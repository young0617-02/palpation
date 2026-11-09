"""E2E 云边协同全链路测试：模型工厂→注册→灰度发布→影子验证→切流/回滚。

验证完整链路：
  模型工厂(七阶段) → ModelRegistry 注册 → 灰度发布(五步节奏)
    → 影子验证(KPI+Wilson) → 切流 / 回滚决策

测试场景覆盖：
1. 模型工厂全链路（异步任务、产物、注册）
2. 灰度发布五步节奏（internal_test→canary→expand_ring→stable_window→full_release）
3. 发布状态机完整流转（pending→downloading→verifying→staged→activating
   →probing→observing→succeeded / rolling_back→rolled_back→frozen）
4. 影子验证 KPI 五指标 + Wilson 置信区间 + 切流决策
5. 回滚决策（信号→层级）+ 冻结发布管理 + 三层版本按层回滚
6. 全链路集成：工厂产出→注册→灰度→影子→切流/回滚
7. 多站点多设备分环（3 站点 × 10 设备 = 30 台）

技术约束：
- tmp_path 隔离产物目录
- sqlite:///:memory: + StaticPool + check_same_thread=False 隔离 DB（支持后台线程）
- datetime.now(UTC) 避免 timezone 问题
- 不依赖外部服务，全部 mock/sim
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cloud.db import Base, EdgeDevice, ModelRegistry, RolloutPlan, RolloutRecord
from cloud.model_factory import (
    DEFAULT_STAGES,
    JobStatus,
    ModelFactory,
    PipelineStage,
)
from cloud.rollback import (
    RollbackManager,
    RollbackPolicy,
    RollbackSignal,
    RollbackTrigger,
    VersionLayer,
)
from cloud.rollout import (
    RingConfig,
    RolloutOrchestrator,
    RolloutState,
    RolloutStep,
)
from cloud.shadow_eval import (
    ShadowEvaluator,
)
from shared.schemas import EdgeEvent, ModelManifest

# ====================================================================
# 常量
# ====================================================================

MODEL_ID = "anomaly-det"
BASELINE_VERSION = "0.1.0"
TARGET_VERSION = "0.2.0"
SITES = ["shanghai", "beijing", "guangzhou"]
HW_MODELS = ["arm64-a", "arm64-b", "x86-c"]
DEVICES_PER_SITE_PER_MODEL = 3  # 3 站点 × 3 型号 × 3 台 = 27 + 3 补足 = 30


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture()
def session_factory():
    """内存 sqlite session_factory，跨线程共享连接以支持 ModelFactory 后台线程。

    用 StaticPool + check_same_thread=False 让内存库在多线程下安全共享，
    因为 :memory: 库每个连接是独立实例，必须共享同一连接。
    """
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def factory(tmp_path: Path, session_factory) -> ModelFactory:
    """模型工厂实例，产物目录隔离在 tmp_path 下。"""
    return ModelFactory(tmp_path / "artifacts", session_factory)


@pytest.fixture()
def orchestrator(session_factory) -> RolloutOrchestrator:
    """灰度发布编排器。"""
    return RolloutOrchestrator(session_factory)


@pytest.fixture()
def evaluator(session_factory) -> ShadowEvaluator:
    """影子验证评估器，min_samples=10 便于测试构造。"""
    return ShadowEvaluator(session_factory, min_samples=10, stable_window_h=24)


@pytest.fixture()
def manager(session_factory) -> RollbackManager:
    """回滚决策管理器。"""
    return RollbackManager(session_factory, RollbackPolicy())


def _make_device(gw_id: str, site: str, hw_model: str) -> EdgeDevice:
    """构造 EdgeDevice。"""
    return EdgeDevice(
        gw_id=gw_id,
        site=site,
        hardware={"model": hw_model, "cpu": "arm64"},
        cert_fingerprint=f"fp-{gw_id}",
    )


@pytest.fixture()
def seeded_devices(session_factory) -> int:
    """插入 30 台设备：3 站点 × 3 型号 × 3 台 + 3 补足 = 30 台。"""
    with session_factory() as s:
        idx = 0
        for site in SITES:
            for hw_model in HW_MODELS:
                for _ in range(DEVICES_PER_SITE_PER_MODEL):
                    idx += 1
                    s.add(_make_device(f"gw-{idx:03d}", site, hw_model))
        # 补 3 台使总数为 30
        for i in range(28, 31):
            s.add(_make_device(f"gw-{i:03d}", "shanghai", "arm64-a"))
        s.commit()
    return 30


@pytest.fixture()
def ring_config() -> RingConfig:
    """测试用 RingConfig，min_sample_size=5 便于测试。"""
    return RingConfig(
        rings=[1, 10, 50, 100],
        stable_window_h=24,
        rollback_window_h=168,
        min_sample_size=5,
    )


# ====================================================================
# 辅助函数
# ====================================================================


def _now() -> datetime:
    return datetime.now(UTC)


def _make_event(
    event_id: str = "e1",
    gw_id: str = "gw-001",
    preliminary: str = "bearing_wear",
) -> EdgeEvent:
    """构造测试用 EdgeEvent。"""
    return EdgeEvent(
        event_id=event_id,
        gw_id=gw_id,
        device_id_hash="hash-device-001",
        seq=1,
        detected_at=_now(),
        rule_id="bearing_rms_high",
        severity="medium",
        metric="vibration_rms",
        value=6.0,
        threshold=4.5,
        preliminary_diagnosis=preliminary,
    )


def _wait_job(factory: ModelFactory, job_id: str, timeout_s: float = 10.0) -> object:
    """轮询工厂任务直到 completed/failed 或超时。"""
    deadline = time.monotonic() + timeout_s
    job = factory.get_job_status(job_id)
    while time.monotonic() < deadline:
        job = factory.get_job_status(job_id)
        if job.status in (JobStatus.completed, JobStatus.failed):
            return job
        time.sleep(0.02)
    return job


def _seed_succeeded(
    session_factory,
    gw_id: str,
    model_id: str,
    version: str,
) -> None:
    """预置一条 state=succeeded 的 RolloutRecord 作为回滚目标。"""
    with session_factory() as s:
        s.add(
            RolloutRecord(
                gw_id=gw_id,
                model_id=model_id,
                version=version,
                state="succeeded",
                detail={},
            )
        )
        s.commit()


def _record_shadow_batch(
    evaluator: ShadowEvaluator,
    gw_id: str,
    model_version: str,
    n: int,
    *,
    n_agree: int | None = None,
    latency_base: float = 10.0,
    latency_step: float = 10.0,
    peak_mem: float = 100.0,
    n_json_ok: int | None = None,
    n_false_positive: int = 0,
    preliminary: str = "bearing_wear",
) -> None:
    """批量记录影子推理，便于构造测试数据。"""
    if n_agree is None:
        n_agree = n
    if n_json_ok is None:
        n_json_ok = n
    for i in range(n):
        event = _make_event(
            event_id=f"{gw_id}-{model_version}-{i}",
            gw_id=gw_id,
            preliminary=preliminary,
        )
        cloud = preliminary if i < n_agree else "other_conclusion"
        evaluator.record_inference(
            gw_id,
            model_version,
            event,
            cloud,
            latency_ms=latency_base + i * latency_step,
            peak_mem_mb=peak_mem,
            json_parse_ok=i < n_json_ok,
            is_false_positive=i < n_false_positive,
        )


def _make_trigger(
    signal: RollbackSignal,
    *,
    gw_id: str = "gw-001",
    model_id: str = MODEL_ID,
    version: str = TARGET_VERSION,
) -> RollbackTrigger:
    """构造 RollbackTrigger，severity 从 _SEVERITY_MAP 推断。"""
    from cloud.rollback import _SEVERITY_MAP

    return RollbackTrigger(
        signal=signal,
        gw_id=gw_id,
        model_id=model_id,
        version=version,
        severity=_SEVERITY_MAP[signal],
        detail={},
        timestamp=_now(),
    )


# ====================================================================
# 1. 模型工厂全链路
# ====================================================================


def test_factory_async_pipeline_produces_all_artifacts(
    factory: ModelFactory,
    session_factory,
    tmp_path: Path,
):
    """异步任务完成七阶段流水线，产出全部产物并注册到 ModelRegistry。

    验证：
    - submit_job 返回 job_id
    - 轮询 get_job_status 直到 completed
    - 七阶段全部完成，每阶段产物文件存在
    - ModelRegistry 中存在对应记录
    """
    job_id = factory.submit_job(
        MODEL_ID, TARGET_VERSION, {"distill_temperature": 3.0}
    )
    assert isinstance(job_id, str)

    job = _wait_job(factory, job_id)
    assert job.status is JobStatus.completed
    assert job.error is None

    # 七阶段全部完成
    assert set(job.artifacts.keys()) == {s.name for s in DEFAULT_STAGES}
    # 每阶段产物文件存在
    for stage_name, artifact_path in job.artifacts.items():
        p = Path(artifact_path)
        assert p.exists(), f"阶段 {stage_name} 产物不存在: {p}"

    # 产物目录在 tmp_path 下（隔离验证）
    artifact_dir = tmp_path / "artifacts"
    assert Path(job.artifacts["register"]).is_relative_to(artifact_dir)

    # ModelRegistry 中存在记录
    with session_factory() as s:
        rows = (
            s.query(ModelRegistry)
            .filter_by(model_id=MODEL_ID, version=TARGET_VERSION)
            .all()
        )
        assert len(rows) == 1
        assert rows[0].sha256 and len(rows[0].sha256) == 64
        assert rows[0].manifest["model_id"] == MODEL_ID
        assert rows[0].manifest["version"] == TARGET_VERSION


def test_factory_manifest_contains_metrics_and_hardware(
    factory: ModelFactory,
    session_factory,
):
    """工厂产出 manifest 携带评测指标与目标硬件信息。"""
    job = factory.create_job(
        MODEL_ID,
        TARGET_VERSION,
        {
            "distill_temperature": 4.0,
            "target_hardware": ["edge-arm64", "edge-x86"],
            "runtime": "llm",
        },
    )
    result = factory.run_all(job)
    manifest: ModelManifest = result.context["manifest"]

    assert manifest.model_id == MODEL_ID
    assert manifest.version == TARGET_VERSION
    assert manifest.target_hardware == ["edge-arm64", "edge-x86"]
    assert manifest.runtime == "llm"
    # 评测指标写入 manifest
    assert "eval_acc" in manifest.metrics_at_register
    assert "p95_latency_ms" in manifest.metrics_at_register
    assert "peak_mem_mb" in manifest.metrics_at_register
    assert manifest.metrics_at_register["replay_source"] == "edge_real_cases"
    assert manifest.metrics_at_register["calibration_corpus"] == "industrial"


def test_factory_pipeline_stage_progression(factory: ModelFactory):
    """工厂流水线阶段顺序推进，current_stage 跟随更新。"""
    job = factory.create_job(MODEL_ID, TARGET_VERSION)
    assert job.status is JobStatus.pending
    assert job.current_stage is None

    # 逐阶段执行，验证 current_stage 跟随
    for stage in DEFAULT_STAGES:
        factory.run_stage(job, stage)
        assert job.current_stage is stage
        assert job.status is JobStatus.stage_done
        assert stage.name in job.artifacts

    # run_all 完成后 status=completed
    job.status = JobStatus.pending  # 重置以演示 run_all
    # 注意：此处不再重新执行 run_all（产物已存在），仅验证状态机语义
    assert job.current_stage is PipelineStage.register


# ====================================================================
# 2. 灰度发布五步节奏
# ====================================================================


def test_rollout_five_steps_full_progression(
    orchestrator: RolloutOrchestrator,
    seeded_devices: int,
    ring_config: RingConfig,
):
    """五步节奏完整推进：internal_test→canary→expand_ring→stable_window→full_release。

    验证：
    - create_plan 初始化到 internal_test
    - advance_step 逐步推进到 full_release
    - 已在最后一步时再推进抛 ValueError
    """
    plan = orchestrator.create_plan(MODEL_ID, TARGET_VERSION, ring_config)
    progress = orchestrator.get_rollout_progress(plan.id)
    assert progress["current_step"] == RolloutStep.internal_test.value

    # 推进 4 次到达 full_release
    expected_steps = [
        RolloutStep.canary,
        RolloutStep.expand_ring,
        RolloutStep.stable_window,
        RolloutStep.full_release,
    ]
    for expected in expected_steps:
        step = orchestrator.advance_step(plan.id)
        assert step == expected

    progress = orchestrator.get_rollout_progress(plan.id)
    assert progress["current_step"] == RolloutStep.full_release.value

    # 已在最后一步，再推进抛异常
    with pytest.raises(ValueError, match="last step"):
        orchestrator.advance_step(plan.id)


def test_rollout_state_machine_full_success_chain(
    orchestrator: RolloutOrchestrator,
    session_factory,
):
    """发布状态机完整成功链路：
    pending→downloading→verifying→staged→activating→probing→observing→succeeded。
    """
    with session_factory() as s:
        rec = RolloutRecord(
            gw_id="gw-001",
            model_id=MODEL_ID,
            version=TARGET_VERSION,
            state=RolloutState.pending.value,
            detail={},
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        rid = rec.id

    chain = [
        RolloutState.downloading,
        RolloutState.verifying,
        RolloutState.staged,
        RolloutState.activating,
        RolloutState.probing,
        RolloutState.observing,
        RolloutState.succeeded,
    ]
    for st in chain:
        rec = orchestrator.transition_state(rid, st, {"step": st.value})
        assert rec.state == st.value


def test_rollout_state_machine_rollback_chain(
    orchestrator: RolloutOrchestrator,
    session_factory,
):
    """发布状态机回滚链路：probing→rolling_back→rolled_back→frozen。"""
    with session_factory() as s:
        rec = RolloutRecord(
            gw_id="gw-001",
            model_id=MODEL_ID,
            version=TARGET_VERSION,
            state=RolloutState.probing.value,
            detail={},
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        rid = rec.id

    rec = orchestrator.transition_state(rid, RolloutState.rolling_back, {"reason": "probe_fail"})
    assert rec.state == RolloutState.rolling_back.value
    rec = orchestrator.transition_state(rid, RolloutState.rolled_back, {"to": BASELINE_VERSION})
    assert rec.state == RolloutState.rolled_back.value
    rec = orchestrator.transition_state(rid, RolloutState.frozen, {"why": "freeze"})
    assert rec.state == RolloutState.frozen.value


def test_rollout_ring_expansion_grows_device_count(
    orchestrator: RolloutOrchestrator,
    seeded_devices: int,
    ring_config: RingConfig,
    session_factory,
):
    """分环扩圈 10%→50%→100%，设备数量按比例增长。"""
    plan = orchestrator.create_plan(MODEL_ID, TARGET_VERSION, ring_config)
    total = seeded_devices  # 30

    # 10% = ceil(30 * 0.10) = 3
    recs_10 = orchestrator.expand_ring(plan.id, 10)
    assert len(recs_10) == 3
    assert all(r.state == RolloutState.pending.value for r in recs_10)

    # 50% = 15，已有 3，新增 12
    recs_50 = orchestrator.expand_ring(plan.id, 50)
    assert len(recs_50) == 12

    # 100% = 30，已有 15，新增 15
    recs_100 = orchestrator.expand_ring(plan.id, 100)
    assert len(recs_100) == 15

    # 总记录数 = 30
    with session_factory() as s:
        total_recs = (
            s.query(RolloutRecord)
            .filter_by(model_id=MODEL_ID, version=TARGET_VERSION)
            .count()
        )
    assert total_recs == total

    # 进度查询验证
    progress = orchestrator.get_rollout_progress(plan.id)
    assert progress["current_ring_percent"] == 100
    assert progress["total_records"] == total


# ====================================================================
# 3. 影子验证 KPI + 置信区间
# ====================================================================


def test_shadow_kpi_all_five_metrics(evaluator: ShadowEvaluator):
    """影子验证 KPI 五指标计算正确：
    一致率 / P95 延迟 / 峰值内存 / JSON 可解析率 / 误报率。
    """
    _record_shadow_batch(
        evaluator,
        gw_id="gw-001",
        model_version="1.0",
        n=10,
        n_agree=8,           # 一致率 0.8
        latency_base=10,
        latency_step=10,     # [10,20,...,100] → P95=95.5
        peak_mem=200,
        n_json_ok=9,         # JSON 可解析率 0.9
        n_false_positive=1,  # 误报率 0.1
    )

    kpi = evaluator.compute_kpi("gw-001", "1.0")
    assert kpi.agreement_rate == pytest.approx(0.8)
    # P95: k = 9 * 0.95 = 8.55, f=8, c=9 → 90 + 0.55*(100-90) = 95.5
    assert kpi.p95_latency_ms == pytest.approx(95.5)
    assert kpi.peak_mem_mb == pytest.approx(200)
    assert kpi.json_parse_rate == pytest.approx(0.9)
    assert kpi.false_positive_rate == pytest.approx(0.1)


def test_shadow_wilson_confidence_interval_properties(evaluator: ShadowEvaluator):
    """Wilson 置信区间计算性质验证。

    - n=0 时返回 (0.0, 1.0)
    - rate=0/1 时不越界 [0, 1]
    - 更高置信度 → 更宽区间
    - 更大样本量 → 更窄区间
    - 经典值 rate=0.5, n=100 → 约 (0.404, 0.596)
    """
    # n=0 全区间
    lo, hi = evaluator.confidence_interval(0.5, 0)
    assert (lo, hi) == (0.0, 1.0)

    # rate=0/1 不越界
    lo0, hi0 = evaluator.confidence_interval(0.0, 100)
    assert lo0 == 0.0
    assert 0.0 <= hi0 <= 1.0
    lo1, hi1 = evaluator.confidence_interval(1.0, 100)
    assert 0.0 <= lo1 <= 1.0
    assert hi1 == 1.0

    # 经典值
    lo, hi = evaluator.confidence_interval(0.5, 100, confidence=0.95)
    assert lo == pytest.approx(0.404, abs=0.005)
    assert hi == pytest.approx(0.596, abs=0.005)
    assert lo < 0.5 < hi

    # 更高置信度 → 更宽区间
    lo95, hi95 = evaluator.confidence_interval(0.5, 50, confidence=0.95)
    lo99, hi99 = evaluator.confidence_interval(0.5, 50, confidence=0.99)
    assert (hi99 - lo99) > (hi95 - lo95)

    # 更大样本量 → 更窄区间
    lo_small, hi_small = evaluator.confidence_interval(0.5, 30)
    lo_large, hi_large = evaluator.confidence_interval(0.5, 300)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_shadow_promote_decision_pass_fail_insufficient(
    evaluator: ShadowEvaluator,
):
    """切流决策三种结果：pass / fail / insufficient_samples。"""
    # === pass：new KPI 优于 baseline 且样本充足 ===
    _record_shadow_batch(
        evaluator, "gw-pass", "1.0", n=10, n_agree=8,
        latency_base=10, latency_step=10, peak_mem=200, n_json_ok=10,
    )
    _record_shadow_batch(
        evaluator, "gw-pass", "2.0", n=10, n_agree=9,
        latency_base=8, latency_step=8, peak_mem=180, n_json_ok=10,
    )
    result_pass = evaluator.compare("gw-pass", "2.0", "1.0")
    assert result_pass.passed is True
    assert result_pass.sample_size == 10

    # === fail：一致率劣化超 2pp ===
    _record_shadow_batch(
        evaluator, "gw-fail", "1.0", n=10, n_agree=8,
        peak_mem=200, n_json_ok=10,
    )
    _record_shadow_batch(
        evaluator, "gw-fail", "2.0", n=10, n_agree=5,  # 0.5 << 0.78
        peak_mem=180, n_json_ok=10,
    )
    result_fail = evaluator.compare("gw-fail", "2.0", "1.0")
    assert result_fail.passed is False

    # === insufficient_samples：样本不足 ===
    _record_shadow_batch(
        evaluator, "gw-insuf", "1.0", n=10, n_agree=8,
        peak_mem=200, n_json_ok=10,
    )
    _record_shadow_batch(
        evaluator, "gw-insuf", "2.0", n=3, n_agree=3,  # 样本不足
        peak_mem=180, n_json_ok=3,
    )
    result_insuf = evaluator.compare("gw-insuf", "2.0", "1.0")
    assert result_insuf.passed is False
    assert result_insuf.sample_size < evaluator.min_samples


# ====================================================================
# 4. 回滚决策 + 冻结
# ====================================================================


def test_rollback_signal_to_layer_mapping(manager: RollbackManager):
    """回滚触发信号→三层版本按层回滚映射：
    - process_start_failure / probe_failure → firmware
    - p95_latency_exceeded / json_parse_rate_drop → model
    - mem_exceeded → config
    - input_link_down → config (但 alert_only 不回滚)
    """
    cases = [
        (RollbackSignal.process_start_failure, VersionLayer.firmware),
        (RollbackSignal.probe_failure, VersionLayer.firmware),
        (RollbackSignal.p95_latency_exceeded, VersionLayer.model),
        (RollbackSignal.json_parse_rate_drop, VersionLayer.model),
        (RollbackSignal.mem_exceeded, VersionLayer.config),
        (RollbackSignal.input_link_down, VersionLayer.config),
    ]
    for signal, expected_layer in cases:
        trigger = _make_trigger(signal)
        layer = manager.select_rollback_layer(trigger)
        assert layer == expected_layer, f"{signal.value} 应映射到 {expected_layer.value}"


def test_rollback_freeze_management_idempotent(
    manager: RollbackManager,
    session_factory,
):
    """冻结发布管理：freeze→is_frozen→idempotent→get_frozen_versions。"""
    # 初始未冻结
    assert manager.is_frozen(MODEL_ID, TARGET_VERSION) is False

    # 冻结
    manager.freeze_version(MODEL_ID, TARGET_VERSION)
    assert manager.is_frozen(MODEL_ID, TARGET_VERSION) is True

    # 幂等：重复冻结不报错
    manager.freeze_version(MODEL_ID, TARGET_VERSION)
    assert manager.is_frozen(MODEL_ID, TARGET_VERSION) is True

    # get_frozen_versions 包含该版本
    frozen = manager.get_frozen_versions()
    assert {"model_id": MODEL_ID, "version": TARGET_VERSION} in frozen

    # 冻结已存在的 RolloutPlan 时更新 frozen 字段
    with session_factory() as s:
        s.add(RolloutPlan(model_id="intent-llm", version="1.0.0", frozen=False))
        s.commit()
    manager.freeze_version("intent-llm", "1.0.0")
    with session_factory() as s:
        plan = (
            s.query(RolloutPlan)
            .filter_by(model_id="intent-llm", version="1.0.0")
            .one()
        )
        assert plan.frozen is True


def test_rollback_execute_updates_records_and_freezes(
    manager: RollbackManager,
    session_factory,
):
    """执行回滚更新 RolloutRecord 状态为 rolled_back 并冻结问题版本。

    验证：
    - immediate 信号（probe_failure）触发回滚
    - execute_rollback 更新记录状态
    - 问题版本被冻结
    - alert_only 信号不触发回滚
    - 无历史版本时不回滚
    """
    # 预置历史成功版本（回滚目标，先插入确保 id 较小）
    _seed_succeeded(session_factory, "gw-001", MODEL_ID, BASELINE_VERSION)
    # 预置当前问题版本的活跃记录（处于 probing 状态，非 succeeded）
    with session_factory() as s:
        s.add(
            RolloutRecord(
                gw_id="gw-001",
                model_id=MODEL_ID,
                version=TARGET_VERSION,
                state="probing",
                detail={},
            )
        )
        s.commit()

    # immediate 信号触发回滚
    trigger = _make_trigger(RollbackSignal.probe_failure)
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is True
    assert decision.target_version == BASELINE_VERSION
    assert decision.frozen_version == TARGET_VERSION
    assert decision.layer == VersionLayer.firmware

    # alert_only 信号不触发回滚（在冻结前测试，避免冻结干扰）
    alert_trigger = _make_trigger(RollbackSignal.input_link_down)
    alert_decision = manager.evaluate_trigger(alert_trigger)
    assert alert_decision.should_rollback is False
    assert "alert_only" in alert_decision.reason

    # 无历史版本时不回滚（新设备，在冻结前测试）
    no_history_trigger = _make_trigger(
        RollbackSignal.probe_failure, gw_id="gw-no-history"
    )
    no_history_decision = manager.evaluate_trigger(no_history_trigger)
    assert no_history_decision.should_rollback is False
    assert "no previous succeeded" in no_history_decision.reason

    # 执行回滚
    record = manager.execute_rollback(decision)
    assert record.state == "rolled_back"
    assert record.detail["rollback_layer"] == "firmware"
    assert record.detail["rollback_target"] == BASELINE_VERSION

    # 问题版本被冻结
    assert manager.is_frozen(MODEL_ID, TARGET_VERSION) is True

    # 冻结后再次评估同版本 → 幂等，不重复回滚
    re_eval = manager.evaluate_trigger(trigger)
    assert re_eval.should_rollback is False
    assert "already frozen" in re_eval.reason


# ====================================================================
# 5. 全链路集成
# ====================================================================


def test_e2e_full_pipeline_factory_to_rollout_to_shadow_pass(
    factory: ModelFactory,
    orchestrator: RolloutOrchestrator,
    evaluator: ShadowEvaluator,
    manager: RollbackManager,
    session_factory,
    seeded_devices: int,
    ring_config: RingConfig,
    tmp_path: Path,
):
    """全链路集成（切流成功路径）：
    工厂产出模型 → 注册到 ModelRegistry → 灰度发布该模型
    → 影子验证（KPI 达标） → 切流到全量。

    验证各阶段产物关联性：灰度计划引用的 model_id/version = 工厂产出的 model_id/version。
    """
    # === 阶段 1：模型工厂产出模型 ===
    job_id = factory.submit_job(MODEL_ID, TARGET_VERSION, {"distill_temperature": 3.0})
    job = _wait_job(factory, job_id)
    assert job.status is JobStatus.completed
    manifest: ModelManifest = job.context["manifest"]
    produced_model_id = manifest.model_id
    produced_version = manifest.version

    # 验证注册到 ModelRegistry
    with session_factory() as s:
        row = (
            s.query(ModelRegistry)
            .filter_by(model_id=produced_model_id, version=produced_version)
            .one()
        )
        assert row.sha256 == manifest.sha256

    # === 阶段 2：灰度发布该模型 ===
    plan = orchestrator.create_plan(produced_model_id, produced_version, ring_config)
    assert plan.model_id == produced_model_id
    assert plan.version == produced_version

    # 推进到 canary 步
    orchestrator.advance_step(plan.id)
    # 扩圈 10%
    canary_recs = orchestrator.expand_ring(plan.id, 10)
    assert len(canary_recs) >= 1
    canary_gw = canary_recs[0].gw_id

    # === 阶段 3：影子验证（模拟 Canary 组并行推理） ===
    # baseline 版本（现网版本）的影子推理记录
    _record_shadow_batch(
        evaluator, canary_gw, BASELINE_VERSION, n=10, n_agree=8,
        latency_base=10, latency_step=10, peak_mem=200, n_json_ok=10,
    )
    # new 版本的影子推理记录（KPI 优于 baseline）
    _record_shadow_batch(
        evaluator, canary_gw, produced_version, n=10, n_agree=9,
        latency_base=8, latency_step=8, peak_mem=180, n_json_ok=10,
    )

    result = evaluator.compare(canary_gw, produced_version, BASELINE_VERSION)
    assert result.passed is True
    assert result.sample_size == 10

    # === 阶段 4：切流（KPI 达标 → 推进到全量） ===
    # 推进到 stable_window
    orchestrator.advance_step(plan.id)  # expand_ring
    orchestrator.advance_step(plan.id)  # stable_window

    # 将 canary 设备记录置为 succeeded（模拟探针通过）
    with session_factory() as s:
        for rec in canary_recs:
            r = s.get(RolloutRecord, rec.id)
            r.state = RolloutState.succeeded.value
        s.commit()

    # 扩圈 50%
    orchestrator.expand_ring(plan.id, 50)
    # 全量发布
    full_recs = orchestrator.full_release(plan.id)
    assert len(full_recs) >= 1

    # 验证进度推进到 full_release
    progress = orchestrator.get_rollout_progress(plan.id)
    assert progress["current_step"] == RolloutStep.full_release.value
    assert progress["current_ring_percent"] == 100

    # 验证版本未冻结（切流成功，无需回滚）
    assert manager.is_frozen(produced_model_id, produced_version) is False


def test_e2e_full_pipeline_with_rollback_on_probe_failure(
    factory: ModelFactory,
    orchestrator: RolloutOrchestrator,
    evaluator: ShadowEvaluator,
    manager: RollbackManager,
    session_factory,
    seeded_devices: int,
    ring_config: RingConfig,
):
    """全链路集成（回滚路径）：
    工厂产出模型 → 注册 → 灰度发布 → 探针失败 → 回滚到 baseline。

    验证：
    - 探针失败触发 immediate 回滚
    - 回滚目标版本为 baseline
    - 问题版本被冻结
    - RolloutRecord 状态转为 rolled_back
    """
    # === 阶段 1：工厂产出 ===
    job_id = factory.submit_job(MODEL_ID, TARGET_VERSION)
    job = _wait_job(factory, job_id)
    assert job.status is JobStatus.completed

    # === 阶段 2：预置 baseline 历史成功版本（回滚目标） ===
    # 模拟 baseline 版本曾在所有设备上 succeeded
    with session_factory() as s:
        for i in range(1, 4):  # 为前 3 台设备预置 baseline succeeded
            s.add(
                RolloutRecord(
                    gw_id=f"gw-{i:03d}",
                    model_id=MODEL_ID,
                    version=BASELINE_VERSION,
                    state="succeeded",
                    detail={},
                )
            )
        s.commit()

    # === 阶段 3：灰度发布 new 版本 ===
    plan = orchestrator.create_plan(MODEL_ID, TARGET_VERSION, ring_config)
    orchestrator.advance_step(plan.id)  # → canary
    canary_recs = orchestrator.expand_ring(plan.id, 10)
    assert len(canary_recs) >= 1

    # 模拟 canary 设备进入 probing 状态
    with session_factory() as s:
        for rec in canary_recs:
            r = s.get(RolloutRecord, rec.id)
            r.state = RolloutState.probing.value
        s.commit()

    # === 阶段 4：探针失败触发回滚 ===
    failed_gw = canary_recs[0].gw_id
    trigger = _make_trigger(
        RollbackSignal.probe_failure,
        gw_id=failed_gw,
        model_id=MODEL_ID,
        version=TARGET_VERSION,
    )
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is True
    assert decision.layer == VersionLayer.firmware
    assert decision.target_version == BASELINE_VERSION

    # 执行回滚
    record = manager.execute_rollback(decision)
    assert record.state == "rolled_back"

    # 问题版本被冻结
    assert manager.is_frozen(MODEL_ID, TARGET_VERSION) is True

    # 冻结后无法继续扩圈
    with pytest.raises(ValueError, match="frozen"):
        orchestrator.expand_ring(plan.id, 50)


def test_e2e_artifacts_linking_across_stages(
    factory: ModelFactory,
    orchestrator: RolloutOrchestrator,
    evaluator: ShadowEvaluator,
    manager: RollbackManager,
    session_factory,
    seeded_devices: int,
    ring_config: RingConfig,
):
    """验证各阶段产物的关联性：
    - 工厂产出的 manifest.model_id/version == ModelRegistry 记录
    - 灰度计划引用的 model_id/version == 工厂产出的
    - 影子验证的 new_version == 工厂产出的 version
    - 回滚决策的 model_id/version == 工厂产出的
    """
    # 工厂产出
    job_id = factory.submit_job(MODEL_ID, TARGET_VERSION, {"distill_temperature": 4.0})
    job = _wait_job(factory, job_id)
    assert job.status is JobStatus.completed
    manifest: ModelManifest = job.context["manifest"]

    # ModelRegistry 关联
    with session_factory() as s:
        row = (
            s.query(ModelRegistry)
            .filter_by(model_id=manifest.model_id, version=manifest.version)
            .one()
        )
        assert row.manifest["sha256"] == manifest.sha256

    # 灰度计划关联
    plan = orchestrator.create_plan(manifest.model_id, manifest.version, ring_config)
    assert plan.model_id == manifest.model_id
    assert plan.version == manifest.version

    # 影子验证关联
    _record_shadow_batch(
        evaluator, "gw-001", BASELINE_VERSION, n=10, n_agree=8,
        peak_mem=200, n_json_ok=10,
    )
    _record_shadow_batch(
        evaluator, "gw-001", manifest.version, n=10, n_agree=9,
        peak_mem=180, n_json_ok=10,
    )
    result = evaluator.compare("gw-001", manifest.version, BASELINE_VERSION)
    assert result.new_version == manifest.version
    assert result.baseline_version == BASELINE_VERSION

    # 回滚决策关联
    _seed_succeeded(session_factory, "gw-001", MODEL_ID, BASELINE_VERSION)
    trigger = _make_trigger(
        RollbackSignal.p95_latency_exceeded,
        gw_id="gw-001",
        model_id=manifest.model_id,
        version=manifest.version,
    )
    decision = manager.evaluate_trigger(trigger)
    assert decision.model_id == manifest.model_id
    assert decision.frozen_version == manifest.version
    assert decision.target_version == BASELINE_VERSION


# ====================================================================
# 6. 多站点多设备分环
# ====================================================================


def test_multi_site_multi_device_ring_selection(
    orchestrator: RolloutOrchestrator,
    seeded_devices: int,
    session_factory,
):
    """多站点多设备分环选择：3 站点 × 10 设备 = 30 台。

    验证：
    - 1% 至少选 1 台
    - 10% = 3 台
    - 50% = 15 台
    - 100% = 30 台
    - 按站点过滤正确
    - 按型号过滤正确
    """
    total = seeded_devices  # 30
    assert total == 30

    one_pct = orchestrator.select_devices(1)
    ten_pct = orchestrator.select_devices(10)
    fifty_pct = orchestrator.select_devices(50)
    full = orchestrator.select_devices(100)

    assert len(one_pct) >= 1
    assert len(ten_pct) == 3
    assert len(fifty_pct) == 15
    assert len(full) == 30

    # 按站点过滤
    shanghai_all = orchestrator.select_devices(100, site="shanghai")
    with session_factory() as s:
        devs = {d.gw_id: d for d in s.query(EdgeDevice).all()}
    assert all(devs[gw].site == "shanghai" for gw in shanghai_all)
    # shanghai: 9 (3 models × 3) + 3 extra = 12
    assert len(shanghai_all) == 12

    # 按型号过滤
    arm64_a_all = orchestrator.select_devices(100, hardware="arm64-a")
    assert all(devs[gw].hardware["model"] == "arm64-a" for gw in arm64_a_all)
    # arm64-a: 3 sites × 3 + 3 extra = 12
    assert len(arm64_a_all) == 12


def test_multi_site_ring_diversity(
    orchestrator: RolloutOrchestrator,
    seeded_devices: int,
    session_factory,
):
    """分环多样性验证：10% 选择应包含不同站点与型号的代表性设备。

    轮询 interleaving 保证前 N 台在站点与型号两个维度都多样。
    """
    selected = orchestrator.select_devices(10)  # 3 台
    with session_factory() as s:
        devs = {d.gw_id: d for d in s.query(EdgeDevice).all()}

    sites = {devs[gw].site for gw in selected}
    models = {devs[gw].hardware["model"] for gw in selected}

    # 3 台应来自不同站点和不同型号
    assert len(sites) >= 2, f"expected diverse sites, got {sites}"
    assert len(models) >= 2, f"expected diverse models, got {models}"

    # 50% 选择应覆盖全部 3 个站点
    selected_50 = orchestrator.select_devices(50)
    sites_50 = {devs[gw].site for gw in selected_50}
    assert sites_50 == set(SITES)


# ====================================================================
# 7. 全链路：五步节奏完整流转（端到端）
# ====================================================================


def test_e2e_five_step_rhythm_full_flow(
    factory: ModelFactory,
    orchestrator: RolloutOrchestrator,
    evaluator: ShadowEvaluator,
    manager: RollbackManager,
    session_factory,
    seeded_devices: int,
    ring_config: RingConfig,
):
    """五步节奏完整流转端到端验证：

    internal_test（1 台实验室设备 succeeded）
    → canary（扩圈 1%，验证升级/推理/回滚链路）
    → expand_ring（10% → 50%，按站点/型号分环）
    → stable_window（观察 succeeded 数 ≥ min_sample_size）
    → full_release（全量发布）

    每步验证状态机转换正确。
    """
    # 工厂产出模型
    job_id = factory.submit_job(MODEL_ID, TARGET_VERSION)
    job = _wait_job(factory, job_id)
    assert job.status is JobStatus.completed

    # 创建灰度计划（初始步 internal_test）
    plan = orchestrator.create_plan(MODEL_ID, TARGET_VERSION, ring_config)
    progress = orchestrator.get_rollout_progress(plan.id)
    assert progress["current_step"] == RolloutStep.internal_test.value

    # === 步骤 1：internal_test ===
    # 模拟 1 台实验室设备完成全状态链路
    with session_factory() as s:
        internal_rec = RolloutRecord(
            gw_id="gw-lab-001",
            model_id=MODEL_ID,
            version=TARGET_VERSION,
            state=RolloutState.pending.value,
            detail={"step": RolloutStep.internal_test.value},
        )
        s.add(internal_rec)
        s.commit()
        s.refresh(internal_rec)
        rid = internal_rec.id

    # 完整状态链路：pending→downloading→verifying→staged→activating→probing→observing→succeeded
    # 注意：can_deploy_to_canary 要求 detail["step"] == "internal_test"，
    # transition_state 会覆盖整个 detail，因此每步都需保留 step 标记。
    success_chain = [
        RolloutState.downloading,
        RolloutState.verifying,
        RolloutState.staged,
        RolloutState.activating,
        RolloutState.probing,
        RolloutState.observing,
        RolloutState.succeeded,
    ]
    for st in success_chain:
        rec = orchestrator.transition_state(
            rid, st, {"step": RolloutStep.internal_test.value, "state": st.value}
        )
        assert rec.state == st.value

    # === 步骤 2：canary ===
    canary_step = orchestrator.advance_step(plan.id)
    assert canary_step == RolloutStep.canary
    # 验证 canary 部署条件满足
    assert orchestrator.can_deploy_to_canary(plan.id) is True
    # 扩圈 1%
    canary_recs = orchestrator.expand_ring(plan.id, 1)
    assert len(canary_recs) >= 1

    # === 步骤 3：expand_ring（10% → 50%） ===
    expand_step = orchestrator.advance_step(plan.id)
    assert expand_step == RolloutStep.expand_ring
    recs_10 = orchestrator.expand_ring(plan.id, 10)
    recs_50 = orchestrator.expand_ring(plan.id, 50)
    assert len(recs_10) + len(recs_50) > 0

    # === 步骤 4：stable_window ===
    stable_step = orchestrator.advance_step(plan.id)
    assert stable_step == RolloutStep.stable_window

    # 将所有记录置为 succeeded（模拟探针通过）
    with session_factory() as s:
        all_recs = (
            s.query(RolloutRecord)
            .filter_by(model_id=MODEL_ID, version=TARGET_VERSION)
            .all()
        )
        for r in all_recs:
            r.state = RolloutState.succeeded.value
        s.commit()

    # 验证稳定窗口满足（succeeded 数 ≥ min_sample_size=5）
    assert orchestrator.check_stable_window(plan.id) is True

    # === 步骤 5：full_release ===
    orchestrator.full_release(plan.id)
    progress = orchestrator.get_rollout_progress(plan.id)
    assert progress["current_step"] == RolloutStep.full_release.value
    assert progress["current_ring_percent"] == 100

    # 验证全量覆盖所有现场设备
    # 注意：gw-lab-001 是实验室设备（不在 EdgeDevice 表中），
    # 其记录单独创建；full_release 覆盖 30 台现场设备。
    with session_factory() as s:
        total_recs = (
            s.query(RolloutRecord)
            .filter_by(model_id=MODEL_ID, version=TARGET_VERSION)
            .count()
        )
        # 30 台现场设备 + 1 台实验室设备
        assert total_recs == seeded_devices + 1
        # 验证所有 30 台现场设备都有记录
        rec_gw_ids = {
            r.gw_id
            for r in s.query(RolloutRecord)
            .filter_by(model_id=MODEL_ID, version=TARGET_VERSION)
            .all()
        }
        device_gw_ids = {d.gw_id for d in s.query(EdgeDevice).all()}
        assert device_gw_ids.issubset(rec_gw_ids), "所有现场设备应有发布记录"
