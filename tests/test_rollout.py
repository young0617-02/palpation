"""cloud/rollout.py 单元测试：灰度发布五步节奏 + 状态机 + 分环。

使用内存 sqlite + sessionmaker，覆盖创建计划、分环选择、状态机转换、
Canary 条件、扩圈、稳定窗口、全量发布、进度查询与边界场景。
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import Base, EdgeDevice, RolloutPlan, RolloutRecord
from cloud.rollout import (
    RingConfig,
    RolloutOrchestrator,
    RolloutState,
    RolloutStep,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def session_factory():
    """内存 sqlite sessionmaker，建全部表。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def orch(session_factory):
    return RolloutOrchestrator(session_factory)


def _make_device(gw_id: str, site: str, hw_model: str) -> EdgeDevice:
    return EdgeDevice(
        gw_id=gw_id,
        site=site,
        hardware={"model": hw_model, "cpu": "arm64"},
        cert_fingerprint=f"fp-{gw_id}",
    )


@pytest.fixture()
def seeded_devices(session_factory):
    """插入 30 台设备：3 站点 × 3 型号 × ~3-4 台，用于分环测试。"""
    sites = ["shanghai", "beijing", "guangzhou"]
    models = ["arm64-a", "arm64-b", "x86-c"]
    with session_factory() as s:
        idx = 0
        for site in sites:
            for model in models:
                for _ in range(3):  # 3*3*3 = 27 台
                    idx += 1
                    s.add(_make_device(f"gw-{idx:03d}", site, model))
                # shanghai 额外 1 台使总数为 30
        # 补 3 台使总数为 30
        for i in range(28, 31):
            s.add(_make_device(f"gw-{i:03d}", "shanghai", "arm64-a"))
        s.commit()
    return 30


@pytest.fixture()
def plan_id(orch, seeded_devices):
    """创建一个灰度计划并返回其 id。"""
    cfg = RingConfig(rings=[1, 10, 50, 100], stable_window_h=24,
                     rollback_window_h=168, min_sample_size=5)
    plan = orch.create_plan("anomaly-det", "0.2.0", cfg)
    return plan.id


# --------------------------------------------------------------------------- #
# 1. 创建灰度计划 → 五步节奏初始化
# --------------------------------------------------------------------------- #

def test_create_plan_initializes_five_steps(orch, session_factory):
    cfg = RingConfig(rings=[1, 10, 50, 100])
    plan = orch.create_plan("anomaly-det", "0.2.0", cfg)
    assert plan.model_id == "anomaly-det"
    assert plan.version == "0.2.0"
    assert plan.strategy == "ring"
    assert plan.rings == [1, 10, 50, 100]
    assert plan.frozen is False
    # 持久化
    with session_factory() as s:
        got = s.get(RolloutPlan, plan.id)
        assert got is not None
        assert got.rings == [1, 10, 50, 100]
    # 进度初始步为 internal_test
    progress = orch.get_rollout_progress(plan.id)
    assert progress["current_step"] == RolloutStep.internal_test.value
    assert progress["current_ring_percent"] == 0


def test_ring_config_defaults():
    cfg = RingConfig()
    assert cfg.rings == [1, 10, 50, 100]
    assert cfg.stable_window_h == 24
    assert cfg.rollback_window_h == 168
    assert cfg.min_sample_size == 100


# --------------------------------------------------------------------------- #
# 2. 分环选择设备 → 按站点/型号分组
# --------------------------------------------------------------------------- #

def test_select_devices_respects_ring_percent(orch, seeded_devices):
    total = seeded_devices  # 30
    one_pct = orch.select_devices(1)
    ten_pct = orch.select_devices(10)
    fifty_pct = orch.select_devices(50)
    full = orch.select_devices(100)
    assert len(one_pct) >= 1            # 至少 1 台
    assert len(ten_pct) == 3            # ceil(30*0.10)=3
    assert len(fifty_pct) == 15         # ceil(30*0.50)=15
    assert len(full) == total


def test_select_devices_interleaves_site_and_hardware(orch, seeded_devices):
    """10% 选择应包含不同站点与型号的代表性设备。"""
    selected = orch.select_devices(10)  # 3 台
    with orch._session_factory() as s:
        devs = {d.gw_id: d for d in s.query(EdgeDevice).all()}
    sites = {devs[gw].site for gw in selected}
    models = {devs[gw].hardware["model"] for gw in selected}
    # 3 台应来自不同站点和不同型号（interleaving 保证多样性）
    assert len(sites) >= 2, f"expected diverse sites, got {sites}"
    assert len(models) >= 2, f"expected diverse models, got {models}"


def test_select_devices_filter_by_site(orch, seeded_devices):
    selected = orch.select_devices(100, site="shanghai")
    with orch._session_factory() as s:
        devs = {d.gw_id: d for d in s.query(EdgeDevice).all()}
    assert all(devs[gw].site == "shanghai" for gw in selected)
    # shanghai: 9 (3 models × 3) + 3 extra = 12
    assert len(selected) == 12


def test_select_devices_filter_by_hardware(orch, seeded_devices):
    selected = orch.select_devices(100, hardware="arm64-a")
    with orch._session_factory() as s:
        devs = {d.gw_id: d for d in s.query(EdgeDevice).all()}
    assert all(devs[gw].hardware["model"] == "arm64-a" for gw in selected)
    # arm64-a: 3 sites × 3 + 3 extra = 12
    assert len(selected) == 12


# --------------------------------------------------------------------------- #
# 3. 状态机转换 → 合法/非法
# --------------------------------------------------------------------------- #

def test_transition_state_legal(orch, session_factory):
    with session_factory() as s:
        rec = RolloutRecord(gw_id="gw-001", model_id="m", version="v",
                            state=RolloutState.pending.value, detail={})
        s.add(rec)
        s.commit()
        s.refresh(rec)
        rid = rec.id
    updated = orch.transition_state(rid, RolloutState.downloading, {"ts": 1})
    assert updated.state == RolloutState.downloading.value
    assert updated.detail == {"ts": 1}


def test_transition_state_illegal_raises(orch, session_factory):
    with session_factory() as s:
        rec = RolloutRecord(gw_id="gw-001", model_id="m", version="v",
                            state=RolloutState.pending.value, detail={})
        s.add(rec)
        s.commit()
        s.refresh(rec)
        rid = rec.id
    # pending → succeeded 非法
    with pytest.raises(ValueError, match="illegal state transition"):
        orch.transition_state(rid, RolloutState.succeeded, {})


def test_transition_state_full_legal_chain(orch, session_factory):
    """验证完整合法链路：pending→downloading→verifying→staged→activating
    →probing→observing→succeeded。"""
    with session_factory() as s:
        rec = RolloutRecord(gw_id="gw-001", model_id="m", version="v",
                            state=RolloutState.pending.value, detail={})
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
        rec = orch.transition_state(rid, st, {"step": st.value})
        assert rec.state == st.value


def test_transition_state_rollback_chain(orch, session_factory):
    """验证回滚链路：probing→rolling_back→rolled_back→frozen。"""
    with session_factory() as s:
        rec = RolloutRecord(gw_id="gw-001", model_id="m", version="v",
                            state=RolloutState.probing.value, detail={})
        s.add(rec)
        s.commit()
        s.refresh(rec)
        rid = rec.id
    rec = orch.transition_state(rid, RolloutState.rolling_back, {"reason": "probe_fail"})
    assert rec.state == RolloutState.rolling_back.value
    rec = orch.transition_state(rid, RolloutState.rolled_back, {"to": "0.1.0"})
    assert rec.state == RolloutState.rolled_back.value
    rec = orch.transition_state(rid, RolloutState.frozen, {"why": "freeze"})
    assert rec.state == RolloutState.frozen.value


def test_transition_state_invalid_record_id(orch):
    with pytest.raises(ValueError, match="not found"):
        orch.transition_state(99999, RolloutState.downloading, {})


# --------------------------------------------------------------------------- #
# 4. Canary 部署条件检查
# --------------------------------------------------------------------------- #

def test_can_deploy_to_canary_false_before_internal_success(orch, plan_id):
    # 未推进步 + 无 internal_test succeeded 记录
    assert orch.can_deploy_to_canary(plan_id) is False


def test_can_deploy_to_canary_true_after_internal_success(orch, plan_id, session_factory):
    # 推进到 canary 步
    orch.advance_step(plan_id)
    # 创建一条 internal_test succeeded 记录
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        s.add(RolloutRecord(
            gw_id="gw-001", model_id=plan.model_id, version=plan.version,
            state=RolloutState.succeeded.value,
            detail={"step": RolloutStep.internal_test.value},
        ))
        s.commit()
    assert orch.can_deploy_to_canary(plan_id) is True


def test_can_deploy_to_canary_false_when_frozen(orch, plan_id, session_factory):
    orch.advance_step(plan_id)
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        plan.frozen = True
        s.add(RolloutRecord(
            gw_id="gw-001", model_id=plan.model_id, version=plan.version,
            state=RolloutState.succeeded.value,
            detail={"step": RolloutStep.internal_test.value},
        ))
        s.commit()
    assert orch.can_deploy_to_canary(plan_id) is False


# --------------------------------------------------------------------------- #
# 5. 扩圈 10% → 50% → 设备列表增长
# --------------------------------------------------------------------------- #

def test_expand_ring_grows_device_list(orch, plan_id, session_factory):
    recs_10 = orch.expand_ring(plan_id, 10)
    assert len(recs_10) == 3  # 30 * 10% = 3
    # 所有新记录为 pending
    assert all(r.state == RolloutState.pending.value for r in recs_10)
    # 扩圈到 50%
    recs_50 = orch.expand_ring(plan_id, 50)
    # 50% = 15 台，已有 3 台，新增 12
    assert len(recs_50) == 12
    # 总记录数 = 15
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        total = s.query(RolloutRecord).filter_by(
            model_id=plan.model_id, version=plan.version).count()
    assert total == 15


def test_expand_ring_idempotent(orch, plan_id):
    """对同一百分比重复扩圈不产生重复记录。"""
    first = orch.expand_ring(plan_id, 10)
    second = orch.expand_ring(plan_id, 10)
    assert len(first) == 3
    assert len(second) == 0  # 已存在，跳过


def test_expand_ring_frozen_plan_raises(orch, plan_id, session_factory):
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        plan.frozen = True
        s.commit()
    with pytest.raises(ValueError, match="frozen"):
        orch.expand_ring(plan_id, 50)


# --------------------------------------------------------------------------- #
# 6. 稳定窗口检查 → 满足/不满足
# --------------------------------------------------------------------------- #

def test_check_stable_window_false_before_step(orch, plan_id):
    # 当前步为 internal_test（索引 0），未到 stable_window（索引 3）
    assert orch.check_stable_window(plan_id) is False


def test_check_stable_window_false_insufficient_samples(orch, plan_id):
    # 推进到 stable_window 步
    for _ in range(3):
        orch.advance_step(plan_id)
    # 无 succeeded 记录，样本不足
    assert orch.check_stable_window(plan_id) is False


def test_check_stable_window_true(orch, plan_id, session_factory):
    # 推进到 stable_window 步
    for _ in range(3):
        orch.advance_step(plan_id)
    # 扩圈 50% = 15 台，全部置 succeeded（min_sample_size=5）
    orch.expand_ring(plan_id, 50)
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        recs = s.query(RolloutRecord).filter_by(
            model_id=plan.model_id, version=plan.version).all()
        for r in recs:
            r.state = RolloutState.succeeded.value
        s.commit()
    assert orch.check_stable_window(plan_id) is True


def test_check_stable_window_false_with_rollback(orch, plan_id, session_factory):
    for _ in range(3):
        orch.advance_step(plan_id)
    orch.expand_ring(plan_id, 50)
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        recs = s.query(RolloutRecord).filter_by(
            model_id=plan.model_id, version=plan.version).all()
        for r in recs:
            r.state = RolloutState.succeeded.value
        recs[0].state = RolloutState.rolling_back.value  # 有回滚
        s.commit()
    assert orch.check_stable_window(plan_id) is False


# --------------------------------------------------------------------------- #
# 7. 全量发布 → 所有设备覆盖
# --------------------------------------------------------------------------- #

def test_full_release_covers_all_devices(orch, plan_id, session_factory):
    recs = orch.full_release(plan_id)
    # 30 台设备全部创建记录
    assert len(recs) == 30
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        total = s.query(RolloutRecord).filter_by(
            model_id=plan.model_id, version=plan.version).count()
    assert total == 30
    # 进度推进到 full_release
    progress = orch.get_rollout_progress(plan_id)
    assert progress["current_step"] == RolloutStep.full_release.value
    assert progress["current_ring_percent"] == 100


def test_full_release_after_partial_expand(orch, plan_id, session_factory):
    """先扩圈 50%，再全量，应补充剩余 15 台。"""
    orch.expand_ring(plan_id, 50)
    recs = orch.full_release(plan_id)
    assert len(recs) == 15  # 30 - 15 已有
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        total = s.query(RolloutRecord).filter_by(
            model_id=plan.model_id, version=plan.version).count()
    assert total == 30


def test_full_release_frozen_raises(orch, plan_id, session_factory):
    with session_factory() as s:
        plan = s.get(RolloutPlan, plan_id)
        plan.frozen = True
        s.commit()
    with pytest.raises(ValueError, match="frozen"):
        orch.full_release(plan_id)


# --------------------------------------------------------------------------- #
# 8. 发布进度查询 → dict 结构
# --------------------------------------------------------------------------- #

def test_get_rollout_progress_structure(orch, plan_id):
    progress = orch.get_rollout_progress(plan_id)
    assert isinstance(progress, dict)
    assert progress["plan_id"] == plan_id
    assert progress["model_id"] == "anomaly-det"
    assert progress["version"] == "0.2.0"
    assert progress["frozen"] is False
    assert progress["current_step"] == RolloutStep.internal_test.value
    assert progress["current_ring_percent"] == 0
    assert progress["total_records"] == 0
    assert progress["state_counts"] == {}
    rc = progress["ring_config"]
    assert rc["rings"] == [1, 10, 50, 100]
    assert rc["stable_window_h"] == 24
    assert rc["rollback_window_h"] == 168
    assert rc["min_sample_size"] == 5


def test_get_rollout_progress_after_expand(orch, plan_id):
    orch.expand_ring(plan_id, 50)
    progress = orch.get_rollout_progress(plan_id)
    assert progress["total_records"] == 15
    assert progress["state_counts"][RolloutState.pending.value] == 15
    assert progress["current_ring_percent"] == 50


# --------------------------------------------------------------------------- #
# 9. 边界：空设备列表、无效 plan_id
# --------------------------------------------------------------------------- #

def test_select_devices_empty_when_no_devices(orch, session_factory):
    """无设备时返回空列表。"""
    assert orch.select_devices(100) == []


def test_select_devices_empty_db(orch, session_factory):
    """空数据库 + 站点过滤返回空。"""
    assert orch.select_devices(50, site="nowhere") == []


def test_get_rollout_progress_invalid_plan_id(orch):
    with pytest.raises(ValueError, match="not found"):
        orch.get_rollout_progress(99999)


def test_advance_step_invalid_plan_id(orch):
    """无效 plan_id 推进：索引默认 0，可推进到 canary（不校验 plan 存在）。"""
    # advance_step 不查 DB，仅操作内存索引；此处验证不抛异常且返回 canary
    step = orch.advance_step(99999)
    assert step == RolloutStep.canary


def test_advance_step_at_last_raises(orch, plan_id):
    """已在最后一步再推进抛 ValueError。"""
    for _ in range(4):
        orch.advance_step(plan_id)
    with pytest.raises(ValueError, match="last step"):
        orch.advance_step(plan_id)


def test_expand_ring_empty_devices(orch, session_factory):
    """无设备时扩圈返回空列表。"""
    cfg = RingConfig(rings=[1, 10, 50, 100])
    plan = orch.create_plan("m", "v", cfg)
    recs = orch.expand_ring(plan.id, 50)
    assert recs == []


def test_can_deploy_to_canary_invalid_plan(orch, session_factory):
    with pytest.raises(ValueError, match="not found"):
        orch.can_deploy_to_canary(99999)


def test_check_stable_window_invalid_plan(orch):
    with pytest.raises(ValueError, match="not found"):
        orch.check_stable_window(99999)