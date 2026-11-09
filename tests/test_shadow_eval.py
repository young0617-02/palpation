"""cloud/shadow_eval.py 单元测试：影子推理 + KPI 比对 + 统计功效 + 切流决策。

覆盖：
1. 计算单版本 KPI → 验证各指标计算正确
2. 比对两版 KPI → 验证 passed 判定
3. 置信区间计算 → 验证 Wilson interval
4. 统计功效检查 → 满足 / 不满足
5. 切流决策 → 达标 / 不达标
6. 批量比对 → 验证多设备聚合
7. 聚合统计 → 验证通过率计算
8. 影子推理记录 → 验证不生效（只记录）
9. 边界：空样本、样本不足、极端值
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import Base, RolloutRecord, ShadowRow
from cloud.shadow_eval import (
    KpiMetric,
    KpiThreshold,
    ShadowEvaluator,
    ShadowInferenceRow,
    ShadowResult,
)
from shared.schemas import EdgeEvent

# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture()
def session_factory():
    """每个用例独立的内存 sqlite session_factory。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def evaluator(session_factory):
    """min_samples=10 便于测试构造。"""
    return ShadowEvaluator(session_factory, min_samples=10, stable_window_h=24)


# ====================================================================
# Helpers
# ====================================================================


def _make_event(**overrides) -> EdgeEvent:
    base = dict(
        event_id="e1",
        gw_id="gw-001",
        device_id_hash="abc",
        seq=1,
        detected_at=datetime.now(UTC),
        rule_id="bearing_rms_high",
        severity="medium",
        metric="vibration_rms",
        value=6.0,
        threshold=4.5,
        preliminary_diagnosis="bearing_wear",
    )
    base.update(overrides)
    return EdgeEvent(**base)


def _record_batch(
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
    """批量记录影子推理，便于构造测试数据。

    - n_agree 条端侧初判与云端结论一致，其余不一致
    - latency = latency_base + i * latency_step
    - n_json_ok 条 JSON 可解析成功
    - n_false_positive 条误报
    """
    if n_agree is None:
        n_agree = n
    if n_json_ok is None:
        n_json_ok = n
    for i in range(n):
        event = _make_event(
            event_id=f"{gw_id}-{model_version}-{i}",
            gw_id=gw_id,
            preliminary_diagnosis=preliminary,
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


# ====================================================================
# 1. 计算单版本 KPI
# ====================================================================


def test_compute_kpi_single_version(evaluator):
    """记录 10 条推理，验证各 KPI 指标计算正确。"""
    _record_batch(
        evaluator,
        gw_id="gw-001",
        model_version="1.0",
        n=10,
        n_agree=8,        # 一致率 0.8
        latency_base=10,
        latency_step=10,  # [10,20,...,100] → P95=95.5
        peak_mem=200,
        n_json_ok=9,      # JSON 可解析率 0.9
        n_false_positive=1,  # 误报率 0.1
    )

    kpi = evaluator.compute_kpi("gw-001", "1.0")
    assert kpi.agreement_rate == pytest.approx(0.8)
    # P95: k = 9 * 0.95 = 8.55, f=8, c=9 → 90 + 0.55*(100-90) = 95.5
    assert kpi.p95_latency_ms == pytest.approx(95.5)
    assert kpi.peak_mem_mb == pytest.approx(200)
    assert kpi.json_parse_rate == pytest.approx(0.9)
    assert kpi.false_positive_rate == pytest.approx(0.1)


def test_compute_kpi_agreement_only_comparable(evaluator):
    """cloud_conclusion 为 None 的记录不计入一致率分母。"""
    # 3 条有 cloud_conclusion（2 条一致），2 条 cloud_conclusion=None
    for i in range(3):
        evaluator.record_inference(
            "gw-001", "1.0", _make_event(event_id=f"e{i}"),
            "bearing_wear" if i < 2 else "other",
        )
    for i in range(2):
        evaluator.record_inference(
            "gw-001", "1.0", _make_event(event_id=f"n{i}"), None,
        )

    kpi = evaluator.compute_kpi("gw-001", "1.0")
    # 可比较的 3 条中 2 条一致 → 2/3
    assert kpi.agreement_rate == pytest.approx(2.0 / 3.0)
    # JSON 可解析率仍按全部 5 条计算
    assert kpi.json_parse_rate == pytest.approx(1.0)


# ====================================================================
# 2. 比对两版 KPI → passed 判定
# ====================================================================


def test_compare_passed(evaluator):
    """new 版 KPI 优于 baseline 且样本充足 → passed=True。"""
    # baseline: agreement=0.8, p95=95.5, peak_mem=200, json=0.9, fp=0.1
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=8,
        latency_base=10, latency_step=10, peak_mem=200,
        n_json_ok=9, n_false_positive=1,
    )
    # new: agreement=0.9, p95=76.4, peak_mem=180, json=1.0, fp=0.0
    _record_batch(
        evaluator, "gw-001", "2.0", n=10, n_agree=9,
        latency_base=8, latency_step=8, peak_mem=180,
        n_json_ok=10, n_false_positive=0,
    )

    result = evaluator.compare("gw-001", "2.0", "1.0")
    assert isinstance(result, ShadowResult)
    assert result.gw_id == "gw-001"
    assert result.new_version == "2.0"
    assert result.baseline_version == "1.0"
    assert result.sample_size == 10
    assert result.passed is True
    # 置信区间应标注
    lo, hi = result.confidence_interval
    assert 0.0 <= lo <= hi <= 1.0


def test_compare_failed_due_to_agreement(evaluator):
    """new 版一致率劣化超 2pp → passed=False。"""
    # baseline: agreement=0.8
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=8,
        peak_mem=200, n_json_ok=10,
    )
    # new: agreement=0.5（远低于 0.8-0.02=0.78）
    _record_batch(
        evaluator, "gw-001", "2.0", n=10, n_agree=5,
        peak_mem=180, n_json_ok=10,
    )

    result = evaluator.compare("gw-001", "2.0", "1.0")
    assert result.passed is False


def test_compare_failed_due_to_latency(evaluator):
    """new 版 P95 延迟超过 baseline × 1.2 → passed=False。"""
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=10,
        latency_base=10, latency_step=10, peak_mem=200, n_json_ok=10,
    )
    # new P95 远超 95.5 * 1.2 = 114.6
    _record_batch(
        evaluator, "gw-001", "2.0", n=10, n_agree=10,
        latency_base=200, latency_step=10, peak_mem=180, n_json_ok=10,
    )

    result = evaluator.compare("gw-001", "2.0", "1.0")
    assert result.passed is False


# ====================================================================
# 3. 置信区间计算 → Wilson interval
# ====================================================================


def test_confidence_interval_wilson(evaluator):
    """验证 Wilson score interval 计算。"""
    # rate=0.5, n=100, confidence=0.95
    lo, hi = evaluator.confidence_interval(0.5, 100, confidence=0.95)
    # z ≈ 1.96, 期望区间约 (0.404, 0.596)
    assert lo == pytest.approx(0.404, abs=0.005)
    assert hi == pytest.approx(0.596, abs=0.005)
    assert lo < 0.5 < hi


def test_confidence_interval_n_zero(evaluator):
    """n=0 时返回 (0.0, 1.0) 全区间。"""
    lo, hi = evaluator.confidence_interval(0.5, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_confidence_interval_extreme_rate(evaluator):
    """rate=0 和 rate=1 时区间不越界。"""
    lo0, hi0 = evaluator.confidence_interval(0.0, 100)
    assert lo0 == 0.0
    assert 0.0 <= hi0 <= 1.0

    lo1, hi1 = evaluator.confidence_interval(1.0, 100)
    assert 0.0 <= lo1 <= 1.0
    assert hi1 == 1.0


def test_confidence_interval_higher_confidence_wider(evaluator):
    """更高置信度 → 更宽区间。"""
    lo95, hi95 = evaluator.confidence_interval(0.5, 50, confidence=0.95)
    lo99, hi99 = evaluator.confidence_interval(0.5, 50, confidence=0.99)
    assert (hi99 - lo99) > (hi95 - lo95)


def test_confidence_interval_larger_n_narrower(evaluator):
    """更大样本量 → 更窄区间。"""
    lo_small, hi_small = evaluator.confidence_interval(0.5, 30)
    lo_large, hi_large = evaluator.confidence_interval(0.5, 300)
    assert (hi_large - lo_large) < (hi_small - lo_small)


# ====================================================================
# 4. 统计功效检查
# ====================================================================


def test_check_statistical_power_satisfied(evaluator):
    """样本量与时长均达标 → True。"""
    assert evaluator.check_statistical_power(sample_size=100, duration_h=24.0) is True
    assert evaluator.check_statistical_power(sample_size=200, duration_h=48.0) is True


def test_check_statistical_power_insufficient_samples(evaluator):
    """样本量不足 → False。"""
    assert evaluator.check_statistical_power(sample_size=9, duration_h=24.0) is False


def test_check_statistical_power_insufficient_duration(evaluator):
    """时长不足 → False。"""
    assert evaluator.check_statistical_power(sample_size=100, duration_h=23.9) is False


def test_check_statistical_power_boundary(evaluator):
    """边界值：恰好等于阈值 → True。"""
    assert evaluator.check_statistical_power(sample_size=10, duration_h=24.0) is True


# ====================================================================
# 5. 切流决策 → should_promote
# ====================================================================


def test_should_promote_all_metrics_pass(evaluator):
    """所有 KPI 达标且样本充足 → True。"""
    result = ShadowResult(
        gw_id="gw-001", model_id="", new_version="2.0", baseline_version="1.0",
        new_kpi=KpiMetric(0.9, 80, 180, 0.96, 0.05),
        baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
        sample_size=100,
        confidence_interval=(0.85, 0.95),
        passed=False,
    )
    thresholds = KpiThreshold(0.78, 120, 230, 0.95, 0.1)
    assert evaluator.should_promote(result, thresholds) is True


def test_should_promote_insufficient_samples(evaluator):
    """样本不足 → False（即使 KPI 达标）。"""
    result = ShadowResult(
        gw_id="gw-001", model_id="", new_version="2.0", baseline_version="1.0",
        new_kpi=KpiMetric(0.9, 80, 180, 0.96, 0.05),
        baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
        sample_size=5,
        confidence_interval=(0.85, 0.95),
        passed=False,
    )
    thresholds = KpiThreshold(0.78, 120, 230, 0.95, 0.1)
    assert evaluator.should_promote(result, thresholds) is False


def test_should_promote_agreement_fail(evaluator):
    """一致率不达标 → False。"""
    result = ShadowResult(
        gw_id="gw-001", model_id="", new_version="2.0", baseline_version="1.0",
        new_kpi=KpiMetric(0.70, 80, 180, 0.96, 0.05),
        baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
        sample_size=100,
        confidence_interval=(0.65, 0.75),
        passed=False,
    )
    thresholds = KpiThreshold(0.78, 120, 230, 0.95, 0.1)
    assert evaluator.should_promote(result, thresholds) is False


def test_should_promote_json_fail(evaluator):
    """JSON 可解析率不达标 → False。"""
    result = ShadowResult(
        gw_id="gw-001", model_id="", new_version="2.0", baseline_version="1.0",
        new_kpi=KpiMetric(0.9, 80, 180, 0.90, 0.05),
        baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
        sample_size=100,
        confidence_interval=(0.85, 0.95),
        passed=False,
    )
    thresholds = KpiThreshold(0.78, 120, 230, 0.95, 0.1)
    assert evaluator.should_promote(result, thresholds) is False


def test_should_promote_false_positive_fail(evaluator):
    """误报率劣化 → False。"""
    result = ShadowResult(
        gw_id="gw-001", model_id="", new_version="2.0", baseline_version="1.0",
        new_kpi=KpiMetric(0.9, 80, 180, 0.96, 0.15),
        baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
        sample_size=100,
        confidence_interval=(0.85, 0.95),
        passed=False,
    )
    thresholds = KpiThreshold(0.78, 120, 230, 0.95, 0.1)
    assert evaluator.should_promote(result, thresholds) is False


# ====================================================================
# 6. 批量比对
# ====================================================================


def test_batch_compare(evaluator):
    """3 个设备批量比对，返回 3 个 ShadowResult。"""
    for gw_id in ["gw-001", "gw-002", "gw-003"]:
        _record_batch(
            evaluator, gw_id, "1.0", n=10, n_agree=8,
            peak_mem=200, n_json_ok=10,
        )
        _record_batch(
            evaluator, gw_id, "2.0", n=10, n_agree=9,
            peak_mem=180, n_json_ok=10,
        )

    results = evaluator.batch_compare(["gw-001", "gw-002", "gw-003"], "2.0", "1.0")
    assert len(results) == 3
    assert all(isinstance(r, ShadowResult) for r in results)
    assert {r.gw_id for r in results} == {"gw-001", "gw-002", "gw-003"}
    assert all(r.passed for r in results)


def test_batch_compare_empty_list(evaluator):
    """空设备列表 → 空结果。"""
    results = evaluator.batch_compare([], "2.0", "1.0")
    assert results == []


# ====================================================================
# 7. 聚合统计
# ====================================================================


def test_aggregate_results(evaluator):
    """验证通过率与平均指标计算。"""
    results = [
        ShadowResult(
            gw_id="gw-001", model_id="", new_version="2.0", baseline_version="1.0",
            new_kpi=KpiMetric(0.9, 80, 180, 0.96, 0.05),
            baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
            sample_size=100, confidence_interval=(0.85, 0.95), passed=True,
        ),
        ShadowResult(
            gw_id="gw-002", model_id="", new_version="2.0", baseline_version="1.0",
            new_kpi=KpiMetric(0.7, 150, 250, 0.90, 0.15),
            baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
            sample_size=100, confidence_interval=(0.65, 0.75), passed=False,
        ),
        ShadowResult(
            gw_id="gw-003", model_id="", new_version="2.0", baseline_version="1.0",
            new_kpi=KpiMetric(0.85, 90, 190, 0.97, 0.03),
            baseline_kpi=KpiMetric(0.8, 100, 200, 0.9, 0.1),
            sample_size=100, confidence_interval=(0.80, 0.90), passed=True,
        ),
    ]
    agg = evaluator.aggregate_results(results)
    assert agg["total"] == 3
    assert agg["passed"] == 2
    assert agg["failed"] == 1
    assert agg["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert agg["avg_agreement_rate"] == pytest.approx((0.9 + 0.7 + 0.85) / 3)
    assert agg["avg_p95_latency_ms"] == pytest.approx((80 + 150 + 90) / 3)
    assert agg["avg_peak_mem_mb"] == pytest.approx((180 + 250 + 190) / 3)
    assert agg["avg_json_parse_rate"] == pytest.approx((0.96 + 0.90 + 0.97) / 3)
    assert agg["avg_false_positive_rate"] == pytest.approx((0.05 + 0.15 + 0.03) / 3)
    assert agg["avg_sample_size"] == pytest.approx(100.0)


def test_aggregate_results_empty(evaluator):
    """空结果列表 → 全零聚合。"""
    agg = evaluator.aggregate_results([])
    assert agg["total"] == 0
    assert agg["passed"] == 0
    assert agg["failed"] == 0
    assert agg["pass_rate"] == 0.0
    assert agg["avg_agreement_rate"] == 0.0


# ====================================================================
# 8. 影子推理记录 → 不生效
# ====================================================================


def test_record_inference_no_side_effect(session_factory):
    """记录影子推理不修改 ShadowRow / RolloutRecord，只写入 ShadowInferenceRow。"""
    # 预置 ShadowRow 和 RolloutRecord
    with session_factory() as s:
        s.add(ShadowRow(
            gw_id="gw-001",
            desired={"model_version": "1.0"},
            reported={"model_version": "1.0"},
        ))
        s.add(RolloutRecord(
            gw_id="gw-001", model_id="anomaly", version="1.0",
            state="succeeded", detail={},
        ))
        s.commit()

    evaluator = ShadowEvaluator(session_factory, min_samples=10)
    event = _make_event(event_id="evt-shadow-1")
    evaluator.record_inference(
        "gw-001", "2.0", event, "bearing_wear",
        latency_ms=50, peak_mem_mb=200,
    )

    with session_factory() as s:
        # ShadowRow 未被修改
        shadow = s.get(ShadowRow, "gw-001")
        assert shadow.desired == {"model_version": "1.0"}
        assert shadow.reported == {"model_version": "1.0"}
        # RolloutRecord 未被修改
        rec = s.query(RolloutRecord).one()
        assert rec.state == "succeeded"
        assert rec.version == "1.0"
        # ShadowInferenceRow 有记录
        inf = s.query(ShadowInferenceRow).one()
        assert inf.gw_id == "gw-001"
        assert inf.model_version == "2.0"
        assert inf.event_id == "evt-shadow-1"
        assert inf.preliminary_diagnosis == "bearing_wear"
        assert inf.cloud_conclusion == "bearing_wear"
        assert inf.latency_ms == 50.0
        assert inf.peak_mem_mb == 200.0


def test_record_inference_default_values(session_factory):
    """record_inference 默认性能指标为 0 / True / False。"""
    evaluator = ShadowEvaluator(session_factory)
    event = _make_event(event_id="evt-default")
    evaluator.record_inference("gw-001", "1.0", event, "bearing_wear")

    with session_factory() as s:
        inf = s.query(ShadowInferenceRow).one()
        assert inf.latency_ms == 0.0
        assert inf.peak_mem_mb == 0.0
        assert inf.json_parse_ok is True
        assert inf.is_false_positive is False


# ====================================================================
# 9. 边界：空样本、样本不足、极端值
# ====================================================================


def test_compute_kpi_empty_samples(evaluator):
    """无记录的版本 → KPI 全 0。"""
    kpi = evaluator.compute_kpi("gw-999", "9.9")
    assert kpi.agreement_rate == 0.0
    assert kpi.p95_latency_ms == 0.0
    assert kpi.peak_mem_mb == 0.0
    assert kpi.json_parse_rate == 0.0
    assert kpi.false_positive_rate == 0.0


def test_compare_empty_samples(evaluator):
    """两版均无记录 → passed=False（样本不足）。"""
    result = evaluator.compare("gw-999", "2.0", "1.0")
    assert result.sample_size == 0
    assert result.passed is False
    assert result.confidence_interval == (0.0, 1.0)


def test_compare_insufficient_samples(evaluator):
    """new 版样本不足 min_samples → passed=False。"""
    # baseline 充足
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=10,
        peak_mem=200, n_json_ok=10,
    )
    # new 仅 5 条（< min_samples=10）
    _record_batch(
        evaluator, "gw-001", "2.0", n=5, n_agree=5,
        peak_mem=180, n_json_ok=5,
    )

    result = evaluator.compare("gw-001", "2.0", "1.0")
    assert result.sample_size == 5
    assert result.passed is False


def test_compute_kpi_extreme_values(evaluator):
    """极端值：全零延迟 / 内存，全一致，全 JSON 可解析，无误报。"""
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=10,
        latency_base=0, latency_step=0, peak_mem=0,
        n_json_ok=10, n_false_positive=0,
    )

    kpi = evaluator.compute_kpi("gw-001", "1.0")
    assert kpi.agreement_rate == 1.0
    assert kpi.p95_latency_ms == 0.0
    assert kpi.peak_mem_mb == 0.0
    assert kpi.json_parse_rate == 1.0
    assert kpi.false_positive_rate == 0.0


def test_compute_kpi_all_false_positive(evaluator):
    """全部误报 → 误报率 1.0。"""
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=10,
        n_json_ok=10, n_false_positive=10,
    )

    kpi = evaluator.compute_kpi("gw-001", "1.0")
    assert kpi.false_positive_rate == 1.0


def test_compute_kpi_all_json_parse_fail(evaluator):
    """全部 JSON 解析失败 → 可解析率 0.0。"""
    _record_batch(
        evaluator, "gw-001", "1.0", n=10, n_agree=10,
        n_json_ok=0,
    )

    kpi = evaluator.compute_kpi("gw-001", "1.0")
    assert kpi.json_parse_rate == 0.0


def test_percentile_single_value(evaluator):
    """单条记录的 P95 = 该值。"""
    evaluator.record_inference(
        "gw-001", "1.0", _make_event(event_id="e0"), "bearing_wear",
        latency_ms=42.0,
    )
    kpi = evaluator.compute_kpi("gw-001", "1.0")
    assert kpi.p95_latency_ms == 42.0


def test_default_thresholds_from_baseline():
    """default_thresholds 从 baseline KPI 正确推导。"""
    baseline = KpiMetric(
        agreement_rate=0.80, p95_latency_ms=100, peak_mem_mb=200,
        json_parse_rate=0.90, false_positive_rate=0.10,
    )
    t = ShadowEvaluator.default_thresholds(baseline)
    assert t.agreement_rate_min == pytest.approx(0.78)
    assert t.p95_latency_max == pytest.approx(120)
    assert t.peak_mem_max == pytest.approx(230)
    assert t.json_parse_rate_min == pytest.approx(0.95)
    assert t.false_positive_rate_max == pytest.approx(0.10)
