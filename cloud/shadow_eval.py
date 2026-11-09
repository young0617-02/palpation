"""影子验证（Shadow Inference）：新模型在 Canary 组与在线模型并行推理，
结果只记录、不生效。稳定窗口内比对两版 KPI，达标才允许切流。

机制（方案 §4.6）：
- 新模型与在线模型并行推理，结果只记录、不生效
- 稳定窗口内比对两版 KPI（一致率 / P95 延迟 / 峰值内存 / JSON 可解析率 / 误报率）
- 统计功效保障：≥24h + 最少样本量 + Wilson 置信区间
- 达标才允许切流

KPI 比对阈值（示例）：
| 比对指标                 | 阈值              |
|--------------------------|-------------------|
| 端侧初判与云端结论一致率 | ≥ 现网版本 − 2pp  |
| 推理 P95 延迟            | ≤ 现网版本 × 1.2  |
| 峰值内存                 | ≤ 现网版本 × 1.15 |
| JSON 可解析率            | ≥ 95%             |
| 误报率                   | 不劣化            |
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from cloud.db import Base
from shared.schemas import EdgeEvent

log = logging.getLogger(__name__)

# Wilson 置信区间默认置信度
_DEFAULT_CONFIDENCE = 0.95
# KPI 阈值默认系数
_AGREEMENT_DEGRADATION_PP = 0.02  # 一致率允许劣化 2pp
_LATENCY_TOLERANCE = 1.2  # P95 延迟允许放大 1.2 倍
_MEM_TOLERANCE = 1.15  # 峰值内存允许放大 1.15 倍
_JSON_PARSE_RATE_MIN = 0.95  # JSON 可解析率最低 95%


class ShadowInferenceRow(Base):
    """影子推理记录表（只记录、不生效）。

    每行记录一次新模型（或基线模型）的影子推理结果，
    不影响 ShadowRow.desired/reported，也不影响 RolloutRecord.state。
    """

    __tablename__ = "shadow_inference"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gw_id: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    event_id: Mapped[str] = mapped_column(String(64))
    preliminary_diagnosis: Mapped[str | None] = mapped_column(String(256), nullable=True)
    cloud_conclusion: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    peak_mem_mb: Mapped[float] = mapped_column(Float, default=0.0)
    json_parse_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    is_false_positive: Mapped[bool] = mapped_column(Boolean, default=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


@dataclass
class KpiMetric:
    """单版本 KPI 指标。"""

    agreement_rate: float  # 端侧初判与云端结论一致率
    p95_latency_ms: float  # 推理 P95 延迟
    peak_mem_mb: float  # 峰值内存
    json_parse_rate: float  # JSON 可解析率
    false_positive_rate: float  # 误报率


@dataclass
class KpiThreshold:
    """KPI 比对阈值。"""

    agreement_rate_min: float  # 现网版本 - 2pp
    p95_latency_max: float  # 现网版本 × 1.2
    peak_mem_max: float  # 现网版本 × 1.15
    json_parse_rate_min: float  # 0.95
    false_positive_rate_max: float  # 不劣化


@dataclass
class ShadowResult:
    """影子验证比对结果。"""

    gw_id: str
    model_id: str
    new_version: str
    baseline_version: str
    new_kpi: KpiMetric
    baseline_kpi: KpiMetric
    sample_size: int
    confidence_interval: tuple[float, float]
    passed: bool


class ShadowEvaluator:
    """影子验证评估器。

    新模型在 Canary 组与在线模型并行推理，结果只记录、不生效。
    稳定窗口内比对两版 KPI，达标才允许切流。

    统计功效保障（RE10 风险应对）：
    - Canary 组维持足够时长（≥ stable_window_h，默认 24h）
    - 最少样本量门槛（≥ min_samples，默认 100）
    - 结论标注 Wilson 置信区间
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        min_samples: int = 100,
        stable_window_h: int = 24,
    ) -> None:
        self.session_factory = session_factory
        self.min_samples = min_samples
        self.stable_window_h = stable_window_h

    # ------------------------------------------------------------------ #
    # 影子推理记录
    # ------------------------------------------------------------------ #

    def record_inference(
        self,
        gw_id: str,
        model_version: str,
        event: EdgeEvent,
        cloud_conclusion: str | None,
        *,
        latency_ms: float = 0.0,
        peak_mem_mb: float = 0.0,
        json_parse_ok: bool = True,
        is_false_positive: bool = False,
    ) -> None:
        """记录一次影子推理结果（只记录、不生效）。

        不修改 ShadowRow 的 desired/reported，不修改 RolloutRecord 的 state。
        额外的性能指标（延迟 / 内存 / JSON 解析 / 误报）通过关键字参数传入，
        因为 EdgeEvent schema 不包含这些字段。
        """
        row = ShadowInferenceRow(
            gw_id=gw_id,
            model_version=model_version,
            event_id=event.event_id,
            preliminary_diagnosis=event.preliminary_diagnosis,
            cloud_conclusion=cloud_conclusion,
            latency_ms=latency_ms,
            peak_mem_mb=peak_mem_mb,
            json_parse_ok=json_parse_ok,
            is_false_positive=is_false_positive,
        )
        with self.session_factory() as s:
            s.add(row)
            s.commit()

    # ------------------------------------------------------------------ #
    # KPI 计算
    # ------------------------------------------------------------------ #

    def compute_kpi(self, gw_id: str, model_version: str) -> KpiMetric:
        """计算某版本的 KPI 指标。

        - 一致率：仅当端侧初判与云端结论均非空时才计入分母
        - P95 延迟：所有记录延迟的第 95 百分位
        - 峰值内存：所有记录中的最大峰值内存
        - JSON 可解析率：json_parse_ok 为 True 的比例
        - 误报率：is_false_positive 为 True 的比例
        """
        with self.session_factory() as s:
            rows = (
                s.query(ShadowInferenceRow)
                .filter_by(gw_id=gw_id, model_version=model_version)
                .all()
            )
            n = len(rows)
            if n == 0:
                return KpiMetric(0.0, 0.0, 0.0, 0.0, 0.0)

            # 一致率：仅当端侧初判与云端结论均非空时才计入分母
            comparable = [
                r
                for r in rows
                if r.preliminary_diagnosis is not None and r.cloud_conclusion is not None
            ]
            if comparable:
                agreed = sum(1 for r in comparable if r.preliminary_diagnosis == r.cloud_conclusion)
                agreement_rate = agreed / len(comparable)
            else:
                agreement_rate = 0.0

            # P95 延迟
            p95 = self._percentile([r.latency_ms for r in rows], 95.0)
            # 峰值内存
            peak_mem = max(r.peak_mem_mb for r in rows)
            # JSON 可解析率
            json_rate = sum(1 for r in rows if r.json_parse_ok) / n
            # 误报率
            fp_rate = sum(1 for r in rows if r.is_false_positive) / n

            return KpiMetric(
                agreement_rate=agreement_rate,
                p95_latency_ms=p95,
                peak_mem_mb=peak_mem,
                json_parse_rate=json_rate,
                false_positive_rate=fp_rate,
            )

    # ------------------------------------------------------------------ #
    # 比对与切流决策
    # ------------------------------------------------------------------ #

    def compare(
        self,
        gw_id: str,
        new_version: str,
        baseline_version: str,
        *,
        model_id: str = "",
    ) -> ShadowResult:
        """比对两版 KPI，返回 ShadowResult。

        使用从 baseline_kpi 推导的默认阈值判断 passed。
        confidence_interval 标注 new_version 一致率的 Wilson 置信区间。
        """
        new_kpi = self.compute_kpi(gw_id, new_version)
        baseline_kpi = self.compute_kpi(gw_id, baseline_version)
        sample_size = self._count_samples(gw_id, new_version)
        ci = self.confidence_interval(new_kpi.agreement_rate, sample_size)
        thresholds = self.default_thresholds(baseline_kpi)

        result = ShadowResult(
            gw_id=gw_id,
            model_id=model_id,
            new_version=new_version,
            baseline_version=baseline_version,
            new_kpi=new_kpi,
            baseline_kpi=baseline_kpi,
            sample_size=sample_size,
            confidence_interval=ci,
            passed=False,
        )
        result.passed = self.should_promote(result, thresholds)
        return result

    def should_promote(self, result: ShadowResult, thresholds: KpiThreshold) -> bool:
        """是否达标可切流。

        需同时满足：
        1. 样本量 ≥ min_samples（统计功效保障）
        2. 一致率 ≥ 阈值
        3. P95 延迟 ≤ 阈值
        4. 峰值内存 ≤ 阈值
        5. JSON 可解析率 ≥ 阈值
        6. 误报率 ≤ 阈值
        """
        if result.sample_size < self.min_samples:
            return False
        kpi = result.new_kpi
        return (
            kpi.agreement_rate >= thresholds.agreement_rate_min
            and kpi.p95_latency_ms <= thresholds.p95_latency_max
            and kpi.peak_mem_mb <= thresholds.peak_mem_max
            and kpi.json_parse_rate >= thresholds.json_parse_rate_min
            and kpi.false_positive_rate <= thresholds.false_positive_rate_max
        )

    def batch_compare(
        self, gw_ids: list[str], new_version: str, baseline_version: str
    ) -> list[ShadowResult]:
        """批量比对多设备。"""
        return [self.compare(gw_id, new_version, baseline_version) for gw_id in gw_ids]

    def aggregate_results(self, results: list[ShadowResult]) -> dict:
        """聚合统计：通过率、平均指标等。"""
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_agreement_rate": 0.0,
                "avg_p95_latency_ms": 0.0,
                "avg_peak_mem_mb": 0.0,
                "avg_json_parse_rate": 0.0,
                "avg_false_positive_rate": 0.0,
                "avg_sample_size": 0.0,
            }
        passed_count = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": passed_count / total,
            "avg_agreement_rate": sum(r.new_kpi.agreement_rate for r in results) / total,
            "avg_p95_latency_ms": sum(r.new_kpi.p95_latency_ms for r in results) / total,
            "avg_peak_mem_mb": sum(r.new_kpi.peak_mem_mb for r in results) / total,
            "avg_json_parse_rate": sum(r.new_kpi.json_parse_rate for r in results) / total,
            "avg_false_positive_rate": sum(r.new_kpi.false_positive_rate for r in results) / total,
            "avg_sample_size": sum(r.sample_size for r in results) / total,
        }

    # ------------------------------------------------------------------ #
    # 统计功效
    # ------------------------------------------------------------------ #

    def check_statistical_power(self, sample_size: int, duration_h: float) -> bool:
        """检查统计功效是否足够。

        需同时满足：
        - 样本量 ≥ min_samples
        - 观测时长 ≥ stable_window_h
        """
        return sample_size >= self.min_samples and duration_h >= self.stable_window_h

    def confidence_interval(
        self, rate: float, n: int, confidence: float = _DEFAULT_CONFIDENCE
    ) -> tuple[float, float]:
        """计算 Wilson 置信区间。

        Wilson score interval 比 Normal approximation 更稳健，
        在 rate 接近 0 或 1 时不会越界。
        """
        if n <= 0:
            return (0.0, 1.0)
        # clamp rate to [0, 1]
        p = max(0.0, min(1.0, rate))
        z = self._z_score(confidence)
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def default_thresholds(baseline: KpiMetric) -> KpiThreshold:
        """根据现网版本 KPI 生成默认阈值。"""
        return KpiThreshold(
            agreement_rate_min=baseline.agreement_rate - _AGREEMENT_DEGRADATION_PP,
            p95_latency_max=baseline.p95_latency_ms * _LATENCY_TOLERANCE,
            peak_mem_max=baseline.peak_mem_mb * _MEM_TOLERANCE,
            json_parse_rate_min=_JSON_PARSE_RATE_MIN,
            false_positive_rate_max=baseline.false_positive_rate,
        )

    def _count_samples(self, gw_id: str, model_version: str) -> int:
        """统计某设备某版本的影子推理记录数。"""
        with self.session_factory() as s:
            return (
                s.query(ShadowInferenceRow)
                .filter_by(gw_id=gw_id, model_version=model_version)
                .count()
            )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """线性插值法计算百分位数。"""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        k = (len(sorted_vals) - 1) * percentile / 100.0
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    @staticmethod
    def _z_score(confidence: float) -> float:
        """根据置信度计算标准正态 z 分数。"""
        return NormalDist().inv_cdf((1.0 + confidence) / 2.0)
