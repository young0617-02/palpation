"""端侧降级策略执行与 hysteresis 恢复。

按方案文档 §3.7 端侧降级映射实现三级降级矩阵：
- 内存 ≤3GB → normal（全功能）
- 内存 3–3.6GB → llm_small（LLM 限定 ≤32 token，T2 简化）
- 内存 >3.6GB → llm_off（LLM 卸载，仅规则+T2）
- 温度 >80°C → temp_throttle（降频，LLM 关闭）
- 离线 >24h → offline_degraded（仅 T1 告警链路，T2/T3 降级或关闭）
- 磁盘 <500MB → disk_critical（拒绝新模型下载）
- 多个高危条件同时触发 → emergency（仅 T1 告警）

DegradeManager 实现 hysteresis 恢复：升级降级立即触发，
恢复需要资源低于阈值 × (1 - margin) 才生效，避免频繁切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from edge.guard import GuardState


class DegradeLevel(Enum):
    """降级级别枚举，数值越大表示降级越严重。"""

    NORMAL = 0
    LLM_SMALL = 1
    LLM_OFF = 2
    TEMP_THROTTLE = 3
    OFFLINE_DEGRADED = 4
    DISK_CRITICAL = 5
    EMERGENCY = 6


@dataclass
class DegradeAction:
    """单个降级动作描述。"""

    name: str
    reason: str
    affected_components: list[str]


@dataclass
class DegradeThresholds:
    """降级阈值配置。

    内存阈值含义：mem_used_mb >= mem_llm_off_mb 触发 LLM 卸载；
    mem_used_mb >= mem_llm_small_mb 触发 LLM 限定。
    """

    mem_normal_mb: int = 3072
    mem_llm_small_mb: int = 3072
    mem_llm_off_mb: int = 3600
    temp_throttle_c: float = 80.0
    disk_critical_mb: int = 500
    offline_degraded_hours: int = 24


@dataclass
class DegradeDecision:
    """降级评估决策结果。"""

    level: DegradeLevel
    actions: list[DegradeAction]
    reason: str
    timestamp: datetime


class DegradePolicy:
    """降级策略：根据 GuardState 评估降级决策。

    评估优先级：多个高危条件（LLM_OFF / OFFLINE_DEGRADED / DISK_CRITICAL）
    同时触发时升级为 EMERGENCY；否则取触发的最高级别。
    """

    # 高危降级级别集合：同时触发 ≥2 个则升级为 EMERGENCY
    _HIGH_RISK: frozenset[DegradeLevel] = frozenset(
        {DegradeLevel.LLM_OFF, DegradeLevel.OFFLINE_DEGRADED, DegradeLevel.DISK_CRITICAL}
    )

    def __init__(self, thresholds: DegradeThresholds) -> None:
        self.thresholds = thresholds

    def evaluate(self, state: GuardState) -> DegradeDecision:
        """评估 GuardState，返回降级决策。"""
        triggered: list[DegradeLevel] = []
        reasons: list[str] = []
        t = self.thresholds

        # 内存降级（互斥：LLM_OFF 优先于 LLM_SMALL）
        if state.mem_used_mb >= t.mem_llm_off_mb:
            triggered.append(DegradeLevel.LLM_OFF)
            reasons.append(f"内存 {state.mem_used_mb}MB ≥ 卸载阈值 {t.mem_llm_off_mb}MB")
        elif state.mem_used_mb >= t.mem_llm_small_mb:
            triggered.append(DegradeLevel.LLM_SMALL)
            reasons.append(f"内存 {state.mem_used_mb}MB ≥ 限定阈值 {t.mem_llm_small_mb}MB")

        # 温度降频
        if state.cpu_temp_c is not None and state.cpu_temp_c > t.temp_throttle_c:
            triggered.append(DegradeLevel.TEMP_THROTTLE)
            reasons.append(f"CPU 温度 {state.cpu_temp_c}°C > 阈值 {t.temp_throttle_c}°C")

        # 离线降级
        offline_threshold_s = t.offline_degraded_hours * 3600
        if state.offline_duration_s > offline_threshold_s:
            triggered.append(DegradeLevel.OFFLINE_DEGRADED)
            reasons.append(f"离线 {state.offline_duration_s}s > 阈值 {offline_threshold_s}s")

        # 磁盘危急
        if state.disk_free_mb < t.disk_critical_mb:
            triggered.append(DegradeLevel.DISK_CRITICAL)
            reasons.append(f"磁盘可用 {state.disk_free_mb}MB < 阈值 {t.disk_critical_mb}MB")

        # 确定最终级别
        triggered_high_risk = set(triggered) & self._HIGH_RISK
        if len(triggered_high_risk) >= 2:
            level = DegradeLevel.EMERGENCY
        elif triggered:
            level = max(triggered, key=lambda lv: lv.value)
        else:
            level = DegradeLevel.NORMAL

        actions = self.actions_for_level(level)
        reason = "; ".join(reasons) if reasons else "资源正常，无需降级"
        return DegradeDecision(
            level=level, actions=actions, reason=reason, timestamp=datetime.now()
        )

    def actions_for_level(self, level: DegradeLevel) -> list[DegradeAction]:
        """返回该级别对应的降级动作列表。"""
        actions: dict[DegradeLevel, list[DegradeAction]] = {
            DegradeLevel.NORMAL: [],
            DegradeLevel.LLM_SMALL: [
                DegradeAction(
                    name="limit_llm_tokens",
                    reason="LLM 输出限定 ≤32 token",
                    affected_components=["t3_llm"],
                ),
                DegradeAction(
                    name="use_small_llm",
                    reason="降为 0.6B 小模型",
                    affected_components=["t3_llm"],
                ),
            ],
            DegradeLevel.LLM_OFF: [
                DegradeAction(
                    name="unload_llm",
                    reason="LLM 卸载，仅规则 + T2",
                    affected_components=["t3_llm"],
                ),
                DegradeAction(
                    name="simplify_t2",
                    reason="T2 简化（仅 Z-Score）",
                    affected_components=["t2_anomaly"],
                ),
            ],
            DegradeLevel.TEMP_THROTTLE: [
                DegradeAction(
                    name="throttle_llm",
                    reason="降低 T3 调用频率",
                    affected_components=["t3_llm"],
                ),
                DegradeAction(
                    name="close_llm_on_temp",
                    reason="高温关闭 LLM",
                    affected_components=["t3_llm"],
                ),
            ],
            DegradeLevel.OFFLINE_DEGRADED: [
                DegradeAction(
                    name="t1_alert_only",
                    reason="仅保留 T1 告警链路",
                    affected_components=["t1_rules"],
                ),
                DegradeAction(
                    name="degrade_t2_t3",
                    reason="T2/T3 降级或关闭",
                    affected_components=["t2_anomaly", "t3_llm"],
                ),
            ],
            DegradeLevel.DISK_CRITICAL: [
                DegradeAction(
                    name="reject_model_download",
                    reason="拒绝新模型下载",
                    affected_components=["ota"],
                ),
                DegradeAction(
                    name="clear_cache",
                    reason="清理 T4 检索缓存",
                    affected_components=["t4_retrieval"],
                ),
            ],
            DegradeLevel.EMERGENCY: [
                DegradeAction(
                    name="t1_alert_only",
                    reason="仅保留 T1 告警链路",
                    affected_components=["t1_rules"],
                ),
                DegradeAction(
                    name="unload_llm",
                    reason="LLM 卸载",
                    affected_components=["t3_llm"],
                ),
                DegradeAction(
                    name="degrade_t2",
                    reason="T2 降级",
                    affected_components=["t2_anomaly"],
                ),
            ],
        }
        return actions[level]


class DegradeManager:
    """降级管理器，含 hysteresis 恢复逻辑。

    升级降级立即触发；恢复需要资源低于阈值 × (1 - margin) 才生效，
    避免在阈值附近频繁切换降级级别。
    """

    def __init__(self, policy: DegradePolicy, hysteresis_margin: float = 0.1) -> None:
        self.policy = policy
        self.hysteresis_margin = hysteresis_margin
        self.current_level: DegradeLevel = DegradeLevel.NORMAL
        self._history: list[DegradeDecision] = []

    def evaluate(self, state: GuardState) -> DegradeDecision:
        """评估并更新降级级别（含 hysteresis 逻辑）。

        - 新级别更严重 → 立即升级
        - 新级别更轻 → 需满足 should_recover 才恢复，否则保持当前级别
        - 级别不变 → 直接返回
        """
        decision = self.policy.evaluate(state)

        if decision.level.value > self.current_level.value:
            self.current_level = decision.level
            self._history.append(decision)
            return decision
        if decision.level.value < self.current_level.value:
            if self.should_recover(state):
                self.current_level = decision.level
                self._history.append(decision)
                return decision
            held = DegradeDecision(
                level=self.current_level,
                actions=self.policy.actions_for_level(self.current_level),
                reason=f"hysteresis 保持 {self.current_level.name}（未达恢复阈值）",
                timestamp=datetime.now(),
            )
            self._history.append(held)
            return held
        self._history.append(decision)
        return decision

    def should_recover(self, state: GuardState) -> bool:
        """是否应该恢复：资源需低于阈值 × (1 - margin) 才允许恢复。

        磁盘方向相反：恢复需高于阈值 × (1 + margin)。
        EMERGENCY 恢复需所有高危条件均满足恢复阈值。
        """
        level = self.current_level
        t = self.policy.thresholds
        m = self.hysteresis_margin

        if level == DegradeLevel.NORMAL:
            return True
        if level == DegradeLevel.LLM_SMALL:
            return state.mem_used_mb < t.mem_llm_small_mb * (1 - m)
        if level == DegradeLevel.LLM_OFF:
            return state.mem_used_mb < t.mem_llm_off_mb * (1 - m)
        if level == DegradeLevel.TEMP_THROTTLE:
            return state.cpu_temp_c is not None and state.cpu_temp_c < t.temp_throttle_c * (1 - m)
        if level == DegradeLevel.OFFLINE_DEGRADED:
            return state.offline_duration_s < t.offline_degraded_hours * 3600 * (1 - m)
        if level == DegradeLevel.DISK_CRITICAL:
            return state.disk_free_mb > t.disk_critical_mb * (1 + m)
        if level == DegradeLevel.EMERGENCY:
            return (
                state.mem_used_mb < t.mem_llm_small_mb * (1 - m)
                and state.disk_free_mb > t.disk_critical_mb * (1 + m)
                and state.offline_duration_s < t.offline_degraded_hours * 3600 * (1 - m)
            )
        return False

    def apply(self, decision: DegradeDecision) -> None:
        """手动应用降级决策（更新当前级别并记录历史）。"""
        self.current_level = decision.level
        self._history.append(decision)

    def history(self) -> list[DegradeDecision]:
        """返回降级历史记录（副本）。"""
        return list(self._history)
