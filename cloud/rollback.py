"""cloud/rollback.py：回滚决策与冻结发布管理。

依据方案 §4.7 + RE4/RE8：
- 回滚触发信号（推理链路级，而非设备在线级）
- 三层版本按层回滚（firmware/model/config，§4.4）
- 冻结发布管理（阻止问题版本扩散）
- 健康趋势检查（持续异常检测）

RE8 风险应对：管理代理与主应用进程隔离。本模块运行于云端控制面，
通过 RolloutRecord/RolloutPlan 表与端侧 mgmt_agent/ota.py 协作执行回滚。
"决策权在云，执行权在边"——本模块只做决策与状态编排，
实际端侧回滚由 mgmt_agent/ota.py 的 OtaEngine/VersionSlots 执行。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from sqlalchemy.orm import sessionmaker

from cloud.db import EdgeHeartbeat, RolloutPlan, RolloutRecord
from shared.schemas import EdgeHealth

log = logging.getLogger(__name__)


class RollbackSignal(StrEnum):
    """回滚触发信号（必须是"推理链路"级，而非"设备在线"级，§4.7）。"""

    process_start_failure = "process_start_failure"
    probe_failure = "probe_failure"
    p95_latency_exceeded = "p95_latency_exceeded"
    mem_exceeded = "mem_exceeded"
    json_parse_rate_drop = "json_parse_rate_drop"
    input_link_down = "input_link_down"


class VersionLayer(StrEnum):
    """三层版本解耦（§4.4）：firmware / model / config。

    - firmware：驱动、采集、推理运行时（需重启/切分区）
    - model：权重、量化文件、前/后处理（可热切换，符号链接）
    - config：阈值、采样策略、开关（秒级生效）
    """

    firmware = "firmware"
    model = "model"
    config = "config"


# 信号 → 严重性（§4.7 触发信号表）
_SEVERITY_MAP: dict[RollbackSignal, Literal["immediate", "delayed", "alert_only"]] = {
    RollbackSignal.process_start_failure: "immediate",
    RollbackSignal.probe_failure: "immediate",
    RollbackSignal.p95_latency_exceeded: "delayed",
    RollbackSignal.mem_exceeded: "delayed",
    RollbackSignal.json_parse_rate_drop: "delayed",
    RollbackSignal.input_link_down: "alert_only",
}


# 信号 → 回滚层级（§4.4 三层解耦 + §4.7 回滚对象）
_LAYER_MAP: dict[RollbackSignal, VersionLayer] = {
    # 进程/探针失败 → firmware 层（需重启/切分区）
    RollbackSignal.process_start_failure: VersionLayer.firmware,
    RollbackSignal.probe_failure: VersionLayer.firmware,
    # 推理质量/格式问题 → model 层（可热切换）
    RollbackSignal.p95_latency_exceeded: VersionLayer.model,
    RollbackSignal.json_parse_rate_drop: VersionLayer.model,
    # 资源异常 → config 层（秒级生效，降低采样率等）
    RollbackSignal.mem_exceeded: VersionLayer.config,
    # 输入链路断开 → alert_only，不回滚（可能是外部原因）
    RollbackSignal.input_link_down: VersionLayer.config,
}


@dataclass
class RollbackTrigger:
    """回滚触发信号实例。"""

    signal: RollbackSignal
    gw_id: str
    model_id: str
    version: str
    severity: Literal["immediate", "delayed", "alert_only"]
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RollbackDecision:
    """回滚决策结果。

    - should_rollback：是否执行回滚
    - layer：按哪层回滚（firmware/model/config）
    - reason：决策原因（人类可读）
    - frozen_version：要冻结的问题版本（None 表示不冻结）
    - target_version：回滚目标版本（上一个稳定版本，空串表示无可用目标）
    - gw_id / model_id：关联设备与模型（execute_rollback 使用）
    """

    should_rollback: bool
    layer: VersionLayer
    reason: str
    frozen_version: str | None
    target_version: str
    gw_id: str = ""
    model_id: str = ""


@dataclass
class RollbackPolicy:
    """回滚策略阈值（§4.7 示例值）。

    - p95_threshold_ms / p95_sustained_min：推理 P95 超安全阈值持续 N 分钟
    - mem_threshold_mb / mem_sustained_min：内存超限持续 N 分钟
    - json_parse_rate_min / json_parse_sustained_count：JSON 可解析率低于阈值持续 N 次
    """

    p95_threshold_ms: float = 3000.0
    p95_sustained_min: int = 10
    mem_threshold_mb: int = 3072
    mem_sustained_min: int = 5
    json_parse_rate_min: float = 0.80
    json_parse_sustained_count: int = 100


class RollbackManager:
    """回滚决策管理器：评估触发信号、执行回滚、冻结发布。

    与端侧 mgmt_agent/ota.py 的 OtaEngine 协作：
    - 云端决策（本类）→ 端侧执行（OtaEngine/VersionSlots）
    - 通过 RolloutRecord 状态机协调：rolling_back → rolled_back
    - 通过 RolloutPlan.frozen 阻止问题版本扩散
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        policy: RollbackPolicy | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.policy = policy or RollbackPolicy()

    # ----------------------------------------------------------------------- #
    # 冻结发布管理
    # ----------------------------------------------------------------------- #

    def freeze_version(self, model_id: str, version: str) -> None:
        """冻结指定版本：设置 RolloutPlan.frozen=True，阻止灰度继续扩散。

        失败收敛逻辑（§4.7）：探针失败 → 自动回滚 → 自动冻结该发布 → 锁定问题版本。
        """
        with self._session_factory() as s:
            plan = s.query(RolloutPlan).filter_by(model_id=model_id, version=version).one_or_none()
            if plan is None:
                s.add(RolloutPlan(model_id=model_id, version=version, frozen=True))
            else:
                plan.frozen = True
            s.commit()

    def is_frozen(self, model_id: str, version: str) -> bool:
        """检查版本是否已冻结。"""
        with self._session_factory() as s:
            plan = s.query(RolloutPlan).filter_by(model_id=model_id, version=version).one_or_none()
            return bool(plan is not None and plan.frozen)

    def get_frozen_versions(self) -> list[dict[str, Any]]:
        """获取所有冻结版本。"""
        with self._session_factory() as s:
            rows = s.query(RolloutPlan).filter_by(frozen=True).all()
            return [{"model_id": r.model_id, "version": r.version} for r in rows]

    # ----------------------------------------------------------------------- #
    # 回滚层级选择
    # ----------------------------------------------------------------------- #

    def select_rollback_layer(self, trigger: RollbackTrigger) -> VersionLayer:
        """根据触发信号选择回滚层级（§4.4 三层解耦）。

        - process_start_failure / probe_failure → firmware（需重启/切分区）
        - p95_latency_exceeded / json_parse_rate_drop → model（可热切换）
        - mem_exceeded → config（秒级生效，降低采样率等）
        - input_link_down → config（默认最轻层，但 alert_only 不回滚）
        """
        return _LAYER_MAP[trigger.signal]

    # ----------------------------------------------------------------------- #
    # 回滚目标版本查找
    # ----------------------------------------------------------------------- #

    def find_rollback_target(self, gw_id: str, layer: VersionLayer) -> str:
        """找到回滚目标版本：该设备上一个 state=succeeded 的版本。

        优先返回最近一次成功的 RolloutRecord.version；无历史时返回空串。
        layer 参数保留用于未来按层筛选（当前 RolloutRecord 无 layer 字段）。
        """
        with self._session_factory() as s:
            row = (
                s.query(RolloutRecord)
                .filter_by(gw_id=gw_id, state="succeeded")
                .order_by(RolloutRecord.id.desc())
                .first()
            )
            return row.version if row is not None else ""

    # ----------------------------------------------------------------------- #
    # 触发信号评估
    # ----------------------------------------------------------------------- #

    def evaluate_trigger(self, trigger: RollbackTrigger) -> RollbackDecision:
        """评估触发信号，决定是否回滚。

        判定逻辑：
        - severity=alert_only → 不回滚（仅告警，§4.7 输入链路断开）
        - severity=immediate/delayed → 应回滚
        - 版本已冻结 → 不重复回滚（幂等）
        - 无回滚目标版本 → 不回滚（边界：无历史版本）
        """
        layer = self.select_rollback_layer(trigger)

        # alert_only 信号不自动回滚（如输入链路断开，可能是外部原因）
        if trigger.severity == "alert_only":
            return RollbackDecision(
                should_rollback=False,
                layer=layer,
                reason=f"alert_only signal {trigger.signal.value}, no auto rollback",
                frozen_version=None,
                target_version="",
                gw_id=trigger.gw_id,
                model_id=trigger.model_id,
            )

        # 已冻结版本不重复回滚（幂等保护）
        if self.is_frozen(trigger.model_id, trigger.version):
            return RollbackDecision(
                should_rollback=False,
                layer=layer,
                reason=f"version {trigger.version} already frozen",
                frozen_version=trigger.version,
                target_version="",
                gw_id=trigger.gw_id,
                model_id=trigger.model_id,
            )

        # 查找回滚目标版本
        target = self.find_rollback_target(trigger.gw_id, layer)
        if not target:
            return RollbackDecision(
                should_rollback=False,
                layer=layer,
                reason="no previous succeeded version to rollback to",
                frozen_version=None,
                target_version="",
                gw_id=trigger.gw_id,
                model_id=trigger.model_id,
            )

        return RollbackDecision(
            should_rollback=True,
            layer=layer,
            reason=f"signal {trigger.signal.value} severity={trigger.severity}",
            frozen_version=trigger.version,
            target_version=target,
            gw_id=trigger.gw_id,
            model_id=trigger.model_id,
        )

    # ----------------------------------------------------------------------- #
    # 执行回滚
    # ----------------------------------------------------------------------- #

    def execute_rollback(self, decision: RollbackDecision) -> RolloutRecord:
        """执行回滚：更新 RolloutRecord 状态为 rolling_back → rolled_back，
        并冻结问题版本。

        注意：实际端侧回滚由 mgmt_agent/ota.py 的 OtaEngine 执行，
        本方法仅更新云端编排状态与冻结标记。
        """
        if not decision.should_rollback:
            raise ValueError("cannot execute rollback for should_rollback=False")

        with self._session_factory() as s:
            # 1. 找到当前问题版本的活跃记录，更新状态
            active = (
                s.query(RolloutRecord)
                .filter_by(
                    gw_id=decision.gw_id,
                    model_id=decision.model_id,
                    version=decision.frozen_version or "",
                )
                .order_by(RolloutRecord.id.desc())
                .first()
            )
            if active is not None:
                active.state = "rolling_back"
                s.commit()
                active.state = "rolled_back"
                active.detail = {
                    **(active.detail or {}),
                    "rollback_layer": decision.layer.value,
                    "rollback_target": decision.target_version,
                    "rollback_reason": decision.reason,
                }
                record = active
            else:
                # 无活跃记录时创建一条回滚记录
                record = RolloutRecord(
                    gw_id=decision.gw_id,
                    model_id=decision.model_id,
                    version=decision.target_version,
                    state="rolled_back",
                    detail={
                        "rollback_layer": decision.layer.value,
                        "rollback_from": decision.frozen_version,
                        "rollback_reason": decision.reason,
                    },
                )
                s.add(record)
            s.commit()

            # 2. 冻结问题版本（阻止扩散）
            if decision.frozen_version:
                plan = (
                    s.query(RolloutPlan)
                    .filter_by(model_id=decision.model_id, version=decision.frozen_version)
                    .one_or_none()
                )
                if plan is None:
                    s.add(
                        RolloutPlan(
                            model_id=decision.model_id,
                            version=decision.frozen_version,
                            frozen=True,
                        )
                    )
                else:
                    plan.frozen = True
            s.commit()

            s.refresh(record)
            return record

    # ----------------------------------------------------------------------- #
    # 批量回滚
    # ----------------------------------------------------------------------- #

    def batch_rollback(
        self,
        gw_ids: list[str],
        decision: RollbackDecision,
    ) -> list[RolloutRecord]:
        """批量回滚多设备：对每个 gw_id 执行一次回滚。

        用于灰度发布中多设备同时异常的场景（如某站点集体故障）。
        """
        records: list[RolloutRecord] = []
        for gw_id in gw_ids:
            per_device = RollbackDecision(
                should_rollback=decision.should_rollback,
                layer=decision.layer,
                reason=decision.reason,
                frozen_version=decision.frozen_version,
                target_version=decision.target_version,
                gw_id=gw_id,
                model_id=decision.model_id,
            )
            records.append(self.execute_rollback(per_device))
        return records

    # ----------------------------------------------------------------------- #
    # 健康趋势检查
    # ----------------------------------------------------------------------- #

    def check_health_trend(self, gw_id: str, window_min: int) -> list[RollbackTrigger]:
        """检查健康趋势，发现持续异常。

        解析最近 window_min 分钟内的 EdgeHeartbeat，对照策略阈值检测：
        - probe_ok=false（任意一次）→ probe_failure（immediate）
        - inference_p95_ms > threshold 持续 p95_sustained_min → p95_latency_exceeded（delayed）
        - mem_used_mb > threshold 持续 mem_sustained_min → mem_exceeded（delayed）
        - input_links 任意 false → input_link_down（alert_only）

        返回检测到的触发信号列表（可能为空）。
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=window_min)
        # SQLite DateTime(timezone=True) 存读时可能丢失 tzinfo，
        # 用 naive datetime 做 SQL 过滤，再在 Python 中用 payload.ts 精确过滤。
        cutoff_naive = cutoff.replace(tzinfo=None)
        with self._session_factory() as s:
            rows = (
                s.query(EdgeHeartbeat)
                .filter(EdgeHeartbeat.gw_id == gw_id, EdgeHeartbeat.ts >= cutoff_naive)
                .order_by(EdgeHeartbeat.ts.asc())
                .all()
            )

        if not rows:
            return []

        # 解析心跳 payload 为 EdgeHealth，用 payload 中的 ts（保留 tzinfo）做精确过滤
        healths: list[EdgeHealth] = []
        for r in rows:
            try:
                h = EdgeHealth.model_validate(r.payload)
                if h.ts >= cutoff:
                    healths.append(h)
            except Exception:
                log.warning("invalid heartbeat payload for %s at %s", gw_id, r.ts)
                continue

        if not healths:
            return []

        triggers: list[RollbackTrigger] = []
        latest = healths[-1]
        model_id = next(iter(latest.model_versions), "unknown")
        version = latest.model_versions.get(model_id, "unknown")

        # 1. 探针失败（任意一次 probe_ok=false → immediate）
        if any(not h.probe_ok for h in healths):
            triggers.append(
                RollbackTrigger(
                    signal=RollbackSignal.probe_failure,
                    gw_id=gw_id,
                    model_id=model_id,
                    version=version,
                    severity="immediate",
                    detail={"reason": "probe_ok=false detected in window"},
                    timestamp=latest.ts,
                )
            )

        # 2. P95 持续超阈值（delayed）
        if self._is_sustained(
            healths,
            self.policy.p95_sustained_min,
            lambda h: h.inference_p95_ms > self.policy.p95_threshold_ms,
        ):
            triggers.append(
                RollbackTrigger(
                    signal=RollbackSignal.p95_latency_exceeded,
                    gw_id=gw_id,
                    model_id=model_id,
                    version=version,
                    severity="delayed",
                    detail={
                        "p95_ms": latest.inference_p95_ms,
                        "threshold_ms": self.policy.p95_threshold_ms,
                        "sustained_min": self.policy.p95_sustained_min,
                    },
                    timestamp=latest.ts,
                )
            )

        # 3. 内存持续超阈值（delayed）
        if self._is_sustained(
            healths,
            self.policy.mem_sustained_min,
            lambda h: h.mem_used_mb > self.policy.mem_threshold_mb,
        ):
            triggers.append(
                RollbackTrigger(
                    signal=RollbackSignal.mem_exceeded,
                    gw_id=gw_id,
                    model_id=model_id,
                    version=version,
                    severity="delayed",
                    detail={
                        "mem_used_mb": latest.mem_used_mb,
                        "threshold_mb": self.policy.mem_threshold_mb,
                        "sustained_min": self.policy.mem_sustained_min,
                    },
                    timestamp=latest.ts,
                )
            )

        # 4. 输入链路断开（alert_only）
        for h in healths:
            broken = [name for name, ok in h.input_links.items() if not ok]
            if broken:
                triggers.append(
                    RollbackTrigger(
                        signal=RollbackSignal.input_link_down,
                        gw_id=gw_id,
                        model_id=model_id,
                        version=version,
                        severity="alert_only",
                        detail={"broken_links": broken},
                        timestamp=h.ts,
                    )
                )
                break

        return triggers

    @staticmethod
    def _is_sustained(
        healths: list[EdgeHealth],
        sustained_min: int,
        predicate: Any,
    ) -> bool:
        """检查最近 sustained_min 分钟内是否所有心跳都满足 predicate。

        判定条件：
        - 窗口内有心跳记录
        - 所有记录都满足 predicate
        - 记录时间跨度 >= sustained_min 分钟（单条记录时放宽为 ts 距 now <= sustained_min）
        """
        if not healths:
            return False
        cutoff = datetime.now(UTC) - timedelta(minutes=sustained_min)
        recent = [h for h in healths if h.ts >= cutoff]
        if not recent or not all(predicate(h) for h in recent):
            return False
        # 时间跨度检查：多条记录需跨度足够，单条记录允许（视为当前正在持续）
        if len(recent) == 1:
            return True
        span_min = (recent[-1].ts - recent[0].ts).total_seconds() / 60.0
        return span_min >= sustained_min * 0.9  # 容忍 10% 抖动
