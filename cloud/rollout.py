"""灰度发布编排器：五步节奏 + 发布状态机 + 按站点/型号分环。

五步节奏（§4.5）：
  1. internal_test  内部实验设备（1 台，实验室）—— 验证版本依赖
  2. canary         Canary（≈1%）—— 验证升级/推理/回滚三条链路
  3. expand_ring    按站点/型号分环扩圈（10% → 50%）—— 验证硬件与现场差异
  4. stable_window  稳定窗口观察（默认 24h）—— 验证推理质量与资源稳定
  5. full_release   全量 + 保留冻结与回滚窗口（默认 7 天）

发布状态机：
  pending → downloading → verifying → staged
    → activating → probing
    → [probe_ok]  observing → succeeded
    → [probe_fail] rolling_back → rolled_back → frozen

关键约束：灰度验证的不是"数量比例"，而是"恢复能力"。
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from cloud.db import EdgeDevice, RolloutPlan, RolloutRecord

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #


class RolloutStep(StrEnum):
    """灰度发布五步节奏。"""

    internal_test = "internal_test"
    canary = "canary"
    expand_ring = "expand_ring"
    stable_window = "stable_window"
    full_release = "full_release"


class RolloutState(StrEnum):
    """发布状态机状态。"""

    pending = "pending"
    downloading = "downloading"
    verifying = "verifying"
    staged = "staged"
    activating = "activating"
    probing = "probing"
    observing = "observing"
    succeeded = "succeeded"
    rolling_back = "rolling_back"
    rolled_back = "rolled_back"
    frozen = "frozen"


# --------------------------------------------------------------------------- #
# 合法状态转换表
# --------------------------------------------------------------------------- #

_VALID_TRANSITIONS: dict[RolloutState, frozenset[RolloutState]] = {
    RolloutState.pending: frozenset({RolloutState.downloading, RolloutState.frozen}),
    RolloutState.downloading: frozenset(
        {
            RolloutState.verifying,
            RolloutState.rolling_back,
            RolloutState.frozen,
        }
    ),
    RolloutState.verifying: frozenset(
        {
            RolloutState.staged,
            RolloutState.rolling_back,
            RolloutState.frozen,
        }
    ),
    RolloutState.staged: frozenset(
        {
            RolloutState.activating,
            RolloutState.rolling_back,
            RolloutState.frozen,
        }
    ),
    RolloutState.activating: frozenset(
        {
            RolloutState.probing,
            RolloutState.rolling_back,
            RolloutState.frozen,
        }
    ),
    RolloutState.probing: frozenset(
        {
            RolloutState.observing,
            RolloutState.rolling_back,
            RolloutState.frozen,
        }
    ),
    RolloutState.observing: frozenset(
        {
            RolloutState.succeeded,
            RolloutState.rolling_back,
            RolloutState.frozen,
        }
    ),
    RolloutState.succeeded: frozenset(),  # 终态
    RolloutState.rolling_back: frozenset({RolloutState.rolled_back, RolloutState.frozen}),
    RolloutState.rolled_back: frozenset({RolloutState.frozen}),
    RolloutState.frozen: frozenset(),  # 终态
}

# 五步节奏顺序（索引即步骤序号）
_STEP_ORDER: tuple[RolloutStep, ...] = (
    RolloutStep.internal_test,
    RolloutStep.canary,
    RolloutStep.expand_ring,
    RolloutStep.stable_window,
    RolloutStep.full_release,
)


# --------------------------------------------------------------------------- #
# RingConfig
# --------------------------------------------------------------------------- #


@dataclass
class RingConfig:
    """灰度分环配置。

    rings: 百分比列表，如 [1, 10, 50, 100] 表示 1% → 10% → 50% → 全量。
    stable_window_h: 稳定窗口观察时长（小时），默认 24。
    rollback_window_h: 回滚保留窗口（小时），默认 168（7 天）。
    min_sample_size: 进入稳定窗口的最少样本量门槛，默认 100。
    """

    rings: list[int] = field(default_factory=lambda: [1, 10, 50, 100])
    stable_window_h: int = 24
    rollback_window_h: int = 168
    min_sample_size: int = 100


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #


def _hardware_model(hardware: dict[str, Any] | None) -> str:
    """从 EdgeDevice.hardware dict 提取型号标识用于分环。

    优先使用 model/type/board/hw_model 字段；缺失时回退到排序键值对，
    确保 不同硬件配置得到不同分组键。
    """
    if not hardware:
        return "unknown"
    for key in ("model", "type", "board", "hw_model"):
        val = hardware.get(key)
        if val:
            return str(val)
    return ",".join(f"{k}={hardware[k]}" for k in sorted(hardware))


def _interleave(groups: dict[str, list[str]]) -> list[str]:
    """轮询 interleaving：从各组交替取元素，保证多样性。

    例：{a: [1,4], b: [2,5], c: [3,6]} → [1,2,3,4,5,6]。
    """
    ordered: list[str] = []
    keys = list(groups.keys())
    cursor = {k: 0 for k in keys}
    total = sum(len(v) for v in groups.values())
    while len(ordered) < total:
        progressed = False
        for k in keys:
            if cursor[k] < len(groups[k]):
                ordered.append(groups[k][cursor[k]])
                cursor[k] += 1
                progressed = True
                if len(ordered) >= total:
                    break
        if not progressed:
            break
    return ordered


def _validate_transition(old: RolloutState, new: RolloutState) -> None:
    """验证状态转换合法性，非法时抛 ValueError。"""
    allowed = _VALID_TRANSITIONS.get(old, frozenset())
    if new not in allowed:
        raise ValueError(
            f"illegal state transition: {old.value} -> {new.value} "
            f"(allowed: {sorted(s.value for s in allowed) or 'none'})"
        )


# --------------------------------------------------------------------------- #
# RolloutOrchestrator
# --------------------------------------------------------------------------- #


class RolloutOrchestrator:
    """灰度发布编排器，协调五步节奏与发布状态机。

    session_factory: SQLAlchemy sessionmaker，每次方法调用创建独立 session。
    运行时编排状态（当前步索引、当前环百分比、RingConfig）由编排器在内存中
    持有，以 plan_id 为键；RolloutPlan.rings 持久化百分比列表作为兜底。
    """

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory
        # plan_id -> 当前步索引
        self._step_index: dict[int, int] = {}
        # plan_id -> 当前环百分比
        self._current_ring: dict[int, int] = {}
        # plan_id -> RingConfig
        self._configs: dict[int, RingConfig] = {}

    # ----- 内部辅助 ----- #

    def _get_plan(self, session: Any, plan_id: int) -> RolloutPlan:
        plan = session.get(RolloutPlan, plan_id)
        if plan is None:
            raise ValueError(f"plan_id={plan_id} not found")
        return plan

    def _get_config(self, plan_id: int) -> RingConfig:
        """取 plan 的 RingConfig，缓存缺失时从 plan.rings 重建默认配置。"""
        if plan_id in self._configs:
            return self._configs[plan_id]
        with self._session_factory() as session:
            plan = self._get_plan(session, plan_id)
        rings = list(plan.rings) if plan.rings else [1, 10, 50, 100]
        cfg = RingConfig(rings=[int(r) for r in rings])
        self._configs[plan_id] = cfg
        return cfg

    def _current_step(self, plan_id: int) -> RolloutStep:
        idx = self._step_index.get(plan_id, 0)
        return _STEP_ORDER[idx]

    # ----- 创建灰度计划 ----- #

    def create_plan(
        self,
        model_id: str,
        version: str,
        ring_config: RingConfig,
    ) -> RolloutPlan:
        """创建灰度发布计划，初始化五步节奏起点（internal_test）。"""
        with self._session_factory() as session:
            plan = RolloutPlan(
                model_id=model_id,
                version=version,
                strategy="ring",
                rings=list(ring_config.rings),
                frozen=False,
            )
            session.add(plan)
            session.commit()
            session.refresh(plan)
            pid = plan.id
        self._step_index[pid] = 0
        self._current_ring[pid] = 0
        self._configs[pid] = ring_config
        log.info(
            "created rollout plan id=%s model=%s version=%s rings=%s",
            pid,
            model_id,
            version,
            ring_config.rings,
        )
        return plan

    # ----- 分环选择设备 ----- #

    def select_devices(
        self,
        ring_percent: int,
        site: str | None = None,
        hardware: str | None = None,
    ) -> list[str]:
        """按站点/型号分环选择设备。

        先按 site/hardware 标签过滤，再按 (site, hardware_model) 分组做
        轮询 interleaving，确保每环包含不同站点/型号的代表性设备，
        最后截取 ring_percent 比例的设备。
        """
        with self._session_factory() as session:
            stmt = select(EdgeDevice)
            if site is not None:
                stmt = stmt.where(EdgeDevice.site == site)
            devices = list(session.execute(stmt).scalars().all())

        if hardware is not None:
            devices = [d for d in devices if _hardware_model(d.hardware) == hardware]

        if not devices:
            return []

        # 两层分环：站点间轮询，站点内按型号轮询。
        # 站点间对型号起始偏移，使前 N 台在站点与型号两个维度都多样。
        by_site: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for d in devices:
            site = d.site or "unknown"
            by_site[site][_hardware_model(d.hardware)].append(d.gw_id)

        site_ordered: dict[str, list[str]] = {}
        sites_list = list(by_site.keys())
        for i, site in enumerate(sites_list):
            model_groups = by_site[site]
            models_list = list(model_groups.keys())
            # 按站点索引轮转型号起始，避免每站点首台都是同一型号
            if models_list:
                offset = i % len(models_list)
                rotated = models_list[offset:] + models_list[:offset]
            else:
                rotated = []
            rotated_groups = {k: model_groups[k] for k in rotated}
            site_ordered[site] = _interleave(rotated_groups)

        ordered = _interleave(site_ordered)

        # 截取 ring_percent 比例（至少 1 台，100% 取全部）
        total = len(devices)
        target = total if ring_percent >= 100 else max(1, math.ceil(total * ring_percent / 100))
        return ordered[:target]

    # ----- 推进步骤 ----- #

    def advance_step(self, plan_id: int) -> RolloutStep:
        """推进到下一步并返回该步；已在最后一步时抛 ValueError。"""
        idx = self._step_index.get(plan_id, 0)
        if idx >= len(_STEP_ORDER) - 1:
            raise ValueError(
                f"plan_id={plan_id} already at last step ({_STEP_ORDER[idx].value}), cannot advance"
            )
        idx += 1
        self._step_index[plan_id] = idx
        step = _STEP_ORDER[idx]
        log.info("plan_id=%s advanced to step=%s", plan_id, step.value)
        return step

    # ----- 状态机转换 ----- #

    def transition_state(
        self,
        record_id: int,
        new_state: RolloutState,
        detail: dict[str, Any],
    ) -> RolloutRecord:
        """状态机转换：验证合法性后更新 record 的 state 与 detail。

        非法转换（如 pending → succeeded）抛 ValueError。
        record 不存在时抛 ValueError。
        """
        with self._session_factory() as session:
            record = session.get(RolloutRecord, record_id)
            if record is None:
                raise ValueError(f"record_id={record_id} not found")
            old = RolloutState(record.state)
            _validate_transition(old, new_state)
            record.state = new_state.value
            record.detail = detail
            session.commit()
            session.refresh(record)
            return record

    # ----- Canary 条件检查 ----- #

    def can_deploy_to_canary(self, plan_id: int) -> bool:
        """检查是否满足 Canary 部署条件。

        条件：
          1. plan 存在且未冻结；
          2. 当前步已推进到 canary（即 internal_test 已完成）；
          3. 存在至少 1 台 internal_test 设备到达 succeeded。
        """
        with self._session_factory() as session:
            plan = self._get_plan(session, plan_id)
            if plan.frozen:
                return False
            records = list(
                session.execute(
                    select(RolloutRecord).where(
                        RolloutRecord.model_id == plan.model_id,
                        RolloutRecord.version == plan.version,
                    )
                )
                .scalars()
                .all()
            )
        # 当前步必须 >= canary
        if self._step_index.get(plan_id, 0) < 1:
            return False
        # 至少 1 台 internal_test 设备 succeeded
        has_internal_success = any(
            r.state == RolloutState.succeeded.value
            and r.detail.get("step") == RolloutStep.internal_test.value
            for r in records
        )
        return has_internal_success

    # ----- 扩圈 ----- #

    def expand_ring(self, plan_id: int, target_percent: int) -> list[RolloutRecord]:
        """扩圈到目标百分比，为入选设备创建 pending 记录。

        已存在同设备同版本的记录则跳过，避免重复。
        """
        with self._session_factory() as session:
            plan = self._get_plan(session, plan_id)
            if plan.frozen:
                raise ValueError(f"plan_id={plan_id} is frozen, cannot expand")
            gw_ids = self.select_devices(target_percent)
            # 查已有记录，避免重复
            existing = {
                r.gw_id
                for r in session.execute(
                    select(RolloutRecord).where(
                        RolloutRecord.model_id == plan.model_id,
                        RolloutRecord.version == plan.version,
                    )
                )
                .scalars()
                .all()
            }
            created: list[RolloutRecord] = []
            for gw_id in gw_ids:
                if gw_id in existing:
                    continue
                rec = RolloutRecord(
                    gw_id=gw_id,
                    model_id=plan.model_id,
                    version=plan.version,
                    state=RolloutState.pending.value,
                    detail={
                        "step": RolloutStep.expand_ring.value,
                        "ring_percent": target_percent,
                    },
                )
                session.add(rec)
                created.append(rec)
            session.commit()
            for r in created:
                session.refresh(r)
        self._current_ring[plan_id] = target_percent
        log.info(
            "plan_id=%s expanded to %s%%, %d new records", plan_id, target_percent, len(created)
        )
        return created

    # ----- 稳定窗口检查 ----- #

    def check_stable_window(self, plan_id: int) -> bool:
        """检查稳定窗口是否满足。

        条件：
          1. 当前步已到达 stable_window；
          2. succeeded 记录数 >= min_sample_size；
          3. 无处于 rolling_back/rolled_back/frozen 的记录（恢复能力验证通过）。
        """
        cfg = self._get_config(plan_id)
        with self._session_factory() as session:
            plan = self._get_plan(session, plan_id)
            records = list(
                session.execute(
                    select(RolloutRecord).where(
                        RolloutRecord.model_id == plan.model_id,
                        RolloutRecord.version == plan.version,
                    )
                )
                .scalars()
                .all()
            )
        # 当前步必须 >= stable_window（索引 3）
        if self._step_index.get(plan_id, 0) < 3:
            return False
        succeeded = [r for r in records if r.state == RolloutState.succeeded.value]
        bad = {
            RolloutState.rolling_back.value,
            RolloutState.rolled_back.value,
            RolloutState.frozen.value,
        }
        has_failure = any(r.state in bad for r in records)
        if has_failure:
            return False
        return len(succeeded) >= cfg.min_sample_size

    # ----- 全量发布 ----- #

    def full_release(self, plan_id: int) -> list[RolloutRecord]:
        """全量发布：选择 100% 设备并创建 pending 记录，推进到 full_release 步。"""
        with self._session_factory() as session:
            plan = self._get_plan(session, plan_id)
            if plan.frozen:
                raise ValueError(f"plan_id={plan_id} is frozen, cannot full release")
            gw_ids = self.select_devices(100)
            existing = {
                r.gw_id
                for r in session.execute(
                    select(RolloutRecord).where(
                        RolloutRecord.model_id == plan.model_id,
                        RolloutRecord.version == plan.version,
                    )
                )
                .scalars()
                .all()
            }
            created: list[RolloutRecord] = []
            for gw_id in gw_ids:
                if gw_id in existing:
                    continue
                rec = RolloutRecord(
                    gw_id=gw_id,
                    model_id=plan.model_id,
                    version=plan.version,
                    state=RolloutState.pending.value,
                    detail={
                        "step": RolloutStep.full_release.value,
                        "ring_percent": 100,
                    },
                )
                session.add(rec)
                created.append(rec)
            session.commit()
            for r in created:
                session.refresh(r)
        self._current_ring[plan_id] = 100
        self._step_index[plan_id] = len(_STEP_ORDER) - 1
        log.info("plan_id=%s full release, %d new records", plan_id, len(created))
        return created

    # ----- 发布进度查询 ----- #

    def get_rollout_progress(self, plan_id: int) -> dict[str, Any]:
        """获取发布进度。

        返回 dict：
          plan_id, model_id, version, frozen,
          current_step, current_ring_percent,
          total_records, state_counts (各状态设备数),
          ring_config (rings/stable_window_h/rollback_window_h/min_sample_size)。
        """
        cfg = self._get_config(plan_id)
        with self._session_factory() as session:
            plan = self._get_plan(session, plan_id)
            records = list(
                session.execute(
                    select(RolloutRecord).where(
                        RolloutRecord.model_id == plan.model_id,
                        RolloutRecord.version == plan.version,
                    )
                )
                .scalars()
                .all()
            )
        state_counts: dict[str, int] = defaultdict(int)
        for r in records:
            state_counts[r.state] += 1
        return {
            "plan_id": plan.id,
            "model_id": plan.model_id,
            "version": plan.version,
            "frozen": plan.frozen,
            "current_step": self._current_step(plan_id).value,
            "current_ring_percent": self._current_ring.get(plan_id, 0),
            "total_records": len(records),
            "state_counts": dict(state_counts),
            "ring_config": {
                "rings": list(cfg.rings),
                "stable_window_h": cfg.stable_window_h,
                "rollback_window_h": cfg.rollback_window_h,
                "min_sample_size": cfg.min_sample_size,
            },
        }
