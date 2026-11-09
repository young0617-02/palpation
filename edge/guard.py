from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from edge.guard_degrade import DegradeLevel


@dataclass
class GuardState:
    # normal | llm_small | llm_off | temp_throttle | offline_degraded | disk_critical | emergency
    level: str
    mem_used_mb: int
    disk_free_mb: int
    cpu_temp_c: float | None
    llm_allowed: bool
    t2_full: bool
    accept_models: bool
    # 增强字段（带默认值，向后兼容现有 snapshot() 调用）
    cpu_percent: float = 0.0
    net_connected: bool = True
    offline_duration_s: int = 0
    degrade_actions: list[str] = field(default_factory=list)


class ResourceGuard:
    def __init__(
        self,
        data_dir: Path,
        warn_mb: int,
        llm_off_mb: int,
        disk_min_mb: int,
        temp_c: float,
        cloud_host: str = "127.0.0.1",
        cloud_port: int = 1883,
        net_timeout_s: float = 2.0,
    ) -> None:
        self.data_dir = data_dir
        self.warn_mb = warn_mb
        self.llm_off_mb = llm_off_mb
        self.disk_min_mb = disk_min_mb
        self.temp_c = temp_c
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.net_timeout_s = net_timeout_s
        self._last_online_at: float | None = None

    def snapshot(self) -> GuardState:
        proc = psutil.Process()
        mem = int(proc.memory_info().rss / (1024 * 1024))
        disk = int(
            psutil.disk_usage(str(self.data_dir if self.data_dir.exists() else Path("."))).free
            / (1024 * 1024)
        )
        temp = None
        try:
            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            if temps:
                first = next(iter(temps.values()))
                if first:
                    temp = float(first[0].current)
        except Exception:
            temp = None
        llm_allowed = mem < self.llm_off_mb
        t2_full = mem < self.llm_off_mb + 200
        level = "normal"
        if mem >= self.llm_off_mb:
            level = "llm_off"
        elif mem >= self.warn_mb:
            level = "llm_small"
        return GuardState(
            level=level,
            mem_used_mb=mem,
            disk_free_mb=disk,
            cpu_temp_c=temp,
            llm_allowed=llm_allowed,
            t2_full=t2_full,
            accept_models=disk >= self.disk_min_mb,
        )

    def check_network(self) -> bool:
        """检查网络连接（尝试 socket 连接云端 MQTT broker）。

        连接成功时更新 _last_online_at 时间戳并返回 True；
        连接失败返回 False，不更新时间戳。
        """
        try:
            with socket.create_connection(
                (self.cloud_host, self.cloud_port), timeout=self.net_timeout_s
            ):
                pass
        except OSError:
            return False
        self._last_online_at = time.time()
        return True

    def check_offline_duration(self) -> int:
        """计算离线时长（秒）。

        若从未成功连接过网络（_last_online_at 为 None），返回 0
        表示未知（视为刚启动，不触发离线降级）。
        """
        if self._last_online_at is None:
            return 0
        return int(time.time() - self._last_online_at)

    def full_snapshot(self) -> GuardState:
        """完整快照（含网络/离线/CPU 使用率/降级建议）。

        在 snapshot() 基础上补充增强字段，并根据完整降级矩阵
        重新计算 level 和 degrade_actions。
        """
        base = self.snapshot()
        net_connected = self.check_network()
        offline_duration_s = self.check_offline_duration()
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu_percent = 0.0

        state = GuardState(
            level=base.level,
            mem_used_mb=base.mem_used_mb,
            disk_free_mb=base.disk_free_mb,
            cpu_temp_c=base.cpu_temp_c,
            llm_allowed=base.llm_allowed,
            t2_full=base.t2_full,
            accept_models=base.accept_models,
            cpu_percent=cpu_percent,
            net_connected=net_connected,
            offline_duration_s=offline_duration_s,
            degrade_actions=[],
        )

        # 根据完整降级矩阵计算级别和动作
        decision = self._degrade_decision(state)
        state.degrade_actions = [a.name for a in decision.actions]
        state.level = decision.level.name.lower()
        return state

    def degrade_level(self, state: GuardState) -> DegradeLevel:
        """根据 GuardState 计算降级级别（委托 DegradePolicy）。"""
        from edge.guard_degrade import DegradeLevel as _Level

        return _Level(self._degrade_decision(state).level.value)

    def _degrade_decision(self, state: GuardState):
        """构建 DegradePolicy 并评估，返回 DegradeDecision。"""
        from edge.guard_degrade import DegradePolicy, DegradeThresholds

        thresholds = DegradeThresholds(
            mem_normal_mb=self.warn_mb,
            mem_llm_small_mb=self.warn_mb,
            mem_llm_off_mb=self.llm_off_mb,
            temp_throttle_c=self.temp_c,
            disk_critical_mb=self.disk_min_mb,
            offline_degraded_hours=24,
        )
        return DegradePolicy(thresholds).evaluate(state)
