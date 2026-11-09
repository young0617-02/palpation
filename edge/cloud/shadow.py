"""设备影子 desired→delta→主动拉取→reported 收敛机制。

核心原则：**边缘永不自己决定版本**。边缘只观察到 desired ≠ reported 时主动拉取，
并将实际状态写入 reported。desired 只能由云端下发（apply_desired / set_desired），
边缘绝不自行修改 desired。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _delta(desired: dict[str, Any], reported: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in desired.items():
        if reported.get(k) != v:
            out[k] = v
    return out


@dataclass
class ShadowBox:
    """设备影子盒：维护 desired / reported / delta 及同步元信息。

    收敛流程：
        1. 云端下发 desired → apply_desired() 计算 delta
        2. needs_pull() 返回非空 delta → 边缘主动拉取对应资源
        3. 拉取并应用后 → update_reported() 逐字段收敛
        4. delta 归零 → reported 与 desired 对齐，同步完成

    边缘只读 desired、只写 reported，永不自行决定 desired 内容。
    """

    desired: dict[str, Any] = field(default_factory=dict)
    reported: dict[str, Any] = field(default_factory=dict)
    # 同步元信息：最后同步时间 / 离线起始时间 / 累计补传条数
    synced_at: datetime | None = None
    offline_since: datetime | None = None
    backlog_count: int = 0

    # --- 兼容旧接口 ---

    def set_desired(self, desired: dict[str, Any]) -> dict[str, Any]:
        """设置云端下发的 desired，返回 delta（待收敛字段）。

        保留以兼容 runtime.py 现有调用。
        """
        self.desired = desired
        return _delta(self.desired, self.reported)

    def set_reported(self, reported: dict[str, Any]) -> dict[str, Any]:
        """批量设置 reported（反映边缘实际状态），返回残余 delta。

        保留以兼容 runtime.py 现有调用。
        """
        self.reported = reported
        self.synced_at = datetime.now(UTC)
        return _delta(self.desired, self.reported)

    def snapshot(self) -> dict[str, Any]:
        return {
            "desired": self.desired,
            "reported": self.reported,
            "delta": _delta(self.desired, self.reported),
            "meta": self.meta(),
        }

    # --- 新增增强接口 ---

    def apply_desired(self, desired: dict[str, Any]) -> dict[str, Any]:
        """应用云端下发的 desired，返回 delta。

        语义化别名，明确表达"云端 → 边缘"方向。边缘不自行构造 desired。
        """
        self.desired = desired
        return _delta(self.desired, self.reported)

    def needs_pull(self) -> dict[str, Any]:
        """返回需要主动拉取的字段（即当前 delta）。

        非空时表示 desired ≠ reported，边缘应主动拉取对应资源以收敛。
        """
        return _delta(self.desired, self.reported)

    def update_reported(self, key: str, value: Any) -> dict[str, Any]:
        """更新单个 reported 字段，刷新同步时间，返回残余 delta。

        边缘在拉取并实际应用某资源后调用，逐字段向 desired 收敛。
        """
        self.reported[key] = value
        self.synced_at = datetime.now(UTC)
        return _delta(self.desired, self.reported)

    def meta(self) -> dict[str, Any]:
        """返回同步元信息字典。"""
        offline_duration_s: float | None = None
        if self.offline_since is not None:
            offline_duration_s = (datetime.now(UTC) - self.offline_since).total_seconds()
        return {
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
            "offline_since": self.offline_since.isoformat() if self.offline_since else None,
            "offline_duration_s": offline_duration_s,
            "backlog_count": self.backlog_count,
        }

    def sync_status(self) -> dict[str, Any]:
        """返回同步状态概览。

        Returns:
            in_sync: delta 是否为空（desired == reported）
            pending_fields: 待拉取字段名列表
            offline_duration_s: 离线时长（秒），未离线时为 None
            backlog_count: 累计补传条数
            synced_at: 最后同步时间 ISO 字符串
        """
        delta = _delta(self.desired, self.reported)
        offline_duration_s: float | None = None
        if self.offline_since is not None:
            offline_duration_s = (datetime.now(UTC) - self.offline_since).total_seconds()
        return {
            "in_sync": len(delta) == 0,
            "pending_fields": list(delta.keys()),
            "offline_duration_s": offline_duration_s,
            "backlog_count": self.backlog_count,
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
        }

    # --- 离线 / 补传跟踪 ---

    def mark_offline(self, at: datetime | None = None) -> None:
        """标记进入离线状态（记录离线起始时间）。"""
        if self.offline_since is None:
            self.offline_since = at or datetime.now(UTC)

    def mark_online(self) -> None:
        """标记恢复在线状态（清除离线起始时间）。"""
        self.offline_since = None

    def incr_backlog(self, n: int = 1) -> None:
        """累计补传条数（离线期间积压的事件数）。"""
        self.backlog_count += n
