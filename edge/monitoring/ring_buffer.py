"""E3 监控可观测：环形缓冲区日志（24h 保留，线程安全）。

用 collections.deque 实现高效环形缓冲，max_entries 限制最大条目数
（超出自动丢弃最旧），retention_hours 限制保留时长（purge_expired 清理）。

支持断网恢复后批量补传：export_batch(since) 返回自 since 起未导出的日志，
mark_exported 标记已导出，避免重复上传。

线程安全：所有公共方法用 threading.Lock 保护。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any


@dataclass
class LogEntry:
    """日志条目。

    Attributes:
        timestamp: 日志时间（UTC）
        level: 日志级别（INFO/WARN/ERROR/DEBUG 等）
        message: 日志消息
        metadata: 附加元数据（如 device_id, trace_id）
        exported: 是否已导出（用于断网恢复后补传去重）
    """

    timestamp: datetime
    level: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    exported: bool = False


class RingBufferLog:
    """环形缓冲区日志，线程安全。

    Args:
        max_entries: 最大条目数，超出自动丢弃最旧（deque maxlen）
        retention_hours: 保留时长（小时），purge_expired 据此清理
    """

    def __init__(self, max_entries: int = 100000, retention_hours: int = 24) -> None:
        if max_entries <= 0:
            msg = f"max_entries must be positive, got {max_entries}"
            raise ValueError(msg)
        if retention_hours <= 0:
            msg = f"retention_hours must be positive, got {retention_hours}"
            raise ValueError(msg)
        self._max_entries = max_entries
        self._retention = timedelta(hours=retention_hours)
        self._buffer: deque[LogEntry] = deque(maxlen=max_entries)
        self._lock = Lock()

    # --- 写入 ---

    def append(self, level: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """追加一条日志。时间戳取当前 UTC。"""
        entry = LogEntry(
            timestamp=datetime.now(UTC),
            level=level,
            message=message,
            metadata=dict(metadata) if metadata else {},
        )
        with self._lock:
            self._buffer.append(entry)

    def purge_expired(self) -> int:
        """清理过期日志（timestamp 早于 now - retention）。

        Returns:
            清理的条目数量
        """
        cutoff = datetime.now(UTC) - self._retention
        removed = 0
        with self._lock:
            # deque 从左侧（最旧）开始清理，遇到第一条未过期即停
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()
                removed += 1
        return removed

    # --- 查询 ---

    def query(
        self,
        since: datetime | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """查询日志。

        Args:
            since: 仅返回 timestamp >= since 的条目（None 则不限）
            level: 仅返回该级别的条目（None 则不限，精确匹配）
            limit: 最多返回条目数（取符合条件中最近的 limit 条，按时间正序）

        Returns:
            符合条件的日志条目列表（按时间正序）
        """
        if limit <= 0:
            return []
        with self._lock:
            matched = [
                e
                for e in self._buffer
                if (since is None or e.timestamp >= since) and (level is None or e.level == level)
            ]
        # 取最近 limit 条，保持正序
        return matched[-limit:] if len(matched) > limit else matched

    def export_batch(self, since: datetime) -> list[LogEntry]:
        """导出批量日志（用于断网恢复后补传）。

        返回 timestamp >= since 且未导出的条目，按时间正序。
        不会修改 exported 标记，需配合 mark_exported 使用。
        """
        with self._lock:
            return [e for e in self._buffer if e.timestamp >= since and not e.exported]

    def mark_exported(self, entries: list[LogEntry]) -> None:
        """标记条目已导出。对不在缓冲区中的条目静默忽略。"""
        with self._lock:
            for entry in entries:
                entry.exported = True

    # --- 统计 ---

    def stats(self) -> dict[str, Any]:
        """返回统计信息。

        Returns:
            dict 包含：
            - total: 总条目数
            - by_level: 各级别计数 dict
            - earliest: 最早 timestamp（None 如果空）
            - latest: 最晚 timestamp（None 如果空）
            - exported: 已导出条目数
            - unexported: 未导出条目数
            - max_entries: 最大容量
            - retention_hours: 保留时长（小时）
        """
        with self._lock:
            total = len(self._buffer)
            by_level: dict[str, int] = {}
            exported_count = 0
            earliest: datetime | None = None
            latest: datetime | None = None
            for e in self._buffer:
                by_level[e.level] = by_level.get(e.level, 0) + 1
                if e.exported:
                    exported_count += 1
                if earliest is None or e.timestamp < earliest:
                    earliest = e.timestamp
                if latest is None or e.timestamp > latest:
                    latest = e.timestamp
        return {
            "total": total,
            "by_level": by_level,
            "earliest": earliest,
            "latest": latest,
            "exported": exported_count,
            "unexported": total - exported_count,
            "max_entries": self._max_entries,
            "retention_hours": int(self._retention.total_seconds() // 3600),
        }

    # --- 辅助 ---

    def __len__(self) -> int:
        """返回当前缓冲区条目数。"""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """清空缓冲区（主要用于测试）。"""
        with self._lock:
            self._buffer.clear()
