from __future__ import annotations

from dataclasses import dataclass

from edge.queue.sqlite_queue import QueueStats, SqliteQueue


def next_batch(queue: SqliteQueue, limit: int = 100, *, skip_acked: bool = False) -> list:
    """获取下一批待补传事件。

    Args:
        queue: 本地事件队列。
        limit: 批次上限。
        skip_acked: 为 True 时基于 ``acked`` 字段过滤（而非仅水位线），
            可跳过已单独 ACK 但水位线尚未推进的事件。默认 False 保持原行为。

    Returns:
        待补传的 EdgeEvent 列表，按 seq 升序排列。
    """
    if skip_acked:
        return queue.unacked_batch(limit=limit)
    return queue.pending_batch(queue.last_acked_seq(), limit=limit)


def apply_ack(queue: SqliteQueue, seq: int) -> None:
    """应用云端 ACK，推进水位线（单调递增，不回退）。"""
    if seq > queue.last_acked_seq():
        queue.set_acked(seq)


def detect_gaps(queue: SqliteQueue) -> list[tuple[int, int]]:
    """检测事件 seq 缺口。

    Returns:
        缺口闭区间列表，例如 ``[(3, 5)]`` 表示 seq 3/4/5 缺失。
    """
    return queue.gap_detection()


@dataclass(frozen=True, slots=True)
class ReplayStatus:
    """补传状态摘要。"""

    last_acked_seq: int
    pending_count: int
    gap_count: int
    gaps: list[tuple[int, int]]
    queue_stats: QueueStats


def replay_status(queue: SqliteQueue) -> ReplayStatus:
    """返回补传状态摘要：水位线、待补传数量、缺口数与缺口区间。"""
    gaps = detect_gaps(queue)
    stats = queue.queue_stats()
    return ReplayStatus(
        last_acked_seq=stats.last_acked_seq,
        pending_count=stats.depth,
        gap_count=sum(end - start + 1 for start, end in gaps),
        gaps=gaps,
        queue_stats=stats,
    )
