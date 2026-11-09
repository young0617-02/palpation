"""断网自治 + 水位线补传 + 幂等去重 + 审计留痕 增强测试。

覆盖：
- 事件入队 / 出队
- 幂等去重（相同 event_id 不重复入队）
- 水位线 ACK 与补传
- 缺口检测
- 审计记录写入与查询
- 断网 → 缓存 → 补传 → 去重 完整链路
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from edge.queue.sqlite_queue import AuditRecord, QueueStats, SqliteQueue
from edge.queue.watermark import (
    ReplayStatus,
    apply_ack,
    detect_gaps,
    next_batch,
    replay_status,
)
from shared.errors import EdgeError
from shared.schemas import EdgeEvent


def _event(eid: str, seq: int = 0) -> EdgeEvent:
    return EdgeEvent(
        event_id=eid,
        gw_id="gw-sim-001",
        device_id_hash="abc",
        seq=seq,
        detected_at=datetime.now(UTC),
        rule_id="bearing_rms_high",
        severity="high",
        metric="vibration_rms",
        value=6.1,
        threshold=4.5,
    )


# ----------------------------------------------------------------------
# 事件入队 / 出队
# ----------------------------------------------------------------------


class TestEnqueueDequeue:
    def test_enqueue_assigns_sequential_seq(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        e1 = q.append_event(_event("evt-001"))
        e2 = q.append_event(_event("evt-002"))
        assert e1.seq == 1
        assert e2.seq == 2

    def test_dequeue_returns_in_seq_order(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(3):
            q.append_event(_event(f"evt-{i:03d}"))
        batch = next_batch(q, 100)
        assert [e.seq for e in batch] == [1, 2, 3]

    def test_dequeue_respects_limit(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        batch = next_batch(q, limit=2)
        assert [e.seq for e in batch] == [1, 2]

    def test_queue_depth_decreases_after_ack(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        assert q.queue_depth() == 5
        apply_ack(q, 2)
        assert q.queue_depth() == 3


# ----------------------------------------------------------------------
# 幂等去重
# ----------------------------------------------------------------------


class TestIdempotentDedup:
    def test_dedup_check_true_for_existing(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.append_event(_event("dup-001"))
        assert q.dedup_check("dup-001") is True

    def test_dedup_check_false_for_new(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        assert q.dedup_check("new-001") is False

    def test_duplicate_append_raises_conflict(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.append_event(_event("dup-002"))
        with pytest.raises(EdgeError) as exc_info:
            q.append_event(_event("dup-002"))
        assert exc_info.value.code == "E_CONFLICT"

    def test_duplicate_does_not_create_extra_row(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.append_event(_event("dup-003"))
        with pytest.raises(EdgeError):
            q.append_event(_event("dup-003"))
        assert q.event_count() == 1

    def test_distinct_event_ids_all_enqueued(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(10):
            q.append_event(_event(f"unique-{i:03d}"))
        assert q.event_count() == 10


# ----------------------------------------------------------------------
# 水位线 ACK 与补传
# ----------------------------------------------------------------------


class TestWatermarkAck:
    def test_ack_advances_watermark(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(4):
            q.append_event(_event(f"evt-{i:03d}"))
        apply_ack(q, 2)
        assert q.last_acked_seq() == 2
        batch = next_batch(q, 100)
        assert [e.seq for e in batch] == [3, 4]

    def test_ack_is_idempotent(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.append_event(_event("evt-001"))
        apply_ack(q, 1)
        apply_ack(q, 1)
        assert q.last_acked_seq() == 1

    def test_ack_does_not_regress(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.append_event(_event("evt-001"))
        q.append_event(_event("evt-002"))
        apply_ack(q, 2)
        apply_ack(q, 1)
        assert q.last_acked_seq() == 2

    def test_next_batch_skip_acked_filters_by_acked_flag(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(4):
            q.append_event(_event(f"evt-{i:03d}"))
        q.set_acked(2)
        batch = next_batch(q, 100, skip_acked=True)
        assert [e.seq for e in batch] == [3, 4]

    def test_next_batch_default_uses_watermark(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(4):
            q.append_event(_event(f"evt-{i:03d}"))
        apply_ack(q, 2)
        batch = next_batch(q, 100)
        assert [e.seq for e in batch] == [3, 4]


# ----------------------------------------------------------------------
# 缺口检测
# ----------------------------------------------------------------------


class TestGapDetection:
    def test_no_gaps_when_contiguous(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        assert q.gap_detection() == []
        assert detect_gaps(q) == []

    def test_empty_queue_no_gaps(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        assert q.gap_detection() == []

    def test_single_gap_in_middle(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        q._conn.execute("DELETE FROM events WHERE seq=3")
        q._conn.commit()
        assert q.gap_detection() == [(3, 3)]

    def test_multiple_gaps(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(8):
            q.append_event(_event(f"evt-{i:03d}"))
        q._conn.execute("DELETE FROM events WHERE seq IN (3, 4, 7)")
        q._conn.commit()
        assert q.gap_detection() == [(3, 4), (7, 7)]

    def test_detect_gaps_delegates_to_queue(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(4):
            q.append_event(_event(f"evt-{i:03d}"))
        q._conn.execute("DELETE FROM events WHERE seq=2")
        q._conn.commit()
        assert detect_gaps(q) == q.gap_detection()


# ----------------------------------------------------------------------
# 审计记录写入与查询
# ----------------------------------------------------------------------


class TestAudit:
    def test_audit_write_and_query_all(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        t0 = datetime.now(UTC) - timedelta(seconds=1)
        q.audit("event", {"event_id": "e1", "rule_id": "r1"})
        q.audit("preliminary", {"event_id": "e1", "text": "bearing wear"})
        t1 = datetime.now(UTC) + timedelta(seconds=1)
        records = q.query_audit(t0, t1)
        assert len(records) == 2
        assert isinstance(records[0], AuditRecord)
        assert records[0].kind == "event"
        assert records[1].kind == "preliminary"
        assert records[0].body == {"event_id": "e1", "rule_id": "r1"}

    def test_audit_with_explicit_timestamp(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        q.audit("event", {"x": 1}, timestamp=ts)
        records = q.query_audit(ts - timedelta(seconds=1), ts + timedelta(seconds=1))
        assert len(records) == 1
        assert records[0].ts == ts

    def test_audit_query_filters_by_time_range(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        ts1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2025, 1, 1, 13, 0, 0, tzinfo=UTC)
        q.audit("a", {"n": 1}, timestamp=ts1)
        q.audit("b", {"n": 2}, timestamp=ts2)
        records = q.query_audit(ts1 - timedelta(seconds=1), ts1 + timedelta(seconds=1))
        assert len(records) == 1
        assert records[0].kind == "a"

    def test_audit_is_append_only(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.audit("event", {"x": 1})
        with pytest.raises(Exception):
            q._conn.execute("UPDATE audit SET kind='x'")
            q._conn.commit()

    def test_audit_delete_blocked(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        q.audit("event", {"x": 1})
        with pytest.raises(Exception):
            q._conn.execute("DELETE FROM audit")
            q._conn.commit()


# ----------------------------------------------------------------------
# 队列统计
# ----------------------------------------------------------------------


class TestQueueStats:
    def test_stats_empty_queue(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        stats = q.queue_stats()
        assert isinstance(stats, QueueStats)
        assert stats.depth == 0
        assert stats.total_events == 0
        assert stats.last_acked_seq == 0
        assert stats.oldest_pending_at is None
        assert stats.newest_seq is None

    def test_stats_with_pending_events(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        stats = q.queue_stats()
        assert stats.depth == 5
        assert stats.total_events == 5
        assert stats.last_acked_seq == 0
        assert stats.newest_seq == 5
        assert stats.oldest_pending_at is not None

    def test_stats_after_partial_ack(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        apply_ack(q, 2)
        stats = q.queue_stats()
        assert stats.depth == 3
        assert stats.total_events == 5
        assert stats.last_acked_seq == 2
        assert stats.newest_seq == 5


# ----------------------------------------------------------------------
# 补传状态摘要
# ----------------------------------------------------------------------


class TestReplayStatus:
    def test_status_no_gaps(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        apply_ack(q, 2)
        status = replay_status(q)
        assert isinstance(status, ReplayStatus)
        assert status.last_acked_seq == 2
        assert status.pending_count == 3
        assert status.gap_count == 0
        assert status.gaps == []

    def test_status_with_gaps(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(6):
            q.append_event(_event(f"evt-{i:03d}"))
        q._conn.execute("DELETE FROM events WHERE seq IN (4, 5)")
        q._conn.commit()
        apply_ack(q, 2)
        status = replay_status(q)
        assert status.gap_count == 2
        assert status.gaps == [(4, 5)]

    def test_status_empty_queue(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        status = replay_status(q)
        assert status.pending_count == 0
        assert status.gap_count == 0


# ----------------------------------------------------------------------
# 断网 → 缓存 → 补传 → 去重 完整链路
# ----------------------------------------------------------------------


class TestOfflineReplayDedupChain:
    def test_offline_cache_replay_dedup(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        # 阶段 1：断网，事件入队（本地缓存）
        events_in = [q.append_event(_event(f"offline-evt-{i:03d}")) for i in range(3)]
        assert q.event_count() == 3

        # 阶段 2：补传（next_batch 取出）
        batch = next_batch(q, 100)
        assert len(batch) == 3

        # 阶段 3：云端 ACK 到 seq=3
        apply_ack(q, 3)
        assert q.queue_depth() == 0

        # 阶段 4：重放后再次入队相同 event_id（幂等去重）
        for ev in events_in:
            with pytest.raises(EdgeError):
                q.append_event(ev)
        assert q.event_count() == 3

    def test_partial_ack_then_replay_remaining(self, tmp_path: Path):
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        apply_ack(q, 2)
        batch = next_batch(q, 100)
        assert [e.seq for e in batch] == [3, 4, 5]
        apply_ack(q, 5)
        assert q.queue_depth() == 0
        assert next_batch(q, 100) == []

    def test_replay_completeness(self, tmp_path: Path):
        """补传完整率 100%：所有未 ACK 事件都能被分批取出并 ACK。"""
        q = SqliteQueue(tmp_path / "q.db")
        n = 20
        for i in range(n):
            q.append_event(_event(f"evt-{i:04d}"))
        replayed: list[int] = []
        while True:
            batch = next_batch(q, limit=4)
            if not batch:
                break
            replayed.extend(e.seq for e in batch)
            apply_ack(q, max(e.seq for e in batch))
        assert len(replayed) == n
        assert set(replayed) == set(range(1, n + 1))
        assert q.queue_depth() == 0

    def test_replay_after_gap_still_complete(self, tmp_path: Path):
        """即使存在缺口，剩余事件仍能被完整补传。"""
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(6):
            q.append_event(_event(f"evt-{i:03d}"))
        q._conn.execute("DELETE FROM events WHERE seq=3")
        q._conn.commit()
        replayed: list[int] = []
        while True:
            batch = next_batch(q, limit=10)
            if not batch:
                break
            replayed.extend(e.seq for e in batch)
            apply_ack(q, max(e.seq for e in batch))
        assert set(replayed) == {1, 2, 4, 5, 6}
        assert q.queue_depth() == 0

    def test_zero_duplicate_replay(self, tmp_path: Path):
        """幂等性 0 重复：补传不会产生重复事件。"""
        q = SqliteQueue(tmp_path / "q.db")
        for i in range(5):
            q.append_event(_event(f"evt-{i:03d}"))
        seen_ids: set[str] = set()
        while True:
            batch = next_batch(q, limit=2)
            if not batch:
                break
            for ev in batch:
                assert ev.event_id not in seen_ids, f"duplicate {ev.event_id}"
                seen_ids.add(ev.event_id)
            apply_ack(q, max(e.seq for e in batch))
        assert len(seen_ids) == 5