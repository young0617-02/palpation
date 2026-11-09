from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.errors import E_CONFLICT, EdgeError
from shared.schemas import EdgeEvent

DDL = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    payload TEXT NOT NULL,
    acked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watermark (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_acked_seq INTEGER NOT NULL
);
INSERT OR IGNORE INTO watermark(id, last_acked_seq) VALUES (1, 0);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit is append-only');
END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit is append-only');
END;
CREATE INDEX IF NOT EXISTS idx_events_acked ON events(acked);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
"""


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """审计记录的只读视图。"""

    id: int
    ts: datetime
    kind: str
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QueueStats:
    """队列统计快照。"""

    depth: int
    total_events: int
    last_acked_seq: int
    oldest_pending_at: datetime | None
    newest_seq: int | None


class SqliteQueue:
    """本地 SQLite 事件队列，支持幂等去重、水位线 ACK、缺口检测与追加式审计。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # 事件入队 / 幂等去重
    # ------------------------------------------------------------------

    def dedup_check(self, event_id: str) -> bool:
        """检查 event_id 是否已存在（幂等）。

        Returns:
            True 表示已存在（重复），False 表示新事件。
        """
        return self.has_event_id(event_id)

    def append_event(self, event: EdgeEvent) -> EdgeEvent:
        """追加事件到队列。

        追加前检查 event_id 幂等：若已存在则抛出 EdgeError(E_CONFLICT)，
        保证同一事件不会被重复入队。
        """
        if self.dedup_check(event.event_id):
            raise EdgeError(E_CONFLICT, f"duplicate event_id={event.event_id}")
        payload = event.model_dump_json()
        cur = self._conn.execute(
            "INSERT INTO events(event_id, payload, created_at) VALUES (?, ?, ?)",
            (event.event_id, payload, datetime.now(UTC).isoformat()),
        )
        event.seq = int(cur.lastrowid or 0)
        self._conn.execute(
            "UPDATE events SET payload=? WHERE seq=?",
            (event.model_dump_json(), event.seq),
        )
        self._conn.commit()
        return event

    # ------------------------------------------------------------------
    # 批量取出 / 水位线
    # ------------------------------------------------------------------

    def pending_batch(self, last_acked: int, limit: int = 100) -> list[EdgeEvent]:
        """返回 seq > last_acked 的待发送批次（基于水位线）。"""
        rows = self._conn.execute(
            "SELECT payload FROM events WHERE seq > ? ORDER BY seq ASC LIMIT ?",
            (last_acked, limit),
        ).fetchall()
        return [EdgeEvent.model_validate_json(r[0]) for r in rows]

    def unacked_batch(self, limit: int = 100) -> list[EdgeEvent]:
        """返回 acked=0 的待发送批次（基于单条 ACK 标志，非水位线）。

        当某些事件被单独 ACK 但水位线尚未推进时，此方法比 pending_batch
        更精确地跳过已 ACK 的事件。
        """
        rows = self._conn.execute(
            "SELECT payload FROM events WHERE acked=0 ORDER BY seq ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [EdgeEvent.model_validate_json(r[0]) for r in rows]

    def last_acked_seq(self) -> int:
        row = self._conn.execute("SELECT last_acked_seq FROM watermark WHERE id=1").fetchone()
        return int(row[0])

    def set_acked(self, seq: int) -> None:
        """推进 ACK 水位线，并将 seq <= 给定值的事件标记为已 ACK。"""
        self._conn.execute("UPDATE watermark SET last_acked_seq=? WHERE id=1", (seq,))
        self._conn.execute("UPDATE events SET acked=1 WHERE seq<=?", (seq,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # 缺口检测
    # ------------------------------------------------------------------

    def gap_detection(self) -> list[tuple[int, int]]:
        """检测事件 seq 中的缺口。

        扫描 [1, max(seq)] 范围内缺失的 seq，返回闭区间列表，
        例如 [(3, 5), (8, 8)] 表示 seq 3/4/5/8 缺失。

        Returns:
            缺口区间列表，无缺口时为空列表。
        """
        rows = self._conn.execute("SELECT seq FROM events ORDER BY seq ASC").fetchall()
        if not rows:
            return []
        seqs = [int(r[0]) for r in rows]
        gaps: list[tuple[int, int]] = []
        if seqs[0] > 1:
            gaps.append((1, seqs[0] - 1))
        for i in range(1, len(seqs)):
            prev = seqs[i - 1]
            curr = seqs[i]
            if curr - prev > 1:
                gaps.append((prev + 1, curr - 1))
        return gaps

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def queue_depth(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE seq > (SELECT last_acked_seq FROM watermark WHERE id=1)"
        ).fetchone()
        return int(row[0])

    def has_event_id(self, event_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM events WHERE event_id=?", (event_id,)).fetchone()
        return row is not None

    def event_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def queue_stats(self) -> QueueStats:
        """返回队列统计快照：深度、总事件数、ACK 水位线、最旧待发送时间、最新 seq。"""
        last_acked = self.last_acked_seq()
        depth = self.queue_depth()
        total = self.event_count()
        oldest_row = self._conn.execute(
            "SELECT MIN(created_at) FROM events WHERE seq > ?",
            (last_acked,),
        ).fetchone()
        oldest = oldest_row[0] if oldest_row else None
        newest_row = self._conn.execute("SELECT MAX(seq) FROM events").fetchone()
        newest = newest_row[0] if newest_row else None
        return QueueStats(
            depth=depth,
            total_events=total,
            last_acked_seq=last_acked,
            oldest_pending_at=datetime.fromisoformat(oldest) if oldest else None,
            newest_seq=int(newest) if newest is not None else None,
        )

    # ------------------------------------------------------------------
    # 审计（追加式，不可修改/删除）
    # ------------------------------------------------------------------

    def audit(
        self,
        kind: str,
        body: dict[str, Any],
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """写入追加式审计记录。

        Args:
            kind: 动作类型（如 "event"、"preliminary"、"shadow_desired"）。
            body: 动作详情（JSON 序列化存储）。
            timestamp: 可选时间戳，默认为当前 UTC 时间。
        """
        ts = (timestamp or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO audit(ts, kind, body) VALUES (?, ?, ?)",
            (ts, kind, json.dumps(body, ensure_ascii=False)),
        )
        self._conn.commit()

    def query_audit(self, start_time: datetime, end_time: datetime) -> list[AuditRecord]:
        """查询 [start_time, end_time] 时间范围内的审计记录，按时间升序返回。"""
        rows = self._conn.execute(
            "SELECT id, ts, kind, body FROM audit WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
            (start_time.isoformat(), end_time.isoformat()),
        ).fetchall()
        return [
            AuditRecord(
                id=int(r[0]),
                ts=datetime.fromisoformat(r[1]),
                kind=r[2],
                body=json.loads(r[3]),
            )
            for r in rows
        ]
