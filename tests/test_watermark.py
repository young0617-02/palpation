from pathlib import Path

from edge.queue.sqlite_queue import SqliteQueue
from edge.queue.watermark import apply_ack, next_batch
from edge.stream.rules import RuleEngine, RuleHit
from shared.schemas import EdgeEvent
from datetime import datetime, timezone


def _event(eid: str) -> EdgeEvent:
    return EdgeEvent(
        event_id=eid,
        gw_id="gw-sim-001",
        device_id_hash="abc",
        seq=0,
        detected_at=datetime.now(timezone.utc),
        rule_id="bearing_rms_high",
        severity="high",
        metric="vibration_rms",
        value=6.1,
        threshold=4.5,
    )


def test_queue_seq_and_ack(tmp_path: Path):
    q = SqliteQueue(tmp_path / "q.db")
    e1 = q.append_event(_event("01HTEST000000000000000001"))
    e2 = q.append_event(_event("01HTEST000000000000000002"))
    assert e1.seq == 1
    assert e2.seq == 2
    assert q.queue_depth() == 2
    batch = next_batch(q, 100)
    assert [e.seq for e in batch] == [1, 2]
    apply_ack(q, 1)
    assert q.last_acked_seq() == 1
    rest = next_batch(q, 100)
    assert [e.seq for e in rest] == [2]


def test_event_id_unique(tmp_path: Path):
    q = SqliteQueue(tmp_path / "q.db")
    q.append_event(_event("dup-1"))
    assert q.has_event_id("dup-1")
    try:
        q.append_event(_event("dup-1"))
        raised = False
    except Exception:
        raised = True
    assert raised


def test_rules_fire():
    eng = RuleEngine([{"id": "bearing_rms_high", "metric": "vibration_rms", "op": ">", "threshold": 4.5, "severity": "high"}])
    hits = eng.evaluate({"vibration_rms": 6.2})
    assert hits and hits[0].rule_id == "bearing_rms_high"
    assert eng.evaluate({"vibration_rms": 1.2}) == []
