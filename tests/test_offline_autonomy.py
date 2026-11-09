from datetime import datetime, timezone
from pathlib import Path

from edge.cloud.masking import hash_device_id
from edge.queue.sqlite_queue import SqliteQueue
from shared.schemas import EdgeEvent


def test_offline_queue_is_local_authority(tmp_path: Path):
    q = SqliteQueue(tmp_path / "q.db")
    for i in range(5):
        q.append_event(
            EdgeEvent(
                event_id=f"e-{i}",
                gw_id="gw-sim-001",
                device_id_hash=hash_device_id("pump-de-01", "salt"),
                seq=0,
                detected_at=datetime.now(timezone.utc),
                rule_id="bearing_rms_high",
                severity="high",
                metric="vibration_rms",
                value=6.0,
                threshold=4.5,
            )
        )
    assert q.event_count() == 5
    assert q.queue_depth() == 5
    q.set_acked(3)
    assert q.queue_depth() == 2


def test_audit_append_only(tmp_path: Path):
    q = SqliteQueue(tmp_path / "q.db")
    q.audit("event", {"x": 1})
    try:
        q._conn.execute("UPDATE audit SET kind='x'")
        q._conn.commit()
        ok = True
    except Exception:
        ok = False
    assert ok is False
