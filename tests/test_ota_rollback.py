from pathlib import Path

from mgmt_agent.version import VersionSlots


def test_activate_and_rollback(tmp_path: Path):
    slots = VersionSlots(tmp_path)
    slots.slot_dir("anomaly", "0.9.1").joinpath("model.onnx").write_bytes(b"old")
    slots.slot_dir("anomaly", "0.9.2").joinpath("model.onnx").write_bytes(b"new")
    slots.activate("anomaly", "0.9.1")
    assert slots.current_version("anomaly") == "0.9.1"
    slots.activate("anomaly", "0.9.2")
    assert slots.current_version("anomaly") == "0.9.2"
    slots.rollback("anomaly", "0.9.1")
    assert slots.current_version("anomaly") == "0.9.1"
