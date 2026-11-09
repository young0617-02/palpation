from pathlib import Path

from edge.cloud.masking import bin_value, hash_device_id
from edge.stream.anomaly_onnx import AnomalyDetector, ensure_ae_onnx
from edge.stream.window import SlidingWindow


def test_hmac_irreversible():
    a = hash_device_id("pump-de-01", "salt")
    b = hash_device_id("pump-de-01", "other")
    assert a != "pump-de-01"
    assert len(a) == 64
    assert a != b


def test_temp_bins():
    assert bin_value(46.2, [0, 40, 60, 80, 100]) == "[40,60)"


def test_t2_scores_fault(tmp_path: Path):
    path = ensure_ae_onnx(tmp_path / "model.onnx", window=32)
    det = AnomalyDetector(path, window=32, threshold=3.0)
    win = SlidingWindow(32)
    for _ in range(32):
        win.push("vibration_rms", 1.2)
    ok, s_n = det.is_anomaly(win)
    assert not ok
    for _ in range(32):
        win.push("vibration_rms", 6.5)
    bad, s_f = det.is_anomaly(win)
    assert s_f > s_n
    assert bad
