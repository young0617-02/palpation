"""T2 ONNX 异常检测器测试.

覆盖：模型加载、正常/异常检测、降级、推理延迟。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge.stream.anomaly_onnx import AnomalyDetector, ensure_ae_onnx
from edge.stream.window import SlidingWindow

MODEL_PATH = Path("models/anomaly/current/model.onnx")
WINDOW = 32
THRESHOLD = 3.0


def _fill_window(win: SlidingWindow, value: float, n: int = WINDOW) -> None:
    for _ in range(n):
        win.push("vibration_rms", value)


@pytest.fixture(scope="module")
def if_model() -> Path:
    """返回已训练的 Isolation Forest ONNX 模型路径."""
    if not MODEL_PATH.exists():
        pytest.skip("ONNX 模型未生成，请先运行 scripts/train_anomaly_model.py")
    return MODEL_PATH


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
class TestModelLoading:
    def test_load_isolation_forest(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        assert det.backend == "onnxruntime"
        assert det._model_kind == "if"
        assert det.session is not None
        assert det._score_scale > 1.0

    def test_load_ae_model(self, tmp_path: Path) -> None:
        path = ensure_ae_onnx(tmp_path / "ae.onnx", window=WINDOW)
        det = AnomalyDetector(path, window=WINDOW, threshold=THRESHOLD)
        assert det.backend == "onnxruntime"
        assert det._model_kind == "ae"

    def test_metadata_read(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        assert det._score_output == "scores"
        assert det._score_sign == -1.0


# --------------------------------------------------------------------------- #
# 异常检测准确性
# --------------------------------------------------------------------------- #
class TestAnomalyDetection:
    def test_normal_not_flagged(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win = SlidingWindow(WINDOW)
        rng = np.random.default_rng(0)
        for v in rng.uniform(0.5, 2.0, size=WINDOW):
            win.push("vibration_rms", float(v))
        is_anom, score = det.is_anomaly(win)
        assert not is_anom
        assert score < THRESHOLD

    def test_anomaly_flagged(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win = SlidingWindow(WINDOW)
        rng = np.random.default_rng(1)
        for v in rng.uniform(3.0, 8.0, size=WINDOW):
            win.push("vibration_rms", float(v))
        is_anom, score = det.is_anomaly(win)
        assert is_anom
        assert score >= THRESHOLD

    def test_score_separation(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win_n = SlidingWindow(WINDOW)
        _fill_window(win_n, 1.2)
        _, s_n = det.is_anomaly(win_n)
        win_a = SlidingWindow(WINDOW)
        _fill_window(win_a, 6.5)
        _, s_a = det.is_anomaly(win_a)
        assert s_a > s_n

    def test_constant_normal_low_score(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win = SlidingWindow(WINDOW)
        _fill_window(win, 1.0)
        _, score = det.is_anomaly(win)
        assert score < THRESHOLD

    def test_high_vibration_flagged(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win = SlidingWindow(WINDOW)
        _fill_window(win, 7.0)
        is_anom, _ = det.is_anomaly(win)
        assert is_anom


# --------------------------------------------------------------------------- #
# 降级
# --------------------------------------------------------------------------- #
class TestDegradation:
    def test_invalid_model_fallback_numpy(self, tmp_path: Path) -> None:
        """无效 ONNX 文件 → 降级到 numpy AE scorer."""
        path = tmp_path / "invalid.onnx"
        path.write_bytes(b"not a valid onnx model")
        det = AnomalyDetector(path, window=WINDOW, threshold=THRESHOLD)
        assert det.backend == "numpy"
        assert det.session is None
        win = SlidingWindow(WINDOW)
        _fill_window(win, 1.2)
        is_anom, score = det.is_anomaly(win)
        assert isinstance(is_anom, bool)
        assert isinstance(score, float)

    def test_missing_model_auto_create_ae(self, tmp_path: Path) -> None:
        """模型文件不存在 → ensure_ae_onnx 自动创建 AE 降级模型."""
        path = tmp_path / "auto.onnx"
        det = AnomalyDetector(path, window=WINDOW, threshold=THRESHOLD)
        assert path.exists()
        win = SlidingWindow(WINDOW)
        _fill_window(win, 1.2)
        is_anom, score = det.is_anomaly(win)
        assert isinstance(is_anom, bool)
        assert isinstance(score, float)

    def test_degraded_flag_returns_safe(self, tmp_path: Path) -> None:
        """手动标记降级 → is_anomaly 返回 (False, 0.0)."""
        path = ensure_ae_onnx(tmp_path / "ae.onnx", window=WINDOW)
        det = AnomalyDetector(path, window=WINDOW, threshold=THRESHOLD)
        det._degraded = True
        win = SlidingWindow(WINDOW)
        _fill_window(win, 6.5)
        is_anom, score = det.is_anomaly(win)
        assert is_anom is False
        assert score == 0.0

    def test_numpy_backend_detects_anomaly(self, tmp_path: Path) -> None:
        """numpy 降级 scorer 仍能区分正常/异常."""
        path = tmp_path / "invalid.onnx"
        path.write_bytes(b"bad")
        det = AnomalyDetector(path, window=WINDOW, threshold=3.0)
        win_n = SlidingWindow(WINDOW)
        _fill_window(win_n, 1.2)
        _, s_n = det.is_anomaly(win_n)
        win_a = SlidingWindow(WINDOW)
        _fill_window(win_a, 6.5)
        is_a, s_a = det.is_anomaly(win_a)
        assert s_a > s_n
        assert is_a


# --------------------------------------------------------------------------- #
# 推理延迟
# --------------------------------------------------------------------------- #
class TestLatency:
    def test_latency_under_200ms(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win = SlidingWindow(WINDOW)
        _fill_window(win, 1.2)
        det.is_anomaly(win)
        assert det.latency_ms < 200.0

    def test_latency_recorded(self, if_model: Path) -> None:
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        assert det.latency_ms == 0.0
        win = SlidingWindow(WINDOW)
        _fill_window(win, 6.5)
        det.is_anomaly(win)
        assert det.latency_ms > 0.0

    def test_p95_latency_over_multiple_runs(self, if_model: Path) -> None:
        """连续推理 50 次，P95 延迟应 < 200ms."""
        det = AnomalyDetector(if_model, window=WINDOW, threshold=THRESHOLD)
        win = SlidingWindow(WINDOW)
        _fill_window(win, 1.2)
        latencies: list[float] = []
        for _ in range(50):
            det.is_anomaly(win)
            latencies.append(det.latency_ms)
        p95 = float(np.percentile(latencies, 95))
        assert p95 < 200.0