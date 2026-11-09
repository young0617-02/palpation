from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from edge.stream.window import SlidingWindow

log = logging.getLogger(__name__)


def ensure_ae_onnx(path: Path, window: int = 32, mu: float = 1.2, sigma: float = 0.25) -> Path:
    """Build a tiny 1-D AE-style scorer: mean(|(x-mu)/sigma|)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, window])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])
    mu_t = numpy_helper.from_array(np.array([mu], dtype=np.float32), name="mu")
    sigma_t = numpy_helper.from_array(np.array([sigma], dtype=np.float32), name="sigma")
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["x", "mu"], ["centered"]),
            helper.make_node("Div", ["centered", "sigma"], ["normed"]),
            helper.make_node("Abs", ["normed"], ["absn"]),
            helper.make_node("ReduceMean", ["absn"], ["score"], keepdims=0),
        ],
        "ae_score",
        [x],
        [y],
        [mu_t, sigma_t],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


class AnomalyDetector:
    """T2 ONNX 异常检测器，支持 Isolation Forest 与 AE 两种模型格式.

    - Isolation Forest：读取 ONNX ``metadata_props`` 中的缩放参数，将
      ``decision_function`` 映射为"越大越异常"的分数。
    - AE scorer：直接使用 ONNX 输出的 score。
    - 降级：ONNX 加载失败时回退到 numpy AE scorer；完全不可用时
      ``is_anomaly`` 返回 ``(False, 0.0)``。
    """

    def __init__(self, model_path: Path, window: int, threshold: float):
        self.model_path = model_path
        self.window = window
        self.threshold = threshold
        self.session: Any = None
        self.backend = "numpy"
        self.latency_ms: float = 0.0
        # 模型元数据（默认 AE scorer 参数）
        self._input_name = "x"
        self._score_output = "score"
        self._score_sign = 1.0
        self._score_scale = 1.0
        self._model_kind = "ae"
        self._degraded = False
        try:
            import onnxruntime as ort

            if not model_path.exists():
                ensure_ae_onnx(model_path, window=window)
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
            self._detect_model_kind()
        except Exception as exc:  # noqa: BLE001
            log.warning("ONNX 模型加载失败，降级到 numpy scorer: %s", exc)
            self.session = None
            self.backend = "numpy"

    def _detect_model_kind(self) -> None:
        """检测 ONNX 模型类型并读取分数缩放 metadata."""
        outputs = [o.name for o in self.session.get_outputs()]
        inputs = [i.name for i in self.session.get_inputs()]
        self._input_name = inputs[0]
        meta = self.session.get_modelmeta()
        custom = getattr(meta, "custom_metadata_map", {}) or {}
        kind = custom.get("model_kind", "")
        if kind == "isolation_forest" or "scores" in outputs:
            self._model_kind = "if"
            self._score_output = custom.get("score_output", "scores")
            if self._score_output not in outputs:
                self._score_output = next((n for n in outputs if "score" in n.lower()), outputs[-1])
            self._score_sign = float(custom.get("score_sign", "-1"))
            self._score_scale = float(custom.get("score_scale", "1.0"))
        else:
            self._model_kind = "ae"
            self._score_output = outputs[0]

    def score(self, values: list[float]) -> float:
        """计算异常分数（越大越异常），同时记录推理延迟."""
        arr = np.array(values, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        if arr.size < self.window:
            pad = np.full(self.window - arr.size, arr.mean() if arr.size else 0.0, dtype=np.float32)
            arr = np.concatenate([pad, arr])
        else:
            arr = arr[-self.window :]
        if self.session is not None:
            t0 = time.perf_counter()
            out = self.session.run(
                [self._score_output], {self._input_name: arr.reshape(1, self.window)}
            )[0]
            self.latency_ms = (time.perf_counter() - t0) * 1000
            raw = float(np.array(out).reshape(-1)[0])
            if self._model_kind == "if":
                return max(0.0, self._score_sign * raw) * self._score_scale
            return raw
        # numpy AE 降级 scorer
        t0 = time.perf_counter()
        mu, sigma = 1.2, 0.25
        s = float(np.mean(np.abs((arr - mu) / sigma)))
        self.latency_ms = (time.perf_counter() - t0) * 1000
        return s

    def is_anomaly(
        self, window: SlidingWindow, metric: str = "vibration_rms"
    ) -> tuple[bool, float]:
        """判断窗口是否异常，返回 (是否异常, 异常分数).

        模型完全不可用时安全降级返回 ``(False, 0.0)``。
        """
        if self._degraded:
            return False, 0.0
        try:
            s = self.score(window.values(metric))
        except Exception as exc:  # noqa: BLE001
            log.warning("异常检测推理失败，降级返回 (False, 0.0): %s", exc)
            return False, 0.0
        return s >= self.threshold, s
