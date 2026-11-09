"""T2 判别式异常检测模型训练脚本.

用 sklearn Isolation Forest 训练振动异常检测模型，并用 skl2onnx 导出 ONNX。
合成数据：正常振动 RMS 0.5-2.0，异常 3.0-8.0。
输出：models/anomaly/current/model.onnx（<20MB）。

缩放策略：IF decision_function 越负越异常，取 ``max(0, -score) * scale`` 作为
异常分数，使异常样本分数 ≈ 5.0（> 默认 threshold=3.0），正常样本分数 ≈ 0。
缩放因子写入 ONNX ``metadata_props``，由 ``anomaly_onnx.py`` 读取应用。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WINDOW = 32
RANDOM_SEED = 42
MODEL_DIR = Path("models/anomaly/current")
MODEL_PATH = MODEL_DIR / "model.onnx"
# 异常样本目标分数（应 > 配置中的 score_threshold=3.0）
TARGET_ANOMALY_SCORE = 5.0


def build_training_data(
    n_normal: int = 2000, n_anomaly: int = 200, seed: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """生成合成训练/验证数据.

    正常样本：每维 ~ Uniform(0.5, 2.0)，模拟健康振动 RMS 窗口。
    异常样本：每维 ~ Uniform(3.0, 8.0)，模拟故障振动。
    """
    rng = np.random.default_rng(seed)
    normal = rng.uniform(0.5, 2.0, size=(n_normal, WINDOW)).astype(np.float32)
    anomaly = rng.uniform(3.0, 8.0, size=(n_anomaly, WINDOW)).astype(np.float32)
    return normal, anomaly


def train_isolation_forest(normal: np.ndarray):
    """训练 Isolation Forest（仅用正常数据）."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=80,
        max_samples=256,
        contamination=0.05,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(normal)
    return model


def compute_scale(model, anomaly: np.ndarray) -> float:
    """计算分数缩放因子.

    ``raw = max(0, -decision_function)``，使 ``mean(raw[anomaly]) * scale ≈ TARGET``。
    """
    decision = model.decision_function(anomaly)
    raw = np.maximum(0.0, -decision)
    mean_raw = float(raw.mean())
    if mean_raw < 1e-9:
        return 1.0
    return TARGET_ANOMALY_SCORE / mean_raw


def convert_to_onnx(model, window: int = WINDOW, opset: int = 15, score_scale: float = 1.0):
    """用 skl2onnx 将 Isolation Forest 转换为 ONNX，并写入缩放 metadata."""
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, window]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": opset, "ai.onnx.ml": 3},
        options={id(model): {"score_samples": True}},
    )
    # 写入 metadata：模型类型 + 分数缩放参数
    onnx.helper.set_model_props(
        onx,
        {
            "model_kind": "isolation_forest",
            "score_output": "scores",
            "score_sign": "-1",
            "score_scale": f"{score_scale:.6f}",
        },
    )
    return onx


def evaluate_onnx(model_path: Path, normal: np.ndarray, anomaly: np.ndarray) -> dict[str, float]:
    """用 onnxruntime 评估模型，返回指标和延迟."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = [o.name for o in sess.get_outputs()]
    meta = sess.get_modelmeta()
    custom = meta.custom_metadata_map if hasattr(meta, "custom_metadata_map") else {}

    score_name = custom.get("score_output", "scores")
    if score_name not in outputs:
        score_name = next((n for n in outputs if "score" in n.lower()), outputs[-1])
    score_sign = float(custom.get("score_sign", "-1"))
    score_scale = float(custom.get("score_scale", "1.0"))

    def _score_batch(batch: np.ndarray) -> tuple[np.ndarray, float]:
        latencies = []
        scores = []
        for row in batch:
            t0 = time.perf_counter()
            out = sess.run([score_name], {input_name: row.reshape(1, -1)})[0]
            latencies.append(time.perf_counter() - t0)
            raw = float(np.asarray(out).reshape(-1)[0])
            scores.append(max(0.0, score_sign * raw) * score_scale)
        return np.array(scores), float(np.mean(latencies) * 1000)

    normal_scores, normal_lat = _score_batch(normal[:200])
    anomaly_scores, anomaly_lat = _score_batch(anomaly)

    return {
        "normal_score_mean": float(normal_scores.mean()),
        "normal_score_max": float(normal_scores.max()),
        "anomaly_score_mean": float(anomaly_scores.mean()),
        "anomaly_score_min": float(anomaly_scores.min()),
        "normal_latency_ms": normal_lat,
        "anomaly_latency_ms": anomaly_lat,
        "separation": float(anomaly_scores.mean() - normal_scores.mean()),
    }


def main() -> int:
    print("[train] 生成合成数据 ...")
    normal, anomaly = build_training_data()
    print(f"  正常样本: {normal.shape}  异常样本: {anomaly.shape}")

    print("[train] 训练 Isolation Forest ...")
    model = train_isolation_forest(normal)

    # sklearn 侧快速验证
    train_pred = model.predict(normal[:200])
    anomaly_pred = model.predict(anomaly)
    train_acc = float((train_pred == 1).mean())
    anomaly_detect = float((anomaly_pred == -1).mean())
    print(f"  sklearn 正常识别率: {train_acc:.3f}  异常检出率: {anomaly_detect:.3f}")

    print("[train] 计算分数缩放因子 ...")
    score_scale = compute_scale(model, anomaly)
    print(f"  score_scale: {score_scale:.4f}")

    print("[train] 转换为 ONNX ...")
    onx = convert_to_onnx(model, window=WINDOW, score_scale=score_scale)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_bytes = onx.SerializeToString()
    MODEL_PATH.write_bytes(onnx_bytes)
    size_mb = len(onnx_bytes) / (1024 * 1024)
    print(f"  模型已保存: {MODEL_PATH}  大小: {size_mb:.4f} MB ({len(onnx_bytes)} bytes)")

    if size_mb >= 20:
        print(f"[ERROR] 模型体积 {size_mb:.2f}MB 超过 20MB 限制", file=sys.stderr)
        return 1

    print("[train] ONNX 模型评估 ...")
    metrics = evaluate_onnx(MODEL_PATH, normal, anomaly)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if metrics["separation"] <= 0:
        print("[ERROR] 异常分数与正常分数无有效分离", file=sys.stderr)
        return 1

    print("[train] 完成 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
