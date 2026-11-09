"""E3 监控可观测：边缘指标收集器 + Prometheus 文本格式导出。

纯 Python 实现，不依赖 prometheus_client 库。指标收集覆盖：
- 推理 P95 延迟（毫秒，滑动窗口最近 1000 次）
- 模型版本（字符串，导出为 info 指标 + version label）
- 本地队列深度
- 离线时长（秒）
- 本地缓存命中率（累计命中/总请求）
- 云端升级率（累计升级/总请求，下降=端侧在进步）

Prometheus 文本格式遵循：
    # HELP <name> <description>
    # TYPE <name> gauge
    <name>[{<labels>}] <value>
"""

from __future__ import annotations

from collections import deque
from typing import Any


class EdgeMetrics:
    """边缘指标收集器，线程不安全（调用方需在同一线程或自行加锁）。

    P95 计算用滑动窗口（最近 ``p95_window`` 次推理，默认 1000），
    与 edge/agent/state_machine.py 中 p95_inference_ms() 算法保持一致：
    sorted 样本取 int(0.95 * (n-1)) 索引。
    """

    #: 滑动窗口默认大小
    DEFAULT_P95_WINDOW = 1000

    def __init__(self, p95_window: int = DEFAULT_P95_WINDOW) -> None:
        if p95_window <= 0:
            msg = f"p95_window must be positive, got {p95_window}"
            raise ValueError(msg)
        self._p95_window = p95_window
        self._latencies: deque[float] = deque(maxlen=p95_window)

        # 静态/可设置指标
        self._model_version: str = "unknown"
        self._queue_depth: int = 0
        self._offline_duration_s: int = 0

        # 累计计数器（用于计算比率）
        self._cache_hits: int = 0
        self._cache_total: int = 0
        self._cloud_escalations: int = 0
        self._cloud_total: int = 0

    # --- 属性访问 ---

    @property
    def inference_p95_ms(self) -> float:
        """推理 P95 延迟（毫秒），无样本时返回 0.0。"""
        return self._compute_p95()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def offline_duration_s(self) -> int:
        return self._offline_duration_s

    @property
    def local_cache_hit_rate(self) -> float:
        """本地缓存命中率，无样本时返回 0.0。"""
        if self._cache_total == 0:
            return 0.0
        return self._cache_hits / self._cache_total

    @property
    def cloud_escalation_rate(self) -> float:
        """云端升级率，无样本时返回 0.0。"""
        if self._cloud_total == 0:
            return 0.0
        return self._cloud_escalations / self._cloud_total

    # --- 记录/设置方法 ---

    def record_inference(self, latency_ms: float) -> None:
        """记录一次推理延迟（毫秒），维护滑动窗口。"""
        if latency_ms < 0:
            msg = f"latency_ms must be non-negative, got {latency_ms}"
            raise ValueError(msg)
        self._latencies.append(float(latency_ms))

    def set_model_version(self, version: str) -> None:
        """设置当前模型版本。"""
        self._model_version = version

    def set_queue_depth(self, depth: int) -> None:
        """设置本地队列深度。"""
        if depth < 0:
            msg = f"depth must be non-negative, got {depth}"
            raise ValueError(msg)
        self._queue_depth = depth

    def set_offline_duration(self, seconds: int) -> None:
        """设置离线时长（秒）。"""
        if seconds < 0:
            msg = f"seconds must be non-negative, got {seconds}"
            raise ValueError(msg)
        self._offline_duration_s = seconds

    def record_cache_hit(self, hit: bool) -> None:
        """记录一次缓存命中/未命中。"""
        self._cache_total += 1
        if hit:
            self._cache_hits += 1

    def record_cloud_escalation(self, escalated: bool) -> None:
        """记录一次云端升级/未升级。"""
        self._cloud_total += 1
        if escalated:
            self._cloud_escalations += 1

    # --- 导出/快照 ---

    def _compute_p95(self) -> float:
        """计算当前滑动窗口的 P95。"""
        if not self._latencies:
            return 0.0
        xs = sorted(self._latencies)
        idx = int(0.95 * (len(xs) - 1))
        return xs[idx]

    def export_prometheus(self) -> str:
        """导出 Prometheus 文本格式。

        格式：
            # HELP <name> <desc>
            # TYPE <name> gauge
            <name>[{labels}] <value>

        每个指标以换行结尾，整体以换行结尾。
        """
        lines: list[str] = [
            "# HELP edge_inference_p95_ms Edge inference p95 latency in milliseconds",
            "# TYPE edge_inference_p95_ms gauge",
            f"edge_inference_p95_ms {self.inference_p95_ms}",
            "# HELP edge_model_version_info Current model version",
            "# TYPE edge_model_version_info gauge",
            f'edge_model_version_info{{version="{self._escape_label(self._model_version)}"}} 1',
            "# HELP edge_queue_depth Local queue depth",
            "# TYPE edge_queue_depth gauge",
            f"edge_queue_depth {self._queue_depth}",
            "# HELP edge_offline_duration_s Offline duration in seconds",
            "# TYPE edge_offline_duration_s gauge",
            f"edge_offline_duration_s {self._offline_duration_s}",
            "# HELP edge_local_cache_hit_rate Local cache hit rate",
            "# TYPE edge_local_cache_hit_rate gauge",
            f"edge_local_cache_hit_rate {self.local_cache_hit_rate}",
            "# HELP edge_cloud_escalation_rate Cloud escalation rate",
            "# TYPE edge_cloud_escalation_rate gauge",
            f"edge_cloud_escalation_rate {self.cloud_escalation_rate}",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _escape_label(value: str) -> str:
        """转义 Prometheus label 值中的特殊字符。"""
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def snapshot(self) -> dict[str, Any]:
        """返回所有指标快照（深拷贝当前值）。"""
        return {
            "inference_p95_ms": self.inference_p95_ms,
            "model_version": self._model_version,
            "queue_depth": self._queue_depth,
            "offline_duration_s": self._offline_duration_s,
            "local_cache_hit_rate": self.local_cache_hit_rate,
            "cloud_escalation_rate": self.cloud_escalation_rate,
            "inference_samples": len(self._latencies),
            "cache_hits": self._cache_hits,
            "cache_total": self._cache_total,
            "cloud_escalations": self._cloud_escalations,
            "cloud_total": self._cloud_total,
        }

    def reset(self) -> None:
        """重置所有指标（主要用于测试）。"""
        self._latencies.clear()
        self._model_version = "unknown"
        self._queue_depth = 0
        self._offline_duration_s = 0
        self._cache_hits = 0
        self._cache_total = 0
        self._cloud_escalations = 0
        self._cloud_total = 0
