"""E3 监控可观测：本地探针集合。

探针覆盖端侧关键健康维度：
- 推理延迟异常（P95 超阈值）
- 磁盘空间不足（影响日志/缓存写入）
- 温度过高（触发降级）
- 队列积压（背压预警）
- 模型未加载（无法推理）
- MQTT 连接断开（影响上报）

每个探针返回 ProbeResult，run_all 聚合运行。critical_failures
返回影响业务的严重失败（推理异常/模型未加载/MQTT 断连）。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from edge.monitoring.metrics import EdgeMetrics


@dataclass
class ProbeResult:
    """探针结果。

    Attributes:
        name: 探针名称（如 "inference_health"）
        passed: 是否通过
        detail: 详细描述（通过/失败原因）
        timestamp: 探针执行时间（UTC）
    """

    name: str
    passed: bool
    detail: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


#: 影响业务的严重探针名称集合
CRITICAL_PROBE_NAMES: frozenset[str] = frozenset(
    {"inference_health", "model_loaded", "mqtt_connected"}
)


class LocalProbes:
    """本地探针集合。

    所有 check_* 方法纯函数式（不持有状态），可独立调用或通过 run_all 聚合。
    """

    # --- 单项探针 ---

    def check_inference_health(self, p95_ms: float, threshold_ms: float = 3000) -> ProbeResult:
        """推理延迟异常探针：P95 超阈值则失败。"""
        passed = p95_ms <= threshold_ms
        if passed:
            detail = f"p95={p95_ms:.2f}ms within threshold {threshold_ms}ms"
        else:
            detail = f"p95={p95_ms:.2f}ms exceeds threshold {threshold_ms}ms"
        return ProbeResult(name="inference_health", passed=passed, detail=detail)

    def check_disk_space(self, path: Path, min_mb: int = 500) -> ProbeResult:
        """磁盘空间探针：可用空间 < min_mb 则失败。

        path 不存在时回退到当前目录检测。
        """
        target = path if path.exists() else Path(".")
        try:
            usage = shutil.disk_usage(str(target))
            free_mb = usage.free // (1024 * 1024)
        except OSError as exc:
            return ProbeResult(
                name="disk_space",
                passed=False,
                detail=f"disk usage query failed: {exc}",
            )
        passed = free_mb >= min_mb
        if passed:
            detail = f"disk free={free_mb}MB >= min {min_mb}MB at {target}"
        else:
            detail = f"disk free={free_mb}MB < min {min_mb}MB at {target}"
        return ProbeResult(name="disk_space", passed=passed, detail=detail)

    def check_temperature(self, temp_c: float | None, threshold_c: float = 80.0) -> ProbeResult:
        """温度探针：温度超阈值则失败，温度不可读（None）则通过（降级由 guard 处理）。"""
        if temp_c is None:
            return ProbeResult(
                name="temperature",
                passed=True,
                detail="temperature sensor unavailable, skipped",
            )
        passed = temp_c <= threshold_c
        if passed:
            detail = f"temp={temp_c:.1f}C within threshold {threshold_c}C"
        else:
            detail = f"temp={temp_c:.1f}C exceeds threshold {threshold_c}C"
        return ProbeResult(name="temperature", passed=passed, detail=detail)

    def check_queue_depth(self, depth: int, max_depth: int = 10000) -> ProbeResult:
        """队列积压探针：深度超 max_depth 则失败。"""
        passed = depth <= max_depth
        if passed:
            detail = f"queue_depth={depth} within max {max_depth}"
        else:
            detail = f"queue_depth={depth} exceeds max {max_depth}"
        return ProbeResult(name="queue_depth", passed=passed, detail=detail)

    def check_model_loaded(self, model_version: str | None) -> ProbeResult:
        """模型加载探针：version 为 None 或空或 "unknown" 则失败。"""
        loaded = model_version is not None and model_version != "" and model_version != "unknown"
        if loaded:
            detail = f"model loaded: version={model_version}"
        else:
            detail = f"model not loaded: version={model_version!r}"
        return ProbeResult(name="model_loaded", passed=loaded, detail=detail)

    def check_mqtt_connected(self, connected: bool) -> ProbeResult:
        """MQTT 连接探针：未连接则失败。"""
        detail = "mqtt connected" if connected else "mqtt disconnected"
        return ProbeResult(name="mqtt_connected", passed=connected, detail=detail)

    # --- 聚合 ---

    def run_all(
        self,
        metrics: EdgeMetrics,
        *,
        disk_path: Path | None = None,
        disk_min_mb: int = 500,
        temp_c: float | None = None,
        temp_threshold_c: float = 80.0,
        mqtt_connected: bool | None = None,
    ) -> list[ProbeResult]:
        """运行所有探针。

        必选参数 metrics 提供 inference_p95_ms / queue_depth / model_version。
        可选参数提供后则对应探针纳入运行，未提供则跳过该探针（不加入结果列表），
        调用方可单独调用对应 check_* 方法补齐。
        """
        results: list[ProbeResult] = [
            self.check_inference_health(metrics.inference_p95_ms),
            self.check_queue_depth(metrics.queue_depth),
            self.check_model_loaded(metrics.model_version),
        ]
        if disk_path is not None:
            results.append(self.check_disk_space(disk_path, min_mb=disk_min_mb))
        if temp_c is not None:
            results.append(self.check_temperature(temp_c, threshold_c=temp_threshold_c))
        if mqtt_connected is not None:
            results.append(self.check_mqtt_connected(mqtt_connected))
        return results

    def has_failures(self, results: list[ProbeResult]) -> bool:
        """是否有探针失败。"""
        return any(not r.passed for r in results)

    def critical_failures(self, results: list[ProbeResult]) -> list[ProbeResult]:
        """返回严重失败（影响业务的探针失败）。

        严重探针集合见 CRITICAL_PROBE_NAMES。
        """
        return [r for r in results if not r.passed and r.name in CRITICAL_PROBE_NAMES]
