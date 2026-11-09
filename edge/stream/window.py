from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass
class WindowStats:
    avg: float
    max: float
    min: float
    p95: float
    std: float
    slope: float
    sample_count: int
    zscore: float
    last: float


class SlidingWindow:
    def __init__(self, size: int = 32):
        self.size = size
        self._bufs: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=size))
        self._ewma: dict[str, float] = {}

    def push(self, metric: str, value: float) -> WindowStats:
        buf = self._bufs[metric]
        buf.append(value)
        arr = np.array(buf, dtype=float)
        avg = float(arr.mean())
        std = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
        z = 0.0 if std < 1e-9 else float((value - avg) / std)
        x = np.arange(len(arr), dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) >= 3 else 0.0
        alpha = 0.2
        prev = self._ewma.get(metric, value)
        self._ewma[metric] = alpha * value + (1 - alpha) * prev
        return WindowStats(
            avg=avg,
            max=float(arr.max()),
            min=float(arr.min()),
            p95=float(np.percentile(arr, 95)),
            std=std,
            slope=slope,
            sample_count=len(arr),
            zscore=z,
            last=value,
        )

    def snapshot_id(self, metric: str) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        return f"win:{metric}:{ts}"

    def values(self, metric: str) -> list[float]:
        return list(self._bufs[metric])
