from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EdgeTelemetry(BaseModel):
    gw_id: str
    device_id_hash: str
    metric: str
    window_start: datetime
    window_end: datetime
    agg: dict[str, float]
    raw_rate_hz: float
    schema_ver: str = "1.0"


class EdgeEvent(BaseModel):
    event_id: str
    gw_id: str
    device_id_hash: str
    seq: int
    detected_at: datetime
    rule_id: str
    severity: Literal["low", "medium", "high"]
    metric: str
    value: float
    threshold: float
    preliminary_diagnosis: str | None = None
    preliminary_confidence: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    degraded: bool = False
    schema_ver: str = "1.0"


class ModelManifest(BaseModel):
    model_id: str
    version: str
    artifact_url: str
    sha256: str
    size_bytes: int
    quant: str
    runtime: str
    target_hardware: list[str]
    requires_firmware: str
    preproc_ver: str
    metrics_at_register: dict[str, Any] = Field(default_factory=dict)
    rollout: dict[str, Any] = Field(default_factory=dict)
    signature: str | None = None
    schema_ver: str = "1.0"


class EdgeHealth(BaseModel):
    gw_id: str
    ts: datetime
    firmware_version: str
    model_versions: dict[str, str]
    config_version: str
    inference_p95_ms: float
    mem_used_mb: int
    disk_free_mb: int
    cpu_temp_c: float | None = None
    input_links: dict[str, bool] = Field(default_factory=dict)
    offline_since: datetime | None = None
    queue_depth: int = 0
    probe_ok: bool = True
    schema_ver: str = "1.0"


class DeviceShadow(BaseModel):
    desired: dict[str, Any] = Field(default_factory=dict)
    reported: dict[str, Any] = Field(default_factory=dict)
    delta: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class InternalSample(BaseModel):
    device_id: str
    source: str
    ts: datetime
    metrics: dict[str, float]


class AckPayload(BaseModel):
    gw_id: str
    last_acked_seq: int
    schema_ver: str = "1.0"


class CloudDiagnosis(BaseModel):
    event_id: str
    gw_id: str
    conclusion: str
    confidence: float
    citations: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    schema_ver: str = "1.0"
