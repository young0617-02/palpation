"""shared/schemas.py 单元测试：Pydantic 模型构造、必填校验、序列化。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AckPayload,
    CloudDiagnosis,
    DeviceShadow,
    EdgeEvent,
    EdgeHealth,
    EdgeTelemetry,
    InternalSample,
    ModelManifest,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# EdgeTelemetry
# --------------------------------------------------------------------------- #

def test_edge_telemetry_construction_and_default():
    t = EdgeTelemetry(
        gw_id="gw-001",
        device_id_hash="abc",
        metric="vibration_rms",
        window_start=_now(),
        window_end=_now(),
        agg={"mean": 5.0, "max": 6.0},
        raw_rate_hz=1000.0,
    )
    assert t.schema_ver == "1.0"
    assert t.agg["max"] == 6.0


def test_edge_telemetry_missing_required():
    with pytest.raises(ValidationError):
        EdgeTelemetry(gw_id="gw-001", device_id_hash="abc")  # 缺 metric/window_*/agg/raw_rate_hz


# --------------------------------------------------------------------------- #
# EdgeEvent
# --------------------------------------------------------------------------- #

def _make_event(**overrides) -> dict:
    base = dict(
        event_id="e1",
        gw_id="gw-001",
        device_id_hash="abc",
        seq=1,
        detected_at=_now(),
        rule_id="r1",
        severity="high",
        metric="vibration_rms",
        value=6.0,
        threshold=4.5,
    )
    base.update(overrides)
    return base


def test_edge_event_defaults():
    ev = EdgeEvent(**_make_event())
    assert ev.degraded is False
    assert ev.evidence_refs == []
    assert ev.preliminary_diagnosis is None
    assert ev.preliminary_confidence is None
    assert ev.schema_ver == "1.0"


def test_edge_event_severity_literal_valid():
    for sev in ("low", "medium", "high"):
        ev = EdgeEvent(**_make_event(severity=sev))
        assert ev.severity == sev


def test_edge_event_severity_literal_invalid():
    with pytest.raises(ValidationError):
        EdgeEvent(**_make_event(severity="critical"))


def test_edge_event_missing_required():
    with pytest.raises(ValidationError):
        EdgeEvent(event_id="e1", gw_id="gw-001")  # 缺大量必填字段


def test_edge_event_serialization_round_trip():
    ev = EdgeEvent(
        **_make_event(
            severity="medium",
            preliminary_diagnosis="overheat",
            preliminary_confidence=0.8,
            evidence_refs=["s1", "s2"],
            degraded=True,
        )
    )
    data = ev.model_dump()
    assert data["severity"] == "medium"
    assert data["evidence_refs"] == ["s1", "s2"]
    assert data["degraded"] is True
    ev2 = EdgeEvent(**data)
    assert ev2.event_id == "e1"
    assert ev2.preliminary_diagnosis == "overheat"


# --------------------------------------------------------------------------- #
# ModelManifest
# --------------------------------------------------------------------------- #

def _make_manifest(**overrides) -> dict:
    base = dict(
        model_id="anomaly",
        version="0.1.0",
        artifact_url="https://x/m.onnx",
        sha256="abc",
        size_bytes=100,
        quant="int8",
        runtime="onnxruntime",
        target_hardware=["x86"],
        requires_firmware="0.1.0",
        preproc_ver="v1",
    )
    base.update(overrides)
    return base


def test_model_manifest_construction_and_defaults():
    m = ModelManifest(**_make_manifest())
    assert m.metrics_at_register == {}
    assert m.rollout == {}
    assert m.signature is None
    assert m.schema_ver == "1.0"


def test_model_manifest_missing_required():
    with pytest.raises(ValidationError):
        ModelManifest(model_id="anomaly", version="0.1.0")  # 缺大量必填字段


def test_model_manifest_with_optional_fields():
    m = ModelManifest(
        **_make_manifest(
            metrics_at_register={"f1": 0.9},
            rollout={"strategy": "ring"},
            signature="sig-abc",
        )
    )
    assert m.metrics_at_register == {"f1": 0.9}
    assert m.rollout["strategy"] == "ring"
    assert m.signature == "sig-abc"


# --------------------------------------------------------------------------- #
# InternalSample
# --------------------------------------------------------------------------- #

def test_internal_sample_construction():
    s = InternalSample(device_id="pump-01", source="modbus", ts=_now(),
                       metrics={"temp": 46.2})
    assert s.metrics["temp"] == 46.2


def test_internal_sample_missing_required():
    with pytest.raises(ValidationError):
        InternalSample(device_id="pump-01")  # 缺 source/ts/metrics


# --------------------------------------------------------------------------- #
# EdgeHealth
# --------------------------------------------------------------------------- #

def test_edge_health_defaults():
    h = EdgeHealth(
        gw_id="gw-001", ts=_now(), firmware_version="0.1.0",
        model_versions={"anomaly": "0.1.0"}, config_version="v0001",
        inference_p95_ms=12.5, mem_used_mb=256, disk_free_mb=1024,
    )
    assert h.cpu_temp_c is None
    assert h.input_links == {}
    assert h.offline_since is None
    assert h.queue_depth == 0
    assert h.probe_ok is True
    assert h.schema_ver == "1.0"


def test_edge_health_with_optional_fields():
    h = EdgeHealth(
        gw_id="gw-001", ts=_now(), firmware_version="0.1.0",
        model_versions={}, config_version="v0001",
        inference_p95_ms=20.0, mem_used_mb=512, disk_free_mb=500,
        cpu_temp_c=55.0, input_links={"modbus": True}, queue_depth=3,
        probe_ok=False,
    )
    assert h.cpu_temp_c == 55.0
    assert h.input_links == {"modbus": True}
    assert h.queue_depth == 3
    assert h.probe_ok is False


# --------------------------------------------------------------------------- #
# DeviceShadow / AckPayload / CloudDiagnosis
# --------------------------------------------------------------------------- #

def test_device_shadow_defaults():
    s = DeviceShadow()
    assert s.desired == {}
    assert s.reported == {}
    assert s.delta == {}
    assert s.meta == {}


def test_device_shadow_with_values():
    s = DeviceShadow(desired={"v": "0.2"}, reported={"v": "0.1"},
                     delta={"v": "0.2"}, meta={"ts": "x"})
    assert s.desired["v"] == "0.2"
    assert s.delta["v"] == "0.2"


def test_ack_payload_defaults():
    a = AckPayload(gw_id="gw-001", last_acked_seq=5)
    assert a.schema_ver == "1.0"


def test_cloud_diagnosis_defaults():
    d = CloudDiagnosis(event_id="e1", gw_id="gw-001", conclusion="bearing wear",
                       confidence=0.9)
    assert d.citations == []
    assert d.requires_human_review is False
    assert d.schema_ver == "1.0"


def test_cloud_diagnosis_with_citations():
    d = CloudDiagnosis(
        event_id="e1", gw_id="gw-001", conclusion="overheat",
        confidence=0.95, citations=["kb-1", "kb-2"], requires_human_review=True,
    )
    assert d.citations == ["kb-1", "kb-2"]
    assert d.requires_human_review is True