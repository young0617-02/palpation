"""shared/topics.py 单元测试：MQTT 主题生成函数。"""

from __future__ import annotations

from shared import topics


def test_telemetry_topic():
    assert topics.telemetry("gw-001") == "edge/gw-001/telemetry"


def test_event_and_ack_topics():
    assert topics.event("gw-001") == "edge/gw-001/event"
    assert topics.event_ack("gw-001") == "edge/gw-001/ack"


def test_diagnosis_topics():
    assert topics.diagnosis_preliminary("gw-001") == "edge/gw-001/diagnosis/preliminary"
    assert topics.diagnosis_cloud("gw-001") == "edge/gw-001/diagnosis/cloud"


def test_health_and_status_topics():
    assert topics.health("gw-001") == "edge/gw-001/health"
    assert topics.status("gw-001") == "edge/gw-001/status"


def test_shadow_topics():
    assert topics.shadow_desired("gw-001") == "edge/gw-001/shadow/desired"
    assert topics.shadow_reported("gw-001") == "edge/gw-001/shadow/reported"


def test_manifest_and_command_topics():
    assert topics.manifest("gw-001") == "edge/gw-001/manifest"
    assert topics.command("gw-001") == "edge/gw-001/command"


def test_sparkplug_ddata_format():
    t = topics.sparkplug_ddata("GROUP1", "edge-node-1", "device-1")
    assert t == "spBv1.0/GROUP1/DDATA/edge-node-1/device-1"


def test_all_gw_topics_contain_gw_id():
    """所有以 gw_id 为参数的主题函数返回值都应包含 gw_id。"""
    gw = "gw-sim-999"
    fns = (
        topics.telemetry,
        topics.event,
        topics.event_ack,
        topics.diagnosis_preliminary,
        topics.diagnosis_cloud,
        topics.health,
        topics.status,
        topics.shadow_desired,
        topics.shadow_reported,
        topics.manifest,
        topics.command,
    )
    for fn in fns:
        assert gw in fn(gw), f"{fn.__name__} 未包含 gw_id"


def test_sparkplug_ddata_contains_all_parts():
    t = topics.sparkplug_ddata("G", "N", "D")
    assert "G" in t and "N" in t and "D" in t
    assert t.startswith("spBv1.0/")