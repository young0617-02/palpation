"""ShadowBox 增强功能与 NorthMqtt 影子 MQTT 集成单元测试。

覆盖：
- desired/reported/delta 收敛计算
- needs_pull() 待拉取字段
- apply_desired / update_reported 逐字段收敛
- sync_status 同步状态概览
- "边缘永不自己决定版本"原则
- NorthMqtt 各 publish_* 方法（mock paho-mqtt 客户端）
- 重连配置
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from edge.cloud.mqtt_north import NorthMqtt
from edge.cloud.shadow import ShadowBox, _delta
from shared import topics
from shared.schemas import EdgeEvent, EdgeHealth, EdgeTelemetry


# ====================================================================
# ShadowBox 增强功能
# ====================================================================

# --- needs_pull ---

def test_needs_pull_empty_when_synced():
    sb = ShadowBox()
    sb.apply_desired({"model_version": "1.0"})
    sb.update_reported("model_version", "1.0")
    assert sb.needs_pull() == {}


def test_needs_pull_returns_delta_fields():
    sb = ShadowBox()
    sb.apply_desired({"model_version": "2.0", "config_version": "v2"})
    sb.update_reported("model_version", "2.0")
    # model_version 已收敛，config_version 尚未收敛
    assert sb.needs_pull() == {"config_version": "v2"}


def test_needs_pull_all_fields_when_reported_empty():
    sb = ShadowBox()
    sb.apply_desired({"a": 1, "b": 2})
    assert sb.needs_pull() == {"a": 1, "b": 2}


def test_needs_pull_clears_after_full_convergence():
    sb = ShadowBox()
    sb.apply_desired({"a": 1, "b": 2})
    sb.update_reported("a", 1)
    assert sb.needs_pull() == {"b": 2}
    sb.update_reported("b", 2)
    assert sb.needs_pull() == {}


# --- apply_desired ---

def test_apply_desired_returns_delta():
    sb = ShadowBox(reported={"model_version": "1.0"})
    delta = sb.apply_desired({"model_version": "2.0", "config_version": "v2"})
    assert delta == {"model_version": "2.0", "config_version": "v2"}


def test_apply_desired_equivalent_to_set_desired():
    """apply_desired 与 set_desired 语义一致。"""
    sb1 = ShadowBox(reported={"x": 1})
    sb2 = ShadowBox(reported={"x": 1})
    d1 = sb1.apply_desired({"x": 2, "y": 3})
    d2 = sb2.set_desired({"x": 2, "y": 3})
    assert d1 == d2
    assert sb1.desired == sb2.desired


def test_apply_desired_overwrites_previous_desired():
    sb = ShadowBox()
    sb.apply_desired({"a": 1})
    sb.apply_desired({"b": 2})
    assert sb.desired == {"b": 2}


# --- update_reported ---

def test_update_reported_single_field():
    sb = ShadowBox()
    sb.apply_desired({"model_version": "2.0", "config_version": "v2"})
    delta = sb.update_reported("model_version", "2.0")
    # model_version 已收敛，config_version 仍待拉取
    assert delta == {"config_version": "v2"}
    assert sb.reported["model_version"] == "2.0"


def test_update_reported_refreshes_synced_at():
    sb = ShadowBox()
    before = datetime.now(UTC)
    sb.update_reported("k", "v")
    after = datetime.now(UTC)
    assert sb.synced_at is not None
    assert before - timedelta(seconds=1) <= sb.synced_at <= after + timedelta(seconds=1)


def test_update_reported_full_convergence():
    sb = ShadowBox()
    sb.apply_desired({"a": 1, "b": 2, "c": 3})
    sb.update_reported("a", 1)
    sb.update_reported("b", 2)
    delta = sb.update_reported("c", 3)
    assert delta == {}


def test_update_reported_does_not_touch_desired():
    """边缘更新 reported 时绝不修改 desired。"""
    sb = ShadowBox()
    sb.apply_desired({"model_version": "2.0"})
    desired_before = dict(sb.desired)
    sb.update_reported("model_version", "1.0")
    assert sb.desired == desired_before


# --- sync_status ---

def test_sync_status_in_sync_when_delta_empty():
    sb = ShadowBox()
    sb.apply_desired({"a": 1})
    sb.update_reported("a", 1)
    status = sb.sync_status()
    assert status["in_sync"] is True
    assert status["pending_fields"] == []


def test_sync_status_not_in_sync_when_delta_nonempty():
    sb = ShadowBox()
    sb.apply_desired({"a": 1, "b": 2})
    sb.update_reported("a", 1)
    status = sb.sync_status()
    assert status["in_sync"] is False
    assert status["pending_fields"] == ["b"]


def test_sync_status_offline_duration():
    sb = ShadowBox()
    offline_start = datetime.now(UTC) - timedelta(seconds=30)
    sb.mark_offline(at=offline_start)
    status = sb.sync_status()
    assert status["offline_duration_s"] is not None
    assert status["offline_duration_s"] >= 29  # 容忍 1s 抖动


def test_sync_status_no_offline_duration_when_online():
    sb = ShadowBox()
    status = sb.sync_status()
    assert status["offline_duration_s"] is None


def test_sync_status_backlog_count():
    sb = ShadowBox()
    sb.incr_backlog(5)
    status = sb.sync_status()
    assert status["backlog_count"] == 5


def test_sync_status_synced_at_string():
    sb = ShadowBox()
    sb.update_reported("a", 1)
    status = sb.sync_status()
    assert status["synced_at"] is not None
    assert isinstance(status["synced_at"], str)
    # 可解析回 datetime
    datetime.fromisoformat(status["synced_at"])


def test_sync_status_synced_at_none_initially():
    sb = ShadowBox()
    status = sb.sync_status()
    assert status["synced_at"] is None


# --- 离线 / 补传跟踪 ---

def test_mark_offline_sets_offline_since():
    sb = ShadowBox()
    sb.mark_offline()
    assert sb.offline_since is not None


def test_mark_offline_idempotent():
    sb = ShadowBox()
    sb.mark_offline()
    first = sb.offline_since
    sb.mark_offline()
    assert sb.offline_since == first


def test_mark_online_clears_offline_since():
    sb = ShadowBox()
    sb.mark_offline()
    sb.mark_online()
    assert sb.offline_since is None


def test_incr_backlog_accumulates():
    sb = ShadowBox()
    sb.incr_backlog(3)
    sb.incr_backlog(2)
    assert sb.backlog_count == 5


def test_incr_backlog_default_one():
    sb = ShadowBox()
    sb.incr_backlog()
    assert sb.backlog_count == 1


# --- "边缘永不自己决定版本"原则 ---

def test_edge_never_modifies_desired_directly():
    """边缘只能通过 apply_desired 接收云端下发的 desired，
    绝不自行构造或修改 desired 内容。"""
    sb = ShadowBox()
    # 云端下发 desired
    sb.apply_desired({"model_version": "2.0"})
    # 边缘观察到需要拉取
    assert sb.needs_pull() == {"model_version": "2.0"}
    # 边缘拉取后更新 reported（而非改 desired）
    sb.update_reported("model_version", "2.0")
    # desired 保持云端下发的值不变
    assert sb.desired == {"model_version": "2.0"}
    assert sb.needs_pull() == {}


def test_edge_reports_observed_state_not_desired():
    """边缘 reported 反映实际观察到的状态，而非直接复制 desired。"""
    sb = ShadowBox()
    sb.apply_desired({"model_version": "2.0"})
    # 边缘实际拉取到的可能是 2.0，但也可能拉取失败保持 1.0
    sb.update_reported("model_version", "1.0")  # 拉取失败，保持旧版
    # delta 仍非空，边缘会继续尝试拉取
    assert sb.needs_pull() == {"model_version": "2.0"}
    assert sb.sync_status()["in_sync"] is False


def test_convergence_only_when_reported_matches_desired():
    """只有 reported 真正等于 desired 时才视为收敛。"""
    sb = ShadowBox()
    sb.apply_desired({"a": 1, "b": 2})
    sb.update_reported("a", 1)
    assert not sb.sync_status()["in_sync"]
    sb.update_reported("b", 2)
    assert sb.sync_status()["in_sync"]


# --- snapshot meta 增强 ---

def test_snapshot_meta_contains_enhanced_fields():
    sb = ShadowBox()
    sb.apply_desired({"a": 1})
    sb.update_reported("a", 1)
    snap = sb.snapshot()
    meta = snap["meta"]
    assert "synced_at" in meta
    assert "offline_since" in meta
    assert "offline_duration_s" in meta
    assert "backlog_count" in meta
    assert meta["synced_at"] is not None


# --- 兼容旧接口 ---

def test_set_desired_still_works():
    sb = ShadowBox()
    delta = sb.set_desired({"a": 1})
    assert delta == {"a": 1}


def test_set_reported_still_works():
    sb = ShadowBox()
    sb.set_desired({"a": 1})
    delta = sb.set_reported({"a": 1})
    assert delta == {}


def test_reported_attribute_accessible():
    sb = ShadowBox(reported={"x": 1})
    assert sb.reported == {"x": 1}


# ====================================================================
# NorthMqtt 影子 MQTT 集成（mock paho-mqtt）
# ====================================================================

@pytest.fixture
def mock_mqtt_client():
    """patch paho.mqtt.client.Client，返回 mock 实例。"""
    with patch("edge.cloud.mqtt_north.mqtt.Client") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


@pytest.fixture
def north(mock_mqtt_client):
    """构造 NorthMqtt 实例，其 _client 为 mock。"""
    return NorthMqtt(gw_id="gw-test", host="localhost", port=1883)


def test_north_init_configures_reconnect(mock_mqtt_client):
    NorthMqtt(gw_id="gw-1", host="h", port=1883)
    mock_mqtt_client.reconnect_delay_set.assert_called_once_with(min_delay=1, max_delay=120)


def test_north_online_initial_false(north):
    assert north.online is False


def test_north_on_connect_sets_online_and_subscribes(north):
    client = north._client
    north._on_connect(client, None, None, 0)
    assert north.online is True
    client.publish.assert_any_call(topics.status("gw-test"), "online", qos=1, retain=True)
    # 订阅关键 topic
    subscribed_topics = [call.args[0] for call in client.subscribe.call_args_list]
    assert topics.shadow_desired("gw-test") in subscribed_topics
    assert topics.event_ack("gw-test") in subscribed_topics
    assert topics.manifest("gw-test") in subscribed_topics


def test_north_on_connect_failure_keeps_offline(north):
    north._on_connect(north._client, None, None, 5)
    assert north.online is False


def test_north_on_disconnect_sets_offline(north):
    north.online = True
    north._on_disconnect(north._client, None, None, 0)
    assert north.online is False


def test_north_on_message_dispatches_handler(north):
    received: list[bytes] = []
    north.on("some/topic", lambda payload: received.append(payload))
    msg = MagicMock()
    msg.topic = "some/topic"
    msg.payload = b'{"k": 1}'
    north._on_message(north._client, None, msg)
    assert received == [b'{"k": 1}']


def test_north_on_message_no_handler_no_error(north):
    msg = MagicMock()
    msg.topic = "no/handler"
    msg.payload = b"x"
    # 不应抛异常
    north._on_message(north._client, None, msg)


def test_north_publish_json(north):
    north.publish_json("t/topic", {"a": 1}, qos=1, retain=True)
    call = north._client.publish.call_args
    assert call.args[0] == "t/topic"
    assert json.loads(call.args[1]) == {"a": 1}
    assert call.kwargs["qos"] == 1
    assert call.kwargs["retain"] is True


def _make_health() -> EdgeHealth:
    return EdgeHealth(
        gw_id="gw-test",
        ts=datetime.now(UTC),
        firmware_version="1.0.0",
        model_versions={"anomaly_det": "0.9.1"},
        config_version="v0001",
        inference_p95_ms=12.5,
        mem_used_mb=256,
        disk_free_mb=1024,
    )


def test_north_publish_health(north):
    health = _make_health()
    north.publish_health(health)
    call = north._client.publish.call_args
    assert call.args[0] == topics.health("gw-test")
    payload = json.loads(call.args[1])
    assert payload["gw_id"] == "gw-test"
    assert payload["firmware_version"] == "1.0.0"
    assert call.kwargs["retain"] is True


def test_north_publish_reported(north):
    reported = {"model_version": "0.9.1", "config_version": "v0001"}
    north.publish_reported(reported)
    call = north._client.publish.call_args
    assert call.args[0] == topics.shadow_reported("gw-test")
    assert json.loads(call.args[1]) == reported
    assert call.kwargs["retain"] is True


def _make_event() -> EdgeEvent:
    return EdgeEvent(
        event_id="evt-001",
        gw_id="gw-test",
        device_id_hash="hash123",
        seq=1,
        detected_at=datetime.now(UTC),
        rule_id="bearing_rms_high",
        severity="high",
        metric="vibration_rms",
        value=8.5,
        threshold=7.0,
    )


def test_north_publish_event(north):
    event = _make_event()
    north.publish_event(event)
    call = north._client.publish.call_args
    assert call.args[0] == topics.event("gw-test")
    payload = json.loads(call.args[1])
    assert payload["event_id"] == "evt-001"


def _make_telemetry() -> EdgeTelemetry:
    return EdgeTelemetry(
        gw_id="gw-test",
        device_id_hash="hash123",
        metric="vibration_rms",
        window_start=datetime.now(UTC),
        window_end=datetime.now(UTC),
        agg={"avg": 5.0, "max": 8.0},
        raw_rate_hz=1000.0,
    )


def test_north_publish_telemetry(north):
    tel = _make_telemetry()
    north.publish_telemetry(tel)
    call = north._client.publish.call_args
    assert call.args[0] == topics.telemetry("gw-test")
    payload = json.loads(call.args[1])
    assert payload["metric"] == "vibration_rms"
    assert call.kwargs["qos"] == 0


def test_north_publish_preliminary(north):
    event = _make_event()
    event.preliminary_diagnosis = "轴承早期磨损"
    event.preliminary_confidence = 0.85
    north.publish_preliminary(event)
    call = north._client.publish.call_args
    assert call.args[0] == topics.diagnosis_preliminary("gw-test")
    payload = json.loads(call.args[1])
    assert payload["event_id"] == "evt-001"
    assert payload["preliminary_diagnosis"] == "轴承早期磨损"
    assert payload["preliminary_confidence"] == 0.85


def test_north_on_registers_handler(north):
    handler = MagicMock()
    north.on("custom/topic", handler)
    assert north._handlers["custom/topic"] is handler


def test_north_start_connects_and_loops(north):
    north.start()
    north._client.connect_async.assert_called_once_with("localhost", 1883, keepalive=45)
    north._client.loop_start.assert_called_once()


def test_north_stop_publishes_offline_and_disconnects(north):
    north.stop()
    # 发布 offline 状态
    publish_calls = [c.args[0] for c in north._client.publish.call_args_list]
    assert topics.status("gw-test") in publish_calls
    north._client.loop_stop.assert_called_once()
    north._client.disconnect.assert_called_once()


# --- 端到端影子收敛流程（mock MQTT） ---

def test_shadow_convergence_flow_with_mqtt(north):
    """模拟完整收敛流程：云端下发 desired → 边缘检测 delta → 拉取 → reported 收敛 → 上报。"""
    sb = ShadowBox(reported={"model_version": "1.0", "config_version": "v0001"})

    # 1. 云端下发新 desired
    delta = sb.apply_desired({"model_version": "2.0", "config_version": "v0001"})
    assert delta == {"model_version": "2.0"}

    # 2. 边缘检测到需要拉取
    assert sb.needs_pull() == {"model_version": "2.0"}
    assert not sb.sync_status()["in_sync"]

    # 3. 边缘主动拉取并应用新模型
    sb.update_reported("model_version", "2.0")

    # 4. delta 归零，收敛完成
    assert sb.needs_pull() == {}
    assert sb.sync_status()["in_sync"]

    # 5. 上报 reported 到云端
    north.publish_reported(sb.reported)
    call = north._client.publish.call_args
    payload = json.loads(call.args[1])
    assert payload["model_version"] == "2.0"


def test_shadow_offline_backlog_flow(north):
    """离线期间积压事件，恢复后补传。"""
    sb = ShadowBox()

    # 进入离线
    sb.mark_offline()
    assert sb.sync_status()["offline_duration_s"] is not None

    # 离线期间积压 3 条事件
    for _ in range(3):
        sb.incr_backlog()
    assert sb.sync_status()["backlog_count"] == 3

    # 恢复在线
    sb.mark_online()
    assert sb.sync_status()["offline_duration_s"] is None
    # backlog_count 累计值保留（由上层清零）
    assert sb.backlog_count == 3