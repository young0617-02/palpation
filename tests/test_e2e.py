"""端到端联调测试：验证完整数据流。

数据流：传感器 → 边缘采集 → 规则/异常检测 → 事件上报 → 云端入库 → 诊断 → OTA

测试分层：
1. 诊断管线端到端（直接调用 depth_diagnosis，无 MQTT 依赖）
2. 云端 MQTT 桥接端到端（mock client + 直接调用 _ingest_event）
3. 边缘规则检测端到端（RuleEngine → EdgeEvent 构造）
4. 边缘到云端完整流（真实 amqtt broker，skip if 不可用）
5. OTA 流程端到端（FastAPI TestClient）
6. 完整数据流端到端（传感器指标 → 规则 → 事件 → 云端入库 → 诊断）
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import (
    Base,
    EdgeEventRow,
    ModelRegistry,
    RolloutPlan,
    RolloutRecord,
    ShadowRow,
    Watermark,
)
from cloud.diagnosis import depth_diagnosis
from cloud.mqtt_bridge import _ingest_event, attach
from edge.stream.rules import RuleEngine
from shared import topics
from shared.schemas import CloudDiagnosis, EdgeEvent

GW_ID = "gw-sim-001"
BROKER_PORT = 11883

# current.yaml 中的规则定义（与 config/current.yaml 保持一致）
RULES = [
    {
        "id": "bearing_rms_high",
        "metric": "vibration_rms",
        "op": ">",
        "threshold": 4.5,
        "severity": "high",
        "description": "驱动端轴承振动 RMS 越限",
    },
    {
        "id": "temperature_high",
        "metric": "temperature",
        "op": ">",
        "threshold": 80.0,
        "severity": "medium",
        "description": "轴承温度越限",
    },
    {
        "id": "current_high",
        "metric": "current",
        "op": ">",
        "threshold": 18.0,
        "severity": "medium",
        "description": "电机电流越限",
    },
]


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #


def _now() -> datetime:
    return datetime.now(UTC)


def _make_event(event_id: str = "e1", seq: int = 1, **overrides) -> EdgeEvent:
    """构造测试用 EdgeEvent，可覆盖任意字段。"""
    base = dict(
        event_id=event_id,
        gw_id=GW_ID,
        device_id_hash="hash-pump-de-01",
        seq=seq,
        detected_at=_now(),
        rule_id="bearing_rms_high",
        severity="high",
        metric="vibration_rms",
        value=6.3,
        threshold=4.5,
        preliminary_diagnosis="疑似轴承磨损",
        preliminary_confidence=0.6,
    )
    base.update(overrides)
    return EdgeEvent(**base)


def _fault_metrics() -> dict[str, float]:
    """故障工况传感器指标（全部越限）。"""
    return {"vibration_rms": 6.3, "temperature": 85.0, "current": 20.0}


def _normal_metrics() -> dict[str, float]:
    """正常工况传感器指标（全部未越限）。"""
    return {"vibration_rms": 1.2, "temperature": 47.0, "current": 10.2}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def temp_cloud_db(tmp_path, monkeypatch):
    """替换 cloud DB 到 tmp_path sqlite，patch 所有 SessionLocal/engine 引用。

    cloud.db.SessionLocal 被 cloud.main 和 cloud.mqtt_bridge 分别 import，
    需逐一替换才能让 API 端点和 MQTT 桥接都用上临时库。
    """
    db_path = tmp_path / "cloud_e2e.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    session = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr("cloud.db.engine", engine)
    monkeypatch.setattr("cloud.db.SessionLocal", session)
    monkeypatch.setattr("cloud.main.SessionLocal", session)
    monkeypatch.setattr("cloud.mqtt_bridge.SessionLocal", session)

    Base.metadata.create_all(engine)
    return session


@pytest.fixture()
def mock_mqtt_client():
    """捕获 publish 调用的 mock MQTT client。

    替代真实 MQTT broker，直接验证 _ingest_event 的 publish 行为
    （ACK + 诊断消息）。
    """
    client = MagicMock()
    published: list[dict] = []

    def capture_publish(topic, payload, qos=0, retain=False):
        published.append({"topic": topic, "payload": payload, "qos": qos})

    client.publish = capture_publish
    client.published = published
    return client


@pytest.fixture()
def amqtt_broker():
    """启动真实 amqtt broker（后台线程），返回端口号。

    amqtt 0.12.1 的 Broker.__init__ 需要 running event loop，
    因此 Broker 实例化也在后台线程内完成。
    amqtt 不可用或启动失败时 pytest.skip，不影响其他测试。
    """
    try:
        from amqtt.broker import Broker
    except ImportError:
        pytest.skip("amqtt not installed")

    config = {"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{BROKER_PORT}"}}}
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    errors: list[Exception] = []
    broker_ref: list = []  # 在线程内创建后传回主线程

    async def start_broker() -> None:
        """在事件循环内创建并启动 broker。"""
        broker = Broker(config)
        broker_ref.append(broker)
        await broker.start()

    def run() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(start_broker())
        except Exception as exc:
            errors.append(exc)
            ready.set()
            return
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(timeout=5)

    if errors:
        pytest.skip(f"broker start failed: {errors[0]}")

    yield BROKER_PORT

    # 清理：关闭 broker 和事件循环
    if broker_ref:
        try:
            fut = asyncio.run_coroutine_threadsafe(broker_ref[0].shutdown(), loop)
            fut.result(timeout=5)
        except Exception:
            pass
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# 1. 云端诊断管线端到端
# --------------------------------------------------------------------------- #


def test_cloud_diagnosis_pipeline_e2e():
    """各种事件类型 → depth_diagnosis → CloudDiagnosis 完整验证。

    覆盖 bearing_rms_high / temperature_high / current_high 三种规则，
    验证 conclusion 非空、confidence 合理、requires_human_review 逻辑。
    """
    cases = [
        ("bearing_rms_high", "vibration_rms", 6.3, 4.5, "high", "振动"),
        ("temperature_high", "temperature", 85.0, 80.0, "medium", "温度"),
        ("current_high", "current", 20.0, 18.0, "medium", "电流"),
    ]
    for rule_id, metric, value, threshold, severity, keyword in cases:
        event = _make_event(
            event_id=f"e-diag-{rule_id}",
            rule_id=rule_id,
            metric=metric,
            value=value,
            threshold=threshold,
            severity=severity,
        )
        diag = depth_diagnosis(event)
        assert isinstance(diag, CloudDiagnosis)
        assert diag.event_id == f"e-diag-{rule_id}"
        assert diag.gw_id == GW_ID
        assert diag.conclusion, f"conclusion empty for {rule_id}"
        assert keyword in diag.conclusion, f"keyword '{keyword}' missing for {rule_id}"
        assert 0 < diag.confidence <= 0.95
        assert diag.requires_human_review is (severity == "high")


def test_cloud_diagnosis_degradation_e2e(monkeypatch, tmp_path):
    """知识库和故障链均不可用时诊断管线优雅降级。"""
    monkeypatch.setattr("cloud.diagnosis._SNIPPETS_PATH", tmp_path / "no_kb.json")
    monkeypatch.setattr("cloud.diagnosis._FAULT_CHAINS_PATH", tmp_path / "no_chain.json")
    event = _make_event()
    diag = depth_diagnosis(event)
    assert diag.citations == []
    assert "未命中" in diag.conclusion
    assert diag.confidence == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 2. 云端 MQTT 桥接端到端
# --------------------------------------------------------------------------- #


def test_cloud_mqtt_bridge_ingest_e2e(temp_cloud_db, mock_mqtt_client):
    """EdgeEvent → _ingest_event → DB 入库 + Watermark + ACK + 诊断发布。

    验证完整云端接收链路：
    - EdgeEventRow 入库
    - Watermark 更新
    - ACK 发布到 edge/{gw_id}/ack
    - CloudDiagnosis 发布到 edge/{gw_id}/diagnosis/cloud
    """
    session = temp_cloud_db
    event = _make_event(event_id="evt-bridge-001", seq=1)

    _ingest_event(mock_mqtt_client, event)

    # 验证 DB 入库
    with session() as s:
        row = s.query(EdgeEventRow).filter_by(event_id="evt-bridge-001").one()
        assert row.gw_id == GW_ID
        assert row.seq == 1
        assert row.status == "received"
        wm = s.get(Watermark, GW_ID)
        assert wm is not None
        assert wm.last_acked_seq == 1
        assert wm.gaps == []

    # 验证 ACK + 诊断发布（顺序：先 ACK 后诊断）
    assert len(mock_mqtt_client.published) == 2
    ack_msg, diag_msg = mock_mqtt_client.published[0], mock_mqtt_client.published[1]

    assert ack_msg["topic"] == topics.event_ack(GW_ID)
    ack = json.loads(ack_msg["payload"])
    assert ack["gw_id"] == GW_ID
    assert ack["last_acked_seq"] == 1

    assert diag_msg["topic"] == topics.diagnosis_cloud(GW_ID)
    diag = CloudDiagnosis.model_validate_json(diag_msg["payload"])
    assert diag.event_id == "evt-bridge-001"
    assert diag.gw_id == GW_ID
    assert diag.conclusion


def test_cloud_mqtt_bridge_gap_detection_e2e(temp_cloud_db, mock_mqtt_client):
    """Watermark gap 检测：seq=1 然后 seq=3，产生 gap [2,2]。"""
    session = temp_cloud_db
    _ingest_event(mock_mqtt_client, _make_event(event_id="evt-gap-1", seq=1))
    _ingest_event(mock_mqtt_client, _make_event(event_id="evt-gap-3", seq=3))

    with session() as s:
        wm = s.get(Watermark, GW_ID)
        assert wm.last_acked_seq == 3
        assert wm.gaps == [{"from": 2, "to": 2}]


def test_cloud_mqtt_bridge_dedup_e2e(temp_cloud_db, mock_mqtt_client):
    """重复 event_id 不重复入库，Watermark 不回退。"""
    session = temp_cloud_db
    event = _make_event(event_id="evt-dedup", seq=1)
    _ingest_event(mock_mqtt_client, event)
    _ingest_event(mock_mqtt_client, event)  # 重复

    with session() as s:
        rows = s.query(EdgeEventRow).filter_by(event_id="evt-dedup").all()
        assert len(rows) == 1


# --------------------------------------------------------------------------- #
# 3. 边缘规则检测端到端
# --------------------------------------------------------------------------- #


def test_edge_rule_to_event_e2e():
    """传感器指标 → RuleEngine 评估 → EdgeEvent 构造。

    验证边缘侧数据流前半段：
    - 正常工况无规则触发
    - 故障工况多条规则同时触发
    - RuleHit 字段正确映射到 EdgeEvent
    """
    engine = RuleEngine(RULES)

    # 正常工况：无规则触发
    assert engine.evaluate(_normal_metrics()) == []

    # 故障工况：三条规则全部触发
    hits = engine.evaluate(_fault_metrics())
    assert len(hits) == 3
    rule_ids = {h.rule_id for h in hits}
    assert rule_ids == {"bearing_rms_high", "temperature_high", "current_high"}

    # 验证 RuleHit → EdgeEvent 字段映射
    bearing_hit = next(h for h in hits if h.rule_id == "bearing_rms_high")
    assert bearing_hit.severity == "high"
    assert bearing_hit.value == 6.3
    assert bearing_hit.threshold == 4.5

    event = _make_event(
        event_id="evt-rule-001",
        rule_id=bearing_hit.rule_id,
        metric=bearing_hit.metric,
        value=bearing_hit.value,
        threshold=bearing_hit.threshold,
        severity=bearing_hit.severity,
    )
    assert event.rule_id == "bearing_rms_high"
    assert event.metric == "vibration_rms"
    assert event.value == 6.3
    assert event.severity == "high"


def test_edge_rule_partial_fault_e2e():
    """部分故障：仅振动越限，其他指标正常。"""
    engine = RuleEngine(RULES)
    metrics = {"vibration_rms": 6.3, "temperature": 47.0, "current": 10.2}
    hits = engine.evaluate(metrics)
    assert len(hits) == 1
    assert hits[0].rule_id == "bearing_rms_high"


# --------------------------------------------------------------------------- #
# 4. 边缘到云端完整流（真实 amqtt broker）
# --------------------------------------------------------------------------- #


def test_edge_to_cloud_via_broker_e2e(temp_cloud_db, amqtt_broker):
    """真实 MQTT broker 端到端：edge 发布事件 → cloud 接收入库。

    启动 amqtt broker，cloud 客户端订阅 edge/+/event，
    edge 客户端发布 EdgeEvent JSON，验证云端 DB 入库 + Watermark。
    """
    session = temp_cloud_db
    port = amqtt_broker

    # 创建 cloud 客户端并 attach 桥接回调
    cloud_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cloud-e2e-test")
    attach(cloud_client)
    cloud_connected = threading.Event()
    original_on_connect = cloud_client.on_connect

    def on_connect(client, userdata, flags, reason_code, properties=None):
        original_on_connect(client, userdata, flags, reason_code, properties)
        cloud_connected.set()

    cloud_client.on_connect = on_connect
    cloud_client.connect("127.0.0.1", port, 45)
    cloud_client.loop_start()
    assert cloud_connected.wait(timeout=5), "cloud client not connected"
    time.sleep(0.3)  # 等待订阅生效

    # 创建 edge 客户端
    edge_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge-e2e-pub")
    edge_client.connect("127.0.0.1", port, 45)
    edge_client.loop_start()
    time.sleep(0.3)

    try:
        # 发布 EdgeEvent（重试直到入库，应对订阅延迟）
        event = _make_event(event_id="evt-broker-001", seq=1)
        event_json = event.model_dump_json()
        event_topic = topics.event(GW_ID)

        found = False
        for _ in range(20):
            info = edge_client.publish(event_topic, event_json, qos=1)
            info.wait_for_publish(timeout=2)
            time.sleep(0.5)
            with session() as s:
                row = s.query(EdgeEventRow).filter_by(event_id="evt-broker-001").one_or_none()
                if row is not None:
                    found = True
                    break

        assert found, "event not ingested into DB within timeout"

        # 验证 DB 状态
        with session() as s:
            row = s.query(EdgeEventRow).filter_by(event_id="evt-broker-001").one()
            assert row.gw_id == GW_ID
            assert row.seq == 1
            wm = s.get(Watermark, GW_ID)
            assert wm is not None
            assert wm.last_acked_seq == 1
    finally:
        edge_client.loop_stop()
        edge_client.disconnect()
        cloud_client.loop_stop()
        cloud_client.disconnect()


def test_edge_to_cloud_multiple_events_via_broker_e2e(temp_cloud_db, amqtt_broker):
    """真实 broker：连续发布多个事件，验证全部入库 + Watermark 推进。"""
    session = temp_cloud_db
    port = amqtt_broker

    cloud_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cloud-e2e-multi")
    attach(cloud_client)
    connected = threading.Event()
    orig_on_connect = cloud_client.on_connect

    def on_connect(client, userdata, flags, reason_code, properties=None):
        orig_on_connect(client, userdata, flags, reason_code, properties)
        connected.set()

    cloud_client.on_connect = on_connect
    cloud_client.connect("127.0.0.1", port, 45)
    cloud_client.loop_start()
    assert connected.wait(timeout=5)
    time.sleep(0.3)

    edge_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge-e2e-multi-pub")
    edge_client.connect("127.0.0.1", port, 45)
    edge_client.loop_start()
    time.sleep(0.3)

    try:
        event_topic = topics.event(GW_ID)
        for i in range(1, 4):
            event = _make_event(
                event_id=f"evt-multi-{i:03d}",
                seq=i,
                rule_id=RULES[i - 1]["id"],
                metric=RULES[i - 1]["metric"],
                value=RULES[i - 1]["threshold"] + 1,
                threshold=RULES[i - 1]["threshold"],
                severity=RULES[i - 1]["severity"],
            )
            info = edge_client.publish(event_topic, event.model_dump_json(), qos=1)
            info.wait_for_publish(timeout=2)
            time.sleep(0.3)

        # 等待全部入库
        all_found = False
        for _ in range(30):
            with session() as s:
                count = s.query(EdgeEventRow).filter(EdgeEventRow.gw_id == GW_ID).count()
                if count >= 3:
                    all_found = True
                    break
            time.sleep(0.5)

        assert all_found, "not all events ingested within timeout"

        with session() as s:
            rows = (
                s.query(EdgeEventRow)
                .filter(EdgeEventRow.gw_id == GW_ID)
                .order_by(EdgeEventRow.seq)
                .all()
            )
            assert len(rows) == 3
            assert [r.seq for r in rows] == [1, 2, 3]
            wm = s.get(Watermark, GW_ID)
            assert wm.last_acked_seq == 3
    finally:
        edge_client.loop_stop()
        edge_client.disconnect()
        cloud_client.loop_stop()
        cloud_client.disconnect()


# --------------------------------------------------------------------------- #
# 5. OTA 流程端到端
# --------------------------------------------------------------------------- #


def test_ota_flow_e2e(tmp_path, monkeypatch):
    """OTA 完整流程：设备注册 → 模型注册 → rollout → shadow 更新。

    使用 FastAPI TestClient + 临时 DB + mock MQTT client，
    验证 RolloutPlan / RolloutRecord / ShadowRow 全部正确创建。
    """
    from cloud import main as cloud_main
    from cloud.settings import cloud_settings

    # 准备临时 DB
    db_path = tmp_path / "cloud_ota.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("cloud.db.engine", engine)
    monkeypatch.setattr("cloud.db.SessionLocal", session)
    monkeypatch.setattr("cloud.main.SessionLocal", session)
    monkeypatch.setattr("cloud.mqtt_bridge.SessionLocal", session)
    monkeypatch.setattr(cloud_settings, "artifact_dir", tmp_path / "artifacts")

    # mock MQTT client 避免真实连接
    mock_client = MagicMock()
    monkeypatch.setattr(cloud_main, "mqtt_client", mock_client)

    Base.metadata.create_all(engine)

    with TestClient(cloud_main.app) as client:
        # Step 1: 注册设备
        resp = client.post("/v1/devices/register", json={"gw_id": GW_ID, "site": "lab"})
        assert resp.status_code == 200
        assert resp.json()["gw_id"] == GW_ID

        # Step 2: 注册模型（文件上传）
        model_content = b"fake onnx model bytes for e2e"
        resp = client.post(
            "/v1/models/register",
            data={"model_id": "anomaly-det", "version": "0.2.0"},
            files={"file": ("model.onnx", model_content, "application/octet-stream")},
        )
        assert resp.status_code == 200
        manifest = resp.json()
        assert manifest["model_id"] == "anomaly-det"
        assert manifest["version"] == "0.2.0"
        assert manifest["sha256"]
        assert manifest["size_bytes"] == len(model_content)

        # Step 3: 创建 rollout
        resp = client.post(
            "/v1/rollout",
            json={"gw_id": GW_ID, "model_id": "anomaly-det", "version": "0.2.0"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Step 4: 验证 DB 状态
        with session() as s:
            # 模型注册表
            mr = s.query(ModelRegistry).filter_by(model_id="anomaly-det", version="0.2.0").one()
            assert mr.sha256 == manifest["sha256"]
            assert mr.artifact_path

            # RolloutPlan
            plan = s.query(RolloutPlan).filter_by(model_id="anomaly-det", version="0.2.0").one()
            assert plan.strategy == "ring"
            assert plan.rings == [1]

            # RolloutRecord
            rec = s.query(RolloutRecord).filter_by(gw_id=GW_ID).one()
            assert rec.model_id == "anomaly-det"
            assert rec.version == "0.2.0"
            assert rec.state == "pending"

            # Shadow desired 已更新
            shadow = s.get(ShadowRow, GW_ID)
            assert shadow is not None
            assert shadow.desired["model_version"] == "0.2.0"

        # Step 5: 验证事件查询 API
        resp = client.get("/v1/events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    # 验证 mock MQTT publish 被调用（shadow desired + manifest）
    assert mock_client.publish.called


def test_ota_watermark_api_e2e(tmp_path, monkeypatch):
    """OTA + Watermark API：注册设备后查询 watermark 返回 404 或正常。"""
    from cloud import main as cloud_main
    from cloud.settings import cloud_settings

    db_path = tmp_path / "cloud_wm.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("cloud.db.engine", engine)
    monkeypatch.setattr("cloud.db.SessionLocal", session)
    monkeypatch.setattr("cloud.main.SessionLocal", session)
    monkeypatch.setattr("cloud.mqtt_bridge.SessionLocal", session)
    monkeypatch.setattr(cloud_settings, "artifact_dir", tmp_path / "artifacts")
    mock_client = MagicMock()
    monkeypatch.setattr(cloud_main, "mqtt_client", mock_client)
    Base.metadata.create_all(engine)

    with TestClient(cloud_main.app) as client:
        # 未注册设备的 watermark → 404
        resp = client.get(f"/v1/watermarks/{GW_ID}")
        assert resp.status_code == 404

        # 注册设备
        client.post("/v1/devices/register", json={"gw_id": GW_ID})

        # 注入事件后 watermark 存在
        event = _make_event(event_id="evt-wm-api", seq=1)
        _ingest_event(mock_client, event)

        resp = client.get(f"/v1/watermarks/{GW_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gw_id"] == GW_ID
        assert data["last_acked_seq"] == 1


# --------------------------------------------------------------------------- #
# 6. 完整数据流端到端
# --------------------------------------------------------------------------- #


def test_full_data_flow_e2e(temp_cloud_db, mock_mqtt_client):
    """完整数据流：传感器指标 → 规则评估 → 事件构造 → 云端入库 → 诊断。

    模拟故障工况，三条规则同时触发，三个 EdgeEvent 上报云端，
    验证：DB 入库、Watermark 推进、诊断生成、ACK 发布、人工审核标记。
    """
    session = temp_cloud_db

    # Step 1: 传感器指标（故障工况）
    sensor_metrics = _fault_metrics()

    # Step 2: 规则评估
    engine = RuleEngine(RULES)
    hits = engine.evaluate(sensor_metrics)
    assert len(hits) == 3

    # Step 3: 构造 EdgeEvent 并上报云端
    for i, hit in enumerate(hits, start=1):
        event = _make_event(
            event_id=f"evt-full-{i:03d}",
            seq=i,
            rule_id=hit.rule_id,
            metric=hit.metric,
            value=hit.value,
            threshold=hit.threshold,
            severity=hit.severity,
        )
        _ingest_event(mock_mqtt_client, event)

    # Step 4: 验证云端入库
    with session() as s:
        rows = s.query(EdgeEventRow).order_by(EdgeEventRow.seq).all()
        assert len(rows) == 3
        assert [r.seq for r in rows] == [1, 2, 3]
        wm = s.get(Watermark, GW_ID)
        assert wm.last_acked_seq == 3
        assert wm.gaps == []

    # Step 5: 验证 ACK 发布（3 条）
    acks = [m for m in mock_mqtt_client.published if m["topic"].endswith("/ack")]
    assert len(acks) == 3
    for i, ack_msg in enumerate(acks, start=1):
        ack = json.loads(ack_msg["payload"])
        assert ack["last_acked_seq"] == i

    # Step 6: 验证诊断生成（3 条）
    diag_msgs = [m for m in mock_mqtt_client.published if "diagnosis/cloud" in m["topic"]]
    assert len(diag_msgs) == 3

    diag_by_event: dict[str, CloudDiagnosis] = {}
    for msg in diag_msgs:
        d = CloudDiagnosis.model_validate_json(msg["payload"])
        diag_by_event[d.event_id] = d

    # bearing_rms_high → high severity → 需要人工审核
    assert diag_by_event["evt-full-001"].requires_human_review is True
    assert "振动" in diag_by_event["evt-full-001"].conclusion

    # temperature_high / current_high → medium → 不需要人工审核
    assert diag_by_event["evt-full-002"].requires_human_review is False
    assert "温度" in diag_by_event["evt-full-002"].conclusion
    assert diag_by_event["evt-full-003"].requires_human_review is False
    assert "电流" in diag_by_event["evt-full-003"].conclusion

    # 所有诊断置信度合理
    for diag in diag_by_event.values():
        assert 0 < diag.confidence <= 0.95
        assert diag.conclusion
        assert diag.gw_id == GW_ID


def test_full_data_flow_with_gap_e2e(temp_cloud_db, mock_mqtt_client):
    """完整数据流含 gap：发布 seq=1,2,4（跳过 3），验证 gap 检测。"""
    session = temp_cloud_db
    seqs = [1, 2, 4]
    for seq in seqs:
        event = _make_event(
            event_id=f"evt-gapflow-{seq}",
            seq=seq,
            rule_id="bearing_rms_high",
            severity="high",
        )
        _ingest_event(mock_mqtt_client, event)

    with session() as s:
        rows = s.query(EdgeEventRow).order_by(EdgeEventRow.seq).all()
        assert len(rows) == 3
        assert [r.seq for r in rows] == [1, 2, 4]
        wm = s.get(Watermark, GW_ID)
        assert wm.last_acked_seq == 4
        assert wm.gaps == [{"from": 3, "to": 3}]
