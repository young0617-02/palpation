"""真实端到端联调测试：真实 mosquitto broker + 真实 LLM 模型。

前置条件：
- mosquitto 2.0.18 运行于 127.0.0.1:1883，允许匿名连接
- llama-cpp-python 0.3.36 已安装（Vulkan 版本）
- 模型文件 models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf 存在

任一前置条件不满足时对应测试 skip，不影响其他测试。
数据流：传感器 → 边缘规则评估 → EdgeEvent → MQTT → 云端入库 → 诊断
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import paho.mqtt.client as mqtt
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import Base, EdgeEventRow, Watermark
from cloud.diagnosis import depth_diagnosis
from cloud.mqtt_bridge import attach
from edge.stream.rules import RuleEngine
from shared import topics
from shared.schemas import CloudDiagnosis, EdgeEvent

GW_ID = "gw-real-001"
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
MODEL_PATH = "models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"

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


def _make_event(event_id: str = "e-real-1", seq: int = 1, **overrides) -> EdgeEvent:
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


def _wait_for_event(session, event_id: str, timeout: float = 10.0) -> bool:
    """轮询 DB 等待指定 event_id 入库。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with session() as s:
            if s.query(EdgeEventRow).filter_by(event_id=event_id).one_or_none() is not None:
                return True
        time.sleep(0.3)
    return False


def _wait_for_count(session, gw_id: str, count: int, timeout: float = 15.0) -> bool:
    """轮询 DB 等待指定 gw_id 的事件数达到 count。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with session() as s:
            if s.query(EdgeEventRow).filter(EdgeEventRow.gw_id == gw_id).count() >= count:
                return True
        time.sleep(0.3)
    return False


def _connect_cloud_client(client_id: str):
    """创建 cloud MQTT client，attach 桥接回调并连接到真实 broker。

    返回 (client, connected_event)，connected_event 在 on_connect 触发后 set。
    """
    cloud_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    attach(cloud_client)
    connected = threading.Event()
    orig_on_connect = cloud_client.on_connect

    def on_connect(client, userdata, flags, reason_code, properties=None):
        orig_on_connect(client, userdata, flags, reason_code, properties)
        connected.set()

    cloud_client.on_connect = on_connect
    cloud_client.connect(BROKER_HOST, BROKER_PORT, 10)
    cloud_client.loop_start()
    return cloud_client, connected


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def temp_cloud_db(tmp_path, monkeypatch):
    """替换 cloud DB 到 tmp_path sqlite，patch 所有 SessionLocal/engine 引用。

    cloud.db.SessionLocal 被 cloud.main 和 cloud.mqtt_bridge 分别 import，
    需逐一替换才能让 API 端点和 MQTT 桥接都用上临时库。
    """
    db_path = tmp_path / "cloud_real_e2e.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    session = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr("cloud.db.engine", engine)
    monkeypatch.setattr("cloud.db.SessionLocal", session)
    monkeypatch.setattr("cloud.main.SessionLocal", session)
    monkeypatch.setattr("cloud.mqtt_bridge.SessionLocal", session)

    Base.metadata.create_all(engine)
    return session


@pytest.fixture()
def real_mqtt_client():
    """连接到真实 mosquitto broker，失败则 skip。"""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test-real-e2e")
    try:
        client.connect(BROKER_HOST, BROKER_PORT, 10)
    except Exception:
        pytest.skip("mosquitto not running on 127.0.0.1:1883")
    client.loop_start()
    time.sleep(0.5)
    yield client
    client.loop_stop()
    client.disconnect()


@pytest.fixture(scope="session")
def real_llm():
    """加载真实 LLM 模型（session 级复用避免重复加载），失败则 skip。"""
    from edge.agent.llm_llamacpp import EdgeLLM

    if not Path(MODEL_PATH).exists():
        pytest.skip("model file not found")
    llm = EdgeLLM(MODEL_PATH, max_tokens=128, num_ctx=512)
    if not llm.available:
        pytest.skip("LLM not available")
    return llm


# --------------------------------------------------------------------------- #
# 1. 真实 MQTT 往返
# --------------------------------------------------------------------------- #


def test_real_mqtt_round_trip(temp_cloud_db, real_mqtt_client):
    """真实 MQTT 往返：发布 EdgeEvent → cloud 接收 → DB 入库 + ACK + 诊断。

    验证完整云端接收链路：
    - EdgeEvent 通过真实 mosquitto broker 送达 cloud
    - EdgeEventRow 入库 + Watermark 更新
    - ACK 发布到 edge/{gw_id}/ack 并被收到
    - CloudDiagnosis 发布到 edge/{gw_id}/diagnosis/cloud 并被收到
    """
    session = temp_cloud_db
    event = _make_event(event_id="evt-real-rt-001", seq=1)

    # 创建 cloud client 并 attach 桥接回调
    cloud_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cloud-real-rt")
    attach(cloud_client)

    ack_payloads: list[str] = []
    diag_payloads: list[str] = []
    ack_received = threading.Event()
    diag_received = threading.Event()
    connected = threading.Event()

    orig_on_connect = cloud_client.on_connect

    def on_connect(client, userdata, flags, reason_code, properties=None):
        orig_on_connect(client, userdata, flags, reason_code, properties)
        client.subscribe(topics.event_ack(GW_ID), qos=1)
        client.subscribe(topics.diagnosis_cloud(GW_ID), qos=1)
        connected.set()

    orig_on_message = cloud_client.on_message

    def on_message(client, userdata, msg):
        orig_on_message(client, userdata, msg)
        if msg.topic == topics.event_ack(GW_ID):
            ack_payloads.append(msg.payload.decode("utf-8"))
            ack_received.set()
        elif msg.topic == topics.diagnosis_cloud(GW_ID):
            diag_payloads.append(msg.payload.decode("utf-8"))
            diag_received.set()

    cloud_client.on_connect = on_connect
    cloud_client.on_message = on_message
    cloud_client.connect(BROKER_HOST, BROKER_PORT, 10)
    cloud_client.loop_start()
    assert connected.wait(timeout=5), "cloud client not connected"
    time.sleep(0.5)  # 等待订阅生效

    try:
        # 发布 EdgeEvent
        info = real_mqtt_client.publish(topics.event(GW_ID), event.model_dump_json(), qos=1)
        info.wait_for_publish(timeout=5)

        # 等待 DB 入库
        assert _wait_for_event(session, "evt-real-rt-001"), "event not ingested into DB"

        # 等待 ACK 和诊断消息
        assert ack_received.wait(timeout=10), "ACK not received"
        assert diag_received.wait(timeout=10), "diagnosis not received"

        # 验证 DB 入库
        with session() as s:
            row = s.query(EdgeEventRow).filter_by(event_id="evt-real-rt-001").one()
            assert row.gw_id == GW_ID
            assert row.seq == 1
            assert row.status == "received"
            wm = s.get(Watermark, GW_ID)
            assert wm is not None
            assert wm.last_acked_seq == 1
            assert wm.gaps == []

        # 验证 ACK 内容
        ack = json.loads(ack_payloads[0])
        assert ack["gw_id"] == GW_ID
        assert ack["last_acked_seq"] == 1

        # 验证诊断内容
        diag = CloudDiagnosis.model_validate_json(diag_payloads[0])
        assert diag.event_id == "evt-real-rt-001"
        assert diag.gw_id == GW_ID
        assert diag.conclusion
        assert 0 < diag.confidence <= 0.95
        assert diag.requires_human_review is True  # high severity
    finally:
        cloud_client.loop_stop()
        cloud_client.disconnect()


# --------------------------------------------------------------------------- #
# 2. 真实 LLM 推理
# --------------------------------------------------------------------------- #


def test_real_llm_inference(real_llm):
    """真实 LLM 推理：加载模型并执行推理，验证推理指标。

    EdgeLLM.infer 返回 dict（成功解析 JSON）或 None（模型未输出有效 JSON）。
    无论结果如何，推理确实执行了：json_total >= 1, latencies_ms 非空, p95_ms > 0。
    """
    llm = real_llm
    assert llm.available is True

    prompt = "设备振动RMS值6.3g超过阈值4.5g，请给出诊断结论。"
    result = llm.infer(prompt)

    # 推理确实执行了
    assert llm.json_total >= 1
    assert len(llm.latencies_ms) >= 1
    assert llm.p95_ms() > 0

    # result 可能为 None（模型未输出有效 JSON）或 dict
    if result is not None:
        assert isinstance(result, dict)


# --------------------------------------------------------------------------- #
# 3. 真实完整管线
# --------------------------------------------------------------------------- #


def test_real_full_pipeline(temp_cloud_db, real_mqtt_client, real_llm):
    """真实完整管线：传感器 → 规则评估 → LLM 初判 → 事件 → MQTT → 云端入库 → 诊断。

    用真实 mosquitto broker 传输事件，用真实 LLM 生成 preliminary_diagnosis，
    验证从传感器指标到云端诊断的完整数据流。
    """
    session = temp_cloud_db

    # Step 1: 传感器指标（故障工况）
    sensor_metrics = {"vibration_rms": 6.3, "temperature": 85.0, "current": 20.0}

    # Step 2: 规则评估
    engine = RuleEngine(RULES)
    hits = engine.evaluate(sensor_metrics)
    assert len(hits) == 3

    # Step 3: 用 LLM 生成 preliminary_diagnosis
    llm_result = real_llm.infer("驱动端振动RMS 6.3g，阈值4.5g，判断故障类型。")
    prelim = llm_result.get("diagnosis", "疑似轴承磨损") if llm_result else "疑似轴承磨损"
    raw_conf = llm_result.get("confidence") if llm_result else None
    prelim_conf = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.6

    # Step 4: 构造 EdgeEvent（取 bearing_rms_high hit）
    bearing_hit = next(h for h in hits if h.rule_id == "bearing_rms_high")
    event = _make_event(
        event_id="evt-real-pipeline-001",
        seq=1,
        rule_id=bearing_hit.rule_id,
        metric=bearing_hit.metric,
        value=bearing_hit.value,
        threshold=bearing_hit.threshold,
        severity=bearing_hit.severity,
        preliminary_diagnosis=prelim,
        preliminary_confidence=prelim_conf,
    )

    # Step 5: 创建 cloud client，attach，发布事件
    cloud_client, connected = _connect_cloud_client("cloud-real-pipeline")
    assert connected.wait(timeout=5), "cloud client not connected"
    time.sleep(0.5)

    try:
        # 发布事件到真实 broker
        info = real_mqtt_client.publish(topics.event(GW_ID), event.model_dump_json(), qos=1)
        info.wait_for_publish(timeout=5)

        # 等待云端入库
        assert _wait_for_event(session, "evt-real-pipeline-001"), "event not ingested"

        # Step 6: 验证 DB 入库 + Watermark
        with session() as s:
            row = s.query(EdgeEventRow).filter_by(event_id="evt-real-pipeline-001").one()
            assert row.gw_id == GW_ID
            assert row.seq == 1
            wm = s.get(Watermark, GW_ID)
            assert wm is not None
            assert wm.last_acked_seq == 1

        # Step 7: 验证诊断输出
        diag = depth_diagnosis(event)
        assert isinstance(diag, CloudDiagnosis)
        assert diag.event_id == "evt-real-pipeline-001"
        assert diag.conclusion
        assert "振动" in diag.conclusion
        assert 0 < diag.confidence <= 0.95
        assert diag.requires_human_review is True  # high severity
    finally:
        cloud_client.loop_stop()
        cloud_client.disconnect()


# --------------------------------------------------------------------------- #
# 4. 真实云端诊断（含 LLM）
# --------------------------------------------------------------------------- #


def test_real_cloud_diagnosis_with_llm(real_llm):
    """真实云端诊断（含 LLM）：LLM 生成初判 → depth_diagnosis → CloudDiagnosis。

    验证 depth_diagnosis 输出包含 conclusion、confidence、citations，
    requires_human_review 对 high severity 为 True，对 medium 为 False。
    """
    # 用 LLM 生成 preliminary_diagnosis
    llm_result = real_llm.infer("振动RMS 6.3g超阈值4.5g，给出诊断结论。")
    prelim = llm_result.get("diagnosis", "疑似轴承磨损") if llm_result else "疑似轴承磨损"

    # 构造 high severity EdgeEvent
    event = _make_event(
        event_id="evt-real-diag-001",
        severity="high",
        preliminary_diagnosis=prelim,
    )

    # 调用 depth_diagnosis
    diag = depth_diagnosis(event)

    # 验证 CloudDiagnosis 基本字段
    assert isinstance(diag, CloudDiagnosis)
    assert diag.event_id == "evt-real-diag-001"
    assert diag.gw_id == GW_ID
    assert diag.conclusion
    assert "振动" in diag.conclusion
    assert 0 < diag.confidence <= 0.95
    assert diag.requires_human_review is True  # high severity

    # 验证 medium severity 不需要人工审核
    medium_event = _make_event(
        event_id="evt-real-diag-002",
        rule_id="temperature_high",
        metric="temperature",
        value=85.0,
        threshold=80.0,
        severity="medium",
    )
    medium_diag = depth_diagnosis(medium_event)
    assert medium_diag.event_id == "evt-real-diag-002"
    assert medium_diag.conclusion
    assert "温度" in medium_diag.conclusion
    assert medium_diag.requires_human_review is False  # medium severity


# --------------------------------------------------------------------------- #
# 5. 多事件真实 MQTT
# --------------------------------------------------------------------------- #


def test_real_mqtt_multiple_events(temp_cloud_db, real_mqtt_client):
    """多事件真实 MQTT：连续发布 3 个 EdgeEvent，验证全部入库 + Watermark 推进。

    发布 seq=1,2,3 三个事件（不同规则触发），验证：
    - 全部 3 条入库
    - Watermark.last_acked_seq == 3
    - 无 gaps
    """
    session = temp_cloud_db

    cloud_client, connected = _connect_cloud_client("cloud-real-multi")
    assert connected.wait(timeout=5), "cloud client not connected"
    time.sleep(0.5)

    try:
        event_topic = topics.event(GW_ID)
        for i in range(1, 4):
            event = _make_event(
                event_id=f"evt-real-multi-{i:03d}",
                seq=i,
                rule_id=RULES[i - 1]["id"],
                metric=RULES[i - 1]["metric"],
                value=RULES[i - 1]["threshold"] + 1,
                threshold=RULES[i - 1]["threshold"],
                severity=RULES[i - 1]["severity"],
            )
            info = real_mqtt_client.publish(event_topic, event.model_dump_json(), qos=1)
            info.wait_for_publish(timeout=5)
            time.sleep(0.3)

        # 等待全部入库
        assert _wait_for_count(session, GW_ID, 3), "not all events ingested within timeout"

        # 验证 DB 状态
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
            assert wm is not None
            assert wm.last_acked_seq == 3
            assert wm.gaps == []
    finally:
        cloud_client.loop_stop()
        cloud_client.disconnect()
