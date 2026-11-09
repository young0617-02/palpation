"""E2E 全链路集成测试：验证 E1 端侧能力闭环的完整链路。

完整链路：
    sim设备 → 适配器 → 规则/异常检测 → LLM初判 → MQTT上行 → 云端入库 → 诊断 → OTA → 影子

测试场景：
1. 完整数据流端到端（mock MQTT）：InternalSample → RuleEngine → AnomalyDetector
   → EdgeEvent → cloud _ingest_event → depth_diagnosis → CloudDiagnosis
2. 断网自治 + 补传全链路：SqliteQueue enqueue → 断网 → gap_detection → dedup_check
   → 水位线补传 → query_audit 审计留痕
3. OTA 下载→校验→切换→探针→回滚全链路：ModelManifest → deploy → comprehensive_probe
   → rollback
4. 设备影子 desired→delta→拉取→reported 收敛全链路：ShadowBox apply_desired
   → needs_pull → update_reported → sync_status
5. Agent 状态机 + LLM 初判全链路（mock LLM）：EdgeEvent → EdgeAgentRuntime.preliminary
   → 标注"未验证" → p95_inference_ms
6. 适配器→规则→异常检测联合链路（mock 设备）：模拟传感器 → ModbusTcpAdapter
   → RuleEngine → AnomalyDetector
7. 完整端到端链路（amqtt broker，skip if 不可用）：amqtt broker → edge 发布 EdgeEvent
   → cloud 订阅入库 → depth_diagnosis
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import socketserver
import threading
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import Base, EdgeEventRow, Watermark
from cloud.diagnosis import depth_diagnosis
from cloud.mqtt_bridge import _ingest_event, attach
from edge.agent.state_machine import EdgeAgentRuntime
from edge.agent.tools import TOOL_SCHEMA, build_tools
from edge.cloud.shadow import ShadowBox
from edge.queue.sqlite_queue import SqliteQueue
from edge.queue.watermark import (
    ReplayStatus,
    apply_ack,
    detect_gaps,
    next_batch,
    replay_status,
)
from edge.stream.anomaly_onnx import AnomalyDetector, ensure_ae_onnx
from edge.stream.rules import RuleEngine, RuleHit
from edge.stream.window import SlidingWindow
from mgmt_agent.ota import (
    OtaState,
    deploy,
    download,
    rollback,
    switch_slot,
)
from mgmt_agent.probe import ProbeResult, comprehensive_probe
from mgmt_agent.version import VersionSlots
from shared import topics
from shared.schemas import (
    CloudDiagnosis,
    EdgeEvent,
    InternalSample,
    ModelManifest,
)

GW_ID = "gw-e2e-full-001"
DEVICE_ID = "pump-de-01"
DEVICE_ID_HASH = "hash-pump-de-01"
BROKER_PORT = 11884

# 与 config/current.yaml 对齐的规则定义
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


def _make_sample(
    device_id: str = DEVICE_ID,
    source: str = "modbus",
    metrics: dict[str, float] | None = None,
) -> InternalSample:
    """构造 InternalSample，默认故障工况指标。"""
    if metrics is None:
        metrics = {"vibration_rms": 6.3, "temperature": 85.0, "current": 20.0}
    return InternalSample(
        device_id=device_id,
        source=source,
        ts=_now(),
        metrics=metrics,
    )


def _make_event(event_id: str = "e-full-1", seq: int = 1, **overrides) -> EdgeEvent:
    """构造测试用 EdgeEvent，可覆盖任意字段。"""
    base = dict(
        event_id=event_id,
        gw_id=GW_ID,
        device_id_hash=DEVICE_ID_HASH,
        seq=seq,
        detected_at=_now(),
        rule_id="bearing_rms_high",
        severity="high",
        metric="vibration_rms",
        value=6.3,
        threshold=4.5,
        preliminary_diagnosis="疑似轴承磨损 端侧初判，未经验证。",
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


def _make_manifest(
    url: str,
    content: bytes,
    *,
    model_id: str = "anomaly-det",
    version: str = "0.2.0",
    requires_firmware: str = "0.1.0",
    runtime: str = "onnxruntime",
) -> ModelManifest:
    """构造 ModelManifest，sha256 自动按 content 计算。"""
    return ModelManifest(
        model_id=model_id,
        version=version,
        artifact_url=url,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        quant="int8",
        runtime=runtime,
        target_hardware=["x86"],
        requires_firmware=requires_firmware,
        preproc_ver="v1",
    )


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """静音的 SimpleHTTPRequestHandler，避免测试输出被日志淹没。"""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _FakeResp:
    """httpx.get 返回的假响应。"""

    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def temp_cloud_db(tmp_path, monkeypatch):
    """替换 cloud DB 到 tmp_path sqlite，patch 所有 SessionLocal/engine 引用。"""
    db_path = tmp_path / "cloud_e2e_full.db"
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
    """捕获 publish 调用的 mock MQTT client。"""
    client = MagicMock()
    published: list[dict] = []

    def capture_publish(topic, payload, qos=0, retain=False):
        published.append({"topic": topic, "payload": payload, "qos": qos})

    client.publish = capture_publish
    client.published = published
    return client


@pytest.fixture()
def file_server(tmp_path: Path):
    """在 tmp_path 上启动本地线程化 HTTP 文件服务器，返回 (base_url, root)。"""
    handler = partial(_QuietHandler, directory=str(tmp_path))
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}", tmp_path
        httpd.shutdown()
        t.join(timeout=2)


@pytest.fixture()
def amqtt_broker():
    """启动真实 amqtt broker（后台线程），返回端口号。

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
    broker_ref: list = []

    async def start_broker() -> None:
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

    if broker_ref:
        try:
            fut = asyncio.run_coroutine_threadsafe(broker_ref[0].shutdown(), loop)
            fut.result(timeout=5)
        except Exception:
            pass
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# 场景1：完整数据流端到端（mock MQTT）
# --------------------------------------------------------------------------- #


def test_e2e_full_dataflow_mock_mqtt(temp_cloud_db, mock_mqtt_client, tmp_path):
    """完整数据流端到端：InternalSample → RuleEngine → AnomalyDetector → EdgeEvent
    → cloud _ingest_event → depth_diagnosis → CloudDiagnosis。

    验证每个环节的输入输出正确衔接：
    - InternalSample.metrics → RuleEngine.evaluate → RuleHit 列表
    - RuleHit + InternalSample → AnomalyDetector.is_anomaly → 异常标记
    - RuleHit → EdgeEvent 构造（字段映射）
    - EdgeEvent → _ingest_event → DB 入库 + Watermark + ACK
    - EdgeEvent → depth_diagnosis → CloudDiagnosis（conclusion/confidence/citations）
    """
    session = temp_cloud_db

    # Step 1: 构造 InternalSample（故障工况）
    sample = _make_sample(metrics=_fault_metrics())
    assert sample.device_id == DEVICE_ID
    assert set(sample.metrics.keys()) == {"vibration_rms", "temperature", "current"}

    # Step 2: RuleEngine 评估 → RuleHit 列表
    engine = RuleEngine(RULES)
    hits = engine.evaluate(sample.metrics)
    assert len(hits) == 3, "故障工况应触发全部 3 条规则"
    rule_ids = {h.rule_id for h in hits}
    assert rule_ids == {"bearing_rms_high", "temperature_high", "current_high"}

    # 验证 RuleHit 字段与输入指标一致
    bearing_hit = next(h for h in hits if h.rule_id == "bearing_rms_high")
    assert bearing_hit.metric == "vibration_rms"
    assert bearing_hit.value == sample.metrics["vibration_rms"]
    assert bearing_hit.threshold == 4.5
    assert bearing_hit.severity == "high"

    # Step 3: AnomalyDetector 检测（使用 tmp_path 自动创建 AE 模型）
    model_path = tmp_path / "anomaly" / "model.onnx"
    detector = AnomalyDetector(model_path, window=32, threshold=3.0)
    window = SlidingWindow(32)
    # 填充窗口：先正常值再异常值，让异常检测器能识别
    for _ in range(20):
        window.push("vibration_rms", 1.2)
    for _ in range(12):
        window.push("vibration_rms", 6.3)
    is_anom, score = detector.is_anomaly(window, metric="vibration_rms")
    assert isinstance(is_anom, bool)
    assert isinstance(score, float)
    # 异常值窗口分数应高于正常值窗口
    window_normal = SlidingWindow(32)
    for _ in range(32):
        window_normal.push("vibration_rms", 1.2)
    _, score_normal = detector.is_anomaly(window_normal, metric="vibration_rms")
    assert score >= score_normal, "异常工况分数应不低于正常工况"

    # Step 4: 构造 EdgeEvent（基于 RuleHit 字段映射）
    event = _make_event(
        event_id="evt-full-dataflow-001",
        seq=1,
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

    # Step 5: cloud _ingest_event → DB 入库 + Watermark + ACK
    _ingest_event(mock_mqtt_client, event)

    with session() as s:
        row = s.query(EdgeEventRow).filter_by(event_id="evt-full-dataflow-001").one()
        assert row.gw_id == GW_ID
        assert row.seq == 1
        wm = s.get(Watermark, GW_ID)
        assert wm is not None
        assert wm.last_acked_seq == 1
        assert wm.gaps == []

    # 验证 ACK 发布
    acks = [m for m in mock_mqtt_client.published if m["topic"].endswith("/ack")]
    assert len(acks) == 1
    ack = json.loads(acks[0]["payload"])
    assert ack["gw_id"] == GW_ID
    assert ack["last_acked_seq"] == 1

    # Step 6: depth_diagnosis → CloudDiagnosis
    diag = depth_diagnosis(event)
    assert isinstance(diag, CloudDiagnosis)
    assert diag.event_id == "evt-full-dataflow-001"
    assert diag.gw_id == GW_ID
    assert diag.conclusion, "conclusion 不应为空"
    assert "振动" in diag.conclusion, "conclusion 应包含振动关键词"
    assert 0 < diag.confidence <= 0.95
    assert diag.requires_human_review is True  # severity=high → 需人工审核

    # Step 7: 验证诊断消息也通过 _ingest_event 发布到 cloud diagnosis topic
    diag_msgs = [
        m for m in mock_mqtt_client.published if "diagnosis/cloud" in m["topic"]
    ]
    assert len(diag_msgs) == 1
    published_diag = CloudDiagnosis.model_validate_json(diag_msgs[0]["payload"])
    assert published_diag.event_id == "evt-full-dataflow-001"
    assert published_diag.conclusion == diag.conclusion


# --------------------------------------------------------------------------- #
# 场景2：断网自治 + 补传全链路
# --------------------------------------------------------------------------- #


def test_e2e_full_offline_autonomy_replay(tmp_path):
    """断网自治 + 补传全链路：SqliteQueue enqueue 多条事件 → 模拟断网
    → gap_detection 发现 gap → dedup_check 幂等 → 水位线补传 → query_audit 审计留痕。

    验证：
    - 事件入队后 seq 连续递增
    - 删除中间事件后 gap_detection 正确发现缺口
    - dedup_check 对已存在 event_id 返回 True（幂等）
    - 水位线 ACK 后 next_batch 只返回未 ACK 事件
    - 审计记录写入并可查询
    """
    q = SqliteQueue(tmp_path / "offline.db")

    # Step 1: enqueue 5 条事件（模拟在线时连续产生事件）
    events: list[EdgeEvent] = []
    for i in range(5):
        ev = _make_event(event_id=f"evt-offline-{i:03d}", seq=0)
        ev = q.append_event(ev)
        events.append(ev)
        # 每条入队写审计
        q.audit("event", {"event_id": ev.event_id, "seq": ev.seq})

    # 验证 seq 连续递增（1..5）
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert q.event_count() == 5

    # Step 2: 模拟断网期间继续产生事件（入队但不 ACK）
    offline_events: list[EdgeEvent] = []
    for i in range(5, 8):
        ev = _make_event(event_id=f"evt-offline-{i:03d}", seq=0)
        ev = q.append_event(ev)
        offline_events.append(ev)
        q.audit("event", {"event_id": ev.event_id, "seq": ev.seq, "offline": True})

    assert q.event_count() == 8

    # Step 3: 模拟部分事件丢失（直接从 DB 删除 seq=4 和 seq=6 制造 gap）
    q._conn.execute("DELETE FROM events WHERE seq IN (4, 6)")
    q._conn.commit()
    assert q.event_count() == 6

    # gap_detection 发现缺口 [4,4] 和 [6,6]
    gaps = q.gap_detection()
    assert (4, 4) in gaps
    assert (6, 6) in gaps

    # watermark.detect_gaps 等价
    assert detect_gaps(q) == gaps

    # Step 4: dedup_check 幂等验证
    assert q.dedup_check("evt-offline-000") is True  # 已存在
    assert q.dedup_check("evt-offline-999") is False  # 不存在

    # 重复入队应抛出 EdgeError
    from shared.errors import EdgeError

    with pytest.raises(EdgeError):
        q.append_event(_make_event(event_id="evt-offline-000"))

    # Step 5: 水位线补传 —— 先 ACK seq=3，然后 next_batch 返回 seq>3 的待补传事件
    apply_ack(q, 3)
    assert q.last_acked_seq() == 3
    batch = next_batch(q, limit=100)
    batch_seqs = [e.seq for e in batch]
    # seq=4 已删除，seq=6 已删除，剩余待补传：5, 7, 8
    assert 5 in batch_seqs
    assert 7 in batch_seqs
    assert 8 in batch_seqs
    assert 4 not in batch_seqs
    assert 6 not in batch_seqs

    # Step 6: 补传完成后 ACK 到最大 seq
    max_seq = max(e.seq for e in batch)
    apply_ack(q, max_seq)
    assert q.queue_depth() == 0

    # Step 7: query_audit 审计留痕
    start = _now().replace(microsecond=0)
    # 审计记录已写入（8 条 event 审计）
    audit_records = q.query_audit(datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))
    assert len(audit_records) == 8
    # 验证审计记录内容
    kinds = {r.kind for r in audit_records}
    assert kinds == {"event"}
    # 验证离线事件审计标记
    offline_audits = [r for r in audit_records if r.body.get("offline") is True]
    assert len(offline_audits) == 3  # seq 5,6,7 标记离线（6 已删除但审计仍在）

    # Step 8: replay_status 综合状态
    status = replay_status(q)
    assert isinstance(status, ReplayStatus)
    assert status.last_acked_seq == max_seq
    assert status.pending_count == 0


# --------------------------------------------------------------------------- #
# 场景3：OTA 下载→校验→切换→探针→回滚全链路
# --------------------------------------------------------------------------- #


def test_e2e_full_ota_deploy_probe_rollback(file_server, tmp_path, monkeypatch):
    """OTA 下载→校验→切换→探针→回滚全链路。

    构造 ModelManifest → deploy（分块下载+sha256校验+A/B切换+综合探针）
    → 验证状态流转 → comprehensive_probe 健康探针 → rollback 回滚。

    验证：
    - deploy 成功路径：succeeded 状态 + current 指向新版本
    - deploy 探针失败路径：rolled_back 状态 + current 回退到旧版本
    - rollback 模块级函数：历史记录回滚
    - comprehensive_probe 各项检查
    """
    base, root = file_server

    # Step 1: 构造 ModelManifest + 准备模型文件
    content = b"fake onnx model bytes for e2e full ota"
    (root / "model.onnx").write_bytes(content)
    manifest = _make_manifest(f"{base}/model.onnx", content, version="0.3.0")
    models_dir = tmp_path / "models"

    # Step 2: mock comprehensive_probe 返回健康
    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(True, "ok", {}),
    )

    # Step 3: deploy 成功路径
    state = deploy(
        manifest,
        models_dir,
        firmware_version="0.1.0",
        health_url="http://edge.local/health",
    )
    assert state == OtaState.succeeded
    # 验证 A/B 槽切换
    assert VersionSlots(models_dir).current_version("anomaly") == "0.3.0"
    # 验证文件落地
    assert (models_dir / "anomaly" / "0.3.0" / "model.onnx").exists()
    assert (models_dir / "anomaly" / "0.3.0" / "manifest.json").exists()

    # Step 4: comprehensive_probe 各项检查（mock httpx.get）

    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": True,
                "mem_used_mb": 1024,
                "disk_free_mb": 10240,
                "cpu_temp_c": 50.0,
                "input_links": {"modbus": True, "opcua": True},
            }
        ),
    )
    probe_result = comprehensive_probe("http://edge.local/health")
    assert probe_result.probe_ok is True
    assert probe_result.inference_ok is True
    assert probe_result.mem_ok is True
    assert probe_result.disk_ok is True
    assert probe_result.temp_ok is True
    assert probe_result.input_links_ok is True
    assert probe_result.reason == "ok"

    # Step 5: deploy 探针失败 → 自动回滚
    content_v2 = b"new model v2 bytes"
    (root / "model_v2.onnx").write_bytes(content_v2)
    manifest_v2 = _make_manifest(f"{base}/model_v2.onnx", content_v2, version="0.4.0")

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(False, "failed: inference", {}),
    )

    state_v2 = deploy(
        manifest_v2,
        models_dir,
        firmware_version="0.1.0",
        health_url="http://edge.local/health",
        previous="0.3.0",
    )
    assert state_v2 == OtaState.rolled_back
    # current 应回退到 0.3.0
    assert VersionSlots(models_dir).current_version("anomaly") == "0.3.0"

    # Step 6: rollback 模块级函数验证
    # 此时 history 应为 [0.3.0]（0.4.0 已被 rollback 移除）
    prev = rollback(models_dir, "anomaly")
    # 只有 0.3.0 在历史中，无法回滚
    assert prev is None

    # Step 7: download + switch_slot 独立验证
    content_v3 = b"v3 model content"
    (root / "model_v3.onnx").write_bytes(content_v3)
    dest = tmp_path / "manual_dl.onnx"
    digest_v3 = hashlib.sha256(content_v3).hexdigest()
    downloaded = download(f"{base}/model_v3.onnx", dest, digest_v3)
    assert downloaded == dest
    assert dest.read_bytes() == content_v3

    # switch_slot 切换到新版本
    (models_dir / "anomaly" / "0.5.0").mkdir(parents=True, exist_ok=True)
    switch_slot(models_dir, "anomaly", "0.5.0")
    assert VersionSlots(models_dir).current_version("anomaly") == "0.5.0"


def test_e2e_full_ota_firmware_incompatible(file_server, tmp_path):
    """OTA 固件不兼容时抛 EdgeError(E_REQUIRES_FIRMWARE)。"""
    from shared.errors import E_REQUIRES_FIRMWARE, EdgeError

    base, root = file_server
    content = b"x"
    (root / "model.onnx").write_bytes(content)
    manifest = _make_manifest(f"{base}/model.onnx", content, requires_firmware="9.9.9")

    with pytest.raises(EdgeError) as exc:
        deploy(
            manifest,
            tmp_path / "models",
            firmware_version="0.1.0",
            health_url="http://x",
        )
    assert exc.value.code == E_REQUIRES_FIRMWARE


def test_e2e_full_ota_hash_mismatch(file_server, tmp_path, monkeypatch):
    """OTA sha256 校验失败时抛 EdgeError(E_MODEL_HASH_MISMATCH)。"""
    from shared.errors import E_MODEL_HASH_MISMATCH, EdgeError

    base, root = file_server
    (root / "model.onnx").write_bytes(b"actual content")
    # 构造 manifest 但 sha256 故意写错
    manifest = _make_manifest(f"{base}/model.onnx", b"different content", version="0.2.0")

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(True, "ok", {}),
    )

    with pytest.raises(EdgeError) as exc:
        deploy(
            manifest,
            tmp_path / "models",
            firmware_version="0.1.0",
            health_url="http://x",
        )
    assert exc.value.code == E_MODEL_HASH_MISMATCH


# --------------------------------------------------------------------------- #
# 场景4：设备影子 desired→delta→拉取→reported 收敛全链路
# --------------------------------------------------------------------------- #


def test_e2e_full_shadow_convergence():
    """设备影子 desired→delta→拉取→reported 收敛全链路。

    ShadowBox apply_desired → needs_pull 返回 delta → update_reported 收敛
    → sync_status 验证。

    验证：
    - apply_desired 返回完整 delta
    - needs_pull 逐字段减少
    - update_reported 逐字段收敛到空 delta
    - sync_status in_sync 从 False 变 True
    - "边缘永不自己决定版本"：update_reported 不修改 desired
    """
    # 初始状态：reported 已有旧版本
    sb = ShadowBox(reported={"model_version": "1.0", "config_version": "v0001"})

    # Step 1: 云端下发新 desired（model_version 升级，config_version 不变）
    delta = sb.apply_desired({"model_version": "2.0", "config_version": "v0001"})
    assert delta == {"model_version": "2.0"}, "delta 应只包含 model_version"

    # Step 2: needs_pull 返回待拉取字段
    assert sb.needs_pull() == {"model_version": "2.0"}
    status = sb.sync_status()
    assert status["in_sync"] is False
    assert status["pending_fields"] == ["model_version"]

    # Step 3: 边缘主动拉取并应用新模型 → update_reported 逐字段收敛
    residual = sb.update_reported("model_version", "2.0")
    assert residual == {}, "model_version 收敛后 residual delta 应为空"

    # Step 4: needs_pull 归零，收敛完成
    assert sb.needs_pull() == {}
    status_final = sb.sync_status()
    assert status_final["in_sync"] is True
    assert status_final["pending_fields"] == []

    # Step 5: "边缘永不自己决定版本"：desired 未被修改
    assert sb.desired == {"model_version": "2.0", "config_version": "v0001"}
    assert sb.reported == {"model_version": "2.0", "config_version": "v0001"}


def test_e2e_full_shadow_multi_field_convergence():
    """多字段逐字段收敛：desired 含 3 个字段变更，逐个 update_reported 直到 delta 归零。"""
    sb = ShadowBox(reported={"a": 1, "b": 2, "c": 3})

    # 云端下发新 desired，3 个字段都变
    delta = sb.apply_desired({"a": 10, "b": 20, "c": 30})
    assert delta == {"a": 10, "b": 20, "c": 30}

    # 逐字段收敛
    residual = sb.update_reported("a", 10)
    assert residual == {"b": 20, "c": 30}
    assert not sb.sync_status()["in_sync"]

    residual = sb.update_reported("b", 20)
    assert residual == {"c": 30}
    assert not sb.sync_status()["in_sync"]

    residual = sb.update_reported("c", 30)
    assert residual == {}
    assert sb.sync_status()["in_sync"]

    # 全部收敛后 needs_pull 为空
    assert sb.needs_pull() == {}


def test_e2e_full_shadow_offline_backlog():
    """影子离线/补传跟踪：mark_offline → incr_backlog → mark_online。"""
    sb = ShadowBox()

    # 进入离线
    sb.mark_offline()
    status = sb.sync_status()
    assert status["offline_duration_s"] is not None
    assert status["offline_duration_s"] >= 0

    # 离线期间积压 5 条事件
    for _ in range(5):
        sb.incr_backlog()
    assert sb.sync_status()["backlog_count"] == 5

    # 恢复在线
    sb.mark_online()
    assert sb.sync_status()["offline_duration_s"] is None
    # backlog_count 保留（由上层清零）
    assert sb.backlog_count == 5


# --------------------------------------------------------------------------- #
# 场景5：Agent 状态机 + LLM 初判全链路（mock LLM）
# --------------------------------------------------------------------------- #


def _make_agent_runtime(
    llm_available: bool = True,
    infer_return: dict | None = None,
    tools: dict | None = None,
) -> tuple[EdgeAgentRuntime, MagicMock, MagicMock, MagicMock]:
    """构造 EdgeAgentRuntime，mock EdgeLLM / FaultChain / LocalKnowledge。"""
    llm = MagicMock()
    llm.available = llm_available
    llm.infer.return_value = infer_return
    chain = MagicMock()
    chain.two_hop.return_value = [
        {"id": "bearing_wear", "actions": ["检查润滑", "检查对中"]}
    ]
    knowledge = MagicMock()
    knowledge.search.return_value = [
        {"id": "kb-1", "title": "振动越限", "text": "轴承早期磨损特征"}
    ]
    rt = EdgeAgentRuntime(
        tools=tools if tools is not None else {},
        llm=llm,
        chain=chain,
        knowledge=knowledge,
    )
    return rt, llm, chain, knowledge


def _make_hit() -> RuleHit:
    return RuleHit(
        rule_id="bearing_rms_high",
        metric="vibration_rms",
        value=6.3,
        threshold=4.5,
        severity="high",
        description="驱动端轴承振动 RMS 越限",
    )


def test_e2e_full_agent_llm_preliminary():
    """Agent 状态机 + LLM 初判全链路（mock LLM）。

    构造 EdgeEvent → EdgeAgentRuntime.preliminary → 验证标注"未验证"
    → p95_inference_ms 延迟统计。

    验证：
    - preliminary 返回 LLM 诊断结果
    - 诊断文本包含"端侧初判，未经验证"标注
    - 推理延迟被记录到 inference_latencies_ms
    - p95_inference_ms 返回合理值
    - build_event 构造完整 EdgeEvent
    """
    # Step 1: 构造 mock LLM 返回诊断结果
    rt, llm, _, _ = _make_agent_runtime(
        infer_return={"diagnosis": "轴承外圈磨损", "confidence": 0.82}
    )

    sample = _make_sample(metrics={"vibration_rms": 6.3})
    hit = _make_hit()

    # Step 2: preliminary 初判
    text, conf = rt.preliminary(sample, hit, degraded=False)
    assert "轴承外圈磨损" in text
    assert conf == 0.82

    # Step 3: 验证"未验证"标注
    assert "端侧初判" in text
    assert "未经验证" in text

    # Step 4: 推理延迟统计
    assert len(rt.inference_latencies_ms) == 1
    assert rt.inference_latencies_ms[0] >= 0.0
    p95 = rt.p95_inference_ms()
    assert p95 >= 0.0

    # Step 5: 多次推理后 p95 统计合理
    # 注意：preliminary 有缓存机制（cache_ttl_s=60），相同 cache_key 会命中缓存
    # 不记录推理延迟。用不同 device_id 构造不同 cache_key 避免缓存命中。
    for i in range(4):
        sample_i = _make_sample(
            device_id=f"pump-extra-{i:02d}",
            metrics={"vibration_rms": 6.3},
        )
        rt.preliminary(sample_i, hit, degraded=False)
    assert len(rt.inference_latencies_ms) == 5
    p95_final = rt.p95_inference_ms()
    assert p95_final >= 0.0
    # p95 应在 latencies 范围内
    assert min(rt.inference_latencies_ms) <= p95_final <= max(rt.inference_latencies_ms)


def test_e2e_full_agent_degraded_fallback():
    """Agent 降级路径：degraded=True 时使用规则 fallback，不调用 LLM。"""
    rt, llm, _, _ = _make_agent_runtime()
    sample = _make_sample()
    hit = _make_hit()

    text, conf = rt.preliminary(sample, hit, degraded=True)
    assert conf == 0.55
    assert "疑似bearing_wear" in text
    assert "端侧初判" in text
    assert "未经验证" in text
    llm.infer.assert_not_called()
    # 降级路径不记录推理延迟
    assert rt.inference_latencies_ms == []
    assert rt.p95_inference_ms() == 0.0


def test_e2e_full_agent_build_event_full():
    """Agent build_event 构造完整 EdgeEvent，字段从 RuleHit 正确映射。"""
    rt, _, _, _ = _make_agent_runtime()
    hit = _make_hit()
    ev = rt.build_event(
        gw_id=GW_ID,
        device_id_hash=DEVICE_ID_HASH,
        hit=hit,
        evidence_refs=["win:vibration_rms:20260101T120000000", "kb-1"],
        diagnosis="疑似轴承磨损 端侧初判，未经验证。",
        confidence=0.78,
        degraded=False,
    )
    assert isinstance(ev, EdgeEvent)
    assert ev.gw_id == GW_ID
    assert ev.device_id_hash == DEVICE_ID_HASH
    assert ev.rule_id == "bearing_rms_high"
    assert ev.severity == "high"
    assert ev.metric == "vibration_rms"
    assert ev.value == 6.3
    assert ev.threshold == 4.5
    assert ev.preliminary_diagnosis == "疑似轴承磨损 端侧初判，未经验证。"
    assert ev.preliminary_confidence == 0.78
    assert ev.evidence_refs == ["win:vibration_rms:20260101T120000000", "kb-1"]
    assert ev.degraded is False
    assert ev.event_id  # ULID 非空
    assert rt.state.last_event_id == ev.event_id


def test_e2e_full_agent_tool_schema_and_get_device_status():
    """工具 schema 一致性 + get_device_status 工具返回设备状态。"""
    window = SlidingWindow()
    window.push("vibration_rms", 5.0)
    window.push("vibration_rms", 6.2)
    window.push("temperature", 75.0)

    tools = build_tools(window, MagicMock(), MagicMock())
    # TOOL_SCHEMA 与 build_tools 键一致
    schema_names = {entry["name"] for entry in TOOL_SCHEMA}
    assert schema_names == set(tools)
    assert "get_device_status" in tools

    # get_device_status 返回窗口内指标
    result = tools["get_device_status"]()
    status = result["device_status"]
    assert "vibration_rms" in status
    assert status["vibration_rms"]["last"] == 6.2
    assert status["vibration_rms"]["count"] == 2
    assert "temperature" in status
    assert "snapshot_id" in status["vibration_rms"]


# --------------------------------------------------------------------------- #
# 场景6：适配器→规则→异常检测联合链路（mock 设备）
# --------------------------------------------------------------------------- #


class _FakeModbusResult:
    """模拟 pymodbus 读取结果。"""

    def __init__(self, registers: list[int], error: bool = False):
        self.registers = registers
        self._error = error

    def isError(self) -> bool:
        return self._error


class _FakeModbusClient:
    """模拟 AsyncModbusTcpClient，可控连接与读取行为。"""

    def __init__(self, registers: list[int] | None = None):
        self._registers = registers if registers is not None else [630, 850, 2000]
        self.connected = False
        self.connect_calls = 0
        self.read_calls = 0

    async def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        return True

    async def read_holding_registers(self, address: int, count: int = 1, device_id: int = 1):
        self.read_calls += 1
        return _FakeModbusResult([self._registers[address]])

    def close(self) -> None:
        self.connected = False


async def _collect_samples(
    queue: asyncio.Queue[InternalSample], count: int, timeout: float = 3.0
) -> list[InternalSample]:
    """从队列收集 count 个样本。"""
    samples: list[InternalSample] = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while len(samples) < count:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            s = await asyncio.wait_for(queue.get(), timeout=remaining)
            samples.append(s)
        except asyncio.TimeoutError:
            break
    return samples


async def _run_adapter_for(
    adapter,
    queue: asyncio.Queue[InternalSample],
    stop: asyncio.Event,
    duration: float = 0.5,
) -> None:
    """运行适配器 duration 秒后设置 stop。"""
    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.sleep(duration)
    stop.set()
    import contextlib

    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=5.0)



def test_e2e_full_adapter_rules_anomaly_chain(tmp_path):
    """适配器→规则→异常检测联合链路（完整异步流程）。

    使用 asyncio.run 完整运行：
    mock Modbus → 产出 InternalSample → RuleEngine → AnomalyDetector。
    """
    from edge.adapters.modbus_tcp import ModbusTcpAdapter

    async def run_chain():
        # mock Modbus 客户端：故障工况寄存器值
        # vibration_rms=6.3 → 630, temperature=85.0 → 850, current=20.0 → 2000
        fake_client = _FakeModbusClient(registers=[630, 850, 2000])
        holder = {"fake": fake_client}

        def constructor(*args: Any, **kw: Any) -> _FakeModbusClient:
            return holder["fake"]

        with patch(
            "edge.adapters.modbus_tcp.AsyncModbusTcpClient", side_effect=constructor
        ):
            adapter = ModbusTcpAdapter(
                "127.0.0.1", 5020, DEVICE_ID, period_s=0.05
            )
            queue: asyncio.Queue[InternalSample] = asyncio.Queue()
            stop = asyncio.Event()

            await _run_adapter_for(adapter, queue, stop, duration=0.3)

            # 收集样本
            samples = await _collect_samples(queue, 1)
            return samples

    samples = asyncio.run(run_chain())
    assert len(samples) >= 1, "适配器应至少产出 1 个样本"
    sample = samples[0]

    # Step 2: 验证 InternalSample 字段
    assert sample.device_id == DEVICE_ID
    assert sample.source == "modbus"
    assert isinstance(sample.ts, datetime)
    assert sample.ts.tzinfo is not None
    assert "vibration_rms" in sample.metrics
    assert "temperature" in sample.metrics
    assert "current" in sample.metrics
    # 寄存器值经缩放：630/100=6.3, 850/10=85.0, 2000/100=20.0
    assert sample.metrics["vibration_rms"] == pytest.approx(6.3, abs=0.01)
    assert sample.metrics["temperature"] == pytest.approx(85.0, abs=0.1)
    assert sample.metrics["current"] == pytest.approx(20.0, abs=0.01)

    # Step 3: RuleEngine 评估 → 故障工况触发规则
    engine = RuleEngine(RULES)
    hits = engine.evaluate(sample.metrics)
    assert len(hits) == 3, "故障工况应触发全部 3 条规则"
    rule_ids = {h.rule_id for h in hits}
    assert rule_ids == {"bearing_rms_high", "temperature_high", "current_high"}

    # 验证 RuleHit 字段与 InternalSample.metrics 一致
    bearing_hit = next(h for h in hits if h.rule_id == "bearing_rms_high")
    assert bearing_hit.value == pytest.approx(sample.metrics["vibration_rms"])
    assert bearing_hit.severity == "high"

    # Step 4: AnomalyDetector 检测
    model_path = tmp_path / "anomaly" / "model.onnx"
    detector = AnomalyDetector(model_path, window=32, threshold=3.0)
    window = SlidingWindow(32)
    # 填充窗口：先正常值再异常值
    for _ in range(20):
        window.push("vibration_rms", 1.2)
    for _ in range(12):
        window.push("vibration_rms", sample.metrics["vibration_rms"])
    is_anom, score = detector.is_anomaly(window, metric="vibration_rms")
    assert isinstance(is_anom, bool)
    assert isinstance(score, float)

    # 正常工况窗口分数应低于异常工况
    window_normal = SlidingWindow(32)
    for _ in range(32):
        window_normal.push("vibration_rms", 1.2)
    _, score_normal = detector.is_anomaly(window_normal, metric="vibration_rms")
    assert score >= score_normal, "异常工况分数应不低于正常工况"


def test_e2e_full_adapter_normal_metrics_no_rule():
    """适配器正常工况指标 → RuleEngine 无规则触发。"""
    engine = RuleEngine(RULES)
    hits = engine.evaluate(_normal_metrics())
    assert hits == [], "正常工况不应触发任何规则"


# --------------------------------------------------------------------------- #
# 场景7：完整端到端链路（amqtt broker，skip if 不可用）
# --------------------------------------------------------------------------- #


def test_e2e_full_amqtt_broker(temp_cloud_db, amqtt_broker):
    """完整端到端链路（amqtt broker）：edge 发布 EdgeEvent → cloud 订阅入库
    → depth_diagnosis → 验证完整链路。

    启动 amqtt broker，cloud 客户端订阅 edge/+/event，
    edge 客户端发布 EdgeEvent JSON，验证云端 DB 入库 + Watermark + 诊断生成。
    broker 不可用时 skip。
    """
    session = temp_cloud_db
    port = amqtt_broker

    # Step 1: 创建 cloud 客户端并 attach 桥接回调
    cloud_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id="cloud-e2e-full"
    )
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

    # Step 2: 创建 edge 客户端
    edge_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id="edge-e2e-full-pub"
    )
    edge_client.connect("127.0.0.1", port, 45)
    edge_client.loop_start()
    time.sleep(0.3)

    try:
        # Step 3: edge 发布 EdgeEvent
        event = _make_event(event_id="evt-broker-full-001", seq=1)
        event_topic = topics.event(GW_ID)

        found = False
        for _ in range(20):
            info = edge_client.publish(event_topic, event.model_dump_json(), qos=1)
            info.wait_for_publish(timeout=2)
            time.sleep(0.5)
            with session() as s:
                row = (
                    s.query(EdgeEventRow)
                    .filter_by(event_id="evt-broker-full-001")
                    .one_or_none()
                )
                if row is not None:
                    found = True
                    break

        assert found, "event not ingested into DB within timeout"

        # Step 4: 验证 DB 入库
        with session() as s:
            row = (
                s.query(EdgeEventRow)
                .filter_by(event_id="evt-broker-full-001")
                .one()
            )
            assert row.gw_id == GW_ID
            assert row.seq == 1
            wm = s.get(Watermark, GW_ID)
            assert wm is not None
            assert wm.last_acked_seq == 1
            assert wm.gaps == []

        # Step 5: 验证 depth_diagnosis 可对入库事件生成诊断
        diag = depth_diagnosis(event)
        assert isinstance(diag, CloudDiagnosis)
        assert diag.event_id == "evt-broker-full-001"
        assert diag.gw_id == GW_ID
        assert diag.conclusion
        assert 0 < diag.confidence <= 0.95
        assert diag.requires_human_review is True  # severity=high
    finally:
        edge_client.loop_stop()
        edge_client.disconnect()
        cloud_client.loop_stop()
        cloud_client.disconnect()


def test_e2e_full_amqtt_broker_multiple_events(temp_cloud_db, amqtt_broker):
    """完整端到端链路：连续发布多个事件，验证全部入库 + Watermark 推进 + 诊断生成。"""
    session = temp_cloud_db
    port = amqtt_broker

    cloud_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id="cloud-e2e-full-multi"
    )
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

    edge_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id="edge-e2e-full-multi-pub"
    )
    edge_client.connect("127.0.0.1", port, 45)
    edge_client.loop_start()
    time.sleep(0.3)

    try:
        event_topic = topics.event(GW_ID)
        published_events: list[EdgeEvent] = []
        for i in range(1, 4):
            event = _make_event(
                event_id=f"evt-broker-full-{i:03d}",
                seq=i,
                rule_id=RULES[i - 1]["id"],
                metric=RULES[i - 1]["metric"],
                value=RULES[i - 1]["threshold"] + 1,
                threshold=RULES[i - 1]["threshold"],
                severity=RULES[i - 1]["severity"],
            )
            published_events.append(event)
            info = edge_client.publish(event_topic, event.model_dump_json(), qos=1)
            info.wait_for_publish(timeout=2)
            time.sleep(0.3)

        # 等待全部入库
        all_found = False
        for _ in range(30):
            with session() as s:
                count = (
                    s.query(EdgeEventRow).filter(EdgeEventRow.gw_id == GW_ID).count()
                )
                if count >= 3:
                    all_found = True
                    break
            time.sleep(0.5)

        assert all_found, "not all events ingested within timeout"

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
            assert wm.last_acked_seq == 3
            assert wm.gaps == []

        # 验证每个事件都能生成诊断
        for ev in published_events:
            diag = depth_diagnosis(ev)
            assert isinstance(diag, CloudDiagnosis)
            assert diag.event_id == ev.event_id
            assert diag.conclusion
            assert 0 < diag.confidence <= 0.95
            # bearing_rms_high → high → 需人工审核；其他 → medium → 不需
            assert diag.requires_human_review is (ev.severity == "high")
    finally:
        edge_client.loop_stop()
        edge_client.disconnect()
        cloud_client.loop_stop()
        cloud_client.disconnect()