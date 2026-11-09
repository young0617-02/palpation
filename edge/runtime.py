from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from starlette.responses import Response

from edge.adapters.modbus_tcp import ModbusTcpAdapter
from edge.adapters.mqtt_sparkplug import MqttSparkplugAdapter
from edge.adapters.opcua_client import OpcuaAdapter
from edge.agent.llm_llamacpp import EdgeLLM
from edge.agent.state_machine import EdgeAgentRuntime
from edge.agent.tools import build_tools
from edge.cloud.masking import hash_device_id
from edge.cloud.mqtt_north import NorthMqtt
from edge.cloud.shadow import ShadowBox
from edge.config_loader import load_yaml
from edge.guard import ResourceGuard
from edge.knowledge.fault_chain import FaultChain
from edge.knowledge.sqlite_vec import LocalKnowledge
from edge.queue.sqlite_queue import SqliteQueue
from edge.queue.watermark import apply_ack, next_batch
from edge.settings import ROOT, settings
from edge.stream.anomaly_onnx import AnomalyDetector
from edge.stream.rules import RuleEngine, RuleHit
from edge.stream.window import SlidingWindow
from shared import topics
from shared.schemas import (
    AckPayload,
    EdgeEvent,
    EdgeHealth,
    EdgeTelemetry,
    InternalSample,
)

log = logging.getLogger(__name__)

T1_HIST = Histogram("edge_t1_ms", "T1 rule path latency ms")
T2_HIST = Histogram("edge_t2_ms", "T2 anomaly latency ms")
T4_HIST = Histogram("edge_t4_ms", "T4 knowledge latency ms")
EVENTS = Counter("edge_events_total", "events emitted")

# --- Grafana dashboard 面板指标（带 gw_id label） ---
QUEUE_DEPTH = Gauge("edge_queue_depth", "Local queue depth", ["gw_id"])
MODEL_VERSION = Gauge("edge_model_version", "Current model version (numeric)", ["gw_id"])
DEGRADE_LEVEL = Gauge("edge_degrade_level", "Degrade level (0=normal..6=emergency)", ["gw_id"])
NETWORK_CONNECTED = Gauge("edge_network_connected", "Network connected (1/0)", ["gw_id"])
OFFLINE_DURATION = Gauge("edge_offline_duration_seconds", "Offline duration in seconds", ["gw_id"])
CACHE_HIT_RATE = Gauge("edge_cache_hit_rate", "Local cache hit rate", ["gw_id"])
CPU_TEMP = Gauge("edge_cpu_temp_celsius", "CPU temperature in celsius", ["gw_id"])
DISK_FREE = Gauge("edge_disk_free_mb", "Disk free space in MB", ["gw_id"])
DESIRED_MODEL_VERSION = Gauge(
    "edge_desired_model_version", "Desired model version (numeric)", ["gw_id"]
)
# 云端升级计数（无 label，与 dashboard `rate(edge_cloud_escalations_total[1h])` 一致）
CLOUD_ESCALATIONS = Counter("edge_cloud_escalations_total", "Cloud escalations total")
# 版本元数据（Info 指标，记录版本字符串，导出为 edge_version_info{...}）
VERSION_INFO = Info("edge_version", "Edge version metadata", ["gw_id"])

# 降级级别字符串 → 数值映射（与 dashboard edge_degrade_level 面板一致）
_DEGRADE_LEVEL_MAP: dict[str, int] = {
    "normal": 0,
    "llm_small": 1,
    "llm_off": 2,
    "temp_throttle": 3,
    "offline_degraded": 4,
    "disk_critical": 5,
    "emergency": 6,
}

# 提取版本字符串中的数字部分（兼容 "0.9.1" / "v1.2.3" / "0.10.3-rc1"）
_VERSION_NUM_RE = re.compile(r"\d+")


def _version_to_gauge(version: str) -> float:
    """将语义版本字符串转为数值用于 Prometheus gauge 导出。

    例如 "0.9.1" -> 0.0901, "1.2.3" -> 1.0203, "v0.10.3" -> 0.1003。
    不同版本映射为不同数值，供 dashboard ``edge_model_version != edge_desired_model_version``
    数值比较使用。解析失败时返回 0.0。
    """
    parts = _VERSION_NUM_RE.findall(version)
    if not parts:
        return 0.0
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return major + minor * 0.01 + patch * 0.0001


class EdgeRuntime:
    def __init__(self) -> None:
        self.cfg = load_yaml(settings.config_path)
        self.gw_id = settings.gw_id
        self.samples: asyncio.Queue[InternalSample] = asyncio.Queue(maxsize=1000)
        self.stop = asyncio.Event()
        self.window = SlidingWindow(self.cfg["sampling"]["vibration_window"])
        self.rules = RuleEngine(self.cfg["rules"])
        model_path = settings.models_dir / "anomaly" / "current" / "model.onnx"
        self.detector = AnomalyDetector(
            model_path,
            window=self.cfg["anomaly"]["window"],
            threshold=self.cfg["anomaly"]["score_threshold"],
        )
        self.queue = SqliteQueue(settings.data_dir / "queue.db")
        chain = FaultChain(ROOT / "edge" / "knowledge" / "fault_chains.json")
        # T4 知识检索配置：相对路径按仓库根解析，缺省回落到 models_dir/knowledge/current
        kc = self.cfg.get("knowledge", {}) or {}
        kdir_raw = str(kc.get("model_dir") or "").strip()
        if kdir_raw:
            kdir = Path(kdir_raw)
            if not kdir.is_absolute():
                kdir = ROOT / kdir
        else:
            kdir = settings.models_dir / "knowledge" / "current"
        knowledge = LocalKnowledge(
            settings.data_dir / "knowledge.db",
            ROOT / "edge" / "knowledge" / "snippets.json",
            model_dir=kdir,
            dim=int(kc.get("dim", settings.knowledge_dim)),
            enabled=bool(kc.get("enabled", settings.knowledge_enabled)),
            onnx_rel=str(kc.get("onnx_file", "onnx/model_quantized.onnx")),
            max_length=int(kc.get("max_length", 512)),
            num_threads=int(kc.get("num_threads", settings.knowledge_num_threads)),
            enable_cpu_mem_arena=bool(kc.get("enable_cpu_mem_arena", True)),
            degrade_to_hash=bool(kc.get("degrade_to_hash", settings.knowledge_degrade_to_hash)),
        )
        self.knowledge = knowledge
        self.llm = EdgeLLM(settings.edge_llm_model, settings.edge_llm_maxtok, settings.edge_llm_ctx)
        tools = build_tools(self.window, chain, knowledge)
        self.agent = EdgeAgentRuntime(tools, self.llm, chain, knowledge)
        self.north = NorthMqtt(self.gw_id, settings.cloud_mqtt_host, settings.cloud_mqtt_port)
        self.shadow = ShadowBox(
            reported={
                "model_version": self._current_model_version(),
                "config_version": self.cfg.get("config_version", "v0001"),
                "firmware_version": settings.firmware_version,
            }
        )
        self.guard = ResourceGuard(
            settings.data_dir,
            self.cfg["degrade"]["mem_warn_mb"],
            self.cfg["degrade"]["mem_llm_off_mb"],
            self.cfg["degrade"]["disk_stop_model_mb"],
            self.cfg["degrade"]["cpu_temp_c"],
        )
        self.modbus = ModbusTcpAdapter(settings.modbus_host, settings.modbus_port, "pump-de-01")
        self.opcua = OpcuaAdapter(settings.opcua_endpoint, "meter-01")
        self.spb = MqttSparkplugAdapter(
            settings.cloud_mqtt_host,
            settings.cloud_mqtt_port,
            self.cfg.get("group_id", "plant1"),
            self.gw_id,
            "pump-de-01",
        )
        self.last_t1_ms = 0.0
        self.last_t2_ms = 0.0
        self.offline_since: datetime | None = None
        self.cloud_escalations = 0
        self.events_emitted = 0
        self._dedupe: dict[str, float] = {}

    def _current_model_version(self) -> str:
        pointer = settings.models_dir / "anomaly" / "current.txt"
        if pointer.exists():
            return pointer.read_text(encoding="utf-8").strip()
        current = settings.models_dir / "anomaly" / "current"
        if current.exists():
            return current.name
        return "0.9.1"

    def input_links(self) -> dict[str, bool]:
        return {
            "modbus:plc1": self.modbus.connected,
            "opcua:meter2": self.opcua.connected,
            "mqtt:sparkplug": self.spb.connected,
            "north": self.north.online,
        }

    async def start(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.north.on(topics.event_ack(self.gw_id), self._on_ack)
        self.north.on(topics.shadow_desired(self.gw_id), self._on_desired)
        self.north.on(topics.manifest(self.gw_id), self._on_manifest_note)
        self.north.start()
        asyncio.create_task(self.modbus.run(self.samples, self.stop), name="modbus")
        asyncio.create_task(self.opcua.run(self.samples, self.stop), name="opcua")
        asyncio.create_task(self.spb.run(self.samples, self.stop), name="sparkplug")
        asyncio.create_task(self._ingest_loop(), name="ingest")
        asyncio.create_task(self._telemetry_loop(), name="telemetry")
        asyncio.create_task(self._health_loop(), name="health")
        asyncio.create_task(self._replay_loop(), name="replay")

    async def stop_all(self) -> None:
        self.stop.set()
        self.north.stop()

    def _on_ack(self, payload: bytes) -> None:
        ack = AckPayload.model_validate_json(payload)
        apply_ack(self.queue, ack.last_acked_seq)

    def _on_desired(self, payload: bytes) -> None:
        desired = json.loads(payload.decode("utf-8"))
        delta = self.shadow.set_desired(desired)
        self.queue.audit("shadow_desired", {"delta": delta})
        # Edge never chooses the version; it only pulls when desired != reported.
        self.north.publish_reported(self.shadow.reported)

    def _on_manifest_note(self, payload: bytes) -> None:
        self.queue.audit("manifest_seen", {"bytes": len(payload)})

    async def _ingest_loop(self) -> None:
        while not self.stop.is_set():
            sample = await self.samples.get()
            await self._handle_sample(sample)

    async def _handle_sample(self, sample: InternalSample) -> None:
        guard = self.guard.snapshot()
        t0 = time.perf_counter()
        stats_by_metric = {}
        for name, value in sample.metrics.items():
            stats_by_metric[name] = self.window.push(name, value)
        hits = self.rules.evaluate(sample.metrics)
        t1 = (time.perf_counter() - t0) * 1000
        self.last_t1_ms = t1
        T1_HIST.observe(t1)

        t2s = time.perf_counter()
        anom, score = False, 0.0
        if self.cfg["anomaly"]["enabled"] and guard.t2_full and "vibration_rms" in sample.metrics:
            anom, score = self.detector.is_anomaly(self.window, "vibration_rms")
        self.last_t2_ms = (time.perf_counter() - t2s) * 1000
        T2_HIST.observe(self.last_t2_ms)

        if anom and not any(h.rule_id == "bearing_rms_high" for h in hits):
            hits.append(
                RuleHit(
                    rule_id="t2_ae_vibration",
                    metric="vibration_rms",
                    value=sample.metrics.get("vibration_rms", score),
                    threshold=self.cfg["anomaly"]["score_threshold"],
                    severity="high",
                    description="轻量自编码器判定振动模式异常",
                )
            )

        degraded = guard.level == "llm_off" or not self.north.online and False
        if guard.level == "llm_off":
            degraded = True

        for hit in hits:
            await self._emit_hit(sample, hit, degraded)

    async def _emit_hit(self, sample: InternalSample, hit: RuleHit, degraded: bool) -> None:
        key = f"{sample.device_id}:{hit.rule_id}"
        now = time.time()
        if now - self._dedupe.get(key, 0) < 5:
            return
        self._dedupe[key] = now
        evidence = [self.window.snapshot_id(hit.metric)]
        t4s = time.perf_counter()
        _ = self.knowledge.search(hit.description, top_k=1)
        T4_HIST.observe((time.perf_counter() - t4s) * 1000)

        event = self.agent.build_event(
            self.gw_id,
            hash_device_id(sample.device_id, settings.hmac_salt),
            hit,
            evidence,
            None,
            None,
            degraded,
        )
        event = self.queue.append_event(event)
        EVENTS.inc()
        self.events_emitted += 1
        self.queue.audit("event", {"event_id": event.event_id, "rule_id": hit.rule_id})
        if self.north.online:
            self.north.publish_event(event)
        self.cloud_escalations += 1
        asyncio.create_task(self._fill_diagnosis(event, sample, hit, degraded))

    async def _fill_diagnosis(
        self, event: EdgeEvent, sample: InternalSample, hit: RuleHit, degraded: bool
    ) -> None:
        text, conf = await asyncio.to_thread(self.agent.preliminary, sample, hit, degraded)
        event.preliminary_diagnosis = text
        event.preliminary_confidence = conf
        event.degraded = degraded or not self.llm.available
        self.queue.audit("preliminary", {"event_id": event.event_id, "text": text})
        if self.north.online:
            self.north.publish_preliminary(event)

    async def _telemetry_loop(self) -> None:
        period = float(self.cfg["sampling"]["telemetry_period_s"])
        while not self.stop.is_set():
            await asyncio.sleep(period)
            for metric in ("vibration_rms", "temperature", "current"):
                vals = self.window.values(metric)
                if not vals:
                    continue
                st = self.window.push(metric, vals[-1])
                tel = EdgeTelemetry(
                    gw_id=self.gw_id,
                    device_id_hash=hash_device_id("pump-de-01", settings.hmac_salt),
                    metric=metric,
                    window_start=datetime.now(UTC),
                    window_end=datetime.now(UTC),
                    agg={
                        "avg": st.avg,
                        "max": st.max,
                        "min": st.min,
                        "p95": st.p95,
                        "std": st.std,
                        "slope": st.slope,
                        "sample_count": float(st.sample_count),
                    },
                    raw_rate_hz=float(self.cfg["sampling"]["raw_rate_hz"]),
                )
                if self.north.online:
                    self.north.publish_telemetry(tel)

    async def _health_loop(self) -> None:
        period = float(self.cfg["sampling"]["health_period_s"])
        while not self.stop.is_set():
            await asyncio.sleep(period)
            if not self.north.online:
                if self.offline_since is None:
                    self.offline_since = datetime.now(UTC)
            else:
                self.offline_since = None
            g = self.guard.snapshot()
            health = EdgeHealth(
                gw_id=self.gw_id,
                ts=datetime.now(UTC),
                firmware_version=settings.firmware_version,
                model_versions={"intent": "1.0.0", "anomaly_det": self._current_model_version()},
                config_version=str(self.cfg.get("config_version", "v0001")),
                inference_p95_ms=self.llm.p95_ms() or self.last_t1_ms,
                mem_used_mb=g.mem_used_mb,
                disk_free_mb=g.disk_free_mb,
                cpu_temp_c=g.cpu_temp_c,
                input_links=self.input_links(),
                offline_since=self.offline_since,
                queue_depth=self.queue.queue_depth(),
                probe_ok=g.accept_models and self.modbus.connected or True,
            )
            self.shadow.set_reported(
                {
                    "model_version": health.model_versions.get("anomaly_det"),
                    "config_version": health.config_version,
                    "firmware_version": health.firmware_version,
                }
            )
            if self.north.online:
                self.north.publish_health(health)
                self.north.publish_reported(self.shadow.reported)

    async def _replay_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(2.0)
            if not self.north.online:
                continue
            batch = next_batch(self.queue, 100)
            for ev in batch:
                self.north.publish_event(ev)


def _export_runtime_metrics(runtime: EdgeRuntime) -> None:
    """从 runtime 状态填充 Grafana dashboard 面板指标值。

    在 /metrics 端点每次被 scrape 时调用，将运行时状态同步到
    prometheus_client 全局注册表，随后由 generate_latest() 统一导出。
    """
    gw_id = runtime.gw_id
    guard_snap = runtime.guard.snapshot()
    full_snap = runtime.guard.full_snapshot()
    shadow_snap = runtime.shadow.snapshot()

    model_version = runtime._current_model_version()
    # 优先取 shadow reported 中的 model_version（与 _health_loop 上报一致）
    reported_version = str(shadow_snap.get("reported", {}).get("model_version") or model_version)
    desired_version = str(shadow_snap.get("desired", {}).get("model_version") or reported_version)

    QUEUE_DEPTH.labels(gw_id=gw_id).set(runtime.queue.queue_depth())
    MODEL_VERSION.labels(gw_id=gw_id).set(_version_to_gauge(reported_version))
    DEGRADE_LEVEL.labels(gw_id=gw_id).set(_DEGRADE_LEVEL_MAP.get(guard_snap.level, 0))
    NETWORK_CONNECTED.labels(gw_id=gw_id).set(1.0 if full_snap.net_connected else 0.0)
    OFFLINE_DURATION.labels(gw_id=gw_id).set(full_snap.offline_duration_s)
    CACHE_HIT_RATE.labels(gw_id=gw_id).set(0.0)
    CPU_TEMP.labels(gw_id=gw_id).set(guard_snap.cpu_temp_c or 0.0)
    DISK_FREE.labels(gw_id=gw_id).set(guard_snap.disk_free_mb)
    DESIRED_MODEL_VERSION.labels(gw_id=gw_id).set(_version_to_gauge(desired_version))
    VERSION_INFO.labels(gw_id=gw_id).info(
        {"model_version": reported_version, "desired_model_version": desired_version}
    )
    # edge_cloud_escalations_total 为 Counter，仅在云升级时递增（见 _emit_hit），
    # 此处不重复递增，避免每次 scrape 虚增计数。


def create_app(runtime: EdgeRuntime) -> FastAPI:
    app = FastAPI(title="edge-agent", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        g = runtime.guard.snapshot()
        return {
            "ok": True,
            "gw_id": runtime.gw_id,
            "links": runtime.input_links(),
            "queue_depth": runtime.queue.queue_depth(),
            "guard": g.level,
            "t1_ms": runtime.last_t1_ms,
            "t2_ms": runtime.last_t2_ms,
            "llm": runtime.llm.available,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        _export_runtime_metrics(runtime)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/shadow")
    def shadow() -> dict[str, Any]:
        return runtime.shadow.snapshot()

    @app.on_event("startup")
    async def _startup() -> None:
        await runtime.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await runtime.stop_all()

    return app
