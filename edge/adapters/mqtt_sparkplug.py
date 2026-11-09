from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import paho.mqtt.client as mqtt

from shared.schemas import InternalSample
from shared.topics import sparkplug_ddata

log = logging.getLogger(__name__)


class MqttSparkplugAdapter:
    """Consumes Sparkplug-like JSON DDATA (Phase 0: JSON, not protobuf)."""

    name = "mqtt_sparkplug"

    def __init__(self, host: str, port: int, group: str, edge_node: str, device_id: str):
        self.host = host
        self.port = port
        self.topic = sparkplug_ddata(group, edge_node, device_id)
        self.device_id = device_id
        self.connected = False
        self._queue: asyncio.Queue[InternalSample] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self, out: asyncio.Queue[InternalSample], stop: asyncio.Event) -> None:
        self._queue = out
        self._loop = asyncio.get_running_loop()
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-spb-{self.device_id}"
        )
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect
        client.connect_async(self.host, self.port, keepalive=45)
        client.loop_start()
        try:
            await stop.wait()
        finally:
            client.loop_stop()
            client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = reason_code == 0 or getattr(reason_code, "value", 1) == 0
        if self.connected:
            client.subscribe(self.topic, qos=1)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        if not self._queue or not self._loop:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            metrics = {}
            for m in payload.get("metrics", []):
                metrics[m["name"]] = float(m["value"])
            if not metrics and "metrics_map" in payload:
                metrics = {k: float(v) for k, v in payload["metrics_map"].items()}
            sample = InternalSample(
                device_id=payload.get("device_id", self.device_id),
                source=self.name,
                ts=datetime.now(UTC),
                metrics=metrics,
            )
            self._loop.call_soon_threadsafe(self._queue.put_nowait, sample)
        except Exception:
            log.exception("sparkplug decode failed")
