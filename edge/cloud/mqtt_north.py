from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

from shared import topics
from shared.schemas import EdgeEvent, EdgeHealth, EdgeTelemetry

log = logging.getLogger(__name__)


class NorthMqtt:
    def __init__(self, gw_id: str, host: str, port: int):
        self.gw_id = gw_id
        self.host = host
        self.port = port
        self.online = False
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"edge-{gw_id}",
        )
        self._client.will_set(topics.status(gw_id), payload="offline", qos=1, retain=True)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        # 指数退避重连：1s → 120s，断线后自动重连
        self._client.reconnect_delay_set(min_delay=1, max_delay=120)
        self._handlers: dict[str, Callable[[bytes], None]] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        self._client.connect_async(self.host, self.port, keepalive=45)
        self._client.loop_start()

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._client.publish(topics.status(self.gw_id), "offline", qos=1, retain=True)
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        rc = reason_code if isinstance(reason_code, int) else getattr(reason_code, "value", 1)
        self.online = rc == 0
        if self.online:
            client.publish(topics.status(self.gw_id), "online", qos=1, retain=True)
            client.subscribe(topics.event_ack(self.gw_id), qos=1)
            client.subscribe(topics.shadow_desired(self.gw_id), qos=1)
            client.subscribe(topics.manifest(self.gw_id), qos=1)
            client.subscribe(topics.command(self.gw_id), qos=1)
            client.subscribe(topics.diagnosis_cloud(self.gw_id), qos=1)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.online = False

    def _on_message(self, client, userdata, msg):
        handler = self._handlers.get(msg.topic)
        if handler:
            handler(msg.payload)

    def on(self, topic: str, handler: Callable[[bytes], None]) -> None:
        self._handlers[topic] = handler

    def publish_json(
        self, topic: str, payload: dict[str, Any], qos: int = 1, retain: bool = False
    ) -> None:
        self._client.publish(topic, json.dumps(payload, default=str), qos=qos, retain=retain)

    def publish_event(self, event: EdgeEvent) -> None:
        self.publish_json(topics.event(self.gw_id), event.model_dump(mode="json"), qos=1)

    def publish_telemetry(self, tel: EdgeTelemetry) -> None:
        self.publish_json(topics.telemetry(self.gw_id), tel.model_dump(mode="json"), qos=0)

    def publish_health(self, health: EdgeHealth) -> None:
        self.publish_json(
            topics.health(self.gw_id), health.model_dump(mode="json"), qos=0, retain=True
        )

    def publish_reported(self, reported: dict[str, Any]) -> None:
        self.publish_json(topics.shadow_reported(self.gw_id), reported, qos=1, retain=True)

    def publish_preliminary(self, event: EdgeEvent) -> None:
        self.publish_json(
            topics.diagnosis_preliminary(self.gw_id),
            {
                "event_id": event.event_id,
                "preliminary_diagnosis": event.preliminary_diagnosis,
                "preliminary_confidence": event.preliminary_confidence,
                "schema_ver": "1.0",
            },
            qos=1,
        )
