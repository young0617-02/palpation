from __future__ import annotations

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from edge.settings import ROOT, settings
from mgmt_agent.ota import OtaEngine, OtaState
from mgmt_agent.probe import probe_edge
from mgmt_agent.version import VersionSlots
from shared import topics
from shared.errors import EdgeError
from shared.schemas import ModelManifest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mgmt")

slots = VersionSlots(ROOT / "models")
engine = OtaEngine(
    slots, settings.firmware_version, f"http://127.0.0.1:{settings.edge_api_port}/health"
)
app = FastAPI(title="mgmt-agent", version="1.0.0")


class RolloutBody(BaseModel):
    manifest: ModelManifest
    previous: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    edge = probe_edge(engine.health_url)
    return {
        "ok": True,
        "isolated": True,
        "ota_state": engine.state.value,
        "edge_probe_ok": edge.probe_ok,
        "current": {
            "anomaly": slots.current_version("anomaly"),
            "intent": slots.current_version("intent"),
            "config": "v0001",
        },
        "frozen": sorted(engine.frozen_versions),
    }


@app.post("/ota/apply")
def ota_apply(body: RolloutBody) -> dict[str, Any]:
    previous = body.previous or slots.current_version(
        "anomaly" if body.manifest.runtime == "onnxruntime" else "intent"
    )
    try:
        state = engine.apply(body.manifest, previous)
    except EdgeError as exc:
        raise HTTPException(
            status_code=400, detail={"code": exc.code, "message": exc.message}
        ) from exc
    return {"state": state.value, "error": engine.last_error}


@app.post("/ota/rollback")
def ota_rollback(layer: str, version: str) -> dict[str, str]:
    slots.rollback(layer, version)
    engine.state = OtaState.rolled_back
    return {"state": engine.state.value, "current": version}


def _mqtt_loop() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"mgmt-{settings.gw_id}")

    def on_connect(c, userdata, flags, reason_code, properties=None):
        c.subscribe(topics.manifest(settings.gw_id), qos=1)
        c.subscribe(topics.command(settings.gw_id), qos=1)

    def on_message(c, userdata, msg):
        try:
            if msg.topic == topics.manifest(settings.gw_id):
                manifest = ModelManifest.model_validate_json(msg.payload)
                engine.apply(manifest, slots.current_version("anomaly"))
            elif msg.topic == topics.command(settings.gw_id):
                cmd = json.loads(msg.payload.decode("utf-8"))
                if cmd.get("op") == "rollback":
                    slots.rollback(cmd["layer"], cmd["version"])
        except Exception:
            log.exception("mgmt mqtt handler failed")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(settings.cloud_mqtt_host, settings.cloud_mqtt_port, keepalive=45)
    client.loop_start()


@app.on_event("startup")
def startup() -> None:
    (ROOT / "models" / "anomaly" / "0.9.1").mkdir(parents=True, exist_ok=True)
    if not slots.current_version("anomaly"):
        slots.activate("anomaly", "0.9.1")
    _mqtt_loop()


if __name__ == "__main__":
    uvicorn.run(
        "mgmt_agent.main:app",
        host="0.0.0.0",
        port=int(__import__("os").environ.get("MGMT_API_PORT", "9001")),
    )
