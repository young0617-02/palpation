from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from starlette.responses import Response

from cloud.db import (
    EdgeDevice,
    EdgeEventRow,
    ModelRegistry,
    RolloutPlan,
    RolloutRecord,
    SessionLocal,
    ShadowRow,
    Watermark,
    init_db,
)
from cloud.model_factory import FactoryJob, ModelFactory
from cloud.mqtt_bridge import attach
from cloud.rollback import (
    RollbackManager,
    RollbackPolicy,
    RollbackSignal,
    RollbackTrigger,
)
from cloud.rollout import RingConfig, RolloutOrchestrator
from cloud.settings import cloud_settings
from cloud.shadow_eval import ShadowEvaluator
from shared import topics
from shared.schemas import ModelManifest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cloud")

app = FastAPI(title="cloud-increment", version="1.0.0")
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cloud-increment")

# --------------------------------------------------------------------------- #
# Prometheus 指标
# --------------------------------------------------------------------------- #
CLOUD_EVENTS_RECEIVED_TOTAL = Counter(
    "cloud_events_received_total",
    "Total edge events received",
    ["gw_id"],
)
CLOUD_MODELS_REGISTERED_TOTAL = Counter(
    "cloud_models_registered_total",
    "Total models registered",
    ["model_id", "version"],
)
CLOUD_ROLLOUT_ACTIVE = Gauge(
    "cloud_rollout_active",
    "Active rollout plans",
    ["model_id", "version"],
)
CLOUD_MQTT_CONNECTED = Gauge("cloud_mqtt_connected", "MQTT connection status (1/0)")
CLOUD_FACTORY_JOBS_TOTAL = Counter(
    "cloud_factory_jobs_total",
    "Total model factory jobs",
    ["model_id"],
)
CLOUD_ROLLBACK_TOTAL = Counter(
    "cloud_rollback_total",
    "Total rollback operations",
    ["model_id", "version"],
)

# --------------------------------------------------------------------------- #
# E2 模块实例：模型工厂 / 灰度发布编排 / 影子验证 / 回滚管理
# --------------------------------------------------------------------------- #
model_factory = ModelFactory(cloud_settings.artifact_dir, SessionLocal)
rollout_orchestrator = RolloutOrchestrator(SessionLocal)
shadow_evaluator = ShadowEvaluator(SessionLocal)
rollback_manager = RollbackManager(SessionLocal, RollbackPolicy())


@app.on_event("startup")
def startup() -> None:
    cloud_settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    Path(cloud_settings.database_url.replace("sqlite:///", "")).parent.mkdir(
        parents=True, exist_ok=True
    ) if cloud_settings.database_url.startswith("sqlite") else None
    init_db()
    attach(mqtt_client)
    mqtt_client.connect_async(
        cloud_settings.cloud_mqtt_host, cloud_settings.cloud_mqtt_port, keepalive=45
    )
    mqtt_client.loop_start()
    CLOUD_MQTT_CONNECTED.set(1)


@app.on_event("shutdown")
def shutdown() -> None:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    CLOUD_MQTT_CONNECTED.set(0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"ok": "true", "role": "cloud-increment-stub"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/devices/register")
def register(body: dict[str, Any]) -> dict[str, str]:
    gw_id = body["gw_id"]
    with SessionLocal() as s:
        s.merge(EdgeDevice(gw_id=gw_id, site=body.get("site"), hardware=body.get("hardware", {})))
        if s.get(ShadowRow, gw_id) is None:
            s.add(
                ShadowRow(
                    gw_id=gw_id,
                    desired={
                        "model_version": "0.9.1",
                        "config_version": "v0001",
                        "sampling_policy": "10hz",
                        "upload_policy": "agg-10s",
                    },
                    reported={},
                )
            )
        s.commit()
    mqtt_client.publish(
        topics.shadow_desired(gw_id),
        '{"model_version":"0.9.1","config_version":"v0001"}',
        qos=1,
        retain=True,
    )
    return {"gw_id": gw_id}


@app.get("/v1/events")
def list_events() -> list[dict[str, Any]]:
    with SessionLocal() as s:
        rows = s.query(EdgeEventRow).order_by(EdgeEventRow.id.desc()).limit(100).all()
        return [
            {"event_id": r.event_id, "gw_id": r.gw_id, "seq": r.seq, "status": r.status}
            for r in rows
        ]


@app.get("/v1/watermarks/{gw_id}")
def watermark(gw_id: str) -> dict[str, Any]:
    with SessionLocal() as s:
        row = s.get(Watermark, gw_id)
        if not row:
            raise HTTPException(404, "not found")
        return {"gw_id": gw_id, "last_acked_seq": row.last_acked_seq, "gaps": row.gaps}


@app.post("/v1/models/register")
async def register_model(
    file: UploadFile,
    model_id: str = Form(...),
    version: str = Form(...),
    runtime: str = Form("onnxruntime"),
    quant: str = Form("int8"),
    requires_firmware: str = Form("0.1.0"),
    preproc_ver: str = Form("ae-v1"),
):
    raw = await file.read()
    digest = hashlib.sha256(raw).hexdigest()
    dest = cloud_settings.artifact_dir / f"{model_id}-{version}-{file.filename}"
    dest.write_bytes(raw)
    url = f"http://127.0.0.1:{cloud_settings.cloud_http_port}/v1/models/artifacts/{dest.name}"
    manifest = ModelManifest(
        model_id=model_id,
        version=version,
        artifact_url=url,
        sha256=digest,
        size_bytes=len(raw),
        quant=quant,
        runtime=runtime,
        target_hardware=["x86-avx2"],
        requires_firmware=requires_firmware,
        preproc_ver=preproc_ver,
        metrics_at_register={},
        rollout={"strategy": "ring", "rings": [1, 10, 50, 100], "stable_window_h": "24"},
    )
    with SessionLocal() as s:
        existing = s.query(ModelRegistry).filter_by(model_id=model_id, version=version).first()
        if existing is not None:
            existing.sha256 = digest
            existing.manifest = manifest.model_dump(mode="json")
            existing.artifact_path = str(dest)
        else:
            s.add(
                ModelRegistry(
                    model_id=model_id,
                    version=version,
                    sha256=digest,
                    manifest=manifest.model_dump(mode="json"),
                    artifact_path=str(dest),
                )
            )
        s.commit()
    CLOUD_MODELS_REGISTERED_TOTAL.labels(model_id=model_id, version=version).inc()
    return manifest.model_dump()


@app.get("/v1/models/artifacts/{name}")
def get_artifact(name: str):
    path = cloud_settings.artifact_dir / name
    if not path.exists():
        raise HTTPException(404, "artifact missing")
    return FileResponse(path)


@app.post("/v1/rollout")
def rollout(body: dict[str, Any]) -> dict[str, Any]:
    gw_id = body["gw_id"]
    with SessionLocal() as s:
        row = (
            s.query(ModelRegistry)
            .filter_by(model_id=body["model_id"], version=body["version"])
            .order_by(ModelRegistry.id.desc())
            .first()
        )
        if row is None:
            raise HTTPException(404, "model not registered")
        plan = RolloutPlan(
            model_id=body["model_id"], version=body["version"], strategy="ring", rings=[1]
        )
        s.add(plan)
        s.add(
            RolloutRecord(
                gw_id=gw_id,
                model_id=body["model_id"],
                version=body["version"],
                state="pending",
                detail={},
            )
        )
        shadow = s.get(ShadowRow, gw_id) or ShadowRow(gw_id=gw_id, desired={}, reported={})
        desired = dict(shadow.desired)
        desired["model_version"] = body["version"]
        shadow.desired = desired
        s.merge(shadow)
        s.commit()
        manifest = row.manifest
    mqtt_client.publish(topics.shadow_desired(gw_id), json_dumps(desired), qos=1, retain=True)
    mqtt_client.publish(topics.manifest(gw_id), json_dumps(manifest), qos=1)
    CLOUD_ROLLOUT_ACTIVE.labels(model_id=body["model_id"], version=body["version"]).inc()
    return {"ok": True, "manifest": manifest}


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)


# --------------------------------------------------------------------------- #
# E2 模块 REST 端点
# --------------------------------------------------------------------------- #


def _factory_job_to_dict(job: FactoryJob) -> dict[str, Any]:
    """将 FactoryJob 序列化为 JSON 可输出的 dict。"""
    return {
        "job_id": job.job_id,
        "model_id": job.model_id,
        "target_version": job.target_version,
        "stages": [s.value for s in job.stages],
        "current_stage": job.current_stage.value if job.current_stage else None,
        "status": job.status.value,
        "artifacts": dict(job.artifacts),
        "metrics": dict(job.metrics),
        "error": job.error,
        "config": dict(job.config),
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


@app.post("/v1/model-factory/jobs")
def create_factory_job(body: dict[str, Any]) -> dict[str, str]:
    """创建模型工厂流水线任务。

    Body: {model_id, target_version, config?}
    """
    model_id = body.get("model_id")
    target_version = body.get("target_version")
    if not model_id or not target_version:
        raise HTTPException(400, "model_id and target_version are required")
    config = body.get("config") or {}
    try:
        job_id = model_factory.submit_job(model_id, target_version, config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    CLOUD_FACTORY_JOBS_TOTAL.labels(model_id=model_id).inc()
    return {"job_id": job_id}


@app.get("/v1/model-factory/jobs/{job_id}")
def get_factory_job(job_id: str) -> dict[str, Any]:
    """查询模型工厂任务状态。"""
    try:
        job = model_factory.get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _factory_job_to_dict(job)


@app.post("/v1/rollout/canary")
def create_canary(body: dict[str, Any]) -> dict[str, Any]:
    """创建 Canary 发布计划。

    Body: {model_id, version, gw_ids?, ring_config?}
    """
    model_id = body.get("model_id")
    version = body.get("version")
    if not model_id or not version:
        raise HTTPException(400, "model_id and version are required")
    ring_cfg_dict = body.get("ring_config") or {}
    try:
        ring_config = RingConfig(
            rings=ring_cfg_dict.get("rings", [1, 10, 50, 100]),
            stable_window_h=ring_cfg_dict.get("stable_window_h", 24),
            rollback_window_h=ring_cfg_dict.get("rollback_window_h", 168),
            min_sample_size=ring_cfg_dict.get("min_sample_size", 100),
        )
        plan = rollout_orchestrator.create_plan(model_id, version, ring_config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    can_deploy = rollout_orchestrator.can_deploy_to_canary(plan.id)
    return {
        "plan_id": plan.id,
        "model_id": plan.model_id,
        "version": plan.version,
        "rings": list(plan.rings),
        "can_deploy_to_canary": can_deploy,
    }


@app.post("/v1/rollout/expand")
def expand_ring(body: dict[str, Any]) -> dict[str, Any]:
    """扩圈到目标百分比。

    Body: {plan_id, target_percent}
    """
    plan_id = body.get("plan_id")
    target_percent = body.get("target_percent")
    if plan_id is None or target_percent is None:
        raise HTTPException(400, "plan_id and target_percent are required")
    try:
        records = rollout_orchestrator.expand_ring(int(plan_id), int(target_percent))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "plan_id": int(plan_id),
        "target_percent": int(target_percent),
        "new_records": [
            {
                "id": r.id,
                "gw_id": r.gw_id,
                "model_id": r.model_id,
                "version": r.version,
                "state": r.state,
                "detail": dict(r.detail or {}),
            }
            for r in records
        ],
        "count": len(records),
    }


@app.post("/v1/rollout/rollback")
def rollback(body: dict[str, Any]) -> dict[str, Any]:
    """回滚：评估触发信号并执行回滚。

    Body: {signal, gw_id, model_id, version, severity, detail?}
    """
    signal_str = body.get("signal")
    gw_id = body.get("gw_id")
    model_id = body.get("model_id")
    version = body.get("version")
    severity = body.get("severity")
    if not all([signal_str, gw_id, model_id, version, severity]):
        raise HTTPException(400, "signal/gw_id/model_id/version/severity are required")
    allowed_severities = {"immediate", "delayed", "alert_only"}
    if severity not in allowed_severities:
        raise HTTPException(400, f"severity must be one of {sorted(allowed_severities)}")
    try:
        signal = RollbackSignal(str(signal_str))
    except ValueError as exc:
        raise HTTPException(400, f"unknown signal: {signal_str}") from exc
    trigger = RollbackTrigger(
        signal=signal,
        gw_id=str(gw_id),
        model_id=str(model_id),
        version=str(version),
        severity=severity,
        detail=body.get("detail") or {},
    )
    decision = rollback_manager.evaluate_trigger(trigger)
    result: dict[str, Any] = {
        "should_rollback": decision.should_rollback,
        "layer": decision.layer.value,
        "reason": decision.reason,
        "frozen_version": decision.frozen_version,
        "target_version": decision.target_version,
        "gw_id": decision.gw_id,
        "model_id": decision.model_id,
    }
    if decision.should_rollback:
        try:
            record = rollback_manager.execute_rollback(decision)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        CLOUD_ROLLBACK_TOTAL.labels(
            model_id=decision.model_id, version=decision.frozen_version
        ).inc()
        result["record"] = {
            "id": record.id,
            "gw_id": record.gw_id,
            "model_id": record.model_id,
            "version": record.version,
            "state": record.state,
            "detail": dict(record.detail or {}),
        }
    return result


@app.post("/v1/rollout/freeze")
def freeze_version(body: dict[str, Any]) -> dict[str, Any]:
    """冻结指定版本。

    Body: {model_id, version}
    """
    model_id = body.get("model_id")
    version = body.get("version")
    if not model_id or not version:
        raise HTTPException(400, "model_id and version are required")
    rollback_manager.freeze_version(model_id, version)
    return {"model_id": model_id, "version": version, "frozen": True}


@app.get("/v1/shadow-eval/{gw_id}")
def shadow_eval_compare(gw_id: str, new_version: str, baseline_version: str) -> dict[str, Any]:
    """影子验证：比对 new_version 与 baseline_version 的 KPI。"""
    result = shadow_evaluator.compare(gw_id, new_version, baseline_version)
    return {
        "gw_id": result.gw_id,
        "model_id": result.model_id,
        "new_version": result.new_version,
        "baseline_version": result.baseline_version,
        "new_kpi": {
            "agreement_rate": result.new_kpi.agreement_rate,
            "p95_latency_ms": result.new_kpi.p95_latency_ms,
            "peak_mem_mb": result.new_kpi.peak_mem_mb,
            "json_parse_rate": result.new_kpi.json_parse_rate,
            "false_positive_rate": result.new_kpi.false_positive_rate,
        },
        "baseline_kpi": {
            "agreement_rate": result.baseline_kpi.agreement_rate,
            "p95_latency_ms": result.baseline_kpi.p95_latency_ms,
            "peak_mem_mb": result.baseline_kpi.peak_mem_mb,
            "json_parse_rate": result.baseline_kpi.json_parse_rate,
            "false_positive_rate": result.baseline_kpi.false_positive_rate,
        },
        "sample_size": result.sample_size,
        "confidence_interval": list(result.confidence_interval),
        "passed": result.passed,
    }


@app.get("/v1/rollout/progress/{plan_id}")
def rollout_progress(plan_id: int) -> dict[str, Any]:
    """查询灰度发布进度。"""
    try:
        return rollout_orchestrator.get_rollout_progress(plan_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("cloud.main:app", host="0.0.0.0", port=cloud_settings.cloud_http_port)
