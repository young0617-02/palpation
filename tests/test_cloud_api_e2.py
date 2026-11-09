"""tests/test_cloud_api_e2.py：cloud/main.py E2 模块 REST 端点测试。

覆盖 8 个新增端点 + 1 个现有端点回归：
1. POST /v1/model-factory/jobs            → 创建模型工厂任务
2. GET  /v1/model-factory/jobs/{job_id}   → 查询任务状态
3. POST /v1/rollout/canary                → 创建 Canary 发布计划
4. POST /v1/rollout/expand                → 扩圈
5. POST /v1/rollout/rollback              → 回滚
6. POST /v1/rollout/freeze                → 冻结版本
7. GET  /v1/shadow-eval/{gw_id}           → 影子验证比对
8. GET  /v1/rollout/progress/{plan_id}    → 发布进度
9. 现有端点 /health、/v1/devices/register 不受影响
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cloud.db import Base, RolloutRecord
from cloud.model_factory import JobStatus, ModelFactory
from cloud.rollback import RollbackManager, RollbackPolicy
from cloud.rollout import RolloutOrchestrator, RolloutState
from cloud.shadow_eval import ShadowEvaluator
from shared.schemas import EdgeEvent

# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """构造隔离的 FastAPI TestClient。

    - 内存 sqlite + StaticPool 支持后台线程共享连接（ModelFactory.submit_job）
    - monkeypatch cloud.db.SessionLocal 与 cloud.main 下的模块实例
    - mock MQTT client 避免真实连接
    """
    import cloud.main as main_module
    from cloud import db as db_module
    from cloud import mqtt_bridge
    from cloud.settings import cloud_settings

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine, expire_on_commit=False)

    # monkeypatch SessionLocal（被 cloud.db / cloud.main / cloud.mqtt_bridge 引用）
    monkeypatch.setattr(db_module, "SessionLocal", test_session)
    monkeypatch.setattr(main_module, "SessionLocal", test_session)
    monkeypatch.setattr(mqtt_bridge, "SessionLocal", test_session)
    monkeypatch.setattr(db_module, "engine", engine)

    # artifact_dir 隔离到 tmp_path
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloud_settings, "artifact_dir", artifact_dir)

    # 重建 4 个 E2 模块实例，绑定 test_session 与隔离的 artifact_dir
    factory = ModelFactory(artifact_dir, test_session)
    orchestrator = RolloutOrchestrator(test_session)
    evaluator = ShadowEvaluator(test_session, min_samples=10, stable_window_h=24)
    manager = RollbackManager(test_session, RollbackPolicy())

    monkeypatch.setattr(main_module, "model_factory", factory)
    monkeypatch.setattr(main_module, "rollout_orchestrator", orchestrator)
    monkeypatch.setattr(main_module, "shadow_evaluator", evaluator)
    monkeypatch.setattr(main_module, "rollback_manager", manager)

    # mock MQTT client 避免真实连接
    mock_client = MagicMock()
    monkeypatch.setattr(main_module, "mqtt_client", mock_client)

    with TestClient(main_module.app) as c:
        yield c


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #


def _make_event(event_id: str = "e1", gw_id: str = "gw-001") -> EdgeEvent:
    return EdgeEvent(
        event_id=event_id,
        gw_id=gw_id,
        device_id_hash="abc",
        seq=1,
        detected_at=datetime.now(UTC),
        rule_id="bearing_rms_high",
        severity="medium",
        metric="vibration_rms",
        value=6.0,
        threshold=4.5,
        preliminary_diagnosis="bearing_wear",
    )


def _wait_job_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    """轮询模型工厂任务状态直到 completed/failed，返回最终状态 dict。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/model-factory/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["status"] in (JobStatus.completed.value, JobStatus.failed.value):
            return data
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _seed_devices(client: TestClient, n: int = 10) -> list[str]:
    """通过 /v1/devices/register 注册 n 台设备，返回 gw_id 列表。"""
    gw_ids = []
    for i in range(n):
        gw_id = f"gw-{i:03d}"
        resp = client.post(
            "/v1/devices/register",
            json={"gw_id": gw_id, "site": "lab", "hardware": {"model": "arm64-a"}},
        )
        assert resp.status_code == 200, resp.text
        gw_ids.append(gw_id)
    return gw_ids


# --------------------------------------------------------------------------- #
# 1. POST /v1/model-factory/jobs → 创建任务返回 job_id
# --------------------------------------------------------------------------- #


def test_create_factory_job_returns_job_id(client: TestClient):
    """POST /v1/model-factory/jobs 返回 {job_id}。"""
    resp = client.post(
        "/v1/model-factory/jobs",
        json={
            "model_id": "anomaly-det",
            "target_version": "0.2.0",
            "config": {"distill_temperature": 3.0},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "job_id" in data
    assert isinstance(data["job_id"], str)
    assert len(data["job_id"]) > 0


def test_create_factory_job_missing_fields_returns_400(client: TestClient):
    """缺少必填字段返回 400。"""
    resp = client.post("/v1/model-factory/jobs", json={"model_id": "anomaly-det"})
    assert resp.status_code == 400


def test_create_factory_job_invalid_temperature_returns_400(client: TestClient):
    """蒸馏温度非法返回 400。"""
    resp = client.post(
        "/v1/model-factory/jobs",
        json={
            "model_id": "anomaly-det",
            "target_version": "0.2.0",
            "config": {"distill_temperature": -1.0},
        },
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# 2. GET /v1/model-factory/jobs/{job_id} → 查询任务状态
# --------------------------------------------------------------------------- #


def test_get_factory_job_status(client: TestClient):
    """GET /v1/model-factory/jobs/{job_id} 返回 FactoryJob 信息。"""
    create = client.post(
        "/v1/model-factory/jobs",
        json={"model_id": "anomaly-det", "target_version": "0.2.0"},
    )
    job_id = create.json()["job_id"]
    final = _wait_job_done(client, job_id)

    assert final["job_id"] == job_id
    assert final["model_id"] == "anomaly-det"
    assert final["target_version"] == "0.2.0"
    assert final["status"] == JobStatus.completed.value
    assert final["error"] is None
    # 七阶段全部完成
    assert len(final["stages"]) == 7
    assert final["current_stage"] == "register"


def test_get_factory_job_not_found_returns_404(client: TestClient):
    """不存在的 job_id 返回 404。"""
    resp = client.get("/v1/model-factory/jobs/nonexistent-job-id")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 3. POST /v1/rollout/canary → 创建 Canary 发布计划
# --------------------------------------------------------------------------- #


def test_create_canary_plan(client: TestClient):
    """POST /v1/rollout/canary 创建灰度计划并返回 plan_id。"""
    resp = client.post(
        "/v1/rollout/canary",
        json={
            "model_id": "anomaly-det",
            "version": "0.2.0",
            "ring_config": {
                "rings": [1, 10, 50, 100],
                "stable_window_h": 24,
                "min_sample_size": 5,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["plan_id"] > 0
    assert data["model_id"] == "anomaly-det"
    assert data["version"] == "0.2.0"
    assert data["rings"] == [1, 10, 50, 100]
    # 初始未推进步，can_deploy_to_canary 为 False
    assert data["can_deploy_to_canary"] is False


def test_create_canary_plan_missing_fields_returns_400(client: TestClient):
    """缺少必填字段返回 400。"""
    resp = client.post("/v1/rollout/canary", json={"model_id": "anomaly-det"})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# 4. POST /v1/rollout/expand → 扩圈
# --------------------------------------------------------------------------- #


def test_expand_ring_creates_records(client: TestClient):
    """POST /v1/rollout/expand 扩圈并创建 RolloutRecord。"""
    _seed_devices(client, n=10)
    canary = client.post(
        "/v1/rollout/canary",
        json={"model_id": "anomaly-det", "version": "0.2.0"},
    )
    plan_id = canary.json()["plan_id"]

    resp = client.post(
        "/v1/rollout/expand",
        json={"plan_id": plan_id, "target_percent": 10},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["plan_id"] == plan_id
    assert data["target_percent"] == 10
    # 10 台设备 10% = 1 台
    assert data["count"] >= 1
    assert all(r["state"] == RolloutState.pending.value for r in data["new_records"])


def test_expand_ring_idempotent(client: TestClient):
    """对同一百分比重复扩圈不产生重复记录。"""
    _seed_devices(client, n=10)
    plan_id = client.post(
        "/v1/rollout/canary",
        json={"model_id": "anomaly-det", "version": "0.2.0"},
    ).json()["plan_id"]

    first = client.post(
        "/v1/rollout/expand", json={"plan_id": plan_id, "target_percent": 10}
    ).json()
    second = client.post(
        "/v1/rollout/expand", json={"plan_id": plan_id, "target_percent": 10}
    ).json()
    assert first["count"] >= 1
    assert second["count"] == 0  # 已存在，跳过


def test_expand_ring_invalid_plan_returns_400(client: TestClient):
    """不存在的 plan_id 返回 400。"""
    resp = client.post(
        "/v1/rollout/expand",
        json={"plan_id": 99999, "target_percent": 10},
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# 5. POST /v1/rollout/rollback → 回滚
# --------------------------------------------------------------------------- #


def test_rollback_with_succeeded_history(client: TestClient):
    """有历史 succeeded 版本时触发回滚并更新记录状态。"""
    # 预置一条 succeeded 历史记录作为回滚目标
    import cloud.main as main_module

    with main_module.SessionLocal() as s:
        s.add(
            RolloutRecord(
                gw_id="gw-001",
                model_id="anomaly-det",
                version="0.1.0",
                state="succeeded",
                detail={},
            )
        )
        s.commit()

    resp = client.post(
        "/v1/rollout/rollback",
        json={
            "signal": "probe_failure",
            "gw_id": "gw-001",
            "model_id": "anomaly-det",
            "version": "0.2.0",
            "severity": "immediate",
            "detail": {"reason": "probe failed"},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["should_rollback"] is True
    assert data["frozen_version"] == "0.2.0"
    assert data["target_version"] == "0.1.0"
    assert "record" in data
    assert data["record"]["state"] == "rolled_back"


def test_rollback_alert_only_no_rollback(client: TestClient):
    """alert_only 信号不触发回滚。"""
    resp = client.post(
        "/v1/rollout/rollback",
        json={
            "signal": "input_link_down",
            "gw_id": "gw-001",
            "model_id": "anomaly-det",
            "version": "0.2.0",
            "severity": "alert_only",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["should_rollback"] is False
    assert "record" not in data


def test_rollback_missing_fields_returns_400(client: TestClient):
    """缺少必填字段返回 400。"""
    resp = client.post(
        "/v1/rollout/rollback",
        json={"signal": "probe_failure", "gw_id": "gw-001"},
    )
    assert resp.status_code == 400


def test_rollback_unknown_signal_returns_400(client: TestClient):
    """未知信号返回 400。"""
    resp = client.post(
        "/v1/rollout/rollback",
        json={
            "signal": "unknown_signal",
            "gw_id": "gw-001",
            "model_id": "anomaly-det",
            "version": "0.2.0",
            "severity": "immediate",
        },
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# 6. POST /v1/rollout/freeze → 冻结版本
# --------------------------------------------------------------------------- #


def test_freeze_version(client: TestClient):
    """POST /v1/rollout/freeze 冻结指定版本。"""
    resp = client.post(
        "/v1/rollout/freeze",
        json={"model_id": "anomaly-det", "version": "0.2.0"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_id"] == "anomaly-det"
    assert data["version"] == "0.2.0"
    assert data["frozen"] is True

    # 验证回滚管理器确实冻结了
    import cloud.main as main_module

    assert main_module.rollback_manager.is_frozen("anomaly-det", "0.2.0") is True


def test_freeze_version_missing_fields_returns_400(client: TestClient):
    """缺少必填字段返回 400。"""
    resp = client.post("/v1/rollout/freeze", json={"model_id": "anomaly-det"})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# 7. GET /v1/shadow-eval/{gw_id} → 影子验证比对
# --------------------------------------------------------------------------- #


def test_shadow_eval_compare(client: TestClient):
    """GET /v1/shadow-eval/{gw_id} 返回影子验证比对结果。"""
    # 通过 evaluator 实例预置影子推理记录
    import cloud.main as main_module

    evaluator = main_module.shadow_evaluator
    # baseline 版本 10 条
    for i in range(10):
        evaluator.record_inference(
            "gw-001", "1.0", _make_event(event_id=f"base-{i}"), "bearing_wear",
            latency_ms=10 + i * 10, peak_mem_mb=200, json_parse_ok=True,
        )
    # new 版本 10 条，KPI 更优
    for i in range(10):
        evaluator.record_inference(
            "gw-001", "2.0", _make_event(event_id=f"new-{i}"), "bearing_wear",
            latency_ms=8 + i * 8, peak_mem_mb=180, json_parse_ok=True,
        )

    resp = client.get(
        "/v1/shadow-eval/gw-001",
        params={"new_version": "2.0", "baseline_version": "1.0"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["gw_id"] == "gw-001"
    assert data["new_version"] == "2.0"
    assert data["baseline_version"] == "1.0"
    assert data["sample_size"] == 10
    assert "new_kpi" in data
    assert "baseline_kpi" in data
    assert "confidence_interval" in data
    assert isinstance(data["confidence_interval"], list)
    assert len(data["confidence_interval"]) == 2


def test_shadow_eval_compare_empty_samples(client: TestClient):
    """无影子推理记录时返回 KPI 全 0、passed=False。"""
    resp = client.get(
        "/v1/shadow-eval/gw-001",
        params={"new_version": "2.0", "baseline_version": "1.0"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["sample_size"] == 0
    assert data["passed"] is False
    assert data["new_kpi"]["agreement_rate"] == 0.0


# --------------------------------------------------------------------------- #
# 8. GET /v1/rollout/progress/{plan_id} → 发布进度
# --------------------------------------------------------------------------- #


def test_rollout_progress(client: TestClient):
    """GET /v1/rollout/progress/{plan_id} 返回发布进度 dict。"""
    _seed_devices(client, n=10)
    plan_id = client.post(
        "/v1/rollout/canary",
        json={"model_id": "anomaly-det", "version": "0.2.0"},
    ).json()["plan_id"]
    # 扩圈 10%
    client.post(
        "/v1/rollout/expand", json={"plan_id": plan_id, "target_percent": 10}
    )

    resp = client.get(f"/v1/rollout/progress/{plan_id}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["plan_id"] == plan_id
    assert data["model_id"] == "anomaly-det"
    assert data["version"] == "0.2.0"
    assert data["frozen"] is False
    assert data["current_step"] == "internal_test"
    assert data["current_ring_percent"] == 10
    assert data["total_records"] >= 1
    assert "state_counts" in data
    assert "ring_config" in data
    assert data["ring_config"]["rings"] == [1, 10, 50, 100]


def test_rollout_progress_not_found_returns_404(client: TestClient):
    """不存在的 plan_id 返回 404。"""
    resp = client.get("/v1/rollout/progress/99999")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 9. 现有端点不受影响
# --------------------------------------------------------------------------- #


def test_health_still_works(client: TestClient):
    """/health 端点仍正常工作。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] == "true"


def test_devices_register_still_works(client: TestClient):
    """/v1/devices/register 端点仍正常工作。"""
    resp = client.post(
        "/v1/devices/register",
        json={"gw_id": "gw-existing", "site": "lab"},
    )
    assert resp.status_code == 200
    assert resp.json()["gw_id"] == "gw-existing"


def test_events_endpoint_still_works(client: TestClient):
    """/v1/events 端点仍正常工作（返回列表）。"""
    resp = client.get("/v1/events")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_watermarks_endpoint_still_works(client: TestClient):
    """/v1/watermarks/{gw_id} 端点仍正常工作（未注册设备返回 404）。"""
    resp = client.get("/v1/watermarks/gw-nonexistent")
    assert resp.status_code == 404
