"""cloud/rollback.py 单元测试：回滚决策 + 冻结发布 + 三层版本按层回滚。

覆盖：
1. 各触发信号 → severity 和 should_rollback 判定
2. 回滚层级选择 → firmware/model/config 层对应关系
3. 冻结版本 → is_frozen 返回正确
4. 执行回滚 → RolloutRecord 状态更新为 rolling_back/rolled_back
5. 找回滚目标版本 → 返回上一个稳定版本
6. 批量回滚 → 多设备同时回滚
7. 健康趋势检查 → 持续异常检测
8. input_link_down → 不回滚只告警
9. 边界：无历史版本、版本已冻结
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cloud.db import Base, EdgeHeartbeat, RolloutPlan, RolloutRecord
from cloud.rollback import (
    RollbackDecision,
    RollbackManager,
    RollbackPolicy,
    RollbackSignal,
    RollbackTrigger,
    VersionLayer,
)
from shared.schemas import EdgeHealth


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def session_factory():
    """每个用例独立的内存 sqlite session_factory。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def manager(session_factory):
    """默认策略的 RollbackManager。"""
    return RollbackManager(session_factory, RollbackPolicy())


def _now() -> datetime:
    return datetime.now(UTC)


def _make_trigger(
    signal: RollbackSignal,
    *,
    gw_id: str = "gw-001",
    model_id: str = "anomaly-det",
    version: str = "0.2.0",
    severity: str | None = None,
    detail: dict | None = None,
) -> RollbackTrigger:
    """构造 RollbackTrigger，severity 默认从 _SEVERITY_MAP 推断。"""
    from cloud.rollback import _SEVERITY_MAP

    return RollbackTrigger(
        signal=signal,
        gw_id=gw_id,
        model_id=model_id,
        version=version,
        severity=severity or _SEVERITY_MAP[signal],
        detail=detail or {},
        timestamp=_now(),
    )


def _seed_succeeded(session_factory, gw_id: str, model_id: str, version: str):
    """预置一条 state=succeeded 的 RolloutRecord 作为回滚目标。"""
    with session_factory() as s:
        s.add(
            RolloutRecord(
                gw_id=gw_id,
                model_id=model_id,
                version=version,
                state="succeeded",
                detail={},
            )
        )
        s.commit()


def _make_health(
    *,
    gw_id: str = "gw-001",
    probe_ok: bool = True,
    p95_ms: float = 100.0,
    mem_mb: int = 512,
    input_links: dict[str, bool] | None = None,
    ts: datetime | None = None,
) -> EdgeHealth:
    return EdgeHealth(
        gw_id=gw_id,
        ts=ts or _now(),
        firmware_version="1.0.0",
        model_versions={"anomaly-det": "0.2.0"},
        config_version="v0001",
        inference_p95_ms=p95_ms,
        mem_used_mb=mem_mb,
        disk_free_mb=10240,
        cpu_temp_c=50.0,
        input_links=input_links or {"modbus": True, "opcua": True},
        probe_ok=probe_ok,
    )


# --------------------------------------------------------------------------- #
# 1. 各触发信号 → severity 和 should_rollback 判定
# --------------------------------------------------------------------------- #

def test_signal_severity_mapping():
    """每个信号对应正确的 severity。"""
    from cloud.rollback import _SEVERITY_MAP

    assert _SEVERITY_MAP[RollbackSignal.process_start_failure] == "immediate"
    assert _SEVERITY_MAP[RollbackSignal.probe_failure] == "immediate"
    assert _SEVERITY_MAP[RollbackSignal.p95_latency_exceeded] == "delayed"
    assert _SEVERITY_MAP[RollbackSignal.mem_exceeded] == "delayed"
    assert _SEVERITY_MAP[RollbackSignal.json_parse_rate_drop] == "delayed"
    assert _SEVERITY_MAP[RollbackSignal.input_link_down] == "alert_only"


def test_immediate_signal_triggers_rollback(manager, session_factory):
    """immediate 信号（进程启动失败）在有历史版本时触发回滚。"""
    _seed_succeeded(session_factory, "gw-001", "anomaly-det", "0.1.0")
    trigger = _make_trigger(RollbackSignal.process_start_failure)
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is True
    assert decision.target_version == "0.1.0"
    assert decision.frozen_version == "0.2.0"


def test_delayed_signal_triggers_rollback(manager, session_factory):
    """delayed 信号（P95 超阈值）在有历史版本时触发回滚。"""
    _seed_succeeded(session_factory, "gw-001", "anomaly-det", "0.1.0")
    trigger = _make_trigger(RollbackSignal.p95_latency_exceeded)
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is True


def test_alert_only_signal_no_rollback(manager):
    """alert_only 信号（输入链路断开）不触发回滚。"""
    trigger = _make_trigger(RollbackSignal.input_link_down)
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is False
    assert "alert_only" in decision.reason


# --------------------------------------------------------------------------- #
# 2. 回滚层级选择 → firmware/model/config 层对应关系
# --------------------------------------------------------------------------- #

def test_layer_selection_process_start_failure(manager):
    """进程启动失败 → firmware 层。"""
    trigger = _make_trigger(RollbackSignal.process_start_failure)
    assert manager.select_rollback_layer(trigger) == VersionLayer.firmware


def test_layer_selection_probe_failure(manager):
    """探针失败 → firmware 层。"""
    trigger = _make_trigger(RollbackSignal.probe_failure)
    assert manager.select_rollback_layer(trigger) == VersionLayer.firmware


def test_layer_selection_p95_exceeded(manager):
    """P95 超阈值 → model 层（热切换）。"""
    trigger = _make_trigger(RollbackSignal.p95_latency_exceeded)
    assert manager.select_rollback_layer(trigger) == VersionLayer.model


def test_layer_selection_json_parse_drop(manager):
    """JSON 可解析率骤降 → model 层。"""
    trigger = _make_trigger(RollbackSignal.json_parse_rate_drop)
    assert manager.select_rollback_layer(trigger) == VersionLayer.model


def test_layer_selection_mem_exceeded(manager):
    """内存异常 → config 层（秒级生效）。"""
    trigger = _make_trigger(RollbackSignal.mem_exceeded)
    assert manager.select_rollback_layer(trigger) == VersionLayer.config


def test_layer_selection_input_link_down(manager):
    """输入链路断开 → config 层（但 alert_only 不回滚）。"""
    trigger = _make_trigger(RollbackSignal.input_link_down)
    assert manager.select_rollback_layer(trigger) == VersionLayer.config


# --------------------------------------------------------------------------- #
# 3. 冻结版本 → is_frozen 返回正确
# --------------------------------------------------------------------------- #

def test_freeze_version_sets_frozen_flag(manager):
    """freeze_version 后 is_frozen 返回 True。"""
    assert manager.is_frozen("anomaly-det", "0.2.0") is False
    manager.freeze_version("anomaly-det", "0.2.0")
    assert manager.is_frozen("anomaly-det", "0.2.0") is True


def test_is_frozen_returns_false_for_unfrozen(manager):
    """未冻结版本 is_frozen 返回 False。"""
    assert manager.is_frozen("anomaly-det", "0.1.0") is False


def test_freeze_version_idempotent(manager):
    """重复冻结不报错。"""
    manager.freeze_version("anomaly-det", "0.2.0")
    manager.freeze_version("anomaly-det", "0.2.0")
    assert manager.is_frozen("anomaly-det", "0.2.0") is True


def test_get_frozen_versions(manager):
    """get_frozen_versions 返回所有冻结版本。"""
    manager.freeze_version("anomaly-det", "0.2.0")
    manager.freeze_version("intent-llm", "1.0.0")
    frozen = manager.get_frozen_versions()
    assert {"model_id": "anomaly-det", "version": "0.2.0"} in frozen
    assert {"model_id": "intent-llm", "version": "1.0.0"} in frozen
    assert len(frozen) == 2


def test_get_frozen_versions_empty(manager):
    """无冻结版本时返回空列表。"""
    assert manager.get_frozen_versions() == []


def test_freeze_existing_rollout_plan(manager, session_factory):
    """冻结已存在的 RolloutPlan 时更新 frozen 字段。"""
    with session_factory() as s:
        s.add(RolloutPlan(model_id="anomaly-det", version="0.2.0", frozen=False))
        s.commit()
    manager.freeze_version("anomaly-det", "0.2.0")
    with session_factory() as s:
        plan = s.query(RolloutPlan).filter_by(model_id="anomaly-det", version="0.2.0").one()
        assert plan.frozen is True


# --------------------------------------------------------------------------- #
# 4. 执行回滚 → RolloutRecord 状态更新为 rolling_back/rolled_back
# --------------------------------------------------------------------------- #

def test_execute_rollback_updates_existing_record(manager, session_factory):
    """执行回滚更新已有 RolloutRecord 状态为 rolled_back。"""
    with session_factory() as s:
        s.add(
            RolloutRecord(
                gw_id="gw-001",
                model_id="anomaly-det",
                version="0.2.0",
                state="succeeded",
                detail={},
            )
        )
        s.commit()
    _seed_succeeded(session_factory, "gw-001", "anomaly-det", "0.1.0")

    decision = RollbackDecision(
        should_rollback=True,
        layer=VersionLayer.model,
        reason="probe_failure",
        frozen_version="0.2.0",
        target_version="0.1.0",
        gw_id="gw-001",
        model_id="anomaly-det",
    )
    record = manager.execute_rollback(decision)
    assert record.state == "rolled_back"
    assert record.detail["rollback_layer"] == "model"
    assert record.detail["rollback_target"] == "0.1.0"


def test_execute_rollback_creates_record_when_no_active(manager, session_factory):
    """无活跃记录时创建一条 rolled_back 记录。"""
    decision = RollbackDecision(
        should_rollback=True,
        layer=VersionLayer.config,
        reason="mem_exceeded",
        frozen_version="0.2.0",
        target_version="0.1.0",
        gw_id="gw-001",
        model_id="anomaly-det",
    )
    record = manager.execute_rollback(decision)
    assert record.state == "rolled_back"
    assert record.version == "0.1.0"
    assert record.detail["rollback_from"] == "0.2.0"


def test_execute_rollback_freezes_problem_version(manager, session_factory):
    """执行回滚后问题版本被冻结。"""
    decision = RollbackDecision(
        should_rollback=True,
        layer=VersionLayer.firmware,
        reason="process_start_failure",
        frozen_version="0.2.0",
        target_version="0.1.0",
        gw_id="gw-001",
        model_id="anomaly-det",
    )
    manager.execute_rollback(decision)
    assert manager.is_frozen("anomaly-det", "0.2.0") is True


def test_execute_rollback_rejects_no_rollback_decision(manager):
    """should_rollback=False 时 execute_rollback 抛异常。"""
    decision = RollbackDecision(
        should_rollback=False,
        layer=VersionLayer.config,
        reason="alert_only",
        frozen_version=None,
        target_version="",
    )
    with pytest.raises(ValueError, match="should_rollback"):
        manager.execute_rollback(decision)


# --------------------------------------------------------------------------- #
# 5. 找回滚目标版本 → 返回上一个稳定版本
# --------------------------------------------------------------------------- #

def test_find_rollback_target_returns_last_succeeded(manager, session_factory):
    """find_rollback_target 返回最近一次 succeeded 的版本。"""
    with session_factory() as s:
        s.add(
            RolloutRecord(
                gw_id="gw-001",
                model_id="anomaly-det",
                version="0.1.0",
                state="succeeded",
            )
        )
        s.add(
            RolloutRecord(
                gw_id="gw-001",
                model_id="anomaly-det",
                version="0.1.1",
                state="succeeded",
            )
        )
        s.commit()
    target = manager.find_rollback_target("gw-001", VersionLayer.model)
    assert target == "0.1.1"


def test_find_rollback_target_empty_when_no_history(manager):
    """无历史版本时返回空串。"""
    target = manager.find_rollback_target("gw-001", VersionLayer.model)
    assert target == ""


def test_find_rollback_target_ignores_non_succeeded(manager, session_factory):
    """find_rollback_target 只找 state=succeeded 的记录。"""
    with session_factory() as s:
        s.add(
            RolloutRecord(
                gw_id="gw-001",
                model_id="anomaly-det",
                version="0.2.0",
                state="rolling_back",
            )
        )
        s.commit()
    target = manager.find_rollback_target("gw-001", VersionLayer.model)
    assert target == ""


# --------------------------------------------------------------------------- #
# 6. 批量回滚 → 多设备同时回滚
# --------------------------------------------------------------------------- #

def test_batch_rollback_multiple_devices(manager, session_factory):
    """batch_rollback 对每个设备生成一条 rolled_back 记录。"""
    decision = RollbackDecision(
        should_rollback=True,
        layer=VersionLayer.model,
        reason="p95_latency_exceeded",
        frozen_version="0.2.0",
        target_version="0.1.0",
        model_id="anomaly-det",
    )
    gw_ids = ["gw-001", "gw-002", "gw-003"]
    records = manager.batch_rollback(gw_ids, decision)
    assert len(records) == 3
    for r, gw_id in zip(records, gw_ids):
        assert r.state == "rolled_back"
        assert r.gw_id == gw_id


def test_batch_rollback_empty_list(manager):
    """空设备列表返回空记录列表。"""
    decision = RollbackDecision(
        should_rollback=True,
        layer=VersionLayer.model,
        reason="test",
        frozen_version="0.2.0",
        target_version="0.1.0",
    )
    records = manager.batch_rollback([], decision)
    assert records == []


def test_batch_rollback_freezes_version(manager):
    """批量回滚后问题版本被冻结。"""
    decision = RollbackDecision(
        should_rollback=True,
        layer=VersionLayer.model,
        reason="p95_latency_exceeded",
        frozen_version="0.2.0",
        target_version="0.1.0",
        model_id="anomaly-det",
    )
    manager.batch_rollback(["gw-001", "gw-002"], decision)
    assert manager.is_frozen("anomaly-det", "0.2.0") is True


# --------------------------------------------------------------------------- #
# 7. 健康趋势检查 → 持续异常检测
# --------------------------------------------------------------------------- #

def _seed_heartbeats(session_factory, gw_id: str, healths: list[EdgeHealth]):
    """批量写入 EdgeHeartbeat。"""
    with session_factory() as s:
        for h in healths:
            s.add(
                EdgeHeartbeat(
                    gw_id=gw_id,
                    ts=h.ts,
                    payload=h.model_dump(mode="json"),
                )
            )
        s.commit()


def test_check_health_trend_probe_failure(manager, session_factory):
    """探针失败（probe_ok=false）被检测为 immediate 触发。"""
    healths = [_make_health(probe_ok=False, ts=_now())]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = manager.check_health_trend("gw-001", window_min=10)

    assert any(t.signal == RollbackSignal.probe_failure for t in triggers)
    probe_trigger = next(t for t in triggers if t.signal == RollbackSignal.probe_failure)
    assert probe_trigger.severity == "immediate"


def test_check_health_trend_p95_sustained(manager, session_factory):
    """P95 持续超阈值被检测为 delayed 触发。"""
    now = _now()
    # 构造 10 分钟窗口内的心跳，P95 全部超阈值（policy 默认 3000ms 持续 10min）
    # 时间跨度 9 分钟（>= sustained_min * 0.9 = 9min）
    healths = [
        _make_health(p95_ms=3500.0, ts=now - timedelta(minutes=9)),
        _make_health(p95_ms=3600.0, ts=now - timedelta(minutes=5)),
        _make_health(p95_ms=3400.0, ts=now),
    ]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = manager.check_health_trend("gw-001", window_min=15)
    assert any(t.signal == RollbackSignal.p95_latency_exceeded for t in triggers)


def test_check_health_trend_p95_not_sustained(manager, session_factory):
    """P95 偶尔超阈值（未持续）不触发。"""
    now = _now()
    healths = [
        _make_health(p95_ms=100.0, ts=now - timedelta(minutes=5)),
        _make_health(p95_ms=3500.0, ts=now),  # 仅最后一次超阈值
    ]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = manager.check_health_trend("gw-001", window_min=15)
    assert not any(t.signal == RollbackSignal.p95_latency_exceeded for t in triggers)


def test_check_health_trend_mem_sustained(manager, session_factory):
    """内存持续超阈值被检测为 delayed 触发。"""
    now = _now()
    # policy 默认 3072MB 持续 5min
    healths = [
        _make_health(mem_mb=4096, ts=now - timedelta(minutes=6)),
        _make_health(mem_mb=4096, ts=now),
    ]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = manager.check_health_trend("gw-001", window_min=10)
    assert any(t.signal == RollbackSignal.mem_exceeded for t in triggers)


def test_check_health_trend_input_link_down(manager, session_factory):
    """输入链路断开被检测为 alert_only 触发。"""
    healths = [
        _make_health(input_links={"modbus": False, "opcua": True}, ts=_now())
    ]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = manager.check_health_trend("gw-001", window_min=10)
    assert any(t.signal == RollbackSignal.input_link_down for t in triggers)
    link_trigger = next(t for t in triggers if t.signal == RollbackSignal.input_link_down)
    assert link_trigger.severity == "alert_only"


def test_check_health_trend_no_anomaly(manager, session_factory):
    """无异常时返回空列表。"""
    healths = [_make_health(ts=_now())]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = manager.check_health_trend("gw-001", window_min=10)
    assert triggers == []


def test_check_health_trend_no_heartbeats(manager):
    """无心跳数据时返回空列表。"""
    triggers = manager.check_health_trend("gw-001", window_min=10)
    assert triggers == []


def test_check_health_trend_custom_policy(session_factory):
    """自定义策略阈值生效。"""
    now = _now()
    # 用更低的阈值使 P95 触发
    policy = RollbackPolicy(p95_threshold_ms=100.0, p95_sustained_min=5)
    mgr = RollbackManager(session_factory, policy)
    healths = [
        _make_health(p95_ms=150.0, ts=now - timedelta(minutes=6)),
        _make_health(p95_ms=150.0, ts=now),
    ]
    _seed_heartbeats(session_factory, "gw-001", healths)
    triggers = mgr.check_health_trend("gw-001", window_min=10)
    assert any(t.signal == RollbackSignal.p95_latency_exceeded for t in triggers)


# --------------------------------------------------------------------------- #
# 8. input_link_down → 不回滚只告警
# --------------------------------------------------------------------------- #

def test_input_link_down_no_rollback(manager):
    """input_link_down 信号 evaluate_trigger 返回 should_rollback=False。"""
    trigger = _make_trigger(RollbackSignal.input_link_down)
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is False
    assert decision.frozen_version is None
    assert decision.target_version == ""


def test_input_link_down_decision_not_executable(manager):
    """input_link_down 的决策不能执行回滚。"""
    trigger = _make_trigger(RollbackSignal.input_link_down)
    decision = manager.evaluate_trigger(trigger)
    with pytest.raises(ValueError):
        manager.execute_rollback(decision)


# --------------------------------------------------------------------------- #
# 9. 边界：无历史版本、版本已冻结
# --------------------------------------------------------------------------- #

def test_no_history_no_rollback(manager):
    """无历史版本时不回滚。"""
    trigger = _make_trigger(RollbackSignal.probe_failure)
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is False
    assert "no previous succeeded" in decision.reason


def test_frozen_version_no_rollback(manager, session_factory):
    """版本已冻结时不重复回滚（幂等）。"""
    _seed_succeeded(session_factory, "gw-001", "anomaly-det", "0.1.0")
    manager.freeze_version("anomaly-det", "0.2.0")
    trigger = _make_trigger(RollbackSignal.probe_failure, version="0.2.0")
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is False
    assert "already frozen" in decision.reason
    assert decision.frozen_version == "0.2.0"


def test_evaluate_trigger_full_flow(manager, session_factory):
    """完整评估流程：有历史 + 未冻结 + immediate → 触发回滚。"""
    _seed_succeeded(session_factory, "gw-001", "anomaly-det", "0.1.0")
    trigger = _make_trigger(
        RollbackSignal.probe_failure,
        gw_id="gw-001",
        model_id="anomaly-det",
        version="0.2.0",
    )
    decision = manager.evaluate_trigger(trigger)
    assert decision.should_rollback is True
    assert decision.layer == VersionLayer.firmware
    assert decision.target_version == "0.1.0"
    assert decision.frozen_version == "0.2.0"
    assert decision.gw_id == "gw-001"
    assert decision.model_id == "anomaly-det"


# --------------------------------------------------------------------------- #
# RollbackPolicy 默认值
# --------------------------------------------------------------------------- #

def test_policy_default_values():
    """RollbackPolicy 默认值符合 §4.7 示例。"""
    p = RollbackPolicy()
    assert p.p95_threshold_ms == 3000.0
    assert p.p95_sustained_min == 10
    assert p.mem_threshold_mb == 3072
    assert p.mem_sustained_min == 5
    assert p.json_parse_rate_min == 0.80
    assert p.json_parse_sustained_count == 100


def test_manager_default_policy(session_factory):
    """RollbackManager 未传 policy 时使用默认策略。"""
    mgr = RollbackManager(session_factory)
    assert mgr.policy.p95_threshold_ms == 3000.0


# --------------------------------------------------------------------------- #
# 枚举完整性
# --------------------------------------------------------------------------- #

def test_rollback_signal_values():
    """RollbackSignal 包含全部 6 个信号。"""
    assert len(RollbackSignal) == 6
    assert RollbackSignal.process_start_failure in RollbackSignal
    assert RollbackSignal.probe_failure in RollbackSignal
    assert RollbackSignal.p95_latency_exceeded in RollbackSignal
    assert RollbackSignal.mem_exceeded in RollbackSignal
    assert RollbackSignal.json_parse_rate_drop in RollbackSignal
    assert RollbackSignal.input_link_down in RollbackSignal


def test_version_layer_values():
    """VersionLayer 包含 firmware/model/config 三层。"""
    assert len(VersionLayer) == 3
    assert VersionLayer.firmware in VersionLayer
    assert VersionLayer.model in VersionLayer
    assert VersionLayer.config in VersionLayer