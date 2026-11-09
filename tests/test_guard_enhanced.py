"""资源护栏增强 + 三级降级矩阵单元测试。

覆盖：
- GuardState 新字段默认值与赋值
- ResourceGuard.full_snapshot() / check_network() / check_offline_duration()
- degrade_level() 各降级级别计算
- DegradePolicy.evaluate() 各种状态组合
- DegradeManager 降级/恢复/hysteresis/历史
- actions_for_level() 各级别动作
- 边界/异常路径
- 温度降频联动、离线降级

psutil / socket / time 通过 monkeypatch mock，只测纯逻辑。
"""
from __future__ import annotations

from collections import namedtuple
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from edge.guard import GuardState, ResourceGuard
from edge.guard_degrade import (
    DegradeAction,
    DegradeDecision,
    DegradeLevel,
    DegradeManager,
    DegradePolicy,
    DegradeThresholds,
)

TempReading = namedtuple("TempReading", ["current", "high", "critical"])


# ---------------------------------------------------------------------------
# fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_psutil(monkeypatch):
    """替换 edge.guard.psutil 为 MagicMock。"""
    mock = MagicMock()
    monkeypatch.setattr("edge.guard.psutil", mock)
    return mock


def _set_mem(mock_psutil, mb: int) -> None:
    mock_psutil.Process.return_value.memory_info.return_value.rss = mb * 1024 * 1024


def _set_disk(mock_psutil, mb: int) -> None:
    mock_psutil.disk_usage.return_value.free = mb * 1024 * 1024


def _set_temp(mock_psutil, temp_c: float) -> None:
    mock_psutil.sensors_temperatures.return_value = {
        "cpu": [TempReading(current=temp_c, high=85, critical=100)]
    }


def _no_temp(mock_psutil) -> None:
    mock_psutil.sensors_temperatures.return_value = {}


def _make_guard(
    tmp_path,
    warn=3000,
    off=3600,
    disk_min=500,
    temp=80.0,
    cloud_host="127.0.0.1",
    cloud_port=1883,
) -> ResourceGuard:
    return ResourceGuard(
        data_dir=tmp_path,
        warn_mb=warn,
        llm_off_mb=off,
        disk_min_mb=disk_min,
        temp_c=temp,
        cloud_host=cloud_host,
        cloud_port=cloud_port,
    )


def _make_state(
    mem: int = 2000,
    disk: int = 10000,
    temp: float | None = None,
    offline: int = 0,
    net: bool = True,
    cpu_percent: float = 0.0,
) -> GuardState:
    """直接构建 GuardState，用于 DegradePolicy/DegradeManager 纯逻辑测试。"""
    return GuardState(
        level="normal",
        mem_used_mb=mem,
        disk_free_mb=disk,
        cpu_temp_c=temp,
        llm_allowed=True,
        t2_full=True,
        accept_models=True,
        cpu_percent=cpu_percent,
        net_connected=net,
        offline_duration_s=offline,
        degrade_actions=[],
    )


def _default_thresholds() -> DegradeThresholds:
    return DegradeThresholds(
        mem_normal_mb=3000,
        mem_llm_small_mb=3000,
        mem_llm_off_mb=3600,
        temp_throttle_c=80.0,
        disk_critical_mb=500,
        offline_degraded_hours=24,
    )


# ===========================================================================
# 1. GuardState 新字段
# ===========================================================================


def test_guard_state_new_fields_defaults():
    """新字段应有合理默认值，保证向后兼容。"""
    st = GuardState(
        level="normal",
        mem_used_mb=1000,
        disk_free_mb=5000,
        cpu_temp_c=None,
        llm_allowed=True,
        t2_full=True,
        accept_models=True,
    )
    assert st.cpu_percent == 0.0
    assert st.net_connected is True
    assert st.offline_duration_s == 0
    assert st.degrade_actions == []


def test_guard_state_new_fields_set():
    """新字段可显式赋值。"""
    st = _make_state(mem=3500, disk=300, temp=85.0, offline=100000, net=False, cpu_percent=55.5)
    assert st.cpu_percent == 55.5
    assert st.net_connected is False
    assert st.offline_duration_s == 100000
    assert st.degrade_actions == []


def test_guard_state_degrade_actions_default_independent():
    """degrade_actions 默认值互不影响（default_factory=list）。"""
    st1 = GuardState(
        level="normal", mem_used_mb=1, disk_free_mb=1, cpu_temp_c=None,
        llm_allowed=True, t2_full=True, accept_models=True,
    )
    st2 = GuardState(
        level="normal", mem_used_mb=1, disk_free_mb=1, cpu_temp_c=None,
        llm_allowed=True, t2_full=True, accept_models=True,
    )
    st1.degrade_actions.append("x")
    assert st2.degrade_actions == []


# ===========================================================================
# 2. ResourceGuard.full_snapshot()
# ===========================================================================


def test_full_snapshot_basic(mock_psutil, tmp_path, monkeypatch):
    """full_snapshot 返回含增强字段的 GuardState。"""
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    mock_psutil.cpu_percent.return_value = 12.5
    monkeypatch.setattr("edge.guard.socket.create_connection", MagicMock())
    g = _make_guard(tmp_path)
    st = g.full_snapshot()
    assert isinstance(st, GuardState)
    assert st.mem_used_mb == 2000
    assert st.cpu_percent == 12.5
    assert st.level == "normal"
    assert st.degrade_actions == []


def test_full_snapshot_includes_network_connected(mock_psutil, tmp_path, monkeypatch):
    """full_snapshot 反映网络连接成功。"""
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    mock_psutil.cpu_percent.return_value = 0.0
    monkeypatch.setattr("edge.guard.socket.create_connection", MagicMock())
    g = _make_guard(tmp_path)
    st = g.full_snapshot()
    assert st.net_connected is True


def test_full_snapshot_includes_network_disconnected(mock_psutil, tmp_path, monkeypatch):
    """full_snapshot 反映网络连接失败。"""
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    mock_psutil.cpu_percent.return_value = 0.0

    def _fail(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr("edge.guard.socket.create_connection", _fail)
    g = _make_guard(tmp_path)
    st = g.full_snapshot()
    assert st.net_connected is False


def test_full_snapshot_offline_duration_zero_initial(mock_psutil, tmp_path, monkeypatch):
    """首次 full_snapshot 离线时长为 0（从未在线）。"""
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    mock_psutil.cpu_percent.return_value = 0.0
    monkeypatch.setattr("edge.guard.socket.create_connection", MagicMock())
    g = _make_guard(tmp_path)
    st = g.full_snapshot()
    assert st.offline_duration_s == 0


def test_full_snapshot_degrade_actions_populated(mock_psutil, tmp_path, monkeypatch):
    """内存超阈值时 full_snapshot 填充 degrade_actions。"""
    _set_mem(mock_psutil, 3700)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    mock_psutil.cpu_percent.return_value = 0.0
    monkeypatch.setattr("edge.guard.socket.create_connection", MagicMock())
    g = _make_guard(tmp_path)
    st = g.full_snapshot()
    assert st.level == "llm_off"
    assert "unload_llm" in st.degrade_actions


# ===========================================================================
# 3. 网络检查 / 离线时长
# ===========================================================================


def test_check_network_success(mock_psutil, tmp_path, monkeypatch):
    """网络连接成功返回 True 并更新时间戳。"""
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    monkeypatch.setattr("edge.guard.socket.create_connection", MagicMock())
    monkeypatch.setattr("edge.guard.time.time", lambda: 1000.0)
    g = _make_guard(tmp_path)
    assert g.check_network() is True
    assert g._last_online_at == 1000.0


def test_check_network_failure(mock_psutil, tmp_path, monkeypatch):
    """网络连接失败返回 False 且不更新时间戳。"""
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)

    def _fail(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr("edge.guard.socket.create_connection", _fail)
    g = _make_guard(tmp_path)
    assert g.check_network() is False
    assert g._last_online_at is None


def test_check_offline_duration_initial_zero(mock_psutil, tmp_path):
    """从未在线时离线时长为 0。"""
    g = _make_guard(tmp_path)
    assert g.check_offline_duration() == 0


def test_check_offline_duration_after_offline(mock_psutil, tmp_path, monkeypatch):
    """离线后时长 = now - last_online_at。"""
    monkeypatch.setattr("edge.guard.time.time", lambda: 1100.0)
    g = _make_guard(tmp_path)
    g._last_online_at = 100.0
    assert g.check_offline_duration() == 1000


# ===========================================================================
# 4. degrade_level() 各降级级别
# ===========================================================================


def test_degrade_level_normal(mock_psutil, tmp_path):
    g = _make_guard(tmp_path)
    assert g.degrade_level(_make_state(mem=2000, disk=10000)) == DegradeLevel.NORMAL


def test_degrade_level_llm_small(mock_psutil, tmp_path):
    g = _make_guard(tmp_path, warn=3000, off=3600)
    assert g.degrade_level(_make_state(mem=3300, disk=10000)) == DegradeLevel.LLM_SMALL


def test_degrade_level_llm_off(mock_psutil, tmp_path):
    g = _make_guard(tmp_path, warn=3000, off=3600)
    assert g.degrade_level(_make_state(mem=3700, disk=10000)) == DegradeLevel.LLM_OFF


def test_degrade_level_temp_throttle(mock_psutil, tmp_path):
    g = _make_guard(tmp_path, temp=80.0)
    assert g.degrade_level(_make_state(mem=2000, disk=10000, temp=85.0)) == DegradeLevel.TEMP_THROTTLE


def test_degrade_level_offline_degraded(mock_psutil, tmp_path):
    g = _make_guard(tmp_path)
    assert g.degrade_level(_make_state(mem=2000, disk=10000, offline=25 * 3600)) == DegradeLevel.OFFLINE_DEGRADED


def test_degrade_level_disk_critical(mock_psutil, tmp_path):
    g = _make_guard(tmp_path, disk_min=500)
    assert g.degrade_level(_make_state(mem=2000, disk=300)) == DegradeLevel.DISK_CRITICAL


def test_degrade_level_emergency(mock_psutil, tmp_path):
    """LLM_OFF + DISK_CRITICAL 同时触发 → EMERGENCY。"""
    g = _make_guard(tmp_path, warn=3000, off=3600, disk_min=500)
    st = _make_state(mem=3700, disk=300)
    assert g.degrade_level(st) == DegradeLevel.EMERGENCY


# ===========================================================================
# 5. DegradePolicy.evaluate() 各种状态组合
# ===========================================================================


def test_policy_evaluate_normal():
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000))
    assert d.level == DegradeLevel.NORMAL
    assert d.actions == []
    assert "正常" in d.reason


def test_policy_evaluate_llm_small():
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3300, disk=10000))
    assert d.level == DegradeLevel.LLM_SMALL
    assert len(d.actions) == 2


def test_policy_evaluate_llm_off():
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3700, disk=10000))
    assert d.level == DegradeLevel.LLM_OFF
    assert any(a.name == "unload_llm" for a in d.actions)


def test_policy_evaluate_temp_throttle():
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, temp=85.0))
    assert d.level == DegradeLevel.TEMP_THROTTLE


def test_policy_evaluate_offline_degraded():
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, offline=30 * 3600))
    assert d.level == DegradeLevel.OFFLINE_DEGRADED


def test_policy_evaluate_disk_critical():
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=400))
    assert d.level == DegradeLevel.DISK_CRITICAL


def test_policy_evaluate_emergency_llm_off_plus_disk():
    """LLM_OFF + DISK_CRITICAL → EMERGENCY。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3700, disk=400))
    assert d.level == DegradeLevel.EMERGENCY
    assert len(d.actions) == 3


def test_policy_evaluate_emergency_llm_off_plus_offline():
    """LLM_OFF + OFFLINE_DEGRADED → EMERGENCY。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3700, disk=10000, offline=30 * 3600))
    assert d.level == DegradeLevel.EMERGENCY


def test_policy_evaluate_temp_plus_llm_off_not_emergency():
    """LLM_OFF + TEMP_THROTTLE → 取最高 TEMP_THROTTLE（TEMP 非高危，不升级 EMERGENCY）。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3700, disk=10000, temp=90.0))
    assert d.level == DegradeLevel.TEMP_THROTTLE


def test_policy_decision_has_timestamp():
    """DegradeDecision 包含有效 timestamp。"""
    policy = DegradePolicy(_default_thresholds())
    before = datetime.now()
    d = policy.evaluate(_make_state())
    after = datetime.now()
    assert before <= d.timestamp <= after


# ===========================================================================
# 6. DegradeManager 降级/恢复/hysteresis/历史
# ===========================================================================


def test_manager_initial_level_normal():
    mgr = DegradeManager(DegradePolicy(_default_thresholds()))
    assert mgr.current_level == DegradeLevel.NORMAL


def test_manager_upgrade_immediate():
    """升级降级立即触发。"""
    mgr = DegradeManager(DegradePolicy(_default_thresholds()))
    d = mgr.evaluate(_make_state(mem=3700, disk=10000))
    assert d.level == DegradeLevel.LLM_OFF
    assert mgr.current_level == DegradeLevel.LLM_OFF


def test_manager_recover_with_hysteresis():
    """恢复需低于阈值 × (1 - margin)，满足后恢复。"""
    thresholds = _default_thresholds()  # mem_llm_off_mb=3600
    mgr = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)
    # 先升级到 LLM_OFF
    mgr.evaluate(_make_state(mem=3700, disk=10000))
    assert mgr.current_level == DegradeLevel.LLM_OFF
    # 内存降到 3300（高于 3600*0.9=3240，不满足恢复）→ 保持
    d = mgr.evaluate(_make_state(mem=3300, disk=10000))
    assert mgr.current_level == DegradeLevel.LLM_OFF
    assert "hysteresis" in d.reason
    # 内存降到 2500（低于 3240 满足 hysteresis；且 <3000 不触发 LLM_SMALL）→ 恢复到 NORMAL
    mgr.evaluate(_make_state(mem=2500, disk=10000))
    assert mgr.current_level == DegradeLevel.NORMAL


def test_manager_no_recover_without_margin():
    """未达恢复阈值时保持当前降级级别。"""
    thresholds = _default_thresholds()
    mgr = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)
    mgr.evaluate(_make_state(mem=3700, disk=10000))  # LLM_OFF
    # 3500 > 3600*0.9=3240 → 不恢复
    mgr.evaluate(_make_state(mem=3500, disk=10000))
    assert mgr.current_level == DegradeLevel.LLM_OFF


def test_manager_recover_temp_throttle():
    """温度降频恢复需低于阈值 × (1 - margin)。"""
    thresholds = _default_thresholds()  # temp_throttle_c=80
    mgr = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)
    mgr.evaluate(_make_state(mem=2000, disk=10000, temp=85.0))
    assert mgr.current_level == DegradeLevel.TEMP_THROTTLE
    # 75 > 80*0.9=72 → 不恢复
    mgr.evaluate(_make_state(mem=2000, disk=10000, temp=75.0))
    assert mgr.current_level == DegradeLevel.TEMP_THROTTLE
    # 70 < 72 → 恢复
    mgr.evaluate(_make_state(mem=2000, disk=10000, temp=70.0))
    assert mgr.current_level == DegradeLevel.NORMAL


def test_manager_recover_disk_critical():
    """磁盘危急恢复需高于阈值 × (1 + margin)。"""
    thresholds = _default_thresholds()  # disk_critical_mb=500
    mgr = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)
    mgr.evaluate(_make_state(mem=2000, disk=400))
    assert mgr.current_level == DegradeLevel.DISK_CRITICAL
    # 520 < 500*1.1=550 → 不恢复
    mgr.evaluate(_make_state(mem=2000, disk=520))
    assert mgr.current_level == DegradeLevel.DISK_CRITICAL
    # 600 > 550 → 恢复
    mgr.evaluate(_make_state(mem=2000, disk=600))
    assert mgr.current_level == DegradeLevel.NORMAL


def test_manager_history():
    """history() 返回所有评估记录。"""
    mgr = DegradeManager(DegradePolicy(_default_thresholds()))
    mgr.evaluate(_make_state(mem=2000, disk=10000))
    mgr.evaluate(_make_state(mem=3700, disk=10000))
    mgr.evaluate(_make_state(mem=2000, disk=10000))
    hist = mgr.history()
    assert len(hist) == 3
    assert hist[0].level == DegradeLevel.NORMAL
    assert hist[1].level == DegradeLevel.LLM_OFF


def test_manager_history_is_copy():
    """history() 返回副本，修改不影响内部状态。"""
    mgr = DegradeManager(DegradePolicy(_default_thresholds()))
    mgr.evaluate(_make_state())
    hist = mgr.history()
    hist.clear()
    assert len(mgr.history()) == 1


def test_manager_apply():
    """apply() 手动应用决策并更新当前级别。"""
    mgr = DegradeManager(DegradePolicy(_default_thresholds()))
    decision = DegradeDecision(
        level=DegradeLevel.LLM_OFF,
        actions=[],
        reason="manual",
        timestamp=datetime.now(),
    )
    mgr.apply(decision)
    assert mgr.current_level == DegradeLevel.LLM_OFF
    assert len(mgr.history()) == 1


def test_manager_emergency_recover_requires_all():
    """EMERGENCY 恢复需内存+磁盘+离线全部满足。"""
    thresholds = _default_thresholds()
    mgr = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)
    # 触发 EMERGENCY（LLM_OFF + DISK_CRITICAL）
    mgr.evaluate(_make_state(mem=3700, disk=400))
    assert mgr.current_level == DegradeLevel.EMERGENCY
    # 仅内存恢复，磁盘未恢复 → 不恢复
    mgr.evaluate(_make_state(mem=2000, disk=400))
    assert mgr.current_level == DegradeLevel.EMERGENCY
    # 内存+磁盘都恢复 → 恢复
    mgr.evaluate(_make_state(mem=2000, disk=600))
    assert mgr.current_level == DegradeLevel.NORMAL


# ===========================================================================
# 7. actions_for_level() 各级别动作
# ===========================================================================


def test_actions_for_normal_empty():
    policy = DegradePolicy(_default_thresholds())
    assert policy.actions_for_level(DegradeLevel.NORMAL) == []


def test_actions_for_llm_small():
    policy = DegradePolicy(_default_thresholds())
    actions = policy.actions_for_level(DegradeLevel.LLM_SMALL)
    names = [a.name for a in actions]
    assert "limit_llm_tokens" in names
    assert "use_small_llm" in names


def test_actions_for_emergency():
    policy = DegradePolicy(_default_thresholds())
    actions = policy.actions_for_level(DegradeLevel.EMERGENCY)
    names = [a.name for a in actions]
    assert "t1_alert_only" in names
    assert "unload_llm" in names
    assert "degrade_t2" in names


def test_actions_for_disk_critical():
    policy = DegradePolicy(_default_thresholds())
    actions = policy.actions_for_level(DegradeLevel.DISK_CRITICAL)
    names = [a.name for a in actions]
    assert "reject_model_download" in names
    assert "clear_cache" in names


def test_actions_have_affected_components():
    """所有动作的 affected_components 非空。"""
    policy = DegradePolicy(_default_thresholds())
    for level in DegradeLevel:
        for action in policy.actions_for_level(level):
            assert len(action.affected_components) > 0


# ===========================================================================
# 8. 边界 / 异常路径
# ===========================================================================


def test_boundary_mem_exact_llm_off_threshold():
    """内存恰好等于卸载阈值 → LLM_OFF（>= 判定）。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3600, disk=10000))
    assert d.level == DegradeLevel.LLM_OFF


def test_boundary_mem_exact_llm_small_threshold():
    """内存恰好等于限定阈值 → LLM_SMALL（>= 判定）。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=3000, disk=10000))
    assert d.level == DegradeLevel.LLM_SMALL


def test_boundary_temp_exact_threshold_not_triggered():
    """温度恰好等于阈值 → 不触发（> 严格判定）。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, temp=80.0))
    assert d.level == DegradeLevel.NORMAL


def test_boundary_disk_exact_threshold_not_triggered():
    """磁盘恰好等于阈值 → 不触发（< 严格判定）。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=500))
    assert d.level == DegradeLevel.NORMAL


def test_boundary_offline_exact_threshold_not_triggered():
    """离线恰好等于阈值 → 不触发（> 严格判定）。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, offline=24 * 3600))
    assert d.level == DegradeLevel.NORMAL


def test_temp_none_does_not_trigger_throttle():
    """cpu_temp_c 为 None 时不触发温度降频。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, temp=None))
    assert d.level == DegradeLevel.NORMAL


# ===========================================================================
# 9. 温度降频联动
# ===========================================================================


def test_temp_throttle_action_closes_llm():
    """温度降频动作包含关闭 LLM。"""
    policy = DegradePolicy(_default_thresholds())
    actions = policy.actions_for_level(DegradeLevel.TEMP_THROTTLE)
    names = [a.name for a in actions]
    assert "close_llm_on_temp" in names
    assert "throttle_llm" in names


def test_temp_throttle_with_normal_mem():
    """内存正常但温度过高 → TEMP_THROTTLE。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, temp=95.0))
    assert d.level == DegradeLevel.TEMP_THROTTLE
    assert "温度" in d.reason


# ===========================================================================
# 10. 离线降级
# ===========================================================================


def test_offline_degraded_action_t1_only():
    """离线降级动作包含仅保留 T1 告警链路。"""
    policy = DegradePolicy(_default_thresholds())
    actions = policy.actions_for_level(DegradeLevel.OFFLINE_DEGRADED)
    names = [a.name for a in actions]
    assert "t1_alert_only" in names
    assert "degrade_t2_t3" in names


def test_offline_degraded_with_disk_emergency():
    """离线 + 磁盘同时触发 → EMERGENCY。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=400, offline=30 * 3600))
    assert d.level == DegradeLevel.EMERGENCY


def test_offline_just_below_threshold_normal():
    """离线略低于阈值 → NORMAL。"""
    policy = DegradePolicy(_default_thresholds())
    d = policy.evaluate(_make_state(mem=2000, disk=10000, offline=24 * 3600 - 1))
    assert d.level == DegradeLevel.NORMAL


# ===========================================================================
# 11. DegradeThresholds 默认值 & DegradeAction dataclass
# ===========================================================================


def test_thresholds_defaults():
    t = DegradeThresholds()
    assert t.mem_normal_mb == 3072
    assert t.mem_llm_small_mb == 3072
    assert t.mem_llm_off_mb == 3600
    assert t.temp_throttle_c == 80.0
    assert t.disk_critical_mb == 500
    assert t.offline_degraded_hours == 24


def test_degrade_action_dataclass():
    a = DegradeAction(name="x", reason="y", affected_components=["c1"])
    assert a.name == "x"
    assert a.reason == "y"
    assert a.affected_components == ["c1"]