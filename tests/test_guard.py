"""ResourceGuard 资源降级单元测试。

覆盖内存/磁盘/温度阈值边界、降级切换（normal/llm_small/llm_off）、
llm_allowed/t2_full/accept_models 派生标志、温度读取异常兜底。
psutil 被 MagicMock 替换，只测纯逻辑。
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from edge.guard import GuardState, ResourceGuard

TempReading = namedtuple("TempReading", ["current", "high", "critical"])


@pytest.fixture
def mock_psutil(monkeypatch):
    """替换 edge.guard.psutil 为 MagicMock，返回 mock 供各用例配置返回值。"""
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


def _make_guard(tmp_path, warn=3000, off=3600, disk_min=500, temp=85.0):
    return ResourceGuard(
        data_dir=tmp_path,
        warn_mb=warn,
        llm_off_mb=off,
        disk_min_mb=disk_min,
        temp_c=temp,
    )


# --- 降级级别边界 ---

def test_normal_level(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _set_temp(mock_psutil, 50.0)
    st = _make_guard(tmp_path, warn=3000, off=3600).snapshot()
    assert st.level == "normal"
    assert st.llm_allowed is True
    assert st.t2_full is True
    assert st.accept_models is True
    assert st.mem_used_mb == 2000
    assert st.disk_free_mb == 10000
    assert st.cpu_temp_c == 50.0


def test_llm_small_at_warn_boundary(mock_psutil, tmp_path):
    # mem == warn_mb → llm_small（>= 判定）
    _set_mem(mock_psutil, 3000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    st = _make_guard(tmp_path, warn=3000, off=3600).snapshot()
    assert st.level == "llm_small"
    assert st.llm_allowed is True  # 3000 < 3600


def test_llm_small_mid_range(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 3300)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    st = _make_guard(tmp_path, warn=3000, off=3600).snapshot()
    assert st.level == "llm_small"
    assert st.llm_allowed is True


def test_llm_off_at_off_boundary(mock_psutil, tmp_path):
    # mem == llm_off_mb → llm_off（>= 判定），llm_allowed False
    _set_mem(mock_psutil, 3600)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    st = _make_guard(tmp_path, warn=3000, off=3600).snapshot()
    assert st.level == "llm_off"
    assert st.llm_allowed is False  # 3600 < 3600 为 False


def test_llm_off_above_threshold(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 4000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    st = _make_guard(tmp_path, warn=3000, off=3600).snapshot()
    assert st.level == "llm_off"
    assert st.llm_allowed is False


# --- t2_full 边界 ---

def test_t2_full_boundary(mock_psutil, tmp_path):
    # t2_full = mem < llm_off_mb + 200
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    g = _make_guard(tmp_path, warn=3000, off=3600)
    _set_mem(mock_psutil, 3799)  # 3799 < 3800
    assert g.snapshot().t2_full is True
    _set_mem(mock_psutil, 3800)  # 3800 < 3800 为 False
    assert g.snapshot().t2_full is False


# --- accept_models 边界 ---

def test_accept_models_boundary(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _no_temp(mock_psutil)
    g = _make_guard(tmp_path, warn=3000, off=3600, disk_min=500)
    _set_disk(mock_psutil, 500)  # == disk_min → True（>=）
    assert g.snapshot().accept_models is True
    _set_disk(mock_psutil, 499)
    assert g.snapshot().accept_models is False


# --- 温度读取 ---

def test_temp_reading(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _set_temp(mock_psutil, 72.5)
    st = _make_guard(tmp_path).snapshot()
    assert st.cpu_temp_c == 72.5


def test_temp_none_on_exception(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    mock_psutil.sensors_temperatures.side_effect = RuntimeError("no sensors")
    st = _make_guard(tmp_path).snapshot()
    assert st.cpu_temp_c is None


def test_temp_none_when_empty_dict(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    assert _make_guard(tmp_path).snapshot().cpu_temp_c is None


def test_temp_none_when_empty_list(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    mock_psutil.sensors_temperatures.return_value = {"cpu": []}
    assert _make_guard(tmp_path).snapshot().cpu_temp_c is None


# --- data_dir 不存在时回退到当前目录 ---

def test_data_dir_not_exists_falls_back_to_cwd(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    g = ResourceGuard(
        data_dir=tmp_path / "nonexistent",
        warn_mb=3000,
        llm_off_mb=3600,
        disk_min_mb=500,
        temp_c=85.0,
    )
    st = g.snapshot()
    assert st.level == "normal"
    mock_psutil.disk_usage.assert_called()


def test_snapshot_returns_guard_state_instance(mock_psutil, tmp_path):
    _set_mem(mock_psutil, 2000)
    _set_disk(mock_psutil, 10000)
    _no_temp(mock_psutil)
    st = _make_guard(tmp_path).snapshot()
    assert isinstance(st, GuardState)