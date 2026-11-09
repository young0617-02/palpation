"""ShadowBox 设备影子 desired/reported/delta 对齐单元测试。

覆盖 _delta 计算、set_desired/set_reported 返回 delta、
幂等收敛、snapshot 结构、嵌套字典浅比较。
"""
from __future__ import annotations

from edge.cloud.shadow import ShadowBox, _delta


# --- _delta 纯函数 ---

def test_delta_empty_when_synced():
    d = {"a": 1, "b": 2}
    assert _delta(d, d) == {}


def test_delta_detects_missing_in_reported():
    desired = {"a": 1, "b": 2}
    reported = {"a": 1}
    assert _delta(desired, reported) == {"b": 2}


def test_delta_detects_value_diff():
    desired = {"a": 1}
    reported = {"a": 99}
    assert _delta(desired, reported) == {"a": 1}


def test_delta_ignores_reported_only_keys():
    desired = {"a": 1}
    reported = {"a": 1, "extra": 5}
    assert _delta(desired, reported) == {}


def test_delta_empty_for_empty_desired():
    assert _delta({}, {"a": 1}) == {}


# --- set_desired / set_reported ---

def test_set_desired_returns_delta():
    sb = ShadowBox()
    delta = sb.set_desired({"mode": "auto", "threshold": 80})
    assert delta == {"mode": "auto", "threshold": 80}


def test_set_reported_returns_delta():
    sb = ShadowBox()
    sb.set_desired({"mode": "auto"})
    delta = sb.set_reported({"mode": "auto"})
    assert delta == {}


def test_set_reported_partial_returns_residual_delta():
    sb = ShadowBox()
    sb.set_desired({"a": 1, "b": 2, "c": 3})
    delta = sb.set_reported({"a": 1, "b": 99})
    assert delta == {"b": 2, "c": 3}


def test_idempotent_report_clears_delta():
    sb = ShadowBox()
    sb.set_desired({"mode": "auto", "threshold": 80})
    sb.set_reported({"mode": "auto", "threshold": 80})
    assert sb.snapshot()["delta"] == {}


def test_re_desired_after_sync_recomputes_delta():
    sb = ShadowBox()
    sb.set_desired({"mode": "auto"})
    sb.set_reported({"mode": "auto"})
    assert sb.snapshot()["delta"] == {}
    delta = sb.set_desired({"mode": "manual"})
    assert delta == {"mode": "manual"}


# --- snapshot ---

def test_snapshot_structure():
    sb = ShadowBox()
    sb.set_desired({"mode": "auto"})
    snap = sb.snapshot()
    assert set(snap.keys()) == {"desired", "reported", "delta", "meta"}
    assert snap["desired"] == {"mode": "auto"}
    assert snap["reported"] == {}
    assert snap["delta"] == {"mode": "auto"}
    assert "synced_at" in snap["meta"]


def test_default_shadow_empty():
    sb = ShadowBox()
    snap = sb.snapshot()
    assert snap["desired"] == {}
    assert snap["reported"] == {}
    assert snap["delta"] == {}


# --- 嵌套字典浅比较 ---

def test_nested_dict_equal_not_in_delta():
    desired = {"cfg": {"x": 1}}
    reported = {"cfg": {"x": 1}}
    assert _delta(desired, reported) == {}


def test_nested_dict_diff_enters_delta():
    desired = {"cfg": {"x": 1}}
    reported = {"cfg": {"x": 2}}
    # 浅比较：整个嵌套 dict 作为 desired 值进入 delta
    assert _delta(desired, reported) == {"cfg": {"x": 1}}