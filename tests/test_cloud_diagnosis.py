"""cloud/diagnosis.py 单元测试：纯 Python 诊断管线。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cloud.diagnosis import depth_diagnosis
from shared.schemas import CloudDiagnosis, EdgeEvent


def _now() -> datetime:
    return datetime.now(UTC)


def _make_event(**overrides):
    base = dict(
        event_id="e1",
        gw_id="gw-001",
        device_id_hash="abc",
        seq=1,
        detected_at=_now(),
        rule_id="bearing_rms_high",
        severity="medium",
        metric="vibration_rms",
        value=6.0,
        threshold=4.5,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# 正常诊断
# --------------------------------------------------------------------------- #

def test_normal_diagnosis():
    """知识库与故障链均可用时返回带引用的完整报告。"""
    event = EdgeEvent(**_make_event())
    diag = depth_diagnosis(event)
    assert isinstance(diag, CloudDiagnosis)
    assert diag.event_id == "e1"
    assert diag.gw_id == "gw-001"
    assert "云端深度诊断" in diag.conclusion
    assert "振动" in diag.conclusion
    assert 0 < diag.confidence <= 0.95
    assert diag.citations
    assert diag.requires_human_review is False


def test_fault_chain_matched_bearing_wear():
    """bearing_rms_high 命中 bearing_wear 故障链。"""
    event = EdgeEvent(**_make_event(rule_id="bearing_rms_high"))
    diag = depth_diagnosis(event)
    assert "bearing_wear" in diag.conclusion
    assert "润滑" in diag.conclusion


def test_fault_chain_matched_overload():
    """current_high 命中 overload 故障链。"""
    event = EdgeEvent(**_make_event(rule_id="current_high", metric="current"))
    diag = depth_diagnosis(event)
    assert "overload" in diag.conclusion


# --------------------------------------------------------------------------- #
# 高风险人工审核
# --------------------------------------------------------------------------- #

def test_high_severity_requires_review():
    """severity=high 时强制人工审核。"""
    event = EdgeEvent(**_make_event(severity="high"))
    diag = depth_diagnosis(event)
    assert diag.requires_human_review is True


def test_medium_severity_no_review():
    """severity=medium 时不要求人工审核。"""
    event = EdgeEvent(**_make_event(severity="medium"))
    diag = depth_diagnosis(event)
    assert diag.requires_human_review is False


def test_low_severity_no_review():
    """severity=low 时不要求人工审核。"""
    event = EdgeEvent(**_make_event(severity="low"))
    diag = depth_diagnosis(event)
    assert diag.requires_human_review is False


# --------------------------------------------------------------------------- #
# 知识库 / 故障链不存在降级
# --------------------------------------------------------------------------- #

def test_knowledge_degradation(monkeypatch, tmp_path):
    """知识库文件不存在时优雅降级：空引用 + 降低置信度。"""
    monkeypatch.setattr("cloud.diagnosis._SNIPPETS_PATH", tmp_path / "nonexistent.json")
    event = EdgeEvent(**_make_event())
    diag = depth_diagnosis(event)
    assert diag.citations == []
    assert diag.confidence < 0.9


def test_fault_chain_degradation(monkeypatch, tmp_path):
    """故障链文件不存在时优雅降级。"""
    monkeypatch.setattr("cloud.diagnosis._FAULT_CHAINS_PATH", tmp_path / "nonexistent.json")
    event = EdgeEvent(**_make_event())
    diag = depth_diagnosis(event)
    assert "未命中" in diag.conclusion
    assert diag.confidence < 0.9


def test_full_degradation(monkeypatch, tmp_path):
    """知识库和故障链均不可用时仍能产出诊断。"""
    monkeypatch.setattr("cloud.diagnosis._SNIPPETS_PATH", tmp_path / "no_kb.json")
    monkeypatch.setattr("cloud.diagnosis._FAULT_CHAINS_PATH", tmp_path / "no_chain.json")
    event = EdgeEvent(**_make_event())
    diag = depth_diagnosis(event)
    assert diag.citations == []
    assert "未命中" in diag.conclusion
    assert diag.confidence == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 各种 metric 类型
# --------------------------------------------------------------------------- #

def test_metric_vibration_rms():
    """vibration_rms 指标分析包含振动关键词。"""
    event = EdgeEvent(**_make_event(metric="vibration_rms"))
    diag = depth_diagnosis(event)
    assert "振动" in diag.conclusion


def test_metric_temperature():
    """temperature 指标分析包含温度关键词。"""
    event = EdgeEvent(**_make_event(metric="temperature", rule_id="temperature_high"))
    diag = depth_diagnosis(event)
    assert "温度" in diag.conclusion


def test_metric_current():
    """current 指标分析包含电流关键词。"""
    event = EdgeEvent(**_make_event(metric="current", rule_id="current_high"))
    diag = depth_diagnosis(event)
    assert "电流" in diag.conclusion


def test_metric_unknown():
    """未知 metric 类型仍能给出通用分析。"""
    event = EdgeEvent(**_make_event(metric="pressure"))
    diag = depth_diagnosis(event)
    assert "pressure" in diag.conclusion


def test_deviation_calculation():
    """偏离度正确反映在结论中。"""
    event = EdgeEvent(**_make_event(value=9.0, threshold=4.5))
    diag = depth_diagnosis(event)
    assert "偏离度 2.00" in diag.conclusion
