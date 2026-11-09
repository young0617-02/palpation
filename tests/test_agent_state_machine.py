"""端侧 Agent 状态机 + LLM 一级初判增强单元测试（E1-3）。

用 MagicMock 替换 EdgeLLM / FaultChain / LocalKnowledge，不依赖真实模型，
覆盖：preliminary 正常/降级路径、"未经验证"标注、build_event、工具调用、
工业诊断 prompt 构建、推理延迟统计、get_device_status 工具、TOOL_SCHEMA 一致性。
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from edge.agent.state_machine import EdgeAgentRuntime
from edge.agent.tools import TOOL_SCHEMA, build_tools
from edge.stream.rules import RuleHit
from edge.stream.window import SlidingWindow
from shared.schemas import EdgeEvent, InternalSample


# ---------- 辅具 ----------


def _hit() -> RuleHit:
    return RuleHit(
        rule_id="bearing_rms_high",
        metric="vibration_rms",
        value=6.2,
        threshold=4.5,
        severity="high",
        description="驱动端轴承振动 RMS 越限",
    )


def _sample() -> InternalSample:
    return InternalSample(
        device_id="pump-de-01",
        source="modbus",
        ts=datetime.now(timezone.utc),
        metrics={"vibration_rms": 6.2},
    )


def _hops() -> list[dict]:
    return [{"id": "bearing_wear", "actions": ["检查润滑", "检查对中"]}]


def _kb() -> list[dict]:
    return [{"id": "kb-1", "title": "振动越限", "text": "轴承早期磨损特征"}]


def _make_runtime(
    llm_available: bool = True,
    infer_return=None,
    hops=None,
    kb=None,
    tools=None,
):
    llm = MagicMock()
    llm.available = llm_available
    llm.infer.return_value = infer_return
    chain = MagicMock()
    chain.two_hop.return_value = hops if hops is not None else _hops()
    knowledge = MagicMock()
    knowledge.search.return_value = kb if kb is not None else _kb()
    rt = EdgeAgentRuntime(
        tools=tools if tools is not None else {},
        llm=llm,
        chain=chain,
        knowledge=knowledge,
    )
    return rt, llm, chain, knowledge


# ---------- preliminary 正常路径（LLM mock） ----------


def test_preliminary_normal_path_returns_llm_diagnosis():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "轴承早期磨损", "confidence": 0.82}
    )
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "轴承早期磨损" in text
    assert conf == 0.82


def test_preliminary_normal_path_appends_unverified_marker():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "轴承外圈磨损", "confidence": 0.7}
    )
    text, _ = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "端侧初判" in text
    assert "未经验证" in text


def test_preliminary_marker_not_duplicated_when_llm_includes_it():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "磨损 端侧初判，未经验证。", "confidence": 0.7}
    )
    text, _ = rt.preliminary(_sample(), _hit(), degraded=False)
    assert text.count("端侧初判") == 1


def test_preliminary_default_confidence_when_missing():
    rt, _, _, _ = _make_runtime(infer_return={"diagnosis": "诊断结果"})
    _, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.6


# ---------- preliminary 降级路径 ----------


def test_preliminary_degraded_flag_uses_fallback():
    rt, llm, _, _ = _make_runtime()
    text, conf = rt.preliminary(_sample(), _hit(), degraded=True)
    assert conf == 0.55
    assert "疑似bearing_wear" in text
    assert "端侧初判" in text
    llm.infer.assert_not_called()


def test_preliminary_llm_unavailable_uses_fallback():
    rt, llm, _, _ = _make_runtime(llm_available=False)
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.6
    assert "疑似bearing_wear" in text
    assert "端侧初判" in text
    llm.infer.assert_not_called()


def test_preliminary_llm_returns_none_falls_back():
    rt, _, _, _ = _make_runtime(infer_return=None)
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "疑似bearing_wear" in text
    assert "端侧初判" in text
    assert conf == 0.62


# ---------- "未验证"标注 ----------


def test_unverified_marker_present_in_all_paths():
    """正常、降级、LLM 不可用、LLM 无输出四条路径均必须含'未经验证'。"""
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "磨损", "confidence": 0.8}
    )
    t_normal, _ = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "未经验证" in t_normal

    rt, _, _, _ = _make_runtime()
    t_degraded, _ = rt.preliminary(_sample(), _hit(), degraded=True)
    assert "未经验证" in t_degraded

    rt, _, _, _ = _make_runtime(llm_available=False)
    t_unavail, _ = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "未经验证" in t_unavail

    rt, _, _, _ = _make_runtime(infer_return=None)
    t_none, _ = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "未经验证" in t_none


# ---------- build_event 生成正确 EdgeEvent ----------


def test_build_event_generates_correct_edge_event():
    rt, _, _, _ = _make_runtime()
    h = _hit()
    ev = rt.build_event(
        gw_id="gw-001",
        device_id_hash="hash-xyz",
        hit=h,
        evidence_refs=["ref-1", "ref-2"],
        diagnosis="疑似轴承磨损 端侧初判，未经验证。",
        confidence=0.78,
        degraded=False,
    )
    assert isinstance(ev, EdgeEvent)
    assert ev.gw_id == "gw-001"
    assert ev.device_id_hash == "hash-xyz"
    assert ev.rule_id == "bearing_rms_high"
    assert ev.severity == "high"
    assert ev.metric == "vibration_rms"
    assert ev.value == 6.2
    assert ev.threshold == 4.5
    assert ev.preliminary_diagnosis == "疑似轴承磨损 端侧初判，未经验证。"
    assert ev.preliminary_confidence == 0.78
    assert ev.evidence_refs == ["ref-1", "ref-2"]
    assert ev.degraded is False
    assert ev.event_id  # ULID 非空
    assert rt.state.last_event_id == ev.event_id


def test_build_event_degraded_flag_propagated():
    rt, _, _, _ = _make_runtime()
    ev = rt.build_event(
        gw_id="gw-1",
        device_id_hash="h",
        hit=_hit(),
        evidence_refs=[],
        diagnosis=None,
        confidence=None,
        degraded=True,
    )
    assert ev.degraded is True
    assert ev.preliminary_diagnosis is None
    assert ev.preliminary_confidence is None


# ---------- 工具调用正确 ----------


def test_tool_invocation_correct():
    called = {}

    def my_tool(**kwargs):
        called.update(kwargs)

    rt, _, _, _ = _make_runtime(
        infer_return={
            "diagnosis": "需停机检查",
            "confidence": 0.75,
            "tool": {"name": "inspect", "device": "pump-de-01"},
        },
        tools={"inspect": my_tool},
    )
    rt.preliminary(_sample(), _hit(), degraded=False)
    assert called == {"device": "pump-de-01"}


def test_tool_failure_does_not_affect_diagnosis():
    def bad_tool(**kwargs):
        raise RuntimeError("boom")

    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "诊断", "confidence": 0.7, "tool": {"name": "bad"}},
        tools={"bad": bad_tool},
    )
    _, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.7


def test_tool_unknown_name_ignored():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "诊断", "confidence": 0.7, "tool": {"name": "nope"}},
        tools={},
    )
    _, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.7


# ---------- 工业诊断 prompt 构建 ----------


def test_build_prompt_contains_device_info():
    rt, _, _, _ = _make_runtime()
    prompt = rt._build_prompt(_sample(), _hit(), _hops(), _kb())
    assert "pump-de-01" in prompt
    assert "modbus" in prompt


def test_build_prompt_contains_metric_and_threshold():
    rt, _, _, _ = _make_runtime()
    prompt = rt._build_prompt(_sample(), _hit(), _hops(), _kb())
    assert "vibration_rms" in prompt
    assert "6.2" in prompt
    assert "4.5" in prompt
    assert "bearing_rms_high" in prompt


def test_build_prompt_contains_fault_chain_and_kb():
    rt, _, _, _ = _make_runtime()
    prompt = rt._build_prompt(_sample(), _hit(), _hops(), _kb())
    assert "bearing_wear" in prompt
    assert "检查润滑" in prompt
    assert "kb-1" in prompt
    assert "轴承早期磨损特征" in prompt


def test_build_prompt_contains_output_constraint():
    rt, _, _, _ = _make_runtime()
    prompt = rt._build_prompt(_sample(), _hit(), _hops(), _kb())
    assert "JSON" in prompt
    assert "diagnosis" in prompt
    assert "confidence" in prompt
    assert "端侧初判" in prompt


def test_preliminary_passes_built_prompt_to_llm():
    rt, llm, _, _ = _make_runtime(
        infer_return={"diagnosis": "诊断", "confidence": 0.8}
    )
    rt.preliminary(_sample(), _hit(), degraded=False)
    sent_prompt = llm.infer.call_args[0][0]
    assert "pump-de-01" in sent_prompt
    assert "vibration_rms" in sent_prompt
    assert "bearing_wear" in sent_prompt


# ---------- 推理延迟统计 ----------


def test_inference_latency_recorded_on_llm_path():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "诊断", "confidence": 0.8}
    )
    rt.preliminary(_sample(), _hit(), degraded=False)
    assert len(rt.inference_latencies_ms) == 1
    assert rt.inference_latencies_ms[0] >= 0.0


def test_inference_latency_not_recorded_on_degraded():
    rt, _, _, _ = _make_runtime()
    rt.preliminary(_sample(), _hit(), degraded=True)
    assert rt.inference_latencies_ms == []


def test_inference_latency_not_recorded_when_unavailable():
    rt, _, _, _ = _make_runtime(llm_available=False)
    rt.preliminary(_sample(), _hit(), degraded=False)
    assert rt.inference_latencies_ms == []


def test_p95_inference_ms_empty_returns_zero():
    rt, _, _, _ = _make_runtime()
    assert rt.p95_inference_ms() == 0.0


def test_p95_inference_ms_calculation():
    rt, _, _, _ = _make_runtime()
    rt.inference_latencies_ms = [10.0, 20.0, 30.0, 40.0, 100.0]
    # sorted=[10,20,30,40,100], idx=int(0.95*4)=3 -> 40.0
    assert rt.p95_inference_ms() == 40.0


def test_p95_inference_ms_single_sample():
    rt, _, _, _ = _make_runtime()
    rt.inference_latencies_ms = [42.0]
    assert rt.p95_inference_ms() == 42.0


# ---------- get_device_status 工具 ----------


def test_get_device_status_returns_empty_when_no_data():
    window = SlidingWindow()
    tools = build_tools(window, MagicMock(), MagicMock())
    result = tools["get_device_status"]()
    assert result == {"device_status": {}}


def test_get_device_status_returns_metrics_with_data():
    window = SlidingWindow()
    window.push("vibration_rms", 5.0)
    window.push("vibration_rms", 6.2)
    tools = build_tools(window, MagicMock(), MagicMock())
    result = tools["get_device_status"]()
    status = result["device_status"]
    assert "vibration_rms" in status
    assert status["vibration_rms"]["last"] == 6.2
    assert status["vibration_rms"]["count"] == 2
    assert "snapshot_id" in status["vibration_rms"]


def test_get_device_status_with_explicit_metrics():
    window = SlidingWindow()
    window.push("temperature", 75.0)
    tools = build_tools(window, MagicMock(), MagicMock())
    result = tools["get_device_status"](metrics=["temperature", "current"])
    status = result["device_status"]
    assert "temperature" in status
    assert "current" not in status  # 无数据不出现


# ---------- TOOL_SCHEMA 一致性 ----------


def test_tool_schema_matches_build_tools_keys():
    tools = build_tools(SlidingWindow(), MagicMock(), MagicMock())
    schema_names = {entry["name"] for entry in TOOL_SCHEMA}
    assert schema_names == set(tools)


def test_tool_schema_entries_well_formed():
    for entry in TOOL_SCHEMA:
        assert "name" in entry
        assert "description" in entry
        assert "args" in entry
        assert "returns" in entry
        assert isinstance(entry["args"], dict)