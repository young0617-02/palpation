"""EdgeAgentRuntime 状态机与 preliminary 诊断单元测试。

用 MagicMock 替换 EdgeLLM / FaultChain / LocalKnowledge 外部依赖，
只测纯逻辑：降级兜底、LLM 路径、缓存命中/过期、tool 调用、
状态流转、build_event、replay_safe_dump。
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from edge.agent.state_machine import AgentState, EdgeAgentRuntime
from edge.stream.rules import RuleHit
from shared.schemas import EdgeEvent, InternalSample


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
    knowledge.search.return_value = kb if kb is not None else [{"id": "kb-1", "text": "振动越限"}]
    rt = EdgeAgentRuntime(
        tools=tools if tools is not None else {},
        llm=llm,
        chain=chain,
        knowledge=knowledge,
    )
    return rt, llm, chain, knowledge


# --- 降级兜底 ---

def test_degraded_uses_fallback_conf_055():
    rt, llm, _, _ = _make_runtime()
    text, conf = rt.preliminary(_sample(), _hit(), degraded=True)
    assert conf == 0.55
    assert "疑似bearing_wear" in text
    assert "端侧初判" in text
    llm.infer.assert_not_called()


def test_llm_unavailable_uses_fallback_conf_06():
    rt, llm, _, _ = _make_runtime(llm_available=False)
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.6
    assert "疑似bearing_wear" in text
    llm.infer.assert_not_called()


# --- LLM 正常路径 ---

def test_llm_returns_diagnosis():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "轴承早期磨损", "confidence": 0.8}
    )
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "轴承早期磨损" in text
    assert "端侧初判" in text  # 自动追加标记
    assert conf == 0.8


def test_llm_diagnosis_with_marker_not_duplicated():
    rt, _, _, _ = _make_runtime(
        infer_return={"diagnosis": "轴承磨损 端侧初判，未经验证。", "confidence": 0.7}
    )
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert text.count("端侧初判") == 1
    assert conf == 0.7


def test_llm_returns_none_uses_fallback():
    rt, _, _, _ = _make_runtime(infer_return=None)
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "疑似bearing_wear" in text
    assert "端侧初判" in text
    # LLM 无输出时 conf 保留初始值 0.62
    assert conf == 0.62


def test_llm_returns_no_diagnosis_key_uses_fallback():
    rt, _, _, _ = _make_runtime(infer_return={"confidence": 0.9})
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert "疑似bearing_wear" in text
    # LLM 返回无 diagnosis 时 conf 保留初始值 0.62
    assert conf == 0.62


def test_llm_default_confidence_when_missing():
    rt, _, _, _ = _make_runtime(infer_return={"diagnosis": "诊断结果"})
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.6  # parsed.get("confidence", 0.6)


# --- 缓存 ---

def test_cache_hit_skips_recompute():
    rt, llm, chain, _ = _make_runtime(
        infer_return={"diagnosis": "诊断A", "confidence": 0.8}
    )
    s, h = _sample(), _hit()
    t1, c1 = rt.preliminary(s, h, degraded=False)
    t2, c2 = rt.preliminary(s, h, degraded=False)
    assert (t1, c1) == (t2, c2)
    assert chain.two_hop.call_count == 1
    assert llm.infer.call_count == 1


def test_cache_expiry_recomputes():
    rt, llm, chain, _ = _make_runtime(
        infer_return={"diagnosis": "诊断A", "confidence": 0.8}
    )
    s, h = _sample(), _hit()
    rt.preliminary(s, h, degraded=False)
    key = f"{s.device_id}:{h.metric}:{h.rule_id}"
    old_ts, text, conf = rt._cache[key]
    rt._cache[key] = (old_ts - rt.cache_ttl_s - 1, text, conf)
    rt.preliminary(s, h, degraded=False)
    assert chain.two_hop.call_count == 2
    assert llm.infer.call_count == 2


def test_different_cache_keys_independent():
    rt, llm, chain, _ = _make_runtime(
        infer_return={"diagnosis": "诊断", "confidence": 0.8}
    )
    h = _hit()
    s1 = _sample()
    s2 = InternalSample(
        device_id="pump-de-02",
        source="modbus",
        ts=datetime.now(timezone.utc),
        metrics={"vibration_rms": 6.2},
    )
    rt.preliminary(s1, h, degraded=False)
    rt.preliminary(s2, h, degraded=False)
    assert chain.two_hop.call_count == 2


# --- tool 调用 ---

def test_tool_invocation():
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


def test_tool_failure_swallowed():
    def bad_tool(**kwargs):
        raise RuntimeError("boom")

    rt, _, _, _ = _make_runtime(
        infer_return={
            "diagnosis": "诊断",
            "confidence": 0.7,
            "tool": {"name": "bad", "x": 1},
        },
        tools={"bad": bad_tool},
    )
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.7  # tool 失败不影响诊断结果


def test_tool_unknown_name_ignored():
    rt, _, _, _ = _make_runtime(
        infer_return={
            "diagnosis": "诊断",
            "confidence": 0.7,
            "tool": {"name": "nonexistent", "x": 1},
        },
        tools={},
    )
    text, conf = rt.preliminary(_sample(), _hit(), degraded=False)
    assert conf == 0.7


# --- fallback 边界 ---

def test_fallback_unknown_fault_when_hops_missing_id():
    rt, _, _, _ = _make_runtime(hops=[{"actions": None}])
    text, conf = rt.preliminary(_sample(), _hit(), degraded=True)
    assert "疑似未知故障" in text
    assert "现场检查" in text  # actions None → 默认 ['现场检查']
    assert conf == 0.55


# --- 状态流转 / build_event / replay_safe_dump ---

def test_initial_state_idle():
    rt, _, _, _ = _make_runtime()
    assert rt.state.name == "IDLE"
    assert rt.state.last_event_id is None


def test_state_returns_to_idle_after_preliminary():
    rt, _, _, _ = _make_runtime()
    rt.preliminary(_sample(), _hit(), degraded=True)
    assert rt.state.name == "IDLE"


def test_build_event_creates_event_and_updates_state():
    rt, _, _, _ = _make_runtime()
    h = _hit()
    ev = rt.build_event(
        gw_id="gw-1",
        device_id_hash="hash-abc",
        hit=h,
        evidence_refs=["ref-1"],
        diagnosis="疑似轴承磨损",
        confidence=0.6,
        degraded=False,
    )
    assert isinstance(ev, EdgeEvent)
    assert ev.gw_id == "gw-1"
    assert ev.rule_id == "bearing_rms_high"
    assert ev.severity == "high"
    assert ev.preliminary_diagnosis == "疑似轴承磨损"
    assert ev.preliminary_confidence == 0.6
    assert ev.degraded is False
    assert ev.evidence_refs == ["ref-1"]
    assert rt.state.last_event_id == ev.event_id


def test_replay_safe_dump_initial():
    rt, _, _, _ = _make_runtime()
    assert rt.replay_safe_dump() == {"state": "IDLE", "last_event_id": None}


def test_replay_safe_dump_after_event():
    rt, _, _, _ = _make_runtime()
    rt.build_event(
        gw_id="gw-1",
        device_id_hash="h",
        hit=_hit(),
        evidence_refs=[],
        diagnosis=None,
        confidence=None,
        degraded=True,
    )
    dump = rt.replay_safe_dump()
    assert dump["state"] == "IDLE"
    assert dump["last_event_id"] is not None


def test_agent_state_default():
    st = AgentState()
    assert st.name == "IDLE"
    assert st.last_event_id is None