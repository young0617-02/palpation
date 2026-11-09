"""EdgeLLM 真实模型集成测试。

现有 tests/test_edge_llm.py 全用 mock，无法覆盖真实推理，导致
"JSON 解析率 = 0%（completion 接口未套 chat template）"这类缺陷在
899 测试全绿时仍被漏过。本文件补上真实推理盲区：

- 当 EDGE_LLM_MODEL 未设置、或对应模型文件不存在时 pytest.skip；
- 存在时用真实模型跑 2 次 infer，断言返回非 None 且包含 diagnosis 字段。

运行（需模型文件存在）:
  EDGE_LLM_MODEL=models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf uv run pytest -q tests/test_edge_llm_real.py
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edge.agent.llm_llamacpp import EdgeLLM
from edge.agent.state_machine import EdgeAgentRuntime
from edge.agent.tools import build_tools
from edge.knowledge.fault_chain import FaultChain
from edge.knowledge.sqlite_vec import LocalKnowledge
from edge.settings import ROOT, settings
from edge.stream.rules import RuleHit
from edge.stream.window import SlidingWindow
from shared.schemas import InternalSample


def _resolve_model_path() -> Path | None:
    """解析模型路径：环境变量优先，其次 settings(.env)；不存在返回 None。"""
    raw = os.environ.get("EDGE_LLM_MODEL") or settings.edge_llm_model
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


_MODEL_PATH = _resolve_model_path()

pytestmark = pytest.mark.skipif(
    _MODEL_PATH is None,
    reason="EDGE_LLM_MODEL 未设置或模型文件不存在；跳过真实模型集成测试",
)


@pytest.fixture(scope="module")
def edge_llm() -> EdgeLLM:
    assert _MODEL_PATH is not None
    llm = EdgeLLM(str(_MODEL_PATH), settings.edge_llm_maxtok, settings.edge_llm_ctx)
    if not llm.available:
        pytest.skip("llama.cpp 模型加载失败，跳过真实推理")
    return llm


def _make_agent(
    llm: EdgeLLM, db_path: Path
) -> tuple[EdgeAgentRuntime, RuleHit, list[dict], list[dict]]:
    """用生产 prompt 构造逻辑装配一个 EdgeAgentRuntime。"""
    hit = RuleHit(
        rule_id="bearing_rms_high",
        metric="vibration_rms",
        value=5.2,
        threshold=4.5,
        severity="high",
        description="驱动端轴承振动 RMS 越限",
    )
    chain = FaultChain(ROOT / "edge" / "knowledge" / "fault_chains.json")
    knowledge = LocalKnowledge(db_path, ROOT / "edge" / "knowledge" / "snippets.json")
    window = SlidingWindow(32)
    tools = build_tools(window, chain, knowledge)
    agent = EdgeAgentRuntime(tools, llm, chain, knowledge)
    hops = chain.two_hop(hit.rule_id)
    kb = knowledge.search(hit.description, top_k=1)
    return agent, hit, hops, kb


def test_real_infer_produces_parsable_diagnosis(tmp_path, edge_llm):
    """真实模型应输出可解析 JSON，且包含 diagnosis 字段（防 chat template 回归）。"""
    agent, hit, hops, kb = _make_agent(edge_llm, tmp_path / "real_knowledge.db")

    def prompt_at(second: int) -> str:
        # ts 不同 -> prompt 唯一 -> 绕开 60s 缓存，确保真实推理
        sample = InternalSample(
            device_id="pump-de-01",
            source="modbus",
            ts=datetime(2026, 1, 1, 0, 0, second, tzinfo=UTC),
            metrics={"vibration_rms": 5.2, "temperature": 84.0, "current": 19.5},
        )
        return agent._build_prompt(sample, hit, hops, kb)

    for i in range(2):
        parsed = edge_llm.infer(prompt_at(i))
        assert parsed is not None, "真实模型未返回可解析 JSON（chat template 可能未生效）"
        assert "diagnosis" in parsed
        assert parsed["diagnosis"]


def test_real_infer_parse_rate_positive(edge_llm):
    """真实推理的解析成功率必须大于 0（回归 0% 静默降级缺陷）。"""
    if edge_llm.json_total == 0:
        pytest.skip("无真实推理计数可评估")
    assert edge_llm.parse_rate() > 0.0