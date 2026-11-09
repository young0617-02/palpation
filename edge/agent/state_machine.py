from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ulid import ULID

from edge.agent.llm_llamacpp import EdgeLLM
from edge.knowledge.fault_chain import FaultChain
from edge.knowledge.sqlite_vec import LocalKnowledge
from edge.stream.rules import RuleHit
from shared.schemas import EdgeEvent, InternalSample

log = logging.getLogger(__name__)


@dataclass
class AgentState:
    name: str = "IDLE"
    last_event_id: str | None = None


class EdgeAgentRuntime:
    """Lightweight state machine. Not LangGraph (scheme §2.2)."""

    def __init__(self, tools: dict, llm: EdgeLLM, chain: FaultChain, knowledge: LocalKnowledge):
        self.tools = tools
        self.llm = llm
        self.chain = chain
        self.knowledge = knowledge
        self.state = AgentState()
        self._cache: dict[str, tuple[float, str, float]] = {}
        self.cache_ttl_s = 60.0
        # preliminary() 中端侧 LLM 推理延迟样本（毫秒），用于 p95 统计
        self.inference_latencies_ms: list[float] = []
        self._latency_window = 200

    def preliminary(
        self, sample: InternalSample, hit: RuleHit, degraded: bool
    ) -> tuple[str | None, float | None]:
        self.state.name = "PROCESSING"
        cache_key = f"{sample.device_id}:{hit.metric}:{hit.rule_id}"
        now = datetime.now(UTC).timestamp()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self.cache_ttl_s:
            return cached[1], cached[2]

        hops = self.chain.two_hop(hit.rule_id)
        kb = self.knowledge.search(hit.description, top_k=1)
        actions = "、".join(hops[0].get("actions") or ["现场检查"])
        fallback = f"疑似{hops[0].get('id', '未知故障')}，建议{actions}。端侧初判，未经验证。"
        conf = 0.62
        if degraded or not self.llm.available:
            text, conf = fallback, 0.55 if degraded else 0.6
        else:
            prompt = self._build_prompt(sample, hit, hops, kb)
            t0 = time.perf_counter()
            parsed = self.llm.infer(prompt)
            self.inference_latencies_ms.append((time.perf_counter() - t0) * 1000)
            if parsed and "diagnosis" in parsed:
                text = str(parsed["diagnosis"])
                if "端侧初判" not in text:
                    text += " 端侧初判，未经验证。"
                conf = float(parsed.get("confidence", 0.6))
                tool = parsed.get("tool")
                if isinstance(tool, dict) and tool.get("name") in self.tools:
                    try:
                        self.tools[tool["name"]](**{k: v for k, v in tool.items() if k != "name"})
                    except Exception:
                        log.warning("tool call rejected or failed")
            else:
                text = fallback
        self._cache[cache_key] = (now, text, conf)
        self.state.name = "IDLE"
        return text, conf

    def _build_prompt(
        self,
        sample: InternalSample,
        hit: RuleHit,
        hops: list[dict[str, Any]],
        kb: list[dict],
    ) -> str:
        """构建工业诊断一级初判 prompt。

        包含设备信息、异常指标（当前值/阈值/严重度）、故障链两跳查询结果、
        本地知识库检索命中，以及输出格式约束。LLM 据此给出一级初判 JSON。
        """
        chain_lines = [
            f"  - {hop.get('id', '未知')}: actions={hop.get('actions') or []}" for hop in hops
        ] or ["  - (无故障链命中)"]
        kb_lines = [f"  - {item.get('id', '')}: {item.get('text', '')}" for item in kb[:1]] or [
            "  - (无知识库命中)"
        ]
        sections = [
            "# 端侧一级诊断（工业设备异常初判）",
            "## 设备信息",
            f"设备ID: {sample.device_id}",
            f"数据源: {sample.source}",
            f"采样时间: {sample.ts.isoformat()}",
            "## 异常指标",
            f"规则ID: {hit.rule_id}",
            f"异常指标: {hit.metric}",
            f"当前值: {hit.value}",
            f"阈值: {hit.threshold}",
            f"严重度: {hit.severity}",
            f"规则描述: {hit.description}",
            "## 故障链查询(两跳)",
            *chain_lines,
            "## 知识库检索",
            *kb_lines,
            "## 要求",
            "给出最可能根因、置信度(0-1)与建议工具调用。",
            "只输出一个 JSON 对象，不要解释文字或花括号简写。示例：",
            '{"diagnosis": "轴承润滑不良", "confidence": 0.85, "tool": null}',
            'confidence 为 0-1 数字，tool 为 null 或 {"name": "工具名", "args": {}}。',
            "必须标注：端侧初判，未经验证。",
        ]
        return "\n".join(sections)

    def p95_inference_ms(self) -> float:
        """preliminary() 端侧推理 p95 延迟（毫秒），无样本时返回 0.0。"""
        if not self.inference_latencies_ms:
            return 0.0
        xs = sorted(self.inference_latencies_ms)[-self._latency_window :]
        return xs[int(0.95 * (len(xs) - 1))]

    def build_event(
        self,
        gw_id: str,
        device_id_hash: str,
        hit: RuleHit,
        evidence_refs: list[str],
        diagnosis: str | None,
        confidence: float | None,
        degraded: bool,
    ) -> EdgeEvent:
        event = EdgeEvent(
            event_id=str(ULID()),
            gw_id=gw_id,
            device_id_hash=device_id_hash,
            seq=0,
            detected_at=datetime.now(UTC),
            rule_id=hit.rule_id,
            severity=hit.severity,  # type: ignore[arg-type]
            metric=hit.metric,
            value=hit.value,
            threshold=hit.threshold,
            preliminary_diagnosis=diagnosis,
            preliminary_confidence=confidence,
            evidence_refs=evidence_refs,
            degraded=degraded,
        )
        self.state.last_event_id = event.event_id
        return event

    def replay_safe_dump(self) -> dict[str, Any]:
        return {"state": self.state.name, "last_event_id": self.state.last_event_id}
