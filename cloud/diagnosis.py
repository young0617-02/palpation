"""云端深度诊断管线（纯 Python 实现，不依赖 LangGraph）。

管线步骤：
1. 传感器数据分析 —— 计算 metric 偏离度与严重性
2. 本地知识检索 —— ONNX 语义检索（bge-small-zh-v1.5 int8 / 512 维 / onnxruntime，numpy 余弦）手册片段作为引用
3. 故障链图谱推理 —— fault_chain ≤2 跳路径推导根因与措施
4. 综合 CloudDiagnosis —— 带引用的正式报告

知识库或图谱文件缺失时优雅降级：返回空引用并降低置信度。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edge.knowledge.fault_chain import FaultChain
from edge.knowledge.sqlite_vec import LocalKnowledge
from shared.schemas import CloudDiagnosis, EdgeEvent

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_SNIPPETS_PATH = _ROOT / "edge" / "knowledge" / "snippets.json"
_FAULT_CHAINS_PATH = _ROOT / "edge" / "knowledge" / "fault_chains.json"
_KB_DB_PATH = _ROOT / "data" / "cloud_kb.db"

# 置信度权重
_BASE_CONFIDENCE = 0.3
_SENSOR_BONUS = 0.2
_KB_BONUS = 0.2
_CHAIN_BONUS = 0.3
_MAX_CONFIDENCE = 0.95
_KB_TOP_K = 3

# metric → 传感器层面初步分析
_METRIC_ANALYSIS: dict[str, str] = {
    "vibration_rms": "振动 RMS 越限，常指向轴承磨损、润滑不良、不平衡或不对中",
    "temperature": "温度越限，可能指向过热、润滑失效或散热异常",
    "current": "电流越限，可能指向过载、机械卡阻或工况负荷异常",
}


@dataclass
class SensorAnalysis:
    """传感器数据分析结果。"""

    deviation: float
    description: str
    severity: str


@dataclass
class KnowledgeRetrieval:
    """本地知识检索结果。"""

    citations: list[str] = field(default_factory=list)
    snippets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FaultInference:
    """故障链图谱推理结果。"""

    root_id: str = "unknown"
    causes: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    hops: list[dict[str, Any]] = field(default_factory=list)


def _analyze_sensor(event: EdgeEvent) -> SensorAnalysis:
    """分析传感器指标偏离度与严重性。"""
    threshold = event.threshold if event.threshold != 0 else 1.0
    deviation = event.value / threshold
    base = _METRIC_ANALYSIS.get(event.metric, f"指标 {event.metric} 越限，需结合工况进一步研判")
    description = (
        f"{base}。实测值 {event.value}，阈值 {event.threshold}，"
        f"偏离度 {deviation:.2f}，边缘判定严重性 {event.severity}。"
    )
    return SensorAnalysis(deviation=deviation, description=description, severity=event.severity)


def _retrieve_knowledge(event: EdgeEvent) -> KnowledgeRetrieval:
    """检索本地知识库相关手册片段，失败时优雅降级返回空。"""
    query = f"{event.rule_id} {event.preliminary_diagnosis or ''} {event.metric}"
    try:
        kb = LocalKnowledge(_KB_DB_PATH, _SNIPPETS_PATH)
        hits = kb.search(query, top_k=_KB_TOP_K)
    except Exception:
        log.warning("knowledge retrieval degraded for event %s", event.event_id, exc_info=True)
        return KnowledgeRetrieval()
    citations = [f"{h['id']}:{h['title']}" for h in hits]
    return KnowledgeRetrieval(citations=citations, snippets=hits)


def _infer_fault_chain(event: EdgeEvent) -> FaultInference:
    """故障链图谱推理（≤2 跳），失败时优雅降级。"""
    try:
        chain = FaultChain(_FAULT_CHAINS_PATH)
        hops = chain.two_hop(event.rule_id)
    except Exception:
        log.warning("fault chain inference degraded for event %s", event.event_id, exc_info=True)
        return FaultInference()
    root = hops[0] if hops else {}
    return FaultInference(
        root_id=str(root.get("id", "unknown")),
        causes=list(root.get("causes", [])),
        actions=list(root.get("actions", [])),
        hops=hops,
    )


def depth_diagnosis(event: EdgeEvent) -> CloudDiagnosis:
    """云端深度诊断管线入口。

    传感器分析 + 本地知识检索 + 故障链图谱推理 → 带引用的 CloudDiagnosis。
    知识库或图谱文件缺失时优雅降级（空引用 + 降低置信度）。
    """
    sensor = _analyze_sensor(event)
    knowledge = _retrieve_knowledge(event)
    fault = _infer_fault_chain(event)

    # 综合结论
    parts: list[str] = [
        f"云端深度诊断：规则 {event.rule_id} 触发。",
        sensor.description,
    ]
    if fault.root_id != "unknown":
        causes = "、".join(fault.causes) if fault.causes else "待定"
        actions = "；".join(fault.actions) if fault.actions else "持续观测"
        parts.append(f"图谱命中故障 {fault.root_id}，可能根因：{causes}。建议措施：{actions}。")
    else:
        parts.append("图谱未命中已知故障链，建议上报人工深度排查。")
    if knowledge.snippets:
        titles = "、".join(s["title"] for s in knowledge.snippets)
        parts.append(f"知识库引用：{titles}。")
    conclusion = "".join(parts)

    # 置信度：基于证据数量
    confidence = _BASE_CONFIDENCE + _SENSOR_BONUS
    if knowledge.citations:
        confidence += _KB_BONUS * min(len(knowledge.citations) / _KB_TOP_K, 1.0)
    if fault.root_id != "unknown":
        confidence += _CHAIN_BONUS
    confidence = min(confidence, _MAX_CONFIDENCE)

    return CloudDiagnosis(
        event_id=event.event_id,
        gw_id=event.gw_id,
        conclusion=conclusion,
        confidence=round(confidence, 2),
        citations=knowledge.citations,
        requires_human_review=event.severity == "high",
    )
