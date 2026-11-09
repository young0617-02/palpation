from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edge.knowledge.fault_chain import FaultChain
from edge.knowledge.sqlite_vec import LocalKnowledge
from edge.stream.window import SlidingWindow

# get_device_status 默认关注的设备测点（工业泵/电机常见指标）
_DEFAULT_METRICS: tuple[str, ...] = ("vibration_rms", "temperature", "current")


def build_tools(
    window: SlidingWindow, chain: FaultChain, knowledge: LocalKnowledge
) -> dict[str, Callable[..., Any]]:
    """构造端侧 Agent 工具白名单。

    工具供 LLM 一级初判通过 `tool` 字段回调，所有工具均为只读查询，
    不含任何执行/停机等高危动作（方案 §2.2 端侧不可一键执行）。
    """

    def lookup_fault_chain(rule_id: str) -> dict:
        """根据规则 ID 查询故障链两跳因果图，返回根因与建议处置动作。"""
        return {"hops": chain.two_hop(rule_id)}

    def get_window_snapshot(metric: str) -> dict:
        """返回指定指标在滑动窗口内的近期采样值与快照 ID。"""
        return {
            "metric": metric,
            "values": window.values(metric),
            "snapshot_id": window.snapshot_id(metric),
        }

    def search_local_knowledge(query: str) -> dict:
        """本地知识库语义检索，返回与查询最相关的运维片段。"""
        return {"hits": knowledge.search(query)}

    def get_device_status(metrics: list[str] | None = None) -> dict:
        """返回当前设备多指标状态摘要（末值、样本数、快照 ID）。

        Args:
            metrics: 关注的指标名列表；为 None 时使用默认工业测点。
        """
        targets = metrics if metrics is not None else list(_DEFAULT_METRICS)
        status: dict[str, Any] = {}
        for metric in targets:
            vals = window.values(metric)
            if not vals:
                continue
            status[metric] = {
                "last": vals[-1],
                "count": len(vals),
                "snapshot_id": window.snapshot_id(metric),
            }
        return {"device_status": status}

    return {
        "lookup_fault_chain": lookup_fault_chain,
        "get_window_snapshot": get_window_snapshot,
        "search_local_knowledge": search_local_knowledge,
        "get_device_status": get_device_status,
    }


# 工具 schema：供 LLM prompt 注明可用工具及参数类型/返回值。
# 与 build_tools 返回的键保持一致。
TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "lookup_fault_chain",
        "description": "根据规则 ID 查询故障链两跳因果图，返回根因与建议处置动作",
        "args": {"rule_id": "str"},
        "returns": "dict[str, list[dict]] — {hops: [...]}",
    },
    {
        "name": "get_window_snapshot",
        "description": "返回指定指标在滑动窗口内的近期采样值与快照 ID",
        "args": {"metric": "str"},
        "returns": "dict — {metric, values: list[float], snapshot_id: str}",
    },
    {
        "name": "search_local_knowledge",
        "description": "本地知识库语义检索，返回与查询最相关的运维片段",
        "args": {"query": "str"},
        "returns": "dict[str, list[dict]] — {hits: [...]}",
    },
    {
        "name": "get_device_status",
        "description": "返回当前设备多指标状态摘要（末值、样本数、快照 ID）",
        "args": {"metrics": "list[str] | None"},
        "returns": "dict[str, dict] — {device_status: {metric: {last, count, snapshot_id}}}",
    },
]
