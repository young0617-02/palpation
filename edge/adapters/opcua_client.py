"""OPC UA 适配器：轮询/订阅节点并产出 InternalSample。

增强点（E1-2）：
- 节点映射可配置（browse name → metric name），默认对齐 sim/opcua_meter.py
- 对任务要求的功率/电压/电流节点也提供默认映射，缺失节点优雅跳过
- 连接失败指数退避重试，上限可配
- namespace 查找失败时回退到 ns=2（asyncua 默认首个自定义命名空间）
- 优雅关闭：stop 后断开客户端，connected 同步反映链路状态
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from shared.schemas import InternalSample

log = logging.getLogger(__name__)


@runtime_checkable
class NodeLike(Protocol):
    """OPC UA 节点最小接口，便于 mypy 与 mock。"""

    async def read_value(self) -> object: ...


@dataclass(frozen=True)
class NodeMap:
    """OPC UA 变量节点到 InternalSample.metrics 的映射。

    browse_name: 服务器侧变量的 BrowseName（s=... 部分）
    metric_name: 输出到 metrics 字典的键名
    required:    True 表示必须存在，缺失则本轮采样失败；False 表示可选，缺失跳过
    """

    browse_name: str
    metric_name: str
    required: bool = True


# 默认节点映射：
#   - Temperature / Current 对齐 sim/opcua_meter.py，required=True 保证 sim 联调有数据
#   - Power / Voltage 为任务要求的扩展指标，required=False，缺失时优雅跳过
DEFAULT_NODES: tuple[NodeMap, ...] = (
    NodeMap(browse_name="Temperature", metric_name="temperature", required=True),
    NodeMap(browse_name="Current", metric_name="current", required=True),
    NodeMap(browse_name="Power", metric_name="power", required=False),
    NodeMap(browse_name="Voltage", metric_name="voltage", required=False),
)

DEFAULT_NAMESPACE_URI = "http://edge.sim/meter"


class OpcuaAdapter:
    """OPC UA 轮询适配器。

    保持 runtime.py 调用接口不变：
        OpcuaAdapter(endpoint, device_id, period_s=0.5)
        await adapter.run(out_queue, stop_event)
    """

    name = "opcua"

    def __init__(
        self,
        endpoint: str,
        device_id: str,
        period_s: float = 0.5,
        *,
        namespace_uri: str = DEFAULT_NAMESPACE_URI,
        nodes: Sequence[NodeMap] = DEFAULT_NODES,
        retry_delay: float = 2.0,
        retry_delay_max: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.device_id = device_id
        self.period_s = period_s
        self.namespace_uri = namespace_uri
        self.nodes = tuple(nodes)
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max
        self.connected = False

    async def run(self, out: asyncio.Queue[InternalSample], stop: asyncio.Event) -> None:
        """轮询循环：连接 → 读节点 → 封装 InternalSample → 放入队列。

        asyncua 未安装时 idle 等待 stop；连接失败按指数退避重试。
        """
        try:
            from asyncua import Client
        except ImportError:
            log.warning("asyncua not installed; OPC UA adapter idle")
            self.connected = False
            await stop.wait()
            return

        delay = self.retry_delay
        while not stop.is_set():
            try:
                async with Client(url=self.endpoint) as client:
                    self.connected = True
                    delay = self.retry_delay  # 连接成功，重置退避
                    ns = await self._resolve_namespace(client)
                    resolved = await self._resolve_nodes(client, ns)

                    while not stop.is_set():
                        sample = await self._poll_once(resolved)
                        if sample is not None:
                            await out.put(sample)
                        await self._sleep(stop, self.period_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.connected = False
                log.warning("opcua reconnecting after error", exc_info=True)
                await self._sleep(stop, delay)
                delay = min(delay * 2, self.retry_delay_max)
        self.connected = False

    async def _resolve_namespace(self, client) -> int:
        """查找命名空间索引，失败时回退到 ns=2。"""
        try:
            return await client.get_namespace_index(self.namespace_uri)
        except Exception:
            log.warning("namespace %r not found, fallback to ns=2", self.namespace_uri)
            return 2

    async def _resolve_nodes(self, client, ns: int) -> list[tuple[NodeMap, NodeLike]]:
        """预解析所有节点，跳过不存在且非 required 的节点。

        required 节点解析失败时抛异常，触发外层重连。
        """
        resolved: list[tuple[NodeMap, NodeLike]] = []
        for node_map in self.nodes:
            node: NodeLike = client.get_node(f"ns={ns};s={node_map.browse_name}")
            try:
                # 读取一次验证节点存在
                await node.read_value()
                resolved.append((node_map, node))
            except Exception:
                if node_map.required:
                    raise
                log.info("opcua optional node %s skipped (not found)", node_map.browse_name)
        return resolved

    async def _poll_once(self, resolved: list[tuple[NodeMap, NodeLike]]) -> InternalSample | None:
        """读取所有已解析节点并构造 InternalSample。"""
        metrics: dict[str, float] = {}
        for node_map, node in resolved:
            try:
                value = await node.read_value()
                # asyncua 返回 DataValue 或裸值，统一取 .Value
                raw = value.Value if hasattr(value, "Value") else value
                metrics[node_map.metric_name] = float(raw)  # type: ignore[arg-type]
            except Exception:
                log.warning("opcua read failed for %s", node_map.browse_name, exc_info=True)
                if node_map.required:
                    return None
                continue
        if not metrics:
            return None
        return InternalSample(
            device_id=self.device_id,
            source=self.name,
            ts=datetime.now(UTC),
            metrics=metrics,
        )

    async def _sleep(self, stop: asyncio.Event, delay: float) -> None:
        """可被 stop 打断的 sleep。"""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=delay)
