"""适配器单元测试：ModbusTcpAdapter / OpcuaAdapter。

测试分层：
1. Mock 测试（不依赖外部 sim 进程）：
   - Modbus 轮询 → InternalSample 生成正确
   - Modbus 连接失败指数退避重试
   - Modbus 读取错误后重连
   - OPC UA 轮询 → InternalSample 生成正确
   - OPC UA 连接失败重试
   - OPC UA 可选节点缺失优雅跳过
2. 真实 Modbus server 集成测试（自建 server，不依赖 sim/modbus_device.py）：
   - 适配器连接真实 server 并产出样本
3. InternalSample schema 验证
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from edge.adapters.modbus_tcp import ModbusTcpAdapter, RegisterMap
from edge.adapters.opcua_client import NodeMap, OpcuaAdapter
from shared.schemas import InternalSample

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 辅助：从队列收集样本 / 运行适配器一段时间
# --------------------------------------------------------------------------- #


async def _collect(
    queue: asyncio.Queue[InternalSample], count: int, timeout: float = 3.0
) -> list[InternalSample]:
    """从队列收集 count 个样本，超时则返回已收集的。"""
    samples: list[InternalSample] = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while len(samples) < count:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            s = await asyncio.wait_for(queue.get(), timeout=remaining)
            samples.append(s)
        except asyncio.TimeoutError:
            break
    return samples


async def _run_for(
    adapter: ModbusTcpAdapter | OpcuaAdapter,
    queue: asyncio.Queue[InternalSample],
    stop: asyncio.Event,
    duration: float = 0.5,
) -> None:
    """运行适配器 duration 秒后设置 stop 并等待其退出。

    适配器停止后 connected 会被置 False，因此连接状态应在运行期间
    通过样本产出或 fake client 状态间接验证。
    """
    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.sleep(duration)
    stop.set()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=5.0)


# --------------------------------------------------------------------------- #
# 1. Modbus Mock 测试
# --------------------------------------------------------------------------- #


class FakeModbusResult:
    """模拟 pymodbus 读取结果。"""

    def __init__(self, registers: list[int], error: bool = False):
        self.registers = registers
        self._error = error

    def isError(self) -> bool:
        return self._error


class FakeModbusClient:
    """模拟 AsyncModbusTcpClient，可控连接与读取行为。"""

    def __init__(
        self,
        registers: list[int] | None = None,
        connect_ok: bool = True,
        read_error: bool = False,
    ):
        self._registers = registers if registers is not None else [120, 460, 1000]
        self._connect_ok = connect_ok
        self._read_error = read_error
        self.connected = False
        self.connect_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    async def connect(self) -> bool:
        self.connect_calls += 1
        if self._connect_ok:
            self.connected = True
        return self._connect_ok

    async def read_holding_registers(self, address: int, count: int = 1, device_id: int = 1):
        self.read_calls += 1
        if self._read_error:
            return FakeModbusResult([], error=True)
        return FakeModbusResult([self._registers[address]])

    def close(self) -> None:
        self.close_calls += 1
        self.connected = False


@pytest.fixture()
def patch_modbus_client():
    """patch AsyncModbusTcpClient，返回 FakeModbusClient 工厂。

    用法：fake = patch_modbus_client(registers=[120, 460, 1000])
    """
    holder: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeModbusClient:
        fake = FakeModbusClient(**kwargs)
        holder["fake"] = fake
        return fake

    def constructor(*args: Any, **kw: Any) -> FakeModbusClient:
        return holder["fake"]

    with patch("edge.adapters.modbus_tcp.AsyncModbusTcpClient", side_effect=constructor):
        yield factory


async def test_modbus_poll_produces_internal_sample(patch_modbus_client):
    """Modbus 轮询正常时产出 InternalSample，字段映射正确。

    寄存器值 [120, 460, 1000] 经缩放：
      vibration_rms = 120/100 = 1.2
      temperature   = 460/10  = 46.0
      current       = 1000/100 = 10.0
    """
    fake = patch_modbus_client(registers=[120, 460, 1000])
    adapter = ModbusTcpAdapter("127.0.0.1", 5020, "pump-de-01", period_s=0.05)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)

    assert fake.connect_calls >= 1
    assert fake.read_calls >= 3  # 至少一轮（3 个寄存器）

    samples = await _collect(queue, 1)
    assert len(samples) >= 1
    s = samples[0]
    assert s.device_id == "pump-de-01"
    assert s.source == "modbus"
    assert isinstance(s.ts, datetime)
    assert s.ts.tzinfo is not None
    assert "vibration_rms" in s.metrics
    assert "temperature" in s.metrics
    assert "current" in s.metrics
    assert s.metrics["vibration_rms"] == pytest.approx(1.2, abs=0.01)
    assert s.metrics["temperature"] == pytest.approx(46.0, abs=0.1)
    assert s.metrics["current"] == pytest.approx(10.0, abs=0.01)


async def test_modbus_custom_register_map(patch_modbus_client):
    """自定义寄存器映射时指标名和缩放正确。"""
    patch_modbus_client(registers=[500, 2300])
    regs = (
        RegisterMap(address=0, scale=0.01, name="vibration_rms"),
        RegisterMap(address=1, scale=0.1, name="voltage"),
    )
    adapter = ModbusTcpAdapter(
        "127.0.0.1", 5020, "dev-custom", period_s=0.05, registers=regs
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)
    samples = await _collect(queue, 1)
    assert len(samples) >= 1
    s = samples[0]
    assert s.device_id == "dev-custom"
    assert s.metrics["vibration_rms"] == pytest.approx(5.0, abs=0.01)
    assert s.metrics["voltage"] == pytest.approx(230.0, abs=0.1)


async def test_modbus_connect_failure_retry(patch_modbus_client):
    """连接失败时 connected 保持 False，适配器不崩溃，指数退避重试。"""
    fake = patch_modbus_client(connect_ok=False)
    adapter = ModbusTcpAdapter(
        "127.0.0.1", 59999, "pump-de-01", period_s=0.05, retry_delay=0.05
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)

    assert fake.connect_calls >= 2  # 多次重试
    assert adapter.connected is False
    assert queue.empty()  # 无样本产出


async def test_modbus_read_error_no_sample(patch_modbus_client):
    """读取返回错误时本轮无样本产出，适配器继续运行不崩溃。"""
    fake = patch_modbus_client(read_error=True)
    adapter = ModbusTcpAdapter("127.0.0.1", 5020, "pump-de-01", period_s=0.05)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)

    assert fake.read_calls >= 1
    assert queue.empty()  # 读取错误不产出样本


async def test_modbus_stop_exits_cleanly(patch_modbus_client):
    """stop 设置后适配器优雅退出，不抛异常。"""
    patch_modbus_client(registers=[120, 460, 1000])
    adapter = ModbusTcpAdapter("127.0.0.1", 5020, "pump-de-01", period_s=0.05)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    # 退出后 connected 应为 False
    assert adapter.connected is False


# --------------------------------------------------------------------------- #
# 2. OPC UA Mock 测试
# --------------------------------------------------------------------------- #


class FakeOpcuaNode:
    """模拟 asyncua Node。"""

    def __init__(self, value: float, fail: bool = False):
        self._value = value
        self._fail = fail

    async def read_value(self) -> float:
        if self._fail:
            raise RuntimeError("node read failed")
        return self._value


class FakeOpcuaClient:
    """模拟 asyncua.Client（async context manager）。"""

    def __init__(
        self,
        nodes: dict[str, FakeOpcuaNode] | None = None,
        namespace_idx: int = 2,
        namespace_fail: bool = False,
    ):
        self._nodes = nodes or {
            "Temperature": FakeOpcuaNode(45.5),
            "Current": FakeOpcuaNode(10.2),
        }
        self._namespace_idx = namespace_idx
        self._namespace_fail = namespace_fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def get_namespace_index(self, uri: str) -> int:
        if self._namespace_fail:
            raise RuntimeError("namespace not found")
        return self._namespace_idx

    def get_node(self, node_id: str) -> FakeOpcuaNode:
        # node_id 形如 "ns=2;s=Temperature"
        browse_name = node_id.split(";s=")[-1]
        if browse_name not in self._nodes:
            # 模拟节点不存在：返回一个读取会失败的 node
            return FakeOpcuaNode(0.0, fail=True)
        return self._nodes[browse_name]


@pytest.fixture()
def patch_opcua_client():
    """patch asyncua.Client，返回 FakeOpcuaClient 工厂。

    通过替换 sys.modules["asyncua"] 实现，因为适配器在 run() 内部 import。
    """
    import sys

    holder: dict[str, Any] = {}
    original = sys.modules.get("asyncua")

    def factory(**kwargs: Any) -> FakeOpcuaClient:
        fake = FakeOpcuaClient(**kwargs)
        holder["fake"] = fake
        return fake

    class FakeAsyncuaModule:
        def Client(self, *args: Any, **kw: Any):  # noqa: N802
            return holder["fake"]

    fake_mod = FakeAsyncuaModule()
    sys.modules["asyncua"] = fake_mod  # type: ignore[assignment]
    try:
        yield factory
    finally:
        if original is not None:
            sys.modules["asyncua"] = original
        else:
            sys.modules.pop("asyncua", None)


async def test_opcua_poll_produces_internal_sample(patch_opcua_client):
    """OPC UA 轮询正常时产出 InternalSample。"""
    patch_opcua_client(
        nodes={
            "Temperature": FakeOpcuaNode(45.5),
            "Current": FakeOpcuaNode(10.2),
        }
    )
    adapter = OpcuaAdapter("opc.tcp://127.0.0.1:4840", "meter-01", period_s=0.05)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)

    samples = await _collect(queue, 1)
    assert len(samples) >= 1
    s = samples[0]
    assert s.device_id == "meter-01"
    assert s.source == "opcua"
    assert isinstance(s.ts, datetime)
    assert s.ts.tzinfo is not None
    assert "temperature" in s.metrics
    assert "current" in s.metrics
    assert s.metrics["temperature"] == pytest.approx(45.5)
    assert s.metrics["current"] == pytest.approx(10.2)


async def test_opcua_optional_nodes_skipped(patch_opcua_client):
    """可选节点（Power/Voltage）不存在时优雅跳过，required 节点正常读取。"""
    patch_opcua_client(
        nodes={
            "Temperature": FakeOpcuaNode(46.0),
            "Current": FakeOpcuaNode(10.0),
            # Power / Voltage 不存在 → get_node 返回 fail node → 跳过
        }
    )
    adapter = OpcuaAdapter("opc.tcp://127.0.0.1:4840", "meter-01", period_s=0.05)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)

    samples = await _collect(queue, 1)
    assert len(samples) >= 1
    s = samples[0]
    assert "temperature" in s.metrics
    assert "current" in s.metrics
    # Power/Voltage 不存在，不应出现在 metrics 中
    assert "power" not in s.metrics
    assert "voltage" not in s.metrics


async def test_opcua_custom_nodes_with_power_voltage(patch_opcua_client):
    """服务器提供 Power/Voltage 节点时，适配器读取并输出。"""
    patch_opcua_client(
        nodes={
            "Power": FakeOpcuaNode(2300.0),
            "Voltage": FakeOpcuaNode(380.0),
            "Current": FakeOpcuaNode(12.5),
        }
    )
    nodes = (
        NodeMap(browse_name="Power", metric_name="power", required=True),
        NodeMap(browse_name="Voltage", metric_name="voltage", required=True),
        NodeMap(browse_name="Current", metric_name="current", required=True),
    )
    adapter = OpcuaAdapter(
        "opc.tcp://127.0.0.1:4840", "meter-01", period_s=0.05, nodes=nodes
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)
    samples = await _collect(queue, 1)
    assert len(samples) >= 1
    s = samples[0]
    assert s.metrics["power"] == pytest.approx(2300.0)
    assert s.metrics["voltage"] == pytest.approx(380.0)
    assert s.metrics["current"] == pytest.approx(12.5)


async def test_opcua_connect_failure_retry():
    """OPC UA 连接异常时指数退避重试，不崩溃。"""
    import sys

    class FailingClient:
        def __init__(self, *args: Any, **kw: Any):
            raise RuntimeError("connection refused")

    original = sys.modules.get("asyncua")

    class FakeAsyncuaModule:
        Client = FailingClient  # noqa: N815

    fake_mod = FakeAsyncuaModule()
    sys.modules["asyncua"] = fake_mod  # type: ignore[assignment]
    try:
        adapter = OpcuaAdapter(
            "opc.tcp://127.0.0.1:59999", "meter-01", period_s=0.05, retry_delay=0.05
        )
        queue: asyncio.Queue[InternalSample] = asyncio.Queue()
        stop = asyncio.Event()

        await _run_for(adapter, queue, stop, duration=0.3)
        assert adapter.connected is False
        assert queue.empty()
    finally:
        if original is not None:
            sys.modules["asyncua"] = original
        else:
            sys.modules.pop("asyncua", None)


async def test_opcua_namespace_fallback(patch_opcua_client):
    """namespace 查找失败时回退到 ns=2，仍能读取节点。"""
    patch_opcua_client(
        nodes={
            "Temperature": FakeOpcuaNode(44.0),
            "Current": FakeOpcuaNode(9.5),
        },
        namespace_fail=True,
    )
    adapter = OpcuaAdapter("opc.tcp://127.0.0.1:4840", "meter-01", period_s=0.05)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.3)
    samples = await _collect(queue, 1)
    assert len(samples) >= 1
    assert samples[0].metrics["temperature"] == pytest.approx(44.0)


# --------------------------------------------------------------------------- #
# 3. InternalSample schema 与默认映射验证
# --------------------------------------------------------------------------- #


def test_internal_sample_schema():
    """InternalSample 字段类型与约束验证。"""
    sample = InternalSample(
        device_id="pump-de-01",
        source="modbus",
        ts=datetime.now(UTC),
        metrics={"vibration_rms": 1.2, "temperature": 46.0, "current": 10.0},
    )
    assert sample.device_id == "pump-de-01"
    assert sample.source == "modbus"
    assert isinstance(sample.ts, datetime)
    assert sample.ts.tzinfo is not None
    assert all(isinstance(v, float) for v in sample.metrics.values())
    # 序列化往返
    j = sample.model_dump_json()
    s2 = InternalSample.model_validate_json(j)
    assert s2.device_id == sample.device_id
    assert s2.metrics == sample.metrics


def test_register_map_defaults():
    """默认寄存器映射与 sim/modbus_device.py 对齐。"""
    from edge.adapters.modbus_tcp import DEFAULT_REGISTERS

    names = [r.name for r in DEFAULT_REGISTERS]
    assert names == ["vibration_rms", "temperature", "current"]
    assert DEFAULT_REGISTERS[0].address == 0
    assert DEFAULT_REGISTERS[1].address == 1
    assert DEFAULT_REGISTERS[2].address == 2


def test_node_map_defaults():
    """默认节点映射包含 required 的 Temperature/Current 和可选 Power/Voltage。"""
    from edge.adapters.opcua_client import DEFAULT_NODES

    by_name = {n.metric_name: n for n in DEFAULT_NODES}
    assert by_name["temperature"].required is True
    assert by_name["current"].required is True
    assert by_name["power"].required is False
    assert by_name["voltage"].required is False


# --------------------------------------------------------------------------- #
# 4. 真实 Modbus server 集成测试
# --------------------------------------------------------------------------- #


async def _start_modbus_server(port: int, values: list[int]) -> asyncio.Task:
    """启动真实 pymodbus async TCP server，返回可取消的 task。

    server 持有给定 values 的保持寄存器（地址 0..len-1）。
    使用 ModbusSparseDataBlock 以 dict 形式指定地址，兼容 pymodbus 3.x。
    """
    from pymodbus.datastore import (
        ModbusDeviceContext,
        ModbusSparseDataBlock,
        ModbusServerContext,
    )
    from pymodbus.server import StartAsyncTcpServer

    addr_map = {i: v for i, v in enumerate(values)}
    block = ModbusSparseDataBlock(addr_map)
    ctx = ModbusDeviceContext(hr=block)
    context = ModbusServerContext(devices={1: ctx}, single=False)

    async def serve() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await StartAsyncTcpServer(context=context, address=("127.0.0.1", port))

    task = asyncio.create_task(serve())
    await asyncio.sleep(0.5)  # 给 server 启动时间
    return task


async def test_modbus_real_server_integration():
    """适配器连接真实 pymodbus server 并产出 InternalSample。

    不依赖 sim/modbus_device.py（其 pymodbus API 已不兼容），
    而是在测试内自建 server，寄存器值固定。
    """
    port = 15102
    # 寄存器值：vib=1.2(×100), temp=46.0(×10), current=10.0(×100)
    values = [120, 460, 1000]
    task = await _start_modbus_server(port, values)

    try:
        adapter = ModbusTcpAdapter(
            "127.0.0.1", port, "pump-real", period_s=0.1, retry_delay=0.2
        )
        queue: asyncio.Queue[InternalSample] = asyncio.Queue()
        stop = asyncio.Event()

        await _run_for(adapter, queue, stop, duration=1.0)

        samples = await _collect(queue, 1)
        assert len(samples) >= 1, "no samples produced from real server"
        s = samples[0]
        assert s.device_id == "pump-real"
        assert s.source == "modbus"
        assert s.metrics["vibration_rms"] == pytest.approx(1.2, abs=0.01)
        assert s.metrics["temperature"] == pytest.approx(46.0, abs=0.1)
        assert s.metrics["current"] == pytest.approx(10.0, abs=0.01)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_modbus_real_server_connection_refused():
    """连接不存在的端口时重试不崩溃，connected 保持 False。"""
    adapter = ModbusTcpAdapter(
        "127.0.0.1", 59998, "pump-noref", period_s=0.05, retry_delay=0.05,
        connect_timeout=0.5,
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()

    await _run_for(adapter, queue, stop, duration=0.4)
    assert adapter.connected is False
    assert queue.empty()