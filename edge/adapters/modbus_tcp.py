"""Modbus TCP 适配器：轮询保持寄存器并产出 InternalSample。

增强点（E1-2）：
- 寄存器映射可配置（地址 → 指标名 + 缩放因子），默认对齐 sim/modbus_device.py
- 连接失败指数退避重试，上限可配
- pymodbus 客户端带 timeout/retries，避免单次轮询挂死
- 优雅关闭：stop 后关闭客户端，connected 同步反映链路状态
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pymodbus.client import AsyncModbusTcpClient

from shared.schemas import InternalSample

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisterMap:
    """单个保持寄存器到 InternalSample.metrics 的映射。

    address:  保持寄存器起始地址（0-based）
    scale:    原始整数值乘以 scale 得到工程量（如 1/100 表示存的是 ×100）
    name:     输出到 metrics 字典的键名
    """

    address: int
    scale: float
    name: str


# 默认映射，与 sim/modbus_device.py 中 Holding.tick() 写入的三个寄存器对齐：
#   values[0] = int(vib*100)   → vibration_rms
#   values[1] = int(temp*10)   → temperature
#   values[2] = int(current*100) → current
DEFAULT_REGISTERS: tuple[RegisterMap, ...] = (
    RegisterMap(address=0, scale=1 / 100.0, name="vibration_rms"),
    RegisterMap(address=1, scale=1 / 10.0, name="temperature"),
    RegisterMap(address=2, scale=1 / 100.0, name="current"),
)


class ModbusTcpAdapter:
    """Modbus TCP 轮询适配器。

    保持 runtime.py 调用接口不变：
        ModbusTcpAdapter(host, port, device_id, period_s=0.1)
        await adapter.run(out_queue, stop_event)
    """

    name = "modbus"

    def __init__(
        self,
        host: str,
        port: int,
        device_id: str,
        period_s: float = 0.1,
        *,
        unit_id: int = 1,
        registers: Sequence[RegisterMap] = DEFAULT_REGISTERS,
        connect_timeout: float = 3.0,
        retry_delay: float = 1.0,
        retry_delay_max: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.device_id = device_id
        self.period_s = period_s
        self.unit_id = unit_id
        self.registers = tuple(registers)
        self.connect_timeout = connect_timeout
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max
        self.connected = False

    async def run(self, out: asyncio.Queue[InternalSample], stop: asyncio.Event) -> None:
        """轮询循环：连接 → 读寄存器 → 封装 InternalSample → 放入队列。

        连接失败或读取异常时按指数退避重试，stop 被设置后优雅退出。
        """
        client = AsyncModbusTcpClient(
            self.host,
            port=self.port,
            timeout=self.connect_timeout,
            retries=0,  # 由本适配器自行控制重试节奏
        )
        delay = self.retry_delay
        try:
            while not stop.is_set():
                try:
                    if not client.connected:
                        ok = await client.connect()
                        self.connected = bool(ok)
                        if not ok:
                            await self._sleep(stop, delay)
                            delay = min(delay * 2, self.retry_delay_max)
                            continue
                        delay = self.retry_delay  # 连接成功，重置退避

                    sample = await self._poll_once(client)
                    if sample is None:
                        await self._sleep(stop, self.period_s)
                        continue
                    self.connected = True
                    await out.put(sample)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("modbus poll failed")
                    self.connected = False
                    with contextlib.suppress(Exception):
                        client.close()
                    await self._sleep(stop, delay)
                    delay = min(delay * 2, self.retry_delay_max)
                    continue
                await self._sleep(stop, self.period_s)
        finally:
            self.connected = False
            with contextlib.suppress(Exception):
                client.close()

    async def _poll_once(self, client: AsyncModbusTcpClient) -> InternalSample | None:
        """读取全部映射寄存器并构造 InternalSample，读取失败返回 None。"""
        metrics: dict[str, float] = {}
        for reg in self.registers:
            rr = await client.read_holding_registers(
                address=reg.address, count=1, device_id=self.unit_id
            )
            if rr.isError():
                log.warning("modbus read error at addr=%d: %s", reg.address, rr)
                self.connected = False
                return None
            metrics[reg.name] = float(rr.registers[0]) * reg.scale
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
