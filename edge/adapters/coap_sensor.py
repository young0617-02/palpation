"""CoAP 低功耗传感器适配器：轮询/observe 推送并产出 InternalSample。

增强点（S1）：
- 可配置轮询间隔（低功耗设备可设 60s–600s）
- 指数退避重连（失败后延迟递增，成功后重置，上限可配）
- CoAP observe 推送模式（服务器主动推送，更省电）
- 连接状态管理与统计（请求数/成功/失败/平均延迟）
- 请求超时控制
- aiocoap 未安装时 idle（MVP Should，不强制依赖）
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from shared.schemas import InternalSample

log = logging.getLogger(__name__)


class CoapAdapter:
    """Best-effort CoAP poller. Idle if aiocoap is missing (MVP Should, not Must).

    支持两种工作模式：
    - 轮询模式（默认）：按 poll_interval_s 间隔 GET 资源
    - observe 模式（observe_mode=True）：订阅资源，服务器推送时产出样本，更省电
    """

    name = "coap"

    def __init__(
        self,
        uri: str = "coap://127.0.0.1:5683/sensor",
        device_id: str = "coap-01",
        *,
        poll_interval_s: float = 10.0,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        observe_mode: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        self.uri = uri
        self.device_id = device_id
        self.poll_interval_s = poll_interval_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.observe_mode = observe_mode
        self.timeout_s = timeout_s
        self.connected = False
        # 统计信息
        self._total_requests = 0
        self._successful = 0
        self._failed = 0
        self._latency_samples: list[float] = []
        # aiocoap client context（惰性创建并在多次轮询间复用）
        self._context: Any = None

    # ------------------------------------------------------------------ #
    # 退避延迟与统计
    # ------------------------------------------------------------------ #

    def _backoff_delay(self, retry: int) -> float:
        """计算指数退避延迟：min(base * 2^retry, max)。"""
        return min(self.backoff_base_s * (2**retry), self.backoff_max_s)

    def stats(self) -> dict[str, Any]:
        """返回适配器统计信息。"""
        avg_latency = (
            sum(self._latency_samples) / len(self._latency_samples)
            if self._latency_samples
            else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "successful": self._successful,
            "failed": self._failed,
            "avg_latency_ms": avg_latency,
            "connected": self.connected,
        }

    # ------------------------------------------------------------------ #
    # 单次轮询
    # ------------------------------------------------------------------ #

    async def _get_context(self) -> Any:
        """惰性创建并复用 aiocoap client context。"""
        if self._context is None:
            from aiocoap import Context

            self._context = await Context.create_client_context()
        return self._context

    async def poll_once(self) -> InternalSample | None:
        """单次轮询：GET uri → 解析 JSON → InternalSample。

        成功返回样本并更新统计；失败（异常/超时/无效负载）返回 None。
        aiocoap 未安装时返回 None。
        """
        try:
            from aiocoap import GET, Message
        except ImportError:
            return None

        self._total_requests += 1
        start = time.monotonic()
        try:
            ctx = await self._get_context()
            req = Message(code=GET, uri=self.uri)
            resp = await asyncio.wait_for(ctx.request(req).response, timeout=self.timeout_s)
            data = json.loads(resp.payload.decode("utf-8"))
            sample = InternalSample(
                device_id=self.device_id,
                source=self.name,
                ts=datetime.now(UTC),
                metrics={k: float(v) for k, v in data.items()},
            )
        except Exception:
            self._failed += 1
            self.connected = False
            log.warning("coap poll_once failed", exc_info=True)
            return None
        latency_ms = (time.monotonic() - start) * 1000.0
        self._latency_samples.append(latency_ms)
        self._successful += 1
        self.connected = True
        return sample

    # ------------------------------------------------------------------ #
    # observe 推送模式
    # ------------------------------------------------------------------ #

    async def observe(self, out: asyncio.Queue[InternalSample], stop: asyncio.Event) -> None:
        """CoAP observe 推送模式：订阅资源，服务器推送时产出样本。

        订阅断开或异常时按指数退避重连，stop 被设置后优雅退出。
        aiocoap 未安装时 idle 等待 stop。
        """
        try:
            from aiocoap import GET, Message
        except ImportError:
            log.info("aiocoap not installed; CoAP observe idle")
            await stop.wait()
            return

        retry = 0
        while not stop.is_set():
            try:
                ctx = await self._get_context()
                req = Message(code=GET, uri=self.uri, observe=0)
                observation = await asyncio.wait_for(
                    ctx.request(req).observation, timeout=self.timeout_s
                )
                self.connected = True
                retry = 0
                await self._drain_observation(observation, out, stop)
                if stop.is_set():
                    break
                self.connected = False
            except asyncio.CancelledError:
                raise
            except Exception:
                self.connected = False
                log.warning("coap observe failed, reconnecting", exc_info=True)

            if stop.is_set():
                break
            if retry < self.max_retries:
                delay = self._backoff_delay(retry)
                retry += 1
            else:
                delay = self.poll_interval_s
                retry = 0
            await self._sleep(stop, delay)

    async def _drain_observation(
        self,
        observation: Any,
        out: asyncio.Queue[InternalSample],
        stop: asyncio.Event,
    ) -> None:
        """迭代 observation 推送，直到订阅结束或 stop 被设置。

        用 asyncio.wait 同时监控消费任务与 stop，保证 stop 能及时打断
        阻塞中的 async for。
        """

        async def _consume() -> None:
            async for data in observation:
                self._total_requests += 1
                start = time.monotonic()
                try:
                    payload = json.loads(data.payload.decode("utf-8"))
                    sample = InternalSample(
                        device_id=self.device_id,
                        source=self.name,
                        ts=datetime.now(UTC),
                        metrics={k: float(v) for k, v in payload.items()},
                    )
                except Exception:
                    self._failed += 1
                    log.warning("coap observe payload parse failed", exc_info=True)
                    continue
                self._latency_samples.append((time.monotonic() - start) * 1000.0)
                self._successful += 1
                await out.put(sample)

        consume_task = asyncio.create_task(_consume())
        stop_task = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({consume_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (consume_task, stop_task):
                if not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    async def run(self, out: asyncio.Queue[InternalSample], stop: asyncio.Event) -> None:
        """主循环：observe_mode 时进入推送模式，否则按间隔轮询。

        失败按指数退避重连，成功后重置退避；aiocoap 未安装时 idle 等待 stop。
        """
        try:
            from aiocoap import GET, Context, Message  # noqa: F401
        except ImportError:
            log.info("aiocoap not installed; CoAP adapter idle")
            await stop.wait()
            return

        try:
            if self.observe_mode:
                await self.observe(out, stop)
                return

            retry = 0
            while not stop.is_set():
                sample = await self.poll_once()
                if sample is not None:
                    await out.put(sample)
                    retry = 0
                    await self._sleep(stop, self.poll_interval_s)
                    continue
                if retry < self.max_retries:
                    delay = self._backoff_delay(retry)
                    retry += 1
                else:
                    delay = self.poll_interval_s
                    retry = 0
                await self._sleep(stop, delay)
        finally:
            await self._close_context()
            self.connected = False

    # ------------------------------------------------------------------ #
    # 健康检查
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        """健康检查：尝试一次 poll_once，成功返回 True。

        aiocoap 未安装时返回 False。
        """
        try:
            import aiocoap  # noqa: F401
        except ImportError:
            return False
        sample = await self.poll_once()
        return sample is not None

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    async def _sleep(self, stop: asyncio.Event, delay: float) -> None:
        """可被 stop 打断的 sleep。"""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=delay)

    async def _close_context(self) -> None:
        """关闭 aiocoap context（若已创建）。"""
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.shutdown()
            self._context = None
