"""CoAP 低功耗链路增强测试。

测试分层（aiocoap 未安装，全程用 fake aiocoap 模块注入 sys.modules）：
1. 初始化参数（默认值/自定义值）
2. 退避延迟计算 _backoff_delay
3. 单次轮询 poll_once（成功/无效负载/超时/连接异常）
4. 轮询循环 run（正常产出/断连重连/退避重置/超过最大重试/优雅退出）
5. observe 推送模式（产出推送/订阅断开重连）
6. 统计信息 stats
7. 健康检查 health_check
8. 低功耗长间隔场景
9. aiocoap 未安装降级 idle
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import types
from typing import Any

import pytest

from edge.adapters.coap_sensor import CoapAdapter
from shared.schemas import InternalSample


# --------------------------------------------------------------------------- #
# Fake aiocoap 基础设施
# --------------------------------------------------------------------------- #


class FakeCoapResponse:
    """模拟 aiocoap 响应消息。"""

    def __init__(self, payload: bytes = b'{"temperature": 25.0, "humidity": 60.0}') -> None:
        self.payload = payload


class FakeCoapRequestHandle:
    """模拟 aiocoap Request 对象（.response / .observation 属性）。

    response/observation 均为返回 awaitable 的 property，与 aiocoap 真实 API 一致。
    """

    def __init__(
        self,
        response: FakeCoapResponse | None = None,
        observation: Any | None = None,
        exc: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self._response = response
        self._observation = observation
        self._exc = exc
        self._delay = delay

    @property
    def response(self) -> Any:
        resp = self._response if self._response is not None else FakeCoapResponse()
        exc = self._exc
        delay = self._delay

        async def _coro() -> FakeCoapResponse:
            if delay > 0:
                await asyncio.sleep(delay)
            if exc is not None:
                raise exc
            return resp

        return _coro()

    @property
    def observation(self) -> Any:
        obs = self._observation
        exc = self._exc
        delay = self._delay

        async def _coro() -> Any:
            if delay > 0:
                await asyncio.sleep(delay)
            if exc is not None:
                raise exc
            return obs

        return _coro()


class FakeCoapContext:
    """模拟 aiocoap Protocol/Context。request() 同步返回 handle（与真实 API 一致）。"""

    def __init__(self, request_factory: Any | None = None) -> None:
        self._factory = request_factory
        self.requests: list[Any] = []
        self.shutdown_calls = 0

    def request(self, msg: Any) -> FakeCoapRequestHandle:
        self.requests.append(msg)
        if self._factory is not None:
            return self._factory(msg)
        return FakeCoapRequestHandle()

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeObservation:
    """模拟 aiocoap Observation（async iterable）。迭代完消息后结束（订阅断开）。"""

    def __init__(self, messages: list[FakeCoapResponse] | None = None) -> None:
        self._messages = list(messages or [])

    def __aiter__(self) -> FakeObservation:
        return self

    async def __anext__(self) -> FakeCoapResponse:
        if self._messages:
            return self._messages.pop(0)
        raise StopAsyncIteration


class ScriptedFactory:
    """按脚本顺序返回 request handle，耗尽后返回默认成功 handle。"""

    def __init__(self, handles: list[FakeCoapRequestHandle]) -> None:
        self._handles = list(handles)
        self.calls = 0

    def __call__(self, msg: Any) -> FakeCoapRequestHandle:
        self.calls += 1
        if self._handles:
            return self._handles.pop(0)
        return FakeCoapRequestHandle()


@pytest.fixture()
def patch_aiocoap():
    """patch sys.modules['aiocoap']，返回 install(context) 函数。

    用法：
        ctx = FakeCoapContext(request_factory=...)
        patch_aiocoap(ctx)
    """
    original = sys.modules.get("aiocoap")

    class _Msg:
        def __init__(
            self, code: Any = None, uri: str | None = None, observe: Any = None
        ) -> None:
            self.code = code
            self.uri = uri
            self.observe = observe

    def install(context: FakeCoapContext) -> None:
        class _Ctx:
            @staticmethod
            async def create_client_context() -> FakeCoapContext:
                return context

        mod = types.ModuleType("aiocoap")
        mod.GET = "GET"  # type: ignore[attr-defined]
        mod.OBSERVE = 0  # type: ignore[attr-defined]
        mod.Message = _Msg  # type: ignore[attr-defined]
        mod.Context = _Ctx  # type: ignore[attr-defined]
        sys.modules["aiocoap"] = mod

    yield install

    if original is not None:
        sys.modules["aiocoap"] = original
    else:
        sys.modules.pop("aiocoap", None)


@contextlib.contextmanager
def no_aiocoap():
    """临时移除 aiocoap 模块，模拟未安装。"""
    original = sys.modules.pop("aiocoap", None)
    try:
        yield
    finally:
        if original is not None:
            sys.modules["aiocoap"] = original


# --------------------------------------------------------------------------- #
# 1. 初始化参数
# --------------------------------------------------------------------------- #


def test_default_init():
    """默认参数符合低功耗链路规格。"""
    a = CoapAdapter()
    assert a.uri == "coap://127.0.0.1:5683/sensor"
    assert a.device_id == "coap-01"
    assert a.poll_interval_s == 10.0
    assert a.max_retries == 5
    assert a.backoff_base_s == 1.0
    assert a.backoff_max_s == 60.0
    assert a.observe_mode is False
    assert a.timeout_s == 5.0
    assert a.connected is False
    assert a.name == "coap"


def test_custom_init():
    """自定义参数全部生效。"""
    a = CoapAdapter(
        "coap://sensor.local:5683/metrics",
        "coap-pump-07",
        poll_interval_s=60.0,
        max_retries=3,
        backoff_base_s=0.5,
        backoff_max_s=30.0,
        observe_mode=True,
        timeout_s=10.0,
    )
    assert a.uri == "coap://sensor.local:5683/metrics"
    assert a.device_id == "coap-pump-07"
    assert a.poll_interval_s == 60.0
    assert a.max_retries == 3
    assert a.backoff_base_s == 0.5
    assert a.backoff_max_s == 30.0
    assert a.observe_mode is True
    assert a.timeout_s == 10.0


def test_init_keyword_only_extra_params():
    """新增参数为关键字-only，避免位置参数误用。"""
    with pytest.raises(TypeError):
        CoapAdapter("coap://x", "dev", 60.0)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 2. 退避延迟计算
# --------------------------------------------------------------------------- #


def test_backoff_delay_retry0_equals_base():
    """retry=0 时退避等于基数。"""
    a = CoapAdapter(backoff_base_s=1.0, backoff_max_s=60.0)
    assert a._backoff_delay(0) == 1.0


def test_backoff_delay_sequence_doubles():
    """退避随 retry 指数翻倍：1, 2, 4, 8。"""
    a = CoapAdapter(backoff_base_s=1.0, backoff_max_s=60.0)
    assert a._backoff_delay(0) == 1.0
    assert a._backoff_delay(1) == 2.0
    assert a._backoff_delay(2) == 4.0
    assert a._backoff_delay(3) == 8.0


def test_backoff_delay_capped_at_max():
    """退避不超过上限。"""
    a = CoapAdapter(backoff_base_s=1.0, backoff_max_s=10.0)
    assert a._backoff_delay(10) == 10.0  # 2^10=1024 > 10
    assert a._backoff_delay(100) == 10.0


def test_backoff_delay_custom_base_and_max():
    """自定义 base/max 下退避序列正确。"""
    a = CoapAdapter(backoff_base_s=0.5, backoff_max_s=30.0)
    assert a._backoff_delay(0) == 0.5
    assert a._backoff_delay(1) == 1.0
    assert a._backoff_delay(2) == 2.0
    assert a._backoff_delay(10) == 30.0  # 0.5*1024=512 > 30


# --------------------------------------------------------------------------- #
# 3. 单次轮询 poll_once
# --------------------------------------------------------------------------- #


async def test_poll_once_success(patch_aiocoap):
    """成功轮询产出 InternalSample，统计与连接状态更新。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b'{"temperature": 25.0, "humidity": 60.0}')
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter()
    sample = await adapter.poll_once()
    assert sample is not None
    assert sample.device_id == "coap-01"
    assert sample.source == "coap"
    assert sample.ts.tzinfo is not None
    assert sample.metrics["temperature"] == 25.0
    assert sample.metrics["humidity"] == 60.0
    assert adapter.connected is True
    assert len(ctx.requests) == 1
    st = adapter.stats()
    assert st["total_requests"] == 1
    assert st["successful"] == 1
    assert st["failed"] == 0


async def test_poll_once_invalid_payload(patch_aiocoap):
    """无效 JSON 负载返回 None，记一次失败。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b"not-json")
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter()
    sample = await adapter.poll_once()
    assert sample is None
    assert adapter.connected is False
    assert adapter.stats()["failed"] == 1
    assert adapter.stats()["successful"] == 0


async def test_poll_once_timeout(patch_aiocoap):
    """请求超时返回 None，记一次失败。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(delay=1.0)
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(timeout_s=0.05)
    sample = await adapter.poll_once()
    assert sample is None
    assert adapter.connected is False
    assert adapter.stats()["failed"] == 1


async def test_poll_once_connection_error(patch_aiocoap):
    """请求抛异常时返回 None，记一次失败。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(exc=RuntimeError("conn refused"))
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter()
    sample = await adapter.poll_once()
    assert sample is None
    assert adapter.connected is False
    assert adapter.stats()["failed"] == 1


# --------------------------------------------------------------------------- #
# 4. 轮询循环 run
# --------------------------------------------------------------------------- #


async def test_run_poll_loop_produces_samples(patch_aiocoap):
    """轮询循环持续产出样本。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b'{"t": 1.0}')
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(poll_interval_s=0.02, timeout_s=1.0)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    s1 = await asyncio.wait_for(queue.get(), timeout=3.0)
    s2 = await asyncio.wait_for(queue.get(), timeout=3.0)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    assert s1.metrics["t"] == 1.0
    assert s2.metrics["t"] == 1.0
    assert adapter.stats()["successful"] >= 2


async def test_run_reconnect_on_failure(patch_aiocoap):
    """前两次失败、第三次成功：产出样本且统计正确。"""
    factory = ScriptedFactory(
        [
            FakeCoapRequestHandle(exc=RuntimeError("conn lost")),
            FakeCoapRequestHandle(exc=RuntimeError("conn lost")),
            FakeCoapRequestHandle(response=FakeCoapResponse(b'{"t": 1.0}')),
        ]
    )
    ctx = FakeCoapContext(request_factory=factory)
    patch_aiocoap(ctx)
    adapter = CoapAdapter(
        poll_interval_s=0.01, backoff_base_s=0.01, backoff_max_s=0.1, timeout_s=1.0
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    sample = await asyncio.wait_for(queue.get(), timeout=3.0)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    assert sample.metrics["t"] == 1.0
    st = adapter.stats()
    assert st["successful"] == 1
    assert st["failed"] == 2


async def test_run_backoff_reset_on_success(patch_aiocoap):
    """成功后退避计数重置：再失败时延迟从 base 重新开始。"""
    factory = ScriptedFactory(
        [
            FakeCoapRequestHandle(exc=RuntimeError("e")),  # fail -> delay=base
            FakeCoapRequestHandle(exc=RuntimeError("e")),  # fail -> delay=2*base
            FakeCoapRequestHandle(response=FakeCoapResponse(b'{"t": 1.0}')),  # ok
            FakeCoapRequestHandle(exc=RuntimeError("e")),  # fail -> delay=base (reset)
            FakeCoapRequestHandle(response=FakeCoapResponse(b'{"t": 2.0}')),  # ok
        ]
    )
    ctx = FakeCoapContext(request_factory=factory)
    patch_aiocoap(ctx)
    adapter = CoapAdapter(
        poll_interval_s=0.01,
        max_retries=10,
        backoff_base_s=0.1,
        backoff_max_s=1.0,
        timeout_s=1.0,
    )
    delays: list[float] = []
    real_sleep = adapter._sleep

    async def spy_sleep(stop_ev: asyncio.Event, delay: float) -> None:
        delays.append(delay)
        await real_sleep(stop_ev, min(delay, 0.01))

    adapter._sleep = spy_sleep  # type: ignore[assignment]

    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.wait_for(queue.get(), timeout=3.0)
    await asyncio.wait_for(queue.get(), timeout=3.0)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    # 失败退避：base, 2*base；成功后 poll_interval；再失败从 base 开始
    assert delays[0] == pytest.approx(0.1)
    assert delays[1] == pytest.approx(0.2)
    assert delays[2] == pytest.approx(0.01)  # 成功后的轮询间隔
    assert delays[3] == pytest.approx(0.1)  # 退避已重置


async def test_run_max_retries_falls_to_long_interval(patch_aiocoap):
    """超过最大重试次数后改用 poll_interval_s 等待并重置计数。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(exc=RuntimeError("down"))
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(
        poll_interval_s=0.05,
        max_retries=2,
        backoff_base_s=0.01,
        backoff_max_s=0.05,
        timeout_s=0.5,
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.sleep(0.3)  # 足过若干退避周期
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    st = adapter.stats()
    assert st["failed"] >= 3
    assert st["successful"] == 0
    assert adapter.connected is False
    assert queue.empty()


async def test_run_stop_exits_cleanly(patch_aiocoap):
    """stop 设置后适配器优雅退出，connected 置 False。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b'{"t": 1.0}')
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(poll_interval_s=0.02, timeout_s=1.0)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    assert adapter.connected is False


# --------------------------------------------------------------------------- #
# 5. observe 推送模式
# --------------------------------------------------------------------------- #


async def test_observe_mode_produces_pushed_samples(patch_aiocoap):
    """observe 模式下服务器推送的消息被产出为样本。"""
    obs = FakeObservation(
        messages=[
            FakeCoapResponse(b'{"t": 1.0}'),
            FakeCoapResponse(b'{"t": 2.0}'),
        ]
    )
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(observation=obs)
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(
        observe_mode=True, timeout_s=1.0, poll_interval_s=0.01, backoff_base_s=0.01
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    s1 = await asyncio.wait_for(queue.get(), timeout=3.0)
    s2 = await asyncio.wait_for(queue.get(), timeout=3.0)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    assert s1.metrics["t"] == 1.0
    assert s2.metrics["t"] == 2.0
    assert adapter.stats()["successful"] == 2


async def test_observe_reconnect_on_subscription_end(patch_aiocoap):
    """订阅结束后按退避重连，新订阅继续产出。"""
    obs1 = FakeObservation(messages=[FakeCoapResponse(b'{"t": 1.0}')])
    obs2 = FakeObservation(messages=[FakeCoapResponse(b'{"t": 2.0}')])
    factory = ScriptedFactory(
        [
            FakeCoapRequestHandle(observation=obs1),
            FakeCoapRequestHandle(observation=obs2),
        ]
    )
    ctx = FakeCoapContext(request_factory=factory)
    patch_aiocoap(ctx)
    adapter = CoapAdapter(
        observe_mode=True, timeout_s=1.0, poll_interval_s=0.01, backoff_base_s=0.01
    )
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    s1 = await asyncio.wait_for(queue.get(), timeout=3.0)
    s2 = await asyncio.wait_for(queue.get(), timeout=3.0)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    assert s1.metrics["t"] == 1.0
    assert s2.metrics["t"] == 2.0


# --------------------------------------------------------------------------- #
# 6. 统计信息 stats
# --------------------------------------------------------------------------- #


def test_stats_initial():
    """初始统计全零且未连接。"""
    a = CoapAdapter()
    assert a.stats() == {
        "total_requests": 0,
        "successful": 0,
        "failed": 0,
        "avg_latency_ms": 0.0,
        "connected": False,
    }


async def test_stats_after_polls(patch_aiocoap):
    """多次轮询后统计聚合正确，平均延迟非负。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b'{"t": 1.0}')
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter()
    s1 = await adapter.poll_once()
    s2 = await adapter.poll_once()
    assert s1 is not None and s2 is not None
    st = adapter.stats()
    assert st["total_requests"] == 2
    assert st["successful"] == 2
    assert st["failed"] == 0
    assert st["avg_latency_ms"] >= 0.0
    assert st["connected"] is True


# --------------------------------------------------------------------------- #
# 7. 健康检查 health_check
# --------------------------------------------------------------------------- #


async def test_health_check_ok(patch_aiocoap):
    """链路正常时健康检查返回 True。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b'{"t": 1.0}')
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter()
    ok = await adapter.health_check()
    assert ok is True
    assert adapter.stats()["successful"] == 1


async def test_health_check_fail(patch_aiocoap):
    """链路异常时健康检查返回 False。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(exc=RuntimeError("down"))
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(timeout_s=0.5)
    ok = await adapter.health_check()
    assert ok is False
    assert adapter.stats()["failed"] == 1


# --------------------------------------------------------------------------- #
# 8. 低功耗长间隔场景
# --------------------------------------------------------------------------- #


def test_low_power_long_interval_config():
    """低功耗设备可配置 600s 长间隔。"""
    adapter = CoapAdapter(poll_interval_s=600.0)
    assert adapter.poll_interval_s == 600.0


async def test_low_power_long_interval_no_rapid_poll(patch_aiocoap):
    """长间隔下不会在短时间内重复请求（仅一次请求）。"""
    ctx = FakeCoapContext(
        request_factory=lambda m: FakeCoapRequestHandle(
            response=FakeCoapResponse(b'{"t": 1.0}')
        )
    )
    patch_aiocoap(ctx)
    adapter = CoapAdapter(poll_interval_s=10.0, backoff_base_s=0.01, timeout_s=1.0)
    queue: asyncio.Queue[InternalSample] = asyncio.Queue()
    stop = asyncio.Event()
    task = asyncio.create_task(adapter.run(queue, stop))
    await asyncio.wait_for(queue.get(), timeout=3.0)
    # 等待一段时间，因间隔 10s 不应有第二个样本
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(queue.get(), timeout=0.3)
    stop.set()
    await asyncio.wait_for(task, timeout=3.0)
    assert len(ctx.requests) == 1


# --------------------------------------------------------------------------- #
# 9. aiocoap 未安装降级
# --------------------------------------------------------------------------- #


async def test_idle_when_aiocoap_missing():
    """aiocoap 未安装时适配器 idle，不产出样本，stop 后优雅退出。"""
    with no_aiocoap():
        adapter = CoapAdapter(poll_interval_s=0.01)
        queue: asyncio.Queue[InternalSample] = asyncio.Queue()
        stop = asyncio.Event()
        task = asyncio.create_task(adapter.run(queue, stop))
        await asyncio.sleep(0.05)
        assert not task.done()
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert queue.empty()
        assert adapter.connected is False


async def test_health_check_false_when_aiocoap_missing():
    """aiocoap 未安装时健康检查返回 False。"""
    with no_aiocoap():
        adapter = CoapAdapter()
        assert await adapter.health_check() is False


async def test_poll_once_none_when_aiocoap_missing():
    """aiocoap 未安装时 poll_once 返回 None。"""
    with no_aiocoap():
        adapter = CoapAdapter()
        assert await adapter.poll_once() is None