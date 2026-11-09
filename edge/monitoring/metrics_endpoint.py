"""S3 Prometheus 指标 HTTP 端点。

使用 asyncio 原生 HTTP 实现（不依赖 aiohttp），提供：
- GET /metrics：返回 Prometheus 文本格式指标
- GET /health：返回 JSON 健康状态

响应 Content-Type 遵循 Prometheus 标准：
    text/plain; version=0.0.4; charset=utf-8

典型用法::

    metrics = EdgeMetrics()
    endpoint = MetricsEndpoint(metrics, host="0.0.0.0", port=9000)
    stop = asyncio.Event()
    await endpoint.serve(stop)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from edge.monitoring.metrics import EdgeMetrics

#: Prometheus 文本格式 Content-Type（遵循 exposition format v0.0.4）
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: 健康端点 JSON Content-Type
JSON_CONTENT_TYPE = "application/json; charset=utf-8"

#: 纯文本 Content-Type（用于错误响应）
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"


@dataclass
class HttpRequest:
    """简易 HTTP 请求表示，用于内部路由与测试。

    Attributes:
        method: HTTP 方法（GET / POST 等）
        path: 请求路径（含查询参数）
        headers: 请求头（键已小写化）
        body: 请求体原始字节
    """

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


class MetricsEndpoint:
    """Prometheus 指标 HTTP 端点。

    使用 ``asyncio.start_server`` 实现轻量 HTTP 服务，不依赖 aiohttp。
    提供 ``/metrics``（Prometheus 文本格式）与 ``/health``（JSON 健康状态）
    两个端点。

    Args:
        metrics: 边缘指标收集器实例
        host: 监听地址，默认 ``0.0.0.0``
        port: 监听端口，默认 ``9000``
    """

    def __init__(
        self,
        metrics: EdgeMetrics,
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        self._metrics = metrics
        self._host = host
        self._port = port

    @property
    def host(self) -> str:
        """监听地址。"""
        return self._host

    @property
    def port(self) -> int:
        """监听端口。"""
        return self._port

    def url(self) -> str:
        """返回端点基础 URL（含 scheme/host/port）。"""
        return f"http://{self._host}:{self._port}"

    async def handle_metrics(self, request: Any) -> str:
        """处理 ``/metrics`` 请求，返回 Prometheus 文本格式指标。

        Args:
            request: HTTP 请求对象（HttpRequest 或测试中传入的任意值）

        Returns:
            Prometheus exposition format 文本
        """
        return self._metrics.export_prometheus()

    async def handle_health(self, request: Any) -> str:
        """处理 ``/health`` 请求，返回 JSON 健康状态。

        Args:
            request: HTTP 请求对象（HttpRequest 或测试中传入的任意值）

        Returns:
            JSON 格式健康状态字符串
        """
        snap = self._metrics.snapshot()
        status: dict[str, Any] = {
            "status": "healthy",
            "metrics": snap,
        }
        return json.dumps(status, ensure_ascii=False)

    async def serve(self, stop: asyncio.Event) -> None:
        """启动 HTTP 服务，直到 ``stop`` 事件被 set。

        使用 ``asyncio.start_server`` 监听，每个连接在独立协程中处理。
        当 ``stop.is_set()`` 为 True 时停止接受新连接并退出。

        Args:
            stop: 停止事件，set 后服务退出
        """
        server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
        )
        try:
            await stop.wait()
        finally:
            server.close()
            await server.wait_closed()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """处理单个 TCP 连接：解析 HTTP 请求并路由到对应处理函数。"""
        try:
            request = await self._parse_request(reader)
            if request is None:
                await self._send_response(writer, 400, "Bad Request\n", TEXT_CONTENT_TYPE)
                return
            body, content_type, status_code = await self._route(request)
            await self._send_response(writer, status_code, body, content_type)
        except Exception:
            # 避免连接异常导致服务崩溃
            await self._send_response(writer, 500, "Internal Server Error\n", TEXT_CONTENT_TYPE)
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    async def _parse_request(self, reader: asyncio.StreamReader) -> HttpRequest | None:
        """从 StreamReader 解析 HTTP 请求行与头部。

        Returns:
            HttpRequest 或 None（解析失败时）
        """
        request_line = await reader.readline()
        if not request_line:
            return None
        try:
            parts = request_line.decode("iso-8859-1").strip().split()
            if len(parts) < 2:
                return None
            method, path = parts[0], parts[1]
        except (UnicodeDecodeError, IndexError):
            return None

        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            try:
                key, _, value = line.decode("iso-8859-1").strip().partition(":")
                if key:
                    headers[key.strip().lower()] = value.strip()
            except UnicodeDecodeError:
                continue

        return HttpRequest(method=method, path=path, headers=headers)

    async def _route(self, request: HttpRequest) -> tuple[str, str, int]:
        """根据请求路径路由到对应处理函数。

        Returns:
            (response_body, content_type, status_code)
        """
        # 去除查询参数
        path = request.path.split("?", 1)[0]

        if request.method != "GET":
            return "Method Not Allowed\n", TEXT_CONTENT_TYPE, 405

        if path == "/metrics":
            body = await self.handle_metrics(request)
            return body, PROMETHEUS_CONTENT_TYPE, 200

        if path == "/health":
            body = await self.handle_health(request)
            return body, JSON_CONTENT_TYPE, 200

        return "Not Found\n", TEXT_CONTENT_TYPE, 404

    @staticmethod
    async def _send_response(
        writer: asyncio.StreamWriter,
        status_code: int,
        body: str,
        content_type: str,
    ) -> None:
        """发送 HTTP/1.1 响应。"""
        reason = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(status_code, "Unknown")

        body_bytes = body.encode("utf-8")
        header_lines = [
            f"HTTP/1.1 {status_code} {reason}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body_bytes)}",
            "Connection: close",
            "",
            "",
        ]
        writer.write("\r\n".join(header_lines).encode("iso-8859-1") + body_bytes)
        await writer.drain()
