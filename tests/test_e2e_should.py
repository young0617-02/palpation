"""E2E Should 全链路集成测试。

验证 Should 项三个模块的端到端联动：
  1. CoAP 适配器（edge/adapters/coap_sensor.py）— 产出 InternalSample
  2. 规则表热更新（edge/stream/rule_loader.py + rules.py）— 评估样本/动态重载
  3. 看板指标导出（edge/monitoring/metrics_endpoint.py + deploy/*.yml + dashboard.json）

测试分层：
  A. CoAP + 规则引擎联动（4 个）
  B. 规则热更新 + 指标端点联动（4 个）
  C. 全链路集成（5 个）
  D. 边界与异常（4 个）

注意：
  - aiocoap 未安装，用 sys.modules 注入 fake 模块（参考 test_coap_enhanced.py）
  - 文件变化用 tmp_path + os.utime 触发 mtime 变化（Windows 兼容）
  - MetricsEndpoint 不启动 HTTP 服务，只测试 handle_metrics/handle_health 与 _route
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from edge.adapters.coap_sensor import CoapAdapter
from edge.monitoring.metrics import EdgeMetrics
from edge.monitoring.metrics_endpoint import (
    JSON_CONTENT_TYPE,
    PROMETHEUS_CONTENT_TYPE,
    TEXT_CONTENT_TYPE,
    HttpRequest,
    MetricsEndpoint,
)
from edge.stream.rule_loader import RuleHotReloader, RuleLoader
from edge.stream.rules import RuleEngine
from shared.schemas import InternalSample


# --------------------------------------------------------------------------- #
# 常量与辅助
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_YML = ROOT / "deploy" / "prometheus.yml"
ALERTS_YML = ROOT / "deploy" / "alerts.yml"
GRAFANA_DASHBOARD = ROOT / "deploy" / "grafana" / "dashboard.json"


def _make_rule(
    rid: str = "r1",
    metric: str = "temperature",
    op: str = ">",
    threshold: float = 80.0,
    severity: str = "high",
) -> dict[str, Any]:
    """构造一条有效规则。"""
    return {
        "id": rid,
        "metric": metric,
        "op": op,
        "threshold": threshold,
        "severity": severity,
    }


def _touch_future(path: Path, offset: float = 2.0) -> None:
    """将文件 mtime 设为未来时间，确保 has_changed 返回 True（Windows 兼容）。"""
    future = time.time() + offset
    os.utime(str(path), (future, future))


# --------------------------------------------------------------------------- #
# Fake aiocoap 基础设施（与 test_coap_enhanced.py 一致风格）
# --------------------------------------------------------------------------- #


class FakeCoapResponse:
    """模拟 aiocoap 响应消息。"""

    def __init__(self, payload: bytes = b'{"temperature": 25.0}') -> None:
        self.payload = payload


class FakeCoapRequestHandle:
    """模拟 aiocoap Request 对象（.response 为返回 awaitable 的 property）。"""

    def __init__(
        self,
        response: FakeCoapResponse | None = None,
        exc: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self._response = response
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


class FakeCoapContext:
    """模拟 aiocoap Protocol/Context。request() 同步返回 handle。"""

    def __init__(self, request_factory: Any | None = None) -> None:
        self._factory = request_factory
        self.requests: list[Any] = []

    def request(self, msg: Any) -> FakeCoapRequestHandle:
        self.requests.append(msg)
        if self._factory is not None:
            return self._factory(msg)
        return FakeCoapRequestHandle()

    async def shutdown(self) -> None:
        return None


@pytest.fixture()
def patch_aiocoap():
    """patch sys.modules['aiocoap']，返回 install(context) 函数。"""
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


# ============================================================================ #
# A. CoAP + 规则引擎联动（4 个）
# ============================================================================ #


class TestCoapRuleEngineIntegration:
    """CoAP 适配器产出样本 → RuleEngine 评估的端到端联动。"""

    async def test_coap_sample_evaluated_by_rule_engine(self, patch_aiocoap) -> None:
        """CoAP 产出 InternalSample → RuleEngine.evaluate() 正确触发规则命中。

        全链路：fake CoAP server 返回 {"temperature": 100.0}
                → CoapAdapter.poll_once() 产出 InternalSample
                → RuleEngine.evaluate(sample.metrics) 返回 RuleHit
        """
        ctx = FakeCoapContext(
            request_factory=lambda m: FakeCoapRequestHandle(
                response=FakeCoapResponse(b'{"temperature": 100.0, "humidity": 50.0}')
            )
        )
        patch_aiocoap(ctx)
        adapter = CoapAdapter(device_id="pump-01")
        engine = RuleEngine([_make_rule("temp_high", "temperature", ">", 80.0, "high")])

        sample = await adapter.poll_once()
        assert sample is not None, "CoAP 应产出样本"
        assert sample.device_id == "pump-01"
        assert sample.metrics["temperature"] == 100.0

        hits = engine.evaluate(sample.metrics)
        assert len(hits) == 1, "温度 100 > 80 应触发规则"
        assert hits[0].rule_id == "temp_high"
        assert hits[0].value == 100.0
        assert hits[0].threshold == 80.0
        assert hits[0].severity == "high"

    async def test_coap_stats_consistency_with_rule_hits(self, patch_aiocoap) -> None:
        """CoAP 统计信息与规则命中结果一致性。

        多次轮询：温度从正常→越限，统计 successful 次数与规则命中次数匹配。
        """
        payloads = [
            b'{"temperature": 50.0}',  # 正常
            b'{"temperature": 70.0}',  # 正常
            b'{"temperature": 95.0}',  # 越限
            b'{"temperature": 110.0}',  # 越限
        ]
        handles = [
            FakeCoapRequestHandle(response=FakeCoapResponse(p)) for p in payloads
        ]
        factory_idx = 0

        def factory(msg: Any) -> FakeCoapRequestHandle:
            nonlocal factory_idx
            h = handles[factory_idx] if factory_idx < len(handles) else handles[-1]
            factory_idx += 1
            return h

        ctx = FakeCoapContext(request_factory=factory)
        patch_aiocoap(ctx)
        adapter = CoapAdapter()
        engine = RuleEngine([_make_rule("temp_high", "temperature", ">", 80.0)])

        hit_count = 0
        for _ in range(4):
            sample = await adapter.poll_once()
            assert sample is not None
            hits = engine.evaluate(sample.metrics)
            hit_count += len(hits)

        st = adapter.stats()
        assert st["successful"] == 4, "4 次轮询全部成功"
        assert st["failed"] == 0
        assert hit_count == 2, "后两次温度越限应触发 2 次规则命中"

    async def test_coap_idle_rule_engine_works_independently(self) -> None:
        """CoAP idle 模式（aiocoap 未安装）下规则引擎仍可独立工作。

        验证 Should 项的优雅降级：CoAP 不可用不影响规则引擎核心功能。
        """
        with no_aiocoap():
            adapter = CoapAdapter()
            engine = RuleEngine(
                [
                    _make_rule("temp_high", "temperature", ">", 80.0),
                    _make_rule("vib_high", "vibration", ">", 5.0, "medium"),
                ]
            )

            # CoAP idle：poll_once 返回 None
            sample = await adapter.poll_once()
            assert sample is None, "aiocoap 未安装时 poll_once 应返回 None"
            assert adapter.connected is False

            # 规则引擎仍可独立评估任意指标
            hits = engine.evaluate({"temperature": 100.0, "vibration": 8.0})
            assert {h.rule_id for h in hits} == {"temp_high", "vib_high"}

            # 健康检查返回 False 但不抛异常
            ok = await adapter.health_check()
            assert ok is False

    async def test_coap_sample_multiple_metrics_multiple_rules(self, patch_aiocoap) -> None:
        """CoAP 样本含多指标 → 多条规则同时触发，severity 多级。

        验证 CoAP payload 与规则引擎多维度评估的端到端联动。
        """
        ctx = FakeCoapContext(
            request_factory=lambda m: FakeCoapRequestHandle(
                response=FakeCoapResponse(
                    b'{"temperature": 95.0, "vibration_rms": 6.0, "current": 12.0}'
                )
            )
        )
        patch_aiocoap(ctx)
        adapter = CoapAdapter(device_id="motor-03")
        engine = RuleEngine(
            [
                _make_rule("temp_high", "temperature", ">", 80.0, "high"),
                _make_rule("vib_high", "vibration_rms", ">", 5.0, "medium"),
                _make_rule("current_low", "current", "<", 10.0, "low"),
            ]
        )

        sample = await adapter.poll_once()
        assert sample is not None
        assert sample.source == "coap"

        hits = engine.evaluate(sample.metrics)
        # temperature 95 > 80 触发，vibration 6 > 5 触发，current 12 < 10 不触发
        assert {h.rule_id for h in hits} == {"temp_high", "vib_high"}
        severities = {h.severity for h in hits}
        assert severities == {"high", "medium"}


# ============================================================================ #
# B. 规则热更新 + 指标端点联动（4 个）
# ============================================================================ #


class TestRuleReloadMetricsIntegration:
    """规则热更新与 MetricsEndpoint 端点的端到端联动。"""

    def test_rule_hot_reload_then_metrics_export(self, tmp_path: Path) -> None:
        """规则文件变更 → RuleHotReloader 自动重载 → MetricsEndpoint 导出成功。

        全链路：写规则文件 → 启动热更新 → 修改文件 → 自动重载
                → MetricsEndpoint.handle_metrics() 仍可正常导出。
        """
        rule_file = tmp_path / "rules.json"
        initial_rules = [_make_rule("r1", "temperature", ">", 80.0)]
        rule_file.write_text(json.dumps(initial_rules), encoding="utf-8")

        engine = RuleEngine([])
        reloader = RuleHotReloader(rule_file, engine, poll_interval_s=0.05)
        reloader.force_reload()
        assert engine.get_version() == 1

        metrics = EdgeMetrics()
        endpoint = MetricsEndpoint(metrics, host="127.0.0.1", port=9001)

        # 启动后台监听
        reloader.start()
        # 修改规则文件
        new_rules = [
            _make_rule("r1", "temperature", ">", 50.0),
            _make_rule("r2", "vibration", ">", 4.0, "medium"),
        ]
        rule_file.write_text(json.dumps(new_rules), encoding="utf-8")
        _touch_future(rule_file)
        time.sleep(0.3)
        reloader.stop()

        # 验证自动重载生效
        assert engine.get_version() == 2, "自动重载应使版本 +1"
        assert engine.get_rule_ids() == {"r1", "r2"}

        # 验证指标端点仍可导出（同步调用 handle_metrics 的底层 export_prometheus）
        prom_text = metrics.export_prometheus()
        assert "edge_inference_p95_ms" in prom_text
        # handle_metrics 是 async，用 asyncio.run 验证
        body = asyncio.run(endpoint.handle_metrics(HttpRequest("GET", "/metrics")))
        assert "edge_queue_depth" in body, "指标端点应能导出重载后的指标"

    def test_rule_validation_failure_preserves_old_rules(
        self, tmp_path: Path
    ) -> None:
        """规则校验失败 → 旧规则保留 → 指标端点仍可导出。

        验证优雅降级：无效规则不破坏现有功能。
        """
        rule_file = tmp_path / "rules.json"
        valid_rules = [_make_rule("r1", "temperature", ">", 80.0)]
        rule_file.write_text(json.dumps(valid_rules), encoding="utf-8")

        engine = RuleEngine([])
        reloader = RuleHotReloader(rule_file, engine, poll_interval_s=0.05)
        reloader.force_reload()
        original_version = engine.get_version()

        metrics = EdgeMetrics()
        metrics.set_model_version("v1.0.0")
        endpoint = MetricsEndpoint(metrics)

        # 写入无效规则
        reloader.start()
        bad_rules = [{"id": "bad", "op": "!="}]  # 缺字段 + 非法 op
        rule_file.write_text(json.dumps(bad_rules), encoding="utf-8")
        _touch_future(rule_file)
        time.sleep(0.3)
        reloader.stop()

        # 旧规则保留
        assert engine.get_version() == original_version, "校验失败不应改变版本"
        assert engine.get_rule_ids() == {"r1"}, "旧规则应保留"
        assert len(reloader.errors) >= 1, "应记录校验错误"

        # 指标端点仍可导出
        body = asyncio.run(endpoint.handle_metrics(HttpRequest("GET", "/metrics")))
        assert "edge_model_version_info" in body
        health_body = asyncio.run(endpoint.handle_health(HttpRequest("GET", "/health")))
        health_data = json.loads(health_body)
        assert health_data["status"] == "healthy"

    def test_force_reload_then_health_endpoint_ok(self, tmp_path: Path) -> None:
        """force_reload 后指标端点健康检查正常。

        全链路：force_reload → 指标记录 → /health 端点返回 healthy + 指标快照。
        """
        rule_file = tmp_path / "rules.json"
        rules = [_make_rule("r1", "temperature", ">", 80.0)]
        rule_file.write_text(json.dumps(rules), encoding="utf-8")

        engine = RuleEngine([])
        reloader = RuleHotReloader(rule_file, engine)
        v = reloader.force_reload()
        assert v == 1
        assert reloader.reload_count() == 1

        metrics = EdgeMetrics()
        metrics.record_inference(120.0)
        metrics.set_model_version("v2.1.0")
        metrics.set_queue_depth(5)
        endpoint = MetricsEndpoint(metrics, host="0.0.0.0", port=9000)

        # /health 端点返回 healthy
        body = asyncio.run(endpoint.handle_health(HttpRequest("GET", "/health")))
        data = json.loads(body)
        assert data["status"] == "healthy"
        snap = data["metrics"]
        assert snap["model_version"] == "v2.1.0"
        assert snap["queue_depth"] == 5
        assert snap["inference_samples"] == 1

    def test_reload_version_reflected_in_snapshot(self, tmp_path: Path) -> None:
        """多次重载后版本递增，指标快照反映最新状态。

        验证热更新与指标端点的版本一致性。
        """
        rule_file = tmp_path / "rules.json"
        rule_file.write_text(
            json.dumps([_make_rule("r1", "temperature", ">", 80.0)]),
            encoding="utf-8",
        )
        engine = RuleEngine([])
        reloader = RuleHotReloader(rule_file, engine)

        versions = [reloader.force_reload()]
        for i in range(2, 5):
            rule_file.write_text(
                json.dumps([_make_rule(f"r{i}", "temperature", ">", float(i) * 10)]),
                encoding="utf-8",
            )
            versions.append(reloader.force_reload())

        assert versions == [1, 2, 3, 4], "版本应随重载递增"
        assert reloader.reload_count() == 4

        # 指标快照
        metrics = EdgeMetrics()
        for v in versions:
            metrics.record_inference(float(v) * 10)
        snap = metrics.snapshot()
        assert snap["inference_samples"] == 4
        endpoint = MetricsEndpoint(metrics)
        body = asyncio.run(endpoint.handle_health(HttpRequest("GET", "/health")))
        data = json.loads(body)
        assert data["metrics"]["inference_samples"] == 4


# ============================================================================ #
# C. 全链路集成（5 个）
# ============================================================================ #


class TestFullPipelineIntegration:
    """CoAP → 规则评估 → 指标记录 → Prometheus 导出 全链路集成。"""

    async def test_full_pipeline_coap_to_prometheus(self, patch_aiocoap) -> None:
        """全链路：CoAP 产出 → 规则评估 → 指标记录 → Prometheus 导出。

        验证三个模块数据流的端到端正确性。
        """
        ctx = FakeCoapContext(
            request_factory=lambda m: FakeCoapRequestHandle(
                response=FakeCoapResponse(
                    b'{"temperature": 95.0, "vibration_rms": 6.5}'
                )
            )
        )
        patch_aiocoap(ctx)
        adapter = CoapAdapter(device_id="pump-01")
        engine = RuleEngine(
            [
                _make_rule("temp_high", "temperature", ">", 80.0, "high"),
                _make_rule("vib_high", "vibration_rms", ">", 5.0, "medium"),
            ]
        )
        metrics = EdgeMetrics()
        endpoint = MetricsEndpoint(metrics)

        # 1. CoAP 产出样本
        sample = await adapter.poll_once()
        assert sample is not None

        # 2. 规则评估
        hits = engine.evaluate(sample.metrics)
        assert len(hits) == 2

        # 3. 指标记录（模拟推理延迟）
        for hit in hits:
            metrics.record_inference(150.0)
        metrics.set_model_version("v1.0.0")
        metrics.set_queue_depth(len(hits))

        # 4. Prometheus 导出
        prom_text = await endpoint.handle_metrics(HttpRequest("GET", "/metrics"))
        assert "edge_inference_p95_ms" in prom_text
        assert "edge_model_version_info" in prom_text
        assert "edge_queue_depth 2" in prom_text

        # 5. /health 端点
        health = await endpoint.handle_health(HttpRequest("GET", "/health"))
        health_data = json.loads(health)
        assert health_data["status"] == "healthy"
        assert health_data["metrics"]["inference_samples"] == 2

    def test_rule_reload_new_rule_applies_to_subsequent_samples(
        self, tmp_path: Path
    ) -> None:
        """规则热更新后 → 新规则对后续样本生效 → 指标反映变化。

        验证热更新对后续评估的实时生效。
        """
        rule_file = tmp_path / "rules.json"
        rule_file.write_text(
            json.dumps([_make_rule("r1", "temperature", ">", 80.0)]),
            encoding="utf-8",
        )
        engine = RuleEngine([])
        reloader = RuleHotReloader(rule_file, engine, poll_interval_s=0.05)
        reloader.force_reload()

        metrics = EdgeMetrics()

        # 旧规则：temperature=70 不触发（阈值 80）
        hits_old = engine.evaluate({"temperature": 70.0})
        assert len(hits_old) == 0
        metrics.record_inference(50.0)

        # 热更新：阈值改为 50
        reloader.start()
        rule_file.write_text(
            json.dumps([_make_rule("r1", "temperature", ">", 50.0)]),
            encoding="utf-8",
        )
        _touch_future(rule_file)
        time.sleep(0.3)
        reloader.stop()

        assert engine.get_version() == 2

        # 新规则：temperature=70 触发（阈值 50）
        hits_new = engine.evaluate({"temperature": 70.0})
        assert len(hits_new) == 1
        assert hits_new[0].threshold == 50.0
        metrics.record_inference(100.0)

        # 指标反映变化
        snap = metrics.snapshot()
        assert snap["inference_samples"] == 2
        assert snap["inference_p95_ms"] >= 50.0

    def test_config_files_cross_consistency(self) -> None:
        """配置文件（prometheus.yml + alerts.yml + dashboard.json）一致性验证。

        验证三个配置文件的跨文件一致性：
        - prometheus.yml 引用 alerts.yml
        - 端口/路径跨文件一致
        - dashboard 指标名与 prometheus.yml 采集目标匹配
        """
        # 1. prometheus.yml
        assert PROMETHEUS_YML.exists(), "prometheus.yml 必须存在"
        prom_data = yaml.safe_load(PROMETHEUS_YML.read_text(encoding="utf-8"))
        jobs = {j["job_name"]: j for j in prom_data["scrape_configs"]}
        assert "edge-agent" in jobs
        edge_targets = jobs["edge-agent"]["static_configs"][0]["targets"]
        assert any("9000" in t for t in edge_targets)
        assert jobs["edge-agent"].get("metrics_path") == "/metrics"

        # 2. alerts.yml 被 prometheus.yml 引用
        assert ALERTS_YML.exists(), "alerts.yml 必须存在"
        rule_files = prom_data.get("rule_files", [])
        assert "alerts.yml" in rule_files, "prometheus.yml 应引用 alerts.yml"
        alerts_data = yaml.safe_load(ALERTS_YML.read_text(encoding="utf-8"))
        alert_rules = alerts_data["groups"][0]["rules"]
        assert len(alert_rules) >= 6, "alerts.yml 应至少 6 条规则"

        # 3. dashboard.json
        assert GRAFANA_DASHBOARD.exists(), "dashboard.json 必须存在"
        dashboard = json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))
        panels = dashboard["panels"]
        assert len(panels) >= 10, "dashboard 应至少 10 个面板"
        grafana_alerts = dashboard.get("alertRuleGroup", {}).get("rules", [])
        assert len(grafana_alerts) >= 6, "dashboard 应至少 6 条告警规则"

        # 4. 跨文件一致性：dashboard refresh 与 prometheus scrape_interval 量级合理
        assert dashboard.get("refresh") == "10s"
        assert jobs["edge-agent"].get("scrape_interval") == "10s"

    async def test_metrics_endpoint_metrics_format(self) -> None:
        """MetricsEndpoint /metrics 端点返回正确 Prometheus 格式。"""
        metrics = EdgeMetrics()
        metrics.record_inference(200.0)
        metrics.record_inference(300.0)
        metrics.set_model_version("v3.0.0")
        metrics.set_queue_depth(10)
        metrics.set_offline_duration(60)
        metrics.record_cache_hit(True)
        metrics.record_cache_hit(False)
        metrics.record_cloud_escalation(False)
        endpoint = MetricsEndpoint(metrics)

        body = await endpoint.handle_metrics(HttpRequest("GET", "/metrics"))
        lines = body.strip().split("\n")

        # 必须包含所有指标
        expected_metrics = {
            "edge_inference_p95_ms",
            "edge_model_version_info",
            "edge_queue_depth",
            "edge_offline_duration_s",
            "edge_local_cache_hit_rate",
            "edge_cloud_escalation_rate",
        }
        help_metrics = {
            ln.split()[2] for ln in lines if ln.startswith("# HELP")
        }
        assert expected_metrics.issubset(help_metrics), (
            f"缺少指标：{expected_metrics - help_metrics}"
        )

        # 每个指标都有 HELP/TYPE/值行
        for name in expected_metrics:
            assert any(ln.startswith(f"# HELP {name}") for ln in lines), f"缺少 {name} HELP"
            assert any(ln.startswith(f"# TYPE {name}") for ln in lines), f"缺少 {name} TYPE"
            assert any(
                ln.startswith(name) and not ln.startswith("#") for ln in lines
            ), f"缺少 {name} 值行"

        # model_version label 正确
        version_line = next(
            ln for ln in lines if ln.startswith("edge_model_version_info{")
        )
        assert 'version="v3.0.0"' in version_line

    async def test_metrics_endpoint_health_format(self) -> None:
        """MetricsEndpoint /health 端点返回正确 JSON 格式。"""
        metrics = EdgeMetrics()
        metrics.record_inference(150.0)
        metrics.set_model_version("v1.5.0")
        metrics.set_queue_depth(42)
        endpoint = MetricsEndpoint(metrics, host="127.0.0.1", port=9001)

        body = await endpoint.handle_health(HttpRequest("GET", "/health"))
        data = json.loads(body)

        assert data["status"] == "healthy"
        snap = data["metrics"]
        expected_keys = {
            "inference_p95_ms",
            "model_version",
            "queue_depth",
            "offline_duration_s",
            "local_cache_hit_rate",
            "cloud_escalation_rate",
            "inference_samples",
            "cache_hits",
            "cache_total",
            "cloud_escalations",
            "cloud_total",
        }
        assert expected_keys.issubset(snap.keys()), (
            f"health 快照缺少键：{expected_keys - snap.keys()}"
        )
        assert snap["model_version"] == "v1.5.0"
        assert snap["queue_depth"] == 42
        assert snap["inference_samples"] == 1


# ============================================================================ #
# D. 边界与异常（4 个）
# ============================================================================ #


class TestBoundaryAndException:
    """边界条件与异常路径的端到端处理。"""

    async def test_empty_rules_with_coap_idle_no_exception(self) -> None:
        """空规则列表 + CoAP idle → 无异常，端点正常导出。

        验证最小配置下的系统稳定性。
        """
        with no_aiocoap():
            adapter = CoapAdapter()
            engine = RuleEngine([])
            metrics = EdgeMetrics()
            endpoint = MetricsEndpoint(metrics)

            # CoAP idle
            sample = await adapter.poll_once()
            assert sample is None

            # 空规则评估任意指标不触发
            hits = engine.evaluate({"temperature": 999.0})
            assert hits == []

            # 指标端点正常导出
            body = await endpoint.handle_metrics(HttpRequest("GET", "/metrics"))
            assert "edge_inference_p95_ms" in body
            health = await endpoint.handle_health(HttpRequest("GET", "/health"))
            data = json.loads(health)
            assert data["status"] == "healthy"
            assert data["metrics"]["inference_samples"] == 0

    def test_invalid_rule_file_graceful_degradation(self, tmp_path: Path) -> None:
        """无效规则文件 → 优雅降级，旧规则保留，端点可用。

        验证 force_reload 抛 ValueError 后系统仍可用。
        """
        rule_file = tmp_path / "rules.json"
        # 先写有效规则
        rule_file.write_text(
            json.dumps([_make_rule("r1", "temperature", ">", 80.0)]),
            encoding="utf-8",
        )
        engine = RuleEngine([])
        reloader = RuleHotReloader(rule_file, engine)
        reloader.force_reload()
        assert engine.get_rule_ids() == {"r1"}

        # 写入无效规则
        rule_file.write_text(
            json.dumps([{"id": "bad", "op": "!="}]),
            encoding="utf-8",
        )

        # force_reload 抛 ValueError
        with pytest.raises(ValueError):
            reloader.force_reload()

        # 旧规则保留
        assert engine.get_rule_ids() == {"r1"}
        assert reloader.reload_count() == 1, "失败重载不应计入 reload_count"

        # 指标端点仍可用
        metrics = EdgeMetrics()
        endpoint = MetricsEndpoint(metrics)
        body = asyncio.run(endpoint.handle_metrics(HttpRequest("GET", "/metrics")))
        assert "edge_inference_p95_ms" in body

    async def test_metrics_endpoint_routing_unknown_path(self) -> None:
        """MetricsEndpoint 路由：未知路径返回 404。"""
        metrics = EdgeMetrics()
        endpoint = MetricsEndpoint(metrics)

        request = HttpRequest(method="GET", path="/unknown")
        body, content_type, status = await endpoint._route(request)
        assert status == 404
        assert content_type == TEXT_CONTENT_TYPE
        assert "Not Found" in body

    async def test_metrics_endpoint_routing_non_get_method(self) -> None:
        """MetricsEndpoint 路由：非 GET 方法返回 405。"""
        metrics = EdgeMetrics()
        endpoint = MetricsEndpoint(metrics)

        # POST /metrics → 405
        request = HttpRequest(method="POST", path="/metrics")
        body, content_type, status = await endpoint._route(request)
        assert status == 405
        assert content_type == TEXT_CONTENT_TYPE
        assert "Method Not Allowed" in body

        # POST /health → 405
        request = HttpRequest(method="POST", path="/health")
        body, content_type, status = await endpoint._route(request)
        assert status == 405

        # DELETE /metrics → 405
        request = HttpRequest(method="DELETE", path="/metrics")
        body, content_type, status = await endpoint._route(request)
        assert status == 405

    async def test_metrics_endpoint_routing_known_paths(self) -> None:
        """MetricsEndpoint 路由：/metrics 和 /health 返回 200 + 正确 Content-Type。"""
        metrics = EdgeMetrics()
        metrics.record_inference(100.0)
        endpoint = MetricsEndpoint(metrics)

        # /metrics → 200 + PROMETHEUS_CONTENT_TYPE
        request = HttpRequest(method="GET", path="/metrics")
        body, content_type, status = await endpoint._route(request)
        assert status == 200
        assert content_type == PROMETHEUS_CONTENT_TYPE
        assert "edge_inference_p95_ms" in body

        # /health → 200 + JSON_CONTENT_TYPE
        request = HttpRequest(method="GET", path="/health")
        body, content_type, status = await endpoint._route(request)
        assert status == 200
        assert content_type == JSON_CONTENT_TYPE
        data = json.loads(body)
        assert data["status"] == "healthy"

        # 带查询参数的 /metrics → 200
        request = HttpRequest(method="GET", path="/metrics?foo=bar")
        body, content_type, status = await endpoint._route(request)
        assert status == 200
        assert content_type == PROMETHEUS_CONTENT_TYPE