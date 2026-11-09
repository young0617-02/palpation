"""S3 自动化看板完善测试。

验证范围：
  - deploy/prometheus.yml：scrape config / job 名称 / targets / scrape_interval
  - deploy/alerts.yml：告警规则存在 / 表达式正确 / severity 标签
  - deploy/grafana/dashboard.json：面板数量≥10 / 告警规则≥6 / templating / time range / refresh
  - edge/monitoring/metrics_endpoint.py：MetricsEndpoint 初始化 / URL / handle_metrics / handle_health
  - Prometheus 格式正确性：Content-Type / 文本格式
  - 配置文件一致性：端口 / 路径跨文件一致
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from edge.monitoring.metrics import EdgeMetrics
from edge.monitoring.metrics_endpoint import (
    JSON_CONTENT_TYPE,
    PROMETHEUS_CONTENT_TYPE,
    HttpRequest,
    MetricsEndpoint,
)

ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_YML = ROOT / "deploy" / "prometheus.yml"
ALERTS_YML = ROOT / "deploy" / "alerts.yml"
GRAFANA_DASHBOARD = ROOT / "deploy" / "grafana" / "dashboard.json"
COMPOSE_FILE = ROOT / "docker-compose.edge.yml"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    """解析 deploy/prometheus.yml。"""
    assert PROMETHEUS_YML.exists(), f"prometheus.yml 不存在：{PROMETHEUS_YML}"
    data = yaml.safe_load(PROMETHEUS_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "prometheus.yml 顶层不是 mapping"
    return data


@pytest.fixture(scope="module")
def alerts_config() -> dict:
    """解析 deploy/alerts.yml。"""
    assert ALERTS_YML.exists(), f"alerts.yml 不存在：{ALERTS_YML}"
    data = yaml.safe_load(ALERTS_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "alerts.yml 顶层不是 mapping"
    return data


@pytest.fixture(scope="module")
def dashboard() -> dict:
    """解析 deploy/grafana/dashboard.json。"""
    assert GRAFANA_DASHBOARD.exists(), f"dashboard.json 不存在：{GRAFANA_DASHBOARD}"
    return json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose_yaml() -> dict:
    """解析 docker-compose.edge.yml（用于跨文件一致性检查）。"""
    assert COMPOSE_FILE.exists(), f"docker-compose.edge.yml 不存在：{COMPOSE_FILE}"
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


@pytest.fixture()
def metrics_endpoint() -> MetricsEndpoint:
    """构造带示例指标的 MetricsEndpoint。"""
    metrics = EdgeMetrics()
    metrics.record_inference(150.0)
    metrics.set_model_version("v1.2.0")
    metrics.set_queue_depth(42)
    metrics.set_offline_duration(120)
    metrics.record_cache_hit(True)
    metrics.record_cache_hit(False)
    return MetricsEndpoint(metrics, host="0.0.0.0", port=9000)


# --------------------------------------------------------------------------- #
# 1. prometheus.yml 验证（4 个）
# --------------------------------------------------------------------------- #


class TestPrometheusConfig:
    """deploy/prometheus.yml 采集配置验证。"""

    def test_scrape_configs_exist(self, prometheus_config: dict) -> None:
        """scrape_configs 必须存在且包含至少 2 个作业。"""
        configs = prometheus_config.get("scrape_configs")
        assert configs, "prometheus.yml 缺少 scrape_configs"
        assert len(configs) >= 2, f"scrape_configs 应至少 2 个作业，实际 {len(configs)}"

    def test_edge_agent_job_config(self, prometheus_config: dict) -> None:
        """edge-agent 作业：job_name / targets / metrics_path / scrape_interval。"""
        jobs = {j["job_name"]: j for j in prometheus_config["scrape_configs"]}
        assert "edge-agent" in jobs, "缺少 edge-agent 作业"
        job = jobs["edge-agent"]
        targets = job["static_configs"][0]["targets"]
        assert "localhost:9000" in targets, f"edge-agent targets 应含 localhost:9000，实际 {targets}"
        assert job.get("metrics_path") == "/metrics", "edge-agent metrics_path 应为 /metrics"
        assert job.get("scrape_interval") == "10s", "edge-agent scrape_interval 应为 10s"

    def test_cloud_api_job_config(self, prometheus_config: dict) -> None:
        """cloud-api 作业：job_name / targets / metrics_path / scrape_interval。"""
        jobs = {j["job_name"]: j for j in prometheus_config["scrape_configs"]}
        assert "cloud-api" in jobs, "缺少 cloud-api 作业"
        job = jobs["cloud-api"]
        targets = job["static_configs"][0]["targets"]
        assert "localhost:8000" in targets, f"cloud-api targets 应含 localhost:8000，实际 {targets}"
        assert job.get("metrics_path") == "/metrics", "cloud-api metrics_path 应为 /metrics"
        assert job.get("scrape_interval") == "15s", "cloud-api scrape_interval 应为 15s"

    def test_alerting_and_rule_files(self, prometheus_config: dict) -> None:
        """alerting 与 rule_files 必须正确配置。"""
        rule_files = prometheus_config.get("rule_files")
        assert rule_files, "prometheus.yml 缺少 rule_files"
        assert "alerts.yml" in rule_files, "rule_files 应引用 alerts.yml"
        alerting = prometheus_config.get("alerting", {})
        managers = alerting.get("alertmanagers", [])
        assert managers, "缺少 alertmanagers 配置"
        targets = managers[0]["static_configs"][0]["targets"]
        assert "localhost:9093" in targets, "alertmanager target 应为 localhost:9093"


# --------------------------------------------------------------------------- #
# 2. alerts.yml 验证（4 个）
# --------------------------------------------------------------------------- #


class TestAlertsConfig:
    """deploy/alerts.yml 告警规则验证。"""

    def test_alert_rules_exist(self, alerts_config: dict) -> None:
        """必须包含至少 6 条告警规则。"""
        rules = alerts_config["groups"][0]["rules"]
        assert len(rules) >= 6, f"告警规则应至少 6 条，实际 {len(rules)}"
        alert_names = {r["alert"] for r in rules}
        expected = {
            "InferenceP95High",
            "QueueDepthHigh",
            "OfflineTooLong",
            "DiskSpaceLow",
            "ModelVersionMismatch",
            "HighCpuTemp",
        }
        missing = expected - alert_names
        assert not missing, f"缺少告警规则：{missing}"

    def test_alert_expressions(self, alerts_config: dict) -> None:
        """各告警表达式必须正确。"""
        rules = {r["alert"]: r for r in alerts_config["groups"][0]["rules"]}
        assert "edge_inference_p95_ms > 3000" in rules["InferenceP95High"]["expr"]
        assert "edge_queue_depth > 10000" in rules["QueueDepthHigh"]["expr"]
        assert "edge_offline_duration_s > 86400" in rules["OfflineTooLong"]["expr"]
        assert "edge_disk_free_mb < 500" in rules["DiskSpaceLow"]["expr"]
        assert "edge_cpu_temp_c > 80" in rules["HighCpuTemp"]["expr"]
        # 模型版本不匹配用 != 比较
        assert "!=" in rules["ModelVersionMismatch"]["expr"]

    def test_alert_severity_labels(self, alerts_config: dict) -> None:
        """每条规则必须有 severity 标签（warning 或 critical）。"""
        rules = alerts_config["groups"][0]["rules"]
        for r in rules:
            labels = r.get("labels", {})
            severity = labels.get("severity")
            assert severity in ("warning", "critical"), (
                f"告警 {r['alert']} severity 应为 warning/critical，实际 {severity}"
            )
        # ModelVersionMismatch 应为 critical
        mismatch = next(r for r in rules if r["alert"] == "ModelVersionMismatch")
        assert mismatch["labels"]["severity"] == "critical", (
            "ModelVersionMismatch 应为 critical 级别"
        )

    def test_alert_annotations_complete(self, alerts_config: dict) -> None:
        """每条规则必须有 summary 与 description 注解及 for 持续时间。"""
        rules = alerts_config["groups"][0]["rules"]
        for r in rules:
            ann = r.get("annotations", {})
            assert ann.get("summary"), f"告警 {r['alert']} 缺少 summary"
            assert ann.get("description"), f"告警 {r['alert']} 缺少 description"
            assert r.get("for"), f"告警 {r['alert']} 缺少 for 持续时间"


# --------------------------------------------------------------------------- #
# 3. dashboard.json 增强验证（5 个）
# --------------------------------------------------------------------------- #


class TestDashboardEnhanced:
    """deploy/grafana/dashboard.json 增强验证。"""

    def test_panel_count_at_least_10(self, dashboard: dict) -> None:
        """面板数量必须 ≥ 10。"""
        panels = dashboard["panels"]
        assert len(panels) >= 10, f"面板数量应 ≥ 10，实际 {len(panels)}"

    def test_alert_rule_count_at_least_6(self, dashboard: dict) -> None:
        """告警规则数量必须 ≥ 6。"""
        rules = dashboard.get("alertRuleGroup", {}).get("rules", [])
        assert len(rules) >= 6, f"告警规则应 ≥ 6，实际 {len(rules)}"

    def test_templating_has_device_id(self, dashboard: dict) -> None:
        """templating 必须包含 device_id 变量（按设备 ID 筛选）。"""
        tmpl_list = dashboard.get("templating", {}).get("list", [])
        var_names = {v.get("name") for v in tmpl_list}
        assert "device_id" in var_names, "templating 缺少 device_id 变量"
        # device_id 变量应为 query 类型
        device_var = next(v for v in tmpl_list if v.get("name") == "device_id")
        assert device_var.get("type") == "query", "device_id 变量应为 query 类型"
        assert device_var.get("includeAll") is True, "device_id 变量应支持 All 选项"

    def test_time_range_default_1h(self, dashboard: dict) -> None:
        """time range 默认最近 1 小时。"""
        time = dashboard.get("time", {})
        assert time.get("from") == "now-1h", f"time.from 应为 now-1h，实际 {time.get('from')}"
        assert time.get("to") == "now", f"time.to 应为 now，实际 {time.get('to')}"

    def test_refresh_interval_10s(self, dashboard: dict) -> None:
        """refresh 间隔必须为 10s。"""
        assert dashboard.get("refresh") == "10s", (
            f"refresh 应为 10s，实际 {dashboard.get('refresh')}"
        )

    def test_new_panels_exist(self, dashboard: dict) -> None:
        """新增面板必须存在：模型版本不匹配 / 降级级别 / 网络连接。"""
        titles = {p["title"] for p in dashboard["panels"]}
        assert "模型版本不匹配告警" in titles, "缺少面板：模型版本不匹配告警"
        assert "降级级别状态" in titles, "缺少面板：降级级别状态"
        assert "网络连接状态" in titles, "缺少面板：网络连接状态"


# --------------------------------------------------------------------------- #
# 4. MetricsEndpoint 验证（5 个）
# --------------------------------------------------------------------------- #


class TestMetricsEndpoint:
    """edge/monitoring/metrics_endpoint.py MetricsEndpoint 验证。"""

    def test_init_defaults(self) -> None:
        """默认 host=0.0.0.0 / port=9000。"""
        ep = MetricsEndpoint(EdgeMetrics())
        assert ep.host == "0.0.0.0", f"默认 host 应为 0.0.0.0，实际 {ep.host}"
        assert ep.port == 9000, f"默认 port 应为 9000，实际 {ep.port}"

    def test_init_custom_host_port(self) -> None:
        """自定义 host/port 正确设置。"""
        ep = MetricsEndpoint(EdgeMetrics(), host="127.0.0.1", port=9091)
        assert ep.host == "127.0.0.1"
        assert ep.port == 9091

    def test_url(self, metrics_endpoint: MetricsEndpoint) -> None:
        """url() 返回正确的端点 URL。"""
        assert metrics_endpoint.url() == "http://0.0.0.0:9000"

    async def test_handle_metrics(self, metrics_endpoint: MetricsEndpoint) -> None:
        """handle_metrics 返回 Prometheus 文本格式。"""
        request = HttpRequest(method="GET", path="/metrics")
        body = await metrics_endpoint.handle_metrics(request)
        assert isinstance(body, str), "handle_metrics 应返回 str"
        assert "edge_inference_p95_ms" in body, "响应应包含 edge_inference_p95_ms 指标"
        assert "# HELP" in body, "响应应包含 HELP 注释"
        assert "# TYPE" in body, "响应应包含 TYPE 注释"

    async def test_handle_health(self, metrics_endpoint: MetricsEndpoint) -> None:
        """handle_health 返回 JSON 健康状态。"""
        request = HttpRequest(method="GET", path="/health")
        body = await metrics_endpoint.handle_health(request)
        assert isinstance(body, str), "handle_health 应返回 str"
        data = json.loads(body)
        assert data["status"] == "healthy", "健康状态应为 healthy"
        assert "metrics" in data, "健康响应应包含 metrics 字段"
        assert "inference_p95_ms" in data["metrics"], "metrics 应包含 inference_p95_ms"


# --------------------------------------------------------------------------- #
# 5. Prometheus 格式正确性（3 个）
# --------------------------------------------------------------------------- #


class TestPrometheusFormat:
    """Prometheus 文本格式与 Content-Type 正确性验证。"""

    def test_prometheus_content_type_value(self) -> None:
        """PROMETHEUS_CONTENT_TYPE 必须为标准值。"""
        assert PROMETHEUS_CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"

    def test_json_content_type_value(self) -> None:
        """JSON_CONTENT_TYPE 必须为标准值。"""
        assert JSON_CONTENT_TYPE == "application/json; charset=utf-8"

    async def test_metrics_text_format_complete(self, metrics_endpoint: MetricsEndpoint) -> None:
        """导出的 Prometheus 文本必须包含完整 HELP/TYPE/值行。"""
        request = HttpRequest(method="GET", path="/metrics")
        body = await metrics_endpoint.handle_metrics(request)
        lines = body.strip().split("\n")
        # 必须有 HELP 行
        help_lines = [ln for ln in lines if ln.startswith("# HELP")]
        assert help_lines, "缺少 # HELP 行"
        # 必须有 TYPE 行
        type_lines = [ln for ln in lines if ln.startswith("# TYPE")]
        assert type_lines, "缺少 # TYPE 行"
        # 必须有指标值行（非注释行）
        value_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
        assert value_lines, "缺少指标值行"
        # 每个指标名应同时有 HELP、TYPE 和值行
        for help_line in help_lines:
            metric_name = help_line.split()[2]
            has_type = any(ln.startswith(f"# TYPE {metric_name}") for ln in type_lines)
            assert has_type, f"指标 {metric_name} 缺少 TYPE 行"


# --------------------------------------------------------------------------- #
# 6. 配置文件一致性（2 个）
# --------------------------------------------------------------------------- #


class TestCrossFileConsistency:
    """跨文件一致性：端口 / 路径在多份配置中保持一致。"""

    def test_port_consistency(
        self, prometheus_config: dict, compose_yaml: dict
    ) -> None:
        """prometheus.yml 中 edge-agent 端口 9000 与 docker-compose 一致。"""
        # prometheus.yml edge-agent target
        jobs = {j["job_name"]: j for j in prometheus_config["scrape_configs"]}
        edge_targets = jobs["edge-agent"]["static_configs"][0]["targets"]
        assert any("9000" in t for t in edge_targets), "prometheus.yml edge-agent 应含端口 9000"
        # docker-compose edge-agent 端口映射
        edge_ports = compose_yaml["services"]["edge-agent"]["ports"]
        assert "9000:9000" in edge_ports, "docker-compose edge-agent 端口应为 9000:9000"
        # cloud-api 端口 8000 一致
        cloud_targets = jobs["cloud-api"]["static_configs"][0]["targets"]
        assert any("8000" in t for t in cloud_targets), "prometheus.yml cloud-api 应含端口 8000"
        cloud_ports = compose_yaml["services"]["cloud-api"]["ports"]
        assert "8000:8000" in cloud_ports, "docker-compose cloud-api 端口应为 8000:8000"

    def test_metrics_path_consistency(self, prometheus_config: dict) -> None:
        """所有 scrape_configs 的 metrics_path 必须为 /metrics。"""
        for job in prometheus_config["scrape_configs"]:
            path = job.get("metrics_path")
            assert path == "/metrics", (
                f"作业 {job['job_name']} metrics_path 应为 /metrics，实际 {path}"
            )