"""E3 部署配置验证测试。

验证范围（不实际构建 Docker 镜像，仅校验配置文件语法与内容）：
  - Dockerfile：非 root 用户 / HEALTHCHECK / 基础镜像
  - docker-compose.edge.yml：资源限制 / 服务依赖 / 环境变量 / healthcheck
  - systemd 服务：After/Before 顺序 / 资源限制 / 安全加固
  - Grafana 看板 JSON：面板 / 数据源 / 告警规则
  - 迁移检查清单：文件存在 / 内容完整
  - 跨文件一致性：端口 / 路径 / 环境变量
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE_FILE = ROOT / "docker-compose.edge.yml"
EDGE_SERVICE = ROOT / "deploy" / "edge-agent.service"
MGMT_SERVICE = ROOT / "deploy" / "mgmt-agent.service"
GRAFANA_DASHBOARD = ROOT / "deploy" / "grafana" / "dashboard.json"
MIGRATION_CHECKLIST = ROOT / "deploy" / "migration_checklist.md"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    """读取 Dockerfile 全文。"""
    assert DOCKERFILE.exists(), f"Dockerfile 不存在：{DOCKERFILE}"
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerfile_lines(dockerfile_text: str) -> list[str]:
    """Dockerfile 按行切分（去除空行与注释）。"""
    lines = []
    for raw in dockerfile_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


@pytest.fixture(scope="module")
def compose_yaml() -> dict:
    """解析 docker-compose.edge.yml。"""
    assert COMPOSE_FILE.exists(), f"docker-compose.edge.yml 不存在：{COMPOSE_FILE}"
    data = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "docker-compose.edge.yml 顶层不是 mapping"
    return data


@pytest.fixture(scope="module")
def edge_service_text() -> str:
    assert EDGE_SERVICE.exists(), f"edge-agent.service 不存在：{EDGE_SERVICE}"
    return EDGE_SERVICE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mgmt_service_text() -> str:
    assert MGMT_SERVICE.exists(), f"mgmt-agent.service 不存在：{MGMT_SERVICE}"
    return MGMT_SERVICE.read_text(encoding="utf-8")


def parse_systemd(text: str) -> dict[str, dict[str, str]]:
    """简易 systemd ini 解析（支持重复键收集到列表）。"""
    sections: dict[str, dict[str, list[str]]] = {}
    current = "DEFAULT"
    sections[current] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        m = re.match(r"\[(.+)]$", line)
        if m:
            current = m.group(1)
            sections.setdefault(current, {})
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        sections[current].setdefault(key, []).append(value)
    # 单值化：若只有一份则返回 str，多份返回 list
    flat: dict[str, dict[str, str | list[str]]] = {}
    for sec, kvs in sections.items():
        flat[sec] = {k: (v[0] if len(v) == 1 else v) for k, v in kvs.items()}
    return flat


@pytest.fixture(scope="module")
def edge_service(edge_service_text: str) -> dict:
    return parse_systemd(edge_service_text)


@pytest.fixture(scope="module")
def mgmt_service(mgmt_service_text: str) -> dict:
    return parse_systemd(mgmt_service_text)


@pytest.fixture(scope="module")
def grafana_dashboard() -> dict:
    assert GRAFANA_DASHBOARD.exists(), f"dashboard.json 不存在：{GRAFANA_DASHBOARD}"
    return json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def migration_checklist() -> str:
    assert MIGRATION_CHECKLIST.exists(), f"migration_checklist.md 不存在：{MIGRATION_CHECKLIST}"
    return MIGRATION_CHECKLIST.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. Dockerfile 验证（3 个）
# --------------------------------------------------------------------------- #


class TestDockerfile:
    """Dockerfile 语法与关键内容验证。"""

    def test_base_image_is_python_312_slim(self, dockerfile_lines: list[str]) -> None:
        """基础镜像必须为 python:3.12-slim。"""
        from_lines = [ln for ln in dockerfile_lines if ln.upper().startswith("FROM ")]
        assert from_lines, "Dockerfile 缺少 FROM 指令"
        # 取第一个 FROM（base 阶段）
        first_from = from_lines[0]
        assert "python:3.12-slim" in first_from, (
            f"基础镜像应为 python:3.12-slim，实际：{first_from}"
        )

    def test_non_root_user(self, dockerfile_lines: list[str]) -> None:
        """必须创建非 root 用户并以该用户运行。"""
        # 1) useradd 创建 edge-agent 用户
        useradd_lines = [ln for ln in dockerfile_lines if "useradd" in ln and "edge-agent" in ln]
        assert useradd_lines, "Dockerfile 未创建非 root 用户 edge-agent"
        # 2) USER 指令切换到非 root
        user_lines = [ln for ln in dockerfile_lines if ln.upper().startswith("USER ")]
        assert user_lines, "Dockerfile 缺少 USER 指令"
        user_target = user_lines[-1].split(None, 1)[1]
        assert user_target != "root", "USER 指令仍为 root"
        assert "edge-agent" in user_target, f"USER 应为 edge-agent，实际：{user_target}"

    def test_healthcheck_present(self, dockerfile_text: str) -> None:
        """必须存在 HEALTHCHECK 指令并指向 /health 端点。

        Dockerfile 中 HEALTHCHECK 可能使用行继续符 ``\\`` 跨多行，因此用全文搜索
        而非按行过滤。
        """
        # 用全文匹配 HEALTHCHECK ... CMD ... /health（容忍行继续符与换行）
        hc_match = re.search(r"HEALTHCHECK\b.*?CMD\b.*?/health", dockerfile_text, re.DOTALL)
        assert hc_match, "Dockerfile 缺少指向 /health 的 HEALTHCHECK 指令"
        assert "curl" in hc_match.group(0), "HEALTHCHECK 应使用 curl 探测"


# --------------------------------------------------------------------------- #
# 2. docker-compose.edge.yml 验证（4 个）
# --------------------------------------------------------------------------- #


class TestDockerCompose:
    """docker-compose.edge.yml 资源限制 / 依赖 / 环境变量 / healthcheck 验证。"""

    def test_edge_agent_resource_limits(self, compose_yaml: dict) -> None:
        """edge-agent 必须有 2C4G 资源限制 + read_only + cap_drop ALL。"""
        svc = compose_yaml["services"]["edge-agent"]
        assert float(svc["cpus"]) == 2.0, f"cpus 应为 2.0，实际：{svc['cpus']}"
        assert svc["mem_limit"] == "4g", f"mem_limit 应为 4g，实际：{svc['mem_limit']}"
        assert svc.get("memswap_limit") == "4g", "memswap_limit 应为 4g"
        assert svc.get("pids_limit") == 256, "pids_limit 应为 256"
        assert svc.get("read_only") is True, "read_only 必须为 true"
        assert "ALL" in svc.get("cap_drop", []), "cap_drop 必须包含 ALL"

    def test_service_dependencies(self, compose_yaml: dict) -> None:
        """edge-agent 必须依赖 mosquitto 启动。"""
        svc = compose_yaml["services"]["edge-agent"]
        deps = svc.get("depends_on", {})
        assert "mosquitto" in deps, "edge-agent 必须依赖 mosquitto"
        # 检查 condition 形式
        cond = deps["mosquitto"]
        if isinstance(cond, dict):
            assert cond.get("condition") == "service_started", (
                "mosquitto 依赖条件应为 service_started"
            )

    def test_edge_agent_environment(self, compose_yaml: dict) -> None:
        """edge-agent 关键环境变量必须设置。"""
        env_list = compose_yaml["services"]["edge-agent"]["environment"]
        env = {}
        for item in env_list:
            key, _, value = item.partition("=")
            env[key] = value
        required = {
            "GW_ID": "gw-sim-001",
            "EDGE_API_PORT": "9000",
            "EDGE_LLM_CTX": "2048",
            "EDGE_LLM_MAXTOK": "32",
        }
        for key, expected in required.items():
            assert env.get(key) == expected, (
                f"环境变量 {key} 应为 {expected}，实际：{env.get(key)}"
            )
        assert env.get("EDGE_LLM_MODEL"), "EDGE_LLM_MODEL 不能为空"
        assert "mosquitto" in env.get("CLOUD_MQTT_HOST", ""), (
            "CLOUD_MQTT_HOST 应指向 mosquitto"
        )

    def test_healthcheck_config(self, compose_yaml: dict) -> None:
        """edge-agent 与 cloud-api 都必须有 healthcheck。"""
        for name in ("edge-agent", "cloud-api", "mgmt-agent"):
            svc = compose_yaml["services"][name]
            hc = svc.get("healthcheck")
            assert hc, f"{name} 缺少 healthcheck"
            test_cmd = hc.get("test", [])
            assert any("/health" in str(t) for t in test_cmd), (
                f"{name} healthcheck 应探测 /health"
            )
            assert hc.get("interval") == "30s", f"{name} healthcheck interval 应为 30s"
            assert hc.get("retries") == 3, f"{name} healthcheck retries 应为 3"


# --------------------------------------------------------------------------- #
# 3. systemd 服务文件验证（3 个）
# --------------------------------------------------------------------------- #


class TestSystemdServices:
    """systemd 服务文件启动顺序 / 资源限制 / 安全加固验证。"""

    def test_startup_ordering(
        self, edge_service: dict, mgmt_service: dict
    ) -> None:
        """管理代理必须 Before= 主应用；主应用必须 After= 管理代理。"""
        mgmt_unit = mgmt_service["Unit"]
        edge_unit = edge_service["Unit"]
        # mgmt-agent Before=edge-agent.service
        before = mgmt_unit.get("Before", "")
        assert "edge-agent.service" in before, (
            "mgmt-agent.service 必须声明 Before=edge-agent.service"
        )
        # edge-agent After=mgmt-agent.service
        after = edge_unit.get("After", "")
        assert "mgmt-agent.service" in after, (
            "edge-agent.service 必须声明 After=mgmt-agent.service"
        )
        # 两者都依赖网络
        assert "network-online.target" in after, "edge-agent 必须依赖 network-online.target"
        assert "network-online.target" in mgmt_unit.get("After", ""), (
            "mgmt-agent 必须依赖 network-online.target"
        )

    def test_resource_limits(self, edge_service: dict, mgmt_service: dict) -> None:
        """资源限制必须与 docker-compose 一致。"""
        edge_svc = edge_service["Service"]
        mgmt_svc = mgmt_service["Service"]
        # 主应用：2C / 4G / 256 tasks
        assert edge_svc["CPUQuota"] == "200%", "edge-agent CPUQuota 应为 200%（2 核）"
        assert edge_svc["MemoryMax"] == "4G", "edge-agent MemoryMax 应为 4G"
        assert edge_svc["TasksMax"] == "256", "edge-agent TasksMax 应为 256"
        # 管理代理：0.5C / 512M / 64 tasks（更轻量）
        assert mgmt_svc["CPUQuota"] == "50%", "mgmt-agent CPUQuota 应为 50%"
        assert mgmt_svc["MemoryMax"] == "512M", "mgmt-agent MemoryMax 应为 512M"
        assert mgmt_svc["TasksMax"] == "64", "mgmt-agent TasksMax 应为 64"

    def test_security_hardening(self, edge_service: dict, mgmt_service: dict) -> None:
        """安全加固选项必须存在。"""
        required = {
            "NoNewPrivileges": "true",
            "ProtectSystem": "strict",
            "ProtectHome": "true",
            "PrivateTmp": "true",
        }
        for name, svc in (("edge-agent", edge_service), ("mgmt-agent", mgmt_service)):
            s = svc["Service"]
            for key, expected in required.items():
                assert s.get(key) == expected, f"{name} {key} 应为 {expected}"
            # ReadWritePaths 必须包含队列与模型目录
            rwp = s.get("ReadWritePaths", "")
            assert "/var/lib/edge-agent" in rwp, f"{name} ReadWritePaths 必须含队列目录"
            assert "/opt/edge-agent/models" in rwp, f"{name} ReadWritePaths 必须含模型目录"
            # 重启策略
            assert "Restart" in s, f"{name} 必须配置 Restart 策略"


# --------------------------------------------------------------------------- #
# 4. Grafana 看板 JSON 验证（3 个）
# --------------------------------------------------------------------------- #


class TestGrafanaDashboard:
    """Grafana dashboard.json 面板 / 数据源 / 告警规则验证。"""

    def test_required_panels_exist(self, grafana_dashboard: dict) -> None:
        """必须包含方案要求的 8 类面板。"""
        panels = grafana_dashboard["panels"]
        titles = {p["title"] for p in panels}
        required_keywords = [
            "推理 P95 延迟",  # gauge
            "模型版本",  # stat
            "队列深度",  # timeseries
            "离线时长",  # stat
            "缓存命中率",  # gauge
            "云端升级率趋势",  # timeseries
            "CPU 使用率",  # timeseries
            "内存使用率",  # timeseries
            "CPU 温度",  # gauge
        ]
        for kw in required_keywords:
            assert any(kw in t for t in titles), f"看板缺少面板：{kw}"
        # 类型分布校验
        by_type = {}
        for p in panels:
            by_type.setdefault(p["type"], []).append(p)
        assert "gauge" in by_type, "看板必须包含 gauge 面板"
        assert "stat" in by_type, "看板必须包含 stat 面板"
        assert "timeseries" in by_type, "看板必须包含 timeseries 面板"

    def test_datasource_configuration(self, grafana_dashboard: dict) -> None:
        """看板必须配置 Prometheus 数据源变量。"""
        templating = grafana_dashboard.get("templating", {}).get("list", [])
        ds_vars = [v for v in templating if v.get("type") == "datasource"]
        assert ds_vars, "看板必须定义 datasource 类型变量"
        prom_var = ds_vars[0]
        assert prom_var.get("query") == "prometheus", "数据源变量必须为 prometheus"
        # 每个面板的 datasource 必须引用该变量
        ds_uid = f"${{{prom_var['name']}}}"
        for p in grafana_dashboard["panels"]:
            ds = p.get("datasource", {})
            uid = ds.get("uid", "")
            assert uid == ds_uid, (
                f"面板 {p['title']} 数据源 uid 应为 {ds_uid}，实际：{uid}"
            )

    def test_alert_rules_present(self, grafana_dashboard: dict) -> None:
        """看板必须包含告警规则。"""
        group = grafana_dashboard.get("alertRuleGroup", {})
        rules = group.get("rules", [])
        assert rules, "看板必须定义至少一条告警规则"
        rule_names = {r["name"] for r in rules}
        # 至少覆盖：推理延迟 / 队列积压 / 离线超时 / 温度 / 内存
        expected_keywords = ["推理", "队列", "离线", "温度", "内存"]
        for kw in expected_keywords:
            assert any(kw in n for n in rule_names), f"告警规则缺少覆盖：{kw}"
        # 每条规则必须有 for 持续时间与 message
        for r in rules:
            assert r.get("for"), f"告警规则 {r['name']} 缺少 for 持续时间"
            assert r.get("message"), f"告警规则 {r['name']} 缺少 message"


# --------------------------------------------------------------------------- #
# 5. 迁移检查清单验证（2 个）
# --------------------------------------------------------------------------- #


class TestMigrationChecklist:
    """迁移检查清单文件存在性与内容完整性验证。"""

    def test_checklist_sections_complete(self, migration_checklist: str) -> None:
        """检查清单必须包含三大核心章节。"""
        required_sections = [
            "零改码迁移验证",
            "配置差异",
            "真机重测指标",
        ]
        for sec in required_sections:
            assert sec in migration_checklist, f"检查清单缺少章节：{sec}"

    def test_checklist_key_items(self, migration_checklist: str) -> None:
        """检查清单必须覆盖关键迁移项。"""
        required_items = [
            "Docker 镜像",
            "docker-compose.edge.yml",
            "systemd",
            "管理代理先于主应用启动",
            "健康检查端点",
            "MQTT 连接",
            "模型加载",
            "GW_ID",
            "CLOUD_MQTT",
            "CLOUD_API",
            "EDGE_LLM_MODEL",
            "告警路径",
            "判别路径",
            "LLM 初判",
            "断网自治",
            "OTA 成功率",
        ]
        for item in required_items:
            assert item in migration_checklist, f"检查清单缺少关键项：{item}"


# --------------------------------------------------------------------------- #
# 6. 部署配置一致性验证（3 个）
# --------------------------------------------------------------------------- #


class TestCrossFileConsistency:
    """跨文件一致性：端口 / 路径 / 环境变量在多份配置中保持一致。"""

    def test_port_consistency(
        self, compose_yaml: dict, edge_service: dict, mgmt_service: dict
    ) -> None:
        """端口在 docker-compose 与 systemd 间一致。"""
        # edge-agent: 9000
        edge_ports = compose_yaml["services"]["edge-agent"]["ports"]
        assert "9000:9000" in edge_ports, "edge-agent 端口映射应为 9000:9000"
        # mgmt-agent: 9001
        mgmt_ports = compose_yaml["services"]["mgmt-agent"]["ports"]
        assert "9001:9001" in mgmt_ports, "mgmt-agent 端口映射应为 9001:9001"
        # cloud-api: 8000
        cloud_ports = compose_yaml["services"]["cloud-api"]["ports"]
        assert "8000:8000" in cloud_ports, "cloud-api 端口映射应为 8000:8000"
        # docker-compose mgmt-agent command 必须包含 9001（uvicorn --port 9001）
        mgmt_cmd = compose_yaml["services"]["mgmt-agent"]["command"]
        assert "9001" in mgmt_cmd, "docker-compose mgmt-agent command 应包含 9001 端口"
        # systemd 通过 Environment=MGMT_API_PORT=9001 传入端口
        mgmt_env = mgmt_service["Service"].get("Environment", "")
        if isinstance(mgmt_env, list):
            mgmt_env = " ".join(mgmt_env)
        assert "MGMT_API_PORT=9001" in mgmt_env, (
            "mgmt-agent.service 必须通过 Environment 设置 MGMT_API_PORT=9001"
        )
        # edge-agent systemd 通过 Environment=EDGE_API_PORT=9000
        edge_env = edge_service["Service"].get("Environment", "")
        if isinstance(edge_env, list):
            edge_env = " ".join(edge_env)
        assert "EDGE_API_PORT=9000" in edge_env, (
            "edge-agent.service 必须通过 Environment 设置 EDGE_API_PORT=9000"
        )

    def test_path_consistency(
        self, compose_yaml: dict, edge_service: dict, mgmt_service: dict
    ) -> None:
        """关键路径在 docker-compose 与 systemd 间一致。"""
        # docker-compose 挂载 /opt/edge-agent/models 与 /var/lib/edge-agent
        edge_vols = compose_yaml["services"]["edge-agent"]["volumes"]
        assert any("/opt/edge-agent/models" in v for v in edge_vols), (
            "docker-compose 必须挂载 /opt/edge-agent/models"
        )
        assert any("/var/lib/edge-agent" in v for v in edge_vols), (
            "docker-compose 必须挂载 /var/lib/edge-agent"
        )
        # systemd ReadWritePaths 同样包含这两个路径
        for name, svc in (("edge-agent", edge_service), ("mgmt-agent", mgmt_service)):
            rwp = svc["Service"]["ReadWritePaths"]
            assert "/opt/edge-agent/models" in rwp, f"{name} ReadWritePaths 缺少模型目录"
            assert "/var/lib/edge-agent" in rwp, f"{name} ReadWritePaths 缺少队列目录"

    def test_env_consistency(
        self, compose_yaml: dict, dockerfile_text: str
    ) -> None:
        """环境变量在 compose 中声明且 Dockerfile 暴露对应端口。"""
        edge_env = {}
        for item in compose_yaml["services"]["edge-agent"]["environment"]:
            k, _, v = item.partition("=")
            edge_env[k] = v
        # EXPOSE 9000 与 EDGE_API_PORT=9000 一致
        expose_lines = [
            ln for ln in dockerfile_text.splitlines() if ln.strip().upper().startswith("EXPOSE")
        ]
        assert expose_lines, "Dockerfile 必须有 EXPOSE 指令"
        exposed_ports = []
        for ln in expose_lines:
            exposed_ports.extend(ln.split()[1:])
        assert "9000" in exposed_ports, "Dockerfile 应 EXPOSE 9000"
        assert edge_env["EDGE_API_PORT"] == "9000", "EDGE_API_PORT 应为 9000"
        # GW_ID 非空
        assert edge_env["GW_ID"], "GW_ID 不能为空"
