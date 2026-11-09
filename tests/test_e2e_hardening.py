"""E2E 硬化集成测试：安全 + 监控 + 降级 + 部署配置全链路验证。

本测试是 E3 硬化的端到端集成测试，覆盖以下场景：
1. 安全链路集成：mTLS + 模型签名 + 脱敏 串联
2. 监控链路集成：指标 + 探针 + Ring buffer 串联
3. 降级链路集成：DegradeManager 矩阵 + 与监控/安全联动
4. 部署配置集成：Dockerfile/compose/systemd/grafana/checklist 跨文件一致性
5. 全链路集成：安全 → 监控 → 降级 → 部署 一站式硬化场景

设计原则：
- 使用 tmp_path 隔离文件 IO，monkeypatch 隔离系统资源
- 不依赖外部服务（Docker/Prometheus/Grafana），全部用 mock/sim
- 安全测试用 cryptography 生成真实证书/密钥
- 降级测试用 mock GuardState 而非真实系统资源
- 部署配置测试只验证文件内容，不实际构建/运行
- 使用 datetime.now(UTC) 而非 datetime.now()
"""

from __future__ import annotations

import json
import re
import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from edge.cloud.masking import (
    SaltRotator,
    hash_device_id,
    mask_payload,
    mask_process_params,
)
from edge.guard import GuardState
from edge.guard_degrade import (
    DegradeLevel,
    DegradeManager,
    DegradePolicy,
    DegradeThresholds,
)
from edge.monitoring.metrics import EdgeMetrics
from edge.monitoring.probes import CRITICAL_PROBE_NAMES, LocalProbes
from edge.monitoring.ring_buffer import RingBufferLog
from edge.security import mtls
from edge.security.model_signing import (
    ModelSignature,
    compute_sha256,
    generate_rsa_key_pair,
    sign_model,
    verify_model,
    verify_model_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE_FILE = ROOT / "docker-compose.edge.yml"
EDGE_SERVICE = ROOT / "deploy" / "edge-agent.service"
MGMT_SERVICE = ROOT / "deploy" / "mgmt-agent.service"
GRAFANA_DASHBOARD = ROOT / "deploy" / "grafana" / "dashboard.json"
MIGRATION_CHECKLIST = ROOT / "deploy" / "migration_checklist.md"


# ===========================================================================
# 辅助函数与 fixtures
# ===========================================================================


def _make_state(
    mem: int = 2000,
    disk: int = 10000,
    temp: float | None = None,
    offline: int = 0,
    net: bool = True,
    cpu_percent: float = 0.0,
) -> GuardState:
    """构建 GuardState，用于 DegradePolicy/DegradeManager 纯逻辑测试。"""
    return GuardState(
        level="normal",
        mem_used_mb=mem,
        disk_free_mb=disk,
        cpu_temp_c=temp,
        llm_allowed=True,
        t2_full=True,
        accept_models=True,
        cpu_percent=cpu_percent,
        net_connected=net,
        offline_duration_s=offline,
        degrade_actions=[],
    )


def _default_thresholds() -> DegradeThresholds:
    """默认降级阈值（与 docker-compose.edge.yml 的 2C4G 对齐）。"""
    return DegradeThresholds(
        mem_normal_mb=3000,
        mem_llm_small_mb=3000,
        mem_llm_off_mb=3600,
        temp_throttle_c=80.0,
        disk_critical_mb=500,
        offline_degraded_hours=24,
    )


def _parse_systemd(text: str) -> dict[str, dict[str, str | list[str]]]:
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
    flat: dict[str, dict[str, str | list[str]]] = {}
    for sec, kvs in sections.items():
        flat[sec] = {k: (v[0] if len(v) == 1 else v) for k, v in kvs.items()}
    return flat


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_yaml() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def edge_service_text() -> str:
    return EDGE_SERVICE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mgmt_service_text() -> str:
    return MGMT_SERVICE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def grafana_dashboard() -> dict:
    return json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def migration_checklist() -> str:
    return MIGRATION_CHECKLIST.read_text(encoding="utf-8")


# ===========================================================================
# 1. 安全链路集成（5 个测试）
# ===========================================================================


class TestSecurityChainIntegration:
    """安全链路集成：mTLS + 模型签名 + 脱敏 全链路。"""

    def test_mtls_full_chain(self, tmp_path: Path) -> None:
        """mTLS 全链路：CA → 设备证书 → SSLContext → 验证 → 指纹。"""
        # 1) 生成 CA
        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestRootCA", days_valid=365)
        assert b"BEGIN CERTIFICATE" in ca_cert_pem

        # 2) 用 CA 签发设备证书
        device_id = "device-001"
        dev_cert_pem, dev_key_pem = mtls.generate_device_cert(
            device_id, ca_cert_pem, ca_key_pem, days_valid=30
        )
        assert b"BEGIN CERTIFICATE" in dev_cert_pem

        # 3) 加载为 SSLContext（mTLS 双向认证）
        ctx = mtls.load_cert_chain(dev_cert_pem, dev_key_pem)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED

        # 4) 验证证书由 CA 签发
        assert mtls.verify_cert(dev_cert_pem, ca_cert_pem) is True

        # 5) 计算指纹（SHA256 hex）
        fp = mtls.fingerprint(dev_cert_pem)
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA256 hex 长度
        int(fp, 16)  # 必须是合法十六进制

        # 6) 自签名 CA 验证自身
        assert mtls.verify_cert(ca_cert_pem, ca_cert_pem) is True

    def test_model_signing_full_chain(self, tmp_path: Path) -> None:
        """模型签名全链路：RSA 密钥对 → 签名 → 验证 → 篡改检测 → 完整性。"""
        # 1) 生成 RSA 密钥对
        priv_pem, pub_pem = generate_rsa_key_pair(key_size=2048)
        assert b"BEGIN RSA PRIVATE KEY" in priv_pem or b"BEGIN PRIVATE KEY" in priv_pem
        assert b"BEGIN PUBLIC KEY" in pub_pem

        # 2) 准备模型文件
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"\x42" * 1024)

        # 3) 计算哈希 + 签名
        original_hash = compute_sha256(model_path)
        assert len(original_hash) == 64

        signature = sign_model(model_path, priv_pem)
        assert isinstance(signature, str)

        # 4) 验证签名通过
        assert verify_model(model_path, signature, pub_pem) is True

        # 5) 完整性校验通过
        assert verify_model_integrity(model_path, original_hash) is True

        # 6) 篡改检测：修改模型文件后签名验证失败
        with model_path.open("ab") as f:
            f.write(b"\x00" * 16)
        assert verify_model(model_path, signature, pub_pem) is False
        assert verify_model_integrity(model_path, original_hash) is False

        # 7) ModelSignature 元数据可用
        sig_meta = ModelSignature(
            model_hash=original_hash,
            signature=signature,
            signed_at=datetime.now(UTC).isoformat(),
            signed_by="test-signer",
        )
        assert sig_meta.signed_by == "test-signer"
        assert sig_meta.model_hash == original_hash

    def test_masking_full_chain(self) -> None:
        """脱敏全链路：SaltRotator → 哈希 → 轮换 → 历史验证 → 综合脱敏。"""
        # 1) SaltRotator 初始化
        rotator = SaltRotator("salt-v1", rotation_interval_days=30)
        assert rotator.current_salt == "salt-v1"

        # 2) 哈希 device_id
        device_id = "device-007"

        hashed_v1 = hash_device_id(device_id, rotator.current_salt)
        assert len(hashed_v1) == 64

        # 3) 轮换 salt
        rotator.rotate("salt-v2")
        assert rotator.current_salt == "salt-v2"
        hashed_v2 = hash_device_id(device_id, rotator.current_salt)
        assert hashed_v1 != hashed_v2  # 不同 salt 产生不同哈希

        # 4) 验证历史哈希（用旧时间戳应匹配旧 salt）
        # 使用当前时间（v2 生效后）验证 v1 时期的哈希需要早于 v2 生效时间
        # 此处验证当前 salt 下的哈希
        current_hashed = hash_device_id(device_id, rotator.current_salt)
        assert rotator.verify(device_id, current_hashed, datetime.now(UTC)) is True
        # 错误哈希验证失败
        assert rotator.verify(device_id, "0" * 64, datetime.now(UTC)) is False

        # 5) mask_process_params 区间化
        assert mask_process_params(35.0, [(0, 40), (40, 60)]) == "[0,40)"
        assert mask_process_params(50.0, [(0, 40), (40, 60)]) == "[40,60)"
        assert mask_process_params(100.0, [(0, 40), (40, 60)]) == ">=60"

        # 6) mask_payload 综合脱敏
        payload = {
            "device_id": "device-007",
            "site": "confidential-site",
            "temperature": 42.5,
            "customer": "secret-customer",
            "trace_id": "abc-123",
        }
        masked = mask_payload(
            payload,
            salt=rotator.current_salt,
            fields_to_hash=["device_id"],
            fields_to_strip=["site", "customer"],
        )
        assert "site" not in masked  # 已剥离
        assert "customer" not in masked
        assert masked["device_id"] == hashed_v2  # 已哈希
        assert masked["temperature"] == 42.5  # 保留
        assert masked["trace_id"] == "abc-123"

    def test_security_modules_interoperation(self, tmp_path: Path) -> None:
        """安全模块互操作：mTLS 指纹 + 模型签名 + 脱敏 在同一测试中串联。

        模拟场景：设备启动时同时进行证书签发、模型签名和脱敏配置，
        三者共用同一 device_id 作为关联键。
        """
        device_id = "edge-gw-001"

        # 1) mTLS：CA 签发设备证书，计算指纹
        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("InteropCA")
        dev_cert_pem, dev_key_pem = mtls.generate_device_cert(
            device_id, ca_cert_pem, ca_key_pem
        )
        cert_fp = mtls.fingerprint(dev_cert_pem)
        assert mtls.verify_cert(dev_cert_pem, ca_cert_pem) is True

        # 2) 模型签名：用同一 device_id 作为签名者
        priv_pem, pub_pem = generate_rsa_key_pair()
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"model-data")
        signature = sign_model(model_path, priv_pem)
        assert verify_model(model_path, signature, pub_pem) is True

        # 3) 脱敏：用证书指纹作为 salt 关联设备身份
        rotator = SaltRotator(cert_fp[:32])  # 取指纹前 32 字符作为 salt
        hashed_device = hash_device_id(device_id, rotator.current_salt)
        assert len(hashed_device) == 64

        # 4) 三者关联：同一设备证书指纹 + 模型签名 + 脱敏哈希
        sig_meta = ModelSignature(
            model_hash=compute_sha256(model_path),
            signature=signature,
            signed_at=datetime.now(UTC).isoformat(),
            signed_by=device_id,
            extra={"cert_fp": cert_fp, "masked_device": hashed_device},
        )
        assert sig_meta.signed_by == device_id
        assert sig_meta.extra["cert_fp"] == cert_fp
        # 验证关联一致性：脱敏 salt 来自证书指纹
        assert rotator.current_salt == cert_fp[:32]

    def test_security_chain_tamper_detection(self, tmp_path: Path) -> None:
        """安全链路篡改检测：证书篡改 + 模型篡改 + salt 轮换后旧哈希失效。"""
        # 1) 证书篡改：用不同 CA 签发的证书验证失败
        ca1_cert, ca1_key = mtls.generate_ca_cert("CA1")
        ca2_cert, ca2_key = mtls.generate_ca_cert("CA2")
        dev_cert, _ = mtls.generate_device_cert("dev", ca1_cert, ca1_key)
        assert mtls.verify_cert(dev_cert, ca1_cert) is True
        assert mtls.verify_cert(dev_cert, ca2_cert) is False  # 不同 CA

        # 2) 模型篡改：替换模型文件后签名失效
        priv, pub = generate_rsa_key_pair()
        model = tmp_path / "m.bin"
        model.write_bytes(b"original")
        sig = sign_model(model, priv)
        model.write_bytes(b"tampered")  # 完全替换
        assert verify_model(model, sig, pub) is False

        # 3) salt 轮换后，用新 salt 验证旧哈希应失败
        rotator = SaltRotator("old-salt")
        old_hashed = hash_device_id("dev", "old-salt")
        rotator.rotate("new-salt")
        # 用当前时间（new-salt 生效后）验证 old-salt 时期的哈希
        # get_salt(now) 返回 new-salt，重新计算不匹配 old_hashed
        assert rotator.verify("dev", old_hashed, datetime.now(UTC)) is False


# ===========================================================================
# 2. 监控链路集成（5 个测试）
# ===========================================================================


class TestMonitoringChainIntegration:
    """监控链路集成：指标 + 探针 + Ring buffer 全链路。"""

    def test_metrics_collection_full_chain(self) -> None:
        """指标收集全链路：推理 → 队列 → 缓存 → 升级 → Prometheus 导出。"""
        m = EdgeMetrics(p95_window=100)

        # 1) 记录多次推理延迟
        for i in range(100):
            m.record_inference(float(i + 1))
        assert m.inference_p95_ms == 95.0  # sorted[int(0.95*99)] = sorted[94] = 95

        # 2) 设置模型版本与队列深度
        m.set_model_version("v1.2.3")
        m.set_queue_depth(42)
        m.set_offline_duration(1800)

        # 3) 记录缓存命中/未命中
        for _ in range(8):
            m.record_cache_hit(True)
        for _ in range(2):
            m.record_cache_hit(False)
        assert m.local_cache_hit_rate == pytest.approx(0.8)

        # 4) 记录云端升级
        for _ in range(3):
            m.record_cloud_escalation(True)
        for _ in range(7):
            m.record_cloud_escalation(False)
        assert m.cloud_escalation_rate == pytest.approx(0.3)

        # 5) Prometheus 格式导出
        prom = m.export_prometheus()
        assert "edge_inference_p95_ms 95.0" in prom
        assert 'edge_model_version_info{version="v1.2.3"} 1' in prom
        assert "edge_queue_depth 42" in prom
        assert "edge_offline_duration_s 1800" in prom
        assert "edge_local_cache_hit_rate 0.8" in prom
        assert "edge_cloud_escalation_rate 0.3" in prom

        # 6) 快照一致性
        snap = m.snapshot()
        assert snap["inference_p95_ms"] == 95.0
        assert snap["model_version"] == "v1.2.3"
        assert snap["queue_depth"] == 42
        assert snap["cache_hits"] == 8
        assert snap["cache_total"] == 10

    def test_probes_full_chain(self, tmp_path: Path) -> None:
        """探针全链路：run_all() → 失败检测 → critical_failures。"""
        m = EdgeMetrics()
        m.set_model_version("v1.0")
        m.set_queue_depth(50)
        m.record_inference(200.0)  # 正常延迟

        probes = LocalProbes()
        results = probes.run_all(
            m,
            disk_path=tmp_path,
            temp_c=45.0,
            mqtt_connected=True,
        )

        # 1) 所有探针都应通过（正常状态）
        assert len(results) == 6  # inference + queue + model + disk + temp + mqtt
        assert probes.has_failures(results) is False
        assert probes.critical_failures(results) == []

        # 2) 异常状态：高延迟 + 队列积压 + 模型未加载 + MQTT 断连
        m2 = EdgeMetrics()
        m2.set_model_version("unknown")  # 模型未加载
        m2.set_queue_depth(20000)  # 队列积压
        m2.record_inference(5000.0)  # P95 超阈值

        results2 = probes.run_all(
            m2,
            disk_path=tmp_path,
            temp_c=95.0,  # 高温
            mqtt_connected=False,  # MQTT 断连
        )
        assert probes.has_failures(results2) is True

        # 3) critical_failures 应包含推理/模型/MQTT
        critical = probes.critical_failures(results2)
        critical_names = {r.name for r in critical}
        assert "inference_health" in critical_names
        assert "model_loaded" in critical_names
        assert "mqtt_connected" in critical_names
        # 温度/队列/磁盘不在 critical 集合
        assert frozenset(
            {"inference_health", "model_loaded", "mqtt_connected"}
        ) == CRITICAL_PROBE_NAMES

    def test_ring_buffer_full_chain(self) -> None:
        """Ring buffer 全链路：追加 → 查询 → 过期清理 → 批量导出 → 标记。"""
        buf = RingBufferLog(max_entries=1000, retention_hours=24)

        # 1) 追加多条日志
        for i in range(10):
            buf.append(
                level="INFO" if i % 2 == 0 else "WARN",
                message=f"event-{i}",
                metadata={"trace_id": f"t-{i}"},
            )
        assert len(buf) == 10

        # 2) 查询全部
        all_logs = buf.query(limit=100)
        assert len(all_logs) == 10
        assert all_logs[0].message == "event-0"
        assert all_logs[-1].message == "event-9"

        # 3) 按级别查询
        warn_logs = buf.query(level="WARN", limit=100)
        assert len(warn_logs) == 5
        assert all(r.level == "WARN" for r in warn_logs)

        # 4) 按 since 查询
        cutoff = datetime.now(UTC) - timedelta(seconds=1)
        recent = buf.query(since=cutoff, limit=100)
        assert len(recent) == 10  # 全部都是刚刚追加的

        # 5) 过期清理（无过期）
        removed = buf.purge_expired()
        assert removed == 0
        assert len(buf) == 10

        # 6) 批量导出 + 标记已导出
        since = datetime.now(UTC) - timedelta(minutes=1)
        batch = buf.export_batch(since=since)
        assert len(batch) == 10
        assert all(not e.exported for e in batch)
        buf.mark_exported(batch)
        # 再次导出应为空
        assert buf.export_batch(since=since) == []

        # 7) 统计
        stats = buf.stats()
        assert stats["total"] == 10
        assert stats["exported"] == 10
        assert stats["unexported"] == 0
        assert stats["by_level"]["INFO"] == 5
        assert stats["by_level"]["WARN"] == 5

    def test_ring_buffer_eviction_and_purge(self) -> None:
        """Ring buffer 容量驱逐与过期清理。"""
        # 1) 容量驱逐：max_entries=5，追加 10 条只保留最后 5 条
        buf = RingBufferLog(max_entries=5, retention_hours=24)
        for i in range(10):
            buf.append("INFO", f"msg-{i}")
        assert len(buf) == 5
        remaining = buf.query(limit=100)
        assert remaining[0].message == "msg-5"
        assert remaining[-1].message == "msg-9"

        # 2) 过期清理：retention_hours=1，追加后手动篡改时间戳使其过期
        buf2 = RingBufferLog(max_entries=100, retention_hours=1)
        buf2.append("INFO", "recent")
        # 手动将最早条目时间戳改为 2 小时前
        with buf2._lock:
            buf2._buffer[0].timestamp = datetime.now(UTC) - timedelta(hours=2)
        removed = buf2.purge_expired()
        assert removed == 1
        assert len(buf2) == 0

    def test_monitoring_modules_interoperation(self, tmp_path: Path) -> None:
        """监控模块互操作：指标 → 探针 → 日志 串联。

        模拟场景：指标收集到异常状态 → 探针检测到失败 → 写入告警日志。
        """
        # 1) 指标：记录异常推理延迟
        m = EdgeMetrics()
        m.set_model_version("v2.0")
        m.set_queue_depth(15000)  # 队列积压
        for _ in range(20):
            m.record_inference(4000.0)  # P95 远超阈值

        # 2) 探针：基于指标运行
        probes = LocalProbes()
        results = probes.run_all(m, disk_path=tmp_path, mqtt_connected=True)
        assert probes.has_failures(results) is True
        critical = probes.critical_failures(results)
        assert any(r.name == "inference_health" for r in critical)

        # 3) 日志：将探针失败结果写入 ring buffer
        buf = RingBufferLog(max_entries=100, retention_hours=24)
        for r in results:
            if not r.passed:
                buf.append(
                    level="WARN",
                    message=f"probe failed: {r.name} - {r.detail}",
                    metadata={"probe": r.name, "timestamp": r.timestamp.isoformat()},
                )

        # 4) 验证日志已记录
        warn_logs = buf.query(level="WARN", limit=100)
        assert len(warn_logs) > 0
        assert any("inference_health" in e.message for e in warn_logs)

        # 5) 导出告警日志（模拟断网恢复后补传）
        batch = buf.export_batch(since=datetime.now(UTC) - timedelta(minutes=1))
        assert len(batch) == len(warn_logs)
        buf.mark_exported(batch)
        assert buf.stats()["unexported"] == 0


# ===========================================================================
# 3. 降级链路集成（5 个测试）
# ===========================================================================


class TestDegradeChainIntegration:
    """降级链路集成：DegradeManager 矩阵 + 与监控/安全联动。"""

    def test_degrade_matrix_full_chain(self) -> None:
        """降级矩阵全链路：NORMAL → LLM_SMALL → LLM_OFF → EMERGENCY → 恢复。"""
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)

        # 1) NORMAL：内存 2000MB，一切正常
        d = manager.evaluate(_make_state(mem=2000, disk=10000))
        assert d.level == DegradeLevel.NORMAL
        assert d.actions == []

        # 2) LLM_SMALL：内存 3200MB（≥ 3000 阈值）
        d = manager.evaluate(_make_state(mem=3200, disk=10000))
        assert d.level == DegradeLevel.LLM_SMALL
        assert any(a.name == "limit_llm_tokens" for a in d.actions)

        # 3) LLM_OFF：内存 3700MB（≥ 3600 阈值）
        d = manager.evaluate(_make_state(mem=3700, disk=10000))
        assert d.level == DegradeLevel.LLM_OFF
        assert any(a.name == "unload_llm" for a in d.actions)

        # 4) EMERGENCY：LLM_OFF + DISK_CRITICAL 同时触发
        d = manager.evaluate(_make_state(mem=3700, disk=300))
        assert d.level == DegradeLevel.EMERGENCY
        assert any(a.name == "t1_alert_only" for a in d.actions)
        assert any(a.name == "unload_llm" for a in d.actions)

        # 5) hysteresis 恢复：从 EMERGENCY 恢复需要所有条件满足恢复阈值
        # 内存需 < 3000 * 0.9 = 2700，磁盘需 > 500 * 1.1 = 550
        d = manager.evaluate(_make_state(mem=2800, disk=600))
        # 2800 > 2700，未达恢复阈值，保持 EMERGENCY
        assert d.level == DegradeLevel.EMERGENCY
        d = manager.evaluate(_make_state(mem=2600, disk=600))
        # 满足恢复阈值，降为 NORMAL
        assert d.level == DegradeLevel.NORMAL

    def test_degrade_with_monitoring(self) -> None:
        """降级 + 监控联动：降级触发时 EdgeMetrics 记录 → 探针检测。"""
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds))
        metrics = EdgeMetrics()
        probes = LocalProbes()
        buf = RingBufferLog(max_entries=100, retention_hours=24)

        # 1) 触发 LLM_OFF 降级
        state = _make_state(mem=3700, disk=10000)
        decision = manager.evaluate(state)
        assert decision.level == DegradeLevel.LLM_OFF

        # 2) 指标记录降级事件
        metrics.set_model_version("v1.0-degraded")
        metrics.set_queue_depth(100)
        for _ in range(10):
            metrics.record_inference(5000.0)  # 降级后延迟变高

        # 3) 探针检测到异常
        results = probes.run_all(metrics, mqtt_connected=True)
        critical = probes.critical_failures(results)
        assert any(r.name == "inference_health" for r in critical)

        # 4) 日志记录降级决策
        buf.append(
            level="WARN",
            message=f"degrade to {decision.level.name}: {decision.reason}",
            metadata={
                "level": decision.level.name,
                "actions": [a.name for a in decision.actions],
            },
        )
        logs = buf.query(level="WARN", limit=10)
        assert len(logs) == 1
        assert "LLM_OFF" in logs[0].message

    def test_degrade_with_security(self, tmp_path: Path) -> None:
        """降级 + 安全联动：降级时模型签名校验仍有效（降级不影响安全）。"""
        # 1) 准备已签名模型
        priv, pub = generate_rsa_key_pair()
        model = tmp_path / "model.bin"
        model.write_bytes(b"signed-model")
        sig = sign_model(model, priv)
        assert verify_model(model, sig, pub) is True

        # 2) 触发降级
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds))
        state = _make_state(mem=3700, disk=10000)
        decision = manager.evaluate(state)
        assert decision.level == DegradeLevel.LLM_OFF

        # 3) 降级后模型签名校验仍然有效（安全不降级）
        assert verify_model(model, sig, pub) is True
        original_hash = compute_sha256(model)
        assert verify_model_integrity(model, original_hash) is True

        # 4) mTLS 证书验证也不受降级影响
        ca_cert, ca_key = mtls.generate_ca_cert("DegradeCA")
        dev_cert, dev_key = mtls.generate_device_cert("dev", ca_cert, ca_key)
        assert mtls.verify_cert(dev_cert, ca_cert) is True

        # 5) 脱敏也不受降级影响
        rotator = SaltRotator("salt-during-degrade")
        hashed = hash_device_id("device-001", rotator.current_salt)
        assert rotator.verify("device-001", hashed, datetime.now(UTC)) is True

    def test_temperature_throttle_chain(self) -> None:
        """温度降频联动：温度超阈值 → TEMP_THROTTLE → LLM 关闭动作 → 指标。"""
        thresholds = _default_thresholds()
        policy = DegradePolicy(thresholds)
        metrics = EdgeMetrics()

        # 1) 温度 85°C > 80°C 阈值
        state = _make_state(mem=2000, disk=10000, temp=85.0)
        decision = policy.evaluate(state)
        assert decision.level == DegradeLevel.TEMP_THROTTLE

        # 2) 验证降级动作包含 LLM 相关
        action_names = {a.name for a in decision.actions}
        assert "throttle_llm" in action_names
        assert "close_llm_on_temp" in action_names
        affected = set()
        for a in decision.actions:
            affected.update(a.affected_components)
        assert "t3_llm" in affected

        # 3) 指标记录温度异常导致的延迟变化
        # 模拟降频后推理延迟升高
        for _ in range(20):
            metrics.record_inference(3500.0)
        assert metrics.inference_p95_ms == 3500.0

        # 4) 探针检测到温度异常
        probes = LocalProbes()
        temp_result = probes.check_temperature(85.0, threshold_c=80.0)
        assert temp_result.passed is False
        assert "85.0C exceeds threshold" in temp_result.detail

        # 5) 温度恢复正常后降级解除（无 hysteresis 时）
        state_normal = _make_state(mem=2000, disk=10000, temp=70.0)
        decision_normal = policy.evaluate(state_normal)
        assert decision_normal.level == DegradeLevel.NORMAL

    def test_degrade_hysteresis_recovery(self) -> None:
        """降级 hysteresis 恢复：升级立即触发，恢复需低于阈值 × (1 - margin)。"""
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)

        # 1) 升级到 LLM_SMALL（mem=3200 ≥ 3000）
        d = manager.evaluate(_make_state(mem=3200, disk=10000))
        assert d.level == DegradeLevel.LLM_SMALL

        # 2) 内存降到 2900（< 3000 但 ≥ 2700 = 3000*0.9），hysteresis 保持
        d = manager.evaluate(_make_state(mem=2900, disk=10000))
        assert d.level == DegradeLevel.LLM_SMALL  # 保持
        assert "hysteresis" in d.reason

        # 3) 内存降到 2600（< 2700），恢复到 NORMAL
        d = manager.evaluate(_make_state(mem=2600, disk=10000))
        assert d.level == DegradeLevel.NORMAL

        # 4) 历史记录验证
        history = manager.history()
        assert len(history) == 3
        assert history[0].level == DegradeLevel.LLM_SMALL
        assert history[1].level == DegradeLevel.LLM_SMALL  # hysteresis 保持
        assert history[2].level == DegradeLevel.NORMAL


# ===========================================================================
# 4. 部署配置集成（4 个测试）
# ===========================================================================


class TestDeploymentConfigIntegration:
    """部署配置集成：Dockerfile/compose/systemd/grafana/checklist 跨文件一致性。"""

    def test_dockerfile_compose_consistency(
        self, dockerfile_text: str, compose_yaml: dict
    ) -> None:
        """Dockerfile + docker-compose 一致性：端口/路径/环境变量。"""
        # 1) Dockerfile EXPOSE 9000 与 compose ports 9000:9000 一致
        expose_match = re.search(r"EXPOSE\s+(\d+)", dockerfile_text)
        assert expose_match, "Dockerfile 缺少 EXPOSE"
        expose_port = expose_match.group(1)
        assert expose_port == "9000"

        edge_svc = compose_yaml["services"]["edge-agent"]
        port_mapping = edge_svc["ports"][0]
        assert port_mapping == "9000:9000", f"compose 端口映射应为 9000:9000，实际：{port_mapping}"

        # 2) EDGE_API_PORT 环境变量与 EXPOSE 一致
        env = {}
        for item in edge_svc["environment"]:
            k, _, v = item.partition("=")
            env[k] = v
        assert env["EDGE_API_PORT"] == expose_port

        # 3) Dockerfile USER 非 root 与 compose cap_drop ALL 一致
        assert "USER edge-agent" in dockerfile_text
        assert "ALL" in edge_svc.get("cap_drop", [])

        # 4) Dockerfile HEALTHCHECK 与 compose healthcheck 一致
        assert "HEALTHCHECK" in dockerfile_text
        assert "/health" in dockerfile_text
        hc = edge_svc["healthcheck"]
        assert any("/health" in str(t) for t in hc["test"])

        # 5) 镜像名一致
        assert "edge-ai-agent" in dockerfile_text
        assert edge_svc["image"] == "edge-ai-agent:1.0.0"

        # 6) mgmt-agent 端口 9001 与 MGMT_API_PORT 一致
        mgmt_svc = compose_yaml["services"]["mgmt-agent"]
        mgmt_env = {}
        for item in mgmt_svc["environment"]:
            k, _, v = item.partition("=")
            mgmt_env[k] = v
        assert mgmt_env["MGMT_API_PORT"] == "9001"
        assert mgmt_svc["ports"][0] == "9001:9001"

    def test_systemd_service_order(
        self, edge_service_text: str, mgmt_service_text: str
    ) -> None:
        """systemd 服务顺序：mgmt-agent Before=edge-agent，资源限制正确。"""
        edge_svc = _parse_systemd(edge_service_text)
        mgmt_svc = _parse_systemd(mgmt_service_text)

        # 1) mgmt-agent 必须声明 Before=edge-agent.service
        before = mgmt_svc["Unit"].get("Before", "")
        assert "edge-agent.service" in before, (
            "mgmt-agent.service 必须声明 Before=edge-agent.service"
        )

        # 2) edge-agent 必须声明 After=mgmt-agent.service
        after = edge_svc["Unit"].get("After", "")
        assert "mgmt-agent.service" in after, (
            "edge-agent.service 必须声明 After=mgmt-agent.service"
        )

        # 3) edge-agent 资源限制：CPUQuota=200% / MemoryMax=4G / TasksMax=256
        assert edge_svc["Service"]["CPUQuota"] == "200%"
        assert edge_svc["Service"]["MemoryMax"] == "4G"
        assert edge_svc["Service"]["TasksMax"] == "256"

        # 4) mgmt-agent 资源限制：CPUQuota=50% / MemoryMax=512M / TasksMax=64
        assert mgmt_svc["Service"]["CPUQuota"] == "50%"
        assert mgmt_svc["Service"]["MemoryMax"] == "512M"
        assert mgmt_svc["Service"]["TasksMax"] == "64"

        # 5) 安全加固一致
        for svc in (edge_svc, mgmt_svc):
            assert svc["Service"]["NoNewPrivileges"] == "true"
            assert svc["Service"]["ProtectSystem"] == "strict"
            assert svc["Service"]["PrivateTmp"] == "true"

        # 6) mgmt-agent Restart=always（永不退出）
        assert mgmt_svc["Service"]["Restart"] == "always"
        # edge-agent Restart=on-failure
        assert edge_svc["Service"]["Restart"] == "on-failure"

    def test_grafana_dashboard_completeness(self, grafana_dashboard: dict) -> None:
        """Grafana 看板完整性：所有关键指标面板存在。"""
        panels = grafana_dashboard["panels"]
        titles = {p["title"] for p in panels}

        # 1) 关键监控面板存在
        required_panels = {
            "推理 P95 延迟 (ms)",
            "模型版本",
            "队列深度",
            "离线时长 (h)",
            "缓存命中率",
            "云端升级率趋势",
            "CPU 使用率",
            "内存使用率",
            "CPU 温度 (°C)",
        }
        missing = required_panels - titles
        assert not missing, f"缺少面板：{missing}"

        # 2) 所有面板都有 Prometheus 数据源
        for p in panels:
            ds = p.get("datasource", {})
            assert ds.get("type") == "prometheus", (
                f"面板 {p['title']} 数据源类型应为 prometheus"
            )

        # 3) 关键面板有 PromQL 表达式
        exprs = []
        for p in panels:
            for t in p.get("targets", []):
                if "expr" in t:
                    exprs.append(t["expr"])
        # 应包含 edge_ 前缀的指标
        assert any("edge_" in e for e in exprs), "应包含 edge_ 前缀指标"

        # 4) 告警规则存在
        alert_group = grafana_dashboard.get("alertRuleGroup", {})
        rules = alert_group.get("rules", [])
        rule_names = {r["name"] for r in rules}
        assert "推理 P95 延迟超阈值" in rule_names
        assert "队列积压告警" in rule_names
        assert "CPU 温度过高" in rule_names
        assert "内存使用率过高" in rule_names

        # 5) 模板变量包含 gw_id
        templating = grafana_dashboard.get("templating", {})
        var_names = {v["name"] for v in templating.get("list", [])}
        assert "gw_id" in var_names
        assert "DS_PROMETHEUS" in var_names

    def test_migration_checklist_completeness(self, migration_checklist: str) -> None:
        """迁移检查清单验证：覆盖所有关键迁移步骤。"""
        content = migration_checklist

        # 1) 必须包含的核心章节
        required_sections = [
            "零改码迁移验证",
            "配置差异",
            "真机重测指标",
            "安全加固验证",
            "可观测性验证",
        ]
        for section in required_sections:
            assert section in content, f"检查清单缺少章节：{section}"

        # 2) 零改码迁移关键步骤
        required_steps = [
            "docker build",
            "docker-compose.edge.yml",
            "systemctl enable",
            "mgmt-agent.service",
            "Before=edge-agent.service",
            "/health",
            "mosquitto",
        ]
        for step in required_steps:
            assert step in content, f"检查清单缺少步骤：{step}"

        # 3) 安全加固项
        required_security = [
            "非 root 用户",
            "read_only",
            "cap_drop",
            "NoNewPrivileges",
            "模型签名校验",
            "mTLS",
        ]
        for item in required_security:
            assert item in content, f"检查清单缺少安全项：{item}"

        # 4) 指标项
        required_metrics = [
            "T1 P95 < 50ms",
            "T2 P95 < 200ms",
            "T4 P95 < 1.2s",
            "缓存命中率",
            "OTA 成功率",
        ]
        for metric in required_metrics:
            assert metric in content, f"检查清单缺少指标项：{metric}"

        # 5) 可观测性项
        assert "Prometheus" in content
        assert "Grafana" in content
        assert "ring buffer" in content


# ===========================================================================
# 5. 全链路集成（4 个测试）
# ===========================================================================


class TestFullHardeningChain:
    """全链路集成：安全 + 监控 + 降级 + 部署 一站式硬化场景。"""

    def test_security_monitoring_degrade_chain(self, tmp_path: Path) -> None:
        """安全 + 监控 + 降级全链路：在一个测试中串联三大模块。"""
        device_id = "edge-gw-full-chain"

        # === 安全模块 ===
        # 1) mTLS 证书
        ca_cert, ca_key = mtls.generate_ca_cert("FullChainCA")
        dev_cert, dev_key = mtls.generate_device_cert(device_id, ca_cert, ca_key)
        assert mtls.verify_cert(dev_cert, ca_cert) is True
        cert_fp = mtls.fingerprint(dev_cert)

        # 2) 模型签名
        priv, pub = generate_rsa_key_pair()
        model = tmp_path / "model.bin"
        model.write_bytes(b"full-chain-model")
        sig = sign_model(model, priv)
        assert verify_model(model, sig, pub) is True

        # 3) 脱敏
        rotator = SaltRotator(cert_fp[:32])
        masked_device = hash_device_id(device_id, rotator.current_salt)

        # === 监控模块 ===
        metrics = EdgeMetrics()
        metrics.set_model_version("v1.0.0")
        metrics.set_queue_depth(100)
        for _ in range(50):
            metrics.record_inference(150.0)
        for _ in range(8):
            metrics.record_cache_hit(True)
        for _ in range(2):
            metrics.record_cache_hit(False)

        buf = RingBufferLog(max_entries=1000, retention_hours=24)
        buf.append("INFO", "device started", metadata={"device": masked_device})

        # === 降级模块 ===
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds))
        state = _make_state(mem=2000, disk=10000, temp=45.0)
        decision = manager.evaluate(state)
        assert decision.level == DegradeLevel.NORMAL

        # === 串联验证 ===
        # 探针基于指标运行
        probes = LocalProbes()
        results = probes.run_all(metrics, mqtt_connected=True)
        assert probes.has_failures(results) is False

        # 导出 Prometheus 指标
        prom = metrics.export_prometheus()
        assert "edge_inference_p95_ms 150.0" in prom
        assert "edge_local_cache_hit_rate 0.8" in prom

        # 日志记录全链路状态
        buf.append(
            level="INFO",
            message="full chain healthy",
            metadata={
                "cert_fp": cert_fp,
                "model_verified": True,
                "degrade_level": decision.level.name,
                "cache_hit_rate": metrics.local_cache_hit_rate,
            },
        )
        logs = buf.query(level="INFO", limit=10)
        assert len(logs) == 2
        assert "full chain healthy" in logs[1].message

    def test_hardening_e2e_scenario(self, tmp_path: Path) -> None:
        """硬化 E2E 场景：设备启动 → mTLS → 模型校验 → 指标 → 降级 → 告警 → 日志。"""
        # === 阶段 1：设备启动 + mTLS 握手 ===
        ca_cert, ca_key = mtls.generate_ca_cert("E2ECA")
        dev_cert, dev_key = mtls.generate_device_cert("e2e-device", ca_cert, ca_key)
        ctx = mtls.load_cert_chain(dev_cert, dev_key)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert mtls.verify_cert(dev_cert, ca_cert) is True

        # === 阶段 2：模型签名校验 ===
        priv, pub = generate_rsa_key_pair()
        model = tmp_path / "e2e_model.bin"
        model.write_bytes(b"e2e-model-content" * 100)
        original_hash = compute_sha256(model)
        sig = sign_model(model, priv)
        assert verify_model(model, sig, pub) is True
        assert verify_model_integrity(model, original_hash) is True

        # === 阶段 3：指标收集 ===
        metrics = EdgeMetrics()
        metrics.set_model_version("v3.1.4")
        metrics.set_queue_depth(500)
        for _ in range(100):
            metrics.record_inference(800.0)
        metrics.set_offline_duration(0)

        # === 阶段 4：资源降级（内存升高触发 LLM_SMALL）===
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)
        state = _make_state(mem=3200, disk=10000, temp=70.0)
        decision = manager.evaluate(state)
        assert decision.level == DegradeLevel.LLM_SMALL
        assert any(a.name == "limit_llm_tokens" for a in decision.actions)

        # 降级后指标变化
        metrics.set_model_version("v3.1.4-degraded")
        for _ in range(20):
            metrics.record_inference(5000.0)  # 降级后延迟升高，P95 超 3000ms 阈值

        # === 阶段 5：探针告警 ===
        probes = LocalProbes()
        results = probes.run_all(metrics, mqtt_connected=True, temp_c=70.0)
        # P95 应超阈值（3000ms 默认）：120 样本中后 20 个为 5000，
        # P95 = sorted[int(0.95*119)] = sorted[113] = 5000
        inference_result = next(r for r in results if r.name == "inference_health")
        assert inference_result.passed is False

        # === 阶段 6：日志记录 ===
        buf = RingBufferLog(max_entries=1000, retention_hours=24)
        # 记录降级事件
        buf.append(
            level="WARN",
            message=f"degrade to {decision.level.name}",
            metadata={"reason": decision.reason, "actions": [a.name for a in decision.actions]},
        )
        # 记录探针失败
        for r in results:
            if not r.passed:
                buf.append(
                    level="WARN",
                    message=f"probe failed: {r.name}",
                    metadata={"probe": r.name, "detail": r.detail},
                )

        # 验证日志
        warn_logs = buf.query(level="WARN", limit=100)
        assert len(warn_logs) >= 2  # 至少降级 + 探针失败
        assert any("degrade to LLM_SMALL" in e.message for e in warn_logs)
        assert any("inference_health" in e.message for e in warn_logs)

        # === 阶段 7：断网恢复后批量导出 ===
        batch = buf.export_batch(since=datetime.now(UTC) - timedelta(minutes=5))
        assert len(batch) == len(warn_logs)
        buf.mark_exported(batch)
        assert buf.stats()["unexported"] == 0

    def test_migration_readiness(
        self,
        dockerfile_text: str,
        compose_yaml: dict,
        edge_service_text: str,
        mgmt_service_text: str,
        grafana_dashboard: dict,
        migration_checklist: str,
    ) -> None:
        """迁移就绪验证：所有部署文件存在且配置正确，代码零改动即可迁移。"""
        # 1) 所有部署文件存在
        assert DOCKERFILE.exists()
        assert COMPOSE_FILE.exists()
        assert EDGE_SERVICE.exists()
        assert MGMT_SERVICE.exists()
        assert GRAFANA_DASHBOARD.exists()
        assert MIGRATION_CHECKLIST.exists()

        # 2) Dockerfile 关键配置
        assert "python:3.12-slim" in dockerfile_text
        assert "USER edge-agent" in dockerfile_text
        assert "HEALTHCHECK" in dockerfile_text
        assert "EXPOSE 9000" in dockerfile_text

        # 3) docker-compose 关键服务
        services = compose_yaml["services"]
        required_services = {"edge-agent", "mgmt-agent", "mosquitto", "cloud-api"}
        assert required_services.issubset(set(services.keys()))

        # 4) systemd 服务顺序正确
        mgmt_svc = _parse_systemd(mgmt_service_text)
        edge_svc = _parse_systemd(edge_service_text)
        assert "edge-agent.service" in mgmt_svc["Unit"].get("Before", "")
        assert "mgmt-agent.service" in edge_svc["Unit"].get("After", "")

        # 5) Grafana 看板有面板和告警
        assert len(grafana_dashboard["panels"]) >= 9
        assert len(grafana_dashboard["alertRuleGroup"]["rules"]) >= 4

        # 6) 迁移检查清单覆盖所有章节
        for section in [
            "零改码迁移验证",
            "配置差异",
            "真机重测指标",
            "安全加固验证",
            "可观测性验证",
        ]:
            assert section in migration_checklist

        # 7) 跨文件端口一致性
        edge_compose = services["edge-agent"]
        env = {}
        for item in edge_compose["environment"]:
            k, _, v = item.partition("=")
            env[k] = v
        assert env["EDGE_API_PORT"] == "9000"
        assert edge_svc["Service"]["Environment"] == "EDGE_API_PORT=9000"
        assert edge_compose["ports"][0] == "9000:9000"

    def test_hardening_resilience_scenario(self, tmp_path: Path) -> None:
        """硬化韧性场景：模拟断网 + 高温 + 内存压力的复合故障。"""
        # 1) 准备安全基线
        ca_cert, ca_key = mtls.generate_ca_cert("ResilienceCA")
        dev_cert, _ = mtls.generate_device_cert("resilient-dev", ca_cert, ca_key)
        assert mtls.verify_cert(dev_cert, ca_cert) is True

        priv, pub = generate_rsa_key_pair()
        model = tmp_path / "model.bin"
        model.write_bytes(b"resilient-model")
        sig = sign_model(model, priv)

        # 2) 复合故障：高温 + 内存压力 + 长时间离线
        thresholds = _default_thresholds()
        manager = DegradeManager(DegradePolicy(thresholds), hysteresis_margin=0.1)

        # 故障 1：内存 3700MB + 高温 85°C + 离线 80000s（< 86400 不触发 OFFLINE_DEGRADED）
        state_1 = _make_state(mem=3700, disk=10000, temp=85.0, offline=80000)
        d_1 = manager.evaluate(state_1)
        # LLM_OFF (high_risk) + TEMP_THROTTLE (非 high_risk)，仅 1 个 high_risk 不升级 EMERGENCY
        # 取触发的最高级别：LLM_OFF (value=2) vs TEMP_THROTTLE (value=3) → TEMP_THROTTLE
        assert d_1.level == DegradeLevel.TEMP_THROTTLE

        # 故障 2：加上磁盘危急 + 长时间离线 → 多个 high_risk → EMERGENCY
        state_2 = _make_state(mem=3700, disk=300, temp=85.0, offline=90000)
        d_2 = manager.evaluate(state_2)
        # LLM_OFF + OFFLINE_DEGRADED + DISK_CRITICAL 三个 high_risk → EMERGENCY
        assert d_2.level == DegradeLevel.EMERGENCY

        # 3) 安全模块在复合故障下仍有效
        assert verify_model(model, sig, pub) is True
        assert mtls.verify_cert(dev_cert, ca_cert) is True

        # 4) 监控模块记录复合故障
        metrics = EdgeMetrics()
        metrics.set_model_version("emergency-mode")
        metrics.set_queue_depth(50000)  # 队列积压
        metrics.set_offline_duration(90000)  # 25h 离线
        for _ in range(50):
            metrics.record_inference(10000.0)  # 延迟极高

        # 5) 探针全面告警
        probes = LocalProbes()
        results = probes.run_all(
            metrics,
            temp_c=85.0,
            mqtt_connected=False,  # 断网
        )
        assert probes.has_failures(results) is True
        critical = probes.critical_failures(results)
        critical_names = {r.name for r in critical}
        assert "inference_health" in critical_names
        assert "model_loaded" not in critical_names  # 模型已加载（非 unknown）
        assert "mqtt_connected" in critical_names

        # 6) 日志记录复合故障
        buf = RingBufferLog(max_entries=1000, retention_hours=24)
        buf.append(
            level="ERROR",
            message=f"EMERGENCY declared: {d_2.reason}",
            metadata={
                "level": d_2.level.name,
                "actions": [a.name for a in d_2.actions],
                "critical_probes": list(critical_names),
            },
        )
        error_logs = buf.query(level="ERROR", limit=10)
        assert len(error_logs) == 1
        assert "EMERGENCY" in error_logs[0].message

        # 7) Prometheus 指标可导出（即使在紧急模式）
        prom = metrics.export_prometheus()
        assert "edge_queue_depth 50000" in prom
        assert "edge_offline_duration_s 90000" in prom
