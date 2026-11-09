"""E3 监控可观测模块单元测试。

覆盖：
- EdgeMetrics 指标收集（P95 滑动窗口/版本/队列/缓存命中率/升级率）
- Prometheus 文本格式导出
- 指标快照
- LocalProbes 各探针正常/异常路径
- LocalProbes 聚合（run_all/has_failures/critical_failures）
- RingBufferLog 追加/查询/过期清理/批量导出/标记已导出/统计
- 线程安全
"""
from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from edge.monitoring.metrics import EdgeMetrics
from edge.monitoring.probes import CRITICAL_PROBE_NAMES, LocalProbes, ProbeResult
from edge.monitoring.ring_buffer import LogEntry, RingBufferLog


# =============================================================================
# EdgeMetrics — 指标收集
# =============================================================================


class TestEdgeMetricsInference:
    """推理 P95 延迟与滑动窗口。"""

    def test_p95_empty_returns_zero(self):
        m = EdgeMetrics()
        assert m.inference_p95_ms == 0.0

    def test_p95_single_sample(self):
        m = EdgeMetrics()
        m.record_inference(42.0)
        assert m.inference_p95_ms == 42.0

    def test_p95_multiple_samples(self):
        """100 个样本 [1..100]，P95 = sorted[int(0.95*99)] = sorted[94] = 95。"""
        m = EdgeMetrics()
        for i in range(1, 101):
            m.record_inference(float(i))
        assert m.inference_p95_ms == 95.0

    def test_p95_sliding_window_drops_old(self):
        """窗口大小 5，记录 6 个样本，最旧的被丢弃。"""
        m = EdgeMetrics(p95_window=5)
        for v in [10, 20, 30, 40, 50, 60]:
            m.record_inference(float(v))
        # 窗口应为 [20,30,40,50,60]，P95=sorted[int(0.95*4)]=sorted[3]=50
        assert m.inference_p95_ms == 50.0

    def test_p95_default_window_1000(self):
        m = EdgeMetrics()
        assert m.DEFAULT_P95_WINDOW == 1000
        for i in range(1500):
            m.record_inference(float(i))
        # 窗口保留最近 1000 个：[500..1499]
        # P95 = sorted[int(0.95*999)] = sorted[949] = 500+949 = 1449
        assert m.inference_p95_ms == 1449.0

    def test_record_inference_negative_raises(self):
        m = EdgeMetrics()
        with pytest.raises(ValueError, match="non-negative"):
            m.record_inference(-1.0)

    def test_invalid_p95_window_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EdgeMetrics(p95_window=0)


class TestEdgeMetricsSetters:
    """版本/队列深度/离线时长设置。"""

    def test_set_model_version(self):
        m = EdgeMetrics()
        assert m.model_version == "unknown"
        m.set_model_version("v2.3.1")
        assert m.model_version == "v2.3.1"

    def test_set_queue_depth(self):
        m = EdgeMetrics()
        m.set_queue_depth(42)
        assert m.queue_depth == 42

    def test_set_queue_depth_negative_raises(self):
        m = EdgeMetrics()
        with pytest.raises(ValueError, match="non-negative"):
            m.set_queue_depth(-1)

    def test_set_offline_duration(self):
        m = EdgeMetrics()
        m.set_offline_duration(3600)
        assert m.offline_duration_s == 3600

    def test_set_offline_duration_negative_raises(self):
        m = EdgeMetrics()
        with pytest.raises(ValueError, match="non-negative"):
            m.set_offline_duration(-1)


class TestEdgeMetricsRates:
    """缓存命中率与云端升级率。"""

    def test_cache_hit_rate_empty(self):
        m = EdgeMetrics()
        assert m.local_cache_hit_rate == 0.0

    def test_cache_hit_rate_all_hits(self):
        m = EdgeMetrics()
        for _ in range(10):
            m.record_cache_hit(True)
        assert m.local_cache_hit_rate == 1.0

    def test_cache_hit_rate_mixed(self):
        m = EdgeMetrics()
        for hit in [True, False, True, True, False]:  # 3/5
            m.record_cache_hit(hit)
        assert m.local_cache_hit_rate == pytest.approx(0.6)

    def test_cloud_escalation_rate_empty(self):
        m = EdgeMetrics()
        assert m.cloud_escalation_rate == 0.0

    def test_cloud_escalation_rate_mixed(self):
        m = EdgeMetrics()
        for esc in [True, False, True, False, False]:  # 2/5
            m.record_cloud_escalation(esc)
        assert m.cloud_escalation_rate == pytest.approx(0.4)

    def test_cloud_escalation_rate_all_escalated(self):
        m = EdgeMetrics()
        for _ in range(5):
            m.record_cloud_escalation(True)
        assert m.cloud_escalation_rate == 1.0


# =============================================================================
# EdgeMetrics — Prometheus 导出
# =============================================================================


class TestExportPrometheus:
    """Prometheus 文本格式导出。"""

    def test_export_contains_required_metrics(self):
        m = EdgeMetrics()
        m.set_model_version("v1.0.0")
        m.set_queue_depth(5)
        m.set_offline_duration(120)
        out = m.export_prometheus()
        assert "# TYPE edge_inference_p95_ms gauge" in out
        assert "# TYPE edge_model_version_info gauge" in out
        assert "# TYPE edge_queue_depth gauge" in out
        assert "# TYPE edge_offline_duration_s gauge" in out
        assert "# TYPE edge_local_cache_hit_rate gauge" in out
        assert "# TYPE edge_cloud_escalation_rate gauge" in out
        assert "edge_queue_depth 5" in out
        assert "edge_offline_duration_s 120" in out

    def test_export_model_version_label(self):
        m = EdgeMetrics()
        m.set_model_version("v2.3.1")
        out = m.export_prometheus()
        assert 'edge_model_version_info{version="v2.3.1"} 1' in out

    def test_export_label_escaping(self):
        m = EdgeMetrics()
        m.set_model_version('a"b\\c\nd')  # 含引号、反斜杠、换行
        out = m.export_prometheus()
        # 换行在 label 中转为 \n 字面，需检查该行不含真实换行
        version_line = [ln for ln in out.splitlines() if ln.startswith("edge_model_version_info")][0]
        assert version_line.count("\n") == 0  # 该行内无真实换行
        assert '\\"' in version_line  # 引号被转义
        assert "\\\\" in version_line  # 反斜杠被转义

    def test_export_ends_with_newline(self):
        m = EdgeMetrics()
        out = m.export_prometheus()
        assert out.endswith("\n")

    def test_export_help_lines_present(self):
        m = EdgeMetrics()
        out = m.export_prometheus()
        assert "# HELP edge_inference_p95_ms" in out
        assert "# HELP edge_queue_depth" in out


# =============================================================================
# EdgeMetrics — 快照
# =============================================================================


class TestSnapshot:
    """指标快照。"""

    def test_snapshot_keys(self):
        m = EdgeMetrics()
        snap = m.snapshot()
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
        assert expected_keys.issubset(snap.keys())

    def test_snapshot_values(self):
        m = EdgeMetrics()
        m.record_inference(10.0)
        m.record_inference(20.0)
        m.set_model_version("v9.9")
        m.set_queue_depth(7)
        m.set_offline_duration(300)
        m.record_cache_hit(True)
        m.record_cache_hit(False)
        m.record_cloud_escalation(True)
        snap = m.snapshot()
        assert snap["inference_samples"] == 2
        assert snap["model_version"] == "v9.9"
        assert snap["queue_depth"] == 7
        assert snap["offline_duration_s"] == 300
        assert snap["cache_hits"] == 1
        assert snap["cache_total"] == 2
        assert snap["cloud_escalations"] == 1
        assert snap["cloud_total"] == 1


# =============================================================================
# LocalProbes — 单项探针
# =============================================================================


class TestProbes:
    """各探针正常/异常路径。"""

    def setup_method(self):
        self.probes = LocalProbes()

    def test_inference_health_pass(self):
        r = self.probes.check_inference_health(100.0, threshold_ms=3000)
        assert r.name == "inference_health"
        assert r.passed is True
        assert "within" in r.detail

    def test_inference_health_fail(self):
        r = self.probes.check_inference_health(3500.0, threshold_ms=3000)
        assert r.passed is False
        assert "exceeds" in r.detail

    def test_inference_health_boundary(self):
        r = self.probes.check_inference_health(3000.0, threshold_ms=3000)
        assert r.passed is True  # <= 阈值即通过

    def test_disk_space_pass(self, tmp_path):
        r = self.probes.check_disk_space(tmp_path, min_mb=1)
        assert r.name == "disk_space"
        assert r.passed is True

    def test_disk_space_fail(self, tmp_path):
        """mock shutil.disk_usage 返回极小空闲空间。"""
        fake_usage = type("U", (), {"total": 100, "used": 99, "free": 100})()  # free < 1MB
        with patch("edge.monitoring.probes.shutil.disk_usage", return_value=fake_usage):
            r = self.probes.check_disk_space(tmp_path, min_mb=500)
        assert r.passed is False
        assert "min 500MB" in r.detail

    def test_disk_space_nonexistent_path_falls_back(self):
        r = self.probes.check_disk_space(Path("Z:/nonexistent_xyz_abc"), min_mb=1)
        assert r.name == "disk_space"
        # 回退到当前目录，应能查询
        assert isinstance(r.passed, bool)

    def test_temperature_pass(self):
        r = self.probes.check_temperature(50.0, threshold_c=80.0)
        assert r.name == "temperature"
        assert r.passed is True

    def test_temperature_fail(self):
        r = self.probes.check_temperature(85.0, threshold_c=80.0)
        assert r.passed is False
        assert "exceeds" in r.detail

    def test_temperature_none_passes(self):
        """温度传感器不可读时通过（降级由 guard 处理）。"""
        r = self.probes.check_temperature(None, threshold_c=80.0)
        assert r.passed is True
        assert "unavailable" in r.detail

    def test_queue_depth_pass(self):
        r = self.probes.check_queue_depth(100, max_depth=10000)
        assert r.name == "queue_depth"
        assert r.passed is True

    def test_queue_depth_fail(self):
        r = self.probes.check_queue_depth(20000, max_depth=10000)
        assert r.passed is False

    def test_model_loaded_pass(self):
        r = self.probes.check_model_loaded("v1.2.3")
        assert r.name == "model_loaded"
        assert r.passed is True

    def test_model_loaded_fail_none(self):
        r = self.probes.check_model_loaded(None)
        assert r.passed is False

    def test_model_loaded_fail_unknown(self):
        r = self.probes.check_model_loaded("unknown")
        assert r.passed is False

    def test_model_loaded_fail_empty(self):
        r = self.probes.check_model_loaded("")
        assert r.passed is False

    def test_mqtt_connected_pass(self):
        r = self.probes.check_mqtt_connected(True)
        assert r.name == "mqtt_connected"
        assert r.passed is True

    def test_mqtt_connected_fail(self):
        r = self.probes.check_mqtt_connected(False)
        assert r.passed is False
        assert "disconnected" in r.detail

    def test_probe_result_timestamp_auto(self):
        r = self.probes.check_mqtt_connected(True)
        assert isinstance(r.timestamp, datetime)


# =============================================================================
# LocalProbes — 聚合
# =============================================================================


class TestProbesAggregation:
    """run_all / has_failures / critical_failures。"""

    def test_run_all_basic(self):
        probes = LocalProbes()
        m = EdgeMetrics()
        m.set_model_version("v1.0")
        m.set_queue_depth(5)
        results = probes.run_all(m)
        names = {r.name for r in results}
        # 必含三项基础探针
        assert {"inference_health", "queue_depth", "model_loaded"}.issubset(names)
        assert all(isinstance(r, ProbeResult) for r in results)

    def test_run_all_with_optional(self, tmp_path):
        probes = LocalProbes()
        m = EdgeMetrics()
        results = probes.run_all(
            m,
            disk_path=tmp_path,
            temp_c=50.0,
            mqtt_connected=True,
        )
        names = {r.name for r in results}
        assert "disk_space" in names
        assert "temperature" in names
        assert "mqtt_connected" in names

    def test_has_failures_true(self):
        probes = LocalProbes()
        m = EdgeMetrics()
        m.record_inference(5000.0)  # P95 超阈值
        results = probes.run_all(m)
        assert probes.has_failures(results) is True

    def test_has_failures_false(self):
        probes = LocalProbes()
        m = EdgeMetrics()
        m.set_model_version("v1.0")
        m.record_inference(100.0)
        results = probes.run_all(m)
        assert probes.has_failures(results) is False

    def test_critical_failures_filters(self):
        probes = LocalProbes()
        m = EdgeMetrics()
        # model 未加载（critical）+ 队列积压（非 critical）
        m.set_queue_depth(999999)
        results = probes.run_all(m)
        critical = probes.critical_failures(results)
        # model_loaded 失败应被识别为 critical
        assert any(r.name == "model_loaded" for r in critical)
        # queue_depth 不在 critical 集合
        assert all(r.name != "queue_depth" for r in critical)

    def test_critical_failures_empty_when_all_pass(self):
        probes = LocalProbes()
        m = EdgeMetrics()
        m.set_model_version("v1.0")
        m.record_inference(100.0)
        results = probes.run_all(m, mqtt_connected=True)
        assert probes.critical_failures(results) == []

    def test_critical_probe_names_constant(self):
        assert "inference_health" in CRITICAL_PROBE_NAMES
        assert "model_loaded" in CRITICAL_PROBE_NAMES
        assert "mqtt_connected" in CRITICAL_PROBE_NAMES
        assert "disk_space" not in CRITICAL_PROBE_NAMES


# =============================================================================
# RingBufferLog — 追加/查询/清理/导出/统计
# =============================================================================


class TestRingBufferAppend:
    """追加与基本查询。"""

    def test_append_single(self):
        buf = RingBufferLog()
        buf.append("INFO", "hello")
        assert len(buf) == 1
        entries = buf.query(limit=10)
        assert len(entries) == 1
        assert entries[0].level == "INFO"
        assert entries[0].message == "hello"
        assert entries[0].metadata == {}
        assert entries[0].exported is False

    def test_append_with_metadata(self):
        buf = RingBufferLog()
        buf.append("ERROR", "boom", metadata={"device": "X1", "code": 500})
        e = buf.query(limit=10)[0]
        assert e.metadata == {"device": "X1", "code": 500}

    def test_append_metadata_isolated(self):
        """传入的 metadata 应被拷贝，避免外部修改影响条目。"""
        buf = RingBufferLog()
        meta = {"k": 1}
        buf.append("INFO", "m", metadata=meta)
        meta["k"] = 999  # 修改原 dict
        e = buf.query(limit=10)[0]
        assert e.metadata == {"k": 1}

    def test_max_entries_eviction(self):
        buf = RingBufferLog(max_entries=3)
        for i in range(5):
            buf.append("INFO", f"msg{i}")
        assert len(buf) == 3
        msgs = [e.message for e in buf.query(limit=10)]
        assert msgs == ["msg2", "msg3", "msg4"]

    def test_invalid_max_entries_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RingBufferLog(max_entries=0)

    def test_invalid_retention_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RingBufferLog(retention_hours=0)


class TestRingBufferQuery:
    """查询过滤。"""

    def test_query_by_level(self):
        buf = RingBufferLog()
        buf.append("INFO", "a")
        buf.append("ERROR", "b")
        buf.append("INFO", "c")
        infos = buf.query(level="INFO", limit=10)
        assert [e.message for e in infos] == ["a", "c"]
        errors = buf.query(level="ERROR", limit=10)
        assert [e.message for e in errors] == ["b"]

    def test_query_by_since(self):
        buf = RingBufferLog()
        buf.append("INFO", "old")
        cutoff = datetime.now(UTC) + timedelta(milliseconds=10)
        # 确保新条目在 cutoff 之后
        import time

        time.sleep(0.02)
        buf.append("INFO", "new")
        result = buf.query(since=cutoff, limit=10)
        assert [e.message for e in result] == ["new"]

    def test_query_limit_returns_recent(self):
        buf = RingBufferLog()
        for i in range(10):
            buf.append("INFO", f"m{i}")
        # limit=3 应返回最近 3 条（按时间正序）
        result = buf.query(limit=3)
        assert [e.message for e in result] == ["m7", "m8", "m9"]

    def test_query_limit_zero(self):
        buf = RingBufferLog()
        buf.append("INFO", "x")
        assert buf.query(limit=0) == []

    def test_query_empty_buffer(self):
        buf = RingBufferLog()
        assert buf.query(limit=10) == []


class TestRingBufferPurge:
    """过期清理。"""

    def test_purge_expired_removes_old(self):
        buf = RingBufferLog(retention_hours=1)
        # 手动注入一条过期条目
        old_entry = LogEntry(
            timestamp=datetime.now(UTC) - timedelta(hours=2),
            level="INFO",
            message="old",
        )
        buf._buffer.append(old_entry)  # type: ignore[attr-defined]
        buf.append("INFO", "fresh")
        removed = buf.purge_expired()
        assert removed == 1
        assert len(buf) == 1
        assert buf.query(limit=10)[0].message == "fresh"

    def test_purge_expired_none_when_all_fresh(self):
        buf = RingBufferLog(retention_hours=24)
        buf.append("INFO", "a")
        assert buf.purge_expired() == 0

    def test_purge_expired_empty_buffer(self):
        buf = RingBufferLog()
        assert buf.purge_expired() == 0


class TestRingBufferExport:
    """批量导出与标记。"""

    def test_export_batch_returns_unexported_since(self):
        buf = RingBufferLog()
        buf.append("INFO", "a")
        buf.append("ERROR", "b")
        since = datetime.now(UTC) - timedelta(minutes=1)
        batch = buf.export_batch(since)
        assert len(batch) == 2
        assert all(not e.exported for e in batch)

    def test_export_batch_excludes_exported(self):
        buf = RingBufferLog()
        buf.append("INFO", "a")
        buf.append("INFO", "b")
        since = datetime.now(UTC) - timedelta(minutes=1)
        batch = buf.export_batch(since)
        buf.mark_exported(batch)
        # 再次导出应空
        assert buf.export_batch(since) == []

    def test_mark_exported_sets_flag(self):
        buf = RingBufferLog()
        buf.append("INFO", "a")
        entries = buf.query(limit=10)
        buf.mark_exported(entries)
        assert all(e.exported for e in entries)

    def test_export_batch_filters_by_since(self):
        buf = RingBufferLog()
        buf.append("INFO", "old")
        cutoff = datetime.now(UTC) + timedelta(milliseconds=10)
        import time

        time.sleep(0.02)
        buf.append("INFO", "new")
        batch = buf.export_batch(cutoff)
        assert [e.message for e in batch] == ["new"]


class TestRingBufferStats:
    """统计信息。"""

    def test_stats_empty(self):
        buf = RingBufferLog()
        s = buf.stats()
        assert s["total"] == 0
        assert s["by_level"] == {}
        assert s["earliest"] is None
        assert s["latest"] is None
        assert s["exported"] == 0
        assert s["unexported"] == 0

    def test_stats_populated(self):
        buf = RingBufferLog(max_entries=1000, retention_hours=12)
        buf.append("INFO", "a")
        buf.append("ERROR", "b")
        buf.append("INFO", "c")
        s = buf.stats()
        assert s["total"] == 3
        assert s["by_level"] == {"INFO": 2, "ERROR": 1}
        assert s["earliest"] is not None
        assert s["latest"] is not None
        assert s["earliest"] <= s["latest"]
        assert s["exported"] == 0
        assert s["unexported"] == 3
        assert s["max_entries"] == 1000
        assert s["retention_hours"] == 12

    def test_stats_exported_counts(self):
        buf = RingBufferLog()
        buf.append("INFO", "a")
        buf.append("INFO", "b")
        entries = buf.query(limit=10)
        buf.mark_exported([entries[0]])
        s = buf.stats()
        assert s["exported"] == 1
        assert s["unexported"] == 1


# =============================================================================
# RingBufferLog — 线程安全
# =============================================================================


class TestRingBufferThreadSafety:
    """多线程并发追加/查询。"""

    def test_concurrent_append(self):
        buf = RingBufferLog(max_entries=10000)
        n_threads = 10
        per_thread = 100

        def worker():
            for i in range(per_thread):
                buf.append("INFO", f"t{i}")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(buf) == n_threads * per_thread

    def test_concurrent_append_and_query(self):
        buf = RingBufferLog(max_entries=5000)
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(200):
                    buf.append("INFO", f"m{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    buf.query(limit=50)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(buf) == 5 * 200