"""tests/test_ota_enhanced.py：OTA 下载/校验/A-B 双槽/自动回滚 + 三层版本 + 综合探针。

用 tmp_path 模拟 models/ 结构，本地 ThreadingHTTPServer 做真实分块下载，
monkeypatch 替换探针避免网络依赖。不修改真实 models/ 目录。
"""

from __future__ import annotations

import hashlib
import http.server
import socketserver
import threading
from functools import partial
from pathlib import Path

import httpx
import pytest

from mgmt_agent.ota import (
    OtaState,
    deploy,
    download,
    rollback,
    switch_slot,
)
from mgmt_agent.probe import ProbeResult, comprehensive_probe, probe_edge
from mgmt_agent.version import VersionManager, VersionSlots
from shared.errors import E_MODEL_HASH_MISMATCH, E_REQUIRES_FIRMWARE, EdgeError
from shared.schemas import ModelManifest


# --------------------------------------------------------------------------- #
# 辅助构造
# --------------------------------------------------------------------------- #

def _make_manifest(
    url: str,
    content: bytes,
    *,
    model_id: str = "anomaly-det",
    version: str = "0.2.0",
    requires_firmware: str = "0.1.0",
    runtime: str = "onnxruntime",
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        version=version,
        artifact_url=url,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        quant="int8",
        runtime=runtime,
        target_hardware=["x86"],
        requires_firmware=requires_firmware,
        preproc_ver="v1",
    )


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """静音的 SimpleHTTPRequestHandler，避免测试输出被日志淹没。"""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _FakeResp:
    """httpx.get 返回的假响应。"""

    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


@pytest.fixture()
def file_server(tmp_path: Path):
    """在 tmp_path 上启动本地线程化 HTTP 文件服务器，返回 (base_url, root)。"""
    handler = partial(_QuietHandler, directory=str(tmp_path))
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}", tmp_path
        httpd.shutdown()
        t.join(timeout=2)


# --------------------------------------------------------------------------- #
# download：分块下载 + sha256 校验
# --------------------------------------------------------------------------- #

def test_download_verifies_sha256_and_writes_file(file_server):
    base, root = file_server
    content = b"x" * 1024
    (root / "model.onnx").write_bytes(content)
    dest = root / "downloaded.onnx"
    digest = hashlib.sha256(content).hexdigest()

    out = download(f"{base}/model.onnx", dest, digest)

    assert out == dest
    assert dest.read_bytes() == content


def test_download_large_file_chunked(file_server):
    """大于单块的内容也能正确分块下载并校验。"""
    base, root = file_server
    content = b"A" * (10 * 1024 * 1024 + 7)  # ~10MB，跨多个 4MB 块
    (root / "big.onnx").write_bytes(content)
    dest = root / "big_dl.onnx"
    digest = hashlib.sha256(content).hexdigest()

    download(f"{base}/big.onnx", dest, digest, chunk_size=1024 * 1024)

    assert dest.read_bytes() == content


def test_download_hash_mismatch_raises(file_server):
    base, root = file_server
    (root / "model.onnx").write_bytes(b"real content")
    dest = root / "downloaded.onnx"

    with pytest.raises(EdgeError) as exc:
        download(f"{base}/model.onnx", dest, "0" * 64)

    assert exc.value.code == E_MODEL_HASH_MISMATCH


def test_download_http_error_propagates(file_server):
    base, root = file_server  # noqa: F841 — root 未用，仅起服务器
    dest = root / "x.onnx"
    with pytest.raises(httpx.HTTPStatusError):
        download(f"{base}/nonexistent.onnx", dest, "0" * 64)


# --------------------------------------------------------------------------- #
# switch_slot：A/B 双槽符号链接切换
# --------------------------------------------------------------------------- #

def test_switch_slot_activates_current(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    (models / "anomaly" / "0.2.0").mkdir(parents=True)

    switch_slot(models, "anomaly", "0.1.0")
    slots = VersionSlots(models)
    assert slots.current_version("anomaly") == "0.1.0"

    switch_slot(models, "anomaly", "0.2.0")
    assert slots.current_version("anomaly") == "0.2.0"


def test_switch_slot_records_history(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    (models / "anomaly" / "0.2.0").mkdir(parents=True)

    switch_slot(models, "anomaly", "0.1.0")
    switch_slot(models, "anomaly", "0.2.0")

    history_file = models / "anomaly" / "history.json"
    assert history_file.exists()
    import json

    history = json.loads(history_file.read_text("utf-8"))
    assert history == ["0.1.0", "0.2.0"]


def test_switch_slot_idempotent_on_same_version(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")
    switch_slot(models, "anomaly", "0.1.0")  # 重复激活不应重复记录

    import json

    history = json.loads((models / "anomaly" / "history.json").read_text("utf-8"))
    assert history == ["0.1.0"]


# --------------------------------------------------------------------------- #
# rollback：自动回滚到上一版本
# --------------------------------------------------------------------------- #

def test_rollback_to_previous_version(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    (models / "anomaly" / "0.2.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")
    switch_slot(models, "anomaly", "0.2.0")

    prev = rollback(models, "anomaly")

    assert prev == "0.1.0"
    assert VersionSlots(models).current_version("anomaly") == "0.1.0"


def test_rollback_removes_failed_version_from_history(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    (models / "anomaly" / "0.2.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")
    switch_slot(models, "anomaly", "0.2.0")
    rollback(models, "anomaly")

    import json

    history = json.loads((models / "anomaly" / "history.json").read_text("utf-8"))
    assert history == ["0.1.0"]


def test_rollback_no_history_returns_none(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    assert rollback(models, "anomaly") is None


def test_rollback_single_version_returns_none(tmp_path):
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")
    assert rollback(models, "anomaly") is None


def test_rollback_preserves_old_version_dir(tmp_path):
    """回滚后旧版本目录仍存在（用于再次切换/回滚）。"""
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    (models / "anomaly" / "0.2.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")
    switch_slot(models, "anomaly", "0.2.0")
    rollback(models, "anomaly")

    assert (models / "anomaly" / "0.2.0").exists()
    assert (models / "anomaly" / "0.1.0").exists()


# --------------------------------------------------------------------------- #
# deploy：完整部署流程
# --------------------------------------------------------------------------- #

def test_deploy_success(file_server, tmp_path, monkeypatch):
    base, root = file_server
    content = b"good model bytes"
    (root / "model.onnx").write_bytes(content)
    manifest = _make_manifest(f"{base}/model.onnx", content, version="0.3.0")
    models = tmp_path / "models"

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(True, "ok", {}),
    )

    state = deploy(manifest, models, firmware_version="0.1.0", health_url="http://x")

    assert state == OtaState.succeeded
    assert VersionSlots(models).current_version("anomaly") == "0.3.0"
    assert (models / "anomaly" / "0.3.0" / "model.onnx").exists()
    assert (models / "anomaly" / "0.3.0" / "manifest.json").exists()


def test_deploy_probe_fail_auto_rollback(file_server, tmp_path, monkeypatch):
    base, root = file_server
    content = b"new model"
    (root / "model.onnx").write_bytes(content)
    manifest = _make_manifest(f"{base}/model.onnx", content, version="0.2.0")
    models = tmp_path / "models"
    # 预置旧版本并激活
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(False, "health_false", {}),
    )

    state = deploy(
        manifest, models, firmware_version="0.1.0", health_url="http://x", previous="0.1.0"
    )

    assert state == OtaState.rolled_back
    assert VersionSlots(models).current_version("anomaly") == "0.1.0"


def test_deploy_probe_fail_rollback_via_history(file_server, tmp_path, monkeypatch):
    """未显式传 previous 时，deploy 通过历史记录自动回滚。"""
    base, root = file_server
    content = b"new model"
    (root / "model.onnx").write_bytes(content)
    manifest = _make_manifest(f"{base}/model.onnx", content, version="0.2.0")
    models = tmp_path / "models"
    (models / "anomaly" / "0.1.0").mkdir(parents=True)
    switch_slot(models, "anomaly", "0.1.0")

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(False, "failed: memory", {}),
    )

    state = deploy(manifest, models, firmware_version="0.1.0", health_url="http://x")

    assert state == OtaState.rolled_back
    assert VersionSlots(models).current_version("anomaly") == "0.1.0"


def test_deploy_firmware_incompatible_raises(file_server, tmp_path):
    base, root = file_server
    content = b"x"
    (root / "model.onnx").write_bytes(content)
    manifest = _make_manifest(f"{base}/model.onnx", content, requires_firmware="0.9.0")

    with pytest.raises(EdgeError) as exc:
        deploy(manifest, tmp_path / "models", firmware_version="0.1.0", health_url="http://x")

    assert exc.value.code == E_REQUIRES_FIRMWARE


def test_deploy_hash_mismatch_raises(file_server, tmp_path, monkeypatch):
    base, root = file_server
    (root / "model.onnx").write_bytes(b"actual content")
    manifest = _make_manifest(f"{base}/model.onnx", b"different content", version="0.2.0")

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(True, "ok", {}),
    )

    with pytest.raises(EdgeError) as exc:
        deploy(manifest, tmp_path / "models", firmware_version="0.1.0", health_url="http://x")

    assert exc.value.code == E_MODEL_HASH_MISMATCH


def test_deploy_intent_layer(file_server, tmp_path, monkeypatch):
    """非 anomaly 模型应落到 intent 层。"""
    base, root = file_server
    content = b"intent model"
    (root / "llm.bin").write_bytes(content)
    manifest = _make_manifest(
        f"{base}/llm.bin", content, model_id="intent-llm", version="1.0.0", runtime="llm"
    )
    models = tmp_path / "models"

    monkeypatch.setattr(
        "mgmt_agent.ota.comprehensive_probe",
        lambda url, **kw: ProbeResult(True, "ok", {}),
    )

    state = deploy(manifest, models, firmware_version="0.1.0", health_url="http://x")

    assert state == OtaState.succeeded
    assert VersionSlots(models).current_version("intent") == "1.0.0"


# --------------------------------------------------------------------------- #
# VersionManager：三层版本解耦
# --------------------------------------------------------------------------- #

def test_version_manager_three_layers(tmp_path):
    vm = VersionManager(tmp_path / "models")
    vm.set_firmware_version("0.1.0")
    vm.set_config_version("v0001")
    vm.update_model_version("0.9.1")

    v = vm.get_versions()
    assert v == {
        "firmware_version": "0.1.0",
        "model_version": "0.9.1",
        "config_version": "v0001",
    }


def test_version_manager_default_unknown(tmp_path):
    vm = VersionManager(tmp_path / "models")
    v = vm.get_versions()
    assert v == {
        "firmware_version": "unknown",
        "model_version": "unknown",
        "config_version": "unknown",
    }


def test_version_manager_check_compatibility_match(tmp_path):
    vm = VersionManager(tmp_path / "models")
    vm.set_firmware_version("0.1.0")
    manifest = _make_manifest("http://x", b"x", requires_firmware="0.1.0")
    assert vm.check_compatibility(manifest) is True


def test_version_manager_check_compatibility_mismatch(tmp_path):
    vm = VersionManager(tmp_path / "models")
    vm.set_firmware_version("0.1.0")
    manifest = _make_manifest("http://x", b"x", requires_firmware="0.2.0")
    assert vm.check_compatibility(manifest) is False


def test_version_manager_update_model_preserves_other_layers(tmp_path):
    vm = VersionManager(tmp_path / "models")
    vm.set_firmware_version("0.1.0")
    vm.set_config_version("v0001")
    vm.update_model_version("0.9.1")
    vm.update_model_version("0.9.2")

    v = vm.get_versions()
    assert v["model_version"] == "0.9.2"
    assert v["firmware_version"] == "0.1.0"
    assert v["config_version"] == "v0001"


def test_version_manager_persists_across_instances(tmp_path):
    root = tmp_path / "models"
    vm1 = VersionManager(root)
    vm1.set_firmware_version("0.1.0")
    vm1.update_model_version("0.9.1")

    vm2 = VersionManager(root)
    v = vm2.get_versions()
    assert v["firmware_version"] == "0.1.0"
    assert v["model_version"] == "0.9.1"


# --------------------------------------------------------------------------- #
# comprehensive_probe：综合健康探针
# --------------------------------------------------------------------------- #

def test_comprehensive_probe_all_ok(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": True,
                "mem_used_mb": 1024,
                "disk_free_mb": 10240,
                "cpu_temp_c": 50.0,
                "input_links": {"a": True, "b": True},
            }
        ),
    )
    r = comprehensive_probe("http://x")
    assert r.probe_ok is True
    assert r.inference_ok is True
    assert r.mem_ok is True
    assert r.disk_ok is True
    assert r.temp_ok is True
    assert r.input_links_ok is True
    assert r.reason == "ok"


def test_comprehensive_probe_mem_exceeds(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": True,
                "mem_used_mb": 99999,
                "disk_free_mb": 10240,
                "cpu_temp_c": 50.0,
                "input_links": {},
            }
        ),
    )
    r = comprehensive_probe("http://x", mem_limit_mb=4096)
    assert r.probe_ok is False
    assert r.mem_ok is False
    assert "memory" in r.reason


def test_comprehensive_probe_disk_low(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": True,
                "mem_used_mb": 1024,
                "disk_free_mb": 10,
                "cpu_temp_c": 50.0,
                "input_links": {},
            }
        ),
    )
    r = comprehensive_probe("http://x", disk_min_mb=512)
    assert r.probe_ok is False
    assert r.disk_ok is False
    assert "disk" in r.reason


def test_comprehensive_probe_temp_high(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": True,
                "mem_used_mb": 1024,
                "disk_free_mb": 10240,
                "cpu_temp_c": 99.0,
                "input_links": {},
            }
        ),
    )
    r = comprehensive_probe("http://x", temp_max_c=85.0)
    assert r.probe_ok is False
    assert r.temp_ok is False
    assert "temperature" in r.reason


def test_comprehensive_probe_temp_none_passes(monkeypatch):
    """温度未知（None）应放行。"""
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {"ok": True, "mem_used_mb": 1024, "disk_free_mb": 10240, "input_links": {}}
        ),
    )
    r = comprehensive_probe("http://x")
    assert r.temp_ok is True
    assert r.probe_ok is True


def test_comprehensive_probe_input_link_down(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": True,
                "mem_used_mb": 1024,
                "disk_free_mb": 10240,
                "cpu_temp_c": 50.0,
                "input_links": {"a": True, "b": False},
            }
        ),
    )
    r = comprehensive_probe("http://x")
    assert r.probe_ok is False
    assert r.input_links_ok is False
    assert "input_links" in r.reason


def test_comprehensive_probe_inference_down(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "ok": False,
                "mem_used_mb": 1024,
                "disk_free_mb": 10240,
                "cpu_temp_c": 50.0,
                "input_links": {},
            }
        ),
    )
    r = comprehensive_probe("http://x")
    assert r.probe_ok is False
    assert r.inference_ok is False


def test_comprehensive_probe_request_failure(monkeypatch):
    def raise_err(url, **kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "get", raise_err)
    r = comprehensive_probe("http://x")
    assert r.probe_ok is False
    assert r.inference_ok is False
    assert r.mem_ok is False
    assert r.disk_ok is False
    assert r.checks == {"health": False}


def test_comprehensive_probe_uses_probe_ok_field(monkeypatch):
    """EdgeHealth 用 probe_ok 字段时也应识别为推理可用。"""
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResp(
            {
                "probe_ok": True,
                "mem_used_mb": 1024,
                "disk_free_mb": 10240,
                "cpu_temp_c": 50.0,
                "input_links": {},
            }
        ),
    )
    r = comprehensive_probe("http://x")
    assert r.inference_ok is True


# --------------------------------------------------------------------------- #
# probe_edge 兼容性回归
# --------------------------------------------------------------------------- #

def test_probe_edge_still_works(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResp({"ok": True}))
    r = probe_edge("http://x")
    assert r.probe_ok is True
    assert r.reason == "ok"
    # 单项探针不填充增强字段
    assert r.inference_ok is None
    assert r.checks == {}


def test_probe_edge_failure_still_works(monkeypatch):
    def raise_err(url, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", raise_err)
    r = probe_edge("http://x")
    assert r.probe_ok is False