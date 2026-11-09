"""mgmt_agent/ota.py 单元测试：OtaEngine/OtaState 状态机。

用 tmp_path 做文件操作，monkeypatch 替换 _download 与 probe_edge 避免网络依赖。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mgmt_agent.ota import OtaEngine, OtaState
from mgmt_agent.probe import ProbeResult
from mgmt_agent.version import VersionSlots
from shared.errors import E_MODEL_HASH_MISMATCH, E_REQUIRES_FIRMWARE, EdgeError
from shared.schemas import ModelManifest


# --------------------------------------------------------------------------- #
# 辅助构造
# --------------------------------------------------------------------------- #

def _make_manifest(
    sha256: str,
    *,
    model_id: str = "anomaly-det",
    version: str = "0.1.0",
    requires_firmware: str = "0.1.0",
    runtime: str = "onnxruntime",
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        version=version,
        artifact_url="https://example.com/model.onnx",
        sha256=sha256,
        size_bytes=100,
        quant="int8",
        runtime=runtime,
        target_hardware=["x86"],
        requires_firmware=requires_firmware,
        preproc_ver="v1",
    )


def _fake_download(content: bytes):
    """返回一个替换 OtaEngine._download 的函数，将 content 写入 dest_dir。"""

    def _download(self, manifest, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "model.onnx"
        dest.write_bytes(content)
        return dest

    return _download


def _probe_ok(*_args, **_kwargs) -> ProbeResult:
    return ProbeResult(True, "ok", {})


def _probe_fail(*_args, **_kwargs) -> ProbeResult:
    return ProbeResult(False, "health_false", {})


@pytest.fixture()
def slots(tmp_path: Path) -> VersionSlots:
    return VersionSlots(tmp_path)


# --------------------------------------------------------------------------- #
# 初始状态
# --------------------------------------------------------------------------- #

def test_initial_state_is_pending(slots):
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    assert eng.state == OtaState.pending
    assert eng.frozen_versions == set()
    assert eng.last_error is None


# --------------------------------------------------------------------------- #
# frozen 分支
# --------------------------------------------------------------------------- #

def test_frozen_version_returns_frozen(slots):
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    manifest = _make_manifest("abc")
    key = f"{manifest.model_id}:{manifest.version}"
    eng.frozen_versions.add(key)
    state = eng.apply(manifest, previous=None)
    assert state == OtaState.frozen


# --------------------------------------------------------------------------- #
# requires_firmware 不匹配
# --------------------------------------------------------------------------- #

def test_requires_firmware_mismatch_raises(slots):
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    manifest = _make_manifest("abc", requires_firmware="0.2.0")
    with pytest.raises(EdgeError) as exc:
        eng.apply(manifest, previous=None)
    assert exc.value.code == E_REQUIRES_FIRMWARE


def test_requires_firmware_match_proceeds(slots, monkeypatch):
    content = b"good"
    digest = hashlib.sha256(content).hexdigest()
    manifest = _make_manifest(digest, version="0.5.0", requires_firmware="0.1.0")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_ok)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    assert eng.apply(manifest, previous=None) == OtaState.succeeded


# --------------------------------------------------------------------------- #
# hash 不匹配 -> rolling_back + EdgeError
# --------------------------------------------------------------------------- #

def test_hash_mismatch_raises_and_rolling_back(slots, monkeypatch):
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    manifest = _make_manifest("0" * 64)  # 期望 sha256 全 0
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(b"wrong content"))
    with pytest.raises(EdgeError) as exc:
        eng.apply(manifest, previous=None)
    assert exc.value.code == E_MODEL_HASH_MISMATCH
    assert eng.state == OtaState.rolling_back


# --------------------------------------------------------------------------- #
# 探测失败 -> rolled_back + frozen + last_error
# --------------------------------------------------------------------------- #

def test_probe_fail_rolls_back_and_freezes(slots, monkeypatch):
    content = b"good model"
    digest = hashlib.sha256(content).hexdigest()
    manifest = _make_manifest(digest, version="0.2.0")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_fail)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    state = eng.apply(manifest, previous="0.1.0")
    assert state == OtaState.rolled_back
    key = f"{manifest.model_id}:{manifest.version}"
    assert key in eng.frozen_versions
    assert eng.last_error == "health_false"


def test_probe_fail_without_previous_still_rolls_back(slots, monkeypatch):
    content = b"good model"
    digest = hashlib.sha256(content).hexdigest()
    manifest = _make_manifest(digest, version="0.2.0")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_fail)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    state = eng.apply(manifest, previous=None)
    assert state == OtaState.rolled_back
    assert eng.last_error == "health_false"


# --------------------------------------------------------------------------- #
# 正常全流程 -> succeeded
# --------------------------------------------------------------------------- #

def test_success_full_flow(slots, monkeypatch):
    content = b"good model"
    digest = hashlib.sha256(content).hexdigest()
    manifest = _make_manifest(digest, version="0.3.0")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_ok)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    state = eng.apply(manifest, previous="0.2.0")
    assert state == OtaState.succeeded
    assert eng.last_error is None


# --------------------------------------------------------------------------- #
# layer 选择：anomaly vs intent
# --------------------------------------------------------------------------- #

def test_layer_selection_anomaly_by_model_id(slots, monkeypatch):
    content = b"anom model"
    digest = hashlib.sha256(content).hexdigest()
    # model_id 含 "anom" -> anomaly layer
    manifest = _make_manifest(digest, model_id="anomaly-det", version="0.4.0",
                              runtime="llm")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_ok)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    assert eng.apply(manifest, previous=None) == OtaState.succeeded
    assert slots.current_version("anomaly") == "0.4.0"


def test_layer_selection_anomaly_by_runtime(slots, monkeypatch):
    content = b"onnx model"
    digest = hashlib.sha256(content).hexdigest()
    # runtime == "onnxruntime" -> anomaly layer
    manifest = _make_manifest(digest, model_id="xyz", version="0.4.0",
                              runtime="onnxruntime")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_ok)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    assert eng.apply(manifest, previous=None) == OtaState.succeeded
    assert slots.current_version("anomaly") == "0.4.0"


def test_layer_selection_intent(slots, monkeypatch):
    content = b"intent model"
    digest = hashlib.sha256(content).hexdigest()
    # model_id 不含 "anom" 且 runtime != "onnxruntime" -> intent layer
    manifest = _make_manifest(digest, model_id="intent-llm", version="0.4.0",
                              runtime="llm")
    monkeypatch.setattr(OtaEngine, "_download", _fake_download(content))
    monkeypatch.setattr("mgmt_agent.ota.probe_edge", _probe_ok)
    eng = OtaEngine(slots, firmware_version="0.1.0", health_url="http://x")
    assert eng.apply(manifest, previous=None) == OtaState.succeeded
    assert slots.current_version("intent") == "0.4.0"