from __future__ import annotations

import hashlib
import json
import logging
import time
from enum import StrEnum
from pathlib import Path

import httpx

from mgmt_agent.probe import comprehensive_probe, probe_edge
from mgmt_agent.version import VersionSlots
from shared.errors import E_MODEL_HASH_MISMATCH, E_REQUIRES_FIRMWARE, EdgeError
from shared.schemas import ModelManifest

log = logging.getLogger(__name__)


class OtaState(StrEnum):
    pending = "pending"
    downloading = "downloading"
    verifying = "verifying"
    staged = "staged"
    activating = "activating"
    probing = "probing"
    observing = "observing"
    succeeded = "succeeded"
    rolling_back = "rolling_back"
    rolled_back = "rolled_back"
    frozen = "frozen"


class OtaEngine:
    def __init__(self, slots: VersionSlots, firmware_version: str, health_url: str):
        self.slots = slots
        self.firmware_version = firmware_version
        self.health_url = health_url
        self.state = OtaState.pending
        self.frozen_versions: set[str] = set()
        self.last_error: str | None = None

    def apply(self, manifest: ModelManifest, previous: str | None) -> OtaState:
        key = f"{manifest.model_id}:{manifest.version}"
        if key in self.frozen_versions:
            self.state = OtaState.frozen
            return self.state
        if manifest.requires_firmware and manifest.requires_firmware != self.firmware_version:
            raise EdgeError(E_REQUIRES_FIRMWARE, f"need {manifest.requires_firmware}")
        layer = (
            "anomaly"
            if "anom" in manifest.model_id or manifest.runtime == "onnxruntime"
            else "intent"
        )
        self.state = OtaState.downloading
        dest_dir = self.slots.slot_dir(layer, manifest.version)
        artifact = self._download(manifest, dest_dir)
        self.state = OtaState.verifying
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != manifest.sha256:
            self.state = OtaState.rolling_back
            raise EdgeError(E_MODEL_HASH_MISMATCH, digest)
        (dest_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        self.state = OtaState.staged
        self.state = OtaState.activating
        self.slots.activate(layer, manifest.version)
        self.state = OtaState.probing
        result = probe_edge(self.health_url)
        if not result.probe_ok:
            self.state = OtaState.rolling_back
            if previous:
                self.slots.rollback(layer, previous)
            self.frozen_versions.add(key)
            self.state = OtaState.rolled_back
            self.last_error = result.reason
            return self.state
        self.state = OtaState.observing
        time.sleep(0.05)
        self.state = OtaState.succeeded
        return self.state

    def _download(self, manifest: ModelManifest, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = Path(httpx.URL(manifest.artifact_url).path).name or "artifact.bin"
        dest = dest_dir / name
        with (
            httpx.Client(timeout=30.0) as client,
            client.stream("GET", manifest.artifact_url) as resp,
        ):
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(4 * 1024 * 1024):
                    f.write(chunk)
        return dest


# --------------------------------------------------------------------------- #
# 增强模块级 API：分块下载 + sha256 校验 / A-B 双槽切换 / 自动回滚 / 完整部署
# --------------------------------------------------------------------------- #

_HISTORY_FILE = "history.json"


def _infer_layer(manifest: ModelManifest) -> str:
    """推断模型所属层：anomaly（异常检测）或 intent（意图理解）。"""
    return (
        "anomaly" if "anom" in manifest.model_id or manifest.runtime == "onnxruntime" else "intent"
    )


def _history_path(models_dir: Path, model_type: str) -> Path:
    return models_dir / model_type / _HISTORY_FILE


def _load_history(models_dir: Path, model_type: str) -> list[str]:
    p = _history_path(models_dir, model_type)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(v) for v in data]
        except json.JSONDecodeError:
            return []
    return []


def _save_history(models_dir: Path, model_type: str, history: list[str]) -> None:
    p = _history_path(models_dir, model_type)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history), encoding="utf-8")


def download(
    url: str,
    dest: Path,
    sha256_expected: str,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    timeout: float = 30.0,
) -> Path:
    """分块下载 url 到 dest，边下边算 sha256；校验失败抛 EdgeError(E_MODEL_HASH_MISMATCH)。

    分块写入并增量更新摘要，避免大模型整块读入内存。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with (
        httpx.Client(timeout=timeout) as client,
        client.stream("GET", url) as resp,
    ):
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size):
                f.write(chunk)
                h.update(chunk)
    digest = h.hexdigest()
    if digest != sha256_expected:
        raise EdgeError(E_MODEL_HASH_MISMATCH, f"expected={sha256_expected} got={digest}")
    return dest


def switch_slot(models_dir: Path, model_type: str, new_version: str) -> Path:
    """A/B 双槽符号链接切换：current → new_version，并记录激活历史。

    旧版本目录不删除，保留在槽位中以供回滚。返回新激活的槽位目录。
    """
    slots = VersionSlots(models_dir)
    target = slots.slot_dir(model_type, new_version)
    history = _load_history(models_dir, model_type)
    if not history or history[-1] != new_version:
        history.append(new_version)
    _save_history(models_dir, model_type, history)
    slots.activate(model_type, new_version)
    return target


def rollback(models_dir: Path, model_type: str) -> str | None:
    """回滚到上一版本（旧版本不删除）。返回回滚到的版本号；无历史可回滚时返回 None。"""
    history = _load_history(models_dir, model_type)
    if len(history) < 2:
        return None
    failed = history.pop()
    prev = history[-1]
    slots = VersionSlots(models_dir)
    slots.activate(model_type, prev)
    _save_history(models_dir, model_type, history)
    log.info("rollback %s: %s -> %s (failed=%s removed)", model_type, failed, prev, failed)
    return prev


def deploy(
    manifest: ModelManifest,
    models_dir: Path,
    *,
    firmware_version: str,
    health_url: str,
    previous: str | None = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> OtaState:
    """完整部署流程：兼容性检查 → 分块下载+sha256校验 → A/B 切换 → 综合探针 → 自动回滚。

    - 固件不兼容：抛 EdgeError(E_REQUIRES_FIRMWARE)
    - sha256 不匹配：抛 EdgeError(E_MODEL_HASH_MISMATCH)
    - 探针失败：自动回滚到上一版本，返回 OtaState.rolled_back
    - 全程成功：返回 OtaState.succeeded
    旧版本不删除，保留以供后续回滚。
    """
    layer = _infer_layer(manifest)
    # 1. 固件硬约束
    if manifest.requires_firmware and manifest.requires_firmware != firmware_version:
        raise EdgeError(E_REQUIRES_FIRMWARE, f"need {manifest.requires_firmware}")
    # 2. 分块下载 + sha256 校验
    name = Path(httpx.URL(manifest.artifact_url).path).name or "artifact.bin"
    dest_dir = models_dir / layer / manifest.version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    download(manifest.artifact_url, dest, manifest.sha256, chunk_size=chunk_size)
    (dest_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    # 3. A/B 双槽切换（记录历史）
    switch_slot(models_dir, layer, manifest.version)
    # 4. 综合探针
    result = comprehensive_probe(health_url)
    if not result.probe_ok:
        # 5. 自动回滚：优先用历史回滚，previous 作为兜底
        rolled = rollback(models_dir, layer)
        if rolled is None and previous:
            VersionSlots(models_dir).rollback(layer, previous)
        log.warning(
            "deploy %s %s probe failed (%s), rolled back to %s",
            manifest.model_id,
            manifest.version,
            result.reason,
            rolled or previous,
        )
        return OtaState.rolled_back
    return OtaState.succeeded
