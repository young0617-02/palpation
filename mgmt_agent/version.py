from __future__ import annotations

import json
import shutil
from pathlib import Path

from shared.schemas import ModelManifest


class VersionSlots:
    """Three-layer versions. Windows uses current.txt when symlink is unavailable."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def slot_dir(self, layer: str, version: str) -> Path:
        p = self.root / layer / version
        p.mkdir(parents=True, exist_ok=True)
        return p

    def current_version(self, layer: str) -> str | None:
        pointer = self.root / layer / "current.txt"
        if pointer.exists():
            return pointer.read_text(encoding="utf-8").strip()
        link = self.root / layer / "current"
        if link.is_symlink():
            return link.resolve().name
        return None

    def activate(self, layer: str, version: str) -> None:
        target = self.slot_dir(layer, version)
        link = self.root / layer / "current"
        pointer = self.root / layer / "current.txt"
        pointer.write_text(version, encoding="utf-8")
        try:
            if link.is_symlink() or link.is_file():
                link.unlink()
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
        except OSError:
            marker = self.root / layer / "current.json"
            marker.write_text(
                json.dumps({"version": version, "path": str(target)}), encoding="utf-8"
            )

    def rollback(self, layer: str, version: str) -> None:
        self.activate(layer, version)

    def copy_into(self, layer: str, version: str, src: Path) -> Path:
        dest = self.slot_dir(layer, version)
        if src.is_file():
            shutil.copy2(src, dest / src.name)
        else:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        return dest


class VersionManager:
    """三层版本解耦：firmware_version / model_version / config_version。

    元数据持久化于 root/versions.json；槽位管理委托给 VersionSlots，
    从而与 A/B 双槽切换复用同一目录结构。固件层为硬约束（check_compatibility），
    模型层与配置层可独立滚动。
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.slots = VersionSlots(root)
        self._meta_path = root / "versions.json"

    def _load(self) -> dict[str, str]:
        if self._meta_path.exists():
            try:
                data = json.loads(self._meta_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {
                        "firmware": str(data.get("firmware", "unknown")),
                        "model": str(data.get("model", "unknown")),
                        "config": str(data.get("config", "unknown")),
                    }
            except json.JSONDecodeError:
                pass
        return {"firmware": "unknown", "model": "unknown", "config": "unknown"}

    def _save(self, data: dict[str, str]) -> None:
        self._meta_path.write_text(json.dumps(data), encoding="utf-8")

    def get_versions(self) -> dict[str, str]:
        """返回当前三层版本：firmware_version / model_version / config_version。"""
        data = self._load()
        return {
            "firmware_version": data["firmware"],
            "model_version": data["model"],
            "config_version": data["config"],
        }

    def check_compatibility(self, manifest: ModelManifest) -> bool:
        """检查 requires_firmware 硬约束是否与当前固件版本匹配。"""
        data = self._load()
        return manifest.requires_firmware == data["firmware"]

    def update_model_version(self, new_version: str) -> None:
        """更新模型版本记录（配置/固件层不变）。"""
        data = self._load()
        data["model"] = new_version
        self._save(data)

    def set_firmware_version(self, version: str) -> None:
        data = self._load()
        data["firmware"] = version
        self._save(data)

    def set_config_version(self, version: str) -> None:
        data = self._load()
        data["config"] = version
        self._save(data)
