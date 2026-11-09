from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from shared.errors import E_DISK_FULL, E_MODEL_HASH_MISMATCH, EdgeError
from shared.schemas import ModelManifest


class DataPlane:
    def __init__(self, cloud_api: str, timeout_s: float = 10.0):
        self.cloud_api = cloud_api.rstrip("/")
        self.timeout_s = timeout_s

    def disk_free_mb(self, path: Path) -> int:
        import shutil

        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return int(usage.free / (1024 * 1024))

    def download_artifact(
        self, manifest: ModelManifest, dest_dir: Path, chunk_size: int = 4 * 1024 * 1024
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        need_mb = (manifest.size_bytes * 2) / (1024 * 1024) + 500
        if self.disk_free_mb(dest_dir) < need_mb:
            raise EdgeError(E_DISK_FULL, "insufficient disk for model download")
        dest = dest_dir / Path(manifest.artifact_url).name
        hasher = hashlib.sha256()
        downloaded = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        with (
            httpx.Client(timeout=self.timeout_s) as client,
            client.stream("GET", manifest.artifact_url, headers=headers) as resp,
        ):
            resp.raise_for_status()
            mode = "ab" if downloaded and resp.status_code == 206 else "wb"
            if mode == "wb":
                downloaded = 0
                hasher = hashlib.sha256()
            with dest.open(mode) as f:
                if mode == "ab":
                    hasher.update(dest.read_bytes())
                for chunk in resp.iter_bytes(chunk_size):
                    f.write(chunk)
                    hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != manifest.sha256:
            dest.unlink(missing_ok=True)
            raise EdgeError(E_MODEL_HASH_MISMATCH, f"got {digest}")
        return dest
