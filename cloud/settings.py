from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class CloudSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = f"sqlite:///{(ROOT / 'data' / 'cloud.db').as_posix()}"
    cloud_mqtt_host: str = "127.0.0.1"
    cloud_mqtt_port: int = 1883
    cloud_http_port: int = 8000
    artifact_dir: Path = ROOT / "data" / "artifacts"


cloud_settings = CloudSettings()
