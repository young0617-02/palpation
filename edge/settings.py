from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gw_id: str = "gw-sim-001"
    firmware_version: str = "0.1.0"
    config_path: Path = ROOT / "config" / "current.yaml"
    data_dir: Path = ROOT / "data" / "edge"
    models_dir: Path = ROOT / "models"
    cloud_mqtt_host: str = "127.0.0.1"
    cloud_mqtt_port: int = 1883
    cloud_api: str = "http://127.0.0.1:8000"
    edge_api_port: int = 9000
    hmac_salt: str = "sim-only-replace-me"
    edge_llm_model: str = ""
    edge_llm_ctx: int = 2048
    edge_llm_maxtok: int = 32
    modbus_host: str = "127.0.0.1"
    modbus_port: int = 5020
    opcua_endpoint: str = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
    log_level: str = "INFO"
    # --- T4 知识检索（bge-small-zh-v1.5 ONNX，512 维；模型缺失时降级 hash-embed） ---
    knowledge_enabled: bool = True
    knowledge_model_dir: str = ""  # 空则回落到 models_dir / "knowledge" / "current"
    knowledge_dim: int = 512
    knowledge_degrade_to_hash: bool = True
    knowledge_num_threads: int = 2


settings = Settings()
