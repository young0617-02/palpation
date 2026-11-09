"""cloud/settings.py 单元测试：pydantic-settings 配置默认值与环境变量覆盖。"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloud.settings import ROOT, CloudSettings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """清理可能影响测试的环境变量，确保默认值用例不受外部环境干扰。"""
    for key in (
        "DATABASE_URL",
        "CLOUD_MQTT_HOST",
        "CLOUD_MQTT_PORT",
        "CLOUD_HTTP_PORT",
        "ARTIFACT_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_values():
    s = CloudSettings()
    assert s.cloud_mqtt_host == "127.0.0.1"
    assert s.cloud_mqtt_port == 1883
    assert s.cloud_http_port == 8000
    assert "cloud.db" in s.database_url
    assert s.artifact_dir == ROOT / "data" / "artifacts"


def test_database_url_default_is_sqlite():
    s = CloudSettings()
    assert s.database_url.startswith("sqlite:///")


def test_env_override_database_and_mqtt(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("CLOUD_MQTT_HOST", "mqtt.example.com")
    monkeypatch.setenv("CLOUD_MQTT_PORT", "8883")
    s = CloudSettings()
    assert s.database_url == "sqlite:///:memory:"
    assert s.cloud_mqtt_host == "mqtt.example.com"
    assert s.cloud_mqtt_port == 8883


def test_env_override_http_port(monkeypatch):
    monkeypatch.setenv("CLOUD_HTTP_PORT", "9000")
    s = CloudSettings()
    assert s.cloud_http_port == 9000


def test_env_override_artifact_dir(monkeypatch):
    target = Path("/tmp/palpation_artifacts_test")
    monkeypatch.setenv("ARTIFACT_DIR", str(target))
    s = CloudSettings()
    assert s.artifact_dir == target


def test_env_override_takes_precedence_over_defaults(monkeypatch):
    """环境变量覆盖默认值，且可反复实例化得到一致结果。"""
    monkeypatch.setenv("CLOUD_MQTT_PORT", "1884")
    a = CloudSettings()
    b = CloudSettings()
    assert a.cloud_mqtt_port == 1884
    assert b.cloud_mqtt_port == 1884