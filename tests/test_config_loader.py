"""YAML 配置加载单元测试。

覆盖正常加载 config/current.yaml、空文件/null 归一为 {}、
文件不存在抛异常、缺字段、嵌套结构。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from edge.config_loader import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_load_current_yaml():
    cfg = load_yaml(ROOT / "config" / "current.yaml")
    assert cfg["gw_id"] == "gw-sim-001"
    assert cfg["config_version"] == "v0001"
    assert cfg["sampling"]["vibration_window"] == 32
    assert cfg["degrade"]["mem_warn_mb"] == 3072


def test_load_returns_dict():
    cfg = load_yaml(ROOT / "config" / "current.yaml")
    assert isinstance(cfg, dict)


def test_empty_file_returns_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml(p) == {}


def test_null_content_returns_empty_dict(tmp_path):
    p = tmp_path / "null.yaml"
    p.write_text("null", encoding="utf-8")
    assert load_yaml(p) == {}


def test_missing_file_raises(tmp_path):
    p = tmp_path / "nope.yaml"
    with pytest.raises((FileNotFoundError, OSError)):
        load_yaml(p)


def test_missing_field_not_in_config(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("gw_id: gw-1\n", encoding="utf-8")
    cfg = load_yaml(p)
    assert cfg["gw_id"] == "gw-1"
    assert "sampling" not in cfg


def test_load_nested_structure(tmp_path):
    p = tmp_path / "nested.yaml"
    p.write_text("a:\n  b: 1\nlist:\n  - 1\n  - 2\n", encoding="utf-8")
    cfg = load_yaml(p)
    assert cfg["a"]["b"] == 1
    assert cfg["list"] == [1, 2]


def test_load_rules_list(tmp_path):
    p = tmp_path / "rules.yaml"
    p.write_text(
        "rules:\n  - id: r1\n    threshold: 4.5\n  - id: r2\n    threshold: 80.0\n",
        encoding="utf-8",
    )
    cfg = load_yaml(p)
    assert len(cfg["rules"]) == 2
    assert cfg["rules"][0]["id"] == "r1"