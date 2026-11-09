"""规则表热更新测试。

覆盖：
- RuleEngine 增强（add_rule / remove_rule / update_rule / reload / get_version / get_rule_ids / validate_rule）
- RuleLoader（load JSON/YAML / save / has_changed / reload_if_changed / validate_rules / start_watching）
- RuleHotReloader（start / stop / force_reload / reload_count / last_reload_time / 自动重载 / 校验失败）
- 文件变化检测（mtime 变化 / 无变化 / 新文件）
- 边界与异常路径
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml

from edge.stream.rule_loader import RuleHotReloader, RuleLoader
from edge.stream.rules import RuleEngine, RuleHit


# ====================================================================== #
# 辅助函数
# ====================================================================== #


def _make_rule(
    rid: str = "r1",
    metric: str = "temperature",
    op: str = ">",
    threshold: float = 80.0,
    severity: str = "high",
) -> dict:
    """构造一条有效规则。"""
    return {"id": rid, "metric": metric, "op": op, "threshold": threshold, "severity": severity}


def _touch_future(path: Path, offset: float = 2.0) -> None:
    """将文件 mtime 设为未来时间，确保 has_changed 返回 True。"""
    future = time.time() + offset
    os.utime(str(path), (future, future))


VALID_RULES = [
    _make_rule("bearing_rms_high", "vibration_rms", ">", 4.5, "high"),
    _make_rule("temperature_high", "temperature", ">", 80.0, "medium"),
]


# ====================================================================== #
# RuleEngine.validate_rule
# ====================================================================== #


class TestValidateRule:
    def test_valid_rule(self):
        assert RuleEngine.validate_rule(_make_rule()) is True

    def test_valid_rule_with_description(self):
        rule = _make_rule()
        rule["description"] = "温度越限"
        assert RuleEngine.validate_rule(rule) is True

    def test_missing_field(self):
        rule = {"id": "r1", "metric": "temp", "op": ">"}
        assert RuleEngine.validate_rule(rule) is False

    def test_invalid_op(self):
        rule = _make_rule(op="!=")
        assert RuleEngine.validate_rule(rule) is False

    def test_non_numeric_threshold(self):
        rule = _make_rule(threshold="80")  # type: ignore[arg-type]
        assert RuleEngine.validate_rule(rule) is False

    def test_bool_threshold_rejected(self):
        rule = _make_rule(threshold=True)  # type: ignore[arg-type]
        assert RuleEngine.validate_rule(rule) is False

    def test_not_a_dict(self):
        assert RuleEngine.validate_rule("not a dict") is False  # type: ignore[arg-type]

    def test_all_ops_valid(self):
        for op in (">", ">=", "<", "<="):
            assert RuleEngine.validate_rule(_make_rule(op=op)) is True


# ====================================================================== #
# RuleEngine 动态增删改
# ====================================================================== #


class TestRuleEngineDynamic:
    def test_add_rule_success(self):
        eng = RuleEngine([])
        assert eng.add_rule(_make_rule()) is True
        assert len(eng.rules) == 1
        assert eng.get_version() == 1

    def test_add_rule_invalid(self):
        eng = RuleEngine([])
        bad = {"id": "r1", "metric": "temp", "op": "!="}
        assert eng.add_rule(bad) is False
        assert len(eng.rules) == 0
        assert eng.get_version() == 0

    def test_remove_rule_success(self):
        eng = RuleEngine([_make_rule("r1"), _make_rule("r2")])
        assert eng.remove_rule("r1") is True
        assert eng.get_rule_ids() == {"r2"}
        assert eng.get_version() == 1

    def test_remove_rule_not_found(self):
        eng = RuleEngine([_make_rule("r1")])
        assert eng.remove_rule("nope") is False
        assert len(eng.rules) == 1
        assert eng.get_version() == 0

    def test_update_rule_success(self):
        eng = RuleEngine([_make_rule("r1", threshold=80.0)])
        assert eng.update_rule("r1", {"threshold": 90.0}) is True
        assert eng.rules[0]["threshold"] == 90.0
        assert eng.get_version() == 1

    def test_update_rule_not_found(self):
        eng = RuleEngine([_make_rule("r1")])
        assert eng.update_rule("nope", {"threshold": 90.0}) is False
        assert eng.get_version() == 0

    def test_update_rule_invalid_keeps_old(self):
        eng = RuleEngine([_make_rule("r1", threshold=80.0)])
        assert eng.update_rule("r1", {"op": "!="}) is False
        assert eng.rules[0]["op"] == ">"
        assert eng.get_version() == 0

    def test_get_rule_ids(self):
        eng = RuleEngine([_make_rule("a"), _make_rule("b"), _make_rule("c")])
        assert eng.get_rule_ids() == {"a", "b", "c"}

    def test_get_rule_ids_empty(self):
        eng = RuleEngine([])
        assert eng.get_rule_ids() == set()


# ====================================================================== #
# RuleEngine 原子重载与版本管理
# ====================================================================== #


class TestRuleEngineReload:
    def test_get_version_initial(self):
        eng = RuleEngine(VALID_RULES)
        assert eng.get_version() == 0

    def test_reload_increments_version(self):
        eng = RuleEngine([])
        v = eng.reload(VALID_RULES)
        assert v == 1
        assert eng.get_version() == 1
        assert len(eng.rules) == 2

    def test_reload_multiple_times(self):
        eng = RuleEngine([])
        assert eng.reload(VALID_RULES) == 1
        assert eng.reload([_make_rule("r3")]) == 2
        assert eng.get_version() == 2
        assert eng.get_rule_ids() == {"r3"}

    def test_reload_invalid_raises(self):
        eng = RuleEngine(VALID_RULES)
        bad_rules = [_make_rule(), {"id": "bad", "metric": "x", "op": "!="}]
        with pytest.raises(ValueError, match="Invalid rule"):
            eng.reload(bad_rules)
        # 原子性：失败后旧规则不变
        assert len(eng.rules) == 2
        assert eng.get_version() == 0

    def test_reload_empty_list(self):
        eng = RuleEngine(VALID_RULES)
        v = eng.reload([])
        assert v == 1
        assert eng.rules == []

    def test_reload_preserves_evaluate(self):
        eng = RuleEngine([])
        eng.reload([_make_rule("temp_high", "temperature", ">", 80.0)])
        hits = eng.evaluate({"temperature": 100.0})
        assert len(hits) == 1
        assert hits[0].rule_id == "temp_high"


# ====================================================================== #
# RuleLoader — 加载
# ====================================================================== #


class TestRuleLoaderLoad:
    def test_load_json(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        rules = loader.load()
        assert len(rules) == 2
        assert rules[0]["id"] == "bearing_rms_high"

    def test_load_yaml(self, tmp_path):
        f = tmp_path / "rules.yaml"
        f.write_text(yaml.safe_dump(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        rules = loader.load()
        assert len(rules) == 2
        assert rules[1]["id"] == "temperature_high"

    def test_load_yml_extension(self, tmp_path):
        f = tmp_path / "rules.yml"
        f.write_text(yaml.safe_dump(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        rules = loader.load()
        assert len(rules) == 2

    def test_load_wrapped_format(self, tmp_path):
        """支持 {"rules": [...]} 包装格式。"""
        f = tmp_path / "rules.json"
        f.write_text(json.dumps({"rules": VALID_RULES}), encoding="utf-8")
        loader = RuleLoader(f)
        rules = loader.load()
        assert len(rules) == 2

    def test_load_unsupported_format(self, tmp_path):
        f = tmp_path / "rules.txt"
        f.write_text("dummy", encoding="utf-8")
        loader = RuleLoader(f)
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load()

    def test_load_nonexistent_file(self, tmp_path):
        loader = RuleLoader(tmp_path / "no_such.json")
        with pytest.raises(FileNotFoundError):
            loader.load()


# ====================================================================== #
# RuleLoader — 保存
# ====================================================================== #


class TestRuleLoaderSave:
    def test_save_json(self, tmp_path):
        f = tmp_path / "rules.json"
        loader = RuleLoader(f)
        loader.save(VALID_RULES)
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_save_yaml(self, tmp_path):
        f = tmp_path / "rules.yaml"
        loader = RuleLoader(f)
        loader.save(VALID_RULES)
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_save_then_load_roundtrip(self, tmp_path):
        f = tmp_path / "rules.json"
        loader = RuleLoader(f)
        loader.save(VALID_RULES)
        # 新 loader 实例加载
        loader2 = RuleLoader(f)
        rules = loader2.load()
        assert len(rules) == 2
        assert rules[0]["id"] == "bearing_rms_high"


# ====================================================================== #
# RuleLoader — 变化检测
# ====================================================================== #


class TestRuleLoaderChanged:
    def test_has_changed_false_initially(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        loader.load()  # 记录 mtime
        assert loader.has_changed() is False

    def test_has_changed_true_after_modify(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        loader.load()
        # 修改文件
        f.write_text(json.dumps([_make_rule("new")]), encoding="utf-8")
        _touch_future(f)
        assert loader.has_changed() is True

    def test_has_changed_nonexistent_file(self, tmp_path):
        loader = RuleLoader(tmp_path / "no_such.json")
        assert loader.has_changed() is False

    def test_has_changed_new_file_created(self, tmp_path):
        f = tmp_path / "rules.json"
        # 初始化时文件不存在
        loader = RuleLoader(f)
        assert loader.has_changed() is False
        # 创建文件
        f.write_text("[]", encoding="utf-8")
        assert loader.has_changed() is True

    def test_reload_if_changed_returns_rules(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        loader.load()
        f.write_text(json.dumps([_make_rule("new")]), encoding="utf-8")
        _touch_future(f)
        result = loader.reload_if_changed()
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "new"

    def test_reload_if_changed_returns_none(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f)
        loader.load()
        assert loader.reload_if_changed() is None


# ====================================================================== #
# RuleLoader — 批量校验
# ====================================================================== #


class TestRuleLoaderValidate:
    def test_validate_rules_all_valid(self):
        ok, errors = RuleLoader.validate_rules(VALID_RULES)
        assert ok is True
        assert errors == []

    def test_validate_rules_with_errors(self):
        bad = [_make_rule(), {"id": "bad", "op": "!="}]
        ok, errors = RuleLoader.validate_rules(bad)
        assert ok is False
        assert len(errors) == 1
        assert "Rule #1" in errors[0]

    def test_validate_rules_empty_list(self):
        ok, errors = RuleLoader.validate_rules([])
        assert ok is True
        assert errors == []

    def test_validate_rules_multiple_errors(self):
        bad = [{"op": "!="}, {"id": "x"}, "not a dict"]
        ok, errors = RuleLoader.validate_rules(bad)  # type: ignore[arg-type]
        assert ok is False
        assert len(errors) == 3


# ====================================================================== #
# RuleLoader — 后台监听
# ====================================================================== #


class TestRuleLoaderWatching:
    def test_start_stop_watching(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        loader = RuleLoader(f, poll_interval_s=0.05)
        loader.load()
        reloaded: list[list] = []
        loader.start_watching(lambda rules: reloaded.append(rules))
        # 修改文件触发回调
        f.write_text(json.dumps([_make_rule("new")]), encoding="utf-8")
        _touch_future(f)
        time.sleep(0.3)
        loader.stop_watching()
        assert len(reloaded) >= 1
        assert reloaded[0][0]["id"] == "new"

    def test_stop_watching_without_start(self, tmp_path):
        f = tmp_path / "rules.json"
        loader = RuleLoader(f)
        # 不应抛异常
        loader.stop_watching()


# ====================================================================== #
# RuleHotReloader — 基本功能
# ====================================================================== #


class TestRuleHotReloader:
    def test_force_reload(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng)
        v = reloader.force_reload()
        assert v == 1
        assert len(eng.rules) == 2
        assert eng.get_version() == 1

    def test_force_reload_invalid_raises(self, tmp_path):
        f = tmp_path / "rules.json"
        bad = [_make_rule(), {"id": "bad", "op": "!="}]
        f.write_text(json.dumps(bad), encoding="utf-8")
        eng = RuleEngine(VALID_RULES)
        reloader = RuleHotReloader(f, eng)
        with pytest.raises(ValueError):
            reloader.force_reload()
        # 旧规则不变
        assert len(eng.rules) == 2
        assert reloader.reload_count() == 0

    def test_reload_count(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng)
        assert reloader.reload_count() == 0
        reloader.force_reload()
        assert reloader.reload_count() == 1
        # 修改文件再 force_reload
        f.write_text(json.dumps([_make_rule("r3")]), encoding="utf-8")
        reloader.force_reload()
        assert reloader.reload_count() == 2

    def test_last_reload_time_none_initially(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng)
        assert reloader.last_reload_time() is None

    def test_last_reload_time_after_reload(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng)
        reloader.force_reload()
        t = reloader.last_reload_time()
        assert t is not None
        # 刚刚重载，时间应该是最近的
        assert (time.time() - t.timestamp()) < 5

    def test_start_stop(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng, poll_interval_s=0.05)
        reloader.start()
        time.sleep(0.1)
        reloader.stop()
        # start/stop 不应抛异常

    def test_auto_reload_on_change(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng, poll_interval_s=0.05)
        reloader.force_reload()
        assert eng.get_version() == 1
        reloader.start()
        # 修改文件为新规则
        new_rules = [_make_rule("r3", "current", ">", 10.0)]
        f.write_text(json.dumps(new_rules), encoding="utf-8")
        _touch_future(f)
        time.sleep(0.3)
        reloader.stop()
        assert eng.get_version() == 2
        assert eng.get_rule_ids() == {"r3"}
        assert reloader.reload_count() == 2


# ====================================================================== #
# RuleHotReloader — 校验失败处理
# ====================================================================== #


class TestHotReloaderValidationFailure:
    def test_invalid_rules_not_reloaded(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng, poll_interval_s=0.05)
        reloader.force_reload()
        original_ids = eng.get_rule_ids()
        reloader.start()
        # 写入无效规则
        bad = [{"id": "bad", "op": "!="}]
        f.write_text(json.dumps(bad), encoding="utf-8")
        _touch_future(f)
        time.sleep(0.3)
        reloader.stop()
        # 旧规则不变
        assert eng.get_rule_ids() == original_ids
        # 版本未增加
        assert eng.get_version() == 1

    def test_error_messages_recorded(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(VALID_RULES), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng, poll_interval_s=0.05)
        reloader.force_reload()
        assert reloader.errors == []
        reloader.start()
        bad = [{"id": "bad", "op": "!="}]
        f.write_text(json.dumps(bad), encoding="utf-8")
        _touch_future(f)
        time.sleep(0.3)
        reloader.stop()
        assert len(reloader.errors) >= 1
        assert "invalid" in reloader.errors[0].lower()


# ====================================================================== #
# 边界 / 异常路径
# ====================================================================== #


class TestEdgeCases:
    def test_empty_rules_file(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text("[]", encoding="utf-8")
        loader = RuleLoader(f)
        rules = loader.load()
        assert rules == []

    def test_empty_yaml_file_raises(self, tmp_path):
        f = tmp_path / "rules.yaml"
        f.write_text("", encoding="utf-8")
        loader = RuleLoader(f)
        with pytest.raises(ValueError):
            loader.load()

    def test_engine_evaluate_after_reload_consistency(self, tmp_path):
        """重载后 evaluate 使用新规则。"""
        f = tmp_path / "rules.json"
        f.write_text(json.dumps([_make_rule("r1", "temperature", ">", 80.0)]), encoding="utf-8")
        eng = RuleEngine([])
        reloader = RuleHotReloader(f, eng)
        reloader.force_reload()
        # 旧阈值 80，值 70 不触发
        assert eng.evaluate({"temperature": 70.0}) == []
        # 写入新阈值 50
        f.write_text(json.dumps([_make_rule("r1", "temperature", ">", 50.0)]), encoding="utf-8")
        reloader.force_reload()
        # 新阈值 50，值 70 触发
        hits = eng.evaluate({"temperature": 70.0})
        assert len(hits) == 1
        assert hits[0].threshold == 50.0

    def test_add_rule_does_not_break_evaluate(self):
        """动态添加规则后 evaluate 正常工作。"""
        eng = RuleEngine([_make_rule("r1", "temperature", ">", 80.0)])
        eng.add_rule(_make_rule("r2", "vibration", ">", 5.0))
        hits = eng.evaluate({"temperature": 100.0, "vibration": 10.0})
        assert {h.rule_id for h in hits} == {"r1", "r2"}

    def test_remove_rule_does_not_break_evaluate(self):
        eng = RuleEngine([_make_rule("r1", "temperature", ">", 80.0),
                          _make_rule("r2", "vibration", ">", 5.0)])
        eng.remove_rule("r1")
        hits = eng.evaluate({"temperature": 100.0, "vibration": 10.0})
        assert {h.rule_id for h in hits} == {"r2"}