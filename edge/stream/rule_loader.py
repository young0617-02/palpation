"""规则文件加载器与热更新管理器。

支持 JSON / YAML 格式的规则文件，通过 mtime 轮询检测文件变化，
变化时自动校验 → 原子重载 → 版本 +1。校验失败则保持旧规则不变。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from edge.stream.rules import RuleEngine

logger = logging.getLogger(__name__)

#: 支持的规则文件后缀
_JSON_SUFFIXES = {".json"}
_YAML_SUFFIXES = {".yaml", ".yml"}


class RuleLoader:
    """规则文件加载与变化检测。

    通过比较文件 mtime 判断是否变化，支持 JSON 和 YAML 格式。
    规则文件可以是顶层列表，也可以是 ``{"rules": [...]}`` 包装格式。
    """

    def __init__(self, rule_file: Path, poll_interval_s: float = 5.0):
        self.rule_file = Path(rule_file)
        self.poll_interval_s = poll_interval_s
        self._last_mtime: float | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._on_reload: Callable[[list[dict[str, Any]]], None] | None = None
        if self.rule_file.exists():
            self._last_mtime = self.rule_file.stat().st_mtime

    # ------------------------------------------------------------------ #
    # 加载 / 保存
    # ------------------------------------------------------------------ #

    def load(self) -> list[dict[str, Any]]:
        """加载规则文件，返回规则列表。

        Raises:
            FileNotFoundError: 规则文件不存在。
            ValueError: 不支持的文件格式或文件内容不是规则列表。
        """
        suffix = self.rule_file.suffix.lower()
        text = self.rule_file.read_text(encoding="utf-8")
        if suffix in _JSON_SUFFIXES:
            data = json.loads(text)
        elif suffix in _YAML_SUFFIXES:
            data = yaml.safe_load(text)
        else:
            raise ValueError(f"Unsupported rule file format: {suffix}")
        # 兼容 {"rules": [...]} 包装格式
        if isinstance(data, dict):
            data = data.get("rules", [])
        if not isinstance(data, list):
            raise ValueError("Rule file must contain a list of rules")
        self._last_mtime = self.rule_file.stat().st_mtime
        return data

    def save(self, rules: list[dict[str, Any]]) -> None:
        """保存规则列表到文件。"""
        suffix = self.rule_file.suffix.lower()
        if suffix in _JSON_SUFFIXES:
            text = json.dumps(rules, ensure_ascii=False, indent=2)
        elif suffix in _YAML_SUFFIXES:
            text = yaml.safe_dump(rules, allow_unicode=True, sort_keys=False)
        else:
            raise ValueError(f"Unsupported rule file format: {suffix}")
        self.rule_file.write_text(text, encoding="utf-8")
        self._last_mtime = self.rule_file.stat().st_mtime

    # ------------------------------------------------------------------ #
    # 变化检测
    # ------------------------------------------------------------------ #

    def has_changed(self) -> bool:
        """检查文件 mtime 是否变化。文件不存在时返回 False。"""
        if not self.rule_file.exists():
            return False
        current_mtime = self.rule_file.stat().st_mtime
        if self._last_mtime is None:
            return True
        return current_mtime != self._last_mtime

    def reload_if_changed(self) -> list[dict[str, Any]] | None:
        """如果文件变化则重载，返回新规则列表；未变化返回 None。"""
        if not self.has_changed():
            return None
        return self.load()

    # ------------------------------------------------------------------ #
    # 后台轮询监听
    # ------------------------------------------------------------------ #

    def start_watching(self, on_reload: Callable[[list[dict[str, Any]]], None]) -> None:
        """启动后台线程轮询文件变化，变化时调用 on_reload 回调。"""
        self._stop_event = threading.Event()
        self._on_reload = on_reload
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop_watching(self) -> None:
        """停止后台监听线程。"""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_s + 1)
            self._thread = None
        self._stop_event = None
        self._on_reload = None

    def _watch_loop(self) -> None:
        assert self._stop_event is not None
        assert self._on_reload is not None
        while not self._stop_event.is_set():
            try:
                new_rules = self.reload_if_changed()
                if new_rules is not None:
                    self._on_reload(new_rules)
            except Exception:  # noqa: BLE001
                logger.exception("Error in rule watch loop")
            self._stop_event.wait(self.poll_interval_s)

    # ------------------------------------------------------------------ #
    # 批量校验
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_rules(rules: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        """批量校验规则列表。

        Returns:
            (全部有效, 错误信息列表)
        """
        errors: list[str] = []
        for i, rule in enumerate(rules):
            if not RuleEngine.validate_rule(rule):
                errors.append(f"Rule #{i} invalid: {rule}")
        return len(errors) == 0, errors


class RuleHotReloader:
    """热更新管理器：组合 RuleLoader + RuleEngine。

    文件变化时自动校验 → 通过则原子重载 → 版本 +1 → 记录历史。
    校验失败则保持旧规则不变，记录错误信息。
    """

    def __init__(
        self,
        rule_file: Path,
        engine: RuleEngine,
        poll_interval_s: float = 5.0,
    ):
        self.loader = RuleLoader(rule_file, poll_interval_s)
        self.engine = engine
        self._reload_count = 0
        self._last_reload_time: datetime | None = None
        self.errors: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # 启动 / 停止
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """启动热更新：开始后台监听文件变化。"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止热更新。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.loader.poll_interval_s + 1)
            self._thread = None

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                new_rules = self.loader.reload_if_changed()
                if new_rules is not None:
                    self._apply_rules(new_rules)
            except Exception:  # noqa: BLE001
                logger.exception("Error in hot reload watch loop")
            self._stop_event.wait(self.loader.poll_interval_s)

    # ------------------------------------------------------------------ #
    # 重载逻辑
    # ------------------------------------------------------------------ #

    def _apply_rules(self, new_rules: list[dict[str, Any]]) -> bool:
        """校验并应用新规则。校验失败时记录错误并返回 False。"""
        ok, errors = RuleLoader.validate_rules(new_rules)
        if not ok:
            self.errors.extend(errors)
            logger.warning("Rule validation failed, keeping old rules: %s", errors)
            return False
        self.engine.reload(new_rules)
        self._reload_count += 1
        self._last_reload_time = datetime.now()
        return True

    def force_reload(self) -> int:
        """强制重载规则文件，返回新版本号。

        Raises:
            ValueError: 规则校验失败时抛出，旧规则保持不变。
        """
        rules = self.loader.load()
        ok, errors = RuleLoader.validate_rules(rules)
        if not ok:
            self.errors.extend(errors)
            raise ValueError(f"Rule validation failed: {errors}")
        self.engine.reload(rules)
        self._reload_count += 1
        self._last_reload_time = datetime.now()
        return self.engine.get_version()

    # ------------------------------------------------------------------ #
    # 状态查询
    # ------------------------------------------------------------------ #

    def reload_count(self) -> int:
        """返回累计重载次数。"""
        return self._reload_count

    def last_reload_time(self) -> datetime | None:
        """返回最后重载时间，未重载过返回 None。"""
        return self._last_reload_time
