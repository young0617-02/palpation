from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class RuleHit:
    rule_id: str
    metric: str
    value: float
    threshold: float
    severity: str
    description: str


OPS = {
    ">": lambda v, t: v > t,
    ">=": lambda v, t: v >= t,
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
}

#: 规则必须包含的字段
_REQUIRED_FIELDS = frozenset({"id", "metric", "threshold", "op", "severity"})


class RuleEngine:
    """规则引擎：评估指标是否触发规则，支持动态增删改与原子重载。

    线程安全说明：``evaluate`` 读取规则快照（原子引用读取），
    ``add_rule`` / ``remove_rule`` / ``update_rule`` / ``reload`` 在锁保护下
    替换规则列表，保证评估期间不会看到半更新的列表。
    """

    def __init__(self, rules: list[dict[str, Any]]):
        self.rules = rules
        self._version = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 原有接口（保持不变）
    # ------------------------------------------------------------------ #

    def evaluate(self, metrics: dict[str, float]) -> list[RuleHit]:
        hits: list[RuleHit] = []
        for rule in self.rules:
            metric = rule["metric"]
            if metric not in metrics:
                continue
            value = float(metrics[metric])
            threshold = float(rule["threshold"])
            op = OPS[rule["op"]]
            if op(value, threshold):
                hits.append(
                    RuleHit(
                        rule_id=rule["id"],
                        metric=metric,
                        value=value,
                        threshold=threshold,
                        severity=rule["severity"],
                        description=rule.get("description", rule["id"]),
                    )
                )
        return hits

    # ------------------------------------------------------------------ #
    # 规则校验
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_rule(rule: dict[str, Any]) -> bool:
        """校验单条规则格式。

        要求：包含 id / metric / threshold / op / severity 五个字段，
        op 必须在 OPS 中，threshold 必须是数值（非 bool）。
        """
        if not isinstance(rule, dict):
            return False
        if not _REQUIRED_FIELDS.issubset(rule.keys()):
            return False
        if rule["op"] not in OPS:
            return False
        threshold = rule["threshold"]
        return isinstance(threshold, int | float) and not isinstance(threshold, bool)

    # ------------------------------------------------------------------ #
    # 动态增删改
    # ------------------------------------------------------------------ #

    def add_rule(self, rule: dict[str, Any]) -> bool:
        """添加单条规则，校验通过后追加。返回是否添加成功。"""
        if not self.validate_rule(rule):
            return False
        with self._lock:
            self.rules = [*self.rules, rule]
            self._version += 1
        return True

    def remove_rule(self, rule_id: str) -> bool:
        """移除指定 ID 的规则。返回是否存在并移除。"""
        with self._lock:
            new_rules = [r for r in self.rules if r.get("id") != rule_id]
            if len(new_rules) == len(self.rules):
                return False
            self.rules = new_rules
            self._version += 1
        return True

    def update_rule(self, rule_id: str, updates: dict[str, Any]) -> bool:
        """更新指定 ID 的规则，合并 updates 后重新校验。

        校验失败或 rule_id 不存在时返回 False，不修改任何规则。
        """
        with self._lock:
            found = False
            new_rules: list[dict[str, Any]] = []
            for r in self.rules:
                if r.get("id") == rule_id:
                    merged = {**r, **updates}
                    if not self.validate_rule(merged):
                        return False
                    new_rules.append(merged)
                    found = True
                else:
                    new_rules.append(r)
            if not found:
                return False
            self.rules = new_rules
            self._version += 1
        return True

    # ------------------------------------------------------------------ #
    # 原子重载与版本管理
    # ------------------------------------------------------------------ #

    def reload(self, new_rules: list[dict[str, Any]]) -> int:
        """原子重载：先校验全部规则，通过后替换并版本 +1。

        Raises:
            ValueError: 当 new_rules 中存在无效规则时。
        """
        for i, rule in enumerate(new_rules):
            if not self.validate_rule(rule):
                raise ValueError(f"Invalid rule at index {i}: {rule}")
        with self._lock:
            self.rules = list(new_rules)
            self._version += 1
        return self._version

    def get_version(self) -> int:
        """获取当前规则版本号。"""
        return self._version

    def get_rule_ids(self) -> set[str]:
        """获取当前所有规则 ID 集合。"""
        return {r["id"] for r in self.rules if "id" in r}
