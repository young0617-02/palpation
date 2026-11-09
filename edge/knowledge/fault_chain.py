from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FaultChain:
    def __init__(self, path: Path):
        self.data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def lookup(self, rule_id: str) -> dict[str, Any]:
        for key, node in self.data.items():
            if rule_id in node.get("symptoms", []) or key in rule_id:
                return {"id": key, **node}
        if rule_id in self.data:
            return {"id": rule_id, **self.data[rule_id]}
        return {
            "id": "unknown",
            "symptoms": [rule_id],
            "causes": [],
            "actions": ["上报云端深度诊断"],
        }

    def two_hop(self, rule_id: str) -> list[dict[str, Any]]:
        root = self.lookup(rule_id)
        hops = [root]
        for cause in root.get("causes", [])[:2]:
            if cause in self.data:
                hops.append({"id": cause, **self.data[cause]})
        return hops
