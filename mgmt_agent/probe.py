from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """探针结果。

    probe_ok/reason/details 为兼容字段（保持 probe_edge 旧接口）；
    inference_ok/mem_ok/disk_ok/temp_ok/input_links_ok/checks 为综合探针增强字段，
    单项探针（probe_edge）不填充，默认 None/空 dict。
    """

    probe_ok: bool
    reason: str
    details: dict
    # 综合探针增强字段（默认 None 以保持向后兼容）
    inference_ok: bool | None = None
    mem_ok: bool | None = None
    disk_ok: bool | None = None
    temp_ok: bool | None = None
    input_links_ok: bool | None = None
    checks: dict[str, bool] = field(default_factory=dict)


def probe_edge(health_url: str, timeout_s: float = 5.0) -> ProbeResult:
    """单点健康探针（兼容旧接口）：仅检查 /health 返回 ok 字段。"""
    try:
        r = httpx.get(health_url, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        ok = bool(data.get("ok"))
        return ProbeResult(ok, "ok" if ok else "health_false", data)
    except Exception as exc:
        return ProbeResult(False, str(exc), {})


def comprehensive_probe(
    edge_url: str,
    *,
    timeout_s: float = 5.0,
    mem_limit_mb: int = 4096,
    disk_min_mb: int = 512,
    temp_max_c: float = 85.0,
) -> ProbeResult:
    """综合健康探针：推理可用性 / 内存 / 磁盘 / 温度 / 输入链路。

    edge_url 指向边缘侧 /health 端点，返回 EdgeHealth 兼容字段：
      ok|probe_ok, mem_used_mb, disk_free_mb, cpu_temp_c, input_links。
    任一项不达标即 probe_ok=False，checks 给出每项明细，reason 列出失败项。

    阈值参数可调以适配不同硬件；默认值适用于典型边缘网关。
    """
    checks: dict[str, bool] = {}
    try:
        r = httpx.get(edge_url, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        # 健康端点不可达：所有检查均失败
        return ProbeResult(
            False,
            f"health_request_failed: {exc}",
            {},
            inference_ok=False,
            mem_ok=False,
            disk_ok=False,
            temp_ok=False,
            input_links_ok=False,
            checks={"health": False},
        )

    # 1. 推理可用性：ok 或 probe_ok 任一为真
    inference_ok = bool(data.get("ok") or data.get("probe_ok"))
    checks["inference"] = inference_ok

    # 2. 内存：mem_used_mb < mem_limit_mb
    mem_used = data.get("mem_used_mb")
    mem_ok = isinstance(mem_used, (int, float)) and mem_used < mem_limit_mb
    checks["memory"] = mem_ok

    # 3. 磁盘：disk_free_mb > disk_min_mb
    disk_free = data.get("disk_free_mb")
    disk_ok = isinstance(disk_free, (int, float)) and disk_free > disk_min_mb
    checks["disk"] = disk_ok

    # 4. 温度：cpu_temp_c < temp_max_c（None 视为未知→放行）
    temp = data.get("cpu_temp_c")
    temp_ok = temp is None or (isinstance(temp, (int, float)) and temp < temp_max_c)
    checks["temperature"] = temp_ok

    # 5. 输入链路：input_links 全部为真（空 dict 视为无链路→放行）
    links = data.get("input_links") or {}
    input_links_ok = all(bool(v) for v in links.values())
    checks["input_links"] = input_links_ok

    all_ok = inference_ok and mem_ok and disk_ok and temp_ok and input_links_ok
    failed = [k for k, v in checks.items() if not v]
    reason = "ok" if all_ok else f"failed: {','.join(failed)}"
    return ProbeResult(
        all_ok,
        reason,
        data,
        inference_ok=inference_ok,
        mem_ok=mem_ok,
        disk_ok=disk_ok,
        temp_ok=temp_ok,
        input_links_ok=input_links_ok,
        checks=checks,
    )
