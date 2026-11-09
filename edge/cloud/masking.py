"""数据脱敏：HMAC 哈希、分箱、字段剥离、salt 轮换、工艺参数区间化与综合脱敏。"""

from __future__ import annotations

import datetime
import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SaltEntry",
    "SaltRotator",
    "hash_device_id",
    "bin_value",
    "strip_site_fields",
    "mask_process_params",
    "mask_payload",
]


def hash_device_id(device_id: str, salt: str) -> str:
    return hmac.new(salt.encode("utf-8"), device_id.encode("utf-8"), hashlib.sha256).hexdigest()


def bin_value(value: float, bins: list[float]) -> str:
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return f"[{bins[i]},{bins[i + 1]})"
    return f">={bins[-1]}"


def strip_site_fields(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"site", "customer", "plant", "device_id"}
    return {k: v for k, v in payload.items() if k not in blocked}


# ---------------------------------------------------------------------------
# salt 定期轮换管理
# ---------------------------------------------------------------------------


def _ensure_aware_utc(ts: datetime.datetime) -> datetime.datetime:
    """将 naive datetime 当作 UTC 处理，返回 aware datetime。"""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=datetime.UTC)
    return ts


@dataclass
class SaltEntry:
    """salt 历史记录条目。

    Attributes:
        salt: salt 值。
        effective_from: 该 salt 生效起始时间（UTC aware）。
    """

    salt: str
    effective_from: datetime.datetime


class SaltRotator:
    """salt 定期轮换管理器。

    维护 salt 历史记录，支持按时间窗口查询历史 salt 并验证历史哈希，
    确保 salt 轮换后仍可校验旧数据。

    Args:
        current_salt: 初始 salt。
        rotation_interval_days: 轮换间隔天数，默认 30。
    """

    def __init__(self, current_salt: str, rotation_interval_days: int = 30) -> None:
        if not current_salt:
            raise ValueError("current_salt 不能为空")
        if rotation_interval_days <= 0:
            raise ValueError("rotation_interval_days 必须为正整数")
        self.rotation_interval_days = rotation_interval_days
        now = _ensure_aware_utc(datetime.datetime.now(datetime.UTC))
        self._history: list[SaltEntry] = [SaltEntry(salt=current_salt, effective_from=now)]

    @property
    def current_salt(self) -> str:
        """当前生效的 salt。"""
        return self._history[-1].salt

    def get_salt(self, timestamp: datetime.datetime | None = None) -> str:
        """获取指定时间生效的 salt。

        Args:
            timestamp: 查询时间点。``None`` 表示当前时间（返回最新 salt）。

        Returns:
            该时间点生效的 salt。若 timestamp 早于最早 salt 生效时间，
            返回最早的 salt。
        """
        if timestamp is None:
            return self._history[-1].salt
        ts = _ensure_aware_utc(timestamp)
        # 从后往前找第一个 effective_from <= ts 的 salt
        for entry in reversed(self._history):
            if entry.effective_from <= ts:
                return entry.salt
        # ts 早于所有 salt 生效时间，返回最早的 salt
        return self._history[0].salt

    def rotate(self, new_salt: str) -> None:
        """轮换 salt。

        Args:
            new_salt: 新 salt 值，不能为空且不能与当前 salt 相同。
        """
        if not new_salt:
            raise ValueError("new_salt 不能为空")
        if new_salt == self.current_salt:
            raise ValueError("new_salt 不能与当前 salt 相同")
        now = _ensure_aware_utc(datetime.datetime.now(datetime.UTC))
        self._history.append(SaltEntry(salt=new_salt, effective_from=now))

    def verify(self, device_id: str, hashed: str, timestamp: datetime.datetime) -> bool:
        """验证历史哈希。

        使用 timestamp 对应的 salt 重新计算 device_id 的哈希，与给定哈希
        恒定时间比较。

        Args:
            device_id: 设备标识。
            hashed: 待验证的哈希值。
            timestamp: 原始哈希生成时的时间点。

        Returns:
            哈希匹配返回 ``True``，否则返回 ``False``。
        """
        salt = self.get_salt(timestamp)
        expected = hash_device_id(device_id, salt)
        return hmac.compare_digest(expected, hashed)

    def is_rotation_due(self, now: datetime.datetime | None = None) -> bool:
        """判断当前 salt 是否已超过轮换间隔，需要轮换。

        Args:
            now: 当前时间，``None`` 表示取系统当前时间。

        Returns:
            当前 salt 生效时长超过 ``rotation_interval_days`` 返回 ``True``。
        """
        if now is None:
            now = datetime.datetime.now(datetime.UTC)
        now = _ensure_aware_utc(now)
        latest = self._history[-1]
        elapsed = now - latest.effective_from
        return elapsed.days >= self.rotation_interval_days


# ---------------------------------------------------------------------------
# 工艺参数区间化
# ---------------------------------------------------------------------------


def mask_process_params(value: float, ranges: list[tuple[float, float]]) -> str:
    """工艺参数区间化。

    将连续数值映射到离散区间标签，避免泄露精确工艺参数。
    区间按 ``(lower, upper)`` 表示，返回值落入的区间标签 ``"[lower,upper)"``。
    若值超出所有区间，返回 ``"out_of_range"``。

    Args:
        value: 工艺参数原始值。
        ranges: 升序排列的区间列表，如 ``[(0, 40), (40, 60), (60, 100)]``。

    Returns:
        区间标签字符串。
    """
    if not ranges:
        return "out_of_range"
    for lower, upper in ranges:
        if lower <= value < upper:
            return f"[{lower},{upper})"
    # 值 >= 最后一个区间的 upper 或 < 第一个区间的 lower
    last_upper = ranges[-1][1]
    if value >= last_upper:
        return f">={last_upper}"
    return "out_of_range"


# ---------------------------------------------------------------------------
# 综合脱敏
# ---------------------------------------------------------------------------


def mask_payload(
    payload: dict[str, Any],
    salt: str,
    fields_to_hash: list[str],
    fields_to_strip: list[str],
) -> dict[str, Any]:
    """综合脱敏：哈希敏感字段 + 剥离禁止字段 + 保留其余字段。

    Args:
        payload: 原始数据载荷。
        salt: 用于 HMAC 哈希的 salt。
        fields_to_hash: 需要哈希处理的字段名列表（值转为 str 后 HMAC-SHA256）。
        fields_to_strip: 需要直接剥离的字段名列表。

    Returns:
        脱敏后的新 dict，原 payload 不被修改。
    """
    strip_set = set(fields_to_strip)
    hash_set = set(fields_to_hash)
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key in strip_set:
            continue
        if key in hash_set:
            result[key] = hash_device_id(str(value), salt)
        else:
            result[key] = value
    return result
