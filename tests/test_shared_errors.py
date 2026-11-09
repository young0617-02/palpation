"""shared/errors.py 单元测试：EdgeError 异常与错误码常量。"""

from __future__ import annotations

import pytest

from shared.errors import (
    E_CONFLICT,
    E_DISK_FULL,
    E_INVALID_ARG,
    E_MODEL_HASH_MISMATCH,
    E_NOT_FOUND,
    E_PERMISSION,
    E_REQUIRES_FIRMWARE,
    E_TIMEOUT,
    E_UNSIGNED_MODEL,
    E_UPSTREAM,
    EdgeError,
)


def test_error_code_constants_values():
    """全部错误码常量等于其自身字符串字面量。"""
    assert E_TIMEOUT == "E_TIMEOUT"
    assert E_NOT_FOUND == "E_NOT_FOUND"
    assert E_INVALID_ARG == "E_INVALID_ARG"
    assert E_UPSTREAM == "E_UPSTREAM"
    assert E_CONFLICT == "E_CONFLICT"
    assert E_PERMISSION == "E_PERMISSION"
    assert E_MODEL_HASH_MISMATCH == "E_MODEL_HASH_MISMATCH"
    assert E_REQUIRES_FIRMWARE == "E_REQUIRES_FIRMWARE"
    assert E_DISK_FULL == "E_DISK_FULL"
    assert E_UNSIGNED_MODEL == "E_UNSIGNED_MODEL"


def test_error_codes_are_distinct():
    codes = [
        E_TIMEOUT, E_NOT_FOUND, E_INVALID_ARG, E_UPSTREAM, E_CONFLICT,
        E_PERMISSION, E_MODEL_HASH_MISMATCH, E_REQUIRES_FIRMWARE,
        E_DISK_FULL, E_UNSIGNED_MODEL,
    ]
    assert len(set(codes)) == len(codes)


def test_edge_error_construction_and_str():
    err = EdgeError(E_TIMEOUT, "request timed out")
    assert err.code == E_TIMEOUT
    assert err.message == "request timed out"
    assert str(err) == "E_TIMEOUT: request timed out"


def test_edge_error_is_exception():
    err = EdgeError(E_NOT_FOUND, "missing resource")
    assert isinstance(err, Exception)


def test_edge_error_raise_and_catch():
    with pytest.raises(EdgeError) as exc:
        raise EdgeError(E_MODEL_HASH_MISMATCH, "digest mismatch")
    assert exc.value.code == E_MODEL_HASH_MISMATCH
    assert "digest mismatch" in exc.value.message
    assert "E_MODEL_HASH_MISMATCH" in str(exc.value)


def test_edge_error_distinct_instances():
    a = EdgeError(E_TIMEOUT, "a")
    b = EdgeError(E_NOT_FOUND, "b")
    assert a.code != b.code
    assert a.message != b.message
    assert str(a) != str(b)