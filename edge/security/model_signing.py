"""模型签名校验：SHA256 完整性 + RSA-PSS 数字签名。

对边缘侧下发的模型文件提供完整性校验与抗篡改数字签名能力，
未签名或校验失败的模型将被拒绝加载。
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

__all__ = [
    "ModelSignature",
    "compute_sha256",
    "generate_rsa_key_pair",
    "sign_model",
    "verify_model",
    "verify_model_integrity",
]


@dataclass
class ModelSignature:
    """模型签名元数据。

    Attributes:
        model_hash: 模型文件 SHA256 哈希（十六进制）。
        signature: RSA-PSS 数字签名（Base64 编码）。
        signed_at: 签名时间（UTC ISO8601 字符串）。
        signed_by: 签名者标识。
    """

    model_hash: str
    signature: str
    signed_at: str
    signed_by: str = "edge-ai-agent"
    extra: dict[str, str] = field(default_factory=dict)


def compute_sha256(file_path: Path) -> str:
    """计算文件 SHA256 哈希。

    采用分块读取以支持大模型文件，避免一次性占用过多内存。

    Args:
        file_path: 模型文件路径。

    Returns:
        SHA256 哈希的十六进制字符串（小写）。
    """
    sha = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_rsa_key_pair(key_size: int = 2048) -> tuple[bytes, bytes]:
    """生成 RSA 密钥对，用于模型签名。

    Args:
        key_size: RSA 密钥位数，默认 2048。

    Returns:
        ``(private_key_pem, public_key_pem)`` 元组，均为 PEM 编码的 bytes。
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _rsa_pss_padding() -> padding.PSS:
    """返回 RSA-PSS 填充策略（SHA256 + MGF1，盐长度与哈希相同）。"""
    return padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=hashes.SHA256().digest_size,
    )


def sign_model(file_path: Path, private_key_pem: bytes) -> str:
    """对模型文件生成数字签名（RSA-PSS）。

    对文件的 SHA256 哈希进行 RSA-PSS 签名，返回 Base64 编码的签名字符串。

    Args:
        file_path: 模型文件路径。
        private_key_pem: RSA 私钥 PEM。

    Returns:
        Base64 编码的签名字符串。
    """
    private_key = cast(
        rsa.RSAPrivateKey, serialization.load_pem_private_key(private_key_pem, password=None)
    )
    file_hash = compute_sha256(file_path)
    signature = private_key.sign(
        file_hash.encode("utf-8"),
        _rsa_pss_padding(),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


def verify_model(file_path: Path, signature: str, public_key_pem: bytes) -> bool:
    """验证模型文件签名。

    Args:
        file_path: 模型文件路径。
        signature: Base64 编码的签名字符串。
        public_key_pem: RSA 公钥 PEM。

    Returns:
        签名验证通过返回 ``True``，失败或异常返回 ``False``。
    """
    try:
        public_key = cast(rsa.RSAPublicKey, serialization.load_pem_public_key(public_key_pem))
        file_hash = compute_sha256(file_path)
        signature_bytes = base64.b64decode(signature)
        public_key.verify(
            signature_bytes,
            file_hash.encode("utf-8"),
            _rsa_pss_padding(),
            hashes.SHA256(),
        )
    except Exception:
        return False
    return True


def verify_model_integrity(file_path: Path, expected_hash: str) -> bool:
    """验证模型完整性（哈希比对）。

    Args:
        file_path: 模型文件路径。
        expected_hash: 期望的 SHA256 哈希（十六进制）。

    Returns:
        实际哈希与期望哈希一致返回 ``True``，否则返回 ``False``。
    """
    try:
        actual_hash = compute_sha256(file_path)
    except Exception:
        return False
    # 使用恒定时间比较防止时序攻击
    return hmac.compare_digest(actual_hash, expected_hash)


def now_iso_utc() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串（用于签名时间戳）。"""
    return datetime.datetime.now(datetime.UTC).isoformat()
