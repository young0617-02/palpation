"""mTLS 证书管理：CA 自签名、设备证书签发、SSLContext 加载与证书验证。

基于 ``cryptography`` 库实现双向 TLS（mTLS）所需的证书生命周期管理，
覆盖 CA 生成、设备证书签发、SSL 上下文构建、证书链验证与指纹计算。
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import ipaddress
import ssl
import tempfile
from pathlib import Path
from typing import cast

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

__all__ = [
    "generate_ca_cert",
    "generate_device_cert",
    "load_cert_chain",
    "verify_cert",
    "fingerprint",
]


def _generate_rsa_key(key_size: int = 2048) -> rsa.RSAPrivateKey:
    """生成 RSA 私钥。"""
    return rsa.generate_private_key(public_exponent=65537, key_size=key_size)


def _build_name(common_name: str) -> x509.Name:
    """构建 X.509 Subject/Issuer 名称。"""
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _now_utc() -> datetime.datetime:
    """返回当前 UTC 时间（带时区信息）。"""
    return datetime.datetime.now(datetime.UTC)


def _serial_number() -> int:
    """生成正整数序列号（x509 要求 > 0）。"""
    return x509.random_serial_number()


def generate_ca_cert(common_name: str, days_valid: int = 3650) -> tuple[bytes, bytes]:
    """生成自签名 CA 证书。

    Args:
        common_name: CA 证书的 Common Name。
        days_valid: 有效天数，默认 3650（约 10 年）。

    Returns:
        ``(cert_pem, key_pem)`` 元组，均为 PEM 编码的 bytes。
    """
    key = _generate_rsa_key()
    name = _build_name(common_name)
    now = _now_utc()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def generate_device_cert(
    device_id: str,
    ca_cert_pem: bytes,
    ca_key_pem: bytes,
    days_valid: int = 365,
) -> tuple[bytes, bytes]:
    """用 CA 签发设备证书。

    Args:
        device_id: 设备标识，作为证书 Common Name 与 SAN DNS/IP。
        ca_cert_pem: CA 证书 PEM。
        ca_key_pem: CA 私钥 PEM。
        days_valid: 有效天数，默认 365。

    Returns:
        ``(cert_pem, key_pem)`` 元组，均为 PEM 编码的 bytes。
    """
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_key = cast(rsa.RSAPrivateKey, serialization.load_pem_private_key(ca_key_pem, password=None))

    device_key = _generate_rsa_key()
    subject = _build_name(device_id)
    now = _now_utc()

    # 构建 SAN：DNS 名 + 可能的 IP 地址
    san_entries: list[x509.GeneralName] = [x509.DNSName(device_id)]
    with contextlib.suppress(ValueError):
        # device_id 不是合法 IP 地址时仅使用 DNS 名
        san_entries.append(x509.IPAddress(ipaddress.ip_address(device_id)))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(device_key.public_key())
        .serial_number(_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [x509.ExtendedKeyUsageOID.CLIENT_AUTH, x509.ExtendedKeyUsageOID.SERVER_AUTH]
            ),
            critical=False,
        )
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = device_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def load_cert_chain(cert_pem: bytes, key_pem: bytes) -> ssl.SSLContext:
    """加载证书链创建 SSLContext（mTLS 双向认证）。

    将 PEM 写入临时文件后加载到 ``ssl.SSLContext``，并强制要求对端证书
    （``CERT_REQUIRED``），实现双向认证。

    Args:
        cert_pem: 证书 PEM。
        key_pem: 私钥 PEM。

    Returns:
        配置好证书链的 ``ssl.SSLContext``，协议为 TLS。
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED

    with tempfile.TemporaryDirectory() as tmp_dir:
        cert_path = Path(tmp_dir) / "cert.pem"
        key_path = Path(tmp_dir) / "key.pem"
        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)
        ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return ctx


def verify_cert(cert_pem: bytes, ca_cert_pem: bytes) -> bool:
    """验证证书是否由指定 CA 签发。

    通过比较证书的 issuer 与 CA 的 subject，并验证 CA 私钥对证书的签名
    来确认签发关系。同时检查证书是否过期。

    Args:
        cert_pem: 待验证证书 PEM。
        ca_cert_pem: CA 证书 PEM。

    Returns:
        签发关系成立且证书未过期返回 ``True``，否则返回 ``False``。
    """
    try:
        cert = x509.load_pem_x509_certificate(cert_pem)
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    except Exception:
        return False

    # 检查 issuer 与 CA subject 是否匹配
    if cert.issuer != ca_cert.subject:
        return False

    # 检查证书有效期
    now = _now_utc()
    not_before = cert.not_valid_before_utc
    not_after = cert.not_valid_after_utc
    if now < not_before or now > not_after:
        return False

    # 验证 CA 签发关系（issuer 匹配 + 签名验证）
    try:
        cert.verify_directly_issued_by(ca_cert)
    except Exception:
        return False
    return True


def fingerprint(cert_pem: bytes) -> str:
    """计算证书指纹（SHA256）。

    Args:
        cert_pem: 证书 PEM。

    Returns:
        SHA256 指纹的十六进制字符串（小写，无分隔符）。
    """
    cert = x509.load_pem_x509_certificate(cert_pem)
    # 优先使用 cryptography 内置指纹 API
    try:
        fp = cert.fingerprint(hashes.SHA256())
    except Exception:
        fp = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).digest()
    return fp.hex()
