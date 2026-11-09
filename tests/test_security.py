"""E3 安全加固模块测试：mTLS 证书管理 + 模型签名校验 + 脱敏增强。"""

from __future__ import annotations

import datetime
import ssl
import time
from pathlib import Path

import pytest

from edge.cloud.masking import (
    SaltRotator,
    bin_value,
    hash_device_id,
    mask_payload,
    mask_process_params,
    strip_site_fields,
)
from edge.security import mtls
from edge.security.model_signing import (
    ModelSignature,
    compute_sha256,
    generate_rsa_key_pair,
    sign_model,
    verify_model,
    verify_model_integrity,
)


# ===========================================================================
# mTLS 证书生成
# ===========================================================================


class TestMTlsCertGeneration:
    """mTLS 证书生成测试。"""

    def test_generate_ca_cert_returns_pem_bytes(self) -> None:
        cert_pem, key_pem = mtls.generate_ca_cert("TestCA")
        assert isinstance(cert_pem, bytes)
        assert isinstance(key_pem, bytes)
        assert b"BEGIN CERTIFICATE" in cert_pem
        assert b"END CERTIFICATE" in cert_pem
        assert b"BEGIN RSA PRIVATE KEY" in key_pem or b"BEGIN PRIVATE KEY" in key_pem

    def test_generate_ca_cert_subject_matches_cn(self) -> None:
        from cryptography import x509

        cert_pem, _ = mtls.generate_ca_cert("MyRootCA")
        cert = x509.load_pem_x509_certificate(cert_pem)
        cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        assert cn == "MyRootCA"

    def test_generate_ca_cert_has_ca_basic_constraints(self) -> None:
        from cryptography import x509

        cert_pem, _ = mtls.generate_ca_cert("TestCA")
        cert = x509.load_pem_x509_certificate(cert_pem)
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        assert bc.ca is True

    def test_generate_device_cert_signed_by_ca(self) -> None:
        from cryptography import x509

        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestCA")
        dev_cert_pem, _ = mtls.generate_device_cert("device-001", ca_cert_pem, ca_key_pem)
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
        dev_cert = x509.load_pem_x509_certificate(dev_cert_pem)
        # 设备证书的 issuer 应等于 CA 的 subject
        assert dev_cert.issuer == ca_cert.subject

    def test_generate_device_cert_has_san(self) -> None:
        from cryptography import x509

        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestCA")
        dev_cert_pem, _ = mtls.generate_device_cert("device-001", ca_cert_pem, ca_key_pem)
        dev_cert = x509.load_pem_x509_certificate(dev_cert_pem)
        san = dev_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        dns_names = san.get_values_for_type(x509.DNSName)
        assert "device-001" in dns_names


# ===========================================================================
# mTLS 证书验证
# ===========================================================================


def _generate_expired_device_cert(ca_cert_pem: bytes, ca_key_pem: bytes) -> bytes:
    """构造一个已过期的设备证书（用于测试 verify_cert 过期路径）。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "device-001")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=10))
        .not_valid_after(now - datetime.timedelta(days=1))
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


class TestMTlsCertVerification:
    """mTLS 证书验证测试。"""

    def test_verify_cert_valid(self) -> None:
        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestCA")
        dev_cert_pem, _ = mtls.generate_device_cert("device-001", ca_cert_pem, ca_key_pem)
        assert mtls.verify_cert(dev_cert_pem, ca_cert_pem) is True

    def test_verify_cert_wrong_ca(self) -> None:
        ca1_pem, ca1_key = mtls.generate_ca_cert("CA1")
        ca2_pem, _ = mtls.generate_ca_cert("CA2")
        dev_cert_pem, _ = mtls.generate_device_cert("device-001", ca1_pem, ca1_key)
        # 用 CA2 验证 CA1 签发的证书应失败
        assert mtls.verify_cert(dev_cert_pem, ca2_pem) is False

    def test_verify_cert_expired(self) -> None:
        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestCA")
        # 直接构造一个已过期的设备证书
        dev_cert_pem = _generate_expired_device_cert(ca_cert_pem, ca_key_pem)
        assert mtls.verify_cert(dev_cert_pem, ca_cert_pem) is False

    def test_verify_cert_invalid_pem(self) -> None:
        ca_cert_pem, _ = mtls.generate_ca_cert("TestCA")
        assert mtls.verify_cert(b"not a certificate", ca_cert_pem) is False

    def test_fingerprint_sha256(self) -> None:
        ca_cert_pem, _ = mtls.generate_ca_cert("TestCA")
        fp = mtls.fingerprint(ca_cert_pem)
        # SHA256 指纹应为 64 个十六进制字符
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_deterministic(self) -> None:
        ca_cert_pem, _ = mtls.generate_ca_cert("TestCA")
        assert mtls.fingerprint(ca_cert_pem) == mtls.fingerprint(ca_cert_pem)


# ===========================================================================
# mTLS SSLContext 加载
# ===========================================================================


class TestMTlsSslContext:
    """mTLS SSLContext 加载测试。"""

    def test_load_cert_chain_returns_sslcontext(self) -> None:
        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestCA")
        dev_cert_pem, dev_key_pem = mtls.generate_device_cert("device-001", ca_cert_pem, ca_key_pem)
        ctx = mtls.load_cert_chain(dev_cert_pem, dev_key_pem)
        assert isinstance(ctx, ssl.SSLContext)

    def test_load_cert_chain_requires_client_cert(self) -> None:
        ca_cert_pem, ca_key_pem = mtls.generate_ca_cert("TestCA")
        dev_cert_pem, dev_key_pem = mtls.generate_device_cert("device-001", ca_cert_pem, ca_key_pem)
        ctx = mtls.load_cert_chain(dev_cert_pem, dev_key_pem)
        # mTLS 双向认证应要求对端证书
        assert ctx.verify_mode == ssl.CERT_REQUIRED


# ===========================================================================
# 模型签名
# ===========================================================================


class TestModelSigning:
    """模型数字签名测试。"""

    def test_compute_sha256(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake model content")
        h = compute_sha256(model_path)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_sha256_consistent(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"consistent content")
        assert compute_sha256(model_path) == compute_sha256(model_path)

    def test_sign_and_verify_model(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"original model bytes")
        private_pem, public_pem = generate_rsa_key_pair()
        signature = sign_model(model_path, private_pem)
        assert verify_model(model_path, signature, public_pem) is True

    def test_verify_model_tampered(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"original model bytes")
        private_pem, public_pem = generate_rsa_key_pair()
        signature = sign_model(model_path, private_pem)
        # 篡改模型文件
        model_path.write_bytes(b"tampered model bytes")
        assert verify_model(model_path, signature, public_pem) is False

    def test_verify_model_bad_signature(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model bytes")
        _, public_pem = generate_rsa_key_pair()
        # 使用无效签名
        assert verify_model(model_path, "invalid-signature-base64", public_pem) is False

    def test_verify_model_wrong_key(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model bytes")
        private_pem, _ = generate_rsa_key_pair()
        _, other_public_pem = generate_rsa_key_pair()
        signature = sign_model(model_path, private_pem)
        # 用不同公钥验证应失败
        assert verify_model(model_path, signature, other_public_pem) is False

    def test_model_signature_dataclass(self) -> None:
        sig = ModelSignature(
            model_hash="abc123",
            signature="base64sig==",
            signed_at="2026-01-01T00:00:00+00:00",
            signed_by="edge-agent",
        )
        assert sig.model_hash == "abc123"
        assert sig.signature == "base64sig=="
        assert sig.signed_at == "2026-01-01T00:00:00+00:00"
        assert sig.signed_by == "edge-agent"
        assert sig.extra == {}

    def test_model_signature_default_signed_by(self) -> None:
        sig = ModelSignature(model_hash="h", signature="s", signed_at="t")
        assert sig.signed_by == "edge-ai-agent"


# ===========================================================================
# 模型完整性校验
# ===========================================================================


class TestModelIntegrity:
    """模型完整性校验测试。"""

    def test_verify_model_integrity_match(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model content")
        expected = compute_sha256(model_path)
        assert verify_model_integrity(model_path, expected) is True

    def test_verify_model_integrity_mismatch(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model content")
        assert verify_model_integrity(model_path, "0" * 64) is False

    def test_verify_model_integrity_missing_file(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent.onnx"
        assert verify_model_integrity(missing_path, "0" * 64) is False

    def test_verify_model_integrity_tampered(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"original")
        expected = compute_sha256(model_path)
        model_path.write_bytes(b"tampered")
        assert verify_model_integrity(model_path, expected) is False


# ===========================================================================
# salt 轮换
# ===========================================================================


class TestSaltRotator:
    """salt 轮换管理测试。"""

    def test_initial_salt(self) -> None:
        rotator = SaltRotator("salt1")
        assert rotator.current_salt == "salt1"
        assert rotator.get_salt() == "salt1"

    def test_rotate_changes_current_salt(self) -> None:
        rotator = SaltRotator("salt1")
        rotator.rotate("salt2")
        assert rotator.current_salt == "salt2"

    def test_get_salt_time_window(self) -> None:
        rotator = SaltRotator("salt1")
        # 记录 salt1 生效期间的时间
        old_time = datetime.datetime.now(datetime.UTC)
        time.sleep(0.01)  # 确保 rotate 的时间严格大于 old_time
        rotator.rotate("salt2")
        # 旧时间应返回 salt1
        assert rotator.get_salt(old_time) == "salt1"
        # 当前时间应返回 salt2
        assert rotator.get_salt() == "salt2"
        # rotate 之后的时间应返回 salt2
        new_time = datetime.datetime.now(datetime.UTC)
        assert rotator.get_salt(new_time) == "salt2"

    def test_get_salt_before_earliest_returns_earliest(self) -> None:
        rotator = SaltRotator("salt1")
        # 早于 salt1 生效时间的时间点
        early_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=100)
        assert rotator.get_salt(early_time) == "salt1"

    def test_verify_historical_hash(self) -> None:
        rotator = SaltRotator("salt1")
        old_time = datetime.datetime.now(datetime.UTC)
        hashed_old = hash_device_id("device-001", "salt1")
        time.sleep(0.01)  # 确保 rotate 的时间严格大于 old_time
        rotator.rotate("salt2")
        # 用旧时间验证旧哈希应成功
        assert rotator.verify("device-001", hashed_old, old_time) is True

    def test_verify_historical_hash_wrong_salt(self) -> None:
        rotator = SaltRotator("salt1")
        old_time = datetime.datetime.now(datetime.UTC)
        time.sleep(0.01)  # 确保 rotate 的时间严格大于 old_time
        rotator.rotate("salt2")
        # 用新 salt 生成的哈希验证旧时间应失败
        hashed_new = hash_device_id("device-001", "salt2")
        assert rotator.verify("device-001", hashed_new, old_time) is False

    def test_is_rotation_due_false_initially(self) -> None:
        rotator = SaltRotator("salt1", rotation_interval_days=30)
        assert rotator.is_rotation_due() is False

    def test_is_rotation_due_true_after_interval(self) -> None:
        rotator = SaltRotator("salt1", rotation_interval_days=30)
        future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=31)
        assert rotator.is_rotation_due(future) is True


# ===========================================================================
# 工艺参数区间化
# ===========================================================================


class TestMaskProcessParams:
    """工艺参数区间化测试。"""

    def test_value_in_first_range(self) -> None:
        ranges = [(0, 40), (40, 60), (60, 100)]
        assert mask_process_params(25.0, ranges) == "[0,40)"

    def test_value_in_middle_range(self) -> None:
        ranges = [(0, 40), (40, 60), (60, 100)]
        assert mask_process_params(50.0, ranges) == "[40,60)"

    def test_value_above_max(self) -> None:
        ranges = [(0, 40), (40, 60), (60, 100)]
        assert mask_process_params(120.0, ranges) == ">=100"

    def test_value_below_min(self) -> None:
        ranges = [(0, 40), (40, 60), (60, 100)]
        assert mask_process_params(-5.0, ranges) == "out_of_range"

    def test_empty_ranges(self) -> None:
        assert mask_process_params(50.0, []) == "out_of_range"

    def test_boundary_value(self) -> None:
        ranges = [(0, 40), (40, 60)]
        # 边界 40 应落入 [40,60) 而非 [0,40)
        assert mask_process_params(40.0, ranges) == "[40,60)"


# ===========================================================================
# 综合脱敏
# ===========================================================================


class TestMaskPayload:
    """综合脱敏测试。"""

    def test_hashes_specified_fields(self) -> None:
        payload = {"device_id": "dev-001", "temperature": 50.0}
        result = mask_payload(payload, "salt", fields_to_hash=["device_id"], fields_to_strip=[])
        assert result["device_id"] == hash_device_id("dev-001", "salt")
        assert result["temperature"] == 50.0

    def test_strips_specified_fields(self) -> None:
        payload = {"device_id": "dev-001", "site": "factory-a", "temperature": 50.0}
        result = mask_payload(payload, "salt", fields_to_hash=[], fields_to_strip=["site"])
        assert "site" not in result
        assert result["device_id"] == "dev-001"
        assert result["temperature"] == 50.0

    def test_preserves_other_fields(self) -> None:
        payload = {"device_id": "dev-001", "temperature": 50.0, "pressure": 1.2}
        result = mask_payload(payload, "salt", fields_to_hash=["device_id"], fields_to_strip=[])
        assert result["temperature"] == 50.0
        assert result["pressure"] == 1.2

    def test_combined_hash_and_strip(self) -> None:
        payload = {
            "device_id": "dev-001",
            "site": "factory-a",
            "customer": "acme",
            "temperature": 50.0,
        }
        result = mask_payload(
            payload,
            "salt",
            fields_to_hash=["device_id"],
            fields_to_strip=["site", "customer"],
        )
        assert result["device_id"] == hash_device_id("dev-001", "salt")
        assert "site" not in result
        assert "customer" not in result
        assert result["temperature"] == 50.0

    def test_mask_payload_does_not_mutate_original(self) -> None:
        payload = {"device_id": "dev-001", "site": "factory-a"}
        original = dict(payload)
        mask_payload(payload, "salt", fields_to_hash=["device_id"], fields_to_strip=["site"])
        assert payload == original

    def test_mask_payload_hashes_non_string_value(self) -> None:
        payload = {"value": 12345}
        result = mask_payload(payload, "salt", fields_to_hash=["value"], fields_to_strip=[])
        assert result["value"] == hash_device_id("12345", "salt")


# ===========================================================================
# 边界/异常路径
# ===========================================================================


class TestEdgeCases:
    """边界与异常路径测试。"""

    def test_salt_rotator_empty_salt_raises(self) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            SaltRotator("")

    def test_salt_rotator_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="正整数"):
            SaltRotator("salt1", rotation_interval_days=0)

    def test_salt_rotator_rotate_same_salt_raises(self) -> None:
        rotator = SaltRotator("salt1")
        with pytest.raises(ValueError, match="不能与当前"):
            rotator.rotate("salt1")

    def test_salt_rotator_rotate_empty_raises(self) -> None:
        rotator = SaltRotator("salt1")
        with pytest.raises(ValueError, match="不能为空"):
            rotator.rotate("")

    def test_existing_functions_unchanged(self) -> None:
        """验证原有脱敏函数行为未被破坏。"""
        # hash_device_id
        h = hash_device_id("dev-001", "salt")
        assert len(h) == 64
        # bin_value
        assert bin_value(46.2, [0, 40, 60, 80, 100]) == "[40,60)"
        # strip_site_fields
        stripped = strip_site_fields({"site": "a", "device_id": "d", "temp": 50})
        assert "site" not in stripped
        assert "device_id" not in stripped
        assert stripped["temp"] == 50