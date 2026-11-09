"""T4 知识检索测试：ONNX 真实语义（512 维）+ hash-embed 降级路径。

覆盖三类场景（对齐 plan §5 测试策略与 §6 验收标准）：

1. 降级路径（默认，CI 无模型也必跑）：``model_dir`` 缺失 / ``model_dir=None`` /
   ``enabled=False`` 时优雅降级为 ``hash-embed``（32 维），``search`` 正常返回且不抛异常；
2. 真实语义（``skipif`` 保护，模型就绪时执行）：``onnx-bge-small-zh``（512 维），
   并对三个代表性 query 断言首位命中；
3. 维度迁移一致性：hash（32 维）与 onnx（512 维）交替打开同一库时，
   ``meta`` 记录的 ``embedding_backend`` / ``embedding_dim`` 被更新，且检索不抛 ``ValueError``。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edge.knowledge.fault_chain import FaultChain
from edge.knowledge.sqlite_vec import (
    HASH_BACKEND,
    HASH_DIM,
    ONNX_BACKEND,
    ONNX_DIM,
    LocalKnowledge,
    OnnxEmbedder,
    cosine,
)

ROOT = Path(__file__).resolve().parents[1]
SNIPPETS = ROOT / "edge" / "knowledge" / "snippets.json"
MODEL_DIR = ROOT / "models" / "knowledge" / "current"
# 保证不存在的路径，用于驱动降级分支（即使模型资产已就绪也不受影响）
MISSING_MODEL_DIR = ROOT / "models" / "knowledge" / "__not_prepared__"


def _onnx_ready() -> bool:
    """真实语义用例的前置条件：模型资产完整且 onnxruntime/tokenizers 可导入。"""
    if not (MODEL_DIR / "onnx" / "model_quantized.onnx").exists():
        return False
    if not (MODEL_DIR / "tokenizer.json").exists():
        return False
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:  # 依赖缺失时同样走降级路径，故跳过真实语义用例
        return False
    return True


requires_onnx = pytest.mark.skipif(
    not _onnx_ready(),
    reason="ONNX 模型资产或 onnxruntime/tokenizers 不可用，跳过真实语义用例",
)


@pytest.fixture()
def make_kb(tmp_path: Path) -> Iterator[Callable[..., LocalKnowledge]]:
    """构造 ``LocalKnowledge`` 的工厂，并在测试结束后统一关闭 sqlite 连接。

    Windows 下未关闭的连接会持有 ``.db`` 文件锁，统一关闭可避免临时目录清理失败。
    """

    created: list[LocalKnowledge] = []

    def _make(
        db_name: str = "kb.db",
        *,
        model_dir: Path | None = None,
        snippets_path: Path = SNIPPETS,
        **kwargs: Any,
    ) -> LocalKnowledge:
        kb = LocalKnowledge(tmp_path / db_name, snippets_path, model_dir=model_dir, **kwargs)
        created.append(kb)
        return kb

    yield _make
    for kb in created:
        kb.conn.close()


# --------------------------------------------------------------------------- #
# 原有用例：保留意图（故障链 + 检索接口可用）
# --------------------------------------------------------------------------- #
def test_two_hop():
    chain = FaultChain(ROOT / "edge" / "knowledge" / "fault_chains.json")
    hops = chain.two_hop("bearing_rms_high")
    assert hops[0]["id"] == "bearing_wear"
    assert len(hops) >= 1


def test_search(make_kb: Callable[..., LocalKnowledge]):
    """检索接口在默认（降级）配置下可用：命中均来自知识库、结构不变且按相似度降序。

    说明：hash-embed 为字节分布占位实现、不具备语义能力，故此处不再断言首位与
    query 主题的语义相关性（语义相关性由 ``@requires_onnx`` 用例覆盖）。
    """
    kb = make_kb()
    known_ids = {item["id"] for item in json.loads(SNIPPETS.read_text(encoding="utf-8"))}
    hits = kb.search("轴承振动 RMS", top_k=3)
    assert len(hits) == 3
    for hit in hits:
        assert set(hit) == {"id", "title", "text", "score"}
        assert hit["id"] in known_ids
        assert isinstance(hit["score"], float)
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_cosine_dot_and_dim_mismatch():
    """向后兼容的 cosine 为等长向量余弦；维度不一致时按 strict 语义抛 ValueError。

    该断言同时守护维度迁移风险：32 维与 512 维向量混算必须显式失败而非静默截断。
    """
    assert cosine([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cosine([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------- #
# 降级路径（默认，CI 无模型也必跑）
# --------------------------------------------------------------------------- #
def test_hash_fallback_when_model_dir_missing(make_kb: Callable[..., LocalKnowledge]):
    """模型目录不存在 → 优雅降级 hash-embed（32 维），不抛异常且检索可用。"""
    kb = make_kb(model_dir=MISSING_MODEL_DIR)
    assert kb.backend == HASH_BACKEND
    assert kb.dim == HASH_DIM

    hits = kb.search("驱动端轴承温度升高怎么办", top_k=5)
    assert len(hits) == 5
    assert all(hit["id"].startswith("kb-") for hit in hits)


def test_hash_fallback_when_model_dir_none(make_kb: Callable[..., LocalKnowledge]):
    """未提供 model_dir（旧签名默认值）→ 降级 hash-embed，保证向后兼容。"""
    kb = make_kb(model_dir=None)
    assert kb.backend == HASH_BACKEND
    assert kb.dim == HASH_DIM
    assert kb.search("轴承") != []


def test_hash_fallback_when_disabled(make_kb: Callable[..., LocalKnowledge]):
    """enabled=False 时即使模型就绪也强制走降级路径。"""
    kb = make_kb(model_dir=MODEL_DIR, enabled=False)
    assert kb.backend == HASH_BACKEND
    assert kb.dim == HASH_DIM
    assert kb.search("轴承") != []


def test_hash_fallback_when_degrade_disabled(make_kb: Callable[..., LocalKnowledge]):
    """degrade_to_hash=False 仅改变告警日志级别，模型缺失时依旧降级且不中断。"""
    kb = make_kb(model_dir=MISSING_MODEL_DIR, degrade_to_hash=False)
    assert kb.backend == HASH_BACKEND
    assert kb.dim == HASH_DIM
    assert kb.search("振动") != []


def test_search_empty_index(tmp_path: Path, make_kb: Callable[..., LocalKnowledge]):
    """空知识库时 search 走提前返回分支，返回空列表而非报错。"""
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    kb = make_kb(snippets_path=empty)
    assert kb.backend == HASH_BACKEND
    assert kb.search("任意查询") == []


# --------------------------------------------------------------------------- #
# 索引重建 / 复用（维度或内容不可用时的自愈）
# --------------------------------------------------------------------------- #
def test_corrupt_embedding_triggers_rebuild(make_kb: Callable[..., LocalKnowledge]):
    """库内 embedding 为非法 JSON 时触发重建，损坏数据被合法向量覆盖。"""
    kb = make_kb()
    kb.conn.execute("UPDATE snippets SET embedding='not-json' WHERE id='kb-bearing-rms'")
    kb.conn.commit()

    kb2 = make_kb("kb.db")  # 重新打开同一库
    assert kb2.backend == HASH_BACKEND
    row = kb2.conn.execute(
        "SELECT embedding FROM snippets WHERE id='kb-bearing-rms'"
    ).fetchone()
    assert row is not None
    assert len(json.loads(row[0])) == kb2.dim
    assert kb2.search("轴承振动 RMS") != []


def test_wrong_dim_embedding_triggers_rebuild(make_kb: Callable[..., LocalKnowledge]):
    """库内向量维度与当前后端不一致时触发重建，恢复为当前维度。"""
    kb = make_kb()
    bad_vector = json.dumps([1.0, 2.0, 3.0])  # 3 维，与 32 维不符
    kb.conn.execute(
        "UPDATE snippets SET embedding=? WHERE id='kb-gearbox-hs-cage'", (bad_vector,)
    )
    kb.conn.commit()

    kb2 = make_kb("kb.db")
    row = kb2.conn.execute(
        "SELECT embedding FROM snippets WHERE id='kb-gearbox-hs-cage'"
    ).fetchone()
    assert row is not None
    assert len(json.loads(row[0])) == kb2.dim
    assert kb2.search("齿轮箱高速轴") != []


@pytest.mark.parametrize("scalar", ["null", "123", "true"])
def test_scalar_embedding_triggers_rebuild_not_crash(
    make_kb: Callable[..., LocalKnowledge], scalar: str
):
    """库内 embedding 为 JSON 标量时触发重建而非崩溃（审查 M1 回归）。

    ``json.loads`` 对标量（null/number/bool）不抛异常却返回非列表对象，旧实现随后
    对其调用 ``len()`` 抛未捕获 ``TypeError``，导致 ``__init__`` 崩溃、网关启动失败。
    修复后应显式判为 stale，重开库时重建为合法向量并降级可用（默认 model_dir=None）。
    """
    kb = make_kb("kb.db")
    kb.conn.execute("UPDATE snippets SET embedding=? WHERE id='kb-bearing-rms'", (scalar,))
    kb.conn.commit()

    kb2 = make_kb("kb.db")  # 重开同一库不得抛异常
    assert kb2.backend == HASH_BACKEND
    assert kb2.dim == HASH_DIM
    row = kb2.conn.execute(
        "SELECT embedding FROM snippets WHERE id='kb-bearing-rms'"
    ).fetchone()
    assert row is not None
    assert len(json.loads(row[0])) == kb2.dim
    assert kb2.search("轴承振动 RMS") != []


@pytest.mark.parametrize(
    "bad_embedding",
    [
        ["a"] * HASH_DIM,  # 字符串数组，长度 == dim
        [[1.0, 2.0]] * HASH_DIM,  # 嵌套数组，长度 == dim
        [1.0] * (HASH_DIM - 1) + ["x"],  # 混合数组，长度 == dim
    ],
    ids=["str_array", "nested_array", "mixed_array"],
)
def test_non_numeric_embedding_triggers_rebuild_not_crash(
    make_kb: Callable[..., LocalKnowledge], bad_embedding: list
):
    """长度==dim 的非数值数组 embedding 触发重建而非崩溃（验证 §4.1 回归）。

    字符串数组 / 嵌套数组 / 混合数组的长度恰好等于 dim，绕过长度校验后
    ``np.asarray(..., dtype=np.float32)`` 会抛未捕获 ``ValueError``，使
    ``LocalKnowledge.__init__`` 崩溃、网关启动失败。修复后在元素级数值校验阶段
    即判为 stale，重开库时重建为合法向量并降级可用（默认 model_dir=None）。
    """
    kb = make_kb("kb.db")
    kb.conn.execute(
        "UPDATE snippets SET embedding=? WHERE id='kb-bearing-rms'",
        (json.dumps(bad_embedding),),
    )
    kb.conn.commit()

    kb2 = make_kb("kb.db")  # 重开同一库不得抛异常
    assert kb2.backend == HASH_BACKEND
    assert kb2.dim == HASH_DIM
    row = kb2.conn.execute(
        "SELECT embedding FROM snippets WHERE id='kb-bearing-rms'"
    ).fetchone()
    assert row is not None
    assert len(json.loads(row[0])) == kb2.dim
    assert kb2.search("轴承振动 RMS") != []


def test_reopen_reuses_existing_index(make_kb: Callable[..., LocalKnowledge]):
    """backend/维度/snippets 内容均未变化时不重建，检索结果保持稳定。"""
    kb1 = make_kb()
    kb2 = make_kb("kb.db")
    assert kb2.backend == kb1.backend
    assert kb2.dim == kb1.dim
    assert kb2.search("轴承温度", top_k=3) == kb1.search("轴承温度", top_k=3)


# --------------------------------------------------------------------------- #
# 真实语义（模型就绪时才执行）
# --------------------------------------------------------------------------- #
@requires_onnx
def test_onnx_backend_and_dim(make_kb: Callable[..., LocalKnowledge]):
    """模型就绪时启用 ONNX 后端且向量维度为 512。"""
    kb = make_kb(model_dir=MODEL_DIR)
    assert kb.backend == ONNX_BACKEND
    assert kb.dim == ONNX_DIM
    assert kb.dim == 512

    hits = kb.search("轴承振动", top_k=3)
    assert len(hits) == 3
    assert all(hit["id"].startswith("kb-") for hit in hits)


@requires_onnx
def test_onnx_dim_overrides_mismatched_config(make_kb: Callable[..., LocalKnowledge]):
    """配置 dim 与模型维度不一致时以模型为准（512），不因配置值降级。"""
    kb = make_kb(model_dir=MODEL_DIR, dim=64)
    assert kb.backend == ONNX_BACKEND
    assert kb.dim == ONNX_DIM


@requires_onnx
def test_semantic_relevance_bearing_rms(make_kb: Callable[..., LocalKnowledge]):
    """「离心泵轴承振动越限与润滑不良」首位命中轴承振动片段。"""
    kb = make_kb(model_dir=MODEL_DIR)
    hits = kb.search("离心泵轴承振动越限与润滑不良", top_k=3)
    assert hits[0]["id"] == "kb-bearing-rms"
    assert hits[0]["score"] > hits[1]["score"]


@requires_onnx
def test_semantic_relevance_temp_current(make_kb: Callable[..., LocalKnowledge]):
    """「单纯电机电流升高是什么原因」首位命中温度/电流联合判据，而非轴承振动。"""
    kb = make_kb(model_dir=MODEL_DIR)
    hits = kb.search("单纯电机电流升高是什么原因", top_k=3)
    assert hits[0]["id"] == "kb-temp-current"
    assert hits[0]["score"] > hits[1]["score"]


@requires_onnx
def test_semantic_relevance_bearing_temperature(make_kb: Callable[..., LocalKnowledge]):
    """「驱动端轴承温度升高怎么办」首要指向轴承振动片段而非通用温升条目。"""
    kb = make_kb(model_dir=MODEL_DIR)
    hits = kb.search("驱动端轴承温度升高怎么办", top_k=3)
    assert hits[0]["id"] == "kb-bearing-rms"
    assert hits[0]["score"] > hits[1]["score"]


@requires_onnx
def test_onnx_embedder_dim_and_encode():
    """OnnxEmbedder 直接编码：维度为 512、dtype 为 float32 且向量 L2 归一化。"""
    embedder = OnnxEmbedder(MODEL_DIR)
    assert embedder.dim == ONNX_DIM

    vectors = embedder.encode(["驱动端轴承振动", "电机电流升高"])
    assert vectors.shape == (2, ONNX_DIM)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@requires_onnx
def test_onnx_embedder_raises_when_model_missing(tmp_path: Path):
    """ONNX 文件缺失时抛出 FileNotFoundError，消息包含缺失路径信息。"""
    with pytest.raises(FileNotFoundError, match="ONNX 模型不存在"):
        OnnxEmbedder(tmp_path / "absent")


@requires_onnx
def test_onnx_embedder_raises_when_tokenizer_missing(tmp_path: Path):
    """ONNX 文件存在但 tokenizer.json 缺失时同样抛出 FileNotFoundError。"""
    fake_model = tmp_path / "fake_model"
    (fake_model / "onnx").mkdir(parents=True)
    (fake_model / "onnx" / "model_quantized.onnx").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="tokenizer.json 不存在"):
        OnnxEmbedder(fake_model)


# --------------------------------------------------------------------------- #
# 维度迁移一致性（32 维 hash ↔ 512 维 onnx 交替打开同一库）
# --------------------------------------------------------------------------- #
@requires_onnx
def test_dimension_migration_hash_to_onnx(make_kb: Callable[..., LocalKnowledge]):
    """先用 hash 建库、再用 onnx 打开同一库：meta 维度更新，检索不抛 ValueError。"""
    kb_hash = make_kb("kb.db", model_dir=MISSING_MODEL_DIR)
    assert kb_hash.backend == HASH_BACKEND
    meta_hash = dict(kb_hash.conn.execute("SELECT key, value FROM meta").fetchall())
    assert meta_hash["embedding_backend"] == HASH_BACKEND
    assert meta_hash["embedding_dim"] == str(HASH_DIM)

    kb_onnx = make_kb("kb.db", model_dir=MODEL_DIR)
    assert kb_onnx.backend == ONNX_BACKEND
    assert kb_onnx.dim == ONNX_DIM
    meta_onnx = dict(kb_onnx.conn.execute("SELECT key, value FROM meta").fetchall())
    assert meta_onnx["embedding_backend"] == ONNX_BACKEND
    assert meta_onnx["embedding_dim"] == str(ONNX_DIM)

    # 库内每条向量均已迁移为 512 维，检索不再因维度不一致抛 ValueError
    rows = kb_onnx.conn.execute("SELECT embedding FROM snippets").fetchall()
    assert all(len(json.loads(row[0])) == ONNX_DIM for row in rows)

    hits = kb_onnx.search("离心泵轴承振动越限与润滑不良", top_k=3)
    assert hits[0]["id"] == "kb-bearing-rms"


@requires_onnx
def test_dimension_migration_onnx_to_hash(make_kb: Callable[..., LocalKnowledge]):
    """反向迁移：先用 onnx 建库、再用 hash 打开同一库，meta 维度回落且检索正常。"""
    kb_onnx = make_kb("kb.db", model_dir=MODEL_DIR)
    assert kb_onnx.backend == ONNX_BACKEND
    assert kb_onnx.dim == ONNX_DIM

    kb_hash = make_kb("kb.db", model_dir=MISSING_MODEL_DIR)
    assert kb_hash.backend == HASH_BACKEND
    assert kb_hash.dim == HASH_DIM
    meta_hash = dict(kb_hash.conn.execute("SELECT key, value FROM meta").fetchall())
    assert meta_hash["embedding_backend"] == HASH_BACKEND
    assert meta_hash["embedding_dim"] == str(HASH_DIM)

    rows = kb_hash.conn.execute("SELECT embedding FROM snippets").fetchall()
    assert all(len(json.loads(row[0])) == HASH_DIM for row in rows)
    assert kb_hash.search("离心泵轴承振动越限与润滑不良", top_k=3) != []