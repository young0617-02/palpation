from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# 后端标识与向量维度常量
ONNX_BACKEND = "onnx-bge-small-zh"
HASH_BACKEND = "hash-embed"
HASH_DIM = 32
ONNX_DIM = 512
DEFAULT_ONNX_REL = "onnx/model_quantized.onnx"
DEFAULT_MAX_LENGTH = 512


def _hash_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    """Deterministic bag-of-bytes embedding used when bge-small is not present."""
    vec = [0.0] * dim
    data = text.encode("utf-8")
    for i, b in enumerate(data):
        vec[i % dim] += (b - 128) / 128.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def _is_number(x: object) -> bool:
    """向量元素是否为可转 float 的数值标量（显式排除 bool）。"""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def cosine(a: list[float], b: list[float]) -> float:
    """等长向量的余弦相似度；调用方须保证维度一致。"""
    return sum(x * y for x, y in zip(a, b, strict=True))


class OnnxEmbedder:
    """bge-small-zh-v1.5 ONNX(int8) 嵌入器：tokenize → ONNX 前向 → CLS pooling → L2 归一化。

    仅依赖 ``tokenizers``（直载 ``tokenizer.json``）与 ``onnxruntime``，**不引入**
    transformers / torch。加载异常由调用方（``LocalKnowledge``）捕获并降级。
    """

    def __init__(
        self,
        model_dir: Path,
        onnx_rel: str = DEFAULT_ONNX_REL,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_threads: int = 2,
        enable_cpu_mem_arena: bool = True,
    ) -> None:
        # 延迟导入：tokenizers / onnxruntime 缺失时由调用方降级，不影响模块导入
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / onnx_rel
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {self.onnx_path}")
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json 不存在: {tokenizer_path}")

        self.max_length = max_length
        # tokenizer：直载 fast tokenizer；自动补 [CLS]/[SEP] 并按批填充/截断
        self._tokenizer: Any = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 0 = 交由 ORT 决定；arena 打开时检索延迟更稳定（见 evidence/rag_onnx_bench.md）
        opts.intra_op_num_threads = num_threads
        opts.enable_cpu_mem_arena = enable_cpu_mem_arena
        self._session: Any = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # 输入名取交集，健壮应对不同导出（是否含 token_type_ids）
        self._input_names: set[str] = {i.name for i in self._session.get_inputs()}
        self._output_name: str = self._session.get_outputs()[0].name

    @property
    def dim(self) -> int:
        """向量维度（bge-small-zh-v1.5 固定 512）。"""
        return ONNX_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码为 (B, 512) 的 L2 归一化 float32 向量。"""
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        all_feeds: dict[str, np.ndarray] = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        }
        feeds = {k: v for k, v in all_feeds.items() if k in self._input_names}
        out = self._session.run([self._output_name], feeds)[0]
        cls = out[:, 0, :]  # CLS pooling（BGE 官方推荐）
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        return (cls / np.maximum(norms, 1e-12)).astype(np.float32)


class LocalKnowledge:
    """本地知识库检索：ONNX 语义嵌入（bge-small-zh-v1.5, 512 维）+ hash-embed 降级。

    - 模型就绪：``backend == "onnx-bge-small-zh"``、``dim == 512``；
    - 模型缺失 / 依赖缺失 / 加载失败：**优雅降级**回 ``_hash_embed``，
      ``backend == "hash-embed"``、``dim == 32``，``search`` 行为与改造前一致；
    - 向后兼容：旧签名 ``LocalKnowledge(db_path, snippets_path)`` 仍可用（新参数均有默认值）。
    """

    def __init__(
        self,
        db_path: Path,
        snippets_path: Path,
        *,
        model_dir: Path | None = None,
        dim: int = ONNX_DIM,
        enabled: bool = True,
        onnx_rel: str = DEFAULT_ONNX_REL,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_threads: int = 2,
        enable_cpu_mem_arena: bool = True,
        degrade_to_hash: bool = True,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS snippets("
            "id TEXT PRIMARY KEY, title TEXT, text TEXT, embedding TEXT)"
        )
        # meta 表记录 embedding_backend / embedding_dim / snippets_sha，用于维度迁移与增量判断
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        self.snippets_path = snippets_path

        self._embedder: OnnxEmbedder | None = self._build_embedder(
            model_dir=model_dir,
            enabled=enabled,
            onnx_rel=onnx_rel,
            max_length=max_length,
            num_threads=num_threads,
            enable_cpu_mem_arena=enable_cpu_mem_arena,
            degrade_to_hash=degrade_to_hash,
        )
        if self._embedder is not None:
            self.backend = ONNX_BACKEND
            self.dim = self._embedder.dim
            if dim != self.dim:
                log.warning("配置 dim=%d 与 ONNX 模型维度 %d 不一致，以模型为准", dim, self.dim)
        else:
            self.backend = HASH_BACKEND
            self.dim = HASH_DIM

        self._docs: list[dict[str, str]] = []
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._rebuild_index(snippets_path)

    # ------------------------------------------------------------------ #
    # 嵌入器构建（含降级）
    # ------------------------------------------------------------------ #
    def _build_embedder(
        self,
        *,
        model_dir: Path | None,
        enabled: bool,
        onnx_rel: str,
        max_length: int,
        num_threads: int,
        enable_cpu_mem_arena: bool,
        degrade_to_hash: bool,
    ) -> OnnxEmbedder | None:
        """尝试构建 ONNX 嵌入器；任何失败均返回 None，由调用方降级 hash-embed。"""
        if not enabled:
            log.info("knowledge.enabled=false，T4 使用 hash-embed 降级路径")
            return None
        if model_dir is None:
            log.info("未提供 knowledge model_dir，T4 使用 hash-embed 降级路径")
            return None
        try:
            embedder = OnnxEmbedder(
                model_dir,
                onnx_rel=onnx_rel,
                max_length=max_length,
                num_threads=num_threads,
                enable_cpu_mem_arena=enable_cpu_mem_arena,
            )
        except Exception as exc:  # 降级路径需吞掉任何加载异常（文件缺失/依赖缺失/损坏）
            level = logging.WARNING if degrade_to_hash else logging.INFO
            log.log(level, "ONNX 知识嵌入器不可用，降级 hash-embed: %s", exc)
            return None
        log.info("T4 知识检索启用 ONNX 语义嵌入: %s (dim=%d)", model_dir, embedder.dim)
        return embedder

    def _embed_text(self, text: str) -> list[float]:
        """用当前后端编码单条文本为归一化向量。"""
        if self._embedder is not None:
            vec = self._embedder.encode([text])[0]
            return [float(x) for x in vec]
        return _hash_embed(text, self.dim)

    # ------------------------------------------------------------------ #
    # 索引构建 / 加载
    # ------------------------------------------------------------------ #
    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _load_matrix(self) -> bool:
        """从库加载向量矩阵与文档到内存；维度/内容不可用时返回 False（触发重建）。"""
        rows = self.conn.execute("SELECT id, title, text, embedding FROM snippets").fetchall()
        try:
            vectors = [json.loads(r[3]) for r in rows]
        except (json.JSONDecodeError, TypeError):
            return False
        # JSON 标量（null/number/bool）在 JSON 层合法但不可作为向量：显式判为 stale，
        # 否则后续 len() 会抛未捕获 TypeError 导致启动崩溃（审查 M1）。
        if any(not isinstance(v, list) for v in vectors):
            return False
        # 元素级校验：长度须等于当前维度，且每个元素为数值标量（bool 视为非法）。
        # 字符串数组 / 嵌套数组 / 混合数组（长度恰好 == dim）若不在此拦截，
        # np.asarray 会抛未捕获 ValueError 使 __init__ 崩溃（验证 §4.1）。
        for v in vectors:
            if len(v) != self.dim:
                return False
            if not all(_is_number(x) for x in v):
                return False
        # np.asarray 作为最后兜底：任何残留的类型问题都不应冒泡出 __init__。
        try:
            matrix = (
                np.asarray(vectors, dtype=np.float32)
                if vectors
                else np.zeros((0, self.dim), dtype=np.float32)
            )
        except (ValueError, TypeError):
            return False
        # 全部校验通过后再赋值，避免 _docs/_matrix 半更新状态。
        self._docs = [{"id": r[0], "title": r[1], "text": r[2]} for r in rows]
        self._matrix = matrix
        return True

    def _rebuild_index(self, snippets_path: Path) -> None:
        """加载/重建向量索引：backend、维度或 snippets 内容变化时清空重建。"""
        snippets: list[dict[str, Any]] = json.loads(snippets_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(snippets_path.read_bytes()).hexdigest()
        meta = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())
        loaded = self._load_matrix()
        stale = (
            not loaded
            or meta.get("embedding_backend") != self.backend
            or meta.get("embedding_dim") != str(self.dim)
            or meta.get("snippets_sha") != digest
        )
        if not stale:
            return
        log.info(
            "重建知识库向量索引: backend=%s dim=%d items=%d",
            self.backend,
            self.dim,
            len(snippets),
        )
        # 编码/落库失败时降级 hash-embed 并重建，保证 __init__ 绝不因索引问题抛出（审查 M2）
        if self._try_rebuild(snippets, digest):
            return
        self._degrade_to_hash()
        if not self._try_rebuild(snippets, digest):
            # 极端兜底：连 hash 重建也失败时保持可用（空索引，search 返回 []）
            self._docs = []
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            log.error("知识库索引重建彻底失败，T4 将返回空结果")

    def _try_rebuild(self, snippets: list[dict[str, Any]], digest: str) -> bool:
        """按当前后端全量重建索引；失败时回滚并返回 False（不抛出）。"""
        try:
            self.conn.execute("DELETE FROM snippets")
            for s in snippets:
                embedding = self._embed_text(f"{s['title']} {s['text']}")
                self.conn.execute(
                    "INSERT OR REPLACE INTO snippets(id, title, text, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (s["id"], s["title"], s["text"], json.dumps(embedding)),
                )
            self._set_meta("embedding_backend", self.backend)
            self._set_meta("embedding_dim", str(self.dim))
            self._set_meta("snippets_sha", digest)
            self.conn.commit()
        except Exception as exc:  # 编码/落库任何异常均降级，不冒泡到 __init__
            log.warning("知识库重建失败(backend=%s dim=%d): %s", self.backend, self.dim, exc)
            with contextlib.suppress(sqlite3.Error):
                self.conn.rollback()
            return False
        self._load_matrix()
        return True

    def _degrade_to_hash(self) -> None:
        """切换到 hash-embed 降级后端，并保持矩阵/维度状态自洽。"""
        self._embedder = None
        self.backend = HASH_BACKEND
        self.dim = HASH_DIM
        self._docs = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        log.warning("T4 知识检索降级为 hash-embed (dim=%d)", self.dim)

    # ------------------------------------------------------------------ #
    # 检索（对外接口，签名与返回结构保持不变）
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索 top_k；返回 ``[{"id","title","text","score"}]``，score 为余弦相似度。"""
        if self._matrix.shape[0] == 0:
            return []
        qv = np.asarray(self._embed_text(query), dtype=np.float32)
        scores = self._matrix @ qv
        k = min(top_k, int(scores.shape[0]))
        order = np.argsort(-scores, kind="stable")[:k]
        hits: list[dict] = []
        for idx in order:
            j = int(idx)
            hits.append(
                {
                    "id": self._docs[j]["id"],
                    "title": self._docs[j]["title"],
                    "text": self._docs[j]["text"],
                    "score": float(scores[j]),
                }
            )
        return hits
