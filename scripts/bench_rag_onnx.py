#!/usr/bin/env python
"""路线 B 资源实测脚本：onnxruntime + bge-small-zh-v1.5 ONNX(int8)。

测量维度：
  1. 磁盘占用：ONNX 模型文件 + tokenizer + 依赖（site-packages）体积。
  2. 进程 RSS：基线 / 导入依赖 / 加载 tokenizer / 加载 session / 单条峰值 / 批量峰值。
  3. 检索延迟：top_k 余弦检索（tokenize + ONNX 前向 + 排序）的 P50/P95，
     语料库规模分别在 2 条（真实 snippets）与 ~1000 条（伪片段）。
  4. 索引构建：批量 embed 整个语料库的耗时与峰值。

设计说明：
  - 全部依赖独立 venv（.bench_rag_onnx/venv），不污染项目 .venv。
  - 推理仅用 onnxruntime，绝不引入 torch；tokenizer 用 transformers.AutoTokenizer。
  - embedder 采用 CLS pooling + L2 normalize，与 BGE 官方推荐用法一致。
  - RSS 峰值由后台高频采样线程捕获，避免错过瞬时分配。

用法：
  python scripts/bench_rag_onnx.py                      # 本机完整测量
  python scripts/bench_rag_onnx.py --docker             # 容器内精简测量（输出 JSON）
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# 本地模型优先，无需联网；HF_ENDPOINT 仅用于兜底下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / ".bench_rag_onnx"
DEFAULT_MODEL_DIR = BENCH_DIR / "model"
DEFAULT_ONNX = "onnx/model_quantized.onnx"
SNIPPETS = ROOT / "edge" / "knowledge" / "snippets.json"
RAW_OUT = ROOT / "evidence" / "raw" / "bench_rag_onnx.json"


# --------------------------------------------------------------------------- #
# RSS 采样
# --------------------------------------------------------------------------- #
def rss_mb() -> float:
    """当前进程 RSS（MiB）。"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@contextlib.contextmanager
def rss_peak_tracker(interval_s: float = 0.002) -> Iterator[Any]:
    """后台高频采样进程 RSS 峰值，yield 一个返回当前峰值(MiB)的可调用对象。"""
    proc = psutil.Process(os.getpid())
    state: dict[str, Any] = {"peak": proc.memory_info().rss, "stop": threading.Event()}

    def _sample() -> None:
        stop: threading.Event = state["stop"]
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > state["peak"]:
                state["peak"] = rss
            stop.wait(interval_s)

    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    try:
        yield lambda: state["peak"] / (1024 * 1024)
    finally:
        state["stop"].set()
        thread.join(timeout=1.0)


# --------------------------------------------------------------------------- #
# 语料库构造
# --------------------------------------------------------------------------- #
_DEVICES = ["离心泵", "轴流风机", "齿轮箱", "压缩机", "给水泵", "输送带电机", "液压站", "冷却塔"]
_LOCS = ["驱动端", "非驱动端", "联轴器侧", "高速轴", "低速轴", "轴承座"]
_FAULTS = ["润滑不良", "滚动体磨损", "对中不良", "保持架损伤", "轴承游隙异常", "基础松动"]
_ACTS = [
    "补充润滑脂并复测",
    "复核联轴器对中",
    "检查地脚螺栓扭矩",
    "更换轴承并做动平衡",
    "降低负荷观察趋势",
]


def _fake_snippet(i: int) -> dict[str, str]:
    """生成第 i 条工业设备故障知识伪片段（标题 + 正文，风格贴近真实 snippet）。"""
    dev = _DEVICES[i % len(_DEVICES)]
    loc = _LOCS[(i // len(_DEVICES)) % len(_LOCS)]
    fault = _FAULTS[(i // 7) % len(_FAULTS)]
    act = _ACTS[(i // 5) % len(_ACTS)]
    rms = 3.0 + (i % 40) * 0.15
    temp = 55.0 + (i % 30) * 0.8
    return {
        "id": f"kb-fake-{i:04d}",
        "title": f"{dev}{loc}轴承振动 RMS {rms:.1f}mm/s 越限",
        "text": (
            f"案例 {i:04d}：{dev}的{loc}轴承振动 RMS 达到 {rms:.1f} mm/s，"
            f"轴承温度 {temp:.0f}℃，疑似{fault}。建议先{act}，"
            f"若 24 小时内 RMS 未回落需停机检查。本结论为手册摘要，非正式诊断报告。"
        ),
    }


def build_corpus(n: int) -> list[dict[str, str]]:
    """前若干条用真实 snippets，其余用伪片段补足到 n 条。"""
    docs: list[dict[str, str]] = json.loads(SNIPPETS.read_text(encoding="utf-8"))
    if len(docs) > n:
        docs = docs[:n]
    for i in range(len(docs), n):
        docs.append(_fake_snippet(i))
    return docs


def build_queries() -> list[str]:
    """检索查询集，覆盖真实片段与伪片段主题。"""
    return [
        "驱动端轴承振动超过阈值，温度也升高，怎么处理？",
        "离心泵振动大伴随温度上升是什么原因",
        "齿轮箱高速轴振动 RMS 越限如何处理",
        "风机轴承润滑不良的现场检查步骤",
        "轴承温度升高同时电流升高说明什么",
        "压缩机低速轴轴承异响和振动怎么办",
        "冷却塔电机振动异常，是否需要停机",
        "基础松动会导致哪些振动特征",
        "给水泵非驱动端振动超标的排查顺序",
        "输送带电机联轴器侧振动偏高的原因",
    ]


# --------------------------------------------------------------------------- #
# ONNX embedder
# --------------------------------------------------------------------------- #
class BgeOnnxEmbedder:
    """bge-small-zh-v1.5 ONNX(int8) 嵌入器：tokenize + ONNX 前向 + CLS pooling + L2。"""

    def __init__(
        self,
        model_dir: Path,
        onnx_rel: str = DEFAULT_ONNX,
        max_length: int = 512,
        num_threads: int = 0,
        enable_arena: bool = True,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.onnx_path = model_dir / onnx_rel
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {self.onnx_path}")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads  # 0 = ORT 默认
        opts.enable_cpu_mem_arena = enable_arena
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._output_name = self.session.get_outputs()[0].name

    def tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return {k: np.asarray(v, dtype=np.int64) for k, v in enc.items() if k in self._input_names}

    def forward(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """返回归一化后的句向量 (B, 512)。"""
        out = self.session.run([self._output_name], feeds)[0]
        cls = out[:, 0, :]
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        return (cls / np.maximum(norms, 1e-12)).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.forward(self.tokenize(texts))

    def encode_timed(self, texts: list[str]) -> tuple[np.ndarray, float, float]:
        t0 = time.perf_counter()
        feeds = self.tokenize(texts)
        t1 = time.perf_counter()
        vecs = self.forward(feeds)
        t2 = time.perf_counter()
        return vecs, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
class Retriever:
    """预计算语料向量 + 查询时余弦 top_k 检索（等效 sqlite-vec 缓存命中路径）。"""

    def __init__(
        self, embedder: BgeOnnxEmbedder, docs: list[dict[str, str]], batch_size: int = 32
    ) -> None:
        self.embedder = embedder
        self.docs = docs
        texts = [f"{d['title']} {d['text']}" for d in docs]
        t0 = time.perf_counter()
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunks.append(self.embedder.encode(texts[start : start + batch_size]))
        t1 = time.perf_counter()
        self.matrix = np.vstack(chunks) if chunks else np.zeros((0, 512), dtype=np.float32)
        self.build_ms = (t1 - t0) * 1000.0

    def search_timed(self, query: str, top_k: int = 3) -> dict[str, Any]:
        qv, t_tok, t_fwd = self.embedder.encode_timed([query])
        q = qv[0]
        t0 = time.perf_counter()
        scores = self.matrix @ q
        k = min(top_k, len(scores))
        order = np.argsort(-scores, kind="stable")[:k]
        t1 = time.perf_counter()
        hits = [
            {
                "id": self.docs[j]["id"],
                "title": self.docs[j]["title"],
                "score": float(scores[j]),
            }
            for j in order
        ]
        return {
            "tokenize_ms": t_tok,
            "forward_ms": t_fwd,
            "rank_ms": (t1 - t0) * 1000.0,
            "total_ms": t_tok + t_fwd + (t1 - t0) * 1000.0,
            "hits": hits,
        }


def _stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bench_latency(
    embedder: BgeOnnxEmbedder, corpus_size: int, iters: int, warmup: int, top_k: int
) -> dict[str, Any]:
    docs = build_corpus(corpus_size)
    retriever = Retriever(embedder, docs)
    queries = build_queries()
    for i in range(warmup):
        retriever.search_timed(queries[i % len(queries)], top_k=top_k)

    tok, fwd, rnk, tot = [], [], [], []
    for i in range(iters):
        r = retriever.search_timed(queries[i % len(queries)], top_k=top_k)
        tok.append(r["tokenize_ms"])
        fwd.append(r["forward_ms"])
        rnk.append(r["rank_ms"])
        tot.append(r["total_ms"])
    sample = retriever.search_timed(queries[0], top_k=top_k)
    return {
        "corpus_size": len(docs),
        "iters": iters,
        "warmup": warmup,
        "top_k": top_k,
        "build_ms": retriever.build_ms,
        "tokenize": _stats(tok),
        "forward": _stats(fwd),
        "rank": _stats(rnk),
        "total": _stats(tot),
        "sample_hits": sample["hits"],
    }


# --------------------------------------------------------------------------- #
# 磁盘
# --------------------------------------------------------------------------- #
def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def disk_report(model_dir: Path) -> dict[str, Any]:
    files = []
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(model_dir)), "bytes": p.stat().st_size})
    venv = BENCH_DIR / "venv"
    site = venv / "Lib" / "site-packages"
    if not site.exists():
        site = venv / "lib"
    return {
        "model_dir_bytes": _dir_size_bytes(model_dir),
        "model_files": files,
        "venv_site_packages_bytes": _dir_size_bytes(site) if site.exists() else None,
        "venv_total_bytes": _dir_size_bytes(venv) if venv.exists() else None,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def env_report() -> dict[str, Any]:
    import onnxruntime as ort
    import tokenizers
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": psutil.virtual_memory().total / (1024 * 1024),
    }


def run_full(
    model_dir: Path,
    onnx_rel: str,
    iters: int,
    warmup: int,
    top_k: int,
    corpus_sizes: list[int],
    num_threads: int,
    enable_arena: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {"mode": "full", "enable_arena": enable_arena, "env": env_report()}
    result["rss_baseline_mb"] = rss_mb()  # 仅标准库 + psutil 已导入

    import onnxruntime as ort  # noqa: F401
    import transformers  # noqa: F401

    result["rss_after_imports_mb"] = rss_mb()

    embedder = BgeOnnxEmbedder(
        model_dir, onnx_rel=onnx_rel, num_threads=num_threads, enable_arena=enable_arena
    )
    result["rss_after_session_mb"] = rss_mb()
    result["rss_model_increment_mb"] = result["rss_after_session_mb"] - result["rss_baseline_mb"]

    # 单条推理峰值
    with rss_peak_tracker() as peak:
        vecs, t_tok, t_fwd = embedder.encode_timed(["驱动端轴承振动 RMS 越限，温度升高怎么处理"])
        _ = vecs[0][:4]
    result["single_rss_peak_mb"] = peak()
    result["single_rss_delta_mb"] = result["single_rss_peak_mb"] - result["rss_after_session_mb"]
    result["single_tokenize_ms"] = t_tok
    result["single_forward_ms"] = t_fwd

    # 批量推理（32 条）峰值
    batch = [d["title"] + " " + d["text"] for d in build_corpus(32)]
    with rss_peak_tracker() as peak:
        _ = embedder.encode(batch)
    result["batch_rss_peak_mb"] = peak()
    result["batch_rss_delta_mb"] = result["batch_rss_peak_mb"] - result["rss_after_session_mb"]

    # 语料库规模检索延迟
    lat = []
    for size in corpus_sizes:
        lat.append(bench_latency(embedder, size, iters, warmup, top_k))
    result["latency"] = lat

    # 稳态：连续单条查询后的常驻 RSS（确认无内存增长）
    steady = Retriever(embedder, build_corpus(1000))
    qs = build_queries()
    for i in range(200):
        steady.search_timed(qs[i % len(qs)], top_k=3)
    result["rss_steady_mb"] = rss_mb()

    result["rss_final_mb"] = rss_mb()
    result["disk"] = disk_report(model_dir)
    return result


def run_docker(
    model_dir: Path, onnx_rel: str, num_threads: int, iters: int = 30, enable_arena: bool = True
) -> dict[str, Any]:
    """容器内精简测量：加载 + 单条/批量峰值 + 检索 P95，输出单行 JSON。"""
    result: dict[str, Any] = {"mode": "docker", "enable_arena": enable_arena, "env": env_report()}
    result["rss_baseline_mb"] = rss_mb()
    try:
        embedder = BgeOnnxEmbedder(
            model_dir, onnx_rel=onnx_rel, num_threads=num_threads, enable_arena=enable_arena
        )
        result["load_ok"] = True
    except MemoryError as exc:  # pragma: no cover - 容器 OOM 时会以非 0 退出
        result["load_ok"] = False
        result["error"] = f"MemoryError: {exc}"
        return result
    result["rss_after_session_mb"] = rss_mb()
    result["rss_model_increment_mb"] = result["rss_after_session_mb"] - result["rss_baseline_mb"]

    with rss_peak_tracker() as peak:
        _ = embedder.encode(["驱动端轴承振动 RMS 越限，温度升高怎么处理"])
    result["single_rss_peak_mb"] = peak()

    batch = [d["title"] + " " + d["text"] for d in build_corpus(32)]
    with rss_peak_tracker() as peak:
        _ = embedder.encode(batch)
    result["batch_rss_peak_mb"] = peak()

    lat = bench_latency(embedder, 1000, iters, 5, 3)
    result["latency_1000"] = lat
    result["latency_1000_total_p95_ms"] = lat["total"]["p95"]
    result["latency_1000_total_p50_ms"] = lat["total"]["p50"]
    result["rss_final_mb"] = rss_mb()
    result["disk"] = disk_report(model_dir)
    result["load_ok"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="路线B ONNX RAG 资源实测")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx", default=DEFAULT_ONNX)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--corpus-sizes", type=int, nargs="+", default=[2, 1000])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--no-arena", action="store_true", help="禁用 ORT CPU arena（4G 场景省内存）"
    )
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--json-out", type=Path, default=RAW_OUT)
    args = parser.parse_args()

    with contextlib.suppress(Exception):  # pragma: no cover
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if args.docker:
        res = run_docker(
            args.model_dir,
            args.onnx,
            args.threads,
            iters=args.iters,
            enable_arena=not args.no_arena,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DOCKER_RESULT_JSON=" + json.dumps(res, ensure_ascii=False))
        return 0 if res.get("load_ok") else 3

    res = run_full(
        args.model_dir,
        args.onnx,
        args.iters,
        args.warmup,
        args.top_k,
        args.corpus_sizes,
        args.threads,
        enable_arena=not args.no_arena,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    e = res["env"]
    print("=" * 72)
    print("路线B：onnxruntime + bge-small-zh-v1.5 ONNX(int8) 资源实测")
    print("=" * 72)
    print(f"python={e['python']} platform={e['platform']}")
    print(f"onnxruntime={e['onnxruntime']} transformers={e['transformers']}")
    print(f"tokenizers={e['tokenizers']} numpy={e['numpy']} psutil={e['psutil']}")
    print(f"cpu(logical/physical)={e['cpu_logical']}/{e['cpu_physical']}")
    print(f"ram_total={e['ram_total_mb']:.0f} MiB")
    print("-" * 72)
    print("[磁盘]")
    print(f"  model_dir total = {res['disk']['model_dir_bytes'] / 1048576:.2f} MiB")
    for f in res["disk"]["model_files"]:
        print(f"    {f['bytes'] / 1024:>10.1f} KiB  {f['path']}")
    site = res["disk"]["venv_site_packages_bytes"]
    if site:
        print(f"  venv site-packages = {site / 1048576:.0f} MiB")
    print("-" * 72)
    print("[RSS MiB]")
    print(f"  baseline              = {res['rss_baseline_mb']:.1f}")
    print(f"  after imports         = {res['rss_after_imports_mb']:.1f}")
    print(
        f"  after session load    = {res['rss_after_session_mb']:.1f}"
        f"  (增量 {res['rss_model_increment_mb']:.1f})"
    )
    print(
        f"  single infer peak     = {res['single_rss_peak_mb']:.1f}"
        f"  (峰值增量 {res['single_rss_delta_mb']:.1f})"
    )
    print(
        f"  batch32 infer peak    = {res['batch_rss_peak_mb']:.1f}"
        f"  (峰值增量 {res['batch_rss_delta_mb']:.1f})"
    )
    print(f"  steady(200 queries)   = {res.get('rss_steady_mb', float('nan')):.1f}")
    print(f"  final                 = {res['rss_final_mb']:.1f}")
    print(f"  arena_enabled         = {res['enable_arena']}")
    print("-" * 72)
    print("[检索延迟 ms]  （tokenize + ONNX 前向 + 余弦排序）")
    for lat in res["latency"]:
        t, tk, fw, rk = lat["total"], lat["tokenize"], lat["forward"], lat["rank"]
        print(
            f"  corpus={lat['corpus_size']:>4}  iters={lat['iters']}  build={lat['build_ms']:.1f}ms"
        )
        print(
            f"    total P50={t['p50']:.2f} P95={t['p95']:.2f}"
            f"  tok P50={tk['p50']:.2f} fwd P50={fw['p50']:.2f} rank P50={rk['p50']:.3f}"
        )
        print(f"    top{lat['top_k']} hits: {[h['id'] for h in lat['sample_hits']]}")
    print("-" * 72)
    print(f"JSON → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
