#!/usr/bin/env python

"""端侧 T4 知识检索「路线 A」资源实测 (bench_rag_torch)。

路线 A = sentence-transformers + PyTorch(CPU) + BAAI/bge-small-zh-v1.5(512 维)。

目的：在 2 核 4GB 边缘约束假设下，实测引入 PyTorch 语义检索方案的真实资源开销，
为「是否把现有 hash-embed 占位实现升级为真实语义检索」提供可复核的决策依据。

实测维度：
  1) 磁盘：模型文件总大小 + 依赖安装后的 site-packages 额外占用。
  2) 内存(RSS, psutil 当前进程)：
     - 基线 RSS（导入 torch/transformers 后、加载模型前）
     - 加载 bge-small 后的 RSS 增量
     - 单条文本 embedding 推理的 RSS 峰值增量
     - 批量 32 条 embedding 的 RSS 峰值增量
  3) 延迟：top_k=3 检索（查询编码 + 余弦排序）的 P50/P95，
     分别在 (a) 现有 2 条 snippets、(b) 扩容到 1000 条伪片段 下测量，迭代 >= 20 次。
  4) 线程：在导入 torch 前锁死 BLAS/OMP 线程数，并在导入后 torch.set_num_threads(2)，
     以模拟 2 核目标设备。

设计要点：
  * 纯只读复用 edge/knowledge/snippets.json 与 edge/knowledge/sqlite_vec.py 的余弦语义，
    不修改任何生产源码；本脚本不写入项目 .venv。
  * 依赖独立虚拟环境 .bench_rag_torch/venv（见 evidence/rag_torch_bench.md）。
  * 输出 UTF-8，无 emoji；同时写 evidence/raw/rag_torch_bench.txt 与 .json。

用法（在项目根，使用实测专用 venv）：
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_XET=1
  .bench_rag_torch/venv/Scripts/python.exe scripts/bench_rag_torch.py

环境变量：
  BENCH_THREADS      模拟核数（默认 2）
  BENCH_ITERS        每个检索场景的迭代次数（默认 25）
  BENCH_BATCH        批量 embedding 条数（默认 32）
  BENCH_PSEUDO_N     伪片段扩容条数（默认 1000）
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

# --- 必须在 import torch/numpy 之前设置，才能生效（模拟 2C） ---
_BENCH_THREADS = os.environ.get("BENCH_THREADS", "2")
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_var, _BENCH_THREADS)

import psutil  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# 参数与路径
# ---------------------------------------------------------------------------
ITERS = int(os.environ.get("BENCH_ITERS", "25"))
BATCH = int(os.environ.get("BENCH_BATCH", "32"))
PSEUDO_N = int(os.environ.get("BENCH_PSEUDO_N", "1000"))
WARMUP = 3
TOP_K = 3

_MODEL_DIR = _ROOT / ".bench_rag_torch" / "models" / "bge-small-zh-v1.5"
_SNIPPETS = _ROOT / "edge" / "knowledge" / "snippets.json"
_DB_DIR = _ROOT / "evidence" / "raw"
_OUT_TXT = _DB_DIR / "rag_torch_bench.txt"
_OUT_JSON = _DB_DIR / "rag_torch_bench.json"
_VENV_SITE = _ROOT / ".bench_rag_torch" / "venv" / "Lib" / "site-packages"

_PROC = psutil.Process()


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _rss_mb() -> float:
    """当前进程 RSS（MB）。"""
    return _PROC.memory_info().rss / (1024 * 1024)


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total / (1024 * 1024)


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def _summary(xs: list[float]) -> dict[str, float]:
    return {
        "n": float(len(xs)),
        "mean": statistics.fmean(xs) if xs else 0.0,
        "p50": _quantile(xs, 0.50),
        "p95": _quantile(xs, 0.95),
        "p99": _quantile(xs, 0.99),
        "min": min(xs) if xs else 0.0,
        "max": max(xs) if xs else 0.0,
    }


class RssSampler:
    """后台线程高频采样当前进程 RSS，用于捕捉推理期间的峰值。"""

    def __init__(self, interval_s: float = 0.001) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb = _rss_mb()
        self.start_mb = self.peak_mb
        self.samples: list[float] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = _rss_mb()
            self.samples.append(v)
            if v > self.peak_mb:
                self.peak_mb = v
            time.sleep(self._interval)

    def __enter__(self) -> RssSampler:
        self.start_mb = _rss_mb()
        self.peak_mb = self.start_mb
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # 采样线程停止后，再补采一次，避免峰值恰好落在窗口边缘
        v = _rss_mb()
        if v > self.peak_mb:
            self.peak_mb = v

    @property
    def peak_delta_mb(self) -> float:
        return self.peak_mb - self.start_mb


class Report:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def emit(self, line: str = "") -> None:
        self._lines.append(line)
        print(line, flush=True)

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


# ---------------------------------------------------------------------------
# 伪片段扩容（确定性，可复现）
# ---------------------------------------------------------------------------
_TERMS = [
    "驱动端轴承振动 RMS 越限",
    "非驱动端轴承温度偏高",
    "电机电流异常升高",
    "联轴器对中不良",
    "润滑脂老化",
    "机械密封泄漏",
    "叶轮气蚀",
    "出口压力波动",
    "入口过滤器堵塞",
    "冷却水流量不足",
    "绕组绝缘下降",
    "变频器直流母线过压",
    "接地故障报警",
    "振动频谱二倍频突出",
    "轴承保持架磨损",
    "齿轮箱油温过限",
    "皮带打滑",
    "紧固螺栓松动",
    "传感器信号漂移",
    "sample_rate 采样缺失",
]


def _gen_pseudo_snippets(n: int) -> list[dict[str, str]]:
    """生成 n 条确定性中文伪片段，模拟知识库扩容。"""
    out: list[dict[str, str]] = []
    for i in range(n):
        t = _TERMS[i % len(_TERMS)]
        out.append(
            {
                "id": f"pseudo-{i:05d}",
                "title": f"{t}（手册条目 {i}）",
                "text": (
                    f"本条摘录自设备维护手册第 {i} 节：{t}。"
                    f"现场排查应先核对历史趋势与工况负荷，再结合温度、电流、压力等联合判据，"
                    f"必要时停机检查。条目编号 {i:05d}，为手册摘要，非正式诊断报告。"
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# numpy 余弦检索（复刻 edge/knowledge/sqlite_vec.py 的语义）
# ---------------------------------------------------------------------------
def _l2_normalize(mat: object) -> object:
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _cosine_rank(query_vec, embed_mat, top_k: int) -> list[int]:
    """对已 L2 归一化的向量做余弦（点积）排序，返回 top_k 下标。"""
    import numpy as np

    sims = embed_mat @ query_vec
    idx = np.argsort(-sims)[:top_k]
    return [int(i) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="T4 路线 A 资源实测 (torch CPU + bge-small-zh)")
    ap.add_argument("--model-dir", default=str(_MODEL_DIR))
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pseudo-n", type=int, default=PSEUDO_N)
    ap.add_argument("--skip-latency", action="store_true", help="仅测资源，跳过延迟")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    rep = Report()
    result: dict[str, object] = {}

    rep.emit("=" * 78)
    rep.emit(" T4 路线 A 资源实测报告 (sentence-transformers + PyTorch CPU + bge-small-zh-v1.5)")
    rep.emit("=" * 78)
    rep.emit(f" 生成时间        : {datetime.now(UTC).isoformat()}")
    rep.emit(f" 平台            : {platform.platform()}")
    rep.emit(f" CPU             : {os.environ.get('PROCESSOR_IDENTIFIER', platform.processor())}")
    rep.emit(f" 逻辑核数        : {os.cpu_count()}")
    rep.emit(f" Python          : {sys.version.split()[0]}")
    rep.emit(f" 模拟核数(BLAS)  : {_BENCH_THREADS}")
    rep.emit(f" 迭代次数        : {args.iters} (warmup {WARMUP})  batch={args.batch}")
    rep.emit(f" 伪片段条数      : {args.pseudo_n}")
    rep.emit(f" 模型目录        : {model_dir}")
    rep.emit("")

    # ---------------- 磁盘 ----------------
    rep.emit("[磁盘占用]")
    model_total = _dir_size_mb(model_dir)
    weights = 0.0
    weight_files: list[str] = []
    for f in ("model.safetensors", "pytorch_model.bin"):
        p = model_dir / f
        if p.exists():
            weights += p.stat().st_size / (1024 * 1024)
            weight_files.append(f"{f} ({p.stat().st_size / (1024 * 1024):.1f} MB)")
    site_mb = _dir_size_mb(_VENV_SITE)
    rep.emit(f"  模型目录总计    : {model_total:.2f} MB")
    rep.emit(f"  权重文件        : {' + '.join(weight_files) or 'N/A'}  = {weights:.2f} MB")
    dep_desc = "含 torch/transformers/sentence-transformers/... 全部依赖"
    rep.emit(f"  site-packages   : {site_mb:.2f} MB ({dep_desc})")
    rep.emit("")

    # ---------------- 导入（基线 RSS） ----------------
    rep.emit("[内存 — 基线 (导入后/加载模型前)]")
    rep.emit(f"  加载前进程 RSS  : {_rss_mb():.2f} MB (本脚本裸进程)")
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    import transformers  # noqa: E402
    from sentence_transformers import SentenceTransformer  # noqa: E402

    torch.set_num_threads(int(_BENCH_THREADS))
    baseline_mb = _rss_mb()
    rep.emit(f"  torch           : {torch.__version__}  (cuda={torch.cuda.is_available()})")
    rep.emit(f"  transformers    : {transformers.__version__}")
    rep.emit(f"  torch 线程      : {torch.get_num_threads()}")
    rep.emit(f"  基线 RSS        : {baseline_mb:.2f} MB  (导入 torch+transformers 后、加载模型前)")
    rep.emit("")
    result["baseline_rss_mb"] = round(baseline_mb, 2)

    # ---------------- 加载模型 ----------------
    rep.emit("[内存 — 加载 bge-small-zh-v1.5]")
    t0 = time.perf_counter()
    with RssSampler() as s_load:
        model = SentenceTransformer(str(model_dir), device="cpu")
        dim = int(model.get_sentence_embedding_dimension())
        # 触发一次前向，确保权重真正落到内存（懒加载兜底）
        _ = model.encode(["warmup"], normalize_embeddings=False)
    load_s = time.perf_counter() - t0
    loaded_mb = _rss_mb()
    rep.emit(f"  向量维度        : {dim}")
    rep.emit(f"  模型加载耗时    : {load_s:.2f} s")
    load_delta = loaded_mb - baseline_mb
    rep.emit(f"  加载后 RSS      : {loaded_mb:.2f} MB")
    rep.emit(f"  加载增量        : {load_delta:+.2f} MB  (峰值窗口 +{s_load.peak_delta_mb:.2f} MB)")
    rep.emit("  （对照）方案预估: 120 MB")
    rep.emit("")
    result.update(
        {
            "dim": dim,
            "load_seconds": round(load_s, 2),
            "loaded_rss_mb": round(loaded_mb, 2),
            "load_delta_mb": round(loaded_mb - baseline_mb, 2),
            "load_peak_delta_mb": round(s_load.peak_delta_mb, 2),
            "model_dir_mb": round(model_total, 2),
            "model_weights_mb": round(weights, 2),
            "site_packages_mb": round(site_mb, 2),
        }
    )

    # 预热编码
    for _ in range(WARMUP):
        model.encode(["预热查询：驱动端轴承振动 RMS 越限"], normalize_embeddings=False)

    # ---------------- 单条 embedding 峰值 ----------------
    single_text = "驱动端轴承振动 RMS 持续越限，伴随温度升高，需检查润滑与对中。"
    pre_single_mb = _rss_mb()
    with RssSampler() as s_single:
        model.encode([single_text], normalize_embeddings=False)
    post_single_mb = _rss_mb()
    rep.emit("[内存 — 单条文本 embedding 峰值]")
    rep.emit(f"  推理前 RSS      : {pre_single_mb:.2f} MB")
    rep.emit(f"  推理后 RSS      : {post_single_mb:.2f} MB")
    rep.emit(f"  采样峰值        : {s_single.peak_mb:.2f} MB")
    rep.emit(f"  峰值增量        : {s_single.peak_delta_mb:+.2f} MB")
    rep.emit("")
    result["single_peak_delta_mb"] = round(s_single.peak_delta_mb, 2)
    result["single_peak_rss_mb"] = round(s_single.peak_mb, 2)

    # ---------------- 批量 embedding 峰值 ----------------
    batch_texts = [
        f"批量片段 {i}：轴承温度与电流联合判据，检查润滑状态。" for i in range(args.batch)
    ]
    pre_batch_mb = _rss_mb()
    with RssSampler() as s_batch:
        emb_batch = model.encode(batch_texts, normalize_embeddings=False)
        _ = emb_batch.shape
    post_batch_mb = _rss_mb()
    rep.emit(f"[内存 — 批量 {args.batch} 条 embedding 峰值]")
    rep.emit(f"  推理前 RSS      : {pre_batch_mb:.2f} MB")
    rep.emit(f"  推理后 RSS      : {post_batch_mb:.2f} MB")
    rep.emit(f"  采样峰值        : {s_batch.peak_mb:.2f} MB")
    rep.emit(f"  峰值增量        : {s_batch.peak_delta_mb:+.2f} MB")
    rep.emit(f"  进程整体峰值    : {max(s_load.peak_mb, s_single.peak_mb, s_batch.peak_mb):.2f} MB")
    rep.emit("")
    result["batch_peak_delta_mb"] = round(s_batch.peak_delta_mb, 2)
    result["batch_peak_rss_mb"] = round(s_batch.peak_mb, 2)
    result["process_peak_rss_mb"] = round(max(s_load.peak_mb, s_single.peak_mb, s_batch.peak_mb), 2)

    if not args.skip_latency:
        rep.emit("=" * 78)
        rep.emit("[检索延迟] top_k=3，含查询编码 + 余弦排序")
        rep.emit("=" * 78)

        real_snippets = json.loads(_SNIPPETS.read_text(encoding="utf-8"))
        queries = [
            "驱动端轴承振动 RMS 越限",
            "轴承温度与电流联合异常",
            "电机电流升高是什么原因",
        ]

        def measure_latency(corpus: list[dict[str, str]], tag: str) -> dict[str, float]:
            texts = [c["title"] + " " + c["text"] for c in corpus]
            with RssSampler() as s_enc:
                mat = model.encode(
                    texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
                )
            mat = np.asarray(mat, dtype=np.float32)
            build_peak = s_enc.peak_delta_mb
            rep.emit(f"  [{tag}] 语料条数={len(corpus)}  建索引峰值增量 +{build_peak:.2f} MB")

            # warmup
            for i in range(WARMUP):
                qv = model.encode([queries[i % len(queries)]], normalize_embeddings=True)[0]
                _cosine_rank(np.asarray(qv, dtype=np.float32), mat, TOP_K)

            samples: list[float] = []
            for i in range(args.iters):
                q = queries[i % len(queries)]
                t = time.perf_counter()
                qv = model.encode([q], normalize_embeddings=True)[0]
                _cosine_rank(np.asarray(qv, dtype=np.float32), mat, TOP_K)
                samples.append((time.perf_counter() - t) * 1000.0)
            st = _summary(samples)
            st["index_peak_delta_mb"] = build_peak
            return st

        rep.emit("")
        st_small = measure_latency(real_snippets, "现有 2 条 snippets")
        rep.emit(
            f"    P50={st_small['p50']:.2f} ms  P95={st_small['p95']:.2f} ms  "
            f"mean={st_small['mean']:.2f} ms  n={int(st_small['n'])}"
        )
        rep.emit("")
        pseudo = _gen_pseudo_snippets(args.pseudo_n)
        st_big = measure_latency(pseudo, f"扩容 {args.pseudo_n} 条伪片段")
        rep.emit(
            f"    P50={st_big['p50']:.2f} ms  P95={st_big['p95']:.2f} ms  "
            f"mean={st_big['mean']:.2f} ms  n={int(st_big['n'])}"
        )
        rep.emit("")
        rep.emit("[延迟汇总]")
        rep.emit("  场景                     P50(ms)   P95(ms)   mean(ms)   n")
        rep.emit(
            f"  2 条 snippets            {st_small['p50']:7.2f}   {st_small['p95']:7.2f}   "
            f"{st_small['mean']:7.2f}   {int(st_small['n'])}"
        )
        big_label = f"{args.pseudo_n} 条伪片段".ljust(22)
        rep.emit(
            f"  {big_label}{st_big['p50']:7.2f}   {st_big['p95']:7.2f}   "
            f"{st_big['mean']:7.2f}   {int(st_big['n'])}"
        )
        rep.emit("  方案 T4 目标             P95 < 80.00 ms")
        rep.emit("")
        result["latency_2"] = st_small
        result["latency_1000"] = st_big

    rep.emit("=" * 78)
    rep.emit(" 原始输出已写入: " + str(_OUT_TXT))
    rep.emit("=" * 78)

    _DB_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_TXT.write_text(rep.text(), encoding="utf-8")
    _OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
