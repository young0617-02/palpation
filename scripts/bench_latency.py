#!/usr/bin/env python

"""端云协同边缘 AI Agent 网关 - 分路径延迟基准实测 (bench_latency)。

依据 plan 方案 §6.4 延迟预算，实测四条路径的 P95 延迟：
  A(T1) 采集 -> 规则告警        目标 P95 < 50 ms
  B(T2) 采集 -> ONNX 判别模型    目标 P95 < 200 ms
  C(T4) 本地知识检索            目标 P95 < 80 ms
  D(T3) 告警 -> LLM 一级初判     目标 P95 < 1.2 s (2C 模拟)

设计要点：
  * 纯只读复用 edge/ 下生产代码，不修改任何源码。
  * 在 import numpy/onnxruntime/llama_cpp 之前限制 BLAS/OMP 线程数
    (BENCH_THREADS，默认 2)，以贴近方案标注的"2C 模拟"目标环境。
  * 输出 UTF-8，无 emoji。

用法:
  uv run python scripts/bench_latency.py

环境变量:
  BENCH_THREADS   限制 BLAS/OMP 线程数 (默认 2)
  BENCH_T3_ITERS  T3 真实推理次数 (默认 20)
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import os

# --- 必须在 import numpy/onnxruntime/llama_cpp 之前设置，才能生效 ---
_BENCH_THREADS = os.environ.get("BENCH_THREADS", "2")
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, _BENCH_THREADS)

import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# 保证脚本可独立运行（uv run python scripts/bench_latency.py）时能 import edge/shared
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edge.agent.llm_llamacpp import SYSTEM, EdgeLLM  # noqa: E402
from edge.agent.state_machine import EdgeAgentRuntime  # noqa: E402
from edge.agent.tools import build_tools  # noqa: E402
from edge.config_loader import load_yaml  # noqa: E402
from edge.knowledge.fault_chain import FaultChain  # noqa: E402
from edge.knowledge.sqlite_vec import LocalKnowledge  # noqa: E402
from edge.settings import ROOT, settings  # noqa: E402
from edge.stream.anomaly_onnx import AnomalyDetector  # noqa: E402
from edge.stream.rules import RuleEngine, RuleHit  # noqa: E402
from edge.stream.window import SlidingWindow  # noqa: E402
from shared.schemas import InternalSample  # noqa: E402

# ---------------------------------------------------------------------------
# 迭代次数（可在不影响结论的前提下调整）
# ---------------------------------------------------------------------------
T1_ITERS = 2000
T1_WARMUP = 300
T2_ITERS = 500
T2_WARMUP = 100
T4_ITERS = 500
T4_WARMUP = 100
T3_ITERS = int(os.environ.get("BENCH_T3_ITERS", "20"))
T3_TOK_SAMPLES = 3

T1_TARGET_MS = 50.0
T2_TARGET_MS = 200.0
T3_TARGET_MS = 1200.0
T4_TARGET_MS = 80.0

_OUT_PATH = ROOT / "evidence" / "raw" / "bench_latency.txt"


def _quantile(xs: list[float], q: float) -> float:
    """线性插值分位数（xs 非空）。"""
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] + (ys[hi] - ys[lo]) * frac


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


def _cpu_model() -> str:
    ident = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if ident:
        return ident
    proc = platform.processor().strip()
    return proc or "unknown"


class Report:
    """同时向 stdout 与文件输出的 UTF-8 报告收集器。"""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def emit(self, line: str = "") -> None:
        self._lines.append(line)
        print(line, flush=True)

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


def bench_t1(rep: Report, rules: list[dict], window_size: int) -> dict[str, float]:
    """T1：滑窗统计 push(3 metric) + 规则评估，严格复刻 runtime._handle_sample。"""
    window = SlidingWindow(window_size)
    engine = RuleEngine(rules)
    rng = np.random.default_rng(20261015)

    def gen() -> dict[str, float]:
        return {
            "vibration_rms": float(rng.normal(3.0, 1.2)),
            "temperature": float(rng.normal(72.0, 9.0)),
            "current": float(rng.normal(15.0, 3.5)),
        }

    # 预热：填满滑窗并让解释器/JIT 路径稳定
    for _ in range(T1_WARMUP):
        m = gen()
        for name, value in m.items():
            window.push(name, value)
        engine.evaluate(m)

    hits_count = 0
    samples: list[float] = []
    for _ in range(T1_ITERS):
        m = gen()
        t0 = time.perf_counter()
        for name, value in m.items():
            window.push(name, value)
        hits = engine.evaluate(m)
        samples.append((time.perf_counter() - t0) * 1000.0)
        hits_count += len(hits)

    st = _summary(samples)
    rep.emit("[T1 确定性告警路径 (滑窗统计 + 规则判定)]")
    rep.emit(f"  迭代次数      : {T1_ITERS} (预热 {T1_WARMUP})")
    rep.emit("  metrics       : vibration_rms / temperature / current")
    rep.emit(f"  规则条数      : {len(rules)} (从 config/current.yaml 加载)")
    rep.emit(f"  窗口 size     : {window_size}")
    rep.emit(f"  mean          : {st['mean']:.4f} ms")
    rep.emit(f"  P50           : {st['p50']:.4f} ms")
    rep.emit(f"  P95           : {st['p95']:.4f} ms")
    rep.emit(f"  P99           : {st['p99']:.4f} ms")
    rep.emit(f"  min / max     : {st['min']:.4f} / {st['max']:.4f} ms")
    rep.emit(f"  规则命中次数  : {hits_count} (仅作正确性参考)")
    rep.emit(f"  目标          : P95 < {T1_TARGET_MS:.0f} ms")
    if st["p95"] < T1_TARGET_MS:
        rep.emit(f"  结论          : PASS  余量 {T1_TARGET_MS / max(st['p95'], 1e-9):.1f}x")
    else:
        rep.emit(f"  结论          : FAIL  超标 {st['p95'] / T1_TARGET_MS:.1f}x")
    return st


def bench_t2(rep: Report, model_path: Path, window_size: int, threshold: float) -> dict[str, float]:
    """T2：ONNX 异常检测 is_anomaly（计时口径与 runtime 一致，不含滑窗 push）。"""
    detector = AnomalyDetector(model_path, window=window_size, threshold=threshold)
    window = SlidingWindow(window_size)
    rng = np.random.default_rng(7)

    def push_next() -> None:
        window.push("vibration_rms", float(rng.normal(3.0, 1.0)))

    for _ in range(T2_WARMUP):
        push_next()
        detector.is_anomaly(window, "vibration_rms")

    samples: list[float] = []
    for _ in range(T2_ITERS):
        push_next()
        t0 = time.perf_counter()
        detector.is_anomaly(window, "vibration_rms")
        samples.append((time.perf_counter() - t0) * 1000.0)

    st = _summary(samples)
    rep.emit("[T2 ONNX 异常检测路径 (参考)]")
    rep.emit(f"  后端          : {detector.backend}  模型: {model_path.name}")
    rep.emit(f"  迭代次数      : {T2_ITERS} (预热 {T2_WARMUP})")
    rep.emit(f"  mean          : {st['mean']:.4f} ms")
    rep.emit(f"  P95           : {st['p95']:.4f} ms")
    rep.emit(f"  min / max     : {st['min']:.4f} / {st['max']:.4f} ms")
    rep.emit(f"  目标          : P95 < {T2_TARGET_MS:.0f} ms")
    if st["p95"] < T2_TARGET_MS:
        rep.emit(f"  结论          : PASS  余量 {T2_TARGET_MS / max(st['p95'], 1e-9):.1f}x")
    else:
        rep.emit(f"  结论          : FAIL  超标 {st['p95'] / T2_TARGET_MS:.1f}x")
    return st


def bench_t4(rep: Report, db_path: Path, snippets_path: Path) -> dict[str, float]:
    """T4：本地知识检索（ONNX 语义嵌入 + 全表余弦；模型缺失时降级 hash-embed）。"""
    knowledge = LocalKnowledge(db_path, snippets_path)
    queries = [
        "驱动端轴承振动 RMS 越限",
        "轴承温度越限",
        "电机电流越限",
    ]
    for i in range(T4_WARMUP):
        knowledge.search(queries[i % len(queries)], top_k=1)

    samples: list[float] = []
    for i in range(T4_ITERS):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        knowledge.search(q, top_k=1)
        samples.append((time.perf_counter() - t0) * 1000.0)

    st = _summary(samples)
    rep.emit("[T4 本地知识检索路径 (参考)]")
    rep.emit(f"  后端          : {knowledge.backend}  top_k=1")
    rep.emit(f"  迭代次数      : {T4_ITERS} (预热 {T4_WARMUP})")
    rep.emit(f"  mean          : {st['mean']:.4f} ms")
    rep.emit(f"  P95           : {st['p95']:.4f} ms")
    rep.emit(f"  min / max     : {st['min']:.4f} / {st['max']:.4f} ms")
    rep.emit(f"  目标          : P95 < {T4_TARGET_MS:.0f} ms")
    if st["p95"] < T4_TARGET_MS:
        rep.emit(f"  结论          : PASS  余量 {T4_TARGET_MS / max(st['p95'], 1e-9):.1f}x")
    else:
        rep.emit(f"  结论          : FAIL  超标 {st['p95'] / T4_TARGET_MS:.1f}x")
    return st


def bench_t3(
    rep: Report,
    model_path: Path,
    max_tokens: int,
    num_ctx: int,
    window_size: int,
) -> dict[str, float] | None:
    """T3：端侧 LLM 真实推理（EdgeLLM.infer），使用生产诊断 prompt 构造逻辑。"""
    rep.emit("[T3 端侧 LLM 一级初判路径 (重点)]")
    rep.emit(f"  模型          : {model_path}")
    rep.emit(f"  max_tokens    : {max_tokens}  num_ctx: {num_ctx}  n_threads: 2 (代码固定)")

    if not model_path.exists():
        rep.emit(f"  结论          : FAIL  模型文件不存在: {model_path}")
        return None

    llm = EdgeLLM(str(model_path), max_tokens, num_ctx)
    if not llm.available:
        # EdgeLLM 内部吞掉异常，这里重新加载一次以捕获完整底层错误
        err = "unknown"
        try:
            from llama_cpp import Llama

            Llama(model_path=str(model_path), n_ctx=num_ctx, n_threads=2, verbose=False)
            err = "(直接加载成功但 EdgeLLM.available=False，疑似代码路径异常)"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        rep.emit("  结论          : FAIL  EdgeLLM 不可用，T3 无法实测")
        rep.emit(f"  加载错误      : {err}")
        return None

    # 复用生产 prompt 构造逻辑，保证是"真实诊断 prompt"
    chain = FaultChain(ROOT / "edge" / "knowledge" / "fault_chains.json")
    knowledge = LocalKnowledge(
        settings.data_dir / "bench_knowledge.db",
        ROOT / "edge" / "knowledge" / "snippets.json",
    )
    window = SlidingWindow(window_size)
    tools = build_tools(window, chain, knowledge)
    agent = EdgeAgentRuntime(tools, llm, chain, knowledge)

    hit = RuleHit(
        rule_id="bearing_rms_high",
        metric="vibration_rms",
        value=5.2,
        threshold=4.5,
        severity="high",
        description="驱动端轴承振动 RMS 越限",
    )
    hops = chain.two_hop(hit.rule_id)
    kb = knowledge.search(hit.description, top_k=1)

    def build_prompt() -> str:
        # ts 每次不同 -> prompt 唯一 -> 绕开 EdgeLLM 的 60s 缓存，测真实推理
        sample = InternalSample(
            device_id="pump-de-01",
            source="modbus",
            ts=datetime.now(UTC),
            metrics={"vibration_rms": 5.2, "temperature": 84.0, "current": 19.5},
        )
        return agent._build_prompt(sample, hit, hops, kb)

    # 预热一次（不计入）
    llm.infer(build_prompt())

    samples: list[float] = []
    ok = 0
    for _ in range(T3_ITERS):
        prompt = build_prompt()
        t0 = time.perf_counter()
        parsed = llm.infer(prompt)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if parsed is not None and "diagnosis" in parsed:
            ok += 1

    st = _summary(samples)
    rep.emit(f"  推理次数      : {T3_ITERS} (预热 1，prompt 每次唯一)")
    rep.emit(f"  有效 JSON 数  : {ok}/{T3_ITERS}")
    rep.emit(f"  mean          : {st['mean']:.1f} ms")
    rep.emit(f"  P50           : {st['p50']:.1f} ms")
    rep.emit(f"  P95           : {st['p95']:.1f} ms")
    rep.emit(f"  min / max     : {st['min']:.1f} / {st['max']:.1f} ms")

    # tok/s：底层 llama.cpp 调用取 usage.completion_tokens（infer 不返回 usage）
    tok_rates: list[float] = []
    comp_tokens: list[int] = []
    for _ in range(T3_TOK_SAMPLES):
        prompt = build_prompt()
        t0 = time.perf_counter()
        raw = llm.llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        dt = time.perf_counter() - t0
        ct = int((raw.get("usage") or {}).get("completion_tokens", 0) or 0)
        if dt > 0 and ct > 0:
            tok_rates.append(ct / dt)
            comp_tokens.append(ct)
    if tok_rates:
        avg_ct = statistics.fmean(comp_tokens)
        avg_tok_s = statistics.fmean(tok_rates)
        rep.emit(f"  输出 token    : 平均 {avg_ct:.1f} /次 (上限 {max_tokens})")
        rep.emit(f"  吞吐          : {avg_tok_s:.1f} tok/s ({T3_TOK_SAMPLES} 次底层采样)")
        # 用主循环平均延迟与平均 token 数反推吞吐，交叉校验
        if st["mean"] > 0:
            rate = avg_ct / (st["mean"] / 1000.0)
            rep.emit(f"  吞吐(校验)    : {rate:.1f} tok/s (按 infer 平均延迟反推)")
    else:
        rep.emit("  吞吐          : N/A (未能从 usage 取到 token 数)")
    rep.emit(f"  parse_rate    : {llm.parse_rate():.3f}  内部 p95_ms: {llm.p95_ms():.1f}")

    # 输出质量探查：打印一次原始生成文本，解释 parse_rate（延迟达标不代表语义有效）
    probe = build_prompt()
    raw_probe = llm.llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": probe}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    raw_text = str(raw_probe["choices"][0]["message"]["content"])
    rep.emit(f"  样本原始输出  : {raw_text[:200]!r}")

    rep.emit(f"  目标          : P95 < {T3_TARGET_MS:.0f} ms (2C 模拟)")
    if st["p95"] < T3_TARGET_MS:
        rep.emit(f"  结论          : PASS  余量 {T3_TARGET_MS / max(st['p95'], 1e-9):.2f}x")
    else:
        rep.emit(f"  结论          : FAIL  超标 {st['p95'] / T3_TARGET_MS:.2f}x")
    return st


def main() -> int:
    rep = Report()
    cfg = load_yaml(settings.config_path)
    window_size = int(cfg["sampling"]["vibration_window"])
    rules = cfg["rules"]
    anomaly_cfg = cfg["anomaly"]
    anomaly_model = settings.models_dir / "anomaly" / "current" / "model.onnx"

    rep.emit("=" * 78)
    rep.emit(" Palpation 分路径延迟基准实测报告 (bench_latency)")
    rep.emit("=" * 78)
    rep.emit(f" 生成时间      : {datetime.now(UTC).isoformat()}")
    rep.emit(" 目标口径      : 方案 §6.4 分路径 P95 预算")
    rep.emit("")
    rep.emit("[测试条件]")
    rep.emit(f"  CPU 型号      : {_cpu_model()}")
    rep.emit(f"  逻辑核数      : {os.cpu_count()}")
    rep.emit(f"  平台          : {platform.platform()}")
    rep.emit(f"  Python        : {sys.version.split()[0]}")
    rep.emit(f"  BLAS/OMP 线程 : {_BENCH_THREADS} (BENCH_THREADS，模拟 2C)")
    rep.emit("  T3 llama.cpp  : n_threads=2 (llm_llamacpp.py 硬编码)")
    rep.emit(f"  滑窗 size     : {window_size}   规则条数: {len(rules)}")
    rep.emit("  说明          : T1/T2/T4 为单核为主的小数组运算，20 核开发机与 2C 目标差异小；")
    rep.emit("                  T3 已固定 n_threads=2，但开发机单核性能可能优于目标 2C 边缘设备。")
    rep.emit("")

    st1 = bench_t1(rep, rules, window_size)
    rep.emit("")
    st2 = bench_t2(
        rep,
        anomaly_model,
        int(anomaly_cfg["window"]),
        float(anomaly_cfg["score_threshold"]),
    )
    rep.emit("")
    st4 = bench_t4(
        rep,
        settings.data_dir / "bench_knowledge.db",
        ROOT / "edge" / "knowledge" / "snippets.json",
    )
    rep.emit("")
    llm_model = Path(settings.edge_llm_model) if settings.edge_llm_model else Path("")
    if llm_model and not llm_model.is_absolute():
        llm_model = ROOT / llm_model
    st3 = bench_t3(
        rep,
        llm_model,
        int(settings.edge_llm_maxtok),
        int(settings.edge_llm_ctx),
        window_size,
    )
    rep.emit("")

    rep.emit("=" * 78)
    rep.emit(" 结论汇总")
    rep.emit("=" * 78)
    rep.emit(
        f"  T1 确定性告警  P95 = {st1['p95']:.3f} ms   目标 < {T1_TARGET_MS:.0f} ms   "
        f"{'PASS' if st1['p95'] < T1_TARGET_MS else 'FAIL'}"
    )
    rep.emit(
        f"  T2 ONNX 判别   P95 = {st2['p95']:.3f} ms   目标 < {T2_TARGET_MS:.0f} ms   "
        f"{'PASS' if st2['p95'] < T2_TARGET_MS else 'FAIL'}"
    )
    rep.emit(
        f"  T4 知识检索    P95 = {st4['p95']:.3f} ms   目标 < {T4_TARGET_MS:.0f} ms   "
        f"{'PASS' if st4['p95'] < T4_TARGET_MS else 'FAIL'}"
    )
    if st3 is not None:
        rep.emit(
            f"  T3 LLM 初判    P95 = {st3['p95']:.1f} ms   目标 < {T3_TARGET_MS:.0f} ms   "
            f"{'PASS' if st3['p95'] < T3_TARGET_MS else 'FAIL'}"
        )
    else:
        rep.emit(f"  T3 LLM 初判    N/A (无法实测)，目标 < {T3_TARGET_MS:.0f} ms")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(rep.text(), encoding="utf-8")
    rep.emit("")
    rep.emit(f" 原始输出已写入: {_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
