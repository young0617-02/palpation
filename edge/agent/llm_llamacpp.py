from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SYSTEM = (
    "你是边缘侧一级诊断助手。只输出一个 JSON 对象，不要使用花括号简写（如 {值1, 值2}）。"
    "必须标注：端侧初判，未经验证。"
    "禁止给出一键执行或远程停机指令。"
)


class EdgeLLM:
    """端侧 LLM 封装（llama.cpp），不可用时优雅降级到规则路径。

    公开接口：__init__ / infer / p95_ms / parse_rate / available。
    - 降级：模型缺失或加载失败时 available=False，infer 返回 None。
    - 缓存：相同 prompt 在 TTL 内复用初判（方案 §6.5 第6条）。
    - 线程安全：infer 可被并发调用，内部以锁串行化模型推理与指标更新。
    """

    # 缓存 TTL（秒），相同 prompt 在窗口内复用结果
    _CACHE_TTL_S: float = 60.0
    # 延迟样本窗口，仅保留最近 N 次用于 p95
    _LATENCY_WINDOW: int = 200

    def __init__(self, model_path: str, max_tokens: int = 32, num_ctx: int = 2048):
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.available = False
        self.llm: Any = None
        self.json_ok = 0
        self.json_total = 0
        self.latencies_ms: list[float] = []
        # prompt -> (monotonic_ts, result)；result 可为 None（负缓存避免重复失败）
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        # 指标/缓存锁；推理锁单独分离以避免持锁等待模型
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

        path = Path(model_path) if model_path else None
        if path and path.exists():
            try:
                from llama_cpp import Llama

                self.llm = Llama(model_path=str(path), n_ctx=num_ctx, n_threads=2, verbose=False)
                self.available = True
                log.info("edge llm ready: %s", path)
            except Exception:
                log.exception("failed to load llama.cpp model: %s", path)
        else:
            log.info("EDGE_LLM_MODEL not set or missing; T3 uses rule-side fallback")

    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        xs = sorted(self.latencies_ms)[-self._LATENCY_WINDOW :]
        idx = int(0.95 * (len(xs) - 1))
        return xs[idx]

    def parse_rate(self) -> float:
        if self.json_total == 0:
            return 1.0
        return self.json_ok / self.json_total

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        """从模型输出中提取首个 JSON 对象，解析失败返回 None。

        先做严格 json.loads；失败时做轻量修正（去尾逗号、单引号改双引号）后再试。
        不做 set 风格的位置猜测，避免引入脆弱启发式。
        """
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # 轻量容错：去掉对象/数组闭合符前的多余逗号，并把单引号字符串改为双引号
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    def infer(self, prompt: str) -> dict[str, Any] | None:
        """运行一次端侧推理并解析 JSON。不可用或失败时返回 None。"""
        if not self.available or self.llm is None:
            return None

        # 缓存命中：TTL 内直接复用，不计入推理指标
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(prompt)
            if cached and now - cached[0] < self._CACHE_TTL_S:
                return cached[1]

        result: dict[str, Any] | None = None
        t0 = time.perf_counter()
        with self._lock:
            self.json_total += 1
        try:
            # llama_cpp.Llama 非线程安全，推理需串行化。
            # 使用 chat completion 以套用 Qwen2.5-Instruct 的 chat template，
            # 否则 completion 接口无法理解对话格式，只会复述 prompt。
            with self._infer_lock:
                out = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                )
            text = out["choices"][0]["message"]["content"]
            result = self._parse_json(text)
            if result is not None:
                with self._lock:
                    self.json_ok += 1
        except Exception:
            log.warning("llm infer failed; prompt head=%r", prompt[:80])
            result = None
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.latencies_ms.append(elapsed_ms)
                # 写入缓存（含负缓存：失败结果也缓存，避免 TTL 内重复失败开销）
                self._cache[prompt] = (now, result)
        return result
