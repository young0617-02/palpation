#!/usr/bin/env python
"""准备 bge-small-zh-v1.5 ONNX(int8) 知识检索模型资产。

两种方式（离线复用优先）：
  1) 离线复用：``uv run python scripts/download_bge_onnx.py --from-local .bench_rag_onnx/model``
  2) 联网下载（走 HF 镜像）：``uv run python scripts/download_bge_onnx.py``

输出目录默认为 ``models/knowledge/current/``（对齐 ``models/anomaly/current/`` 惯例），
可用 ``--out`` 覆盖。复制/下载完成后可选 ``--verify``：加载 tokenizer 与 ONNX session
各一次，打印向量维度与 L2 范数，提前暴露资产不完整问题。

依赖：仅标准库（``urllib`` / ``shutil``）；仅 ``--verify`` 需要项目已安装
``tokenizers`` 与 ``onnxruntime``。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Xenova/bge-small-zh-v1.5"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUT = ROOT / "models" / "knowledge" / "current"
DEFAULT_ONNX = "onnx/model_quantized.onnx"

# 需要落地的文件（相对镜像仓库根）。model_quantized.onnx 为 int8 量化权重。
FILES: tuple[str, ...] = (
    DEFAULT_ONNX,
    "tokenizer.json",
    "vocab.txt",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

# 已知期望字节数（用于对账 evidence/rag_onnx_bench.md §2；未列出则不校验大小）。
EXPECTED_BYTES: dict[str, int] = {
    DEFAULT_ONNX: 24010842,
}


def _expected(rel: str) -> int | None:
    """返回相对路径对应的期望字节数（无约定则返回 None）。"""
    return EXPECTED_BYTES.get(rel)


def _copy_from_local(src_dir: Path, out_dir: Path, force: bool) -> list[tuple[str, int, str]]:
    """从本地已下载目录复制资产；返回 [(相对路径, 字节数, 来源说明)]。"""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"--from-local 源目录不存在: {src_dir}")
    result: list[tuple[str, int, str]] = []
    for rel in FILES:
        src = src_dir / rel
        if not src.is_file():
            raise FileNotFoundError(f"源资产缺失: {src}")
        dst = out_dir / rel
        if dst.exists() and not force and dst.stat().st_size == src.stat().st_size:
            result.append((rel, dst.stat().st_size, f"skip(已存在) <- {src}"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        result.append((rel, dst.stat().st_size, f"copy <- {src}"))
    return result


def _download_one(
    url: str, rel: str, dst: Path, force: bool, retries: int = 3
) -> tuple[str, int, str]:
    """下载单个文件；已存在且大小匹配则跳过；失败按指数退避重试。"""
    exp = _expected(rel)
    if dst.exists() and not force and (exp is None or dst.stat().st_size == exp):
        return rel, dst.stat().st_size, "skip(已存在)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = dst.parent / (dst.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(dst)
            return rel, dst.stat().st_size, f"download <- {url}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"下载失败 {url}: {last_err}")


def _download_all(
    endpoint: str, repo: str, out_dir: Path, force: bool
) -> list[tuple[str, int, str]]:
    """按文件清单依次下载到 out_dir。"""
    result: list[tuple[str, int, str]] = []
    for rel in FILES:
        url = f"{endpoint}/{repo}/resolve/main/{rel}"
        result.append(_download_one(url, rel, out_dir / rel, force))
    return result


def _verify(out_dir: Path, onnx_rel: str, max_length: int = 512) -> None:
    """加载 tokenizer 与 ONNX session，校验向量维度与 L2 归一化。"""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0
    opts.enable_cpu_mem_arena = True
    session = ort.InferenceSession(
        str(out_dir / onnx_rel), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_names = {i.name for i in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    sample = "驱动端轴承振动 RMS 越限，温度升高怎么处理"
    enc = tokenizer.encode(sample)
    print(f"  tokenizer ids[:5]  = {enc.ids[:5]}")
    print(f"  tokenizer ids[-3:] = {enc.ids[-3:]}")
    if enc.ids[:1] != [101] or enc.ids[-1:] != [102]:
        print("  [warn] 未自动补 [CLS]/[SEP]（101/102），语义质量可能下降")

    ids = np.asarray([enc.ids], dtype=np.int64)
    mask = np.asarray([enc.attention_mask], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    feeds = {k: v for k, v in feeds.items() if k in input_names}
    out = session.run([output_name], feeds)[0]
    cls = out[:, 0, :]
    vec = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
    dim = int(vec.shape[-1])
    norm = float(np.linalg.norm(vec[0]))
    print(f"  vec_dim = {dim}")
    print(f"  vec_norm = {norm:.6f}")
    if dim != 512:
        raise SystemExit(f"[fail] 向量维度应为 512，实际 {dim}")
    if abs(norm - 1.0) > 1e-3:
        raise SystemExit(f"[fail] L2 范数应接近 1.0，实际 {norm:.6f}")
    print("  [ok] 资产自检通过")


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 bge-small-zh-v1.5 ONNX(int8) 模型资产")
    parser.add_argument("--from-local", type=Path, default=None, help="从本地已下载目录复制")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="目标目录")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF 仓库 id")
    parser.add_argument("--onnx", default=DEFAULT_ONNX, help="ONNX 文件相对路径")
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT), help="HF 镜像端点"
    )
    parser.add_argument("--force", action="store_true", help="强制覆盖已有文件")
    parser.add_argument("--verify", action="store_true", help="加载模型自检")
    args = parser.parse_args()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bge-onnx] 目标目录: {out_dir}")

    try:
        if args.from_local is not None:
            print(f"[bge-onnx] 模式: --from-local {args.from_local}")
            report = _copy_from_local(args.from_local, out_dir, args.force)
        else:
            endpoint = args.endpoint.rstrip("/")
            print(f"[bge-onnx] 模式: 下载 repo={args.repo} endpoint={endpoint}")
            report = _download_all(endpoint, args.repo, out_dir, args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[bge-onnx][fail] {exc}", file=sys.stderr)
        return 1

    print("[bge-onnx] 文件清单:")
    all_ok = True
    for rel, size, src in report:
        exp = _expected(rel)
        if exp is None or size == exp:
            flag = "ok"
        else:
            flag = f"MISMATCH(exp={exp})"
            all_ok = False
        print(f"  {size:>12,} B  [{flag}] {rel}  ({src})")

    if args.verify:
        print("[bge-onnx] --verify 自检:")
        try:
            _verify(out_dir, args.onnx)
        except ImportError as exc:
            print(f"[bge-onnx][warn] --verify 需要项目依赖 tokenizers/onnxruntime: {exc}")
            return 1
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
