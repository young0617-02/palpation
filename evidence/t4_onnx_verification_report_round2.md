# T4 ONNX 检索 — 修复后重验报告（round 2）

- 任务：team task id=30（T4-RAG 验证工程师）
- 前次判定：task id=26 → **PARTIAL**（2 项未闭合：§4.1 元素级脏数据崩溃、§4.2 check.sh 被 bench 目录污染）
- 本次对象：任务 29 的修复（`edge/knowledge/sqlite_vec.py` `_load_matrix` 加固 + `.gitignore` 忽略 `.bench_rag_*/`、`.resultverify/`）
- 日期：2026-10-22
- 约束：`uv run --no-sync`（不用 pip）；未修改任何被测文件；验证在临时目录执行

---

## 0. 最终判定

> **VERDICT: PASS**

前次 2 项未闭合**均已闭合**，且全部回归项**无退化**：`check.sh` 原文 RC=0 全绿、全量 932 passed、真实语义确凿生效、降级 8/8、T4 延迟 P95=6.64ms ≪ 80ms、无禁依赖。

---

## 1. 修复确认（代码 diff）

`edge/knowledge/sqlite_vec.py` `_load_matrix()`：
```python
def _is_number(x: object) -> bool:
    """向量元素是否为可转 float 的数值标量（显式排除 bool）。"""
    return isinstance(x, (int, float)) and not isinstance(x, bool)

# _load_matrix 内：
for v in vectors:
    if len(v) != self.dim:
        return False
    if not all(_is_number(x) for x in v):   # 元素级校验：拦截字符串/嵌套/混合/bool 数组
        return False
try:                                         # np.asarray 兜底，异常不冒泡出 __init__
    matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, self.dim), np.float32)
except (ValueError, TypeError):
    return False
# 全部校验通过后再赋值，避免 _docs/_matrix 半更新
```
`.gitignore` 追加：
```
+.bench_rag_torch/
+.bench_rag_onnx/
+.resultverify/
```

---

## 2. §4.1 闭合证据（重跑前次暴露缺陷的对抗用例）

`rv_c1b.py`（hash 后端 dim=32，注入脏 embedding 后重开库）：
```
[scalar_null           ] backend=hash-embed dim=32 hits=3 rebuilt=True -> OK
[scalar_int            ] ... -> OK
[scalar_bool           ] ... -> OK
[json_object           ] ... -> OK
[json_string           ] ... -> OK
[empty_list            ] ... -> OK
[short_list            ] ... -> OK
[null_array_matchdim   ] backend=hash-embed dim=32 hits=3 rebuilt=True -> OK   # 前次“静默”项，现已正确重建
[bool_array_matchdim   ] ... -> OK
[str_array_matchdim    ] backend=hash-embed dim=32 hits=3 rebuilt=True -> OK   # 前次崩溃 ❌→✅
[nested_array_matchdim ] backend=hash-embed dim=32 hits=3 rebuilt=True -> OK   # 前次崩溃 ❌→✅
[mixed_array           ] backend=hash-embed dim=32 hits=3 rebuilt=True -> OK   # 前次崩溃 ❌→✅

>> §4.1 重验汇总: ALL OK (12 cases)
```
结论：**12/12 不再抛异常**，全部优雅降级 `hash-embed(dim=32)`、`search` 返回 3 条、脏行被合法向量覆盖（`rebuilt=True`）。前次 3 个崩溃用例与 1 个静默项均已修复。

新增回归测试同步覆盖（`tests/test_knowledge.py`）：
```
test_non_numeric_embedding_triggers_rebuild_not_crash[str_array]   PASSED
test_non_numeric_embedding_triggers_rebuild_not_crash[nested_array] PASSED
test_non_numeric_embedding_triggers_rebuild_not_crash[mixed_array]  PASSED
============================= 27 passed in 8.00s ==============================
```

## 3. §4.2 闭合证据

```
$ uv run --no-sync ruff check .
All checks passed!                              # RC=0；.bench/.resultverify 命中数=0
$ uv run --no-sync ruff format --check . --exclude tests --exclude plan
85 files already formatted                      # RC=0
$ bash scripts/check.sh
=== ruff check ===        All checks passed!
=== ruff format --check === 85 files already formatted
=== mypy ===              Success: no issues found in 64 source files
=== pytest ===            932 passed, 8 warnings in 143.31s
=== ALL CHECKS PASSED ===                       # RC=0
```
前次 `Found 186659 errors` 的遗留目录污染已消除。

---

## 4. 回归确认（不得退化）

| 项 | 命令/证据 | 结果 |
|---|---|---|
| 全量 pytest | `.venv/Scripts/python.exe -m pytest -q` | **932 passed**（RC=0，基线一致）|
| ruff check（源码） | `ruff check edge cloud mgmt_agent shared sim scripts` | All checks passed |
| ruff format --check（源码）| `ruff format --check ...` | 69 files already formatted |
| mypy | `mypy cloud edge mgmt_agent shared sim` | Success 64 files |
| 真实语义生效 | `rv_a.py` | backend=onnx-bge-small-zh, dim=512, matrix(30,512) |
| 语义锚点 | 3 query | top1 全命中（kb-bearing-rms / kb-temp-current / kb-bearing-rms），与 plan §3.7 一致 |
| 非降级伪装 | ONNX vs hash top3 | 3/3 differ=YES |
| embedder 直测 | encode | (2,512) float32 norm=1 |
| 降级链路 | `rv_b.py` | **8/8 OK**（缺失/None/disabled/degrade_false/依赖缺失/onnx截断/tokenizer缺失/onnx缺失）|
| T4 延迟 | `rv_d.py` N=300 top_k=1 | ONNX **P95=6.64ms** < 80ms；hash P95=0.03ms |
| 并发 | `rv_c2.py` 16线程×60 | errors=0 mismatches=0 |
| 幂等 | 重复打开 | 结果一致 YES，meta 不变 |
| 维度迁移 | hash↔onnx 交替 | all_rows_512=True / all_rows_32=True |
| snippets 自愈 | `rv_f.py` | sha 变化 → 重建 PASS |
| 禁依赖 | uv.lock + 环境 | torch/transformers/sentence-transformers **均缺席** |

---

## 5. 残留观察（非阻塞，不改变 PASS 判定）

1. `search(q, top_k=-1)` 返回 n-1 条（`np.argsort(...)[:-1]`）：仓内调用点均传正数，属契约外行为，非本次改动范围。
2. 极端 `NaN/Infinity` 数值数组仍会通过 `_is_number` 校验（`float('nan')` 是 float），但 `json.dumps` 默认不产生 NaN，需外部构造，可达性极低；不影响正常路径。
3. 本报告的验证在临时目录进行；模型资产位于 `models/knowledge/current/`（受 `.gitignore` 保护）。CI 无模型时自动走 hash 降级（已由 `rv_b.py` 覆盖）。

## 6. 未验证点（如实声明）

1. 仍未在 **2C4G 容器**内复测本改动延迟/RSS；本机 P95=6.64ms 与 `evidence/rag_onnx_bench.md` §4（容器 2C arena ON P95=5.78ms）方向一致。
2. 未走真实 MQTT/端到端链路触发 `edge_t4_ms`（T4_HIST）打点；延迟以 `LocalKnowledge.search` 直接复刻 `runtime.py:277` 参数测得。集成路径由既有全量测试（含 e2e）间接覆盖。

---

## 7. 结论

- §4.1、§4.2 **均已闭合**，且无任何回归或新增缺陷。
- 综合判定：**VERDICT: PASS**。
- 报告位置：`evidence/t4_onnx_verification_report_round2.md`
- 原始输出：`/tmp/rv/{verify_c1b,verify_a2,verify_b2,verify_d2,check_sh,ruff_check_ws,ruff_fmt_ws}.txt`