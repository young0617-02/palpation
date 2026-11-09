# T4 端侧知识检索（ONNX 真实语义）独立验证报告

- 任务：team task id=26（T4-RAG 验证工程师 / team-result-verifier）
- 验证对象：T4 从 `hash-embed` 占位升级为 **onnxruntime + bge-small-zh-v1.5 ONNX(int8) 真实语义检索（512 维）**，并保留 hash-embed 优雅降级（含残留脏数据健壮性修复）
- 验证日期：2026-10-22
- 场景识别：`new-feature`（写入 `.resultverify/verification_scenario.json`；本环境未注册 `scenario-recognition`/`new-feature-verification` 技能，按角色 Step1 语义自行完成识别并留痕）
- 约束遵守：全程 `uv run --no-sync` / `.venv/Scripts/python.exe`，**未使用 pip**；未修改任何被测文件；验证均在临时目录进行

---

## 0. 结论（VERDICT）

> **VERDICT: PARTIAL**

- **核心目标 100% 达成**：真实语义检索确凿生效（非降级伪装）、512 维、降级链路 8/8 场景健壮、接口不变、无回归（929 passed）、T4 延迟 P95=3.58ms ≪ 80ms、交付源码静态检查全绿、无禁依赖。
- **未完全达成 2 项**（详见 §4）：
  1. **【真实缺陷·低危】** 元素级脏 embedding（字符串数组 / 嵌套数组 / 混合数组，且长度恰好等于 dim）仍会使 `LocalKnowledge.__init__` 抛 `ValueError` 崩溃 —— 与代码注释 L252「保证 `__init__` 绝不因索引问题抛出」直接矛盾，属「残留脏数据健壮性修复」未闭合的**同类漏洞**（审查 M1 只覆盖了 JSON 标量）。
  2. **【环境性·非本次交付代码】** `bash scripts/check.sh` 原文的 `ruff check .` / `ruff format --check .` 因**未跟踪的遗留 bench 产物** `.bench_rag_torch/docker_deps/torch/**` 报 **186659 errors**（RC=1），导致「check.sh 全绿」这一明列验收条件在当前工作区不成立。对**本次交付源码**单独执行静态检查则全绿。

---

## 1. 执行流程与方法

| 阶段 | 方法 | Oracle 类型 |
|---|---|---|
| 静态检查 | ruff check / ruff format --check / mypy（交付源码 + 工作区两种口径） | 全绿断言 |
| 回归 | pytest 全量 + test_knowledge.py 逐用例 | 基线一致（929） |
| 功能A | 独立脚本构造 `LocalKnowledge`，就绪态断言 + 与 hash 后端 top3 对照 + embedder 直测 | 语义 Oracle + 差异化对照 |
| 功能B | 8 种降级场景（缺失/None/disabled/依赖缺失/资产损坏） | 降级断言（backend=hash, dim=32, 不抛异常） |
| 对抗 | 11 类脏 embedding 类型矩阵 / 边界 / 并发 / 幂等 / 迁移 | 崩溃探测 |
| 性能 | N=300 次 `search(top_k=1)` 复刻 `runtime.py:277`，测 P50/P95/P99/max | P95 < 80ms |
| 接线 | 复刻 `runtime.py:124-141` 解析 `config/current.yaml` 构造 | 非死配置断言 |
| 资产 | `download_bge_onnx.py --from-local --verify` | 字节对账 + 维度/范数 |

复现脚本：`<TMPDIR>/rv_{a,b,c1,c2,d,e,f}.py`；原始输出：`/tmp/rv/verify_{a,b,c1,c2,d,e,f}.txt`。

---

## 2. 通过项（含原始证据）

### 2.1 全量回归基线
```
$ .venv/Scripts/python.exe -m pytest -q
929 passed, 8 warnings in 102.35s (0:01:42)        # RC=0，无失败/无回归
```
`tests/test_knowledge.py` 24/24 PASSED，**真实语义用例未 skip**（模型就绪路径确实执行）：
```
test_onnx_backend_and_dim PASSED ... test_dimension_migration_onnx_to_hash PASSED
============================= 24 passed in 4.77s ==============================
```

### 2.2 静态检查（交付源码口径）
```
$ uv run --no-sync ruff check edge cloud mgmt_agent shared sim scripts
All checks passed!                                 # RC=0
$ uv run --no-sync ruff format --check edge cloud mgmt_agent shared sim scripts
69 files already formatted                         # RC=0
$ .venv/Scripts/python.exe -m mypy cloud edge mgmt_agent shared sim
Success: no issues found in 64 source files        # RC=0
```

### 2.3 禁依赖排查
```
$ grep -inE '^name = "(torch|transformers|sentence-transformers)"' uv.lock  -> NONE
$ python -c "importlib.find_spec" torch/transformers/sentence_transformers -> absent
uv.lock 含 tokenizers（新增，符合决策）          # 合规
```

### 2.4 真实语义检索确凿生效（非降级伪装）
```
backend='onnx-bge-small-zh' dim=512 embedder=ONNX
matrix.shape=(30, 512) docs=30
stored embedding len=512
>> 模型就绪断言: PASS
```
语义相关性（与 plan §3.7 验收锚点一致）：
```
Q='离心泵轴承振动越限与润滑不良' top1=kb-bearing-rms  scores=[0.8546,0.7167,0.672]  gap=0.1379  OK
Q='单纯电机电流升高是什么原因'   top1=kb-temp-current scores=[0.6720,0.6390,0.592]  gap=0.0329  OK
Q='驱动端轴承温度升高怎么办'     top1=kb-bearing-rms  scores=[0.7352,0.6860,0.6860] gap=0.0493  OK
```
**差异化对照（证明不是降级伪装）** —— 同一 query 的 ONNX 与 hash 后端 top3 完全不同：
```
Q='单纯电机电流升高是什么原因'
  onnx=['kb-temp-current','kb-conveyor-motor-vib','kb-motor-stator-overheat']
  hash=['kb-gearbox-hs-cage','kb-pump-impeller-wear','kb-pump-nd-seal']   differ=YES
>> 结果可区分（非伪装）: PASS
OnnxEmbedder 直测: shape=(2,512) dtype=float32 norms=[1. 1.]  PASS
```

### 2.5 降级链路 8/8
```
[missing_dir]   hash-embed dim=32 hits=3  OK      [model_none]      hash-embed  OK
[disabled]      hash-embed dim=32 hits=3  OK      [degrade_false]   hash-embed  OK
[deps_missing]  hash-embed dim=32 hits=3  OK（模拟 tokenizers/onnxruntime import 失败）
[onnx_truncated] 降级 hash（ONNXRuntimeError INVALID_PROTOBUF） OK
[tokenizer_missing] tokenizer.json 不存在 -> 降级 OK
[onnx_missing]  ONNX 模型不存在 -> 降级 OK
>> 降级路径汇总: 8/8 OK
```

### 2.6 性能
```
== T4 search 延迟（N=300，top_k=1，复刻 runtime 调用）==
[onnx] backend=onnx-bge-small-zh dim=512  P50=2.05ms P95=3.58ms P99=4.36ms max=5.41ms mean=2.20ms
[hash] backend=hash-embed dim=32          P50=0.01ms P95=0.03ms P99=0.10ms max=0.14ms
>> ONNX P95=3.58ms 目标<80ms : PASS       （余量 ~22×）
```

### 2.7 对抗性探测通过
```
== 并发 == 60 并发检索: errors=0 mismatches=0 -> OK   （16 线程 × 60 次，结果与串行一致）
== 幂等 == 重复打开 backend/dim/meta 不变，结果一致: YES
== 迁移 == hash->onnx->hash 同一库：all_rows_512=True / all_rows_32=True，meta 同步
== 边界 == top_k=0->[]; top_k=1->1; top_k=1000->30；空/空格/超长(10k)/emoji/数字 query 均正常
== 自愈 == snippets 内容变更 -> snippets_sha 变化并重建 PASS
== 配置 == yaml(knowledge.model_dir=models/knowledge/current) -> backend=onnx dim=512 PASS（非死配置）
== 资产 == download_bge_onnx.py --from-local --verify:
  24,010,842 B [ok] onnx/model_quantized.onnx    tokenizer ids[:5]=[101,7721,1220,4999,6766]
  439,125 B tokenizer.json                        ids[-3:]=[1905,4415,102]
  vec_dim=512  vec_norm=1.000000  [ok] 资产自检通过   RC=0
```

---

## 3. 对抗性探测结果明细（脏 embedding 类型矩阵）

对 `snippets.embedding` 注入 11 类数据后重开库（hash 后端，dim=32）：

| 注入值 | 结果 |
|---|---|
| `null` / `123` / `true`（JSON 标量） | 正常重建，不崩 ✅ |
| `{"a":1}`（object）/ `"abc"`（string） | 非 list → 正常重建 ✅ |
| `[]`（空数组）/ `[1.0,2.0]`（短数组） | len≠dim → 正常重建 ✅ |
| `[null]*32`（null 数组，长度=dim） | 不崩（matrix 含 nan，检索未抛异常）⚠️ 静默 |
| `["a"]*32`（字符串数组，长度=dim） | **`__init__` 抛 `ValueError: could not convert string to float: 'a'`** ❌ |
| `[[1.0,2.0]]*32`（嵌套数组，长度=dim） | **`__init__` 抛 `ValueError: setting an array element with a sequence…`** ❌ |
| `[1.0×31, "x"]`（混合数组，长度=dim） | **`__init__` 抛 `ValueError: could not convert string to float: 'x'`** ❌ |

**定位**：`edge/knowledge/sqlite_vec.py` → `_load_matrix()`
- L220 只校验 `isinstance(v, list)`；
- L222 只校验 `len(v) != self.dim`；
- L225-229 `np.asarray(vectors, dtype=np.float32)` **未捕获**元素类型非数值/嵌套抛出的 `ValueError`，异常冒泡出 `__init__`（L160 `self._rebuild_index` → L237 `_load_matrix`），使网关启动失败。

**可达性评估（低）**：正常数据流中 embedding 恒为 `json.dumps(float list)`（`_embed_text` → `float(x)`），该崩溃需**外部篡改 db 文件**或异常存储损坏才可触发。但 `_load_matrix` 的 docstring 与 L252 注释均承诺「内容不可用→返回 False」「绝不因索引问题抛出」，属**承诺未闭合**；审查 M1 已识别同类（JSON 标量），本类为同一逻辑漏网。

---

## 4. 未通过 / 未完全达成项

### 4.1 ❌ 元素级脏 embedding 导致启动崩溃（真实缺陷，低危）
- 证据：§3 表格；复现：`rv_c1.py`（`str_array_matchdim` / `nested_array_matchdim` / `mixed_array` 三例 INIT EXCEPTION）。
- 影响：`LocalKnowledge.__init__` 崩溃 → 网关启动失败（但正常路径不可达）。
- 违反：`sqlite_vec.py` L252 显式承诺「保证 `__init__` 绝不因索引问题抛出」。
- 建议修复（1 行级加固）：`_load_matrix` 在 `len` 校验后追加「元素必须为数值标量」校验（如 `all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v)`），非法即返回 `False` 触发重建；并将 `np.asarray(...)` 包裹 try/except 兜底。

### 4.2 ❌ `check.sh` 全绿条件在当前工作区不成立（环境性，非本次交付代码）
```
$ uv run --no-sync ruff check .
warning: Invalid `# noqa` directive on .bench_rag_torch\docker_deps\torch\...\nested_tensor.py
...
Found 186659 errors.                              # RC=1
```
- 根因：未跟踪目录 `.bench_rag_torch/docker_deps/torch/**`（路线 A 实测遗留的 torch 源码拷贝）与 `.bench_rag_onnx/`，`git check-ignore` 显示**均未被 `.gitignore` 忽略**，`ruff` 默认扫描（`pyproject` 仅 exclude `tests`/`plan`）。
- 归因：该目录为任务 20（路线 A 资源实测）临时产物，非任务 23/24/28 交付内容；**交付源码本身静态检查全绿**（§2.2）。
- 建议：将 `.bench_rag_*/` 加入 `.gitignore` 或清理，使 `scripts/check.sh` 原文可全绿。

### 4.3 ⚠️ 边界观察（非缺陷）
- `search(q, top_k=-1)` 返回 29/30 条（`np.argsort(...)[:-1]`）：仅当调用方传入负值才出现，仓内调用点均传正数，属契约外行为，记录不改判。

---

## 5. 不确定 / 未能验证的点（如实声明）

1. **未在 2C4G 容器内实测**本改动的 T4 延迟与 RSS；本机 P95=3.58ms 与 `evidence/rag_onnx_bench.md` §4（容器 2C arena ON 1000 条 P95=5.78ms）方向一致，但容器侧本改动未复测。
2. **未通过真实 MQTT/端到端链路触发 T4_HIST**（`edge_t4_ms`）打点；仅以 `LocalKnowledge.search` 直接调用复刻 `runtime.py:277` 的路径与参数测得延迟。集成链路由既有全量测试（含 e2e）间接覆盖。
3. 本机为共享开发机，延迟绝对值存在环境噪声；已 warmup 20 次、N=300，P95 稳定。
4. §3 的 `[null]*32` 场景检索未抛异常但可能产生 nan 分数（未深入定位排序退化程度），列为观察。

---

## 6. 复现步骤（关键）

```bash
# 静态检查（交付源码口径）
uv run --no-sync ruff check edge cloud mgmt_agent shared sim scripts
uv run --no-sync ruff format --check edge cloud mgmt_agent shared sim scripts
.venv/Scripts/python.exe -m mypy cloud edge mgmt_agent shared sim
# 全量回归
.venv/Scripts/python.exe -m pytest -q          # 期望 929 passed
# 独立验证（脚本见 §1）
.venv/Scripts/python.exe "$TEMP\rv_a.py"   # 功能A
.venv/Scripts/python.exe "$TEMP\rv_b.py"   # 降级B
.venv/Scripts/python.exe "$TEMP\rv_c1.py"  # 脏数据对抗（暴露 §4.1）
.venv/Scripts/python.exe "$TEMP\rv_d.py"   # 延迟
```

---

## 7. 结论与交付建议

- **核心改动确凿达成**：ONNX 真实语义检索已生效（有差异化对照硬证据）、512 维、降级完备、无回归、延迟达标、无禁依赖。**主功能可交付。**
- **建议在合入前补 2 处**：(a) `_load_matrix` 元素数值校验 + `np.asarray` 异常兜底（闭合 §4.1，约 3 行）；(b) `.gitignore` 忽略 `.bench_rag_*/`（恢复 `check.sh` 全绿，§4.2）。
- 因存在上述 1 个真实（低危）缺陷与 1 个阻塞性验收条件未满足，本报告判定 **PARTIAL** 而非 PASS；二者均不影响正常功能路径，整改成本低。