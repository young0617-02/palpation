# 路线 B 资源实测报告：onnxruntime + bge-small-zh-v1.5 ONNX(int8)

> 目标：为端侧 T4 知识检索（当前 `edge/knowledge/sqlite_vec.py` 为 hash-embed 32 维占位）评估
> **不引入 PyTorch** 的真实语义检索方案在 **2 核 4G** 下的资源可行性。
> 全部数字来自真实执行，脚本与原始输出可复核（见文末原始文件清单）。

- 任务 id：21（owner：RAG-ONNX实测工程师）
- 执行日期：2026-10-22
- 主机：Windows 11 (10.0.26200)，20 逻辑核 / 14 物理核 / 32560 MiB RAM，Docker 29.6.2（Linux/WSL2 引擎，10 cpu / 20001 MiB）
- 依赖管理：`uv` 0.11.31（全程未使用 pip）；独立 venv 位于 `.bench_rag_onnx/venv`（**未污染项目 `.venv`，未改 `pyproject.toml` / `uv.lock`**）

---

## 0. 结论速览

| 维度 | 结果 | 判定 |
|---|---|---|
| 磁盘 · int8 权重 | **22.90 MiB**（`model_quantized.onnx` 24,010,842 B） | 极省 |
| 磁盘 · 依赖(site-packages) | **174 MiB**（本机）/ 231 MB（容器） | 远小于路线 A 的 768 MB |
| 冷启动 RSS 增量 | **+46.8 ~ +47.8 MiB**（跨本机/容器高度一致） | 稳定 |
| 单条推理峰值增量 | **+0.0 ~ +2.0 MiB** | 极小 |
| 批量32 峰值增量 | +65.4 MiB(arena off) / +91.6 MiB(arena on) | 一次性，可接受 |
| 检索延迟 P95 · 2 条 | 5.44 ms（本机2线程） | 远低于 80 ms 目标 |
| 检索延迟 P95 · 1000 条 | **5.78 ms**（容器 2C，n=100） | 远低于 80 ms 目标 |
| Docker `--cpus=2 --memory=4g` | **加载成功、无 OOM**，进程峰值 344.6 MiB | PASS |
| 叠加现有 edge 后合计 | **≈ 763 MiB**（同进程），余量 ≈ **3.25 GiB** | 远 < 4G |
| **2C4G 路线 B 可行性** | **可行**，且余量充足 | ✅ |

---

## 1. 环境与复现命令

### 1.1 版本

| 组件 | 版本 |
|---|---|
| Python | 3.12.8（本机）/ 3.12.15（容器） |
| onnxruntime | 1.31.0（与项目 T2 一致） |
| transformers | 5.20.2 |
| tokenizers | 0.23.3 |
| huggingface_hub | 2.2.0 |
| numpy | 2.5.4 |
| psutil | 7.2.2 |
| **torch** | **未安装（无 torch 依赖）** |

模型来源：`Xenova/bge-small-zh-v1.5`（社区 ONNX 转换仓库，HF 镜像 `hf-mirror.com` 下载），
选用 `onnx/model_quantized.onnx`（int8 量化，transformers.js 标准量化版）。
模型签名：输入 `input_ids/attention_mask/token_type_ids`（int64），输出 `last_hidden_state [B,S,512]`，
采用 **CLS pooling + L2 normalize**（BGE 官方用法）。

### 1.2 复现命令

```bash
# 1) 独立 venv（不污染项目 .venv）
uv venv .bench_rag_onnx/venv --python 3.12
uv pip install --python .bench_rag_onnx/venv/Scripts/python.exe \
    onnxruntime==1.31.0 transformers tokenizers huggingface_hub numpy psutil

# 2) 下载 int8 模型 + tokenizer
HF_ENDPOINT=https://hf-mirror.com .bench_rag_onnx/venv/Scripts/python.exe .bench_rag_onnx/_download.py

# 3) 本机完整测量（2 线程贴近 2C 目标；--no-arena 为对照）
.bench_rag_onnx/venv/Scripts/python.exe scripts/bench_rag_onnx.py \
    --iters 30 --warmup 5 --corpus-sizes 2 1000 --threads 2 \
    --json-out evidence/raw/bench_rag_onnx_t2_arena.json

# 4) edge 基线（用项目 .venv，只读加载）
.venv/Scripts/python.exe .bench_rag_onnx/edge_baseline.py

# 5) Docker 2C4G 约束验证
docker build -t rag-onnx-bench:1.0 .bench_rag_onnx/ctx
docker run --rm --cpus=2 --memory=4g --memory-swap=4g rag-onnx-bench:1.0 \
    python /app/scripts/bench_rag_onnx.py --docker --model-dir /app/.bench_rag_onnx/model --iters 100
```

> 构建上下文 `.bench_rag_onnx/ctx` 为最小化副本（24 MB），避免把 `models/llm/*.gguf`(469 MB) 送入构建。
> 容器内先由官方脚本引导安装 `uv`，再用 `uv pip install --system` 装依赖（未使用 pip 装项目依赖）。

---

## 2. 磁盘占用

| 项目 | 大小 | 说明 |
|---|---|---|
| `onnx/model_quantized.onnx` (int8) | **22.90 MiB**（24,010,842 B） | 实测 |
| 模型目录合计（含 tokenizer） | **23.43 MiB**（24,564,937 B） | 实测 |
| ├ `tokenizer.json` | 428.8 KiB | |
| ├ `vocab.txt` | 107.0 KiB | |
| └ config/特殊 token 等 | < 1 KiB | |
| 依赖 site-packages（本机 venv） | **174 MiB**（du 计 187 M） | onnxruntime 43M + transformers 57M + numpy 24M + tokenizers 7.9M + 其他 |
| bench venv 总计 | **188 MiB** | |
| 依赖 site-packages（容器） | 231 MB | 含容器内基础包 |
| Docker 镜像 `rag-onnx-bench:1.0` | 902 MB | 含 `python:3.12-slim` 基础 + apt 层，**部署需瘦身** |
| （对照）fp32 `model.onnx` | 90.45 MiB（94,851,877 B） | 来自 HF metadata，未下载 |
| （对照）仓库其他量化 | fp16 45.3 MiB / uint8 22.80 MiB / q4 50.0 MiB | HF metadata |

**int8 相比 fp32 权重缩小 3.94×**（22.90 vs 90.45 MiB）。

---

## 3. 进程 RSS（psutil 实测，峰值由后台高频采样线程捕获）

### 表 3-1 本机（`--threads 2`，贴近 2C）

| 阶段 | arena ON (MiB) | arena OFF (MiB) |
|---|---|---|
| 基线（python+numpy+psutil） | 79.2 | 79.7 |
| 导入 onnxruntime+transformers 后 | 79.2 | 79.7 |
| 加载 session 后 | 126.0 | 126.4 |
| **模型加载增量** | **+46.8** | **+46.7** |
| 单条推理采样峰值 | 127.5（增量 +1.4） | 126.4（增量 +0.0） |
| 批量32 推理采样峰值 | 217.7（增量 +91.6） | 191.8（增量 +65.4） |
| 稳态（200 次单条查询后） | 334.3 | **142.2** |
| 进程结束前 | 334.3 | 142.2 |

### 表 3-2 容器 `--cpus=2 --memory=4g`（`--memory-swap=4g` 硬限）

| 阶段 | arena ON (MiB) | arena OFF (MiB) |
|---|---|---|
| 基线 | 94.4 | 94.7 |
| 加载 session 后 | 142.1 | 140.9 |
| **模型加载增量** | **+47.8** | **+46.2** |
| 单条推理峰值 | 143.8（+1.7） | 142.6（+1.7） |
| 批量32 推理峰值 | 232.2 | 220.0 |
| 进程结束前 | **344.6** | **167.0 ~ 168.8** |

**关键洞察**：ORT CPU arena 在批量推理后不会归还内存（稳态 arena ON = 334~345 MiB，
arena OFF = 142~168 MiB）。**4G 场景建议 `enable_cpu_mem_arena=False`**，可省 ~190 MiB 常驻，
但需权衡延迟抖动（见 §4 与 §8）。

> 跨环境一致性极好：模型加载增量本机 46.8 / 容器 47.8 MiB；负载增量几乎相同，
> 说明该方案内存占用与平台无关，可控性强。

---

## 4. 检索延迟（tokenize + ONNX 前向 + 余弦排序，top_k=3）

### 表 4-1 本机（`--threads 2`，n=30，warmup=5）

| 语料规模 | 总 P50 (ms) | 总 P95 (ms) | tokenize P50 | forward P50 | rank P50 | 建索引(embed 全部) |
|---|---|---|---|---|---|---|
| 2 条 | 4.19 | 5.44 | 0.23 | 3.92 | 0.034 | 31.8 ms |
| 1000 条 | 5.15 | 6.55 | 0.27 | 4.35 | 0.423 | 21.1 s |

### 表 4-2 容器 2C4G（n=100，warmup=5，语料 1000 条）

| 配置 | 总 P50 (ms) | 总 P95 (ms) | mean (ms) | max (ms) | forward P95 | 建索引 |
|---|---|---|---|---|---|---|
| arena ON | **1.60** | **5.78** | 5.05 | 72.7 | 5.40 | 17.4 s |
| arena OFF（第 1 次） | 3.66 | **73.83** | 12.17 | 79.1 | 73.16 | 15.8 s |
| arena OFF（第 2 次复现） | 3.78 | **73.53** | — | 77.4 | 72.83 | — |

**要点**：
- 检索延迟（含查询编码 + 余弦排序）在 2 条与 1000 条语料间几乎不增长（排序 0.02→0.42 ms），
  瓶颈完全在**查询编码的 ONNX 前向**，与语料规模无关。
- 容器 2C 下 **arena ON 的 P95 = 5.78 ms**，远低于 T4 目标 80 ms（余量 ~14×）。
- **arena OFF 出现 P95 ≈ 73.8 ms 的抖动（两次复现）**，mean 抬到 12 ms、max ~79 ms，
  接近 80 ms 阈值；而 arena ON 尾部仅 5.8 ms。→ **2C 场景优先保留默认 arena**。
- 注意：本机为 20 核共享开发机（测量期间有约 28% 后台负载），故 m
  本机绝对延迟会波动（首跑空闲时 P50 曾达 1.31 ms）；**容器 2C 数据为准**。
- 建索引（一次性 embed 1000 条，batch=32）在真实 2C 下约 **16~17 s**，属启动期一次性开销，
  可离线预计算落盘规避。

---

## 5. Docker 4G 准约束验证

```text
docker run --rm --cpus=2 --memory=4g --memory-swap=4g rag-onnx-bench:1.0 \
  python /app/scripts/bench_rag_onnx.py --docker --model-dir /app/.bench_rag_onnx/model --iters 100
```

| 项 | arena ON | arena OFF |
|---|---|---|
| 模型加载 | ✅ 成功 | ✅ 成功 |
| 进程峰值 RSS | 344.6 MiB | 220.0 MiB（结束 167 MiB） |
| OOM | **否** | **否** |
| 退出码 | 0 | 0 |

**4G 限制下内存使用 < 9%**，加载与批量推理均通过。镜像构建成功（`.bench_rag_onnx/ctx/Dockerfile`）。

---

## 6. 共存预算（叠加到现有 edge 后）

### 6.1 edge 基线实测（项目 `.venv`，只读加载）

| 阶段 | RSS (MiB) |
|---|---|
| 裸进程（python+psutil） | 18.2 |
| + numpy + onnxruntime | 50.5 |
| + T2 ONNX session | 76.7 |
| + fastapi / uvicorn / prometheus_client / sqlite3 | 100.4 |
| + **Qwen2.5-0.5B Q4 LLM 加载** | 252.7 |
| + **LLM 推理一次后** | **623.5**（KV cache / compute buffer 分配） |

> edge 常驻按**较高实测路径**取 **623.5 MiB**（含一次 LLM 推理的峰值，保守）。
> 注：`llm.infer` 本次返回 `None`（该 prompt 未产出可解析 JSON），但 llama.cpp 的内存分配真实发生；
> LLM 权重按需 mmap，加载后 RSS 增量仅 ~152 MiB，推理时方升至峰值。

### 6.2 叠加计算

**方案 1：同进程集成（与现有 `edge/knowledge` 结构一致，推荐）**

| 项 | MiB |
|---|---|
| edge 基线常驻（含 LLM 推理） | 623.5 |
| RAG ONNX 模块加载增量 | +47.8 |
| RAG 单条查询峰值增量 | +1.7 |
| RAG 批量索引峰值增量（arena ON，一次性） | +91.6 |
| **合计峰值** | **≈ 764.6 MiB ≈ 0.747 GiB** |
| **4G 余量** | **≈ 3331 MiB ≈ 3.25 GiB（占用 18.7%）** |

**方案 2：RAG 独立 sidecar 进程**

| 项 | MiB |
|---|---|
| edge 基线 | 623.5 |
| RAG 独立进程峰值（容器 arena ON） | 344.6 |
| **合计** | **≈ 968.1 MiB** |
| **4G 余量** | **≈ 3128 MiB ≈ 3.05 GiB** |

即便叠加 30% 安全系数（同进程 ≈ 994 MiB），**2C4G 下合计仍远低于 4096 MiB**。

---

## 7. 与路线 A（sentence-transformers + PyTorch CPU）量化对比

> 路线 A 数据来源：`evidence/raw/rag_torch_bench.{txt,json}`（队友实测，同机 Windows，BLAS 模拟 2 核）。
> 路线 B 取本机 `--threads 2` 与容器 2C 实测。两侧均未在同一配置下互测，对比为**同机不同约束**的量化近似。

| 维度 | 路线 A (PyTorch) | 路线 B (ONNX int8) | 优势方 |
|---|---|---|---|
| 模型权重磁盘 | 182.79 MB（safetensors 91.4 + pytorch_model.bin 91.4） | **22.90 MiB** | **B 省 87%** |
| 模型目录磁盘 | 183.35 MB | **23.43 MiB** | **B 省 87%** |
| 依赖 site-packages | 768.27 MB（含 torch） | **174 MiB**（本机）/231 MB（容器） | **B 省 ~77%** |
| 导入后基线 RSS | 401.91 MB | **79.2 ~ 94.4 MiB** | **B 少 ~310 MiB** |
| 模型加载增量 | +107.26 MB | **+46.8 ~ +47.8 MiB** | **B 少 ~60 MiB** |
| 单条峰值增量 | +0.70 MB | +0.0 ~ +2.0 MiB | 相当 |
| 批量32 峰值增量 | +33.05 MB | +65.4(off) / +91.6(on) MiB | A 略优（注） |
| 进程整体峰值 | 543.47 MB | **217.7 MiB(本机) / 344.6 MiB(容器)** | **B 绝对更低** |
| 检索 P95 · 1000 条 | 21.28 ms（本机 BLAS2） | **5.78 ms**（容器 2C，n=100） | **B 快 ~3.7×** |
| Docker 4G 验证 | 进行中（`rag_torch_docker.txt` 截至本报告时仅含起始信息） | ✅ 通过，峰值 344.6 MiB | B |
| torch 依赖 | 必需（~200+ MB） | **无** | B |

> 注：批量峰值增量一项 A 更小，因 PyTorch 基线本就预留了 400+ MB 大内存池；从**绝对峰值**看
> 仍是 B 更低（217.7 vs 543.47 MiB）。检索质量（召回/排序）两侧均未评测，仅比资源。

**对比结论**：路线 B 在**磁盘、内存、延迟、依赖复杂度**四个维度全面优于路线 A；
唯一需关注的尾部抖动（arena OFF，容器 2C 下 P95≈73.8 ms）通过保留默认 arena 即可消除。

---

## 8. 障碍与局限

1. **arena 尾部抖动**：容器 2C + `enable_cpu_mem_arena=False` 时 100 次中 P95 稳定在 ~73.8 ms（两次复现），
   接近 T4 目标 80 ms。推测为 WSL2 容器内频繁分配/释放导致，**未进一步定位**。
   → 建议保留默认 arena（内存代价 ~190 MiB 常驻，延迟 P95 5.8 ms）。
2. **本机延迟受并发负载干扰**：测量期间同机存在其他 agent 任务（CPU ~28%），本机绝对延迟波动较大
   （首跑空闲 P50 1.31 ms，穿插负载时可达 13 ms）。**以容器 2C 数据为权威**。
3. **模型来源非官方**：int8 权重取自社区 `Xenova/bge-small-zh-v1.5`，非 BAAI 官方 ONNX；
   本报告**仅评资源，未评检索质量**（需另做召回/排序评测）。
4. **transformers 5.20.2 为较新版本**：无 torch 时 `AutoTokenizer` 正常（解析为 `BertTokenizer` 快 tokenizer），
   但版本较新，落地时建议锁定版本并做兼容性回归。
5. **共存预算为分进程相加近似**：未在同一容器内同时加载 edge LLM + RAG；同进程实际会共享部分库与页缓存，
   真实占用应**低于**本估算（估算偏保守）。
6. **未测并发**：全部为串行查询；多查询并发下的内存/延迟未覆盖。
7. **向量维度变更影响**：bge-small 输出 **512 维**，与现有 hash-embed(32 维) 不兼容，
   落地需同步更新 `snippets` 表 schema 与存量数据（当前 `edge/knowledge/sqlite_vec.py`）。
8. **Docker 镜像 902 MB 偏大**：含 apt/uv 引导层，部署前需多阶段精简（生产可仅装 onnxruntime+tokenizers，
   免去 transformers 57 MB、用 tokenizers 直接加载）。

---

## 9. 待补事项

- 路线 A 的 Docker 4G 约束数据：截至本报告生成时 `evidence/raw/rag_torch_docker.txt` 仅含起始信息（未完成），`evidence/rag_torch_bench.md` 亦未产出，队列任务 20 进行中。
- 两路线**检索质量**对比（同一批 query 的 top-k 命中/排序一致性），建议后续单独评测。
- 并发与长时间稳定性（内存是否随请求增长）——本报告已用 200 次串行查询确认无增长。

---

## 附：原始输出文件清单（均在 `evidence/raw/`）

| 文件 | 内容 |
|---|---|
| `bench_rag_onnx_t2_arena.json/.txt` | 本机 2 线程 arena ON 完整结果 |
| `bench_rag_onnx_t2_noarena.json/.txt` | 本机 2 线程 arena OFF 完整结果 |
| `bench_rag_onnx_arena.json/.txt` | 本机默认线程 arena ON |
| `bench_rag_onnx_noarena.json/.txt` | 本机默认线程 arena OFF |
| `docker_build.txt` | 镜像构建日志 |
| `docker_arena.json/.txt` | 容器 2C4G arena ON（n=100） |
| `docker_noarena.json/.txt` | 容器 2C4G arena OFF（n=100） |
| `docker_noarena2.json/.txt` | 容器 arena OFF 复现验证 |
| `docker_run.txt` | 容器首跑（n=30）原始输出 |
| `edge_baseline.txt` | edge 基线 RSS 分阶段实测 |
| `rag_torch_bench.json/.txt` | 路线 A 原始数据（引用） |

脚本与镜像定义：

| 路径 | 说明 |
|---|---|
| `scripts/bench_rag_onnx.py` | 测量脚本（`--docker` / `--no-arena` / `--threads` 等） |
| `.bench_rag_onnx/venv/` | 独立 venv（保留供复核） |
| `.bench_rag_onnx/model/` | int8 ONNX 模型 + tokenizer |
| `.bench_rag_onnx/ctx/Dockerfile` | 容器构建定义（最小上下文） |
| `.bench_rag_onnx/edge_baseline.py` | edge 基线测量脚本 || `.bench_rag_onnx/_download.py` | 模型下载脚本（记录来源与文件清单） |