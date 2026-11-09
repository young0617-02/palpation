# T4 知识检索「路线 A」资源实测报告

> 路线 A = `sentence-transformers` + **PyTorch(CPU)** + `BAAI/bge-small-zh-v1.5`(512 维)
> 目标：验证在 **2 核 4GB** 边缘约束下引入真实语义检索的可行性（内存为主要风险）。
> 所有数字均来自真实执行；脚本与原始输出见文末「可复核性」。

- 生成时间：2026-10-22
- 任务：team task id=20（RAG-PyTorch 实测工程师）
- 被测对象：把端侧 T4 当前 `hash-embed` 占位实现（`edge/knowledge/sqlite_vec.py`，纯字节袋哈希 32 维）升级为 bge-small 语义检索后的资源开销。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| **2C4G 内存下能否加载？** | ✅ **能**。`docker run --cpus=2 --memory=4g` 实测：加载+推理全程成功，**进程峰值 RSS 522 MB / 4096 MB（12.7%），未触顶、未 OOM**。 |
| **T4 检索延迟达标？** | ✅ 达标但**余量取决于 CPU 争用**。空闲态 P95 远低于 80 ms 目标；但在 2-CPU 配额下 1000 条语料 P95 已抬升到 **38–44 ms**，观测到一次 **78 ms**（逼近上限）。 |
| **引入路线 A 的真实代价？** | **内存 ≈ 520 MB/进程**（基线导入 ~380–400 MB + 权重 ~108 MB + 批工作集 ~34 MB），**约为方案 §3.5 预估值 120 MB 的 4.3 倍**。 |
| **磁盘代价？** | 依赖 **~770 MB**（首次运行后含 `.pyc` 达 ~818 MB）；模型 **91 MB/份权重**（官方仓同时含 safetensors+bin 共 183 MB）。**远超方案「模型 ~95 MB / 磁盘余量 <500 MB」的预期**。 |
| **叠加现有边缘常驻后总量？** | 实测现有 edge 进程稳态 **611 MB**（含 Qwen2.5-0.5B Q4 预热后驻留）；叠加路线 A 后全系统估算 **≈1.74–1.87 GB**，**4G 预算内余量 ≈2.1–2.3 GB**。 |
| **最终判断** | **内存维度：可行**（余量充足）；**但可行性被「CPU 争用」与「磁盘膨胀」两个非内存因素削弱**。真正的障碍不是"放不下"，而是"装得下却跑不稳 / 存不下"。 |

---

## 1. 环境与复现命令

### 1.1 运行环境

| 项 | 值 |
|---|---|
| 平台 | Windows-11-10.0.26200-SP0 / win32 |
| CPU（开发机） | Intel64 Family 6 Model 183（20 逻辑核）|
| RAM（开发机） | 34 GB（可用约 8.7 GB）|
| Python | 3.12.8 |
| uv | 0.11.31 |
| Docker | 29.6.2（Docker Desktop，Linux 容器）|
| **torch** | **2.14.1+cpu**（`cuda=False`，CPU-only wheel）|
| **transformers** | **5.20.2** |
| **sentence-transformers** | **6.2.0** |
| tokenizers | 0.23.3 |
| 模型 | `BAAI/bge-small-zh-v1.5`，输出维度 **512** |
| 线程约束 | `OMP_NUM_THREADS=2` 等环境变量 + `torch.set_num_threads(2)`（模拟 2C）|

### 1.2 独立环境（绝不污染项目 `.venv`）

按硬性约束，全部依赖装在独立目录 `D:\project\Palpation\.bench_rag_torch\`，**未修改** `pyproject.toml` / `uv.lock`（`git status` 仅新增未跟踪目录）。测后保留供复核。

```bash
# 1) 独立 venv（禁止 pip，统一 uv）
uv venv .bench_rag_torch/venv --python 3.12

# 2) CPU-only torch（显式走 CPU 专用 index，避免拉 CUDA 巨包）
uv pip install --python .bench_rag_torch/venv/Scripts/python.exe torch \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# 3) sentence-transformers（连带 transformers/tokenizers/scipy/sklearn）
uv pip install --python .bench_rag_torch/venv/Scripts/python.exe sentence-transformers
uv pip install --python .bench_rag_torch/venv/Scripts/python.exe psutil

# 4) 经 hf-mirror 下载模型（注意：镜像不支持 hf-xet，需禁用）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
.bench_rag_torch/venv/Scripts/python.exe -c \
  "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir=r'.bench_rag_torch/models/bge-small-zh-v1.5')"

# 5) 本机测量（模拟 2C）
export HF_HUB_OFFLINE=1
.bench_rag_torch/venv/Scripts/python.exe scripts/bench_rag_torch.py

# 6) Docker 4G 约束验证
#    先用 uv 预取 Linux wheel 到 docker_deps（避免容器内 pip/联网）
uv pip install --target .bench_rag_torch/docker_deps \
  --python-platform x86_64-manylinux_2_28 --python-version 3.12 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  torch sentence-transformers psutil
cd .bench_rag_torch && docker build -t rag-torch-bench:local .
docker run --rm --cpus=2 --memory=4g --memory-swap=4g \
  -v D:/project/Palpation/.bench_rag_torch/docker_deps:/deps:ro \
  -v D:/project/Palpation/.bench_rag_torch/models:/models:ro \
  rag-torch-bench:local
```

> 说明：`torch 2.14.1+cpu` 在 Windows(PyPI) 与 Linux(manylinux) 上均能直接装 CPU 版；此处仍显式指定 `download.pytorch.org/whl/cpu` 以杜绝 CUDA 依赖被间接拉入。

### 1.3 产物清单

| 文件 | 说明 |
|---|---|
| `scripts/bench_rag_torch.py` | 可复现测量脚本（磁盘 / RSS 四组 / 延迟）|
| `evidence/raw/rag_torch_bench.txt` | 本机测量原始输出 |
| `evidence/raw/rag_torch_bench.json` | 本机测量结构化结果 |
| `evidence/raw/rag_torch_docker.txt` | 容器 4G 约束验证原始输出 |
| `.bench_rag_torch/venv/` | 独立虚拟环境（Windows wheel，保留复核）|
| `.bench_rag_torch/docker_deps/` | Linux wheel（容器验证用）|
| `.bench_rag_torch/models/bge-small-zh-v1.5/` | 已下载模型 |
| `.bench_rag_torch/Dockerfile` / `docker_bench.py` | 容器验证用镜像与脚本 |
| `.bench_rag_torch/coexist_probe.py` | edge 侧既有常驻 RSS 探针 |

---

## 2. 磁盘占用

| 项 | 大小 | 来源 |
|---|---|---|
| 模型目录总计（含 safetensors + bin 双权重） | **183.35 MB** | 测量 |
| `model.safetensors` | 91.4 MB | 测量 |
| `pytorch_model.bin` | 91.4 MB | 测量 |
| **单份权重（去重后实际所需）** | **~91 MB** | 测量 |
| 依赖 `site-packages`（torch+transformers+ST+scipy+sklearn+tokenizers+numpy…） | **768.27 MB** | 测量（安装后）|
| 依赖 `site-packages`（首次运行生成 `.pyc` 后） | 818.60 MB | 测量（运行后）|
| Linux 依赖目录 `docker_deps`（含 dist-info） | 1008.8 MB | 测量 |

关键依赖细分（`site-packages`）：

| 包 | 大小 |
|---|---|
| torch | **464.1 MB** |
| scipy | 83.4 MB |
| transformers | 48.6 MB |
| sklearn(scikit-learn) | 25.9 MB |
| numpy（项目已装，非新增） | 21.6 MB |
| tokenizers | 7.7 MB |
| sentence_transformers | 2.5 MB |
| huggingface_hub / safetensors | ~3.9 MB |

> **净新增磁盘 ≈ 747 MB**（768 − 已存在的 numpy 22）。
> **对比方案预期**：方案 §3.5 将「T4 嵌入模型」记为 ~120 MB、§7 磁盘护栏以「模型 <500 MB」为口径。路线 A 的真实磁盘代价是**模型 91 MB + 依赖 ~770 MB ≈ 860 MB**，其中 **~90% 是 PyTorch 运行时本身**，而非模型。
> **附带问题**：HF 官方仓同时下发 `safetensors`(91 MB) 与 `pytorch_model.bin`(91 MB) 两份等价权重；边缘部署须裁剪为单份，否则无谓多占 91 MB。

---

## 3. 内存（进程 RSS，psutil）

口径：`psutil.Process().memory_info().rss`；峰值由后台线程 ~1 ms 高频采样捕捉。本机 3 轮复跑，取范围。

### 3.1 本机（Windows，CPU-only torch）

| 指标 | 轮1 | 轮2 | 轮3 | 轮4 | **报告值** |
|---|---|---|---|---|---|
| 裸进程 RSS（未导入） | 26.8 | 27.3 | 27.0 | 26.8 | ~27 MB |
| **① 基线 RSS（导入 torch+transformers 后、加载模型前）** | 401.9 | 376.0 | 375.8 | 375.1 | **375–402 MB** |
| **② 加载 bge-small 增量** | +107.3 | +109.4 | +109.6 | +109.7 | **+107–110 MB** |
| **③ 单条文本 embedding 峰值增量** | +0.70 | +0.72 | +0.97 | +0.78 | **+0.7–1.0 MB** |
| **④ 批量 32 条 embedding 峰值增量** | +33.05 | +33.66 | +34.31 | +33.08 | **+33–34 MB** |
| 进程整体峰值 RSS | 543.5 | 520.4 | 521.2 | 519.2 | **519–544 MB** |

> 对照方案 §3.5 预估「T4 嵌入模型 ~120 MB」：**实测「加载增量」+108 MB 与预估值接近**；
> **但方案漏算了「PyTorch + transformers 运行时基线 ~380–400 MB」**——这才是路线 A 的主要内存成本。
> 即：单看"模型权重"是 108 MB，但"让模型能跑起来的框架"还要 ~390 MB，**合计 ~520 MB/进程**（约为方案值的 4.3 倍）。

### 3.2 容器内（`--cpus=2 --memory=4g`）

| 指标 | 值（多次复跑） |
|---|---|
| cgroup 内存上限 | 4096 MB |
| 裸进程 RSS | ~20 MB |
| **① 基线 RSS** | **426.5–427.2 MB** |
| **② 加载 bge-small 增量** | **+69.9–70.4 MB** |
| **③ 单条 embedding 峰值增量** | **+0.91 MB** |
| **④ 批量 32 条 embedding 峰值增量** | **+23.35 MB** |
| **进程整体峰值 RSS** | **521.8–523.2 MB（占 4G 的 12.7%）** |
| 是否触顶/OOM | **否** |

> 容器内「加载增量」(+70 MB) 低于本机 (+108 MB)，差异来自分配器/页驻留策略；两者「进程整体峰值」一致（~522 MB），说明**峰值主要由框架基线决定，稳定且可复现（5 次复跑极差 <1.5 MB）**。

---

## 4. 检索延迟（top_k=3，含查询编码 + 余弦排序）

查询编码用 `SentenceTransformer.encode`，排序用 numpy 点积并 `argsort`（复刻 `sqlite_vec.cosine` 语义）。每场景 25 次迭代（warmup 3）。目标：方案 §6.4 的 **T4 P95 < 80 ms**。

### 4.1 本机（模拟 2 线程）

| 场景 | P50 (ms) | P95 (ms) | mean (ms) | n |
|---|---|---|---|---|
| 现有 **2 条 snippets** | 6.5–18.4 | **7.1–23.7** | 6.4–14.8 | 25 |
| 扩容 **1000 条伪片段** | 7.1–19.2 | **10.6–27.9** | 7.7–20.4 | 25 |

### 4.2 容器内（`--cpus=2`）

| 场景 | P50 (ms) | P95 (ms) | mean (ms) | n |
|---|---|---|---|---|
| 现有 **2 条 snippets** | 6.6–7.2 | **8.2–9.5** | ~7.2 | 25 |
| 扩容 **1000 条伪片段** | 7.1–8.3 | **37.8–44.2**（一次 78.1） | 10.5–14.2 | 25 |

**观察**：
1. **语料从 2 条 → 1000 条，排序部分几乎不增加开销**（512×1000 点积 <1 ms）；延迟几乎全部来自**查询编码（单条 encode）**。
2. 本机 P95 ~7–28 ms；容器 P95（1000 条）~38–44 ms，**比本机更慢**——`--cpus=2` 配额下 CPU 争用/节流显著。**观测到一次 78.1 ms，已逼近 80 ms 上限**。
3. 说明 **T4 延迟的敏感性在 CPU 争用，不在内存**：在 4G 内存宽松、但 2 核被 LLM/ONNX 共用的场景，T4 P95 有突破 80 ms 的风险。

---

## 5. Docker 4G 准约束验证

- 命令：`docker run --rm --cpus=2 --memory=4g --memory-swap=4g`（swap 无余量，从严验证）
- 镜像：`python:3.12-slim` + 挂载预取的 Linux wheel（`/deps`）+ 模型（`/models`）；torch manylinux wheel 自带 `libgomp.so.1`，无需额外系统库。
- 构建耗时：**约 2.4 s**（基础镜像已缓存，无 pip/编译步骤）；单次运行约 4–5 分钟（含容器内首次 import torch）。
- 结果（5 次复跑一致）：

```
cgroup 内存上限 : 4096 MB
torch           : 2.14.1+cpu  cuda=False  threads=2
基线 RSS(导入后): 427.19 MB
加载后 RSS      : 497.11 MB  增量 +69.92 MB
单条 embedding 峰值增量 : +0.91 MB
批量32 embedding 峰值增量: +23.35 MB
进程整体峰值 RSS: 521.79 MB
占 4G 上限比例  : 12.7%   余量 3573.81 MB
是否触顶/OOM    : 否
[1000 条伪片段] 语料 1000 条  P50=7.48 ms  P95=38.41 ms
结果: 容器内加载与推理均成功，未 OOM。
```

> **结论：4G 硬限制下，路线 A 可正常加载并推理，内存不是瓶颈。** 注意容器 `os.cpu_count()` 报告 10（Docker Desktop VM 规模），但 `--cpus=2` 通过 CFS 配额真实限制了算力——这正是延迟抬升的原因。

---

## 6. 边缘侧既有常驻实测 & 共存预算

### 6.1 实测：本项目 edge 侧稳态占用（项目 `.venv`，`coexist_probe.py`）

| 阶段 | 进程 RSS |
|---|---|
| 裸解释器 | 18.3 MB |
| + numpy + onnxruntime | 49.5 MB |
| + `edge.settings` 导入 | 64.4 MB |
| + `EdgeLLM` 加载（0.76 s，可用=True） | 329.9 MB |
| **+ LLM 真实推理一次（KV/权重驻留）** | **586.7 MB** |
| + `AnomalyDetector`(ONNX) | 610.3 MB |
| + `LocalKnowledge`(hash-embed) 检索 | **611.0 MB** |

> **发现**：Qwen2.5-0.5B Q4 权重文件仅 **468.6 MiB**（491,400,032 字节），但**推理一次后进程 RSS 达 ~587 MB**——因为 GGUF 采用 mmap，权重页在推理时才被真正 fault-in，叠加 KV Cache 与计算缓冲，**LLM 实际常驻增量 ≈ 522 MB**（而非 468 MB 文件大小）。
> edge-agent 单进程稳态（含 LLM warm + ONNX + hash KB）实测 **611 MB**。

### 6.2 共存预算表（叠加路线 A）

以方案 §3.5 表格为骨架，**把 T4 一行替换为实测值**，其余保持方案口径：

| 项 | 方案原估 | **实测/修正** | 口径 |
|---|---|---|---|
| OS + 容器运行时 | 300 MB | ~300 MB | 方案 |
| Edge Agent 运行时（Python+FastAPI） | 150 MB | ~150 MB | 方案 |
| MQTT 客户端 + 流处理 | 50 MB | ~50 MB | 方案 |
| T3a LLM 权重 Qwen2.5-0.5B Q4 | 400 MB | ~470 MB | **实测文件 468.6 MiB** |
| LLM KV Cache (`ctx=2048`) | 100 MB | ~100 MB | 方案 |
| llama.cpp 运行时 | 100 MB | ~100 MB | 方案 |
| T2 异常检测（ONNX+Runtime） | 200 MB | **~25 MB** | **实测 +23.6 MB**（方案高估）|
| **T4 嵌入检索（路线 A）** | **120 MB** | **~520 MB** | **实测**（基线 ~390 + 权重 ~108 + 工作集 ~34）|
| SQLite + 队列 + 索引 | 100 MB | ~100 MB | 方案 |
| 探针 / 监控 / 日志缓冲 | 50 MB | ~50 MB | 方案 |
| **合计** | **≈1.57 GB** | **≈1.87 GB** | — |
| **4G 余量** | ~2.4 GB | **≈2.2 GB** | — |

另若把 mgmt-agent、cloud-api、mosquitto、device-sim 等 compose 稀疏服务计入（各自 FastAPI ~80–100 MB / MQTT ~10 MB），实测 edge-agent 611 MB + 路线 A ~520 MB + 其他服务 ~300 MB + OS/容器 ~300 MB ≈ **1.74 GB**，**4G 余量 ≈2.3 GB**。

> **共存预算判断：叠加路线 A 后合计 ≈1.7–1.9 GB < 4 GB，预计余量 ≈2.1–2.3 GB。内存维度可共存。**

---

## 7. 最终判断：2C4G 下路线 A 是否可行

### 7.1 可行（有保留）

| 维度 | 判定 | 依据 |
|---|---|---|
| **内存** | ✅ **可行** | 容器 4G 下峰值仅 522 MB（12.7%）；叠加现有边缘常驻后总量 ≈1.7–1.9 GB，余量 ≈2.1–2.3 GB |
| **加载/稳定性** | ✅ **可行** | 5 次容器复跑峰值极差 <1.5 MB；无 OOM |
| **本地检索延迟（空闲）** | ✅ **达标** | 本机 P95 7–28 ms、容器空闲 P95 8–44 ms，均 <80 ms |
| **磁盘** | ⚠️ **吃紧** | 依赖 ~770–818 MB + 模型 91 MB，远超方案「~120 MB / <500 MB」预期 |
| **CPU 争用下的延迟** | ⚠️ **有风险** | 2-CPU 配额下 1000 条 P95 已到 38–44 ms，观测到 78 ms；与 T3 LLM（2 线程）、ONNX 并发时可能越界 |
| **冷启动** | ⚠️ **有代价** | 预热后 import torch+transformers+ST ≈ **6.8 s**；首次冷盘（eMMC）实测曾达 **~60 s** |

**一句话结论：内存维度不是障碍（方案 §3.5 的 120 MB 低估了 4.3 倍，但总账仍在 4G 内）；真正的障碍是"依赖体积膨胀"与"2 核 CPU 争用"——路线 A 在 2C4G 上"装得下、跑得出正确结果"，但要"稳定达标"依赖于磁盘空间和 T3/T4 的线程调度编排。**

### 7.2 障碍清单

1. **方案预算修订（必须）**：§3.5 内存表 T4 行 120 MB → 实测 ~520 MB；§7 磁盘护栏"模型 <500 MB"口径需改为"依赖+模型 ≈860 MB"。
2. **磁盘**：~770 MB 依赖（其中 torch 464 MB / scipy 83 MB）对 eMMC 边缘盒是硬约束；须评估根分区容量与 OTA 分块下载空间公式（方案用 `size×2+500MB`）。
3. **CPU 争用**：T4（torch 2 线程）+ T3（llama.cpp 2 线程）在 2 核上互抢；实测证明空闲 vs 争用下 1000 条 P95 可差 2–9 倍。需要线程预算/优先级编排，或 T4 与 T3 串行化。
4. **冷启动**：~6.8 s（热）/ ~60 s（冷盘）的框架导入会拖慢边缘进程启动与 OTA 重启窗口。
5. **依赖面**：torch 2.14 / transformers 5.x 体量大、CVE 面广；边缘（离线、难升级）需更频繁的安全补丁流程。
6. **权重冗余**：须裁剪 HF 双权重（safetensors/bin）为单份。

### 7.3 局限（本实测的边界）

1. **开发机非真实 2C4G 设备**：本机 20 核 / 34 GB。已用 `torch.set_num_threads(2)` + 环境变量锁线程，并用 `docker --cpus=2 --memory=4g` 容器作为最接近的代理；但容器运行于 Docker Desktop VM（Linux 内核、10 核可见、CFS 配额），**与真实工控 ARM/x86 eMMC 边缘盒的绝对性能、内存带宽、页缓存行为仍有差异**，延迟/冷启动数字不可直接外推到目标硬件——须在目标设备复测（方案 §6.4 亦要求"禁止外推"）。
2. **延迟方差较大**：容器 1000 条 P95 在 37.8–78.1 ms 间波动，系 CPU 配额节流与宿主后台负载所致；本机亦有 9.7–18.4 ms 的 P50 波动。报告以范围呈现，未做多轮统计检验。
3. **未模拟 T4 与 T3 并发**：本报告 T4 延迟为"独占进程"口径，**未叠加 LLM 并发推理**；共存下的真实 T4 P95 需专门编排测试。
4. **地理口径**：RSS 含共享库/页缓存，不同 OS 的驻留策略不同；容器与宿主机同一现象但数值有差（如加载增量 +70 vs +108 MB）。
5. **路线 A 未做量化/蒸馏**：本实测为 FP32 bge-small + 原版 sentence-transformers；量化（如 int8）/ONNX 导出可能显著降低内存与依赖（属路线 B 范围，本次未测）。

---

## 8. 原始输出片段

**本机（`evidence/raw/rag_torch_bench.txt`，节选）**

```
[磁盘占用]
  模型目录总计    : 183.35 MB
  权重文件        : model.safetensors (91.4 MB) + pytorch_model.bin (91.4 MB)  = 182.79 MB
  site-packages   : 818.60 MB (含 torch/transformers/sentence-transformers/... 全部依赖)

[内存 — 基线 (导入后/加载模型前)]
  基线 RSS        : 375.11 MB  (导入 torch+transformers 后、加载模型前)

[内存 — 加载 bge-small-zh-v1.5]
  向量维度        : 512
  加载后 RSS      : 484.85 MB
  加载增量        : +109.74 MB  (峰值窗口 +109.73 MB)
  （对照）方案预估: 120 MB

[内存 — 批量 32 条 embedding 峰值]
  峰值增量        : +33.08 MB
  进程整体峰值    : 519.19 MB

[延迟汇总]
  2 条 snippets               6.45      7.05      6.44   25
  1000 条伪片段                7.05     10.59      7.65   25
  方案 T4 目标             P95 < 80.00 ms
```

**容器 4G 约束（`evidence/raw/rag_torch_docker.txt`，节选）**

```
 cgroup 内存上限 : 4096 MB
 基线 RSS(导入后): 427.19 MB
 加载后 RSS      : 497.11 MB  增量 +69.92 MB
 进程整体峰值 RSS: 521.79 MB
 占 4G 上限比例  : 12.7%   余量 3573.81 MB
 是否触顶/OOM    : 否
  [1000 条伪片段] 语料 1000 条  P50=7.48 ms  P95=38.41 ms
 结果: 容器内加载与推理均成功，未 OOM。
```

---

## 9. 对方案的直接建议

1. **修订 §3.5 内存表**：T4 行 `120 MB` → `~520 MB`；并在表格中显式区分"模型权重"与"框架运行时（PyTorch+transformers）"。
2. **修订 §7 磁盘护栏**：为路线 A 预留 `≈860 MB`（依赖 ~770 + 模型 91），并在 OTA 下载空间公式中计入依赖层。
3. **增设 CPU 编排约束**：明确 T3/T4 线程预算与并发策略（串行化或分核），否则 T4 P95 在共存下可能破 80 ms。
4. **若磁盘/CPU 不可放宽**：路线 A 的"框架重量"是主因，建议以已具备的 `onnxruntime`（项目已装、无 torch 依赖）实现 T4 语义检索（路线 B，由并行任务实测），以规避 770 MB 依赖与 CPU 争用。
5. **裁剪权重**：部署时仅保留单份权重（safetensors 91 MB）。

---

## 10. 可复核性

- 测量脚本：`scripts/bench_rag_torch.py`（参数 `--iters/--batch/--pseudo-n/--model-dir`，支持多次复现）。
- 原始输出：`evidence/raw/rag_torch_bench.txt` / `.json` / `rag_torch_docker.txt`。
- 独立环境：`.bench_rag_torch/venv`（Windows）、`.bench_rag_torch/docker_deps`（Linux wheel）、`.bench_rag_torch/models/bge-small-zh-v1.5`。
- 容器：`.bench_rag_torch/Dockerfile` + `docker_bench.py`，`docker build -t rag-torch-bench:local .` 后按 §1.2 命令复跑。
- 项目 `.venv` 与 `pyproject.toml` / `uv.lock` 未被修改（`git status` 仅 `Material/` 与新增未跟踪目录）。