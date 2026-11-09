# edge-ai-agent-gateway

> 边云协同 AI Agent 网关（方案 V1.0，Phase 0）
>
> 版本：`1.0.0`　｜　Python：`>= 3.12`　｜　依赖管理：`uv`

---

## 一、项目简介

**edge-ai-agent-gateway** 是一个面向工业现场的「边云协同 AI Agent 网关」，对应方案 V1.0 的 Phase 0 实现。它把「实时采集 — 规则与异常检测 — 本地知识诊断 — 离线自治 — 云端灰度 — OTA 升级回滚」串成一条完整的边云协同链路：

- **边缘侧（edge）**：部署在工业现场设备侧，负责多协议本地实时采集、滑动窗口统计、规则引擎判定（T1）、轻量 ONNX 异常检测（T2）、故障链与本地知识检索诊断（T4），并具备完全离线自治能力（断网时事件进入本地队列，恢复后自动补传）。
- **云端（cloud）**：承担增量服务职责，负责设备注册、事件查询、水位（watermark）管理、模型注册与环状灰度下发、工件（artifact）下载，并基于 SQLite + MQTT 对外提供服务。
- **管理代理（mgmt_agent）**：负责模型槽位管理、OTA 应用与回滚编排，以及对边缘网关健康状态的探测。

---

## 二、架构与模块

下表逐项说明每个模块「是什么」以及「做什么」。

### 1. `edge/` —— 边缘网关

- **是什么**：边缘侧主服务，基于 **FastAPI** 实现，**默认端口 `9000`**（由 `EDGE_API_PORT` 控制）。
- **做什么**：作为现场自治核心，向上通过 MQTT 与云端通信，向下通过多协议适配器采集设备数据，并在本地完成检测与诊断。
- `edge/main.py`：应用入口，创建 `EdgeRuntime` 并挂载 FastAPI 应用，使用 `uvicorn` 启动。
- `edge/runtime.py`：运行时编排中枢，负责采集、窗口、规则、异常、知识、Agent、队列、影子、资源守护等组件的装配与生命周期调度。
- `edge/adapters/`：**采集适配层**，包含 `modbus_tcp.py`（Modbus TCP）、`opcua_client.py`（OPC UA）、`mqtt_sparkplug.py`（MQTT Sparkplug B）、`coap_sensor.py`（CoAP 传感器）与 `base.py`（适配器基类）。
- `edge/stream/`：**流处理层**，含 `window.py`（滑动窗口统计，输出均值/最值/p95/标准差/斜率等）、`rules.py`（规则引擎，T1 阈值与越限判定）、`anomaly_onnx.py`（ONNX 轻量自编码器异常检测，T2）。
- `edge/knowledge/`：**本地知识诊断**，含 `fault_chain.py`（故障链推理）、`sqlite_vec.py`（**ONNX 语义检索**：`bge-small-zh-v1.5` int8 / 512 维 / `onnxruntime`，`numpy` 余弦，模型缺失时降级 hash-embed）、`fault_chains.json`（故障链定义）、`snippets.json`（知识片段，30 条）。
- `edge/agent/`：**端侧 Agent**，含 `state_machine.py`（状态机编排）、`llm_llamacpp.py`（端侧 LLM，可选启用）、`tools.py`（Agent 可调用工具集）。
- `edge/queue/`：**离线队列**，含 `sqlite_queue.py`（基于 SQLite 的事件本地持久化队列）与 `watermark.py`（水位推进、缺号检测与批量补传）。
- `edge/cloud/`：**边缘侧云协同组件**，含 `masking.py`（设备标识 HMAC 脱敏）、`mqtt_north.py`（北向 MQTT 通道）、`shadow.py`（设备影子 desired/reported 对齐）、`data_plane.py`（数据面）。
- `edge/guard.py`：**资源降级守护**，依据内存、磁盘、CPU 温度等指标在 `normal / llm_small / llm_off` 之间切换降级等级。
- `edge/config_loader.py`、`edge/settings.py`：YAML 配置加载与环境变量设置。

### 2. `cloud/` —— 云端增量服务

- **是什么**：云端服务，基于 **FastAPI** 实现，**默认端口 `8000`**（由 `CLOUD_API` / 内部 `cloud_http_port` 控制）。
- **做什么**：面向边缘网关提供注册、查询、模型分发等增量能力。
- `cloud/main.py`：应用入口与全部 HTTP 接口（设备注册 `POST /v1/devices/register`、事件查询 `GET /v1/events`、水位查询 `GET /v1/watermarks/{gw_id}`、模型注册 `POST /v1/models/register`、工件下载 `GET /v1/models/artifacts/{name}`、灰度下发 `POST /v1/rollout`）。
- `cloud/db.py`：数据模型与数据库初始化（设备、事件、水位、模型注册表、灰度计划与记录、影子），基于 SQLAlchemy。
- `cloud/mqtt_bridge.py`：云端 MQTT 桥接，订阅边缘事件、写入数据库并回送 ACK。
- `cloud/diagnosis.py`：云端诊断处理。
- `cloud/settings.py`：云端配置（数据库地址、MQTT 主机/端口、HTTP 端口、工件目录）。

### 3. `mgmt_agent/` —— 管理代理

- **是什么**：管理代理服务，基于 **FastAPI** 实现，**默认端口 `9001`**（由 `MGMT_API_PORT` 控制）。
- **做什么**：负责模型槽位管理与 OTA 应用/回滚，并探测边缘网关健康状态。
- `mgmt_agent/main.py`：应用入口，提供 `GET /health`（含边缘探测结果与当前槽位状态）、`POST /ota/apply`（应用新模型清单）、`POST /ota/rollback`（回滚指定层与版本），并通过 MQTT 订阅清单与命令。
- `mgmt_agent/ota.py`：OTA 引擎与状态机（下载、校验、暂存、激活、探测、观察、成功、回滚、冻结）。
- `mgmt_agent/probe.py`：边缘网关健康探测。
- `mgmt_agent/version.py`：模型版本槽位管理（当前版本、激活、回滚）。
- `mgmt_agent/__init__.py`：包初始化。

### 4. `shared/` —— 共享定义

- **是什么**：跨模块共享的数据与协议定义。
- **做什么**：
  - `shared/schemas.py`：核心数据模型（`InternalSample`、`EdgeTelemetry`、`EdgeEvent`、`EdgeHealth`、`AckPayload`、`ModelManifest` 等，基于 Pydantic）。
  - `shared/topics.py`：MQTT 主题族（telemetry、event、ack、diagnosis、health、status、shadow desired/reported、manifest、command、Sparkplug DDATA 等）。
  - `shared/errors.py`：统一错误码（`E_TIMEOUT`、`E_NOT_FOUND`、`E_MODEL_HASH_MISMATCH`、`E_REQUIRES_FIRMWARE`、`E_DISK_FULL` 等）与 `EdgeError` 异常类型。

### 5. `sim/` —— 设备模拟器

- **是什么**：用于 Phase 0 联调的现场设备模拟器集合。
- **做什么**：`modbus_device.py`（Modbus 设备）、`opcua_meter.py`（OPC UA 电表）、`mqtt_pump.py`（MQTT 泵机）、`coap_sensor.py`（CoAP 传感器）。

### 6. `config/` —— 配置目录

- **是什么**：YAML 配置存放目录。
- **做什么**：
  - `config/current.yaml`：当前生效配置（Windows 仿真下作为实时副本，POSIX 生产环境为指向版本文件的符号链接）。
  - `config/v0001.yaml`：`v0001` 版本化配置快照。

---

## 三、环境要求

- **Python**：`>= 3.12`
- **包管理器**：[uv](https://docs.astral.sh/uv/)（负责虚拟环境创建、依赖安装与锁定）

---

## 四、安装

在项目根目录执行：

```bash
uv sync
```

该命令会依据 `pyproject.toml` 自动创建 `.venv` 虚拟环境，并安装**全部依赖（含开发依赖）**。

---

## 五、配置

1. 复制环境变量模板：

```bash
copy .env.example .env      # Windows (cmd)
# 或
cp .env.example .env        # POSIX
```

2. `.env` 中的关键变量（与 `.env.example` 对应）：

| 变量 | 说明 |
| --- | --- |
| `GW_ID` | 网关标识，默认 `gw-sim-001` |
| `FIRMWARE_VERSION` | 固件版本，默认 `0.1.0` |
| `CONFIG_VERSION` | 配置版本，默认 `v0001` |
| `CLOUD_MQTT_HOST` / `CLOUD_MQTT_PORT` | 云端 MQTT Broker 地址与端口，默认 `127.0.0.1:1883` |
| `CLOUD_API` | 云端 HTTP 地址，默认 `http://127.0.0.1:8000` |
| `EDGE_API_PORT` | 边缘网关端口，默认 `9000` |
| `MGMT_API_PORT` | 管理代理端口，默认 `9001` |
| `HMAC_SALT` | 设备标识脱敏用盐值（仿真默认值须替换） |
| `EDGE_LLM_MODEL` / `EDGE_LLM_CTX` / `EDGE_LLM_MAXTOK` | 端侧 LLM 模型路径、上下文长度与最大生成 token |
| `DATABASE_URL` | 云端数据库地址，默认 `sqlite:///./data/cloud.db` |
| `MODBUS_HOST` / `MODBUS_PORT` | Modbus 设备地址与端口 |
| `OPCUA_ENDPOINT` | OPC UA 服务端点 |
| `LOG_LEVEL` | 日志级别，默认 `INFO` |

3. `config/current.yaml` 的关键配置项：

| 配置项 | 说明 |
| --- | --- |
| `gw_id` / `firmware_version` / `config_version` | 网关标识、固件版本与配置版本 |
| `site` / `group_id` | 现场站点与分组标识 |
| `sampling.vibration_window` | 振动滑动窗口长度（采样点数） |
| `sampling.telemetry_period_s` | 遥测上报周期（秒） |
| `sampling.health_period_s` | 健康上报周期（秒） |
| `sampling.raw_rate_hz` | 原始采样频率（Hz） |
| `upload.*_qos` | 遥测/事件/健康上报的 MQTT QoS 等级 |
| `rules[]` | 阈值规则列表（`id`、`metric`、`op`、`threshold`、`severity`、`description`） |
| `anomaly.*` | 异常检测开关、窗口、分数阈值、预处理版本 |
| `llm.*` | 端侧 LLM 开关、最大 token、上下文、缓存 TTL |
| `degrade.*` | 资源降级阈值（内存告警/关闭 LLM、磁盘停止加载模型、CPU 温度） |
| `privacy.*` | 脱敏盐值环境变量名与温度分箱 |
| `devices.*` | 设备接入参数（设备 ID、Modbus 地址与端口等） |

---

## 六、运行

以下命令均在项目根目录执行。

### 1. 边缘网关（默认端口 9000）

```bash
uv run python -m edge.main
```

或直接使用 uvicorn：

```bash
uv run uvicorn edge.main:app --port 9000
```

### 2. 云端增量服务（默认端口 8000）

```bash
uv run python -m cloud.main
```

### 3. 管理代理（默认端口 9001）

```bash
uv run python -m mgmt_agent.main
```

---

## 七、测试

```bash
uv run pytest
```

测试代码位于 `tests/` 目录，覆盖知识诊断、脱敏与异常检测、离线自治、OTA 回滚、水位管理等核心链路。

---

## 八、目录结构

```text
Palpation/
├── cloud/          # 云端增量服务（FastAPI，端口 8000：注册/事件/水位/模型注册与灰度/工件下载）
├── config/         # YAML 配置（current.yaml、v0001.yaml）
├── edge/           # 边缘网关（FastAPI，端口 9000）
│   ├── adapters/   #   采集适配（Modbus/OPC UA/MQTT Sparkplug/CoAP）
│   ├── stream/     #   滑动窗口 + 规则引擎(T1) + ONNX 异常检测(T2)
│   ├── knowledge/  #   故障链与本地检索(T4)
│   ├── agent/      #   Agent 状态机与端侧 LLM
│   ├── queue/      #   离线 SQLite 队列与水位
│   └── cloud/      #   masking / mqtt_north / shadow
├── mgmt_agent/     # 管理代理（FastAPI，端口 9001：模型槽位管理与 OTA 应用/回滚、边缘探测）
├── shared/         # 共享定义（schemas 数据模型、topics MQTT 主题、errors 错误码）
├── sim/            # 设备模拟器（modbus/opcua/mqtt/coap）
├── tests/          # 测试用例（pytest）
├── .env.example    # 环境变量模板
├── pyproject.toml  # 项目与依赖声明
└── README.md
```

---

## 九、依赖管理声明

> **本项目依赖由 uv 统一管理，依赖声明位于 `pyproject.toml`（运行期在 `[project.dependencies]`，开发期在 `[dependency-groups].dev`），版本由 `uv.lock` 锁定；请勿新增或使用 `requirements.txt`。**