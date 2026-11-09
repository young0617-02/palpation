#!/usr/bin/env bash
# 端到端联调脚本
# 启动 mosquitto + sim 模拟器 + edge + cloud，验证完整数据流：
#   传感器 → 边缘采集 → 规则/异常检测 → 事件上报 → 云端入库 → 诊断 → OTA
#
# 前置条件：
#   - mosquitto 已安装并监听 127.0.0.1:1883（或通过环境变量指定）
#   - .venv 已创建且依赖已安装
#
# 用法：
#   bash scripts/e2e.sh
#
# 环境变量（可选）：
#   MQTT_HOST       MQTT broker 地址（默认 127.0.0.1）
#   MQTT_PORT       MQTT broker 端口（默认 1883）
#   CLOUD_PORT      cloud HTTP 端口（默认 8000）
#   EDGE_PORT       edge HTTP 端口（默认 9000）
#   FAULT_AFTER     sim 注入故障延迟秒数（默认 10）
#   WAIT_SECONDS    数据流等待秒数（默认 20）
#   PYTHON          Python 解释器路径（默认自动检测）
set -euo pipefail

# --------------------------------------------------------------------------- #
# 路径与配置
# --------------------------------------------------------------------------- #
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MQTT_HOST="${MQTT_HOST:-127.0.0.1}"
MQTT_PORT="${MQTT_PORT:-1883}"
CLOUD_PORT="${CLOUD_PORT:-8000}"
EDGE_PORT="${EDGE_PORT:-9000}"
FAULT_AFTER="${FAULT_AFTER:-10}"
WAIT_SECONDS="${WAIT_SECONDS:-20}"

# Python 解释器检测
if [[ -f .venv/Scripts/python.exe ]]; then
  PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
elif [[ -f .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

# 后台进程 PID 收集
PIDS=()

cleanup() {
  echo ""
  echo "=== 清理后台进程 ==="
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  终止 PID $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "=== 清理完成 ==="
}
trap cleanup EXIT INT TERM

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# --------------------------------------------------------------------------- #
# 1. 前置检查
# --------------------------------------------------------------------------- #
echo "=== 端到端联调脚本 ==="
echo "  MQTT:  ${MQTT_HOST}:${MQTT_PORT}"
echo "  Cloud: http://127.0.0.1:${CLOUD_PORT}"
echo "  Edge:  http://127.0.0.1:${EDGE_PORT}"
echo "  Python: ${PYTHON}"
echo ""

log "[1/8] 前置检查..."

# 检查 Python
if ! "$PYTHON" --version &>/dev/null; then
  echo "ERROR: Python 解释器不可用: ${PYTHON}" >&2
  exit 1
fi

# 检查 MQTT broker 连通性
MQTT_OK=false
if command -v mosquitto_sub &>/dev/null; then
  # 用 mosquitto_sub 探测（超时 3 秒）
  if timeout 3 mosquitto_sub -h "$MQTT_HOST" -p "$MQTT_PORT" -t "#" -C 1 &>/dev/null; then
    MQTT_OK=true
  fi
fi
if [[ "$MQTT_OK" == "false" ]]; then
  # 尝试 TCP 连接
  if (echo > "/dev/tcp/${MQTT_HOST}/${MQTT_PORT}") 2>/dev/null; then
    MQTT_OK=true
  fi
fi
if [[ "$MQTT_OK" == "false" ]]; then
  echo "WARNING: MQTT broker ${MQTT_HOST}:${MQTT_PORT} 不可达" >&2
  echo "  请先启动 mosquitto:  mosquitto -p ${MQTT_PORT}" >&2
  echo "  或安装 amqtt:  pip install amqtt" >&2
  # 不退出，继续尝试（broker 可能在后续启动）
fi
log "  MQTT broker: $([[ "$MQTT_OK" == "true" ]] && echo "可达" || echo "不可达")"

# --------------------------------------------------------------------------- #
# 2. 启动 cloud
# --------------------------------------------------------------------------- #
log "[2/8] 启动 cloud (port ${CLOUD_PORT})..."
"$PYTHON" -m uvicorn cloud.main:app --host 0.0.0.0 --port "$CLOUD_PORT" &
CLOUD_PID=$!
PIDS+=("$CLOUD_PID")
log "  cloud PID=${CLOUD_PID}"

# 等待 cloud 就绪
for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:${CLOUD_PORT}/health" &>/dev/null; then
    log "  cloud 就绪"
    break
  fi
  sleep 1
  if [[ $i -eq 15 ]]; then
    echo "ERROR: cloud 启动超时" >&2
    exit 1
  fi
done

# --------------------------------------------------------------------------- #
# 3. 启动 edge
# --------------------------------------------------------------------------- #
log "[3/8] 启动 edge (port ${EDGE_PORT})..."
CLOUD_MQTT_HOST="$MQTT_HOST" CLOUD_MQTT_PORT="$MQTT_PORT" \
  "$PYTHON" -m uvicorn edge.main:app --host 0.0.0.0 --port "$EDGE_PORT" &
EDGE_PID=$!
PIDS+=("$EDGE_PID")
log "  edge PID=${EDGE_PID}"
sleep 3
log "  edge 已启动"

# --------------------------------------------------------------------------- #
# 4. 启动 sim 模拟器
# --------------------------------------------------------------------------- #
log "[4/8] 启动 sim 模拟器 (fault_after=${FAULT_AFTER}s)..."
"$PYTHON" -m sim.mqtt_pump \
  --host "$MQTT_HOST" \
  --port "$MQTT_PORT" \
  --gw "gw-sim-001" \
  --device "pump-de-01" \
  --rate 2.0 \
  --fault-after "$FAULT_AFTER" &
SIM_PID=$!
PIDS+=("$SIM_PID")
log "  sim PID=${SIM_PID}"

# --------------------------------------------------------------------------- #
# 5. 等待数据流
# --------------------------------------------------------------------------- #
log "[5/8] 等待数据流 (${WAIT_SECONDS}s)..."
log "  sim 发布传感器数据 → edge 规则检测 → cloud 入库 + 诊断"
sleep "$WAIT_SECONDS"

# --------------------------------------------------------------------------- #
# 6. 查询 cloud API 验证
# --------------------------------------------------------------------------- #
log "[6/8] 查询 cloud API 验证数据流..."

echo ""
echo "--- /health ---"
curl -sf "http://127.0.0.1:${CLOUD_PORT}/health" | "$PYTHON" -m json.tool 2>/dev/null || \
  curl -s "http://127.0.0.1:${CLOUD_PORT}/health"

echo ""
echo "--- /v1/events ---"
EVENTS=$(curl -s "http://127.0.0.1:${CLOUD_PORT}/v1/events")
echo "$EVENTS" | "$PYTHON" -m json.tool 2>/dev/null || echo "$EVENTS"

EVENT_COUNT=$(echo "$EVENTS" | "$PYTHON" -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
log "  事件数量: ${EVENT_COUNT}"

echo ""
echo "--- /v1/watermarks/gw-sim-001 ---"
curl -s "http://127.0.0.1:${CLOUD_PORT}/v1/watermarks/gw-sim-001" | "$PYTHON" -m json.tool 2>/dev/null || \
  echo "  (watermark 尚未生成)"

# --------------------------------------------------------------------------- #
# 7. OTA 流程验证
# --------------------------------------------------------------------------- #
log "[7/8] OTA 流程验证..."

# 注册设备
echo ""
echo "--- POST /v1/devices/register ---"
curl -sf -X POST "http://127.0.0.1:${CLOUD_PORT}/v1/devices/register" \
  -H "Content-Type: application/json" \
  -d '{"gw_id":"gw-sim-001","site":"lab"}' | "$PYTHON" -m json.tool 2>/dev/null || \
  log "  设备注册失败或已注册"

# 创建临时模型文件并注册
MODEL_FILE=$(mktemp /tmp/model_XXXX.onnx)
echo "fake onnx model bytes" > "$MODEL_FILE"
echo ""
echo "--- POST /v1/models/register ---"
curl -sf -X POST "http://127.0.0.1:${CLOUD_PORT}/v1/models/register" \
  -F "file=@${MODEL_FILE}" \
  -F "model_id=anomaly-det" \
  -F "version=0.2.0" | "$PYTHON" -m json.tool 2>/dev/null || \
  log "  模型注册失败"
rm -f "$MODEL_FILE"

# 创建 rollout
echo ""
echo "--- POST /v1/rollout ---"
curl -sf -X POST "http://127.0.0.1:${CLOUD_PORT}/v1/rollout" \
  -H "Content-Type: application/json" \
  -d '{"gw_id":"gw-sim-001","model_id":"anomaly-det","version":"0.2.0"}' \
  | "$PYTHON" -m json.tool 2>/dev/null || \
  log "  rollout 创建失败"

# --------------------------------------------------------------------------- #
# 8. 结果汇总
# --------------------------------------------------------------------------- #
log "[8/8] 联调结果汇总"
echo ""
echo "=== 联调结果 ==="
echo "  cloud 健康:     $(curl -sf http://127.0.0.1:${CLOUD_PORT}/health &>/dev/null && echo "OK" || echo "FAIL")"
echo "  edge 健康:      $(curl -sf http://127.0.0.1:${EDGE_PORT}/health &>/dev/null && echo "OK" || echo "FAIL")"
echo "  事件入库:       ${EVENT_COUNT} 条"
if [[ "$EVENT_COUNT" -gt 0 ]]; then
  echo "  数据流验证:     PASS (传感器→边缘→云端)"
else
  echo "  数据流验证:     WARN (未检测到事件，检查 MQTT 连通性)"
fi
echo ""

log "端到端联调脚本执行完毕"