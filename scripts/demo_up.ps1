# =============================================================================
# demo_up.ps1 — 一键启动 Palpation 端到端演示环境（Windows）
#   顺序：mosquitto broker(1883) -> cloud(8000) -> edge(9000) -> sim 数据注入
#   前置：已执行 uv sync（见 docs/用户操作手册.md 第 4.2 节）
#   用法：powershell -ExecutionPolicy Bypass -File scripts\demo_up.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# 依赖与虚拟环境由 uv 统一管理（uv sync 创建 .venv）；
# 服务统一经 `uv run --no-sync` 启动，避免每次触发依赖同步检查。
$uv     = "uv"
$uvArgs = @("run", "--no-sync")

# 0) mosquitto 配置（本地允许匿名连接）
$conf = "C:\mosquitto\demo.conf"
if (-not (Test-Path $conf)) {
@"
listener 1883 127.0.0.1
allow_anonymous true
"@ | Set-Content -Encoding ascii $conf
}

$env:CLOUD_MQTT_HOST = "127.0.0.1"
$env:CLOUD_MQTT_PORT = "1883"

# 1) broker
Start-Process -FilePath "C:\mosquitto\mosquitto.exe" -ArgumentList "-c", $conf `
  -RedirectStandardOutput "$logs\mosquitto.log" -RedirectStandardError "$logs\mosquitto.err" -WindowStyle Hidden
Start-Sleep -Seconds 2

# 2) cloud
Start-Process -FilePath $uv -ArgumentList ($uvArgs + @("uvicorn", "cloud.main:app", "--host", "0.0.0.0", "--port", "8000")) `
  -WorkingDirectory $root -RedirectStandardOutput "$logs\cloud.log" -RedirectStandardError "$logs\cloud.err" -WindowStyle Hidden
Start-Sleep -Seconds 4

# 3) edge
Start-Process -FilePath $uv -ArgumentList ($uvArgs + @("uvicorn", "edge.main:app", "--host", "0.0.0.0", "--port", "9000")) `
  -WorkingDirectory $root -RedirectStandardOutput "$logs\edge.log" -RedirectStandardError "$logs\edge.err" -WindowStyle Hidden
Start-Sleep -Seconds 4

# 4) 设备模拟器（持续注入传感器数据，5s 后触发故障）
Start-Process -FilePath $uv -ArgumentList ($uvArgs + @("python", "-m", "sim.mqtt_pump", "--host", "127.0.0.1", "--gw", "gw-sim-001", "--device", "pump-de-01", "--rate", "2.0", "--fault-after", "5")) `
  -WorkingDirectory $root -RedirectStandardOutput "$logs\sim.log" -RedirectStandardError "$logs\sim.err" -WindowStyle Hidden

Write-Host "演示环境已启动: cloud=http://127.0.0.1:8000  edge=http://127.0.0.1:9000  broker=127.0.0.1:1883"