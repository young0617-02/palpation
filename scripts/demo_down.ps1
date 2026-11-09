# =============================================================================
# demo_down.ps1 — 停止 Palpation 演示环境
#   停止 mosquitto 及经 `uv run` 启动的 cloud/edge/sim（进程链 uv.exe -> uvicorn.exe -> python.exe）
#   用法：powershell -ExecutionPolicy Bypass -File scripts\demo_down.ps1
# =============================================================================
$ErrorActionPreference = "SilentlyContinue"
Get-Process mosquitto | Stop-Process -Force
# 经 `uv run` 启动的服务形成 uv.exe -> uvicorn.exe -> python.exe 进程链，
# 故按进程名（三类）+ 命令行关键字匹配，确保全部清理。
Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -in @("python.exe", "uv.exe", "uvicorn.exe") -and
        $_.CommandLine -match "uvicorn|sim\.mqtt_pump"
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Host "演示环境已停止"