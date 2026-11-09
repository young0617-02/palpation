#!/usr/bin/env bash
# 通用 CI 检查脚本：ruff check + ruff format --check + mypy + pytest
# 端云协同边缘 AI Agent 网关 (Phase 0)
# 用法：bash scripts/check.sh
set -euo pipefail

# 平台适配：Windows 用 Scripts/python.exe，Linux/macOS 用 bin/python
if [[ -f .venv/Scripts/python.exe ]]; then
  VENV_PY=".venv/Scripts/python.exe"
elif [[ -f .venv/bin/python ]]; then
  VENV_PY=".venv/bin/python"
else
  echo "ERROR: 找不到虚拟环境 python，请先运行 uv sync --all-groups" >&2
  exit 1
fi

# mypy 类型检查的源码目录
SRC_DIRS="cloud edge mgmt_agent shared sim"

echo "=== ruff check ==="
ruff check .

echo "=== ruff format --check ==="
ruff format --check . --exclude tests --exclude plan

echo "=== mypy ==="
"$VENV_PY" -m mypy $SRC_DIRS

echo "=== pytest ==="
"$VENV_PY" -m pytest -q

echo "=== ALL CHECKS PASSED ==="