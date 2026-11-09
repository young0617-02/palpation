# Makefile - 通用 CI 检查（ruff + mypy + pytest + uv）
# 端云协同边缘 AI Agent 网关 (Phase 0)
# 各 CI 平台可复用：make check / bash scripts/check.sh

# 平台适配：Windows 用 Scripts/python.exe，Linux/macOS 用 bin/python
ifeq ($(OS),Windows_NT)
	VENV_PY := .venv/Scripts/python.exe
else
	VENV_PY := .venv/bin/python
endif

RUFF   := ruff
MYPY   := $(VENV_PY) -m mypy
PYTEST := $(VENV_PY) -m pytest

# mypy 类型检查的源码目录
SRC_DIRS := cloud edge mgmt_agent shared sim

.PHONY: all check lint typecheck test install sync clean

# 默认目标：一键检查
all: check

# 一键检查（lint + typecheck + test）
check: lint typecheck test

# 代码风格检查（ruff check + ruff format --check）
# format 排除 tests（pyproject exclude 对 format 不生效）和 plan（文档伪代码）
lint:
	$(RUFF) check .
	$(RUFF) format --check . --exclude tests --exclude plan

# 类型检查（mypy）
typecheck:
	$(MYPY) $(SRC_DIRS)

# 单元测试（pytest）
test:
	$(PYTEST) -q

# 安装依赖（uv sync 全部分组）
install:
	uv sync --all-groups

# 同步依赖（离线模式，不联网）
sync:
	uv sync --no-sync --all-groups

# 清理缓存
clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +