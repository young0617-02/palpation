# syntax=docker/dockerfile:1.7
# =============================================================================
# Edge AI Agent - 边缘 Agent 镜像
# 方案文档 §7.3：开发期 Docker 模拟环境（Phase 0 完整复现）
#   docker run --cpus=2 --memory=4g --read-only --tmpfs /tmp ...
# 安全要求（§7.6）：
#   - 容器非 root 用户运行
#   - read_only 根文件系统兼容（仅 /tmp 与挂载卷可写）
#   - 最小权限，运行时 cap_drop ALL（在 docker-compose 中声明）
# =============================================================================
FROM python:3.12-slim AS base

# 元数据标签
LABEL org.opencontainers.image.title="edge-ai-agent" \
      org.opencontainers.image.description="Edge-cloud collaborative AI Agent gateway" \
      org.opencontainers.image.source="https://github.com/palpation/edge-ai-agent" \
      org.opencontainers.image.version="1.0.0"

# ----- 1. 系统运行时依赖 -------------------------------------------------------
# libgomp1: onnxruntime OpenMP 运行时依赖
# libgl1 / libglib2.0-0: 部分 wheel（如 llama-cpp-python 预编译版）的间接依赖
# curl: 用于容器 HEALTHCHECK 探测 /health 端点
# tini: 轻量 init，正确处理信号转发与僵尸进程回收
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# ----- 2. 创建非 root 用户 ----------------------------------------------------
# 固定 UID/GID 10001 以保证宿主机挂载卷权限一致
RUN groupadd --system --gid 10001 edge-agent \
    && useradd --system --uid 10001 --gid edge-agent \
        --home-dir /opt/edge-agent --shell /usr/sbin/nologin edge-agent

# ----- 3. 安装 Python 依赖 ----------------------------------------------------
# 优先使用 uv（astral）做快速依赖解析与安装；离线/无 uv 环境回退到 pip install -e .
WORKDIR /opt/edge-agent
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/edge-agent/.venv/bin:${PATH}"

# 先复制依赖清单以最大化 Docker 层缓存命中
COPY pyproject.toml uv.lock* ./

# 安装 uv 并用其同步依赖到独立 venv（不含可选的 llm 编译依赖）
RUN pip install --no-cache-dir "uv>=0.4.0" \
    && uv venv /opt/edge-agent/.venv \
    && uv pip install --python /opt/edge-agent/.venv/bin/python . \
    && rm -rf /root/.cache /root/.local

# ----- 4. 复制源码与配置 ------------------------------------------------------
COPY edge/ ./edge/
COPY cloud/ ./cloud/
COPY mgmt_agent/ ./mgmt_agent/
COPY shared/ ./shared/
COPY sim/ ./sim/
COPY config/ ./config/
COPY README.md ./

# 模型与数据目录由 volume 挂载，这里仅建立占位目录与权限
RUN mkdir -p /opt/edge-agent/models /var/lib/edge-agent \
    && chown -R edge-agent:edge-agent /opt/edge-agent /var/lib/edge-agent

# ----- 5. 运行时配置 ----------------------------------------------------------
USER edge-agent
WORKDIR /opt/edge-agent

# 暴露边缘 API 端口（默认 9000，由 EDGE_API_PORT 覆盖）
EXPOSE 9000

# 健康检查：探测 FastAPI /health 端点
# --cpus=2 --memory=4g 模拟边缘 2C4G 环境（在 docker-compose.edge.yml 声明）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:9000/health || exit 1

# tini 作为 PID 1 处理信号转发与僵尸进程回收
# ENTRYPOINT 仅包含 tini，实际启动命令由 CMD 提供
# compose 的 command 会覆盖 CMD（而非追加到 ENTRYPOINT），从而正确切换服务入口
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/edge-agent/.venv/bin/python", "-m", "edge.main"]