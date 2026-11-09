# syntax=docker/dockerfile:1.7
# =============================================================================
# sim/Dockerfile — 设备模拟器镜像（Modbus / OPC UA / MQTT pump / CoAP sensor）
# 用于 docker-compose.edge.yml 中 device-sim-* 服务
# =============================================================================
FROM python:3.12-slim AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# 非 root 用户
RUN groupadd --system --gid 10002 sim-agent \
    && useradd --system --uid 10002 --gid sim-agent \
        --home-dir /app --shell /usr/sbin/nologin sim-agent

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/.venv/bin:${PATH}"

# 先复制依赖清单以最大化层缓存
COPY pyproject.toml uv.lock* ./

# 安装 uv 并用其同步依赖到独立 venv
RUN pip install --no-cache-dir "uv>=0.4.0" \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python . \
    && rm -rf /root/.cache /root/.local

# 复制源码
COPY sim/ ./sim/
COPY shared/ ./shared/
COPY config/ ./config/

USER sim-agent
WORKDIR /app

# 默认启动 Modbus 设备模拟器，compose 可通过 command 覆盖
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/.venv/bin/python", "sim/modbus_device.py"]