# 真机迁移检查清单

> 本清单对应方案文档 §7.3「开发期 Docker 模拟方案」到真机的迁移验证流程。
> **核心承诺**：模拟环境与生产环境共用同一份镜像与同一份配置，差异只在
> `docker-compose.edge.yml` 的资源限制与硬件加速开关。**代码零改动即可上真机**。

## 零改码迁移验证

- [ ] Docker 镜像在目标硬件上构建成功（`docker build -t edge-ai-agent:1.0.0 .`）
- [ ] `docker-compose.edge.yml` 资源限制正确（`cpus: 2.0` / `mem_limit: 4g`）
- [ ] systemd 服务文件安装并启动（`systemctl enable --now edge-agent.service`）
- [ ] 管理代理先于主应用启动（`mgmt-agent.service` 的 `Before=edge-agent.service`）
- [ ] 健康检查端点响应正常（`curl http://localhost:9000/health` 返回 200）
- [ ] MQTT 连接建立（`mosquitto_sub -t '#' -v` 能收到 telemetry）
- [ ] 模型加载成功（`/health` 返回 `llm: true` 或显式降级标记）
- [ ] 推理延迟在可接受范围（P95 < 1.2s）
- [ ] 内存使用在 4GB 限制内（峰值 ≤ 3.0GB，预留 1GB 余量）
- [ ] 日志正常输出到 journal（`journalctl -u edge-agent -f`）

## 配置差异（仅配置改动，零代码改动）

- [ ] `GW_ID` 设置为目标设备 ID（替换 `gw-sim-001`）
- [ ] `CLOUD_MQTT` 设置为目标 broker 地址（替换 `mosquitto:1883`）
- [ ] `CLOUD_API` 设置为目标云端 API 地址（替换 `host.docker.internal:8000`）
- [ ] `EDGE_LLM_MODEL` 根据硬件能力选择模型（NPU 可用更大模型）
- [ ] NPU 加速开关（如适用，设置 `EDGE_NPU_DEVICE` 环境变量）
- [ ] `EDGE_LLM_CTX` / `EDGE_LLM_MAXTOK` 根据内存预算调整
- [ ] mTLS 证书与设备证书配置（替换模拟用的 `hmac_salt`）

## 真机重测指标

> 指标口径来自方案文档 §3.5 与 §7.5。

- [ ] 告警路径 T1 P95 < 50ms
- [ ] 判别路径 T2 P95 < 200ms
- [ ] LLM 初判 T4 P95 < 1.2s
- [ ] 容器峰值内存 ≤ 3.0GB（4GB 限制下留 1GB 余量）
- [ ] 断网自治 ≥ 24h（本地队列 + ring buffer 不丢事件）
- [ ] OTA 成功率 ≥ 99%（A/B 双槽切换 + sha256 校验）
- [ ] 缓存命中率 ≥ 80%（本地知识库命中）
- [ ] 离线补传零丢失（重连后 watermark 对账一致）

## 安全加固验证

- [ ] 容器以非 root 用户运行（`docker inspect` 确认 `User=10001`）
- [ ] 根文件系统 read_only（`docker inspect` 确认 `ReadonlyRootfs=true`）
- [ ] 能力裁剪 `cap_drop: ALL`（`docker inspect` 确认无残留 capability）
- [ ] systemd `NoNewPrivileges=true` / `ProtectSystem=strict` 生效
- [ ] 模型签名校验通过（未签名模型拒绝加载）
- [ ] mTLS 双向认证建立（设备证书 + 云端证书互信）

## 可观测性验证

- [ ] Prometheus 抓取 `/metrics` 端点成功
- [ ] Grafana 看板 `deploy/grafana/dashboard.json` 导入成功
- [ ] 关键指标可见：`inference_p95_ms` / `queue_depth` / `offline_duration`
- [ ] 告警规则生效（离线超时 / 队列积压 / 温度过高 / 内存过高）
- [ ] 日志 ring buffer 保留 24h，断网期不丢