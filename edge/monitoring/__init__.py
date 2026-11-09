"""E3 监控可观测模块：Prometheus 指标导出 + 本地探针 + ring buffer 日志。

子模块：
- metrics: 边缘指标收集器（纯 Python 实现 Prometheus 文本格式导出）
- probes: 本地探针集合（推理/磁盘/温度/队列/模型/MQTT）
- ring_buffer: 环形缓冲区日志（24h 保留，线程安全）
"""
