"""七阶段模型工厂流水线 + 异步任务管理（不依赖 Celery）。

流水线阶段（详见 plan/端云协同边缘 AI Agent 网关方案（V1.0）.md §4.2）：
① 数据回流    云端接收 → 影子评估结果 / 人工审核修正 / 端侧误报样本
② 清洗标注    去重、脱敏、分层抽样；审核结论作为黄金标签
③ 蒸馏        teacher(cloud) → student(0.6B/1.7B)；温度 T 需调优
④ 剪枝        结构化剪枝（去冗余注意力头/FFN 通道）
⑤ 量化        FP16 → AWQ/GPTQ → GGUF Q4_K_M（校准集含工业语料）
⑥ 校准评测    端侧回放集（Top-50 真实案例）→ 准确率 / P95 延迟 / 峰值内存
⑦ 注册        写入 model_registry（含 sha256 + 指标 + 前置依赖），产出 ModelManifest

流水线纪律：
- 每阶段都留基线产物（artifacts dict 记录路径，蒸馏/剪枝/量化各留一版产物）
- 校准集必须含工业语料
- 评测必须含端侧回放
- 蒸馏温度不可拍脑袋（显式配置或经调优的默认值，非正数则拒绝）
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from cloud.db import ModelRegistry
from shared.schemas import ModelManifest

log = logging.getLogger(__name__)

# 蒸馏温度默认值（经调优，不可拍脑袋；调用方可显式覆盖）
_DEFAULT_DISTILL_TEMPERATURE = 4.0
# 量化格式默认值
_DEFAULT_QUANT_FORMAT = "q4_k_m"
# 评测端侧回放集大小（Top-50 真实案例）
_EVAL_REPLAY_TOP_K = 50


class PipelineStage(Enum):
    """七阶段流水线阶段。"""

    data_reflux = "data_reflux"
    clean_label = "clean_label"
    distill = "distill"
    prune = "prune"
    quantize = "quantize"
    evaluate = "evaluate"
    register = "register"


# 默认完整流水线阶段顺序
DEFAULT_STAGES: list[PipelineStage] = [
    PipelineStage.data_reflux,
    PipelineStage.clean_label,
    PipelineStage.distill,
    PipelineStage.prune,
    PipelineStage.quantize,
    PipelineStage.evaluate,
    PipelineStage.register,
]


class JobStatus(Enum):
    """任务状态。"""

    pending = "pending"
    running = "running"
    stage_done = "stage_done"
    completed = "completed"
    failed = "failed"


@dataclass
class FactoryJob:
    """模型工厂流水线任务。"""

    job_id: str
    model_id: str
    target_version: str
    stages: list[PipelineStage]
    current_stage: PipelineStage | None = None
    status: JobStatus = JobStatus.pending
    # stage.name -> 产物路径
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # 阶段间中间数据传递（raw_data / cleaned / manifest）
    context: dict[str, Any] = field(default_factory=dict)
    # 失败时的错误信息
    error: str | None = None
    # 流水线配置（distill_temperature / quant_format / target_hardware …）
    config: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """更新 updated_at 时间戳。"""
        self.updated_at = datetime.now(UTC)


class ModelFactory:
    """七阶段模型工厂流水线 + 异步任务管理。

    用 threading 管理后台任务，不依赖 Celery。
    """

    def __init__(
        self,
        artifact_dir: Path,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.artifact_dir = artifact_dir
        self.session_factory = session_factory
        # 异步任务管理
        self._jobs: dict[str, FactoryJob] = {}
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------ #
    # 任务创建
    # ------------------------------------------------------------------ #

    def create_job(
        self,
        model_id: str,
        target_version: str,
        config: dict[str, Any] | None = None,
    ) -> FactoryJob:
        """创建流水线任务（status=pending）。

        Args:
            model_id: 模型标识，如 "anomaly-det"。
            target_version: 目标版本号，如 "0.2.0"。
            config: 流水线配置，可包含：
                - stages: 自定义阶段列表（默认全部七阶段）
                - distill_temperature: 蒸馏温度（默认 4.0，必须为正数）
                - quant_format: 量化格式（默认 "q4_k_m"）
                - target_hardware: 目标硬件列表
                - runtime: 运行时（如 "llm" / "onnxruntime"）

        Returns:
            FactoryJob: 创建的任务对象。

        Raises:
            ValueError: model_id / target_version 为空或蒸馏温度非法。
        """
        if not model_id:
            raise ValueError("model_id 不能为空")
        if not target_version:
            raise ValueError("target_version 不能为空")

        cfg = dict(config or {})
        stages = self._parse_stages(cfg.pop("stages", None))
        # 蒸馏温度校验（不可拍脑袋：显式配置或经调优默认值，非正数则拒绝）
        temperature = cfg.get("distill_temperature", _DEFAULT_DISTILL_TEMPERATURE)
        if not isinstance(temperature, int | float) or temperature <= 0:
            raise ValueError(f"distill_temperature 必须为正数，得到 {temperature!r}")
        cfg["distill_temperature"] = float(temperature)

        job = FactoryJob(
            job_id=uuid.uuid4().hex,
            model_id=model_id,
            target_version=target_version,
            stages=stages,
            config=cfg,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        log.info(
            "created job %s for %s@%s (%d stages)",
            job.job_id,
            model_id,
            target_version,
            len(stages),
        )
        return job

    @staticmethod
    def _parse_stages(stages: Any) -> list[PipelineStage]:
        """解析阶段配置，支持 None（默认全部）/ list[str] / list[PipelineStage]。"""
        if stages is None:
            return list(DEFAULT_STAGES)
        result: list[PipelineStage] = []
        for s in stages:
            if isinstance(s, PipelineStage):
                result.append(s)
            elif isinstance(s, str):
                result.append(PipelineStage(s))
            else:
                raise TypeError(f"stage 元素类型不支持: {s!r}")
        if not result:
            raise ValueError("stages 不能为空")
        return result

    # ------------------------------------------------------------------ #
    # 阶段执行
    # ------------------------------------------------------------------ #

    def run_stage(self, job: FactoryJob, stage: PipelineStage) -> FactoryJob:
        """执行单个阶段，更新 job 状态与产物。

        Raises:
            ValueError: 前置依赖未满足。
            RuntimeError: 阶段执行失败。
        """
        job.status = JobStatus.running
        job.current_stage = stage
        job.touch()
        log.info("job %s running stage %s", job.job_id, stage.name)
        try:
            self._dispatch_stage(job, stage)
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            job.touch()
            log.exception("job %s stage %s failed", job.job_id, stage.name)
            raise
        job.status = JobStatus.stage_done
        job.touch()
        return job

    def _dispatch_stage(self, job: FactoryJob, stage: PipelineStage) -> None:
        """根据阶段枚举分发到对应实现，并记录产物。"""
        if stage is PipelineStage.data_reflux:
            raw = self._data_reflux(job)
            job.context["raw_data"] = raw
            job.artifacts[stage.name] = str(self._write_artifact(job, "raw_data.json", raw))
        elif stage is PipelineStage.clean_label:
            raw_data = job.context.get("raw_data")
            if raw_data is None:
                raise ValueError("clean_label 前置阶段 data_reflux 未执行")
            cleaned = self._clean_label(job, raw_data)
            job.context["cleaned"] = cleaned
            job.artifacts[stage.name] = str(self._write_artifact(job, "cleaned.json", cleaned))
        elif stage is PipelineStage.distill:
            cleaned_data = job.context.get("cleaned")
            if cleaned_data is None:
                raise ValueError("distill 前置阶段 clean_label 未执行")
            job.artifacts[stage.name] = str(self._distill(job, cleaned_data))
        elif stage is PipelineStage.prune:
            distilled = job.artifacts.get(PipelineStage.distill.name)
            if not distilled:
                raise ValueError("prune 前置阶段 distill 未执行")
            job.artifacts[stage.name] = str(self._prune(job, Path(distilled)))
        elif stage is PipelineStage.quantize:
            pruned = job.artifacts.get(PipelineStage.prune.name)
            if not pruned:
                raise ValueError("quantize 前置阶段 prune 未执行")
            job.artifacts[stage.name] = str(self._quantize(job, Path(pruned)))
        elif stage is PipelineStage.evaluate:
            quantized = job.artifacts.get(PipelineStage.quantize.name)
            if not quantized:
                raise ValueError("evaluate 前置阶段 quantize 未执行")
            metrics = self._evaluate(job, Path(quantized))
            job.metrics.update(metrics)
            job.artifacts[stage.name] = str(self._write_artifact(job, "metrics.json", metrics))
        elif stage is PipelineStage.register:
            quantized = job.artifacts.get(PipelineStage.quantize.name)
            if not quantized:
                raise ValueError("register 前置阶段 quantize 未执行")
            manifest = self._register(job, Path(quantized), job.metrics)
            job.context["manifest"] = manifest
            job.artifacts[stage.name] = str(
                self._write_artifact(job, "manifest.json", manifest.model_dump())
            )
        else:  # pragma: no cover — 枚举穷尽
            raise RuntimeError(f"未知阶段: {stage}")

    def run_all(self, job: FactoryJob) -> FactoryJob:
        """执行完整流水线（按 job.stages 顺序）。"""
        job.status = JobStatus.running
        job.touch()
        for stage in job.stages:
            self.run_stage(job, stage)
        job.status = JobStatus.completed
        job.touch()
        log.info(
            "job %s completed: %d stages, %d artifacts",
            job.job_id,
            len(job.stages),
            len(job.artifacts),
        )
        return job

    # ------------------------------------------------------------------ #
    # 阶段实现
    # ------------------------------------------------------------------ #

    def _job_dir(self, job: FactoryJob) -> Path:
        """获取任务产物目录（自动创建）。"""
        d = self.artifact_dir / job.job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_artifact(self, job: FactoryJob, filename: str, data: Any) -> Path:
        """将阶段产物以 JSON 写入产物目录。"""
        p = self._job_dir(job) / filename
        p.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2), "utf-8")
        return p

    def _data_reflux(self, job: FactoryJob) -> dict[str, Any]:
        """阶段①数据回流（mock：返回影子评估 + 人工审核 + 端侧误报样本）。"""
        samples = [
            {
                "id": f"s{i}",
                "metric": "vibration_rms",
                "value": 5.0 + i * 0.1,
                "severity": "medium",
                "label": "bearing_wear",
                "site": "shanghai",
            }
            for i in range(20)
        ]
        false_positives = [
            {
                "id": f"fp{i}",
                "metric": "temperature",
                "value": 80.0 + i,
                "severity": "low",
                "label": "normal",
            }
            for i in range(5)
        ]
        return {
            "source": "edge_shadow+human_review+false_positive",
            "samples": samples,
            "false_positives": false_positives,
            "count": len(samples) + len(false_positives),
        }

    def _clean_label(self, job: FactoryJob, raw_data: dict[str, Any]) -> dict[str, Any]:
        """阶段②清洗标注：去重、脱敏、分层抽样；审核结论作为黄金标签。"""
        all_samples = list(raw_data.get("samples", [])) + list(raw_data.get("false_positives", []))

        # 去重（按 id）
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for s in all_samples:
            sid = s.get("id")
            if sid and sid not in seen:
                seen.add(sid)
                deduped.append(dict(s))  # 浅拷贝避免污染输入

        # 脱敏：掩码 site 字段
        for s in deduped:
            if "site" in s:
                s["site"] = "***"

        # 分层抽样（按 severity 分层统计）
        layers: dict[str, int] = {}
        for s in deduped:
            key = s.get("severity", "unknown")
            layers[key] = layers.get(key, 0) + 1

        # 审核结论作为黄金标签
        for s in deduped:
            s["gold_label"] = s.get("label")

        return {
            "cleaned_samples": deduped,
            "layers": layers,
            "count": len(deduped),
            "dedup_removed": len(all_samples) - len(deduped),
        }

    def _distill(self, job: FactoryJob, cleaned: dict[str, Any]) -> Path:
        """阶段③蒸馏（mock：生成 mock 模型文件）。

        teacher(cloud) → student(0.6B/1.7B)；温度 T 需调优。
        """
        temperature = job.config.get("distill_temperature", _DEFAULT_DISTILL_TEMPERATURE)
        student_size = job.config.get("student_size", "0.6B")
        content = {
            "stage": "distill",
            "teacher": "cloud-llm-7b",
            "student": student_size,
            "temperature": temperature,
            "sample_count": cleaned.get("count", 0),
        }
        p = self._job_dir(job) / "distilled.bin"
        p.write_text(json.dumps(content, ensure_ascii=False), "utf-8")
        return p

    def _prune(self, job: FactoryJob, distilled: Path) -> Path:
        """阶段④剪枝（mock：复制 + 标记剪枝信息）。

        结构化剪枝（去冗余注意力头/FFN 通道）。
        """
        base = distilled.read_text("utf-8") if distilled.exists() else "{}"
        content = json.loads(base)
        content["stage"] = "prune"
        content["pruned_heads"] = 12
        content["pruned_ffn_channels"] = 64
        content["sparsity"] = 0.3
        p = self._job_dir(job) / "pruned.bin"
        p.write_text(json.dumps(content, ensure_ascii=False), "utf-8")
        return p

    def _quantize(self, job: FactoryJob, pruned: Path) -> Path:
        """阶段⑤量化（mock：生成 mock GGUF）。

        FP16 → AWQ/GPTQ → GGUF Q4_K_M（校准集必须包含工业语料）。
        """
        quant_format = job.config.get("quant_format", _DEFAULT_QUANT_FORMAT)
        base = pruned.read_text("utf-8") if pruned.exists() else "{}"
        content = json.loads(base)
        content["stage"] = "quantize"
        content["quant_format"] = quant_format
        content["calibration_set"] = "industrial_corpus"  # 校准集含工业语料
        p = self._job_dir(job) / "quantized.gguf"
        p.write_text(json.dumps(content, ensure_ascii=False), "utf-8")
        return p

    def _evaluate(self, job: FactoryJob, quantized: Path) -> dict[str, Any]:
        """阶段⑥校准评测（mock：返回指标 dict）。

        端侧回放集（Top-50 真实案例）→ 准确率 / P95 延迟 / 峰值内存。
        """
        return {
            "eval_acc": 0.92,
            "p95_latency_ms": 45.0,
            "peak_mem_mb": 512,
            "replay_top_k": _EVAL_REPLAY_TOP_K,
            "replay_source": "edge_real_cases",
            "calibration_corpus": "industrial",
            "quantized_path": str(quantized),
        }

    def _register(self, job: FactoryJob, quantized: Path, metrics: dict[str, Any]) -> ModelManifest:
        """阶段⑦注册到 ModelRegistry，产出 ModelManifest。"""
        data = quantized.read_bytes() if quantized.exists() else b""
        sha256 = hashlib.sha256(data).hexdigest()
        size_bytes = len(data)

        target_hardware = job.config.get("target_hardware", ["edge-arm64"])
        runtime = job.config.get("runtime", "llm")
        requires_firmware = job.config.get("requires_firmware", "0.1.0")
        preproc_ver = job.config.get("preproc_ver", "v1")
        quant = job.config.get("quant_format", _DEFAULT_QUANT_FORMAT)

        manifest = ModelManifest(
            model_id=job.model_id,
            version=job.target_version,
            artifact_url=f"file://{quantized}",
            sha256=sha256,
            size_bytes=size_bytes,
            quant=quant,
            runtime=runtime,
            target_hardware=list(target_hardware),
            requires_firmware=requires_firmware,
            preproc_ver=preproc_ver,
            metrics_at_register=dict(metrics),
            rollout={"strategy": "ring", "rings": []},
        )

        # 写入 ModelRegistry
        with self.session_factory() as session:
            row = ModelRegistry(
                model_id=manifest.model_id,
                version=manifest.version,
                sha256=manifest.sha256,
                manifest=manifest.model_dump(),
                artifact_path=str(quantized),
            )
            session.add(row)
            session.commit()

        return manifest

    # ------------------------------------------------------------------ #
    # 异步任务管理
    # ------------------------------------------------------------------ #

    def submit_job(
        self,
        model_id: str,
        target_version: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """提交异步任务，返回 job_id。

        内部用 threading 管理后台任务（不依赖 Celery）。
        """
        job = self.create_job(model_id, target_version, config)

        def _run() -> None:
            try:
                self.run_all(job)
            except Exception:  # noqa: BLE001 — 后台线程需吞掉异常
                # run_stage 已设置 status=failed，此处仅记录日志
                log.exception("async job %s failed", job.job_id)

        t = threading.Thread(target=_run, daemon=True, name=f"factory-{job.job_id}")
        with self._lock:
            self._threads[job.job_id] = t
        t.start()
        return job.job_id

    def get_job_status(self, job_id: str) -> FactoryJob:
        """查询任务状态。

        Raises:
            KeyError: job_id 不存在。
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"job_id 不存在: {job_id}")
        return job
