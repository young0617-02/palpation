"""tests/test_model_factory.py：模型工厂七阶段流水线 + 异步任务管理测试。

用 tmp_path 隔离产物目录，内存 sqlite 隔离数据库，不依赖真实模型文件。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cloud.db import Base, ModelRegistry
from cloud.model_factory import (
    DEFAULT_STAGES,
    FactoryJob,
    JobStatus,
    ModelFactory,
    PipelineStage,
)
from shared.schemas import ModelManifest


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def session_factory():
    """每个用例独立的内存 sqlite session_factory。

    用 StaticPool + check_same_thread=False 让内存库跨线程共享同一连接，
    以支持 submit_job 后台线程访问主线程建好的表。
    """
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def factory(tmp_path: Path, session_factory):
    """模型工厂实例，产物目录在 tmp_path 下隔离。"""
    return ModelFactory(tmp_path / "artifacts", session_factory)


# --------------------------------------------------------------------------- #
# 1. 创建任务 → 执行完整流水线 → 七阶段全部完成
# --------------------------------------------------------------------------- #

def test_full_pipeline_all_seven_stages_completed(factory):
    """创建任务 → 执行完整流水线 → 七阶段全部完成。"""
    job = factory.create_job("anomaly-det", "0.2.0", {"distill_temperature": 3.0})
    assert job.status is JobStatus.pending
    assert len(job.stages) == 7
    assert job.stages == DEFAULT_STAGES

    result = factory.run_all(job)

    assert result.status is JobStatus.completed
    assert result.error is None
    # 七阶段全部完成，每阶段都有产物
    assert set(result.artifacts.keys()) == {s.name for s in DEFAULT_STAGES}


def test_full_pipeline_current_stage_is_last(factory):
    """完整流水线完成后 current_stage 为最后一个阶段。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    assert result.current_stage is PipelineStage.register


# --------------------------------------------------------------------------- #
# 2. 单个阶段执行 → 状态流转
# --------------------------------------------------------------------------- #

def test_single_stage_status_transition(factory):
    """单个阶段执行 → 状态从 pending → running → stage_done。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    assert job.status is JobStatus.pending

    result = factory.run_stage(job, PipelineStage.data_reflux)

    assert result.status is JobStatus.stage_done
    assert result.current_stage is PipelineStage.data_reflux
    assert PipelineStage.data_reflux.name in result.artifacts


def test_stage_done_after_each_stage(factory):
    """每执行一个阶段，current_stage 更新且产物累积。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    factory.run_stage(job, PipelineStage.data_reflux)
    assert job.current_stage is PipelineStage.data_reflux
    assert job.status is JobStatus.stage_done

    factory.run_stage(job, PipelineStage.clean_label)
    assert job.current_stage is PipelineStage.clean_label
    assert len(job.artifacts) == 2


# --------------------------------------------------------------------------- #
# 3. 每阶段基线产物保留 → artifacts dict
# --------------------------------------------------------------------------- #

def test_every_stage_has_artifact(factory):
    """每阶段基线产物保留 → artifacts dict 记录每阶段产物路径。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    for stage in DEFAULT_STAGES:
        assert stage.name in result.artifacts
        p = Path(result.artifacts[stage.name])
        assert p.exists(), f"阶段 {stage.name} 产物不存在: {p}"


def test_distill_prune_quantize_artifacts_are_files(factory):
    """蒸馏/剪枝/量化各留一版产物文件。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    assert Path(result.artifacts["distill"]).is_file()
    assert Path(result.artifacts["prune"]).is_file()
    assert Path(result.artifacts["quantize"]).is_file()
    # 量化产物为 .gguf
    assert result.artifacts["quantize"].endswith(".gguf")


# --------------------------------------------------------------------------- #
# 4. 评测指标生成 → metrics 包含 eval_acc, p95_latency_ms, peak_mem_mb
# --------------------------------------------------------------------------- #

def test_evaluate_metrics_contain_required_keys(factory):
    """评测指标包含 eval_acc, p95_latency_ms, peak_mem_mb。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    assert "eval_acc" in result.metrics
    assert "p95_latency_ms" in result.metrics
    assert "peak_mem_mb" in result.metrics
    assert 0 <= result.metrics["eval_acc"] <= 1
    assert result.metrics["p95_latency_ms"] > 0
    assert result.metrics["peak_mem_mb"] > 0


def test_evaluate_contains_edge_replay(factory):
    """评测必须含端侧回放（replay_source / replay_top_k）。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    assert result.metrics["replay_source"] == "edge_real_cases"
    assert result.metrics["replay_top_k"] == 50
    # 校准集必须含工业语料
    assert result.metrics["calibration_corpus"] == "industrial"


# --------------------------------------------------------------------------- #
# 5. 注册到 ModelRegistry → manifest 写入
# --------------------------------------------------------------------------- #

def test_register_writes_model_registry(factory, session_factory):
    """注册到 ModelRegistry → manifest 写入数据库。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    manifest: ModelManifest = result.context["manifest"]
    assert manifest.model_id == "anomaly-det"
    assert manifest.version == "0.2.0"
    assert len(manifest.sha256) == 64
    assert manifest.size_bytes > 0

    with session_factory() as s:
        rows = (
            s.query(ModelRegistry)
            .filter_by(model_id="anomaly-det", version="0.2.0")
            .all()
        )
        assert len(rows) == 1
        assert rows[0].sha256 == manifest.sha256
        assert rows[0].manifest["model_id"] == "anomaly-det"
        assert rows[0].manifest["version"] == "0.2.0"


def test_register_manifest_has_metrics(factory):
    """注册产出的 manifest 携带评测指标。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    result = factory.run_all(job)
    manifest: ModelManifest = result.context["manifest"]
    assert "eval_acc" in manifest.metrics_at_register
    assert "p95_latency_ms" in manifest.metrics_at_register


# --------------------------------------------------------------------------- #
# 6. 异步任务提交 + 状态查询
# --------------------------------------------------------------------------- #

def test_submit_job_and_get_status(factory):
    """异步任务提交 + 状态查询 → 最终 completed。"""
    job_id = factory.submit_job(
        "anomaly-det", "0.2.0", {"distill_temperature": 2.0}
    )
    assert isinstance(job_id, str)
    # 轮询直到完成或失败
    job = factory.get_job_status(job_id)
    for _ in range(200):
        job = factory.get_job_status(job_id)
        if job.status in (JobStatus.completed, JobStatus.failed):
            break
        time.sleep(0.05)
    assert job.status is JobStatus.completed
    assert len(job.artifacts) == 7
    assert job.error is None


def test_get_job_status_unknown_raises(factory):
    """查询不存在的 job_id 抛 KeyError。"""
    with pytest.raises(KeyError):
        factory.get_job_status("nonexistent-id")


def test_submit_job_initial_status_pending(factory):
    """提交后立即查询，状态应为 pending 或 running（线程可能已开始）。"""
    job_id = factory.submit_job("anomaly-det", "0.2.0")
    job = factory.get_job_status(job_id)
    assert job.status in (JobStatus.pending, JobStatus.running, JobStatus.stage_done,
                          JobStatus.completed)


# --------------------------------------------------------------------------- #
# 7. 失败处理：某阶段失败 → status=failed
# --------------------------------------------------------------------------- #

def test_stage_failure_sets_failed_status(factory):
    """某阶段失败 → status=failed，error 记录原因。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    # 跳过 data_reflux，直接执行 clean_label → 前置依赖未满足
    with pytest.raises(ValueError):
        factory.run_stage(job, PipelineStage.clean_label)
    assert job.status is JobStatus.failed
    assert job.error is not None
    assert "data_reflux" in job.error


def test_run_all_stops_on_failure(factory):
    """流水线中途失败则停止，status=failed。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    # 篡改 stages，让 distill 前缺少 clean_label → distill 失败
    job.stages = [PipelineStage.data_reflux, PipelineStage.distill]
    with pytest.raises(ValueError):
        factory.run_all(job)
    assert job.status is JobStatus.failed
    # data_reflux 成功执行，distill 失败
    assert "data_reflux" in job.artifacts
    assert "distill" not in job.artifacts


def test_async_job_failure_status(factory):
    """异步任务失败时 status=failed。"""
    job_id = factory.submit_job(
        "anomaly-det",
        "0.2.0",
        {"stages": ["data_reflux", "distill"]},  # 缺 clean_label
    )
    job = factory.get_job_status(job_id)
    for _ in range(200):
        job = factory.get_job_status(job_id)
        if job.status in (JobStatus.completed, JobStatus.failed):
            break
        time.sleep(0.05)
    assert job.status is JobStatus.failed
    assert job.error is not None


# --------------------------------------------------------------------------- #
# 8. 边界：空输入、无效 model_id
# --------------------------------------------------------------------------- #

def test_empty_model_id_raises(factory):
    """空 model_id 抛 ValueError。"""
    with pytest.raises(ValueError):
        factory.create_job("", "0.2.0")


def test_empty_version_raises(factory):
    """空 target_version 抛 ValueError。"""
    with pytest.raises(ValueError):
        factory.create_job("anomaly-det", "")


def test_invalid_distill_temperature_raises(factory):
    """无效蒸馏温度（非正数）抛 ValueError（不可拍脑袋）。"""
    with pytest.raises(ValueError):
        factory.create_job("anomaly-det", "0.2.0", {"distill_temperature": -1})


def test_zero_distill_temperature_raises(factory):
    """蒸馏温度为 0 抛 ValueError。"""
    with pytest.raises(ValueError):
        factory.create_job("anomaly-det", "0.2.0", {"distill_temperature": 0})


def test_clean_label_handles_empty_input(factory):
    """边界：clean_label 接收空 raw_data 不崩溃。"""
    job = factory.create_job("anomaly-det", "0.2.0")
    cleaned = factory._clean_label(job, {})
    assert cleaned["count"] == 0
    assert cleaned["cleaned_samples"] == []
    assert cleaned["dedup_removed"] == 0


def test_custom_stages_subset(factory):
    """边界：自定义阶段子集可执行。"""
    job = factory.create_job(
        "anomaly-det",
        "0.2.0",
        {"stages": ["data_reflux", "clean_label"]},
    )
    assert len(job.stages) == 2
    result = factory.run_all(job)
    assert result.status is JobStatus.completed
    assert set(result.artifacts.keys()) == {"data_reflux", "clean_label"}


def test_invalid_stage_string_raises(factory):
    """边界：无效阶段字符串抛 ValueError。"""
    with pytest.raises(ValueError):
        factory.create_job(
            "anomaly-det", "0.2.0", {"stages": ["data_reflux", "nonexistent"]}
        )


def test_empty_stages_list_raises(factory):
    """边界：空阶段列表抛 ValueError。"""
    with pytest.raises(ValueError):
        factory.create_job("anomaly-det", "0.2.0", {"stages": []})