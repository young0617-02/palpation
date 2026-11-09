"""EdgeLLM 单元测试。

当前环境无法安装 llama-cpp-python（无 C++ 编译器、GitHub/HuggingFace 网络隔离），
故用 mock 替换 llama_cpp.Llama，覆盖 EdgeLLM 的：
- 降级行为（available=False / infer 返回 None）
- 错误处理（模型加载失败、推理失败、JSON 解析失败）
- 性能指标（p95_ms / parse_rate）
- 缓存（相同 prompt TTL 内复用、过期重算、负缓存）
- 线程安全（并发 infer）

真实模型集成测试见 tests/test_edge_llm_real.py（mock 模式无法覆盖真实推理）。
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from edge.agent.llm_llamacpp import EdgeLLM

# EdgeLLM.__init__ 内部 `from llama_cpp import Llama` 为局部导入，
# 各测试用 patch.dict(sys.modules, {"llama_cpp": ...}) 注入行为可控的 mock；
# 降级用例（path 为空/不存在）不触发该导入，无需全局 stub。


# ---------- 辅具 ----------


def _fake_module_with_output(text: str) -> tuple[MagicMock, MagicMock]:
    """构造 mock llama_cpp 模块，其 chat completion 固定返回 text。

    EdgeLLM.infer 走 create_chat_completion，返回结构为
    {"choices": [{"message": {"content": text}}]}。
    """
    instance = MagicMock()
    instance.create_chat_completion.return_value = {"choices": [{"message": {"content": text}}]}
    fake_mod = MagicMock()
    fake_mod.Llama = MagicMock(return_value=instance)
    return fake_mod, instance


def _fake_module_with_instance(instance: MagicMock) -> MagicMock:
    """构造 mock llama_cpp 模块，绑定给定 instance。"""
    fake_mod = MagicMock()
    fake_mod.Llama = MagicMock(return_value=instance)
    return fake_mod


def _dummy_model(tmp_path) -> Path:
    """创建一个占位模型文件（使 path.exists() 为真）。"""
    p = tmp_path / "model.gguf"
    p.write_bytes(b"dummy")
    return p


# ---------- 降级行为 ----------


def test_empty_model_path_unavailable():
    llm = EdgeLLM("")
    assert llm.available is False
    assert llm.infer("anything") is None


def test_nonexistent_model_path_unavailable(tmp_path):
    llm = EdgeLLM(str(tmp_path / "nope.gguf"))
    assert llm.available is False
    assert llm.infer("x") is None


def test_unavailable_default_metrics():
    llm = EdgeLLM("")
    assert llm.p95_ms() == 0.0
    assert llm.parse_rate() == 1.0


def test_unavailable_infer_does_not_touch_metrics():
    llm = EdgeLLM("")
    assert llm.infer("x") is None
    assert llm.json_total == 0
    assert llm.json_ok == 0
    assert llm.latencies_ms == []


# ---------- 模型加载失败 ----------


def test_model_load_exception_marks_unavailable(tmp_path):
    bad = _dummy_model(tmp_path)
    fake_mod = MagicMock()
    fake_mod.Llama = MagicMock(side_effect=RuntimeError("bad model"))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(bad))
    assert llm.available is False
    assert llm.infer("x") is None


def test_model_load_missing_attr_marks_unavailable(tmp_path):
    """llama_cpp 模块无 Llama 属性时优雅降级。"""
    bad = _dummy_model(tmp_path)
    fake_mod = MagicMock(spec=[])  # 无任何属性
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(bad))
    assert llm.available is False


# ---------- 正常推理 + JSON 解析 ----------


def test_infer_parses_json(tmp_path):
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "轴承早期磨损", "confidence": 0.8, "tool": None}
    fake_mod, _ = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.available is True

    result = llm.infer("驱动端振动 RMS 6.2 阈值 4.5")
    assert result == payload
    assert llm.json_total == 1
    assert llm.json_ok == 1
    assert llm.parse_rate() == 1.0
    assert len(llm.latencies_ms) == 1


def test_infer_extracts_json_from_surrounding_noise(tmp_path):
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "磨损", "confidence": 0.9}
    fake_mod, _ = _fake_module_with_output(f"分析结果如下：{json.dumps(payload)} 以上。")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") == payload


def test_infer_non_json_returns_none(tmp_path):
    model = _dummy_model(tmp_path)
    fake_mod, _ = _fake_module_with_output("这不是 JSON")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") is None
    assert llm.json_total == 1
    assert llm.json_ok == 0
    assert llm.parse_rate() == 0.0


def test_infer_malformed_json_returns_none(tmp_path):
    model = _dummy_model(tmp_path)
    fake_mod, _ = _fake_module_with_output('{"diagnosis": "x", "confidence":}')  # 坏 json
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") is None
    assert llm.json_ok == 0


def test_infer_single_quoted_json_recovered(tmp_path):
    """单引号 JSON 经轻量容错（单引号改双引号）后可解析。"""
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "轴承润滑不良", "confidence": 0.85}
    fake_mod, _ = _fake_module_with_output("{'diagnosis': '轴承润滑不良', 'confidence': 0.85}")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") == payload
    assert llm.json_ok == 1


def test_infer_trailing_comma_recovered(tmp_path):
    """尾随逗号经轻量容错（去尾逗号）后可解析。"""
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "轴承润滑不良", "confidence": 0.85, "tool": None}
    fake_mod, _ = _fake_module_with_output(
        '{"diagnosis": "轴承润滑不良", "confidence": 0.85, "tool": null,}'
    )
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") == payload
    assert llm.json_ok == 1


def test_infer_uses_chat_completion_message_content(tmp_path):
    """infer 从 chat completion 的 message.content 取文本。"""
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "轴承磨损", "confidence": 0.9, "tool": None}
    fake_mod, instance = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") == payload
    # 必须走 chat completion（而非原始 completion 接口）
    instance.create_chat_completion.assert_called_once()
    kwargs = instance.create_chat_completion.call_args.kwargs
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"


# ---------- 推理异常 ----------


def test_infer_exception_returns_none_and_records_latency(tmp_path):
    model = _dummy_model(tmp_path)
    instance = MagicMock()
    instance.create_chat_completion.side_effect = RuntimeError("infer boom")
    fake_mod = _fake_module_with_instance(instance)
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") is None
    assert llm.json_total == 1
    assert llm.json_ok == 0
    assert len(llm.latencies_ms) == 1  # 延迟仍记录


def test_infer_partial_output_no_brace_returns_none(tmp_path):
    model = _dummy_model(tmp_path)
    fake_mod, _ = _fake_module_with_output("no braces here at all")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("x") is None


# ---------- 性能指标 ----------


def test_p95_ms_calculation():
    llm = EdgeLLM("")
    llm.latencies_ms = [10.0, 20.0, 30.0, 40.0, 100.0]
    # sorted=[10,20,30,40,100], len=5, idx=int(0.95*4)=3 -> 40.0
    assert llm.p95_ms() == 40.0


def test_p95_ms_single_sample():
    llm = EdgeLLM("")
    llm.latencies_ms = [42.0]
    assert llm.p95_ms() == 42.0


def test_p95_ms_window_capped():
    llm = EdgeLLM("")
    llm.latencies_ms = [float(i) for i in range(300)]
    # 仅取最近 200 个：[100..299]，idx=int(0.95*199)=189 -> 100+189=289.0
    assert llm.p95_ms() == 289.0


def test_parse_rate_calculation():
    llm = EdgeLLM("")
    llm.json_total = 10
    llm.json_ok = 7
    assert llm.parse_rate() == pytest.approx(0.7)


def test_parse_rate_zero_total_is_one():
    llm = EdgeLLM("")
    assert llm.parse_rate() == 1.0


# ---------- 缓存 ----------


def test_cache_reuses_result_within_ttl(tmp_path):
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "x", "confidence": 0.7}
    fake_mod, instance = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    r1 = llm.infer("same prompt")
    r2 = llm.infer("same prompt")
    assert r1 == r2 == payload
    assert instance.create_chat_completion.call_count == 1  # 仅推理一次
    assert llm.json_total == 1  # 缓存命中不计入


def test_cache_expiry_recomputes(tmp_path):
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "x", "confidence": 0.7}
    fake_mod, instance = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    llm.infer("p")
    assert instance.create_chat_completion.call_count == 1
    # 手动使缓存条目过期
    ts, res = llm._cache["p"]
    llm._cache["p"] = (ts - llm._CACHE_TTL_S - 1, res)
    llm.infer("p")
    assert instance.create_chat_completion.call_count == 2
    assert llm.json_total == 2


def test_cache_negative_caching_avoids_repeated_failure(tmp_path):
    """推理失败的结果也被缓存，TTL 内不重复调用模型。"""
    model = _dummy_model(tmp_path)
    instance = MagicMock()
    instance.create_chat_completion.side_effect = RuntimeError("always fails")
    fake_mod = _fake_module_with_instance(instance)
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    assert llm.infer("bad") is None
    assert llm.infer("bad") is None
    assert instance.create_chat_completion.call_count == 1  # 负缓存命中


def test_cache_different_prompts_independent(tmp_path):
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "x", "confidence": 0.7}
    fake_mod, instance = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))
    llm.infer("prompt-a")
    llm.infer("prompt-b")
    assert instance.create_chat_completion.call_count == 2


# ---------- 线程安全 ----------


def test_concurrent_infer_thread_safe(tmp_path):
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "x", "confidence": 0.7}
    fake_mod, instance = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))

    errors: list[BaseException] = []

    def worker(wid: int) -> None:
        try:
            for j in range(25):
                llm.infer(f"prompt-{wid}-{j}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # 4 workers * 25 unique prompts = 100 次实际推理
    assert instance.create_chat_completion.call_count == 100
    assert llm.json_total == 100
    assert llm.json_ok == 100


def test_concurrent_infer_same_prompt_uses_cache(tmp_path):
    """并发相同 prompt：仅推理一次，其余命中缓存。"""
    model = _dummy_model(tmp_path)
    payload = {"diagnosis": "x", "confidence": 0.7}
    fake_mod, instance = _fake_module_with_output(json.dumps(payload))
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        llm = EdgeLLM(str(model))

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                llm.infer("shared")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert instance.create_chat_completion.call_count == 1  # 仅一次实际推理


# ---------- 公开接口稳定性 ----------


def test_public_interface_attributes():
    llm = EdgeLLM("")
    assert hasattr(llm, "available")
    assert hasattr(llm, "infer")
    assert hasattr(llm, "p95_ms")
    assert hasattr(llm, "parse_rate")
    assert llm.available is False
