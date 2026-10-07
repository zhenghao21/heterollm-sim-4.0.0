import pytest

from heterollm_sim.model_presets import materialize_model_payload
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.web import (
    HttpError,
    _scenario_failure_diagnostic,
    scenario_or_http_error,
    scenario_to_payload,
    validation_payload,
)


def test_device_memory_capacity_rejection_is_not_reported_as_malformed_json():
    payload = scenario_to_payload(build_llama_default_scenario())
    payload["model"] = materialize_model_payload("llama3_1-405b")
    payload["profiles"]["llama_cpp"] = {
        "policy": "llama_cpp", "gpu_layers": -1, "batch": 512,
        "ubatch": 512, "context": 640, "parallel": 1,
        "offload_kqv": True, "device_memory_tiering": True,
    }
    for key in ("op_to_component", "tensor_to_component", "tensor_bytes"):
        payload["placement"][key] = {}
    payload["placement"]["parallel"].update(layer_to_stage={}, rank_mapping=[])
    payload["placement"]["metadata"].pop("control_plane", None)

    validation = validation_payload(payload)
    assert not validation["valid"]
    issue = validation["errors"]["scenario"][0]
    assert issue["code"] == "device_memory_capacity_exhausted"
    assert "显存容量不足" in issue["message_zh"]
    assert "gpu0" in issue["message_zh"]
    assert "mlp_weights" in issue["message_zh"]
    assert "字节" in issue["message_zh"]
    assert "JSON" not in issue["message_zh"]

    with pytest.raises(HttpError) as caught:
        scenario_or_http_error(payload)
    error = caught.value
    assert error.status == 422
    assert error.code == issue["code"]
    assert error.message == issue["message_zh"]
    assert error.details["errors"]["scenario"][0] == issue


def test_unrelated_value_errors_keep_the_generic_chinese_diagnostic():
    code, message_zh, _ = _scenario_failure_diagnostic(ValueError("unrelated internal detail"))
    assert code == "parse_error"
    assert message_zh == "场景 JSON 解析失败：字段值或结构不符合配置约束"
