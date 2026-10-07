from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from heterollm_sim.config import model_from_dict, scenario_from_dict
from heterollm_sim import control_plane_planner as control, planner
from heterollm_sim.ir import model_graph_execution_view
from heterollm_sim.model_presets import get_model_preset, materialize_model_payload, _layer_weight_bytes
from heterollm_sim.projection_descriptors import resolve_attention_qk_norm


QWEN3 = ("qwen3-0_6b", "qwen3-1_7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b", "qwen3-235b-a22b")


@pytest.mark.parametrize("preset_id", QWEN3)
def test_all_qwen3_presets_declare_norm_and_count_its_weight_capacity(preset_id):
    model = model_from_dict(materialize_model_payload(preset_id))
    pattern = get_model_preset(preset_id).patterns[0]
    for instance in model_graph_execution_view(model.graph).layer_instances:
        layer = instance.layer
        contract = resolve_attention_qk_norm(layer.metadata, head_dim=layer.effective_attention_head_dim)
        assert contract["kind"] == "rmsnorm"
        assert contract["weight_bytes"] == 2 * layer.effective_attention_head_dim * 2
        assert layer.weight_bytes == _layer_weight_bytes(pattern) + contract["weight_bytes"]


def test_qwen25_does_not_acquire_qwen3_norm():
    model = model_from_dict(materialize_model_payload("qwen2_5-0_5b"))
    for instance in model_graph_execution_view(model.graph).layer_instances:
        layer = instance.layer
        assert resolve_attention_qk_norm(layer.metadata, head_dim=layer.effective_attention_head_dim) is None


@pytest.mark.parametrize("tokens", [1, 16])
def test_real_gguf_qk_norm_is_costed_before_rope_and_in_control_plane(tokens):
    path = Path(__file__).resolve().parents[1] / "docs/frontend_native_validation_2026-10-07/scenario_qwen3_0_6b_f16_512_128.json"
    scenario = scenario_from_dict(json.loads(path.read_text(encoding="utf-8")))
    layer = planner._execution_layers(scenario)[0]
    contract = resolve_attention_qk_norm(layer.metadata, head_dim=128)
    assert {name: row["n_bytes"] for name, row in contract["weight_bindings"].items()} == {"q": 512, "k": 512}
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_layer_body(builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), layer, token_batch=tokens, context_tokens=tokens,
            kv_read_tokens=0, kv_append_tokens=tokens, kv_materialized_tokens=tokens,
            linear_state_runtime=None, phase="prefill", dependencies=())
    for prefix, heads in (("q", 16), ("k", 8)):
        tasks = [task for task in builder.tasks if task.metadata.get("event_kind") == f"attention_{prefix}_norm_apply"]
        assert tasks
        assert any(sum(demand.service_ns for demand in task.demands) > 0 for task in tasks)
        assert all(task.metadata["norm_groups"] == tokens * heads for task in tasks)
        assert all(task.metadata["norm_weight_binding"] == contract["weight_bindings"][prefix] for task in tasks)
    join = next(task for task in builder.tasks if task.metadata.get("event_kind") == "attention_qk_norm_ready")
    assert len(join.dependencies) == 2
    assert any(join.task_id in task.dependencies for task in builder.tasks)
    rope = [task for task in builder.tasks if task.metadata.get("event_kind") == "rope"
            and task.metadata.get("phase") != "kernel_launch"]
    assert len(rope) == 2
    assert {task.metadata["rope_operand"] for task in rope} == {"q", "k"}
    assert sum(task.metadata["cost_model"]["special_function_operations"] for task in rope) == tokens * 3072
    assert sum(task.metadata["cost_model"]["read_bytes"] for task in rope) == tokens * (3072 * 4 + 8)
    assert all(task.metadata["rope_table_strategy"] == "runtime_sin_cos" for task in rope)
    assert all(task.metadata["timing_completeness"] == "partial" for task in rope)
    requirements = control._derive_requirements(scenario)
    norm_requirements = [row for row in requirements if row.layer == layer and row.kind in {
        "q_norm_reduce", "q_norm_apply", "k_norm_reduce", "k_norm_apply"}]
    assert len(norm_requirements) == 4
    keys = {row.mapping_key for row in norm_requirements}
    assert keys <= planner._reference_lowering_op_mapping_keys(scenario)
    assert keys <= {row[1] for row in planner._typed_primitive_mapping_uses(scenario)}
    explicit = replace(scenario, placement=replace(scenario.placement,
        op_to_component={**scenario.placement.op_to_component, **dict.fromkeys(keys, "gpu0")}))
    validation = planner.validate_scenario(explicit)
    assert validation.is_valid, validation.errors_en


def test_qk_norm_rejects_wrong_head_shape_and_missing_pair():
    contract = {"kind": "rmsnorm", "head_dim": 128, "weight_bindings": {
        "q": {"shape": [128], "type": "F32", "n_bytes": 512},
        "k": {"shape": [128], "type": "F32", "n_bytes": 512}}}
    resolve_attention_qk_norm({"attention_qk_norm": contract}, head_dim=128)
    invalid = deepcopy(contract)
    invalid["weight_bindings"]["q"]["shape"] = [64]
    with pytest.raises(ValueError, match="physical weight shape"):
        resolve_attention_qk_norm({"attention_qk_norm": invalid}, head_dim=128)
    invalid = deepcopy(contract)
    del invalid["weight_bindings"]["k"]
    with pytest.raises(ValueError, match="both Q and K"):
        resolve_attention_qk_norm({"attention_qk_norm": invalid}, head_dim=128)
