"""Source-required Q/K/V bias must contribute work without inventing a full bias batch."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim import control_plane_planner as control, planner
from heterollm_sim.attention_bias import resolve_attention_qkv_bias, projection_bias_workload
from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import ModelSpec, build_model_graph_from_layer_specs, model_graph_execution_view
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.model_presets import get_model_preset, list_model_presets, materialize_model_payload, _layer_weight_bytes
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.reporting import run_scenario, report_dict
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


QWEN25 = tuple(row["id"] for row in list_model_presets() if row["id"].startswith("qwen2_5-"))


def _scenario(*, f32=False, bias=True, cpu_bias=False):
    model = model_from_dict(materialize_model_payload("qwen2_5-0_5b"))
    full = model_graph_execution_view(model.graph).layer_instances[0].layer
    metadata = dict(full.metadata)
    if not bias:
        del metadata["attention_qkv_bias"]
    h = full.hidden_size
    q, kv = full.attention_heads * full.effective_attention_head_dim, full.effective_kv_heads * full.effective_attention_head_dim
    bias_bytes = (q + 2 * kv) * 2 if bias else 0
    layer = replace(full, layer_id="layer0", intermediate_size=128,
        weight_bytes=2 * (2 * h * q + 2 * h * kv + 3 * h * 128) + bias_bytes,
        metadata=metadata)
    base = build_llama_default_scenario()
    graph = build_model_graph_from_layer_specs("bias", (layer,), architecture="qwen2",
        vocabulary_size=32, max_sequence_length=4096, embedding_weight_bytes=64 * h,
        tie_word_embeddings=True)
    scenario = prepare_llama_scenario(replace(base, model=ModelSpec(name="bias", graph=graph),
        placement=replace(base.placement, model_name="bias",
            parallel=replace(base.placement.parallel, layer_to_stage={})),
        workload=replace(base.workload, metadata={**base.workload.metadata, "llama_cpp_f32_hidden_storage": f32}),
        llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1)))
    if cpu_bias:
        cpu = next(c.component_id for c in scenario.hardware.components if c.kind == "cpu")
        scenario = replace(scenario, placement=replace(scenario.placement,
            op_to_component={**scenario.placement.op_to_component,
                **{"layer0.attention.{}_bias".format(key): cpu for key in ("q", "k", "v")}}))
    return scenario


def _compile(scenario, tokens):
    layer = planner._execution_layers(scenario)[0]
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_layer_body(builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), layer, token_batch=tokens, context_tokens=tokens,
            kv_read_tokens=0, kv_append_tokens=tokens, kv_materialized_tokens=tokens,
            linear_state_runtime=None, phase="prefill" if tokens > 1 else "decode", dependencies=())
    return builder


@pytest.mark.parametrize("preset_id", QWEN25)
def test_every_qwen25_size_declares_correct_vectors_and_capacity(preset_id):
    model = model_from_dict(materialize_model_payload(preset_id))
    pattern = get_model_preset(preset_id).patterns[0]
    for instance in model_graph_execution_view(model.graph).layer_instances:
        layer = instance.layer
        contract = resolve_attention_qkv_bias(layer.metadata,
            query_width=layer.attention_heads * layer.effective_attention_head_dim,
            kv_width=layer.effective_kv_heads * layer.effective_attention_head_dim)
        assert contract.storage_bits == 16
        assert layer.weight_bytes == _layer_weight_bytes(pattern) + contract.weight_bytes


@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("f32", [False, True])
def test_bias_work_is_physical_and_precedes_rope(tokens, f32):
    scenario = _scenario(f32=f32)
    builder = _compile(scenario, tokens)
    layer = planner._execution_layers(scenario)[0]
    total_ops = 0
    for label, width in (("q", 896), ("k", 128), ("v", 128)):
        tasks = [task for task in builder.tasks if task.metadata.get("event_kind") == f"attention_{label}_bias"]
        assert tasks
        # One runtime operation can expose several phases, including launch.
        costed = next(task for task in tasks if any(d.work_units > 0 for d in task.demands))
        metadata = costed.metadata
        assert metadata["bias_add_operations"] == tokens * width
        assert metadata["bias_activation_read_bytes"] == tokens * width * (4 if f32 else 2)
        assert metadata["bias_vector_read_bytes"] == width * 2
        assert metadata["bias_output_write_bytes"] == metadata["bias_activation_read_bytes"]
        assert any(d.bytes_moved > 0 for task in tasks for d in task.demands)
        assert any(d.service_ns > 0 for task in tasks for d in task.demands)
        assert costed.metadata["cost_model"]["operations"] == tokens * width
        assert costed.metadata["cost_model"]["read_bytes"] == metadata["bias_activation_read_bytes"] + width * 2
        assert costed.metadata["cost_model"]["write_bytes"] == metadata["bias_output_write_bytes"]
        total_ops += metadata["bias_add_operations"]
    assert total_ops == tokens * (896 + 128 + 128)
    ready = next(t for t in builder.tasks if t.metadata.get("event_kind") == "attention_qkv_bias_ready")
    assert len(ready.dependencies) == 3
    assert any(ready.task_id in t.dependencies for t in builder.tasks)
    qkv = [t for t in builder.tasks if t.metadata.get("projection_id") == "attention.qkv"]
    assert qkv and all(t.metadata.get("fusion_applied") is not True for t in qkv)
    requirements = control._derive_requirements(scenario)
    for label, width in (("q", 896), ("k", 128), ("v", 128)):
        requirement = next(r for r in requirements if r.kind == f"{label}_bias")
        assert requirement.elements_per_token == width
        assert requirement.fixed_read_bytes == width * 2
    opportunities = control._derive_fusion_opportunities(scenario, control.PlacementPolicy(), requirements)
    assert not any(row.group == "qkv_rope" for row in opportunities)
    assert sum(r.tensor_bytes for r in requirements if r.layer == layer and r.kind in {"attention", "mlp"}) == layer.weight_bytes


def test_explicit_cpu_bias_mapping_preserves_result_return_dependency():
    scenario = _scenario(cpu_bias=True)
    builder = _compile(scenario, 4)
    for label in ("q", "k", "v"):
        tasks = [t for t in builder.tasks if t.metadata.get("event_kind") == f"attention_{label}_bias"]
        assert tasks and all(t.metadata["target_component"] == "cpu0" for t in tasks)
    returns = [t for t in builder.tasks if t.metadata.get("event_kind") == "attention_bias_output_transfer"]
    assert returns and any(d.bytes_moved > 0 for t in returns for d in t.demands)


def test_bias_mapping_survives_full_validation_and_execution():
    scenario = _scenario()
    request = replace(scenario.workload.requests[0], prompt_tokens=4, output_tokens=3)
    scenario = replace(scenario, workload=replace(scenario.workload, requests=(request,)))
    keys = {"layer0.attention.{}_bias".format(label) for label in ("q", "k", "v")}
    assert keys <= planner._reference_lowering_op_mapping_keys(scenario)
    uses = {row[1] for row in planner._typed_primitive_mapping_uses(scenario)}
    assert keys <= uses
    result = report_dict(run_scenario(scenario))
    assert result["summary"]["completed_requests"] == 1


def test_models_without_bias_do_not_gain_bias_work():
    for preset_id in ("llama3_1-8b", "qwen3-0_6b"):
        model = model_from_dict(materialize_model_payload(preset_id))
        assert all("attention_qkv_bias" not in i.layer.metadata for i in model_graph_execution_view(model.graph).layer_instances)
    builder = _compile(_scenario(bias=False), 4)
    assert not any("bias" in str(t.metadata.get("event_kind", "")) for t in builder.tasks)


@pytest.mark.parametrize("field,value", [("query_elements", 895), ("key_elements", True),
    ("value_elements", 64), ("storage_bits", 4), ("weight_bytes", 1), ("source", ""),
    ("schema_version", "unknown")])
def test_invalid_bias_contract_is_rejected(field, value):
    scenario = _scenario()
    metadata = deepcopy(planner._execution_layers(scenario)[0].metadata)
    metadata["attention_qkv_bias"][field] = value
    with pytest.raises(ValueError, match="attention_qkv_bias"):
        resolve_attention_qkv_bias(metadata, query_width=896, kv_width=128)


def test_broadcast_vector_traffic_is_constant_when_token_count_grows():
    for bits in (16, 32):
        one = projection_bias_workload(tokens=1, local_width=64, activation_bits=32, bias_bits=bits, name="q_bias")
        many = projection_bias_workload(tokens=512, local_width=64, activation_bits=32, bias_bits=bits, name="q_bias")
        assert one.read_bytes - one.write_bytes == many.read_bytes - many.write_bytes == 64 * bits // 8
        assert many.operations == 512 * one.operations
