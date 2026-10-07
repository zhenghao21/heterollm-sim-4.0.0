"""Explicit head dimensions must survive ordinary attention lowering."""
from dataclasses import replace

import pytest

from heterollm_sim import control_plane_planner as control, planner
from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.model_presets import get_model_preset, list_model_presets
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


AFFECTED = ("qwen3-0_6b", "qwen3-4b", "qwen3-32b", "qwen3-30b-a3b", "qwen3-235b-a22b")


def _scenario(preset_id):
    # One dense layer isolates attention while retaining each real preset's
    # hidden/head geometry; no expert execution or long simulation is needed.
    pattern = get_model_preset(preset_id).patterns[0]
    h = pattern.hidden_size
    head_dim = pattern.attention_head_dim or h // pattern.attention_heads
    q, kv = pattern.attention_heads * head_dim, pattern.kv_heads * head_dim
    layer = LayerSpec(
        layer_id="layer0", kind="dense", hidden_size=h, intermediate_size=128,
        attention_heads=pattern.attention_heads, kv_heads=pattern.kv_heads,
        attention_head_dim=pattern.attention_head_dim, dtype="fp16",
        weight_bytes=2 * (2 * h * q + 2 * h * kv + 3 * h * 128),
    )
    base = build_llama_default_scenario()
    graph = build_model_graph_from_layer_specs(
        "geometry", (layer,), architecture="llama", vocabulary_size=32,
        max_sequence_length=4096, embedding_weight_bytes=64 * h,
        tie_word_embeddings=True,
    )
    return prepare_llama_scenario(replace(
        base, model=ModelSpec(name="geometry", graph=graph),
        placement=replace(base.placement, model_name="geometry",
                          parallel=replace(base.placement.parallel, layer_to_stage={})),
        llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1),
    ))


def test_catalog_nonstandard_query_widths_are_explicit():
    affected = set()
    for item in list_model_presets():
        for pattern in get_model_preset(item["id"]).patterns:
            if (pattern.sequence_mixer == "full_attention" and pattern.attention_head_dim
                    and pattern.hidden_size != pattern.attention_heads * pattern.attention_head_dim):
                affected.add(item["id"])
    assert affected == set(AFFECTED)


@pytest.mark.parametrize("preset_id", AFFECTED + ("qwen3-8b", "qwen2_5-0_5b"))
@pytest.mark.parametrize("tokens", (1, 4))
def test_qkv_qk_pv_and_output_use_declared_head_geometry(preset_id, tokens):
    scenario = _scenario(preset_id)
    layer = planner._execution_layers(scenario)[0]
    q = layer.attention_heads * layer.effective_attention_head_dim
    kv = layer.effective_kv_heads * layer.effective_attention_head_dim
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_layer_body(
            builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), layer,
            token_batch=tokens, context_tokens=tokens, kv_read_tokens=0,
            kv_append_tokens=tokens, kv_materialized_tokens=tokens,
            linear_state_runtime=None, phase="prefill", dependencies=(),
        )
    matrices = {task.name.split(".")[-2]: task for task in builder.tasks
                if task.metadata.get("phase") == "gpu_gemm"}
    expected = {"qkv": (layer.hidden_size, q + 2 * kv),
                "attention_qk": (q, tokens), "attention_pv": (tokens, q),
                "attention_output": (q, layer.hidden_size)}
    for name, (k, n) in expected.items():
        task = matrices[name]
        assert (task.metadata["gemm_k"], task.metadata["gemm_n"]) == (k, n)
    # Capacity requirements and actual physical weight reads describe the same
    # Q/K/V/O matrices, including the larger query/output projections.
    actual_attention_bytes = sum(
        matrices[name].metadata["cost_model"]["weight_bytes"]
        for name in ("qkv", "attention_output"))
    assert actual_attention_bytes == 2 * (2 * layer.hidden_size * q + 2 * layer.hidden_size * kv)


@pytest.mark.parametrize("preset_id", ("qwen3-0_6b", "qwen3-8b"))
def test_control_plane_flash_working_set_matches_runtime_geometry(preset_id, monkeypatch):
    scenario = _scenario(preset_id)
    layer = planner._execution_layers(scenario)[0]
    captured = []
    constructor = control.FusedAttentionWorkload

    def capture(**kwargs):
        captured.append(kwargs)
        return constructor(**kwargs)

    monkeypatch.setattr(control, "FusedAttentionWorkload", capture)
    control._derive_fusion_opportunities(
        scenario, control.PlacementPolicy(design_prefill_tokens=4),
        control._derive_requirements(scenario),
    )
    assert len(captured) == 1
    assert captured[0]["hidden_size"] == layer.attention_heads * layer.effective_attention_head_dim
    assert captured[0]["score_heads"] == layer.attention_heads
