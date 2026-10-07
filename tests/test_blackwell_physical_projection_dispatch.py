"""Regression cases from the two imported GGUF/frontend validation scenarios."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import estimate_gpu_gemm
from heterollm_sim.kernel_model import llama_blackwell_analytical_profile
import heterollm_sim.planner as planner


@lru_cache(maxsize=1)
def _mixed_case():
    source = (Path(__file__).resolve().parents[1] / "docs" /
              "frontend_native_validation_2026-10-07" /
              "scenario_qwen3_8_27b_mixed_512_128.json")
    return scenario_from_dict(json.loads(source.read_text(encoding="utf-8")))


def _projection(scenario, *, mixed, m=16):
    layers = planner._execution_layers(scenario)
    layer = next(layer for layer in layers
                 if (layer.linear_attention is None) == mixed)
    projection = "attention.qkv" if mixed else "linear_attention.qkv"
    workload = planner._layer_gemm(
        layer, m, layer.hidden_size, 14336 if mixed else 10240,
        name="qkv_tp", projection_id=projection, f32_storage=True,
    )
    return layer, workload, {"layer_id": layer.layer_id, "projection_id": projection}


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("epilogue", ["", "rope", "swiglu"])
def test_dense_f32_storage_preserves_f16_compute_declaration(phase, epilogue):
    profile = llama_blackwell_analytical_profile("test-gpu", "test-runtime")
    args = dict(shape=(16, 3072, 1024), formats=("fp16",), dtype="fp16",
                phase=phase, accumulator_bits=32, epilogue=epilogue)
    original = profile.dispatch(**args, output_bits=16)
    f32 = profile.dispatch(**args, output_bits=32)
    assert original is not None and f32 is not None
    assert replace(f32, output_bits=16) == original
    assert f32.internal_dtype == "fp16"


@pytest.mark.parametrize("m", [1, 16])
def test_real_mixed_projection_calls_conserve_physical_work(m):
    scenario = _mixed_case()
    layer, workload, metadata = _projection(scenario, mixed=True, m=m)
    name = ("decode." if m == 1 else "prefill.") + layer.layer_id + ".qkv"
    calls = planner._mixed_projection_calls(scenario, workload, name, metadata)
    assert [call.n for call, _ in calls] == [12288, 1024, 1024]
    assert [call.packed_weight_formats for call, _ in calls] == [
        ("IQ4_XS",), ("IQ4_XS",), ("Q5_K",)]
    assert [call.weight_bytes for call, _ in calls] == [33423360, 2785280, 3604480]
    assert sum(call.weight_bytes for call, _ in calls) == workload.weight_bytes
    assert sum(call.output_bytes for call, _ in calls) == workload.output_bytes
    assert sum(call.packed_weight_transform_operations for call, _ in calls) == workload.packed_weight_transform_operations
    assert all(call.activation_bytes == workload.activation_bytes for call, _ in calls)
    assert len({meta["weight_buffer_id"] for _, meta in calls}) == 3

    plan = planner._parallel_plan(scenario)
    builder = planner._TaskBuilder(scenario.workload.requests[0])
    planner._add_rank_gemm(
        builder, scenario, planner._topology_router(scenario), plan, plan.ranks[0],
        workload, "gpu0", name, (), keep_output_on_target=True,
        weight_tensor_id=layer.layer_id + ".attention_weights", metadata=metadata,
    )
    matrices = [task for task in builder.tasks if task.metadata.get("phase") == "gpu_gemm"]
    launches = [task for task in builder.tasks if task.metadata.get("phase") == "kernel_launch"]
    assert len(matrices) == len(launches) == 3
    assert sum(task.metadata["cost_model"]["weight_bytes"] for task in matrices) == workload.weight_bytes
    assert sum(task.metadata["cost_model"]["output_bytes"] for task in matrices) == workload.output_bytes
    for index, task in enumerate(matrices):
        assert len(task.metadata["projection_segments"]) == 1
        assert task.metadata["projection_segments"][0]["local_physical_bytes"] == calls[index][0].weight_bytes
        assert task.metadata["gemm_n"] == calls[index][0].n
        assert launches[index].task_id in task.dependencies
        if index:
            # Charge every physical call in order, rather than hiding two
            # extra matrices behind the first call's elapsed time.
            assert matrices[index - 1].task_id in launches[index].dependencies


def test_single_format_projection_keeps_existing_cost_path():
    scenario = _mixed_case()
    layer, workload, metadata = _projection(scenario, mixed=False)
    name = "prefill." + layer.layer_id + ".qkv"
    assert planner._mixed_projection_calls(scenario, workload, name, metadata) == ()
    plan = planner._parallel_plan(scenario)
    builder = planner._TaskBuilder(scenario.workload.requests[0])
    planner._add_rank_gemm(
        builder, scenario, planner._topology_router(scenario), plan, plan.ranks[0],
        workload, "gpu0", name, (), keep_output_on_target=True,
        weight_tensor_id=layer.layer_id + ".linear_attention_weights", metadata=metadata,
    )
    gpu, memory = planner._gpu_profiles(scenario, "gpu0", "gddr0")
    expected = estimate_gpu_gemm(gpu, memory, replace(workload, execution_phase="prefill"))
    assert len(builder.tasks) == len(expected.phases) == 2
    for task, phase in zip(builder.tasks, expected.phases):
        assert task.metadata["phase"] == phase.name
        assert "physical_projection_segment_index" not in task.metadata
        assert task.demands == phase.demands


def test_mixed_projection_write_direction_audit_uses_each_physical_output():
    scenario = _mixed_case()
    layer, workload, metadata = _projection(scenario, mixed=True, m=16)
    metadata["modeled_memory_write_bytes"] = workload.output_bytes
    calls = planner._mixed_projection_calls(scenario, workload, "prefill." + layer.layer_id + ".qkv", metadata)
    assert [meta["modeled_memory_write_bytes"] for _, meta in calls] == [786432, 65536, 65536]
    assert sum(meta["modeled_memory_write_bytes"] for _, meta in calls) == workload.output_bytes
    for call, meta in calls:
        assert meta["modeled_memory_write_bytes"] == call.output_bytes


def test_undeclared_quantized_fusion_requires_separate_operator():
    scenario = _mixed_case()
    _layer, workload, _metadata = _projection(scenario, mixed=False)
    rank = planner._parallel_plan(scenario).ranks[0]
    assert not planner._gemm_epilogue_declared(scenario, rank, "gpu0", workload, "swiglu", "prefill")
    assert not planner._gemm_epilogue_declared(scenario, rank, "gpu0", workload, "rope", "prefill")


def test_mixed_projection_rejects_unmapped_output_and_fused_work():
    scenario = _mixed_case()
    layer, workload, metadata = _projection(scenario, mixed=True)
    name = "prefill." + layer.layer_id + ".qkv"
    with pytest.raises(ValueError, match="separate operator lowering"):
        planner._mixed_projection_calls(scenario, replace(workload, epilogue_name="rope"), name, metadata)
    with pytest.raises(ValueError, match="per-column output storage"):
        planner._mixed_projection_calls(scenario, replace(workload, output_storage_bytes=32), name, metadata)
