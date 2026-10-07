"""F16/F32 decode dispatch from llama.cpp d3146f2, not latency fitting."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import GemmWorkload, estimate_gpu_gemm
from heterollm_sim.kernel_model import llama_blackwell_analytical_profile
from heterollm_sim.mmvf_work import derive_mmvf_work
import heterollm_sim.planner as planner


@lru_cache(maxsize=1)
def _case():
    path = Path(__file__).resolve().parents[1] / "docs/frontend_native_validation_2026-10-07/scenario_qwen3_0_6b_f16_512_128.json"
    scenario = scenario_from_dict(json.loads(path.read_text(encoding="utf-8")))
    profiles = {kind: dict(items) for kind, items in scenario.component_profiles.items()}
    for name, gpu in profiles["gpu"].items():
        profiles["gpu"][name] = replace(gpu, kernel_model=llama_blackwell_analytical_profile(
            "nvidia-rtx-5080", "source-dispatch-test"))
    return replace(scenario, component_profiles=profiles)


def _qkv(scenario):
    layer = planner._execution_layers(scenario)[0]
    tensors = [item for item in layer.metadata["gguf_tensor_bindings"]
               if item["name"].endswith((".attn_q.weight", ".attn_k.weight", ".attn_v.weight"))]
    workload = GemmWorkload(1, 1024, 4096, activation_bits=16, weight_bits=16,
                            output_bits=32, activation_storage_bytes=4096)
    return workload, {"layer_id": layer.layer_id, "projection_id": "attention.qkv"}, tensors


def test_source_qkv_calls_conserve_weights_and_charge_each_launch():
    scenario = _case()
    workload, metadata, tensors = _qkv(scenario)
    plan = planner._parallel_plan(scenario)
    builder = planner._TaskBuilder(scenario.workload.requests[0])
    planner._add_rank_gemm(builder, scenario, planner._topology_router(scenario), plan,
        plan.ranks[0], workload, "gpu0", "decode.layer_0.qkv", (),
        keep_output_on_target=True, metadata=metadata)
    matrices = [task for task in builder.tasks if task.metadata.get("phase") == "gpu_gemm"]
    launches = [task for task in builder.tasks if task.metadata.get("phase") == "kernel_launch"]
    assert len(matrices) == len(launches) == 3
    assert [task.metadata["gemm_n"] for task in matrices] == [2048, 1024, 1024]
    assert sum(task.metadata["cost_model"]["weight_bytes"] for task in matrices) == workload.weight_bytes
    assert sum(task.metadata["cost_model"]["output_bytes"] for task in matrices) == workload.output_bytes
    for i, task in enumerate(matrices):
        audit = task.metadata["mmvf_source_work"]
        assert audit["cta_count"] == task.metadata["gemm_n"]
        assert audit["block_threads"] == 256
        assert audit["input_storage_dtype"] == audit["output_storage_dtype"] == "f32"
        assert task.metadata["mmvf_weight_tensors"][0]["name"].endswith(
            (".attn_q.weight", ".attn_k.weight", ".attn_v.weight")[i])
        assert not any("tensor" in demand.resource_id for demand in task.demands)
        if i:
            assert matrices[i-1].task_id in launches[i].dependencies


def test_source_f16_prefill_keeps_three_real_calls_without_mmvf():
    scenario = _case()
    workload, metadata, _ = _qkv(scenario)
    workload = replace(workload, m=16, activation_storage_bytes=16 * 1024 * 4)
    plan = planner._parallel_plan(scenario)
    builder = planner._TaskBuilder(scenario.workload.requests[0])
    planner._add_rank_gemm(builder, scenario, planner._topology_router(scenario), plan,
        plan.ranks[0], workload, "gpu0", "prefill.layer_0.qkv", (),
        keep_output_on_target=True, metadata=metadata)
    matrices = [task for task in builder.tasks if task.metadata.get("phase") == "gpu_gemm"]
    assert len(matrices) == 3
    assert sum(task.metadata["cost_model"]["weight_bytes"] for task in matrices) == workload.weight_bytes
    assert sum(task.metadata["cost_model"]["output_bytes"] for task in matrices) == workload.output_bytes
    assert all("mmvf_source_work" not in task.metadata for task in matrices)
    assert all(any("tensor" in demand.resource_id for demand in task.demands) for task in matrices)


@pytest.mark.parametrize("changes", [
    {"m": 2}, {"activation_bits": 32}, {"activation_storage_bytes": 2048},
    {"output_bits": 16}, {"weight_bits": 8}, {"layout": "strided"},
    {"packed_weight_formats": ("Q8_0",)}, {"epilogue_name": "rope"},
])
def test_unqualified_work_does_not_acquire_mmvf(changes):
    scenario = _case()
    workload, metadata, _ = _qkv(scenario)
    assert planner._source_f16_projection_tensors(
        scenario, replace(workload, **changes), "gpu0", "decode.layer_0.qkv", metadata) == ()


def test_missing_source_contract_or_tp_or_non_gpu_does_not_acquire_mmvf():
    scenario = _case()
    workload, metadata, _ = _qkv(scenario)
    unsigned = replace(scenario, workload=replace(scenario.workload,
        metadata={key: value for key, value in scenario.workload.metadata.items()
                  if key != "llama_cpp_tensor_storage_contract"}))
    assert planner._source_f16_projection_tensors(unsigned, workload, "gpu0", "decode.layer_0.qkv", metadata) == ()
    assert planner._source_f16_projection_tensors(scenario, workload, "gpu0", "decode.layer_0.qkv",
        {**metadata, "projection_tp_degree": 2}) == ()
    assert planner._source_f16_projection_tensors(scenario, workload, "cpu0", "decode.layer_0.qkv", metadata) == ()
    assert planner._source_f16_projection_tensors(scenario, workload, "gpu0", "decode.layer_0.qkv",
        {**metadata, "rhs_operand_kind": "activation"}) == ()
    with pytest.raises(ValueError, match="physical dimensions"):
        planner._source_f16_projection_tensors(scenario, replace(workload, n=3072),
            "gpu0", "decode.layer_0.qkv", metadata)


def test_source_forbids_rope_fusion_and_prefill_glu_fusion():
    scenario = _case()
    workload, _metadata, _ = _qkv(scenario)
    rank = planner._parallel_plan(scenario).ranks[0]
    assert not planner._gemm_epilogue_declared(scenario, rank, "gpu0", workload, "rope", "decode")
    assert not planner._gemm_epilogue_declared(scenario, rank, "gpu0", replace(workload, m=16), "swiglu", "prefill")


def test_fused_gate_has_one_grid_and_charges_both_weights():
    scenario = _case()
    layer = planner._execution_layers(scenario)[0]
    workload = GemmWorkload(1, 1024, 6144, activation_bits=16, weight_bits=16,
        output_bits=32, activation_storage_bytes=4096, epilogue_name="swiglu",
        epilogue_output_elements=3072, epilogue_operations=3072 * 3,
        epilogue_transcendental_operations=3072)
    tensors = planner._source_f16_projection_tensors(scenario, workload, "gpu0",
        "decode.layer_0.mlp", {"layer_id": layer.layer_id, "projection_id": "mlp.up_gate"})
    assert len(tensors) == 2
    source = derive_mmvf_work(1024, 6144, fused_gate=True)
    gpu, memory = planner._gpu_profiles(scenario, "gpu0", "gddr0")
    estimate = estimate_gpu_gemm(gpu, memory, replace(workload,
        execution_phase="decode", mmvf_work=source))
    assert source.cta_count == 3072
    assert source.weight_bytes == sum(tensor["n_bytes"] for tensor in tensors)
    assert source.output_bytes == 3072 * 4
    assert source.shared_bytes == 256
    assert len(estimate.phases) == 2


def test_unqualified_dense_dispatch_and_cost_unchanged():
    scenario = _case()
    gpu, memory = planner._gpu_profiles(scenario, "gpu0", "gddr0")
    profile = gpu.kernel_model
    ordinary = replace(profile, kernels=tuple(kernel for kernel in profile.kernels
                                              if "mmvf" not in kernel.kernel_family))
    for m in (1, 16):
        for output in (16, 32):
            workload = GemmWorkload(m, 1024, 2048, activation_bits=16,
                weight_bits=16, output_bits=output, execution_phase="decode")
            assert estimate_gpu_gemm(gpu, memory, workload) == estimate_gpu_gemm(
                replace(gpu, kernel_model=ordinary), memory, workload)


def test_mmvf_work_requires_matching_storage_and_source_shape():
    source = derive_mmvf_work(1024, 2048)
    workload = GemmWorkload(1, 1024, 2048, activation_bits=16, weight_bits=16,
        output_bits=32, activation_storage_bytes=4096, mmvf_work=source)
    for changes in ({"m": 2}, {"n": 1024}, {"weight_bits": 8},
                    {"activation_storage_bytes": 2048}, {"output_bits": 16},
                    {"packed_weight_formats": ("Q8_0",)}):
        with pytest.raises(ValueError, match="mmvf_work"):
            replace(workload, **changes)
    with pytest.raises(ValueError, match="even contiguous K"):
        derive_mmvf_work(1023, 2048)


def test_mmvf_source_work_cannot_fall_back_to_tensor_cost_without_kernel_profile():
    scenario = _case()
    gpu, memory = planner._gpu_profiles(scenario, "gpu0", "gddr0")
    workload = GemmWorkload(1, 1024, 2048, activation_bits=16, weight_bits=16,
        output_bits=32, activation_storage_bytes=4096, mmvf_work=derive_mmvf_work(1024, 2048))
    with pytest.raises(ValueError, match="MMVF source work requires a GPU kernel model"):
        estimate_gpu_gemm(replace(gpu, kernel_model=None), memory, workload)


@pytest.mark.parametrize("k,threads", [(64, 32), (128, 64), (192, 96), (1024, 256)])
def test_block_size_is_source_loop_minimizer(k, threads):
    assert derive_mmvf_work(k, 17).block_threads == threads
