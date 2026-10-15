"""Kernel callers keep current typed contracts and distinct model semantics."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.calibration import _invocation_fields
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import (
    CacheHierarchyProfile, CacheLevelProfile, FusedAttentionWorkload, GemmWorkload,
    GPUProfile, HBMProfile, TensorCoreProfile, estimate_gpu_gemm,
)
from heterollm_sim.kernel_mapping import classify_kernel, map_kernel_row
from heterollm_sim.kernel_model import (
    KernelCapability, KernelModelProfile, apply_captured_graph_launch,
    estimate_kernel, execution_phase, kernel_calibration_dispatch_signature,
)
from heterollm_sim.kernel_query_ledger import _source_mmvq_executed_k, summarize_kernel_queries
from heterollm_sim.mmvq_work import MMVQSourceContract, SOURCE_SHA256, derive_mmvq_work


def _gpu(profile=None):
    return GPUProfile(TensorCoreProfile(1, 1, 1.),
        CacheHierarchyProfile((CacheLevelProfile("gpu.l2", 1024, 64, 0, 1000.),)),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1, reduction_ops_per_cycle_per_sm=1,
        special_function_units_per_sm=1, special_function_ops_per_cycle=1,
        kernel_model=profile)


@pytest.mark.parametrize("attention", [False, True])
def test_current_workloads_retain_optional_dtype_and_analytical_kernel_models(attention):
    capability = KernelCapability("attention" if attention else "gemm", ("fp16",),
        "fp16", "decode", "tensor", "analytical declaration", internal_dtype="fp16",
        operator="attention" if attention else "gemm")
    profile = KernelModelProfile("gpu", "runtime", "arch", kernels=(capability,))
    workload = (FusedAttentionWorkload(1, 16, 16, score_heads=1, execution_phase="decode")
                if attention else GemmWorkload(1, 16, 16, activation_bits=16,
                                               weight_bits=16, execution_phase="decode"))
    assert workload.activation_dtype is None
    result = estimate_kernel(_gpu(profile), HBMProfile(1000.), workload, attention=attention)
    assert result is not None and result.service_ns > 0
    assert result.metadata["prediction"]["model"] == "analytical"
    assert execution_phase(workload) == "decode"


def test_no_kernel_profile_remains_an_explicit_analytical_mode():
    workload = GemmWorkload(1, 16, 16)
    assert execution_phase(workload) == "unspecified"
    assert kernel_calibration_dispatch_signature(workload) is None
    assert estimate_kernel(_gpu(), HBMProfile(1000.), workload) is None
    estimate = estimate_gpu_gemm(_gpu(), HBMProfile(1000.), workload)
    assert estimate.metadata["prediction"]["phase"] == "unspecified"
    assert estimate.metadata["prediction"]["model"] == "analytical"
    with pytest.raises(AttributeError, match="execution_phase"):
        execution_phase(SimpleNamespace())


def test_current_cuda_capture_caller_preserves_physical_accounting():
    profile = KernelModelProfile("gpu", "runtime", "arch", graph_enabled=True,
        graph_device_launch_ns=7, graph_device_launch_evidence="device replay measurement")
    tasks = tuple(TaskSpec(f"launch-{i}", "request", "launch", TaskCategory.COMPUTE,
        demands=(ResourceDemand("gpu.frontend", 10, bytes_moved=64, work_units=2),),
        metadata={"phase": "kernel_launch", "cuda_graph_captured": True,
                  "cuda_graph_id": "replay-1"}) for i in range(2))
    first, second = apply_captured_graph_launch(tasks, profile)
    assert (first.demands[0].service_ns, second.demands[0].service_ns) == (7, 0)
    assert (first.demands[0].bytes_moved, first.demands[0].work_units) == (128, 4)
    assert (second.demands[0].bytes_moved, second.demands[0].work_units) == (0, 0)
    assert second.dependencies == (first.task_id,)


def test_kernel_annotations_use_current_stage_and_semantic_owner_fields():
    assert classify_kernel("mul_mat_q", {"semantic_operator_id": "blk0:ffn_gate"}).stage == "ffn"
    assert classify_kernel("mul_mat_q", {"stage": "attention_qkv"}).stage == "attention_qkv"
    # Nsight's original CSV columns remain a supported input format.
    result = map_kernel_row({"Name": "mul_mat_q", "Total Time (ns)": "1,000",
                             "Instances": "4", "Avg (ns)": "250"})
    assert (result.total_ns, result.instances, result.avg_ns) == (1000, 4, 250)
    assert result.stage == "unknown"


@pytest.mark.parametrize("alias", ["semantic_operator", "operator_id", "graph_stage", "node_stage", "op_stage"])
def test_retired_kernel_annotations_are_rejected(alias):
    with pytest.raises(ValueError, match="unsupported kernel annotation fields"):
        classify_kernel("mul_mat_q", {alias: "ffn", "stage": "ffn"})


def _source_query():
    contract = MMVQSourceContract(1200, 1200, 32, SOURCE_SHA256, "a" * 64, True, False, True)
    work = derive_mmvq_work(m=1, k=256, n=128, weight_format="Q4_K", contract=contract,
                            allow_k_formats=True)
    audit = {**work.to_metadata(), "status": "source_geometry_unpriced", "stage": "matrix",
             "execution_component": "gpu", "channels": 1, "samples": 1}
    geometry = {"m": 1, "n": 128, "k_logical": 256, "weight_formats": ("q4_k",),
                "activation_storage_bytes": work.logical_input_f32_bytes,
                "output_storage_bytes": work.output_f32_bytes, "accumulator_bits": 32,
                "model_weight_read": True, "rhs_is_activation": False}
    meta = {"phase": "gpu_gemm", "target_component": "gpu", "mmvq_source_work": audit,
            "mmq_source_work": {"status": "mmvq_precedes_mmq"}, "kernel_query_geometry": geometry,
            "kernel_main_consumer_storage_bytes": work.consumer_q8_1_unique_bytes}
    return meta, geometry


def test_kernel_query_ledger_uses_current_source_dimensions_without_native_claims():
    meta, geometry = _source_query()
    assert _source_mmvq_executed_k(meta, geometry) == (256, "source_derived_mmvq_logical_K")
    task = TaskSpec("gemm", "request", "gemm", TaskCategory.COMPUTE, metadata=meta)
    summary = summarize_kernel_queries((task, task))
    assert summary["gpu_gemm_tasks"] == 1
    signature = summary["signatures"][0]["key"]
    assert signature["k_executed"] == 256
    assert signature["native_dispatch_proven"] is False
    assert signature["calibration_eligible"] is False
    conflicting = replace(task, metadata={**meta, "target_component": "other-gpu"})
    with pytest.raises(ValueError, match="conflicting geometry"):
        summarize_kernel_queries((task, conflicting))


@pytest.mark.parametrize("alias", ["channel", "nchannels", "channel_count", "sample", "nsamples", "sample_count"])
def test_old_query_dimension_aliases_cannot_qualify_source_work(alias):
    meta, geometry = _source_query()
    meta["mmvq_source_work"][alias] = 1
    assert _source_mmvq_executed_k(meta, geometry) == (None, "unsupported_mmvq_source_fields")


def test_current_layer_coverage_and_calibration_stage_have_distinct_semantics():
    common = {"execution_phase": "decode", "token_shape": "4x1x1x1"}
    assert _invocation_fields({**common, "coverage_component": "full_attention",
        "projection_id": "attention.qkv"}) == ("attention_qkv", "decode", "4x1x1x1")
    assert _invocation_fields({**common, "coverage_component": "full_attention",
        "projection_id": "attention.output"})[0] is None
    assert _invocation_fields({**common, "coverage_component": "dense_ffn",
        "projection_id": "mlp.up"})[0] == "ffn"
    assert _invocation_fields({**common, "coverage_component": "full_attention",
        "calibration_stage": "attention_output"})[0] == "attention_output"
