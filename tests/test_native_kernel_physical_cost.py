"""Native kernel envelopes must not retain a replaced DRAM estimate."""
from dataclasses import replace

import pytest

from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.cost_models import (
    CacheHierarchyProfile, CacheLevelProfile, GemmWorkload, GPUProfile,
    HBMProfile, TensorCoreProfile,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.data_motion import lower_physical_allocations
from heterollm_sim.kernel_memory import attach_l2_contract
from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile, estimate_kernel
from heterollm_sim.memory_types import make_hbm_config
from heterollm_sim.mmq_work import derive_mmq_work


def native_task(bandwidth, *, l2=False, compute_cycles=1., launch=0.):
    capability = KernelCapability("declared", ("fp16",), "fp16", "decode", "tensor",
        "analytical", internal_dtype="fp16", dependency_cycles=7.)
    model = KernelModelProfile("device", "runtime", "architecture", kernels=(capability,),
                               stateful_l2=l2)
    cache = CacheHierarchyProfile((CacheLevelProfile("gpu.l2", 1 << 20, 64, 3., 1000.),))
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1., cycles_per_mma=compute_cycles), cache,
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1, reduction_ops_per_cycle_per_sm=1,
        special_function_units_per_sm=1, special_function_ops_per_cycle=1,
         kernel_launch_ns=launch,  kernel_model=model)
    hbm = HBMProfile(bandwidth)
    work = GemmWorkload(1, 16, 16, activation_bits=16, weight_bits=16, execution_phase="decode")
    estimate = estimate_kernel(gpu, hbm, work)
    phase = estimate.phases[-1]
    reads, writes = work.activation_bytes + work.weight_bytes, work.output_bytes
    task = TaskSpec("kernel", "request", "gemm", TaskCategory.COMPUTE, demands=phase.demands,
        metadata={"physical_memory_config": make_hbm_config(capacity_bytes=1 << 20),
            "physical_owner": hbm.resource_id,
            "phase_metadata": phase.metadata, "cost_model": estimate.metadata,
            "memory_accesses": tuple({"operation": op, "address": address, "byte_count": count,
                "resource_id": hbm.resource_id, "buffer_id": buffer, "offset_bytes": 0,
                "allocation_size_bytes": count}
                for op, address, count, buffer in (("read", 0, reads, "input"),
                                                   ("write", 8192, writes, "output")))})
    if l2:
        task = attach_l2_contract(task, gpu=gpu, hbm=hbm, memory_resource=hbm.resource_id,
            cache_resource="gpu.l2", owner="gpu.l2", accesses=(
                CacheAccess("input", 0, reads, "read", reads),
                CacheAccess("output", 0, writes, "write", writes)))
    return lower_physical_allocations(task), estimate


@pytest.mark.parametrize("l2", [False, True])
@pytest.mark.parametrize("compute_cycles", [1., 1000.])
def test_native_physical_envelope_replaces_only_backing_estimate(l2, compute_cycles):
    events = []
    for bandwidth in (1000., 1.):
        task, _ = native_task(bandwidth, l2=l2, compute_cycles=compute_cycles)
        event = UnifiedEventKernel.from_closed_graph((task,)).step()
        assert "hbm.channel" not in {d.resource_id for d in event.demands}
        compute = next(d for d in event.demands if d.resource_id == "gpu.tensor_core")
        original = next(d for d in task.demands if d.resource_id == compute.resource_id)
        assert compute == original
        core_span = event.task.metadata["physical_completion_ns"] - event.start_ns
        assert event.end_ns - event.start_ns == pytest.approx(max(core_span, compute.service_ns) + 7.)
        events.append(event)
    assert events[0].end_ns == events[1].end_ns
    assert events[0].demands == events[1].demands


def test_unattached_native_estimate_still_has_memory_roofline():
    _, fast = native_task(1000.)
    _, slow = native_task(1.)
    assert slow.service_ns > fast.service_ns
    assert next(d.service_ns for d in slow.phases[-1].demands
                if d.resource_id == "hbm.channel") == pytest.approx(1536.)


def test_native_physical_envelope_keeps_launch_and_namespaced_stream():
    task, estimate = native_task(1., launch=11.)
    launch_phase = estimate.phases[0]
    launch = TaskSpec("launch", "request", "launch", TaskCategory.COMPUTE, demands=launch_phase.demands)
    task = replace(task, dependencies=("launch",), demands=tuple(
        replace(d, resource_id="rank-1." + d.resource_id) if d.resource_id.endswith(".device_stream") else d
        for d in task.demands))
    kernel = UnifiedEventKernel.from_closed_graph((launch, task))
    first, second = kernel.step(), kernel.step()
    assert first.end_ns == second.start_ns == 11.
    assert second.end_ns == second.task.metadata["physical_completion_ns"] + 7.
    assert next(d.service_ns for d in second.demands if d.resource_id.endswith(".device_stream")) == (
        second.end_ns - second.start_ns)


def test_native_warm_l2_retains_cache_lookup_without_core_or_old_memory_floor():
    task, _ = native_task(1., l2=True)
    next_task = replace(task, task_id="next", dependencies=("kernel",))
    kernel = UnifiedEventKernel.from_closed_graph((task, next_task))
    cold, warm = kernel.step(), kernel.step()
    assert warm.task.metadata["l2_execution"]["hbm_read_bytes"] == 0
    assert warm.task.metadata["l2_execution"]["hbm_write_bytes"] == 0
    assert "physical_completion_ns" not in warm.task.metadata
    cache = next(d for d in warm.demands if d.resource_id == "gpu.l2")
    assert cache.service_ns > 0
    assert warm.end_ns - warm.start_ns == max(d.service_ns for d in warm.demands)
    assert warm.end_ns - warm.start_ns < cold.end_ns - cold.start_ns


@pytest.mark.parametrize("m", [1, 16, 17])
def test_native_tensor_work_uses_the_same_issued_tiles_as_service(m):
    capability = KernelCapability("declared", ("fp16",), "fp16", "decode", "tensor",
        "analytical", internal_dtype="fp16")
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1.),
        CacheHierarchyProfile((CacheLevelProfile("cache", 1024, 64, 0, 1000.),)),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1, reduction_ops_per_cycle_per_sm=1,
        special_function_units_per_sm=1, special_function_ops_per_cycle=1,
        kernel_model=KernelModelProfile("device", "runtime", "architecture", kernels=(capability,)))
    result = estimate_kernel(gpu, HBMProfile(1000.), GemmWorkload(m, 16, 16,
        activation_bits=16, weight_bits=16, execution_phase="decode"))
    demand = next(d for d in result.phases[-1].demands if d.resource_id == gpu.tensor_core.resource_id)
    issued = ((m + 15) // 16) * 8192
    assert result.useful_ops == 2 * m * 16 * 16
    assert demand.work_units == issued
    assert result.metadata["kernel_model"]["issued_dot_operations"] == issued
    assert demand.service_ns == issued / (gpu.tensor_core.peak_tops("fp16") * 1000.)


def test_native_mmq_keeps_unfused_source_extent():
    source = derive_mmq_work(m=17, k=256, n=129, weight_format="Q4_K", sm_count=1,
                             shared_memory_per_block=65536)
    capability = KernelCapability("source_mmq", ("q4_k",), "fp32", "prefill", "tensor",
        "analytical", internal_dtype="int8", output_bits=32)
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1.),
        CacheHierarchyProfile((CacheLevelProfile("cache", 1024, 64, 0, 1000.),)),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1, reduction_ops_per_cycle_per_sm=1,
        special_function_units_per_sm=1, special_function_ops_per_cycle=1,

        kernel_model=KernelModelProfile("device", "runtime", "architecture", kernels=(capability,)))
    work = GemmWorkload(17, 256, 129, activation_bits=32, weight_bits=4, output_bits=32,
        activation_dtype="fp32", packed_weight_formats=("q4_k",), execution_phase="prefill",
        weight_storage_bytes=source.logical_weight_bytes,
        activation_storage_bytes=source.consumer_unique_bytes, mmq_work=source)
    result = estimate_kernel(gpu, HBMProfile(1000.), work)
    demands = {d.resource_id: d for d in result.phases[-1].demands}
    tensor = demands[gpu.tensor_core.resource_id]
    assert tensor.work_units == source.source_nominal_arithmetic_operations
    assert result.useful_ops == work.operations
    assert demands[gpu.scalar_resource_id].work_units == 0
    assert demands[gpu.special_function_resource_id].work_units == 0
