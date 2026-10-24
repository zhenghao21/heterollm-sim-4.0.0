"""Native cache traffic is not a nonnegative affine context coordinate."""

import pytest

from heterollm_sim.cost_models import (
    CacheHierarchyProfile, CacheLevelProfile, GPUProfile, HBMProfile,
    ReductionWorkload, TensorCoreProfile, estimate_gpu_reduction,
)


def _softmax_reduce(context_tokens, *, streaming_fraction=0.):
    # Same native BF16 softmax_reduce workload as planner.py: R=2*T,
    # working_set=2*T, reuse=2, and the two max/sum output fields.
    cache = CacheHierarchyProfile((CacheLevelProfile(
        "L2", 128, 32, 1., 100., resource_id="gpu.l2"),))
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1.), cache,
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1.,
        reduction_ops_per_cycle_per_sm=1., special_function_units_per_sm=1,
        special_function_ops_per_cycle=1.)
    workload = ReductionWorkload(input_elements=context_tokens,
        output_elements=1, operations_per_combine=2,
        input_bits=16, output_bits=32,
        dependency_depth=(context_tokens-1).bit_length(),
        working_set_bytes=2*context_tokens, reuse_factor=2.,
        streaming_fraction=streaming_fraction, name="softmax_reduce_max_sum")
    return estimate_gpu_reduction(gpu, HBMProfile(1.), workload)


def test_native_softmax_cache_capacity_crossing_exceeds_context_affine_bulk_bound():
    left, right = _softmax_reduce(64), _softmax_reduce(128)
    assert left.metadata["model"] == right.metadata["model"] == "gpu_reduction_roofline"
    a, b = left.metadata["cache"], right.metadata["cache"]
    assert a["cache_model"] == b["cache_model"] == "v4_directional_closed_interval"
    assert a["cross_invocation_cache_state"] is b["cross_invocation_cache_state"] is False
    assert (a["physical_read_bytes"], b["physical_read_bytes"]) == (64, 192)
    assert (a["physical_write_bytes"], b["physical_write_bytes"]) == (4, 4)
    # The actual native context grows 2x while its backing read grows 3x.
    # A phase-only row/refresh guard cannot repair this bulk-byte deficit.
    assert b["physical_read_bytes"] > 2*a["physical_read_bytes"]
    assert (a["backing_service_ns"], b["backing_service_ns"]) == (68., 196.)
    assert b["backing_service_ns"] > 2*a["backing_service_ns"]


def test_native_streaming_reads_retain_the_context_affine_bulk_coordinate():
    left, right = (_softmax_reduce(value, streaming_fraction=1.) for value in (64, 128))
    a, b = left.metadata["cache"], right.metadata["cache"]
    assert (a["physical_read_bytes"], b["physical_read_bytes"]) == (128, 256)
    assert b["physical_read_bytes"] == 2*a["physical_read_bytes"]
    assert a["physical_write_bytes"] == b["physical_write_bytes"] == 4
    assert b["backing_service_ns"] <= 2*a["backing_service_ns"]
