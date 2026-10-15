"""Physical and cache dispatch use the current phase, not an operator estimate."""
from dataclasses import replace

import pytest

from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import (
    CacheHierarchyProfile, CacheLevelProfile, GPUProfile, HBMProfile, TensorCoreProfile,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.kernel_memory import attach_l2_contract, resolve_l2_task
from heterollm_sim.memory_types import NandConfig, Operation
from test_native_kernel_physical_cost import native_task


def test_nand_host_direction_uses_the_current_query_without_duplicate_properties():
    config = NandConfig(host_bandwidth_gb_s=8., host_read_bandwidth_gb_s=6.,
                        host_write_bandwidth_gb_s=2.)
    assert config.host_bandwidth_for(Operation.READ) == 6.
    assert config.host_bandwidth_for("write") == 2.
    assert config.host_bandwidth_gb_s == 8.
    with pytest.raises(ValueError, match="read or write"):
        config.host_bandwidth_for(Operation.ERASE)
    for retired in ("effective_host_read_bandwidth_gb_s", "effective_host_write_bandwidth_gb_s"):
        with pytest.raises(AttributeError):
            getattr(config, retired)


@pytest.mark.parametrize("l2", [False, True])
def test_physical_dispatch_updates_its_phase_and_preserves_the_operator_estimate(l2):
    task, estimate = native_task(1., l2=l2)
    task = replace(task, metadata={**task.metadata,
        "kernel_prediction": task.metadata["phase_metadata"]["kernel_model"]})
    event = UnifiedEventKernel.from_closed_graph((task,)).step()
    metadata = event.task.metadata
    phase = metadata["phase_metadata"]["kernel_model"]
    assert phase["analytical_service_ns"] == pytest.approx(event.end_ns - event.start_ns)
    assert phase["prediction"]["physical_memory_recost"]["memory_completion_span_ns"] > 0
    assert metadata["kernel_prediction"] == phase
    assert metadata["cost_model"] == estimate.metadata
    assert metadata["cost_model"]["kernel_model"] != phase


def test_physical_auxiliary_phase_does_not_borrow_the_operator_kernel_envelope():
    task, estimate = native_task(1.)
    original_stream = next(d for d in task.demands if d.resource_id.endswith(".device_stream"))
    task = replace(task, metadata={**task.metadata, "phase_metadata": {"auxiliary": True}})
    event = UnifiedEventKernel.from_closed_graph((task,)).step()
    assert next(d for d in event.demands if d.resource_id == original_stream.resource_id) == original_stream
    assert event.task.metadata["phase_metadata"] == {"auxiliary": True}
    assert event.task.metadata["cost_model"] == estimate.metadata


def attach_cache(phase_metadata):
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1.),
        CacheHierarchyProfile((CacheLevelProfile("gpu.l2", 4096, 64, 3., 1000.),)),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1., reduction_ops_per_cycle_per_sm=1.,
        special_function_units_per_sm=1, special_function_ops_per_cycle=1.)
    hbm = HBMProfile(100.)
    task = TaskSpec("auxiliary", "request", "memory", TaskCategory.COMPUTE,
        demands=(ResourceDemand(hbm.resource_id, 1., bytes_moved=64),),
        metadata={"phase_metadata": phase_metadata, "cost_model": {"kernel_model": {
            "prediction": {"model": "measured"}, "achieved_bandwidth_gb_s": 2.,
            "pipeline_ns": {"dependency": 99.}}}})
    return attach_l2_contract(task, gpu=gpu, hbm=hbm, memory_resource=hbm.resource_id,
        cache_resource="gpu.l2", owner="gpu.l2", accesses=(CacheAccess.read("input", 0, 64),))


def test_l2_auxiliary_phase_uses_its_profile_without_borrowing_an_operator_prediction():
    contract = attach_cache({}).metadata["stateful_l2"]
    assert contract["bandwidth_gb_s"] == 100.
    assert contract["dependency_ns"] == 0.


def test_l2_native_phase_uses_its_own_prediction_and_dependency():
    contract = attach_cache({"kernel_model": {
        "prediction": {"model": "analytical"}, "achieved_bandwidth_gb_s": 17.,
        "pipeline_ns": {"dependency": 9.}}}).metadata["stateful_l2"]
    assert contract["bandwidth_gb_s"] == 17.
    assert contract["dependency_ns"] == 9.
    with pytest.raises(ValueError, match="measured kernel cache protocol"):
        attach_cache({"kernel_model": {"prediction": {"model": "measured"}}})


def test_generic_analytical_phase_prediction_tracks_dispatched_cache_service():
    task = attach_cache({"kernel_prediction": {
        "model": "analytical", "prediction_ns": 50., "reason": "analytical_gpu_profile"}})
    resolved = resolve_l2_task(task, {})
    prediction = resolved.metadata["phase_metadata"]["kernel_prediction"]
    assert prediction["prediction_ns"] == resolved.metadata["analytical_service_ns"]
    assert prediction["prediction_ns"] != 50.
    assert prediction["reason"] == "stateful_l2_runtime_recost"
    assert resolved.metadata["cost_model"] == task.metadata["cost_model"]
