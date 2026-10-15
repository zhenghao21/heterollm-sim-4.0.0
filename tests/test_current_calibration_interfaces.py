"""Current calibration and serving metadata have explicit owners and fields."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.calibration import (
    NativeCalibrationProfile, calibrate_cost_phase, request_marker_calibration_ns,
)
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import CostPhase
from heterollm_sim.cuda_graph_costs import _apply_graph_device_dispatch
from heterollm_sim.kernel_model import KernelModelProfile
from heterollm_sim.planner import _ordered_residency_accesses
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import _OnlineRuntime, _serving_resource_policy


def test_current_graph_profile_keeps_optional_device_calibration_explicit():
    task = TaskSpec("launch", "request", "launch", TaskCategory.COMPUTE,
                    demands=(ResourceDemand("gpu.frontend", 20),))
    profile = KernelModelProfile("gpu", "runtime", "arch")
    replacements, audit = _apply_graph_device_dispatch((task,), profile, captured=True)
    assert replacements == {}
    assert audit["applied"] is False
    calibrated = replace(profile, graph_device_launch_ns=7,
                         graph_device_launch_evidence="independent device timing")
    replacements, audit = _apply_graph_device_dispatch((task,), calibrated, captured=True)
    assert replacements["launch"].demands[0].service_ns == 7
    assert audit["applied"] is True
    with pytest.raises(AttributeError, match="graph_device_launch_ns"):
        _apply_graph_device_dispatch((task,), SimpleNamespace(), captured=True)


def _exact_profile_and_phase():
    profile = NativeCalibrationProfile(kernel_stage_mapping={"exact_operator_keys": {
        "attention_qkv|decode|4x1x1x1|f32|contiguous|mmvq": {
            "status": "calibrated", "train_ns_per_instance": 12, "train_instances": 2,
        },
    }})
    phase = CostPhase("gemm", TaskCategory.COMPUTE, (ResourceDemand("gpu.compute", 100),))
    metadata = {"projection_id": "attention.qkv", "execution_phase": "decode",
                "token_shape": "4x1x1x1", "semantic_dtype": "f32",
                "semantic_layout": "contiguous", "kernel_family": "mmvq"}
    return profile, phase, metadata


def test_exact_operator_calibration_uses_current_semantic_dimensions():
    profile, phase, metadata = _exact_profile_and_phase()
    result = calibrate_cost_phase(phase, metadata, profile)
    assert result.demands[0].service_ns == 12
    assert result.metadata["native_calibration_operator_key_resolution"]["mode"] == "exact"


@pytest.mark.parametrize("current,retired", [
    ("execution_phase", "phase"), ("token_shape", "shape"),
    ("semantic_dtype", "semantic_type"), ("semantic_dtype", "dtype"),
    ("semantic_dtype", "calibration_dtype"), ("semantic_layout", "layout"),
    ("semantic_layout", "calibration_layout"), ("kernel_family", "calibration_kernel_family"),
])
def test_retired_operator_fields_cannot_select_a_measurement(current, retired):
    profile, phase, metadata = _exact_profile_and_phase()
    metadata[retired] = metadata.pop(current)
    result = calibrate_cost_phase(phase, metadata, profile)
    assert result.demands == phase.demands
    assert not result.metadata.get("native_calibration_applied", False)


@pytest.mark.parametrize("current,retired", [("request_begin", "request_arrival"),
                                             ("request_end", "request_done")])
def test_request_marker_names_are_distinct_from_trace_event_names(current, retired):
    profile = NativeCalibrationProfile(request_marker_policy="additive_once_per_request",
        request_marker_ns={current: 9}, request_marker_evidence={current: {"status": "calibrated"}},
        request_shape={"prompt_tokens": 1, "output_tokens": 2})
    args = {"prompt_tokens": 1, "output_tokens": 2, "required": True}
    assert request_marker_calibration_ns(profile, current, **args) == 9
    with pytest.raises(ValueError, match="marker"):
        request_marker_calibration_ns(profile, retired, **args)


def _runtime_scenario(runtime):
    scenario = build_reference_scenario()
    return replace(scenario, workload=replace(scenario.workload, metadata={"serving_runtime": runtime}))


def test_serving_runtime_uses_workload_configuration_and_nested_prompt_cache():
    scenario = _runtime_scenario({"contention_enabled": True, "residency_granule_bytes": 256,
        "prompt_cache": {"enabled": True, "slot_allocation_bytes": 512,
                         "resident_entry_bytes": 128, "graph_residency_mode": "explicit"}})
    policy = _serving_resource_policy(scenario)
    assert policy.enabled and policy.residency_granule_bytes == 256
    assert policy.prompt_cache.slot_allocation_bytes == 512
    assert policy.prompt_cache.resident_entry_bytes == 128
    assert policy.prompt_cache.graph_residency_mode == "explicit"
    old = replace(scenario, placement=replace(scenario.placement,
                  metadata={"serving_runtime": {"contention_enabled": False}}))
    with pytest.raises(ValueError, match="workload.metadata"):
        _serving_resource_policy(old)


@pytest.mark.parametrize("runtime", [
    {"enabled": True}, {"resource_contention": {"contention_enabled": True}},
    {"prompt_cache_enabled": True}, {"prompt_cache": True},
    {"prompt_cache": {"full_slot_bytes": 512}},
    {"prompt_cache": {"logical_entry_bytes": 512}},
    {"prompt_cache": {"graph_resident_bytes": 128}},
    {"prompt_cache": {"retain_completed_entries": True}},
    {"prompt_cache": {"enabled": True, "graph_residency_mode": "full"}},
    {"fault_latency_scope": "cohort"},
])
def test_serving_runtime_rejects_retired_configuration_forms(runtime):
    with pytest.raises(ValueError):
        _serving_resource_policy(_runtime_scenario(runtime))


def test_residency_trace_preserves_owner_view_and_distinct_range_semantics():
    task = TaskSpec("read", "request", "read", TaskCategory.MEMORY,
        metadata={"event_kind": "model_weight_read", "weight_tensor_id": "weight.view",
                  "logical_weight_tensor": "weight.owner", "tensor": "weight.physical",
                  "weight_read_bytes": 512, "weight_read_invocation_id": "invocation",
                  "operator_invocation_group_id": "group", "weight_read_offset_bytes": 32,
                  "weight_read_size_bytes": 128})
    trace = _ordered_residency_accesses((task,))
    access, = _OnlineRuntime._owner_residency_trace({"residency_accesses": trace})
    assert access["requested_tensor_id"] == "weight.view"
    assert access["canonical_owner_id"] == "weight.owner"
    assert access["physical_tensor"] == "weight.physical"
    assert (access["offset_bytes"], _OnlineRuntime._owner_residency_access_size(access)) == (32, 128)
    whole = replace(task, metadata={key: value for key, value in task.metadata.items()
                                   if key not in {"weight_read_offset_bytes", "weight_read_size_bytes"}})
    whole_access, = _ordered_residency_accesses((whole,))
    assert whole_access["byte_count"] == 512
    assert _OnlineRuntime._owner_residency_access_size(whole_access) is None
    with pytest.raises(ValueError, match="unsupported residency access fields"):
        _OnlineRuntime._owner_residency_trace({"residency_accesses": [{**access, "tensor_id": "old"}]})
