"""Compilation and control-plane planning share the current typed contracts."""
from dataclasses import replace

import pytest

from heterollm_sim.compiler_ir import CompilerOptions, _build_workload_graph
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.control_plane import (
    _control_plane_evidence_equal, _runtime_placement_equal,
)
from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.ir import (
    ComponentSpec, HardwareSpec, LayerSpec, ModelSpec, RequestSpec,
    build_model_graph_from_layer_specs,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v4 import ControllerProfile
from heterollm_sim.web import scenario_to_payload


def test_compiler_preserves_explicit_request_fields_and_synthetic_arrivals():
    scenario = build_reference_scenario()
    request = RequestSpec("explicit", 15., 23, 7, priority=2,
                          deadline_ns=100., metadata={"session": "a"})
    explicit = replace(scenario, workload=replace(scenario.workload, requests=(request,)))
    row, = _build_workload_graph(explicit, CompilerOptions()).requests
    assert (row.request_id, row.arrival_ns, row.prompt_tokens, row.output_tokens,
            row.priority, row.deadline_ns, row.attributes) == (
        "explicit", 15., 23, 7, 2, 100., {"session": "a"})
    synthetic = replace(scenario, workload=replace(scenario.workload,
        requests=(), request_count=3, prompt_tokens=17, output_tokens=5, arrival_rate_rps=2.))
    assert _build_workload_graph(synthetic, CompilerOptions()).requests == ()
    rows = _build_workload_graph(synthetic, CompilerOptions(expand_synthetic_requests=True)).requests
    assert [(row.request_id, row.arrival_ns, row.prompt_tokens, row.output_tokens)
            for row in rows] == [("request-0000", 0., 17, 5),
                                ("request-0001", 500_000_000., 17, 5),
                                ("request-0002", 1_000_000_000., 17, 5)]


def test_placement_runtime_equality_keeps_evidence_separate_from_semantics():
    placement = replace(build_reference_scenario().placement,
        metadata={"control_plane": {"decision": {"owner": "gpu0"},
            "evidence": {"input_fingerprint": "first", "fingerprint_schema": "current"}}})
    changed_evidence = replace(placement,
        metadata={"control_plane": {"decision": {"owner": "gpu0"},
            "evidence": {"input_fingerprint": "second", "fingerprint_schema": "current"}}})
    assert _runtime_placement_equal(placement, changed_evidence)
    assert not _control_plane_evidence_equal(placement, changed_evidence)
    assert not _runtime_placement_equal(placement, replace(placement,
        tensor_bytes={**placement.tensor_bytes, "extra": 128}))
    assert _control_plane_evidence_equal(replace(placement, metadata={}), replace(placement, metadata={}))


def test_default_controllers_use_parsed_hardware_and_omitted_runtime_profile():
    hardware = HardwareSpec("two-gpus", components=(
        ComponentSpec("gpu-b", "gpu"), ComponentSpec("gpu-a", "gpu"),
        ComponentSpec("cpu", "cpu")), links=())
    assert tuple(ControllerProfile.architecture_default(hardware).gpu_controllers) == ("gpu-a", "gpu-b")
    with pytest.raises(AttributeError):
        ControllerProfile.architecture_default({"components": [{"kind": "gpu", "component_id": "old"}]})
    payload = scenario_to_payload(build_reference_scenario())
    payload["profiles"].pop("runtime")
    assert tuple(scenario_from_dict(payload).runtime_profile.gpu_controllers) == ("gpu0",)


@pytest.mark.parametrize("head_dim,expected_width", [(0, 64), (24, 96)])
def test_control_plane_uses_layer_geometry_instead_of_retired_metadata(head_dim, expected_width):
    scenario = build_reference_scenario()
    layer = LayerSpec("audit", "dense", 64, 128, 4, attention_head_dim=head_dim,
                      metadata={"attention_head_dim": 999})
    model = ModelSpec(name="current-geometry", graph=build_model_graph_from_layer_specs(
        "current-geometry", (layer,), vocabulary_size=128, max_sequence_length=256))
    scenario = replace(scenario, model=model,
                       placement=replace(scenario.placement, model_name=model.name))
    attention = next(row for row in _derive_requirements(scenario) if row.kind == "attention")
    assert [(matrix.k, matrix.n) for matrix in attention.matrices] == [
        (64, 3 * expected_width), (expected_width, 64)]
