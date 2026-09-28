"""Shared llama preparation and direct HBM/HBF capacity regression checks."""
from dataclasses import replace

import pytest

from heterollm_sim.ir import model_graph_execution_view
from heterollm_sim.llama_scenario import (
    apply_llama_runtime_config, llama_cpp_kv_layer_mapping, prepare_llama_scenario,
)
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.serving import compile_serving_plan
from tests.model_helpers import model_from_layer_specs
from tests.test_memory_tier_placement import _scenario


def authored(*, hbm_bytes=8192, **runtime):
    scenario = _scenario(hybrid=True)
    layers = tuple(item.layer for item in model_graph_execution_view(scenario.model.graph).layer_instances)
    model = model_from_layer_specs(scenario.model.name, layers,
                                  vocabulary_size=64, embedding_weight_bytes=8192)
    return replace(scenario, model=model,
        hardware=replace(scenario.hardware, components=tuple(
            replace(component, capacity_bytes=hbm_bytes) if component.normalized_kind == "hbm"
            else component for component in scenario.hardware.components)),
        llama_cpp_config=LlamaCppRuntimeConfig(batch=32, ubatch=16, context=256,
                                               device_memory_tiering=True, **runtime))


def test_shared_capacity_allocates_weights_kv_and_linear_state_to_direct_buffers():
    prepared = prepare_llama_scenario(authored())
    metadata = prepared.placement.metadata
    policy = metadata["llama_device_memory_policy"]
    assert policy["mode"] == "source_rule_approximation"
    assert policy["native_dispatch_proven"] is False
    assert policy["state_reserved_bytes"]["hbf0"] > 0
    assert policy["weight_bytes"]["hbf0"] > 0
    assert policy["state_reserved_bytes"]["hbm0"] > 0
    for component, amount in policy["total_allocated_bytes"].items():
        assert amount <= policy["capacity_bytes"][component]
        assert amount == policy["state_reserved_bytes"].get(component, 0) + policy["weight_bytes"].get(component, 0)
    assert metadata["llama_backend_memory"]["hbf0"] == {
        "device_id": "gpu0", "buffer_kind": "HBF", "access": "direct", "priority": 1,
    }
    assert prepared.placement.kv_policy.offload_component is None
    assert set(metadata["llama_cpp_kv_layer_components"].values()) == {"hbf0"}
    assert set(metadata["llama_cpp_linear_state_layer_components"].values()) == {"hbm0", "hbm1"}
    assert prepare_llama_scenario(prepared) is prepared


def test_hbm_is_used_before_hbf_and_capacity_edit_recomputes_placement():
    roomy = prepare_llama_scenario(authored(hbm_bytes=1024 * 1024))
    assert "hbf0" not in roomy.placement.metadata["llama_device_memory_policy"]["total_allocated_bytes"]
    cramped = replace(roomy, hardware=authored().hardware)
    updated = prepare_llama_scenario(cramped)
    assert updated is not cramped
    assert updated.placement.metadata["llama_device_memory_policy"]["weight_bytes"]["hbf0"] > 0


def test_shared_capacity_fails_instead_of_overcommitting():
    scenario = authored()
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        replace(c, capacity_bytes=100) if c.component_id == "hbf0" else c
        for c in scenario.hardware.components)))
    with pytest.raises(ValueError, match="capacity exhausted"):
        prepare_llama_scenario(scenario)


def test_partial_offload_counts_linear_layers_and_output_in_kv_boundary():
    scenario = authored(gpu_layers=2)
    mapping = llama_cpp_kv_layer_mapping(scenario, scenario.llama_cpp_config)
    # full0, full1, linear0, linear1, output: two units offload only
    # linear1 + output. Neither full-attention layer owns GPU KV.
    assert mapping["kv_layer_components"] == {"full0": "hostmem0", "full1": "hostmem0"}
    prepared = prepare_llama_scenario(scenario)
    assert prepared.placement.tensor_to_component["linear0.linear_state"] == "hostmem0"
    assert prepared.placement.tensor_to_component["linear1.linear_state"] == "hbm0"


def test_existing_native_evidence_and_generic_policy_are_preserved():
    scenario = authored(hbm_bytes=1024 * 1024)
    generic = replace(scenario, llama_cpp_config=replace(scenario.llama_cpp_config, policy="generic"))
    assert prepare_llama_scenario(generic) is generic
    config = replace(scenario.llama_cpp_config, device_memory_tiering=False)
    native = apply_llama_runtime_config(scenario, config)
    native = replace(native, workload=replace(native.workload,
        metadata={**native.workload.metadata, "native_test_evidence": {"sha256": "bound"}}))
    assert prepare_llama_scenario(native) is native


def test_hbf_template_binding_requires_explicit_write_throughput():
    scenario = authored()
    profile = scenario.component_profiles["host_memory"]["hbf-memory"]
    from heterollm_sim.serde import to_primitive
    template = to_primitive(profile)
    hbf = scenario.hardware.get_component("hbf0")
    profiles = {kind: dict(registry) for kind, registry in scenario.component_profiles.items()}
    del profiles["host_memory"]["hbf-memory"]
    hbf = replace(hbf, cost_profile_id=None,
        metadata={**hbf.metadata, "access_mode": "remote_flash", "cost_profile_template": template})
    scenario = replace(scenario, component_profiles=profiles, hardware=replace(scenario.hardware,
        components=tuple(hbf if c.component_id == hbf.component_id else c for c in scenario.hardware.components)))
    prepared = prepare_llama_scenario(scenario)
    assert prepared.hardware.get_component("hbf0").is_active_memory
    assert prepared.resolve_component_profile("hbf0").write_latency_ns == 200
    # A template-provided directional write value is explicit configuration;
    # a zero component field may inherit it and is never guessed from reads.


def test_serving_honors_direct_hbf_kv_owner():
    prepared = prepare_llama_scenario(authored())
    plan = compile_serving_plan(prepared)
    assert set(plan.kv_layer_components.values()) == {"hbf0"}
    assert set(plan.kv_component_bytes_per_page) == {"hbf0"}
