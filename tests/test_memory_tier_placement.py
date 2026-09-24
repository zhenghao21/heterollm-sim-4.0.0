"""Opt-in memory-tier constraints: ownership, capacity and legacy isolation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.config import (
    _hbm_profile_from_dict, _host_memory_profile_from_dict,
    normalize_cost_profile_kind, scenario_from_dict,
)
from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.control_plane_state import mapping_fingerprint_status
from heterollm_sim.cost_models import HBMProfile, HostMemoryProfile
from heterollm_sim.ir import ComponentSpec, LayerSpec, LinearAttentionSpec, model_graph_execution_view
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.planner import _kv_components, _linear_state_components
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import (
    _KVLedger, _LinearStateLedger, _PhysicalCapacityLedger, _physical_runtime_limits,
    _kv_bytes_per_token, _linear_state_bytes_per_request,
    compile_serving_plan, simulate_online,
)
from tests.model_helpers import model_from_layer_specs
from tests.test_config_v05 import reference_payload
from tests.test_storage_v03 import _add_storage


def _scenario(*, mode="memory", hybrid=False):
    scenario, _ = _add_storage(build_reference_scenario(), "hbf", "UCIe")
    if hybrid:
        full = LayerSpec("full0", "dense", hidden_size=64, intermediate_size=128,
                         attention_heads=4, dtype="int8", quantization="w8a8")
        linear = LayerSpec("linear0", "dense", hidden_size=64, intermediate_size=128,
                           attention_heads=4, sequence_mixer="linear_attention",
                           linear_attention=LinearAttentionSpec(2, 4, 16, 16, 4),
                           dtype="int8", quantization="w8a8")
        model = model_from_layer_specs("tier-hybrid", (full, replace(full, layer_id="full1"),
                                                        linear, replace(linear, layer_id="linear1")))
        scenario = replace(scenario, model=model, placement=replace(
            scenario.placement, model_name=model.name,
            parallel=replace(scenario.placement.parallel, layer_to_stage={
                name: 0 for name in ("full0", "full1", "linear0", "linear1")}),
        ))
    if mode == "memory":
        profile = HostMemoryProfile(
            bandwidth_gb_s=4.0, read_latency_ns=100.0, write_latency_ns=200.0,
            transaction_bytes=4096, max_outstanding_requests=32,
            resource_id="hbf0.media",
        )
        profiles = {kind: dict(registry) for kind, registry in scenario.component_profiles.items()}
        profiles["host_memory"]["hbf-memory"] = profile
        scenario = replace(scenario, component_profiles=profiles, hardware=replace(
            scenario.hardware, components=tuple(
                replace(c, cost_profile_id="hbf-memory", metadata={
                    **c.metadata, "access_mode": "memory", "write_buffer_bytes": 0,
                    "memory_service_owner": "hbf0.media", "max_outstanding_requests": 32,
                }) if c.component_id == "hbf0" else c for c in scenario.hardware.components
            )))
    scenario = replace(scenario, workload=replace(
        scenario.workload, requests=(), request_count=1, prompt_tokens=8, output_tokens=3,
        mtp=None, scheduler=replace(scenario.workload.scheduler, max_num_seqs=1),
    ), placement=replace(scenario.placement, kv_policy=replace(
        scenario.placement.kv_policy, offload_component=None,
    )))
    return scenario


def _solve(scenario, **options):
    decision = plan_runtime_placement(scenario, PlacementPolicy(**options))
    assert decision.fully_placed, decision.unplaced
    return decision.apply(scenario)


def test_hbf_memory_is_an_explicit_interface_not_a_kind_alias():
    assert ComponentSpec("f", "hbf").memory_class == "offload"
    assert ComponentSpec("f", "hbf", metadata={"access_mode": "remote_flash"}).memory_class == "offload"
    memory = ComponentSpec("f", "hbf", metadata={"access_mode": "memory", "write_buffer_bytes": 0})
    assert memory.is_active_memory and memory.is_storage and memory.normalized_kind == "hbf"
    assert normalize_cost_profile_kind("hbf") is None
    for metadata in ({"access_mode": "dram"}, {"access_mode": None},
                     {"access_mode": "memory"}, {"access_mode": "memory", "write_buffer_bytes": 1},
                     {"access_mode": "memory", "write_buffer_bytes": False}):
        with pytest.raises(ValueError):
            ComponentSpec("f", "hbf", metadata=metadata)


@pytest.mark.parametrize("parse", [_hbm_profile_from_dict, _host_memory_profile_from_dict])
def test_optional_memory_profile_parameters(parse):
    profile = parse({"bandwidth_gb_s": 12, "read_latency_ns": 70, "write_latency_ns": 90,
                     "transaction_bytes": 4096, "max_outstanding_requests": 4})
    assert (profile.read_latency_ns, profile.write_latency_ns,
            profile.transaction_bytes, profile.max_outstanding_requests) == (70, 90, 4096, 4)
    assert parse({"bandwidth_gb_s": 12}).read_latency_ns == 0
    for name, value in (("read_latency_ns", True), ("write_latency_ns", -1),
                        ("transaction_bytes", 2.5), ("max_outstanding_requests", False)):
        with pytest.raises(ValueError):
            parse({"bandwidth_gb_s": 12, name: value})


def test_hbf_profile_binding_is_component_sensitive_and_unambiguous():
    scenario = _scenario()
    assert isinstance(scenario.resolve_component_profile("hbf0"), HostMemoryProfile)
    profiles = {kind: dict(registry) for kind, registry in scenario.component_profiles.items()}
    host = profiles["host_memory"].pop("hbf-memory")
    profiles["hbm"]["hbf-memory"] = HBMProfile(
        bandwidth_gb_s=host.bandwidth_gb_s, read_latency_ns=host.read_latency_ns,
        write_latency_ns=host.write_latency_ns, transaction_bytes=host.transaction_bytes,
        max_outstanding_requests=host.max_outstanding_requests, resource_id=host.resource_id,
    )
    assert isinstance(replace(scenario, component_profiles=profiles).resolve_component_profile("hbf0"), HBMProfile)
    profiles["host_memory"]["hbf-memory"] = host
    with pytest.raises(ValueError, match="exactly one"):
        replace(scenario, component_profiles=profiles)
    assert _scenario(mode="remote_flash").hardware.get_component("hbf0").cost_profile_id is None


@pytest.mark.parametrize("field,value", [("memory_service_owner", "other"),
    ("transfer_granularity_bytes", 64), ("max_outstanding_requests", 1),
    ("read_latency_ns", 0), ("write_latency_ns", 1)])
def test_hbf_media_and_profile_must_share_owner_and_transaction_semantics(field, value):
    scenario = _scenario()
    with pytest.raises(ValueError, match="HBF"):
        replace(scenario, hardware=replace(scenario.hardware, components=tuple(
            replace(c, metadata={**c.metadata, field: value}) if c.component_id == "hbf0" else c
            for c in scenario.hardware.components)))


def test_hbf_single_bandwidth_profile_is_conservatively_bounded():
    scenario = _scenario()
    profiles = {kind: dict(registry) for kind, registry in scenario.component_profiles.items()}
    profiles["host_memory"]["hbf-memory"] = replace(profiles["host_memory"]["hbf-memory"], bandwidth_gb_s=5)
    with pytest.raises(ValueError, match="either explicit read/write"):
        replace(scenario, component_profiles=profiles)


def test_policy_parser_accepts_only_explicit_targets_not_manual_placement():
    payload = reference_payload()
    options = {"weight_tensor_targets": {"dense0.mlp_weights": "hbm1"},
               "kv_cache_target": "hbm0", "kv_layer_targets": {"moe1": "hbm1"},
               "linear_state_target": None, "linear_state_layer_targets": {}}
    payload["placement"]["metadata"]["control_plane"] = {"policy": {"options": options}}
    assert scenario_from_dict(payload).placement.metadata["control_plane"]["policy"]["options"] == options
    for name, value in (("weight_tensor_targets", []), ("kv_layer_targets", {"x": 1}),
                        ("kv_cache_target", False)):
        payload["placement"]["metadata"]["control_plane"]["policy"]["options"] = {name: value}
        with pytest.raises(ValueError):
            scenario_from_dict(payload)


def test_weight_targets_override_default_local_pool_but_remain_v4_output():
    mode = "memory"
    original = _scenario(mode=mode)
    mapped = _solve(original, weight_tensor_targets={"dense0.mlp_weights": "hbf0"})
    assert not original.placement.tensor_to_component
    assert mapped.placement.tensor_to_component["dense0.mlp_weights"] == "hbf0"
    decisions = mapped.placement.metadata["control_plane"]["decision"]
    assert "dense0.mlp_weights" in decisions["generated_tensor_ids"]
    assert {shard["component_id"] for shard in decisions["rank_weight_shards"]["dense0.mlp_weights"]} == {"hbf0"}
    assert plan_runtime_placement(mapped).placement == mapped.placement


def test_remote_flash_weight_target_is_fail_closed_until_load_lowering_exists():
    decision = plan_runtime_placement(_scenario(mode="remote_flash"), PlacementPolicy(
        weight_tensor_targets={"dense0.mlp_weights": "hbf0"}))
    assert not decision.fully_placed
    assert any(item.item_id == "scenario_validation" for item in decision.unplaced)


def test_kv_and_linear_state_global_and_layer_targets_are_separate():
    scenario = _scenario(hybrid=True)
    mapped = _solve(scenario, linear_state_target="hbm1",
                    kv_layer_targets={"full1": "hbf0"},
                    linear_state_layer_targets={"linear1": "hbf0"})
    targets = mapped.placement.tensor_to_component
    assert targets["kv_cache"] == "hbm0" and targets["linear_state"] == "hbm1"
    assert targets["full1.kv_cache"] == targets["linear1.linear_state"] == "hbf0"
    assert mapped.placement.metadata["memory_tiers"] == {
        "kv_layer_components": {"full1": "hbf0"},
        "linear_state_layer_components": {"linear1": "hbf0"},
    }
    rank = build_parallel_plan(mapped).ranks[0]
    layers = {item.layer.layer_id: item.layer for item in model_graph_execution_view(mapped.model.graph).layer_instances}
    assert _kv_components(mapped, rank, layer=layers["full1"])[0] == "hbf0"
    assert _linear_state_components(mapped, rank, layer=layers["linear1"])[0] == "hbf0"
    plan = compile_serving_plan(mapped)
    assert sum(plan.kv_policy.component_bytes_per_page.values()) == _kv_bytes_per_token(mapped, None) * plan.kv_policy.tokens_per_page
    assert sum(plan.linear_state_policy.component_bytes_per_request.values()) == _linear_state_bytes_per_request(mapped)
    assert set(plan.kv_policy.component_bytes_per_page) == {"hbm0", "hbf0"}
    assert set(plan.linear_state_policy.component_bytes_per_request) == {"hbm1", "hbf0"}
    assert set(_physical_runtime_limits(plan)) == {"hbm0", "hbm1", "hbf0"}


def test_kv_partitioned_allocations_are_atomic_and_released_per_physical_owner():
    mapped = _solve(_scenario(), kv_layer_targets={"moe1": "hbf0"})
    plan = compile_serving_plan(mapped)
    quanta = plan.kv_policy.component_bytes_per_page
    physical = _PhysicalCapacityLedger(quanta)
    ledger = _KVLedger(plan.kv_policy, physical)
    request = SimpleNamespace(kv_pages=0, peak_kv_pages=0, temporary_kv_pages=0)
    assert ledger.resize(request, 1)
    assert physical.used_bytes == quanta
    assert not ledger.resize(request, 2)
    assert physical.used_bytes == quanta and request.kv_pages == 1
    assert not ledger.offload(request)
    assert ledger.resize(request, 0)
    assert physical.used_bytes == dict.fromkeys(quanta, 0)


def test_linear_partitioned_allocation_cannot_overcommit_secondary_tier():
    mapped = _solve(_scenario(hybrid=True), linear_state_target="hbm1",
                    linear_state_layer_targets={"linear1": "hbf0"})
    plan = compile_serving_plan(mapped)
    quanta = plan.linear_state_policy.component_bytes_per_request
    physical = _PhysicalCapacityLedger({**quanta, "hbf0": quanta["hbf0"] - 1})
    ledger = _LinearStateLedger(plan.linear_state_policy, physical)
    request = SimpleNamespace(linear_state_resident=False)
    assert not ledger.allocate(request)
    assert all(n == 0 for n in physical.used_bytes.values())
    physical.limits["hbf0"] += 1
    assert ledger.allocate(request)
    assert physical.used_bytes == quanta
    ledger.release(request)
    assert all(n == 0 for n in physical.used_bytes.values())


def test_partition_capacity_is_limited_by_smallest_tier_not_summed_capacity():
    scenario = _scenario()
    # No weights are allowed to consume hbf0; only the one layer's 8192-byte KV quantum.
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        replace(c, capacity_bytes=8192) if c.component_id == "hbf0" else c
        for c in scenario.hardware.components)))
    mapped = _solve(scenario, kv_layer_targets={"moe1": "hbf0"})
    plan = compile_serving_plan(mapped)
    assert plan.kv_policy.capacity_pages == 1
    assert plan.kv_policy.capacity_bytes == 16384


def test_half_kv_layer_partition_runs_without_discarding_full_attention_history():
    baseline = _solve(_scenario())
    mapped = _solve(_scenario(), kv_layer_targets={"moe1": "hbf0"})
    original, changed = simulate_online(baseline), simulate_online(mapped)
    assert changed.kv_metrics.rejected_requests == 0
    assert changed.kv_metrics.logical_decode_read_bytes == original.kv_metrics.logical_decode_read_bytes
    assert changed.kv_metrics.logical_decode_read_bytes > 0
    assert changed.kv_metrics.logical_decode_write_bytes == original.kv_metrics.logical_decode_write_bytes
    assert changed.kv_metrics.swap_events == changed.kv_metrics.migration_events == 0


def test_replanning_clears_old_layer_output_instead_of_turning_it_into_a_lock():
    mapped = _solve(_scenario(), kv_layer_targets={"moe1": "hbf0"})
    remapped = _solve(mapped)
    assert "memory_tiers" not in remapped.placement.metadata
    assert "moe1.kv_cache" not in remapped.placement.tensor_to_component


@pytest.mark.parametrize("options", [
    {"weight_tensor_targets": {"made_up.weights": "hbf0"}},
    {"kv_cache_target": "hbf0"},  # Conflicts with explicit kv_policy hbm0.
    {"kv_layer_targets": {"linear0": "hbf0"}},
    {"linear_state_layer_targets": {"full0": "hbf0"}},
    {"linear_state_target": "missing"},
])
def test_invalid_or_conflicting_runtime_constraints_fail_closed(options):
    with pytest.raises(ValueError):
        plan_runtime_placement(_scenario(hybrid=True), PlacementPolicy(**options))


def test_remote_flash_cannot_be_primary_state_even_under_explicit_policy():
    with pytest.raises(ValueError, match="writable active"):
        plan_runtime_placement(_scenario(mode="remote_flash"), PlacementPolicy(kv_layer_targets={"moe1": "hbf0"}))


def test_layer_partition_does_not_claim_distributed_swap_support():
    mapped = _solve(_scenario(), kv_layer_targets={"moe1": "hbf0"})
    mapped = replace(mapped, placement=replace(mapped.placement, kv_policy=replace(
        mapped.placement.kv_policy, offload_component="hbm2")))
    with pytest.raises(ValueError, match="static layer partitioning"):
        compile_serving_plan(mapped)


def test_global_targets_are_honored_by_lowering_and_capacity():
    scenario = _scenario(hybrid=True)
    scenario = replace(scenario, placement=replace(scenario.placement,
        kv_policy=replace(scenario.placement.kv_policy, cache_component=None)))
    mapped = _solve(scenario, kv_cache_target="hbf0", linear_state_target="hbm1")
    layers = [item.layer for item in model_graph_execution_view(mapped.model.graph).layer_instances]
    rank = build_parallel_plan(mapped).ranks[0]
    assert _kv_components(mapped, rank, "gpu0", layers[0])[0] == "hbf0"
    assert _linear_state_components(mapped, rank, "gpu0", layers[2])[0] == "hbm1"
    plan = compile_serving_plan(mapped)
    assert plan.kv_policy.cache_component == "hbf0"
    assert plan.linear_state_policy.cache_component == "hbm1"
    result = simulate_online(mapped)
    assert result.kv_metrics.rejected_requests == 0
    assert result.linear_state_metrics.peak_used_bytes == _linear_state_bytes_per_request(mapped)


def test_constraint_change_invalidates_mapping_fingerprint():
    mapped = _solve(_scenario(), kv_layer_targets={"moe1": "hbf0"})
    assert not mapping_fingerprint_status(mapped)["mapping_stale"]
    control = mapped.placement.metadata["control_plane"]
    changed = replace(mapped, placement=replace(mapped.placement, metadata={
        **mapped.placement.metadata,
        "control_plane": {**control, "policy": {"options": {
            **control["policy"]["options"], "kv_layer_targets": {"moe1": "hbm1"},
        }}},
    }))
    assert mapping_fingerprint_status(changed)["mapping_stale"]


def test_linear_state_offload_mode_invalidates_mapping_fingerprint():
    mapped = _solve(_scenario(mode="remote_flash", hybrid=True), linear_state_target="hbm0", linear_state_offload_target="hbf0")
    baseline = mapping_fingerprint_status(mapped)
    assert not baseline["mapping_stale"]
    changed = replace(mapped, placement=replace(mapped.placement, metadata={
        **mapped.placement.metadata,
        "linear_state_offload_mode": "pressure",
    }))
    assert mapping_fingerprint_status(changed)["mapping_stale"]


def test_linear_state_offload_policy_roundtrip_and_generated_output():
    payload = reference_payload()
    payload["placement"]["metadata"]["control_plane"] = {"policy": {"options": {
        "linear_state_offload_target": "hbf0"}}}
    assert scenario_from_dict(payload).placement.metadata["control_plane"]["policy"]["options"]["linear_state_offload_target"] == "hbf0"
    for bad in (False, 1, "", [], {}):
        with pytest.raises(ValueError):
            PlacementPolicy(linear_state_offload_target=bad)
    scenario = _scenario(mode="remote_flash", hybrid=True)
    mapped = _solve(scenario, linear_state_target="hbm0", linear_state_offload_target="hbf0")
    assert mapped.placement.tensor_to_component["linear_state_offload"] == "hbf0"
    assert "linear_state_offload" in mapped.placement.metadata["control_plane"]["decision"]["generated_tensor_ids"]
    assert "linear_state_offload" not in mapped.placement.tensor_bytes
    assert plan_runtime_placement(mapped).placement == mapped.placement
    assert "linear_state_offload" not in _solve(mapped).placement.tensor_to_component


def test_generated_linear_state_offload_reuses_real_swap_ledger():
    scenario = _scenario(mode="remote_flash", hybrid=True)
    scenario = replace(scenario, placement=replace(scenario.placement,
        kv_policy=replace(scenario.placement.kv_policy, offload_component="hbf0", preemption_mode="swap")))
    plan = compile_serving_plan(_solve(scenario, linear_state_target="hbm0", linear_state_offload_target="hbf0"))
    physical = _PhysicalCapacityLedger(_physical_runtime_limits(plan))
    ledger = _LinearStateLedger(plan.linear_state_policy, physical)
    request = SimpleNamespace(linear_state_resident=False, linear_state_swapped_bytes=0)
    assert ledger.allocate(request)
    assert ledger.offload(request)
    assert not request.linear_state_resident
    assert physical.used_bytes["hbf0"] == plan.linear_state_policy.bytes_per_request
    assert ledger.restore(request)
    assert request.linear_state_resident and physical.used_bytes["hbf0"] == 0
    ledger.release(request)
    assert all(n == 0 for n in physical.used_bytes.values())
    assert ledger.offload_events == ledger.restore_events == 1


@pytest.mark.parametrize("change", ["zero", "small", "read_only", "missing", "same", "no_state", "partition", "no_route"])
def test_linear_state_offload_constraints_fail_closed(change):
    scenario = _scenario(mode="remote_flash", hybrid=change != "no_state")
    options = {"linear_state_target": "hbm0", "linear_state_offload_target": "hbf0"}
    if change in ("zero", "small", "read_only"):
        changes = ({"capacity_bytes": 0 if change == "zero" else 1} if change != "read_only"
                   else {"metadata": {"read_only": True}})
        scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
            replace(c, **changes) if c.component_id == "hbf0" else c for c in scenario.hardware.components)))
    elif change == "missing":
        options["linear_state_offload_target"] = "missing"
    elif change == "same":
        options["linear_state_offload_target"] = "hbm0"
    elif change == "partition":
        options["linear_state_layer_targets"] = {"linear1": "hbm1"}
    elif change == "no_route":
        scenario = replace(scenario, hardware=replace(scenario.hardware,
            links=tuple(l for l in scenario.hardware.links if "hbf0" not in (l.source_component, l.target_component))))
        decision = plan_runtime_placement(scenario, PlacementPolicy(**options))
        assert not decision.fully_placed
        assert any(r.tensor_id == "linear_state_offload" for r in decision.unplaced)
        return
    with pytest.raises(ValueError):
        plan_runtime_placement(scenario, PlacementPolicy(**options))


def test_linear_offload_policy_does_not_depend_on_current_workload():
    scenario = _scenario(mode="remote_flash", hybrid=True)
    policy = PlacementPolicy(linear_state_target="hbm0", linear_state_offload_target="hbf0")
    first = plan_runtime_placement(scenario, policy)
    changed = replace(scenario, workload=replace(scenario.workload, prompt_tokens=4096, request_count=9))
    second = plan_runtime_placement(changed, policy)
    assert first.fully_placed and second.fully_placed
    assert first.placement == second.placement


def test_linear_offload_target_requires_return_route():
    scenario = _scenario(mode="remote_flash", hybrid=True)
    scenario = replace(scenario, hardware=replace(scenario.hardware, links=tuple(
        replace(link, bidirectional=False) if link.target_component == "hbf0" else link
        for link in scenario.hardware.links)))
    decision = plan_runtime_placement(scenario, PlacementPolicy(
        linear_state_target="hbm0", linear_state_offload_target="hbf0"))
    assert not decision.fully_placed
    assert {item.tensor_id for item in decision.unplaced} >= {"linear_state", "linear_state_offload"}


@pytest.mark.parametrize("bad", [None, False, "", "stream", [], {}])
def test_linear_state_offload_mode_rejected_by_parser_and_runtime(bad):
    payload = reference_payload()
    payload["placement"]["metadata"]["linear_state_offload_mode"] = bad
    with pytest.raises(ValueError, match="linear_state_offload_mode"):
        scenario_from_dict(payload)
    scenario = _scenario(hybrid=True)
    scenario = replace(scenario, placement=replace(scenario.placement,
        metadata={**scenario.placement.metadata, "linear_state_offload_mode": bad}))
    with pytest.raises(ValueError, match="linear_state_offload_mode"):
        compile_serving_plan(scenario)


def test_pressure_offload_keeps_capacity_and_swap_but_not_per_call_mirroring():
    scenario = _scenario(mode="remote_flash", hybrid=True)
    mapped = _solve(scenario, linear_state_target="hbm0", linear_state_offload_target="hbf0")
    plans = {}
    rank = build_parallel_plan(mapped).ranks[0]
    layer = next(item.layer for item in model_graph_execution_view(mapped.model.graph).layer_instances
                 if item.layer.is_linear_attention)
    assert _linear_state_components(mapped, rank, layer=layer)[2] == 1.0
    for mode in ("mirror", "pressure"):
        configured = replace(mapped, placement=replace(mapped.placement,
            metadata={**mapped.placement.metadata, "linear_state_offload_mode": mode}))
        plans[mode] = compile_serving_plan(configured)
        active, offload, access_ratio = _linear_state_components(configured, rank, layer=layer)
        assert (active, offload) == ("hbm0", "hbf0")
        assert access_ratio == (0.0 if mode == "pressure" else 1.0)
        assert plans[mode].linear_state_policy.offload_ratio == 1.0
    assert plans["mirror"].linear_state_policy == plans["pressure"].linear_state_policy
    plan = plans["pressure"]
    ledger = _LinearStateLedger(plan.linear_state_policy, _PhysicalCapacityLedger(_physical_runtime_limits(plan)))
    request = SimpleNamespace(linear_state_resident=False, linear_state_swapped_bytes=0)
    assert ledger.allocate(request) and ledger.offload(request) and ledger.restore(request)
    assert ledger.offload_bytes == plan.linear_state_policy.bytes_per_request
    ledger.release(request)
    assert all(n == 0 for n in ledger.physical.used_bytes.values())


def test_cim_floating_contract_roundtrips_and_controls_candidates():
    from heterollm_sim.config import _cim_profile_from_dict
    from heterollm_sim.serde import to_primitive
    base = _scenario()
    profile = replace(base.cim_profile, arithmetic_mode="fp16_fp32_analytical",
        float_cycles_per_eval=2.5, float_accumulator_outputs_per_cycle=4.0,
        float_contract_basis="analytical FP16 MAC / FP32 accumulator assumption",
        supported_activation_bits=(16,), supported_weight_bits=(16,), accumulator_bits=32)
    assert _cim_profile_from_dict(to_primitive(profile)) == profile
    assert _cim_profile_from_dict(to_primitive(base.cim_profile)) == base.cim_profile
    layer = LayerSpec("float0", "dense", hidden_size=64, intermediate_size=128,
                      attention_heads=4, dtype="fp16")
    model = model_from_layer_specs("float-cim", (layer,))
    scenario = replace(base, model=model, placement=replace(base.placement,
        model_name=model.name, parallel=replace(base.placement.parallel, layer_to_stage={"float0": 0})),
        component_profiles={**base.component_profiles, "cim": {"legacy-cim": profile}})
    mapped = _solve(scenario, weight_tensor_targets={"float0.mlp_weights": "cim0"})
    assert mapped.placement.tensor_to_component["float0.mlp_weights"] == "cim0"
    integer = replace(scenario, component_profiles={**scenario.component_profiles,
        "cim": {"legacy-cim": replace(base.cim_profile, accumulator_bits=64)}})
    rejected = plan_runtime_placement(integer, PlacementPolicy(weight_tensor_targets={"float0.mlp_weights": "cim0"}))
    assert not rejected.fully_placed  # Wider integer accumulation does not establish FP16 capability.


def test_cim_candidate_rejects_packed_float_weight_without_conversion():
    from heterollm_sim.control_plane_planner import _derive_requirements, _cim_rank_padded_weight_bytes, _requirement_execution_ranks
    base = _scenario()
    layer = LayerSpec("float0", "dense", hidden_size=64, intermediate_size=128,
                      attention_heads=4, dtype="fp16")
    model = model_from_layer_specs("float-cim", (layer,))
    profile = replace(base.cim_profile, arithmetic_mode="fp16_fp32_analytical",
        float_cycles_per_eval=2.0, float_accumulator_outputs_per_cycle=4.0,
        float_contract_basis="analytical assumption", supported_activation_bits=(16,),
        supported_weight_bits=(16,), accumulator_bits=32)
    scenario = replace(base, model=model, placement=replace(base.placement,
        model_name=model.name, parallel=replace(base.placement.parallel, layer_to_stage={"float0": 0})),
        component_profiles={**base.component_profiles, "cim": {"legacy-cim": profile}})
    requirement = next(r for r in _derive_requirements(scenario) if r.tensor_id == "float0.mlp_weights")
    rank = _requirement_execution_ranks(scenario, requirement, (scenario.hardware.get_component("gpu0"),))[0]
    for fmt in ("IQ3_S", "IQ4_XS"):
        packed = replace(requirement, matrices=tuple(replace(m, packed_weight_formats=(fmt,)) for m in requirement.matrices))
        with pytest.raises(ValueError, match="conversion"):
            _cim_rank_padded_weight_bytes(scenario, packed, rank, "cim0")
