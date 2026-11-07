"""Seed capacity is rejected before the first cohort is lowered."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from heterollm_sim import native_stationary as stationary, serving
from heterollm_sim.native_layer_sampling import NativeLayerSamplingGuardError
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
import native_toy


def runtime(loaded=(0, 1, 2), degree=2):
    seeds = [NS(spec=NS(request_id=f"s{index}", metadata={"warmup_seed": True},
                       arrival_ns=0., output_tokens=8-index), ep_group=index % degree,
                status=serving.RequestStatus.WAITING)
             for index in range(4)]
    fresh = [NS(spec=NS(request_id=f"f{index}", metadata={}, arrival_ns=0., output_tokens=8),
                ep_group=None) for index in range(2)]
    for state in fresh:
        state.spec.prompt_tokens = 40
    return NS(plan=NS(scheduler=NS(max_num_seqs=4)), now=10., _ep_degree=degree,
              _state_values=tuple(seeds + fresh), _kv_block_layout=None,
              _external_ingress={"preload_warmup_seed_kv": True, "max_inflight_requests": 2},
              _external_preloaded_requests=len(loaded),
              _external_initialized={f"s{index}" for index in loaded} | {"f0"})


@pytest.mark.parametrize("loaded", [(), (0,), (0, 1), (0, 1, 2)])
def test_short_preloaded_batch_is_a_capacity_rejection(loaded):
    active, diagnostics = runtime(loaded), {}
    with pytest.raises(NativeLayerSamplingGuardError, match="first_cohort_not_a_full_batch"):
        stationary._preloaded_seed_admission(active, diagnostics)
    assert diagnostics == {"physical_admission_cutoff_ns": 10.}


@pytest.mark.parametrize("pool_blocks,reason", [(8, "kv_block_peak_exceeds_capacity"),
                                                (9, "first_cohort_not_a_full_batch")])
def test_a_refill_cycle_beyond_the_pools_is_rejected_before_any_seed_check(pool_blocks, reason):
    # Seeds at ages 0, 2 and 1, 3 of 8 in two groups, prompt 40, 16-token
    # blocks: every slot holds 3 blocks at every age, plus one prepared
    # import's 3 history blocks per group.
    active, diagnostics = runtime(degree=2), {}
    active._kv_block_layout = {"tokens_per_block": 16, "pool_blocks": {"full": pool_blocks}, "ring": 0}
    peaks = stationary.kv_block_peak([[0, 2], [1, 3]], 40, 8, 1, 16)
    assert peaks == {"full": [9, 9]}
    with pytest.raises(NativeLayerSamplingGuardError, match=reason):
        stationary._preloaded_seed_admission(active, diagnostics)
    assert diagnostics["kv_block_peak_by_group"] == peaks
    assert diagnostics["kv_block_capacity"] == {"full": pool_blocks}


@pytest.mark.parametrize("contract", ["not_preloaded", "seed_count", "fresh_lengths", "no_fresh", "future_seed"])
def test_other_admission_contracts_are_not_reclassified_as_capacity(contract):
    active, diagnostics = runtime(), {"untouched": True}
    if contract == "not_preloaded":
        active._external_ingress["preload_warmup_seed_kv"] = False
    elif contract == "seed_count":
        active.plan.scheduler.max_num_seqs = 5
    elif contract == "fresh_lengths":
        active._state_values[-1].spec.output_tokens = 9
    elif contract == "no_fresh":
        active._state_values = active._state_values[:4]
    else:
        active._state_values[3].spec.arrival_ns = 11.
    stationary._preloaded_seed_admission(active, diagnostics)
    assert diagnostics == {"untouched": True}


def test_full_batch_and_invalid_group_assignments_keep_existing_path():
    active, diagnostics = runtime((0, 1, 2, 3)), {}
    stationary._preloaded_seed_admission(active, diagnostics)
    assert diagnostics == {}
    active = runtime()
    active._state_values[0].ep_group = None
    stationary._preloaded_seed_admission(active, diagnostics)
    assert diagnostics == {}


def test_mixed_pre_admission_logical_rejection_is_not_a_preload_capacity_cutoff():
    from heterollm_sim.control_plane import bootstrap_control_plane

    scenario = native_toy.moe_toy_scenario(8, batch_size=4, output=6, warmup=5, measurement=10, ep=2)
    scenario = bootstrap_control_plane(scenario, capture_physical_details=False).scenario
    plan = serving.compile_serving_plan(scenario)
    logical_limit = max(serving._max_live_kv_tokens(row.prompt_tokens, row.output_tokens)
                        for row in plan.requests)
    oversized = replace(plan.requests[0], prompt_tokens=logical_limit + 1)
    assert oversized.metadata["warmup_seed"]
    plan = replace(plan, requests=(oversized, *plan.requests[1:]),
                   kv_policy=replace(plan.kv_policy, logical_context_tokens=logical_limit))
    active = serving._OnlineRuntime(plan, lambda *_: serving.BatchCost(1000., {}))
    active._external_destination_ranges = lambda state: []
    active._reject_impossible_requests()
    active._arrive()
    active._preload_warmup_seed_kv()

    rejected = active.states[oversized.request_id]
    assert rejected.status == serving.RequestStatus.REJECTED
    assert "logical context" in rejected.rejection_reason
    assert active._external_preloaded_requests == 3
    diagnostics = {"untouched": True}
    stationary._preloaded_seed_admission(active, diagnostics)
    assert diagnostics == {"untouched": True}
    assert rejected.status == serving.RequestStatus.REJECTED


def boundary_runtime():
    order = []
    active = NS(now=10., _request_count=4)
    for name in ("_complete_prefetches", "_complete_external_imports", "_process_agent_events",
                 "_arrive", "_preload_warmup_seed_kv", "_priority_preempt", "_fill_slots",
                 "_start_external_imports", "_prefetch_waiting"):
        setattr(active, name, lambda key=name: order.append(key))
    return active, order


def test_optional_guard_runs_once_just_after_preload():
    active, order = boundary_runtime()
    active._native_preload_admission_guard = lambda same: order.append("guard")
    serving._OnlineRuntime._stabilize_boundary(active)
    serving._OnlineRuntime._stabilize_boundary(active)
    assert order.count("guard") == 1
    position = order.index("guard")
    assert order[position-1:position+2] == ["_preload_warmup_seed_kv", "guard", "_priority_preempt"]
    assert active._native_preload_admission_guard is None


def test_default_boundary_has_no_guard_or_extra_state():
    active, order = boundary_runtime()
    serving._OnlineRuntime._stabilize_boundary(active)
    assert not hasattr(active, "_native_preload_admission_guard")
    assert order == ["_complete_prefetches", "_complete_external_imports", "_process_agent_events",
                     "_arrive", "_preload_warmup_seed_kv", "_priority_preempt", "_fill_slots",
                     "_start_external_imports", "_fill_slots", "_prefetch_waiting"]


@pytest.mark.parametrize("loaded", [0, 3])
def test_real_native_preload_capacity_stops_before_lower_guard_imports_or_samples(monkeypatch, loaded):
    from heterollm_sim.reporting import run_scenario

    scenario = native_toy.moe_toy_scenario(8, batch_size=4, output=6, warmup=5, measurement=10, ep=2)
    seeds = [request.request_id for request in scenario.workload.requests if request.metadata.get("warmup_seed")]
    refused = set(seeds[loaded:])
    original = serving._KVLedger.can_resize

    def limited(ledger, state, pages):
        if state.spec.request_id in refused and state.kv_pages == 0 and pages > 0:
            return False
        return original(ledger, state, pages)

    def unexpected(*args, **kwargs):
        pytest.fail("capacity shortfall entered lowering, query guard, fresh imports, or samples")

    monkeypatch.setattr(serving._KVLedger, "can_resize", limited)
    monkeypatch.setattr(serving, "_lower_cost", unexpected)
    monkeypatch.setattr(serving._OnlineRuntime, "_start_external_imports", unexpected)
    monkeypatch.setattr(stationary, "prepare_stationary", unexpected)
    monkeypatch.setattr(stationary, "_query_decode_guard", unexpected)
    monkeypatch.setattr(stationary, "sample_stationary", unexpected)
    result = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(calibration_layers=4, holdout_layers=4))),
        retention_policy="aggregate")
    diagnostics = result.approximation
    assert diagnostics["rejection_reasons"] == ["native_stationary_first_cohort_not_a_full_batch"]
    assert not diagnostics["kv_block_peak_exceeds_capacity"]
    assert diagnostics["physical_ingress_accounting"]["preloaded_seed_requests"] == loaded
    assert diagnostics["physical_boundary_accounting"]["source_pending_carry_upper_ns"] is None
    assert diagnostics["physical_boundary_accounting"]["layer_composition"] == {}
    assert result.layer_window_samples == () and result.native_window_forecast is None


def test_sequential_layout_has_no_synthetic_block_open_rate():
    from heterollm_sim.kv_placement import kv_sequential_layout
    layout = kv_sequential_layout(1660, 373, 2, 16, 25, 64)
    assert layout["placement_mode"] == "sequential_channel"
