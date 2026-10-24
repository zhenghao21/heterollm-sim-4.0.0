"""Serving side of EP attention data parallelism.

A request is owned by one EP group for its whole life: seeds round-robin,
later requests go to the least-loaded group when their KV is first placed,
slots and the import pool are split per group, the KV ledger charges only the
owner group's components, and KV capacity excludes resident weight shards.
Also covers the incremental scheduler trace and the reference-counted release
of request-owned physical buffers.
"""
from dataclasses import replace
from functools import lru_cache
from random import Random

import pytest

import native_toy
from heterollm_sim import serving
from heterollm_sim.control_plane import bootstrap_control_plane
from heterollm_sim.control_plane_state import control_plane_decision
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.llama_trace_diff import diff_scheduler_traces
from heterollm_sim.serving import (BatchCost, RequestStatus, ServingRequest, _KVLedger,
    _MutableRequest, _OnlineRuntime, _PhysicalCapacityLedger, compile_serving_plan)


@lru_cache(maxsize=None)
def _scenario(ep=2, tp=1, batch=4, **ingress):
    """Bootstrapped Maverick-pattern toy (16 requests, the first ``batch`` are seeds)."""
    scenario = native_toy.maverick_pattern_scenario(layer_count=4, batch_size=batch, tp=tp, ep=ep)
    scenario = bootstrap_control_plane(scenario, capture_physical_details=False).scenario
    if ingress:
        metadata = scenario.workload.metadata
        scenario = replace(scenario, workload=replace(scenario.workload, metadata={
            **metadata, "external_kv_ingress": {**metadata["external_kv_ingress"], **ingress}}))
    return scenario


def _runtime(scenario, plan=None):
    """Runtime with a fixed cohort cost; imports have no physical destinations."""
    runtime = _OnlineRuntime(plan or compile_serving_plan(scenario),
                             lambda _scenario, _cohort: BatchCost(1000.0, {}))
    runtime._external_destination_ranges = lambda state: []
    return runtime


def _admitted_seeds(runtime):
    runtime._arrive()
    runtime._preload_warmup_seed_kv()
    runtime._fill_slots()
    return [state for state in runtime._state_values if state.spec.metadata.get("warmup_seed")]


def _by_id(runtime, index):
    return runtime.states["native-request-{:08d}".format(index)]


# ------------------------------------------------------------ plan contract

def test_one_ep_group_keeps_a_single_pool_and_ledger():
    plan = compile_serving_plan(_scenario(ep=1, tp=2))
    policy = plan.kv_policy
    assert plan.ep_degree == 1 and len(policy.ep_group_components) == 1
    assert sum(policy.component_bytes_per_page.values()) == policy.bytes_per_page
    runtime = _runtime(_scenario(ep=1, tp=2), plan)
    assert runtime._free_slot_ids_by_group == [[0, 1, 2, 3]]
    assert runtime.ledger.groups == 1 and runtime.ledger._group_page_bytes is None


def test_kv_policy_describes_one_group_and_lists_every_group_component():
    policy = compile_serving_plan(_scenario(ep=2, tp=2)).kv_policy
    assert policy.ep_degree == 2
    assert policy.ep_group_components == (("hbf0", "hbf1"), ("hbf2", "hbf3"))
    assert set(policy.component_bytes_per_page) == {"hbf0", "hbf1", "hbf2", "hbf3"}
    assert policy.bytes_per_page == sum(
        policy.component_bytes_per_page[component] for component in policy.ep_group_components[0])
    assert sum(policy.component_bytes_per_page.values()) == 2 * policy.bytes_per_page
    for layer, groups in policy.kv_layer_group_shards.items():
        assert groups == (("hbf0", "hbf1"), ("hbf2", "hbf3"))
        assert policy.kv_layer_component_shards[layer] == ("hbf0", "hbf1", "hbf2", "hbf3")


@pytest.mark.parametrize("field, value, message", [
    ("max_num_seqs", 3, "max_num_seqs 3 must be divisible by ep_degree 2"),
    ("max_inflight_requests", 3, "max_inflight_requests 3 must be divisible by ep_degree 2"),
])
def test_plan_fails_closed_when_pools_do_not_split_over_groups(field, value, message):
    scenario = _scenario(ep=2)
    if field == "max_num_seqs":
        scenario = replace(scenario, workload=replace(scenario.workload,
            scheduler=replace(scenario.workload.scheduler, max_num_seqs=value)))
    else:
        metadata = scenario.workload.metadata
        scenario = replace(scenario, workload=replace(scenario.workload, metadata={
            **metadata, "external_kv_ingress": {**metadata["external_kv_ingress"], field: value}}))
    with pytest.raises(ValueError, match=message):
        compile_serving_plan(scenario)


def test_placement_validation_workload_does_not_reject_a_legal_ep_scenario():
    """The control-plane pass uses a one-sequence placeholder scheduler."""
    scenario = _scenario(ep=2)
    probe = replace(scenario, workload=replace(
        scenario.workload, requests=(), request_count=1, prompt_tokens=1, output_tokens=0,
        arrival_rate_rps=0.0,
        metadata={**dict(scenario.workload.metadata),
                  "_control_plane_placement_validation": True},
        scheduler=replace(scenario.workload.scheduler, max_num_seqs=1)))
    assert compile_serving_plan(probe).ep_degree == 2
    # The real workload still fails closed on a pool that does not split.
    real = replace(probe, workload=replace(
        probe.workload, metadata=scenario.workload.metadata))
    with pytest.raises(ValueError, match="max_num_seqs 1 must be divisible"):
        compile_serving_plan(real)


# ---------------------------------------------------- ownership and slots

def test_seeds_round_robin_and_run_only_in_their_group_slots():
    runtime = _runtime(_scenario(ep=2))
    assert runtime._free_slot_ids_by_group == [[0, 2], [1, 3]]
    runtime._arrive()
    runtime._preload_warmup_seed_kv()
    seeds = [state for state in runtime._state_values if state.spec.metadata.get("warmup_seed")]
    assert [state.ep_group for state in seeds] == [0, 1, 0, 1]
    assert runtime._ep_load == [2, 2]
    # Preloaded seeds fill both groups' import pools (cap 4 // 2 per group).
    runtime._start_external_imports()
    assert runtime._ep_load == [2, 2]
    assert all(state.ep_group is None for state in runtime._state_values[4:])
    runtime._fill_slots()
    assert [(state.status, state.slot_id) for state in seeds] == [
        (RequestStatus.RUNNING, 0), (RequestStatus.RUNNING, 1),
        (RequestStatus.RUNNING, 2), (RequestStatus.RUNNING, 3)]
    assert runtime._free_slot_ids_by_group == [[], []]


def test_imports_pick_the_least_loaded_group_within_its_pool_and_refill_the_freed_group():
    runtime = _runtime(_scenario(ep=2))
    seeds = _admitted_seeds(runtime)
    runtime._start_external_imports()
    assert [_by_id(runtime, index).ep_group for index in range(4, 9)] == [0, 1, 0, 1, None]
    assert runtime._ep_load == [4, 4]
    assert runtime.ledger.group_used_pages[0] == runtime.ledger.group_used_pages[1] > 0

    # Seed 1 leaves group 1: its slot returns to group 1 and only a group-1
    # request may take it, although group-0 request 4 ranks first.
    finished = seeds[1]
    runtime.ledger.resize(finished, 0)
    runtime._set_status(finished, RequestStatus.FINISHED)
    assert finished.ep_group == 1 and runtime._ep_load == [4, 3]
    assert runtime._free_slot_ids_by_group == [[], [1]]
    runtime._fill_slots()
    assert _by_id(runtime, 4).status is RequestStatus.WAITING
    refill = _by_id(runtime, 5)
    assert (refill.status, refill.slot_id, refill.ep_group) == (RequestStatus.RUNNING, 1, 1)

    # Group 0 still holds two ready imports, so only group 1 has pool room.
    runtime._start_external_imports()
    assert _by_id(runtime, 8).ep_group == 1 and _by_id(runtime, 9).ep_group is None
    assert runtime._ep_load == [4, 4]


def test_admission_without_import_places_the_request_in_the_least_loaded_free_group():
    runtime = _runtime(_scenario(ep=2))
    seeds = _admitted_seeds(runtime)
    for seed in (seeds[0], seeds[2]):
        runtime.ledger.resize(seed, 0)
        runtime._set_status(seed, RequestStatus.FINISHED)
    state = _by_id(runtime, 4)
    assert state.ep_group is None
    assert runtime._assign_slot(state) == 0
    assert state.ep_group == 0 and runtime._ep_load == [1, 2]
    other = _by_id(runtime, 5)
    runtime._assign_ep_group(other, 1)
    with pytest.raises(RuntimeError, match="slot pool is exhausted"):
        runtime._assign_slot(other)


def test_cancelled_imports_return_their_group_assignment():
    runtime = _runtime(_scenario(ep=2))
    _admitted_seeds(runtime)
    runtime._start_external_imports()
    assert runtime._ep_load == [4, 4]
    runtime._cancel_external_imports()
    assert runtime._ep_load == [2, 2]
    assert all(state.ep_group is None for state in runtime._state_values[4:])
    assert runtime.ledger.group_used_pages == [
        sum(state.kv_pages for state in runtime._state_values if state.ep_group == group)
        for group in (0, 1)]


def test_external_destination_ranges_place_kv_only_on_the_owner_group():
    scenario = _scenario(ep=2, tp=2)
    runtime = _OnlineRuntime(compile_serving_plan(scenario), lambda _s, _c: BatchCost(1000.0, {}))
    state = runtime._state_values[0]
    runtime._assign_ep_group(state, 1)
    destinations = runtime._external_destination_ranges(state)
    assert {row["component_id"] for row in destinations} == {"hbf2", "hbf3"}
    owned = runtime._execution_resource_kernel._request_physical_buffers[state.spec.request_id]
    assert owned and {owner for owner, _buffer, _generation in owned} <= {
        service.physical_owner for component, service in runtime._external_executor.services.items()
        if component in {"hbf2", "hbf3"}}


def test_inventory_receives_the_owner_group_and_a_failed_import_returns_it(monkeypatch):
    from heterollm_sim import planner
    calls = []

    def inventory(scenario, request_id, history, *, ep_group, execution_control=None):
        calls.append((request_id, ep_group))
        raise RuntimeError("injected inventory failure")

    monkeypatch.setattr(planner, "native_kv_ingress_inventory", inventory)
    runtime = _OnlineRuntime(compile_serving_plan(_scenario(ep=2)), lambda _s, _c: BatchCost(1000.0, {}))
    runtime._arrive()
    with pytest.raises(RuntimeError, match="injected inventory failure"):
        runtime._preload_warmup_seed_kv()
    assert calls == [("native-request-00000000", 0)]
    assert runtime._state_values[0].ep_group is None and runtime._ep_load == [0, 0]
    assert runtime.ledger.used_pages == 0 and runtime.ledger.group_used_pages == [0, 0]


# ------------------------------------------------- scheduling per group

def test_priority_preemption_only_evicts_the_waiter_group(monkeypatch):
    scenario = _scenario(ep=2)
    plan = compile_serving_plan(scenario)
    plan = replace(plan, scheduler=replace(plan.scheduler, preemption_enabled=True))
    runtime = _runtime(scenario, plan)
    _admitted_seeds(runtime)
    victims = []
    monkeypatch.setattr(runtime, "_preempt", lambda victim, reason: victims.append(victim) or True)
    waiter = _by_id(runtime, 4)
    waiter.spec = replace(waiter.spec, priority=10)
    runtime._assign_ep_group(waiter, 0)
    runtime._priority_preempt()
    # Seed 3 ranks last overall, but only a group-0 slot can serve the waiter.
    assert [victim.spec.request_id for victim in victims] == ["native-request-00000002"]

    victims.clear()
    seed = _by_id(runtime, 0)
    runtime.ledger.resize(seed, 0)
    runtime._set_status(seed, RequestStatus.FINISHED)
    runtime._priority_preempt()
    assert victims == []


def test_full_batch_gate_waits_only_for_requests_a_free_slot_can_take():
    scenario = _scenario(ep=2, require_full_batch=True)
    runtime = _runtime(scenario)
    seeds = _admitted_seeds(runtime)
    assert runtime._select_cohort() is not None
    runtime.ledger.resize(seeds[1], 0)
    runtime._set_status(seeds[1], RequestStatus.FINISHED)
    waiting = runtime._state_values[4:]
    # An unplaced waiter could still take group 1's free slot.
    assert runtime._select_cohort() is None
    for state in waiting:
        runtime._assign_ep_group(state, 0)
    assert runtime._select_cohort() is not None
    runtime._release_ep_group(waiting[-1])
    runtime._assign_ep_group(waiting[-1], 1)
    assert runtime._select_cohort() is None


def test_failed_execution_restores_group_slot_pools_and_loads():
    runtime = _runtime(_scenario(ep=2))
    _admitted_seeds(runtime)
    runtime._start_external_imports()
    cohort = runtime._select_cohort()
    before = ([list(slots) for slots in runtime._free_slot_ids_by_group],
              dict(runtime._slot_bindings), list(runtime._ep_load),
              list(runtime.ledger.group_used_pages))
    victim = runtime.states[cohort.items[0].request_id]

    def broken(_cohort):
        runtime.ledger.resize(victim, 0)
        runtime._set_status(victim, RequestStatus.FINISHED)
        raise RuntimeError("injected execution failure")

    runtime._execute_cohort = broken
    with pytest.raises(RuntimeError, match="injected execution failure"):
        runtime._execute(cohort)
    after = ([list(slots) for slots in runtime._free_slot_ids_by_group],
             dict(runtime._slot_bindings), list(runtime._ep_load),
             list(runtime.ledger.group_used_pages))
    assert after == before and victim.status is RequestStatus.RUNNING
    states = runtime._schedule_trace[-1]["states"]
    assert next(row for row in states if row["request_id"] == victim.spec.request_id)["status"] == "running"


def test_full_run_keeps_each_request_in_one_group_and_balances_cohorts():
    runtime = _runtime(_scenario(ep=2))
    result = runtime.run()
    groups = {request_id: state.ep_group for request_id, state in result.request_states.items()}
    assert all(group in (0, 1) for request_id, group in groups.items()
               if result.request_states[request_id].status is not RequestStatus.WAITING)
    assert result.batches
    for batch in result.batches:
        assert all(item.ep_group == groups[item.request_id] for item in batch.items)
        assert [sum(item.ep_group == group for item in batch.items) for group in (0, 1)] == [2, 2]
    seen = {}
    for row in result.schedule_trace:
        for item in row.get("logical_batch", {}).get("rows", ()):
            # A row recorded after its request finished reports no slot.
            assert item["slot_id"] is None or item["slot_id"] % 2 == item["ep_group"]
        for state in row["states"]:
            if state["ep_group"] is not None and state["status"] != "waiting":
                assert seen.setdefault(state["request_id"], state["ep_group"]) == state["ep_group"]
    kv = result.kv_metrics
    assert kv.ep_degree == 2 and len(kv.peak_used_pages_by_group) == 2
    assert kv.peak_used_pages == max(kv.peak_used_pages_by_group) > 0


# ------------------------------------------------------------- KV ledger

def test_ledger_charges_only_the_owner_group_against_a_per_group_page_budget():
    policy = replace(compile_serving_plan(_scenario(ep=2, tp=2)).kv_policy, capacity_pages=3)
    physical = _PhysicalCapacityLedger({component: 10**15 for component in policy.component_bytes_per_page})
    ledger = _KVLedger(policy, physical)

    def request(name, group):
        state = _MutableRequest(ServingRequest(name, 0.0, 8, 4))
        state.ep_group = group
        return state

    first, second, other = request("a", 0), request("b", 0), request("c", 1)
    assert ledger.resize(first, 3)
    page = policy.component_bytes_per_page
    assert physical.used_bytes == {"hbf0": 3 * page["hbf0"], "hbf1": 3 * page["hbf1"], "hbf2": 0, "hbf3": 0}
    assert not ledger.can_resize(second, 1)
    assert ledger.resize(other, 3)
    assert ledger.group_used_pages == [3, 3] and ledger.used_pages == 6
    assert ledger.resize(first, 1)
    assert ledger.group_used_pages == [1, 3] and ledger.group_peak_pages == [3, 3]
    assert ledger.peak_group_pages == 3 and ledger.max_group_used_pages == 3
    assert physical.used_bytes["hbf0"] == page["hbf0"] and physical.used_bytes["hbf2"] == 3 * page["hbf2"]

    unplaced = _MutableRequest(ServingRequest("u", 0.0, 8, 4))
    assert ledger.can_resize(unplaced, 0) and ledger.resize(unplaced, 0)
    with pytest.raises(RuntimeError, match="no EP group"):
        ledger.can_resize(unplaced, 1)


@pytest.mark.parametrize("ep", (1, 2))
def test_kv_capacity_excludes_resident_rank_weight_shards(monkeypatch, ep):
    scenario = _scenario(ep=ep, tp=2)
    resident = {}
    for entries in control_plane_decision(scenario)["rank_weight_shards"].values():
        for entry in entries:
            component = entry["storage_component_id"]
            resident[component] = resident.get(component, 0) + entry["physical_bytes"]
    after = compile_serving_plan(scenario).kv_policy.capacity_bytes_by_component
    monkeypatch.setattr(serving, "_resident_rank_weight_bytes", lambda *_args: {})
    before = compile_serving_plan(scenario).kv_policy.capacity_bytes_by_component
    assert set(after) == {f"hbf{index}" for index in range(2 * ep)}
    for component, capacity in after.items():
        assert resident[component] > 0
        assert before[component] - capacity == resident[component]
    assert len(set(after.values())) == 1


def test_maverick_tp2_ep4_hbm_kv_capacity_is_component_minus_resident_weights():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario

    scenario = build_native_comparison_scenario("llama4_maverick", "HBM4", "SHORT",
                                                batch_size=8, tp=2, pp=1, ep=4)
    scenario = bootstrap_control_plane(scenario, capture_physical_details=False).scenario
    resident = sum(entry["physical_bytes"]
                   for entries in control_plane_decision(scenario)["rank_weight_shards"].values()
                   for entry in entries if entry["storage_component_id"] == "hbm0")
    policy = compile_serving_plan(scenario).kv_policy
    capacity = scenario.hardware.get_component("hbm0").capacity_bytes
    assert capacity == 288 * 10**9 and resident == pytest.approx(110.8e9, abs=0.1e9)
    assert policy.capacity_bytes_by_component["hbm0"] == capacity - resident
    assert policy.ep_group_components == (("hbm0", "hbm1"), ("hbm2", "hbm3"),
                                          ("hbm4", "hbm5"), ("hbm6", "hbm7"))


# -------------------------------------------------------- scheduler trace

def test_incremental_trace_rebuilds_the_full_per_row_states_with_far_fewer_records():
    runtime = _runtime(_scenario(ep=2))
    full_rows = []
    record = runtime._record_schedule_trace

    def recording(*args, **kwargs):
        record(*args, **kwargs)
        full_rows.append(tuple(runtime._state_trace(state) for state in
                               sorted(runtime._state_values, key=lambda state: state.spec.request_id)))

    runtime._record_schedule_trace = recording
    result = runtime.run()
    rows = list(result.schedule_trace)
    assert len(rows) == len(full_rows) == len(result.schedule_trace) > 10
    assert [row["states"] for row in rows] == full_rows
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert result.schedule_trace[5] == rows[5] and result.schedule_trace[-1] == rows[-1]
    assert result.schedule_trace[2:4] == tuple(rows[2:4])
    assert result.schedule_trace == rows
    assert diff_scheduler_traces({"schedule_trace": result.schedule_trace}, rows)["equal"]
    stored = sum(len(changed) for changed in result.schedule_trace._deltas)
    assert stored * 3 < len(rows) * len(runtime.states)


def test_trace_rechecks_states_rewritten_behind_the_change_hook():
    runtime = _runtime(_scenario(ep=2))
    runtime._record_schedule_trace("probe")
    state = runtime._state_values[3]
    state.__dict__["committed"] = 2
    runtime._record_schedule_trace("probe")
    assert runtime._schedule_trace[-1]["states"][3]["committed_tokens"] == 0
    runtime._mark_all_trace_states_pending()
    runtime._record_schedule_trace("probe")
    assert runtime._schedule_trace[-1]["states"][3]["committed_tokens"] == 2
    assert runtime._schedule_trace._deltas[-1] == ((state.spec.request_id, runtime._state_trace(state)),)


# ------------------------------------------------- request buffer release

def test_release_counts_holders_instead_of_rescanning_every_request(monkeypatch):
    rng = Random(7)
    kernel = UnifiedEventKernel()
    monkeypatch.setattr(kernel, "_reclaim_physical_allocations", lambda _arrival: None)
    keys = [("owner", "buffer-{}".format(index), 0) for index in range(12)]
    owned = {"r{}".format(index): set(rng.sample(keys, rng.randrange(1, 5))) for index in range(9)}
    for request_id, held in owned.items():
        kernel.register_request_physical_buffers(request_id, held)
        kernel.register_request_physical_buffers(request_id, held)
    live = dict(owned)
    for request_id in rng.sample(sorted(owned), len(owned)):
        expected = live.pop(request_id) - set().union(*live.values())
        kernel._pending_physical_releases.clear()
        kernel.release_request_physical_allocations(request_id)
        assert set(kernel._pending_physical_releases) == expected
    assert not kernel._request_buffer_holders and not kernel._request_physical_buffers


def test_forgetting_a_failed_registration_releases_nothing(monkeypatch):
    kernel = UnifiedEventKernel()
    monkeypatch.setattr(kernel, "_reclaim_physical_allocations", lambda _arrival: None)
    shared, private = ("owner", "shared", 0), ("owner", "private", 0)
    kernel.register_request_physical_buffers("a", (shared,))
    kernel.register_request_physical_buffers("b", (shared, private))
    kernel.forget_request_physical_buffers("b")
    assert kernel._request_buffer_holders == {shared: 1} and not kernel._pending_physical_releases
    kernel.release_request_physical_allocations("a")
    assert set(kernel._pending_physical_releases) == {shared}
