"""Preloaded native admission preserves order without repeatedly sorting seeds."""
from dataclasses import replace

import pytest

from heterollm_sim.serving import RequestStatus, compile_serving_plan
from test_native_ep_serving import _runtime, _scenario


def prepared_runtime(*, ep=2, batch=4, allocation="eager", capacity_pages=None):
    scenario = _scenario(ep=ep, batch=batch)
    plan = compile_serving_plan(scenario)
    policy = replace(plan.kv_policy, allocation_policy=allocation)
    if capacity_pages is not None:
        policy = replace(policy, capacity_pages=capacity_pages)
    runtime = _runtime(scenario, replace(plan, kv_policy=policy))
    runtime._arrive()
    runtime._preload_warmup_seed_kv()
    return runtime


def request_ids(*indexes):
    return [f"native-request-{index:08d}" for index in indexes]


def admission_events(runtime):
    return [event for event in runtime.events
            if event.event_type in ("request_admitted", "request_resumed")]


def kv_accounting(runtime):
    return (runtime.ledger.used_pages, runtime.ledger.persistent_used_pages,
            runtime.ledger.allocations, runtime.ledger.releases,
            tuple(runtime.ledger.group_used_pages),
            tuple(runtime.physical_ledger.used_bytes.items()))


@pytest.mark.parametrize("allocation", ["lazy", "eager"])
@pytest.mark.parametrize("ep", [1, 2])
def test_preloaded_admission_preserves_priority_deadline_ep_slots_and_capacity(allocation, ep):
    runtime = prepared_runtime(ep=ep, allocation=allocation)
    seeds = runtime._state_values[:4]
    for state, priority, deadline in zip(seeds, [1, 0, 1, 0], [20., 5., 10., 2.]):
        state.spec = replace(state.spec, priority=priority, deadline_ns=deadline)
    before = kv_accounting(runtime)

    runtime._fill_slots()

    expected = request_ids(2, 0, 3, 1)
    assert [event.request_id for event in admission_events(runtime)] == expected
    transitions = [row["request_id"] for row in runtime._schedule_trace
                   if row["kind"] == "state_transition" and row["details"]["to"] == "running"]
    assert transitions == expected
    assert [runtime.states[key].admission_sequence for key in expected] == [0, 1, 2, 3]
    assert [runtime.states[key].slot_id for key in expected] == ([0, 1, 2, 3] if ep == 1 else [0, 2, 1, 3])
    assert runtime._running_count == 4
    assert runtime._free_slot_ids_by_group == [[] for _ in range(ep)]
    assert kv_accounting(runtime) == before
    assert all(state.status is RequestStatus.WAITING for state in runtime._state_values[4:])


@pytest.mark.parametrize("batch", [4, 16, 64])
@pytest.mark.parametrize("allocation", ["lazy", "eager"])
def test_successful_preloaded_batch_ranks_at_most_once_per_candidate(monkeypatch, batch, allocation):
    runtime = prepared_runtime(batch=batch, allocation=allocation)
    rank = runtime._service_rank
    count = 0

    def counted(state):
        nonlocal count
        count += 1
        return rank(state)

    monkeypatch.setattr(runtime, "_service_rank", counted)
    runtime._fill_slots()

    assert runtime._running_count == batch
    assert len(admission_events(runtime)) == batch
    assert count <= len(runtime._state_values)
    # Full groups make a repeated boundary call a no-op, with no stale cache.
    before = count
    runtime._fill_slots()
    assert count == before
    assert len(admission_events(runtime)) == batch


@pytest.mark.parametrize("all_later_fail", [False, True])
def test_failed_candidate_waits_for_later_success_before_retry(monkeypatch, all_later_fail):
    runtime = prepared_runtime()
    admit = runtime._admit_candidate
    attempts = []
    first, second = request_ids(0, 1)

    def controlled(state):
        key = state.spec.request_id
        attempts.append(key)
        if all_later_fail and key != first:
            return False
        if key == second and attempts.count(second) == 1:
            return False
        return admit(state)

    monkeypatch.setattr(runtime, "_admit_candidate", controlled)
    runtime._fill_slots()

    assert attempts == request_ids(*(range(4) if all_later_fail else [0, 1, 2, 1, 3]))
    expected = request_ids(0) if all_later_fail else request_ids(0, 2, 1, 3)
    assert [event.request_id for event in admission_events(runtime)] == expected
    assert runtime._running_count == len(expected)


@pytest.mark.parametrize("advance_during_swap", [False, True])
def test_time_advance_recomputes_priority_aging(monkeypatch, advance_during_swap):
    runtime = prepared_runtime()
    runtime.plan = replace(runtime.plan, scheduler=replace(runtime.plan.scheduler,
        policy="decode_first_aging", starvation_ns=10.))
    first = runtime._state_values[0]
    first.spec = replace(first.spec, priority=100)
    runtime._state_values[2].queued_since_ns = 5.
    if advance_during_swap:
        runtime._set_status(first, RequestStatus.SWAPPED)
        first.preemption_strategy = "swap"

        def resume(state):
            assert state is first
            runtime.now += 12.5
            return True

        monkeypatch.setattr(runtime, "_resume_swap", resume)
    else:
        admit = runtime._admit_candidate

        def admit_and_advance(state):
            accepted = admit(state)
            if state is first:
                runtime.now += 12.5
            return accepted

        monkeypatch.setattr(runtime, "_admit_candidate", admit_and_advance)

    runtime._fill_slots()

    # At time 0, IDs would order 1,2,3. At 12.5, only 1 and 3 gained aging.
    events = admission_events(runtime)
    assert [event.request_id for event in events] == request_ids(0, 1, 3, 2)
    assert runtime.now.hex() == (12.5).hex()
    assert [event.timestamp_ns.hex() for event in events[1:]] == [(12.5).hex()] * 3


def test_preemption_adds_candidate_and_returns_its_ep_slot(monkeypatch):
    runtime = prepared_runtime()
    runtime.plan = replace(runtime.plan, scheduler=replace(runtime.plan.scheduler, preemption_enabled=True))
    victim = runtime._state_values[0]
    victim.spec = replace(victim.spec, priority=100)
    assert runtime._admit_candidate(victim)
    admit = runtime._admit_candidate
    attempts = []

    def admit_after_pressure(state):
        attempts.append(state.spec.request_id)
        if len(attempts) == 1:
            # Exercise the candidate/slot mutation of pressure preemption
            # without launching a transfer or a compute cohort.
            runtime._set_status(victim, RequestStatus.SWAPPED)
            victim.preemption_strategy = "recompute"
        return admit(state)

    monkeypatch.setattr(runtime, "_admit_candidate", admit_after_pressure)
    runtime._fill_slots()

    assert attempts == request_ids(1, 0, 2, 3)
    assert [(event.request_id, event.event_type) for event in admission_events(runtime)[1:]] == [
        (request_ids(1)[0], "request_admitted"), (request_ids(0)[0], "request_resumed"),
        (request_ids(2)[0], "request_admitted"), (request_ids(3)[0], "request_admitted")]
    assert victim.slot_id == 0 and victim.admission_sequence == 0
    assert runtime._free_slot_ids_by_group == [[], []]


def test_prompt_save_time_advance_returns_to_arrival_boundary(monkeypatch):
    runtime = prepared_runtime()
    runtime.prompt_cache.policy = replace(runtime.prompt_cache.policy, enabled=True)
    before = len(runtime.events)

    def save():
        runtime.now += 7.25

    def unexpected(_state):
        pytest.fail("admission must wait for the boundary to process arrivals during the save")

    monkeypatch.setattr(runtime, "_save_pending_prompt_cache_states", save)
    monkeypatch.setattr(runtime, "_admit_candidate", unexpected)
    runtime._fill_slots()

    assert runtime.now.hex() == (7.25).hex()
    assert runtime._running_count == 0
    assert len(runtime.events) == before
    assert runtime._free_slot_ids_by_group == [[0, 2], [1, 3]]


def test_zero_preload_capacity_does_not_admit_uninitialized_requests(monkeypatch):
    runtime = prepared_runtime(capacity_pages=0)
    before = kv_accounting(runtime)

    def unexpected(_state):
        pytest.fail("uninitialized requests must not be admitted")

    monkeypatch.setattr(runtime, "_admit_candidate", unexpected)
    runtime._fill_slots()

    assert runtime._external_preloaded_requests == 0
    assert not runtime._external_initialized
    assert runtime._running_count == 0 and runtime.ledger.used_pages == 0
    assert not admission_events(runtime)
    assert runtime._free_slot_ids_by_group == [[0, 2], [1, 3]]
    assert runtime._ep_load == [0, 0]
    assert kv_accounting(runtime) == before
