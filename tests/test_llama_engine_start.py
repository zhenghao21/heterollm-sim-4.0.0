"""llama request engine boundary is distinct from early host preparation."""
from dataclasses import replace

import pytest

from heterollm_sim import serving
from heterollm_sim.ir import RequestSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


def scenario(*, count=4, prompt=4, output=1, batch=4, chunk=4, llama=True, origin=0.0):
    base = build_reference_scenario()
    requests = tuple(RequestSpec(f"request-{i:04d}", origin, prompt, output) for i in range(count))
    workload = replace(base.workload, requests=requests,
                       scheduler=replace(base.workload.scheduler, max_num_seqs=count,
                                         max_num_batched_tokens=batch, max_num_ubatch_tokens=batch,
                                         prefill_chunk_tokens=chunk, prefill_stop_offsets=(),
                                         preemption_enabled=False, mixed_phase_batching=False),
                       mtp=replace(base.workload.mtp, method="disabled"),
                       metadata={"llama_cpp_runtime": {"batch": batch}} if llama else {})
    return replace(base, workload=workload, llama_cpp_config=None)


def staged_cost(_scenario, cohort):
    return serving.BatchCost(120.0, 0.0, {
        "host_orchestration_ns": 20.0, "device_execution_ns": 100.0,
        "execution_stages": [
            {"stage_id": "cpu", "component_id": "cpu0", "service_ns": 10.0,
             "request_ids": cohort.request_ids, "dependencies": ()},
            {"stage_id": "gpu", "component_id": "gpu0", "service_ns": 90.0,
             "request_ids": cohort.request_ids, "dependencies": ("cpu",)},
        ],
    })


def runtime(case):
    return serving._OnlineRuntime(serving.compile_serving_plan(case), staged_cost,
                                   apply_measurement_start_residency=False)


def engine_events(result):
    return [event for event in result.events if event.event_type == "engine_request_begin"]


def test_serial_prompt_cohorts_start_engine_when_selected_not_early_host():
    result = serving.simulate_online(scenario(), staged_cost)
    batches = [batch for batch in result.batches if batch.kind == "prefill"]
    assert len(batches) == 4
    assert len(engine_events(result)) == 4
    previous_end = 0.0
    for batch in batches:
        request_id = batch.request_ids[0]
        state, metric = result.request_states[request_id], result.request_metrics[request_id]
        assert state.engine_start_ns == metric.engine_start_ns == previous_end
        assert metric.engine_start_ns < metric.first_token_ns
        if previous_end > 0:
            assert batch.cost.metadata["host_start_ns"] < metric.engine_start_ns
        event = next(item for item in engine_events(result) if item.request_id == request_id)
        assert event.timestamp_ns == metric.engine_start_ns
        assert event.details["boundary"] == "first_prompt_batch_processing"
        assert event.cohort_id is None
        previous_end = batch.end_ns


def test_non_llama_timing_is_unchanged_and_engine_start_is_absent():
    generic = serving.simulate_online(scenario(llama=False), staged_cost)
    llama = serving.simulate_online(scenario(), staged_cost)
    assert generic.batches == llama.batches
    assert generic.makespan_ns == llama.makespan_ns
    assert not engine_events(generic)
    assert [e for e in llama.events if e.event_type != "engine_request_begin"] == list(generic.events)
    for key in generic.request_metrics:
        assert generic.request_metrics[key].engine_start_ns is None
        assert generic.request_states[key].engine_start_ns is None
        assert replace(llama.request_metrics[key], engine_start_ns=None) == generic.request_metrics[key]
        assert replace(llama.request_states[key], engine_start_ns=None) == generic.request_states[key]


def test_same_cohort_requests_start_together_at_arrival():
    case = scenario(count=2, prompt=2, batch=4, chunk=2, origin=75.0)
    result = serving.simulate_online(case, staged_cost)
    assert len(result.batches) == 1
    assert [event.timestamp_ns for event in engine_events(result)] == [75.0, 75.0]
    assert all(metric.engine_start_ns == 75.0 for metric in result.request_metrics.values())


def test_chunked_prompt_and_decode_keep_single_first_engine_event():
    result = serving.simulate_online(scenario(count=1, prompt=12, chunk=4, output=3), staged_cost)
    assert len([batch for batch in result.batches if batch.kind == "prefill"]) == 3
    assert len(result.batches) > 3
    assert len(engine_events(result)) == 1
    assert result.request_metrics["request-0000"].engine_start_ns == 0.0


def test_typed_llama_runtime_enables_event_without_metadata():
    case = scenario(count=1, llama=False)
    case = replace(case, llama_cpp_config=LlamaCppRuntimeConfig())
    result = serving.simulate_online(case, staged_cost)
    assert result.request_metrics["request-0000"].engine_start_ns == 0.0
    assert len(engine_events(result)) == 1


def test_selection_before_reservation_wait_keeps_wait_inside_engine(monkeypatch):
    runner = runtime(scenario(count=1))
    runner._stabilize_boundary()
    runner.now = 50.0
    state = runner.states["request-0000"]
    reserve = runner._reserve_with_pressure
    calls = []
    def pressure(owner, pages, protected):
        assert owner.engine_start_ns == 50.0
        calls.append(runner.now)
        runner.now += 100.0
        return reserve(owner, pages, protected)
    monkeypatch.setattr(runner, "_reserve_with_pressure", pressure)
    cohort = runner._prefill_cohort([state], recompute=False)
    assert cohort is not None and cohort.start_ns == 150.0
    assert state.engine_start_ns == 50.0
    runner._execute(cohort)
    result = runner._result()
    assert result.request_metrics[state.spec.request_id].engine_start_ns == 50.0
    assert len(calls) == 1
    assert len(engine_events(result)) == 1


def test_failed_reservation_retains_first_entry_for_later_retry(monkeypatch):
    runner = runtime(scenario(count=1))
    runner._stabilize_boundary()
    runner.now = 25.0
    state = runner.states["request-0000"]
    reserve = runner._reserve_with_pressure
    monkeypatch.setattr(runner, "_reserve_with_pressure", lambda *_: False)
    assert runner._prefill_cohort([state], recompute=False) is None
    assert state.engine_start_ns == 25.0
    runner.now = 225.0
    monkeypatch.setattr(runner, "_reserve_with_pressure", reserve)
    assert runner._prefill_cohort([state], recompute=False) is not None
    assert state.engine_start_ns == 25.0
    assert len(engine_events(runner)) == 1


@pytest.mark.parametrize("mode", ["token_budget", "sequence_budget", "protected", "not_running", "zero_chunk"])
def test_ineligible_prefill_candidate_does_not_start_engine(mode):
    runner = runtime(scenario(count=1))
    runner._stabilize_boundary()
    state = runner.states["request-0000"]
    kwargs = {}
    if mode == "token_budget":
        kwargs["token_budget"] = 0
    elif mode == "sequence_budget":
        kwargs["sequence_budget"] = 0
    elif mode == "protected":
        kwargs["protected_request_ids"] = [state.spec.request_id]
    elif mode == "not_running":
        runner._set_status(state, serving.RequestStatus.WAITING)
    else:
        state.prefill_cursor = state.spec.prompt_tokens
    assert runner._prefill_cohort([state], recompute=False, **kwargs) is None
    assert state.engine_start_ns is None
    assert not engine_events(runner)


def test_recompute_entry_never_synthesizes_first_prompt_start():
    runner = runtime(scenario(count=1))
    runner._stabilize_boundary()
    state = runner.states["request-0000"]
    state.recompute_target = 4
    assert runner._prefill_cohort([state], recompute=True) is not None
    assert state.engine_start_ns is None
    assert not engine_events(runner)


@pytest.mark.parametrize("policy", ["swap", "recompute"])
def test_preempt_resume_and_recompute_do_not_reset_engine_start(policy):
    runner = runtime(scenario(count=1, prompt=12, chunk=4, output=2))
    runner.plan = replace(runner.plan, scheduler=replace(runner.plan.scheduler, preemption_enabled=True, preemption_policy=policy))
    runner._stabilize_boundary()
    state = runner.states["request-0000"]
    cohort = runner._select_cohort()
    assert cohort is not None
    runner._execute(cohort)
    first_start = state.engine_start_ns
    assert first_start == 0.0 and state.prefill_cursor == 4
    assert runner._preempt(state, "synthetic_preemption") is True
    assert state.status == serving.RequestStatus.SWAPPED
    assert state.engine_start_ns == first_start
    runner._stabilize_boundary()
    assert state.status == serving.RequestStatus.RUNNING
    assert state.engine_start_ns == first_start
    resumed = runner._select_cohort()
    assert resumed is not None
    if policy == "recompute":
        assert resumed.items[0].phase == "recompute"
    runner._execute(resumed)
    assert state.engine_start_ns == first_start
    result = runner.run()
    assert result.request_metrics[state.spec.request_id].engine_start_ns == first_start
    assert len(engine_events(result)) == 1
    assert result.request_states[state.spec.request_id].status == serving.RequestStatus.FINISHED
