"""Exact-prefix stops preserve the original request and resume semantics."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

import pytest

from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim.serving import BatchCost, RequestStatus, _OnlineRuntime, compile_serving_plan
from test_hbf_host_staging_integration import strict_staging_scenario


def runtime(output=9, *, control=None, lowerer=None):
    scenario = strict_staging_scenario(prompt=7, output=max(1, output), all_hbf=True, capacity=65536)
    scenario = replace(scenario, placement=replace(scenario.placement,
        kv_policy=replace(scenario.placement.kv_policy, offload_ratio=0., prefetch_distance=0)),
        workload=replace(scenario.workload, requests=tuple(
            replace(request, output_tokens=output) for request in scenario.workload.requests)))
    return _OnlineRuntime(compile_serving_plan(scenario),
        lowerer or (lambda _scenario, cohort: BatchCost(31.7 + cohort.token_count * .3)),
        execution_control=control)


def exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, float):
        assert left.hex() == right.hex()
    elif is_dataclass(left):
        for field in fields(left):
            exact(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for item, expected in zip(left, right, strict=True):
            exact(item, expected)
    else:
        assert left == right


def test_multiple_stops_resume_to_identical_result_and_progress():
    uninterrupted_progress, resumed_progress = [], []
    reference = runtime(control=ExecutionControl(progress_callback=uninterrupted_progress.append)).run()
    runner = runtime(control=ExecutionControl(progress_callback=resumed_progress.append))
    assert runner.run_until_decode_steps(0) is False
    state = runner.states["r0"]
    assert state.spec.output_tokens == 9 and state.committed == 1
    assert state.first_token_ns is not None and state.finished_ns is None
    assert state.status is RequestStatus.RUNNING and state.kv_pages > 0
    assert not any(event.event_type == "request_finished" for event in runner.events)
    assert runner.run_until_decode_steps(2) is False
    assert state.committed == 3
    boundary = (runner.now, runner.rounds, len(runner.events), len(runner.batches))
    assert runner.run_until_decode_steps(2) is False
    assert (runner.now, runner.rounds, len(runner.events), len(runner.batches)) == boundary
    assert runner.run_until_decode_steps(4) is False
    exact(runner.run(), reference)
    exact(resumed_progress, uninterrupted_progress)
    assert sum(progress.message == "online serving started" for progress in resumed_progress) == 1


@pytest.mark.parametrize("output", [0, 1, 2])
def test_short_outputs_finish_naturally_without_phantom_decode(output):
    reference = runtime(output).run()
    runner = runtime(output)
    complete = runner.run_until_decode_steps(100)
    assert complete is True
    assert runner._completed_decode_steps == max(0, output - 1)
    exact(runner.run(), reference)


@pytest.mark.parametrize("target,exception", [(-1, ValueError), (True, TypeError), (1.0, TypeError), (None, TypeError)])
def test_invalid_stop_does_not_start_or_mutate_runtime(target, exception):
    runner = runtime()
    with pytest.raises(exception, match="non-negative integer"):
        runner.run_until_decode_steps(target)
    assert runner._run_initialized is False
    assert runner.rounds == 0 and not runner.batches and not runner.events


def test_cancelled_prefix_does_not_finish_and_can_continue_at_same_boundary():
    cancelled = False
    progress = []
    runner = runtime(control=ExecutionControl(progress_callback=progress.append,
        cancellation_callback=lambda: cancelled))
    assert runner.run_until_decode_steps(2) is False
    before = (runner.now, runner.rounds, len(runner.batches), deepcopy(runner.states["r0"].__dict__))
    cancelled = True
    with pytest.raises(ExecutionCancelledError):
        runner.run_until_decode_steps(4)
    assert before == (runner.now, runner.rounds, len(runner.batches), runner.states["r0"].__dict__)
    assert not any(row.stage == "serving_complete" for row in progress)
    cancelled = False
    exact(runner.run(), runtime().run())


def test_lowering_failure_propagates_without_finalizing_prefix():
    progress = []
    def lower(_scenario, cohort):
        if cohort.kind == "decode":
            raise RuntimeError("injected decode lowering failure")
        return BatchCost(30.)
    runner = runtime(control=ExecutionControl(progress_callback=progress.append), lowerer=lower)
    with pytest.raises(RuntimeError, match="injected decode lowering failure"):
        runner.run_until_decode_steps(2)
    assert runner.states["r0"].finished_ns is None
    assert runner.states["r0"].committed == 1
    assert runner._completed_decode_steps == 0
    assert not any(row.stage == "serving_complete" for row in progress)


def test_full_request_plan_is_not_rebuilt_at_each_stop(monkeypatch):
    runner = runtime()
    plan = runner.plan
    rejected = []
    original = runner._reject_impossible_requests
    def check():
        rejected.append(True)
        original()
    monkeypatch.setattr(runner, "_reject_impossible_requests", check)
    runner.run_until_decode_steps(0)
    runner.run_until_decode_steps(1)
    runner.run_until_decode_steps(3)
    runner.run()
    assert runner.plan is plan
    assert plan.requests[0].output_tokens == 9
    assert rejected == [True]


def test_physical_hbf_prefix_resume_keeps_transactions_and_controller_endpoint():
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.planner import TopologyAwareBatchCostProvider, _compilation_scope

    def execute(stops):
        scenario = runtime(output=5).plan.scenario
        bootstrap = bootstrap_control_plane(scenario, capture_physical_details=False)
        scenario = bootstrap.scenario
        lowerer = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
        with _compilation_scope(scenario, lowerer._compilation_context):
            runner = _OnlineRuntime(compile_serving_plan(scenario), lowerer,
                execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
            for target in stops:
                assert runner.run_until_decode_steps(target) is False
            result = runner.run()
            physical = runner._execution_resource_kernel.physical_runtime
            controllers = {owner: value["controller"].snapshot()
                           for owner, value in physical.hbf_staging.items()}
            cores = {}
            for owner, device in physical.runtimes.items():
                core = device.core
                cores[owner] = {name: deepcopy(getattr(core, name)) for name in (
                    "_banks", "_inflight", "_acceptance_ns", "_host_inflight",
                    "_host_acceptance_ns", "_next_program_page", "_read_cache",
                ) if hasattr(core, name)}
            return result, controllers, cores

    expected, expected_controllers, expected_cores = execute(())
    actual, controllers, cores = execute((0, 1, 3))
    exact(actual, expected)
    assert controllers == expected_controllers
    exact(cores, expected_cores)
