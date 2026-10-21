"""Initial native PD publishes only its proved phase without stale mirrors."""
from types import MethodType
from dataclasses import replace
from queue import SimpleQueue

import pytest

from heterollm_sim.serving import _OnlineRuntime
from test_native_external_kv import ingress_scenario, native_runtime


def _same_mirror(left, right):
    assert list(left) == list(right)
    assert {key: float(value).hex() for key, value in left.items()} == {
        key: float(value).hex() for key, value in right.items()}


@pytest.mark.parametrize("case", ("hbm_tp1", "hbf_weights_kv_tp1"))
def test_real_initial_PD_sparse_and_complete_publication_match_after_every_phase(case, monkeypatch):
    from heterollm_sim import _resource_ledger

    sparse, full = (native_runtime(ingress_scenario(case, count=2)) for _ in range(2))
    original = _OnlineRuntime._publish_prefetch_resource_availability
    full._publish_prefetch_resource_availability = MethodType(
        lambda self, **_kwargs: original(self), full)
    calls, complete_calls = [], []
    publish = _resource_ledger.publish_timeline_frontiers
    def observe(lanes, available):
        if lanes is sparse._execution_resource_kernel.physical_runtime.timeline.lane_available:
            calls.append(len(lanes))
        if lanes is full._execution_resource_kernel.physical_runtime.timeline.lane_available:
            complete_calls.append(len(lanes))
        return publish(lanes, available)
    monkeypatch.setattr(_resource_ledger, "publish_timeline_frontiers", observe)
    for runtime in (sparse, full):
        timeline = runtime._execution_resource_kernel.physical_runtime.timeline
        for index in range(200):
            timeline.reserve("unrelated-gpu:"+str(index), index+.125, .375)
        runtime._arrive()
        runtime._start_external_imports()
    phase_count = 0
    while sparse._external_pending:
        assert sparse._next_external_phase() == full._next_external_phase()
        for runtime in (sparse, full):
            runtime.now = runtime._next_external_phase()
            runtime._complete_external_imports()
        phase_count += 1
        _same_mirror(sparse._execution_resource_kernel.resource_available,
                     full._execution_resource_kernel.resource_available)
        assert sparse._external_executor.cost_ledger() == full._external_executor.cost_ledger()
    assert sparse.events == full.events
    assert sparse._external_imported_bytes == full._external_imported_bytes
    from test_token_owned_history import compare_physical_state
    compare_physical_state(sparse._execution_resource_kernel, full._execution_resource_kernel)
    if case == "hbf_weights_kv_tp1":
        assert sparse._native_prefetch_publication is not None
    assert len(calls) < len(complete_calls)


def test_public_shared_lane_lists_keep_complete_publication():
    runtime = native_runtime(ingress_scenario(count=1))
    lanes = runtime._execution_resource_kernel.physical_runtime.timeline.lane_available
    lanes["retained-public-alias"] = next(iter(lanes.values()))
    runtime._arrive()
    runtime._start_external_imports()
    while runtime._external_pending:
        runtime.now = runtime._next_external_phase()
        runtime._complete_external_imports()
        assert runtime._native_prefetch_publication is None


def test_authored_plan_producer_cannot_hide_a_foreign_lane_from_publication(monkeypatch):
    runtime = native_runtime(ingress_scenario(count=1))
    executor = runtime._external_executor
    prepare = executor.prepare_external_import
    def authored(**kwargs):
        plan = prepare(**kwargs)
        executor.runtime.timeline.lane_available["authored:foreign"] = [1234.]
        return plan
    monkeypatch.setattr(executor, "prepare_external_import", authored)
    runtime._arrive()
    runtime._start_external_imports()
    assert runtime._execution_resource_kernel.resource_available["authored:foreign"] == 1234.
    assert runtime._native_prefetch_publication is None


def test_after_model_execution_publication_keeps_the_complete_mutation_domain():
    runtime = native_runtime(ingress_scenario(count=1))
    runtime._arrive()
    runtime._start_external_imports()
    plan = runtime._external_pending["r0"]["plan"]
    runtime._batch_kind_counts["decode"] = 1
    runtime._execution_resource_kernel.physical_runtime.timeline.lane_available["other-model-lane"] = [321.]
    runtime._publish_prefetch_resource_availability(plan=plan, executor=runtime._external_executor)
    assert runtime._execution_resource_kernel.resource_available["other-model-lane"] == 321.
    assert runtime._native_prefetch_publication is None


@pytest.mark.parametrize("kind", ("progress", "cancellation"))
def test_author_control_callback_updates_a_foreign_lane_and_forces_full_publication(kind):
    from heterollm_sim.execution_control import ExecutionControl

    runtime = native_runtime(ingress_scenario(count=1))
    runtime._arrive()
    runtime._start_external_imports()
    plan = runtime._external_pending["r0"]["plan"]
    lanes = runtime._execution_resource_kernel.physical_runtime.timeline.lane_available
    # Keep dictionary size unchanged: a size-only cache cannot detect this.
    foreign = next(iter(lanes))
    def mutate(*_args):
        lanes[foreign][0] = 98765.
        return False
    if kind == "progress":
        runtime.execution_control = ExecutionControl(progress_callback=mutate)
        runtime.execution_control.report("serving_boundary", 0)
    else:
        runtime.execution_control = ExecutionControl(cancellation_callback=mutate)
        runtime.execution_control.raise_if_cancelled()
    runtime._publish_prefetch_resource_availability(plan=plan, executor=runtime._external_executor)
    assert runtime._execution_resource_kernel.resource_available[foreign] == min(lanes[foreign])
    assert runtime._native_prefetch_publication is None


def test_native_progress_queue_preserves_real_cumulative_Token_and_batch_counts():
    runtime = native_runtime(ingress_scenario(count=1))
    queue = SimpleQueue()
    runtime.execution_control = replace(runtime.execution_control, progress_queue=queue)
    result = runtime.run()
    observations = []
    while not queue.empty():
        observations.append(queue.get_nowait())
    progress = next(row.metadata["cumulative_progress"] for row in reversed(observations)
                    if "cumulative_progress" in row.metadata)
    assert progress["cohorts"] > 0 and progress["prefill_compute_tokens"] == 1
    assert progress["visible_output_tokens"] == result.request_metrics["r0"].visible_output_tokens == 4
