"""Owned native Token frames retain compact history and sparse frontiers."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel
from test_token_program import consume, graph


def bits(value):
    if type(value) is float:
        return float, value.hex()
    if isinstance(value, Mapping):
        return tuple((key, bits(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return type(value), tuple(bits(item) for item in value)
    if is_dataclass(value):
        return type(value), tuple((field.name, bits(getattr(value, field.name)))
                                  for field in fields(value))
    return value


def prefixed(prefix):
    return tuple(replace(task, task_id=prefix + task.task_id,
                         dependencies=tuple(prefix + key for key in task.dependencies))
                 for task in graph())


def compare_physical_state(left, right):
    for field in ("metrics", "resource_available", "resource_last_interval",
                  "_resource_lane_available", "_resource_lane_last_interval",
                  "completed_end_ns", "completion_leases"):
        assert bits(getattr(left, field)) == bits(getattr(right, field)), field
    # This lookup index is populated at submit versus dispatch; no numerical
    # reduction observes its dictionary insertion order.
    assert left._dependents == right._dependents
    saved = [kernel.physical_runtime.snapshot() for kernel in (left, right)]
    for field in ("timeline", "allocator_state", "committed_owners", "hbf_staging"):
        assert bits(saved[0][field]) == bits(saved[1][field]), field
    assert saved[0]["runtime_state"].keys() == saved[1]["runtime_state"].keys()
    for owner, actual in saved[0]["runtime_state"].items():
        expected = saved[1]["runtime_state"][owner]
        assert bits(actual["clock_ns"]) == bits(expected["clock_ns"])
        assert bits(actual["core"]) == bits(expected["core"])


def test_shared_suffix_history_keeps_all_ids_without_expanding_per_token():
    reference, candidate = UnifiedEventKernel(), UnifiedEventKernel()
    suffixes = tuple(task.task_id for task in graph())
    for prefix in ("token.0001.", "token.0002.", "token.0003."):
        tasks = prefixed(prefix)
        reference.submit(tasks)
        assert candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
            seen_namespace=(prefix, suffixes)) == consume(reference)
        assert all(candidate._has_seen_task_id(task.task_id) for task in tasks)
        compare_physical_state(candidate, reference)
    assert not candidate._seen_ids
    assert len(candidate._compact_seen_namespaces) == 3
    assert all(values is suffixes for values in candidate._compact_seen_namespaces.values())
    for kernel in (reference, candidate):
        with pytest.raises(ValueError, match="duplicate task_id"):
            kernel.submit((prefixed("token.0001.")[0],))


def test_duplicate_or_colliding_namespace_rejection_keeps_state_unchanged():
    kernel = UnifiedEventKernel()
    tasks = prefixed("outer.inner.")
    suffixes = tuple(task.task_id for task in graph())
    kernel._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                      seen_namespace=("outer.inner.", suffixes))
    count = kernel.completed_count
    with pytest.raises(ValueError, match="already registered"):
        kernel._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                          seen_namespace=("outer.inner.", suffixes))
    with pytest.raises(ValueError, match="duplicate task_id"):
        kernel._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
            seen_namespace=("outer.", tuple("inner." + key for key in suffixes)))
    assert kernel.completed_count == count
    assert len(kernel._compact_seen_namespaces) == 1
    assert not kernel.has_active_tasks


@pytest.mark.parametrize("invalid", ("mismatched", "nonfinite", "explicit_duplicate"))
def test_compact_namespace_validation_is_atomic(invalid):
    kernel = UnifiedEventKernel()
    tasks = prefixed("token.1.")
    suffixes = tuple(task.task_id for task in graph())
    if invalid == "mismatched":
        suffixes = tuple("wrong." + key for key in suffixes)
    elif invalid == "nonfinite":
        tasks = (replace(tasks[0]), *tasks[1:])
        object.__setattr__(tasks[0], "earliest_start_ns", float("nan"))
    else:
        kernel.submit((tasks[0],))
        consume(kernel)
    completed = kernel.completed_count
    with pytest.raises(ValueError):
        kernel._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                          seen_namespace=("token.1.", suffixes))
    assert not kernel._compact_seen_namespaces
    assert kernel.completed_count == completed
    assert not kernel.has_active_tasks


def test_suspended_compact_frame_can_export_to_incremental_and_reject_old_ids():
    tasks = prefixed("token.1.")
    reference, candidate = UnifiedEventKernel(), UnifiedEventKernel()
    reference.submit(tasks)
    expected_prefix = (reference.step(), reference.step())
    observed = []
    def suspend(event):
        observed.append(event)
        if len(observed) == 2:
            raise InterruptedError("suspend")
    with pytest.raises(InterruptedError, match="suspend"):
        candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
            event_observer=suspend, seen_namespace=("token.1.", tuple(task.task_id for task in graph())))
    assert tuple(observed) == expected_prefix
    extra = TaskSpec("extra", "request", "extra", TaskCategory.COMPUTE,
        dependencies=(tasks[0].task_id,), demands=(ResourceDemand("gpu", .125),))
    for kernel in (reference, candidate):
        kernel.submit((extra,))
        with pytest.raises(ValueError, match="duplicate task_id"):
            kernel.submit((tasks[1],))
    assert consume(candidate) == consume(reference)
    compare_physical_state(candidate, reference)


@pytest.mark.parametrize("owned", (False, True))
def test_physical_frontiers_include_zero_busy_reservations_and_exact_ledger(monkeypatch, owned):
    from heterollm_sim import data_motion
    from test_native_kernel_physical_cost import native_task
    first, _ = native_task(1.)
    second = replace(first, task_id="next", dependencies=(first.task_id,))
    final = TaskSpec("zero-consumer", "request", "zero-consumer", TaskCategory.COMPUTE,
        dependencies=(second.task_id,), demands=(ResourceDemand("zero:next", .1),))
    tasks = (first, second, final)
    original = data_motion.resolve_physical_task
    def with_zero(task, runtime, arrival_ns=0., **kwargs):
        result = original(task, runtime, arrival_ns, **kwargs)
        key = "zero:" + task.task_id
        time = float(result.metadata["physical_completion_ns"])
        runtime.timeline.reserve(key, time, 0.)
        return replace(result, metadata={**result.metadata,
            "physical_resource_last_intervals": {
                **result.metadata["physical_resource_last_intervals"], key: (time, time)}})
    monkeypatch.setattr(data_motion, "resolve_physical_task", with_zero)
    reference, candidate = UnifiedEventKernel(capture_physical_details=False), UnifiedEventKernel(capture_physical_details=False)
    reference.submit(tasks)
    expected = consume(reference)
    observed = candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
        owned_physical_frontiers=owned)
    assert observed == expected
    assert candidate.resource_available["zero:next"] == reference.resource_available["zero:next"]
    compare_physical_state(candidate, reference)


def test_unknown_callback_keeps_full_frontier_publication_of_unrelated_edits():
    from test_native_kernel_physical_cost import native_task
    first, _ = native_task(1.)
    second = replace(first, task_id="next", dependencies=(first.task_id,))
    final = TaskSpec("outside-consumer", "request", "outside-consumer", TaskCategory.COMPUTE,
        dependencies=(second.task_id,), demands=(ResourceDemand("outside", .1),))
    tasks = (first, second, final)
    reference, candidate = UnifiedEventKernel(capture_physical_details=False), UnifiedEventKernel(capture_physical_details=False)
    reference.submit(tasks)
    expected = []
    for index in range(len(tasks)):
        event = reference.step()
        expected.append(event)
        if not index:
            reference.physical_runtime.timeline.reserve("outside", 1.e6, 0.)
    actual = []
    def observe(event):
        actual.append(event)
        if len(actual) == 1:
            candidate.physical_runtime.timeline.reserve("outside", 1.e6, 0.)
    candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                         event_observer=observe)
    assert actual == expected
    compare_physical_state(candidate, reference)


def test_successful_unknown_core_callback_publishes_foreign_lane_on_later_task(monkeypatch):
    from heterollm_sim.dram_core import DramCore
    from test_native_kernel_physical_cost import native_task

    first, _ = native_task(1.)
    second = replace(first, task_id="next", dependencies=(first.task_id,))
    final = TaskSpec("outside-consumer", "request", "outside-consumer", TaskCategory.COMPUTE,
        dependencies=(second.task_id,), demands=(ResourceDemand("outside", .1),))
    tasks = (first, second, final)
    original = DramCore.submit
    calls = {}

    def foreign_writer(core, request, **kwargs):
        result = original(core, request, **kwargs)
        calls[id(core)] = calls.get(id(core), 0) + 1
        if calls[id(core)] == 3:
            # A successful unknown hook need not use ResourceTimeline.reserve
            # or mark this unrelated resource in the current task's cost bill.
            core.timeline.lane_available["outside"] = [1.e6]
        return result

    monkeypatch.setattr(DramCore, "submit", foreign_writer)
    reference, candidate = (UnifiedEventKernel(capture_physical_details=False)
                            for _ in range(2))
    reference.submit(tasks)
    expected = consume(reference)
    observed = candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
        owned_physical_frontiers=True)
    assert expected[-1].start_ns == 1.e6
    assert observed == expected
    compare_physical_state(candidate, reference)


@pytest.mark.parametrize("observer_kind", ("progress", "cancellation"))
def test_native_authored_control_observer_keeps_foreign_calendar_frontier(observer_kind):
    from heterollm_sim._token_binding import AuthoredTokenProgram
    from heterollm_sim.contracts import _PreparedExecutionStage, _PreparedExecutionTask
    from heterollm_sim.execution_control import ExecutionControl
    from heterollm_sim.serving import _OnlineRuntime
    from test_native_kernel_physical_cost import native_task

    first, _ = native_task(1.)
    rows = tuple(_PreparedExecutionTask(
        "read" + str(index), ("read" + str(index - 1),) if index else (),
        ("request",), first.demands, category=first.category, metadata=first.metadata)
        for index in range(257))
    consumer = _PreparedExecutionTask("outside-consumer", ("read256",), ("request",),
        (ResourceDemand("outside", .1),), category=TaskCategory.COMPUTE)
    stages = (_PreparedExecutionStage("body", 0, (), ("request",), "memory", 0.,
                                      (*rows, consumer)),)
    program = AuthoredTokenProgram.compile(stages, _OnlineRuntime._stage_task_specs)
    kernels, observations = [], []
    for owned in (False, True):
        kernel = UnifiedEventKernel(capture_physical_details=False)
        if not owned:
            original_drain = kernel._drain_prevalidated_compiled

            def full_publication(*args, _drain=original_drain, **kwargs):
                kwargs["owned_physical_frontiers"] = False
                return _drain(*args, **kwargs)

            kernel._drain_prevalidated_compiled = full_publication
        events, changed = [], []

        class Observer:
            def observe(self, event):
                events.append(event)

        def change_calendar(_progress=None):
            if len(events) == 256:
                kernel.physical_runtime.timeline.reserve("outside", 1.e12, 0.)
                changed.append(True)
            return False

        runtime = object.__new__(_OnlineRuntime)
        runtime._execution_resource_kernel = kernel
        runtime._live_cohort_metrics = Observer()
        runtime._pending_prefetches = ()
        runtime._external_pending = {}
        runtime.execution_control = ExecutionControl(
            progress_callback=change_calendar if observer_kind == "progress" else None,
            cancellation_callback=change_calendar if observer_kind == "cancellation" else None)
        runtime._replay_authored_token_program(stages, {"body": 0.},
                                               "native.callback", program)
        assert changed == [True]
        assert events[-1].start_ns == 1.e12
        kernels.append(kernel)
        observations.append(events)
    assert observations[0] == observations[1]
    compare_physical_state(kernels[0], kernels[1])
