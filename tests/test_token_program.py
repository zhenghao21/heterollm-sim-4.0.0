"""Closed Token execution has the same live transactions as incremental steps."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel


def graph():
    def task(key, resources=(), dependencies=(), earliest=0.0):
        return TaskSpec(key, "request", key, TaskCategory.COMPUTE,
                        dependencies=dependencies,
                        demands=tuple(ResourceDemand(*item) for item in resources),
                        earliest_start_ns=earliest)
    return (task("a", (("logical.x", 0.1),)),
            task("b", (("physical.x", 0.2),)),
            task("c", (("gpu", 0.3),), ("a",)),
            task("d", (("logical.x", 0.0),), ("b",), 3),
            task("e", (("gpu", 0.7), ("physical.x", 0.3)), ("c", "d")),
            task("f", (), ("e",)))


def pair(tasks, capacities=None, owners=None):
    reference = UnifiedEventKernel(capacities, owners, capture_physical_details=False)
    reference.submit(tasks)
    candidate = UnifiedEventKernel(capacities, owners, capture_physical_details=False)
    return reference, candidate


def consume(kernel):
    result = []
    while kernel.has_active_tasks:
        result.append(kernel.step())
    return tuple(result)


def same_state(left, right):
    for field in ("metrics", "resource_available", "resource_last_interval",
                  "_resource_lane_available", "_resource_lane_last_interval",
                  "completed_end_ns", "completion_leases", "_seen_ids",
                  "_dependents", "_physical_allocation_uses",
                  "_physical_allocation_task_keys", "_pending_physical_releases"):
        assert getattr(left, field) == getattr(right, field), field


@pytest.mark.parametrize("lanes", [1, 2, 3])
def test_live_owner_lanes_and_order_are_exact(lanes):
    tasks = graph()
    reference, candidate = pair(tasks, {"physical.x": lanes}, {"logical.x": "physical.x"})
    expected = consume(reference)
    actual = candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks))
    assert actual == expected
    assert [(type(e.start_ns), type(e.end_ns)) for e in actual] == [
        (type(e.start_ns), type(e.end_ns)) for e in expected]
    same_state(reference, candidate)


def test_observer_stream_and_before_step_peek_follow_identical_boundaries():
    tasks = graph()
    reference, candidate = pair(tasks)
    expected, keys = [], []
    while reference.has_active_tasks:
        keys.append(reference.peek_ready_key())
        expected.append(reference.step())
    actual, observed_keys = [], []
    result = candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
        event_observer=actual.append,
        before_step=lambda kernel: observed_keys.append(kernel.peek_ready_key()))
    assert result == ()
    assert tuple(actual) == tuple(expected)
    assert observed_keys == keys
    same_state(reference, candidate)


def test_observer_cancellation_retains_committed_prefix_and_can_resume():
    tasks = graph()
    reference, candidate = pair(tasks)
    prefix = [reference.step(), reference.step()]
    observed = []
    class Cancelled(Exception):
        pass
    def observe(event):
        observed.append(event)
        if len(observed) == 2:
            raise Cancelled
    with pytest.raises(Cancelled):
        candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                              event_observer=observe)
    assert observed == prefix
    assert candidate.peek_ready_key() == reference.peek_ready_key()
    assert consume(candidate) == consume(reference)
    same_state(reference, candidate)


def test_dynamic_submission_transfers_pending_program_without_reexecuting_prefix():
    tasks = graph()
    reference, candidate = pair(tasks)
    first = reference.step()
    def interrupt(event):
        assert event == first
        raise InterruptedError
    with pytest.raises(InterruptedError):
        candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                              event_observer=interrupt)
    extra = TaskSpec("extra", "request", "extra", TaskCategory.COMPUTE,
                     dependencies=(first.task.task_id,), demands=(ResourceDemand("gpu", 1.1),))
    for kernel in (reference, candidate):
        kernel.submit((extra,))
    assert consume(candidate) == consume(reference)
    same_state(reference, candidate)


def test_failed_transaction_keeps_ready_item_for_retry(monkeypatch):
    tasks = graph()
    reference, candidate = pair(tasks)
    original = candidate._resolve_task_execution
    def reject(task, start, sorted_demands=None):
        raise ValueError("reject before commit")
    monkeypatch.setattr(candidate, "_resolve_task_execution", reject)
    with pytest.raises(ValueError, match="reject before commit"):
        candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                              event_observer=lambda event: None)
    assert candidate.peek_ready_key() == reference.peek_ready_key()
    assert candidate.completed_count == 0
    monkeypatch.setattr(candidate, "_resolve_task_execution", original)
    assert consume(candidate) == consume(reference)
    same_state(reference, candidate)


@pytest.mark.parametrize("l2", [False, True])
def test_physical_and_l2_transactions_match_original_incremental_events(l2):
    from test_native_kernel_physical_cost import native_task
    task, _ = native_task(1.0, l2=l2)
    tasks = (task, replace(task, task_id=task.task_id + ".next", dependencies=(task.task_id,)))
    reference, candidate = pair(tasks)
    expected = consume(reference)
    actual = candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks))
    assert actual == expected
    same_state(reference, candidate)
    for name in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "directions"):
        assert getattr(candidate.physical_runtime.timeline, name) == getattr(reference.physical_runtime.timeline, name)
