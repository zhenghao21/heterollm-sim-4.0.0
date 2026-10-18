"""Active callbacks preserve the incremental kernel's fixed event budget."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel
from heterollm_sim.runtime_ir import RUNTIME_PHASE_METADATA_KEY
from test_token_program import consume, graph, pair, same_state


def _append(kernel, mode, completed=None):
    dependencies = (completed.task.task_id,) if completed is not None and mode == "submit" else ()
    first = TaskSpec("extra.0", "request", "extra.0", TaskCategory.COMPUTE,
                     demands=(ResourceDemand("gpu", .25),), dependencies=dependencies)
    second = replace(first, task_id="extra.1", name="extra.1", dependencies=(first.task_id,))
    tasks = (first, second)
    if mode == "submit_compiled":
        return kernel.submit_compiled(tasks, CompiledGraphLayout.compile(tasks))
    return kernel.submit(tasks)


@pytest.mark.parametrize("when", ["before_step", "observer"])
@pytest.mark.parametrize("mode", ["submit", "submit_compiled"])
@pytest.mark.parametrize("boundary", [0, 2, 5])
def test_active_callback_submission_matches_fixed_incremental_event_budget(when, mode, boundary):
    tasks = graph()
    reference, candidate = pair(tasks)
    expected, reference_receipts = [], []
    for index in range(len(tasks)):
        if when == "before_step" and index == boundary:
            reference_receipts.append(_append(reference, mode))
        event = reference.step()
        expected.append(event)
        if when == "observer" and index == boundary:
            reference_receipts.append(_append(reference, mode, event))
    actual, receipts = [], []
    calls = 0
    def before(kernel):
        nonlocal calls
        if when == "before_step" and calls == boundary:
            receipts.append(_append(kernel, mode))
        calls += 1
    def observe(event):
        actual.append(event)
        if when == "observer" and len(actual) - 1 == boundary:
            receipts.append(_append(candidate, mode, event))
    result = candidate._drain_prevalidated_compiled(
        tasks, CompiledGraphLayout.compile(tasks), before_step=before, event_observer=observe)
    assert result == ()
    assert actual == expected
    assert receipts == reference_receipts
    assert candidate.active_task_count == reference.active_task_count
    assert candidate.peek_ready_key() == reference.peek_ready_key()
    # The stage runner owns the original event count. Appended tasks can
    # remain; continuation must retain their exact timing and dependencies.
    assert consume(candidate) == consume(reference)
    same_state(reference, candidate)


@pytest.mark.parametrize("mode", ["submit", "submit_compiled"])
def test_rejected_active_callback_submission_does_not_change_committed_prefix(mode):
    tasks = graph()
    reference, candidate = pair(tasks)
    rejected = (replace(tasks[0], name="duplicate id"),)
    layout = CompiledGraphLayout.compile(rejected)
    def reject(kernel):
        with pytest.raises(ValueError, match="duplicate task_id"):
            if mode == "submit_compiled":
                kernel.submit_compiled(rejected, layout)
            else:
                kernel.submit(rejected)
    expected = []
    for _ in tasks:
        reject(reference)
        expected.append(reference.step())
    actual = []
    candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                         event_observer=actual.append, before_step=reject)
    assert actual == expected
    same_state(reference, candidate)


def test_empty_program_clears_frame_and_allows_following_submission():
    kernel = UnifiedEventKernel()
    submitted, before, observed = [], [], []
    assert kernel._drain_prevalidated_compiled((), CompiledGraphLayout.compile(()),
        event_observer=observed.append, before_step=lambda value: before.append(value),
        on_submitted=lambda: submitted.append(True)) == ()
    assert submitted == [True]
    assert not before and not observed
    assert getattr(kernel, "_token_frame", None) is None
    assert kernel.peek_ready_key() is None
    kernel.submit(graph())
    assert len(consume(kernel)) == len(graph())


def test_submission_callback_failure_keeps_full_pending_graph_for_resume():
    tasks = graph()
    reference, candidate = pair(tasks)
    def fail():
        raise InterruptedError("cancel after submission")
    with pytest.raises(InterruptedError, match="after submission"):
        candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                             on_submitted=fail)
    assert candidate.completed_count == 0
    assert candidate.peek_ready_key() == reference.peek_ready_key()
    assert consume(candidate) == consume(reference)
    same_state(reference, candidate)


@pytest.mark.parametrize("failure", ["nonfinite", "owner_alias", "runtime_metadata", "layout"])
def test_failed_closed_program_validation_is_atomic_and_retryable(failure):
    tasks = graph()
    layout = CompiledGraphLayout.compile(tasks)
    candidate = UnifiedEventKernel(resource_owners={"logical.x": "physical.x"})
    bad = list(tasks)
    if failure == "nonfinite":
        bad[0] = replace(bad[0])
        object.__setattr__(bad[0], "earliest_start_ns", float("nan"))
    elif failure == "owner_alias":
        bad[0] = replace(bad[0], demands=(ResourceDemand("logical.x", 1.), ResourceDemand("physical.x", 2.)))
        # The graph itself is valid; the live owner contract must reject
        # these two logical IDs before any frame state is committed.
        layout = CompiledGraphLayout.compile(tuple(bad))
    elif failure == "runtime_metadata":
        bad[0] = replace(bad[0], metadata={RUNTIME_PHASE_METADATA_KEY: True})
    else:
        bad[0] = replace(bad[0], demands=(ResourceDemand("different", 1.),))
    with pytest.raises((ValueError, TypeError)):
        candidate._drain_prevalidated_compiled(tuple(bad), layout, event_observer=lambda event: None)
    assert candidate.active_task_count == 0
    assert candidate.completed_count == 0
    assert not candidate._seen_ids
    assert getattr(candidate, "_token_frame", None) is None
    assert candidate.peek_ready_key() is None
    reference = UnifiedEventKernel(resource_owners={"logical.x": "physical.x"})
    reference.submit(tasks)
    assert candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks)) == consume(reference)
    same_state(reference, candidate)
