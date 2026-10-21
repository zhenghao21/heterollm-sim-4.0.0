"""Native progress delivery cannot call back into physical simulation state."""
from pathlib import Path
from queue import SimpleQueue
from time import perf_counter

import pytest

from heterollm_sim.execution_control import (ExecutionControl, ExecutionCancelledError,
    NativeFastWallBudgetError, native_callback_free_control)


def test_native_queue_receives_progress_before_the_optional_author_callback():
    queue = SimpleQueue()
    observed = []
    def callback(progress):
        assert queue.get_nowait() is progress
        observed.append(progress)
    control = ExecutionControl(progress_queue=queue, progress_callback=callback)
    control.report("cohort_tasks", 7, 10, metadata={"tokens": 9})
    assert len(observed) == 1 and observed[0].completed == 7
    assert observed[0].metadata == {"tokens": 9}
    assert not native_callback_free_control(control)


def test_native_stop_wins_over_wall_deadline_and_user_cancel(tmp_path):
    stop = tmp_path/"stop.request"
    stop.write_text("pause", encoding="utf-8")
    calls = []
    control = ExecutionControl(stop_request_path=stop, deadline_monotonic=perf_counter()-1.,
        cancellation_callback=lambda: calls.append(1) or False)
    assert type(control.stop_request_path) is str
    with pytest.raises(ExecutionCancelledError):
        control.raise_if_cancelled()
    assert not calls and stop.exists()


def test_native_wall_deadline_uses_the_existing_budget_rejection_type():
    with pytest.raises(NativeFastWallBudgetError, match="wall budget exhausted"):
        ExecutionControl(deadline_monotonic=perf_counter()-1.).raise_if_cancelled()
    ExecutionControl(deadline_monotonic=perf_counter()+60.).raise_if_cancelled()


def test_native_user_cancel_still_wins_over_an_expired_deadline():
    with pytest.raises(ExecutionCancelledError):
        ExecutionControl(cancellation_callback=lambda: True,
                         deadline_monotonic=perf_counter()-1.).raise_if_cancelled()


def test_native_stop_deadline_and_queue_are_closed_without_author_callbacks(tmp_path):
    control = ExecutionControl(stop_request_path=tmp_path/"absent", deadline_monotonic=perf_counter()+60.,
                               progress_queue=SimpleQueue())
    assert native_callback_free_control(control)
    control.raise_if_cancelled()
    before = perf_counter()
    control.report("serving_boundary", 0)
    progress = control.progress_queue.get_nowait()
    assert progress.stage == "serving_boundary"
    assert before <= progress.emitted_monotonic <= perf_counter()
    assert native_callback_free_control(ExecutionControl())


@pytest.mark.parametrize("deadline", (True, -1., float("nan"), float("inf"), "10"))
def test_native_deadline_rejects_non_clock_values(deadline):
    with pytest.raises(ValueError):
        ExecutionControl(deadline_monotonic=deadline)


def test_progress_queue_rejects_an_author_put_callback():
    class AuthoredQueue:
        def put(self, progress):
            raise AssertionError("must not run during construction")
    with pytest.raises(TypeError, match="native SimpleQueue"):
        ExecutionControl(progress_queue=AuthoredQueue())


def test_unknown_control_subclass_or_class_property_is_not_a_closed_native_writer(monkeypatch):
    class AuthoredControl(ExecutionControl):
        pass
    assert not native_callback_free_control(AuthoredControl())
    control = ExecutionControl()
    def callback_property(self):
        raise AssertionError("qualification must not call authored descriptors")
    monkeypatch.setattr(ExecutionControl, "progress_callback", property(callback_property))
    assert not native_callback_free_control(control)
