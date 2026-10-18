"""Deferred previews reduce native events at the original commit boundary."""

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.engine import ScheduleIR
from heterollm_sim.event_kernel import CompiledGraphExecutor, UnifiedEventKernel
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim import scalable_serving


def schedule(count=300):
    tasks = tuple(TaskSpec(f"task.{index:04d}", "request", f"task.{index:04d}",
        TaskCategory.COMPUTE if index % 2 else TaskCategory.MEMORY,
        dependencies=((f"task.{index - 3:04d}",) if index >= 3 else ()),
        demands=(ResourceDemand("logical_cpu", 0.0 if index % 7 == 0 else 0.1, index),),
        earliest_start_ns=float(index % 3),
        metadata={"coverage_component": "cpu", "analytical_ops": float(index)})
        for index in range(count))
    return ScheduleIR(None, tasks, {"cpu": 2}, {"logical_cpu": "cpu"})


def test_native_preview_matches_every_summary_record_and_progress():
    progress = []
    def run(executor):
        rows = []
        result = scalable_serving.execute_cost_schedule(schedule(),
            defer_physical_execution=True, retain_task_metadata=False,
            compiled_executor=executor,
            control=ExecutionControl(progress_callback=rows.append))
        progress.append(rows)
        return result
    assert run(None) == run(CompiledGraphExecutor())
    assert progress[0] == progress[1]


def test_preview_cancellation_preserves_committed_prefix_and_ready_frontier(monkeypatch):
    kernels = []
    class CapturingKernel(UnifiedEventKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kernels.append(self)
    monkeypatch.setattr(scalable_serving, "UnifiedEventKernel", CapturingKernel)
    progress = []
    for executor in (None, CompiledGraphExecutor()):
        rows = []
        def observe(item):
            rows.append(item)
            if item.completed == 256:
                raise ExecutionCancelledError("stop at the original boundary")
        with pytest.raises(ExecutionCancelledError):
            scalable_serving.execute_cost_schedule(schedule(),
                defer_physical_execution=True, retain_task_metadata=False,
                compiled_executor=executor,
                control=ExecutionControl(progress_callback=observe))
        progress.append(rows)
    assert progress[0] == progress[1]
    first, second = kernels
    assert first.completed_count == second.completed_count == 256
    assert first.active_task_count == second.active_task_count == 44
    assert first.peek_ready_key() == second.peek_ready_key()
    assert first.resource_available == second.resource_available
    assert first.metrics == second.metrics
    assert first.completed_end_ns == second.completed_end_ns
