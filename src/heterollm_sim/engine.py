"""Deterministic execution engine for ScheduleIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

from .contracts import (
    ResourceInterval,
    RunManifest,
    SimulationTrace,
    TaskResult,
    TaskSpec,
)
from .event_kernel import (
    UnifiedEventKernel,
    validate_task_graph,
)

if TYPE_CHECKING:
    from .execution_control import ExecutionControl


_CONTROL_CHECK_INTERVAL = 256


@dataclass(frozen=True)
class ScheduleIR:
    """A deterministic, dependency-constrained task schedule."""

    manifest: RunManifest
    tasks: Tuple[TaskSpec, ...]
    # Optional hardware concurrency declaration.  Omitting it retains the
    # historical single-lane behavior; callers with multi-lane resources can
    # now use the same capacities as the online runtime.
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    # Explicit logical demand -> shared physical service owner. Unknown
    # relationships remain independent instead of inventing a device lock.
    resource_owners: Mapping[str, str] = field(default_factory=dict)


def simulate_schedule(
    schedule: ScheduleIR,
    *,
    control: Optional["ExecutionControl"] = None,
) -> SimulationTrace:
    """Execute ScheduleIR with deterministic resource contention.

    Ready tasks are selected by effective ready time, then dependency ready
    time, earliest start, and task_id.  All demands of a task start together
    once every dependency and required resource is available.
    """

    if control is not None:
        control.raise_if_cancelled()

    tasks_by_id = validate_task_graph(schedule.tasks)
    total_tasks = len(tasks_by_id)
    if control is not None:
        control.report(
            "schedule",
            0,
            total_tasks,
            message="开始执行详细离散事件计划",
        )
        # A progress callback may itself request cancellation.  Check again
        # before the first event rather than waiting for the first interval.
        control.raise_if_cancelled()
    results: List[TaskResult] = []
    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks, resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    while kernel.has_active_tasks:
        event = kernel.step()
        if event is None:
            kernel.assert_drained()
            break
        task = event.task
        intervals: List[ResourceInterval] = []
        for demand in event.demands:
            end_ns = event.start_ns + demand.service_ns
            intervals.append(
                ResourceInterval(
                    resource_id=demand.resource_id,
                    start_ns=event.start_ns,
                    end_ns=end_ns,
                    bytes_moved=demand.bytes_moved,
                    energy_pj=demand.energy_pj,
                )
            )
        metadata = _merge_engine_metadata(
            task.metadata,
            effective_ready_ns=event.effective_ready_ns,
            dependency_ids=task.dependencies,
            resource_predecessors=event.resource_predecessors,
            resource_lanes=event.resource_lanes,
        )
        result = TaskResult(
            task_id=task.task_id,
            request_id=task.request_id,
            name=task.name,
            category=task.category,
            start_ns=event.start_ns,
            end_ns=event.end_ns,
            dependency_ready_ns=event.dependency_ready_ns,
            marker=task.marker,
            token_index=task.token_index,
            resource_intervals=tuple(intervals),
            metadata=metadata,
        )
        results.append(result)
        if (
            control is not None
            and kernel.completed_count % _CONTROL_CHECK_INTERVAL == 0
            and kernel.completed_count < total_tasks
        ):
            # Check both sides of the callback: the first check responds to an
            # external cancellation, while the second makes progress-driven
            # cancellation deterministic at this exact event boundary.
            control.raise_if_cancelled()
            control.report(
                "schedule",
                kernel.completed_count,
                total_tasks,
                message="正在执行详细离散事件计划",
                simulated_time_ns=kernel.makespan_ns,
            )
            control.raise_if_cancelled()

    if kernel.completed_count != len(tasks_by_id):
        # validate_task_graph() already rejects cycles.  Keep this guard so a
        # future validation change cannot turn a malformed schedule into a
        # partial trace.
        raise ValueError("schedule contains a dependency cycle")

    if control is not None:
        control.raise_if_cancelled()

    # Sort the sole result list in place before freezing it.  The previous
    # ``tuple(sorted(results))`` briefly retained the original list, a second
    # sorted list, and the tuple at once for very long token traces.
    results.sort(key=lambda item: (item.start_ns, item.end_ns, item.task_id))
    trace_tasks = tuple(results)
    trace = SimulationTrace(
        manifest=schedule.manifest,
        tasks=trace_tasks,
        resource_busy_ns=dict(sorted(kernel.resource_busy_ns.items())),
        makespan_ns=kernel.makespan_ns,
        resource_capacities=dict(sorted(kernel.resource_capacities.items())),
    )
    if control is not None:
        control.report(
            "schedule",
            total_tasks,
            total_tasks,
            message="详细离散事件计划执行完成",
            simulated_time_ns=trace.makespan_ns,
        )
    return trace


def _merge_engine_metadata(
    metadata: Mapping[str, object],
    *,
    effective_ready_ns: float,
    dependency_ids: Tuple[str, ...],
    resource_predecessors: Mapping[str, Mapping[str, object]],
    resource_lanes: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    merged = dict(metadata)
    merged["_engine_effective_ready_ns"] = effective_ready_ns
    merged["_engine_dependency_ids"] = tuple(dependency_ids)
    merged["_engine_resource_predecessors"] = dict(resource_predecessors)
    if resource_lanes is not None:
        merged["_engine_resource_lanes"] = dict(resource_lanes)
    return merged
