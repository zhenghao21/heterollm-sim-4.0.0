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


def _task_resource_intervals(
    task: TaskSpec,
    demands: Tuple[object, ...],
    start_ns: float,
) -> List[ResourceInterval]:
    """Project physical reservations and retained ordinary demands together."""
    metadata = task.metadata
    physical = metadata.get("physical_execution")
    if not isinstance(physical, Mapping):
        return [
            ResourceInterval(
                resource_id=demand.resource_id,
                start_ns=start_ns,
                end_ns=start_ns + demand.service_ns,
                bytes_moved=demand.bytes_moved,
                energy_pj=demand.energy_pj,
            )
            for demand in demands
        ]

    intervals: List[ResourceInterval] = []
    payloads = physical.get("resource_interval_payloads", {})
    if isinstance(payloads, Mapping):
        for resource_id, values in payloads.items():
            for item in values or ():
                if isinstance(item, (tuple, list)) and len(item) >= 4:
                    begin, end, moved, energy = item[:4]
                    intervals.append(ResourceInterval(
                        resource_id=str(resource_id), start_ns=begin, end_ns=end,
                        bytes_moved=int(moved), energy_pj=float(energy),
                    ))
    if not intervals and not physical.get("details_truncated", False):
        raw = physical.get("resource_intervals", {})
        resource_bytes = physical.get("resource_bytes", {})
        resource_energy = physical.get("resource_energy_pj", {})
        if isinstance(raw, Mapping):
            for resource_id, values in raw.items():
                for item in values or ():
                    if not isinstance(item, (tuple, list)) or len(item) < 2:
                        continue
                    begin, end = item[:2]
                    intervals.append(ResourceInterval(
                        resource_id=str(resource_id), start_ns=begin, end_ns=end,
                        bytes_moved=int(resource_bytes.get(resource_id, 0)),
                        energy_pj=float(resource_energy.get(resource_id, 0.0)),
                    ))
    if not intervals and not physical.get("details_truncated", False):
        reservations = metadata.get("physical_resource_intervals", ())
        for item in reservations:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                resource_id, begin, end = item[:3]
                moved = int(item[3]) if len(item) > 3 else 0
                energy = float(item[4]) if len(item) > 4 else 0.0
            else:
                resource_id = getattr(item, "resource_id", None)
                begin = getattr(item, "start_ns", None)
                end = getattr(item, "end_ns", None)
                moved = int(getattr(item, "bytes", 0))
                energy = 0.0
            if resource_id is not None and begin is not None and end is not None:
                intervals.append(ResourceInterval(
                    resource_id=str(resource_id), start_ns=begin, end_ns=end,
                    bytes_moved=moved, energy_pj=energy,
                ))

    # Physical tasks retain ordinary compute/link demands alongside replaced
    # memory previews.  Do not invent intervals when physical detail is absent.
    retained_ids = set(str(item) for item in metadata.get("physical_nonmemory_demand_ids", ()))
    if not retained_ids:
        physical_ids = set(str(item) for item in metadata.get("physical_demands_resource_ids", ()))
        retained_ids = {d.resource_id for d in demands if d.resource_id not in physical_ids}
    for demand in demands:
        if demand.resource_id in retained_ids:
            intervals.append(ResourceInterval(
                resource_id=demand.resource_id,
                start_ns=start_ns,
                end_ns=start_ns + demand.service_ns,
                bytes_moved=demand.bytes_moved,
                energy_pj=demand.energy_pj,
            ))
    return intervals


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
        intervals = _task_resource_intervals(task, event.demands, event.start_ns)
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
