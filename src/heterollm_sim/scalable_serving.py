"""Exact, allocation-light primitives for topology-aware serving costs.

This module intentionally returns aggregate cost fields rather than a
``SimulationTrace``.  Online serving never exposes each cohort's internal
planner trace, so materializing thousands of ``TaskResult`` and
``ResourceInterval`` objects only to immediately reduce them is avoidable.
The scheduler and critical-path ordering are identical to ``engine.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple

from .contracts import ResourceDemand, TaskCategory
from .engine import ScheduleIR
from .event_kernel import CompiledGraphExecutor, UnifiedEventKernel
from .execution_control import ExecutionControl


@dataclass(frozen=True)
class TaskExecutionRecord:
    """Immutable task-timeline fact retained only for immediate reduction."""

    task_id: str
    start_ns: float
    end_ns: float
    category: TaskCategory
    dependencies: Tuple[str, ...]
    metadata: Mapping[str, Any]
    demands: Tuple[ResourceDemand, ...]


@dataclass(frozen=True)
class ScheduleCostResult:
    makespan_ns: float
    task_count: int
    energy_pj: float
    bytes_moved: int
    resource_busy_ns: Mapping[str, float]
    category_time_ns: Mapping[TaskCategory, float]
    critical_path_category_ns: Mapping[TaskCategory, float]
    coverage: Mapping[str, Mapping[str, object]]
    linear_state_bytes: Mapping[str, int]
    execution_records: Tuple[TaskExecutionRecord, ...]


class _PathLink:
    __slots__ = (
        "previous",
        "duration_ns",
        "category",
        "node_key",
        "distance",
        "materialized_path",
    )

    def __init__(
        self,
        previous: Optional["_PathLink"],
        duration_ns: float,
        category: Optional[TaskCategory],
        node_key: Optional[str],
    ) -> None:
        self.previous = previous
        self.duration_ns = duration_ns
        self.category = category
        self.node_key = node_key
        self.distance = (previous.distance if previous is not None else 0.0) + duration_ns
        self.materialized_path: Optional[Tuple[str, ...]] = None


def _materialize_path(link: Optional[_PathLink]) -> Tuple[str, ...]:
    if link is not None and link.materialized_path is not None:
        return link.materialized_path
    reverse_keys: List[str] = []
    cursor = link
    prefix: Tuple[str, ...] = ()
    while cursor is not None:
        if cursor.materialized_path is not None:
            prefix = cursor.materialized_path
            break
        if cursor.node_key is not None:
            reverse_keys.append(cursor.node_key)
        cursor = cursor.previous
    reverse_keys.reverse()
    result = prefix + tuple(reverse_keys)
    if link is not None:
        link.materialized_path = result
    return result


def _better_path(
    candidate: Optional[_PathLink], current: Optional[_PathLink]
) -> Optional[_PathLink]:
    candidate_distance = candidate.distance if candidate is not None else 0.0
    current_distance = current.distance if current is not None else 0.0
    if candidate_distance > current_distance:
        return candidate
    if candidate_distance == current_distance:
        if _materialize_path(candidate) < _materialize_path(current):
            return candidate
    return current


def _extend_path(
    previous: Optional[_PathLink],
    duration_ns: float,
    category: TaskCategory,
    node_key: str,
) -> Optional[_PathLink]:
    candidate = _PathLink(previous, duration_ns, category, node_key)
    # Every graph node begins with a zero-distance empty path.  Preserve the
    # historical lexicographic rule: a non-empty zero path does not replace it.
    return _better_path(candidate, None)


def execute_cost_schedule(
    schedule: ScheduleIR,
    *,
    control: Optional[ExecutionControl] = None,
    compiled_executor: Optional[CompiledGraphExecutor] = None,
) -> ScheduleCostResult:
    """Execute and reduce a cohort schedule in one exact pass.

    Cancellation is checked at least every 256 scheduled tasks.  Progress is
    reported at the same granularity, avoiding callback overhead per task.
    """

    execution_control = control or ExecutionControl()
    execution_control.raise_if_cancelled()
    bulk_events = None
    if compiled_executor is not None:
        compiled_tasks, compiled_layout = compiled_executor._compiled_layout(
            schedule.tasks
        )
        kernel = UnifiedEventKernel(
            resource_capacities=getattr(schedule, "resource_capacities", {})
        )
        bulk_events = kernel._drain_prevalidated_compiled(
            compiled_tasks,
            compiled_layout,
            validate_tasks=False,
        )
    else:
        kernel = UnifiedEventKernel.from_closed_graph(
            schedule.tasks,
            resource_capacities=getattr(schedule, "resource_capacities", {}),
        )
    task_by_id = {task.task_id: task for task in schedule.tasks}
    total_tasks = len(schedule.tasks)
    execution_control.report("cohort_tasks", 0, total_tasks)
    resource_last_path: Dict[
        Tuple[str, int], Tuple[float, Optional[_PathLink]]
    ] = {}
    end_by_task: Dict[str, float] = {}
    end_path_by_task: Dict[str, Optional[_PathLink]] = {}
    # start, end, id, category, duration, metadata, sorted demands
    records: List[
        Tuple[
            float,
            float,
            str,
            TaskCategory,
            float,
            Mapping[str, Any],
            Tuple[ResourceDemand, ...],
        ]
    ] = []

    bulk_iterator = iter(bulk_events) if bulk_events is not None else None
    while bulk_iterator is not None or kernel.has_active_tasks:
        if bulk_iterator is not None:
            try:
                event = next(bulk_iterator)
            except StopIteration:
                bulk_iterator = None
                continue
        else:
            event = kernel.step()
            if event is None:
                kernel.assert_drained()
                break
        task = event.task
        task_id = task.task_id
        effective_ready_ns = event.effective_ready_ns
        start_ns = event.start_ns
        start_path: Optional[_PathLink] = None
        for dependency_id in sorted(task.dependencies):
            if dependency_id in end_path_by_task:
                start_path = _better_path(
                    end_path_by_task.get(dependency_id), start_path
                )

        sorted_demands = event.demands
        for demand in sorted_demands:
            lane = int(event.resource_lanes.get(demand.resource_id, 0))
            predecessor = resource_last_path.get((demand.resource_id, lane))
            if predecessor is not None and demand.resource_id in event.resource_predecessors:
                start_path = _better_path(predecessor[1], start_path)

        raw_task_duration = max(
            (demand.service_ns for demand in sorted_demands), default=0.0
        )
        end_ns = event.end_ns
        # TaskResult.duration_ns and ResourceInterval durations subtract
        # absolute timestamps.  Preserve that floating-point order so all
        # retention policies remain bit-for-bit equal.
        task_duration = end_ns - start_ns
        end_path = _extend_path(
            start_path, task_duration, task.category, task.task_id
        )
        end_by_task[task_id] = end_ns
        end_path_by_task[task_id] = end_path

        for demand in sorted_demands:
            interval_end_ns = start_ns + demand.service_ns
            interval_duration = interval_end_ns - start_ns
            interval_path = _extend_path(
                start_path,
                interval_duration,
                task.category,
                "{}@{}".format(task.task_id, demand.resource_id),
            )
            # A zero-duration resource interval is still the event kernel's
            # latest resource predecessor.  Store its empty path explicitly
            # so a later task cannot inherit an older non-zero interval.
            resource_last_path[(demand.resource_id, lane)] = (
                interval_end_ns,
                interval_path,
            )

        records.append(
            (
                start_ns,
                end_ns,
                task.task_id,
                task.category,
                task_duration,
                task.metadata,
                sorted_demands,
            )
        )
        if kernel.completed_count % 256 == 0:
            execution_control.raise_if_cancelled()
            execution_control.report(
                "cohort_tasks", kernel.completed_count, total_tasks
            )

    if kernel.completed_count != total_tasks:
        raise ValueError("schedule contains a dependency cycle")
    execution_control.raise_if_cancelled()

    ordered = sorted(records, key=lambda item: (item[0], item[1], item[2]))
    category_time: Dict[TaskCategory, float] = {}
    coverage: Dict[str, Dict[str, Any]] = {}
    state_bytes = {"read": 0, "write": 0, "prefetch": 0, "offload": 0}
    energy_pj = 0.0
    bytes_moved = 0
    for _start, _end, task_id, category, duration, metadata, demands in ordered:
        category_time[category] = category_time.get(category, 0.0) + duration
        task_energy = sum(demand.energy_pj for demand in demands)
        task_bytes = sum(demand.bytes_moved for demand in demands)
        energy_pj += task_energy
        bytes_moved += task_bytes
        event_kind = str(metadata.get("event_kind", ""))
        if event_kind.startswith("linear_state_"):
            state_kind = event_kind[len("linear_state_") :]
            if state_kind in state_bytes:
                state_bytes[state_kind] += int(metadata.get("bytes", 0))
        component = _coverage_component(task_id, metadata)
        if component:
            row = coverage.setdefault(
                component,
                {
                    "task_count": 0.0,
                    "operations": 0.0,
                    "bytes": 0.0,
                    "latency_ns": 0.0,
                    "energy_pj": 0.0,
                    "operator_ids": set(),
                    "tensor_ids": set(),
                },
            )
            row["task_count"] = float(row["task_count"]) + 1.0
            row["operations"] = float(row["operations"]) + float(
                metadata.get("analytical_ops", 0.0)
            )
            row["bytes"] = float(row["bytes"]) + float(task_bytes)
            row["latency_ns"] = float(row["latency_ns"]) + duration
            row["energy_pj"] = float(row["energy_pj"]) + float(task_energy)
            operator_id = metadata.get(
                "model_operator_id", metadata.get("operator_id")
            )
            tensor_id = metadata.get(
                "weight_tensor_id", metadata.get("tensor_id")
            )
            if operator_id:
                operator_ids = row["operator_ids"]
                if isinstance(operator_ids, set):
                    operator_ids.add(str(operator_id))
            if tensor_id:
                tensor_ids = row["tensor_ids"]
                if isinstance(tensor_ids, set):
                    tensor_ids.add(str(tensor_id))

    makespan_ns = max((record[1] for record in ordered), default=0.0)
    target_ids = sorted(
        record[2] for record in ordered if record[1] == makespan_ns
    )
    best_path: Optional[_PathLink] = None
    for task_id in target_ids:
        candidate = end_path_by_task.get(task_id)
        best_path = _better_path(candidate, best_path)
    critical_categories: Dict[TaskCategory, float] = {}
    cursor = best_path
    while cursor is not None:
        if cursor.category is not None:
            critical_categories[cursor.category] = (
                critical_categories.get(cursor.category, 0.0)
                + cursor.duration_ns
            )
        cursor = cursor.previous

    execution_control.report(
        "cohort_tasks",
        total_tasks,
        total_tasks,
        simulated_time_ns=makespan_ns,
    )
    return ScheduleCostResult(
        makespan_ns=makespan_ns,
        task_count=total_tasks,
        energy_pj=energy_pj,
        bytes_moved=bytes_moved,
        resource_busy_ns=dict(sorted(kernel.resource_busy_ns.items())),
        category_time_ns=category_time,
        critical_path_category_ns=critical_categories,
        coverage={
            name: {
                key: (
                    sorted(str(item) for item in value)
                    if isinstance(value, set)
                    else value
                )
                for key, value in values.items()
            }
            for name, values in sorted(coverage.items())
        },
        linear_state_bytes=state_bytes,
        execution_records=tuple(
            TaskExecutionRecord(
                task_id=task_id,
                start_ns=start_ns,
                end_ns=end_ns,
                category=category,
                dependencies=tuple(task_by_id[task_id].dependencies),
                metadata=metadata,
                demands=demands,
            )
            for start_ns, end_ns, task_id, category, _duration, metadata, demands in ordered
        ),
    )


def _coverage_component(task_id: str, metadata: Mapping[str, Any]) -> str:
    component = str(metadata.get("coverage_component", ""))
    event_kind = str(metadata.get("event_kind", ""))
    if component:
        return component
    if event_kind.startswith("linear_state_"):
        return "linear_state"
    if event_kind.startswith("kv_"):
        return "kv_cache"
    if event_kind.startswith("mtp_") or task_id.startswith("mtp"):
        return "mtp_policy"
    if "shared_expert" in task_id:
        return "shared_expert"
    if "expert" in task_id or "moe_router" in task_id:
        return "routed_expert"
    if "linear_" in task_id:
        return "linear_attention"
    if any(marker in task_id for marker in (".qkv", ".attention_", ".kv.")):
        return "full_attention"
    if ".mlp_" in task_id:
        return "dense_ffn"
    return ""


class ExactTemplateCache:
    """Bounded LRU of exact cohort cost templates.

    Keys must include every cost-relevant cohort field.  Values are returned
    unchanged; the cache never interpolates or approximates unseen inputs.
    """

    def __init__(self, max_entries: int = 512) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: "OrderedDict[Hashable, Mapping[str, object]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], Mapping[str, object]],
    ) -> Mapping[str, object]:
        cached = self._values.get(key)
        if cached is not None:
            self._values.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        value = factory()
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) > self.max_entries:
            self._values.popitem(last=False)
        return value

    @property
    def size(self) -> int:
        return len(self._values)


__all__ = [
    "CompiledGraphExecutor",
    "ExactTemplateCache",
    "ScheduleCostResult",
    "TaskExecutionRecord",
    "execute_cost_schedule",
]
