"""Exact, allocation-light primitives for topology-aware serving costs.

This module intentionally returns aggregate cost fields rather than a
``SimulationTrace``.  Online serving never exposes each cohort's internal
planner trace, so materializing thousands of ``TaskResult`` and
``ResourceInterval`` objects only to immediately reduce them is avoidable.
The scheduler and critical-path ordering are identical to ``engine.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple

from .contracts import ResourceDemand, TaskCategory, TaskSpec
from .engine import ScheduleIR, _resource_release_points
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
    original_task: Optional[TaskSpec] = None


@dataclass(frozen=True)
class ScheduleCostResult:
    makespan_ns: float
    task_count: int
    bytes_moved: int
    resource_busy_ns: Mapping[str, float]
    category_time_ns: Mapping[TaskCategory, float]
    critical_path_category_ns: Mapping[TaskCategory, float]
    coverage: Mapping[str, Mapping[str, object]]
    linear_state_bytes: Mapping[str, int]
    execution_records: Tuple[TaskExecutionRecord, ...]


class _PathLink:
    """One node of the critical-path forest, linked to its predecessor.

    ``depth`` and ``jump`` are skew-binary jump pointers (Myers): the jump
    target's depth is a function of the node's depth alone, so ancestors at
    any depth are reached in O(log depth) steps with one extra pointer per
    node. They replace materialized key tuples for the lexicographic
    tie-break (see ``_path_precedes``).
    """

    __slots__ = (
        "previous",
        "duration_ns",
        "category",
        "node_key",
        "distance",
        "depth",
        "jump",
    )

    def __init__(
        self,
        previous: Optional["_PathLink"],
        duration_ns: float,
        category: Optional[TaskCategory],
        node_key: str,
    ) -> None:
        self.previous = previous
        self.duration_ns = duration_ns
        self.category = category
        self.node_key = node_key
        self.distance = (previous.distance if previous is not None else 0.0) + duration_ns
        if previous is None:
            self.depth = 1
            self.jump = None
        else:
            self.depth = previous.depth + 1
            skip = previous.jump
            if skip is not None and previous.depth - skip.depth == skip.depth - (
                    skip.jump.depth if skip.jump is not None else 0):
                self.jump = skip.jump
            else:
                self.jump = previous


def _ancestor_at(link: _PathLink, depth: int) -> _PathLink:
    while link.depth > depth:
        skip = link.jump
        link = skip if skip is not None and skip.depth >= depth else link.previous
    return link


def _path_precedes(left: Optional[_PathLink], right: Optional[_PathLink]) -> bool:
    """Whether ``left``'s root-to-link node-key sequence sorts before ``right``'s.

    Exactly the tuple order of the two key sequences, without building them.
    Links are shared prefixes of one forest, so the first position where the
    sequences can differ is just below the lowest common ancestor. Distinct
    links built independently with the same key (and ancestors with the same
    keys) are compared position by position, so equal sequences stay equal.
    """
    if left is right or right is None:
        return False
    if left is None:
        return True
    if left.previous is right.previous:
        return left.node_key < right.node_key
    shared = min(left.depth, right.depth)
    low, high = _ancestor_at(left, shared), _ancestor_at(right, shared)
    if low is high:
        # One sequence is a proper prefix of the other.
        return left.depth < right.depth
    low_child = high_child = low
    while low is not high:
        if low.jump is not high.jump:
            low, high = low.jump, high.jump
        else:
            low_child, high_child = low, high
            low, high = low.previous, high.previous
    if low_child.node_key != high_child.node_key:
        return low_child.node_key < high_child.node_key
    for depth in range(low_child.depth + 1, shared + 1):
        left_key = _ancestor_at(left, depth).node_key
        right_key = _ancestor_at(right, depth).node_key
        if left_key != right_key:
            return left_key < right_key
    return left.depth < right.depth


def _better_path(
    candidate: Optional[_PathLink], current: Optional[_PathLink]
) -> Optional[_PathLink]:
    candidate_distance = candidate.distance if candidate is not None else 0.0
    current_distance = current.distance if current is not None else 0.0
    if candidate_distance > current_distance:
        return candidate
    if candidate_distance == current_distance:
        if _path_precedes(candidate, current):
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


def compact_execution_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep scalar/audit execution facts, without expanded physical traces."""
    record_metadata = {
        key: metadata[key]
        for key in (
            "persistent_request_buffers", "persistent_buffer_ids",
            "hbf_host_staging",
            "native_kv_work", "source_f32_kv", "native_rope_source_contract",
            "mmvf_source_work", "mmq_source_work", "native_get_rows_storage",
            "kernel_prediction", "qwen35_attention_work", "qwen35_tensor_geometry",
            "qwen35_shared_input_work", "qwen35_attention_geometry",
            "final_layer_output_selection", "output_selection_tensor_geometry",
            "runtime_phase", "instruction_class", "instruction_count",
            "transaction_kind", "transaction_batches", "aggregation",
            "analytical_ops", "bytes", "cohort_id", "coverage_component", "model_operator_id",
            "memory_direction", "phase", "submission_count",
            "dma_engine_resource_id", "event_kind",
            "source_stage", "target_stage",
            "execution_component", "gpu_consumer_component_id",
            "gpu_consumer_request_ids", "layer_id", "modeled_memory_write_bytes",
            "opaque_device_fence", "operator_id", "operator_invocation_group_id",
            "orchestration_stage",
            "output_source_component_id", "output_transfer_source_component_id",
            "physical_invocation_group_ids", "physical_memory_config",
            "physical_execution_by_owner",
            "request_ids", "resource_directions", "serving_cohort_complete",
            "serving_output_stage", "target_component", "tensor_id",
            "transfer_execution", "weight_tensor_id",
            "native_cost_model", "operator_class", "rank", "expert_index",
            "cache_input_bits", "cache_output_bits", "cache_artifact_format",
            "conversion_instructions_priced", "timing_completeness", "unpriced_terms",
        )
        if key in metadata
    }
    if metadata.get("cost_model"):
        # Stage refinement only checks presence; retaining the full
        # kernel model per task needlessly repeats a large mapping.
        record_metadata["cost_model"] = True
        cost = metadata["cost_model"]
        if isinstance(cost, Mapping):
            # Native prediction needs the actual estimator identity even for
            # nonphysical phases whose expanded metadata is discarded. Keep
            # only formula qualification inputs, never kernel traces or a
            # user assertion that an arbitrary function is monotone.
            small = {key: cost[key] for key in (
                "model", "profile", "device", "launch_only", "tensor_dtype",
                "quantized_weight_path", "source_partial_service_ns",
                "quantized_shape_efficiency", "mmvq_work", "mmq_work",
            ) if key in cost}
            cache = cost.get("cache")
            if isinstance(cache, Mapping):
                small["cache"] = {key: cache[key] for key in (
                    "cache_model", "logical_read_bytes", "logical_write_bytes",
                    "cross_invocation_cache_state",
                ) if key in cache}
            record_metadata["native_cost_model"] = small
    by_owner = metadata.get("physical_execution_by_owner")
    if isinstance(by_owner, Mapping):
        excluded = {"resource_intervals", "resource_interval_payloads", "resource_last_intervals",
                    "physical_resource_intervals", "physical_resource_last_intervals", "physical_resource_lane_completions"}
        record_metadata["physical_execution_by_owner"] = {
            owner: {key: value for key, value in values.items() if key not in excluded}
            for owner, values in by_owner.items() if isinstance(values, Mapping)
        }
    return record_metadata


def _unresolved_physical_task(task: TaskSpec) -> TaskSpec:
    """Remove only explicitly marked endpoint construction-preview billing."""
    if task.metadata.get("physical_execution_role") != "construction_preview":
        return task
    owner_resource = task.metadata.get("physical_resource_id")
    if (not owner_resource or task.metadata.get("memory_accesses") is None
            or not any(d.resource_id == owner_resource for d in task.demands)
            or "physical_demands_resource_ids" in task.metadata):
        raise ValueError("endpoint construction preview lacks its original physical access contract")
    metadata = {key: value for key, value in task.metadata.items()
                if key not in {"physical_execution_by_owner", "physical_execution_role"}}
    return replace(task, metadata=metadata, demands=tuple(
        replace(demand, service_ns=0.0, bytes_moved=0)
        if demand.resource_id == owner_resource else demand for demand in task.demands))


def execute_cost_schedule(
    schedule: ScheduleIR,
    *,
    control: Optional[ExecutionControl] = None,
    compiled_executor: Optional[CompiledGraphExecutor] = None,
    retain_task_metadata: bool = True,
    defer_physical_execution: bool = False,
) -> ScheduleCostResult:
    """Execute and reduce a cohort schedule in one exact pass.

    Cancellation is checked at least every 256 scheduled tasks.  Progress is
    reported at the same granularity, avoiding callback overhead per task.
    """

    execution_control = control or ExecutionControl()
    execution_control.raise_if_cancelled()
    # Live physical cohorts execute once against their persistent controllers.
    # A preview only organizes the authored host/compute/link envelope. Staged
    # HBF also depends on writes from earlier cohorts, so a fresh preview core
    # cannot execute those reads. Preserve the complete physical tasks for
    # the live kernel; provisional envelope times are not physical timings.
    original_physical_tasks = {task.task_id: _unresolved_physical_task(task) for task in schedule.tasks
        if task.metadata.get("physical_memory_config") is not None
        or task.metadata.get("stateful_l2") is not None}
    submitted_tasks = schedule.tasks
    if defer_physical_execution:
        submitted_tasks = tuple(replace(task, metadata={
            key: value for key, value in task.metadata.items()
            if not key.startswith(("physical_", "hbf_"))
            and key not in {"memory_accesses", "stateful_l2", "l2_execution"}})
            if task.task_id in original_physical_tasks else task for task in schedule.tasks)
    bulk_events = None
    compiled_preview = None
    # Deferred physical previews contain only their authored analytical
    # envelope.  Keep their exact event/reduction results while executing the
    # resident position graph; owner aliases alone do not require incremental
    # dispatch.  Real physical/stateful previews still consume events one at
    # a time so completed DRAM/L2 payloads can be reclaimed promptly.
    has_physical_or_stateful = bool(
        schedule.resource_owners
        or any(
            task.metadata.get("stateful_l2") is not None
            or task.metadata.get("physical_memory_config") is not None
            for task in submitted_tasks
        )
    )
    if compiled_executor is not None and (
        defer_physical_execution or not has_physical_or_stateful
    ):
        compiled_tasks, compiled_layout = compiled_executor._compiled_layout(
            submitted_tasks
        )
        kernel = UnifiedEventKernel(
            resource_capacities=schedule.resource_capacities,
            resource_owners=schedule.resource_owners,
            capture_physical_details=retain_task_metadata,
        )
        if defer_physical_execution:
            # Bind the resident graph after its observer exists, then reduce
            # each committed event before the next dispatch.  Cancellation
            # therefore retains exactly the incremental execution prefix.
            compiled_preview = (compiled_tasks, compiled_layout)
        else:
            bulk_events = kernel._drain_prevalidated_compiled(
                compiled_tasks,
                compiled_layout,
                validate_tasks=False,
            )
    else:
        kernel = UnifiedEventKernel.from_closed_graph(
            submitted_tasks,
            resource_capacities=schedule.resource_capacities,
            resource_owners=schedule.resource_owners,
            capture_physical_details=retain_task_metadata,
        )
    total_tasks = len(schedule.tasks)
    progress_stage = "cohort_preview_tasks" if defer_physical_execution else "cohort_tasks"
    if compiled_preview is None:
        execution_control.report(progress_stage, 0, total_tasks)
    resource_last_path: Dict[
        Tuple[str, int], Tuple[float, Optional[_PathLink]]
    ] = {}
    end_path_by_task: Dict[str, Optional[_PathLink]] = {}
    # start, end, id, category, duration, metadata, sorted demands, dependencies
    records: List[
        Tuple[
            float,
            float,
            str,
            TaskCategory,
            float,
            Mapping[str, Any],
            Tuple[ResourceDemand, ...],
            Tuple[str, ...],
        ]
    ] = []

    def observe_event(event):
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
            info = event.resource_predecessors.get(demand.resource_id)
            if info is None:
                continue
            if info["end_ns"] > start_ns:
                continue  # Internal Core wait is already in this task's span.
            lane = int(info.get("lane", event.resource_lanes.get(demand.resource_id, 0)))
            predecessor_resource = str(info["resource_id"])
            predecessor = resource_last_path.get((predecessor_resource, lane))
            if predecessor is not None:
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
        end_path_by_task[task_id] = end_path

        for resource_id, lane, interval_end_ns in _resource_release_points(
                task, sorted_demands, start_ns, event.resource_lanes):
            interval_duration = interval_end_ns - start_ns
            interval_path = _extend_path(
                start_path,
                interval_duration,
                task.category,
                "{}@{}".format(task.task_id, resource_id),
            )
            # A zero-duration resource interval is still the event kernel's
            # latest resource predecessor.  Store its empty path explicitly
            # so a later task cannot inherit an older non-zero interval.
            resource_last_path[(resource_id, lane)] = (
                interval_end_ns,
                interval_path,
            )

        record_metadata = task.metadata if retain_task_metadata else compact_execution_metadata(task.metadata)
        records.append(
            (
                start_ns,
                end_ns,
                task.task_id,
                task.category,
                task_duration,
                record_metadata,
                sorted_demands,
                tuple(task.dependencies),
            )
        )
        if kernel.completed_count % 256 == 0:
            execution_control.raise_if_cancelled()
            execution_control.report(
                progress_stage, kernel.completed_count, total_tasks
            )

    if compiled_preview is not None:
        compiled_tasks, compiled_layout = compiled_preview
        kernel._drain_prevalidated_compiled(
            compiled_tasks, compiled_layout, validate_tasks=False,
            event_observer=observe_event,
            on_submitted=lambda: execution_control.report(progress_stage, 0, total_tasks),
        )
    else:
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
            observe_event(event)

    if kernel.completed_count != total_tasks:
        raise ValueError("schedule contains a dependency cycle")
    execution_control.raise_if_cancelled()

    ordered = sorted(records, key=lambda item: (item[0], item[1], item[2]))
    category_time: Dict[TaskCategory, float] = {}
    coverage: Dict[str, Dict[str, Any]] = {}
    state_bytes = {"read": 0, "write": 0, "prefetch": 0, "offload": 0}
    bytes_moved = 0
    for _start, _end, task_id, category, duration, metadata, demands, _dependencies in ordered:
        category_time[category] = category_time.get(category, 0.0) + duration
        task_bytes = sum(demand.bytes_moved for demand in demands)
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
                dependencies=dependencies,
                metadata=metadata,
                demands=demands,
                original_task=original_physical_tasks.get(task_id),
            )
            for start_ns, end_ns, task_id, category, _duration, metadata, demands, dependencies in ordered
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
    "compact_execution_metadata",
]
