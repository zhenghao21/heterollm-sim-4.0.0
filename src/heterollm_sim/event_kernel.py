"""Version-4 deterministic event kernel.

The kernel is the only implementation of dependency readiness, resource
contention, and deterministic task ordering.  Exact traces, bounded streaming
traces, and aggregate serving costs are observers over the same event stream;
they are retention policies rather than independent simulators.
"""

from __future__ import annotations

import bisect
import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .contracts import ResourceDemand, TaskSpec
from .runtime_ir import (
    KernelCompletion,
    RUNTIME_ACTION_METADATA_KEY,
    RUNTIME_CONTROLLER_BATCH_METADATA_KEY,
    RUNTIME_INSTRUCTION_BATCH_METADATA_KEY,
    RUNTIME_PAYLOAD_METADATA_KEY,
    RUNTIME_PHASE_METADATA_KEY,
    RUNTIME_SEQUENCE_METADATA_KEY,
    decode_aggregate_metadata,
    decode_runtime_metadata,
)


_RUNTIME_ORDER_METADATA_KEYS = (
    RUNTIME_ACTION_METADATA_KEY,
    RUNTIME_PAYLOAD_METADATA_KEY,
    RUNTIME_PHASE_METADATA_KEY,
    RUNTIME_SEQUENCE_METADATA_KEY,
)
_RUNTIME_AGGREGATE_METADATA_KEYS = (
    RUNTIME_INSTRUCTION_BATCH_METADATA_KEY,
    RUNTIME_CONTROLLER_BATCH_METADATA_KEY,
)


def _validated_phase_sequence(task: TaskSpec) -> Tuple[int, int]:
    """Validate reserved runtime metadata without taxing ordinary DAG tasks."""

    metadata = task.metadata
    if type(metadata) is dict:
        if metadata.keys().isdisjoint(_RUNTIME_ORDER_METADATA_KEYS):
            return (0, 0)
    elif isinstance(metadata, Mapping) and not any(
        key in metadata for key in _RUNTIME_ORDER_METADATA_KEYS
    ):
        return (0, 0)
    _action, _payload, phase, sequence = decode_runtime_metadata(task)
    return (phase, sequence)


def _validate_aggregate_metadata(task: TaskSpec) -> None:
    metadata = task.metadata
    if isinstance(metadata, Mapping) and not any(
        key in metadata for key in _RUNTIME_AGGREGATE_METADATA_KEYS
    ):
        return
    decode_aggregate_metadata(task)


@dataclass(frozen=True)
class KernelEvent:
    """One realized task emitted by :class:`UnifiedEventKernel`."""

    task: TaskSpec
    start_ns: float
    end_ns: float
    dependency_ready_ns: float
    effective_ready_ns: float
    demands: Tuple[ResourceDemand, ...]
    resource_predecessors: Mapping[str, Mapping[str, object]]
    # Chosen physical lane for each demand resource.  This is emitted even
    # when no predecessor exists, allowing incremental observers to maintain
    # one critical-path frontier per lane instead of one per resource.
    resource_lanes: Mapping[str, int] = field(
        default_factory=dict, compare=False
    )

    @property
    def queue_wait_ns(self) -> float:
        return self.start_ns - self.effective_ready_ns

    @property
    def service_ns(self) -> float:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class SubmissionReceipt:
    """Result of one fully validated, atomically committed submission."""

    task_ids: Tuple[str, ...]
    retained_dependencies: Tuple[str, ...]
    active_task_count: int


ReadyKey = Tuple[float, float, float, float, str]
InternalReadyKey = Tuple[float, float, float, float, int, int, str]
StaticReadyKey = Tuple[float, float, float, int, int, str]
ResourceGroup = Tuple[str, ...]
CompiledStructureKey = Tuple[
    int,
    Tuple[Tuple[int, ...], ...],
    Tuple[int, ...],
    Tuple[str, ...],
    Tuple[Tuple[str, ...], ...],
]
GlobalReadyEntry = Tuple[
    float,
    float,
    float,
    float,
    int,
    int,
    str,
    ResourceGroup,
    int,
]


@dataclass(frozen=True)
class CompiledGraphLayout:
    """Validated, task-id-independent layout for one closed task graph.

    Dynamic costs and task payloads stay on the real :class:`TaskSpec`
    instances.  Only scheduling structure is compiled: dependency positions,
    task-id ordering, finite release times, demand resource ordering, sorted
    dependents, resource groups, and demand sorting permutations.
    """

    task_count: int
    dependency_positions: Tuple[Tuple[int, ...], ...]
    task_id_order: Tuple[int, ...]
    earliest_start_hex: Tuple[str, ...]
    demand_resource_ids: Tuple[Tuple[str, ...], ...]
    dependent_positions: Tuple[Tuple[int, ...], ...]
    resource_groups: Tuple[ResourceGroup, ...]
    demand_order: Tuple[Tuple[int, ...], ...]
    root_positions: Tuple[int, ...]
    indegree_template: Tuple[int, ...]
    resource_ids: Tuple[str, ...]
    demand_resource_indices: Tuple[Tuple[int, ...], ...]
    group_indices: Tuple[int, ...]
    group_count: int

    @classmethod
    def compile(cls, tasks: Sequence[TaskSpec]) -> "CompiledGraphLayout":
        """Fully validate and compile one closed graph."""

        chunk = tuple(tasks)
        validate_task_graph(chunk)
        return cls._from_validated_tasks(chunk)

    @classmethod
    def _from_validated_tasks(
        cls,
        tasks: Tuple[TaskSpec, ...],
    ) -> "CompiledGraphLayout":
        positions = {task.task_id: position for position, task in enumerate(tasks)}
        dependency_positions = tuple(
            tuple(positions[dependency_id] for dependency_id in task.dependencies)
            for task in tasks
        )
        task_id_order = tuple(
            sorted(range(len(tasks)), key=lambda position: tasks[position].task_id)
        )
        demand_resource_ids = tuple(
            tuple(demand.resource_id for demand in task.demands) for task in tasks
        )
        dependent_lists: List[List[int]] = [[] for _task in tasks]
        for dependent_position, dependencies in enumerate(dependency_positions):
            for dependency_position in dependencies:
                dependent_lists[dependency_position].append(dependent_position)
        dependent_positions = tuple(
            tuple(
                sorted(
                    positions_for_dependency,
                    key=lambda position: tasks[position].task_id,
                )
            )
            for positions_for_dependency in dependent_lists
        )
        resource_groups = tuple(
            tuple(sorted(set(resource_ids)))
            for resource_ids in demand_resource_ids
        )
        demand_order = tuple(
            tuple(
                sorted(
                    range(len(resource_ids)),
                    key=lambda position: resource_ids[position],
                )
            )
            for resource_ids in demand_resource_ids
        )
        resource_ids = tuple(
            sorted(
                {
                    resource_id
                    for task_resource_ids in demand_resource_ids
                    for resource_id in task_resource_ids
                }
            )
        )
        resource_position = {
            resource_id: position
            for position, resource_id in enumerate(resource_ids)
        }
        unique_resource_groups = tuple(sorted(set(resource_groups)))
        group_position = {
            resource_group: position
            for position, resource_group in enumerate(unique_resource_groups)
        }
        return cls(
            task_count=len(tasks),
            dependency_positions=dependency_positions,
            task_id_order=task_id_order,
            earliest_start_hex=tuple(
                float(task.earliest_start_ns).hex() for task in tasks
            ),
            demand_resource_ids=demand_resource_ids,
            dependent_positions=dependent_positions,
            resource_groups=resource_groups,
            demand_order=demand_order,
            root_positions=tuple(
                position
                for position, dependencies in enumerate(dependency_positions)
                if not dependencies
            ),
            indegree_template=tuple(
                len(dependencies) for dependencies in dependency_positions
            ),
            resource_ids=resource_ids,
            demand_resource_indices=tuple(
                tuple(
                    resource_position[resource_id]
                    for resource_id in task_resource_ids
                )
                for task_resource_ids in demand_resource_ids
            ),
            group_indices=tuple(
                group_position[resource_group]
                for resource_group in resource_groups
            ),
            group_count=len(unique_resource_groups),
        )

    def matches_structure(self, tasks: Sequence[TaskSpec]) -> bool:
        """Return whether ``tasks`` match every indexed layout field.

        ``CompiledGraphLayout`` is frozen, but its constructor is public and
        nested tuple fields can still be supplied incorrectly.  Callers use
        this predicate before indexing those fields during a compiled
        submission, so validate the complete shape and the derived values in
        one fail-closed pass.  In particular, checking only dependencies and
        demand resource ids is insufficient: ``resource_groups``,
        ``demand_order``, ``dependent_positions``, and ``root_positions`` are
        all consumed later by the kernel commit path.
        """

        def valid_position(value: object) -> bool:
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < self.task_count
            )

        try:
            task_count = self.task_count
            if (
                isinstance(task_count, bool)
                or not isinstance(task_count, int)
                or task_count < 0
                or len(tasks) != task_count
            ):
                return False

            # Compiled layouts are immutable tuples produced by ``compile``.
            # Requiring the same shape also prevents a mutable list supplied
            # through the public constructor from changing after this guard.
            outer_fields = (
                self.dependency_positions,
                self.task_id_order,
                self.earliest_start_hex,
                self.demand_resource_ids,
                self.dependent_positions,
                self.resource_groups,
                self.demand_order,
                self.indegree_template,
                self.demand_resource_indices,
                self.group_indices,
            )
            if any(
                not isinstance(field, tuple) or len(field) != task_count
                for field in outer_fields
            ):
                return False
            if not isinstance(self.root_positions, tuple):
                return False
            if not isinstance(self.resource_ids, tuple):
                return False
            if (
                isinstance(self.group_count, bool)
                or not isinstance(self.group_count, int)
                or self.group_count < 0
            ):
                return False
            if any(
                not isinstance(value, str)
                or not math.isfinite(float.fromhex(value))
                for value in self.earliest_start_hex
            ):
                return False

            positions: Dict[str, int] = {}
            for position, task in enumerate(tasks):
                task_id = task.task_id
                if task_id in positions:
                    return False
                positions[task_id] = position

            expected_task_id_order = tuple(
                sorted(
                    range(task_count),
                    key=lambda position: tasks[position].task_id,
                )
            )
            if any(
                not valid_position(position)
                for position in self.task_id_order
            ) or len(set(self.task_id_order)) != task_count:
                return False
            if self.task_id_order != expected_task_id_order:
                return False

            expected_dependency_positions: List[Tuple[int, ...]] = []
            expected_demand_resource_ids: List[Tuple[str, ...]] = []
            expected_resource_groups: List[ResourceGroup] = []
            expected_demand_order: List[Tuple[int, ...]] = []
            for position, task in enumerate(tasks):
                dependency_positions = tuple(
                    positions[dependency_id]
                    for dependency_id in task.dependencies
                )
                demand_resource_ids = tuple(
                    demand.resource_id for demand in task.demands
                )
                expected_dependency_positions.append(dependency_positions)
                expected_demand_resource_ids.append(demand_resource_ids)
                expected_resource_groups.append(
                    tuple(sorted(set(demand_resource_ids)))
                )
                expected_demand_order.append(
                    tuple(
                        sorted(
                            range(len(demand_resource_ids)),
                            key=lambda demand_position: demand_resource_ids[
                                demand_position
                            ],
                        )
                    )
                )

                field_values = (
                    self.dependency_positions[position],
                    self.demand_resource_ids[position],
                    self.resource_groups[position],
                    self.demand_order[position],
                )
                if any(
                    not isinstance(field, tuple) for field in field_values
                ):
                    return False
                if any(
                    not valid_position(dependency_position)
                    for dependency_position in self.dependency_positions[
                        position
                    ]
                ):
                    return False
                if any(
                    not valid_position(demand_position)
                    for demand_position in self.demand_order[position]
                ):
                    return False
                if (
                    len(set(self.demand_order[position]))
                    != len(self.demand_order[position])
                ):
                    return False

            expected_dependent_lists: List[List[int]] = [
                [] for _task in tasks
            ]
            for dependent_position, dependencies in enumerate(
                expected_dependency_positions
            ):
                for dependency_position in dependencies:
                    expected_dependent_lists[dependency_position].append(
                        dependent_position
                    )
            expected_dependent_positions = tuple(
                tuple(
                    sorted(
                        dependent_positions,
                        key=lambda dependent_position: tasks[
                            dependent_position
                        ].task_id,
                    )
                )
                for dependent_positions in expected_dependent_lists
            )
            if any(
                any(
                    not valid_position(dependent_position)
                    for dependent_position in dependent_positions
                )
                or len(set(dependent_positions)) != len(dependent_positions)
                for dependent_positions in self.dependent_positions
            ):
                return False

            expected_root_positions = tuple(
                position
                for position, dependencies in enumerate(
                    expected_dependency_positions
                )
                if not dependencies
            )
            expected_resource_ids = tuple(
                sorted(
                    {
                        resource_id
                        for task_resource_ids in expected_demand_resource_ids
                        for resource_id in task_resource_ids
                    }
                )
            )
            expected_resource_position = {
                resource_id: position
                for position, resource_id in enumerate(expected_resource_ids)
            }
            expected_demand_resource_indices = tuple(
                tuple(
                    expected_resource_position[resource_id]
                    for resource_id in task_resource_ids
                )
                for task_resource_ids in expected_demand_resource_ids
            )
            expected_unique_resource_groups = tuple(
                sorted(set(expected_resource_groups))
            )
            expected_group_position = {
                resource_group: position
                for position, resource_group in enumerate(
                    expected_unique_resource_groups
                )
            }
            expected_group_indices = tuple(
                expected_group_position[resource_group]
                for resource_group in expected_resource_groups
            )
            if any(
                not valid_position(position) for position in self.root_positions
            ) or len(set(self.root_positions)) != len(self.root_positions):
                return False

            if (
                self.dependency_positions
                != tuple(expected_dependency_positions)
                or self.demand_resource_ids
                != tuple(expected_demand_resource_ids)
                or self.dependent_positions != expected_dependent_positions
                or self.resource_groups != tuple(expected_resource_groups)
                or self.demand_order != tuple(expected_demand_order)
                or self.root_positions != expected_root_positions
                or self.indegree_template
                != tuple(
                    len(dependencies)
                    for dependencies in expected_dependency_positions
                )
                or self.resource_ids != expected_resource_ids
                or self.demand_resource_indices
                != expected_demand_resource_indices
                or self.group_indices != expected_group_indices
                or self.group_count != len(expected_unique_resource_groups)
            ):
                return False
            return True
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # A malformed public-constructor layout is simply not reusable.
            # The compiled submission path turns this ``False`` into its
            # normal pre-commit ``ValueError`` without mutating kernel state.
            return False

    @staticmethod
    def structure_key(
        tasks: Sequence[TaskSpec],
    ) -> Optional[CompiledStructureKey]:
        """Return the complete id-independent fields consumed by a layout.

        The tuple is exact rather than digest-based.  Resource groups, demand
        order, dependent positions, and roots are deterministic derivations
        of these fields, so an equal key selects the same compiled structure
        without rebuilding those derived collections on every cache hit.
        """

        positions: Dict[str, int] = {}
        for position, task in enumerate(tasks):
            if task.task_id in positions:
                return None
            positions[task.task_id] = position
        try:
            dependencies = tuple(
                tuple(
                    positions[dependency_id]
                    for dependency_id in task.dependencies
                )
                for task in tasks
            )
        except KeyError:
            return None
        task_id_order = tuple(
            sorted(
                range(len(tasks)),
                key=lambda position: tasks[position].task_id,
            )
        )
        demand_resource_ids = tuple(
            tuple(demand.resource_id for demand in task.demands)
            for task in tasks
        )
        return (
            len(tasks),
            dependencies,
            task_id_order,
            tuple(float(task.earliest_start_ns).hex() for task in tasks),
            demand_resource_ids,
        )

    @property
    def structure_signature(self) -> CompiledStructureKey:
        """Return this validated layout's exact lookup signature."""

        return (
            self.task_count,
            self.dependency_positions,
            self.task_id_order,
            self.earliest_start_hex,
            self.demand_resource_ids,
        )

    def matches(self, tasks: Sequence[TaskSpec]) -> bool:
        """Return whether real tasks match every compiled scheduling field."""

        if not self.matches_structure(tasks):
            return False
        for position, task in enumerate(tasks):
            earliest_start_ns = float(task.earliest_start_ns)
            if (
                not math.isfinite(earliest_start_ns)
                or earliest_start_ns.hex() != self.earliest_start_hex[position]
            ):
                return False
        return True


class CompiledGraphExecutor:
    """Bounded per-run LRU for strictly matching compiled graph layouts."""

    def __init__(self, max_layouts: int = 8) -> None:
        if max_layouts <= 0:
            raise ValueError("max_layouts must be positive")
        self.max_layouts = max_layouts
        self._layouts: "OrderedDict[CompiledStructureKey, CompiledGraphLayout]" = (
            OrderedDict()
        )
        self._hits = 0
        self._misses = 0
        self._fallbacks = 0

    def create_kernel(self, tasks: Sequence[TaskSpec]) -> "UnifiedEventKernel":
        """Create a kernel, reusing a layout only after an exact match."""

        chunk, layout = self._compiled_layout(tasks)
        return UnifiedEventKernel._from_compiled_layout(chunk, layout)

    def _compiled_layout(
        self, tasks: Sequence[TaskSpec]
    ) -> Tuple[Tuple[TaskSpec, ...], CompiledGraphLayout]:
        """Return the identity-checked layout while preserving LRU accounting."""

        chunk = tuple(tasks)
        structure_key = CompiledGraphLayout.structure_key(chunk)
        layout = (
            self._layouts.get(structure_key)
            if structure_key is not None
            else None
        )
        if layout is not None:
            self._layouts.move_to_end(structure_key)
            self._hits += 1
            return chunk, layout

        had_cached_layout = bool(self._layouts)
        self._misses += 1
        if had_cached_layout:
            self._fallbacks += 1
        layout = CompiledGraphLayout.compile(chunk)
        structure_key = layout.structure_signature
        self._layouts[structure_key] = layout
        self._layouts.move_to_end(structure_key)
        if len(self._layouts) > self.max_layouts:
            self._layouts.popitem(last=False)
        return chunk, layout

    @property
    def stats(self) -> Mapping[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "fallbacks": self._fallbacks,
            "size": len(self._layouts),
        }


def task_start_ns(
    demands: Sequence[ResourceDemand],
    effective_ready_ns: float,
    resource_available: Mapping[str, float],
) -> float:
    demand_iterator = iter(demands)
    try:
        first_demand = next(demand_iterator)
    except StopIteration:
        resources_ready_ns = 0.0
    else:
        resource_available_get = resource_available.get
        resources_ready_ns = resource_available_get(
            first_demand.resource_id,
            0.0,
        )
        for demand in demand_iterator:
            available_ns = resource_available_get(demand.resource_id, 0.0)
            if available_ns > resources_ready_ns:
                resources_ready_ns = available_ns
    return max(effective_ready_ns, resources_ready_ns)


def validate_task_graph(tasks: Sequence[TaskSpec]) -> Dict[str, TaskSpec]:
    """Validate one closed task graph and return its stable id index."""

    tasks_by_id: Dict[str, TaskSpec] = {}
    for task in tasks:
        _validate_task_time(task)
        if task.task_id in tasks_by_id:
            raise ValueError("duplicate task_id: {}".format(task.task_id))
        tasks_by_id[task.task_id] = task

    missing: Dict[str, Tuple[str, ...]] = {}
    for task in tasks:
        absent = tuple(
            dependency
            for dependency in task.dependencies
            if dependency not in tasks_by_id
        )
        if absent:
            missing[task.task_id] = absent
    if missing:
        details = ", ".join(
            "{}->{}".format(task_id, list(dependencies))
            for task_id, dependencies in sorted(missing.items())
        )
        raise ValueError("missing dependencies: {}".format(details))

    _validate_acyclic(tasks_by_id)
    return tasks_by_id


def _validate_task_time(task: TaskSpec) -> None:
    if not math.isfinite(float(task.earliest_start_ns)):
        raise ValueError(
            "task {} earliest_start_ns must be finite".format(task.task_id)
        )


def _validate_acyclic(tasks_by_id: Mapping[str, TaskSpec]) -> None:
    indegree = {
        task_id: len(task.dependencies) for task_id, task in tasks_by_id.items()
    }
    dependents: Dict[str, List[str]] = {
        task_id: [] for task_id in tasks_by_id
    }
    for task_id, task in tasks_by_id.items():
        for dependency_id in task.dependencies:
            dependents[dependency_id].append(task_id)

    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    visited_count = 0
    while ready:
        task_id = heapq.heappop(ready)
        visited_count += 1
        for dependent_id in sorted(dependents[task_id]):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                heapq.heappush(ready, dependent_id)
    if visited_count != len(tasks_by_id):
        raise ValueError("schedule contains a dependency cycle")


class UnifiedEventKernel:
    """Incremental deterministic multi-resource event scheduler.

    ``add_tasks`` accepts either a complete graph or a bounded chunk whose
    external dependencies were completed earlier.  This lets static exact,
    static streaming, and online cohort execution share the same scheduler.
    """

    def __init__(
        self,
        resource_capacities: Optional[Mapping[str, int]] = None,
    ) -> None:
        capacities = dict(resource_capacities or {})
        for resource_id, capacity in capacities.items():
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("resource capacity ids must not be empty")
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity <= 0
            ):
                raise ValueError("resource capacities must be positive integers")
        self._tasks: Dict[str, TaskSpec] = {}
        self._indegree: Dict[str, int] = {}
        self._dependents: Dict[str, List[str]] = {}
        self._dependency_ready: Dict[str, float] = {}
        self._completed_end: Dict[str, float] = {}
        self._completion_leases: Dict[str, int] = {}
        self._seen_ids: Set[str] = set()
        self._compact_seen_namespaces: Dict[str, Tuple[str, ...]] = {}
        self._compact_seen_base_id_sets: Dict[str, FrozenSet[str]] = {}
        self._compact_seen_base_id_cache: Dict[
            int, Tuple[Tuple[str, ...], FrozenSet[str]]
        ] = {}
        self._compact_seen_prefix_lengths: Set[int] = set()
        self._compact_seen_sorted_prefixes: List[str] = []

        self.resource_capacities: Dict[str, int] = capacities
        self.resource_available: Dict[str, float] = {}
        self.resource_last_interval: Dict[str, Dict[str, object]] = {}
        self.resource_busy_ns: Dict[str, float] = {}
        self.resource_queue_wait_ns: Dict[str, float] = {}
        self.resource_task_count: Dict[str, int] = {}
        self.total_queue_wait_ns = 0.0
        self.total_service_ns = 0.0
        self.completed_count = 0
        self.makespan_ns = 0.0

        self._resource_lane_available: Dict[str, List[float]] = {
            resource_id: [0.0] * capacity
            for resource_id, capacity in capacities.items()
        }
        self._resource_lane_last_interval: Dict[
            Tuple[str, int], Dict[str, object]
        ] = {}

        self._ready_by_group: Dict[ResourceGroup, List[StaticReadyKey]] = {}
        self._group_versions: Dict[ResourceGroup, int] = {}
        self._ready_heap: List[GlobalReadyEntry] = []
        self._resource_groups: Dict[str, ResourceGroup] = {}
        self._sorted_demands: Dict[str, Tuple[ResourceDemand, ...]] = {}
        self._phase_sequence: Dict[str, Tuple[int, int]] = {}

    @classmethod
    def from_closed_graph(
        cls,
        tasks: Sequence[TaskSpec],
        *,
        resource_capacities: Optional[Mapping[str, int]] = None,
    ) -> "UnifiedEventKernel":
        chunk = tuple(tasks)
        layout = CompiledGraphLayout.compile(chunk)
        return cls._from_compiled_layout(
            chunk,
            layout,
            resource_capacities=resource_capacities,
        )

    @classmethod
    def from_compiled_graph(
        cls,
        tasks: Sequence[TaskSpec],
        layout: CompiledGraphLayout,
        *,
        resource_capacities: Optional[Mapping[str, int]] = None,
    ) -> "UnifiedEventKernel":
        """Load a graph only when it exactly matches a validated layout."""

        chunk = tuple(tasks)
        if not layout.matches(chunk):
            raise ValueError("task graph does not match compiled layout")
        return cls._from_compiled_layout(
            chunk,
            layout,
            resource_capacities=resource_capacities,
        )

    @classmethod
    def _from_compiled_layout(
        cls,
        tasks: Tuple[TaskSpec, ...],
        layout: CompiledGraphLayout,
        *,
        resource_capacities: Optional[Mapping[str, int]] = None,
    ) -> "UnifiedEventKernel":
        kernel = cls(resource_capacities=resource_capacities)
        phase_sequence: Dict[str, Tuple[int, int]] = {}
        for task in tasks:
            _validate_aggregate_metadata(task)
            phase_sequence[task.task_id] = _validated_phase_sequence(task)
        task_ids = tuple(task.task_id for task in tasks)
        kernel._tasks = {task.task_id: task for task in tasks}
        kernel._seen_ids = set(task_ids)
        kernel._indegree = {
            task_ids[position]: len(layout.dependency_positions[position])
            for position in range(layout.task_count)
        }
        kernel._dependents = {
            task_ids[position]: [
                task_ids[dependent_position]
                for dependent_position in layout.dependent_positions[position]
            ]
            for position in range(layout.task_count)
        }
        kernel._dependency_ready = {task_id: 0.0 for task_id in task_ids}
        kernel._resource_groups = {
            task_ids[position]: layout.resource_groups[position]
            for position in range(layout.task_count)
        }
        kernel._sorted_demands = {
            task_ids[position]: tuple(
                tasks[position].demands[demand_position]
                for demand_position in layout.demand_order[position]
            )
            for position in range(layout.task_count)
        }
        kernel._phase_sequence = phase_sequence

        groups_to_refresh: Set[ResourceGroup] = set()
        for position in layout.root_positions:
            changed_group = kernel._add_ready_task(task_ids[position])
            if changed_group is not None:
                groups_to_refresh.add(changed_group)
        for resource_group in sorted(groups_to_refresh):
            kernel._refresh_group(resource_group)
        return kernel

    @property
    def active_task_count(self) -> int:
        return len(self._tasks)

    @property
    def has_active_tasks(self) -> bool:
        return bool(self._tasks)

    @property
    def completed_end_ns(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._completed_end))

    @property
    def completion_leases(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._completion_leases))

    @property
    def metrics(self) -> Mapping[str, object]:
        return {
            "completed_count": self.completed_count,
            "makespan_ns": self.makespan_ns,
            "queue_wait_ns": self.total_queue_wait_ns,
            "service_ns": self.total_service_ns,
            "resource_capacities": dict(sorted(self.resource_capacities.items())),
            "resource_service_ns": dict(self.resource_busy_ns),
            "resource_queue_wait_ns": dict(self.resource_queue_wait_ns),
            "resource_task_count": dict(self.resource_task_count),
        }

    def ensure_resource_capacities(
        self,
        capacities: Mapping[str, int],
    ) -> None:
        """Declare untouched resource lanes before appending another DAG.

        A live V4 run may bootstrap its control plane first and append serving
        work later.  Capacity can be added for resources the bootstrap never
        touched; changing the capacity of a resource with scheduling history
        would rewrite that history and therefore fails closed.
        """

        for resource_id, capacity in dict(capacities).items():
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("resource capacity ids must not be empty")
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity <= 0
            ):
                raise ValueError("resource capacities must be positive integers")
            existing = self.resource_capacities.get(resource_id)
            lanes = self._resource_lane_available.get(resource_id)
            if existing is not None:
                if existing != capacity:
                    raise ValueError(
                        "resource capacity mismatch for {}: {} != {}".format(
                            resource_id,
                            existing,
                            capacity,
                        )
                    )
                continue
            if lanes is not None:
                if len(lanes) != capacity:
                    raise ValueError(
                        "cannot change capacity for used resource {}".format(
                            resource_id
                        )
                    )
            else:
                self._resource_lane_available[resource_id] = [0.0] * capacity
            self.resource_capacities[resource_id] = capacity

    def _resource_ready_ns(self, resource_id: str) -> float:
        # ``resource_available`` is maintained as the earliest free lane for
        # every touched resource.  Untouched declared lanes are all zero, so a
        # missing entry has the same value without scanning the lane list.
        return self.resource_available.get(resource_id, 0.0)

    def _ready_key(self, task_id: str) -> InternalReadyKey:
        task = self._tasks[task_id]
        dependency_ready_ns = self._dependency_ready[task_id]
        earliest_start_ns = task.earliest_start_ns
        effective_ready_ns = dependency_ready_ns
        if earliest_start_ns > effective_ready_ns:
            effective_ready_ns = earliest_start_ns
        demands = task.demands
        if demands:
            demand_iterator = iter(demands)
            resources_ready_ns = self._resource_ready_ns(
                next(demand_iterator).resource_id
            )
            for demand in demand_iterator:
                available_ns = self._resource_ready_ns(demand.resource_id)
                if available_ns > resources_ready_ns:
                    resources_ready_ns = available_ns
        else:
            resources_ready_ns = 0.0
        actual_start_ns = effective_ready_ns
        if resources_ready_ns > actual_start_ns:
            actual_start_ns = resources_ready_ns
        phase, sequence = self._phase_sequence[task_id]
        return (
            actual_start_ns,
            effective_ready_ns,
            dependency_ready_ns,
            earliest_start_ns,
            phase,
            sequence,
            task_id,
        )

    def _static_ready_key(self, task_id: str) -> StaticReadyKey:
        task = self._tasks[task_id]
        dependency_ready_ns = self._dependency_ready[task_id]
        effective_ready_ns = dependency_ready_ns
        if task.earliest_start_ns > effective_ready_ns:
            effective_ready_ns = task.earliest_start_ns
        phase, sequence = self._phase_sequence[task_id]
        return (
            effective_ready_ns,
            dependency_ready_ns,
            task.earliest_start_ns,
            phase,
            sequence,
            task_id,
        )

    def _resource_group(self, task_id: str) -> ResourceGroup:
        return self._resource_groups[task_id]

    def _refresh_group(self, resource_group: ResourceGroup) -> None:
        version = self._group_versions.get(resource_group, 0) + 1
        self._group_versions[resource_group] = version
        group_ready = self._ready_by_group.get(resource_group)
        if not group_ready:
            return
        task_id = group_ready[0][-1]
        heapq.heappush(
            self._ready_heap,
            (*self._ready_key(task_id), resource_group, version),
        )

    def _add_ready_task(self, task_id: str) -> Optional[ResourceGroup]:
        resource_group = self._resource_group(task_id)
        group_ready = self._ready_by_group.setdefault(resource_group, [])
        previous_top = group_ready[0] if group_ready else None
        heapq.heappush(group_ready, self._static_ready_key(task_id))
        if previous_top is None or group_ready[0] != previous_top:
            return resource_group
        return None

    def _has_seen_task_id(self, task_id: str) -> bool:
        """Return whether ``task_id`` is present in explicit or compact history."""

        if task_id in self._seen_ids:
            return True
        for prefix_length in self._compact_seen_prefix_lengths:
            if prefix_length >= len(task_id):
                continue
            base_task_ids = self._compact_seen_base_id_sets.get(
                task_id[:prefix_length]
            )
            if (
                base_task_ids is not None
                and task_id[prefix_length:] in base_task_ids
            ):
                return True
        return False

    def _compact_seen_collision(
        self,
        prefix: str,
        base_task_ids: Tuple[str, ...],
        base_task_id_set: FrozenSet[str],
    ) -> Optional[str]:
        """Return one compact/compact duplicate id, if namespaces overlap."""

        prefix_length = len(prefix)
        for existing_length in self._compact_seen_prefix_lengths:
            if existing_length >= prefix_length:
                continue
            existing_prefix = prefix[:existing_length]
            existing_base_ids = self._compact_seen_base_id_sets.get(
                existing_prefix
            )
            if existing_base_ids is None:
                continue
            suffix_prefix = prefix[existing_length:]
            for base_task_id in base_task_ids:
                if suffix_prefix + base_task_id in existing_base_ids:
                    return prefix + base_task_id

        descendant_index = bisect.bisect_right(
            self._compact_seen_sorted_prefixes,
            prefix,
        )
        while descendant_index < len(self._compact_seen_sorted_prefixes):
            existing_prefix = self._compact_seen_sorted_prefixes[
                descendant_index
            ]
            if not existing_prefix.startswith(prefix):
                break
            suffix_prefix = existing_prefix[prefix_length:]
            for existing_base_id in self._compact_seen_namespaces[
                existing_prefix
            ]:
                base_task_id = suffix_prefix + existing_base_id
                if base_task_id in base_task_id_set:
                    return prefix + base_task_id
            descendant_index += 1
        return None

    def _compact_seen_base_ids(
        self,
        base_task_ids: Sequence[str],
    ) -> Tuple[int, Tuple[str, ...], FrozenSet[str]]:
        """Return validated shared base ids and their membership set."""

        if isinstance(base_task_ids, (str, bytes)):
            raise TypeError("compact seen base task ids must be a sequence")
        if type(base_task_ids) is tuple:
            cache_key = id(base_task_ids)
            cached = self._compact_seen_base_id_cache.get(cache_key)
            if cached is not None and cached[0] is base_task_ids:
                return (cache_key, *cached)
            base_ids = base_task_ids
        else:
            base_ids = tuple(base_task_ids)
            cache_key = id(base_ids)

        seen_base_ids: Set[str] = set()
        for base_task_id in base_ids:
            if not isinstance(base_task_id, str) or not base_task_id:
                raise ValueError(
                    "compact seen base task ids must be non-empty strings"
                )
            if base_task_id in seen_base_ids:
                raise ValueError(
                    "compact seen namespace contains duplicate task ids"
                )
            seen_base_ids.add(base_task_id)

        return (cache_key, base_ids, frozenset(seen_base_ids))

    def _prepare_compact_seen_namespace(
        self,
        prefix: str,
        base_task_ids: Sequence[str],
    ) -> Tuple[str, int, Tuple[str, ...], FrozenSet[str]]:
        """Validate one compact namespace without mutating kernel history.

        This private path is for trusted replay code that already owns the
        immutable base id sequence.  The kernel records that sequence by
        namespace instead of expanding every historical id into ``_seen_ids``.
        """

        if not isinstance(prefix, str) or not prefix:
            raise ValueError(
                "compact seen namespace prefix must be a non-empty string"
            )
        if prefix in self._compact_seen_namespaces:
            raise ValueError(
                "compact seen namespace already registered: {}".format(prefix)
            )
        cache_key, base_ids, base_id_set = self._compact_seen_base_ids(
            base_task_ids
        )
        for task_id in self._seen_ids:
            if (
                task_id.startswith(prefix)
                and task_id[len(prefix) :] in base_id_set
            ):
                raise ValueError("duplicate task_id: {}".format(task_id))
        duplicate = self._compact_seen_collision(
            prefix,
            base_ids,
            base_id_set,
        )
        if duplicate is not None:
            raise ValueError("duplicate task_id: {}".format(duplicate))

        return (prefix, cache_key, base_ids, base_id_set)

    def _commit_compact_seen_namespace(
        self,
        prepared: Tuple[str, int, Tuple[str, ...], FrozenSet[str]],
    ) -> None:
        """Commit a namespace prepared inside one closed kernel operation."""

        prefix, cache_key, base_ids, base_id_set = prepared

        self._compact_seen_namespaces[prefix] = base_ids
        self._compact_seen_base_id_sets[prefix] = base_id_set
        self._compact_seen_base_id_cache[cache_key] = (base_ids, base_id_set)
        self._compact_seen_prefix_lengths.add(len(prefix))
        bisect.insort(self._compact_seen_sorted_prefixes, prefix)

    def _register_compact_seen_namespace(
        self,
        prefix: str,
        base_task_ids: Sequence[str],
    ) -> None:
        """Atomically validate and register compact historical task ids."""

        prepared = self._prepare_compact_seen_namespace(
            prefix,
            base_task_ids,
        )
        self._commit_compact_seen_namespace(prepared)

    def submit(
        self,
        tasks: Iterable[TaskSpec],
        *,
        retain_dependencies: Iterable[str] = (),
    ) -> SubmissionReceipt:
        """Fully validate a dynamic chunk before changing any kernel state."""

        chunk = tuple(tasks)
        retained = tuple(retain_dependencies)

        # Phase 1: validation and complete staging.  Nothing below this block
        # mutates scheduler, resource, completion, or metric state.
        phase_sequence: Dict[str, Tuple[int, int]] = {}
        resource_groups: Dict[str, ResourceGroup] = {}
        sorted_demands: Dict[str, Tuple[ResourceDemand, ...]] = {}
        for task in chunk:
            if not isinstance(task, TaskSpec):
                raise TypeError("kernel submissions must contain TaskSpec instances")
            for field_name, value in (
                ("task_id", task.task_id),
                ("request_id", task.request_id),
                ("name", task.name),
            ):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        "task {} must be a non-empty string".format(field_name)
                    )
            for dependency_id in task.dependencies:
                if not isinstance(dependency_id, str) or not dependency_id:
                    raise ValueError(
                        "task dependencies must be non-empty strings"
                    )
            for demand in task.demands:
                if not isinstance(demand, ResourceDemand):
                    raise TypeError(
                        "task demands must contain ResourceDemand instances"
                    )
                if not isinstance(demand.resource_id, str) or not demand.resource_id:
                    raise ValueError(
                        "demand resource_id must be a non-empty string"
                    )
            _validate_task_time(task)
            phase_sequence[task.task_id] = _validated_phase_sequence(task)
            _validate_aggregate_metadata(task)
            resource_groups[task.task_id] = tuple(
                sorted({demand.resource_id for demand in task.demands})
            )
            sorted_demands[task.task_id] = tuple(
                sorted(task.demands, key=lambda demand: demand.resource_id)
            )

        chunk_ids = {task.task_id for task in chunk}
        if len(chunk_ids) != len(chunk):
            raise ValueError("task chunk contains duplicate task ids")
        for task_id in chunk_ids:
            if self._has_seen_task_id(task_id):
                raise ValueError("duplicate task_id: {}".format(task_id))

        for dependency_id in retained:
            if not isinstance(dependency_id, str) or not dependency_id:
                raise ValueError(
                    "retained dependency ids must be non-empty strings"
                )
        if len(set(retained)) != len(retained):
            raise ValueError("retain_dependencies contains duplicate ids")
        for dependency_id in retained:
            if self._completion_leases.get(dependency_id, 0) <= 0:
                raise ValueError(
                    "cannot retain unavailable completion {}".format(dependency_id)
                )

        for task in chunk:
            for dependency_id in task.dependencies:
                if (
                    dependency_id not in chunk_ids
                    and dependency_id not in self._tasks
                    and dependency_id not in self._completed_end
                ):
                    raise ValueError(
                        "task {} has unavailable dependency {}".format(
                            task.task_id, dependency_id
                        )
                    )

        local_indegree = {
            task.task_id: sum(
                dependency_id in chunk_ids
                for dependency_id in task.dependencies
            )
            for task in chunk
        }
        local_dependents: Dict[str, List[str]] = {
            task_id: [] for task_id in chunk_ids
        }
        for task in chunk:
            for dependency_id in task.dependencies:
                if dependency_id in chunk_ids:
                    local_dependents[dependency_id].append(task.task_id)
        local_ready = [
            task_id for task_id, degree in local_indegree.items() if degree == 0
        ]
        heapq.heapify(local_ready)
        local_visited = 0
        while local_ready:
            task_id = heapq.heappop(local_ready)
            local_visited += 1
            for dependent_id in sorted(local_dependents[task_id]):
                local_indegree[dependent_id] -= 1
                if local_indegree[dependent_id] == 0:
                    heapq.heappush(local_ready, dependent_id)
        if local_visited != len(chunk):
            raise ValueError("schedule contains a dependency cycle")

        staged_indegree: Dict[str, int] = {}
        staged_ready: Dict[str, float] = {}
        staged_external_dependents: Dict[str, List[str]] = {}
        for task in chunk:
            unresolved = 0
            dependency_ready_ns = 0.0
            for dependency_id in task.dependencies:
                completed_ns = self._completed_end.get(dependency_id)
                if completed_ns is not None:
                    dependency_ready_ns = max(dependency_ready_ns, completed_ns)
                else:
                    unresolved += 1
                    staged_external_dependents.setdefault(dependency_id, []).append(
                        task.task_id
                    )
            staged_indegree[task.task_id] = unresolved
            staged_ready[task.task_id] = dependency_ready_ns

        # Construct every sorted dependent list before commit.  Together with
        # the string-only id/resource validation above, phase 2 contains only
        # assignments and heap operations over homogeneous, staged keys.
        staged_sorted_dependents = {
            dependency_id: sorted(
                (
                    *self._dependents.get(dependency_id, ()),
                    *dependent_ids,
                )
            )
            for dependency_id, dependent_ids in staged_external_dependents.items()
        }

        # Phase 2: commit the already validated/staged chunk.
        for dependency_id in retained:
            self._completion_leases[dependency_id] += 1
        for task in chunk:
            task_id = task.task_id
            self._tasks[task_id] = task
            self._dependents.setdefault(task_id, [])
            self._dependency_ready[task_id] = staged_ready[task_id]
            self._indegree[task_id] = staged_indegree[task_id]
            self._seen_ids.add(task_id)
            self._resource_groups[task_id] = resource_groups[task_id]
            self._sorted_demands[task_id] = sorted_demands[task_id]
            self._phase_sequence[task_id] = phase_sequence[task_id]

        for dependency_id, dependent_ids in staged_sorted_dependents.items():
            self._dependents[dependency_id] = dependent_ids

        groups_to_refresh: Set[ResourceGroup] = set()
        for task in chunk:
            if staged_indegree[task.task_id] == 0:
                changed_group = self._add_ready_task(task.task_id)
                if changed_group is not None:
                    groups_to_refresh.add(changed_group)
        for resource_group in sorted(groups_to_refresh):
            self._refresh_group(resource_group)

        return SubmissionReceipt(
            task_ids=tuple(task.task_id for task in chunk),
            retained_dependencies=retained,
            active_task_count=len(self._tasks),
        )

    def add_tasks(self, tasks: Iterable[TaskSpec]) -> None:
        """Backward-compatible wrapper over atomic :meth:`submit`."""

        self.submit(tasks)

    def submit_compiled(
        self,
        tasks: Iterable[TaskSpec],
        layout: CompiledGraphLayout,
        *,
        retain_dependencies: Iterable[str] = (),
    ) -> SubmissionReceipt:
        """Atomically append a closed graph through a validated layout.

        Release times and service quantities remain live on ``tasks``.  The
        compiled object supplies only dependency positions, dependent order,
        resource groups, and demand permutations that were validated when the
        layout was created.  A structural mismatch fails before kernel state
        changes.
        """

        return self._submit_compiled(
            tasks,
            layout,
            retain_dependencies=retain_dependencies,
            validate_structure=True,
        )

    def _submit_prevalidated_compiled(
        self,
        tasks: Iterable[TaskSpec],
        layout: CompiledGraphLayout,
        *,
        retain_dependencies: Iterable[str] = (),
    ) -> SubmissionReceipt:
        """Append an internally keyed layout without re-deriving its graph.

        This private entry point is reserved for replay caches that already
        validated the complete id-independent structure and retained the
        exact structural key alongside ``layout``.  Public and otherwise
        untrusted callers continue through :meth:`submit_compiled`, which
        performs the full fail-closed structural predicate before commit.
        """

        return self._submit_compiled(
            tasks,
            layout,
            retain_dependencies=retain_dependencies,
            validate_structure=False,
        )

    def _submit_compiled(
        self,
        tasks: Iterable[TaskSpec],
        layout: CompiledGraphLayout,
        *,
        retain_dependencies: Iterable[str],
        validate_structure: bool,
    ) -> SubmissionReceipt:
        if not isinstance(layout, CompiledGraphLayout):
            raise TypeError("layout must be a CompiledGraphLayout")
        chunk = tuple(tasks)
        retained = tuple(retain_dependencies)
        phase_sequence: Dict[str, Tuple[int, int]] = {}
        for task in chunk:
            if not isinstance(task, TaskSpec):
                raise TypeError("kernel submissions must contain TaskSpec instances")
            for field_name, value in (
                ("task_id", task.task_id),
                ("request_id", task.request_id),
                ("name", task.name),
            ):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        "task {} must be a non-empty string".format(field_name)
                    )
            for dependency_id in task.dependencies:
                if not isinstance(dependency_id, str) or not dependency_id:
                    raise ValueError(
                        "task dependencies must be non-empty strings"
                    )
            for demand in task.demands:
                if not isinstance(demand, ResourceDemand):
                    raise TypeError(
                        "task demands must contain ResourceDemand instances"
                    )
                if not isinstance(demand.resource_id, str) or not demand.resource_id:
                    raise ValueError(
                        "demand resource_id must be a non-empty string"
                    )
            _validate_task_time(task)
            phase_sequence[task.task_id] = _validated_phase_sequence(task)
            _validate_aggregate_metadata(task)

        if validate_structure and not layout.matches_structure(chunk):
            raise ValueError("task graph does not match compiled structure")
        task_ids = tuple(task.task_id for task in chunk)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task chunk contains duplicate task ids")
        for task_id in task_ids:
            if self._has_seen_task_id(task_id):
                raise ValueError("duplicate task_id: {}".format(task_id))

        for dependency_id in retained:
            if not isinstance(dependency_id, str) or not dependency_id:
                raise ValueError(
                    "retained dependency ids must be non-empty strings"
                )
        if len(set(retained)) != len(retained):
            raise ValueError("retain_dependencies contains duplicate ids")
        for dependency_id in retained:
            if self._completion_leases.get(dependency_id, 0) <= 0:
                raise ValueError(
                    "cannot retain unavailable completion {}".format(
                        dependency_id
                    )
                )

        sorted_demands = tuple(
            tuple(task.demands[index] for index in layout.demand_order[position])
            for position, task in enumerate(chunk)
        )
        dependent_ids = tuple(
            tuple(task_ids[position] for position in positions)
            for positions in layout.dependent_positions
        )

        for dependency_id in retained:
            self._completion_leases[dependency_id] += 1
        for position, task in enumerate(chunk):
            task_id = task_ids[position]
            self._tasks[task_id] = task
            self._dependents[task_id] = list(dependent_ids[position])
            self._dependency_ready[task_id] = 0.0
            self._indegree[task_id] = len(layout.dependency_positions[position])
            self._seen_ids.add(task_id)
            self._resource_groups[task_id] = layout.resource_groups[position]
            self._sorted_demands[task_id] = sorted_demands[position]
            self._phase_sequence[task_id] = phase_sequence[task_id]

        groups_to_refresh: Set[ResourceGroup] = set()
        for position in layout.root_positions:
            changed_group = self._add_ready_task(task_ids[position])
            if changed_group is not None:
                groups_to_refresh.add(changed_group)
        for resource_group in sorted(groups_to_refresh):
            self._refresh_group(resource_group)

        return SubmissionReceipt(
            task_ids=task_ids,
            retained_dependencies=retained,
            active_task_count=len(self._tasks),
        )

    def _drain_prevalidated_compiled(
        self,
        tasks: Sequence[TaskSpec],
        layout: CompiledGraphLayout,
        *,
        validate_tasks: bool = True,
    ) -> Tuple[KernelEvent, ...]:
        """Execute one internally keyed closed graph with array-local state.

        The persistent resource lanes, predecessor intervals, metrics, and
        completion leases are updated exactly as repeated ``step`` calls do.
        Only transient task-id dictionaries and versioned heap entries move
        to position-indexed local arrays.  The caller must own an exact
        structural key for ``layout`` and a drained kernel; public submission
        continues through the fully validating incremental path.
        """

        if self.has_active_tasks:
            raise ValueError("prevalidated compiled drain requires an idle kernel")
        chunk = tuple(tasks)
        if len(chunk) != layout.task_count:
            raise ValueError("task graph does not match compiled structure")
        task_ids = tuple(task.task_id for task in chunk)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task chunk contains duplicate task ids")
        duplicate = next(
            (task_id for task_id in task_ids if self._has_seen_task_id(task_id)),
            None,
        )
        if duplicate is not None:
            raise ValueError("duplicate task_id: {}".format(duplicate))

        phase_sequence: List[Tuple[int, int]] = []
        for task in chunk:
            if validate_tasks:
                _validate_task_time(task)
            phase_sequence.append(_validated_phase_sequence(task))
            if validate_tasks:
                _validate_aggregate_metadata(task)

        indegree = list(layout.indegree_template)
        dependency_ready = [0.0] * len(chunk)
        sorted_demands = tuple(
            tuple(
                chunk[position].demands[demand_position]
                for demand_position in layout.demand_order[position]
            )
            for position in range(len(chunk))
        )
        sorted_demand_resource_indices = tuple(
            tuple(
                layout.demand_resource_indices[position][demand_position]
                for demand_position in layout.demand_order[position]
            )
            for position in range(len(chunk))
        )
        resource_available_ns = [
            self.resource_available.get(resource_id, 0.0)
            for resource_id in layout.resource_ids
        ]
        ready_by_group: List[
            List[Tuple[float, float, float, int, int, str, int]]
        ] = [[] for _group in range(layout.group_count)]
        group_versions = [0] * layout.group_count
        ready_heap: List[
            Tuple[
                float,
                float,
                float,
                float,
                int,
                int,
                str,
                ResourceGroup,
                int,
                int,
            ]
        ] = []

        def static_key(
            position: int,
        ) -> Tuple[float, float, float, int, int, str, int]:
            task = chunk[position]
            dependency_ready_ns = dependency_ready[position]
            effective_ready_ns = max(
                dependency_ready_ns,
                task.earliest_start_ns,
            )
            phase, sequence = phase_sequence[position]
            return (
                effective_ready_ns,
                dependency_ready_ns,
                task.earliest_start_ns,
                phase,
                sequence,
                task.task_id,
                position,
            )

        def ready_key(
            position: int,
        ) -> Tuple[float, float, float, float, int, int, str]:
            task = chunk[position]
            dependency_ready_ns = dependency_ready[position]
            effective_ready_ns = max(
                dependency_ready_ns,
                task.earliest_start_ns,
            )
            resources_ready_ns = 0.0
            for resource_index in layout.demand_resource_indices[position]:
                available_ns = resource_available_ns[resource_index]
                if available_ns > resources_ready_ns:
                    resources_ready_ns = available_ns
            phase, sequence = phase_sequence[position]
            return (
                max(effective_ready_ns, resources_ready_ns),
                effective_ready_ns,
                dependency_ready_ns,
                task.earliest_start_ns,
                phase,
                sequence,
                task.task_id,
            )

        def refresh_group(group_index: int) -> None:
            version = group_versions[group_index] + 1
            group_versions[group_index] = version
            group_ready = ready_by_group[group_index]
            if not group_ready:
                return
            position = group_ready[0][-1]
            heapq.heappush(
                ready_heap,
                (
                    *ready_key(position),
                    layout.resource_groups[position],
                    version,
                    position,
                ),
            )

        def add_ready(position: int) -> Optional[int]:
            group_index = layout.group_indices[position]
            group_ready = ready_by_group[group_index]
            previous_top = group_ready[0] if group_ready else None
            heapq.heappush(group_ready, static_key(position))
            if previous_top is None or group_ready[0] != previous_top:
                return group_index
            return None

        initial_groups: List[int] = []
        initial_group_seen = [False] * layout.group_count
        for position in layout.root_positions:
            changed_group = add_ready(position)
            if (
                changed_group is not None
                and not initial_group_seen[changed_group]
            ):
                initial_group_seen[changed_group] = True
                initial_groups.append(changed_group)
        initial_groups.sort()
        for group_index in initial_groups:
            refresh_group(group_index)

        events: List[KernelEvent] = []
        completed_positions: List[int] = []
        resource_last_interval = self.resource_last_interval
        resource_available = self.resource_available
        resource_busy_ns = self.resource_busy_ns
        resource_queue_wait_ns = self.resource_queue_wait_ns
        resource_task_count = self.resource_task_count
        completed_end = self._completed_end
        completion_leases = self._completion_leases
        resource_ids = layout.resource_ids
        resource_count = len(resource_ids)
        resource_busy_values = [
            resource_busy_ns.get(resource_id, 0.0)
            for resource_id in resource_ids
        ]
        resource_queue_wait_values = [
            resource_queue_wait_ns.get(resource_id, 0.0)
            for resource_id in resource_ids
        ]
        resource_task_count_values = [
            resource_task_count.get(resource_id, 0)
            for resource_id in resource_ids
        ]
        resource_lanes: List[Optional[List[float]]] = [None] * resource_count
        resource_lanes_missing = [False] * resource_count
        resource_lane_intervals: List[
            Optional[List[Optional[Dict[str, object]]]]
        ] = [None] * resource_count
        resource_last_values: List[Optional[Dict[str, object]]] = (
            [None] * resource_count
        )
        resource_touched = [False] * resource_count
        resource_touch_order: List[int] = []
        lane_touch_order: List[Tuple[int, int]] = []
        lane_available = self._resource_lane_available
        lane_last_interval = self._resource_lane_last_interval
        completed_count = self.completed_count
        total_queue_wait_ns = self.total_queue_wait_ns
        total_service_ns = self.total_service_ns
        makespan_ns = self.makespan_ns
        state_committed = False

        def commit_state() -> None:
            """Reduce indexed state while preserving persistent key order."""

            nonlocal state_committed
            if state_committed:
                return
            state_committed = True
            for resource_index in resource_touch_order:
                resource_id = resource_ids[resource_index]
                lanes = resource_lanes[resource_index]
                interval = resource_last_values[resource_index]
                if lanes is None or interval is None:
                    continue
                if resource_lanes_missing[resource_index]:
                    lane_available[resource_id] = lanes
                resource_available[resource_id] = resource_available_ns[
                    resource_index
                ]
                resource_last_interval[resource_id] = interval
                resource_busy_ns[resource_id] = resource_busy_values[
                    resource_index
                ]
                resource_queue_wait_ns[resource_id] = (
                    resource_queue_wait_values[resource_index]
                )
                resource_task_count[resource_id] = (
                    resource_task_count_values[resource_index]
                )
            for resource_index, lane_index in lane_touch_order:
                intervals = resource_lane_intervals[resource_index]
                if intervals is None:
                    continue
                interval = intervals[lane_index]
                if interval is not None:
                    lane_last_interval[
                        (resource_ids[resource_index], lane_index)
                    ] = interval
            for position, event in zip(completed_positions, events):
                task_id = task_ids[position]
                completed_end[task_id] = event.end_ns
                completion_leases[task_id] = 1
                self._dependents[task_id] = [
                    task_ids[dependent_position]
                    for dependent_position in layout.dependent_positions[
                        position
                    ]
                ]
            self.completed_count = completed_count
            self.total_queue_wait_ns = total_queue_wait_ns
            self.total_service_ns = total_service_ns
            self.makespan_ns = makespan_ns

        while ready_heap:
            queued = heapq.heappop(ready_heap)
            version = queued[8]
            position = queued[9]
            group_index = layout.group_indices[position]
            if version != group_versions[group_index]:
                continue
            group_ready = ready_by_group[group_index]
            if not group_ready or group_ready[0][-1] != position:
                continue
            current_key = ready_key(position)
            if current_key != queued[:7]:
                refresh_group(group_index)
                continue
            (
                start_ns,
                effective_ready_ns,
                dependency_ready_ns,
                _earliest_start_ns,
                _phase,
                _sequence,
                task_id,
            ) = current_key
            task = chunk[position]
            demands = sorted_demands[position]
            if not math.isfinite(start_ns) or any(
                not math.isfinite(start_ns + demand.service_ns)
                for demand in demands
            ):
                commit_state()
                raise ValueError(
                    "task {} timing exceeds finite simulation range".format(
                        task_id
                    )
                )
            heapq.heappop(group_ready)

            resource_predecessors: Dict[str, Dict[str, object]] = {}
            chosen_lanes: Dict[str, int] = {}
            end_ns = start_ns
            queue_wait_ns = start_ns - effective_ready_ns
            demand_resource_indices = sorted_demand_resource_indices[position]
            for demand_index, demand in enumerate(demands):
                resource_index = demand_resource_indices[demand_index]
                resource_id = resource_ids[resource_index]
                demand_end_ns = start_ns + demand.service_ns
                lanes = resource_lanes[resource_index]
                intervals = resource_lane_intervals[resource_index]
                if lanes is None:
                    lanes = lane_available.get(resource_id)
                    if lanes is None:
                        lanes = [0.0] * self.resource_capacities.get(
                            resource_id, 1
                        )
                        resource_lanes_missing[resource_index] = True
                    resource_lanes[resource_index] = lanes
                    intervals = [None] * len(lanes)
                    resource_lane_intervals[resource_index] = intervals
                if not resource_touched[resource_index]:
                    resource_touched[resource_index] = True
                    resource_touch_order.append(resource_index)
                if len(lanes) == 1:
                    lane_index = 0
                    available_ns = lanes[0]
                else:
                    lane_index = 0
                    available_ns = lanes[0]
                    for candidate_lane in range(1, len(lanes)):
                        candidate_available_ns = lanes[candidate_lane]
                        if candidate_available_ns < available_ns:
                            lane_index = candidate_lane
                            available_ns = candidate_available_ns
                chosen_lanes[resource_id] = lane_index
                if intervals is None:
                    raise AssertionError("resource interval lanes were not initialized")
                interval = intervals[lane_index]
                previous_interval = interval
                if previous_interval is None:
                    previous_interval = lane_last_interval.get(
                        (resource_id, lane_index)
                    )
                if previous_interval is not None and available_ns == start_ns:
                    resource_predecessors[resource_id] = dict(previous_interval)
                if demand_end_ns > end_ns:
                    end_ns = demand_end_ns
                if interval is None:
                    interval = {
                        "task_id": task_id,
                        "resource_id": resource_id,
                        "start_ns": start_ns,
                        "end_ns": demand_end_ns,
                    }
                    if len(lanes) > 1:
                        interval["lane"] = lane_index
                    intervals[lane_index] = interval
                    lane_touch_order.append((resource_index, lane_index))
                else:
                    interval["task_id"] = task_id
                    interval["resource_id"] = resource_id
                    interval["start_ns"] = start_ns
                    interval["end_ns"] = demand_end_ns
                lanes[lane_index] = demand_end_ns
                resource_available_ns[resource_index] = (
                    demand_end_ns if len(lanes) == 1 else min(lanes)
                )
                resource_last_values[resource_index] = interval
                resource_busy_values[resource_index] = (
                    resource_busy_values[resource_index]
                    + demand.service_ns
                )
                resource_queue_wait_values[resource_index] = (
                    resource_queue_wait_values[resource_index]
                    + queue_wait_ns
                )
                resource_task_count_values[resource_index] = (
                    resource_task_count_values[resource_index] + 1
                )

            completed_count += 1
            total_queue_wait_ns += queue_wait_ns
            total_service_ns += end_ns - start_ns
            if end_ns > makespan_ns:
                makespan_ns = end_ns

            groups_to_refresh = [group_index]
            for dependent_position in layout.dependent_positions[position]:
                if end_ns > dependency_ready[dependent_position]:
                    dependency_ready[dependent_position] = end_ns
                indegree[dependent_position] -= 1
                if indegree[dependent_position] == 0:
                    changed_group = add_ready(dependent_position)
                    if (
                        changed_group is not None
                        and changed_group not in groups_to_refresh
                    ):
                        groups_to_refresh.append(changed_group)
            if len(groups_to_refresh) == 2:
                if groups_to_refresh[1] < groups_to_refresh[0]:
                    groups_to_refresh.reverse()
            elif len(groups_to_refresh) > 2:
                groups_to_refresh.sort()
            for changed_group in groups_to_refresh:
                refresh_group(changed_group)

            events.append(
                KernelEvent(
                    task=task,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    dependency_ready_ns=dependency_ready_ns,
                    effective_ready_ns=effective_ready_ns,
                    demands=demands,
                    resource_predecessors=resource_predecessors,
                    resource_lanes=chosen_lanes,
                )
            )
            completed_positions.append(position)

        if len(events) != len(chunk):
            commit_state()
            raise ValueError("schedule contains a dependency cycle")
        commit_state()
        self._seen_ids.update(task_ids)
        return tuple(events)

    def peek_ready_key(self) -> Optional[ReadyKey]:
        """Return the next valid event key without consuming it."""

        queued = self._pop_ready_entry()
        if queued is None:
            return None
        heapq.heappush(self._ready_heap, queued)
        return (queued[0], queued[1], queued[2], queued[3], queued[6])

    def _pop_ready_entry(self) -> Optional[GlobalReadyEntry]:
        """Remove and return the next valid global ready entry.

        Ready entries are versioned because a task's resource clock can move
        after the entry is queued.  Keeping the validation in one consuming
        primitive lets :meth:`step` avoid the old pop/push/pop round trip;
        :meth:`peek_ready_key` restores the validated entry before returning.
        """

        ready_heap = self._ready_heap
        heappop = heapq.heappop
        group_versions_get = self._group_versions.get
        ready_by_group_get = self._ready_by_group.get
        while ready_heap:
            queued = heappop(ready_heap)
            resource_group = queued[7]
            version = queued[8]
            if version != group_versions_get(resource_group):
                continue
            group_ready = ready_by_group_get(resource_group)
            if not group_ready:
                continue
            task_id = group_ready[0][-1]
            current = self._ready_key(task_id)
            if current != queued[:7]:
                self._refresh_group(resource_group)
                continue
            return queued
        return None

    def _lanes_for(self, resource_id: str) -> List[float]:
        lanes = self._resource_lane_available.get(resource_id)
        if lanes is None:
            capacity = self.resource_capacities.get(resource_id, 1)
            lanes = [0.0] * capacity
            self._resource_lane_available[resource_id] = lanes
        return lanes

    def _select_lane(
        self, resource_id: str
    ) -> Tuple[List[float], int, float]:
        lanes = self._lanes_for(resource_id)
        if len(lanes) == 1:
            return lanes, 0, lanes[0]
        lane_index = min(
            range(len(lanes)), key=lambda index: (lanes[index], index)
        )
        return lanes, lane_index, lanes[lane_index]

    def step(self) -> Optional[KernelEvent]:
        """Execute and return one task, or ``None`` when no task is ready."""

        queued = self._pop_ready_entry()
        if queued is None:
            return None
        (
            start_ns,
            effective_ready_ns,
            dependency_ready_ns,
            _earliest_start_ns,
            _phase,
            _sequence,
            task_id,
            resource_group,
            _version,
        ) = queued
        task = self._tasks[task_id]
        demands = self._sorted_demands[task_id]
        isfinite = math.isfinite
        timing_is_finite = isfinite(start_ns)
        if timing_is_finite:
            for demand in demands:
                if not isfinite(start_ns + demand.service_ns):
                    timing_is_finite = False
                    break
        if not timing_is_finite:
            raise ValueError(
                "task {} timing exceeds finite simulation range".format(
                    task_id
                )
            )

        group_ready = self._ready_by_group[resource_group]
        heapq.heappop(group_ready)

        self._tasks.pop(task_id)
        self._dependency_ready.pop(task_id)
        self._resource_groups.pop(task_id)
        self._sorted_demands.pop(task_id)
        self._phase_sequence.pop(task_id)
        resource_predecessors: Dict[str, Dict[str, object]] = {}
        resource_lanes: Dict[str, int] = {}
        end_ns = start_ns
        resource_last_interval = self.resource_last_interval
        resource_available = self.resource_available
        resource_busy_ns = self.resource_busy_ns
        resource_queue_wait_ns = self.resource_queue_wait_ns
        resource_task_count = self.resource_task_count
        resource_busy_ns_get = resource_busy_ns.get
        queue_wait_ns = start_ns - effective_ready_ns
        for demand in demands:
            demand_end_ns = start_ns + demand.service_ns
            lanes, lane_index, available_ns = self._select_lane(
                demand.resource_id
            )
            previous_interval = self._resource_lane_last_interval.get(
                (demand.resource_id, lane_index)
            )
            resource_lanes[demand.resource_id] = lane_index
            if previous_interval is not None and available_ns == start_ns:
                resource_predecessors[demand.resource_id] = dict(
                    previous_interval
                )
            if demand_end_ns > end_ns:
                end_ns = demand_end_ns
            interval: Dict[str, object] = {
                "task_id": task.task_id,
                "resource_id": demand.resource_id,
                "start_ns": start_ns,
                "end_ns": demand_end_ns,
            }
            if len(lanes) > 1:
                interval["lane"] = lane_index
            lanes[lane_index] = demand_end_ns
            resource_available[demand.resource_id] = (
                demand_end_ns if len(lanes) == 1 else min(lanes)
            )
            resource_last_interval[demand.resource_id] = interval
            self._resource_lane_last_interval[
                (demand.resource_id, lane_index)
            ] = interval
            resource_busy_ns[demand.resource_id] = (
                resource_busy_ns_get(demand.resource_id, 0.0)
                + demand.service_ns
            )
            resource_queue_wait_ns[demand.resource_id] = (
                resource_queue_wait_ns.get(demand.resource_id, 0.0)
                + queue_wait_ns
            )
            resource_task_count[demand.resource_id] = (
                resource_task_count.get(demand.resource_id, 0) + 1
            )

        self._completed_end[task_id] = end_ns
        self._completion_leases[task_id] = 1
        self.completed_count += 1
        self.total_queue_wait_ns += queue_wait_ns
        self.total_service_ns += end_ns - start_ns
        if end_ns > self.makespan_ns:
            self.makespan_ns = end_ns

        groups_to_refresh = [resource_group]
        for dependent_id in self._dependents.get(task_id, ()):
            previous_ready_ns = self._dependency_ready[dependent_id]
            if end_ns > previous_ready_ns:
                self._dependency_ready[dependent_id] = end_ns
            self._indegree[dependent_id] -= 1
            if self._indegree[dependent_id] == 0:
                changed_group = self._add_ready_task(dependent_id)
                if (
                    changed_group is not None
                    and changed_group not in groups_to_refresh
                ):
                    groups_to_refresh.append(changed_group)
        self._indegree.pop(task_id, None)
        if len(groups_to_refresh) == 2:
            left_group, right_group = groups_to_refresh
            if right_group < left_group:
                groups_to_refresh[0], groups_to_refresh[1] = (
                    right_group,
                    left_group,
                )
        elif len(groups_to_refresh) > 2:
            groups_to_refresh.sort()
        for changed_group in groups_to_refresh:
            self._refresh_group(changed_group)

        return KernelEvent(
            task=task,
            start_ns=start_ns,
            end_ns=end_ns,
            dependency_ready_ns=dependency_ready_ns,
            effective_ready_ns=effective_ready_ns,
            demands=demands,
            resource_predecessors=resource_predecessors,
            resource_lanes=resource_lanes,
        )

    def step_completion(self) -> Optional[KernelCompletion]:
        """Execute one task and return its typed runtime completion view."""

        event = self.step()
        if event is None:
            return None
        action, payload, phase, sequence = decode_runtime_metadata(event.task)
        instruction_batch, controller_batch = decode_aggregate_metadata(event.task)
        resource_metrics = {
            demand.resource_id: {
                "queue_wait_ns": event.queue_wait_ns,
                "service_ns": demand.service_ns,
                "capacity": float(
                    self.resource_capacities.get(demand.resource_id, 1)
                ),
            }
            for demand in event.demands
        }
        return KernelCompletion(
            task=event.task,
            action=action,
            payload=payload,
            phase=phase,
            sequence=sequence,
            start_ns=event.start_ns,
            end_ns=event.end_ns,
            dependency_ready_ns=event.dependency_ready_ns,
            effective_ready_ns=event.effective_ready_ns,
            demands=event.demands,
            resource_predecessors=event.resource_predecessors,
            queue_wait_ns=event.queue_wait_ns,
            service_ns=event.service_ns,
            resource_metrics=resource_metrics,
            instruction_batch=instruction_batch,
            controller_batch=controller_batch,
        )

    def retain_completed(self, task_id: str) -> int:
        """Acquire one lease on a still-retained completion timestamp."""

        leases = self._completion_leases.get(task_id, 0)
        if leases <= 0 or task_id not in self._completed_end:
            raise ValueError("completion is unavailable: {}".format(task_id))
        leases += 1
        self._completion_leases[task_id] = leases
        return leases

    def release_completed(self, task_id: str) -> int:
        """Release completed dependency state after dynamic consumers attach."""

        leases = self._completion_leases.get(task_id, 0)
        if leases <= 0:
            raise ValueError("completion is unavailable: {}".format(task_id))
        leases -= 1
        if leases:
            self._completion_leases[task_id] = leases
            return leases

        # All registered dependents have already inherited the completion
        # timestamp at task completion, so even an empty list can be released.
        self._completion_leases.pop(task_id, None)
        self._dependents.pop(task_id, None)
        self._completed_end.pop(task_id, None)
        return 0

    def assert_drained(self) -> None:
        if self._tasks:
            raise ValueError("schedule contains a dependency cycle")


__all__ = [
    "CompiledGraphExecutor",
    "CompiledGraphLayout",
    "KernelEvent",
    "SubmissionReceipt",
    "UnifiedEventKernel",
    "task_start_ns",
    "validate_task_graph",
]
