"""Deterministic analysis-level online serving scheduler.

This module deliberately models scheduling policy rather than individual model
kernels.  A cost provider can lower each homogeneous serving cohort into a
calibrated duration; the default provider is a small, non-zero analytical
roofline.  Preemption is only considered between completed prefill/decode/MTP
units, so the resulting trace is deterministic and straightforward to audit.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping as _ABCMapping, Sequence as _ABCSequence
from dataclasses import dataclass, field, replace
from enum import Enum
import heapq
import inspect
import math
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .config import ScenarioConfig
from .control_plane import _execution_resource_capacities
from .contracts import (
    ResourceDemand,
    TaskCategory,
    TaskSpec,
    _PreparedExecutionStage as _ExecutionStage,
    _PreparedExecutionTask as _ExecutionTask,
)
from .cost_models import (
    CPUProfile,
    GPUProfile,
    HBMProfile,
    MemoryWorkload,
    _dma_setup_service,
    estimate_cpu_logical_stream,
    estimate_cpu_memory,
)
from .event_kernel import (
    CompiledGraphLayout,
    UnifiedEventKernel,
    _validated_phase_sequence,
)
from .execution_control import ExecutionControl
from .control_plane_state import (
    control_plane_decision,
    is_control_plane_generated_tensor,
)
from .mtp import (
    MTPRequestCursor,
    expected_draft_prefix_tokens,
    round_accepted_prefix,
)
from .planner import (
    _component_resource_id,
    _compilation_scope,
    _cpu_profiles,
    _execution_layers,
    _execution_view,
    _kind,
    _kv_dtype_bits,
    _kv_tensor_bytes,
    _parallel_plan,
    _request_modalities,
    _topology_router,
    _trusted_execution_stages,
    materialize_requests,
    TopologyAwareBatchCostProvider,
)
from .precision import dtype_bits
from .residency import (
    AccessOperation,
    AllocationLifecycle,
    AllocationResidencyManager,
    MemoryAccess,
    Migration,
    MigrationKind,
)


class RequestStatus(str, Enum):
    """Lifecycle states retained in the realized serving result."""

    ARRIVALS = "arrivals"
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"
    REJECTED = "rejected"


_TERMINAL_STATUSES = (RequestStatus.FINISHED, RequestStatus.REJECTED)
@dataclass(frozen=True)
class ServingPolicy:
    mode: str = "continuous_batching"
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 512
    max_num_ubatch_tokens: Optional[int] = None
    prefill_chunk_tokens: int = 128
    mixed_phase_batching: bool = False
    policy: str = "decode_first_aging"
    phase_candidate_order: str = "least_recently_served"
    starvation_ns: float = 1_000_000.0
    preemption_enabled: bool = True
    preemption_granularity: str = "boundary"
    preemption_policy: str = "auto"
    slo_ttft_ns: Optional[float] = None
    slo_tbt_ns: Optional[float] = None
    prefill_stop_offsets: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _PromptCachePolicy:
    """Optional physical contract for retained prompt-cache checkpoints.

    Prompt-cache entries are intentionally kept separate from the active KV
    and linear-state ledgers.  ``slot_allocation_bytes`` describes the
    logical full-slot commitment (normally the full Q4 KV slot plus one
    recurrent-state allocation); ``resident_entry_bytes`` may instead
    describe the graph-referenced checkpoint working set.  Neither value is
    a concurrency lookup or a fitted multiplier.
    """

    enabled: bool = False
    context_checkpoints: int = 1
    component: Optional[str] = None
    offload_component: Optional[str] = None
    # Explicit full-slot/logical allocation.  If omitted, serving derives it
    # from the model context, KV page quantum, and linear-state allocation.
    slot_allocation_bytes: Optional[int] = None
    # ``entry_bytes`` is an explicit graph/log-backed entry size.  It is also
    # used as the logical allocation when no slot allocation is supplied.
    entry_bytes: Optional[int] = None
    resident_entry_bytes: Optional[int] = None
    kv_slot_allocation_bytes: Optional[int] = None
    state_slot_allocation_bytes: Optional[int] = None
    graph_residency_mode: str = "actual_tokens"
    retain_completed: bool = True
    # This is deliberately opt-in: the default does not model prompt-save
    # work, preserving the historical analytical serving path.
    save_implementation: str = "unmodeled"
    # Experimental analytical assumption, not a measured copy-API service.
    # Charge only the known ordinary K/V GPU read count. Defaults stay off.
    apply_tensor_get_submission_service: bool = False
    # Experimental migration-controller proxy. A resident tensor read does
    # not itself prove make-resident work or translation misses. Defaults off.
    apply_tensor_get_controller_phases: bool = False
    # Pinned pageable-copy implementation; payload instruction issue only.
    # This is an explicit source contract, never a measured API duration.
    driver_cpu_copy_issue_contract: Optional[str] = None
    unified_kv: bool = False
    host_state_layout: Optional[str] = None
    host_pointer_bytes: Optional[int] = None
    # One current, unshared FP32 R/S row; no P tensor or rollback planes.
    recurrent_state_layout: Optional[str] = None
    # Observed log values are provenance/configuration facts only.  They are
    # never selected by request count, concurrency, or another empirical key.
    observed_entry_sizes_bytes: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _ServingResourcePolicy:
    """Optional physical residency model for online serving.

    Existing scenarios keep their historical numbers unless a workload opts
    in through ``workload.metadata`` (or ``placement.metadata``) under the
    ``serving_runtime`` key.  Once enabled, the runtime only uses declared
    hardware rates, the unified resident working set, and explicit transfer
    controls.  There are deliberately no fitted concurrency multipliers.
    """

    enabled: bool = False
    # Bytes held back from the physical cache component for the driver,
    # allocator, and other non-runtime users.  Zero preserves the existing
    # dynamic component capacity semantics.
    vram_reserve_bytes: int = 0
    # ``None`` means derive the transfer service from the declared topology;
    # the host DMA profile is the fallback when no route is available.
    page_transfer_bandwidth_gb_s: Optional[float] = None
    # ``None`` uses HostOrchestrationProfile.dma_latency_ns only when the
    # submission latency is known.  A runtime can explicitly mark it unknown;
    # in that case the simulator reports partial timing and charges only the
    # independently modeled bulk transfer service.
    page_fault_latency_ns: Optional[float] = None
    page_fault_latency_known: bool = True
    # Physical residency granularity is an OS/backend property and is not the
    # same thing as the model's logical KV page.  Allocation-specific contracts
    # may override this value.
    residency_granule_bytes: Optional[int] = None
    # Consecutive make-resident requests in one cohort may be coalesced by the
    # driver.  ``None`` preserves the legacy one-latency-per-granule behavior;
    # a positive value charges one latency for each batch of this many bytes.
    fault_batch_bytes: Optional[int] = None
    # ``granule`` is the compatibility default.  ``make_resident_batch``
    # charges once for each coalesced direction/route group in a cohort.
    fault_latency_scope: str = "granule"
    timing_completeness: str = "complete"
    include_custom_lowerers: bool = False
    # Optional per-slot logical reservation.  If omitted, the graph's model
    # max_sequence_length is used for the diagnostic unified capacity.
    kv_slot_context_tokens: Optional[int] = None
    prompt_cache: Optional[_PromptCachePolicy] = None


@dataclass(frozen=True)
class KVCachePolicy:
    cache_component: Optional[str]
    offload_component: Optional[str]
    tokens_per_page: int
    dtype: Optional[str]
    offload_ratio: float
    allocation_policy: str
    preemption_mode: str
    prefetch_distance: int
    bytes_per_page: int
    logical_bytes_per_token: int
    capacity_bytes: int
    capacity_pages: int
    offload_capacity_bytes: int


@dataclass(frozen=True)
class LinearStatePolicy:
    cache_component: Optional[str]
    offload_component: Optional[str]
    bytes_per_request: int
    capacity_bytes: int
    capacity_requests: int
    offload_capacity_bytes: int
    offload_ratio: float = 1.0


@dataclass(frozen=True)
class MTPPolicy:
    method: str = "disabled"
    candidate_tokens: int = 1
    min_draft_tokens: int = 0
    continuation_threshold: Optional[float] = None
    proposal_length_model: str = "max"
    expected_draft_tokens_per_round: Optional[float] = None
    draft_length_trace: Tuple[int, ...] = ()
    acceptance_model: str = "expected_prefix"
    acceptance_rate: float = 0.0
    proposal_cost_scale: float = 1.0
    acceptance_trace: Tuple[float, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.method.lower() not in ("", "none", "disabled", "off")


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    arrival_ns: float
    prompt_tokens: int
    output_tokens: int
    priority: int = 0
    deadline_ns: Optional[float] = None


class _SyntheticServingRequestSequence(Sequence[ServingRequest]):
    """Lazy serving-request view for homogeneous synthetic workloads."""

    def __init__(self, scenario: ScenarioConfig) -> None:
        workload = scenario.workload
        self._count = max(0, int(workload.request_count))
        self._prompt_tokens = int(workload.prompt_tokens)
        self._output_tokens = int(workload.output_tokens)
        self._interval_ns = (
            1_000_000_000.0 / float(workload.arrival_rate_rps)
            if workload.arrival_rate_rps > 0
            else 0.0
        )

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for index in range(self._count):
            yield self._at(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(
                self._at(position)
                for position in range(*index.indices(self._count))
            )
        if not isinstance(index, int):
            raise TypeError("request index must be an integer or slice")
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("request index out of range")
        return self._at(index)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _SyntheticServingRequestSequence):
            return (
                self._count,
                self._prompt_tokens,
                self._output_tokens,
                self._interval_ns,
            ) == (
                other._count,
                other._prompt_tokens,
                other._output_tokens,
                other._interval_ns,
            )
        if isinstance(other, _ABCSequence):
            return len(other) == self._count and all(
                left == right for left, right in zip(self, other)
            )
        return False

    def _at(self, index: int) -> ServingRequest:
        return ServingRequest(
            request_id="request-{:04d}".format(index),
            arrival_ns=index * self._interval_ns,
            prompt_tokens=self._prompt_tokens,
            output_tokens=self._output_tokens,
        )


@dataclass(frozen=True)
class ServingPlan:
    scenario: ScenarioConfig
    requests: Sequence[ServingRequest]
    scheduler: ServingPolicy
    kv_policy: KVCachePolicy
    linear_state_policy: LinearStatePolicy
    mtp: MTPPolicy


@dataclass(frozen=True)
class BatchItem:
    request_id: str
    phase: str
    token_count: int
    context_tokens: int
    proposed_tokens: int = 0
    expected_accepted_tokens: float = 0.0
    kv_append_tokens: Optional[int] = None
    # KV entries actually materialized by the backbone in this batch.  For
    # ordinary prefill/decode this is the same as ``kv_append_tokens``.  MTP
    # verification materializes every proposal, while only the accepted prefix
    # in ``kv_append_tokens`` survives as persistent cache state.
    kv_materialized_tokens: Optional[int] = None
    # Explicit MTP accounting.  These are optional for compatibility with
    # third-party/custom BatchItem construction; runtime-created MTP items
    # always populate all four values.
    main_tokens: Optional[int] = None
    draft_tokens: Optional[int] = None
    verifier_tokens: Optional[int] = None
    committed_tokens: Optional[int] = None
    # Number of positions whose logits are requested from the output head.
    # Runtime-created prompt chunks set this to zero except for the final
    # prompt position. MTP verification requests logits for every verifier
    # position. ``None`` retains the historical all-positions behavior for
    # third-party/custom BatchItem construction.
    logit_tokens: Optional[int] = None
    # Cursor visible when this item's terminal DAG task completes.  Planner
    # metadata uses it to distinguish multiple chunks/rounds of one phase.
    completion_cursor: Optional[int] = None


@dataclass(frozen=True)
class BatchCohort:
    cohort_id: str
    kind: str
    start_ns: float
    items: Tuple[BatchItem, ...]
    proposal_cost_scale: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def request_ids(self) -> Tuple[str, ...]:
        return tuple(item.request_id for item in self.items)

    @property
    def token_count(self) -> int:
        return sum(item.token_count for item in self.items)

    @property
    def phase_token_counts(self) -> Mapping[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.phase] = counts.get(item.phase, 0) + item.token_count
        return counts

    @property
    def phase_item_counts(self) -> Mapping[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.phase] = counts.get(item.phase, 0) + 1
        return counts

    @property
    def is_mixed(self) -> bool:
        return len(self.phase_token_counts) > 1


@dataclass(frozen=True)
class BatchCost:
    duration_ns: float
    energy_pj: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ExecutionStageTaskLayout:
    """One task's immutable position in a cached stage replay graph."""

    task_id: str
    stage_position: int
    task_position: int
    dependencies: Tuple[str, ...]


@dataclass(frozen=True)
class _ExecutionStageReplayLayout:
    """Validated task topology for planner-owned execution stages.

    Ready times and demand quantities remain live inputs.  The cache retains
    only task ids, dependency wiring, stage/task positions, and the event
    kernel's compiled structural layout; no executed event or timestamp is
    reused.
    """

    tasks: Tuple[_ExecutionStageTaskLayout, ...]
    compiled: CompiledGraphLayout
    phase_sequence: Tuple[Tuple[int, int], ...]
    stage_positions: Tuple[int, ...]
    base_task_ids: Tuple[str, ...]
    indegree_template: Tuple[int, ...]
    resource_ids: Tuple[str, ...]
    demand_resource_indices: Tuple[Tuple[int, ...], ...]
    group_indices: Tuple[int, ...]
    group_count: int
    structure_key: Optional[Tuple[object, ...]] = None

    @classmethod
    def compile(
        cls,
        stages: Sequence[_ExecutionStage],
        *,
        structure_key: Optional[Tuple[object, ...]] = None,
    ) -> "_ExecutionStageReplayLayout":
        namespace = "serving.cached"
        namespaced_id: Dict[Tuple[str, str], str] = {}
        terminal_ids_by_stage: Dict[str, Tuple[str, ...]] = {}
        for stage_position, stage in enumerate(stages):
            prefix = "{}.stage{:04d}.".format(namespace, stage_position)
            for task_position, task in enumerate(stage.execution_tasks):
                namespaced_id[(stage.stage_id, task.task_id)] = (
                    "{}task{:06d}".format(prefix, task_position)
                )
            dependency_ids = {
                dependency
                for task in stage.execution_tasks
                for dependency in task.dependencies
            }
            terminal_ids_by_stage[stage.stage_id] = tuple(
                namespaced_id[(stage.stage_id, task.task_id)]
                for task in stage.execution_tasks
                if task.task_id not in dependency_ids
            )

        task_layouts: List[_ExecutionStageTaskLayout] = []
        specs: List[TaskSpec] = []
        for stage_position, stage in enumerate(stages):
            stage_parent_ids = tuple(
                terminal_id
                for dependency_stage_id in stage.dependencies
                for terminal_id in terminal_ids_by_stage[dependency_stage_id]
            )
            for task_position, task in enumerate(stage.execution_tasks):
                task_id = namespaced_id[(stage.stage_id, task.task_id)]
                dependencies = tuple(
                    namespaced_id[(stage.stage_id, dependency)]
                    for dependency in task.dependencies
                )
                if not dependencies:
                    dependencies = stage_parent_ids
                task_layout = _ExecutionStageTaskLayout(
                    task_id,
                    stage_position,
                    task_position,
                    dependencies,
                )
                task_layouts.append(task_layout)
                specs.append(
                    TaskSpec(
                        task_id=task_id,
                        request_id=task.request_ids[0],
                        name=task.task_id,
                        category=task.category,
                        dependencies=dependencies,
                        demands=task.demands,
                        earliest_start_ns=0.0,
                        metadata={
                            **dict(task.metadata),
                            "execution_stage_id": stage.stage_id,
                        },
                    )
                )
        compiled_specs = tuple(specs)
        compiled_layout = CompiledGraphLayout.compile(compiled_specs)
        resource_ids = tuple(
            sorted(
                {
                    resource_id
                    for task_resource_ids in compiled_layout.demand_resource_ids
                    for resource_id in task_resource_ids
                }
            )
        )
        resource_position = {
            resource_id: position
            for position, resource_id in enumerate(resource_ids)
        }
        resource_groups = tuple(sorted(set(compiled_layout.resource_groups)))
        group_position = {
            resource_group: position
            for position, resource_group in enumerate(resource_groups)
        }
        compiled_tasks = tuple(task_layouts)
        return cls(
            compiled_tasks,
            # Planner stages are parser-validated, but runtime residency may
            # splice causal transfer stages into the final graph.  Validate
            # the complete replay layout once so both the live and isolated
            # kernels can safely reuse the compiled structure.
            compiled_layout,
            tuple(_validated_phase_sequence(task) for task in compiled_specs),
            tuple(task.stage_position for task in compiled_tasks),
            tuple(task.task_id for task in compiled_tasks),
            tuple(
                len(dependencies)
                for dependencies in compiled_layout.dependency_positions
            ),
            resource_ids,
            tuple(
                tuple(resource_position[resource_id] for resource_id in task_resources)
                for task_resources in compiled_layout.demand_resource_ids
            ),
            tuple(
                group_position[resource_group]
                for resource_group in compiled_layout.resource_groups
            ),
            len(resource_groups),
            structure_key,
        )

    def instantiate(
        self,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        namespace: Optional[str] = None,
    ) -> Tuple[TaskSpec, ...]:
        specs: List[TaskSpec] = []
        if namespace is None:
            task_ids = tuple(layout.task_id for layout in self.tasks)
        else:
            prefix = namespace + "."
            task_ids = tuple(prefix + layout.task_id for layout in self.tasks)
        for position, layout in enumerate(self.tasks):
            stage = stages[layout.stage_position]
            task = stage.execution_tasks[layout.task_position]
            specs.append(
                TaskSpec(
                    task_id=task_ids[position],
                    request_id=task.request_ids[0],
                    name=task.task_id,
                    category=task.category,
                    dependencies=tuple(
                        task_ids[dependency_position]
                        for dependency_position in (
                            self.compiled.dependency_positions[position]
                        )
                    ),
                    demands=task.demands,
                    earliest_start_ns=ready_by_stage[stage.stage_id],
                    metadata={
                        **dict(task.metadata),
                        "execution_stage_id": stage.stage_id,
                    },
                )
            )
        return tuple(specs)

    def instantiate_trusted(
        self,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        namespace: str,
    ) -> Tuple[TaskSpec, ...]:
        """Bind a planner-validated layout without copying audit metadata.

        ``replay_layout`` is available only for the in-process prepared-stage
        handoff after one ordinary ``TaskSpec``/``CompiledGraphLayout``
        validation.  Structure keys prove the dependency and resource-id
        shape on reuse.  Live demand objects retain their own constructor
        validation, and stage readiness is checked once per stage here.  The
        metadata mapping can therefore stay shared/read-only; the event kernel
        only decodes reserved runtime fields from it.
        """

        expected_fields = (
            "task_id",
            "request_id",
            "name",
            "category",
            "dependencies",
            "demands",
            "earliest_start_ns",
            "marker",
            "token_index",
            "metadata",
        )
        if tuple(TaskSpec.__dataclass_fields__) != expected_fields:
            return self.instantiate(
                stages,
                ready_by_stage,
                namespace=namespace,
            )
        ready_values: List[float] = []
        for stage in stages:
            ready_ns = float(ready_by_stage[stage.stage_id])
            if not math.isfinite(ready_ns) or ready_ns < 0.0:
                # Preserve the public constructor's exact validation error.
                return self.instantiate(
                    stages,
                    ready_by_stage,
                    namespace=namespace,
                )
            ready_values.append(ready_ns)
        prefix = namespace + "."
        task_ids = tuple(prefix + layout.task_id for layout in self.tasks)
        specs: List[TaskSpec] = []
        for position, layout in enumerate(self.tasks):
            stage = stages[layout.stage_position]
            task = stage.execution_tasks[layout.task_position]
            spec = object.__new__(TaskSpec)
            object.__setattr__(spec, "task_id", task_ids[position])
            object.__setattr__(spec, "request_id", task.request_ids[0])
            object.__setattr__(spec, "name", task.task_id)
            object.__setattr__(spec, "category", task.category)
            object.__setattr__(
                spec,
                "dependencies",
                tuple(
                    task_ids[dependency_position]
                    for dependency_position in (
                        self.compiled.dependency_positions[position]
                    )
                ),
            )
            object.__setattr__(spec, "demands", task.demands)
            object.__setattr__(
                spec,
                "earliest_start_ns",
                ready_values[layout.stage_position],
            )
            object.__setattr__(spec, "marker", None)
            object.__setattr__(spec, "token_index", None)
            object.__setattr__(spec, "metadata", task.metadata)
            specs.append(spec)
        return tuple(specs)


class _UniformIsolatedReplay:
    """Exact isolated timing shadow for a uniformly shifted live graph.

    The live event order is reused only while it agrees with an independent
    zero-origin scheduler selector.  The shadow retains the dependency and
    resource clocks needed for duration, but none of the live kernel's trace,
    predecessor, busy-time, or completed-task bookkeeping.  Any ordering
    disagreement invalidates the shadow and the caller falls back to the
    ordinary isolated kernel replay.
    """

    def __init__(
        self,
        *,
        resource_capacities: Optional[Mapping[str, int]] = None,
        origin_ns: float = 0.0,
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
        try:
            normalized_origin_ns = float(origin_ns)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("shadow origin must be finite") from None
        if not math.isfinite(normalized_origin_ns):
            raise ValueError("shadow origin must be finite")
        self._resource_capacities = capacities
        self._origin_ns = normalized_origin_ns
        self._valid = False
        self._remaining = 0
        self._tasks: Dict[str, TaskSpec] = {}
        self._indegree: Dict[str, int] = {}
        self._dependents: Dict[str, Tuple[str, ...]] = {}
        self._dependency_ready: Dict[str, float] = {}
        self._resource_groups: Dict[str, Tuple[str, ...]] = {}
        self._sorted_demands: Dict[str, Tuple[ResourceDemand, ...]] = {}
        self._ready_by_group: Dict[
            Tuple[str, ...],
            List[Tuple[float, str]],
        ] = {}
        self._resource_lane_available: Dict[str, List[float]] = {}
        self._makespan_ns = 0.0

    def _lanes_for(self, resource_id: str) -> List[float]:
        lanes = self._resource_lane_available.get(resource_id)
        if lanes is None:
            lanes = [0.0] * self._resource_capacities.get(resource_id, 1)
            self._resource_lane_available[resource_id] = lanes
        return lanes

    def bind(
        self,
        specs: Sequence[TaskSpec],
        compiled: Optional[CompiledGraphLayout] = None,
        *,
        structure_prevalidated: bool = False,
    ) -> None:
        tasks = tuple(specs)
        tasks_by_id = {task.task_id: task for task in tasks}
        if len(tasks_by_id) != len(tasks):
            return
        task_ids = tuple(task.task_id for task in tasks)
        if compiled is not None:
            if (
                not structure_prevalidated
                and not compiled.matches_structure(tasks)
            ):
                return
            indegree = {
                task_ids[position]: len(dependencies)
                for position, dependencies in enumerate(
                    compiled.dependency_positions
                )
            }
            dependents_by_id = {
                task_ids[position]: tuple(
                    task_ids[dependent_position]
                    for dependent_position in dependent_positions
                )
                for position, dependent_positions in enumerate(
                    compiled.dependent_positions
                )
            }
            resource_groups = {
                task_ids[position]: compiled.resource_groups[position]
                for position in range(len(tasks))
            }
            sorted_demands = {
                task_ids[position]: tuple(
                    tasks[position].demands[demand_position]
                    for demand_position in compiled.demand_order[position]
                )
                for position in range(len(tasks))
            }
            root_ids = tuple(task_ids[position] for position in compiled.root_positions)
        else:
            dependent_lists: Dict[str, List[str]] = {
                task_id: [] for task_id in tasks_by_id
            }
            for task in tasks:
                for dependency_id in task.dependencies:
                    dependents = dependent_lists.get(dependency_id)
                    if dependents is None:
                        return
                    dependents.append(task.task_id)
            indegree = {
                task.task_id: len(task.dependencies) for task in tasks
            }
            dependents_by_id = {
                task_id: tuple(sorted(dependents))
                for task_id, dependents in dependent_lists.items()
            }
            resource_groups = {
                task.task_id: tuple(
                    sorted({demand.resource_id for demand in task.demands})
                )
                for task in tasks
            }
            sorted_demands = {
                task.task_id: tuple(
                    sorted(task.demands, key=lambda demand: demand.resource_id)
                )
                for task in tasks
            }
            root_ids = tuple(
                task.task_id for task in tasks if indegree[task.task_id] == 0
            )

        self._tasks = tasks_by_id
        self._indegree = indegree
        self._dependents = dependents_by_id
        self._dependency_ready = {
            task.task_id: 0.0 for task in tasks
        }
        self._resource_groups = resource_groups
        self._sorted_demands = sorted_demands
        self._ready_by_group = {}
        self._resource_lane_available = {
            resource_id: [0.0] * capacity
            for resource_id, capacity in self._resource_capacities.items()
        }
        self._makespan_ns = 0.0
        self._remaining = len(tasks)
        for task_id in root_ids:
            group = self._resource_groups[task_id]
            heapq.heappush(
                self._ready_by_group.setdefault(group, []),
                (0.0, task_id),
            )
        self._valid = True

    def observe(self, event: Any) -> None:
        self.observe_values(
            event.task.task_id,
            event.start_ns,
            event.end_ns,
        )

    def observe_values(
        self,
        live_task_id: str,
        live_start_ns: float,
        live_end_ns: float,
    ) -> None:
        """Advance from one live event without requiring a KernelEvent."""

        if not self._valid:
            return
        best_key: Optional[Tuple[float, float, float, float, str]] = None
        best_group: Optional[Tuple[str, ...]] = None
        for resource_group, group_ready in self._ready_by_group.items():
            if not group_ready:
                continue
            dependency_ready_ns, task_id = group_ready[0]
            resources_ready_ns = 0.0
            for resource_id in resource_group:
                lanes = self._resource_lane_available.get(resource_id)
                available_ns = min(lanes) if lanes is not None else 0.0
                if available_ns > resources_ready_ns:
                    resources_ready_ns = available_ns
            actual_start_ns = max(
                dependency_ready_ns,
                resources_ready_ns,
            )
            ready_key = (
                actual_start_ns,
                dependency_ready_ns,
                dependency_ready_ns,
                0.0,
                task_id,
            )
            if best_key is None or ready_key < best_key:
                best_key = ready_key
                best_group = resource_group
        if (
            best_key is None
            or best_group is None
            or best_key[-1] != live_task_id
        ):
            self._valid = False
            return

        task_id = best_key[-1]
        start_ns = best_key[0]
        end_ns = start_ns
        lane_choices: List[Tuple[List[float], int, float]] = []
        for demand in self._sorted_demands[task_id]:
            demand_end_ns = start_ns + demand.service_ns
            if not math.isfinite(demand_end_ns):
                self._valid = False
                return
            if demand_end_ns > end_ns:
                end_ns = demand_end_ns
            lanes = self._lanes_for(demand.resource_id)
            lane_index = min(
                range(len(lanes)),
                key=lambda index: (lanes[index], index),
            )
            lane_choices.append((lanes, lane_index, demand_end_ns))

        # The shadow runs in a zero-origin coordinate system.  A uniform live
        # graph is equivalent only when both its predicted event times agree
        # after translating that origin; matching the task id alone would let
        # a stale/mis-seeded lane clock under-report the isolated duration.
        try:
            live_start_ns = float(live_start_ns)
            live_end_ns = float(live_end_ns)
        except (TypeError, ValueError, OverflowError):
            self._valid = False
            return
        if (
            not math.isfinite(live_start_ns)
            or not math.isfinite(live_end_ns)
            or live_start_ns != self._origin_ns + start_ns
            or live_end_ns != self._origin_ns + end_ns
        ):
            self._valid = False
            return

        heapq.heappop(self._ready_by_group[best_group])
        for lanes, lane_index, demand_end_ns in lane_choices:
            lanes[lane_index] = demand_end_ns
        if end_ns > self._makespan_ns:
            self._makespan_ns = end_ns

        for dependent_id in self._dependents[task_id]:
            if end_ns > self._dependency_ready[dependent_id]:
                self._dependency_ready[dependent_id] = end_ns
            self._indegree[dependent_id] -= 1
            if self._indegree[dependent_id] == 0:
                resource_group = self._resource_groups[dependent_id]
                heapq.heappush(
                    self._ready_by_group.setdefault(resource_group, []),
                    (
                        self._dependency_ready[dependent_id],
                        dependent_id,
                    ),
                )
        self._remaining -= 1

    @property
    def duration_ns(self) -> Optional[float]:
        if not self._valid or self._remaining != 0:
            return None
        return self._makespan_ns


_ExecutionStageLayoutKey = Tuple[
    Tuple[
        Tuple[int, ...],
        Tuple[
            Tuple[Tuple[int, ...], Tuple[str, ...], bool, Tuple[int, int]],
            ...,
        ],
    ],
    ...,
]


class _ExecutionStageMetadataCache:
    """Bound parsed planner stages to the lifetime of one batch lowerer.

    The topology-aware lowerer owns an exact-template cache and returns the
    same planner-created ``execution_stages`` tuple on a template hit.  Local
    replay deliberately reuses that lowerer across warmup and measurements;
    reparsing thousands of detailed task mappings cannot change the event
    graph.  Entries retain the source tuple and compare it by identity, so a
    replacement tuple is always reparsed.  Only planner-authored stage data is
    cached; custom or fallback metadata continues through the fail-closed
    parser on every call.  Compiled replay layouts live in a separate bounded
    LRU keyed by the normalized structure of the final, runtime-adjusted graph.
    """

    def __init__(
        self,
        max_entries: int = 512,
        max_final_layouts: int = 128,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_final_layouts = max(1, int(max_final_layouts))
        self._values: Dict[
            int,
            Tuple[
                object,
                Tuple[Tuple[_ExecutionStage, ...], Optional[str]],
                bool,
            ],
        ] = {}
        self._final_layouts: OrderedDict[
            _ExecutionStageLayoutKey, _ExecutionStageReplayLayout
        ] = OrderedDict()
        self._identity_final_layout_keys: OrderedDict[
            Tuple[object, ...],
            Tuple[Tuple[object, ...], _ExecutionStageLayoutKey],
        ] = OrderedDict()
        self._identity_task_layout_keys: OrderedDict[
            int,
            Tuple[
                object,
                Tuple[
                    Tuple[
                        Tuple[int, ...],
                        Tuple[str, ...],
                        bool,
                        Tuple[int, int],
                    ],
                    ...,
                ],
            ],
        ] = OrderedDict()
        # Keep replay-layout accounting separate from the identity/parser
        # cache above.  A lookup is counted only after the planner-owned
        # handoff, expected stage count, and normalized key have all passed;
        # rejected metadata therefore cannot masquerade as a layout miss.
        self._final_layout_hits = 0
        self._final_layout_misses = 0
        self._final_layout_evictions = 0
        self._final_layout_guard_rejects = 0
        # Standalone cohort duration is an exact zero-origin replay.  The
        # planner tuple is identity-guarded above and immutable for the life
        # of this cache; only runtime overlay stages can add timing inputs.
        # Retain a bounded result memo so warmup/measured replays do not drain
        # the same tens-of-thousands-task isolated graph again.
        self._isolated_durations: OrderedDict[
            Tuple[int, int, Tuple[object, ...], Tuple[Tuple[str, int], ...]],
            Tuple[object, _ExecutionStageReplayLayout, float],
        ] = OrderedDict()
        self._isolated_duration_hits = 0
        self._isolated_duration_misses = 0
    @staticmethod
    def _final_layout_key(
        stages: Sequence[_ExecutionStage],
    ) -> Optional[_ExecutionStageLayoutKey]:
        """Return the complete id-independent structure of the final graph."""

        stage_positions = {
            stage.stage_id: position for position, stage in enumerate(stages)
        }
        if len(stage_positions) != len(stages):
            return None
        task_positions_by_stage: List[Dict[str, int]] = []
        for stage in stages:
            task_positions = {
                task.task_id: position
                for position, task in enumerate(stage.execution_tasks)
            }
            if len(task_positions) != len(stage.execution_tasks):
                return None
            task_positions_by_stage.append(task_positions)

        normalized_stages: List[
            Tuple[
                Tuple[int, ...],
                Tuple[
                    Tuple[
                        Tuple[int, ...],
                        Tuple[str, ...],
                        bool,
                        Tuple[int, int],
                    ],
                    ...,
                ],
            ]
        ] = []
        try:
            for stage_position, stage in enumerate(stages):
                task_positions = task_positions_by_stage[stage_position]
                normalized_tasks = tuple(
                    (
                        tuple(
                            task_positions[dependency]
                            for dependency in task.dependencies
                        ),
                        tuple(
                            demand.resource_id for demand in task.demands
                        ),
                        bool(task.opaque_device_fence),
                        _validated_phase_sequence(task),
                    )
                    for task in stage.execution_tasks
                )
                normalized_stages.append(
                    (
                        tuple(
                            stage_positions[dependency]
                            for dependency in stage.dependencies
                        ),
                        normalized_tasks,
                    )
                )
        except KeyError:
            return None
        return tuple(normalized_stages)

    @staticmethod
    def _layout_matches_stages(
        layout: _ExecutionStageReplayLayout,
        stages: Sequence[_ExecutionStage],
        structure_key: _ExecutionStageLayoutKey,
    ) -> bool:
        if layout.structure_key is not None:
            return layout.structure_key == structure_key
        zero_ready = {stage.stage_id: 0.0 for stage in stages}
        specs = layout.instantiate(stages, zero_ready)
        return layout.compiled.matches(specs)

    def _cached_final_layout_key(
        self,
        stages: Sequence[_ExecutionStage],
    ) -> Optional[_ExecutionStageLayoutKey]:
        """Reuse normalized task topology for identical prepared task tuples."""

        sources = tuple(stage.execution_tasks for stage in stages)
        identity_key = tuple(
            (
                stage.stage_id,
                stage.dependencies,
                id(stage.execution_tasks),
            )
            for stage in stages
        )
        cached = self._identity_final_layout_keys.get(identity_key)
        if cached is not None and len(cached[0]) == len(sources) and all(
            retained is source
            for retained, source in zip(cached[0], sources)
        ):
            self._identity_final_layout_keys.move_to_end(identity_key)
            return cached[1]
        stage_positions = {
            stage.stage_id: position for position, stage in enumerate(stages)
        }
        if len(stage_positions) != len(stages):
            return None
        normalized_stages = []
        try:
            for stage in stages:
                task_source = stage.execution_tasks
                task_identity = id(task_source)
                cached_tasks = self._identity_task_layout_keys.get(task_identity)
                if cached_tasks is not None and cached_tasks[0] is task_source:
                    normalized_tasks = cached_tasks[1]
                    self._identity_task_layout_keys.move_to_end(task_identity)
                else:
                    task_positions = {
                        task.task_id: position
                        for position, task in enumerate(task_source)
                    }
                    if len(task_positions) != len(task_source):
                        return None
                    normalized_tasks = tuple(
                        (
                            tuple(
                                task_positions[dependency]
                                for dependency in task.dependencies
                            ),
                            tuple(
                                demand.resource_id for demand in task.demands
                            ),
                            bool(task.opaque_device_fence),
                            _validated_phase_sequence(task),
                        )
                        for task in task_source
                    )
                    self._identity_task_layout_keys[task_identity] = (
                        task_source,
                        normalized_tasks,
                    )
                    self._identity_task_layout_keys.move_to_end(task_identity)
                    task_layout_capacity = self.max_final_layouts * 256
                    while (
                        len(self._identity_task_layout_keys)
                        > task_layout_capacity
                    ):
                        self._identity_task_layout_keys.popitem(last=False)
                normalized_stages.append(
                    (
                        tuple(
                            stage_positions[dependency]
                            for dependency in stage.dependencies
                        ),
                        normalized_tasks,
                    )
                )
        except KeyError:
            return None
        key: _ExecutionStageLayoutKey = tuple(normalized_stages)
        self._identity_final_layout_keys[identity_key] = (sources, key)
        self._identity_final_layout_keys.move_to_end(identity_key)
        while len(self._identity_final_layout_keys) > self.max_final_layouts:
            self._identity_final_layout_keys.popitem(last=False)
        return key

    def resolve(
        self, metadata: Mapping[str, Any]
    ) -> Tuple[Tuple[_ExecutionStage, ...], Optional[str]]:
        raw_stages = metadata.get("execution_stages")
        if (
            metadata.get("execution_stage_source")
            != "executed_task_dag_kernel_timeline"
            or not isinstance(raw_stages, tuple)
        ):
            return _execution_stages_from_metadata(metadata)
        key = id(raw_stages)
        cached = self._values.get(key)
        if cached is not None and cached[0] is raw_stages:
            return cached[1]
        prepared_stages = _trusted_execution_stages(raw_stages)
        trusted_handoff = prepared_stages is not None
        parsed = (
            (prepared_stages, None)
            if trusted_handoff
            else _execution_stages_from_metadata(metadata)
        )
        if len(self._values) >= self.max_entries:
            # Dict insertion order gives a small, deterministic FIFO bound.
            self._values.pop(next(iter(self._values)))
        self._values[key] = (raw_stages, parsed, trusted_handoff)
        return parsed

    def replay_layout(
        self,
        metadata: Mapping[str, Any],
        stages: Sequence[_ExecutionStage],
        *,
        has_runtime_extension: bool,
        runtime_overlay_stage_count: int = 0,
    ) -> Optional[_ExecutionStageReplayLayout]:
        """Return an identity-guarded structural layout after validation."""

        raw_stages = metadata.get("execution_stages")
        if (
            metadata.get("execution_stage_source")
            != "executed_task_dag_kernel_timeline"
            or not isinstance(raw_stages, tuple)
        ):
            return None
        cached = self._values.get(id(raw_stages))
        if cached is None or cached[0] is not raw_stages:
            return None
        if not cached[2]:
            return None
        parsed_stages, reason = cached[1]
        if runtime_overlay_stage_count < 0:
            return None
        expected_count = (
            len(parsed_stages)
            + int(has_runtime_extension)
            + runtime_overlay_stage_count
        )
        if reason is not None or len(stages) != expected_count:
            return None
        key = self._cached_final_layout_key(stages)
        if key is None:
            return None
        layout = self._final_layouts.get(key)
        if layout is not None:
            # The tuple key is complete rather than digest-only, but the event
            # kernel's own structural predicate remains the final safety
            # guard before its private compiled-layout constructor is used.
            if not self._layout_matches_stages(layout, stages, key):
                # A guard failure means the key and the retained compiled
                # layout disagree (for example after a future key-schema
                # regression or accidental in-process mutation).  Never use
                # the suspect entry; discard and rebuild it from the live
                # specs so the caller remains on the exact replay path.
                self._final_layout_guard_rejects += 1
                self._final_layouts.pop(key, None)
            else:
                self._final_layout_hits += 1
                self._final_layouts.move_to_end(key)
                return layout
        self._final_layout_misses += 1
        layout = _ExecutionStageReplayLayout.compile(
            stages,
            structure_key=key,
        )
        self._final_layouts[key] = layout
        if len(self._final_layouts) > self.max_final_layouts:
            self._final_layouts.popitem(last=False)
            self._final_layout_evictions += 1
        return layout

    @staticmethod
    def _runtime_overlay_timing_key(
        stages: Sequence[_ExecutionStage],
    ) -> Tuple[object, ...]:
        """Return exact timing inputs not owned by the cached planner tuple."""

        overlays: List[object] = []
        for stage in stages:
            runtime_stage = stage.stage_id.startswith("runtime.")
            runtime_tasks = tuple(
                (
                    task.task_id,
                    task.dependencies,
                    tuple(
                        (
                            demand.resource_id,
                            float(demand.service_ns).hex(),
                        )
                        for demand in task.demands
                    ),
                )
                for task in stage.execution_tasks
                if runtime_stage or task.task_id.startswith("runtime.")
            )
            if not runtime_stage and not runtime_tasks:
                continue
            # ``raw_stages`` and the replay-layout identity own the immutable
            # planner DAG.  Runtime tasks may be injected into those planner
            # stages, so retain the containing stage topology while excluding
            # base planner task/demand details from this overlay-only key.  A
            # runtime-owned stage retains every task because its task ids need
            # not themselves use the reserved runtime namespace.
            overlays.append(
                (
                    stage.stage_id,
                    stage.dependencies,
                    runtime_tasks,
                )
            )
        return tuple(overlays)

    def isolated_duration(
        self,
        raw_stages: object,
        replay_layout: _ExecutionStageReplayLayout,
        stages: Sequence[_ExecutionStage],
        resource_capacities: Mapping[str, int],
        *,
        overlay_timing_key: Optional[Tuple[object, ...]] = None,
    ) -> Optional[float]:
        """Return an exact memoized zero-origin duration, if present."""

        if overlay_timing_key is None:
            overlay_timing_key = self._runtime_overlay_timing_key(stages)
        key = (
            id(raw_stages),
            id(replay_layout),
            overlay_timing_key,
            tuple(sorted(resource_capacities.items())),
        )
        cached = self._isolated_durations.get(key)
        if (
            cached is None
            or cached[0] is not raw_stages
            or cached[1] is not replay_layout
        ):
            self._isolated_duration_misses += 1
            return None
        self._isolated_duration_hits += 1
        self._isolated_durations.move_to_end(key)
        return cached[2]

    def remember_isolated_duration(
        self,
        raw_stages: object,
        replay_layout: _ExecutionStageReplayLayout,
        stages: Sequence[_ExecutionStage],
        resource_capacities: Mapping[str, int],
        duration_ns: float,
        *,
        overlay_timing_key: Optional[Tuple[object, ...]] = None,
    ) -> None:
        """Memoize one already executed, exact zero-origin task DAG."""

        if overlay_timing_key is None:
            overlay_timing_key = self._runtime_overlay_timing_key(stages)
        key = (
            id(raw_stages),
            id(replay_layout),
            overlay_timing_key,
            tuple(sorted(resource_capacities.items())),
        )
        self._isolated_durations[key] = (
            raw_stages,
            replay_layout,
            duration_ns,
        )
        self._isolated_durations.move_to_end(key)
        if len(self._isolated_durations) > self.max_entries:
            self._isolated_durations.popitem(last=False)

    @property
    def final_layout_cache_stats(self) -> Mapping[str, int]:
        """Return immutable counters for the normalized replay-layout LRU."""

        return {
            "hits": self._final_layout_hits,
            "misses": self._final_layout_misses,
            "evictions": self._final_layout_evictions,
            "guard_rejects": self._final_layout_guard_rejects,
            "size": len(self._final_layouts),
            "capacity": self.max_final_layouts,
            "isolated_duration_hits": self._isolated_duration_hits,
            "isolated_duration_misses": self._isolated_duration_misses,
            "isolated_duration_size": len(self._isolated_durations),
        }

    @property
    def final_layout_stats(self) -> Mapping[str, int]:
        """Compatibility alias for callers that use the shorter name."""

        return self.final_layout_cache_stats

    @property
    def stats(self) -> Mapping[str, int]:
        """Expose the layout counters using the event-kernel cache idiom."""

        return self.final_layout_cache_stats


def _execution_stage_cache_for_lowerer(
    lowerer: object,
) -> _ExecutionStageMetadataCache:
    cache = getattr(lowerer, "_execution_stage_metadata_cache", None)
    if isinstance(cache, _ExecutionStageMetadataCache):
        return cache
    cache = _ExecutionStageMetadataCache()
    try:
        setattr(lowerer, "_execution_stage_metadata_cache", cache)
    except (AttributeError, TypeError):
        # Slot-based or otherwise immutable custom lowerers still get a
        # runtime-local cache without changing their public contract.
        pass
    return cache

@dataclass(frozen=True)
class ServingBatch:
    cohort_id: str
    kind: str
    start_ns: float
    end_ns: float
    request_ids: Tuple[str, ...]
    token_count: int
    cost: BatchCost
    items: Tuple[BatchItem, ...]
    proposal_cost_scale: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingEvent:
    timestamp_ns: float
    event_type: str
    request_id: Optional[str] = None
    cohort_id: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingRequestState:
    request_id: str
    status: RequestStatus
    arrival_ns: float
    prompt_tokens: int
    output_tokens: int
    prefill_cursor: int
    committed: int
    proposed: int
    accepted: int
    priority: int
    deadline_ns: Optional[float]
    kv_pages: int
    peak_kv_pages: int
    started_ns: Optional[float]
    first_token_ns: Optional[float]
    finished_ns: Optional[float]
    preemptions: int
    swaps: int
    recomputes: int
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class ServingRequestMetrics:
    request_id: str
    status: RequestStatus
    arrival_ns: float
    start_ns: Optional[float]
    first_token_ns: Optional[float]
    finish_ns: Optional[float]
    queue_delay_ns: Optional[float]
    ttft_ns: Optional[float]
    tpot_ns: Optional[float]
    prompt_tokens: int
    requested_output_tokens: int
    visible_output_tokens: int
    proposed_tokens: int
    accepted_tokens: int
    preemptions: int
    swaps: int
    recomputes: int
    deadline_met: Optional[bool]
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class KVCacheMetrics:
    tokens_per_page: int
    bytes_per_page: int
    capacity_pages: int
    capacity_bytes: int
    peak_used_pages: int
    peak_used_bytes: int
    allocation_events: int
    release_events: int
    swap_events: int
    swap_bytes: int
    recompute_events: int
    recompute_tokens: int
    offload_peak_bytes: int
    rejected_requests: int
    swap_in_bytes: int = 0
    swap_transfer_time_ns: float = 0.0
    swap_transfer_energy_pj: float = 0.0
    logical_prefill_read_bytes: int = 0
    logical_prefill_write_bytes: int = 0
    logical_decode_read_bytes: int = 0
    logical_decode_write_bytes: int = 0
    logical_decode_append_bytes: int = 0
    physical_prefill_read_bytes: int = 0
    physical_prefill_write_bytes: int = 0
    physical_decode_read_bytes: int = 0
    physical_decode_write_bytes: int = 0
    physical_decode_append_bytes: int = 0
    prefetch_events: int = 0
    prefetch_bytes: int = 0
    offload_events: int = 0
    offload_bytes: int = 0
    migration_events: int = 0
    migration_bytes: int = 0
    swap_out_bytes: int = 0
    physical_swap_bytes: int = 0
    logical_bytes_per_token: int = 0
    logical_swap_out_bytes: int = 0
    logical_swap_in_bytes: int = 0
    logical_migration_bytes: int = 0
    physical_offload_bytes: int = 0
    max_live_tokens_per_request: int = 0
    peak_semantics: str = "realized_prompt_plus_output_minus_one"
    prefetch_distance_modeled: bool = False
    traffic_semantics: str = "logical_kv_traffic_separate_from_resource_accounting"
    persistent_peak_used_pages: int = 0
    persistent_peak_used_bytes: int = 0
    mtp_materialized_tokens: int = 0
    mtp_temporary_tokens: int = 0
    mtp_temporary_peak_pages: int = 0
    mtp_temporary_peak_bytes: int = 0
    mtp_temporary_allocation_events: int = 0
    mtp_temporary_release_events: int = 0
    logical_mtp_materialized_write_bytes: int = 0
    physical_mtp_materialized_write_bytes: int = 0
    logical_mtp_temporary_write_bytes: int = 0
    physical_mtp_temporary_write_bytes: int = 0
    logical_mtp_verification_read_bytes: int = 0
    physical_mtp_verification_read_bytes: int = 0


@dataclass(frozen=True)
class LinearStateMetrics:
    bytes_per_request: int
    capacity_bytes: int
    capacity_requests: int
    peak_used_bytes: int
    peak_used_requests: int
    allocation_events: int
    release_events: int
    offload_events: int
    offload_bytes: int
    offload_peak_bytes: int
    restore_events: int
    swap_in_bytes: int = 0
    swap_transfer_time_ns: float = 0.0
    swap_transfer_energy_pj: float = 0.0
    swap_routed_bytes: int = 0


@dataclass(frozen=True)
class PromptCacheMetrics:
    """Realized prompt-cache allocation, residency, and eviction facts."""

    enabled: bool = False
    context_checkpoints: int = 0
    component: Optional[str] = None
    offload_component: Optional[str] = None
    entry_count: int = 0
    allocation_events: int = 0
    logical_allocated_bytes: int = 0
    peak_logical_allocated_bytes: int = 0
    logical_capacity_bytes: int = 0
    logical_overcommit_bytes: int = 0
    resident_bytes: int = 0
    peak_resident_bytes: int = 0
    resident_capacity_bytes: int = 0
    resident_overcommit_bytes: int = 0
    host_backed_bytes: int = 0
    peak_host_backed_bytes: int = 0
    unbacked_bytes: int = 0
    logical_entry_bytes: int = 0
    graph_resident_entry_bytes: int = 0
    graph_resident_segments: int = 0
    eviction_events: int = 0
    eviction_entries: int = 0
    eviction_bytes: int = 0
    refault_events: int = 0
    refault_bytes: int = 0
    checkpoint_segments_allocated: int = 0
    checkpoint_segments_resident: int = 0
    observed_entry_sizes_bytes: Tuple[int, ...] = ()
    residency_mode: str = "actual_tokens"
    semantics: str = (
        "logical_full_slot_commitment_separate_from_graph_referenced_residency"
    )


@dataclass(frozen=True)
class OwnerResidencyMetrics:
    """Realized facts from the optional unified owner-aware physical pool."""

    enabled: bool = False
    component_id: Optional[str] = None
    capacity_bytes: int = 0
    committed_bytes: int = 0
    resident_bytes: int = 0
    peak_resident_bytes: int = 0
    available_bytes: int = 0
    allocation_count: int = 0
    view_count: int = 0
    access_count: int = 0
    weight_access_count: int = 0
    kv_access_count: int = 0
    state_access_count: int = 0
    temporary_allocation_count: int = 0
    temporary_release_count: int = 0
    migration_event_count: int = 0
    fault_batch_count: int = 0
    clean_eviction_batch_count: int = 0
    page_in_bytes: int = 0
    page_out_bytes: int = 0
    clean_discard_bytes: int = 0
    dirty_writeback_bytes: int = 0
    clean_discard_time_ns: float = 0.0
    clean_discard_energy_pj: float = 0.0
    transfer_time_ns: float = 0.0
    transfer_energy_pj: float = 0.0
    initialization_migration_event_count: int = 0
    initialization_page_in_bytes: int = 0
    initialization_page_out_bytes: int = 0
    initialization_clean_discard_bytes: int = 0
    initialization_dirty_writeback_bytes: int = 0
    timing_completeness: str = "complete"
    submission_latency_known: bool = True
    unmodeled_timing_terms: Tuple[str, ...] = ()
    # Per-runtime replay-window residency.  These fields deliberately do not
    # replace ``resident_bytes``/``peak_resident_bytes``: the latter are the
    # manager's current/cumulative session facts, while these values describe
    # only the interval beginning when this runtime received its manager.
    interval_baseline_resident_bytes: int = 0
    interval_peak_resident_bytes: int = 0
    interval_resident_delta_bytes: int = 0


@dataclass(frozen=True)
class SchedulerMetrics:
    scheduling_rounds: int
    total_batches: int
    prefill_batches: int
    decode_batches: int
    mtp_batches: int
    preemptions: int
    priority_preemptions: int
    memory_preemptions: int
    max_batch_sequences: int
    max_batch_tokens: int
    idle_time_ns: float
    kv_transfer_batches: int = 0
    linear_state_transfer_batches: int = 0


@dataclass(frozen=True)
class ServingResult:
    plan: ServingPlan
    events: Tuple[ServingEvent, ...]
    batches: Tuple[ServingBatch, ...]
    request_states: Mapping[str, ServingRequestState]
    request_metrics: Mapping[str, ServingRequestMetrics]
    kv_metrics: KVCacheMetrics
    linear_state_metrics: LinearStateMetrics
    scheduler_metrics: SchedulerMetrics
    makespan_ns: float
    prompt_cache_metrics: PromptCacheMetrics = field(
        default_factory=PromptCacheMetrics
    )
    owner_residency_metrics: OwnerResidencyMetrics = field(
        default_factory=OwnerResidencyMetrics
    )
    runtime_kernel_metrics: Mapping[str, object] = field(default_factory=dict)

    @property
    def trace(self) -> Tuple[ServingEvent, ...]:
        return self.events

    @property
    def prompt_cache_save_timing(self) -> Mapping[str, object]:
        saves = [event for event in self.events if event.event_type == "prompt_cache_save_end"]
        if not saves:
            return {}
        return {
            "scope": "prompt_cache_save_only",
            "save_events": len(saves),
            "timing_completeness": "partial",
            "known_service_ns": sum(float(event.details["service_ns"]) for event in saves),
            "total_service_ns": None,
            "unmodeled_timing_terms": tuple(sorted({
                str(term) for event in saves for term in event.details.get(
                    "unmodeled_timing_terms", ("legacy_prompt_save_proxy_incomplete",)
                )
            })),
            "request_latency_semantics": "modeled_components_only_not_complete_latency",
        }


def _uniform_rank_profile(
    scenario: ScenarioConfig,
    component_ids: Sequence[str],
    expected_type: type,
    label: str,
) -> Tuple[Any, Tuple[str, ...]]:
    """Resolve aggregate rank owners and reject a non-equivalent profile mix."""

    unique_ids = tuple(sorted({str(item) for item in component_ids if item}))
    if not unique_ids:
        raise ValueError(
            "analytical batch roofline has no {} rank owner".format(label)
        )
    profiles = tuple(
        scenario.resolve_component_profile(component_id, expected_type)
        for component_id in unique_ids
    )
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise ValueError(
            "analytical batch roofline cannot aggregate non-equivalent {} "
            "profiles across components {}".format(
                label, ", ".join(unique_ids)
            )
        )
    return profiles[0], unique_ids


class BatchCostProvider(Protocol):
    """Structural interface for a calibrated deterministic cohort lowerer."""

    def estimate(self, scenario: ScenarioConfig, cohort: BatchCohort) -> BatchCost:
        ...


BatchLowerer = Union[
    BatchCostProvider,
    Callable[[ScenarioConfig, BatchCohort], Union[BatchCost, float, Mapping[str, Any]]],
]


def compile_serving_plan(scenario: ScenarioConfig) -> ServingPlan:
    """Compile serving policy and capacity with one graph projection."""

    with _compilation_scope(scenario):
        return _compile_serving_plan_in_context(scenario)


def _compile_serving_plan_in_context(scenario: ScenarioConfig) -> ServingPlan:
    """Compile the typed V4 scheduler, KV, MTP, and capacity contracts."""

    scheduler_spec = scenario.workload.scheduler
    if scheduler_spec is None:
        raise ValueError("V4 workload.scheduler is required")
    max_num_seqs = int(scheduler_spec.max_num_seqs)
    prefill_chunk = int(scheduler_spec.prefill_chunk_tokens)
    requested_max_tokens = int(scheduler_spec.max_num_batched_tokens)
    raw_ubatch_tokens = scheduler_spec.max_num_ubatch_tokens
    requested_max_ubatch_tokens = (
        requested_max_tokens
        if raw_ubatch_tokens is None
        else int(raw_ubatch_tokens)
    )
    # n_batch/n_ubatch are submission budgets and may exceed one slot's
    # context length.  Per-request context admission is validated separately;
    # clipping these values here would silently change the llama.cpp runtime
    # contract for multi-request batches.
    max_tokens = requested_max_tokens
    max_ubatch_tokens = min(requested_max_ubatch_tokens, max_tokens)
    if (
        max_num_seqs <= 0
        or prefill_chunk <= 0
        or max_tokens <= 0
        or max_ubatch_tokens <= 0
    ):
        raise ValueError("serving sequence, token, and prefill chunk budgets must be positive")
    scheduler = ServingPolicy(
        mode=str(scheduler_spec.mode),
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_tokens,
        max_num_ubatch_tokens=max_ubatch_tokens,
        prefill_chunk_tokens=min(prefill_chunk, max_tokens),
        prefill_stop_offsets=scheduler_spec.prefill_stop_offsets,
        mixed_phase_batching=bool(scheduler_spec.mixed_phase_batching),
        policy=str(scheduler_spec.policy),
        phase_candidate_order=str(scheduler_spec.phase_candidate_order),
        starvation_ns=float(scheduler_spec.starvation_ns),
        preemption_enabled=bool(scheduler_spec.preemption_enabled),
        preemption_granularity=str(scheduler_spec.preemption_granularity),
        preemption_policy=str(scheduler_spec.preemption_policy),
        slo_ttft_ns=_optional_float(scheduler_spec.slo_ttft_ns),
        slo_tbt_ns=_optional_float(scheduler_spec.slo_tbt_ns),
    )
    if scheduler.starvation_ns <= 0:
        raise ValueError("scheduler starvation_ns must be positive")
    if scheduler.preemption_granularity != "boundary":
        raise ValueError("online preemption is supported only at unit boundaries")
    if scheduler.preemption_policy not in ("auto", "swap", "recompute"):
        raise ValueError("unsupported scheduler preemption_policy")

    kv_spec = scenario.placement.kv_policy
    if kv_spec is None:
        raise ValueError("V4 placement.kv_policy is required")
    # ``kv_policy`` carries user policy while ``tensor_to_component`` carries
    # the concrete location selected by the control plane.  Keep an explicit
    # policy target authoritative, but allow the generated tensor placement to
    # supply the active KV component when the policy intentionally leaves it
    # unspecified.
    cache_component = _optional_str(
        kv_spec.cache_component
        or scenario.placement.tensor_to_component.get("kv_cache")
    )
    offload_component = _optional_str(kv_spec.offload_component)
    page_tokens = int(kv_spec.tokens_per_page)
    if page_tokens <= 0:
        raise ValueError("KV tokens_per_page must be positive")
    kv_dtype = _optional_str(kv_spec.dtype)
    physical_bytes_per_token = _kv_bytes_per_token(scenario, kv_dtype)
    logical_bytes_per_token = _logical_kv_bytes_per_token(scenario, kv_dtype)
    bytes_per_page = physical_bytes_per_token * page_tokens
    logical_linear_state_bytes = _linear_state_bytes_per_request(scenario)
    state_contract = _linear_state_runtime_contract(scenario)
    contract_linear_state_bytes = _linear_state_contract_live_bytes(
        state_contract
    )
    linear_state_bytes = (
        contract_linear_state_bytes or logical_linear_state_bytes
    )
    state_cache_component = _optional_str(
        scenario.placement.tensor_to_component.get("linear_state")
        or state_contract.get("component_id")
        or cache_component
    )
    state_offload_component = _optional_str(
        scenario.placement.tensor_to_component.get("linear_state_offload")
    )
    state_offload_ratio = float(
        scenario.placement.metadata.get("linear_state_offload_ratio", 1.0)
    )
    if not 0.0 <= state_offload_ratio <= 1.0:
        raise ValueError("linear_state_offload_ratio must be in [0, 1]")
    contract_state_capacity = _linear_state_contract_capacity(
        state_contract
    )
    declared_state_capacity = (
        contract_state_capacity
        if contract_state_capacity > 0
        else _declared_runtime_tensor_capacity(scenario, "linear_state")
    )
    active_dynamic_tensors = {"linear_state"}
    if state_cache_component == cache_component:
        active_dynamic_tensors.add("kv_cache")
    component_state_capacity = _dynamic_component_capacity(
        scenario,
        state_cache_component,
        active_dynamic_tensors,
    )
    if _strict_unknown_storage_capacity(scenario, state_cache_component):
        # V4 treats an omitted/zero physical capacity as unknown.  Keep the
        # runtime plan fail-closed so admission explains the rejected set.
        state_capacity = 0
    elif linear_state_bytes <= 0:
        state_capacity = 0
    elif declared_state_capacity > 0:
        state_capacity = (
            min(declared_state_capacity, component_state_capacity)
            if component_state_capacity
            else declared_state_capacity
        )
    else:
        requested_state_capacity = linear_state_bytes * max_num_seqs
        state_capacity = (
            min(requested_state_capacity, component_state_capacity)
            if component_state_capacity
            else requested_state_capacity
        )
    state_capacity_requests = (
        state_capacity // linear_state_bytes if linear_state_bytes else 0
    )
    state_capacity = state_capacity_requests * linear_state_bytes
    capacity_bytes = _kv_capacity_bytes(scenario, cache_component)
    explicit_kv_capacity = (
        _declared_runtime_tensor_capacity(scenario, "kv_cache") > 0
    )
    explicit_state_capacity = declared_state_capacity > 0
    if state_cache_component == cache_component and cache_component:
        active_budget = _dynamic_component_capacity(
            scenario,
            cache_component,
            {"kv_cache", "linear_state"},
        )
        if capacity_bytes > active_budget:
            if explicit_kv_capacity:
                raise ValueError(
                    "KV capacity exceeds shared physical capacity on {}".format(
                        cache_component
                    )
                )
            capacity_bytes = active_budget
        if state_capacity + capacity_bytes > active_budget:
            if explicit_state_capacity and explicit_kv_capacity:
                raise ValueError(
                    "KV and linear-state capacities exceed shared physical "
                    "capacity on {}".format(cache_component)
                )
            if explicit_kv_capacity:
                state_capacity = max(0, active_budget - capacity_bytes)
                state_capacity -= state_capacity % linear_state_bytes
                state_capacity_requests = (
                    state_capacity // linear_state_bytes
                    if linear_state_bytes
                    else 0
                )
            else:
                capacity_bytes = max(0, active_budget - state_capacity)
    capacity_pages = capacity_bytes // bytes_per_page if bytes_per_page else 0
    offload_ratio = float(kv_spec.offload_ratio)
    if not 0.0 <= offload_ratio <= 1.0:
        raise ValueError("KV offload_ratio must be in [0, 1]")
    shared_offload = (
        state_offload_component is not None
        and state_offload_component == offload_component
    )
    state_offload_dynamic = {"linear_state_offload"}
    kv_offload_dynamic = {"kv_cache_offload"}
    if shared_offload:
        state_offload_dynamic.add("kv_cache_offload")
        kv_offload_dynamic.add("linear_state_offload")
    state_offload_budget = _dynamic_component_capacity(
        scenario,
        state_offload_component,
        state_offload_dynamic,
    )
    declared_state_offload = _declared_runtime_tensor_capacity(
        scenario, "linear_state_offload"
    )
    if declared_state_offload > 0:
        state_offload_capacity = min(
            declared_state_offload, state_offload_budget
        )
    else:
        state_offload_capacity = min(
            state_capacity,
            int(state_offload_budget * state_offload_ratio),
        )
    offload_budget = _dynamic_component_capacity(
        scenario,
        offload_component,
        kv_offload_dynamic,
    )
    declared_kv_offload = _declared_runtime_tensor_capacity(
        scenario, "kv_cache_offload"
    )
    offload_capacity = (
        min(declared_kv_offload, offload_budget)
        if declared_kv_offload > 0
        else int(offload_budget * offload_ratio)
    )
    if shared_offload:
        offload_capacity = max(
            0, min(offload_capacity, offload_budget - state_offload_capacity)
        )
    normalized_capacities = _normalize_physical_role_capacities(
        scenario,
        (
            (
                "linear_state_active",
                state_cache_component,
                state_capacity,
                explicit_state_capacity,
                max(1, linear_state_bytes),
                "linear_state",
            ),
            (
                "kv_active",
                cache_component,
                capacity_pages * bytes_per_page,
                explicit_kv_capacity,
                max(1, bytes_per_page),
                "kv_cache",
            ),
            (
                "linear_state_offload",
                state_offload_component,
                state_offload_capacity,
                declared_state_offload > 0,
                max(1, linear_state_bytes),
                "linear_state_offload",
            ),
            (
                "kv_offload",
                offload_component,
                offload_capacity,
                declared_kv_offload > 0,
                max(1, bytes_per_page),
                "kv_cache_offload",
            ),
        ),
    )
    state_capacity = normalized_capacities["linear_state_active"]
    state_capacity_requests = (
        state_capacity // linear_state_bytes if linear_state_bytes else 0
    )
    capacity_bytes = normalized_capacities["kv_active"]
    capacity_pages = capacity_bytes // bytes_per_page if bytes_per_page else 0
    state_offload_capacity = normalized_capacities["linear_state_offload"]
    offload_capacity = normalized_capacities["kv_offload"]
    linear_state_policy = LinearStatePolicy(
        cache_component=state_cache_component,
        offload_component=state_offload_component,
        bytes_per_request=linear_state_bytes,
        capacity_bytes=state_capacity,
        capacity_requests=state_capacity_requests,
        offload_capacity_bytes=state_offload_capacity,
        offload_ratio=state_offload_ratio,
    )
    kv_policy = KVCachePolicy(
        cache_component=cache_component,
        offload_component=offload_component,
        tokens_per_page=page_tokens,
        dtype=kv_dtype,
        offload_ratio=offload_ratio,
        allocation_policy=str(kv_spec.allocation_policy),
        preemption_mode=str(kv_spec.preemption_mode),
        prefetch_distance=int(kv_spec.prefetch_distance),
        bytes_per_page=bytes_per_page,
        logical_bytes_per_token=logical_bytes_per_token,
        capacity_bytes=capacity_pages * bytes_per_page,
        capacity_pages=capacity_pages,
        offload_capacity_bytes=offload_capacity,
    )
    if kv_policy.allocation_policy not in ("lazy", "eager"):
        raise ValueError("KV allocation_policy must be lazy or eager")
    if kv_policy.preemption_mode not in ("auto", "swap", "recompute"):
        raise ValueError("unsupported KV preemption_mode")
    if kv_policy.prefetch_distance < 0:
        raise ValueError("KV prefetch_distance must be non-negative")
    if kv_policy.prefetch_distance > 0:
        raise ValueError(
            "V4 placement does not support non-zero prefetch_distance; "
            "proactive KV lookahead is not modeled"
        )

    mtp_spec = scenario.workload.mtp
    if mtp_spec is None:
        mtp = MTPPolicy()
    else:
        raw_acceptance_model = str(mtp_spec.acceptance_model).strip().lower()
        if raw_acceptance_model in ("expected", "expected_prefix"):
            acceptance_model = "expected"
        elif raw_acceptance_model == "trace":
            acceptance_model = "trace"
        else:
            raise ValueError(
                "unsupported MTP acceptance_model: {}".format(
                    mtp_spec.acceptance_model
                )
            )
        rate = float(mtp_spec.acceptance_rate or 0.0)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("MTP acceptance_rate must be in [0, 1]")
        acceptance_trace = tuple(
            float(value) for value in mtp_spec.acceptance_trace
        )
        if acceptance_model == "trace" and not acceptance_trace:
            raise ValueError(
                "MTP acceptance_model='trace' requires acceptance_trace"
            )
        mtp = MTPPolicy(
            method=str(mtp_spec.method),
            candidate_tokens=int(mtp_spec.candidate_tokens),
            min_draft_tokens=int(mtp_spec.min_draft_tokens),
            continuation_threshold=(
                None
                if mtp_spec.continuation_threshold is None
                else float(mtp_spec.continuation_threshold)
            ),
            proposal_length_model=str(mtp_spec.proposal_length_model),
            expected_draft_tokens_per_round=(
                None
                if mtp_spec.expected_draft_tokens_per_round is None
                else float(mtp_spec.expected_draft_tokens_per_round)
            ),
            draft_length_trace=tuple(
                int(value) for value in mtp_spec.draft_length_trace
            ),
            acceptance_model=acceptance_model,
            acceptance_rate=rate,
            proposal_cost_scale=float(mtp_spec.proposal_cost_scale),
            acceptance_trace=acceptance_trace,
        )
    if mtp.proposal_cost_scale <= 0:
        raise ValueError("MTP proposal_cost_scale must be positive")

    if scenario.workload.requests:
        requests = []
        for request in materialize_requests(scenario):
            metadata = request.metadata if isinstance(request.metadata, _ABCMapping) else {}
            priority = int(getattr(request, "priority", metadata.get("priority", 0)))
            deadline = getattr(request, "deadline_ns", metadata.get("deadline_ns"))
            requests.append(
                ServingRequest(
                    request_id=request.request_id,
                    arrival_ns=float(request.arrival_ns),
                    prompt_tokens=int(request.prompt_tokens),
                    output_tokens=int(request.output_tokens),
                    priority=priority,
                    deadline_ns=_optional_float(deadline),
                )
            )
        requests.sort(key=lambda item: (item.arrival_ns, item.request_id))
        plan_requests: Sequence[ServingRequest] = tuple(requests)
    elif scenario.workload.request_count > 0:
        plan_requests = _SyntheticServingRequestSequence(scenario)
    else:
        plan_requests = ()
    return ServingPlan(
        scenario,
        plan_requests,
        scheduler,
        kv_policy,
        linear_state_policy,
        mtp,
    )


def _request_admission_reason(
    plan: ServingPlan,
    request: ServingRequest,
) -> Optional[str]:
    required_tokens = _max_live_kv_tokens(
        request.prompt_tokens, request.output_tokens
    )
    page_tokens = plan.kv_policy.tokens_per_page
    required_pages = (
        (required_tokens + page_tokens - 1) // page_tokens
        if required_tokens > 0 and plan.kv_policy.bytes_per_page > 0
        else 0
    )
    required_bytes = required_pages * plan.kv_policy.bytes_per_page
    if required_pages > plan.kv_policy.capacity_pages:
        return (
            "request {} KV working set requires {} bytes ({} pages for {} "
            "tokens), exceeding cache capacity {} bytes ({} pages)"
        ).format(
            request.request_id,
            required_bytes,
            required_pages,
            required_tokens,
            plan.kv_policy.capacity_bytes,
            plan.kv_policy.capacity_pages,
        )

    shared_component = plan.kv_policy.cache_component
    if (
        shared_component
        and shared_component == plan.linear_state_policy.cache_component
    ):
        shared_required = (
            required_bytes + plan.linear_state_policy.bytes_per_request
        )
        shared_capacity = _physical_runtime_limits(plan).get(
            shared_component, math.inf
        )
        if shared_required > shared_capacity:
            return (
                "request {} KV and linear-state working sets require {} bytes, "
                "exceeding shared physical cache capacity {} bytes on {}"
            ).format(
                request.request_id,
                shared_required,
                int(shared_capacity),
                shared_component,
            )
    if (
        plan.linear_state_policy.bytes_per_request > 0
        and plan.linear_state_policy.capacity_requests < 1
    ):
        return (
            "request {} linear state requires {} bytes, exceeding state cache "
            "capacity {} bytes"
        ).format(
            request.request_id,
            plan.linear_state_policy.bytes_per_request,
            plan.linear_state_policy.capacity_bytes,
        )
    full_sequence_tokens = max(0, int(request.prompt_tokens)) + max(
        0, int(request.output_tokens)
    )
    model_limit = _model_max_sequence_length(plan.scenario)
    if model_limit and full_sequence_tokens > model_limit:
        return (
            "request {} sequence has {} tokens, exceeding model "
            "max_sequence_length {}"
        ).format(request.request_id, full_sequence_tokens, model_limit)
    if (
        plan.scheduler.max_num_seqs > 0
        and len(plan.requests) > 0
        and _serving_resource_policy(plan.scenario).enabled
        and plan.kv_policy.allocation_policy == "eager"
    ):
        slot_count = min(plan.scheduler.max_num_seqs, len(plan.requests))
        eager_reason = _eager_slot_admission_reason(plan, slot_count)
        if eager_reason is not None:
            return "request {} {}".format(request.request_id, eager_reason)
    return None


def serving_admission_diagnostics(scenario: ScenarioConfig) -> Tuple[str, ...]:
    """Return deterministic per-request admission failures before simulation."""

    with _compilation_scope(scenario):
        plan = _compile_serving_plan_in_context(scenario)
        requests = (
            plan.requests
            if scenario.workload.requests
            else plan.requests[:1]
        )
        return tuple(
            reason
            for request in requests
            for reason in (_request_admission_reason(plan, request),)
            if reason is not None
        )


def simulate_online(
    scenario: Union[ScenarioConfig, ServingPlan],
    batch_lowerer: Optional[BatchLowerer] = None,
    *,
    execution_control: Optional[ExecutionControl] = None,
    residency_manager: Optional[AllocationResidencyManager] = None,
    apply_measurement_start_residency: bool = False,
    execution_kernel: Optional[UnifiedEventKernel] = None,
    runtime_origin_ns: float = 0.0,
) -> ServingResult:
    """Run deterministic continuous batching and return a realized trace.

    ``residency_manager`` is an optional session-scoped physical residency
    pool.  When supplied, the runtime starts from that pool as-is: allocation
    construction, constructor clears, and any measurement-start reshaping
    have already happened in the owning session.  A fresh manager is built
    only when the argument is omitted.

    The legacy ``measurement_start_residency`` contract is opt-in here.  The
    normal path begins at the constructor-clear endpoint so callers can run a
    real untimed warmup and carry its physical resident/dirty/LRU state into
    measured replays.
    """

    bound_scenario = scenario.scenario if isinstance(scenario, ServingPlan) else scenario
    lowerer = batch_lowerer or TopologyAwareBatchCostProvider(bound_scenario)
    lowerer_context = getattr(lowerer, "_compilation_context", None)
    if getattr(lowerer_context, "scenario", None) is not bound_scenario:
        lowerer_context = None
    with _compilation_scope(bound_scenario, lowerer_context):
        plan = (
            scenario
            if isinstance(scenario, ServingPlan)
            else compile_serving_plan(scenario)
        )
        runtime = _OnlineRuntime(
            plan,
            lowerer,
            execution_control=execution_control,
            residency_manager=residency_manager,
            apply_measurement_start_residency=apply_measurement_start_residency,
            execution_kernel=execution_kernel,
            runtime_origin_ns=runtime_origin_ns,
        )
        return runtime.run()


@dataclass
class _MutableRequest:
    spec: ServingRequest
    status: RequestStatus = RequestStatus.ARRIVALS
    prefill_cursor: int = 0
    committed: int = 0
    proposed: int = 0
    accepted: int = 0
    kv_pages: int = 0
    peak_kv_pages: int = 0
    temporary_kv_pages: int = 0
    queued_since_ns: float = 0.0
    started_ns: Optional[float] = None
    first_token_ns: Optional[float] = None
    finished_ns: Optional[float] = None
    preemptions: int = 0
    swaps: int = 0
    recomputes: int = 0
    swapped_pages: int = 0
    swap_bytes: int = 0
    preemption_strategy: Optional[str] = None
    recompute_cursor: int = 0
    recompute_target: int = 0
    mtp_round: int = 0
    mtp_cursor: Optional[MTPRequestCursor] = None
    # Monotonic runtime service order used to rotate otherwise equivalent
    # phase candidates.  Keep this separate from ``queued_since_ns``: age is
    # a starvation signal, while this sequence is a deterministic
    # least-recently-served tie-breaker for same-arrival work.
    last_service_sequence: int = 0
    # Stable request-lifetime order assigned only after the first successful
    # admission.  A resumed request keeps its original position.
    admission_sequence: Optional[int] = None
    linear_state_resident: bool = False
    linear_state_swapped_bytes: int = 0
    rejection_reason: Optional[str] = None
    kv_cache_range_count: int = 0
    kv_cache_range_tokens: List[int] = field(default_factory=list)
    kv_cache_range_error: Optional[str] = None
    # Captured before logical release; residency is rechecked at save time.
    recurrent_save_identity: Optional[Mapping[str, Any]] = None

    @property
    def cached_tokens(self) -> int:
        # Prefill materializes prompt KV only.  A generated output becomes a
        # KV entry when a later decode/MTP step consumes it, so the live bound
        # remains prompt + output - 1.
        return self.prefill_cursor + max(0, self.committed - 1)


class _PhysicalCapacityLedger:
    def __init__(self, limits: Mapping[str, int]) -> None:
        self.limits = {str(key): max(0, int(value)) for key, value in limits.items()}
        self.used_bytes: Dict[str, int] = {key: 0 for key in self.limits}

    def can_adjust(self, component_id: Optional[str], delta_bytes: int) -> bool:
        if not component_id or delta_bytes <= 0:
            return True
        component = str(component_id)
        limit = self.limits.get(component)
        if limit is None:
            return True
        return self.used_bytes.get(component, 0) + delta_bytes <= limit

    def adjust(self, component_id: Optional[str], delta_bytes: int) -> bool:
        if not self.can_adjust(component_id, delta_bytes):
            return False
        if not component_id or delta_bytes == 0:
            return True
        component = str(component_id)
        if component not in self.limits:
            return True
        updated = self.used_bytes.get(component, 0) + delta_bytes
        if updated < 0:
            raise RuntimeError("physical capacity ledger underflow on {}".format(component))
        self.used_bytes[component] = updated
        return True

    def can_transfer(
        self,
        source_component: Optional[str],
        target_component: Optional[str],
        byte_count: int,
        target_extra_bytes: int = 0,
    ) -> bool:
        """Check an atomic move, optionally followed by target allocation.

        KV and linear-state offload can legally use the same physical component
        for active and spill storage.  Checking the destination before releasing
        the source would reject an otherwise capacity-neutral transfer.  This
        method reasons about the net move first and is shared by both ledgers.
        """

        byte_count = int(byte_count)
        target_extra_bytes = int(target_extra_bytes)
        if byte_count < 0 or target_extra_bytes < 0:
            return False
        source = str(source_component) if source_component else None
        target = str(target_component) if target_component else None
        if source == target:
            if target is None or target not in self.limits:
                return True
            if self.used_bytes.get(target, 0) < byte_count:
                return False
            return (
                self.used_bytes.get(target, 0) + target_extra_bytes
                <= self.limits[target]
            )
        if source is not None and source in self.limits:
            if self.used_bytes.get(source, 0) < byte_count:
                return False
        if target is not None and target in self.limits:
            return (
                self.used_bytes.get(target, 0)
                + byte_count
                + target_extra_bytes
                <= self.limits[target]
            )
        return True

    def transfer(
        self,
        source_component: Optional[str],
        target_component: Optional[str],
        byte_count: int,
        target_extra_bytes: int = 0,
    ) -> bool:
        """Atomically move bytes between physical components."""

        byte_count = int(byte_count)
        target_extra_bytes = int(target_extra_bytes)
        if not self.can_transfer(
            source_component,
            target_component,
            byte_count,
            target_extra_bytes,
        ):
            return False
        source = str(source_component) if source_component else None
        target = str(target_component) if target_component else None
        if source != target:
            if source is not None and source in self.limits:
                self.used_bytes[source] = self.used_bytes.get(source, 0) - byte_count
            if target is not None and target in self.limits:
                self.used_bytes[target] = (
                    self.used_bytes.get(target, 0) + byte_count
                )
        if target is not None and target_extra_bytes > 0 and target in self.limits:
            self.used_bytes[target] = (
                self.used_bytes.get(target, 0) + target_extra_bytes
            )
        return True


class _KVLedger:
    def __init__(
        self, policy: KVCachePolicy, physical: _PhysicalCapacityLedger
    ) -> None:
        self.policy = policy
        self.physical = physical
        self.used_pages = 0
        self.peak_pages = 0
        self.persistent_used_pages = 0
        self.persistent_peak_pages = 0
        self.temporary_used_pages = 0
        self.temporary_peak_pages = 0
        self.temporary_allocations = 0
        self.temporary_releases = 0
        self.allocations = 0
        self.releases = 0
        self.offload_used_bytes = 0
        self.offload_peak_bytes = 0

    def pages_for_tokens(self, token_count: int) -> int:
        if token_count <= 0 or self.policy.bytes_per_page <= 0:
            return 0
        return (token_count + self.policy.tokens_per_page - 1) // self.policy.tokens_per_page

    def can_resize(self, request: _MutableRequest, pages: int) -> bool:
        delta_bytes = (pages - request.kv_pages) * self.policy.bytes_per_page
        return (
            self.used_pages + pages - request.kv_pages <= self.policy.capacity_pages
            and self.physical.can_adjust(
                self.policy.cache_component, delta_bytes
            )
        )

    def resize(self, request: _MutableRequest, pages: int) -> bool:
        if pages < 0 or not self.can_resize(request, pages):
            return False
        delta = pages - request.kv_pages
        if not self.physical.adjust(
            self.policy.cache_component,
            delta * self.policy.bytes_per_page,
        ):
            return False
        self.used_pages += delta
        self.persistent_used_pages += delta
        request.kv_pages = pages
        request.peak_kv_pages = max(request.peak_kv_pages, pages)
        self.peak_pages = max(self.peak_pages, self.used_pages)
        self.persistent_peak_pages = max(
            self.persistent_peak_pages, self.persistent_used_pages
        )
        if delta > 0:
            self.allocations += 1
        elif delta < 0:
            self.releases += 1
        return True

    def reserve_temporary(
        self, request: _MutableRequest, target_total_pages: int
    ) -> bool:
        """Reserve proposal pages without treating them as persistent KV."""

        if target_total_pages < 0:
            return False
        if request.temporary_kv_pages:
            raise RuntimeError("temporary KV reservation already active")
        temporary_pages = max(0, target_total_pages - request.kv_pages)
        delta_bytes = temporary_pages * self.policy.bytes_per_page
        if (
            self.used_pages + temporary_pages > self.policy.capacity_pages
            or not self.physical.can_adjust(
                self.policy.cache_component, delta_bytes
            )
        ):
            return False
        if not self.physical.adjust(
            self.policy.cache_component, delta_bytes
        ):
            return False
        request.temporary_kv_pages = temporary_pages
        self.used_pages += temporary_pages
        self.temporary_used_pages += temporary_pages
        self.peak_pages = max(self.peak_pages, self.used_pages)
        self.temporary_peak_pages = max(
            self.temporary_peak_pages, self.temporary_used_pages
        )
        if temporary_pages > 0:
            self.temporary_allocations += 1
        return True

    def release_temporary(self, request: _MutableRequest) -> None:
        temporary_pages = request.temporary_kv_pages
        if temporary_pages <= 0:
            return
        byte_count = temporary_pages * self.policy.bytes_per_page
        self.physical.adjust(self.policy.cache_component, -byte_count)
        self.used_pages -= temporary_pages
        self.temporary_used_pages -= temporary_pages
        request.temporary_kv_pages = 0
        self.temporary_releases += 1

    def offload(self, request: _MutableRequest) -> bool:
        pages = max(0, int(request.kv_pages))
        byte_count = pages * self.policy.bytes_per_page
        if pages <= 0:
            return True
        if self.offload_used_bytes + byte_count > self.policy.offload_capacity_bytes:
            return False
        if not self.physical.transfer(
            self.policy.cache_component,
            self.policy.offload_component,
            byte_count,
        ):
            return False
        self.used_pages -= pages
        self.persistent_used_pages -= pages
        if self.used_pages < 0 or self.persistent_used_pages < 0:
            raise RuntimeError("KV ledger underflow while offloading")
        self.releases += 1
        request.kv_pages = 0
        self.offload_used_bytes += byte_count
        self.offload_peak_bytes = max(self.offload_peak_bytes, self.offload_used_bytes)
        request.swapped_pages = pages
        request.swap_bytes = byte_count
        return True

    def can_restore(
        self, request: _MutableRequest, target_pages: Optional[int] = None
    ) -> bool:
        swapped_pages = max(0, int(request.swapped_pages))
        byte_count = max(0, int(request.swap_bytes))
        if swapped_pages <= 0:
            return byte_count <= 0
        if byte_count != swapped_pages * self.policy.bytes_per_page:
            return False
        target = swapped_pages if target_pages is None else int(target_pages)
        if target < swapped_pages or target > self.policy.capacity_pages:
            return False
        extra_bytes = (target - swapped_pages) * self.policy.bytes_per_page
        if self.used_pages + target > self.policy.capacity_pages:
            return False
        if self.offload_used_bytes < byte_count:
            return False
        return self.physical.can_transfer(
            self.policy.offload_component,
            self.policy.cache_component,
            byte_count,
            extra_bytes,
        )

    def restore(
        self, request: _MutableRequest, target_pages: Optional[int] = None
    ) -> bool:
        if not self.can_restore(request, target_pages):
            return False
        swapped_pages = max(0, int(request.swapped_pages))
        byte_count = max(0, int(request.swap_bytes))
        target = swapped_pages if target_pages is None else int(target_pages)
        extra_pages = target - swapped_pages
        if not self.physical.transfer(
            self.policy.offload_component,
            self.policy.cache_component,
            byte_count,
            extra_pages * self.policy.bytes_per_page,
        ):
            return False
        self.used_pages += target
        self.persistent_used_pages += target
        self.peak_pages = max(self.peak_pages, self.used_pages)
        self.persistent_peak_pages = max(
            self.persistent_peak_pages, self.persistent_used_pages
        )
        if extra_pages > 0:
            self.allocations += 1
        self.offload_used_bytes -= byte_count
        if self.offload_used_bytes < 0:
            raise RuntimeError("KV offload ledger underflow while restoring")
        request.kv_pages = target
        request.swapped_pages = 0
        request.swap_bytes = 0
        return True

    def discard_offload(self, request: _MutableRequest) -> None:
        self.physical.adjust(self.policy.offload_component, -request.swap_bytes)
        self.offload_used_bytes -= request.swap_bytes
        request.swapped_pages = 0
        request.swap_bytes = 0


class _LinearStateLedger:
    def __init__(
        self, policy: LinearStatePolicy, physical: _PhysicalCapacityLedger
    ) -> None:
        self.policy = policy
        self.physical = physical
        self.used_requests = 0
        self.peak_requests = 0
        self.allocations = 0
        self.releases = 0
        self.offload_events = 0
        self.offload_bytes = 0
        self.offload_used_bytes = 0
        self.offload_peak_bytes = 0
        self.restore_events = 0

    def allocate(self, request: _MutableRequest) -> bool:
        if self.policy.bytes_per_request <= 0 or request.linear_state_resident:
            return True
        if self.used_requests >= self.policy.capacity_requests:
            return False
        if not self.physical.adjust(
            self.policy.cache_component, self.policy.bytes_per_request
        ):
            return False
        self.used_requests += 1
        self.peak_requests = max(self.peak_requests, self.used_requests)
        self.allocations += 1
        request.linear_state_resident = True
        return True

    def release(self, request: _MutableRequest) -> None:
        if not request.linear_state_resident:
            return
        self.used_requests -= 1
        self.physical.adjust(
            self.policy.cache_component, -self.policy.bytes_per_request
        )
        self.releases += 1
        request.linear_state_resident = False

    def offload(self, request: _MutableRequest) -> bool:
        if not request.linear_state_resident:
            return True
        byte_count = self.policy.bytes_per_request
        if not self.policy.offload_component:
            return False
        if self.offload_used_bytes + byte_count > self.policy.offload_capacity_bytes:
            return False
        if not self.physical.transfer(
            self.policy.cache_component,
            self.policy.offload_component,
            byte_count,
        ):
            return False
        self.used_requests -= 1
        if self.used_requests < 0:
            raise RuntimeError("linear-state ledger underflow while offloading")
        self.releases += 1
        request.linear_state_resident = False
        request.linear_state_swapped_bytes = byte_count
        self.offload_used_bytes += byte_count
        self.offload_peak_bytes = max(
            self.offload_peak_bytes, self.offload_used_bytes
        )
        self.offload_events += 1
        self.offload_bytes += byte_count
        return True

    def restore(self, request: _MutableRequest) -> bool:
        if request.linear_state_swapped_bytes <= 0:
            return self.allocate(request)
        if self.used_requests >= self.policy.capacity_requests:
            return False
        byte_count = request.linear_state_swapped_bytes
        if self.offload_used_bytes < byte_count:
            return False
        if not self.physical.transfer(
            self.policy.offload_component,
            self.policy.cache_component,
            byte_count,
        ):
            return False
        self.used_requests += 1
        self.peak_requests = max(self.peak_requests, self.used_requests)
        self.offload_used_bytes -= byte_count
        request.linear_state_swapped_bytes = 0
        request.linear_state_resident = True
        self.restore_events += 1
        return True

    def discard_offload(self, request: _MutableRequest) -> None:
        self.physical.adjust(
            self.policy.offload_component,
            -request.linear_state_swapped_bytes,
        )
        self.offload_used_bytes -= request.linear_state_swapped_bytes
        request.linear_state_swapped_bytes = 0


@dataclass
class _PromptCacheEntry:
    """One retained prompt-cache object and its checkpoint segments."""

    key: str
    request_id: str
    created_ns: float
    last_access_ns: float
    allocation_bytes: int
    resident_bytes: int
    host_backed_bytes: int
    unbacked_bytes: int
    segment_sizes: Tuple[int, ...]
    resident_segments: int


class _PromptCacheLedger:
    """Track retained prompt-cache allocations independently from active KV.

    The active request ledgers describe pages/state needed to execute the
    current batch.  This ledger describes completed prompt-cache objects that
    may remain graph-referenced after a request finishes.  A cache entry has a
    logical full-slot allocation and a separate resident segment demand, so an
    untouched context is not silently treated as resident GPU memory.
    """

    def __init__(
        self,
        policy: _PromptCachePolicy,
        plan: ServingPlan,
        physical: _PhysicalCapacityLedger,
    ) -> None:
        self.policy = policy
        self.plan = plan
        self.physical = physical
        self.entries: Dict[str, _PromptCacheEntry] = {}
        self.logical_allocated_bytes = 0
        self.peak_logical_allocated_bytes = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.host_backed_bytes = 0
        self.peak_host_backed_bytes = 0
        self.unbacked_bytes = 0
        self.logical_entry_bytes = 0
        self.graph_resident_entry_bytes = 0
        self.graph_resident_segments = 0
        self.checkpoint_segments_allocated = 0
        self.checkpoint_segments_resident = 0
        self.allocation_events = 0
        self.eviction_events = 0
        self.eviction_entries = 0
        self.eviction_bytes = 0
        self.refault_events = 0
        self.refault_bytes = 0

    @property
    def enabled(self) -> bool:
        return bool(self.policy.enabled)

    @property
    def component(self) -> Optional[str]:
        return self.policy.component or self.plan.kv_policy.cache_component

    @property
    def offload_component(self) -> Optional[str]:
        return self.policy.offload_component or self.plan.kv_policy.offload_component

    def _slot_context_tokens(self) -> int:
        value = _model_max_sequence_length(self.plan.scenario)
        resource = _serving_resource_policy(self.plan.scenario)
        if resource.kv_slot_context_tokens is not None:
            value = resource.kv_slot_context_tokens
        return max(0, int(value or 0))

    def _derived_entry_sizes(self, state: _MutableRequest) -> Tuple[int, int]:
        """Return ``(logical_full_slot, graph_resident)`` bytes for a request."""

        page_tokens = max(1, int(self.plan.kv_policy.tokens_per_page))
        page_bytes = max(0, int(self.plan.kv_policy.bytes_per_page))
        slot_tokens = self._slot_context_tokens()
        slot_pages = (
            (slot_tokens + page_tokens - 1) // page_tokens
            if slot_tokens > 0
            else 0
        )
        kv_slot_bytes = slot_pages * page_bytes
        declared_kv_per_token, declared_kv_slot_bytes = (
            _declared_serving_kv_quantum(self.plan)
        )
        if declared_kv_per_token > 0:
            page_bytes = declared_kv_per_token * page_tokens
        if declared_kv_slot_bytes > 0:
            kv_slot_bytes = declared_kv_slot_bytes
        elif declared_kv_per_token > 0:
            kv_slot_bytes = slot_pages * page_bytes
        if self.policy.kv_slot_allocation_bytes is not None:
            kv_slot_bytes = max(0, int(self.policy.kv_slot_allocation_bytes))
        state_slot_bytes = max(
            0, int(self.plan.linear_state_policy.bytes_per_request)
        )
        if self.policy.state_slot_allocation_bytes is not None:
            state_slot_bytes = max(
                0, int(self.policy.state_slot_allocation_bytes)
            )
        derived_slot_bytes = kv_slot_bytes + state_slot_bytes
        if self.policy.slot_allocation_bytes is not None:
            logical_bytes = max(0, int(self.policy.slot_allocation_bytes))
        elif self.policy.entry_bytes is not None:
            logical_bytes = max(0, int(self.policy.entry_bytes))
        else:
            logical_bytes = derived_slot_bytes

        live_tokens = _max_live_kv_tokens(
            state.spec.prompt_tokens, state.spec.output_tokens
        )
        actual_pages = (
            (live_tokens + page_tokens - 1) // page_tokens
            if live_tokens > 0 and page_bytes > 0
            else 0
        )
        actual_bytes = (
            actual_pages * page_bytes
            + max(0, int(self.plan.linear_state_policy.bytes_per_request))
        )
        mode = self.policy.graph_residency_mode.strip().lower()
        if self.policy.resident_entry_bytes is not None:
            resident_bytes = max(0, int(self.policy.resident_entry_bytes))
        elif self.policy.entry_bytes is not None and self.policy.slot_allocation_bytes is None:
            # An explicitly observed entry size is a direct graph-residency
            # fact unless a separate full-slot commitment was supplied.
            resident_bytes = max(0, int(self.policy.entry_bytes))
        elif mode in {"slot_allocation", "full_slot", "allocation", "logical"}:
            resident_bytes = logical_bytes
        else:
            resident_bytes = actual_bytes
        return logical_bytes, max(0, resident_bytes)

    @staticmethod
    def _split_segments(total_bytes: int, segment_count: int) -> Tuple[int, ...]:
        total = max(0, int(total_bytes))
        count = max(1, int(segment_count))
        base, remainder = divmod(total, count)
        return tuple(base + (1 if index < remainder else 0) for index in range(count))

    @staticmethod
    def _resident_segment_count(
        segments: Tuple[int, ...], resident_bytes: int
    ) -> int:
        remaining = max(0, int(resident_bytes))
        count = 0
        for segment_bytes in segments:
            if remaining <= 0:
                break
            count += 1
            remaining -= segment_bytes
        return count

    def _component_limit(self) -> Optional[int]:
        component = self.component
        if not component:
            return None
        return self.physical.limits.get(str(component))

    def _logical_capacity(self) -> int:
        limit = self._component_limit()
        return max(0, int(limit)) if limit is not None else 0

    def _remove_entry(self, key: str) -> None:
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        if entry.resident_bytes:
            self.physical.adjust(self.component, -entry.resident_bytes)
            self.resident_bytes -= entry.resident_bytes
            self.graph_resident_entry_bytes -= entry.resident_bytes
        if entry.host_backed_bytes:
            self.physical.adjust(self.offload_component, -entry.host_backed_bytes)
            self.host_backed_bytes -= entry.host_backed_bytes
        self.logical_allocated_bytes -= entry.allocation_bytes
        self.logical_entry_bytes -= entry.allocation_bytes
        self.unbacked_bytes -= entry.unbacked_bytes
        self.graph_resident_segments -= entry.resident_segments
        self.checkpoint_segments_allocated -= len(entry.segment_sizes)
        self.checkpoint_segments_resident -= entry.resident_segments

    def _evict_entry(self, entry: _PromptCacheEntry) -> None:
        resident = max(0, int(entry.resident_bytes))
        if resident <= 0:
            return
        offload = self.offload_component
        component = self.component
        moved = False
        # An offload target is a real host-backed allocation.  Do not pretend
        # that a component shared with the device is an independent spill.
        if offload and component and str(offload) != str(component):
            moved = self.physical.transfer(component, offload, resident)
        if moved:
            entry.host_backed_bytes += resident
            self.host_backed_bytes += resident
            self.peak_host_backed_bytes = max(
                self.peak_host_backed_bytes, self.host_backed_bytes
            )
        else:
            # Without a declared host backing path an evicted object is
            # removed, matching the runtime's "remove oldest entry" event.
            self._remove_entry(entry.key)
        if moved:
            entry.resident_bytes = 0
            entry.resident_segments = 0
            self.resident_bytes -= resident
            self.graph_resident_entry_bytes -= resident
            self.graph_resident_segments = max(
                0, self.graph_resident_segments - self._resident_segment_count(
                    entry.segment_sizes, resident
                )
            )
            self.checkpoint_segments_resident = max(
                0,
                self.checkpoint_segments_resident
                - self._resident_segment_count(entry.segment_sizes, resident),
            )
        self.eviction_events += 1
        self.eviction_entries += 1
        self.eviction_bytes += resident

    def _make_room(self, additional_bytes: int, protected_key: Optional[str] = None) -> None:
        need = max(0, int(additional_bytes))
        limit = self._component_limit()
        if limit is None:
            return
        used = max(0, int(self.physical.used_bytes.get(str(self.component), 0)))
        need = max(0, used + need - int(limit))
        while need > 0:
            candidates = [
                entry
                for entry in self.entries.values()
                if entry.resident_bytes > 0 and entry.key != protected_key
            ]
            if not candidates:
                return
            victim = min(
                candidates,
                key=lambda entry: (entry.last_access_ns, entry.created_ns, entry.key),
            )
            resident_before = victim.resident_bytes
            self._evict_entry(victim)
            if victim.key not in self.entries:
                released = resident_before
            else:
                released = resident_before - victim.resident_bytes
            if released <= 0:
                return
            used = max(0, int(self.physical.used_bytes.get(str(self.component), 0)))
            need = max(0, used + max(0, int(additional_bytes)) - int(limit))

    def ensure_capacity(self, additional_bytes: int) -> bool:
        """Evict old prompt entries so an active allocation can proceed."""

        if not self.enabled or additional_bytes <= 0:
            return True
        self._make_room(additional_bytes)
        component = self.component
        return self.physical.can_adjust(component, int(additional_bytes))

    def add_completed(
        self, state: _MutableRequest, timestamp_ns: float
    ) -> Optional[_PromptCacheEntry]:
        if not self.enabled or not self.policy.retain_completed:
            return None
        key = str(state.spec.request_id)
        existing = self.entries.get(key)
        if existing is not None:
            existing.last_access_ns = float(timestamp_ns)
            return existing
        logical_bytes, resident_demand = self._derived_entry_sizes(state)
        if logical_bytes <= 0 and resident_demand <= 0:
            return None
        segment_sizes = self._split_segments(
            logical_bytes, self.policy.context_checkpoints
        )
        resident_demand = min(max(0, resident_demand), max(0, logical_bytes))
        entry = _PromptCacheEntry(
            key=key,
            request_id=key,
            created_ns=float(timestamp_ns),
            last_access_ns=float(timestamp_ns),
            allocation_bytes=logical_bytes,
            resident_bytes=0,
            host_backed_bytes=0,
            unbacked_bytes=0,
            segment_sizes=segment_sizes,
            resident_segments=0,
        )
        self.entries[key] = entry
        self.logical_allocated_bytes += logical_bytes
        self.logical_entry_bytes += logical_bytes
        self.peak_logical_allocated_bytes = max(
            self.peak_logical_allocated_bytes, self.logical_allocated_bytes
        )
        self.checkpoint_segments_allocated += len(segment_sizes)
        self.allocation_events += 1

        self._make_room(resident_demand, protected_key=key)
        granted = 0
        if resident_demand > 0 and self.physical.adjust(
            self.component, resident_demand
        ):
            granted = resident_demand
        remaining = resident_demand - granted
        if remaining > 0:
            offload = self.offload_component
            component = self.component
            if offload and (not component or str(offload) != str(component)):
                if self.physical.adjust(offload, remaining):
                    entry.host_backed_bytes = remaining
                    self.host_backed_bytes += remaining
                    self.peak_host_backed_bytes = max(
                        self.peak_host_backed_bytes, self.host_backed_bytes
                    )
                    remaining = 0
        entry.resident_bytes = granted
        entry.unbacked_bytes = remaining
        entry.resident_segments = self._resident_segment_count(
            segment_sizes, granted
        )
        self.resident_bytes += granted
        self.graph_resident_entry_bytes += granted
        self.graph_resident_segments += entry.resident_segments
        self.unbacked_bytes += remaining
        self.checkpoint_segments_resident += entry.resident_segments
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return entry

    def touch(self, key: str, timestamp_ns: float, required_bytes: Optional[int] = None) -> bool:
        """Refault a host-backed/overcommitted entry when its graph is reused."""

        entry = self.entries.get(str(key))
        if entry is None:
            return False
        entry.last_access_ns = float(timestamp_ns)
        if required_bytes is None:
            required = entry.resident_bytes + entry.host_backed_bytes + entry.unbacked_bytes
        else:
            required = max(0, int(required_bytes))
        additional = max(0, required - entry.resident_bytes)
        if additional <= 0:
            return True
        self._make_room(additional, protected_key=entry.key)
        component = self.component
        offload = self.offload_component
        restored = 0
        if entry.host_backed_bytes > 0 and offload and (
            not component or str(offload) != str(component)
        ):
            restored = min(additional, entry.host_backed_bytes)
            if not self.physical.transfer(offload, component, restored):
                restored = 0
            else:
                entry.host_backed_bytes -= restored
                self.host_backed_bytes -= restored
        remaining = additional - restored
        if remaining > 0 and self.physical.adjust(component, remaining):
            restored += remaining
            entry.unbacked_bytes = max(0, entry.unbacked_bytes - remaining)
            self.unbacked_bytes = max(0, self.unbacked_bytes - remaining)
        if restored <= 0:
            return False
        prior_resident_segments = entry.resident_segments
        entry.resident_bytes += restored
        entry.resident_segments = self._resident_segment_count(
            entry.segment_sizes, entry.resident_bytes
        )
        self.resident_bytes += restored
        self.graph_resident_entry_bytes += restored
        self.graph_resident_segments += max(
            0, entry.resident_segments - prior_resident_segments
        )
        self.checkpoint_segments_resident += max(
            0, entry.resident_segments - prior_resident_segments
        )
        self.refault_events += 1
        self.refault_bytes += restored
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return restored >= additional

    def snapshot(self) -> Mapping[str, object]:
        capacity = self._logical_capacity()
        logical_overcommit = (
            max(0, self.logical_allocated_bytes - capacity)
            if capacity > 0
            else 0
        )
        resident_overcommit = (
            max(0, self.resident_bytes - capacity) if capacity > 0 else 0
        )
        return {
            "prompt_cache_enabled": self.enabled,
            "prompt_cache_context_checkpoints": int(self.policy.context_checkpoints),
            "prompt_cache_component": self.component,
            "prompt_cache_offload_component": self.offload_component,
            "prompt_cache_entry_count": len(self.entries),
            "prompt_cache_allocation_events": self.allocation_events,
            "prompt_cache_logical_allocated_bytes": self.logical_allocated_bytes,
            "prompt_cache_peak_logical_allocated_bytes": self.peak_logical_allocated_bytes,
            "prompt_cache_logical_capacity_bytes": capacity,
            "prompt_cache_logical_overcommit_bytes": logical_overcommit,
            "prompt_cache_resident_bytes": self.resident_bytes,
            "prompt_cache_peak_resident_bytes": self.peak_resident_bytes,
            "prompt_cache_resident_capacity_bytes": capacity,
            "prompt_cache_resident_overcommit_bytes": resident_overcommit,
            "prompt_cache_host_backed_bytes": self.host_backed_bytes,
            "prompt_cache_peak_host_backed_bytes": self.peak_host_backed_bytes,
            "prompt_cache_unbacked_bytes": self.unbacked_bytes,
            "prompt_cache_logical_entry_bytes": self.logical_entry_bytes,
            "prompt_cache_graph_resident_entry_bytes": self.graph_resident_entry_bytes,
            "prompt_cache_graph_resident_segments": self.graph_resident_segments,
            "prompt_cache_checkpoint_segments_allocated": self.checkpoint_segments_allocated,
            "prompt_cache_checkpoint_segments_resident": self.checkpoint_segments_resident,
            "prompt_cache_eviction_events": self.eviction_events,
            "prompt_cache_eviction_entries": self.eviction_entries,
            "prompt_cache_eviction_bytes": self.eviction_bytes,
            "prompt_cache_refault_events": self.refault_events,
            "prompt_cache_refault_bytes": self.refault_bytes,
            "prompt_cache_observed_entry_sizes_bytes": self.policy.observed_entry_sizes_bytes,
            "prompt_cache_residency_mode": self.policy.graph_residency_mode,
            "prompt_cache_semantics": (
                "logical_full_slot_commitment_separate_from_graph_referenced_residency"
            ),
        }

    def metrics(self) -> PromptCacheMetrics:
        capacity = self._logical_capacity()
        return PromptCacheMetrics(
            enabled=self.enabled,
            context_checkpoints=int(self.policy.context_checkpoints),
            component=self.component,
            offload_component=self.offload_component,
            entry_count=len(self.entries),
            allocation_events=self.allocation_events,
            logical_allocated_bytes=self.logical_allocated_bytes,
            peak_logical_allocated_bytes=self.peak_logical_allocated_bytes,
            logical_capacity_bytes=capacity,
            logical_overcommit_bytes=(
                max(0, self.logical_allocated_bytes - capacity)
                if capacity > 0
                else 0
            ),
            resident_bytes=self.resident_bytes,
            peak_resident_bytes=self.peak_resident_bytes,
            resident_capacity_bytes=capacity,
            resident_overcommit_bytes=(
                max(0, self.resident_bytes - capacity) if capacity > 0 else 0
            ),
            host_backed_bytes=self.host_backed_bytes,
            peak_host_backed_bytes=self.peak_host_backed_bytes,
            unbacked_bytes=self.unbacked_bytes,
            logical_entry_bytes=self.logical_entry_bytes,
            graph_resident_entry_bytes=self.graph_resident_entry_bytes,
            graph_resident_segments=self.graph_resident_segments,
            eviction_events=self.eviction_events,
            eviction_entries=self.eviction_entries,
            eviction_bytes=self.eviction_bytes,
            refault_events=self.refault_events,
            refault_bytes=self.refault_bytes,
            checkpoint_segments_allocated=self.checkpoint_segments_allocated,
            checkpoint_segments_resident=self.checkpoint_segments_resident,
            observed_entry_sizes_bytes=self.policy.observed_entry_sizes_bytes,
            residency_mode=self.policy.graph_residency_mode,
        )


_RUNTIME_RESIDENCY_TENSOR_IDS = frozenset(
    {
        "kv_cache",
        "kv_cache_offload",
        "linear_state",
        "linear_state_offload",
        "prompt_cache",
        "prompt_cache_offload",
    }
)
_CLEAN_EVICTION_FREE_DISCARD = "free_discard"
_CLEAN_EVICTION_BACKING_MIGRATION = "backing_migration"
_CLEAN_EVICTION_SERVICES = frozenset(
    {
        _CLEAN_EVICTION_FREE_DISCARD,
        _CLEAN_EVICTION_BACKING_MIGRATION,
    }
)
_OWNER_RESIDENCY_TRANSFER_SETUP_ENVELOPE_RESOURCE_ID = (
    "owner_residency.transfer_setup_serial_envelope"
)


@dataclass(frozen=True)
class _PhysicalCapacityClaim:
    """One fixed, pinned owner of bytes in a physical component."""

    claim_id: str
    component_id: str
    byte_count: int


def _mapping_or_empty(value: object) -> Mapping[object, object]:
    return value if isinstance(value, _ABCMapping) else {}


def _runtime_weight_copy_ids(
    decision: Mapping[object, object],
) -> FrozenSet[str]:
    """Return aliases explicitly materialized as distinct runtime copies."""

    aliases = _mapping_or_empty(decision.get("logical_weight_aliases"))
    details = _mapping_or_empty(decision.get("weight_tensor_details"))
    copies = set()
    for raw_tensor_id, raw_detail in details.items():
        detail = _mapping_or_empty(raw_detail)
        if detail.get("runtime_copy_of") is None:
            continue
        tensor_id = str(raw_tensor_id)
        owner_id = str(detail.get("runtime_copy_of") or "").strip()
        file_owner_id = str(
            detail.get("file_owner_tensor_id") or ""
        ).strip()
        if (
            not owner_id
            or str(aliases.get(tensor_id) or "") != owner_id
            or file_owner_id != owner_id
        ):
            raise ValueError(
                "runtime weight copy {} must match its logical/file owner alias"
                .format(tensor_id)
            )
        copies.add(tensor_id)
    return frozenset(copies)


def _physical_capacity_claims(
    scenario: ScenarioConfig,
) -> Tuple[_PhysicalCapacityClaim, ...]:
    """Parse the authoritative fixed-capacity owners, failing closed.

    ``placement.metadata.capacity_claims`` is the sole declaration point for
    pre-existing physical owners such as driver reservations.  Every entry is
    therefore required to identify a non-negative, non-evictable, pinned
    physical owner.  A zero-byte row is an explicit no-op; silently dropping a
    malformed row would add its bytes back to the runtime budget.
    """

    metadata = scenario.placement.metadata
    if not isinstance(metadata, _ABCMapping):
        raise ValueError("placement.metadata must be a mapping")
    if "capacity_claims" not in metadata:
        return ()
    raw_claims = metadata["capacity_claims"]
    if not isinstance(raw_claims, _ABCMapping):
        raise ValueError("placement.metadata.capacity_claims must be a mapping")

    claims: List[_PhysicalCapacityClaim] = []
    seen_ids = set()
    for raw_claim_id, raw_claim in sorted(
        raw_claims.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(raw_claim_id, str) or not raw_claim_id.strip():
            raise ValueError("capacity_claims claim id must be non-empty text")
        claim_id = raw_claim_id.strip()
        if claim_id != raw_claim_id:
            raise ValueError(
                "capacity_claims claim id must not contain surrounding whitespace"
            )
        if claim_id in seen_ids:
            raise ValueError("duplicate capacity_claims claim id: {}".format(claim_id))
        seen_ids.add(claim_id)
        key = "capacity_claims[{}]".format(claim_id)
        if not isinstance(raw_claim, _ABCMapping):
            raise ValueError("{} must be a mapping".format(key))

        raw_component_id = raw_claim.get("component_id")
        if (
            not isinstance(raw_component_id, str)
            or not raw_component_id.strip()
            or raw_component_id != raw_component_id.strip()
        ):
            raise ValueError("{}.component_id must be non-empty text".format(key))
        component_id = raw_component_id
        try:
            scenario.hardware.get_component(component_id)
        except KeyError:
            raise ValueError(
                "{}.component_id names unknown component: {}".format(
                    key, component_id
                )
            )

        raw_bytes = raw_claim.get("bytes")
        if (
            isinstance(raw_bytes, bool)
            or not isinstance(raw_bytes, int)
            or raw_bytes < 0
        ):
            raise ValueError("{}.bytes must be a non-negative integer".format(key))
        if raw_claim.get("capacity_accounting_role") != "physical_owner":
            raise ValueError(
                "{}.capacity_accounting_role must be physical_owner".format(key)
            )
        if raw_claim.get("non_evictable") is not True:
            raise ValueError("{}.non_evictable must be true".format(key))
        if raw_claim.get("pinned") is not True:
            raise ValueError("{}.pinned must be true".format(key))
        claims.append(
            _PhysicalCapacityClaim(
                claim_id=claim_id,
                component_id=component_id,
                byte_count=raw_bytes,
            )
        )
    return tuple(claims)


def _residency_capacity_contract(plan: ServingPlan) -> Mapping[object, object]:
    """Return the explicit unified-pool contract, or an empty mapping.

    The owner-aware runtime is intentionally opt-in.  A physical component
    capacity alone is insufficient to infer that eager slots, recurrent
    state, and clean model weights share one pageable allocation domain.
    Local/runtime adapters declare that fact through ``capacity_ledger``.
    """

    metadata = _mapping_or_empty(plan.scenario.placement.metadata)
    contract = _mapping_or_empty(metadata.get("capacity_ledger"))
    unified = _mapping_or_empty(contract.get("unified_pool"))
    if not contract or not unified:
        return {}
    try:
        capacity = int(contract.get("physical_capacity_bytes", 0))
    except (TypeError, ValueError, OverflowError):
        return {}
    component_id = str(contract.get("component_id", "")).strip()
    if capacity <= 0 or not component_id:
        return {}
    if plan.kv_policy.cache_component and component_id != str(
        plan.kv_policy.cache_component
    ):
        return {}
    return contract


def _runtime_allocation_contract(plan: ServingPlan) -> Mapping[object, object]:
    """Return adapter-declared live allocation geometry, if available.

    Capacity partitions and allocation lifetimes are different facts.  This
    nested contract lets an adapter describe a shared KV pool, recurrent
    rollback planes, initial WDDM residency, and OS backing without teaching
    the generic runtime a model name or a concurrency threshold.
    """

    capacity = _residency_capacity_contract(plan)
    return _mapping_or_empty(capacity.get("runtime_allocation_contract"))


def _clean_eviction_service(
    row: Mapping[object, object],
    *,
    label: str,
    default: str = _CLEAN_EVICTION_FREE_DISCARD,
) -> str:
    """Return a fail-closed clean-eviction service mode."""

    raw_service = row.get("clean_eviction_service", default)
    service = str(raw_service).strip().lower()
    if service not in _CLEAN_EVICTION_SERVICES:
        raise ValueError(
            "{} clean_eviction_service must be one of {}".format(
                label,
                ", ".join(sorted(_CLEAN_EVICTION_SERVICES)),
            )
        )
    return service


def _runtime_clean_eviction_services(
    runtime_contract: Mapping[object, object],
) -> Tuple[str, Mapping[str, str]]:
    default = str(
        runtime_contract.get(
            "clean_eviction_service_default",
            _CLEAN_EVICTION_FREE_DISCARD,
        )
    ).strip().lower()
    if default not in _CLEAN_EVICTION_SERVICES:
        raise ValueError(
            "runtime allocation clean_eviction_service_default must be one of "
            "{}".format(", ".join(sorted(_CLEAN_EVICTION_SERVICES)))
        )
    services: Dict[str, str] = {}
    for key in ("weight_backend", "kv", "linear_state", "mtp_draft_kv"):
        row = _mapping_or_empty(runtime_contract.get(key))
        if not row:
            continue
        allocation_id = str(row.get("allocation_id", "")).strip()
        if not allocation_id:
            if "clean_eviction_service" in row:
                raise ValueError(
                    "runtime allocation {} declares clean_eviction_service "
                    "but has no allocation_id".format(key)
                )
            continue
        services[allocation_id] = _clean_eviction_service(
            row,
            label="runtime allocation {}".format(allocation_id),
            default=default,
        )
    return default, services


def _contract_nonnegative_int(
    values: Mapping[object, object],
    *keys: str,
    default: int = 0,
) -> int:
    for key in keys:
        if key not in values:
            continue
        raw = values.get(key)
        if isinstance(raw, bool):
            continue
        try:
            parsed = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed == raw and parsed >= 0:
            return parsed
    return max(0, int(default))


def _scenario_runtime_allocation_contract(
    scenario: ScenarioConfig,
) -> Mapping[object, object]:
    metadata = _mapping_or_empty(scenario.placement.metadata)
    capacity_ledger = _mapping_or_empty(metadata.get("capacity_ledger"))
    contract = _mapping_or_empty(
        capacity_ledger.get("runtime_allocation_contract")
    )
    if contract:
        return contract
    return _mapping_or_empty(metadata.get("runtime_allocation_contract"))


def _linear_state_runtime_contract(
    scenario: ScenarioConfig,
) -> Mapping[object, object]:
    contract = _scenario_runtime_allocation_contract(scenario)
    state_contract = _mapping_or_empty(contract.get("linear_state"))
    if state_contract:
        return state_contract
    metadata = _mapping_or_empty(scenario.placement.metadata)
    return _mapping_or_empty(metadata.get("linear_state_contract"))


def _linear_state_contract_live_bytes(
    contract: Mapping[object, object],
) -> int:
    live_bytes = _contract_nonnegative_int(
        contract,
        "device_live_bytes_per_slot",
        "live_bytes_per_slot",
        "live_bytes_per_request",
        "bytes_per_request",
    )
    if live_bytes > 0:
        return live_bytes
    base_bytes = _contract_nonnegative_int(
        contract,
        "device_base_bytes_per_seq",
        "base_bytes_per_seq",
        "current_state_bytes_per_sequence",
        "bytes_per_plane",
    )
    if base_bytes <= 0:
        return 0
    live_planes = _contract_nonnegative_int(
        contract,
        "live_plane_count",
        default=1,
    )
    return base_bytes * max(1, live_planes)


def _linear_state_contract_capacity(
    contract: Mapping[object, object],
) -> int:
    resident = _contract_nonnegative_int(
        contract,
        "resident_budget_bytes",
        "initial_resident_bytes",
        "capacity_bytes",
    )
    if resident > 0:
        return resident
    return _contract_nonnegative_int(
        contract,
        "committed_bytes",
        "requested_bytes",
        "unified_total_requested_bytes",
    )


def _canonical_residency_owner(
    alias_id: str,
    aliases: Mapping[object, object],
) -> str:
    current = str(alias_id)
    visited = set()
    while current in aliases and current not in visited:
        visited.add(current)
        target = aliases[current]
        if target is None:
            break
        current = str(target)
    return current


def _normalized_make_resident_scope(value: object) -> str:
    """Return the exact physical extent selected by a residency contract."""

    scope = str(value).strip().lower()
    if scope == "allocation":
        # Historical spelling meant the physical backend owner.
        scope = "backend_allocation"
    if scope not in {"range", "allocation_view", "backend_allocation"}:
        raise ValueError(
            "make_resident_scope must be range, allocation_view, or "
            "backend_allocation"
        )
    return scope


def _residency_physical_owner_id(
    manager: AllocationResidencyManager,
    target_id: str,
) -> str:
    if target_id in manager.views:
        return str(manager.views[target_id].owner_id)
    return str(target_id)


def _make_resident_for_scope(
    manager: AllocationResidencyManager,
    target_id: str,
    scope: str,
) -> None:
    """Apply only the pre-access residency extent selected by ``scope``."""

    if scope == "range":
        return
    if scope == "allocation_view":
        manager.touch(target_id)
        return
    manager.touch(_residency_physical_owner_id(manager, target_id))


def _make_resident_scope_physical_bytes(
    manager: AllocationResidencyManager,
    target_id: str,
    scope: str,
    range_physical_bytes: int,
) -> int:
    """Report the backing bytes covered by the selected residency extent."""

    if scope == "range":
        return int(range_physical_bytes)
    owner_id = _residency_physical_owner_id(manager, target_id)
    owner = manager.allocations[owner_id]
    if scope == "backend_allocation" or target_id not in manager.views:
        return int(owner.committed_bytes)
    view = manager.views[target_id]
    granule = max(1, int(owner.residency_granule_bytes))
    start = (int(view.offset_bytes) // granule) * granule
    end = min(
        int(owner.committed_bytes),
        (
            (int(view.offset_bytes) + int(view.size_bytes) + granule - 1)
            // granule
        )
        * granule,
    )
    return max(0, end - start)


def _settle_measurement_start_residency(
    manager: AllocationResidencyManager,
    runtime_contract: Mapping[object, object],
) -> None:
    """Apply an adapter-declared untimed warmup residency snapshot.

    Backend-buffer construction and zero-fill happen before a server benchmark
    warmup.  The constructor endpoint is therefore not necessarily the
    residency state from which timed requests start.  This contract reshapes
    only physical owner ranges; it does not change commitment or allocate a
    second copy of any live pool.
    """

    raw_snapshot = runtime_contract.get("measurement_start_residency")
    if raw_snapshot is None:
        return
    if not isinstance(raw_snapshot, _ABCMapping):
        raise ValueError(
            "measurement_start_residency must be a mapping"
        )
    enabled = raw_snapshot.get("enabled", False)
    if enabled is False or enabled is None:
        return
    if enabled is not True:
        raise ValueError(
            "measurement_start_residency.enabled must be boolean"
        )

    raw_targets = raw_snapshot.get("owner_resident_bytes")
    if not isinstance(raw_targets, _ABCMapping) or not raw_targets:
        raise ValueError(
            "measurement-start residency requires owner_resident_bytes"
        )
    targets: Dict[str, int] = {}
    for raw_owner_id, raw_target in raw_targets.items():
        owner_id = str(raw_owner_id).strip()
        if not owner_id:
            raise ValueError(
                "measurement-start residency owner id must not be empty"
            )
        if owner_id in targets:
            raise ValueError(
                "duplicate measurement-start residency owner: {}".format(
                    owner_id
                )
            )
        if owner_id not in manager.allocations:
            raise ValueError(
                "measurement-start residency references unknown owner: {}"
                .format(owner_id)
            )
        if isinstance(raw_target, bool) or not isinstance(raw_target, int):
            raise ValueError(
                "measurement-start resident bytes must be an integer"
            )
        target = raw_target
        if target < 0:
            raise ValueError(
                "measurement-start resident bytes must be an integer >= 0"
            )
        committed = manager.allocations[owner_id].committed_bytes
        if target > committed:
            raise ValueError(
                "measurement-start resident bytes exceed committed bytes for {}"
                .format(owner_id)
            )
        granule = manager.allocations[owner_id].residency_granule_bytes
        if target != committed and target % granule != 0:
            raise ValueError(
                "measurement-start resident bytes must end on a backing-granule "
                "boundary for {} (granule {} bytes)".format(
                    owner_id, granule
                )
            )
        targets[owner_id] = target

    raw_priority = raw_snapshot.get("resident_priority")
    if (
        isinstance(raw_priority, (str, bytes, _ABCMapping))
        or not isinstance(raw_priority, _ABCSequence)
    ):
        raise ValueError(
            "measurement-start resident_priority must be an ordered sequence"
        )
    priority: List[str] = []
    for raw_owner_id in raw_priority:
        owner_id = str(raw_owner_id).strip()
        if owner_id not in targets:
            raise ValueError(
                "measurement-start resident_priority references an owner not "
                "declared in owner_resident_bytes: {}".format(owner_id)
            )
        if owner_id in priority:
            raise ValueError(
                "measurement-start resident_priority contains duplicate owner: {}"
                .format(owner_id)
            )
        priority.append(owner_id)
    missing_priority = set(targets).difference(priority)
    if missing_priority:
        raise ValueError(
            "measurement-start resident_priority omits declared owners: {}"
            .format(", ".join(sorted(missing_priority)))
        )

    target_total = sum(targets.values())
    if target_total > manager.pool.capacity_bytes:
        raise ValueError(
            "measurement-start residency targets exceed physical capacity"
        )
    non_target_resident = sum(
        owner.resident_bytes
        for owner_id, owner in manager.allocations.items()
        if owner_id not in targets
    )
    if target_total + non_target_resident > manager.pool.capacity_bytes:
        raise ValueError(
            "measurement-start residency targets plus undeclared resident "
            "owners exceed physical capacity"
        )

    # Full constructor clears leave runtime owners resident and can displace
    # weights.  First trim every declared owner to its steady-state budget so
    # later restores have enough capacity without relying on incidental LRU
    # order.  Explicit eviction selects high addresses, preserving a target
    # prefix for range-based request accesses.
    for owner_id in reversed(priority):
        owner = manager.allocations[owner_id]
        excess = owner.resident_bytes - targets[owner_id]
        if excess > 0:
            manager.evict(owner_id, excess)

    # Restore only owners that are actually short.  Snapshot targets were
    # validated above as complete backing-unit prefixes (or full allocations),
    # so this cannot manufacture a half-granule steady state.
    for owner_id in priority:
        target = targets[owner_id]
        owner = manager.allocations[owner_id]
        if owner.resident_bytes >= target:
            continue
        manager.read(owner_id, offset_bytes=0, size_bytes=target)
        excess = owner.resident_bytes - target
        if excess > 0:
            manager.evict(owner_id, excess)

    for owner_id, target in targets.items():
        actual = manager.allocations[owner_id].resident_bytes
        if actual != target:
            raise ValueError(
                "measurement-start residency did not converge for {}: "
                "expected {}, got {}".format(owner_id, target, actual)
            )
    manager.assert_consistent()


def _build_allocation_residency_manager(
    plan: ServingPlan,
    *,
    apply_measurement_start_residency: bool = True,
) -> Optional[AllocationResidencyManager]:
    """Materialize one authoritative owner-aware device residency pool.

    Static placement tensors are unique physical owners.  Logical weight
    aliases become zero-capacity views unless control-plane detail explicitly
    marks a distinct runtime copy.  Eager serving slots are registered as full
    allocations according to the declared allocation policy.  Clean weights/KV
    pages with explicit host backing may be discarded/refaulted; recurrent
    state without a declared offload path remains non-evictable.
    """

    contract = _residency_capacity_contract(plan)
    if not contract:
        return None
    component_id = str(contract["component_id"])
    capacity_bytes = int(contract["physical_capacity_bytes"])
    declared_kv_per_token, _ = _declared_serving_kv_quantum(plan)
    resource_policy = _serving_resource_policy(plan.scenario)
    page_size_bytes = max(
        1,
        int(
            resource_policy.residency_granule_bytes
            or (
                declared_kv_per_token * plan.kv_policy.tokens_per_page
                if declared_kv_per_token > 0
                else (plan.kv_policy.bytes_per_page or 1)
            )
        ),
    )
    manager = AllocationResidencyManager(
        component_id,
        capacity_bytes,
        page_size_bytes=page_size_bytes,
    )

    placement = plan.scenario.placement
    metadata = _mapping_or_empty(placement.metadata)
    claimed_ids = set()
    for claim in _physical_capacity_claims(plan.scenario):
        if claim.component_id != component_id:
            continue
        claimed_ids.add(claim.claim_id)
        if claim.byte_count:
            manager.claim_system(claim.claim_id, claim.byte_count)

    runtime_contract = _runtime_allocation_contract(plan)
    decision = control_plane_decision(plan.scenario)
    aliases = {
        str(key): value
        for key, value in _mapping_or_empty(
            decision.get("logical_weight_aliases")
        ).items()
    }
    raw_logical_views = decision.get("logical_weight_views", ())
    if isinstance(raw_logical_views, _ABCMapping):
        logical_views = set(str(tensor_id) for tensor_id in raw_logical_views)
    elif isinstance(raw_logical_views, _ABCSequence) and not isinstance(
        raw_logical_views, (str, bytes)
    ):
        logical_views = set(str(tensor_id) for tensor_id in raw_logical_views)
    else:
        logical_views = set()
    special = _mapping_or_empty(metadata.get("special_weight_placement"))
    default_weight_backing = plan.kv_policy.offload_component
    details = _mapping_or_empty(decision.get("weight_tensor_details"))
    runtime_weight_copies = _runtime_weight_copy_ids(decision)
    weight_contract = _mapping_or_empty(
        runtime_contract.get("weight_backend")
    )
    weight_member_ids = set()

    # llama.cpp allocates all tensors sharing one backend buffer type from a
    # single backend allocation.  Adapter-declared tensor members are thus
    # zero-capacity byte-range views of one physical owner, not independent
    # residency allocations.
    if weight_contract:
        weight_owner_id = str(
            weight_contract.get(
                "allocation_id", "model.weights.backend.{}".format(component_id)
            )
        ).strip()
        weight_committed = _contract_nonnegative_int(
            weight_contract,
            "committed_bytes",
            "size_bytes",
        )
        weight_initial = min(
            weight_committed,
            _contract_nonnegative_int(
                weight_contract,
                "initial_resident_bytes",
                default=weight_committed,
            ),
        )
        raw_weight_backing = weight_contract.get(
            "os_backing_component_id",
            weight_contract.get("backing_component_id", default_weight_backing),
        )
        weight_backing = (
            str(raw_weight_backing).strip() if raw_weight_backing else None
        )
        if weight_backing == component_id:
            weight_backing = None
        weight_granule = max(
            1,
            _contract_nonnegative_int(
                weight_contract,
                "residency_granule_bytes",
                default=page_size_bytes,
            ),
        )
        raw_members = weight_contract.get("members", ())
        if isinstance(raw_members, (str, bytes, _ABCMapping)) or not isinstance(
            raw_members, _ABCSequence
        ):
            raise ValueError("weight backend members must be an ordered sequence")
        parsed_members: List[Tuple[str, int, int]] = []
        covered_until = 0
        for index, raw_member in enumerate(raw_members):
            member = _mapping_or_empty(raw_member)
            tensor_id = str(member.get("tensor_id", "")).strip()
            offset = _contract_nonnegative_int(member, "offset_bytes")
            size = _contract_nonnegative_int(member, "size_bytes")
            if not tensor_id or size <= 0:
                raise ValueError(
                    "weight backend member {} requires tensor_id and positive size"
                    .format(index)
                )
            if tensor_id in weight_member_ids:
                raise ValueError(
                    "duplicate weight backend member: {}".format(tensor_id)
                )
            if offset != covered_until:
                raise ValueError(
                    "weight backend members must form one contiguous ordered view"
                )
            placed_size = placement.tensor_bytes.get(tensor_id)
            placed_component = placement.tensor_to_component.get(tensor_id)
            if (
                placed_size is None
                or int(placed_size) != size
                or str(placed_component) != component_id
            ):
                raise ValueError(
                    "weight backend member {} disagrees with placement"
                    .format(tensor_id)
                )
            parsed_members.append((tensor_id, offset, size))
            weight_member_ids.add(tensor_id)
            covered_until += size
        if weight_committed <= 0 or covered_until != weight_committed:
            raise ValueError(
                "weight backend members must exactly cover committed bytes"
            )
        manager.register(
            weight_owner_id,
            weight_committed,
            kind="model_weights",
            backing=weight_backing,
            committed_bytes=weight_committed,
            resident_bytes=weight_initial,
            evictable=bool(weight_backing),
            read_only=True,
            residency_granule_bytes=weight_granule,
        )
        for tensor_id, offset, size in parsed_members:
            manager.create_view(
                tensor_id,
                weight_owner_id,
                offset_bytes=offset,
                size_bytes=size,
            )

    for raw_tensor_id, raw_size in placement.tensor_bytes.items():
        tensor_id = str(raw_tensor_id)
        if tensor_id in claimed_ids or tensor_id in _RUNTIME_RESIDENCY_TENSOR_IDS:
            continue
        if placement.tensor_to_component.get(tensor_id) != component_id:
            continue
        non_owning_alias = (
            (
                tensor_id in aliases or tensor_id in logical_views
            )
            and tensor_id not in runtime_weight_copies
        )
        if non_owning_alias or tensor_id in weight_member_ids:
            continue
        tensor_special = _mapping_or_empty(special.get(tensor_id))
        if (
            tensor_id not in runtime_weight_copies
            and str(tensor_special.get("capacity_accounting_role", ""))
            in {"logical_view", "non_owning_alias"}
        ):
            continue
        try:
            size_bytes = int(raw_size)
        except (TypeError, ValueError, OverflowError):
            continue
        if size_bytes <= 0:
            continue
        is_weight = "weight" in tensor_id.casefold()
        detail = _mapping_or_empty(details.get(tensor_id))
        raw_backing = detail.get("backing_component_id")
        backing = str(raw_backing) if raw_backing else None
        if not backing and is_weight and default_weight_backing:
            backing = str(default_weight_backing)
        if backing == component_id:
            backing = None
        pinned = not is_weight
        manager.register(
            tensor_id,
            size_bytes,
            kind="model_weight" if is_weight else "fixed_runtime",
            backing=backing,
            committed_bytes=size_bytes,
            resident_bytes=size_bytes,
            evictable=bool(backing) and is_weight,
            pinned=pinned,
            read_only=is_weight,
        )

    # Views are created only after every owner/member exists.  Chained aliases
    # are flattened deterministically all the way through a physical member
    # view, retaining its exact owner offset and extent without charging
    # capacity again.
    for raw_alias in sorted(aliases, key=str):
        alias_id = str(raw_alias)
        # A runtime copy retains the file alias but owns distinct device bytes.
        if alias_id in runtime_weight_copies:
            continue
        target_id = _canonical_residency_owner(alias_id, aliases)
        if alias_id in manager.allocations or alias_id in manager.views:
            continue
        if target_id in manager.allocations:
            owner_id = target_id
            offset_bytes = 0
            size_bytes = manager.allocations[target_id].committed_bytes
        elif target_id in manager.views:
            target_view = manager.views[target_id]
            owner_id = target_view.owner_id
            offset_bytes = target_view.offset_bytes
            size_bytes = target_view.size_bytes
        else:
            continue
        manager.create_view(
            alias_id,
            owner_id,
            offset_bytes=offset_bytes,
            size_bytes=size_bytes,
        )

    # Every scheduler slot has a stable logical identity, but the adapter may
    # declare that several identities are views of one live backend pool.  This
    # is essential for runtimes where ``n_ctx`` is a global cell budget rather
    # than one full context allocation per sequence.
    _, declared_kv_slot_bytes = _declared_serving_kv_quantum(plan)
    kv_slot_bytes = max(0, int(declared_kv_slot_bytes))
    if kv_slot_bytes <= 0:
        slot_tokens = _serving_resource_policy(plan.scenario).kv_slot_context_tokens
        if slot_tokens is None:
            slot_tokens = _model_max_sequence_length(plan.scenario)
        kv_slot_bytes = (
            (int(slot_tokens) + plan.kv_policy.tokens_per_page - 1)
            // plan.kv_policy.tokens_per_page
        ) * plan.kv_policy.bytes_per_page
    state_slot_bytes = max(
        0, int(plan.linear_state_policy.bytes_per_request)
    )
    kv_contract = _mapping_or_empty(runtime_contract.get("kv"))
    mtp_draft_kv_contract = _mapping_or_empty(
        runtime_contract.get("mtp_draft_kv")
    )
    state_contract = _mapping_or_empty(runtime_contract.get("linear_state"))
    eager_resident = (
        str(plan.kv_policy.allocation_policy).strip().lower() == "eager"
    )
    slot_count = max(0, int(plan.scheduler.max_num_seqs))

    if kv_contract:
        kv_committed = _contract_nonnegative_int(
            kv_contract,
            "committed_bytes",
            "requested_bytes",
            default=kv_slot_bytes * slot_count,
        )
        kv_initial = min(
            kv_committed,
            _contract_nonnegative_int(
                kv_contract,
                "initial_resident_bytes",
                "resident_budget_bytes",
            ),
        )
        kv_owner_id = str(
            kv_contract.get("allocation_id", "runtime.kv.pool")
        ).strip() or "runtime.kv.pool"
        raw_kv_backing = kv_contract.get(
            "os_backing_component_id",
            kv_contract.get("backing_component_id", plan.kv_policy.offload_component),
        )
        kv_backing = str(raw_kv_backing).strip() if raw_kv_backing else None
        if kv_backing == component_id:
            kv_backing = None
        kv_granule = max(
            1,
            _contract_nonnegative_int(
                kv_contract,
                "residency_granule_bytes",
                default=page_size_bytes,
            ),
        )
        shared_pool = bool(
            kv_contract.get("shared_pool", False)
            or str(kv_contract.get("allocation_mode", "")).strip().lower()
            in {"shared", "shared_pool", "global_pool", "unified_pool"}
        )
        kv_full_initialization = bool(
            str(kv_contract.get("full_buffer_initialization", "")).strip()
        )
        kv_registration_resident = 0 if kv_full_initialization else kv_initial
        if kv_committed > 0 and shared_pool:
            manager.register(
                kv_owner_id,
                kv_committed,
                kind="kv_cache",
                backing=kv_backing,
                committed_bytes=kv_committed,
                resident_bytes=kv_registration_resident,
                evictable=bool(kv_backing),
                read_only=False,
                residency_granule_bytes=kv_granule,
            )
            # Static byte-range views give each scheduler identity a bounded
            # share of the global cell pool without charging capacity twice.
            # The last view absorbs any alignment remainder.
            if slot_count > 0:
                offset = 0
                base_view, remainder = divmod(kv_committed, slot_count)
                for slot_index in range(slot_count):
                    view_size = base_view + (
                        1 if slot_index < remainder else 0
                    )
                    if view_size <= 0:
                        continue
                    manager.create_view(
                        "runtime.kv.slot.{:04d}".format(slot_index),
                        kv_owner_id,
                        offset_bytes=offset,
                        size_bytes=view_size,
                    )
                    offset += view_size
        elif kv_committed > 0 and slot_count > 0:
            per_slot = _contract_nonnegative_int(
                kv_contract,
                "live_bytes_per_slot",
                "bytes_per_slot",
                default=max(1, kv_committed // slot_count),
            )
            initial_base, initial_remainder = divmod(
                kv_registration_resident, slot_count
            )
            for slot_index in range(slot_count):
                resident = min(
                    per_slot,
                    initial_base + (1 if slot_index < initial_remainder else 0),
                )
                manager.register(
                    "runtime.kv.slot.{:04d}".format(slot_index),
                    per_slot,
                    kind="kv_cache",
                    backing=kv_backing,
                    committed_bytes=per_slot,
                    resident_bytes=resident,
                    evictable=bool(kv_backing),
                    read_only=False,
                    residency_granule_bytes=kv_granule,
                )
    else:
        for slot_index in range(slot_count):
            if kv_slot_bytes <= 0:
                continue
            manager.register(
                "runtime.kv.slot.{:04d}".format(slot_index),
                kv_slot_bytes,
                kind="kv_cache",
                backing=plan.kv_policy.offload_component,
                committed_bytes=kv_slot_bytes,
                resident_bytes=kv_slot_bytes if eager_resident else 0,
                evictable=bool(plan.kv_policy.offload_component),
                read_only=False,
            )

    if state_contract and slot_count > 0:
        state_committed = _contract_nonnegative_int(
            state_contract,
            "committed_bytes",
            "requested_bytes",
            default=state_slot_bytes * slot_count,
        )
        live_state_per_slot = _contract_nonnegative_int(
            state_contract,
            "live_bytes_per_slot",
            "bytes_per_slot",
            default=(state_committed // slot_count if slot_count else 0),
        )
        state_initial = min(
            state_committed,
            _contract_nonnegative_int(
                state_contract,
                "initial_resident_bytes",
                "resident_budget_bytes",
            ),
        )
        raw_state_backing = state_contract.get(
            "os_backing_component_id",
            state_contract.get(
                "backing_component_id",
                plan.linear_state_policy.offload_component,
            ),
        )
        state_backing = (
            str(raw_state_backing).strip() if raw_state_backing else None
        )
        if state_backing == component_id:
            state_backing = None
        state_granule = max(
            1,
            _contract_nonnegative_int(
                state_contract,
                "residency_granule_bytes",
                default=page_size_bytes,
            ),
        )
        state_full_initialization = bool(
            str(state_contract.get("full_buffer_initialization", "")).strip()
        )
        state_registration_resident = (
            0 if state_full_initialization else state_initial
        )
        state_shared_pool = bool(
            state_contract.get("shared_pool", False)
            or str(state_contract.get("allocation_scope", "")).strip().lower()
            in {
                "global_pool",
                "shared_pool",
                "unified_pool",
                "global_recurrent_backend_pool",
            }
        )
        state_owner_id = str(
            state_contract.get(
                "allocation_id", "runtime.linear_state.pool"
            )
        ).strip() or "runtime.linear_state.pool"
        if state_shared_pool and state_committed > 0:
            if live_state_per_slot <= 0 or (
                live_state_per_slot * slot_count != state_committed
            ):
                raise ValueError(
                    "shared recurrent backend pool must equal live bytes per "
                    "slot times slot count"
                )
            manager.register(
                state_owner_id,
                state_committed,
                kind="linear_state",
                backing=state_backing,
                committed_bytes=state_committed,
                resident_bytes=state_registration_resident,
                evictable=bool(state_backing),
                read_only=False,
                residency_granule_bytes=state_granule,
            )
            for slot_index in range(slot_count):
                manager.create_view(
                    "runtime.linear_state.slot.{:04d}".format(slot_index),
                    state_owner_id,
                    offset_bytes=slot_index * live_state_per_slot,
                    size_bytes=live_state_per_slot,
                )
        else:
            initial_base, initial_remainder = divmod(
                state_registration_resident, slot_count
            )
            for slot_index in range(slot_count):
                if live_state_per_slot <= 0:
                    continue
                resident = min(
                    live_state_per_slot,
                    initial_base
                    + (1 if slot_index < initial_remainder else 0),
                )
                manager.register(
                    "runtime.linear_state.slot.{:04d}".format(slot_index),
                    live_state_per_slot,
                    kind="linear_state",
                    backing=state_backing,
                    committed_bytes=live_state_per_slot,
                    resident_bytes=resident,
                    evictable=bool(state_backing),
                    read_only=False,
                    residency_granule_bytes=state_granule,
                )
    else:
        for slot_index in range(slot_count):
            if state_slot_bytes <= 0:
                continue
            manager.register(
                "runtime.linear_state.slot.{:04d}".format(slot_index),
                state_slot_bytes,
                kind="linear_state",
                backing=plan.linear_state_policy.offload_component,
                committed_bytes=state_slot_bytes,
                resident_bytes=state_slot_bytes if eager_resident else 0,
                evictable=bool(plan.linear_state_policy.offload_component),
                read_only=False,
            )

    if mtp_draft_kv_contract:
        draft_committed = _contract_nonnegative_int(
            mtp_draft_kv_contract,
            "committed_bytes",
            "requested_bytes",
        )
        draft_initial = min(
            draft_committed,
            _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "initial_resident_bytes",
                "resident_budget_bytes",
            ),
        )
        draft_slot_count = _contract_nonnegative_int(
            mtp_draft_kv_contract,
            "slot_count",
            default=slot_count,
        )
        if draft_slot_count != slot_count:
            raise ValueError(
                "MTP draft KV slot count must match scheduler slot count"
            )
        draft_owner_id = str(
            mtp_draft_kv_contract.get(
                "allocation_id", "runtime.mtp_draft.kv.pool"
            )
        ).strip() or "runtime.mtp_draft.kv.pool"
        if mtp_draft_kv_contract.get("shares_target_kv_owner") is False:
            target_owner_id = str(
                kv_contract.get("allocation_id", "runtime.kv.pool")
            ).strip() or "runtime.kv.pool"
            if draft_owner_id == target_owner_id:
                raise ValueError(
                    "MTP draft KV must use an owner independent from target KV"
                )
        if mtp_draft_kv_contract.get("cell_geometry_exact") is True:
            draft_stream_count = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "n_stream",
                "stream_count",
            )
            draft_cells_per_stream = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "cells_per_stream",
            )
            draft_total_cells = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "total_cells",
            )
            draft_n_ctx_slot = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "n_ctx_slot",
            )
            draft_server_n_ctx = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "server_n_ctx_tokens",
            )
            draft_bytes_per_cell = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "bytes_per_cell",
                "bytes_per_token",
            )
            if min(
                draft_stream_count,
                draft_cells_per_stream,
                draft_total_cells,
                draft_n_ctx_slot,
                draft_server_n_ctx,
                draft_bytes_per_cell,
            ) <= 0:
                raise ValueError(
                    "exact MTP draft KV geometry requires positive stream, "
                    "cell, context, and byte dimensions"
                )
            if draft_server_n_ctx != draft_n_ctx_slot * draft_slot_count:
                raise ValueError(
                    "MTP draft KV global n_ctx must equal n_ctx_slot times slots"
                )
            if draft_total_cells != draft_cells_per_stream * draft_stream_count:
                raise ValueError(
                    "MTP draft KV total cells must equal cells per stream times streams"
                )
            if draft_total_cells != draft_server_n_ctx:
                raise ValueError(
                    "MTP draft KV cells must inherit target global n_ctx"
                )
            if (
                mtp_draft_kv_contract.get("unified_streams") is True
                and draft_stream_count != 1
            ):
                raise ValueError(
                    "unified MTP draft KV must contain exactly one stream"
                )
            if draft_committed != draft_total_cells * draft_bytes_per_cell:
                raise ValueError(
                    "MTP draft KV committed bytes must equal cells times bytes per cell"
                )
        raw_draft_backing = mtp_draft_kv_contract.get(
            "os_backing_component_id",
            mtp_draft_kv_contract.get(
                "backing_component_id", plan.kv_policy.offload_component
            ),
        )
        draft_backing = (
            str(raw_draft_backing).strip() if raw_draft_backing else None
        )
        if draft_backing == component_id:
            draft_backing = None
        draft_granule = max(
            1,
            _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "residency_granule_bytes",
                default=page_size_bytes,
            ),
        )
        draft_full_initialization = bool(
            str(
                mtp_draft_kv_contract.get(
                    "full_buffer_initialization", ""
                )
            ).strip()
        )
        draft_registration_resident = (
            0 if draft_full_initialization else draft_initial
        )
        draft_shared_pool = bool(
            mtp_draft_kv_contract.get("shared_pool", False)
            or str(
                mtp_draft_kv_contract.get("allocation_scope", "")
            ).strip().lower()
            in {
                "global_pool",
                "shared_pool",
                "unified_pool",
                "global_mtp_draft_live_pool",
            }
        )
        if draft_committed > 0 and draft_shared_pool:
            manager.register(
                draft_owner_id,
                draft_committed,
                kind="mtp_draft_kv_cache",
                backing=draft_backing,
                committed_bytes=draft_committed,
                resident_bytes=draft_registration_resident,
                evictable=bool(draft_backing),
                read_only=False,
                residency_granule_bytes=draft_granule,
            )
            if slot_count > 0:
                offset = 0
                base_view, remainder = divmod(draft_committed, slot_count)
                for slot_index in range(slot_count):
                    view_size = base_view + (
                        1 if slot_index < remainder else 0
                    )
                    if view_size <= 0:
                        continue
                    manager.create_view(
                        "runtime.mtp_draft.kv.slot.{:04d}".format(
                            slot_index
                        ),
                        draft_owner_id,
                        offset_bytes=offset,
                        size_bytes=view_size,
                    )
                    offset += view_size
        elif draft_committed > 0 and slot_count > 0:
            per_slot = _contract_nonnegative_int(
                mtp_draft_kv_contract,
                "live_bytes_per_slot",
                "bytes_per_slot",
                default=max(1, draft_committed // slot_count),
            )
            initial_base, initial_remainder = divmod(
                draft_registration_resident, slot_count
            )
            for slot_index in range(slot_count):
                resident = min(
                    per_slot,
                    initial_base
                    + (1 if slot_index < initial_remainder else 0),
                )
                manager.register(
                    "runtime.mtp_draft.kv.slot.{:04d}".format(slot_index),
                    per_slot,
                    kind="mtp_draft_kv_cache",
                    backing=draft_backing,
                    committed_bytes=per_slot,
                    resident_bytes=resident,
                    evictable=bool(draft_backing),
                    read_only=False,
                    residency_granule_bytes=draft_granule,
                )
    raw_initialization = runtime_contract.get("initialization_sequence", ())
    if raw_initialization is None:
        raw_initialization = ()
    if isinstance(raw_initialization, (str, bytes, _ABCMapping)) or not isinstance(
        raw_initialization, _ABCSequence
    ):
        raise ValueError(
            "runtime allocation initialization_sequence must be ordered"
        )
    initialization_rows = []
    for index, raw_row in enumerate(raw_initialization):
        row = _mapping_or_empty(raw_row)
        try:
            order = int(row.get("order", index))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "runtime allocation initialization order must be an integer"
            ) from exc
        initialization_rows.append((order, index, row))
    for _order, _index, row in sorted(initialization_rows):
        allocation_id = str(row.get("allocation_id", "")).strip()
        operation = str(row.get("operation", "")).strip().lower()
        scope = str(row.get("scope", "")).strip().lower()
        if not allocation_id:
            raise ValueError(
                "runtime allocation initialization requires allocation_id"
            )
        if allocation_id not in manager.allocations:
            raise ValueError(
                "runtime initialization references missing allocation: {}"
                .format(allocation_id)
            )
        if operation not in {"write", "write_zero", "clear_zero"}:
            raise ValueError(
                "unsupported runtime allocation initialization operation: {}"
                .format(operation)
            )
        if scope not in {"allocation", "full_allocation", "full_backend_buffer"}:
            raise ValueError(
                "runtime allocation initialization must cover a full allocation"
            )
        # A constructor-side buffer clear is a write allocation: it makes the
        # complete backend buffer resident, dirties it, and may evict older
        # clean weight granules.  These migrations predate request timing.
        manager.write(allocation_id)
    if apply_measurement_start_residency and not initialization_rows:
        _settle_measurement_start_residency(manager, runtime_contract)
    manager.assert_consistent()
    return manager


class _OnlineRuntime:
    def __init__(
        self,
        plan: ServingPlan,
        lowerer: BatchLowerer,
        *,
        execution_control: Optional[ExecutionControl] = None,
        residency_manager: Optional[AllocationResidencyManager] = None,
        apply_measurement_start_residency: bool = True,
        execution_kernel: Optional[UnifiedEventKernel] = None,
        runtime_origin_ns: float = 0.0,
    ) -> None:
        if not isinstance(runtime_origin_ns, (int, float)) or isinstance(
            runtime_origin_ns, bool
        ):
            raise TypeError("runtime_origin_ns must be a finite non-negative number")
        runtime_origin_ns = float(runtime_origin_ns)
        if not math.isfinite(runtime_origin_ns) or runtime_origin_ns < 0.0:
            raise ValueError("runtime_origin_ns must be a finite non-negative number")
        if execution_kernel is not None and not isinstance(
            execution_kernel, UnifiedEventKernel
        ):
            raise TypeError("execution_kernel must be a UnifiedEventKernel")
        if execution_kernel is not None and execution_kernel.has_active_tasks:
            raise ValueError("execution_kernel must be drained before serving append")
        self.plan = plan
        self.lowerer = lowerer
        self._execution_stage_metadata_cache = (
            _execution_stage_cache_for_lowerer(lowerer)
        )
        self.execution_control = execution_control or ExecutionControl()
        self.resource_policy = _serving_resource_policy(plan.scenario)
        self.runtime_allocation_contract = _runtime_allocation_contract(plan)
        self._weight_allocation_contract = _mapping_or_empty(
            self.runtime_allocation_contract.get("weight_backend")
        )
        self._kv_allocation_contract = _mapping_or_empty(
            self.runtime_allocation_contract.get("kv")
        )
        self._mtp_draft_kv_allocation_contract = _mapping_or_empty(
            self.runtime_allocation_contract.get("mtp_draft_kv")
        )
        self._state_allocation_contract = _mapping_or_empty(
            self.runtime_allocation_contract.get("linear_state")
        )
        (
            self._clean_eviction_service_default,
            self._clean_eviction_service_by_allocation,
        ) = _runtime_clean_eviction_services(self.runtime_allocation_contract)
        # Staged host weights always use free clean-discard semantics.  Keep
        # their per-allocation overrides until migration costing observes the
        # release, then remove them so long sessions do not accumulate ids.
        self._staged_weight_clean_discard_ids: set[str] = set()
        # Resolve aggregate profiles only when the opt-in physical model will
        # consume them.  The historical serving path accepts custom/minimal
        # scenarios that need no GPU/HBM profile validation.
        self._gpu_profile = None
        self._hbm_profile = None
        self._gpu_controller_resource_domains: Dict[
            str, Tuple[Optional[str], frozenset[str]]
        ] = {}
        self.states: Dict[str, _MutableRequest] = {
            item.request_id: _MutableRequest(item, queued_since_ns=item.arrival_ns)
            for item in plan.requests
        }
        self._state_values = tuple(self.states.values())
        self.pending = plan.requests
        self._pending_index = 0
        self._request_count = len(self._state_values)
        self._running_count = 0
        self._terminal_count = 0
        self._finished_count = 0
        self._rejected_count = 0
        physical_limits = dict(_physical_runtime_limits(plan))
        _add_prompt_cache_runtime_limits(
            plan, self.resource_policy.prompt_cache, physical_limits
        )
        self.physical_ledger = _PhysicalCapacityLedger(physical_limits)
        self.ledger = _KVLedger(plan.kv_policy, self.physical_ledger)
        self.state_ledger = _LinearStateLedger(
            plan.linear_state_policy, self.physical_ledger
        )
        self.prompt_cache = _PromptCacheLedger(
            self.resource_policy.prompt_cache or _PromptCachePolicy(),
            plan,
            self.physical_ledger,
        )
        self._last_kv_allocation_request_id: Optional[str] = None
        self._last_prompt_cache_allocation_request_id: Optional[str] = None
        self._prompt_cache_appended_tokens = 0
        self._prompt_cache_wave_requests: Set[str] = set()
        self._pending_prompt_cache_states: List[_MutableRequest] = []
        self._kv_scan_wave_requests: Set[str] = set()
        scan_requested = plan.scenario.workload.metadata.get(
            "llama_cpp_unified_kv_scan", False
        ) is True
        scan_scenario = plan.scenario
        scan_layers = tuple(layer for layer in _execution_layers(scan_scenario)
                            if not layer.is_linear_attention)
        q4_materialization_requested = bool(
            scan_scenario.workload.metadata.get("llama_cpp_q4_kv_mma_materialization") is True
            and scan_layers
            and all(
                (artifact := _kv_dtype_bits(scan_scenario, layer)[1]) is not None
                and artifact.name.casefold() == "q4_0"
                for layer in scan_layers
            )
        )
        self._kv_scan_enabled = bool(
            scan_layers and not plan.mtp.enabled
            and scan_scenario.fusion_policy.flash_attention
            and (q4_materialization_requested or (
                scan_requested
                and str(plan.kv_policy.dtype).lower() in {"fp16", "f16", "float16", "half"}
            ))
            and plan.kv_policy.offload_ratio == 0.0
            and _parallel_plan(scan_scenario).world_size == 1
            and self.prompt_cache.enabled
            and self.prompt_cache.policy.unified_kv
            and self.prompt_cache.policy.retain_completed
            and all(scan_scenario.placement.op_to_component.get(layer.layer_id + ".attention")
                    == scan_scenario.host_orchestration_profile.gpu_component_id
                    for layer in scan_layers)
        )
        self._q4_mma_materialization_enabled = (
            self._kv_scan_enabled and q4_materialization_requested
        )
        self._kv_scan_cache_cells = 0
        if self._kv_scan_enabled:
            observed = scan_scenario.workload.metadata.get("runtime_observed_config", {})
            self._kv_scan_cache_cells = _metadata_nonnegative_int(
                observed.get("context_length", 0), "runtime_observed_config.context_length"
            )
            if self._kv_scan_cache_cells < 256:
                raise ValueError("KV scan lower bound requires declared cache cells >= 256")
        manager_created_here = residency_manager is None
        if manager_created_here:
            self.residency_manager = _build_allocation_residency_manager(
                plan,
                apply_measurement_start_residency=(
                    apply_measurement_start_residency
                ),
            )
        else:
            # A shared session manager is deliberately treated as persistent
            # physical state.  Do not rebuild allocations, clear backend
            # buffers, or apply the declarative measurement-start snapshot on
            # every fresh logical runtime.
            self.residency_manager = residency_manager
        if self.residency_manager is None:
            self._residency_interval_baseline_bytes: Optional[int] = None
            self._residency_interval_peak_bytes: Optional[int] = None
        else:
            replay_baseline = max(
                0, int(self.residency_manager.snapshot().resident_bytes)
            )
            self._residency_interval_baseline_bytes = replay_baseline
            self._residency_interval_peak_bytes = replay_baseline
        self._request_residency_slots: Dict[str, int] = {}
        self._free_residency_slots: List[int] = list(
            range(max(0, int(plan.scheduler.max_num_seqs)))
        )
        self._residency_migration_cursor = (
            self.residency_manager.migration_count
            if self.residency_manager is not None
            else 0
        )
        self._residency_measurement_migration_start = (
            self._residency_migration_cursor
        )
        self._residency_initialization_migrations = (
            self.residency_manager.migrations_since(0)
            if manager_created_here and self.residency_manager is not None
            else ()
        )
        self.residency_initialization_page_in_bytes = sum(
            item.page_in_bytes
            for item in self._residency_initialization_migrations
        )
        self.residency_initialization_page_out_bytes = sum(
            item.page_out_bytes
            for item in self._residency_initialization_migrations
        )
        self.residency_initialization_clean_discard_bytes = sum(
            item.clean_discard_bytes
            for item in self._residency_initialization_migrations
        )
        self.residency_initialization_dirty_writeback_bytes = sum(
            item.dirty_writeback_bytes
            for item in self._residency_initialization_migrations
        )
        self.residency_page_in_bytes = 0
        self.residency_page_out_bytes = 0
        self.residency_dirty_writeback_bytes = 0
        self.residency_clean_discard_bytes = 0
        self.residency_clean_discard_time_ns = 0.0
        self.residency_clean_discard_energy_pj = 0.0
        self.residency_transfer_time_ns = 0.0
        self.residency_transfer_energy_pj = 0.0
        self.residency_accesses = 0
        self.residency_weight_accesses = 0
        self.residency_kv_accesses = 0
        self.residency_state_accesses = 0
        self.residency_temporary_allocations = 0
        self.residency_temporary_releases = 0
        self._residency_lease_sequence = 0
        self._residency_group_lease_ids: List[str] = []
        self.residency_fault_batches = 0
        self.residency_clean_eviction_batches = 0
        self.events: List[ServingEvent] = []
        self.batches: List[ServingBatch] = []
        self._runtime_origin_ns = runtime_origin_ns
        self.now = max(
            runtime_origin_ns,
            min(
                (state.spec.arrival_ns for state in self._state_values),
                default=0.0,
            ),
        )
        self._host_available_ns = self.now
        self._gpu_available_ns = self.now
        self._last_gpu_start_ns = self.now
        self._last_serial_device_end_ns = self.now
        self._stage_resource_available_ns: Dict[str, float] = {}
        resource_capacities = _execution_resource_capacities(
            self.plan.scenario
        )
        if execution_kernel is None:
            self._execution_resource_kernel = UnifiedEventKernel(
                resource_capacities=resource_capacities
            )
        else:
            execution_kernel.ensure_resource_capacities(resource_capacities)
            self._execution_resource_kernel = execution_kernel
        self._execution_task_sequence = 0
        self._request_stage_ready_ns: Dict[Tuple[str, str, int], float] = {}
        self._request_device_ready_ns: Dict[str, float] = {
            state.spec.request_id: max(
                state.spec.arrival_ns,
                runtime_origin_ns,
            )
            for state in self._state_values
        }
        # Every successfully executed compute item receives the next sequence
        # value.  The value is deliberately runtime-local (rather than based
        # on request ids or arrival counts) so equal same-arrival candidates
        # rotate in the order they were actually served.
        self._service_sequence = 0
        self._admission_sequence = 0
        self.rounds = 0
        self.idle_ns = self.now
        self.preemptions = 0
        self.priority_preemptions = 0
        self.memory_preemptions = 0
        self.swap_events = 0
        self.swap_bytes = 0
        self.swap_in_bytes = 0
        self.swap_transfer_time_ns = 0.0
        self.swap_transfer_energy_pj = 0.0
        self.logical_swap_out_bytes = 0
        self.logical_swap_in_bytes = 0
        self.linear_state_swap_in_bytes = 0
        self.linear_state_swap_transfer_time_ns = 0.0
        self.linear_state_swap_transfer_energy_pj = 0.0
        self.linear_state_swap_routed_bytes = 0
        self.recompute_events = 0
        self.recompute_tokens = 0
        self.logical_prefill_read_bytes = 0
        self.logical_prefill_write_bytes = 0
        self.logical_decode_read_bytes = 0
        self.logical_decode_write_bytes = 0
        self.physical_prefill_read_bytes = 0
        self.physical_prefill_write_bytes = 0
        self.physical_decode_read_bytes = 0
        self.physical_decode_write_bytes = 0
        self.mtp_materialized_tokens = 0
        self.mtp_temporary_tokens = 0
        self.logical_mtp_materialized_write_bytes = 0
        self.physical_mtp_materialized_write_bytes = 0
        self.logical_mtp_temporary_write_bytes = 0
        self.physical_mtp_temporary_write_bytes = 0
        self.logical_mtp_verification_read_bytes = 0
        self.physical_mtp_verification_read_bytes = 0
        self._batch_kind_counts: Dict[str, int] = {}
        self._batch_phase_counts: Dict[str, int] = {}
        self._max_batch_sequences = 0
        self._max_batch_tokens = 0
        # Eager KV allocation is a logical address reservation.  Keep the
        # pages that are currently backed outside the device separately from
        # the physical resident ledger so a full context slot is never
        # mistaken for a fully touched GPU working set.
        self._resource_kv_host_backed_pages = 0
        self._resource_state_host_backed_allocations = 0

    def _capture_residency_interval_peak(self) -> None:
        """Record current physical residency for this replay interval.

        ``AllocationResidencyManager.peak_resident_bytes`` is cumulative over
        a shared session.  Sampling the current physical total at mutation
        boundaries lets a runtime retain a peak that belongs to this replay,
        including short-lived workspace residency before its release, without
        attributing an earlier warmup/replay peak to the current one.
        """

        manager = self.residency_manager
        if manager is None:
            return
        current = max(0, int(manager.resident_bytes))
        previous = self._residency_interval_peak_bytes
        self._residency_interval_peak_bytes = (
            current if previous is None else max(previous, current)
        )

    @staticmethod
    def _residency_consumer_task_ids(
        metadata: Optional[Mapping[str, Any]],
    ) -> Tuple[str, ...]:
        if metadata is None:
            return ()
        raw_ids = metadata.get("consumer_task_ids", ())
        ids = (
            tuple(str(item) for item in raw_ids if item)
            if isinstance(raw_ids, _ABCSequence)
            and not isinstance(raw_ids, (str, bytes, _ABCMapping))
            else ()
        )
        terminal = metadata.get("terminal_task_id")
        if terminal:
            ids = (*ids, str(terminal))
        return tuple(dict.fromkeys(ids))

    def _lease_residency_result(
        self,
        result: object,
        consumer_task_ids: Sequence[str],
    ) -> None:
        """Protect the accessed backing range until its DAG consumer anchor."""

        manager = self.residency_manager
        if manager is None:
            return
        size_bytes = max(0, int(getattr(result, "size_bytes", 0)))
        if size_bytes <= 0:
            return
        owner_id = str(getattr(result, "owner_id"))
        owner = manager.allocations.get(owner_id)
        if owner is not None and owner.lifecycle is AllocationLifecycle.TEMPORARY:
            return
        offset_bytes = max(0, int(getattr(result, "offset_bytes", 0)))
        for consumer_task_id in tuple(dict.fromkeys(consumer_task_ids)):
            if not consumer_task_id:
                continue
            self._residency_lease_sequence += 1
            lease_id = "runtime.residency.lease{:08d}".format(
                self._residency_lease_sequence
            )
            manager.acquire_lease(
                lease_id,
                owner_id,
                str(consumer_task_id),
                offset_bytes=offset_bytes,
                size_bytes=size_bytes,
            )
            # Keep bytes non-evictable while the current invocation group is
            # still being assembled.  Releasing at the group boundary records
            # the consumer anchor, so a later PAGE_OUT can reclaim the range
            # only after this exact DAG consumer completes.
            self._residency_group_lease_ids.append(lease_id)

    def _release_residency_group_leases(self) -> None:
        manager = self.residency_manager
        if manager is None:
            self._residency_group_lease_ids.clear()
            return
        for lease_id in reversed(self._residency_group_lease_ids):
            if lease_id in manager.leases:
                manager.release_lease(lease_id)
        self._residency_group_lease_ids.clear()

    def _component_participates_in_device_fence(
        self, component_id: str
    ) -> bool:
        component = self.plan.scenario.hardware.component_map().get(
            component_id
        )
        if component is None:
            return True
        return component.normalized_kind not in {
            "cpu",
            "host_memory",
            "dram",
            "ddr",
            "ddr_memory",
            "cxl_memory",
        }

    def _resource_snapshot(
        self, items: Sequence[BatchItem]
    ) -> Mapping[str, object]:
        """Capture scheduler and unified-residency state at cohort creation."""

        active = [
            state
            for state in self._state_values
            if state.status == RequestStatus.RUNNING
        ]
        active_decode = [
            state
            for state in active
            if state.recompute_cursor >= state.recompute_target
            and state.prefill_cursor >= state.spec.prompt_tokens
            and state.committed < state.spec.output_tokens
        ]
        cache_component = self.plan.kv_policy.cache_component
        kv_offload_component = self.plan.kv_policy.offload_component
        state_offload_component = self.plan.linear_state_policy.offload_component
        # The physical ledger can contain both resident and offloaded bytes
        # when roles share one component.  Runtime pressure is about bytes
        # resident on the device, so derive it from the role ledgers instead
        # of treating the component aggregate as a single residency bucket.
        resident_kv_bytes = (
            max(0, self.ledger.used_pages)
            * max(0, self.plan.kv_policy.bytes_per_page)
        )
        resident_state_bytes = (
            max(0, self.state_ledger.used_requests)
            * max(0, self.plan.linear_state_policy.bytes_per_request)
        )
        active_working_set = resident_kv_bytes + resident_state_bytes
        cache_capacity = (
            self.physical_ledger.limits.get(
                str(cache_component), self.plan.kv_policy.capacity_bytes
            )
            if cache_component
            else self.plan.kv_policy.capacity_bytes
        )
        physical_cache_used = (
            self.physical_ledger.used_bytes.get(
                str(cache_component), active_working_set
            )
            if cache_component
            else active_working_set
        )
        kv_offload_used = max(0, int(self.ledger.offload_used_bytes))
        state_offload_used = max(
            0, int(self.state_ledger.offload_used_bytes)
        )
        host_spill_used = kv_offload_used + state_offload_used
        kv_offload_capacity = max(
            0, int(self.plan.kv_policy.offload_capacity_bytes)
        )
        state_offload_capacity = max(
            0, int(self.plan.linear_state_policy.offload_capacity_bytes)
        )
        host_spill_capacity = kv_offload_capacity + state_offload_capacity
        host_spill_component = kv_offload_component or state_offload_component
        physical_host_spill_capacity = (
            self.physical_ledger.limits.get(
                str(host_spill_component), host_spill_capacity
            )
            if host_spill_component
            else host_spill_capacity
        )
        prompt_snapshot = dict(self.prompt_cache.snapshot())
        prompt_component = self.prompt_cache.component
        prompt_resident_bytes = max(
            0, int(prompt_snapshot.get("prompt_cache_resident_bytes", 0))
        )
        prompt_logical_bytes = max(
            0,
            int(
                prompt_snapshot.get(
                    "prompt_cache_logical_allocated_bytes", 0
                )
            ),
        )
        prompt_capacity = (
            self.physical_ledger.limits.get(str(prompt_component), 0)
            if prompt_component
            else 0
        )
        prompt_effective_capacity = max(
            0,
            int(prompt_capacity)
            - int(self.resource_policy.vram_reserve_bytes),
        )
        slot_context_tokens = self.resource_policy.kv_slot_context_tokens
        if slot_context_tokens is None:
            slot_context_tokens = _model_max_sequence_length(
                self.plan.scenario
            )
        logical_capacity_tokens = max(
            0,
            int(self.plan.scheduler.max_num_seqs)
            * max(0, int(slot_context_tokens)),
        )
        slot_reservation_tokens = 0
        if self.plan.kv_policy.allocation_policy == "eager":
            slot_reservation_tokens = len(active) * max(
                0, int(slot_context_tokens)
            )
        slot_count = len(active) if self.plan.kv_policy.allocation_policy == "eager" else 0
        slot_pages = (
            (max(0, int(slot_context_tokens)) + max(1, int(self.plan.kv_policy.tokens_per_page)) - 1)
            // max(1, int(self.plan.kv_policy.tokens_per_page))
            if slot_context_tokens > 0
            else 0
        )
        kv_reservation_pages = (
            slot_count * slot_pages
            if slot_count > 0 and slot_pages > 0
            else 0
        )
        _, declared_kv_slot_bytes = _declared_serving_kv_quantum(self.plan)
        kv_reservation_bytes = (
            slot_count
            * (
                declared_kv_slot_bytes
                if declared_kv_slot_bytes > 0
                else slot_pages * max(0, self.plan.kv_policy.bytes_per_page)
            )
        )
        # Lazy mode reports actual committed pages.  Eager mode reserves one
        # full context window per active slot in the *logical* address space,
        # while physical pressure remains the touched/committed resident set.
        # A reservation therefore cannot turn an untouched 65K context into a
        # fabricated GPU allocation; it is checked for backing capacity during
        # admission and exposed separately in the snapshot.
        resident_kv_demand_bytes = resident_kv_bytes
        active_working_set = resident_kv_bytes + resident_state_bytes
        effective_capacity = max(
            0,
            int(cache_capacity) - int(self.resource_policy.vram_reserve_bytes),
        )
        # Prompt-cache segments share the declared component only when they
        # are mapped there.  Keep active-page spill accounting separate from
        # prompt-cache LRU eviction, but expose the combined resident pressure
        # so a c4-sized retained cache is visible as a real device demand.
        prompt_shared_resident = (
            prompt_resident_bytes
            if prompt_component
            and cache_component
            and str(prompt_component) == str(cache_component)
            else 0
        )
        active_effective_capacity = max(
            0, effective_capacity - prompt_shared_resident
        )
        resident_excess = max(
            0, active_working_set - active_effective_capacity
        )
        unified_resident_working_set = active_working_set + prompt_resident_bytes
        if prompt_component and cache_component and str(prompt_component) == str(cache_component):
            unified_resident_excess = max(
                0, unified_resident_working_set - effective_capacity
            )
        else:
            unified_resident_excess = max(
                0, active_working_set - effective_capacity
            ) + max(0, prompt_resident_bytes - prompt_effective_capacity)
        reservation_terms = _eager_slot_reservation(
            self.plan, len(active)
        )
        if (
            self.resource_policy.enabled
            and self.plan.kv_policy.allocation_policy == "eager"
        ):
            if self.plan.scenario.workload.metadata.get("explicit_shared_kv_pool") is True:
                kv_reservation_bytes = int(reservation_terms["kv_reservation_bytes"])
            logical_reservation_bytes = int(
                reservation_terms["logical_reservation_bytes"]
            )
            logical_host_backed_bytes = int(
                reservation_terms["host_backed_bytes"]
            )
            logical_kv_host_backed_bytes = int(
                reservation_terms["kv_host_backed_bytes"]
            )
            logical_state_host_backed_bytes = int(
                reservation_terms["state_host_backed_bytes"]
            )
        else:
            logical_reservation_bytes = resident_kv_bytes + resident_state_bytes
            logical_host_backed_bytes = 0
            logical_kv_host_backed_bytes = 0
            logical_state_host_backed_bytes = 0
        logical_reservation_shortfall = max(
            0, logical_reservation_bytes - effective_capacity
        )
        # Attribute a capacity shortfall to role-specific backing instead of
        # using the largest object as a fake page size.  KV migrates in cache
        # pages; recurrent state migrates in its declared request allocation
        # unit.  The order follows the only generic backing contract available
        # here: KV has a cache offload path before state does.
        kv_backing_available = max(0, kv_offload_capacity - kv_offload_used)
        state_backing_available = max(
            0, state_offload_capacity - state_offload_used
        )
        kv_page_bytes = max(1, self.plan.kv_policy.bytes_per_page)
        kv_available_pages = kv_backing_available // kv_page_bytes
        kv_needed_pages = (
            int(
                math.ceil(
                    min(resident_excess, resident_kv_demand_bytes)
                    / float(kv_page_bytes)
                )
            )
            if resident_excess > 0 and resident_kv_demand_bytes > 0
            else 0
        )
        kv_spill_pages = min(kv_needed_pages, kv_available_pages)
        kv_spill = kv_spill_pages * kv_page_bytes
        remaining_excess = max(0, resident_excess - kv_spill)
        state_allocation_bytes = max(
            1, self.plan.linear_state_policy.bytes_per_request
        )
        state_available_allocations = (
            state_backing_available // state_allocation_bytes
        )
        state_needed_allocations = (
            int(
                math.ceil(
                    min(remaining_excess, resident_state_bytes)
                    / float(state_allocation_bytes)
                )
            )
            if remaining_excess > 0 and resident_state_bytes > 0
            else 0
        )
        state_spill_allocations = min(
            state_needed_allocations, state_available_allocations
        )
        state_spill = state_spill_allocations * state_allocation_bytes
        actual_spill = kv_spill + state_spill
        backing_available = kv_backing_available + state_backing_available
        spill_pages = kv_spill_pages + state_spill_allocations
        owner_residency_snapshot: Dict[str, object] = {
            "owner_residency_enabled": False,
        }
        if self.residency_manager is not None:
            owner_snapshot = self.residency_manager.snapshot()
            owner_totals = owner_snapshot.migrations
            owner_residency_snapshot = {
                "owner_residency_enabled": True,
                "owner_residency_component_id": owner_snapshot.component_id,
                "owner_residency_capacity_bytes": owner_snapshot.capacity_bytes,
                "owner_residency_committed_bytes": owner_snapshot.committed_bytes,
                "owner_residency_resident_bytes": owner_snapshot.resident_bytes,
                "owner_residency_peak_resident_bytes": (
                    owner_snapshot.peak_resident_bytes
                ),
                "owner_residency_available_bytes": owner_snapshot.available_bytes,
                "owner_residency_allocation_count": owner_snapshot.allocation_count,
                "owner_residency_view_count": owner_snapshot.view_count,
                "owner_residency_page_in_bytes_total": (
                    owner_totals.page_in_bytes
                ),
                "owner_residency_page_out_bytes_total": (
                    owner_totals.page_out_bytes
                ),
                "owner_residency_clean_discard_bytes_total": (
                    owner_totals.clean_discard_bytes
                ),
                "owner_residency_dirty_writeback_bytes_total": (
                    owner_totals.dirty_writeback_bytes
                ),
            }
        return {
            **prompt_snapshot,
            **owner_residency_snapshot,
            "active_sequence_count": len(active),
            "active_decode_sequence_count": len(active_decode),
            "batch_sequence_count": len(items),
            "batch_token_count": sum(
                max(0, int(item.token_count)) for item in items
            ),
            "kv_resident_pages": self.ledger.used_pages,
            "kv_persistent_pages": self.ledger.persistent_used_pages,
            "kv_temporary_pages": self.ledger.temporary_used_pages,
            "kv_capacity_pages": self.plan.kv_policy.capacity_pages,
            "kv_device_resident_capacity_pages": (
                self.plan.kv_policy.capacity_pages
            ),
            "kv_device_resident_capacity_bytes": (
                self.plan.kv_policy.capacity_bytes
            ),
            "kv_resident_bytes": resident_kv_bytes,
            "kv_reservation_tokens": slot_reservation_tokens,
            "kv_reservation_pages": kv_reservation_pages,
            "kv_reservation_bytes": kv_reservation_bytes,
            "kv_resident_demand_bytes": resident_kv_demand_bytes,
            "linear_state_resident_bytes": resident_state_bytes,
            "active_working_set_bytes": active_working_set,
            "unified_resident_working_set_bytes": unified_resident_working_set,
            "unified_resident_capacity_shortfall_bytes": unified_resident_excess,
            "active_resident_capacity_bytes": active_effective_capacity,
            "prompt_cache_effective_resident_capacity_bytes": prompt_effective_capacity,
            "logical_reservation_bytes": logical_reservation_bytes,
            "unified_logical_reservation_bytes": (
                logical_reservation_bytes + prompt_logical_bytes
            ),
            "prompt_cache_logical_reservation_shortfall_bytes": max(
                0,
                prompt_logical_bytes
                - max(0, effective_capacity - logical_reservation_bytes),
            ),
            "unified_logical_reservation_shortfall_bytes": max(
                0,
                logical_reservation_bytes
                + prompt_logical_bytes
                - effective_capacity,
            ),
            "logical_reservation_shortfall_bytes": logical_reservation_shortfall,
            "logical_host_backed_bytes": logical_host_backed_bytes,
            "logical_kv_host_backed_bytes": logical_kv_host_backed_bytes,
            "logical_state_host_backed_bytes": logical_state_host_backed_bytes,
            "kv_resident_tokens": (
                self.ledger.used_pages * self.plan.kv_policy.tokens_per_page
            ),
            "kv_unified_logical_capacity_tokens": logical_capacity_tokens,
            "kv_slot_context_tokens": max(0, int(slot_context_tokens)),
            "kv_residency_mode": self.plan.kv_policy.allocation_policy,
            "physical_cache_component": cache_component,
            "physical_cache_used_bytes": max(0, int(active_working_set)),
            "physical_cache_component_used_bytes": max(
                0, int(physical_cache_used)
            ),
            "physical_cache_capacity_bytes": max(0, int(cache_capacity)),
            "effective_resident_capacity_bytes": effective_capacity,
            "resident_capacity_shortfall_bytes": resident_excess,
            "host_spill_component": host_spill_component,
            "host_spill_used_bytes": host_spill_used,
            "host_spill_capacity_bytes": host_spill_capacity,
            "physical_host_spill_capacity_bytes": max(
                0, int(physical_host_spill_capacity)
            ),
            "host_spill_available_bytes": backing_available,
            "actual_spill_bytes": actual_spill,
            "actual_kv_spill_bytes": kv_spill,
            "actual_linear_state_spill_bytes": state_spill,
            "unbacked_pressure_bytes": max(0, resident_excess - actual_spill),
            "spill_page_bytes": self.plan.kv_policy.bytes_per_page,
            "spill_page_count": spill_pages,
            "kv_spill_page_bytes": self.plan.kv_policy.bytes_per_page,
            "kv_spill_page_count": kv_spill_pages,
            "linear_state_spill_allocation_bytes": state_allocation_bytes,
            "linear_state_spill_allocation_count": state_spill_allocations,
            "kv_offload_used_bytes": kv_offload_used,
            "kv_offload_capacity_bytes": kv_offload_capacity,
            "linear_state_offload_used_bytes": state_offload_used,
            "linear_state_offload_capacity_bytes": state_offload_capacity,
            "resource_contention_model": "serving_physical_residency_v1",
        }

    def _cohort_hbm_bytes(self, cohort: BatchCohort) -> int:
        """Derive persisted KV traffic for lowerers without byte metadata."""

        bytes_per_token = (
            self.plan.kv_policy.bytes_per_page
            // self.plan.kv_policy.tokens_per_page
            if self.plan.kv_policy.tokens_per_page > 0
            else 0
        )
        if bytes_per_token <= 0:
            return 0
        total_tokens = 0
        for item in cohort.items:
            materialized = _batch_item_kv_materialized_tokens(item)
            persisted_read = (
                0
                if item.phase in {"prefill", "recompute"}
                else max(0, int(item.context_tokens))
                * max(0, int(item.token_count))
            )
            total_tokens += materialized + persisted_read
        return total_tokens * bytes_per_token

    def _ensure_resource_profiles(self) -> None:
        """Resolve the aggregate rates needed by the opt-in resource model."""

        if self._gpu_profile is not None:
            return
        parallel_plan = _parallel_plan(self.plan.scenario)
        self._gpu_profile, _ = _uniform_rank_profile(
            self.plan.scenario,
            tuple(rank.component_id for rank in parallel_plan.ranks),
            GPUProfile,
            "GPU",
        )
        hbm_component_ids = tuple(
            rank.memory_component_id
            for rank in parallel_plan.ranks
            if rank.memory_component_id is not None
        )
        if hbm_component_ids:
            self._hbm_profile, _ = _uniform_rank_profile(
                self.plan.scenario,
                hbm_component_ids,
                HBMProfile,
                "HBM",
            )

    def _page_transfer_cost(
        self,
        byte_count: int,
        page_count: int,
        source_component: Optional[str],
        target_component: Optional[str],
    ) -> Tuple[float, float, str]:
        """Return migration time/energy from topology or declared DMA rates."""

        transfer_ns, energy_pj, route_kind, _phases, _components, _latency = (
            self._page_transfer_cost_details(
                byte_count,
                page_count,
                source_component,
                target_component,
                include_controller_phases=False,
            )
        )
        return transfer_ns, energy_pj, route_kind

    def _owner_residency_controller_phases(
        self,
        byte_count: int,
        submission_batch_count: int,
        residency_page_count: int,
        source_component: str,
        target_component: str,
        route_component_ids: Sequence[str],
        submission_latency_ns: float,
    ) -> Tuple[
        Tuple[Mapping[str, object], ...],
        Tuple[Tuple[Mapping[str, object], ...], ...],
        Tuple[Mapping[str, object], ...],
        str,
    ]:
        """Lower one migration into a constant-size controller task chain.

        CPU page-table work uses one task.  IOMMU, DMA setup, GPU command
        submission, and GPU MMU translation are four dependent physical tasks;
        an optional private zero-service join preserves the controller-chain
        audit boundary without consuming a physical resource.  L2/VRAM
        controller occupancy is returned for pipelining into the existing
        topology bulk phase.  The task count remains independent of page,
        granule, transaction, and vocabulary counts.
        """

        scenario = self.plan.scenario
        orchestration = scenario.host_orchestration_profile
        runtime_profile = scenario.runtime_profile
        cpu_control = runtime_profile.cpu
        transport = runtime_profile.pcie_dma_iommu
        cpu_id = orchestration.cpu_component_id
        cpu_profile, _host_memory_profile = _cpu_profiles(scenario, cpu_id)
        if not isinstance(cpu_profile, CPUProfile):
            raise ValueError("owner residency CPU profile is invalid")
        cpu_pipeline = cpu_profile.pipeline
        cpu_pipeline_resource = _component_resource_id(
            cpu_pipeline.resource_id,
            reference_component_id=orchestration.cpu_component_id,
            target_component_id=cpu_id,
        )

        pages = max(1, int(residency_page_count))
        submission_batches = max(1, int(submission_batch_count))

        def _ceil_div(numerator: int, denominator: int) -> int:
            return (numerator + denominator - 1) // denominator

        def _demand(
            resource_id: str,
            service_ns: float,
            *,
            controller_service_ns: float,
            transaction_count: int,
            transaction_batches: int,
            bytes_moved: int,
            work_units: float,
            runtime_phase: str,
            envelope: bool = False,
        ) -> Mapping[str, object]:
            return {
                "resource_id": resource_id,
                "service_ns": float(service_ns),
                "bytes_moved": int(bytes_moved),
                "energy_pj": 0.0,
                "work_units": float(work_units),
                "runtime_phase": runtime_phase,
                "aggregation": "controller_transaction_batch",
                "controller_service_ns": float(controller_service_ns),
                "transaction_count": int(transaction_count),
                "transaction_batches": int(transaction_batches),
                "transfer_bytes": int(byte_count),
                "serial_envelope": envelope,
            }

        # One page-table control item per residency granule plus one item per
        # page batch.  Throughput comes only from the declared CPU pipeline;
        # there is no fitted instruction multiplier.
        control_batches = _ceil_div(pages, cpu_control.request_batch_size)
        issue_width = max(
            1,
            min(
                cpu_pipeline.decode_width,
                cpu_pipeline.issue_width,
                cpu_pipeline.retire_width,
            ),
        )
        pipeline_parallelism = max(1, cpu_pipeline.core_count * issue_width)
        pipeline_work_items = pages + control_batches
        pipeline_waves = _ceil_div(
            pipeline_work_items,
            pipeline_parallelism,
        )
        pipeline_service_ns = (
            pipeline_waves / max(1.0e-12, cpu_pipeline.frequency_ghz)
        )
        # Core multiplicity belongs to the event-kernel capacity of the shared
        # scheduler resource.  This per-lane service may still batch requests
        # up to the authored queue/outstanding limits, but must not divide by
        # core_count a second time.
        scheduler_parallelism = max(
            1,
            min(
                orchestration.max_inflight_batches,
                cpu_control.max_outstanding_requests,
            ),
        )
        scheduler_waves = _ceil_div(control_batches, scheduler_parallelism)
        scheduler_service_ns = scheduler_waves * (
            orchestration.batch_fixed_ns
            + orchestration.kv_page_lookup_ns
            + orchestration.kv_descriptor_ns
        ) + max(0.0, float(submission_latency_ns))
        descriptor_bytes = pages * orchestration.kv_descriptor_bytes
        cache_lines = _ceil_div(
            descriptor_bytes,
            cpu_control.cache_line_bytes,
        )
        cache_batches = _ceil_div(
            cache_lines,
            cpu_control.request_batch_size,
        )
        cache_parallelism = max(
            1,
            min(
                cpu_control.memory_channels,
                cpu_control.request_queue_depth,
                cpu_control.max_outstanding_requests,
            ),
        )
        cache_waves = _ceil_div(cache_batches, cache_parallelism)
        cache_service_ns = cache_waves * cpu_control.cache_hit_latency_ns
        cpu_envelope_ns = (
            pipeline_service_ns + scheduler_service_ns + cache_service_ns
        )
        cpu_phase = (
            _demand(
                cpu_pipeline_resource,
                cpu_envelope_ns,
                controller_service_ns=pipeline_service_ns,
                transaction_count=pipeline_work_items,
                transaction_batches=pipeline_waves,
                bytes_moved=descriptor_bytes,
                work_units=float(pipeline_work_items),
                runtime_phase="owner_residency_cpu_make_resident",
                envelope=True,
            ),
            _demand(
                orchestration.scheduler_resource_id,
                scheduler_service_ns,
                controller_service_ns=scheduler_service_ns,
                transaction_count=pages,
                transaction_batches=scheduler_waves,
                bytes_moved=descriptor_bytes,
                work_units=float(pages),
                runtime_phase="owner_residency_cpu_make_resident",
            ),
            _demand(
                "{}.control_cache".format(cpu_id),
                cache_service_ns,
                controller_service_ns=cache_service_ns,
                transaction_count=cache_lines,
                transaction_batches=cache_waves,
                bytes_moved=descriptor_bytes,
                work_units=float(cache_lines),
                runtime_phase="owner_residency_cpu_make_resident",
            ),
        )

        iommu_pages = _ceil_div(byte_count, transport.iommu_page_size_bytes)
        iommu_misses = _ceil_div(iommu_pages, transport.iommu_tlb_entries)
        iommu_walk_waves = _ceil_div(
            iommu_misses,
            transport.iommu_max_outstanding_walks,
        )
        iommu_service_ns = (
            iommu_walk_waves * transport.iommu_miss_latency_ns
        )
        dma_setup = _dma_setup_service(
            byte_count,
            batch_bytes=transport.dma_batch_bytes,
            queue_depth=transport.dma_queue_depth,
            max_outstanding=transport.dma_max_outstanding,
            fixed_latency_ns=orchestration.dma_latency_ns,
            submission_ns_per_wave=(
                orchestration.dma_queue_submission_ns
            ),
        )

        def _component_kind(component_id: str) -> Optional[str]:
            try:
                return _kind(scenario.hardware.get_component(component_id))
            except (KeyError, ValueError):
                return None

        route_gpu_ids = tuple(
            component_id
            for component_id in route_component_ids
            if component_id in runtime_profile.gpu_controllers
            and _component_kind(component_id) == "gpu"
        )
        if route_gpu_ids:
            gpu_id = route_gpu_ids[-1]
        else:
            gpu_id = orchestration.gpu_component_id
            if (
                gpu_id not in runtime_profile.gpu_controllers
                or _component_kind(gpu_id) != "gpu"
            ):
                raise ValueError(
                    "owner residency route has no declared GPU controller"
                )
        gpu_control = runtime_profile.gpu_controllers[gpu_id]
        source_kind = _component_kind(source_component)
        target_kind = _component_kind(target_component)
        gpu_memory_kinds = {"gpu", "hbm"}
        if source_kind in gpu_memory_kinds and target_kind not in gpu_memory_kinds:
            direction = "page_out"
        elif target_kind in gpu_memory_kinds and source_kind not in gpu_memory_kinds:
            direction = "page_in"
        else:
            gpu_position = (
                tuple(route_component_ids).index(gpu_id)
                if gpu_id in route_component_ids
                else len(route_component_ids)
            )
            direction = (
                "page_out"
                if gpu_position * 2 < len(route_component_ids)
                else "page_in"
            )

        command = gpu_control.command_processor
        command_batches = _ceil_div(
            submission_batches,
            command.launch_batch_size,
        )
        command_parallelism = max(
            1,
            command.command_processor_count
            * min(
                command.hardware_queue_count,
                command.queue_depth,
                command.max_outstanding_kernels,
            ),
        )
        command_waves = _ceil_div(command_batches, command_parallelism)
        command_service_ns = (
            command_waves * command.command_submission_latency_ns
        )

        mmu = gpu_control.mmu_tlb
        gpu_pages = _ceil_div(byte_count, mmu.page_size_bytes)
        translation_batches = _ceil_div(
            gpu_pages,
            mmu.translation_batch_size,
        )
        translation_waves = _ceil_div(
            translation_batches,
            mmu.max_outstanding_page_walks,
        )
        mmu_service_ns = translation_waves * mmu.page_walk_latency_ns

        setup_envelope_resource_id = (
            _OWNER_RESIDENCY_TRANSFER_SETUP_ENVELOPE_RESOURCE_ID
        )
        setup_envelope_member_resource_ids = (
            "{}.iommu".format(cpu_id),
            orchestration.dma_resource_id,
            "{}.command_processor".format(gpu_id),
            "{}.mmu_tlb".format(gpu_id),
        )
        setup_phases = (
            (
                _demand(
                    "{}.iommu".format(cpu_id),
                    iommu_service_ns,
                    controller_service_ns=iommu_service_ns,
                    transaction_count=iommu_pages,
                    transaction_batches=iommu_walk_waves,
                    bytes_moved=byte_count,
                    work_units=float(iommu_pages),
                    runtime_phase="owner_residency_iommu_translation",
                ),
            ),
            (
                {
                    **_demand(
                        orchestration.dma_resource_id,
                        dma_setup.service_ns,
                        controller_service_ns=dma_setup.service_ns,
                        transaction_count=dma_setup.transaction_count,
                        transaction_batches=dma_setup.wave_count,
                        bytes_moved=byte_count,
                        work_units=float(dma_setup.transaction_count),
                        runtime_phase="owner_residency_dma_setup",
                    ),
                    "queue_parallelism": dma_setup.queue_parallelism,
                    "queue_depth": transport.dma_queue_depth,
                    "max_outstanding": transport.dma_max_outstanding,
                    "dma_engine_count_accounting": (
                        "event_kernel_resource_capacity"
                    ),
                },
            ),
            (
                _demand(
                    "{}.command_processor".format(gpu_id),
                    command_service_ns,
                    controller_service_ns=command_service_ns,
                    transaction_count=submission_batches,
                    transaction_batches=command_batches,
                    bytes_moved=byte_count,
                    work_units=float(command_batches),
                    runtime_phase="owner_residency_gpu_command_submission",
                ),
            ),
            (
                _demand(
                    "{}.mmu_tlb".format(gpu_id),
                    mmu_service_ns,
                    controller_service_ns=mmu_service_ns,
                    transaction_count=gpu_pages,
                    transaction_batches=translation_waves,
                    bytes_moved=byte_count,
                    work_units=float(gpu_pages),
                    runtime_phase="owner_residency_gpu_mmu_translation",
                ),
            ),
            (
                {
                    **_demand(
                        setup_envelope_resource_id,
                        0.0,
                        controller_service_ns=0.0,
                        transaction_count=0,
                        transaction_batches=0,
                        bytes_moved=0,
                        work_units=0.0,
                        runtime_phase="owner_residency_controller_chain_join",
                        envelope=True,
                    ),
                    "aggregation": "controller_chain_join",
                    "logical_resource": True,
                    "logical_join": True,
                    "transfer_bytes": 0,
                    "covered_transfer_bytes": int(byte_count),
                    "envelope_member_resource_ids": (
                        setup_envelope_member_resource_ids
                    ),
                    "envelope_semantics": (
                        "zero_service_join_after_serial_physical_controllers"
                    ),
                },
            ),
        )

        l2 = gpu_control.l2_cache
        cache_requests = _ceil_div(byte_count, l2.line_size_bytes)
        l2_batches = _ceil_div(cache_requests, l2.request_batch_size)
        l2_waves = _ceil_div(l2_batches, l2.max_outstanding_misses)
        l2_service_ns = l2_waves * l2.hit_latency_ns

        vram = gpu_control.vram_controller
        vram_requests = _ceil_div(byte_count, l2.line_size_bytes)
        vram_batches = _ceil_div(vram_requests, vram.request_batch_size)
        vram_parallelism = max(
            1,
            min(
                vram.controller_count
                * vram.channel_count
                * vram.lanes_per_channel,
                vram.max_outstanding_requests,
            ),
        )
        vram_waves = _ceil_div(vram_batches, vram_parallelism)
        # The topology endpoint already charges bulk VRAM bandwidth.  The V4
        # controller demand therefore contributes only queue/access waves and
        # pipelines in the same phase instead of serially charging the bytes a
        # second time.
        vram_service_ns = vram_waves * vram.access_latency_ns
        gpu_data_demands = (
            _demand(
                "{}.l2_controller".format(gpu_id),
                l2_service_ns,
                controller_service_ns=l2_service_ns,
                transaction_count=cache_requests,
                transaction_batches=l2_waves,
                bytes_moved=byte_count,
                work_units=float(cache_requests),
                runtime_phase="owner_residency_gpu_memory_pipeline",
            ),
            _demand(
                "{}.vram_controller".format(gpu_id),
                vram_service_ns,
                controller_service_ns=vram_service_ns,
                transaction_count=vram_requests,
                transaction_batches=vram_waves,
                bytes_moved=byte_count,
                work_units=float(vram_requests),
                runtime_phase="owner_residency_gpu_memory_pipeline",
            ),
        )
        return cpu_phase, setup_phases, gpu_data_demands, direction

    def _topology_transfer_cost_details(
        self,
        byte_count: int,
        source_component: Optional[str],
        target_component: Optional[str],
        *,
        name: str,
    ) -> Tuple[
        float,
        float,
        str,
        Tuple[Tuple[Mapping[str, object], ...], ...],
        Tuple[str, ...],
    ]:
        """Return bulk transfer work without residency or translation work."""

        if byte_count <= 0 or not source_component or not target_component:
            return 0.0, 0.0, "no_declared_backing_path", (), ()
        source = str(source_component)
        target = str(target_component)
        if source == target:
            return 0.0, 0.0, "same_component", (), (source,)
        route_ns = 0.0
        route_energy = 0.0
        route_kind = "topology"
        topology_phases: List[Tuple[Mapping[str, object], ...]] = []
        route_component_ids: Tuple[str, ...] = (source, target)
        try:
            router = _topology_router(self.plan.scenario)
            hops = router.route(
                source,
                target,
                byte_count,
                policy=self.plan.scenario.placement.parallel.routing_policy,
            )
            route_component_ids = tuple(
                dict.fromkeys((source, *(hop.target_component for hop in hops)))
            )
            for phase in router.transfer_phases(
                source,
                target,
                byte_count,
                policy=self.plan.scenario.placement.parallel.routing_policy,
                name=name,
            ):
                demands = tuple(
                    {
                        "resource_id": demand.resource_id,
                        "service_ns": float(demand.service_ns),
                        "bytes_moved": int(demand.bytes_moved),
                        "energy_pj": float(demand.energy_pj),
                    }
                    for demand in phase.demands
                )
                if demands:
                    topology_phases.append(demands)
                route_ns += max((demand["service_ns"] for demand in demands), default=0.0)
                route_energy += sum(demand["energy_pj"] for demand in demands)
        except ValueError:
            route_kind = "declared_dma_fallback"
        if route_ns > 0.0:
            return route_ns, route_energy, route_kind, tuple(topology_phases), route_component_ids
        bandwidth = self.resource_policy.page_transfer_bandwidth_gb_s
        if bandwidth is None:
            bandwidth = float(self.plan.scenario.host_orchestration_profile.dma_bandwidth_gb_s)
        bulk_ns = byte_count / max(1.0e-12, float(bandwidth))
        return (
            bulk_ns,
            route_energy,
            route_kind,
            (({
                "resource_id": "owner_residency.dma.{}->{}".format(source, target),
                "service_ns": float(bulk_ns),
                "bytes_moved": int(byte_count),
                "energy_pj": 0.0,
            },),),
            route_component_ids,
        )

    def _tensor_get_copy_cost_details(
        self,
        byte_count: int,
        source_component: Optional[str],
        target_component: Optional[str],
    ) -> Tuple[float, float, str, Mapping[str, object]]:
        """Price one resident tensor_get with bulk transfer plus DMA descriptors only."""

        bulk_ns, energy_pj, route_kind, _phases, _components = (
            self._topology_transfer_cost_details(
                byte_count,
                source_component,
                target_component,
                name="serving_runtime_tensor_get_copy",
            )
        )
        if route_kind != "topology" or bulk_ns <= 0.0:
            return bulk_ns, energy_pj, route_kind, {}
        transport = self.plan.scenario.runtime_profile.pcie_dma_iommu
        setup = _dma_setup_service(
            byte_count,
            batch_bytes=transport.dma_batch_bytes,
            queue_depth=transport.dma_queue_depth,
            max_outstanding=transport.dma_max_outstanding,
            fixed_latency_ns=0.0,
            submission_ns_per_wave=(
                self.plan.scenario.host_orchestration_profile.dma_queue_submission_ns
            ),
        )
        return (
            bulk_ns + setup.service_ns,
            energy_pj,
            route_kind,
            {
                "dma_descriptor_service_ns": setup.service_ns,
                "dma_descriptor_transaction_count": setup.transaction_count,
                "dma_descriptor_wave_count": setup.wave_count,
                "dma_descriptor_queue_parallelism": setup.queue_parallelism,
            },
        )

    def _page_transfer_cost_details(
        self,
        byte_count: int,
        page_count: int,
        source_component: Optional[str],
        target_component: Optional[str],
        *,
        residency_page_count: Optional[int] = None,
        include_controller_phases: bool = True,
    ) -> Tuple[
        float,
        float,
        str,
        Tuple[Tuple[Mapping[str, object], ...], ...],
        Tuple[str, ...],
        float,
    ]:
        """Return migration totals plus aggregate V4 controller phases."""

        bulk_ns, route_energy, route_kind, topology_phases, route_component_ids = (
            self._topology_transfer_cost_details(
                byte_count,
                source_component,
                target_component,
                name="serving_runtime_page_migration",
            )
        )
        if route_kind in {"no_declared_backing_path", "same_component"}:
            return bulk_ns, route_energy, route_kind, topology_phases, route_component_ids, 0.0
        source = str(source_component)
        target = str(target_component)
        fault_latency = 0.0
        if self.resource_policy.page_fault_latency_known:
            declared_fault_latency = (
                self.resource_policy.page_fault_latency_ns
            )
            if declared_fault_latency is None:
                declared_fault_latency = float(
                    self.plan.scenario.host_orchestration_profile.dma_latency_ns
            )
            fault_latency = max(0.0, float(declared_fault_latency))
        submission_latency_ns = max(0, int(page_count)) * fault_latency
        try:
            source_kind = _kind(
                self.plan.scenario.hardware.get_component(source)
            )
            target_kind = _kind(
                self.plan.scenario.hardware.get_component(target)
            )
        except (KeyError, ValueError):
            source_kind = target_kind = None
        gpu_memory_kinds = {"gpu", "hbm"}
        gpu_endpoint_transfer = (
            (source_kind in gpu_memory_kinds)
            != (target_kind in gpu_memory_kinds)
        )
        if not include_controller_phases or not gpu_endpoint_transfer:
            legacy_phases = list(topology_phases)
            if submission_latency_ns > 0.0:
                legacy_phases.append(
                    (
                        {
                            "resource_id": (
                                "owner_residency.page_fault_submission"
                            ),
                            "service_ns": submission_latency_ns,
                            "bytes_moved": 0,
                            "energy_pj": 0.0,
                        },
                    )
                )
            legacy_transfer_ns = sum(
                max(
                    (
                        float(demand["service_ns"])
                        for demand in phase
                    ),
                    default=0.0,
                )
                for phase in legacy_phases
            )
            return (
                legacy_transfer_ns,
                max(0.0, route_energy),
                route_kind,
                tuple(legacy_phases),
                route_component_ids,
                submission_latency_ns,
            )
        controller_pages = (
            max(1, int(residency_page_count))
            if residency_page_count is not None
            else max(
                1,
                (
                    byte_count
                    + self.plan.scenario.runtime_profile.pcie_dma_iommu.iommu_page_size_bytes
                    - 1
                )
                // self.plan.scenario.runtime_profile.pcie_dma_iommu.iommu_page_size_bytes,
            )
        )
        (
            cpu_phase,
            setup_phases,
            gpu_data_demands,
            direction,
        ) = self._owner_residency_controller_phases(
            byte_count,
            max(1, int(page_count)),
            controller_pages,
            source,
            target,
            route_component_ids,
            submission_latency_ns,
        )
        controller_topology_phases = list(topology_phases)
        if direction == "page_out":
            controller_topology_phases[0] = (
                *gpu_data_demands,
                *controller_topology_phases[0],
            )
        else:
            controller_topology_phases[-1] = (
                *controller_topology_phases[-1],
                *gpu_data_demands,
            )
        resource_phases = (
            cpu_phase,
            *setup_phases,
            *tuple(controller_topology_phases),
        )
        conserved_transfer_ns = sum(
            max(
                (float(demand["service_ns"]) for demand in phase),
                default=0.0,
            )
            for phase in resource_phases
        )
        return (
            conserved_transfer_ns,
            max(0.0, route_energy),
            route_kind,
            tuple(resource_phases),
            route_component_ids,
            submission_latency_ns,
        )

    def _cohort_touched_kv_pages(self, cohort: BatchCohort) -> int:
        """Estimate distinct KV pages touched by this realized cohort.

        This is a page-granular upper bound from the cohort's context and
        materialized tokens.  It is used only to decide how many currently
        host-backed pages fault back for this cohort; it never expands the
        logical eager reservation into resident bytes.
        """

        total_pages = 0
        for item in cohort.items:
            token_count = max(0, int(item.token_count))
            context_tokens = max(0, int(item.context_tokens))
            materialized_tokens = max(
                0, int(_batch_item_kv_materialized_tokens(item))
            )
            if item.phase in {"prefill", "recompute"}:
                touched_tokens = materialized_tokens
            else:
                touched_tokens = context_tokens + materialized_tokens
            total_pages += self.ledger.pages_for_tokens(touched_tokens)
            # A custom/hand-authored cohort may omit materialized KV metadata;
            # token_count is still a conservative page-touch fallback.
            if touched_tokens <= 0 and token_count > 0:
                total_pages += self.ledger.pages_for_tokens(token_count)
        return max(0, int(total_pages))

    def _cohort_touched_state_allocations(self, cohort: BatchCohort) -> int:
        """Return state allocations that can fault for this cohort."""

        if self.plan.linear_state_policy.bytes_per_request <= 0:
            return 0
        return max(0, len(cohort.items))

    def _apply_resource_contention(
        self, cohort: BatchCohort, cost: BatchCost
    ) -> BatchCost:
        """Account for physical service demand and resident-page migration.

        The lowerer supplies the work demand.  GPU/HBM service rates come from
        the declared profiles, while migration is derived from the active
        working-set shortfall and the actual host backing path.  Custom
        lowerers stay unchanged unless they explicitly opt in through metadata.
        """

        policy = self.resource_policy
        metadata = dict(cost.metadata)
        model = str(metadata.get("model", ""))
        supported_model = model in {
            "analytical_batch_roofline",
            "topology_aware_parallel_lowering",
        }
        if not policy.enabled or (
            not supported_model and not policy.include_custom_lowerers
        ):
            return cost
        if cohort.kind not in {
            "prefill",
            "decode",
            "mtp",
            "mixed",
            "recompute",
        }:
            return cost

        self._ensure_resource_profiles()
        snapshot = dict(self._resource_snapshot(cohort.items))
        host_ns = max(0.0, float(metadata.get("host_orchestration_ns", 0.0)))
        device_ns = max(
            0.0,
            float(
                metadata.get("device_execution_ns", cost.duration_ns - host_ns)
            ),
        )
        resource_demand = metadata.get("resource_demand", {})
        if not isinstance(resource_demand, _ABCMapping):
            resource_demand = {}
        raw_operations = resource_demand.get(
            "gpu_operations", metadata.get("operations", 0.0)
        )
        raw_hbm_bytes = resource_demand.get(
            "hbm_bytes", metadata.get("bytes_moved")
        )
        if raw_hbm_bytes is None:
            raw_hbm_bytes = self._cohort_hbm_bytes(cohort)
        try:
            operations = max(0.0, float(raw_operations))
        except (TypeError, ValueError):
            operations = 0.0
        try:
            hbm_bytes = max(0.0, float(raw_hbm_bytes))
        except (TypeError, ValueError):
            hbm_bytes = 0.0
        gpu_rate = max(
            0.0,
            float(getattr(self._gpu_profile, "attainable_tops", 0.0)) * 1_000.0,
        )
        hbm_rate = max(
            0.0,
            float(
                getattr(self._hbm_profile, "effective_bandwidth_gb_s", 0.0)
            ),
        )
        gpu_demand_ns = operations / gpu_rate if gpu_rate > 0 else 0.0
        hbm_demand_ns = hbm_bytes / hbm_rate if hbm_rate > 0 else 0.0
        shared_service_ns = max(device_ns, gpu_demand_ns, hbm_demand_ns)
        actual_spill = max(0, int(snapshot.get("actual_spill_bytes", 0)))
        spill_pages = max(0, int(snapshot.get("spill_page_count", 0)))
        cache_component = snapshot.get("physical_cache_component")

        # The resource model keeps a small residency state machine.  A new
        # shortfall evicts pages once; a subsequent cohort that touches those
        # pages incurs host->device refaults and re-evicts them if the
        # shortfall remains.  A capacity release restores pages without
        # charging a second device->host move.
        target_kv_pages = max(
            0, int(snapshot.get("kv_spill_page_count", 0))
        )
        target_state_allocations = max(
            0,
            int(snapshot.get("linear_state_spill_allocation_count", 0)),
        )
        current_kv_pages = max(0, int(self._resource_kv_host_backed_pages))
        current_state_allocations = max(
            0, int(self._resource_state_host_backed_allocations)
        )
        if self.residency_manager is not None:
            # The unified owner-aware pool below is the sole migration
            # authority when an adapter declares shared weight/KV/state
            # residency.  Retain this method's compute/HBM roofline, but do
            # not also run the legacy role-local spill approximation.
            target_kv_pages = 0
            target_state_allocations = 0
            current_kv_pages = 0
            current_state_allocations = 0
        capacity_restore_kv_pages = max(0, current_kv_pages - target_kv_pages)
        capacity_restore_state_allocations = max(
            0, current_state_allocations - target_state_allocations
        )
        touched_kv_pages = min(
            current_kv_pages, self._cohort_touched_kv_pages(cohort)
        )
        touched_state_allocations = min(
            current_state_allocations,
            self._cohort_touched_state_allocations(cohort),
        )
        page_in_kv_pages = max(capacity_restore_kv_pages, touched_kv_pages)
        page_in_state_allocations = max(
            capacity_restore_state_allocations, touched_state_allocations
        )
        page_out_kv_pages = max(0, target_kv_pages - current_kv_pages) + max(
            0, touched_kv_pages - capacity_restore_kv_pages
        )
        page_out_state_allocations = max(
            0, target_state_allocations - current_state_allocations
        ) + max(0, touched_state_allocations - capacity_restore_state_allocations)

        kv_page_bytes = max(1, int(self.plan.kv_policy.bytes_per_page))
        state_allocation_bytes = max(
            1, int(self.plan.linear_state_policy.bytes_per_request)
        )
        kv_page_out_bytes = page_out_kv_pages * kv_page_bytes
        kv_page_in_bytes = page_in_kv_pages * kv_page_bytes
        state_page_out_bytes = page_out_state_allocations * state_allocation_bytes
        state_page_in_bytes = page_in_state_allocations * state_allocation_bytes

        page_out_ns = 0.0
        page_out_energy = 0.0
        page_in_ns = 0.0
        page_in_energy = 0.0
        transfer_routes: List[str] = []
        kv_out_ns, kv_out_energy, kv_out_route = self._page_transfer_cost(
            kv_page_out_bytes,
            page_out_kv_pages,
            cache_component,
            self.plan.kv_policy.offload_component,
        )
        kv_in_ns, kv_in_energy, kv_in_route = self._page_transfer_cost(
            kv_page_in_bytes,
            page_in_kv_pages,
            self.plan.kv_policy.offload_component,
            cache_component,
        )
        page_out_ns += kv_out_ns
        page_out_energy += kv_out_energy
        page_in_ns += kv_in_ns
        page_in_energy += kv_in_energy
        if kv_out_ns > 0.0:
            transfer_routes.append("kv_out:{}".format(kv_out_route))
        if kv_in_ns > 0.0:
            transfer_routes.append("kv_in:{}".format(kv_in_route))
        state_out_ns, state_out_energy, state_out_route = self._page_transfer_cost(
            state_page_out_bytes,
            page_out_state_allocations,
            cache_component,
            self.plan.linear_state_policy.offload_component,
        )
        state_in_ns, state_in_energy, state_in_route = self._page_transfer_cost(
            state_page_in_bytes,
            page_in_state_allocations,
            self.plan.linear_state_policy.offload_component,
            cache_component,
        )
        page_out_ns += state_out_ns
        page_out_energy += state_out_energy
        page_in_ns += state_in_ns
        page_in_energy += state_in_energy
        if state_out_ns > 0.0:
            transfer_routes.append("linear_state_out:{}".format(state_out_route))
        if state_in_ns > 0.0:
            transfer_routes.append("linear_state_in:{}".format(state_in_route))
        spill_ns = page_out_ns + page_in_ns
        spill_energy = page_out_energy + page_in_energy
        spill_route = "+".join(transfer_routes) if transfer_routes else "none"
        self._resource_kv_host_backed_pages = target_kv_pages
        self._resource_state_host_backed_allocations = target_state_allocations
        adjusted_device_ns = shared_service_ns + spill_ns
        adjusted_duration_ns = host_ns + adjusted_device_ns
        base_device_ns = max(device_ns, 1.0e-9)
        factor = adjusted_device_ns / base_device_ns
        pressure_capacity = max(
            0, int(snapshot.get("effective_resident_capacity_bytes", 0))
        )
        pressure_bytes = max(
            0, int(snapshot.get("resident_capacity_shortfall_bytes", 0))
        )
        pressure_ratio = (
            pressure_bytes / float(pressure_capacity)
            if pressure_capacity > 0
            else (1.0 if pressure_bytes > 0 else 0.0)
        )
        metadata.update(
            {
                **snapshot,
                "resource_contention_enabled": True,
                "resource_contention_factor": factor,
                "resource_contention_vram_pressure": pressure_ratio,
                "resource_contention_base_duration_ns": cost.duration_ns,
                "resource_contention_base_device_ns": device_ns,
                "resource_contention_gpu_operations": operations,
                "resource_contention_hbm_bytes": hbm_bytes,
                "resource_contention_gpu_service_rate_ops_per_ns": gpu_rate,
                "resource_contention_hbm_service_rate_bytes_per_ns": hbm_rate,
                "resource_contention_gpu_demand_ns": gpu_demand_ns,
                "resource_contention_hbm_demand_ns": hbm_demand_ns,
                "resource_contention_shared_service_ns": shared_service_ns,
                "resource_contention_actual_spill_bytes": actual_spill,
                "resource_contention_spill_page_count": spill_pages,
                "resource_contention_page_transfer_ns": spill_ns,
                "resource_contention_page_transfer_energy_pj": spill_energy,
                "resource_contention_page_transfer_route": spill_route,
                "resource_contention_page_out_ns": page_out_ns,
                "resource_contention_page_out_energy_pj": page_out_energy,
                "resource_contention_page_in_ns": page_in_ns,
                "resource_contention_page_in_energy_pj": page_in_energy,
                "resource_contention_kv_page_out_count": page_out_kv_pages,
                "resource_contention_kv_page_in_count": page_in_kv_pages,
                "resource_contention_kv_refault_page_count": touched_kv_pages,
                "resource_contention_linear_state_page_out_count": (
                    page_out_state_allocations
                ),
                "resource_contention_linear_state_page_in_count": (
                    page_in_state_allocations
                ),
                "resource_contention_host_backed_committed_bytes": (
                    snapshot.get("logical_host_backed_bytes", 0)
                ),
                "resource_contention_logical_reservation_bytes": (
                    snapshot.get("logical_reservation_bytes", 0)
                ),
                "device_execution_ns": adjusted_device_ns,
            }
        )
        return BatchCost(
            duration_ns=max(1.0e-9, adjusted_duration_ns),
            energy_pj=max(0.0, cost.energy_pj + spill_energy),
            metadata=metadata,
        )

    @staticmethod
    def _owner_residency_trace(
        metadata: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], ...]:
        """Validate the planner's ordered physical-access trace.

        An absent trace is deliberately empty.  Coverage summaries list what
        a lowering contains, not the execution order or access count, so they
        must never be promoted to residency facts here.
        """

        raw_trace = metadata.get("residency_accesses", ())
        if raw_trace is None:
            return ()
        if isinstance(raw_trace, (str, bytes, _ABCMapping)) or not isinstance(
            raw_trace, _ABCSequence
        ):
            raise ValueError("residency_accesses must be an ordered sequence")
        trace: List[Mapping[str, Any]] = []
        for index, raw_access in enumerate(raw_trace):
            if not isinstance(raw_access, _ABCMapping):
                raise ValueError(
                    "residency_accesses[{}] must be a mapping".format(index)
                )
            trace.append(raw_access)
        return tuple(trace)

    def _owner_residency_target(
        self,
        access: Mapping[str, Any],
    ) -> Optional[str]:
        manager = self.residency_manager
        if manager is None:
            return None
        requested = access.get(
            "requested_tensor_id",
            access.get("tensor_id", access.get("requested_tensor")),
        )
        canonical = access.get(
            "canonical_owner_id",
            access.get("canonical_owner", access.get("owner_id")),
        )
        explicit_target = access.get("allocation_id")
        candidates = tuple(
            str(value)
            for value in (requested, explicit_target, canonical)
            if value is not None and str(value)
        )
        for target_id in candidates:
            if target_id in manager.allocations or target_id in manager.views:
                return target_id

        raw_component = access.get(
            "weight_target_component",
            access.get(
                "target_component_id",
                access.get("component_id"),
            ),
        )
        target_component = str(raw_component) if raw_component else None
        if target_component and target_component != manager.pool.component_id:
            return None
        placement = self.plan.scenario.placement
        placed_component = None
        for target_id in reversed(candidates):
            if target_id in placement.tensor_to_component:
                placed_component = str(
                    placement.tensor_to_component[target_id]
                )
                break
        if placed_component and placed_component != manager.pool.component_id:
            return None
        if target_component == manager.pool.component_id or (
            placed_component == manager.pool.component_id
        ):
            raise ValueError(
                "residency trace references missing owner/view: {}".format(
                    candidates[0] if candidates else "<missing id>"
                )
            )
        # A weight owned by host/CIM may have no component annotation in an
        # older trace.  Skipping it is backward compatible and does not invent
        # device residency.
        return None

    @staticmethod
    def _owner_residency_access_size(
        access: Mapping[str, Any],
    ) -> Optional[int]:
        raw_size = access.get("size_bytes", access.get("byte_count"))
        if raw_size is None:
            return None
        if isinstance(raw_size, bool):
            raise ValueError("residency access byte_count must be an integer")
        try:
            size = int(raw_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "residency access byte_count must be an integer"
            ) from exc
        if size < 0 or size != raw_size:
            raise ValueError(
                "residency access byte_count must be a non-negative integer"
            )
        return size

    def _apply_traced_owner_accesses(
        self,
        trace: Sequence[Mapping[str, Any]],
        *,
        cohort_id: str,
    ) -> Tuple[List[Mapping[str, object]], List[str], bool, bool]:
        """Apply planner accesses and return audit rows plus temp owners."""

        manager = self.residency_manager
        if manager is None:
            return ([], [], False, False)
        audit: List[Mapping[str, object]] = []
        temporary_ids: List[str] = []
        explicit_kv = False
        explicit_state = False
        try:
            for index, access in enumerate(trace):
                kind = str(
                    access.get("kind", "model_weight")
                ).strip().lower()
                raw_operation = str(
                    access.get("operation", "read")
                ).strip().lower()
                if kind in {"kv", "kv_cache"}:
                    explicit_kv = True
                if kind in {"state", "linear_state", "recurrent_state"}:
                    explicit_state = True
                lifecycle = str(access.get("lifecycle", "")).strip().lower()
                is_staged_weight = kind in {
                    "staged_weight",
                    "temporary_model_weight",
                }
                is_temporary = lifecycle == "temporary" or kind in {
                    "activation",
                    "temporary",
                    "workspace",
                }
                if is_staged_weight:
                    raw_id = access.get(
                        "allocation_id",
                        access.get("requested_tensor_id"),
                    )
                    target_id = str(raw_id or "").strip()
                    if not target_id:
                        raise ValueError(
                            "staged weight residency access requires "
                            "allocation_id"
                        )
                    size = self._owner_residency_access_size(access)
                    if size is None or size <= 0:
                        raise ValueError(
                            "staged weight residency access requires positive "
                            "byte_count"
                        )
                    residency_component = access.get(
                        "residency_component_id"
                    )
                    if (
                        residency_component is not None
                        and str(residency_component)
                        != manager.pool.component_id
                    ):
                        continue
                    if raw_operation == "register":
                        if target_id in manager.allocations:
                            raise ValueError(
                                "staged weight allocation already registered: "
                                "{}".format(target_id)
                            )
                        raw_backing = access.get(
                            "backing_component_id",
                            access.get("weight_owner_component_id"),
                        )
                        backing = (
                            str(raw_backing).strip() if raw_backing else None
                        )
                        if not backing or backing == manager.pool.component_id:
                            raise ValueError(
                                "staged weight allocation requires remote host "
                                "backing"
                            )
                        manager.register(
                            target_id,
                            size,
                            kind="model_weight",
                            backing=backing,
                            committed_bytes=size,
                            resident_bytes=size,
                            evictable=True,
                            lifecycle="temporary",
                            read_only=True,
                        )
                        self._capture_residency_interval_peak()
                        temporary_ids.append(target_id)
                        self.residency_temporary_allocations += 1
                        self._clean_eviction_service_by_allocation[
                            target_id
                        ] = _CLEAN_EVICTION_FREE_DISCARD
                        self._staged_weight_clean_discard_ids.add(target_id)
                        audit.append(
                            {
                                "target_id": target_id,
                                "owner_id": target_id,
                                "operation": "register",
                                "offset_bytes": 0,
                                "size_bytes": size,
                                "physical_residency_bytes": size,
                                "make_resident_scope": "registration",
                                "kind": kind,
                                "read_only": True,
                                "fully_resident": True,
                                "h2d_already_charged": True,
                                "backing_component_id": backing,
                                "operator_invocation_group_id": access.get(
                                    "operator_invocation_group_id"
                                ),
                                "weight_read_invocation_id": access.get(
                                    "weight_read_invocation_id",
                                    access.get("invocation_id"),
                                ),
                            }
                        )
                        continue
                    if raw_operation == "release":
                        if target_id not in manager.allocations:
                            raise ValueError(
                                "staged weight release references missing "
                                "allocation: {}".format(target_id)
                            )
                        owner = manager.allocations[target_id]
                        resident_bytes = int(owner.resident_bytes)
                        if resident_bytes:
                            manager.evict(target_id)
                        manager.release(target_id)
                        self._capture_residency_interval_peak()
                        if target_id in temporary_ids:
                            temporary_ids.remove(target_id)
                        self.residency_temporary_releases += 1
                        audit.append(
                            {
                                "target_id": target_id,
                                "owner_id": target_id,
                                "operation": "release",
                                "offset_bytes": 0,
                                "size_bytes": size,
                                "physical_residency_bytes": resident_bytes,
                                "make_resident_scope": "clean_discard_release",
                                "kind": kind,
                                "read_only": True,
                                "release_semantics": "clean_discard",
                                "dirty_writeback": False,
                                "operator_invocation_group_id": access.get(
                                    "operator_invocation_group_id"
                                ),
                                "weight_read_invocation_id": access.get(
                                    "weight_read_invocation_id",
                                    access.get("invocation_id"),
                                ),
                            }
                        )
                        continue
                    if raw_operation != "read":
                        raise ValueError(
                            "unsupported staged weight lifecycle operation: "
                            "{}".format(raw_operation)
                        )
                    if target_id not in manager.allocations:
                        raise ValueError(
                            "staged weight read references missing allocation: "
                            "{}".format(target_id)
                        )
                    target = target_id
                elif is_temporary:
                    raw_id = access.get(
                        "allocation_id",
                        access.get(
                            "requested_tensor_id",
                            "runtime.temporary.{}.{:04d}".format(
                                cohort_id, index
                            ),
                        ),
                    )
                    target_id = str(raw_id)
                    size = self._owner_residency_access_size(access)
                    if size is None or size <= 0:
                        raise ValueError(
                            "temporary residency access requires positive "
                            "byte_count"
                        )
                    if target_id not in manager.allocations:
                        manager.register_temporary(
                            target_id,
                            size,
                            kind=kind if kind else "workspace",
                        )
                        self._capture_residency_interval_peak()
                        temporary_ids.append(target_id)
                        self.residency_temporary_allocations += 1
                    target = target_id
                else:
                    target = self._owner_residency_target(access)
                    if target is None:
                        continue
                    # Planner byte_count is GEMM read traffic.  An aggregate
                    # physical owner has no submatrix offset contract, so a
                    # fault makes the whole owner/view resident.  Explicit
                    # physical size/offset metadata opts into a subrange.
                    if kind in {
                        "model_weight",
                        "weight",
                        "weights",
                    } and not (
                        "size_bytes" in access or "offset_bytes" in access
                    ):
                        size = None
                    else:
                        size = self._owner_residency_access_size(access)
                try:
                    operation = AccessOperation(raw_operation)
                except ValueError as exc:
                    raise ValueError(
                        "unsupported residency access operation: {}".format(
                            raw_operation
                        )
                    ) from exc
                try:
                    offset = int(access.get("offset_bytes", 0))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "residency access offset_bytes must be an integer"
                    ) from exc
                make_resident_scope = "range"
                if kind in {
                    "model_weight",
                    "weight",
                    "weights",
                    "staged_weight",
                    "temporary_model_weight",
                }:
                    make_resident_scope = _normalized_make_resident_scope(
                        (
                            "range"
                            if is_staged_weight
                            else self._weight_allocation_contract.get(
                                "make_resident_scope", "range"
                            )
                        )
                    )
                _make_resident_for_scope(
                    manager, target, make_resident_scope
                )
                self._capture_residency_interval_peak()
                result = manager.access(
                    MemoryAccess(
                        target,
                        operation,
                        offset_bytes=offset,
                        size_bytes=size,
                    )
                )
                self._lease_residency_result(
                    result,
                    self._residency_consumer_task_ids(access),
                )
                self._capture_residency_interval_peak()
                self.residency_accesses += 1
                if kind in {
                    "model_weight",
                    "weight",
                    "weights",
                    "staged_weight",
                    "temporary_model_weight",
                }:
                    self.residency_weight_accesses += 1
                elif kind in {"kv", "kv_cache"}:
                    self.residency_kv_accesses += 1
                elif kind in {"state", "linear_state", "recurrent_state"}:
                    self.residency_state_accesses += 1
                audit.append(
                    {
                        "target_id": target,
                        "owner_id": result.owner_id,
                        "operation": result.operation.value,
                        "offset_bytes": result.offset_bytes,
                        "size_bytes": result.size_bytes,
                        "physical_residency_bytes": (
                            _make_resident_scope_physical_bytes(
                                manager,
                                target,
                                make_resident_scope,
                                result.physical_residency_bytes,
                            )
                        ),
                        "make_resident_scope": make_resident_scope,
                        "kind": kind,
                        "operator_invocation_group_id": access.get(
                            "operator_invocation_group_id"
                        ),
                        "weight_read_invocation_id": access.get(
                            "weight_read_invocation_id",
                            access.get("invocation_id"),
                        ),
                    }
                )
        except Exception:
            # If validation or capacity fails after a temporary was
            # registered, the caller cannot receive its id for cleanup.
            for temporary_id in reversed(temporary_ids):
                if temporary_id in manager.allocations:
                    manager.release(temporary_id)
                    self._capture_residency_interval_peak()
                    self.residency_temporary_releases += 1
                if temporary_id in self._staged_weight_clean_discard_ids:
                    self._clean_eviction_service_by_allocation.pop(
                        temporary_id, None
                    )
                    self._staged_weight_clean_discard_ids.discard(
                        temporary_id
                    )
            for staged_id in tuple(
                self._staged_weight_clean_discard_ids
            ):
                if staged_id in manager.allocations:
                    continue
                self._clean_eviction_service_by_allocation.pop(
                    staged_id, None
                )
                self._staged_weight_clean_discard_ids.discard(staged_id)
            raise
        return (audit, temporary_ids, explicit_kv, explicit_state)

    def _apply_mtp_draft_kv_accesses(
        self,
        cohort: BatchCohort,
        *,
        access_phase: str,
        invocation_group: Mapping[str, Any],
    ) -> List[Mapping[str, object]]:
        """Apply one MTP side invocation to its independent draft KV context."""

        manager = self.residency_manager
        contract = self._mtp_draft_kv_allocation_contract
        if (
            manager is None
            or not contract
        ):
            return []
        if access_phase not in {"read", "write"}:
            raise ValueError("MTP draft KV access_phase must be read or write")
        bytes_per_cell = _contract_nonnegative_int(
            contract,
            "bytes_per_cell",
            "bytes_per_token",
        )
        if bytes_per_cell <= 0:
            return []
        page_tokens = max(1, int(self.plan.kv_policy.tokens_per_page))
        page_bytes = bytes_per_cell * page_tokens
        make_resident_scope = _normalized_make_resident_scope(
            contract.get("make_resident_scope", "range")
        )
        raw_request_ids = invocation_group.get("request_ids", ())
        request_ids = (
            {str(item) for item in raw_request_ids}
            if isinstance(raw_request_ids, _ABCSequence)
            and not isinstance(raw_request_ids, (str, bytes))
            else set()
        )
        try:
            draft_step = max(0, int(invocation_group.get("draft_step", 0)))
        except (TypeError, ValueError, OverflowError):
            draft_step = 0
        residency_role = str(
            invocation_group.get("residency_role", "mtp_draft_context")
        ).strip().lower()
        catchup = residency_role == "mtp_draft_context_catchup"

        def _request_token_values(key: str) -> Optional[Dict[str, int]]:
            raw_values = invocation_group.get(key)
            if raw_values is None:
                return None
            if isinstance(raw_values, _ABCMapping):
                rows = tuple(raw_values.items())
            elif isinstance(raw_values, _ABCSequence) and not isinstance(
                raw_values, (str, bytes)
            ):
                rows = tuple(raw_values)
            else:
                raise ValueError("{} must be a request/value mapping".format(key))
            values: Dict[str, int] = {}
            for row in rows:
                if not isinstance(row, _ABCSequence) or isinstance(
                    row, (str, bytes)
                ) or len(row) != 2:
                    raise ValueError(
                        "{} entries must be request/value pairs".format(key)
                    )
                request_id = str(row[0])
                if request_id in values:
                    raise ValueError(
                        "{} request ids must be unique".format(key)
                    )
                try:
                    value = int(row[1])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "{} values must be non-negative integers".format(key)
                    ) from exc
                if value < 0:
                    raise ValueError(
                        "{} values must be non-negative integers".format(key)
                    )
                values[request_id] = value
            return values

        catchup_token_counts = (
            _request_token_values("token_count_by_request") if catchup else None
        )
        catchup_token_offsets = (
            _request_token_values("token_offset_by_request") if catchup else None
        )
        declares_physical_catchup = catchup and any(
            key in invocation_group
            for key in (
                "physical_chunk_index",
                "physical_row_start",
                "physical_ubatch_rows",
                "token_count_by_request",
                "token_offset_by_request",
            )
        )
        if declares_physical_catchup:
            if not request_ids:
                raise ValueError(
                    "physical MTP catchup requires request identity"
                )
            if catchup_token_counts is None or catchup_token_offsets is None:
                raise ValueError(
                    "physical MTP catchup requires count and offset maps"
                )
            if (
                set(catchup_token_counts) != request_ids
                or set(catchup_token_offsets) != request_ids
            ):
                raise ValueError(
                    "physical MTP catchup maps must match request_ids"
                )
            if any(value <= 0 for value in catchup_token_counts.values()):
                raise ValueError(
                    "physical MTP catchup counts must be positive"
                )
            try:
                lane_count = int(invocation_group["lane_count"])
                physical_rows = int(invocation_group["physical_ubatch_rows"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "physical MTP catchup requires integer row counts"
                ) from exc
            counted_rows = sum(catchup_token_counts.values())
            if counted_rows != lane_count or lane_count != physical_rows:
                raise ValueError(
                    "physical MTP catchup row counts must agree"
                )
            item_token_counts = {
                str(item.request_id): max(0, int(item.token_count))
                for item in cohort.items
            }
            if any(
                request_id not in item_token_counts
                or catchup_token_offsets[request_id]
                + catchup_token_counts[request_id]
                > item_token_counts[request_id]
                for request_id in request_ids
            ):
                raise ValueError(
                    "physical MTP catchup request span exceeds item tokens"
                )
        audit: List[Mapping[str, object]] = []

        def _extent(target_id: str) -> int:
            if target_id in manager.allocations:
                return max(
                    0, int(manager.allocations[target_id].committed_bytes)
                )
            if target_id in manager.views:
                return max(0, int(manager.views[target_id].size_bytes))
            return 0

        for item in cohort.items:
            if not catchup and item.phase != "mtp":
                continue
            request_id = str(item.request_id)
            if request_ids and request_id not in request_ids:
                continue
            slot = self._request_residency_slots.get(request_id)
            if slot is None:
                raise RuntimeError(
                    "request {} has no owner-aware residency slot".format(
                        request_id
                    )
                )
            target_id = "runtime.mtp_draft.kv.slot.{:04d}".format(slot)
            if (
                target_id not in manager.allocations
                and target_id not in manager.views
            ):
                continue
            target_extent = _extent(target_id)
            base_context_tokens = max(0, int(item.context_tokens))
            context_tokens = base_context_tokens + draft_step
            span_tokens = 1
            catchup_token_offset = 0
            if catchup:
                span_tokens = (
                    max(0, int(item.token_count))
                    if catchup_token_counts is None
                    else catchup_token_counts.get(request_id, 0)
                )
                catchup_token_offset = (
                    0
                    if catchup_token_offsets is None
                    else catchup_token_offsets.get(request_id, 0)
                )
                if span_tokens <= 0:
                    continue
            catchup_context_tokens = base_context_tokens + catchup_token_offset
            if access_phase == "read":
                read_context_tokens = (
                    catchup_context_tokens if catchup else context_tokens
                )
                touched_bytes = min(
                    target_extent,
                    int(math.ceil(read_context_tokens / float(page_tokens)))
                    * page_bytes,
                )
                if touched_bytes <= 0:
                    continue
                _make_resident_for_scope(
                    manager, target_id, make_resident_scope
                )
                self._capture_residency_interval_peak()
                result = manager.read(target_id, size_bytes=touched_bytes)
            else:
                write_start = min(
                    target_extent,
                    (
                        catchup_context_tokens
                        if catchup
                        else context_tokens
                    )
                    * bytes_per_cell,
                )
                write_end = min(
                    target_extent,
                    (
                        (catchup_context_tokens + span_tokens) * bytes_per_cell
                        if catchup
                        else int(
                            math.ceil(
                                (context_tokens + 1) / float(page_tokens)
                            )
                        )
                        * page_bytes
                    ),
                )
                if write_end <= write_start:
                    continue
                touched_bytes = write_end - write_start
                _make_resident_for_scope(
                    manager, target_id, make_resident_scope
                )
                self._capture_residency_interval_peak()
                result = manager.write(
                    target_id,
                    offset_bytes=write_start,
                    size_bytes=touched_bytes,
                )
            self._lease_residency_result(
                result,
                self._residency_consumer_task_ids(invocation_group),
            )
            self._capture_residency_interval_peak()
            self.residency_accesses += 1
            self.residency_kv_accesses += 1
            audit.append(
                {
                    "target_id": target_id,
                    "owner_id": result.owner_id,
                    "operation": access_phase,
                    "offset_bytes": result.offset_bytes,
                    "size_bytes": result.size_bytes,
                    "logical_touched_bytes": touched_bytes,
                    "physical_residency_bytes": (
                        _make_resident_scope_physical_bytes(
                            manager,
                            target_id,
                            make_resident_scope,
                            result.physical_residency_bytes,
                        )
                    ),
                    "make_resident_scope": make_resident_scope,
                    "kind": "mtp_draft_kv_cache",
                    "request_id": request_id,
                    "draft_step": draft_step,
                    "mtp_draft_access_role": (
                        "catchup" if catchup else "proposer"
                    ),
                    "catchup_rows": span_tokens if catchup else 0,
                    "catchup_token_offset": (
                        catchup_token_offset if catchup else 0
                    ),
                    "catchup_context_tokens": (
                        catchup_context_tokens if catchup else 0
                    ),
                    "operator_invocation_group_id": invocation_group.get(
                        "group_id"
                    ),
                }
            )
        return audit

    def _apply_slot_owner_accesses(
        self,
        cohort: BatchCohort,
        *,
        include_kv: bool,
        include_state: bool,
        access_phase: str,
        invocation_group: Optional[Mapping[str, Any]] = None,
        speculative_state_ids: Optional[Dict[str, str]] = None,
        temporary_ids: Optional[List[str]] = None,
    ) -> List[Mapping[str, object]]:
        """Apply one invocation group's request-local KV/state phase."""

        manager = self.residency_manager
        if manager is None:
            return []
        if access_phase not in {"read", "write"}:
            raise ValueError("owner residency access_phase must be read or write")
        audit: List[Mapping[str, object]] = []
        page_tokens = max(1, int(self.plan.kv_policy.tokens_per_page))
        declared_kv_per_token, _ = _declared_serving_kv_quantum(self.plan)
        device_kv_per_token = _contract_nonnegative_int(
            self._kv_allocation_contract,
            "device_bytes_per_cell",
            "bytes_per_cell",
            default=max(0, int(declared_kv_per_token)),
        )
        bytes_per_token = (
            max(0, int(device_kv_per_token))
            if device_kv_per_token > 0
            else int(math.ceil(max(0, int(self.plan.kv_policy.bytes_per_page)) / float(page_tokens)))
        )
        page_bytes = max(0, int(self.plan.kv_policy.bytes_per_page)) or bytes_per_token * page_tokens
        state_bytes = _contract_nonnegative_int(
            self._state_allocation_contract,
            "base_bytes_per_sequence",
            "base_bytes_per_seq",
            "current_state_bytes_per_sequence",
            "bytes_per_plane",
            default=max(
                0, int(self.plan.linear_state_policy.bytes_per_request)
            ),
        )
        rollback_plane_count = _contract_nonnegative_int(
            self._state_allocation_contract,
            "rollback_plane_count",
            "n_rs_seq",
        )
        uses_embedded_rollback_planes = (
            state_bytes > 0 and rollback_plane_count > 0
        )
        kv_make_resident_scope = _normalized_make_resident_scope(
            self._kv_allocation_contract.get("make_resident_scope", "range")
        )
        state_make_resident_scope = _normalized_make_resident_scope(
            self._state_allocation_contract.get(
                "make_resident_scope", "range"
            )
        )

        def _target_extent(target_id: str) -> int:
            if target_id in manager.allocations:
                return max(
                    0, int(manager.allocations[target_id].committed_bytes)
                )
            if target_id in manager.views:
                return max(0, int(manager.views[target_id].size_bytes))
            return 0

        def _state_read(
            target_id: str,
            *,
            offset_bytes: int = 0,
            size_bytes: Optional[int] = None,
        ) -> object:
            logical_size = state_bytes if size_bytes is None else size_bytes
            _make_resident_for_scope(
                manager, target_id, state_make_resident_scope
            )
            self._capture_residency_interval_peak()
            result = manager.read(
                target_id,
                offset_bytes=offset_bytes,
                size_bytes=logical_size,
            )
            self._lease_residency_result(
                result,
                self._residency_consumer_task_ids(invocation_group),
            )
            self._capture_residency_interval_peak()
            return result

        def _state_write(
            target_id: str,
            *,
            offset_bytes: int = 0,
            size_bytes: Optional[int] = None,
        ) -> object:
            logical_size = state_bytes if size_bytes is None else size_bytes
            _make_resident_for_scope(
                manager, target_id, state_make_resident_scope
            )
            self._capture_residency_interval_peak()
            result = manager.write(
                target_id,
                offset_bytes=offset_bytes,
                size_bytes=logical_size,
            )
            self._lease_residency_result(
                result,
                self._residency_consumer_task_ids(invocation_group),
            )
            self._capture_residency_interval_peak()
            return result

        runtime_semantics = (
            str(
                invocation_group.get("linear_state_runtime_semantics", "")
            ).strip()
            if invocation_group is not None
            else ""
        )
        group_uses_speculative_state = runtime_semantics in {
            "speculative_verification",
            "mixed_persistent_and_speculative_update",
        }
        if group_uses_speculative_state and (
            speculative_state_ids is None or temporary_ids is None
        ):
            raise RuntimeError(
                "speculative state residency requires lifecycle tracking"
            )

        def _string_tuple(value: object) -> Tuple[str, ...]:
            if not isinstance(value, _ABCSequence) or isinstance(value, (str, bytes)):
                return ()
            return tuple(str(item) for item in value)

        def _integer_tuple(value: object) -> Tuple[int, ...]:
            if not isinstance(value, _ABCSequence) or isinstance(value, (str, bytes)):
                return ()
            parsed: List[int] = []
            for item in value:
                try:
                    parsed.append(int(item))
                except (TypeError, ValueError, OverflowError):
                    continue
            return tuple(parsed)

        group_lanes = (
            tuple(
                lane
                for lane in invocation_group.get("lanes", ())
                if isinstance(lane, _ABCMapping)
            )
            if invocation_group is not None
            and isinstance(invocation_group.get("lanes", ()), _ABCSequence)
            and not isinstance(
                invocation_group.get("lanes", ()), (str, bytes)
            )
            else ()
        )
        commit_lane_ids = set(
            _string_tuple(
                invocation_group.get(
                    "linear_state_commit_snapshot_lane_ids", ()
                )
                if invocation_group is not None
                else ()
            )
        )
        commit_positions = set(
            _integer_tuple(
                invocation_group.get(
                    "linear_state_commit_snapshot_positions", ()
                )
                if invocation_group is not None
                else ()
            )
        )
        read_source = (
            str(invocation_group.get("linear_state_read_source", "committed"))
            .strip()
            .lower()
            if invocation_group is not None
            else "committed"
        )

        def _request_lanes(request_id: str) -> Tuple[Mapping[str, Any], ...]:
            return tuple(
                lane
                for lane in group_lanes
                if str(lane.get("request_id", "")) == request_id
            )

        def _commits_request(
            request_id: str, positions: Sequence[int]
        ) -> bool:
            lanes = _request_lanes(request_id)
            if commit_lane_ids:
                return any(
                    str(lane.get("lane_id", "")) in commit_lane_ids
                    for lane in lanes
                )
            if commit_positions:
                return any(int(position) in commit_positions for position in positions)
            return False

        def _releases_request(request_id: str) -> bool:
            lanes = _request_lanes(request_id)
            explicit_lane_boundaries = tuple(
                lane for lane in lanes if "linear_state_release_boundary" in lane
            )
            if explicit_lane_boundaries:
                return any(
                    bool(lane.get("linear_state_release_boundary"))
                    for lane in explicit_lane_boundaries
                )
            return bool(
                invocation_group.get("linear_state_temporary_release", False)
                if invocation_group is not None
                else False
            )

        def _state_audit(
            result: object,
            *,
            request_id: str,
            persistence: str,
            operation: str,
        ) -> Mapping[str, object]:
            return {
                "target_id": result.target_id,
                "owner_id": result.owner_id,
                "operation": operation,
                "offset_bytes": result.offset_bytes,
                "size_bytes": result.size_bytes,
                "logical_touched_bytes": state_bytes,
                "physical_residency_bytes": (
                    _make_resident_scope_physical_bytes(
                        manager,
                        str(result.target_id),
                        state_make_resident_scope,
                        result.physical_residency_bytes,
                    )
                ),
                "make_resident_scope": state_make_resident_scope,
                "kind": "linear_state",
                "state_persistence": persistence,
                "linear_state_read_source": read_source,
                "request_id": request_id,
                "operator_invocation_group_id": (
                    invocation_group.get("group_id")
                    if invocation_group is not None
                    else None
                ),
            }

        indexed_items: List[Tuple[BatchItem, Tuple[int, ...]]] = []
        if invocation_group is None:
            indexed_items = [
                (
                    item,
                    tuple(range(max(0, int(item.token_count)))),
                )
                for item in cohort.items
            ]
        else:
            raw_lanes = invocation_group.get("lanes", ())
            if isinstance(raw_lanes, _ABCSequence) and not isinstance(
                raw_lanes, (str, bytes)
            ):
                positions_by_item: Dict[int, List[int]] = {}
                for raw_lane in raw_lanes:
                    if not isinstance(raw_lane, _ABCMapping):
                        continue
                    try:
                        item_index = int(raw_lane.get("item_index", -1))
                        position = int(raw_lane.get("position", 0))
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if 0 <= item_index < len(cohort.items) and position >= 0:
                        positions_by_item.setdefault(item_index, []).append(
                            position
                        )
                indexed_items = [
                    (
                        cohort.items[item_index],
                        tuple(sorted(set(positions))),
                    )
                    for item_index, positions in sorted(
                        positions_by_item.items()
                    )
                ]
            if not indexed_items:
                raw_request_ids = invocation_group.get("request_ids", ())
                request_ids = (
                    tuple(str(item) for item in raw_request_ids)
                    if isinstance(raw_request_ids, _ABCSequence)
                    and not isinstance(raw_request_ids, (str, bytes))
                    else ()
                )
                indexed_items = [
                    (
                        item,
                        tuple(range(max(0, int(item.token_count)))),
                    )
                    for item in cohort.items
                    if str(item.request_id) in request_ids
                ]
        for item, positions in indexed_items:
            if not positions:
                continue
            request_id = str(item.request_id)
            uses_speculative_state = (
                group_uses_speculative_state and item.phase == "mtp"
            )
            if request_id not in self._request_residency_slots:
                raise RuntimeError(
                    "request {} has no owner-aware residency slot".format(
                        request_id
                    )
                )
            slot = self._request_residency_slots[request_id]
            if include_kv and page_bytes > 0:
                kv_owner_id = "runtime.kv.slot.{:04d}".format(slot)
                if (
                    kv_owner_id in manager.allocations
                    or kv_owner_id in manager.views
                ):
                    target_extent = _target_extent(kv_owner_id)
                    context_tokens = max(0, int(item.context_tokens))
                    materialized_tokens = max(
                        0, int(_batch_item_kv_materialized_tokens(item))
                    )
                    if (
                        access_phase == "read"
                        and item.phase not in {"prefill", "recompute"}
                    ):
                        group_context_tokens = context_tokens + max(positions)
                        read_bytes = min(
                            target_extent,
                            self.ledger.pages_for_tokens(group_context_tokens)
                            * page_bytes,
                        )
                        if read_bytes > 0:
                            _make_resident_for_scope(
                                manager,
                                kv_owner_id,
                                kv_make_resident_scope,
                            )
                            self._capture_residency_interval_peak()
                            result = manager.read(
                                kv_owner_id,
                                size_bytes=read_bytes,
                            )
                            self._lease_residency_result(
                                result,
                                self._residency_consumer_task_ids(
                                    invocation_group
                                ),
                            )
                            self._capture_residency_interval_peak()
                            self.residency_accesses += 1
                            self.residency_kv_accesses += 1
                            audit.append(
                                {
                                    "target_id": kv_owner_id,
                                    "owner_id": result.owner_id,
                                    "operation": "read",
                                    "offset_bytes": 0,
                                    "size_bytes": result.size_bytes,
                                    "physical_residency_bytes": (
                                        _make_resident_scope_physical_bytes(
                                            manager,
                                            kv_owner_id,
                                            kv_make_resident_scope,
                                            result.physical_residency_bytes,
                                        )
                                    ),
                                    "logical_touched_bytes": read_bytes,
                                    "make_resident_scope": (
                                        kv_make_resident_scope
                                    ),
                                    "kind": "kv_cache",
                                    "request_id": request_id,
                                    "operator_invocation_group_id": (
                                        invocation_group.get("group_id")
                                        if invocation_group is not None
                                        else None
                                    ),
                                }
                            )
                    materialized_positions = tuple(
                        position
                        for position in positions
                        if position < materialized_tokens
                    )
                    if access_phase == "write" and materialized_positions:
                        write_start = min(
                            target_extent,
                            (
                                context_tokens
                                + min(materialized_positions)
                            )
                            * bytes_per_token,
                        )
                        write_end = min(
                            target_extent,
                            self.ledger.pages_for_tokens(
                                context_tokens
                                + max(materialized_positions)
                                + 1
                            )
                            * page_bytes,
                        )
                        if write_end > write_start:
                            _make_resident_for_scope(
                                manager,
                                kv_owner_id,
                                kv_make_resident_scope,
                            )
                            self._capture_residency_interval_peak()
                            result = manager.write(
                                kv_owner_id,
                                offset_bytes=write_start,
                                size_bytes=write_end - write_start,
                            )
                            self._lease_residency_result(
                                result,
                                self._residency_consumer_task_ids(
                                    invocation_group
                                ),
                            )
                            self._capture_residency_interval_peak()
                            self.residency_accesses += 1
                            self.residency_kv_accesses += 1
                            audit.append(
                                {
                                    "target_id": kv_owner_id,
                                    "owner_id": result.owner_id,
                                    "operation": "write",
                                    "offset_bytes": write_start,
                                    "size_bytes": write_end - write_start,
                                    "physical_residency_bytes": (
                                        _make_resident_scope_physical_bytes(
                                            manager,
                                            kv_owner_id,
                                            kv_make_resident_scope,
                                            result.physical_residency_bytes,
                                        )
                                    ),
                                    "logical_touched_bytes": (
                                        write_end - write_start
                                    ),
                                    "make_resident_scope": (
                                        kv_make_resident_scope
                                    ),
                                    "kind": "kv_cache",
                                    "request_id": request_id,
                                    "operator_invocation_group_id": (
                                        invocation_group.get("group_id")
                                        if invocation_group is not None
                                        else None
                                    ),
                                }
                            )
            if include_state and state_bytes > 0:
                state_owner_id = "runtime.linear_state.slot.{:04d}".format(
                    slot
                )
                if (
                    state_owner_id not in manager.allocations
                    and state_owner_id not in manager.views
                ):
                    continue
                if uses_speculative_state:
                    assert speculative_state_ids is not None
                    assert temporary_ids is not None
                    group_context_tokens = max(
                        0, int(item.context_tokens)
                    ) + min(positions)
                    if uses_embedded_rollback_planes:
                        # llama.cpp's recurrent live buffer already owns the
                        # current row plus ``n_rs_seq`` rollback rows.  Reusing
                        # those planes avoids fabricating a separate full-state
                        # temporary allocation for every MTP cohort.
                        position = min(positions)
                        if access_phase == "read":
                            if read_source == "committed":
                                plane_index = 0
                                should_read = group_context_tokens > 0
                            elif read_source == "speculative":
                                plane_index = max(
                                    1,
                                    min(rollback_plane_count, position),
                                )
                                should_read = True
                            else:
                                raise ValueError(
                                    "unsupported linear_state_read_source: {}".format(
                                        read_source
                                    )
                                )
                            if should_read:
                                result = _state_read(
                                    state_owner_id,
                                    offset_bytes=plane_index * state_bytes,
                                    size_bytes=state_bytes,
                                )
                                self.residency_accesses += 1
                                self.residency_state_accesses += 1
                                row = dict(
                                    _state_audit(
                                        result,
                                        request_id=request_id,
                                        persistence=(
                                            "committed"
                                            if plane_index == 0
                                            else "embedded_rollback"
                                        ),
                                        operation="read",
                                    )
                                )
                                row["rollback_plane_index"] = plane_index
                                audit.append(row)
                        else:
                            plane_index = max(
                                1,
                                min(rollback_plane_count, position + 1),
                            )
                            result = _state_write(
                                state_owner_id,
                                offset_bytes=plane_index * state_bytes,
                                size_bytes=state_bytes,
                            )
                            self.residency_accesses += 1
                            self.residency_state_accesses += 1
                            row = dict(
                                _state_audit(
                                    result,
                                    request_id=request_id,
                                    persistence="embedded_rollback",
                                    operation="write",
                                )
                            )
                            row["rollback_plane_index"] = plane_index
                            audit.append(row)
                            if _commits_request(request_id, positions):
                                result = _state_write(
                                    state_owner_id,
                                    offset_bytes=0,
                                    size_bytes=state_bytes,
                                )
                                self.residency_accesses += 1
                                self.residency_state_accesses += 1
                                committed_row = dict(
                                    _state_audit(
                                        result,
                                        request_id=request_id,
                                        persistence="committed",
                                        operation="write",
                                    )
                                )
                                committed_row["rollback_plane_index"] = 0
                                audit.append(committed_row)
                        continue

                    temporary_id = speculative_state_ids.get(request_id)
                    if access_phase == "read":
                        if read_source == "committed":
                            if group_context_tokens > 0:
                                result = _state_read(state_owner_id)
                                self.residency_accesses += 1
                                self.residency_state_accesses += 1
                                audit.append(
                                    _state_audit(
                                        result,
                                        request_id=request_id,
                                        persistence="committed",
                                        operation="read",
                                    )
                                )
                            if temporary_id is None:
                                temporary_id = (
                                    "runtime.linear_state.speculative.{}.slot.{:04d}"
                                ).format(cohort.cohort_id, slot)
                                manager.register_temporary(
                                    temporary_id,
                                    state_bytes,
                                    kind="linear_state_speculative",
                                )
                                self._capture_residency_interval_peak()
                                speculative_state_ids[request_id] = temporary_id
                                temporary_ids.append(temporary_id)
                                self.residency_temporary_allocations += 1
                        elif read_source == "speculative":
                            if (
                                temporary_id is None
                                or temporary_id not in manager.allocations
                            ):
                                raise RuntimeError(
                                    "request {} has no rolling speculative "
                                    "linear state".format(request_id)
                                )
                            result = _state_read(temporary_id)
                            self.residency_accesses += 1
                            self.residency_state_accesses += 1
                            audit.append(
                                _state_audit(
                                    result,
                                    request_id=request_id,
                                    persistence="temporary_speculative",
                                    operation="read",
                                )
                            )
                        else:
                            raise ValueError(
                                "unsupported linear_state_read_source: {}".format(
                                    read_source
                                )
                            )
                    else:
                        if (
                            temporary_id is None
                            or temporary_id not in manager.allocations
                        ):
                            raise RuntimeError(
                                "request {} has no materialized speculative "
                                "linear state".format(request_id)
                            )
                        result = _state_write(temporary_id)
                        self.residency_accesses += 1
                        self.residency_state_accesses += 1
                        audit.append(
                            _state_audit(
                                result,
                                request_id=request_id,
                                persistence="temporary_speculative",
                                operation="write",
                            )
                        )
                        if _commits_request(request_id, positions):
                            result = _state_write(state_owner_id)
                            self.residency_accesses += 1
                            self.residency_state_accesses += 1
                            audit.append(
                                _state_audit(
                                    result,
                                    request_id=request_id,
                                    persistence="committed",
                                    operation="write",
                                )
                            )
                        if _releases_request(request_id):
                            manager.release(temporary_id)
                            self._capture_residency_interval_peak()
                            self.residency_temporary_releases += 1
                            speculative_state_ids.pop(request_id, None)
                    continue
                # The first prefill/recompute write initializes state; later
                # invocations read then update that same slot-local owner.
                group_context_tokens = max(0, int(item.context_tokens)) + min(
                    positions
                )
                if access_phase == "read" and group_context_tokens > 0:
                    result = _state_read(state_owner_id)
                    self.residency_accesses += 1
                    self.residency_state_accesses += 1
                    audit.append(
                        {
                            "target_id": state_owner_id,
                            "owner_id": result.owner_id,
                            "operation": "read",
                            "offset_bytes": 0,
                            "size_bytes": state_bytes,
                            "kind": "linear_state",
                            "request_id": request_id,
                            "operator_invocation_group_id": (
                                invocation_group.get("group_id")
                                if invocation_group is not None
                                else None
                            ),
                        }
                    )
                if access_phase == "write":
                    result = _state_write(state_owner_id)
                    self.residency_accesses += 1
                    self.residency_state_accesses += 1
                    audit.append(
                        {
                            "target_id": state_owner_id,
                            "owner_id": result.owner_id,
                            "operation": "write",
                            "offset_bytes": 0,
                            "size_bytes": state_bytes,
                            "kind": "linear_state",
                            "request_id": request_id,
                            "operator_invocation_group_id": (
                                invocation_group.get("group_id")
                                if invocation_group is not None
                                else None
                            ),
                        }
                    )
        return audit

    def _owner_migration_cost_details(
        self,
        migrations: Sequence[Migration],
        migration_group_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[
        Tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            Tuple[str, ...],
            int,
            int,
        ],
        Tuple[Mapping[str, object], ...],
    ]:
        """Return conserved transfer totals plus causal coalescing facts."""

        if migration_group_ids is None:
            causal_group_ids: Tuple[Optional[str], ...] = (None,) * len(
                migrations
            )
        else:
            causal_group_ids = tuple(migration_group_ids)
            if len(causal_group_ids) != len(migrations):
                raise ValueError(
                    "migration_group_ids must align one-to-one with migrations"
                )

        fault_batch_bytes = self.resource_policy.fault_batch_bytes
        grouped: List[Dict[str, object]] = []
        routes: List[str] = []
        coalesce_open = False
        for migration, raw_group_id in zip(migrations, causal_group_ids):
            group_id = None
            if raw_group_id is not None:
                group_id = str(raw_group_id)
                if not group_id:
                    raise ValueError(
                        "migration causal invocation-group id must be non-empty"
                    )
            clean_discard_service = None
            if migration.kind is MigrationKind.CLEAN_DISCARD:
                clean_discard_service = self._clean_eviction_service_by_allocation.get(
                    str(migration.allocation_id),
                    self._clean_eviction_service_default,
                )
            causal_group_key = (group_id,) if group_id is not None else ()
            causal_consumer_key = tuple(
                str(task_id) for task_id in migration.consumer_task_ids
            )
            key = (
                migration.kind,
                migration.source,
                migration.destination,
                # A direct discard has no transfer submission and must not be
                # coalesced with a backing migration merely because the
                # allocation route is otherwise identical.
                clean_discard_service
                if migration.kind is MigrationKind.CLEAN_DISCARD
                else None,
                # Released leases and invocation groups are the causal
                # boundary for later DAG placement.  Same-route migrations
                # may share transfer resources only while they share those
                # anchors; crossing them would erase required dependencies.
                causal_group_key,
                causal_consumer_key,
            )
            if coalesce_open and grouped and grouped[-1]["key"] == key:
                grouped[-1]["byte_count"] = int(
                    grouped[-1]["byte_count"]
                ) + int(migration.byte_count)
                grouped[-1]["granule_count"] = int(
                    grouped[-1]["granule_count"]
                ) + int(migration.granule_count)
                grouped[-1]["migration_count"] = int(
                    grouped[-1]["migration_count"]
                ) + 1
                allocation_ids = grouped[-1]["allocation_ids"]
                assert isinstance(allocation_ids, list)
                allocation_id = str(migration.allocation_id)
                if allocation_id not in allocation_ids:
                    allocation_ids.append(allocation_id)
                grouped[-1]["migration_sequence_end"] = int(
                    migration.sequence
                )
                covered = grouped[-1]["operator_invocation_group_ids"]
                assert isinstance(covered, list)
                if group_id is not None and group_id not in covered:
                    covered.append(group_id)
                consumer_task_ids = grouped[-1]["consumer_task_ids"]
                assert isinstance(consumer_task_ids, list)
                for task_id in migration.consumer_task_ids:
                    if task_id not in consumer_task_ids:
                        consumer_task_ids.append(task_id)
            else:
                grouped.append(
                    {
                        "key": key,
                        "byte_count": int(migration.byte_count),
                        "granule_count": int(migration.granule_count),
                        "migration_count": 1,
                        "allocation_ids": [str(migration.allocation_id)],
                        "migration_sequence_start": int(migration.sequence),
                        "migration_sequence_end": int(migration.sequence),
                        "operator_invocation_group_ids": (
                            [group_id] if group_id is not None else []
                        ),
                        "consumer_task_ids": list(
                            migration.consumer_task_ids
                        ),
                    }
                )
            coalesce_open = True

        page_in_ns = page_in_energy = 0.0
        writeback_ns = writeback_energy = 0.0
        clean_discard_ns = clean_discard_energy = 0.0
        fault_batch_count = 0
        clean_eviction_batch_count = 0
        service_batches: List[Mapping[str, object]] = []
        for batch_index, group in enumerate(grouped):
            (
                kind,
                source,
                destination,
                clean_discard_service,
                _causal_group_key,
                _causal_consumer_key,
            ) = group["key"]
            byte_count = int(group["byte_count"])
            direct_clean_discard = (
                kind is MigrationKind.CLEAN_DISCARD
                and clean_discard_service == _CLEAN_EVICTION_FREE_DISCARD
            )
            if direct_clean_discard:
                # The bytes still belong to the page-out accounting domain,
                # but free discard does not submit a DMA transfer or incur a
                # fault-latency charge.
                batches = 0
            elif (
                self.resource_policy.fault_latency_scope
                == "make_resident_batch"
            ):
                batches = 1
            elif fault_batch_bytes is None:
                batches = max(1, int(group["granule_count"]))
            else:
                batches = max(
                    1,
                    int(
                        math.ceil(
                            byte_count / float(max(1, fault_batch_bytes))
                        )
                    ),
                )
            if direct_clean_discard:
                transfer_ns, transfer_energy, route = 0.0, 0.0, "direct"
                resource_phases: Tuple[
                    Tuple[Mapping[str, object], ...], ...
                ] = ()
                route_component_ids = tuple(
                    dict.fromkeys((str(source), str(destination)))
                )
                submission_latency_ns = 0.0
            else:
                (
                    transfer_ns,
                    transfer_energy,
                    route,
                    resource_phases,
                    route_component_ids,
                    submission_latency_ns,
                ) = self._page_transfer_cost_details(
                    byte_count,
                    batches,
                    source,
                    destination,
                    residency_page_count=int(group["granule_count"]),
                )
            migration_sequence_start = int(group["migration_sequence_start"])
            migration_sequence_end = int(group["migration_sequence_end"])
            movement_id = "owner-residency-migration-batch{:04d}".format(
                batch_index
            )
            private_setup_envelope_resource_id = (
                "{}.batch{:04d}.migration{:08d}_{:08d}".format(
                    _OWNER_RESIDENCY_TRANSFER_SETUP_ENVELOPE_RESOURCE_ID,
                    batch_index,
                    migration_sequence_start,
                    migration_sequence_end,
                )
            )
            annotated_phases: List[Tuple[Mapping[str, object], ...]] = []
            service_domains: List[Mapping[str, object]] = []
            seen_bulk_domains = set()
            for phase_index, phase in enumerate(resource_phases):
                annotated_phase: List[Mapping[str, object]] = []
                for demand in phase:
                    resource_id = str(demand.get("resource_id", ""))
                    logical_resource_base_id: Optional[str] = None
                    if (
                        demand.get("logical_resource") is True
                        and resource_id
                        == _OWNER_RESIDENCY_TRANSFER_SETUP_ENVELOPE_RESOURCE_ID
                    ):
                        logical_resource_base_id = resource_id
                        resource_id = private_setup_envelope_resource_id
                    controller_phase = bool(demand.get("runtime_phase"))
                    service_role = (
                        "controller_owner"
                        if controller_phase
                        else "bulk_owner"
                    )
                    service_domain = resource_id
                    if service_role == "bulk_owner":
                        owner_key = (movement_id, service_domain)
                        if owner_key in seen_bulk_domains:
                            raise ValueError(
                                "migration movement has duplicate bulk service owner"
                            )
                        seen_bulk_domains.add(owner_key)
                    annotated = {
                        **dict(demand),
                        "resource_id": resource_id,
                        "movement_id": movement_id,
                        "service_domain": service_domain,
                        "service_role": service_role,
                        "bulk_service_owner": service_role == "bulk_owner",
                    }
                    if logical_resource_base_id is not None:
                        annotated["logical_resource_base_id"] = (
                            logical_resource_base_id
                        )
                        annotated["logical_resource_scope"] = (
                            "owner_residency_migration_batch"
                        )
                    annotated_phase.append(annotated)
                    service_domains.append(
                        {
                            "phase_index": phase_index,
                            "resource_id": resource_id,
                            "service_domain": service_domain,
                            "service_role": service_role,
                            "observed_bytes": max(
                                0, int(demand.get("bytes_moved", 0))
                            ),
                            "service_ns": max(
                                0.0, float(demand.get("service_ns", 0.0))
                            ),
                            "bulk_service_ns": (
                                max(
                                    0.0,
                                    float(demand.get("service_ns", 0.0)),
                                )
                                if service_role == "bulk_owner"
                                else 0.0
                            ),
                        }
                    )
                annotated_phases.append(tuple(annotated_phase))
            resource_phases = tuple(annotated_phases)
            if kind is MigrationKind.PAGE_IN:
                fault_batch_count += batches
                page_in_ns += transfer_ns
                page_in_energy += transfer_energy
                routes.append("page_in:{}".format(route))
            elif kind is MigrationKind.DIRTY_WRITEBACK:
                fault_batch_count += batches
                writeback_ns += transfer_ns
                writeback_energy += transfer_energy
                routes.append("dirty_writeback:{}".format(route))
            elif kind is MigrationKind.CLEAN_DISCARD:
                clean_eviction_batch_count += batches
                clean_discard_ns += transfer_ns
                clean_discard_energy += transfer_energy
                routes.append("clean_discard:{}".format(route))
            covered_group_ids = tuple(
                str(item)
                for item in group["operator_invocation_group_ids"]
            )
            service_batches.append(
                {
                    "batch_index": batch_index,
                    "movement_id": movement_id,
                    "kind": kind.value,
                    "source_component_id": source,
                    "destination_component_id": destination,
                    "byte_count": byte_count,
                    "granule_count": int(group["granule_count"]),
                    "coalesced_migration_count": int(
                        group["migration_count"]
                    ),
                    "allocation_ids": tuple(group["allocation_ids"]),
                    "migration_sequence_start": migration_sequence_start,
                    "migration_sequence_end": migration_sequence_end,
                    "submission_batch_count": batches,
                    "transfer_ns": transfer_ns,
                    "transfer_energy_pj": transfer_energy,
                    "route": route,
                    "route_component_ids": route_component_ids,
                    "resource_phases": resource_phases,
                    "service_domains": tuple(service_domains),
                    "bulk_service_owner_count": len(seen_bulk_domains),
                    "service_ownership": "exactly_once_per_movement_domain",
                    "resource_demands": (
                        resource_phases[0]
                        if len(resource_phases) == 1
                        else tuple(
                            demand
                            for phase in resource_phases
                            for demand in phase
                        )
                    ),
                    "submission_latency_ns": submission_latency_ns,
                    "operator_invocation_group_ids": covered_group_ids,
                    "consumer_task_ids": tuple(
                        str(item) for item in group["consumer_task_ids"]
                    ),
                    "earliest_operator_invocation_group_id": (
                        covered_group_ids[0] if covered_group_ids else None
                    ),
                    "causal_placement": (
                        "before_earliest_covered_invocation_group"
                        if covered_group_ids
                        else "cohort_tail_unattributed"
                    ),
                }
            )
        return (
            (
                page_in_ns,
                page_in_energy,
                writeback_ns,
                writeback_energy,
                clean_discard_ns,
                clean_discard_energy,
                tuple(routes),
                fault_batch_count,
                clean_eviction_batch_count,
            ),
            tuple(service_batches),
        )

    def _owner_migration_cost(
        self,
        migrations: Sequence[Migration],
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Tuple[str, ...],
        int,
        int,
    ]:
        """Return the legacy conserved migration-cost aggregate."""

        totals, _service_batches = self._owner_migration_cost_details(
            migrations
        )
        return totals

    def _apply_owner_residency(
        self,
        cohort: BatchCohort,
        cost: BatchCost,
    ) -> BatchCost:
        """Apply ordered owner accesses and add real migration service cost."""

        manager = self.residency_manager
        if manager is None or cohort.kind not in {
            "prefill",
            "decode",
            "mtp",
            "mixed",
            "recompute",
        }:
            return cost
        metadata = dict(cost.metadata)
        trace = self._owner_residency_trace(metadata)
        migration_start = self._residency_migration_cursor
        migration_group_spans: List[Tuple[int, int, str]] = []
        audit: List[Mapping[str, object]] = []
        temporary_ids: List[str] = []
        speculative_state_ids: Dict[str, str] = {}
        explicit_kv = any(
            str(item.get("kind", "")).strip().lower()
            in {"kv", "kv_cache"}
            for item in trace
        )
        explicit_state = any(
            str(item.get("kind", "")).strip().lower()
            in {"state", "linear_state", "recurrent_state"}
            for item in trace
        )

        def apply_trace_chunk(
            rows: Sequence[Mapping[str, Any]],
        ) -> None:
            if not rows:
                return
            traced_audit, traced_temporary, _kv, _state = (
                self._apply_traced_owner_accesses(
                    rows,
                    cohort_id=cohort.cohort_id,
                )
            )
            audit.extend(traced_audit)
            temporary_ids.extend(
                item for item in traced_temporary if item not in temporary_ids
            )
            # Traced workspace allocations are still resident here; capture
            # the interval peak before the caller's finally block releases
            # them.
            self._capture_residency_interval_peak()

        raw_proposer_groups = metadata.get(
            "mtp_proposer_invocation_groups", ()
        )
        if raw_proposer_groups is None:
            raw_proposer_groups = ()
        if isinstance(
            raw_proposer_groups, (str, bytes, _ABCMapping)
        ) or not isinstance(raw_proposer_groups, _ABCSequence):
            raise ValueError(
                "mtp_proposer_invocation_groups must be an ordered sequence"
            )
        proposer_groups: List[Mapping[str, Any]] = []
        known_group_ids = set()
        for index, raw_group in enumerate(raw_proposer_groups):
            if not isinstance(raw_group, _ABCMapping):
                raise ValueError(
                    "mtp_proposer_invocation_groups[{}] must be a mapping"
                    .format(index)
                )
            group_id = str(raw_group.get("group_id", ""))
            if not group_id or group_id in known_group_ids:
                raise ValueError(
                    "operator invocation group ids must be non-empty and unique"
                )
            known_group_ids.add(group_id)
            proposer_groups.append(raw_group)

        raw_groups = metadata.get("operator_invocation_groups", ())
        if raw_groups is None:
            raw_groups = ()
        if isinstance(raw_groups, (str, bytes, _ABCMapping)) or not isinstance(
            raw_groups, _ABCSequence
        ):
            raise ValueError(
                "operator_invocation_groups must be an ordered sequence"
            )
        invocation_groups: List[Mapping[str, Any]] = []
        for index, raw_group in enumerate(raw_groups):
            if not isinstance(raw_group, _ABCMapping):
                raise ValueError(
                    "operator_invocation_groups[{}] must be a mapping".format(
                        index
                    )
                )
            group_id = str(raw_group.get("group_id", ""))
            if not group_id or group_id in known_group_ids:
                raise ValueError(
                    "operator invocation group ids must be non-empty and unique"
                )
            known_group_ids.add(group_id)
            invocation_groups.append(raw_group)
        raw_catchup_groups = metadata.get(
            "mtp_draft_catchup_invocation_groups", ()
        )
        if raw_catchup_groups is None:
            raw_catchup_groups = ()
        if isinstance(
            raw_catchup_groups, (str, bytes, _ABCMapping)
        ) or not isinstance(raw_catchup_groups, _ABCSequence):
            raise ValueError(
                "mtp_draft_catchup_invocation_groups must be an ordered sequence"
            )
        catchup_groups: List[Mapping[str, Any]] = []
        for index, raw_group in enumerate(raw_catchup_groups):
            if not isinstance(raw_group, _ABCMapping):
                raise ValueError(
                    "mtp_draft_catchup_invocation_groups[{}] must be a mapping"
                    .format(index)
                )
            group_id = str(raw_group.get("group_id", ""))
            if not group_id or group_id in known_group_ids:
                raise ValueError(
                    "operator invocation group ids must be non-empty and unique"
                )
            known_group_ids.add(group_id)
            catchup_groups.append(raw_group)
        expected_catchup_offsets: Dict[str, int] = {}
        catchup_item_tokens: Dict[str, int] = {}
        for item in cohort.items:
            item_tokens = max(0, int(item.token_count))
            if item_tokens <= 0:
                continue
            request_id = str(item.request_id)
            catchup_item_tokens[request_id] = (
                catchup_item_tokens.get(request_id, 0) + item_tokens
            )
        has_physical_catchup = False
        logical_catchup_group_count = 0
        for group in catchup_groups:
            declares_physical_catchup = any(
                key in group
                for key in (
                    "physical_chunk_index",
                    "physical_row_start",
                    "physical_ubatch_rows",
                    "token_count_by_request",
                    "token_offset_by_request",
                )
            )
            if not declares_physical_catchup:
                logical_catchup_group_count += 1
                continue
            has_physical_catchup = True
            raw_counts = group.get("token_count_by_request")
            raw_offsets = group.get("token_offset_by_request")
            count_rows = (
                tuple(raw_counts.items())
                if isinstance(raw_counts, _ABCMapping)
                else raw_counts
            )
            offset_rows = (
                tuple(raw_offsets.items())
                if isinstance(raw_offsets, _ABCMapping)
                else raw_offsets
            )
            if (
                not isinstance(count_rows, _ABCSequence)
                or isinstance(count_rows, (str, bytes))
                or not isinstance(offset_rows, _ABCSequence)
                or isinstance(offset_rows, (str, bytes))
            ):
                raise ValueError(
                    "physical MTP catchup requires count and offset maps"
                )
            try:
                counts = {str(key): int(value) for key, value in count_rows}
                offsets = {str(key): int(value) for key, value in offset_rows}
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "physical MTP catchup maps require integer values"
                )
            raw_request_ids = group.get("request_ids", ())
            group_request_ids = (
                {str(item) for item in raw_request_ids}
                if isinstance(raw_request_ids, _ABCSequence)
                and not isinstance(raw_request_ids, (str, bytes))
                else set()
            )
            if set(counts) != group_request_ids or set(offsets) != group_request_ids:
                raise ValueError(
                    "physical MTP catchup maps must match request_ids"
                )
            if not group_request_ids or any(value <= 0 for value in counts.values()):
                raise ValueError(
                    "physical MTP catchup counts must be positive"
                )
            if any(value < 0 for value in offsets.values()):
                raise ValueError(
                    "physical MTP catchup offsets must be non-negative"
                )
            try:
                lane_count = int(group["lane_count"])
                physical_rows = int(group["physical_ubatch_rows"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "physical MTP catchup requires integer row counts"
                ) from exc
            if sum(counts.values()) != lane_count or lane_count != physical_rows:
                raise ValueError(
                    "physical MTP catchup row counts must agree"
                )
            for request_id in counts:
                expected_offset = expected_catchup_offsets.get(request_id, 0)
                if offsets[request_id] != expected_offset:
                    raise ValueError(
                        "physical MTP catchup request spans must be contiguous"
                    )
                expected_catchup_offsets[request_id] = (
                    offsets[request_id] + counts[request_id]
                )
                if (
                    request_id not in catchup_item_tokens
                    or expected_catchup_offsets[request_id]
                    > catchup_item_tokens[request_id]
                ):
                    raise ValueError(
                        "physical MTP catchup request span exceeds item tokens"
                    )
        if logical_catchup_group_count and (
            has_physical_catchup or logical_catchup_group_count != 1
        ):
            raise ValueError(
                "multiple MTP catchup groups require physical chunk metadata"
            )
        if has_physical_catchup and expected_catchup_offsets != catchup_item_tokens:
            raise ValueError(
                "physical MTP catchup groups must cover all item tokens"
            )
        manager.begin_deferred_validation()
        try:
            if proposer_groups or invocation_groups or catchup_groups:
                declared_groups = [
                    *proposer_groups,
                    *invocation_groups,
                    *catchup_groups,
                ]
                rows_by_group: Dict[str, List[Mapping[str, Any]]] = {
                    str(group["group_id"]): []
                    for group in declared_groups
                }
                before_group: Dict[str, List[Mapping[str, Any]]] = {}
                pending_ungrouped: List[Mapping[str, Any]] = []
                first_seen_groups = set()
                unknown_groups: List[Mapping[str, Any]] = []
                for row in trace:
                    raw_group_id = row.get("operator_invocation_group_id")
                    if raw_group_id is None:
                        pending_ungrouped.append(row)
                        continue
                    group_id = str(raw_group_id)
                    if group_id not in rows_by_group:
                        request_ids = row.get("request_ids", ())
                        synthetic_group: Mapping[str, Any] = {
                            "group_id": group_id,
                            "residency_role": "trace_only",
                            "request_ids": (
                                tuple(str(item) for item in request_ids)
                                if isinstance(request_ids, _ABCSequence)
                                and not isinstance(request_ids, (str, bytes))
                                else (
                                    (str(row["request_id"]),)
                                    if row.get("request_id") is not None
                                    else ()
                                )
                            ),
                        }
                        rows_by_group[group_id] = []
                        unknown_groups.append(synthetic_group)
                    if group_id not in first_seen_groups:
                        before_group[group_id] = list(pending_ungrouped)
                        pending_ungrouped.clear()
                        first_seen_groups.add(group_id)
                    rows_by_group[group_id].append(row)
                ordered_groups = [*declared_groups, *unknown_groups]
                for group in ordered_groups:
                    group_id = str(group["group_id"])
                    group_migration_start = manager.migration_count
                    apply_trace_chunk(before_group.get(group_id, ()))
                    residency_role = str(
                        group.get("residency_role", "target_context")
                    ).strip().lower()
                    if residency_role == "mtp_draft_context":
                        audit.extend(
                            self._apply_mtp_draft_kv_accesses(
                                cohort,
                                access_phase="read",
                                invocation_group=group,
                            )
                        )
                        self._capture_residency_interval_peak()
                        apply_trace_chunk(rows_by_group.get(group_id, ()))
                        audit.extend(
                            self._apply_mtp_draft_kv_accesses(
                                cohort,
                                access_phase="write",
                                invocation_group=group,
                            )
                        )
                        self._capture_residency_interval_peak()
                    elif residency_role == "mtp_draft_context_catchup":
                        audit.extend(
                            self._apply_mtp_draft_kv_accesses(
                                cohort,
                                access_phase="read",
                                invocation_group=group,
                            )
                        )
                        self._capture_residency_interval_peak()
                        apply_trace_chunk(rows_by_group.get(group_id, ()))
                        audit.extend(
                            self._apply_mtp_draft_kv_accesses(
                                cohort,
                                access_phase="write",
                                invocation_group=group,
                            )
                        )
                        self._capture_residency_interval_peak()
                    elif residency_role == "trace_only":
                        apply_trace_chunk(rows_by_group.get(group_id, ()))
                    else:
                        audit.extend(
                            self._apply_slot_owner_accesses(
                                cohort,
                                include_kv=not explicit_kv,
                                include_state=not explicit_state,
                                access_phase="read",
                                invocation_group=group,
                                speculative_state_ids=speculative_state_ids,
                                temporary_ids=temporary_ids,
                            )
                        )
                        self._capture_residency_interval_peak()
                        apply_trace_chunk(rows_by_group.get(group_id, ()))
                        audit.extend(
                            self._apply_slot_owner_accesses(
                                cohort,
                                include_kv=not explicit_kv,
                                include_state=not explicit_state,
                                access_phase="write",
                                invocation_group=group,
                                speculative_state_ids=speculative_state_ids,
                                temporary_ids=temporary_ids,
                            )
                        )
                        self._capture_residency_interval_peak()
                    # Residency mutations are planned group-by-group.  The
                    # range stays non-evictable while this consumer is being
                    # assembled, then its released lease becomes the causal
                    # dependency of any later PAGE_OUT for the same bytes.
                    self._release_residency_group_leases()
                    group_migration_end = manager.migration_count
                    if group_migration_end > group_migration_start:
                        migration_group_spans.append(
                            (
                                group_migration_start,
                                group_migration_end,
                                group_id,
                            )
                        )
                apply_trace_chunk(pending_ungrouped)
            else:
                # Backward-compatible minimum ordering for lowerers that do
                # not yet expose backend invocation groups.
                audit.extend(
                    self._apply_slot_owner_accesses(
                        cohort,
                        include_kv=not explicit_kv,
                        include_state=not explicit_state,
                        access_phase="read",
                    )
                )
                self._capture_residency_interval_peak()
                apply_trace_chunk(trace)
                audit.extend(
                    self._apply_slot_owner_accesses(
                        cohort,
                        include_kv=not explicit_kv,
                        include_state=not explicit_state,
                        access_phase="write",
                    )
                )
                self._capture_residency_interval_peak()
        finally:
            try:
                self._release_residency_group_leases()
                # Workspace/activation allocations are live for this cohort
                # only. Reverse release mirrors nested allocation order.
                self._capture_residency_interval_peak()
                for temporary_id in reversed(temporary_ids):
                    if temporary_id in manager.allocations:
                        manager.release(temporary_id)
                        self.residency_temporary_releases += 1
            finally:
                # Thousands of tensor/view touches may occur in one cohort.
                # Validate the whole physical registry once at this atomic
                # boundary instead of rescanning every owner/view per touch.
                manager.end_deferred_validation()
        new_migrations = manager.migrations_since(migration_start)
        self._residency_migration_cursor = manager.migration_count
        migration_group_ids: List[Optional[str]] = [
            None
        ] * len(new_migrations)
        for span_start, span_end, group_id in migration_group_spans:
            local_start = max(0, span_start - migration_start)
            local_end = min(len(new_migrations), span_end - migration_start)
            for index in range(local_start, local_end):
                migration_group_ids[index] = group_id
        page_in_bytes = sum(item.page_in_bytes for item in new_migrations)
        page_out_bytes = sum(item.page_out_bytes for item in new_migrations)
        clean_discard_bytes = sum(
            item.clean_discard_bytes for item in new_migrations
        )
        dirty_writeback_bytes = sum(
            item.dirty_writeback_bytes for item in new_migrations
        )
        released_staged_weight_ids = tuple(
            allocation_id
            for allocation_id in self._staged_weight_clean_discard_ids
            if allocation_id not in manager.allocations
        )
        try:
            migration_totals, migration_service_batches = (
                self._owner_migration_cost_details(
                    new_migrations,
                    migration_group_ids,
                )
            )
            (
                page_in_ns,
                page_in_energy,
                writeback_ns,
                writeback_energy,
                clean_discard_ns,
                clean_discard_energy,
                routes,
                fault_batch_count,
                clean_eviction_batch_count,
            ) = migration_totals
        finally:
            for allocation_id in released_staged_weight_ids:
                self._clean_eviction_service_by_allocation.pop(
                    allocation_id, None
                )
                self._staged_weight_clean_discard_ids.discard(allocation_id)
        migration_ns = page_in_ns + writeback_ns + clean_discard_ns
        migration_energy = (
            page_in_energy + writeback_energy + clean_discard_energy
        )
        self.residency_page_in_bytes += page_in_bytes
        self.residency_page_out_bytes += page_out_bytes
        self.residency_clean_discard_bytes += clean_discard_bytes
        self.residency_dirty_writeback_bytes += dirty_writeback_bytes
        self.residency_clean_discard_time_ns += clean_discard_ns
        self.residency_clean_discard_energy_pj += clean_discard_energy
        self.residency_transfer_time_ns += migration_ns
        self.residency_transfer_energy_pj += migration_energy
        self.residency_fault_batches += fault_batch_count
        self.residency_clean_eviction_batches += clean_eviction_batch_count
        snapshot = manager.snapshot()
        base_device_ns = max(
            0.0,
            float(
                metadata.get(
                    "device_execution_ns",
                    cost.duration_ns
                    - max(
                        0.0,
                        float(metadata.get("host_orchestration_ns", 0.0)),
                    ),
                )
            ),
        )
        metadata.update(
            {
                "owner_residency_enabled": True,
                "owner_residency_component_id": snapshot.component_id,
                "owner_residency_capacity_bytes": snapshot.capacity_bytes,
                "owner_residency_committed_bytes": snapshot.committed_bytes,
                "owner_residency_resident_bytes": snapshot.resident_bytes,
                "owner_residency_peak_resident_bytes": (
                    snapshot.peak_resident_bytes
                ),
                "owner_residency_available_bytes": snapshot.available_bytes,
                "owner_residency_allocation_count": snapshot.allocation_count,
                "owner_residency_view_count": snapshot.view_count,
                "owner_residency_access_count": len(audit),
                "owner_residency_accesses": tuple(audit),
                "owner_residency_migration_event_count": len(new_migrations),
                "owner_residency_fault_batch_count": fault_batch_count,
                "owner_residency_clean_eviction_batch_count": (
                    clean_eviction_batch_count
                ),
                "owner_residency_page_in_bytes": page_in_bytes,
                "owner_residency_page_out_bytes": page_out_bytes,
                "owner_residency_clean_discard_bytes": clean_discard_bytes,
                "owner_residency_dirty_writeback_bytes": (
                    dirty_writeback_bytes
                ),
                "owner_residency_page_in_ns": page_in_ns,
                "owner_residency_page_in_energy_pj": page_in_energy,
                "owner_residency_dirty_writeback_ns": writeback_ns,
                "owner_residency_dirty_writeback_energy_pj": (
                    writeback_energy
                ),
                "owner_residency_clean_discard_ns": clean_discard_ns,
                "owner_residency_clean_discard_energy_pj": (
                    clean_discard_energy
                ),
                "owner_residency_transfer_ns": migration_ns,
                "owner_residency_transfer_energy_pj": migration_energy,
                "owner_residency_transfer_routes": routes,
                "owner_residency_transfer_batches": (
                    migration_service_batches
                ),
                "owner_residency_transfer_placement_policy": (
                    "before_earliest_covered_invocation_group"
                ),
                "owner_residency_timing_completeness": (
                    self.resource_policy.timing_completeness
                ),
                "owner_residency_submission_latency_known": (
                    self.resource_policy.page_fault_latency_known
                ),
                "owner_residency_unmodeled_timing_terms": (
                    ()
                    if self.resource_policy.page_fault_latency_known
                    else ("make_resident_submission_latency",)
                ),
                "owner_residency_page_in_bytes_total": (
                    self.residency_page_in_bytes
                ),
                "owner_residency_page_out_bytes_total": (
                    self.residency_page_out_bytes
                ),
                "owner_residency_clean_discard_bytes_total": (
                    self.residency_clean_discard_bytes
                ),
                "owner_residency_dirty_writeback_bytes_total": (
                    self.residency_dirty_writeback_bytes
                ),
                "owner_residency_initialization_migration_event_count": (
                    len(self._residency_initialization_migrations)
                ),
                "owner_residency_initialization_page_in_bytes": (
                    self.residency_initialization_page_in_bytes
                ),
                "owner_residency_initialization_page_out_bytes": (
                    self.residency_initialization_page_out_bytes
                ),
                "owner_residency_fault_batch_count_total": (
                    self.residency_fault_batches
                ),
                "owner_residency_clean_eviction_batch_count_total": (
                    self.residency_clean_eviction_batches
                ),
                "resource_contention_page_transfer_ns": (
                    max(
                        0.0,
                        float(
                            metadata.get(
                                "resource_contention_page_transfer_ns", 0.0
                            )
                        ),
                    )
                    + migration_ns
                ),
                "resource_contention_page_transfer_energy_pj": (
                    max(
                        0.0,
                        float(
                            metadata.get(
                                "resource_contention_page_transfer_energy_pj",
                                0.0,
                            )
                        ),
                    )
                    + migration_energy
                ),
                "resource_contention_page_in_ns": (
                    max(
                        0.0,
                        float(
                            metadata.get(
                                "resource_contention_page_in_ns", 0.0
                            )
                        ),
                    )
                    + page_in_ns
                ),
                "resource_contention_page_out_ns": (
                    max(
                        0.0,
                        float(
                            metadata.get(
                                "resource_contention_page_out_ns", 0.0
                            )
                        ),
                    )
                    + writeback_ns
                    + clean_discard_ns
                ),
                "device_execution_ns": base_device_ns + migration_ns,
            }
        )
        return BatchCost(
            duration_ns=max(1.0e-9, cost.duration_ns + migration_ns),
            energy_pj=max(0.0, cost.energy_pj + migration_energy),
            metadata=metadata,
        )

    def _assign_residency_slot(self, request_id: str) -> None:
        if self.residency_manager is None:
            return
        request = str(request_id)
        if request in self._request_residency_slots:
            return
        if not self._free_residency_slots:
            raise RuntimeError("owner-aware residency slot pool is exhausted")
        slot = self._free_residency_slots.pop(0)
        self._request_residency_slots[request] = slot

    def _release_residency_slot(self, request_id: str) -> None:
        request = str(request_id)
        slot = self._request_residency_slots.pop(request, None)
        if slot is None:
            return
        self._free_residency_slots.append(slot)
        self._free_residency_slots.sort()

    def _set_status(
        self,
        state: _MutableRequest,
        status: RequestStatus,
    ) -> None:
        if state.status == status:
            return
        if status == RequestStatus.RUNNING:
            self._assign_residency_slot(state.spec.request_id)
        if state.status == RequestStatus.RUNNING:
            self._running_count -= 1
        elif state.status == RequestStatus.FINISHED:
            self._finished_count -= 1
            self._terminal_count -= 1
        elif state.status == RequestStatus.REJECTED:
            self._rejected_count -= 1
            self._terminal_count -= 1
        state.status = status
        if status == RequestStatus.RUNNING:
            self._running_count += 1
        elif status == RequestStatus.FINISHED:
            self._finished_count += 1
            self._terminal_count += 1
        elif status == RequestStatus.REJECTED:
            self._rejected_count += 1
            self._terminal_count += 1
        if status in _TERMINAL_STATUSES:
            self._release_residency_slot(state.spec.request_id)

    def _append_batch(self, batch: ServingBatch) -> None:
        self.batches.append(batch)
        self._batch_kind_counts[batch.kind] = (
            self._batch_kind_counts.get(batch.kind, 0) + 1
        )
        phase_families = {
            "prefill" if item.phase == "recompute" else item.phase
            for item in batch.items
        }
        for phase in phase_families:
            self._batch_phase_counts[phase] = (
                self._batch_phase_counts.get(phase, 0) + 1
            )
        self._max_batch_sequences = max(
            self._max_batch_sequences, len(batch.request_ids)
        )
        self._max_batch_tokens = max(self._max_batch_tokens, batch.token_count)

    def run(self) -> ServingResult:
        self.execution_control.raise_if_cancelled()
        self.execution_control.report(
            "serving_cohorts",
            0,
            message="online serving started",
            simulated_time_ns=self.now,
            metadata={"request_count": self._request_count},
        )
        self._reject_impossible_requests()
        safety_limit = max(10_000, sum(state.spec.prompt_tokens + state.spec.output_tokens + 1 for state in self._state_values) * 32)
        while not self._terminal():
            self.execution_control.raise_if_cancelled()
            self.rounds += 1
            if self.rounds > safety_limit:
                raise RuntimeError("online scheduler made no progress")
            self._stabilize_boundary()
            cohort = self._select_cohort()
            if cohort is not None:
                self._execute(cohort)
                self.execution_control.report(
                    "serving_cohorts",
                    len(self.batches),
                    simulated_time_ns=self.now,
                    metadata={
                        "cohort_kind": cohort.kind,
                        "finished_requests": self._finished_count,
                        "request_count": self._request_count,
                    },
                )
                continue
            next_arrival = self._next_arrival()
            if next_arrival is not None and next_arrival > self.now:
                self.idle_ns += next_arrival - self.now
                self.now = next_arrival
                continue
            if self._recover_stalled():
                continue
            break
        self.execution_control.raise_if_cancelled()
        result = self._result()
        self.execution_control.report(
            "serving_complete",
            len(self.batches),
            len(self.batches),
            message="online serving completed",
            simulated_time_ns=self.now,
        )
        return result

    def _stabilize_boundary(self) -> None:
        """Process arrivals/admission until any boundary I/O has completed.

        Swap transfers advance simulated time.  Arrivals that occur during
        them must be visible before the next compute cohort is selected.
        """

        limit = max(4, self._request_count * 4)
        for _ in range(limit):
            before = self.now
            self._arrive()
            self._priority_preempt()
            self._fill_slots()
            if self.now == before:
                return
        raise RuntimeError("online scheduler could not stabilize a boundary")

    def _terminal(self) -> bool:
        return self._terminal_count >= self._request_count

    def _reject_impossible_requests(self) -> None:
        for request in self.plan.requests:
            state = self.states[request.request_id]
            reason = _request_admission_reason(self.plan, request)
            if reason is not None:
                self._set_status(state, RequestStatus.REJECTED)
                state.finished_ns = request.arrival_ns
                state.rejection_reason = reason
                self.events.append(
                    ServingEvent(request.arrival_ns, "request_rejected", request.request_id, details={"reason": reason})
                )

    def _arrive(self) -> None:
        pending_count = len(self.pending)
        while self._pending_index < pending_count:
            request = self.pending[self._pending_index]
            if request.arrival_ns > self.now:
                return
            self._pending_index += 1
            state = self.states[request.request_id]
            if state.status == RequestStatus.REJECTED:
                continue
            self._set_status(state, RequestStatus.WAITING)
            state.queued_since_ns = request.arrival_ns
            self.events.append(ServingEvent(request.arrival_ns, "request_arrival", request.request_id))
            if request.prompt_tokens == 0 and request.output_tokens == 0:
                self._finish(state, request.arrival_ns)

    def _next_arrival(self) -> Optional[float]:
        if self._pending_index >= len(self.pending):
            return None
        return self.pending[self._pending_index].arrival_ns

    def _active(self) -> List[_MutableRequest]:
        return [state for state in self._state_values if state.status == RequestStatus.RUNNING]

    def _rank(self, state: _MutableRequest) -> Tuple[float, float, float, str]:
        aging = 0
        if self.plan.scheduler.policy.strip().lower() == "decode_first_aging":
            age = max(0.0, self.now - state.queued_since_ns)
            aging = math.floor(age / self.plan.scheduler.starvation_ns)
        deadline = state.spec.deadline_ns if state.spec.deadline_ns is not None else math.inf
        return (-(state.spec.priority + aging), deadline, state.spec.arrival_ns, state.spec.request_id)

    def _service_rank(
        self, state: _MutableRequest
    ) -> Tuple[float, float, float, int, str]:
        """Rank a phase candidate with the configured deterministic order.

        Priority, deadline, arrival, and (when enabled) policy aging retain
        their existing meaning.  By default the service sequence resolves the
        final request-id tie for candidates in the same phase, preventing a
        lexicographically early request from consuming every prefill chunk or
        decode unit.  ``stable_admission`` instead preserves the first
        successful admission order across later service and resume boundaries.
        Request id remains the ultimate deterministic fallback for requests
        that have not yet been admitted.
        """

        priority, deadline, arrival, request_id = self._rank(state)
        if (
            self.plan.scheduler.phase_candidate_order.strip().lower()
            == "stable_admission"
        ):
            candidate_order = (
                self._request_count
                if state.admission_sequence is None
                else int(state.admission_sequence)
            )
        else:
            candidate_order = int(state.last_service_sequence)
        return (
            priority,
            deadline,
            arrival,
            candidate_order,
            request_id,
        )

    def _is_starved(self, state: _MutableRequest) -> bool:
        """Return whether a runnable non-decode phase has waited too long."""

        return (
            max(0.0, self.now - state.queued_since_ns)
            >= self.plan.scheduler.starvation_ns
        )

    def _fill_slots(self) -> None:
        while self._running_count < self.plan.scheduler.max_num_seqs:
            candidates = [
                state for state in self._state_values
                if state.status in (RequestStatus.WAITING, RequestStatus.SWAPPED)
            ]
            if not candidates:
                return
            # llama.cpp serializes idle prompt saves before it makes a later
            # waiting slot runnable; do this before any candidate reservation.
            before_save_ns = self.now
            self._save_pending_prompt_cache_states()
            if self.now > before_save_ns:
                # Requests may arrive while the synchronous save is running.
                # Return to the boundary loop so it can admit those arrivals
                # and rebuild the priority ordering before consuming a slot.
                return
            candidates.sort(key=self._service_rank)
            admitted = False
            for state in candidates:
                if not self._eager_slot_available(1):
                    # Eager allocation commits a full logical slot even when
                    # only a prefix has been touched.  Host backing may make
                    # that commitment legal without turning the untouched
                    # address range into resident GPU pages.
                    continue
                prior_pages = state.kv_pages
                if state.status == RequestStatus.SWAPPED and state.preemption_strategy == "swap":
                    if not self._resume_swap(state):
                        continue
                elif self.plan.kv_policy.allocation_policy == "eager":
                    target_pages = self.ledger.pages_for_tokens(
                        _max_live_kv_tokens(
                            state.spec.prompt_tokens, state.spec.output_tokens
                        )
                    )
                    if not self._reserve_with_pressure(state, target_pages):
                        continue
                if not self.prompt_cache.ensure_capacity(
                    max(
                        0,
                        int(self.plan.linear_state_policy.bytes_per_request)
                        if not state.linear_state_resident
                        else 0,
                    )
                ):
                    if state.kv_pages != prior_pages:
                        self.ledger.resize(state, prior_pages)
                    continue
                if not self.state_ledger.restore(state):
                    self.ledger.resize(state, prior_pages)
                    continue
                if self.prompt_cache.policy.save_implementation == "llama_cpp_host_tensor_get_combined":
                    if self._running_count == 0:
                        self._prompt_cache_wave_requests.clear()
                    elif any(self.states[key].status == RequestStatus.FINISHED
                             for key in self._prompt_cache_wave_requests):
                        # seq_rm can rewind the pool head into a freed slot,
                        # even before capacity wrap. No allocator is modeled.
                        for key in self._prompt_cache_wave_requests | {state.spec.request_id}:
                            self.states[key].kv_cache_range_error = "unknown_physical_slot_reuse"
                    self._prompt_cache_wave_requests.add(state.spec.request_id)
                if self._kv_scan_enabled:
                    if self._running_count == 0:
                        self._kv_scan_wave_requests.clear()
                    elif any(self.states[key].status == RequestStatus.FINISHED
                             for key in self._kv_scan_wave_requests):
                        raise ValueError("KV scan lower bound does not support slot reuse inside a wave")
                    self._kv_scan_wave_requests.add(state.spec.request_id)
                self._set_status(state, RequestStatus.RUNNING)
                if state.admission_sequence is None:
                    state.admission_sequence = self._admission_sequence
                    self._admission_sequence += 1
                state.queued_since_ns = self.now
                event_type = "request_admitted" if state.preemption_strategy is None else "request_resumed"
                self.events.append(ServingEvent(self.now, event_type, state.spec.request_id))
                state.preemption_strategy = None
                admitted = True
                break
            if not admitted:
                return

    def _prompt_cache_save_enabled(self) -> bool:
        return (
            self.prompt_cache.enabled
            and self.prompt_cache.policy.retain_completed
            and self.prompt_cache.policy.save_implementation
            in {"llama_cpp_host_tensor_get", "llama_cpp_host_tensor_get_combined"}
        )

    def _prompt_cache_gpu_attention_layer_count(self) -> int:
        scenario = self.plan.scenario
        gpu_id = scenario.host_orchestration_profile.gpu_component_id
        mapping = scenario.placement.op_to_component
        return sum(
            1
            for layer in _execution_layers(scenario)
            if not layer.is_linear_attention
            and mapping.get("{}.attention".format(layer.layer_id)) == gpu_id
        )

    def _prompt_cache_payload_service(
        self, state: _MutableRequest, details: Optional[Dict[str, object]] = None
    ) -> Tuple[Optional[int], Optional[float], str]:
        """Price known owner partitions; incomplete service remains explicit."""

        scenario = self.plan.scenario
        if self.plan.mtp.enabled:
            return None, None, "unsupported_speculative_state"
        if not scenario.fusion_policy.flash_attention:
            return None, None, "unsupported_value_layout"
        parallel = _parallel_plan(scenario)
        if parallel.world_size != 1 or len(parallel.ranks) != 1:
            return None, None, "unsupported_parallelism"
        gpu_id = scenario.host_orchestration_profile.gpu_component_id
        cpu_id = scenario.host_orchestration_profile.cpu_component_id
        layers = tuple(
            layer for layer in _execution_layers(scenario)
            if not layer.is_linear_attention
        )
        tokens = max(0, int(state.prefill_cursor)) + max(0, int(state.committed) - 1)
        if not layers or tokens <= 0:
            return 0, 0.0, "no_attention_state"
        rank = parallel.ranks[0]
        source = rank.memory_component_id
        target = self.prompt_cache.offload_component or self.prompt_cache.component
        if not target or _kind(scenario.hardware.get_component(target)) != "host_memory":
            return None, None, "unsupported_host_prompt_destination"
        owner_layers: Dict[str, list] = {"gpu": [], "cpu": []}
        for layer in layers:
            component = scenario.placement.op_to_component.get(layer.layer_id + ".attention")
            if component not in {gpu_id, cpu_id}:
                return None, None, "unsupported_attention_owner"
            owner_layers["gpu" if component == gpu_id else "cpu"].append(layer)
        contract = self._kv_allocation_contract
        cpu_owner_unknown = bool(owner_layers["gpu"] and owner_layers["cpu"]
                                 and not contract and self.plan.kv_policy.offload_ratio == 0.0)
        if cpu_owner_unknown:
            # Preserve the already supported local GPU subset. Placement of
            # computation alone does not declare the CPU cache memory owner.
            layers = tuple(owner_layers["gpu"])
            owner_layers["cpu"] = []
        row_bytes: Counter = Counter()
        reads: Dict[str, Counter] = {"gpu": Counter(), "cpu": Counter()}
        ranges = Counter(state.kv_cache_range_tokens)
        valid_ranges = (
            bool(ranges) and all(length > 0 for length in ranges)
            and sum(state.kv_cache_range_tokens) == tokens
            and len(state.kv_cache_range_tokens) == state.kv_cache_range_count
        )
        for layer in layers:
            try:
                bits, artifact = _kv_dtype_bits(scenario, layer)
            except ValueError:
                return None, None, "unsupported_kv_layout"
            if artifact is None:
                if bits != 16:
                    return None, None, "unsupported_kv_layout"
            elif artifact.name != "Q4_0":
                return None, None, "unsupported_kv_layout"
            elif (layer.effective_kv_heads * layer.effective_attention_head_dim) % artifact.block_size:
                return None, None, "unaligned_quantized_kv_row"
            owner = "gpu" if layer in owner_layers["gpu"] else "cpu"
            row_bytes[owner] += 2 * _kv_tensor_bytes(scenario, layer, 1, 1)
            if valid_ranges:
                for length, count in ranges.items():
                    reads[owner][_kv_tensor_bytes(scenario, layer, 1, length)] += 2 * count
        partitioned = False
        cpu_source = None
        if owner_layers["gpu"]:
            if (rank.component_id != gpu_id or not source
                    or self.plan.kv_policy.cache_component != source
                    or scenario.placement.tensor_to_component.get("kv_cache", source) != source
                    or _kind(scenario.hardware.get_component(source)) != "hbm"):
                return None, None, "unsupported_nonlocal_or_offloaded_kv"
            if owner_layers["cpu"] or self.plan.kv_policy.offload_ratio != 0.0:
                cells = contract.get("total_cells")
                partitioned = bool(
                    self.prompt_cache.policy.unified_kv
                    and contract.get("context_owner") == "target"
                    and contract.get("partition_scope") == "gpu_placed_target_attention_layers"
                    and contract.get("allocation_scope") == "global_live_pool"
                    and contract.get("shared_pool") is True
                    and contract.get("unified_streams") is True
                    and contract.get("cell_geometry_exact") is True
                    and contract.get("stream_count") == 1
                    and isinstance(cells, int) and not isinstance(cells, bool) and cells > 0
                    and tokens <= cells
                    and contract.get("component_id") == source
                    and contract.get("cells_per_stream", cells) == cells
                    and contract.get("n_stream", 1) == 1
                    and contract.get("device_bytes_per_cell") == row_bytes["gpu"]
                    and contract.get("cpu_bytes_per_cell") == row_bytes["cpu"]
                    and contract.get("logical_bytes_per_cell") == sum(row_bytes.values())
                    and contract.get("committed_bytes") == cells * row_bytes["gpu"]
                    and contract.get("cpu_committed_bytes") == cells * row_bytes["cpu"]
                )
                if not partitioned:
                    return None, None, "unsupported_nonlocal_or_offloaded_kv"
                cpu_source = contract.get("cpu_component_id")
        else:
            cpu_source = self.plan.kv_policy.cache_component
            if (self.plan.kv_policy.offload_ratio != 0.0
                    or scenario.placement.tensor_to_component.get("kv_cache", cpu_source) != cpu_source):
                return None, None, "unsupported_nonlocal_or_offloaded_kv"
        if owner_layers["cpu"] and (cpu_source != target
                or _kind(scenario.hardware.get_component(cpu_source)) != "host_memory"):
            return None, None, "unsupported_cpu_payload_owner"
        byte_count = tokens * sum(row_bytes.values())
        audit = details if details is not None else {}
        audit.update(
            payload_layout_scope=("verified_gpu_subset_cpu_owner_unknown" if cpu_owner_unknown
                                  else "ordinary_full_attention_key_value_subset"),
            payload_unmodeled_terms=("cpu_kv_storage_owner",) if cpu_owner_unknown else (),
            cpu_full_attention_layers=len(owner_layers["cpu"]),
            gpu_payload_bytes=tokens * row_bytes["gpu"],
            cpu_payload_bytes=None if cpu_owner_unknown else tokens * row_bytes["cpu"],
            gpu_payload_service_ns=None if owner_layers["gpu"] else 0.0,
            cpu_payload_service_ns=None if owner_layers["cpu"] or cpu_owner_unknown else 0.0,
            known_payload_service_ns=0.0,
            gpu_tensor_get_count=None, cpu_tensor_get_count=None,
            tensor_get_count_known=False,
            tensor_get_dma_descriptor_setup_applied=False,
            tensor_get_dma_descriptor_service_ns=0.0,
            tensor_get_dma_descriptor_transaction_count=0,
            tensor_get_dma_descriptor_wave_count=0,
        )
        if state.swaps > 0 or state.recomputes > 0 or state.kv_cache_range_error or not valid_ranges:
            return byte_count, None, (
                "unknown_physical_ranges_after_swap" if state.swaps > 0
                else "unknown_physical_ranges_after_recompute" if state.recomputes > 0
                else state.kv_cache_range_error or "unknown_physical_range_lengths"
            )
        audit.update(gpu_tensor_get_count=sum(reads["gpu"].values()),
                     cpu_tensor_get_count=None if cpu_owner_unknown else sum(reads["cpu"].values()),
                     tensor_get_count_known=True)
        cpu_ns = sum(count * self._prompt_cache_host_memory_service(size, copy_payload=True)
                     for size, count in reads["cpu"].items())
        audit.update(cpu_payload_service_ns=None if cpu_owner_unknown else cpu_ns,
                     known_payload_service_ns=cpu_ns)
        gpu_scope = None
        if owner_layers["gpu"] and (partitioned or self.residency_manager is not None):
            # Slot views are capacity aliases, not actual saved tensor ranges.
            # With no physical row starts, only a complete current pool proves
            # these GPU reads resident.  Never call access/read to test this.
            manager = self.residency_manager
            owner_id = str(contract.get("allocation_id", "runtime.kv.pool")).strip() or "runtime.kv.pool"
            owner = manager.allocations.get(owner_id) if manager is not None else None
            committed = contract.get("committed_bytes")
            cells = contract.get("total_cells")
            resident = bool(owner is not None and committed and committed > 0
                            and isinstance(cells, int) and not isinstance(cells, bool)
                            and cells >= tokens and committed == cells * row_bytes["gpu"]
                            and owner.lifecycle is not AllocationLifecycle.RELEASED
                            and owner.kind == "kv_cache"
                            and owner.physical_owner == manager.pool.component_id == source
                            and owner.committed_bytes == committed
                            and owner.resident_bytes == committed
                            and owner.resident_ranges == ((0, committed),))
            audit["gpu_residency_check"] = {
                "allocation_id": owner_id, "fully_resident": resident,
                "committed_bytes": committed,
                "current_resident_bytes": owner.resident_bytes if owner else None,
                "scope": "current_complete_physical_pool",
            }
            if not resident:
                gpu_scope = "unknown_current_gpu_payload_residency"
        if gpu_scope:
            return byte_count, None, gpu_scope
        prompt_cache_d2h_spec = None
        raw_prompt_cache_d2h = scenario.hardware.metadata.get(
            "pageable_prompt_cache_d2h_component_model"
        )
        if (
            isinstance(raw_prompt_cache_d2h, Mapping)
            and raw_prompt_cache_d2h.get("enabled") is True
            and self.prompt_cache.policy.save_implementation
            == "llama_cpp_host_tensor_get_combined"
            and owner_layers["gpu"]
            and (not owner_layers["cpu"] or cpu_owner_unknown)
            and self.prompt_cache.policy.driver_cpu_copy_issue_contract
            == "nvcuda_616_64_payload_max16_v1"
            and raw_prompt_cache_d2h.get("driver_sha256")
            == "3348ef5cb38aed1a9a1e403d7b1ab0b18089e86d2e29a027568e494177edd954"
            and raw_prompt_cache_d2h.get("runtime_sha256")
            == "c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344"
        ):
            try:
                d2h_bandwidth_gbps = float(
                    raw_prompt_cache_d2h.get("bandwidth_gbps", 0.0)
                )
                d2h_fixed_latency_ns = float(
                    raw_prompt_cache_d2h.get("fixed_latency_ns", 0.0)
                )
            except (TypeError, ValueError):
                d2h_bandwidth_gbps = 0.0
                d2h_fixed_latency_ns = 0.0
            if d2h_bandwidth_gbps > 0.0 and d2h_fixed_latency_ns >= 0.0:
                prompt_cache_d2h_spec = {
                    "bandwidth_gbps": d2h_bandwidth_gbps,
                    "fixed_latency_ns": d2h_fixed_latency_ns,
                    "resource_id": str(
                        raw_prompt_cache_d2h.get(
                            "resource_id", "pageable_prompt_cache_d2h_component"
                        )
                    ),
                    "evidence": raw_prompt_cache_d2h.get("evidence"),
                }
                audit["pageable_prompt_cache_d2h_component"] = {
                    "bandwidth_gbps": d2h_bandwidth_gbps,
                    "fixed_latency_ns": d2h_fixed_latency_ns,
                    "resource_id": prompt_cache_d2h_spec["resource_id"],
                    "evidence": prompt_cache_d2h_spec["evidence"],
                    "scope": "gpu_tensor_get_device_to_pageable_host_with_immediate_stream_sync",
                    "replaces": (
                        "topology_gpu_to_host_bulk_service",
                        "driver_cpu_copy_issue_contract",
                    ),
                }
        driver_copy_ns = 0.0
        if (
            prompt_cache_d2h_spec is None
            and self.prompt_cache.policy.driver_cpu_copy_issue_contract
            and reads["gpu"]
        ):
            driver_work = self._prompt_cache_driver_copy_issue(reads["gpu"])
            audit["driver_cpu_copy_issue"] = driver_work
            driver_copy_ns = float(driver_work["service_ns"])
            audit["known_payload_service_ns"] = cpu_ns + driver_copy_ns
            audit["payload_unmodeled_terms"] = (
                *audit.get("payload_unmodeled_terms", ()),
                "driver_cpu_copy_remaining_control_and_memory_work",
            )
        # Each synchronous tensor_get is a separate transfer.  Aggregate only
        # identical sizes, not bytes across calls: route startup and transaction
        # rounding belong to each actual call. Reuse the existing route model.
        service_ns = 0.0
        descriptor_service_ns = 0.0
        descriptor_transactions = 0
        descriptor_waves = 0
        component_service_ns = 0.0
        try:
            for read_bytes, count in reads["gpu"].items():
                if prompt_cache_d2h_spec is not None:
                    cost_ns = prompt_cache_d2h_spec["fixed_latency_ns"] + (
                        8.0
                        * float(read_bytes)
                        / prompt_cache_d2h_spec["bandwidth_gbps"]
                    )
                    component_service_ns += count * cost_ns
                    route = "topology"
                    if self.prompt_cache.policy.apply_tensor_get_controller_phases:
                        _ignored_cost, _energy, _ignored_route, descriptor = (
                            self._tensor_get_copy_cost_details(
                                read_bytes, source, target
                            )
                        )
                        descriptor_service_ns += count * float(
                            descriptor.get("dma_descriptor_service_ns", 0.0)
                        )
                        descriptor_transactions += count * int(
                            descriptor.get("dma_descriptor_transaction_count", 0)
                        )
                        descriptor_waves += count * int(
                            descriptor.get("dma_descriptor_wave_count", 0)
                        )
                elif self.prompt_cache.policy.apply_tensor_get_controller_phases:
                    cost_ns, _energy, route, descriptor = (
                        self._tensor_get_copy_cost_details(
                            read_bytes, source, target
                        )
                    )
                    descriptor_service_ns += count * float(
                        descriptor.get("dma_descriptor_service_ns", 0.0)
                    )
                    descriptor_transactions += count * int(
                        descriptor.get("dma_descriptor_transaction_count", 0)
                    )
                    descriptor_waves += count * int(
                        descriptor.get("dma_descriptor_wave_count", 0)
                    )
                else:
                    cost_ns, _energy, route, _phases, _components, _latency = (
                        self._page_transfer_cost_details(
                            read_bytes,
                            0,
                            source,
                            target,
                            include_controller_phases=False,
                        )
                    )
                if route != "topology" or cost_ns <= 0.0:
                    return byte_count, None, "unknown_topology_route"
                if prompt_cache_d2h_spec is None:
                    service_ns += count * cost_ns
        except ValueError:
            return byte_count, None, "unknown_topology_route"
        if prompt_cache_d2h_spec is not None:
            service_ns = component_service_ns
            if self.prompt_cache.policy.apply_tensor_get_controller_phases:
                service_ns += descriptor_service_ns
        audit.update(gpu_payload_service_ns=service_ns,
                     known_payload_service_ns=service_ns + cpu_ns + driver_copy_ns)
        audit["controller_phases_applied"] = bool(
            self.prompt_cache.policy.apply_tensor_get_controller_phases
        )
        if self.prompt_cache.policy.apply_tensor_get_controller_phases:
            audit.update(
                tensor_get_dma_descriptor_setup_applied=True,
                tensor_get_dma_descriptor_service_ns=descriptor_service_ns,
                tensor_get_dma_descriptor_transaction_count=descriptor_transactions,
                tensor_get_dma_descriptor_wave_count=descriptor_waves,
                tensor_get_translation_service="unknown_unmodeled",
                tensor_get_driver_submission_and_completion="unknown_unmodeled",
            )
        scope = "topology:{}->{}".format(source, target) if owner_layers["gpu"] else "host_copy:{}".format(target)
        if owner_layers["cpu"] and owner_layers["gpu"]:
            scope += ";host_copy:{}".format(target)
        return byte_count, service_ns + cpu_ns + driver_copy_ns, scope

    def _prompt_cache_driver_copy_issue(self, reads: Mapping[int, int]) -> Dict[str, object]:
        """Partial payload issue for an explicitly declared driver-copy path.

        The maximum 16-byte payload operand gives an aggregate load/store
        instruction-count minimum, including scalar heads and tails. It is
        neither the exact XMM count nor a latency lower bound. Source chunk
        identity and payload coverage belong to the declared implementation.
        """
        cpu, _memory = _cpu_profiles(
            self.plan.scenario,
            self.plan.scenario.host_orchestration_profile.cpu_component_id,
        )
        source_cpu = replace(cpu, pipeline=replace(cpu.pipeline, simd_width_bits=128))
        service_ns = 0.0
        charged_gets = logical_bytes = instructions = 0
        unmodeled: Dict[int, int] = {}
        parts = []
        for byte_count, count in sorted(reads.items()):
            if (not 0 < byte_count < (1 << 64) or count <= 0
                    or cpu.pipeline.simd_width_bits < 128):
                unmodeled[byte_count] = count
                continue
            cost = estimate_cpu_logical_stream(
                source_cpu,
                MemoryWorkload(read_bytes=byte_count, write_bytes=byte_count,
                               name="driver_copy_payload_issue_minimum"),
                serial_repetitions=count,
            )
            service_ns += cost.service_ns
            charged_gets += count
            logical_bytes += byte_count * count
            instructions += ((byte_count + 15) // 16) * count
            parts.append({"bytes": byte_count, "count": count,
                          "service_ns": cost.service_ns,
                          "resource_model": dict(cost.metadata)})
        return {
            "contract": self.prompt_cache.policy.driver_cpu_copy_issue_contract,
            "source_driver_sha256": "3348ef5cb38aed1a9a1e403d7b1ab0b18089e86d2e29a027568e494177edd954",
            "source_routine_rva": 0x10a1e0,
            "source_length_domain": "positive_uint64_valid_ranges",
            "scope": "minimum_total_payload_load_store_issue",
            "instruction_count_semantics": "aggregate_minimum_not_exact_xmm_count",
            "service_ns": service_ns, "charged_get_count": charged_gets,
            "unmodeled_get_sizes": unmodeled,
            "minimum_load_instructions": instructions,
            "minimum_store_instructions": instructions,
            "logical_read_bytes": logical_bytes, "logical_write_bytes": logical_bytes,
            "hardware_simd_width_bits": cpu.pipeline.simd_width_bits,
            "max_payload_operand_bytes": 16,
            "resource_id": cpu.compute_resource_id, "parts": parts,
            "physical_memory_bytes_added": 0, "dma_bytes_added": 0,
            "physical_memory_traffic_status": "unknown_not_charged",
            "timing_completeness": "partial", "strict_latency_lower_bound": False,
            "unmodeled_terms": ("alignment_specific_extra_payload_instructions",
                                "control_stack_and_helper_instructions",
                                "cache_memory_and_store_completion"),
        }

    def _prompt_cache_host_initialization_service(self, byte_count: int) -> float:
        # server_prompt_cache::alloc creates a fresh vector<uint8_t> and resizes
        # it before serialization. Count only the verified payload subset of
        # its zero-initialized bytes; headers and allocation remain unknown.
        # This is one host thread inside an existing call, not a new operator
        # dispatch. The typed CPU/cache/memory model owns its service rate.
        return self._prompt_cache_host_memory_service(byte_count, copy_payload=False)

    def _capture_prompt_cache_recurrent_identity(self, state: _MutableRequest) -> None:
        """Keep the original owner alive in evidence, not its residency result."""
        manager = self.residency_manager
        slot = self._request_residency_slots.get(state.spec.request_id)
        view_id = "runtime.linear_state.slot.{:04d}".format(slot) if slot is not None else None
        view = manager.views.get(view_id) if manager is not None and view_id else None
        owner_id = view.owner_id if view is not None else view_id
        owner = manager.allocations.get(owner_id) if manager is not None and owner_id else None
        state.recurrent_save_identity = {
            "active_state": state.linear_state_resident,
            "slot": slot, "view_id": view_id, "owner_id": owner_id,
            "manager": manager, "view": view, "allocation": owner,
        }

    def _prompt_cache_recurrent_payload_service(self, state: _MutableRequest) -> Mapping[str, object]:
        """Price only the explicitly declared current R/S rows, once per save."""
        layout = self.prompt_cache.policy.recurrent_state_layout
        if layout is None:
            return {}
        scenario = self.plan.scenario
        layers = tuple(layer for layer in _execution_layers(scenario) if layer.is_linear_attention)
        if not layers:
            return {}
        result: Dict[str, object] = {
            "layout": layout, "scope": "unknown_recurrent_state_layout",
            "payload_bytes": None, "payload_service_ns": None,
            "known_service_ns": 0.0, "total_service_ns": None,
            "tensor_reads": (), "active_cell_count": None,
            "unmodeled_timing_terms": ("recurrent_state_layout",),
        }

        def unknown(reason: str) -> Mapping[str, object]:
            result.update(scope=reason, unmodeled_timing_terms=(reason,))
            return result

        if state.cached_tokens <= 0:
            result.update(scope="empty_state", payload_bytes=0, payload_service_ns=0.0,
                          active_cell_count=0, unmodeled_timing_terms=())
            return result
        parallel = _parallel_plan(scenario)
        if self.plan.mtp.enabled or parallel.world_size != 1 or len(parallel.ranks) != 1:
            return unknown("unsupported_recurrent_parallel_or_speculative_state")
        if state.swaps or state.recomputes or state.linear_state_swapped_bytes:
            return unknown("unknown_recurrent_source_after_swap_or_recompute")
        contract = self._state_allocation_contract
        if (any(key in contract and (type(contract[key]) is not int or contract[key] != expected)
                for key, expected in (("rollback_plane_count", 0), ("live_plane_count", 1), ("n_rs_seq", 0)))
                or contract.get("shared_sequence_state", False)
                or contract.get("p_tensor_present", False)):
            return unknown("unsupported_recurrent_rollback_shared_or_p_state")
        identity = state.recurrent_save_identity
        if not identity or identity.get("active_state") is not True:
            return unknown("unknown_recurrent_active_cell_identity")
        cpu_id = scenario.host_orchestration_profile.cpu_component_id
        gpu_id = scenario.host_orchestration_profile.gpu_component_id
        source = contract.get("component_id")
        target = self.prompt_cache.offload_component or self.prompt_cache.component
        try:
            if not target or _kind(scenario.hardware.get_component(target)) != "host_memory":
                return unknown("unsupported_recurrent_host_destination")
            reads: List[Dict[str, object]] = []
            # The public writer emits all R rows first, then all S rows.
            for tensor in ("R", "S"):
                for layer in layers:
                    geometry = layer.linear_attention
                    if (geometry is None or geometry.state_dtype.lower() not in {"fp32", "f32", "float32"}
                            or geometry.key_head_dim != geometry.value_head_dim
                            or geometry.conv_kernel_size <= 1):
                        return unknown("unsupported_recurrent_rs_geometry_or_dtype")
                    component = scenario.placement.op_to_component.get(layer.layer_id + ".linear_attention")
                    elements = (geometry.convolution_state_elements if tensor == "R"
                                else geometry.recurrent_state_elements)
                    reads.append({"layer_id": layer.layer_id, "tensor": tensor,
                                  "owner": "cpu" if component == cpu_id else "gpu" if component == gpu_id else "unknown",
                                  "bytes": 4 * elements})
            # A fresh host destination is zero-filled before state export.
            # Its size is independent of source ownership or GPU residency.
            amount = sum(int(row["bytes"]) for row in reads)
            init_ns = self._prompt_cache_host_initialization_service(amount)
            result.update(payload_bytes=amount, active_cell_count=1,
                          tensor_reads=tuple({key: value for key, value in row.items() if key != "owner"}
                                             for row in reads),
                          cpu_payload_service_ns=None, gpu_payload_service_ns=None,
                          host_initialization_bytes=amount, host_initialization_service_ns=init_ns,
                          known_service_ns=init_ns)
            if not contract:
                return unknown("unknown_recurrent_storage_contract")
            if any(row["owner"] == "unknown" for row in reads):
                return unknown("unsupported_recurrent_layer_owner")
            by_owner = {owner: sum(int(row["bytes"]) for row in reads if row["owner"] == owner)
                        for owner in ("cpu", "gpu")}
            slots = contract.get("slot_count")
            if type(slots) is not int or slots <= 0:
                return unknown("unknown_recurrent_slot_geometry")
            expected = {
                "base_bytes_per_seq": by_owner["gpu"],
                "device_base_bytes_per_seq": by_owner["gpu"],
                "logical_base_bytes_per_seq": sum(by_owner.values()),
                "cpu_base_bytes_per_seq": by_owner["cpu"],
                "live_bytes_per_slot": by_owner["gpu"],
                "committed_bytes": slots * by_owner["gpu"],
                "cpu_committed_bytes": slots * by_owner["cpu"],
                "live_row_count": slots,
            }
            if (contract.get("partition_scope") != "gpu_placed_target_recurrent_layers"
                    or any(type(contract.get(key)) is not int or contract[key] != value
                           for key, value in expected.items())):
                return unknown("recurrent_partition_contract_geometry_mismatch")
            if by_owner["cpu"] and (contract.get("cpu_component_id") != target
                    or _kind(scenario.hardware.get_component(contract["cpu_component_id"])) != "host_memory"):
                return unknown("unsupported_recurrent_cpu_storage_owner")
            if by_owner["gpu"] and (source != parallel.ranks[0].memory_component_id
                    or _kind(scenario.hardware.get_component(source)) != "hbm"):
                return unknown("unsupported_recurrent_gpu_storage_owner")
        except (KeyError, ValueError):
            return unknown("unsupported_recurrent_storage_profile")

        cpu_ns = sum(self._prompt_cache_host_memory_service(int(row["bytes"]), copy_payload=True)
                     for row in reads if row["owner"] == "cpu")
        result.update(payload_bytes=amount, tensor_reads=tuple(reads), active_cell_count=1,
                      gpu_payload_bytes=by_owner["gpu"], cpu_payload_bytes=by_owner["cpu"],
                      gpu_tensor_get_count=sum(row["owner"] == "gpu" for row in reads),
                      cpu_tensor_get_count=sum(row["owner"] == "cpu" for row in reads),
                      cpu_payload_service_ns=cpu_ns, gpu_payload_service_ns=0.0 if not by_owner["gpu"] else None,
                      host_initialization_bytes=amount, host_initialization_service_ns=init_ns,
                      known_service_ns=init_ns + cpu_ns)
        if by_owner["gpu"]:
            manager = self.residency_manager
            owner_id, view_id, slot = (identity.get(key) for key in ("owner_id", "view_id", "slot"))
            owner = manager.allocations.get(owner_id) if manager is not None and owner_id else None
            view = manager.views.get(view_id) if manager is not None and view_id else None
            shared_pool = contract.get("shared_pool") is True
            expected_owner = (str(contract.get("allocation_id", "runtime.linear_state.pool"))
                              if shared_pool else view_id)
            expected_commitment = expected["committed_bytes"] if shared_pool else by_owner["gpu"]
            view_valid = (view is identity.get("view") and view is not None
                          and view.owner_id == owner_id and view.offset_bytes == slot * by_owner["gpu"]
                          and view.size_bytes == by_owner["gpu"]) if shared_pool and type(slot) is int else (
                              not shared_pool and view is None)
            resident = bool(manager is not None and manager is identity.get("manager")
                            and owner is not None and owner is identity.get("allocation")
                            and owner_id == expected_owner and type(slot) is int and 0 <= slot < slots
                            and slot not in self._request_residency_slots.values() and view_valid
                            and owner.lifecycle is not AllocationLifecycle.RELEASED
                            and owner.kind == "linear_state"
                            and owner.physical_owner == manager.pool.component_id == source
                            and owner.committed_bytes == expected_commitment
                            and owner.resident_bytes == expected_commitment
                            and owner.resident_ranges == ((0, expected_commitment),))
            result["gpu_residency_check"] = {
                "allocation_id": owner_id, "slot": slot, "fully_resident": resident,
                "scope": "same_captured_current_complete_physical_owner",
                "committed_bytes": expected_commitment,
                "current_resident_bytes": owner.resident_bytes if owner else None,
            }
            if not resident:
                return unknown("unknown_current_recurrent_gpu_residency")
        gpu_ns = 0.0
        try:
            for row in reads:
                if row["owner"] != "gpu":
                    continue
                cost_ns, _energy, route, _phases, _components, _latency = self._page_transfer_cost_details(
                    int(row["bytes"]), 0, source, target, include_controller_phases=False)
                if route != "topology" or cost_ns <= 0:
                    return unknown("unknown_recurrent_topology_route")
                gpu_ns += cost_ns
        except ValueError:
            return unknown("unknown_recurrent_topology_route")
        result.update(scope="declared_one_live_recurrent_rs_row", gpu_payload_service_ns=gpu_ns,
                      payload_service_ns=cpu_ns + gpu_ns, known_service_ns=init_ns + cpu_ns + gpu_ns,
                      unmodeled_timing_terms=("recurrent_metadata_allocator_and_checkpoint_work",
                          *(("driver_copy_submission_and_completion_notification",) if by_owner["gpu"] else ())))
        return result

    def _prompt_cache_host_memory_service(self, byte_count: int, *, copy_payload: bool) -> float:
        if byte_count <= 0:
            return 0.0
        scenario = self.plan.scenario
        cpu, memory = _cpu_profiles(
            scenario, scenario.host_orchestration_profile.cpu_component_id,
            self.prompt_cache.offload_component or self.prompt_cache.component,
        )
        serial_cpu = replace(cpu, dispatch_ns=0.0,
                             pipeline=replace(cpu.pipeline, core_count=1))
        return estimate_cpu_memory(
            serial_cpu, memory,
            MemoryWorkload(read_bytes=byte_count if copy_payload else 0, write_bytes=byte_count,
                           name="prompt_cache_payload_copy" if copy_payload else "prompt_cache_payload_zero_init"),
        ).service_ns

    def _prompt_cache_host_state_work(self, state: _MutableRequest) -> Mapping[str, object]:
        """Count the declared plain state format, retaining unsupported work."""
        policy = self.prompt_cache.policy
        if policy.host_state_layout is None:
            return {}
        scenario = self.plan.scenario
        layers = _execution_layers(scenario)
        parallel = _parallel_plan(scenario)
        gpu_id = scenario.host_orchestration_profile.gpu_component_id
        supported = (
            policy.unified_kv and not self.plan.mtp.enabled
            and scenario.fusion_policy.flash_attention and bool(layers)
            and parallel.world_size == 1 and len(parallel.ranks) == 1
            and all(not layer.is_linear_attention
                    and scenario.placement.op_to_component.get(layer.layer_id + ".attention") == gpu_id
                    for layer in layers)
        )
        scope = "unsupported_plain_kv_structure"
        if supported:
            rank = parallel.ranks[0]
            source = rank.memory_component_id
            target = self.prompt_cache.offload_component
            kv_owner = _mapping_or_empty(_runtime_allocation_contract(self.plan).get("kv"))
            try:
                supported = bool(
                    rank.component_id == gpu_id and source and target
                    and self.plan.kv_policy.cache_component == source
                    and scenario.placement.tensor_to_component.get("kv_cache", source) == source
                    and kv_owner.get("component_id", source) == source
                    and self.plan.kv_policy.offload_ratio == 0.0
                    and _kind(scenario.hardware.get_component(source)) == "hbm"
                    and _kind(scenario.hardware.get_component(target)) == "host_memory"
                )
            except (KeyError, ValueError):
                supported = False
            scope = "unsupported_plain_kv_storage_owner"
        if supported and scenario.workload.requests:
            request = next((item for item in scenario.workload.requests
                            if item.request_id == state.spec.request_id), None)
            supported = request is not None and set(_request_modalities(request)) <= {"text"}
            scope = "unsupported_non_text_request" if request is not None else "unknown_plain_kv_request_metadata"
        if not supported:
            return {"layout": policy.host_state_layout, "scope": scope,
                    "known_service_ns": 0.0, "total_service_ns": None, "terms": None,
                    "unmodeled_timing_terms": (scope,)}
        tokens = state.cached_tokens
        if tokens <= 0:
            return {"layout": policy.host_state_layout, "scope": "empty_state",
                    "known_service_ns": 0.0, "total_service_ns": None, "terms": {},
                    "known_logical_read_bytes": 0, "known_logical_write_bytes": 0,
                    "unmodeled_timing_terms": ()}
        header_bytes = 24 + 12 * tokens + 24 * len(layers)
        observed = scenario.workload.metadata.get("runtime_observed_config", {})
        capacity = observed.get("context_length") if isinstance(observed, _ABCMapping) else None
        capacity = capacity if isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0 else None
        ranges_known = (
            not state.swaps and not state.recomputes and not state.kv_cache_range_error
            and bool(state.kv_cache_range_tokens) and all(n > 0 for n in state.kv_cache_range_tokens)
            and sum(state.kv_cache_range_tokens) == tokens
            and len(state.kv_cache_range_tokens) == state.kv_cache_range_count
        )
        gets = 2 * len(layers) * len(state.kv_cache_range_tokens) if ranges_known else None
        descriptor_bytes = 4 * policy.host_pointer_bytes if policy.host_pointer_bytes else None
        cpu, _memory = _cpu_profiles(scenario, scenario.host_orchestration_profile.cpu_component_id)
        terms: Dict[str, object] = {}
        # Actual writer calls remain 4/8-byte serial accesses. The token clone
        # and vector initialization are real contiguous vector operations.
        specifications = (
            ("header_zero_initialization", 0, header_bytes, 1),
            ("header_fields_4_bytes", 4, 4, 6 + 3 * tokens + 2 * len(layers)),
            ("header_fields_8_bytes", 8, 8, 2 * len(layers)),
            ("plain_token_clone", 4 * tokens, 4 * tokens, 1),
            ("pool_position_scan", 4, 0, 2 * capacity if capacity else None),
            ("tensor_read_descriptors", descriptor_bytes, descriptor_bytes,
             gets if descriptor_bytes is not None else None),
        )
        known_ns = 0.0
        known_read = known_write = 0
        for name, read_bytes, write_bytes, repetitions in specifications:
            if repetitions is None:
                terms[name] = None
                continue
            estimate = estimate_cpu_logical_stream(
                cpu, MemoryWorkload(read_bytes=read_bytes, write_bytes=write_bytes, name=name),
                serial_repetitions=repetitions,
            )
            terms[name] = {"read_bytes": read_bytes * repetitions,
                           "write_bytes": write_bytes * repetitions, "repetitions": repetitions,
                           "service_ns": estimate.service_ns, "resource_model": dict(estimate.metadata)}
            known_ns += estimate.service_ns
            known_read += read_bytes * repetitions
            known_write += write_bytes * repetitions
        return {
            "layout": policy.host_state_layout, "scope": "declared_plain_kv_logical_work",
            "source_revision": "0f3a71be15af836d277c9f918adfafb45732677e",
            "source_files": ("src/llama-kv-cache.cpp:2045,2195,2228", "src/llama-context.cpp:2576",
                             "tools/server/server-task.cpp:1760", "tools/server/server-common.cpp:774"),
            "persistent_tokens": tokens, "serialized_layers": len(layers), "header_bytes": header_bytes,
            "host_pointer_bytes": policy.host_pointer_bytes, "descriptor_bytes": descriptor_bytes,
            "tensor_get_count": gets, "terms": terms, "known_service_ns": known_ns,
            "known_logical_read_bytes": known_read, "known_logical_write_bytes": known_write,
            "physical_memory_traffic_status": "unknown_not_charged", "total_service_ns": None,
            "unmodeled_timing_terms": (
                "state_sizing_membership_and_control_instructions", "metadata_and_clone_control_instructions",
                "checkpoint_contents_and_cache_management", "compiler_and_physical_memory_traffic",
                *(name for name, value in terms.items() if value is None),
            ),
        }

    def _record_prompt_cache_ranges(self, cohort: BatchCohort, cost: BatchCost) -> None:
        """Reuse the planner's physical lane order; missing evidence stays unknown."""
        if self.prompt_cache.policy.save_implementation != "llama_cpp_host_tensor_get_combined":
            return
        expected = sum(_batch_item_kv_append_tokens(item) for item in cohort.items)
        if expected <= 0:
            return
        self._prompt_cache_appended_tokens += expected
        observed = self.plan.scenario.workload.metadata.get("runtime_observed_config", {})
        capacity = int(observed.get("context_length", 0))
        error = None
        sequence: List[str] = []
        try:
            if capacity <= 0 or self._prompt_cache_appended_tokens > capacity:
                raise ValueError("unknown_physical_pool_position_or_wrap")
            groups = cost.metadata.get("operator_invocation_groups", ())
            if not groups:
                raise ValueError("unknown_physical_invocation_order")
            seen: Set[Tuple[int, int]] = set()
            previous = None
            for group in groups:
                group_id = group["group_id"]
                if not group_id or group.get("predecessor_group_id") != previous:
                    raise ValueError("unknown_physical_invocation_order")
                previous = group_id
                group_appends = 0
                for lane in group["lanes"]:
                    index, position = int(lane["item_index"]), int(lane["position"])
                    if index < 0 or index >= len(cohort.items):
                        raise ValueError("unknown_physical_lane_coverage")
                    item = cohort.items[index]
                    if (lane["request_id"] != item.request_id
                            or not 0 <= position < item.token_count
                            or (index, position) in seen):
                        raise ValueError("unknown_physical_lane_coverage")
                    seen.add((index, position))
                    if position < _batch_item_kv_append_tokens(item):
                        sequence.append(item.request_id)
                        group_appends += 1
                if group_appends != group["kv_append_tokens"]:
                    raise ValueError("unknown_physical_lane_coverage")
            if len(seen) != sum(item.token_count for item in cohort.items) or len(sequence) != expected:
                raise ValueError("unknown_physical_lane_coverage")
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            error = str(exc) if isinstance(exc, ValueError) else "unknown_physical_invocation_order"
        if error:
            affected = {item.request_id for item in cohort.items}
            for state in self._state_values:
                if state.kv_cache_range_tokens or state.spec.request_id in affected:
                    state.kv_cache_range_error = error
            self._last_prompt_cache_allocation_request_id = None
            return
        for request_id in sequence:
            state = self.states[request_id]
            if not state.kv_cache_range_error:
                if (not state.kv_cache_range_tokens or self.prompt_cache.policy.unified_kv
                        and self._last_prompt_cache_allocation_request_id != request_id):
                    state.kv_cache_range_tokens.append(0)
                state.kv_cache_range_tokens[-1] += 1
            self._last_prompt_cache_allocation_request_id = request_id

    def _save_pending_prompt_cache_states(self) -> None:
        """Serialize llama.cpp-like checkpoint saves before a new slot enters."""

        if not self._prompt_cache_save_enabled():
            return
        scenario = self.plan.scenario
        gpu_id = scenario.host_orchestration_profile.gpu_component_id
        combined = self.prompt_cache.policy.save_implementation == "llama_cpp_host_tensor_get_combined"
        controller = scenario.runtime_profile.gpu_controllers.get(gpu_id)
        if controller is None and not combined:
            raise ValueError(
                "llama_cpp_host_tensor_get requires a GPU controller for {}"
                .format(gpu_id)
            )
        submission_ns = (float(controller.command_processor.command_submission_latency_ns)
                         if controller is not None else 0.0)
        gpu_attention_layers = self._prompt_cache_gpu_attention_layer_count()
        while self._pending_prompt_cache_states:
            state = self._pending_prompt_cache_states.pop(0)
            range_count = max(0, int(state.kv_cache_range_count))
            extra_ranges = max(0, range_count - 1)
            extra_tensor_gets = 2 * gpu_attention_layers * extra_ranges
            service_ns = extra_tensor_gets * submission_ns
            refinement_details: Dict[str, object] = {}
            no_gpu_reads = False
            if combined:
                base_tensor_gets = 2 * gpu_attention_layers if range_count > 0 else 0
                apply_submission_service = bool(
                    self.prompt_cache.policy.apply_tensor_get_submission_service
                )
                owner_work: Dict[str, object] = {}
                payload_bytes, payload_service_ns, payload_scope = self._prompt_cache_payload_service(state, owner_work)
                no_gpu_reads = (owner_work.get("gpu_tensor_get_count") == 0
                                and owner_work.get("cpu_tensor_get_count") is not None)
                init_ns = (
                    self._prompt_cache_host_initialization_service(payload_bytes)
                    if payload_bytes is not None else None
                )
                # Keep a timeline of known work, never label it full service.
                # The optional price is a predeclared analytical assumption.
                # Unknown layout/ranges must not produce a guessed read count.
                # Recurrent R/S reads are a separate, unchanged scope below.
                known_payload_ns = float(owner_work.get("known_payload_service_ns", payload_service_ns or 0.0))
                submission_count = owner_work.get("gpu_tensor_get_count")
                submission_count_known = (
                    type(submission_count) is int and submission_count >= 0
                    and owner_work.get("tensor_get_count_known") is True
                )
                submission_service_ns = (
                    submission_count * submission_ns
                    if apply_submission_service and submission_count_known else 0.0
                )
                submission_split_known = (
                    submission_count_known
                    and submission_count == base_tensor_gets + extra_tensor_gets
                )
                service_ns = (init_ns or 0.0) + known_payload_ns + submission_service_ns
                refinement_details = {
                    "base_tensor_gets": base_tensor_gets,
                    "base_submission_service_ns": (
                        base_tensor_gets * submission_ns
                        if apply_submission_service and submission_split_known else None
                    ),
                    "extra_submission_service_ns": (
                        extra_tensor_gets * submission_ns
                        if apply_submission_service and submission_split_known else None
                    ),
                    "tensor_get_submission_service_ns": submission_service_ns,
                    "submission_proxy_applied": apply_submission_service,
                    "payload_bytes": payload_bytes,
                    "payload_service_ns": payload_service_ns,
                    "payload_scope": payload_scope,
                    "payload_timing_known": payload_service_ns is not None,
                    "tensor_get_count_known": payload_service_ns is not None,
                    "controller_phases_applied": owner_work.get(
                        "controller_phases_applied", False
                    ),
                    **owner_work,
                    "range_token_histogram": dict(Counter(state.kv_cache_range_tokens)),
                    "host_payload_initialization_bytes": payload_bytes if init_ns is not None else None,
                    "host_payload_initialization_service_ns": init_ns,
                    "known_service_ns": service_ns,
                    "total_service_ns": None,
                    "unmodeled_timing_terms": (
                        "state_sizing_and_metadata_serialization",
                        "allocator_page_faults_and_descriptor_growth",
                        *(() if no_gpu_reads else ("driver_copy_submission_and_completion_notification",)),
                        *owner_work.get("payload_unmodeled_terms", ()),
                        *(() if payload_service_ns is not None else (payload_scope,)),
                    ),
                    "service_semantics": "modeled_components_only_not_complete_latency",
                }
                if apply_submission_service:
                    refinement_details.update(
                        submission_price_provenance="predeclared_analytical_controller_default",
                        submission_physical_service_verified=False,
                        submission_scope="ordinary_full_attention_kv_gpu_reads_only",
                        submission_tensor_get_count=submission_count if submission_count_known else None,
                        submission_count_status="known" if submission_count_known else "unknown",
                    )
                    if not submission_count_known:
                        refinement_details["unmodeled_timing_terms"] += ("tensor_get_submission_count_unknown",)
                host_work = self._prompt_cache_host_state_work(state)
                if host_work:
                    service_ns += float(host_work["known_service_ns"])
                    refinement_details.update(host_state_work=host_work, known_service_ns=service_ns)
                    old_unknown = refinement_details["unmodeled_timing_terms"]
                    if host_work["scope"] == "declared_plain_kv_logical_work":
                        old_unknown = tuple(term for term in old_unknown if term != "state_sizing_and_metadata_serialization")
                    refinement_details["unmodeled_timing_terms"] = tuple(old_unknown) + tuple(host_work["unmodeled_timing_terms"])
                recurrent_work = self._prompt_cache_recurrent_payload_service(state)
                if recurrent_work:
                    service_ns += float(recurrent_work["known_service_ns"])
                    refinement_details.update(recurrent_state_work=recurrent_work, known_service_ns=service_ns)
                    refinement_details["unmodeled_timing_terms"] = tuple(dict.fromkeys((
                        *refinement_details["unmodeled_timing_terms"], *recurrent_work["unmodeled_timing_terms"])))
                    if recurrent_work.get("gpu_tensor_get_count") != 0:
                        no_gpu_reads = False
            start_ns = max(
                self.now, self._host_available_ns, self._gpu_available_ns
            )
            end_ns = start_ns + service_ns
            details = {
                "implementation": self.prompt_cache.policy.save_implementation,
                "slot_request_id": state.spec.request_id,
                "kv_cache_range_count": range_count,
                "extra_ranges": extra_ranges,
                "gpu_full_attention_layers": gpu_attention_layers,
                "extra_tensor_gets": extra_tensor_gets,
                "command_submission_latency_ns_per_get": submission_ns,
                "service_ns": service_ns,
                "partial_timing": True,
                "timing_completeness": "partial",
                "component_placement_assumption": (
                    "verified_full_attention_backend_owners_in_payload_layout_scope" if combined else
                    "placement.op_to_component[layer_id + '.attention'] "
                    "equals host_orchestration_profile.gpu_component_id"
                ),
                "cuda_stream_synchronize_wall_time": (
                    "not_applicable_no_gpu_tensor_gets" if no_gpu_reads else "unknown_unmodeled"),
                **refinement_details,
            }
            self.events.append(
                ServingEvent(
                    start_ns,
                    "prompt_cache_save_start",
                    state.spec.request_id,
                    details=details,
                )
            )
            self.events.append(
                ServingEvent(
                    end_ns,
                    "prompt_cache_save_end",
                    state.spec.request_id,
                    details=details,
                )
            )
            self.now = end_ns
            self._host_available_ns = end_ns
            self._gpu_available_ns = end_ns
            self._last_gpu_start_ns = start_ns
            self._last_serial_device_end_ns = end_ns
            self.prompt_cache.add_completed(state, end_ns)

    def _eager_slot_available(self, additional_slots: int) -> bool:
        """Check logical eager commitments against remaining host backing."""

        if (
            not self.resource_policy.enabled
            or self.plan.kv_policy.allocation_policy != "eager"
        ):
            return True
        terms = _eager_slot_reservation(
            self.plan, self._running_count + max(0, int(additional_slots))
        )
        kv_available = max(
            0,
            int(self.plan.kv_policy.offload_capacity_bytes)
            - int(self.ledger.offload_used_bytes),
        )
        state_available = max(
            0,
            int(self.plan.linear_state_policy.offload_capacity_bytes)
            - int(self.state_ledger.offload_used_bytes),
        )
        return (
            int(terms["kv_host_backed_bytes"]) <= kv_available
            and int(terms["state_host_backed_bytes"]) <= state_available
        )

    def _resume_swap(self, state: _MutableRequest) -> bool:
        byte_count = state.swap_bytes
        swapped_pages = state.swapped_pages
        state_byte_count = state.linear_state_swapped_bytes
        target_pages = swapped_pages
        if self.plan.kv_policy.allocation_policy == "eager":
            target_pages = self.ledger.pages_for_tokens(
                _max_live_kv_tokens(
                    state.spec.prompt_tokens, state.spec.output_tokens
                )
            )
        # Keep offloaded KV in the host ledger until the transfer cohort has
        # completed.  ``can_restore`` checks the net move, so active and spill
        # storage may share one physical component without a transient double
        # charge.
        kv_restore_pending = byte_count > 0
        if kv_restore_pending:
            if not self.ledger.can_restore(state, target_pages):
                return False
            # KV and linear-state restore share the same physical cache.  The
            # individual ledgers cannot see each other's pending destination
            # bytes, so preflight both moves on a scratch physical ledger
            # before mutating either ledger.
            restore_check = _PhysicalCapacityLedger(
                self.physical_ledger.limits
            )
            restore_check.used_bytes.update(self.physical_ledger.used_bytes)
            state_cache = self.plan.linear_state_policy.cache_component
            state_offload = self.plan.linear_state_policy.offload_component
            if state_byte_count > 0:
                if not state_cache or not state_offload:
                    return False
                if (
                    self.state_ledger.offload_used_bytes < state_byte_count
                    or self.state_ledger.used_requests
                    >= self.state_ledger.policy.capacity_requests
                    or not restore_check.transfer(
                        state_offload,
                        state_cache,
                        state_byte_count,
                    )
                ):
                    return False
            elif (
                self.plan.linear_state_policy.bytes_per_request > 0
                and not state.linear_state_resident
            ):
                if (
                    self.state_ledger.used_requests
                    >= self.state_ledger.policy.capacity_requests
                    or not restore_check.adjust(
                        state_cache,
                        self.plan.linear_state_policy.bytes_per_request,
                    )
                ):
                    return False
            extra_bytes = (
                target_pages - swapped_pages
            ) * self.plan.kv_policy.bytes_per_page
            if not restore_check.transfer(
                self.plan.kv_policy.offload_component,
                self.plan.kv_policy.cache_component,
                byte_count,
                extra_bytes,
            ):
                return False
        elif not self._reserve_with_pressure(state, target_pages):
            return False
        if not self.state_ledger.restore(state):
            if not kv_restore_pending:
                self.ledger.resize(state, 0)
            return False
        if byte_count > 0:
            source = self.plan.kv_policy.offload_component
            target = self.plan.kv_policy.cache_component
            if not source or not target:
                raise ValueError("swapped KV state is missing a valid transfer")
            self.events.append(
                ServingEvent(
                    self.now,
                    "kv_swap_in_start",
                    state.spec.request_id,
                    details={"pages": swapped_pages, "bytes": byte_count},
                )
            )
            self._execute_swap_transfer(
                state,
                "kv_swap_in",
                source,
                target,
                byte_count,
                swapped_pages,
            )
            if not self.ledger.restore(state, target_pages):
                raise RuntimeError("failed to restore KV state after transfer")
            self.swap_in_bytes += byte_count
            self.logical_swap_in_bytes += self._logical_bytes_for_physical(
                byte_count
            )
            self.events.append(
                ServingEvent(
                    self.now,
                    "kv_swap_in",
                    state.spec.request_id,
                    details={"pages": swapped_pages, "bytes": byte_count},
                )
            )
        elif swapped_pages > 0:
            raise ValueError("swapped KV pages are missing transfer bytes")
        if state_byte_count > 0:
            state_source = self.plan.linear_state_policy.offload_component
            state_target = self.plan.linear_state_policy.cache_component
            if not state_source or not state_target:
                raise ValueError("swapped linear state is missing a valid transfer")
            state_details = {"bytes": state_byte_count}
            self.events.append(
                ServingEvent(
                    self.now,
                    "linear_state_swap_in_start",
                    state.spec.request_id,
                    details=state_details,
                )
            )
            self._execute_swap_transfer(
                state,
                "linear_state_swap_in",
                state_source,
                state_target,
                state_byte_count,
            )
            self.linear_state_swap_in_bytes += state_byte_count
            self.events.append(
                ServingEvent(
                    self.now,
                    "linear_state_swap_in",
                    state.spec.request_id,
                    details=state_details,
                )
            )
        return True

    def _priority_preempt(self) -> None:
        if not self.plan.scheduler.preemption_enabled:
            return
        if self._running_count < self.plan.scheduler.max_num_seqs:
            return
        best = None
        best_rank = None
        active: List[_MutableRequest] = []
        for state in self._state_values:
            if state.status in (RequestStatus.WAITING, RequestStatus.SWAPPED):
                rank = self._rank(state)
                if best_rank is None or rank < best_rank:
                    best = state
                    best_rank = rank
            elif state.status == RequestStatus.RUNNING:
                active.append(state)
        if best is None or best_rank is None or not active:
            return
        for victim in sorted(active, key=self._rank, reverse=True):
            if best_rank >= self._rank(victim):
                return
            if self._preempt(victim, "higher_priority"):
                self.priority_preemptions += 1
                return

    def _select_cohort(self) -> Optional[BatchCohort]:
        active: List[_MutableRequest] = []
        decode: List[_MutableRequest] = []
        recompute: List[_MutableRequest] = []
        prefill: List[_MutableRequest] = []
        best_decode_rank = None
        best_recompute_rank = None
        best_prefill_rank = None
        for state in self._state_values:
            if state.status != RequestStatus.RUNNING:
                continue
            active.append(state)
            rank = self._rank(state)
            is_recompute = state.recompute_cursor < state.recompute_target
            is_prefill = state.prefill_cursor < state.spec.prompt_tokens
            if is_recompute:
                recompute.append(state)
                if best_recompute_rank is None or rank < best_recompute_rank:
                    best_recompute_rank = rank
            if is_prefill:
                prefill.append(state)
                if best_prefill_rank is None or rank < best_prefill_rank:
                    best_prefill_rank = rank
            if (
                not is_recompute
                and not is_prefill
                and state.committed < state.spec.output_tokens
            ):
                decode.append(state)
                if best_decode_rank is None or rank < best_decode_rank:
                    best_decode_rank = rank

        # Decode is the default tie-breaker, while priority/deadline/aging can
        # promote a prefill chunk.  queued_since_ns is refreshed whenever a
        # request receives service, preventing an endless decode stream from
        # starving a prompt.
        best_non_decode_rank = None
        serve_recompute = False
        if best_recompute_rank is not None and (
            best_prefill_rank is None or best_recompute_rank <= best_prefill_rank
        ):
            best_non_decode_rank = best_recompute_rank
            serve_recompute = True
        elif best_prefill_rank is not None:
            best_non_decode_rank = best_prefill_rank
        serve_non_decode = best_non_decode_rank is not None and (
            best_decode_rank is None or best_non_decode_rank < best_decode_rank
        )
        if self.plan.scheduler.mixed_phase_batching and decode:
            decode_cohort = self._decode_cohort(decode)
            if decode_cohort is not None:
                mixed_items: List[BatchItem] = list(decode_cohort.items)
                protected_request_ids: List[str] = list(
                    decode_cohort.request_ids
                )
                budget = (
                    self.plan.scheduler.max_num_batched_tokens
                    - sum(item.token_count for item in mixed_items)
                )
                sequence_budget = (
                    self.plan.scheduler.max_num_seqs
                    - len(set(protected_request_ids))
                )
                non_decode_groups: List[Tuple[Sequence[_MutableRequest], bool]] = []
                if best_recompute_rank is not None and (
                    best_prefill_rank is None
                    or best_recompute_rank <= best_prefill_rank
                ):
                    non_decode_groups.append((recompute, True))
                    if best_prefill_rank is not None:
                        non_decode_groups.append((prefill, False))
                elif best_prefill_rank is not None:
                    non_decode_groups.append((prefill, False))
                    if best_recompute_rank is not None:
                        non_decode_groups.append((recompute, True))
                for group_candidates, group_recompute in non_decode_groups:
                    if budget <= 0 or sequence_budget <= 0:
                        break
                    filler = self._prefill_cohort(
                        group_candidates,
                        recompute=group_recompute,
                        token_budget=budget,
                        sequence_budget=sequence_budget,
                        protected_request_ids=protected_request_ids,
                    )
                    if filler is None:
                        continue
                    mixed_items.extend(filler.items)
                    protected_request_ids.extend(
                        request_id
                        for request_id in filler.request_ids
                        if request_id not in protected_request_ids
                    )
                    budget = (
                        self.plan.scheduler.max_num_batched_tokens
                        - sum(item.token_count for item in mixed_items)
                    )
                    sequence_budget = (
                        self.plan.scheduler.max_num_seqs
                        - len(set(protected_request_ids))
                    )
                if len(mixed_items) == len(decode_cohort.items):
                    return decode_cohort
                return BatchCohort(
                    "cohort-{:06d}".format(len(self.batches)),
                    "mixed",
                    self.now,
                    tuple(mixed_items),
                    (
                        self.plan.mtp.proposal_cost_scale
                        if any(item.phase == "mtp" for item in mixed_items)
                        else 1.0
                    ),
                    self._resource_snapshot(mixed_items)
                    if self.resource_policy.enabled
                    else {},
                )
        if serve_non_decode:
            cohort = self._prefill_cohort(
                recompute if serve_recompute else prefill,
                recompute=serve_recompute,
            )
            if cohort is not None:
                return cohort
        if decode:
            cohort = self._decode_cohort(decode)
            if cohort is not None:
                return cohort
        if recompute:
            cohort = self._prefill_cohort(recompute, recompute=True)
            if cohort is not None:
                return cohort
        if prefill:
            return self._prefill_cohort(prefill, recompute=False)
        for state in active:
            if state.spec.output_tokens == 0 and state.prefill_cursor >= state.spec.prompt_tokens:
                self._finish(state, self.now)
        return None

    def _decode_cohort(
        self,
        candidates: Sequence[_MutableRequest],
        token_budget: Optional[int] = None,
        sequence_budget: Optional[int] = None,
        protected_request_ids: Sequence[str] = (),
    ) -> Optional[BatchCohort]:
        items: List[BatchItem] = []
        protected = set(protected_request_ids)
        batch_protected_request_ids: List[str] = list(protected_request_ids)
        available_sequences = max(
            0, self.plan.scheduler.max_num_seqs - len(protected)
        )
        budget = (
            self.plan.scheduler.max_num_batched_tokens
            if token_budget is None
            else min(
                self.plan.scheduler.max_num_batched_tokens,
                max(0, int(token_budget)),
            )
        )
        max_sequences = (
            available_sequences
            if sequence_budget is None
            else min(
                available_sequences,
                max(0, int(sequence_budget)),
            )
        )
        candidates = sorted(candidates, key=self._service_rank)
        for state in candidates:
            if len(items) >= max_sequences or budget <= 0:
                break
            if state.status != RequestStatus.RUNNING:
                continue
            request_id = state.spec.request_id
            if request_id in protected:
                continue
            remaining = state.spec.output_tokens - state.committed
            phase = "mtp" if self.plan.mtp.enabled else "decode"
            if phase == "mtp":
                current_cursor = state.mtp_cursor or MTPRequestCursor(
                    self.plan.mtp
                )
                tentative_cursor = current_cursor.clone()
                mtp_round = tentative_cursor.next_round(
                    min(remaining, budget)
                )
                proposed = mtp_round.verifier_tokens
                accepted = mtp_round.committed_tokens
                expected = 1.0 + mtp_round.expected_accepted_draft_tokens
                main_tokens = mtp_round.main_tokens
                draft_tokens = mtp_round.draft_tokens
                verifier_tokens = mtp_round.verifier_tokens
                committed_tokens = mtp_round.committed_tokens
            else:
                tentative_cursor = None
                proposed = min(remaining, 1, budget)
                accepted = proposed
                expected = float(accepted)
                main_tokens = None
                draft_tokens = None
                verifier_tokens = None
                committed_tokens = None
            if proposed <= 0:
                continue
            target_live_tokens = _max_live_kv_tokens(
                state.prefill_cursor, state.committed + accepted
            )
            append_tokens = max(
                0, target_live_tokens - state.cached_tokens
            )
            materialized_tokens = proposed if phase == "mtp" else append_tokens
            if phase == "mtp":
                materialized_live_tokens = (
                    state.cached_tokens + materialized_tokens
                )
                target_pages = self.ledger.pages_for_tokens(
                    materialized_live_tokens
                )
                reserved = self._reserve_temporary_with_pressure(
                    state, target_pages, batch_protected_request_ids
                )
            else:
                target_pages = self.ledger.pages_for_tokens(target_live_tokens)
                reserved = self._reserve_with_pressure(
                    state, target_pages, batch_protected_request_ids
                )
            if not reserved:
                continue
            if tentative_cursor is not None:
                state.mtp_cursor = tentative_cursor
                state.mtp_round = tentative_cursor.round_index
            items.append(
                BatchItem(
                    state.spec.request_id,
                    phase,
                    proposed,
                    state.cached_tokens,
                    proposed_tokens=proposed,
                    expected_accepted_tokens=expected,
                    kv_append_tokens=append_tokens,
                    kv_materialized_tokens=materialized_tokens,
                    main_tokens=main_tokens,
                    draft_tokens=draft_tokens,
                    verifier_tokens=verifier_tokens,
                    committed_tokens=committed_tokens,
                    logit_tokens=proposed,
                    completion_cursor=state.committed + accepted,
                )
            )
            protected.add(request_id)
            batch_protected_request_ids.append(request_id)
            budget -= proposed
        if not items:
            return None
        kind = "mtp" if self.plan.mtp.enabled else "decode"
        return BatchCohort(
            "cohort-{:06d}".format(len(self.batches)),
            kind,
            self.now,
            tuple(items),
            self.plan.mtp.proposal_cost_scale if kind == "mtp" else 1.0,
            self._resource_snapshot(items)
            if self.resource_policy.enabled
            else {},
        )

    def _prefill_cohort(
        self,
        candidates: Sequence[_MutableRequest],
        recompute: bool,
        token_budget: Optional[int] = None,
        sequence_budget: Optional[int] = None,
        protected_request_ids: Sequence[str] = (),
    ) -> Optional[BatchCohort]:
        items: List[BatchItem] = []
        protected = set(protected_request_ids)
        batch_protected_request_ids: List[str] = list(protected_request_ids)
        available_sequences = max(
            0, self.plan.scheduler.max_num_seqs - len(protected)
        )
        budget = (
            self.plan.scheduler.max_num_batched_tokens
            if token_budget is None
            else min(
                self.plan.scheduler.max_num_batched_tokens,
                max(0, int(token_budget)),
            )
        )
        max_sequences = (
            available_sequences
            if sequence_budget is None
            else min(
                available_sequences,
                max(0, int(sequence_budget)),
            )
        )
        candidates = sorted(candidates, key=self._service_rank)
        for state in candidates:
            if len(items) >= max_sequences or budget <= 0:
                break
            if state.status != RequestStatus.RUNNING:
                continue
            request_id = state.spec.request_id
            if request_id in protected:
                continue
            remaining = (
                state.recompute_target - state.recompute_cursor
                if recompute
                else state.spec.prompt_tokens - state.prefill_cursor
            )
            chunk = min(remaining, self.plan.scheduler.prefill_chunk_tokens, budget)
            if not recompute and not self.plan.mtp.enabled:
                for offset in self.plan.scheduler.prefill_stop_offsets:
                    if remaining > offset:
                        chunk = min(chunk, remaining - offset)
            if chunk <= 0:
                continue
            current_cached = state.recompute_cursor if recompute else state.prefill_cursor
            target_pages = self.ledger.pages_for_tokens(current_cached + chunk)
            if not self._reserve_with_pressure(
                state, target_pages, batch_protected_request_ids
            ):
                continue
            items.append(
                BatchItem(
                    request_id,
                    "recompute" if recompute else "prefill",
                    chunk,
                    current_cached,
                    kv_append_tokens=chunk,
                    logit_tokens=(
                        1
                        if current_cached + chunk
                        >= (
                            state.recompute_target
                            if recompute
                            else state.spec.prompt_tokens
                        )
                        else 0
                    ),
                    completion_cursor=current_cached + chunk,
                )
            )
            protected.add(request_id)
            batch_protected_request_ids.append(request_id)
            budget -= chunk
        if not items:
            return None
        return BatchCohort(
            "cohort-{:06d}".format(len(self.batches)),
            "prefill",
            self.now,
            tuple(items),
            metadata=(
                self._resource_snapshot(items)
                if self.resource_policy.enabled
                else {}
            ),
        )

    def _reserve_with_pressure(
        self,
        owner: _MutableRequest,
        target_pages: int,
        protected_request_ids: Sequence[str] = (),
    ) -> bool:
        self.prompt_cache.ensure_capacity(
            max(0, int(target_pages) - int(owner.kv_pages))
            * max(0, int(self.plan.kv_policy.bytes_per_page))
        )
        return self._reserve_capacity_with_pressure(
            owner,
            lambda: self.ledger.resize(owner, target_pages),
            protected_request_ids,
        )

    def _reserve_temporary_with_pressure(
        self,
        owner: _MutableRequest,
        target_total_pages: int,
        protected_request_ids: Sequence[str] = (),
    ) -> bool:
        self.prompt_cache.ensure_capacity(
            max(0, int(target_total_pages) - int(owner.kv_pages))
            * max(0, int(self.plan.kv_policy.bytes_per_page))
        )
        return self._reserve_capacity_with_pressure(
            owner,
            lambda: self.ledger.reserve_temporary(
                owner, target_total_pages
            ),
            protected_request_ids,
        )

    def _reserve_capacity_with_pressure(
        self,
        owner: _MutableRequest,
        reserve: Callable[[], bool],
        protected_request_ids: Sequence[str],
    ) -> bool:
        if reserve():
            return True
        if not self.plan.scheduler.preemption_enabled:
            return False
        protected = set(protected_request_ids)
        victims = [
            state for state in self._state_values
            if state is not owner
            and state.status == RequestStatus.RUNNING
            and state.spec.request_id not in protected
            and state.kv_pages > 0
        ]
        victims.sort(key=self._rank, reverse=True)
        for victim in victims:
            if not self._preempt(victim, "memory_pressure"):
                continue
            self.memory_preemptions += 1
            if reserve():
                return True
        return False

    def _preempt(self, state: _MutableRequest, reason: str) -> bool:
        if state.status != RequestStatus.RUNNING:
            return False
        if self._kv_scan_enabled:
            raise ValueError("KV scan lower bound does not support preemption or swapping")
        # Cohort execution is synchronous, so this is normally already zero at
        # a scheduling boundary.  Keep preemption defensive: proposal pages
        # must never be swapped, recomputed, or reported as persistent KV.
        self.ledger.release_temporary(state)
        strategy = self.plan.scheduler.preemption_policy.lower()
        if strategy == "auto":
            strategy = self.plan.kv_policy.preemption_mode.lower()
        kv_swap_bytes = state.kv_pages * self.plan.kv_policy.bytes_per_page
        state_swap_bytes = (
            self.plan.linear_state_policy.bytes_per_request
            if state.linear_state_resident
            else 0
        )
        can_swap = kv_swap_bytes <= 0 or (
            self.plan.kv_policy.offload_component is not None
            and self.ledger.offload_used_bytes + kv_swap_bytes
            <= self.plan.kv_policy.offload_capacity_bytes
        )
        state_policy = self.plan.linear_state_policy
        can_state_swap = (
            state_swap_bytes <= 0
            or (
                state_policy.offload_component is not None
                and state_policy.offload_capacity_bytes
                - self.state_ledger.offload_used_bytes
                >= state_swap_bytes
            )
        )
        # Check the two role moves as one physical transaction.  A cache and
        # its host backing may intentionally share a component; destination-
        # first ``can_adjust`` would reject that capacity-neutral move.  A
        # scratch ledger also preserves the aggregate-capacity check when KV
        # and linear-state bytes share one destination.
        swap_check = _PhysicalCapacityLedger(self.physical_ledger.limits)
        swap_check.used_bytes.update(self.physical_ledger.used_bytes)
        can_physical_swap = True
        for source, target, byte_count in (
            (
                self.plan.kv_policy.cache_component,
                self.plan.kv_policy.offload_component,
                kv_swap_bytes,
            ),
            (
                state_policy.cache_component,
                state_policy.offload_component,
                state_swap_bytes,
            ),
        ):
            if byte_count <= 0:
                continue
            if not swap_check.transfer(source, target, byte_count):
                can_physical_swap = False
                break
        use_swap = (
            strategy in ("auto", "swap")
            and can_swap
            and can_state_swap
            and can_physical_swap
        )
        if strategy == "swap" and not use_swap:
            # An explicit swap policy is fail-closed.  Only ``auto`` may
            # choose recomputation when the declared offload path is full or
            # unavailable.
            return False
        swapped = False
        if use_swap:
            kv_offloaded = self.ledger.offload(state)
            state_offloaded = kv_offloaded and self.state_ledger.offload(state)
            swapped = kv_offloaded and state_offloaded
            if not swapped and kv_offloaded:
                if not self.ledger.restore(state):
                    raise RuntimeError("failed to roll back incomplete KV swap")
            if strategy == "swap" and not swapped:
                return False
        if swapped:
            state.preemption_strategy = "swap"
            state.swaps += 1
            self.swap_bytes += state.swap_bytes
            self.logical_swap_out_bytes += self._logical_bytes_for_physical(
                state.swap_bytes
            )
            swapped_pages = state.swapped_pages
            byte_count = state.swap_bytes
            state_byte_count = state.linear_state_swapped_bytes
            details = {
                "reason": reason,
                "pages": swapped_pages,
                "bytes": byte_count,
            }
            if byte_count > 0:
                self.swap_events += 1
                source = self.plan.kv_policy.cache_component
                target = self.plan.kv_policy.offload_component
                if not source or not target:
                    raise ValueError("KV swap requires cache and offload components")
                self.events.append(
                    ServingEvent(
                        self.now,
                        "kv_swap_out_start",
                        state.spec.request_id,
                        details=details,
                    )
                )
                self._execute_swap_transfer(
                    state,
                    "kv_swap_out",
                    source,
                    target,
                    byte_count,
                    swapped_pages,
                )
                event = "kv_swap_out"
            else:
                event = "linear_state_swap_out"
                details = {"reason": reason, "bytes": state_byte_count}
            if state_byte_count > 0:
                state_source = state_policy.cache_component
                state_target = state_policy.offload_component
                if not state_source or not state_target:
                    raise ValueError(
                        "linear state swap requires cache and offload components"
                    )
                state_details = {"bytes": state_byte_count}
                self.events.append(
                    ServingEvent(
                        self.now,
                        "linear_state_swap_out_start",
                        state.spec.request_id,
                        details=state_details,
                    )
                )
                self._execute_swap_transfer(
                    state,
                    "linear_state_swap_out",
                    state_source,
                    state_target,
                    state_byte_count,
                )
                if byte_count > 0:
                    self.events.append(
                        ServingEvent(
                            self.now,
                            "linear_state_swap_out",
                            state.spec.request_id,
                            details=state_details,
                        )
                    )
        else:
            if state.swap_bytes:
                self.ledger.discard_offload(state)
            self.ledger.resize(state, 0)
            self.state_ledger.release(state)
            if state.linear_state_swapped_bytes:
                self.state_ledger.discard_offload(state)
            state.preemption_strategy = "recompute"
            state.recompute_cursor = 0
            state.recompute_target = state.cached_tokens
            state.kv_cache_range_count = 0
            state.kv_cache_range_tokens.clear()
            state.kv_cache_range_error = None
            if self._last_prompt_cache_allocation_request_id == state.spec.request_id:
                self._last_prompt_cache_allocation_request_id = None
            if self._last_kv_allocation_request_id == state.spec.request_id:
                self._last_kv_allocation_request_id = None
            state.recomputes += 1
            self.recompute_events += 1
            event = "kv_recompute_required"
            details = {"reason": reason, "tokens": state.recompute_target}
        if self.prompt_cache.policy.save_implementation == "llama_cpp_host_tensor_get_combined":
            # A released slot can rewind the allocator into holes among live
            # peers. Logical append order cannot reconstruct their locations.
            for peer in self._state_values:
                if peer.status == RequestStatus.RUNNING:
                    peer.kv_cache_range_error = "unknown_physical_ranges_after_release"
            self._last_prompt_cache_allocation_request_id = None
        state.preemptions += 1
        self.preemptions += 1
        self._set_status(state, RequestStatus.SWAPPED)
        state.queued_since_ns = self.now
        self.events.append(ServingEvent(self.now, event, state.spec.request_id, details=details))
        self.events.append(ServingEvent(self.now, "request_preempted", state.spec.request_id, details={"reason": reason, "strategy": state.preemption_strategy}))
        return True

    def _execute_swap_transfer(
        self,
        state: _MutableRequest,
        kind: str,
        source_component: str,
        target_component: str,
        byte_count: int,
        page_count: int = 0,
    ) -> None:
        cohort_id = "cohort-{:06d}".format(len(self.batches))
        item = BatchItem(
            request_id=state.spec.request_id,
            phase=kind,
            token_count=0,
            context_tokens=state.cached_tokens,
        )
        cohort = BatchCohort(
            cohort_id=cohort_id,
            kind=kind,
            start_ns=self.now,
            items=(item,),
            metadata={
                "source_component": source_component,
                "target_component": target_component,
                "byte_count": byte_count,
                "page_count": page_count,
            },
        )
        cost = _lower_cost(self.lowerer, self.plan.scenario, cohort)
        end_ns = self.now + cost.duration_ns
        self.events.append(
            ServingEvent(
                self.now,
                "batch_start",
                cohort_id=cohort_id,
                details={
                    "kind": kind,
                    "request_ids": cohort.request_ids,
                    "bytes": byte_count,
                },
            )
        )
        self._append_batch(
            ServingBatch(
                cohort_id,
                kind,
                self.now,
                end_ns,
                cohort.request_ids,
                0,
                cost,
                cohort.items,
                cohort.proposal_cost_scale,
                cohort.metadata,
            )
        )
        self.events.append(
            ServingEvent(
                end_ns,
                "batch_end",
                cohort_id=cohort_id,
                details={"kind": kind, "bytes": byte_count},
            )
        )
        if kind.startswith("kv_"):
            self.swap_transfer_time_ns += cost.duration_ns
            self.swap_transfer_energy_pj += cost.energy_pj
        else:
            self.linear_state_swap_transfer_time_ns += cost.duration_ns
            self.linear_state_swap_transfer_energy_pj += cost.energy_pj
            self.linear_state_swap_routed_bytes += int(
                cost.metadata.get("resource_accounted_bytes", byte_count)
            )
        self._host_available_ns = max(self._host_available_ns, end_ns)
        self._gpu_available_ns = max(self._gpu_available_ns, end_ns)
        self._last_gpu_start_ns = self.now
        self._request_device_ready_ns[state.spec.request_id] = end_ns
        self.now = end_ns

    def _expected_acceptance(self, state: _MutableRequest, proposed: int) -> float:
        mtp = self.plan.mtp
        if proposed <= 1:
            return 1.0
        if mtp.acceptance_model == "trace":
            value = mtp.acceptance_trace[state.mtp_round % len(mtp.acceptance_trace)]
            rate = max(0.0, min(1.0, value))
        else:
            rate = mtp.acceptance_rate
        # ``proposed`` is verifier width.  The main token always commits and
        # the remaining values are draft-only prefix candidates.
        return 1.0 + expected_draft_prefix_tokens(proposed - 1, rate)

    def _record_kv_traffic(self, item: BatchItem) -> None:
        """Record logical KV traffic independently of resource demands.

        Planner kernels may already account local HBM traffic.  These counters
        therefore describe KV semantics and physical stored-byte volume; they
        are deliberately not added to resource-accounted bytes a second time.
        """

        # Prefill/recompute still performs attention against historical prompt
        # state; a fused chunk loads that persisted KV once through its first
        # lane and reuses it across the remaining query rows.  Decode/MTP
        # retain their per-query external persisted-KV access semantics.
        read_tokens = (
            max(0, item.context_tokens)
            if item.phase in {"prefill", "recompute"}
            else max(0, item.context_tokens) * max(0, item.token_count)
        )
        append_tokens = _batch_item_kv_append_tokens(item)
        materialized_tokens = _batch_item_kv_materialized_tokens(item)
        logical_per_token = max(0, self.plan.kv_policy.logical_bytes_per_token)
        physical_per_token = (
            self.plan.kv_policy.bytes_per_page
            // self.plan.kv_policy.tokens_per_page
            if self.plan.kv_policy.tokens_per_page > 0
            else 0
        )
        if item.phase in {"prefill", "recompute"}:
            self.logical_prefill_read_bytes += read_tokens * logical_per_token
            self.logical_prefill_write_bytes += append_tokens * logical_per_token
            self.physical_prefill_read_bytes += read_tokens * physical_per_token
            self.physical_prefill_write_bytes += append_tokens * physical_per_token
        else:
            self.logical_decode_read_bytes += read_tokens * logical_per_token
            self.logical_decode_write_bytes += append_tokens * logical_per_token
            self.physical_decode_read_bytes += read_tokens * physical_per_token
            self.physical_decode_write_bytes += append_tokens * physical_per_token
            if item.phase == "mtp":
                temporary_tokens = max(
                    0, materialized_tokens - append_tokens
                )
                verification_read_tokens = (
                    max(0, int(item.token_count))
                    * (max(0, int(item.token_count)) + 1)
                    // 2
                )
                self.mtp_materialized_tokens += materialized_tokens
                self.mtp_temporary_tokens += temporary_tokens
                self.logical_mtp_materialized_write_bytes += (
                    materialized_tokens * logical_per_token
                )
                self.physical_mtp_materialized_write_bytes += (
                    materialized_tokens * physical_per_token
                )
                self.logical_mtp_temporary_write_bytes += (
                    temporary_tokens * logical_per_token
                )
                self.physical_mtp_temporary_write_bytes += (
                    temporary_tokens * physical_per_token
                )
                self.logical_mtp_verification_read_bytes += (
                    verification_read_tokens * logical_per_token
                )
                self.physical_mtp_verification_read_bytes += (
                    verification_read_tokens * physical_per_token
                )

    def _logical_bytes_for_physical(self, physical_bytes: int) -> int:
        physical_per_token = (
            self.plan.kv_policy.bytes_per_page
            // self.plan.kv_policy.tokens_per_page
            if self.plan.kv_policy.tokens_per_page > 0
            else 0
        )
        if physical_per_token <= 0:
            return 0
        return int(
            math.ceil(
                max(0, physical_bytes)
                * self.plan.kv_policy.logical_bytes_per_token
                / float(physical_per_token)
            )
        )

    @staticmethod
    def _runtime_stage_extension_ns(
        metadata: Mapping[str, Any],
        device_ns: float,
    ) -> float:
        if (
            metadata.get("execution_stages_include_host_orchestration") is True
            and metadata.get("execution_stage_source")
            == "executed_task_dag_kernel_timeline"
        ):
            base_stage_ns = max(
                0.0,
                float(metadata.get("execution_stage_makespan_ns", 0.0)),
            )
            return max(0.0, device_ns - base_stage_ns)
        owner_ns = max(
            0.0, float(metadata.get("owner_residency_transfer_ns", 0.0))
        )
        if "resource_contention_base_device_ns" not in metadata:
            return owner_ns
        base_ns = max(
            0.0,
            float(metadata.get("resource_contention_base_device_ns", 0.0)),
        )
        contention_ns = max(0.0, device_ns - owner_ns - base_ns)
        return owner_ns + contention_ns

    @staticmethod
    def _inject_owner_residency_transfer_stages(
        stages: Sequence[_ExecutionStage],
        metadata: Mapping[str, Any],
    ) -> Tuple[Tuple[_ExecutionStage, ...], float, int]:
        """Place each attributed transfer before its earliest covered group.

        This is the causal placement policy selected for owner residency.  It
        changes only where already-accounted transfer service sits in the
        execution DAG; transfer bytes, service, and energy remain untouched.
        A batch is left for the legacy cohort-tail extension unless every
        covered invocation group has a concrete planner stage.
        """

        if (
            metadata.get("owner_residency_enabled") is not True
            or metadata.get("owner_residency_transfer_placement_policy")
            != "before_earliest_covered_invocation_group"
        ):
            return tuple(stages), 0.0, 0
        raw_batches = metadata.get("owner_residency_transfer_batches", ())
        if raw_batches is None:
            raw_batches = ()
        if isinstance(raw_batches, (str, bytes, _ABCMapping)) or not isinstance(
            raw_batches, _ABCSequence
        ):
            raise ValueError(
                "owner_residency_transfer_batches must be an ordered sequence"
            )
        if not raw_batches:
            return tuple(stages), 0.0, 0

        trusted_planner_stages = (
            metadata.get("execution_stage_source")
            == "executed_task_dag_kernel_timeline"
            and _trusted_execution_stages(
                metadata.get("execution_stages")
            )
            is not None
        )

        def fail_trusted_page_in_placement(reason: str) -> None:
            if trusted_planner_stages:
                raise ValueError(
                    "trusted owner-residency PAGE_IN cannot be causally "
                    "placed: {}".format(reason)
                )

        if not stages:
            for raw_batch in raw_batches:
                if not isinstance(raw_batch, _ABCMapping):
                    raise ValueError(
                        "owner residency transfer batch must be a mapping"
                    )
                if (
                    str(raw_batch.get("kind", ""))
                    == MigrationKind.PAGE_IN.value
                    and float(raw_batch.get("transfer_ns", 0.0)) > 0.0
                ):
                    fail_trusted_page_in_placement(
                        "execution stage graph is empty"
                    )
            return tuple(stages), 0.0, 0

        first_position_by_group: Dict[str, int] = {}
        stage_id_by_task_id: Dict[str, str] = {}
        for position, stage in enumerate(stages):
            for task in stage.execution_tasks:
                stage_id_by_task_id[task.task_id] = stage.stage_id
            for group_id in stage.causal_invocation_group_ids:
                if group_id not in first_position_by_group:
                    first_position_by_group[group_id] = position
        if not first_position_by_group:
            for raw_batch in raw_batches:
                if not isinstance(raw_batch, _ABCMapping):
                    raise ValueError(
                        "owner residency transfer batch must be a mapping"
                    )
                if (
                    str(raw_batch.get("kind", ""))
                    == MigrationKind.PAGE_IN.value
                    and float(raw_batch.get("transfer_ns", 0.0)) > 0.0
                ):
                    fail_trusted_page_in_placement(
                        "execution stage graph has no invocation groups"
                    )
            return tuple(stages), 0.0, 0

        original_stages = tuple(stages)
        stage_dependencies_by_id = {
            stage.stage_id: tuple(stage.dependencies)
            for stage in original_stages
        }
        prior_anchor_ancestor_ids: set[str] = set()

        def record_strict_ancestors(stage_id: str) -> None:
            pending = list(stage_dependencies_by_id.get(stage_id, ()))
            while pending:
                dependency_id = pending.pop()
                if dependency_id in prior_anchor_ancestor_ids:
                    continue
                prior_anchor_ancestor_ids.add(dependency_id)
                pending.extend(
                    stage_dependencies_by_id.get(dependency_id, ())
                )

        insertions: Dict[int, List[_ExecutionStage]] = {}
        dependency_additions: Dict[str, List[str]] = {}
        last_placed_transfer_id: Optional[str] = None
        placement_blocked = False
        existing_stage_ids = {stage.stage_id for stage in original_stages}
        next_stage_index = max(
            (stage.stage_index for stage in original_stages),
            default=-1,
        ) + 1
        placed_service_ns = 0.0
        placed_batch_count = 0

        for row_position, raw_batch in enumerate(raw_batches):
            if not isinstance(raw_batch, _ABCMapping):
                raise ValueError(
                    "owner residency transfer batch must be a mapping"
                )
            raw_service_ns = raw_batch.get("transfer_ns", 0.0)
            if isinstance(raw_service_ns, bool):
                raise ValueError(
                    "owner residency transfer batch service is invalid"
                )
            try:
                service_ns = float(raw_service_ns)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "owner residency transfer batch service is invalid"
                ) from exc
            if not math.isfinite(service_ns) or service_ns < 0.0:
                raise ValueError(
                    "owner residency transfer batch service is invalid"
                )
            if service_ns == 0.0:
                continue
            if placement_blocked:
                if str(raw_batch.get("kind", "")) == MigrationKind.PAGE_IN.value:
                    fail_trusted_page_in_placement(
                        "a preceding positive-service migration has no safe "
                        "DAG position"
                    )
                continue
            raw_group_ids = raw_batch.get(
                "operator_invocation_group_ids", ()
            )
            if isinstance(
                raw_group_ids, (str, bytes, _ABCMapping)
            ) or not isinstance(raw_group_ids, _ABCSequence):
                raise ValueError(
                    "owner residency transfer batch group ids are invalid"
                )
            group_ids = tuple(dict.fromkeys(str(item) for item in raw_group_ids))
            if any(not group_id for group_id in group_ids):
                raise ValueError(
                    "owner residency transfer batch group ids are invalid"
                )
            missing_group_ids = tuple(
                group_id
                for group_id in group_ids
                if group_id not in first_position_by_group
            )
            # Unknown/trace-only groups cannot be blocked exactly.  Custom
            # lowerers retain the explicit degraded tail path.  A trusted V4
            # PAGE_IN must fail closed because executing it after the
            # consumer would violate residency causality.
            if not group_ids or missing_group_ids:
                if str(raw_batch.get("kind", "")) == MigrationKind.PAGE_IN.value:
                    fail_trusted_page_in_placement(
                        "missing group ids {}".format(
                            missing_group_ids or ("<unattributed>",)
                        )
                    )
                placement_blocked = True
                continue

            earliest_position = min(
                first_position_by_group[group_id]
                for group_id in group_ids
            )
            covered_stage_ids = tuple(
                original_stages[first_position_by_group[group_id]].stage_id
                for group_id in group_ids
            )
            if any(
                stage_id in prior_anchor_ancestor_ids
                for stage_id in covered_stage_ids
            ):
                # Transfers retain global migration order.  Adding the current
                # transfer before a consumer that is an ancestor of any prior
                # anchor would therefore close a dependency cycle.  Stage-list
                # position alone is not causal: independent request branches
                # are routinely interleaved by round, while residency events
                # are emitted request-by-request.
                if str(raw_batch.get("kind", "")) == MigrationKind.PAGE_IN.value:
                    fail_trusted_page_in_placement(
                        "consumer order regresses behind an earlier migration "
                        "dependency"
                    )
                placement_blocked = True
                continue
            earliest_stage = original_stages[earliest_position]
            stage_id = "runtime.owner_residency_transfer.batch{:04d}".format(
                row_position
            )
            if stage_id in existing_stage_ids:
                raise ValueError(
                    "owner residency transfer stage id collides with planner stage"
                )
            existing_stage_ids.add(stage_id)
            raw_consumer_task_ids = raw_batch.get("consumer_task_ids", ())
            if isinstance(
                raw_consumer_task_ids, (str, bytes, _ABCMapping)
            ) or not isinstance(raw_consumer_task_ids, _ABCSequence):
                raise ValueError(
                    "owner residency transfer consumer task ids are invalid"
                )
            consumer_task_ids = tuple(
                dict.fromkeys(str(item) for item in raw_consumer_task_ids)
            )
            if any(not task_id for task_id in consumer_task_ids):
                raise ValueError(
                    "owner residency transfer consumer task ids are invalid"
                )
            missing_consumer_task_ids = tuple(
                task_id
                for task_id in consumer_task_ids
                if task_id not in stage_id_by_task_id
            )
            current_consumer_task_ids = tuple(
                task_id
                for task_id in consumer_task_ids
                if task_id in stage_id_by_task_id
            )
            consumer_stage_ids = tuple(
                dict.fromkeys(
                    stage_id_by_task_id[task_id]
                    for task_id in current_consumer_task_ids
                )
            )
            dependency_consumer_stage_ids = tuple(
                stage_id
                for stage_id in consumer_stage_ids
                if stage_id != earliest_stage.stage_id
            )
            same_stage_consumer_task_ids = tuple(
                task_id
                for task_id in current_consumer_task_ids
                if stage_id_by_task_id[task_id] == earliest_stage.stage_id
            )
            if same_stage_consumer_task_ids and str(
                raw_batch.get("kind", "")
            ) in {
                MigrationKind.CLEAN_DISCARD.value,
                MigrationKind.DIRTY_WRITEBACK.value,
            }:
                # A same-stage PAGE_OUT cannot be inserted before the stage
                # that still needs the bytes.  Leave the positive service to
                # the conservative device-extension tail rather than making
                # the graph acyclic by violating the lease's consumer order.
                placement_blocked = True
                continue
            dependencies = tuple(
                dict.fromkeys(
                    (
                        *earliest_stage.dependencies,
                        *dependency_consumer_stage_ids,
                        *(
                            (last_placed_transfer_id,)
                            if last_placed_transfer_id is not None
                            else ()
                        ),
                    )
                )
            )
            request_ids = tuple(
                dict.fromkeys(
                    request_id
                    for group_id in group_ids
                    for request_id in original_stages[
                        first_position_by_group[group_id]
                    ].request_ids
                )
            )
            raw_resource_phases = raw_batch.get("resource_phases")
            transfer_tasks: List[_ExecutionTask] = []
            if raw_resource_phases is not None:
                if isinstance(
                    raw_resource_phases, (str, bytes, _ABCMapping)
                ) or not isinstance(raw_resource_phases, _ABCSequence):
                    raise ValueError(
                        "owner residency transfer resource phases are invalid"
                    )
                previous_task_id: Optional[str] = None
                phase_service_total = 0.0
                for phase_index, raw_phase in enumerate(raw_resource_phases):
                    if isinstance(
                        raw_phase, (str, bytes, _ABCMapping)
                    ) or not isinstance(raw_phase, _ABCSequence):
                        raise ValueError(
                            "owner residency transfer resource phase is invalid"
                        )
                    demands: List[ResourceDemand] = []
                    seen_resources = set()
                    for raw_demand in raw_phase:
                        if not isinstance(raw_demand, _ABCMapping):
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            )
                        resource_id = raw_demand.get("resource_id")
                        raw_demand_ns = raw_demand.get("service_ns")
                        if (
                            not isinstance(resource_id, str)
                            or not resource_id
                            or resource_id in seen_resources
                            or isinstance(raw_demand_ns, bool)
                        ):
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            )
                        try:
                            demand_ns = float(raw_demand_ns)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            ) from exc
                        if not math.isfinite(demand_ns) or demand_ns < 0.0:
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            )
                        raw_bytes_moved = raw_demand.get("bytes_moved", 0)
                        if (
                            isinstance(raw_bytes_moved, bool)
                            or not isinstance(raw_bytes_moved, int)
                            or raw_bytes_moved < 0
                        ):
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            )
                        raw_energy_pj = raw_demand.get("energy_pj", 0.0)
                        raw_work_units = raw_demand.get("work_units", 0.0)
                        if isinstance(raw_energy_pj, bool) or isinstance(
                            raw_work_units, bool
                        ):
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            )
                        try:
                            energy_pj = float(raw_energy_pj)
                            work_units = float(raw_work_units)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            ) from exc
                        if (
                            not math.isfinite(energy_pj)
                            or energy_pj < 0.0
                            or not math.isfinite(work_units)
                            or work_units < 0.0
                        ):
                            raise ValueError(
                                "owner residency transfer demand is invalid"
                            )
                        demands.append(
                            ResourceDemand(
                                resource_id,
                                demand_ns,
                                bytes_moved=raw_bytes_moved,
                                energy_pj=energy_pj,
                                work_units=work_units,
                            )
                        )
                        seen_resources.add(resource_id)
                    phase_service_ns = max(
                        (demand.service_ns for demand in demands),
                        default=0.0,
                    )
                    logical_join = bool(demands) and all(
                        raw_demand.get("logical_resource") is True
                        and raw_demand.get("logical_join") is True
                        and demand.service_ns == 0.0
                        for demand, raw_demand in zip(demands, raw_phase)
                    )
                    if phase_service_ns <= 0.0 and not logical_join:
                        continue
                    task_id = "{}.task{:04d}".format(stage_id, phase_index)
                    transfer_tasks.append(
                        _ExecutionTask(
                            task_id=task_id,
                            dependencies=(
                                (previous_task_id,)
                                if previous_task_id is not None
                                else ()
                            ),
                            request_ids=request_ids,
                            demands=tuple(demands),
                            opaque_device_fence=True,
                            category=TaskCategory.COMMUNICATION,
                            metadata={
                                "event_kind": (
                                    "owner_residency_transfer_phase"
                                ),
                                "runtime_phase": (
                                    "owner_residency_transfer"
                                ),
                                "movement_id": raw_batch.get(
                                    "movement_id",
                                    "owner-residency-migration-batch{:04d}".format(
                                        row_position
                                    ),
                                ),
                                "migration_kind": raw_batch.get("kind"),
                                "consumer_task_ids": consumer_task_ids,
                                "current_consumer_task_ids": (
                                    current_consumer_task_ids
                                ),
                                "already_satisfied_consumer_task_ids": (
                                    missing_consumer_task_ids
                                ),
                                "same_stage_consumer_task_ids": (
                                    same_stage_consumer_task_ids
                                ),
                                "service_domains": tuple(
                                    {
                                        "resource_id": demand.resource_id,
                                        "service_domain": raw_demand.get(
                                            "service_domain",
                                            demand.resource_id,
                                        ),
                                        "service_role": raw_demand.get(
                                            "service_role",
                                            "bulk_owner",
                                        ),
                                        "bulk_service_owner": bool(
                                            raw_demand.get(
                                                "bulk_service_owner", False
                                            )
                                        ),
                                    }
                                    for demand, raw_demand in zip(
                                        demands, raw_phase
                                    )
                                ),
                            },
                        )
                    )
                    previous_task_id = task_id
                    phase_service_total += phase_service_ns
                if transfer_tasks and not math.isclose(
                    phase_service_total,
                    service_ns,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-6,
                ):
                    raise ValueError(
                        "owner residency transfer phases do not conserve service"
                    )
            if not transfer_tasks:
                transfer_tasks.append(
                    _ExecutionTask(
                        task_id=stage_id + ".task0000",
                        dependencies=(),
                        request_ids=request_ids,
                        demands=(
                            ResourceDemand(
                                earliest_stage.component_id,
                                service_ns,
                            ),
                        ),
                        opaque_device_fence=True,
                        category=TaskCategory.COMMUNICATION,
                        metadata={
                            "event_kind": "owner_residency_transfer_phase",
                            "runtime_phase": "owner_residency_transfer",
                            "movement_id": raw_batch.get(
                                "movement_id",
                                "owner-residency-migration-batch{:04d}".format(
                                    row_position
                                ),
                            ),
                            "migration_kind": raw_batch.get("kind"),
                            "consumer_task_ids": consumer_task_ids,
                            "current_consumer_task_ids": (
                                current_consumer_task_ids
                            ),
                            "already_satisfied_consumer_task_ids": (
                                missing_consumer_task_ids
                            ),
                            "same_stage_consumer_task_ids": (
                                same_stage_consumer_task_ids
                            ),
                            "service_domains": (),
                            "service_ownership": "legacy_unclassified",
                        },
                    )
                )
            transfer_stage = _ExecutionStage(
                stage_id,
                next_stage_index + placed_batch_count,
                dependencies,
                request_ids,
                earliest_stage.component_id,
                service_ns,
                tuple(transfer_tasks),
                None,
                group_ids,
            )
            insertions.setdefault(earliest_position, []).append(
                transfer_stage
            )
            # Residency mutations are produced in capacity-causal order: an
            # eviction that makes room must complete before the later page-in
            # can consume that room.  Preserve that order across invocation
            # groups even when their topology resources would otherwise let
            # the event kernel overlap opposite-direction transfers.
            last_placed_transfer_id = stage_id
            record_strict_ancestors(earliest_stage.stage_id)
            for group_id in group_ids:
                first_stage = original_stages[
                    first_position_by_group[group_id]
                ]
                additions = dependency_additions.setdefault(
                    first_stage.stage_id, []
                )
                if stage_id not in additions:
                    additions.append(stage_id)
            placed_service_ns += service_ns
            placed_batch_count += 1

        if placed_batch_count == 0:
            return original_stages, 0.0, 0

        scheduled: List[_ExecutionStage] = []
        for position, stage in enumerate(original_stages):
            scheduled.extend(insertions.get(position, ()))
            additions = dependency_additions.get(stage.stage_id, ())
            if additions:
                stage = _ExecutionStage(
                    stage.stage_id,
                    stage.stage_index,
                    tuple(dict.fromkeys((*stage.dependencies, *additions))),
                    stage.request_ids,
                    stage.component_id,
                    stage.service_ns,
                    stage.execution_tasks,
                    stage.invocation_group_id,
                    stage.covered_invocation_group_ids,
                )
            scheduled.append(stage)
        return tuple(scheduled), placed_service_ns, placed_batch_count

    @staticmethod
    def _stage_task_specs(
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        namespace: str,
    ) -> Tuple[Tuple[TaskSpec, ...], Mapping[str, Tuple[str, _ExecutionTask]]]:
        """Build one closed, namespaced task graph for a cohort's stages."""

        namespaced_id: Dict[Tuple[str, str], str] = {}
        terminal_ids_by_stage: Dict[str, Tuple[str, ...]] = {}
        for stage_position, stage in enumerate(stages):
            prefix = "{}.stage{:04d}.".format(namespace, stage_position)
            for task_position, task in enumerate(stage.execution_tasks):
                namespaced_id[(stage.stage_id, task.task_id)] = (
                    "{}task{:06d}".format(prefix, task_position)
                )
            dependency_ids = {
                dependency
                for task in stage.execution_tasks
                for dependency in task.dependencies
            }
            terminal_ids_by_stage[stage.stage_id] = tuple(
                namespaced_id[(stage.stage_id, task.task_id)]
                for task in stage.execution_tasks
                if task.task_id not in dependency_ids
            )

        specs: List[TaskSpec] = []
        source_by_id: Dict[str, Tuple[str, _ExecutionTask]] = {}
        for stage in stages:
            stage_parent_ids = tuple(
                terminal_id
                for dependency_stage_id in stage.dependencies
                for terminal_id in terminal_ids_by_stage[dependency_stage_id]
            )
            for task in stage.execution_tasks:
                task_id = namespaced_id[(stage.stage_id, task.task_id)]
                dependencies = tuple(
                    namespaced_id[(stage.stage_id, dependency)]
                    for dependency in task.dependencies
                )
                if not dependencies:
                    dependencies = stage_parent_ids
                specs.append(
                    TaskSpec(
                        task_id=task_id,
                        request_id=task.request_ids[0],
                        name=task.task_id,
                        category=task.category,
                        dependencies=dependencies,
                        demands=task.demands,
                        earliest_start_ns=ready_by_stage[stage.stage_id],
                        metadata={
                            **dict(task.metadata),
                            "execution_stage_id": stage.stage_id,
                        },
                    )
                )
                source_by_id[task_id] = (stage.stage_id, task)
        return tuple(specs), source_by_id

    @classmethod
    def _replay_execution_stage_tasks(
        cls,
        kernel: UnifiedEventKernel,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        namespace: str,
        isolated_shadow: Optional[_UniformIsolatedReplay] = None,
    ) -> Mapping[str, Tuple[float, float]]:
        specs, source_by_id = cls._stage_task_specs(
            stages, ready_by_stage, namespace
        )
        if isolated_shadow is not None:
            isolated_shadow.bind(specs)
        kernel.add_tasks(specs)
        min_start_by_stage: Dict[str, float] = {}
        max_end_by_stage: Dict[str, float] = {}
        remaining = len(specs)
        completed_ids: List[str] = []
        while remaining:
            event = kernel.step()
            if event is None:
                raise ValueError("execution stage task graph could not drain")
            if isolated_shadow is not None:
                isolated_shadow.observe(event)
            stage_id, _source = source_by_id[event.task.task_id]
            min_start_by_stage[stage_id] = min(
                min_start_by_stage.get(stage_id, event.start_ns),
                event.start_ns,
            )
            max_end_by_stage[stage_id] = max(
                max_end_by_stage.get(stage_id, event.end_ns),
                event.end_ns,
            )
            completed_ids.append(event.task.task_id)
            remaining -= 1
        for task_id in completed_ids:
            kernel.release_completed(task_id)
        return {
            stage.stage_id: (
                min_start_by_stage.get(
                    stage.stage_id, ready_by_stage[stage.stage_id]
                ),
                max_end_by_stage.get(
                    stage.stage_id, ready_by_stage[stage.stage_id]
                ),
            )
            for stage in stages
        }

    @staticmethod
    def _replay_prevalidated_execution_layout(
        kernel: UnifiedEventKernel,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        replay_layout: _ExecutionStageReplayLayout,
        specs: Optional[Sequence[TaskSpec]],
        isolated_shadow: Optional[_UniformIsolatedReplay] = None,
        *,
        namespace: Optional[str] = None,
    ) -> Optional[Mapping[str, Tuple[float, float]]]:
        """Drain a trusted layout directly to exact stage envelopes.

        This is the compiled bulk kernel's scheduling algorithm with the same
        ready keys, demand order, lane selection, floating-point operations,
        and persistent metric updates.  The online caller immediately releases
        every completion and retains no per-task events, so this specialization
        omits only those transient ``KernelEvent``/lease objects and their
        second traversal.
        """

        if kernel.has_active_tasks:
            return None
        layout = replay_layout.compiled
        task_count = layout.task_count
        stage_positions = replay_layout.stage_positions
        if (
            len(replay_layout.tasks) != task_count
            or len(stage_positions) != task_count
            or len(replay_layout.indegree_template) != task_count
            or len(replay_layout.demand_resource_indices) != task_count
            or len(replay_layout.group_indices) != task_count
            or len(replay_layout.base_task_ids) != task_count
            or len(replay_layout.phase_sequence) != task_count
        ):
            return None
        if specs is None:
            if isolated_shadow is not None or namespace is None:
                return None
            try:
                stage_ids = tuple(stage.stage_id for stage in stages)
                earliest_start_by_stage = tuple(
                    float(ready_by_stage[stage_id]) for stage_id in stage_ids
                )
                service_ns_by_position = tuple(
                    tuple(
                        demand.service_ns
                        for demand in stages[item.stage_position].execution_tasks[
                            item.task_position
                        ].demands
                    )
                    for item in replay_layout.tasks
                )
            except (
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                return None
            prefix = namespace + "."
            # This path is reachable only for a trusted, structure-keyed
            # layout and an internally allocated monotonic namespace.  The
            # prefix is common to every task, so relative ids preserve the
            # exact lexicographic tie order without materializing tens of
            # thousands of namespaced strings per cohort.  The kernel keeps
            # the completed-id contract through its compact namespace
            # registry; later public submission of any expanded id is still
            # rejected as a duplicate.
            task_ids = replay_layout.base_task_ids
            compact_seen_namespace = True
            relative_task_ids = True
            earliest_start_ns = None
            phase_sequence = replay_layout.phase_sequence
        else:
            chunk = tuple(specs)
            if len(chunk) != task_count:
                return None
            stage_ids = tuple(stage.stage_id for stage in stages)
            task_ids = tuple(task.task_id for task in chunk)
            earliest_start_ns = tuple(task.earliest_start_ns for task in chunk)
            earliest_start_by_stage = None
            service_ns_by_position = tuple(
                tuple(demand.service_ns for demand in task.demands)
                for task in chunk
            )
            phase_sequence = tuple(
                _validated_phase_sequence(task) for task in chunk
            )
            if phase_sequence != replay_layout.phase_sequence:
                return None
            if namespace is None:
                prefix = ""
                compact_seen_namespace = False
            else:
                prefix = namespace + "."
                if any(
                    not task_id.startswith(prefix)
                    or task_id[len(prefix) :] != base_task_id
                    for task_id, base_task_id in zip(
                        task_ids,
                        replay_layout.base_task_ids,
                    )
                ):
                    return None
                compact_seen_namespace = True
            relative_task_ids = False
        demand_resource_indices = replay_layout.demand_resource_indices
        if any(
            len(service_ns) != len(resource_indices)
            for service_ns, resource_indices in zip(
                service_ns_by_position,
                demand_resource_indices,
            )
        ):
            return None
        compact_seen_registration = None
        if compact_seen_namespace:
            compact_seen_registration = kernel._prepare_compact_seen_namespace(
                prefix,
                replay_layout.base_task_ids,
            )
        else:
            if len(set(task_ids)) != task_count:
                raise ValueError("task chunk contains duplicate task ids")
            duplicate = next(
                (
                    task_id
                    for task_id in task_ids
                    if kernel._has_seen_task_id(task_id)
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError("duplicate task_id: {}".format(duplicate))

        indegree = list(replay_layout.indegree_template)
        dependency_ready = [0.0] * task_count
        ready_by_group: List[Optional[List[Tuple[Any, ...]]]] = [
            None
        ] * replay_layout.group_count
        group_versions = [0] * replay_layout.group_count
        ready_heap: List[Tuple[Any, ...]] = []
        resource_ids = replay_layout.resource_ids
        resource_available = kernel.resource_available
        resource_available_values = [
            resource_available.get(resource_id, 0.0)
            for resource_id in resource_ids
        ]
        resource_busy_ns = kernel.resource_busy_ns
        resource_busy_values = [
            resource_busy_ns.get(resource_id, 0.0)
            for resource_id in resource_ids
        ]
        resource_queue_wait_ns = kernel.resource_queue_wait_ns
        resource_queue_wait_values = [
            resource_queue_wait_ns.get(resource_id, 0.0)
            for resource_id in resource_ids
        ]
        resource_task_count = kernel.resource_task_count
        resource_task_count_values = [
            resource_task_count.get(resource_id, 0)
            for resource_id in resource_ids
        ]
        resource_lanes = [
            kernel._resource_lane_available.get(resource_id)
            for resource_id in resource_ids
        ]
        resource_lane_intervals: List[
            Optional[List[Optional[Dict[str, object]]]]
        ] = [None] * len(resource_ids)
        resource_touched = [False] * len(resource_ids)

        def static_key(position: int) -> Tuple[Any, ...]:
            dependency_ready_ns = dependency_ready[position]
            release_ns = (
                earliest_start_ns[position]
                if earliest_start_ns is not None
                else earliest_start_by_stage[stage_positions[position]]
            )
            effective_ready_ns = max(
                dependency_ready_ns,
                release_ns,
            )
            phase, sequence = phase_sequence[position]
            return (
                effective_ready_ns,
                dependency_ready_ns,
                release_ns,
                phase,
                sequence,
                task_ids[position],
                position,
            )

        def ready_key(position: int) -> Tuple[Any, ...]:
            dependency_ready_ns = dependency_ready[position]
            release_ns = (
                earliest_start_ns[position]
                if earliest_start_ns is not None
                else earliest_start_by_stage[stage_positions[position]]
            )
            effective_ready_ns = max(
                dependency_ready_ns,
                release_ns,
            )
            resources_ready_ns = 0.0
            for resource_index in demand_resource_indices[position]:
                available_ns = resource_available_values[resource_index]
                if available_ns > resources_ready_ns:
                    resources_ready_ns = available_ns
            phase, sequence = phase_sequence[position]
            return (
                max(effective_ready_ns, resources_ready_ns),
                effective_ready_ns,
                dependency_ready_ns,
                release_ns,
                phase,
                sequence,
                task_ids[position],
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
                (*ready_key(position), group_index, version, position),
            )

        def add_ready(position: int) -> Optional[int]:
            group_index = replay_layout.group_indices[position]
            group_ready = ready_by_group[group_index]
            if group_ready is None:
                group_ready = []
                ready_by_group[group_index] = group_ready
            previous_top = group_ready[0] if group_ready else None
            heapq.heappush(group_ready, static_key(position))
            if previous_top is None or group_ready[0] != previous_top:
                return group_index
            return None

        initial_groups = set()
        for position in layout.root_positions:
            changed_group = add_ready(position)
            if changed_group is not None:
                initial_groups.add(changed_group)
        for group_index in sorted(initial_groups):
            refresh_group(group_index)

        min_start_by_stage: List[Optional[float]] = [None] * len(stages)
        max_end_by_stage: List[Optional[float]] = [None] * len(stages)
        resource_last_interval = kernel.resource_last_interval
        completed_count = 0
        while ready_heap:
            queued = heapq.heappop(ready_heap)
            group_index = queued[7]
            version = queued[8]
            position = queued[9]
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
                _dependency_ready_ns,
                _earliest_start_ns,
                _phase,
                _sequence,
                task_id,
            ) = current_key
            service_ns_values = service_ns_by_position[position]
            resource_indices = demand_resource_indices[position]
            demand_order = layout.demand_order[position]
            timing_is_finite = math.isfinite(start_ns)
            if timing_is_finite:
                for demand_position in demand_order:
                    if not math.isfinite(
                        start_ns + service_ns_values[demand_position]
                    ):
                        timing_is_finite = False
                        break
            if not timing_is_finite:
                raise ValueError(
                    "task {} timing exceeds finite simulation range".format(
                        (prefix + task_id) if relative_task_ids else task_id
                    )
                )
            heapq.heappop(group_ready)

            end_ns = start_ns
            queue_wait_ns = start_ns - effective_ready_ns
            for demand_position in demand_order:
                service_ns = service_ns_values[demand_position]
                resource_index = resource_indices[demand_position]
                resource_id = resource_ids[resource_index]
                demand_end_ns = start_ns + service_ns
                lanes = resource_lanes[resource_index]
                if lanes is None:
                    lanes = kernel._lanes_for(resource_id)
                    resource_lanes[resource_index] = lanes
                if len(lanes) == 1:
                    lane_index = 0
                else:
                    lane_index = min(
                        range(len(lanes)),
                        key=lambda index: (lanes[index], index),
                    )
                if demand_end_ns > end_ns:
                    end_ns = demand_end_ns
                lane_intervals = resource_lane_intervals[resource_index]
                if lane_intervals is None:
                    lane_intervals = [None] * len(lanes)
                    resource_lane_intervals[resource_index] = lane_intervals
                interval = lane_intervals[lane_index]
                if interval is None:
                    interval = {
                        "task_id": task_id,
                        "resource_id": resource_id,
                        "start_ns": start_ns,
                        "end_ns": demand_end_ns,
                    }
                    if len(lanes) > 1:
                        interval["lane"] = lane_index
                    lane_intervals[lane_index] = interval
                else:
                    interval["task_id"] = task_id
                    interval["start_ns"] = start_ns
                    interval["end_ns"] = demand_end_ns
                lanes[lane_index] = demand_end_ns
                resource_available_values[resource_index] = (
                    demand_end_ns if len(lanes) == 1 else min(lanes)
                )
                resource_last_interval[resource_id] = interval
                kernel._resource_lane_last_interval[
                    (resource_id, lane_index)
                ] = interval
                resource_busy_values[resource_index] += service_ns
                resource_queue_wait_values[resource_index] += queue_wait_ns
                resource_task_count_values[resource_index] += 1
                if not resource_touched[resource_index]:
                    resource_touched[resource_index] = True
                    # Preserve the public dictionaries' first-touch insertion
                    # order.  Their values are synchronized after the closed
                    # graph drains; no callback can observe this private bulk
                    # specialization mid-drain.
                    resource_available[resource_id] = (
                        resource_available_values[resource_index]
                    )
                    resource_busy_ns[resource_id] = (
                        resource_busy_values[resource_index]
                    )
                    resource_queue_wait_ns[resource_id] = (
                        resource_queue_wait_values[resource_index]
                    )
                    resource_task_count[resource_id] = (
                        resource_task_count_values[resource_index]
                    )

            kernel.completed_count += 1
            kernel.total_queue_wait_ns += queue_wait_ns
            kernel.total_service_ns += end_ns - start_ns
            if end_ns > kernel.makespan_ns:
                kernel.makespan_ns = end_ns
            stage_position = stage_positions[position]
            current_min_start = min_start_by_stage[stage_position]
            if current_min_start is None or start_ns < current_min_start:
                min_start_by_stage[stage_position] = start_ns
            current_max_end = max_end_by_stage[stage_position]
            if current_max_end is None or end_ns > current_max_end:
                max_end_by_stage[stage_position] = end_ns
            if isolated_shadow is not None:
                isolated_shadow.observe_values(task_id, start_ns, end_ns)

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
            completed_count += 1

        if completed_count != task_count:
            raise ValueError("schedule contains a dependency cycle")
        for resource_index, touched in enumerate(resource_touched):
            if not touched:
                continue
            resource_id = resource_ids[resource_index]
            resource_available[resource_id] = resource_available_values[
                resource_index
            ]
            resource_busy_ns[resource_id] = resource_busy_values[resource_index]
            resource_queue_wait_ns[resource_id] = (
                resource_queue_wait_values[resource_index]
            )
            resource_task_count[resource_id] = resource_task_count_values[
                resource_index
            ]
        if relative_task_ids:
            # Only the final interval per resource lane is externally
            # retained.  Expand those few ids after the closed replay drains;
            # all scheduling comparisons above remain exactly equivalent
            # because every relative id had the same omitted prefix.
            for lane_intervals in resource_lane_intervals:
                if lane_intervals is None:
                    continue
                for interval in lane_intervals:
                    if interval is not None:
                        interval["task_id"] = prefix + str(interval["task_id"])
        elif not compact_seen_namespace:
            kernel._seen_ids.update(task_ids)
        if compact_seen_registration is not None:
            # Commit completed-id history only after the entire closed replay
            # has drained and every persistent metric has been synchronized.
            # A timing overflow or malformed layout therefore leaves the
            # compact duplicate guard untouched, matching ordinary atomic
            # submission semantics.
            kernel._commit_compact_seen_namespace(
                compact_seen_registration
            )
        return {
            stage_id: (
                (
                    min_start_by_stage[stage_position]
                    if min_start_by_stage[stage_position] is not None
                    else ready_by_stage[stage_id]
                ),
                (
                    max_end_by_stage[stage_position]
                    if max_end_by_stage[stage_position] is not None
                    else ready_by_stage[stage_id]
                ),
            )
            for stage_position, stage_id in enumerate(stage_ids)
        }

    def _replay_cached_execution_stage_tasks(
        self,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        replay_layout: _ExecutionStageReplayLayout,
        isolated_shadow: Optional[_UniformIsolatedReplay] = None,
    ) -> Mapping[str, Tuple[float, float]]:
        namespace = "serving.online.{:08d}".format(
            self._execution_task_sequence
        )

        if replay_layout.structure_key is not None and isolated_shadow is None:
            replayed = self._replay_prevalidated_execution_layout(
                self._execution_resource_kernel,
                stages,
                ready_by_stage,
                replay_layout,
                None,
                namespace=namespace,
            )
            if replayed is not None:
                return replayed
        specs = replay_layout.instantiate_trusted(
            stages,
            ready_by_stage,
            namespace,
        )
        position_by_task_id = {
            task.task_id: position for position, task in enumerate(specs)
        }
        if isolated_shadow is not None:
            isolated_shadow.bind(
                specs,
                replay_layout.compiled,
                structure_prevalidated=(
                    replay_layout.structure_key is not None
                ),
            )
        if replay_layout.structure_key is not None:
            replayed = self._replay_prevalidated_execution_layout(
                self._execution_resource_kernel,
                stages,
                ready_by_stage,
                replay_layout,
                specs,
                isolated_shadow,
                namespace=namespace,
            )
            if replayed is not None:
                return replayed
        # A compiled layout is only a task-template optimization.  The live
        # resource clocks, completed-parent state, controller queues, and
        # runtime metrics belong to the one persistent kernel for this run.
        # Replacing that kernel and transplanting selected private fields can
        # silently lose dynamic control-plane state, so cached task instances
        # are appended through the same validated incremental path as an
        # uncached cohort.
        kernel = self._execution_resource_kernel
        bulk_events: Optional[Tuple[Any, ...]] = None
        if replay_layout.structure_key is None:
            kernel.submit_compiled(specs, replay_layout.compiled)
        else:
            bulk_events = kernel._drain_prevalidated_compiled(
                specs,
                replay_layout.compiled,
                validate_tasks=False,
            )
        min_start_by_stage: Dict[str, float] = {}
        max_end_by_stage: Dict[str, float] = {}
        completed_ids: List[str] = []
        remaining = len(specs)
        while remaining:
            if bulk_events is None:
                event = kernel.step()
                if event is None:
                    raise ValueError(
                        "execution stage task graph could not drain"
                    )
            else:
                event = bulk_events[len(specs) - remaining]
            if isolated_shadow is not None:
                isolated_shadow.observe(event)
            event_position = position_by_task_id[event.task.task_id]
            stage_id = stages[
                replay_layout.tasks[event_position].stage_position
            ].stage_id
            min_start_by_stage[stage_id] = min(
                min_start_by_stage.get(stage_id, event.start_ns),
                event.start_ns,
            )
            max_end_by_stage[stage_id] = max(
                max_end_by_stage.get(stage_id, event.end_ns),
                event.end_ns,
            )
            completed_ids.append(event.task.task_id)
            remaining -= 1
        for task_id in completed_ids:
            kernel.release_completed(task_id)
        return {
            stage.stage_id: (
                min_start_by_stage.get(
                    stage.stage_id, ready_by_stage[stage.stage_id]
                ),
                max_end_by_stage.get(
                    stage.stage_id, ready_by_stage[stage.stage_id]
                ),
            )
            for stage in stages
        }

    @staticmethod
    def _runtime_controller_ledger(
        stages: Sequence[_ExecutionStage],
    ) -> Mapping[str, object]:
        """Return a compact exactly-once ledger for runtime GPU overlays."""

        controller_tasks = tuple(
            task
            for stage in stages
            for task in stage.execution_tasks
            if task.task_id.startswith("runtime.controller.")
        )
        domain_totals: Dict[
            Tuple[str, str, str, Tuple[str, ...]], Dict[str, object]
        ] = {}
        for task in controller_tasks:
            raw_domains = task.metadata.get("service_domains")
            if raw_domains is None:
                raw_domains = (
                    {
                        "resource_id": (
                            task.demands[0].resource_id
                            if task.demands
                            else ""
                        ),
                        "service_domain": task.metadata.get(
                            "service_domain", "unclassified"
                        ),
                        "service_role": task.metadata.get(
                            "service_role", "controller_owner"
                        ),
                        "observed_resource_ids": (),
                    },
                )
            if isinstance(raw_domains, (str, bytes, _ABCMapping)) or not isinstance(
                raw_domains, _ABCSequence
            ):
                raise ValueError(
                    "runtime controller service_domains must be an ordered sequence"
                )
            demand_by_resource = {
                demand.resource_id: demand for demand in task.demands
            }
            for raw_domain in raw_domains:
                if not isinstance(raw_domain, _ABCMapping):
                    raise ValueError(
                        "runtime controller service domain must be a mapping"
                    )
                resource_id = str(raw_domain.get("resource_id", ""))
                service_domain = str(
                    raw_domain.get("service_domain", "unclassified")
                )
                service_role = str(
                    raw_domain.get("service_role", "controller_owner")
                )
                raw_observed = raw_domain.get("observed_resource_ids")
                if raw_observed is None:
                    observed = raw_domain.get("observed_resource_id")
                    raw_observed = (observed,) if observed else ()
                observed_resource_ids = tuple(
                    str(item) for item in raw_observed if item
                )
                demand = demand_by_resource.get(resource_id)
                service_ns = (
                    max(0.0, float(demand.service_ns))
                    if demand is not None
                    else max(0.0, float(raw_domain.get("service_ns", 0.0)))
                )
                observed_bytes = (
                    max(0, int(demand.bytes_moved))
                    if demand is not None
                    else max(0, int(raw_domain.get("observed_bytes", 0)))
                )
                work_units = (
                    max(0.0, float(demand.work_units))
                    if demand is not None
                    else 0.0
                )
                key = (
                    service_domain,
                    service_role,
                    resource_id,
                    observed_resource_ids,
                )
                row = domain_totals.setdefault(
                    key,
                    {
                        "service_domain": service_domain,
                        "service_role": service_role,
                        "resource_id": resource_id,
                        "observed_resource_ids": observed_resource_ids,
                        "task_count": 0,
                        "observed_bytes": 0,
                        "service_ns": 0.0,
                        "work_units": 0.0,
                        "bulk_service_ns": 0.0,
                    },
                )
                row["task_count"] = int(row["task_count"]) + 1
                row["observed_bytes"] = int(row["observed_bytes"]) + observed_bytes
                row["service_ns"] = float(row["service_ns"]) + service_ns
                row["work_units"] = float(row["work_units"]) + work_units
        return {
            "schema_version": "heterollm.runtime-controller-ledger/v1",
            "task_count": len(controller_tasks),
            "domains": tuple(
                domain_totals[key] for key in sorted(domain_totals)
            ),
            "bulk_service_ownership": "planner_or_topology_only",
        }

    def _with_gpu_controller_stages(
        self,
        stages: Sequence[_ExecutionStage],
        cohort: BatchCohort,
    ) -> Tuple[_ExecutionStage, ...]:
        """Overlay aggregate GPU translation and memory-controller work.

        Existing planner tasks already carry the useful compute and memory
        quantities.  V4 exposes MMU translation as a short causal prefix for
        each GPU execution root.  Linked L2 traffic is an observer of the
        planner-owned cache domain, while independent VRAM access waves are
        composed in parallel with the root memory phase.  Quantities remain
        aggregate per cohort/root; this must never expand per page, cache
        line, memory request, or PCIe packet.
        """

        original = tuple(stages)
        if not original:
            return original
        replacements: Dict[str, _ExecutionStage] = {}
        inserted_before: Dict[str, Tuple[_ExecutionStage, ...]] = {}
        runtime_profile = self.plan.scenario.runtime_profile

        for gpu_id, controllers in sorted(
            runtime_profile.gpu_controllers.items()
        ):
            gpu_candidates = tuple(
                stage
                for stage in original
                if stage.component_id == gpu_id
                and any(
                    demand.bytes_moved > 0 or demand.work_units > 0
                    for task in stage.execution_tasks
                    for demand in task.demands
                )
                and any(
                    not any(
                        marker in demand.resource_id
                        for marker in (
                            "command_processor",
                            "command_queue",
                            "launch",
                            "mmu_tlb",
                            "l2_controller",
                            "vram_controller",
                        )
                    )
                    for task in stage.execution_tasks
                    for demand in task.demands
                )
            )
            if not gpu_candidates:
                continue
            candidate_ids = {stage.stage_id for stage in gpu_candidates}
            roots = tuple(
                stage
                for stage in gpu_candidates
                if not any(
                    dependency in candidate_ids
                    for dependency in stage.dependencies
                )
            )
            if not roots:
                continue
            # Resolve the planner-owned cache/backing domains before adding
            # runtime controller observers.  The analytical GPU cost model
            # already owns L2 hit/bandwidth service on its declared cache
            # resource; charging the same hit latency again in a serial
            # controller prefix would violate exactly-once service ownership.
            # VRAM backing demands own only bulk bandwidth, so the independent
            # controller access-wave term remains a runtime-owned domain.
            cached_domains = self._gpu_controller_resource_domains.get(gpu_id)
            if cached_domains is None:
                l2_resource_id: Optional[str] = None
                memory_resource_ids = set()
                try:
                    gpu_profile = self.plan.scenario.resolve_component_profile(
                        gpu_id, GPUProfile
                    )
                    l2_level = next(
                        (
                            level
                            for level in reversed(
                                gpu_profile.cache_hierarchy.levels
                            )
                            if str(level.name).strip().lower() == "l2"
                        ),
                        None,
                    )
                    reference_gpu_id = (
                        self.plan.scenario.host_orchestration_profile.gpu_component_id
                    )
                    if l2_level is not None:
                        l2_resource_id = _component_resource_id(
                            l2_level.resource_id,
                            reference_component_id=reference_gpu_id,
                            target_component_id=gpu_id,
                        )
                    parallel_plan = _parallel_plan(self.plan.scenario)
                    for rank in parallel_plan.ranks:
                        if rank.component_id != gpu_id:
                            continue
                        memory_component_id = rank.memory_component_id
                        if memory_component_id is None:
                            continue
                        hbm_profile = (
                            self.plan.scenario.resolve_component_profile(
                                memory_component_id, HBMProfile
                            )
                        )
                        memory_resource_ids.add(
                            _component_resource_id(
                                hbm_profile.resource_id,
                                reference_component_id=reference_gpu_id,
                                target_component_id=memory_component_id,
                            )
                        )
                except (KeyError, StopIteration, TypeError, ValueError):
                    # Custom lowerers/profiles without a linkable cache domain
                    # retain the conservative controller-owned fallback below.
                    l2_resource_id = None
                    memory_resource_ids.clear()
                cached_domains = (
                    l2_resource_id,
                    frozenset(memory_resource_ids),
                )
                self._gpu_controller_resource_domains[gpu_id] = cached_domains
            l2_resource_id, frozen_memory_resource_ids = cached_domains
            memory_resource_ids = set(frozen_memory_resource_ids)

            # A task can expose the same byte fact on concurrent compute and
            # memory resources.  ``fallback_total_bytes`` therefore retains
            # the historical max-per-task rule only for profiles that do not
            # expose planner resource provenance.
            fallback_total_bytes = sum(
                max(
                    (demand.bytes_moved for demand in task.demands),
                    default=0,
                )
                for stage in gpu_candidates
                for task in stage.execution_tasks
            )
            l2_total_bytes = sum(
                demand.bytes_moved
                for stage in gpu_candidates
                for task in stage.execution_tasks
                for demand in task.demands
                if l2_resource_id is not None
                and demand.resource_id == l2_resource_id
            )
            vram_total_bytes = sum(
                demand.bytes_moved
                for stage in gpu_candidates
                for task in stage.execution_tasks
                for demand in task.demands
                if demand.resource_id in memory_resource_ids
            )
            root_count = len(roots)

            def root_bytes(total: int, root_index: int) -> int:
                base, extra = divmod(max(0, int(total)), root_count)
                return base + (1 if root_index < extra else 0)

            for root_index, root in enumerate(roots):
                fallback_bytes = root_bytes(
                    fallback_total_bytes, root_index
                )
                l2_bytes = root_bytes(l2_total_bytes, root_index)
                vram_bytes = root_bytes(vram_total_bytes, root_index)
                translation_bytes = max(
                    1,
                    l2_bytes if l2_bytes > 0 else fallback_bytes,
                )
                prefix = "runtime.controller.{}.{}.root{:04d}".format(
                    cohort.cohort_id,
                    gpu_id,
                    root_index,
                )
                request_ids = root.request_ids or cohort.request_ids
                mmu = controllers.mmu_tlb
                pages = max(
                    1,
                    (translation_bytes + mmu.page_size_bytes - 1)
                    // mmu.page_size_bytes,
                )
                translation_batches = max(
                    1,
                    (pages + mmu.translation_batch_size - 1)
                    // mmu.translation_batch_size,
                )
                translation_waves = max(
                    1,
                    (
                        translation_batches
                        + mmu.max_outstanding_page_walks
                        - 1
                    )
                    // mmu.max_outstanding_page_walks,
                )
                mmu_task_id = prefix + ".mmu_tlb"
                mmu_task = _ExecutionTask(
                    task_id=mmu_task_id,
                    dependencies=(),
                    request_ids=request_ids,
                    demands=(
                        ResourceDemand(
                            "{}.mmu_tlb".format(gpu_id),
                            translation_waves * mmu.page_walk_latency_ns,
                            bytes_moved=translation_bytes,
                            work_units=float(pages),
                        ),
                    ),
                    category=TaskCategory.MEMORY,
                    metadata={
                        "event_kind": "gpu_mmu_tlb_batch",
                        "runtime_phase": "gpu_mmu_tlb",
                        "aggregation": "controller_transaction_batch",
                        "transaction_count": pages,
                        "movement_id": prefix + ".gpu_memory_access",
                        "service_domain": "gpu_address_translation",
                        "service_role": "controller_owner",
                        "bulk_service_owner": False,
                    },
                )
                mmu_stage = _ExecutionStage(
                    stage_id=mmu_task_id,
                    stage_index=root.stage_index * 10 - 3,
                    dependencies=root.dependencies,
                    request_ids=request_ids,
                    component_id=gpu_id,
                    service_ns=sum(
                        demand.service_ns for demand in mmu_task.demands
                    ),
                    execution_tasks=(mmu_task,),
                    invocation_group_id=root.invocation_group_id,
                    covered_invocation_group_ids=(
                        root.covered_invocation_group_ids
                    ),
                )

                l2 = controllers.l2_cache
                observed_l2_bytes = (
                    l2_bytes if l2_bytes > 0 else fallback_bytes
                )
                cache_requests = (
                    (observed_l2_bytes + l2.line_size_bytes - 1)
                    // l2.line_size_bytes
                    if observed_l2_bytes > 0
                    else 0
                )
                cache_batches = max(
                    0,
                    (cache_requests + l2.request_batch_size - 1)
                    // l2.request_batch_size,
                )
                cache_waves = max(
                    0,
                    (cache_batches + l2.max_outstanding_misses - 1)
                    // l2.max_outstanding_misses,
                )
                planner_owns_l2_service = (
                    l2_resource_id is not None and l2_total_bytes > 0
                )
                # CacheHierarchyProfile already charges both L2 bandwidth and
                # hit-latency waves.  The runtime L2 controller is therefore
                # a byte/work observer for linked planner tasks.  A custom
                # lowerer with no declared L2 provenance retains the former
                # controller-owned latency as an explicit degraded fallback.
                l2_service_ns = (
                    0.0
                    if planner_owns_l2_service
                    else cache_waves * l2.hit_latency_ns
                )

                vram = controllers.vram_controller
                observed_vram_bytes = (
                    vram_bytes if vram_bytes > 0 else fallback_bytes
                )
                controller_requests = (
                    (observed_vram_bytes + l2.line_size_bytes - 1)
                    // l2.line_size_bytes
                    if observed_vram_bytes > 0
                    else 0
                )
                controller_batches = max(
                    0,
                    (controller_requests + vram.request_batch_size - 1)
                    // vram.request_batch_size,
                )
                controller_parallelism = max(
                    1,
                    vram.controller_count
                    * vram.channel_count
                    * vram.lanes_per_channel,
                )
                controller_waves = max(
                    0,
                    (
                        controller_batches
                        + min(
                            controller_parallelism,
                            vram.max_outstanding_requests,
                        )
                        - 1
                    )
                    // min(
                        controller_parallelism,
                        vram.max_outstanding_requests,
                    ),
                )
                # Planner roots already include useful VRAM bulk bandwidth.
                # This overlay models only the independent access-wave domain
                # and runs in the root memory phase instead of serially
                # charging the same physical movement after L2.
                vram_service_ns = controller_waves * vram.access_latency_ns
                memory_task_id = prefix + ".memory_pipeline"
                memory_task = _ExecutionTask(
                    task_id=memory_task_id,
                    dependencies=(),
                    request_ids=request_ids,
                    demands=(
                        ResourceDemand(
                            "{}.l2_controller".format(gpu_id),
                            l2_service_ns,
                            bytes_moved=observed_l2_bytes,
                            work_units=float(cache_requests),
                        ),
                        ResourceDemand(
                            "{}.vram_controller".format(gpu_id),
                            vram_service_ns,
                            bytes_moved=observed_vram_bytes,
                            work_units=float(controller_requests),
                        ),
                    ),
                    category=TaskCategory.MEMORY,
                    metadata={
                        "event_kind": "gpu_memory_controller_pipeline",
                        "runtime_phase": "gpu_memory_pipeline",
                        "aggregation": "controller_transaction_batch",
                        "movement_id": prefix + ".gpu_memory_access",
                        "service_domains": (
                            {
                                "resource_id": "{}.l2_controller".format(
                                    gpu_id
                                ),
                                "service_domain": "gpu_l2_data_access",
                                "service_role": (
                                    "controller_observer"
                                    if planner_owns_l2_service
                                    else "controller_owner_fallback"
                                ),
                                "observed_resource_id": l2_resource_id,
                                "observed_bytes": observed_l2_bytes,
                                "service_ns": l2_service_ns,
                                "transaction_count": cache_requests,
                            },
                            {
                                "resource_id": "{}.vram_controller".format(
                                    gpu_id
                                ),
                                "service_domain": "gpu_vram_access_latency",
                                "service_role": "controller_owner",
                                "observed_resource_ids": tuple(
                                    sorted(memory_resource_ids)
                                ),
                                "observed_bytes": observed_vram_bytes,
                                "service_ns": vram_service_ns,
                                "transaction_count": controller_requests,
                            },
                        ),
                        "bulk_service_owner": False,
                        "composition": "parallel_roofline_with_gpu_root",
                    },
                )
                inserted_before[root.stage_id] = (mmu_stage,)
                replacements[root.stage_id] = _ExecutionStage(
                    stage_id=root.stage_id,
                    stage_index=root.stage_index,
                    dependencies=(mmu_stage.stage_id,),
                    request_ids=root.request_ids,
                    component_id=root.component_id,
                    service_ns=max(
                        root.service_ns,
                        l2_service_ns,
                        vram_service_ns,
                    ),
                    execution_tasks=(
                        *root.execution_tasks,
                        memory_task,
                    ),
                    invocation_group_id=root.invocation_group_id,
                    covered_invocation_group_ids=(
                        root.covered_invocation_group_ids
                    ),
                )

        if not inserted_before:
            return original
        expanded: List[_ExecutionStage] = []
        for stage in original:
            expanded.extend(inserted_before.get(stage.stage_id, ()))
            expanded.append(replacements.get(stage.stage_id, stage))
        return tuple(expanded)

    @classmethod
    def _replay_execution_stage_duration(
        cls,
        kernel: UnifiedEventKernel,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
        namespace: str,
        replay_layout: Optional[_ExecutionStageReplayLayout] = None,
    ) -> float:
        """Replay a stage graph while retaining only its device makespan.

        The isolated replay is used only to derive the cohort's standalone
        duration.  It must follow the same task construction and kernel event
        order as the live replay, but it has no schedule observer to retain.
        In particular, the isolated kernel is discarded by the caller, so
        completed dependency state does not need to be released.
        """

        if (
            replay_layout is not None
            and replay_layout.structure_key is not None
            and not kernel.has_active_tasks
        ):
            replayed = cls._replay_prevalidated_execution_layout(
                kernel,
                stages,
                ready_by_stage,
                replay_layout,
                None,
                namespace=namespace,
            )
            if replayed is not None:
                return max(
                    (end_ns for _start_ns, end_ns in replayed.values()),
                    default=0.0,
                )

        if replay_layout is None:
            specs, _source_by_id = cls._stage_task_specs(
                stages, ready_by_stage, namespace
            )
            kernel.add_tasks(specs)
        else:
            specs = replay_layout.instantiate(stages, ready_by_stage)
            kernel = UnifiedEventKernel._from_compiled_layout(
                specs,
                replay_layout.compiled,
                resource_capacities=kernel.resource_capacities,
            )
        remaining = len(specs)
        max_end_ns = 0.0
        while remaining:
            event = kernel.step()
            if event is None:
                raise ValueError("execution stage task graph could not drain")
            max_end_ns = max(max_end_ns, event.end_ns)
            remaining -= 1
        return max_end_ns

    @staticmethod
    def _uniform_live_replay_origin_ns(
        kernel: UnifiedEventKernel,
        stages: Sequence[_ExecutionStage],
        ready_by_stage: Mapping[str, float],
    ) -> Optional[float]:
        """Return a safe common origin for live/isolated shadow replay."""

        if kernel.has_active_tasks or not stages:
            return None
        origin_ns = ready_by_stage.get(stages[0].stage_id)
        if origin_ns is None:
            return None
        origin_ns = float(origin_ns)
        if not math.isfinite(origin_ns):
            return None
        resource_available = kernel.resource_available
        resource_lane_available = kernel._resource_lane_available
        touched_resources = {
            demand.resource_id
            for stage in stages
            for task in stage.execution_tasks
            for demand in task.demands
        }
        for stage in stages:
            stage_ready_ns = ready_by_stage.get(stage.stage_id)
            if stage_ready_ns is None or float(stage_ready_ns) != origin_ns:
                return None
        for resource_id in touched_resources:
            lanes = resource_lane_available.get(resource_id)
            if lanes is None:
                lanes = [
                    0.0
                ] * kernel.resource_capacities.get(resource_id, 1)
            if not lanes:
                return None
            for lane_available_ns in lanes:
                try:
                    lane_available_ns = float(lane_available_ns)
                except (TypeError, ValueError, OverflowError):
                    return None
                if (
                    not math.isfinite(lane_available_ns)
                    or lane_available_ns > origin_ns
                ):
                    return None
            try:
                resource_ready_ns = float(
                    resource_available.get(resource_id, 0.0)
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                not math.isfinite(resource_ready_ns)
                or resource_ready_ns > origin_ns
            ):
                return None
        return origin_ns

    @staticmethod
    def _trusted_execution_stage_duration(
        metadata: Mapping[str, Any],
        *,
        host_ns: float,
        device_ns: float,
        extension_ns: float,
    ) -> Optional[float]:
        """Reuse the planner's executed-DAG makespan when it is still exact.

        The topology-aware lowerer already executes the task DAG and records
        its device makespan.  Resource contention and residency may append one
        serial runtime extension.  When the recorded makespan plus that exact
        extension still agrees with the adjusted device duration, replaying an
        isolated copy of the same DAG cannot add information.  Hand-authored,
        stale, or inconsistent metadata deliberately falls back to the full
        event-kernel replay.
        """

        if (
            metadata.get("execution_stage_source")
            != "executed_task_dag_kernel_timeline"
        ):
            return None
        raw_makespan = metadata.get("execution_stage_makespan_ns")
        if isinstance(raw_makespan, bool):
            return None
        try:
            stage_makespan_ns = float(raw_makespan)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(stage_makespan_ns) or stage_makespan_ns < 0.0:
            return None
        projected_device_ns = stage_makespan_ns + extension_ns
        if not math.isclose(
            projected_device_ns,
            device_ns,
            rel_tol=1.0e-12,
            abs_tol=1.0e-6,
        ):
            return None
        return host_ns + projected_device_ns

    def _schedule_execution_stages(
        self,
        cohort: BatchCohort,
        cost: BatchCost,
        stages: Sequence[_ExecutionStage],
        host_ns: float,
        device_ns: float,
        request_ready_ns: float,
        runtime_controller_stage_count: int = 0,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Tuple[Mapping[str, Any], ...],
        float,
    ]:
        component_count = len({stage.component_id for stage in stages})
        pipeline_prefill = (
            cohort.kind == "prefill"
            and all(item.phase == "prefill" for item in cohort.items)
            and component_count > 1
        )
        host_request_ready_ns = (
            max(
                (
                    max(
                        self.states[item.request_id].spec.arrival_ns,
                        self._runtime_origin_ns,
                    )
                    for item in cohort.items
                ),
                default=self.now,
            )
            if pipeline_prefill
            else request_ready_ns
        )
        previous_device_available_ns = self._gpu_available_ns
        host_start_ns = max(self._host_available_ns, host_request_ready_ns)
        host_end_ns = host_start_ns + host_ns

        (
            causally_placed_stages,
            causally_placed_owner_ns,
            causal_transfer_stage_count,
        ) = self._inject_owner_residency_transfer_stages(
            stages,
            cost.metadata,
        )
        scheduled_stages = list(causally_placed_stages)
        total_extension_ns = self._runtime_stage_extension_ns(
            cost.metadata,
            device_ns,
        )
        owner_transfer_ns = max(
            0.0,
            float(
                cost.metadata.get("owner_residency_transfer_ns", 0.0)
            ),
        )
        if causally_placed_owner_ns > owner_transfer_ns and not math.isclose(
            causally_placed_owner_ns,
            owner_transfer_ns,
            rel_tol=1.0e-12,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "causally placed residency service exceeds conserved total"
            )
        extension_ns = max(
            0.0,
            total_extension_ns - min(
                causally_placed_owner_ns,
                owner_transfer_ns,
            ),
        )
        if extension_ns > 0.0:
            dependency_ids = {
                dependency
                for stage in scheduled_stages
                for dependency in stage.dependencies
            }
            terminal_ids = tuple(
                stage.stage_id
                for stage in scheduled_stages
                if stage.stage_id not in dependency_ids
            )
            last_stage = scheduled_stages[-1]
            extension_stage_index = max(
                stage.stage_index for stage in scheduled_stages
            ) + 1
            scheduled_stages.append(
                _ExecutionStage(
                    "runtime.device_extension",
                    extension_stage_index,
                    terminal_ids,
                    cohort.request_ids,
                    last_stage.component_id,
                    extension_ns,
                    (
                        _ExecutionTask(
                            "runtime.device_extension.task0000",
                            (),
                            cohort.request_ids,
                            (
                                ResourceDemand(
                                    last_stage.component_id, extension_ns
                                ),
                            ),
                            True,
                        ),
                    ),
                )
            )

        cohort_request_ids = set(cohort.request_ids)
        for stage in scheduled_stages:
            invalid_stage_request_ids = tuple(
                request_id
                for request_id in stage.request_ids
                if request_id not in cohort_request_ids
            )
            if invalid_stage_request_ids:
                raise ValueError(
                    "execution stage {} request_ids are not in current "
                    "cohort: {}".format(
                        stage.stage_id, invalid_stage_request_ids
                    )
                )
            for task in stage.execution_tasks:
                invalid_task_request_ids = tuple(
                    request_id
                    for request_id in task.request_ids
                    if request_id not in cohort_request_ids
                )
                if invalid_task_request_ids:
                    raise ValueError(
                        "execution task {} request_ids are not in current "
                        "cohort: {}".format(
                            task.task_id, invalid_task_request_ids
                        )
                    )

        participates_in_device_fence = {
            stage.stage_id: self._component_participates_in_device_fence(
                stage.component_id
            )
            for stage in scheduled_stages
        }
        requires_device_fence = {
            stage.stage_id: (
                stage.requires_device_fence
                and participates_in_device_fence[stage.stage_id]
            )
            for stage in scheduled_stages
        }

        ready_by_stage: Dict[str, float] = {}
        for stage in scheduled_stages:
            device_fence_ready_ns = 0.0
            if participates_in_device_fence[stage.stage_id]:
                device_fence_ready_ns = self._last_serial_device_end_ns
                if requires_device_fence[stage.stage_id]:
                    device_fence_ready_ns = max(
                        device_fence_ready_ns,
                        previous_device_available_ns,
                    )
            cross_chunk_ready = max(
                (
                    self._request_stage_ready_ns.get(
                        (request_id, stage.component_id, stage.stage_index),
                        max(
                            self.states[request_id].spec.arrival_ns,
                            self._runtime_origin_ns,
                        ),
                    )
                    for request_id in stage.request_ids
                ),
                default=host_end_ns,
            )
            ready_by_stage[stage.stage_id] = max(
                host_end_ns,
                cross_chunk_ready,
                device_fence_ready_ns,
            )

        isolated_shadow: Optional[_UniformIsolatedReplay] = None
        uniform_origin_ns: Optional[float] = None
        if causal_transfer_stage_count > 0:
            uniform_origin_ns = self._uniform_live_replay_origin_ns(
                self._execution_resource_kernel,
                scheduled_stages,
                ready_by_stage,
            )
        raw_execution_stages = cost.metadata.get("execution_stages")

        # A uniform causal graph is replayed once while the exact zero-origin
        # shadow derives its isolated duration.  Non-uniform causal graphs
        # still share one validated final layout between the live and
        # isolated kernels.
        replay_layout = self._execution_stage_metadata_cache.replay_layout(
            cost.metadata,
            scheduled_stages,
            has_runtime_extension=extension_ns > 0.0,
            runtime_overlay_stage_count=(
                causal_transfer_stage_count
                + max(0, runtime_controller_stage_count)
            ),
        )
        overlay_timing_key: Optional[Tuple[object, ...]] = None
        if (
            replay_layout is not None
            and replay_layout.structure_key is not None
            and isinstance(raw_execution_stages, tuple)
        ):
            # The exact runtime-overlay timing tuple participates in every
            # isolated-duration lookup and insert for this cohort.  Large
            # trusted DAGs may reach this path several times; derive the
            # immutable key once instead of rescanning every overlay task.
            overlay_timing_key = (
                self._execution_stage_metadata_cache._runtime_overlay_timing_key(
                    scheduled_stages
                )
            )
        cached_isolated_device_duration_ns: Optional[float] = None
        if (
            uniform_origin_ns is not None
            and replay_layout is not None
            and replay_layout.structure_key is not None
            and isinstance(raw_execution_stages, tuple)
        ):
            cached_isolated_device_duration_ns = (
                self._execution_stage_metadata_cache.isolated_duration(
                    raw_execution_stages,
                    replay_layout,
                    scheduled_stages,
                    self._execution_resource_kernel.resource_capacities,
                    overlay_timing_key=overlay_timing_key,
                )
            )
        if (
            uniform_origin_ns is not None
            and cached_isolated_device_duration_ns is None
        ):
            isolated_shadow = _UniformIsolatedReplay(
                resource_capacities=(
                    self._execution_resource_kernel.resource_capacities
                ),
                origin_ns=uniform_origin_ns,
            )
        self._execution_task_sequence += 1
        if replay_layout is None:
            stage_envelopes = self._replay_execution_stage_tasks(
                self._execution_resource_kernel,
                scheduled_stages,
                ready_by_stage,
                "serving.online.{:08d}".format(
                    self._execution_task_sequence
                ),
                isolated_shadow,
            )
        else:
            stage_envelopes = self._replay_cached_execution_stage_tasks(
                scheduled_stages,
                ready_by_stage,
                replay_layout,
                isolated_shadow,
            )
        realized: List[Mapping[str, Any]] = []
        for stage in scheduled_stages:
            stage_start_ns, stage_end_ns = stage_envelopes[stage.stage_id]
            self._stage_resource_available_ns[stage.component_id] = stage_end_ns
            for request_id in stage.request_ids:
                self._request_stage_ready_ns[
                    (request_id, stage.component_id, stage.stage_index)
                ] = stage_end_ns
            realized_stage: Dict[str, Any] = {
                "stage_id": stage.stage_id,
                "stage_index": stage.stage_index,
                "invocation_group_id": stage.invocation_group_id,
                "covered_invocation_group_ids": (
                    stage.covered_invocation_group_ids
                ),
                "dependencies": stage.dependencies,
                "request_ids": stage.request_ids,
                "component_id": stage.component_id,
                "service_ns": stage.service_ns,
                "realized_service_ns": stage_end_ns - stage_start_ns,
                "start_ns": stage_start_ns,
                "end_ns": stage_end_ns,
            }
            realized.append(realized_stage)

        trusted_planner_stages = (
            cost.metadata.get("execution_stage_source")
            == "executed_task_dag_kernel_timeline"
            and _trusted_execution_stages(
                cost.metadata.get("execution_stages")
            )
            is not None
        )
        if trusted_planner_stages:
            realized_by_id = {
                str(stage["stage_id"]): stage for stage in realized
            }
            earliest_consumer_start_by_group: Dict[str, float] = {}
            for row in realized:
                row_id = str(row["stage_id"])
                if row_id.startswith("runtime.owner_residency_transfer."):
                    continue
                causal_ids = tuple(
                    dict.fromkeys(
                        (
                            *((
                                str(row["invocation_group_id"]),
                            ) if row.get("invocation_group_id") else ()),
                            *tuple(
                                str(item)
                                for item in row.get(
                                    "covered_invocation_group_ids", ()
                                )
                            ),
                        )
                    )
                )
                start_ns = float(row["start_ns"])
                for group_id in causal_ids:
                    earliest_consumer_start_by_group[group_id] = min(
                        start_ns,
                        earliest_consumer_start_by_group.get(
                            group_id, start_ns
                        ),
                    )
            raw_transfer_batches = cost.metadata.get(
                "owner_residency_transfer_batches", ()
            )
            if isinstance(raw_transfer_batches, _ABCSequence) and not isinstance(
                raw_transfer_batches, (str, bytes, _ABCMapping)
            ):
                for batch_index, raw_batch in enumerate(raw_transfer_batches):
                    if not isinstance(raw_batch, _ABCMapping):
                        continue
                    if (
                        str(raw_batch.get("kind", ""))
                        != MigrationKind.PAGE_IN.value
                        or float(raw_batch.get("transfer_ns", 0.0)) <= 0.0
                    ):
                        continue
                    transfer_id = (
                        "runtime.owner_residency_transfer.batch{:04d}".format(
                            batch_index
                        )
                    )
                    transfer_row = realized_by_id.get(transfer_id)
                    if transfer_row is None:
                        raise ValueError(
                            "trusted owner-residency PAGE_IN was not placed "
                            "in the execution DAG"
                        )
                    group_ids = tuple(
                        str(item)
                        for item in raw_batch.get(
                            "operator_invocation_group_ids", ()
                        )
                    )
                    if not group_ids or any(
                        group_id not in earliest_consumer_start_by_group
                        for group_id in group_ids
                    ):
                        raise ValueError(
                            "trusted owner-residency PAGE_IN has no realized "
                            "consumer stage"
                        )
                    earliest_consumer_start = min(
                        earliest_consumer_start_by_group[group_id]
                        for group_id in group_ids
                    )
                    if (
                        float(transfer_row["end_ns"])
                        > earliest_consumer_start + 1.0e-6
                    ):
                        raise ValueError(
                            "trusted owner-residency PAGE_IN completes after "
                            "its earliest consumer"
                        )
        device_start_ns = min(
            (float(stage["start_ns"]) for stage in realized),
            default=host_end_ns,
        )
        end_ns = max(
            previous_device_available_ns,
            max((float(stage["end_ns"]) for stage in realized), default=host_end_ns),
        )

        isolated_duration_ns = None
        if (
            causal_transfer_stage_count == 0
            and runtime_controller_stage_count == 0
        ):
            isolated_duration_ns = self._trusted_execution_stage_duration(
                cost.metadata,
                host_ns=host_ns,
                device_ns=device_ns,
                extension_ns=extension_ns,
            )
        if (
            isolated_duration_ns is None
            and cached_isolated_device_duration_ns is not None
        ):
            isolated_duration_ns = (
                host_ns + cached_isolated_device_duration_ns
            )
        if isolated_duration_ns is None and isolated_shadow is not None:
            isolated_device_duration_ns = isolated_shadow.duration_ns
            if isolated_device_duration_ns is not None:
                isolated_duration_ns = host_ns + isolated_device_duration_ns
                if (
                    replay_layout is not None
                    and replay_layout.structure_key is not None
                    and isinstance(raw_execution_stages, tuple)
                ):
                    self._execution_stage_metadata_cache.remember_isolated_duration(
                        raw_execution_stages,
                        replay_layout,
                        scheduled_stages,
                        self._execution_resource_kernel.resource_capacities,
                        isolated_device_duration_ns,
                        overlay_timing_key=overlay_timing_key,
                    )
        if isolated_duration_ns is None:
            resource_capacities = (
                self._execution_resource_kernel.resource_capacities
            )
            raw_stages = raw_execution_stages
            isolated_device_duration_ns = None
            if (
                replay_layout is not None
                and replay_layout.structure_key is not None
                and isinstance(raw_stages, tuple)
            ):
                isolated_device_duration_ns = (
                    self._execution_stage_metadata_cache.isolated_duration(
                        raw_stages,
                        replay_layout,
                        scheduled_stages,
                        resource_capacities,
                        overlay_timing_key=overlay_timing_key,
                    )
                )
            if isolated_device_duration_ns is None:
                isolated_device_duration_ns = (
                    self._replay_execution_stage_duration(
                        UnifiedEventKernel(
                            resource_capacities=resource_capacities
                        ),
                        scheduled_stages,
                        {
                            stage.stage_id: 0.0
                            for stage in scheduled_stages
                        },
                        "serving.isolated.{:08d}".format(
                            self._execution_task_sequence
                        ),
                        replay_layout,
                    )
                )
                if (
                    replay_layout is not None
                    and replay_layout.structure_key is not None
                    and isinstance(raw_stages, tuple)
                ):
                    self._execution_stage_metadata_cache.remember_isolated_duration(
                        raw_stages,
                        replay_layout,
                        scheduled_stages,
                        resource_capacities,
                        isolated_device_duration_ns,
                        overlay_timing_key=overlay_timing_key,
                    )
            isolated_duration_ns = host_ns + isolated_device_duration_ns
        overlap_ns = max(
            0.0,
            min(host_end_ns, previous_device_available_ns)
            - max(host_start_ns, self._last_gpu_start_ns),
        )
        device_idle_ns = max(
            0.0, device_start_ns - previous_device_available_ns
        )
        self._host_available_ns = host_end_ns
        self._gpu_available_ns = end_ns
        self._last_gpu_start_ns = device_start_ns
        if any(requires_device_fence.values()):
            device_fence_end_ns = max(
                (
                    float(stage["end_ns"])
                    for stage in realized
                    if requires_device_fence[str(stage["stage_id"])]
                ),
                default=end_ns,
            )
            self._last_serial_device_end_ns = max(
                self._last_serial_device_end_ns, device_fence_end_ns
            )
        return (
            host_start_ns,
            host_end_ns,
            device_start_ns,
            end_ns,
            overlap_ns,
            device_idle_ns,
            tuple(realized),
            isolated_duration_ns,
        )

    def _with_kv_scan_lower_bound(self, cohort: BatchCohort) -> BatchCohort:
        prefill_scan = (
            self._kv_scan_enabled and cohort.kind in {"prefill", "mixed"}
            and (self._q4_mma_materialization_enabled
                 or self.plan.scenario.workload.metadata.get("llama_cpp_prefill_kv_scan") is True)
        )
        if prefill_scan:
            scenario = self.plan.scenario
            physical_rows = self.plan.scheduler.max_num_ubatch_tokens or self.plan.scheduler.max_num_batched_tokens
            if physical_rows >= 1024:
                raise ValueError("prefill KV scan does not support mask-trimming query counts")
            if (any(layer.is_linear_attention for layer in _execution_layers(scenario))
                    and not (scenario.workload.metadata.get("supports_batched_stateful_execution") is True
                             and scenario.workload.metadata.get("supports_equal_length_stateful_ubatches") is True)):
                raise ValueError("prefill KV scan requires explicit serial equal-length stateful groups")
            if any(item.phase not in {"prefill", "decode"}
                   or _batch_item_kv_append_tokens(item) != item.token_count for item in cohort.items):
                raise ValueError("prefill KV scan requires ordinary persistent rows")
        if (not self._kv_scan_enabled or cohort.kind != "decode" or not cohort.items
                or any(item.phase != "decode" or item.token_count != 1
                       or _batch_item_kv_append_tokens(item) != 1 for item in cohort.items)) and not prefill_scan:
            return cohort
        occupied = sum(self.states[key].cached_tokens for key in self._kv_scan_wave_requests)
        occupied += sum(_batch_item_kv_append_tokens(item) for item in cohort.items)
        if occupied > self._kv_scan_cache_cells:
            raise ValueError("KV scan lower bound does not support cache wraparound")
        span = min(self._kv_scan_cache_cells, ((max(1, occupied) + 255) // 256) * 256)
        span_key = (
            "llama_cpp_q4_kv_materialization_view_tokens_lower_bound"
            if self._q4_mma_materialization_enabled else "llama_cpp_kv_scan_tokens"
        )
        return replace(cohort, metadata={
            **cohort.metadata, span_key: span,
            "llama_cpp_kv_occupied_rows": occupied,
            "llama_cpp_kv_scan_bound": "occupied_rows_lower_bound",
            "llama_cpp_kv_scan_timing_completeness": "partial",
            **({"llama_cpp_kv_scan_phase": "prefill_rectangular"} if prefill_scan else {}),
        })

    def _execute(self, cohort: BatchCohort) -> None:
        cohort = self._with_kv_scan_lower_bound(cohort)
        cost = _lower_cost(self.lowerer, self.plan.scenario, cohort)
        cost = self._apply_resource_contention(cohort, cost)
        cost = self._apply_owner_residency(cohort, cost)
        reported_host_ns = max(
            0.0, float(cost.metadata.get("host_orchestration_ns", 0.0))
        )
        request_ready_ns = max(
            (
                self._request_device_ready_ns.get(
                    request_id,
                    self.states[request_id].spec.arrival_ns,
                )
                for request_id in cohort.request_ids
            ),
            default=self.now,
        )
        stages, stage_fallback_reason = (
            self._execution_stage_metadata_cache.resolve(cost.metadata)
        )
        host_orchestration_is_explicit = bool(
            stages
            and cost.metadata.get(
                "execution_stages_include_host_orchestration"
            )
            is True
            and cost.metadata.get("execution_stage_source")
            == "executed_task_dag_kernel_timeline"
        )
        host_ns = 0.0 if host_orchestration_is_explicit else reported_host_ns
        device_ns = max(
            0.0,
            float(
                cost.duration_ns
                if host_orchestration_is_explicit
                else cost.metadata.get(
                    "device_execution_ns", cost.duration_ns - host_ns
                )
            ),
        )
        runtime_controller_stage_count = 0
        runtime_controller_ledger: Mapping[str, object] = {
            "schema_version": "heterollm.runtime-controller-ledger/v1",
            "task_count": 0,
            "domains": (),
            "bulk_service_ownership": "planner_or_topology_only",
        }
        if stages:
            base_stage_count = len(stages)
            stages = self._with_gpu_controller_stages(stages, cohort)
            runtime_controller_stage_count = len(stages) - base_stage_count
            runtime_controller_ledger = self._runtime_controller_ledger(
                stages
            )
        realized_stages: Tuple[Mapping[str, Any], ...] = ()
        if stages:
            (
                host_start_ns,
                host_end_ns,
                device_start_ns,
                end_ns,
                overlap_ns,
                gpu_idle_wait_ns,
                realized_stages,
                isolated_duration_ns,
            ) = self._schedule_execution_stages(
                cohort,
                cost,
                stages,
                host_ns,
                device_ns,
                request_ready_ns,
                runtime_controller_stage_count,
            )
            cost = BatchCost(
                max(1.0e-9, isolated_duration_ns),
                cost.energy_pj,
                cost.metadata,
            )
        else:
            previous_gpu_available_ns = self._gpu_available_ns
            host_start_ns = max(self._host_available_ns, request_ready_ns)
            host_end_ns = host_start_ns + host_ns
            device_start_ns = max(previous_gpu_available_ns, host_end_ns)
            if device_start_ns == host_end_ns:
                # Preserve the unified-kernel addition order when this cohort does
                # not overlap an earlier GPU interval.
                end_ns = host_start_ns + cost.duration_ns
            else:
                end_ns = device_start_ns + device_ns
            overlap_ns = max(
                0.0,
                min(host_end_ns, previous_gpu_available_ns)
                - max(host_start_ns, self._last_gpu_start_ns),
            )
            gpu_idle_wait_ns = max(
                0.0, device_start_ns - previous_gpu_available_ns
            )
            self._host_available_ns = host_end_ns
            self._gpu_available_ns = end_ns
            self._last_gpu_start_ns = device_start_ns
            self._last_serial_device_end_ns = end_ns
        task_stage_ids = {
            task.task_id: stage.stage_id
            for stage in stages
            for task in stage.execution_tasks
        }
        realized_end_by_stage = {
            str(stage.get("stage_id", "")): float(stage.get("end_ns", end_ns))
            for stage in realized_stages
        }
        realized_end_by_group: Dict[str, float] = {}
        for stage in realized_stages:
            stage_end = float(stage.get("end_ns", end_ns))
            group_ids = tuple(
                dict.fromkeys(
                    (
                        *((
                            str(stage.get("invocation_group_id")),
                        ) if stage.get("invocation_group_id") else ()),
                        *tuple(
                            str(item)
                            for item in stage.get(
                                "covered_invocation_group_ids", ()
                            )
                        ),
                    )
                )
            )
            for group_id in group_ids:
                realized_end_by_group[group_id] = max(
                    stage_end,
                    realized_end_by_group.get(group_id, stage_end),
                )
        item_completion_ns: Dict[int, float] = {}
        item_terminal_schedule: List[Mapping[str, object]] = []
        raw_item_terminals = cost.metadata.get("item_terminal_tasks", ())
        if isinstance(raw_item_terminals, _ABCSequence) and not isinstance(
            raw_item_terminals, (str, bytes, _ABCMapping)
        ):
            for raw_terminal in raw_item_terminals:
                if not isinstance(raw_terminal, _ABCMapping):
                    continue
                try:
                    item_index = int(raw_terminal.get("item_index"))
                except (TypeError, ValueError, OverflowError):
                    continue
                terminal_task_id = str(
                    raw_terminal.get("terminal_task_id", "")
                )
                terminal_stage_id = task_stage_ids.get(terminal_task_id)
                raw_group_ids = raw_terminal.get(
                    "operator_invocation_group_ids", ()
                )
                group_ids = (
                    tuple(str(item) for item in raw_group_ids)
                    if isinstance(raw_group_ids, _ABCSequence)
                    and not isinstance(raw_group_ids, (str, bytes, _ABCMapping))
                    else ()
                )
                group_terminal_ns = max(
                    (
                        realized_end_by_group[group_id]
                        for group_id in group_ids
                        if group_id in realized_end_by_group
                    ),
                    default=end_ns,
                )
                terminal_ns = (
                    realized_end_by_stage.get(
                        terminal_stage_id, group_terminal_ns
                    )
                    if terminal_stage_id is not None
                    else group_terminal_ns
                )
                item_completion_ns[item_index] = terminal_ns
                item_terminal_schedule.append(
                    {
                        **dict(raw_terminal),
                        "terminal_stage_id": terminal_stage_id,
                        "terminal_ns": terminal_ns,
                    }
                )
        execution_start_ns = host_start_ns if host_ns > 0.0 else device_start_ns
        for request_id in cohort.request_ids:
            state = self.states[request_id]
            if state.started_ns is None or execution_start_ns < state.started_ns:
                state.started_ns = execution_start_ns
        causal_transfer_rows = tuple(
            stage
            for stage in realized_stages
            if str(stage.get("stage_id", "")).startswith(
                "runtime.owner_residency_transfer."
            )
        )
        causally_placed_transfer_ns = sum(
            max(0.0, float(stage.get("service_ns", 0.0)))
            for stage in causal_transfer_rows
        )
        owner_transfer_ns = max(
            0.0,
            float(cost.metadata.get("owner_residency_transfer_ns", 0.0)),
        )
        placed_transfer_ids = {
            str(stage.get("stage_id", "")) for stage in causal_transfer_rows
        }
        unattributed_positive_batch_count = 0
        unattributed_positive_bytes = 0
        unattributed_positive_service_ns = 0.0
        unattributed_page_in_bytes = 0
        unattributed_page_in_service_ns = 0.0
        raw_transfer_batches = cost.metadata.get(
            "owner_residency_transfer_batches", ()
        )
        if isinstance(raw_transfer_batches, (str, bytes, _ABCMapping)) or not isinstance(
            raw_transfer_batches, _ABCSequence
        ):
            raw_transfer_batches = ()
        for batch_index, raw_batch in enumerate(raw_transfer_batches):
            if not isinstance(raw_batch, _ABCMapping):
                continue
            try:
                transfer_ns = max(
                    0.0, float(raw_batch.get("transfer_ns", 0.0))
                )
                byte_count = max(0, int(raw_batch.get("byte_count", 0)))
            except (TypeError, ValueError, OverflowError):
                continue
            if transfer_ns <= 0.0:
                continue
            transfer_id = (
                "runtime.owner_residency_transfer.batch{:04d}".format(
                    batch_index
                )
            )
            if transfer_id in placed_transfer_ids:
                continue
            unattributed_positive_batch_count += 1
            unattributed_positive_bytes += byte_count
            unattributed_positive_service_ns += transfer_ns
            if (
                str(raw_batch.get("kind", ""))
                == MigrationKind.PAGE_IN.value
            ):
                unattributed_page_in_bytes += byte_count
                unattributed_page_in_service_ns += transfer_ns
        causal_violation_count = 0
        earliest_consumer_start_by_group: Dict[str, float] = {}
        for row in realized_stages:
            row_id = str(row.get("stage_id", ""))
            if row_id.startswith("runtime.owner_residency_transfer."):
                continue
            row_group_ids = tuple(
                dict.fromkeys(
                    (
                        *((
                            str(row.get("invocation_group_id")),
                        ) if row.get("invocation_group_id") else ()),
                        *tuple(
                            str(item)
                            for item in row.get(
                                "covered_invocation_group_ids", ()
                            )
                        ),
                    )
                )
            )
            start_ns = float(row.get("start_ns", 0.0))
            for group_id in row_group_ids:
                earliest_consumer_start_by_group[group_id] = min(
                    start_ns,
                    earliest_consumer_start_by_group.get(group_id, start_ns),
                )
        for transfer_row in causal_transfer_rows:
            covered_ids = tuple(
                str(item)
                for item in transfer_row.get(
                    "covered_invocation_group_ids", ()
                )
            )
            if not covered_ids:
                causal_violation_count += 1
                continue
            if (
                any(
                    group_id not in earliest_consumer_start_by_group
                    for group_id in covered_ids
                )
                or float(transfer_row.get("end_ns", 0.0))
                > min(
                    earliest_consumer_start_by_group[group_id]
                    for group_id in covered_ids
                )
                + 1.0e-6
            ):
                causal_violation_count += 1
        tail_transfer_ns = max(
            0.0,
            owner_transfer_ns - causally_placed_transfer_ns,
        )
        causality_degraded_reasons = []
        if unattributed_positive_batch_count:
            causality_degraded_reasons.append(
                "positive_transfer_without_resolvable_consumer_anchor"
            )
        if causal_violation_count:
            causality_degraded_reasons.append(
                "placed_transfer_completes_after_or_without_consumer"
            )
        if tail_transfer_ns > 1.0e-6 and not unattributed_positive_batch_count:
            causality_degraded_reasons.append(
                "owner_transfer_service_remains_in_cohort_tail"
            )
        causality_degraded = bool(causality_degraded_reasons)
        cost = BatchCost(
            cost.duration_ns,
            cost.energy_pj,
            {
                **dict(cost.metadata),
                "host_start_ns": host_start_ns,
                "host_end_ns": host_end_ns,
                "device_start_ns": device_start_ns,
                "device_end_ns": end_ns,
                "cpu_gpu_overlap_ns": overlap_ns,
                "gpu_idle_waiting_for_host_ns": gpu_idle_wait_ns,
                "execution_stage_schedule": realized_stages,
                "execution_stage_schedule_mode": (
                    "resource_dag" if realized_stages else "serial_fallback"
                ),
                "execution_stage_serial_fallback_reason": (
                    None if realized_stages else stage_fallback_reason
                ),
                "item_terminal_schedule": tuple(item_terminal_schedule),
                "owner_residency_causal_transfer_stage_count": len(
                    causal_transfer_rows
                ),
                "owner_residency_causally_placed_transfer_ns": (
                    causally_placed_transfer_ns
                ),
                "owner_residency_tail_transfer_ns": tail_transfer_ns,
                "owner_residency_causality_degraded": causality_degraded,
                "owner_residency_causality_degraded_reasons": tuple(
                    causality_degraded_reasons
                ),
                "owner_residency_causal_violation_count": (
                    causal_violation_count
                ),
                "owner_residency_unattributed_positive_batch_count": (
                    unattributed_positive_batch_count
                ),
                "owner_residency_unattributed_positive_bytes": (
                    unattributed_positive_bytes
                ),
                "owner_residency_unattributed_positive_service_ns": (
                    unattributed_positive_service_ns
                ),
                "owner_residency_unattributed_page_in_bytes": (
                    unattributed_page_in_bytes
                ),
                "owner_residency_unattributed_page_in_service_ns": (
                    unattributed_page_in_service_ns
                ),
                "runtime_controller_stage_count": (
                    runtime_controller_stage_count
                ),
                "runtime_controller_task_count": int(
                    runtime_controller_ledger.get("task_count", 0)
                ),
                "runtime_controller_ledger": runtime_controller_ledger,
            },
        )
        self.events.append(
            ServingEvent(
                device_start_ns,
                "batch_start",
                cohort_id=cohort.cohort_id,
                details={
                    "kind": cohort.kind,
                    "request_ids": cohort.request_ids,
                    "tokens": cohort.token_count,
                    "host_start_ns": host_start_ns,
                    "host_end_ns": host_end_ns,
                    "device_start_ns": device_start_ns,
                    "cpu_gpu_overlap_ns": overlap_ns,
                },
            )
        )
        self._append_batch(
            ServingBatch(
                cohort.cohort_id,
                cohort.kind,
                device_start_ns,
                end_ns,
                cohort.request_ids,
                cohort.token_count,
                cost,
                cohort.items,
                cohort.proposal_cost_scale,
                cohort.metadata,
            )
        )
        self._record_prompt_cache_ranges(cohort, cost)
        for item_index, item in enumerate(cohort.items):
            item_end_ns = item_completion_ns.get(item_index, end_ns)
            state = self.states[item.request_id]
            if _batch_item_kv_append_tokens(item) > 0:
                if self.prompt_cache.policy.unified_kv:
                    if self._last_kv_allocation_request_id != item.request_id:
                        state.kv_cache_range_count += 1
                    self._last_kv_allocation_request_id = item.request_id
                elif state.kv_cache_range_count == 0:
                    state.kv_cache_range_count = 1
                if (self.prompt_cache.policy.save_implementation == "llama_cpp_host_tensor_get_combined"
                        and not state.kv_cache_range_error):
                    state.kv_cache_range_count = len(state.kv_cache_range_tokens)
            self._record_kv_traffic(item)
            if item.phase == "prefill":
                state.prefill_cursor += item.token_count
                self.events.append(ServingEvent(item_end_ns, "prefill_chunk_complete", item.request_id, cohort.cohort_id, {"cursor": state.prefill_cursor, "chunk_tokens": item.token_count}))
                if (
                    state.prefill_cursor >= state.spec.prompt_tokens
                    and state.spec.prompt_tokens > 0
                    and state.spec.output_tokens > 0
                    and state.committed == 0
                ):
                    # Authoritative llama.cpp commit 18443257a samples the
                    # target model's final prompt logits immediately after
                    # prompt decode.  Token 1 is therefore visible at the
                    # prefill boundary; speculative drafting and verification
                    # only begin with the still-uncommitted suffix.  Recompute
                    # deliberately cannot enter this branch because it has its
                    # own phase below.
                    state.proposed += 1
                    state.accepted += 1
                    state.committed += 1
                    state.first_token_ns = item_end_ns
                    self.events.append(
                        ServingEvent(
                            item_end_ns,
                            "tokens_committed",
                            item.request_id,
                            cohort.cohort_id,
                            {
                                "source": "prefill",
                                "proposed": 1,
                                "accepted": 1,
                                "main_tokens": 1,
                                "draft_tokens": 0,
                                "verifier_tokens": 1,
                                "committed_tokens": 1,
                                "accepted_draft_tokens": 0,
                                "rejected_draft_tokens": 0,
                                "committed": state.committed,
                                "visible_tokens": 1,
                                "kv_materialized_tokens": 0,
                                "kv_persistent_append_tokens": 0,
                                "kv_temporary_tokens": 0,
                            },
                        )
                    )
                    if state.committed >= state.spec.output_tokens:
                        self._finish(state, item_end_ns)
                elif (
                    state.prefill_cursor >= state.spec.prompt_tokens
                    and state.spec.output_tokens == 0
                ):
                    self._finish(state, item_end_ns)
            elif item.phase == "recompute":
                state.recompute_cursor += item.token_count
                self.recompute_tokens += item.token_count
                self.events.append(ServingEvent(item_end_ns, "kv_recompute_chunk_complete", item.request_id, cohort.cohort_id, {"cursor": state.recompute_cursor, "target": state.recompute_target}))
            else:
                proposed = _batch_item_verifier_tokens(item)
                # Admission has already advanced the request-wide cumulative
                # expectation cursor and stored the exact integer commit.
                # Execution must never round the expectation a second time.
                accepted = _batch_item_committed_tokens(item)
                accepted = min(accepted, state.spec.output_tokens - state.committed)
                materialized = _batch_item_kv_materialized_tokens(item)
                temporary = max(
                    0, materialized - _batch_item_kv_append_tokens(item)
                )
                if item.phase == "mtp":
                    # Proposal capacity is live through verification, then all
                    # temporary pages are released before the accepted prefix
                    # is installed as persistent state.
                    self.ledger.release_temporary(state)
                state.proposed += proposed
                state.accepted += accepted
                state.committed += accepted
                committed_pages = self.ledger.pages_for_tokens(state.cached_tokens)
                self.ledger.resize(state, committed_pages)
                if state.first_token_ns is None:
                    state.first_token_ns = item_end_ns
                self.events.append(
                    ServingEvent(
                        item_end_ns,
                        "tokens_committed",
                        item.request_id,
                        cohort.cohort_id,
                        {
                            "proposed": proposed,
                            "accepted": accepted,
                            "main_tokens": _batch_item_main_tokens(item),
                            "draft_tokens": _batch_item_draft_tokens(item),
                            "verifier_tokens": proposed,
                            "committed_tokens": accepted,
                            "accepted_draft_tokens": max(
                                0,
                                accepted - _batch_item_main_tokens(item),
                            ),
                            "rejected_draft_tokens": max(
                                0,
                                _batch_item_draft_tokens(item)
                                - max(
                                    0,
                                    accepted - _batch_item_main_tokens(item),
                                ),
                            ),
                            "committed": state.committed,
                            "visible_tokens": accepted,
                            "kv_materialized_tokens": materialized,
                            "kv_persistent_append_tokens": (
                                _batch_item_kv_append_tokens(item)
                            ),
                            "kv_temporary_tokens": temporary,
                        },
                    )
                )
                if state.committed >= state.spec.output_tokens:
                    self._finish(state, item_end_ns)
            # Advance fairness only after this compute item has completed and
            # its request state has been committed.  Admission, preemption,
            # swap, and candidates skipped during reservation never reach
            # this point and therefore cannot masquerade as service.
            self._service_sequence += 1
            state.last_service_sequence = self._service_sequence
            state.queued_since_ns = item_end_ns
            self._request_device_ready_ns[item.request_id] = item_end_ns
        self.events.append(ServingEvent(end_ns, "batch_end", cohort_id=cohort.cohort_id, details={"kind": cohort.kind}))
        self.now = end_ns

    def _finish(self, state: _MutableRequest, timestamp_ns: float) -> None:
        if state.status == RequestStatus.FINISHED:
            return
        if self._prompt_cache_save_enabled() and self.prompt_cache.policy.recurrent_state_layout is not None:
            self._capture_prompt_cache_recurrent_identity(state)
        self.ledger.release_temporary(state)
        self.ledger.resize(state, 0)
        self.state_ledger.release(state)
        if state.swap_bytes:
            self.ledger.discard_offload(state)
        if state.linear_state_swapped_bytes:
            self.state_ledger.discard_offload(state)
        # llama.cpp's prompt_save returns false for an empty prompt; do not
        # manufacture a zero-cost save event when no physical KV range exists.
        defer_prompt_cache_save = (
            self._prompt_cache_save_enabled()
            and state.kv_cache_range_count > 0
        )
        prompt_cache_entry = (
            None
            if defer_prompt_cache_save
            else self.prompt_cache.add_completed(state, timestamp_ns)
        )
        if defer_prompt_cache_save:
            self._pending_prompt_cache_states.append(state)
        self._set_status(state, RequestStatus.FINISHED)
        state.finished_ns = timestamp_ns
        finish_details: Dict[str, object] = {"committed": state.committed}
        if defer_prompt_cache_save:
            finish_details["prompt_cache_save_deferred"] = True
        if prompt_cache_entry is not None:
            finish_details.update(
                {
                    "prompt_cache_entry_bytes": prompt_cache_entry.allocation_bytes,
                    "prompt_cache_resident_bytes": prompt_cache_entry.resident_bytes,
                    "prompt_cache_host_backed_bytes": prompt_cache_entry.host_backed_bytes,
                    "prompt_cache_checkpoint_segments": len(
                        prompt_cache_entry.segment_sizes
                    ),
                }
            )
        self.events.append(
            ServingEvent(
                timestamp_ns,
                "request_finished",
                state.spec.request_id,
                details=finish_details,
            )
        )

    def _recover_stalled(self) -> bool:
        unfinished = [
            state for state in self._state_values
            if state.status not in _TERMINAL_STATUSES
            and state.status != RequestStatus.ARRIVALS
        ]
        if not unfinished:
            return False
        # A lone swapped request must always be resumable because impossible
        # per-request KV footprints were rejected before the simulation.
        for state in sorted(unfinished, key=self._rank):
            if state.status == RequestStatus.SWAPPED:
                if state.preemption_strategy == "swap" and not self._resume_swap(state):
                    continue
                if (
                    state.preemption_strategy != "swap"
                    and self.plan.kv_policy.allocation_policy == "eager"
                ):
                    target_pages = self.ledger.pages_for_tokens(
                        _max_live_kv_tokens(
                            state.spec.prompt_tokens, state.spec.output_tokens
                        )
                    )
                    if not self._reserve_with_pressure(state, target_pages):
                        continue
                self._set_status(state, RequestStatus.RUNNING)
                state.preemption_strategy = None
                state.queued_since_ns = self.now
                self.events.append(
                    ServingEvent(
                        self.now,
                        "request_resumed",
                        state.spec.request_id,
                    )
                )
                return True
        for state in unfinished:
            reason = "scheduler could not make progress"
            self._set_status(state, RequestStatus.REJECTED)
            state.finished_ns = self.now
            state.rejection_reason = reason
            self.events.append(ServingEvent(self.now, "request_rejected", state.spec.request_id, details={"reason": reason}))
        return False

    def _result(self) -> ServingResult:
        states: Dict[str, ServingRequestState] = {}
        metrics: Dict[str, ServingRequestMetrics] = {}
        for request_id in sorted(self.states):
            state = self.states[request_id]
            spec = state.spec
            states[request_id] = ServingRequestState(
                request_id, state.status, spec.arrival_ns, spec.prompt_tokens, spec.output_tokens,
                state.prefill_cursor, state.committed, state.proposed, state.accepted,
                spec.priority, spec.deadline_ns, state.kv_pages, state.peak_kv_pages,
                state.started_ns, state.first_token_ns, state.finished_ns,
                state.preemptions, state.swaps, state.recomputes,
                state.rejection_reason,
            )
            queue_delay = state.started_ns - spec.arrival_ns if state.started_ns is not None else None
            ttft = state.first_token_ns - spec.arrival_ns if state.first_token_ns is not None else None
            tpot = None
            if state.first_token_ns is not None and state.finished_ns is not None and state.committed > 1:
                tpot = (state.finished_ns - state.first_token_ns) / (state.committed - 1)
            if spec.deadline_ns is None:
                deadline_met = None
            elif state.status == RequestStatus.FINISHED and state.finished_ns is not None:
                deadline_met = state.finished_ns <= spec.deadline_ns
            else:
                deadline_met = False
            metrics[request_id] = ServingRequestMetrics(
                request_id, state.status, spec.arrival_ns, state.started_ns, state.first_token_ns,
                state.finished_ns, queue_delay, ttft, tpot, spec.prompt_tokens,
                spec.output_tokens, state.committed, state.proposed, state.accepted,
                state.preemptions, state.swaps, state.recomputes, deadline_met,
                state.rejection_reason,
            )
        self.events.sort(key=_event_sort_key)
        events = tuple(self.events)
        batch_kind_counts = self._batch_kind_counts
        kv_metrics = KVCacheMetrics(
            self.plan.kv_policy.tokens_per_page,
            self.plan.kv_policy.bytes_per_page,
            self.plan.kv_policy.capacity_pages,
            self.plan.kv_policy.capacity_bytes,
            self.ledger.peak_pages,
            self.ledger.peak_pages * self.plan.kv_policy.bytes_per_page,
            self.ledger.allocations,
            self.ledger.releases,
            self.swap_events,
            self.swap_bytes,
            self.recompute_events,
            self.recompute_tokens,
            self.ledger.offload_peak_bytes,
            self._rejected_count,
            self.swap_in_bytes,
            self.swap_transfer_time_ns,
            self.swap_transfer_energy_pj,
            logical_prefill_read_bytes=self.logical_prefill_read_bytes,
            logical_prefill_write_bytes=self.logical_prefill_write_bytes,
            logical_decode_read_bytes=self.logical_decode_read_bytes,
            logical_decode_write_bytes=self.logical_decode_write_bytes,
            logical_decode_append_bytes=self.logical_decode_write_bytes,
            physical_prefill_read_bytes=self.physical_prefill_read_bytes,
            physical_prefill_write_bytes=self.physical_prefill_write_bytes,
            physical_decode_read_bytes=self.physical_decode_read_bytes,
            physical_decode_write_bytes=self.physical_decode_write_bytes,
            physical_decode_append_bytes=self.physical_decode_write_bytes,
            offload_events=self.swap_events,
            offload_bytes=self.swap_bytes,
            migration_events=self.swap_events
            + batch_kind_counts.get("kv_swap_in", 0),
            migration_bytes=self.swap_bytes + self.swap_in_bytes,
            swap_out_bytes=self.swap_bytes,
            physical_swap_bytes=self.swap_bytes + self.swap_in_bytes,
            logical_bytes_per_token=self.plan.kv_policy.logical_bytes_per_token,
            logical_swap_out_bytes=self.logical_swap_out_bytes,
            logical_swap_in_bytes=self.logical_swap_in_bytes,
            logical_migration_bytes=self.logical_swap_out_bytes
            + self.logical_swap_in_bytes,
            physical_offload_bytes=self.swap_bytes,
            max_live_tokens_per_request=max(
                (
                    _max_live_kv_tokens(
                        state.spec.prompt_tokens, state.spec.output_tokens
                    )
                    for state in self._state_values
                    if state.status != RequestStatus.REJECTED
                ),
                default=0,
            ),
            peak_semantics=(
                "realized_persistent_plus_mtp_temporary_reservation"
                if self.plan.mtp.enabled
                else "realized_prompt_plus_output_minus_one"
            ),
            traffic_semantics=(
                "persistent_kv_traffic_separate_from_mtp_temporary_"
                "verification_and_resource_accounting"
            ),
            persistent_peak_used_pages=self.ledger.persistent_peak_pages,
            persistent_peak_used_bytes=(
                self.ledger.persistent_peak_pages
                * self.plan.kv_policy.bytes_per_page
            ),
            mtp_materialized_tokens=self.mtp_materialized_tokens,
            mtp_temporary_tokens=self.mtp_temporary_tokens,
            mtp_temporary_peak_pages=self.ledger.temporary_peak_pages,
            mtp_temporary_peak_bytes=(
                self.ledger.temporary_peak_pages
                * self.plan.kv_policy.bytes_per_page
            ),
            mtp_temporary_allocation_events=(
                self.ledger.temporary_allocations
            ),
            mtp_temporary_release_events=self.ledger.temporary_releases,
            logical_mtp_materialized_write_bytes=(
                self.logical_mtp_materialized_write_bytes
            ),
            physical_mtp_materialized_write_bytes=(
                self.physical_mtp_materialized_write_bytes
            ),
            logical_mtp_temporary_write_bytes=(
                self.logical_mtp_temporary_write_bytes
            ),
            physical_mtp_temporary_write_bytes=(
                self.physical_mtp_temporary_write_bytes
            ),
            logical_mtp_verification_read_bytes=(
                self.logical_mtp_verification_read_bytes
            ),
            physical_mtp_verification_read_bytes=(
                self.physical_mtp_verification_read_bytes
            ),
        )
        linear_state_metrics = LinearStateMetrics(
            self.plan.linear_state_policy.bytes_per_request,
            self.plan.linear_state_policy.capacity_bytes,
            self.plan.linear_state_policy.capacity_requests,
            self.state_ledger.peak_requests
            * self.plan.linear_state_policy.bytes_per_request,
            self.state_ledger.peak_requests,
            self.state_ledger.allocations,
            self.state_ledger.releases,
            self.state_ledger.offload_events,
            self.state_ledger.offload_bytes,
            self.state_ledger.offload_peak_bytes,
            self.state_ledger.restore_events,
            self.linear_state_swap_in_bytes,
            self.linear_state_swap_transfer_time_ns,
            self.linear_state_swap_transfer_energy_pj,
            self.linear_state_swap_routed_bytes,
        )
        scheduler_metrics = SchedulerMetrics(
            self.rounds,
            len(self.batches),
            self._batch_phase_counts.get("prefill", 0),
            self._batch_phase_counts.get("decode", 0),
            self._batch_phase_counts.get("mtp", 0),
            self.preemptions,
            self.priority_preemptions,
            self.memory_preemptions,
            self._max_batch_sequences,
            self._max_batch_tokens,
            self.idle_ns,
            batch_kind_counts.get("kv_swap_out", 0)
            + batch_kind_counts.get("kv_swap_in", 0),
            batch_kind_counts.get("linear_state_swap_out", 0)
            + batch_kind_counts.get("linear_state_swap_in", 0),
        )
        if self.residency_manager is None:
            owner_residency_metrics = OwnerResidencyMetrics()
        else:
            # ``_capture_residency_interval_peak`` is normally called after
            # every owner mutation.  Sample once more at the result boundary
            # so a replay with no explicit access trace still reports its
            # physical endpoint and so the final value is never stale.
            self._capture_residency_interval_peak()
            owner_snapshot = self.residency_manager.snapshot()
            interval_baseline = int(
                self._residency_interval_baseline_bytes or 0
            )
            interval_peak = max(
                interval_baseline,
                int(self._residency_interval_peak_bytes or 0),
                int(owner_snapshot.resident_bytes),
            )
            owner_residency_metrics = OwnerResidencyMetrics(
                enabled=True,
                component_id=owner_snapshot.component_id,
                capacity_bytes=owner_snapshot.capacity_bytes,
                committed_bytes=owner_snapshot.committed_bytes,
                resident_bytes=owner_snapshot.resident_bytes,
                peak_resident_bytes=owner_snapshot.peak_resident_bytes,
                available_bytes=owner_snapshot.available_bytes,
                allocation_count=owner_snapshot.allocation_count,
                view_count=owner_snapshot.view_count,
                access_count=self.residency_accesses,
                weight_access_count=self.residency_weight_accesses,
                kv_access_count=self.residency_kv_accesses,
                state_access_count=self.residency_state_accesses,
                temporary_allocation_count=(
                    self.residency_temporary_allocations
                ),
                temporary_release_count=self.residency_temporary_releases,
                migration_event_count=(
                    self.residency_manager.migration_count
                    - self._residency_measurement_migration_start
                ),
                fault_batch_count=self.residency_fault_batches,
                clean_eviction_batch_count=(
                    self.residency_clean_eviction_batches
                ),
                page_in_bytes=self.residency_page_in_bytes,
                page_out_bytes=self.residency_page_out_bytes,
                clean_discard_bytes=self.residency_clean_discard_bytes,
                dirty_writeback_bytes=self.residency_dirty_writeback_bytes,
                clean_discard_time_ns=self.residency_clean_discard_time_ns,
                clean_discard_energy_pj=(
                    self.residency_clean_discard_energy_pj
                ),
                transfer_time_ns=self.residency_transfer_time_ns,
                transfer_energy_pj=self.residency_transfer_energy_pj,
                initialization_migration_event_count=(
                    len(self._residency_initialization_migrations)
                ),
                initialization_page_in_bytes=(
                    self.residency_initialization_page_in_bytes
                ),
                initialization_page_out_bytes=(
                    self.residency_initialization_page_out_bytes
                ),
                initialization_clean_discard_bytes=(
                    self.residency_initialization_clean_discard_bytes
                ),
                initialization_dirty_writeback_bytes=(
                    self.residency_initialization_dirty_writeback_bytes
                ),
                timing_completeness=(
                    self.resource_policy.timing_completeness
                ),
                submission_latency_known=(
                    self.resource_policy.page_fault_latency_known
                ),
                unmodeled_timing_terms=(
                    ()
                    if self.resource_policy.page_fault_latency_known
                    else ("make_resident_submission_latency",)
                ),
                interval_baseline_resident_bytes=interval_baseline,
                interval_peak_resident_bytes=interval_peak,
                interval_resident_delta_bytes=max(
                    0, interval_peak - interval_baseline
                ),
            )
        makespan = max((state.finished_ns or 0.0 for state in self._state_values), default=0.0)
        return ServingResult(
            self.plan,
            events,
            tuple(self.batches),
            states,
            metrics,
            kv_metrics,
            linear_state_metrics,
            scheduler_metrics,
            makespan,
            prompt_cache_metrics=self.prompt_cache.metrics(),
            owner_residency_metrics=owner_residency_metrics,
            runtime_kernel_metrics=dict(
                self._execution_resource_kernel.metrics
            ),
        )


def _lower_cost(lowerer: BatchLowerer, scenario: ScenarioConfig, cohort: BatchCohort) -> BatchCost:
    estimator = getattr(lowerer, "estimate", None)
    if callable(estimator):
        raw = estimator(scenario, cohort)
    elif callable(lowerer):
        signature = inspect.signature(lowerer)
        positional = [
            parameter for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is parameter.empty
        ]
        raw = lowerer(cohort) if len(positional) <= 1 else lowerer(scenario, cohort)
    else:
        raise TypeError("batch_lowerer must be callable or define estimate(scenario, cohort)")
    if isinstance(raw, BatchCost):
        cost = raw
    elif isinstance(raw, _ABCMapping):
        cost = BatchCost(float(raw["duration_ns"]), float(raw.get("energy_pj", 0.0)), dict(raw.get("metadata", {})))
    else:
        cost = BatchCost(float(raw))
    if not math.isfinite(cost.duration_ns) or cost.duration_ns <= 0:
        raise ValueError("batch cost duration_ns must be finite and positive")
    return cost


def _execution_stages_from_metadata(
    metadata: Mapping[str, Any],
) -> Tuple[Tuple[_ExecutionStage, ...], Optional[str]]:
    raw_stages = metadata.get("execution_stages")
    if raw_stages is None:
        return (), "batch cost has no execution_stages"
    raw_stages_type = type(raw_stages)
    if (
        raw_stages_type is not tuple
        and raw_stages_type is not list
        and (
            isinstance(raw_stages, (str, bytes, _ABCMapping))
            or not isinstance(raw_stages, _ABCSequence)
        )
    ):
        return (), "execution_stages is not an ordered sequence"
    stages: List[_ExecutionStage] = []
    seen = set()
    detailed_task_source = (
        metadata.get("execution_stage_source")
        == "executed_task_dag_kernel_timeline"
    )
    for index, raw_stage in enumerate(raw_stages):
        if type(raw_stage) is not dict and not isinstance(
            raw_stage, _ABCMapping
        ):
            return (), "execution_stages[{}] is not a mapping".format(index)
        raw_stage_id = raw_stage.get("stage_id")
        raw_component_id = raw_stage.get("component_id")
        raw_group_id = raw_stage.get("group_id")
        raw_covered_group_ids = raw_stage.get(
            "covered_invocation_group_ids", ()
        )
        if not isinstance(raw_stage_id, str) or not isinstance(
            raw_component_id, str
        ):
            return (), "execution stage identity/component is invalid"
        stage_id = raw_stage_id
        component_id = raw_component_id
        if raw_group_id is not None and (
            not isinstance(raw_group_id, str) or not raw_group_id
        ):
            return (), "execution stage invocation-group identity is invalid"
        invocation_group_id = (
            str(raw_group_id) if raw_group_id is not None else None
        )
        if isinstance(
            raw_covered_group_ids, (str, bytes, _ABCMapping)
        ) or not isinstance(raw_covered_group_ids, _ABCSequence):
            return (), "execution stage covered invocation groups are invalid"
        if any(
            not isinstance(item, str) or not item
            for item in raw_covered_group_ids
        ):
            return (), "execution stage covered invocation groups are invalid"
        covered_invocation_group_ids = tuple(raw_covered_group_ids)
        if len(covered_invocation_group_ids) != len(
            set(covered_invocation_group_ids)
        ):
            return (), "execution stage covered invocation groups are duplicated"
        if not stage_id or stage_id in seen or not component_id:
            return (), "execution stage identity/component is invalid"
        raw_dependencies = raw_stage.get("dependencies", ())
        raw_request_ids = raw_stage.get("request_ids", ())
        raw_dependencies_type = type(raw_dependencies)
        raw_request_ids_type = type(raw_request_ids)
        if (
            (
                raw_dependencies_type is not tuple
                and raw_dependencies_type is not list
                and (
                    isinstance(raw_dependencies, (str, bytes, _ABCMapping))
                    or not isinstance(raw_dependencies, _ABCSequence)
                )
            )
            or (
                raw_request_ids_type is not tuple
                and raw_request_ids_type is not list
                and (
                    isinstance(raw_request_ids, (str, bytes, _ABCMapping))
                    or not isinstance(raw_request_ids, _ABCSequence)
                )
            )
        ):
            return (), "execution stage dependencies/request_ids are invalid"
        if any(
            not isinstance(item, str) or not item
            for item in raw_dependencies
        ) or any(
            not isinstance(item, str) or not item
            for item in raw_request_ids
        ):
            return (), "execution stage dependencies/request_ids are invalid"
        dependencies = tuple(raw_dependencies)
        dependency_set = set(dependencies)
        if len(dependencies) != len(dependency_set):
            return (), "execution stage dependencies are duplicated"
        if any(dependency not in seen for dependency in dependencies):
            return (), "execution stages are not in topological order"
        try:
            raw_service_ns = raw_stage.get("service_ns", 0.0)
            raw_stage_index = raw_stage.get("stage_index", index)
            if isinstance(raw_service_ns, bool) or isinstance(
                raw_stage_index, bool
            ):
                raise ValueError
            service_ns = float(raw_service_ns)
            stage_index = int(raw_stage_index)
            if (
                not math.isfinite(float(raw_stage_index))
                or float(raw_stage_index) != stage_index
            ):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return (), "execution stage service/index is invalid"
        if not math.isfinite(service_ns) or service_ns <= 0.0:
            return (), "execution stage service must be finite and positive"
        request_ids = tuple(raw_request_ids)
        if not request_ids:
            return (), "execution stage has no request identity"
        request_id_set = set(request_ids)
        if len(request_ids) != len(request_id_set):
            return (), "execution stage request_ids are duplicated"

        raw_tasks = raw_stage.get("execution_tasks")
        if raw_tasks is None:
            if detailed_task_source:
                return (), "execution stage is missing execution_tasks"
            execution_tasks = (
                _ExecutionTask(
                    stage_id + ".task0000",
                    (),
                    request_ids,
                    (ResourceDemand(component_id, service_ns),),
                    True,
                ),
            )
        else:
            raw_tasks_type = type(raw_tasks)
            if (
                raw_tasks_type is not tuple
                and raw_tasks_type is not list
                and (
                    isinstance(raw_tasks, (str, bytes, _ABCMapping))
                    or not isinstance(raw_tasks, _ABCSequence)
                )
            ):
                return (), (
                    "execution stage execution_tasks is not an ordered sequence"
                )
            if not raw_tasks:
                return (), "execution stage execution_tasks is empty"
            execution_task_rows: List[_ExecutionTask] = []
            seen_task_ids = set()
            for raw_task in raw_tasks:
                if type(raw_task) is not dict and not isinstance(
                    raw_task, _ABCMapping
                ):
                    return (), "execution task is not a mapping"
                raw_task_id = raw_task.get("task_id")
                if (
                    not isinstance(raw_task_id, str)
                    or not raw_task_id
                    or raw_task_id in seen_task_ids
                ):
                    return (), "execution task identity is invalid"
                raw_task_dependencies = raw_task.get("dependencies", ())
                raw_task_request_ids = raw_task.get("request_ids", ())
                raw_demands = raw_task.get("resource_demands")
                raw_opaque_device_fence = raw_task.get(
                    "opaque_device_fence", False
                )
                if not isinstance(raw_opaque_device_fence, bool):
                    return (), "execution task opaque device fence is invalid"
                raw_task_dependencies_type = type(raw_task_dependencies)
                raw_task_request_ids_type = type(raw_task_request_ids)
                raw_demands_type = type(raw_demands)
                if (
                    (
                        raw_task_dependencies_type is not tuple
                        and raw_task_dependencies_type is not list
                        and (
                            isinstance(
                                raw_task_dependencies,
                                (str, bytes, _ABCMapping),
                            )
                            or not isinstance(
                                raw_task_dependencies, _ABCSequence
                            )
                        )
                    )
                    or (
                        raw_task_request_ids_type is not tuple
                        and raw_task_request_ids_type is not list
                        and (
                            isinstance(
                                raw_task_request_ids,
                                (str, bytes, _ABCMapping),
                            )
                            or not isinstance(
                                raw_task_request_ids, _ABCSequence
                            )
                        )
                    )
                    or (
                        raw_demands_type is not tuple
                        and raw_demands_type is not list
                        and (
                            isinstance(raw_demands, (str, bytes, _ABCMapping))
                            or not isinstance(raw_demands, _ABCSequence)
                        )
                    )
                ):
                    return (), (
                        "execution task dependencies/request_ids/demands "
                        "are invalid"
                    )
                if any(
                    not isinstance(item, str) or not item
                    for item in raw_task_dependencies
                ) or any(
                    not isinstance(item, str) or not item
                    for item in raw_task_request_ids
                ):
                    return (), "execution task dependencies/request_ids are invalid"
                task_dependencies = tuple(raw_task_dependencies)
                task_request_ids = tuple(raw_task_request_ids)
                task_dependency_set = set(task_dependencies)
                if (
                    len(task_dependencies) != len(task_dependency_set)
                    or any(
                        dependency not in seen_task_ids
                        for dependency in task_dependencies
                    )
                ):
                    return (), "execution tasks are not in topological order"
                task_request_id_set = set(task_request_ids)
                if (
                    not task_request_ids
                    or len(task_request_ids) != len(task_request_id_set)
                    or not task_request_id_set.issubset(request_id_set)
                ):
                    return (), "execution task request_ids are invalid"
                demands: List[ResourceDemand] = []
                demand_resource_ids = set()
                for raw_demand in raw_demands:
                    if type(raw_demand) is not dict and not isinstance(
                        raw_demand, _ABCMapping
                    ):
                        return (), (
                            "execution task resource demand is not a mapping"
                        )
                    resource_id = raw_demand.get("resource_id")
                    raw_demand_service = raw_demand.get("service_ns")
                    if (
                        not isinstance(resource_id, str)
                        or not resource_id
                        or resource_id in demand_resource_ids
                        or isinstance(raw_demand_service, bool)
                    ):
                        return (), "execution task resource demand is invalid"
                    try:
                        demand_service_ns = float(raw_demand_service)
                    except (TypeError, ValueError, OverflowError):
                        return (), "execution task resource demand is invalid"
                    if (
                        not math.isfinite(demand_service_ns)
                        or demand_service_ns < 0.0
                    ):
                        return (), "execution task resource demand is invalid"
                    raw_bytes_moved = raw_demand.get("bytes_moved", 0)
                    raw_energy_pj = raw_demand.get("energy_pj", 0.0)
                    raw_work_units = raw_demand.get("work_units", 0.0)
                    if (
                        isinstance(raw_bytes_moved, bool)
                        or not isinstance(raw_bytes_moved, int)
                        or raw_bytes_moved < 0
                        or isinstance(raw_energy_pj, bool)
                        or isinstance(raw_work_units, bool)
                    ):
                        return (), "execution task resource demand is invalid"
                    try:
                        bytes_moved_numeric = float(raw_bytes_moved)
                        energy_pj = float(raw_energy_pj)
                        work_units = float(raw_work_units)
                    except (TypeError, ValueError, OverflowError):
                        return (), "execution task resource demand is invalid"
                    if (
                        not math.isfinite(bytes_moved_numeric)
                        or not math.isfinite(energy_pj)
                        or energy_pj < 0.0
                        or not math.isfinite(work_units)
                        or work_units < 0.0
                    ):
                        return (), "execution task resource demand is invalid"
                    demands.append(
                        ResourceDemand(
                            resource_id,
                            demand_service_ns,
                            bytes_moved=raw_bytes_moved,
                            energy_pj=energy_pj,
                            work_units=work_units,
                        )
                    )
                    demand_resource_ids.add(resource_id)
                raw_metadata = raw_task.get("metadata")
                source_metadata = {
                    key: dict(value)
                    for key in ("mmq_source_work", "native_kv_work")
                    for value in ((raw_metadata.get(key) if isinstance(raw_metadata, _ABCMapping) else None),)
                    if isinstance(value, _ABCMapping)
                }
                execution_task_rows.append(
                    _ExecutionTask(
                        raw_task_id,
                        task_dependencies,
                        task_request_ids,
                        tuple(demands),
                        opaque_device_fence=raw_opaque_device_fence,
                        metadata=source_metadata,
                    )
                )
                seen_task_ids.add(raw_task_id)
            if not any(
                demand.service_ns > 0.0
                for task in execution_task_rows
                for demand in task.demands
            ):
                return (), "execution stage task graph has no positive service"
            execution_tasks = tuple(execution_task_rows)
        stages.append(
            _ExecutionStage(
                stage_id,
                stage_index,
                dependencies,
                request_ids,
                component_id,
                service_ns,
                execution_tasks,
                invocation_group_id,
                covered_invocation_group_ids,
            )
        )
        seen.add(stage_id)
    if not stages:
        return (), "execution_stages is empty"
    return tuple(stages), None


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None or str(value) == "" else str(value)


_SERVING_RUNTIME_METADATA_KEY = "serving_runtime"


def _metadata_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("{} must be a boolean".format(key))


def _metadata_nonnegative_float(value: Any, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite non-negative number".format(key))
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("{} must be a finite non-negative number".format(key))
    return number


def _metadata_positive_float(value: Any, key: str) -> float:
    number = _metadata_nonnegative_float(value, key)
    if number <= 0.0:
        raise ValueError("{} must be positive".format(key))
    return number


def _metadata_nonnegative_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be a non-negative integer".format(key))
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be a non-negative integer".format(key))
    if number < 0 or (isinstance(value, float) and value != number):
        raise ValueError("{} must be a non-negative integer".format(key))
    return number


def _metadata_byte_sequence(value: Any, key: str) -> Tuple[int, ...]:
    """Parse observed byte facts without turning them into a lookup table."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raw_values = [part.strip() for part in str(value).split(",") if part.strip()]
    elif isinstance(value, _ABCSequence) and not isinstance(value, (str, bytes)):
        raw_values = list(value)
    else:
        raw_values = [value]
    result: List[int] = []
    for item in raw_values:
        result.append(
            _metadata_nonnegative_int(item, "{} entry size".format(key))
        )
    return tuple(result)


def _serving_resource_policy(scenario: ScenarioConfig) -> _ServingResourcePolicy:
    """Read optional serving resource controls from scenario metadata.

    ``workload.metadata.serving_runtime`` is the canonical per-workload
    location.  ``placement.metadata.serving_runtime`` is accepted as a
    scenario-wide fallback, which is useful when a mapper owns the runtime
    contract.  Workload values override placement values.  A nested
    ``resource_contention`` mapping is accepted for readability, while flat
    keys remain the documented form.
    """

    values: Dict[str, Any] = {}
    prompt_values: Dict[str, Any] = {}
    placement_metadata = getattr(scenario.placement, "metadata", {})
    workload_metadata = getattr(scenario.workload, "metadata", {})
    for metadata in (placement_metadata, workload_metadata):
        if not isinstance(metadata, _ABCMapping):
            continue
        section = metadata.get(_SERVING_RUNTIME_METADATA_KEY)
        if not isinstance(section, _ABCMapping):
            continue
        values.update(
            {
                str(key): value
                for key, value in section.items()
                if key != "resource_contention"
            }
        )
        nested = section.get("resource_contention")
        if isinstance(nested, _ABCMapping):
            values.update({str(key): value for key, value in nested.items()})
        raw_prompt = section.get("prompt_cache")
        if isinstance(raw_prompt, _ABCMapping):
            prompt_values.update(
                {str(key): value for key, value in raw_prompt.items()}
            )
        elif raw_prompt is not None:
            prompt_values["enabled"] = raw_prompt
        for key in (
            "prompt_cache_enabled",
            "prompt_cache_context_checkpoints",
            "prompt_cache_component",
            "prompt_cache_offload_component",
            "prompt_cache_slot_allocation_bytes",
            "prompt_cache_entry_bytes",
            "prompt_cache_resident_entry_bytes",
            "prompt_cache_graph_residency_mode",
            "prompt_cache_retain_completed",
            "prompt_cache_observed_entry_sizes_bytes",
            "prompt_cache_observed_entry_sizes_mib",
            "prompt_cache_save_implementation",
            "prompt_cache_unified_kv",
            "prompt_cache_apply_tensor_get_submission_service",
            "prompt_cache_apply_tensor_get_controller_phases",
        ):
            if key in section:
                prompt_values[key] = section[key]

    enabled = values.get("contention_enabled", values.get("enabled", False))
    raw_bandwidth = values.get("page_transfer_bandwidth_gb_s")
    raw_fault_latency = values.get("page_fault_latency_ns")
    fault_latency_known = _metadata_bool(
        values.get("page_fault_latency_known", True),
        "serving_runtime.page_fault_latency_known",
    )
    if not fault_latency_known and raw_fault_latency is not None:
        raise ValueError(
            "serving_runtime.page_fault_latency_ns must be omitted or null "
            "when page_fault_latency_known is false"
        )
    timing_completeness = str(
        values.get(
            "timing_completeness",
            "complete" if fault_latency_known else "partial",
        )
    ).strip().lower()
    if timing_completeness not in {"complete", "partial"}:
        raise ValueError(
            "serving_runtime.timing_completeness must be complete or partial"
        )
    if not fault_latency_known and timing_completeness != "partial":
        raise ValueError(
            "unknown page-fault/submission latency requires partial timing"
        )
    raw_residency_granule = values.get("residency_granule_bytes")
    raw_fault_batch = values.get("fault_batch_bytes")
    fault_latency_scope = str(
        values.get("fault_latency_scope", "granule")
    ).strip().lower()
    if fault_latency_scope in {"batch", "migration", "cohort"}:
        fault_latency_scope = "make_resident_batch"
    if fault_latency_scope not in {"granule", "make_resident_batch"}:
        raise ValueError(
            "unsupported serving_runtime.fault_latency_scope"
        )
    raw_slot_context = values.get("kv_slot_context_tokens")
    prompt_enabled = prompt_values.get(
        "enabled", prompt_values.get("prompt_cache_enabled", False)
    )
    prompt_context_checkpoints = prompt_values.get(
        "context_checkpoints",
        prompt_values.get("prompt_cache_context_checkpoints", 1),
    )
    prompt_component = prompt_values.get(
        "component", prompt_values.get("prompt_cache_component")
    )
    prompt_offload_component = prompt_values.get(
        "offload_component",
        prompt_values.get("prompt_cache_offload_component"),
    )
    raw_slot_allocation = prompt_values.get(
        "slot_allocation_bytes",
        prompt_values.get(
            "full_slot_bytes",
            prompt_values.get(
                "logical_entry_bytes",
                prompt_values.get("prompt_cache_slot_allocation_bytes"),
            ),
        ),
    )
    raw_entry_bytes = prompt_values.get(
        "entry_bytes", prompt_values.get("prompt_cache_entry_bytes")
    )
    raw_resident_entry_bytes = prompt_values.get(
        "resident_entry_bytes",
        prompt_values.get(
            "graph_resident_bytes",
            prompt_values.get("prompt_cache_resident_entry_bytes"),
        ),
    )
    raw_kv_slot_allocation = prompt_values.get(
        "kv_slot_allocation_bytes"
    )
    raw_state_slot_allocation = prompt_values.get(
        "state_slot_allocation_bytes"
    )
    save_implementation = str(
        prompt_values.get(
            "save_implementation",
            prompt_values.get("prompt_cache_save_implementation", "unmodeled"),
        )
    ).strip().lower()
    if save_implementation not in {
        "unmodeled", "llama_cpp_host_tensor_get", "llama_cpp_host_tensor_get_combined"
    }:
        raise ValueError(
            "unsupported serving_runtime.prompt_cache.save_implementation"
        )
    if any(prompt_values.get(key) is not None for key in (
        "measured_tensor_get_service_ns", "measured_tensor_get_service_source"
    )):
        raise ValueError(
            "measured tensor-get service is retired: remove calibration fields; "
            "combined saving derives known work from hardware profiles and reports unknown service"
        )
    submission_enabled = _metadata_bool(
        prompt_values.get("apply_tensor_get_submission_service",
                          prompt_values.get("prompt_cache_apply_tensor_get_submission_service", False)),
        "serving_runtime.prompt_cache.apply_tensor_get_submission_service",
    )
    controllers_enabled = _metadata_bool(
        prompt_values.get("apply_tensor_get_controller_phases",
                          prompt_values.get("prompt_cache_apply_tensor_get_controller_phases", False)),
        "serving_runtime.prompt_cache.apply_tensor_get_controller_phases",
    )
    if submission_enabled and controllers_enabled:
        raise ValueError("tensor-get submission and controller proxies are mutually exclusive")
    driver_copy_contract = prompt_values.get("driver_cpu_copy_issue_contract")
    if driver_copy_contract is not None and (
        driver_copy_contract != "nvcuda_616_64_payload_max16_v1"
        or save_implementation != "llama_cpp_host_tensor_get_combined"
    ):
        raise ValueError(
            "driver_cpu_copy_issue_contract requires combined saving and "
            "nvcuda_616_64_payload_max16_v1"
        )
    host_state_layout = prompt_values.get("host_state_layout")
    host_pointer_bytes = prompt_values.get("host_pointer_bytes")
    recurrent_state_layout = prompt_values.get("recurrent_state_layout")
    if recurrent_state_layout is not None and (
        recurrent_state_layout != "llama_cpp_recurrent_rs_v1"
        or save_implementation != "llama_cpp_host_tensor_get_combined"
    ):
        raise ValueError("recurrent_state_layout requires combined saving and llama_cpp_recurrent_rs_v1")
    if host_state_layout is not None and (
        host_state_layout != "llama_cpp_plain_kv_v1"
        or save_implementation != "llama_cpp_host_tensor_get_combined"
    ):
        raise ValueError("host_state_layout requires combined saving and llama_cpp_plain_kv_v1")
    if host_pointer_bytes is not None and (
        host_state_layout is None or isinstance(host_pointer_bytes, bool)
        or not isinstance(host_pointer_bytes, int) or host_pointer_bytes not in (4, 8)
    ):
        raise ValueError("host_pointer_bytes requires the declared state layout and integer 4 or 8")
    unified_kv = _metadata_bool(
        prompt_values.get(
            "unified_kv", prompt_values.get("prompt_cache_unified_kv", False)
        ),
        "serving_runtime.prompt_cache.unified_kv",
    )
    raw_observed = prompt_values.get(
        "observed_entry_sizes_bytes",
        prompt_values.get("prompt_cache_observed_entry_sizes_bytes"),
    )
    observed_entry_sizes = _metadata_byte_sequence(
        raw_observed, "serving_runtime.prompt_cache.observed_entry_sizes_bytes"
    )
    observed_mib = prompt_values.get(
        "observed_entry_sizes_mib",
        prompt_values.get("prompt_cache_observed_entry_sizes_mib"),
    )
    if observed_mib is not None:
        if isinstance(observed_mib, (str, bytes)):
            raw_mib_values = [
                part.strip()
                for part in str(observed_mib).split(",")
                if part.strip()
            ]
        elif isinstance(observed_mib, _ABCSequence):
            raw_mib_values = list(observed_mib)
        else:
            raw_mib_values = [observed_mib]
        parsed_mib: List[int] = []
        for item in raw_mib_values:
            try:
                mib = float(item)
            except (TypeError, ValueError):
                raise ValueError(
                    "serving_runtime.prompt_cache.observed_entry_sizes_mib "
                    "must contain finite non-negative numbers"
                )
            if not math.isfinite(mib) or mib < 0:
                raise ValueError(
                    "serving_runtime.prompt_cache.observed_entry_sizes_mib "
                    "must contain finite non-negative numbers"
                )
            parsed_mib.append(int(round(mib * 1024.0 * 1024.0)))
        observed_entry_sizes = tuple(parsed_mib)
    prompt_policy: Optional[_PromptCachePolicy]
    if _metadata_bool(
        prompt_enabled, "serving_runtime.prompt_cache.enabled"
    ):
        context_checkpoints = _metadata_nonnegative_int(
            prompt_context_checkpoints,
            "serving_runtime.prompt_cache.context_checkpoints",
        )
        if context_checkpoints <= 0:
            raise ValueError(
                "serving_runtime.prompt_cache.context_checkpoints must be positive"
            )
        graph_mode = str(
            prompt_values.get(
                "graph_residency_mode",
                prompt_values.get(
                    "residency_mode",
                    prompt_values.get(
                        "prompt_cache_graph_residency_mode", "actual_tokens"
                    ),
                ),
            )
        ).strip().lower()
        if graph_mode in {"full", "full_slot", "slot", "allocation"}:
            graph_mode = "slot_allocation"
        elif graph_mode in {"actual", "tokens", "actual_token", "token"}:
            graph_mode = "actual_tokens"
        elif graph_mode not in {"actual_tokens", "slot_allocation", "explicit"}:
            raise ValueError(
                "unsupported serving_runtime.prompt_cache.graph_residency_mode"
            )
        retain_completed = _metadata_bool(
            prompt_values.get(
                "retain_completed",
                prompt_values.get("retain_completed_entries", True),
            ),
            "serving_runtime.prompt_cache.retain_completed",
        )
        prompt_policy = _PromptCachePolicy(
            enabled=True,
            context_checkpoints=context_checkpoints,
            component=_optional_str(prompt_component),
            offload_component=_optional_str(prompt_offload_component),
            slot_allocation_bytes=(
                None
                if raw_slot_allocation is None
                else _metadata_nonnegative_int(
                    raw_slot_allocation,
                    "serving_runtime.prompt_cache.slot_allocation_bytes",
                )
            ),
            entry_bytes=(
                None
                if raw_entry_bytes is None
                else _metadata_nonnegative_int(
                    raw_entry_bytes,
                    "serving_runtime.prompt_cache.entry_bytes",
                )
            ),
            resident_entry_bytes=(
                None
                if raw_resident_entry_bytes is None
                else _metadata_nonnegative_int(
                    raw_resident_entry_bytes,
                    "serving_runtime.prompt_cache.resident_entry_bytes",
                )
            ),
            kv_slot_allocation_bytes=(
                None
                if raw_kv_slot_allocation is None
                else _metadata_nonnegative_int(
                    raw_kv_slot_allocation,
                    "serving_runtime.prompt_cache.kv_slot_allocation_bytes",
                )
            ),
            state_slot_allocation_bytes=(
                None
                if raw_state_slot_allocation is None
                else _metadata_nonnegative_int(
                    raw_state_slot_allocation,
                    "serving_runtime.prompt_cache.state_slot_allocation_bytes",
                )
            ),
            graph_residency_mode=graph_mode,
            retain_completed=retain_completed,
            save_implementation=save_implementation,
            apply_tensor_get_submission_service=submission_enabled,
            apply_tensor_get_controller_phases=controllers_enabled,
            driver_cpu_copy_issue_contract=driver_copy_contract,
            unified_kv=unified_kv,
            host_state_layout=host_state_layout,
            host_pointer_bytes=host_pointer_bytes,
            recurrent_state_layout=recurrent_state_layout,
            observed_entry_sizes_bytes=observed_entry_sizes,
        )
    else:
        prompt_policy = None
    return _ServingResourcePolicy(
        enabled=_metadata_bool(enabled, "serving_runtime.contention_enabled"),
        vram_reserve_bytes=_metadata_nonnegative_int(
            values.get("vram_reserve_bytes", 0),
            "serving_runtime.vram_reserve_bytes",
        ),
        page_transfer_bandwidth_gb_s=(
            None
            if raw_bandwidth is None
            else _metadata_positive_float(
                raw_bandwidth,
                "serving_runtime.page_transfer_bandwidth_gb_s",
            )
        ),
        page_fault_latency_ns=(
            None
            if raw_fault_latency is None
            else _metadata_nonnegative_float(
                raw_fault_latency,
                "serving_runtime.page_fault_latency_ns",
            )
        ),
        page_fault_latency_known=fault_latency_known,
        residency_granule_bytes=(
            None
            if raw_residency_granule is None
            else _metadata_nonnegative_int(
                raw_residency_granule,
                "serving_runtime.residency_granule_bytes",
            )
            or None
        ),
        fault_batch_bytes=(
            None
            if raw_fault_batch is None
            else _metadata_nonnegative_int(
                raw_fault_batch,
                "serving_runtime.fault_batch_bytes",
            )
            or None
        ),
        fault_latency_scope=fault_latency_scope,
        timing_completeness=timing_completeness,
        include_custom_lowerers=_metadata_bool(
            values.get("include_custom_lowerers", False),
            "serving_runtime.include_custom_lowerers",
        ),
        kv_slot_context_tokens=(
            None
            if raw_slot_context is None
            else _metadata_nonnegative_int(
                raw_slot_context,
                "serving_runtime.kv_slot_context_tokens",
            )
        ),
        prompt_cache=prompt_policy,
    )


def _declared_serving_kv_quantum(plan: ServingPlan) -> Tuple[int, int]:
    """Read optional full-slot KV facts emitted by a placement adapter.

    The default page quantum is schema-level; the explicit quantized-page
    capability includes supported Q4 scale metadata. Local runtimes may also
    declare device-local or full-slot allocation facts in their capacity
    ledger; consume those when sizing checkpoints and eager commitments.
    """

    placement_metadata = getattr(plan.scenario.placement, "metadata", {})
    capacity_ledger = (
        placement_metadata.get("capacity_ledger", {})
        if isinstance(placement_metadata, _ABCMapping)
        else {}
    )
    if not isinstance(capacity_ledger, _ABCMapping):
        return (0, 0)
    raw_per_token = capacity_ledger.get("kv_bytes_per_token", 0)
    raw_per_slot = capacity_ledger.get(
        "kv_bytes_per_slot",
        capacity_ledger.get("kv_per_slot_requested_bytes", 0),
    )
    try:
        per_token = max(0, int(raw_per_token))
        per_slot = max(0, int(raw_per_slot))
    except (TypeError, ValueError, OverflowError):
        return (0, 0)
    return (per_token, per_slot)


def _eager_slot_reservation(
    plan: ServingPlan, slot_count: int
) -> Mapping[str, int]:
    """Return logical eager commitment and its role-specific host backing.

    The commitment normally reserves ``slot_count`` full context windows.
    An explicit research capability may instead use one declared shared pool.
    It is deliberately separate from the pages that have
    actually been touched and allocated by the resident KV ledger.  State is
    given first use of the physical cache in this conservative admission
    projection because a linear-state spill path may be absent.
    """

    policy = _serving_resource_policy(plan.scenario)
    slots = max(0, int(slot_count))
    if not policy.enabled or plan.kv_policy.allocation_policy != "eager":
        return {
            "slot_count": slots,
            "kv_reservation_bytes": 0,
            "state_reservation_bytes": 0,
            "logical_reservation_bytes": 0,
            "effective_capacity_bytes": 0,
            "kv_host_backed_bytes": 0,
            "state_host_backed_bytes": 0,
            "host_backed_bytes": 0,
        }
    slot_context_tokens = policy.kv_slot_context_tokens
    if slot_context_tokens is None:
        slot_context_tokens = _model_max_sequence_length(plan.scenario)
    page_tokens = max(1, int(plan.kv_policy.tokens_per_page))
    kv_slot_pages = (
        (int(slot_context_tokens) + page_tokens - 1) // page_tokens
        if int(slot_context_tokens) > 0
        else 0
    )
    _, declared_kv_slot_bytes = _declared_serving_kv_quantum(plan)
    kv_per_slot_bytes = (
        declared_kv_slot_bytes
        if declared_kv_slot_bytes > 0
        else kv_slot_pages * max(0, int(plan.kv_policy.bytes_per_page))
    )
    kv_contract = _mapping_or_empty(_runtime_allocation_contract(plan).get("kv"))
    shared_kv_pool = (
        plan.scenario.workload.metadata.get("explicit_shared_kv_pool") is True
        and (
            kv_contract.get("shared_pool") is True
            or str(kv_contract.get("allocation_mode", "")).strip().lower()
            in {"shared", "shared_pool", "global_pool", "unified_pool"}
        )
    )
    if shared_kv_pool and slots > 0:
        kv_reservation_bytes = _contract_nonnegative_int(
            kv_contract, "committed_bytes", "requested_bytes"
        )
        if kv_reservation_bytes <= 0 and plan.kv_policy.bytes_per_page > 0:
            raise ValueError("shared KV pool requires positive committed_bytes")
    else:
        kv_reservation_bytes = slots * kv_per_slot_bytes
    state_reservation_bytes = slots * max(
        0, int(plan.linear_state_policy.bytes_per_request)
    )
    cache_component = plan.kv_policy.cache_component
    physical_capacity = _physical_runtime_limits(plan).get(
        str(cache_component),
        max(0, int(plan.kv_policy.capacity_bytes)),
    ) if cache_component else max(0, int(plan.kv_policy.capacity_bytes))
    effective_capacity = max(
        0, int(physical_capacity) - int(policy.vram_reserve_bytes)
    )
    state_resident_bytes = min(state_reservation_bytes, effective_capacity)
    state_host_backed_bytes = max(0, state_reservation_bytes - state_resident_bytes)
    kv_physical_capacity = max(0, effective_capacity - state_resident_bytes)
    kv_host_backed_bytes = max(0, kv_reservation_bytes - kv_physical_capacity)
    return {
        "slot_count": slots,
        "kv_reservation_bytes": kv_reservation_bytes,
        "state_reservation_bytes": state_reservation_bytes,
        "logical_reservation_bytes": (
            kv_reservation_bytes + state_reservation_bytes
        ),
        "effective_capacity_bytes": effective_capacity,
        "kv_host_backed_bytes": kv_host_backed_bytes,
        "state_host_backed_bytes": state_host_backed_bytes,
        "host_backed_bytes": kv_host_backed_bytes + state_host_backed_bytes,
    }


def _eager_slot_admission_reason(
    plan: ServingPlan, slot_count: int
) -> Optional[str]:
    """Return a fail-closed reason when eager commitments lack backing."""

    terms = _eager_slot_reservation(plan, slot_count)
    if not terms["logical_reservation_bytes"]:
        return None
    state_host_capacity = max(
        0, int(plan.linear_state_policy.offload_capacity_bytes)
    )
    kv_host_capacity = max(0, int(plan.kv_policy.offload_capacity_bytes))
    if terms["state_host_backed_bytes"] > state_host_capacity:
        return (
            "eager linear-state reservation requires {} host-backed bytes, "
            "but only {} bytes are available"
        ).format(
            terms["state_host_backed_bytes"], state_host_capacity
        )
    if terms["kv_host_backed_bytes"] > kv_host_capacity:
        return (
            "eager KV reservation requires {} host-backed bytes, but only {} "
            "bytes are available"
        ).format(terms["kv_host_backed_bytes"], kv_host_capacity)
    return None


def _max_live_kv_tokens(prompt_tokens: int, output_tokens: int) -> int:
    """Maximum committed KV entries for ordinary autoregressive serving."""

    return max(0, int(prompt_tokens)) + max(0, int(output_tokens) - 1)


def _model_max_sequence_length(scenario: ScenarioConfig) -> int:
    return max(0, int(_execution_view(scenario).max_sequence_length or 0))


def _dtype_bits(dtype: str) -> int:
    return dtype_bits(
        dtype,
        unsupported_message="unsupported KV dtype: {}".format(dtype),
    )


def _batch_item_kv_append_tokens(item: BatchItem) -> int:
    value = item.token_count if item.kv_append_tokens is None else item.kv_append_tokens
    return max(0, int(value))


def _batch_item_kv_materialized_tokens(item: BatchItem) -> int:
    value = (
        _batch_item_kv_append_tokens(item)
        if item.kv_materialized_tokens is None
        else item.kv_materialized_tokens
    )
    return max(0, int(value))


def _batch_item_verifier_tokens(item: BatchItem) -> int:
    if item.verifier_tokens is not None:
        return max(0, int(item.verifier_tokens))
    return max(
        0,
        int(item.proposed_tokens or item.token_count),
    )


def _batch_item_main_tokens(item: BatchItem) -> int:
    if item.main_tokens is not None:
        return max(0, int(item.main_tokens))
    return 1 if item.phase == "mtp" and _batch_item_verifier_tokens(item) else 0


def _batch_item_draft_tokens(item: BatchItem) -> int:
    if item.draft_tokens is not None:
        return max(0, int(item.draft_tokens))
    return max(
        0,
        _batch_item_verifier_tokens(item) - _batch_item_main_tokens(item),
    )


def _batch_item_committed_tokens(item: BatchItem) -> int:
    if item.committed_tokens is not None:
        return max(0, int(item.committed_tokens))
    proposed = _batch_item_verifier_tokens(item)
    return round_accepted_prefix(proposed, item.expected_accepted_tokens)


def _kv_bytes_for_layer(
    scenario: ScenarioConfig,
    layer: Any,
    override_dtype: Optional[str],
) -> Tuple[int, int]:
    """Return logical and aggregate TP-physical KV bytes for one token."""

    if layer.is_linear_attention:
        return (0, 0)
    if not layer.attention_head_dim and layer.hidden_size % layer.attention_heads:
        raise ValueError(
            "layer {} without explicit attention_head_dim must divide "
            "hidden_size by attention_heads for KV sizing".format(layer.layer_id)
        )
    policy_dtype = scenario.placement.kv_policy.dtype
    effective_dtype = override_dtype or policy_dtype or layer.dtype
    element_bits = _dtype_bits(str(effective_dtype))
    head_dim = layer.effective_attention_head_dim
    kv_heads = layer.effective_kv_heads
    logical_bits = 2 * kv_heads * head_dim * element_bits
    tp_degree = max(1, int(scenario.placement.parallel.tp_degree))
    local_heads = int(math.ceil(kv_heads / float(tp_degree)))
    physical_bits = 2 * local_heads * tp_degree * head_dim * element_bits
    if (scenario.workload.metadata.get("explicit_quantized_kv_pages") is True
            and element_bits == 4 and override_dtype in (None, policy_dtype)):
        _, artifact = _kv_dtype_bits(scenario, layer)
        if (artifact is not None and artifact.name == "Q4_0"
                and (local_heads * head_dim) % artifact.block_size == 0):
            return ((logical_bits + 7) // 8,
                    2 * tp_degree * _kv_tensor_bytes(scenario, layer, tp_degree, 1))
    return ((logical_bits + 7) // 8, (physical_bits + 7) // 8)


def _logical_kv_bytes_per_token(
    scenario: ScenarioConfig, override_dtype: Optional[str]
) -> int:
    total_bytes = 0
    for layer in _execution_layers(scenario):
        if layer.is_linear_attention:
            continue
        logical_bytes, _ = _kv_bytes_for_layer(scenario, layer, override_dtype)
        total_bytes += logical_bytes
    return total_bytes


def _kv_bytes_per_token(scenario: ScenarioConfig, override_dtype: Optional[str]) -> int:
    """Return aggregate physical KV bytes across tensor-parallel ranks.

    Attention KV heads normally shard across TP ranks.  When there are fewer
    KV heads than ranks (common for MQA/GQA), each rank still needs a local
    head and the KV is replicated.  Counting ``ceil(heads / TP) * TP`` keeps
    the scheduler capacity ledger from under-reporting that replication.
    """

    total_bytes = 0
    for layer in _execution_layers(scenario):
        if layer.is_linear_attention:
            continue
        _, physical_bytes = _kv_bytes_for_layer(scenario, layer, override_dtype)
        total_bytes += physical_bytes
    return total_bytes


def _linear_state_bytes_per_layer(layer: Any) -> int:
    if not getattr(layer, "is_linear_attention", False):
        return 0
    geometry = getattr(layer, "linear_attention", None)
    if geometry is None:
        raise ValueError("linear attention layer is missing geometry")
    elements = (
        geometry.recurrent_state_elements
        + geometry.convolution_state_elements
    )
    return int(math.ceil(elements * _dtype_bits(geometry.state_dtype) / 8.0))


def _linear_state_bytes_per_request(scenario: ScenarioConfig) -> int:
    return sum(
        _linear_state_bytes_per_layer(layer)
        for layer in _execution_layers(scenario)
    )


_EVENT_TYPE_ORDER = {
    "request_arrival": 0,
    "prefill_chunk_complete": 1,
    "kv_recompute_chunk_complete": 1,
    "tokens_committed": 1,
    "batch_end": 2,
    "request_finished": 3,
    "kv_swap_out": 4,
    "linear_state_swap_out": 4,
    "kv_recompute_required": 4,
    "request_preempted": 5,
    "kv_swap_out_start": 6,
    "kv_swap_in_start": 6,
    "linear_state_swap_out_start": 6,
    "linear_state_swap_in_start": 6,
    "kv_swap_in": 6,
    "linear_state_swap_in": 6,
    "request_admitted": 6,
    "request_resumed": 6,
    "batch_start": 7,
    "request_rejected": 8,
}


def _event_sort_key(event: ServingEvent) -> Tuple[float, int, str, str, str]:
    return (
        event.timestamp_ns,
        _EVENT_TYPE_ORDER.get(event.event_type, 9),
        event.request_id or "",
        event.cohort_id or "",
        event.event_type,
    )


def _component_capacity(scenario: ScenarioConfig, component_id: Optional[str]) -> int:
    if not component_id:
        return 0
    try:
        return int(scenario.hardware.get_component(component_id).capacity_bytes)
    except KeyError:
        raise ValueError("unknown KV component: {}".format(component_id))


def _strict_unknown_storage_capacity(
    scenario: ScenarioConfig, component_id: Optional[str]
) -> bool:
    """Whether a V4 storage component omits required physical capacity."""

    if not component_id:
        return False
    component = scenario.hardware.get_component(str(component_id))
    return component.capacity_bytes <= 0 and component.memory_class in {
        "active",
        "offload",
    }


def _declared_runtime_tensor_capacity(
    scenario: ScenarioConfig, tensor_id: str
) -> int:
    """Return a user-declared runtime-state cap, excluding mapper estimates."""

    if is_control_plane_generated_tensor(scenario, tensor_id):
        return 0
    return max(0, int(scenario.placement.tensor_bytes.get(tensor_id, 0)))


def _normalize_physical_role_capacities(
    scenario: ScenarioConfig,
    roles: Sequence[Tuple[str, Optional[str], int, bool, int, str]],
) -> Mapping[str, int]:
    """Partition each physical component budget across all runtime roles."""

    normalized = {name: max(0, int(capacity)) for name, _, capacity, _, _, _ in roles}
    by_component: Dict[str, List[Tuple[str, int, bool, int, str]]] = {}
    for name, component_id, capacity, explicit, quantum, tensor_name in roles:
        if _strict_unknown_storage_capacity(scenario, component_id):
            normalized[name] = 0
            continue
        if not component_id or capacity <= 0:
            continue
        by_component.setdefault(str(component_id), []).append(
            (name, int(capacity), bool(explicit), max(1, int(quantum)), tensor_name)
        )
    for component_id, component_roles in by_component.items():
        if _component_capacity(scenario, component_id) <= 0:
            continue
        budget = _dynamic_component_capacity(
            scenario,
            component_id,
            tuple(role[4] for role in component_roles),
        )
        explicit_total = sum(
            capacity for _, capacity, explicit, _, _ in component_roles if explicit
        )
        if explicit_total > budget:
            raise ValueError(
                "explicit runtime capacities exceed shared physical capacity on {}".format(
                    component_id
                )
            )
        available = budget - explicit_total
        for name, capacity, explicit, quantum, _ in component_roles:
            if explicit:
                normalized[name] = capacity
                continue
            allocated = min(capacity, available)
            allocated -= allocated % quantum
            normalized[name] = allocated
            available -= allocated
    return normalized


def _physical_runtime_limits(plan: ServingPlan) -> Mapping[str, int]:
    # Validate the authoritative fixed owners even when a referenced storage
    # component has unknown capacity and no dynamic budget can be derived.
    _physical_capacity_claims(plan.scenario)
    roles = (
        (
            plan.kv_policy.cache_component,
            "kv_cache",
            plan.kv_policy.capacity_bytes,
        ),
        (
            plan.kv_policy.offload_component,
            "kv_cache_offload",
            plan.kv_policy.offload_capacity_bytes,
        ),
        (
            plan.linear_state_policy.cache_component,
            "linear_state",
            plan.linear_state_policy.capacity_bytes,
        ),
        (
            plan.linear_state_policy.offload_component,
            "linear_state_offload",
            plan.linear_state_policy.offload_capacity_bytes,
        ),
    )
    names_by_component: Dict[str, set] = {}
    fallback_by_component: Dict[str, int] = {}
    for component_id, tensor_name, fallback in roles:
        if not component_id:
            continue
        component = str(component_id)
        names_by_component.setdefault(component, set()).add(tensor_name)
        fallback_by_component[component] = (
            fallback_by_component.get(component, 0) + int(fallback)
        )
    limits: Dict[str, int] = {}
    for component_id, dynamic_tensors in names_by_component.items():
        physical = _component_capacity(plan.scenario, component_id)
        if _strict_unknown_storage_capacity(plan.scenario, component_id):
            limits[component_id] = 0
        elif physical > 0:
            limits[component_id] = _dynamic_component_capacity(
                plan.scenario, component_id, tuple(dynamic_tensors)
            )
        else:
            limits[component_id] = fallback_by_component[component_id]
    return limits


def _add_prompt_cache_runtime_limits(
    plan: ServingPlan,
    policy: Optional[_PromptCachePolicy],
    limits: Dict[str, int],
) -> None:
    """Add capacity for a prompt-cache-only component when one is declared."""

    if policy is None or not policy.enabled:
        return
    roles = (
        (policy.component or plan.kv_policy.cache_component, "prompt_cache"),
        (
            policy.offload_component or plan.kv_policy.offload_component,
            "prompt_cache_offload",
        ),
    )
    for component_id, tensor_name in roles:
        if not component_id:
            continue
        component = str(component_id)
        if component in limits:
            continue
        if _strict_unknown_storage_capacity(plan.scenario, component):
            limits[component] = 0
            continue
        physical = _component_capacity(plan.scenario, component)
        if physical > 0:
            limits[component] = _dynamic_component_capacity(
                plan.scenario, component, (tensor_name,)
            )
        else:
            # Unknown capacity is intentionally left unbounded here.  The
            # physical ledger's existing fallback semantics apply, while all
            # derived prompt-cache sizes remain explicit in metadata.
            continue


def _dynamic_component_capacity(
    scenario: ScenarioConfig,
    component_id: Optional[str],
    dynamic_tensors: Sequence[str],
) -> int:
    """Physical bytes available to colocated runtime-managed state."""

    claims = _physical_capacity_claims(scenario)
    capacity = _component_capacity(scenario, component_id)
    if capacity <= 0:
        return 0
    dynamic = set(str(tensor_id) for tensor_id in dynamic_tensors)
    component_claims = tuple(
        claim for claim in claims if claim.component_id == component_id
    )
    claimed_ids = set(claim.claim_id for claim in component_claims)
    claimed_bytes = sum(claim.byte_count for claim in component_claims)
    decision = control_plane_decision(scenario)
    logical_views_raw = decision.get("logical_weight_views", {})
    if isinstance(logical_views_raw, _ABCMapping):
        logical_views = set(str(tensor_id) for tensor_id in logical_views_raw)
    elif isinstance(logical_views_raw, _ABCSequence) and not isinstance(
        logical_views_raw, (str, bytes)
    ):
        logical_views = set(str(tensor_id) for tensor_id in logical_views_raw)
    else:
        logical_views = set()
    runtime_weight_copies = _runtime_weight_copy_ids(decision)
    logical_views.difference_update(runtime_weight_copies)
    aliases = {
        str(tensor_id)
        for tensor_id in _mapping_or_empty(
            decision.get("logical_weight_aliases")
        )
        if str(tensor_id) not in runtime_weight_copies
    }
    special = _mapping_or_empty(
        _mapping_or_empty(scenario.placement.metadata).get(
            "special_weight_placement"
        )
    )
    non_owning_special = {
        str(tensor_id)
        for tensor_id, raw_contract in special.items()
        if str(
            _mapping_or_empty(raw_contract).get("capacity_accounting_role", "")
        )
        in {"logical_view", "non_owning_alias"}
        and str(tensor_id) not in runtime_weight_copies
    }
    occupied = sum(
        int(size)
        for tensor, size in scenario.placement.tensor_bytes.items()
        if tensor not in dynamic
        and tensor not in claimed_ids
        and tensor not in logical_views
        and tensor not in aliases
        and tensor not in non_owning_special
        and scenario.placement.tensor_to_component.get(tensor) == component_id
    )
    return max(0, capacity - claimed_bytes - occupied)


def _kv_capacity_bytes(
    scenario: ScenarioConfig, component_id: Optional[str]
) -> int:
    if _kv_bytes_per_token(scenario, None) <= 0:
        return 0
    if not component_id:
        raise ValueError("V4 full-attention workloads require a KV cache component")
    if _strict_unknown_storage_capacity(scenario, component_id):
        return 0
    component_capacity = _component_capacity(scenario, component_id)
    declared = _declared_runtime_tensor_capacity(scenario, "kv_cache")
    if declared > 0:
        capacity = min(declared, component_capacity) if component_capacity else declared
    elif component_capacity > 0:
        capacity = _dynamic_component_capacity(
            scenario, component_id, ("kv_cache",)
        )
    else:
        capacity = 0
    if capacity < 0:
        raise ValueError("KV capacity_bytes must be non-negative")
    return capacity


__all__ = [
    "BatchCohort",
    "BatchCost",
    "BatchCostProvider",
    "BatchItem",
    "KVCacheMetrics",
    "KVCachePolicy",
    "LinearStateMetrics",
    "LinearStatePolicy",
    "MTPPolicy",
    "PromptCacheMetrics",
    "RequestStatus",
    "SchedulerMetrics",
    "ServingBatch",
    "ServingEvent",
    "ServingPlan",
    "ServingPolicy",
    "ServingRequest",
    "ServingRequestMetrics",
    "ServingRequestState",
    "ServingResult",
    "compile_serving_plan",
    "serving_admission_diagnostics",
    "simulate_online",
]
