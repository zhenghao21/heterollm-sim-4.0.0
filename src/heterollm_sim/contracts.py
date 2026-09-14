"""Stable contracts shared by planners, cost models, and the event engine.

The analytical models describe *resource demand*.  Only the discrete-event
engine turns those demands into elapsed time under contention.  Keeping this
boundary explicit prevents memory or link latency from being counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Mapping, Optional, Tuple

from .schema_v4 import SIMULATION_SCHEMA_VERSION


class TaskCategory(str, Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    CIM = "cim"
    COLLECTIVE = "collective"
    POLICY = "policy"
    SYNCHRONIZATION = "synchronization"
    OUTPUT = "output"


class OperatorClass(str, Enum):
    """Stable cost-model primitive classes.

    These classes describe *how* an operator consumes hardware rather than a
    model-layer name.  A compound Transformer operator may lower to more than
    one primitive, while compute and memory demand inside one fused primitive
    remain concurrent.
    """

    GEMM = "gemm"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    MEMORY = "memory"
    COMMUNICATION = "communication"


class TraceMarker(str, Enum):
    REQUEST_ARRIVAL = "request_arrival"
    FIRST_TOKEN = "first_token"
    TOKEN_EMIT = "token_emit"
    REQUEST_DONE = "request_done"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    CALIBRATED = "calibrated"
    ANALYTICAL = "analytical"
    BLACK_BOX = "black_box"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


class TraceFidelity(str, Enum):
    """Semantic fidelity of a visualization trace payload.

    ``EXACT`` rows are realized DES intervals.  ``REPRESENTATIVE`` rows are
    selected realized samples, while ``AGGREGATE`` rows describe envelopes or
    counters that cannot be expanded back into an operator-level timeline.
    """

    EXACT = "exact"
    REPRESENTATIVE = "representative"
    AGGREGATE = "aggregate"


ANALYTICAL_MODEL_VERSION = "task-transaction-analytical-v4"


class RetentionPolicy(str, Enum):
    """V4 event-retention policies over the single unified kernel."""

    EXACT = "exact"
    STREAMING = "streaming"
    AGGREGATE = "aggregate"


VISUALIZATION_SCHEMA_VERSION = "1.0"
DEFAULT_VISUALIZATION_EVENT_LIMIT = 2_000
MAX_VISUALIZATION_EVENT_LIMIT = 5_000
DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT = 4_096
MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT = 4_096


# Public component time-series contract.  Keep the metric spellings stable:
# the web UI and exported JSON reports use them as machine-readable keys.
COMPONENT_TIMESERIES_SCHEMA_VERSION = "1.0"
DEFAULT_COMPONENT_TIMESERIES_POINT_LIMIT = 500
MAX_COMPONENT_TIMESERIES_POINT_LIMIT = 2_000
COMPONENT_TIMESERIES_METRICS = frozenset(
    {
        "busy_fraction",
        "modeled_compute_utilization",
        "weight_residency_bytes",
        "kv_cache_residency_bytes",
        "linear_state_residency_bytes",
        "activation_residency_bytes",
        "temporary_residency_bytes",
        "memory_read_bandwidth_utilization",
        "memory_write_bandwidth_utilization",
        "storage_occupancy_bytes",
        "storage_io_utilization",
        "storage_read_bandwidth_utilization",
        "storage_write_bandwidth_utilization",
        "dma_engine_utilization",
        "fabric_bandwidth_utilization",
        "link_bandwidth_utilization",
    }
)


class SeriesQuality(str, Enum):
    """Evidence quality of one component time-series curve."""

    MODELED = "modeled"
    ESTIMATED = "estimated"
    DECLARED = "declared"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChangePointInterval:
    """A finite, half-open interval in a bounded component series."""

    start_ns: float
    end_ns: float
    value: float

    def __post_init__(self) -> None:
        values = (self.start_ns, self.end_ns, self.value)
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError("时序区间必须使用有限数值")
        if self.start_ns < 0 or self.end_ns < self.start_ns:
            raise ValueError("时序区间必须满足 0 <= start_ns <= end_ns")


@dataclass(frozen=True)
class VisualizationTraceOptions:
    """Bounded paging options for the public visualization contract."""

    event_offset: int = 0
    event_limit: int = DEFAULT_VISUALIZATION_EVENT_LIMIT
    memory_segment_limit: int = DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT

    def __post_init__(self) -> None:
        if (
            isinstance(self.event_offset, bool)
            or not isinstance(self.event_offset, int)
            or self.event_offset < 0
        ):
            raise ValueError("event_offset must be a non-negative integer")
        if (
            isinstance(self.event_limit, bool)
            or not isinstance(self.event_limit, int)
            or self.event_limit < 1
            or self.event_limit > MAX_VISUALIZATION_EVENT_LIMIT
        ):
            raise ValueError(
                "event_limit must be between 1 and {}".format(
                    MAX_VISUALIZATION_EVENT_LIMIT
                )
            )
        if (
            isinstance(self.memory_segment_limit, bool)
            or not isinstance(self.memory_segment_limit, int)
            or self.memory_segment_limit < 1
            or self.memory_segment_limit > MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT
        ):
            raise ValueError(
                "memory_segment_limit must be between 1 and {}".format(
                    MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT
                )
            )


@dataclass(frozen=True)
class ResourceDemand:
    """Exclusive service demand on one simulator resource.

    Full-duplex links and independent channels are represented as separate
    resources.  A task may contain multiple demands; they start together once
    every required resource is available and may have different release times.
    """

    resource_id: str
    service_ns: float
    bytes_moved: int = 0
    energy_pj: float = 0.0
    work_units: float = 0.0

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must not be empty")
        numeric_values = (
            self.service_ns,
            self.bytes_moved,
            self.energy_pj,
            self.work_units,
        )
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError("demand quantities must be finite numbers")
        if self.service_ns < 0:
            raise ValueError("service_ns must be non-negative")
        if self.bytes_moved < 0 or self.energy_pj < 0 or self.work_units < 0:
            raise ValueError("demand quantities must be non-negative")


@dataclass(frozen=True)
class _PreparedExecutionTask:
    """Planner-authored task fact reused by online stage scheduling.

    This is an internal handoff object, not a serialized public contract.
    Planner construction already validates ordering and resource demands; a
    normal/custom metadata mapping still goes through serving's fail-closed
    parser instead of constructing this type.
    """

    task_id: str
    dependencies: Tuple[str, ...]
    request_ids: Tuple[str, ...]
    demands: Tuple[ResourceDemand, ...]
    opaque_device_fence: bool = False
    category: TaskCategory = TaskCategory.COMPUTE
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedExecutionStage:
    """Planner-authored execution stage for an in-process trusted handoff."""

    stage_id: str
    stage_index: int
    dependencies: Tuple[str, ...]
    request_ids: Tuple[str, ...]
    component_id: str
    service_ns: float
    execution_tasks: Tuple[_PreparedExecutionTask, ...]
    invocation_group_id: Optional[str] = None
    covered_invocation_group_ids: Tuple[str, ...] = ()

    @property
    def causal_invocation_group_ids(self) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *((self.invocation_group_id,) if self.invocation_group_id else ()),
                    *self.covered_invocation_group_ids,
                )
            )
        )

    @property
    def requires_device_fence(self) -> bool:
        return any(
            task.opaque_device_fence for task in self.execution_tasks
        )


@dataclass(frozen=True)
class TaskSpec:
    """An atomic task in ScheduleIR."""

    task_id: str
    request_id: str
    name: str
    category: TaskCategory
    dependencies: Tuple[str, ...] = ()
    demands: Tuple[ResourceDemand, ...] = ()
    earliest_start_ns: float = 0.0
    marker: Optional[TraceMarker] = None
    token_index: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.request_id or not self.name:
            raise ValueError("task_id, request_id, and name must not be empty")
        if not math.isfinite(self.earliest_start_ns):
            raise ValueError("earliest_start_ns must be finite")
        if self.earliest_start_ns < 0:
            raise ValueError("earliest_start_ns must be non-negative")
        resource_ids = [d.resource_id for d in self.demands]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("a task may demand each resource at most once")


@dataclass(frozen=True)
class ResourceInterval:
    resource_id: str
    start_ns: float
    end_ns: float
    bytes_moved: int = 0
    energy_pj: float = 0.0


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    request_id: str
    name: str
    category: TaskCategory
    start_ns: float
    end_ns: float
    dependency_ready_ns: float
    marker: Optional[TraceMarker] = None
    token_index: Optional[int] = None
    resource_intervals: Tuple[ResourceInterval, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> float:
        return self.end_ns - self.start_ns

    @property
    def wait_ns(self) -> float:
        return self.start_ns - self.dependency_ready_ns


@dataclass(frozen=True, kw_only=True)
class RunManifest:
    schema_version: str
    run_id: str
    random_seed: int
    simulator_version: str
    model_name: str
    hardware_name: str
    workload_name: str
    # Compatibility field retained in the V4 report schema.  Its value names
    # the analytical lowering model, not a silicon-calibration claim.
    calibration_version: str = ANALYTICAL_MODEL_VERSION
    evidence: EvidenceStatus = EvidenceStatus.ANALYTICAL
    assumptions: Tuple[str, ...] = ()
    assumptions_zh: Tuple[str, ...] = ()
    assumptions_en: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SIMULATION_SCHEMA_VERSION:
            raise ValueError(
                "manifest schema_version must be exactly {}; got {}".format(
                    SIMULATION_SCHEMA_VERSION,
                    self.schema_version,
                )
            )


@dataclass
class SimulationTrace:
    manifest: RunManifest
    tasks: Tuple[TaskResult, ...]
    resource_busy_ns: Dict[str, float]
    makespan_ns: float
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    def by_task_id(self) -> Dict[str, TaskResult]:
        return {task.task_id: task for task in self.tasks}
