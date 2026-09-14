"""Closed, aggregate runtime IR for the version-4 control-plane simulator.

The runtime IR deliberately stops at batches and controller transactions.  It
must never be expanded into one task per instruction, cache line, page, or
packet: those populations are represented by counts on the aggregate objects
below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Mapping, Optional, Tuple, TypeAlias, Union

from .contracts import ResourceDemand, TaskCategory, TaskSpec


RUNTIME_ACTION_METADATA_KEY = "runtime.action"
RUNTIME_PAYLOAD_METADATA_KEY = "runtime.payload"
RUNTIME_PHASE_METADATA_KEY = "runtime.phase"
RUNTIME_SEQUENCE_METADATA_KEY = "runtime.sequence"
RUNTIME_INSTRUCTION_BATCH_METADATA_KEY = "runtime.instruction_batch"
RUNTIME_CONTROLLER_BATCH_METADATA_KEY = "runtime.controller_batch"


class RuntimePhase(IntEnum):
    """Stable phase order used before the submission sequence tie-break."""

    CAPACITY_CHECK = 0
    PLACEMENT_DECISION = 10
    ALLOCATION = 20
    CACHE = 30
    STORAGE = 40
    ADDRESS_TRANSLATION = 50
    TRANSFER = 60
    SCHEDULING = 70
    COMMAND = 80
    GPU_FRONTEND = 90
    MEMORY = 100
    COMPUTE = 110
    INTERRUPT = 120
    CPU_COMPLETION = 130

    PLACEMENT = PLACEMENT_DECISION
    ALLOCATOR = ALLOCATION
    NVME = STORAGE
    IOMMU_DMA_PCIE = TRANSFER
    GPU_COMMAND_PROCESSOR = GPU_FRONTEND
    MMU_TLB_L2_VRAM = MEMORY


class InstructionClass(StrEnum):
    """Coarse instruction population carried by one aggregate batch."""

    CONTROL = "control"
    MEMORY = "memory"
    DMA = "dma"
    GPU_COMMAND = "gpu_command"
    COMPUTE = "compute"
    INTERRUPT = "interrupt"


class ControllerKind(StrEnum):
    """Closed controller population represented by transaction batches."""

    CPU = "cpu"
    ALLOCATOR = "allocator"
    WEIGHT_CACHE = "weight_cache"
    PAGE_CACHE = "page_cache"
    NVME = "nvme"
    IOMMU = "iommu"
    DMA = "dma"
    PCIE = "pcie"
    SCHEDULER = "scheduler"
    COMMAND_BUILDER = "command_builder"
    GPU_COMMAND_PROCESSOR = "gpu_command_processor"
    MMU_TLB = "mmu_tlb"
    L2 = "l2"
    VRAM = "vram"
    INTERRUPT = "interrupt"


class RuntimeAction(StrEnum):
    """Closed set of completion actions understood by ``RuntimeDispatcher``."""

    NOOP = "noop"
    CAPACITY_CHECK = "capacity_check"
    PLACEMENT_DECISION = "placement_decision"
    ALLOCATE = "allocate"
    WEIGHT_CACHE_LOOKUP = "weight_cache_lookup"
    PAGE_CACHE_LOOKUP = "page_cache_lookup"
    NVME_READ = "nvme_read"
    IOMMU_TRANSLATE = "iommu_translate"
    DMA_MAP = "dma_map"
    PCIE_TRANSFER = "pcie_transfer"
    BATCH_SCHEDULE = "batch_schedule"
    OPERATOR_SCHEDULE = "operator_schedule"
    COMMAND_BUILD = "command_build"
    COMMAND_SUBMIT = "command_submit"
    GPU_COMMAND_PROCESS = "gpu_command_process"
    MMU_TLB_LOOKUP = "mmu_tlb_lookup"
    L2_LOOKUP = "l2_lookup"
    VRAM_CONTROLLER = "vram_controller"
    COMPUTE = "compute"
    INTERRUPT = "interrupt"
    CPU_COMPLETE = "cpu_complete"

    ALLOCATION = ALLOCATE
    GPU_COMMAND_PROCESSOR = GPU_COMMAND_PROCESS
    MMU_TLB = MMU_TLB_LOOKUP
    CPU_COMPLETION = CPU_COMPLETE


def decode_runtime_metadata(
    task: TaskSpec,
) -> Tuple[RuntimeAction, Mapping[str, object], int, int]:
    """Validate and decode the reserved runtime fields on a ``TaskSpec``."""

    raw_action = task.metadata.get(RUNTIME_ACTION_METADATA_KEY, RuntimeAction.NOOP)
    try:
        action = (
            raw_action
            if isinstance(raw_action, RuntimeAction)
            else RuntimeAction(str(raw_action))
        )
    except ValueError as exc:
        raise ValueError("task {} has unknown runtime action".format(task.task_id)) from exc
    payload = task.metadata.get(RUNTIME_PAYLOAD_METADATA_KEY, {})
    if not isinstance(payload, Mapping):
        raise TypeError("task {} runtime payload must be a mapping".format(task.task_id))
    _require_data_only(payload)
    phase = task.metadata.get(RUNTIME_PHASE_METADATA_KEY, 0)
    sequence = task.metadata.get(RUNTIME_SEQUENCE_METADATA_KEY, 0)
    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 0:
        raise ValueError("task {} runtime phase must be a non-negative integer".format(task.task_id))
    _require_non_negative_int(sequence, "runtime sequence")
    return action, dict(payload), phase, sequence


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must not be empty".format(field_name))


def _require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(field_name))


def _require_data_only(value: object, path: str = "payload") -> None:
    """Reject executable or mutable-control objects at the IR boundary."""

    if callable(value):
        raise TypeError("{} must contain data, not callable objects".format(path))
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{} must contain finite numbers".format(path))
        return
    if isinstance(value, (RuntimeAction, RuntimePhase, InstructionClass, ControllerKind)):
        return
    if isinstance(value, TaskSpec):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("{} mapping keys must be non-empty strings".format(path))
            _require_data_only(item, "{}.{}".format(path, key))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_data_only(item, "{}[{}]".format(path, index))
        return
    raise TypeError(
        "{} contains unsupported runtime data type {}".format(
            path,
            type(value).__name__,
        )
    )


@dataclass(frozen=True)
class InstructionBatch:
    """A population of like instructions, never individual instructions."""

    batch_id: str
    instruction_class: InstructionClass
    instruction_count: int
    operation_count: int = 0
    byte_count: int = 0

    def __post_init__(self) -> None:
        _require_non_empty(self.batch_id, "batch_id")
        if not isinstance(self.instruction_class, InstructionClass):
            raise TypeError("instruction_class must be an InstructionClass")
        _require_non_negative_int(self.instruction_count, "instruction_count")
        if self.instruction_count == 0:
            raise ValueError("instruction_count must be positive")
        _require_non_negative_int(self.operation_count, "operation_count")
        _require_non_negative_int(self.byte_count, "byte_count")

    def to_metadata(self) -> Mapping[str, object]:
        return {
            "batch_id": self.batch_id,
            "instruction_class": self.instruction_class.value,
            "instruction_count": self.instruction_count,
            "operation_count": self.operation_count,
            "byte_count": self.byte_count,
        }

    @property
    def count(self) -> int:
        return self.instruction_count


@dataclass(frozen=True)
class ControllerTransactionBatch:
    """An aggregate transaction population for one hardware controller."""

    batch_id: str
    controller: ControllerKind
    transaction_count: int
    byte_count: int = 0

    def __post_init__(self) -> None:
        _require_non_empty(self.batch_id, "batch_id")
        if not isinstance(self.controller, ControllerKind):
            raise TypeError("controller must be a ControllerKind")
        _require_non_negative_int(self.transaction_count, "transaction_count")
        if self.transaction_count == 0:
            raise ValueError("transaction_count must be positive")
        _require_non_negative_int(self.byte_count, "byte_count")

    def to_metadata(self) -> Mapping[str, object]:
        return {
            "batch_id": self.batch_id,
            "controller": self.controller.value,
            "transaction_count": self.transaction_count,
            "byte_count": self.byte_count,
        }

    @property
    def controller_kind(self) -> ControllerKind:
        return self.controller

    @property
    def count(self) -> int:
        return self.transaction_count


def decode_aggregate_metadata(
    task: TaskSpec,
) -> Tuple[Optional[InstructionBatch], Optional[ControllerTransactionBatch]]:
    """Decode typed aggregate batches carried through the legacy ``TaskSpec``."""

    instruction_batch: Optional[InstructionBatch] = None
    raw_instruction = task.metadata.get(RUNTIME_INSTRUCTION_BATCH_METADATA_KEY)
    if raw_instruction is not None:
        if not isinstance(raw_instruction, Mapping):
            raise TypeError("runtime instruction batch metadata must be a mapping")
        instruction_batch = InstructionBatch(
            batch_id=str(raw_instruction.get("batch_id", "")),
            instruction_class=InstructionClass(
                str(raw_instruction.get("instruction_class", ""))
            ),
            instruction_count=raw_instruction.get("instruction_count", 0),
            operation_count=raw_instruction.get("operation_count", 0),
            byte_count=raw_instruction.get("byte_count", 0),
        )

    controller_batch: Optional[ControllerTransactionBatch] = None
    raw_controller = task.metadata.get(RUNTIME_CONTROLLER_BATCH_METADATA_KEY)
    if raw_controller is not None:
        if not isinstance(raw_controller, Mapping):
            raise TypeError("runtime controller batch metadata must be a mapping")
        controller_batch = ControllerTransactionBatch(
            batch_id=str(raw_controller.get("batch_id", "")),
            controller=ControllerKind(str(raw_controller.get("controller", ""))),
            transaction_count=raw_controller.get("transaction_count", 0),
            byte_count=raw_controller.get("byte_count", 0),
        )
    return instruction_batch, controller_batch


@dataclass(frozen=True)
class HardwareTask:
    """One aggregate control-plane or hardware-service task."""

    task_id: str
    request_id: str
    name: str
    action: RuntimeAction
    phase: int
    sequence: int
    category: TaskCategory = TaskCategory.POLICY
    dependencies: Tuple[str, ...] = ()
    demands: Tuple[ResourceDemand, ...] = ()
    earliest_start_ns: float = 0.0
    payload: Mapping[str, object] = field(default_factory=dict)
    instruction_batch: Optional[InstructionBatch] = None
    controller_batch: Optional[ControllerTransactionBatch] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.request_id, "request_id")
        _require_non_empty(self.name, "name")
        if not isinstance(self.action, RuntimeAction):
            raise TypeError("action must be a RuntimeAction")
        if isinstance(self.phase, bool) or not isinstance(self.phase, int) or self.phase < 0:
            raise ValueError("phase must be a non-negative integer")
        _require_non_negative_int(self.sequence, "sequence")
        if not isinstance(self.category, TaskCategory):
            raise TypeError("category must be a TaskCategory")
        if not math.isfinite(float(self.earliest_start_ns)) or self.earliest_start_ns < 0:
            raise ValueError("earliest_start_ns must be a finite non-negative number")
        _require_data_only(self.payload)
        _require_data_only(self.metadata, "metadata")
        reserved = {
            RUNTIME_ACTION_METADATA_KEY,
            RUNTIME_PAYLOAD_METADATA_KEY,
            RUNTIME_PHASE_METADATA_KEY,
            RUNTIME_SEQUENCE_METADATA_KEY,
            RUNTIME_INSTRUCTION_BATCH_METADATA_KEY,
            RUNTIME_CONTROLLER_BATCH_METADATA_KEY,
        }
        overlap = reserved.intersection(self.metadata)
        if overlap:
            raise ValueError(
                "metadata may not override runtime keys: {}".format(sorted(overlap))
            )

    def to_task_spec(self) -> TaskSpec:
        """Lower to the existing scheduler contract without losing runtime type."""

        metadata = dict(self.metadata)
        metadata.update(
            {
                RUNTIME_ACTION_METADATA_KEY: self.action.value,
                RUNTIME_PAYLOAD_METADATA_KEY: dict(self.payload),
                RUNTIME_PHASE_METADATA_KEY: int(self.phase),
                RUNTIME_SEQUENCE_METADATA_KEY: self.sequence,
            }
        )
        if self.instruction_batch is not None:
            metadata[RUNTIME_INSTRUCTION_BATCH_METADATA_KEY] = dict(
                self.instruction_batch.to_metadata()
            )
        if self.controller_batch is not None:
            metadata[RUNTIME_CONTROLLER_BATCH_METADATA_KEY] = dict(
                self.controller_batch.to_metadata()
            )
        return TaskSpec(
            task_id=self.task_id,
            request_id=self.request_id,
            name=self.name,
            category=self.category,
            dependencies=tuple(self.dependencies),
            demands=tuple(self.demands),
            earliest_start_ns=float(self.earliest_start_ns),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RuntimeDelta:
    """Closed state mutation produced by one completion dispatch."""

    capacity_reservations: Mapping[str, int] = field(default_factory=dict)
    placement_decisions: Mapping[str, str] = field(default_factory=dict)
    allocations: Mapping[str, int] = field(default_factory=dict)
    weight_cache_additions: Tuple[str, ...] = ()
    page_cache_additions: Tuple[str, ...] = ()
    completed_requests: Tuple[str, ...] = ()
    metric_increments: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, values in (
            ("capacity_reservations", self.capacity_reservations),
            ("allocations", self.allocations),
        ):
            for key, value in values.items():
                _require_non_empty(key, "{} key".format(field_name))
                _require_non_negative_int(value, "{} value".format(field_name))
        for key, value in self.placement_decisions.items():
            _require_non_empty(key, "placement decision key")
            _require_non_empty(value, "placement decision value")
        for field_name, additions in (
            ("weight_cache_additions", self.weight_cache_additions),
            ("page_cache_additions", self.page_cache_additions),
            ("completed_requests", self.completed_requests),
        ):
            for value in additions:
                _require_non_empty(value, "{} value".format(field_name))
        for key, value in self.metric_increments.items():
            _require_non_empty(key, "metric key")
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("metric increments must be finite numbers")


RuntimeTask: TypeAlias = Union[HardwareTask, TaskSpec]


@dataclass(frozen=True)
class ExpansionResult:
    """Pure-data result of dispatching a closed runtime action."""

    tasks: Tuple[RuntimeTask, ...] = ()
    delta: RuntimeDelta = field(default_factory=RuntimeDelta)
    retain_dependencies: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(task, (HardwareTask, TaskSpec)) for task in self.tasks):
            raise TypeError("expansion tasks must be HardwareTask or TaskSpec instances")
        if not isinstance(self.delta, RuntimeDelta):
            raise TypeError("delta must be a RuntimeDelta")

    @property
    def new_tasks(self) -> Tuple[RuntimeTask, ...]:
        return self.tasks

    @property
    def state_delta(self) -> RuntimeDelta:
        return self.delta


@dataclass(frozen=True)
class KernelCompletion:
    """Typed runtime view of a completed kernel event."""

    task: TaskSpec
    action: RuntimeAction
    payload: Mapping[str, object]
    phase: int
    sequence: int
    start_ns: float
    end_ns: float
    dependency_ready_ns: float
    effective_ready_ns: float
    demands: Tuple[ResourceDemand, ...]
    resource_predecessors: Mapping[str, Mapping[str, object]]
    queue_wait_ns: float
    service_ns: float
    resource_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    instruction_batch: Optional[InstructionBatch] = None
    controller_batch: Optional[ControllerTransactionBatch] = None

    @property
    def duration_ns(self) -> float:
        return self.end_ns - self.start_ns


__all__ = [
    "ControllerKind",
    "ControllerTransactionBatch",
    "ExpansionResult",
    "HardwareTask",
    "InstructionBatch",
    "InstructionClass",
    "KernelCompletion",
    "RuntimeAction",
    "RuntimeDelta",
    "RuntimePhase",
    "RuntimeTask",
    "decode_aggregate_metadata",
    "decode_runtime_metadata",
]
