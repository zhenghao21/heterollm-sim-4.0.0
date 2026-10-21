"""Progress and cooperative-cancellation contracts for V4 execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from queue import SimpleQueue
from time import perf_counter
from typing import Any, Callable, Mapping, Optional


PROGRESS_UNIT_LABELS_ZH = {
    "serving_batches": "在线批次",
    "schedule_tasks": "拓扑任务",
    "workflow_steps": "流程步骤",
    "work_units": "工作单元",
}

_STAGE_PROGRESS_UNITS = {
    "serving_cohorts": "serving_batches",
    "serving_complete": "serving_batches",
    "cohort_tasks": "schedule_tasks",
    "cohort_preview_tasks": "schedule_tasks",
    "schedule": "schedule_tasks",
    "queued": "workflow_steps",
    "starting": "workflow_steps",
    "validation": "workflow_steps",
    "compilation": "workflow_steps",
    "reporting": "workflow_steps",
    "completed": "workflow_steps",
}


def progress_unit_for_stage(stage: str) -> str:
    """Return the logical unit counted by a known execution stage."""

    return _STAGE_PROGRESS_UNITS.get(str(stage), "work_units")


def progress_unit_label_zh(unit: str) -> str:
    """Return a stable Chinese display label for a logical progress unit."""

    return PROGRESS_UNIT_LABELS_ZH.get(str(unit), "工作单元")


class ExecutionCancelledError(RuntimeError):
    """Raised at a deterministic execution boundary after cancellation."""


class NativeFastWallBudgetError(RuntimeError):
    """The whole native fast candidate exhausted its allowed wall time."""


class _NativeLayerSamplingCutoff(BaseException):
    """Intentional incomplete cohort boundary, never a committed Token.

    Only the native fast layer orchestrator consumes this internal signal.
    It must discard the enclosing runtime rather than resume its controller
    state or turn its isolated physical layer windows into a ServingResult.
    """


@dataclass(frozen=True)
class ExecutionProgress:
    """Kernel-neutral progress over logical work units, not wall time."""

    stage: str
    completed: int
    total: Optional[int] = None
    message: str = ""
    simulated_time_ns: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unit: str = ""
    emitted_monotonic: Optional[float] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.unit:
            object.__setattr__(self, "unit", progress_unit_for_stage(self.stage))


ProgressCallback = Callable[[ExecutionProgress], None]
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True)
class ExecutionControl:
    """Shared cooperative progress/cancellation channel."""

    progress_callback: Optional[ProgressCallback] = None
    cancellation_callback: Optional[CancellationCallback] = None
    stop_request_path: Optional[str | Path] = None
    deadline_monotonic: Optional[float] = None
    progress_queue: Optional[SimpleQueue] = None

    def __post_init__(self) -> None:
        if self.stop_request_path is not None:
            if not isinstance(self.stop_request_path, (str, Path)):
                raise TypeError("stop_request_path must be a path or string")
            object.__setattr__(self, "stop_request_path", str(self.stop_request_path))
        if self.deadline_monotonic is not None:
            if (type(self.deadline_monotonic) not in (int, float)
                    or not math.isfinite(self.deadline_monotonic)
                    or self.deadline_monotonic < 0):
                raise ValueError("deadline_monotonic must be a finite non-negative clock value")
            object.__setattr__(self, "deadline_monotonic", float(self.deadline_monotonic))
        if self.progress_queue is not None and type(self.progress_queue) is not SimpleQueue:
            raise TypeError("progress_queue must be a native SimpleQueue")

    def raise_if_cancelled(self) -> None:
        # A user's stop always wins over a simultaneous wall-time deadline.
        if self.stop_request_path is not None and os.path.exists(self.stop_request_path):
            raise ExecutionCancelledError("仿真任务已取消")
        if (
            self.cancellation_callback is not None
            and self.cancellation_callback()
        ):
            raise ExecutionCancelledError("仿真任务已取消")
        if self.deadline_monotonic is not None and perf_counter() >= self.deadline_monotonic:
            raise NativeFastWallBudgetError("native total candidate wall budget exhausted")

    def report(
        self,
        stage: str,
        completed: int,
        total: Optional[int] = None,
        *,
        message: str = "",
        simulated_time_ns: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        unit: Optional[str] = None,
    ) -> None:
        if self.progress_callback is None and self.progress_queue is None:
            return
        progress = ExecutionProgress(
                stage=stage,
                completed=completed,
                total=total,
                message=message,
                simulated_time_ns=simulated_time_ns,
                metadata=dict(metadata or {}),
                unit=unit or progress_unit_for_stage(stage),
                emitted_monotonic=perf_counter() if self.progress_queue is not None else None,
            )
        if self.progress_queue is not None:
            self.progress_queue.put(progress)
        if self.progress_callback is not None:
            self.progress_callback(progress)


_NATIVE_CONTROL_REPORT = ExecutionControl.report
_NATIVE_CONTROL_CANCEL = ExecutionControl.raise_if_cancelled
_NATIVE_CONTROL_GETATTRIBUTE = ExecutionControl.__getattribute__
_NATIVE_CONTROL_NAMESPACE = tuple((name, value) for name, value in vars(ExecutionControl).items()
                                  if name != "__slotnames__")


def native_callback_free_control(control) -> bool:
    """Native stop/deadline/queue work cannot edit a simulation calendar."""
    namespace = vars(ExecutionControl)
    slotnames = namespace.get("__slotnames__")
    return (type(control) is ExecutionControl
        and (slotnames is None or type(slotnames) is list and not slotnames)
        and len(namespace) - ("__slotnames__" in namespace) == len(_NATIVE_CONTROL_NAMESPACE)
        and all(namespace.get(name) is value for name, value in _NATIVE_CONTROL_NAMESPACE)
        and ExecutionControl.__getattribute__ is _NATIVE_CONTROL_GETATTRIBUTE
        and getattr(control.report, "__func__", None) is _NATIVE_CONTROL_REPORT
        and getattr(control.raise_if_cancelled, "__func__", None) is _NATIVE_CONTROL_CANCEL
        and control.progress_callback is None and control.cancellation_callback is None
        and (control.stop_request_path is None or type(control.stop_request_path) is str)
        and (control.deadline_monotonic is None or type(control.deadline_monotonic) is float)
        and (control.progress_queue is None or type(control.progress_queue) is SimpleQueue))


__all__ = [
    "CancellationCallback",
    "NativeFastWallBudgetError",
    "ExecutionCancelledError",
    "ExecutionControl",
    "ExecutionProgress",
    "PROGRESS_UNIT_LABELS_ZH",
    "ProgressCallback",
    "progress_unit_for_stage",
    "progress_unit_label_zh",
]
