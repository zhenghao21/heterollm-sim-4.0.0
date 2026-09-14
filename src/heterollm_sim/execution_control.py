"""Progress and cooperative-cancellation contracts for V4 execution."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def raise_if_cancelled(self) -> None:
        if (
            self.cancellation_callback is not None
            and self.cancellation_callback()
        ):
            raise ExecutionCancelledError("仿真任务已取消")

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
        if self.progress_callback is None:
            return
        self.progress_callback(
            ExecutionProgress(
                stage=stage,
                completed=completed,
                total=total,
                message=message,
                simulated_time_ns=simulated_time_ns,
                metadata=dict(metadata or {}),
                unit=unit or progress_unit_for_stage(stage),
            )
        )


__all__ = [
    "CancellationCallback",
    "ExecutionCancelledError",
    "ExecutionControl",
    "ExecutionProgress",
    "PROGRESS_UNIT_LABELS_ZH",
    "ProgressCallback",
    "progress_unit_for_stage",
    "progress_unit_label_zh",
]
