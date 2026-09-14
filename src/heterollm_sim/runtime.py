"""Dynamic control-plane runtime over the single unified event kernel."""

from __future__ import annotations

import heapq
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .contracts import TaskSpec
from .event_kernel import SubmissionReceipt, UnifiedEventKernel
from .runtime_ir import HardwareTask, KernelCompletion, RuntimeDelta, RuntimeTask
from .runtime_lowering import (
    RuntimeDispatcher,
    _reject_callbacks,
    build_initial_control_tasks,
)
from .runtime_state import RuntimeState


@dataclass(frozen=True)
class RuntimeRunResult:
    """Completed dynamic DAG plus aggregate scheduler/runtime state."""

    completions: Tuple[KernelCompletion, ...]
    submissions: Tuple[SubmissionReceipt, ...]
    state: RuntimeState
    kernel_metrics: Mapping[str, object]

    @property
    def makespan_ns(self) -> float:
        return float(self.kernel_metrics.get("makespan_ns", 0.0))

    @property
    def completed_count(self) -> int:
        return len(self.completions)


class ControlPlaneRuntime:
    """Dispatch typed completions and atomically submit their downstream DAG."""

    def __init__(
        self,
        profile: Optional[Mapping[str, object]] = None,
        state: Optional[RuntimeState] = None,
        *,
        resource_capacities: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.profile = dict(profile or {})
        _reject_callbacks(self.profile, "profile")
        self.state = state if state is not None else RuntimeState()
        if not isinstance(self.state, RuntimeState):
            raise TypeError("state must be a RuntimeState")

        profile_capacities = self.profile.get("resource_capacities", {})
        if not isinstance(profile_capacities, Mapping):
            raise TypeError("profile.resource_capacities must be a mapping")
        capacities = dict(profile_capacities)
        capacities.update(dict(resource_capacities or {}))
        self.resource_capacities = capacities

        profile_bytes = self.profile.get("capacity_bytes", {})
        if not isinstance(profile_bytes, Mapping):
            raise TypeError("profile.capacity_bytes must be a mapping")
        for resource_id, capacity in profile_bytes.items():
            if resource_id not in self.state.capacity_bytes:
                if (
                    not isinstance(resource_id, str)
                    or not resource_id
                    or isinstance(capacity, bool)
                    or not isinstance(capacity, int)
                    or capacity < 0
                ):
                    raise ValueError(
                        "profile capacity bytes require non-negative integer values"
                    )
                self.state.capacity_bytes[resource_id] = capacity

        self.dispatcher = RuntimeDispatcher(self.profile)
        self.kernel: Optional[UnifiedEventKernel] = None
        self._run_capacity_reservations: Dict[str, int] = {}
        self._run_allocations: Dict[str, int] = {}

    def _begin_run_transaction(self) -> None:
        self.state.release_transient_ledger(
            self._run_capacity_reservations,
            self._run_allocations,
        )
        self._run_capacity_reservations = {}
        self._run_allocations = {}

    def _record_run_delta(self, delta: RuntimeDelta) -> None:
        for resource_id, amount in delta.capacity_reservations.items():
            self._run_capacity_reservations[resource_id] = (
                self._run_capacity_reservations.get(resource_id, 0) + amount
            )
        for allocation_id, amount in delta.allocations.items():
            self._run_allocations[allocation_id] = (
                self._run_allocations.get(allocation_id, 0) + amount
            )

    @staticmethod
    def _lower_tasks(tasks: Iterable[RuntimeTask]) -> Tuple[TaskSpec, ...]:
        lowered = []
        for task in tasks:
            if isinstance(task, HardwareTask):
                lowered.append(task.to_task_spec())
            elif isinstance(task, TaskSpec):
                lowered.append(task)
            else:
                raise TypeError(
                    "runtime tasks must be HardwareTask or TaskSpec instances"
                )
        return tuple(lowered)

    def run(
        self,
        initial_tasks: Optional[Iterable[RuntimeTask]],
        context: Mapping[str, object],
    ) -> RuntimeRunResult:
        """Run one dynamic DAG to completion without executable callbacks."""

        if not isinstance(context, Mapping):
            raise TypeError("runtime context must be a mapping")
        _reject_callbacks(context)
        context_data = dict(context)
        self._begin_run_transaction()

        context_capacities = context_data.get("capacity_bytes", {})
        if context_capacities and not isinstance(context_capacities, Mapping):
            raise TypeError("context.capacity_bytes must be a mapping")
        if isinstance(context_capacities, Mapping):
            for resource_id, capacity in context_capacities.items():
                if resource_id not in self.state.capacity_bytes:
                    if (
                        not isinstance(resource_id, str)
                        or not resource_id
                        or isinstance(capacity, bool)
                        or not isinstance(capacity, int)
                        or capacity < 0
                    ):
                        raise ValueError(
                            "context capacity bytes require non-negative integers"
                        )
                    self.state.capacity_bytes[resource_id] = capacity

        roots: Iterable[RuntimeTask]
        if initial_tasks is None:
            roots = build_initial_control_tasks(
                self.profile,
                self.state,
                context_data,
            )
        else:
            roots = tuple(initial_tasks)
        lowered_roots = self._lower_tasks(roots)

        kernel = UnifiedEventKernel(
            resource_capacities=self.resource_capacities
        )
        self.kernel = kernel
        submissions = [kernel.submit(lowered_roots)]
        completions = []
        pending_completions = []
        retained_for_run = []

        while kernel.has_active_tasks or pending_completions:
            ready_key = kernel.peek_ready_key()
            next_start_ns = ready_key[0] if ready_key is not None else None

            # Kernel tasks are selected by start time, but their runtime state
            # must not become visible until their end time.  Deliver every
            # completion at a timestamp before allowing an old or newly
            # submitted ready task at that same timestamp to compete.
            if pending_completions and (
                next_start_ns is None
                or pending_completions[0][0] <= next_start_ns
            ):
                _key_end, _key_phase, _key_sequence, _key_id, completion = (
                    heapq.heappop(pending_completions)
                )
                expansion = self.dispatcher.dispatch(
                    completion,
                    self.state,
                    context_data,
                )
                # Preflight state on an isolated copy before submit so an
                # invalid delta changes neither state nor kernel.  The real
                # commit is safe after submit because it repeats the already
                # validated transition from the same state.
                projected_state = deepcopy(self.state)
                projected_state.apply(expansion.delta)
                if expansion.tasks or expansion.retain_dependencies:
                    receipt = kernel.submit(
                        self._lower_tasks(expansion.tasks),
                        retain_dependencies=expansion.retain_dependencies,
                    )
                    submissions.append(receipt)
                    retained_for_run.extend(receipt.retained_dependencies)
                self.state.apply(expansion.delta)
                self._record_run_delta(expansion.delta)
                kernel.release_completed(completion.task.task_id)
                completions.append(completion)
                continue

            completion = kernel.step_completion()
            if completion is None:
                kernel.assert_drained()
                raise RuntimeError("active runtime DAG has no ready task")
            heapq.heappush(
                pending_completions,
                (
                    completion.end_ns,
                    completion.phase,
                    completion.sequence,
                    completion.task.task_id,
                    completion,
                ),
            )

        kernel.assert_drained()
        for task_id in reversed(retained_for_run):
            kernel.release_completed(task_id)
        return RuntimeRunResult(
            completions=tuple(completions),
            submissions=tuple(submissions),
            state=self.state,
            kernel_metrics=dict(kernel.metrics),
        )


__all__ = ["ControlPlaneRuntime", "RuntimeRunResult"]
