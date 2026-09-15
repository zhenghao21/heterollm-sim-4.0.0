"""Explicit runtime/graph invocation lifecycle over existing task DAGs.

This module models when a declared overhead occurs; it does not estimate that
cost from an LLM's latency.  Callers supply independently sourced costs and a
structural graph signature (kernel path, shapes and layouts).  No model-name,
prompt fingerprint or token-index table participates in lifecycle transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .contracts import ResourceDemand, TaskCategory, TaskSpec
from .event_kernel import validate_task_graph


@dataclass(frozen=True)
class InvocationOverheads:
    """Costs in ns, separately owned by host launch and synchronization.

    Nonzero costs require provenance.  ``evidence_type`` is only a description
    of the supplied evidence, never an accuracy/acceptance certification.
    """

    initialization_ns: float = 0.0
    graph_capture_ns: float = 0.0
    graph_update_ns: float = 0.0
    launch_ns: float = 0.0
    synchronize_ns: float = 0.0
    host_resource_id: str = "runtime.host_launch"
    sync_resource_id: str = "runtime.host_sync"
    evidence_type: str = "analytical"
    source_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (self.initialization_ns, self.graph_capture_ns,
                  self.graph_update_ns, self.launch_ns, self.synchronize_ns)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invocation overheads must be finite non-negative ns")
        for value in (self.host_resource_id, self.sync_resource_id):
            if not isinstance(value, str) or not value:
                raise ValueError("invocation resource ids must be non-empty text")
        if self.evidence_type not in {"analytical", "microbenchmark", "source_trace"}:
            raise ValueError("unknown invocation evidence type")
        if not isinstance(self.source_refs, tuple) or any(
            not isinstance(ref, str) or not ref for ref in self.source_refs
        ):
            raise ValueError("source_refs must be a tuple of non-empty references")
        if any(values) and not self.source_refs:
            raise ValueError("nonzero invocation overheads require source_refs")


@dataclass(frozen=True)
class InvocationPlan:
    tasks: Tuple[TaskSpec, ...]
    completion_task_id: str
    invocation_id: str
    context_id: str
    graph_key: Optional[str]
    graph_signature: Optional[str]
    transition: str
    initialization_applied: bool
    overhead_ns: Mapping[str, float]


class InvocationLifecycle:
    """One runtime identity's ordered graph-executable lifetimes.

    Each ``lower`` call consumes one invocation ID and returns a closed graph
    plus explicit dependencies on prior lifecycle tasks.  Submit all returned
    tasks together, or retain the prior completion leases when appending them.
    Warmup uses the same state transitions and is tagged, not subtracted or
    averaged into later tokens.  Graph updates wait for the previous use of
    that executable; independent contexts retain independent lifetimes.
    """

    def __init__(self, runtime_identity: str, overheads: InvocationOverheads) -> None:
        if not isinstance(runtime_identity, str) or not runtime_identity:
            raise ValueError("runtime_identity must be non-empty text")
        if not isinstance(overheads, InvocationOverheads):
            raise TypeError("overheads must be InvocationOverheads")
        self.runtime_identity = runtime_identity
        self.overheads = overheads
        self._context_ready: Dict[str, str] = {}
        self._graphs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._seen_invocations: set[str] = set()
        self._seen_task_ids: set[str] = set()

    def lower(
        self,
        tasks: Sequence[TaskSpec],
        *,
        invocation_id: str,
        context_id: str,
        graph_key: Optional[str] = None,
        graph_signature: Optional[str] = None,
        timing_role: str = "measured",
        allow_graph_update: bool = False,
    ) -> InvocationPlan:
        for value in (invocation_id, context_id):
            if not isinstance(value, str) or not value:
                raise ValueError("invocation/context ids must be non-empty text")
        if invocation_id in self._seen_invocations:
            raise ValueError("invocation_id is already lowered")
        if timing_role not in {"warmup", "measured"}:
            raise ValueError("timing_role must be warmup or measured")
        if not isinstance(allow_graph_update, bool):
            raise ValueError("allow_graph_update must be boolean")
        if (graph_key is None) != (graph_signature is None):
            raise ValueError("graph_key and structural graph_signature are required together")
        if graph_key is not None and any(
            not isinstance(value, str) or not value for value in (graph_key, graph_signature)
        ):
            raise ValueError("graph key/signature must be non-empty text")
        chunk = tuple(tasks)
        if not chunk:
            raise ValueError("an invocation requires an executed task graph")
        validate_task_graph(chunk)
        prefix = "invocation.{}.".format(invocation_id)
        reserved = {prefix + phase for phase in (
            "initialize", "graph_capture", "graph_update", "launch", "synchronize"
        )}
        input_ids = {task.task_id for task in chunk}
        invocation_ready_ns = min(task.earliest_start_ns for task in chunk if not task.dependencies)
        if input_ids & (reserved | self._seen_task_ids):
            raise ValueError("invocation task IDs collide with lifecycle history")
        first_context = context_id not in self._context_ready
        previous_graph = self._graphs.get((context_id, graph_key)) if graph_key else None
        transition = "eager"
        if graph_key:
            if previous_graph is None:
                transition = "capture"
            elif previous_graph[0] == graph_signature:
                transition = "replay"
            else:
                transition = "update" if allow_graph_update else "recapture"
        metadata = {
            "runtime_identity": self.runtime_identity,
            "invocation_id": invocation_id, "context_id": context_id,
            "graph_key": graph_key, "graph_signature": graph_signature,
            "lifecycle_transition": transition, "timing_role": timing_role,
            "prediction_source": self.overheads.evidence_type,
            "parameter_source_refs": self.overheads.source_refs,
            "validation_status": "mechanism_only",
        }
        lowered = []
        dependencies: Tuple[str, ...] = ()
        overhead_ns: Dict[str, float] = {}
        context_ready_id = self._context_ready.get(context_id)

        def add_phase(phase: str, duration: float, resource: str,
                      predecessors: Tuple[str, ...]) -> str:
            task_id = prefix + phase
            lowered.append(TaskSpec(
                task_id=task_id, request_id=chunk[0].request_id,
                name=phase, category=TaskCategory.SYNCHRONIZATION,
                dependencies=tuple(dict.fromkeys(predecessors)),
                earliest_start_ns=invocation_ready_ns,
                # Zero-cost barriers are dependencies, not occupied lanes.
                demands=(ResourceDemand(resource, duration),) if duration else (),
                metadata={**metadata, "lifecycle_event": phase,
                          "one_time_event": phase in {"initialize", "graph_capture", "graph_update"}},
            ))
            overhead_ns[phase] = duration
            return task_id

        if first_context:
            context_ready_id = add_phase("initialize", self.overheads.initialization_ns,
                                         self.overheads.host_resource_id, ())
        if context_ready_id:
            dependencies = (context_ready_id,)
        # One graph executable's update/use order is explicit; other graph keys
        # and contexts are not merged into a device-wide serialization lock.
        if previous_graph:
            dependencies += (previous_graph[1],)
        if transition in {"capture", "recapture", "update"}:
            phase = "graph_update" if transition == "update" else "graph_capture"
            duration = self.overheads.graph_update_ns if transition == "update" else self.overheads.graph_capture_ns
            dependencies = (add_phase(phase, duration, self.overheads.host_resource_id, dependencies),)
        launch_id = add_phase("launch", self.overheads.launch_ns,
                              self.overheads.host_resource_id, dependencies)
        for task in chunk:
            lowered.append(replace(
                task,
                dependencies=task.dependencies or (launch_id,),
                metadata={**task.metadata, **metadata},
            ))
        parent_ids = {dep for task in chunk for dep in task.dependencies}
        leaves = tuple(sorted(input_ids - parent_ids))
        done_id = add_phase("synchronize", self.overheads.synchronize_ns,
                            self.overheads.sync_resource_id, leaves)
        all_ids = {task.task_id for task in lowered}
        if all_ids & self._seen_task_ids:
            raise ValueError("lifecycle task IDs collide with a previous invocation")
        # Commit state only after all validation and task construction succeeds.
        self._seen_invocations.add(invocation_id)
        self._seen_task_ids.update(all_ids)
        self._context_ready[context_id] = context_ready_id or launch_id
        if graph_key:
            self._graphs[(context_id, graph_key)] = (graph_signature, done_id)
        return InvocationPlan(
            tuple(lowered), done_id, invocation_id, context_id, graph_key,
            graph_signature, transition, first_context, overhead_ns,
        )


__all__ = ["InvocationOverheads", "InvocationPlan", "InvocationLifecycle"]
