"""CUDA Graph transitions from the pinned llama.cpp CUDA backend.

This is distinct from CPU GGML graph reuse and from planner template caches.
The runtime owns this state. Preparing a cost must not advance it; only a
successful execution commits it. Inputs are *modeled backend invocations*,
not observed native decisions or target-model latency measurements.

The adapter must provide the complete property snapshot compared by
ggml_cuda_graph_update_required: the ggml_tensor (including symbolic pointer
identities, view offsets, op parameters and names), plus every source data
identity, shape and stride. A kernel shape or an ubatch ID is insufficient.
Unknown initialization or an unknown executable-update outcome is rejected.

Source: llama.cpp d3146f2b56c2db4711ac8391871c9e529d1946d7,
ggml/src/ggml-cuda/ggml-cuda.cu:2550-2655,4378-4467;
ggml/src/ggml-cuda/common.cuh:1265-1295,1468-1493.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Hashable, Mapping, Sequence

from .contracts import TaskCategory, TaskSpec


SOURCE_REVISION = "d3146f2b56c2db4711ac8391871c9e529d1946d7"
SCHEMA = "heterollm.llama-cuda-graph-lifecycle/v1"
SWEEP_INTERVAL_US = 5_000_000
EVICTION_AGE_US = 10_000_000


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty source-bound identity")


def _immutable(value):
    if value is None or type(value) in (str, int, bool, bytes):
        return
    if isinstance(value, tuple):
        for item in value:
            _immutable(item)
        return
    raise ValueError("CUDA graph node snapshots must contain immutable exact values")


@dataclass(frozen=True)
class CudaGraphInvocation:
    """One GGML CUDA backend graph_compute, not one LLM token.

    ``graph_key`` identifies the first GGML tensor object, scoped to context.
    ``node_properties`` is the complete ordered equality snapshot. Its count
    is GGML nodes, *not* CUDA launch nodes (one GGML op can launch many kernels).
    An adapter must derive compatibility from the actual operator dispatch.
    ``update_result`` is needed only when updating a previous executable;
    constraints_failure models native's explicit destroy/reinstantiate path.
    """

    context_id: str
    graph_key: str
    graph_uid: int
    node_properties: tuple[Hashable, ...]
    compatible: bool
    compatibility_reason: str
    enabled: bool = True
    source_revision: str = SOURCE_REVISION
    update_result: str | None = None

    def __post_init__(self):
        for name in ("context_id", "graph_key", "compatibility_reason"):
            _text(getattr(self, name), name)
        if type(self.graph_uid) is not int or self.graph_uid < 0:
            raise ValueError("graph_uid must be a nonnegative integer")
        if not isinstance(self.node_properties, tuple) or not self.node_properties:
            raise ValueError("complete nonempty ordered GGML node properties are required")
        _immutable(self.node_properties)
        if type(self.compatible) is not bool or type(self.enabled) is not bool:
            raise ValueError("Graph enablement and compatibility must be known booleans")
        if self.source_revision != SOURCE_REVISION:
            raise ValueError("CUDA Graph lifecycle source revision is unsupported")
        if self.update_result not in (None, "success", "constraints_failure"):
            raise ValueError("unknown CUDA Graph executable update result")


@dataclass(frozen=True)
class CudaGraphEntry:
    context_id: str
    graph_key: str
    graph_uid: int = 0
    node_properties: tuple[Hashable, ...] = ()
    warmup_complete: bool = False
    graph_exists: bool = False
    executable_generation: int = 0
    capture_generation: int = 0
    last_used_us: int = 0

    @property
    def executable_id(self):
        if not self.executable_generation:
            return None
        return f"{self.context_id}:{self.graph_key}:exec:{self.executable_generation}"


@dataclass(frozen=True)
class CudaGraphState:
    """Known cold by default; do not use it for unobserved native warmup."""

    entries: tuple[CudaGraphEntry, ...] = ()
    last_sweeps: tuple[tuple[str, int], ...] = ()
    revision: int = 0
    known: bool = True
    unresolved_update_count: int = 0

    @classmethod
    def unknown(cls):
        return cls(known=False)


@dataclass(frozen=True)
class CudaGraphTransition:
    revision: int
    state: CudaGraphState
    context_id: str
    graph_key: str
    use_graph: bool
    reason: str
    properties_changed: bool | None
    events: tuple[str, ...]
    replay_id: str | None = None
    executable_id: str | None = None
    evicted: tuple[CudaGraphEntry, ...] = ()
    previous_state: CudaGraphState | None = None
    update_compatibility_pending: bool = False

    def metadata(self) -> dict:
        return {
            "schema": SCHEMA,
            "source_revision": SOURCE_REVISION,
            "context_id": self.context_id,
            "graph_key": self.graph_key,
            "invocation_id": f"{self.context_id}:{self.graph_key}:invocation:{self.revision + 1}",
            "decision": "graph_launch" if self.use_graph else "direct",
            "reason": self.reason,
            "properties_changed": self.properties_changed,
            "events": self.events,
            "replay_id": self.replay_id,
            "executable_id": self.executable_id if not self.state.unresolved_update_count else None,
            "evicted_executable_ids": tuple(e.executable_id for e in self.evicted),
            "body_executions": 1,
            "capture_executes_body": False,
            "native_execution_verified": False,
            "update_compatibility_pending": self.update_compatibility_pending,
            "pricing_ready": self.state.unresolved_update_count == 0,
            "unresolved_update_count": self.state.unresolved_update_count,
        }


def prepare_cuda_graph(state: CudaGraphState, invocation: CudaGraphInvocation,
                       *, host_time_us: int, allow_unknown_update: bool = False) -> CudaGraphTransition:
    """Derive one transition without mutating runtime or task-cache state.

    The clock is modeled host time. A graph-cache lookup sweeps the current
    device context, exactly at the pinned backend's 5s/10s thresholds. Callers
    making other timed backend lookups must model those lookups too.
    """
    if type(host_time_us) is not int or host_time_us < 0:
        raise ValueError("modeled host_time_us must be a nonnegative integer")
    if type(allow_unknown_update) is not bool:
        raise ValueError("allow_unknown_update must be an explicit diagnostic boolean")
    if not state.known:
        raise ValueError("CUDA Graph lifecycle initialization is unknown; model warmup explicitly")
    if host_time_us < max((*[e.last_used_us for e in state.entries],
                           *[time for _, time in state.last_sweeps]), default=0):
        raise ValueError("CUDA Graph host clock must be monotonic")
    entries = {(e.context_id, e.graph_key): e for e in state.entries}
    sweeps = dict(state.last_sweeps)
    context = invocation.context_id
    evicted = ()
    if host_time_us - sweeps.get(context, 0) >= SWEEP_INTERVAL_US:
        sweeps[context] = host_time_us
        evicted = tuple(e for e in entries.values() if e.context_id == context
                        and host_time_us - e.last_used_us >= EVICTION_AGE_US)
        for entry in evicted:
            del entries[(entry.context_id, entry.graph_key)]
    key = (context, invocation.graph_key)
    entry = entries.get(key, CudaGraphEntry(*key))
    entry = replace(entry, last_used_us=host_time_us)
    changed = None
    use_graph = False
    update_pending = False
    events = ()
    reason = "disabled"
    if invocation.enabled and not invocation.compatible:
        reason = "incompatible:" + invocation.compatibility_reason
    if invocation.enabled and invocation.compatible:
        if invocation.graph_uid != 0 and invocation.graph_uid == entry.graph_uid:
            if len(invocation.node_properties) != len(entry.node_properties):
                raise ValueError("same nonzero GGML uid cannot change node count")
            changed = False
        else:
            changed = invocation.node_properties != entry.node_properties
            entry = replace(entry, graph_uid=invocation.graph_uid,
                            node_properties=invocation.node_properties)
        capture = False
        if not entry.warmup_complete:
            if changed:
                reason = "warmup_properties_changed"
            else:
                reason = "warmup_stable_capture"
                entry = replace(entry, warmup_complete=True)
                use_graph = capture = True
        elif changed:
            entry = replace(entry, warmup_complete=False)
            reason = "properties_changed_reset_warmup"
        else:
            use_graph = True
            capture = entry.executable_generation == 0
            reason = "stable_capture" if capture else "stable_replay"
        if use_graph:
            emitted = []
            new_executable = entry.executable_generation == 0
            if capture:
                if entry.graph_exists:
                    emitted.append("destroy_graph")
                emitted.append("capture")
                entry = replace(entry, graph_exists=True,
                                capture_generation=entry.capture_generation + 1)
            if new_executable:
                emitted.append("instantiate")
                entry = replace(entry, executable_generation=state.revision + 1)
            if capture:
                emitted.append("update")
                # A fresh executable is instantiated from exactly this graph.
                # Updating the very same graph satisfies the update constraints.
                if not new_executable:
                    if invocation.update_result is None:
                        if not allow_unknown_update:
                            raise ValueError("recapture requires source-derived executable update compatibility")
                        update_pending = True
                    if invocation.update_result == "constraints_failure":
                        emitted[-1] = "update_failure"
                        emitted.extend(("destroy_exec", "instantiate"))
                        entry = replace(entry, executable_generation=state.revision + 1)
                        new_executable = True
                elif invocation.update_result == "constraints_failure":
                    raise ValueError("fresh executable is instantiated from the identical captured graph")
            emitted.append("launch_submit_unresolved" if update_pending else (
                "first_launch_submit" if new_executable else "replay_submit"))
            events = tuple(emitted)
    if not use_graph:
        events = ("ordinary_submit",)
    entries[key] = entry
    new_state = CudaGraphState(tuple(entries.values()), tuple(sweeps.items()),
                               state.revision + 1, unresolved_update_count=(
                                   state.unresolved_update_count + int(update_pending)))
    return CudaGraphTransition(state.revision, new_state, context, invocation.graph_key,
        use_graph, reason, changed, events,
        replay_id=f"{context}:replay:{state.revision + 1}" if use_graph else None,
        executable_id=entry.executable_id if use_graph else None, evicted=evicted,
        previous_state=state, update_compatibility_pending=update_pending)


class CudaGraphRuntime:
    """Runtime-owned transaction boundary; never store this in template caches."""

    def __init__(self, state: CudaGraphState | None = None):
        self.state = state if state is not None else CudaGraphState()

    def prepare(self, invocation: CudaGraphInvocation, *, host_time_us: int,
                allow_unknown_update: bool = False):
        return prepare_cuda_graph(self.state, invocation, host_time_us=host_time_us,
                                  allow_unknown_update=allow_unknown_update)

    def commit(self, transition: CudaGraphTransition):
        if transition.revision != self.state.revision or transition.previous_state is not self.state:
            raise ValueError("stale or already committed CUDA Graph transition")
        self.state = transition.state

    def failed(self):
        # A failure is not evidence that native cleaned up or retained capture.
        self.state = CudaGraphState(revision=self.state.revision + 1, known=False)


def _is_direct_device_memory_task(task: TaskSpec, device: str,
                                  memory_components: Sequence[str]) -> bool:
    """Recognize a physical load/store on this GPU's mapped local memory.

    Writes name memory as their target, unlike compute tasks and reads. They
    remain real memory work inside the backend body, never cost-free markers.
    """
    meta = task.metadata
    storage = meta.get("direct_memory_component")
    if (storage not in memory_components or task.category != TaskCategory.MEMORY
            or not task.demands or meta.get("resource_accounting") != "direct_memory_access"
            or meta.get("orchestration_stage") or meta.get("serving_output_stage")
            or meta.get("transfer_kind") or meta.get("resource_transfer_bytes") != 0
            or not isinstance(meta.get("physical_memory_config"), Mapping)):
        return False
    direction = meta.get("access_kind")
    endpoints = ((storage, device) if direction == "READ" else
                 (device, storage) if direction == "WRITE" else None)
    if endpoints != (meta.get("source_component"), meta.get("target_component")):
        return False
    accesses = meta.get("memory_access")
    if isinstance(accesses, Mapping):
        accesses = (accesses,)
    owner = meta.get("physical_owner")
    return bool(owner and any(demand.resource_id == owner for demand in task.demands)
                and isinstance(accesses, (tuple, list)) and accesses
                and all(isinstance(access, Mapping)
                        and access.get("physical_owner") == owner
                        and access.get("operation") == direction.lower()
                        for access in accesses))


def bind_cuda_graph_tasks(tasks: Sequence[TaskSpec], transition: CudaGraphTransition,
                          *, member_task_ids: Sequence[str], topology: str | None = None,
                          node_count: int | None = None,
                          node_count_basis: str | None = None,
                          phase_structures: Mapping | None = None,
                          device_marker_task_ids: Sequence[str] = (),
                          device_memory_component_ids: Sequence[str] = (),
                          structure_id: str | None = None) -> tuple[TaskSpec, ...]:
    """Attach a proven modeled backend split to its exact planner task members.

    This emits metadata, not a price or a physical-work replacement. The cost
    owner reads lifecycle events from the first launch. All external inputs to
    the captured body gate that launch; internal compute/DRAM dependencies are
    preserved. The caller supplies the *complete* split, including device work,
    not only its kernel_launch tasks. Native verification remains separate.
    """
    ids = tuple(member_task_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("CUDA Graph binding requires unique explicit member task IDs")
    members = set(ids)
    task_map = {task.task_id: task for task in tasks}
    if len(task_map) != len(tasks) or not members.issubset(task_map):
        raise ValueError("CUDA Graph binding references missing or duplicate task IDs")
    ordered = [task for task in tasks if task.task_id in members]
    markers = set(device_marker_task_ids)
    if not markers.issubset(members) or any(task_map[key].demands for key in markers):
        raise ValueError("CUDA device markers must be cost-free members of this split")
    launches = [task for task in ordered if task.metadata.get("phase") == "kernel_launch"]
    if not launches:
        raise ValueError("CUDA Graph split has no modeled launch tasks")
    if any(task.metadata.get("cuda_graph_lifecycle") is not None for task in ordered):
        raise ValueError("CUDA Graph task membership is already bound")
    first = launches[0]
    # Do not accept a host task inside the capture: CUDA cannot record CPU work.
    target = first.metadata.get("target_component")
    if not target or any(task.metadata.get("target_component") != target for task in launches):
        raise ValueError("one CUDA Graph binding must target one CUDA backend device")
    if any(task.metadata.get("target_component") not in (None, target)
           and task.task_id not in markers
           and not _is_direct_device_memory_task(task, target, device_memory_component_ids)
           for task in ordered):
        raise ValueError("CUDA Graph body contains a foreign-device or host task")
    if topology is not None:
        _text(topology, "CUDA Graph topology")
    if node_count is not None and (type(node_count) is not int or node_count < 1):
        raise ValueError("CUDA graph node_count must be positive")
    if node_count is not None:
        _text(node_count_basis, "CUDA node count basis")
    external = tuple(dict.fromkeys(dep for task in ordered for dep in task.dependencies
                                   if dep not in members))
    # An external input produced by a descendant of the body creates a cycle,
    # exposing an invalid backend split instead of launching before that input.
    children = defaultdict(list)
    for task in tasks:
        for dep in task.dependencies:
            children[dep].append(task.task_id)
    descendants = set(members)
    pending = deque(members)
    while pending:
        for child in children[pending.popleft()]:
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    if any(dep in descendants for dep in external):
        raise ValueError("CUDA Graph membership crosses a host dependency split")
    result = []
    lifecycle_metadata = transition.metadata()
    lifecycle_reference = {key: lifecycle_metadata[key] for key in (
        "schema", "source_revision", "invocation_id", "decision", "replay_id", "executable_id")}
    for task in tasks:
        if task.task_id not in members:
            result.append(task)
            continue
        metadata = {**task.metadata, "cuda_graph_captured": transition.use_graph,
                    "cuda_graph_device": target,
                    "cuda_graph_id": transition.replay_id,
                    "cuda_graph_executable_id": transition.executable_id,
                    "cuda_graph_lifecycle": (lifecycle_metadata if task.task_id == first.task_id
                                             else lifecycle_reference)}
        if task.task_id == first.task_id:
            metadata["cuda_graph_lifecycle_events"] = transition.events
            metadata["cuda_graph_member_task_ids"] = ids
            metadata["cuda_graph_topology"] = topology
            metadata["cuda_graph_node_count"] = node_count
            metadata["cuda_graph_node_count_basis"] = node_count_basis
            if structure_id is not None:
                metadata["cuda_graph_structure_id"] = structure_id
            if phase_structures is not None:
                metadata["cuda_graph_phase_structures"] = dict(phase_structures)
        dependencies = task.dependencies
        if transition.use_graph and task.task_id == first.task_id:
            dependencies = tuple(dict.fromkeys((*dependencies, *external)))
        result.append(replace(task, metadata=metadata, dependencies=dependencies))
    return tuple(result)
