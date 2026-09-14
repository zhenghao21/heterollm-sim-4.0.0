"""Incremental V4 schedule execution with policy-controlled retention.

All execution modes lower the same request chunks into the same event kernel.
``exact`` keeps every realized task, ``streaming`` keeps a bounded head/tail
sample, and ``aggregate`` keeps no realized tasks. Counters and numerical
aggregates are accumulated before retention and are therefore identical.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
from threading import Lock
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from weakref import WeakValueDictionary

from .contracts import (
    ResourceDemand,
    ResourceInterval,
    SIMULATION_SCHEMA_VERSION,
    RetentionPolicy,
    SimulationTrace,
    TaskCategory,
    TaskResult,
    TaskSpec,
    TraceMarker,
)
from .engine import _merge_engine_metadata
from .event_kernel import UnifiedEventKernel
from .execution_control import ExecutionControl
from .metrics import MetricsSummary, NS_PER_SECOND, RequestMetrics
from .planner import (
    RequestTaskChunk,
    StreamingScheduleIR,
    _compilation_scope,
    iter_request_task_chunks,
)


_CONTROL_CHECK_INTERVAL = 256
DEFAULT_RETAINED_TASK_LIMIT = 2_000


@dataclass(frozen=True)
class ScheduleExecutionResult:
    """Exact aggregates plus a trace selected by one V4 retention mode."""

    trace: SimulationTrace
    metrics: MetricsSummary
    task_count: int
    total_energy_pj: float
    resource_accounted_bytes: int
    proposed_tokens: int
    accepted_tokens: int
    kv_event_counts: Mapping[str, int]
    kv_logical_event_bytes: Mapping[str, int]
    kv_physical_event_bytes: Mapping[str, int]
    kv_phase_bytes: Mapping[Tuple[str, str, str], int]
    state_event_counts: Mapping[str, int]
    state_event_bytes: Mapping[str, int]
    analytical_coverage: Mapping[str, Mapping[str, object]]
    retention_policy: RetentionPolicy
    retained_task_limit: Optional[int]
    runtime_kernel_metrics: Mapping[str, object] = field(default_factory=dict)


@dataclass
class _RequestStream:
    request_id: str
    chunks: Iterator[RequestTaskChunk]
    terminal_task_id: str
    final: bool


class _PathKeyNode:
    """One canonical node in the trie of realized critical-path keys."""

    __slots__ = (
        "index",
        "key",
        "parent",
        "depth",
        "jumps",
        "__weakref__",
    )

    def __init__(
        self,
        index: "_PathKeyIndex",
        key: Optional[str],
        parent: Optional["_PathKeyNode"],
        jumps: Tuple["_PathKeyNode", ...] = (),
    ) -> None:
        self.index = index
        self.key = key
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.jumps = jumps


class _PathKeyIndex:
    """Canonical persistent trie with exact logarithmic lexicographic compare.

    Canonicalizing ``(parent, key)`` makes identity mean exact prefix equality;
    binary-lifting ancestors can therefore find the first differing key without
    hashing path contents or materializing either full path.  Weak values let
    abandoned branches disappear once no path state references them.
    """

    __slots__ = ("root", "_nodes", "_lock", "__weakref__")

    def __init__(self, *, thread_safe: bool = False) -> None:
        self._nodes = WeakValueDictionary()
        self._lock = Lock() if thread_safe else None
        self.root = _PathKeyNode(self, None, None)

    def extend(self, parent: _PathKeyNode, key: str) -> _PathKeyNode:
        cache_key = (parent, key)
        lock = self._lock
        if lock is None:
            return self._extend_unlocked(cache_key, parent, key)
        # Only the process-global compatibility index can be shared.  A
        # simulation-owned index is mutated by its single accumulator and
        # deliberately avoids a lock on every realized path extension.
        with lock:
            return self._extend_unlocked(cache_key, parent, key)

    def _extend_unlocked(
        self,
        cache_key: Tuple[_PathKeyNode, str],
        parent: _PathKeyNode,
        key: str,
    ) -> _PathKeyNode:
        existing = self._nodes.get(cache_key)
        if existing is not None:
            return existing

        depth = parent.depth + 1
        jumps: List[_PathKeyNode] = [parent]
        level = 1
        while (1 << level) <= depth:
            halfway = jumps[level - 1]
            jumps.append(halfway.jumps[level - 1])
            level += 1
        node = _PathKeyNode(self, key, parent, tuple(jumps))
        self._nodes[cache_key] = node
        return node


_PATH_KEY_INDEX = _PathKeyIndex(thread_safe=True)


class _PathState:
    __slots__ = (
        "distance",
        "previous",
        "node_key",
        "end_ns",
        "duration_ns",
        "category",
        "key_path",
    )

    def __init__(
        self,
        distance: float,
        previous: Optional["_PathState"] = None,
        node_key: Optional[str] = None,
        end_ns: float = 0.0,
        duration_ns: float = 0.0,
        category: Optional[TaskCategory] = None,
        *,
        key_path: Optional[_PathKeyNode] = None,
    ) -> None:
        self.distance = distance
        self.previous = previous
        self.node_key = node_key
        self.end_ns = end_ns
        self.duration_ns = duration_ns
        self.category = category
        if key_path is not None:
            self.key_path = key_path
        else:
            parent_key_path = (
                previous.key_path
                if previous is not None
                else _PATH_KEY_INDEX.root
            )
            self.key_path = (
                parent_key_path
                if node_key is None
                else parent_key_path.index.extend(parent_key_path, node_key)
            )


def _path_keys(state: _PathState) -> Tuple[str, ...]:
    reverse: List[str] = []
    cursor: Optional[_PathState] = state
    while cursor is not None:
        if cursor.node_key is not None:
            reverse.append(cursor.node_key)
        cursor = cursor.previous
    reverse.reverse()
    return tuple(reverse)


def _ancestor(node: _PathKeyNode, distance: int) -> _PathKeyNode:
    """Return the exact ancestor ``distance`` edges above ``node``."""

    cursor = node
    remaining = distance
    while remaining:
        level = remaining.bit_length() - 1
        cursor = cursor.jumps[level]
        remaining -= 1 << level
    return cursor


def _lowest_common_ancestor(
    left: _PathKeyNode, right: _PathKeyNode
) -> _PathKeyNode:
    """Return the canonical longest common prefix of two key paths."""

    if left.depth > right.depth:
        left = _ancestor(left, left.depth - right.depth)
    elif right.depth > left.depth:
        right = _ancestor(right, right.depth - left.depth)
    if left is right:
        return left

    for level in range(len(left.jumps) - 1, -1, -1):
        if level >= len(left.jumps):
            # A higher jump may already have moved both cursors close enough
            # to the root that this lower table entry is not present.  Its
            # conceptual saturated ancestor would be the shared root.
            continue
        left_up = left.jumps[level]
        right_up = right.jumps[level]
        if left_up is not right_up:
            left = left_up
            right = right_up
    parent = left.parent
    if parent is None:  # pragma: no cover - distinct roots are impossible
        raise AssertionError("canonical path nodes must share one root")
    return parent


def _compare_path_keys(left: _PathState, right: _PathState) -> int:
    """Compare ``_path_keys`` exactly in O(log(path length)) time."""

    left_path = left.key_path
    right_path = right.key_path
    if left_path.index is not right_path.index:
        # Private callers that manually assemble states may use independent
        # indexes.  Production accumulators never mix indexes; retain exact
        # historical tuple semantics for this compatibility edge case.
        left_keys = _path_keys(left)
        right_keys = _path_keys(right)
        return (left_keys > right_keys) - (left_keys < right_keys)
    if left_path is right_path:
        return 0

    prefix = _lowest_common_ancestor(left_path, right_path)
    if prefix is left_path:
        return -1
    if prefix is right_path:
        return 1

    left_child = _ancestor(left_path, left_path.depth - prefix.depth - 1)
    right_child = _ancestor(right_path, right_path.depth - prefix.depth - 1)
    if left_child.key == right_child.key:  # pragma: no cover - canonical invariant
        raise AssertionError("distinct canonical children must have distinct keys")
    return -1 if left_child.key < right_child.key else 1


def _better_path(left: _PathState, right: _PathState) -> _PathState:
    if left.distance != right.distance:
        return left if left.distance > right.distance else right
    return left if _compare_path_keys(left, right) < 0 else right


def _extend_path(
    source: _PathState,
    duration_ns: float,
    category: TaskCategory,
    node_key: str,
    end_ns: float,
) -> _PathState:
    distance = source.distance + duration_ns
    if distance == 0.0:
        # The implicit empty path wins every zero-distance tie, but retain the
        # realized end timestamp for subsequent dependency-edge checks.
        return _PathState(
            0.0,
            end_ns=end_ns,
            key_path=source.key_path.index.root,
        )
    return _PathState(
        distance,
        source,
        node_key,
        end_ns,
        duration_ns,
        category,
    )


def _category_mapping(state: _PathState) -> Dict[TaskCategory, float]:
    """Match ``metrics._CriticalPathIndex.categories`` accumulation order.

    The critical-path aggregator walks the selected predecessor chain from
    target to source.  Repeating that order here preserves both zero-duration
    categories and bit-for-bit floating-point totals across backends.
    """

    values: Dict[TaskCategory, float] = {}
    cursor: Optional[_PathState] = state
    while cursor is not None:
        if cursor.category is not None:
            values[cursor.category] = (
                values.get(cursor.category, 0.0) + cursor.duration_ns
            )
        cursor = cursor.previous
    return values


class _CriticalPathAccumulator:
    """Incremental equivalent of metrics._critical_path_categories."""

    def __init__(self) -> None:
        self._path_index = _PathKeyIndex()
        self._zero_path = _PathState(
            0.0,
            key_path=self._path_index.root,
        )
        self._task_states: Dict[str, _PathState] = {}
        self._dependency_users: Dict[str, int] = {}
        self._pinned: Set[str] = set()
        self._resource_states: Dict[Tuple[str, int], _PathState] = {}
        self._request_categories: Dict[
            str, Dict[TaskCategory, float]
        ] = {}
        self._request_latest: Dict[str, Tuple[float, _PathState]] = {}
        self._global_latest_end = -1.0
        self._global_target: Optional[_PathState] = None

    def register_chunk(self, chunk: RequestTaskChunk) -> None:
        self._pinned.add(chunk.terminal_task_id)
        for task in chunk.tasks:
            for dependency_id in task.dependencies:
                self._dependency_users[dependency_id] = (
                    self._dependency_users.get(dependency_id, 0) + 1
                )

    def unpin(self, task_id: str) -> None:
        self._pinned.discard(task_id)
        if self._dependency_users.get(task_id, 0) <= 0:
            self._task_states.pop(task_id, None)

    def observe(
        self,
        task: TaskSpec,
        *,
        start_ns: float,
        end_ns: float,
        demands: Sequence[ResourceDemand],
        resource_predecessors: Mapping[str, Mapping[str, object]],
        resource_lanes: Optional[Mapping[str, int]] = None,
    ) -> _PathState:
        candidates: List[_PathState] = [self._zero_path]
        dependency_ids = task.dependencies
        for dependency_id in dependency_ids:
            state = self._task_states.get(str(dependency_id))
            if state is not None:
                candidates.append(state)

        for resource_id, info in resource_predecessors.items():
            lane = int(info.get("lane", 0)) if isinstance(info, Mapping) else 0
            state = self._resource_states.get((str(resource_id), lane))
            if state is not None:
                candidates.append(state)

        start = candidates[0]
        for candidate in candidates[1:]:
            start = _better_path(start, candidate)

        end_state = _extend_path(
            start,
            end_ns - start_ns,
            task.category,
            task.task_id,
            end_ns,
        )
        self._task_states[task.task_id] = end_state
        for demand in demands:
            interval_end_ns = start_ns + demand.service_ns
            lane = int(resource_lanes.get(demand.resource_id, 0)) if resource_lanes else 0
            self._resource_states[(demand.resource_id, lane)] = _extend_path(
                start,
                interval_end_ns - start_ns,
                task.category,
                "{}@{}".format(task.task_id, demand.resource_id),
                interval_end_ns,
            )

        if task.marker == TraceMarker.REQUEST_DONE:
            self._request_categories[task.request_id] = _category_mapping(
                end_state
            )
        latest = self._request_latest.get(task.request_id)
        if latest is None or end_ns > latest[0]:
            self._request_latest[task.request_id] = (end_ns, end_state)
        elif end_ns == latest[0]:
            self._request_latest[task.request_id] = (
                end_ns,
                _better_path(latest[1], end_state),
            )

        if task.marker == TraceMarker.REQUEST_DONE:
            # Request metrics retain only the exact category totals.  The
            # global/resource frontier may still reference the shared path,
            # but completed requests no longer pin one full chain each.
            self._request_latest.pop(task.request_id, None)

        if end_ns > self._global_latest_end:
            self._global_latest_end = end_ns
            self._global_target = end_state
        elif end_ns == self._global_latest_end:
            self._global_target = (
                end_state
                if self._global_target is None
                else _better_path(self._global_target, end_state)
            )

        for dependency_id in dependency_ids:
            key = str(dependency_id)
            remaining = self._dependency_users.get(key, 0) - 1
            if remaining <= 0:
                self._dependency_users.pop(key, None)
                if key not in self._pinned:
                    self._task_states.pop(key, None)
            else:
                self._dependency_users[key] = remaining

        if (
            self._dependency_users.get(task.task_id, 0) <= 0
            and task.task_id not in self._pinned
        ):
            self._task_states.pop(task.task_id, None)
        return end_state

    def request_categories(self, request_id: str) -> Dict[TaskCategory, float]:
        completed = self._request_categories.get(request_id)
        if completed is not None:
            return dict(completed)
        latest = self._request_latest.get(request_id)
        target = latest[1] if latest is not None else None
        return _category_mapping(target) if target is not None else {}

    def global_categories(self) -> Dict[TaskCategory, float]:
        return (
            _category_mapping(self._global_target)
            if self._global_target is not None
            else {}
        )

@dataclass
class _RequestMetricsAccumulator:
    arrival_ns: Optional[float] = None
    first_token_ns: Optional[float] = None
    done_ns: Optional[float] = None
    token_events: List[Tuple[Optional[int], float, str]] = field(default_factory=list)
    category_ns: Dict[TaskCategory, float] = field(default_factory=dict)

    def observe(
        self, task: TaskSpec, *, start_ns: float, end_ns: float
    ) -> None:
        self.category_ns[task.category] = (
            self.category_ns.get(task.category, 0.0) + end_ns - start_ns
        )
        marker_time = (
            start_ns
            if task.marker == TraceMarker.REQUEST_ARRIVAL
            else end_ns
        )
        if task.marker == TraceMarker.REQUEST_ARRIVAL:
            self.arrival_ns = (
                marker_time
                if self.arrival_ns is None
                else min(self.arrival_ns, marker_time)
            )
        elif task.marker in (TraceMarker.FIRST_TOKEN, TraceMarker.TOKEN_EMIT):
            self.token_events.append((task.token_index, marker_time, task.task_id))
            if task.marker == TraceMarker.FIRST_TOKEN:
                self.first_token_ns = (
                    marker_time
                    if self.first_token_ns is None
                    else min(self.first_token_ns, marker_time)
                )
        elif task.marker == TraceMarker.REQUEST_DONE:
            self.done_ns = (
                marker_time
                if self.done_ns is None
                else max(self.done_ns, marker_time)
            )

    def build(
        self,
        request_id: str,
        critical: Mapping[TaskCategory, float],
    ) -> RequestMetrics:
        events = self.token_events
        if all(index is not None for index, _, _ in events):
            events.sort(key=lambda item: (item[0], item[1], item[2]))
        else:
            events.sort(key=lambda item: (item[1], item[2]))
        times = tuple(item[1] for item in events)
        first_token = self.first_token_ns
        if first_token is None and times:
            first_token = times[0]
        tbt = tuple(times[index] - times[index - 1] for index in range(1, len(times)))
        tpot = (
            (times[-1] - times[0]) / (len(times) - 1)
            if len(times) >= 2
            else None
        )
        return RequestMetrics(
            request_id=request_id,
            arrival_ns=self.arrival_ns,
            first_token_ns=first_token,
            last_token_ns=(times[-1] if times else None),
            done_ns=self.done_ns,
            ttft_ns=(
                first_token - self.arrival_ns
                if first_token is not None and self.arrival_ns is not None
                else None
            ),
            tbt_ns=tbt,
            tpot_ns=tpot,
            e2e_ns=(
                self.done_ns - self.arrival_ns
                if self.done_ns is not None and self.arrival_ns is not None
                else None
            ),
            client_e2e_ns=(
                times[-1] - self.arrival_ns
                if times and self.arrival_ns is not None
                else None
            ),
            visible_output_tokens=len(times),
            category_time_ns=dict(self.category_ns),
            critical_path_category_ns=dict(critical),
        )


_TaskHistoryRecord = Tuple[
    TaskSpec,
    float,
    float,
    float,
    float,
    Tuple[ResourceDemand, ...],
    Mapping[str, Mapping[str, object]],
    Mapping[str, int],
]


def _materialize_history_task(record: _TaskHistoryRecord) -> TaskResult:
    (
        task,
        start_ns,
        end_ns,
        dependency_ready_ns,
        effective_ready_ns,
        demands,
        resource_predecessors,
        resource_lanes,
    ) = record
    intervals = tuple(
        ResourceInterval(
            resource_id=demand.resource_id,
            start_ns=start_ns,
            end_ns=start_ns + demand.service_ns,
            bytes_moved=demand.bytes_moved,
            energy_pj=demand.energy_pj,
        )
        for demand in demands
    )
    return TaskResult(
        task_id=task.task_id,
        request_id=task.request_id,
        name=task.name,
        category=task.category,
        start_ns=start_ns,
        end_ns=end_ns,
        dependency_ready_ns=dependency_ready_ns,
        marker=task.marker,
        token_index=task.token_index,
        resource_intervals=intervals,
        metadata=_merge_engine_metadata(
            task.metadata,
            effective_ready_ns=effective_ready_ns,
            dependency_ids=task.dependencies,
            resource_predecessors=resource_predecessors,
            resource_lanes=resource_lanes,
        ),
    )


class _TaskHistory:
    def __init__(
        self,
        retention_policy: RetentionPolicy,
        retained_task_limit: int,
    ) -> None:
        self.retention_policy = RetentionPolicy(retention_policy)
        if self.retention_policy is RetentionPolicy.STREAMING and retained_task_limit < 1:
            raise ValueError("retained task limit must be positive for streaming")
        self.limit = retained_task_limit
        self.head_limit = (
            (retained_task_limit + 1) // 2
            if self.retention_policy is RetentionPolicy.STREAMING
            else 0
        )
        self.tail_limit = (
            retained_task_limit - self.head_limit
            if self.retention_policy is RetentionPolicy.STREAMING
            else 0
        )
        self.exact: List[_TaskHistoryRecord] = []
        self.head: List[_TaskHistoryRecord] = []
        self.tail: Deque[_TaskHistoryRecord] = deque(maxlen=self.tail_limit or None)
        self.total = 0

    def append(self, record: _TaskHistoryRecord) -> None:
        self.total += 1
        if self.retention_policy is RetentionPolicy.EXACT:
            self.exact.append(record)
        elif self.retention_policy is RetentionPolicy.AGGREGATE:
            return
        elif len(self.head) < self.head_limit:
            self.head.append(record)
        elif self.tail_limit:
            self.tail.append(record)

    def tasks(self) -> Tuple[TaskResult, ...]:
        tasks = [
            _materialize_history_task(record)
            for record in self.exact + self.head + list(self.tail)
        ]
        return tuple(
            sorted(
                tasks,
                key=lambda item: (item.start_ns, item.end_ns, item.task_id),
            )
        )


class _ResultAccumulator:
    def __init__(
        self,
        retention_policy: RetentionPolicy,
        retained_task_limit: int,
    ) -> None:
        self.retention_policy = RetentionPolicy(retention_policy)
        self.history = _TaskHistory(self.retention_policy, retained_task_limit)
        self.critical = _CriticalPathAccumulator()
        self.requests: Dict[str, _RequestMetricsAccumulator] = {}
        self.category_ns: Dict[TaskCategory, float] = {}
        self.resource_busy_ns: Dict[str, float] = {}
        self.total_energy_pj = 0.0
        self._energy_compensation_pj = 0.0
        self.bytes_moved = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.kv_event_counts: Dict[str, int] = {}
        self.kv_logical_event_bytes: Dict[str, int] = {}
        self.kv_physical_event_bytes: Dict[str, int] = {}
        self.kv_phase_bytes: Dict[Tuple[str, str, str], int] = {}
        self.state_event_counts: Dict[str, int] = {}
        self.state_event_bytes: Dict[str, int] = {}
        self.coverage: Dict[str, Dict[str, object]] = {}
        self.task_count = 0
        self.makespan_ns = 0.0

    def register_chunk(self, chunk: RequestTaskChunk) -> None:
        self.critical.register_chunk(chunk)
        grouped: Dict[
            Tuple[str, str, str, object, object], Dict[str, object]
        ] = {}
        for task in chunk.tasks:
            metadata = task.metadata
            event_kind = str(metadata.get("event_kind", ""))
            if event_kind == "mtp_commit":
                self.proposed_tokens += int(metadata.get("proposed_tokens", 0))
                self.accepted_tokens += int(metadata.get("accepted_tokens", 0))
            if event_kind.startswith("linear_state_"):
                self.state_event_counts[event_kind] = (
                    self.state_event_counts.get(event_kind, 0) + 1
                )
                self.state_event_bytes[event_kind] = (
                    self.state_event_bytes.get(event_kind, 0)
                    + int(metadata.get("bytes", 0))
                )
            if event_kind not in {
                "kv_read",
                "kv_append",
                "kv_prefetch",
                "kv_offload",
            }:
                continue
            logical_name = str(metadata.get("kv_access_id", task.name))
            if "kv_access_id" not in metadata:
                suffix = logical_name.rsplit(".", 1)[-1]
                if suffix == "local" or (
                    suffix.startswith("link") and suffix[4:].isdigit()
                ):
                    logical_name = logical_name.rsplit(".", 1)[0]
            key = (
                event_kind,
                task.request_id,
                logical_name,
                metadata.get("rank"),
                metadata.get("layer_id"),
            )
            physical = max(
                0,
                int(metadata.get("physical_bytes", metadata.get("bytes", 0))),
            )
            logical = max(0, int(metadata.get("logical_bytes", physical)))
            lowered = logical_name.lower()
            phase = (
                "prefill"
                if "prefill" in lowered
                else "decode"
                if "decode" in lowered or "mtp" in lowered
                else "other"
            )
            row = grouped.setdefault(
                key, {"logical": 0, "physical": 0, "phase": phase}
            )
            row["logical"] = max(int(row["logical"]), logical)
            row["physical"] = max(int(row["physical"]), physical)

        for (event_kind, _request, _name, _rank, _layer), row in grouped.items():
            logical = int(row["logical"])
            physical = int(row["physical"])
            phase = str(row["phase"])
            self.kv_event_counts[event_kind] = (
                self.kv_event_counts.get(event_kind, 0) + 1
            )
            self.kv_logical_event_bytes[event_kind] = (
                self.kv_logical_event_bytes.get(event_kind, 0) + logical
            )
            self.kv_physical_event_bytes[event_kind] = (
                self.kv_physical_event_bytes.get(event_kind, 0) + physical
            )
            self.kv_phase_bytes[(phase, event_kind, "logical")] = (
                self.kv_phase_bytes.get((phase, event_kind, "logical"), 0)
                + logical
            )
            self.kv_phase_bytes[(phase, event_kind, "physical")] = (
                self.kv_phase_bytes.get((phase, event_kind, "physical"), 0)
                + physical
            )

    def observe(
        self,
        task: TaskSpec,
        *,
        start_ns: float,
        end_ns: float,
        dependency_ready_ns: float,
        effective_ready_ns: float,
        demands: Tuple[ResourceDemand, ...],
        resource_predecessors: Mapping[str, Mapping[str, object]],
        resource_lanes: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.task_count += 1
        self.makespan_ns = max(self.makespan_ns, end_ns)
        self.history.append(
            (
                task,
                start_ns,
                end_ns,
                dependency_ready_ns,
                effective_ready_ns,
                demands,
                resource_predecessors,
                dict(resource_lanes or {}),
            )
        )
        self.critical.observe(
            task,
            start_ns=start_ns,
            end_ns=end_ns,
            demands=demands,
            resource_predecessors=resource_predecessors,
            resource_lanes=resource_lanes,
        )
        request = self.requests.setdefault(
            task.request_id, _RequestMetricsAccumulator()
        )
        request.observe(task, start_ns=start_ns, end_ns=end_ns)
        duration_ns = end_ns - start_ns
        self.category_ns[task.category] = (
            self.category_ns.get(task.category, 0.0) + duration_ns
        )
        for demand in demands:
            self.resource_busy_ns[demand.resource_id] = (
                self.resource_busy_ns.get(demand.resource_id, 0.0)
                + demand.service_ns
            )
            # Incremental execution observes demands in completion order,
            # while the exact trace reports them in deterministic task order.
            # Kahan summation keeps the aggregate invariant to the harmless
            # rounding drift exposed by large KV traffic energy totals.
            adjusted_energy = (
                float(demand.energy_pj) - self._energy_compensation_pj
            )
            updated_energy = self.total_energy_pj + adjusted_energy
            self._energy_compensation_pj = (
                updated_energy - self.total_energy_pj
            ) - adjusted_energy
            self.total_energy_pj = updated_energy
            self.bytes_moved += demand.bytes_moved
        self._observe_coverage(task, demands, duration_ns)

    def _observe_coverage(
        self,
        task: TaskSpec,
        demands: Sequence[ResourceDemand],
        duration_ns: float,
    ) -> None:
        metadata = task.metadata
        component = str(metadata.get("coverage_component", ""))
        name = task.name
        event_kind = str(metadata.get("event_kind", ""))
        if not component:
            if event_kind.startswith("linear_state_"):
                component = "linear_state"
            elif event_kind.startswith("kv_"):
                component = "kv_cache"
            elif event_kind.startswith("mtp_") or name.startswith("mtp"):
                component = "mtp_policy"
            elif "shared_expert" in name:
                component = "shared_expert"
            elif "expert" in name or "moe_router" in name:
                component = "routed_expert"
            elif "linear_" in name:
                component = "linear_attention"
            elif any(marker in name for marker in (".qkv", ".attention_", ".kv.")):
                component = "full_attention"
            elif ".mlp_" in name:
                component = "dense_ffn"
        if not component:
            return
        row = self.coverage.setdefault(
            component,
            {
                "task_count": 0.0,
                "operations": 0.0,
                "bytes": 0.0,
                "latency_ns": 0.0,
                "energy_pj": 0.0,
                "operator_ids": set(),
                "tensor_ids": set(),
            },
        )
        row["task_count"] = float(row["task_count"]) + 1.0
        row["operations"] = float(row["operations"]) + float(
            metadata.get("analytical_ops", 0.0)
        )
        row["bytes"] = float(row["bytes"]) + sum(
            float(demand.bytes_moved) for demand in demands
        )
        row["energy_pj"] = float(row["energy_pj"]) + sum(
            float(demand.energy_pj) for demand in demands
        )
        row["latency_ns"] = float(row["latency_ns"]) + duration_ns
        operator_id = metadata.get("model_operator_id", metadata.get("operator_id"))
        tensor_id = metadata.get("weight_tensor_id", metadata.get("tensor_id"))
        if operator_id:
            cast = row["operator_ids"]
            if isinstance(cast, set):
                cast.add(str(operator_id))
        if tensor_id:
            cast = row["tensor_ids"]
            if isinstance(cast, set):
                cast.add(str(tensor_id))

    def build(
        self,
        manifest: Any,
        retained_task_limit: int,
        resource_capacities: Mapping[str, int] = (),
    ) -> ScheduleExecutionResult:
        request_metrics = {
            request_id: request.build(
                request_id, self.critical.request_categories(request_id)
            )
            for request_id, request in sorted(self.requests.items())
        }
        completed_metrics = [
            metric for metric in request_metrics.values() if metric.done_ns is not None
        ]
        if completed_metrics:
            arrivals = [
                metric.arrival_ns for metric in completed_metrics
                if metric.arrival_ns is not None
            ]
            dones = [
                metric.done_ns for metric in completed_metrics
                if metric.done_ns is not None
            ]
            active_window_ns = max(0.0, max(dones) - (min(arrivals) if arrivals else 0.0))
        else:
            active_window_ns = 0.0
        if active_window_ns <= 0.0:
            active_window_ns = self.makespan_ns
        if active_window_ns > 0.0:
            seconds = active_window_ns / NS_PER_SECOND
            throughput = {
                "requests_per_s": sum(
                    metric.done_ns is not None for metric in completed_metrics
                )
                / seconds,
                "visible_output_tokens_per_s": sum(
                    metric.visible_output_tokens for metric in completed_metrics
                )
                / seconds,
            }
            utilization = {
                resource: busy
                / (
                    self.makespan_ns
                    * max(1, int(resource_capacities.get(resource, 1)))
                )
                for resource, busy in sorted(self.resource_busy_ns.items())
            }
        else:
            throughput = {
                "requests_per_s": 0.0,
                "visible_output_tokens_per_s": 0.0,
            }
            utilization = {
                resource: 0.0 for resource in sorted(self.resource_busy_ns)
            }
        metrics = MetricsSummary(
            makespan_ns=self.makespan_ns,
            request_metrics=request_metrics,
            throughput=throughput,
            category_time_ns=dict(self.category_ns),
            critical_path_category_ns=self.critical.global_categories(),
            resource_utilization=utilization,
        )
        coverage: Dict[str, Dict[str, object]] = {}
        for component, values in sorted(self.coverage.items()):
            coverage[component] = {
                key: (
                    sorted(str(item) for item in value)
                    if isinstance(value, set)
                    else value
                )
                for key, value in values.items()
            }
        metadata = dict(manifest.metadata)
        metadata.update(
            {
                "retention_policy": self.retention_policy.value,
                "event_kernel": "UnifiedEventKernel",
                "scheduling_semantics_version": SIMULATION_SCHEMA_VERSION,
                "trace_fidelity": {
                    RetentionPolicy.EXACT: "exact",
                    RetentionPolicy.STREAMING: "representative",
                    RetentionPolicy.AGGREGATE: "aggregate",
                }[self.retention_policy],
            }
        )
        manifest = replace(
            manifest,
            schema_version=SIMULATION_SCHEMA_VERSION,
            metadata=metadata,
        )
        warnings = ()
        if self.retention_policy is RetentionPolicy.STREAMING:
            warnings = (
                "streaming retention keeps exact aggregates and a bounded task trace",
            )
        elif self.retention_policy is RetentionPolicy.AGGREGATE:
            warnings = (
                "aggregate retention keeps exact aggregates without a task trace",
            )
        trace = SimulationTrace(
            manifest=manifest,
            tasks=self.history.tasks(),
            resource_busy_ns=dict(self.resource_busy_ns),
            makespan_ns=self.makespan_ns,
            resource_capacities=dict(sorted(resource_capacities.items())),
            warnings=warnings,
        )
        return ScheduleExecutionResult(
            trace=trace,
            metrics=metrics,
            task_count=self.task_count,
            total_energy_pj=self.total_energy_pj,
            resource_accounted_bytes=self.bytes_moved,
            proposed_tokens=self.proposed_tokens,
            accepted_tokens=self.accepted_tokens,
            kv_event_counts=dict(sorted(self.kv_event_counts.items())),
            kv_logical_event_bytes=dict(sorted(self.kv_logical_event_bytes.items())),
            kv_physical_event_bytes=dict(sorted(self.kv_physical_event_bytes.items())),
            kv_phase_bytes=dict(self.kv_phase_bytes),
            state_event_counts=dict(sorted(self.state_event_counts.items())),
            state_event_bytes=dict(sorted(self.state_event_bytes.items())),
            analytical_coverage=coverage,
            retention_policy=self.retention_policy,
            retained_task_limit=(
                retained_task_limit
                if self.retention_policy is RetentionPolicy.STREAMING
                else None
            ),
        )


def _request_iterator(schedule: StreamingScheduleIR) -> Iterator[Any]:
    requests = schedule.requests
    if schedule.scenario.workload.requests:
        yield from sorted(
            requests, key=lambda item: (item.arrival_ns, item.request_id)
        )
        return
    yield from requests


def _request_task_chunks(
    schedule: StreamingScheduleIR,
    request: Any,
) -> Iterator[RequestTaskChunk]:
    """Lower one request and split only at stable visible-token boundaries."""

    yield from iter_request_task_chunks(schedule.scenario, request)


def _arrival_key(request: Any) -> Tuple[float, float, float, float, str]:
    arrival = float(request.arrival_ns)
    task_id = "{}.00001.request_arrival".format(request.request_id)
    return (arrival, arrival, 0.0, arrival, task_id)


def execute_incremental_schedule(
    schedule: StreamingScheduleIR,
    *,
    retention_policy: RetentionPolicy = RetentionPolicy.EXACT,
    control: Optional[ExecutionControl] = None,
    retained_task_limit: int = DEFAULT_RETAINED_TASK_LIMIT,
    execution_kernel: Optional[UnifiedEventKernel] = None,
    runtime_origin_ns: float = 0.0,
) -> ScheduleExecutionResult:
    """Execute one static schedule on an optional live V4 runtime kernel."""

    retention_policy = RetentionPolicy(retention_policy)
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
        raise ValueError("execution_kernel must be drained before schedule append")
    execution_control = control or ExecutionControl()
    execution_control.raise_if_cancelled()
    execution_control.report(
        "schedule",
        0,
        None,
        message="开始执行统一事件计划",
        metadata={"retention_policy": retention_policy.value},
    )

    accumulator = _ResultAccumulator(retention_policy, retained_task_limit)
    kernel = execution_kernel or UnifiedEventKernel()
    resource_capacities = getattr(schedule, "resource_capacities", {})
    # Static streaming and online serving must use the same declared
    # execution lanes. ``ensure_resource_capacities`` is idempotent for an
    # already-bootstrapped kernel and rejects conflicting capacities.
    kernel.ensure_resource_capacities(resource_capacities)
    streams_by_terminal: Dict[str, _RequestStream] = {}

    requests = _request_iterator(schedule)
    next_request = next(requests, None)

    def add_chunk(stream: _RequestStream, chunk: RequestTaskChunk) -> None:
        accumulator.register_chunk(chunk)
        tasks = tuple(
            task
            if task.marker is TraceMarker.REQUEST_ARRIVAL
            or task.earliest_start_ns >= runtime_origin_ns
            else replace(task, earliest_start_ns=runtime_origin_ns)
            for task in chunk.tasks
        )
        kernel.add_tasks(tasks)
        stream.terminal_task_id = chunk.terminal_task_id
        stream.final = chunk.final
        streams_by_terminal[chunk.terminal_task_id] = stream

    def add_request(request: Any) -> None:
        chunks = _request_task_chunks(schedule, request)
        chunk = next(chunks, None)
        if chunk is None:
            return
        stream = _RequestStream(
            request_id=request.request_id,
            chunks=chunks,
            terminal_task_id=chunk.terminal_task_id,
            final=chunk.final,
        )
        add_chunk(stream, chunk)

    with _compilation_scope(schedule.scenario):
        while kernel.has_active_tasks or next_request is not None:
            current = kernel.peek_ready_key()
            if next_request is not None and (
                current is None or _arrival_key(next_request) <= current
            ):
                add_request(next_request)
                next_request = next(requests, None)
                continue
            if current is None:
                if kernel.has_active_tasks:
                    raise ValueError("event schedule contains a dependency cycle")
                break
            event = kernel.step()
            if event is None:  # pragma: no cover - guarded by peek_ready_key
                raise AssertionError("ready kernel did not emit an event")
            task = event.task
            accumulator.observe(
                task,
                start_ns=event.start_ns,
                end_ns=event.end_ns,
                dependency_ready_ns=event.dependency_ready_ns,
                effective_ready_ns=event.effective_ready_ns,
                demands=event.demands,
                resource_predecessors=event.resource_predecessors,
                resource_lanes=event.resource_lanes,
            )

            stream = streams_by_terminal.pop(task.task_id, None)
            if stream is not None:
                if not stream.final:
                    next_chunk = next(stream.chunks, None)
                    if next_chunk is None:
                        raise ValueError(
                            "request stream ended before final chunk: {}".format(
                                stream.request_id
                            )
                        )
                    add_chunk(stream, next_chunk)
                    accumulator.critical.unpin(task.task_id)
                else:
                    accumulator.critical.unpin(task.task_id)
            kernel.release_completed(task.task_id)

            if accumulator.task_count % _CONTROL_CHECK_INTERVAL == 0:
                execution_control.raise_if_cancelled()
                execution_control.report(
                    "schedule",
                    accumulator.task_count,
                    None,
                    message="正在执行统一事件计划",
                    simulated_time_ns=accumulator.makespan_ns,
                    metadata={
                        "active_task_count": kernel.active_task_count,
                        "retention_policy": retention_policy.value,
                        "retained_task_limit": (
                            retained_task_limit
                            if retention_policy is RetentionPolicy.STREAMING
                            else None
                        ),
                    },
                )
                execution_control.raise_if_cancelled()

    execution_control.raise_if_cancelled()
    result = replace(
        accumulator.build(
            schedule.manifest,
            retained_task_limit,
            resource_capacities,
        ),
        runtime_kernel_metrics=dict(kernel.metrics),
    )
    execution_control.report(
        "schedule",
        result.task_count,
        result.task_count,
        message="统一事件计划执行完成",
        simulated_time_ns=result.trace.makespan_ns,
        metadata={
            "retained_task_count": len(result.trace.tasks),
            "retention_policy": retention_policy.value,
        },
    )
    return result


__all__ = [
    "DEFAULT_RETAINED_TASK_LIMIT",
    "ScheduleExecutionResult",
    "execute_incremental_schedule",
]
