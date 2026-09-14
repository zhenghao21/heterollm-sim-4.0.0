"""Metric aggregation for deterministic simulation traces."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contracts import (
    ChangePointInterval,
    DEFAULT_COMPONENT_TIMESERIES_POINT_LIMIT,
    MAX_COMPONENT_TIMESERIES_POINT_LIMIT,
    SimulationTrace,
    TaskCategory,
    TaskResult,
    TraceMarker,
)

NS_PER_SECOND = 1_000_000_000.0


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    arrival_ns: Optional[float]
    first_token_ns: Optional[float]
    last_token_ns: Optional[float]
    done_ns: Optional[float]
    ttft_ns: Optional[float]
    tbt_ns: Tuple[float, ...]
    tpot_ns: Optional[float]
    e2e_ns: Optional[float]
    client_e2e_ns: Optional[float]
    visible_output_tokens: int
    category_time_ns: Mapping[TaskCategory, float]
    critical_path_category_ns: Mapping[TaskCategory, float]


@dataclass(frozen=True)
class MetricsSummary:
    makespan_ns: float
    request_metrics: Mapping[str, RequestMetrics]
    throughput: Mapping[str, float]
    category_time_ns: Mapping[TaskCategory, float]
    critical_path_category_ns: Mapping[TaskCategory, float]
    resource_utilization: Mapping[str, float]


@dataclass(frozen=True)
class CostMetricsSummary:
    """Metrics required by an analytical cohort cost calculation."""

    category_time_ns: Mapping[TaskCategory, float]
    critical_path_category_ns: Mapping[TaskCategory, float]


@dataclass(frozen=True)
class BoundedChangePointSeries:
    """Server-merged interval series with deterministic size metadata."""

    points: Tuple[ChangePointInterval, ...]
    raw_point_count: int
    point_limit: int

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def merged(self) -> bool:
        return self.point_count < self.raw_point_count


def bound_change_point_intervals(
    intervals: Iterable[ChangePointInterval],
    *,
    point_limit: int = DEFAULT_COMPONENT_TIMESERIES_POINT_LIMIT,
    clamp_ratio: bool = False,
) -> BoundedChangePointSeries:
    """Coalesce and deterministically bound a contiguous interval series.

    Adjacent equal values are losslessly merged first.  If the result still
    exceeds ``point_limit``, contiguous rows are grouped and represented by a
    duration-weighted mean.  This keeps response size independent of trace
    length while preserving the time-weighted mean of the curve.
    """

    if (
        isinstance(point_limit, bool)
        or not isinstance(point_limit, int)
        or point_limit < 1
        or point_limit > MAX_COMPONENT_TIMESERIES_POINT_LIMIT
    ):
        raise ValueError(
            "每条时序曲线的点数上限必须在 1 到 {} 之间".format(
                MAX_COMPONENT_TIMESERIES_POINT_LIMIT
            )
        )

    normalized: List[ChangePointInterval] = []
    for interval in intervals:
        if not isinstance(interval, ChangePointInterval):
            raise TypeError("时序曲线必须由 ChangePointInterval 组成")
        if interval.end_ns <= interval.start_ns:
            continue
        value = float(interval.value)
        if clamp_ratio:
            value = min(1.0, max(0.0, value))
        normalized.append(
            ChangePointInterval(
                float(interval.start_ns),
                float(interval.end_ns),
                value,
            )
        )
    normalized.sort(key=lambda item: (item.start_ns, item.end_ns, item.value))
    raw_point_count = len(normalized)

    coalesced: List[ChangePointInterval] = []
    for interval in normalized:
        if coalesced and interval.start_ns < coalesced[-1].end_ns:
            raise ValueError("同一时序曲线的区间不能重叠")
        if coalesced and interval.start_ns > coalesced[-1].end_ns:
            raise ValueError("同一时序曲线的区间必须连续")
        if (
            coalesced
            and interval.start_ns == coalesced[-1].end_ns
            and math.isclose(
                interval.value,
                coalesced[-1].value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            previous = coalesced[-1]
            coalesced[-1] = ChangePointInterval(
                previous.start_ns,
                interval.end_ns,
                previous.value,
            )
        else:
            coalesced.append(interval)

    if len(coalesced) <= point_limit:
        return BoundedChangePointSeries(
            tuple(coalesced), raw_point_count, point_limit
        )

    # Integer bucket boundaries avoid floating point partition drift and
    # produce exactly ``point_limit`` rows whenever there is enough input.
    bounded: List[ChangePointInterval] = []
    count = len(coalesced)
    for bucket_index in range(point_limit):
        start_index = bucket_index * count // point_limit
        end_index = (bucket_index + 1) * count // point_limit
        bucket = coalesced[start_index:end_index]
        if not bucket:
            continue
        duration = sum(item.end_ns - item.start_ns for item in bucket)
        if duration <= 0.0:
            continue
        value = sum(
            item.value * (item.end_ns - item.start_ns) for item in bucket
        ) / duration
        if clamp_ratio:
            value = min(1.0, max(0.0, value))
        bounded.append(
            ChangePointInterval(bucket[0].start_ns, bucket[-1].end_ns, value)
        )
    return BoundedChangePointSeries(tuple(bounded), raw_point_count, point_limit)


def summarize_metrics(trace: SimulationTrace) -> MetricsSummary:
    """Aggregate request latency, critical path, and utilization metrics."""

    tasks = tuple(trace.tasks)
    tasks_by_request = _tasks_by_request(tasks)
    critical_path_index = _build_critical_path_index(tasks)
    request_metrics = {
        request_id: _summarize_request(
            request_id,
            request_tasks,
            critical_path_index,
        )
        for request_id, request_tasks in sorted(tasks_by_request.items())
    }
    category_time_ns = _category_time(tasks)
    critical_path_category_ns = critical_path_index.categories(
        _global_targets(tasks)
    )
    resource_utilization = _resource_utilization(
        trace.resource_busy_ns, trace.makespan_ns, trace.resource_capacities
    )
    throughput = _throughput(request_metrics, trace.makespan_ns)

    return MetricsSummary(
        makespan_ns=trace.makespan_ns,
        request_metrics=request_metrics,
        throughput=throughput,
        category_time_ns=category_time_ns,
        critical_path_category_ns=critical_path_category_ns,
        resource_utilization=resource_utilization,
    )


def summarize_cost_metrics(trace: SimulationTrace) -> CostMetricsSummary:
    """Aggregate only the global cost fields consumed by online lowering.

    A serving cohort is represented as one synthetic request.  Calling the
    general report aggregator would still query both request and global
    targets and aggregate latency and throughput fields that the cost provider
    discards.
    """

    tasks = tuple(trace.tasks)
    return CostMetricsSummary(
        category_time_ns=_category_time(tasks),
        critical_path_category_ns=_critical_path_categories(
            tasks, _global_targets(tasks)
        ),
    )


def _summarize_request(
    request_id: str,
    request_tasks: Sequence[TaskResult],
    critical_path_index: "_CriticalPathIndex",
) -> RequestMetrics:
    arrival_ns = _first_marker_time(request_tasks, TraceMarker.REQUEST_ARRIVAL)
    token_events = _token_events(request_tasks)
    token_times = tuple(event_time for _, event_time, _ in token_events)
    first_token_ns = _first_marker_time(request_tasks, TraceMarker.FIRST_TOKEN)
    if first_token_ns is None and token_times:
        first_token_ns = token_times[0]
    done_ns = _last_marker_time(request_tasks, TraceMarker.REQUEST_DONE)

    ttft_ns = _delta(first_token_ns, arrival_ns)
    tbt_ns = tuple(
        token_times[index] - token_times[index - 1]
        for index in range(1, len(token_times))
    )
    tpot_ns = None
    if len(token_times) >= 2:
        tpot_ns = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
    e2e_ns = _delta(done_ns, arrival_ns)
    last_token_ns = token_times[-1] if token_times else None
    client_e2e_ns = _delta(last_token_ns, arrival_ns)
    targets = _request_targets(request_id, request_tasks)

    return RequestMetrics(
        request_id=request_id,
        arrival_ns=arrival_ns,
        first_token_ns=first_token_ns,
        last_token_ns=last_token_ns,
        done_ns=done_ns,
        ttft_ns=ttft_ns,
        tbt_ns=tbt_ns,
        tpot_ns=tpot_ns,
        e2e_ns=e2e_ns,
        client_e2e_ns=client_e2e_ns,
        visible_output_tokens=len(token_times),
        category_time_ns=_category_time(request_tasks),
        critical_path_category_ns=critical_path_index.categories(targets),
    )


def _tasks_by_request(tasks: Iterable[TaskResult]) -> Dict[str, List[TaskResult]]:
    grouped: Dict[str, List[TaskResult]] = {}
    for task in tasks:
        grouped.setdefault(task.request_id, []).append(task)
    for request_id in grouped:
        grouped[request_id].sort(key=lambda task: (task.start_ns, task.end_ns, task.task_id))
    return grouped


def _marker_time(task: TaskResult) -> float:
    if task.marker == TraceMarker.REQUEST_ARRIVAL:
        return task.start_ns
    return task.end_ns


def _first_marker_time(
    tasks: Sequence[TaskResult], marker: TraceMarker
) -> Optional[float]:
    times = [_marker_time(task) for task in tasks if task.marker == marker]
    return min(times) if times else None


def _last_marker_time(
    tasks: Sequence[TaskResult], marker: TraceMarker
) -> Optional[float]:
    times = [_marker_time(task) for task in tasks if task.marker == marker]
    return max(times) if times else None


def _token_events(tasks: Sequence[TaskResult]) -> Tuple[Tuple[int, float, str], ...]:
    events: List[Tuple[Optional[int], float, str]] = []
    for task in tasks:
        if task.marker in (TraceMarker.FIRST_TOKEN, TraceMarker.TOKEN_EMIT):
            events.append((task.token_index, _marker_time(task), task.task_id))

    if all(token_index is not None for token_index, _, _ in events):
        events.sort(key=lambda event: (event[0], event[1], event[2]))
    else:
        events.sort(key=lambda event: (event[1], event[2]))
    return tuple((index if index is not None else offset, time, task_id) for offset, (index, time, task_id) in enumerate(events))


def _delta(
    later_ns: Optional[float], earlier_ns: Optional[float]
) -> Optional[float]:
    if later_ns is None or earlier_ns is None:
        return None
    return later_ns - earlier_ns


def _category_time(tasks: Sequence[TaskResult]) -> Dict[TaskCategory, float]:
    totals: Dict[TaskCategory, float] = {}
    for task in tasks:
        totals[task.category] = totals.get(task.category, 0.0) + task.duration_ns
    return totals


def _resource_utilization(
    resource_busy_ns: Mapping[str, float], makespan_ns: float,
    resource_capacities: Optional[Mapping[str, int]] = None,
) -> Dict[str, float]:
    if makespan_ns <= 0:
        return {resource_id: 0.0 for resource_id in sorted(resource_busy_ns)}
    return {
        resource_id: busy_ns / (makespan_ns * max(1, int((resource_capacities or {}).get(resource_id, 1))))
        for resource_id, busy_ns in sorted(resource_busy_ns.items())
    }


def _throughput(
    request_metrics: Mapping[str, RequestMetrics], makespan_ns: float
) -> Dict[str, float]:
    completed_items = [
        item for item in request_metrics.values() if item.done_ns is not None
    ]
    if not completed_items:
        return {"requests_per_s": 0.0, "visible_output_tokens_per_s": 0.0}
    arrivals = [item.arrival_ns for item in completed_items if item.arrival_ns is not None]
    dones = [item.done_ns for item in completed_items if item.done_ns is not None]
    start_ns = min(arrivals) if arrivals else 0.0
    end_ns = max(dones) if dones else makespan_ns
    # Exclude an idle prefix before the first request, matching steady-state
    # runtime measurements from llama.cpp.
    window_ns = max(0.0, end_ns - start_ns)
    if window_ns <= 0.0:
        window_ns = max(0.0, makespan_ns)
    if window_ns <= 0.0:
        return {"requests_per_s": 0.0, "visible_output_tokens_per_s": 0.0}
    visible_tokens = sum(item.visible_output_tokens for item in completed_items)
    seconds = window_ns / NS_PER_SECOND
    return {
        "requests_per_s": len(completed_items) / seconds,
        "visible_output_tokens_per_s": visible_tokens / seconds,
    }


def _global_targets(tasks: Sequence[TaskResult]) -> Tuple[str, ...]:
    if not tasks:
        return ()
    max_end = max(task.end_ns for task in tasks)
    return tuple(sorted(task.task_id for task in tasks if task.end_ns == max_end))


def _request_targets(
    request_id: str, request_tasks: Sequence[TaskResult]
) -> Tuple[str, ...]:
    done_tasks = [
        task.task_id
        for task in request_tasks
        if task.request_id == request_id and task.marker == TraceMarker.REQUEST_DONE
    ]
    if done_tasks:
        return tuple(sorted(done_tasks))
    if not request_tasks:
        return ()
    max_end = max(task.end_ns for task in request_tasks)
    return tuple(sorted(task.task_id for task in request_tasks if task.end_ns == max_end))


_CriticalPathNode = Tuple[str, str, str]
_CriticalPathEdge = Tuple[
    _CriticalPathNode,
    float,
    Optional[TaskCategory],
    Optional[str],
]
_CriticalPathPredecessor = Tuple[
    _CriticalPathNode,
    float,
    Optional[TaskCategory],
    Optional[str],
]


def _critical_path_start_node(task_id: str) -> _CriticalPathNode:
    return ("start", task_id, "")


def _critical_path_end_node(task_id: str) -> _CriticalPathNode:
    return ("end", task_id, "")


def _critical_path_interval_node(
    task_id: str, resource_id: str
) -> _CriticalPathNode:
    return ("interval", task_id, resource_id)


class _CriticalPathIndex:
    """One topological critical-path solve shared by every target query."""

    def __init__(self, tasks: Sequence[TaskResult]) -> None:
        task_by_id = {task.task_id: task for task in tasks}
        adjacency: Dict[_CriticalPathNode, List[_CriticalPathEdge]] = {}
        indegree: Dict[_CriticalPathNode, int] = {}

        def ensure(node: _CriticalPathNode) -> None:
            adjacency.setdefault(node, [])
            indegree.setdefault(node, 0)

        def add_edge(
            source: _CriticalPathNode,
            target: _CriticalPathNode,
            duration_ns: float = 0.0,
            category: Optional[TaskCategory] = None,
            node_key: Optional[str] = None,
        ) -> None:
            ensure(source)
            ensure(target)
            adjacency[source].append(
                (target, duration_ns, category, node_key)
            )
            indegree[target] += 1

        interval_nodes: Dict[Tuple[str, str], _CriticalPathNode] = {}
        for task in tasks:
            start = _critical_path_start_node(task.task_id)
            end = _critical_path_end_node(task.task_id)
            ensure(start)
            add_edge(
                start,
                end,
                task.duration_ns,
                task.category,
                task.task_id,
            )
            for interval in task.resource_intervals:
                node = _critical_path_interval_node(
                    task.task_id, interval.resource_id
                )
                interval_nodes[(task.task_id, interval.resource_id)] = node
                add_edge(
                    start,
                    node,
                    interval.end_ns - interval.start_ns,
                    task.category,
                    "{}@{}".format(task.task_id, interval.resource_id),
                )

        for task in tasks:
            start = _critical_path_start_node(task.task_id)
            dependency_ids = task.metadata.get("_engine_dependency_ids", ())
            if isinstance(dependency_ids, tuple):
                for predecessor_id in sorted(
                    str(item) for item in dependency_ids
                ):
                    predecessor = task_by_id.get(predecessor_id)
                    if predecessor is not None:
                        # Causality remains valid across queue wait; filtering
                        # on end==start dropped delayed dependencies.
                        add_edge(
                            _critical_path_end_node(predecessor_id), start
                        )

            resource_predecessors = task.metadata.get(
                "_engine_resource_predecessors", {}
            )
            if isinstance(resource_predecessors, dict):
                for resource_id, raw_info in sorted(
                    resource_predecessors.items()
                ):
                    if not isinstance(raw_info, dict):
                        continue
                    predecessor_id = str(raw_info.get("task_id", ""))
                    if predecessor_id not in task_by_id:
                        continue
                    predecessor_node = interval_nodes.get(
                        (predecessor_id, str(resource_id)),
                        _critical_path_end_node(predecessor_id),
                    )
                    add_edge(predecessor_node, start)

        self._distances: Dict[_CriticalPathNode, float] = {
            node: 0.0 for node in indegree
        }
        self._predecessors: Dict[
            _CriticalPathNode, _CriticalPathPredecessor
        ] = {}
        self._target_cache: Dict[
            Tuple[str, ...], Dict[TaskCategory, float]
        ] = {}
        self._node_category_cache: Dict[
            _CriticalPathNode, Dict[TaskCategory, float]
        ] = {}

        ready = [node for node, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        visited = 0
        while ready:
            source = heapq.heappop(ready)
            visited += 1
            for target, duration_ns, category, node_key in sorted(
                adjacency[source]
            ):
                candidate_distance = self._distances[source] + duration_ns
                current_distance = self._distances[target]
                replace = candidate_distance > current_distance
                if candidate_distance == current_distance:
                    candidate_path = self._materialize_path(source)
                    if node_key is not None:
                        candidate_path += (node_key,)
                    current_path = self._materialize_path(target)
                    replace = candidate_path < current_path
                if replace:
                    self._distances[target] = candidate_distance
                    self._predecessors[target] = (
                        source,
                        duration_ns,
                        category,
                        node_key,
                    )
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, target)

        if visited != len(indegree):
            raise ValueError("critical-path graph contains a cycle")

    def _materialize_path(
        self, node: _CriticalPathNode
    ) -> Tuple[str, ...]:
        """Build a tie-break path only when an equal-distance edge needs it."""

        reverse_keys: List[str] = []
        cursor = node
        while cursor in self._predecessors:
            source, _duration_ns, _category, node_key = self._predecessors[
                cursor
            ]
            if node_key is not None:
                reverse_keys.append(node_key)
            cursor = source
        reverse_keys.reverse()
        return tuple(reverse_keys)

    def categories(
        self, target_task_ids: Sequence[str]
    ) -> Dict[TaskCategory, float]:
        target_key = tuple(target_task_ids)
        cached = self._target_cache.get(target_key)
        if cached is not None:
            return dict(cached)
        if not target_key or not self._distances:
            return {}

        best_node: Optional[_CriticalPathNode] = None
        best_distance = 0.0
        best_path: Tuple[str, ...] = ()
        for task_id in sorted(set(target_key)):
            candidate_node = _critical_path_end_node(task_id)
            if candidate_node not in self._distances:
                continue
            candidate_distance = self._distances[candidate_node]
            candidate_path = self._materialize_path(candidate_node)
            if candidate_distance > best_distance or (
                candidate_distance == best_distance
                and candidate_path < best_path
            ):
                best_node = candidate_node
                best_distance = candidate_distance
                best_path = candidate_path

        if best_node is None:
            totals: Dict[TaskCategory, float] = {}
        else:
            cached_node_totals = self._node_category_cache.get(best_node)
            if cached_node_totals is not None:
                totals = dict(cached_node_totals)
            else:
                totals = {}
                cursor: Optional[_CriticalPathNode] = best_node
                while cursor is not None and cursor in self._predecessors:
                    source, duration_ns, category, _node_key = (
                        self._predecessors[cursor]
                    )
                    if category is not None:
                        totals[category] = (
                            totals.get(category, 0.0) + duration_ns
                        )
                    cursor = source
                self._node_category_cache[best_node] = dict(totals)

        self._target_cache[target_key] = dict(totals)
        return dict(totals)


def _build_critical_path_index(
    tasks: Sequence[TaskResult],
) -> _CriticalPathIndex:
    return _CriticalPathIndex(tasks)


def _critical_path_categories(
    tasks: Sequence[TaskResult], target_task_ids: Sequence[str]
) -> Dict[TaskCategory, float]:
    if not tasks or not target_task_ids:
        return {}
    return _build_critical_path_index(tasks).categories(target_task_ids)
