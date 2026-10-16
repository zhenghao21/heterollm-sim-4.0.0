"""Reduce actual persistent-kernel events without retaining physical traces."""
from __future__ import annotations

from math import isfinite
from types import SimpleNamespace
from typing import Mapping, Optional

from .contracts import TaskCategory
from .event_kernel import KernelEvent
from .engine import _resource_release_points
from .scalable_serving import (
    ScheduleCostResult, TaskExecutionRecord, _better_path, _coverage_component,
    _extend_path, compact_execution_metadata,
)


def execution_time_breakdown(records, *, origin_ns=None, end_ns=None, host_stages=None):
    """Describe overlapping host/non-host task intervals, not a serial split.

    These are unions of task spans, including waits within a task; they are
    not hardware utilization. In particular device time is not the residual
    after subtracting host work from wall time. The explicit window may add
    idle time before the first or after the last task.
    """
    rows = tuple(records)
    host_stages = frozenset(host_stages if host_stages is not None else (
        "host_prefix", "host_target_frontend", "host_suffix", "gpu_consumer_frontend"))
    host, device = [], []
    host_task_time = 0.0
    for row in rows:
        start, end = float(row.start_ns), float(row.end_ns)
        if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
            raise ValueError("execution time intervals must be finite and ordered")
        if row.metadata.get("orchestration_stage") in host_stages:
            host.append((start, end))
            host_task_time += end - start
        else:
            device.append((start, end))

    def merge(intervals):
        merged = []
        for start, end in sorted(intervals):
            if start == end:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    first = min((float(row.start_ns) for row in rows), default=0.0)
    last = max((float(row.end_ns) for row in rows), default=first)
    origin = first if origin_ns is None else float(origin_ns)
    finish = last if end_ns is None else float(end_ns)
    if (not isfinite(origin) or not isfinite(finish) or origin < 0 or finish < origin
            or (rows and (origin > first or finish < last))):
        raise ValueError("execution time window must contain all task intervals")
    host_union, device_union = merge(host), merge(device)
    host_ns = sum(end - start for start, end in host_union)
    device_ns = sum(end - start for start, end in device_union)
    total_ns = sum(end - start for start, end in merge((*host_union, *device_union)))
    # Intersect the two disjoint lists directly, avoiding subtraction of
    # nearly equal accumulated times for a small overlap.
    overlap = 0.0
    left = right = 0
    while left < len(host_union) and right < len(device_union):
        h_start, h_end = host_union[left]
        d_start, d_end = device_union[right]
        overlap += max(0.0, min(h_end, d_end) - max(h_start, d_start))
        if h_end <= d_end:
            left += 1
        else:
            right += 1
    return {
        "host_orchestration_ns": host_ns,
        "device_execution_ns": device_ns,
        "host_task_time_ns": host_task_time,
        "host_device_overlap_ns": overlap,
        "execution_idle_ns": max(0.0, finish - origin - total_ns),
        "execution_time_semantics": "overlapping_task_interval_unions",
    }


class LiveCohortMetrics:
    """One cohort's realized events, with the closed-schedule reduction rules.

    Records retain absolute timestamps so subtraction and floating-point
    ordering agree with the event kernel. Only the reported cohort span is
    relative to the first observed start (or an explicitly supplied origin).
    Dependencies and resource predecessors outside this cohort contribute
    waiting time to the span, but are not charged as this cohort's own work.
    """

    def __init__(self, *, resource_owners: Optional[Mapping[str, str]] = None,
                 origin_ns: Optional[float] = None):
        if origin_ns is not None and (isinstance(origin_ns, bool)
                or not isfinite(float(origin_ns)) or float(origin_ns) < 0):
            raise ValueError("live cohort origin_ns must be finite and non-negative")
        self.origin_ns = None if origin_ns is None else float(origin_ns)
        self.resource_owners = dict(resource_owners or {})
        self._records = []
        self._names = {}
        self._end_paths = {}
        self._resource_paths = {}
        self._resource_busy = {}
        self._task_bytes = {}
        self._summary_cache = None

    def observe(self, event: KernelEvent) -> None:
        from ._resource_ledger import observe_live_metrics
        observe_live_metrics(self, event, _better_path, _extend_path,
                             TaskExecutionRecord, compact_execution_metadata)

    def summary(self) -> ScheduleCostResult:
        if self._summary_cache is not None:
            return self._summary_cache
        ordered = tuple(sorted(self._records, key=lambda row: (row.start_ns, row.end_ns, row.task_id)))
        categories, coverage = {}, {}
        state_bytes = {"read": 0, "write": 0, "prefetch": 0, "offload": 0}
        byte_count = 0
        for record in ordered:
            duration = record.end_ns - record.start_ns
            metadata = record.metadata
            categories[record.category] = categories.get(record.category, 0.0) + duration
            task_bytes = self._task_bytes[record.task_id]
            byte_count += task_bytes
            event_kind = str(metadata.get("event_kind", ""))
            if event_kind.startswith("linear_state_"):
                kind = event_kind[len("linear_state_"):]
                if kind in state_bytes:
                    state_bytes[kind] += int(metadata.get("bytes", 0))
            component = _coverage_component(record.task_id, metadata)
            if not component:
                continue
            row = coverage.setdefault(component, {
                "task_count": 0.0, "operations": 0.0, "bytes": 0.0,
                "latency_ns": 0.0, "operator_ids": set(), "tensor_ids": set()})
            row["task_count"] += 1.0
            row["operations"] += float(metadata.get("analytical_ops", 0.0))
            row["bytes"] += float(task_bytes)
            row["latency_ns"] += duration
            operator = metadata.get("model_operator_id", metadata.get("operator_id"))
            tensor = metadata.get("weight_tensor_id", metadata.get("tensor_id"))
            if operator:
                row["operator_ids"].add(str(operator))
            if tensor:
                row["tensor_ids"].add(str(tensor))
        last_end = max((record.end_ns for record in ordered), default=0.0)
        best_path = None
        for task_id in sorted(record.task_id for record in ordered if record.end_ns == last_end):
            best_path = _better_path(self._end_paths[task_id], best_path)
        critical = {}
        cursor = best_path
        while cursor is not None:
            if cursor.category is not None:
                critical[cursor.category] = critical.get(cursor.category, 0.0) + cursor.duration_ns
            cursor = cursor.previous
        origin = self.origin_ns if self.origin_ns is not None else min(
            (record.start_ns for record in ordered), default=0.0)
        if ordered and origin > min(record.start_ns for record in ordered):
            raise ValueError("live cohort origin_ns is later than an observed task start")
        self._summary_cache = ScheduleCostResult(
            makespan_ns=last_end - origin if ordered else 0.0,
            task_count=len(ordered), bytes_moved=byte_count,
            resource_busy_ns=dict(sorted(self._resource_busy.items())),
            category_time_ns=categories, critical_path_category_ns=critical,
            coverage={name: {key: sorted(str(item) for item in value) if isinstance(value, set) else value
                            for key, value in row.items()} for name, row in sorted(coverage.items())},
            linear_state_bytes=state_bytes, execution_records=ordered)
        return self._summary_cache

    def metadata(self) -> Mapping[str, object]:
        # Planner imports the cohort execution primitives, so use a lazy
        # import for its existing physical traffic and direction reducers.
        from .planner import (_GPU_CONSUMER_FRONTEND_STAGE, _resource_busy_by_direction,
                              _summarize_dram_task_traffic, _summarize_nand_task_traffic)
        summary = self.summary()
        tasks = tuple(SimpleNamespace(task_id=row.task_id, name=self._names[row.task_id],
                                      metadata=row.metadata, demands=row.demands)
                      for row in summary.execution_records)
        host = tuple(row for row in summary.execution_records
                     if row.metadata.get("orchestration_stage") in {
                         "host_prefix", "host_target_frontend", "host_suffix", _GPU_CONSUMER_FRONTEND_STAGE})
        origin = self.origin_ns if self.origin_ns is not None else min(
            (row.start_ns for row in summary.execution_records), default=0.0)
        time_breakdown = execution_time_breakdown(
            summary.execution_records, origin_ns=origin,
            end_ns=max((row.end_ns for row in summary.execution_records), default=origin))
        return {
            "task_count": summary.task_count,
            "resource_accounted_bytes": summary.bytes_moved,
            "resource_busy_ns": dict(summary.resource_busy_ns),
            "resource_busy_by_direction_ns": _resource_busy_by_direction(tasks),
            "category_time_ns": {key.value: value for key, value in summary.category_time_ns.items()},
            "critical_path_category_ns": {key.value: value for key, value in summary.critical_path_category_ns.items()},
            "coverage": summary.coverage,
            "linear_state_bytes": summary.linear_state_bytes,
            "dram_traffic": _summarize_dram_task_traffic(tasks, resource_owners=self.resource_owners),
            "storage_traffic": _summarize_nand_task_traffic(tasks, resource_owners=self.resource_owners),
            "execution_stage_makespan_ns": summary.makespan_ns,
            **time_breakdown,
            "host_orchestration_task_count": len(host),
            "host_submission_count": sum(int(row.metadata.get("submission_count", 0)) for row in host),
            "physical_execution_scope": "persistent_live_kernel",
            "cost_duration_semantics": "live_cohort_span",
        }


__all__ = ["LiveCohortMetrics", "execution_time_breakdown"]
