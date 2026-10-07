"""Reduce actual persistent-kernel events without retaining physical traces."""
from __future__ import annotations

from math import isfinite
from types import SimpleNamespace
from typing import Mapping, Optional

from .contracts import TaskCategory
from .event_kernel import KernelEvent
from .scalable_serving import (
    ScheduleCostResult, TaskExecutionRecord, _better_path, _coverage_component,
    _extend_path, compact_execution_metadata,
)


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
        self._summary_cache = None

    def observe(self, event: KernelEvent) -> None:
        task = event.task
        if task.task_id in self._names:
            raise ValueError("live cohort observed a duplicate task: {}".format(task.task_id))
        start_path = None
        for dependency_id in sorted(task.dependencies):
            if dependency_id in self._end_paths:
                start_path = _better_path(self._end_paths[dependency_id], start_path)
        for demand in event.demands:
            lane = int(event.resource_lanes.get(demand.resource_id, 0))
            info = event.resource_predecessors.get(demand.resource_id)
            if info is None:
                continue
            resource_id = str(info.get("resource_id", demand.resource_id))
            previous = self._resource_paths.get((resource_id, lane))
            # Other cohorts can interleave on the same lane. An older task
            # from this cohort must not replace the actual external predecessor.
            if previous is not None and str(info.get("task_id", "")) == previous[0]:
                start_path = _better_path(previous[1], start_path)
        duration = event.end_ns - event.start_ns
        self._end_paths[task.task_id] = _extend_path(
            start_path, duration, task.category, task.task_id)
        for demand in event.demands:
            lane = int(event.resource_lanes.get(demand.resource_id, 0))
            interval_end = event.start_ns + demand.service_ns
            path = _extend_path(start_path, interval_end - event.start_ns,
                                task.category, "{}@{}".format(task.task_id, demand.resource_id))
            self._resource_paths[(demand.resource_id, lane)] = (task.task_id, path)
            # Physical demands already contain core interval busy time. The
            # event kernel sums this field exactly once, also for zero work.
            self._resource_busy[demand.resource_id] = (
                self._resource_busy.get(demand.resource_id, 0.0) + demand.service_ns)
        self._records.append(TaskExecutionRecord(
            task_id=task.task_id, start_ns=event.start_ns, end_ns=event.end_ns,
            category=task.category, dependencies=tuple(task.dependencies),
            metadata=compact_execution_metadata(task.metadata), demands=event.demands))
        self._names[task.task_id] = task.name
        self._summary_cache = None

    def summary(self) -> ScheduleCostResult:
        if self._summary_cache is not None:
            return self._summary_cache
        ordered = tuple(sorted(self._records, key=lambda row: (row.start_ns, row.end_ns, row.task_id)))
        categories, coverage = {}, {}
        state_bytes = {"read": 0, "write": 0, "prefetch": 0, "offload": 0}
        energy, byte_count = 0.0, 0
        for record in ordered:
            duration = record.end_ns - record.start_ns
            metadata = record.metadata
            categories[record.category] = categories.get(record.category, 0.0) + duration
            task_energy = sum(demand.energy_pj for demand in record.demands)
            task_bytes = sum(demand.bytes_moved for demand in record.demands)
            energy += task_energy
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
                "latency_ns": 0.0, "energy_pj": 0.0,
                "operator_ids": set(), "tensor_ids": set()})
            row["task_count"] += 1.0
            row["operations"] += float(metadata.get("analytical_ops", 0.0))
            row["bytes"] += float(task_bytes)
            row["latency_ns"] += duration
            row["energy_pj"] += float(task_energy)
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
            task_count=len(ordered), energy_pj=energy, bytes_moved=byte_count,
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
        host_ns = sum(row.end_ns - row.start_ns for row in host)
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
            "host_orchestration_ns": host_ns,
            "host_orchestration_task_count": len(host),
            "host_submission_count": sum(int(row.metadata.get("submission_count", 0)) for row in host),
            "device_execution_ns": max(0.0, summary.makespan_ns - host_ns),
            "physical_execution_scope": "persistent_live_kernel",
            "cost_duration_semantics": "live_cohort_span",
        }


__all__ = ["LiveCohortMetrics"]
