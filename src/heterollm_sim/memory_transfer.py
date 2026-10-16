"""Shared physical data-path resource timing.

All interfaces that represent the same physical wire must use one
``resource_id``.  A timeline reserves only that resource, so independent
channels overlap while a shared path serializes deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Optional

from .memory_types import BatchResult, TransactionResult
from .memory_service import DRAM_OPTIONAL_COUNTER_FIELDS


def _number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return value


@dataclass(frozen=True)
class Transfer:
    resource_id: str
    bytes: int
    bandwidth_gb_s: float
    start_ns: float
    end_ns: float

    @property
    def duration_ns(self) -> float:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class TransferResource:
    """Description of one physical data path used by a timeline."""

    resource_id: str
    bandwidth_gb_s: float
    granularity_bytes: int = 1
    shared: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise ValueError("resource_id must be a non-empty string")
        _number("bandwidth_gb_s", self.bandwidth_gb_s, positive=True)
        if isinstance(self.granularity_bytes, bool) or not isinstance(self.granularity_bytes, int) or self.granularity_bytes <= 0:
            raise ValueError("granularity_bytes must be a positive integer")


@dataclass
class ResourceTimeline:
    """A tiny deterministic reservation calendar keyed by physical resource."""

    ready_ns: Dict[str, float] = field(default_factory=dict)
    directions: Dict[str, str] = field(default_factory=dict)
    busy_ns: Dict[str, float] = field(default_factory=dict)
    bytes_moved: Dict[str, int] = field(default_factory=dict)
    last_intervals: Dict[str, tuple[float, float]] = field(default_factory=dict)
    _intervals: Dict[str, list[tuple[float, float]]] = field(default_factory=dict, repr=False)
    _interval_limit: int = field(default=0, repr=False)
    _interval_count: int = field(default=0, repr=False)
    _intervals_truncated: bool = field(default=False, repr=False)
    _touched: set[str] = field(default_factory=set, repr=False)
    _interval_payloads: Dict[str, list[tuple[float, float, int]]] = field(default_factory=dict, repr=False)
    # Optional lane calendar supplied by the event kernel.  Keeping this as a
    # shared mapping makes ordinary kernel demands and physical reservations
    # consult one authority while preserving the scalar ``ready_ns`` API.
    lane_available: Dict[str, list[float]] = field(default_factory=dict, repr=False)
    lane_directions: Dict[str, list[Optional[str]]] = field(default_factory=dict, repr=False)
    resource_aliases: Mapping[str, str] = field(default_factory=dict, repr=False)
    _metric_key_orders: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def canonical_resource_id(self, resource_id: str) -> str:
        """Resolve a wire name idempotently, rejecting cyclic aliases."""
        if resource_id not in self.resource_aliases:
            return resource_id
        seen = set()
        while resource_id in self.resource_aliases:
            target = self.resource_aliases[resource_id]
            if not isinstance(target, str) or not target:
                raise ValueError("physical resource alias must be a non-empty string")
            if target == resource_id:
                break
            if resource_id in seen:
                raise ValueError("physical resource aliases must not form a cycle")
            seen.add(resource_id)
            resource_id = target
        return resource_id

    def available_ns(self, resource_id: str) -> float:
        resource_id = self.canonical_resource_id(resource_id)
        lanes = self.lane_available.get(resource_id)
        if lanes:
            return float(min(lanes))
        return float(self.ready_ns.get(resource_id, 0.0))

    def reserve(self, resource_id: str, earliest_ns: float, duration_ns: float, *, direction: Optional[str] = None, switch_ns: float = 0.0) -> Transfer:
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        resource_id = self.canonical_resource_id(resource_id)
        earliest_ns = _number("earliest_ns", earliest_ns)
        duration_ns = _number("duration_ns", duration_ns)
        switch_ns = _number("switch_ns", switch_ns)
        lanes = self.lane_available.get(resource_id)
        if lanes is None:
            lanes = [self.available_ns(resource_id)]
        if not lanes:
            raise ValueError("resource must have at least one execution lane")
        # Single-lane accelerators maintain the scalar direction. Multiple
        # execution slots need their own history: another slot's last write
        # must neither erase nor invent this slot's read-to-write penalty.
        previous = self.directions.get(resource_id)
        histories = self.lane_directions.get(resource_id)
        if len(lanes) == 1 or histories is None or len(histories) != len(lanes):
            histories = [previous] * len(lanes)
        if len(lanes) == 1:
            ready = _number("lane readiness", lanes[0])
            if direction and histories[0] and histories[0] != direction:
                ready += switch_ns
            start = max(earliest_ns, ready)
            lane_index = 0
        else:
            def eligible(index):
                ready = _number("lane readiness", lanes[index])
                if direction and histories[index] and histories[index] != direction:
                    ready += switch_ns
                return max(earliest_ns, ready)
            starts = [eligible(index) for index in range(len(lanes))]
            lane_index = min(range(len(lanes)), key=lambda index: (starts[index], lanes[index], index))
            start = starts[lane_index]
        end = start + duration_ns
        busy = self.busy_ns.get(resource_id, 0.0) + duration_ns
        if not math.isfinite(end) or not math.isfinite(busy):
            raise ValueError("resource reservation exceeds finite timing range")
        self.lane_available[resource_id] = lanes
        lanes[lane_index] = end
        self.ready_ns[resource_id] = end if len(lanes) == 1 else min(lanes)
        self.busy_ns[resource_id] = busy
        self.last_intervals[resource_id] = (start, end)
        self._touched.add(resource_id)
        if self._interval_count < self._interval_limit:
            self._intervals.setdefault(resource_id, []).append((start, end))
            self._interval_count += 1
        else:
            self._intervals_truncated = True
        if direction:
            self.directions[resource_id] = direction
            histories[lane_index] = direction
        if len(lanes) > 1:
            self.lane_directions[resource_id] = histories
        else:
            self.lane_directions.pop(resource_id, None)
        return Transfer(resource_id, 0, 0.0, start, end)

    def transfer(self, resource_id: str, bytes: int, earliest_ns: float, bandwidth_gb_s: float, *, direction: Optional[str] = None, switch_ns: float = 0.0) -> Transfer:
        resource_id = self.canonical_resource_id(resource_id)
        if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes < 0:
            raise ValueError("bytes must be a non-negative integer")
        bandwidth_gb_s = _number("bandwidth_gb_s", bandwidth_gb_s, positive=True)
        duration = bytes / bandwidth_gb_s
        reservation = self.reserve(resource_id, earliest_ns, duration, direction=direction, switch_ns=switch_ns)
        self.bytes_moved[resource_id] = self.bytes_moved.get(resource_id, 0) + bytes
        if (resource_id in self._intervals and self._intervals[resource_id]
                and self._intervals[resource_id][-1] == (reservation.start_ns, reservation.end_ns)):
            self._interval_payloads.setdefault(resource_id, []).append(
                (reservation.start_ns, reservation.end_ns, bytes)
            )
        return Transfer(resource_id, bytes, bandwidth_gb_s, reservation.start_ns, reservation.end_ns)

    def metrics_snapshot(self, *, max_intervals: int = 0) -> dict:
        """Begin per-request accounting with bounded real reservation details.

        Pipeline stages are latency annotations and never enter this recorder.
        The detail bound does not change resource calendars or summary counters.
        """
        self._intervals = {}
        self._interval_limit = max_intervals
        self._interval_count = 0
        self._intervals_truncated = False
        self._touched = set()
        self._interval_payloads = {}
        return {
            "busy_ns": dict(self.busy_ns),
            "bytes_moved": dict(self.bytes_moved),
            "last_intervals": dict(self.last_intervals),
            "resource_interval_payloads": {},
        }

    def metrics_delta(self, before: dict) -> dict:
        from ._resource_ledger import timeline_metrics_delta
        return timeline_metrics_delta(self, before)

    def _core_metrics_delta(self, before: dict) -> dict:
        """Reduce reservations made by a Core, whose writers mark touched IDs.

        The public reducer still detects arbitrary dictionary edits. Core
        execution has no caller callback between snapshot and reduction, so
        unrelated resources cannot change and need no counter lookups.
        """
        if (type(self) is not ResourceTimeline or type(self.busy_ns) is not dict
                or type(self.bytes_moved) is not dict):
            return self.metrics_delta(before)
        touched = self._touched
        if len(touched) * 2 >= len(self.busy_ns):
            return self.metrics_delta(before)
        from ._resource_ledger import core_timeline_metrics_delta
        return core_timeline_metrics_delta(self, before)

    def _ordered_metric_touches(self, name, values, touched):
        # Constructing the key tuple runs in C. Comparing it on every call
        # also observes public insertions, deletions and same-size reordering;
        # cached ordinals never hide a caller's changed dictionary order.
        keys = tuple(values)
        cached = self._metric_key_orders.get(name)
        if cached is None or cached[0] != keys:
            order = {key: index for index, key in enumerate(keys)}
            self._metric_key_orders[name] = (keys, order)
        else:
            order = cached[1]
        return sorted((rid for rid in touched if rid in order), key=order.__getitem__)

    def snapshot(self) -> dict[str, float]:
        return dict(self.ready_ns)


def transfer_duration_ns(bytes: int, bandwidth_gb_s: float) -> float:
    """Return decimal-GB/s transfer duration in ns."""
    if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes < 0:
        raise ValueError("bytes must be a non-negative integer")
    return bytes / _number("bandwidth_gb_s", bandwidth_gb_s, positive=True)


def summarize_batch(results, *, bandwidth_ceiling_gb_s: float = 0.0) -> BatchResult:
    """Compute public batch latency and achieved bandwidth once.

    ``logical_bytes`` is intentionally used for achieved bandwidth; physical
    page/burst traffic remains available through each transaction's counters.
    """
    items = tuple(results)
    if not all(isinstance(item, TransactionResult) for item in items):
        raise TypeError("results must contain TransactionResult values")
    if not items:
        return BatchResult((), 0.0, 0.0, 0, 0, {
            "bandwidth_ceiling_gb_s": float(bandwidth_ceiling_gb_s),
            "internal_transfer_bytes": 0,
        })
    first = min(item.arrival_ns for item in items)
    last = max(item.completion_ns for item in items)
    counters = {
        "bandwidth_ceiling_gb_s": float(bandwidth_ceiling_gb_s),
        "read_bandwidth_ceiling_gb_s": max(
            (item.bandwidth_ceiling_gb_s for item in items if item.operation.value == "read"),
            default=0.0,
        ),
        "write_bandwidth_ceiling_gb_s": max(
            (item.bandwidth_ceiling_gb_s for item in items if item.operation.value == "write"),
            default=0.0,
        ),
        "physical_read_bytes": sum(item.physical_read_bytes for item in items),
        "physical_write_bytes": sum(item.physical_write_bytes for item in items),
        "host_transfer_bytes": sum(item.host_transfer_bytes for item in items),
        "internal_transfer_bytes": sum(item.internal_transfer_bytes for item in items),
        "pages_read": sum(item.pages_read for item in items),
        "pages_programmed": sum(item.pages_programmed for item in items),
        "erase_operations": sum(item.erase_operations for item in items),
        "read_cache_hits": sum(item.counters.get("read_cache_hits", 0) for item in items),
        "read_cache_misses": sum(item.counters.get("read_cache_misses", 0) for item in items),
        "queue_wait_ns": sum(item.queue_wait_ns for item in items),
        "write_buffer_wait_ns": sum(item.write_buffer_wait_ns for item in items),
        "write_buffer_bytes": sum(item.write_buffer_bytes for item in items),
        "write_buffer_peak_bytes": max(
            (item.write_buffer_occupancy_bytes for item in items), default=0
        ),
    }
    for key in DRAM_OPTIONAL_COUNTER_FIELDS:
        values = [item.counters.get(key) for item in items]
        counters[key] = (sum(values) if all(isinstance(value, (int, float))
                        and not isinstance(value, bool) for value in values) else None)
    return BatchResult(
        requests=items,
        first_arrival_ns=first,
        last_completion_ns=last,
        logical_bytes=sum(item.logical_bytes for item in items),
        transfer_bytes=sum(item.transfer_bytes for item in items),
        counters=counters,
    )


__all__ = ["ResourceTimeline", "Transfer", "TransferResource", "summarize_batch", "transfer_duration_ns"]
