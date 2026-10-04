"""Shared physical data-path resource timing.

All interfaces that represent the same physical wire must use one
``resource_id``.  A timeline reserves only that resource, so independent
channels overlap while a shared path serializes deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional

from .memory_types import BatchResult, TransactionResult


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

    def available_ns(self, resource_id: str) -> float:
        return float(self.ready_ns.get(resource_id, 0.0))

    def reserve(self, resource_id: str, earliest_ns: float, duration_ns: float, *, direction: Optional[str] = None, switch_ns: float = 0.0) -> Transfer:
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        earliest_ns = _number("earliest_ns", earliest_ns)
        duration_ns = _number("duration_ns", duration_ns)
        switch_ns = _number("switch_ns", switch_ns)
        start = max(earliest_ns, self.available_ns(resource_id))
        previous = self.directions.get(resource_id)
        if direction and previous and previous != direction:
            start += switch_ns
        end = start + duration_ns
        self.ready_ns[resource_id] = end
        if direction:
            self.directions[resource_id] = direction
        return Transfer(resource_id, 0, 0.0, start, end)

    def transfer(self, resource_id: str, bytes: int, earliest_ns: float, bandwidth_gb_s: float, *, direction: Optional[str] = None, switch_ns: float = 0.0) -> Transfer:
        if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes < 0:
            raise ValueError("bytes must be a non-negative integer")
        bandwidth_gb_s = _number("bandwidth_gb_s", bandwidth_gb_s, positive=True)
        duration = bytes / bandwidth_gb_s
        reservation = self.reserve(resource_id, earliest_ns, duration, direction=direction, switch_ns=switch_ns)
        return Transfer(resource_id, bytes, bandwidth_gb_s, reservation.start_ns, reservation.end_ns)

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
        return BatchResult((), 0.0, 0.0, 0, 0, {"bandwidth_ceiling_gb_s": float(bandwidth_ceiling_gb_s)})
    first = min(item.arrival_ns for item in items)
    last = max(item.completion_ns for item in items)
    counters = {
        "bandwidth_ceiling_gb_s": float(bandwidth_ceiling_gb_s),
        "physical_read_bytes": sum(item.physical_read_bytes for item in items),
        "physical_write_bytes": sum(item.physical_write_bytes for item in items),
        "host_transfer_bytes": sum(item.host_transfer_bytes for item in items),
        "internal_transfer_bytes": sum(item.internal_transfer_bytes for item in items),
        "pages_read": sum(item.pages_read for item in items),
        "pages_programmed": sum(item.pages_programmed for item in items),
        "erase_operations": sum(item.erase_operations for item in items),
        "queue_wait_ns": sum(item.queue_wait_ns for item in items),
    }
    return BatchResult(
        requests=items,
        first_arrival_ns=first,
        last_completion_ns=last,
        logical_bytes=sum(item.logical_bytes for item in items),
        transfer_bytes=sum(item.transfer_bytes for item in items),
        counters=counters,
    )


__all__ = ["ResourceTimeline", "Transfer", "TransferResource", "summarize_batch", "transfer_duration_ns"]
