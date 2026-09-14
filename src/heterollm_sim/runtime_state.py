"""Mutable, validated state for the closed version-4 runtime dispatcher."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Set

from .runtime_ir import RuntimeDelta


def _checked_bytes(values: Mapping[str, int], field_name: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("{} keys must be non-empty strings".format(field_name))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("{} values must be non-negative integers".format(field_name))
        result[key] = value
    return result


def _checked_string_set(
    values: Iterable[object],
    field_name: str,
) -> Set[str]:
    result: Set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("{} values must be non-empty strings".format(field_name))
        result.add(value)
    return result


def _checked_placements(values: Mapping[str, str]) -> Dict[str, str]:
    result = dict(values)
    for key, value in result.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError("placement decisions require non-empty strings")
    return result


def _checked_metrics(values: Mapping[str, float]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("runtime metric ids must be non-empty strings")
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("runtime metrics must be finite numbers")
        result[key] = float(value)
    return result


@dataclass
class RuntimeState:
    """Control-plane state changed only through :class:`RuntimeDelta`.

    Placement decisions are inputs, not an embedded optimizer.  A caller may
    pre-populate ``placement_decisions`` or provide the same closed mapping in
    the run context.
    """

    capacity_bytes: Dict[str, int] = field(default_factory=dict)
    reserved_bytes: Dict[str, int] = field(default_factory=dict)
    placement_decisions: Dict[str, str] = field(default_factory=dict)
    allocations: Dict[str, int] = field(default_factory=dict)
    weight_cache: Set[str] = field(default_factory=set)
    page_cache: Set[str] = field(default_factory=set)
    completed_requests: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capacity_bytes = _checked_bytes(self.capacity_bytes, "capacity_bytes")
        self.reserved_bytes = _checked_bytes(self.reserved_bytes, "reserved_bytes")
        self.allocations = _checked_bytes(self.allocations, "allocations")
        self.placement_decisions = _checked_placements(self.placement_decisions)
        self.weight_cache = _checked_string_set(
            self.weight_cache,
            "weight_cache",
        )
        self.page_cache = _checked_string_set(self.page_cache, "page_cache")
        self.completed_requests = _checked_string_set(
            self.completed_requests,
            "completed_requests",
        )
        self.metrics = _checked_metrics(self.metrics)
        self._validate_capacities(self.reserved_bytes)

    def _validate_capacities(self, reservations: Mapping[str, int]) -> None:
        for resource_id, reserved in reservations.items():
            capacity = self.capacity_bytes.get(resource_id)
            if capacity is not None and reserved > capacity:
                raise ValueError(
                    "capacity exceeded for {}: {} > {}".format(
                        resource_id,
                        reserved,
                        capacity,
                    )
                )

    def available_bytes(self, resource_id: str) -> int | None:
        capacity = self.capacity_bytes.get(resource_id)
        if capacity is None:
            return None
        return capacity - self.reserved_bytes.get(resource_id, 0)

    def release_transient_ledger(
        self,
        capacity_reservations: Mapping[str, int],
        allocations: Mapping[str, int],
    ) -> None:
        """Atomically remove only the ledger increments owned by a prior run."""

        releases = _checked_bytes(
            capacity_reservations, "capacity_reservation_releases"
        )
        allocation_releases = _checked_bytes(
            allocations, "allocation_releases"
        )
        next_reserved = dict(self.reserved_bytes)
        for resource_id, amount in releases.items():
            current = next_reserved.get(resource_id, 0)
            if amount > current:
                raise ValueError(
                    "transient reservation release exceeds current ledger for {}"
                    .format(resource_id)
                )
            remaining = current - amount
            if remaining:
                next_reserved[resource_id] = remaining
            else:
                next_reserved.pop(resource_id, None)

        next_allocations = dict(self.allocations)
        for allocation_id, amount in allocation_releases.items():
            current = next_allocations.get(allocation_id, 0)
            if amount > current:
                raise ValueError(
                    "transient allocation release exceeds current ledger for {}"
                    .format(allocation_id)
                )
            remaining = current - amount
            if remaining:
                next_allocations[allocation_id] = remaining
            else:
                next_allocations.pop(allocation_id, None)

        next_reserved = _checked_bytes(next_reserved, "reserved_bytes")
        next_allocations = _checked_bytes(next_allocations, "allocations")
        self._validate_capacities(next_reserved)
        self.reserved_bytes = next_reserved
        self.allocations = next_allocations

    def apply(self, delta: RuntimeDelta) -> None:
        """Validate the complete delta, then commit it atomically."""

        if not isinstance(delta, RuntimeDelta):
            raise TypeError("delta must be a RuntimeDelta")

        next_reserved = dict(self.reserved_bytes)
        for resource_id, amount in delta.capacity_reservations.items():
            next_reserved[resource_id] = next_reserved.get(resource_id, 0) + amount
        next_reserved = _checked_bytes(next_reserved, "reserved_bytes")
        self._validate_capacities(next_reserved)

        next_allocations = dict(self.allocations)
        for allocation_id, amount in delta.allocations.items():
            next_allocations[allocation_id] = (
                next_allocations.get(allocation_id, 0) + amount
            )
        next_allocations = _checked_bytes(next_allocations, "allocations")

        next_placements = dict(self.placement_decisions)
        next_placements.update(delta.placement_decisions)
        next_placements = _checked_placements(next_placements)

        next_weight_cache = _checked_string_set(
            (*self.weight_cache, *delta.weight_cache_additions),
            "weight_cache",
        )
        next_page_cache = _checked_string_set(
            (*self.page_cache, *delta.page_cache_additions),
            "page_cache",
        )
        next_completed_requests = _checked_string_set(
            (*self.completed_requests, *delta.completed_requests),
            "completed_requests",
        )

        next_metrics = dict(self.metrics)
        for metric_id, amount in delta.metric_increments.items():
            next_value = next_metrics.get(metric_id, 0.0) + float(amount)
            if not math.isfinite(next_value):
                raise ValueError("runtime metric exceeds finite range")
            next_metrics[metric_id] = next_value
        next_metrics = _checked_metrics(next_metrics)

        # No mutation occurs before every derived structure is valid.
        self.reserved_bytes = next_reserved
        self.allocations = next_allocations
        self.placement_decisions = next_placements
        self.weight_cache = next_weight_cache
        self.page_cache = next_page_cache
        self.completed_requests = next_completed_requests
        self.metrics = next_metrics


__all__ = ["RuntimeState"]
