"""Owner-aware physical allocation residency.

The serving runtime has several logical users of the same memory component:
model weights, KV cache, recurrent state, and short-lived workspaces.  This
module gives those users one capacity authority without embedding any serving
policy, calibration data, or hardware-specific assumptions.

An :class:`Allocation` is the sole capacity-owning object.  An
:class:`AllocationView` only maps a byte range onto an allocation and therefore
never consumes capacity itself.  All access through a view is charged to its
physical owner while refreshing only the resident granules covered by that
view access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union


class ResidencyError(RuntimeError):
    """Base class for residency failures."""


class ResidencyValidationError(ResidencyError, ValueError):
    """Raised when an allocation, alias, or access contract is invalid."""


class CapacityExceededError(ResidencyError):
    """Raised when capacity cannot be made available without unsafe eviction."""


class AllocationNotFoundError(ResidencyError, KeyError):
    """Raised when an allocation or view identifier is unknown."""


class AllocationLifecycle(str, Enum):
    """Lifecycle classification used by callers for cleanup policy."""

    PERSISTENT = "persistent"
    TEMPORARY = "temporary"
    SYSTEM = "system"
    RELEASED = "released"


class AccessOperation(str, Enum):
    TOUCH = "touch"
    READ = "read"
    WRITE = "write"


class MigrationKind(str, Enum):
    PAGE_IN = "page_in"
    CLEAN_DISCARD = "clean_discard"
    DIRTY_WRITEBACK = "dirty_writeback"


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ResidencyValidationError("{} must be an integer".format(name))
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidencyValidationError("{} must be an integer".format(name)) from exc
    if result != value or result < minimum:
        raise ResidencyValidationError(
            "{} must be an integer >= {}".format(name, minimum)
        )
    return result


@dataclass
class PhysicalPool:
    """One physical component and its authoritative resident-byte total."""

    component_id: str
    capacity_bytes: int
    page_size_bytes: int = 1
    resident_bytes: int = field(default=0, init=False)
    peak_resident_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.component_id = str(self.component_id).strip()
        if not self.component_id:
            raise ResidencyValidationError("component_id must not be empty")
        self.capacity_bytes = _integer("capacity_bytes", self.capacity_bytes)
        self.page_size_bytes = _integer(
            "page_size_bytes", self.page_size_bytes, minimum=1
        )

    @property
    def available_bytes(self) -> int:
        return self.capacity_bytes - self.resident_bytes


_Range = Tuple[int, int]
_RecencyRange = Tuple[int, int, int]


def _range_bytes(ranges: Sequence[_Range]) -> int:
    return sum(end - start for start, end in ranges)


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return (value + divisor - 1) // divisor


def _is_backing_unit_boundary(offset: int, committed: int, granule: int) -> bool:
    return 0 <= offset <= committed and (
        offset == committed or offset % granule == 0
    )


def _ensure_backing_unit_ranges(
    label: str,
    ranges: Iterable[_Range],
    committed: int,
    granule: int,
) -> None:
    committed = _integer("committed_bytes", committed)
    granule = _integer("residency_granule_bytes", granule, minimum=1)
    for start, end in ranges:
        if end <= start:
            continue
        if not (
            _is_backing_unit_boundary(start, committed, granule)
            and _is_backing_unit_boundary(end, committed, granule)
        ):
            raise ResidencyError(
                "{} cuts through allocation backing granule".format(label)
            )


def _prefix_backing_unit_ranges(
    committed: int, granule: int, byte_count: int
) -> list[_Range]:
    if byte_count <= 0:
        return []
    return [(0, min(committed, _ceil_div(byte_count, granule) * granule))]


def _backing_unit_prefix_for_bytes(
    start: int,
    end: int,
    granule: int,
    byte_count: int,
) -> Optional[_Range]:
    """Select the shortest whole-unit prefix satisfying ``byte_count``.

    ``start`` and ``end`` must already be validated backing-unit boundaries.
    In particular, ``end`` may be the allocation's short final backing unit.
    """

    if byte_count <= 0 or end <= start:
        return None
    selection_end = min(end, start + _ceil_div(byte_count, granule) * granule)
    return start, selection_end


def _iter_backing_unit_ranges(
    committed: int,
    granule: int,
    ranges: Sequence[_Range],
) -> Iterable[_Range]:
    _ensure_backing_unit_ranges("backing unit range", ranges, committed, granule)
    for start, end in _merge_ranges(ranges):
        cursor = start
        while cursor < end:
            unit_end = min(committed, cursor + granule)
            if unit_end <= cursor or unit_end > end:
                raise ResidencyError("backing unit iteration escaped selected range")
            yield cursor, unit_end
            cursor = unit_end


def _merge_ranges(ranges: Iterable[_Range]) -> list[_Range]:
    ordered = sorted((int(start), int(end)) for start, end in ranges if end > start)
    merged: list[_Range] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _subtract_ranges(base: Sequence[_Range], cuts: Sequence[_Range]) -> list[_Range]:
    result = list(base)
    for cut_start, cut_end in _merge_ranges(cuts):
        updated: list[_Range] = []
        for start, end in result:
            if cut_end <= start or cut_start >= end:
                updated.append((start, end))
                continue
            if start < cut_start:
                updated.append((start, cut_start))
            if cut_end < end:
                updated.append((cut_end, end))
        result = updated
    return result


def _intersect_ranges(left: Sequence[_Range], right: Sequence[_Range]) -> list[_Range]:
    result: list[_Range] = []
    left_ranges = _merge_ranges(left)
    right_ranges = _merge_ranges(right)
    left_index = right_index = 0
    while left_index < len(left_ranges) and right_index < len(right_ranges):
        left_start, left_end = left_ranges[left_index]
        right_start, right_end = right_ranges[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if start < end:
            result.append((start, end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return result


def _merge_recency_ranges(
    ranges: Iterable[_RecencyRange],
) -> list[_RecencyRange]:
    """Merge adjacent address ranges only when their recency is identical."""

    ordered = sorted(
        (int(start), int(end), int(sequence))
        for start, end, sequence in ranges
        if end > start
    )
    merged: list[_RecencyRange] = []
    for start, end, sequence in ordered:
        if merged and start < merged[-1][1]:
            raise ResidencyError("overlapping allocation recency ranges")
        if merged and start == merged[-1][1] and sequence == merged[-1][2]:
            merged[-1] = (merged[-1][0], end, sequence)
            continue
        merged.append((start, end, sequence))
    return merged


def _subtract_recency_ranges(
    base: Sequence[_RecencyRange], cuts: Sequence[_Range]
) -> list[_RecencyRange]:
    """Remove byte ranges while retaining recency on surviving fragments."""

    result = list(base)
    for cut_start, cut_end in _merge_ranges(cuts):
        updated: list[_RecencyRange] = []
        for start, end, sequence in result:
            if cut_end <= start or cut_start >= end:
                updated.append((start, end, sequence))
                continue
            if start < cut_start:
                updated.append((start, cut_start, sequence))
            if cut_end < end:
                updated.append((cut_end, end, sequence))
        result = updated
    return _merge_recency_ranges(result)


def _overwrite_consumer_range(
    anchors: list[Tuple[int, int, str]],
    start: int,
    end: int,
    consumer_task_id: str,
) -> None:
    """Overwrite one range in a sorted, non-overlapping consumer map.

    Released residency leases form an interval map: the most recently
    released consumer for a byte range is the dependency of its next
    eviction.  The generic implementation rebuilt every existing interval
    for every released lease.  A serving invocation releases thousands of
    tensor-view leases, so that otherwise turns a bounded (roughly 122-range)
    owner map into millions of Python-level range operations.

    ``anchors`` is kept in exactly the same ``(start, end, task)`` order as
    the former subtract-all-then-sort implementation.  Only intervals that
    overlap the replacement are inspected; list splicing handles the
    unaffected prefix and suffix in C.
    """

    if end <= start:
        return

    # Non-overlapping intervals sorted by start also have monotonic ends.
    # Find the first interval whose end is strictly beyond ``start`` so an
    # interval ending exactly at the replacement boundary remains untouched.
    low = 0
    high = len(anchors)
    while low < high:
        middle = (low + high) // 2
        if anchors[middle][1] <= start:
            low = middle + 1
        else:
            high = middle
    first = low

    last = first
    left_fragment: Optional[Tuple[int, int, str]] = None
    right_fragment: Optional[Tuple[int, int, str]] = None
    while last < len(anchors) and anchors[last][0] < end:
        anchor_start, anchor_end, task_id = anchors[last]
        if anchor_start < start:
            left_fragment = (anchor_start, start, task_id)
        if anchor_end > end:
            right_fragment = (end, anchor_end, task_id)
        last += 1

    replacement: list[Tuple[int, int, str]] = []
    if left_fragment is not None:
        replacement.append(left_fragment)
    replacement.append((start, end, consumer_task_id))
    if right_fragment is not None:
        replacement.append(right_fragment)
    anchors[first:last] = replacement


@dataclass
class Allocation:
    """The unique physical owner of a logical allocation.

    ``committed_bytes`` is logical address space promised to the allocation;
    only ``resident_bytes`` consumes the pool.  Resident and dirty extents are
    maintained by :class:`AllocationResidencyManager`.
    """

    allocation_id: str
    physical_owner: str
    size_bytes: int
    backing: Optional[str]
    kind: str
    committed_bytes: int
    evictable: bool
    pinned: bool
    lifecycle: AllocationLifecycle
    read_only: bool
    last_access: int
    residency_granule_bytes: int = 1
    resident_bytes: int = field(default=0, init=False)
    dirty: bool = field(default=False, init=False)
    _registration_order: int = field(default=0, repr=False)
    _resident_ranges: list[_Range] = field(default_factory=list, repr=False)
    _dirty_ranges: list[_Range] = field(default_factory=list, repr=False)
    _resident_recency_ranges: list[_RecencyRange] = field(
        default_factory=list, repr=False
    )

    @property
    def owner_id(self) -> str:
        return self.allocation_id

    @property
    def home(self) -> Optional[str]:
        return self.backing

    @property
    def dirty_bytes(self) -> int:
        return _range_bytes(self._dirty_ranges)

    @property
    def resident_ranges(self) -> Tuple[_Range, ...]:
        return tuple(self._resident_ranges)

    @property
    def dirty_ranges(self) -> Tuple[_Range, ...]:
        return tuple(self._dirty_ranges)

    @property
    def resident_recency_ranges(self) -> Tuple[_RecencyRange, ...]:
        """Resident address extents as ``(start, end, access_sequence)``."""

        return tuple(self._resident_recency_ranges)

    def _sync_derived_state(self) -> None:
        self._resident_ranges = _merge_ranges(self._resident_ranges)
        self._dirty_ranges = _intersect_ranges(
            _merge_ranges(self._dirty_ranges), self._resident_ranges
        )
        self.resident_bytes = _range_bytes(self._resident_ranges)
        self.dirty = bool(self._dirty_ranges)


@dataclass(frozen=True)
class AllocationView:
    """A zero-capacity byte-range alias of exactly one Allocation."""

    view_id: str
    owner_id: str
    offset_bytes: int
    size_bytes: int
    residency_granule_bytes: int = 1

    @property
    def capacity_bytes(self) -> int:
        return 0

    @property
    def resident_bytes(self) -> int:
        return 0

    @property
    def committed_bytes(self) -> int:
        return 0

    @property
    def physical_owner(self) -> str:
        return self.owner_id


@dataclass(frozen=True)
class MemoryAccess:
    """A byte-range access processed in caller-provided sequence order."""

    target_id: str
    operation: Union[AccessOperation, str] = AccessOperation.TOUCH
    offset_bytes: int = 0
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class Migration:
    """One classified movement caused by residency management."""

    sequence: int
    allocation_id: str
    kind: MigrationKind
    byte_count: int
    source: Optional[str]
    destination: Optional[str]
    residency_granule_bytes: int = 1
    granule_count: int = 0
    consumer_task_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            kind = MigrationKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ResidencyValidationError(
                "unsupported migration kind: {}".format(self.kind)
            ) from exc
        byte_count = _integer("byte_count", self.byte_count)
        granule = _integer(
            "residency_granule_bytes",
            self.residency_granule_bytes,
            minimum=1,
        )
        expected_count = _ceil_div(byte_count, granule)
        if isinstance(self.granule_count, bool):
            raise ResidencyValidationError("granule_count must be an integer")
        if self.granule_count == 0:
            granule_count = expected_count
        else:
            granule_count = _integer("granule_count", self.granule_count, minimum=1)
            if granule_count != expected_count:
                raise ResidencyValidationError(
                    "granule_count must equal ceil(byte_count / residency_granule_bytes)"
                )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "residency_granule_bytes", granule)
        object.__setattr__(self, "granule_count", granule_count)
        object.__setattr__(
            self,
            "consumer_task_ids",
            tuple(dict.fromkeys(str(item) for item in self.consumer_task_ids if item)),
        )

    @property
    def page_in_bytes(self) -> int:
        return self.byte_count if self.kind is MigrationKind.PAGE_IN else 0

    @property
    def page_out_bytes(self) -> int:
        """Bytes removed from the resident pool, including clean discard."""

        return self.byte_count if self.kind is not MigrationKind.PAGE_IN else 0

    @property
    def clean_discard_bytes(self) -> int:
        return self.byte_count if self.kind is MigrationKind.CLEAN_DISCARD else 0

    @property
    def dirty_writeback_bytes(self) -> int:
        return self.byte_count if self.kind is MigrationKind.DIRTY_WRITEBACK else 0


@dataclass(frozen=True)
class MigrationTotals:
    page_in_bytes: int = 0
    page_out_bytes: int = 0
    clean_discard_bytes: int = 0
    dirty_writeback_bytes: int = 0


@dataclass(frozen=True)
class AccessResult:
    sequence: int
    target_id: str
    owner_id: str
    operation: AccessOperation
    offset_bytes: int
    size_bytes: int
    page_in_bytes: int
    migrations: Tuple[Migration, ...]
    physical_residency_bytes: int = 0
    residency_ranges: Tuple[_Range, ...] = ()


@dataclass(frozen=True)
class ResidencySnapshot:
    component_id: str
    capacity_bytes: int
    resident_bytes: int
    peak_resident_bytes: int
    available_bytes: int
    committed_bytes: int
    allocation_count: int
    view_count: int
    migrations: MigrationTotals


@dataclass(frozen=True)
class ResidencyLease:
    """A resident owner range protected until its consumer task completes."""

    lease_id: str
    consumer_task_id: str
    target_id: str
    owner_id: str
    offset_bytes: int
    size_bytes: int
    residency_ranges: Tuple[_Range, ...]


class AllocationResidencyManager:
    """Own allocations and arbitrate a single physical pool.

    Capacity acquisition is fail-closed: the manager first proves that enough
    eligible bytes can be evicted and only then mutates residency.  Eviction
    uses one deterministic LRU order across all allocation kinds.
    """

    def __init__(
        self,
        pool: Union[PhysicalPool, str],
        capacity_bytes: Optional[int] = None,
        *,
        page_size_bytes: int = 1,
    ) -> None:
        if isinstance(pool, PhysicalPool):
            if capacity_bytes is not None:
                raise ResidencyValidationError(
                    "capacity_bytes must be omitted when pool is a PhysicalPool"
                )
            if pool.resident_bytes != 0:
                raise ResidencyValidationError(
                    "a PhysicalPool must be unused when attached to a manager"
                )
            self.pool = pool
        else:
            if capacity_bytes is None:
                raise ResidencyValidationError(
                    "capacity_bytes is required when pool is a component id"
                )
            self.pool = PhysicalPool(
                str(pool), capacity_bytes, page_size_bytes=page_size_bytes
            )
        self._allocations: dict[str, Allocation] = {}
        self._views: dict[str, AllocationView] = {}
        self._access_sequence = 0
        self._registration_sequence = 0
        self._migration_sequence = 0
        self._migrations: list[Migration] = []
        # Snapshots are taken once per serving batch.  Re-summing the complete
        # append-only migration history in every snapshot makes a long replay
        # quadratic in the number of residency events, even though the four
        # counters are additive.  Keep the auditable event history and update
        # its aggregate once, at the mutation boundary that creates an event.
        self._migration_totals = MigrationTotals()
        self._leases: dict[str, ResidencyLease] = {}
        # Released leases remain causal anchors for the next eviction of the
        # covered bytes.  The residency planner can therefore release a lease
        # after planning its consumer and still make the later PAGE_OUT task
        # depend on that real consumer.
        self._released_consumer_ranges: dict[
            str, list[Tuple[int, int, str]]
        ] = {}
        self._deferred_validation_depth = 0

    def _validate_mutation_boundary(self) -> None:
        if self._deferred_validation_depth == 0:
            self.assert_consistent()

    def begin_deferred_validation(self) -> None:
        """Defer full registry scans until a caller-owned mutation batch ends."""

        if self._deferred_validation_depth == 0:
            self.assert_consistent()
        self._deferred_validation_depth += 1

    def end_deferred_validation(self) -> None:
        """Close one mutation batch and validate once at its outer boundary."""

        if self._deferred_validation_depth <= 0:
            raise ResidencyError("deferred validation is not active")
        self._deferred_validation_depth -= 1
        if self._deferred_validation_depth == 0:
            self.assert_consistent()

    @property
    def allocations(self) -> Mapping[str, Allocation]:
        return MappingProxyType(self._allocations)

    @property
    def views(self) -> Mapping[str, AllocationView]:
        return MappingProxyType(self._views)

    @property
    def migrations(self) -> Tuple[Migration, ...]:
        return tuple(self._migrations)

    @property
    def migration_count(self) -> int:
        """Return the append-only migration cursor without copying history."""

        return len(self._migrations)

    def migrations_since(self, cursor: int) -> Tuple[Migration, ...]:
        """Return the append-only history suffix without copying its prefix."""

        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise TypeError("migration cursor must be an integer")
        if cursor < 0 or cursor > len(self._migrations):
            raise ValueError("migration cursor is outside the retained history")
        return tuple(self._migrations[cursor:])

    @property
    def leases(self) -> Mapping[str, ResidencyLease]:
        return MappingProxyType(self._leases)

    @property
    def resident_bytes(self) -> int:
        return self.pool.resident_bytes

    @property
    def committed_bytes(self) -> int:
        return sum(owner.committed_bytes for owner in self._allocations.values())

    @property
    def available_bytes(self) -> int:
        return self.pool.available_bytes

    @property
    def migration_totals(self) -> MigrationTotals:
        return self._migration_totals

    def snapshot(self) -> ResidencySnapshot:
        return ResidencySnapshot(
            component_id=self.pool.component_id,
            capacity_bytes=self.pool.capacity_bytes,
            resident_bytes=self.pool.resident_bytes,
            peak_resident_bytes=self.pool.peak_resident_bytes,
            available_bytes=self.pool.available_bytes,
            committed_bytes=self.committed_bytes,
            allocation_count=len(self._allocations),
            view_count=len(self._views),
            migrations=self.migration_totals,
        )

    def _next_access_sequence(self) -> int:
        self._access_sequence += 1
        return self._access_sequence

    @staticmethod
    def _refresh_recency(
        owner: Allocation,
        ranges: Sequence[_Range],
        sequence: int,
    ) -> None:
        """Assign one access sequence to every resident byte in ``ranges``."""

        normalized = _merge_ranges(ranges)
        if not normalized:
            return
        _ensure_backing_unit_ranges(
            "allocation recency refresh",
            normalized,
            owner.committed_bytes,
            owner.residency_granule_bytes,
        )
        resident = _intersect_ranges(normalized, owner._resident_ranges)
        if resident != normalized:
            raise ResidencyError(
                "cannot refresh non-resident bytes for {}".format(
                    owner.allocation_id
                )
            )
        if len(normalized) == 1:
            # ``access`` always refreshes one contiguous, granule-aligned
            # physical extent with a new sequence.  Both the resident ranges
            # and their recency partition are already normalized at mutation
            # boundaries, so replacing that extent is an ordered splice.  The
            # generic subtract+merge path below sorts and rebuilds the whole
            # partition twice; that became quadratic-looking work when one
            # backend weight allocation contained many tensor views.
            refresh_start, refresh_end = normalized[0]
            refreshed: list[_RecencyRange] = []
            inserted = False
            replacement_may_merge = False
            for start, end, previous_sequence in owner._resident_recency_ranges:
                if previous_sequence == sequence:
                    replacement_may_merge = True
                if end <= refresh_start:
                    refreshed.append((start, end, previous_sequence))
                    continue
                if start >= refresh_end:
                    if not inserted:
                        refreshed.append(
                            (refresh_start, refresh_end, sequence)
                        )
                        inserted = True
                    refreshed.append((start, end, previous_sequence))
                    continue
                if start < refresh_start:
                    refreshed.append(
                        (start, refresh_start, previous_sequence)
                    )
                if not inserted:
                    refreshed.append((refresh_start, refresh_end, sequence))
                    inserted = True
                if end > refresh_end:
                    refreshed.append((refresh_end, end, previous_sequence))
            if not inserted:
                refreshed.append((refresh_start, refresh_end, sequence))
            owner._resident_recency_ranges = (
                _merge_recency_ranges(refreshed)
                if replacement_may_merge
                else refreshed
            )
            return
        retained = _subtract_recency_ranges(
            owner._resident_recency_ranges, normalized
        )
        owner._resident_recency_ranges = _merge_recency_ranges(
            [
                *retained,
                *((start, end, sequence) for start, end in normalized),
            ]
        )

    def _adjust_pool(self, delta_bytes: int) -> None:
        updated = self.pool.resident_bytes + int(delta_bytes)
        if updated < 0:
            raise ResidencyError(
                "physical pool underflow on {}".format(self.pool.component_id)
            )
        if updated > self.pool.capacity_bytes:
            raise CapacityExceededError(
                "physical pool {} capacity exceeded: {} > {}".format(
                    self.pool.component_id, updated, self.pool.capacity_bytes
                )
            )
        self.pool.resident_bytes = updated
        self.pool.peak_resident_bytes = max(
            self.pool.peak_resident_bytes, updated
        )

    def _record_migration(
        self,
        owner: Allocation,
        kind: MigrationKind,
        byte_count: int,
        source: Optional[str],
        destination: Optional[str],
        consumer_task_ids: Sequence[str] = (),
    ) -> Migration:
        if byte_count <= 0:
            raise ResidencyError("cannot record a zero-byte migration")
        self._migration_sequence += 1
        migration = Migration(
            sequence=self._migration_sequence,
            allocation_id=owner.allocation_id,
            kind=kind,
            byte_count=byte_count,
            source=source,
            destination=destination,
            residency_granule_bytes=owner.residency_granule_bytes,
            consumer_task_ids=tuple(consumer_task_ids),
        )
        self._migrations.append(migration)
        totals = self._migration_totals
        self._migration_totals = MigrationTotals(
            page_in_bytes=totals.page_in_bytes + migration.page_in_bytes,
            page_out_bytes=totals.page_out_bytes + migration.page_out_bytes,
            clean_discard_bytes=(
                totals.clean_discard_bytes + migration.clean_discard_bytes
            ),
            dirty_writeback_bytes=(
                totals.dirty_writeback_bytes + migration.dirty_writeback_bytes
            ),
        )
        return migration

    def register(
        self,
        allocation_id: str,
        size_bytes: int,
        *,
        kind: str = "generic",
        backing: Optional[str] = None,
        home: Optional[str] = None,
        committed_bytes: Optional[int] = None,
        resident_bytes: int = 0,
        dirty: bool = False,
        evictable: Optional[bool] = None,
        pinned: bool = False,
        lifecycle: Union[AllocationLifecycle, str] = AllocationLifecycle.PERSISTENT,
        read_only: Optional[bool] = None,
        residency_granule_bytes: Optional[int] = None,
    ) -> Allocation:
        """Register one unique capacity owner.

        ``resident_bytes`` initially covers at least that much allocation
        prefix, rounded out to complete backing granules.  Later accesses may
        make arbitrary granule-aligned ranges resident.
        """

        owner_id = str(allocation_id).strip()
        if not owner_id:
            raise ResidencyValidationError("allocation_id must not be empty")
        if owner_id in self._allocations or owner_id in self._views:
            raise ResidencyValidationError(
                "allocation/view id already registered: {}".format(owner_id)
            )
        size = _integer("size_bytes", size_bytes, minimum=1)
        committed = size if committed_bytes is None else _integer(
            "committed_bytes", committed_bytes
        )
        resident = _integer("resident_bytes", resident_bytes)
        if committed > size:
            raise ResidencyValidationError("committed_bytes exceeds size_bytes")
        if resident > committed:
            raise ResidencyValidationError("resident_bytes exceeds committed_bytes")
        granule = (
            self.pool.page_size_bytes
            if residency_granule_bytes is None
            else _integer(
                "residency_granule_bytes",
                residency_granule_bytes,
                minimum=1,
            )
        )
        if backing is not None and home is not None and str(backing) != str(home):
            raise ResidencyValidationError("backing and home identify different components")
        resolved_backing = backing if backing is not None else home
        if resolved_backing is not None:
            resolved_backing = str(resolved_backing).strip()
            if not resolved_backing:
                raise ResidencyValidationError("backing must not be empty")
        try:
            resolved_lifecycle = AllocationLifecycle(lifecycle)
        except ValueError as exc:
            raise ResidencyValidationError(
                "unsupported allocation lifecycle: {}".format(lifecycle)
            ) from exc
        if resolved_lifecycle is AllocationLifecycle.RELEASED:
            raise ResidencyValidationError("cannot register a released allocation")
        if pinned and resolved_lifecycle is AllocationLifecycle.TEMPORARY:
            raise ResidencyValidationError("temporary allocations cannot be pinned")
        if evictable is None:
            resolved_evictable = bool(resolved_backing) and not pinned
        else:
            resolved_evictable = bool(evictable)
        if pinned and resolved_evictable:
            raise ResidencyValidationError("pinned allocations cannot be evictable")
        if resolved_evictable and not resolved_backing:
            raise ResidencyValidationError(
                "evictable allocations require a backing component"
            )
        if resolved_evictable and resolved_backing == self.pool.component_id:
            raise ResidencyValidationError(
                "an evictable allocation backing must differ from its physical pool"
            )
        normalized_kind = str(kind).strip().lower()
        if not normalized_kind:
            raise ResidencyValidationError("kind must not be empty")
        inferred_read_only = normalized_kind in {
            "weight",
            "weights",
            "model_weight",
            "model_weights",
        }
        resolved_read_only = inferred_read_only if read_only is None else bool(read_only)
        if dirty and resident == 0:
            raise ResidencyValidationError("an allocation with no resident bytes cannot be dirty")
        if dirty and resolved_read_only:
            raise ResidencyValidationError("read-only allocations cannot start dirty")
        if dirty and resolved_evictable and not resolved_backing:
            raise ResidencyValidationError("dirty evictable bytes require backing")

        initial_resident_ranges = _prefix_backing_unit_ranges(
            committed, granule, resident
        )
        initial_resident = _range_bytes(initial_resident_ranges)
        self._validate_mutation_boundary()
        additional = initial_resident
        self._make_room(additional)
        self._registration_sequence += 1
        access_sequence = self._next_access_sequence()
        owner = Allocation(
            allocation_id=owner_id,
            physical_owner=self.pool.component_id,
            size_bytes=size,
            backing=resolved_backing,
            kind=normalized_kind,
            committed_bytes=committed,
            evictable=resolved_evictable,
            pinned=bool(pinned),
            lifecycle=resolved_lifecycle,
            read_only=resolved_read_only,
            last_access=access_sequence,
            residency_granule_bytes=granule,
            _registration_order=self._registration_sequence,
        )
        if initial_resident:
            owner._resident_ranges = initial_resident_ranges
            if dirty:
                owner._dirty_ranges = list(initial_resident_ranges)
            owner._sync_derived_state()
            self._refresh_recency(owner, owner._resident_ranges, access_sequence)
            self._adjust_pool(initial_resident)
        self._allocations[owner_id] = owner
        self._validate_mutation_boundary()
        return owner

    def claim_system(
        self,
        allocation_id: str,
        size_bytes: int,
        *,
        residency_granule_bytes: Optional[int] = None,
    ) -> Allocation:
        """Register a fully resident, pinned system claim."""

        return self.register(
            allocation_id,
            size_bytes,
            kind="system",
            committed_bytes=size_bytes,
            resident_bytes=size_bytes,
            evictable=False,
            pinned=True,
            lifecycle=AllocationLifecycle.SYSTEM,
            residency_granule_bytes=residency_granule_bytes,
        )

    def register_temporary(
        self,
        allocation_id: str,
        size_bytes: int,
        *,
        kind: str = "workspace",
        resident_bytes: Optional[int] = None,
        backing: Optional[str] = None,
        evictable: Optional[bool] = None,
        residency_granule_bytes: Optional[int] = None,
    ) -> Allocation:
        """Register a temporary workspace/activation allocation.

        Temporary memory is fully resident by default and non-evictable when
        it has no backing.  Call :meth:`release` at the lifecycle boundary.
        """

        resident = size_bytes if resident_bytes is None else resident_bytes
        return self.register(
            allocation_id,
            size_bytes,
            kind=kind,
            backing=backing,
            committed_bytes=size_bytes,
            resident_bytes=resident,
            evictable=evictable,
            lifecycle=AllocationLifecycle.TEMPORARY,
            residency_granule_bytes=residency_granule_bytes,
        )

    def create_view(
        self,
        view_id: str,
        owner_id: str,
        *,
        offset_bytes: int = 0,
        size_bytes: Optional[int] = None,
    ) -> AllocationView:
        alias_id = str(view_id).strip()
        if not alias_id:
            raise ResidencyValidationError("view_id must not be empty")
        if alias_id in self._allocations or alias_id in self._views:
            raise ResidencyValidationError(
                "allocation/view id already registered: {}".format(alias_id)
            )
        owner = self.get_allocation(owner_id)
        offset = _integer("offset_bytes", offset_bytes)
        view_size = owner.size_bytes - offset if size_bytes is None else _integer(
            "size_bytes", size_bytes, minimum=1
        )
        if offset >= owner.size_bytes or offset + view_size > owner.size_bytes:
            raise ResidencyValidationError("allocation view exceeds owner bounds")
        view = AllocationView(
            alias_id,
            owner.allocation_id,
            offset,
            view_size,
            owner.residency_granule_bytes,
        )
        self._views[alias_id] = view
        return view

    def release_view(self, view_id: str) -> AllocationView:
        alias_id = str(view_id)
        if any(lease.target_id == alias_id for lease in self._leases.values()):
            raise ResidencyValidationError(
                "release active allocation-view leases first: {}".format(alias_id)
            )
        try:
            return self._views.pop(alias_id)
        except KeyError as exc:
            raise AllocationNotFoundError(
                "unknown allocation view: {}".format(alias_id)
            ) from exc

    def get_allocation(self, allocation_id: str) -> Allocation:
        owner_id = str(allocation_id)
        try:
            return self._allocations[owner_id]
        except KeyError as exc:
            if owner_id in self._views:
                raise ResidencyValidationError(
                    "{} is an AllocationView, not a physical owner".format(owner_id)
                ) from exc
            raise AllocationNotFoundError(
                "unknown allocation: {}".format(owner_id)
            ) from exc

    def get_view(self, view_id: str) -> AllocationView:
        alias_id = str(view_id)
        try:
            return self._views[alias_id]
        except KeyError as exc:
            raise AllocationNotFoundError(
                "unknown allocation view: {}".format(alias_id)
            ) from exc

    def release(self, allocation_id: str, *, force_pinned: bool = False) -> Allocation:
        owner = self.get_allocation(allocation_id)
        active_leases = sorted(
            lease.lease_id
            for lease in self._leases.values()
            if lease.owner_id == owner.allocation_id
        )
        if active_leases:
            raise ResidencyValidationError(
                "release allocation leases first: {}".format(
                    ", ".join(active_leases)
                )
            )
        aliases = sorted(
            view.view_id for view in self._views.values() if view.owner_id == owner.allocation_id
        )
        if aliases:
            raise ResidencyValidationError(
                "release allocation views first: {}".format(", ".join(aliases))
            )
        if owner.pinned and not force_pinned:
            raise ResidencyValidationError(
                "pinned allocation requires force_pinned=True to release"
            )
        released_bytes = owner.resident_bytes
        if released_bytes:
            self._adjust_pool(-released_bytes)
        self._allocations.pop(owner.allocation_id)
        self._released_consumer_ranges.pop(owner.allocation_id, None)
        owner._resident_ranges = []
        owner._dirty_ranges = []
        owner._resident_recency_ranges = []
        owner._sync_derived_state()
        owner.lifecycle = AllocationLifecycle.RELEASED
        self._validate_mutation_boundary()
        return owner

    def acquire_lease(
        self,
        lease_id: str,
        target_id: str,
        consumer_task_id: str,
        *,
        offset_bytes: int = 0,
        size_bytes: Optional[int] = None,
    ) -> ResidencyLease:
        """Protect one already-resident target range for a DAG consumer."""

        normalized_lease_id = str(lease_id).strip()
        normalized_task_id = str(consumer_task_id).strip()
        if not normalized_lease_id:
            raise ResidencyValidationError("lease_id must not be empty")
        if not normalized_task_id:
            raise ResidencyValidationError("consumer_task_id must not be empty")
        if normalized_lease_id in self._leases:
            raise ResidencyValidationError(
                "residency lease already exists: {}".format(normalized_lease_id)
            )
        owner, owner_offset, requested_size, resident_start, resident_end = (
            self._resolve_access(
                MemoryAccess(
                    str(target_id),
                    AccessOperation.TOUCH,
                    offset_bytes,
                    size_bytes,
                )
            )
        )
        ranges = (
            ()
            if resident_start == resident_end
            else ((resident_start, resident_end),)
        )
        if _intersect_ranges(ranges, owner._resident_ranges) != list(ranges):
            raise ResidencyValidationError(
                "cannot lease non-resident bytes for {}".format(
                    owner.allocation_id
                )
            )
        lease = ResidencyLease(
            lease_id=normalized_lease_id,
            consumer_task_id=normalized_task_id,
            target_id=str(target_id),
            owner_id=owner.allocation_id,
            offset_bytes=owner_offset,
            size_bytes=requested_size,
            residency_ranges=tuple(ranges),
        )
        self._leases[normalized_lease_id] = lease
        self._validate_mutation_boundary()
        return lease

    def lease(
        self,
        target_id: str,
        consumer_task_id: str,
        *,
        lease_id: Optional[str] = None,
        offset_bytes: int = 0,
        size_bytes: Optional[int] = None,
    ) -> ResidencyLease:
        """Convenience spelling for :meth:`acquire_lease`."""

        resolved_lease_id = lease_id or "{}:{}".format(
            consumer_task_id, len(self._leases)
        )
        return self.acquire_lease(
            resolved_lease_id,
            target_id,
            consumer_task_id,
            offset_bytes=offset_bytes,
            size_bytes=size_bytes,
        )

    def release_lease(self, lease_id: str) -> ResidencyLease:
        """Release protection while retaining the consumer eviction anchor."""

        normalized = str(lease_id)
        try:
            lease = self._leases.pop(normalized)
        except KeyError as exc:
            raise ResidencyValidationError(
                "unknown residency lease: {}".format(normalized)
            ) from exc
        anchors = self._released_consumer_ranges.setdefault(
            lease.owner_id, []
        )
        # ``acquire_lease`` always resolves one contiguous physical extent.
        # Retain a generic fallback for defensive compatibility with any
        # hand-built internal lease containing multiple ranges.
        if len(lease.residency_ranges) <= 1:
            for start, end in lease.residency_ranges:
                _overwrite_consumer_range(
                    anchors,
                    start,
                    end,
                    lease.consumer_task_id,
                )
        else:
            retained: list[Tuple[int, int, str]] = []
            for start, end, task_id in anchors:
                retained.extend(
                    (fragment_start, fragment_end, task_id)
                    for fragment_start, fragment_end in _subtract_ranges(
                        [(start, end)], lease.residency_ranges
                    )
                )
            retained.extend(
                (start, end, lease.consumer_task_id)
                for start, end in lease.residency_ranges
            )
            retained.sort(key=lambda item: (item[0], item[1], item[2]))
            self._released_consumer_ranges[lease.owner_id] = retained
        self._validate_mutation_boundary()
        return lease

    def release_consumer_leases(
        self, consumer_task_id: str
    ) -> Tuple[ResidencyLease, ...]:
        """Release every lease owned by one completed/planned consumer."""

        task_id = str(consumer_task_id)
        lease_ids = tuple(
            lease_id
            for lease_id, lease in self._leases.items()
            if lease.consumer_task_id == task_id
        )
        return tuple(self.release_lease(lease_id) for lease_id in lease_ids)

    def commit(self, allocation_id: str, committed_bytes: int) -> Allocation:
        """Grow an allocation's logical commitment without consuming capacity."""

        owner = self.get_allocation(allocation_id)
        committed = _integer("committed_bytes", committed_bytes)
        if committed < owner.committed_bytes:
            raise ResidencyValidationError(
                "commit cannot shrink; explicitly evict and release instead"
            )
        if committed > owner.size_bytes:
            raise ResidencyValidationError("committed_bytes exceeds size_bytes")
        try:
            _ensure_backing_unit_ranges(
                "resident commit growth",
                owner._resident_ranges,
                committed,
                owner.residency_granule_bytes,
            )
            _ensure_backing_unit_ranges(
                "dirty commit growth",
                owner._dirty_ranges,
                committed,
                owner.residency_granule_bytes,
            )
            _ensure_backing_unit_ranges(
                "recency commit growth",
                ((start, end) for start, end, _ in owner._resident_recency_ranges),
                committed,
                owner.residency_granule_bytes,
            )
        except ResidencyError as exc:
            raise ResidencyValidationError(
                "commit would split an existing residency backing granule"
            ) from exc
        owner.committed_bytes = committed
        self._validate_mutation_boundary()
        return owner

    def _resolve_access(
        self, access: MemoryAccess
    ) -> Tuple[Allocation, int, int, int, int]:
        target_id = str(access.target_id)
        if target_id in self._allocations:
            owner = self._allocations[target_id]
            view_offset = 0
            view_size = owner.size_bytes
        elif target_id in self._views:
            view = self._views[target_id]
            owner = self.get_allocation(view.owner_id)
            view_offset = view.offset_bytes
            view_size = view.size_bytes
        else:
            raise AllocationNotFoundError("unknown allocation/view: {}".format(target_id))
        offset = _integer("offset_bytes", access.offset_bytes)
        if offset > view_size:
            raise ResidencyValidationError("memory access offset exceeds target bounds")
        requested_size = (
            view_size - offset
            if access.size_bytes is None
            else _integer("size_bytes", access.size_bytes)
        )
        if offset + requested_size > view_size:
            raise ResidencyValidationError("memory access exceeds target bounds")
        owner_offset = view_offset + offset
        if owner_offset + requested_size > owner.committed_bytes:
            raise ResidencyValidationError("memory access exceeds committed bytes")
        if requested_size == 0:
            return owner, owner_offset, requested_size, owner_offset, owner_offset
        granule = owner.residency_granule_bytes
        resident_start = (owner_offset // granule) * granule
        resident_end = min(
            owner.committed_bytes,
            _ceil_div(owner_offset + requested_size, granule) * granule,
        )
        return owner, owner_offset, requested_size, resident_start, resident_end

    @staticmethod
    def _operation(value: Union[AccessOperation, str]) -> AccessOperation:
        try:
            return AccessOperation(value)
        except ValueError as exc:
            raise ResidencyValidationError(
                "unsupported memory access operation: {}".format(value)
            ) from exc

    @staticmethod
    def _write_refault_ranges(
        owner: Allocation,
        owner_offset: int,
        requested_size: int,
        resident_start: int,
        resident_end: int,
        missing_ranges: Sequence[_Range],
    ) -> list[_Range]:
        """Return missing bytes whose granule is not fully overwritten.

        Fully covered granules can be allocated directly because no old bytes
        must be preserved.  Only the first and last granules can be partially
        covered by a contiguous write, so this stays compact for very large
        full-allocation clears.
        """

        if requested_size == 0 or not missing_ranges:
            return []
        write_end = owner_offset + requested_size
        granule = owner.residency_granule_bytes
        preserve_granules: list[_Range] = []
        if owner_offset > resident_start:
            preserve_granules.append(
                (
                    resident_start,
                    min(owner.committed_bytes, resident_start + granule),
                )
            )
        if write_end < resident_end:
            last_start = ((write_end - 1) // granule) * granule
            preserve_granules.append(
                (
                    last_start,
                    min(owner.committed_bytes, last_start + granule),
                )
            )
        return _intersect_ranges(missing_ranges, preserve_granules)

    def access(self, access: MemoryAccess) -> AccessResult:
        """Apply one touch/read/write and return only its migrations."""

        if not isinstance(access, MemoryAccess):
            raise ResidencyValidationError("access must be a MemoryAccess")
        self._validate_mutation_boundary()
        operation = self._operation(access.operation)
        owner, owner_offset, requested_size, resident_start, resident_end = (
            self._resolve_access(access)
        )
        if operation is AccessOperation.WRITE and owner.read_only:
            raise ResidencyValidationError(
                "cannot write read-only allocation {}".format(owner.allocation_id)
            )
        required_ranges = (
            [] if resident_start == resident_end else [(resident_start, resident_end)]
        )
        missing_ranges = _subtract_ranges(required_ranges, owner._resident_ranges)
        missing_bytes = _range_bytes(missing_ranges)
        page_in_ranges = list(missing_ranges)
        if operation is AccessOperation.WRITE:
            page_in_ranges = self._write_refault_ranges(
                owner,
                owner_offset,
                requested_size,
                resident_start,
                resident_end,
                missing_ranges,
            )
        page_in_bytes = _range_bytes(page_in_ranges)
        if page_in_bytes and owner.backing is None:
            raise ResidencyValidationError(
                "cannot refault unbacked allocation {}".format(owner.allocation_id)
            )

        migration_start = len(self._migrations)
        if missing_bytes:
            self._make_room(
                missing_bytes,
                protected_owner=owner.allocation_id,
                protected_ranges=required_ranges,
            )
            owner._resident_ranges = _merge_ranges(
                [*owner._resident_ranges, *missing_ranges]
            )
            owner._sync_derived_state()
            self._adjust_pool(missing_bytes)
            if page_in_bytes:
                self._record_migration(
                    owner,
                    MigrationKind.PAGE_IN,
                    page_in_bytes,
                    owner.backing,
                    self.pool.component_id,
                )
        if operation is AccessOperation.WRITE and required_ranges:
            owner._dirty_ranges = _merge_ranges(
                [*owner._dirty_ranges, *required_ranges]
            )
            owner._sync_derived_state()
        sequence = self._next_access_sequence()
        owner.last_access = sequence
        self._refresh_recency(owner, required_ranges, sequence)
        migrations = tuple(self._migrations[migration_start:])
        self._validate_mutation_boundary()
        return AccessResult(
            sequence=sequence,
            target_id=str(access.target_id),
            owner_id=owner.allocation_id,
            operation=operation,
            offset_bytes=owner_offset,
            size_bytes=requested_size,
            page_in_bytes=sum(item.page_in_bytes for item in migrations),
            migrations=migrations,
            physical_residency_bytes=_range_bytes(required_ranges),
            residency_ranges=tuple(required_ranges),
        )

    def access_many(self, accesses: Iterable[MemoryAccess]) -> Tuple[AccessResult, ...]:
        """Process accesses in iterable order; no set/dict reordering is applied."""

        return tuple(self.access(access) for access in accesses)

    def touch(
        self, target_id: str, *, offset_bytes: int = 0, size_bytes: Optional[int] = None
    ) -> AccessResult:
        return self.access(
            MemoryAccess(target_id, AccessOperation.TOUCH, offset_bytes, size_bytes)
        )

    def read(
        self, target_id: str, *, offset_bytes: int = 0, size_bytes: Optional[int] = None
    ) -> AccessResult:
        return self.access(
            MemoryAccess(target_id, AccessOperation.READ, offset_bytes, size_bytes)
        )

    def write(
        self, target_id: str, *, offset_bytes: int = 0, size_bytes: Optional[int] = None
    ) -> AccessResult:
        return self.access(
            MemoryAccess(target_id, AccessOperation.WRITE, offset_bytes, size_bytes)
        )

    def evict(self, allocation_id: str, byte_count: Optional[int] = None) -> Tuple[Migration, ...]:
        """Explicitly evict at least ``byte_count`` bytes, highest units first."""

        self._validate_mutation_boundary()
        owner = self.get_allocation(allocation_id)
        if owner.pinned or not owner.evictable:
            raise ResidencyValidationError(
                "allocation {} is not evictable".format(owner.allocation_id)
            )
        amount = owner.resident_bytes if byte_count is None else _integer(
            "byte_count", byte_count
        )
        if amount > owner.resident_bytes:
            raise ResidencyValidationError(
                "eviction byte_count exceeds resident bytes"
            )
        if amount == 0:
            return ()
        migration_start = len(self._migrations)
        self._evict_owner_bytes(owner, amount, ())
        self._validate_mutation_boundary()
        return tuple(self._migrations[migration_start:])

    def _candidate_ranges(
        self,
        owner: Allocation,
        protected_owner: Optional[str],
        protected_ranges: Sequence[_Range],
    ) -> list[_Range]:
        if owner.pinned or not owner.evictable or not owner._resident_ranges:
            return []
        if owner.backing is None:
            return []
        cuts = list(self._leased_ranges(owner.allocation_id))
        if owner.allocation_id == protected_owner:
            cuts.extend(protected_ranges)
        return _subtract_ranges(owner._resident_ranges, cuts)

    def _leased_ranges(self, owner_id: str) -> list[_Range]:
        return _merge_ranges(
            residency_range
            for lease in self._leases.values()
            if lease.owner_id == owner_id
            for residency_range in lease.residency_ranges
        )

    def _candidate_recency_ranges(
        self,
        owner: Allocation,
        protected_owner: Optional[str],
        protected_ranges: Sequence[_Range],
    ) -> list[_RecencyRange]:
        """Return evictable fragments with their range-level access sequence."""

        if owner.pinned or not owner.evictable or not owner._resident_ranges:
            return []
        if owner.backing is None:
            return []
        cuts = list(self._leased_ranges(owner.allocation_id))
        if owner.allocation_id == protected_owner:
            cuts.extend(protected_ranges)
        candidates: list[_RecencyRange] = []
        for start, end, sequence in owner._resident_recency_ranges:
            candidates.extend(
                (fragment_start, fragment_end, sequence)
                for fragment_start, fragment_end in _subtract_ranges(
                    [(start, end)], cuts
                )
            )
        return candidates

    def _make_room(
        self,
        additional_bytes: int,
        *,
        protected_owner: Optional[str] = None,
        protected_ranges: Sequence[_Range] = (),
    ) -> None:
        additional = _integer("additional_bytes", additional_bytes)
        need = max(0, self.pool.resident_bytes + additional - self.pool.capacity_bytes)
        if need == 0:
            return
        candidates: list[
            Tuple[int, int, str, int, int, Allocation]
        ] = []
        for owner in self._allocations.values():
            for start, end, sequence in self._candidate_recency_ranges(
                owner, protected_owner, protected_ranges
            ):
                _ensure_backing_unit_ranges(
                    "eviction candidate",
                    [(start, end)],
                    owner.committed_bytes,
                    owner.residency_granule_bytes,
                )
                candidates.append(
                    (
                        sequence,
                        owner._registration_order,
                        owner.allocation_id,
                        start,
                        end,
                        owner,
                    )
                )
        candidates.sort(key=lambda item: item[:4])

        plan_order: list[str] = []
        planned_ranges: dict[str, list[_Range]] = {}
        planned_owners: dict[str, Allocation] = {}
        remaining = need
        for _, _, owner_id, start, end, owner in candidates:
            if remaining <= 0:
                break
            selected = _backing_unit_prefix_for_bytes(
                start,
                end,
                owner.residency_granule_bytes,
                remaining,
            )
            if selected is None:
                continue
            if owner_id not in planned_ranges:
                plan_order.append(owner_id)
                planned_ranges[owner_id] = []
                planned_owners[owner_id] = owner
            planned_ranges[owner_id].append(selected)
            remaining -= selected[1] - selected[0]
        if remaining > 0:
            raise CapacityExceededError(
                "{} needs {} additional resident bytes, but only {} of {} can be made available".format(
                    self.pool.component_id,
                    additional,
                    need - remaining,
                    need,
                )
            )
        for owner_id in plan_order:
            self._evict_owner_ranges(
                planned_owners[owner_id], planned_ranges[owner_id]
            )

    def _evict_owner_bytes(
        self,
        owner: Allocation,
        byte_count: int,
        protected_ranges: Sequence[_Range],
    ) -> None:
        if owner.pinned or not owner.evictable or owner.backing is None:
            raise ResidencyError(
                "unsafe eviction plan for {}".format(owner.allocation_id)
            )
        available_ranges = self._candidate_ranges(
            owner, owner.allocation_id, protected_ranges
        )
        available = _range_bytes(available_ranges)
        if byte_count < 0 or byte_count > available:
            raise ResidencyError(
                "eviction underflow for {}".format(owner.allocation_id)
            )
        remaining = byte_count
        selected: list[_Range] = []
        for start, end in reversed(available_ranges):
            if remaining <= 0:
                break
            units = list(
                _iter_backing_unit_ranges(
                    owner.committed_bytes,
                    owner.residency_granule_bytes,
                    [(start, end)],
                )
            )
            for unit_start, unit_end in reversed(units):
                if remaining <= 0:
                    break
                selected.append((unit_start, unit_end))
                remaining -= unit_end - unit_start
        if remaining > 0:
            raise ResidencyError(
                "eviction selection underflow for {}".format(owner.allocation_id)
            )
        self._evict_owner_ranges(owner, selected)

    def _evict_owner_ranges(
        self,
        owner: Allocation,
        selected_ranges: Sequence[_Range],
    ) -> None:
        """Evict already-selected resident ranges without re-running policy."""

        if owner.pinned or not owner.evictable or owner.backing is None:
            raise ResidencyError(
                "unsafe eviction plan for {}".format(owner.allocation_id)
            )
        selected = _merge_ranges(selected_ranges)
        if not selected:
            return
        _ensure_backing_unit_ranges(
            "eviction selection",
            selected,
            owner.committed_bytes,
            owner.residency_granule_bytes,
        )
        selected_bytes = _range_bytes(selected)
        resident_selection = _intersect_ranges(selected, owner._resident_ranges)
        if resident_selection != selected:
            raise ResidencyError(
                "eviction selection includes non-resident bytes for {}".format(
                    owner.allocation_id
                )
            )
        dirty_bytes = _range_bytes(
            _intersect_ranges(selected, owner._dirty_ranges)
        )
        clean_bytes = selected_bytes - dirty_bytes
        consumer_task_ids = tuple(
            dict.fromkeys(
                task_id
                for start, end, task_id in self._released_consumer_ranges.get(
                    owner.allocation_id, ()
                )
                if _intersect_ranges([(start, end)], selected)
            )
        )
        retained_consumer_ranges: list[Tuple[int, int, str]] = []
        for start, end, task_id in self._released_consumer_ranges.get(
            owner.allocation_id, ()
        ):
            retained_consumer_ranges.extend(
                (fragment_start, fragment_end, task_id)
                for fragment_start, fragment_end in _subtract_ranges(
                    [(start, end)], selected
                )
            )
        if retained_consumer_ranges:
            self._released_consumer_ranges[owner.allocation_id] = (
                retained_consumer_ranges
            )
        else:
            self._released_consumer_ranges.pop(owner.allocation_id, None)
        owner._resident_ranges = _subtract_ranges(owner._resident_ranges, selected)
        owner._dirty_ranges = _subtract_ranges(owner._dirty_ranges, selected)
        owner._resident_recency_ranges = _subtract_recency_ranges(
            owner._resident_recency_ranges, selected
        )
        owner._sync_derived_state()
        self._adjust_pool(-selected_bytes)
        # Fixed category order keeps mixed clean/dirty eviction deterministic.
        if dirty_bytes:
            self._record_migration(
                owner,
                MigrationKind.DIRTY_WRITEBACK,
                dirty_bytes,
                self.pool.component_id,
                owner.backing,
                consumer_task_ids,
            )
        if clean_bytes:
            self._record_migration(
                owner,
                MigrationKind.CLEAN_DISCARD,
                clean_bytes,
                self.pool.component_id,
                owner.backing,
                consumer_task_ids,
            )

    def assert_consistent(self) -> None:
        """Fail closed if public accounting invariants have been violated."""

        expected = 0
        for owner_id, owner in self._allocations.items():
            if owner.allocation_id != owner_id:
                raise ResidencyError("allocation registry owner mismatch")
            if owner.physical_owner != self.pool.component_id:
                raise ResidencyError("allocation belongs to another physical pool")
            owner._sync_derived_state()
            if owner.resident_bytes > owner.committed_bytes:
                raise ResidencyError("resident bytes exceed commitment")
            if owner.dirty_bytes > owner.resident_bytes:
                raise ResidencyError("dirty bytes exceed resident bytes")
            granule = _integer(
                "residency_granule_bytes",
                owner.residency_granule_bytes,
                minimum=1,
            )
            _ensure_backing_unit_ranges(
                "allocation resident range",
                owner._resident_ranges,
                owner.committed_bytes,
                granule,
            )
            _ensure_backing_unit_ranges(
                "allocation dirty range",
                owner._dirty_ranges,
                owner.committed_bytes,
                granule,
            )
            normalized_recency = _merge_recency_ranges(
                owner._resident_recency_ranges
            )
            if normalized_recency != owner._resident_recency_ranges:
                raise ResidencyError(
                    "allocation recency ranges are not normalized"
                )
            recency_coverage = _merge_ranges(
                (start, end)
                for start, end, _ in owner._resident_recency_ranges
            )
            if recency_coverage != owner._resident_ranges:
                raise ResidencyError(
                    "allocation recency does not cover exactly its resident bytes"
                )
            _ensure_backing_unit_ranges(
                "allocation recency range",
                ((start, end) for start, end, _ in owner._resident_recency_ranges),
                owner.committed_bytes,
                granule,
            )
            for start, end, sequence in owner._resident_recency_ranges:
                if start < 0 or end > owner.committed_bytes:
                    raise ResidencyError(
                        "allocation recency range exceeds commitment"
                    )
                if sequence <= 0 or sequence > owner.last_access:
                    raise ResidencyError(
                        "allocation recency sequence is invalid"
                    )
            if owner.last_access <= 0 or owner.last_access > self._access_sequence:
                raise ResidencyError("allocation last_access sequence is invalid")
            if owner.pinned and owner.evictable:
                raise ResidencyError("pinned allocation is marked evictable")
            expected += owner.resident_bytes
        for view_id, view in self._views.items():
            if view.view_id != view_id or view.owner_id not in self._allocations:
                raise ResidencyError("stale or malformed allocation view")
            owner = self._allocations[view.owner_id]
            if view.offset_bytes < 0 or view.offset_bytes + view.size_bytes > owner.size_bytes:
                raise ResidencyError("allocation view exceeds owner bounds")
            if view.residency_granule_bytes != owner.residency_granule_bytes:
                raise ResidencyError("allocation view residency granule is stale")
        for lease_id, lease in self._leases.items():
            if lease.lease_id != lease_id or lease.owner_id not in self._allocations:
                raise ResidencyError("stale or malformed residency lease")
            owner = self._allocations[lease.owner_id]
            if _intersect_ranges(lease.residency_ranges, owner._resident_ranges) != list(
                lease.residency_ranges
            ):
                raise ResidencyError("residency lease covers non-resident bytes")
        if expected != self.pool.resident_bytes:
            raise ResidencyError(
                "physical pool accounting mismatch: allocations={} pool={}".format(
                    expected, self.pool.resident_bytes
                )
            )
        if expected < 0 or expected > self.pool.capacity_bytes:
            raise ResidencyError("physical pool accounting is outside capacity")


__all__ = [
    "AccessOperation",
    "AccessResult",
    "Allocation",
    "AllocationLifecycle",
    "AllocationNotFoundError",
    "AllocationResidencyManager",
    "AllocationView",
    "CapacityExceededError",
    "MemoryAccess",
    "Migration",
    "MigrationKind",
    "MigrationTotals",
    "PhysicalPool",
    "ResidencyError",
    "ResidencyLease",
    "ResidencySnapshot",
    "ResidencyValidationError",
]
