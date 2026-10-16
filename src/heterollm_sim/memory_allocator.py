"""Deterministic physical allocation for explicit memory transactions.

The planner may describe a buffer by name and a local byte range, but a
descriptor hash is not an allocator: it can collide and it changes when the
requested extent changes.  This small first-fit allocator gives each
``(buffer_id, generation)`` one persistent extent for the lifetime of a
simulation.  Overlap is accepted only through an explicit alias declaration.
"""

from __future__ import annotations

from bisect import bisect_left

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple


class AllocationError(ValueError):
    """Raised when a physical allocation cannot be represented safely."""


@dataclass(frozen=True)
class PhysicalAllocation:
    buffer_id: str
    base_address: int
    size_bytes: int
    generation: int = 0
    alias_of: Optional[str] = None
    alias_generation: Optional[int] = None
    alias_offset_bytes: int = 0
    inferred: bool = False
    allocation_arena: Optional[str] = None

    def address(self, offset_bytes: int, size_bytes: int = 1) -> int:
        if type(offset_bytes) is not int or offset_bytes < 0:
            raise AllocationError("offset_bytes must be a non-negative integer")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise AllocationError("size_bytes must be a positive integer")
        if offset_bytes + size_bytes > self.size_bytes:
            raise AllocationError(
                "access exceeds allocation {} (offset {} + length {} > {})".format(
                    self.buffer_id, offset_bytes, size_bytes, self.size_bytes
                )
            )
        return self.base_address + offset_bytes


class PhysicalAddressAllocator:
    """A run-local, aligned, non-moving physical address allocator.

    Allocations are keyed by buffer identity and generation.  Re-registering
    an existing key returns the same base address as long as the declared
    extent is compatible.  ``alias_of`` is the only way to deliberately share
    an address range with another allocation.
    """

    def __init__(self, capacity_bytes: int, alignment_bytes: int = 64, *, workspace_capacity_bytes: int = 0) -> None:
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise AllocationError("capacity_bytes must be a positive integer")
        if type(alignment_bytes) is not int or alignment_bytes <= 0:
            raise AllocationError("alignment_bytes must be a positive integer")
        if type(workspace_capacity_bytes) is not int or not 0 <= workspace_capacity_bytes < capacity_bytes:
            raise AllocationError("workspace_capacity_bytes must be an integer in [0, capacity_bytes)")
        if workspace_capacity_bytes and (capacity_bytes - workspace_capacity_bytes) % alignment_bytes:
            raise AllocationError("workspace partition boundary must be aligned")
        self.capacity_bytes = capacity_bytes
        self.alignment_bytes = alignment_bytes
        self.workspace_capacity_bytes = workspace_capacity_bytes
        self._allocations: Dict[Tuple[str, int], PhysicalAllocation] = {}
        self._address_keys = []
        self._address_rows = []
        self._largest_allocation = 0

    def _store_allocation(self, key, allocation):
        previous = self._allocations.get(key)
        if previous is allocation:
            return
        if previous is not None:
            old_key = (previous.base_address, previous.buffer_id, previous.generation)
            position = bisect_left(self._address_keys, old_key)
            self._address_keys.pop(position)
            self._address_rows.pop(position)
        address_key = (allocation.base_address, allocation.buffer_id, allocation.generation)
        position = bisect_left(self._address_keys, address_key)
        self._address_keys.insert(position, address_key)
        self._address_rows.insert(position, allocation)
        self._largest_allocation = max(self._largest_allocation, allocation.size_bytes)
        self._allocations[key] = allocation

    def _restore_allocations(self, allocations):
        """Rebuild the address index at the existing rollback boundary."""
        self._allocations.clear()
        self._allocations.update(allocations)
        self._address_rows = sorted(allocations.values(),
            key=lambda item: (item.base_address, item.buffer_id, item.generation))
        self._address_keys = [(item.base_address, item.buffer_id, item.generation)
                              for item in self._address_rows]
        self._largest_allocation = max((item.size_bytes for item in self._address_rows), default=0)

    @staticmethod
    def _key(buffer_id: str, generation: int) -> Tuple[str, int]:
        if not isinstance(buffer_id, str) or not buffer_id.strip():
            raise AllocationError("buffer_id must be non-empty text")
        if type(generation) is not int or generation < 0:
            raise AllocationError("generation must be a non-negative integer")
        return buffer_id.strip(), generation

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    def allocations(self) -> Tuple[PhysicalAllocation, ...]:
        return tuple(self._address_rows)

    def _check_extent(self, base: int, size: int) -> None:
        if type(base) is not int or base < 0:
            raise AllocationError("base address must be a non-negative integer")
        if type(size) is not int or size <= 0:
            raise AllocationError("size_bytes must be a positive integer")
        if base + size > self.capacity_bytes:
            raise AllocationError(
                "allocation range {} + {} exceeds capacity {}".format(
                    base, size, self.capacity_bytes
                )
            )

    def _overlaps(self, base: int, size: int, *, ignore: Optional[Tuple[str, int]] = None) -> Iterable[PhysicalAllocation]:
        end = base + size
        for key, item in self._allocations.items():
            if key == ignore or item.alias_of is not None:
                continue
            if base < item.base_address + item.size_bytes and item.base_address < end:
                yield item

    def _arena(self, generation: int, allocation_arena: Optional[str] = None) -> Tuple[int, int]:
        if not self.workspace_capacity_bytes:
            return 0, self.capacity_bytes
        boundary = self.capacity_bytes - self.workspace_capacity_bytes
        resident = allocation_arena == "resident" if allocation_arena is not None else generation == 0
        return (0, boundary) if resident else (boundary, self.capacity_bytes)

    def _check_arena_extent(self, base: int, size: int, generation: int, allocation_arena: Optional[str] = None) -> None:
        self._check_extent(base, size)
        lower, upper = self._arena(generation, allocation_arena)
        if base < lower or base + size > upper:
            raise AllocationError("allocation generation {} range {} + {} exceeds its {} arena [{}, {})".format(
                generation, base, size, allocation_arena or ("resident" if generation == 0 else "workspace"), lower, upper))

    def _find_first_fit(self, size: int, generation: int = 0, allocation_arena: Optional[str] = None) -> int:
        lower, upper = self._arena(generation, allocation_arena)
        cursor = lower
        # No allocation starting earlier than this can intersect the arena.
        # The conservative maximum size also covers overlapping extents in
        # imported snapshots; a single preceding row would not suffice there.
        position = bisect_left(self._address_keys, (lower - self._largest_allocation,))
        for index in range(position, len(self._address_rows)):
            item = self._address_rows[index]
            if item.base_address >= upper:
                break
            if item.alias_of is not None:
                continue
            if item.base_address + item.size_bytes <= lower or item.base_address >= upper:
                continue
            cursor = self._align(cursor, self.alignment_bytes)
            if cursor + size <= item.base_address:
                return cursor
            cursor = max(cursor, item.base_address + item.size_bytes)
        cursor = self._align(cursor, self.alignment_bytes)
        if cursor + size > upper:
            raise AllocationError(
                "unable to allocate {} bytes in capacity {}{}".format(size, self.capacity_bytes,
                    " ({} arena [{}, {}))".format(allocation_arena or ("resident" if generation == 0 else "workspace"), lower, upper)
                    if self.workspace_capacity_bytes else "")
            )
        return cursor

    def allocate(
        self,
        buffer_id: str,
        size_bytes: int,
        generation: int = 0,
        *,
        alias_of: Optional[str] = None,
        alias_generation: Optional[int] = None,
        address: Optional[int] = None,
        alias_offset_bytes: int = 0,
        inferred: bool = False,
        allocation_arena: Optional[str] = None,
    ) -> PhysicalAllocation:
        key = self._key(buffer_id, generation)
        if type(size_bytes) is not int or size_bytes <= 0:
            raise AllocationError("size_bytes must be a positive integer")
        existing = self._allocations.get(key)
        if allocation_arena is not None and allocation_arena not in ("resident", "workspace"):
            raise AllocationError("allocation_arena must be resident or workspace")
        if type(inferred) is not bool:
            raise AllocationError("inferred must be a boolean")
        if alias_generation is not None and (type(alias_generation) is not int or alias_generation < 0):
            raise AllocationError("alias_generation must be a non-negative integer")
        if alias_of is not None:
            alias_of = str(alias_of).strip()
            if not alias_of:
                raise AllocationError("alias_of must be non-empty text")
            if alias_generation is None:
                alias_generation = key[1]
        if type(alias_offset_bytes) is not int or alias_offset_bytes < 0:
            raise AllocationError("alias_offset_bytes must be non-negative")
        if existing is not None:
            if (allocation_arena is not None and existing.allocation_arena is not None
                    and allocation_arena != existing.allocation_arena):
                raise AllocationError("allocation arena changed for {}".format(buffer_id))
            arena = existing.allocation_arena if allocation_arena is None else allocation_arena
            if allocation_arena is not None:
                self._check_arena_extent(existing.base_address, existing.size_bytes, generation, arena)
            if address is not None and (type(address) is not int or address != existing.base_address):
                raise AllocationError("allocation address changed for {}".format(buffer_id))
            if existing.alias_of != alias_of or existing.alias_generation != alias_generation:
                raise AllocationError("allocation alias changed for {}".format(buffer_id))
            if existing.alias_offset_bytes != alias_offset_bytes:
                raise AllocationError("allocation alias offset changed for {}".format(buffer_id))
            if size_bytes != existing.size_bytes and size_bytes > existing.size_bytes:
                if existing.alias_of is not None or not existing.inferred or not inferred:
                    raise AllocationError("allocation {} cannot grow".format(buffer_id))
                self._check_arena_extent(existing.base_address, size_bytes, generation, arena)
                if tuple(self._overlaps(existing.base_address, size_bytes, ignore=key)):
                    raise AllocationError(
                        "allocation {} generation {} cannot grow without moving".format(*key)
                    )
                existing = PhysicalAllocation(
                    buffer_id=existing.buffer_id, base_address=existing.base_address,
                    size_bytes=size_bytes, generation=existing.generation,
                    alias_of=existing.alias_of, alias_generation=existing.alias_generation,
                    alias_offset_bytes=existing.alias_offset_bytes, inferred=existing.inferred,
                    allocation_arena=existing.allocation_arena,
                )
            elif size_bytes != existing.size_bytes and (not existing.inferred or not inferred):
                raise AllocationError("allocation size changed for {}".format(buffer_id))
            if arena != existing.allocation_arena:
                existing = replace(existing, allocation_arena=arena)
            self._store_allocation(key, existing)
            return existing

        alias_key = None
        base: int
        if alias_of is not None:
            target_key = (alias_of, alias_generation)
            target = self._allocations.get(target_key)
            if target is None:
                raise AllocationError("alias target {} generation {} is not allocated".format(*target_key))
            target_arena = target.allocation_arena or ("resident" if target.generation == 0 else "workspace")
            if allocation_arena is not None and allocation_arena != target_arena:
                raise AllocationError("alias arena differs from target allocation")
            allocation_arena = target_arena
            if alias_offset_bytes + size_bytes > target.size_bytes:
                raise AllocationError("alias range exceeds target allocation {} generation {}".format(*target_key))
            base = target.base_address + alias_offset_bytes
            alias_key = alias_of
            self._check_extent(base, size_bytes)
            if address is not None and address != base:
                raise AllocationError("allocation address changed for {}".format(buffer_id))
        elif address is not None:
            if type(address) is not int or address < 0:
                raise AllocationError("address must be a non-negative integer")
            base = address
            self._check_arena_extent(base, size_bytes, generation, allocation_arena)
            conflicts = tuple(self._overlaps(base, size_bytes))
            if conflicts:
                raise AllocationError(
                    "allocation {} overlaps {} without an explicit alias".format(
                        buffer_id, conflicts[0].buffer_id
                    )
                )
        else:
            base = self._find_first_fit(size_bytes, generation, allocation_arena)

        allocation = PhysicalAllocation(buffer_id=key[0], base_address=base, size_bytes=size_bytes,
                                        generation=key[1], alias_of=alias_key,
                                        alias_generation=alias_generation,
                                        alias_offset_bytes=alias_offset_bytes,
                                        inferred=inferred, allocation_arena=allocation_arena)
        self._store_allocation(key, allocation)
        return allocation

    def register(self, declaration: dict) -> PhysicalAllocation:
        """Register one descriptor declaration without accepting hidden overlap."""
        if not isinstance(declaration, dict):
            raise AllocationError("physical allocation declaration must be a mapping")
        obsolete = {"allocation_size_bytes", "allocation_generation"}.intersection(declaration)
        if obsolete:
            raise AllocationError("allocation record uses size_bytes and generation; unsupported fields: " + ", ".join(sorted(obsolete)))
        buffer_id = declaration.get("buffer_id")
        size = declaration.get("size_bytes")
        if size is None:
            raise AllocationError("physical allocation requires size_bytes")
        return self.allocate(
            # Keep the allocator's strict type checks in force.  Coercing
            # descriptor values with ``int`` silently truncated malformed
            # floating-point/string input (for example ``64.9`` -> ``64``),
            # producing an extent different from the one the caller declared.
            buffer_id,
            size,
            declaration.get("generation", 0),
            alias_of=declaration.get("alias_of"),
            alias_generation=declaration.get("alias_generation"),
            address=declaration.get("base_address"),
            alias_offset_bytes=declaration.get("alias_offset_bytes", 0),
            allocation_arena=declaration.get("allocation_arena"),
        )

    def lookup(self, buffer_id: str, generation: int = 0) -> PhysicalAllocation:
        allocation = self._allocations.get(self._key(buffer_id, generation))
        if allocation is None:
            raise AllocationError("allocation {} generation {} is not registered".format(buffer_id, generation))
        return allocation

    def get_allocation(self, buffer_id: str, generation: int = 0) -> Optional[PhysicalAllocation]:
        """Return a registered allocation for adapters that use optional lookup."""
        return self._allocations.get(self._key(buffer_id, generation))

    def canonical_range(
        self, buffer_id: str, offset_bytes: int, size_bytes: int, generation: int = 0
    ) -> Tuple[str, int, int]:
        """Return the root allocation identity and offset for a (possibly aliased) range."""
        allocation = self.lookup(buffer_id, generation)
        allocation.address(offset_bytes, size_bytes)
        offset = offset_bytes
        visited = set()
        while allocation.alias_of is not None:
            key = (allocation.buffer_id, allocation.generation)
            if key in visited:
                raise AllocationError("cyclic allocation alias")
            visited.add(key)
            target = self.lookup(allocation.alias_of, allocation.alias_generation)
            offset += allocation.base_address - target.base_address
            allocation = target
        return allocation.buffer_id, allocation.generation, offset

    def address(self, buffer_id: str, offset_bytes: int, size_bytes: int, generation: int = 0) -> int:
        key = self._key(buffer_id, generation)
        allocation = self._allocations.get(key)
        if allocation is None:
            raise AllocationError(
                "allocation {} generation {} is not registered".format(
                    buffer_id, generation
                )
            )
        return allocation.address(offset_bytes, size_bytes)

    def release(self, buffer_id: str, generation: int = 0) -> None:
        key = self._key(buffer_id, generation)
        if key not in self._allocations:
            return
        for item in self._allocations.values():
            if item.alias_of is None:
                continue
            current = item
            visited = set()
            while current.alias_of is not None:
                current_key = (current.buffer_id, current.generation)
                if current_key in visited:
                    break
                visited.add(current_key)
                target_key = (current.alias_of, current.alias_generation)
                if target_key == key:
                    raise AllocationError("cannot release allocation {} generation {} with live aliases".format(*key))
                target = self._allocations.get(target_key)
                if target is None:
                    break
                current = target
        previous = self._allocations.pop(key)
        position = bisect_left(self._address_keys,
            (previous.base_address, previous.buffer_id, previous.generation))
        self._address_keys.pop(position)
        self._address_rows.pop(position)

    def snapshot(self) -> dict:
        snapshot = {
            "capacity_bytes": self.capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
            "allocations": tuple(self.allocations()),
        }
        if self.workspace_capacity_bytes:
            snapshot["workspace_capacity_bytes"] = self.workspace_capacity_bytes
        return snapshot

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "PhysicalAddressAllocator":
        allocator = cls(snapshot["capacity_bytes"], snapshot["alignment_bytes"],
                        workspace_capacity_bytes=snapshot.get("workspace_capacity_bytes", 0))
        # Serialized snapshots are address/name ordered, not alias ordered.
        # Register targets before views, including chains of nested aliases.
        pending = list(snapshot["allocations"])
        ordered = []
        available = set()
        while pending:
            deferred = []
            for item in pending:
                if isinstance(item, PhysicalAllocation):
                    buffer_id, generation = item.buffer_id, item.generation
                    alias_of, alias_generation = item.alias_of, item.alias_generation
                else:
                    item = dict(item)
                    obsolete = {"allocation_size_bytes", "allocation_generation"}.intersection(item)
                    if obsolete:
                        raise AllocationError("allocation record uses size_bytes and generation; unsupported fields: " + ", ".join(sorted(obsolete)))
                    buffer_id, generation = item["buffer_id"], item.get("generation", 0)
                    alias_of, alias_generation = item.get("alias_of"), item.get("alias_generation")
                if alias_of is not None and (alias_of, generation if alias_generation is None else alias_generation) not in available:
                    deferred.append(item)
                    continue
                ordered.append(item)
                available.add((buffer_id, generation))
            if len(deferred) == len(pending):
                raise AllocationError("snapshot alias target is missing or aliases form a cycle")
            pending = deferred
        for item in ordered:
            if isinstance(item, PhysicalAllocation):
                if item.alias_of is not None and item.alias_generation is None:
                    item = PhysicalAllocation(
                        buffer_id=item.buffer_id, base_address=item.base_address,
                        size_bytes=item.size_bytes, generation=item.generation,
                        alias_of=item.alias_of, alias_generation=item.generation,
                        alias_offset_bytes=item.alias_offset_bytes, inferred=item.inferred,
                        allocation_arena=item.allocation_arena,
                    )
                allocator._store_allocation((item.buffer_id, item.generation), item)
            else:
                item = dict(item)
                size = item.get("size_bytes")
                if size is None:
                    raise AllocationError("physical allocation requires size_bytes")
                alias_generation = item.get("alias_generation")
                if item.get("alias_of") is not None and alias_generation is None:
                    alias_generation = item.get("generation", 0)
                allocator.allocate(
                    item["buffer_id"], size, item.get("generation", 0),
                    alias_of=item.get("alias_of"), alias_generation=alias_generation,
                    address=item.get("base_address"), alias_offset_bytes=item.get("alias_offset_bytes", 0),
                    inferred=item.get("inferred", False),
                    allocation_arena=item.get("allocation_arena"),
                )
        # Snapshot order is a stable sort of insertion order. Rebuild once
        # here, rather than let incremental insertion change that public
        # order for native snapshots containing non-total-order values.
        allocator._restore_allocations(dict(allocator._allocations))
        return allocator


__all__ = ["AllocationError", "PhysicalAllocation", "PhysicalAddressAllocator"]
