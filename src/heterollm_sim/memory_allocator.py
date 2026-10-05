"""Deterministic physical allocation for explicit memory transactions.

The planner may describe a buffer by name and a local byte range, but a
descriptor hash is not an allocator: it can collide and it changes when the
requested extent changes.  This small first-fit allocator gives each
``(buffer_id, generation)`` one persistent extent for the lifetime of a
simulation.  Overlap is accepted only through an explicit alias declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def __init__(self, capacity_bytes: int, alignment_bytes: int = 64) -> None:
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise AllocationError("capacity_bytes must be a positive integer")
        if type(alignment_bytes) is not int or alignment_bytes <= 0:
            raise AllocationError("alignment_bytes must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self.alignment_bytes = alignment_bytes
        self._allocations: Dict[Tuple[str, int], PhysicalAllocation] = {}

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
        return tuple(
            sorted(self._allocations.values(), key=lambda item: (item.base_address, item.buffer_id, item.generation))
        )

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

    def _find_first_fit(self, size: int) -> int:
        cursor = 0
        for item in self.allocations():
            if item.alias_of is not None:
                continue
            cursor = self._align(cursor, self.alignment_bytes)
            if cursor + size <= item.base_address:
                return cursor
            cursor = max(cursor, item.base_address + item.size_bytes)
        cursor = self._align(cursor, self.alignment_bytes)
        if cursor + size > self.capacity_bytes:
            raise AllocationError(
                "unable to allocate {} bytes in capacity {}".format(size, self.capacity_bytes)
            )
        return cursor

    def allocate(
        self,
        buffer_id: str,
        size_bytes: int,
        generation: int = 0,
        *,
        alias_of: Optional[str] = None,
        address: Optional[int] = None,
        alias_offset_bytes: int = 0,
    ) -> PhysicalAllocation:
        key = self._key(buffer_id, generation)
        if type(size_bytes) is not int or size_bytes <= 0:
            raise AllocationError("size_bytes must be a positive integer")
        existing = self._allocations.get(key)
        if existing is not None:
            if size_bytes > existing.size_bytes:
                if existing.alias_of is not None:
                    raise AllocationError("aliased allocation {} cannot grow".format(buffer_id))
                self._check_extent(existing.base_address, size_bytes)
                if tuple(self._overlaps(existing.base_address, size_bytes, ignore=key)):
                    raise AllocationError(
                        "allocation {} generation {} cannot grow without moving".format(*key)
                    )
                existing = PhysicalAllocation(
                    existing.buffer_id, existing.base_address, size_bytes,
                    existing.generation, existing.alias_of,
                )
                self._allocations[key] = existing
            if address is not None and address != existing.base_address:
                raise AllocationError("allocation address changed for {}".format(buffer_id))
            return existing

        alias_key = None
        base: int
        if alias_of is not None:
            alias_name = str(alias_of).strip()
            if not alias_name:
                raise AllocationError("alias_of must be non-empty text")
            candidates = [item for (name, _gen), item in self._allocations.items() if name == alias_name]
            if not candidates:
                raise AllocationError("alias target {} is not allocated".format(alias_name))
            target = max(candidates, key=lambda item: item.generation)
            if type(alias_offset_bytes) is not int or alias_offset_bytes < 0:
                raise AllocationError("alias_offset_bytes must be non-negative")
            if alias_offset_bytes + size_bytes > target.size_bytes:
                raise AllocationError("alias range exceeds target allocation {}".format(alias_name))
            base = target.base_address + alias_offset_bytes
            alias_key = alias_name
            self._check_extent(base, size_bytes)
        elif address is not None:
            if type(address) is not int or address < 0:
                raise AllocationError("address must be a non-negative integer")
            base = address
            self._check_extent(base, size_bytes)
            conflicts = tuple(self._overlaps(base, size_bytes))
            if conflicts:
                raise AllocationError(
                    "allocation {} overlaps {} without an explicit alias".format(
                        buffer_id, conflicts[0].buffer_id
                    )
                )
        else:
            base = self._find_first_fit(size_bytes)

        allocation = PhysicalAllocation(buffer_id=key[0], base_address=base, size_bytes=size_bytes,
                                        generation=key[1], alias_of=alias_key)
        self._allocations[key] = allocation
        return allocation

    def register(self, declaration: dict) -> PhysicalAllocation:
        """Register one descriptor declaration without accepting hidden overlap."""
        if not isinstance(declaration, dict):
            raise AllocationError("physical allocation declaration must be a mapping")
        buffer_id = declaration.get("buffer_id")
        size = declaration.get("size_bytes", declaration.get("allocation_size_bytes"))
        if size is None:
            raise AllocationError("physical allocation requires size_bytes")
        return self.allocate(
            str(buffer_id), int(size), int(declaration.get("generation", declaration.get("allocation_generation", 0))),
            alias_of=declaration.get("alias_of"),
            address=declaration.get("base_address"),
            alias_offset_bytes=int(declaration.get("alias_offset_bytes", 0)),
        )

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
        self._allocations.pop(self._key(buffer_id, generation), None)

    def snapshot(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
            "allocations": tuple(self.allocations()),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "PhysicalAddressAllocator":
        allocator = cls(snapshot["capacity_bytes"], snapshot["alignment_bytes"])
        for item in snapshot.get("allocations", ()):
            if isinstance(item, PhysicalAllocation):
                allocator._allocations[(item.buffer_id, item.generation)] = item
            else:
                allocator.register(dict(item))
        return allocator


__all__ = ["AllocationError", "PhysicalAllocation", "PhysicalAddressAllocator"]
