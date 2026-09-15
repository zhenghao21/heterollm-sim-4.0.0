"""Explicit opt-in stateful on-chip cache model.

This module is deliberately separate from HBM/KV residency.  It models only
an explicitly addressed cache working set: callers must provide a stable
``buffer_id`` and byte range for every access.  Callers that do not have those
facts should keep using the aggregate closed-interval model in
:mod:`heterollm_sim.cost_models`.

The model is line-granular and deterministic.  It does not infer addresses
from model names, operator names, or measured latency.  A cache line key is
``(buffer_id, line_index)`` so two buffers never alias merely because their
local offsets match.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple


Operation = Literal["read", "write"]
LineKey = Tuple[str, int]


class CacheStateError(ValueError):
    """Raised when an explicit cache access or state transition is invalid."""


@dataclass(frozen=True)
class CacheAccess:
    """One explicit byte-range access to a named backing buffer.

    ``buffer_size_bytes`` is optional.  When supplied on the first access for
    a buffer, it bounds the final physical cache line; later accesses must use
    the same size when they supply one.  Without it, a touched line occupies a
    full ``line_bytes`` in the cache and on dirty eviction.
    """

    buffer_id: str
    offset_bytes: int
    size_bytes: int
    operation: Operation
    buffer_size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.buffer_id, str) or not self.buffer_id.strip():
            raise CacheStateError("buffer_id must be non-empty text")
        object.__setattr__(self, "buffer_id", self.buffer_id.strip())
        if (
            isinstance(self.offset_bytes, bool)
            or not isinstance(self.offset_bytes, int)
            or self.offset_bytes < 0
        ):
            raise CacheStateError("offset_bytes must be a non-negative integer")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise CacheStateError("size_bytes must be a positive integer")
        operation = str(self.operation).strip().lower()
        if operation not in {"read", "write"}:
            raise CacheStateError("operation must be read or write")
        object.__setattr__(self, "operation", operation)
        if self.buffer_size_bytes is not None:
            if (
                isinstance(self.buffer_size_bytes, bool)
                or not isinstance(self.buffer_size_bytes, int)
                or self.buffer_size_bytes <= 0
            ):
                raise CacheStateError(
                    "buffer_size_bytes must be a positive integer or None"
                )
            if self.offset_bytes + self.size_bytes > self.buffer_size_bytes:
                raise CacheStateError(
                    "access range exceeds declared buffer_size_bytes"
                )

    @classmethod
    def read(
        cls,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
    ) -> "CacheAccess":
        return cls(
            buffer_id,
            offset_bytes,
            size_bytes,
            "read",
            buffer_size_bytes,
        )

    @classmethod
    def write(
        cls,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
    ) -> "CacheAccess":
        return cls(
            buffer_id,
            offset_bytes,
            size_bytes,
            "write",
            buffer_size_bytes,
        )


@dataclass(frozen=True)
class CacheLine:
    """A resident line snapshot."""

    buffer_id: str
    line_index: int
    size_bytes: int
    dirty: bool

    @property
    def key(self) -> LineKey:
        return (self.buffer_id, self.line_index)


@dataclass(frozen=True)
class CacheAccessResult:
    """Accounting produced by one stateful access."""

    access: CacheAccess
    touched_lines: int
    hit_lines: int
    miss_lines: int
    allocated_lines: int
    read_fill_bytes: int
    read_for_ownership_bytes: int
    write_through_bytes: int
    bypass_write_bytes: int
    dirty_eviction_bytes: int
    clean_eviction_bytes: int
    dirty_eviction_lines: int
    clean_eviction_lines: int

    @property
    def backing_read_bytes(self) -> int:
        return self.read_fill_bytes

    @property
    def backing_write_bytes(self) -> int:
        return (
            self.write_through_bytes
            + self.bypass_write_bytes
            + self.dirty_eviction_bytes
        )

    @property
    def evicted_lines(self) -> int:
        return self.dirty_eviction_lines + self.clean_eviction_lines

    @property
    def requested_bytes(self) -> int:
        return self.access.size_bytes


@dataclass(frozen=True)
class CacheFlushResult:
    """Accounting produced by flushing dirty resident lines."""

    buffer_id: Optional[str]
    flushed_lines: int
    writeback_bytes: int

    @property
    def backing_write_bytes(self) -> int:
        return self.writeback_bytes


@dataclass(frozen=True)
class CacheStateSnapshot:
    """Serializable state summary for evidence and cross-invocation replay."""

    capacity_bytes: int
    line_bytes: int
    capacity_lines: int
    resident_lines: int
    resident_bytes: int
    dirty_lines: int
    dirty_bytes: int
    access_count: int
    hit_lines: int
    miss_lines: int
    eviction_lines: int
    dirty_eviction_bytes: int
    flush_writeback_bytes: int
    write_through_bytes: int
    bypass_write_bytes: int
    read_fill_bytes: int
    buffers: Tuple[str, ...]

    @property
    def backing_write_bytes(self) -> int:
        return (
            self.dirty_eviction_bytes + self.flush_writeback_bytes
            + self.write_through_bytes + self.bypass_write_bytes
        )


class ExplicitCacheState:
    """Finite, fully associative line-granular LRU cache with explicit write policy.

    The instance is persistent: calling :meth:`access` repeatedly preserves
    line residency and recency, which is the only supported cross-invocation
    cache state.  It is independent of any physical HBM/KV residency manager.
    """

    def __init__(
        self,
        capacity_bytes: int,
        line_bytes: int,
        *,
        write_back: bool = True,
        write_allocate: bool = True,
    ) -> None:
        self._positive_int(capacity_bytes, "capacity_bytes")
        self._positive_int(line_bytes, "line_bytes")
        if capacity_bytes < line_bytes:
            raise CacheStateError("capacity_bytes must fit at least one line")
        if capacity_bytes % line_bytes:
            raise CacheStateError(
                "capacity_bytes must be an integer multiple of line_bytes"
            )
        if not isinstance(write_back, bool):
            raise CacheStateError("write_back must be boolean")
        if not isinstance(write_allocate, bool):
            raise CacheStateError("write_allocate must be boolean")
        self.capacity_bytes = capacity_bytes
        self.line_bytes = line_bytes
        self.capacity_lines = capacity_bytes // line_bytes
        self.write_back = write_back
        self.write_allocate = write_allocate
        self._lines: "OrderedDict[LineKey, CacheLine]" = OrderedDict()
        self._buffer_sizes: Dict[str, Optional[int]] = {}
        self._access_count = 0
        self._hit_lines = 0
        self._miss_lines = 0
        self._eviction_lines = 0
        self._dirty_eviction_bytes = 0
        self._flush_writeback_bytes = 0
        self._write_through_bytes = 0
        self._bypass_write_bytes = 0
        self._read_fill_bytes = 0

    @staticmethod
    def _positive_int(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CacheStateError("{} must be a positive integer".format(name))

    def _buffer_size(self, access: CacheAccess) -> Optional[int]:
        buffer_id = access.buffer_id
        if buffer_id in self._buffer_sizes:
            known = self._buffer_sizes[buffer_id]
            if access.buffer_size_bytes is not None and known != access.buffer_size_bytes:
                raise CacheStateError(
                    "buffer_size_bytes must be fixed at first access for buffer {}".format(buffer_id)
                )
        else:
            known = access.buffer_size_bytes
        if known is not None and access.offset_bytes + access.size_bytes > known:
            raise CacheStateError("access range exceeds remembered buffer_size_bytes")
        return known

    def _line_specs(
        self, access: CacheAccess
    ) -> Tuple[Tuple[LineKey, int, int, bool], ...]:
        size = self._buffer_size(access)
        start = access.offset_bytes
        end = start + access.size_bytes
        first = start // self.line_bytes
        last = (end - 1) // self.line_bytes
        specs: List[Tuple[LineKey, int, int, bool]] = []
        for line_index in range(first, last + 1):
            line_start = line_index * self.line_bytes
            line_end = line_start + self.line_bytes
            if size is not None:
                line_end = min(line_end, size)
            physical_size = line_end - line_start
            if size is not None and line_start >= size:
                raise CacheStateError("access resolved beyond declared buffer")
            if physical_size <= 0:
                raise CacheStateError("access resolved to an empty cache line")
            covered_start = max(start, line_start)
            covered_end = min(end, line_end)
            if covered_end <= covered_start:
                raise CacheStateError("access range does not cover cache line")
            full_line = (
                covered_start == line_start and covered_end == line_end
            )
            specs.append(
                ((access.buffer_id.strip(), line_index), physical_size,
                 covered_end - covered_start, full_line)
            )
        return tuple(specs)

    def _evict_one(self) -> Tuple[CacheLine, int, int]:
        if not self._lines:
            raise CacheStateError("cache has no resident line to evict")
        _key, victim = self._lines.popitem(last=False)
        dirty_bytes = victim.size_bytes if victim.dirty else 0
        clean_bytes = victim.size_bytes if not victim.dirty else 0
        self._eviction_lines += 1
        self._dirty_eviction_bytes += dirty_bytes
        return victim, dirty_bytes, clean_bytes

    def _ensure_slot(self) -> Tuple[int, int, int, int]:
        dirty_bytes = clean_bytes = dirty_lines = clean_lines = 0
        if len(self._lines) >= self.capacity_lines:
            victim, dirty, clean = self._evict_one()
            dirty_bytes += dirty
            clean_bytes += clean
            if victim.dirty:
                dirty_lines += 1
            else:
                clean_lines += 1
        return dirty_bytes, clean_bytes, dirty_lines, clean_lines

    def access(self, access: CacheAccess) -> CacheAccessResult:
        if not isinstance(access, CacheAccess):
            raise CacheStateError("access must be a CacheAccess")
        specs = self._line_specs(access)
        # Validate the complete range before remembering size or mutating LRU.
        self._buffer_sizes.setdefault(access.buffer_id, access.buffer_size_bytes)
        hit_lines = miss_lines = allocated_lines = 0
        read_fill_bytes = read_for_ownership_bytes = 0
        write_through_bytes = bypass_write_bytes = 0
        dirty_eviction_bytes = clean_eviction_bytes = 0
        dirty_eviction_lines = clean_eviction_lines = 0

        for key, physical_size, covered_bytes, full_line in specs:
            line = self._lines.pop(key, None)
            if line is not None:
                hit_lines += 1
                self._hit_lines += 1
                if access.operation == "write":
                    if self.write_back:
                        line = CacheLine(
                            line.buffer_id,
                            line.line_index,
                            line.size_bytes,
                            True,
                        )
                    else:
                        write_through_bytes += covered_bytes
                        self._write_through_bytes += covered_bytes
                self._lines[key] = line
                continue

            miss_lines += 1
            self._miss_lines += 1
            if access.operation == "write" and not self.write_allocate:
                bypass_write_bytes += covered_bytes
                self._bypass_write_bytes += covered_bytes
                continue

            dirty, clean, dirty_lines, clean_lines = self._ensure_slot()
            dirty_eviction_bytes += dirty
            clean_eviction_bytes += clean
            dirty_eviction_lines += dirty_lines
            clean_eviction_lines += clean_lines

            # A full-line write can allocate the line without a read-for-
            # ownership transfer.  Partial writes need a line fill first.
            if access.operation == "read" or not full_line:
                read_fill_bytes += physical_size
                self._read_fill_bytes += physical_size
                if access.operation == "write":
                    read_for_ownership_bytes += physical_size

            dirty_line = access.operation == "write" and self.write_back
            self._lines[key] = CacheLine(
                key[0], key[1], physical_size, dirty_line
            )
            allocated_lines += 1
            if access.operation == "write" and not self.write_back:
                write_through_bytes += covered_bytes
                self._write_through_bytes += covered_bytes

        self._access_count += 1
        return CacheAccessResult(
            access=access,
            touched_lines=len(specs),
            hit_lines=hit_lines,
            miss_lines=miss_lines,
            allocated_lines=allocated_lines,
            read_fill_bytes=read_fill_bytes,
            read_for_ownership_bytes=read_for_ownership_bytes,
            write_through_bytes=write_through_bytes,
            bypass_write_bytes=bypass_write_bytes,
            dirty_eviction_bytes=dirty_eviction_bytes,
            clean_eviction_bytes=clean_eviction_bytes,
            dirty_eviction_lines=dirty_eviction_lines,
            clean_eviction_lines=clean_eviction_lines,
        )

    def access_many(
        self, accesses: Iterable[CacheAccess]
    ) -> Tuple[CacheAccessResult, ...]:
        return tuple(self.access(access) for access in accesses)

    def read(
        self,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
    ) -> CacheAccessResult:
        return self.access(
            CacheAccess.read(
                buffer_id,
                offset_bytes,
                size_bytes,
                buffer_size_bytes=buffer_size_bytes,
            )
        )

    def write(
        self,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
    ) -> CacheAccessResult:
        return self.access(
            CacheAccess.write(
                buffer_id,
                offset_bytes,
                size_bytes,
                buffer_size_bytes=buffer_size_bytes,
            )
        )

    def flush(self, buffer_id: Optional[str] = None) -> CacheFlushResult:
        if buffer_id is not None and (
            not isinstance(buffer_id, str) or not buffer_id.strip()
        ):
            raise CacheStateError("buffer_id must be non-empty text or None")
        normalized = None if buffer_id is None else buffer_id.strip()
        flushed_lines = 0
        writeback_bytes = 0
        for key, line in tuple(self._lines.items()):
            if normalized is not None and key[0] != normalized:
                continue
            if not line.dirty:
                continue
            flushed_lines += 1
            writeback_bytes += line.size_bytes
            self._lines[key] = CacheLine(
                line.buffer_id,
                line.line_index,
                line.size_bytes,
                False,
            )
        self._flush_writeback_bytes += writeback_bytes
        return CacheFlushResult(normalized, flushed_lines, writeback_bytes)

    def resident_lines(self) -> Tuple[CacheLine, ...]:
        """Return lines in LRU-to-MRU order."""
        return tuple(self._lines.values())

    def snapshot(self) -> CacheStateSnapshot:
        dirty_lines = sum(1 for line in self._lines.values() if line.dirty)
        dirty_bytes = sum(
            line.size_bytes for line in self._lines.values() if line.dirty
        )
        resident_bytes = sum(line.size_bytes for line in self._lines.values())
        return CacheStateSnapshot(
            capacity_bytes=self.capacity_bytes,
            line_bytes=self.line_bytes,
            capacity_lines=self.capacity_lines,
            resident_lines=len(self._lines),
            resident_bytes=resident_bytes,
            dirty_lines=dirty_lines,
            dirty_bytes=dirty_bytes,
            access_count=self._access_count,
            hit_lines=self._hit_lines,
            miss_lines=self._miss_lines,
            eviction_lines=self._eviction_lines,
            dirty_eviction_bytes=self._dirty_eviction_bytes,
            flush_writeback_bytes=self._flush_writeback_bytes,
            write_through_bytes=self._write_through_bytes,
            bypass_write_bytes=self._bypass_write_bytes,
            read_fill_bytes=self._read_fill_bytes,
            buffers=tuple(sorted({line.buffer_id for line in self._lines.values()})),
        )


__all__ = [
    "CacheAccess",
    "CacheAccessResult",
    "CacheFlushResult",
    "CacheLine",
    "CacheStateError",
    "CacheStateSnapshot",
    "ExplicitCacheState",
]
