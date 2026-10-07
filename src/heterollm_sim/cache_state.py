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
LineKey = Tuple[str, int, int]


class CacheStateError(ValueError):
    """Raised when an explicit cache access or state transition is invalid."""


@dataclass(frozen=True)
class CacheAccess:
    """One explicit byte-range access to a named backing buffer.

    ``buffer_size_bytes`` is optional.  When supplied on the first access for
    a buffer, it bounds the final physical cache line; later accesses must use
    the same size when they supply one.  Without it, a touched line occupies a
    full ``line_bytes`` in the cache and on dirty eviction.

    ``allocation_generation`` separates reused allocator identities.  A line
    from generation ``n`` can never hit a line from generation ``n + 1``.
    """

    buffer_id: str
    offset_bytes: int
    size_bytes: int
    operation: Operation
    buffer_size_bytes: Optional[int] = None
    allocation_generation: int = 0

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
        if (
            isinstance(self.allocation_generation, bool)
            or not isinstance(self.allocation_generation, int)
            or self.allocation_generation < 0
        ):
            raise CacheStateError(
                "allocation_generation must be a non-negative integer"
            )
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
        allocation_generation: int = 0,
    ) -> "CacheAccess":
        return cls(
            buffer_id,
            offset_bytes,
            size_bytes,
            "read",
            buffer_size_bytes,
            allocation_generation,
        )

    @classmethod
    def write(
        cls,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
        allocation_generation: int = 0,
    ) -> "CacheAccess":
        return cls(
            buffer_id,
            offset_bytes,
            size_bytes,
            "write",
            buffer_size_bytes,
            allocation_generation,
        )


@dataclass(frozen=True)
class CacheLine:
    """A resident line snapshot."""

    buffer_id: str
    line_index: int
    size_bytes: int
    dirty: bool
    allocation_generation: int = 0

    @property
    def key(self) -> LineKey:
        return (self.buffer_id, self.allocation_generation, self.line_index)


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
    backing_accesses: Tuple[CacheAccess, ...]

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
        # OrderedDict does not expose a constant-time successor lookup.  Keep
        # a tiny side index so transaction journaling stays O(touched lines),
        # even when the cache holds a large resident working set.
        self._lru_prev = {}
        self._lru_next = {}
        self._buffer_line_keys: Dict[Tuple[str, int], set[LineKey]] = {}
        self._buffer_sizes: Dict[Tuple[str, int], Optional[int]] = {}
        self._access_count = 0
        self._hit_lines = 0
        self._miss_lines = 0
        self._eviction_lines = 0
        self._dirty_eviction_bytes = 0
        self._flush_writeback_bytes = 0
        self._write_through_bytes = 0
        self._bypass_write_bytes = 0
        self._read_fill_bytes = 0
        self._transaction = None

    def begin_transaction(self):
        """Begin a cheap, line-level rollback journal.

        A cache may contain hundreds of thousands of lines.  Copying the
        complete ``OrderedDict`` for every event makes a stateful simulation
        quadratic in both time and peak memory.  The journal records only
        lines and counters touched by the current access; rollback is used
        only on an exceptional dispatch path.
        """
        if self._transaction is not None:
            raise CacheStateError("cache transaction already active")
        self._transaction = {
            "lines": [],
            "line_seen": set(),
            "buffers": [],
            "buffer_seen": set(),
            "scalars": {
                name: getattr(self, name)
                for name in (
                    "_access_count", "_hit_lines", "_miss_lines",
                    "_eviction_lines", "_dirty_eviction_bytes",
                    "_flush_writeback_bytes", "_write_through_bytes",
                    "_bypass_write_bytes", "_read_fill_bytes",
                )
            },
        }
        return self._transaction

    def commit_transaction(self, transaction) -> None:
        if transaction is not self._transaction:
            raise CacheStateError("invalid cache transaction")
        self._transaction = None

    def rollback_transaction(self, transaction) -> None:
        if transaction is not self._transaction:
            raise CacheStateError("invalid cache transaction")
        # Undo line mutations in reverse order.  Normal dispatch never scans
        # the cache; the exceptional rollback path restores membership and
        # values and then appends restored lines to the LRU tail.  The event
        # kernel retries the rejected task, so this path is intentionally
        # bounded by the failed access rather than the full cache size.
        for key, existed, old_line, successor in reversed(transaction["lines"]):
            if existed:
                self._lines.pop(key, None)
                if successor is None or successor not in self._lines:
                    self._lines[key] = old_line
                else:
                    restored = OrderedDict()
                    for current_key, current_line in self._lines.items():
                        if current_key == successor:
                            restored[key] = old_line
                        restored[current_key] = current_line
                    self._lines = restored
            else:
                self._lines.pop(key, None)
        for key, existed, old_value in reversed(transaction["buffers"]):
            if existed:
                self._buffer_sizes[key] = old_value
            else:
                self._buffer_sizes.pop(key, None)
        for name, value in transaction["scalars"].items():
            setattr(self, name, value)
        self._rebuild_lru_links()
        self._transaction = None

    def _record_line(self, key) -> None:
        transaction = self._transaction
        if transaction is None or key in transaction["line_seen"]:
            return
        transaction["line_seen"].add(key)
        transaction["lines"].append((key, key in self._lines,
                                      self._lines.get(key), self._lru_next.get(key)))

    def _pop_line(self, key, default=None):
        self._record_line(key)
        if key not in self._lines:
            return default
        previous = self._lru_prev.pop(key, None)
        successor = self._lru_next.pop(key, None)
        if previous is not None:
            self._lru_next[previous] = successor
        if successor is not None:
            self._lru_prev[successor] = previous
        buffer_key = key[:2]
        buffer_lines = self._buffer_line_keys.get(buffer_key)
        if buffer_lines is not None:
            buffer_lines.discard(key)
            if not buffer_lines:
                self._buffer_line_keys.pop(buffer_key, None)
        return self._lines.pop(key)

    def _set_line(self, key, value) -> None:
        self._record_line(key)
        if key in self._lines:
            self._lines[key] = value
            return
        previous = next(reversed(self._lines), None)
        self._lines[key] = value
        self._buffer_line_keys.setdefault(key[:2], set()).add(key)
        self._lru_prev[key] = previous
        self._lru_next[key] = None
        if previous is not None:
            self._lru_next[previous] = key

    def _rebuild_lru_links(self) -> None:
        self._lru_prev.clear()
        self._lru_next.clear()
        self._buffer_line_keys.clear()
        previous = None
        for key in self._lines:
            self._buffer_line_keys.setdefault(key[:2], set()).add(key)
            self._lru_prev[key] = previous
            self._lru_next[key] = None
            if previous is not None:
                self._lru_next[previous] = key
            previous = key

    def _remember_buffer(self, key, value) -> None:
        transaction = self._transaction
        if transaction is None or key in transaction["buffer_seen"]:
            return
        transaction["buffer_seen"].add(key)
        if key in self._buffer_sizes:
            transaction["buffers"].append((key, True, self._buffer_sizes[key]))
        else:
            transaction["buffers"].append((key, False, None))

    @staticmethod
    def _positive_int(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CacheStateError("{} must be a positive integer".format(name))

    def _buffer_size(self, access: CacheAccess) -> Optional[int]:
        buffer_key = (access.buffer_id, access.allocation_generation)
        if buffer_key in self._buffer_sizes:
            known = self._buffer_sizes[buffer_key]
            if access.buffer_size_bytes is not None and known != access.buffer_size_bytes:
                raise CacheStateError(
                    "buffer_size_bytes must be fixed at first access for buffer {} generation {}".format(
                        access.buffer_id, access.allocation_generation
                    )
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
                ((access.buffer_id.strip(), access.allocation_generation, line_index), physical_size,
                 covered_end - covered_start, full_line)
            )
        return tuple(specs)

    def _evict_one(self) -> Tuple[CacheLine, int, int]:
        if not self._lines:
            raise CacheStateError("cache has no resident line to evict")
        _key = next(iter(self._lines))
        victim = self._pop_line(_key)
        dirty_bytes = victim.size_bytes if victim.dirty else 0
        clean_bytes = victim.size_bytes if not victim.dirty else 0
        self._eviction_lines += 1
        self._dirty_eviction_bytes += dirty_bytes
        return victim, dirty_bytes, clean_bytes

    def _ensure_slot(
        self, backing_accesses: List[CacheAccess]
    ) -> Tuple[int, int, int, int]:
        dirty_bytes = clean_bytes = dirty_lines = clean_lines = 0
        if len(self._lines) >= self.capacity_lines:
            victim, dirty, clean = self._evict_one()
            dirty_bytes += dirty
            clean_bytes += clean
            if victim.dirty:
                dirty_lines += 1
                backing_accesses.append(CacheAccess.write(
                    victim.buffer_id,
                    victim.line_index * self.line_bytes,
                    victim.size_bytes,
                    buffer_size_bytes=self._buffer_sizes[
                        (victim.buffer_id, victim.allocation_generation)
                    ],
                    allocation_generation=victim.allocation_generation,
                ))
            else:
                clean_lines += 1
        return dirty_bytes, clean_bytes, dirty_lines, clean_lines

    def access(self, access: CacheAccess) -> CacheAccessResult:
        if not isinstance(access, CacheAccess):
            raise CacheStateError("access must be a CacheAccess")
        specs = self._line_specs(access)
        # Validate the complete range before remembering size or mutating LRU.
        buffer_key = (access.buffer_id, access.allocation_generation)
        self._remember_buffer(buffer_key, access.buffer_size_bytes)
        self._buffer_sizes.setdefault(
            buffer_key,
            access.buffer_size_bytes,
        )
        hit_lines = miss_lines = allocated_lines = 0
        read_fill_bytes = read_for_ownership_bytes = 0
        write_through_bytes = bypass_write_bytes = 0
        dirty_eviction_bytes = clean_eviction_bytes = 0
        dirty_eviction_lines = clean_eviction_lines = 0
        backing_accesses: List[CacheAccess] = []

        for key, physical_size, covered_bytes, full_line in specs:
            line_offset = key[2] * self.line_bytes
            covered_offset = max(access.offset_bytes, line_offset)
            buffer_size = self._buffer_sizes[
                (access.buffer_id, access.allocation_generation)
            ]
            line = self._pop_line(key, None)
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
                            line.allocation_generation,
                        )
                    else:
                        write_through_bytes += covered_bytes
                        self._write_through_bytes += covered_bytes
                        backing_accesses.append(CacheAccess.write(
                            access.buffer_id, covered_offset, covered_bytes,
                            buffer_size_bytes=buffer_size,
                            allocation_generation=access.allocation_generation,
                        ))
                self._set_line(key, line)
                continue

            miss_lines += 1
            self._miss_lines += 1
            if access.operation == "write" and not self.write_allocate:
                bypass_write_bytes += covered_bytes
                self._bypass_write_bytes += covered_bytes
                backing_accesses.append(CacheAccess.write(
                    access.buffer_id, covered_offset, covered_bytes,
                    buffer_size_bytes=buffer_size,
                    allocation_generation=access.allocation_generation,
                ))
                continue

            dirty, clean, dirty_lines, clean_lines = self._ensure_slot(backing_accesses)
            dirty_eviction_bytes += dirty
            clean_eviction_bytes += clean
            dirty_eviction_lines += dirty_lines
            clean_eviction_lines += clean_lines

            # A full-line write can allocate the line without a read-for-
            # ownership transfer.  Partial writes need a line fill first.
            if access.operation == "read" or not full_line:
                read_fill_bytes += physical_size
                self._read_fill_bytes += physical_size
                backing_accesses.append(CacheAccess.read(
                    access.buffer_id, line_offset, physical_size,
                    buffer_size_bytes=buffer_size,
                    allocation_generation=access.allocation_generation,
                ))
                if access.operation == "write":
                    read_for_ownership_bytes += physical_size

            dirty_line = access.operation == "write" and self.write_back
            self._set_line(key, CacheLine(
                key[0], key[2], physical_size, dirty_line, key[1]
            ))
            allocated_lines += 1
            if access.operation == "write" and not self.write_back:
                write_through_bytes += covered_bytes
                self._write_through_bytes += covered_bytes
                backing_accesses.append(CacheAccess.write(
                    access.buffer_id, covered_offset, covered_bytes,
                    buffer_size_bytes=buffer_size,
                    allocation_generation=access.allocation_generation,
                ))

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
            backing_accesses=tuple(backing_accesses),
        )

    def access_many(
        self, accesses: Iterable[CacheAccess]
    ) -> Tuple[CacheAccessResult, ...]:
        # Validate the entire batch before any line, counter, or remembered
        # buffer-size mutation. Materialize generators once, so a late invalid
        # access cannot leave a partially applied kernel memory transaction.
        accesses = tuple(accesses)
        sizes = {}
        for access in accesses:
            if not isinstance(access, CacheAccess):
                raise CacheStateError("access must be a CacheAccess")
            buffer_key = (access.buffer_id, access.allocation_generation)
            if buffer_key not in sizes:
                if buffer_key in self._buffer_sizes:
                    sizes[buffer_key] = self._buffer_sizes[buffer_key]
                else:
                    sizes[buffer_key] = access.buffer_size_bytes
            size = sizes[buffer_key]
            if access.buffer_size_bytes is not None and size != access.buffer_size_bytes:
                raise CacheStateError(
                    "buffer_size_bytes must be fixed at first access for buffer {} generation {}".format(
                        access.buffer_id, access.allocation_generation
                    )
                )
            if size is not None and access.offset_bytes + access.size_bytes > size:
                raise CacheStateError("access range exceeds remembered buffer_size_bytes")
        return tuple(self.access(access) for access in accesses)

    def read(
        self,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
        allocation_generation: int = 0,
    ) -> CacheAccessResult:
        return self.access(
            CacheAccess.read(
                buffer_id,
                offset_bytes,
                size_bytes,
                buffer_size_bytes=buffer_size_bytes,
                allocation_generation=allocation_generation,
            )
        )

    def write(
        self,
        buffer_id: str,
        offset_bytes: int,
        size_bytes: int,
        *,
        buffer_size_bytes: Optional[int] = None,
        allocation_generation: int = 0,
    ) -> CacheAccessResult:
        return self.access(
            CacheAccess.write(
                buffer_id,
                offset_bytes,
                size_bytes,
                buffer_size_bytes=buffer_size_bytes,
                allocation_generation=allocation_generation,
            )
        )

    def discard_buffer(self, buffer_id: str, allocation_generation: int = 0) -> None:
        """Invalidate a freed allocation without writing dead data back.

        Allocation lifetime belongs to the physical event kernel.  Retaining
        dirty lines after it frees their backing buffer would make a future
        eviction recreate an invalid physical destination.
        """

        key = (buffer_id, allocation_generation)
        for line_key in tuple(self._buffer_line_keys.get(key, ())):
            self._pop_line(line_key)
        self._remember_buffer(key, None)
        self._buffer_sizes.pop(key, None)

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
                line.allocation_generation,
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
