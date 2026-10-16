"""Finite HOST DRAM staging and out-of-place mapping for strict OCP HBF.

This is a simulator state machine, not an early-ack buffer inside NandCore.
Every byte moved goes through the supplied DRAM, link and NAND callbacks.
Only complete pages reach NAND; unpublished tail bytes remain readable in
HOST DRAM. One host page is reserved for GC and one physical block for GC.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from functools import cached_property
from itertools import repeat
from threading import Lock
from typing import Callable, Hashable

import numpy as np

from .memory_types import NandConfig


PAGE = 4096
Key = tuple[Hashable, int, int]
Range = tuple[int, int]
MemoryIO = Callable[[int, int, float], float]


@dataclass(frozen=True)
class HbfStagingIO:
    read_host: MemoryIO
    write_host: MemoryIO
    read_flash: MemoryIO
    write_flash: MemoryIO
    transfer: Callable[[str, int, float], float]


def _merge(ranges: list[Range]) -> list[Range]:
    result: list[Range] = []
    for lo, hi in sorted(ranges):
        if lo == hi:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = result[-1][0], max(hi, result[-1][1])
        else:
            result.append((lo, hi))
    return result


def _subtract(ranges: list[Range], removed: list[Range]) -> list[Range]:
    result = list(ranges)
    for a, b in removed:
        parts = []
        for lo, hi in result:
            if b <= lo or a >= hi:
                parts.append((lo, hi))
            else:
                if lo < a:
                    parts.append((lo, a))
                if b < hi:
                    parts.append((b, hi))
        result = parts
    return result


def _intersection(ranges: list[Range], lo: int, hi: int) -> list[Range]:
    return [(max(a, lo), min(b, hi)) for a, b in ranges if a < hi and b > lo]


@dataclass(slots=True, eq=False)
class _Page:
    valid: list[Range] = field(default_factory=list)
    dirty: list[Range] = field(default_factory=list)
    slot: int | None = None
    flash: tuple[int, int] | None = None  # physical block id, page within block
    _ready_ns: float = 0.0
    _read_state: object = field(default=None, repr=False, compare=False)
    _read_index: int = field(default=0, repr=False, compare=False)

    @property
    def ready_ns(self):
        state = self._read_state
        return self._ready_ns if state is None else float(state.ready[self._read_index])

    @ready_ns.setter
    def ready_ns(self, value):
        if self._read_state is None:
            self._ready_ns = value
        else:
            self._read_state.ready[self._read_index] = value

    def __eq__(self, other):
        if not isinstance(other, _Page):
            return NotImplemented
        return (self.valid, self.dirty, self.slot, self.flash, self.ready_ns) == (
            other.valid, other.dirty, other.slot, other.flash, other.ready_ns)


class _BufferReadState:
    """Dense page metadata and authoritative read dependencies for one buffer.

    Clean KV pages remain numeric across Decode tokens. Dirty/partial pages
    still use the ordinary HOST/flash merge path, and every scalar operation
    sees the same ready value through its page object.
    """

    __slots__ = ("ready", "valid", "addresses", "plans")

    def __init__(self, size):
        capacity = 1 << (size - 1).bit_length()
        self.ready = np.zeros(capacity)
        self.valid = np.zeros(capacity, dtype=np.bool_)
        self.addresses = np.full(capacity, -1, dtype=np.int64)
        self.plans = OrderedDict()

    def ensure(self, index):
        if index < len(self.ready):
            return
        capacity = 1 << index.bit_length()
        extra = capacity - len(self.ready)
        self.ready = np.pad(self.ready, (0, extra))
        self.valid = np.pad(self.valid, (0, extra))
        self.addresses = np.pad(self.addresses, (0, extra), constant_values=-1)

    def update(self, index, page, address):
        self.ensure(index)
        full = page.valid == [(0, PAGE)]
        clean_address = address if full and not page.dirty and page.slot is None else -1
        if self.addresses[index] != clean_address or self.valid[index] != full:
            for bounds in tuple(self.plans):
                if bounds[0] <= index < bounds[1]:
                    self.plans.pop(bounds)
            self.addresses[index] = clean_address
            self.valid[index] = full
        if page._read_state is not self:
            self.ready[index] = page.ready_ns
            page._read_state, page._read_index = self, index

    def plan(self, first, stop, lanes, pages_per_block):
        key = first, stop
        plan = self.plans.get(key)
        if plan is not None:
            self.plans.move_to_end(key)
            return plan
        addresses = np.frombuffer(self.addresses[first:stop].tobytes(), dtype=np.int64)
        physical = addresses // PAGE
        block_ids = physical // (pages_per_block * lanes) * lanes + physical % lanes
        blocks, inverse = np.unique(block_ids, return_inverse=True)
        plan = addresses, tuple(blocks.tolist()), inverse
        if len(self.plans) >= 4:
            self.plans.popitem(last=False)
        self.plans[key] = plan
        return plan


@dataclass(frozen=True)
class _Preload:
    byte_count: int
    block_ranges: tuple[range, ...]

    @cached_property
    def block_count(self) -> int:
        return sum(len(span) for span in self.block_ranges)

    def block_at(self, index: int) -> int:
        for span in self.block_ranges:
            if index < len(span):
                return span[index]
            index -= len(span)
        raise IndexError(index)

    def blocks(self):
        for span in self.block_ranges:
            yield from span

    def page_location(self, index: int, lanes: int, pages_per_block: int) -> tuple[int, int]:
        group, within = divmod(index, lanes * pages_per_block)
        first_block = group * lanes
        width = min(lanes, self.block_count - first_block)
        page, lane = divmod(within, width)
        return self.block_at(first_block + lane), page

    def __deepcopy__(self, memo):
        # Loaded weights and their range map never mutate. Snapshots share
        # this descriptor even when it represents tens of millions of pages.
        return self


# This cache contains derived, immutable address descriptions only. It never
# holds controller clocks, residency, program frontiers or other live state.
# A byte limit also bounds plans for large models with many distinct weights.
_PRELOAD_READ_PLANS = OrderedDict()
_PRELOAD_READ_PLAN_BYTES = 0
_PRELOAD_READ_PLAN_LIMIT = 64 * 1024 * 1024
_PRELOAD_READ_PLAN_LOCK = Lock()


def _preload_read_plan(preload, lanes, pages_per_block, first, stop):
    global _PRELOAD_READ_PLAN_BYTES
    key = (preload, lanes, pages_per_block, first, stop)
    with _PRELOAD_READ_PLAN_LOCK:
        plan = _PRELOAD_READ_PLANS.get(key)
        if plan is not None:
            _PRELOAD_READ_PLANS.move_to_end(key)
            return plan
    indexes = np.arange(first, stop, dtype=np.int64)
    group, within = np.divmod(indexes, lanes * pages_per_block)
    first_block = group * lanes
    width = np.minimum(lanes, preload.block_count - first_block)
    page, lane = np.divmod(within, width)
    ranks = first_block + lane
    unique, inverse = np.unique(ranks, return_inverse=True)
    blocks = tuple(preload.block_at(int(rank)) for rank in unique)
    physical = np.asarray(blocks, dtype=np.int64)[inverse]
    addresses = ((physical // lanes * pages_per_block * lanes
                  + page * lanes + physical % lanes) * PAGE)
    # Bytes-backed arrays cannot be made writeable by a caller retaining a
    # view. Repeated Decode accesses reuse these exact validated addresses.
    addresses = np.frombuffer(addresses.tobytes(), dtype=np.int64)
    inverse = np.frombuffer(inverse.tobytes(), dtype=np.int64)
    plan = addresses, blocks, inverse
    size = addresses.nbytes + inverse.nbytes + len(blocks) * 8
    if size <= _PRELOAD_READ_PLAN_LIMIT:
        with _PRELOAD_READ_PLAN_LOCK:
            # Another scenario may have prepared the same immutable stream
            # while NumPy released the GIL; do not count that plan twice.
            previous = _PRELOAD_READ_PLANS.get(key)
            if previous is not None:
                _PRELOAD_READ_PLANS.move_to_end(key)
                return previous
            while _PRELOAD_READ_PLANS and _PRELOAD_READ_PLAN_BYTES + size > _PRELOAD_READ_PLAN_LIMIT:
                _, previous = _PRELOAD_READ_PLANS.popitem(last=False)
                _PRELOAD_READ_PLAN_BYTES -= (previous[0].nbytes + previous[2].nbytes
                                            + len(previous[1]) * 8)
            _PRELOAD_READ_PLANS[key] = plan
            _PRELOAD_READ_PLAN_BYTES += size
    return plan


class _BlockPool:
    """Free physical blocks as intervals, never a device-sized Python queue."""

    def __init__(self, start: int, stop: int):
        self.ranges = deque([range(start, stop)])
        self.count = stop - start

    def __len__(self):
        return self.count

    def __eq__(self, other):
        if not isinstance(other, _BlockPool):
            return NotImplemented
        return self.count == other.count and self.ranges == other.ranges

    def take(self, count: int) -> tuple[range, ...]:
        if count > self.count:
            raise IndexError("insufficient free blocks")
        result = []
        self.count -= count
        while count:
            span = self.ranges.popleft()
            taken = min(count, len(span))
            result.append(range(span.start, span.start + taken))
            if taken < len(span):
                self.ranges.appendleft(range(span.start + taken, span.stop))
            count -= taken
        return tuple(result)

    def popleft(self) -> int:
        return self.take(1)[0].start

    def extend_ranges(self, spans: tuple[range, ...]) -> None:
        self.ranges.extend(spans)
        self.count += sum(len(span) for span in spans)

    def copy(self):
        result = object.__new__(type(self))
        result.ranges = self.ranges.copy()
        result.count = self.count
        return result


_MISSING = object()
_JOURNALED_MAPS = ("_used", "_slot_ready", "_block_ready")


class _JournalDict(dict):
    """Scalar dependencies/frontiers with first-write transaction journals."""

    __slots__ = ("_journals",)

    def __init__(self, values=()):
        super().__init__(values)
        self._journals = []

    def begin(self):
        token = {"values": {}, "order": None}
        self._journals.append(token)
        return token

    def commit(self, token):
        assert self._journals[-1] is token
        self._journals.pop()

    def rollback(self, token):
        self.commit(token)
        for key, value in token["values"].items():
            if value is _MISSING:
                dict.pop(self, key, None)
            else:
                dict.__setitem__(self, key, value)
        if token["order"] is not None:
            # Deletion is rare (GC), but a restored program-frontier mapping
            # must retain insertion order because it breaks victim ties.
            items = [(key, self[key]) for key in token["order"] if key in self]
            dict.clear(self)
            dict.update(self, items)

    def _remember(self, key, *, deleting=False):
        for token in self._journals:
            values = token["values"]
            if key not in values:
                values[key] = self.get(key, _MISSING)
            if deleting and token["order"] is None:
                token["order"] = tuple(self)

    def __setitem__(self, key, value):
        self._remember(key)
        dict.__setitem__(self, key, value)

    def update_max(self, keys, values):
        """Commit a unique block stream without a Python setter per block."""
        previous = list(map(self.get, keys, repeat(_MISSING)))
        for token in self._journals:
            saved = token["values"]
            if not saved:
                saved.update(zip(keys, previous))
            else:
                saved.update((key, value) for key, value in zip(keys, previous) if key not in saved)
        old_ready = np.asarray(
            [0.0 if value is _MISSING else value for value in previous]
            if _MISSING in previous else previous, dtype=np.float64)
        # max(new, old) retains new on a tie, including signed zero.
        ready = np.where(np.greater(old_ready, values), old_ready, values).tolist()
        dict.update(self, zip(keys, ready))

    def pop(self, key, default=_MISSING):
        if key in self:
            self._remember(key, deleting=True)
            return dict.pop(self, key)
        if default is _MISSING:
            raise KeyError(key)
        return default


def _copy_controller_state(state, *, include_pages=True):
    """Copy mutable indexes once, sharing only immutable keys and descriptors.

    Rollback does not need to recursively reconstruct every buffer identity,
    range tuple, configuration field and page dataclass. All mutable page
    fields and containers still receive independent copies, including when
    the same saved state is restored more than once.
    """
    result = {key: value for key, value in state.items()
              if key not in {"_pages", "_buffer_pages", "_owners", "_transaction_stack",
                             "_read_states", "_slots", *_JOURNALED_MAPS}}
    for name in ("_preloads", "stats"):
        result[name] = state[name].copy()
    if include_pages:
        for name in _JOURNALED_MAPS:
            result[name] = state[name].copy()
        result["_pages"] = {
            key: _Page(page.valid.copy(), page.dirty.copy(), page.slot, page.flash, page.ready_ns)
            for key, page in state["_pages"].items()}
        result["_buffer_pages"] = {owner: indexes.copy() for owner, indexes in state["_buffer_pages"].items()}
        result["_owners"] = {block: pages.copy() for block, pages in state["_owners"].items()}
        result["_transaction_stack"] = []
        result["_slots"] = state["_slots"].copy()
    for name in ("_resident", "_active"):
        result[name] = state[name].copy()
    result["_free_blocks"] = state["_free_blocks"].copy()
    return result


class HbfStagingController:
    """Page merging with explicit host residency and a small log-structured FTL.

    Buffers are identified by ``(buffer_id, generation)`` and offsets are
    relative to that buffer. The caller canonicalizes aliases. ``preload``
    registers immutable weights without charging initial loading. Its final
    block is padded and charged to capacity; arbitrary dynamic buffers must
    be initialized with ``write``. ``release`` discards a completed buffer,
    including its dirty tail, without inventing a final durable write.

    Times are ns. The IO callbacks must return completion times and maintain
    the actual hardware resource calendars. Snapshotting this object does
    not snapshot those calendars; callers restore both together.
    """

    def __init__(self, config: NandConfig, host_capacity_bytes: int):
        if config.protocol_mode != "ocp_hbf_0_7":
            raise ValueError("HBF staging requires strict ocp_hbf_0_7")
        if (not isinstance(host_capacity_bytes, int) or isinstance(host_capacity_bytes, bool)
                or host_capacity_bytes < 2 * PAGE or host_capacity_bytes % PAGE):
            raise ValueError("HOST DRAM staging requires at least two aligned 4 KiB pages")
        self.config = config
        self.host_capacity_bytes = host_capacity_bytes
        self.host_reserved_bytes = PAGE
        self._lanes = (config.channels * config.targets_per_channel * config.dies_per_target
                       * config.luns_per_die * config.planes_per_lun)
        # Only complete physical blocks may be reused: a truncated capacity
        # must not let page-0 erasure reach outside the declared medium.
        stripe, tail = divmod(config.effective_capacity_bytes // PAGE,
                             config.pages_per_block * self._lanes)
        self._block_count = stripe * self._lanes + max(0, tail - (config.pages_per_block - 1) * self._lanes)
        if self._block_count < 2:
            raise ValueError("HBF staging requires at least two complete physical blocks")
        self.reserved_bytes = config.block_bytes
        self.unusable_tail_bytes = config.effective_capacity_bytes - self._block_count * config.block_bytes
        self.usable_capacity_bytes = (self._block_count - 1) * config.block_bytes
        self._spare = self._block_count - 1
        self._free_blocks = _BlockPool(0, self._block_count - 1)
        self._used: dict[int, int] = _JournalDict()
        self._owners: dict[int, dict[int, Key]] = {}
        self._preloads: dict[tuple[Hashable, int], _Preload] = {}
        self._preload_occupied_bytes = 0
        self._pages: dict[Key, _Page] = {}
        self._buffer_pages: dict[tuple[Hashable, int], dict[int, None]] = {}
        self._read_states = {}
        self._resident: OrderedDict[Key, None] = OrderedDict()
        self._active: deque[int] = deque()
        self._slots = deque(range(host_capacity_bytes // PAGE - 1))
        self._slot_ready: dict[int, float] = _JournalDict()
        self._block_ready: dict[int, float] = _JournalDict()
        self._scratch_ready = 0.0
        self.stats: dict[str, int] = {}
        self._transaction_stack: list[dict] = []

    def _count(self, name: str, count: int = 1) -> None:
        self.stats[name] = self.stats.get(name, 0) + count

    @property
    def occupied_bytes(self) -> int:
        """Reserved logical media space, including preloaded block padding."""
        return len(self._pages) * PAGE + self._preload_occupied_bytes

    @property
    def host_occupied_bytes(self) -> int:
        """Dirty-page residency; the separately reported scratch page is reserved."""
        return len(self._resident) * PAGE

    @property
    def programmed_page_counts(self) -> dict[int, int]:
        """Physical block page-0 address -> sequential programming frontier.

        After preload the owner also seeds NandCore's initial programming
        frontiers, without submitting timed initial writes.
        """
        result = {self._address(block, 0): count for block, count in self._used.items() if count}
        for preload in self._preloads.values():
            result.update((self._address(block, 0), self.config.pages_per_block) for block in preload.blocks())
        return result

    def preload_page_counts(self, buffer_id: Hashable, generation: int):
        """Iterate only one preload's initial frontiers, without a large dictionary."""
        preload = self._preloads[(buffer_id, generation)]
        for block in preload.blocks():
            yield self._address(block, 0), self.config.pages_per_block

    def snapshot(self) -> dict:
        return _copy_controller_state(self.__dict__)

    def restore(self, state: dict) -> None:
        self.__dict__.clear()
        self.__dict__.update(_copy_controller_state(state))
        self._read_states = {}
        for name in _JOURNALED_MAPS:
            self.__dict__[name] = _JournalDict(self.__dict__[name])

    def begin_transaction(self):
        """Journal changed pages instead of copying every retained KV page."""
        token = {"state": _copy_controller_state(self.__dict__, include_pages=False),
                 "pages": {}, "page_ready": {}, "owners": {}, "released_order": {},
                 "read_ready": {},
                 "maps": {name: self.__dict__[name].begin() for name in _JOURNALED_MAPS}}
        self._transaction_stack.append(token)
        return token

    def _finish_transaction(self, token):
        if not self._transaction_stack or self._transaction_stack[-1] is not token:
            raise ValueError("HBF transactions must finish once in reverse begin order")
        self._transaction_stack.pop()

    def commit_transaction(self, token):
        self._finish_transaction(token)
        for name, journal in token["maps"].items():
            self.__dict__[name].commit(journal)

    def rollback_transaction(self, token):
        self._finish_transaction(token)
        for name, journal in token["maps"].items():
            self.__dict__[name].rollback(journal)
        for state, ready in token["read_ready"].values():
            state.ready[:len(ready)] = ready
        for key, page in token["pages"].items():
            owner = key[:2]
            if page is None:
                self._pages.pop(key, None)
                indexes = self._buffer_pages.get(owner)
                if indexes is not None:
                    indexes.pop(key[2], None)
                    if not indexes:
                        self._buffer_pages.pop(owner)
            else:
                self._pages[key] = page
                self._buffer_pages.setdefault(owner, {})[key[2]] = None
        for owner, indexes in token["released_order"].items():
            remaining = self._buffer_pages.get(owner)
            if remaining is not None:
                self._buffer_pages[owner] = {index: None for index in indexes if index in remaining}
        for key, ready in token["page_ready"].items():
            self._pages[key].ready_ns = ready
        for block, owners in token["owners"].items():
            if owners is None:
                self._owners.pop(block, None)
            else:
                self._owners[block] = owners
        self.__dict__.update(token["state"])
        # Rollback is exceptional. Materialize once so restored page objects
        # cannot retain a plan that described an overwritten/collected page.
        # Active outer transactions retain their original scalar values.
        for key, page in self._pages.items():
            if page._read_state is not None:
                self._touch_page_ready(key, page)
                page._ready_ns = page.ready_ns
                page._read_state = None
        self._read_states.clear()

    @staticmethod
    def _ready_before(token, page):
        state = page._read_state
        saved = None if state is None else token["read_ready"].get(id(state))
        if saved is not None and page._read_index < len(saved[1]):
            return float(saved[1][page._read_index])
        return page.ready_ns

    def _touch_page(self, key):
        for token in self._transaction_stack:
            if key not in token["pages"]:
                page = self._pages.get(key)
                token["pages"][key] = None if page is None else _Page(
                    page.valid.copy(), page.dirty.copy(), page.slot, page.flash,
                    token["page_ready"].pop(key, self._ready_before(token, page)))

    def _touch_slots(self):
        # Most physical tasks only read existing pages. The free HOST queue
        # is large, but needs an undo copy only when a slot is taken/returned.
        for token in self._transaction_stack:
            state = token["state"]
            if "_slots" not in state:
                state["_slots"] = self._slots.copy()

    def _touch_page_ready(self, key, page):
        # Reads only advance a scalar dependency. Save that scalar until a
        # later write/release in the same transaction requires the full page.
        for token in self._transaction_stack:
            if key not in token["pages"]:
                token["page_ready"].setdefault(key, self._ready_before(token, page))

    def _touch_read_ready(self, state):
        for token in self._transaction_stack:
            if id(state) not in token["read_ready"]:
                token["read_ready"][id(state)] = state, state.ready.copy()

    def _read_state(self, owner):
        state = self._read_states.get(owner)
        if state is not None:
            return state
        indexes = self._buffer_pages.get(owner, {})
        if len(indexes) < 16 or 0 not in indexes:
            return None
        size = max(indexes) + 1
        if size > 2 * len(indexes):
            return None  # Sparse offsets retain the ordinary sparse page map.
        state = _BufferReadState(size)
        for index in indexes:
            page = self._pages[(*owner, index)]
            address = -1 if page.flash is None else self._address(*page.flash)
            state.update(index, page, address)
        self._read_states[owner] = state
        return state

    def _refresh_read_state(self, key, page):
        state = self._read_states.get(key[:2])
        if state is not None:
            if key[2] >= len(state.ready) and key[2] + 1 > 2 * len(self._buffer_pages[key[:2]]):
                # A later sparse write must not turn a compact cached prefix
                # into an allocation proportional to its largest byte offset.
                for index in self._buffer_pages[key[:2]]:
                    bound_key = (*key[:2], index)
                    bound = self._pages[bound_key]
                    if bound._read_state is state:
                        self._touch_page_ready(bound_key, bound)
                        bound._ready_ns = bound.ready_ns
                        bound._read_state = None
                self._read_states.pop(key[:2])
                return
            address = -1 if page.flash is None else self._address(*page.flash)
            state.update(key[2], page, address)

    def _touch_owners(self, block):
        for token in self._transaction_stack:
            if block not in token["owners"]:
                owners = self._owners.get(block)
                token["owners"][block] = None if owners is None else owners.copy()

    def _page_for_write(self, key):
        page = self._pages.get(key)
        if page is None:
            page = self._pages[key] = _Page()
            self._buffer_pages.setdefault(key[:2], {})[key[2]] = None
        return page

    def _address(self, block: int, page: int) -> int:
        bank, lane = divmod(block, self._lanes)
        return (bank * self.config.pages_per_block * self._lanes + page * self._lanes + lane) * PAGE

    @staticmethod
    def _check_range(offset: int, byte_count: int) -> None:
        if (not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
                or not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0):
            raise ValueError("buffer offset and byte_count must be non-negative integers")

    def _host(self, io: HbfStagingIO, write: bool, address: int, count: int, time: float) -> float:
        self._count("host_write_bytes" if write else "host_read_bytes", count)
        return (io.write_host if write else io.read_host)(address, count, time)

    def _transfer(self, io: HbfStagingIO, route: str, count: int, time: float) -> float:
        self._count(route + "_bytes", count)
        return io.transfer(route, count, time)

    def _read_flash(self, io: HbfStagingIO, address: int, count: int, time: float) -> tuple[float, int]:
        start = address // 64 * 64
        length = (address + count + 63) // 64 * 64 - start
        self._count("flash_read_bytes", length)
        return io.read_flash(start, length, time), length

    def preload(self, buffer_id: Hashable, generation: int, byte_count: int) -> None:
        """Register immutable loaded data; all physical block padding consumes space."""
        self._check_range(0, byte_count)
        owner = (buffer_id, generation)
        if owner in self._preloads or owner in self._buffer_pages:
            raise ValueError("buffer generation is already initialized")
        blocks_needed = (byte_count + self.config.block_bytes - 1) // self.config.block_bytes
        if (blocks_needed > len(self._free_blocks)
                or self.occupied_bytes + blocks_needed * self.config.block_bytes > self.usable_capacity_bytes):
            raise MemoryError("insufficient HBF space for immutable preload and GC reserve")
        blocks = self._free_blocks.take(blocks_needed)
        self._preloads[owner] = _Preload(byte_count, blocks)
        self._preload_occupied_bytes += blocks_needed * self.config.block_bytes
        self._count("preload_bytes", byte_count)
        self._count("preload_padding_bytes", blocks_needed * self.config.block_bytes - byte_count)

    def _new_slot(self, time: float, io: HbfStagingIO) -> tuple[int, float]:
        if not self._slots:
            key = next(iter(self._resident))
            victim = self._pages[key]
            time = self._flush_page(key, victim, time, io)
            self._count("pressure_flushes")
        self._touch_slots()
        slot = self._slots.popleft()
        return slot, max(time, self._slot_ready.get(slot, 0.0))

    def _allocate(self, time: float, io: HbfStagingIO, replacing: Key) -> tuple[tuple[int, int], float]:
        pages_per_block = self.config.pages_per_block
        while len(self._active) < self._lanes and self._free_blocks:
            block = self._free_blocks.popleft()
            self._used[block] = 0
            self._active.append(block)
        if not self._active:
            candidates = []
            for block in self._used:
                if block == self._spare:
                    continue
                owners = self._owners.get(block, {})
                live = len(owners) - int(replacing in owners.values())
                if live < pages_per_block:
                    candidates.append((live, block))
            if not candidates:
                raise MemoryError("HBF has no reclaimable block after reserving GC space")
            _, victim = min(candidates)
            time = self._collect(victim, replacing, time, io)
        block = self._active.popleft()
        page = self._used[block]
        self._used[block] += 1
        if self._used[block] < pages_per_block:
            self._active.append(block)
        # Page-0 may erase an old incarnation of a block. Other pages are
        # independently admitted; the NAND core schedules same-bank hazards.
        if page == 0:
            time = max(time, self._block_ready.get(block, 0.0))
        return (block, page), time

    def _collect(self, victim: int, replacing: Key, time: float, io: HbfStagingIO) -> float:
        target = self._spare
        time = max(time, self._scratch_ready, self._block_ready.get(victim, 0.0),
                   self._block_ready.get(target, 0.0))
        self._used[target] = 0
        target_owners: dict[int, Key] = {}
        scratch = self.host_capacity_bytes - PAGE
        for old_index, key in sorted(self._owners.get(victim, {}).items()):
            if key == replacing:
                continue
            self._touch_page(key)
            page = self._pages[key]
            time, _ = self._read_flash(io, self._address(victim, old_index), PAGE,
                                       max(time, page.ready_ns))
            time = self._transfer(io, "from_hbf", PAGE, time)
            time = self._host(io, True, scratch, PAGE, time)
            time = self._host(io, False, scratch, PAGE, time)
            time = self._transfer(io, "to_hbf", PAGE, time)
            new_index = self._used[target]
            time = io.write_flash(self._address(target, new_index), PAGE, time)
            self._count("flash_write_bytes", PAGE)
            self._count("gc_copied_pages")
            self._used[target] += 1
            target_owners[new_index] = key
            page.flash = (target, new_index)
            page.ready_ns = time
            self._refresh_read_state(key, page)
        self._touch_owners(victim)
        self._touch_owners(target)
        self._owners.pop(victim, None)
        self._owners[target] = target_owners
        self._used.pop(victim, None)
        self._block_ready[victim] = time
        self._block_ready[target] = time
        self._scratch_ready = time
        self._spare = victim
        self._active.append(target)
        self._count("gc_cycles")
        return time

    def _flush_page(self, key: Key, page: _Page, time: float, io: HbfStagingIO) -> float:
        if page.slot is None:
            return max(time, page.ready_ns)
        self._touch_page(key)
        time = max(time, page.ready_ns)
        slot = page.slot
        address = slot * PAGE
        # Preserve only initialized old bytes. NAND padding is never promoted
        # into valid KV merely because a complete physical page was written.
        old_ranges = _subtract(page.valid, page.dirty)
        if old_ranges:
            assert page.flash is not None
            self._count("rmw_pages")
            old_address = self._address(*page.flash)
            for lo, hi in old_ranges:
                time, fetched = self._read_flash(io, old_address + lo, hi - lo, time)
                time = self._transfer(io, "from_hbf", fetched, time)
                time = self._host(io, True, address + lo, hi - lo, time)
                self._count("rmw_read_bytes", fetched)
        for lo, hi in _subtract([(0, PAGE)], page.valid):
            time = self._host(io, True, address + lo, hi - lo, time)
            self._count("padding_bytes", hi - lo)
        physical, time = self._allocate(time, io, key)
        time = self._host(io, False, address, PAGE, time)
        time = self._transfer(io, "to_hbf", PAGE, time)
        time = io.write_flash(self._address(*physical), PAGE, time)
        self._count("flash_write_bytes", PAGE)
        self._count("flushed_pages")
        if page.flash is not None:
            self._touch_owners(page.flash[0])
            self._owners.get(page.flash[0], {}).pop(page.flash[1], None)
        page.flash = physical
        self._touch_owners(physical[0])
        self._owners.setdefault(physical[0], {})[physical[1]] = key
        self._block_ready[physical[0]] = max(time, self._block_ready.get(physical[0], 0.0))
        page.slot = None
        self._resident.pop(key)
        page.dirty = []
        page.ready_ns = time
        self._refresh_read_state(key, page)
        self._slot_ready[slot] = time
        self._touch_slots()
        self._slots.append(slot)
        return time

    def write(self, buffer_id: Hashable, generation: int, offset: int, byte_count: int,
              time_ns: float, io: HbfStagingIO) -> float:
        self._check_range(offset, byte_count)
        if not byte_count:
            return float(time_ns)
        if (buffer_id, generation) in self._preloads:
            raise ValueError("preloaded weights are immutable; release before reusing their generation")
        first, stop = offset // PAGE, (offset + byte_count + PAGE - 1) // PAGE
        new_pages = sum((buffer_id, generation, index) not in self._pages for index in range(first, stop))
        if self.occupied_bytes + new_pages * PAGE > self.usable_capacity_bytes:
            raise MemoryError("HBF logical capacity exhausted, including GC reserve")
        completion = float(time_ns)
        self._count("logical_write_bytes", byte_count)
        bulk_write = getattr(io, "write_flash_pages", None)
        if (offset % PAGE == 0 and byte_count % PAGE == 0 and stop - first >= 16
                and bulk_write is not None):
            completion = self._write_full_pages(buffer_id, generation, first, stop,
                                                 float(time_ns), io, bulk_write)
            if completion is not None:
                return completion
            completion = float(time_ns)
        for index in range(first, stop):
            lo, hi = max(0, offset - index * PAGE), min(PAGE, offset + byte_count - index * PAGE)
            key = (buffer_id, generation, index)
            self._touch_page(key)
            page = self._page_for_write(key)
            time = max(float(time_ns), page.ready_ns)
            if lo == 0 and hi == PAGE:
                # A full overwrite needs neither old data nor host staging.
                # Keep it on the initiator -> HBF path, with the same legal
                # out-of-place allocator and GC as fragment flushes.
                old_ready = time
                physical, time = self._allocate(time, io, key)
                time = self._transfer(io, "write_hbf", PAGE, time)
                time = io.write_flash(self._address(*physical), PAGE, time)
                self._count("flash_write_bytes", PAGE)
                self._count("direct_write_pages")
                if page.flash is not None:
                    self._touch_owners(page.flash[0])
                    self._owners.get(page.flash[0], {}).pop(page.flash[1], None)
                if page.slot is not None:
                    self._slot_ready[page.slot] = old_ready
                    self._touch_slots()
                    self._slots.append(page.slot)
                    self._resident.pop(key)
                page.flash = physical
                self._touch_owners(physical[0])
                self._owners.setdefault(physical[0], {})[physical[1]] = key
                self._block_ready[physical[0]] = max(time, self._block_ready.get(physical[0], 0.0))
                page.slot = None
                page.valid = [(0, PAGE)]
                page.dirty = []
                page.ready_ns = time
                self._refresh_read_state(key, page)
                completion = max(completion, time)
                continue
            if page.slot is None:
                page.slot, time = self._new_slot(time, io)
                self._resident[key] = None
            self._resident.move_to_end(key)
            was_full = page.valid == [(0, PAGE)]
            time = self._transfer(io, "to_host", hi - lo, time)
            time = self._host(io, True, page.slot * PAGE + lo, hi - lo, time)
            page.valid = _merge(page.valid + [(lo, hi)])
            page.dirty = _merge(page.dirty + [(lo, hi)])
            page.ready_ns = time
            if page.dirty == [(0, PAGE)] or (not was_full and page.valid == [(0, PAGE)]):
                time = self._flush_page(key, page, time, io)
            self._refresh_read_state(key, page)
            completion = max(completion, time)
        return completion

    def _write_full_pages(self, buffer_id, generation, first, stop, time_ns, io, bulk_write):
        # Allocation is independent of page completions while no collection
        # is needed: each block reaches page zero only once in this batch.
        # Collection retains the ordinary read/host-copy/program sequence.
        available = (len(self._free_blocks) * self.config.pages_per_block
                     + sum(self.config.pages_per_block - self._used[block] for block in self._active))
        if stop - first > available:
            return None
        token = self.begin_transaction()
        locations, arrivals, old_ready = [], [], []
        try:
            for index in range(first, stop):
                key = (buffer_id, generation, index)
                page = self._pages.get(key)
                ready = max(time_ns, page.ready_ns if page is not None else 0.0)
                physical, arrival = self._allocate(ready, io, key)
                locations.append(physical)
                arrivals.append(arrival)
                old_ready.append(ready)
            completions = bulk_write([self._address(*location) for location in locations], arrivals)
        except BaseException:
            self.rollback_transaction(token)
            raise
        if completions is None:
            self.rollback_transaction(token)
            return None
        self.commit_transaction(token)
        completion = time_ns
        indexes = self._buffer_pages.setdefault((buffer_id, generation), {})
        if isinstance(completions, np.ndarray):
            completions = completions.tolist()
        for index, physical, ready, time in zip(range(first, stop), locations, old_ready, completions):
            key = (buffer_id, generation, index)
            self._touch_page(key)
            page = self._pages.get(key)
            if page is None:
                self._pages[key] = _Page([(0, PAGE)], [], None, physical, time)
                indexes[index] = None
            else:
                if page.flash is not None:
                    self._touch_owners(page.flash[0])
                    self._owners.get(page.flash[0], {}).pop(page.flash[1], None)
                if page.slot is not None:
                    self._slot_ready[page.slot] = ready
                    self._touch_slots()
                    self._slots.append(page.slot)
                    self._resident.pop(key)
                page.flash = physical
                page.slot = None
                page.valid = [(0, PAGE)]
                page.dirty = []
                page.ready_ns = time
            self._touch_owners(physical[0])
            self._owners.setdefault(physical[0], {})[physical[1]] = key
            self._refresh_read_state(key, self._pages[key])
            self._block_ready[physical[0]] = max(time, self._block_ready.get(physical[0], 0.0))
            completion = max(completion, time)
        self._count("flash_write_bytes", (stop - first) * PAGE)
        self._count("write_hbf_bytes", (stop - first) * PAGE)
        self._count("direct_write_pages", stop - first)
        return completion

    def read(self, buffer_id: Hashable, generation: int, offset: int, byte_count: int,
             time_ns: float, io: HbfStagingIO) -> float:
        self._check_range(offset, byte_count)
        if not byte_count:
            return float(time_ns)
        preload = self._preloads.get((buffer_id, generation))
        first, stop = offset // PAGE, (offset + byte_count + PAGE - 1) // PAGE
        bulk_read = getattr(io, "read_flash_pages", None)
        read_state = (self._read_state((buffer_id, generation))
                      if preload is None and bulk_read is not None else None)
        # Validate the whole read before submitting any hardware work.
        if preload is not None:
            if offset + byte_count > preload.byte_count:
                raise ValueError("read touches uninitialized HBF buffer bytes: "
                    f"{buffer_id!r}, generation={generation}, offset={offset}, "
                    f"size={byte_count}, preloaded_size={preload.byte_count}")
        else:
            unchecked = range(first, stop)
            if read_state is not None:
                checked_stop = min(stop, len(read_state.valid))
                unchecked = (np.flatnonzero(~read_state.valid[first:checked_stop]) + first).tolist()
                unchecked.extend(range(max(first, checked_stop), stop))
            for index in unchecked:
                lo, hi = max(0, offset - index * PAGE), min(PAGE, offset + byte_count - index * PAGE)
                page = self._pages.get((buffer_id, generation, index))
                if page is None or _subtract([(lo, hi)], page.valid):
                    raise ValueError("read touches uninitialized HBF buffer bytes: "
                        f"{buffer_id!r}, generation={generation}, offset={offset}, "
                        f"size={byte_count}, page={index}, "
                        f"valid={None if page is None else page.valid}")
        completion = float(time_ns)
        self._count("logical_read_bytes", byte_count)
        index = first
        while index < stop:
            time = float(time_ns)
            lo, hi = max(0, offset - index * PAGE), min(PAGE, offset + byte_count - index * PAGE)
            if bulk_read is not None and lo == 0 and hi == PAGE:
                end = (offset + byte_count) // PAGE
                if preload is None:
                    if read_state is not None:
                        dirty = np.flatnonzero(read_state.addresses[index:end] < 0)
                        if len(dirty):
                            end = index + int(dirty[0])
                    else:
                        end = index
                        while end < (offset + byte_count) // PAGE:
                            page = self._pages[(buffer_id, generation, end)]
                            if page.dirty or page.flash is None:
                                break
                            end += 1
                if end - index >= 16:
                    bulk_end = self._read_full_pages(buffer_id, generation, index, end,
                                                     preload, time, bulk_read)
                    if bulk_end is not None:
                        completion = max(completion, bulk_end)
                        index = end
                        continue
                    # An unsupported core/calendar cannot become supported
                    # halfway through this logical access.
                    bulk_read = None
            if preload is not None:
                block, page_index = preload.page_location(index, self._lanes, self.config.pages_per_block)
                time, fetched = self._read_flash(io, self._address(block, page_index) + lo, hi - lo, time)
                time = self._transfer(io, "read_hbf", fetched, time)
                self._block_ready[block] = max(time, self._block_ready.get(block, 0.0))
                completion = max(completion, time)
                index += 1
                continue
            page = self._pages[(buffer_id, generation, index)]
            self._touch_page_ready((buffer_id, generation, index), page)
            time = max(time, page.ready_ns)
            dirty = _intersection(page.dirty, lo, hi)
            for a, b in dirty:
                assert page.slot is not None
                time = self._host(io, False, page.slot * PAGE + a, b - a, time)
                time = self._transfer(io, "from_host", b - a, time)
            for a, b in _subtract([(lo, hi)], dirty):
                assert page.flash is not None
                time, fetched = self._read_flash(io, self._address(*page.flash) + a, b - a, time)
                time = self._transfer(io, "read_hbf", fetched, time)
                block = page.flash[0]
                self._block_ready[block] = max(time, self._block_ready.get(block, 0.0))
            page.ready_ns = time
            if page.slot is not None:
                self._resident.move_to_end((buffer_id, generation, index))
            completion = max(completion, time)
            index += 1
        return completion

    def _read_full_pages(self, buffer_id, generation, first, stop, preload, time_ns, bulk_read):
        # Preserve the complete-page prefix of KV reads even when their last
        # bytes still reside in HOST DRAM. Partial/dirty pages execute in the
        # original order around this clean run.
        read_state = self._read_states.get((buffer_id, generation)) if preload is None else None
        if preload is not None or read_state is not None:
            if preload is not None:
                addresses, blocks, inverse = _preload_read_plan(
                    preload, self._lanes, self.config.pages_per_block, first, stop)
                arrivals = time_ns
            else:
                addresses, blocks, inverse = read_state.plan(
                    first, stop, self._lanes, self.config.pages_per_block)
                arrivals = np.maximum(time_ns, read_state.ready[first:stop])
            completions = bulk_read(addresses, arrivals)
            if completions is None:
                return None
            if read_state is not None:
                self._touch_read_ready(read_state)
                read_state.ready[first:stop] = completions
            self._count("flash_read_bytes", (stop - first) * PAGE)
            self._count("read_hbf_bytes", (stop - first) * PAGE)
            ready = np.full(len(blocks), -np.inf)
            np.maximum.at(ready, inverse, completions)
            self._block_ready.update_max(blocks, ready)
            return max(time_ns, float(ready.max()))
        pages = [
            self._pages[(buffer_id, generation, index)] for index in range(first, stop)]
        locations = [page.flash for page in pages]
        arrivals = [max(time_ns, page.ready_ns) for page in pages]
        completions = bulk_read([self._address(*location) for location in locations], arrivals)
        if completions is None:
            return None
        self._count("flash_read_bytes", (stop - first) * PAGE)
        self._count("read_hbf_bytes", (stop - first) * PAGE)
        completion = time_ns
        if isinstance(completions, np.ndarray):
            completions = completions.tolist()
        for index, (block, _page), time in zip(range(first, stop), locations, completions):
            page = pages[index - first]
            self._touch_page_ready((buffer_id, generation, index), page)
            page.ready_ns = time
            if page.slot is not None:
                self._resident.move_to_end((buffer_id, generation, index))
            self._block_ready[block] = max(time, self._block_ready.get(block, 0.0))
            completion = max(completion, time)
        return completion

    def flush(self, buffer_id: Hashable, generation: int, time_ns: float, io: HbfStagingIO) -> float:
        time = float(time_ns)
        for index in self._buffer_pages.get((buffer_id, generation), ()):
            key = (buffer_id, generation, index)
            time = max(time, self._flush_page(key, self._pages[key], float(time_ns), io))
        return time

    def _release_clean_pages(self, owner, indexes, time):
        """Discard a clean numeric buffer without reconstructing each page.

        Only a committed release can discard the objects directly. A live
        transaction still needs its per-page undo entries, and HOST-resident
        tails still release their slots in the original page order.
        """
        state = self._read_states.get(owner)
        if state is None or self._transaction_stack or not indexes:
            return None
        selected = np.fromiter(indexes, dtype=np.int64, count=len(indexes))
        addresses, ready = state.addresses[selected], state.ready[selected]
        if (np.any(addresses < 0) or not np.isfinite(ready).all()
                or not np.isfinite(time)):
            return None
        # argmax retains the first equal maximum (including signed zero),
        # matching max(time, page.ready_ns) in the buffer's insertion order.
        time = max(time, float(ready[int(np.argmax(ready))]))
        physical = addresses // PAGE
        physical_blocks = (physical // (self.config.pages_per_block * self._lanes)
                           * self._lanes + physical % self._lanes)
        blocks, inverse, counts = np.unique(physical_blocks,
                                            return_inverse=True, return_counts=True)
        partial = []
        for position, (block, count) in enumerate(zip(blocks.tolist(), counts.tolist())):
            owners = self._owners.get(block)
            if owners is None:
                continue
            if len(owners) == count:
                owners.clear()
            else:
                partial.append((position, owners))
        if partial:
            order = np.argsort(inverse, kind="stable")
            ends = np.cumsum(counts)
            pages = physical // self._lanes % self.config.pages_per_block
            for position, owners in partial:
                start = 0 if position == 0 else int(ends[position - 1])
                numbers = pages[order[start:int(ends[position])]].tolist()
                deque(map(owners.pop, numbers, repeat(None)), maxlen=0)
        # Both map and zip iterate in C; each physical mapping is still
        # removed, and existing dictionaries keep their identity and order.
        deque(map(self._pages.pop, zip(repeat(owner[0]), repeat(owner[1]), indexes)), maxlen=0)
        return time

    def release(self, buffer_id: Hashable, generation: int, time_ns: float = 0.0) -> float:
        owner = (buffer_id, generation)
        time = float(time_ns)
        preload = self._preloads.pop(owner, None)
        if preload is not None:
            self._preload_occupied_bytes -= preload.block_count * self.config.block_bytes
            for block in self._block_ready:
                if not any(block in span for span in preload.block_ranges):
                    continue
                # Mark as reclaimable. Its next reuse still performs page-0
                # erase through the ordinary strict NAND write callback.
                time = max(time, self._block_ready.get(block, 0.0))
            self._free_blocks.extend_ranges(preload.block_ranges)
        indexes = self._buffer_pages.pop(owner, ())
        journaled = bool(self._transaction_stack)
        clean_time = self._release_clean_pages(owner, indexes, time)
        if clean_time is not None:
            self._read_states.pop(owner)
            self._count("released_buffers")
            return clean_time
        for token in self._transaction_stack:
            if owner not in token["released_order"]:
                token["released_order"][owner] = tuple(indexes)
        for index in indexes:
            key = (*owner, index)
            if journaled:
                self._touch_page(key)
            page = self._pages.pop(key)
            time = max(time, page.ready_ns)
            if page.slot is not None:
                self._slot_ready[page.slot] = time
                self._touch_slots()
                self._slots.append(page.slot)
                self._resident.pop(key)
            if page.flash is not None:
                if journaled:
                    self._touch_owners(page.flash[0])
                self._owners.get(page.flash[0], {}).pop(page.flash[1], None)
        self._read_states.pop(owner, None)
        self._count("released_buffers")
        return time


__all__ = ["HbfStagingController", "HbfStagingIO"]
