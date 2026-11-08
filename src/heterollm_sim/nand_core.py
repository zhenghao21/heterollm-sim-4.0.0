"""Shared page-level NAND core for SSD and HBF configurations."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import deepcopy
from dataclasses import dataclass, replace
from heapq import heapify, heappop, heappush
from math import isfinite, sqrt
from typing import Dict, Iterable, Optional

from .memory_mapping import map_nand_address, split_nand_request, validate_nand_request
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, AddressMapping, NandConfig, Operation, StageTiming, TransactionResult


class _HostList(list):
    """Logical list operations for state inspection without publishing Core."""

    def __eq__(self, other):
        if isinstance(other, list):
            return list(self) == list(other)
        return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __repr__(self):
        return repr(list(self))

    def __getitem__(self, index):
        return list(self)[index]


class _HostPopulation(_HostList):
    """Ordered live commands with completion heaps and incremental counts.

    The list protocol is used only by the runtime's existing snapshot copier.
    Public Core properties publish ordinary, detached lists; controller code
    never reads those properties. No mutable index is shared with a snapshot.
    """

    def __init__(self, commands=()):
        self.commands = dict(enumerate(commands))
        self.next_id = len(self.commands)
        self.ends = [(end, index) for index, (end, _) in self.commands.items()]
        self.write_ends = [(end, index) for index, (end, op) in self.commands.items()
                           if op is Operation.WRITE]
        heapify(self.ends)
        heapify(self.write_ends)
        self.reads = sum(op is Operation.READ for _, op in self.commands.values())
        self.writes = len(self.write_ends)

    def __len__(self):
        return len(self.commands)

    def __iter__(self):
        return iter(self.commands.values())

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(command, memo) for command in self)
        return result

    def preview(self, effective, limit=None, write_limit=None):
        # Admission must not retire live commands before execution succeeds.
        # Temporarily pop expired heap entries, then restore them. Each entry
        # is inspected only when it expires, never on every new submission.
        expired, expired_writes = [], []
        reads, writes, count = self.reads, self.writes, len(self.commands)
        try:
            while True:
                while self.ends and self.ends[0][0] <= effective:
                    item = heappop(self.ends)
                    expired.append(item)
                    op = self.commands[item[1]][1]
                    reads -= op is Operation.READ
                    writes -= op is Operation.WRITE
                    count -= 1
                while self.write_ends and self.write_ends[0][0] <= effective:
                    expired_writes.append(heappop(self.write_ends))
                blocked = []
                if limit is not None and count >= limit:
                    blocked.append(self.ends[0][0])
                if write_limit is not None and writes >= write_limit:
                    blocked.append(self.write_ends[0][0])
                if not blocked:
                    return _HostAdmission(self, effective, count, reads, writes)
                effective = max(effective, max(blocked))
        finally:
            for item in expired:
                heappush(self.ends, item)
            for item in expired_writes:
                heappush(self.write_ends, item)

    def record(self, effective, completion, operation):
        while self.ends and self.ends[0][0] <= effective:
            _, index = heappop(self.ends)
            _, op = self.commands.pop(index)
            self.reads -= op is Operation.READ
            self.writes -= op is Operation.WRITE
        while self.write_ends and self.write_ends[0][0] <= effective:
            heappop(self.write_ends)
        index = self.next_id
        self.next_id += 1
        self.commands[index] = (completion, operation)
        heappush(self.ends, (completion, index))
        if operation is Operation.WRITE:
            heappush(self.write_ends, (completion, index))
            self.writes += 1
        elif operation is Operation.READ:
            self.reads += 1


class _HostAdmission:
    """Immutable counts for one admission, including after queue publication."""

    __slots__ = ("population", "effective", "count", "reads", "writes")

    def __init__(self, population, effective, count, reads, writes):
        self.population, self.effective = population, effective
        self.count, self.reads, self.writes = count, reads, writes

    def __len__(self):
        return self.count

    def __iter__(self):
        return (command for command in self.population if command[0] > self.effective)


class _HostCompletions(_HostList):
    """Lazy projection for the independently observable completion list."""

    def __init__(self, population):
        self.population = population

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return (end for end, _ in self.population)

    def __deepcopy__(self, memo):
        result = list(self)
        memo[id(self)] = result
        return result


@dataclass
class _StagedPage:
    """Host-side state for one dirty NAND page.

    The simulator intentionally does not model cell contents.  Keeping the
    physical mapping and byte counters is enough to preserve the important
    contract: repeated small writes merge into one eventual page program and
    reads observe the staged copy until the program drains to media.
    """

    mapping: object
    host_bytes: int = 0
    logical_bytes: int = 0
    merged_host_bytes: int = 0
    request_count: int = 0
    ready_ns: float = 0.0
    reader_completion_ns: float = 0.0
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass
class _ReadCacheSlot:
    """A physical page buffer, including scheduled fills and live readers."""

    page_key: Optional[tuple[int, ...]] = None
    ready_ns: float = 0.0
    reader_completion_ns: float = 0.0
    last_use: int = 0


def _merge_ranges(ranges):
    """Union of half-open byte ranges, ascending and coalesced."""
    merged = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            if right > merged[-1][1]:
                merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    return tuple(merged)


def _page_share(page):
    """A fixed, evenly spread position of a page index in [0, 1) (Fibonacci hashing)."""
    return (page * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF) / 18446744073709551616.


_RESOURCE_MAP_FIELDS = ("resource_busy_ns", "resource_bytes", "resource_last_intervals")


def _without_resource_maps(result):
    """The result whose resource maps a private task sink has consumed."""
    counters = dict(result.counters)
    for name in _RESOURCE_MAP_FIELDS:
        counters.pop(name, None)
    return replace(result, counters=counters)


def _copy_nand_read_cache(cache, memo=None):
    """Copy page buffers directly while retaining extension aliases/cycles."""
    if memo is None:
        memo = {}
    if type(cache) is not dict:
        return deepcopy(cache, memo)
    result, rows, slots = {}, [], []
    memo[id(cache)] = result
    # Register every ordinary list/slot before copying extension attributes:
    # extensions may refer to a later bank, another slot or the cache itself.
    for row in cache.values():
        if type(row) is not list or id(row) in memo:
            continue
        cloned = []
        memo[id(row)] = cloned
        rows.append((row, cloned))
        for slot in row:
            if type(slot) is _ReadCacheSlot and id(slot) not in memo:
                cloned_slot = object.__new__(_ReadCacheSlot)
                memo[id(slot)] = cloned_slot
                memo[id(vars(slot))] = vars(cloned_slot)
                slots.append((slot, cloned_slot))

    def clone(value):
        if type(value) in (int, float, str, bool, type(None)):
            return value
        if id(value) in memo:
            return memo[id(value)]
        if type(value) is tuple and all(type(item) is int for item in value):
            return value
        return deepcopy(value, memo)

    for slot, cloned in slots:
        values = vars(slot)
        page_key = slot.page_key
        if (values.keys() == {"page_key", "ready_ns", "reader_completion_ns", "last_use"}
                and (page_key is None or type(page_key) is tuple
                     and all(type(item) is int for item in page_key))
                and type(slot.ready_ns) in (int, float)
                and type(slot.reader_completion_ns) in (int, float)
                and type(slot.last_use) is int):
            # Ordinary controller slots contain only immutable scalar values.
            # Copy their fields in C; extension objects still use the shared
            # deepcopy memo registered above, retaining aliases and cycles.
            cloned.__dict__.update(values)
        else:
            cloned.__dict__.update((name, clone(value)) for name, value in values.items())
    for row, cloned in rows:
        cloned.extend(clone(slot) for slot in row)
    result.update((clone(key), clone(row)) for key, row in cache.items())
    return result


class _PageAbsent:
    """A journaled page that had no table entry; one instance across copies."""

    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return "_PAGE_ABSENT"


_PAGE_ABSENT = _PageAbsent()


class _NandTransaction:
    """Undo only media indexes changed while this transaction is open."""

    def __init__(self, core):
        # Granted only by PhysicalRuntimeContext after this phase's complete
        # dynamic native-owner/timeline qualification. It dies with the undo
        # token; direct/public Core submissions never infer this permission.
        self.native_timeline_qualified = False
        self.bank_log = None
        self.write_pointer = tuple(core.__dict__.get("_write_pointer", ()))
        self.seed_history_pages = int(core.__dict__.get("_seed_history_pages", 0))
        self.write_order_errors = int(core.__dict__.get("_write_order_errors", 0))
        # The page table is journaled, not copied: its writers record each
        # page's entry before this transaction's first write of that page.
        # A public snapshot materializes the begin-time table instead.
        self.page_table_root = core.__dict__.get("_logical_page_table", {})
        self.page_table_undo = {}
        self.page_table = None
        self.page_table_identity = bool(core.__dict__.get("_logical_page_table_identity", True))
        self.block_wear = {}
        self.cache_banks = {}
        self.cache_slots = {}
        self.cache_snapshot = None
        lazy_frontiers = (not core._nand_state_exposed
                          and type(core.__dict__["_array_ready"]) is dict
                          and type(core.__dict__["_buffer_ready"]) is dict
                          and _native_frontier_writers(core))
        self.frontier_roots = (core.__dict__["_array_ready"], core.__dict__["_buffer_ready"])
        self.frontier_undo = ({}, {})
        self.array_frontiers = None if lazy_frontiers else dict(self.frontier_roots[0])
        self.buffer_frontiers = None if lazy_frontiers else dict(self.frontier_roots[1])
        self.numeric_at_begin = core._nand_numeric_state
        self.numeric = None
        # Staged tail pages are journaled per page; the runtime snapshot
        # copies their small per-channel counters and in-flight programs.
        self.tail_pages = {}
        self.tail_scopes = None
        self.tail_spilled = None
        self.tail_counters = (core._tail_valid, core._tail_peak, list(core._tail_flushing),
                              core._tail_full_page_fills, core._tail_flush_started_pages,
                              core._tail_flush_discarded_pages, core._tail_spilled_pages,
                              core._tail_evicted_pages)

    def remember_tail(self, core, page):
        if page not in self.tail_pages:
            self.tail_pages[page] = core._tail_pages.get(page)

    def remember_pages(self, table, pages):
        if self.page_table is not None or table is not self.page_table_root:
            return
        undo = self.page_table_undo
        for page in pages:
            if page not in undo:
                undo[page] = table.get(page, _PAGE_ABSENT)

    def _undo_pages(self, table):
        for page, previous in self.page_table_undo.items():
            if previous is _PAGE_ABSENT:
                table.pop(page, None)
            else:
                table[page] = previous
        self.page_table_undo = {}
        return table

    def materialize_page_table(self):
        if self.page_table is None:
            self.page_table = self._undo_pages(dict(self.page_table_root))

    def remember_tail_scopes(self, core):
        if self.tail_scopes is None:
            self.tail_scopes = {scope: set(pages)
                                for scope, pages in core._tail_full_scopes.items()}

    def remember_tail_spilled(self, core):
        if self.tail_spilled is None:
            self.tail_spilled = set(core._tail_spilled)

    def capture_numeric(self, packed):
        if packed is self.numeric_at_begin and self.numeric is None:
            self.numeric = packed.checkpoint()

    def remember_frontiers(self, key):
        if self.array_frontiers is not None:
            return
        for root, undo in zip(self.frontier_roots, self.frontier_undo):
            if key not in undo:
                undo[key] = root.get(key)

    def materialize_frontiers(self):
        if self.array_frontiers is not None:
            return
        saved = tuple(dict(root) for root in self.frontier_roots)
        for target, undo in zip(saved, self.frontier_undo):
            for key, previous in undo.items():
                if previous is None:
                    target.pop(key, None)
                else:
                    target[key] = previous
        self.array_frontiers, self.buffer_frontiers = saved
        self.frontier_undo = ({}, {})

    def remember_cache(self, core, bank):
        if self.cache_snapshot is not None or bank in self.cache_banks:
            return
        slots = core.__dict__["_read_cache"].get(bank)
        if slots is not None and (type(slots) is not list
                                  or any(type(slot) is not _ReadCacheSlot for slot in slots)):
            self.materialize_cache(core)
            return
        self.cache_banks[bank] = slots
        for slot in slots or ():
            self.cache_slots.setdefault(id(slot), (
                slot, slot.page_key, slot.ready_ns, slot.reader_completion_ns, slot.last_use))

    def materialize_cache(self, core):
        """Detach undo state before an external snapshot replaces live slots."""
        self.materialize_frontiers()
        if self.cache_snapshot is not None:
            return
        memo = {}
        saved = _copy_nand_read_cache(core.__dict__["_read_cache"], memo)
        for bank, previous in self.cache_banks.items():
            if previous is None:
                saved.pop(bank, None)
            else:
                saved[bank] = deepcopy(previous, memo)
        for slot, page_key, ready, readers, use in self.cache_slots.values():
            cloned = deepcopy(slot, memo)
            cloned.page_key, cloned.ready_ns = page_key, ready
            cloned.reader_completion_ns, cloned.last_use = readers, use
        if self.numeric_at_begin is not None:
            self.numeric_at_begin.publish(saved, self.array_frontiers, self.buffer_frontiers,
                                          checkpoint=self.numeric)
            self.numeric_at_begin = self.numeric = None
        self.cache_snapshot = saved

    def rollback(self, core):
        if self.bank_log is not None:
            # Tokens can be restored by a public snapshot and used again.
            core._bank_log = list(self.bank_log)
        if self.write_pointer:
            core._write_pointer = list(self.write_pointer)
            core._seed_history_pages = self.seed_history_pages
            core._write_order_errors = self.write_order_errors
            core._logical_page_table = (dict(self.page_table) if self.page_table is not None
                                        else self._undo_pages(self.page_table_root))
            core._logical_page_table_identity = self.page_table_identity
        for key, previous in self.block_wear.items():
            if previous is None:
                core._block_wear_state.pop(key, None)
            else:
                core._block_wear_state[key] = previous
        if self.cache_snapshot is not None:
            # Tokens can be restored by a public snapshot and used again.
            core.__dict__["_read_cache"] = _copy_nand_read_cache(self.cache_snapshot)
        else:
            for bank, previous in self.cache_banks.items():
                if previous is None:
                    core.__dict__["_read_cache"].pop(bank, None)
                else:
                    core.__dict__["_read_cache"][bank] = previous
            for slot, page_key, ready, readers, use in self.cache_slots.values():
                slot.page_key, slot.ready_ns = page_key, ready
                slot.reader_completion_ns, slot.last_use = readers, use
        if self.array_frontiers is None:
            for name, root, undo in zip(("_array_ready", "_buffer_ready"),
                                        self.frontier_roots, self.frontier_undo):
                for key, previous in undo.items():
                    if previous is None:
                        root.pop(key, None)
                    else:
                        root[key] = previous
                core.__dict__[name] = root
        else:
            core.__dict__["_array_ready"] = dict(self.array_frontiers)
            core.__dict__["_buffer_ready"] = dict(self.buffer_frontiers)
        if self.numeric is not None:
            self.numeric_at_begin.restore(self.numeric)
        core._nand_numeric_state = self.numeric_at_begin
        core._tail_valid, core._tail_peak = self.tail_counters[:2]
        core._tail_flushing = list(self.tail_counters[2])
        (core._tail_full_page_fills, core._tail_flush_started_pages,
         core._tail_flush_discarded_pages, core._tail_spilled_pages,
         core._tail_evicted_pages) = self.tail_counters[3:]
        if self.tail_spilled is not None:
            core._tail_spilled = set(self.tail_spilled)
        if self.tail_pages:
            pages = core._tail_pages
            for page, previous in self.tail_pages.items():
                if previous is None:
                    pages.pop(page, None)
                else:
                    pages[page] = previous
            core._tail_index = None
        if self.tail_scopes is not None:
            core._tail_full_scopes = {scope: set(pages)
                                      for scope, pages in self.tail_scopes.items()}


class NandCore:
    """Page read/program/erase with optional explicit erase wear.

    There is no FTL, garbage collector or cell-content validity model.
    ``ocp_hbf_0_7`` models complete 4 KiB write units and the host's per-bank
    log of page numbers; opening a block is the host's rate term in
    native_stationary. It does not expose AXI fragments/pending responses,
    IDs/error packets, ECC, retention or wear-level remapping.
    Read-cache banks are the mapped channel/target/die/LUN/plane coordinates;
    replacement is deterministic LRU, an explicit controller policy choice.
    """

    def __init__(
        self,
        config: NandConfig,
        timeline: Optional[ResourceTimeline] = None,
        *,
        capture_details: bool = True,
    ) -> None:
        if not isinstance(config, NandConfig):
            raise TypeError("config must be NandConfig")
        if not isinstance(capture_details, bool):
            raise TypeError("capture_details must be a boolean")
        self.config = config
        self.timeline = timeline or ResourceTimeline()
        # Keep exact stage/mapping traces for normal runs; aggregate serving
        # estimates can disable them while preserving scalar timing/counters.
        self.capture_details = capture_details
        self.__dict__["_array_ready"]: Dict[str, float] = {}
        self.__dict__["_buffer_ready"]: Dict[str, float] = {}
        self._inflight: list[float] = []
        # Host request admission occupancy, bounded by max_outstanding_requests.
        # These are not per-plane page/read/program queues. Buffered writes
        # leave this host-level population at acknowledgement, before drain.
        self._inflight_operations: list[tuple[float, Operation]] = []
        self._acceptance_ns = 0.0
        self._host_inflight: dict[int, list[tuple[float, Operation]]] = {}
        self._host_acceptance_ns: dict[int, float] = {}
        # A positive write_buffer_bytes models a shared host-side staging
        # budget.  Entries are released when their media program completes;
        # zero preserves the existing write-through path.
        self._write_buffer_pending: list[tuple[float, int]] = []
        self._write_buffer_occupancy = 0
        # A positive capacity always uses host acknowledgements and page merge.
        self._buffered_write_mode = config.buffered_write_mode
        self._staged_pages: Dict[tuple[int, ...], _StagedPage] = {}
        self._buffer_merge_bytes = 0
        self._media_drain_completion_ns = 0.0
        self._block_wear_state: Dict[str, dict] = {}
        self.__dict__["_read_cache"]: Dict[tuple[int, ...], list[_ReadCacheSlot]] = {}
        self._cache_use = 0
        # OCP 5.4.1 state: one next local DLU for every full-duplex host
        # channel/die.  The legacy bank counter is retained only for old
        # snapshot readers; admission and validation use _write_pointer.
        self._write_pointer: list[int] = (
            [0] * config.host_channels if config.protocol_mode == "ocp_hbf_0_7" else []
        )
        self._seed_history_pages = 0
        self._write_order_errors = 0
        # Logical KV pages are remapped to the next physical DLU in paper
        # OCP serving mode.  Generic and strict protocol cores leave this
        # table empty and retain direct-address semantics.
        self._logical_page_table: dict[int, object] = {}
        self._logical_page_table_identity = True
        self._bank_log: Optional[list[int]] = (
            [0] * (config.channels * config.targets_per_channel * config.dies_per_target
                   * config.luns_per_die * config.planes_per_lun)
            if config.protocol_mode == "ocp_hbf_0_7" else None)
        self._nand_transactions: list[_NandTransaction] = []
        self._nand_numeric_state = None
        self._nand_numeric_enabled = False
        self._nand_state_exposed = False
        self._observation_events: list[dict] = []
        self._last_media_queue_wait_ns = 0.0
        self._observe_media = bool(config.detailed_observability or config.wear_model)
        # Page staging: page index -> (valid ranges, valid bytes, data ready,
        # program end or -1 while filling, host channel). Valid bytes count
        # against their channel until the page's program completes.
        self._tail_pages: Dict[int, tuple] = {}
        self._tail_valid = (0,) * config.host_channels
        self._tail_peak = (0,) * config.host_channels
        self._tail_flushing: list[tuple[float, int]] = []
        self._tail_index = None
        # Full pages may be held by an attention-scoped background task.  The
        # scope index lets a flush touch only its own pages; there is no timer
        # heap or global page scan.
        self._tail_full_scopes: dict[str, set[int]] = {}
        self._tail_defer_scope: Optional[str] = None
        self._tail_full_page_fills = 0
        self._tail_flush_started_pages = 0
        self._tail_flush_discarded_pages = 0
        # Pages whose rows go to the array because the store had no room when
        # they started or grew; they stay there until their allocation is
        # released (their later rows read-modify-write the array page).
        self._tail_spilled: set[int] = set()
        self._tail_spilled_pages = 0
        self._tail_evicted_pages = 0
        # Bytes per channel of other data interleaved over the same store
        # (logic-die SRAM activations); the physical runtime refreshes it.
        self._tail_shared = 0
        # Share of new tail pages the store keeps in the step a sampled run
        # stands for (the stationary sampler sets it); 1 admits by room only.
        self._tail_fraction = 1.
        # Detailed OCP traffic evidence is opt-in and lives only on the live
        # core.  Summary/default runs keep this as None, so no event list is
        # allocated and the existing execution path remains unchanged.
        import os
        self._ocp_diag_events = (
            [] if str(os.environ.get("HBF_OCP_DIAGNOSTICS", "0")).lower()
            in {"1", "true", "yes"} and config.protocol_mode == "ocp_hbf_0_7"
            else None
        )

    def _ocp_diag_append_event(self, kind: str, mapping, *, issue_ns: float,
                               end_ns: Optional[float] = None,
                               logical_page: Optional[int] = None,
                               length: Optional[int] = None) -> None:
        """Append one physical OCP event when evidence collection is enabled."""
        events = self.__dict__.get("_ocp_diag_events")
        if events is None:
            return
        page_bytes = int(self.config.page_bytes)
        base_units = (int(self.config.channels) * int(self.config.targets_per_channel)
                      * int(self.config.dies_per_target) * int(self.config.luns_per_die))
        physical_page = int(mapping.address) // page_bytes
        metadata = getattr(self.config, "metadata", {}) or {}
        owner = metadata.get("physical_owner")
        if not owner:
            host_resource = str(metadata.get("host_resource_id", ""))
            owner = host_resource.split(":nand:host", 1)[0] if ":nand:host" in host_resource else None
        if not owner:
            array_resource = str(metadata.get("array_resource_prefix", ""))
            owner = array_resource.split(":nand:array", 1)[0] if ":nand:array" in array_resource else None
        row = {
            "kind": str(kind),
            "issue_ns": float(issue_ns),
            "end_ns": None if end_ns is None else float(end_ns),
            "channel": int(mapping.channel),
            "host_channel": int(mapping.channel) % int(self.config.host_channels),
            "address": int(mapping.address),
            "length": int(page_bytes if length is None else length),
            "local_dlu": int(physical_page // base_units),
            "plane": int(mapping.plane),
            "block": int(mapping.block),
            "page": int(mapping.page),
            "physical_owner": str(owner or "unscoped"),
        }
        if logical_page is not None:
            row["logical_page"] = int(logical_page)
        events.append(row)

    def begin_transaction(self):
        """Protect program-order, wear and read buffers; caller owns other state."""
        token = _NandTransaction(self)
        if self._nand_state_exposed:
            token.materialize_cache(self)
        self._nand_transactions.append(token)
        return token

    def _finish_transaction(self, token):
        if not self._nand_transactions or self._nand_transactions[-1] is not token:
            raise ValueError("NAND transactions must finish once in reverse begin order")
        self._nand_transactions.pop()

    def commit_transaction(self, token):
        self._finish_transaction(token)

    def rollback_transaction(self, token):
        self._finish_transaction(token)
        token.rollback(self)

    def _nand_before_reference(self):
        self._nand_publish_numeric()
        for token in self._nand_transactions:
            token.materialize_cache(self)
            token.materialize_page_table()

    def _nand_publish_numeric(self, *, exposed=False):
        packed = self.__dict__.get("_nand_numeric_state")
        if exposed:
            self.__dict__["_nand_numeric_enabled"] = False
            self.__dict__["_nand_state_exposed"] = True
            for token in self.__dict__.get("_nand_transactions", ()):
                token.materialize_cache(self)
        if packed is None:
            return
        for token in self._nand_transactions:
            token.capture_numeric(packed)
        packed.publish(self.__dict__["_read_cache"], self.__dict__["_array_ready"],
                       self.__dict__["_buffer_ready"], core=self)
        self._nand_numeric_state = None

    @property
    def _read_cache(self):
        self._nand_publish_numeric(exposed=True)
        return self.__dict__["_read_cache"]

    @_read_cache.setter
    def _read_cache(self, value):
        if "_read_cache" in self.__dict__:
            self._nand_publish_numeric(exposed=True)
        self.__dict__["_read_cache"] = value

    @property
    def _array_ready(self):
        self._nand_publish_numeric(exposed=True)
        return self.__dict__["_array_ready"]

    @_array_ready.setter
    def _array_ready(self, value):
        if "_array_ready" in self.__dict__:
            self._nand_publish_numeric(exposed=True)
        self.__dict__["_array_ready"] = value

    @property
    def _buffer_ready(self):
        self._nand_publish_numeric(exposed=True)
        return self.__dict__["_buffer_ready"]

    @_buffer_ready.setter
    def _buffer_ready(self, value):
        if "_buffer_ready" in self.__dict__:
            self._nand_publish_numeric(exposed=True)
        self.__dict__["_buffer_ready"] = value

    def _touch_cache_bank(self, bank):
        for token in self._nand_transactions:
            token.remember_cache(self, bank)

    def _touch_frontiers(self, key):
        for token in self._nand_transactions:
            token.remember_frontiers(key)

    def _log_programs(self, pages):
        """Advance the strict per-channel OCP write pointer after a program."""
        pages = tuple(int(page) for page in pages)
        if self.config.protocol_mode != "ocp_hbf_0_7":
            return
        next_pointers = self._validate_write_order(pages)
        log = self._bank_log
        if log is not None:
            for token in self._nand_transactions:
                if token.bank_log is None:
                    token.bank_log = list(log)
        base_units = (self.config.channels * self.config.targets_per_channel
                      * self.config.dies_per_target * self.config.luns_per_die)
        per_channel = base_units // self.config.channels
        if next_pointers is not None:
            self._write_pointer = list(next_pointers)
        for page in pages:
            channel = (page % base_units) // per_channel
            if log is not None:
                log[page % len(log)] += 1

    def seed_write_history(self, pages: int) -> None:
        """Seed the OCP write cursors from already resident host history.

        Warmup KV is deliberately preloaded without issuing paid NAND writes,
        but it still occupies physical pages in the host's sequential stream.
        Starting every cursor at zero would therefore make the first sampled
        flush look like a block boundary on every die.  The host keeps the
        cursors balanced: the first ``pages % host_channels`` channels have
        one extra historical page, so channel phases differ by at most one.
        """
        if self.config.protocol_mode != "ocp_hbf_0_7":
            return
        if type(pages) is not int or pages < 0:
            raise ValueError("seed write history pages must be a non-negative integer")
        if self._logical_page_table or self._staged_pages or self._tail_flushing:
            raise ValueError("seed write history must be installed before NAND writes")
        if self._write_order_errors or any(self._write_pointer):
            raise ValueError("seed write history may only initialize empty write cursors")
        channels = int(self.config.host_channels)
        quotient, remainder = divmod(int(pages), channels)
        self._write_pointer = [quotient + int(index < remainder) for index in range(channels)]
        self._seed_history_pages = int(pages)

    def _mapping_for_local_dlu(self, mapping, channel: int, local_dlu: int):
        base_units = (self.config.channels * self.config.targets_per_channel
                      * self.config.dies_per_target * self.config.luns_per_die)
        plane_count = int(self.config.planes_per_lun)
        plane = int(local_dlu) % plane_count
        page_in_block = (int(local_dlu) // plane_count) % int(self.config.pages_per_block)
        block = int(local_dlu) // (plane_count * int(self.config.pages_per_block))
        unit = channel + plane * base_units if self.config.planes_independent else channel
        physical_page = int(local_dlu) * base_units + int(channel)
        return replace(mapping, address=physical_page * self.config.page_bytes,
                       unit=unit, plane=plane, block=block, page=page_in_block,
                       page_offset=0)

    def _phase_batch_mappings(self, pages):
        """Return physical page indices, mappings and next cursors for a write batch."""
        if not self._sequential_write_mode():
            return None, (), tuple(self._write_pointer)
        pages = tuple(int(page) for page in pages)
        pointers = list(self._write_pointer)
        physical_pages = []
        mappings = []
        base_units = (self.config.channels * self.config.targets_per_channel
                      * self.config.dies_per_target * self.config.luns_per_die)
        config = self.config
        page_bytes, capacity = config.page_bytes, config.effective_capacity_bytes
        luns, dies, targets = config.luns_per_die, config.dies_per_target, config.targets_per_channel
        planes, pages_per_block = int(config.planes_per_lun), int(config.pages_per_block)
        host_channels, independent = int(config.host_channels), config.planes_independent
        for page in pages:
            if page < 0:
                raise ValueError("ocp_hbf_0_7 write page must be non-negative")
            # map_nand_address's check and die decode, then the mapping of
            # _mapping_for_local_dlu built directly rather than by replace().
            address = page * page_bytes
            if address >= capacity:
                raise ValueError(f"address {address} is outside modeled capacity {capacity}")
            base = page % base_units
            lun = base % luns
            base //= luns
            die = base % dies
            base //= dies
            target = base % targets
            die_channel = base // targets
            channel = die_channel % host_channels
            local_dlu = pointers[channel]
            plane = local_dlu % planes
            physical_page = local_dlu * base_units + channel
            mappings.append(AddressMapping(
                address=physical_page * page_bytes, channel=die_channel, target=target, die=die, lun=lun,
                unit=channel + plane * base_units if independent else channel, plane=plane,
                block=local_dlu // (planes * pages_per_block), page=(local_dlu // planes) % pages_per_block))
            physical_pages.append(physical_page)
            pointers[channel] += 1
        return tuple(physical_pages), tuple(mappings), tuple(pointers)

    def _commit_phase_batch(self, pages, mappings, next_pointers):
        """Commit a successful native phase write and its compact page table."""
        pages = tuple(int(page) for page in pages)
        mappings = tuple(mappings)
        if len(pages) != len(mappings):
            raise ValueError("phase write pages and mappings must have equal length")
        if self._nand_transactions:
            for token in self._nand_transactions:
                if token.bank_log is None and self._bank_log is not None:
                    token.bank_log = list(self._bank_log)
        self._write_pointer = list(next_pointers)
        for token in self._nand_transactions:
            token.remember_pages(self._logical_page_table, pages)
        for page, mapping in zip(pages, mappings):
            self._logical_page_table[int(page)] = mapping
            if int(mapping.address) // int(self.config.page_bytes) != int(page):
                self._logical_page_table_identity = False
            if self._bank_log is not None:
                self._bank_log[page % len(self._bank_log)] += 1

    def _validate_write_order(self, pages):
        """Check local-DLU continuity without mutating clocks or pointers.

        Paper serving windows can begin in the middle of a previously written
        local-DLU stream.  Such configurations declare ``write_pointer_phase``
        and use the first observed page on each channel as the replay phase;
        every later page is still required to advance by exactly one DLU.
        Strict protocol configs omit the metadata and therefore require DLU 0.
        """
        if self.config.protocol_mode != "ocp_hbf_0_7":
            return None
        pages = tuple(int(page) for page in pages)
        if not pages:
            return tuple(self._write_pointer)
        base_units = (self.config.channels * self.config.targets_per_channel
                      * self.config.dies_per_target * self.config.luns_per_die)
        per_channel = base_units // self.config.channels
        pointers = list(self._write_pointer)
        phase_replay = self.config.metadata.get("write_pointer_phase") == "first_write"
        for page in pages:
            if page < 0:
                raise ValueError("ocp_hbf_0_7 write page must be non-negative")
            channel = (page % base_units) // per_channel
            local_dlu = page // base_units
            expected = pointers[channel]
            if phase_replay:
                # The request address is a logical page-table key.  The
                # physical DLU is chosen by this channel's write pointer;
                # every program therefore advances exactly one location even
                # when a logical tail page is rewritten.
                pointers[channel] += 1
                continue
            if local_dlu != expected:
                self._write_order_errors += 1
                raise ValueError(
                    "ocp_hbf_0_7 write order violation on host channel {}: "
                    "local DLU {}, expected {}".format(channel, local_dlu, expected)
                )
            pointers[channel] += 1
        return tuple(pointers)

    def _sequential_write_mode(self) -> bool:
        return (self.config.protocol_mode == "ocp_hbf_0_7"
                and self.config.metadata.get("write_pointer_phase") == "first_write")

    def _program_mapping(self, page, mapping):
        """Map a logical page to this channel's next physical DLU."""
        if not self._sequential_write_mode():
            return mapping
        channel = int(mapping.channel) % self.config.host_channels
        local_dlu = int(self._write_pointer[channel])
        return self._mapping_for_local_dlu(mapping, channel, local_dlu)

    def _read_mapping(self, page, mapping):
        if not self._sequential_write_mode():
            return mapping
        return self._logical_page_table.get(int(page), mapping)

    def _read_pages(self, first, count):
        """Physical page indices of the ``count`` logical pages from ``first``.

        The compiled page machines decode a bank and tag from these indices.
        Under the OCP sequential write mode a programmed page lives where its
        channel's write pointer put it (the host page table); a page the run
        never programmed reads its own address, as in ``_read_mapping``.
        """
        import numpy as np
        if (not self._sequential_write_mode()
                or self.__dict__.get("_logical_page_table_identity", False)):
            return np.arange(first, first + count, dtype=np.int64)
        pages = range(first, first + count)
        table = self._logical_page_table
        try:
            # An unmapped page's default is its own index, so a span the
            # table does not touch converts without building mappings.
            return np.fromiter(map(table.get, pages, pages), dtype=np.int64, count=count)
        except TypeError:
            page_bytes = self.config.page_bytes
            return np.fromiter((page if mapping is None else mapping.address // page_bytes
                                for page, mapping in zip(pages, map(table.get, pages))),
                               dtype=np.int64, count=count)

    def _remember_program(self, page, mapping):
        if self._sequential_write_mode():
            for token in self._nand_transactions:
                token.remember_pages(self._logical_page_table, (int(page),))
            self._logical_page_table[int(page)] = mapping
            if int(mapping.address) // int(self.config.page_bytes) != int(page):
                self._logical_page_table_identity = False

    def _auto_erase_needed(self, mapping) -> bool:
        if self.config.protocol_mode != "ocp_hbf_0_7":
            return False
        # OCP 5.7 maps one NAND block over every plane in the channel.  The
        # sequential local-DLU stream reaches that block exactly once, at
        # plane 0/page 0; the following plane/page-zero programs belong to the
        # same already-open block and must not charge another erase.
        # ``mapping`` is the physical page selected before the write cursor is
        # advanced, so this remains correct for phase-replay writes too.
        return int(mapping.plane) == 0 and int(mapping.page) == 0

    def _reserve_auto_erase(self, mapping, earliest, *, eligible_ns, stages=None) -> float:
        """Reserve one OCP block erase across every plane in its channel.

        OCP 5.7's R1/R2/R3 address arithmetic makes the block unit span all
        planes of a channel.  The page-zero command is therefore represented
        by one observable erase event, while every plane array receives the
        same busy interval before its page program can start.
        """
        if not self._auto_erase_needed(mapping):
            return float(earliest)
        base_units = (self.config.channels * self.config.targets_per_channel
                      * self.config.dies_per_target * self.config.luns_per_die)
        base_unit = int(mapping.unit) % base_units
        if self.config.planes_independent:
            plane_mappings = tuple(replace(mapping, unit=base_unit + plane * base_units, plane=plane)
                                   for plane in range(int(self.config.planes_per_lun)))
        else:
            plane_mappings = (mapping,)
        common_start = float(earliest)
        for plane_mapping in plane_mappings:
            array_id = self._array_id(plane_mapping)
            resource = self._array_resource_id(plane_mapping)
            lanes = self.timeline.lane_available.get(resource)
            common_start = max(common_start, self.__dict__["_array_ready"].get(array_id, 0.0),
                               self.__dict__["_buffer_ready"].get(array_id, 0.0),
                               min(lanes) if lanes else self.timeline.available_ns(resource))
        erase = self._media_stage("BLOCK_ERASE", plane_mappings[0], common_start, self.config.block_erase_ns,
                                  eligible_ns=eligible_ns)
        if stages is not None and self.capture_details:
            stages.append(erase)
        erase_end = float(erase.end_ns)
        for plane_mapping in plane_mappings[1:]:
            array_id = self._array_id(plane_mapping)
            self._touch_frontiers(array_id)
            shadow = self._array_stage("BLOCK_ERASE", self._array_resource_id(plane_mapping),
                                       common_start, self.config.block_erase_ns)
            erase_end = max(erase_end, float(shadow.end_ns))
            self.__dict__["_array_ready"][array_id] = float(shadow.end_ns)
            self.__dict__["_buffer_ready"][array_id] = float(shadow.end_ns)
            self.timeline._remember_resource(array_id)
            self.timeline.ready_ns[array_id] = float(shadow.end_ns)
        # Keep the representative plane's Python frontier aligned as well;
        # the caller will advance it again when PAGE_PROGRAM is reserved.
        first_id = self._array_id(plane_mappings[0])
        self._touch_frontiers(first_id)
        self.__dict__["_array_ready"][first_id] = erase_end
        self.__dict__["_buffer_ready"][first_id] = erase_end
        self.timeline._remember_resource(first_id)
        self.timeline.ready_ns[first_id] = erase_end
        return erase_end



    def buffered_write_mode(self) -> bool:
        """Return whether this core exposes host-ack/media-drain semantics."""

        return self._buffered_write_mode

    @property
    def pending_write_buffer_bytes(self) -> int:
        """Dirty host payload awaiting flush, excluding scheduled media drains."""

        return int(self._write_buffer_occupancy if self._buffered_write_mode else 0)

    @property
    def buffered_page_count(self) -> int:
        """Number of dirty physical pages awaiting a media program."""

        return len(self._staged_pages)

    @property
    def inflight_write_buffer_bytes(self) -> int:
        """Bytes retained by a media drain or staging reader at the last clock observation."""

        return sum(byte_count for _release_ns, byte_count in self._write_buffer_pending) if self._buffered_write_mode else 0

    def _held_write_buffer_bytes(self) -> int:
        return self._write_buffer_occupancy + self.inflight_write_buffer_bytes

    def _array_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("array_resource_prefix", "nand:array"))
        return f"{prefix}:{mapping.unit}"

    def _array_resource_id(self, mapping) -> str:
        """Return the execution slot, without changing physical mapping."""
        array_id = self._array_id(mapping)
        if self.config.parallel_units is None:
            return self.timeline.canonical_resource_id(array_id)
        prefix = str(self.config.metadata.get("parallel_resource_prefix", "nand:parallel"))
        return self.timeline.canonical_resource_id(f"{prefix}:{mapping.unit % self.config.parallel_units}")

    def _channel_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("channel_resource_prefix", "nand:channel"))
        return self.timeline.canonical_resource_id(f"{prefix}:{mapping.channel}")

    def _host_id(self, mapping=None, operation=None) -> str:
        prefix = str(self.config.metadata.get("host_resource_id", "nand:host"))
        if self.config.host_channels > 1:
            if mapping is None:
                raise ValueError("a physical mapping is required for a multi-channel host transfer")
            prefix += ":" + str(mapping.channel % self.config.host_channels)
        if self.config.host_full_duplex:
            if operation is None:
                raise ValueError("a transfer direction is required for a full-duplex host interface")
            prefix += ":" + Operation(operation).value
        return self.timeline.canonical_resource_id(prefix)

    def _scratchpad_id(self, mapping, operation) -> str:
        """Per-channel high-priority logic-die SRAM lane (OCP 5.4.2/11.3)."""
        prefix = str(self.config.metadata.get("scratchpad_resource_prefix", "nand:scratchpad"))
        resource = f"{prefix}:{mapping.channel % self.config.host_channels}:{Operation(operation).value}"
        return self.timeline.canonical_resource_id(resource)

    def _staging_host_id(self, mapping, operation) -> str:
        """Resource for a logic-die staging transfer.

        OCP HBF+ and CONV+ expose the logic-die SRAM as a per-channel
        scratchpad.  Generic NAND keeps its historical host link resource so
        its resident numeric flush and scalar fallback remain bit exact.
        """
        if self.config.protocol_mode == "ocp_hbf_0_7":
            return self._scratchpad_id(mapping, operation)
        return self._host_id(mapping, operation)

    def _host_bandwidth(self, operation) -> float:
        """Configured rates are device aggregates, divided among host channels."""
        return self.config.host_bandwidth_for(operation) / self.config.host_channels

    def _admission_depth_scope(self) -> str:
        if self.config.queue_scope == "host_channel":
            return "outstanding page commands per host channel sampled at admission, including the admitted command"
        return "outstanding host requests sampled at admission, including the admitted request; not NAND page queues"

    @property
    def _inflight_operations(self):
        value = self.__dict__["_inflight_operations"]
        if type(value) is _HostPopulation:
            value = self.__dict__["_inflight_operations"] = list(value)
        return value

    @_inflight_operations.setter
    def _inflight_operations(self, value):
        self.__dict__["_inflight_operations"] = value

    @property
    def _inflight(self):
        value = self.__dict__["_inflight"]
        if type(value) is _HostCompletions:
            value = self.__dict__["_inflight"] = list(value)
        return value

    @_inflight.setter
    def _inflight(self, value):
        self.__dict__["_inflight"] = value

    @property
    def _host_inflight(self):
        value = self.__dict__["_host_inflight"]
        for host, commands in value.items():
            if type(commands) is _HostPopulation:
                value[host] = list(commands)
        return value

    @_host_inflight.setter
    def _host_inflight(self, value):
        self.__dict__["_host_inflight"] = value

    def _admit_host_request(self, request, mapping=None):
        """Find a free command credit without changing another host's clock."""
        host_channel = None
        if self.config.queue_scope == "host_channel":
            if mapping is None:
                mapping = map_nand_address(self.config, request.address)
            host_channel = mapping.channel % self.config.host_channels
            population = self.__dict__["_host_inflight"].get(host_channel, ())
            # _acceptance_ns is a device clock only when the caller explicitly
            # advances/drains the core; ordinary host admissions never set it.
            acceptance = max(self._host_acceptance_ns.get(host_channel, 0.), self._acceptance_ns)
        else:
            population = self.__dict__["_inflight_operations"]
            acceptance = self._acceptance_ns
        effective = max(request.arrival_ns, acceptance)
        if type(population) is not _HostPopulation:
            population = _HostPopulation(population)
        admitted = population.preview(effective, self.config.max_outstanding_requests,
            self.config.max_outstanding_write_requests if request.operation is Operation.WRITE else None)
        return admitted.effective, admitted, host_channel

    def _record_host_admission(self, host_channel, effective, admitted, completion, operation):
        population = admitted.population
        population.record(effective, completion, operation)
        if host_channel is None:
            self._acceptance_ns = effective
            self.__dict__["_inflight_operations"] = population
            self.__dict__["_inflight"] = _HostCompletions(population)
        else:
            self._host_acceptance_ns[host_channel] = effective
            self.__dict__["_host_inflight"][host_channel] = population

    @staticmethod
    def _bank_key(mapping) -> tuple[int, ...]:
        return (mapping.channel, mapping.target, mapping.die, mapping.lun, mapping.plane)

    def _block_key(self, mapping) -> tuple[int, ...]:
        return (*self._bank_key(mapping), mapping.block)

    def _cache_slot(self, mapping) -> tuple[Optional[_ReadCacheSlot], bool]:
        count = self.config.read_cache_pages_per_bank
        if not count:
            return None, False
        bank = self._bank_key(mapping)
        if self._nand_transactions:
            self._touch_cache_bank(bank)
        slots = self.__dict__["_read_cache"].get(bank)
        if slots is None:
            slots = [_ReadCacheSlot() for _ in range(count)]
            self.__dict__["_read_cache"][bank] = slots
        key = self._staged_key(mapping)
        slot = next((slot for slot in slots if slot.page_key == key), None)
        hit = slot is not None
        if slot is None:
            slot = min(slots, key=lambda candidate: (candidate.page_key is not None, candidate.last_use))
        self._cache_use += 1
        slot.last_use = self._cache_use
        return slot, hit

    def _invalidate_cache(self, mapping, *, block=False) -> None:
        bank = self._bank_key(mapping)
        for slot in self.__dict__["_read_cache"].get(bank, ()):
            key = slot.page_key
            if key is not None and (key == self._staged_key(mapping)
                                    or (block and key[-2] == mapping.block)):
                # Invalidate the tag, not the physical buffer's outstanding
                # read lifetime. Its next fill must still wait for all readers.
                if self._nand_transactions:
                    self._touch_cache_bank(bank)
                slot.page_key = None

    def _bandwidth_ceiling_gb_s(self, operation: Optional[Operation | str] = None) -> float:
        """Return the end-to-end payload ceiling for the NAND data path.

        The configured host rates aggregate all independent host channels.
        The smaller of host and summed internal bandwidth is the transfer ceiling;
        a request touching fewer channels or paying array latency can be slower.
        The reservation timeline still models each leg separately; this helper
        only fixes the derived bottleneck/utilization metric.
        """
        if operation is not None and Operation(operation) is Operation.ERASE:
            return 0.0  # Erase operates on the array and carries no host payload.
        # Batch summaries use the shared interface ceiling. Each transaction
        # uses its own directional host link.
        host_bandwidth = (
            float(self.config.host_bandwidth_gb_s)
            if operation is None
            else self.config.host_bandwidth_for(operation)
        )
        return min(host_bandwidth, self.config.channels * float(self.config.internal_bandwidth_gb_s))

    def _array_stage(self, name: str, resource: str, earliest: float, duration: float, bytes: int = 0) -> StageTiming:
        reservation = self.timeline.reserve(resource, earliest, duration)
        return StageTiming(name, reservation.start_ns, reservation.end_ns, resource, bytes)

    def _media_stage(self, name, mapping, earliest, duration, *, eligible_ns):
        """Execute an actual mapped array operation with optional feedback."""
        resource = self._array_resource_id(mapping)
        if not self._observe_media:
            return self._array_stage(name, resource, earliest, duration)
        prefix = str(self.config.metadata.get("array_resource_prefix", "nand:array"))
        plane = prefix+":plane:"+":".join(str(getattr(mapping, key)) for key in
            ("channel", "target", "die", "lun", "plane"))
        block = plane+":block:"+str(mapping.block)
        lanes = self.timeline.lane_available.get(resource)
        start = max(float(earliest), min(lanes) if lanes else self.timeline.available_ns(resource))
        wear_config = self.config.wear_model
        wear = self._block_wear_state.get(block, {"programs": 0, "erases": 0})
        pe_before = (wear_config.initial_pe_cycles if wear_config else 0)+wear["erases"]
        wear_scale = 1.
        if name == "PAGE_PROGRAM" and wear_config and wear_config.program_latency_growth_at_rated_pe:
            growth = wear_config.program_latency_growth_at_rated_pe
            try:
                extra = growth*pe_before/wear_config.rated_pe_cycles
            except OverflowError:
                extra = None
            if extra is None or not isfinite(extra):
                # Counts are integers without a float-sized bound. Compute
                # the rare oversized ratio exactly before applying its finite
                # authored cap; even zero growth must not coerce a huge count.
                from fractions import Fraction
                extra = min(Fraction.from_float(float(wear_config.max_program_latency_scale-1.)),
                    Fraction.from_float(float(growth))*pe_before/wear_config.rated_pe_cycles)
            wear_scale = min(wear_config.max_program_latency_scale, 1.+float(extra))
        work = float(duration)*wear_scale
        stage = self._array_stage(name, resource, start, work)
        if name in {"PAGE_PROGRAM", "BLOCK_ERASE"}:
            self._ocp_diag_append_event(
                "program" if name == "PAGE_PROGRAM" else "erase",
                mapping, issue_ns=stage.start_ns, end_ns=stage.end_ns,
            )
        if wear_config:
            wear = {"programs": wear["programs"]+int(name == "PAGE_PROGRAM"),
                    "erases": wear["erases"]+int(name == "BLOCK_ERASE")}
            if self._nand_transactions:
                previous = self._block_wear_state.get(block)
                for token in self._nand_transactions:
                    token.block_wear.setdefault(block, previous)
            self._block_wear_state[block] = wear
        self._observation_events.append({
            "operation": "program" if name == "PAGE_PROGRAM" else "erase" if name == "BLOCK_ERASE" else "read",
            "stage": name, "plane_id": plane, "block_id": block,
            "execution_resource": resource, "eligible_ns": float(eligible_ns),
            "start_ns": stage.start_ns, "end_ns": stage.end_ns,
            "baseline_service_ns": float(duration), "wear_extra_ns": float(duration)*(wear_scale-1.),
            "physical_bytes": self.config.page_bytes if name != "BLOCK_ERASE" else 0,
            "pe_cycles_before": pe_before if wear_config else None,
            "pe_cycles_after": pe_before+int(name == "BLOCK_ERASE") if wear_config else None,
            "block_programs_after": wear["programs"] if wear_config else None,
            "block_erases_after": wear["erases"] if wear_config else None,
            "wear_enabled": wear_config is not None,
            "rated_pe_cycles": wear_config.rated_pe_cycles if wear_config else None,
        })
        return stage

    def _observations(self, first=0):
        if not self._observe_media:
            return {}
        return {"nand_observations": {
            "events": tuple(self._observation_events[first:]),
            "configured_planes_by_owner": {
                str(self.config.metadata.get("array_resource_prefix", "nand:array")):
                    self.config.channels*self.config.targets_per_channel*self.config.dies_per_target*self.config.luns_per_die*self.config.planes_per_lun,
            },
            "wear_enabled": self.config.wear_model is not None,
            "scope": "actual modeled physical page array operations; queue is eligible-to-array-start waiting, excluding in-service operations; physical mapping geometry is authored, not calibrated",
        }}

    def _write_buffer_payload_bytes(self, request: AccessRequest) -> int:
        if request.operation is not Operation.WRITE or self.config.write_buffer_bytes <= 0:
            return 0
        return sum(
            int(segment.host_transfer_bytes)
            for segment in split_nand_request(request, self.config)
        )

    def _validate_request(self, request: AccessRequest) -> None:
        """Reject invalid geometry or admission size before mutating calendars."""
        validate_nand_request(request, self.config)
        payload_bytes = self._write_buffer_payload_bytes(request)
        if payload_bytes > self.config.write_buffer_bytes:
            raise ValueError(
                "write request requires {} write-buffer bytes, exceeds configured capacity {}"
                .format(payload_bytes, self.config.write_buffer_bytes)
            )

    @staticmethod
    def _staged_key(mapping) -> tuple[int, ...]:
        """Build a stable physical-page key from an address mapping."""

        return (
            int(mapping.stack), int(mapping.lane), int(mapping.subchannel),
            int(mapping.pseudo_channel), int(mapping.rank),
            int(mapping.bank_group), int(mapping.bank),
            int(mapping.channel), int(mapping.target), int(mapping.die),
            int(mapping.lun), int(mapping.plane), int(mapping.block), int(mapping.page),
        )

    def _stage_page(self, segment, request: AccessRequest, ready_ns: float) -> tuple[int, int]:
        """Merge one write segment into staging and return (merge, new)."""

        key = self._staged_key(segment.mapping)
        page = self._staged_pages.get(key)
        merge_bytes = int(segment.host_transfer_bytes) if page is not None else 0
        if page is None:
            page = _StagedPage(mapping=segment.mapping)
            self._staged_pages[key] = page
        page.host_bytes += int(segment.host_transfer_bytes)
        page.logical_bytes += int(segment.logical_bytes)
        page.merged_host_bytes += merge_bytes
        page.request_count += 1
        page.ready_ns = max(page.ready_ns, ready_ns)
        start = max(request.address, segment.address) - segment.address
        end = min(request.address + request.byte_count, segment.address + self.config.page_bytes) - segment.address
        merged_ranges = []
        for left, right in sorted((*page.ranges, (start, end))):
            if merged_ranges and left <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(right, merged_ranges[-1][1]))
            else:
                merged_ranges.append((left, right))
        page.ranges = tuple(merged_ranges)
        self._buffer_merge_bytes += merge_bytes
        return merge_bytes, 1 if merge_bytes else 0

    def _release_write_buffer(self, now_ns: float) -> None:
        if not self._write_buffer_pending:
            return
        retained = []
        released = 0
        for release_ns, byte_count in self._write_buffer_pending:
            if release_ns <= now_ns:
                released += byte_count
            else:
                retained.append((release_ns, byte_count))
        if released:
            self._write_buffer_pending = retained

    def _admit_write_buffer(
        self, request: AccessRequest, effective_arrival: float,
    ) -> tuple[float, int, float, dict]:
        """Delay admission until a finite write buffer has room.

        Capacity is measured in host payload bytes, matching the host transfer
        stages and avoiding a second page-level accounting convention.
        """
        payload_bytes = self._write_buffer_payload_bytes(request)
        if not payload_bytes:
            if self._buffered_write_mode and request.operation is Operation.ERASE and self._staged_pages:
                drained = self.flush_write_buffer(now_ns=effective_arrival)
                arrival = max(effective_arrival, drained["media_drain_completion_ns"])
                self._release_write_buffer(arrival)
                return arrival, 0, arrival - effective_arrival, drained
            self._release_write_buffer(effective_arrival)
            return effective_arrival, 0, 0.0, {}
        capacity = self.config.write_buffer_bytes
        if payload_bytes > capacity:
            raise ValueError(
                "write request requires {} write-buffer bytes, exceeds configured capacity {}"
                .format(payload_bytes, capacity)
            )
        initial_arrival = effective_arrival
        drain_counters = {}
        # Acknowledged payload remains occupied until a media drain completes.
        self._release_write_buffer(effective_arrival)
        while self._held_write_buffer_bytes() + payload_bytes > capacity:
            if self._staged_pages:
                drained = self.flush_write_buffer(now_ns=effective_arrival)
                for key in ("physical_programs", "physical_write_bytes", "pages_read", "physical_read_bytes", "internal_transfer_bytes"):
                    drain_counters[key] = drain_counters.get(key, 0) + drained.get(key, 0)
                effective_arrival = max(effective_arrival, float(drained["media_drain_completion_ns"]))
            else:
                # An explicit flush retains its sources until media completion.
                future = [end for end, _count in self._write_buffer_pending if end > effective_arrival]
                if not future:
                    raise RuntimeError("write buffer occupancy cannot be drained")
                effective_arrival = min(future)
            self._release_write_buffer(effective_arrival)
        return effective_arrival, payload_bytes, max(0.0, effective_arrival - initial_arrival), drain_counters

    def _reserve_write_buffer(self, payload_bytes: int) -> None:
        if payload_bytes <= 0:
            return
        self._write_buffer_occupancy += payload_bytes

    def discard_staged_pages(self, address: int, byte_count: int, *, now_ns: Optional[float] = None) -> dict:
        """Discard dirty pages from a caller-owned complete physical-page range.

        The caller must own every addressed page exclusively. Already scheduled
        media work is never cancelled here. Prior host traffic remains billed,
        and its staging source is retained until both host arrival and any read
        of that source finish. No replacement program or device-wide flush is
        introduced. The returned release time concerns the discarded sources.
        """
        page_bytes = self.config.page_bytes
        if (type(address) is not int or address < 0 or address % page_bytes
                or type(byte_count) is not int or byte_count <= 0 or byte_count % page_bytes
                or address + byte_count > self.config.effective_capacity_bytes):
            raise ValueError("discard range must cover whole aligned NAND pages within capacity")
        if now_ns is not None and (isinstance(now_ns, bool) or not isinstance(now_ns, (int, float))
                                   or not isfinite(now_ns) or now_ns < 0):
            raise ValueError("now_ns must be finite and non-negative")
        if now_ns is not None:
            self._acceptance_ns = max(self._acceptance_ns, float(now_ns))
        self._release_write_buffer(self._acceptance_ns)
        selected = [(key, staged) for key, staged in self._staged_pages.items()
                    if address <= staged.mapping.address < address + byte_count]
        released_at = self._acceptance_ns
        host_bytes = logical_bytes = merge_bytes = 0
        for key, staged in selected:
            del self._staged_pages[key]
            host_bytes += staged.host_bytes
            logical_bytes += staged.logical_bytes
            merge_bytes += staged.merged_host_bytes
            release = max(staged.ready_ns, staged.reader_completion_ns)
            released_at = max(released_at, release)
            if release > self._acceptance_ns:
                self._write_buffer_pending.append((release, staged.host_bytes))
        self._write_buffer_occupancy -= host_bytes
        self._buffer_merge_bytes -= merge_bytes
        return {
            "buffered_write_mode": self._buffered_write_mode,
            "discarded_pages": len(selected), "discarded_host_bytes": host_bytes,
            "discarded_logical_bytes": logical_bytes, "discarded_merge_bytes": merge_bytes,
            "write_buffer_release_completion_ns": released_at,
            "physical_programs": 0, "physical_read_bytes": 0, "physical_write_bytes": 0,
            "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
        }

    def flush_write_buffer(self, *, now_ns: Optional[float] = None) -> dict:
        """Program all dirty staged pages and return drain accounting.

        Host acknowledgements returned by buffered writes are deliberately
        separate from this media completion.  The return value is a small
        JSON-friendly mapping so serving/reporting code can include it without
        introducing a synthetic request into the event stream.
        """

        self._nand_publish_numeric()
        observation_start = len(self._observation_events)
        if now_ns is not None and (isinstance(now_ns, bool) or not isfinite(now_ns) or now_ns < 0):
            raise ValueError("now_ns must be finite and non-negative")
        if now_ns is not None:
            # An explicit clock advance releases completed drains, and later
            # submissions cannot travel back before that observed instant.
            self._acceptance_ns = max(self._acceptance_ns, float(now_ns))
            self._release_write_buffer(self._acceptance_ns)
        if not self._buffered_write_mode:
            return {
                "buffered_write_mode": False,
                "flushed_pages": 0,
                "merge_bytes": 0,
                "physical_programs": 0,
                "physical_write_bytes": 0,
                "media_drain_start_ns": self._acceptance_ns,
                "media_drain_completion_ns": self._acceptance_ns,
                "write_completion": "media_program_complete",
            }
        if not self._staged_pages:
            value = float(
                max(now_ns, self._acceptance_ns) if now_ns is not None else max(
                    self._acceptance_ns, max(self.timeline.ready_ns.values(), default=0.0)
                )
            )
            return {
                "buffered_write_mode": True,
                "flushed_pages": 0,
                "merge_bytes": 0,
                "physical_programs": 0,
                "physical_write_bytes": 0,
                "media_drain_start_ns": value,
                "media_drain_completion_ns": max(value, self._media_drain_completion_ns),
                "write_buffer_release_completion_ns": max(value, max(
                    (end for end, _count in self._write_buffer_pending), default=value)),
                "write_completion": "media_program_complete",
            }
        cursor = float(
            max(now_ns, self._acceptance_ns) if now_ns is not None else max(
                self._acceptance_ns, max(self.timeline.ready_ns.values(), default=0.0)
            )
        )
        start_ns: Optional[float] = None
        earliest_ns = cursor
        physical_programs = 0
        physical_write_bytes = 0
        internal_transfer_bytes = 0
        pages_read = 0
        host_bytes = 0
        staged_items = tuple(sorted(self._staged_pages.values(),
                                    key=lambda item: int(item.mapping.address // self.config.page_bytes)))
        self._validate_write_order(tuple(int(item.mapping.address // self.config.page_bytes)
                                        for item in staged_items))
        erases = 0
        for staged in staged_items:
            m = staged.mapping
            self._invalidate_cache(m)
            array_id = self._array_id(m)
            array_resource = self._array_resource_id(m)
            dependency = max(earliest_ns, staged.ready_ns, self.__dict__["_array_ready"].get(array_id, 0.0), self.__dict__["_buffer_ready"].get(array_id, 0.0))
            if staged.ranges != ((0, self.config.page_bytes),):
                read = self._media_stage("RMW_PAGE_READ", m, dependency, self.config.page_read_ns,
                                         eligible_ns=max(earliest_ns, staged.ready_ns))
                old_data = self.timeline.transfer(
                    self._channel_id(m), self.config.transfer_page_bytes,
                    read.end_ns, self.config.internal_bandwidth_gb_s, direction="read",
                )
                dependency = old_data.end_ns
                internal_transfer_bytes += old_data.bytes
                pages_read += 1
                start_ns = read.start_ns if start_ns is None else min(start_ns, read.start_ns)
            internal = self.timeline.transfer(
                self._channel_id(m), self.config.transfer_page_bytes,
                dependency, self.config.internal_bandwidth_gb_s,
                direction="write",
            )
            erase_end = self._reserve_auto_erase(m, internal.end_ns,
                                                 eligible_ns=internal.end_ns)
            if erase_end > internal.end_ns:
                erases += 1
            program = self._media_stage(
                "PAGE_PROGRAM", m, max(internal.end_ns, erase_end),
                self.config.page_program_ns, eligible_ns=internal.end_ns
            )
            self._log_programs((int(m.address // self.config.page_bytes),))
            self._touch_frontiers(array_id)
            self.__dict__["_array_ready"][array_id] = program.end_ns
            self._touch_frontiers(array_id)
            self.__dict__["_buffer_ready"][array_id] = program.end_ns
            self.timeline._remember_resource(array_id)
            self.timeline.ready_ns[array_id] = program.end_ns
            start_ns = internal.start_ns if start_ns is None else min(start_ns, internal.start_ns)
            cursor = max(cursor, program.end_ns)
            physical_programs += 1
            physical_write_bytes += self.config.page_bytes
            internal_transfer_bytes += internal.bytes
            host_bytes += staged.host_bytes
            # A read can still be sending this host copy through another host
            # lane after programming finishes. Keep its source allocation until
            # both users finish, without charging that reader time as media work.
            self._write_buffer_pending.append((max(program.end_ns, staged.reader_completion_ns), staged.host_bytes))
        merge_bytes = int(self._buffer_merge_bytes)
        flushed_pages = len(self._staged_pages)
        self._staged_pages.clear()
        self._buffer_merge_bytes = 0
        self._write_buffer_occupancy = 0
        cursor = max(cursor, self._media_drain_completion_ns)
        self._media_drain_completion_ns = cursor
        return {
            "buffered_write_mode": True,
            "flushed_pages": flushed_pages,
            "merge_bytes": merge_bytes,
            "physical_programs": physical_programs,
            "erase_operations": erases,
            "physical_write_bytes": physical_write_bytes,
            "pages_read": pages_read,
            "rmw_read_operations": pages_read,
            "physical_read_bytes": pages_read * self.config.page_bytes,
            "host_transfer_bytes": host_bytes,
            "internal_transfer_bytes": internal_transfer_bytes,
            "media_drain_start_ns": float(start_ns if start_ns is not None else cursor),
            "media_drain_completion_ns": float(cursor),
            "write_buffer_release_completion_ns": max(cursor, max(
                (end for end, _count in self._write_buffer_pending), default=cursor)),
            "write_completion": "media_program_complete",
            **self._observations(observation_start),
        }

    def submit(self, request: AccessRequest, *, _resource_sink=None) -> TransactionResult:
        depth = len(self.timeline._owned_transactions)
        try:
            if _resource_sink is None:
                return self._submit_live(request)
            result = self._submit_live(request)
            if any(name in result.counters for name in _RESOURCE_MAP_FIELDS):
                from ._resource_ledger import consume_task_resource_sink
                consume_task_resource_sink(_resource_sink, result.counters)
                result = _without_resource_maps(result)
            return result
        finally:
            self.timeline._finish_core_accounting(depth)

    def submit_many(self, requests, *, _resource_sink=None):
        """Submit requests that arrive together as one controller window.

        A native generic READ batch is scheduled by the out-of-order
        controller: host credits are still taken in request order, but
        channels and the host link serve pages as they become ready, so one
        request's busy plane no longer holds back the others. Every page is
        read where the host page table put it (``_read_pages``). OCP HBF
        v0.7 5.3.1 lets reads of different AXI IDs complete out of order and
        keeps only the reads of one bank in order, which the controller's
        per-plane FIFO does. Other batches are submitted one request at a
        time.

        With a private ``_resource_sink`` the native batches return no
        resource maps; ``fold_read_bills`` bills the sink from their compact
        per-request bills, in input order.
        """
        requests = tuple(requests)
        if (len(requests) >= 2 and self._nand_numeric_enabled
                and all(type(request) is AccessRequest and request.operation is Operation.READ
                        for request in requests)):
            from ._nand_acceleration import execute_read_batch, fold_read_bills
            bills = None if _resource_sink is None else []
            if self.config.queue_scope == "device":
                results = execute_read_batch(self, requests, _bills=bills)
                if results is not None:
                    if bills:
                        fold_read_bills(_resource_sink, self.timeline.last_intervals,
                                        [(range(len(requests)), bills[0])], {})
                    return tuple(results)
            elif self.config.queue_scope == "host_channel":
                # A host-channel queue has an independent credit clock.  Run
                # one native batch per channel so the compiled controller's
                # global credit counter is exactly the channel-local 256
                # command population.  A request is admitted on the host
                # channel of its first page, but its later pages can lie on
                # other host channels' NAND channels, so a sink is billed
                # only after every group ran, in input order.  A one-request
                # group remains on the scalar path because batching cannot
                # amortize packing.
                groups = {}
                for index, request in enumerate(requests):
                    mapping = map_nand_address(self.config, request.address)
                    host = mapping.channel % self.config.host_channels
                    groups.setdefault(host, []).append((index, request))
                results = [None] * len(requests)
                native, scalar = [], {}
                for host, group in groups.items():
                    collected = None if bills is None else []
                    batch = None
                    if len(group) >= 2:
                        batch = execute_read_batch(
                            self, tuple(request for _index, request in group),
                            _bills=collected, _host_channel=host)
                    if batch is None:
                        for index, request in group:
                            results[index] = self.submit(request)
                            if bills is not None:
                                scalar[index] = results[index].counters
                    else:
                        for (index, _request), result in zip(group, batch):
                            results[index] = result
                        if collected:
                            native.append(([index for index, _request in group], collected[0]))
                if bills is not None:
                    fold_read_bills(_resource_sink, self.timeline.last_intervals, native, scalar)
                    for index in scalar:
                        if any(name in results[index].counters for name in _RESOURCE_MAP_FIELDS):
                            results[index] = _without_resource_maps(results[index])
                return tuple(results)
        if _resource_sink is None:
            results = tuple(self.submit(request) for request in requests)
        else:
            results = tuple(self.submit(request, _resource_sink=_resource_sink) for request in requests)
        return results

    def _submit_live(self, request: AccessRequest) -> TransactionResult:
        if self.config.page_staging != "none" and type(request) is AccessRequest:
            result = self._submit_staged(request)
            if result is not None:
                return result
        # A native batch executor may decline a single request that is wholly
        # served by deferred pages.  Its scalar staging branch does not need
        # the packed NAND calendars, so keep those arrays resident instead of
        # publishing every bank merely to account for a host transfer.
        if (self._nand_numeric_state is not None
                and type(request) is AccessRequest
                and request.operation is Operation.READ):
            first = request.address // self.config.page_bytes
            last = (request.address + request.byte_count - 1) // self.config.page_bytes
            readable = self._tail_readable(first, last, request.arrival_ns)
            if len(readable) == last - first + 1:
                self._validate_request(request)
                return self._submit_scalar_request(request)
        # Ordinary native requests can bill the packed cumulative calendars
        # directly. The outer physical transaction still owns their undo;
        # an additional per-request timeline transaction is unnecessary.
        if (self._nand_numeric_enabled and not self._nand_state_exposed
                and type(request) is AccessRequest
                and _generic_bulk_span(request, self.config)
                and (self._nand_numeric_state is not None
                     or (request.address + request.byte_count - 1) // 4096 - request.address // 4096 + 1 >= BULK_MIN_PAGES)
                and not self._write_buffer_pending and not self._write_buffer_occupancy):
            from ._nand_acceleration import _generic_resident_writers
            if (_generic_resident_writers(self)
                    and (self._nand_numeric_state is None
                         or self._nand_numeric_state.scalar_resource_contract == self._resident_resource_contract())):
                self._validate_request(request)
                result = self._submit_generic_summary(request)
                if result is not None:
                    return result
        # GC interleaves a scalar read and PROGRAM with bulk page execution.
        # Hand only that bank to the unchanged scalar state machine instead
        # of publishing and repacking all banks for every copied page.
        # A staged read can fall back from the native batch executor to the
        # one-bank scalar bridge.  Keep that bridge resident while deferred
        # pages are still below both gates; publishing all banks here would
        # turn every such submission into a full numeric-state round trip.
        if (self._nand_numeric_state is not None
                and "_resident_scalar_eligible" not in self.__dict__
                and type(self)._resident_scalar_eligible is _RESIDENT_SCALAR_METHODS["_resident_scalar_eligible"]
                and self._resident_scalar_eligible(request)):
            self._validate_request(request)
            result = self._submit_resident_generic(request)
            if result is not None:
                return result
        if not (self._nand_numeric_state is not None
                and "_resident_generic_bulk_eligible" not in self.__dict__
                and type(self)._resident_generic_bulk_eligible is _RESIDENT_SCALAR_METHODS["_resident_generic_bulk_eligible"]
                and self._resident_generic_bulk_eligible(request)):
            self._nand_publish_numeric()
        self._validate_request(request)
        return self._submit_scalar_request(request)

    def _submit_scalar_request(self, request):
        """Account a validated request through the ordinary scalar machine."""
        self._observation_events = []
        effective_arrival, admitted, host_channel = self._admit_host_request(request)
        before_metrics = self.timeline._core_metrics_snapshot(
            max_intervals=(
                self.config.max_expanded_segments
                if self.capture_details
                else 0
            )
        )
        effective_arrival, write_buffer_bytes, write_buffer_wait_ns, drained = self._admit_write_buffer(
            request, effective_arrival
        )
        if effective_arrival != admitted.effective:
            admitted = admitted.population.preview(effective_arrival)
        admission_depths = {
            "host_outstanding_at_admission_peak": len(admitted) + 1,
            "host_read_outstanding_at_admission_peak":
                admitted.reads
                + int(request.operation is Operation.READ),
            "host_write_outstanding_at_admission_peak":
                admitted.writes
                + int(request.operation is Operation.WRITE),
        }
        result = self._execute_accepted(
            request if effective_arrival == request.arrival_ns
            else replace(request, arrival_ns=effective_arrival)
        )
        if drained:
            counters = dict(result.counters)
            counters["rmw_read_operations"] = (
                counters.get("pages_read", 0) if request.operation is Operation.WRITE else 0
            ) + drained.get("pages_read", 0)
            for key in ("physical_write_bytes", "physical_read_bytes", "pages_read", "internal_transfer_bytes"):
                counters[key] = counters.get(key, 0) + drained.get(key, 0)
            counters["pages_programmed"] = counters.get("pages_programmed", 0) + drained.get("physical_programs", 0)
            counters["auto_drain_physical_programs"] = drained.get("physical_programs", 0)
            result = replace(result, counters=counters, transfer_bytes=result.transfer_bytes + drained.get("internal_transfer_bytes", 0))
        result = replace(result, arrival_ns=request.arrival_ns, counters={
            **result.counters,
            "pages_touched": result.counters.get("page_count", 0),
            "read_operations": result.counters.get("pages_read", 0),
            "program_operations": result.counters.get("pages_programmed", 0),
            "rmw_read_operations": result.counters.get("rmw_read_operations",
                result.counters.get("pages_read", 0) if request.operation is Operation.WRITE else 0),
            "page_access_payload_bytes": request.byte_count if request.operation is not Operation.ERASE else 0,
            "page_access_capacity_bytes": (result.counters.get("page_count", 0) * self.config.page_bytes
                                            if request.operation is not Operation.ERASE else 0),
            "host_admission_samples": 1,
            **admission_depths,
            "host_admission_depth_scope": self._admission_depth_scope(),
            "queue_wait_ns": effective_arrival - request.arrival_ns,
            "write_buffer_wait_ns": write_buffer_wait_ns,
            "write_buffer_bytes": write_buffer_bytes,
            "write_buffer_occupancy_bytes": self._held_write_buffer_bytes() + write_buffer_bytes,
            "write_buffer_peak_bytes": self._held_write_buffer_bytes() + write_buffer_bytes,
            "write_buffer_draining_bytes": self.inflight_write_buffer_bytes,
            "buffered_write_mode": self._buffered_write_mode,
            **self.timeline._core_metrics_delta(before_metrics),
            **self._observations(),
        })
        self._reserve_write_buffer(write_buffer_bytes)
        self._record_host_admission(host_channel, effective_arrival, admitted,
                                    result.completion_ns, request.operation)
        return result

    def _submit_generic_summary(self, request):
        """Keep host admission exact while billing native calendar deltas."""
        from ._nand_acceleration import execute_buffered_full_pages
        from .memory_transfer import _OWNED_TIMELINE_TOKEN_TYPE, _OWNED_TIMELINE_REMEMBER

        # An authored undo callback can observe the normal accounting token
        # and its per-request lifetime. Keep that protocol for unknown writers.
        if any(type(token) is not _OWNED_TIMELINE_TOKEN_TYPE or token.accounting
               or getattr(token.remember, "__func__", token.remember) is not _OWNED_TIMELINE_REMEMBER
               for token in self.timeline._owned_transactions):
            return None

        self._observation_events = []
        effective, admitted, host_channel = self._admit_host_request(request)
        self.timeline._reset_metric_recorder(0)
        metrics = {}
        count = (request.address + request.byte_count - 1) // 4096 - request.address // 4096 + 1
        effective_request = (request if effective == request.arrival_ns else
                             replace(request, arrival_ns=effective))
        result = execute_buffered_full_pages(self, effective_request, count, _metrics=metrics)
        if result is None:
            return None
        # No write buffer, staged pages, media observations or authored
        # callbacks qualify for this path. These are the same fields and
        # insertion order as _submit_live's general request accounting.
        counters = result.counters
        result = replace(result, arrival_ns=request.arrival_ns, counters={
            **counters,
            "pages_touched": counters["page_count"],
            "read_operations": counters["pages_read"],
            "program_operations": counters["pages_programmed"],
            "rmw_read_operations": counters["pages_read"] if request.operation is Operation.WRITE else 0,
            "page_access_payload_bytes": request.byte_count,
            "page_access_capacity_bytes": counters["page_count"] * self.config.page_bytes,
            "host_admission_samples": 1,
            "host_outstanding_at_admission_peak": len(admitted) + 1,
            "host_read_outstanding_at_admission_peak":
                admitted.reads + int(request.operation is Operation.READ),
            "host_write_outstanding_at_admission_peak":
                admitted.writes + int(request.operation is Operation.WRITE),
            "host_admission_depth_scope": self._admission_depth_scope(),
            "queue_wait_ns": effective - request.arrival_ns,
            "write_buffer_wait_ns": 0.0,
            "write_buffer_bytes": 0,
            "write_buffer_occupancy_bytes": 0,
            "write_buffer_peak_bytes": 0,
            "write_buffer_draining_bytes": 0,
            "buffered_write_mode": self._buffered_write_mode,
            **metrics,
        })
        self._record_host_admission(host_channel, effective, admitted, result.completion_ns, request.operation)
        return result

    def _resident_scalar_eligible(self, request):
        if (type(self) is not NandCore or type(request) is not AccessRequest
                or type(self.timeline) is not ResourceTimeline or self.capture_details
                or self._observe_media or self._nand_state_exposed
                or self._cache_use >= 2**63 - 1
                or self.config.page_bytes != 4096):
            return False
        # An authored replacement can access arbitrary controller banks. Such
        # callbacks continue to receive the fully published controller state.
        from ._nand_acceleration import _generic_resident_writers, has_native_timeline_hooks
        if not has_native_timeline_hooks() or not _generic_resident_writers(self):
            return False
        if _generic_bulk_span(request, self.config):
            return False
        return (all(name not in self.__dict__ and getattr(NandCore, name) is method
                    for name, method in _RESIDENT_SCALAR_METHODS.items())
                and getattr(self._nand_numeric_state, "scalar_resource_contract", None)
                == self._resident_resource_contract())

    def _resident_generic_bulk_eligible(self, request):
        from ._nand_acceleration import _generic_resident_writers, has_native_timeline_hooks
        config = self.config
        return (type(self) is NandCore and type(request) is AccessRequest
                and self._nand_numeric_enabled and not self._nand_state_exposed
                and config.page_bytes == 4096
                and _generic_bulk_span(request, config)
                and has_native_timeline_hooks()
                and _generic_resident_writers(self)
                and self._nand_numeric_state.scalar_resource_contract == self._resident_resource_contract())

    def _resident_resource_contract(self):
        """Immutable layout values, including mutable routing metadata.

        Capture the same authored geometry/prefixes/aliases used by bulk
        packing. A scalar handoff cannot use an old array name after a caller
        changes a resource prefix or route between physical submissions.
        """
        config = self.config
        return (config.channels, config.targets_per_channel, config.dies_per_target,
                config.luns_per_die, config.planes_per_lun, config.planes_independent,
                config.parallel_units, config.host_channels, config.host_full_duplex,
                config.pages_per_block,
                tuple(str(config.metadata.get(name, default)) for name, default in (
                    ("array_resource_prefix", "nand:array"),
                    ("parallel_resource_prefix", "nand:parallel"),
                    ("channel_resource_prefix", "nand:channel"),
                    ("host_resource_id", "nand:host"),
                    ("scratchpad_resource_prefix", "nand:scratchpad"))),
                tuple(self.timeline.resource_aliases.items()))

    def _resident_handoff(self, banks, units):
        """Publish only these resident banks and units to their Python slots.

        The numeric arrays stay authoritative for every other bank. Journals
        capture the resident preimage first, just as for a complete
        publication. The returned callable writes the scalar state back.
        """
        packed = self._nand_numeric_state
        arrays, buffers, tags, ready, readers, uses, present = packed.arrays
        bank_keys, prefixes, array_names, pages_per_block = packed.layout
        array_values = self.__dict__["_array_ready"]
        buffer_values = self.__dict__["_buffer_ready"]
        cache = self.__dict__["_read_cache"]
        for token in self._nand_transactions:
            token.capture_numeric(packed)
        for unit in units:
            if packed.dirty_units[unit]:
                name = array_names[unit]
                self._touch_frontiers(name)
                array_values[name], buffer_values[name] = float(arrays[unit]), float(buffers[unit])
        for bank in banks:
            if not packed.dirty_banks[bank]:
                continue
            key = bank_keys[bank]
            if self._nand_transactions:
                self._touch_cache_bank(key)
            slots = cache.get(key)
            if slots is None:
                slots = [_ReadCacheSlot(), _ReadCacheSlot()]
                cache[key] = slots
            for index, slot in enumerate(slots):
                page = int(tags[bank, index])
                slot.page_key = None if page < 0 else (*prefixes[bank], *divmod(page, pages_per_block))
                slot.ready_ns, slot.reader_completion_ns, slot.last_use = (
                    float(ready[bank, index]), float(readers[bank, index]), int(uses[bank, index]))

        def write_back():
            # Preserve even a failing page machine's exact partial state;
            # an outer transaction still owns restoration of its preimage.
            for unit in units:
                name = array_names[unit]
                if name in array_values:
                    arrays[unit] = array_values[name]
                    buffers[unit] = buffer_values.get(name, 0.)
                    packed.dirty_units[unit] = True
            for bank in banks:
                slots = cache.get(bank_keys[bank])
                if slots is None:
                    continue
                present[bank] = True
                packed.dirty_banks[bank] = True
                for index, slot in enumerate(slots):
                    page = slot.page_key
                    tags[bank, index] = -1 if page is None else page[-2] * pages_per_block + page[-1]
                    ready[bank, index], readers[bank, index], uses[bank, index] = (
                        slot.ready_ns, slot.reader_completion_ns, slot.last_use)
        return write_back

    def _submit_resident_generic(self, request):
        """Hand only the addressed banks to the original generic request path.

        Short multi-page and unaligned spans have the same admission and
        exact page machine as before. Other banks remain in private arrays;
        the temporary handoff does not fabricate future controller state.
        """
        packed, config = self._nand_numeric_state, self.config
        base_units = (config.channels * config.targets_per_channel
                      * config.dies_per_target * config.luns_per_die)
        bank_count = base_units * config.planes_per_lun
        first = request.address // 4096
        last = (request.address + request.byte_count - 1) // 4096
        if request.operation is Operation.READ and self._sequential_write_mode():
            # The scalar machine reads each page where the page table put it.
            banks = tuple(dict.fromkeys((self._read_pages(first, last - first + 1) % bank_count).tolist()))
        elif request.operation is Operation.WRITE and self._sequential_write_mode():
            # ... and programs it at its channel's write pointer.
            banks = tuple(dict.fromkeys(page % bank_count for page in
                                        self._phase_batch_mappings(range(first, last + 1))[0]))
        else:
            banks = tuple((first + offset) % bank_count
                          for offset in range(min(last - first + 1, bank_count)))
        units = tuple(dict.fromkeys(bank if config.planes_independent else bank % base_units
                                   for bank in banks))
        write_back = self._resident_handoff(banks, units)
        enabled = self._nand_numeric_enabled
        self._nand_numeric_state = None
        self._nand_numeric_enabled = False
        try:
            return self._submit_scalar_request(request)
        finally:
            self._nand_numeric_state = packed
            self._nand_numeric_enabled = enabled
            write_back()

    # Page staging. A partial page is held in the intermediate store and
    # reaches the media only once all of its bytes are valid; reads of a
    # staged page are served from the store until that program completes.

    def _tail_channel(self, page):
        config = self.config
        base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
        return (page % base_units) // (base_units // config.channels) % config.host_channels

    def _tail_remember(self, page):
        for token in self._nand_transactions:
            token.remember_tail(self, page)

    def _tail_scope_remember(self):
        for token in self._nand_transactions:
            token.remember_tail_scopes(self)

    def _tail_spill(self, page):
        for token in self._nand_transactions:
            token.remember_tail_spilled(self)
        self._tail_spilled.add(page)
        self._tail_spilled_pages += 1

    def _tail_admit(self, pages, start, end, arrival):
        """The pages of a write the store can hold; the others spill to the array.

        A page enters the store only if its valid bytes fit beside the shared
        data. A sampled run stands for a whole step whose other layers' tails
        hold the rest of the store: there a new page is kept only if its
        index falls in the step's stored share (``_tail_fraction``), so every
        layer keeps the same share. A staged page that would overflow is
        programmed from the store now and its new rows go to the array. A
        full store therefore costs array programs and read-modify-writes
        instead of refusing the batch.
        """
        config = self.config
        page_bytes = config.page_bytes
        self._tail_release(arrival)
        room = config.page_staging_bytes_per_host_channel - self._tail_shared_bytes()
        fraction = self._tail_fraction
        valid, entries, kept = list(self._tail_valid), self._tail_pages, set()
        for page in pages:
            base = page * page_bytes
            lo, hi = max(start, base) - base, min(end, base + page_bytes) - base
            channel = self._tail_channel(page)
            entry = entries.get(page)
            if entry is None or entry[3] >= 0:
                if fraction < 1. and _page_share(page) >= fraction:
                    self._tail_spill(page)
                    continue
                # A new copy holds the page's whole valid prefix; it replaces
                # a copy whose program is running.
                grow = hi - (entry[1] if entry is not None else 0)
            else:
                grow = sum(right - left for left, right in _merge_ranges(entry[0] + ((lo, hi),))) - entry[1]
            if valid[channel] + grow > room:
                if entry is not None and entry[3] < 0:
                    self._tail_evict(page, entry, arrival)
                self._tail_spill(page)
                continue
            valid[channel] += grow
            kept.add(page)
        return kept

    def _tail_evict(self, page, entry, now):
        """Program a staged partial page from the store to make room."""
        mapping = map_nand_address(self.config, page * self.config.page_bytes)
        end, _internal, _host, _erase = self._tail_program(
            page, mapping, entry[2], [], start_override=max(float(now), float(entry[2])))
        self._tail_remember(page)
        self._tail_pages[page] = (entry[0], entry[1], entry[2], end, entry[4])
        heappush(self._tail_flushing, (end, page))
        self._tail_evicted_pages += 1

    def set_staging_scope(self, scope):
        """Select the decode layer/rank scope whose full pages remain staged."""
        if scope is not None and (not isinstance(scope, str) or not scope):
            raise ValueError("staging scope must be a non-empty string or None")
        self._tail_defer_scope = scope

    def flush_staged_pages(self, scope, now, pages=()):
        """Program full pages for one attention scope after its read wave."""
        if self.config.page_staging != "logic_die":
            return {"programs": 0, "physical_write_bytes": 0,
                    "internal_transfer_bytes": 0, "host_transfer_bytes": 0,
                    "erase_operations": 0}
        scope = str(scope)
        self._tail_scope_remember()
        selected = self._tail_full_scopes.get(scope, set())
        if pages:
            wanted = {int(page) for page in pages}
            selected = selected.intersection(wanted)
        if not selected:
            return {"programs": 0, "physical_write_bytes": 0,
                    "internal_transfer_bytes": 0, "host_transfer_bytes": 0,
                    "erase_operations": 0}
        before = self.timeline._core_metrics_snapshot(
            max_intervals=self.config.max_expanded_segments if self.capture_details else 0)
        programs = internal_bytes = host_bytes = erases = 0
        active_pages, active_entries, ready_times, start_times = [], [], [], []
        for page in tuple(sorted(selected)):
            entry = self._tail_pages.get(page)
            if entry is None or entry[3] >= 0:
                selected.discard(page)
                continue
            active_pages.append(page)
            active_entries.append(entry)
            start = max(float(now), float(entry[2]))
            ready_times.append(self._tail_program_channel_start(entry[2], start))
            start_times.append(start)
            self._tail_remember(page)
        batch = None
        if active_pages:
            # The phase batch maps each logical page from the current host
            # cursor and advances the cursor exactly once.  Pre-validating
            # here would advance that cursor before mapping and skip every
            # page-zero boundary.  The scalar fallback validates inside
            # _tail_program, after it has selected the physical mapping.
            if not self._sequential_write_mode():
                self._validate_write_order(active_pages)
            from ._nand_acceleration import execute_staged_flush_batch
            batch = execute_staged_flush_batch(self, active_pages, ready_times, start_times)
        if batch is not None:
            ends = batch["ends"]
            self._tail_flushing.extend((float(end), page)
                                        for page, end in zip(active_pages, ends.tolist()))
            heapify(self._tail_flushing)
            for page, entry, end in zip(active_pages, active_entries, ends.tolist()):
                self._tail_pages[page] = (entry[0], entry[1], entry[2], float(end), entry[4])
                selected.discard(page)
            programs = len(active_pages)
            phase_mappings = batch.get("_phase_mappings", ())
            if phase_mappings:
                self._commit_phase_batch(active_pages, phase_mappings,
                                         batch.get("_phase_next_pointers", self._write_pointer))
            else:
                self._log_programs(active_pages)
            erases = int(batch.get("erase_operations", 0))
            internal_bytes = int(batch["internal_transfer_bytes"])
            host_bytes = int(batch["host_transfer_bytes"])
            self._tail_flush_started_pages += programs
        else:
            for page, entry, start in zip(active_pages, active_entries, start_times):
                mapping = map_nand_address(self.config, page * self.config.page_bytes)
                end, moved_internal, moved_host, moved_erases = self._tail_program(
                    page, mapping, entry[2], [], start_override=start)
                self._tail_pages[page] = (entry[0], entry[1], entry[2], end, entry[4])
                heappush(self._tail_flushing, (end, page))
                selected.discard(page)
                programs += 1
                internal_bytes += moved_internal
                host_bytes += moved_host
                erases += moved_erases
        if not selected:
            self._tail_full_scopes.pop(scope, None)
        else:
            self._tail_full_scopes[scope] = selected
        metrics = self.timeline._core_metrics_delta(before)
        return {"programs": programs, "physical_write_bytes": programs * self.config.page_bytes,
                "erase_operations": erases,
                "internal_transfer_bytes": internal_bytes, "host_transfer_bytes": host_bytes,
                **metrics}

    def _tail_release(self, now):
        """Retire staged copies whose page program completed by ``now``."""
        heap = self._tail_flushing
        if not heap or heap[0][0] > now:
            return
        pages, valid = self._tail_pages, list(self._tail_valid)
        while heap and heap[0][0] <= now:
            end, page = heappop(heap)
            entry = pages.get(page)
            if entry is not None and entry[3] == end:
                self._tail_remember(page)
                del pages[page]
                valid[entry[4]] -= entry[1]
        self._tail_valid = tuple(valid)
        self._tail_index = None

    def _tail_staged(self, first, last):
        """Staged page indexes in [first, last], ascending."""
        pages = self._tail_pages
        if not pages:
            return ()
        if last - first < 32:
            return tuple(page for page in range(first, last + 1) if page in pages)
        index = self._tail_index
        if index is None:
            index = self._tail_index = tuple(sorted(pages))
        return index[bisect_left(index, first):bisect_right(index, last)]

    def _tail_readable(self, first, last, arrival):
        """Staged pages a read arriving at ``arrival`` takes from the store.

        A spilled page is read from the array: its later rows never entered
        the store.
        """
        pages, spilled = self._tail_pages, self._tail_spilled
        return tuple(page for page in self._tail_staged(first, last)
                     if (pages[page][3] < 0 or pages[page][3] > arrival) and page not in spilled)

    def _tail_check_read(self, page, start, end):
        """A staged page serves only bytes that some write put in the store."""
        base = page * self.config.page_bytes
        lo, hi = max(start, base) - base, min(end, base + self.config.page_bytes) - base
        if not any(a <= lo and hi <= b for a, b in self._tail_pages[page][0]):
            raise ValueError("read of bytes [{}, {}) of staged NAND page {} that no write "
                             "put in the store".format(lo, hi, page))

    def _tail_shared_bytes(self):
        """Per-channel share of other data in the same store (interleaved)."""
        return self._tail_shared

    def page_staging_state(self):
        """Valid staged bytes and their peaks per host channel."""
        return {"staged_pages": len(self._tail_pages), "valid_bytes": list(self._tail_valid),
                "peak_bytes": list(self._tail_peak), "shared_bytes": self._tail_shared_bytes(),
                "capacity_bytes": self.config.page_staging_bytes_per_host_channel,
                "deferred_pages": sum(len(pages) for pages in self._tail_full_scopes.values()),
                "deferred_scopes": len(self._tail_full_scopes),
                "full_page_fills": int(self._tail_full_page_fills),
                "started_programs": int(self._tail_flush_started_pages),
                "discarded_pages": int(self._tail_flush_discarded_pages),
                "pages_programmed": int(self._tail_flush_started_pages),
                "spilled_pages": int(self._tail_spilled_pages),
                "evicted_pages": int(self._tail_evicted_pages),
                "stored_fraction": float(self._tail_fraction)}

    def discard_tail_pages(self, address, byte_count):
        """Drop staged copies of a released allocation; their bytes are dead.

        A program already scheduled for a filled page is not cancelled.
        """
        page_bytes = self.config.page_bytes
        first, last = address // page_bytes, (address + byte_count - 1) // page_bytes
        spilled = self._tail_spilled
        if spilled:
            for token in self._nand_transactions:
                token.remember_tail_spilled(self)
            if last - first < len(spilled):
                spilled.difference_update(range(first, last + 1))
            else:
                spilled.difference_update([page for page in spilled if first <= page <= last])
        staged = self._tail_staged(first, last)
        if not staged:
            return 0
        pages, valid = self._tail_pages, list(self._tail_valid)
        discarded = 0
        for page in staged:
            self._tail_remember(page)
            entry = pages.pop(page)
            if entry[3] < 0:
                self._tail_scope_remember()
                for scope, scoped_pages in tuple(self._tail_full_scopes.items()):
                    scoped_pages.discard(page)
                    if not scoped_pages:
                        self._tail_full_scopes.pop(scope, None)
                self._tail_flush_discarded_pages += 1
                discarded += 1
            valid[entry[4]] -= entry[1]
        self._tail_valid = tuple(valid)
        self._tail_index = None
        return len(staged)

    def _submit_staged(self, request):
        """Route a request that touches staged pages; None keeps the normal path."""
        page_bytes = self.config.page_bytes
        start, end = request.address, request.address + request.byte_count
        first, last = start // page_bytes, (end - 1) // page_bytes
        if request.operation is Operation.READ:
            readable = self._tail_readable(first, last, request.arrival_ns)
            if not readable and not self._tail_flushing:
                return None
            self._validate_request(request)
            if (self._nand_numeric_enabled and not self._sequential_write_mode()
                    and self.config.queue_scope == "device"):
                from ._nand_acceleration import execute_read_batch
                results = execute_read_batch(self, (request,))
                if results is not None:
                    return results[0]
            return None  # The scalar page machine serves staged pages.
        if request.operation is not Operation.WRITE:
            return None
        pages = self._tail_pages
        staged = {page for page in self._tail_staged(first, last) if pages[page][3] < 0}
        if start % page_bytes:
            staged.add(first)
        if end % page_bytes:
            staged.add(last)
        if self._tail_spilled:
            staged -= self._tail_spilled
        if staged and self.config.page_staging_bytes_per_host_channel:
            staged = self._tail_admit(sorted(staged), start, end, request.arrival_ns)
        if not staged:
            return None
        self._validate_request(request)
        runs, cursor = [], start
        for page in sorted(staged):
            lo, hi = max(start, page * page_bytes), min(end, (page + 1) * page_bytes)
            if cursor < lo:
                runs.append((False, cursor, lo))
            if runs and runs[-1][0] and runs[-1][2] == lo:
                runs[-1] = (True, runs[-1][1], hi)
            else:
                runs.append((True, lo, hi))
            cursor = hi
        if cursor < end:
            runs.append((False, cursor, end))
        if len(runs) == 1:
            return self._submit_tail_write(request)
        results = []
        for index, (tail, lo, hi) in enumerate(runs):
            part = replace(request, request_id=f"{request.request_id}:{index}", address=lo, byte_count=hi - lo)
            results.append(self._submit_tail_write(part) if tail else self._submit_live(part))
        return self._merge_split(request, results)

    def _submit_tail_write(self, request):
        """Merge a write into staged pages; a page that fills is programmed.

        The write is acknowledged once its bytes reach the store. Logic-die
        staging moves them over the page's host channel; external staging is
        billed by the memory that holds the tail. A full page then programs
        without a read-modify-write, after the acknowledgement.
        """
        config, timeline = self.config, self.timeline
        self._observation_events = []
        effective, admitted, host_channel = self._admit_host_request(request)
        before = timeline._core_metrics_snapshot(
            max_intervals=config.max_expanded_segments if self.capture_details else 0)
        arrival = effective + config.front_ns
        self._tail_release(arrival)
        page_bytes, burst = config.page_bytes, config.page_staging_burst_bytes
        latency, logic = config.page_staging_latency_ns, config.page_staging == "logic_die"
        limit = config.page_staging_bytes_per_host_channel
        shared = self._tail_shared_bytes() if limit else 0
        pages, valid, peak = self._tail_pages, list(self._tail_valid), list(self._tail_peak)
        end = request.address + request.byte_count
        first, last = request.address // page_bytes, (end - 1) // page_bytes
        completion = arrival
        host_bytes = flush_host_bytes = internal_bytes = programs = erases = 0
        stages, mappings = [], []
        for page in range(first, last + 1):
            base = page * page_bytes
            lo, hi = max(request.address, base) - base, min(end, base + page_bytes) - base
            channel = self._tail_channel(page)
            entry = pages.get(page)
            if entry is None or entry[3] >= 0:
                # Staged pages are append-only (KV tails): the bytes before a
                # page's first staged write were written earlier and wait in
                # the store too, e.g. an unpaid seed history or rows before a
                # sampled window. A page is programmed only once all are valid.
                ranges, held, ready = ((0, hi),), 0, 0.
                if entry is not None:
                    # A refill supersedes the copy whose program is running.
                    valid[entry[4]] -= entry[1]
            else:
                ranges, held, ready = _merge_ranges(entry[0] + ((lo, hi),)), entry[1], entry[2]
            filled = sum(right - left for left, right in ranges)
            staged = valid[channel] + filled - held
            if limit and staged + shared > limit:
                # _tail_admit spills every page that would not fit.
                from .memory_allocator import AllocationError
                raise AllocationError(
                    "unable to allocate {} bytes in capacity {} of NAND page staging on host channel {} "
                    "({} staged + {} shared)".format(staged + shared, limit, channel, staged, shared))
            mapping = map_nand_address(config, base)
            if self.capture_details:
                mappings.append(mapping)
            data = arrival + latency
            if logic:
                moved = -(-hi // burst) * burst - lo // burst * burst
                host = timeline.transfer(self._staging_host_id(mapping, Operation.WRITE), moved, arrival,
                                         self._host_bandwidth(Operation.WRITE), direction="write")
                if self.capture_details:
                    stages.append(StageTiming("STAGING_WRITE", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                data = host.end_ns + latency
                host_bytes += moved
            ready = max(ready, data)
            completion = max(completion, data)
            flush = -1.
            if ranges == ((0, page_bytes),):
                self._tail_full_page_fills += 1
                if logic and self._tail_defer_scope is not None:
                    self._tail_scope_remember()
                    self._tail_full_scopes.setdefault(self._tail_defer_scope, set()).add(page)
                else:
                    flush, moved_internal, moved_host, moved_erases = self._tail_program(page, mapping, ready, stages)
                    programs += 1
                    erases += moved_erases
                    internal_bytes += moved_internal
                    flush_host_bytes += moved_host
            self._tail_remember(page)
            pages[page] = (ranges, filled, ready, flush, channel)
            valid[channel] = staged
            peak[channel] = max(peak[channel], staged + shared)
            if flush >= 0:
                heappush(self._tail_flushing, (flush, page))
        self._tail_valid, self._tail_peak, self._tail_index = tuple(valid), tuple(peak), None
        count = last - first + 1
        result = TransactionResult(
            request_id=request.request_id, operation=Operation.WRITE, arrival_ns=request.arrival_ns,
            completion_ns=completion, logical_bytes=request.byte_count, transfer_bytes=internal_bytes,
            mapping=tuple(mappings), stages=tuple(stages), counters={
                "host_transfer_bytes": host_bytes + flush_host_bytes,
                "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": 0, "physical_write_bytes": programs * page_bytes,
                "pages_read": 0, "pages_programmed": programs,
                "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": count,
                "staging_read_hits": 0, "staging_read_merges": 0,
                "read_cache_hits": 0, "read_cache_misses": 0, "erase_operations": erases,
                "page_count": count, "details_truncated": False,
                "page_staging_write_bytes": host_bytes, "page_staging_flush_programs": programs,
                "page_staging_peak_bytes": max(peak),
                "bandwidth_ceiling_gb_s": config.host_bandwidth_for(Operation.WRITE),
                "bandwidth_ceiling_scope": "directional host upper bound for staged writes; "
                                           "a filled page programs after the acknowledgement",
                "array_units": config.array_units,
                "write_completion": "page_staging_ack_pending_full_page_program",
                "host_ack_completion_ns": completion, "media_drain_completion_ns": None,
                "pages_touched": count, "read_operations": 0, "program_operations": programs,
                "rmw_read_operations": 0, "page_access_payload_bytes": request.byte_count,
                "page_access_capacity_bytes": count * page_bytes, "host_admission_samples": 1,
                "host_outstanding_at_admission_peak": len(admitted) + 1,
                "host_read_outstanding_at_admission_peak": admitted.reads,
                "host_write_outstanding_at_admission_peak": admitted.writes + 1,
                "host_admission_depth_scope": self._admission_depth_scope(),
                "queue_wait_ns": effective - request.arrival_ns,
                "write_buffer_wait_ns": 0.0, "write_buffer_bytes": 0,
                "write_buffer_occupancy_bytes": 0, "write_buffer_peak_bytes": 0,
                "write_buffer_draining_bytes": 0, "buffered_write_mode": self._buffered_write_mode,
                **timeline._core_metrics_delta(before), **self._observations(),
            })
        self._record_host_admission(host_channel, effective, admitted, completion, Operation.WRITE)
        return result

    def _tail_program(self, page, mapping, ready, stages, *, start_override=None):
        """Program one filled staged page from ``ready`` without reading it from the array.

        The host writes the page, 4 KiB as always. A logic-die scratchpad is
        the host's too, with no internal path to NAND (OCP 5.4.2, 11.3), so
        the host reads the page back first; an external store's reads are
        billed by the memory that holds it.
        """
        config, timeline = self.config, self.timeline
        mapping = self._program_mapping(page, mapping)
        self._validate_write_order((int(page),))
        write_back = None
        if self._nand_numeric_state is not None:
            base_units = (config.channels * config.targets_per_channel
                          * config.dies_per_target * config.luns_per_die)
            bank = page % (base_units * config.planes_per_lun)
            write_back = self._resident_handoff(
                (bank,), (bank if config.planes_independent else bank % base_units,))
        try:
            self._invalidate_cache(mapping)
            erase_needed = self._auto_erase_needed(mapping)
            array_id = self._array_id(mapping)
            earliest, host_bytes = self._tail_program_channel_start(ready, start_override), 0
            if config.page_staging == "logic_die":
                back = timeline.transfer(self._staging_host_id(mapping, Operation.READ), config.page_bytes, earliest,
                                         self._host_bandwidth(Operation.READ), direction="read")
                if self.capture_details:
                    stages.append(StageTiming("STAGING_FLUSH_READ", back.start_ns, back.end_ns,
                                              back.resource_id, back.bytes))
                earliest, host_bytes = back.end_ns, config.page_bytes
            host = timeline.transfer(self._staging_host_id(mapping, Operation.WRITE), config.page_bytes, earliest,
                                     self._host_bandwidth(Operation.WRITE), direction="write")
            if self.capture_details:
                stages.append(StageTiming("STAGING_FLUSH_HOST", host.start_ns, host.end_ns,
                                          host.resource_id, host.bytes))
            earliest, host_bytes = host.end_ns, host_bytes + config.page_bytes
            array_front = self.__dict__["_array_ready"].get(array_id, 0.0)
            buffer_front = self.__dict__["_buffer_ready"].get(array_id, 0.0)
            internal_earliest = max(earliest, buffer_front)
            if start_override is not None:
                # A deferred logic-die flush is a background task that is
                # released by attention.  Its die-channel transfer must not
                # be reserved before that release just because the page was
                # filled earlier.
                internal_earliest = max(internal_earliest, float(start_override))
            internal = timeline.transfer(
                self._channel_id(mapping), config.transfer_page_bytes,
                internal_earliest,
                config.internal_bandwidth_gb_s, direction="write")
            if self.capture_details:
                stages.append(StageTiming("STAGING_FLUSH_INTERNAL", internal.start_ns,
                                          internal.end_ns, internal.resource_id, internal.bytes))
            # A deferred page may have been ready while the plane was idle.
            # Keep the internal transfer on its normal channel calendar, then
            # reserve the array program at the calculated idle slot whenever
            # that slot is still free.
            program_start = max(internal.end_ns,
                                self.__dict__["_array_ready"].get(array_id, 0.0))
            if start_override is not None:
                program_start = max(program_start, float(start_override))
            erase_end = self._reserve_auto_erase(mapping, program_start,
                                                 eligible_ns=internal.end_ns,
                                                 stages=stages)
            program = self._media_stage("PAGE_PROGRAM", mapping,
                                        max(program_start, erase_end),
                                        config.page_program_ns, eligible_ns=internal.end_ns)
            self._log_programs((page,))
            self._remember_program(page, mapping)
            if self.capture_details:
                stages.append(program)
            self._touch_frontiers(array_id)
            self.__dict__["_array_ready"][array_id] = program.end_ns
            self.__dict__["_buffer_ready"][array_id] = program.end_ns
            self.timeline._remember_resource(array_id)
            self.timeline.ready_ns[array_id] = program.end_ns
            self._tail_flush_started_pages += 1
        finally:
            if write_back is not None:
                write_back()
        return program.end_ns, internal.bytes, host_bytes, int(erase_needed)

    @staticmethod
    def _tail_program_channel_start(ready, start_override=None):
        """Return the shared host-transfer origin for scalar and batch flushes.

        The host releases a deferred flush at ``start_override``, once the
        page's bytes are all in the store.
        """
        return float(ready) if start_override is None else max(float(ready), float(start_override))

    def _merge_split(self, request, results):
        """One result for a write split into direct and staged page runs."""
        additive = ("host_transfer_bytes", "internal_transfer_bytes", "physical_read_bytes",
                    "physical_write_bytes", "pages_read", "pages_programmed", "staging_write_pages",
                    "page_staging_write_bytes", "page_staging_flush_programs", "read_cache_hits",
                    "read_cache_misses", "erase_operations", "page_count", "pages_touched",
                    "read_operations", "program_operations", "rmw_read_operations",
                    "page_access_capacity_bytes", "host_admission_samples", "queue_wait_ns")
        peaks = ("host_outstanding_at_admission_peak", "host_read_outstanding_at_admission_peak",
                 "host_write_outstanding_at_admission_peak", "page_staging_peak_bytes")
        counters = dict(results[0].counters)
        for result in results[1:]:
            other = result.counters
            for key in additive:
                if key in other:
                    counters[key] = counters.get(key, 0) + other[key]
            for key in peaks:
                if key in other:
                    counters[key] = max(counters.get(key, 0), other[key])
            for key in ("resource_busy_ns", "resource_bytes"):
                if key in other:
                    merged = dict(counters.get(key, {}))
                    for rid, value in other[key].items():
                        merged[rid] = merged.get(rid, 0) + value
                    counters[key] = merged
            if "resource_last_intervals" in other:
                merged = dict(counters.get("resource_last_intervals", {}))
                for rid, interval in other["resource_last_intervals"].items():
                    previous = merged.get(rid)
                    if previous is None or (interval[1], interval[0]) > (previous[1], previous[0]):
                        merged[rid] = interval
                counters["resource_last_intervals"] = merged
            for key in ("resource_intervals", "resource_interval_payloads"):
                if key in other:
                    merged = dict(counters.get(key, {}))
                    for rid, values in other[key].items():
                        merged[rid] = tuple(merged.get(rid, ())) + tuple(values)
                    counters[key] = merged
            counters["intervals_truncated"] = bool(counters.get("intervals_truncated")) or bool(
                other.get("intervals_truncated"))
            counters["details_truncated"] = bool(counters.get("details_truncated")) or bool(
                other.get("details_truncated"))
            drains = [value for value in (counters.get("media_drain_completion_ns"),
                                          other.get("media_drain_completion_ns")) if value is not None]
            counters["media_drain_completion_ns"] = max(drains) if drains else None
        acknowledgements = [result.counters["host_ack_completion_ns"] for result in results
                            if result.counters.get("host_ack_completion_ns") is not None]
        counters["host_ack_completion_ns"] = max(acknowledgements) if acknowledgements else None
        counters["write_completion"] = "page_staging_ack_pending_full_page_program"
        counters["page_access_payload_bytes"] = request.byte_count
        return TransactionResult(
            request_id=request.request_id, operation=request.operation, arrival_ns=request.arrival_ns,
            completion_ns=max(result.completion_ns for result in results), logical_bytes=request.byte_count,
            transfer_bytes=sum(result.transfer_bytes for result in results),
            mapping=tuple(item for result in results for item in result.mapping),
            stages=tuple(item for result in results for item in result.stages), counters=counters)

    def _execute_accepted(self, request: AccessRequest) -> TransactionResult:
        first_page = request.address // self.config.page_bytes
        last_page = (request.address + request.byte_count - 1) // self.config.page_bytes
        page_count = last_page - first_page + 1
        if (not self.capture_details and not self._observe_media
                and self.config.read_cache_pages_per_bank == 2
                and page_count >= _bulk_min_pages(self, request.operation)):
            from ._nand_acceleration import execute_buffered_full_pages
            result = execute_buffered_full_pages(self, request, page_count)
            if result is not None:
                return result
            # A declined bulk can still have a live private controller from
            # an earlier request. The scalar machine must see that state.
            self._nand_publish_numeric()
        if (
            not self.capture_details
            and not self._observe_media
            and self.config.host_channels == 1
            and not self.config.host_full_duplex
            and not self.config.read_cache_pages_per_bank
            and request.operation is not Operation.ERASE
            and page_count >= 128
            and not self._staged_pages
            and not self._tail_pages
            # Positive write buffers are admitted at request granularity so
            # every buffered write observes the same capacity gate.
            and not (
                request.operation is Operation.WRITE
                and self.config.write_buffer_bytes > 0
            )
        ):
            return self._execute_accelerated_accepted(request, page_count)
        return self._execute_detailed_accepted(request)

    def _execute_detailed_accepted(self, request: AccessRequest) -> TransactionResult:
        segments = split_nand_request(request, self.config)
        stages = []
        mappings = []
        segment_count = 0
        details_truncated = False
        def record(stage):
            if self.capture_details and not details_truncated:
                stages.append(stage)
        completion = request.arrival_ns + self.config.front_ns
        logical_bytes = 0
        host_bytes = internal_bytes = 0
        pages_read = pages_programmed = erases = 0
        media_queue_wait_ns = 0.0
        # Read-wait is an observation of work inherited by this request.  The
        # scalar page loop mutates the same array/buffer/lane calendars as it
        # goes, so retain each execution slot's front at its first visit, which
        # precedes this request's first reservation of that slot, and each
        # logical array's initial front before the first page can change it.
        # This keeps same-request page serialization out of the metric while
        # preserving a real frontier left by an earlier request.
        inherited_array_frontiers = {}
        inherited_slot_frontiers = {}
        merge_bytes = 0
        merged_pages = 0
        staging_write_pages = 0
        staging_read_hits = 0
        staging_read_merges = 0
        read_cache_hits = read_cache_misses = 0
        for segment in segments:
            segment_count += 1
            if segment_count > self.config.max_expanded_segments and not details_truncated:
                stages.clear(); mappings.clear(); details_truncated = True
            m = segment.mapping
            page_number = int(segment.address // self.config.page_bytes)
            if request.operation is Operation.READ:
                m = self._read_mapping(page_number, m)
            elif request.operation is Operation.WRITE:
                m = self._program_mapping(page_number, m)
            if self.capture_details and not details_truncated:
                mappings.append(m)
            logical_bytes += segment.logical_bytes
            array_id = self._array_id(m)
            array_resource = self._array_resource_id(m)
            array_ready = self.__dict__["_array_ready"].get(array_id, 0.0)
            buffer_ready = self.__dict__["_buffer_ready"].get(array_id, 0.0)
            if array_id not in inherited_array_frontiers:
                slot_frontier = inherited_slot_frontiers.get(array_resource)
                if slot_frontier is None:
                    lanes = self.timeline.lane_available.get(array_resource)
                    slot_frontier = min(lanes) if lanes else self.timeline.ready_ns.get(array_resource, 0.0)
                    inherited_slot_frontiers[array_resource] = slot_frontier
                inherited_array_frontiers[array_id] = max(array_ready, buffer_ready, slot_frontier)
            inherited_frontier = inherited_array_frontiers[array_id]
            dependency = max(request.arrival_ns + self.config.front_ns, array_ready, buffer_ready)
            if request.operation is Operation.ERASE:
                self._invalidate_cache(m, block=True)
                stage = self._media_stage("BLOCK_ERASE", m, dependency, self.config.block_erase_ns,
                                          eligible_ns=request.arrival_ns+self.config.front_ns)
                record(stage)
                self._touch_frontiers(array_id)
                self.__dict__["_array_ready"][array_id] = stage.end_ns
                self.timeline._remember_resource(array_id)
                self.timeline.ready_ns[array_id] = stage.end_ns
                completion = max(completion, stage.end_ns)
                erases += 1
                continue

            host_bytes += segment.host_transfer_bytes
            internal_bytes += segment.internal_transfer_bytes
            partial = segment.logical_bytes != self.config.page_bytes
            tail = (self._tail_pages.get(segment.address // self.config.page_bytes)
                    if request.operation is Operation.READ and self._tail_pages else None)
            tail_lo = max(request.address, segment.address) - segment.address
            tail_hi = min(request.address + request.byte_count,
                          segment.address + self.config.page_bytes) - segment.address
            tail_covered = (tail is not None and any(
                left <= tail_lo and tail_hi <= right for left, right in tail[0]))
            if (request.operation is Operation.READ and tail is not None
                    and (tail[3] < 0 or tail[3] > request.arrival_ns)
                    and not tail_covered
                    and not (tail_lo == 0 and tail_hi == self.config.page_bytes)):
                self._tail_check_read(
                    segment.address // self.config.page_bytes,
                    request.address,
                    request.address + request.byte_count,
                )
            if (tail_covered and (tail[3] < 0 or tail[3] > request.arrival_ns)):
                # The staged copy serves the read: no array read or die
                # transfer, only the store and, on a logic die, its channel.
                self._tail_check_read(segment.address // self.config.page_bytes,
                                      request.address, request.address + request.byte_count)
                host_bytes -= segment.host_transfer_bytes
                internal_bytes -= segment.internal_transfer_bytes
                data = max(request.arrival_ns + self.config.front_ns + self.config.page_staging_latency_ns,
                           tail[2])
                if self.config.page_staging == "logic_die":
                    burst = self.config.page_staging_burst_bytes
                    lo = max(request.address, segment.address) - segment.address
                    hi = min(request.address + request.byte_count,
                             segment.address + self.config.page_bytes) - segment.address
                    host = self.timeline.transfer(
                        self._staging_host_id(m, Operation.READ), -(-hi // burst) * burst - lo // burst * burst,
                        data, self._host_bandwidth(Operation.READ), direction="read")
                    record(StageTiming("STAGING_READ", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                    host_bytes += host.bytes
                    data = host.end_ns
                completion = max(completion, data)
                staging_read_hits += 1
                continue
            if request.operation is Operation.WRITE:
                self._invalidate_cache(m)
                # Validate before any host/internal reservation so an OCP
                # out-of-order page cannot leave a partially billed request.
                self._validate_write_order((segment.address // self.config.page_bytes,))
            if (
                request.operation is Operation.WRITE
                and self._buffered_write_mode
                and self.config.write_buffer_bytes > 0
            ):
                # Host acknowledgement completes when the payload reaches the
                # staging buffer.  A later flush performs the one physical
                # page program for each distinct dirty page.
                internal_bytes -= segment.internal_transfer_bytes
                host = self.timeline.transfer(
                    self._host_id(m, Operation.WRITE), segment.host_transfer_bytes,
                    request.arrival_ns + self.config.front_ns,
                    self._host_bandwidth(Operation.WRITE),
                    direction="write",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                merged, merged_page = self._stage_page(segment, request, host.end_ns)
                merge_bytes += merged
                merged_pages += merged_page
                staging_write_pages += 1
                completion = max(completion, host.end_ns)
                continue
            host_write_origin = request.arrival_ns + self.config.front_ns
            host_write_bytes = segment.host_transfer_bytes
            if request.operation is Operation.WRITE and partial and self.config.partial_page_policy == "host_merge":
                # OCP writes are whole pages (4.1): the host reads the old
                # page, merges the new bytes and writes 4 KiB out of place on
                # the same bank. This is intentionally not a persistent page
                # cache.
                read = self._media_stage("MERGE_PAGE_READ", m, dependency, self.config.page_read_ns,
                                         eligible_ns=request.arrival_ns+self.config.front_ns)
                media_queue_wait_ns = max(
                    media_queue_wait_ns,
                    max(0.0, inherited_frontier - (request.arrival_ns + self.config.front_ns)),
                )
                record(read)
                dependency = read.end_ns
                read_xfer = self.timeline.transfer(
                    self._channel_id(m), self.config.transfer_page_bytes,
                    dependency, self.config.internal_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("MERGE_INTERNAL_TRANSFER", read_xfer.start_ns, read_xfer.end_ns, read_xfer.resource_id, read_xfer.bytes))
                internal_bytes += read_xfer.bytes
                self._touch_frontiers(array_id)
                self.__dict__["_buffer_ready"][array_id] = read_xfer.end_ns
                pages_read += 1
                merge = self.timeline.transfer(
                    self._host_id(m, Operation.READ), self.config.page_bytes, read_xfer.end_ns,
                    self._host_bandwidth(Operation.READ), direction="read")
                record(StageTiming("MERGE_HOST_TRANSFER", merge.start_ns, merge.end_ns, merge.resource_id, merge.bytes))
                dependency = host_write_origin = merge.end_ns
                host_write_bytes = self.config.page_bytes
                host_bytes += merge.bytes + host_write_bytes - segment.host_transfer_bytes
            if request.operation is Operation.READ:
                staged = self._staged_pages.get(self._staged_key(m)) if self._buffered_write_mode else None
                logical_start = max(request.address, segment.address) - segment.address
                logical_end = min(request.address + request.byte_count, segment.address + self.config.page_bytes) - segment.address
                if staged is not None and any(left <= logical_start and logical_end <= right for left, right in staged.ranges):
                    # The host copy is authoritative until media drain.  Do
                    # not charge a stale PAGE_READ or internal NAND transfer.
                    internal_bytes -= segment.internal_transfer_bytes
                    host = self.timeline.transfer(
                        self._host_id(m, Operation.READ), segment.host_transfer_bytes,
                        max(request.arrival_ns + self.config.front_ns, staged.ready_ns),
                        self._host_bandwidth(Operation.READ),
                        direction="read",
                    )
                    record(StageTiming("STAGING_READ", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                    staged.reader_completion_ns = max(staged.reader_completion_ns, host.end_ns)
                    completion = max(completion, host.end_ns)
                    staging_read_hits += 1
                    continue
                cache_slot, cache_hit = self._cache_slot(m)
                staging_overlay = staged is not None and any(left < logical_end and logical_start < right
                                                             for left, right in staged.ranges)
                if cache_hit:
                    internal_bytes -= segment.internal_transfer_bytes
                    host_ready = max(request.arrival_ns + self.config.front_ns, cache_slot.ready_ns)
                    if staging_overlay:
                        host_ready = max(host_ready, staged.ready_ns)
                        staging_read_merges += 1
                    host = self.timeline.transfer(
                        self._host_id(m, Operation.READ), segment.host_transfer_bytes,
                        host_ready, self._host_bandwidth(Operation.READ), direction="read")
                    record(StageTiming("CACHE_READ", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                    cache_slot.reader_completion_ns = max(cache_slot.reader_completion_ns, host.end_ns)
                    if staging_overlay:
                        staged.reader_completion_ns = max(staged.reader_completion_ns, host.end_ns)
                    read_cache_hits += 1
                    completion = max(completion, host.end_ns)
                    continue
                read_cache_misses += int(cache_slot is not None)
                read = self._media_stage("PAGE_READ", m, dependency, self.config.page_read_ns,
                                         eligible_ns=request.arrival_ns+self.config.front_ns)
                media_queue_wait_ns = max(
                    media_queue_wait_ns,
                    max(0.0, inherited_frontier - (request.arrival_ns + self.config.front_ns)),
                )
                record(read)
                pages_read += 1
                dependency = read.end_ns
                internal = self.timeline.transfer(
                    self._channel_id(m), segment.internal_transfer_bytes,
                    max(dependency, cache_slot.reader_completion_ns if cache_slot is not None else 0.0),
                    self.config.internal_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("INTERNAL_TRANSFER", internal.start_ns, internal.end_ns, internal.resource_id, internal.bytes))
                host_ready = internal.end_ns
                if staging_overlay:
                    # A partial staging hit still reads old media for the
                    # untouched bytes, then overlays dirty host bytes. With
                    # multiple host lanes, bus contention alone cannot enforce
                    # that the earlier write's payload has actually arrived.
                    host_ready = max(host_ready, staged.ready_ns)
                    staging_read_merges += 1
                host = self.timeline.transfer(
                    self._host_id(m, Operation.READ), segment.host_transfer_bytes,
                    host_ready, self._host_bandwidth(Operation.READ),
                    direction="read",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                if staging_overlay:
                    staged.reader_completion_ns = max(staged.reader_completion_ns, host.end_ns)
                if cache_slot is not None:
                    cache_slot.page_key = self._staged_key(m)
                    cache_slot.ready_ns = internal.end_ns
                    cache_slot.reader_completion_ns = host.end_ns
                completion = max(completion, host.end_ns)
                self._touch_frontiers(array_id)
                self.__dict__["_array_ready"][array_id] = max(
                    read.end_ns, self.__dict__["_array_ready"].get(array_id, 0.))
                self._touch_frontiers(array_id)
                self.__dict__["_buffer_ready"][array_id] = max(
                    internal.end_ns, self.__dict__["_buffer_ready"].get(array_id, 0.))
                self.timeline._remember_resource(array_id)
                self.timeline.ready_ns[array_id] = max(self.__dict__["_array_ready"].get(array_id, 0.),
                                                       self.timeline.ready_ns.get(array_id, 0.))
            else:
                host = self.timeline.transfer(
                    self._host_id(m, Operation.WRITE), host_write_bytes, host_write_origin,
                    self._host_bandwidth(Operation.WRITE),
                    direction="write",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                internal = self.timeline.transfer(
                    self._channel_id(m), segment.internal_transfer_bytes,
                    max(host.end_ns, self.__dict__["_buffer_ready"].get(array_id, 0.0)),
                    self.config.internal_bandwidth_gb_s,
                    direction="write",
                )
                record(StageTiming("INTERNAL_TRANSFER", internal.start_ns, internal.end_ns, internal.resource_id, internal.bytes))
                program_start = max(internal.end_ns, self.__dict__["_array_ready"].get(array_id, 0.0))
                erase_end = self._reserve_auto_erase(m, program_start,
                                                     eligible_ns=internal.end_ns,
                                                     stages=stages)
                if erase_end > program_start:
                    erases += 1
                program = self._media_stage("PAGE_PROGRAM", m,
                                            max(program_start, erase_end),
                                            self.config.page_program_ns,
                                            eligible_ns=internal.end_ns)
                record(program)
                self._log_programs((segment.address // self.config.page_bytes,))
                self._remember_program(segment.address // self.config.page_bytes, m)
                pages_programmed += 1
                completion = max(completion, program.end_ns)
                self._touch_frontiers(array_id)
                self.__dict__["_array_ready"][array_id] = program.end_ns
                self._touch_frontiers(array_id)
                self.__dict__["_buffer_ready"][array_id] = program.end_ns
                self.timeline._remember_resource(array_id)
                self.timeline.ready_ns[array_id] = program.end_ns
        counters = {
            "host_transfer_bytes": host_bytes,
            "internal_transfer_bytes": internal_bytes,
            "physical_read_bytes": pages_read * self.config.page_bytes,
            "physical_write_bytes": pages_programmed * self.config.page_bytes,
            "pages_read": pages_read,
            "pages_programmed": pages_programmed,
            "merge_bytes": merge_bytes,
            "merged_pages": merged_pages,
            "staging_write_pages": staging_write_pages,
            "staging_read_hits": staging_read_hits,
            "staging_read_merges": staging_read_merges,
            "read_cache_hits": read_cache_hits,
            "read_cache_misses": read_cache_misses,
            "erase_operations": erases,
            "page_count": segment_count,
            "details_truncated": details_truncated,
            "bandwidth_ceiling_gb_s": (
                self.config.host_bandwidth_for(request.operation)
                if staging_write_pages or staging_read_hits or read_cache_hits
                else self._bandwidth_ceiling_gb_s(request.operation)
            ),
            "bandwidth_ceiling_scope": (
                "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain"
                if staging_write_pages or staging_read_hits or read_cache_hits
                else "device transfer upper bound: min(directional host, summed independent channels); excludes array latency"
            ),
            "array_units": self.config.array_units,
            "write_completion": (
                "host_buffer_ack_pending_media_drain"
                if request.operation is Operation.WRITE and staging_write_pages
                else "media_program_complete" if request.operation is Operation.WRITE else "n/a"
            ),
            "host_ack_completion_ns": completion if staging_write_pages else None,
            "media_drain_completion_ns": (
                None if staging_write_pages
                else completion if request.operation is Operation.WRITE else None
            ),
            "media_queue_wait_ns": float(media_queue_wait_ns),
        }
        self._last_media_queue_wait_ns = float(media_queue_wait_ns)
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=logical_bytes,
            transfer_bytes=internal_bytes,
            mapping=tuple(mappings), stages=tuple(stages),
            counters=counters,
        )

    def _execute_accelerated_accepted(self, request: AccessRequest, page_count: int) -> TransactionResult:
        from ._nand_acceleration import execute_full_pages

        page = self.config.page_bytes
        end = request.address + request.byte_count
        cursor = request.address
        parts = []
        accelerated_pages = 0
        # Partial boundary pages retain the original RMW/transfer transitions.
        if cursor % page:
            length = min(end - cursor, page - cursor % page)
            parts.append(self._execute_detailed_accepted(replace(request, address=cursor, byte_count=length)))
            cursor += length
        full_pages = (end - cursor) // page
        if full_pages:
            if request.operation is Operation.WRITE:
                self._validate_write_order(range(cursor // page, cursor // page + full_pages))
            result = execute_full_pages(self, request, cursor // page, full_pages)
            if result is not None:
                accelerated_pages = result["accelerated_pages"]
                count = result["page_count"]
                parts.append(TransactionResult(
                    request_id=request.request_id, operation=request.operation,
                    arrival_ns=request.arrival_ns, completion_ns=result["completion_ns"],
                    logical_bytes=count * page, transfer_bytes=result["internal_transfer_bytes"],
                    counters={
                        "host_transfer_bytes": result["host_transfer_bytes"],
                        "internal_transfer_bytes": result["internal_transfer_bytes"],
                        "pages_read": count if request.operation is Operation.READ else 0,
                        "pages_programmed": count if request.operation is Operation.WRITE else 0,
                        "erase_operations": result.get("erase_operations", 0),
                        "media_queue_wait_ns": result.get("media_queue_wait_ns", 0.0),
                    },
                ))
                cursor += count * page
        if cursor < end:
            parts.append(self._execute_detailed_accepted(replace(request, address=cursor, byte_count=end - cursor)))
        pages_read = sum(part.counters["pages_read"] for part in parts)
        pages_programmed = sum(part.counters["pages_programmed"] for part in parts)
        host_bytes = sum(part.counters["host_transfer_bytes"] for part in parts)
        internal_bytes = sum(part.counters["internal_transfer_bytes"] for part in parts)
        erases = sum(part.counters.get("erase_operations", 0) for part in parts)
        media_queue_wait_ns = max(
            (float(part.counters.get("media_queue_wait_ns", 0.0)) for part in parts),
            default=0.0,
        )
        self._last_media_queue_wait_ns = media_queue_wait_ns
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=max(part.completion_ns for part in parts),
            logical_bytes=request.byte_count, transfer_bytes=internal_bytes,
            counters={
                "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": pages_read * page, "physical_write_bytes": pages_programmed * page,
                "pages_read": pages_read, "pages_programmed": pages_programmed,
                "erase_operations": erases, "page_count": page_count,
                "details_truncated": page_count > self.config.max_expanded_segments,
                "accelerated_pages": accelerated_pages,
                "bandwidth_ceiling_gb_s": self._bandwidth_ceiling_gb_s(request.operation),
                "bandwidth_ceiling_scope": "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
                "array_units": self.config.array_units,
                "write_completion": "media_program_complete" if request.operation is Operation.WRITE else "n/a",
                "media_queue_wait_ns": media_queue_wait_ns,
            },
        )

    def execute_batch(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        requests = tuple(requests)
        # Match scalar submission's pre-admission validation for the whole
        # public batch. Unexpected execution failures still require the owning
        # PhysicalRuntimeContext transaction for rollback.
        for request in requests:
            self._validate_request(request)
        return tuple(self.submit(request) for request in requests)

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return one aggregate with achieved bandwidth."""
        from .memory_transfer import summarize_batch
        results = self.execute_batch(requests)
        return summarize_batch(results, bandwidth_ceiling_gb_s=max(
            (result.bandwidth_ceiling_gb_s for result in results),
            default=self._bandwidth_ceiling_gb_s()))


# Fewest touched pages the compiled generic page machine takes. Without a
# resident controller every bulk request packs and publishes all banks, which
# only pays off for long spans; with one, packing and publication cover the
# request's own footprint, so any multi-page read or write runs compiled.
BULK_MIN_PAGES = 128
RESIDENT_MIN_PAGES = 2


def _bulk_min_pages(core, operation):
    if operation in (Operation.READ, Operation.WRITE) and core._nand_numeric_state is not None:
        return RESIDENT_MIN_PAGES
    return BULK_MIN_PAGES


def _generic_bulk_span(request, config):
    """A span the generic two-buffer page machine runs compiled beside a resident controller.

    A READ or WRITE touching at least RESIDENT_MIN_PAGES 4 KiB pages. A READ
    may start or end inside a page (its boundary pages move their
    host-granularity span); a partial WRITE is the scalar machine's host
    merge, or its reject policy's error.
    """
    if request.operation not in (Operation.READ, Operation.WRITE) or request.byte_count <= 0:
        return False
    aligned = not request.address % 4096 and not request.byte_count % 4096
    partial = not 4096 % config.host_granularity_bytes and request.operation is Operation.READ
    return ((aligned or partial)
            and (request.address + request.byte_count - 1) // 4096 - request.address // 4096
            >= RESIDENT_MIN_PAGES - 1)


_RESIDENT_SCALAR_METHODS = {name: getattr(NandCore, name) for name in (
    "_validate_request", "_write_buffer_payload_bytes",
    "_resident_scalar_eligible", "_resident_generic_bulk_eligible", "_submit_resident_generic",
    "_admit_host_request", "_record_host_admission", "_submit_scalar_request", "_cache_slot", "_invalidate_cache",
    "_array_id", "_array_resource_id", "_channel_id", "_host_id", "_host_bandwidth",
    "_staged_key", "_bank_key", "_block_key", "_log_programs", "_scratchpad_id", "_staging_host_id", "_touch_cache_bank",
    "_bandwidth_ceiling_gb_s", "_admission_depth_scope", "_resident_resource_contract",
    "_resident_handoff")}


def nand_service(config: NandConfig, request: AccessRequest, *, timeline: Optional[ResourceTimeline] = None) -> TransactionResult:
    return NandCore(config, timeline).submit(request)


def summarize_nand_observations(observations):
    """Combine actual page event populations, retaining all idle plane zeros."""
    observations = tuple(item for item in observations if item is not None)
    if not observations:
        return None
    events = tuple(event for item in observations for event in item.get("events", ()))
    planes_by_owner = {}
    for item in observations:
        for owner, count in item["configured_planes_by_owner"].items():
            if owner in planes_by_owner and planes_by_owner[owner] != count:
                raise ValueError("NAND observation geometry changed within one physical owner")
            planes_by_owner[owner] = count
    start = min((e["eligible_ns"] for e in events), default=0.)
    end = max((e["end_ns"] for e in events), default=start)
    window = end-start
    queues = {}
    for operation in ("read", "program"):
        selected = [e for e in events if e["operation"] == operation]
        changes = {}
        for event in selected:
            if event["start_ns"] > event["eligible_ns"]:
                changes[event["eligible_ns"]] = changes.get(event["eligible_ns"], 0)+1
                changes[event["start_ns"]] = changes.get(event["start_ns"], 0)-1
        histogram = {}
        depth, peak, last = 0, 0, start
        for timestamp, delta in sorted(changes.items()):
            histogram[depth] = histogram.get(depth, 0.)+timestamp-last
            depth += delta
            peak = max(peak, depth)
            last = timestamp
        histogram[depth] = histogram.get(depth, 0.)+end-last
        def quantile(q):
            accumulated = 0.
            for level, duration in sorted(histogram.items()):
                accumulated += duration
                if accumulated >= q*window:
                    return level
            return 0
        wait = sum(e["start_ns"]-e["eligible_ns"] for e in selected)
        queues[operation] = {"operations": len(selected), "peak": peak,
            "mean_time_weighted": wait/window if window else 0.,
            "p50_time_weighted": quantile(.5), "p95_time_weighted": quantile(.95),
            "p99_time_weighted": quantile(.99), "p999_time_weighted": quantile(.999),
            "total_page_wait_ns": wait, "depth_duration_ns": histogram,
            "scope": "physical array-operation queue, eligible-to-start, excluding in-service; global sum across recorded owners"}
    loads, blocks = {}, {}
    for event in events:
        row = loads.setdefault(event["plane_id"], {"read_operations": 0, "program_operations": 0,
            "erase_operations": 0, "busy_ns": 0., "read_bytes": 0, "write_bytes": 0})
        row[event["operation"]+"_operations"] += 1
        row["busy_ns"] += event["end_ns"]-event["start_ns"]
        if event["operation"] != "erase":
            row["read_bytes" if event["operation"] == "read" else "write_bytes"] += event["physical_bytes"]
        if event["wear_enabled"]:
            old = blocks.get(event["block_id"])
            if old is None or event["end_ns"] >= old["end_ns"]:
                blocks[event["block_id"]] = {key: event[key] for key in
                    ("end_ns", "pe_cycles_after", "block_programs_after", "block_erases_after", "rated_pe_cycles")}
                try:
                    endurance_fraction = event["pe_cycles_after"]/event["rated_pe_cycles"]
                except OverflowError:
                    endurance_fraction = None
                blocks[event["block_id"]].update(
                    authored_endurance_fraction_consumed=endurance_fraction,
                    authored_endurance_fraction_status=("outside_float_range" if endurance_fraction is None else "finite"),
                    authored_pe_cycles_remaining=max(0,event["rated_pe_cycles"]-event["pe_cycles_after"]),
                    at_or_above_authored_endurance=event["pe_cycles_after"]>=event["rated_pe_cycles"],
                    endurance_scope="assumption budget only; reaching this value does not simulate data loss or failure")
    count = sum(planes_by_owner.values())
    busy = [r["busy_ns"] for r in loads.values()]
    mean = sum(busy)/count if count else 0.
    variance = (sum((x-mean)**2 for x in busy)+(count-len(busy))*mean**2)/count if count else 0.
    return {"events": events, "configured_planes_by_owner": planes_by_owner,
        "wear_enabled": any(item.get("wear_enabled") for item in observations),
        "measurement_window": {"start_ns": start, "end_ns": end, "duration_ns": window},
        "read_queue_depth": queues["read"], "program_queue_depth": queues["program"],
        "physical_plane_loads": loads,
        "physical_plane_imbalance": {"configured_planes": count, "observed_planes": len(loads),
            "idle_planes": count-len(loads), "mean_busy_ns": mean,
            "coefficient_of_variation": sqrt(variance)/mean if mean else None,
            "max_over_mean": max(busy, default=0.)/mean if mean else None,
            "scope": "authored physical channel/target/die/LUN/plane mapping, includes configured idle planes; distinct from effective service lane IDs; uncalibrated geometry"},
        "wear_blocks": blocks,
        "actual_erase_operations": sum(e["operation"] == "erase" for e in events),
        "actual_program_operations": sum(e["operation"] == "program" for e in events),
        "wear_extra_ns": sum(e.get("wear_extra_ns", 0.) for e in events),
        "scope": "actual executed array stages; explicit erase-count wear feedback when enabled; no FTL or implicit erase inference"}


def _native_frontier_writers(core):
    if type(core) is not NandCore:
        return False
    for name, original in _OWNED_OWNER_NAND_METHODS:
        method = getattr(core, name)
        if getattr(method, "__func__", method) is not original:
            return False
    from ._nand_acceleration import has_native_timeline_hooks
    return has_native_timeline_hooks()


_OWNED_OWNER_NAND_METHODS = tuple((name, getattr(NandCore, name))
    for name, value in vars(NandCore).items() if callable(value))

__all__ = ["NandCore", "nand_service", "summarize_nand_observations"]
