"""Shared page-level NAND core for SSD and HBF configurations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from typing import Dict, Iterable, Optional

from .memory_mapping import map_nand_address, split_nand_request, validate_nand_request
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, NandConfig, Operation, StageTiming, TransactionResult


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


class _NandTransaction:
    """Undo only media indexes changed while this transaction is open."""

    def __init__(self, core):
        self.program_frontiers = {}
        self.block_wear = {}
        self.cache_banks = {}
        self.cache_slots = {}
        self.cache_snapshot = None
        self.array_frontiers = dict(core.__dict__["_array_ready"])
        self.buffer_frontiers = dict(core.__dict__["_buffer_ready"])
        self.numeric_at_begin = core._nand_numeric_state
        self.numeric = None

    def capture_numeric(self, packed):
        if packed is self.numeric_at_begin and self.numeric is None:
            self.numeric = packed.checkpoint()

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
        for target, saved in ((core._next_program_page, self.program_frontiers),
                              (core._block_wear_state, self.block_wear)):
            for key, previous in saved.items():
                if previous is None:
                    target.pop(key, None)
                else:
                    target[key] = previous
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
        core.__dict__["_array_ready"] = dict(self.array_frontiers)
        core.__dict__["_buffer_ready"] = dict(self.buffer_frontiers)
        if self.numeric is not None:
            self.numeric_at_begin.restore(self.numeric)
        core._nand_numeric_state = self.numeric_at_begin


class NandCore:
    """Page read/program/erase with optional explicit erase wear.

    There is no FTL, garbage collector or cell-content validity model.
    ``ocp_hbf_0_7`` models complete 4 KiB write units, page order and page-0
    automatic erase. It does not expose AXI fragments/pending responses,
    IDs/error packets, ECC, retention, scratchpad or wear-level remapping.
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
        self._next_program_page: Dict[tuple[int, ...], int] = {}
        self._nand_transactions: list[_NandTransaction] = []
        self._nand_numeric_state = None
        self._nand_numeric_enabled = False
        self._nand_state_exposed = False
        self._observation_events: list[dict] = []
        self._observe_media = bool(config.detailed_observability or config.wear_model)

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

    def _set_program_frontier(self, key, value):
        if self._nand_transactions:
            previous = self._next_program_page.get(key)
            for token in self._nand_transactions:
                token.program_frontiers.setdefault(key, previous)
        self._next_program_page[key] = value

    def _update_program_frontiers(self, frontiers):
        if self._nand_transactions:
            for key in frontiers:
                previous = self._next_program_page.get(key)
                for token in self._nand_transactions:
                    token.program_frontiers.setdefault(key, previous)
        self._next_program_page.update(frontiers)

    @property
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

    def _host_bandwidth(self, operation) -> float:
        """Configured rates are device aggregates, divided among host channels."""
        return self.config.host_bandwidth_for(operation) / self.config.host_channels

    def _admission_depth_scope(self) -> str:
        if self.config.queue_scope == "host_channel":
            return "outstanding page commands per host channel sampled at admission, including the admitted command"
        return "outstanding host requests sampled at admission, including the admitted request; not NAND page queues"

    def _admit_host_request(self, request, mapping=None):
        """Find a free command credit without changing another host's clock."""
        host_channel = None
        if self.config.queue_scope == "host_channel":
            if mapping is None:
                mapping = map_nand_address(self.config, request.address)
            host_channel = mapping.channel % self.config.host_channels
            population = self._host_inflight.get(host_channel, ())
            # _acceptance_ns is a device clock only when the caller explicitly
            # advances/drains the core; ordinary host admissions never set it.
            acceptance = max(self._host_acceptance_ns.get(host_channel, 0.), self._acceptance_ns)
        else:
            population = self._inflight_operations
            acceptance = self._acceptance_ns
        effective = max(request.arrival_ns, acceptance)
        admitted = [(end, operation) for end, operation in population if end > effective]
        write_limit = self.config.max_outstanding_write_requests
        while True:
            blocked_until = []
            if len(admitted) >= self.config.max_outstanding_requests:
                blocked_until.append(min(end for end, _ in admitted))
            if request.operation is Operation.WRITE and write_limit is not None:
                writes = [end for end, operation in admitted if operation is Operation.WRITE]
                if len(writes) >= write_limit:
                    blocked_until.append(min(writes))
            if not blocked_until:
                return effective, admitted, host_channel
            effective = max(effective, max(blocked_until))
            admitted = [(end, operation) for end, operation in admitted if end > effective]

    def _record_host_admission(self, host_channel, effective, admitted, completion, operation):
        population = [*admitted, (completion, operation)]
        if host_channel is None:
            self._acceptance_ns = effective
            self._inflight_operations = population
            self._inflight = [end for end, _ in population]
        else:
            self._host_acceptance_ns[host_channel] = effective
            self._host_inflight[host_channel] = population

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

    def _validate_program_order(self, request, expected=None) -> None:
        if self.config.protocol_mode != "ocp_hbf_0_7" or request.operation is not Operation.WRITE:
            return
        expected = {} if expected is None else expected
        for segment in split_nand_request(request, self.config):
            mapping = segment.mapping
            block = self._block_key(mapping)
            next_page = expected.get(block, self._next_program_page.get(block, 0))
            if mapping.page != 0 and mapping.page != next_page:
                raise ValueError("ocp_hbf_0_7 write order violation: block {} requires page {}, received {}"
                                 .format(block, next_page, mapping.page))
            expected[block] = mapping.page + 1

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

    def _validate_request(self, request: AccessRequest, *, program_order=None) -> None:
        """Reject invalid geometry or admission size before mutating calendars."""
        validate_nand_request(request, self.config)
        self._validate_program_order(request, program_order)
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
        for staged in tuple(self._staged_pages.values()):
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
            program = self._media_stage(
                "PAGE_PROGRAM", m, internal.end_ns, self.config.page_program_ns, eligible_ns=internal.end_ns
            )
            self.__dict__["_array_ready"][array_id] = program.end_ns
            self.__dict__["_buffer_ready"][array_id] = program.end_ns
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

    def submit(self, request: AccessRequest) -> TransactionResult:
        # GC interleaves a scalar read and PROGRAM with bulk page execution.
        # Hand only that bank to the unchanged scalar state machine instead
        # of publishing and repacking all banks for every copied page.
        if (self._nand_numeric_state is not None
                and "_resident_scalar_eligible" not in self.__dict__
                and type(self)._resident_scalar_eligible is _RESIDENT_SCALAR_METHODS["_resident_scalar_eligible"]
                and self._resident_scalar_eligible(request)):
            self._validate_request(request)
            result = self._submit_resident_scalar(request)
            if result is not None:
                return result
        self._nand_publish_numeric()
        self._validate_request(request)
        if (self.config.protocol_mode == "ocp_hbf_0_7"
                and request.address // self.config.page_bytes
                != (request.address + request.byte_count - 1) // self.config.page_bytes):
            return self._submit_ocp_page_commands(request)
        if (type(self) is NandCore and not self.capture_details and not self._observe_media
                and self.config.protocol_mode == "ocp_hbf_0_7"
                and request.address // self.config.page_bytes
                == (request.address + request.byte_count - 1) // self.config.page_bytes):
            result = self._submit_ocp_single_page_summary(request)
            if result is not None:
                return result
        self._observation_events = []
        effective_arrival, admitted, host_channel = self._admit_host_request(request)
        before_metrics = self.timeline.metrics_snapshot(
            max_intervals=(
                self.config.max_expanded_segments
                if self.capture_details
                else 0
            )
        )
        effective_arrival, write_buffer_bytes, write_buffer_wait_ns, drained = self._admit_write_buffer(
            request, effective_arrival
        )
        admitted = [(end, operation) for end, operation in admitted if end > effective_arrival]
        admission_depths = {
            "host_outstanding_at_admission_peak": len(admitted) + 1,
            "host_read_outstanding_at_admission_peak":
                sum(operation is Operation.READ for _end, operation in admitted)
                + int(request.operation is Operation.READ),
            "host_write_outstanding_at_admission_peak":
                sum(operation is Operation.WRITE for _end, operation in admitted)
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

    def _submit_ocp_page_commands(self, request):
        """Split a validated host span into separately admitted page commands."""
        # submit() prevalidated the complete span and every program frontier.
        # Reaching a later invalid page must never leave earlier pages scheduled.
        counters = {}
        mappings, stages, observations = [], [], []
        completion = request.arrival_ns
        transfer_bytes = 0
        page_count = 0
        interval_counts = {"resource_intervals": 0, "resource_interval_payloads": 0}
        intervals_truncated = False
        details_truncated = False
        maxima = {"array_units", "bandwidth_ceiling_gb_s", "write_buffer_peak_bytes",
                  "write_buffer_occupancy_bytes", "write_buffer_draining_bytes",
                  "host_outstanding_at_admission_peak", "host_read_outstanding_at_admission_peak",
                  "host_write_outstanding_at_admission_peak", "host_ack_completion_ns",
                  "media_drain_completion_ns"}
        for segment in split_nand_request(request, self.config):
            address = max(request.address, segment.address)
            size = min(request.address + request.byte_count,
                       segment.address + self.config.page_bytes) - address
            result = self.submit(replace(request, address=address, byte_count=size))
            page_count += 1
            completion = max(completion, result.completion_ns)
            transfer_bytes += result.transfer_bytes
            observations.extend(self._observation_events)
            if page_count <= self.config.max_expanded_segments and not details_truncated:
                mappings.extend(result.mapping)
                stages.extend(result.stages)
            else:
                mappings.clear()
                stages.clear()
                details_truncated = True
            for key, value in result.counters.items():
                if key == "nand_observations":
                    continue
                if key in {"resource_busy_ns", "resource_bytes"}:
                    target = counters.setdefault(key, {})
                    for resource, count in value.items():
                        target[resource] = target.get(resource, 0) + count
                elif key == "resource_last_intervals":
                    counters.setdefault(key, {}).update(value)
                elif key in {"resource_intervals", "resource_interval_payloads"}:
                    target = counters.setdefault(key, {})
                    for resource, intervals in value.items():
                        remaining = max(0, self.config.max_expanded_segments - interval_counts[key])
                        retained = intervals[:remaining]
                        if retained:
                            target.setdefault(resource, []).extend(retained)
                            interval_counts[key] += len(retained)
                        intervals_truncated |= len(retained) != len(intervals)
                elif isinstance(value, bool):
                    counters[key] = counters.get(key, False) or value
                elif isinstance(value, (int, float)):
                    counters[key] = (max(counters.get(key, value), value) if key in maxima
                                     else counters.get(key, 0) + value)
                elif value is not None or key not in counters:
                    counters[key] = value
        self._observation_events = observations
        counters.update(self._observations())
        if any(counters.get(key, 0) for key in ("staging_write_pages", "staging_read_hits", "read_cache_hits")):
            counters["bandwidth_ceiling_gb_s"] = self.config.host_bandwidth_for(request.operation)
            counters["bandwidth_ceiling_scope"] = (
                "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain")
        for key in ("resource_intervals", "resource_interval_payloads"):
            counters[key] = {resource: tuple(intervals)
                             for resource, intervals in counters.get(key, {}).items()}
        counters["intervals_truncated"] = counters.get("intervals_truncated", False) or intervals_truncated
        counters["details_truncated"] = details_truncated
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=request.byte_count, transfer_bytes=transfer_bytes,
            mapping=tuple(mappings), stages=tuple(stages), counters=counters)

    def _resident_scalar_eligible(self, request):
        if (type(self) is not NandCore or type(request) is not AccessRequest
                or type(self.timeline) is not ResourceTimeline or self.capture_details
                or self._observe_media or self._nand_state_exposed
                or self._cache_use >= 2**63 - 1
                or self.config.protocol_mode != "ocp_hbf_0_7"
                or request.address // self.config.page_bytes
                != (request.address + request.byte_count - 1) // self.config.page_bytes):
            return False
        # An authored replacement can access arbitrary controller banks. Such
        # callbacks continue to receive the fully published controller state.
        return (all(name not in self.__dict__ and getattr(NandCore, name) is method
                    for name, method in _RESIDENT_SCALAR_METHODS.items())
                and getattr(self._nand_numeric_state, "scalar_resource_contract", None)
                == self._resident_resource_contract())

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
                    ("host_resource_id", "nand:host"))),
                tuple(self.timeline.resource_aliases.items()))

    def _submit_resident_scalar(self, request):
        """Bridge one ordinary scalar command to its resident bank exactly.

        The numeric arrays remain the authoritative state. Python slots for
        this bank retain their identities, and the existing scalar executor
        performs every cache/admission/calendar transition and bill. Journals
        capture the resident preimage before this partial publication, just
        as for a complete publication.
        """
        packed = self._nand_numeric_state
        config = self.config
        base_units = (config.channels * config.targets_per_channel
                      * config.dies_per_target * config.luns_per_die)
        bank = request.address // 4096 % (base_units * config.planes_per_lun)
        unit = bank if config.planes_independent else bank % base_units
        arrays, buffers, tags, ready, readers, uses, present = packed.arrays
        bank_keys, tag_prefixes, array_names, pages_per_block = packed.layout
        name, key = array_names[unit], bank_keys[bank]
        array_values = self.__dict__["_array_ready"]
        buffer_values = self.__dict__["_buffer_ready"]
        cache = self.__dict__["_read_cache"]
        for token in self._nand_transactions:
            token.capture_numeric(packed)
        if packed.dirty_units[unit]:
            array_values[name], buffer_values[name] = float(arrays[unit]), float(buffers[unit])
        if packed.dirty_banks[bank]:
            if self._nand_transactions:
                self._touch_cache_bank(key)
            slots = cache.get(key)
            if slots is None:
                slots = [_ReadCacheSlot(), _ReadCacheSlot()]
                cache[key] = slots
            prefix = tag_prefixes[bank]
            for index, slot in enumerate(slots):
                page = int(tags[bank, index])
                slot.page_key = None if page < 0 else (*prefix, *divmod(page, pages_per_block))
                slot.ready_ns = float(ready[bank, index])
                slot.reader_completion_ns = float(readers[bank, index])
                slot.last_use = int(uses[bank, index])
        try:
            return self._submit_ocp_single_page_summary(request)
        finally:
            # The scalar executor can raise after a reservation. Retain its
            # exact partial state; a transaction still restores the captured
            # preimage, and a later public inspection publishes this state.
            if name in array_values:
                arrays[unit] = array_values[name]
                buffers[unit] = buffer_values.get(name, 0.)
                packed.dirty_units[unit] = True
            slots = cache.get(key)
            if slots is not None:
                present[bank] = True
                packed.dirty_banks[bank] = True
                for index, slot in enumerate(slots):
                    page = slot.page_key
                    tags[bank, index] = -1 if page is None else page[-2] * pages_per_block + page[-1]
                    ready[bank, index], readers[bank, index] = slot.ready_ns, slot.reader_completion_ns
                    uses[bank, index] = slot.last_use

    def submit_read_pages(self, addresses, arrival_ns, *, link_program=()):
        """Execute full-page reads interleaved with external links in order.

        Return page completions, one NAND bill and per-link bills, or None if
        the ordinary per-page executor is needed for the authored features.
        Arrival may be one timestamp or one timestamp per page. No page,
        cache or admission transition is skipped by the compiled loop.
        """
        try:
            from ._nand_execution import execute_pages
        except (ImportError, OSError):
            return None
        result = execute_pages(self, addresses, arrival_ns, link_program)
        if result is not None and self._nand_numeric_state is not None:
            self._nand_numeric_state.scalar_resource_contract = self._resident_resource_contract()
        return result

    def submit_write_pages(self, addresses, arrival_ns, *, link_program=()):
        """Program full pages with each page's incoming links before admission."""
        try:
            from ._nand_execution import execute_pages
        except (ImportError, OSError):
            return None
        result = execute_pages(self, addresses, arrival_ns, link_program, operation=Operation.WRITE)
        if result is not None and self._nand_numeric_state is not None:
            self._nand_numeric_state.scalar_resource_contract = self._resident_resource_contract()
        return result

    def _submit_ocp_single_page_summary(self, request: AccessRequest) -> Optional[TransactionResult]:
        """Execute the same strict page transitions without trace allocations.

        Host staging interleaves each page with its external link reservation,
        so these requests cannot be replaced by a single bulk NAND transfer.
        Only the three touched resources need accounting baselines; scanning
        all resources in the shared machine timeline for every page is wasteful.
        Multi-lane/custom timelines retain the detailed reservation executor.
        """
        timeline, config = self.timeline, self.config
        if type(timeline) is not ResourceTimeline:
            return None
        mapping = map_nand_address(config, request.address // config.page_bytes * config.page_bytes)
        array_id = self._array_id(mapping)
        array_resource = self._array_resource_id(mapping)
        channel = self._channel_id(mapping)
        host_resource = self._host_id(mapping, request.operation)
        resources = (array_resource, channel, host_resource)
        if any(len(timeline.lane_available.get(resource, (0.,))) != 1 for resource in resources):
            return None
        if any(not isfinite(value) or value < 0 for resource in resources
               for value in timeline.lane_available.get(resource, (timeline.ready_ns.get(resource, 0.),))):
            return None
        effective_arrival, admitted, host_channel = self._admit_host_request(request, mapping)
        before_busy = {resource: timeline.busy_ns.get(resource, 0.) for resource in resources}
        before_bytes = {resource: timeline.bytes_moved.get(resource, 0) for resource in resources}
        timeline._intervals = {}
        timeline._interval_limit = timeline._interval_count = 0
        timeline._intervals_truncated = False
        timeline._interval_payloads = {}
        timeline._touched = set()
        self._observation_events = []

        def reserve(resource, earliest, duration, direction=None, payload=None):
            lanes = timeline.lane_available.get(resource)
            if lanes is None:
                lanes = [timeline.ready_ns.get(resource, 0.)]
                timeline.lane_available[resource] = lanes
            start = max(float(earliest), float(lanes[0]))
            end = start + float(duration)
            busy = timeline.busy_ns.get(resource, 0.) + float(duration)
            if not isfinite(end) or not isfinite(busy):
                raise ValueError("resource reservation exceeds finite timing range")
            lanes[0] = end
            timeline.ready_ns[resource] = end
            timeline.busy_ns[resource] = busy
            timeline.last_intervals[resource] = (start, end)
            timeline._touched.add(resource)
            timeline._intervals_truncated = True
            if direction:
                timeline.directions[resource] = direction
            timeline.lane_directions.pop(resource, None)
            if payload is not None:
                timeline.bytes_moved[resource] = timeline.bytes_moved.get(resource, 0) + payload
            return end

        base = effective_arrival + config.front_ns
        dependency = max(base, self.__dict__["_array_ready"].get(array_id, 0.), self.__dict__["_buffer_ready"].get(array_id, 0.))
        host_bytes = request.byte_count  # Strict requests are already 64-byte aligned.
        internal_bytes = config.transfer_page_bytes
        host_ns = host_bytes / self._host_bandwidth(request.operation)
        internal_ns = internal_bytes / config.internal_bandwidth_gb_s
        pages_read = pages_programmed = erases = read_cache_hits = read_cache_misses = 0
        if request.operation is Operation.READ:
            slot, hit = self._cache_slot(mapping)
            if hit:
                internal_bytes = 0
                completion = reserve(host_resource, max(base, slot.ready_ns), host_ns, "read", host_bytes)
                slot.reader_completion_ns = max(slot.reader_completion_ns, completion)
                read_cache_hits = 1
            else:
                read_cache_misses = int(slot is not None)
                read_end = reserve(array_resource, dependency, config.page_read_ns)
                internal_end = reserve(channel, max(read_end, slot.reader_completion_ns if slot else 0.),
                                       internal_ns, "read", internal_bytes)
                completion = reserve(host_resource, internal_end, host_ns, "read", host_bytes)
                if slot is not None:
                    slot.page_key = self._staged_key(mapping)
                    slot.ready_ns = internal_end
                    slot.reader_completion_ns = completion
                self.__dict__["_array_ready"][array_id] = read_end
                self.__dict__["_buffer_ready"][array_id] = internal_end
                timeline.ready_ns[array_id] = read_end
                pages_read = 1
        else:
            self._invalidate_cache(mapping)
            if mapping.page == 0:
                self._invalidate_cache(mapping, block=True)
                erase_end = reserve(array_resource, dependency, config.block_erase_ns)
                self.__dict__["_array_ready"][array_id] = erase_end
                timeline.ready_ns[array_id] = erase_end
                erases = 1
            host_end = reserve(host_resource, base, host_ns, "write", host_bytes)
            internal_end = reserve(channel, max(host_end, self.__dict__["_buffer_ready"].get(array_id, 0.)),
                                   internal_ns, "write", internal_bytes)
            completion = reserve(array_resource, max(internal_end, self.__dict__["_array_ready"].get(array_id, 0.)),
                                 config.page_program_ns)
            self.__dict__["_array_ready"][array_id] = self.__dict__["_buffer_ready"][array_id] = completion
            timeline.ready_ns[array_id] = completion
            self._set_program_frontier(self._block_key(mapping), mapping.page + 1)
            pages_programmed = 1
        completion = max(base, completion)
        moved = {resource: timeline.bytes_moved.get(resource, 0) - before_bytes[resource]
                 for resource in timeline._touched
                 if timeline.bytes_moved.get(resource, 0) != before_bytes[resource]}
        busy = {resource: timeline.busy_ns[resource] - before_busy[resource]
                for resource in timeline._touched if timeline.busy_ns[resource] != before_busy[resource]}
        self._record_host_admission(host_channel, effective_arrival, admitted,
                                    completion, request.operation)
        return TransactionResult(
            request_id=request.request_id, operation=request.operation, arrival_ns=request.arrival_ns,
            completion_ns=completion, logical_bytes=request.byte_count, transfer_bytes=internal_bytes,
            counters={
                "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": pages_read * config.page_bytes,
                "physical_write_bytes": pages_programmed * config.page_bytes,
                "pages_read": pages_read, "pages_programmed": pages_programmed,
                "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": 0,
                "staging_read_hits": 0, "staging_read_merges": 0,
                "read_cache_hits": read_cache_hits, "read_cache_misses": read_cache_misses,
                "erase_operations": erases, "page_count": 1, "details_truncated": False,
                "bandwidth_ceiling_gb_s": config.host_bandwidth_for(request.operation) if read_cache_hits
                    else self._bandwidth_ceiling_gb_s(request.operation),
                "bandwidth_ceiling_scope":
                    "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain"
                    if read_cache_hits else
                    "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
                "array_units": config.array_units,
                "write_completion": "media_program_complete" if pages_programmed else "n/a",
                "host_ack_completion_ns": None,
                "media_drain_completion_ns": completion if pages_programmed else None,
                "pages_touched": 1, "read_operations": pages_read, "program_operations": pages_programmed,
                "rmw_read_operations": 0, "page_access_payload_bytes": request.byte_count,
                "page_access_capacity_bytes": config.page_bytes, "host_admission_samples": 1,
                "host_outstanding_at_admission_peak": len(admitted) + 1,
                "host_read_outstanding_at_admission_peak":
                    sum(operation is Operation.READ for _, operation in admitted) + int(request.operation is Operation.READ),
                "host_write_outstanding_at_admission_peak":
                    sum(operation is Operation.WRITE for _, operation in admitted) + int(request.operation is Operation.WRITE),
                "host_admission_depth_scope": self._admission_depth_scope(),
                "queue_wait_ns": effective_arrival - request.arrival_ns,
                "write_buffer_wait_ns": 0., "write_buffer_bytes": 0, "write_buffer_occupancy_bytes": 0,
                "write_buffer_peak_bytes": 0, "write_buffer_draining_bytes": 0, "buffered_write_mode": False,
                "resource_busy_ns": busy, "resource_bytes": moved,
                "resource_last_intervals": {resource: timeline.last_intervals[resource] for resource in timeline._touched},
                "resource_intervals": {}, "resource_interval_payloads": {},
                "intervals_truncated": timeline._intervals_truncated,
            },
        )

    def _execute_accepted(self, request: AccessRequest) -> TransactionResult:
        first_page = request.address // self.config.page_bytes
        last_page = (request.address + request.byte_count - 1) // self.config.page_bytes
        page_count = last_page - first_page + 1
        if (
            not self.capture_details
            and not self._observe_media
            and self.config.protocol_mode == "generic_nand"
            and self.config.host_channels == 1
            and not self.config.host_full_duplex
            and not self.config.read_cache_pages_per_bank
            and request.operation is not Operation.ERASE
            and page_count >= 128
            and not self._staged_pages
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
            if self.capture_details and not details_truncated:
                mappings.append(m)
            logical_bytes += segment.logical_bytes
            array_id = self._array_id(m)
            array_resource = self._array_resource_id(m)
            array_ready = self.__dict__["_array_ready"].get(array_id, 0.0)
            buffer_ready = self.__dict__["_buffer_ready"].get(array_id, 0.0)
            dependency = max(request.arrival_ns + self.config.front_ns, array_ready, buffer_ready)
            if request.operation is Operation.ERASE:
                self._invalidate_cache(m, block=True)
                stage = self._media_stage("BLOCK_ERASE", m, dependency, self.config.block_erase_ns,
                                          eligible_ns=request.arrival_ns+self.config.front_ns)
                record(stage)
                self.__dict__["_array_ready"][array_id] = stage.end_ns
                self.timeline.ready_ns[array_id] = stage.end_ns
                completion = max(completion, stage.end_ns)
                erases += 1
                continue

            host_bytes += segment.host_transfer_bytes
            internal_bytes += segment.internal_transfer_bytes
            partial = segment.logical_bytes != self.config.page_bytes
            if request.operation is Operation.WRITE:
                self._invalidate_cache(m)
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
            if request.operation is Operation.WRITE and partial and self.config.partial_page_policy == "read_modify_write":
                # Boundary pages need a simplified old-data read before the
                # new bytes can be programmed.  This is intentionally not a
                # persistent page cache.
                read = self._media_stage("RMW_PAGE_READ", m, dependency, self.config.page_read_ns,
                                         eligible_ns=request.arrival_ns+self.config.front_ns)
                record(read)
                dependency = read.end_ns
                read_xfer = self.timeline.transfer(
                    self._channel_id(m), self.config.transfer_page_bytes,
                    dependency, self.config.internal_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("RMW_INTERNAL_TRANSFER", read_xfer.start_ns, read_xfer.end_ns, read_xfer.resource_id, read_xfer.bytes))
                internal_bytes += read_xfer.bytes
                dependency = read_xfer.end_ns
                self.__dict__["_buffer_ready"][array_id] = read_xfer.end_ns
                pages_read += 1
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
                self.__dict__["_array_ready"][array_id] = read.end_ns
                self.__dict__["_buffer_ready"][array_id] = internal.end_ns
                self.timeline.ready_ns[array_id] = read.end_ns
            else:
                if self.config.protocol_mode == "ocp_hbf_0_7" and m.page == 0:
                    self._invalidate_cache(m, block=True)
                    erase = self._media_stage("BLOCK_ERASE", m, dependency, self.config.block_erase_ns,
                                              eligible_ns=request.arrival_ns + self.config.front_ns)
                    record(erase)
                    self.__dict__["_array_ready"][array_id] = erase.end_ns
                    self.timeline.ready_ns[array_id] = erase.end_ns
                    erases += 1
                host = self.timeline.transfer(
                    self._host_id(m, Operation.WRITE), segment.host_transfer_bytes,
                    request.arrival_ns + self.config.front_ns,
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
                program = self._media_stage("PAGE_PROGRAM", m,
                                            max(internal.end_ns, self.__dict__["_array_ready"].get(array_id, 0.0)),
                                            self.config.page_program_ns,
                                            eligible_ns=internal.end_ns)
                record(program)
                pages_programmed += 1
                completion = max(completion, program.end_ns)
                self.__dict__["_array_ready"][array_id] = program.end_ns
                self.__dict__["_buffer_ready"][array_id] = program.end_ns
                self.timeline.ready_ns[array_id] = program.end_ns
                if self.config.protocol_mode == "ocp_hbf_0_7":
                    self._set_program_frontier(self._block_key(m), m.page + 1)
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=logical_bytes,
            transfer_bytes=internal_bytes,
            mapping=tuple(mappings), stages=tuple(stages),
            counters={
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
            },
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
                    },
                ))
                cursor += count * page
        if cursor < end:
            parts.append(self._execute_detailed_accepted(replace(request, address=cursor, byte_count=end - cursor)))
        pages_read = sum(part.counters["pages_read"] for part in parts)
        pages_programmed = sum(part.counters["pages_programmed"] for part in parts)
        host_bytes = sum(part.counters["host_transfer_bytes"] for part in parts)
        internal_bytes = sum(part.counters["internal_transfer_bytes"] for part in parts)
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=max(part.completion_ns for part in parts),
            logical_bytes=request.byte_count, transfer_bytes=internal_bytes,
            counters={
                "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": pages_read * page, "physical_write_bytes": pages_programmed * page,
                "pages_read": pages_read, "pages_programmed": pages_programmed,
                "erase_operations": 0, "page_count": page_count,
                "details_truncated": page_count > self.config.max_expanded_segments,
                "accelerated_pages": accelerated_pages,
                "bandwidth_ceiling_gb_s": self._bandwidth_ceiling_gb_s(request.operation),
                "bandwidth_ceiling_scope": "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
                "array_units": self.config.array_units,
                "write_completion": "media_program_complete" if request.operation is Operation.WRITE else "n/a",
            },
        )

    def execute_batch(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        requests = tuple(requests)
        # Match scalar submission's pre-admission validation for the whole
        # public batch. Unexpected execution failures still require the owning
        # PhysicalRuntimeContext transaction for rollback.
        program_order = dict(self._next_program_page)
        for request in requests:
            self._validate_request(request, program_order=program_order)
        return tuple(self.submit(request) for request in requests)

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return one aggregate with achieved bandwidth."""
        from .memory_transfer import summarize_batch
        results = self.execute_batch(requests)
        return summarize_batch(results, bandwidth_ceiling_gb_s=max(
            (result.bandwidth_ceiling_gb_s for result in results),
            default=self._bandwidth_ceiling_gb_s()))


_RESIDENT_SCALAR_METHODS = {name: getattr(NandCore, name) for name in (
    "_validate_request", "_validate_program_order", "_write_buffer_payload_bytes",
    "_resident_scalar_eligible", "_submit_resident_scalar", "_submit_ocp_single_page_summary",
    "_admit_host_request", "_record_host_admission", "_cache_slot", "_invalidate_cache",
    "_array_id", "_array_resource_id", "_channel_id", "_host_id", "_host_bandwidth",
    "_staged_key", "_bank_key", "_block_key", "_set_program_frontier", "_touch_cache_bank",
    "_bandwidth_ceiling_gb_s", "_admission_depth_scope", "_resident_resource_contract")}


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


__all__ = ["NandCore", "nand_service", "summarize_nand_observations"]
