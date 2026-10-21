"""Finite-buffer bulk memory execution for the Exploring HBF comparison.

This is an explicitly selected *paper bandwidth abstraction*.  It does not
replace NandCore, implement an OCP FTL, infer NAND lanes from bandwidth, or
make SRAM perform arithmetic.  One event transfers a bounded tile, rather
than expanding its contents into 4-KiB physical transactions.  Table I's
directional bandwidth and the first-page latency are service constraints;
undisclosed SRAM and GPU-link inputs are mandatory research assumptions.

Resource reservations are deterministic FIFO reservations in submission
order.  Prefetch and compute are submitted separately so real dependencies,
finite credits and the lifetime of each buffer remain visible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
from functools import lru_cache
import heapq
import math
from typing import Any, Iterable, Mapping
from weakref import WeakSet

import numpy as np


def _positive(name: str, value: float, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'non-negative' if zero else 'positive'}")


def _bytes(name: str, value: int, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if zero else 1}")


class PaperMemoryCapacityError(ValueError):
    """A stack-local SRAM or device capacity cannot admit the request."""


class PaperBufferCreditError(RuntimeError):
    """Both read buffers still belong to unconsumed reads."""


@dataclass(frozen=True)
class PaperMemoryAssumptions:
    """Required local assumptions; none is claimed as a paper measurement.

    GB/s is decimal and therefore numerically equal to bytes/ns.  The shared
    I/O resource models an optional common GPU ingress/egress bottleneck;
    it is an actual resource, independent of memory bandwidth.  Supplying a
    very large value is an explicit ideal-link research choice, not a default.
    ``memory_full_duplex`` also specifies the assumed SRAM read/write ports;
    false makes staging fills and drains contend on the same stack-local port.
    Optional HBM overlap allocates a separately named GPU-side SRAM read
    buffer, using those same explicitly supplied SRAM port assumptions.  Its
    capacity is a local research input, not a Table I HBF SRAM attribution.
    Zero capacity leaves HBM read/compute serial; positive capacity enables
    a paid finite-buffer pipeline.  Its GPU-side location differs from HBF's
    logic-die SRAM: HBF prefetch is local and only consumption crosses the
    output link.  That drain uses explicit cut-through byte service, retains
    each link's own occupancy, and holds its logic slot through compute.
    """

    sram_read_gb_s_per_stack: float
    sram_write_gb_s_per_stack: float
    sram_access_ns: float
    host_read_gb_s: float
    host_write_gb_s: float
    host_full_duplex: bool
    memory_full_duplex: bool
    shared_io_gb_s: float
    shared_io_full_duplex: bool
    hbm_access_ns: float
    basis: str
    hbm_read_buffer_bytes_per_stack: int = 0

    def __post_init__(self) -> None:
        for key in ("sram_read_gb_s_per_stack", "sram_write_gb_s_per_stack",
                    "host_read_gb_s", "host_write_gb_s", "shared_io_gb_s"):
            _positive(key, getattr(self, key))
        for key in ("sram_access_ns", "hbm_access_ns"):
            _positive(key, getattr(self, key), zero=True)
        for key in ("host_full_duplex", "memory_full_duplex", "shared_io_full_duplex"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("undisclosed link/SRAM inputs require an explicit assumption basis")
        _bytes("hbm_read_buffer_bytes_per_stack", self.hbm_read_buffer_bytes_per_stack, zero=True)
        if self.hbm_read_buffer_bytes_per_stack % 2:
            raise ValueError("HBM read-buffer capacity must split into two equal nonempty slots")


@dataclass(frozen=True)
class PaperMemoryEvent:
    operation: str
    tier: str
    stack: int | None
    size_bytes: int
    ready_ns: float
    start_ns: float
    end_ns: float
    resources: tuple[str, ...]
    label: str = ""


@dataclass
class ReadTicket:
    ticket_id: int
    stack: int
    slot: int
    size_bytes: int
    read: PaperMemoryEvent
    consumed_at_ns: float | None = None
    released_at_ns: float | None = None
    cancelled: bool = False


@dataclass
class IntermediateLease:
    allocation_id: int
    owner: str
    stack: int
    size_bytes: int
    allocated_at_ns: float
    released_at_ns: float | None = None


class _SnapshotWatch:
    __slots__ = ("first_event_ns", "last_slot_tie_ns", "__weakref__")

    def __init__(self) -> None:
        self.first_event_ns = math.inf
        self.last_slot_tie_ns = -math.inf


@dataclass(frozen=True)
class PaperMemorySnapshot:
    """Immutable counters and boundary state for a quiescent-period proof."""

    owner: object
    serial: int
    signature: Any
    calendars: tuple[tuple[str, float, float, int], ...]
    byte_counts: tuple[tuple[str, int], ...]
    event_count: int
    cancelled_reads: int
    next_ticket: int
    next_lease: int
    bulk_advanced_waves: int
    pending_tickets: int
    slots: tuple[tuple[tuple[int | None, float], ...], ...]
    slot_releases: tuple[tuple[int, ...], ...]
    leases: tuple[tuple[int, float | None], ...]
    staging_live: tuple[int, ...]
    watch: _SnapshotWatch


@dataclass(frozen=True)
class PaperQuiescentTransition:
    """A same-instance, immutable whole-program idle-to-idle transition."""

    owner: object
    profile: tuple
    signature: Any
    duration_ns: float
    allow_reserved_frontiers: bool
    input_frontiers: tuple[tuple[str, float, float], ...]
    resources: tuple[tuple[str, float, float, int], ...]
    byte_counts: tuple[tuple[str, int], ...]
    events: int
    cancelled_reads: int
    tickets: int
    leases: int
    bulk_waves: int
    slots: tuple[tuple[tuple[float | None, int], ...], ...]
    allocation_offset_ns: float | None
    first_event_offset_ns: float


@dataclass(frozen=True)
class _QuiescentStreamTemplate:
    resources: tuple[tuple[str, float, float, int], ...]
    byte_counts: tuple[tuple[str, int], ...]
    events: int
    tickets: int
    bulk_waves: int
    # Per stack and oldest-first physical slot: release offset and count.
    slots: tuple[tuple[tuple[float | None, int], ...], ...]
    peak_bytes: tuple[int, ...]
    result: PaperMemoryEvent


_STREAM_TEMPLATES: OrderedDict[tuple, _QuiescentStreamTemplate] = OrderedDict()


def _immutable_signature(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable_signature(item) for item in value)
    return False


class _Calendar:
    __slots__ = ("available_ns", "busy_ns", "operations")

    def __init__(self) -> None:
        self.available_ns = 0.0
        self.busy_ns = 0.0
        self.operations = 0


class _BulkWaveTransition:
    """Exact max-plus form of one full-wave consume + full-wave prefetch.

    State is each touched FIFO resource frontier, both in-flight waves' read
    completion times, and the fixed external ready time.  Every operation in
    that fixed sequence is max and addition.  Matrix composition therefore
    performs the same finite-buffer state transition, including lookahead
    reads, rather than replacing it by a bandwidth/compute roofline.
    """

    def __init__(self, memory_tier: str, buffer_tier: str, stacks: int, tile: int, stack_rate: float, latency: float,
                 sram_read: float, sram_write: float, sram_latency: float,
                 host: float, shared: float, memory_duplex: bool,
                 host_duplex: bool, shared_duplex: bool, compute_ns: float) -> None:
        memory_keys = [f"{memory_tier}:{stack}:" + ("read" if memory_duplex else "shared") for stack in range(stacks)]
        array_keys = [f"hbf_array:{stack}:" + ("read" if memory_duplex else "shared")
                      for stack in range(stacks)] if memory_tier == "hbf" else []
        array_credit_keys = [f"hbf_read_credit:{stack}" for stack in range(stacks)] if memory_tier == "hbf" else []
        fill_keys = [f"{buffer_tier}:{stack}:" + ("write" if memory_duplex else "shared") for stack in range(stacks)]
        drain_keys = [f"{buffer_tier}:{stack}:" + ("read" if memory_duplex else "shared") for stack in range(stacks)]
        host_key = "host:" + ("read" if host_duplex else "shared")
        shared_key = "gpu_io:" + ("read" if shared_duplex else "shared")
        self.keys = tuple(dict.fromkeys(memory_keys + array_keys + array_credit_keys + fill_keys + drain_keys + [host_key, shared_key, "gpu:compute"]))
        self.stacks = stacks
        size = len(self.keys) + 2 * stacks + 1
        identity = np.full((size, size), -np.inf)
        np.fill_diagonal(identity, 0.0)
        calendars = {key: identity[index].copy() for index, key in enumerate(self.keys)}
        current = [identity[len(self.keys) + stack].copy() for stack in range(stacks)]
        following = [identity[len(self.keys) + stacks + stack].copy() for stack in range(stacks)]
        ready = identity[-1].copy()
        drain_ns = sram_latency + tile / sram_read
        fill_ns = sram_latency + tile / sram_write

        def maximum(*values: np.ndarray) -> np.ndarray:
            return np.maximum.reduce(values)

        ends = []
        for stack, key in enumerate(drain_keys):
            if memory_tier == "hbf":
                # This SRAM is on the logic die.  Its consumer, rather than
                # array prefetch, owns the physical stack-to-GPU output path.
                start = maximum(ready, current[stack], calendars[key],
                                calendars[memory_keys[stack]] - sram_latency,
                                calendars[host_key] - sram_latency,
                                calendars[shared_key] - sram_latency)
                transfer = start + sram_latency
                calendars[key] = transfer + tile / sram_read
                calendars[memory_keys[stack]] = transfer + tile / stack_rate
                calendars[host_key] = transfer + tile / host
                calendars[shared_key] = transfer + tile / shared
                end = maximum(calendars[key], calendars[memory_keys[stack]],
                              calendars[host_key], calendars[shared_key])
            else:
                end = maximum(ready, current[stack], calendars[key]) + drain_ns
                calendars[key] = end
            ends.append(end)
        compute_end = maximum(ready, calendars["gpu:compute"], *ends) + compute_ns
        calendars["gpu:compute"] = compute_end
        new_wave = []
        for stack, key in enumerate(memory_keys):
            if memory_tier == "hbf":
                # The replacement arrives in logic-die SRAM, never in an
                # undeclared GPU buffer.  Only its local fill waits for the
                # prior slot owner to finish compute.
                array_end = maximum(ready, calendars[array_keys[stack]],
                                    calendars[array_credit_keys[stack]]) + latency
                calendars[array_keys[stack]] = array_end
                fill_start = maximum(compute_end, ready, array_end,
                                     calendars[fill_keys[stack]])
                calendars[array_credit_keys[stack]] = fill_start
                fill_end = fill_start + fill_ns
            else:
                start = maximum(compute_end, ready, calendars[key],
                                calendars[host_key] - latency, calendars[shared_key] - latency)
                transfer = start + latency
                media_end = transfer + tile / stack_rate
                host_end = transfer + tile / host
                shared_end = transfer + tile / shared
                calendars[key] = media_end
                calendars[host_key] = host_end
                calendars[shared_key] = shared_end
                read_end = maximum(media_end, host_end, shared_end)
                fill_end = maximum(read_end, calendars[fill_keys[stack]]) + fill_ns
            calendars[fill_keys[stack]] = fill_end
            new_wave.append(fill_end)
        self.powers = [np.stack([calendars[key] for key in self.keys] + following + new_wave + [ready])]
        # Per transition exactly these services execute.  Busy time and event
        # counts are additive observers, independent of queueing frontiers.
        self.busy: dict[str, tuple[float, int]] = {}

        def busy(key: str, duration: float) -> None:
            old_time, old_count = self.busy.get(key, (0.0, 0))
            self.busy[key] = (old_time + duration, old_count + 1)

        for stack in range(stacks):
            busy(drain_keys[stack], drain_ns)
            busy(memory_keys[stack], (latency if memory_tier != "hbf" else 0.0) + tile / stack_rate)
            if memory_tier == "hbf":
                busy(array_keys[stack], latency)
                busy(array_credit_keys[stack], 0.0)
            busy(host_key, tile / host)
            busy(shared_key, tile / shared)
            busy(fill_keys[stack], fill_ns)
        busy("gpu:compute", compute_ns)

    def advance(self, vector: np.ndarray, count: int) -> np.ndarray:
        bit = 0
        while count:
            if bit == len(self.powers):
                previous = self.powers[-1]
                self.powers.append(np.max(previous[:, :, None] + previous[None, :, :], axis=1))
            if count & 1:
                vector = np.max(self.powers[bit] + vector[None, :], axis=1)
            count >>= 1
            bit += 1
        return vector


@lru_cache(maxsize=128)
def _bulk_wave_transition(memory_tier: str, buffer_tier: str, stacks: int, tile: int, stack_rate: float, latency: float,
                          sram_read: float, sram_write: float, sram_latency: float,
                          host: float, shared: float, memory_duplex: bool,
                          host_duplex: bool, shared_duplex: bool,
                          compute_ns: float) -> _BulkWaveTransition:
    return _BulkWaveTransition(memory_tier, buffer_tier, stacks, tile, stack_rate, latency, sram_read, sram_write,
                               sram_latency, host, shared, memory_duplex, host_duplex,
                               shared_duplex, compute_ns)


class PaperMemorySystem:
    """One GPU's real HBM/HBF/SRAM resources, initialized from the catalog.

    ``hardware`` accepts a PaperHardware object or its dictionary.  Plus
    configurations must have eight independent 40-MB intermediate arenas;
    each HBF stack also has a separate 3.13-MB staging arena, split into two
    finite buffers.  Neither arena spills into host DRAM or hidden HBM.

    Callers should submit jobs in execution order.  A reservation can wait
    for a future dependency; later submissions do not jump ahead of it.
    """

    def __init__(self, hardware: Any, *, assumptions: PaperMemoryAssumptions,
                 retain_trace: bool = False, bulk_wave_acceleration: bool = True,
                 quiescent_stream_reuse: bool = True) -> None:
        if not isinstance(assumptions, PaperMemoryAssumptions):
            raise TypeError("assumptions must be explicit PaperMemoryAssumptions")
        self.assumptions = assumptions
        self.hardware = dict(hardware) if isinstance(hardware, Mapping) else asdict(hardware)
        h = self.hardware
        if h.get("configuration_id") not in {"HBM4", "HBF", "HBF+", "CONV", "CONV+"}:
            raise ValueError("unknown paper memory configuration")
        self.hbm_stacks = h["hbm_stacks"]
        self.hbf_stacks = h["hbf_stacks"]
        for key in ("hbm_stacks", "hbf_stacks", "intermediate_sram_bytes_per_stack",
                    "read_staging_bytes_per_hbf_stack"):
            _bytes(key, h[key], zero=True)
        if self.hbm_stacks + self.hbf_stacks != 8:
            raise ValueError("paper configurations contain eight stacks")
        self.intermediate_storage = h["intermediate_storage"]
        self.intermediate_capacity = h["intermediate_sram_bytes_per_stack"]
        if self.intermediate_storage == "logic_die_SRAM":
            if self.hbm_stacks or self.intermediate_capacity != 40_000_000:
                raise ValueError("plus configurations require 40 MB SRAM per stack and zero HBM")
        elif self.intermediate_storage != "HBM" or not self.hbm_stacks or self.intermediate_capacity:
            raise ValueError("HBM intermediates require actual HBM, without a hidden SRAM tier")
        staging = h["read_staging_bytes_per_hbf_stack"]
        if self.hbf_stacks and staging != 3_130_000:
            raise ValueError("each HBF stack requires the paper's separate 3.13 MB staging capacity")
        if not self.hbf_stacks and staging:
            raise ValueError("HBM4 must not acquire HBF staging")
        for tier in ("hbm", "hbf"):
            present = bool(getattr(self, tier + "_stacks"))
            for direction in ("read", "write"):
                rate = h[tier + "_" + direction + "_gb_s"]
                _positive(tier + "_" + direction, rate, zero=True)
                if present != (rate > 0):
                    raise ValueError("memory bandwidth must match actual stack presence")
        self.staging_slot_bytes = staging // 2
        self.hbf_array_units_per_stack = (h["dies_per_hbf_stack"] * h["planes_per_hbf_die"]
                                          if self.hbf_stacks else 0)
        if self.hbf_stacks and self.hbf_array_units_per_stack <= 0:
            raise ValueError("HBF reads require the published finite die-plane array count")
        # HBM GPU-side read buffers are an explicit, independently named
        # research allocation.  They are not HBF logic-die SRAM and do not
        # contribute to its 40-MB intermediate or 3.13-MB staging capacities.
        hbm_buffer = assumptions.hbm_read_buffer_bytes_per_stack if not self.hbf_stacks else 0
        self._prefetch_tier = "hbf" if self.hbf_stacks else "hbm"
        self._buffer_tier = "staging" if self.hbf_stacks else "hbm_buffer"
        self._buffer_stacks = self.hbf_stacks if self.hbf_stacks else (self.hbm_stacks if hbm_buffer else 0)
        self._read_slot_bytes = (min(self.staging_slot_bytes, self.hbf_array_units_per_stack * 4096)
                                 if self.hbf_stacks else hbm_buffer // 2)
        self._calendars: dict[str, _Calendar] = {}
        self._slots: list[list[tuple[int | None, float]]] = [
            [(None, 0.0), (None, 0.0)] for _ in range(self._buffer_stacks)]
        self._slot_releases = [[0, 0] for _ in range(self._buffer_stacks)]
        self._tickets: dict[int, ReadTicket] = {}
        self._leases: dict[int, IntermediateLease] = {}
        self._release_heap: list[tuple[float, int]] = []
        self._intermediate_live = [0] * self.hbf_stacks
        self._intermediate_peak = [0] * self.hbf_stacks
        self._allocation_frontier_ns = 0.0
        self._next_ticket = 0
        self._next_lease = 0
        self.retain_trace = retain_trace
        self.bulk_wave_acceleration = bulk_wave_acceleration
        self._bulk_advanced_waves = 0
        self.quiescent_stream_reuse = quiescent_stream_reuse
        self._stream_cache_hits = 0
        self._stream_cache_misses = 0
        self.events: list[PaperMemoryEvent] = []
        self.bytes_by_operation: dict[str, int] = {}
        self.event_count = 0
        self.cancelled_reads = 0
        self.staging_peak_bytes = [0] * self.hbf_stacks
        self.hbm_read_buffer_peak_bytes = [0] * (self._buffer_stacks if not self.hbf_stacks else 0)
        self._read_buffer_peak = self.staging_peak_bytes if self.hbf_stacks else self.hbm_read_buffer_peak_bytes
        self._staging_live_bytes = [0] * self._buffer_stacks
        self._snapshot_owner = object()
        self._snapshot_serial = 0
        self._snapshot_min_event_start_ns = math.inf
        self._snapshot_max_slot_tie_ns = -math.inf
        self._snapshot_watches: WeakSet[_SnapshotWatch] = WeakSet()

    def _observe_event_start(self, start_ns: float) -> None:
        self._snapshot_min_event_start_ns = min(self._snapshot_min_event_start_ns, start_ns)
        for watch in self._snapshot_watches:
            watch.first_event_ns = min(watch.first_event_ns, start_ns)

    def _observe_slot_tie(self, free_at_ns: float) -> None:
        self._snapshot_max_slot_tie_ns = max(self._snapshot_max_slot_tie_ns, free_at_ns)
        for watch in self._snapshot_watches:
            watch.last_slot_tie_ns = max(watch.last_slot_tie_ns, free_at_ns)

    def _reserve(self, *, operation: str, tier: str, stack: int | None,
                 size_bytes: int, ready_ns: float, duration_ns: float,
                 resources: Iterable[str], label: str = "") -> PaperMemoryEvent:
        _positive("ready_ns", ready_ns, zero=True)
        _positive("duration_ns", duration_ns, zero=True)
        keys = tuple(dict.fromkeys(resources))
        calendars = [self._calendars.setdefault(key, _Calendar()) for key in keys]
        start = max([ready_ns] + [calendar.available_ns for calendar in calendars])
        end = start + duration_ns
        for calendar in calendars:
            calendar.available_ns = end
            calendar.busy_ns += duration_ns
            calendar.operations += 1
        event = PaperMemoryEvent(operation, tier, stack, size_bytes, ready_ns, start, end, keys, label)
        self._observe_event_start(start)
        self.event_count += 1
        key = tier + ":" + operation
        self.bytes_by_operation[key] = self.bytes_by_operation.get(key, 0) + size_bytes
        if self.retain_trace:
            self.events.append(event)
        return event

    def _stack(self, tier: str, stack: int) -> None:
        count = getattr(self, tier + "_stacks")
        if isinstance(stack, bool) or not isinstance(stack, int) or not 0 <= stack < count:
            raise ValueError(f"{tier} stack does not exist: {stack}")

    def access(self, tier: str, direction: str, size_bytes: int, *, stack: int,
               ready_ns: float = 0.0, label: str = "",
               data_ready_ns: float | None = None) -> PaperMemoryEvent:
        """A paid bulk transfer, contending on stack, host and shared GPU I/O.

        HBM/SRAM use paid cut-through transfers.  NAND reads separately
        prepare a finite array group and drain its data port; NAND writes
        separately pay host ingress and the max(tPROG, Table-I service)
        program budget.  No per-page FTL/erase behavior is invented here.

        ``size_bytes`` is already the caller's chosen service byte count.
        This bandwidth abstraction does not round a NAND write to 4 KiB,
        silently combine logical writes, or add read-modify-write.  A caller
        modeling whole-page programs must supply its separately calculated
        physical program bytes and retain logical bytes as a separate metric.
        """
        if tier not in {"hbm", "hbf", "sram"} or direction not in {"read", "write"}:
            raise ValueError("invalid memory tier or direction")
        _bytes("size_bytes", size_bytes)
        a = self.assumptions
        if data_ready_ns is not None:
            _positive("data_ready_ns", data_ready_ns, zero=True)
        if tier == "sram":
            if self.intermediate_storage != "logic_die_SRAM":
                raise ValueError("this configuration has no intermediate SRAM")
            self._stack("hbf", stack)
            if size_bytes > self.intermediate_capacity:
                raise PaperMemoryCapacityError("one SRAM access exceeds the stack-local arena")
            rate = getattr(a, "sram_" + direction + "_gb_s_per_stack")
            latency = a.sram_access_ns
        else:
            self._stack(tier, stack)
            rate = self.hardware[tier + "_" + direction + "_gb_s"] / getattr(self, tier + "_stacks")
            latency = (self.hardware["page_read_ns" if direction == "read" else "page_program_ns"]
                       if tier == "hbf" else a.hbm_access_ns)
        if tier == "hbf" and direction == "read":
            return self._hbf_read_access(size_bytes, stack, ready_ns=ready_ns,
                                          data_ready_ns=ready_ns if data_ready_ns is None else data_ready_ns,
                                          label=label)
        if tier == "hbf" and direction == "write":
            return self._hbf_program_access(size_bytes, stack,
                                             ready_ns=max(ready_ns, data_ready_ns or 0.0), label=label)
        if data_ready_ns is not None:
            ready_ns = max(ready_ns, data_ready_ns)
        host = getattr(a, "host_" + direction + "_gb_s")
        resources = [f"{tier}:{stack}:" + (direction if a.memory_full_duplex else "shared"),
                     "host:" + (direction if a.host_full_duplex else "shared"),
                     "gpu_io:" + (direction if a.shared_io_full_duplex else "shared")]
        # A shared bus owns only its own byte-service window, not the longer
        # service window of a slower stack.  Holding it for the stack's full
        # latency would accidentally serialize eight independent stacks.
        # This is a declared cut-through bulk-transfer abstraction: media
        # startup precedes transfer; link and stack byte service overlap.
        _positive("ready_ns", ready_ns, zero=True)
        calendars = [self._calendars.setdefault(key, _Calendar()) for key in resources]
        start = max(ready_ns, calendars[0].available_ns,
                    calendars[1].available_ns - latency, calendars[2].available_ns - latency)
        transfer_start = start + latency
        durations = (size_bytes / rate, size_bytes / host, size_bytes / a.shared_io_gb_s)
        ends = [transfer_start + duration for duration in durations]
        for index, (calendar, duration, end) in enumerate(zip(calendars, durations, ends)):
            calendar.available_ns = end
            calendar.busy_ns += duration + (latency if index == 0 else 0.0)
            calendar.operations += 1
        event = PaperMemoryEvent(direction, tier, stack, size_bytes, ready_ns, start, max(ends),
                                 tuple(resources), label)
        self._observe_event_start(start)
        self.event_count += 1
        key = tier + ":" + direction
        self.bytes_by_operation[key] = self.bytes_by_operation.get(key, 0) + size_bytes
        if self.retain_trace:
            self.events.append(event)
        return event

    def _hbf_program_access(self, size_bytes: int, stack: int, *,
                            ready_ns: float, label: str) -> PaperMemoryEvent:
        """Paid host ingress followed by the Table-I program-service budget.

        Table write bandwidth already includes array programming throughput;
        program time is max(tPROG, bytes/rate), never tPROG + bytes/rate.
        A small page still pays its complete tPROG floor.  The ingress ports
        are independent paid resources and can overlap an earlier program.
        This is a declared bulk-service envelope, not per-plane FTL/GC or
        a claim about the paper's unspecified program command granularity.
        The caller is responsible for bounded import buffers and page pads.
        """
        _positive("ready_ns", ready_ns, zero=True)
        a = self.assumptions
        keys = ("host:" + ("write" if a.host_full_duplex else "shared"),
                "gpu_io:" + ("write" if a.shared_io_full_duplex else "shared"))
        calendars = [self._calendars.setdefault(key, _Calendar()) for key in keys]
        ingress_start = max(ready_ns, *(calendar.available_ns for calendar in calendars))
        ingress_ends = []
        for calendar, rate in zip(calendars, (a.host_write_gb_s, a.shared_io_gb_s)):
            duration = size_bytes / rate
            calendar.available_ns = ingress_start + duration
            calendar.busy_ns += duration
            calendar.operations += 1
            ingress_ends.append(calendar.available_ns)
        program_key = (f"hbf:{stack}:write" if a.memory_full_duplex else f"hbf_array:{stack}:shared")
        program = self._calendars.setdefault(program_key, _Calendar())
        program_start = max(*ingress_ends, program.available_ns)
        rate = self.hardware["hbf_write_gb_s"] / self.hbf_stacks
        duration = max(self.hardware["page_program_ns"], size_bytes / rate)
        program.available_ns = program_start + duration
        program.busy_ns += duration
        program.operations += 1
        event = PaperMemoryEvent("write", "hbf", stack, size_bytes, ready_ns,
                                 ingress_start, program.available_ns, (*keys, program_key), label)
        self._observe_event_start(ingress_start)
        self.event_count += 1
        self.bytes_by_operation["hbf:write"] = self.bytes_by_operation.get("hbf:write", 0) + size_bytes
        if self.retain_trace:
            self.events.append(event)
        return event

    def _hbf_read_access(self, size_bytes: int, stack: int, *, ready_ns: float,
                         data_ready_ns: float, label: str,
                         to_staging: bool = False) -> PaperMemoryEvent:
        """Finite array-group preparation followed by an independent data port.

        One collective burst uses at most the published dies*planes 4-KiB
        read resources; a short final burst conservatively reserves the
        whole group.  Bounded streaming tiles never exceed one such group.
        The group is free after tREAD, so it may prepare the next admitted
        read while this read drains through the data port.  A separate
        handoff credit prevents new preparation before the previous tile's
        data phase begins: at most one prepared/pending tile can stand
        behind the transferring tile, even when GPU consumption is delayed.
        Logic-die double-buffer credits independently bound their residence.
        A prefetch stops on that die and never reserves the GPU output path;
        a direct access instead transfers immediately to its external caller.
        """
        _positive("ready_ns", ready_ns, zero=True)
        if size_bytes > self.hbf_array_units_per_stack * 4096:
            raise PaperMemoryCapacityError("direct HBF read exceeds one finite array group; use bounded stream_read_compute")
        a = self.assumptions
        array_key = f"hbf_array:{stack}:" + ("read" if a.memory_full_duplex else "shared")
        port_key = f"hbf:{stack}:" + ("read" if a.memory_full_duplex else "shared")
        host_key = "host:" + ("read" if a.host_full_duplex else "shared")
        shared_key = "gpu_io:" + ("read" if a.shared_io_full_duplex else "shared")
        array = self._calendars.setdefault(array_key, _Calendar())
        credit_key = f"hbf_read_credit:{stack}"
        credit = self._calendars.setdefault(credit_key, _Calendar())
        bursts = 1
        array_start = max(ready_ns, array.available_ns, credit.available_ns)
        array_duration = bursts * self.hardware["page_read_ns"]
        array_end = array_start + array_duration
        array.available_ns = array_end
        array.busy_ns += array_duration
        array.operations += bursts
        if to_staging:
            # A finite local slot and its actual write port must be ready
            # before the prepared array result can leave the handoff stage.
            fill_key = "staging:" + str(stack) + ":" + ("write" if a.memory_full_duplex else "shared")
            fill_frontier = self._calendars.get(fill_key)
            credit.available_ns = max(array_end, data_ready_ns,
                                      fill_frontier.available_ns if fill_frontier is not None else 0.)
            credit.operations += 1
            event = PaperMemoryEvent("read", "hbf", stack, size_bytes, ready_ns,
                                     array_start, array_end, (array_key, credit_key), label)
            self._observe_event_start(array_start)
            self.event_count += 1
            self.bytes_by_operation["hbf:read"] = self.bytes_by_operation.get("hbf:read", 0) + size_bytes
            if self.retain_trace:
                self.events.append(event)
            return event
        keys = (port_key, host_key, shared_key)
        calendars = [self._calendars.setdefault(key, _Calendar()) for key in keys]
        transfer = max(array_end, data_ready_ns, *(calendar.available_ns for calendar in calendars))
        credit.available_ns = transfer
        credit.operations += 1
        rates = (self.hardware["hbf_read_gb_s"] / self.hbf_stacks,
                 a.host_read_gb_s, a.shared_io_gb_s)
        ends = []
        for calendar, rate in zip(calendars, rates):
            duration = size_bytes / rate
            calendar.available_ns = transfer + duration
            calendar.busy_ns += duration
            calendar.operations += 1
            ends.append(calendar.available_ns)
        event = PaperMemoryEvent("read", "hbf", stack, size_bytes, ready_ns,
                                 array_start, max(ends), (array_key, credit_key, *keys), label)
        self._observe_event_start(array_start)
        self.event_count += 1
        self.bytes_by_operation["hbf:read"] = self.bytes_by_operation.get("hbf:read", 0) + size_bytes
        if self.retain_trace:
            self.events.append(event)
        return event

    def prefetch(self, stack: int, size_bytes: int, *, ready_ns: float = 0.0,
                 label: str = "") -> ReadTicket:
        self._stack(self._prefetch_tier, stack)
        if not self._buffer_stacks:
            raise ValueError("HBM prefetch requires an explicit finite HBM read-buffer capacity")
        _bytes("size_bytes", size_bytes)
        _positive("ready_ns", ready_ns, zero=True)
        if size_bytes > self._read_slot_bytes:
            raise PaperMemoryCapacityError("prefetch exceeds one half of stack-local staging capacity")
        candidates = [(free_at, slot) for slot, (owner, free_at) in enumerate(self._slots[stack]) if owner is None]
        if not candidates:
            raise PaperBufferCreditError("both staging buffers await consumption; prefetch has no credit")
        if len(candidates) == 2 and candidates[0][0] == candidates[1][0]:
            self._observe_slot_tie(candidates[0][0])
        free_at, slot = min(candidates)
        if self.hbf_stacks:
            read = self._hbf_read_access(size_bytes, stack, ready_ns=ready_ns,
                                         data_ready_ns=max(ready_ns, free_at), label=label,
                                         to_staging=True)
        else:
            read = self.access(self._prefetch_tier, "read", size_bytes, stack=stack,
                               ready_ns=ready_ns, data_ready_ns=max(ready_ns, free_at), label=label)
        # Memory data lands in its explicitly allocated read buffer.  The
        # SRAM write port is paid for HBM GPU buffers as well as HBF staging.
        fill = self._reserve(operation=self._buffer_tier + "_fill", tier=self._buffer_tier, stack=stack,
                             size_bytes=size_bytes, ready_ns=max(read.end_ns, free_at),
                             duration_ns=self.assumptions.sram_access_ns + size_bytes /
                             self.assumptions.sram_write_gb_s_per_stack,
                             resources=(f"{self._buffer_tier}:{stack}:" +
                                        ("write" if self.assumptions.memory_full_duplex else "shared"),), label=label)
        ticket = ReadTicket(self._next_ticket, stack, slot, size_bytes, fill)
        self._next_ticket += 1
        self._tickets[ticket.ticket_id] = ticket
        self._slots[stack][slot] = (ticket.ticket_id, fill.end_ns)
        self._staging_live_bytes[stack] += size_bytes
        self._read_buffer_peak[stack] = max(self._read_buffer_peak[stack], self._staging_live_bytes[stack])
        return ticket

    def _ticket(self, ticket: ReadTicket) -> ReadTicket:
        if self._tickets.get(ticket.ticket_id) is not ticket:
            raise ValueError("ticket does not belong to this memory system")
        if ticket.cancelled or ticket.consumed_at_ns is not None:
            raise ValueError("ticket has already been consumed or cancelled")
        return ticket

    def consume_many(self, tickets: Iterable[ReadTicket], compute_ns: float, *,
                     ready_ns: float = 0.0, label: str = "") -> PaperMemoryEvent:
        """GPU compute waits for all reads and SRAM reads; slots live to its end."""
        items = tuple(tickets)
        if not items or len({ticket.ticket_id for ticket in items}) != len(items):
            raise ValueError("consume requires distinct, non-empty read tickets")
        for ticket in items:
            self._ticket(ticket)
        _positive("compute_ns", compute_ns, zero=True)
        _positive("ready_ns", ready_ns, zero=True)
        read_ends = []
        for ticket in items:
            if self.hbf_stacks:
                drain = self._drain_logic_staging(ticket, max(ready_ns, ticket.read.end_ns), label)
            else:
                drain = self._reserve(operation=self._buffer_tier + "_read", tier=self._buffer_tier, stack=ticket.stack,
                                      size_bytes=ticket.size_bytes, ready_ns=max(ready_ns, ticket.read.end_ns),
                                      duration_ns=self.assumptions.sram_access_ns + ticket.size_bytes /
                                      self.assumptions.sram_read_gb_s_per_stack,
                                      resources=(f"{self._buffer_tier}:{ticket.stack}:" +
                                                 ("read" if self.assumptions.memory_full_duplex else "shared"),), label=label)
            read_ends.append(drain.end_ns)
        event = self.compute(compute_ns, ready_ns=max([ready_ns] + read_ends), label=label)
        for ticket in items:
            ticket.consumed_at_ns = event.start_ns
            ticket.released_at_ns = event.end_ns
            self._slots[ticket.stack][ticket.slot] = (None, event.end_ns)
            self._slot_releases[ticket.stack][ticket.slot] += 1
            self._staging_live_bytes[ticket.stack] -= ticket.size_bytes
            del self._tickets[ticket.ticket_id]
        return event

    def _drain_logic_staging(self, ticket: ReadTicket, ready_ns: float,
                             label: str) -> PaperMemoryEvent:
        """Paid cut-through logic-SRAM-to-GPU transfer, with no GPU tile cache.

        SRAM startup precedes byte service; the SRAM read, stack output port,
        aggregate host link and shared GPU ingress serve the same bytes in
        cut-through overlap.  Every resource retains its own paid frontier.
        Compute waits for the slowest completion.  The logic-die slot stays
        leased through compute, conservatively avoiding any undeclared GPU
        storage that could otherwise release it at transfer completion.
        """
        a = self.assumptions
        keys = (f"staging:{ticket.stack}:" + ("read" if a.memory_full_duplex else "shared"),
                f"hbf:{ticket.stack}:" + ("read" if a.memory_full_duplex else "shared"),
                "host:" + ("read" if a.host_full_duplex else "shared"),
                "gpu_io:" + ("read" if a.shared_io_full_duplex else "shared"))
        calendars = [self._calendars.setdefault(key, _Calendar()) for key in keys]
        start = max(ready_ns, calendars[0].available_ns,
                    *(calendar.available_ns - a.sram_access_ns for calendar in calendars[1:]))
        transfer = start + a.sram_access_ns
        rates = (a.sram_read_gb_s_per_stack, self.hardware["hbf_read_gb_s"] / self.hbf_stacks,
                 a.host_read_gb_s, a.shared_io_gb_s)
        ends = []
        for index, (calendar, rate) in enumerate(zip(calendars, rates)):
            duration = ticket.size_bytes / rate
            calendar.available_ns = transfer + duration
            calendar.busy_ns += duration + (a.sram_access_ns if index == 0 else 0.)
            calendar.operations += 1
            ends.append(calendar.available_ns)
        event = PaperMemoryEvent("staging_read", "staging", ticket.stack, ticket.size_bytes,
                                 ready_ns, start, max(ends), keys, label)
        self._observe_event_start(start)
        self.event_count += 1
        key = "staging:staging_read"
        self.bytes_by_operation[key] = self.bytes_by_operation.get(key, 0) + ticket.size_bytes
        if self.retain_trace:
            self.events.append(event)
        return event

    def consume(self, ticket: ReadTicket, compute_ns: float, *, ready_ns: float = 0.0,
                label: str = "") -> PaperMemoryEvent:
        return self.consume_many((ticket,), compute_ns, ready_ns=ready_ns, label=label)

    def compute(self, compute_ns: float, *, ready_ns: float = 0.0,
                label: str = "") -> PaperMemoryEvent:
        return self._reserve(operation="compute", tier="gpu", stack=None, size_bytes=0,
                             ready_ns=ready_ns, duration_ns=compute_ns, resources=("gpu:compute",), label=label)

    def cancel_prefetch(self, ticket: ReadTicket) -> float:
        """Discard an unconsumed read; in-flight resource service is still paid."""
        self._ticket(ticket)
        ticket.cancelled = True
        ticket.released_at_ns = ticket.read.end_ns
        self._slots[ticket.stack][ticket.slot] = (None, ticket.read.end_ns)
        self._slot_releases[ticket.stack][ticket.slot] += 1
        self._staging_live_bytes[ticket.stack] -= ticket.size_bytes
        del self._tickets[ticket.ticket_id]
        self.cancelled_reads += 1
        return ticket.read.end_ns

    def stream_read_compute(self, total_bytes: int, compute_ns: float, *,
                            ready_ns: float = 0.0, label: str = "") -> PaperMemoryEvent:
        """Execute a finite-buffer stream, reusing only a proven idle template.

        A template is made by the ordinary priming, all waves and tail path.
        Reuse translates that complete same-work result only when every
        resource used by the stream and every read buffer are already idle.
        Unrelated full-duplex write calendars are never moved or ignored by
        a shared-port qualification.  Active intermediate leases are neither
        released nor duplicated: this stream uses the separate read buffers.
        """
        _bytes("total_bytes", total_bytes)
        _positive("compute_ns", compute_ns, zero=True)
        _positive("ready_ns", ready_ns, zero=True)
        if self.retain_trace or not self.bulk_wave_acceleration or not self.quiescent_stream_reuse:
            return self._stream_read_compute(total_bytes, compute_ns, ready_ns=ready_ns, label=label)
        h = self.hardware
        key = (h["configuration_id"], self.hbm_stacks, self.hbf_stacks,
               h["hbm_read_gb_s"], h["hbf_read_gb_s"], h["page_read_ns"],
               self.hbf_array_units_per_stack,
               self.assumptions, total_bytes, compute_ns)
        template = _STREAM_TEMPLATES.get(key)
        if template is not None:
            if self._stream_is_idle(tuple(item[0] for item in template.resources), ready_ns):
                _STREAM_TEMPLATES.move_to_end(key)
                self._stream_cache_hits += 1
                return self._reuse_stream_template(template, ready_ns, label)
            return self._stream_read_compute(total_bytes, compute_ns, ready_ns=ready_ns, label=label)
        keys = self._stream_resource_keys(total_bytes)
        if not self._stream_is_idle(keys, ready_ns):
            return self._stream_read_compute(total_bytes, compute_ns, ready_ns=ready_ns, label=label)
        self._stream_cache_misses += 1
        before = {name: (value.available_ns, value.busy_ns, value.operations)
                  for name in keys if (value := self._calendars.get(name)) is not None}
        bytes_before = dict(self.bytes_by_operation)
        events, tickets, bulk = self.event_count, self._next_ticket, self._bulk_advanced_waves
        slot_order = [sorted(range(2), key=lambda slot: (stack[slot][1], slot)) for stack in self._slots]
        release_before = [list(stack) for stack in self._slot_releases]
        result = self._stream_read_compute(total_bytes, compute_ns, ready_ns=ready_ns, label=label)
        resources = []
        for name in keys:
            calendar = self._calendars.get(name)
            _, old_busy, old_operations = before.get(name, (0.0, 0.0, 0))
            if calendar is not None and calendar.operations > old_operations:
                resources.append((name, calendar.available_ns - ready_ns,
                                  calendar.busy_ns - old_busy, calendar.operations - old_operations))
        slots = []
        for stack, order in enumerate(slot_order):
            slots.append(tuple((self._slots[stack][slot][1] - ready_ns
                                if self._slot_releases[stack][slot] > release_before[stack][slot] else None,
                                self._slot_releases[stack][slot] - release_before[stack][slot])
                               for slot in order))
        # Intrinsic live buffering depends on the first two admitted waves,
        # not a source instance's historical high-water mark.
        peak = []
        if self._buffer_stacks:
            capacity = self._buffer_stacks * self._read_slot_bytes
            first, second = min(total_bytes, capacity), min(max(total_bytes - capacity, 0), capacity)
            qa, ra = divmod(first, self._buffer_stacks)
            qb, rb = divmod(second, self._buffer_stacks)
            peak = [qa + (stack < ra) + qb + (stack < rb) for stack in range(self._buffer_stacks)]
        template = _QuiescentStreamTemplate(
            tuple(resources), tuple((name, value - bytes_before.get(name, 0))
                                    for name, value in self.bytes_by_operation.items()
                                    if value != bytes_before.get(name, 0)),
            self.event_count - events, self._next_ticket - tickets, self._bulk_advanced_waves - bulk,
            tuple(slots), tuple(peak),
            PaperMemoryEvent(result.operation, result.tier, result.stack, result.size_bytes,
                             result.ready_ns - ready_ns, result.start_ns - ready_ns,
                             result.end_ns - ready_ns, result.resources))
        _STREAM_TEMPLATES[key] = template
        if len(_STREAM_TEMPLATES) > 256:
            _STREAM_TEMPLATES.popitem(last=False)
        return result

    def _stream_resource_keys(self, total_bytes: int) -> tuple[str, ...]:
        a = self.assumptions
        count = self._buffer_stacks or self.hbm_stacks
        active = min(count, total_bytes)
        tier = self._prefetch_tier
        keys = [f"{tier}:{stack}:" + ("read" if a.memory_full_duplex else "shared")
                for stack in range(active)]
        if self.hbf_stacks:
            keys.extend(f"hbf_array:{stack}:" + ("read" if a.memory_full_duplex else "shared")
                        for stack in range(active))
            keys.extend(f"hbf_read_credit:{stack}" for stack in range(active))
        if self._buffer_stacks:
            for operation in ("read", "write"):
                keys.extend(f"{self._buffer_tier}:{stack}:" +
                            (operation if a.memory_full_duplex else "shared") for stack in range(active))
        keys.extend(("host:" + ("read" if a.host_full_duplex else "shared"),
                     "gpu_io:" + ("read" if a.shared_io_full_duplex else "shared"), "gpu:compute"))
        return tuple(dict.fromkeys(keys))

    def _stream_is_idle(self, resource_keys: tuple[str, ...], ready_ns: float) -> bool:
        return (not self._tickets
                and not any(owner is not None or free_at > ready_ns
                            for stack in self._slots for owner, free_at in stack)
                and all(self._calendars[key].available_ns <= ready_ns
                        for key in resource_keys if key in self._calendars))

    def _reuse_stream_template(self, template: _QuiescentStreamTemplate,
                               ready_ns: float, label: str) -> PaperMemoryEvent:
        for key, available, busy, operations in template.resources:
            calendar = self._calendars.setdefault(key, _Calendar())
            calendar.available_ns = ready_ns + available
            calendar.busy_ns += busy
            calendar.operations += operations
        for key, value in template.byte_counts:
            self.bytes_by_operation[key] = self.bytes_by_operation.get(key, 0) + value
        # The compute observer exists even when its byte count is zero.
        self.bytes_by_operation.setdefault("gpu:compute", 0)
        self.event_count += template.events
        self._next_ticket += template.tickets
        self._bulk_advanced_waves += template.bulk_waves
        for stack, values in enumerate(template.slots):
            if any(releases for _, releases in values) and self._slots[stack][0][1] == self._slots[stack][1][1]:
                self._observe_slot_tie(self._slots[stack][0][1])
            order = sorted(range(2), key=lambda slot: (self._slots[stack][slot][1], slot))
            for slot, (offset, releases) in zip(order, values):
                if releases:
                    self._slots[stack][slot] = (None, ready_ns + offset)
                    self._slot_releases[stack][slot] += releases
            self._read_buffer_peak[stack] = max(self._read_buffer_peak[stack], template.peak_bytes[stack])
        self._observe_event_start(ready_ns)
        result = template.result
        return PaperMemoryEvent(result.operation, result.tier, result.stack, result.size_bytes,
                                ready_ns + result.ready_ns, ready_ns + result.start_ns,
                                ready_ns + result.end_ns, result.resources, label)

    def _stream_read_compute(self, total_bytes: int, compute_ns: float, *,
                             ready_ns: float = 0.0, label: str = "") -> PaperMemoryEvent:
        """Striped two-wave prefetch pipeline; work is linear in bounded tiles.

        Each wave has at most one tile per stack.  At most two waves are in
        flight.  Compute is on the GPU, shared by all stacks, and proportional
        to each wave's actual byte count.  Final uneven tiles are paid exactly.
        """
        _bytes("total_bytes", total_bytes)
        _positive("compute_ns", compute_ns, zero=True)
        _positive("ready_ns", ready_ns, zero=True)
        if not self._buffer_stacks:
            events = []
            quotient, remainder = divmod(total_bytes, self.hbm_stacks)
            for stack in range(self.hbm_stacks):
                size = quotient + (stack < remainder)
                if size:
                    events.append(self.access("hbm", "read", size, stack=stack, ready_ns=ready_ns, label=label))
            return self.compute(compute_ns, ready_ns=max(event.end_ns for event in events), label=label)
        remaining = total_bytes
        waves: list[tuple[list[ReadTicket], int]] = []

        def issue() -> None:
            nonlocal remaining
            wave_bytes = min(remaining, self._buffer_stacks * self._read_slot_bytes)
            q, r = divmod(wave_bytes, self._buffer_stacks)
            tickets = [self.prefetch(stack, q + (stack < r), ready_ns=ready_ns, label=label)
                       for stack in range(self._buffer_stacks) if q + (stack < r)]
            waves.append((tickets, wave_bytes))
            remaining -= wave_bytes

        issue()
        if remaining:
            issue()
        # Only identical complete waves qualify.  Two real waves establish
        # finite-buffer ownership.  Keep the final two waves and any partial
        # tail on the ordinary event path.  Retained per-tile traces always
        # use that path, since aggregate powers cannot emit omitted events.
        full_wave_bytes = self._buffer_stacks * self._read_slot_bytes
        full_waves = total_bytes // full_wave_bytes
        bulk_count = full_waves - 2
        if self.bulk_wave_acceleration and not self.retain_trace and bulk_count >= 32:
            waves = self._advance_full_waves(waves, bulk_count,
                                             compute_ns * full_wave_bytes / total_bytes,
                                             ready_ns=ready_ns, label=label)
            remaining -= bulk_count * full_wave_bytes
        result = None
        while waves:
            tickets, wave_bytes = waves.pop(0)  # the queue is bounded to two waves
            result = self.consume_many(tickets, compute_ns * wave_bytes / total_bytes,
                                       ready_ns=ready_ns, label=label)
            if remaining:
                issue()
        assert result is not None
        return result

    def _advance_full_waves(self, waves: list[tuple[list[ReadTicket], int]], count: int,
                            compute_ns: float, *, ready_ns: float,
                            label: str) -> list[tuple[list[ReadTicket], int]]:
        """Advance an already admitted two-wave ring without relaxing credit.

        The slot identity rotates once per wave.  It is not an extra buffer:
        the newly issued wave takes exactly the slot of the consumed wave,
        and its earliest readiness is that wave's compute completion.  The
        matrix explicitly includes both lookahead read completion vectors.
        """
        stacks = self._buffer_stacks
        tile = self._read_slot_bytes
        if len(waves) != 2 or any(len(items) != stacks or size != stacks * tile
                                  or any(ticket.size_bytes != tile for ticket in items)
                                  for items, size in waves):
            raise ValueError("bulk transition requires exactly two homogeneous admitted full waves")
        a = self.assumptions
        transition = _bulk_wave_transition(
            self._prefetch_tier, self._buffer_tier, stacks, tile,
            self.hardware[self._prefetch_tier + "_read_gb_s"] / stacks,
            self.hardware["page_read_ns"] if self.hbf_stacks else a.hbm_access_ns, a.sram_read_gb_s_per_stack,
            a.sram_write_gb_s_per_stack, a.sram_access_ns, a.host_read_gb_s,
            a.shared_io_gb_s, a.memory_full_duplex, a.host_full_duplex,
            a.shared_io_full_duplex, compute_ns)
        vector = np.array([self._calendars.get(key, _Calendar()).available_ns for key in transition.keys]
                          + [ticket.read.end_ns for ticket in waves[0][0]]
                          + [ticket.read.end_ns for ticket in waves[1][0]] + [ready_ns])
        vector = transition.advance(vector, count)
        for index, key in enumerate(transition.keys):
            calendar = self._calendars.setdefault(key, _Calendar())
            calendar.available_ns = float(vector[index])
            busy, operations = transition.busy[key]
            calendar.busy_ns += busy * count
            calendar.operations += operations * count
        original_slots = [[ticket.slot for ticket in items] for items, _ in waves]
        for stack in range(stacks):
            self._slot_releases[stack][original_slots[0][stack]] += (count + 1) // 2
            self._slot_releases[stack][original_slots[1][stack]] += count // 2
        for items, _ in waves:
            for ticket in items:
                del self._tickets[ticket.ticket_id]
        first_new_id = self._next_ticket
        self._next_ticket += count * stacks
        result = []
        offset = len(transition.keys)
        for wave in range(2):
            items = []
            for stack in range(stacks):
                end = float(vector[offset + wave * stacks + stack])
                slot = original_slots[(count + wave) % 2][stack]
                ticket_id = first_new_id + (count - 2 + wave) * stacks + stack
                fill_duration = a.sram_access_ns + tile / a.sram_write_gb_s_per_stack
                read = PaperMemoryEvent(self._buffer_tier + "_fill", self._buffer_tier, stack, tile,
                                        end - fill_duration, end - fill_duration, end,
                                        (f"{self._buffer_tier}:{stack}:" +
                                         ("write" if a.memory_full_duplex else "shared"),), label)
                ticket = ReadTicket(ticket_id, stack, slot, tile, read)
                items.append(ticket)
                self._tickets[ticket_id] = ticket
                self._slots[stack][slot] = (ticket_id, end)
            result.append((items, stacks * tile))
        self.event_count += (3 * stacks + 1) * count
        for key in (self._prefetch_tier + ":read", self._buffer_tier + ":" + self._buffer_tier + "_fill",
                    self._buffer_tier + ":" + self._buffer_tier + "_read"):
            self.bytes_by_operation[key] = self.bytes_by_operation.get(key, 0) + count * stacks * tile
        self.bytes_by_operation.setdefault("gpu:compute", 0)
        self._bulk_advanced_waves += count
        return result

    def allocate_intermediate(self, owner: str, size_bytes: int, *, stack: int,
                              ready_ns: float = 0.0) -> IntermediateLease:
        """Admit a real stack-local lifetime; callers allocate in time order."""
        if self.intermediate_storage != "logic_die_SRAM":
            raise ValueError("intermediate SRAM is absent; use the actual HBM intermediate tier")
        self._stack("hbf", stack)
        _bytes("size_bytes", size_bytes)
        _positive("ready_ns", ready_ns, zero=True)
        if not isinstance(owner, str) or not owner:
            raise ValueError("intermediate allocation requires an owner")
        if ready_ns < self._allocation_frontier_ns:
            raise ValueError("intermediate allocation must be submitted in nondecreasing time order")
        self._allocation_frontier_ns = ready_ns
        while self._release_heap and self._release_heap[0][0] <= ready_ns:
            _, allocation = heapq.heappop(self._release_heap)
            old = self._leases.pop(allocation)
            self._intermediate_live[old.stack] -= old.size_bytes
        if self._intermediate_live[stack] + size_bytes > self.intermediate_capacity:
            raise PaperMemoryCapacityError(f"stack {stack} intermediate SRAM capacity exceeded")
        lease = IntermediateLease(self._next_lease, owner, stack, size_bytes, ready_ns)
        self._next_lease += 1
        self._leases[lease.allocation_id] = lease
        self._intermediate_live[stack] += size_bytes
        self._intermediate_peak[stack] = max(self._intermediate_peak[stack], self._intermediate_live[stack])
        return lease

    def release_intermediate(self, lease: IntermediateLease, *, at_ns: float) -> None:
        if self._leases.get(lease.allocation_id) is not lease or lease.released_at_ns is not None:
            raise ValueError("allocation is unknown or already released")
        _positive("at_ns", at_ns, zero=True)
        if at_ns < lease.allocated_at_ns:
            raise ValueError("an intermediate cannot be released before allocation")
        lease.released_at_ns = at_ns
        heapq.heappush(self._release_heap, (at_ns, lease.allocation_id))

    def snapshot_counters(self, *, signature: Any = None) -> PaperMemorySnapshot:
        """Capture a period boundary without changing scheduling or counters.

        The signature must describe the caller's immutable work program,
        including tensor sizes, placement, operator/compute service and
        dependencies.  Counter equality cannot establish future program
        equality, so repetition requires the caller to supply that same
        explicit signature again.  No signature means no compression.
        Consecutive period replay uses the latest snapshot.  Whole-program
        transitions also support an outer live snapshot while inner layer
        periods execute.  Weak observers retain two timing extrema, never
        an event history; retired snapshots remove their observers.
        """
        if not _immutable_signature(signature):
            raise ValueError("period work signature must contain immutable scalar/tuple values")
        self._snapshot_serial += 1
        self._snapshot_min_event_start_ns = math.inf
        self._snapshot_max_slot_tie_ns = -math.inf
        watch = _SnapshotWatch()
        self._snapshot_watches.add(watch)
        return PaperMemorySnapshot(
            owner=self._snapshot_owner, serial=self._snapshot_serial, signature=signature,
            calendars=tuple((key, value.available_ns, value.busy_ns, value.operations)
                            for key, value in self._calendars.items()),
            byte_counts=tuple(self.bytes_by_operation.items()), event_count=self.event_count,
            cancelled_reads=self.cancelled_reads, next_ticket=self._next_ticket,
            next_lease=self._next_lease, bulk_advanced_waves=self._bulk_advanced_waves,
            pending_tickets=len(self._tickets),
            slots=tuple(tuple(stack) for stack in self._slots),
            slot_releases=tuple(tuple(stack) for stack in self._slot_releases),
            leases=tuple((lease.allocation_id, lease.released_at_ns) for lease in self._leases.values()),
            staging_live=tuple(self._staging_live_bytes), watch=watch)

    def _can_quiescent_period(self, before: PaperMemorySnapshot, start_ns: float,
                              end_ns: float, *, signature: Any = None,
                              allow_slot_permutation: bool = False,
                              allow_outer_snapshot: bool = False,
                              allow_reserved_frontiers: bool = False) -> bool:
        """Check the idle-to-idle proof without changing any system state.

        There is no approximation or hidden scheduling fallback.  False
        leaves state unchanged and tells the caller to execute normally.
        All resources (including a future PD reservation), pending read
        tickets and SRAM lifetimes must be clear at *both* boundaries.

        Free-buffer ordering must also be invariant.  If an odd read count
        swaps the two physical slot identities, use two actual periods as
        the repeated unit rather than resetting/adding a buffer.  Used
        resource frontiers move by the elapsed period; unused resources
        retain their actual frontier.  Trace retention refuses compression.
        """
        _positive("start_ns", start_ns, zero=True)
        _positive("end_ns", end_ns, zero=True)
        if end_ns <= start_ns:
            raise ValueError("period end must follow its start")
        if not isinstance(before, PaperMemorySnapshot):
            return False
        if allow_slot_permutation and before.watch.last_slot_tie_ns > start_ns:
            # Initial free-slot ties may be mapped oldest-first.  A later
            # tie uses the physical slot index and breaks arbitrary label
            # permutation, so generic transition reuse must refuse it.
            return False
        if (self.retain_trace or before.owner is not self._snapshot_owner
                or before.watch not in self._snapshot_watches
                or (not allow_outer_snapshot and before.serial != self._snapshot_serial)
                or before.watch.first_event_ns < start_ns or signature is None
                or not _immutable_signature(signature) or before.signature != signature):
            return False
        if before.pending_tickets or self._tickets or any(before.staging_live) or any(self._staging_live_bytes):
            return False
        if any(release is None or release > start_ns for _, release in before.leases):
            return False
        if any(lease.released_at_ns is None or lease.released_at_ns > end_ns
               for lease in self._leases.values()):
            return False
        previous = {key: (available, busy, operations)
                    for key, available, busy, operations in before.calendars}
        if not allow_reserved_frontiers and any(available > start_ns for available, _, _ in previous.values()):
            return False
        if any(calendar.available_ns > end_ns for key, calendar in self._calendars.items()
               if not allow_reserved_frontiers or calendar.operations > previous.get(key, (0.0, 0.0, 0))[2]):
            return False
        if len(before.slots) != len(self._slots):
            return False
        slot_deltas = []
        for stack, current in enumerate(self._slots):
            old = before.slots[stack]
            if any(owner is not None or free_at > start_ns for owner, free_at in old):
                return False
            if any(owner is not None or free_at > end_ns for owner, free_at in current):
                return False
            releases = [self._slot_releases[stack][slot] - before.slot_releases[stack][slot]
                        for slot in range(2)]
            if any(count < 0 for count in releases):
                return False
            if not allow_slot_permutation and any(releases) and sorted(range(2), key=lambda slot: (old[slot][1], slot)) != sorted(
                    range(2), key=lambda slot: (current[slot][1], slot)):
                return False
            slot_deltas.append(releases)
        counters = ((self.event_count, before.event_count),
                    (self.cancelled_reads, before.cancelled_reads),
                    (self._next_ticket, before.next_ticket), (self._next_lease, before.next_lease),
                    (self._bulk_advanced_waves, before.bulk_advanced_waves))
        if any(after < old for after, old in counters):
            return False
        byte_before = dict(before.byte_counts)
        if any(self.bytes_by_operation.get(key, 0) < value for key, value in byte_before.items()):
            return False
        for key, calendar in self._calendars.items():
            old_available, old_busy, old_operations = previous.get(key, (0.0, 0.0, 0))
            if calendar.operations < old_operations or calendar.busy_ns < old_busy:
                return False
            # A resource with no service in this period is not part of its
            # affine translation.  Never move a foreign/old idle frontier.
            if calendar.operations == old_operations and calendar.available_ns != old_available:
                return False
        return True

    def can_repeat_quiescent_period(self, before: PaperMemorySnapshot, start_ns: float,
                                    end_ns: float, *, signature: Any = None) -> bool:
        """Side-effect-free qualification of an identical slot-order period."""
        return self._can_quiescent_period(before, start_ns, end_ns, signature=signature)

    def _transition_profile(self) -> tuple:
        fields = ("configuration_id", "hbm_stacks", "hbf_stacks", "total_capacity_gb",
                  "hbm_read_gb_s", "hbm_write_gb_s", "hbf_read_gb_s", "hbf_write_gb_s",
                  "dies_per_hbf_stack", "planes_per_hbf_die", "page_read_ns", "page_program_ns",
                  "intermediate_storage", "intermediate_sram_bytes_per_stack",
                  "read_staging_bytes_per_hbf_stack")
        return (tuple(self.hardware[field] for field in fields), self.assumptions,
                self._prefetch_tier, self._buffer_tier, self._buffer_stacks,
                self._read_slot_bytes, self.hbf_array_units_per_stack)

    def make_quiescent_transition(self, before: PaperMemorySnapshot, start_ns: float,
                                  end_ns: float, *, signature: Any = None,
                                  allow_reserved_frontiers: bool = False) -> PaperQuiescentTransition | None:
        """Record an actually executed whole-program transition, without mutation.

        This same-instance template can be reused non-consecutively.  All
        source resources/lifetimes must be idle at both boundaries.  Physical
        buffer labels may rotate, but only when no index-dependent in-period
        free-slot tie occurred.  Initial oldest-first slot labels are mapped
        explicitly on application.  Retained traces are never compressed.

        Opting into reserved frontiers records the exact positive input FIFO
        state of each used resource.  Reuse must match that state within
        floating subtraction roundoff; future service is never discarded.
        Source/target read buffers and SRAM lifetimes still must be clear.
        """
        if not isinstance(allow_reserved_frontiers, bool):
            raise ValueError("allow_reserved_frontiers must be boolean")
        if not self._can_quiescent_period(before, start_ns, end_ns, signature=signature,
                                          allow_slot_permutation=True, allow_outer_snapshot=True,
                                          allow_reserved_frontiers=allow_reserved_frontiers):
            return None
        previous = {key: (available, busy, operations)
                    for key, available, busy, operations in before.calendars}
        resources = []
        frontiers = []
        for key, calendar in self._calendars.items():
            available, busy, operations = previous.get(key, (0.0, 0.0, 0))
            if calendar.operations > operations:
                resources.append((key, calendar.available_ns - start_ns,
                                  calendar.busy_ns - busy, calendar.operations - operations))
                offset = max(0.0, available - start_ns)
                frontiers.append((key, offset, 4 * (math.ulp(available) + math.ulp(start_ns)) if offset else 0.0))
        byte_before = dict(before.byte_counts)
        slots = []
        for stack, old in enumerate(before.slots):
            order = sorted(range(2), key=lambda slot: (old[slot][1], slot))
            values = []
            for slot in order:
                releases = self._slot_releases[stack][slot] - before.slot_releases[stack][slot]
                values.append((self._slots[stack][slot][1] - start_ns if releases else None, releases))
            slots.append(tuple(values))
        return PaperQuiescentTransition(
            self._snapshot_owner, self._transition_profile(), signature, end_ns - start_ns,
            allow_reserved_frontiers, tuple(frontiers),
            tuple(resources), tuple((key, value - byte_before.get(key, 0))
                                    for key, value in self.bytes_by_operation.items()
                                    if value != byte_before.get(key, 0)),
            self.event_count - before.event_count, self.cancelled_reads - before.cancelled_reads,
            self._next_ticket - before.next_ticket, self._next_lease - before.next_lease,
            self._bulk_advanced_waves - before.bulk_advanced_waves, tuple(slots),
            self._allocation_frontier_ns - start_ns if self._next_lease > before.next_lease else None,
            before.watch.first_event_ns - start_ns)

    def can_apply_quiescent_transition(self, template: PaperQuiescentTransition, ready_ns: float, *,
                                       signature: Any = None) -> bool:
        """Read-only whole-step preflight; no resource or expired lease is drained."""
        _positive("ready_ns", ready_ns, zero=True)
        if (self.retain_trace or not isinstance(template, PaperQuiescentTransition)
                or template.owner is not self._snapshot_owner or signature is None
                or not _immutable_signature(signature) or template.signature != signature
                or template.profile != self._transition_profile()):
            return False
        if self._tickets or any(self._staging_live_bytes):
            return False
        if any(owner is not None or free_at > ready_ns for stack in self._slots for owner, free_at in stack):
            return False
        if any(lease.released_at_ns is None or lease.released_at_ns > ready_ns for lease in self._leases.values()):
            return False
        for key, expected, source_error in template.input_frontiers:
            calendar = self._calendars.get(key)
            available = calendar.available_ns if calendar is not None else 0.0
            if not expected:
                if available > ready_ns:
                    return False
            else:
                # Never turn a positive reservation into an idle resource,
                # even when absolute timestamps have a coarse float ULP.
                if available <= ready_ns:
                    return False
                observed = available - ready_ns
                roundoff = source_error + 4 * (math.ulp(available) + math.ulp(ready_ns))
                if abs(observed - expected) > roundoff:
                    return False
        return True

    def apply_quiescent_transition(self, template: PaperQuiescentTransition, ready_ns: float, *,
                                   signature: Any = None) -> float | None:
        """Replay a qualified same-instance transition or return None unchanged.

        A multi-GPU caller first checks every actor with ``can_apply...``.
        Source and destination are the same actor, whose historical SRAM and
        read-buffer peaks are monotone; replay cannot import another GPU's
        peak or lower an already observed peak.  Unused PD/write resources
        are intentionally left at their actual current frontier.
        """
        if not self.can_apply_quiescent_transition(template, ready_ns, signature=signature):
            return None
        for key, available, busy, operations in template.resources:
            calendar = self._calendars.setdefault(key, _Calendar())
            calendar.available_ns = ready_ns + available
            calendar.busy_ns += busy
            calendar.operations += operations
        for key, value in template.byte_counts:
            self.bytes_by_operation[key] = self.bytes_by_operation.get(key, 0) + value
        self.event_count += template.events
        self.cancelled_reads += template.cancelled_reads
        self._next_ticket += template.tickets
        self._next_lease += template.leases
        self._bulk_advanced_waves += template.bulk_waves
        for stack, values in enumerate(template.slots):
            if any(releases for _, releases in values) and self._slots[stack][0][1] == self._slots[stack][1][1]:
                self._observe_slot_tie(self._slots[stack][0][1])
            order = sorted(range(2), key=lambda slot: (self._slots[stack][slot][1], slot))
            for slot, (offset, releases) in zip(order, values):
                if releases:
                    self._slots[stack][slot] = (None, ready_ns + offset)
                    self._slot_releases[stack][slot] += releases
        if template.allocation_offset_ns is not None:
            self._allocation_frontier_ns = ready_ns + template.allocation_offset_ns
        self._leases.clear()
        self._release_heap.clear()
        self._intermediate_live[:] = [0] * self.hbf_stacks
        self._observe_event_start(ready_ns + template.first_event_offset_ns)
        return ready_ns + template.duration_ns

    def repeat_quiescent_period(self, before: PaperMemorySnapshot, start_ns: float,
                                end_ns: float, repetitions: int, *,
                                signature: Any = None) -> bool:
        """Copy a qualified period ``repetitions`` additional times.

        ``can_repeat_quiescent_period`` performs exactly the same proof, so
        a multi-GPU caller may qualify all same-instance snapshots first,
        then replay each GPU without copying live objects.  Callers must not
        schedule intervening work between qualification and replay.
        """
        _bytes("repetitions", repetitions, zero=True)
        if not self.can_repeat_quiescent_period(before, start_ns, end_ns, signature=signature):
            return False
        previous = {key: (available, busy, operations)
                    for key, available, busy, operations in before.calendars}
        byte_before = dict(before.byte_counts)
        slot_deltas = [[self._slot_releases[stack][slot] - before.slot_releases[stack][slot]
                        for slot in range(2)] for stack in range(len(self._slots))]
        observed_tie = before.watch.last_slot_tie_ns
        shift = (end_ns - start_ns) * repetitions
        for key, calendar in self._calendars.items():
            _, busy, operations = previous.get(key, (0.0, 0.0, 0))
            if calendar.operations > operations:
                calendar.available_ns += shift
                calendar.busy_ns += (calendar.busy_ns - busy) * repetitions
                calendar.operations += (calendar.operations - operations) * repetitions
        for key, value in list(self.bytes_by_operation.items()):
            self.bytes_by_operation[key] += (value - byte_before.get(key, 0)) * repetitions
        self.event_count += (self.event_count - before.event_count) * repetitions
        self.cancelled_reads += (self.cancelled_reads - before.cancelled_reads) * repetitions
        self._next_ticket += (self._next_ticket - before.next_ticket) * repetitions
        self._next_lease += (self._next_lease - before.next_lease) * repetitions
        self._bulk_advanced_waves += (self._bulk_advanced_waves - before.bulk_advanced_waves) * repetitions
        for stack, releases in enumerate(slot_deltas):
            for slot, count in enumerate(releases):
                if count:
                    self._slots[stack][slot] = (None, self._slots[stack][slot][1] + shift)
                    self._slot_releases[stack][slot] += count * repetitions
            if (repetitions and any(releases)
                    and self._slots[stack][0][1] == self._slots[stack][1][1]):
                self._observe_slot_tie(self._slots[stack][0][1] - (end_ns - start_ns))
        if repetitions and observed_tie > start_ns:
            self._observe_slot_tie(observed_tie + shift)
        if self._next_lease > before.next_lease:
            self._allocation_frontier_ns += shift
        # The proof above guarantees that all extant leases have retired.
        # Cleaning them does not reduce lifetime peaks or refund capacity.
        self._leases.clear()
        self._release_heap.clear()
        self._intermediate_live[:] = [0] * self.hbf_stacks
        self._snapshot_serial += 1
        return True

    def summary(self) -> dict[str, Any]:
        return {"configuration_id": self.hardware["configuration_id"],
                "abstraction": "bounded_tile_bandwidth_events_not_OCP_physical_transactions",
                "transfer_byte_semantics": "caller_supplied_service_bytes_no_implicit_page_rounding_or_RMW",
                "research_assumptions": asdict(self.assumptions),
                "events": self.event_count, "bytes_by_operation": dict(self.bytes_by_operation),
                "end_ns": max((item.available_ns for item in self._calendars.values()), default=0.0),
                "staging_capacity_bytes_per_stack": self.staging_slot_bytes * 2,
                "hbf_array_units_per_stack": self.hbf_array_units_per_stack,
                "hbf_read_tile_bytes": self._read_slot_bytes if self.hbf_stacks else 0,
                "hbf_read_pipeline": "finite_die_plane_group_preparation_then_independent_table_bandwidth_data_port",
                "hbf_array_group_policy": "4KiB_pages; one_full_group_per_bounded_burst; short_burst_reserves_group; finite_prepare_and_local_fill_handoff; at_most_two_logic_die_staging_tickets_per_stack",
                "hbf_read_path": "array_prepare_then_local_logic_die_staging_fill; consumption_pays_cut_through_SRAM_read_stack_output_host_GPU_link; slot_held_through_compute; no_GPU_tile_cache",
                "hbf_program_service_policy": "paid_host_ingress_then_max_tPROG_or_TableI_program_bandwidth; caller_page_padding; no_per_plane_FTL_or_GC",
                "staging_peak_bytes_per_stack": list(self.staging_peak_bytes),
                "hbm_read_buffer_capacity_bytes_per_stack": self._read_slot_bytes * 2 if not self.hbf_stacks else 0,
                "hbm_read_buffer_peak_bytes_per_stack": list(self.hbm_read_buffer_peak_bytes),
                "intermediate_capacity_bytes_per_stack": self.intermediate_capacity,
                "intermediate_peak_bytes_per_stack": list(self._intermediate_peak),
                "cancelled_reads": self.cancelled_reads,
                "resources": {key: {"available_ns": value.available_ns, "busy_ns": value.busy_ns,
                                    "operations": value.operations} for key, value in self._calendars.items()}}
