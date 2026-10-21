"""Finite HOST DRAM streaming for disaggregated prefill/decode KV import.

Each GPU has an explicit two-slot host arena. Network receive, HOST writes,
HOST reads and actual GPU-memory writes are paid, ordered stages. No HBM
staging is introduced for an HBF-only configuration. Identical full chunks
may use a max-plus power of the very same FIFO resource/slot transition;
this is event compression, not a bandwidth forecast or an unbounded buffer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import ceil, isfinite
from typing import Any

import numpy as np

from .paper_memory import PaperMemorySystem, _Calendar

MIB = 1024 * 1024
PAGE_BYTES = 4096


def _integer(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(name + " must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + " must be " + ("positive" if positive else "non-negative"))


@dataclass(frozen=True)
class PaperPDHostAssumptions:
    buffer_bytes: int = 64 * MIB
    read_gb_s: float = 200.
    write_gb_s: float = 200.
    basis: str = ("Local PD research assumption: each GPU owns 64 MiB HOST DRAM split "
                  "into two 32 MiB slots, with separate 200 GB/s HOST read/write ports. "
                  "These capacities and rates are not disclosed by the paper.")

    def __post_init__(self):
        _integer("buffer_bytes", self.buffer_bytes, minimum=2)
        if self.buffer_bytes % 2:
            raise ValueError("HOST DRAM arena must divide into two equal slots")
        _number("read_gb_s", self.read_gb_s, positive=True)
        _number("write_gb_s", self.write_gb_s, positive=True)
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("HOST DRAM parameters require an explicit research basis")


@dataclass(frozen=True)
class PDChunkTrace:
    slot: int
    logical_bytes: int
    physical_program_bytes: int
    network_start_ns: float
    network_end_ns: float
    host_write_start_ns: float
    host_write_end_ns: float
    host_read_start_ns: float
    host_read_end_ns: float
    end_ns: float


@dataclass(frozen=True)
class PDImportResult:
    end_ns: float
    logical_bytes: int
    physical_program_bytes: int
    transfer_bytes: int
    host_buffer_peak_bytes: int
    chunk_count: int
    compressed_chunks: int
    explicit_chunks: int
    counters: dict[str, Any]


@dataclass
class _HostPort:
    available_ns: float = 0.
    busy_ns: float = 0.
    operations: int = 0
    bytes: int = 0


class _ImportPairTransition:
    """Compiled pair of bounded imports, preserving physical slot identity."""

    def __init__(self, *, tier, stacks, chunk, first_slot, nic_key, nic_latency,
                 nic_rate, host_read, host_write, stack_rate, memory_latency,
                 memory_host_rate, shared_rate, memory_duplex, host_duplex,
                 shared_duplex):
        self.tier, self.stacks, self.chunk = tier, stacks, chunk
        if tier == "hbf":
            quotient, remainder = divmod((chunk+PAGE_BYTES-1)//PAGE_BYTES, stacks)
            sizes = [(quotient + (stack < remainder))*PAGE_BYTES for stack in range(stacks)]
        else:
            quotient, remainder = divmod(chunk, stacks)
            sizes = [quotient + (stack < remainder) for stack in range(stacks)]
        active = [stack for stack, size in enumerate(sizes) if size]
        media_keys = {stack: (f"hbf_array:{stack}:shared" if tier == "hbf" and not memory_duplex
                              else f"{tier}:{stack}:" + ("write" if memory_duplex else "shared"))
                      for stack in active}
        host_key = "host:" + ("write" if host_duplex else "shared")
        shared_key = "gpu_io:" + ("write" if shared_duplex else "shared")
        self.memory_keys = tuple(dict.fromkeys(list(media_keys.values()) + [host_key, shared_key]))
        self.keys = (("network", "host_write", "host_read", "slot0", "slot1", "start0", "start1")
                     + tuple("memory:" + key for key in self.memory_keys) + ("ready",))
        identity = np.full((len(self.keys), len(self.keys)), -np.inf)
        np.fill_diagonal(identity, 0.)
        state = {key: identity[index].copy() for index, key in enumerate(self.keys)}
        memory_busy: dict[str, tuple[float, int]] = {}

        def maximum(*items):
            return np.maximum.reduce(items)

        def add_busy(key, duration):
            old_time, old_count = memory_busy.get(key, (0., 0))
            memory_busy[key] = old_time + duration, old_count + 1

        self.physical_per_chunk = sum(sizes) if tier == "hbf" else 0
        self.memory_bytes_per_chunk = sum(sizes)
        self.memory_events_per_chunk = len(active)
        for slot in (first_slot, 1-first_slot):
            network_start = maximum(state["ready"], state[f"slot{slot}"], state["network"])
            state[f"start{slot}"] = network_start
            state["network"] = network_start + nic_latency + chunk / nic_rate
            state["host_write"] = maximum(state["network"], state["host_write"]) + chunk / host_write
            state["host_read"] = maximum(state["host_write"], state["host_read"]) + chunk / host_read
            ends = []
            for stack, byte_count in enumerate(sizes):
                if not byte_count:
                    continue
                mk = "memory:" + media_keys[stack]
                hk, sk = "memory:" + host_key, "memory:" + shared_key
                if tier == "hbf":
                    ingress = maximum(state["host_read"], state[hk], state[sk])
                    state[hk] = ingress + byte_count / memory_host_rate
                    state[sk] = ingress + byte_count / shared_rate
                    program_start = maximum(state[hk], state[sk], state[mk])
                    program_duration = max(memory_latency, byte_count / stack_rate)
                    state[mk] = program_start + program_duration
                    ends.append(state[mk])
                    add_busy(media_keys[stack], program_duration)
                    add_busy(host_key, byte_count / memory_host_rate)
                    add_busy(shared_key, byte_count / shared_rate)
                    continue
                options = [state["host_read"], state[mk],
                           state[hk]-memory_latency, state[sk]-memory_latency]
                transfer = maximum(*options) + memory_latency
                state[mk] = transfer + byte_count / stack_rate
                state[hk] = transfer + byte_count / memory_host_rate
                state[sk] = transfer + byte_count / shared_rate
                ends.extend((state[mk], state[hk], state[sk]))
                add_busy(media_keys[stack], memory_latency + byte_count / stack_rate)
                add_busy(host_key, byte_count / memory_host_rate)
                add_busy(shared_key, byte_count / shared_rate)
            state[f"slot{slot}"] = maximum(*ends)
        self.powers = [np.stack([state[key] for key in self.keys])]
        self.memory_busy = memory_busy
        self.network_busy = 2 * (nic_latency + chunk / nic_rate)
        self.host_write_busy = 2 * chunk / host_write
        self.host_read_busy = 2 * chunk / host_read
        self.nic_key = nic_key

    def advance(self, vector, count):
        bit = 0
        while count:
            if bit == len(self.powers):
                last = self.powers[-1]
                self.powers.append(np.max(last[:, :, None] + last[None, :, :], axis=1))
            if count & 1:
                vector = np.max(self.powers[bit] + vector[None, :], axis=1)
            count >>= 1
            bit += 1
        return vector


@lru_cache(maxsize=128)
def _pair_transition(tier, stacks, chunk, first_slot, nic_key, nic_latency, nic_rate,
                     host_read, host_write, stack_rate, memory_latency,
                     memory_host_rate, shared_rate, memory_duplex, host_duplex, shared_duplex):
    return _ImportPairTransition(tier=tier, stacks=stacks, chunk=chunk, first_slot=first_slot,
        nic_key=nic_key, nic_latency=nic_latency, nic_rate=nic_rate,
        host_read=host_read, host_write=host_write, stack_rate=stack_rate,
        memory_latency=memory_latency, memory_host_rate=memory_host_rate,
        shared_rate=shared_rate, memory_duplex=memory_duplex,
        host_duplex=host_duplex, shared_duplex=shared_duplex)


class PaperPDImporter:
    """Persistent per-GPU HOST DRAM arena sharing actual NIC/memory calendars.

    The small private-memory-calendar adapter in ``_advance_pairs`` mirrors
    ``PaperMemorySystem.access(..., direction='write')`` only. The test suite
    compares this adapter to ordinary real writes, including half-duplex
    array/link resources. Read and GPU compute frontiers are never reset.
    """

    def __init__(self, network, memory: PaperMemorySystem, rank: int, *,
                 assumptions: PaperPDHostAssumptions | None = None,
                 accelerate: bool = True, retain_trace: bool = False):
        if not isinstance(memory, PaperMemorySystem):
            raise TypeError("memory must be a private PaperMemorySystem")
        _integer("rank", rank)
        if not isinstance(accelerate, bool) or not isinstance(retain_trace, bool):
            raise ValueError("acceleration and trace controls must be boolean")
        self.network, self.memory, self.rank = network, memory, rank
        self.assumptions = assumptions or PaperPDHostAssumptions()
        if not isinstance(self.assumptions, PaperPDHostAssumptions):
            raise TypeError("assumptions must be PaperPDHostAssumptions")
        self.accelerate, self.retain_trace = accelerate, retain_trace
        self.slot_bytes = self.assumptions.buffer_bytes // 2
        self.slot_free_ns = [0., 0.]
        self.slot_start_ns = [0., 0.]
        self.slot_sizes = [0, 0]
        self.slot_uses = [0, 0]
        self.next_slot = 0
        self.host_write, self.host_read = _HostPort(), _HostPort()
        self.buffer_peak_bytes = 0
        self.chunks = self.compressed_chunks = self.explicit_chunks = 0
        self.trace: list[PDChunkTrace] = []

    def chunk_bytes(self, tier):
        if tier not in {"hbm", "hbf"}:
            raise ValueError("PD destination must be hbm or hbf")
        count = getattr(self.memory, tier + "_stacks")
        if not count:
            raise ValueError("PD destination memory does not exist")
        if tier == "hbf":
            h = self.memory.hardware
            page_group = count * h["dies_per_hbf_stack"] * h["planes_per_hbf_die"] * PAGE_BYTES
            return min(self.slot_bytes, page_group)
        return self.slot_bytes

    def _one(self, logical_bytes, now, tier, label):
        slot = self.next_slot
        self.next_slot = 1-slot
        key, _, _ = self.network.pd_receive_parameters(self.rank)
        ready = max(now, self.slot_free_ns[slot])
        network_start = max(ready, self.network.calendars.get(key, 0.))
        network_end = self.network.receive_pd(self.rank, logical_bytes, ready)
        write_start = max(network_end, self.host_write.available_ns)
        write_end = write_start + logical_bytes / self.assumptions.write_gb_s
        self.host_write.available_ns = write_end
        self.host_write.busy_ns += logical_bytes / self.assumptions.write_gb_s
        self.host_write.operations += 1
        self.host_write.bytes += logical_bytes
        read_start = max(write_end, self.host_read.available_ns)
        read_end = read_start + logical_bytes / self.assumptions.read_gb_s
        self.host_read.available_ns = read_end
        self.host_read.busy_ns += logical_bytes / self.assumptions.read_gb_s
        self.host_read.operations += 1
        self.host_read.bytes += logical_bytes
        stacks = getattr(self.memory, tier + "_stacks")
        if tier == "hbf":
            quotient, remainder = divmod((logical_bytes+PAGE_BYTES-1)//PAGE_BYTES, stacks)
        else:
            quotient, remainder = divmod(logical_bytes, stacks)
        ends = []
        program = 0
        for stack in range(stacks):
            size = quotient + (stack < remainder)
            if not size:
                continue
            if tier == "hbf":
                size *= PAGE_BYTES
                program += size
            ends.append(self.memory.access(tier, "write", size, stack=stack,
                                           ready_ns=read_end, label=label).end_ns)
        end = max(ends)
        other = 1-slot
        overlap = (self.slot_sizes[other] if self.slot_start_ns[other] <= network_start
                   < self.slot_free_ns[other] else 0)
        self.buffer_peak_bytes = max(self.buffer_peak_bytes, logical_bytes + overlap)
        self.slot_free_ns[slot] = end
        self.slot_start_ns[slot] = network_start
        self.slot_sizes[slot] = logical_bytes
        self.slot_uses[slot] += 1
        self.chunks += 1
        self.explicit_chunks += 1
        if self.retain_trace:
            self.trace.append(PDChunkTrace(slot, logical_bytes, program, network_start, network_end,
                write_start, write_end, read_start, read_end, end))
        return end

    def _advance_pairs(self, pairs, chunk, now, tier):
        memory, network, a = self.memory, self.network, self.memory.assumptions
        nic_key, nic_latency, nic_rate = network.pd_receive_parameters(self.rank)
        stacks = getattr(memory, tier + "_stacks")
        transition = _pair_transition(tier, stacks, chunk, self.next_slot,
            nic_key, nic_latency, nic_rate, self.assumptions.read_gb_s,
            self.assumptions.write_gb_s, memory.hardware[tier + "_write_gb_s"] / stacks,
            memory.hardware["page_program_ns"] if tier == "hbf" else a.hbm_access_ns,
            a.host_write_gb_s, a.shared_io_gb_s, a.memory_full_duplex,
            a.host_full_duplex, a.shared_io_full_duplex)
        values = {"network": network.calendars.get(nic_key, 0.),
            "host_write": self.host_write.available_ns, "host_read": self.host_read.available_ns,
            "slot0": self.slot_free_ns[0], "slot1": self.slot_free_ns[1],
            "start0": self.slot_start_ns[0], "start1": self.slot_start_ns[1], "ready": now}
        for key in transition.memory_keys:
            calendar = memory._calendars.get(key)
            values["memory:" + key] = calendar.available_ns if calendar is not None else 0.
        vector = np.array([values[key] for key in transition.keys])
        # Arbitrary queued frontiers are already inputs of the exact map, so
        # no explicit "warmup chunks" are needed. Inspect only the first pair
        # to retain the old/new slot overlap high-water mark exactly.
        first = transition.advance(vector, 1)
        first_values = dict(zip(transition.keys, first))
        first_slot, other_slot = self.next_slot, 1-self.next_slot
        first_start = float(first_values[f"start{first_slot}"])
        second_start = float(first_values[f"start{other_slot}"])
        old_overlap = (self.slot_sizes[other_slot]
            if self.slot_start_ns[other_slot] <= first_start < self.slot_free_ns[other_slot] else 0)
        pair_overlap = chunk if second_start < first_values[f"slot{first_slot}"] else 0
        self.buffer_peak_bytes = max(self.buffer_peak_bytes, chunk+old_overlap, chunk+pair_overlap)
        if pairs > 1:
            # Once both old slots have been replaced, the third network RX
            # starts at the second RX completion or a later slot release.
            # The second chunk still has positive HOST/write service left;
            # hence two full chunks overlap, without permitting a third.
            self.buffer_peak_bytes = max(self.buffer_peak_bytes, 2*chunk)
        output = transition.advance(first, pairs-1)
        frontiers = dict(zip(transition.keys, output))
        network.calendars[nic_key] = float(frontiers["network"])
        network.busy[nic_key] = network.busy.get(nic_key, 0.) + pairs * transition.network_busy
        network.bytes += 2 * pairs * chunk
        network.operations += 2 * pairs
        for key in transition.memory_keys:
            calendar = memory._calendars.setdefault(key, _Calendar())
            calendar.available_ns = float(frontiers["memory:" + key])
            busy, operations = transition.memory_busy[key]
            calendar.busy_ns += pairs * busy
            calendar.operations += pairs * operations
        self.host_write.available_ns = float(frontiers["host_write"])
        self.host_write.busy_ns += pairs * transition.host_write_busy
        self.host_write.operations += 2 * pairs
        self.host_write.bytes += 2 * pairs * chunk
        self.host_read.available_ns = float(frontiers["host_read"])
        self.host_read.busy_ns += pairs * transition.host_read_busy
        self.host_read.operations += 2 * pairs
        self.host_read.bytes += 2 * pairs * chunk
        for slot in range(2):
            self.slot_free_ns[slot] = float(frontiers[f"slot{slot}"])
            self.slot_start_ns[slot] = float(frontiers[f"start{slot}"])
            self.slot_sizes[slot] = chunk
            self.slot_uses[slot] += pairs
        memory.bytes_by_operation[tier + ":write"] = memory.bytes_by_operation.get(tier + ":write", 0) + (
            2 * pairs * transition.memory_bytes_per_chunk)
        memory.event_count += 2 * pairs * transition.memory_events_per_chunk
        self.chunks += 2 * pairs
        self.compressed_chunks += 2 * pairs
        return max(self.slot_free_ns)

    def import_kv(self, logical_bytes: int, *, now_ns: float = 0., tier: str = "hbf",
                  label: str = "pd-import") -> PDImportResult:
        _integer("logical_bytes", logical_bytes, minimum=1)
        _number("now_ns", now_ns)
        chunk = self.chunk_bytes(tier)
        full, tail = divmod(logical_bytes, chunk)
        before_program = self.memory.bytes_by_operation.get("hbf:write", 0)
        before_memory_write = self.memory.bytes_by_operation.get(tier + ":write", 0)
        before_memory_events = self.memory.event_count
        before_network = self.network.bytes, self.network.operations
        before_chunks = self.chunks, self.compressed_chunks, self.explicit_chunks
        before_host_write, before_host_read = asdict(self.host_write), asdict(self.host_read)
        end = now_ns
        pairs = full//2
        if (self.accelerate and not self.retain_trace and not self.memory.retain_trace
                and pairs):
            end = max(end, self._advance_pairs(pairs, chunk, now_ns, tier))
            full -= 2*pairs
        for _ in range(full):
            end = max(end, self._one(chunk, now_ns, tier, label))
        if tail:
            end = max(end, self._one(tail, now_ns, tier, label))
        return PDImportResult(end, logical_bytes,
            self.memory.bytes_by_operation.get("hbf:write", 0)-before_program,
            self.network.bytes-before_network[0], self.buffer_peak_bytes,
            self.chunks-before_chunks[0], self.compressed_chunks-before_chunks[1],
            self.explicit_chunks-before_chunks[2], {
                "network_operations": self.network.operations-before_network[1],
                "memory_write_bytes": self.memory.bytes_by_operation.get(tier + ":write", 0)-before_memory_write,
                "memory_events": self.memory.event_count-before_memory_events,
                "host_write_bytes": self.host_write.bytes-before_host_write["bytes"],
                "host_read_bytes": self.host_read.bytes-before_host_read["bytes"],
                "host_write_busy_ns": self.host_write.busy_ns-before_host_write["busy_ns"],
                "host_read_busy_ns": self.host_read.busy_ns-before_host_read["busy_ns"],
                "host_write_operations": self.host_write.operations-before_host_write["operations"],
                "host_read_operations": self.host_read.operations-before_host_read["operations"]})

    def summary(self):
        return {"rank": self.rank, "research_assumptions": asdict(self.assumptions),
            "host_buffer_capacity_bytes": self.assumptions.buffer_bytes,
            "host_buffer_slot_bytes": self.slot_bytes,
            "host_buffer_peak_bytes": self.buffer_peak_bytes,
            "host_write": asdict(self.host_write), "host_read": asdict(self.host_read),
            "slot_free_ns": list(self.slot_free_ns), "slot_start_ns": list(self.slot_start_ns),
            "slot_sizes": list(self.slot_sizes), "slot_uses": list(self.slot_uses),
            "next_slot": self.next_slot, "chunks": self.chunks,
            "compressed_chunks": self.compressed_chunks, "explicit_chunks": self.explicit_chunks,
            "path": "IB_receive_then_HOST_write_then_HOST_read_then_actual_GPU_memory_write",
            "compression_semantics": "identical full-chunk FIFO max-plus transitions including both slot "
                "frontiers from the first complete pair; odd full chunk and partial tail use real events; "
                "old/new slot overlap is accounted exactly; any retained trace disables compression",
            "hidden_HBM_staging_bytes": 0}
