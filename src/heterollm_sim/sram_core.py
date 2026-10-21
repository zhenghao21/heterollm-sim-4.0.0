"""Native ordinary SRAM port service on the shared physical timeline.

This is neither DRAM nor compute-in-memory. A contiguous access stripes its
bursts over finite, shared read/write ports. Equal-share port bandwidths
enforce the aggregate directional and total ceilings. The port pipeline
latency precedes service; it does not serialize every byte on a fake row or
page cycle. An access is coalesced into at most ``ports`` reservations, so a
large workspace does not allocate one event for every 64-byte burst.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from .memory_transfer import ResourceTimeline, summarize_batch
from .memory_types import AccessRequest, AddressMapping, Operation, SramConfig, StageTiming, TransactionResult


class SramCore:
    """Finite-capacity, pipelined SRAM using the native transaction contract."""

    def __init__(self, config: SramConfig, timeline: Optional[ResourceTimeline] = None,
                 *, capture_details: bool = True) -> None:
        if not isinstance(config, SramConfig):
            raise TypeError("config must be SramConfig")
        if not isinstance(capture_details, bool):
            raise TypeError("capture_details must be a boolean")
        self.config = config
        self.timeline = timeline or ResourceTimeline()
        self.capture_details = capture_details
        self._inflight: list[float] = []
        self._acceptance_ns = 0.0

    def _validate(self, request: AccessRequest) -> None:
        if not isinstance(request, AccessRequest):
            raise TypeError("request must be AccessRequest")
        if request.operation is Operation.ERASE:
            raise ValueError("SRAM does not support ERASE operations")
        if request.address + request.byte_count > self.config.capacity_bytes:
            raise ValueError("SRAM access exceeds capacity_bytes")

    def submit(self, request: AccessRequest) -> TransactionResult:
        depth = len(self.timeline._owned_transactions)
        try:
            return self._submit_live(request)
        finally:
            self.timeline._finish_core_accounting(depth)

    def _submit_live(self, request: AccessRequest) -> TransactionResult:
        self._validate(request)
        c, timeline = self.config, self.timeline
        arrival = max(request.arrival_ns, self._acceptance_ns)
        inflight = [end for end in self._inflight if end > arrival]
        if len(inflight) >= c.max_outstanding_requests:
            arrival = min(inflight)
            inflight = [end for end in inflight if end > arrival]
        read = request.operation is Operation.READ
        latency = c.read_latency_ns if read else c.write_latency_ns
        eligible = arrival + latency
        if not math.isfinite(eligible):
            raise ValueError("SRAM request exceeds finite timing range")
        first = request.address // c.burst_bytes
        stop = (request.address + request.byte_count + c.burst_bytes - 1) // c.burst_bytes
        count = stop - first
        prefix = str(c.metadata.get("port_resource_prefix", "sram:port"))
        rate = c.directional_bandwidth_gb_s(request.operation) / c.ports
        # Plan first: invalid resource calendars or overflow cannot partially
        # publish a successful prefix. Aliases may fold authored ports, but
        # cannot grant additional undeclared lanes to one physical port.
        plans, pending, pending_busy = [], {}, {}
        for offset in range(min(count, c.ports)):
            port = (first + offset) % c.ports
            bursts = 1 + (count - offset - 1) // c.ports
            amount = bursts * c.burst_bytes
            resource = timeline.canonical_resource_id(f"{prefix}:{port}")
            lanes = timeline.lane_available.get(resource)
            if lanes is not None and len(lanes) != 1:
                raise ValueError("one SRAM port must have exactly one physical service lane")
            ready = pending.get(resource, timeline.available_ns(resource))
            start = max(eligible, ready)
            try:
                duration = amount / rate
            except OverflowError as exc:
                raise ValueError("SRAM request exceeds finite timing range") from exc
            end = start + duration
            busy = pending_busy.get(resource, timeline.busy_ns.get(resource, 0.)) + duration
            if ready < 0 or not all(math.isfinite(value) for value in (ready, start, duration, end, busy)):
                raise ValueError("SRAM request exceeds finite timing range")
            pending[resource], pending_busy[resource] = end, busy
            plans.append((resource, port, amount, start, end))
        before = timeline._core_metrics_snapshot(max_intervals=c.max_expanded_segments if self.capture_details else 0)
        stages, mappings = [], []
        completion = eligible
        if self.capture_details:
            stages.append(StageTiming("SRAM_READ_LATENCY" if read else "SRAM_WRITE_LATENCY", arrival, eligible))
        for resource, port, amount, start, end in plans:
            transfer = timeline.transfer(resource, amount, start, rate, direction=request.operation.value)
            completion = max(completion, transfer.end_ns)
            if self.capture_details and len(mappings) < c.max_expanded_segments:
                stages.append(StageTiming("SRAM_PORT_TRANSFER", transfer.start_ns, transfer.end_ns,
                                          transfer.resource_id, amount))
                mappings.append(AddressMapping(address=(first + (port - first) % c.ports) * c.burst_bytes,
                                               channel=port, unit=port))
        physical = count * c.burst_bytes
        counters = {
            "physical_memory_model": "native_pipelined_sram_ports",
            "mapping_scope": "burst_interleaved_ports_coalesced_contiguous_request",
            "burst_count": count, "sram_ports_touched": len(plans), "sram_port_count": c.ports,
            "queue_wait_ns": arrival - request.arrival_ns,
            "physical_read_bytes": physical if read else 0,
            "physical_write_bytes": 0 if read else physical,
            "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
            "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
            "bandwidth_ceiling_gb_s": c.directional_bandwidth_gb_s(request.operation),
            "write_completion": "sram_port_write_complete" if not read else "n/a",
            "details_truncated": self.capture_details and len(plans) > c.max_expanded_segments,
        }
        counters.update(timeline._core_metrics_delta(before))
        self._acceptance_ns = arrival
        self._inflight = [*inflight, completion]
        return TransactionResult(request.request_id, request.operation, request.arrival_ns, completion,
                                 request.byte_count, physical, tuple(mappings), tuple(stages), counters)

    def submit_many(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        requests = tuple(requests)
        for request in requests:
            self._validate(request)
        return tuple(self.submit(request) for request in requests)

    def run(self, requests: Iterable[AccessRequest]):
        return summarize_batch(self.submit_many(requests), bandwidth_ceiling_gb_s=self.config.bandwidth_gb_s)


_OWNED_OWNER_SRAM_METHODS = tuple((name, getattr(SramCore, name))
    for name, value in vars(SramCore).items() if callable(value))

__all__ = ["SramCore"]
