"""Shared page-level NAND core for SSD and HBF configurations."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Optional

from .memory_mapping import split_nand_request
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, NandConfig, Operation, StageTiming, TransactionResult


class NandCore:
    """Model page read, page program and block erase without FTL/GC state."""

    def __init__(self, config: NandConfig, timeline: Optional[ResourceTimeline] = None) -> None:
        if not isinstance(config, NandConfig):
            raise TypeError("config must be NandConfig")
        self.config = config
        self.timeline = timeline or ResourceTimeline()
        self._array_ready: Dict[str, float] = {}
        self._buffer_ready: Dict[str, float] = {}
        self._inflight: list[float] = []
        self._acceptance_ns = 0.0

    def _array_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("array_resource_prefix", "nand:array"))
        return f"{prefix}:{mapping.unit}"

    def _array_resource_id(self, mapping) -> str:
        """Return the execution slot, without changing physical mapping."""
        array_id = self._array_id(mapping)
        if self.config.parallel_units is None:
            return array_id
        prefix = str(self.config.metadata.get("parallel_resource_prefix", "nand:parallel"))
        return f"{prefix}:{mapping.unit % self.config.parallel_units}"

    def _channel_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("channel_resource_prefix", "nand:channel"))
        return f"{prefix}:{mapping.channel}"

    def _host_id(self) -> str:
        return str(self.config.metadata.get("host_resource_id", "nand:host"))

    def _array_stage(self, name: str, resource: str, earliest: float, duration: float, bytes: int = 0) -> StageTiming:
        reservation = self.timeline.reserve(resource, earliest, duration)
        return StageTiming(name, reservation.start_ns, reservation.end_ns, resource, bytes)

    def execute(self, request: AccessRequest) -> TransactionResult:
        if not isinstance(request, AccessRequest):
            raise TypeError("request must be AccessRequest")
        segments = split_nand_request(request, self.config)
        stages = []
        mappings = []
        segment_count = 0
        details_truncated = False
        def record(stage):
            if not details_truncated:
                stages.append(stage)
        completion = request.arrival_ns + self.config.front_ns
        logical_bytes = 0
        host_bytes = internal_bytes = 0
        pages_read = pages_programmed = erases = 0
        for segment in segments:
            segment_count += 1
            if segment_count > self.config.max_expanded_segments and not details_truncated:
                stages.clear(); mappings.clear(); details_truncated = True
            m = segment.mapping
            if not details_truncated:
                mappings.append(m)
            logical_bytes += segment.logical_bytes
            array_id = self._array_id(m)
            array_resource = self._array_resource_id(m)
            array_ready = self._array_ready.get(array_id, 0.0)
            buffer_ready = self._buffer_ready.get(array_id, 0.0)
            dependency = max(request.arrival_ns + self.config.front_ns, array_ready, buffer_ready)
            if request.operation is Operation.ERASE:
                stage = self._array_stage("BLOCK_ERASE", array_resource, dependency, self.config.block_erase_ns)
                record(stage)
                self._array_ready[array_id] = stage.end_ns
                self.timeline.ready_ns[array_id] = stage.end_ns
                completion = max(completion, stage.end_ns)
                erases += 1
                continue

            host_bytes += segment.host_transfer_bytes
            internal_bytes += segment.internal_transfer_bytes
            partial = segment.logical_bytes != self.config.page_bytes
            if request.operation is Operation.WRITE and partial and self.config.partial_page_policy == "read_modify_write":
                # Boundary pages need a simplified old-data read before the
                # new bytes can be programmed.  This is intentionally not a
                # persistent page cache.
                read = self._array_stage("RMW_PAGE_READ", array_resource, dependency, self.config.page_read_ns)
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
                self._buffer_ready[array_id] = read_xfer.end_ns
                pages_read += 1
            if request.operation is Operation.READ:
                read = self._array_stage("PAGE_READ", array_resource, dependency, self.config.page_read_ns)
                record(read)
                pages_read += 1
                dependency = read.end_ns
                internal = self.timeline.transfer(
                    self._channel_id(m), segment.internal_transfer_bytes,
                    dependency, self.config.internal_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("INTERNAL_TRANSFER", internal.start_ns, internal.end_ns, internal.resource_id, internal.bytes))
                host = self.timeline.transfer(
                    self._host_id(), segment.host_transfer_bytes,
                    internal.end_ns, self.config.host_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                completion = max(completion, host.end_ns)
                self._array_ready[array_id] = read.end_ns
                self._buffer_ready[array_id] = internal.end_ns
                self.timeline.ready_ns[array_id] = read.end_ns
            else:
                host = self.timeline.transfer(
                    self._host_id(), segment.host_transfer_bytes,
                    request.arrival_ns + self.config.front_ns,
                    self.config.host_bandwidth_gb_s,
                    direction="write",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                internal = self.timeline.transfer(
                    self._channel_id(m), segment.internal_transfer_bytes,
                    max(host.end_ns, self._buffer_ready.get(array_id, 0.0)),
                    self.config.internal_bandwidth_gb_s,
                    direction="write",
                )
                record(StageTiming("INTERNAL_TRANSFER", internal.start_ns, internal.end_ns, internal.resource_id, internal.bytes))
                program = self._array_stage("PAGE_PROGRAM", array_resource, internal.end_ns, self.config.page_program_ns)
                record(program)
                pages_programmed += 1
                completion = max(completion, program.end_ns)
                self._array_ready[array_id] = program.end_ns
                self._buffer_ready[array_id] = program.end_ns
                self.timeline.ready_ns[array_id] = program.end_ns
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=logical_bytes,
            transfer_bytes=internal_bytes,
            mapping=tuple(mappings), stages=tuple(stages),
            counters={
                "host_transfer_bytes": host_bytes,
                "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": pages_read * self.config.transfer_page_bytes,
                "physical_write_bytes": pages_programmed * self.config.transfer_page_bytes,
                "pages_read": pages_read,
                "pages_programmed": pages_programmed,
                "erase_operations": erases,
                "page_count": segment_count,
                "details_truncated": details_truncated,
                "bandwidth_ceiling_gb_s": self.config.host_bandwidth_gb_s,
                "array_units": self.config.array_units,
                "write_completion": "media_program_complete" if request.operation is Operation.WRITE else "n/a",
            },
        )

    def execute_batch(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        results = []
        for request in requests:
            if not isinstance(request, AccessRequest):
                raise TypeError("requests must contain AccessRequest values")
            effective_arrival = max(request.arrival_ns, self._acceptance_ns)
            self._inflight = [end for end in self._inflight if end > effective_arrival]
            if len(self._inflight) >= self.config.max_outstanding_requests:
                effective_arrival = max(effective_arrival, min(self._inflight))
                self._inflight = [end for end in self._inflight if end > effective_arrival]
            result = self.execute(
                request if effective_arrival == request.arrival_ns
                else replace(request, arrival_ns=effective_arrival)
            )
            queue_wait = max(0.0, effective_arrival - request.arrival_ns)
            if queue_wait:
                result = replace(
                    result,
                    arrival_ns=request.arrival_ns,
                    counters={**result.counters, "queue_wait_ns": queue_wait},
                )
            self._acceptance_ns = max(self._acceptance_ns, effective_arrival)
            self._inflight.append(result.completion_ns)
            results.append(result)
        return tuple(results)

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return one aggregate with achieved bandwidth."""
        from .memory_transfer import summarize_batch
        return summarize_batch(self.execute_batch(requests), bandwidth_ceiling_gb_s=self.config.host_bandwidth_gb_s)


def nand_service(config: NandConfig, request: AccessRequest, *, timeline: Optional[ResourceTimeline] = None) -> TransactionResult:
    return NandCore(config, timeline).execute(request)


__all__ = ["NandCore", "nand_service"]
