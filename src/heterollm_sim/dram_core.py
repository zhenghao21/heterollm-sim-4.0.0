"""Shared lightweight DRAM read/write core for DDR, LPDDR and HBM."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional

from .memory_mapping import split_dram_request
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, DramConfig, Operation, StageTiming, TransactionResult


@dataclass
class _BankState:
    open_row: Optional[int] = None
    ready_ns: float = 0.0


class DramCore:
    """Generate DRAM array and burst stages on a caller-owned time line."""

    def __init__(self, config: DramConfig, timeline: Optional[ResourceTimeline] = None) -> None:
        if isinstance(config, Mapping):
            config = DramConfig.from_mapping(config)
        if not isinstance(config, DramConfig):
            raise TypeError("config must be DramConfig or a mapping")
        self.config = config
        self.timeline = timeline or ResourceTimeline()
        self._banks: Dict[str, _BankState] = {}
        self._inflight: list[float] = []
        self._acceptance_ns = 0.0

    def _bank_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("bank_resource_prefix", "dram:bank"))
        return f"{prefix}:{mapping.stack}:{mapping.die}:{mapping.lane}:{mapping.rank}:{mapping.bank_group}:{mapping.bank}"

    def _data_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("data_resource_prefix", "dram:data"))
        return f"{prefix}:{mapping.lane}"

    def _command_id(self, mapping) -> str:
        # HBM pseudo-channels may share a command/address path.  Callers can
        # provide ``shared_command_resource`` in metadata; the data path stays
        # per lane.
        shared = self.config.metadata.get("shared_command_resource")
        prefix = str(self.config.metadata.get("command_resource_prefix", "dram:command"))
        return str(shared) if shared else f"{prefix}:{mapping.lane}"

    def execute(self, request: AccessRequest) -> TransactionResult:
        if not isinstance(request, AccessRequest):
            raise TypeError("request must be AccessRequest")
        if request.operation not in {Operation.READ, Operation.WRITE}:
            raise ValueError("DRAM supports READ and WRITE only")
        segments = split_dram_request(request, self.config)
        stages = []
        mappings = []
        completion = request.arrival_ns
        logical = 0
        physical = 0
        row_hits = row_misses = row_conflicts = 0
        for segment in segments:
            m = segment.mapping
            mappings.append(m)
            logical += segment.logical_bytes
            physical += segment.transfer_bytes
            bank_id = self._bank_id(m)
            bank = self._banks.setdefault(bank_id, _BankState())
            dependency = request.arrival_ns
            row_hit = bank.open_row == m.row
            if row_hit:
                row_hits += 1
            else:
                if bank.open_row is not None:
                    row_conflicts += 1
                    pre = self.timeline.reserve(bank_id, max(dependency, bank.ready_ns), self.config.close_ns)
                    stages.append(StageTiming("PRECHARGE", pre.start_ns, pre.end_ns, bank_id))
                    dependency = pre.end_ns
                else:
                    row_misses += 1
                act = self.timeline.reserve(bank_id, max(dependency, bank.ready_ns), self.config.open_ns)
                stages.append(StageTiming("ACTIVATE", act.start_ns, act.end_ns, bank_id))
                dependency = act.end_ns
                bank.open_row = m.row
                bank.ready_ns = act.end_ns

            command = self.timeline.reserve(
                self._command_id(m), dependency,
                self.config.burst_interval_ns,
            )
            latency = self.config.read_latency_ns if request.operation is Operation.READ else self.config.write_latency_ns
            data_ready = command.start_ns + latency
            stages.append(StageTiming(
                "READ_PIPELINE" if request.operation is Operation.READ else "WRITE_PIPELINE",
                command.start_ns, data_ready, self._command_id(m),
            ))
            data_id = self._data_id(m)
            previous = self.timeline.directions.get(data_id)
            switch = 0.0
            if previous and previous != request.operation.value:
                switch = self.config.read_to_write_ns if previous == Operation.READ.value else self.config.write_to_read_ns
            transfer = self.timeline.transfer(
                data_id, segment.transfer_bytes, data_ready,
                self.config.directional_lane_bandwidth_gb_s(request.operation),
                direction=request.operation.value, switch_ns=switch,
            )
            stages.append(StageTiming("BURST_TRANSFER", transfer.start_ns, transfer.end_ns, data_id, segment.transfer_bytes))
            recovery = self.config.read_recovery_ns if request.operation is Operation.READ else self.config.write_recovery_ns
            # A row remains open while later column commands are issued.  The
            # recovery gate constrains a future PRECHARGE/ACTIVATE, not the
            # next same-row command or the data bus.  Keeping it out of the
            # shared timeline is what permits read latency and burst transfer
            # to pipeline instead of charging first-data latency per burst.
            bank.ready_ns = max(bank.ready_ns, transfer.end_ns + recovery)
            completion = max(completion, transfer.end_ns + (recovery if request.operation is Operation.WRITE else 0.0))
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=logical, transfer_bytes=physical,
            mapping=tuple(mappings), stages=tuple(stages),
            counters={
                "row_hits": row_hits, "row_misses": row_misses,
                "row_conflicts": row_conflicts,
                "burst_count": len(segments),
                "physical_read_bytes": physical if request.operation is Operation.READ else 0,
                "physical_write_bytes": physical if request.operation is Operation.WRITE else 0,
                "queue_wait_ns": 0.0,
                "bandwidth_ceiling_gb_s": self.config.bandwidth_gb_s,
                "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
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
            self._acceptance_ns = effective_arrival
            self._inflight.append(result.completion_ns)
            results.append(result)
        return tuple(results)

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return the unified aggregate result."""
        from .memory_transfer import summarize_batch
        return summarize_batch(self.execute_batch(requests), bandwidth_ceiling_gb_s=self.config.bandwidth_gb_s)


def dram_service(config: DramConfig, request: AccessRequest, *, timeline: Optional[ResourceTimeline] = None) -> TransactionResult:
    """Functional convenience wrapper around :class:`DramCore`."""
    return DramCore(config, timeline).execute(request)


__all__ = ["DramCore", "dram_service"]
