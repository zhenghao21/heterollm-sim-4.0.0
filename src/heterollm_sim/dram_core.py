"""Shared lightweight DRAM read/write core for DDR, LPDDR, HBM and GDDR."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Dict, Iterable, Optional

from .memory_mapping import map_dram_address, split_dram_request, validate_dram_request
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, DramConfig, Operation, StageTiming, TransactionResult


@dataclass
class _BankState:
    open_row: Optional[int] = None
    ready_ns: float = 0.0


def _repeat_add(value: float, increment: float, count: int) -> float:
    """Fast-forward serial IEEE additions within a constant-spacing binade.

    Within one exponent range, adding a fixed increment advances a constant
    number of ULPs. One actual addition resolves ties-to-even first; the
    remaining integer ULP steps can be combined without changing rounding.
    Re-enter at each exponent boundary instead of multiplying the original
    increment, which would drift from serial addition at large timestamps.
    """
    while count:
        value += increment
        count -= 1
        if not count or not math.isfinite(value):
            break
        step = (value + increment) - value
        if not math.isfinite(step):
            return value + increment
        if step == 0.0:
            break
        exponent = math.frexp(value)[1]
        upper = float.fromhex("0x1.fffffffffffffp+1023") if exponent == 1024 else math.ldexp(1.0, exponent)
        steps = min(count, max(0, int((upper - value) / step) - 1))
        if steps:
            value += steps * step
            count -= steps
    return value


@lru_cache(maxsize=128)
def _mapping_cycle(lanes, banks, groups, banks_per_group, inner,
                   bank_prefix, command_prefix, data_prefix, shared, stack, die):
    """Immutable resource IDs for one row-periodic address-mapping cycle.

    This is the lane/bank decoding in map_dram_address, with row/column
    omitted because the caller already bounds execution to one row. Cached
    entries contain no mutable bank state, so transaction rollback remains
    owned by the live core's _banks dictionary.
    """
    entries = []
    for bank in range(banks):
        rank, bank_in_rank = divmod(bank, groups * banks_per_group)
        group, bank_index = divmod(bank_in_rank, banks_per_group)
        for lane in range(lanes):
            entry = (f"{bank_prefix}:{stack}:{die}:{lane}:{rank}:{group}:{bank_index}",
                     shared or f"{command_prefix}:{lane}", f"{data_prefix}:{lane}")
            entries.extend([entry] * inner)
    return tuple(entries)


class DramCore:
    """Generate DRAM array and burst stages on a caller-owned time line."""

    def __init__(
        self,
        config: DramConfig,
        timeline: Optional[ResourceTimeline] = None,
        *,
        capture_details: bool = True,
    ) -> None:
        if not isinstance(config, DramConfig):
            raise TypeError("config must be DramConfig")
        if not isinstance(capture_details, bool):
            raise TypeError("capture_details must be a boolean")
        self.config = config
        self.timeline = timeline or ResourceTimeline()
        # Aggregate serving estimates only need scalar timing/counters.  Keep
        # this opt-in so exact event/replay paths retain their full stage and
        # mapping audit trail.
        self.capture_details = capture_details
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
        """Submit one request through the same admission gate as batches."""
        return self.submit(request)

    def submit(self, request: AccessRequest) -> TransactionResult:
        if request.operation is Operation.ERASE:
            raise ValueError("DRAM does not support ERASE operations")
        validate_dram_request(request, self.config)
        effective_arrival = max(request.arrival_ns, self._acceptance_ns)
        inflight = [end for end in self._inflight if end > effective_arrival]
        if len(inflight) >= self.config.max_outstanding_requests:
            effective_arrival = max(effective_arrival, min(inflight))
            inflight = [end for end in inflight if end > effective_arrival]
        before_metrics = self.timeline.metrics_snapshot(
            max_intervals=(
                self.config.max_expanded_segments
                if self.capture_details
                else 0
            )
        )
        result = self._execute_accepted(
            request if effective_arrival == request.arrival_ns
            else replace(request, arrival_ns=effective_arrival)
        )
        result = replace(result, arrival_ns=request.arrival_ns, counters={
            **result.counters, "queue_wait_ns": effective_arrival - request.arrival_ns,
            **self.timeline.metrics_delta(before_metrics),
        })
        self._acceptance_ns = effective_arrival
        self._inflight = [*inflight, result.completion_ns]
        return result

    def submit_many(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        """Execute the same ordered requests, amortizing numeric state packing."""
        items = tuple(requests)
        for request in items:
            if request.operation is Operation.ERASE:
                raise ValueError("DRAM does not support ERASE operations")
            validate_dram_request(request, self.config)
        if len(items) > 1 and self._can_accelerate():
            from ._compiled_dram import execute_compiled_many
            results = execute_compiled_many(self, items)
            if results is not None:
                return results
        return tuple(self.submit(request) for request in items)

    def _execute_accepted(self, request: AccessRequest) -> TransactionResult:
        # Summary execution can compress repeated row-hit cycles, but must
        # retain the same bank and command/data state as the burst loop below.
        start = request.address
        end = start + request.byte_count
        first = start // self.config.burst_bytes
        last = (end - 1) // self.config.burst_bytes
        segment_count = last - first + 1
        # Even a short row prefix benefits from the summary-only reservation
        # loop, without constructing Segment/StageTiming objects per burst.
        if segment_count >= 64 and self._can_accelerate():
            from ._compiled_dram import execute_compiled
            compiled = execute_compiled(self, request)
            if compiled is not None:
                return compiled
            return self._execute_accelerated_accepted(request, first, last + 1)
        segments = split_dram_request(request, self.config)
        stages = []
        mappings = []
        segment_count = 0
        details_truncated = False
        def record(stage):
            if self.capture_details and not details_truncated:
                stages.append(stage)
        completion = request.arrival_ns
        logical = 0
        physical = 0
        row_hits = row_misses = row_conflicts = 0
        for segment in segments:
            segment_count += 1
            if segment_count > self.config.max_expanded_segments and not details_truncated:
                stages.clear(); mappings.clear(); details_truncated = True
            m = segment.mapping
            if self.capture_details and not details_truncated:
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
                    record(StageTiming("PRECHARGE", pre.start_ns, pre.end_ns, bank_id))
                    dependency = pre.end_ns
                else:
                    row_misses += 1
                act = self.timeline.reserve(bank_id, max(dependency, bank.ready_ns), self.config.open_ns)
                record(StageTiming("ACTIVATE", act.start_ns, act.end_ns, bank_id))
                dependency = act.end_ns
                bank.open_row = m.row
                bank.ready_ns = act.end_ns

            command = self.timeline.reserve(
                self._command_id(m), dependency,
                self.config.burst_interval_ns,
            )
            record(StageTiming("COMMAND_RESERVATION", command.start_ns, command.end_ns, self._command_id(m)))
            latency = self.config.read_latency_ns if request.operation is Operation.READ else self.config.write_latency_ns
            data_ready = command.start_ns + latency
            record(StageTiming(
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
            record(StageTiming("BURST_TRANSFER", transfer.start_ns, transfer.end_ns, data_id, segment.transfer_bytes))
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
                "burst_count": segment_count,
                "details_truncated": details_truncated,
                "physical_read_bytes": physical if request.operation is Operation.READ else 0,
                "physical_write_bytes": physical if request.operation is Operation.WRITE else 0,
                "queue_wait_ns": 0.0,
                "bandwidth_ceiling_gb_s": self.config.directional_bandwidth_gb_s(request.operation),
                "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
            },
        )

    def _can_accelerate(self) -> bool:
        """Use the closed form only for a single-slot, row-periodic calendar."""
        config = self.config
        interleave = config.interleave_bytes or config.burst_bytes
        if (self.capture_details or interleave > config.row_bytes
                or config.row_bytes % interleave
                or type(self.timeline) is not ResourceTimeline):
            return False
        command = str(config.metadata.get("command_resource_prefix", "dram:command"))
        data = str(config.metadata.get("data_resource_prefix", "dram:data"))
        bank = str(config.metadata.get("bank_resource_prefix", "dram:bank"))
        shared = config.metadata.get("shared_command_resource")
        external_ids = {f"{prefix}:{lane}" for prefix in (command, data) for lane in range(config.lane_count)}
        if shared:
            external_ids.add(str(shared))
        if any(len(slots) != 1 and (resource_id in external_ids or resource_id.startswith(bank + ":"))
               for resource_id, slots in self.timeline.lane_available.items()):
            return False
        memory_slots = [slots for resource_id, slots in self.timeline.lane_available.items()
                        if resource_id in external_ids or resource_id.startswith(bank + ":")]
        if len({id(slots) for slots in memory_slots}) != len(memory_slots):
            return False
        # Overlapping resource namespaces couple otherwise independent
        # recurrences; leave these unusual custom calendars to the burst loop.
        if command == data or command == bank or data == bank:
            return False
        if shared and (str(shared).startswith(bank + ":")
                       or str(shared) in {f"{data}:{lane}" for lane in range(config.lane_count)}):
            return False
        # Prefixes are arbitrary metadata: a nested command/data prefix can
        # name an actual bank resource even when the three prefixes differ.
        bank_limits = (config.stacks, config.dies_per_stack, config.lane_count,
                       config.ranks_per_channel, config.bank_groups_per_rank, config.banks_per_group)
        for prefix in (command, data):
            for lane in range(config.lane_count):
                resource_id = f"{prefix}:{lane}"
                if resource_id.startswith(bank + ":"):
                    coordinates = resource_id[len(bank) + 1:].split(":")
                    if (len(coordinates) == 6 and all(value.isdecimal() for value in coordinates)
                            and all(0 <= int(value) < limit for value, limit in zip(coordinates, bank_limits))):
                        return False
        return True

    def _execute_accelerated_accepted(
        self, request: AccessRequest, first: int, stop: int
    ) -> TransactionResult:
        """Execute row boundaries; fast-forward only identical row-hit cycles.

        After the first complete mapping cycle, each bank has its target row
        open and each data resource has the request's direction. A repeated
        cycle has C' = C + Q and D' = max(D + T, C + A). Its K-fold max-plus
        composition is closed form. Executing the final cycle normally then
        restores every bank's final recovery gate and last resource interval.
        No ACT/PRE, direction switch or row boundary is skipped.
        """
        config, timeline = self.config, self.timeline
        burst = config.burst_bytes
        lanes, banks = config.lane_count, config.bank_count
        inner = (config.interleave_bytes or burst) // burst
        period = lanes * banks * inner
        row_span = lanes * banks * (config.row_bytes // burst)
        q = config.burst_interval_ns
        p = burst / config.directional_lane_bandwidth_gb_s(request.operation)
        read = request.operation is Operation.READ
        latency = config.read_latency_ns if read else config.write_latency_ns
        recovery = config.read_recovery_ns if read else config.write_recovery_ns
        direction = request.operation.value
        shared = config.metadata.get("shared_command_resource")
        command_prefix = str(config.metadata.get("command_resource_prefix", "dram:command"))
        data_prefix = str(config.metadata.get("data_resource_prefix", "dram:data"))
        bank_prefix = str(config.metadata.get("bank_resource_prefix", "dram:bank"))
        command_ids = [str(shared) if shared else f"{command_prefix}:{lane}" for lane in range(lanes)]
        data_ids = [f"{data_prefix}:{lane}" for lane in range(lanes)]
        slots, ready, busy = timeline.lane_available, timeline.ready_ns, timeline.busy_ns
        moved, last, touched = timeline.bytes_moved, timeline.last_intervals, timeline._touched
        cursor = first
        cycle_table = ()
        row = 0
        completion = request.arrival_ns
        hits = misses = conflicts = skipped = 0

        def reserve(resource_id, earliest, duration, *, transfer=False):
            lane_slots = slots.get(resource_id)
            if lane_slots is None:
                lane_slots = [float(ready.get(resource_id, 0.0))]
                slots[resource_id] = lane_slots
            previous_end = float(lane_slots[0])
            if transfer:
                previous = timeline.directions.get(resource_id)
                if previous and previous != direction:
                    previous_end += config.read_to_write_ns if previous == "read" else config.write_to_read_ns
            start = max(earliest, previous_end)
            end = start + duration
            lane_slots[0] = ready[resource_id] = end
            busy[resource_id] = busy.get(resource_id, 0.0) + duration
            last[resource_id] = (start, end)
            touched.add(resource_id)
            timeline._intervals_truncated = True
            if transfer:
                timeline.directions[resource_id] = direction
                moved[resource_id] = moved.get(resource_id, 0) + burst
            return start, end

        def execute_until(target):
            nonlocal cursor, completion, hits, misses, conflicts
            while cursor < target:
                bank_id, command_id, data_id = cycle_table[cursor % period]
                bank = self._banks.get(bank_id)
                if bank is None:
                    bank = _BankState()
                    self._banks[bank_id] = bank
                dependency = request.arrival_ns
                if bank.open_row == row:
                    hits += 1
                else:
                    if bank.open_row is not None:
                        conflicts += 1
                        _start, dependency = reserve(bank_id, max(dependency, bank.ready_ns), config.close_ns)
                    else:
                        misses += 1
                    _start, dependency = reserve(bank_id, max(dependency, bank.ready_ns), config.open_ns)
                    bank.open_row = row
                    bank.ready_ns = dependency
                command_start, _end = reserve(command_id, dependency, q)
                _start, end = reserve(data_id, command_start + latency, p, transfer=True)
                bank.ready_ns = max(bank.ready_ns, end + recovery)
                completion = max(completion, end + (recovery if not read else 0.0))
                cursor += 1

        while cursor < stop:
            device, row = divmod(cursor // row_span, config.rows_per_bank)
            stack, die = divmod(device, config.dies_per_stack)
            cycle_table = _mapping_cycle(lanes, banks, config.bank_groups_per_rank, config.banks_per_group,
                                   inner, bank_prefix, command_prefix, data_prefix,
                                   str(shared) if shared else None, stack, die)
            row_stop = min(stop, (cursor // row_span + 1) * row_span)
            aligned = cursor % period == 0
            execute_until(min(row_stop, (cursor // period + 1) * period))
            # A partial prefix does not visit all banks; warm one full cycle
            # before applying the row-hit recurrence.
            if not aligned and cursor + period <= row_stop:
                execute_until(cursor + period)
            cycles = (row_stop - cursor) // period
            count = max(0, cycles - 1)
            if count:
                per_lane = banks * inner
                transfers = count * per_lane
                commands = count * (period if shared else per_lane)
                command_before = [ready[resource_id] for resource_id in command_ids]
                for lane, data_id in enumerate(data_ids):
                    # In real arithmetic, max-plus extrema occur at the
                    # first/last cycle and bank/inner endpoints. Serial-add
                    # evaluation reduces float drift in these suffixes;
                    # IEEE rounding can still move an extremum by an ULP,
                    # so this is not a universal bitwise-equality claim.
                    end = _repeat_add(ready[data_id], p, transfers)
                    if shared:
                        for cycle in {0, count - 1}:
                            for bank_index in {0, banks - 1}:
                                for inner_index in {0, inner - 1}:
                                    command_index = cycle * period + bank_index * lanes * inner + lane * inner + inner_index
                                    transfer_index = cycle * per_lane + bank_index * inner + inner_index
                                    start = _repeat_add(command_before[lane], q, command_index)
                                    end = max(end, _repeat_add(start + latency + p, p, transfers - transfer_index - 1))
                    else:
                        end = max(end, _repeat_add(command_before[lane] + latency + p, p, transfers - 1),
                                  _repeat_add(command_before[lane], q, commands - 1) + latency + p)
                    slots[data_id][0] = ready[data_id] = end
                    busy[data_id] = _repeat_add(busy[data_id], p, transfers)
                    moved[data_id] += transfers * burst
                    last[data_id] = (end - p, end)
                for resource_id in dict.fromkeys(command_ids):
                    end = _repeat_add(ready[resource_id], q, commands)
                    slots[resource_id][0] = ready[resource_id] = end
                    busy[resource_id] = _repeat_add(busy[resource_id], q, commands)
                    last[resource_id] = (end - q, end)
                skipped += count * period
                hits += count * period
                cursor += count * period
            # Includes the last full cycle and any partial tail. It updates
            # all banks touched by the skipped cycles with their last access.
            execute_until(row_stop)

        segment_count = stop - first
        physical = segment_count * burst
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=request.byte_count, transfer_bytes=physical,
            mapping=(), stages=(), counters={
                "row_hits": hits, "row_misses": misses, "row_conflicts": conflicts,
                "burst_count": segment_count,
                "details_truncated": segment_count > config.max_expanded_segments,
                "accelerated_row_hit_bursts": skipped,
                "physical_read_bytes": physical if read else 0,
                "physical_write_bytes": 0 if read else physical,
                "queue_wait_ns": 0.0,
                "bandwidth_ceiling_gb_s": config.directional_bandwidth_gb_s(request.operation),
                "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
            },
        )

    def execute_batch(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        return tuple(self.submit(request) for request in requests)

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return the unified aggregate result."""
        from .memory_transfer import summarize_batch
        return summarize_batch(self.execute_batch(requests), bandwidth_ceiling_gb_s=self.config.bandwidth_gb_s)


def dram_service(config: DramConfig, request: AccessRequest, *, timeline: Optional[ResourceTimeline] = None) -> TransactionResult:
    """Functional convenience wrapper around :class:`DramCore`."""
    return DramCore(config, timeline).execute(request)


__all__ = ["DramCore", "dram_service"]
