"""Shared lightweight DRAM read/write core for DDR, LPDDR, HBM and GDDR."""
from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Dict, Iterable, Optional

from .memory_mapping import map_dram_address, split_dram_request, validate_dram_request, dram_request_channels
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, DramConfig, Operation, StageTiming, TransactionResult


# Numeric state and mapping tables are sized dynamically. Keep allocation
# bounded while admitting standard multi-stack HBM3/HBM4 organizations.
MAX_ACCELERATED_BANKS = 65_536


@dataclass
class _BankState:
    open_row: Optional[int] = None
    ready_ns: float = 0.0
    # Column commands must wait for the current row's ACT even when the
    # command resource has multiple slots. This is separate from ready_ns,
    # which includes data recovery and only gates a subsequent PRE/ACT.
    activation_ready_ns: float = 0.0


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
def _mapping_cycle(lanes, banks, groups, banks_per_group,
                   bank_prefix, command_prefix, data_prefix, shared, stack, die,
                   independent_stack_lanes=0):
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
            physical_stack = lane // independent_stack_lanes if independent_stack_lanes else stack
            entry = (f"{bank_prefix}:{physical_stack}:{die}:{lane}:{rank}:{group}:{bank_index}",
                     shared or f"{command_prefix}:{lane}", f"{data_prefix}:{lane}")
            # Inner bursts share exactly the same resources. Repeating the
            # tuple once per burst can allocate gigabytes for a legal large
            # interleave even when the request touches only a few bursts.
            entries.append(entry)
    return tuple(entries)


class DramCore:
    """Generate ordered DRAM command, array and data-path timing.

    Analytical mode uses the reduced model: PRE/ACT reserve a bank for
    close_ns/open_ns, columns reserve per-lane command resources, and optional
    group gates use (stack, die, lane, rank). It does not model refresh/tRAS.

    jedec_hbm dispatches to the normal-operation HBM controller: both PCs
    share their physical channel's separate row/column command buses, ACT/PRE
    occupancy is distinct from array timing, effective row halves may share an
    activated page, maximum row-active deadlines are enforced, and an
    explicit periodic all-bank refresh policy is supported. Its configured
    timing inputs do not establish a vendor speed
    grade or support for training or the full standard.
    submit executes immediately. submit_many optionally selects whole arrived
    requests from a finite window; this is not a per-command FR-FCFS model.
    """

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
        # Kept together so runtime snapshots can copy all group histories.
        # Per-group histories retain A -> B -> A dependencies; one scalar
        # "last group" would lose the older A event. The sorted ACT calendar
        # retains future reservations until the admission time proves they
        # cannot affect another four-activation window.
        self._bank_group_state: Dict[tuple[int, int, int, int], dict] = {}
        self._hbm_control_state: dict = {}
        self._hbm_transactions: list = []
        self._hbm_numeric_state = None
        self._hbm_numeric_enabled = False
        self._inflight: list[float] = []
        self._channel_inflight: dict[tuple[int, int], list[float]] = {}
        self._acceptance_ns = 0.0
        self._scheduler_direction = None
        self._scheduler_streak = 0
        self._largest_stage_ns = max(
            config.open_ns, config.close_ns, config.read_latency_ns,
            config.write_latency_ns, config.burst_interval_ns,
            config.read_recovery_ns, config.write_recovery_ns,
            config.read_to_write_ns, config.write_to_read_ns,
            config.column_same_group_ns, config.column_different_group_ns,
            config.activate_same_group_ns, config.activate_different_group_ns,
            config.four_activate_window_ns,
            config.write_to_read_same_group_ns, config.write_to_read_different_group_ns,
            config.burst_bytes / config.directional_lane_bandwidth_gb_s(Operation.READ),
            config.burst_bytes / config.directional_lane_bandwidth_gb_s(Operation.WRITE),
        )
        if config.control_mode == "jedec_hbm":
            from .memory_types import HBM_CONTROL_TIMING_FIELDS
            self._largest_stage_ns = max(self._largest_stage_ns,
                *(getattr(config, name) for name in HBM_CONTROL_TIMING_FIELDS))

    @property
    def _banks(self):
        self._hbm_publish_numeric()
        return self.__dict__["_banks"]

    @_banks.setter
    def _banks(self, value):
        if "_banks" in self.__dict__:
            self._hbm_publish_numeric()
        self.__dict__["_banks"] = value

    @property
    def _hbm_control_state(self):
        self._hbm_publish_numeric()
        return self.__dict__["_hbm_control_state"]

    @_hbm_control_state.setter
    def _hbm_control_state(self, value):
        if "_hbm_control_state" in self.__dict__:
            self._hbm_publish_numeric()
        self.__dict__["_hbm_control_state"] = value

    def _hbm_publish_numeric(self):
        """Expose current Python state and end exclusive numeric ownership.

        A caller may retain and mutate either dictionary after reading it.
        Therefore a public read disables array residency, rather than relying
        on an unsafe identity check to detect later mutations through aliases.
        """
        self.__dict__["_hbm_numeric_enabled"] = False
        packed = self.__dict__.get("_hbm_numeric_state")
        if packed is None:
            return
        for token in self._hbm_transactions:
            token.materialize(self)
        if packed._controller_commits:
            packed.commit_controller()
        self._hbm_numeric_state = None

    def begin_transaction(self):
        """Protect bank/controller state; the caller owns queues and timeline."""
        from ._hbm_transaction import HBMTransaction
        token = HBMTransaction(self)
        if self.capture_details or self.config.control_mode != "jedec_hbm":
            token.materialize(self)
        self._hbm_transactions.append(token)
        return token

    def _finish_transaction(self, token):
        if not self._hbm_transactions or self._hbm_transactions[-1] is not token:
            raise ValueError("HBM transactions must finish once in reverse begin order")
        self._hbm_transactions.pop()

    def commit_transaction(self, token):
        self._finish_transaction(token)

    def rollback_transaction(self, token):
        self._finish_transaction(token)
        token.rollback(self)

    def _hbm_before_reference(self):
        self._hbm_publish_numeric()
        for token in self._hbm_transactions:
            token.materialize(self)

    def _hbm_remember_bank(self, key):
        for token in self._hbm_transactions:
            token.remember_bank(self.__dict__["_banks"], key)

    def _hbm_remember_entry(self, namespace, key):
        for token in self._hbm_transactions:
            token.remember_entry(self.__dict__["_hbm_control_state"][namespace], key)

    def _bank_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("bank_resource_prefix", "dram:bank"))
        return f"{prefix}:{mapping.stack}:{mapping.die}:{mapping.lane}:{mapping.rank}:{mapping.bank_group}:{mapping.bank}"

    def _data_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("data_resource_prefix", "dram:data"))
        return self.timeline.canonical_resource_id(f"{prefix}:{mapping.lane}")

    def _command_id(self, mapping) -> str:
        # HBM pseudo-channels may share a command/address path.  Callers can
        # provide ``shared_command_resource`` in metadata; the data path stays
        # per lane.
        shared = self.config.metadata.get("shared_command_resource")
        prefix = str(self.config.metadata.get("command_resource_prefix", "dram:command"))
        return self.timeline.canonical_resource_id(str(shared) if shared else f"{prefix}:{mapping.lane}")

    def _bank_group_history(self, mapping) -> dict:
        scope = (mapping.stack, mapping.die, mapping.lane, mapping.rank)
        state = self._bank_group_state.get(scope)
        if state is None:
            state = {
                "column": {"read": {}, "write": {}},
                "activate": {},
                "activate_window": [],
                "write_end": {},
            }
            self._bank_group_state[scope] = state
        return state

    @staticmethod
    def _group_history_ready(history, group, same_ns, different_ns, earliest, *, clock_quantum=None):
        for previous_group, timestamp in history.items():
            gap = same_ns if previous_group == group else different_ns
            # Each zero-valued timing is individually disabled. In
            # particular, enabling ACT/column gates must not silently add
            # WRITE-data-end -> READ-command ordering when WTR is zero.
            if gap > 0:
                if clock_quantum is None:
                    end = timestamp + gap
                else:
                    from ._hbm_clock import clock_after
                    end = clock_after(timestamp, gap, clock_quantum)
                earliest = max(earliest, end)
        return earliest

    def _bank_group_activate_ready(self, state, group, earliest, arrival_ns=0.0, *, clock_quantum=None):
        config = self.config
        earliest = self._group_history_ready(
            state["activate"], group, config.activate_same_group_ns,
            config.activate_different_group_ns, earliest, clock_quantum=clock_quantum,
        )
        width = config.four_activate_window_ns
        if width <= 0:
            return earliest
        def window_end(start):
            if clock_quantum is None:
                return start + width
            from ._hbm_clock import clock_after
            return clock_after(start, width, clock_quantum)

        calendar = state["activate_window"]
        # A reservation on one bank may be far in the future while another
        # bank remains free now. Only monotonic admission time, not that
        # future ACT, proves old starts can no longer constrain an insertion.
        expired = 0
        while expired < len(calendar) and window_end(calendar[expired]) <= arrival_ns:
            expired += 1
        del calendar[:expired]
        while True:
            position = bisect_right(calendar, earliest)
            first = max(0, position - 4)
            neighbors = calendar[first:position] + [earliest] + calendar[position:position + 4]
            candidate = position - first
            next_start = earliest
            # Only the consecutive five-ACT windows containing this candidate
            # can have become invalid. Existing windows were already legal.
            for offset in range(max(0, candidate - 4), min(candidate, len(neighbors) - 5) + 1):
                window = neighbors[offset:offset + 5]
                if window[-1] < window_end(window[0]):
                    # To repair this violating window without moving before
                    # the lower bound, pass the oldest of its other four ACTs
                    # by a full window. Compare deadlines by addition so
                    # float64 rounding cannot cause a zero-progress loop.
                    other_starts = [value for index, value in enumerate(window)
                                    if index != candidate - offset]
                    next_start = max(next_start, window_end(min(other_starts)))
            if next_start == earliest:
                return earliest
            earliest = next_start

    def _bank_group_column_ready(self, state, group, operation, earliest):
        config = self.config
        earliest = self._group_history_ready(
            state["column"][operation.value], group, config.column_same_group_ns,
            config.column_different_group_ns, earliest,
        )
        if operation is Operation.READ:
            earliest = self._group_history_ready(
                state["write_end"], group, config.write_to_read_same_group_ns,
                config.write_to_read_different_group_ns, earliest,
            )
        return earliest

    def submit(self, request: AccessRequest) -> TransactionResult:
        validate_dram_request(request, self.config)
        if self._finite_guard_required(request):
            # Ordinary requests need no extra state copies. At extreme
            # float64 magnitudes, validate the actual transitions privately
            # rather than returning inf or partially modifying live banks.
            # A conservative bound selects the check; it does not reject a
            # request whose actual timing still fits in finite float64.
            import copy
            # Config and the scheduler's alias map are read-only contracts;
            # the latter can be MappingProxyType, which is not pickleable.
            shadow = copy.deepcopy(self, {
                id(self.config): self.config,
                id(self.timeline.resource_aliases): self.timeline.resource_aliases,
            })
            result = shadow._submit_validated(request)
            if (not math.isfinite(result.completion_ns)
                    or any(not math.isfinite(bank.ready_ns) for bank in shadow._banks.values())
                    or any(not math.isfinite(value) for value in shadow.timeline.busy_ns.values())):
                raise ValueError("DRAM request exceeds finite timing range")
        return self._submit_validated(request)

    def _finite_guard_required(self, request: AccessRequest) -> bool:
        burst = self.config.burst_bytes
        count = (request.address + request.byte_count - 1) // burst - request.address // burst + 1
        try:
            # Below half an ULP near float64's maximum, additions cannot
            # overflow; seven serial stage/turnaround costs per burst bound
            # all timing increases, with an extra factor for rounding.
            return self._largest_stage_ns * count * 16 >= float.fromhex("0x1p+970")
        except OverflowError:
            return True

    def _admission_time(self, request, earliest=0.0):
        """Read-only queue eligibility, also used by the bounded scheduler."""
        ready = max(request.arrival_ns, self._acceptance_ns, earliest)
        calendars = ([self._inflight] if self.config.queue_scope == "device" else
                     [self._channel_inflight.get(channel, ())
                      for channel in dram_request_channels(self.config, request)])
        for calendar in calendars:
            active = [end for end in calendar if end > ready]
            if len(active) >= self.config.max_outstanding_requests:
                ready = max(ready, min(active))
        return ready

    def _submit_validated(self, request: AccessRequest, *, execute=None) -> TransactionResult:
        effective_arrival = self._admission_time(request)
        inflight = [end for end in self._inflight if end > effective_arrival]
        before_metrics = self.timeline.metrics_snapshot(
            max_intervals=(
                self.config.max_expanded_segments
                if self.capture_details
                else 0
            )
        )
        result = (self._execute_accepted if execute is None else execute)(
            request if effective_arrival == request.arrival_ns
            else replace(request, arrival_ns=effective_arrival)
        )
        result = replace(result, arrival_ns=request.arrival_ns, counters={
            **result.counters, "queue_wait_ns": effective_arrival - request.arrival_ns,
            "scheduler_reordered_requests": 0, "scheduler_batches": 0,
            "controller_direction_switches": int(self._scheduler_direction is not None
                and self._scheduler_direction != request.operation.value),
            **self.timeline.metrics_delta(before_metrics),
            **({
                "bank_group_timing_enabled": True,
                "bank_group_timing_fallback": self.config.control_mode == "analytical" and not self.capture_details,
            } if self.config.bank_group_timing_enabled else {}),
        })
        self._acceptance_ns = effective_arrival
        if self.config.queue_scope == "device":
            self._inflight = [*inflight, result.completion_ns]
        else:
            for channel in dram_request_channels(self.config, request):
                self._channel_inflight[channel] = [
                    end for end in self._channel_inflight.get(channel, ()) if end > effective_arrival
                ] + [result.completion_ns]
        self._scheduler_streak = (self._scheduler_streak + 1
                                 if self._scheduler_direction == request.operation.value else 1)
        self._scheduler_direction = request.operation.value
        return result

    def submit_many(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        """Schedule a finite arrived window; return results in input order.

        A single submit has no waiting window and cannot reorder requests.
        Whole-request credits are held until the aggregate completion, even
        when the request spans several channels or thousands of bursts.
        """
        items = tuple(requests)
        for request in items:
            validate_dram_request(request, self.config)
        previous_direction = self._scheduler_direction
        if self.config.scheduler != "fcfs" and len(items) > 1:
            from ._dram_scheduler import schedule_requests
            results = schedule_requests(self, items)
        else:
            results = self._submit_many_ordered(items)
        tagged = []
        for index, (request, result) in enumerate(zip(items, results)):
            counters = dict(result.counters)
            counters["scheduler_batches"] = int(index == 0)
            counters.setdefault("scheduler_reordered_requests", 0)
            # The generic compiled FCFS batch performs admission internally.
            if "controller_direction_switches" not in counters:
                counters["controller_direction_switches"] = int(previous_direction is not None
                    and previous_direction != request.operation.value)
                self._scheduler_streak = (self._scheduler_streak + 1
                    if previous_direction == request.operation.value else 1)
                self._scheduler_direction = request.operation.value
            previous_direction = request.operation.value
            tagged.append(replace(result, counters=counters))
        return tuple(tagged)

    def _submit_many_ordered(self, items):
        """FCFS execution, amortizing numeric state packing when possible."""
        if (len(items) > 1 and self.config.control_mode == "jedec_hbm" and not self.capture_details
                and not any(self._finite_guard_required(request) for request in items)):
            try:
                from ._compiled_hbm import execute_hbm_numeric_many
            except (ImportError, OSError):
                pass
            else:
                results = execute_hbm_numeric_many(self, items)
                if results is not None:
                    return results
        if (len(items) > 1 and self.config.queue_scope == "device" and self._can_accelerate()
                and not any(self._finite_guard_required(request) for request in items)):
            from ._compiled_dram import execute_compiled_many
            results = execute_compiled_many(self, items)
            if results is not None:
                return results
        return tuple(self.submit(request) for request in items)

    def _execute_accepted(self, request: AccessRequest) -> TransactionResult:
        if self.config.control_mode == "jedec_hbm":
            from ._hbm_control import execute_hbm
            return execute_hbm(self, request)
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
            group_state = (self._bank_group_history(m)
                           if self.config.bank_group_timing_enabled else None)
            dependency = max(request.arrival_ns, bank.activation_ready_ns)
            row_hit = bank.open_row == m.row
            if row_hit:
                row_hits += 1
            else:
                if bank.open_row is not None:
                    row_conflicts += 1
                    pre = self.timeline.reserve(bank_id, max(dependency, bank.ready_ns), self.config.close_ns)
                    record(StageTiming("PRECHARGE", pre.start_ns, pre.end_ns, pre.resource_id))
                    dependency = pre.end_ns
                else:
                    row_misses += 1
                act_ready = max(dependency, bank.ready_ns)
                if group_state is not None:
                    # Window eligibility can have holes: a free time before
                    # four future ACTs is valid while a later time is not.
                    # Resolve the actual bank calendar first; reserve below
                    # then cannot move the checked start into such a hole.
                    act_ready = max(act_ready, self.timeline.available_ns(bank_id))
                    act_ready = self._bank_group_activate_ready(
                        group_state, m.bank_group, act_ready, request.arrival_ns,
                    )
                act = self.timeline.reserve(bank_id, act_ready, self.config.open_ns)
                if group_state is not None:
                    group_state["activate"][m.bank_group] = max(
                        group_state["activate"].get(m.bank_group, 0.0), act.start_ns,
                    )
                    if self.config.four_activate_window_ns > 0:
                        insort(group_state["activate_window"], act.start_ns)
                record(StageTiming("ACTIVATE", act.start_ns, act.end_ns, act.resource_id))
                dependency = act.end_ns
                bank.open_row = m.row
                bank.ready_ns = act.end_ns
                bank.activation_ready_ns = act.end_ns

            if group_state is not None:
                dependency = self._bank_group_column_ready(
                    group_state, m.bank_group, request.operation, dependency,
                )
            command = self.timeline.reserve(
                self._command_id(m), dependency,
                self.config.burst_interval_ns,
            )
            if group_state is not None:
                history = group_state["column"][request.operation.value]
                history[m.bank_group] = max(history.get(m.bank_group, 0.0), command.start_ns)
            record(StageTiming("COMMAND_RESERVATION", command.start_ns, command.end_ns, self._command_id(m)))
            latency = self.config.read_latency_ns if request.operation is Operation.READ else self.config.write_latency_ns
            data_ready = command.start_ns + latency
            record(StageTiming(
                "READ_PIPELINE" if request.operation is Operation.READ else "WRITE_PIPELINE",
                command.start_ns, data_ready, self._command_id(m),
            ))
            data_id = self._data_id(m)
            # A resource may have several execution slots with different
            # histories. The timeline decides whether the selected slot
            # switches direction; the scalar last direction is insufficient.
            switch = (self.config.write_to_read_ns if request.operation is Operation.READ
                      else self.config.read_to_write_ns)
            transfer = self.timeline.transfer(
                data_id, segment.transfer_bytes, data_ready,
                self.config.directional_lane_bandwidth_gb_s(request.operation),
                direction=request.operation.value, switch_ns=switch,
            )
            if group_state is not None and request.operation is Operation.WRITE:
                group_state["write_end"][m.bank_group] = max(
                    group_state["write_end"].get(m.bank_group, 0.0), transfer.end_ns,
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
        if (config.control_mode == "jedec_hbm" or config.address_mapping != "bank_interleaved"
                or self.capture_details or config.bank_group_timing_enabled
                or config.addressable_stack_count * config.addressable_die_count
                * config.lane_count * config.bank_count > MAX_ACCELERATED_BANKS
                or interleave > config.row_bytes
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
        if any(source != target and (source in external_ids or source.startswith(bank + ":"))
               for source, target in self.timeline.resource_aliases.items()):
            # Folded/native recurrences use unaliased IDs. Only aliases that
            # touch this core's resources require the exact burst fallback.
            return False
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
        bank_limits = (config.stacks, config.addressable_die_count, config.lane_count,
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
                    previous_end += config.write_to_read_ns if read else config.read_to_write_ns
            start = max(earliest, previous_end)
            end = start + duration
            lane_slots[0] = ready[resource_id] = end
            # This path only admits one slot. Match ResourceTimeline.reserve
            # when a caller contracts a previously multi-slot calendar: old
            # per-slot history must not revive if it is expanded again.
            timeline.lane_directions.pop(resource_id, None)
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
                bank_id, command_id, data_id = cycle_table[(cursor % period) // inner]
                bank = self._banks.get(bank_id)
                if bank is None:
                    bank = _BankState()
                    self._banks[bank_id] = bank
                dependency = max(request.arrival_ns, bank.activation_ready_ns)
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
                    bank.activation_ready_ns = dependency
                command_start, _end = reserve(command_id, dependency, q)
                _start, end = reserve(data_id, command_start + latency, p, transfer=True)
                bank.ready_ns = max(bank.ready_ns, end + recovery)
                completion = max(completion, end + (recovery if not read else 0.0))
                cursor += 1

        while cursor < stop:
            device, row = divmod(cursor // row_span, config.rows_per_bank)
            stack, die = divmod(device, config.addressable_die_count)
            cycle_table = _mapping_cycle(lanes, banks, config.bank_groups_per_rank, config.banks_per_group,
                                   bank_prefix, command_prefix, data_prefix,
                                   str(shared) if shared else None, stack, die,
                                   config.lanes_per_stack if config.stack_interface_mode == "independent" else 0)
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

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return the unified aggregate result."""
        from .memory_transfer import summarize_batch
        return summarize_batch(self.submit_many(requests), bandwidth_ceiling_gb_s=self.config.bandwidth_gb_s)


def dram_service(config: DramConfig, request: AccessRequest, *, timeline: Optional[ResourceTimeline] = None) -> TransactionResult:
    """Functional convenience wrapper around :class:`DramCore`."""
    return DramCore(config, timeline).submit(request)


__all__ = ["DramCore", "dram_service"]
