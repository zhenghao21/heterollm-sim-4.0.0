"""Pack native calendars for the compiled, float64 DRAM timing loop."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from .memory_types import Operation, TransactionResult

_INT64_MAX = (1 << 63) - 1


@lru_cache(maxsize=32)
def _resource_ids(stacks, dies, lanes, banks, groups, banks_per_group,
                  bank_prefix, command_prefix, data_prefix, shared):
    ids = []
    for stack in range(stacks):
        for die in range(dies):
            for lane in range(lanes):
                for bank in range(banks):
                    rank, within = divmod(bank, groups * banks_per_group)
                    group, index = divmod(within, banks_per_group)
                    ids.append(f"{bank_prefix}:{stack}:{die}:{lane}:{rank}:{group}:{index}")
    ids.extend([shared] if shared else [f"{command_prefix}:{lane}" for lane in range(lanes)])
    ids.extend(f"{data_prefix}:{lane}" for lane in range(lanes))
    return tuple(ids)


def execute_compiled(core, request):
    """Return None without changing state when the numeric path is unsuitable."""
    from .dram_core import DramCore, _BankState

    if (type(core) is not DramCore or request.operation not in (Operation.READ, Operation.WRITE)
            or request.byte_count <= 0 or not core._can_accelerate()):
        return None
    config, timeline = core.config, core.timeline
    lanes, banks = config.lane_count, config.bank_count
    total_banks = config.stacks * config.dies_per_stack * lanes * banks
    burst = config.burst_bytes
    first = request.address // burst
    stop = (request.address + request.byte_count - 1) // burst + 1
    physical = (stop - first) * burst
    if (total_banks > 8192 or burst > 2 ** 53 or config.computed_capacity_bytes > _INT64_MAX
            or request.address < 0 or request.address + request.byte_count > config.effective_capacity_bytes
            or stop > _INT64_MAX or physical > _INT64_MAX):
        return None
    metadata = config.metadata
    bank_prefix = str(metadata.get("bank_resource_prefix", "dram:bank"))
    command_prefix = str(metadata.get("command_resource_prefix", "dram:command"))
    data_prefix = str(metadata.get("data_resource_prefix", "dram:data"))
    shared_value = metadata.get("shared_command_resource")
    shared = str(shared_value) if shared_value else None
    ids = _resource_ids(config.stacks, config.dies_per_stack, lanes, banks,
                        config.bank_groups_per_rank, config.banks_per_group,
                        bank_prefix, command_prefix, data_prefix, shared)
    if len(set(ids)) != len(ids):
        return None
    read = request.operation is Operation.READ
    geometry = np.array([lanes, banks, (config.interleave_bytes or burst) // burst,
                         config.row_bytes // burst, config.rows_per_bank, config.dies_per_stack], dtype=np.int64)
    durations = np.array([
        burst, config.burst_interval_ns, burst / config.directional_lane_bandwidth_gb_s(request.operation),
        config.read_latency_ns if read else config.write_latency_ns,
        config.read_recovery_ns if read else config.write_recovery_ns,
        config.open_ns, config.close_ns, config.read_to_write_ns, config.write_to_read_ns,
    ], dtype=np.float64)
    count = len(ids)
    bankrows = np.full(total_banks, -1, dtype=np.int64)
    bankready = np.zeros(total_banks, dtype=np.float64)
    banktouched = np.zeros(total_banks, dtype=np.bool_)
    calendar = np.zeros(count, dtype=np.float64)
    busy = np.zeros(count, dtype=np.float64)
    moved = np.zeros(count, dtype=np.int64)
    directions = np.zeros(count, dtype=np.int8)
    last_start, last_end = np.zeros(count), np.zeros(count)
    touched = np.zeros(count, dtype=np.bool_)
    live_slots = set()
    try:
        for index, rid in enumerate(ids):
            slots = timeline.lane_available.get(rid)
            if slots is not None:
                if len(slots) != 1 or id(slots) in live_slots:
                    return None
                live_slots.add(id(slots))
            calendar[index] = slots[0] if slots is not None else timeline.available_ns(rid)
            busy[index] = timeline.busy_ns.get(rid, 0.0)
            previous_bytes = timeline.bytes_moved.get(rid, 0)
            if (isinstance(previous_bytes, bool) or not isinstance(previous_bytes, (int, np.integer))
                    or previous_bytes < 0 or previous_bytes > _INT64_MAX - physical):
                return None
            moved[index] = previous_bytes
            previous_direction = timeline.directions.get(rid)
            directions[index] = 1 if previous_direction == "read" else 2 if previous_direction == "write" else 3 if previous_direction else 0
            last_start[index], last_end[index] = timeline.last_intervals.get(rid, (0.0, 0.0))
            if index < total_banks:
                bank = core._banks.get(rid)
                if bank is not None:
                    if bank.open_row is not None:
                        if (isinstance(bank.open_row, bool) or not isinstance(bank.open_row, int)
                                or not 0 <= bank.open_row <= _INT64_MAX):
                            return None
                        bankrows[index] = bank.open_row
                    bankready[index] = bank.ready_ns
    except (ValueError, TypeError, OverflowError):
        return None
    if (not math.isfinite(request.arrival_ns)
            or not all(np.isfinite(values).all() for values in
                       (bankready, calendar, busy, last_start, last_end, durations))):
        return None
    try:
        from ._dram_numeric import run_numeric
    except (ImportError, OSError):
        # Old environments can still execute the exact Python model. The
        # project's declared dependency supplies this accelerator on install.
        return None
    completion, hits, misses, conflicts, skipped = run_numeric(
        first, stop, float(request.arrival_ns), read, geometry, durations, bool(shared),
        bankrows, bankready, banktouched, calendar, busy, moved, directions, last_start, last_end, touched)
    if (not math.isfinite(completion)
            or not all(np.isfinite(values).all() for values in (bankready, calendar, busy, last_start, last_end))):
        return None
    # Commit only visited entries; calendars supplied by the kernel retain
    # their list identity. No BankState object survives transaction rollback.
    for index in np.flatnonzero(banktouched):
        rid = ids[index]
        bank = core._banks.get(rid)
        if bank is None:
            bank = _BankState()
            core._banks[rid] = bank
        bank.open_row, bank.ready_ns = int(bankrows[index]), float(bankready[index])
    first_data = total_banks + (1 if shared else lanes)
    for index in np.flatnonzero(touched):
        rid = ids[index]
        timeline.lane_available.setdefault(rid, [0.0])[0] = float(calendar[index])
        timeline.ready_ns[rid] = float(calendar[index])
        timeline.busy_ns[rid] = float(busy[index])
        timeline.last_intervals[rid] = (float(last_start[index]), float(last_end[index]))
        timeline._touched.add(rid)
        if index >= first_data:
            timeline.bytes_moved[rid] = int(moved[index])
            timeline.directions[rid] = request.operation.value
    timeline._intervals_truncated = True
    return TransactionResult(
        request_id=request.request_id, operation=request.operation,
        arrival_ns=request.arrival_ns, completion_ns=float(completion),
        logical_bytes=request.byte_count, transfer_bytes=physical,
        counters={
            "row_hits": int(hits), "row_misses": int(misses), "row_conflicts": int(conflicts),
            "burst_count": stop - first, "details_truncated": stop - first > config.max_expanded_segments,
            "accelerated_row_hit_bursts": int(skipped), "compiled_float64": True,
            "physical_read_bytes": physical if read else 0, "physical_write_bytes": 0 if read else physical,
            "queue_wait_ns": 0.0, "bandwidth_ceiling_gb_s": config.directional_bandwidth_gb_s(request.operation),
            "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
            "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
        },
    )
