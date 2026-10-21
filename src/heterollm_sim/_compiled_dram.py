"""Pack native calendars for the compiled, float64 DRAM timing loop."""
from __future__ import annotations

from functools import lru_cache
import math
from weakref import WeakKeyDictionary

import numpy as np

from .memory_types import Operation, TransactionResult

_INT64_MAX = (1 << 63) - 1
# Per-request result matrices use 33 bytes per resource cell. This bounds
# those matrices to 264 MiB even when the bank limit admits larger stacks.
_MAX_BATCH_RESOURCE_CELLS = 8_388_608
_WORKSPACES = WeakKeyDictionary()


class _NumericWorkspace:
    """Reusable scratch, never authoritative bank or calendar state.

    Every resource the kernel may access is repacked from the live objects.
    This workspace can therefore survive rollback without a second state
    journal, and retained public bank/calendar aliases remain observable.
    """

    def __init__(self, banks, resources):
        self.shape = banks, resources
        self.floats = np.zeros((6, resources))
        self.arrays = (np.full(banks, -1, dtype=np.int64),
            self.floats[0, :banks], self.floats[1, :banks], np.zeros(banks, dtype=np.bool_),
            self.floats[2], self.floats[3], np.zeros(resources, dtype=np.int64),
            np.zeros(resources, dtype=np.int8), self.floats[4], self.floats[5],
            np.zeros(resources, dtype=np.bool_))


@lru_cache(maxsize=128)
def _numeric_inputs(geometry, durations):
    geometry_array = np.frombuffer(np.asarray(geometry, dtype=np.int64).tobytes(), dtype=np.int64)
    duration_array = np.frombuffer(np.asarray(durations, dtype=np.float64).tobytes(), dtype=np.float64)
    return geometry_array, duration_array, bool(np.isfinite(duration_array).all())


@lru_cache(maxsize=4096)
def _short_resource_indexes(first, stop, lanes, banks, inner, row_bursts, rows_per_bank,
                            total_banks, shared):
    selected = set()
    row_span = lanes * banks * row_bursts
    for stripe in range(first // inner, (stop - 1) // inner + 1):
        cursor = max(first, stripe * inner)
        slot = stripe % (lanes * banks)
        lane, bank = slot % lanes, slot // lanes
        unit = cursor // row_span // rows_per_bank
        selected.add(unit * lanes * banks + lane * banks + bank)
        selected.add(total_banks + (0 if shared else lane))
        selected.add(total_banks + (1 if shared else lanes) + lane)
    return tuple(sorted(selected))


@lru_cache(maxsize=32)
def _resource_ids(stacks, dies, lanes, banks, groups, banks_per_group,
                  bank_prefix, command_prefix, data_prefix, shared,
                  independent_stack_lanes=0):
    ids = []
    for stack in range(stacks):
        for die in range(dies):
            for lane in range(lanes):
                physical_stack = lane // independent_stack_lanes if independent_stack_lanes else stack
                for bank in range(banks):
                    rank, within = divmod(bank, groups * banks_per_group)
                    group, index = divmod(within, banks_per_group)
                    ids.append(f"{bank_prefix}:{physical_stack}:{die}:{lane}:{rank}:{group}:{index}")
    ids.extend([shared] if shared else [f"{command_prefix}:{lane}" for lane in range(lanes)])
    ids.extend(f"{data_prefix}:{lane}" for lane in range(lanes))
    return tuple(ids)


def _publish_numeric(core, arrays, ids, first_data, banktouched, touched):
    from .dram_core import _BankState
    (bankrows, bankready, bankactivation, _, calendar, busy, moved, directions,
     last_start, last_end, _) = arrays
    timeline = core.timeline
    bank_indexes = np.flatnonzero(banktouched)
    for index, row, ready, activation in zip(bank_indexes.tolist(),
            bankrows[bank_indexes].tolist(), bankready[bank_indexes].tolist(),
            bankactivation[bank_indexes].tolist()):
        rid = ids[index]
        core._hbm_remember_bank(rid)
        bank = core.__dict__["_banks"].get(rid)
        if bank is None:
            bank = _BankState()
            core.__dict__["_banks"][rid] = bank
        bank.open_row, bank.ready_ns, bank.activation_ready_ns = row, ready, activation
    resource_indexes = np.flatnonzero(touched)
    for index, ready, work, start, end, byte_count, direction in zip(resource_indexes.tolist(),
            calendar[resource_indexes].tolist(), busy[resource_indexes].tolist(),
            last_start[resource_indexes].tolist(), last_end[resource_indexes].tolist(),
            moved[resource_indexes].tolist(), directions[resource_indexes].tolist()):
        rid = ids[index]
        slots = timeline.lane_available.get(rid)
        if slots is None:
            timeline.lane_available[rid] = [ready]
        else:
            slots[0] = ready
        timeline.lane_directions.pop(rid, None)
        timeline.ready_ns[rid], timeline.busy_ns[rid] = ready, work
        timeline.last_intervals[rid] = (start, end)
        if index >= first_data:
            timeline.bytes_moved[rid] = byte_count
            timeline.directions[rid] = "read" if direction == 1 else "write"


def execute_compiled(core, request, *, _submission=None, _scope=None):
    """Return None without changing state when the numeric path is unsuitable."""
    from .dram_core import DramCore, MAX_ACCELERATED_BANKS

    if (type(core) is not DramCore or request.operation not in (Operation.READ, Operation.WRITE)
            or request.byte_count <= 0
            or ((_scope is None or _scope.workspace is None) and not core._can_accelerate())):
        return None
    config, timeline = core.config, core.timeline
    reused = _scope is not None and _scope.workspace is not None
    if reused:
        lanes, banks, total_banks, burst, ids, shared = _scope.layout
    else:
        lanes, banks = config.lane_count, config.bank_count
        total_banks = config.addressable_stack_count * config.addressable_die_count * lanes * banks
        burst = config.burst_bytes
    first = request.address // burst
    stop = (request.address + request.byte_count - 1) // burst + 1
    physical = (stop - first) * burst
    if (total_banks > MAX_ACCELERATED_BANKS or burst > 2 ** 53 or config.computed_capacity_bytes > _INT64_MAX
            or request.address < 0 or request.address + request.byte_count > config.effective_capacity_bytes
            or stop > _INT64_MAX or physical > _INT64_MAX):
        return None
    if not reused:
        metadata = config.metadata
        bank_prefix = str(metadata.get("bank_resource_prefix", "dram:bank"))
        command_prefix = str(metadata.get("command_resource_prefix", "dram:command"))
        data_prefix = str(metadata.get("data_resource_prefix", "dram:data"))
        shared_value = metadata.get("shared_command_resource")
        shared = str(shared_value) if shared_value else None
        ids = _resource_ids(config.addressable_stack_count, config.addressable_die_count, lanes, banks,
                            config.bank_groups_per_rank, config.banks_per_group,
                            bank_prefix, command_prefix, data_prefix, shared,
                            config.lanes_per_stack if config.stack_interface_mode == "independent" else 0)
        if len(set(ids)) != len(ids):
            return None
    read = request.operation is Operation.READ
    geometry, durations, finite_durations = _numeric_inputs(
        (lanes, banks, (config.interleave_bytes or burst) // burst,
         config.row_bytes // burst, config.rows_per_bank, config.addressable_die_count), (
        burst, config.burst_interval_ns, burst / config.directional_lane_bandwidth_gb_s(request.operation),
        config.read_latency_ns if read else config.write_latency_ns,
        config.read_recovery_ns if read else config.write_recovery_ns,
        config.open_ns, config.close_ns, config.read_to_write_ns, config.write_to_read_ns,
    ))
    count = len(ids)
    workspace = _scope.workspace if reused else _WORKSPACES.get(core)
    if workspace is None or workspace.shape != (total_banks, count):
        workspace = _WORKSPACES[core] = _NumericWorkspace(total_banks, count)
    (bankrows, bankready, bankactivation, banktouched, calendar, busy, moved, directions,
     last_start, last_end, touched) = workspace.arrays
    banktouched.fill(False)
    touched.fill(False)
    indexes = range(count)
    if stop - first < 64:
        # A Decode HOST append often touches four bursts out of dozens of
        # banks. Pack only its actual banks and shared command/data lanes;
        # the unchanged kernel never reads or commits the other entries.
        indexes = _short_resource_indexes(first, stop, lanes, banks,
            (config.interleave_bytes or burst) // burst, config.row_bytes // burst,
            config.rows_per_bank, total_banks, bool(shared))
    live_slots = set()
    before_busy = {} if _submission is not None else None
    before_bytes = {} if _submission is not None else None
    pack_indexes = indexes
    if reused:
        selected = list(indexes)
        busy_values, byte_values = busy[selected].tolist(), moved[selected].tolist()
        if any(value > _INT64_MAX - physical for value in byte_values):
            return None
        selected_ids = tuple(ids[index] for index in selected)
        before_busy, before_bytes = dict(zip(selected_ids, busy_values)), dict(zip(selected_ids, byte_values))
        pack_indexes = [index for index in selected if not _scope.loaded[index]]
    try:
        for index in pack_indexes:
            rid = ids[index]
            slots = timeline.lane_available.get(rid)
            if slots is not None:
                if _scope is not None and type(slots) is not list:
                    return None
                if len(slots) != 1 or id(slots) in live_slots:
                    return None
                live_slots.add(id(slots))
            calendar[index] = slots[0] if slots is not None else timeline.available_ns(rid)
            previous_busy = timeline.busy_ns.get(rid, 0.0)
            busy[index] = previous_busy
            previous_bytes = timeline.bytes_moved.get(rid, 0)
            if (isinstance(previous_bytes, bool) or not isinstance(previous_bytes, (int, np.integer))
                    or previous_bytes < 0 or previous_bytes > _INT64_MAX - physical):
                return None
            moved[index] = previous_bytes
            if before_busy is not None:
                before_busy[rid] = previous_busy
                before_bytes[rid] = previous_bytes
            previous_direction = timeline.directions.get(rid)
            directions[index] = 1 if previous_direction == "read" else 2 if previous_direction == "write" else 3 if previous_direction else 0
            last_start[index], last_end[index] = timeline.last_intervals.get(rid, (0.0, 0.0))
            if index < total_banks:
                bank = core.__dict__["_banks"].get(rid)
                bankrows[index] = -1
                bankready[index] = bankactivation[index] = 0.0
                if bank is not None:
                    if bank.open_row is not None:
                        if (isinstance(bank.open_row, bool) or not isinstance(bank.open_row, int)
                                or not 0 <= bank.open_row <= _INT64_MAX):
                            return None
                        bankrows[index] = bank.open_row
                    bankready[index] = bank.ready_ns
                    bankactivation[index] = bank.activation_ready_ns
    except (ValueError, TypeError, OverflowError):
        workspace.floats.fill(0.0)
        return None
    if (not math.isfinite(request.arrival_ns) or not finite_durations
            or ((not reused or pack_indexes) and not np.isfinite(workspace.floats).all())):
        workspace.floats.fill(0.0)
        return None
    try:
        from ._dram_numeric import run_numeric, run_numeric_short
    except (ImportError, OSError):
        # Old environments can still execute the exact Python model. The
        # project's declared dependency supplies this accelerator on install.
        return None
    execute = run_numeric if stop - first >= 64 else run_numeric_short
    try:
        completion, hits, misses, conflicts, skipped = execute(
            first, stop, float(request.arrival_ns if _submission is None else _submission), read, geometry, durations, bool(shared),
            bankrows, bankready, bankactivation, banktouched, calendar, busy, moved, directions, last_start, last_end, touched)
    except BaseException:
        # Scratch is not live state, but a failed kernel must not leave
        # nonfinite untouched cells that poison a later short request.
        workspace.floats.fill(0.0)
        raise
    if not math.isfinite(completion) or not np.isfinite(workspace.floats).all():
        workspace.floats.fill(0.0)
        return None
    if _submission is not None:
        # There is no callback inside the numeric execution. Begin accounting
        # only after it succeeds, so a declined pack never changes the live
        # timeline. The recorded old values came from that same live pack.
        timeline._intervals = {}
        timeline._interval_payloads = {}
        timeline._interval_limit = 0
        timeline._interval_count = 0
        timeline._touched = set()
    first_data = total_banks + (1 if shared else lanes)
    resource_indexes = np.flatnonzero(touched)
    if _scope is None:
        _publish_numeric(core, workspace.arrays, ids, first_data, banktouched, touched)
        timeline._touched.update(ids[index] for index in resource_indexes.tolist())
    else:
        if not reused:
            _scope.workspace, _scope.ids, _scope.first_data = workspace, ids, first_data
            _scope.layout = lanes, banks, total_banks, burst, ids, shared
            _scope.resource_ids = frozenset(ids)
            _scope.calendar_slot_ids = set()
            _scope.dirty_banks = banktouched.copy()
            _scope.dirty_resources = touched.copy()
            _scope.loaded = np.zeros(count, dtype=np.bool_)
        else:
            _scope.dirty_banks |= banktouched
            _scope.dirty_resources |= touched
        _scope.loaded[list(indexes)] = True
        current_busy = dict(zip((ids[index] for index in resource_indexes.tolist()),
                               busy[resource_indexes].tolist()))
        current_bytes = dict(zip((ids[index] for index in resource_indexes.tolist()),
                                moved[resource_indexes].tolist()))
        current_last = dict(zip(current_busy, zip(last_start[resource_indexes].tolist(),
                                                  last_end[resource_indexes].tolist())))
        # Insert new accounting keys at their ordinary submission position;
        # unrelated link reservations may insert keys before the next HOST
        # call. Values stay numeric until the scope publishes. This preserves
        # the original reduction order without rescanning every resource.
        for index in resource_indexes.tolist():
            rid = ids[index]
            timeline.busy_ns.setdefault(rid, before_busy[rid])
            if rid not in timeline.ready_ns:
                timeline.ready_ns[rid] = float(calendar[index])
            if rid not in timeline.lane_available:
                timeline.lane_available[rid] = [float(calendar[index])]
            _scope.calendar_slot_ids.add(id(timeline.lane_available[rid]))
            timeline.last_intervals.setdefault(rid, current_last[rid])
            if index >= first_data:
                timeline.bytes_moved.setdefault(rid, before_bytes[rid])
                timeline.directions.setdefault(rid, request.operation.value)
            timeline._touched.add(rid)
        # Preserve insertion order even when a task first visits high banks
        # and then low banks. Their values remain in the numeric workspace.
        from .dram_core import _BankState
        for index in np.flatnonzero(banktouched).tolist():
            rid = ids[index]
            if rid not in core.__dict__["_banks"]:
                core._hbm_remember_bank(rid)
                core.__dict__["_banks"][rid] = _BankState()
    timeline._intervals_truncated = True
    counters = {
            "row_hits": int(hits), "row_misses": int(misses), "row_conflicts": int(conflicts),
            "burst_count": stop - first, "details_truncated": stop - first > config.max_expanded_segments,
            "accelerated_row_hit_bursts": int(skipped), "compiled_float64": True,
            "physical_read_bytes": physical if read else 0, "physical_write_bytes": 0 if read else physical,
            "queue_wait_ns": 0.0, "bandwidth_ceiling_gb_s": config.directional_bandwidth_gb_s(request.operation),
            "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
            "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
    }
    if _submission is not None:
        touched_ids = timeline._touched
        # Match the public reducer's insertion order: its consumers sum
        # floating-point costs in that order, including shared resources.
        busy_ids = timeline._ordered_metric_touches("busy_ns", timeline.busy_ns, touched_ids)
        byte_ids = timeline._ordered_metric_touches("bytes_moved", timeline.bytes_moved, touched_ids)
        counters.update({
            "queue_wait_ns": _submission - request.arrival_ns,
            "scheduler_reordered_requests": 0, "scheduler_batches": 0,
            "controller_direction_switches": int(core._scheduler_direction is not None
                and core._scheduler_direction != request.operation.value),
            "resource_busy_ns": {rid: value - before_busy[rid] for rid in busy_ids
                if (value := current_busy[rid]) != before_busy[rid]},
            "resource_bytes": {rid: value - before_bytes[rid] for rid in byte_ids
                if (value := current_bytes[rid]) != before_bytes[rid]},
            "resource_last_intervals": {rid: current_last[rid] for rid in touched_ids},
            "resource_intervals": {}, "resource_interval_payloads": {},
            "intervals_truncated": True,
        })
    return TransactionResult(
        request_id=request.request_id, operation=request.operation,
        arrival_ns=request.arrival_ns, completion_ns=float(completion),
        logical_bytes=request.byte_count, transfer_bytes=physical,
        counters=counters,
    )


def execute_compiled_many(core, requests):
    """Pack once for a bounded ordered batch, or decline without mutation."""
    from .memory_types import AccessRequest
    if not requests:
        return ()
    if len(requests) > 1024 or any(r.operation not in (Operation.READ, Operation.WRITE) for r in requests):
        return None
    # A conservative envelope is used only for integer-overflow validation;
    # execution below retains every separate address, size and arrival.
    low = min(r.address for r in requests)
    high = max(r.address + r.byte_count for r in requests)
    request = AccessRequest("batch-pack", requests[0].operation, low, high - low,
                            min(r.arrival_ns for r in requests))
    from .dram_core import DramCore, _BankState, MAX_ACCELERATED_BANKS

    if (type(core) is not DramCore or request.operation not in (Operation.READ, Operation.WRITE)
            or request.byte_count <= 0 or not core._can_accelerate()):
        return None
    config, timeline = core.config, core.timeline
    lanes, banks = config.lane_count, config.bank_count
    total_banks = config.addressable_stack_count * config.addressable_die_count * lanes * banks
    burst = config.burst_bytes
    first = request.address // burst
    stop = (request.address + request.byte_count - 1) // burst + 1
    physical = (stop - first) * burst
    if (total_banks > MAX_ACCELERATED_BANKS or burst > 2 ** 53 or config.computed_capacity_bytes > _INT64_MAX
            or request.address < 0 or request.address + request.byte_count > config.effective_capacity_bytes
            or stop > _INT64_MAX or physical > _INT64_MAX):
        return None
    metadata = config.metadata
    bank_prefix = str(metadata.get("bank_resource_prefix", "dram:bank"))
    command_prefix = str(metadata.get("command_resource_prefix", "dram:command"))
    data_prefix = str(metadata.get("data_resource_prefix", "dram:data"))
    shared_value = metadata.get("shared_command_resource")
    shared = str(shared_value) if shared_value else None
    resource_count = total_banks + (1 if shared else lanes) + lanes
    if len(requests) * resource_count > _MAX_BATCH_RESOURCE_CELLS:
        # submit_many falls back to individually compiled requests, retaining
        # acceleration without allocating unbounded batch result matrices.
        return None
    ids = _resource_ids(config.addressable_stack_count, config.addressable_die_count, lanes, banks,
                        config.bank_groups_per_rank, config.banks_per_group,
                        bank_prefix, command_prefix, data_prefix, shared,
                        config.lanes_per_stack if config.stack_interface_mode == "independent" else 0)
    if len(set(ids)) != len(ids):
        return None
    read = request.operation is Operation.READ
    geometry = np.array([lanes, banks, (config.interleave_bytes or burst) // burst,
                         config.row_bytes // burst, config.rows_per_bank, config.addressable_die_count], dtype=np.int64)
    durations = np.array([
        burst, config.burst_interval_ns, burst / config.directional_lane_bandwidth_gb_s(request.operation),
        config.read_latency_ns if read else config.write_latency_ns,
        config.read_recovery_ns if read else config.write_recovery_ns,
        config.open_ns, config.close_ns, config.read_to_write_ns, config.write_to_read_ns,
    ], dtype=np.float64)
    count = len(ids)
    bankrows = np.full(total_banks, -1, dtype=np.int64)
    bankready = np.zeros(total_banks, dtype=np.float64)
    bankactivation = np.zeros(total_banks, dtype=np.float64)
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
                bank = core.__dict__["_banks"].get(rid)
                if bank is not None:
                    if bank.open_row is not None:
                        if (isinstance(bank.open_row, bool) or not isinstance(bank.open_row, int)
                                or not 0 <= bank.open_row <= _INT64_MAX):
                            return None
                        bankrows[index] = bank.open_row
                    bankready[index] = bank.ready_ns
                    bankactivation[index] = bank.activation_ready_ns
    except (ValueError, TypeError, OverflowError):
        return None
    if (not math.isfinite(request.arrival_ns)
            or not all(np.isfinite(values).all() for values in
                       (bankready, bankactivation, calendar, busy, last_start, last_end, durations))):
        return None
    if (not math.isfinite(core._acceptance_ns)
            or not all(math.isfinite(value) for value in core._inflight)
            or not all(math.isfinite(r.arrival_ns) for r in requests)):
        return None
    physical_total = sum(((r.address + r.byte_count - 1) // burst - r.address // burst + 1) * burst for r in requests)
    if any(value > _INT64_MAX - physical_total for value in moved):
        return None
    duration_pairs = np.array([[burst, config.burst_interval_ns,
        burst / config.directional_lane_bandwidth_gb_s(operation),
        config.read_latency_ns if operation is Operation.READ else config.write_latency_ns,
        config.read_recovery_ns if operation is Operation.READ else config.write_recovery_ns,
        config.open_ns, config.close_ns, config.read_to_write_ns, config.write_to_read_ns]
        for operation in (Operation.READ, Operation.WRITE)], dtype=np.float64)
    if not np.isfinite(duration_pairs).all():
        return None
    firsts = np.array([r.address // burst for r in requests], dtype=np.int64)
    stops = np.array([(r.address + r.byte_count - 1) // burst + 1 for r in requests], dtype=np.int64)
    arrivals = np.array([r.arrival_ns for r in requests], dtype=np.float64)
    reads = np.array([r.operation is Operation.READ for r in requests], dtype=np.bool_)
    inflight = np.empty(len(core._inflight) + len(requests), dtype=np.float64)
    inflight[:len(core._inflight)] = core._inflight
    try:
        from ._dram_numeric import run_numeric_many
    except (ImportError, OSError):
        # Match execute_compiled: packing is still local, so submit_many can
        # safely fall back to the ordered Python submissions without losing
        # admission, bank, calendar or accounting state.
        return None
    output = run_numeric_many(firsts, stops, arrivals, reads, config.max_outstanding_requests,
        float(core._acceptance_ns), inflight, len(core._inflight), geometry, duration_pairs, bool(shared),
        bankrows, bankready, bankactivation, banktouched, calendar, busy, moved, directions, last_start, last_end, touched)
    ends, waits, counts, busy_delta, byte_delta, starts_by_request, ends_by_request, touched_by_request, acceptance, inflight_count = output
    if (not np.isfinite(ends).all() or not all(np.isfinite(values).all()
            for values in (bankready, bankactivation, calendar, busy, last_start, last_end))):
        return None
    results = []
    for index, item in enumerate(requests):
        transfers = int(stops[index] - firsts[index]) * burst
        hits, misses, conflicts, skipped = (int(value) for value in counts[index])
        result_counters = {"row_hits": hits, "row_misses": misses, "row_conflicts": conflicts,
            "burst_count": int(stops[index] - firsts[index]),
            "details_truncated": int(stops[index] - firsts[index]) > config.max_expanded_segments,
            "compiled_batch_float64": True,
            "physical_read_bytes": transfers if reads[index] else 0,
            "physical_write_bytes": 0 if reads[index] else transfers,
            "queue_wait_ns": float(waits[index]),
            "bandwidth_ceiling_gb_s": config.directional_bandwidth_gb_s(item.operation),
            "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
            "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
            "resource_busy_ns": {ids[i]: float(busy_delta[index, i]) for i in np.flatnonzero(busy_delta[index])},
            "resource_bytes": {ids[i]: int(byte_delta[index, i]) for i in np.flatnonzero(byte_delta[index])},
            "resource_last_intervals": {ids[i]: (float(starts_by_request[index, i]), float(ends_by_request[index, i]))
                                        for i in np.flatnonzero(touched_by_request[index])},
            "resource_intervals": {}, "resource_interval_payloads": {}, "intervals_truncated": True}
        if stops[index] - firsts[index] >= 64:
            result_counters.update(accelerated_row_hit_bursts=skipped, compiled_float64=True)
        results.append(TransactionResult(request_id=item.request_id, operation=item.operation,
            arrival_ns=item.arrival_ns, completion_ns=float(ends[index]), logical_bytes=item.byte_count,
            transfer_bytes=transfers, counters=result_counters))
    # Commit only after every input/result has been checked, retaining the
    # caller's lane-calendar list identities and unrelated resource entries.
    for index in np.flatnonzero(banktouched):
        rid = ids[index]
        core._hbm_remember_bank(rid)
        bank = core.__dict__["_banks"].get(rid)
        if bank is None:
            bank = _BankState()
            core.__dict__["_banks"][rid] = bank
        bank.open_row, bank.ready_ns = int(bankrows[index]), float(bankready[index])
        bank.activation_ready_ns = float(bankactivation[index])
    first_data = total_banks + (1 if shared else lanes)
    all_touched = touched_by_request.any(axis=0)
    for index in np.flatnonzero(all_touched):
        rid = ids[index]
        timeline.lane_available.setdefault(rid, [0.0])[0] = float(calendar[index])
        timeline.lane_directions.pop(rid, None)
        timeline.ready_ns[rid] = float(calendar[index])
        timeline.busy_ns[rid] = float(busy[index])
        timeline.last_intervals[rid] = (float(last_start[index]), float(last_end[index]))
        if index >= first_data:
            timeline.bytes_moved[rid] = int(moved[index])
            timeline.directions[rid] = "read" if directions[index] == 1 else "write"
    timeline._touched = {ids[index] for index in np.flatnonzero(touched)}
    timeline._intervals = {}
    timeline._interval_payloads = {}
    timeline._interval_limit = timeline._interval_count = 0
    timeline._intervals_truncated = True
    core._acceptance_ns = float(acceptance)
    core._inflight = [float(value) for value in inflight[:inflight_count]]
    return tuple(results)


def owned_resource_footprint(core):
    """Closed native analytical-DRAM footprint for an outer undo journal.

    Its packed publisher writes dictionaries directly. The outer transaction
    records this small owner footprint before execution instead of copying
    every GPU's unrelated resource calendars for each HOST input chunk.
    """
    from .dram_core import DramCore, MAX_ACCELERATED_BANKS, _BankState
    from .memory_types import DramConfig, MemoryKind
    from ._nand_pack import can_own_timeline
    if (type(core) is not DramCore or type(core.config) is not DramConfig
            or DramConfig is not _OWNED_DRAM_CONFIG_TYPE
            or len(vars(DramConfig)) != len(_OWNED_DRAM_CONFIG_ITEMS)
            or any(vars(DramConfig).get(name) is not value for name, value in _OWNED_DRAM_CONFIG_ITEMS)):
        return None
    bank_class = vars(_BankState)
    slotnames = bank_class.get("__slotnames__")
    if (_BankState is not _OWNED_BANK_STATE_TYPE
            or slotnames is not None and (type(slotnames) is not list or slotnames)
            or len(bank_class) - ("__slotnames__" in bank_class) != len(_OWNED_BANK_STATE_ITEMS)
            or any(vars(_BankState).get(name) is not value for name, value in _OWNED_BANK_STATE_ITEMS)):
        return None
    banks = vars(core).get("_banks")
    if type(banks) is not dict:
        return None
    if vars(core).get("_analytical_banks_exposed"):
        # A retained public mapping can bind even an ordinary native bank
        # object to a sibling Core. Its fields alone cannot prove exclusive
        # ownership, so public bank state keeps the complete owner undo.
        return None
    config = core.config
    if (any(type(value) not in (int, float, str, bool, type(None), MemoryKind)
            for key, value in vars(config).items() if key != "metadata")
            or config.control_mode == "jedec_hbm" or core.capture_details):
        return None
    if any(globals().get(name) is not value for name, value in _OWNED_NATIVE_FUNCTIONS):
        return None
    if any(getattr(_owned_numeric, name) is not value for name, value in _OWNED_NATIVE_NUMERIC):
        return None
    if (type(_NumericWorkspace) is not type or _NumericWorkspace is not _OWNED_WORKSPACE_TYPE
            or len(vars(_NumericWorkspace)) != len(_OWNED_WORKSPACE_CLASS_ITEMS)
            or any(vars(_NumericWorkspace).get(name) is not value
                   for name, value in _OWNED_WORKSPACE_CLASS_ITEMS)
            or _NumericWorkspace.__bases__ != (object,)
            or _WORKSPACES is not _OWNED_WORKSPACE_CACHE
            or type(_WORKSPACES) is not WeakKeyDictionary
            or any(vars(WeakKeyDictionary).get(name) is not value
                   for name, value in _OWNED_WORKSPACE_CACHE_METHODS)
            or vars(DramCore).get("__hash__", object.__hash__) is not object.__hash__
            or vars(DramCore).get("__eq__", object.__eq__) is not object.__eq__):
        return None
    workspace = _WORKSPACES.get(core)
    if workspace is not None:
        # Cached scratch is replaceable Python state. A caller's property,
        # ndarray protocol or object-dtype conversion can touch a foreign
        # owner, so such execution keeps the original complete journal.
        if type(workspace) is not _NumericWorkspace:
            return None
        values = vars(workspace)
        if (set(values) != {"shape", "floats", "arrays"}
                or type(values["shape"]) is not tuple or len(values["shape"]) != 2
                or any(type(value) is not int for value in values["shape"])
                or type(values["floats"]) is not np.ndarray
                or values["floats"].dtype != np.dtype(np.float64)
                or type(values["arrays"]) is not tuple or len(values["arrays"]) != 11
                or any(type(array) is not np.ndarray or array.dtype not in (
                       np.dtype(np.float64), np.dtype(np.int64), np.dtype(np.int8), np.dtype(np.bool_))
                       for array in values["arrays"])):
            return None
    config = core.config
    if type(config.metadata) is not dict or any(type(key) is not str for key in config.metadata):
        return None
    if config.addressable_stack_count*config.addressable_die_count*config.lane_count*config.bank_count > MAX_ACCELERATED_BANKS:
        return None
    prefixes = tuple(config.metadata.get(name, default) for name, default in (
        ("bank_resource_prefix", "dram:bank"), ("command_resource_prefix", "dram:command"),
        ("data_resource_prefix", "dram:data")))
    shared = config.metadata.get("shared_command_resource")
    if any(type(value) is not str for value in prefixes) or shared is not None and type(shared) is not str:
        return None
    ids = _resource_ids(config.addressable_stack_count, config.addressable_die_count,
        config.lane_count, config.bank_count, config.bank_groups_per_rank, config.banks_per_group,
        *prefixes, shared or None,
        config.lanes_per_stack if config.stack_interface_mode == "independent" else 0)
    resources = tuple(dict.fromkeys(core.timeline.canonical_resource_id(rid) for rid in ids))
    return resources if can_own_timeline(core.timeline, resources) else None


from . import _dram_numeric as _owned_numeric
from .memory_types import DramConfig as _OWNED_DRAM_CONFIG_TYPE
_OWNED_DRAM_CONFIG_ITEMS = tuple(vars(_OWNED_DRAM_CONFIG_TYPE).items())
from .dram_core import _BankState as _OWNED_BANK_STATE_TYPE
_OWNED_BANK_STATE_ITEMS = tuple((name, value) for name, value in vars(_OWNED_BANK_STATE_TYPE).items()
                              if name != "__slotnames__")
_OWNED_NATIVE_NUMERIC = tuple((name, value) for name, value in vars(_owned_numeric).items()
    if callable(value) and getattr(value, "__module__", None) == _owned_numeric.__name__)
_OWNED_NATIVE_FUNCTIONS = tuple((name, value) for name, value in tuple(globals().items())
    if callable(value) and getattr(value, "__module__", None) == __name__ and name != "owned_resource_footprint")
_OWNED_WORKSPACE_TYPE = _NumericWorkspace
_OWNED_WORKSPACE_CLASS_ITEMS = tuple(vars(_NumericWorkspace).items())
_OWNED_WORKSPACE_CACHE = _WORKSPACES
_OWNED_WORKSPACE_CACHE_METHODS = tuple((name, getattr(WeakKeyDictionary, name))
    for name in ("get", "__setitem__"))
