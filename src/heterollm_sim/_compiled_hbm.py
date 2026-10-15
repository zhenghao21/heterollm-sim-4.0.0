"""Pack the JEDEC HBM controller state for its separate compiled executor."""
from __future__ import annotations

import math
import numpy as np

from .memory_types import Operation, TransactionResult
from .memory_mapping import map_dram_address


# Cover the 4 TiB, 64-stack HBM reference without raising the analytical
# controller's separate limit. The three bank arrays occupy 9 MiB here;
# batch checkpoints and Python bank mappings require additional memory.
MAX_HBM_NUMERIC_BANKS = 131_072


def _pack_hbm_numeric(core, request):
    """Return packed state, or None for unsupported acceleration cases.

    The detail controller remains the reference for tracing, aliased resources,
    and nonzero tFAW. A decline never selects the analytical DRAM timing model.
    """
    from .dram_core import DramCore
    from .memory_transfer import ResourceTimeline

    c, timeline = core.config, core.timeline
    if (type(core) is not DramCore or type(timeline) is not ResourceTimeline
            or core.capture_details or timeline._interval_limit > 0
            or c.four_activate_window_ns > 0
            or c.control_mode != "jedec_hbm" or c.ranks_per_channel != 1
            or c.pseudo_channels_per_channel != 2 or c.subchannels_per_channel != 1
            or c.addressable_stack_count != 1 or c.addressable_die_count != 1
            or c.stack_interface_mode != "independent"
            or request.operation not in (Operation.READ, Operation.WRITE)):
        return None
    lanes, banks, groups = c.lane_count, c.bank_count, c.bank_groups_per_rank
    if 16 % c.banks_per_group and c.banks_per_group % 16:
        return None
    history_stride = min(16, c.banks_per_group)
    history_entries = banks // history_stride
    channels = lanes // 2
    total_banks = lanes * banks
    first = request.address // c.burst_bytes
    stop = (request.address + request.byte_count - 1) // c.burst_bytes + 1
    physical = (stop - first) * c.burst_bytes
    if (total_banks > MAX_HBM_NUMERIC_BANKS or request.address < 0
            or request.address + request.byte_count > c.effective_capacity_bytes
            or c.effective_capacity_bytes > (1 << 63) - 1 or physical > (1 << 63) - 1
            or stop - first > 2 ** 53 or not math.isfinite(request.arrival_ns)):
        return None
    try:
        from ._hbm_numeric import TIMING_FIELDS, run_hbm_numeric
    except (ImportError, OSError):
        return None
    metadata = c.metadata
    prefix = str(metadata.get("bank_resource_prefix", "dram:bank"))
    row_prefix = str(metadata.get("row_command_resource_prefix", "dram:row_command"))
    col_prefix = str(metadata.get("column_command_resource_prefix", "dram:column_command"))
    data_prefix = str(metadata.get("data_resource_prefix", "dram:data"))
    channel_keys = [(ch // c.channels, ch % c.channels) for ch in range(channels)]
    pc_keys = [(*channel_keys[lane // 2], lane % 2) for lane in range(lanes)]
    ids = ([f"{row_prefix}:{stack}:{channel}" for stack, channel in channel_keys]
           + [f"{col_prefix}:{stack}:{channel}" for stack, channel in channel_keys]
           + [f"{data_prefix}:{lane}" for lane in range(lanes)])
    if len(set(ids)) != len(ids):
        return None
    # The event kernel also contains ownership aliases for unrelated
    # resources. Only aliases affecting this controller require fallback.
    if any(timeline.canonical_resource_id(rid) != rid for rid in ids):
        return None
    bank_ids = [f"{prefix}:{lane // c.lanes_per_stack}:0:{lane}:0:{bank // c.banks_per_group}:{bank % c.banks_per_group}"
                for lane in range(lanes) for bank in range(banks)]
    bank_lookup = {rid: index for index, rid in enumerate(bank_ids)}
    state = core._hbm_control_state
    rows = np.full(total_banks, -1, dtype=np.int64)
    bf = np.zeros((total_banks, 7), dtype=np.float64)
    bf[:, 0] = -1
    addresses = np.full(total_banks, -1, dtype=np.int64)
    pc = np.zeros((lanes, 4), dtype=np.float64)
    pc[:, 0] = -1
    history = np.full((lanes, history_entries, 4), -1., dtype=np.float64)
    pc_seen = np.zeros(lanes, dtype=np.bool_)
    channel = np.full((channels, 2), -1., dtype=np.float64)
    channel_int = np.full((channels, 2), -1, dtype=np.int64)
    resources = np.zeros((len(ids), 4), dtype=np.float64)
    moved = np.zeros(len(ids), dtype=np.int64)
    directions = np.zeros(len(ids), dtype=np.int8)
    touched = np.zeros(len(ids), dtype=np.bool_)
    live_slots = set()
    try:
        for index, rid in enumerate(ids):
            slots = timeline.lane_available.get(rid)
            if slots is not None:
                if len(slots) != 1 or id(slots) in live_slots:
                    return None
                live_slots.add(id(slots))
            resources[index, 0] = timeline.available_ns(rid)
            resources[index, 1] = timeline.busy_ns.get(rid, 0.)
            resources[index, 2:] = timeline.last_intervals.get(rid, (0., 0.))
            previous_bytes = timeline.bytes_moved.get(rid, 0)
            if not isinstance(previous_bytes, (int, np.integer)) or not 0 <= previous_bytes <= (1 << 63) - 1 - physical:
                return None
            moved[index] = previous_bytes
            direction = timeline.directions.get(rid)
            directions[index] = 1 if direction == "read" else 2 if direction == "write" else 3 if direction else 0
        for index, rid in enumerate(bank_ids):
            old = core._banks.get(rid)
            if old is not None:
                rows[index] = old.open_row if old.open_row is not None else -1
                bf[index, 5], bf[index, 6] = old.ready_ns, old.activation_ready_ns
            bank = state.get("banks", {}).get(rid)
            if bank is not None:
                addresses[index] = bank["mapping"].address
                bf[index, :5] = [bank["last_activate"] if bank["last_activate"] is not None else -1.,
                                bank["activate_ready"], bank["precharge_ready"], bank["read_ready"], bank["write_ready"]]
        for lane, key in enumerate(pc_keys):
            previous = state.get("pcs", {}).get(key)
            if previous is None:
                continue
            pc_seen[lane] = True
            pc[lane] = [previous["last_precharge"] if previous["last_precharge"] is not None else -1.,
                        previous["blocked_until"], previous["next_data_start"]["read"], previous["next_data_start"]["write"]]
            for group, value in previous["activate"].items():
                history[lane, group, 0] = value
            for which, entries in ((1, previous["column"]["read"]),
                                   (2, previous["column"]["write"]), (3, previous["write_end"])):
                for (sid, group), value in entries.items():
                    entry = max(sid * 16, group * c.banks_per_group) // history_stride
                    if (not 0 <= entry < history_entries or entry * history_stride // 16 != sid
                            or entry * history_stride // c.banks_per_group != group):
                        return None
                    history[lane, entry, which] = value
        for ch, key in enumerate(channel_keys):
            previous = state.get("channels", {}).get(key, {})
            channel[ch] = [previous.get("last_activate_start", -1.), previous.get("next_refresh", -1.)]
            last_bank = previous.get("last_activate_bank")
            if last_bank is not None:
                channel_int[ch, 0] = bank_lookup[last_bank]
            last_pc = previous.get("last_activate_pc")
            if last_pc is not None:
                channel_int[ch, 1] = pc_keys.index(last_pc)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    g = np.array([lanes, banks, (c.interleave_bytes or c.burst_bytes) // c.burst_bytes,
                  c.row_bytes // c.burst_bytes, c.banks_per_group, groups,
                  -1 if c.activated_row_msb is None else c.activated_row_msb,
                  c.generation in {"HBM2", "HBM2E"}, c.refresh_policy == "all_bank", c.burst_bytes,
                  history_stride], dtype=np.int64)
    t = np.array([getattr(c, field) for field in TIMING_FIELDS] + [
        c.burst_bytes / c.effective_lane_bandwidth_gb_s,
        c.burst_bytes / c.directional_lane_bandwidth_gb_s(Operation.READ),
        c.burst_bytes / c.directional_lane_bandwidth_gb_s(Operation.WRITE),
        c.maximum_active_ns], dtype=np.float64)
    if not all(np.isfinite(a).all() for a in (bf, pc, history, channel, resources, t)):
        return None
    return _HBMNumericState(core, bank_ids, pc_keys, channel_keys, ids, g, t,
        run_hbm_numeric, (rows, bf, addresses, pc, history, pc_seen, channel,
                          channel_int, resources, moved, directions, touched))


class _HBMNumericState:
    """One packed controller; requests still execute in submission order."""

    def __init__(self, core, bank_ids, pc_keys, channel_keys, ids, g, t, run_numeric, arrays):
        self.core = core
        self.bank_ids, self.pc_keys, self.channel_keys, self.ids = bank_ids, pc_keys, channel_keys, ids
        self.g, self.t, self.run_numeric, self.arrays = g, t, run_numeric, arrays

    def execute(self, request):
        c = self.core.config
        first = request.address // c.burst_bytes
        stop = (request.address + request.byte_count - 1) // c.burst_bytes + 1
        physical = (stop - first) * c.burst_bytes
        (rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
         resources, moved, directions, touched) = self.arrays
        if (request.operation not in (Operation.READ, Operation.WRITE)
                or request.address < 0 or request.address + request.byte_count > c.effective_capacity_bytes
                or physical > (1 << 63) - 1 or stop - first > 2 ** 53
                or not math.isfinite(request.arrival_ns)
                or any(int(value) > (1 << 63) - 1 - physical for value in moved)):
            return None
        touched.fill(False)
        g, t, run_hbm_numeric = self.g, self.t, self.run_numeric
        read = request.operation is Operation.READ
        completion, hits, misses, conflicts, counts = run_hbm_numeric(
            first, stop, float(request.arrival_ns), read, g, t, rows, bf, addresses,
            pc, history, pc_seen, channel, channel_int, resources, moved, directions, touched)
        if not math.isfinite(completion) or not all(np.isfinite(a).all() for a in (bf, pc, history, channel, resources)):
            return None
        counters = dict(activate_commands=int(counts[0]), precharge_commands=int(counts[1]),
                        read_commands=int(counts[2]), write_commands=int(counts[3]),
                        row_command_bus_busy_ns=float(counts[5]), column_command_bus_busy_ns=float(counts[6]),
                        control_mode="jedec_hbm", row_hits=int(hits), row_misses=int(misses),
                        row_conflicts=int(conflicts), burst_count=stop - first,
                        details_truncated=stop - first > c.max_expanded_segments,
                        compiled_hbm_float64=True,
                        physical_read_bytes=physical if read else 0, physical_write_bytes=0 if read else physical,
                        queue_wait_ns=0., bandwidth_ceiling_gb_s=c.directional_bandwidth_gb_s(request.operation),
                        host_transfer_bytes=0, internal_transfer_bytes=0,
                        pages_read=0, pages_programmed=0, erase_operations=0)
        if c.refresh_policy == "all_bank":
            counters["refresh_commands"] = int(counts[4])
        return TransactionResult(request.request_id, request.operation, request.arrival_ns,
                                 float(completion), request.byte_count, physical, counters=counters)

    def commit_controller(self):
        from .dram_core import _BankState

        core, c = self.core, self.core.config
        state = core._hbm_control_state
        bank_ids, pc_keys, channel_keys = self.bank_ids, self.pc_keys, self.channel_keys
        history_stride = int(self.g[-1])
        (rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
         resources, moved, directions, touched) = self.arrays
        saved_banks = state.setdefault("banks", {})
        saved_pcs = state.setdefault("pcs", {})
        saved_channels = state.setdefault("channels", {})
        for index in np.flatnonzero(addresses >= 0):
            rid = bank_ids[index]
            old = core._banks.setdefault(rid, _BankState())
            old.open_row = int(rows[index]) if rows[index] >= 0 else None
            old.ready_ns, old.activation_ready_ns = float(bf[index, 5]), float(bf[index, 6])
            if rid not in saved_banks:
                saved_banks[rid] = {"mapping": map_dram_address(c, int(addresses[index]))}
            previous = saved_banks[rid]
            previous.update(last_activate=float(bf[index, 0]) if bf[index, 0] >= 0 else None,
                            activate_ready=float(bf[index, 1]), precharge_ready=float(bf[index, 2]),
                            read_ready=float(bf[index, 3]), write_ready=float(bf[index, 4]))
        for lane in np.flatnonzero(pc_seen):
            previous = saved_pcs.setdefault(pc_keys[lane], {"activate_window": []})
            previous.update(activate={int(group): float(history[lane, group, 0])
                                      for group in np.flatnonzero(history[lane, :, 0] >= 0)},
                            column={op: {(int(entry) * history_stride // 16, int(entry) * history_stride // c.banks_per_group): float(history[lane, entry, index])
                                         for entry in np.flatnonzero(history[lane, :, index] >= 0)}
                                    for op, index in (("read", 1), ("write", 2))},
                            write_end={(int(entry) * history_stride // 16, int(entry) * history_stride // c.banks_per_group): float(history[lane, entry, 3])
                                       for entry in np.flatnonzero(history[lane, :, 3] >= 0)},
                            last_precharge=float(pc[lane, 0]) if pc[lane, 0] >= 0 else None,
                            blocked_until=float(pc[lane, 1]),
                            next_data_start={"read": float(pc[lane, 2]), "write": float(pc[lane, 3])})
        for ch, key in enumerate(channel_keys):
            if channel[ch, 0] >= 0 or channel[ch, 1] >= 0:
                previous = saved_channels.setdefault(key, {})
                if channel[ch, 0] >= 0:
                    previous["last_activate_start"] = float(channel[ch, 0])
                    previous["last_activate_bank"] = bank_ids[channel_int[ch, 0]]
                    previous["last_activate_pc"] = pc_keys[channel_int[ch, 1]]
                if channel[ch, 1] >= 0:
                    previous["next_refresh"] = float(channel[ch, 1])

    def commit_timeline(self, request):
        timeline, ids = self.core.timeline, self.ids
        channels = len(self.channel_keys)
        (rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
         resources, moved, directions, touched) = self.arrays
        for index in np.flatnonzero(touched):
            rid = ids[index]
            timeline.lane_available.setdefault(rid, [0.])[0] = float(resources[index, 0])
            timeline.ready_ns[rid] = float(resources[index, 0])
            timeline.busy_ns[rid] = float(resources[index, 1])
            timeline.last_intervals[rid] = (float(resources[index, 2]), float(resources[index, 3]))
            timeline.lane_directions.pop(rid, None)
            timeline._touched.add(rid)
            if index >= 2 * channels:
                timeline.bytes_moved[rid] = int(moved[index])
                timeline.directions[rid] = request.operation.value
        timeline._intervals_truncated = True


def execute_hbm_numeric(core, request):
    """Execute one accepted request, or decline without changing live state."""
    packed = _pack_hbm_numeric(core, request)
    if packed is None:
        return None
    result = packed.execute(request)
    if result is not None:
        packed.commit_controller()
        packed.commit_timeline(request)
    return result


class _NumericDeclined(Exception):
    pass


def execute_hbm_numeric_many(core, requests):
    """Amortize packing while retaining each request's admission and metrics."""
    if not requests:
        return ()
    packed = _pack_hbm_numeric(core, requests[0])
    if packed is None:
        return None

    def execute(request):
        result = packed.execute(request)
        if result is None:
            raise _NumericDeclined
        packed.commit_timeline(request)
        return result

    results = []
    for index, request in enumerate(requests):
        # Numeric buffers are contiguous and cheap to copy. Keep the last
        # successful state so an exception or decline cannot commit half of
        # the current request or discard requests already accepted in order.
        before = tuple(array.copy() for array in packed.arrays)
        try:
            result = core._submit_validated(request, execute=execute)
        except _NumericDeclined:
            packed.arrays = before
            if results:
                packed.commit_controller()
            return tuple(results) + tuple(core.submit(item) for item in requests[index:])
        except BaseException:
            packed.arrays = before
            if results:
                packed.commit_controller()
            raise
        results.append(result)
    packed.commit_controller()
    return tuple(results)
