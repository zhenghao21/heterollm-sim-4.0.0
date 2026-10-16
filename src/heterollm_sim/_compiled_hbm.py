"""Pack the JEDEC HBM controller state for its separate compiled executor."""
from __future__ import annotations

import math
import copy
from functools import lru_cache
from types import MappingProxyType
import numpy as np

from .memory_types import Operation, TransactionResult
from .memory_mapping import map_dram_address


# Cover the 4 TiB, 64-stack HBM reference without raising the analytical
# controller's separate limit. The three bank arrays occupy 9 MiB here;
# batch checkpoints and Python bank mappings require additional memory.
MAX_HBM_NUMERIC_BANKS = 131_072


@lru_cache(maxsize=8)
def _numeric_resource_layout(lanes, banks, channels_per_stack, lanes_per_stack,
                             banks_per_group, prefix, row_prefix, col_prefix, data_prefix):
    """Cache only names derived from geometry, never live controller state."""
    channel_keys = tuple((ch // channels_per_stack, ch % channels_per_stack)
                         for ch in range(lanes // 2))
    pc_keys = tuple((*channel_keys[lane // 2], lane % 2) for lane in range(lanes))
    ids = (tuple(f"{row_prefix}:{stack}:{channel}" for stack, channel in channel_keys)
           + tuple(f"{col_prefix}:{stack}:{channel}" for stack, channel in channel_keys)
           + tuple(f"{data_prefix}:{lane}" for lane in range(lanes)))
    bank_ids = tuple(
        f"{prefix}:{lane // lanes_per_stack}:0:{lane}:0:{bank // banks_per_group}:{bank % banks_per_group}"
        for lane in range(lanes) for bank in range(banks))
    return (channel_keys, pc_keys, ids, bank_ids,
            MappingProxyType({rid: index for index, rid in enumerate(bank_ids)}),
            MappingProxyType({key: index for index, key in enumerate(pc_keys)}))


def _pack_hbm_numeric(core, request):
    """Return packed state, or None for unsupported acceleration cases.

    The detail controller remains the reference for tracing and aliased
    resources. A decline never selects the analytical DRAM timing model.
    """
    from .dram_core import DramCore
    from .memory_transfer import ResourceTimeline

    c, timeline = core.config, core.timeline
    if (type(core) is not DramCore or type(timeline) is not ResourceTimeline
            or core.capture_details or timeline._interval_limit > 0
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
    # The parallel traversal strides one complete address stripe. Validate in
    # Python integers before constructing the int64 geometry vector; a tiny
    # valid request may still have an authored stripe wider than int64.
    if (c.address_mapping != "row_local"
            and lanes * ((c.interleave_bytes or c.burst_bytes) // c.burst_bytes) > (1 << 63) - 1):
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
    channel_keys, pc_keys, ids, bank_ids, bank_lookup, pc_lookup = _numeric_resource_layout(
        lanes, banks, c.channels, c.lanes_per_stack, c.banks_per_group,
        prefix, row_prefix, col_prefix, data_prefix)
    if len(set(ids)) != len(ids):
        return None
    # The event kernel also contains ownership aliases for unrelated
    # resources. Only aliases affecting this controller require fallback.
    if any(timeline.canonical_resource_id(rid) != rid for rid in ids):
        return None
    cached = core._hbm_numeric_state
    if cached is not None:
        if c != cached._config:
            core._hbm_publish_numeric()
            return _pack_hbm_numeric(core, request)
        if cached.refresh_resources(physical):
            return cached
        return None
    state = core.__dict__["_hbm_control_state"]
    rows = np.full(total_banks, -1, dtype=np.int64)
    bf = np.zeros((total_banks, 7), dtype=np.float64)
    bf[:, 0] = -1
    addresses = np.full(total_banks, -1, dtype=np.int64)
    pc = np.zeros((lanes, 4), dtype=np.float64)
    pc[:, 0] = -1
    history = np.full((lanes, history_entries, 4), -1., dtype=np.float64)
    pc_seen = np.zeros(lanes, dtype=np.bool_)
    # At most four unexpired past ACTs plus the existing future reservations.
    # Pruning uses the monotonic shared row-port frontier, never a speculative
    # candidate on one bank. No storage is allocated in proportion to bursts.
    window_capacity = max((len(value["activate_window"])
                           for value in state.get("pcs", {}).values()), default=0) + 5
    windows = np.zeros((lanes, window_capacity), dtype=np.float64)
    window_counts = np.zeros(lanes, dtype=np.int64)
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
        # Convert each numeric column in one pass. Avoid both NumPy scalar
        # assignment per bank and thousands of temporary bank-record tuples
        # that trigger cyclic-GC scans of an otherwise unchanged controller.
        live_banks = [core.__dict__["_banks"].get(rid) for rid in bank_ids]
        rows[:] = np.fromiter((old.open_row if old is not None and old.open_row is not None else -1
                              for old in live_banks), dtype=np.int64, count=total_banks)
        bf[:, 5] = np.fromiter((old.ready_ns if old is not None else 0. for old in live_banks),
                              dtype=np.float64, count=total_banks)
        bf[:, 6] = np.fromiter((old.activation_ready_ns if old is not None else 0. for old in live_banks),
                              dtype=np.float64, count=total_banks)
        saved_banks = state.get("banks", {})
        controller_banks = [saved_banks.get(rid) for rid in bank_ids]
        addresses[:] = np.fromiter((bank["mapping"].address if bank is not None else -1
                                   for bank in controller_banks), dtype=np.int64, count=total_banks)
        for column, field in enumerate(("last_activate", "activate_ready", "precharge_ready", "read_ready", "write_ready")):
            default = -1. if column == 0 else 0.
            bf[:, column] = np.fromiter((bank[field] if bank is not None and bank[field] is not None else default
                                        for bank in controller_banks), dtype=np.float64, count=total_banks)
        for lane, key in enumerate(pc_keys):
            previous = state.get("pcs", {}).get(key)
            if previous is None:
                continue
            pc_seen[lane] = True
            calendar = previous["activate_window"]
            if any(not math.isfinite(value) or value < 0 for value in calendar) or calendar != sorted(calendar):
                return None
            window_counts[lane] = len(calendar)
            windows[lane, :len(calendar)] = calendar
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
                channel_int[ch, 1] = pc_lookup[last_pc]
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    g = np.array([lanes, banks, (c.interleave_bytes or c.burst_bytes) // c.burst_bytes,
                  c.row_bytes // c.burst_bytes, c.banks_per_group, groups,
                  -1 if c.activated_row_msb is None else c.activated_row_msb,
                  c.generation in {"HBM2", "HBM2E"}, c.refresh_policy == "all_bank", c.burst_bytes,
                  history_stride, c.address_mapping == "row_local"], dtype=np.int64)
    t = np.array([getattr(c, field) for field in TIMING_FIELDS] + [
        c.burst_bytes / c.effective_lane_bandwidth_gb_s,
        c.burst_bytes / c.directional_lane_bandwidth_gb_s(Operation.READ),
        c.burst_bytes / c.directional_lane_bandwidth_gb_s(Operation.WRITE),
        c.maximum_active_ns, c.four_activate_window_ns], dtype=np.float64)
    if not all(np.isfinite(a).all() for a in (bf, pc, history, channel, resources, t, windows)):
        return None
    packed = _HBMNumericState(core, bank_ids, pc_keys, channel_keys, ids, g, t,
        run_hbm_numeric, (rows, bf, addresses, pc, history, pc_seen, channel,
                          channel_int, resources, moved, directions, touched, windows, window_counts))
    if core._hbm_numeric_enabled:
        core._hbm_numeric_state = packed
        for token in core._hbm_transactions:
            token.capture_numeric(packed)
    return packed


class _HBMNumericState:
    """One packed controller; requests still execute in submission order."""

    def __init__(self, core, bank_ids, pc_keys, channel_keys, ids, g, t, run_numeric, arrays):
        self.core = core
        self.bank_ids, self.pc_keys, self.channel_keys, self.ids = bank_ids, pc_keys, channel_keys, ids
        self.g, self.t, self.run_numeric, self.arrays = g, t, run_numeric, arrays
        self._controller_before = self._copy_controller_arrays()
        self._controller_commits = 0
        self._config = copy.deepcopy(core.config)

    def refresh_resources(self, physical):
        """External link reservations remain authoritative on the timeline."""
        timeline = self.core.timeline
        resources, moved, directions = self.arrays[8:11]
        ids = self.ids
        slots = [timeline.lane_available.get(rid) for rid in ids]
        present = [slot for slot in slots if slot is not None]
        if any(len(slot) != 1 for slot in present) or len({id(slot) for slot in present}) != len(present):
            return False
        previous_bytes = [timeline.bytes_moved.get(rid, 0) for rid in ids]
        limit = (1 << 63) - 1 - physical
        if any(not isinstance(value, (int, np.integer)) or not 0 <= value <= limit for value in previous_bytes):
            return False
        # Bulk column conversion avoids four scalar NumPy writes per port.
        # Alias validation above proves each present calendar has one slot,
        # so this is exactly ResourceTimeline.available_ns for these ids.
        resources[:, 0] = np.fromiter((float(slot[0]) if slot is not None else float(timeline.ready_ns.get(rid, 0.))
                                      for rid, slot in zip(ids, slots)), dtype=np.float64, count=len(ids))
        resources[:, 1] = np.fromiter((timeline.busy_ns.get(rid, 0.) for rid in ids),
                                      dtype=np.float64, count=len(ids))
        resources[:, 2:] = [timeline.last_intervals.get(rid, (0., 0.)) for rid in ids]
        moved[:] = previous_bytes
        direction_values = [timeline.directions.get(rid) for rid in ids]
        directions[:] = [1 if value == "read" else 2 if value == "write" else 3 if value else 0
                         for value in direction_values]
        return bool(np.isfinite(resources).all())

    def _copy_controller_arrays(self):
        # Baseline of the currently published Python objects. A resident core
        # retains it across submissions until that state is exposed publicly.
        return tuple(self.arrays[index].copy() for index in (0, 1, 2, 6, 7))

    def execute(self, request):
        c = self.core.config
        first = request.address // c.burst_bytes
        stop = (request.address + request.byte_count - 1) // c.burst_bytes + 1
        physical = (stop - first) * c.burst_bytes
        (rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
         resources, moved, directions, touched, windows, window_counts) = self.arrays
        if (request.operation not in (Operation.READ, Operation.WRITE)
                or request.address < 0 or request.address + request.byte_count > c.effective_capacity_bytes
                or physical > (1 << 63) - 1 or stop - first > 2 ** 53
                or not math.isfinite(request.arrival_ns)
                or any(int(value) > (1 << 63) - 1 - physical for value in moved)):
            return None
        touched.fill(False)
        g, t, run_hbm_numeric = self.g, self.t, self.run_numeric
        read = request.operation is Operation.READ
        try:
            completion, hits, misses, conflicts, counts = run_hbm_numeric(
                first, stop, float(request.arrival_ns), read, g, t, rows, bf, addresses,
                pc, history, pc_seen, channel, channel_int, resources, moved, directions, touched,
                windows, window_counts)
        except OverflowError:
            # Integer clock indices exceeded the compiled safe range. These
            # private arrays are discarded (or rolled back by the batch
            # caller); the Python controller uses unbounded integer indices.
            return None
        if not math.isfinite(completion) or not all(np.isfinite(a).all() for a in (bf, pc, history, channel, resources, windows)):
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

        core, c = self.core, self._config
        state = core.__dict__["_hbm_control_state"]
        live_banks = core.__dict__["_banks"]
        bank_ids, pc_keys, channel_keys = self.bank_ids, self.pc_keys, self.channel_keys
        history_stride = int(self.g[10])
        (rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
         resources, moved, directions, touched, windows, window_counts) = self.arrays
        old_rows, old_bf, old_addresses, old_channel, old_channel_int = self._controller_before
        # Compare float bits rather than tolerances, including signed zero.
        changed_banks = ((rows != old_rows) | (addresses != old_addresses)
                         | np.any(bf.view(np.uint64) != old_bf.view(np.uint64), axis=1))
        changed_channels = (np.any(channel.view(np.uint64) != old_channel.view(np.uint64), axis=1)
                            | np.any(channel_int != old_channel_int, axis=1))
        saved_banks = state.setdefault("banks", {})
        saved_pcs = state.setdefault("pcs", {})
        saved_channels = state.setdefault("channels", {})
        # Preserve materialization of any pre-existing control-only bank, too.
        # Normally both dictionaries are populated together by the controller.
        for index in np.flatnonzero(addresses >= 0):
            if bank_ids[index] not in live_banks:
                changed_banks[index] = True
        bank_indices = np.flatnonzero((addresses >= 0) & changed_banks)
        # Converting the selected rows in C avoids many individual NumPy scalar
        # conversions while preserving the same Python int/float state values.
        bank_rows = rows[bank_indices].tolist()
        bank_times = bf[bank_indices].tolist()
        for index, row, values in zip(bank_indices, bank_rows, bank_times):
            rid = bank_ids[index]
            core._hbm_remember_bank(rid)
            core._hbm_remember_entry("banks", rid)
            old = live_banks.get(rid)
            if old is None:
                old = live_banks.setdefault(rid, _BankState())
            old.open_row = row if row >= 0 else None
            old.ready_ns, old.activation_ready_ns = values[5], values[6]
            if rid not in saved_banks:
                saved_banks[rid] = {"mapping": map_dram_address(c, int(addresses[index]))}
            previous = saved_banks[rid]
            previous.update(last_activate=values[0] if values[0] >= 0 else None,
                            activate_ready=values[1], precharge_ready=values[2],
                            read_ready=values[3], write_ready=values[4])
        # Rebuild every seen PC's nested dictionaries just as the original
        # controller did. Extension fields may retain aliases to old histories;
        # skipping an unchanged PC would incorrectly keep those aliases live.
        pc_indices = np.flatnonzero(pc_seen)
        pc_values = pc[pc_indices].tolist()
        history_values = history[pc_indices].tolist()
        history_keys = tuple((entry * history_stride // 16,
                              entry * history_stride // c.banks_per_group)
                             for entry in range(history.shape[1]))
        for lane, values, entries in zip(pc_indices, pc_values, history_values):
            core._hbm_remember_entry("pcs", pc_keys[lane])
            previous = saved_pcs.setdefault(pc_keys[lane], {"activate_window": []})
            previous["activate_window"] = windows[lane, :window_counts[lane]].tolist()
            previous.update(activate={group: entry[0] for group, entry in enumerate(entries)
                                      if entry[0] >= 0},
                            column={op: {key: entry[index] for key, entry in zip(history_keys, entries)
                                         if entry[index] >= 0}
                                    for op, index in (("read", 1), ("write", 2))},
                            write_end={key: entry[3] for key, entry in zip(history_keys, entries)
                                       if entry[3] >= 0},
                            last_precharge=values[0] if values[0] >= 0 else None,
                            blocked_until=values[1],
                            next_data_start={"read": values[2], "write": values[3]})
        for ch in np.flatnonzero(changed_channels):
            key = channel_keys[ch]
            if channel[ch, 0] >= 0 or channel[ch, 1] >= 0:
                core._hbm_remember_entry("channels", key)
                previous = saved_channels.setdefault(key, {})
                if channel[ch, 0] >= 0:
                    previous["last_activate_start"] = float(channel[ch, 0])
                    previous["last_activate_bank"] = bank_ids[channel_int[ch, 0]]
                    previous["last_activate_pc"] = pc_keys[channel_int[ch, 1]]
                if channel[ch, 1] >= 0:
                    previous["next_refresh"] = float(channel[ch, 1])
        self._controller_before = self._copy_controller_arrays()

    def commit_timeline(self, request):
        timeline, ids = self.core.timeline, self.ids
        channels = len(self.channel_keys)
        (rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
         resources, moved, directions, touched, windows, window_counts) = self.arrays
        indices = np.flatnonzero(touched)
        values = resources[indices].tolist()
        transfer_bytes = moved[indices].tolist()
        for index, resource, byte_count in zip(indices, values, transfer_bytes):
            rid = ids[index]
            timeline.lane_available.setdefault(rid, [0.])[0] = resource[0]
            timeline.ready_ns[rid] = resource[0]
            timeline.busy_ns[rid] = resource[1]
            timeline.last_intervals[rid] = (resource[2], resource[3])
            timeline.lane_directions.pop(rid, None)
            timeline._touched.add(rid)
            if index >= 2 * channels:
                timeline.bytes_moved[rid] = byte_count
                timeline.directions[rid] = request.operation.value
        timeline._intervals_truncated = True


def execute_hbm_numeric(core, request):
    """Execute one accepted request, or decline without changing live state."""
    packed = _pack_hbm_numeric(core, request)
    if packed is None:
        return None
    resident = core._hbm_numeric_state is packed
    before = tuple(array.copy() for array in packed.arrays) if resident else None
    try:
        result = packed.execute(request)
    except BaseException:
        if resident:
            packed.arrays = before
        raise
    if result is not None:
        packed._controller_commits += 1
        if not resident:
            packed.commit_controller()
        packed.commit_timeline(request)
    elif resident:
        packed.arrays = before
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
        packed._controller_commits += 1
        return result

    results = []
    for index, request in enumerate(requests):
        # Numeric buffers are contiguous and cheap to copy. Keep the last
        # successful state so an exception or decline cannot commit half of
        # the current request or discard requests already accepted in order.
        before = tuple(array.copy() for array in packed.arrays)
        before_commits = packed._controller_commits
        try:
            result = core._submit_validated(request, execute=execute)
        except _NumericDeclined:
            packed.arrays = before
            packed._controller_commits = before_commits
            if results and core._hbm_numeric_state is not packed:
                packed.commit_controller()
            return tuple(results) + tuple(core.submit(item) for item in requests[index:])
        except BaseException:
            packed.arrays = before
            packed._controller_commits = before_commits
            if results and core._hbm_numeric_state is not packed:
                packed.commit_controller()
            raise
        results.append(result)
    if core._hbm_numeric_state is not packed:
        packed.commit_controller()
    return tuple(results)
