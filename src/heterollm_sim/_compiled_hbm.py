"""Pack the JEDEC HBM controller state for its separate compiled executor."""
from __future__ import annotations

import math
import copy
from functools import lru_cache
from operator import itemgetter
from types import MappingProxyType
import numpy as np

from .memory_types import Operation, TransactionResult
from .memory_mapping import map_dram_address
from ._hbm_pack import (refresh_resource_columns, publish_resource_columns,
                       can_capture_resource_columns, consume_resource_columns)


# Cover the 4 TiB, 64-stack HBM reference without raising the analytical
# controller's separate limit. The three bank arrays occupy 9 MiB here;
# batch checkpoints and Python bank mappings require additional memory.
MAX_HBM_NUMERIC_BANKS = 131_072


class _BorrowedControllerCheckpoint(tuple):
    """Controller rows are owned by an enclosing immutable undo image."""


@lru_cache(maxsize=512)
def _request_array_selection(lanes, banks, first_lane, lane_count):
    """Array rows a sparse request may mutate, including channel maintenance.

    Refresh and maximum-active maintenance can change the sibling PC and any
    bank in the same physical channel.  Other channels share no controller or
    CA/data resources.  Dense requests use the cheaper contiguous full copy.
    """
    channels = lanes // 2
    if lane_count >= lanes - 1:
        return None
    selected = tuple(sorted({((first_lane + offset) % lanes) // 2
                             for offset in range(lane_count + 1)}))
    if len(selected) * 2 >= channels:
        return None

    def rows(width):
        if selected[-1] - selected[0] + 1 == len(selected):
            return slice(selected[0] * width, (selected[-1] + 1) * width)
        indices = np.fromiter((ch * width + offset for ch in selected
                               for offset in range(width)), dtype=np.intp)
        indices.flags.writeable = False
        return indices

    bank_rows, pc_rows, channel_rows = rows(2 * banks), rows(2), rows(1)
    resources = np.array((*selected, *(channels + ch for ch in selected),
                          *(2 * channels + 2 * ch + pc for ch in selected for pc in range(2))),
                         dtype=np.intp)
    resources.flags.writeable = False
    # touched is reset globally before every numeric invocation; preserve its
    # complete previous bitmap even when all other writes are channel-local.
    return (bank_rows, bank_rows, bank_rows, pc_rows, pc_rows, pc_rows,
            channel_rows, channel_rows, resources, resources, resources,
            slice(None), pc_rows, pc_rows)


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


def _pack_hbm_numeric(core, request, *, refresh_requests=None):
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
    if any(type(values) is not dict for values in (
            timeline.lane_available, timeline.ready_ns, timeline.busy_ns,
            timeline.last_intervals, timeline.lane_directions,
            timeline.bytes_moved, timeline.directions)):
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
    cached = core._hbm_numeric_state
    if cached is not None:
        if c != cached._config:
            core._hbm_publish_numeric()
            return _pack_hbm_numeric(core, request, refresh_requests=refresh_requests)
        if not cached.validate_aliases():
            return None
        for token in core._hbm_transactions:
            for item in (refresh_requests or (request,)):
                token.capture_numeric_request(cached, item)
        if cached.refresh_resources(physical, refresh_requests or (request,)):
            return cached
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
    if timeline.resource_aliases and any(timeline.canonical_resource_id(rid) != rid for rid in ids):
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
    # Column commands are emitted on the clock grid. Restored/authored state
    # may contain arbitrary timestamps, so prove that invariant before using
    # the monotone, grouped history reduction in the numeric controller.
    half_clock = t[0] / 2
    columns = history[:, :, 1:3]
    ticks = np.rint(columns / half_clock)
    grid_history = bool(np.all((columns < 0) | ((np.abs(ticks) < 2. ** 49)
                                              & (columns == ticks * half_clock))))
    g = np.append(g, np.int64(grid_history))
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
        c = self._config
        self._request_lane_span = (c.row_bytes * c.bank_count if c.address_mapping == "row_local"
                                   else c.interleave_bytes or c.burst_bytes)
        self._dense_request_bytes = max(1, c.lane_count // 4) * self._request_lane_span
        aliases = core.timeline.resource_aliases
        self._alias_state = (dict(aliases) if type(aliases) is dict
                             and all(type(key) is str and type(value) is str
                                     for key, value in aliases.items()) else None)
        self._resource_getter = itemgetter(*ids)
        self._calendar_ids = None
        # Conversion scratch never owns controller state and is overwritten
        # before use; it is deliberately absent from transaction checkpoints.
        self._resource_scratch = np.empty((len(ids), 2), dtype=np.float64)
        self._resource_lookup = {rid: index for index, rid in enumerate(ids)}
        self._capture_positions = {}
        self._refreshed_resource_ids = ids

    def validate_aliases(self):
        aliases = self.core.timeline.resource_aliases
        if type(aliases) is dict and self._alias_state is not None and aliases == self._alias_state:
            return True
        if aliases and any(self.core.timeline.canonical_resource_id(rid) != rid for rid in self.ids):
            return False
        self._alias_state = (dict(aliases) if type(aliases) is dict
                             and all(type(key) is str and type(value) is str
                                     for key, value in aliases.items()) else None)
        return True

    def refresh_resources(self, physical, requests):
        """External link reservations remain authoritative on the timeline."""
        timeline = self.core.timeline
        resources, moved, directions = self.arrays[8:11]
        all_ids = self.ids
        if type(timeline.lane_available) is dict:
            try:
                slots = self._resource_getter(timeline.lane_available)
            except KeyError:
                slots = tuple(map(timeline.lane_available.get, all_ids))
        else:
            slots = tuple(map(timeline.lane_available.get, all_ids))
        try:
            lengths = set(map(len, slots))
            present = slots
        except TypeError:
            present = tuple(slot for slot in slots if slot is not None)
            lengths = set(map(len, present))
        if lengths - {1}:
            return False
        identities = tuple(map(id, present))
        if identities != self._calendar_ids:
            if len(set(identities)) != len(identities):
                return False
            self._calendar_ids = identities
        # Only calendar identity uniqueness is cached. Lengths and every live
        # value are reread, including in-place edits and restored timelines.
        # Only channels reached by this ordered request set can execute a
        # command, including tRASmax/refresh maintenance on the other PC.
        # Refresh those channels from the public timeline every time; this
        # does not rely on hidden mutation versions or trusted caller state.
        c = self.core.config
        lane_span = (c.row_bytes * c.bank_count if c.address_mapping == "row_local"
                     else c.interleave_bytes or c.burst_bytes)
        channels = len(self.channel_keys)
        selected = set()
        for request in requests:
            first = request.address // lane_span
            last = (request.address + request.byte_count - 1) // lane_span
            if last - first >= c.lane_count - 1:
                selected = None
                break
            selected.update((lane % c.lane_count) // 2 for lane in range(first, last + 1))
            if len(selected) == channels:
                selected = None
                break
        if selected is None:
            indices = slice(None)
            ids = all_ids
        else:
            indices = sorted(index for ch in selected
                             for index in (ch, channels + ch, 2 * channels + 2 * ch,
                                           2 * channels + 2 * ch + 1))
            ids = tuple(all_ids[index] for index in indices)
            slots = [slots[index] for index in indices]
        if not refresh_resource_columns(resources, moved, directions,
                None if isinstance(indices, slice) else indices, ids, slots,
                timeline.ready_ns, timeline.busy_ns, timeline.last_intervals,
                timeline.bytes_moved, timeline.directions, physical, self._resource_scratch):
            return False
        self._refreshed_resource_ids = ids
        return bool(np.isfinite(resources).all())

    def _copy_controller_arrays(self):
        # Baseline of the currently published Python objects. A resident core
        # retains it across submissions until that state is exposed publicly.
        return tuple(self.arrays[index].copy() for index in (0, 1, 2, 6, 7))

    def request_selection(self, request):
        if request.byte_count >= self._dense_request_bytes:
            return None
        c = self._config
        lane_span = self._request_lane_span
        first_lane = request.address // lane_span
        last_lane = (request.address + request.byte_count - 1) // lane_span
        return _request_array_selection(c.lane_count, c.bank_count,
                                        first_lane % c.lane_count, last_lane - first_lane)

    def checkpoint(self, request):
        selection = self.request_selection(request)
        if selection is None:
            for token in reversed(self.core._hbm_transactions):
                undo = token.numeric
                if (undo is not None and undo.packed is self and undo.whole is not None
                        and undo.commits == self._controller_commits):
                    # The first dense access has the enclosing transaction's
                    # exact controller start. Resource rows were refreshed
                    # from the authoritative public timeline after that undo
                    # image, so they still need their own current checkpoint.
                    return None, _BorrowedControllerCheckpoint(
                        array.copy() if index in (8, 9, 10, 11) else undo.whole[index]
                        for index, array in enumerate(self.arrays))
            return None, tuple(array.copy() for array in self.arrays)
        channel_rows = selection[6]
        if isinstance(channel_rows, slice) and channel_rows.stop - channel_rows.start == 1:
            channel = channel_rows.start
            for token in reversed(self.core._hbm_transactions):
                undo = token.numeric
                if (undo is None or undo.packed is not self
                        or undo.commits != self._controller_commits):
                    continue
                saved = undo.channels.get(channel)
                if undo.whole is not None or saved is not None:
                    # Single-channel maintenance includes both PCs. Its
                    # immutable outer start image is already the request's
                    # controller checkpoint; resources alone were refreshed
                    # from the public timeline after that image was taken.
                    return selection, _BorrowedControllerCheckpoint(
                        (array[index].copy() if isinstance(index, slice) else array[index])
                        if number in (8, 9, 10, 11) else
                        undo.whole[number][index] if undo.whole is not None else saved[1][number]
                        for number, (array, index) in enumerate(zip(self.arrays, selection)))
        # Advanced indexing already returns an independent array.  Avoid
        # copying it a second time for the wrapped-channel/resource indices.
        return selection, tuple(array[index].copy() if isinstance(index, slice) else array[index]
                                for array, index in zip(self.arrays, selection))

    def restore_checkpoint(self, checkpoint):
        selection, saved = checkpoint
        if selection is None:
            # A declined/failed request may continue before the outer caller
            # rolls back. Never make its live arrays aliases of that caller's
            # immutable undo point; copies are needed only on this rare path.
            self.arrays = (tuple(array.copy() for array in saved)
                           if isinstance(saved, _BorrowedControllerCheckpoint) else saved)
        else:
            for array, index, values in zip(self.arrays, selection, saved):
                array[index] = values

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
                or int(moved.max()) > (1 << 63) - 1 - physical):
            return None
        if self.core._hbm_numeric_state is self:
            for token in self.core._hbm_transactions:
                token.capture_numeric_request(self, request)
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
        selection = self.request_selection(request)
        finite_indices = (1, 3, 4, 6, 8, 12)
        # Packing validates the complete state once.  An accepted numeric
        # execution only writes the selected channels, so every untouched
        # floating-point value retains that finite invariant.
        if not math.isfinite(completion) or not all(
                np.isfinite(self.arrays[index] if selection is None
                            else self.arrays[index][selection[index]]).all()
                for index in finite_indices):
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
        timeline._remember_resources(ids[index] for index in indices)
        publish_resource_columns(ids, resources, moved, indices,
            timeline.lane_available, timeline.ready_ns, timeline.busy_ns,
            timeline.last_intervals, timeline.bytes_moved, timeline.directions,
            timeline.lane_directions, timeline._touched, channels, request.operation.value)
        timeline._intervals_truncated = True


def execute_hbm_numeric(core, request):
    """Execute one accepted request, or decline without changing live state."""
    packed = _pack_hbm_numeric(core, request)
    if packed is None:
        return None
    resident = core._hbm_numeric_state is packed
    before = packed.checkpoint(request) if resident else None
    try:
        result = packed.execute(request)
    except BaseException:
        if resident:
            packed.restore_checkpoint(before)
        raise
    if result is not None:
        packed._controller_commits += 1
        if not resident:
            packed.commit_controller()
        packed.commit_timeline(request)
    elif resident:
        packed.restore_checkpoint(before)
    return result


class _NumericDeclined(Exception):
    pass


class _NumericResourceCapture:
    """Consume each successful request's checkpoint directly into a task bill."""

    def __init__(self, packed, accumulator):
        self.packed = packed
        self.accumulator = accumulator
        self.before = None

    def begin(self):
        timeline = self.packed.core.timeline
        timeline._intervals = {}
        timeline._interval_limit = 0
        timeline._interval_count = 0
        timeline._intervals_truncated = False
        timeline._touched = set()
        timeline._interval_payloads = {}

    def finish(self):
        packed, (selection, saved) = self.packed, self.before
        positions = None
        if selection is not None:
            key = id(selection)
            cached = packed._capture_positions.get(key)
            if cached is None or cached[0] is not selection:
                # Keep the selection alive, making identity reuse impossible.
                positions = {int(index): position for position, index in enumerate(selection[8])}
                if len(packed._capture_positions) >= 512:
                    packed._capture_positions.clear()
                packed._capture_positions[key] = (selection, positions)
            else:
                positions = cached[1]
        timeline = packed.core.timeline
        consume_resource_columns(self.accumulator, packed._resource_lookup,
            packed.arrays[8], packed.arrays[9], saved[8], saved[9], positions, timeline)
        return {"resource_intervals": {}, "resource_interval_payloads": {},
                "intervals_truncated": timeline._intervals_truncated}


def execute_hbm_numeric_many(core, requests, *, resource_accumulator=None):
    """Amortize packing while retaining each request's admission and metrics."""
    if not requests:
        return ()
    packed = _pack_hbm_numeric(core, requests[0], refresh_requests=requests)
    if packed is None:
        return None
    capture = (_NumericResourceCapture(packed, resource_accumulator)
               if resource_accumulator is not None
               and can_capture_resource_columns(core.timeline, packed._refreshed_resource_ids) else None)
    def execute(request):
        result = packed.execute(request)
        if result is None:
            raise _NumericDeclined
        packed.commit_timeline(request)
        packed._controller_commits += 1
        return result

    results = []
    for index, request in enumerate(requests):
        # Keep the last successful state, including both PCs and maintenance
        # on every reached channel.  A failed request cannot discard earlier
        # accepted requests or expose half of its own controller mutations.
        before = packed.checkpoint(request)
        before_commits = packed._controller_commits
        if capture is not None:
            capture.before = before
        try:
            result = core._submit_validated(request, execute=execute,
                _resource_accumulator=resource_accumulator, _metrics_capture=capture)
        except _NumericDeclined:
            packed.restore_checkpoint(before)
            packed._controller_commits = before_commits
            if results and core._hbm_numeric_state is not packed:
                packed.commit_controller()
            remaining = (tuple(core.submit(item) for item in requests[index:])
                         if resource_accumulator is None else tuple(
                             core._submit_validated(item, _resource_accumulator=resource_accumulator)
                             for item in requests[index:]))
            return tuple(results) + remaining
        except BaseException:
            packed.restore_checkpoint(before)
            packed._controller_commits = before_commits
            if results and core._hbm_numeric_state is not packed:
                packed.commit_controller()
            raise
        results.append(result)
    if core._hbm_numeric_state is not packed:
        packed.commit_controller()
    return tuple(results)


# Sparse timeline undo is valid only for this complete owned executor path.
# A wrapper around any of these hooks can write a different resource before
# raising, so checking DramCore.submit alone does not establish ownership.
_OWNED_TIMELINE_HBM_FUNCTIONS = tuple((name, globals()[name]) for name in (
    "_pack_hbm_numeric", "execute_hbm_numeric", "execute_hbm_numeric_many",
    "_request_array_selection", "can_capture_resource_columns",
    "refresh_resource_columns", "publish_resource_columns", "consume_resource_columns"))
_OWNED_TIMELINE_HBM_METHODS = tuple((name, getattr(_HBMNumericState, name)) for name in (
    "validate_aliases", "refresh_resources", "request_selection", "checkpoint",
    "restore_checkpoint", "execute", "commit_controller", "commit_timeline"))
_OWNED_TIMELINE_HBM_CLASS = _HBMNumericState
_OWNED_TIMELINE_CAPTURE_METHODS = tuple((name, getattr(_NumericResourceCapture, name))
    for name in ("__init__", "begin", "finish"))
_OWNED_TIMELINE_CAPTURE_CLASS = _NumericResourceCapture

from . import _hbm_numeric as _owned_numeric_module
from . import _hbm_control as _owned_control_module

_OWNED_TIMELINE_NUMERIC_FUNCTIONS = tuple((name, getattr(_owned_numeric_module, name))
    for name in ("run_hbm_numeric", "_run_hbm_numeric", "_run_hbm_channel_range"))
_OWNED_TIMELINE_CONTROL_ENTRY = _owned_control_module.execute_hbm
_OWNED_TIMELINE_CONTROL_CLASS = _owned_control_module._Controller
_OWNED_TIMELINE_CONTROL_METHODS = tuple((name, getattr(_OWNED_TIMELINE_CONTROL_CLASS, name))
    for name, value in vars(_OWNED_TIMELINE_CONTROL_CLASS).items() if callable(value))


def _has_native_timeline_hooks(packed=None):
    """Reject replaceable executor callbacks before using sparse rollback."""
    if (_HBMNumericState is not _OWNED_TIMELINE_HBM_CLASS
            or (packed is not None and type(packed) is not _HBMNumericState)
            or _NumericResourceCapture is not _OWNED_TIMELINE_CAPTURE_CLASS
            or _owned_control_module.execute_hbm is not _OWNED_TIMELINE_CONTROL_ENTRY
            or _owned_control_module._Controller is not _OWNED_TIMELINE_CONTROL_CLASS):
        return False
    for name, original in _OWNED_TIMELINE_HBM_FUNCTIONS:
        if globals()[name] is not original:
            return False
    for name, original in _OWNED_TIMELINE_HBM_METHODS:
        method = getattr(_HBMNumericState if packed is None else packed, name)
        if getattr(method, "__func__", method) is not original:
            return False
    for name, original in _OWNED_TIMELINE_CAPTURE_METHODS:
        if getattr(_NumericResourceCapture, name) is not original:
            return False
    for name, original in _OWNED_TIMELINE_NUMERIC_FUNCTIONS:
        if getattr(_owned_numeric_module, name) is not original:
            return False
    for name, original in _OWNED_TIMELINE_CONTROL_METHODS:
        if getattr(_owned_control_module._Controller, name) is not original:
            return False
    return packed is None or packed.run_numeric is _OWNED_TIMELINE_NUMERIC_FUNCTIONS[0][1]
