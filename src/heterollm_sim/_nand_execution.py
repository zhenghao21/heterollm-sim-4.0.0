"""Compiled strict OCP page IO with interleaved external link calendars.

Every page performs host admission, cache transitions and array/bus
reservations. Incoming write links precede NAND admission; outgoing read
links follow NAND completion. This is the normal per-page state machine.
"""
from __future__ import annotations

import math
from functools import lru_cache
import numpy as np

from .memory_mapping import map_nand_address
from .memory_transfer import ResourceTimeline
from .memory_types import Operation, TransactionResult
from ._nand_pack import (
    pack_single_lane as _single_lane_timeline_state, pack_multi_lane,
    publish_timeline, pack_host_admission, publish_host_admission, host_admission_mask,
)


from . import _compiled_nand as _numeric


_GEOMETRY_PLANS = {}


@lru_cache(maxsize=128)
def _numeric_program(geometry, timing, names, link_program, direction_names):
    """Prepare immutable kernel inputs, separate from every live calendar.

    The complete values form the key, so mutable caller routes, timing
    metadata and unusual direction labels cannot retain an obsolete plan.
    Bytes-backed arrays cannot be made writable through a retained view.
    """
    lookup = {name: index for index, name in enumerate(names)}
    directions = {name: index for index, name in enumerate(direction_names)}
    offsets, ids, durations, payloads, link_directions = [0], [], [], [], []
    for _, direction, program in link_program:
        for name, duration, payload in program:
            if not math.isfinite(duration) or duration < 0 or not isinstance(payload, int) or payload < 0:
                return None
            ids.append(lookup[name])
            durations.append(duration)
            payloads.append(payload)
        offsets.append(len(ids))
        link_directions.append(directions[direction])
    def frozen(values, dtype):
        return np.frombuffer(np.asarray(values, dtype=dtype).tobytes(), dtype=dtype)
    return (frozen(geometry, np.int64), frozen(timing, np.float64),
            frozen(offsets, np.int64), frozen(ids, np.int64), frozen(durations, np.float64),
            frozen(payloads, np.int64), frozen(link_directions, np.int64), tuple(offsets), tuple(ids))


def _geometry_plan(core, operation, banks, units, per_channel):
    """Cache immutable address/resource descriptions, never live calendars."""
    config = core.config
    prefixes = tuple(str(config.metadata.get(name, default)) for name, default in (
        ("array_resource_prefix", "nand:array"), ("parallel_resource_prefix", "nand:parallel"),
        ("channel_resource_prefix", "nand:channel"), ("host_resource_id", "nand:host")))
    key = (config.channels, config.targets_per_channel, config.dies_per_target, config.luns_per_die,
           config.planes_per_lun, config.planes_independent, config.parallel_units,
           config.host_channels, config.host_full_duplex, prefixes,
           tuple(core.timeline.resource_aliases.items()), operation)
    plan = _GEOMETRY_PLANS.get(key)
    if plan is not None:
        return plan
    mappings = tuple(map_nand_address(config, bank * 4096) for bank in range(banks))
    bank_keys = tuple(core._bank_key(mapping) for mapping in mappings)
    tag_prefixes = tuple(core._staged_key(mapping)[:-2] for mapping in mappings)
    array_names = tuple(core._array_id(mappings[unit]) for unit in range(units))
    array_resources = tuple(core._array_resource_id(mappings[unit]) for unit in range(units))
    channel_names = tuple(core._channel_id(mappings[channel * per_channel]) for channel in range(config.channels))
    host_names = tuple(core._host_id(mappings[channel * per_channel], operation)
                       for channel in range(config.host_channels))
    nand_resources = tuple(dict.fromkeys((*array_resources, *channel_names, *host_names)))
    lookup = {name: index for index, name in enumerate(nand_resources)}
    ids = tuple(np.array([lookup[name] for name in names], dtype=np.int64)
                for names in (array_resources, channel_names, host_names))
    for values in ids:
        values.flags.writeable = False
    plan = (bank_keys, tag_prefixes, array_names, array_resources, channel_names,
            host_names, nand_resources, *ids)
    if len(_GEOMETRY_PLANS) >= 64:
        _GEOMETRY_PLANS.clear()
    _GEOMETRY_PLANS[key] = plan
    return plan


class _NandNumericState:
    """Exclusive native controller state; fabric calendars remain shared."""

    def __init__(self, layout, arrays):
        self.layout = layout
        self.arrays = arrays
        self.dirty_units = np.zeros(len(arrays[0]), dtype=np.bool_)
        self.dirty_banks = np.zeros(len(arrays[2]), dtype=np.bool_)

    def checkpoint(self):
        return (tuple(array.copy() for array in self.arrays),
                self.dirty_units.copy(), self.dirty_banks.copy())

    def restore(self, checkpoint):
        arrays, units, banks = checkpoint
        self.arrays = tuple(array.copy() for array in arrays)
        self.dirty_units, self.dirty_banks = units.copy(), banks.copy()

    def publish(self, cache, array_frontiers, buffer_frontiers, *, checkpoint=None, core=None):
        from .nand_core import _ReadCacheSlot

        arrays, units, banks = ((self.arrays, self.dirty_units, self.dirty_banks)
                               if checkpoint is None else checkpoint)
        array_ready, buffer_ready, tags, ready, readers, uses, _present = arrays
        bank_keys, tag_prefixes, array_names, pages_per_block = self.layout
        array_values, buffer_values = array_ready.tolist(), buffer_ready.tolist()
        for unit in np.flatnonzero(units).tolist():
            name = array_names[unit]
            array_frontiers[name], buffer_frontiers[name] = array_values[unit], buffer_values[unit]
        tag_values, ready_values = tags.tolist(), ready.tolist()
        reader_values, use_values = readers.tolist(), uses.tolist()
        for bank in np.flatnonzero(banks).tolist():
            key = bank_keys[bank]
            if core is not None and core._nand_transactions:
                core._touch_cache_bank(key)
            slots = cache.get(key)
            if slots is None:
                slots = [_ReadCacheSlot(), _ReadCacheSlot()]
                cache[key] = slots
            for index, slot in enumerate(slots):
                page = tag_values[bank][index]
                slot.page_key = None if page < 0 else (*tag_prefixes[bank], *divmod(page, pages_per_block))
                slot.ready_ns = ready_values[bank][index]
                slot.reader_completion_ns, slot.last_use = reader_values[bank][index], use_values[bank][index]


def execute_pages(core, addresses, arrival_ns, link_program, *, operation=Operation.READ):
    """Return exact aggregate page/link execution or decline before mutation."""
    from .nand_core import NandCore, _ReadCacheSlot
    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or core.capture_details or core._observe_media or config.protocol_mode != "ocp_hbf_0_7"
            or config.queue_scope != "host_channel"):
        return None
    addresses = np.asarray(addresses)
    if (addresses.ndim != 1 or addresses.dtype.kind not in "iu" or len(addresses) < 2
            or config.effective_capacity_bytes > 2**63 - 1
            or np.any(addresses % 4096) or np.any(addresses < 0)
            or np.any(addresses > config.effective_capacity_bytes - 4096)):
        return None
    addresses = np.asarray(addresses, dtype=np.int64)
    count = len(addresses)
    arrivals = np.asarray(arrival_ns)
    if arrivals.dtype.kind not in "iuf" or arrivals.ndim > 1:
        return None
    if arrivals.ndim == 0:
        arrivals = np.array([float(arrivals)], dtype=np.float64)
    elif len(arrivals) != count:
        return None
    else:
        arrivals = np.asarray(arrivals, dtype=np.float64)
    if np.any(~np.isfinite(arrivals)) or np.any(arrivals < 0):
        return None
    read = operation is Operation.READ
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base_units * config.planes_per_lun
    units = banks if config.planes_independent else base_units
    if (banks > 65536 or config.effective_capacity_bytes < banks * 4096
            or core._cache_use > 2**63 - 1 - count):
        return None
    per_channel = base_units // config.channels
    (bank_keys, tag_prefixes, array_names, array_resources, channel_names, host_names,
     nand_resources, array_ids, channel_ids, host_ids) = _geometry_plan(core, operation, banks, units, per_channel)
    program_frontiers = {}
    if not read:
        for address in addresses.tolist():
            page_index = address // 4096
            bank, physical_page = page_index % banks, page_index // banks
            block, page = divmod(physical_page, config.pages_per_block)
            key = (*bank_keys[bank], block)
            expected = program_frontiers.get(key, core._next_program_page.get(key, 0))
            if page != 0 and page != expected:
                raise ValueError("ocp_hbf_0_7 write order violation: block {} requires page {}, received {}"
                                 .format(key, expected, page))
            program_frontiers[key] = page + 1
    # Shared array slots are supported; independently named buses must remain
    # distinct to preserve per-owner billing without ambiguous double counting.
    if (set(array_resources + array_names) & set(channel_names + host_names)
            or set(channel_names) & set(host_names)):
        return None
    link_program = tuple(link_program)
    link_names = [resource for _, _, program in link_program for resource, _, _ in program]
    if (len(set(link_names)) != len(link_names) or set(link_names) & set(nand_resources)
            or any(name in link_names for name in array_names)):
        return None
    names = list(dict.fromkeys([*nand_resources, *link_names]))
    if any(timeline.canonical_resource_id(name) != name for name in names):
        return None
    starts, lane_counts, lanes, lane_directions = [], [], [], []
    direction_names = [None, "read", "write"]
    direction_lookup = {name: index for index, name in enumerate(direction_names)}
    def direction_id(value):
        if value not in direction_lookup:
            direction_lookup[value] = len(direction_names)
            direction_names.append(value)
        return direction_lookup[value]
    single_lane_state = _single_lane_timeline_state(timeline, names, count, config.transfer_page_bytes)
    if single_lane_state is not None:
        starts, lane_counts, lanes, lane_directions, state, moved, directions = single_lane_state
    else:
        packed_timeline = pack_multi_lane(timeline, names, count, config.transfer_page_bytes,
                                         direction_names, direction_lookup)
        if packed_timeline is None:
            return None
        starts, lane_counts, lanes, lane_directions, state, moved, directions = packed_timeline
    layout = (bank_keys, tag_prefixes, array_names, config.pages_per_block)
    packed = core._nand_numeric_state
    if packed is not None and packed.layout != layout:
        core._nand_publish_numeric()
        packed = None
    if packed is None:
        arrays = np.array([core.__dict__["_array_ready"].get(name, 0.) for name in array_names], dtype=np.float64)
        buffers = np.array([core.__dict__["_buffer_ready"].get(name, 0.) for name in array_names], dtype=np.float64)
        tag_values, ready_values, reader_values, use_values = [-1] * (banks * 2), [0.] * (banks * 2), [0.] * (banks * 2), [0] * (banks * 2)
        present_values = [False] * banks
        for bank, key in enumerate(bank_keys):
            slots = core.__dict__["_read_cache"].get(key, ())
            if len(slots) not in (0, 2):
                return None
            present_values[bank] = bool(slots)
            for slot, old in enumerate(slots):
                index = bank * 2 + slot
                if old.page_key is not None:
                    tag_values[index] = old.page_key[-2] * config.pages_per_block + old.page_key[-1]
                ready_values[index], reader_values[index], use_values[index] = old.ready_ns, old.reader_completion_ns, old.last_use
        tags = np.array(tag_values, dtype=np.int64).reshape(banks, 2)
        ready = np.array(ready_values, dtype=np.float64).reshape(banks, 2)
        readers = np.array(reader_values, dtype=np.float64).reshape(banks, 2)
        uses = np.array(use_values, dtype=np.int64).reshape(banks, 2)
        cache_present = np.array(present_values, dtype=np.bool_)
        if core._nand_numeric_enabled:
            packed = _NandNumericState(layout, (arrays, buffers, tags, ready, readers, uses, cache_present))
    else:
        arrays, buffers, tags, ready, readers, uses, cache_present = packed.arrays
    admission, admission_ops, admission_counts, acceptance = pack_host_admission(
        core._host_inflight, core._host_acceptance_ns, core._acceptance_ns,
        config.host_channels, config.max_outstanding_requests, Operation.READ)
    for _, direction, program in link_program:
        direction_id(direction)
    starts, lane_counts = np.asarray(starts, dtype=np.int64), np.asarray(lane_counts, dtype=np.int64)
    lanes, lane_directions = np.asarray(lanes, dtype=np.float64), np.asarray(lane_directions, dtype=np.int64)
    program = _numeric_program(
        (base_units, config.planes_per_lun, config.planes_independent, per_channel,
         config.host_channels, config.max_outstanding_requests,
         config.max_outstanding_write_requests or config.max_outstanding_requests,
         config.pages_per_block),
        (config.page_read_ns, config.transfer_page_bytes / config.internal_bandwidth_gb_s,
         4096 / core._host_bandwidth(operation), config.front_ns, config.transfer_page_bytes,
         config.page_program_ns, config.block_erase_ns), tuple(names),
        tuple((owner, direction, tuple(tuple(demand) for demand in demands))
              for owner, direction, demands in link_program), tuple(direction_names))
    if program is None:
        return None
    (geometry, timing, numeric_offsets, numeric_ids, durations, payloads,
     link_dirs, link_offsets, link_ids) = program
    nand_busy = np.zeros(len(names), dtype=np.float64)
    nand_bytes = np.zeros(len(names), dtype=np.int64)
    link_busy = np.zeros(len(link_ids), dtype=np.float64)
    link_moved = np.zeros(len(link_ids), dtype=np.int64)
    resident_before = packed.checkpoint() if packed is core._nand_numeric_state and packed is not None else None
    if packed is not None:
        for token in core._nand_transactions:
            token.capture_numeric(packed)
    try:
        answer = _numeric._run(addresses, arrivals, read, geometry, timing, array_ids, channel_ids, host_ids,
                      arrays, buffers, tags, ready, readers, uses, cache_present, core._cache_use,
                      admission, admission_ops, admission_counts, acceptance,
                      starts, lane_counts, lanes, lane_directions, state, moved, directions,
                      numeric_offsets, numeric_ids, durations, payloads, link_dirs,
                      nand_busy, nand_bytes, link_busy, link_moved)
        (completions, counters, cache_clock, admitted_count, acceptance, touched, arrays_touched,
         banks_touched, link_arrivals, link_completions, link_last) = answer
        hosts_touched = host_admission_mask(
            banks_touched, arrays_touched, base_units, per_channel, config.host_channels)
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(completions)):
            raise ValueError("resource reservation exceeds finite timing range")
    except BaseException:
        if resident_before is not None:
            packed.restore(resident_before)
        raise
    if packed is not None:
        packed.dirty_units |= arrays_touched
        packed.dirty_banks |= banks_touched
        core._nand_numeric_state = packed
    # Shared fabric state is committed after numeric execution succeeds.
    # Exclusive controller arrays remain resident until a scalar handoff or
    # public read publishes them; the core transaction owns their undo state.
    # Convert whole columns in C. Per-element ndarray indexing otherwise
    # allocates thousands of NumPy scalars for every repeated Decode access.
    state_values, moved_values, direction_values = state.tolist(), moved.tolist(), directions.tolist()
    lane_values, lane_direction_values = lanes.tolist(), lane_directions.tolist()
    start_values, lane_count_values = starts.tolist(), lane_counts.tolist()
    busy_values, byte_values = nand_busy.tolist(), nand_bytes.tolist()
    active_units, active_banks = np.flatnonzero(arrays_touched).tolist(), np.flatnonzero(banks_touched).tolist()
    active_resources = {name for name, busy, size in zip(names, busy_values, byte_values) if busy or size}
    active_resources.update(array_resources[index] for index in active_units)
    active_resources.update(link_names)
    publish_timeline(timeline, names, state_values, moved_values, direction_values,
                     lane_values, lane_direction_values, start_values, lane_count_values,
                     active_resources, link_names, byte_values, direction_names,
                     single_lane_state is not None)
    array_values = arrays.tolist()
    for unit in active_units:
        timeline.ready_ns[array_names[unit]] = array_values[unit]
    if packed is None:
        array_values, buffer_values = arrays.tolist(), buffers.tolist()
        for unit in active_units:
            name = array_names[unit]
            core.__dict__["_array_ready"][name], core.__dict__["_buffer_ready"][name] = array_values[unit], buffer_values[unit]
            timeline.ready_ns[name] = array_values[unit]
        tag_values, ready_values, reader_values, use_values = tags.tolist(), ready.tolist(), readers.tolist(), uses.tolist()
        for bank in active_banks:
            key = bank_keys[bank]
            if core._nand_transactions:
                core._touch_cache_bank(key)
            slots = core.__dict__["_read_cache"].get(key)
            if slots is None:
                slots = [_ReadCacheSlot(), _ReadCacheSlot()]
                core.__dict__["_read_cache"][key] = slots
            for index, slot in enumerate(slots):
                page = tag_values[bank][index]
                slot.page_key = None if page < 0 else (*tag_prefixes[bank], *divmod(page, config.pages_per_block))
                slot.ready_ns, slot.reader_completion_ns, slot.last_use = ready_values[bank][index], reader_values[bank][index], use_values[bank][index]
    core._cache_use = int(cache_clock)
    core._update_program_frontiers(program_frontiers)
    publish_host_admission(
        core._host_inflight, core._host_acceptance_ns, admission,
        admission_ops, admitted_count, acceptance, hosts_touched,
        Operation.READ, Operation.WRITE)
    core._observation_events = []
    timeline._intervals, timeline._interval_payloads = {}, {}
    timeline._interval_count = timeline._interval_limit = 0
    timeline._intervals_truncated = True
    timeline._touched = {names[index] for index in np.flatnonzero(touched)}
    misses, hits = int(counters[4]), int(counters[5])
    busy = {name: value for name, value in zip(names, busy_values) if value}
    byte_counts = {name: value for name, value in zip(names, byte_values) if value}
    last = {name: timeline.last_intervals[name] for name in active_resources - set(link_names)}
    programmed = 0 if read else count
    internal_bytes = (misses + programmed) * config.transfer_page_bytes
    bill = TransactionResult("hbf-" + operation.value + "-pages", operation, float(counters[7]), float(counters[6]),
        count * 4096, internal_bytes, counters={
            "physical_operation_count": count,
            "host_transfer_bytes": count * 4096, "internal_transfer_bytes": internal_bytes,
            "physical_read_bytes": misses * 4096, "physical_write_bytes": programmed * 4096,
            "pages_read": misses, "pages_programmed": programmed, "read_cache_hits": hits, "read_cache_misses": misses,
            "read_operations": misses, "program_operations": programmed, "rmw_read_operations": 0,
            "page_count": count, "pages_touched": count, "page_access_payload_bytes": count * 4096,
            "page_access_capacity_bytes": count * 4096, "erase_operations": int(counters[8]),
            "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": 0,
            "staging_read_hits": 0, "staging_read_merges": 0, "details_truncated": False,
            "bandwidth_ceiling_gb_s": config.host_bandwidth_for(operation) if hits else core._bandwidth_ceiling_gb_s(operation),
            "bandwidth_ceiling_scope": "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain"
                if hits else "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
            "array_units": config.array_units, "write_completion": "n/a" if read else "media_program_complete", "host_ack_completion_ns": None,
            "media_drain_completion_ns": None if read else float(counters[6]), "host_admission_samples": count,
            "host_outstanding_at_admission_peak": int(counters[1]),
            "host_read_outstanding_at_admission_peak": int(counters[2]),
            "host_write_outstanding_at_admission_peak": int(counters[3]),
            "host_admission_depth_scope": "outstanding page commands per host channel sampled at admission, including the admitted command",
            "queue_wait_ns": float(counters[0]), "write_buffer_wait_ns": 0., "write_buffer_bytes": 0,
            "write_buffer_occupancy_bytes": 0, "write_buffer_peak_bytes": 0, "write_buffer_draining_bytes": 0,
            "buffered_write_mode": False, "resource_busy_ns": busy, "resource_bytes": byte_counts,
            "resource_last_intervals": last, "resource_intervals": {}, "resource_interval_payloads": {},
            "intervals_truncated": True,
        })
    links = []
    for link, (owner, _, _) in enumerate(link_program):
        busy, byte_counts, last = {}, {}, {}
        for transfer in range(link_offsets[link], link_offsets[link + 1]):
            name = names[link_ids[transfer]]
            busy[name] = busy.get(name, 0.) + float(link_busy[transfer])
            byte_counts[name] = byte_counts.get(name, 0) + int(link_moved[transfer])
            last[name] = tuple(float(value) for value in link_last[transfer])
        links.append((owner, TransactionResult("hbf-read-pages-link", Operation.READ, float(link_arrivals[link]),
            float(link_completions[link]), count * 4096, count * 4096, counters={
                "physical_operation_count": count,
                "resource_busy_ns": busy, "resource_bytes": byte_counts, "resource_last_intervals": last,
                "resource_intervals": {}, "resource_interval_payloads": {}, "internal_transfer_bytes": 0,
            })))
    return completions, bill, links
