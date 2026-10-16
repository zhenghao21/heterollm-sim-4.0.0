"""Compiled strict OCP page IO with interleaved external link calendars.

Every page performs host admission, cache transitions and array/bus
reservations. Incoming write links precede NAND admission; outgoing read
links follow NAND completion. This is the normal per-page state machine.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

from .memory_mapping import map_nand_address
from .memory_transfer import ResourceTimeline
from .memory_types import Operation, TransactionResult


_GEOMETRY_PLANS = {}


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


@njit(cache=True, nogil=True, fastmath=False)
def _reserve(rid, earliest, duration, payload, direction, starts, lane_counts,
             lanes, lane_directions, resource_state, moved, directions, touched):
    first, count = starts[rid], lane_counts[rid]
    chosen = first
    for lane in range(first + 1, first + count):
        if lanes[lane] < lanes[chosen]:
            chosen = lane
    start = max(earliest, lanes[chosen])
    end = start + duration
    lanes[chosen] = end
    ready = lanes[first]
    for lane in range(first + 1, first + count):
        ready = min(ready, lanes[lane])
    resource_state[rid, 0] = ready
    resource_state[rid, 1] += duration
    resource_state[rid, 2] = start
    resource_state[rid, 3] = end
    if direction:
        directions[rid] = direction
        lane_directions[chosen] = direction
    if payload >= 0:
        moved[rid] += payload
    touched[rid] = True
    return end


@njit(cache=True, nogil=True, fastmath=False)
def _transfer_links(end, link_offsets, link_ids, link_durations, link_bytes, link_directions,
                    starts, lane_counts, lanes, lane_directions, resource_state, moved,
                    directions, touched, link_busy, link_moved, link_arrivals, link_completions, link_last):
    for link in range(len(link_offsets) - 1):
        first, stop = link_offsets[link], link_offsets[link + 1]
        link_arrivals[link] = min(link_arrivals[link], end)
        link_start = end
        for transfer in range(first, stop):
            rid = link_ids[transfer]
            first_lane, lane_count = starts[rid], lane_counts[rid]
            ready = lanes[first_lane]
            for lane in range(first_lane + 1, first_lane + lane_count):
                ready = min(ready, lanes[lane])
            link_start = max(link_start, ready)
        for transfer in range(first, stop):
            rid = link_ids[transfer]
            duration, payload = link_durations[transfer], link_bytes[transfer]
            transfer_end = _reserve(rid, link_start, duration, payload, link_directions[link],
                                    starts, lane_counts, lanes, lane_directions, resource_state,
                                    moved, directions, touched)
            end = max(end, transfer_end)
            link_busy[transfer] += duration
            link_moved[transfer] += payload
            link_last[transfer, 0] = resource_state[rid, 2]
            link_last[transfer, 1] = resource_state[rid, 3]
        link_completions[link] = max(link_completions[link], end)
    return end


@njit(cache=True, nogil=True, fastmath=False)
def _run(addresses, arrivals, read, geometry, timing, array_ids, channel_ids, host_ids,
         array_ready, buffer_ready, cache_tags, cache_ready, cache_readers, cache_use,
         cache_present, cache_clock, admission, admission_ops, admission_count, acceptance,
         starts, lane_counts, lanes, lane_directions, resource_state, moved, directions,
         link_offsets, link_ids, link_durations, link_bytes, link_directions,
         nand_busy, nand_bytes, link_busy, link_moved):
    base_units, planes, independent, per_channel, host_channels, max_outstanding, pages_per_block = geometry
    page_read, internal_ns, host_ns, front, internal_bytes, page_program, block_erase = timing
    count = len(addresses)
    completions = np.empty(count, dtype=np.float64)
    touched = np.zeros(len(starts), dtype=np.bool_)
    arrays_touched = np.zeros(len(array_ready), dtype=np.bool_)
    banks_touched = np.zeros(len(cache_tags), dtype=np.bool_)
    # queue wait sum, overall/read/write admission peaks, misses, hits, max NAND completion
    counters = np.zeros(9, dtype=np.float64)
    counters[6] = np.min(arrivals)
    counters[7] = np.inf
    link_arrivals = np.full(len(link_offsets) - 1, np.inf, dtype=np.float64)
    link_completions = np.zeros(len(link_offsets) - 1, dtype=np.float64)
    link_last = np.zeros((len(link_ids), 2), dtype=np.float64)
    for index in range(count):
        touched[:] = False  # A real core submission starts a fresh detail recorder.
        arrival = arrivals[index]
        if not read:
            arrival = _transfer_links(arrival, link_offsets, link_ids, link_durations, link_bytes, link_directions,
                starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched,
                link_busy, link_moved, link_arrivals, link_completions, link_last)
            touched[:] = False  # NAND submission follows the incoming links.
        counters[7] = min(counters[7], arrival)
        effective = max(arrival, acceptance)
        kept = 0
        for old in range(admission_count):
            if admission[old] > effective:
                admission[kept], admission_ops[kept] = admission[old], admission_ops[old]
                kept += 1
        admission_count = kept
        if admission_count >= max_outstanding:
            earliest = admission[0]
            for old in range(1, admission_count):
                earliest = min(earliest, admission[old])
            effective = max(effective, earliest)
            kept = 0
            for old in range(admission_count):
                if admission[old] > effective:
                    admission[kept], admission_ops[kept] = admission[old], admission_ops[old]
                    kept += 1
            admission_count = kept
        reads = int(read)
        writes = int(not read)
        for old in range(admission_count):
            reads += admission_ops[old] == 1
            writes += admission_ops[old] == 2
        counters[0] += effective - arrival
        counters[1] = max(counters[1], admission_count + 1)
        counters[2] = max(counters[2], reads)
        counters[3] = max(counters[3], writes)
        acceptance = effective
        base = effective + front
        page_index = addresses[index] // 4096
        base_unit = page_index % base_units
        plane = (page_index // base_units) % planes
        bank = base_unit + plane * base_units
        physical_page = page_index // (base_units * planes)
        unit = bank if independent else base_unit
        channel = base_unit // per_channel
        ar, ch, host = array_ids[unit], channel_ids[channel], host_ids[channel % host_channels]
        if not read:
            if cache_present[bank]:
                for candidate in range(2):
                    tag = cache_tags[bank, candidate]
                    if tag >= 0 and (tag == physical_page or (physical_page % pages_per_block == 0
                            and tag // pages_per_block == physical_page // pages_per_block)):
                        cache_tags[bank, candidate] = -1
                banks_touched[bank] = True
            before_array, before_channel, before_host = resource_state[ar, 1], resource_state[ch, 1], resource_state[host, 1]
            if physical_page % pages_per_block == 0:
                erased = _reserve(ar, max(base, array_ready[unit], buffer_ready[unit]), block_erase, -1, 0,
                    starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched)
                array_ready[unit] = erased
                counters[8] += 1
            host_end = _reserve(host, base, host_ns, 4096, 2, starts, lane_counts, lanes,
                                lane_directions, resource_state, moved, directions, touched)
            internal_end = _reserve(ch, max(host_end, buffer_ready[unit]), internal_ns,
                int(internal_bytes), 2, starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched)
            end = _reserve(ar, max(internal_end, array_ready[unit]), page_program, -1, 0,
                starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched)
            array_ready[unit], buffer_ready[unit] = end, end
            arrays_touched[unit] = True
            nand_busy[ar] += resource_state[ar, 1] - before_array
            nand_busy[ch] += resource_state[ch, 1] - before_channel
            nand_busy[host] += resource_state[host, 1] - before_host
            nand_bytes[ch] += int(internal_bytes)
            nand_bytes[host] += 4096
            end = max(base, end)
            counters[6] = max(counters[6], end)
            admission[admission_count], admission_ops[admission_count] = end, 2
            admission_count += 1
            completions[index] = end
            continue
        cache_clock += 1
        slot = -1
        for candidate in range(2):
            if cache_tags[bank, candidate] == physical_page:
                slot = candidate
                break
        hit = slot >= 0
        if not hit:
            slot = 0
            # Invalid slots sort before valid slots; LRU breaks ties.
            if ((cache_tags[bank, 1] < 0 and cache_tags[bank, 0] >= 0)
                    or ((cache_tags[bank, 1] < 0) == (cache_tags[bank, 0] < 0)
                        and cache_use[bank, 1] < cache_use[bank, 0])):
                slot = 1
        cache_use[bank, slot] = cache_clock
        banks_touched[bank] = True
        before_host = resource_state[host, 1]
        if hit:
            end = _reserve(host, max(base, cache_ready[bank, slot]), host_ns, 4096, 1,
                           starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched)
            cache_readers[bank, slot] = max(cache_readers[bank, slot], end)
            counters[5] += 1
        else:
            before_array, before_channel = resource_state[ar, 1], resource_state[ch, 1]
            read_end = _reserve(ar, max(base, array_ready[unit], buffer_ready[unit]), page_read, -1, 0,
                                starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched)
            internal_end = _reserve(ch, max(read_end, cache_readers[bank, slot]), internal_ns,
                                    int(internal_bytes), 1, starts, lane_counts, lanes, lane_directions,
                                    resource_state, moved, directions, touched)
            end = _reserve(host, internal_end, host_ns, 4096, 1, starts, lane_counts, lanes,
                           lane_directions, resource_state, moved, directions, touched)
            cache_tags[bank, slot] = physical_page
            cache_ready[bank, slot] = internal_end
            cache_readers[bank, slot] = end
            array_ready[unit], buffer_ready[unit] = read_end, internal_end
            arrays_touched[unit] = True
            nand_busy[ar] += resource_state[ar, 1] - before_array
            nand_busy[ch] += resource_state[ch, 1] - before_channel
            nand_bytes[ch] += int(internal_bytes)
            counters[4] += 1
        nand_busy[host] += resource_state[host, 1] - before_host
        nand_bytes[host] += 4096
        end = max(base, end)
        counters[6] = max(counters[6], end)
        admission[admission_count], admission_ops[admission_count] = end, 1
        admission_count += 1
        end = _transfer_links(end, link_offsets, link_ids, link_durations, link_bytes, link_directions,
            starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched,
            link_busy, link_moved, link_arrivals, link_completions, link_last)
        completions[index] = end
    return (completions, counters, cache_clock, admission_count, acceptance,
            touched, arrays_touched, banks_touched, link_arrivals, link_completions, link_last)


def execute_pages(core, addresses, arrival_ns, link_program, *, operation=Operation.READ):
    """Return exact aggregate page/link execution or decline before mutation."""
    from .nand_core import NandCore, _ReadCacheSlot
    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or core.capture_details or core._observe_media or config.protocol_mode != "ocp_hbf_0_7"):
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
        arrivals = np.full(count, float(arrivals), dtype=np.float64)
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
    lookup = {name: index for index, name in enumerate(names)}
    starts, lane_counts, lanes, lane_directions = [], [], [], []
    direction_names = [None, "read", "write"]
    direction_lookup = {name: index for index, name in enumerate(direction_names)}
    def direction_id(value):
        if value not in direction_lookup:
            direction_lookup[value] = len(direction_names)
            direction_names.append(value)
        return direction_lookup[value]
    state = np.zeros((len(names), 4), dtype=np.float64)
    moved = np.zeros(len(names), dtype=np.int64)
    directions = np.zeros(len(names), dtype=np.int64)
    seen_lanes = set()
    for index, name in enumerate(names):
        existing = timeline.lane_available.get(name)
        if existing is not None:
            if not existing or id(existing) in seen_lanes:
                return None
            seen_lanes.add(id(existing))
        values = existing if existing is not None else [timeline.ready_ns.get(name, 0.)]
        if any(not math.isfinite(value) or value < 0 for value in values):
            return None
        starts.append(len(lanes)); lane_counts.append(len(values)); lanes.extend(values)
        previous = timeline.directions.get(name)
        histories = timeline.lane_directions.get(name)
        if len(values) == 1 or histories is None or len(histories) != len(values):
            histories = [previous] * len(values)
        lane_directions.extend(direction_id(value) for value in histories)
        state[index] = [timeline.ready_ns.get(name, 0.), timeline.busy_ns.get(name, 0.),
                        *timeline.last_intervals.get(name, (0., 0.))]
        byte_count = timeline.bytes_moved.get(name, 0)
        if not isinstance(byte_count, int) or not 0 <= byte_count <= 2**63 - 1 - count * max(4096, config.transfer_page_bytes):
            return None
        moved[index] = byte_count
        directions[index] = direction_id(previous)
    arrays = np.array([core._array_ready.get(name, 0.) for name in array_names], dtype=np.float64)
    buffers = np.array([core._buffer_ready.get(name, 0.) for name in array_names], dtype=np.float64)
    tags = np.full((banks, 2), -1, dtype=np.int64)
    ready = np.zeros((banks, 2), dtype=np.float64)
    readers = np.zeros((banks, 2), dtype=np.float64)
    uses = np.zeros((banks, 2), dtype=np.int64)
    cache_present = np.zeros(banks, dtype=np.bool_)
    for bank, key in enumerate(bank_keys):
        slots = core._read_cache.get(key, ())
        if len(slots) not in (0, 2):
            return None
        cache_present[bank] = bool(slots)
        for slot, old in enumerate(slots):
            if old.page_key is not None:
                tags[bank, slot] = old.page_key[-2] * config.pages_per_block + old.page_key[-1]
            ready[bank, slot], readers[bank, slot], uses[bank, slot] = old.ready_ns, old.reader_completion_ns, old.last_use
    if [end for end, _ in core._inflight_operations] != core._inflight:
        return None
    admission = np.zeros(max(config.max_outstanding_requests, len(core._inflight)) + 1, dtype=np.float64)
    admission_ops = np.zeros(len(admission), dtype=np.int64)
    for index, (end, previous_operation) in enumerate(core._inflight_operations):
        admission[index], admission_ops[index] = end, 1 if previous_operation is Operation.READ else 2
    link_offsets, link_ids, durations, payloads, link_dirs = [0], [], [], [], []
    for _, direction, program in link_program:
        for name, duration, payload in program:
            if not math.isfinite(duration) or duration < 0 or not isinstance(payload, int) or payload < 0:
                return None
            link_ids.append(lookup[name]); durations.append(duration); payloads.append(payload)
        link_offsets.append(len(link_ids))
        link_dirs.append(direction_id(direction))
    starts, lane_counts = np.array(starts, dtype=np.int64), np.array(lane_counts, dtype=np.int64)
    lanes, lane_directions = np.array(lanes, dtype=np.float64), np.array(lane_directions, dtype=np.int64)
    geometry = np.array([base_units, config.planes_per_lun, config.planes_independent,
                         per_channel, config.host_channels, config.max_outstanding_requests,
                         config.pages_per_block], dtype=np.int64)
    timing = np.array([config.page_read_ns, config.transfer_page_bytes / config.internal_bandwidth_gb_s,
                       4096 / core._host_bandwidth(operation), config.front_ns, config.transfer_page_bytes,
                       config.page_program_ns, config.block_erase_ns])
    nand_busy = np.zeros(len(names), dtype=np.float64)
    nand_bytes = np.zeros(len(names), dtype=np.int64)
    link_busy = np.zeros(len(link_ids), dtype=np.float64)
    link_moved = np.zeros(len(link_ids), dtype=np.int64)
    answer = _run(addresses, arrivals, read, geometry, timing, array_ids, channel_ids, host_ids,
                  arrays, buffers, tags, ready, readers, uses, cache_present, core._cache_use,
                  admission, admission_ops, len(core._inflight), core._acceptance_ns,
                  starts, lane_counts, lanes, lane_directions, state, moved, directions,
                  np.array(link_offsets, dtype=np.int64), np.array(link_ids, dtype=np.int64),
                  np.array(durations, dtype=np.float64), np.array(payloads, dtype=np.int64),
                  np.array(link_dirs, dtype=np.int64), nand_busy, nand_bytes, link_busy, link_moved)
    (completions, counters, cache_clock, admitted_count, acceptance, touched, arrays_touched,
     banks_touched, link_arrivals, link_completions, link_last) = answer
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(completions)):
        raise ValueError("resource reservation exceeds finite timing range")
    # Commit only after numeric execution succeeds; no independent shadow state
    # survives the call, so ordinary snapshots/restores remain authoritative.
    active_resources = {names[index] for index in range(len(names)) if nand_busy[index] or nand_bytes[index]}
    active_resources.update(array_resources[index] for index in np.flatnonzero(arrays_touched))
    active_resources.update(link_names)
    for index, name in enumerate(names):
        if name not in active_resources:
            continue
        first, lane_count = starts[index], lane_counts[index]
        values = lanes[first:first + lane_count].tolist()
        if name in timeline.lane_available:
            timeline.lane_available[name][:] = values
        else:
            timeline.lane_available[name] = values
        timeline.ready_ns[name], timeline.busy_ns[name] = float(state[index, 0]), float(state[index, 1])
        timeline.last_intervals[name] = (float(state[index, 2]), float(state[index, 3]))
        if name in link_names or nand_bytes[index] or name in timeline.bytes_moved:
            timeline.bytes_moved[name] = int(moved[index])
        if directions[index]:
            timeline.directions[name] = direction_names[directions[index]]
        if lane_count > 1:
            timeline.lane_directions[name] = [direction_names[value] for value in lane_directions[first:first + lane_count]]
        else:
            timeline.lane_directions.pop(name, None)
    for unit in np.flatnonzero(arrays_touched):
        name = array_names[unit]
        core._array_ready[name], core._buffer_ready[name] = float(arrays[unit]), float(buffers[unit])
        timeline.ready_ns[name] = float(arrays[unit])
    for bank in np.flatnonzero(banks_touched):
        key = bank_keys[bank]
        slots = core._read_cache.get(key)
        if slots is None:
            slots = [_ReadCacheSlot(), _ReadCacheSlot()]
            core._read_cache[key] = slots
        for index, slot in enumerate(slots):
            page = int(tags[bank, index])
            slot.page_key = None if page < 0 else (*tag_prefixes[bank], *divmod(page, config.pages_per_block))
            slot.ready_ns, slot.reader_completion_ns, slot.last_use = float(ready[bank, index]), float(readers[bank, index]), int(uses[bank, index])
    core._cache_use, core._acceptance_ns = int(cache_clock), float(acceptance)
    core._next_program_page.update(program_frontiers)
    core._inflight = admission[:admitted_count].tolist()
    core._inflight_operations = [(float(admission[index]), Operation.READ if admission_ops[index] == 1 else Operation.WRITE)
                                 for index in range(admitted_count)]
    core._observation_events = []
    timeline._intervals, timeline._interval_payloads = {}, {}
    timeline._interval_count = timeline._interval_limit = 0
    timeline._intervals_truncated = True
    timeline._touched = {names[index] for index in np.flatnonzero(touched)}
    misses, hits = int(counters[4]), int(counters[5])
    busy = {name: float(nand_busy[index]) for index, name in enumerate(names) if nand_busy[index]}
    byte_counts = {name: int(nand_bytes[index]) for index, name in enumerate(names) if nand_bytes[index]}
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
            "host_admission_depth_scope": "outstanding host requests sampled at admission, including the admitted request; not NAND page queues",
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
