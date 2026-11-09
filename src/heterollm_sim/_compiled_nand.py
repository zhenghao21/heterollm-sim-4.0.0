"""Numba state machines for generic NAND execution.

Packing and publication live in _nand_acceleration and _nand_pack.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def run_generic_full_pages(first_page, count, read, page_indices, base, geometry, timing,
                           array_ids, channel_ids, host_ids, array_ready, buffer_ready,
                           tags, cache_ready, readers, uses, present, cache_clock,
                           starts, lane_counts, lanes, lane_directions, state, moved,
                           directions, head_bytes, head_ns, tail_bytes, tail_ns,
                           trace_program_start, trace_program_end,
                           trace_erase_start, trace_erase_end):
    """The ordinary generic page machine, with ordered IEEE additions.

    Admission remains one host request in NandCore.submit. Unlike the OCP
    kernel below, generic NAND has neither implicit page-0 erase nor PROGRAM
    frontier validation. No page or cache transition is skipped here. The
    first and last page move ``head_bytes``/``tail_bytes`` over the host
    (their host-granularity span of a read that starts or ends inside a
    page); every other page moves a full page. Writes are whole pages: the
    host's read-merge-write of a partial page runs in the scalar machine.
    """
    base_units, planes, independent, per_channel, host_channels = geometry
    page_read, page_program, block_erase, internal_ns, page_host_ns, internal_bytes, pages_per_block = timing
    touched = np.zeros(len(state), dtype=np.bool_)
    arrays_touched = np.zeros(len(array_ready), dtype=np.bool_)
    banks_touched = np.zeros(len(tags), dtype=np.bool_)
    # A request may queue behind a frontier left by an earlier transaction.
    # Snapshot it before this batch so later pages in the same batch do not
    # turn ordinary same-batch serialization into media queue wait.
    initial_frontier = np.maximum(array_ready.copy(), buffer_ready.copy())
    for unit in range(len(array_ready)):
        rid = array_ids[unit]
        first, count_lanes = starts[rid], lane_counts[rid]
        lane_frontier = lanes[first]
        for lane in range(first + 1, first + count_lanes):
            if lanes[lane] < lane_frontier:
                lane_frontier = lanes[lane]
        if lane_frontier > initial_frontier[unit]:
            initial_frontier[unit] = lane_frontier
    # Short resident requests previously ran scalar. Preserve reserve's
    # first-touch order, including set collision order in request metrics.
    resource_order = np.empty(len(state), dtype=np.int64)
    ordered = np.zeros(len(state), dtype=np.bool_)
    order_count = 0
    hits = misses = 0
    max_read_wait = 0.0
    completion = base
    erase_count = 0
    for offset in range(count):
        page_index = page_indices[offset]
        base_unit = page_index % base_units
        plane = (page_index // base_units) % planes
        bank = base_unit + plane * base_units
        physical_page = page_index // (base_units * planes)
        unit = bank if independent else base_unit
        channel = base_unit // per_channel
        ar, ch, host = array_ids[unit], channel_ids[channel], host_ids[channel % host_channels]
        host_ns, host_bytes = page_host_ns, 4096
        if offset == 0:
            host_ns, host_bytes = head_ns, head_bytes
        elif offset == count - 1:
            host_ns, host_bytes = tail_ns, tail_bytes
        if read:
            cache_clock += 1
            slot = -1
            for candidate in range(2):
                if tags[bank, candidate] == physical_page:
                    slot = candidate
                    break
            hit = slot >= 0
            if not hit:
                slot = 0
                if ((tags[bank, 1] < 0 and tags[bank, 0] >= 0)
                        or ((tags[bank, 1] < 0) == (tags[bank, 0] < 0)
                            and uses[bank, 1] < uses[bank, 0])):
                    slot = 1
            uses[bank, slot] = cache_clock
            present[bank] = True
            banks_touched[bank] = True
            if hit:
                end = _reserve(host, max(base, cache_ready[bank, slot]), host_ns, host_bytes, 1,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                readers[bank, slot] = max(readers[bank, slot], end)
                hits += 1
            else:
                eligible = max(base, array_ready[unit], buffer_ready[unit])
                read_end = _reserve(ar, eligible, page_read, -1, 0,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                # Report only the inherited array/buffer frontier.  The
                # snapshot excludes queueing caused by pages admitted by this
                # invocation, while retaining a real prior PROGRAM/read
                # collision.
                frontier_wait = initial_frontier[unit] - base
                if frontier_wait > max_read_wait:
                    max_read_wait = frontier_wait
                internal_end = _reserve(ch, max(read_end, readers[bank, slot]), internal_ns,
                    int(internal_bytes), 1, starts, lane_counts, lanes, lane_directions,
                    state, moved, directions, touched)
                end = _reserve(host, internal_end, host_ns, host_bytes, 1,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                tags[bank, slot], cache_ready[bank, slot] = physical_page, internal_end
                readers[bank, slot] = end
                array_ready[unit], buffer_ready[unit] = read_end, internal_end
                arrays_touched[unit] = True
                misses += 1
        else:
            # Invalidating a tag retains its live reader and its LRU age.
            if present[bank]:
                for slot in range(2):
                    if tags[bank, slot] == physical_page:
                        tags[bank, slot] = -1
                        banks_touched[bank] = True
            host_end = _reserve(host, base, host_ns, host_bytes, 2,
                starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
            internal_end = _reserve(ch, max(host_end, buffer_ready[unit]), internal_ns,
                int(internal_bytes), 2, starts, lane_counts, lanes, lane_directions,
                state, moved, directions, touched)
            program_start = max(internal_end, array_ready[unit])
            # OCP's block spans all planes in one channel.  Only the first
            # local-DLU (plane 0/page 0) opens it; reserve the same busy
            # interval on every plane array while charging one erase event.
            if (block_erase > 0.0 and plane == 0
                    and physical_page % int(pages_per_block) == 0):
                common_start = program_start
                group_first = channel * per_channel
                group_last = group_first + per_channel
                for group_base in range(group_first, group_last):
                    plane_count = planes if independent else 1
                    for group_plane in range(plane_count):
                        group_unit = group_base + group_plane * base_units if independent else group_base
                        group_front = array_ready[group_unit]
                        if group_front > common_start:
                            common_start = group_front
                common_end = common_start
                for group_base in range(group_first, group_last):
                    plane_count = planes if independent else 1
                    for group_plane in range(plane_count):
                        group_unit = group_base + group_plane * base_units if independent else group_base
                        group_rid = array_ids[group_unit]
                        end = _reserve(group_rid, common_start, block_erase, -1, 0,
                                       starts, lane_counts, lanes, lane_directions,
                                       state, moved, directions, touched)
                        array_ready[group_unit] = end
                        if end > common_end:
                            common_end = end
                        arrays_touched[group_unit] = True
                program_start = common_end
                erase_count += 1
                if trace_erase_start.size:
                    trace_erase_start[offset] = common_start
                    trace_erase_end[offset] = common_end
            if trace_program_start.size:
                trace_program_start[offset] = program_start
            end = _reserve(ar, program_start, page_program, -1, 0,
                starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
            if trace_program_end.size:
                trace_program_end[offset] = end
            array_ready[unit], buffer_ready[unit] = end, end
            arrays_touched[unit] = True
        if count < 128:
            if read and hit:
                first_rid, second_rid, third_rid = host, host, host
            elif read:
                first_rid, second_rid, third_rid = ar, ch, host
            else:
                first_rid, second_rid, third_rid = host, ch, ar
            for rid in (first_rid, second_rid, third_rid):
                if not ordered[rid]:
                    ordered[rid] = True
                    resource_order[order_count] = rid
                    order_count += 1
        completion = max(completion, end)
    return (completion, hits, misses, cache_clock, touched,
            arrays_touched, banks_touched, resource_order[:order_count], max_read_wait,
            erase_count)


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
def run_generic_flush_batch(pages, ready_times, start_times, geometry, page_program_ns,
                            block_erase_ns, pages_per_block,
                            internal_ns, internal_bytes, host_read_ns, host_write_ns,
                            array_ids, channel_ids, host_read_ids, host_write_ids,
                            array_ready, buffer_ready, tags,
                            starts, lane_counts, lanes, lane_directions, state, moved,
                            directions, trace_program_start, trace_program_end,
                            trace_erase_start, trace_erase_end):
    """Reserve a scratchpad page-program batch against resident NAND arrays.

    ``pages`` is already in the scalar flush order.  The scratchpad is the
    host's, with no internal path to NAND: for every page the host reads it
    back and writes 4 KiB, then the channel transfer and array PROGRAM run,
    while the controller/cache arrays and fabric calendars remain resident.
    """
    base_units, planes, independent, per_channel, host_channels = geometry
    touched = np.zeros(len(state), dtype=np.bool_)
    arrays_touched = np.zeros(len(array_ready), dtype=np.bool_)
    banks_touched = np.zeros(len(tags), dtype=np.bool_)
    ends = np.empty(len(pages), dtype=np.float64)
    banks = len(tags)
    slots = tags.shape[1]
    erase_count = 0

    # OCP cannot interrupt a plane erase.  A page-zero program therefore
    # reserves its erase before the program transfer reaches the array.  Do
    # that reservation in one pre-pass so all banks which cross a block in
    # this flush share the same release frontier.  Host/scratchpad transfers
    # still run in the main pass and may overlap this common erase window.
    erase_ends = np.empty(len(pages), dtype=np.float64)
    for index in range(len(pages)):
        erase_ends[index] = -1.0
    if block_erase_ns > 0.0:
        common_erase_start = 0.0
        for index in range(len(pages)):
            page_index = pages[index]
            base_unit = page_index % base_units
            plane = (page_index // base_units) % planes
            bank = base_unit + plane * base_units
            if bank < 0 or bank >= banks:
                raise ValueError("logic-die flush page is outside the resident bank geometry")
            physical_page = page_index // (base_units * planes)
            if (plane == 0 and physical_page % int(pages_per_block) == 0):
                unit = bank if independent else base_unit
                release = max(ready_times[index], start_times[index], array_ready[unit])
                if release > common_erase_start:
                    common_erase_start = release
        for index in range(len(pages)):
            page_index = pages[index]
            base_unit = page_index % base_units
            plane = (page_index // base_units) % planes
            bank = base_unit + plane * base_units
            physical_page = page_index // (base_units * planes)
            if (plane == 0 and physical_page % int(pages_per_block) == 0):
                channel = base_unit // per_channel
                group_first = channel * per_channel
                group_last = group_first + per_channel
                common_start = common_erase_start
                for group_base in range(group_first, group_last):
                    plane_count = planes if independent else 1
                    for group_plane in range(plane_count):
                        group_unit = group_base + group_plane * base_units if independent else group_base
                        if array_ready[group_unit] > common_start:
                            common_start = array_ready[group_unit]
                common_end = common_start
                for group_base in range(group_first, group_last):
                    plane_count = planes if independent else 1
                    for group_plane in range(plane_count):
                        group_unit = group_base + group_plane * base_units if independent else group_base
                        ar = array_ids[group_unit]
                        erase_end = _reserve(ar, common_start, block_erase_ns, -1, 0,
                                             starts, lane_counts, lanes, lane_directions,
                                             state, moved, directions, touched)
                        array_ready[group_unit] = erase_end
                        if erase_end > common_end:
                            common_end = erase_end
                        arrays_touched[group_unit] = True
                erase_ends[index] = common_end
                erase_count += 1
                if trace_erase_start.size:
                    trace_erase_start[index] = common_start
                    trace_erase_end[index] = common_end

    for index in range(len(pages)):
        page_index = pages[index]
        base_unit = page_index % base_units
        plane = (page_index // base_units) % planes
        bank = base_unit + plane * base_units
        if bank < 0 or bank >= banks:
            raise ValueError("logic-die flush page is outside the resident bank geometry")
        physical_page = page_index // (base_units * planes)
        unit = bank if independent else base_unit
        channel = base_unit // per_channel
        ar = array_ids[unit]
        ch = channel_ids[channel]
        link = channel % host_channels

        # _invalidate_cache only removes the tag; reader lifetimes and LRU
        # ages remain live until the next fill, so preserve all other fields.
        for slot in range(slots):
            if tags[bank, slot] == physical_page:
                tags[bank, slot] = -1
                banks_touched[bank] = True

        # Keep the same ordering as NandCore._tail_program: host read and
        # write, channel transfer, then the plane PROGRAM after both channel
        # and array fronts. ``start_times`` is the release frontier for the
        # whole program transfer: nothing begins before that point.
        release = max(ready_times[index], start_times[index])
        host_end = _reserve(host_read_ids[link], release, host_read_ns, internal_bytes, 1,
                            starts, lane_counts, lanes, lane_directions, state,
                            moved, directions, touched)
        host_end = _reserve(host_write_ids[link], host_end, host_write_ns, internal_bytes, 2,
                            starts, lane_counts, lanes, lane_directions, state,
                            moved, directions, touched)
        internal_start = max(host_end, buffer_ready[unit], start_times[index])
        internal_end = _reserve(ch, internal_start, internal_ns, internal_bytes, 2,
                                starts, lane_counts, lanes, lane_directions, state,
                                moved, directions, touched)
        program_start = max(internal_end, array_ready[unit], start_times[index])
        if erase_ends[index] >= 0.0 and erase_ends[index] > program_start:
            program_start = erase_ends[index]
        if trace_program_start.size:
            trace_program_start[index] = program_start
        program_end = _reserve(ar, program_start, page_program_ns, -1, 0,
                               starts, lane_counts, lanes, lane_directions, state,
                               moved, directions, touched)
        if trace_program_end.size:
            trace_program_end[index] = program_end
        array_ready[unit] = program_end
        buffer_ready[unit] = program_end
        arrays_touched[unit] = True
        ends[index] = program_end
    return ends, touched, arrays_touched, banks_touched, erase_count


# Read-batch events. At one timestamp, completions release buffers, lanes
# and credits before admission; enqueues precede the dispatch that chooses
# among them, so equal-ready pages always leave in submission order.
_HOST_END, _CHANNEL_END, _READ_END, _ADMIT, _ENQUEUE, _DISPATCH = 0, 1, 2, 3, 4, 5
_KIND_SHIFT = 56
_PAYLOAD_MASK = (1 << 56) - 1


@njit(cache=True, nogil=True, fastmath=False)
def _event_push(times, keys, size, time, key):
    at = size
    while at:
        parent = (at - 1) >> 1
        if times[parent] < time or (times[parent] == time and keys[parent] <= key):
            break
        times[at], keys[at] = times[parent], keys[parent]
        at = parent
    times[at], keys[at] = time, key
    return size + 1


@njit(cache=True, nogil=True, fastmath=False)
def _event_pop(times, keys, size):
    """Remove the minimum (time, key); the caller reads it first."""
    size -= 1
    time, key = times[size], keys[size]
    at = 0
    while True:
        child = 2 * at + 1
        if child >= size:
            break
        if child + 1 < size and (times[child + 1] < times[child] or (
                times[child + 1] == times[child] and keys[child + 1] < keys[child])):
            child += 1
        if time < times[child] or (time == times[child] and key <= keys[child]):
            break
        times[at], keys[at] = times[child], keys[child]
        at = child
    times[at], keys[at] = time, key
    return size


@njit(cache=True, nogil=True, fastmath=False)
def _queue_push(times, pages, first, size, time, page):
    at = size
    while at:
        parent = (at - 1) >> 1
        if times[first + parent] < time or (times[first + parent] == time and pages[first + parent] < page):
            break
        times[first + at], pages[first + at] = times[first + parent], pages[first + parent]
        at = parent
    times[first + at], pages[first + at] = time, page


@njit(cache=True, nogil=True, fastmath=False)
def _queue_pop(times, pages, first, size):
    page = pages[first]
    size -= 1
    time, last = times[first + size], pages[first + size]
    at = 0
    while True:
        child = 2 * at + 1
        if child >= size:
            break
        left, right = first + child, first + child + 1
        if child + 1 < size and (times[right] < times[left] or (
                times[right] == times[left] and pages[right] < pages[left])):
            child += 1
        if time < times[first + child] or (time == times[first + child] and last < pages[first + child]):
            break
        times[first + at], pages[first + at] = times[first + child], pages[first + child]
        at = child
    times[first + at], pages[first + at] = time, last
    return page


@njit(cache=True, nogil=True, fastmath=False)
def _arrive(page, rid, time, now, queue_times, queue_pages, queue_first, queue_size,
            dispatch_now, event_times, event_keys, size):
    """Make a page ready for one resource at ``time`` (never before now)."""
    if time > now:
        return _event_push(event_times, event_keys, size, time, (_ENQUEUE << _KIND_SHIFT) | page)
    _queue_push(queue_times, queue_pages, queue_first[rid], queue_size[rid], now, page)
    queue_size[rid] += 1
    if dispatch_now[rid] != now:
        dispatch_now[rid] = now
        size = _event_push(event_times, event_keys, size, now, (_DISPATCH << _KIND_SHIFT) | rid)
    return size


@njit(cache=True, nogil=True, fastmath=False)
def run_generic_read_batch(page_indices, page_counts, arrivals, head_bytes, head_ns, tail_bytes, tail_ns,
                           staged_ready, staging_ns, staging_host, request_queues, queue_acceptance,
                           prior_ends, prior_offsets, limit, front, geometry, timing,
                           array_ids, channel_ids, host_ids, stage_ids, array_ready, buffer_ready,
                           tags, cache_ready, readers, uses, present, cache_clock,
                           starts, lane_counts, lanes, lane_directions, state, moved, directions):
    """Out-of-order generic-NAND controller for one batch of READ requests.

    Admission, buffer choice and per-plane order follow submission order:
    every request takes a credit of its own queue (``request_queues``: one
    queue per OCP host channel, or the single device queue) in submission
    order, and queues admit independently, so a host channel whose credits
    are spent never delays another channel's requests. Queue ``q`` starts
    no earlier than ``queue_acceptance[q]`` and its credits are held by the
    sorted completions ``prior_ends[prior_offsets[q]:prior_offsets[q + 1]]``.
    Every page makes the same two-slot LRU hit/victim decision as a
    sequential controller, and each array unit reads its pages first-in
    first-out. A unit reads its next
    page only after the previous one left its page register over the
    channel. Shared resources are not first-come calendars: a channel
    moves whichever read page became eligible first (its target slot's
    previous content fully read by the host), the host link sends pages in
    the order their data reached the buffer, ties in submission order.
    Calendars start at their current frontiers, so later batches still
    queue behind this one. Durations, payloads and frontier, busy, byte and
    last-interval updates are those of the sequential page machine.

    A staged page (``staged_ready`` >= 0, the time its staged bytes are in
    the store) has no array read, die transfer or buffer slot. Its data is
    ready ``staging_ns`` after admission, but not before those bytes; with
    ``staging_host`` it then crosses its host channel's staging link
    (``stage_ids``: the OCP scratchpad lane, or the host link itself),
    otherwise no NAND resource serves it.

    ``page_indices`` holds every page's physical index, request after
    request: the bank, plane and cache tag follow where the host page
    table put the page.
    """
    base_units, planes, per_channel, host_channels = geometry[0], geometry[1], geometry[3], geometry[4]
    independent = geometry[2] != 0
    page_read, internal_ns, page_host_ns = timing[0], timing[1], timing[2]
    internal_bytes = np.int64(timing[3])
    requests = len(page_counts)
    offsets = np.empty(requests + 1, dtype=np.int64)
    offsets[0] = 0
    for request in range(requests):
        offsets[request + 1] = offsets[request] + page_counts[request]
    pages = offsets[requests]
    banks, units, resources = tags.shape[0], len(array_ready), len(starts)

    page_request = np.empty(pages, dtype=np.int64)
    page_unit = np.empty(pages, dtype=np.int64)
    page_channel = np.empty(pages, dtype=np.int64)
    page_host = np.empty(pages, dtype=np.int64)
    page_generation = np.empty(pages, dtype=np.int64)
    page_hit = np.zeros(pages, dtype=np.bool_)
    page_stage = np.empty(pages, dtype=np.int8)
    unit_next = np.empty(pages, dtype=np.int64)
    hit_next = np.empty(pages, dtype=np.int64)
    read_start = np.empty(pages, dtype=np.float64)
    read_eligible = np.full(pages, -1., dtype=np.float64)
    channel_start = np.empty(pages, dtype=np.float64)
    host_start = np.empty(pages, dtype=np.float64)
    # A generation is one fill of one buffer slot: its readers are the page
    # that filled it and later hits; the next fill waits for all of them.
    capacity = pages + 2 * min(banks, pages)
    generation_fill = np.empty(capacity, dtype=np.int64)
    generation_fill_end = np.empty(capacity, dtype=np.float64)
    generation_done = np.empty(capacity, dtype=np.bool_)
    generation_readers = np.empty(capacity, dtype=np.float64)
    generation_pending = np.empty(capacity, dtype=np.int64)
    generation_waiter = np.empty(capacity, dtype=np.int64)
    generation_hits = np.empty(capacity, dtype=np.int64)
    generation_previous = np.empty(capacity, dtype=np.int64)
    current = np.full((banks, 2), -1, dtype=np.int64)
    generations = 0
    unit_head = np.full(units, -1, dtype=np.int64)
    unit_tail = np.full(units, -1, dtype=np.int64)
    hits = np.zeros(requests, dtype=np.int64)
    misses = np.zeros(requests, dtype=np.int64)
    touched = np.zeros(resources, dtype=np.bool_)
    arrays_touched = np.zeros(units, dtype=np.bool_)
    banks_touched = np.zeros(banks, dtype=np.bool_)
    queue_capacity = np.zeros(resources, dtype=np.int64)
    for request in range(requests):
        for offset in range(page_counts[request]):
            page = offsets[request] + offset
            page_index = page_indices[page]
            base_unit = page_index % base_units
            bank = base_unit + ((page_index // base_units) % planes) * base_units
            physical_page = page_index // (base_units * planes)
            unit = bank if independent else base_unit
            channel = base_unit // per_channel
            page_request[page], page_unit[page], page_channel[page] = request, unit, channel
            page_host[page] = host_ids[channel % host_channels]
            if staged_ready[page] >= 0:
                page_unit[page] = -1
                page_stage[page] = 2
                page_generation[page] = -1
                page_host[page] = stage_ids[channel % host_channels]
                if staging_host:
                    queue_capacity[page_host[page]] += 1
                continue
            cache_clock += 1
            slot = -1
            for candidate in range(2):
                if tags[bank, candidate] == physical_page:
                    slot = candidate
                    break
            hit = slot >= 0
            if not hit:
                slot = 0
                # Invalid slots sort before valid slots; LRU breaks ties.
                if ((tags[bank, 1] < 0 and tags[bank, 0] >= 0)
                        or ((tags[bank, 1] < 0) == (tags[bank, 0] < 0)
                            and uses[bank, 1] < uses[bank, 0])):
                    slot = 1
            uses[bank, slot] = cache_clock
            present[bank] = True
            banks_touched[bank] = True
            generation = current[bank, slot]
            if generation < 0:
                generation = generations
                generations += 1
                generation_fill[generation] = -1
                generation_fill_end[generation] = cache_ready[bank, slot]
                generation_done[generation] = True
                generation_readers[generation] = readers[bank, slot]
                generation_pending[generation] = 0
                generation_waiter[generation] = -1
                generation_hits[generation] = -1
                generation_previous[generation] = -1
                current[bank, slot] = generation
            queue_capacity[page_host[page]] += 1
            if hit:
                page_hit[page] = True
                page_stage[page] = 2
                page_generation[page] = generation
                generation_pending[generation] += 1
                hits[request] += 1
                continue
            fill = generations
            generations += 1
            generation_fill[fill] = page
            generation_fill_end[fill] = 0.
            generation_done[fill] = False
            generation_readers[fill] = 0.
            generation_pending[fill] = 1
            generation_waiter[fill] = -1
            generation_hits[fill] = -1
            generation_previous[fill] = generation
            current[bank, slot] = fill
            tags[bank, slot] = physical_page
            page_generation[page] = fill
            page_stage[page] = 0
            unit_next[page] = -1
            if unit_tail[unit] < 0:
                unit_head[unit] = page
            else:
                unit_next[unit_tail[unit]] = page
            unit_tail[unit] = page
            arrays_touched[unit] = True
            misses[request] += 1
    # Each unit holds at most one page between its array read and channel end.
    for unit in range(units):
        if arrays_touched[unit]:
            base_unit = unit % base_units if independent else unit
            queue_capacity[array_ids[unit]] += 1
            queue_capacity[channel_ids[base_unit // per_channel]] += 1
    queue_first = np.empty(resources, dtype=np.int64)
    total = 0
    for rid in range(resources):
        queue_first[rid] = total
        total += queue_capacity[rid]
    queue_times = np.empty(total, dtype=np.float64)
    queue_pages = np.empty(total, dtype=np.int64)
    queue_size = np.zeros(resources, dtype=np.int64)
    dispatch_now = np.full(resources, -1., dtype=np.float64)
    dispatch_later = np.full(resources, -1., dtype=np.float64)
    # Every page owns at most one pending event; admission retries are
    # bounded by credit releases, inherited-frontier dispatches by lanes.
    event_capacity = pages + 2 * requests + len(prior_ends) + resources + len(lanes) + 16
    event_times = np.empty(event_capacity, dtype=np.float64)
    event_keys = np.empty(event_capacity, dtype=np.int64)

    unit_busy = np.zeros(units, dtype=np.bool_)
    unit_free = np.empty(units, dtype=np.float64)
    for unit in range(units):
        unit_free[unit] = max(array_ready[unit], buffer_ready[unit])
    admitted = np.zeros(requests, dtype=np.bool_)
    bases = np.zeros(requests, dtype=np.float64)
    admissions = np.zeros(requests, dtype=np.float64)
    completions = np.zeros(requests, dtype=np.float64)
    remaining = page_counts.copy()
    done_at = np.zeros(requests, dtype=np.float64)
    # Each queue admits its own requests in submission order.
    queues = len(queue_acceptance)
    queue_end = np.zeros(queues, dtype=np.int64)
    for request in range(requests):
        queue_end[request_queues[request]] += 1
    queue_next = np.empty(queues, dtype=np.int64)
    total = 0
    for queue in range(queues):
        queue_next[queue] = total
        total += queue_end[queue]
        queue_end[queue] = total
    queue_order = np.empty(requests, dtype=np.int64)
    for request in range(requests):
        queue = request_queues[request]
        queue_order[queue_next[queue]] = request
        queue_next[queue] += 1
    outstanding = np.empty(queues, dtype=np.int64)
    prior = np.empty(queues, dtype=np.int64)
    blocked = np.zeros(queues, dtype=np.bool_)
    size = 0
    for queue in range(queues):
        queue_next[queue] = queue_end[queue - 1] if queue else 0
        outstanding[queue] = prior_offsets[queue + 1] - prior_offsets[queue]
        prior[queue] = prior_offsets[queue]
        if queue_next[queue] < queue_end[queue]:
            request = queue_order[queue_next[queue]]
            size = _event_push(event_times, event_keys, size, max(arrivals[request], queue_acceptance[queue]),
                               (_ADMIT << _KIND_SHIFT) | request)
    while size:
        now, key = event_times[0], event_keys[0]
        size = _event_pop(event_times, event_keys, size)
        kind, item = key >> _KIND_SHIFT, key & _PAYLOAD_MASK
        if kind == _HOST_END:
            generation = page_generation[item]
            if generation >= 0:
                generation_pending[generation] -= 1
                if now > generation_readers[generation]:
                    generation_readers[generation] = now
            if generation >= 0 and generation_pending[generation] == 0 and generation_waiter[generation] >= 0:
                waiter = generation_waiter[generation]
                generation_waiter[generation] = -1
                size = _arrive(waiter, channel_ids[page_channel[waiter]],
                    max(now, generation_readers[generation]), now, queue_times, queue_pages,
                    queue_first, queue_size, dispatch_now, event_times, event_keys, size)
            request = page_request[item]
            if now > done_at[request]:
                done_at[request] = now
            remaining[request] -= 1
            if not remaining[request]:
                completions[request] = max(bases[request], done_at[request])
                queue = request_queues[request]
                outstanding[queue] -= 1
                if blocked[queue]:
                    size = _event_push(event_times, event_keys, size, now,
                                       (_ADMIT << _KIND_SHIFT) | queue_order[queue_next[queue]])
            rid = page_host[item]
            if queue_size[rid] and dispatch_now[rid] != now:
                dispatch_now[rid] = now
                size = _event_push(event_times, event_keys, size, now, (_DISPATCH << _KIND_SHIFT) | rid)
        elif kind == _CHANNEL_END:
            unit = page_unit[item]
            unit_busy[unit] = False
            unit_free[unit] = now
            array_ready[unit] = read_start[item] + page_read
            buffer_ready[unit] = now
            generation = page_generation[item]
            generation_done[generation] = True
            generation_fill_end[generation] = now
            page_stage[item] = 2
            host = page_host[item]
            size = _arrive(item, host, now, now, queue_times, queue_pages, queue_first, queue_size,
                           dispatch_now, event_times, event_keys, size)
            waiting = generation_hits[generation]
            generation_hits[generation] = -1
            while waiting >= 0:
                following = hit_next[waiting]
                size = _arrive(waiting, host, max(now, bases[page_request[waiting]]), now, queue_times,
                               queue_pages, queue_first, queue_size, dispatch_now, event_times, event_keys, size)
                waiting = following
            head = unit_head[unit]
            if head >= 0 and admitted[page_request[head]]:
                unit_busy[unit] = True
                unit_head[unit] = unit_next[head]
                arrival = max(now, bases[page_request[head]])
                read_eligible[head] = bases[page_request[head]]
                size = _arrive(head, array_ids[unit], arrival, now,
                               queue_times, queue_pages, queue_first, queue_size, dispatch_now,
                               event_times, event_keys, size)
            rid = channel_ids[page_channel[item]]
            if queue_size[rid] and dispatch_now[rid] != now:
                dispatch_now[rid] = now
                size = _event_push(event_times, event_keys, size, now, (_DISPATCH << _KIND_SHIFT) | rid)
        elif kind == _READ_END:
            page_stage[item] = 1
            previous = generation_previous[page_generation[item]]
            if generation_pending[previous]:
                generation_waiter[previous] = item
            else:
                size = _arrive(item, channel_ids[page_channel[item]], max(now, generation_readers[previous]),
                               now, queue_times, queue_pages, queue_first, queue_size, dispatch_now,
                               event_times, event_keys, size)
            rid = array_ids[page_unit[item]]
            if queue_size[rid] and dispatch_now[rid] != now:
                dispatch_now[rid] = now
                size = _event_push(event_times, event_keys, size, now, (_DISPATCH << _KIND_SHIFT) | rid)
        elif kind == _ADMIT:
            request = item
            queue = request_queues[request]
            position = queue_next[queue]
            if position == queue_end[queue] or queue_order[position] != request:
                continue
            stop = prior_offsets[queue + 1]
            while prior[queue] < stop and prior_ends[prior[queue]] <= now:
                prior[queue] += 1
                outstanding[queue] -= 1
            if outstanding[queue] >= limit:
                # Retry at the queue's next known prior completion; a
                # completing batch request of the queue also retries.
                blocked[queue] = True
                if prior[queue] < stop:
                    size = _event_push(event_times, event_keys, size, prior_ends[prior[queue]],
                                       (_ADMIT << _KIND_SHIFT) | request)
                continue
            blocked[queue] = False
            admitted[request] = True
            admissions[request] = now
            base = now + front
            bases[request] = base
            outstanding[queue] += 1
            position += 1
            queue_next[queue] = position
            if position < queue_end[queue]:
                following = queue_order[position]
                size = _event_push(event_times, event_keys, size, max(arrivals[following], now),
                                   (_ADMIT << _KIND_SHIFT) | following)
            for page in range(offsets[request], offsets[request + 1]):
                if staged_ready[page] >= 0:
                    if staging_host:
                        size = _arrive(page, page_host[page],
                                       max(base + staging_ns, staged_ready[page]),
                                       now, queue_times, queue_pages, queue_first, queue_size, dispatch_now,
                                       event_times, event_keys, size)
                    else:
                        host_start[page] = -1.
                        size = _event_push(event_times, event_keys, size,
                                           max(base + staging_ns, staged_ready[page]),
                                           (np.int64(_HOST_END) << _KIND_SHIFT) | page)
                    continue
                if page_hit[page]:
                    generation = page_generation[page]
                    if generation_done[generation]:
                        size = _arrive(page, page_host[page],
                                       max(base, generation_fill_end[generation]), now, queue_times,
                                       queue_pages, queue_first, queue_size, dispatch_now,
                                       event_times, event_keys, size)
                    else:
                        hit_next[page] = generation_hits[generation]
                        generation_hits[generation] = page
                    continue
                unit = page_unit[page]
                if not unit_busy[unit] and unit_head[unit] == page:
                    unit_busy[unit] = True
                    unit_head[unit] = unit_next[page]
                    arrival = max(base, unit_free[unit])
                    # Keep the pre-array request readiness as the metric
                    # origin.  ``unit_free`` may be an inherited program
                    # frontier; folding it into eligible would hide the
                    # collision from the published read-wait field.
                    read_eligible[page] = base
                    size = _arrive(page, array_ids[unit], arrival, now,
                                   queue_times, queue_pages, queue_first, queue_size, dispatch_now,
                                   event_times, event_keys, size)
        elif kind == _ENQUEUE:
            stage = page_stage[item]
            if stage == 0:
                rid = array_ids[page_unit[item]]
            elif stage == 1:
                rid = channel_ids[page_channel[item]]
            else:
                rid = page_host[item]
            size = _arrive(item, rid, now, now, queue_times, queue_pages, queue_first, queue_size,
                           dispatch_now, event_times, event_keys, size)
        else:
            rid = item
            if dispatch_now[rid] == now:
                dispatch_now[rid] = -1.
            first, count = starts[rid], lane_counts[rid]
            while queue_size[rid]:
                lane = first
                for candidate in range(first + 1, first + count):
                    if lanes[candidate] < lanes[lane]:
                        lane = candidate
                if lanes[lane] > now:
                    # Only a frontier inherited from earlier work can lie
                    # ahead; lanes freed in this batch dispatch at their end.
                    if dispatch_later[rid] != lanes[lane]:
                        dispatch_later[rid] = lanes[lane]
                        size = _event_push(event_times, event_keys, size, lanes[lane],
                                           (_DISPATCH << _KIND_SHIFT) | rid)
                    break
                page = _queue_pop(queue_times, queue_pages, queue_first[rid], queue_size[rid])
                queue_size[rid] -= 1
                stage = page_stage[page]
                direction = 1
                if stage == 0:
                    duration, payload, direction, ending = page_read, np.int64(-1), 0, _READ_END
                    read_start[page] = now
                elif stage == 1:
                    duration, payload, ending = internal_ns, internal_bytes, _CHANNEL_END
                    channel_start[page] = now
                else:
                    request = page_request[page]
                    offset = page - offsets[request]
                    if offset == 0:
                        duration, payload = head_ns[request], head_bytes[request]
                    elif offset == page_counts[request] - 1:
                        duration, payload = tail_ns[request], tail_bytes[request]
                    else:
                        duration, payload = page_host_ns, np.int64(4096)
                    ending = _HOST_END
                    host_start[page] = now
                end = now + duration
                lanes[lane] = end
                ready = lanes[first]
                for candidate in range(first + 1, first + count):
                    ready = min(ready, lanes[candidate])
                state[rid, 0] = ready
                state[rid, 1] += duration
                state[rid, 2] = now
                state[rid, 3] = end
                if direction:
                    directions[rid] = direction
                    lane_directions[lane] = direction
                if payload >= 0:
                    moved[rid] += payload
                touched[rid] = True
                size = _event_push(event_times, event_keys, size, end, (np.int64(ending) << _KIND_SHIFT) | page)
    for request in range(requests):
        if remaining[request]:
            raise RuntimeError("NAND read batch left a request unfinished")
    for bank in range(banks):
        if banks_touched[bank]:
            for slot in range(2):
                generation = current[bank, slot]
                if generation >= 0:
                    readers[bank, slot] = generation_readers[generation]
                    if generation_fill[generation] >= 0:
                        cache_ready[bank, slot] = generation_fill_end[generation]
    return (admissions, completions, hits, misses, cache_clock, touched, arrays_touched, banks_touched,
            offsets, page_hit, page_unit, page_channel, page_host, read_start, channel_start, host_start,
            read_eligible,
            )


@njit(cache=True, nogil=True, fastmath=False)
def read_batch_request_metrics(offsets, page_counts, page_hit, page_unit, page_channel, page_host,
                               read_start, read_eligible, channel_start, host_start,
                               head_bytes, head_ns,
                               tail_bytes, tail_ns, timing,
                               array_ids, channel_ids, initial_frontier,
                               resources):
    """Each request's own bills plus inherited-frontier read wait.

    ``initial_frontier`` is captured before the batch starts; it prevents
    pages queued behind one another in this invocation from inflating the
    published media wait metric.
    """
    page_read, internal_ns, page_host_ns = timing[0], timing[1], timing[2]
    internal_bytes = np.int64(timing[3])
    requests = len(page_counts)
    stamp = np.full(resources, -1, dtype=np.int64)
    position = np.empty(resources, dtype=np.int64)
    request_offsets = np.empty(requests + 1, dtype=np.int64)
    request_offsets[0] = 0
    total = 0
    for request in range(requests):
        for page in range(offsets[request], offsets[request + 1]):
            if not page_hit[page] and page_unit[page] >= 0:
                rid = array_ids[page_unit[page]]
                if stamp[rid] != request:
                    stamp[rid] = request
                    total += 1
                rid = channel_ids[page_channel[page]]
                if stamp[rid] != request:
                    stamp[rid] = request
                    total += 1
            if host_start[page] < 0:
                continue
            rid = page_host[page]
            if stamp[rid] != request:
                stamp[rid] = request
                total += 1
        request_offsets[request + 1] = total
    ids = np.empty(total, dtype=np.int64)
    busy = np.zeros(total, dtype=np.float64)
    moved = np.zeros(total, dtype=np.int64)
    first = np.empty(total, dtype=np.float64)
    last = np.empty(total, dtype=np.float64)
    read_wait = np.zeros(requests, dtype=np.float64)
    stamp[:] = -1
    count = 0
    for request in range(requests):
        for page in range(offsets[request], offsets[request + 1]):
            if not page_hit[page] and page_unit[page] >= 0 and read_eligible[page] >= 0.:
                # ``read_eligible`` is the request base, not the live array
                # queue.  Compare the immutable pre-batch frontier so pages
                # queued behind one another in this batch are not charged.
                wait = initial_frontier[page_unit[page]] - read_eligible[page]
                if wait > read_wait[request]:
                    read_wait[request] = wait
            for stage in range(3):
                if stage < 2 and (page_hit[page] or page_unit[page] < 0):
                    continue
                if stage == 2 and host_start[page] < 0:
                    continue
                if stage == 0:
                    rid, duration, payload, start = array_ids[page_unit[page]], page_read, np.int64(0), read_start[page]
                elif stage == 1:
                    rid, duration, payload, start = channel_ids[page_channel[page]], internal_ns, internal_bytes, channel_start[page]
                else:
                    rid, start = page_host[page], host_start[page]
                    offset = page - offsets[request]
                    if offset == 0:
                        duration, payload = head_ns[request], head_bytes[request]
                    elif offset == page_counts[request] - 1:
                        duration, payload = tail_ns[request], tail_bytes[request]
                    else:
                        duration, payload = page_host_ns, np.int64(4096)
                end = start + duration
                if stamp[rid] != request:
                    stamp[rid] = request
                    position[rid] = count
                    ids[count], first[count], last[count] = rid, start, end
                    count += 1
                index = position[rid]
                busy[index] += duration
                moved[index] += payload
                if end > last[index] or (end == last[index] and start > first[index]):
                    first[index], last[index] = start, end
    return request_offsets, ids, busy, moved, first, last, read_wait


@njit(cache=True, nogil=True, fastmath=False)
def read_batch_sink_totals(ids, busy, moved, first, last, task_busy, owner_busy, resources):
    """Fold per-request bills into a task sink as sequential request additions.

    ``task_busy``/``owner_busy`` hold each resource's current bill (zero when
    absent). Busy time is added request by request, exactly as consuming
    each request's map would; the latest interval keeps the earliest
    request on ties. ``order`` lists resources by first appearance and
    ``byte_order`` by first nonzero payload, the order in which request byte
    maps (which omit zero payloads) would insert them.
    """
    seen = np.zeros(resources, dtype=np.bool_)
    moved_seen = np.zeros(resources, dtype=np.bool_)
    order = np.empty(len(ids), dtype=np.int64)
    byte_order = np.empty(len(ids), dtype=np.int64)
    count = 0
    byte_count = 0
    byte_totals = np.zeros(resources, dtype=np.int64)
    last_first = np.zeros(resources, dtype=np.float64)
    last_end = np.zeros(resources, dtype=np.float64)
    for entry in range(len(ids)):
        rid = ids[entry]
        task_busy[rid] += busy[entry]
        owner_busy[rid] += busy[entry]
        byte_totals[rid] += moved[entry]
        if moved[entry] != 0 and not moved_seen[rid]:
            moved_seen[rid] = True
            byte_order[byte_count] = rid
            byte_count += 1
        if not seen[rid]:
            seen[rid] = True
            order[count] = rid
            count += 1
            last_first[rid], last_end[rid] = first[entry], last[entry]
        elif last[entry] > last_end[rid] or (last[entry] == last_end[rid] and first[entry] > last_first[rid]):
            last_first[rid], last_end[rid] = first[entry], last[entry]
    return order[:count], byte_order[:byte_count], byte_totals, last_first, last_end
