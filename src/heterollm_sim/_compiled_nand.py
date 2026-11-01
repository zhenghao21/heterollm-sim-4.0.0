"""Numba state machines for generic NAND execution.

Packing, publication and public transactions live in _nand_execution.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def run_generic_full_pages(first_page, count, read, base, geometry, timing,
                           array_ids, channel_ids, host_ids, array_ready, buffer_ready,
                           tags, cache_ready, readers, uses, present, cache_clock,
                           starts, lane_counts, lanes, lane_directions, state, moved,
                           directions, head_bytes, head_ns, tail_bytes, tail_ns,
                           head_rmw, tail_rmw):
    """The ordinary generic page machine, with ordered IEEE additions.

    Admission remains one host request in NandCore.submit. Unlike the OCP
    kernel below, generic NAND has neither implicit page-0 erase nor PROGRAM
    frontier validation. No page or cache transition is skipped here. The
    first and last page move ``head_bytes``/``tail_bytes`` over the host
    (their host-granularity span of a span that starts or ends inside a
    page); every other page moves a full page. A partially written first or
    last page (``head_rmw``/``tail_rmw``) first pays the read-modify-write
    old-page read and its internal transfer, as the scalar machine does.
    """
    base_units, planes, independent, per_channel, host_channels = geometry
    page_read, page_program, internal_ns, page_host_ns, internal_bytes = timing
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
    hits = misses = rmw_reads = 0
    max_read_wait = 0.0
    completion = base
    for offset in range(count):
        page_index = first_page + offset
        base_unit = page_index % base_units
        plane = (page_index // base_units) % planes
        bank = base_unit + plane * base_units
        physical_page = page_index // (base_units * planes)
        unit = bank if independent else base_unit
        channel = base_unit // per_channel
        ar, ch, host = array_ids[unit], channel_ids[channel], host_ids[channel % host_channels]
        host_ns, host_bytes, rmw = page_host_ns, 4096, False
        if offset == 0:
            host_ns, host_bytes, rmw = head_ns, head_bytes, head_rmw
        elif offset == count - 1:
            host_ns, host_bytes, rmw = tail_ns, tail_bytes, tail_rmw
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
            if rmw:
                # The old page is read into the page buffer before the new
                # bytes arrive; the array frontier itself is not advanced.
                eligible = max(base, array_ready[unit], buffer_ready[unit])
                read_end = _reserve(ar, eligible, page_read, -1, 0,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                frontier_wait = initial_frontier[unit] - base
                if frontier_wait > max_read_wait:
                    max_read_wait = frontier_wait
                buffer_ready[unit] = _reserve(ch, read_end, internal_ns, int(internal_bytes), 1,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                arrays_touched[unit] = True
                rmw_reads += 1
            host_end = _reserve(host, base, host_ns, host_bytes, 2,
                starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
            internal_end = _reserve(ch, max(host_end, buffer_ready[unit]), internal_ns,
                int(internal_bytes), 2, starts, lane_counts, lanes, lane_directions,
                state, moved, directions, touched)
            end = _reserve(ar, max(internal_end, array_ready[unit]), page_program, -1, 0,
                starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
            array_ready[unit], buffer_ready[unit] = end, end
            arrays_touched[unit] = True
        if count < 128:
            if read and hit:
                first_rid, second_rid, third_rid = host, host, host
            elif read or rmw:
                first_rid, second_rid, third_rid = ar, ch, host
            else:
                first_rid, second_rid, third_rid = host, ch, ar
            for rid in (first_rid, second_rid, third_rid):
                if not ordered[rid]:
                    ordered[rid] = True
                    resource_order[order_count] = rid
                    order_count += 1
        completion = max(completion, end)
    return (completion, hits, misses, rmw_reads, cache_clock, touched,
            arrays_touched, banks_touched, resource_order[:order_count], max_read_wait)


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
                            internal_ns, internal_bytes, array_ids, channel_ids,
                            array_ready, buffer_ready, tags,
                            starts, lane_counts, lanes, lane_directions, state, moved,
                            directions):
    """Reserve a logic-die page-program batch against resident NAND arrays.

    ``pages`` is already in the scalar flush order.  The kernel performs the
    same channel transfer followed by array PROGRAM for every page while the
    controller/cache arrays and fabric calendars remain resident.  No host
    transfer or read-modify-write is involved for a full logic-die page.
    """
    base_units, planes, independent, per_channel, _host_channels = geometry
    touched = np.zeros(len(state), dtype=np.bool_)
    arrays_touched = np.zeros(len(array_ready), dtype=np.bool_)
    banks_touched = np.zeros(len(tags), dtype=np.bool_)
    ends = np.empty(len(pages), dtype=np.float64)
    banks = len(tags)
    slots = tags.shape[1]
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

        # _invalidate_cache only removes the tag; reader lifetimes and LRU
        # ages remain live until the next fill, so preserve all other fields.
        for slot in range(slots):
            if tags[bank, slot] == physical_page:
                tags[bank, slot] = -1
                banks_touched[bank] = True

        # Keep the same ordering as NandCore._tail_program: channel transfer
        # first, then the plane PROGRAM after both channel and array fronts.
        # ``start_times`` is the release frontier for the whole program
        # transfer.  A channel transfer must not begin before that point;
        # scalar ``_tail_program_channel_start`` uses the same rule.
        internal_start = max(ready_times[index], buffer_ready[unit], start_times[index])
        internal_end = _reserve(ch, internal_start, internal_ns, internal_bytes, 2,
                                starts, lane_counts, lanes, lane_directions, state,
                                moved, directions, touched)
        program_start = max(internal_end, array_ready[unit], start_times[index])
        program_end = _reserve(ar, program_start, page_program_ns, -1, 0,
                               starts, lane_counts, lanes, lane_directions, state,
                               moved, directions, touched)
        array_ready[unit] = program_end
        buffer_ready[unit] = program_end
        arrays_touched[unit] = True
        ends[index] = program_end
    return ends, touched, arrays_touched, banks_touched


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
def run_generic_read_batch(first_pages, page_counts, arrivals, head_bytes, head_ns, tail_bytes, tail_ns,
                           staged_ready, staging_ns, staging_host, acceptance, prior_ends, limit, front, geometry, timing,
                           array_ids, channel_ids, host_ids, array_ready, buffer_ready,
                           tags, cache_ready, readers, uses, present, cache_clock,
                           starts, lane_counts, lanes, lane_directions, state, moved, directions):
    """Out-of-order generic-NAND controller for one batch of READ requests.

    Admission, buffer choice and per-plane order follow submission order:
    requests take device credits in order (``prior_ends`` are the sorted
    completions already holding credits), every page makes the same
    two-slot LRU hit/victim decision as a sequential controller, and each
    array unit reads its pages first-in first-out. A unit reads its next
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
    ``staging_host`` it then crosses the host link like any other page,
    otherwise no NAND resource serves it.
    """
    base_units, planes, per_channel, host_channels = geometry[0], geometry[1], geometry[3], geometry[4]
    independent = geometry[2] != 0
    page_read, internal_ns, page_host_ns = timing[0], timing[1], timing[2]
    internal_bytes = np.int64(timing[3])
    requests = len(first_pages)
    offsets = np.empty(requests + 1, dtype=np.int64)
    offsets[0] = 0
    for request in range(requests):
        offsets[request + 1] = offsets[request] + page_counts[request]
    pages = offsets[requests]
    banks, units, resources = tags.shape[0], len(array_ready), len(starts)

    page_request = np.empty(pages, dtype=np.int64)
    page_unit = np.empty(pages, dtype=np.int64)
    page_channel = np.empty(pages, dtype=np.int64)
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
            page_index = first_pages[request] + offset
            base_unit = page_index % base_units
            bank = base_unit + ((page_index // base_units) % planes) * base_units
            physical_page = page_index // (base_units * planes)
            unit = bank if independent else base_unit
            channel = base_unit // per_channel
            page_request[page], page_unit[page], page_channel[page] = request, unit, channel
            if staged_ready[page] >= 0:
                page_unit[page] = -1
                page_stage[page] = 2
                page_generation[page] = -1
                if staging_host:
                    queue_capacity[host_ids[channel % host_channels]] += 1
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
            queue_capacity[host_ids[channel % host_channels]] += 1
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
    outstanding = len(prior_ends)
    prior = 0
    next_request = 0
    blocked = False
    size = _event_push(event_times, event_keys, 0, max(arrivals[0], acceptance), (_ADMIT << _KIND_SHIFT) | 0)
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
                outstanding -= 1
                if blocked:
                    size = _event_push(event_times, event_keys, size, now, (_ADMIT << _KIND_SHIFT) | next_request)
            rid = host_ids[page_channel[item] % host_channels]
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
            host = host_ids[page_channel[item] % host_channels]
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
            if request != next_request:
                continue
            while prior < len(prior_ends) and prior_ends[prior] <= now:
                prior += 1
                outstanding -= 1
            if outstanding >= limit:
                # Retry at the next known prior completion; a completing
                # batch request also retries while admission is blocked.
                blocked = True
                if prior < len(prior_ends):
                    size = _event_push(event_times, event_keys, size, prior_ends[prior],
                                       (_ADMIT << _KIND_SHIFT) | request)
                continue
            blocked = False
            admitted[request] = True
            admissions[request] = now
            base = now + front
            bases[request] = base
            outstanding += 1
            next_request += 1
            if next_request < requests:
                size = _event_push(event_times, event_keys, size, max(arrivals[next_request], now),
                                   (_ADMIT << _KIND_SHIFT) | next_request)
            for page in range(offsets[request], offsets[request + 1]):
                if staged_ready[page] >= 0:
                    if staging_host:
                        size = _arrive(page, host_ids[page_channel[page] % host_channels],
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
                        size = _arrive(page, host_ids[page_channel[page] % host_channels],
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
                rid = host_ids[page_channel[item] % host_channels]
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
            offsets, page_hit, page_unit, page_channel, read_start, channel_start, host_start,
            read_eligible,
            )


@njit(cache=True, nogil=True, fastmath=False)
def read_batch_request_metrics(offsets, page_counts, page_hit, page_unit, page_channel,
                               read_start, read_eligible, channel_start, host_start,
                               head_bytes, head_ns,
                               tail_bytes, tail_ns, timing, host_channels,
                               array_ids, channel_ids, host_ids, initial_frontier,
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
            rid = host_ids[page_channel[page] % host_channels]
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
                    rid, start = host_ids[page_channel[page] % host_channels], host_start[page]
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
    request on ties. ``order`` lists resources by first appearance.
    """
    seen = np.zeros(resources, dtype=np.bool_)
    order = np.empty(len(ids), dtype=np.int64)
    count = 0
    byte_totals = np.zeros(resources, dtype=np.int64)
    last_first = np.zeros(resources, dtype=np.float64)
    last_end = np.zeros(resources, dtype=np.float64)
    for entry in range(len(ids)):
        rid = ids[entry]
        task_busy[rid] += busy[entry]
        owner_busy[rid] += busy[entry]
        byte_totals[rid] += moved[entry]
        if not seen[rid]:
            seen[rid] = True
            order[count] = rid
            count += 1
            last_first[rid], last_end[rid] = first[entry], last[entry]
        elif last[entry] > last_end[rid] or (last[entry] == last_end[rid] and first[entry] > last_first[rid]):
            last_first[rid], last_end[rid] = first[entry], last[entry]
    return order[:count], byte_totals, last_first, last_end


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
def _admission_push(times, operations, sequence, size, end, operation, serial):
    at = size
    while at:
        parent = (at - 1) // 2
        if times[parent] <= end:
            break
        times[at], operations[at], sequence[at] = times[parent], operations[parent], sequence[parent]
        at = parent
    times[at], operations[at], sequence[at] = end, operation, serial
    return size + 1


@njit(cache=True, nogil=True, fastmath=False)
def _admission_pop(times, operations, sequence, size):
    operation = operations[0]
    size -= 1
    if size:
        end, last_operation, serial = times[size], operations[size], sequence[size]
        at = 0
        while 2 * at + 1 < size:
            child = 2 * at + 1
            if child + 1 < size and times[child + 1] < times[child]:
                child += 1
            if end <= times[child]:
                break
            times[at], operations[at], sequence[at] = times[child], operations[child], sequence[child]
            at = child
        times[at], operations[at], sequence[at] = end, last_operation, serial
    return size, operation


@njit(cache=True, nogil=True, fastmath=False)
def _admission_append(times, operations, sequence, size, ordered, head, end, operation, serial):
    if ordered:
        tail = head + size - 1
        if tail >= len(times):
            tail -= len(times)
        if not size or end >= times[tail]:
            at = head + size
            if at >= len(times):
                at -= len(times)
            times[at], operations[at], sequence[at] = end, operation, serial
            return size + 1, True, head
        # A faster page may overtake a queued page. Convert the same live
        # entries to a heap once; their separate serials preserve output order.
        old_times, old_ops, old_serials = np.empty(size), np.empty(size, np.int64), np.empty(size, np.int64)
        for index in range(size):
            at = head + index
            if at >= len(times):
                at -= len(times)
            old_times[index], old_ops[index], old_serials[index] = times[at], operations[at], sequence[at]
        for index in range(size):
            _admission_push(times, operations, sequence, index,
                            old_times[index], old_ops[index], old_serials[index])
    return _admission_push(times, operations, sequence, size, end, operation, serial), False, 0


@njit(cache=True, nogil=True, fastmath=False)
def _run(addresses, arrivals, read, geometry, timing, array_ids, channel_ids, host_ids,
         array_ready, buffer_ready, cache_tags, cache_ready, cache_readers, cache_use,
         cache_present, cache_clock, admission, admission_ops, admission_count, acceptance,
         starts, lane_counts, lanes, lane_directions, resource_state, moved, directions,
         link_offsets, link_ids, link_durations, link_bytes, link_directions,
         nand_busy, nand_bytes, link_busy, link_moved):
    (base_units, planes, independent, per_channel, host_channels, max_outstanding,
     max_writes, pages_per_block) = geometry
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
    # Snapshot inherited array/buffer/lane frontiers before this page batch;
    # pages admitted by this invocation must not inflate media wait.
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
    max_read_wait = 0.0
    last_read_wait = 0.0
    link_arrivals = np.full(len(link_offsets) - 1, np.inf, dtype=np.float64)
    link_completions = np.zeros(len(link_offsets) - 1, dtype=np.float64)
    link_last = np.zeros((len(link_ids), 2), dtype=np.float64)
    # Admission needs the earliest completion and directional populations,
    # not a full request-list scan for every page. Preserve submission order
    # separately so the published live queue remains exactly the same.
    # Each UCIe host channel owns independent credits and an issue clock.
    # A blocked channel must not advance another channel's admission time.
    width = admission.shape[1]
    sequence = np.empty(admission.shape, dtype=np.int64)
    initial_count = admission_count.copy()
    ordered = np.ones(host_channels, dtype=np.bool_)
    heads = np.zeros(host_channels, dtype=np.int64)
    read_depths = np.zeros(host_channels, dtype=np.int64)
    write_depths = np.zeros(host_channels, dtype=np.int64)
    for host_channel in range(host_channels):
        size = admission_count[host_channel]
        for old in range(size):
            sequence[host_channel, old] = old
            if old and admission[host_channel, old] < admission[host_channel, old - 1]:
                ordered[host_channel] = False
            read_depths[host_channel] += admission_ops[host_channel, old] == 1
            write_depths[host_channel] += admission_ops[host_channel, old] == 2
        if not ordered[host_channel]:
            old_times = admission[host_channel, :size].copy()
            old_ops = admission_ops[host_channel, :size].copy()
            for old in range(size):
                _admission_push(admission[host_channel], admission_ops[host_channel],
                                sequence[host_channel], old, old_times[old], old_ops[old], old)
    shared_arrival = len(arrivals) == 1
    for index in range(count):
        # Only the final page's detail recorder survives a sequence of scalar
        # submissions. Earlier masks are never read by the page state machine.
        if index == count - 1:
            touched[:] = False
        arrival = arrivals[0] if shared_arrival else arrivals[index]
        if not read:
            last_read_wait = 0.0
            arrival = _transfer_links(arrival, link_offsets, link_ids, link_durations, link_bytes, link_directions,
                starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched,
                link_busy, link_moved, link_arrivals, link_completions, link_last)
            if index == count - 1:
                touched[:] = False  # NAND submission follows the incoming links.
        counters[7] = min(counters[7], arrival)
        page_index = addresses[index] // 4096
        base_unit = page_index % base_units
        plane = (page_index // base_units) % planes
        bank = base_unit + plane * base_units
        physical_page = page_index // (base_units * planes)
        unit = bank if independent else base_unit
        channel = base_unit // per_channel
        host_channel = channel % host_channels
        times, operations, serials = admission[host_channel], admission_ops[host_channel], sequence[host_channel]
        size, head = admission_count[host_channel], heads[host_channel]
        in_order = ordered[host_channel]
        read_depth, write_depth = read_depths[host_channel], write_depths[host_channel]
        effective = max(arrival, acceptance[host_channel])
        while True:
            while size and times[head] <= effective:
                if in_order:
                    removed = operations[head]
                    size -= 1
                    head += 1
                    if head == width:
                        head = 0
                else:
                    size, removed = _admission_pop(times, operations, serials, size)
                read_depth -= removed == 1
                write_depth -= removed == 2
            if size < max_outstanding and (read or write_depth < max_writes):
                break
            deadline = times[head] if size >= max_outstanding else np.inf
            if not read and write_depth >= max_writes:
                # The earliest completed read cannot release a write credit.
                write_deadline = np.inf
                for entry in range(size):
                    at = (head + entry) % width if in_order else entry
                    if operations[at] == 2:
                        write_deadline = min(write_deadline, times[at])
                deadline = max(deadline, write_deadline) if size >= max_outstanding else write_deadline
            effective = max(effective, deadline)
        reads, writes = read_depth + int(read), write_depth + int(not read)
        counters[0] += effective - arrival
        counters[1] = max(counters[1], size + 1)
        counters[2] = max(counters[2], reads)
        counters[3] = max(counters[3], writes)
        acceptance[host_channel] = effective
        base = effective + front
        ar, ch, host = array_ids[unit], channel_ids[channel], host_ids[host_channel]
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
            size, in_order, head = _admission_append(times, operations, serials,
                size, in_order, head, end, 2, initial_count[host_channel] + index)
            write_depth += 1
            admission_count[host_channel], ordered[host_channel], heads[host_channel] = size, in_order, head
            read_depths[host_channel], write_depths[host_channel] = read_depth, write_depth
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
        cache_present[bank] = True
        banks_touched[bank] = True
        before_host = resource_state[host, 1]
        if hit:
            end = _reserve(host, max(base, cache_ready[bank, slot]), host_ns, 4096, 1,
                           starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched)
            cache_readers[bank, slot] = max(cache_readers[bank, slot], end)
            counters[5] += 1
        else:
            before_array, before_channel = resource_state[ar, 1], resource_state[ch, 1]
            frontier_wait = initial_frontier[unit] - base
            last_read_wait = max(0.0, frontier_wait)
            if last_read_wait > max_read_wait:
                max_read_wait = last_read_wait
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
        size, in_order, head = _admission_append(times, operations, serials,
            size, in_order, head, end, 1, initial_count[host_channel] + index)
        read_depth += 1
        admission_count[host_channel], ordered[host_channel], heads[host_channel] = size, in_order, head
        read_depths[host_channel], write_depths[host_channel] = read_depth, write_depth
        end = _transfer_links(end, link_offsets, link_ids, link_durations, link_bytes, link_directions,
            starts, lane_counts, lanes, lane_directions, resource_state, moved, directions, touched,
            link_busy, link_moved, link_arrivals, link_completions, link_last)
        completions[index] = end
    for host_channel in range(host_channels):
        size, head = admission_count[host_channel], heads[host_channel]
        times, operations = admission[host_channel], admission_ops[host_channel]
        if ordered[host_channel]:
            kept_times, kept_ops = np.empty(size), np.empty(size, np.int64)
            for index in range(size):
                at = (head + index) % width
                kept_times[index], kept_ops[index] = times[at], operations[at]
            times[:size], operations[:size] = kept_times, kept_ops
        else:
            order = np.argsort(sequence[host_channel, :size])
            kept_times, kept_ops = times[:size].copy(), operations[:size].copy()
            for index in range(size):
                times[index], operations[index] = kept_times[order[index]], kept_ops[order[index]]
    return (completions, counters, cache_clock, admission_count, acceptance,
            touched, arrays_touched, banks_touched, link_arrivals, link_completions, link_last,
            max_read_wait, last_read_wait)
