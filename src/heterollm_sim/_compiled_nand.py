"""Numba state machine for strict per-page NAND execution.

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
                           directions):
    """The ordinary generic page machine, with ordered IEEE additions.

    Admission remains one host request in NandCore.submit. Unlike the OCP
    kernel below, generic NAND has neither implicit page-0 erase nor PROGRAM
    frontier validation. No page or cache transition is skipped here.
    """
    base_units, planes, independent, per_channel, host_channels = geometry
    page_read, page_program, internal_ns, host_ns, internal_bytes = timing
    touched = np.zeros(len(state), dtype=np.bool_)
    arrays_touched = np.zeros(len(array_ready), dtype=np.bool_)
    banks_touched = np.zeros(len(tags), dtype=np.bool_)
    hits = misses = 0
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
                end = _reserve(host, max(base, cache_ready[bank, slot]), host_ns, 4096, 1,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                readers[bank, slot] = max(readers[bank, slot], end)
                hits += 1
            else:
                read_end = _reserve(ar, max(base, array_ready[unit], buffer_ready[unit]), page_read, -1, 0,
                    starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
                internal_end = _reserve(ch, max(read_end, readers[bank, slot]), internal_ns,
                    int(internal_bytes), 1, starts, lane_counts, lanes, lane_directions,
                    state, moved, directions, touched)
                end = _reserve(host, internal_end, host_ns, 4096, 1,
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
            host_end = _reserve(host, base, host_ns, 4096, 2,
                starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
            internal_end = _reserve(ch, max(host_end, buffer_ready[unit]), internal_ns,
                int(internal_bytes), 2, starts, lane_counts, lanes, lane_directions,
                state, moved, directions, touched)
            end = _reserve(ar, max(internal_end, array_ready[unit]), page_program, -1, 0,
                starts, lane_counts, lanes, lane_directions, state, moved, directions, touched)
            array_ready[unit], buffer_ready[unit] = end, end
            arrays_touched[unit] = True
        completion = max(completion, end)
    return completion, hits, misses, cache_clock, touched, arrays_touched, banks_touched


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
            touched, arrays_touched, banks_touched, link_arrivals, link_completions, link_last)
