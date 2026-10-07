"""Exact max-plus cycle evaluation for summary-only NAND page streams.

Repeated pages are skipped only after proving every max branch and every
calendar transition over the skipped cycles.  Binary fixed-point integers
make that proof exact; conversion back to floats changes only rounding, not
the modeled operations, dependencies, geometry, or resource contention.
"""
from __future__ import annotations

from math import gcd, isfinite, lcm

from .memory_mapping import map_nand_address
from .memory_transfer import ResourceTimeline
from .memory_types import Operation


def full_page_period(config):
    units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    if config.planes_independent:
        units *= config.planes_per_lun
    # Host rounding is relative to the absolute address, including when the
    # host quantum does not divide the page size.
    return lcm(units, config.host_granularity_bytes // gcd(config.page_bytes, config.host_granularity_bytes))


def execute_full_pages(core, request, first_page, page_count):
    """Return a scalar result, or None when ordinary page execution is needed."""
    config, timeline = core.config, core.timeline
    period = full_page_period(config)
    # Bound compilation independently of unusual user-defined geometry.
    if type(timeline) is not ResourceTimeline or period > 4096 or page_count < 3 * period:
        return None
    pages = []
    resources = set()
    arrays = set()
    host_total = 0
    for offset in range(period):
        index = first_page + offset
        mapping = map_nand_address(config, index * config.page_bytes)
        array, resource, channel = core._array_id(mapping), core._array_resource_id(mapping), core._channel_id(mapping)
        start, end = index * config.page_bytes, (index + 1) * config.page_bytes
        quantum = config.host_granularity_bytes
        host_bytes = ((end + quantum - 1) // quantum - start // quantum) * quantum
        pages.append((array, resource, channel, host_bytes))
        arrays.add(array)
        resources.update((resource, channel, core._host_id()))
        host_total += host_bytes
    if any(not rid or len(timeline.lane_available.get(rid, [0.0])) != 1 for rid in resources):
        return None

    # Keys keep virtual array ready times separate from actual slot calendars.
    # Resource aliases share the same lane/ready keys throughout the program.
    values = {("completion", ""): request.arrival_ns + config.front_ns}
    for rid in resources:
        values["lane", rid] = timeline.lane_available.get(rid, [timeline.available_ns(rid)])[0]
        values["ready", rid] = timeline.ready_ns.get(rid, 0.0)
        start, end = timeline.last_intervals.get(rid, (0.0, 0.0))
        values["start", rid], values["end", rid] = start, end
    for array in arrays:
        values["array", array] = core._array_ready.get(array, 0.0)
        values["buffer", array] = core._buffer_ready.get(array, 0.0)
        values["ready", array] = timeline.ready_ns.get(array, 0.0)
    read_ns, program_ns = config.page_read_ns, config.page_program_ns
    internal_ns = config.transfer_page_bytes / config.internal_bandwidth_gb_s
    base = request.arrival_ns + config.front_ns
    constants = [base, read_ns, program_ns, internal_ns, *(count / config.host_bandwidth_gb_s for *_, count in pages)]
    if not all(isfinite(value) for value in (*values.values(), *constants)):
        return None
    ratios = [float(value).as_integer_ratio() for value in (*values.values(), *constants)]
    scale = max(denominator for _, denominator in ratios)

    def ticks(value):
        numerator, denominator = float(value).as_integer_ratio()
        return numerator * (scale // denominator)

    state = {key: ticks(value) for key, value in values.items()}
    base_ticks = ticks(base)
    busy = {rid: 0 for rid in resources}
    moved = {rid: 0 for rid in resources}
    directions = {}
    # Compile stable page geometry once.  Durations use the same per-page
    # float division as ResourceTimeline.transfer, before exact accumulation.
    page_program = [(a, r, c, count, ticks(count / config.host_bandwidth_gb_s)) for a, r, c, count in pages]
    for _, resource, channel, count, host_ns in page_program:
        busy[resource] += ticks(read_ns if request.operation is Operation.READ else program_ns)
        busy[channel] += ticks(internal_ns)
        busy[core._host_id()] += host_ns
        moved[channel] += config.transfer_page_bytes
        moved[core._host_id()] += count

    def cycle(current, delta=None):
        affine = {key: (value, 0 if delta is None else delta[key]) for key, value in current.items()}
        bound = None

        def maximum(*terms):
            nonlocal bound
            winner = max(terms, key=lambda term: term[0])
            if delta is not None:
                for other in terms:
                    gap, slope = winner[0] - other[0], winner[1] - other[1]
                    if slope < 0:
                        # A cycle executes at t=0..k-1; ties are safe because
                        # both candidates have the same value at that point.
                        limit = gap // -slope + 1
                        bound = limit if bound is None else min(bound, limit)
            return winner

        def reserve(rid, earliest, duration, direction=None):
            start = maximum(earliest, affine["lane", rid])
            end = (start[0] + duration, start[1])
            affine["lane", rid] = affine["ready", rid] = end
            affine["start", rid], affine["end", rid] = start, end
            if direction:
                directions[rid] = direction
            return end

        for array, resource, channel, count, host_ns in page_program:
            if request.operation is Operation.READ:
                dependency = maximum((base_ticks, 0), affine["array", array], affine["buffer", array])
                read = reserve(resource, dependency, ticks(read_ns))
                internal = reserve(channel, read, ticks(internal_ns), "read")
                host = reserve(core._host_id(), internal, host_ns, "read")
                affine["array", array], affine["buffer", array] = read, internal
                affine["ready", array] = read
                affine["completion", ""] = maximum(affine["completion", ""], host)
            else:
                host = reserve(core._host_id(), (base_ticks, 0), host_ns, "write")
                dependency = maximum(host, affine["buffer", array])
                internal = reserve(channel, dependency, ticks(internal_ns), "write")
                program = reserve(resource, internal, ticks(program_ns))
                affine["array", array] = affine["buffer", array] = program
                affine["ready", array] = program
                affine["completion", ""] = maximum(affine["completion", ""], program)
        return affine, bound

    cycles, tail = divmod(page_count, period)
    executed = skipped = 0
    while executed < cycles:
        result, _ = cycle(state)
        next_state = {key: value[0] for key, value in result.items()}
        delta = {key: next_state[key] - state[key] for key in state}
        jump = 1
        if cycles - executed > 1 and all(value >= 0 for value in delta.values()):
            proven, bound = cycle(state, delta)
            # This is a complete-state algebraic certificate, not a heuristic
            # based on two equal request completion times.
            if all(proven[key] == (next_state[key], delta[key]) for key in state):
                jump = min(cycles - executed, bound if bound is not None else cycles)
                next_state = {key: state[key] + jump * delta[key] for key in state}
                skipped += (jump - 1) * period
        state = next_state
        executed += jump

    for rid in resources:
        timeline.lane_available.setdefault(rid, [0.0])[0] = state["lane", rid] / scale
        timeline.last_intervals[rid] = (state["start", rid] / scale, state["end", rid] / scale)
        timeline.busy_ns[rid] = timeline.busy_ns.get(rid, 0.0) + busy[rid] * cycles / scale
        if moved[rid]:
            timeline.bytes_moved[rid] = timeline.bytes_moved.get(rid, 0) + moved[rid] * cycles
    for key, value in state.items():
        field, rid = key
        if field == "ready":
            timeline.ready_ns[rid] = value / scale
        elif field == "array":
            core._array_ready[rid] = value / scale
        elif field == "buffer":
            core._buffer_ready[rid] = value / scale
    timeline.directions.update(directions)
    timeline._touched.update(resources)
    timeline._intervals_truncated = True
    return {
        "completion_ns": state["completion", ""] / scale,
        "page_count": cycles * period,
        "host_transfer_bytes": host_total * cycles,
        "internal_transfer_bytes": cycles * period * config.transfer_page_bytes,
        "accelerated_pages": skipped,
        "tail_pages": tail,
    }
