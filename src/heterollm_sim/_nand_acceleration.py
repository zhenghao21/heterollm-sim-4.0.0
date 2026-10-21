"""Strict native pages and max-plus cycles for summary-only NAND streams.

Repeated pages are skipped only after proving every max branch and every
calendar transition over the skipped cycles.  Binary fixed-point integers
make that proof exact; conversion back to floats changes only rounding, not
the modeled operations, dependencies, geometry, or resource contention.
The two-buffer generic path instead executes every page in the existing
native reservation machine, preserving ordered IEEE float results as well.
"""
from __future__ import annotations

from math import gcd, isfinite, lcm
from functools import lru_cache

from .memory_mapping import map_nand_address
from .memory_transfer import ResourceTimeline
from .memory_types import Operation


def _generic_resident_writers(core):
    """Keep private controller arrays only while all writers remain native."""
    from .nand_core import _native_frontier_writers
    from .memory_transfer import _TIMELINE_MAPS, _OWNED_TIMELINE_METHODS
    from ._nand_pack import can_own_resource_aliases
    timeline = core.timeline
    return (not core.capture_details and not core._observe_media and not core._nand_state_exposed
            and core.config.protocol_mode == "generic_nand" and core.config.queue_scope == "device"
            and core.config.read_cache_pages_per_bank == 2 and core.config.page_bytes == 4096
            and not core.config.write_buffer_bytes and not core._staged_pages
            and type(timeline) is ResourceTimeline
            and all(type(getattr(timeline, name)) is dict for name in _TIMELINE_MAPS)
            and type(timeline._touched) is set
            and type(timeline.resource_aliases) is dict
            and can_own_resource_aliases(timeline.resource_aliases)
            and type(core.config.metadata) is dict
            and all(type(core.config.metadata.get(name, "")) is str for name in (
                "array_resource_prefix", "parallel_resource_prefix", "channel_resource_prefix", "host_resource_id"))
            and all(getattr(getattr(timeline, name), "__func__", getattr(timeline, name)) is original
                    for name, original in _OWNED_TIMELINE_METHODS)
            and _native_frontier_writers(core))


@lru_cache(maxsize=128)
def _buffered_layout_contract(array_names, array_resources, channels, hosts, names, aliases):
    """Only immutable geometry, aliases and bus membership; no live clocks.

    Native qualification before this call excludes authored getters and
    conversion hooks. All mutable alias entries participate in the key, so
    a later route edit cannot keep a cached proof of an older layout.
    """
    alias_timeline = ResourceTimeline(resource_aliases=dict(aliases))
    arrays, array_buses, channel_set, host_set = set(array_names), set(array_resources), set(channels), set(hosts)
    buses = channel_set | host_set
    valid = not ((arrays | array_buses) & buses or channel_set & host_set
        or any(name != alias_timeline.canonical_resource_id(name) for name in array_names)
        or (arrays - array_buses) & set(names))
    return valid, tuple(int(name in buses) for name in names)


def execute_buffered_full_pages(core, request, page_count):
    """Strict native generic-NAND pages with the existing two-slot buffers.

    This is a compiled form of the scalar page loop, not the affine shortcut
    below. It retains every float addition, cache access and reservation. The
    normal submit path still owns request-level admission and transactions.
    Unaligned boundaries, authored callbacks and unusual aliases retain the
    ordinary scalar machine rather than changing their semantics.
    """
    import numpy as np
    from .nand_core import NandCore, _ReadCacheSlot
    from .memory_types import TransactionResult
    from ._nand_execution import _geometry_plan, _NandNumericState
    from ._nand_pack import (pack_single_lane, pack_multi_lane, publish_timeline,
                            validate_buffered_cache, pack_buffered_cache, publish_buffered_cache,
                            publish_active_timeline, generic_request_resource_indexes,
                            pack_selected_single_lane)
    from ._compiled_nand import run_generic_full_pages

    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or core.capture_details or core._observe_media or core._staged_pages
            or config.protocol_mode != "generic_nand" or config.queue_scope != "device"
            or config.page_bytes != 4096 or config.read_cache_pages_per_bank != 2
            or 4096 % config.host_granularity_bytes
            or request.operation not in (Operation.READ, Operation.WRITE)
            or request.address % 4096 or request.byte_count % 4096
            or (request.address + request.byte_count - 1) // 4096 > 2**63 - 1
            or request.operation is Operation.WRITE and config.write_buffer_bytes
            or type(core._cache_use) is not int
            or not 0 <= core._cache_use <= 2**63 - 1 - page_count):
        return None
    if any(name in core.__dict__ for name in (
            "_array_id", "_array_resource_id", "_channel_id", "_host_id", "_host_bandwidth",
            "_cache_slot", "_invalidate_cache", "_media_stage", "_staged_key", "_bank_key")):
        return None
    if any(type(getattr(timeline, name)) is not dict for name in (
            "ready_ns", "directions", "busy_ns", "bytes_moved", "last_intervals",
            "lane_available", "lane_directions")):
        return None
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base_units * config.planes_per_lun
    units = banks if config.planes_independent else base_units
    # The request-level dispatcher uses this same 128-page boundary. PD
    # imports on wide devices are small relative to total bank count, but
    # retain their controller arrays and amortize initialization over many
    # chunks. Falling back by total banks routed every chunk through Python.
    if (banks > 65536 or config.effective_capacity_bytes < banks * 4096
            or page_count < 128):
        return None
    per_channel = base_units // config.channels
    (bank_keys, prefixes, array_names, array_resources, channels, hosts, names,
     array_ids, channel_ids, host_ids) = _geometry_plan(
        core, request.operation, banks, units, per_channel)
    # Scalar virtual frontiers may not alias a bus or a different array's
    # execution calendar. The general scalar path handles such authored IDs.
    bus_mask = None
    if has_native_timeline_hooks() and _generic_resident_writers(core):
        valid, bus_mask = _buffered_layout_contract(array_names, array_resources, channels, hosts,
            names, tuple(timeline.resource_aliases.items()))
        if not valid:
            return None
    elif (set(array_resources + array_names) & set(channels + hosts)
            or set(channels) & set(hosts)
            or any(name != timeline.canonical_resource_id(name) for name in array_names)
            or any(name in names for name in set(array_names) - set(array_resources))):
        return None
    directions = [None, "read", "write"]
    packed = None
    if (bus_mask is not None and core._nand_transactions
            and core._nand_transactions[-1].native_timeline_qualified):
        selected = generic_request_resource_indexes(request.address // 4096, page_count,
            base_units, config.planes_per_lun, config.planes_independent, per_channel,
            config.host_channels, array_ids, channel_ids, host_ids)
        packed = pack_selected_single_lane(timeline, names, selected, page_count, config.transfer_page_bytes)
    if packed is None:
        packed = pack_single_lane(timeline, names, page_count, config.transfer_page_bytes)
    single_lane = packed is not None
    if packed is None:
        packed = pack_multi_lane(timeline, names, page_count, config.transfer_page_bytes,
                                 directions, {name: index for index, name in enumerate(directions)})
    if packed is None:
        return None
    starts, lane_counts, lanes, lane_directions, state, moved, direction_values = packed
    # Ordinary core state is numeric and finite. Do not reinterpret unusual
    # caller objects, extension slots, shared slots, or corrupted cache tags.
    cache = core.__dict__["_read_cache"]
    layout = (bank_keys, prefixes, array_names, config.pages_per_block)
    packed = core._nand_numeric_state
    if packed is not None and packed.layout != layout:
        core._nand_publish_numeric()
        packed = None
    if packed is None:
        rows = validate_buffered_cache(cache, bank_keys, prefixes, _ReadCacheSlot, config.pages_per_block, isfinite)
        if rows is None:
            return None
        arrays = np.array([core.__dict__["_array_ready"].get(name, 0.) for name in array_names], dtype=np.float64)
        buffers = np.array([core.__dict__["_buffer_ready"].get(name, 0.) for name in array_names], dtype=np.float64)
        tags, ready, readers, uses, present = pack_buffered_cache(rows, banks, config.pages_per_block)
        if (core._nand_numeric_enabled and not core._nand_state_exposed
                and has_native_timeline_hooks() and _generic_resident_writers(core)):
            packed = _NandNumericState(layout, (arrays, buffers, tags, ready, readers, uses, present))
            packed.scalar_resource_contract = core._resident_resource_contract()
    else:
        arrays, buffers, tags, ready, readers, uses, present = packed.arrays
    if not all(np.all(np.isfinite(values)) and np.all(values >= 0) for values in (arrays, buffers, state)):
        return None
    start_values, lane_counts = np.asarray(starts, dtype=np.int64), np.asarray(lane_counts, dtype=np.int64)
    lanes, lane_directions = np.asarray(lanes, dtype=np.float64), np.asarray(lane_directions, dtype=np.int64)
    timing = np.array((config.page_read_ns, config.page_program_ns,
                       config.transfer_page_bytes / config.internal_bandwidth_gb_s,
                       4096 / core._host_bandwidth(request.operation), config.transfer_page_bytes))
    geometry = np.array((base_units, config.planes_per_lun, config.planes_independent,
                         per_channel, config.host_channels), dtype=np.int64)
    before = packed.checkpoint() if packed is core._nand_numeric_state and packed is not None else None
    if packed is not None:
        for token in core._nand_transactions:
            token.capture_numeric(packed)
    try:
        completion, hits, misses, clock, touched, active_units, active_banks = run_generic_full_pages(
            request.address // 4096, page_count, request.operation is Operation.READ,
            request.arrival_ns + config.front_ns, geometry, timing, array_ids, channel_ids, host_ids,
            arrays, buffers, tags, ready, readers, uses, present, core._cache_use,
            start_values, lane_counts, lanes, lane_directions, state, moved, direction_values)
        if not np.all(np.isfinite(state)) or not isfinite(completion):
            raise ValueError("resource reservation exceeds finite timing range")
    except BaseException:
        if before is not None:
            packed.restore(before)
        raise
    if packed is not None:
        packed.dirty_units |= active_units
        packed.dirty_banks |= active_banks
        core._nand_numeric_state = packed
    resource_indexes = np.flatnonzero(touched)
    resources = {names[index] for index in resource_indexes.tolist()}
    if bus_mask is not None:
        publish_active_timeline(timeline, names, state, moved, direction_values,
            lanes, lane_directions, start_values, lane_counts, resource_indexes,
            resources, bus_mask, directions, single_lane)
    else:
        publish_timeline(timeline, names, state.tolist(), moved.tolist(), direction_values.tolist(),
                         lanes.tolist(), lane_directions.tolist(), start_values.tolist(), lane_counts.tolist(),
                         resources, (), [int(name in resources and name in channels + hosts) for name in names],
                         directions, single_lane)
    unit_indexes = np.flatnonzero(active_units)
    array_values = arrays[unit_indexes].tolist()
    buffer_values = buffers[unit_indexes].tolist() if packed is None else None
    for index, unit in enumerate(unit_indexes.tolist()):
        name = array_names[unit]
        if packed is None:
            core._touch_frontiers(name)
            core.__dict__["_array_ready"][name], core.__dict__["_buffer_ready"][name] = array_values[index], buffer_values[index]
        elif name not in core.__dict__["_array_ready"]:
            # Keep the original first-touch dictionary order and slot object
            # identities while the actual values remain private in arrays.
            core._touch_frontiers(name)
            core.__dict__["_array_ready"][name] = 0.
            core.__dict__["_buffer_ready"].setdefault(name, 0.)
        timeline._remember_resource(name)
        timeline.ready_ns[name] = array_values[index]
    if packed is None:
        publish_buffered_cache(core, cache, bank_keys, prefixes, config.pages_per_block,
                               tags, ready, readers, uses, active_banks, _ReadCacheSlot)
    else:
        for bank in np.flatnonzero(active_banks).tolist():
            key = bank_keys[bank]
            if key not in cache:
                if core._nand_transactions:
                    core._touch_cache_bank(key)
                cache[key] = [_ReadCacheSlot(), _ReadCacheSlot()]
    core._cache_use = int(clock)
    timeline._touched.update(resources)
    timeline._intervals_truncated = bool(resources) or timeline._intervals_truncated
    read = request.operation is Operation.READ
    programmed = 0 if read else page_count
    internal_bytes = (int(misses) + programmed) * config.transfer_page_bytes
    return TransactionResult(request.request_id, request.operation, request.arrival_ns, float(completion),
        request.byte_count, internal_bytes, counters={
            "host_transfer_bytes": request.byte_count, "internal_transfer_bytes": internal_bytes,
            "physical_read_bytes": int(misses) * 4096, "physical_write_bytes": programmed * 4096,
            "pages_read": int(misses), "pages_programmed": programmed,
            "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": 0,
            "staging_read_hits": 0, "staging_read_merges": 0,
            "read_cache_hits": int(hits), "read_cache_misses": int(misses),
            "erase_operations": 0, "page_count": page_count,
            "details_truncated": page_count > config.max_expanded_segments,
            "native_fullpage_pages": page_count,
            "bandwidth_ceiling_gb_s": config.host_bandwidth_for(request.operation) if hits
                else core._bandwidth_ceiling_gb_s(request.operation),
            "bandwidth_ceiling_scope":
                "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain" if hits else
                "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
            "array_units": config.array_units, "write_completion": "n/a" if read else "media_program_complete",
            "host_ack_completion_ns": None, "media_drain_completion_ns": None if read else float(completion),
        })


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
    # This affine program has no page-tag, cache-reader, or OCP page-order
    # state. Those configurations execute the same detailed NAND state
    # machine; never skip their protocol/cache transitions.
    if (config.protocol_mode != "generic_nand" or config.host_channels != 1
            or config.host_full_duplex or config.read_cache_pages_per_bank):
        return None
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
    virtual_arrays = {array for array, resource, _channel, _count in pages if array != resource}
    if any(rid not in timeline.lane_available for rid in virtual_arrays & resources):
        # A virtual array clock can alias a real wire. ResourceTimeline creates
        # that wire's first lane lazily from its ready clock, which an earlier
        # page in this request may change. The affine program assumes existing
        # lanes, so use exact ordinary transitions until those lanes exist.
        # Default array/channel/host IDs are disjoint and retain acceleration.
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
    host_bandwidth = config.host_bandwidth_for(request.operation)
    constants = [base, read_ns, program_ns, internal_ns, *(count / host_bandwidth for *_, count in pages)]
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
    page_program = [(a, r, c, count, ticks(count / host_bandwidth)) for a, r, c, count in pages]
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
                program = reserve(resource, maximum(internal, affine["array", array]), ticks(program_ns))
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

    timeline._remember_resources(resources)
    timeline._remember_resources(rid for field, rid in state if field == "ready")
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


@lru_cache(maxsize=128)
def _owned_resource_footprint(read_names, write_names, array_names):
    return tuple(dict.fromkeys((*read_names, *write_names, *array_names)))


def owned_resource_footprint(core):
    """All possible native calendars for this owner, including virtual arrays."""
    from ._nand_execution import _geometry_plan
    config = core.config
    base = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base * config.planes_per_lun
    if banks > 65536:
        return None
    units = banks if config.planes_independent else base
    per_channel = base // config.channels
    read = _geometry_plan(core, Operation.READ, banks, units, per_channel)
    write = _geometry_plan(core, Operation.WRITE, banks, units, per_channel)
    return _owned_resource_footprint(read[6], write[6], read[2])


from . import _nand_pack as _owned_pack_module
from . import _compiled_nand as _owned_numeric_module
from . import _nand_execution as _owned_execution_module

_OWNED_NAND_FUNCTIONS = tuple((name, globals()[name]) for name in (
    "execute_buffered_full_pages", "execute_full_pages", "full_page_period", "_generic_resident_writers",
    "_buffered_layout_contract"))
_OWNED_NAND_PACK_FUNCTIONS = tuple((name, getattr(_owned_pack_module, name)) for name in (
    "pack_single_lane", "pack_multi_lane", "publish_timeline", "validate_buffered_cache",
    "pack_buffered_cache", "publish_buffered_cache", "can_own_resource_aliases", "publish_active_timeline",
    "generic_request_resource_indexes", "pack_selected_single_lane"))
_OWNED_NAND_NUMERIC_ENTRY = _owned_numeric_module.run_generic_full_pages
_OWNED_NAND_GEOMETRY_PLAN = _owned_execution_module._geometry_plan


def has_native_timeline_hooks():
    for name, original in _OWNED_NAND_FUNCTIONS:
        if globals()[name] is not original:
            return False
    for name, original in _OWNED_NAND_PACK_FUNCTIONS:
        if getattr(_owned_pack_module, name) is not original:
            return False
    return (_owned_numeric_module.run_generic_full_pages is _OWNED_NAND_NUMERIC_ENTRY
            and _owned_execution_module._geometry_plan is _OWNED_NAND_GEOMETRY_PLAN)
