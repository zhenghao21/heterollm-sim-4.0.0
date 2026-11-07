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
from operator import attrgetter

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
            # Strict OCP uses the same resident controller for per-channel
            # admission.  The page kernel already routes every page to the
            # mapped host link; request admission is kept in NandCore so the
            # 256/64 limits remain observable without a Python page loop.
            and core.config.queue_scope in {"device", "host_channel"}
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


def execute_staged_flush_batch(core, pages, ready_times, start_times):
    """Run one logic-die scope flush through the resident numeric controller.

    The caller has already selected full pages belonging to one layer/rank.
    This bridge only packs the shared fabric once, reserves all host, channel
    and array work in ``run_generic_flush_batch``, then publishes the touched
    calendars back.  It returns ``None`` before mutation when an authored or
    unusual timeline cannot be represented by the native arrays; the scalar
    flush remains the exact fallback for those cases.
    """
    import numpy as np
    from .nand_core import NandCore
    from ._nand_execution import _geometry_plan
    from ._nand_pack import pack_single_lane, pack_multi_lane, publish_active_timeline
    from ._compiled_nand import run_generic_flush_batch

    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or config.page_staging != "logic_die"
            or core.capture_details or core._observe_media
            or core._nand_numeric_state is None
            or not core._nand_numeric_enabled
            or not has_native_timeline_hooks()):
        return None
    pages = np.asarray(pages, dtype=np.int64)
    ready_times = np.asarray(ready_times, dtype=np.float64)
    start_times = np.asarray(start_times, dtype=np.float64)
    if (pages.ndim != 1 or ready_times.ndim != 1 or start_times.ndim != 1
            or len(pages) == 0 or len(pages) != len(ready_times)
            or len(pages) != len(start_times)):
        return None
    if (np.any(pages < 0) or np.any(~np.isfinite(ready_times))
            or np.any(~np.isfinite(start_times))):
        return None
    phase_mappings, phase_next_pointers = (), tuple(core._write_pointer)
    physical_pages = pages
    if core._sequential_write_mode():
        physical_list, phase_mappings, phase_next_pointers = core._phase_batch_mappings(pages.tolist())
        physical_pages = np.asarray(physical_list, dtype=np.int64)
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base_units * config.planes_per_lun
    units = banks if config.planes_independent else base_units
    per_channel = base_units // config.channels
    (bank_keys, prefixes, array_names, array_resources, channels, hosts, names,
     array_ids, channel_ids, host_write_ids) = _geometry_plan(
        core, Operation.WRITE, banks, units, per_channel)
    read_hosts = _geometry_plan(core, Operation.READ, banks, units, per_channel)[5]
    # A flush touches every host link in both directions, its die channels
    # and arrays.
    links = (*read_hosts, *hosts)
    if config.host_channels * 2 != len(set(links)):
        return None
    names = tuple(dict.fromkeys((*array_resources, *channels, *links)))
    lookup = {name: index for index, name in enumerate(names)}
    array_ids = np.asarray([lookup[array_resources[unit]] for unit in range(units)], dtype=np.int64)
    channel_ids = np.asarray([lookup[name] for name in channels], dtype=np.int64)
    host_read_ids = np.asarray([lookup[name] for name in read_hosts], dtype=np.int64)
    host_write_ids = np.asarray([lookup[name] for name in hosts], dtype=np.int64)
    direction_names = [None, "read", "write"]
    packed_timeline = pack_single_lane(timeline, names, len(pages), config.transfer_page_bytes)
    single_lane = packed_timeline is not None
    if packed_timeline is None:
        packed_timeline = pack_multi_lane(
            timeline, names, len(pages), config.transfer_page_bytes,
            direction_names, {name: index for index, name in enumerate(direction_names)})
    if packed_timeline is None:
        return None
    starts, lane_counts, lanes, lane_directions, state, moved, direction_values = packed_timeline
    starts = np.asarray(starts, dtype=np.int64)
    lane_counts = np.asarray(lane_counts, dtype=np.int64)
    lanes = np.asarray(lanes, dtype=np.float64)
    lane_directions = np.asarray(lane_directions, dtype=np.int64)
    state = np.asarray(state, dtype=np.float64)
    moved = np.asarray(moved, dtype=np.int64)
    direction_values = np.asarray(direction_values, dtype=np.int64)
    packed = core._nand_numeric_state
    layout = (bank_keys, prefixes, array_names, config.pages_per_block)
    if packed.layout != layout:
        return None
    arrays, buffers, tags, _cache_ready, _readers, _uses, _present = packed.arrays
    if not all(np.all(np.isfinite(values)) and np.all(values >= 0)
               for values in (arrays, buffers, state, lanes)):
        return None
    before = packed.checkpoint()
    for token in core._nand_transactions:
        token.capture_numeric(packed)
    geometry = np.array((base_units, config.planes_per_lun,
                         int(config.planes_independent), per_channel,
                         config.host_channels), dtype=np.int64)
    internal_ns = config.transfer_page_bytes / config.internal_bandwidth_gb_s
    trace_enabled = (getattr(core, "_ocp_diag_events", None) is not None
                     and config.protocol_mode == "ocp_hbf_0_7")
    trace_program_start = (np.full(len(pages), -1.0, dtype=np.float64)
                           if trace_enabled else np.empty(0, dtype=np.float64))
    trace_program_end = (np.full(len(pages), -1.0, dtype=np.float64)
                         if trace_enabled else np.empty(0, dtype=np.float64))
    trace_erase_start = (np.full(len(pages), -1.0, dtype=np.float64)
                         if trace_enabled else np.empty(0, dtype=np.float64))
    trace_erase_end = (np.full(len(pages), -1.0, dtype=np.float64)
                       if trace_enabled else np.empty(0, dtype=np.float64))
    try:
        ends, touched, active_units, active_banks, erase_count = run_generic_flush_batch(
            physical_pages, ready_times, start_times, geometry, config.page_program_ns,
            config.block_erase_ns if config.protocol_mode == "ocp_hbf_0_7" else 0.0,
            config.pages_per_block,
            internal_ns, int(config.transfer_page_bytes),
            config.transfer_page_bytes / core._host_bandwidth(Operation.READ),
            config.transfer_page_bytes / core._host_bandwidth(Operation.WRITE),
            array_ids, channel_ids, host_read_ids, host_write_ids,
            arrays, buffers, tags, starts, lane_counts, lanes, lane_directions,
            state, moved, direction_values, trace_program_start, trace_program_end,
            trace_erase_start, trace_erase_end)
    except BaseException:
        packed.restore(before)
        raise
    packed.dirty_units |= active_units
    packed.dirty_banks |= active_banks
    resource_indexes = np.flatnonzero(touched)
    resources = {names[index] for index in resource_indexes.tolist()}
    publish_active_timeline(
        timeline, names, state, moved, direction_values, lanes, lane_directions,
        starts, lane_counts, resource_indexes, resources,
        tuple(int(name in channels or name in links) for name in names),
        direction_names, single_lane)
    timeline._touched.update(resources)
    timeline._intervals_truncated = bool(resources) or timeline._intervals_truncated
    core._nand_numeric_state = packed
    if trace_enabled:
        for index, logical_page in enumerate(pages.tolist()):
            mapping = (phase_mappings[index] if len(phase_mappings) == len(pages)
                       else map_nand_address(config, int(logical_page) * config.page_bytes))
            if trace_erase_start[index] >= 0.0:
                core._ocp_diag_append_event(
                    "erase", mapping, issue_ns=trace_erase_start[index],
                    end_ns=trace_erase_end[index],
                )
            core._ocp_diag_append_event(
                "program", mapping, issue_ns=trace_program_start[index],
                end_ns=trace_program_end[index], logical_page=logical_page,
            )
    return {"pages": pages, "ends": ends, "internal_transfer_bytes": int(len(pages) * config.transfer_page_bytes),
            "host_transfer_bytes": int(2 * len(pages) * config.transfer_page_bytes),
            "erase_operations": int(erase_count),
            "_phase_mappings": phase_mappings,
            "_phase_next_pointers": phase_next_pointers}


def execute_buffered_full_pages(core, request, page_count, *, _metrics=None):
    """Strict native generic-NAND pages with the existing two-slot buffers.

    This is a compiled form of the scalar page loop, not the affine shortcut
    below. It retains every float addition, cache access and reservation. The
    normal submit path still owns request-level admission and transactions.
    A span may start or end inside a page: as in ``split_nand_request`` its
    boundary pages move their host-granularity span over the host and a
    full page internally, and a partially written boundary page pays the
    read-modify-write old-page read first. Partial programs under the
    reject policy, authored callbacks and unusual aliases retain the
    ordinary scalar machine rather than changing their semantics.
    """
    import numpy as np
    from .nand_core import NandCore, _ReadCacheSlot, _bulk_min_pages
    from .memory_types import TransactionResult
    from ._nand_execution import _geometry_plan, _NandNumericState
    from ._nand_pack import (pack_single_lane, pack_multi_lane,
                            validate_buffered_cache, pack_buffered_cache, publish_buffered_cache,
                            publish_active_timeline, generic_request_resource_indexes,
                            pack_selected_single_lane, generic_metrics_delta,
                            publish_generic_controller)
    from ._compiled_nand import run_generic_full_pages

    config, timeline = core.config, core.timeline
    phase_write = (request.operation is Operation.WRITE and core._sequential_write_mode())
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or core.capture_details or core._observe_media or core._staged_pages
            or config.queue_scope not in {"device", "host_channel"}
            or config.page_bytes != 4096 or config.read_cache_pages_per_bank != 2
            or 4096 % config.host_granularity_bytes
            or request.operation not in (Operation.READ, Operation.WRITE)
            or request.operation is Operation.WRITE and (request.address % 4096 or request.byte_count % 4096)
            or (request.address + request.byte_count - 1) // 4096 > 2**63 - 1
            or request.operation is Operation.WRITE and config.write_buffer_bytes
            or type(core._cache_use) is not int
            or not 0 <= core._cache_use <= 2**63 - 1 - page_count):
        return None
    if request.operation is Operation.WRITE and not phase_write:
        # Reject a skipped local DLU before packing or reserving any native
        # calendars; the scalar path exposes the same check.
        core._validate_write_order(range(request.address // 4096,
                                         request.address // 4096 + page_count))
    logical_pages = tuple(range(request.address // 4096,
                                request.address // 4096 + page_count))
    if phase_write:
        physical_list, phase_mappings, phase_next_pointers = core._phase_batch_mappings(logical_pages)
        page_indices = np.asarray(physical_list, dtype=np.int64)
    else:
        phase_mappings, phase_next_pointers = (), tuple(core._write_pointer)
        page_indices = np.arange(request.address // 4096,
                                 request.address // 4096 + page_count, dtype=np.int64)
    if any(name in core.__dict__ for name in (
            "_array_id", "_array_resource_id", "_channel_id", "_host_id", "_host_bandwidth",
            "_cache_slot", "_invalidate_cache", "_media_stage", "_staged_key", "_bank_key")):
        return None
    if any(type(getattr(timeline, name)) is not dict for name in (
            "ready_ns", "directions", "busy_ns", "bytes_moved", "last_intervals",
            "lane_available", "lane_directions")):
        return None
    # Staged pages are served by the staging store, page by page.
    if core._tail_staged(request.address // 4096, (request.address + request.byte_count - 1) // 4096):
        return None
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base_units * config.planes_per_lun
    units = banks if config.planes_independent else base_units
    # The request-level dispatcher uses this same page boundary. PD imports
    # on wide devices are small relative to total bank count, but retain
    # their controller arrays and amortize initialization over many chunks.
    # Falling back by total banks routed every chunk through Python.
    if (banks > 65536 or config.effective_capacity_bytes < banks * 4096
            or page_count < _bulk_min_pages(core, request.operation)):
        return None
    per_channel = base_units // config.channels
    (bank_keys, prefixes, array_names, array_resources, channels, hosts, names,
     array_ids, channel_ids, host_ids) = _geometry_plan(
        core, request.operation, banks, units, per_channel)
    # Scalar virtual frontiers may not alias a bus or a different array's
    # execution calendar. The general scalar path handles such authored IDs.
    bus_mask = None
    # Direct request accounting already proved every writer before admission;
    # no callback occurs between that proof and this native executor.
    resident_writers = _metrics is not None or (has_native_timeline_hooks() and _generic_resident_writers(core))
    if resident_writers:
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
        selected = generic_request_resource_indexes(int(page_indices[0]), page_count,
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
    before_busy = state[:, 1].copy() if _metrics is not None else None
    before_moved = moved.copy() if _metrics is not None else None
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
                and resident_writers):
            packed = _NandNumericState(layout, (arrays, buffers, tags, ready, readers, uses, present))
            packed.scalar_resource_contract = core._resident_resource_contract()
    else:
        arrays, buffers, tags, ready, readers, uses, present = packed.arrays
    if not all(np.all(np.isfinite(values)) and np.all(values >= 0) for values in (arrays, buffers, state)):
        return None
    start_values, lane_counts = np.asarray(starts, dtype=np.int64), np.asarray(lane_counts, dtype=np.int64)
    lanes, lane_directions = np.asarray(lanes, dtype=np.float64), np.asarray(lane_directions, dtype=np.int64)
    host_bandwidth = core._host_bandwidth(request.operation)
    # split_nand_request's host span of the first and last page.
    granularity, end = config.host_granularity_bytes, request.address + request.byte_count
    head_end = min(end, (request.address // 4096 + 1) * 4096)
    head_bytes = -(-head_end // granularity) * granularity - request.address // granularity * granularity
    tail_start = max(request.address, (end - 1) // 4096 * 4096)
    tail_bytes = -(-end // granularity) * granularity - tail_start // granularity * granularity
    host_bytes = head_bytes if page_count == 1 else head_bytes + tail_bytes + (page_count - 2) * 4096
    geometry = np.array((base_units, config.planes_per_lun, config.planes_independent,
                         per_channel, config.host_channels), dtype=np.int64)
    before = packed.checkpoint() if packed is core._nand_numeric_state and packed is not None else None
    if packed is not None:
        for token in core._nand_transactions:
            token.capture_numeric(packed)
    timing = np.array((config.page_read_ns, config.page_program_ns,
                       config.block_erase_ns if config.protocol_mode == "ocp_hbf_0_7" else 0.0,
                       config.transfer_page_bytes / config.internal_bandwidth_gb_s,
                       4096 / host_bandwidth, config.transfer_page_bytes,
                       config.pages_per_block))
    trace_enabled = (getattr(core, "_ocp_diag_events", None) is not None
                     and config.protocol_mode == "ocp_hbf_0_7")
    trace_program_start = (np.full(page_count, -1.0, dtype=np.float64)
                           if trace_enabled else np.empty(0, dtype=np.float64))
    trace_program_end = (np.full(page_count, -1.0, dtype=np.float64)
                         if trace_enabled else np.empty(0, dtype=np.float64))
    trace_erase_start = (np.full(page_count, -1.0, dtype=np.float64)
                         if trace_enabled else np.empty(0, dtype=np.float64))
    trace_erase_end = (np.full(page_count, -1.0, dtype=np.float64)
                       if trace_enabled else np.empty(0, dtype=np.float64))
    try:
        completion, hits, misses, clock, touched, active_units, active_banks, resource_order, max_read_wait, erase_count = run_generic_full_pages(
            request.address // 4096, page_count, request.operation is Operation.READ,
            page_indices,
            request.arrival_ns + config.front_ns, geometry, timing, array_ids, channel_ids, host_ids,
            arrays, buffers, tags, ready, readers, uses, present, core._cache_use,
            start_values, lane_counts, lanes, lane_directions, state, moved, direction_values,
            head_bytes, head_bytes / host_bandwidth, tail_bytes, tail_bytes / host_bandwidth,
            trace_program_start, trace_program_end, trace_erase_start, trace_erase_end)
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
    if page_count < 128:
        # This span previously used scalar reserve(). Retain its first-touch
        # dictionary/set order as well as its float arithmetic, including a
        # newly introduced directional host or an as-yet untouched bank.
        resource_indexes = resource_order
    resources = {names[index] for index in resource_indexes.tolist()}
    if bus_mask is None:
        bus_mask = tuple(int(name in channels or name in hosts) for name in names)
    publish_active_timeline(timeline, names, state, moved, direction_values,
        lanes, lane_directions, start_values, lane_counts, resource_indexes,
        resources, bus_mask, directions, single_lane, _metrics is not None)
    unit_indexes = np.flatnonzero(active_units)
    publish_generic_controller(core, array_names, unit_indexes, arrays, buffers,
                               packed is not None, resources, _metrics is not None)
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
    if page_count < 128:
        for index in resource_indexes.tolist():
            timeline._touched.add(names[index])
    else:
        timeline._touched.update(resources)
    timeline._intervals_truncated = bool(resources) or timeline._intervals_truncated
    if _metrics is not None:
        _metrics.update(generic_metrics_delta(timeline, names, resource_indexes, before_busy, before_moved))
    read = request.operation is Operation.READ
    programmed = 0 if read else page_count
    if trace_enabled and programmed:
        for offset, logical_page in enumerate(logical_pages):
            mapping = (phase_mappings[offset] if len(phase_mappings) == programmed
                       else map_nand_address(config, int(logical_page) * config.page_bytes))
            if trace_erase_start[offset] >= 0.0:
                core._ocp_diag_append_event(
                    "erase", mapping, issue_ns=trace_erase_start[offset],
                    end_ns=trace_erase_end[offset],
                )
            core._ocp_diag_append_event(
                "program", mapping, issue_ns=trace_program_start[offset],
                end_ns=trace_program_end[offset], logical_page=logical_page,
            )
    if programmed:
        if phase_write:
            core._commit_phase_batch(logical_pages, phase_mappings, phase_next_pointers)
        else:
            core._log_programs(logical_pages)
    pages_read = int(misses)
    internal_bytes = (pages_read + programmed) * config.transfer_page_bytes
    core._last_media_queue_wait_ns = float(max_read_wait)
    return TransactionResult(request.request_id, request.operation, request.arrival_ns, float(completion),
        request.byte_count, internal_bytes, counters={
            "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
            "physical_read_bytes": pages_read * 4096, "physical_write_bytes": programmed * 4096,
            "pages_read": pages_read, "pages_programmed": programmed,
            "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": 0,
            "staging_read_hits": 0, "staging_read_merges": 0,
            "read_cache_hits": int(hits), "read_cache_misses": int(misses),
            "erase_operations": int(erase_count), "page_count": page_count,
            "details_truncated": page_count > config.max_expanded_segments,
            "native_fullpage_pages": page_count,
            "bandwidth_ceiling_gb_s": config.host_bandwidth_for(request.operation) if hits
                else core._bandwidth_ceiling_gb_s(request.operation),
            "bandwidth_ceiling_scope":
                "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain" if hits else
                "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
            "array_units": config.array_units, "write_completion": "n/a" if read else "media_program_complete",
            "host_ack_completion_ns": None, "media_drain_completion_ns": None if read else float(completion),
            "media_queue_wait_ns": float(max_read_wait),
        })


def execute_read_batch(core, requests, *, _resource_sink=None, _host_channel=None):
    """Schedule one submit_many READ batch on the out-of-order controller.

    The batch is one controller window (run_generic_read_batch): requests
    still take host credits in order and every page keeps its buffer
    decision, but channels and the host link serve pages as they become
    ready instead of in request order. Each request's host admission is
    then recorded exactly as a sequential submit would, from the scheduled
    completions. Returns results in input order, or None before any
    mutation when the core is not a native generic controller.
    """
    import numpy as np
    from .nand_core import NandCore, _ReadCacheSlot
    from .memory_types import AccessRequest, TransactionResult
    from .memory_transfer import _OWNED_TIMELINE_TOKEN_TYPE, _OWNED_TIMELINE_REMEMBER
    from ._nand_execution import _geometry_plan, _NandNumericState
    from ._nand_pack import (pack_single_lane, pack_multi_lane, validate_buffered_cache,
                            pack_buffered_cache, publish_active_timeline, publish_generic_controller)
    from ._compiled_nand import run_generic_read_batch, read_batch_request_metrics, read_batch_sink_totals

    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or not core._nand_numeric_enabled or core._nand_state_exposed
            or 4096 % config.host_granularity_bytes
            or core._write_buffer_pending or core._write_buffer_occupancy
            or type(core._cache_use) is not int
            or not has_native_timeline_hooks() or not _generic_resident_writers(core)):
        return None
    if any(name in core.__dict__ for name in (
            "_array_id", "_array_resource_id", "_channel_id", "_host_id", "_host_bandwidth",
            "_cache_slot", "_invalidate_cache", "_media_stage", "_staged_key", "_bank_key",
            "_admit_host_request", "_record_host_admission", "_validate_request")):
        return None
    # Undo journals are written by the ordinary native publishers below.
    if any(type(token) is not _OWNED_TIMELINE_TOKEN_TYPE or token.accounting
           or getattr(token.remember, "__func__", token.remember) is not _OWNED_TIMELINE_REMEMBER
           for token in timeline._owned_transactions):
        return None
    if any(type(request) is not AccessRequest or request.operation is not Operation.READ
           or request.byte_count <= 0 for request in requests):
        return None
    counts = [(request.address + request.byte_count - 1) // 4096 - request.address // 4096 + 1
              for request in requests]
    pages = sum(counts)
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base_units * config.planes_per_lun
    units = banks if config.planes_independent else base_units
    if (banks > 65536 or config.effective_capacity_bytes < banks * 4096
            or not 0 <= core._cache_use <= 2**63 - 1 - pages):
        return None
    for request in requests:
        core._validate_request(request)
    per_channel = base_units // config.channels
    (bank_keys, prefixes, array_names, array_resources, channels, hosts, names,
     array_ids, channel_ids, host_ids) = _geometry_plan(core, Operation.READ, banks, units, per_channel)
    valid, bus_mask = _buffered_layout_contract(array_names, array_resources, channels, hosts,
        names, tuple(timeline.resource_aliases.items()))
    if not valid:
        return None
    directions = [None, "read", "write"]
    calendar = pack_single_lane(timeline, names, pages, config.transfer_page_bytes)
    single_lane = calendar is not None
    if calendar is None:
        calendar = pack_multi_lane(timeline, names, pages, config.transfer_page_bytes,
                                   directions, {name: index for index, name in enumerate(directions)})
    if calendar is None:
        return None
    starts, lane_counts, lanes, lane_directions, state, moved, direction_values = calendar
    cache = core.__dict__["_read_cache"]
    layout = (bank_keys, prefixes, array_names, config.pages_per_block)
    packed = core._nand_numeric_state
    if packed is not None and packed.layout != layout:
        core._nand_publish_numeric()
        packed = None
    resident = packed is not None
    if not resident:
        rows = validate_buffered_cache(cache, bank_keys, prefixes, _ReadCacheSlot, config.pages_per_block, isfinite)
        if rows is None:
            return None
        packed = _NandNumericState(layout, (
            np.array([core.__dict__["_array_ready"].get(name, 0.) for name in array_names], dtype=np.float64),
            np.array([core.__dict__["_buffer_ready"].get(name, 0.) for name in array_names], dtype=np.float64),
            *pack_buffered_cache(rows, banks, config.pages_per_block)))
        packed.scalar_resource_contract = core._resident_resource_contract()
    arrays, buffers, tags, ready, readers, uses, present = packed.arrays
    if not all(np.all(np.isfinite(values)) and np.all(values >= 0) for values in (arrays, buffers, state)):
        return None
    # Preserve only frontiers inherited from work before this batch.  The
    # numeric kernel mutates ``arrays``/``buffers`` as pages are dispatched;
    # using this snapshot keeps same-batch queueing out of the wait metric.
    initial_frontier = np.maximum(arrays.copy(), buffers.copy())
    start_values, lane_counts = np.asarray(starts, dtype=np.int64), np.asarray(lane_counts, dtype=np.int64)
    lanes, lane_directions = np.asarray(lanes, dtype=np.float64), np.asarray(lane_directions, dtype=np.int64)
    for unit, resource in enumerate(array_ids):
        first = int(start_values[resource])
        stop = first + int(lane_counts[resource])
        lane_frontier = float(np.min(lanes[first:stop]))
        if lane_frontier > initial_frontier[unit]:
            initial_frontier[unit] = lane_frontier
    host_bandwidth = core._host_bandwidth(Operation.READ)
    granularity = config.host_granularity_bytes
    # Pages a staging store holds at each request's arrival. A staged
    # boundary page moves only its burst-rounded span; an external store
    # bills its own reads, so its pages cross no NAND resource.
    staging_host = config.page_staging == "logic_die"
    staged_ready = np.full(pages, -1., dtype=np.float64)
    staged_counts = [0] * len(requests)
    staged_host_bytes = [0] * len(requests)
    head_bytes, tail_bytes = [], []
    offset = 0
    for index, request in enumerate(requests):
        end = request.address + request.byte_count
        first, last = request.address // 4096, (end - 1) // 4096
        staged = core._tail_readable(first, last, request.arrival_ns) if core._tail_pages else ()
        head_end = min(end, (first + 1) * 4096)
        tail_start = max(request.address, last * 4096)
        unit_head = config.page_staging_burst_bytes if first in staged else granularity
        unit_tail = config.page_staging_burst_bytes if last in staged else granularity
        head_bytes.append(-(-head_end // unit_head) * unit_head - request.address // unit_head * unit_head)
        tail_bytes.append(-(-end // unit_tail) * unit_tail - tail_start // unit_tail * unit_tail)
        for page in staged:
            entry = core._tail_pages[page]
            base = page * 4096
            lo, hi = max(request.address, base) - base, min(end, base + 4096) - base
            if not any(left <= lo and hi <= right for left, right in entry[0]):
                # The scalar page machine can merge a partially staged page
                # with media bytes; the native batch kernel only handles a
                # complete staged span.
                return None
            core._tail_check_read(page, request.address, end)
            staged_ready[offset + page - first] = core._tail_pages[page][2]
            if not staging_host:
                staged_host_bytes[index] += (head_bytes[-1] if page == first else
                                             tail_bytes[-1] if page == last else 4096)
        staged_counts[index] = len(staged)
        offset += counts[index]
    head_bytes = np.array(head_bytes, dtype=np.int64)
    tail_bytes = np.array(tail_bytes, dtype=np.int64)
    page_counts = np.array(counts, dtype=np.int64)
    if config.queue_scope == "host_channel":
        if _host_channel is None:
            return None
        request_hosts = {
            map_nand_address(config, request.address).channel % config.host_channels
            for request in requests
        }
        if request_hosts != {_host_channel}:
            return None
        population = core.__dict__["_host_inflight"].get(_host_channel, ())
        acceptance = max(core._host_acceptance_ns.get(_host_channel, 0.),
                         core._acceptance_ns)
    else:
        population = core.__dict__["_inflight_operations"]
        acceptance = core._acceptance_ns
    prior_ends = np.sort(np.array([end for end, _operation in population], dtype=np.float64))
    geometry = np.array((base_units, config.planes_per_lun, config.planes_independent,
                         per_channel, config.host_channels), dtype=np.int64)
    timing = np.array((config.page_read_ns, config.transfer_page_bytes / config.internal_bandwidth_gb_s,
                       4096 / host_bandwidth, config.transfer_page_bytes))
    before = packed.checkpoint() if resident else None
    if resident:
        for token in core._nand_transactions:
            token.capture_numeric(packed)
    try:
        (admissions, completions, hits, misses, clock, touched, active_units, active_banks,
         offsets, page_hit, page_unit, page_channel, read_start, channel_start, host_start,
         read_eligible) = run_generic_read_batch(
            np.array([request.address // 4096 for request in requests], dtype=np.int64), page_counts,
            np.array([request.arrival_ns for request in requests], dtype=np.float64),
            head_bytes, head_bytes / host_bandwidth, tail_bytes, tail_bytes / host_bandwidth,
            staged_ready, config.page_staging_latency_ns, staging_host,
            acceptance, prior_ends, config.max_outstanding_requests, config.front_ns,
            geometry, timing, array_ids, channel_ids, host_ids,
            arrays, buffers, tags, ready, readers, uses, present, core._cache_use,
            start_values, lane_counts, lanes, lane_directions, state, moved, direction_values)
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(completions)):
            raise ValueError("resource reservation exceeds finite timing range")
    except BaseException:
        if before is not None:
            packed.restore(before)
        raise
    packed.dirty_units |= active_units
    packed.dirty_banks |= active_banks
    core._nand_numeric_state = packed
    core._observation_events = []
    timeline._reset_metric_recorder(0)
    resource_indexes = np.flatnonzero(touched)
    resources = {names[index] for index in resource_indexes.tolist()}
    publish_active_timeline(timeline, names, state, moved, direction_values,
        lanes, lane_directions, start_values, lane_counts, resource_indexes,
        resources, bus_mask, directions, single_lane, True)
    publish_generic_controller(core, array_names, np.flatnonzero(active_units), arrays, buffers,
                               True, resources, True)
    for bank in np.flatnonzero(active_banks).tolist():
        key = bank_keys[bank]
        if key not in cache:
            if core._nand_transactions:
                core._touch_cache_bank(key)
            cache[key] = [_ReadCacheSlot(), _ReadCacheSlot()]
    core._cache_use = int(clock)
    timeline._touched.update(resources)
    timeline._intervals_truncated = bool(resources) or timeline._intervals_truncated
    truncated = timeline._intervals_truncated

    request_offsets, ids, busy, byte_counts, first, last, read_wait = read_batch_request_metrics(
        offsets, page_counts, page_hit, page_unit, page_channel, read_start, read_eligible, channel_start,
        host_start, head_bytes, head_bytes / host_bandwidth, tail_bytes, tail_bytes / host_bandwidth,
        timing, config.host_channels, array_ids, channel_ids, host_ids, initial_frontier, len(names))
    if _resource_sink is not None:
        # Fold the per-request bills into the task and owner bills with the
        # same request-ordered additions as consuming each request's maps,
        # without building those maps. A latest interval equal to the
        # published one is that published tuple.
        task_busy, task_bytes, task_last, owner_busy, owner_bytes, owner_last = _resource_sink
        task_values = np.zeros(len(names), dtype=np.float64)
        owner_values = np.zeros(len(names), dtype=np.float64)
        for index in resource_indexes.tolist():
            task_values[index] = task_busy.get(names[index], 0.0)
            owner_values[index] = owner_busy.get(names[index], 0.0)
        order, byte_totals, last_first, last_end = read_batch_sink_totals(
            ids, busy, byte_counts, first, last, task_values, owner_values, len(names))
        published = timeline.last_intervals
        for index in order.tolist():
            rid = names[index]
            task_busy[rid] = float(task_values[index])
            owner_busy[rid] = float(owner_values[index])
            payload = int(byte_totals[index])
            if payload:
                task_bytes[rid] = task_bytes.get(rid, 0) + payload
                owner_bytes[rid] = owner_bytes.get(rid, 0) + payload
            interval = published[rid]
            if interval != (last_first[index], last_end[index]):
                interval = (float(last_first[index]), float(last_end[index]))
            for bill in (task_last, owner_last):
                previous = bill.get(rid)
                if previous is None or (interval[1], interval[0]) > (previous[1], previous[0]):
                    bill[rid] = interval
        request_metrics = None
    else:
        request_metrics = (request_offsets.tolist(), [names[rid] for rid in ids.tolist()],
                           busy.tolist(), byte_counts.tolist(), first.tolist(), last.tolist())

    hit_ceiling = config.host_bandwidth_for(Operation.READ)
    core._last_media_queue_wait_ns = float(np.max(read_wait)) if len(read_wait) else 0.0
    miss_ceiling = core._bandwidth_ceiling_gb_s(Operation.READ)
    depth_scope = core._admission_depth_scope()
    admission_values, completion_values = admissions.tolist(), completions.tolist()
    hit_values, miss_values = hits.tolist(), misses.tolist()
    results = []
    for index, request in enumerate(requests):
        effective, admitted, host_channel = core._admit_host_request(request)
        if effective != admission_values[index]:
            raise RuntimeError("native NAND read batch admission diverged from the host queue")
        completion, hit_count, miss_count, count = (completion_values[index], hit_values[index],
                                                    miss_values[index], counts[index])
        host_bytes = (int(head_bytes[index]) if count == 1
                      else int(head_bytes[index]) + int(tail_bytes[index]) + (count - 2) * 4096)
        host_bytes -= staged_host_bytes[index]
        staged_count = staged_counts[index]
        internal_bytes = miss_count * config.transfer_page_bytes
        if request_metrics is None:
            metrics = {"resource_intervals": {}, "resource_interval_payloads": {},
                       "intervals_truncated": truncated}
        else:
            request_offsets, rid_names, busy, byte_counts, first, last = request_metrics
            span = range(request_offsets[index], request_offsets[index + 1])
            metrics = {
                "resource_busy_ns": {rid_names[at]: busy[at] for at in span},
                "resource_bytes": {rid_names[at]: byte_counts[at] for at in span if byte_counts[at]},
                "resource_last_intervals": {rid_names[at]: (first[at], last[at]) for at in span},
                "resource_intervals": {}, "resource_interval_payloads": {},
                "intervals_truncated": truncated}
        results.append(TransactionResult(request.request_id, Operation.READ, request.arrival_ns,
            completion, request.byte_count, internal_bytes, counters={
                "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": miss_count * 4096, "physical_write_bytes": 0,
                "pages_read": miss_count, "pages_programmed": 0,
                "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": 0,
                "staging_read_hits": staged_count, "staging_read_merges": 0,
                "read_cache_hits": hit_count, "read_cache_misses": miss_count,
                "erase_operations": 0, "page_count": count,
                "details_truncated": count > config.max_expanded_segments,
                "native_fullpage_pages": count,
                "bandwidth_ceiling_gb_s": hit_ceiling if hit_count or staged_count else miss_ceiling,
                "bandwidth_ceiling_scope":
                    "directional host upper bound for transactions using cached/staged data; host acknowledgement is not media drain" if hit_count or staged_count else
                    "device transfer upper bound: min(directional host, summed independent channels); excludes array latency",
                "array_units": config.array_units, "write_completion": "n/a",
                "host_ack_completion_ns": None, "media_drain_completion_ns": None,
                "pages_touched": count, "read_operations": miss_count, "program_operations": 0,
                "rmw_read_operations": 0, "page_access_payload_bytes": request.byte_count,
                "page_access_capacity_bytes": count * config.page_bytes,
                "host_admission_samples": 1,
                "host_outstanding_at_admission_peak": len(admitted) + 1,
                "host_read_outstanding_at_admission_peak": admitted.reads + 1,
                "host_write_outstanding_at_admission_peak": admitted.writes,
                "host_admission_depth_scope": depth_scope,
                "queue_wait_ns": effective - request.arrival_ns,
                "media_queue_wait_ns": float(read_wait[index]),
                "write_buffer_wait_ns": 0.0, "write_buffer_bytes": 0,
                "write_buffer_occupancy_bytes": 0, "write_buffer_peak_bytes": 0,
                "write_buffer_draining_bytes": 0,
                "buffered_write_mode": core._buffered_write_mode,
                **metrics,
            }))
        core._record_host_admission(host_channel, effective, admitted, completion, Operation.READ)
    return tuple(results)


def execute_sequential_read_group(core, requests, resource_sink=None):
    """One submit_many READ window, every request as if submitted alone.

    Under the OCP sequential write mode reads follow the host page table, so
    the out-of-order window (execute_read_batch) does not apply and each
    request is submitted on its own. NandCore then serves a span by the
    scalar page machine (_execute_detailed_accepted) when it touches a
    staged tail page, or when it is shorter than BULK_MIN_PAGES with no
    resident controller. Consecutive such requests run here in one compiled
    pass with the same host admissions, page order, cache and frontier
    updates, float operations and per-request bills; the touched state is
    published once. Every other request is submitted normally, in order.
    Returns None before any mutation when the core, timeline or sink is not
    the native one.
    """
    from .nand_core import NandCore, BULK_MIN_PAGES, RESIDENT_MIN_PAGES, _generic_bulk_span
    from .memory_types import AccessRequest
    from .memory_mapping import validate_nand_request
    from .memory_transfer import _OWNED_TIMELINE_TOKEN_TYPE, _OWNED_TIMELINE_REMEMBER

    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or resource_sink is not None and (
                type(resource_sink) is not tuple or len(resource_sink) != 6
                or any(type(value) is not dict for value in resource_sink)
                or len({id(value) for value in resource_sink}) != 6)
            or config.protocol_mode != "ocp_hbf_0_7" or not core._sequential_write_mode()
            or config.queue_scope != "host_channel" or config.page_staging == "none"
            or config.parallel_units is not None or not config.planes_independent
            or config.host_channels != config.channels
            or (config.targets_per_channel, config.dies_per_target, config.luns_per_die) != (1, 1, 1)
            or 4096 % config.host_granularity_bytes or config.page_staging_burst_bytes <= 0
            or core._write_buffer_pending or type(core._cache_use) is not int
            or timeline._discarded or not _generic_resident_writers(core)
            or any(type(token) is not _OWNED_TIMELINE_TOKEN_TYPE or token.accounting
                   or getattr(token.remember, "__func__", token.remember) is not _OWNED_TIMELINE_REMEMBER
                   for token in timeline._owned_transactions)):
        return None
    spans = []
    for request in requests:
        span = None
        if (type(request) is AccessRequest and request.operation is Operation.READ
                and request.byte_count > 0):
            try:
                validate_nand_request(request, config)
            except Exception:
                pass
            else:
                first, last = request.address // 4096, (request.address + request.byte_count - 1) // 4096
                span = (last - first + 1, bool(core._tail_staged(first, last)))
        spans.append(span)

    def scalar(span, resident):
        # NandCore._submit_live's route to the scalar page machine.
        if span is None:
            return False
        pages, staged = span
        if resident:
            return staged and pages >= RESIDENT_MIN_PAGES
        return staged or pages < BULK_MIN_PAGES

    def resident_bulk(request, span):
        # A plain bulk READ: NandCore._submit_live sends it to
        # _submit_generic_summary whether or not the controller is resident,
        # and that executor packs a published controller unchanged.
        return (span is not None and not span[1] and span[0] >= BULK_MIN_PAGES
                and _generic_bulk_span(request, config)
                and core._nand_numeric_enabled and not core._nand_state_exposed
                and not core._write_buffer_pending and not core._write_buffer_occupancy
                and _generic_resident_writers(core)
                and core._nand_numeric_state.scalar_resource_contract == core._resident_resource_contract())

    results = [None] * len(requests)
    # A run on a resident controller updates its arrays instead of publishing
    # them as the scalar machine does. Until the next request is a bulk READ
    # (which repacks that published state), the controller is then
    # published at the latest before anything else sees it.
    index, batching, published = 0, True, False
    try:
        while index < len(requests):
            if batching and scalar(spans[index], core._nand_numeric_state is not None and not published):
                stop = index + 1
                while stop < len(requests) and scalar(spans[stop], False):
                    stop += 1
                done = _sequential_read_run(core, requests[index:stop], resource_sink, results, index)
                if done is None:
                    batching = False
                    continue
                if done:
                    published = core._nand_numeric_state is not None
                    index += done
                    continue
            if published and not resident_bulk(requests[index], spans[index]):
                core._nand_publish_numeric()
            published = False
            results[index] = (core.submit(requests[index]) if resource_sink is None
                              else core.submit(requests[index], _resource_sink=resource_sink))
            index += 1
    finally:
        if published:
            core._nand_publish_numeric()
    return tuple(results)


_PAGE_ADDRESS = attrgetter("address")


def _sequential_read_run(core, run, resource_sink, results, at):
    """Run a prefix of ``run`` (scalar-routed READs) and return its length.

    None declines the window's structure (lanes, cache objects or layout);
    0 leaves the first request to NandCore.submit, whose scalar page machine
    raises on a staged page it cannot read.
    """
    import numpy as np
    from .nand_core import _ReadCacheSlot
    from .memory_types import TransactionResult
    from ._nand_execution import _geometry_plan
    from ._compiled_nand import run_sequential_read_group, fold_sequential_read_bill

    config, timeline = core.config, core.timeline
    base_units = config.channels
    banks = base_units * config.planes_per_lun
    pages_per_block = config.pages_per_block
    if banks > 65536 or config.effective_capacity_bytes < banks * 4096:
        return None
    (bank_keys, prefixes, array_names, array_resources, channel_names, host_names, names,
     array_ids, channel_ids, host_ids) = _geometry_plan(core, Operation.READ, banks, banks, 1)
    if array_names != array_resources:
        return None
    valid, _bus_mask = _buffered_layout_contract(array_names, array_resources, channel_names, host_names,
                                                 names, tuple(timeline.resource_aliases.items()))
    if not valid:
        return None

    # Host channels: acceptance clocks and live command completions, ascending.
    hosts, request_host, arrivals = {}, [], []
    for request in run:
        host = map_nand_address(config, request.address).channel % config.host_channels
        request_host.append(hosts.setdefault(host, len(hosts)))
        arrivals.append(request.arrival_ns)
    inflight = core.__dict__["_host_inflight"]
    waiting = np.bincount(np.array(request_host, dtype=np.int64), minlength=len(hosts))
    host_low = np.zeros(len(hosts), dtype=np.int64)
    host_high = np.zeros(len(hosts), dtype=np.int64)
    ends_rows, offset = [], 0
    for host, local in hosts.items():
        ends = sorted(end for end, _operation in inflight.get(host, ()))
        host_low[local], host_high[local] = offset, offset + len(ends)
        ends_rows.append(ends + [0.0] * int(waiting[local]))
        offset += len(ends) + int(waiting[local])
    host_ends = np.array([end for row in ends_rows for end in row], dtype=np.float64)
    acceptance = np.array([max(core._host_acceptance_ns.get(host, 0.), core._acceptance_ns) for host in hosts],
                          dtype=np.float64)
    limit = config.max_outstanding_requests
    limit = 2**62 if limit is None else int(limit)

    # Pages: the page-table mapping, host spans and staged tail pages.
    granularity, burst = config.host_granularity_bytes, config.page_staging_burst_bytes
    logic = config.page_staging == "logic_die"
    host_bandwidth = float(core._host_bandwidth(Operation.READ))
    table = core._logical_page_table
    tails = core._tail_pages
    counts, physical, head_tail, staged_rows = [], [], [], []
    for request in run:
        end = request.address + request.byte_count
        first, last = request.address // 4096, (end - 1) // 4096
        counts.append(last - first + 1)
        mapped = list(map(table.get, range(first, last + 1)))
        try:
            physical.append(np.fromiter(map(_PAGE_ADDRESS, mapped), dtype=np.int64, count=len(mapped)) // 4096)
        except AttributeError:  # an unmapped page reads its own address
            physical.append(np.array([page if mapping is None else mapping.address // 4096
                                      for page, mapping in zip(range(first, last + 1), mapped)],
                                     dtype=np.int64))
        head_end = min(end, (first + 1) * 4096)
        head_tail.append((-(-head_end // granularity) * granularity - request.address // granularity * granularity,
                          -(-end // granularity) * granularity - last * 4096 if last > first else 0))
        rows = []
        for page in core._tail_staged(first, last):
            tail = tails[page]
            base = page * 4096
            lo, hi = max(request.address, base) - base, min(end, base + 4096) - base
            rows.append((page - first, any(left <= lo and hi <= right for left, right in tail[0]),
                         lo == 0 and hi == 4096, tail[2], tail[3],
                         -(-hi // burst) * burst - lo // burst * burst))
        staged_rows.append(rows)
    request_pages = np.array(counts, dtype=np.int64)
    total = int(request_pages.sum())
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(request_pages, out=starts[1:])
    pages = np.concatenate(physical)
    host_bytes = np.full(total, 4096, dtype=np.int64)
    host_bytes[starts[:-1]] = [row[0] for row in head_tail]
    multi = request_pages > 1
    host_bytes[(starts[1:] - 1)[multi]] = np.array([row[1] for row in head_tail], dtype=np.int64)[multi]
    page_staged = np.zeros(total, dtype=np.bool_)
    page_covered = np.zeros(total, dtype=np.bool_)
    page_full = np.zeros(total, dtype=np.bool_)
    page_ready = np.zeros(total, dtype=np.float64)
    page_end = np.zeros(total, dtype=np.float64)
    scratch_bytes = np.zeros(total, dtype=np.int64)
    for request, rows in enumerate(staged_rows):
        for offset, covered, full, ready, program_end, moved in rows:
            position = starts[request] + offset
            page_staged[position], page_covered[position], page_full[position] = True, covered, full
            page_ready[position], page_end[position], scratch_bytes[position] = ready, program_end, moved
    bank = pages % banks
    channel = bank % base_units
    tag = pages // banks
    # Compact the window's banks and resources; scratchpad lanes follow.
    used_banks, page_bank = np.unique(bank, return_inverse=True)
    page_bank = page_bank.astype(np.int64)
    used_banks = used_banks.tolist()
    used_resources, inverse = np.unique(np.concatenate((array_ids[bank], channel_ids[channel],
                                                         host_ids[channel % config.host_channels])),
                                        return_inverse=True)
    resource_names = [names[index] for index in used_resources.tolist()]
    page_ar = inverse[:total].astype(np.int64)
    page_ch = inverse[total:2 * total].astype(np.int64)
    page_host = inverse[2 * total:].astype(np.int64)
    page_scratch = np.zeros(total, dtype=np.int64)
    scratch_ns = np.zeros(total, dtype=np.float64)
    if logic and np.any(page_staged):
        plan_names = set(names)
        scratch_ids = {}
        for position in np.flatnonzero(page_staged).tolist():
            lane_channel = int(channel[position])
            rid = scratch_ids.get(lane_channel)
            if rid is None:
                name = core._staging_host_id(map_nand_address(config, lane_channel * 4096), Operation.READ)
                if name in plan_names or name in resource_names:
                    return None
                rid = scratch_ids[lane_channel] = len(resource_names)
                resource_names.append(name)
            page_scratch[position] = rid
            scratch_ns[position] = int(scratch_bytes[position]) / host_bandwidth
    host_ns = host_bytes / host_bandwidth
    array_local = {name: index for index, name in enumerate(resource_names)}
    unit_array = np.array([array_local[array_resources[unit]] for unit in used_banks], dtype=np.int64)

    # Pack the touched banks, units and calendars as the scalar machine reads them.
    cache = core.__dict__["_read_cache"]
    array_frontiers, buffer_frontiers = core.__dict__["_array_ready"], core.__dict__["_buffer_ready"]
    size = len(used_banks)
    tags = np.full((size, 2), -1, dtype=np.int64)
    cache_ready = np.zeros((size, 2), dtype=np.float64)
    readers = np.zeros((size, 2), dtype=np.float64)
    uses = np.zeros((size, 2), dtype=np.int64)
    array_ready = np.zeros(size, dtype=np.float64)
    buffer_ready = np.zeros(size, dtype=np.float64)
    packed = core._nand_numeric_state
    resident = packed is not None and packed.layout == (bank_keys, prefixes, array_names, pages_per_block)
    if packed is not None and not resident:
        # The first request publishes the resident controller before its
        # scalar page machine; the window runs on the published state.
        core._nand_publish_numeric()
    rows = np.array(used_banks, dtype=np.int64)
    if resident:
        # The scalar machine reads exactly these values once the controller
        # is published; the window reads them in place instead.
        frontier_array, held_array, tag_array, ready_array, reader_array, use_array, present = packed.arrays
        tags, cache_ready, readers, uses = tag_array[rows], ready_array[rows], reader_array[rows], use_array[rows]
        array_ready, buffer_ready = frontier_array[rows], held_array[rows]
        if not all(np.all(np.isfinite(values)) and np.all(values >= 0)
                   for values in (cache_ready, readers, array_ready, buffer_ready)):
            return None
    for local, unit in enumerate(() if resident else used_banks):
        slots = cache.get(bank_keys[unit])
        if slots is not None:
            if type(slots) is not list or len(slots) != 2:
                return None
            prefix = prefixes[unit]
            for index, slot in enumerate(slots):
                if type(slot) is not _ReadCacheSlot:
                    return None
                key, ready, reader, use = slot.page_key, slot.ready_ns, slot.reader_completion_ns, slot.last_use
                if (type(ready) not in (int, float) or type(reader) not in (int, float) or type(use) is not int
                        or not (isfinite(ready) and isfinite(reader)) or ready < 0 or reader < 0 or use < 0):
                    return None
                if key is not None:
                    if (type(key) is not tuple or len(key) != len(prefix) + 2 or key[:-2] != prefix
                            or type(key[-2]) is not int or type(key[-1]) is not int
                            or key[-2] < 0 or not 0 <= key[-1] < pages_per_block):
                        return None
                    tags[local, index] = key[-2] * pages_per_block + key[-1]
                cache_ready[local, index], readers[local, index], uses[local, index] = ready, reader, use
        name = array_names[unit]
        frontier, held = array_frontiers.get(name, 0.0), buffer_frontiers.get(name, 0.0)
        if (type(frontier) not in (int, float) or type(held) not in (int, float)
                or not (isfinite(frontier) and isfinite(held)) or frontier < 0 or held < 0):
            return None
        array_ready[local], buffer_ready[local] = frontier, held
    lane_map, ready_map, busy_map = timeline.lane_available, timeline.ready_ns, timeline.busy_ns
    moved_map, interval_map = timeline.bytes_moved, timeline.last_intervals
    lanes = np.zeros(len(resource_names), dtype=np.float64)
    state = np.zeros((len(resource_names), 4), dtype=np.float64)
    moved = np.zeros(len(resource_names), dtype=np.int64)
    lane_lists = set()
    for local, name in enumerate(resource_names):
        slots = lane_map.get(name)
        if slots is not None:
            # A lane list shared by two calendars would couple them.
            if id(slots) in lane_lists:
                return None
            lane_lists.add(id(slots))
        lane = ready_map.get(name, 0.0) if slots is None else (
            slots[0] if type(slots) is list and len(slots) == 1 else None)
        busy, byte_count = busy_map.get(name, 0.0), moved_map.get(name, 0)
        if (type(lane) not in (int, float) or not isfinite(lane) or lane < 0
                or type(busy) not in (int, float) or not isfinite(busy)
                or type(byte_count) is not int or not 0 <= byte_count <= 2**62):
            return None
        lanes[local], state[local, 1], moved[local] = lane, busy, byte_count

    (processed, effective, completions, waits, hits, misses, staged, staged_host, staged_bytes, reserved,
     clock, entry_offsets, entry_rid, busy_before, busy_after, moved_before, moved_after, entry_start,
     entry_end, resource_order, transferred, bank_order, unit_order, slot_used, slot_filled,
     slot_read) = run_sequential_read_group(
        request_pages, np.array(arrivals, dtype=np.float64), np.array(request_host, dtype=np.int64),
        acceptance, float(core._acceptance_ns), host_ends, host_low, host_high, limit,
        float(config.front_ns), float(config.page_staging_latency_ns), logic,
        page_bank, tag, page_ar, page_ch, page_host, page_scratch, host_ns, host_bytes,
        page_staged, page_covered, page_full, page_ready, page_end, scratch_ns, scratch_bytes,
        float(config.page_read_ns), config.transfer_page_bytes / float(config.internal_bandwidth_gb_s),
        int(config.transfer_page_bytes), unit_array, array_ready, buffer_ready,
        tags, cache_ready, readers, uses, core._cache_use, lanes, state, moved)
    if not processed:
        return 0
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(completions[:processed]))):
        raise ValueError("resource reservation exceeds finite timing range")
    run = run[:processed]

    # Publish each touched calendar once, in first-touch order, after its
    # undo journals have their preimage (the scalar recorder resets first).
    timeline._reset_metric_recorder(0)
    order = resource_order.tolist()
    touched_names = [resource_names[rid] for rid in order]
    timeline._remember_resources(touched_names)
    lane_values, state_values = lanes.tolist(), state.tolist()
    moved_values, transfer_flags = moved.tolist(), transferred.tolist()
    direction_map, lane_direction_map = timeline.directions, timeline.lane_directions
    for rid, name in zip(order, touched_names):
        slots = lane_map.get(name)
        if slots is None:
            lane_map[name] = [lane_values[rid]]
        else:
            slots[0] = lane_values[rid]
        row = state_values[rid]
        ready_map[name] = row[0]
        busy_map[name] = row[1]
        interval_map[name] = (row[2], row[3])
        if transfer_flags[rid]:
            direction_map[name] = "read"
        lane_direction_map.pop(name, None)
        if transfer_flags[rid]:
            moved_map[name] = moved_values[rid]
    last_request = processed - 1
    timeline._touched = set(resource_names[rid] for rid in
                            entry_rid[entry_offsets[last_request]:entry_offsets[last_request + 1]].tolist())
    timeline._intervals_truncated = bool(reserved[last_request])
    if resident:
        for token in core._nand_transactions:
            token.capture_numeric(packed)
        tag_array[rows], ready_array[rows], reader_array[rows], use_array[rows] = tags, cache_ready, readers, uses
        frontier_array[rows], held_array[rows] = array_ready, buffer_ready
        accessed = rows[bank_order]
        packed.dirty_banks[accessed] = True
        present[accessed] = True
        packed.dirty_units[rows[unit_order]] = True
    else:
        frontier_values, held_values = array_ready.tolist(), buffer_ready.tolist()
        for local in unit_order.tolist():
            name = array_names[used_banks[local]]
            core._touch_frontiers(name)
            array_frontiers[name] = frontier_values[local]
            buffer_frontiers[name] = held_values[local]
        tag_values, ready_values, reader_values, use_values = (
            tags.tolist(), cache_ready.tolist(), readers.tolist(), uses.tolist())
        used, filled, read = slot_used.tolist(), slot_filled.tolist(), slot_read.tolist()
        for local in bank_order.tolist():
            unit = used_banks[local]
            key = bank_keys[unit]
            if core._nand_transactions:
                core._touch_cache_bank(key)
            slots = cache.get(key)
            if slots is None:
                slots = [_ReadCacheSlot(), _ReadCacheSlot()]
                cache[key] = slots
            for index in (0, 1):
                if not used[local][index]:
                    continue
                slot = slots[index]
                if filled[local][index]:
                    slot.page_key = (*prefixes[unit], *divmod(tag_values[local][index], pages_per_block))
                    slot.ready_ns = ready_values[local][index]
                    slot.reader_completion_ns = reader_values[local][index]
                elif read[local][index]:
                    slot.reader_completion_ns = reader_values[local][index]
                slot.last_use = use_values[local][index]
    core._cache_use = int(clock)
    core._observation_events = []
    core._last_media_queue_wait_ns = float(waits[last_request])

    if resource_sink is None:
        bills = _sequential_read_bills(timeline, resource_names, entry_offsets, entry_rid, busy_before,
                                       busy_after, moved_before, moved_after, entry_start, entry_end)
    else:
        bills = None
        _fold_sequential_read_sink(timeline, resource_sink, resource_names, entry_offsets, entry_rid,
                                   busy_before, busy_after, moved_before, moved_after, entry_start, entry_end,
                                   fold_sequential_read_bill)

    page_bytes, transfer_bytes = config.page_bytes, config.transfer_page_bytes
    hit_ceiling = config.host_bandwidth_for(Operation.READ)
    miss_ceiling = core._bandwidth_ceiling_gb_s(Operation.READ)
    hit_scope = ("directional host upper bound for transactions using cached/staged data; "
                 "host acknowledgement is not media drain")
    miss_scope = ("device transfer upper bound: min(directional host, summed independent channels); "
                  "excludes array latency")
    array_units, depth_scope = config.array_units, core._admission_depth_scope()
    held_bytes, draining = core._held_write_buffer_bytes(), core.inflight_write_buffer_bytes
    buffered = core._buffered_write_mode
    for position, request in enumerate(run):
        arrival, admitted, host_channel = core._admit_host_request(request)
        if arrival != effective[position]:
            raise RuntimeError("sequential NAND read admission diverged from the host queue")
        count, hit, miss = counts[position], int(hits[position]), int(misses[position])
        staged_count = int(staged[position])
        head, tail = head_tail[position]
        host_total = head + (tail + (count - 2) * 4096 if count > 1 else 0) - int(staged_host[position]) + (
            int(staged_bytes[position]) if logic else 0)
        internal = miss * transfer_bytes
        completion = float(completions[position])
        cached = bool(staged_count or hit)
        counters = {
            "host_transfer_bytes": host_total, "internal_transfer_bytes": internal,
            "physical_read_bytes": miss * page_bytes, "physical_write_bytes": 0,
            "pages_read": miss, "pages_programmed": 0,
            "merge_bytes": 0, "merged_pages": 0, "staging_write_pages": 0,
            "staging_read_hits": staged_count, "staging_read_merges": 0,
            "read_cache_hits": hit, "read_cache_misses": miss,
            "erase_operations": 0, "page_count": count,
            "details_truncated": count > config.max_expanded_segments,
            "bandwidth_ceiling_gb_s": hit_ceiling if cached else miss_ceiling,
            "bandwidth_ceiling_scope": hit_scope if cached else miss_scope,
            "array_units": array_units, "write_completion": "n/a",
            "host_ack_completion_ns": None, "media_drain_completion_ns": None,
            "media_queue_wait_ns": float(waits[position]),
            "pages_touched": count, "read_operations": miss, "program_operations": 0,
            "rmw_read_operations": 0, "page_access_payload_bytes": request.byte_count,
            "page_access_capacity_bytes": count * page_bytes, "host_admission_samples": 1,
            "host_outstanding_at_admission_peak": len(admitted) + 1,
            "host_read_outstanding_at_admission_peak": admitted.reads + 1,
            "host_write_outstanding_at_admission_peak": admitted.writes,
            "host_admission_depth_scope": depth_scope,
            "queue_wait_ns": arrival - request.arrival_ns,
            "write_buffer_wait_ns": 0.0, "write_buffer_bytes": 0,
            "write_buffer_occupancy_bytes": held_bytes, "write_buffer_peak_bytes": held_bytes,
            "write_buffer_draining_bytes": draining, "buffered_write_mode": buffered}
        if bills is not None:
            (counters["resource_busy_ns"], counters["resource_bytes"],
             counters["resource_last_intervals"]) = bills[position]
        counters["resource_intervals"], counters["resource_interval_payloads"] = {}, {}
        counters["intervals_truncated"] = bool(reserved[position])
        results[at + position] = TransactionResult(
            request.request_id, Operation.READ, request.arrival_ns, completion, request.byte_count, internal,
            counters=counters)
        core._record_host_admission(host_channel, arrival, admitted, completion, Operation.READ)
    return processed


def _sequential_read_bills(timeline, resource_names, entry_offsets, entry_rid, busy_before, busy_after,
                           moved_before, moved_after, starts, ends):
    """Each request's metric recorder maps, as ResourceTimeline._core_metrics_delta returns them.

    Busy and byte entries exist where the counter changed, in the timeline
    column's order; intervals follow the request's touched set.
    """
    import numpy as np

    busy_order = {rid: index for index, rid in enumerate(timeline.busy_ns)}
    byte_order = {rid: index for index, rid in enumerate(timeline.bytes_moved)}
    busy_position = np.array([busy_order.get(name, -1) for name in resource_names], dtype=np.int64)
    byte_position = np.array([byte_order.get(name, -1) for name in resource_names], dtype=np.int64)
    busy_delta = busy_after - busy_before
    byte_delta = moved_after - moved_before
    busy_changed = busy_after != busy_before
    byte_changed = moved_after != moved_before
    bills = []
    for request in range(len(entry_offsets) - 1):
        lo, hi = int(entry_offsets[request]), int(entry_offsets[request + 1])
        rids = entry_rid[lo:hi]
        mask = busy_changed[lo:hi]
        selected = rids[mask]
        order = np.argsort(busy_position[selected], kind="stable")
        busy = dict(zip([resource_names[rid] for rid in selected[order].tolist()],
                        busy_delta[lo:hi][mask][order].tolist()))
        mask = byte_changed[lo:hi]
        selected = rids[mask]
        order = np.argsort(byte_position[selected], kind="stable")
        moved = dict(zip([resource_names[rid] for rid in selected[order].tolist()],
                         byte_delta[lo:hi][mask][order].tolist()))
        touched = [resource_names[rid] for rid in rids.tolist()]
        intervals = dict(zip(touched, zip(starts[lo:hi].tolist(), ends[lo:hi].tolist())))
        bills.append((busy, moved, {rid: intervals[rid] for rid in set(touched)}))
    return bills


def _fold_sequential_read_sink(timeline, resource_sink, resource_names, entry_offsets, entry_rid,
                               busy_before, busy_after, moved_before, moved_after, starts, ends, fold):
    """Bill each request into the task and owner sinks, as consume_task_resource_sink.

    New sink keys keep the order consuming the per-request maps would give:
    the first request that carries them, then that map's order (busy and
    bytes follow the timeline's columns, intervals the request's touched set).
    """
    import numpy as np

    size = len(resource_names)
    columns = {}

    def column_order(name):
        if name not in columns:
            values = getattr(timeline, name)
            columns[name] = {rid: index for index, rid in enumerate(values)}
        return columns[name]

    touched_orders = {}

    def touched_order(request):
        if request not in touched_orders:
            touched = set()
            for rid in entry_rid[entry_offsets[request]:entry_offsets[request + 1]].tolist():
                touched.add(resource_names[rid])
            touched_orders[request] = {rid: index for index, rid in enumerate(touched)}
        return touched_orders[request]

    for busy_map, byte_map, last_map in (resource_sink[:3], resource_sink[3:]):
        busy = np.array([float(busy_map.get(name, 0.0)) for name in resource_names], dtype=np.float64)
        has_busy = np.array([name in busy_map for name in resource_names], dtype=np.bool_)
        byte_counts = np.array([byte_map.get(name, 0) for name in resource_names], dtype=np.int64)
        has_bytes = np.array([name in byte_map for name in resource_names], dtype=np.bool_)
        previous = [last_map.get(name) for name in resource_names]
        has_last = np.array([value is not None for value in previous], dtype=np.bool_)
        last_start = np.array([0.0 if value is None else value[0] for value in previous], dtype=np.float64)
        last_end = np.array([0.0 if value is None else value[1] for value in previous], dtype=np.float64)
        initial_start, initial_end = last_start.copy(), last_end.copy()
        initial_busy = has_busy.copy()
        first_busy, first_bytes, first_last = fold(
            entry_offsets, entry_rid, busy_before, busy_after, moved_before, moved_after, starts, ends,
            busy, has_busy, byte_counts, has_bytes, last_start, last_end, has_last)
        changed = np.zeros(size, dtype=np.bool_)
        changed[entry_rid[busy_after != busy_before]] = True
        busy_values = busy.tolist()
        for rid in np.flatnonzero(changed & initial_busy).tolist():
            busy_map[resource_names[rid]] = busy_values[rid]
        fresh = np.flatnonzero(first_busy >= 0).tolist()
        if fresh:
            order = column_order("busy_ns")
            for rid in sorted(fresh, key=lambda rid: (first_busy[rid], order[resource_names[rid]])):
                busy_map[resource_names[rid]] = busy_values[rid]
        changed[:] = False
        changed[entry_rid[moved_after != moved_before]] = True
        byte_values = byte_counts.tolist()
        for rid in np.flatnonzero(changed & ~(first_bytes >= 0)).tolist():
            byte_map[resource_names[rid]] = byte_values[rid]
        fresh = np.flatnonzero(first_bytes >= 0).tolist()
        if fresh:
            order = column_order("bytes_moved")
            for rid in sorted(fresh, key=lambda rid: (first_bytes[rid], order[resource_names[rid]])):
                byte_map[resource_names[rid]] = byte_values[rid]
        start_values, end_values = last_start.tolist(), last_end.tolist()
        replaced = (last_start != initial_start) | (last_end != initial_end)
        for rid in np.flatnonzero(replaced & (first_last < 0)).tolist():
            last_map[resource_names[rid]] = (start_values[rid], end_values[rid])
        fresh = np.flatnonzero(first_last >= 0).tolist()
        if fresh:
            for rid in sorted(fresh, key=lambda rid: (first_last[rid],
                                                      touched_order(int(first_last[rid]))[resource_names[rid]])):
                last_map[resource_names[rid]] = (start_values[rid], end_values[rid])


def _native_numeric_qualified(core):
    """Prove the resident controller arrays and layout are the native ones."""
    import numpy as np
    from ._nand_execution import _geometry_plan

    packed = core._nand_numeric_state
    arrays = packed.arrays
    if type(arrays) is not tuple or len(arrays) != 7:
        return False
    values = (*arrays, packed.dirty_units, packed.dirty_banks)
    dtypes = (np.dtype("float64"), np.dtype("float64"), np.dtype("int64"),
              np.dtype("float64"), np.dtype("float64"), np.dtype("int64"),
              np.dtype("bool"), np.dtype("bool"), np.dtype("bool"))
    # Owning arrays have independent allocations unless the same array
    # object is repeated. Only array ready must not overlap any other field.
    if (any(type(item) is not np.ndarray for item in values)
            or any(item.dtype != dtype for item, dtype in zip(values, dtypes))
            or any(not item.flags.owndata for item in values)
            or any(item is arrays[0] for item in values[1:])
            or arrays[0].ndim != 1 or not arrays[0].flags.c_contiguous):
        return False
    config = core.config
    base = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base * config.planes_per_lun
    units = banks if config.planes_independent else base
    shapes = ((units,), (units,), (banks, 2), (banks, 2), (banks, 2),
              (banks, 2), (banks,), (units,), (banks,))
    if any(item.shape != shape for item, shape in zip(values, shapes)):
        return False
    layout = packed.layout
    if type(layout) is not tuple or len(layout) != 4 or type(layout[3]) is not int:
        return False
    expected = _geometry_plan(core, Operation.READ, banks, units, base // config.channels)
    for actual, native, atom in zip(layout[:3], expected[:3], (int, int, str)):
        if actual is native:
            continue
        if type(actual) is not tuple:
            return False
        if atom is str:
            valid = all(type(item) is str for item in actual)
        else:
            valid = all(type(item) is tuple
                        and all(type(part) is int for part in item) for item in actual)
        if not valid or actual != native:
            return False
    if layout[3] != config.pages_per_block:
        return False
    contract = packed.scalar_resource_contract
    if (type(contract) is not tuple or len(contract) != 12
            or any(type(item) not in (int, bool, type(None)) for item in contract[:10])
            # The fifth prefix is the per-channel logic-die scratchpad lane;
            # generic NAND keeps it in the immutable contract as well so a
            # later staging-mode edit cannot reuse a stale resident pack.
            or type(contract[10]) is not tuple or len(contract[10]) != 5
            or any(type(item) is not str for item in contract[10])
            or type(contract[11]) is not tuple
            or any(type(item) is not tuple or len(item) != 2
                   or any(type(part) is not str for part in item) for item in contract[11])):
        return False
    return contract == core._resident_resource_contract()


def _native_sink_qualified(core):
    """A resident native READ core whose batches bill a task sink directly.

    Physical task billing hands this core a private resource sink. Prove
    that its constructor, admission and validation callbacks, controller
    arrays and undo tokens are the native ones before doing so.
    """
    from . import nand_core as owner
    from .memory_transfer import _OWNED_TIMELINE_TOKEN_TYPE, _OWNED_TIMELINE_TOKEN_METHODS
    if (type(core._nand_numeric_state) is not _NATIVE_NUMERIC_TYPE
            or type(core.config) is not _FOOTPRINT_CONFIG_TYPE
            or not _footprint_namespace_matches(_FOOTPRINT_CORE_TYPE, _FOOTPRINT_CORE_ITEMS)
            or any(type(getattr(core, name)) is not bool for name in (
                "_nand_numeric_enabled", "_nand_state_exposed", "capture_details",
                "_observe_media", "_buffered_write_mode"))
            or type(core._staged_pages) is not dict
            or core.config.parallel_units is not None
            or not _footprint_namespace_matches(_FOOTPRINT_CONFIG_TYPE, _FOOTPRINT_CONFIG_ITEMS)
            or any(not _footprint_namespace_matches(cls, items) for cls, items in _NATIVE_CLASSES)
            or any(getattr(module, name) is not value for module, name, value in _NATIVE_BINDINGS)):
        return False
    if any(type(key) is not str for key in core.config.metadata):
        return False
    for name, value in vars(core.config).items():
        if name in ("kind", "metadata"):
            continue
        if not (value is None or type(value) is str or type(value) is int
                or type(value) is float or type(value) is bool):
            return False
    if not _native_numeric_qualified(core):
        return False
    if ((type(core._acceptance_ns) is not int and type(core._acceptance_ns) is not float)
            or type(core._write_buffer_occupancy) is not int
            or type(core._write_buffer_pending) is not list
            or not _footprint_methods_absent(core._nand_numeric_state, ("checkpoint", "restore", "publish"))):
        return False
    population = core.__dict__["_inflight_operations"]
    if (type(population) is not owner._HostPopulation
            or type(population.commands) is not dict
            or type(population.ends) is not list or type(population.write_ends) is not list
            or any(type(getattr(population, name)) is not int for name in ("next_id", "reads", "writes"))):
        return False
    if not _footprint_methods_absent(population, ("preview", "record")):
        return False
    for key, item in population.commands.items():
        if (type(key) is not int or type(item) is not tuple or len(item) != 2
                or (type(item[0]) is not int and type(item[0]) is not float)
                or item[1] is not Operation.READ and item[1] is not Operation.WRITE):
            return False
    for item in (*population.ends, *population.write_ends):
        if (type(item) is not tuple or len(item) != 2 or type(item[1]) is not int
                or (type(item[0]) is not int and type(item[0]) is not float)):
            return False
    if (type(core._nand_transactions) is not list
            or any(type(token) is not owner._NandTransaction
                   or not _footprint_methods_absent(token, ("capture_numeric", "remember_frontiers",
                       "materialize_frontiers", "remember_cache", "materialize_cache", "rollback"))
                   for token in core._nand_transactions)
            or type(core.timeline._owned_transactions) is not list):
        return False
    public_roots = tuple(getattr(core.timeline, name) for name in (
        "ready_ns", "busy_ns", "bytes_moved", "last_intervals",
        "lane_available", "directions", "lane_directions")) + (
            core.__dict__["_array_ready"], core.__dict__["_buffer_ready"], core.__dict__["_read_cache"])
    for token in core._nand_transactions:
        # Only missing native cache banks call remember_cache in the READ
        # executor; cache_slots is not read when the previous bank is absent.
        if (type(token.cache_banks) is not dict
                or any(token.cache_banks is item for item in public_roots)
                or any(type(key) is not tuple or any(type(part) is not int for part in key)
                       for key in token.cache_banks)):
            return False
    for token in core.timeline._owned_transactions:
        if (type(token) is not _OWNED_TIMELINE_TOKEN_TYPE
                or type(token.accounting) is not bool or token.accounting
                or any(getattr(getattr(token, name), "__func__", getattr(token, name)) is not value
                       for name, value in _OWNED_TIMELINE_TOKEN_METHODS)
                or any(type(getattr(token, name)) is not dict
                       for name in ("roots", "entries", "objects", "deleted_orders"))
                or any(type(value) is not dict for value in token.entries.values())
                or any(type(key) is not str for key in token.roots)
                or any(type(key) is not str for key in token.entries)
                or any(type(key) is not int for key in token.objects)):
            return False
        # Recorder roots are replaced before execution; only these seven
        # persistent calendar journals are queried by native publication.
        for name in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals",
                     "lane_available", "directions", "lane_directions"):
            saved = token.entries.get(name)
            if saved is not None and any(type(key) is not str for key in saved):
                return False
    return True


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
    if (config.host_channels != 1
            or config.host_full_duplex or config.read_cache_pages_per_bank
            or config.page_staging != "none"):
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
    # Match the compiled page runner's inherited-frontier metric.  Snapshot
    # the resource state before this invocation so serialization introduced by
    # later pages in the same affine batch is not reported as media wait.
    max_read_wait_ticks = 0
    if request.operation is Operation.READ:
        for array, resource, _channel, _count, _host_ns in page_program:
            inherited = max(state["array", array], state["buffer", array], state["lane", resource])
            max_read_wait_ticks = max(max_read_wait_ticks, inherited - base_ticks)
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
        "media_queue_wait_ns": float(max_read_wait_ticks / scale),
    }


@lru_cache(maxsize=128)
def _owned_resource_footprint(read_names, write_names, array_names):
    return tuple(dict.fromkeys((*read_names, *write_names, *array_names)))


def owned_resource_footprint(core):
    """All possible native calendars for this owner, including virtual arrays."""
    from ._nand_execution import _geometry_plan
    config = core.config
    # Cache only immutable geometry descriptions, after rechecking every
    # callback that builds them. Live calendar values are still validated by
    # PhysicalRuntimeContext on every call. Unknown hooks execute uncached.
    key = _footprint_cache_key(core)
    if key is not None and key in _FOOTPRINT_CACHE:
        return _FOOTPRINT_CACHE[key]
    base = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    banks = base * config.planes_per_lun
    if banks > 65536:
        return None
    units = banks if config.planes_independent else base
    per_channel = base // config.channels
    read = _geometry_plan(core, Operation.READ, banks, units, per_channel)
    write = _geometry_plan(core, Operation.WRITE, banks, units, per_channel)
    resources = _owned_resource_footprint(read[6], write[6], read[2])
    if key is not None:
        if len(_FOOTPRINT_CACHE) >= 128:
            _FOOTPRINT_CACHE.clear()
        _FOOTPRINT_CACHE[key] = resources
    return resources


from . import _nand_pack as _owned_pack_module
from . import _compiled_nand as _owned_numeric_module
from . import _nand_execution as _owned_execution_module
from . import memory_mapping as _footprint_mapping_module
from .memory_types import NandConfig as _FOOTPRINT_CONFIG_TYPE, AddressMapping as _FOOTPRINT_MAPPING_TYPE
from .nand_core import NandCore as _FOOTPRINT_CORE_TYPE
from ._nand_pack import native_namespace_matches as _footprint_namespace_matches
from ._nand_pack import native_instance_methods_absent as _footprint_methods_absent

_FOOTPRINT_CACHE = {}
_FOOTPRINT_CONFIG_ITEMS = tuple((name, value) for name, value in vars(_FOOTPRINT_CONFIG_TYPE).items() if name != "__slotnames__")
_FOOTPRINT_MAPPING_ITEMS = tuple((name, value) for name, value in vars(_FOOTPRINT_MAPPING_TYPE).items() if name != "__slotnames__")
_FOOTPRINT_CORE_ITEMS = tuple((name, value) for name, value in vars(_FOOTPRINT_CORE_TYPE).items() if name != "__slotnames__")
_FOOTPRINT_CORE_METHOD_NAMES = tuple(name for name, value in _FOOTPRINT_CORE_ITEMS if callable(value))
_FOOTPRINT_MAPPING_FUNCTIONS = tuple((name, getattr(_footprint_mapping_module, name))
    for name in ("map_nand_address", "_check_address", "AddressMapping", "NandConfig"))
_FOOTPRINT_EXECUTION_MAP = _owned_execution_module.map_nand_address
_FOOTPRINT_CANONICAL = ResourceTimeline.canonical_resource_id


def _footprint_cache_key(core):
    from .physical_contract import _physical_content_key
    if (type(core) is not _FOOTPRINT_CORE_TYPE or type(core.config) is not _FOOTPRINT_CONFIG_TYPE
            or type(core.timeline) is not ResourceTimeline
            or _owned_execution_module._geometry_plan is not _OWNED_NAND_GEOMETRY_PLAN
            or _owned_execution_module.map_nand_address is not _FOOTPRINT_EXECUTION_MAP
            or _owned_pack_module.native_namespace_matches is not _footprint_namespace_matches
            or _owned_pack_module.native_instance_methods_absent is not _footprint_methods_absent
            or getattr(core.timeline.canonical_resource_id, "__func__", None) is not _FOOTPRINT_CANONICAL):
        return None
    for cls, items in ((_FOOTPRINT_CONFIG_TYPE, _FOOTPRINT_CONFIG_ITEMS),
                       (_FOOTPRINT_MAPPING_TYPE, _FOOTPRINT_MAPPING_ITEMS),
                       (_FOOTPRINT_CORE_TYPE, _FOOTPRINT_CORE_ITEMS)):
        # deepcopy may add an empty slot-name cache; it changes no behavior.
        if not _footprint_namespace_matches(cls, items):
            return None
    if any(getattr(_footprint_mapping_module, name) is not value
           for name, value in _FOOTPRINT_MAPPING_FUNCTIONS):
        return None
    # Instance overrides can depend on mutable closure state, so an identity
    # key alone is insufficient: never cache such callbacks.
    if not _footprint_methods_absent(core, _FOOTPRINT_CORE_METHOD_NAMES):
        return None
    aliases = core.timeline.resource_aliases
    if type(aliases) is not dict or any(type(k) is not str or type(v) is not str for k, v in aliases.items()):
        return None
    return _physical_content_key((vars(core.config), aliases))

_OWNED_NAND_FUNCTIONS = tuple((name, globals()[name]) for name in (
    "execute_buffered_full_pages", "execute_read_batch", "execute_full_pages", "full_page_period",
    "_generic_resident_writers", "_buffered_layout_contract", "_native_numeric_qualified",
    "_native_sink_qualified"))
_OWNED_NAND_PACK_FUNCTIONS = tuple((name, getattr(_owned_pack_module, name)) for name in (
    "pack_single_lane", "pack_multi_lane", "publish_timeline", "validate_buffered_cache",
    "pack_buffered_cache", "publish_buffered_cache", "can_own_resource_aliases", "publish_active_timeline",
    "generic_request_resource_indexes", "pack_selected_single_lane", "generic_metrics_delta",
    "publish_generic_controller", "remember_generic_resources"))
_OWNED_NAND_NUMERIC_ENTRIES = tuple((name, getattr(_owned_numeric_module, name)) for name in (
    "run_generic_full_pages", "run_generic_read_batch", "run_generic_flush_batch",
    "read_batch_request_metrics",
    "read_batch_sink_totals"))
_OWNED_NAND_GEOMETRY_PLAN = _owned_execution_module._geometry_plan


def has_native_timeline_hooks():
    for name, original in _OWNED_NAND_FUNCTIONS:
        if globals()[name] is not original:
            return False
    for name, original in _OWNED_NAND_PACK_FUNCTIONS:
        if getattr(_owned_pack_module, name) is not original:
            return False
    return (all(getattr(_owned_numeric_module, name) is original for name, original in _OWNED_NAND_NUMERIC_ENTRIES)
            and _owned_execution_module._geometry_plan is _OWNED_NAND_GEOMETRY_PLAN)

# These actual constructor/admission/validation callbacks execute while a
# native READ core bills a private task sink (_native_sink_qualified).
from . import nand_core as _native_owner_module
from . import memory_types as _native_types_module
from . import _resource_ledger as _native_ledger_module
_NATIVE_REQUEST = _native_types_module.AccessRequest
_NATIVE_NUMERIC_TYPE = _owned_execution_module._NandNumericState
_NATIVE_CLASSES = tuple((cls, tuple((name, value) for name, value in vars(cls).items()
    if name != "__slotnames__")) for cls in (
        _native_types_module.AccessRequest, _native_types_module.TransactionResult,
        _native_owner_module._HostPopulation, _native_owner_module._HostAdmission,
        _native_owner_module._HostCompletions, _native_owner_module._NandTransaction,
        _native_owner_module._ReadCacheSlot, _owned_execution_module._NandNumericState))
_NATIVE_BINDINGS = (
    (_owned_execution_module, "_NandNumericState", _NATIVE_NUMERIC_TYPE),
    *tuple((_native_ledger_module, name, getattr(_native_ledger_module, name)) for name in (
        "remember_owned_timeline_resources", "_get", "_put", "_remember_owned_resource")
        if hasattr(_native_ledger_module, name)),
    *tuple((_owned_pack_module, name, getattr(_owned_pack_module, name)) for name in (
        "can_own_timeline", "_native_timeline_columns", "_can_own_native_columns",
        "_get", "_put", "_item", "_double") if hasattr(_owned_pack_module, name)),
    *tuple((_native_owner_module, name, getattr(_native_owner_module, name)) for name in (
        "validate_nand_request", "map_nand_address", "replace", "_generic_bulk_span", "_bulk_min_pages",
        "_HostPopulation", "_HostAdmission", "_HostCompletions", "_NandTransaction", "_ReadCacheSlot",
        "heappop", "heappush", "heapify")),
    *tuple((_footprint_mapping_module, name, getattr(_footprint_mapping_module, name)) for name in (
        "validate_nand_request", "_check_address")),
    *tuple((_native_types_module, name, getattr(_native_types_module, name)) for name in (
        "AccessRequest", "TransactionResult", "_positive_int", "_nonnegative")),
)
