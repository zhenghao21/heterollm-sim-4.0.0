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


def execute_buffered_full_pages(core, request, page_count, *, _metrics=None, _resource_sink=None):
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
    from ._nand_pack import (pack_single_lane, pack_multi_lane, publish_timeline,
                            validate_buffered_cache, pack_buffered_cache, publish_buffered_cache,
                            publish_active_timeline, generic_request_resource_indexes,
                            pack_selected_single_lane, generic_metrics_delta,
                            publish_generic_controller, defer_active_timeline,
                            generic_deferred_metrics_delta)
    from ._compiled_nand import run_generic_full_pages

    config, timeline = core.config, core.timeline
    if (type(core) is not NandCore or type(timeline) is not ResourceTimeline
            or core.capture_details or core._observe_media or core._staged_pages
            or config.protocol_mode != "generic_nand" or config.queue_scope != "device"
            or config.page_bytes != 4096 or config.read_cache_pages_per_bank != 2
            or 4096 % config.host_granularity_bytes
            or request.operation not in (Operation.READ, Operation.WRITE)
            or request.operation is Operation.WRITE and (request.address % 4096 or request.byte_count % 4096)
            and config.partial_page_policy != "read_modify_write"
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
    scope = core.__dict__.get("_nand_request_metric_orders") if _metrics is not None else None
    shared_pack = scope is not None and "pack_pages" in scope
    calendar = scope.get("calendar") if shared_pack else None
    packed = None
    if calendar is not None:
        packed, single_lane = calendar
    elif shared_pack:
        # A submit_many scope has no external callbacks. Pack the complete
        # immutable resource layout once; later requests execute against the
        # same live arrays that were just published to the public calendars.
        packed = pack_single_lane(timeline, names, scope["pack_pages"], config.transfer_page_bytes)
    elif (bus_mask is not None and core._nand_transactions
            and core._nand_transactions[-1].native_timeline_qualified):
        selected = generic_request_resource_indexes(request.address // 4096, page_count,
            base_units, config.planes_per_lun, config.planes_independent, per_channel,
            config.host_channels, array_ids, channel_ids, host_ids)
        packed = pack_selected_single_lane(timeline, names, selected, page_count, config.transfer_page_bytes)
    if packed is None:
        packed = pack_single_lane(timeline, names, page_count, config.transfer_page_bytes)
    if calendar is None:
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
    if shared_pack and calendar is None:
        scope["calendar"] = ((start_values, lane_counts, lanes, lane_directions, state, moved, direction_values), single_lane)
    if shared_pack:
        _start_deferred_calendar(core, scope, names, array_names, array_resources, bus_mask,
            single_lane, (start_values, lane_counts, lanes, lane_directions, state, moved, direction_values))
    deferred = scope.get("deferred") if shared_pack else None
    host_bandwidth = core._host_bandwidth(request.operation)
    timing = np.array((config.page_read_ns, config.page_program_ns,
                       config.transfer_page_bytes / config.internal_bandwidth_gb_s,
                       4096 / host_bandwidth, config.transfer_page_bytes))
    # split_nand_request's host span of the first and last page.
    granularity, end = config.host_granularity_bytes, request.address + request.byte_count
    head_end = min(end, (request.address // 4096 + 1) * 4096)
    head_bytes = -(-head_end // granularity) * granularity - request.address // granularity * granularity
    tail_start = max(request.address, (end - 1) // 4096 * 4096)
    tail_bytes = -(-end // granularity) * granularity - tail_start // granularity * granularity
    host_bytes = head_bytes if page_count == 1 else head_bytes + tail_bytes + (page_count - 2) * 4096
    # A partially written boundary page pays the scalar read-modify-write.
    write = request.operation is Operation.WRITE
    head_rmw, tail_rmw = write and head_end - request.address != 4096, write and end - tail_start != 4096
    geometry = np.array((base_units, config.planes_per_lun, config.planes_independent,
                         per_channel, config.host_channels), dtype=np.int64)
    before = packed.checkpoint() if packed is core._nand_numeric_state and packed is not None else None
    if packed is not None:
        for token in core._nand_transactions:
            token.capture_numeric(packed)
    calendar_before = (tuple(value.copy() for value in deferred["calendar"][2:])
                       if deferred is not None else None)
    if deferred is not None:
        deferred["request_checkpoint"] = (deferred["publication_serial"], calendar_before)
    try:
        completion, hits, misses, rmw_reads, clock, touched, active_units, active_banks, resource_order = run_generic_full_pages(
            request.address // 4096, page_count, request.operation is Operation.READ,
            request.arrival_ns + config.front_ns, geometry, timing, array_ids, channel_ids, host_ids,
            arrays, buffers, tags, ready, readers, uses, present, core._cache_use,
            start_values, lane_counts, lanes, lane_directions, state, moved, direction_values,
            head_bytes, head_bytes / host_bandwidth, tail_bytes, tail_bytes / host_bandwidth,
            head_rmw, tail_rmw)
        if not np.all(np.isfinite(state)) or not isfinite(completion):
            raise ValueError("resource reservation exceeds finite timing range")
    except BaseException:
        if before is not None:
            packed.restore(before)
        if deferred is not None:
            _restore_deferred_calendar(deferred, calendar_before)
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
    if deferred is not None:
        publication_serial = deferred["publication_serial"]
        try:
            defer_active_timeline(timeline, names, state, resource_indexes, resources, scope)
        except BaseException:
            # Numeric/controller state is already successful at this stage;
            # only calendars still expose the preceding published prefix.
            if deferred["publication_serial"] == publication_serial:
                _restore_deferred_calendar(deferred, calendar_before)
            raise
    elif bus_mask is not None:
        publish_active_timeline(timeline, names, state, moved, direction_values,
            lanes, lane_directions, start_values, lane_counts, resource_indexes,
            resources, bus_mask, directions, single_lane, _metrics is not None,
            scope if shared_pack else None)
    else:
        publish_timeline(timeline, names, state.tolist(), moved.tolist(), direction_values.tolist(),
                         lanes.tolist(), lane_directions.tolist(), start_values.tolist(), lane_counts.tolist(),
                         resources, (), [int(name in resources and name in channels + hosts) for name in names],
                         directions, single_lane)
    # A qualified READ miss stores the same end in arrays[unit] and the
    # single-lane array calendar. That calendar can publish its ready value
    # once at the existing batch/public boundary. Resident controller maps
    # already contain every key, so this publisher would only repeat ready.
    if deferred is None:
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
        if deferred is None:
            _metrics.update(generic_metrics_delta(timeline, names, resource_indexes, before_busy, before_moved,
                core.__dict__.get("_nand_request_metric_orders")))
        else:
            if _resource_sink is None:
                _metrics.update(generic_deferred_metrics_delta(timeline, names, state, moved, touched,
                    before_busy, before_moved, deferred))
            else:
                _metrics.update(generic_deferred_metrics_delta(timeline, names, state, moved, touched,
                    before_busy, before_moved, deferred, _resource_sink=_resource_sink))
    read = request.operation is Operation.READ
    programmed = 0 if read else page_count
    pages_read = int(misses) + int(rmw_reads)
    internal_bytes = (pages_read + programmed) * config.transfer_page_bytes
    return TransactionResult(request.request_id, request.operation, request.arrival_ns, float(completion),
        request.byte_count, internal_bytes, counters={
            "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
            "physical_read_bytes": pages_read * 4096, "physical_write_bytes": programmed * 4096,
            "pages_read": pages_read, "pages_programmed": programmed,
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




def _deferred_numeric_qualified(core):
    """Prove only descriptors and buffers read while calendars are private."""
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
            or type(contract[10]) is not tuple or len(contract[10]) != 4
            or any(type(item) is not str for item in contract[10])
            or type(contract[11]) is not tuple
            or any(type(item) is not tuple or len(item) != 2
                   or any(type(part) is not str for part in item) for item in contract[11])):
        return False
    return contract == core._resident_resource_contract()


def _deferred_batch_qualified(core, requests, *, _core_only=False):
    """Only closed, already resident native READ batches may hide calendars."""
    from . import nand_core as owner
    from .memory_transfer import _OWNED_TIMELINE_TOKEN_TYPE, _OWNED_TIMELINE_TOKEN_METHODS
    if ((not _core_only and len(requests) < 8)
            or type(core._nand_numeric_state) is not _DEFER_NUMERIC_TYPE
            or type(core.config) is not _FOOTPRINT_CONFIG_TYPE
            or not _footprint_namespace_matches(_FOOTPRINT_CORE_TYPE, _FOOTPRINT_CORE_ITEMS)
            or any(type(getattr(core, name)) is not bool for name in (
                "_nand_numeric_enabled", "_nand_state_exposed", "capture_details",
                "_observe_media", "_buffered_write_mode"))
            or type(core._staged_pages) is not dict
            or core.config.parallel_units is not None
            or not _footprint_namespace_matches(_FOOTPRINT_CONFIG_TYPE, _FOOTPRINT_CONFIG_ITEMS)
            or any(not _footprint_namespace_matches(cls, items) for cls, items in _DEFER_CLASSES)
            or any(getattr(module, name) is not value for module, name, value in _DEFER_BINDINGS)):
        return False
    if any(type(key) is not str for key in core.config.metadata):
        return False
    for name, value in vars(core.config).items():
        if name in ("kind", "metadata"):
            continue
        if not (value is None or type(value) is str or type(value) is int
                or type(value) is float or type(value) is bool):
            return False
    if not _deferred_numeric_qualified(core):
        return False
    for request in requests:
        if (type(request) is not _DEFER_REQUEST or request.operation is not Operation.READ
                or type(request.request_id) is not str or type(request.address) is not int
                or type(request.byte_count) is not int
                or (type(request.arrival_ns) is not int and type(request.arrival_ns) is not float)):
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
        # Only missing native cache banks call remember_cache in this READ
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
        # persistent calendar journals are queried inside the private scope.
        for name in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals",
                     "lane_available", "directions", "lane_directions"):
            saved = token.entries.get(name)
            if saved is not None and any(type(key) is not str for key in saved):
                return False
    return True


def _deferred_calendar_eligible(core, names, array_names, array_resources, bus_mask, single_lane):
    """No new public key, aliased map or shared physical array in this scope."""
    import numpy as np
    from ._nand_pack import can_own_timeline
    timeline = core.timeline
    columns = tuple(getattr(timeline, name) for name in (
        "ready_ns", "busy_ns", "bytes_moved", "last_intervals",
        "lane_available", "directions", "lane_directions"))
    # Frontier membership executes inside the controller publisher. Authored
    # containers/keys can inspect every calendar there, so they must retain
    # immediate publication, not merely the original controller publisher.
    # These checks occur once when this submit_many scope first qualifies.
    frontiers = (core.__dict__["_array_ready"], core.__dict__["_buffer_ready"])
    cache = core.__dict__["_read_cache"]
    if (not single_lane or core.config.parallel_units is not None
            or tuple(array_names) != tuple(array_resources)
            or len(set(array_names)) != len(array_names)
            or any(type(column) is not dict for column in (*columns, *frontiers, cache))
            or len({id(column) for column in (*columns, *frontiers, cache)}) != len(columns) + 3
            or any(type(key) is not str for column in (*columns, *frontiers) for key in column)
            or any(type(key) is not tuple or any(type(part) is not int for part in key) for key in cache)
            or not can_own_timeline(timeline, tuple(names))
            or any(name in timeline.lane_directions for name in names)
            or len({id(value) for value in timeline.lane_available.values()}) != len(timeline.lane_available)):
        return False
    # All output-order keys are stable for the whole batch. A missing key
    # retains the existing publisher so later batches may qualify naturally.
    for index, name in enumerate(names):
        if (name not in timeline.ready_ns or name not in timeline.busy_ns
                or name not in timeline.last_intervals or name not in timeline.lane_available
                or len(timeline.lane_available[name]) != 1
                or bus_mask[index] and (name not in timeline.bytes_moved or name not in timeline.directions)):
            return False
    if any(name not in core.__dict__["_array_ready"]
           or name not in core.__dict__["_buffer_ready"] for name in array_names):
        return False
    # The current native pack has already validated scalar values/unique
    # one-element lanes. Controller arrays must not carry authored ndarray hooks.
    return all(type(array) is np.ndarray for array in core._nand_numeric_state.arrays)


def _start_deferred_calendar(core, scope, names, array_names, array_resources, bus_mask,
                             single_lane, calendar):
    import numpy as np
    if (scope is None or not scope.get("defer_candidate")
            or scope.get("defer_checked")):
        return
    scope["defer_checked"] = True
    if not _deferred_calendar_eligible(core, names, array_names, array_resources, bus_mask, single_lane):
        return
    lookup = {name: index for index, name in enumerate(names)}
    scope["deferred"] = {
        "names": names, "arrays": frozenset(array_names), "bus_mask": bus_mask,
        "calendar": calendar, "seen": np.zeros(len(names), dtype=np.bool_),
        "intervals": {name: core.timeline.last_intervals[name] for name in names},
        "publication_serial": 0, "controller_ready_complete": False,
        "busy_order": np.array([lookup[name] for name in core.timeline.busy_ns if name in lookup], dtype=np.int64),
        "moved_order": np.array([lookup[name] for name in core.timeline.bytes_moved if name in lookup], dtype=np.int64),
    }


def _restore_deferred_calendar(deferred, before):
    # The native kernel mutates shared calendars in place. Controller undo
    # does not include them. Preserve the previous public-success prefix.
    for target, saved in zip(deferred["calendar"][2:], before):
        target[:] = saved


def _flush_deferred_calendar(core):
    """Publish this batch's union without changing the current recorder."""
    import numpy as np
    from ._nand_pack import publish_active_timeline
    scope = core.__dict__.get("_nand_request_metric_orders")
    deferred = None if scope is None else scope.pop("deferred", None)
    if deferred is None:
        return
    scope["defer_candidate"] = False
    checkpoint = deferred.get("request_checkpoint")
    if checkpoint is not None and checkpoint[0] == deferred["publication_serial"]:
        _restore_deferred_calendar(deferred, checkpoint[1])
    indexes = np.flatnonzero(deferred["seen"])
    if not len(indexes):
        return
    starts, counts, lanes, lane_directions, state, moved, directions = deferred["calendar"]
    names = deferred["names"]
    resources = {names[index] for index in indexes.tolist()}
    # The qualified READ calendar also owns array ready until this boundary.
    # Its successful prefix supplies the same miss end as the controller.
    # Do not retain numeric-array pointers: restore() may replace them after
    # a failed request. Untouched/hit-only arrays are absent from the union.
    skip_ready = deferred["arrays"] if deferred["controller_ready_complete"] else ()
    publish_active_timeline(core.timeline, names, state, moved, directions,
        lanes, lane_directions, starts, counts, indexes, resources,
        deferred["bus_mask"], [None, "read", "write"], True,
        _remember=False, _last_intervals=deferred["intervals"], _skip_ready=skip_ready)

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
    "execute_buffered_full_pages", "execute_full_pages", "full_page_period", "_generic_resident_writers",
    "_buffered_layout_contract", "_deferred_numeric_qualified", "_deferred_batch_qualified", "_deferred_calendar_eligible",
    "_start_deferred_calendar", "_restore_deferred_calendar", "_flush_deferred_calendar"))
_OWNED_NAND_PACK_FUNCTIONS = tuple((name, getattr(_owned_pack_module, name)) for name in (
    "pack_single_lane", "pack_multi_lane", "publish_timeline", "validate_buffered_cache",
    "pack_buffered_cache", "publish_buffered_cache", "can_own_resource_aliases", "publish_active_timeline",
    "generic_request_resource_indexes", "pack_selected_single_lane", "generic_metrics_delta",
    "publish_generic_controller", "remember_generic_resources", "generic_ordered_metric_touches",
    "defer_active_timeline", "generic_deferred_metrics_delta"))
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

# These actual constructor/admission/validation callbacks execute inside a
# deferred batch. Public replacements keep ordinary immediate publication.
from . import nand_core as _defer_owner_module
from . import memory_types as _defer_types_module
from . import _resource_ledger as _defer_ledger_module
_DEFER_REQUEST = _defer_types_module.AccessRequest
_DEFER_NUMERIC_TYPE = _owned_execution_module._NandNumericState
_DEFER_CLASSES = tuple((cls, tuple((name, value) for name, value in vars(cls).items()
    if name != "__slotnames__")) for cls in (
        _defer_types_module.AccessRequest, _defer_types_module.TransactionResult,
        _defer_owner_module._HostPopulation, _defer_owner_module._HostAdmission,
        _defer_owner_module._HostCompletions, _defer_owner_module._NandTransaction,
        _defer_owner_module._ReadCacheSlot, _owned_execution_module._NandNumericState))
_DEFER_BINDINGS = (
    (_owned_execution_module, "_NandNumericState", _DEFER_NUMERIC_TYPE),
    *tuple((_defer_ledger_module, name, getattr(_defer_ledger_module, name)) for name in (
        "remember_owned_timeline_resources", "_get", "_put", "_remember_owned_resource")
        if hasattr(_defer_ledger_module, name)),
    *tuple((_owned_pack_module, name, getattr(_owned_pack_module, name)) for name in (
        "can_own_timeline", "_native_timeline_columns", "_can_own_native_columns",
        "_get", "_put", "_item", "_double") if hasattr(_owned_pack_module, name)),
    *tuple((_defer_owner_module, name, getattr(_defer_owner_module, name)) for name in (
        "validate_nand_request", "map_nand_address", "replace", "_generic_bulk_span", "_bulk_min_pages",
        "_HostPopulation", "_HostAdmission", "_HostCompletions", "_NandTransaction", "_ReadCacheSlot",
        "heappop", "heappush", "heapify")),
    *tuple((_footprint_mapping_module, name, getattr(_footprint_mapping_module, name)) for name in (
        "validate_nand_request", "_check_address")),
    *tuple((_defer_types_module, name, getattr(_defer_types_module, name)) for name in (
        "AccessRequest", "TransactionResult", "_positive_int", "_nonnegative")),
)
