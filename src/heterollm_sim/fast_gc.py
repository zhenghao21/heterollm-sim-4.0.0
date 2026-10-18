"""Read-only, conditional maintenance bounds for an exhausted HBF free pool.

Closed-victim reservoirs bound the scalar FTL's live copies. Moving a live
page also needs an independently proved cold-read domain; its placement
uncertainty is two-sided, unlike the added maintenance IO. These are serial
service bounds, not predictions of critical-path service, and never divide
by banks.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import math

from .hbf_staging import HbfStagingController, PAGE, _EMPTY_GC_METHODS
from .memory_types import DramConfig, NandConfig, Operation
from .nand_core import NandCore

_NAND_CACHE_METHODS = {name: getattr(NandCore, name) for name in (
    "_invalidate_cache", "_cache_slot", "_submit_ocp_single_page_summary", "_submit_ocp_page_commands")}


def _same_physical_config(bound, authored):
    # PhysicalRuntimeContext adds owner resource names to metadata. All
    # authored values and every actual geometry/timing field must agree.
    return (type(bound) is type(authored)
        and all(getattr(bound, field.name) == getattr(authored, field.name)
                for field in fields(authored) if field.name != "metadata")
        and all(bound.metadata.get(key) == value for key, value in authored.metadata.items()))


def qualify_immutable_reads(physical, controller, owner, rows):
    """A resident annotation alone does not prove an immutable loaded page."""
    from .memory_allocator import AllocationError, PhysicalAddressAllocator
    result = {"qualified": False, "canonical_read_range_count": 0,
              "physical_state_advanced": False, "reason": None}
    allocator = physical.allocators.get(owner)
    if type(allocator) is not PhysicalAddressAllocator:
        result["reason"] = "gc_resident_read_allocator_unqualified"
        return result
    canonical = set()
    try:
        for buffer_id, generation, offset, size in rows:
            root, generation, relative = allocator.canonical_range(buffer_id, offset, size, generation)
            preload = controller._preloads.get((root, generation))
            if preload is None or relative < 0 or relative + size > preload.byte_count:
                raise ValueError("gc_resident_read_not_fully_preloaded")
            canonical.add((root, generation, relative, size))
    except (AllocationError, KeyError, TypeError, ValueError) as error:
        result["reason"] = str(error)
        return result
    result.update(qualified=True, canonical_read_range_count=len(canonical),
        method="read_only_allocator_alias_resolution_against_immutable_controller_preloads")
    return result


def _finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non_numeric_service_or_clock")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("non_finite_service_or_clock")
    return value


def _route_page_cost(routes, name):
    hops = routes.get(name)
    if not isinstance(hops, (tuple, list)) or not hops:
        raise ValueError("gc_route_unavailable:" + name)
    result = 0.
    for hop in hops:
        if (not isinstance(hop, Mapping)
                or not all(isinstance(hop.get(key), str) and hop[key]
                           for key in ("link_id", "resource_id"))
                or type(hop.get("queue_depth", 1)) is not int
                or hop.get("queue_depth", 1) <= 0):
            raise ValueError("gc_route_format_unqualified:" + name)
        # Charge both send and in-flight demands serially. They normally
        # overlap, but resource aliases can combine them on one calendar.
        # This sum remains conservative for either physical composition.
        result += 2 * PAGE / _finite(hop.get("bandwidth_gb_s"), positive=True)
        result += _finite(hop.get("latency_ns"))
    return _finite(result)


def _host_page_cost(config, operation):
    """Bound an analytical DRAM request with every burst serialized.

    Every burst may conflict, change bus direction and hit every group gate.
    Charging all gates, including the full four-ACT window, bounds the
    reduced controller independent of row placement. JEDEC/refresh controls
    require a different proof and are deliberately outside this domain.
    """
    if (type(config) is not DramConfig or config.control_mode != "analytical"
            or config.refresh_policy != "disabled" or config.scheduler != "fcfs"):
        raise ValueError("gc_host_controller_unqualified")
    timing = ("close_ns", "open_ns", "burst_interval_ns",
        "read_latency_ns" if operation is Operation.READ else "write_latency_ns",
        "read_recovery_ns", "write_recovery_ns", "read_to_write_ns", "write_to_read_ns",
        "column_same_group_ns", "column_different_group_ns",
        "activate_same_group_ns", "activate_different_group_ns",
        "four_activate_window_ns", "write_to_read_same_group_ns",
        "write_to_read_different_group_ns")
    per_burst = math.fsum(_finite(getattr(config, field)) for field in timing)
    per_burst += config.burst_bytes / _finite(
        config.directional_lane_bandwidth_gb_s(operation), positive=True)
    # The scratch and slot pages are aligned, but retain an extra boundary
    # burst to cover unusual legal HOST base alignment too.
    bursts = (PAGE + 2 * (config.burst_bytes - 1)) // config.burst_bytes
    return _finite(bursts * per_burst)


def _calendar_carry_in(physical, controller, cutoff_ns):
    """Inspect authoritative Python/native clocks without publishing them."""
    from .memory_transfer import ResourceTimeline
    from .nand_core import NandCore
    from .dram_core import DramCore

    cutoff = _finite(cutoff_ns)
    if getattr(physical, "_hbf_host_task", None) is not None:
        raise ValueError("gc_deferred_host_scope_not_closed")
    maxima = [cutoff, _finite(controller._scratch_ready)]
    maxima.extend(_finite(value) for value in controller._block_ready.values())
    maxima.extend(_finite(value) for value in controller._slot_ready.values())
    maxima.extend(_finite(controller._pages[key].ready_ns) for key in controller._resident)
    timelines = [physical.timeline]
    for active in physical.runtimes.values():
        core = active.core
        if type(core) not in (NandCore, DramCore):
            raise ValueError("gc_calendar_core_unqualified")
        data = core.__dict__
        maxima.append(_finite(active.clock_ns))
        timelines.append(core.timeline)
        for field in ("_acceptance_ns", "_media_drain_completion_ns"):
            if field in data:
                maxima.append(_finite(data[field]))
        for field in ("_array_ready", "_buffer_ready", "_host_acceptance_ns"):
            maxima.extend(_finite(value) for value in data.get(field, {}).values())
        maxima.extend(_finite(value) for value in data.get("_inflight", ()))
        for population in data.get("_host_inflight", {}).values():
            maxima.extend(_finite(row[0]) for row in population)
        for population in data.get("_channel_inflight", {}).values():
            maxima.extend(_finite(value) for value in population)
        for bank in data.get("_banks", {}).values():
            maxima.extend((_finite(bank.ready_ns), _finite(bank.activation_ready_ns)))
        packed = data.get("_nand_numeric_state")
        if packed is not None:
            # [0,1] array/buffer frontiers, [3,4] cache ready/readers.
            # Reading these ndarrays leaves numeric ownership unchanged.
            for index in (0, 1, 3, 4):
                values = packed.arrays[index]
                if values.size:
                    import numpy as np
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError("gc_numeric_calendar_unqualified")
                    maxima.append(float(values.max()))
        for slots in data.get("_read_cache", {}).values():
            for slot in slots:
                maxima.extend((_finite(slot.ready_ns), _finite(slot.reader_completion_ns)))
        if data.get("_hbm_numeric_state") is not None:
            raise ValueError("gc_unqualified_host_numeric_controller")
    seen = set()
    for timeline in timelines:
        if id(timeline) in seen:
            continue
        seen.add(id(timeline))
        if type(timeline) is not ResourceTimeline:
            raise ValueError("gc_calendar_timeline_unqualified")
        maxima.extend(_finite(value) for value in timeline.ready_ns.values())
        for lanes in timeline.lane_available.values():
            maxima.extend(_finite(value) for value in lanes)
    return _finite(max(maxima) - cutoff)


def gc_service_bounds(physical, controller, binding, cutoff_ns):
    """Static complete-page costs and already outstanding cleanup service."""
    config = controller.config
    variants = tuple(binding) if isinstance(binding, (tuple, list)) else (binding,)
    if (type(config) is not NandConfig or config.protocol_mode != "ocp_hbf_0_7"
            or config.page_bytes != PAGE or config.write_buffer_bytes
            or config.wear_model is not None or not variants
            or any(not isinstance(item, Mapping) for item in variants)):
        raise ValueError("gc_physical_service_contract_unqualified")
    binding = variants[0]
    owners = [owner for owner, state in physical.hbf_staging.items()
              if state.get("controller") is controller]
    if len(owners) != 1:
        raise ValueError("gc_controller_physical_owner_unqualified")
    nand = physical.runtimes.get(owners[0])
    if (nand is None or type(nand.core) is not NandCore
            or not _same_physical_config(nand.core.config, config)
            or any(name in nand.core.__dict__ or getattr(NandCore, name) is not method
                   for name, method in _NAND_CACHE_METHODS.items())):
        raise ValueError("gc_nand_cache_contract_unqualified")
    host = binding.get("host_config")
    if isinstance(host, Mapping):
        from .physical_contract import parse_execution_physical_config
        host = parse_execution_physical_config(host)
    host_owner = binding.get("host_owner")
    active_host = physical.runtimes.get(host_owner)
    if (active_host is None or type(host) is not DramConfig or host.kind.value == "HBM"
            or not _same_physical_config(active_host.core.config, host)):
        raise ValueError("gc_host_binding_unqualified")
    route_variants = []
    for variant in variants:
        variant_host = variant.get("host_config")
        if isinstance(variant_host, Mapping):
            from .physical_contract import parse_execution_physical_config
            variant_host = parse_execution_physical_config(variant_host)
        if (variant.get("host_owner") != host_owner or variant_host != host
                or variant.get("capacity_bytes") != controller.host_capacity_bytes):
            raise ValueError("gc_host_binding_variants_unqualified")
        routes = variant.get("routes")
        if not isinstance(routes, Mapping):
            raise ValueError("gc_routes_unavailable")
        route_variants.append({name: _route_page_cost(routes, name)
            for name in ("from_hbf", "to_hbf", "write_hbf", "read_hbf")})
    route = {name: max(variant[name] for variant in route_variants)
             for name in route_variants[0]}
    read = (config.front_ns + config.page_read_ns
        + config.transfer_page_bytes / config.internal_bandwidth_gb_s
        + PAGE * config.host_channels / config.host_bandwidth_for(Operation.READ))
    write = (config.front_ns + config.page_program_ns
        + config.transfer_page_bytes / config.internal_bandwidth_gb_s
        + PAGE * config.host_channels / config.host_bandwidth_for(Operation.WRITE))
    host_read = _host_page_cost(host, Operation.READ)
    host_write = _host_page_cost(host, Operation.WRITE)
    cache_capacity = config.read_cache_pages_per_bank
    # Delete temporary reads from the reference cache trace. Inserting one
    # access into a k-slot LRU changes at most the next k distinct-reference
    # misses, plus its own miss; after those references the LRU states agree.
    # Charging the full service also covers any cache-slot reader lifetime.
    read_placement = (cache_capacity + 1) * read + route["read_hbf"]
    cache_initial = cache_capacity * config.array_units * read
    copy = math.fsum(((cache_capacity + 1) * read, route["from_hbf"], host_write, host_read,
                     route["to_hbf"], write))
    # Zero-copy GC leaves HOST fragment/slot allocation unchanged. Charge
    # the entire placement-dependent ordinary NAND program and either its
    # direct initiator route or the existing HOST flush route. This permits
    # all their overlap to disappear; no bank-parallelism factor is used.
    ordinary = write + max(route["to_hbf"], route["write_hbf"])
    return {"per_unit_costs": {
        "gc_copy_page_serial_upper_ns": _finite(copy),
        "ordinary_program_page_serial_upper_ns": _finite(ordinary),
        "ordinary_read_page_serial_upper_ns": _finite(read_placement),
        "read_cache_initial_state_serial_upper_ns": _finite(cache_initial),
        "nand_read_page_serial_upper_ns": _finite(read),
        "kv_read_remap_page_serial_upper_ns": _finite(read + route["read_hbf"]),
        "persistent_cache_miss_page_serial_upper_ns": _finite(read),
        "nand_program_page_serial_upper_ns": _finite(write),
        "host_read_page_serial_upper_ns": host_read,
        "host_write_page_serial_upper_ns": host_write,
        "link_from_hbf_page_serial_upper_ns": route["from_hbf"],
        "link_to_hbf_page_serial_upper_ns": route["to_hbf"],
        "link_write_hbf_page_serial_upper_ns": route["write_hbf"],
        "link_read_hbf_page_serial_upper_ns": route["read_hbf"],
        "read_cache_lru_slots_per_bank": cache_capacity,
        "temporary_read_cache_miss_multiplier": cache_capacity + 1,
        "erase_included_in_page_costs": False,
        "semantics": "all_page_stages_serialized_no_bank_or_queue_divisor"},
        "carry_in_serial_guard_ns": _calendar_carry_in(physical, controller, cutoff_ns),
        "carry_in_cutoff_ns": cutoff_ns,
        "carry_in_clock_semantics": "absolute_live_token_commit_timestamp",
        "qualified_initiator_route_variants": len(variants),
        "read_cache_mapping_proof": "scalar_and_numeric_OCP_writes_invalidate_only_matching_page_or_target_block_tags_no_whole_bank_invalidation",
        "persistent_write_invalidation_extra_miss_upper": 0,
        "persistent_write_invalidation_zero_basis": "gc_target_is_spare_before_first_program_later_programs_append_monotonic_kv_does_not_overwrite_backed_pages_and_warm_weights_are_immutable",
        "temporary_read_cache_bound_method": "persistent_only_reference_trace_plus_each_temporary_access_lru_insertion_k_plus_1_and_initial_k_distinct_misses_per_bank"}


def project_gc_maintenance(controller, *, program_pages, free_program_pages,
                         geometry_qualified, temporary_read_pages_per_decode=None,
                         remaining_decode_steps=None, service_bounds=None,
                         ordinary_program_prefix_upper_by_decode=None,
                         gc_read_qualification=None):
    """Bound a tail using a sufficient reservoir of closed low-live blocks.

    Each collection consumes at most one initial low-live victim. Closed
    blocks cannot be written without being selected as a victim; release
    can only increase this reservoir. Active blocks are excluded because
    the initial allocation prefix can still populate them.
    """
    result = {"required": True, "qualified": False, "reason": None,
        "method": "closed_empty_victim_reservoir_with_serial_service_bounds/v1",
        "physical_state_advanced": False, "physical_io_executed": False,
        "copied_pages_upper": None, "zero_copy_proven": False,
        "live_copy_proven": False, "live_projection_qualified": False,
        "future_gc_read_remap_symmetric_guard_ns": None,
        "future_gc_persistent_cache_symmetric_guard_ns": None}
    if (type(program_pages) is not int or program_pages < 0
            or type(free_program_pages) is not int or free_program_pages < 0):
        result["reason"] = "gc_program_budget_unqualified"
        return result
    if (type(controller) is not HbfStagingController or any(
            name in controller.__dict__ or getattr(HbfStagingController, name) is not method
            for name, method in _EMPTY_GC_METHODS.items())):
        result["reason"] = "gc_custom_allocator_or_controller_unqualified"
        return result
    if not geometry_qualified:
        result["reason"] = "gc_fixed_write_geometry_unqualified"
        return result
    if (type(temporary_read_pages_per_decode) is not int or temporary_read_pages_per_decode < 0
            or type(remaining_decode_steps) is not int or remaining_decode_steps < 0):
        result["reason"] = "gc_temporary_read_budget_unqualified"
        return result
    active = set(controller._active)
    actual_headroom = len(controller._free_blocks) * controller.config.pages_per_block + sum(
        controller.config.pages_per_block - controller._used[block] for block in active)
    if actual_headroom != free_program_pages:
        result["reason"] = "gc_program_headroom_does_not_match_live_controller"
        return result
    immutable = {block for preload in controller._preloads.values() for block in preload.blocks()}
    per_block = controller.config.pages_per_block
    histogram = [0] * per_block
    for block in controller._used:
        if block in active or block == controller._spare or block in immutable:
            continue
        live = len(controller._owners.get(block, ()))
        if live < per_block:
            histogram[live] += 1
    closed = histogram[0]
    deficit = max(0, program_pages - free_program_pages)
    cycles = (deficit + per_block - 1) // per_block
    reservoir_by_q, required_by_q, pool = [], [], 0
    for candidate, count in enumerate(histogram):
        pool += count
        reservoir_by_q.append(pool)
        required_by_q.append((deficit + per_block - candidate - 1) // (per_block - candidate))
    result.update(empty_victim_reservoir=closed, cycles_upper=None,
        zero_copy_cycles_required=cycles,
        closed_victim_live_histogram=tuple(histogram),
        closed_victim_reservoir_by_live_page_upper=tuple(reservoir_by_q),
        gc_cycles_required_by_live_page_upper=tuple(required_by_q),
        reservoir_excludes_active=True, reservoir_excludes_spare=True,
        reservoir_excludes_immutable_preloads=True,
        ordinary_program_page_upper=program_pages,
        initial_program_headroom_pages=free_program_pages,
        pages_per_collected_block=per_block)
    q, pool = None, 0
    for candidate, (pool, required) in enumerate(zip(reservoir_by_q, required_by_q)):
        if pool >= required:
            q, cycles = candidate, required
            break
    if q is None:
        result["reason"] = "live_gc_copy_remap_bound_unavailable"
        # A useful diagnostic only: no live-copy domain is accepted without
        # a bound on the resulting ordinary read/cache remapping effects.
        count = len(controller._used)
        q = (len(controller._pages) + program_pages) // count if count else per_block
        result["diagnostic_live_copy_pages_per_cycle_upper"] = min(per_block - 1, q)
        result["diagnostic_live_copy_bound_used_for_acceptance"] = False
        result["future_gc_read_remap_symmetric_guard_ns"] = None
        result["future_gc_persistent_cache_symmetric_guard_ns"] = None
        return result
    result.update(closed_victim_live_page_upper=q,
        closed_victim_reservoir=pool, closed_victim_live_histogram=tuple(histogram),
        cycles_upper=cycles, zero_copy_proven=q == 0, copied_pages_upper=q * cycles,
        existing_persistent_backing_mapping_unchanged=q == 0,
        new_kv_read_growth_semantics="existing_nonnegative_affine_context_envelope_not_exact_future_mapping",
        future_gc_target_page_zero_erases_upper=cycles,
        future_gc_erase_serial_ns=cycles * controller.config.block_erase_ns,
        future_gc_non_erase_serial_ns=0. if q == 0 else None)
    if q:
        result["method"] = "closed_low_live_victim_reservoir_with_cold_kv_read_bounds/v1"
        result["gc_read_qualification"] = (dict(gc_read_qualification)
            if isinstance(gc_read_qualification, Mapping) else gc_read_qualification)
        if (not isinstance(gc_read_qualification, Mapping)
                or gc_read_qualification.get("qualified") is not True
                or gc_read_qualification.get("physical_state_advanced") is not False):
            result["reason"] = "live_gc_cold_read_qualification_unavailable"
            result["future_gc_read_remap_symmetric_guard_ns"] = None
            result["future_gc_persistent_cache_symmetric_guard_ns"] = None
            return result
        names = ("kv_reads_per_page_per_decode_upper", "immutable_repeat_read_pages_per_decode_upper",
                 "initial_warm_read_pages_per_decode_upper")
        if (any(type(gc_read_qualification.get(name)) is not int
                or gc_read_qualification[name] < 0 for name in names)
                or gc_read_qualification[names[0]] <= 0):
            result["reason"] = "live_gc_cold_read_counts_unqualified"
            result["future_gc_read_remap_symmetric_guard_ns"] = None
            result["future_gc_persistent_cache_symmetric_guard_ns"] = None
            return result
        prefix = ordinary_program_prefix_upper_by_decode
        if (not isinstance(prefix, (tuple, list)) or len(prefix) != remaining_decode_steps
                or any(type(value) is not int or value < 0 for value in prefix)
                or any(left > right for left, right in zip(prefix, prefix[1:]))
                or (prefix[-1] if prefix else 0) != program_pages):
            result["reason"] = "live_gc_cumulative_program_bound_unqualified"
            result["future_gc_read_remap_symmetric_guard_ns"] = None
            result["future_gc_persistent_cache_symmetric_guard_ns"] = None
            return result
        gc_prefix = tuple((max(0, value - free_program_pages) + per_block - q - 1)
                          // (per_block - q) for value in prefix)
        result.update(gc_read_qualification=dict(gc_read_qualification),
            **{name: gc_read_qualification[name] for name in names},
            ordinary_program_prefix_upper_by_decode=tuple(prefix),
            gc_cycles_prefix_upper_by_decode=gc_prefix,
            future_gc_migrated_kv_page_token_upper=q * sum(gc_prefix),
            future_gc_read_remap_reference="cutoff_mapping_with_time_weighted_copy_count",
            live_copy_proven=True)
    else:
        result.update(copy_zero_basis="sufficient_closed_empty_victim_reservoir_not_an_unobserved_zero",
            future_gc_read_remap_symmetric_guard_ns=0.,
            future_gc_persistent_cache_symmetric_guard_ns=0.,
            gc_copy_nand_read_serial_upper_ns=0., gc_copy_nand_write_serial_upper_ns=0.,
            gc_copy_host_read_serial_upper_ns=0., gc_copy_host_write_serial_upper_ns=0.,
            gc_copy_link_from_hbf_serial_upper_ns=0., gc_copy_link_to_hbf_serial_upper_ns=0.)
    if (not isinstance(service_bounds, Mapping)
            or not isinstance(service_bounds.get("per_unit_costs"), Mapping)):
        result["reason"] = "gc_complete_service_bounds_unavailable"
        return result
    result.update(service_bounds)
    units = service_bounds["per_unit_costs"]
    required_units = ("ordinary_program_page_serial_upper_ns",
        "ordinary_read_page_serial_upper_ns", "read_cache_initial_state_serial_upper_ns")
    try:
        _finite(service_bounds.get("carry_in_serial_guard_ns"))
        for name in required_units:
            _finite(units.get(name))
    except (TypeError, ValueError) as error:
        result["reason"] = "gc_complete_service_bounds_unavailable:" + str(error)
        return result
    if q:
        required_units = ("gc_copy_page_serial_upper_ns", "nand_read_page_serial_upper_ns",
            "nand_program_page_serial_upper_ns", "host_read_page_serial_upper_ns",
            "host_write_page_serial_upper_ns", "link_from_hbf_page_serial_upper_ns",
            "link_to_hbf_page_serial_upper_ns", "read_cache_lru_slots_per_bank",
            "kv_read_remap_page_serial_upper_ns", "persistent_cache_miss_page_serial_upper_ns")
        try:
            for name in required_units:
                _finite(units.get(name))
        except (TypeError, ValueError) as error:
            result.update(reason="live_gc_complete_service_bounds_unavailable:" + str(error),
                future_gc_read_remap_symmetric_guard_ns=None,
                future_gc_persistent_cache_symmetric_guard_ns=None)
            return result
        copies = result["copied_pages_upper"]
        result["future_gc_non_erase_serial_ns"] = _finite(copies * units["gc_copy_page_serial_upper_ns"])
        for output, unit in (("gc_copy_nand_read_serial_upper_ns", "nand_read_page_serial_upper_ns"),
                ("gc_copy_nand_write_serial_upper_ns", "nand_program_page_serial_upper_ns"),
                ("gc_copy_host_read_serial_upper_ns", "host_read_page_serial_upper_ns"),
                ("gc_copy_host_write_serial_upper_ns", "host_write_page_serial_upper_ns"),
                ("gc_copy_link_from_hbf_serial_upper_ns", "link_from_hbf_page_serial_upper_ns"),
                ("gc_copy_link_to_hbf_serial_upper_ns", "link_to_hbf_page_serial_upper_ns")):
            result[output] = _finite(copies * units[unit])
        result["gc_copy_cache_pollution_serial_upper_ns"] = _finite(copies
            * units["read_cache_lru_slots_per_bank"] * units["nand_read_page_serial_upper_ns"])
        result["future_gc_read_remap_symmetric_guard_ns"] = _finite(
            result["future_gc_migrated_kv_page_token_upper"]
            * result["kv_reads_per_page_per_decode_upper"]
            * units["kv_read_remap_page_serial_upper_ns"])
        result["future_gc_persistent_cache_symmetric_guard_ns"] = _finite(remaining_decode_steps
            * (result["immutable_repeat_read_pages_per_decode_upper"]
               + result["initial_warm_read_pages_per_decode_upper"])
            * units["persistent_cache_miss_page_serial_upper_ns"])
    per_page = service_bounds["per_unit_costs"]["ordinary_program_page_serial_upper_ns"]
    result["future_write_placement_serial_guard_ns"] = _finite(program_pages * per_page)
    read_unit = service_bounds["per_unit_costs"]["ordinary_read_page_serial_upper_ns"]
    result["temporary_read_pages_per_decode_upper"] = temporary_read_pages_per_decode
    result["future_temporary_read_pages_upper"] = temporary_read_pages_per_decode * remaining_decode_steps
    result["future_read_placement_serial_guard_ns"] = _finite(
        result["future_temporary_read_pages_upper"] * read_unit
        + service_bounds["per_unit_costs"]["read_cache_initial_state_serial_upper_ns"])
    result["qualified"] = True
    result["live_projection_qualified"] = q > 0
    return result
