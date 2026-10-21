"""Native IR facts for conditional continuous-window timing intervals.

No function in this module executes an IO or advances a controller. The
affine, fixed-topology continuation is a declared research model, not a
certificate of an unexecuted physical trace. Geometry-dependent phase costs
remain explicit additions to that model and can make it reject.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from functools import lru_cache
import math
from typing import Mapping

from .memory_types import DramConfig, NandConfig, SramConfig
from .dram_core import DramCore
from .nand_core import NandCore
from .sram_core import SramCore


_CORE_SUBMIT = {core_type: core_type.submit for core_type in (DramCore, NandCore, SramCore)}


def _native_cores_supported(physical):
    return all(type(active.core) in _CORE_SUBMIT
        and getattr(active.core.submit, "__func__", None) is _CORE_SUBMIT[type(active.core)]
        for active in physical.runtimes.values())


def _bandwidth(config, operation):
    if isinstance(config, (DramConfig, SramConfig)):
        return config.directional_bandwidth_gb_s(operation)
    if isinstance(config, NandConfig):
        return config.host_bandwidth_for(operation)
    raise ValueError("unsupported native memory contract")


def native_context_features(scenario, contexts):
    """Positive per-layer work coordinates, including fixed-block resets."""
    view = getattr(scenario.model, "_execution_view", None)
    if view is None:
        return (sum(contexts),)  # Pure formula fixtures have no executable IR.
    layers = [instance.layer for instance in view.layer_instances]
    chunks = sorted({layer.attention_chunk_size for layer in layers if layer.attention_chunk_size})
    features = []
    if any(not layer.attention_chunk_size for layer in layers):
        features.append(sum(context+1 for context in contexts))
    features.extend(sum(context % chunk+1 for context in contexts) for chunk in chunks)
    return tuple(features)


def _context_sensitive(metadata):
    # Read native operator identities, never a user-supplied certificate.
    name = str(metadata.get("operator_class", "")).lower()
    model_kind = str(metadata.get("model_operator_kind", "")).lower()
    event_kind = str(metadata.get("event_kind", "")).lower()
    if ("attention" in name or "attention" in model_kind
            or name.startswith("kv_") or event_kind.startswith("kv_")
            or any(metadata.get(key) is not None for key in (
                "generic_kv_storage", "kv_operation", "kv_cache_component", "kv_operand",
                "kv_read_semantics", "native_kv_work", "source_f32_kv"))
            or metadata.get("persistent_request_buffers")):
        return True
    # Transfer/materialization Tasks can retain their native storage
    # descriptors after losing the higher-level operator metadata. Their
    # KV rows still grow with context, so they cannot share a fixed-shape
    # atomic REF envelope merely because the generic marker is absent.
    def kv_identity(identity):
        return (isinstance(identity, str) and (identity.startswith(("kv:", "native_kv:"))
            or ".k_cache" in identity or ".v_cache" in identity))
    if any(kv_identity(identity) for identity in metadata.get("persistent_buffer_ids", ())):
        return True
    return any(isinstance(access, Mapping) and kv_identity(access.get("buffer_id"))
        for key in ("memory_accesses", "physical_allocations", "buffer_accesses",
                    "source_buffer_accesses", "target_buffer_accesses", "rhs_buffer_accesses",
                    "kv_writeback_accesses") for access in metadata.get(key, ()))


def _native_dag_ancestry(tasks):
    """Exact reachability of a bounded, original native Task DAG.

    Each bit represents an existing Task. Only authored edges whose two
    endpoints are present participate; an absent parent's own ancestry is
    unknown and never supplies a path. Kahn ordering checks the whole known
    graph, so input enumeration need not already be topological. The 4096
    node limit caps bit storage at 2 MiB before ordinary mapping overhead.
    """
    from .contracts import _PreparedExecutionTask

    count = len(tasks)
    unavailable = {"qualified": False, "task_count": count,
        "physical_state_advanced": False}
    if count > 4096:
        return None, {**unavailable, "reason": "native_DAG_exceeds_bounded_layer_scope"}
    if any(type(task) is not _PreparedExecutionTask or type(task.task_id) is not str
            or type(task.dependencies) is not tuple
            or any(type(parent) is not str for parent in task.dependencies) for task in tasks):
        return None, {**unavailable, "reason": "native_DAG_has_unknown_Task_or_dependency"}
    indexes = {task.task_id: index for index, task in enumerate(tasks)}
    if len(indexes) != count:
        return None, {**unavailable, "reason": "native_DAG_has_duplicate_Task_identity"}
    parents, children = [], [[] for _ in tasks]
    external = set()
    for task in tasks:
        known = tuple(dict.fromkeys(indexes[parent] for parent in task.dependencies if parent in indexes))
        parents.append(known)
        external.update(parent for parent in task.dependencies if parent not in indexes)
        for parent in known:
            children[parent].append(indexes[task.task_id])
    remaining = [len(row) for row in parents]
    ready = deque(index for index, value in enumerate(remaining) if not value)
    ancestors, completed = [0]*count, 0
    while ready:
        index = ready.popleft()
        for parent in parents[index]:
            ancestors[index] |= ancestors[parent] | (1 << parent)
        completed += 1
        for child in children[index]:
            remaining[child] -= 1
            if not remaining[child]:
                ready.append(child)
    if completed != count:
        return None, {**unavailable, "reason": "native_DAG_contains_a_cycle"}
    return (indexes, ancestors), {"qualified": True, "task_count": count,
        "known_edge_count": sum(map(len, parents)), "missing_parent_count": len(external),
        "basis": "original_known_native_Task_edges_exact_bitset_transitive_closure; missing_parents_supply_no_path",
        "physical_state_advanced": False}


def _native_dag_precedes(reachability, previous, current):
    indexes, ancestors = reachability
    before, after = indexes.get(previous), indexes.get(current)
    return (before is not None and after is not None
        and (before == after or bool(ancestors[after] & (1 << before))))


def _hbm_channels_independent(config, physical_runtime, *, owner=None):
    """Prove that native row/column/DQ buses do not join two channels."""
    if (physical_runtime is None or config.control_mode != "jedec_hbm"
            or config.stack_interface_mode != "independent"
            or config.addressable_stack_count != 1 or config.addressable_die_count != 1
            or config.ranks_per_channel != 1 or config.subchannels_per_channel != 1
            or config.pseudo_channels_per_channel != 2
            or config.lanes_per_stack != 2*config.channels):
        return False
    metadata = config.metadata
    prefixes = tuple(metadata.get(name, default if owner is None else f"{owner}:{default}") for name, default in (
        ("row_command_resource_prefix", "dram:row_command"),
        ("column_command_resource_prefix", "dram:column_command"),
        ("data_resource_prefix", "dram:data")))
    if any(type(value) is not str for value in prefixes) or len(set(prefixes)) != 3:
        return False
    timeline = physical_runtime.timeline
    if not timeline.resource_aliases:
        return True
    ids = [f"{prefix}:{stack}:{channel}" for prefix in prefixes[:2]
           for stack in range(config.stacks) for channel in range(config.channels)]
    ids.extend(f"{prefixes[2]}:{lane}" for lane in range(config.lane_count))
    return all(timeline.canonical_resource_id(identity) == identity for identity in ids)


def _native_refresh_source_qualification(config, owner, physical_runtime, native_runtime):
    """Read a closed native REF state without packing or publishing a Core.

    The runtime origin is only one prerequisite. Every mutable calendar and
    controller inequality is reread, including after a native restore. Public
    bank/control getters terminate exclusive numeric ownership; unknown future
    callbacks or an externally supplied kernel keep the old serial envelope.
    """
    from .execution_control import native_callback_free_control
    from .kv_transfer import native_external_publication_supported
    from .native_hbm_cold_bound import _native_refresh_source_limits

    def reject(reason):
        return {"qualified": False, "reason": reason, "physical_state_advanced": False}

    if (type(native_runtime) is not _NATIVE_SOURCE_RUNTIME
            or vars(native_runtime).get("_native_window_owns_execution_kernel") is not True
            or not native_callback_free_control(vars(native_runtime).get("execution_control"))):
        return reject("native_refresh_source_has_external_kernel_or_callback")
    kernel = vars(native_runtime).get("_execution_resource_kernel")
    if (type(kernel) is not _NATIVE_SOURCE_KERNEL
            or vars(kernel).get("physical_runtime") is not physical_runtime
            or type(physical_runtime) is not _NATIVE_SOURCE_PHYSICAL):
        return reject("native_refresh_source_has_unknown_kernel_or_physical_context")
    lowerer = vars(native_runtime).get("lowerer")
    if type(lowerer) is not _NATIVE_SOURCE_LOWERER:
        return reject("native_refresh_source_has_unknown_lowerer")
    for instance, methods in ((native_runtime, _NATIVE_SOURCE_RUNTIME_METHODS),
            (kernel, _NATIVE_SOURCE_KERNEL_METHODS),
            (physical_runtime, _NATIVE_SOURCE_PHYSICAL_METHODS),
            (lowerer, _NATIVE_SOURCE_LOWERER_METHODS)):
        if (type(instance).__getattribute__ is not object.__getattribute__
                or any(getattr(getattr(instance, name), "__func__", getattr(instance, name)) is not original
                       for name, original in methods)):
            return reject("native_refresh_source_has_unknown_future_writer")
    for name in ("_external_executor", "_paged_pool_transfer_executor"):
        executor = vars(native_runtime).get(name)
        if executor is not None and not native_external_publication_supported(executor):
            return reject("native_refresh_source_has_unknown_transfer_writer")
    if type(config) is not DramConfig or config.refresh_policy != "all_bank":
        return reject("native_refresh_source_has_shared_or_unsupported_HBM_buses")
    runtimes = vars(physical_runtime).get("runtimes")
    if (type(runtimes) is not dict
            or not physical_runtime._owners_have_native_hooks(runtimes)):
        return reject("native_refresh_source_has_unknown_memory_writer")
    timeline = physical_runtime.timeline
    # A native sibling may be configured to use these exact ports without
    # an alias. That is still shared hardware, not a private HBM channel.
    active = runtimes.get(owner)
    actual_config = config if active is None else active.core.config
    if not _hbm_channels_independent(actual_config, physical_runtime, owner=owner):
        return reject("native_refresh_source_has_shared_or_unsupported_HBM_buses")
    prefixes = tuple(actual_config.metadata.get(name, f"{owner}:{default}") for name, default in (
        ("row_command_resource_prefix", "dram:row_command"),
        ("column_command_resource_prefix", "dram:column_command"),
        ("data_resource_prefix", "dram:data")))
    ids = _hbm_phase_resource_ids(prefixes, config.stacks, config.channels, config.lane_count)
    for identity, sibling in runtimes.items():
        if identity == owner:
            continue
        for name, prefix in sibling.core.config.metadata.items():
            if name.endswith("resource_prefix") and type(prefix) is str:
                if any(resource == prefix or resource.startswith(prefix+":") for resource in ids):
                    return reject("native_refresh_source_has_shared_memory_ports")
            elif name in {"host_resource_id", "shared_command_resource"} and prefix in ids:
                return reject("native_refresh_source_has_shared_memory_ports")
    lane_ids = set()
    live_ready = {}
    for resource, slots in timeline.lane_available.items():
        if (type(resource) is not str or type(slots) is not list or id(slots) in lane_ids
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0.
                       for value in slots)):
            return reject("native_refresh_source_has_unknown_or_shared_calendar")
        lane_ids.add(id(slots))
        if resource in ids:
            if len(slots) != 1:
                return reject("native_refresh_source_has_shared_HBM_lane")
            live_ready[resource] = slots[0]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0.
            for value in timeline.ready_ns.values())
            or type(vars(native_runtime).get("now")) not in (int, float)):
        return reject("native_refresh_source_has_unknown_frontier")
    # ResourceTimeline.available_ns uses ready_ns when a resource has no
    # lane list. Read that same authority without invoking any public hook.
    for resource in ids:
        live_ready.setdefault(resource, timeline.ready_ns.get(resource, 0.))
    frontier = max(float(native_runtime.now), max(timeline.ready_ns.values(), default=0.),
                   max(live_ready.values(), default=0.))
    limits = _native_refresh_source_limits(config)
    history_upper = limits["initial_history_upper_ns"]
    proof = {"qualified": True, "physical_state_advanced": False,
        "basis": "fresh_native_kernel_callback_free_owned_hooks_and_reread_REF_epoch_state",
        "current_physical_frontier_ns": frontier,
        "initial_history_upper_ns": history_upper,
        "controller_state_published": False,
        "source_carry_scope": "all_real_source_CA_DQ_frontiers_paid_before_independent_samples"}
    if active is None:
        if any(live_ready.values()):
            return reject("native_refresh_uncreated_owner_has_prior_bus_reservations")
        return {**proof, "state": "uncreated_native_cold_owner"}
    core = active.core
    values = vars(core)
    if (type(core) is not DramCore or values.get("timeline") is not timeline
            or values.get("_hbm_numeric_enabled") is not True
            or values.get("_hbm_transactions")):
        return reject("native_refresh_source_state_exposed_or_transaction_active")
    # Bound configs add owner prefixes; executable geometry/timing must be
    # the same normalized scalar contract as the future IR descriptor.
    if any(value != vars(config).get(name) for name, value in vars(core.config).items()
           if name != "metadata"):
        return reject("native_refresh_source_future_config_differs")
    packed = values.get("_hbm_numeric_state")
    if packed is None:
        if (values.get("_banks") or values.get("_hbm_control_state")
                or any(live_ready.values())):
            return reject("native_refresh_source_nonempty_unowned_python_state")
        return {**proof, "state": "empty_native_cold_controller"}
    from ._compiled_hbm import _has_native_timeline_hooks
    import numpy as np
    if not _has_native_timeline_hooks(packed) or packed.core is not core:
        return reject("native_refresh_source_has_unknown_numeric_writer")
    arrays = packed.arrays
    lanes, banks, channels = config.lane_count, config.bank_count, config.lane_count//2
    shapes = ((lanes*banks,), (lanes*banks, 7), (lanes*banks,), (lanes, 4),
              (lanes, banks//min(16, config.banks_per_group), 4), (lanes,),
              (channels, 2), (channels, 2))
    dtypes = (np.int64, np.float64, np.int64, np.float64,
              np.float64, np.bool_, np.float64, np.int64)
    if (type(arrays) is not tuple or len(arrays) != 14
            or any(type(array) is not np.ndarray or array.shape != shape
                or array.dtype != dtype for array, shape, dtype in zip(arrays, shapes, dtypes))
            or any(not np.isfinite(array).all() for array in arrays[:8])):
        return reject("native_refresh_source_numeric_columns_invalid")
    from ._hbm_numeric import TIMING_FIELDS
    expected_g = (lanes, banks, (config.interleave_bytes or config.burst_bytes)//config.burst_bytes,
        config.row_bytes//config.burst_bytes, config.banks_per_group, config.bank_groups_per_rank,
        -1 if config.activated_row_msb is None else config.activated_row_msb,
        int(config.generation in {"HBM2", "HBM2E"}), 1, config.burst_bytes,
        min(16, config.banks_per_group), int(config.address_mapping == "row_local"))
    expected_t = tuple(getattr(config, name) for name in TIMING_FIELDS)+( 
        config.burst_bytes/config.effective_lane_bandwidth_gb_s,
        config.burst_bytes/config.directional_lane_bandwidth_gb_s("read"),
        config.burst_bytes/config.directional_lane_bandwidth_gb_s("write"),
        config.maximum_active_ns, config.four_activate_window_ns)
    if (type(packed.g) is not np.ndarray or packed.g.dtype != np.int64
            or packed.g.shape != (len(expected_g)+1,)
            or tuple(packed.g[:len(expected_g)]) != expected_g or packed.g[-1] not in (0, 1)
            or type(packed.t) is not np.ndarray or packed.t.dtype != np.float64
            or tuple(packed.t) != expected_t):
        return reject("native_refresh_source_instruction_or_REF_contract_changed")
    rows, bf, addresses, pcs, history, seen, channel, _ = arrays[:8]
    if packed.g[-1]:
        columns, quantum = history[:, :, 1:3], config.command_clock_ns/2
        ticks = np.rint(columns/quantum)
        if not np.all((columns < 0.) | ((np.abs(ticks) < 2.**49) & (columns == ticks*quantum))):
            return reject("native_refresh_source_grid_history_claim_not_proven")
    if any(not math.isfinite(value) or value < 0. for value in (frontier, history_upper)):
        return reject("native_refresh_source_clock_domain_invalid")
    interval, clock = config.refresh_interval_ns, config.command_clock_ns
    expected_ids = tuple(f"{prefix}:{stack}:{ch}" for prefix in prefixes[:2]
        for stack in range(config.stacks) for ch in range(config.channels))
    expected_ids += tuple(f"{prefixes[2]}:{lane}" for lane in range(lanes))
    if packed.ids != expected_ids:
        return reject("native_refresh_source_numeric_port_layout_changed")
    for index in range(channels):
        first, stop = 2*index*banks, (2*index+2)*banks
        pc_first, pc_stop = 2*index, 2*index+2
        due = float(channel[index, 1])
        pc = pcs[pc_first:pc_stop]
        bh = bf[first:stop]
        hs = history[pc_first:pc_stop]
        row_id, col_id = packed.ids[index], packed.ids[channels+index]
        dq_ids = packed.ids[2*channels+2*index:2*channels+2*index+2]
        if due < 0.:
            if (np.any(rows[first:stop] >= 0) or np.any(addresses[first:stop] >= 0)
                    or np.any(bh[:, 0] != -1.) or np.any(bh[:, 1:] != 0.)
                    or np.any(pc[:, 0] != -1.) or np.any(pc[:, 1:] != 0.)
                    or np.any(hs != -1.) or np.any(seen[pc_first:pc_stop])
                    or channel[index, 0] != -1.
                    or any(live_ready.get(rid, 0.) != 0. for rid in (row_id, col_id, *dq_ids))):
                return reject("native_refresh_source_unseen_epoch_is_not_cold")
            continue
        if (due > frontier+interval or due < interval
                or np.any(pc[:, 1] > min(frontier, due))
                or np.any(pc[:, 0] > min(frontier, due))
                or channel[index, 0] > min(frontier, due)
                or np.any(hs[:, :, :3] > min(frontier, due))
                or np.any(hs[:, :, 3] > min(frontier, due)+limits["source_final_tail_upper_ns"])
                or np.any(bh[:, 0] > min(frontier, due))
                or np.any(bh[:, 1:] > min(frontier, due)+history_upper)
                or np.any(bh[:, 1] > due+config.precharge_to_activate_ns)
                or np.any(bh[:, 2] > due+limits["source_precharge_tail_upper_ns"])
                or np.any(bh[:, 3] > due+config.activate_to_read_ns)
                or np.any(bh[:, 4] > due+config.activate_to_write_ns)
                or np.any(bh[:, 5] > due+max(config.precharge_to_activate_ns,
                                             limits["source_precharge_tail_upper_ns"]))
                or np.any(bh[:, 6] > due+min(config.activate_to_read_ns, config.activate_to_write_ns))
                or np.any(pc[:, 2:] > min(frontier, due)+limits["source_final_tail_upper_ns"])):
            return reject("native_refresh_source_epoch_history_or_future_due_invalid")
        if due > interval and np.any(pc[:, 1] < due-interval+config.refresh_cycle_ns):
            return reject("native_refresh_source_epoch_missing_both_PC_refresh_recovery")
        opened = rows[first:stop] >= 0
        if np.any(bh[opened, 0] < due-interval-limits["source_ACT_due_rounding_margin_ns"]):
            return reject("native_refresh_source_open_row_predates_last_REF_epoch")
        # The real timeline remains authoritative; no cached pack is refreshed.
        if (max(live_ready.get(row_id, 0.), live_ready.get(col_id, 0.)) > due+limits["source_CA_tail_upper_ns"]
                or max((live_ready.get(rid, 0.) for rid in dq_ids), default=0.)
                    > due+limits["source_final_tail_upper_ns"]):
            return reject("native_refresh_source_unpaid_historical_REF_debt")
    return {**proof, "state": "owned_numeric_REF_epoch_state", "channel_count": channels}


def _channel_burst_upper(config, bursts):
    """Exact address-independent maximum from the native periodic mapping.

    Within one lane stripe a physical channel owns two adjacent PC blocks.
    In an arbitrary contiguous interval, its complete periods contribute
    exactly two blocks and the remaining fraction contributes at most two.
    No payload/bandwidth service estimate is used here.
    """
    block = (config.bank_count*(config.row_bytes//config.burst_bytes)
             if config.address_mapping == "row_local" else
             (config.interleave_bytes or config.burst_bytes)//config.burst_bytes)
    period = config.lane_count*block
    whole, residual = divmod(bursts, period)
    return whole*2*block+min(residual, 2*block)


def _refresh_serial_terms(config, byte_count, operation, *, independent_channels=False,
                          allow_bounded_clock=False, native_refresh_reachable=False):
    """Conservative active-IO refresh feedback, without charging GPU idle.

    The current native controller refreshes idle epochs before admission.
    Those completed epochs cannot delay this IO; one residual epoch can.
    During the active request, serializing every burst and every configured
    dependency gate gives a cold no-refresh service upper D. Proven native
    independent channels need only their largest contiguous burst count;
    all bursts remain serial within that channel. Otherwise keep the full
    request count. With blocking refresh cost R and interval I, an initial
    residual REF plus floor(D/(I-R)) later epochs covers refresh feedback.
    A fully elapsed idle epoch is not a new active-request stall.
    This intentionally retains nominal refresh already present in samples.
    It is broad until a tighter native row-cycle upper is available.
    """
    if not isinstance(config, DramConfig) or config.refresh_policy != "all_bank":
        return None
    interval = config.refresh_interval_ns
    if byte_count == 0:
        return None
    clock = config.command_clock_ns
    recovery = max(config.minimum_active_ns, config.read_to_precharge_ns,
        config.precharge_to_precharge_ns,
        config.write_latency_ns+config.burst_bytes/config.effective_lane_bandwidth_gb_s
            +config.write_to_precharge_ns,
        2*clock)
    # Both PCs issue their native PREab/REFab on this channel's shared row
    # bus. PC0 books REF after its tRP before PC1 may issue PRE, so both
    # tRP gaps remain serial even though the tRFC array periods overlap.
    blocked = (config.refresh_cycle_ns+recovery
        +config.pseudo_channels_per_channel*config.precharge_to_activate_ns
        +(2*config.pseudo_channels_per_channel*config.precharge_command_cycles
          +config.activate_command_cycles+config.pseudo_channels_per_channel+2)*clock)
    if blocked >= interval:
        raise ValueError("native refresh serial feedback has no positive service interval")
    gates = (config.open_ns, config.close_ns, config.read_latency_ns, config.write_latency_ns,
        config.burst_interval_ns, config.read_recovery_ns, config.write_recovery_ns,
        config.read_to_write_ns, config.write_to_read_ns, config.column_same_group_ns,
        config.column_different_group_ns, config.activate_same_group_ns,
        config.activate_different_group_ns, config.four_activate_window_ns,
        config.write_to_read_same_group_ns, config.write_to_read_different_group_ns,
        config.activate_to_read_ns, config.activate_to_write_ns,
        config.precharge_to_activate_ns, config.minimum_active_ns, config.activate_cycle_ns,
        config.read_to_precharge_ns, config.write_to_precharge_ns, config.column_different_sid_ns,
        config.read_to_write_command_ns, config.precharge_to_precharge_ns,
        (config.activate_command_cycles+config.precharge_command_cycles+config.column_command_cycles+2)*config.command_clock_ns,
        config.burst_bytes/config.directional_lane_bandwidth_gb_s(operation))
    bursts = (byte_count+2*config.burst_bytes-2)//config.burst_bytes
    if independent_channels:
        bursts = _channel_burst_upper(config, bursts)
    # Existing rows in other banks can expire during this access. Each has
    # at most one outstanding initial PRE; every subsequent ACT/PRE is
    # already overcharged by the complete per-burst gate sum above.
    initial_pre = (config.pseudo_channels_per_channel*config.bank_count
                   *(config.precharge_command_cycles+1)*clock)
    active_upper = bursts*sum(gates)+initial_pre
    cold = (_native_cold_ref_service(config, byte_count, operation,
                                    independent_channels=independent_channels,
                                    native_refresh_reachable=native_refresh_reachable)
            if allow_bounded_clock else None)
    if cold is not None:
        active_upper = min(active_upper, cold["conditional_no_refresh_serial_upper_ns"])
    return interval, blocked, active_upper


def _native_cold_ref_service(config, byte_count, operation, *, independent_channels,
                             native_refresh_reachable=False):
    # This domain-constrained upper is paired with an explicit summary
    # clock limit. The final whole-window evaluator must close that domain
    # over every future transaction before accepting any forecast.
    if not independent_channels:
        return None
    reachable = None
    if native_refresh_reachable:
        from .native_hbm_cold_bound import native_hbm_refresh_service_upper
        proof = native_hbm_refresh_service_upper(config, byte_count, operation,
            independent_channels=True, native_refresh_reachable=True)
        if proof["qualified"]:
            reachable = proof
    if byte_count < 256*1024:
        return reachable
    from .native_hbm_cold_bound import native_hbm_no_refresh_upper
    proof = native_hbm_no_refresh_upper(config, byte_count, operation,
                                      independent_channels=independent_channels)
    if not proof["qualified"]:
        return reachable
    if reachable is not None:
        selected = (reachable if reachable["active_service_upper_ns"]
                    < proof["conditional_no_refresh_serial_upper_ns"] else proof)
        # The REF-on source invariant also supports the row-state envelope.
        # Keep its absolute clock guard even if the cold active-work bound
        # happens to be smaller for this IO shape.
        raw_limit = min(reachable["required_native_hbm_clock_limit_ns"],
                        proof["required_native_hbm_clock_limit_ns"])
        reserve = max(reachable["hbm_clock_domain_comparison_reserve_ns"],
                      proof["hbm_clock_domain_comparison_reserve_ns"])
        return {**selected, "required_native_hbm_clock_limit_ns": raw_limit,
            "hbm_clock_domain_limit_ns": math.nextafter(raw_limit-reserve, -math.inf),
            "hbm_clock_domain_comparison_reserve_ns": reserve,
            "native_REF_on_source_clock_retained": True}
    return proof


def _native_row_source_proven(config, byte_count, operation, *, independent_channels,
                              allow_bounded_clock, native_refresh_reachable):
    if not (allow_bounded_clock and native_refresh_reachable and independent_channels):
        return False
    from .native_hbm_cold_bound import native_hbm_refresh_service_upper
    return native_hbm_refresh_service_upper(config, byte_count, operation,
        independent_channels=True, native_refresh_reachable=True)["qualified"]


def _refresh_serial_bound(config, byte_count, operation, *, independent_channels=False,
                         allow_bounded_clock=False, native_refresh_reachable=False):
    terms = _refresh_serial_terms(config, byte_count, operation,
                                 independent_channels=independent_channels,
                                 allow_bounded_clock=allow_bounded_clock,
                                 native_refresh_reachable=native_refresh_reachable)
    if terms is None:
        return 0.
    interval, blocked, active_upper = terms
    return (1+math.floor(active_upper/(interval-blocked)))*blocked


def _phase(config, byte_count, operation, *, independent_hbm_channels=False,
           allow_bounded_clock=False, native_refresh_reachable=False):
    """Residual initial-state phase used by the declared steady model.

    This is deliberately separate from bulk service, which is retained in
    measured cohorts. It bounds one initial row/page/refresh phase, not the
    total time of an arbitrary adversarial sequence of future IOs.
    """
    if isinstance(config, SramConfig):
        return 0., 0., 0.
    if isinstance(config, DramConfig):
        row = max(config.open_ns + config.close_ns,
                  config.precharge_to_activate_ns + config.activate_to_read_ns,
                  config.precharge_to_activate_ns + config.activate_to_write_ns,
                  config.activate_cycle_ns, config.four_activate_window_ns)
        row += max(config.read_to_write_ns, config.write_to_read_ns,
                   config.read_to_write_command_ns,
                   config.write_to_read_same_group_ns,
                   config.write_to_read_different_group_ns)
        # A contiguous request can enter at one different row per bank;
        # bank array operations overlap, but row-command issue does not.
        touched = min(config.bank_count, max(1, math.ceil(byte_count / config.burst_bytes / config.lane_count)))
        row += touched * config.command_clock_ns * (
            config.activate_command_cycles + config.precharge_command_cycles)
        if config.control_mode == "jedec_hbm":
            from .native_hbm_cold_bound import native_hbm_initial_row_upper
            initial = native_hbm_initial_row_upper(config, byte_count, operation,
                independent_channels=independent_hbm_channels,
                native_refresh_reachable=_native_row_source_proven(config, byte_count, operation,
                    independent_channels=independent_hbm_channels,
                    allow_bounded_clock=allow_bounded_clock,
                    native_refresh_reachable=native_refresh_reachable))
            if initial["qualified"]:
                # A wrong initial row also serializes its complete tRP/ACT
                # startup chain. CA issue occupancy alone is insufficient.
                row = max(row, initial["initial_state_upper_ns"])
        refresh = _refresh_serial_bound(config, byte_count, operation,
                                       independent_channels=independent_hbm_channels,
                                       allow_bounded_clock=allow_bounded_clock,
                                       native_refresh_reachable=native_refresh_reachable)
        return row, refresh, 0.
    if isinstance(config, NandConfig):
        if config.protocol_mode != "generic_nand" or config.write_buffer_bytes or config.wear_model is not None:
            raise ValueError("native window does not bound FTL, staging or wear state")
        # Independent banks can each contribute a cached-page phase in
        # parallel. Payload still crosses the charged host/internal buses.
        touched_banks = min(config.channels*config.targets_per_channel*config.dies_per_target
            *config.luns_per_die*config.planes_per_lun,
            max(1, math.ceil((byte_count+config.page_bytes-1)/config.page_bytes)))
        # A cache miss also performs a full-page internal transfer; a hit
        # omits it. Charge the possible saved array time plus serial internal
        # bus payload across all touched cache banks. We do not assume that
        # those internal transfers overlap merely because arrays do.
        internal_page = config.transfer_page_bytes/config.internal_bandwidth_gb_s
        cache_array_rounds = max(1, math.ceil(touched_banks/config.array_units))
        cache = (config.read_cache_pages_per_bank*(cache_array_rounds*config.page_read_ns+touched_banks*internal_page)
                 if operation == "read" else 0.)
        # At most the first/last page of a contiguous write are partial.
        # RMW adds their read array and full internal-page transfers; program
        # and write payload remain in the original measured bulk service.
        page = (2*(config.page_read_ns+internal_page)
                if operation == "write" and config.partial_page_policy == "read_modify_write" else 0.)
        return page, 0., cache
    raise ValueError("unsupported native memory contract")


def _nand_cache_release_phase(config, byte_count, physical_runtime):
    """Native parallel page fills delay mandatory host payload release.

    A contiguous request has balanced native channel/array mapping. At
    most C cached pages per bank differ between initial states. Fully
    cold release adds at most the maximum per-array serial read chain
    plus the maximum per-channel serial internal-page chain. The unchanged
    host payload/credits remain in measured service, and may serialize
    afterwards. This is a release-delay guard, not a bandwidth division of
    NAND array latency. Unknown/shared physical buses keep the coarse sum.
    """
    if not isinstance(config, NandConfig) or physical_runtime is None or not config.read_cache_pages_per_bank:
        return None
    canonical = physical_runtime.timeline.canonical_resource_id
    channel_prefix = str(config.metadata.get("channel_resource_prefix", "nand:channel"))
    array_prefix = str(config.metadata.get("parallel_resource_prefix" if config.parallel_units is not None
                                        else "array_resource_prefix",
                                        "nand:parallel" if config.parallel_units is not None else "nand:array"))
    base_units = config.channels*config.targets_per_channel*config.dies_per_target*config.luns_per_die
    native_units = base_units*(config.planes_per_lun if config.planes_independent else 1)
    units = min(config.parallel_units or native_units, native_units)
    if native_units % units:
        return None
    channel_ids = {canonical(f"{channel_prefix}:{index}") for index in range(config.channels)}
    array_ids = {canonical(f"{array_prefix}:{index}") for index in range(units)}
    host_prefix = str(config.metadata.get("host_resource_id", "nand:host"))
    host_names = ({host_prefix} if config.host_channels == 1 else
                  {f"{host_prefix}:{index}" for index in range(config.host_channels)})
    if config.host_full_duplex:
        host_names = {f"{name}:{operation}" for name in host_names for operation in ("read", "write")}
    host_ids = {canonical(name) for name in host_names}
    if (len(channel_ids) != config.channels or len(array_ids) != units
            or channel_ids & array_ids or host_ids & (channel_ids | array_ids)):
        return None
    pages = math.ceil((byte_count+config.page_bytes-1)/config.page_bytes)
    banks = (config.channels*config.targets_per_channel*config.dies_per_target
             *config.luns_per_die*config.planes_per_lun)
    cache = config.read_cache_pages_per_bank
    array_chain = min(math.ceil(pages/units), math.ceil(banks/units)*cache)
    per_channel = base_units//config.channels
    cycles, tail = divmod(pages, base_units)
    max_channel_pages = cycles*per_channel+min(tail, per_channel)
    channel_chain = min(max_channel_pages, (banks//config.channels)*cache)
    return array_chain*config.page_read_ns+channel_chain*config.transfer_page_bytes/config.internal_bandwidth_gb_s


@lru_cache(maxsize=128)
def _hbm_phase_resource_ids(prefixes, stacks, channels, lanes):
    result = {f"{prefix}:{stack}:{channel}" for prefix in prefixes[:2]
              for stack in range(stacks) for channel in range(channels)}
    result.update(f"{prefixes[2]}:{lane}" for lane in range(lanes))
    return frozenset(result)


def _atomic_task_refresh_bound(task, terms, configs, physical_runtime, old_refresh, *,
                               independent_channels=False):
    """One native Task does not wait for a different owner's completion.

    execute_prepared_physical_task starts every direct descriptor at the
    Task arrival; only same-owner overlapping RAW/WAR/WAW can raise it to
    an earlier same-owner descriptor's completion. The wall time spent
    calculating a NAND reservation never advances that simulated arrival.
    Thus the complete serialized HBM IO uppers cover this atomic descriptor
    batch without adding NAND/compute gaps. Staging/L2 and any shared bus
    are excluded. This contracts no Task edge and hides no between-Task wait.
    """
    if (len(terms) != 1 or physical_runtime is None or not independent_channels
            or _context_sensitive(task.metadata)
            or task.metadata.get("hbf_host_staging_by_owner")
            or task.metadata.get("hbf_host_staging") or task.metadata.get("stateful_l2")
            or task.metadata.get("l2_execution")):
        return old_refresh, None
    owner, values = next(iter(terms.items()))
    if len(values) < 2 or len({(interval, blocked) for interval, blocked, _d in values}) != 1:
        return old_refresh, None
    hbm = configs[owner]
    if (type(hbm) is not DramConfig or hbm.control_mode != "jedec_hbm"
            or hbm.scheduler != "fcfs"):
        return old_refresh, None
    prefixes = tuple(str(hbm.metadata.get(name, default)) for name, default in (
        ("row_command_resource_prefix", "dram:row_command"),
        ("column_command_resource_prefix", "dram:column_command"),
        ("data_resource_prefix", "dram:data")))
    hb_ids = _hbm_phase_resource_ids(prefixes, hbm.stacks, hbm.channels, hbm.lane_count)
    timeline = physical_runtime.timeline
    for other, config in configs.items():
        if other == owner:
            continue
        if type(config) is NandConfig and config.protocol_mode == "generic_nand":
            names = (("array_resource_prefix", "nand:array"),
                     ("parallel_resource_prefix", "nand:parallel"),
                     ("channel_resource_prefix", "nand:channel"),
                     ("host_resource_id", "nand:host"))
        elif type(config) is SramConfig:
            names = (("port_resource_prefix", "sram:port"),)
        else:
            return old_refresh, None
        other_prefixes = tuple(str(config.metadata.get(name, default)) for name, default in names)
        def in_other(identity):
            return any(identity == prefix or identity.startswith(prefix+":") for prefix in other_prefixes)
        if (any(in_other(identity) for identity in hb_ids)
                or any(in_other(identity) and timeline.canonical_resource_id(identity) in hb_ids
                       for identity in timeline.resource_aliases)):
            return old_refresh, None
    interval, blocked = values[0][:2]
    service = math.fsum(value[2] for value in values)
    grouped = min(old_refresh, (1+math.floor(service/(interval-blocked)))*blocked)
    return grouped, {"task_id": task.task_id, "physical_owner": owner,
        "descriptor_count": len(values), "old_per_descriptor_refresh_upper_ns": old_refresh,
        "atomic_task_refresh_upper_ns": grouped,
        "serialized_native_no_refresh_service_upper_ns": service,
        "basis": "native_direct_descriptor_common_Task_arrival_and_same_owner_overlap_only; one_residual_REF_plus_full_serial_feedback",
        "non_HBM_resources_disjoint": True, "between_Task_waits_removed": False,
        "physical_state_advanced": False}


def _native_task_hbm_phase_mask(task, configs, physical_runtime, native_runtime, *,
                                source_proofs, owner_tasks, exact_ancestry,
                                direct_port_writers, compute_resources,
                                physical_prefixes, hidden_physical_writers=False):
    """Native compute or mandatory NAND payload can cover an HBM branch.

    This uses an absolute REF-on service upper, not REF counters or a replay
    with commands removed. Every other user of that owner must be ordered
    through the original Task endpoint, so an earlier HBM release cannot
    unblock an otherwise independent branch. Later Tasks retain their own
    full controller-state protection.
    """
    from .native_hbm_cold_bound import native_hbm_refresh_service_upper

    proof = {"task_id": task.task_id, "qualified": False,
        "physical_state_advanced": False, "later_controller_state_guards_retained": True}

    def reject(reason):
        return {**proof, "reason": reason}

    hbms = [(owner, config) for owner, config in configs.items() if type(config) is DramConfig]
    if len(hbms) != 1 or physical_runtime is None or exact_ancestry is None:
        return reject("native_single_HBM_owner_or_exact_Task_DAG_unavailable")
    owner, config = hbms[0]
    proof["physical_owner"] = owner
    source = source_proofs.get((owner, id(config)), {})
    if source.get("qualified") is not True or hidden_physical_writers:
        return reject("native_REF_source_or_complete_physical_writer_set_unproven")
    if owner in direct_port_writers:
        return reject("native_HBM_owner_has_unmodelled_direct_port_writer")
    md = task.metadata
    if (task.opaque_device_fence or _context_sensitive(md)
            or any(md.get(key) for key in ("hbf_host_staging_by_owner", "hbf_host_staging",
                "stateful_l2", "l2_execution"))):
        return reject("native_Task_has_context_sensitive_or_indirect_physical_work")
    if any(not (_native_dag_precedes(exact_ancestry, other.task_id, task.task_id)
                    or _native_dag_precedes(exact_ancestry, task.task_id, other.task_id))
           for other in owner_tasks.get(owner, ())):
        return reject("native_HBM_owner_has_unordered_Task_branch")
    # The source contract already checked instantiated siblings. Unknown
    # future owners cannot silently share a CA/DQ prefix with this branch.
    if any(other not in physical_runtime.runtimes for other in owner_tasks):
        return reject("native_physical_owner_prefix_inventory_incomplete")
    for other, declared in configs.items():
        actual = physical_runtime.runtimes[other].core.config
        if any(actual.metadata.get(name) != value for name, value in declared.metadata.items()
               if name.endswith("resource_prefix") or name in {"host_resource_id", "shared_command_resource"}):
            return reject("native_future_physical_prefix_differs_from_live_owner")
    kernel = native_runtime._execution_resource_kernel
    origin = max(float(native_runtime.now), float(kernel.makespan_ns),
                 float(source["current_physical_frontier_ns"]))
    canonical = physical_runtime.timeline.canonical_resource_id
    prefixes = tuple(config.metadata.get(name, f"{owner}:{default}") for name, default in (
        ("row_command_resource_prefix", "dram:row_command"),
        ("column_command_resource_prefix", "dram:column_command"),
        ("data_resource_prefix", "dram:data")))
    hb_ids = set(_hbm_phase_resource_ids(prefixes, config.stacks, config.channels, config.lane_count))
    if any(identity not in hb_ids and canonical(identity) in hb_ids
           for identity in physical_runtime.timeline.resource_aliases):
        return reject("native_foreign_resource_aliases_HBM_port")
    # Mirror execute_prepared_physical_task's placeholder identities before
    # taking a lower bound. A physical owner itself can be a preview demand,
    # even when the access.resource_id uses another label.
    physical_ids = {canonical(str(access[key]))
                    for access in md.get("memory_accesses", ())
                    for key in ("resource_id", "physical_owner") if access.get(key)}
    if md.get("physical_owner"):
        physical_ids.add(canonical(str(md["physical_owner"])))
    physical_ids.update(canonical(str(identity)) for key in (
        "physical_demands_resource_ids", "physical_placeholder_resource_ids",
        "hbf_staging_replaced_link_ids") for identity in md.get(key, ()))
    audit = md.get("phase_metadata", {}).get("kernel_model")
    contract = audit.get("physical_memory_envelope") if isinstance(audit, Mapping) else None
    suffix = contract.get("resource_suffix") if isinstance(contract, Mapping) else None
    if contract is not None and (not isinstance(suffix, str) or not suffix):
        return reject("native_recost_envelope_identity_unavailable")
    lower = 0.
    lower_resources = []
    for demand in task.demands:
        resource = canonical(demand.resource_id)
        if (resource in physical_ids or (suffix and demand.resource_id.endswith(suffix))
                or demand.resource_id not in compute_resources or demand.work_units <= 0):
            continue
        if any(resource == prefix or resource.startswith(prefix+":") for prefix in prefixes):
            return reject("native_compute_or_link_resource_shares_HBM_port")
        # Native dispatch removes *all* actual Core resources, including
        # NAND host links and SRAM ports, before retaining compute/link
        # demands. An authored link label is not sufficient provenance.
        if any(resource == prefix or resource.startswith(prefix+":") for prefix in physical_prefixes):
            continue
        if max(kernel.resource_available.get(resource, 0.),
               kernel.resource_available.get(demand.resource_id, 0.)) > origin:
            return reject("native_nonphysical_source_queue_carry_unbounded")
        if not math.isfinite(demand.service_ns) or demand.service_ns < 0:
            return reject("native_nonphysical_service_lower_invalid")
        if demand.service_ns > lower:
            lower = demand.service_ns
            lower_resources = [demand.resource_id]
    components = []
    for access in md.get("memory_accesses", ()):
        if access.get("physical_owner", md.get("physical_owner")) != owner:
            continue
        count, operation = access.get("byte_count"), access.get("operation")
        if count == 0:
            continue
        bound = native_hbm_refresh_service_upper(config, count, operation,
            independent_channels=_hbm_channels_independent(config, physical_runtime, owner=owner),
            native_refresh_reachable=True)
        if bound.get("qualified") is not True:
            return reject("native_absolute_HBM_service_upper_unavailable:"+bound.get("reason", "unknown"))
        active, blocked = bound["active_service_upper_ns"], bound["refresh_epoch_upper_ns"]
        refresh = (1+math.floor(active/(config.refresh_interval_ns-blocked)))*blocked
        components.append({"byte_count": count, "operation": operation,
            "active_service_upper_ns": active, "residual_and_feedback_REF_upper_ns": refresh,
            "complete_service_upper_ns": math.nextafter(active+refresh, math.inf),
            "hbm_clock_domain_limit_ns": bound["hbm_clock_domain_limit_ns"],
            "hbm_clock_domain_comparison_reserve_ns": bound["hbm_clock_domain_comparison_reserve_ns"]})
    if not components:
        return reject("native_positive_HBM_descriptor_missing")
    upper = math.nextafter(math.fsum(row["complete_service_upper_ns"] for row in components), math.inf)
    clock_limit = min(row["hbm_clock_domain_limit_ns"] for row in components)
    # A generic NAND read always moves its complete logical payload through
    # the host ports, including page-buffer hits. The configured bandwidth
    # is the sum over host channels, so even perfect distribution cannot
    # finish a descriptor sooner than bytes / aggregate bandwidth. Use the
    # largest *single* descriptor; never add overlapping IO lower bounds.
    # This is the actual Core contract, not a replaced analytical demand.
    nand_lowers, nand_rejections = [], []
    for access in md.get("memory_accesses", ()):
        nand_owner = access.get("physical_owner", md.get("physical_owner"))
        nand = configs.get(nand_owner)
        count = access.get("byte_count")
        if type(nand) is not NandConfig or access.get("operation") != "read" or not count:
            continue
        if (nand.protocol_mode != "generic_nand" or nand.write_buffer_bytes
                or nand.wear_model is not None):
            nand_rejections.append("native_NAND_payload_has_unsupported_buffer_or_protocol")
            continue
        actual = physical_runtime.runtimes[nand_owner].core.config
        from dataclasses import replace
        if (replace(nand, metadata=actual.metadata) != actual
                or any(actual.metadata.get(key) != value for key, value in nand.metadata.items())):
            nand_rejections.append("native_NAND_payload_contract_differs_from_live_owner")
            continue
        host = actual.metadata["host_resource_id"]
        host_ids = ({host} if actual.host_channels == 1 else
                    {f"{host}:{index}" for index in range(actual.host_channels)})
        if actual.host_full_duplex:
            host_ids = {identity+":read" for identity in host_ids}
        if (any(canonical(identity) != identity for identity in host_ids)
                or any(identity not in host_ids and canonical(identity) in host_ids
                       for identity in physical_runtime.timeline.resource_aliases)
                or any(len(physical_runtime.timeline.lane_available.get(identity, [0.])) != 1
                       or kernel.resource_capacities.get(identity, 1) != 1 for identity in host_ids)):
            nand_rejections.append("native_NAND_payload_host_has_alias_or_multiple_lanes")
            continue
        bandwidth = actual.host_bandwidth_for("read")
        pages = (count+2*actual.page_bytes-2)//actual.page_bytes
        # Each host division/addition can round down. Charge every possible
        # page at the full admitted lifecycle clock, even though only one
        # channel's chain determines the endpoint. Page alignment only adds
        # traffic; using logical bytes remains a lower bound.
        page_service = (actual.page_bytes+2*actual.host_granularity_bytes)/(
            bandwidth/actual.host_channels)
        reserve = math.nextafter(pages*(math.ulp(clock_limit)+math.ulp(page_service)), math.inf)
        payload_lower = max(0., math.nextafter(count/bandwidth, -math.inf)-reserve)
        nand_lowers.append({"physical_owner": nand_owner, "byte_count": count,
            "aggregate_host_bandwidth_gb_s": bandwidth, "host_channels": actual.host_channels,
            "host_ports": sorted(host_ids), "floating_rounding_reserve_ns": reserve,
            "mandatory_payload_lower_ns": math.nextafter(payload_lower, -math.inf)
                if payload_lower else 0., "cache_hits_retain_complete_host_payload": True})
    payload_lower = max((row["mandatory_payload_lower_ns"] for row in nand_lowers), default=0.)
    independent_lower = max(lower, payload_lower)
    if independent_lower <= 0:
        proof.update(nand_payload_lower_rejections=nand_rejections)
        return reject("native_independent_compute_or_link_lower_unavailable")
    proof.update(nonphysical_service_lower_ns=lower, lower_bound_resources=lower_resources,
        independent_service_lower_ns=independent_lower,
        native_NAND_payload_lowers=nand_lowers, nand_payload_lower_rejections=nand_rejections,
        complete_HBM_service_upper_ns=upper, descriptor_bounds=components,
        source_boundary_origin_ns=origin, source_boundary_wait_ns=max(0., origin-native_runtime.now),
        unpaid_source_carry_upper_ns=0.,
        hbm_clock_domain_limit_ns=clock_limit,
        hbm_clock_domain_comparison_reserve_ns=max(row["hbm_clock_domain_comparison_reserve_ns"] for row in components),
        basis="native_common_Task_arrival; same_owner_descriptors_serialized_including_RAW_WAR_WAW; all_other_owner_users_ordered_by_original_DAG; complete_REF_on_HBM_upper_covered_by_retained_compute_or_mandatory_native_NAND_payload")
    if upper > independent_lower:
        return reject("native_independent_lower_does_not_cover_complete_HBM_upper")
    if origin+upper > proof["hbm_clock_domain_limit_ns"]:
        return reject("native_source_outside_absolute_HBM_service_clock_domain")
    return {**proof, "qualified": True}


def _native_fixed_kv_partial_write(config, access, metadata, physical_runtime):
    """One native token-major row always pays one identical partial RMW.

    Generic NAND performs the old-page read even if that page happens to
    be cached. With page-aligned storage and row_bytes dividing a page,
    every integer token position stays inside exactly one partial page.
    RMW/PROGRAM therefore belong to measured bulk work, and contribute
    no additional initial-state *difference*. Their physical events and
    counters are never removed. Unknown/multirow/alias geometry keeps the
    ordinary two-edge-page protection.
    """
    if (not isinstance(config, NandConfig) or config.protocol_mode != "generic_nand"
            or config.partial_page_policy != "read_modify_write" or config.write_buffer_bytes
            or config.wear_model is not None or access.get("operation") != "write"
            or physical_runtime is None or access.get("allocation_generation", 0) != 0):
        return False
    contract = metadata.get("generic_kv_storage")
    if (not isinstance(contract, Mapping) or contract.get("schema") != "analytical-token-major-kv/v1"
            or contract.get("layout") != "separate_token_major_k_and_v"
            or type(contract.get("materialized_tokens")) is not int
            or contract["materialized_tokens"] <= 0):
        return False
    all_owners = contract.get("owners", ())
    if sum(len(owner.get("write_positions", ())) for owner in all_owners) != contract["materialized_tokens"]:
        return False
    row = contract.get("row_bytes")
    if type(row) is not int or row <= 0 or row >= config.page_bytes or config.page_bytes % row:
        return False
    identity = access.get("buffer_id")
    owners = [owner for owner in all_owners
              if identity in (owner.get("id", "")+":k", owner.get("id", "")+":v")]
    if len(owners) != 1 or access.get("byte_count") != row:
        return False
    positions = tuple(owners[0].get("write_positions", ()))
    if (len(positions) != 1 or type(positions[0]) is not int or positions[0] < 0
            or access.get("offset_bytes") != positions[0]*row):
        return False
    owner = access.get("physical_owner", metadata.get("physical_owner"))
    allocator = physical_runtime.allocators.get(owner)
    if allocator is None or allocator.alignment_bytes < config.page_bytes:
        return False
    allocation = allocator.get_allocation(identity, 0)
    if (allocation is None or allocation.alias_of is not None
            or allocation.allocation_arena != "resident"
            or allocation.base_address % config.page_bytes
            or access["offset_bytes"]+row > allocation.size_bytes
            or allocator._aliases_by_target.get((identity, 0))):
        return False
    return True


def _cold_resident_nand_weights(stages, physical_runtime):
    """Prove full resident scans miss even when prior scan suffix is cached.

    A native page-aligned, non-aliased resident weight extent is read once
    per repeated cohort, completely and in order. It spans >C pages per
    bank for a C-page LRU. In a repeated ordered cycle of N>C distinct
    pages, each page's reuse distance is N-1>=C, so every access misses.
    Intervening reads of disjoint extents can only evict more cycle pages.
    Other byte ranges cannot introduce a prefix cache hit because native
    page-aligned allocations do not overlap. Unknown or partial use keeps
    the ordinary cache phase; this proof removes no program or RMW cost.
    """
    if physical_runtime is None:
        return set()
    uses, candidates, declarations, future_aliases = Counter(), {}, {}, set()
    for stage in stages:
        for task in stage.execution_tasks:
            md = task.metadata
            for row in md.get("physical_allocations", ()):
                owner = row.get("physical_owner", md.get("physical_owner"))
                key = (owner, row.get("buffer_id"), row.get("generation", 0))
                prior = declarations.get(key)
                if prior is not None and prior != row:
                    # Conflicting native declarations cannot establish the
                    # complete extent, even if one happens to be resident.
                    declarations[key] = None
                elif key not in declarations:
                    declarations[key] = row
                if row.get("alias_of"):
                    future_aliases.add((owner, row["alias_of"], row.get("alias_generation", row.get("generation", 0))))
    for stage in stages:
        for task in stage.execution_tasks:
            md = task.metadata
            for access in md.get("memory_accesses", ()):
                owner = access.get("physical_owner", md.get("physical_owner"))
                identity = access.get("buffer_id")
                generation = access.get("allocation_generation", 0)
                allocator = physical_runtime.allocators.get(owner)
                active = physical_runtime.runtimes.get(owner)
                if allocator is None or active is None:
                    continue
                config = active.core.config
                if not isinstance(config, NandConfig) or not config.read_cache_pages_per_bank:
                    continue
                key = (owner, identity, generation)
                uses[key] += 1
                allocation = allocator.get_allocation(identity, generation) if identity else None
                declaration = declarations.get(key)
                if allocation is not None:
                    size, arena, alias = allocation.size_bytes, allocation.allocation_arena, allocation.alias_of
                    aligned = allocation.base_address % config.page_bytes == 0
                elif declaration is not None:
                    # Before first query, native weight buffers have only
                    # authored declarations. The original allocator gives
                    # every nonaliased extent its own page-aligned range;
                    # its exact future address is not needed for an
                    # address-phase-independent full-scan eviction proof.
                    size, arena, alias = declaration.get("size_bytes"), declaration.get("allocation_arena"), declaration.get("alias_of")
                    aligned = (type(size) is int and size > 0
                        and declaration.get("address", 0) % config.page_bytes == 0)
                else:
                    continue
                pages = size//config.page_bytes
                cache_banks = (config.channels*config.targets_per_channel*config.dies_per_target
                               *config.luns_per_die*config.planes_per_lun)
                if (access.get("operation") == "read" and generation == 0
                        and alias is None
                        and arena == "resident"
                        and allocator.alignment_bytes >= config.page_bytes
                        and aligned and size % config.page_bytes == 0
                        and access.get("offset_bytes", 0) == 0
                        and access.get("byte_count") == size
                        and md.get("weight_lifecycle_mode") == "preloaded_resident"
                        and md.get("weight_tensor_id")
                        and pages >= (config.read_cache_pages_per_bank+1)*cache_banks
                        and key not in future_aliases
                        and not allocator._aliases_by_target.get((identity, generation))):
                    candidates[key] = True
    return {key for key in candidates if uses[key] == 1}


def _native_swept_kv_reads(stages, physical_runtime, cold_weights):
    """A completed dominating resident scan evicts all older KV cache tags."""
    if physical_runtime is None or not cold_weights:
        return set()
    tasks = [task for stage in stages for task in stage.execution_tasks]
    owner_bits = {owner: 1 << index for index, owner in enumerate(sorted({row[0] for row in cold_weights}))}
    uses, unknown_owners, future_aliases = Counter(), set(), set()
    for task in tasks:
        md = task.metadata
        for access in md.get("memory_accesses", ()):
            owner = access.get("physical_owner", md.get("physical_owner"))
            identity = access.get("buffer_id")
            if access.get("operation") == "read":
                if not identity:
                    unknown_owners.add(owner)
                uses[(owner, identity, access.get("allocation_generation", 0))] += 1
        for row in md.get("physical_allocations", ()):
            if row.get("alias_of"):
                future_aliases.add((row.get("physical_owner", md.get("physical_owner")),
                    row["alias_of"], row.get("alias_generation", row.get("generation", 0))))
    inherited, result = {}, set()
    for task in tasks:
        before = 0
        for dependency in task.dependencies:
            before |= inherited.get(dependency, 0)
        md, completed = task.metadata, before
        for access in md.get("memory_accesses", ()):
            owner = access.get("physical_owner", md.get("physical_owner"))
            identity, generation = access.get("buffer_id"), access.get("allocation_generation", 0)
            key = (owner, identity, generation)
            if key in cold_weights:
                completed |= owner_bits[owner]
            if (access.get("operation") != "read" or not isinstance(identity, str)
                    or not identity.startswith("kv:") or generation != 0 or uses[key] != 1
                    or not before & owner_bits.get(owner, 0) or owner in unknown_owners
                    or key in future_aliases):
                continue
            contract = md.get("generic_kv_storage", {})
            if contract.get("schema") != "analytical-token-major-kv/v1":
                continue
            active, allocator = physical_runtime.runtimes.get(owner), physical_runtime.allocators.get(owner)
            if active is None or allocator is None:
                continue
            config, allocation = active.core.config, allocator.get_allocation(identity, generation)
            if (not isinstance(config, NandConfig) or config.protocol_mode != "generic_nand"
                    or config.write_buffer_bytes or config.wear_model is not None
                    or allocation is None or allocation.alias_of is not None
                    or allocation.allocation_arena != "resident"
                    or allocator.alignment_bytes < config.page_bytes
                    or allocation.base_address % config.page_bytes
                    or allocator._aliases_by_target.get((identity, generation))):
                continue
            result.add(key)
        inherited[task.task_id] = completed
    return result


def summarize_native_stages(stages, scenario, physical_runtime=None, *, allow_bounded_clock=False,
                            native_runtime=None, allocation_scope=None):
    """Reduce already-authored native stages before aggregate retention.

    The summary is built by serving from its private prepared tasks. Its
    quantities are recomputed; no metadata ``qualified`` flag is accepted.
    """
    from .contracts import _PreparedExecutionStage, _PreparedExecutionTask
    from .physical_contract import parse_execution_physical_config

    native_tasks = tuple(task for stage in stages if type(stage) is _PreparedExecutionStage
                         for task in stage.execution_tasks)
    exact_ancestry, ancestry_proof = _native_dag_ancestry(native_tasks)
    owner_tasks = defaultdict(list)
    # Classify every authored demand once, including ordinary Tasks with no
    # memory descriptors. Such a Task may write the same physical calendar
    # through an explicit CA/DQ resource (or an alias). It cannot be hidden
    # merely because the descriptor-only owner inventory looks ordered.
    physical_prefix_owners = defaultdict(set)
    if physical_runtime is not None:
        for owner, active in physical_runtime.runtimes.items():
            for name, value in active.core.config.metadata.items():
                if isinstance(value, str) and (name.endswith("resource_prefix")
                        or name in {"host_resource_id", "shared_command_resource"}):
                    physical_prefix_owners[value].add(owner)
        # Alias targets need the same physical identity. A NAND/SRAM port
        # aliased to a tensor label is still a replaced physical demand.
        aliased_physical_owners = defaultdict(set)
        for identity in physical_runtime.timeline.resource_aliases:
            resource = identity
            while resource:
                aliased_physical_owners[
                    physical_runtime.timeline.canonical_resource_id(identity)].update(
                        physical_prefix_owners.get(resource, ()))
                resource, separator, _tail = resource.rpartition(":")
                if not separator:
                    break
        for resource, owners in aliased_physical_owners.items():
            if owners:
                physical_prefix_owners[resource].update(owners)
    physical_prefixes = tuple(physical_prefix_owners)
    direct_port_writers = set()
    from .cost_models import GPUProfile
    compute_resources = {identity
        for profiles in getattr(scenario, "component_profiles", {}).values()
        for profile in profiles.values() if type(profile) is GPUProfile
        for identity in (profile.tensor_core.resource_id, profile.scalar_resource_id,
                         profile.special_function_resource_id)}
    hidden_physical_writers = False
    for task in native_tasks:
        if type(task) is not _PreparedExecutionTask:
            hidden_physical_writers = True
            continue
        md = task.metadata
        hidden_physical_writers |= any(md.get(key) for key in (
            "hbf_host_staging_by_owner", "hbf_host_staging", "stateful_l2", "l2_execution"))
        touched = {access.get("physical_owner", md.get("physical_owner"))
                   for access in md.get("memory_accesses", ())}
        if physical_runtime is not None:
            for demand in task.demands:
                resource = physical_runtime.timeline.canonical_resource_id(demand.resource_id)
                while resource:
                    direct_port_writers.update(physical_prefix_owners.get(resource, ()))
                    resource, separator, _tail = resource.rpartition(":")
                    if not separator:
                        break
        for owner in touched-{None}:
            owner_tasks[owner].append(task)
    owner_ranks = defaultdict(set)
    if hasattr(scenario, "placement"):
        from .ir import default_memory_resource_id
        for rank in scenario.placement.parallel.rank_mapping:
            component_ids = {rank.memory_component_id}
            component_ids.add(scenario.placement.metadata.get("rank_kv_components", {}).get(str(rank.rank)))
            for component_id in component_ids-{None}:
                component = scenario.hardware.get_component(component_id)
                owner_ranks[default_memory_resource_id(component)].add(rank.rank)

    cold_weights = _cold_resident_nand_weights(stages, physical_runtime)
    swept_kv = _native_swept_kv_reads(stages, physical_runtime, cold_weights)
    owners, allocated, finish, signature = {}, defaultdict(dict), {}, Counter()
    weight_finish, expert_signature = {}, set()
    parsed_configs = {}
    independent_channels = {}
    refresh_source_proofs = {}
    row_phase = refresh_phase = cache_phase = 0.
    hbm_clock_limits = []
    hbm_clock_comparison_reserves = []
    atomic_task_refresh_proofs = []
    atomic_task_phase_mask_proofs = []
    unknown, stateful_cache, task_count = [], False, 0
    if exact_ancestry is None and ancestry_proof["reason"] != "native_DAG_exceeds_bounded_layer_scope":
        unknown.append(ancestry_proof["reason"])
    cleanup = 0.
    stage_phase_bounds = []
    order_rows = []
    layer_row_reuse, io_row_guards, row_guard_overrides = None, {}, {}
    io_active_uppers, io_active_overrides = {}, {}
    if (allocation_scope is not None and allow_bounded_clock and native_runtime is not None
            and physical_runtime is not None):
        from .native_collective_row_reuse import native_layer_row_reuse
        hbm_configs, source_qualified = {}, {}
        for task in native_tasks:
            md = task.metadata
            for access in md.get("memory_accesses", ()):
                owner = access.get("physical_owner", md.get("physical_owner"))
                raw = md.get("physical_memory_configs", {}).get(owner, md.get("physical_memory_config"))
                config = raw
                if isinstance(raw, Mapping):
                    config = parsed_configs.get(id(raw))
                    if config is None:
                        config = parsed_configs[id(raw)] = parse_execution_physical_config(raw)
                if not (isinstance(config, DramConfig) and config.control_mode == "jedec_hbm"):
                    continue
                previous = hbm_configs.setdefault(owner, config)
                if previous != config:
                    source_qualified[owner] = False
                    continue
                key = (owner, id(config))
                if key not in refresh_source_proofs:
                    refresh_source_proofs[key] = _native_refresh_source_qualification(
                        config, owner, physical_runtime, native_runtime)
                source_qualified.setdefault(owner, refresh_source_proofs[key].get("qualified") is True
                    and _hbm_channels_independent(config, physical_runtime))
        layer_row_reuse = native_layer_row_reuse(native_tasks, exact_ancestry=exact_ancestry,
            physical_runtime=physical_runtime, allocation_scope=allocation_scope,
            configs=hbm_configs, source_qualified=source_qualified)
        io_row_guards = layer_row_reuse["io_guards"]
        io_active_uppers = layer_row_reuse["io_active_uppers"]
    graph_dependencies, graph_position = {}, {}
    reach_budget = max(64, 8*sum(len(stage.execution_tasks) for stage in stages))
    reach_examined = 0
    for stage in stages:
        if type(stage) is not _PreparedExecutionStage:
            unknown.append("non_native_prepared_stage")
            continue
        stage_cleanup = "output_cpu_commit" in stage.stage_id
        stage_sum, stage_paths, previous_resource, unordered = 0., {}, {}, set()
        private_rows = []
        refresh_bills = {}

        def precedes(previous, current):
            """Prove ordering from authored edges; never invent a queue order."""
            nonlocal reach_budget, reach_examined
            if exact_ancestry is not None:
                return _native_dag_precedes(exact_ancestry, previous, current.task_id)
            if previous == current:
                return True
            cutoff = graph_position[previous]
            pending, seen = list(current.dependencies), set()
            local_budget = 64
            while pending:
                if reach_budget <= 0 or local_budget <= 0:
                    return False
                reach_budget -= 1
                local_budget -= 1
                reach_examined += 1
                task_id = pending.pop()
                if task_id == previous:
                    return True
                if task_id in seen or graph_position.get(task_id, -1) < cutoff:
                    continue
                seen.add(task_id)
                pending.extend(graph_dependencies.get(task_id, ()))
            return False

        for task in stage.execution_tasks:
            task_count += 1
            if type(task) is not _PreparedExecutionTask:
                unknown.append("non_native_prepared_task")
                continue
            md = task.metadata
            graph_position[task.task_id] = len(graph_position)
            graph_dependencies[task.task_id] = task.dependencies
            task_phase, owner_resources = 0., set()
            stateful_cache |= md.get("stateful_l2") is not None
            sensitive = _context_sensitive(md)
            signature[(str(md.get("operator_class", "")),
                       str(md.get("model_operator_kind", "")), sensitive)] += 1
            # Demands start together. Physical demand service may be zero
            # in deferred IR, hence add a byte-rate lower bound explicitly.
            physical_resources = {str(access.get("resource_id", ""))
                                  for access in md.get("memory_accesses", ())}
            weight = max((d.service_ns for d in task.demands
                          if d.resource_id not in physical_resources), default=0.)
            invariant_weight = 0.
            physical_upper = 0.
            refresh_terms = defaultdict(list)
            task_physical_configs = {}
            task_refresh = 0.
            task_hbm_row = 0.
            refresh_qualified, other_physical = True, False
            nonphysical_upper = math.fsum(d.service_ns for d in task.demands
                                         if d.resource_id not in physical_resources)
            configs = md.get("physical_memory_configs", {})
            default_owner = md.get("physical_owner")
            default_config = md.get("physical_memory_config")
            for access_index, access in enumerate(md.get("memory_accesses", ())):
                owner = access.get("physical_owner", default_owner)
                config = configs.get(owner, default_config)
                if isinstance(config, Mapping):
                    original = config
                    config = parsed_configs.get(id(original))
                    if config is None:
                        config = parse_execution_physical_config(original)
                        parsed_configs[id(original)] = config
                count, operation = access.get("byte_count"), access.get("operation")
                if (not owner or type(count) is not int or count < 0
                        or operation not in {"read", "write"}
                        or not isinstance(config, (DramConfig, NandConfig, SramConfig))):
                    unknown.append("unresolved_native_physical_access")
                    continue
                task_physical_configs[owner] = config
                entry = owners.setdefault(owner, {"read_count": 0, "write_count": 0,
                    "read_bytes": 0, "write_bytes": 0, "context_read_bytes": 0,
                    "row_phase_ns": 0., "cache_phase_ns": 0., "refresh_cycle_ns": 0.,
                    "refresh_active_io_serial_upper_ns": 0.})
                entry[operation+"_count"] += 1
                entry[operation+"_bytes"] += count
                if sensitive and operation == "read":
                    entry["context_read_bytes"] += count
                try:
                    bandwidth = _bandwidth(config, operation)
                    key = id(config)
                    if key not in independent_channels:
                        independent_channels[key] = (isinstance(config, DramConfig)
                            and _hbm_channels_independent(config, physical_runtime))
                    source_key = (owner, key)
                    if (allow_bounded_clock and isinstance(config, DramConfig)
                            and config.control_mode == "jedec_hbm"
                            and (config.refresh_policy == "all_bank" or native_runtime is not None)
                            and source_key not in refresh_source_proofs):
                        refresh_source_proofs[source_key] = _native_refresh_source_qualification(
                            config, owner, physical_runtime, native_runtime)
                    source_proof = refresh_source_proofs.get(source_key, {})
                    native_refresh = source_proof.get("qualified") is True
                    if source_key in refresh_source_proofs:
                        entry["native_refresh_source_proof"] = source_proof
                        if native_runtime is not None and not native_refresh:
                            unknown.append("native_refresh_source_not_closed:"+source_proof["reason"])
                    row, refresh, cache = _phase(config, count, operation,
                        independent_hbm_channels=independent_channels[key],
                        allow_bounded_clock=allow_bounded_clock,
                        native_refresh_reachable=native_refresh)
                    if isinstance(config, DramConfig) and config.control_mode == "jedec_hbm":
                        from .native_hbm_cold_bound import native_hbm_initial_row_upper
                        initial = native_hbm_initial_row_upper(config, count, operation,
                            independent_channels=independent_channels[key],
                            native_refresh_reachable=_native_row_source_proven(config, count, operation,
                                independent_channels=independent_channels[key],
                                allow_bounded_clock=allow_bounded_clock,
                                native_refresh_reachable=native_refresh))
                        if initial.get("untouched_old_row_expiry_excluded"):
                            # This is a row proof dependency even when the
                            # active REF envelope chooses a different upper.
                            hbm_clock_limits.append(initial["hbm_clock_domain_limit_ns"])
                            hbm_clock_comparison_reserves.append(
                                initial["hbm_clock_domain_comparison_reserve_ns"])
                        initial_proofs = entry.setdefault("native_initial_row_bounds", [])
                        if not any(proof["byte_count"] == count and proof["operation"] == operation
                                   for proof in initial_proofs):
                            initial_proofs.append({"byte_count": count, "operation": operation,
                                "applied_row_upper_ns": row, **initial})
                except ValueError as error:
                    unknown.append(str(error))
                    continue
                guard = io_row_guards.get((task.task_id, access_index))
                if guard is not None and isinstance(config, DramConfig) and guard < row:
                    row = guard
                    row_guard_overrides[task.task_id] = None
                active_nc = io_active_uppers.get((task.task_id, access_index))
                if active_nc is not None and isinstance(config, DramConfig):
                    io_active_overrides[task.task_id] = max(active_nc, io_active_overrides.get(task.task_id, 0.))
                    nc_terms = _refresh_serial_terms(config, count, operation,
                        independent_channels=independent_channels[key],
                        allow_bounded_clock=allow_bounded_clock, native_refresh_reachable=native_refresh)
                    if nc_terms is not None:
                        interval_nc, blocked_nc, active_old = nc_terms
                        refresh = min(refresh, (1+math.floor(min(active_old, active_nc)
                            /(interval_nc-blocked_nc)))*blocked_nc)
                access_key = (owner, access.get("buffer_id"), access.get("allocation_generation", 0))
                if operation == "read" and isinstance(config, NandConfig):
                    if access_key in cold_weights or access_key in swept_kv:
                        cache = 0.
                    else:
                        release_phase = _nand_cache_release_phase(config, count, physical_runtime)
                        if release_phase is not None:
                            cache = min(cache, release_phase)
                elif operation == "write" and _native_fixed_kv_partial_write(config, access, md, physical_runtime):
                    row = 0.
                    entry["fixed_kv_rmw_bulk_access_count"] = entry.get("fixed_kv_rmw_bulk_access_count", 0)+1
                weight = max(weight, count/bandwidth)
                if (operation == "read" and access.get("allocation_generation", 0) == 0
                        and md.get("weight_tensor_id")
                        and md.get("weight_lifecycle_mode") == "preloaded_resident"
                        and access.get("buffer_id", "").startswith("tensor:")):
                    invariant_weight = max(invariant_weight, count/bandwidth)
                entry["row_phase_ns"] += row
                entry["cache_phase_ns"] += cache
                entry["refresh_cycle_ns"] = max(entry["refresh_cycle_ns"],
                    getattr(config, "refresh_cycle_ns", 0.))
                entry["refresh_active_io_serial_upper_ns"] += refresh
                task_refresh += refresh
                if isinstance(config, DramConfig) and config.refresh_policy == "all_bank":
                    task_hbm_row += row
                    refresh_qualified &= independent_channels[key]
                    terms = _refresh_serial_terms(config, count, operation,
                        independent_channels=independent_channels[key],
                        allow_bounded_clock=allow_bounded_clock,
                        native_refresh_reachable=native_refresh)
                    if terms is not None:
                        refresh_terms[owner].append(terms if active_nc is None
                            else (*terms[:2], min(terms[2], active_nc)))
                    cold = (_native_cold_ref_service(config, count, operation,
                        independent_channels=independent_channels[key],
                        native_refresh_reachable=native_refresh) if allow_bounded_clock else None)
                    if (cold is not None and terms is not None
                            and (terms[2] == cold["conditional_no_refresh_serial_upper_ns"]
                                or cold.get("qualified_source_required")
                                or cold.get("native_REF_on_source_clock_retained"))):
                        reserve = cold["hbm_clock_domain_comparison_reserve_ns"]
                        # A padded counterfactual may finish after its real
                        # IO. Reserve that whole comparison path too, so a
                        # final lifecycle edge just below the raw tick limit
                        # cannot rely on a proof that crosses it.
                        hbm_clock_limits.append(cold["hbm_clock_domain_limit_ns"])
                        hbm_clock_comparison_reserves.append(reserve)
                        proof_name = ("native_REF_on_service_bounds" if cold.get("qualified_source_required")
                                      else "native_no_refresh_cold_bounds")
                        cold_proofs = entry.setdefault(proof_name, [])
                        if not any(row["byte_count"] == count and row["operation"] == operation
                                   for row in cold_proofs):
                            cold_proofs.append({"byte_count": count, "operation": operation, **cold})
                else:
                    other_physical = True
                physical_upper += count/bandwidth + row + cache + refresh
                task_phase += row+cache+refresh
                owner_resources.add("@physical-owner:"+owner)
                # Different logical owners can explicitly share a physical
                # command/data/array prefix. Such ranks are not independent
                # even when their top-level resource names differ.
                for name in ("bank_resource_prefix", "command_resource_prefix",
                        "row_command_resource_prefix", "column_command_resource_prefix",
                        "data_resource_prefix", "array_resource_prefix", "parallel_resource_prefix",
                        "channel_resource_prefix", "host_resource_id", "port_resource_prefix"):
                    prefix = config.metadata.get(name)
                    if isinstance(prefix, str) and prefix:
                        owner_resources.add("@physical-prefix:"+prefix)
            if task.task_id in row_guard_overrides:
                row_guard_overrides[task.task_id] = task_hbm_row
            grouped_refresh, atomic = _atomic_task_refresh_bound(task, dict(refresh_terms),
                task_physical_configs, physical_runtime, task_refresh,
                independent_channels=refresh_qualified)
            if atomic is not None:
                task_phase -= task_refresh-grouped_refresh
                task_refresh = grouped_refresh
                atomic_task_refresh_proofs.append(atomic)
            if task_hbm_row and allow_bounded_clock and native_runtime is not None:
                mask = _native_task_hbm_phase_mask(task, task_physical_configs,
                    physical_runtime, native_runtime, source_proofs=refresh_source_proofs,
                    owner_tasks=owner_tasks, exact_ancestry=exact_ancestry,
                    direct_port_writers=direct_port_writers, compute_resources=compute_resources,
                    physical_prefixes=physical_prefixes,
                    hidden_physical_writers=hidden_physical_writers)
                if mask["qualified"]:
                    removed = task_hbm_row+task_refresh
                    task_phase = max(0., task_phase-removed)
                    task_refresh = 0.
                    refresh_terms.clear()
                    mask["removed_HBM_phase_ns"] = removed
                    hbm_clock_limits.append(mask["hbm_clock_domain_limit_ns"])
                    hbm_clock_comparison_reserves.append(mask["hbm_clock_domain_comparison_reserve_ns"])
                atomic_task_phase_mask_proofs.append(mask)
            for allocation in md.get("physical_allocations", ()):
                owner = allocation.get("physical_owner", default_owner)
                if (owner and type(allocation.get("size_bytes")) is int
                        and allocation.get("allocation_arena") != "resident"
                        and allocation.get("generation", 0) > 0):
                    identity = (str(allocation.get("buffer_id")), allocation.get("generation", 0))
                    allocated[owner][identity] = max(allocated[owner].get(identity, 0), allocation["size_bytes"])
            predecessor = max((finish.get(dep, 0.) for dep in task.dependencies), default=0.)
            finish[task.task_id] = predecessor + (0. if sensitive else weight)
            weight_finish[task.task_id] = max((weight_finish.get(dep, 0.)
                for dep in task.dependencies), default=0.)+invariant_weight
            if md.get("routing_model") == "deterministic_balanced_histogram" and "expert_index" in md:
                expert_signature.add((str(md.get("layer_id")), md.get("rank"),
                    md["expert_index"], md.get("expert_tokens"), md.get("routed_tokens")))
            resources = owner_resources | {d.resource_id for d in task.demands if d.service_ns > 0.}
            if physical_runtime is not None:
                canonical = physical_runtime.timeline.canonical_resource_id
                resources = {canonical(resource) if not resource.startswith("@physical-owner:")
                             else resource for resource in resources}
            private_rows.append((task, task_phase, resources))
            order_rows.append((stage.stage_id, task, resources))
            refresh_bills[task.task_id] = {"refresh_ns": task_refresh,
                "terms": dict(refresh_terms), "qualified": refresh_qualified and not other_physical,
                "other_physical": other_physical, "nonphysical_upper_ns": nonphysical_upper}
            for resource in resources:
                previous = previous_resource.get(resource)
                if previous is not None and not precedes(previous, task):
                    unordered.add(resource)
                previous_resource[resource] = task.task_id
            stage_sum += task_phase
            stage_paths[task.task_id] = max((stage_paths.get(dep, 0.)
                for dep in task.dependencies), default=0.)+task_phase
            if stage_cleanup:
                cleanup += max(weight, physical_upper)
        path = max(stage_paths.values(), default=0.)
        private_bound, blocks = _rank_private_phase_bound(private_rows, owner_ranks=owner_ranks)
        segment_bound, segment_proof = _busy_segment_phase_bound(private_rows, refresh_bills,
            owner_ranks=owner_ranks, exact_ancestry=exact_ancestry)
        from .native_collective_refresh_bound import native_collective_refresh_bound
        # Closed private-rank REF segments and closed collectives are disjoint
        # contractions; feed the merged segment REF into the collective bound.
        merged = segment_proof.get("segment_phase_by_task", {})
        collective_rows = [(task, merged.get(task.task_id, phase), resources)
                           for task, phase, resources in private_rows] if merged else private_rows
        collective_bound, collective_proof = native_collective_refresh_bound(
            collective_rows, refresh_bills, physical_runtime=physical_runtime,
            native_runtime=native_runtime if allow_bounded_clock else None,
            owner_ranks=owner_ranks, exact_ancestry=exact_ancestry,
            rank_bound=_rank_private_phase_bound, all_tasks=native_tasks,
            parsed_configs=parsed_configs, source_proofs=refresh_source_proofs,
            allocation_scope=allocation_scope, row_guard_overrides=row_guard_overrides,
            io_active_overrides=io_active_overrides)
        if collective_proof.get("qualified"):
            hbm_clock_limits.append(collective_proof["hbm_clock_domain_limit_ns"])
        # If every shared resource already follows authored DAG edges,
        # resource arbitration adds no order not present in this DAG. The
        # max-plus Lipschitz bound is its phase-weighted longest path. Any
        # incomparable queue keeps the full serial sum for this stage.
        stage_phase_bounds.append({"stage_id": stage.stage_id,
            "serial_access_phase_ns": stage_sum, "dependency_path_phase_ns": path,
            "applied_phase_ns": min(collective_bound, segment_bound, private_bound,
                                    stage_sum if unordered else path),
            "closed_hbm_busy_segments": segment_proof,
            "closed_collective_refresh": collective_proof,
            "resource_orders_proven": not unordered,
            "incomparable_resources": sorted(unordered),
            "rank_private_blocks": blocks,
            "basis": "all_demanded_and_physical_owner_resource_accesses_ordered_by_native_dag"})
    for row in owners.values():
        row_phase += row["row_phase_ns"]
        cache_phase += row["cache_phase_ns"]
        refresh_phase += row["refresh_cycle_ns"]
    from .native_issue_order import native_issue_order_proof
    issue_order = native_issue_order_proof(stages, order_rows, physical_runtime,
        native_runtime if allow_bounded_clock else None)
    return {"schema": "native-window-ir-observations/v1", "task_count": task_count,
        "native_issue_order_proof": issue_order,
        "fixed_path_floor_ns": max(finish.values(), default=0.),
        "fixed_path_basis": "native_task_dag_with_context_sensitive_task_services_removed",
        "batch_invariant_weight_read_floor_ns": max(weight_finish.values(), default=0.),
        "batch_invariant_weight_floor_basis": "native_preloaded_generation_zero_tensor_read_bytes_over_directional_bandwidth_only",
        "owner_accesses": owners, "row_phase_per_cohort_ns": row_phase,
        "cache_phase_per_cohort_ns": cache_phase,
        "ordered_resource_phase_per_cohort_ns": sum(row["applied_phase_ns"] for row in stage_phase_bounds),
        "hbm_clock_domain_limit_ns": min(hbm_clock_limits, default=None),
        "hbm_clock_domain_comparison_reserve_ns": max(hbm_clock_comparison_reserves, default=0.),
        "atomic_task_refresh_proofs": atomic_task_refresh_proofs,
        "atomic_task_phase_mask_proofs": atomic_task_phase_mask_proofs,
        # Same-class layers must share the real allocator layout behind any
        # applied row-reuse proof; collective names alone differ per layer.
        "collective_row_reuse_layouts": sorted(
            (group["closed_collective_row_reuse"].get("applied") is True,
             repr(group["closed_collective_row_reuse"].get("address_layout_signature")))
            for row in stage_phase_bounds
            for group in row["closed_collective_refresh"].get("collectives", ())) + (
            [] if layer_row_reuse is None else [(True, repr(sorted(
                (owner, row["address_layout_sha256"]) for owner, row in layer_row_reuse["owners"].items())))]),
        "native_layer_row_reuse": None if layer_row_reuse is None else {
            key: layer_row_reuse[key] for key in ("owners", "rejections", "basis", "scope")}
            | {"reduced_task_count": len(row_guard_overrides), "reduced_io_guard_count": len(io_row_guards)},
        "native_resource_order_proof": stage_phase_bounds,
        "native_DAG_ancestry_proof": ancestry_proof,
        "ordering_reachability_nodes_examined": reach_examined,
        "cold_resident_weight_cache_proof": {"qualified_scan_count": len(cold_weights),
            "basis": "one_full_page_aligned_nonaliased_resident_weight_scan_per_cohort; cyclic_LRU_reuse_distance_at_least_cache_capacity_per_bank; native_declaration_before_first_allocation_is_sufficient",
            "scope": "repeated_fixed_native_dense_cohort; first observed cohort remains real; no latency or payload removed"},
        "dominating_resident_scan_KV_cache_proof": {"qualified_read_count": len(swept_kv),
            "basis": "native_DAG_strictly_prior_complete_owner_scan_replaces_all_LRU_tags; nonaliased_resident_KV_extent_read_once; generic_KV_writes_invalidate_without_populating_read_cache"},
        "native_nand_cache_phase_basis": "cyclic_resident_scan_proof_else_native_independent_array_and_channel_fill_release_delay; aliased_unknown_buses_keep_serial_bound; host_payload_credits_program_and_RMW_retained",
        "fixed_native_KV_RMW_difference_basis": "only_one_token_major_row_dividing_a_page_in_real_nonaliased_page_aligned_resident_allocation; every_future_integer_row_is_one_identical_paid_partial_RMW_and_PROGRAM; unknown_geometry_retains_two_edge_page_guard",
        "refresh_initial_phase_ns": refresh_phase,
        "refresh_active_io_serial_upper_ns": sum(row["refresh_active_io_serial_upper_ns"] for row in owners.values()),
        "refresh_bound_basis": "native_contiguous_channel_burst_upper_only_after_bus_independence_proof_else_full_serial_bursts; full_gate_sum_and_initial_PRE; residual_epoch_plus_REFI_minus_blocked_feedback; nominal_refresh_double_count_retained",
        "allocation_declared_bytes_by_owner": {key: sum(rows.values()) for key, rows in allocated.items()},
        "output_commit_cleanup_upper_ns": cleanup,
        "operator_signature": [[*key, count] for key, count in sorted(signature.items())],
        "native_balanced_expert_signature": [list(row) for row in sorted(expert_signature, key=str)],
        "stateful_gpu_cache": stateful_cache, "unsupported": sorted(set(unknown)),
        "physical_state_advanced": False}


def _busy_segment_phase_bound(rows, bills, *, owner_ranks, exact_ancestry=None):
    """Contract closed native rank subgraphs and charge one REF arrival.

    Every contracted group gains a completion barrier, so work cannot leave
    the group early. External dependencies must precede its first task;
    this rules out an unbounded foreign wait in its active REF horizon.
    Arbitrary branches *inside* the group are serialized conservatively.
    Shared resources, unknown rank/owner and non-HBM physical operations
    keep their original boundary and per-access envelope.
    """
    from types import SimpleNamespace

    if len(rows) > 4096:
        # Layer windows are bounded subgraphs. A whole-model IR need not
        # perform a quadratic cross-boundary proof merely to obtain an
        # optional tighter envelope; its existing serial guard remains.
        return math.inf, {"segments": [], "qualified": False,
            "reason": "closed_segment_proof_graph_exceeds_bounded_layer_scope",
            "physical_state_advanced": False}
    if exact_ancestry is None:
        exact_ancestry, ancestry_proof = _native_dag_ancestry(tuple(row[0] for row in rows))
        if exact_ancestry is None:
            return math.inf, {"segments": [], "qualified": False,
                "reason": ancestry_proof["reason"], "physical_state_advanced": False}
    tasks = {task.task_id: task for task, _phase, _resources in rows}
    rank_by_id, resource_ranks = {}, defaultdict(set)
    for task, phase, resources in rows:
        rank = task.metadata.get("rank")
        touched = {resource.removeprefix("@physical-owner:") for resource in resources
                   if resource.startswith("@physical-owner:")}
        if type(rank) is not int:
            assigned = set().union(*(owner_ranks.get(owner, ()) for owner in touched)) if touched else set()
            if len(assigned) == 1 and all(len(owner_ranks.get(owner, ())) == 1 for owner in touched):
                rank = assigned.pop()
        if type(rank) is not int and not phase and not resources:
            incoming = {rank_by_id.get(identity) for identity in task.dependencies}
            if len(incoming) == 1 and None not in incoming:
                rank = incoming.pop()
        rank = rank if type(rank) is int else None
        rank_by_id[task.task_id] = rank
        for resource in resources:
            resource_ranks[resource].add(rank)

    def before_first(identity, first):
        return _native_dag_precedes(exact_ancestry, identity, first.task_id)

    pending, groups = defaultdict(list), []
    def close(rank):
        group = pending.pop(rank, None)
        if group:
            groups.append(group)
    def close_all():
        for rank in tuple(pending):
            close(rank)

    for position, row in enumerate(rows):
        task, phase, resources = row
        rank, bill = rank_by_id[task.task_id], bills[task.task_id]
        private = (rank is not None and all(resource_ranks[resource] == {rank} for resource in resources))
        entry = (position, task, phase, resources, bill)
        if not private:
            close_all()
            groups.append([entry])
            continue
        current = pending.get(rank, ())
        if current:
            ids = {item[1].task_id for item in current}
            # An incoming dependency outside this group may wait for an
            # arbitrarily long unrelated native branch. Starting a new
            # segment pays a new residual REF instead of hiding that wait.
            # Its first Task must also precede every later member: an
            # independent root could execute before that first Task's long
            # external wait, producing two separated residual REF epochs.
            if (not before_first(current[0][1].task_id, task)
                    or any(parent not in ids and not before_first(parent, current[0][1])
                           for parent in task.dependencies)):
                close(rank)
        if bill["other_physical"]:
            close(rank)
            groups.append([entry])
        else:
            pending[rank].append(entry)
    close_all()
    groups.sort(key=lambda group: group[0][0])
    owner_tasks = defaultdict(list)
    for task, _phase, _resources in rows:
        for owner in bills[task.task_id]["terms"]:
            owner_tasks[owner].append(task)
    replacements = {item[1].task_id: "@native-ref-segment:"+group[0][1].task_id
                    for group in groups for item in group}
    contracted, proof, segment_phases = [], [], {}
    for group in groups:
        identity = replacements[group[0][1].task_id]
        old_phase = math.fsum(item[2] for item in group)
        old_ref = math.fsum(item[4]["refresh_ns"] for item in group)
        owner_terms = defaultdict(list)
        qualified = True
        for _position, _task, _phase_value, _resources, bill in group:
            qualified &= bill["qualified"]
            for owner, terms in bill["terms"].items():
                owner_terms[owner].extend(terms)
        nonphysical_upper = math.fsum(item[4]["nonphysical_upper_ns"] for item in group)
        new_ref = 0.
        if len(owner_terms) != 1:
            qualified = False
        resources = set().union(*(item[3] for item in group))
        rank = rank_by_id[group[0][1].task_id]
        qualified &= rank is not None and all(resource_ranks[resource] == {rank} for resource in resources)
        group_ids = {item[1].task_id for item in group}
        roots = [item[1] for item in group
                 if not any(parent in group_ids for parent in item[1].dependencies)]
        used = {parent for item in group for parent in item[1].dependencies if parent in group_ids}
        terminals = [item[1] for item in group if item[1].task_id not in used]
        # An artificial barrier is conservative for a DAG bound, but it
        # cannot erase physical accesses that the original graph allows to
        # interleave with this controller. Prove isolation on that original
        # graph before sharing even the first residual REF across requests.
        isolated = all(
            all(before_first(outside.task_id, root) for root in roots)
            or all(before_first(terminal.task_id, outside) for terminal in terminals)
            for owner in owner_terms for outside in owner_tasks[owner]
            if outside.task_id not in group_ids)
        qualified &= isolated
        for terms in owner_terms.values():
            if len({(interval, blocked) for interval, blocked, _active in terms}) != 1:
                qualified = False
                continue
            interval, blocked = terms[0][:2]
            # Summing the complete native gate upper for each access and
            # every intervening nonphysical demand bounds the whole closed
            # group's no-REF span. This retains RAW and command ordering.
            active_upper = nonphysical_upper+math.fsum(active for _i, _b, active in terms)
            new_ref += (1+math.floor(active_upper/(interval-blocked)))*blocked
        new_ref = min(old_ref, new_ref) if qualified else old_ref
        phase = old_phase-old_ref+new_ref
        if qualified and new_ref < old_ref:
            # Every member is private to one rank, and a rank's tasks form one
            # serial chain in each private block. Charging the merged REF to
            # the first member therefore reaches every later member and every
            # edge leaving the segment, exactly like the contracted node.
            for index, (_position, task, member_phase, _resources, bill) in enumerate(group):
                segment_phases[task.task_id] = member_phase-bill["refresh_ns"]+(new_ref if index == 0 else 0.)
        dependencies = tuple(dict.fromkeys(replacements.get(parent, parent)
            for item in group for parent in item[1].dependencies
            if replacements.get(parent, parent) != identity))
        node = SimpleNamespace(task_id=identity, dependencies=dependencies,
                               metadata={} if rank is None else {"rank": rank})
        contracted.append((node, phase, resources))
        if old_ref:
            proof.append({"first_task_id": group[0][1].task_id,
                "task_count": len(group), "rank": rank, "qualified": qualified,
                "old_per_access_refresh_upper_ns": old_ref,
                "segment_refresh_upper_ns": new_ref,
                "nonphysical_serial_upper_ns": nonphysical_upper,
                "original_same_owner_IO_isolated_by_native_DAG": isolated,
                "added_complete_group_barrier": True,
                "basis": "closed_native_DAG_private_rank_segment_full_gate_serial_upper_and_one_residual_REF_plus_feedback"})
    bound, rank_proof = _rank_private_phase_bound(contracted, owner_ranks=owner_ranks)
    return bound, {"segments": proof, "contracted_rank_blocks": rank_proof,
                   "segment_phase_by_task": segment_phases, "physical_state_advanced": False}


def _rank_private_phase_bound(rows, *, owner_ranks=None):
    """Serial sum across blocks, max only across isolated native rank graphs.

    A block ends at every task without a native rank identity. This retains
    ring/network/HOST tasks as explicit synchronization separators. Rank
    subgraphs may have arbitrary internal branches: their entire phase sum
    is retained. Cross-rank edges or shared resources invalidate the max.
    """
    total, proof, pending, known_ranks = 0., [], [], {}

    def close():
        nonlocal total
        if not pending:
            return
        by_rank, task_ranks, resource_ranks = Counter(), {}, defaultdict(set)
        for task, phase, resources, rank in pending:
            by_rank[rank] += phase
            task_ranks[task.task_id] = rank
            for resource in resources:
                resource_ranks[resource].add(rank)
        shared = sorted(resource for resource, ranks in resource_ranks.items() if len(ranks) > 1)
        cross = any(task_ranks.get(dependency, rank) != rank
            for task, _phase, _resources, rank in pending for dependency in task.dependencies)
        serial, parallel = sum(by_rank.values()), max(by_rank.values(), default=0.)
        # Keep all native cross-rank dependencies and additionally serialize
        # each private rank in authored topological order. This only removes
        # overlap inside that rank; it never discards a network/barrier edge.
        # With no resource shared across ranks, no hidden arbitration edge
        # can connect these chains. Arbitrary within-rank branch contention
        # is covered by the complete chain, rather than assumed absent.
        paths, rank_frontiers = {}, {}
        for task, phase, _resources, rank in pending:
            predecessor = max((paths.get(dep, 0.) for dep in task.dependencies), default=0.)
            paths[task.task_id] = phase+max(predecessor, rank_frontiers.get(rank, 0.))
            rank_frontiers[rank] = paths[task.task_id]
        chained = max(paths.values(), default=0.)
        qualified = not shared
        applied = chained if qualified else serial
        total += applied
        proof.append({"rank_count": len(by_rank), "task_count": len(pending),
            "serial_phase_ns": serial, "applied_phase_ns": applied,
            "private_rank_chain_native_cross_edge_path_ns": chained,
            "qualified": qualified, "shared_resources": shared,
            "cross_rank_dependencies": cross,
            "basis": "private_rank_serial_chains_with_all_native_cross_rank_edges_between_unranked_tasks"})
        pending.clear()

    unranked = 0.
    for task, phase, resources in rows:
        rank = task.metadata.get("rank")
        if type(rank) is not int and owner_ranks is not None:
            # A communication endpoint that touches one explicitly private
            # memory owner belongs to that rank; the network/link Task is
            # still unranked and remains an explicit separator.
            touched = {resource.removeprefix("@physical-owner:")
                       for resource in resources if resource.startswith("@physical-owner:")}
            assigned = set().union(*(owner_ranks.get(owner, ()) for owner in touched)) if touched else set()
            if len(assigned) == 1 and all(len(owner_ranks.get(owner, ())) == 1 for owner in touched):
                rank = assigned.pop()
        if type(rank) is not int and not phase and not resources:
            # Pure zero-time markers inside one rank are not a shared
            # synchronization boundary. Derive their rank only when every
            # known incoming edge selects that same private graph.
            incoming = {known_ranks.get(dep) for dep in task.dependencies}
            if len(incoming) == 1 and None not in incoming:
                rank = incoming.pop()
        known_ranks[task.task_id] = rank if type(rank) is int else None
        if type(rank) is int:
            pending.append((task, phase, resources, rank))
        else:
            close()
            total += phase
            unranked += phase
    close()
    if unranked:
        proof.append({"rank_count": 0, "qualified": True,
            "applied_phase_ns": unranked, "basis": "unranked_communication_and_shared_host_serial_sum"})
    return total, proof


def _allocation_evidence(runtime, summaries, *, resident_declarations=()):
    """Check future resident declarations with the native allocator itself.

    This clones only address metadata. It never snapshots or submits a Core.
    Keeping every authored request allocated simultaneously is stricter than
    the bounded native serving lifecycle and its pending ingress slots.
    """
    from .memory_allocator import AllocationError, PhysicalAddressAllocator
    from .planner import native_kv_ingress_inventory

    physical = runtime._execution_resource_kernel.physical_runtime
    logical_pages = sum(runtime.ledger.pages_for_tokens(request.prompt_tokens+request.output_tokens-1)
                        for request in runtime.plan.requests)
    if logical_pages > runtime.plan.kv_policy.capacity_pages:
        return {"qualified": False, "reason": "all_authored_logical_kv_exceeds_native_page_capacity",
                "required_pages": logical_pages, "capacity_pages": runtime.plan.kv_policy.capacity_pages}
    clones = {owner: PhysicalAddressAllocator.from_snapshot(allocator.snapshot())
              for owner, allocator in physical.allocators.items()}
    future_bytes, workspace, resident_bytes, resident_keys = Counter(), Counter(), Counter(), set()
    try:
        # Layer windows begin before the first complete query, so source
        # allocators may not contain any model weights yet. Register the
        # entire native graph's real resident declarations in address-only
        # clones; selected physical layers alone cannot establish capacity.
        pending = list(resident_declarations)
        while pending:
            deferred = []
            for declaration in pending:
                owner = declaration["physical_owner"]
                clone = clones.get(owner)
                if clone is None:
                    return {"qualified": False, "reason": "resident_owner_has_no_native_allocator"}
                alias = declaration.get("alias_of")
                alias_generation = declaration.get("alias_generation", declaration.get("generation", 0))
                if alias and clone.get_allocation(alias, alias_generation) is None:
                    deferred.append(declaration)
                    continue
                clone.register(dict(declaration))
                key = (owner, declaration["buffer_id"], declaration.get("generation", 0))
                if key not in resident_keys:
                    resident_keys.add(key)
                    resident_bytes[owner] += declaration["size_bytes"]
            if len(deferred) == len(pending):
                raise ValueError("resident_native_alias_target_unavailable")
            pending = deferred
        for request in runtime.plan.requests:
            inventory = native_kv_ingress_inventory(runtime.plan.scenario,
                request.request_id, request.prompt_tokens-1)
            for declaration in inventory["physical_allocations"]:
                owner = declaration["physical_owner"]
                clone = clones.get(owner)
                if clone is None:
                    return {"qualified": False, "reason": "future_owner_has_no_native_allocator"}
                clone.register(dict(declaration))
                future_bytes[owner] += declaration["size_bytes"]
        # Native workspace is closed per cohort and lives in its own arena
        # where configured. The all-declaration sum is a conservative peak,
        # not an estimate of live activation tensors.
        for summary in summaries:
            for owner, size in summary["allocation_declared_bytes_by_owner"].items():
                workspace[owner] = max(workspace[owner], size)
        max_context = max(row.prompt_tokens+row.output_tokens-1 for row in runtime.plan.requests)
        min_context = min(row.prompt_tokens for row in runtime.plan.requests)
        expansion = max_context/min_context
        for owner, clone in clones.items():
            extra = math.ceil(workspace[owner]*expansion)
            if extra:
                clone.allocate("@native-window-workspace-upper", extra,
                    allocation_arena="workspace" if clone.workspace_capacity_bytes else None)
    except (AllocationError, KeyError, ValueError) as error:
        return {"qualified": False, "reason": str(error),
                "basis": "native_allocator_all_authored_kv_plus_closed_cohort_workspace"}
    return {"qualified": True, "physical_workspace_and_allocator_fragmentation_covered": True,
        "basis": "native_allocator_all_authored_kv_plus_closed_cohort_workspace",
        "future_kv_declaration_bytes_by_owner": dict(future_bytes),
        "full_native_resident_declaration_bytes_by_owner": dict(resident_bytes),
        "full_native_resident_declaration_count": len(resident_keys),
        "workspace_all_declaration_bytes_by_owner": dict(workspace),
        "context_workspace_expansion_upper": expansion,
        "physical_state_advanced": False}


def _native_pd_evidence(runtime):
    """Bound future native chunk imports from their own memory geometry.

    This reads the original inventory and executor ledger only. Query
    compute/weight phases cannot stand in for source/read, relay or target
    write initial-state costs. Different authored prompt shapes expand the
    measured bulk span by both bytes and chunk population, retaining every
    future request's separate memory phase in the serial uncertainty sum.
    """
    from .planner import native_kv_ingress_inventory

    executor, config = runtime._external_executor, runtime._external_ingress
    starts, completed = {}, []
    for event in runtime.events:
        if event.event_type == "external_kv_import_start":
            starts[event.request_id] = event.timestamp_ns
        elif event.event_type == "external_kv_import_complete" and event.request_id in starts:
            completed.append((event.request_id, event.timestamp_ns-starts[event.request_id]))
    if executor is None or not completed:
        return {"qualified": False, "reason": "native_paid_PD_sample_unavailable"}
    chunk = min(config.get("chunk_bytes", 1024**2),
        config.get("buffer_bytes", 64*1024**2)//config.get("max_inflight_requests", 2))
    component_by_owner = {service.physical_owner: component
        for component, service in executor.services.items()}
    by_id = {request.request_id: request for request in runtime.plan.requests}
    by_shape = {}

    def geometry(request):
        shape = (request.prompt_tokens, request.output_tokens)
        if shape in by_shape:
            return by_shape[shape]
        inventory = native_kv_ingress_inventory(runtime.plan.scenario,
            request.request_id, request.prompt_tokens-1,
            execution_control=runtime.execution_control)
        allocations_by_id = defaultdict(list)
        for declaration in inventory["physical_allocations"]:
            allocations_by_id[declaration["buffer_id"]].append(declaration)
        # The native contracts already contain the exact logical/physical
        # clipped history ranges. Do not invent a second KV layout.
        extents = defaultdict(set)
        for contract in inventory["contracts"]:
            row_bytes = int(contract["row_bytes"])
            for owner in contract["owners"]:
                if owner["request_id"] != request.request_id:
                    continue
                physical = owner.get("read_ranges", ())
                logical = owner.get("logical_read_ranges", physical)
                if len(physical) != len(logical):
                    raise ValueError("native_PD_range_mapping_unavailable")
                for (logical_start, logical_length), (physical_start, physical_length) in zip(logical, physical):
                    if logical_length != physical_length:
                        raise ValueError("native_PD_range_mapping_inconsistent")
                    lo, hi = max(0, logical_start), min(request.prompt_tokens-1, logical_start+logical_length)
                    if hi <= lo:
                        continue
                    for part in ("k", "v"):
                        identity = owner["id"]+":"+part
                        matches = allocations_by_id.get(identity, ())
                        if len(matches) != 1:
                            raise ValueError("native_PD_destination_owner_unavailable")
                        physical_owner = matches[0]["physical_owner"]
                        extents[(physical_owner, identity)].add(
                            ((physical_start+lo-logical_start)*row_bytes,
                             (physical_start+hi-logical_start)*row_bytes))
        phases, total_bytes, total_chunks = Counter(), 0, 0
        for (owner, _identity), ranges in extents.items():
            merged = []
            for lo, hi in sorted(ranges):
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append((lo, hi))
            for lo, hi in merged:
                count, amount = math.ceil((hi-lo)/chunk), min(chunk, hi-lo)
                total_bytes += hi-lo; total_chunks += count
                phases[(config["source_component"], "read", amount)] += count
                for component in executor.relay_components:
                    phases[(component, "write", amount)] += count
                    phases[(component, "read", amount)] += count
                phases[(component_by_owner[owner], "write", amount)] += count
        charge, rows = 0., []
        for (component, operation, amount), count in phases.items():
            memory = executor._configs.get(component)
            if memory is None:
                raise ValueError("native_PD_memory_configuration_unavailable")
            phase = sum(_phase(memory, amount, operation))
            charge += count*phase
            rows.append({"component_id": component, "operation": operation,
                "chunk_count_upper": count, "chunk_bytes_upper": amount,
                "initial_state_phase_per_chunk_ns": phase,
                "initial_state_phase_serial_upper_ns": count*phase})
        value = {"logical_bytes": total_bytes, "chunk_count": total_chunks,
            "initial_state_phase_serial_upper_ns": charge, "memory_phases": rows}
        by_shape[shape] = value
        return value

    try:
        observed = [geometry(by_id[identity]) for identity, _span in completed]
        if any(row["logical_bytes"] <= 0 or row["chunk_count"] <= 0 for row in observed):
            raise ValueError("native_PD_observed_geometry_empty")
        min_bytes, min_chunks = min(row["logical_bytes"] for row in observed), min(row["chunk_count"] for row in observed)
        span = max(value for _identity, value in completed)
        future = []
        for request in runtime.plan.requests:
            if request.request_id in runtime._external_initialized:
                continue
            row = geometry(request)
            scale = max(1., row["logical_bytes"]/min_bytes, row["chunk_count"]/min_chunks)
            future.append({"request_id": request.request_id, **row,
                "observed_bulk_shape_expansion": scale,
                "native_import_upper_ns": span*scale+row["initial_state_phase_serial_upper_ns"]})
    except (KeyError, TypeError, ValueError) as error:
        return {"qualified": False, "reason": str(error)}
    return {"qualified": True,
        "basis": "native_PD_inventory_exact_chunk_geometry_and_real_import_span_plus_source_relay_target_memory_phase",
        "conditional_bulk_assumption": "same_native_ordered_chunk_policy_bulk_service_is_nonnegative_byte_and_chunk_affine; initial_state_variation_explicitly_charged",
        "actual_import_spans_ns": [value for _identity, value in completed],
        "future_requests": future,
        "future_PD_serial_upper_ns": math.fsum(row["native_import_upper_ns"] for row in future),
        "physical_state_advanced": False}


def native_continuation_evidence(runtime, sample_records, options):
    """Produce a closed conditional B=1 continuation from native facts."""
    physical = runtime._execution_resource_kernel.physical_runtime
    if not _native_cores_supported(physical):
        return None
    summaries = [row.get("physical_observations", {}).get("native_fast_budget_observations")
                 for row in sample_records]
    if (not summaries or any(not isinstance(row, Mapping)
            or row.get("schema") != "native-window-ir-observations/v1"
            or row.get("unsupported") or row.get("stateful_gpu_cache") for row in summaries)):
        return None
    scenario, plan = runtime.plan.scenario, runtime.plan
    view = scenario.model._execution_view
    shape_equal = len({(row.prompt_tokens, row.output_tokens) for row in plan.requests}) == 1
    washout = None
    if plan.scheduler.max_num_seqs > 1 and not shape_equal:
        from .native_window_lifecycle import native_admission_washout
        washout = native_admission_washout(runtime)
        # These aged seeds change batch occupancy, not total sequence
        # storage capacity. The multislot evaluator must still bound every
        # unobserved shape and its TTFT contribution separately.
        shape_equal = (washout["qualified"]
            and len({row.prompt_tokens+row.output_tokens for row in plan.requests}) == 1)
    moe_layers = [instance.layer for instance in view.layer_instances if instance.layer.is_moe]
    supported_moe = all(layer.metadata.get("moe_routing", {}).get("load_model") == "deterministic_balanced_histogram"
                        for layer in moe_layers)
    if (any(instance.layer.is_linear_attention for instance in view.layer_instances) or not supported_moe
            or plan.scenario.placement.parallel.pp_degree != 1
            or plan.scenario.placement.parallel.ep_degree != 1
            or not shape_equal):
        return None
    decode = [(sample, summary) for sample, summary in zip(sample_records, summaries)
              if sample["kind"] == "decode"]
    query = [(sample, summary) for sample, summary in zip(sample_records, summaries)
             if sample["kind"] == "prefill"]
    if not decode or not query or len({repr(row["operator_signature"]) for _, row in decode}) != 1:
        return None
    if moe_layers:
        signatures = {repr(row["native_balanced_expert_signature"]) for _, row in decode}
        if len(signatures) != 1 or not decode[-1][1]["native_balanced_expert_signature"]:
            return None
    capacity = _allocation_evidence(runtime, summaries)
    phase = max(row["ordered_resource_phase_per_cohort_ns"] for _, row in decode)
    query_phase = max(row["ordered_resource_phase_per_cohort_ns"] for _, row in query)
    pd = _native_pd_evidence(runtime)
    if not pd["qualified"]:
        return None
    pd_upper = max((row["native_import_upper_ns"] for row in pd["future_requests"]), default=0.)
    outstanding = [row["request_id"] for row in pd["future_requests"]]
    timeline = runtime._execution_resource_kernel.physical_runtime.timeline
    carry = max(0., max(timeline.ready_ns.values(), default=runtime.now)-runtime.now)
    cleanup = max(row["output_commit_cleanup_upper_ns"] for row in summaries)
    actual_lag = max((sample["end_ns"]-commit["timestamp_ns"]
        for sample in sample_records for commit in sample.get("visible_commits", ())), default=0.)
    lag = actual_lag+cleanup
    guards = {name: {"qualified": True, "lower_guard_ns": 0., "upper_guard_ns": 0.,
        "basis": "charged_to_each_forecast_event_not_subtracted_globally"} for name in (
        "pd_service_and_interference", "cache_and_page_phase", "refresh_and_row_state",
        "allocation_reuse", "cutoff_carry_in", "prefill_query")}
    # Already-imported requests do not pay for their KV twice. Remaining PD
    # can contend with any measured cohort, so retain its whole serial span
    # once for the complete future window/request uncertainty, not once at
    # each query and again as Decode interference.
    guards["pd_service_and_interference"].update(
        upper_guard_ns=pd["future_PD_serial_upper_ns"],
        basis="each_not_yet_imported_authored_request_native_span_plus_initial_state_phase_once")
    return {"schema": "native-window-budget-evidence/v1", "capacity": capacity,
        "steady_contract": {"qualified": True,
            "basis": "native_fixed_sample_shape_task_topology_with_nonnegative_layer_context_feature_envelope",
            "research_assumptions": [
                "fixed native Task topology; constant work plus nonnegative context-linear work",
                "ordered contiguous IO has at most one residual initial row/page/cache phase per authored access",
                "future same-geometry PD import span is at most observed native maximum plus recorded initial-state phase",
                "holdout disagreement and drift expand the interval; they do not prove unseen detailed error"],
            "future_core_execution_claimed": False,
            "future_fresh_request_labels_may_permute": True,
            "fresh_shape_and_arrival_equal": True,
            "multislot_shape_timing_must_be_checked_by_evaluator": plan.scheduler.max_num_seqs > 1,
            "native_seed_washout": washout,
            "native_balanced_expert_signature": decode[-1][1]["native_balanced_expert_signature"],
            "moe_assignment_proof": "native_balanced_histogram_divmod_B_times_topk_by_num_experts_EP1" if moe_layers else "not_moe",
            "context_features": {"full_attention": any(not instance.layer.attention_chunk_size for instance in view.layer_instances),
                "chunk_sizes": sorted({instance.layer.attention_chunk_size for instance in view.layer_instances if instance.layer.attention_chunk_size}),
                "basis": "native_LayerSpec_attention_block_start_query_position_modulo_plus_current_query"}},
        "future_guards": guards,
        "event_guards": {"decode": {"lower_ns": phase, "upper_ns": phase},
            "prefill": {"lower_ns": query_phase, "upper_ns": query_phase},
            "cutoff_carry_in_upper_ns": carry},
        "phase_budget": {"decode_initial_state_phase_ns": phase,
            "query_initial_state_phase_ns": query_phase,
            "future_pd_per_request_upper_ns": pd_upper,
            "future_pd_unimported_request_ids": outstanding,
            "future_pd_total_serial_upper_ns": pd["future_PD_serial_upper_ns"],
            "native_PD_geometry_and_phase_budget": pd,
            "actual_import_spans_ns": pd["actual_import_spans_ns"], "cutoff_carry_in_ns": carry,
            "owner_accesses": decode[-1][1]["owner_accesses"],
            "refresh_bound_basis": decode[-1][1]["refresh_bound_basis"],
            "nominal_refresh_already_in_samples_not_subtracted": True},
        "constant_path_floor_ns": min(row["fixed_path_floor_ns"] for _, row in decode),
        "constant_path_floor_proof": "all_paths_constant_floor",
        "batch_invariant_weight_read_floor_ns": (0. if moe_layers else
            min(row["batch_invariant_weight_read_floor_ns"] for _, row in decode)),
        "batch_invariant_weight_floor_applicable": not moe_layers,
        "batch_invariant_path_floor_ns": (0. if moe_layers else
            min(row["batch_invariant_weight_read_floor_ns"] for _, row in decode)),
        "batch_invariant_path_floor_proof": "native_preloaded_generation_zero_tensor_read_bytes_over_directional_bandwidth_only" if not moe_layers else "unavailable_for_unsampled_MoE_active_expert_shapes",
        "visible_commit_lag_ns": {"qualified": True, "lower_ns": 0., "upper_ns": lag,
            "basis": "actual_visible_commit_lag_plus_native_output_commit_serial_cleanup_upper",
            "observed_max_lag_ns": actual_lag, "native_cleanup_serial_upper_ns": cleanup},
        "generic_nand_gc": {"applicable": False, "basis": "generic_nand_has_no_ftl_or_implicit_gc"},
        "physical_state_advanced": False}


# Original execution hooks, not flags accepted from candidate metadata. These
# identities are checked again when building every independent source summary.
from .serving import _OnlineRuntime as _NATIVE_SOURCE_RUNTIME
from .planner import TopologyAwareBatchCostProvider as _NATIVE_SOURCE_LOWERER
from .event_kernel import UnifiedEventKernel as _NATIVE_SOURCE_KERNEL
from .data_motion import PhysicalRuntimeContext as _NATIVE_SOURCE_PHYSICAL


def _native_source_methods(cls):
    return tuple((name, getattr(getattr(cls, name), "__func__", getattr(cls, name)))
                 for name, value in vars(cls).items()
                 if callable(value) or isinstance(value, (staticmethod, classmethod)))


_NATIVE_SOURCE_RUNTIME_METHODS = _native_source_methods(_NATIVE_SOURCE_RUNTIME)
_NATIVE_SOURCE_LOWERER_METHODS = _native_source_methods(_NATIVE_SOURCE_LOWERER)
_NATIVE_SOURCE_KERNEL_METHODS = _native_source_methods(_NATIVE_SOURCE_KERNEL)
_NATIVE_SOURCE_PHYSICAL_METHODS = _native_source_methods(_NATIVE_SOURCE_PHYSICAL)


__all__ = ["summarize_native_stages", "native_continuation_evidence", "native_context_features"]
