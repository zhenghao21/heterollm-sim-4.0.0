"""Conditional service bounds for native contiguous HBM4 requests.

This invokes the existing native command machine on a private cold controller.
It is a budget construction, not a physical serving sample or a detailed
reference. No source controller, allocator, event or visible Token is changed.
The REF-on reachable-source bound instead serializes the existing physical
command chains under an explicit source-state contract. The finite clock
horizon is a required final-window guard, never an assumption silently
supplied by either helper.
"""
from __future__ import annotations

from dataclasses import fields
import math

from ._hbm_control import row_lookahead_depth
from .memory_types import AccessRequest, DramConfig


_CACHE = {}


def _key(config):
    # All executable timing/mapping quantities are normalized scalar fields.
    # Resource-prefix metadata affects names only; the caller separately
    # proves that native channels and their CA/DQ ports do not alias.
    return tuple(getattr(config, field.name) for field in fields(config)
                 if field.name != "metadata")


def _prepare(config):
    from .dram_core import DramCore
    from ._compiled_hbm import _pack_hbm_numeric
    from ._hbm_numeric import _new_channel_scratch

    shadow = DramCore(config, capture_details=False)
    packed = _pack_hbm_numeric(shadow, AccessRequest("native-budget-cold", "read", 0, 32))
    if packed is None:
        return None
    # A public normalized HBM4 configuration must retain standard REF
    # timings. This explicitly counterfactual budget call disables only
    # the REF branch of a private packed instruction vector, never a
    # physical configuration. Every other native timing and tRASmax stays.
    packed.g = packed.g.copy()
    packed.g[8] = 0
    return packed, _new_channel_scratch(packed.g, packed.arrays)


def _cold_duration(prepared, byte_count, operation):
    from ._hbm_numeric import _run_hbm_channel_range

    packed, scratch = prepared
    arrays = tuple(array.copy() for array in packed.arrays)
    # Full global bank-row periods visit every homogeneous independent
    # physical channel identically. Running one real channel is exact for
    # their maximum; it does not divide an array latency by channel count.
    result = _run_hbm_channel_range(0, byte_count//packed.g[9], 0., operation == "read",
        packed.g, packed.t, *arrays, scratch, 0, 1)
    return float(result[0])


def _clock_limit(config, prepared, operation, *, extra_timing=()):
    from ._hbm_numeric import _column_tick_profile

    packed = prepared[0]
    profile = _column_tick_profile(packed.g, packed.t, operation == "read")
    if not profile[0]:
        return None
    limit = float(profile[1])*config.command_clock_ns
    quantum = config.command_clock_ns/2
    # Outside the column recurrence, a new row has one analog tRCD, ACT
    # spacing or recovery before the next CA-grid alignment. Exact-grid
    # delays use the existing integer clock_after rule. An analog delay must
    # remain strictly inside the same half-CK bin throughout the declared
    # absolute horizon; 64 ULPs covers its bounded add/subtract/compare chain.
    for value in (config.activate_to_read_ns, config.activate_to_write_ns,
                  config.precharge_to_activate_ns, config.minimum_active_ns,
                  config.activate_cycle_ns, config.read_to_precharge_ns,
                  config.write_to_precharge_ns, config.precharge_to_precharge_ns,
                  config.activate_same_group_ns, config.activate_different_group_ns,
                  config.four_activate_window_ns, *extra_timing):
        ticks = round(value/quantum)
        if value == ticks*quantum:
            continue
        lower = math.floor(value/quantum)*quantum
        margin = min(value-lower, lower+quantum-value)
        if margin <= 0:
            return None
        limit = min(limit, margin/(64*2.**-52)-abs(value))
    return limit if limit > 0 else None


def native_hbm_no_refresh_upper(config, byte_count, operation, *, independent_channels=False):
    """Return a sufficient intrinsic service upper, conditional on a clock domain.

    For the supported homogeneous FCFS mapping, pad an arbitrary contiguous
    interval to complete global bank-row cycles. A padded cold execution
    contains its burst sequence in the native channel order. Row lookahead
    keeps that containment: one-burst bank interleaving gives every physical
    bank one burst per two channel bank sweeps, longer than the lookahead
    window, so no early PRE is ever deferred, and each burst's PRE is placed
    a fixed number of channel bursts before its ACT. The retained bursts'
    commands therefore keep their standalone row-bus order once the
    prefix/suffix commands are removed. Removing those removes resource
    constraints; its possible first-row hit advantage over another initial
    row is covered by clearing every bank. With REF and new tRASmax expiry
    absent, all remaining CA/DQ dependencies, including tRRD/tFAW, are FIFO
    max-plus constraints. A different initial row only adds or removes a
    first-visit PRE/ACT at its fixed position, so clearing the initial
    row/history state in serial gives the additive initial-state upper below.

    New rows cannot introduce tRASmax branches if every row's residence is
    bounded by two complete cold cycles plus the initial-state allowance.
    This is checked before using the bound, and applies inductively at the
    first possible expiry. The original positive tRASmax remains enabled in
    every cold native invocation. Outstanding earlier work is a separate
    paid boundary carry; it must not be hidden inside intrinsic IO service.
    """
    rejected = {"qualified": False, "physical_state_advanced": False}
    if (type(config) is not DramConfig or type(byte_count) is not int or byte_count <= 0
            or operation not in {"read", "write"} or not independent_channels
            or config.control_mode != "jedec_hbm" or config.generation != "HBM4"
            or config.scheduler != "fcfs" or config.address_mapping != "bank_interleaved"
            or (config.interleave_bytes or config.burst_bytes) != config.burst_bytes
            or config.bank_count != 64 or config.row_bytes != 1024 or config.banks_per_group != 8
            or config.activate_command_cycles != 1.5):
        return {**rejected, "reason": "native_cold_geometry_or_private_channel_contract_unsupported"}
    key = (_key(config), byte_count, operation)
    if key in _CACHE:
        return dict(_CACHE[key])
    prepared = _prepare(config)
    if prepared is None:
        return {**rejected, "reason": "native_cold_original_command_machine_unavailable"}
    limit = _clock_limit(config, prepared, operation)
    if limit is None:
        return {**rejected, "reason": "native_cold_clock_translation_profile_unavailable"}
    period = config.lane_count*config.bank_count*config.row_bytes
    cycles = (byte_count+2*period-2)//period
    padded = cycles*period
    # Row identities are used only for equality by the native machine.
    # An activated-row mask may identify rows separated by 2**msb, so a
    # shorter consecutive interval cannot contain a new alias/reuse edge.
    if (padded > config.capacity_bytes
            or (config.activated_row_msb is not None and cycles >= 2**config.activated_row_msb)):
        return {**rejected, "reason": "native_cold_padding_capacity_or_row_identity_bound_unavailable"}
    clock = config.command_clock_ns
    history = max(config.minimum_active_ns, config.activate_cycle_ns,
        config.read_to_precharge_ns,
        config.write_latency_ns+config.burst_bytes/config.effective_lane_bandwidth_gb_s
            +config.write_to_precharge_ns,
        config.read_to_write_ns, config.write_to_read_ns, config.read_to_write_command_ns,
        config.write_to_read_same_group_ns, config.write_to_read_different_group_ns,
        config.column_same_group_ns, config.column_different_group_ns,
        config.column_different_sid_ns, config.four_activate_window_ns,
        config.activate_same_group_ns, config.activate_different_group_ns, 2*clock)
    # A padded prefix may already have opened the first row needed by the
    # retained interval. Charge the complete per-bank startup chain, not
    # just PRE issue occupancy: a wrong initial row pays PRE/tRP/ACT/tRCD.
    # If the first retained row is only a short tail, the inserted ACT can
    # additionally constrain its following PRE/ACT through tRAS/tRC and the
    # following ACTs through tRRD/tFAW. Those successors share the same ACT
    # anchor, so their maximum is sufficient after the initial PRE+tRP, and
    # all banks are serialized conservatively.
    pre = max(config.precharge_command_cycles*clock, config.precharge_to_precharge_ns)
    after_act = max(config.activate_to_read_ns, config.activate_to_write_ns,
        config.minimum_active_ns+config.precharge_command_cycles*clock
            +config.precharge_to_activate_ns,
        config.activate_cycle_ns, config.four_activate_window_ns,
        config.activate_same_group_ns, config.activate_different_group_ns)
    per_bank = (pre+config.precharge_to_activate_ns
        +config.activate_command_cycles*clock+after_act+2*clock)
    initial = history+config.pseudo_channels_per_channel*config.bank_count*per_bank+2*clock
    try:
        two_cycle = _cold_duration(prepared, 2*period, operation)
    except (ValueError, OverflowError):
        return {**rejected, "reason": "native_cold_new_row_tRASmax_residence_not_proven"}
    residence = two_cycle+2*initial+2*clock
    if residence >= config.maximum_active_ns:
        result = {**rejected, "reason": "native_cold_new_row_tRASmax_residence_not_proven",
            "new_row_residence_upper_ns": residence,
            "maximum_active_ns": config.maximum_active_ns}
    else:
        try:
            cold = _cold_duration(prepared, padded, operation)
        except (ValueError, OverflowError):
            return {**rejected, "reason": "native_cold_complete_padding_upper_unavailable"}
        result = {"qualified": True, "physical_state_advanced": False,
            "conditional_only": True,
            "basis": "original_native_REF_disabled_cold_full_row_cycles_plus_serial_initial_PRE_history; no_new_tRASmax_by_two_cycle_residence",
            "conditional_no_refresh_serial_upper_ns": cold+initial+clock,
            "required_native_hbm_clock_limit_ns": limit,
            "hbm_clock_domain_limit_ns": math.nextafter(limit-(cold+initial+clock), -math.inf),
            "hbm_clock_domain_comparison_reserve_ns": cold+initial+clock,
            "padded_byte_count": padded, "global_bank_row_period_bytes": period,
            "native_cold_duration_ns": cold, "initial_state_upper_ns": initial,
            "new_row_residence_upper_ns": residence,
            "maximum_active_ns": config.maximum_active_ns,
            "initial_outstanding_work_scope": "paid_separate_boundary_carry; private_serial_intrinsic_service",
            "counterfactual_is_physical_sample": False}
    if len(_CACHE) >= 128:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return dict(result)


def _native_refresh_history_upper(config):
    """Latent native history after its authored source carry is paid."""
    dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
    pacing = max(config.burst_bytes/config.directional_lane_bandwidth_gb_s(which)
                 for which in ("read", "write"))
    return max(config.activate_to_read_ns, config.activate_to_write_ns,
        config.precharge_to_activate_ns, config.minimum_active_ns, config.activate_cycle_ns,
        config.read_to_precharge_ns, config.write_to_precharge_ns,
        config.column_same_group_ns, config.column_different_group_ns,
        config.column_different_sid_ns, config.write_to_read_same_group_ns,
        config.write_to_read_different_group_ns, config.read_to_write_command_ns,
        config.precharge_to_precharge_ns, config.read_to_write_ns,
        config.write_to_read_ns, config.four_activate_window_ns,
        config.activate_same_group_ns, config.activate_different_group_ns,
        2*config.command_clock_ns, max(0., pacing-dq))


def _native_refresh_source_limits(config):
    """Pure-configuration limits for the separately checked live source."""
    clock = config.command_clock_ns
    dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
    pacing = max(config.burst_bytes/config.directional_lane_bandwidth_gb_s(which)
                 for which in ("read", "write"))
    return {"initial_history_upper_ns": _native_refresh_history_upper(config),
        "source_final_tail_upper_ns": max(config.read_latency_ns,
            config.write_latency_ns)+max(dq, pacing),
        "source_precharge_tail_upper_ns": max(config.minimum_active_ns,
            config.read_to_precharge_ns, config.write_latency_ns+dq
                +config.write_to_precharge_ns, config.precharge_to_precharge_ns, 2*clock),
        "source_CA_tail_upper_ns": 2*clock,
        "source_ACT_due_rounding_margin_ns": 2*clock}


def native_hbm_initial_row_upper(config, byte_count, operation, *,
                                 independent_channels=False,
                                 native_refresh_reachable=False):
    """Pay complete initial bank startup, not only CA issue occupancy.

    A native cold row may already be open in another source image. Changing
    that initial row pays the whole PRE/recovery/tRP/ACT/tRCD chain; a short
    retained first-row tail also pays its tRAS/tRC successors. The banks'
    arrays cannot be called parallel here: the native burst traversal
    advances the shared CA calendar through each complete row transition.

    Proven REF-on reachable state excludes untouched old-row expiry, so
    only the actual maximum initial banks touched by the request need the
    complete chain. Otherwise every initial bank is charged, together with
    its possible old tRASmax deadline horizon. Unknown shared channels do
    not divide either count by a nominal lane count. All old ACT anchors
    must be no later than the separately paid native source frontier. The
    reachable-source reduction also requires the caller to close the
    original REF arithmetic clock domain over the whole forecast window.
    """
    rejected = {"qualified": False, "physical_state_advanced": False}
    if (type(config) is not DramConfig or config.control_mode != "jedec_hbm"
            or type(byte_count) is not int or byte_count < 0
            or operation not in {"read", "write"}):
        return {**rejected, "reason": "native_initial_row_contract_unsupported"}
    if byte_count == 0:
        return {"qualified": True, "physical_state_advanced": False,
            "initial_state_upper_ns": 0., "initial_bank_upper": 0,
            "old_row_expiry_horizon_upper_ns": 0.}
    clock = config.command_clock_ns
    history = max(_native_refresh_history_upper(config),
        config.write_latency_ns+config.burst_bytes/config.effective_lane_bandwidth_gb_s
            +config.write_to_precharge_ns)
    pre = max(config.precharge_command_cycles*clock, config.precharge_to_precharge_ns)
    after_act = max(config.activate_to_read_ns, config.activate_to_write_ns,
        config.minimum_active_ns+config.precharge_command_cycles*clock
            +config.precharge_to_activate_ns,
        config.activate_cycle_ns, config.four_activate_window_ns,
        config.activate_same_group_ns, config.activate_different_group_ns)
    per_bank = (pre+config.precharge_to_activate_ns
        +config.activate_command_cycles*clock+after_act+2*clock)
    refresh_proof = (native_hbm_refresh_service_upper(config, byte_count, operation,
        independent_channels=independent_channels, native_refresh_reachable=True)
        if native_refresh_reachable else None)
    source_proven = refresh_proof is not None and refresh_proof["qualified"]
    bursts = (byte_count+2*config.burst_bytes-2)//config.burst_bytes
    if independent_channels:
        initial_banks = config.pseudo_channels_per_channel*config.bank_count
        if source_proven:
            block = (config.bank_count*(config.row_bytes//config.burst_bytes)
                if config.address_mapping == "row_local" else
                (config.interleave_bytes or config.burst_bytes)//config.burst_bytes)
            whole, residual = divmod(bursts, config.lane_count*block)
            channel_bursts = whole*2*block+min(residual, 2*block)
            initial_banks = min(initial_banks, channel_bursts)
    else:
        initial_banks = config.lane_count*config.bank_count
    expiry = 0. if source_proven else max(0., config.maximum_active_ns)
    upper = math.nextafter(history+initial_banks*per_bank+2*clock+expiry, math.inf)
    unique = {"qualified": False,
        "reason": "single_contiguous_unique_bank_REF_source_contract_unavailable"}
    if source_proven and bursts <= config.lane_count*config.bank_count:
        # The qualified native REF helper already proves one-burst
        # bank-interleaving. A consecutive sequence no longer than one
        # global bank sweep touches each physical bank at most once, even
        # across its address-period boundary. Its new ACT therefore has no
        # subsequent same-bank PRE/tRAS/tRC successor.
        #
        # Between REF epochs the original row-CA chain contains only old
        # history plus PRE/tRP/ACT and ACT spacing. No new column or DQ
        # event feeds back into it when banks are unique. Bank j's ACT ends
        # within one row_chain of bank j-1's: row lookahead issued its own
        # PRE no later than an earlier ACT, at most one later early PRE
        # precedes it on the row bus, and tRRD follows the previous ACT.
        # tFAW follows the ACT four banks earlier, so a quarter of it per
        # bank suffices; older ACTs are settled by the history bridge. The
        # request's first burst precharges its whole lookahead window ahead
        # of its own ACT, one PRE slot each. Its original column/DQ max-plus
        # graph receives at most one row-to-column bridge per REF-free
        # segment, rather than one full tRCD/latency per bank. A REF can
        # feed a previous column's recovery back into row CA; charge a
        # fresh complete history/RCD/latency/DQ bridge on *every* segment.
        # The epoch count uses the independent unchanged full active-work
        # upper above, so this reduction cannot justify its own REF count.
        blocked = refresh_proof["refresh_epoch_upper_ns"]
        epochs = 1+math.floor(refresh_proof["active_service_upper_ns"]
            /(config.refresh_interval_ns-blocked))
        act = config.activate_command_cycles*clock+3*clock
        row_chain = max(pre+max(config.precharge_to_activate_ns,
                                config.precharge_command_cycles*clock)+act,
            max(config.activate_same_group_ns, config.activate_different_group_ns)+act,
            (config.four_activate_window_ns+act)/4)
        first_step = row_lookahead_depth(config)*(pre+clock)
        latency = config.read_latency_ns if operation == "read" else config.write_latency_ns
        rcd = config.activate_to_read_ns if operation == "read" else config.activate_to_write_ns
        dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
        bridge = history+rcd+latency+dq+2*clock
        unique_upper = math.nextafter(first_step+initial_banks*row_chain
                                      +(epochs+1)*bridge+2*clock, math.inf)
        applied = unique_upper < upper
        unique = {"qualified": True, "applied": applied,
            "basis": "one_contiguous_native_bank_sweep; independent_row_CA_column_CA_DQ; no_new_same_bank_successor; lookahead_PRE_and_ACT_spacing_in_row_chain; full_history_RCD_latency_DQ_bridge_per_REF_free_segment",
            "scope": "initial_state_perturbation_of_common_native_column_DQ_graph; not_a_total_service_upper",
            "single_request_operation": operation,
            "maximum_global_burst_count": bursts,
            "maximum_unique_physical_bank_count": config.lane_count*config.bank_count,
            "serialized_initial_row_CA_upper_per_bank_ns": row_chain,
            "lookahead_first_burst_PRE_upper_ns": first_step,
            "row_CA_clock_alignment_and_anchor_margin_ns": 3*clock,
            "independently_proven_maximum_REF_epochs": epochs,
            "REF_free_segment_upper": epochs+1,
            "complete_history_and_column_bridge_per_segment_ns": bridge,
            "initial_state_upper_ns": unique_upper,
            "original_complete_serial_initial_state_upper_ns": upper,
            "REF_epoch_blocking_charged_separately": True,
            "physical_state_advanced": False}
        if applied:
            upper = unique_upper
    elif source_proven:
        unique["reason"] = "request_may_revisit_a_physical_bank"
    clock_domain = ({name: refresh_proof[name] for name in (
        "required_native_hbm_clock_limit_ns", "hbm_clock_domain_limit_ns",
        "hbm_clock_domain_comparison_reserve_ns")} if source_proven else {})
    return {"qualified": True, "physical_state_advanced": False,
        "conditional_only": True, "initial_state_upper_ns": upper,
        **clock_domain,
        "initial_bank_upper": initial_banks, "initial_history_upper_ns": history,
        "per_initial_bank_startup_upper_ns": per_bank,
        "old_row_expiry_horizon_upper_ns": expiry,
        "untouched_old_row_expiry_excluded": bool(source_proven),
        "unique_bank_initial_row_proof": unique,
        "initial_outstanding_work_scope": "paid_native_source_frontier; all_old_ACT_anchors_no_later_than_frontier",
        "basis": ("native_unique_bank_row_CA_chain_with_complete_bridge_per_REF_free_segment"
            if unique.get("applied") else
            "complete_serial_initial_bank_PRE_history_tRP_ACT_tRCD_tRAS_tRC; unknown_source_all_banks_and_old_expiry_horizon")}


def native_hbm_refresh_service_upper(config, byte_count, operation, *,
                                     independent_channels=False,
                                     native_refresh_reachable=False):
    """Bound REF-on active work for a qualified native reachable controller.

    This is not a no-REF execution from an arbitrary initial bank image.
    The caller must prove native cold ancestry, unchanged all-bank REF,
    independent CA/DQ buses and the source limits returned here. In
    particular, before next due ``d``, every open ACT is at least ``d-I``;
    all preceding column/ACT/PRE stamps are no later than ``d``. The last
    DQ/recovery tails, unlike arbitrary future reservations, are bounded.

    Each completed REF closes both PCs before incrementing due by I. An
    ACT after that REF is no earlier than its due. Hence its expiry is
    later than the next due by M-I (less an explicit clock margin). The
    native expiry branch therefore first performs REF rather than the
    arbitrary old-bank expiry cascade. This invariant is preserved by
    every following native ACT/REF; unknown restored state does not get it.

    Once source carry is paid, H settles old command histories. Serialize
    every burst of the busiest actual channel, including the whole
    PRE/recovery/tRP/ACT/tRC/tRRD/tFAW/tRCD/COL/latency/DQ path. Row
    lookahead only moves a burst's PRE earlier in the fixed row-bus order;
    the extra PREs of a request's first burst are charged once. Early PREs
    pass the same maintenance check as burst commands and open no row, so
    the invariant below is unchanged. Alternative gates take a maximum,
    not their sum. All nominal REF stays charged separately
    by the caller's residual-plus-feedback envelope. Stale epochs complete
    before the origin except at most its final residual: source stamps
    and tails limit the first stale epoch to R<I, and subsequent closed
    epochs have the same bound. Ordinary carry alone is not that proof.
    """
    rejected = {"qualified": False, "physical_state_advanced": False}
    if (type(config) is not DramConfig or type(byte_count) is not int or byte_count <= 0
            or byte_count > config.effective_capacity_bytes
            or operation not in {"read", "write"} or not independent_channels
            or not native_refresh_reachable or config.control_mode != "jedec_hbm"
            or config.generation != "HBM4" or config.refresh_policy != "all_bank"
            or config.scheduler != "fcfs" or config.address_mapping != "bank_interleaved"
            or (config.interleave_bytes or config.burst_bytes) != config.burst_bytes
            or config.bank_count != 64 or config.row_bytes != 1024
            or config.banks_per_group != 8 or config.pseudo_channels_per_channel != 2
            or config.activate_command_cycles != 1.5):
        return {**rejected, "reason": "native_REF_reachable_source_or_geometry_not_proven"}
    key = ("reachable_REF", _key(config), byte_count, operation)
    if key in _CACHE:
        return dict(_CACHE[key])
    clock, interval = config.command_clock_ns, config.refresh_interval_ns
    dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
    pacing = max(config.burst_bytes/config.directional_lane_bandwidth_gb_s(which)
                 for which in ("read", "write"))
    source_limits = _native_refresh_source_limits(config)
    initial = source_limits["initial_history_upper_ns"]
    recovery = source_limits["source_precharge_tail_upper_ns"]
    # The native routine completely reserves PC0's PRE/tRP/REF before
    # visiting PC1. PC1's PRE cannot be inserted into PC0's earlier tRP
    # gap on the already advanced shared CA calendar, so both tRP waits
    # must be paid even though the final tRFC array periods overlap.
    blocked = (config.refresh_cycle_ns+recovery
        +config.pseudo_channels_per_channel*config.precharge_to_activate_ns
        +(2*config.pseudo_channels_per_channel*config.precharge_command_cycles
          +config.activate_command_cycles+config.pseudo_channels_per_channel+2)*clock)
    if not 0 < blocked < interval:
        return {**rejected, "reason": "native_REF_reachable_positive_service_interval_unavailable"}
    prepared = _prepare(config)
    if prepared is None:
        return {**rejected, "reason": "native_REF_reachable_clock_profile_unavailable"}
    limit = _clock_limit(config, prepared, operation, extra_timing=(interval,
        config.refresh_cycle_ns, config.maximum_active_ns, config.read_latency_ns,
        config.write_latency_ns, dq, pacing))
    if limit is None:
        return {**rejected, "reason": "native_REF_reachable_clock_profile_unavailable"}
    # One arbitrary unaligned interval adds at most one partially touched
    # burst. Two adjacent PC blocks belong to one physical channel.
    bursts = (byte_count+2*config.burst_bytes-2)//config.burst_bytes
    whole, residual = divmod(bursts, config.lane_count)
    channel_bursts = whole*2+min(residual, 2)
    pre_align = clock if config.generation == "HBM4" else clock/2
    pre_history = max(config.minimum_active_ns, config.read_to_precharge_ns,
        config.write_to_precharge_ns, config.precharge_to_precharge_ns, 2*clock)
    # tRC, tRRD and tFAW count from earlier ACTs, all inside earlier
    # serialized bursts. The one later early PRE that row lookahead may
    # place before this ACT is gated only by earlier bursts' stamps, so it
    # ends inside the same PRE/tRP slot.
    pre_to_act = max(pre_history+pre_align
        +max(config.precharge_to_activate_ns, config.precharge_command_cycles*clock),
        config.activate_cycle_ns, config.activate_same_group_ns,
        config.activate_different_group_ns, config.four_activate_window_ns)
    latency = config.read_latency_ns if operation == "read" else config.write_latency_ns
    rcd = config.activate_to_read_ns if operation == "read" else config.activate_to_write_ns
    row_to_column = pre_to_act+clock+max(config.activate_command_cycles*clock, clock+rcd)
    column_history = max(config.column_same_group_ns, config.column_different_group_ns,
        config.column_different_sid_ns, config.write_to_read_same_group_ns,
        config.write_to_read_different_group_ns, config.read_to_write_command_ns,
        config.read_to_write_ns, config.write_to_read_ns, max(0., pacing-dq))
    # Two CA edges cover alignment plus the native data-ready subtraction
    # check. A full clock also covers the specified command occupancy.
    per_burst = max(row_to_column, column_history)+2*clock+max(
        config.column_command_cycles*clock, latency+dq)
    # The first burst precharges its whole lookahead window before its ACT.
    first_step = row_lookahead_depth(config)*(max(config.precharge_command_cycles*clock,
        config.precharge_to_precharge_ns)+pre_align)
    active = math.nextafter(initial+first_step+channel_bursts*per_burst, math.inf)
    # The eight-clock gap is much larger than the bounded add/compare ULP
    # chain. It is kept explicit instead of treating real-number equality
    # ACT >= next_due-I as exact for an analog tREFI.
    expiry_margin = config.maximum_active_ns-interval-8*clock
    if not active+blocked < expiry_margin:
        return {**rejected, "reason": "native_REF_reachable_row_lifetime_margin_unavailable",
            "active_service_upper_ns": active, "refresh_epoch_upper_ns": blocked,
            "row_lifetime_margin_ns": expiry_margin}
    reserve = math.nextafter(active+blocked, math.inf)
    result = {"qualified": True, "physical_state_advanced": False,
        "conditional_only": True, "counterfactual_is_physical_sample": False,
        "qualified_source_required": True,
        "basis": "native_REF_on_reachable_source; original_channel_burst_mapping; complete_serial_PRE_ACT_COL_DQ_chain; residual_epoch_and_feedback_retained",
        "active_service_upper_ns": active,
        # The common envelope consumes this name; it denotes active work,
        # not a physical no-REF counterfactual from arbitrary bank state.
        "conditional_no_refresh_serial_upper_ns": active,
        **source_limits, "per_burst_serial_upper_ns": per_burst,
        "lookahead_first_burst_PRE_upper_ns": first_step,
        "channel_burst_upper": channel_bursts, "refresh_epoch_upper_ns": blocked,
        "row_lifetime_margin_ns": expiry_margin,
        "required_native_hbm_clock_limit_ns": limit,
        "hbm_clock_domain_limit_ns": math.nextafter(limit-reserve, -math.inf),
        "hbm_clock_domain_comparison_reserve_ns": reserve,
        "source_contract": "native_cold_REF_on_ancestry; pc_blocked_and_ACT_PRE_COL_stamps_le_due; bounded_DQ_and_PRE_tails; open_ACT_ge_due_minus_REFI; paid_boundary_carry"}
    if len(_CACHE) >= 128:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return dict(result)


__all__ = ["native_hbm_no_refresh_upper", "native_hbm_refresh_service_upper",
           "native_hbm_initial_row_upper"]
