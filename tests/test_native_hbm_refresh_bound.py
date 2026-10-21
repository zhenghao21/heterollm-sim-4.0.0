"""The REF envelope follows native address/channel and command rules."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from heterollm_sim.memory_mapping import map_dram_address
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.native_window_producer import (_channel_burst_upper,
    _hbm_channels_independent, _refresh_serial_bound)
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


@pytest.mark.parametrize("mapping,interleave", (("bank_interleaved", 32),
    ("bank_interleaved", 128), ("bank_interleaved", 4096), ("row_local", None)))
def test_periodic_channel_burst_upper_covers_actual_native_mapping(mapping, interleave):
    config = configuration(address_mapping=mapping, interleave_bytes=interleave,
                           channels=3)
    for first in (0, 1, 5, 31, 63, 129, 1023, 1024, 3071, 8191):
        for count in (1, 2, 5, 16, 63, 65, 127, 259, 1027):
            distribution = Counter()
            for burst in range(first, first+count):
                location = map_dram_address(config, burst*config.burst_bytes)
                distribution[location.stack, location.channel] += 1
            assert max(distribution.values()) <= _channel_burst_upper(config, count)


def test_channel_splitting_requires_actual_unaliased_native_buses():
    config = configuration()
    timeline = ResourceTimeline()
    runtime = NS(timeline=timeline)
    assert _hbm_channels_independent(config, runtime)
    timeline.resource_aliases["unrelated"] = "other"
    assert _hbm_channels_independent(config, runtime)
    timeline.resource_aliases["dram:row_command:0:1"] = "dram:row_command:0:0"
    assert not _hbm_channels_independent(config, runtime)
    assert not _hbm_channels_independent(config, None)
    same_ports = replace(config, metadata={"row_command_resource_prefix": "same",
                                         "column_command_resource_prefix": "same"})
    assert not _hbm_channels_independent(same_ports, NS(timeline=ResourceTimeline()))


@pytest.mark.parametrize("operation", ("read", "write"))
def test_bound_covers_native_active_refresh_at_different_grid_phases(operation):
    config = configuration(refresh_policy="all_bank", refresh_interval_ns=400.,
                           refresh_cycle_ns=40.)
    no_refresh = replace(config, refresh_policy="disabled",
                         refresh_interval_ns=0., refresh_cycle_ns=0.)
    observed_stalls = []
    for address, count in ((3, 1), (31, 65), (125, 8193), (32765, 32771)):
        guard = _refresh_serial_bound(config, count, operation, independent_channels=True)
        for arrival in (0., 1., 355., 399., 400., 439., 799.5, 1160.):
            results = []
            for setting in (config, no_refresh):
                core = DramCore(setting, capture_details=False)
                result = core.submit(AccessRequest("sample", operation, address, count, arrival))
                results.append(result.completion_ns-arrival)
            extra = results[0]-results[1]
            assert extra <= guard
            observed_stalls.append(extra)
    assert max(observed_stalls) > config.refresh_cycle_ns


def test_native_channel_proof_tightens_a_large_contiguous_request_without_dropping_feedback():
    config = configuration(channels=32, refresh_policy="all_bank",
                           refresh_interval_ns=400., refresh_cycle_ns=40.)
    full = _refresh_serial_bound(config, 1024**2, "read")
    channel = _refresh_serial_bound(config, 1024**2, "read", independent_channels=True)
    assert config.refresh_cycle_ns < channel < full/8
    assert _refresh_serial_bound(config, 0, "read", independent_channels=True) == 0.


def _native_segment_summary(*, foreign_wait=False, alias=False):
    from heterollm_sim.contracts import _PreparedExecutionStage
    from heterollm_sim.native_window_producer import summarize_native_stages
    from test_native_window_producer import task
    config = configuration(refresh_policy="all_bank", refresh_interval_ns=3900., refresh_cycle_ns=40.)
    rows = []
    if foreign_wait:
        rows.append(task("foreign", 1, resources=("gpu1",)))
    for index in range(10):
        dependencies = () if not index else (str(index-1),)
        if index == 5 and foreign_wait:
            dependencies += ("foreign",)
        rows.append(task(str(index), 0, dependencies, ("gpu0",), {
            "physical_owner": "memory", "physical_memory_config": config,
            "memory_accesses": ({"operation": "read", "byte_count": 32,
                "physical_owner": "memory", "resource_id": "memory"},)}))
    timeline = ResourceTimeline()
    if alias:
        timeline.resource_aliases["dram:row_command:0:1"] = "dram:row_command:0:0"
    runtime = NS(timeline=timeline, allocators={}, runtimes={})
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., tuple(rows))
    return summarize_native_stages((stage,), NS(), runtime)


def test_closed_native_chain_pays_one_segment_residual_and_retains_feedback():
    summary = _native_segment_summary()
    proof = summary["native_resource_order_proof"][0]
    segments = proof["closed_hbm_busy_segments"]["segments"]
    assert len(segments) == 1 and segments[0]["qualified"]
    assert segments[0]["task_count"] == 10
    assert segments[0]["segment_refresh_upper_ns"] < segments[0]["old_per_access_refresh_upper_ns"]
    assert proof["applied_phase_ns"] < proof["serial_access_phase_ns"]


def test_new_foreign_dependency_starts_a_new_native_busy_segment():
    summary = _native_segment_summary(foreign_wait=True)
    segments = summary["native_resource_order_proof"][0]["closed_hbm_busy_segments"]["segments"]
    assert [row["task_count"] for row in segments] == [5, 5]
    assert all(row["added_complete_group_barrier"] for row in segments)


def test_aliased_controller_cannot_use_segment_ref_tightening():
    summary = _native_segment_summary(alias=True)
    proof = summary["native_resource_order_proof"][0]
    segments = proof["closed_hbm_busy_segments"]["segments"]
    assert all(not row["qualified"] for row in segments)
    assert all(row["segment_refresh_upper_ns"] == row["old_per_access_refresh_upper_ns"] for row in segments)


def test_possible_outside_same_owner_IO_prevents_busy_segment_tightening():
    from heterollm_sim.native_window_producer import _busy_segment_phase_bound
    from test_native_window_producer import task
    rows, bills = [], {}
    for index in range(11):
        identity = "outside" if index == 5 else str(index)
        deps = () if index in (0, 5) else (str(index-1 if index != 6 else 4),)
        row = task(identity, 0, deps)
        rows.append((row, 100., {"@physical-owner:memory"}))
        bills[identity] = {"refresh_ns": 100., "terms": {"memory": [(3900., 100., 100.)]},
            "qualified": index != 5, "other_physical": index == 5, "nonphysical_upper_ns": 0.}
    bound, proof = _busy_segment_phase_bound(rows, bills, owner_ranks={"memory": {0}})
    assert bound == 1100.
    assert all(not row["qualified"] for row in proof["segments"])
    assert any(not row["original_same_owner_IO_isolated_by_native_DAG"] for row in proof["segments"])


def _formal_staging_hbm4(**changes):
    """The real paper HBF staging HBM organization, not a toy REF machine."""
    from exploring_hbf_paper_config import build_paper_hbm_config
    return replace(build_paper_hbm_config("HBM4"), stacks=1,
        capacity_bytes=36_000_000_000, interface_bandwidth_gb_s=1600.,
        read_bandwidth_gb_s=1600., write_bandwidth_gb_s=1600., **changes)


@pytest.mark.parametrize("operation", ("read", "write"))
def test_reachable_REF_bound_retains_feedback_and_does_not_require_a_fake_no_REF_sample(operation):
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    config = _formal_staging_hbm4()
    proof = native_hbm_refresh_service_upper(config, 32768, operation,
        independent_channels=True, native_refresh_reachable=True)
    assert proof["qualified"] and proof["qualified_source_required"]
    assert proof["channel_burst_upper"] == 33
    assert not proof["physical_state_advanced"]
    assert not proof["counterfactual_is_physical_sample"]
    assert proof["active_service_upper_ns"] < 3300.
    interval, blocked = config.refresh_interval_ns, proof["refresh_epoch_upper_ns"]
    assert blocked == pytest.approx(598.80)
    assert 1+int(proof["active_service_upper_ns"]//(interval-blocked)) == 1
    assert proof["active_service_upper_ns"]+blocked < proof["row_lifetime_margin_ns"]
    assert (proof["hbm_clock_domain_limit_ns"]
        +proof["hbm_clock_domain_comparison_reserve_ns"]
        <= proof["required_native_hbm_clock_limit_ns"])
    assert config.refresh_policy == "all_bank"


@pytest.mark.parametrize("operation", ("read", "write"))
@pytest.mark.parametrize("warm_operation", ("read", "write"))
@pytest.mark.parametrize("absolute", (0., 1e9+.17, 1e11+.31))
def test_reachable_service_covers_real_native_warm_rows_stale_REF_and_unaligned_bursts(
        operation, warm_operation, absolute):
    from heterollm_sim._compiled_hbm import _pack_hbm_numeric
    from heterollm_sim._hbm_numeric import _new_channel_scratch, _run_hbm_channel_range
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    config = _formal_staging_hbm4()
    for count, address in ((1, 31), (32768, 1), (32768, 131071), (65536, 8177)):
        core = DramCore(config, capture_details=False)
        packed = _pack_hbm_numeric(core, AccessRequest("native-source", "read", 0, 32))
        assert packed.g[8]  # The actual original controller has REF enabled.
        arrays = packed.arrays
        scratch = _new_channel_scratch(packed.g, arrays)
        warm = _run_hbm_channel_range(0, 8*1024**2//32, absolute,
            warm_operation == "read", packed.g, packed.t, *arrays, scratch, 0, 1)
        due = arrays[6][0, 1]
        opened = (arrays[0][:128] >= 0) & (arrays[2][:128] >= 0)
        assert all(arrays[1][:128, 0][opened] >= due-config.refresh_interval_ns-2*config.command_clock_ns)
        # This includes an initial pending DQ tail, the last edge before a
        # due, and stale historical epochs. They all originate from genuine
        # REF-on execution, not hand-written arbitrary expiry deadlines.
        arrival = max(warm[0], due-.001)+7*config.refresh_interval_ns
        origin = max(arrival, float(max(arrays[8][:, 0])))
        proof = native_hbm_refresh_service_upper(config, count, operation,
            independent_channels=True, native_refresh_reachable=True)
        assert proof["qualified"]
        service, blocked = proof["active_service_upper_ns"], proof["refresh_epoch_upper_ns"]
        epochs = 1+int(service//(config.refresh_interval_ns-blocked))
        result = _run_hbm_channel_range(address//32, (address+count-1)//32+1,
            arrival, operation == "read", packed.g, packed.t, *arrays, scratch, 0, 1)
        assert result[0]-origin <= service+epochs*blocked


def test_reachable_tightening_requires_both_source_and_bus_proofs_and_row_margin():
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    config = _formal_staging_hbm4()
    assert not native_hbm_refresh_service_upper(config, 32768, "read",
        independent_channels=True)["qualified"]
    assert not native_hbm_refresh_service_upper(config, 32768, "read",
        native_refresh_reachable=True)["qualified"]
    unknown = replace(config, four_activate_window_ns=16.)
    assert not native_hbm_refresh_service_upper(unknown, 32768, "read",
        independent_channels=True, native_refresh_reachable=True)["qualified"]
    short = replace(config, maximum_active_ns=5000.)
    result = native_hbm_refresh_service_upper(short, 32768, "read",
        independent_channels=True, native_refresh_reachable=True)
    assert not result["qualified"] and "row_lifetime" in result["reason"]


def test_reachable_large_IO_does_not_drop_periodic_feedback_to_one_REF():
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    config = _formal_staging_hbm4()
    proof = native_hbm_refresh_service_upper(config, 65536, "read",
        independent_channels=True, native_refresh_reachable=True)
    assert proof["qualified"]
    service, blocked = proof["active_service_upper_ns"], proof["refresh_epoch_upper_ns"]
    assert 1+int(service//(config.refresh_interval_ns-blocked)) >= 2


def test_reachable_epoch_cost_pays_the_actual_PRE_to_PRE_gate():
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    normal = _formal_staging_hbm4()
    slow = replace(normal, precharge_to_precharge_ns=100.)
    proofs = [native_hbm_refresh_service_upper(config, 32768, "read",
        independent_channels=True, native_refresh_reachable=True)
        for config in (normal, slow)]
    assert all(proof["qualified"] for proof in proofs)
    assert proofs[1]["source_precharge_tail_upper_ns"] == 100.
    assert proofs[1]["refresh_epoch_upper_ns"] > proofs[0]["refresh_epoch_upper_ns"]


def test_actual_two_PC_refresh_reserves_both_tRP_waits_on_the_shared_row_calendar():
    from heterollm_sim._compiled_hbm import _pack_hbm_numeric
    from heterollm_sim._hbm_numeric import (_new_channel_scratch,
        _refresh_until, _run_hbm_channel_range)
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    # A legal short tRCD exposes the calendar ordering without relying on
    # an arbitrary restored ACT image. Both rows and all recovery debt are
    # created by the original native write command machine near the due.
    config = _formal_staging_hbm4(activate_to_read_ns=0., activate_to_write_ns=0.)
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False),
        AccessRequest("two-native-PCs", "write", 0, 64))
    arrays = packed.arrays
    result = _run_hbm_channel_range(0, 2, 3890., False, packed.g, packed.t,
        *arrays, _new_channel_scratch(packed.g, arrays), 0, 1)
    assert arrays[0][0] >= 0 and arrays[0][64] >= 0
    due = arrays[6][0, 1]
    _refresh_until(0, due, result[0], packed.g, packed.t,
        arrays[0], arrays[1], arrays[2], arrays[3], arrays[5], arrays[6],
        arrays[7], arrays[8], arrays[11], result[4])
    actual = max(arrays[3][:2, 1])-due
    proof = native_hbm_refresh_service_upper(config, 32768, "write",
        independent_channels=True, native_refresh_reachable=True)
    corrected = proof["refresh_epoch_upper_ns"]
    old_single_tRP = corrected-config.precharge_to_activate_ns
    assert actual > old_single_tRP
    assert actual <= corrected


@pytest.mark.parametrize("operation", ("read", "write"))
@pytest.mark.parametrize("warm_operation", ("read", "write"))
def test_initial_row_guard_covers_actual_REF_on_wrong_rows_and_fresh_same_arrival(
        operation, warm_operation):
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = _formal_staging_hbm4()
    count = 32768
    target = config.lane_count*config.bank_count*config.row_bytes
    source = DramCore(config, capture_details=False)
    warm = source.submit(AccessRequest("source-row", warm_operation, 0, count, 0.))
    actual = source.submit(AccessRequest("different-row", operation, target, count,
        warm.completion_ns))
    fresh = DramCore(config, capture_details=False).submit(AccessRequest("fresh",
        operation, target, count, warm.completion_ns))
    assert actual.counters["row_conflicts"] == count//config.burst_bytes
    assert actual.counters["refresh_commands"] == 0
    difference = abs(actual.completion_ns-fresh.completion_ns)
    proof = native_hbm_initial_row_upper(config, count, operation,
        independent_channels=True, native_refresh_reachable=True)
    assert proof["qualified"] and not proof["physical_state_advanced"]
    assert proof["initial_bank_upper"] == 33
    assert proof["untouched_old_row_expiry_excluded"]
    assert difference <= proof["initial_state_upper_ns"]
    if operation == warm_operation == "read":
        assert actual.completion_ns-warm.completion_ns == pytest.approx(567.68)
        assert fresh.completion_ns-warm.completion_ns == pytest.approx(76.16)
        assert difference == pytest.approx(491.52)
        # The former guard paid just one shared-array latency plus CA
        # occupancy. Even charging it on both sides missed this real case.
        old_row_guard = config.activate_cycle_ns+16*config.command_clock_ns*2
        assert difference > 2*old_row_guard


def test_initial_row_touched_reduction_requires_source_and_real_bus_independence():
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = _formal_staging_hbm4()
    proven = native_hbm_initial_row_upper(config, 32768, "read",
        independent_channels=True, native_refresh_reachable=True)
    unknown = native_hbm_initial_row_upper(config, 32768, "read",
        independent_channels=True)
    shared = native_hbm_initial_row_upper(config, 32768, "read",
        native_refresh_reachable=True)
    assert proven["initial_bank_upper"] == 33
    assert unknown["initial_bank_upper"] == 2*config.bank_count
    assert shared["initial_bank_upper"] == config.lane_count*config.bank_count
    assert unknown["old_row_expiry_horizon_upper_ns"] == config.maximum_active_ns
    assert shared["old_row_expiry_horizon_upper_ns"] == config.maximum_active_ns
    assert shared["initial_state_upper_ns"] > unknown["initial_state_upper_ns"] > proven["initial_state_upper_ns"]


@pytest.mark.parametrize("changes", ({"maximum_active_ns": 5000.},
    {"four_activate_window_ns": 16.},
    {"address_mapping": "row_local", "interleave_bytes": None}))
def test_row_touched_reduction_also_requires_this_IO_REF_lifetime_and_clock_proof(changes):
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = replace(_formal_staging_hbm4(), **changes)
    proof = native_hbm_initial_row_upper(config, 32768, "read",
        independent_channels=True, native_refresh_reachable=True)
    assert proof["qualified"]
    assert not proof["untouched_old_row_expiry_excluded"]
    assert proof["initial_bank_upper"] == 2*config.bank_count
    assert proof["old_row_expiry_horizon_upper_ns"] == config.maximum_active_ns
    assert "hbm_clock_domain_limit_ns" not in proof


@pytest.mark.parametrize("count,epochs,expected", ((32768, 1, 816.8),
    (65536, 2, 1527.12)))
def test_unique_bank_row_graph_pays_a_complete_bridge_for_every_REF_free_segment(
        count, epochs, expected):
    from heterollm_sim.native_hbm_cold_bound import (native_hbm_initial_row_upper,
        native_hbm_refresh_service_upper)
    config = _formal_staging_hbm4()
    proof = native_hbm_initial_row_upper(config, count, "read",
        independent_channels=True, native_refresh_reachable=True)
    unique = proof["unique_bank_initial_row_proof"]
    active = native_hbm_refresh_service_upper(config, count, "read",
        independent_channels=True, native_refresh_reachable=True)
    assert unique["qualified"] and unique["applied"]
    assert unique["independently_proven_maximum_REF_epochs"] == epochs
    assert epochs == 1+int(active["active_service_upper_ns"]
        //(config.refresh_interval_ns-active["refresh_epoch_upper_ns"]))
    assert unique["REF_free_segment_upper"] == epochs+1
    assert unique["REF_epoch_blocking_charged_separately"]
    assert unique["complete_history_and_column_bridge_per_segment_ns"] >= (
        config.read_latency_ns+config.burst_bytes/config.effective_lane_bandwidth_gb_s)
    assert proof["initial_state_upper_ns"] == pytest.approx(expected)
    assert proof["initial_state_upper_ns"] < unique["original_complete_serial_initial_state_upper_ns"]


def test_unique_bank_bound_uses_the_global_mapping_period_and_worst_unaligned_burst():
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = _formal_staging_hbm4()
    banks = config.lane_count*config.bank_count
    last_unique_bytes = banks*config.burst_bytes-(config.burst_bytes-1)
    unique = native_hbm_initial_row_upper(config, last_unique_bytes, "read",
        independent_channels=True, native_refresh_reachable=True)
    repeated = native_hbm_initial_row_upper(config, last_unique_bytes+1, "read",
        independent_channels=True, native_refresh_reachable=True)
    assert unique["unique_bank_initial_row_proof"]["maximum_global_burst_count"] == banks
    assert unique["unique_bank_initial_row_proof"]["qualified"]
    assert not repeated["unique_bank_initial_row_proof"]["qualified"]
    assert repeated["unique_bank_initial_row_proof"]["reason"] == "request_may_revisit_a_physical_bank"
    assert repeated["initial_state_upper_ns"] == pytest.approx(
        repeated["initial_history_upper_ns"]
        +2*config.bank_count*repeated["per_initial_bank_startup_upper_ns"]
        +2*config.command_clock_ns)


@pytest.mark.parametrize("operation", ("read", "write"))
@pytest.mark.parametrize("warm_operation", ("read", "write"))
def test_unique_bank_bound_covers_real_hit_wrong_cold_and_REF_edge_states(operation, warm_operation):
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = _formal_staging_hbm4()
    period = config.lane_count*config.bank_count*config.row_bytes
    for count, offset in ((32768, 1), (65536, 31)):
        proof = native_hbm_initial_row_upper(config, count, operation,
            independent_channels=True, native_refresh_reachable=True)
        assert proof["unique_bank_initial_row_proof"]["applied"]
        for gap in (0., 3900.-120., 3900.+.17, 8*3900.+.31):
            completions = []
            for source_kind in ("cold", "hit", "wrong"):
                core = DramCore(config, capture_details=False)
                # The source states are all built by genuine native calls.
                # A row's tRAS/write recovery history is not hand-written.
                warm_address = period if source_kind == "hit" else 0
                warm = core.submit(AccessRequest("old", warm_operation,
                    warm_address, count+64, 0.))
                arrival = warm.completion_ns+gap
                if source_kind == "cold":
                    core = DramCore(config, capture_details=False)
                result = core.submit(AccessRequest("unique", operation,
                    period+offset, count, arrival))
                completions.append(result.completion_ns-arrival)
            # REF phase guards remain paid on both sides. This row-only
            # comparison uses identical arrivals and hence the same due
            # grid, while still exercising native maintenance at edges.
            assert max(completions)-min(completions) <= proof["initial_state_upper_ns"]


@pytest.mark.parametrize("changes", (
    {"read_to_write_ns": 100., "write_to_read_ns": 100.,
     "write_to_read_same_group_ns": 100., "write_to_read_different_group_ns": 100.},
    {"read_latency_ns": 17*.64, "write_latency_ns": 4*.64},
    {"read_bandwidth_gb_s": 16., "write_bandwidth_gb_s": 16.}))
@pytest.mark.parametrize("operation", ("read", "write"))
def test_unique_bank_difference_keeps_native_old_direction_history_and_bulk_pacing(changes, operation):
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = replace(_formal_staging_hbm4(), **changes)
    count = 32768
    target = config.lane_count*config.bank_count*config.row_bytes
    other = "write" if operation == "read" else "read"
    warm = DramCore(config, capture_details=False)
    prefix = warm.submit(AccessRequest("opposite", other, 0, count, 0.))
    actual = warm.submit(AccessRequest("wrong", operation, target, count,
        prefix.completion_ns))
    cold = DramCore(config, capture_details=False).submit(AccessRequest("cold",
        operation, target, count, prefix.completion_ns))
    proof = native_hbm_initial_row_upper(config, count, operation,
        independent_channels=True, native_refresh_reachable=True)
    if "read_to_write_ns" in changes:
        # A large analog turnaround is outside the native integer-column
        # translation proof. Keeping the complete conservative row state
        # is required; this case must not silently obtain the new branch.
        assert not proof["unique_bank_initial_row_proof"]["qualified"]
        assert proof["old_row_expiry_horizon_upper_ns"] == config.maximum_active_ns
    else:
        assert proof["unique_bank_initial_row_proof"]["qualified"]
    assert abs(actual.completion_ns-cold.completion_ns) <= proof["initial_state_upper_ns"]
    if config.read_bandwidth_gb_s == 16. and operation == "read":
        # This is an initial-state perturbation bound around a common
        # native COL/DQ graph. The substantial common bulk/pacing service
        # must remain paid: the row bound is deliberately not a total
        # service upper and cannot replace native byte/bandwidth service.
        assert cold.completion_ns-prefix.completion_ns > proof["initial_state_upper_ns"]


@pytest.mark.parametrize("arrival", (1e9+.17, 1.7e11+.31))
@pytest.mark.parametrize("operation", ("read", "write"))
def test_unique_bank_initial_difference_retains_the_declared_large_clock_domain(arrival, operation):
    from heterollm_sim.native_hbm_cold_bound import native_hbm_initial_row_upper
    config = _formal_staging_hbm4()
    count = 32768
    target = config.lane_count*config.bank_count*config.row_bytes
    core = DramCore(config, capture_details=False)
    source = core.submit(AccessRequest("old-clock", operation, 0, count, arrival))
    current = core.submit(AccessRequest("wrong-clock", operation, target+31, count,
        source.completion_ns))
    cold = DramCore(config, capture_details=False).submit(AccessRequest("fresh-clock",
        operation, target+31, count, source.completion_ns))
    proof = native_hbm_initial_row_upper(config, count, operation,
        independent_channels=True, native_refresh_reachable=True)
    assert current.completion_ns < proof["hbm_clock_domain_limit_ns"]
    assert abs(current.completion_ns-cold.completion_ns) <= proof["initial_state_upper_ns"]


def test_merged_private_segment_REF_is_charged_once_to_its_first_member():
    summary = _native_segment_summary()
    proof = summary["native_resource_order_proof"][0]
    segments = proof["closed_hbm_busy_segments"]
    phases = segments["segment_phase_by_task"]
    segment = segments["segments"][0]
    assert sorted(phases, key=int) == [str(index) for index in range(10)]
    saving = segment["old_per_access_refresh_upper_ns"]-segment["segment_refresh_upper_ns"]
    assert saving > 0
    assert sum(phases.values()) == pytest.approx(proof["serial_access_phase_ns"]-saving)
    assert phases["0"] == max(phases.values())


def test_unqualified_or_split_segments_export_only_their_own_reassignment():
    assert _native_segment_summary(alias=True)["native_resource_order_proof"][0][
        "closed_hbm_busy_segments"]["segment_phase_by_task"] == {}
    split = _native_segment_summary(foreign_wait=True)["native_resource_order_proof"][0][
        "closed_hbm_busy_segments"]
    phases = split["segment_phase_by_task"]
    assert set(phases) <= {str(index) for index in range(10)}
    for segment in split["segments"]:
        if segment["qualified"] and segment["segment_refresh_upper_ns"] < segment["old_per_access_refresh_upper_ns"]:
            assert segment["task_count"] == 5
