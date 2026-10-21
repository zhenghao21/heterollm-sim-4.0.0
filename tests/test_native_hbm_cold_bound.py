"""Counterfactual native bounds never become serving physical samples."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from exploring_hbf_paper_config import build_paper_hbm_config
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.native_hbm_cold_bound import native_hbm_no_refresh_upper
from heterollm_sim.native_window_producer import _refresh_serial_bound, _refresh_serial_terms


@pytest.mark.parametrize("operation", ("read", "write"))
def test_full_cycle_native_bound_has_explicit_absolute_clock_and_row_lifetime_guards(operation):
    config = build_paper_hbm_config("HBM4")
    proof = native_hbm_no_refresh_upper(config, 128*1024**2, operation, independent_channels=True)
    assert proof["qualified"] and not proof["physical_state_advanced"]
    assert not proof["counterfactual_is_physical_sample"]
    assert proof["padded_byte_count"] == 160*1024**2
    assert proof["new_row_residence_upper_ns"] < config.maximum_active_ns
    assert proof["required_native_hbm_clock_limit_ns"] > 1e11
    assert proof["conditional_no_refresh_serial_upper_ns"] < 50_000.
    # The real normalized physical configuration remains unchanged.
    assert config.refresh_policy == "all_bank" and config.maximum_active_ns == 35100.


@pytest.mark.parametrize("operation", ("read", "write"))
def test_native_bound_covers_warm_opposite_direction_rows_and_unaligned_address_tails(operation):
    from heterollm_sim.native_hbm_cold_bound import _prepare
    from heterollm_sim._hbm_numeric import _run_hbm_channel_range
    config = build_paper_hbm_config("HBM4")
    proof = native_hbm_no_refresh_upper(config, 128*1024**2, operation, independent_channels=True)
    other = "write" if operation == "read" else "read"
    for address, offset in ((1, .001), (31, .17), (8177, .32), (33554431, 1.07)):
        packed, scratch = _prepare(config)
        assert not packed.g[8] and config.refresh_policy == "all_bank"
        arrays = packed.arrays
        _run_hbm_channel_range(0, 32*1024**2//32, 0., other == "read",
            packed.g, packed.t, *arrays, scratch, 0, packed.g[0]//2)
        arrival = float(max(arrays[8][:, 0]))+offset
        first, stop = address//32, (address+128*1024**2-1)//32+1
        result = _run_hbm_channel_range(first, stop, arrival, operation == "read",
            packed.g, packed.t, *arrays, scratch, 0, packed.g[0]//2)
        assert result[0]-arrival <= proof["conditional_no_refresh_serial_upper_ns"]


def test_cold_tightening_keeps_active_refresh_feedback_instead_of_one_tRFC():
    config = build_paper_hbm_config("HBM4")
    interval, blocked, service = _refresh_serial_terms(config, 128*1024**2, "read",
        independent_channels=True, allow_bounded_clock=True)
    guard = _refresh_serial_bound(config, 128*1024**2, "read", independent_channels=True,
        allow_bounded_clock=True)
    assert service < 50_000.
    assert guard == (1+int(service//(interval-blocked)))*blocked
    assert guard > config.refresh_cycle_ns


def test_native_cold_clock_translation_covers_large_absolute_grid_indices():
    from heterollm_sim.native_hbm_cold_bound import _prepare
    from heterollm_sim._hbm_numeric import _run_hbm_channel_range
    config = build_paper_hbm_config("HBM4")
    count = 128*1024**2
    proof = native_hbm_no_refresh_upper(config, count, "read", independent_channels=True)
    for arrival in (1e9+.17, 1e11+.31):
        packed, scratch = _prepare(config)
        address = 33554431
        result = _run_hbm_channel_range(address//32, (address+count-1)//32+1,
            arrival, True, packed.g, packed.t, *packed.arrays, scratch, 0, packed.g[0]//2)
        assert result[0] <= proof["required_native_hbm_clock_limit_ns"]
        assert result[0]-arrival <= proof["conditional_no_refresh_serial_upper_ns"]


def test_initial_state_allowance_pays_the_actual_per_bank_tRP_ACT_startup_chain():
    from heterollm_sim.native_hbm_cold_bound import _prepare, _cold_duration
    from heterollm_sim._hbm_numeric import _run_hbm_channel_range
    config = build_paper_hbm_config("HBM4")
    period = config.lane_count*config.bank_count*config.row_bytes
    sweep = config.lane_count*config.bank_count*config.burst_bytes
    proof = native_hbm_no_refresh_upper(config, 128*1024**2, "read", independent_channels=True)
    prepared = _prepare(config)
    first_cold_sweep = _cold_duration(prepared, sweep, "read")
    packed, scratch = prepared
    warm = _run_hbm_channel_range(0, period//32, 0., True,
        packed.g, packed.t, *packed.arrays, scratch, 0, 1)
    arrival = warm[0]
    wrong_rows = _run_hbm_channel_range(period//32, (period+sweep)//32, arrival, True,
        packed.g, packed.t, *packed.arrays, scratch, 0, 1)
    excess = wrong_rows[0]-arrival-first_cold_sweep
    assert excess > 1000.  # The old 311ns PRE-only allowance was insufficient.
    assert excess <= proof["initial_state_upper_ns"]


@pytest.mark.parametrize("operation", ("read", "write"))
def test_tighter_feedback_covers_actual_native_REF_stalls_at_address_and_grid_boundaries(operation):
    from heterollm_sim.native_hbm_cold_bound import _prepare
    from heterollm_sim._hbm_numeric import _run_hbm_channel_range
    config = build_paper_hbm_config("HBM4")
    count = 128*1024**2
    guard = _refresh_serial_bound(config, count, operation, independent_channels=True,
        allow_bounded_clock=True)
    for address, arrival in ((0, 0.), (1, 3899.999), (33554431, 3900.), (1234567, 1e9+.17)):
        completions = []
        for refresh in (False, True):
            packed, scratch = _prepare(config)
            packed.g[8] = int(refresh)
            result = _run_hbm_channel_range(address//32, (address+count-1)//32+1,
                arrival, operation == "read", packed.g, packed.t, *packed.arrays,
                scratch, 0, packed.g[0]//2)
            completions.append(result[0])
        assert completions[1]-completions[0] <= guard


def test_unclosed_cohort_PD_and_direct_phase_paths_keep_the_original_upper():
    from heterollm_sim.native_window_producer import _phase
    config = build_paper_hbm_config("HBM4")
    old = _refresh_serial_terms(config, 128*1024**2, "read", independent_channels=True)
    guarded = _refresh_serial_terms(config, 128*1024**2, "read",
        independent_channels=True, allow_bounded_clock=True)
    assert old[2] > 3_000_000. and guarded[2] < old[2]/10
    assert _phase(config, 128*1024**2, "read", independent_hbm_channels=True)[1] > 100_000.


def test_summary_reserves_the_padded_comparison_before_the_raw_clock_limit():
    from heterollm_sim.contracts import _PreparedExecutionStage
    from heterollm_sim.memory_transfer import ResourceTimeline
    from heterollm_sim.native_window_producer import summarize_native_stages
    from test_native_window_producer import task
    config = build_paper_hbm_config("HBM4")
    item = task("weight", 0, (), ("gpu0",), {
        "physical_owner": "memory", "physical_memory_config": config,
        "memory_accesses": ({"operation": "read", "byte_count": 128*1024**2,
            "physical_owner": "memory", "resource_id": "memory"},)})
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., (item,))
    physical = NS(timeline=ResourceTimeline(), runtimes={}, allocators={})
    coarse = summarize_native_stages((stage,), NS(), physical)
    bounded = summarize_native_stages((stage,), NS(), physical, allow_bounded_clock=True)
    proof = bounded["owner_accesses"]["memory"]["native_no_refresh_cold_bounds"][0]
    assert coarse["hbm_clock_domain_limit_ns"] is None
    assert bounded["hbm_clock_domain_limit_ns"] < proof["required_native_hbm_clock_limit_ns"]
    assert bounded["hbm_clock_domain_limit_ns"]+bounded["hbm_clock_domain_comparison_reserve_ns"] <= proof["required_native_hbm_clock_limit_ns"]
    assert bounded["ordered_resource_phase_per_cohort_ns"] < coarse["ordered_resource_phase_per_cohort_ns"]


@pytest.mark.parametrize("changes", ({"address_mapping": "row_local", "interleave_bytes": None},
    {"four_activate_window_ns": 16.}, {"activate_same_group_ns": 1.28}))
def test_unknown_native_dependency_profiles_do_not_get_a_cold_upper(changes):
    config = replace(build_paper_hbm_config("HBM4"), **changes)
    proof = native_hbm_no_refresh_upper(config, 128*1024**2, "read", independent_channels=True)
    assert not proof["qualified"]


def test_shared_channels_and_capacity_padding_refuse_the_tight_upper():
    config = build_paper_hbm_config("HBM4")
    assert not native_hbm_no_refresh_upper(config, 128*1024**2, "read")["qualified"]
    result = native_hbm_no_refresh_upper(config, config.capacity_bytes, "read", independent_channels=True)
    assert not result["qualified"] and "capacity" in result["reason"]


def test_tRASmax_that_cannot_contain_the_native_row_residence_refuses():
    config = replace(build_paper_hbm_config("HBM4"), maximum_active_ns=1000.)
    result = native_hbm_no_refresh_upper(config, 128*1024**2, "read", independent_channels=True)
    assert not result["qualified"] and "tRASmax" in result["reason"]
