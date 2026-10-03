import math
from dataclasses import replace

import pytest

from heterollm_sim.dram import DramProfile, DramState, dram_service


def profile(**overrides):
    values = dict(
        channels=2,
        banks_per_channel=2,
        burst_bytes=16,
        row_bytes=64,
        t_rcd_ns=2.0,
        t_rp_ns=3.0,
        t_ras_ns=4.0,
        read_latency_ns=1.0,
        write_latency_ns=2.0,
        read_to_write_ns=5.0,
        write_to_read_ns=6.0,
        read_recovery_ns=1.0,
        write_recovery_ns=2.0,
        refresh_interval_ns=100.0,
        refresh_duration_ns=10.0,
        evidence="unit analytical contract",
        aggregate_policy="cold_contiguous",
    )
    values.update(overrides)
    return DramProfile(**values)


def service(p, accesses, **kwargs):
    return dram_service(p, tuple(accesses), read_bandwidth_gb_s=16.0, write_bandwidth_gb_s=16.0, max_outstanding_requests=4, **kwargs)


def test_profile_rejects_bad_geometry_refresh_and_evidence():
    with pytest.raises(ValueError):
        profile(row_bytes=15)
    with pytest.raises(ValueError):
        profile(refresh_interval_ns=0, refresh_duration_ns=1)
    with pytest.raises(ValueError):
        profile(refresh_interval_ns=10, refresh_duration_ns=10)
    with pytest.raises(ValueError):
        profile(evidence="")
    with pytest.raises(ValueError):
        profile(channels=True)


def test_unaligned_access_rounds_to_bursts_and_maps_channel_bank_column_row():
    metrics, state = service(profile(refresh_interval_ns=0, refresh_duration_ns=0), [{"operation": "read", "offset_bytes": 3, "byte_count": 17}])
    assert metrics["burst_count"] == 2
    assert metrics["logical_read_bytes"] == 17
    assert metrics["physical_read_bytes"] == 32
    assert metrics["physical_bytes"] == 32
    assert len(metrics["boundaries"]) == 2
    assert all(0 <= row["channel"] < 2 and 0 <= row["bank"] < 2 for row in metrics["boundaries"])
    assert isinstance(state, DramState)


def test_row_hit_then_conflict_and_state_is_immutable_preview():
    p = profile(refresh_interval_ns=0, refresh_duration_ns=0)
    first, state1 = service(p, [{"operation": "read", "offset_bytes": 0, "byte_count": 16}])
    second, state2 = service(p, [{"operation": "read", "offset_bytes": 64, "byte_count": 16}], state=state1)
    conflict, state3 = service(p, [{"operation": "read", "offset_bytes": 256, "byte_count": 16}], state=state2)
    assert first["row_misses"] == 1
    assert second["row_hits"] == 1
    assert conflict["row_conflicts"] == 1
    assert state1 != state2 != state3
    # Replaying from state1 gives the same result and never mutates state1.
    replay, replay_state = service(p, [{"operation": "read", "offset_bytes": 64, "byte_count": 16}], state=state1)
    assert replay == second
    assert replay_state == state2


def test_channels_and_banks_are_parallel_addressed_and_mixed_turnaround_is_once():
    p = profile(refresh_interval_ns=0, refresh_duration_ns=0)
    metrics, _ = service(p, [
        {"operation": "read", "offset_bytes": 0, "byte_count": 16},
        {"operation": "write", "offset_bytes": 32, "byte_count": 16},
        {"operation": "read", "offset_bytes": 64, "byte_count": 16},
        {"operation": "read", "offset_bytes": 16, "byte_count": 16},
    ])
    assert metrics["turnaround_ns"] >= p.read_to_write_ns + p.write_to_read_ns
    assert metrics["physical_bytes"] == 64
    assert {row["channel"] for row in metrics["boundaries"]} == {0, 1}


def test_refresh_closes_rows_and_queue_budget_rejects_before_state_change():
    p = profile(refresh_interval_ns=10, refresh_duration_ns=4)
    metrics, state = service(p, [{"operation": "read", "offset_bytes": 0, "byte_count": 16}], start_ns=10.0)
    assert metrics["refresh_wait_ns"] > 0
    assert metrics["row_misses"] == 1
    with pytest.raises(ValueError):
        limited = replace(p, max_bursts_per_access=1)
        dram_service(limited, ({"operation": "read", "offset_bytes": 0, "byte_count": 32},), read_bandwidth_gb_s=16, write_bandwidth_gb_s=16, max_outstanding_requests=1)


def test_zero_timing_is_valid_with_evidence_and_output_is_finite():
    p = profile(**{name: 0.0 for name in ("t_rcd_ns", "t_rp_ns", "t_ras_ns", "read_latency_ns", "write_latency_ns", "read_to_write_ns", "write_to_read_ns", "read_recovery_ns", "write_recovery_ns", "refresh_interval_ns", "refresh_duration_ns")})
    metrics, _ = service(p, [{"operation": "read", "offset_bytes": 0, "byte_count": 16}])
    assert math.isfinite(metrics["service_ns"])


def test_optional_dram_organization_coordinates_are_mapped_without_double_charging():
    p = profile(
        channels=2,
        banks_per_channel=8,
        subchannels_per_channel=2,
        ranks_per_channel=2,
        bank_groups_per_channel=4,
        stack_count=2,
        pseudo_channels_per_channel=2,
        refresh_interval_ns=0,
        refresh_duration_ns=0,
    )
    metrics, state = service(p, [{"operation": "read", "offset_bytes": 0, "byte_count": 16 * 256}])
    assert metrics["lane_count"] == 16
    assert metrics["physical_bytes"] == 16 * 256
    assert len(state.channels) == 16
    assert len(state.banks) == 16 * 2 * 8
    assert {row["stack"] for row in metrics["boundaries"]} == {0, 1}
    assert {row["subchannel"] for row in metrics["boundaries"]} == {0, 1}
    assert {row["pseudo_channel"] for row in metrics["boundaries"]} == {0, 1}
    assert {row["rank"] for row in metrics["boundaries"]} == {0, 1}
    assert {row["bank_group"] for row in metrics["boundaries"]} == {0, 1}


def test_unknown_bank_group_geometry_remains_unknown():
    metrics, _ = service(profile(refresh_interval_ns=0, refresh_duration_ns=0), [{"operation": "read", "offset_bytes": 0, "byte_count": 16}])
    assert metrics["boundaries"][0]["bank_group"] is None
