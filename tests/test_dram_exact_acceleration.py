from dataclasses import replace

import pytest

from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.dram_core import DramCore, _repeat_add
from heterollm_sim.memory_types import AccessRequest, DramConfig, MemoryKind, Operation, parse_physical_memory_config


class BurstReference(DramCore):
    """The original state-transition loop, without the accelerated dispatch."""

    def _can_accelerate(self):
        return False


def _native():
    return parse_physical_memory_config(get_component_preset(
        "gddr7-16gb-30_0-256bit",
    ).component.metadata["physical_memory_config"])


def _generic(**changes):
    config = DramConfig(
        kind=MemoryKind.GDDR, data_lanes=2, bank_groups_per_rank=2,
        banks_per_group=2, rows_per_bank=1024, row_bytes=8192,
        interface_bandwidth_gb_s=80.0, max_expanded_segments=8192,
    )
    return replace(config, capacity_bytes=None, **changes)


def _assert_floats(actual, expected, *, tolerance=0.00001):
    assert actual.keys() == expected.keys()
    for key in expected:
        value = expected[key]
        if isinstance(value, (list, tuple)):
            assert actual[key] == pytest.approx(value, rel=0, abs=tolerance)
        else:
            assert actual[key] == pytest.approx(value, rel=0, abs=tolerance)


def _assert_state(actual, expected):
    assert actual._banks.keys() == expected._banks.keys()
    for key in expected._banks:
        assert actual._banks[key].open_row == expected._banks[key].open_row
        assert actual._banks[key].ready_ns == pytest.approx(expected._banks[key].ready_ns, rel=0, abs=0.00001)
    for field in ("ready_ns", "lane_available", "busy_ns", "last_intervals"):
        _assert_floats(getattr(actual.timeline, field), getattr(expected.timeline, field))
    assert actual.timeline.bytes_moved == expected.timeline.bytes_moved
    assert actual.timeline.directions == expected.timeline.directions
    assert actual.timeline._touched == expected.timeline._touched
    assert actual._inflight == pytest.approx(expected._inflight, rel=0, abs=0.00001)
    assert actual._acceptance_ns == pytest.approx(expected._acceptance_ns, rel=0, abs=0.00001)


def _compare_requests(config, requests, *, seed=False, multi_slot=False):
    actual = DramCore(config, capture_details=False)
    expected = BurstReference(config, capture_details=False)
    if seed:
        for core in (actual, expected):
            for lane in range(config.lane_count):
                core.timeline.reserve(f"dram:command:{lane}", 0.0, 50000.0 + lane)
                core.timeline.transfer(f"dram:data:{lane}", 64, 900.0 + lane, 1.0, direction="write")
            core.timeline.reserve("dram:bank:0:0:0:0:0:0", 0.0, 30000.0)
    if multi_slot:
        for core in (actual, expected):
            core.timeline.lane_available["dram:data:0"] = [5.0, 7.0]
            core.timeline.ready_ns["dram:data:0"] = 5.0
    results = []
    for request in requests:
        a, e = actual.submit(request), expected.submit(request)
        assert a.completion_ns == pytest.approx(e.completion_ns, rel=0, abs=0.00001)
        assert a.logical_bytes == e.logical_bytes
        assert a.transfer_bytes == e.transfer_bytes
        for key in ("row_hits", "row_misses", "row_conflicts", "burst_count",
                    "physical_read_bytes", "physical_write_bytes"):
            assert a.counters[key] == e.counters[key]
        _assert_state(actual, expected)
        results.append((a, e))
    return results


@pytest.mark.parametrize("config,address,size,seed", [
    (_native(), 0, 6 * 1024 * 1024, False),
    (replace(_native(), metadata={"shared_command_resource": "dram:command:shared"}), 64, 9217 * 64 - 9, False),
    (_generic(data_lanes=3, banks_per_group=1, row_bytes=512, interleave_bytes=128,
              metadata={"shared_command_resource": "dram:command:shared"}), 193, 10000 * 64 + 7, False),
    (_generic(burst_interval_ns=0.1, read_bandwidth_gb_s=20, interleave_bytes=256,
              metadata={"shared_command_resource": "dram:command:shared"}), 193, 10000 * 64 + 7, False),
    (_generic(), 193, 10000 * 64 + 7, True),
    (_generic(interleave_bytes=128), 193, 10000 * 64 + 7, False),
    (_generic(interleave_bytes=256), 193, 10000 * 64 + 7, False),
    (_generic(read_to_write_ns=37, write_to_read_ns=71,
              read_recovery_ns=11, write_recovery_ns=19,
              read_bandwidth_gb_s=40, write_bandwidth_gb_s=20), 17, 9217 * 64 - 11, False),
    (_generic(burst_interval_ns=0, open_ns=0, close_ns=0), 64, 10000 * 64 + 7, False),
    (_generic(max_outstanding_requests=1), 64, 10000 * 64 + 7, False),
    (_generic(row_bytes=512, stacks=2, dies_per_stack=2), 2 * 4 * 1024 * 512 - 1024, 10000 * 64 + 7, False),
    (_generic(row_bytes=512, stacks=2, dies_per_stack=2), 2 * 2 * 4 * 1024 * 512 - 1024, 10000 * 64 + 7, False),
])
def test_acceleration_matches_burst_state_and_mixed_followups(config, address, size, seed):
    requests = (
        AccessRequest("warm", Operation.READ, address, 64, 0.0),
        AccessRequest("large-read", Operation.READ, address, size, 0.0),
        AccessRequest("large-write", Operation.WRITE, address, size, 0.0),
        AccessRequest("large-read-again", Operation.READ, address, size, 0.0),
        AccessRequest("same-last-row", Operation.READ, address + size - 64, 64, 1000000.0),
        AccessRequest("other-row-write", Operation.WRITE, address + size + 8192, 64, 1000000.0),
    )
    results = _compare_requests(config, requests, seed=seed)
    assert results[1][0].counters["accelerated_row_hit_bursts"] > 0


@pytest.mark.parametrize("size", [8192 * 64, 8193 * 64, 8193 * 64 + 1])
def test_threshold_does_not_change_the_state_model(size):
    results = _compare_requests(_native(), (
        AccessRequest("boundary", Operation.READ, 0, size, 0.0),
        AccessRequest("probe", Operation.WRITE, size - 64, 64, 0.0),
    ))
    if size > 8192 * 64:
        assert results[0][0].counters["accelerated_row_hit_bursts"] > 0


@pytest.mark.parametrize("multi_slot,interleave,row", [(True, 64, 8192), (False, 16384, 8192), (False, 128, 192)])
def test_nonperiodic_or_multiple_slot_calendar_falls_back(multi_slot, interleave, row):
    results = _compare_requests(_generic(interleave_bytes=interleave, row_bytes=row), (
        AccessRequest("fallback", Operation.READ, 3, 9217 * 64, 0.0),
        AccessRequest("probe", Operation.WRITE, 128, 64, 0.0),
    ), multi_slot=multi_slot)
    assert "accelerated_row_hit_bursts" not in results[0][0].counters


def test_detailed_capture_retains_burst_mapping_and_stages():
    config = _native()
    request = AccessRequest("detailed", Operation.READ, 64, 8193 * 64, 0.0)
    actual = DramCore(config).submit(request)
    expected = BurstReference(config).submit(request)
    assert actual == expected


def test_nested_resource_prefix_alias_falls_back():
    config = _generic(metadata={"data_resource_prefix": "dram:bank:0:0:0:0:0"})
    results = _compare_requests(config, (
        AccessRequest("alias", Operation.READ, 64, 9217 * 64, 0.0),
        AccessRequest("probe", Operation.WRITE, 128, 64, 0.0),
    ))
    assert "accelerated_row_hit_bursts" not in results[0][0].counters


@pytest.mark.parametrize("start,increment,count", [
    (0.0, 64 / 120, 100000), (1e12, 64 / 120, 10000),
    (1023.9, 0.1, 20000), (0.0, 0.0, 1000),
    (2.0 ** 50, 0.125, 1000), (2.0 ** 50 + 0.25, 0.125, 1000),
    (2.0 ** 50, 0.375, 1000), (2.0 ** 50 + 0.25, 0.375, 1000),
    (1e308, 1e293, 3), (1e308, 1e307, 1000),
])
def test_repeated_add_preserves_serial_rounding(start, increment, count):
    expected = start
    for _ in range(count):
        expected += increment
    assert _repeat_add(start, increment, count) == expected


def test_large_absolute_arrival_preserves_burst_state():
    _compare_requests(_native(), (
        AccessRequest("large-clock", Operation.READ, 64, 9217 * 64 - 9, 1e12),
        AccessRequest("switch", Operation.WRITE, 128, 64, 1e12),
    ))


def test_unrelated_parallel_resource_keeps_memory_acceleration():
    core = DramCore(_native(), capture_details=False)
    core.timeline.lane_available["gpu0.compute"] = [0.0, 10.0]
    result = core.submit(AccessRequest("unrelated", Operation.READ, 0, 9217 * 64, 0.0))
    assert result.counters["accelerated_row_hit_bursts"] > 0
    assert core.timeline.lane_available["gpu0.compute"] == [0.0, 10.0]


@pytest.mark.parametrize("kind", [MemoryKind.DDR, MemoryKind.LPDDR, MemoryKind.HBM, MemoryKind.GDDR])
def test_all_dram_families_use_the_same_exact_state_transition(kind):
    config = _generic(
        kind=kind, channels=2, subchannels_per_channel=2,
        pseudo_channels_per_channel=2, data_lanes=3,
        read_to_write_ns=37, write_to_read_ns=71,
        read_recovery_ns=11, write_recovery_ns=19,
        read_bandwidth_gb_s=40, write_bandwidth_gb_s=20,
        metadata={"shared_command_resource": "dram:command:shared"},
    )
    results = _compare_requests(config, (
        AccessRequest("warm", Operation.READ, 193, 64, 0.0),
        AccessRequest("family-read", Operation.READ, 193, 10000 * 64 + 7, 0.0),
        AccessRequest("family-write", Operation.WRITE, 193, 10000 * 64 + 7, 0.0),
        AccessRequest("family-read-after-write", Operation.READ, 193, 10000 * 64 + 7, 0.0),
        AccessRequest("last-row-probe", Operation.WRITE, 10000 * 64 + 128, 64, 1000000.0),
    ))
    assert results[1][0].counters["accelerated_row_hit_bursts"] > 0
    assert results[2][0].counters["accelerated_row_hit_bursts"] > 0


@pytest.mark.parametrize("kind", [MemoryKind.DDR, MemoryKind.LPDDR, MemoryKind.HBM, MemoryKind.GDDR])
@pytest.mark.parametrize("bursts", [512, 2048, 8192])
def test_small_projection_uses_exact_acceleration_for_all_families(kind, bursts):
    config = replace(_native(), kind=kind, generation="GDDR7" if kind is MemoryKind.GDDR else "")
    results = _compare_requests(config, (
        AccessRequest("small-projection-read", Operation.READ, 64, bursts * 64, 0.0),
        AccessRequest("small-projection-write", Operation.WRITE, 64, bursts * 64, 0.0),
        AccessRequest("small-projection-probe", Operation.READ, bursts * 64, 64, 1000000.0),
    ))
    assert results[0][0].counters["accelerated_row_hit_bursts"] > 0
    assert results[1][0].counters["accelerated_row_hit_bursts"] > 0
