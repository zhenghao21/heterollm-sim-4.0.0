from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig


def test_dram_records_absolute_reservations():
    cfg = DramConfig(banks_per_group=1, rows_per_bank=4, row_bytes=128,
                     burst_bytes=64, open_ns=10, read_latency_ns=5,
                     burst_interval_ns=1, lane_bandwidth_gb_s=64)
    result = DramCore(cfg).execute(AccessRequest("r", "read", 0, 64))
    got = result.counters["resource_intervals"]
    assert got["dram:bank:0:0:0:0:0:0"] == ((0.0, 10.0),)
    assert got["dram:command:0"] == ((10.0, 11.0),)
    assert got["dram:data:0"] == ((15.0, 16.0),)


def test_idle_time_satisfies_direction_gap():
    timeline = ResourceTimeline()
    timeline.transfer("data", 1, 0, 1, direction="read")
    write = timeline.transfer("data", 1, 100, 1, direction="write", switch_ns=10)
    assert write.start_ns == 100


def test_reservation_detail_cap_does_not_change_summary():
    common = dict(banks_per_group=1, rows_per_bank=4, row_bytes=128,
                  burst_bytes=64, open_ns=10, read_latency_ns=5,
                  burst_interval_ns=1, lane_bandwidth_gb_s=64)
    full = DramCore(DramConfig(**common, max_expanded_segments=100)).execute(
        AccessRequest("r", "read", 0, 128))
    low = DramCore(DramConfig(**common, max_expanded_segments=1)).execute(
        AccessRequest("r", "read", 0, 128))
    assert (low.completion_ns, low.latency_ns) == (full.completion_ns, full.latency_ns)
    assert low.counters["details_truncated"]
    assert low.counters["intervals_truncated"]
