"""Behavioral comparisons for mapping, finite scheduling and queue scope."""
from dataclasses import replace

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import map_dram_address, dram_request_channels
from heterollm_sim.memory_types import AccessRequest, DramConfig
from heterollm_sim.physical_presets import resolve_physical_config


def config(**changes):
    values = dict(channels=1, banks_per_group=2, rows_per_bank=8,
                  row_bytes=128, burst_bytes=32, lane_bandwidth_gb_s=32.,
                  open_ns=10., close_ns=10., read_latency_ns=5., write_latency_ns=5.,
                  burst_interval_ns=1., write_recovery_ns=0.,
                  address_mapping="row_local")
    values.update(changes)
    return DramConfig(**values)


@pytest.mark.parametrize("patch", [dict(address_mapping="unknown"), dict(scheduler="frfcfs"),
    dict(scheduler_window=0), dict(scheduler_window=True), dict(queue_scope="bank"),
    dict(interleave_bytes=32)])
def test_invalid_controller_choices_are_rejected(patch):
    with pytest.raises(ValueError):
        config(**patch)


@pytest.mark.parametrize("layout", ["bank_interleaved", "row_local"])
@pytest.mark.parametrize("scope", ["per_die", "per_stack"])
@pytest.mark.parametrize("interface", ["shared", "independent"])
def test_non_power_of_two_mapping_is_bijective_and_channel_coverage_matches(layout, scope, interface):
    c = config(address_mapping=layout, channels=3, banks_per_group=3,
               stacks=2, dies_per_stack=3, rows_per_bank=3,
               bank_geometry_scope=scope, stack_interface_mode=interface)
    coordinates = set()
    for address in range(0, c.capacity_bytes, c.burst_bytes):
        m = map_dram_address(c, address)
        coordinates.add((m.stack, m.die, m.lane, m.rank, m.bank_group, m.bank, m.row, m.column))
        assert m.row < c.rows_per_bank
    assert len(coordinates) == c.capacity_bytes // c.burst_bytes
    for start, length in ((0, 1), (31, 130), (127, 600), (c.capacity_bytes - 257, 257), (0, c.capacity_bytes)):
        request = AccessRequest("coverage", "read", start, length)
        expected = set()
        for address in range(start // 32 * 32, start + length, 32):
            m = map_dram_address(c, address)
            expected.add((m.stack if interface == "independent" else 0, m.channel))
        assert set(dram_request_channels(c, request)) == expected


def test_row_local_24_gbit_12_high_standard_geometry_keeps_last_address():
    c = resolve_physical_config("hbm3e_reference", overrides={
        "capacity_bytes": 36 * 1024**3, "stacks": 1,
        "dies_per_stack": 12, "die_density_gbits": 24,
        "address_mapping": "row_local", "interleave_bytes": None})
    last = map_dram_address(c, c.capacity_bytes - 1)
    assert (last.lane, last.bank_group, last.bank, last.row, last.column) == (31, 11, 3, 24575, 31)
    assert map_dram_address(c, 32).lane == 0
    assert map_dram_address(c, c.row_bytes).bank == 1


def test_row_hit_first_reorders_inside_window_and_returns_input_order():
    results = {}
    for policy in ("fcfs", "row_hit_first"):
        core = DramCore(config(scheduler=policy))
        core.submit(AccessRequest("warm", "read", 0, 32))
        results[policy] = core.run([
            AccessRequest("miss", "read", 256, 32, 100),
            AccessRequest("hit", "read", 32, 32, 100)])
    changed = results["row_hit_first"]
    assert [r.request_id for r in changed.requests] == ["miss", "hit"]
    assert changed.requests[1].completion_ns < changed.requests[0].completion_ns
    assert sum(r.counters["row_conflicts"] for r in changed.requests) == 1
    assert changed.duration_ns < results["fcfs"].duration_ns
    assert sum(r.counters["scheduler_reordered_requests"] for r in changed.requests) == 1
    assert sum(r.counters["scheduler_batches"] for r in changed.requests) == 1


@pytest.mark.parametrize("window,hit_arrival", [(1, 100.), (16, 1000.)])
def test_scheduler_never_looks_beyond_window_or_waits_for_future_hit(window, hit_arrival):
    core = DramCore(config(scheduler="row_hit_first", scheduler_window=window))
    core.submit(AccessRequest("warm", "read", 0, 32))
    miss, hit = core.submit_many([
        AccessRequest("miss", "read", 256, 32, 100),
        AccessRequest("hit", "read", 32, 32, hit_arrival)])
    assert miss.completion_ns < hit.completion_ns
    assert sum(r.counters["scheduler_reordered_requests"] for r in (miss, hit)) == 0
    assert hit.arrival_ns == hit_arrival


def test_read_write_batch_reduces_turnarounds_and_measured_latency():
    traces = {}
    requests = [AccessRequest(str(i), "read" if i % 2 == 0 else "write", i * 32, 32)
                for i in range(4)]
    for policy in ("fcfs", "read_write_batch"):
        traces[policy] = DramCore(config(scheduler=policy, read_to_write_ns=20,
                                        write_to_read_ns=20)).run(requests)
    assert sum(r.counters["controller_direction_switches"] for r in traces["fcfs"].requests) == 3
    assert sum(r.counters["controller_direction_switches"] for r in traces["read_write_batch"].requests) == 1
    assert traces["read_write_batch"].duration_ns < traces["fcfs"].duration_ns


@pytest.mark.parametrize("policy", ["row_hit_first", "read_write_batch"])
@pytest.mark.parametrize("first,second", [("read", "write"), ("write", "read"), ("write", "write")])
def test_overlap_dependencies_wait_for_completion_even_with_disjoint_logical_bytes(policy, first, second):
    core = DramCore(config(scheduler=policy, write_latency_ns=1))
    before, after = core.submit_many([
        AccessRequest("before", first, 0, 1), AccessRequest("after", second, 31, 1)])
    assert after.arrival_ns == 0
    assert after.queue_wait_ns >= before.completion_ns
    assert min(stage.start_ns for stage in after.stages) >= before.completion_ns


def test_independent_arrived_request_passes_a_future_dependency_chain():
    core = DramCore(config(scheduler="read_write_batch"))
    future_write, dependent_read, ready_read = core.submit_many([
        AccessRequest("future", "write", 0, 32, 1000),
        AccessRequest("dependent", "read", 0, 32),
        AccessRequest("ready", "read", 128, 32)])
    assert ready_read.completion_ns < 1000
    assert dependent_read.queue_wait_ns >= future_write.completion_ns


def test_channel_credits_allow_independent_channels_and_charge_multichannel_aggregates():
    c = config(address_mapping="bank_interleaved", channels=2, max_outstanding_requests=1)
    requests = [AccessRequest("a", "read", 0, 32), AccessRequest("b", "read", 32, 32)]
    device = DramCore(c).submit_many(requests)
    channel = DramCore(replace(c, queue_scope="channel")).submit_many(requests)
    assert device[1].queue_wait_ns == device[0].completion_ns
    assert channel[1].queue_wait_ns == 0
    core = DramCore(replace(c, queue_scope="channel"))
    aggregate = core.submit(AccessRequest("both", "read", 0, 64))
    follow = core.submit(AccessRequest("one", "read", 32, 32))
    assert follow.queue_wait_ns == aggregate.completion_ns
    assert set(core._channel_inflight) == {(0, 0), (0, 1)}


def test_row_hit_policy_can_issue_another_channel_when_one_queue_is_full():
    c = config(address_mapping="bank_interleaved", channels=2, max_outstanding_requests=1,
               scheduler="row_hit_first", queue_scope="channel")
    core = DramCore(c)
    busy = core.submit(AccessRequest("busy", "read", 0, 32))
    same, independent = core.submit_many([
        AccessRequest("same", "read", 0, 32), AccessRequest("independent", "read", 32, 32)])
    assert independent.queue_wait_ns == 0
    assert same.queue_wait_ns >= busy.completion_ns
    assert independent.counters["scheduler_reordered_requests"] == 1


def test_single_submit_never_claims_batch_reordering():
    core = DramCore(config(scheduler="row_hit_first"))
    result = core.submit(AccessRequest("single", "read", 0, 32))
    assert result.counters["scheduler_reordered_requests"] == result.counters["scheduler_batches"] == 0


def test_window_age_bounds_read_write_starvation():
    core = DramCore(config(scheduler="read_write_batch", scheduler_window=3))
    requests = [AccessRequest("read-first", "read", 0, 32),
                AccessRequest("write", "write", 128, 32)]
    requests.extend(AccessRequest(f"read-{i}", "read", 32, 32) for i in range(12))
    results = core.submit_many(requests)
    assert results[1].completion_ns < results[-1].completion_ns
    assert [r.request_id for r in results] == [r.request_id for r in requests]


def test_runtime_rollback_restores_channel_credits_and_direction_on_repeated_retry():
    import copy
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    c = config(channels=2, queue_scope="channel", max_outstanding_requests=1,
               scheduler="read_write_batch")
    runtime = PhysicalRuntimeContext(c, "memory")
    core = runtime.runtime(c, "memory").core
    core.submit(AccessRequest("warm", "read", 0, 32))
    expected_channels = copy.deepcopy(core._channel_inflight)
    expected_direction = (core._scheduler_direction, core._scheduler_streak)
    saved = runtime.snapshot()
    requests = [AccessRequest("write", "write", 128, 32),
                AccessRequest("read", "read", 256, 32)]
    first = core.submit_many(requests)
    for _ in range(2):
        runtime.restore(saved)
        assert core._channel_inflight == expected_channels
        assert (core._scheduler_direction, core._scheduler_streak) == expected_direction
        assert core.submit_many(requests) == first
