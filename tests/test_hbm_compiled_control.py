"""Compiled HBM must preserve the authoritative normal-operation controller."""
from dataclasses import replace
import math

import pytest

from heterollm_sim import _compiled_hbm
from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim._hbm_control import _Controller
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, DramConfig


def configuration(**changes):
    values = dict(kind="HBM", generation="HBM3E", channels=2,
        pseudo_channels_per_channel=2, stacks=1, dies_per_stack=4,
        bank_groups_per_rank=8, banks_per_group=4, rows_per_bank=32,
        row_bytes=1024, burst_bytes=32, data_width_bits=32, data_rate_mt_s=4000.,
        bank_geometry_scope="per_stack", stack_interface_mode="independent",
        control_mode="jedec_hbm", command_clock_ns=1., activate_command_cycles=1.5,
        precharge_command_cycles=.5, column_command_cycles=1., activated_row_msb=4,
        activate_to_read_ns=4., activate_to_write_ns=3., precharge_to_activate_ns=3.,
        minimum_active_ns=7., maximum_active_ns=35100., activate_cycle_ns=10., read_to_precharge_ns=3.,
        write_to_precharge_ns=4., read_latency_ns=8., write_latency_ns=4.,
        column_same_group_ns=4., column_different_group_ns=2.,
        column_different_sid_ns=3., precharge_to_precharge_ns=2.,
        activate_same_group_ns=5., activate_different_group_ns=2.,
        write_to_read_same_group_ns=3., write_to_read_different_group_ns=2.,
        read_to_write_command_ns=7., write_to_read_ns=1., read_to_write_ns=2.)
    values.update(changes)
    if "maximum_active_ns" not in changes and values.get("refresh_interval_ns", 0.) > 0:
        values["maximum_active_ns"] = 9 * values["refresh_interval_ns"]
    return DramConfig(**values)


def assert_same(reference, numeric, request):
    reference.timeline.metrics_snapshot()
    numeric.timeline.metrics_snapshot()
    expected = _Controller(reference, request).execute()
    actual = execute_hbm_numeric(numeric, request)
    assert actual is not None
    assert actual.completion_ns == expected.completion_ns
    assert actual.logical_bytes == expected.logical_bytes
    assert actual.transfer_bytes == expected.transfer_bytes
    assert {k: v for k, v in actual.counters.items() if k != "compiled_hbm_float64"} == expected.counters
    assert numeric._hbm_control_state == reference._hbm_control_state
    assert numeric._banks == reference._banks
    for field in ("ready_ns", "busy_ns", "bytes_moved", "directions", "last_intervals", "lane_available"):
        assert getattr(numeric.timeline, field) == getattr(reference.timeline, field), field


@pytest.mark.parametrize("quantum", [2.56, 1.28, 20.48, 10.24])
def test_clock_alignment_is_idempotent_without_accepting_later_arrivals(quantum):
    from heterollm_sim._hbm_numeric import _align

    tick = round(21072650.24 / quantum)
    grid = tick * quantum
    for align in (_Controller.align, _align):
        assert align(grid, quantum) == grid
        below = math.nextafter(grid, -math.inf)
        assert align(below, quantum) == grid
        above = math.nextafter(grid, math.inf)
        following = align(above, quantum)
        assert following == (tick + 1) * quantum
        assert following > above
        assert align(following, quantum) == following


def deadline_configuration():
    return configuration(channels=1, command_clock_ns=20.48, data_rate_mt_s=195.3125,
        read_latency_ns=655.36, write_latency_ns=163.84, activate_to_read_ns=14.,
        activate_to_write_ns=14., read_to_precharge_ns=0., column_same_group_ns=81.92,
        column_different_group_ns=40.96, column_different_sid_ns=61.44,
        precharge_to_precharge_ns=40.96)


def test_repeated_column_alignment_keeps_read_before_active_deadline():
    c = deadline_configuration()
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    assert_same(reference, numeric, AccessRequest("warm", "read", 0, 32, 21037547.52))
    for core in (reference, numeric):
        ready = core.timeline.available_ns("dram:column_command:0:0")
        core.timeline.reserve("dram:column_command:0:0", ready, 21072650.24 - ready)
    assert_same(reference, numeric,
                AccessRequest("before-deadline", "read", 0, 32, 21038547.52))
    for core in (reference, numeric):
        bank = next(iter(core._hbm_control_state["banks"].values()))
        deadline = bank["last_activate"] + c.maximum_active_ns
        assert bank["precharge_ready"] <= deadline
        assert core.timeline.last_intervals["dram:column_command:0:0"][0] == 21072650.240000002
    assert_same(reference, numeric,
                AccessRequest("after-deadline", "read", 0, 32, 21072691.2))


@pytest.mark.parametrize("numeric", [False, True])
def test_clock_alignment_still_rejects_blocked_precharge_deadline(numeric):
    c = deadline_configuration()
    core = DramCore(c, capture_details=False)
    execute = execute_hbm_numeric if numeric else lambda core, request: _Controller(core, request).execute()
    execute(core, AccessRequest("warm", "read", 0, 32, 21037547.52))
    bank = next(iter(core._hbm_control_state["banks"].values()))
    deadline = bank["last_activate"] + c.maximum_active_ns
    ready = core.timeline.available_ns("dram:row_command:0:0")
    core.timeline.reserve("dram:row_command:0:0", ready, deadline + c.command_clock_ns - ready)
    with pytest.raises(ValueError, match="tRASmax deadline cannot be met by a legal PRE"):
        execute(core, AccessRequest("blocked", "read", 0, 32, deadline))


@pytest.mark.parametrize("generation,clock,refresh", [
    ("HBM2E", 1., False), ("HBM3E", 1., False),
    ("HBM3E", .7, True), ("HBM4", .625, True),
])
def test_compiled_matches_mixed_commands_and_refresh(generation, clock, refresh):
    hbm2 = generation == "HBM2E"
    c = configuration(generation=generation, command_clock_ns=clock,
        activated_row_msb=None if hbm2 else 4,
        activate_command_cycles=2. if hbm2 else 1.5,
        precharge_command_cycles=1. if hbm2 else .5,
        data_width_bits=64 if hbm2 else 32, data_rate_mt_s=(2000. if hbm2 else 4000.) / clock,
        read_latency_ns=(32. if generation == "HBM4" else 8.) * clock,
        write_latency_ns=4. * clock, refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    row_span = c.lane_count * c.bank_count * c.row_bytes
    commands = [
        ("read", 3, 30007, 0.), ("write", 65, 9000, 0.),
        ("read", row_span + 5, 45000, 5.),
        ("write", 16 * row_span + 1, 8100, 8.),
        ("read", row_span, 33001, 20000.),
    ]
    for index, (op, address, count, time) in enumerate(commands):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, time))


def test_compiled_preserves_interleaving_stacks_and_direction_budget():
    c = configuration(stacks=2, interleave_bytes=128,
                      write_bandwidth_gb_s=64., read_bandwidth_gb_s=128.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for core in (reference, numeric):
        core.timeline.reserve("dram:row_command:0:0", 0., 100.)
        core.timeline.reserve("dram:column_command:1:1", 0., 70.)
        core.timeline.reserve("dram:data:3", 0., 200., direction="write")
    for index, op in enumerate(("read", "write", "read")):
        assert_same(reference, numeric, AccessRequest(str(index), op, index * 4111, 270011, index * 3.))


def test_compiled_reuses_saved_bank_mappings(monkeypatch):
    c = configuration()
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    mapped_addresses = []
    original_mapping = _compiled_hbm.map_dram_address

    def observe_mapping(config, address):
        mapped_addresses.append(address)
        return original_mapping(config, address)

    monkeypatch.setattr(_compiled_hbm, "map_dram_address", observe_mapping)
    row_span = c.lane_count * c.bank_count * c.row_bytes
    for index, (address, size) in enumerate(((0, 32), (row_span, 32), (32, 32), (0, 4096), (0, 4096))):
        assert_same(reference, numeric, AccessRequest(str(index), "read", address, size))
        assert len(mapped_addresses) == len(numeric._hbm_control_state["banks"])


def test_numeric_accepts_nonzero_faw_and_preserves_controller_state():
    c = configuration(four_activate_window_ns=20.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    assert_same(reference, numeric, AccessRequest("a", "read", 0, 4096))


def test_flat_bank_group_keeps_separate_sid_histories():
    c = configuration(bank_groups_per_rank=1, banks_per_group=64,
                      column_different_sid_ns=9., refresh_policy="all_bank",
                      refresh_interval_ns=900., refresh_cycle_ns=45.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for index, (op, address, count) in enumerate((
            ("read", 0, 163841), ("write", 513, 81921),
            ("read", 64 * 4 * 1024, 131071), ("read", 2048, 32000))):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, 2. * index))


def test_large_idle_refresh_folding_keeps_exact_state_and_accounting():
    c = configuration(refresh_policy="all_bank", refresh_interval_ns=100., refresh_cycle_ns=20.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    assert_same(reference, numeric, AccessRequest("warm", "read", 0, 4096))
    assert_same(reference, numeric, AccessRequest("after-idle", "write", 0, 8192, 1.e9))


def test_unrelated_kernel_aliases_do_not_disable_compiled_entry():
    c = configuration()
    core = DramCore(c, capture_details=False)
    core.timeline.resource_aliases = {"unrelated:port": "owner:wire", "dram:data:0": "dram:data:0"}
    result = core.submit(AccessRequest("aliased-kernel", "read", 0, 4096))
    assert result.counters["compiled_hbm_float64"] is True
    assert result.counters["read_commands"] == 128
    blocked = DramCore(c, capture_details=False)
    blocked.timeline.resource_aliases = {"dram:data:0": "dram:data:1"}
    assert execute_hbm_numeric(blocked, AccessRequest("shared-wire", "read", 0, 4096)) is None
    assert not blocked._hbm_control_state


@pytest.mark.parametrize("refresh", [False, True])
def test_maximum_active_deadlines_match_before_reads_writes_and_idle(refresh):
    c = configuration(maximum_active_ns=50., refresh_policy="all_bank" if refresh else "disabled",
                      refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for index, (op, address, count, arrival) in enumerate((
            ("read", 0, 256, 0.), ("read", 0, 32, 49.),
            ("write", 128, 96, 95.), ("read", 32, 128, 20000.))):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, arrival))


@pytest.mark.parametrize("generation", ["HBM3E", "HBM4"])
def test_real_builder_maximum_active_and_idle_match(generation):
    from heterollm_sim.hbm_geometry import make_hbm_geometry_config
    c = DramConfig(**make_hbm_geometry_config(generation=generation, capacity_bytes=36 * 2 ** 30,
        read_bandwidth_gbps=9600., write_bandwidth_gbps=9600., die_density_gbits=24, dies_per_stack=12))
    assert c.maximum_active_ns == 35100.
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    assert_same(reference, numeric, AccessRequest("normal", "read", 0, 32768))
    assert_same(reference, numeric, AccessRequest("idle", "write", 0, 2048, 1.e6))


def test_maximum_active_delays_activation_until_reserved_dq_slot():
    c = configuration(maximum_active_ns=50.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for core in (reference, numeric):
        core.timeline.reserve("dram:data:0", 0., 1000.)
    assert_same(reference, numeric, AccessRequest("late-dq", "read", 0, 32))


def test_impossible_maximum_active_window_is_rejected_before_either_backend_runs():
    with pytest.raises(ValueError, match="READ recovery cannot fit before tRASmax"):
        configuration(minimum_active_ns=1., maximum_active_ns=5.)


def test_refresh_cannot_precharge_an_open_row_after_its_deadline():
    c = configuration(maximum_active_ns=2000., refresh_policy="all_bank",
                      refresh_interval_ns=1000., refresh_cycle_ns=40.)
    for numeric in (False, True):
        core = DramCore(c, capture_details=False)
        warm = AccessRequest("warm", "read", 0, 32)
        execute_hbm_numeric(core, warm) if numeric else _Controller(core, warm).execute()
        core.timeline.reserve("dram:row_command:0:0", 0., 3000.)
        request = AccessRequest("blocked", "read", 0, 32, 1000.)
        with pytest.raises(ValueError, match="tRASmax.*refresh PRE"):
            execute_hbm_numeric(core, request) if numeric else _Controller(core, request).execute()


def assert_core_state_equal(actual, expected):
    assert {key: value for key, value in vars(actual).items() if key != "timeline"} == {
        key: value for key, value in vars(expected).items() if key != "timeline"}
    assert vars(actual.timeline) == vars(expected.timeline)


def assert_batch_matches_scalar(actual, expected):
    # The scheduler now reports the submit_many boundary; service and every
    # per-request observation must still equal the ordered scalar execution.
    assert [row.counters["scheduler_batches"] for row in actual] == [1] + [0] * (len(actual) - 1)
    assert tuple(replace(row, counters={**row.counters, "scheduler_batches": 0})
                 for row in actual) == expected


@pytest.mark.parametrize("refresh", [False, True])
def test_four_tib_profile_uses_compiled_hbm_at_last_bank(refresh):
    from heterollm_sim.memory_mapping import map_dram_address
    from heterollm_sim.physical_contract import reconfigure_physical_config
    from heterollm_sim.physical_presets import resolve_physical_config

    c = resolve_physical_config("hbm3e_reference", overrides={
        "capacity_bytes": 4 * 1024 ** 4, "read_bandwidth_gb_s": 1600.,
        "write_bandwidth_gb_s": 1600., "interface_bandwidth_gb_s": 1600.})
    if refresh:
        c = reconfigure_physical_config(c, overrides={"refresh_policy": "all_bank",
            "refresh_interval_ns": 3900., "refresh_cycle_ns": 400.})
    assert c.lane_count * c.bank_count == 131_072
    assert c.stacks == 64 and c.stack_interface_mode == "independent"
    end = c.effective_capacity_bytes
    last = map_dram_address(c, end - 1)
    assert (last.stack, last.lane, last.channel, last.pseudo_channel,
            last.bank_group, last.bank, last.row, last.column) == (
        63, 2047, 15, 1, 15, 3, 32767, 31)
    last_bank = (c.lane_count * c.bank_count - 1) * c.burst_bytes
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for index, (op, address, size, arrival) in enumerate((
            ("read", 0, 1, 0.), ("read", last_bank + 3, 29, 0.),
            ("write", end - 1, 1, 1.), ("read", end - 65, 65, 40000.))):
        request = AccessRequest(str(index), op, address, size, arrival)
        actual = numeric._submit_validated(request,
            execute=lambda accepted: execute_hbm_numeric(numeric, accepted))
        assert actual.counters["compiled_hbm_float64"] is True
        expected = reference._submit_validated(request,
            execute=lambda accepted: _Controller(reference, accepted).execute())
        assert replace(actual, counters={key: value for key, value in actual.counters.items()
                                        if key != "compiled_hbm_float64"}) == expected
        assert_core_state_equal(numeric, reference)
    if refresh:
        assert actual.counters["refresh_commands"] > 0


def test_numeric_declines_above_hbm_bank_limit_without_mutation():
    c = configuration(stacks=65, channels=16, bank_groups_per_rank=16)
    assert c.lane_count * c.bank_count > _compiled_hbm.MAX_HBM_NUMERIC_BANKS
    core = DramCore(c, capture_details=False)
    assert execute_hbm_numeric(core, AccessRequest("too-many-banks", "read", 0, 32)) is None
    assert_core_state_equal(core, DramCore(c, capture_details=False))


@pytest.mark.parametrize("queue_depth", [1, 3])
@pytest.mark.parametrize("refresh", [False, True])
def test_compiled_batch_preserves_each_request_and_following_submission(queue_depth, refresh):
    c = configuration(max_outstanding_requests=queue_depth, maximum_active_ns=2000.,
        refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    sequential, batch = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for core in (sequential, batch):
        core.timeline.reserve("dram:data:3", 0., 200., direction="write")
        core.submit(AccessRequest("warm", "read", 0, 4096))
    row_span = c.lane_count * c.bank_count * c.row_bytes
    requests = tuple(AccessRequest(str(index), op, address, count, arrival)
        for index, (op, address, count, arrival) in enumerate((
            ("read", 3, 61, 0.), ("write", 65, 65, 0.),
            ("read", 2055, 257, 1.), ("write", row_span + 31, 33, 5.),
            ("read", 32, 96, 5000.), ("write", 2 * row_span + 1, 64, 5000.),
            ("read", 0, 129, 2.))))
    expected = tuple(sequential.submit(request) for request in requests)
    actual = batch.submit_many(requests)
    assert_batch_matches_scalar(actual, expected)
    assert any(result.counters["queue_wait_ns"] > 0 for result in actual)
    assert all(result.counters["compiled_hbm_float64"] for result in actual)
    assert_core_state_equal(batch, sequential)
    following = AccessRequest("following", "write", row_span + 7, 1025, 6000.)
    assert batch.submit(following) == sequential.submit(following)
    assert_core_state_equal(batch, sequential)


@pytest.mark.parametrize("details,faw", [(True, 0.), (False, 20.)])
def test_compiled_batch_preserves_detail_and_nonzero_faw_modes(details, faw):
    c = configuration(four_activate_window_ns=faw)
    sequential, batch = DramCore(c, capture_details=details), DramCore(c, capture_details=details)
    requests = (AccessRequest("read", "read", 3, 129), AccessRequest("write", "write", 32, 97))
    assert_batch_matches_scalar(batch.submit_many(requests), tuple(sequential.submit(request) for request in requests))
    assert_core_state_equal(batch, sequential)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("faw", [0., 31.])
def test_compiled_batch_keeps_successful_prefix_when_numeric_stops(failure, faw, monkeypatch):
    c = configuration(max_outstanding_requests=1, four_activate_window_ns=faw)
    sequential, batch = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    requests = tuple(AccessRequest(str(i), "write" if i % 2 else "read", i * 67, 97) for i in range(4))
    original = _compiled_hbm._HBMNumericState.execute

    def stop_after_numeric_mutation(packed, request):
        result = original(packed, request)
        if request.request_id == "2":
            if failure:
                raise ValueError("injected numeric failure")
            return None
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm._HBMNumericState, "execute", stop_after_numeric_mutation)
        if failure:
            with pytest.raises(ValueError, match="injected numeric failure"):
                tuple(sequential.submit(request) for request in requests)
            with pytest.raises(ValueError, match="injected numeric failure"):
                batch.submit_many(requests)
        else:
            expected = tuple(sequential.submit(request) for request in requests)
            assert_batch_matches_scalar(batch.submit_many(requests), expected)
        assert_core_state_equal(batch, sequential)
    if failure:
        assert_batch_matches_scalar(batch.submit_many(requests[2:]), tuple(sequential.submit(request) for request in requests[2:]))
        assert_core_state_equal(batch, sequential)
