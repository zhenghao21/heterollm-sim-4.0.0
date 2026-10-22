"""Normal-operation HBM command traces, independently of vendor timing bins."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import rotated_bank_step
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig


def config(**overrides):
    values = dict(kind="HBM", generation="HBM3E", channels=2,
        pseudo_channels_per_channel=2, stacks=1, dies_per_stack=4,
        bank_groups_per_rank=4, banks_per_group=4, rows_per_bank=8,
        row_bytes=1024, burst_bytes=32, data_width_bits=32, data_rate_mt_s=4000.,
        bank_geometry_scope="per_stack", stack_interface_mode="independent",
        control_mode="jedec_hbm", command_clock_ns=1., activate_command_cycles=1.5,
        precharge_command_cycles=.5, column_command_cycles=1., activated_row_msb=2,
        activate_to_read_ns=4., activate_to_write_ns=3., precharge_to_activate_ns=3.,
        minimum_active_ns=7., maximum_active_ns=35100., activate_cycle_ns=10., read_to_precharge_ns=3.,
        write_to_precharge_ns=4., read_latency_ns=8., write_latency_ns=4.,
        column_same_group_ns=4., column_different_group_ns=2.,
        column_different_sid_ns=3., precharge_to_precharge_ns=2.,
        open_ns=100., close_ns=100., burst_interval_ns=100.,
        write_recovery_ns=100.)
    values.update(overrides)
    if "maximum_active_ns" not in overrides and values.get("refresh_interval_ns", 0.) > 0:
        values["maximum_active_ns"] = 9 * values["refresh_interval_ns"]
    return DramConfig(**values)


def address(c, *, stack=0, channel=0, pc=0, bank=0, row=0, column=0):
    lane = (stack * c.channels + channel) * 2 + pc
    step = rotated_bank_step(bank, c.bank_count, c.banks_per_group)
    return (((row * (c.row_bytes // c.burst_bytes) + column) * c.bank_count + step)
            * c.lane_count + lane) * c.burst_bytes


def request(c, op="read", *, time=0., **coordinates):
    return AccessRequest(str(coordinates), op, address(c, **coordinates), 32, time)


def stage(result, name):
    return next(item for item in result.stages if item.name == name)


def test_act_occupancy_and_second_edge_array_anchor_are_separate():
    core = DramCore(config())
    result = core.submit(request(core.config))
    act = stage(result, "ACTIVATE")
    assert (act.start_ns, act.end_ns) == (0., 1.5)
    assert (stage(result, "READ_COMMAND").start_ns, result.completion_ns) == (5., 15.)
    assert result.counters["activate_commands"] == 1
    assert result.counters["row_command_bus_busy_ns"] == 1.5
    assert result.counters["column_command_bus_busy_ns"] == 1.
    assert not any(key.startswith("dram:bank:") for key in result.counters["resource_busy_ns"])
    assert "refresh_commands" not in result.counters


@pytest.mark.parametrize("ras,rp,rc", [(7., 3., 30.), (20., 10., 5.)])
def test_same_bank_reactivation_enforces_independent_rc_and_ras_rp_constraints(ras, rp, rc):
    c = config(minimum_active_ns=ras, precharge_to_activate_ns=rp, activate_cycle_ns=rc)
    core = DramCore(c)
    first = core.submit(request(c))
    changed_row = core.submit(request(c, row=1))
    first_anchor = stage(first, "ACTIVATE").start_ns + c.command_clock_ns
    precharge = stage(changed_row, "PRECHARGE").start_ns
    next_act = stage(changed_row, "ACTIVATE").start_ns
    assert precharge >= first_anchor + ras
    assert next_act >= precharge + rp
    assert next_act + c.command_clock_ns >= first_anchor + rc
    assert next_act + c.command_clock_ns - first_anchor >= max(rc, ras + rp)


def test_pseudo_channels_share_ca_but_group_timings_are_independent():
    c = config()
    core = DramCore(c)
    first = core.submit(request(c, pc=0))
    other = core.submit(request(c, pc=1))
    assert stage(first, "ACTIVATE").resource_id == stage(other, "ACTIVATE").resource_id
    assert stage(other, "ACTIVATE").start_ns == 2.
    assert stage(other, "READ_COMMAND").start_ns == 7.  # no PC0 tCCDL=4 penalty
    assert stage(other, "ACTIVATE").start_ns < stage(first, "READ_COMMAND").start_ns
    a = core.submit(request(c, pc=0, time=100))
    b = core.submit(request(c, pc=1, time=100))
    assert stage(a, "READ_COMMAND").start_ns == 100.
    assert stage(b, "READ_COMMAND").start_ns == 101.  # the shared CA wire serializes
    assert stage(a, "BURST_TRANSFER").resource_id != stage(b, "BURST_TRANSFER").resource_id


def test_physical_channels_and_stacks_have_separate_command_buses():
    c = config(stacks=2)
    core = DramCore(c)
    results = [core.submit(request(c, stack=stack, channel=channel))
               for stack, channel in ((0, 0), (0, 1), (1, 0))]
    assert {stage(item, "ACTIVATE").start_ns for item in results} == {0.}
    assert len({stage(item, "ACTIVATE").resource_id for item in results}) == 3


@pytest.mark.parametrize("generation,rows,bit,latency", [
    ("HBM3E", 24576, 14, 8.), ("HBM4", 12288, 13, 17.),
])
def test_effective_row_msb_selects_the_same_activated_page(generation, rows, bit, latency):
    c = config(generation=generation, rows_per_bank=rows, activated_row_msb=bit,
               read_latency_ns=latency)
    core = DramCore(c)
    core.submit(request(c, row=0))
    other_half = core.submit(request(c, row=1 << bit))
    assert other_half.counters["row_hits"] == 1
    assert other_half.counters["activate_commands"] == 0
    different = core.submit(request(c, row=1))
    assert different.counters["row_conflicts"] == 1
    assert different.counters["precharge_commands"] == 1


def test_read_and_write_recovery_gate_precharge_without_charging_write_completion():
    c = config()
    core = DramCore(c)
    first = core.submit(request(c))
    conflict = core.submit(request(c, row=1))
    assert stage(first, "READ_COMMAND").start_ns == 5.
    assert stage(conflict, "PRECHARGE").start_ns == 8.  # max(tRAS anchor+7, RD+3)
    assert stage(conflict, "ACTIVATE").start_ns == 11.  # PRE+tRP; rising edge
    writer = DramCore(c)
    written = writer.submit(request(c, "write"))
    assert written.completion_ns == 10.  # DQ end; old write_recovery_ns=100 is unused
    after = writer.submit(request(c, row=1))
    assert stage(after, "PRECHARGE").start_ns == 14.  # write DQ end+tWR


def test_activate_cycle_and_half_cycle_precharge_constraints():
    c = config(activate_cycle_ns=30., minimum_active_ns=7.2)
    core = DramCore(c)
    core.submit(request(c))
    result = core.submit(request(c, row=1))
    assert stage(result, "PRECHARGE").start_ns == 8.5
    assert stage(result, "ACTIVATE").start_ns == 30.


def test_activate_group_gaps_and_four_activate_window_are_per_pc():
    c = config(activate_same_group_ns=10., activate_different_group_ns=3.)
    core = DramCore(c)
    first = core.submit(request(c, bank=0))
    same = core.submit(request(c, bank=1))
    different = core.submit(request(c, bank=4))
    assert [stage(item, "ACTIVATE").start_ns for item in (first, same, different)] == [0., 10., 13.]
    window = DramCore(config(four_activate_window_ns=20.))
    results = [window.submit(request(window.config, bank=index)) for index in range(5)]
    assert [stage(item, "ACTIVATE").start_ns for item in results] == [0., 2., 4., 6., 20.]


def test_precharge_same_bank_cannot_use_acts_reserved_last_half_cycle():
    c = config(activate_to_read_ns=0., activate_to_write_ns=0., minimum_active_ns=0.,
               activate_cycle_ns=0., read_to_precharge_ns=0., precharge_to_activate_ns=0.)
    core = DramCore(c)
    core.submit(request(c))
    conflict = core.submit(request(c, row=1))
    assert stage(conflict, "PRECHARGE").start_ns == 2.


def test_sid_read_spacing_is_distinct_from_write_spacing():
    c = config(bank_groups_per_rank=8, column_different_sid_ns=7.)
    reader = DramCore(c)
    a = reader.submit(request(c, bank=0))
    b = reader.submit(request(c, bank=16))
    assert stage(b, "READ_COMMAND").start_ns - stage(a, "READ_COMMAND").start_ns == 7.
    writer = DramCore(c)
    a = writer.submit(request(c, "write", bank=0))
    b = writer.submit(request(c, "write", bank=16))
    assert stage(b, "WRITE_COMMAND").start_ns - stage(a, "WRITE_COMMAND").start_ns == 2.


def test_read_write_command_turnaround_and_write_end_to_read_gates():
    c = config(read_to_write_command_ns=12., write_to_read_same_group_ns=6.)
    core = DramCore(c)
    read = core.submit(request(c))
    write = core.submit(request(c, "write"))
    assert stage(write, "WRITE_COMMAND").start_ns >= stage(read, "READ_COMMAND").start_ns + 12.
    following = core.submit(request(c))
    assert stage(following, "READ_COMMAND").start_ns >= stage(write, "BURST_TRANSFER").end_ns + 6.


@pytest.mark.parametrize("operation,spacing", [("read", 8.), ("write", 16.)])
def test_directional_cap_paces_commands_without_stretching_dq_bursts(operation, spacing):
    c = config(interface_bandwidth_gb_s=64., read_bandwidth_gb_s=16., write_bandwidth_gb_s=8.)
    core = DramCore(c)
    results = [core.submit(request(c, operation, column=index)) for index in range(3)]
    transfers = [stage(result, "BURST_TRANSFER") for result in results]
    assert [item.end_ns - item.start_ns for item in transfers] == [2.] * 3
    assert [transfers[i + 1].start_ns - transfers[i].start_ns for i in range(2)] == [spacing] * 2
    assert sum(result.counters["resource_busy_ns"]["dram:data:0"] for result in results) == 6.


def test_periodic_refresh_targets_each_pc_all_sids_once_and_reopens_after_rfc():
    c = config(bank_groups_per_rank=8, refresh_policy="all_bank",
               refresh_interval_ns=100., refresh_cycle_ns=20.)
    core = DramCore(c)
    core.submit(request(c, bank=0))
    core.submit(request(c, bank=16))
    result = core.submit(request(c, bank=0, time=105.))
    assert result.counters["refresh_commands"] == 2  # two PCs, not PC times SID
    assert result.counters["precharge_commands"] == 1  # one PREab for PC0
    refresh = [item for item in result.stages if item.name == "REFRESH_ALL"]
    assert len(refresh) == 2
    assert all(item.start_ns % c.command_clock_ns == 0 for item in refresh)
    assert stage(result, "ACTIVATE").start_ns >= refresh[0].start_ns + 20.
    assert result.counters["row_misses"] == 1
    assert result.counters["activate_commands"] == 1


def test_long_idle_refresh_aggregation_matches_individual_epochs():
    c = config(refresh_policy="all_bank", refresh_interval_ns=100., refresh_cycle_ns=20.)
    reference, summary = DramCore(c), DramCore(c, capture_details=False)
    for core in (reference, summary):
        core.submit(request(c))
    expected = reference.submit(request(c, time=10000.))
    actual = summary.submit(request(c, time=10000.))
    assert actual.completion_ns == expected.completion_ns
    assert actual.counters["refresh_commands"] == expected.counters["refresh_commands"] == 200
    assert actual.counters["resource_busy_ns"] == expected.counters["resource_busy_ns"]
    assert summary._hbm_control_state == reference._hbm_control_state
    assert summary._banks == reference._banks
    assert summary.timeline.last_intervals == reference.timeline.last_intervals
    assert summary.timeline.ready_ns == reference.timeline.ready_ns


def test_long_idle_does_not_expand_unbounded_detail_stages():
    c = config(refresh_policy="all_bank", refresh_interval_ns=100., refresh_cycle_ns=20.,
               max_expanded_segments=2)
    core = DramCore(c)
    core.submit(request(c))
    result = core.submit(request(c, time=1e9))
    assert result.counters["refresh_commands"] == 20_000_000
    assert result.counters["details_truncated"] is True
    assert result.stages == ()


def test_hbm2_command_encoding_uses_two_clock_activate():
    c = config(generation="HBM2E", data_width_bits=64, data_rate_mt_s=2000.,
               activate_command_cycles=2., precharge_command_cycles=1., activated_row_msb=None)
    result = DramCore(c).submit(request(c))
    assert stage(result, "ACTIVATE").end_ns == 2.
    assert stage(result, "READ_COMMAND").start_ns == 5.  # second rising edge + tRCD
    assert stage(result, "BURST_TRANSFER").end_ns - stage(result, "BURST_TRANSFER").start_ns == 2.


@pytest.mark.parametrize("resource", ["dram:row_command:0:0", "dram:column_command:0:0", "dram:data:0"])
def test_physical_ca_and_dq_resources_reject_multiple_execution_slots(resource):
    c = config()
    core = DramCore(c, ResourceTimeline(lane_available={resource: [0., 0.]}))
    with pytest.raises(ValueError, match="one execution slot"):
        core.submit(request(c))
    assert core.timeline.busy_ns == {}
    assert core._banks == {}


def test_summary_and_batch_use_new_control_semantics_without_legacy_acceleration():
    c = config()
    requests = (request(c), request(c, "write", row=1), request(c, pc=1))
    reference = DramCore(c)
    expected = reference.submit_many(requests)
    summary = DramCore(c, capture_details=False)
    assert not summary._can_accelerate()
    with patch("heterollm_sim._compiled_dram.execute_compiled", side_effect=AssertionError("legacy path")), \
         patch("heterollm_sim._compiled_dram.execute_compiled_many", side_effect=AssertionError("legacy batch")):
        actual = summary.submit_many(requests)
    assert summary._hbm_control_state == reference._hbm_control_state
    assert summary._banks == reference._banks
    for a, b in zip(actual, expected):
        assert a.completion_ns == b.completion_ns
        assert a.counters["resource_busy_ns"] == b.counters["resource_busy_ns"]
        assert a.stages == a.mapping == ()


def test_maximum_active_closes_a_row_during_long_idle_without_refresh():
    c = config(maximum_active_ns=20.25)
    core = DramCore(c)
    core.submit(request(c))  # ACT anchor 1; tRASmax deadline 21.25.
    result = core.submit(request(c, time=100.))
    assert stage(result, "PRECHARGE").start_ns == 21.  # last legal half-CK edge
    assert stage(result, "ACTIVATE").start_ns == 100.
    assert result.counters["row_hits"] == 0
    assert result.counters["row_misses"] == 1
    assert result.counters["precharge_commands"] == 1
    assert "refresh_commands" not in result.counters


def test_write_recovery_crossing_maximum_active_precharges_before_issuing_write():
    c = config(maximum_active_ns=20.)
    core = DramCore(c)
    core.submit(request(c))
    result = core.submit(request(c, "write", time=15.))
    assert stage(result, "PRECHARGE").start_ns == 21.
    assert stage(result, "ACTIVATE").start_ns == 24.
    assert stage(result, "WRITE_COMMAND").start_ns == 28.
    assert stage(result, "BURST_TRANSFER").end_ns + c.write_to_precharge_ns <= 25. + c.maximum_active_ns


def test_maximum_active_closes_banks_across_both_pcs_on_the_shared_row_bus():
    c = config(maximum_active_ns=20.)
    core = DramCore(c)
    initial = [core.submit(request(c, pc=pc, bank=bank)) for pc, bank in ((0, 0), (1, 0), (0, 1))]
    deadlines = [stage(result, "ACTIVATE").start_ns + c.command_clock_ns + c.maximum_active_ns
                 for result in initial]
    result = core.submit(request(c, time=100.))
    precharges = [item for item in result.stages if item.name == "PRECHARGE"]
    assert [item.start_ns for item in precharges] == deadlines == [21., 23., 25.]
    assert len({item.resource_id for item in precharges}) == 1
    assert result.counters["precharge_commands"] == 3


def test_maximum_active_plan_leaves_room_for_other_precharge_deadlines():
    c = config(maximum_active_ns=20., precharge_to_precharge_ns=10.)
    core = DramCore(c)
    core.submit(request(c, bank=0))
    core.submit(request(c, bank=1))
    result = core.submit(request(c, time=100.))
    precharges = [item.start_ns for item in result.stages if item.name == "PRECHARGE"]
    assert precharges == [13., 23.]  # PC0 tPPD=10 requires the first PRE before its deadline 21.


def test_maximum_active_delays_new_activate_until_a_paced_dq_slot_is_near():
    c = config(maximum_active_ns=20., read_bandwidth_gb_s=1.)
    core = DramCore(c)
    core.submit(request(c))
    result = core.submit(request(c))
    assert stage(result, "PRECHARGE").start_ns == 21.
    assert stage(result, "ACTIVATE").start_ns == 115.
    assert stage(result, "READ_COMMAND").start_ns == 133.
    assert stage(result, "READ_COMMAND").start_ns + c.read_to_precharge_ns == 116. + c.maximum_active_ns


def test_maximum_active_reports_an_unreachable_recovery_or_precharge_deadline():
    with pytest.raises(ValueError, match="WRITE recovery cannot fit before tRASmax"):
        config(maximum_active_ns=8.)
    c = config(maximum_active_ns=20.)
    core = DramCore(c)
    core.submit(request(c))
    core.timeline.reserve("dram:row_command:0:0", 50., 1.)
    with pytest.raises(ValueError, match="tRASmax deadline cannot be met"):
        core.submit(request(c, time=100.))


def test_periodic_refresh_precedes_a_later_maximum_active_deadline():
    c = config(maximum_active_ns=900., refresh_policy="all_bank",
               refresh_interval_ns=100., refresh_cycle_ns=20.)
    core = DramCore(c)
    core.submit(request(c))
    result = core.submit(request(c, time=1000.))
    assert stage(result, "PRECHARGE_ALL").start_ns == 100.
    assert not any(item.name == "PRECHARGE" for item in result.stages)


def test_refresh_precharge_cannot_bypass_the_maximum_active_deadline():
    c = config(maximum_active_ns=20., refresh_policy="all_bank",
               refresh_interval_ns=20., refresh_cycle_ns=5.)
    core = DramCore(c)
    core.submit(request(c))
    core.timeline.reserve("dram:row_command:0:0", 50., 1.)
    with pytest.raises(ValueError, match="tRASmax deadline cannot be met by the refresh PRE"):
        core.submit(request(c, time=100.))
