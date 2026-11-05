"""JEDEC dependent values follow their inputs at every public config boundary."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, DramConfig
from heterollm_sim.physical_contract import (
    parse_execution_physical_config, reconfigure_physical_config,
)
from heterollm_sim.physical_presets import resolve_physical_config


PATHS = ("reconfigure", "dataclass", "replace", "from_mapping", "execution")


def reference(generation="HBM4", density=24, height=8, clock=1.):
    lanes = 64 if generation == "HBM4" else 16 if generation in {"HBM2", "HBM2E"} else 32
    interface = lanes * 16. / clock
    return DramConfig(**make_hbm_geometry_config(
        generation=generation, die_density_gbits=density, dies_per_stack=height,
        capacity_bytes=8 * 2**30, interface_bandwidth_gbps=interface * 8,
        read_bandwidth_gbps=interface * 8, write_bandwidth_gbps=interface * 4))


def update(config, values, path):
    if path == "reconfigure":
        return reconfigure_physical_config(config, overrides=values)
    if path == "replace":
        return replace(config, **values)
    payload = asdict(config)
    payload.update(deepcopy(values))
    if path == "dataclass":
        return DramConfig(**payload)
    if path == "from_mapping":
        return DramConfig.from_mapping(payload)
    return parse_execution_physical_config(payload)


def status(config, field):
    return config.metadata["parameter_provenance"][field]["status"]


@pytest.mark.parametrize("path", PATHS)
def test_height_only_rebuilds_organization_and_refresh_table(path):
    old = reference()
    new = update(old, {"dies_per_stack": 16}, path)
    direct = reference(height=16)
    assert old.refresh_cycle_ns == 410.
    assert new.refresh_cycle_ns == direct.refresh_cycle_ns == 490.
    assert new.bank_count == direct.bank_count == 64
    assert new.bank_groups_per_rank == direct.bank_groups_per_rank == 8
    assert new.rows_per_bank == direct.rows_per_bank == 12288
    assert new.metadata["die_density_gbits"] == 24
    assert new.command_clock_ns == old.command_clock_ns == 1.
    assert new.computed_capacity_bytes == direct.computed_capacity_bytes
    assert new.capacity_bytes == old.capacity_bytes
    assert status(new, "refresh_cycle_ns") == "standard_reference"
    assert old.dies_per_stack == 8  # normalization must not mutate its source


@pytest.mark.parametrize("path", PATHS)
def test_physical_frequency_recomputes_cycle_based_timings_and_mr_latencies(path):
    old = reference("HBM3", 16)
    new = update(old, {"interface_bandwidth_gb_s": old.interface_bandwidth_gb_s / 2,
                       "read_bandwidth_gb_s": old.read_bandwidth_gb_s / 2,
                       "write_bandwidth_gb_s": old.write_bandwidth_gb_s / 2}, path)
    direct = reference("HBM3", 16, clock=2.)
    for name in ("command_clock_ns", "data_rate_mt_s", "column_different_group_ns",
                 "column_same_group_ns", "precharge_to_precharge_ns", "column_different_sid_ns",
                 "read_latency_ns", "write_latency_ns", "minimum_active_ns",
                 "precharge_to_activate_ns", "activate_cycle_ns"):
        assert getattr(new, name) == getattr(direct, name), name
    assert (old.command_clock_ns, new.command_clock_ns) == (1., 2.)
    assert (old.column_different_group_ns, new.column_different_group_ns) == (2., 4.)
    assert new.precharge_to_precharge_ns == 4.
    assert new.read_latency_ns / new.command_clock_ns == 32
    assert new.write_latency_ns / new.command_clock_ns == 8
    assert (new.minimum_active_ns, new.precharge_to_activate_ns, new.activate_cycle_ns) == (34., 16., 50.)


def test_reconfigure_accepts_clock_as_an_explicit_physical_operating_point():
    old = reference("HBM3", 16)
    new = reconfigure_physical_config(old, overrides={"command_clock_ns": 2.})
    assert new.command_clock_ns == 2.
    assert new.physical_interface_bandwidth_gb_s == old.physical_interface_bandwidth_gb_s / 2
    assert new.data_rate_mt_s == old.data_rate_mt_s / 2
    assert new.column_different_group_ns == new.precharge_to_precharge_ns == 4.
    assert new.read_latency_ns == 64.
    assert new.write_latency_ns == 16.


@pytest.mark.parametrize("path", PATHS)
def test_directional_caps_leave_the_physical_clock_and_timing_unchanged(path):
    old = reference("HBM3", 16)
    new = update(old, {"read_bandwidth_gb_s": 100., "write_bandwidth_gb_s": 50.}, path)
    assert (new.read_bandwidth_gb_s, new.write_bandwidth_gb_s) == (100., 50.)
    assert new.physical_interface_bandwidth_gb_s == old.physical_interface_bandwidth_gb_s
    for field in ("command_clock_ns", "data_rate_mt_s", "column_different_group_ns",
                  "column_same_group_ns", "precharge_to_precharge_ns", "read_latency_ns",
                  "write_latency_ns", "refresh_cycle_ns"):
        assert getattr(new, field) == getattr(old, field), field


@pytest.mark.parametrize("path", PATHS)
def test_custom_unknown_density_cannot_keep_previous_standard_refresh(path):
    old = reference()
    new = update(old, {"rows_per_bank": 12000}, path)
    assert new.rows_per_bank == 12000
    assert new.metadata.get("die_density_gbits") is None
    assert new.refresh_policy == "disabled"
    assert new.refresh_cycle_ns == 0.
    assert status(new, "refresh_cycle_ns") == "unknown"
    assert "refresh_cycle_ns" in new.metadata["unknown_parameters"]


@pytest.mark.parametrize("path", PATHS)
def test_hbm3_tbd_density_never_inherits_a_known_density_refresh_value(path):
    old = reference("HBM3", 16)
    # The legal 24 Gbit row-address count uniquely identifies this imported
    # organization; the corresponding JESD238 tRFC entries are explicitly TBD.
    new = update(old, {"rows_per_bank": 24576}, path)
    assert new.metadata["die_density_gbits"] == 24
    assert new.refresh_cycle_ns == 0.
    assert new.refresh_policy == "disabled"
    assert status(new, "refresh_cycle_ns") == "unknown"


def test_explicit_experimental_refresh_survives_reconfiguration_as_an_assumption():
    old = reference()
    experimental = reconfigure_physical_config(old, overrides={"refresh_cycle_ns": 777.})
    assert status(experimental, "refresh_cycle_ns") == "explicit_assumption"
    for values in ({"dies_per_stack": 16}, {"rows_per_bank": 12000}):
        changed = reconfigure_physical_config(experimental, overrides=values)
        assert changed.refresh_cycle_ns == 777.
        # tRFCpb is known only for a standard organization.
        assert changed.refresh_policy == ("per_bank" if "dies_per_stack" in values else "all_bank")
        assert status(changed, "refresh_cycle_ns") == "explicit_assumption"


def test_unchanged_derived_values_recompute_but_manually_changed_values_remain():
    old = reference()
    payload = asdict(old)
    payload.update(dies_per_stack=16)
    derived = parse_execution_physical_config(payload)
    assert derived.refresh_cycle_ns == 490.  # the unchanged prior table value follows height
    payload["refresh_cycle_ns"] = 888.
    edited = parse_execution_physical_config(payload)
    assert edited.refresh_cycle_ns == 888.  # a manual value cannot hide behind its old source label
    assert status(edited, "refresh_cycle_ns") == "explicit_assumption"
    payload["metadata"]["parameter_provenance"]["refresh_cycle_ns"] = {
        "status": "explicit_assumption", "source": "test_experiment", "basis": "authored timing input"}
    authored = parse_execution_physical_config(payload)
    assert authored.refresh_cycle_ns == 888.
    assert status(authored, "refresh_cycle_ns") == "explicit_assumption"


@pytest.mark.parametrize("interval,maximum", [(1950., 17550.), (975., 8775.)])
@pytest.mark.parametrize("path", PATHS)
def test_refresh_rate_changes_recompute_the_maximum_active_limit(path, interval, maximum):
    old = reference()
    new = update(old, {"refresh_interval_ns": interval}, path)
    assert new.refresh_interval_ns == interval
    assert new.maximum_active_ns == maximum
    assert new.refresh_cycle_ns == old.refresh_cycle_ns


def test_explicit_ras_rp_inputs_update_cycle_and_do_not_claim_device_validation():
    old = reference("HBM3", 16)
    new = reconfigure_physical_config(old, overrides={"minimum_active_ns": 40., "precharge_to_activate_ns": 18.})
    assert new.activate_cycle_ns == 58.
    for field in ("minimum_active_ns", "precharge_to_activate_ns"):
        assert status(new, field) == "explicit_assumption"
    explicit = reconfigure_physical_config(new, overrides={"activate_cycle_ns": 60.})
    assert explicit.activate_cycle_ns == 60.
    assert status(explicit, "activate_cycle_ns") == "explicit_assumption"
    explicit = reconfigure_physical_config(explicit, overrides={"minimum_active_ns": 42.})
    assert explicit.activate_cycle_ns == 60.
    assert status(explicit, "activate_cycle_ns") == "explicit_assumption"


def test_numerical_array_assumptions_still_leave_the_real_device_parameters_unknown():
    config = reference()
    for field in ("activate_to_read_ns", "activate_to_write_ns", "write_to_precharge_ns"):
        assert getattr(config, field) > 0
        assert status(config, field) == "unknown_with_assumption"
        assert field in config.metadata["unknown_parameters"]
    assert config.metadata["timing_validation"] == "not_a_vendor_speed_bin"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("before,after,density,standard,filename", [
    ("HBM2", "HBM3", 16, "JESD238", "JESD238-2022.pdf"),
    ("HBM3", "HBM2", 16, "JESD235B", "JESD235B-2018.pdf"),
    ("HBM3", "HBM4", 24, "JESD270-4A", "JESD270-4A-2025.pdf"),
    ("HBM4", "HBM3", 24, "JESD238", "JESD238-2022.pdf"),
])
def test_generation_switch_refreshes_the_entire_protocol_source(path, before, after, density, standard, filename):
    old = reference(before, density)
    changed = update(old, {"generation": after}, path)
    direct = reference(after, density)
    metadata = changed.metadata
    assert metadata["jedec_standard"] == standard
    assert metadata["jedec_reference_file"].endswith(filename)
    assert metadata["source_documents"] == direct.metadata["source_documents"]
    assert len(metadata["source_documents"]) == 1
    assert metadata["source_documents"][0]["local_file"].endswith(filename)
    assert metadata["control_basis"] == direct.metadata["control_basis"]
    assert standard in metadata["control_basis"]
    assert metadata["row_bytes_basis"] == direct.metadata["row_bytes_basis"]
    assert metadata["channel_operation_mode_basis"] == direct.metadata["channel_operation_mode_basis"]
    for field in ("row_bytes", "rows_per_bank", "column_same_group_ns", "activate_command_cycles"):
        assert metadata["parameter_provenance"][field]["source"] == metadata["source_documents"][0]["source"]
    assert old.metadata["jedec_standard"] != standard


@pytest.mark.parametrize("change", ["height", "frequency"])
def test_recomputed_parameters_reach_command_execution_and_bus_accounting(change):
    if change == "height":
        old = reference()
        revised = reconfigure_physical_config(old, overrides={"dies_per_stack": 16})
        direct = reference(height=16)
    else:
        old = reference("HBM3", 16)
        direct = reference("HBM3", 16, clock=2.)
        revised = reconfigure_physical_config(old, overrides={
            "command_clock_ns": 2., "read_bandwidth_gb_s": direct.read_bandwidth_gb_s,
            "write_bandwidth_gb_s": direct.write_bandwidth_gb_s})
    cores = [DramCore(config, capture_details=True) for config in (revised, direct)]
    for request in (AccessRequest("first", "read", 0, 32),
                    AccessRequest("after-refresh", "write", 0, 32, 4000.)):
        actual, expected = [core.submit(request) for core in cores]
        assert actual.completion_ns == expected.completion_ns
        assert actual.counters == expected.counters
        assert actual.stages == expected.stages
        assert cores[0]._hbm_control_state == cores[1]._hbm_control_state
        assert cores[0].timeline.busy_ns == cores[1].timeline.busy_ns
        if request.request_id == "after-refresh":
            assert actual.counters["refresh_commands"] > 0
    baseline = DramCore(old, capture_details=True).submit(AccessRequest("baseline", "read", 0, 32))
    revised_result = DramCore(revised, capture_details=True).submit(AccessRequest("revised", "read", 0, 32))
    baseline_dq = next(stage for stage in baseline.stages if stage.name == "BURST_TRANSFER")
    revised_dq = next(stage for stage in revised_result.stages if stage.name == "BURST_TRANSFER")
    assert revised_dq.end_ns - revised_dq.start_ns == 2 * revised.command_clock_ns
    if change == "frequency":
        assert revised_dq.end_ns - revised_dq.start_ns == 2 * (baseline_dq.end_ns - baseline_dq.start_ns)


@pytest.mark.parametrize("path", PATHS)
def test_synthetic_column_floor_survives_clock_changes_until_explicitly_replaced(path):
    original = resolve_physical_config("hbm3e_reference_synthetic_timing", overrides={
        "capacity_bytes": 8 * 2**30, "stacks": 1, "command_clock_ns": .5})
    assert original.command_clock_ns == .5
    assert original.column_same_group_ns == 4.

    def change_clock(config, clock):
        if path == "reconfigure":
            return update(config, {"command_clock_ns": clock}, path)
        # Raw/replace inputs use the physical interface as the authoritative
        # operating point, so their clock changes through that primary input.
        interface = 512. / clock
        return update(config, {"interface_bandwidth_gb_s": interface,
            "read_bandwidth_gb_s": min(config.read_bandwidth_gb_s, interface),
            "write_bandwidth_gb_s": min(config.write_bandwidth_gb_s, interface)}, path)

    slower = change_clock(original, 2.)
    assert slower.command_clock_ns == 2.
    assert slower.column_same_group_ns == 8.
    faster_again = change_clock(slower, .5)
    assert faster_again.column_same_group_ns == 4.
    for config in (original, slower, faster_again):
        provenance = config.metadata["parameter_provenance"]["column_same_group_ns"]
        assert provenance["rule"] == "max_standard_minimum_and_analytical_floor"
        assert provenance["analytical_floor_ns"] == 4.

    fixed = reconfigure_physical_config(original, overrides={"column_same_group_ns": 7.})
    provenance = fixed.metadata["parameter_provenance"]["column_same_group_ns"]
    assert provenance["status"] == "explicit_assumption"
    assert "rule" not in provenance
    assert "analytical_floor_ns" not in provenance
    with pytest.raises(ValueError, match="column_same_group_ns"):
        change_clock(fixed, 2.)  # 7 ns falls below the new 4 CK = 8 ns minimum.
    assert change_clock(fixed, .5).column_same_group_ns == 7.
    conservative = reconfigure_physical_config(fixed, overrides={"column_same_group_ns": 9.})
    assert change_clock(conservative, 2.).column_same_group_ns == 9.
