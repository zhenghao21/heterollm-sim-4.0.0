"""Engineering packs remain traceable and reach real command timing gates."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import rotated_bank_step
from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.hbm_timing_profiles import apply_hbm_timing_profile, timing_profile_catalog
from heterollm_sim.memory_types import AccessRequest, DramConfig
from heterollm_sim.physical_contract import parse_execution_physical_config, reconfigure_physical_config


PATHS = ("reconfigure", "dataclass", "replace", "from_mapping", "execution")


def baseline():
    return DramConfig(**make_hbm_geometry_config(
        generation="HBM3E", capacity_bytes=64 * 2**30, stacks=1,
        die_density_gbits=32, dies_per_stack=16,
        interface_bandwidth_gbps=819.2 * 8,
        read_bandwidth_gbps=819.2 * 8, write_bandwidth_gbps=819.2 * 8))


def update(config, changes, path):
    if path == "reconfigure":
        return reconfigure_physical_config(config, overrides=changes)
    if path == "replace":
        return replace(config, **changes)
    raw = asdict(config)
    raw.update(deepcopy(changes))
    if path == "dataclass":
        return DramConfig(**raw)
    if path == "from_mapping":
        return DramConfig.from_mapping(raw)
    return parse_execution_physical_config(raw)


def request(config, op="read", *, bank=0, row=0, time=0.):
    step = rotated_bank_step(bank, config.bank_count, config.banks_per_group)
    address = ((row * (config.row_bytes // 32)) * config.bank_count + step) * config.lane_count * 32
    return AccessRequest(str((op, bank, row, time)), op, address, 32, time)


def stage(result, name):
    return next(item for item in result.stages if item.name == name)


def half_frequency(config, path):
    # Public reconfigure accepts a CK-only input. Complete raw payloads use
    # their bandwidth_input_mode primary input, with CK derived from it.
    changes = ({"command_clock_ns": 1.25} if path == "reconfigure" else
               {"interface_bandwidth_gb_s": config.interface_bandwidth_gb_s / 2,
                "read_bandwidth_gb_s": config.read_bandwidth_gb_s / 2,
                "write_bandwidth_gb_s": config.write_bandwidth_gb_s / 2})
    return update(config, changes, path)


def test_catalog_records_all_inputs_ranges_and_non_vendor_status_independently():
    catalog = timing_profile_catalog()
    assert set(catalog) == {"none", "fast", "reference", "slow", "ramulator2"}
    assert json.loads(json.dumps(catalog)) == catalog
    for name in ("fast", "reference", "slow"):
        pack = catalog[name]
        assert pack["vendor_calibrated"] is False
        assert len(pack["parameters"]) == 20
        assert pack["parameters"]["refresh_interval_ns"]["input_value"] == 3900.
        assert pack["parameters"]["read_to_precharge_ns"]["input_unit"] == "nCK"
        assert pack["parameters"]["refresh_cycle_ns"]["sensitivity_range"] == [350., 650.]
        assert "TBD" in pack["parameters"]["refresh_cycle_ns"]["basis"]
        assert "IDD example" in pack["parameters"]["minimum_active_ns"]["basis"]
        assert pack["parameters"]["read_to_write_command_ns"]["formula"].startswith("ceil(")
        assert pack["phy_assumptions"]["tdqsqtra_max_ns"] == .020
        assert "not a JEDEC" in pack["phy_assumptions"]["parameter_basis"]["system_turnaround_margin_ns"]
        assert all(entry["status"] == "explicit_assumption" and entry["vendor_value_status"] == "unknown"
                   and entry["source"] and entry["basis"] and entry["input_unit"]
                   for entry in pack["parameters"].values())
        assert all(source["url"].startswith("https://") and source["finding"]
                   for source in pack["vendor_sources_checked"])
    catalog["reference"]["parameters"]["read_latency_ns"]["input_value"] = -1
    assert timing_profile_catalog()["reference"]["parameters"]["read_latency_ns"]["input_value"] == 32


@pytest.mark.parametrize("name", ["REFERENCE", " central", "reference ", "unknown", None, 1])
def test_names_are_exact_and_unknown_names_never_silently_fall_back(name):
    with pytest.raises(ValueError, match="unknown HBM timing profile"):
        apply_hbm_timing_profile(baseline(), name)


def test_none_keeps_explicit_baseline_and_profiles_do_not_change_geometry_or_clock():
    base = baseline()
    assert apply_hbm_timing_profile(base, "none") is base
    for name in ("fast", "reference", "slow"):
        config = apply_hbm_timing_profile(base, name)
        assert config is not base
        for field in ("capacity_bytes", "stacks", "dies_per_stack", "rows_per_bank", "bank_count",
                      "channels", "pseudo_channels_per_channel", "command_clock_ns",
                      "physical_interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s"):
            assert getattr(config, field) == getattr(base, field)
        assert config.command_clock_ns == .625
        assert config.refresh_policy == "all_bank"
        assert config.metadata["hbm_parameter_profile"]["name"] == name
        assert config.metadata["target_hardware_calibrated"] is False
        assert "hbm_parameter_profile" not in base.metadata


@pytest.mark.parametrize("name,rl,wl,rtw,rfc", [
    ("fast", 17.5, 3.75, 16.25, 350.),
    ("reference", 20., 5., 18.125, 500.),
    ("slow", 25., 6.25, 23.125, 650.),
])
def test_selected_clock_converts_cycle_units_and_formula_independently(name, rl, wl, rtw, rfc):
    c = apply_hbm_timing_profile(baseline(), name)
    assert (c.read_latency_ns, c.write_latency_ns, c.read_to_write_command_ns, c.refresh_cycle_ns) == (rl, wl, rtw, rfc)
    # Reference analog RAS 33 ns becomes 106 half-cycles at CK=.625 ns.
    if name == "reference":
        assert (c.minimum_active_ns, c.precharge_to_activate_ns, c.activate_cycle_ns) == (33.125, 15., 48.125)
        assert (c.read_to_precharge_ns, c.write_to_read_same_group_ns, c.write_to_read_different_group_ns) == (3.75, 5., 3.75)
    assert c.maximum_active_ns == 35100.
    for field in c.metadata["hbm_parameter_profile"]["parameters"]:
        assert c.metadata["parameter_provenance"][field]["output_value"] == getattr(c, field)


@pytest.mark.parametrize("path", PATHS)
def test_serialization_and_frequency_changes_recompute_profile_dependencies(path):
    original = apply_hbm_timing_profile(baseline())
    same = update(original, {}, path)
    assert asdict(same) == asdict(original)
    changed = half_frequency(original, path)
    assert changed.command_clock_ns == 1.25
    assert changed.metadata["hbm_parameter_profile"]["overridden_fields"] == []
    assert (changed.read_latency_ns, changed.write_latency_ns) == (40., 10.)
    assert (changed.read_to_precharge_ns, changed.write_to_read_same_group_ns, changed.column_different_sid_ns) == (7.5, 10., 5.)
    assert changed.activate_to_read_ns == 14.  # Analog ns inputs do not scale with CK.
    assert (changed.minimum_active_ns, changed.precharge_to_activate_ns, changed.activate_cycle_ns) == (33.125, 15., 48.125)
    # 40-10+2.5*1.25-0.2+1.5+0.1=34.525 -> 28 CK=35 ns.
    assert changed.read_to_write_command_ns == 35.
    assert changed.refresh_cycle_ns == original.refresh_cycle_ns
    assert changed.metadata["parameter_provenance"]["read_to_precharge_ns"]["input_value"] == 6
    assert changed.metadata["parameter_provenance"]["read_latency_ns"]["cycles"] == 32
    assert asdict(update(changed, {}, path)) == asdict(changed)


@pytest.mark.parametrize("path", PATHS)
def test_authored_primitive_inputs_keep_their_units_and_update_dependent_outputs(path):
    original = apply_hbm_timing_profile(baseline())
    changed = update(original, {"minimum_active_ns": 40., "read_latency_ns": 22.5,
                                "read_to_precharge_ns": 7.}, path)
    assert changed.activate_cycle_ns == 55.
    assert changed.read_to_write_command_ns == 20.625
    assert changed.metadata["hbm_parameter_profile"]["overridden_fields"] == [
        "minimum_active_ns", "read_latency_ns", "read_to_precharge_ns"]
    for field in changed.metadata["hbm_parameter_profile"]["overridden_fields"]:
        provenance = changed.metadata["parameter_provenance"][field]
        assert provenance["input_unit"] == "ns"
        assert provenance["input_value"] == getattr(changed, field)
        assert provenance["source"] == "physical_config_override"
    slower = half_frequency(changed, path)
    assert (slower.minimum_active_ns, slower.read_latency_ns, slower.read_to_precharge_ns) == (40., 22.5, 7.)
    assert slower.write_latency_ns == 10.
    assert slower.read_to_write_command_ns == 17.5
    reset = apply_hbm_timing_profile(slower)
    assert (reset.minimum_active_ns, reset.read_latency_ns, reset.read_to_precharge_ns) == (33.125, 40., 7.5)
    assert reset.metadata["hbm_parameter_profile"]["overridden_fields"] == []


@pytest.mark.parametrize("path", PATHS)
def test_disabling_refresh_keeps_independent_maximum_open_row_limit(path):
    original = apply_hbm_timing_profile(baseline())
    disabled = update(original, {"refresh_policy": "disabled", "refresh_interval_ns": 0., "refresh_cycle_ns": 0.}, path)
    assert (disabled.refresh_policy, disabled.refresh_interval_ns, disabled.refresh_cycle_ns) == ("disabled", 0., 0.)
    assert disabled.maximum_active_ns == 35100.
    assert update(disabled, {}, path).maximum_active_ns == 35100.
    assert half_frequency(disabled, path).maximum_active_ns == 35100.
    doubled = update(original, {"refresh_interval_ns": 1950.}, path)
    assert doubled.maximum_active_ns == 17550.


def test_reference_profile_enforces_independent_array_read_and_write_command_expectations():
    c = apply_hbm_timing_profile(baseline())
    core = DramCore(c)
    read = core.submit(request(c))
    write = core.submit(request(c, "write"))
    following = core.submit(request(c))
    # ACT's array anchor=.625; ceil((.625+14)/.625)=24 rising edges.
    assert stage(read, "READ_COMMAND").start_ns == 15.
    assert (stage(read, "BURST_TRANSFER").start_ns, read.completion_ns) == (35., 36.25)
    # RTW=29 CK wins over DQ occupancy and WL; WTR_L=8 CK follows write DQ end.
    assert stage(write, "WRITE_COMMAND").start_ns == 33.125
    assert write.completion_ns == 39.375
    assert stage(following, "READ_COMMAND").start_ns == 44.375
    writer = DramCore(c)
    first = writer.submit(request(c, "write"))
    conflict = writer.submit(request(c, row=1))
    assert stage(first, "WRITE_COMMAND").start_ns == 15.
    assert stage(conflict, "PRECHARGE").start_ns == 36.25  # write DQ end 21.25 + WR 15.
    assert stage(conflict, "ACTIVATE").start_ns == 51.25  # PRE + RP 15.


def test_reference_profile_activation_windows_and_late_read_recovery_are_active():
    c = apply_hbm_timing_profile(baseline())
    core = DramCore(c)
    same = [core.submit(request(c, bank=bank)) for bank in (0, 1)]
    assert [stage(item, "ACTIVATE").start_ns for item in same] == [0., 6.25]
    core = DramCore(c)
    groups = [core.submit(request(c, bank=bank)) for bank in (0, 4, 8, 12, 16)]
    assert [stage(item, "ACTIVATE").start_ns for item in groups] == [0., 4.375, 8.75, 13.125, 20.]
    core = DramCore(c)
    core.submit(request(c))
    late = core.submit(request(c, time=100.))
    conflict = core.submit(request(c, time=100., row=1))
    assert stage(late, "READ_COMMAND").start_ns == 100.
    assert stage(conflict, "PRECHARGE").start_ns == 103.75  # RTP=6 CK.


def test_reference_refresh_executes_the_assumed_rfc_and_interval():
    c = apply_hbm_timing_profile(baseline())
    result = DramCore(c).submit(request(c, time=3900.))
    recovery = [item for item in result.stages if item.name == "REFRESH_RECOVERY"]
    assert [(item.start_ns, item.end_ns) for item in recovery] == [(3900., 4400.), (3900.625, 4400.625)]
    assert result.counters["refresh_commands"] == 2
    assert stage(result, "ACTIVATE").start_ns == 4400.
    assert c.metadata["parameter_provenance"]["refresh_cycle_ns"]["vendor_value_status"] == "unknown"


def test_reference_different_sid_and_different_group_turnaround_are_active():
    c = apply_hbm_timing_profile(baseline())
    core = DramCore(c)
    for bank in (0, 16):
        core.submit(request(c, bank=bank))
    first = core.submit(request(c, bank=0, time=100.))
    second = core.submit(request(c, bank=16, time=100.))
    assert stage(first, "READ_COMMAND").start_ns == 100.
    assert stage(second, "READ_COMMAND").start_ns == 102.5  # tCCDR=4 CK, not tCCDS=2 CK.
    core = DramCore(c)
    for bank in (0, 4):
        core.submit(request(c, bank=bank))
    written = core.submit(request(c, "write", bank=0, time=100.))
    read = core.submit(request(c, bank=4, time=100.))
    assert written.completion_ns == 106.25
    assert stage(read, "READ_COMMAND").start_ns == 110.  # WTR_S=6 CK after write DQ end.


def test_ramulator_profile_preserves_source_cycles_and_maps_to_local_command_inputs():
    c = apply_hbm_timing_profile(baseline(), "ramulator2")
    expected = {
        "read_latency_ns": 12.5, "write_latency_ns": 6.25,
        "activate_to_read_ns": 19.375, "activate_to_write_ns": 9.375,
        "precharge_to_activate_ns": 16.25, "minimum_active_ns": 28.125,
        "activate_cycle_ns": 44.375, "write_to_precharge_ns": 20.625,
        "read_to_precharge_ns": 5.625,
        "activate_different_group_ns": 2.5, "activate_same_group_ns": 3.125,
        "four_activate_window_ns": 15., "write_to_read_different_group_ns": 4.375,
        "write_to_read_same_group_ns": 6.25, "column_different_group_ns": 1.25,
        "column_same_group_ns": 2.5, "column_different_sid_ns": 1.875,
        "precharge_to_precharge_ns": 1.25, "refresh_cycle_ns": 550.,
        "refresh_interval_ns": 3900., "read_to_write_command_ns": 10.625,
        "maximum_active_ns": 35100., "refresh_policy": "all_bank",
    }
    pack = c.metadata["hbm_parameter_profile"]
    assert set(pack["parameters"]) == set(expected)
    for field, value in expected.items():
        assert getattr(c, field) == value
        assert c.metadata["parameter_provenance"][field]["output_value"] == value
    source = pack["upstream_source"]
    assert source["commit"] == "54e90367191062067e9be74840439b0daca40343"
    assert source["commit"] in source["url"]
    assert source["timing_preset"] == "HBM3_6400Mbps"
    assert source["organization_preset"] == "HBM3_32Gb_16hi"
    assert pack["upstream_raw_timings"]["nCL"] == 20  # Do not double-count nBL in RL.
    assert pack["upstream_raw_timings"]["nBL"] == 2
    assert pack["upstream_raw_timings"]["nRTW"] == 17
    assert pack["upstream_raw_timings"]["nRFC"] == 880
    assert set(pack["unmapped_upstream_parameters"]) == {"nRFCpb", "nRFMab", "nRFMpb", "nRREFD", "nREFIpb"}
    provenance = c.metadata["parameter_provenance"]
    assert provenance["refresh_cycle_ns"]["source_kind"] == "upstream_estimate"
    assert provenance["refresh_cycle_ns"]["upstream_input_ns"] == 550.
    assert provenance["refresh_cycle_ns"]["source"] == source["url"]
    assert provenance["column_same_group_ns"]["source_kind"] == "upstream_formula"
    assert provenance["maximum_active_ns"]["source_kind"] == "local_standard_supplement"
    assert provenance["refresh_policy"]["source_kind"] == "local_controller_policy"
    assert "do not reproduce its command timeline" in pack["mapping_notes"]["execution"]
    assert pack["vendor_calibrated"] is False
    assert c.metadata["target_hardware_calibrated"] is False
    assert c.metadata["vendor_verified"] is False


@pytest.mark.parametrize("path", PATHS)
def test_ramulator_profile_roundtrips_and_refuses_unavailable_operating_points(path):
    c = apply_hbm_timing_profile(baseline(), "ramulator2")
    assert asdict(update(c, {}, path)) == asdict(c)
    with pytest.raises(ValueError, match="ramulator2 timing profile requires"):
        half_frequency(c, path)
    changed = update(c, {"read_latency_ns": 15., "minimum_active_ns": 30.}, path)
    assert changed.read_to_write_command_ns == 13.125
    assert changed.activate_cycle_ns == 46.25
    assert changed.metadata["hbm_parameter_profile"]["overridden_fields"] == ["minimum_active_ns", "read_latency_ns"]
    assert changed.metadata["parameter_provenance"]["read_latency_ns"]["source_kind"] == "caller_override"
    assert asdict(update(changed, {}, path)) == asdict(changed)
    assert asdict(apply_hbm_timing_profile(changed, "ramulator2")) == asdict(c)


def test_ramulator_inputs_reach_existing_command_gates_without_replacing_controller():
    c = apply_hbm_timing_profile(baseline(), "ramulator2")
    core = DramCore(c)
    read = core.submit(request(c))
    write = core.submit(request(c, "write"))
    following = core.submit(request(c))
    # Local ACT anchor=.625, plus imported 31CK RCDRD; RL excludes nBL.
    assert stage(read, "READ_COMMAND").start_ns == 20.
    assert (stage(read, "BURST_TRANSFER").start_ns, read.completion_ns) == (32.5, 33.75)
    assert stage(write, "WRITE_COMMAND").start_ns == 30.625  # Imported 17CK RTW.
    assert write.completion_ns == 38.125
    assert stage(following, "READ_COMMAND").start_ns == 44.375  # 10CK WTRL from write DQ end.
    writer = DramCore(c)
    first = writer.submit(request(c, "write"))
    conflict = writer.submit(request(c, row=1))
    assert stage(first, "WRITE_COMMAND").start_ns == 10.
    assert stage(conflict, "PRECHARGE").start_ns == 38.125
    assert stage(conflict, "ACTIVATE").start_ns == 54.375
    refresh = DramCore(c).submit(request(c, time=3900.))
    assert stage(refresh, "REFRESH_RECOVERY").end_ns == 4450.
    assert stage(refresh, "ACTIVATE").start_ns == 4450.
