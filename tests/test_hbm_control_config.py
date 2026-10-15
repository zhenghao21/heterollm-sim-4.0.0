"""Protocol rules, reference selections and unavailable vendor inputs stay distinct."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.memory_types import DramConfig
from heterollm_sim.physical_contract import parse_execution_physical_config, reconfigure_physical_config
from heterollm_sim.physical_presets import resolve_physical_config


def reference(generation="HBM3", density=16, height=8):
    return parse_execution_physical_config(make_hbm_geometry_config(
        generation=generation, capacity_bytes=density * height * 2**30 // 8,
        die_density_gbits=density, dies_per_stack=height,
        read_bandwidth_gbps=4096, write_bandwidth_gbps=2048))


@pytest.mark.parametrize("generation,density,rf,occupancy", [
    ("HBM2", 16, 450, (2, 1)), ("HBM2E", 16, 450, (2, 1)),
    ("HBM3", 16, 350, (1.5, .5)), ("HBM3E", 16, 350, (1.5, .5)),
    ("HBM4", 24, 410, (1.5, .5)),
])
def test_generational_command_clock_latency_and_refresh(generation, density, rf, occupancy):
    config = reference(generation, density)
    ck = config.command_clock_ns
    assert config.control_mode == "jedec_hbm"
    assert (config.activate_command_cycles, config.precharge_command_cycles) == occupancy
    assert config.read_latency_ns == 32 * ck
    assert config.write_latency_ns == 8 * ck
    assert config.burst_bytes / (config.physical_interface_bandwidth_gb_s / config.lane_count) == 2 * ck
    assert config.column_different_group_ns == 2 * ck
    assert config.precharge_to_precharge_ns == (1 if generation in {"HBM2", "HBM2E"} else 2) * ck
    assert config.maximum_active_ns == 35100
    assert config.refresh_policy == "all_bank"
    assert (config.refresh_interval_ns, config.refresh_cycle_ns) == (3900, rf)
    assert config.metadata["refresh_interval_scope"]
    p = config.metadata["parameter_provenance"]
    assert p["minimum_active_ns"]["status"] == "jedec_idd_example"
    assert p["activate_to_read_ns"]["status"] == "unknown_with_assumption"
    assert "activate_to_read_ns" in config.metadata["unknown_parameters"]
    assert p["read_to_write_command_ns"]["status"] == "unknown_disabled"
    assert config.metadata["timing_validation"] == "not_a_vendor_speed_bin"


def test_hbm3_tbd_refresh_is_not_copied_from_hbm4():
    third, fourth = reference("HBM3", 24, 12), reference("HBM4", 24, 12)
    assert (third.refresh_policy, third.refresh_cycle_ns) == ("disabled", 0)
    assert third.maximum_active_ns == fourth.maximum_active_ns == 35100
    assert "refresh_cycle_ns" in third.metadata["unconfigured_timing_parameters"]
    assert (fourth.refresh_policy, fourth.refresh_cycle_ns) == ("all_bank", 450)


@pytest.mark.parametrize("update", [
    {"command_clock_ns": -1.}, {"activate_command_cycles": 2},
    {"read_latency_ns": 4.5}, {"write_latency_ns": 17},
    {"activated_row_msb": 0}, {"refresh_cycle_ns": 0, "refresh_policy": "all_bank"},
    {"ranks_per_channel": 2}, {"data_lanes": 64},
])
def test_invalid_authored_control_contract_is_rejected(update):
    with pytest.raises(ValueError):
        reconfigure_physical_config(reference(), overrides=update)


def test_missing_control_contract_does_not_silently_use_legacy_or_default_fields():
    config = reference()
    payload = asdict(config)
    del payload["read_to_precharge_ns"]
    with pytest.raises(ValueError, match="missing explicit fields"):
        parse_execution_physical_config(payload)
    payload = asdict(config)
    payload.update(control_mode="analytical", refresh_policy="disabled")
    with pytest.raises(ValueError, match="JEDEC control status"):
        parse_execution_physical_config(payload)


def test_known_refresh_minimum_cannot_be_disabled_by_an_experimental_zero():
    for updates in ({"refresh_cycle_ns": 0.},
                    {"refresh_cycle_ns": 0., "refresh_policy": "disabled"}):
        with pytest.raises(ValueError, match="refresh_cycle_ns"):
            reconfigure_physical_config(reference(), overrides=updates)


def test_raw_unchanged_derived_fields_follow_inputs_and_edited_timings_remain_explicit():
    config = reference()
    # Unchanged serialized derived values follow the changed primary clock.
    rebuilt = replace(config, interface_bandwidth_gb_s=256.,
                      read_bandwidth_gb_s=256., write_bandwidth_gb_s=256.)
    assert (rebuilt.command_clock_ns, rebuilt.read_latency_ns, rebuilt.write_latency_ns) == (2., 64., 16.)
    assert rebuilt.refresh_cycle_ns == config.refresh_cycle_ns
    # Manual edits remain authored values even with old derived source tags.
    edited = replace(config, read_latency_ns=40., write_latency_ns=10., refresh_cycle_ns=500.)
    for field, value in (("read_latency_ns", 40.), ("write_latency_ns", 10.), ("refresh_cycle_ns", 500.)):
        assert getattr(edited, field) == value
        assert edited.metadata["parameter_provenance"][field]["status"] == "explicit_assumption"


def test_overrides_record_assumptions_and_keep_command_anchored_fields_effective():
    old = reference()
    config = reconfigure_physical_config(old, overrides={
        "activate_to_read_ns": 17., "precharge_to_activate_ns": 20., "write_to_precharge_ns": 21.,
        "activate_to_write_ns": 18., "read_to_precharge_ns": 8.})
    assert (config.activate_to_read_ns, config.activate_to_write_ns) == (17, 18)
    assert config.precharge_to_activate_ns == 20
    assert config.activate_cycle_ns == config.minimum_active_ns + 20
    assert config.write_to_precharge_ns == 21
    assert "read_to_precharge_ns" not in config.metadata["unconfigured_timing_parameters"]
    for name in ("activate_to_read_ns", "write_to_precharge_ns", "precharge_to_activate_ns"):
        assert config.metadata["parameter_provenance"][name]["status"] == "explicit_assumption"
    assert old.activate_to_read_ns == 14


def test_micro_reference_rejects_unencoded_latency_instead_of_rounding_it():
    with pytest.raises(ValueError, match="whole command clocks"):
        resolve_physical_config("research_hbm_relay", overrides={"read_latency_ns": 9.})


def test_shorter_reference_refresh_interval_also_shortens_maximum_open_time():
    updated = reconfigure_physical_config(reference(), overrides={"refresh_interval_ns": 1950.})
    assert updated.maximum_active_ns == 17550.
    assert updated.metadata["parameter_provenance"]["maximum_active_ns"]["status"] == "derived_from_input"
