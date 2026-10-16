"""Named physical presets expose engineering timing packs to every caller."""
from dataclasses import asdict
import json

import pytest

from heterollm_sim.memory_types import DramConfig
from heterollm_sim.physical_contract import parse_execution_physical_config, reconfigure_physical_config
from heterollm_sim.physical_presets import physical_config_profile_ids, resolve_physical_config


@pytest.mark.parametrize("name,rfc,rtw", [("fast", 350., 16.25), ("reference", 500., 18.125), ("slow", 650., 23.125), ("ramulator2", 550., 10.625)])
def test_timing_presets_are_discoverable_complete_and_correctly_sourced(name, rfc, rtw):
    profile_id = "hbm3e_timing_" + name
    assert profile_id in physical_config_profile_ids()
    config = resolve_physical_config(profile_id)
    assert (config.generation, config.dies_per_stack, config.rows_per_bank) == ("HBM3E", 16, 32768)
    assert config.capacity_bytes == config.computed_capacity_bytes == 512 * 2**30
    assert config.stacks == 8
    assert config.command_clock_ns == .625
    assert config.interface_bandwidth_gb_s == 6553.6
    assert (config.refresh_cycle_ns, config.read_to_write_command_ns) == (rfc, rtw)
    assert config.metadata["physical_config_profile_id"] == profile_id
    assert config.metadata["physical_config_override_fields"] == []
    assert config.metadata["hbm_parameter_profile"]["name"] == name
    assert config.metadata["hbm_parameter_profile"]["vendor_calibrated"] is False
    assert config.metadata["vendor_verified"] is False
    assert config.metadata["target_hardware_calibrated"] is False
    for field in config.metadata["hbm_parameter_profile"]["parameters"]:
        entry = config.metadata["parameter_provenance"][field]
        assert entry["profile_name"] == name
        assert entry["output_value"] == getattr(config, field)
        assert entry["vendor_value_status"] == "unknown"
    raw = json.loads(json.dumps(asdict(config)))
    assert asdict(DramConfig.from_mapping(raw)) == asdict(config)
    assert asdict(parse_execution_physical_config(raw)) == asdict(config)
    assert asdict(reconfigure_physical_config(config)) == asdict(config)


@pytest.mark.parametrize("capacity,stacks", [(256 * 2**30, 8), (1024 * 2**30, 16)])
def test_capacity_uses_existing_covering_stack_rule_without_changing_clock(capacity, stacks):
    config = resolve_physical_config("hbm3e_timing_reference", overrides={"capacity_bytes": capacity})
    assert config.capacity_bytes == capacity
    assert config.stacks == stacks
    assert config.command_clock_ns == .625
    assert config.interface_bandwidth_gb_s == stacks * 819.2
    assert config.computed_capacity_bytes == stacks * 64 * 2**30
    assert config.metadata["physical_config_override_fields"] == ["capacity_bytes"]


def test_explicit_clock_array_timing_capacity_and_controller_inputs_take_precedence():
    overrides = dict(capacity_bytes=64 * 2**30, stacks=1, command_clock_ns=1.25,
        activate_to_read_ns=21., activate_different_group_ns=0.,
        scheduler="row_hit_first", scheduler_window=32, queue_scope="channel",
        max_outstanding_requests=128, read_bandwidth_gb_s=200., write_bandwidth_gb_s=100.)
    config = resolve_physical_config("hbm3e_timing_reference", overrides=overrides)
    for field, expected in overrides.items():
        assert getattr(config, field) == expected
    assert config.interface_bandwidth_gb_s == 409.6
    assert (config.read_latency_ns, config.write_latency_ns, config.read_to_precharge_ns) == (40., 10., 7.5)
    assert config.read_to_write_command_ns == 35.
    assert config.metadata["physical_config_override_fields"] == sorted(overrides)
    assert config.metadata["hbm_parameter_profile"]["overridden_fields"] == [
        "activate_different_group_ns", "activate_to_read_ns"]
    assert config.metadata["parameter_provenance"]["activate_to_read_ns"]["source"] == "physical_config_override"
    # Existing configuration edits still use the same public contract and do
    # not require resolving or applying the preset a second time.
    changed = reconfigure_physical_config(config, overrides={"command_clock_ns": .625})
    assert changed.activate_to_read_ns == 21.
    assert changed.activate_different_group_ns == 0.
    assert (changed.read_latency_ns, changed.read_to_write_command_ns) == (20., 18.125)
    assert asdict(parse_execution_physical_config(asdict(changed))) == asdict(changed)


def test_explicit_rate_mode_and_cleared_interface_are_applied_after_pack_selection():
    config = resolve_physical_config("hbm3e_timing_fast", overrides={
        "interface_bandwidth_gb_s": None, "bandwidth_input_mode": "data_rate", "command_clock_ns": 1.25,
        "read_to_precharge_ns": 9.})
    assert config.command_clock_ns == 1.25
    assert config.data_rate_mt_s == 3200.
    assert config.read_latency_ns == 35.
    assert config.read_to_precharge_ns == 9.
    assert config.metadata["hbm_parameter_profile"]["name"] == "fast"


def test_metadata_identity_tracks_the_selected_preset_and_inputs_are_not_mutated():
    overrides = {"activate_to_read_ns": 30., "metadata": {
        "physical_config_profile_id": "caller_label", "experiment_note": "test",
        "hbm_parameter_profile": {"name": "slow", "version": 1}}}
    before = json.loads(json.dumps(overrides))
    config = resolve_physical_config("hbm3e_timing_reference", overrides=overrides)
    assert overrides == before
    assert config.metadata["physical_config_profile_id"] == "hbm3e_timing_reference"
    assert config.metadata["hbm_parameter_profile"]["name"] == "reference"
    assert config.metadata["experiment_note"] == "test"
    assert config.activate_to_read_ns == 30.
    assert resolve_physical_config("hbm3e_timing_reference").activate_to_read_ns == 14.


def test_refresh_ablation_survives_named_preset_resolution_and_readback():
    config = resolve_physical_config("hbm3e_timing_slow", overrides={
        "refresh_policy": "disabled", "refresh_interval_ns": 0., "refresh_cycle_ns": 0.})
    assert (config.refresh_policy, config.refresh_interval_ns, config.refresh_cycle_ns) == ("disabled", 0., 0.)
    assert config.maximum_active_ns == 35100.
    assert asdict(parse_execution_physical_config(asdict(config))) == asdict(config)


@pytest.mark.parametrize("overrides", [{"generation": "HBM4"}, {"die_density_gbits": 24, "capacity_bytes": 128 * 2**30}])
def test_packs_do_not_silently_apply_to_unsupported_organizations(overrides):
    with pytest.raises(ValueError, match="HBM timing packs require"):
        resolve_physical_config("hbm3e_timing_reference", overrides=overrides)


@pytest.mark.parametrize("profile_id,control", [
    ("hbm3e_reference", "standard"), ("hbm3e_reference_flat", "flat"),
    ("hbm3e_reference_synthetic_timing", "timed"), ("hbm4_reference", "standard"),
])
def test_existing_geometry_control_presets_do_not_acquire_timing_packs(profile_id, control):
    config = resolve_physical_config(profile_id)
    assert "hbm_parameter_profile" not in config.metadata
    assert config.metadata["geometry_control"] == control
    assert config.metadata["physical_config_profile_id"] == profile_id
    assert config.interface_bandwidth_gb_s == 4000.


def test_ramulator_preset_keeps_the_upstream_operating_point_and_independent_caps():
    c = resolve_physical_config("hbm3e_timing_ramulator2", overrides={
        "stacks": 1, "capacity_bytes": 64 * 2**30,
        "read_bandwidth_gb_s": 200., "write_bandwidth_gb_s": 100.})
    assert c.command_clock_ns == .625
    assert c.interface_bandwidth_gb_s == 819.2
    assert (c.read_bandwidth_gb_s, c.write_bandwidth_gb_s) == (200., 100.)
    assert c.metadata["hbm_parameter_profile"]["name"] == "ramulator2"
    assert c.metadata["hbm_parameter_profile"]["overridden_fields"] == []
    for changes in ({"command_clock_ns": 1.25}, {"interface_bandwidth_gb_s": 3276.8},
                    {"bandwidth_input_mode": "data_rate", "data_rate_mt_s": 8000.}):
        with pytest.raises(ValueError, match="ramulator2 timing profile requires"):
            resolve_physical_config("hbm3e_timing_ramulator2", overrides=changes)
