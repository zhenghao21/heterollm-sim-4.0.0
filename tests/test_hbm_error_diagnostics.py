"""Geometry and budget rejections retain actionable HBM diagnostics."""

import pytest

from heterollm_sim._hbm_errors import HBMConfigError
from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.memory_types import DramConfig


def reference(**updates):
    payload = make_hbm_geometry_config(
        generation="HBM4", capacity_bytes=1 << 30,
        read_bandwidth_gbps=8192., write_bandwidth_gbps=8192.,
        die_density_gbits=24, dies_per_stack=8,
    )
    payload.update(updates)
    return payload


def test_pc_geometry_error_identifies_the_unsupported_field_and_required_value():
    with pytest.raises(HBMConfigError, match="requires PC-mode HBM geometry") as captured:
        DramConfig.from_mapping(reference(pseudo_channels_per_channel=1))
    error = captured.value
    assert error.details["category"] == "model_limit"
    assert error.details["field"] == "pseudo_channels_per_channel"
    assert error.details["value"] == 1
    assert error.details["related_values"]["conflicts"]["pseudo_channels_per_channel"] == {
        "actual": 1, "required": 2,
    }
    assert error.details["remedy_zh"] and error.details["remedy_en"]


def test_capacity_error_reports_available_physical_capacity():
    with pytest.raises(HBMConfigError, match="capacity_bytes cannot exceed the declared physical geometry") as captured:
        DramConfig.from_mapping(reference(capacity_bytes=1 << 40))
    error = captured.value
    assert error.details["field"] == "capacity_bytes"
    assert error.details["value"] == 1 << 40
    assert error.details["category"] == "invalid_input"
    assert error.details["related_values"]["computed_capacity_bytes"] == 24 << 30
    assert "capacity_bytes" in error.message_en
    assert "容量" in error.message_zh


@pytest.mark.parametrize("field", ["read_bandwidth_gb_s", "write_bandwidth_gb_s"])
def test_directional_budget_error_reports_the_physical_interface_limit(field):
    with pytest.raises(HBMConfigError, match="directional bandwidth cannot exceed") as captured:
        DramConfig.from_mapping(reference(**{field: 2048.}))
    error = captured.value
    assert error.details["field"] == field
    assert error.details["value"] == 2048.
    assert error.details["category"] == "invalid_input"
    assert error.details["related_values"]["physical_interface_bandwidth_gb_s"] == 1024.


def test_explicit_raw_clock_conflict_reports_the_ck_dq_relationship():
    # No provenance requests automatic dependency rewriting. All authored
    # timing gates remain legal at 0.5 ns; the physical interface requires 1 ns.
    with pytest.raises(HBMConfigError, match="command_clock_ns conflicts with the JEDEC CK/DQ ratio") as captured:
        DramConfig.from_mapping(reference(metadata={}, command_clock_ns=0.5))
    error = captured.value
    assert error.details["category"] == "standard_limit"
    assert error.details["field"] == "command_clock_ns"
    assert error.details["value"] == 0.5
    assert error.details["related_values"]["required_clock_ns"] == 1.
    assert error.details["related_values"]["transfers_per_command_clock"] == 4
    assert error.details["source"] == "JESD270-4A"


@pytest.mark.parametrize("value,category", [(-1, "invalid_input"), (99, "model_limit")])
def test_explicit_row_address_msb_error_keeps_the_authored_value(value, category):
    with pytest.raises(HBMConfigError, match="activated_row_msb") as captured:
        DramConfig.from_mapping(reference(metadata={}, activated_row_msb=value))
    error = captured.value
    assert error.details["field"] == "activated_row_msb"
    assert error.details["value"] == value
    assert error.details["category"] == category
    assert error.details["related_values"]["rows_per_bank"] == 12288
    if value == 99:
        assert error.details["related_values"]["required_value"] == 13


def test_non_mapping_metadata_keeps_the_normalizer_diagnostic():
    with pytest.raises(HBMConfigError, match="metadata must be a mapping") as captured:
        DramConfig.from_mapping(reference(metadata=None))
    assert captured.value.details["field"] == "metadata"
    assert captured.value.details["value"] is None


def test_capacity_alignment_is_identified_as_a_model_requirement():
    with pytest.raises(HBMConfigError, match="capacity_bytes must be a multiple of burst_bytes") as captured:
        DramConfig.from_mapping(reference(capacity_bytes=(1 << 30) + 1))
    assert captured.value.details["category"] == "model_limit"
    assert captured.value.details["related_values"]["burst_bytes"] == 32


def test_legacy_directional_budget_error_retains_its_existing_exception_type():
    with pytest.raises(ValueError, match="directional bandwidth cannot exceed") as captured:
        DramConfig.from_mapping(reference(control_mode="analytical", refresh_policy="disabled",
                                          read_bandwidth_gb_s=2048.))
    assert type(captured.value) is ValueError
