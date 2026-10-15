"""Hard protocol limits apply independently of timing provenance or entry point."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from heterollm_sim._hbm_errors import HBMConfigError
from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.memory_types import DramConfig
from heterollm_sim.physical_contract import (
    parse_execution_physical_config, reconfigure_physical_config,
)


PATHS = ("reconfigure", "dataclass", "replace", "from_mapping", "execution")
GENERATIONS = ("HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4")
FIELDS = ("column_same_group_ns", "column_different_group_ns",
          "column_different_sid_ns", "precharge_to_precharge_ns")


def reference(generation="HBM3", clock=1.):
    lanes = 16 if generation in {"HBM2", "HBM2E"} else 64 if generation == "HBM4" else 32
    interface_gbps = lanes * 16. * 8. / clock
    return DramConfig(**make_hbm_geometry_config(
        generation=generation, die_density_gbits=24 if generation == "HBM4" else 16,
        dies_per_stack=8, capacity_bytes=8 * 2**30,
        interface_bandwidth_gbps=interface_gbps,
        read_bandwidth_gbps=interface_gbps, write_bandwidth_gbps=interface_gbps))


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


def lower_bounds(generation, clock):
    hbm2 = generation in {"HBM2", "HBM2E"}
    return dict(zip(FIELDS, (max(4 * clock, 2.8 if hbm2 else 2.5),
                            2 * clock, (2 if hbm2 else 3) * clock,
                            (1 if hbm2 else 2) * clock)))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("generation", GENERATIONS)
@pytest.mark.parametrize("field", FIELDS)
def test_all_config_entries_reject_authored_values_below_protocol_minima(path, generation, field):
    config = reference(generation)
    minimum = lower_bounds(generation, config.command_clock_ns)[field]
    # Keep the existing standard provenance: a source label cannot make an
    # explicitly supplied illegal value valid before normalization.
    with pytest.raises(ValueError, match=field):
        update(config, {field: minimum - .125}, path)


@pytest.mark.parametrize("generation", GENERATIONS)
def test_exact_minima_and_conservative_longer_delays_are_allowed(generation):
    config = reference(generation, clock=.5)
    bounds = lower_bounds(generation, config.command_clock_ns)
    exact = reconfigure_physical_config(config, overrides=bounds)
    assert {name: getattr(exact, name) for name in FIELDS} == bounds
    longer = reconfigure_physical_config(exact, overrides={name: value + 10. for name, value in bounds.items()})
    assert all(getattr(longer, name) == bounds[name] + 10. for name in FIELDS)
    assert longer.metadata["timing_validation"] == "not_a_vendor_speed_bin"
    assert longer.metadata["vendor_verified"] is False
    assert "column_different_sid_ns" in longer.metadata["unknown_parameters"]


@pytest.mark.parametrize("generation", GENERATIONS)
def test_same_group_minimum_includes_absolute_ns_limit_at_high_frequency(generation):
    config = reference(generation, clock=.5)
    with pytest.raises(ValueError, match="column_same_group_ns"):
        reconfigure_physical_config(config, overrides={"column_same_group_ns": 2.4})


@pytest.mark.parametrize("status", ["explicit_assumption", "unknown_with_assumption", "unknown",
                                     "standard_minimum", "derived_from_input", "selected_reference"])
def test_metadata_labels_cannot_bypass_an_illegal_timing(status):
    payload = asdict(reference())
    payload["column_same_group_ns"] = 0.
    payload["metadata"]["parameter_provenance"]["column_same_group_ns"] = {
        "status": status, "source": "test input", "basis": "self-declared timing"}
    with pytest.raises(ValueError, match="column_same_group_ns"):
        parse_execution_physical_config(payload)


@pytest.mark.parametrize("path", PATHS)
def test_frequency_change_rejects_a_fixed_experiment_that_becomes_too_short(path):
    fixed = reconfigure_physical_config(reference(), overrides={"column_same_group_ns": 5.})
    values = {"command_clock_ns": 2.} if path == "reconfigure" else {
        "interface_bandwidth_gb_s": 256., "read_bandwidth_gb_s": 256., "write_bandwidth_gb_s": 256.}
    with pytest.raises(ValueError, match="column_same_group_ns"):
        update(fixed, values, path)


@pytest.mark.parametrize("path", PATHS)
def test_generation_change_rejects_previous_generation_specific_experiment(path):
    hbm2 = reconfigure_physical_config(reference("HBM2"), overrides={
        "precharge_to_precharge_ns": 1., "column_different_sid_ns": 2.})
    with pytest.raises(ValueError):
        update(hbm2, {"generation": "HBM3"}, path)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("field,value", [
    ("refresh_cycle_ns", 409.), ("refresh_cycle_ns", 0.),
    ("maximum_active_ns", 35101.), ("maximum_active_ns", 0.),
    ("refresh_interval_ns", 3901.),
])
def test_refresh_and_maximum_active_bounds_cannot_be_disabled_by_manual_values(path, field, value):
    with pytest.raises(ValueError, match=field):
        update(reference("HBM4"), {field: value}, path)


def test_conservative_refresh_and_maximum_active_assumptions_remain_explicit():
    config = reconfigure_physical_config(reference("HBM4"), overrides={
        "refresh_cycle_ns": 500., "maximum_active_ns": 20000.})
    assert config.refresh_cycle_ns == 500.
    assert config.maximum_active_ns == 20000.
    for field in ("refresh_cycle_ns", "maximum_active_ns"):
        assert config.metadata["parameter_provenance"][field]["status"] == "explicit_assumption"


def test_periodic_refresh_rejects_a_cycle_longer_than_its_interval():
    with pytest.raises(ValueError):
        reconfigure_physical_config(reference("HBM4"), overrides={
            "refresh_cycle_ns": 500., "refresh_interval_ns": 450.})


def test_unknown_device_array_delays_do_not_acquire_idd_example_minima():
    config = reconfigure_physical_config(reference(), overrides={
        "minimum_active_ns": 7., "precharge_to_activate_ns": 3., "activate_cycle_ns": 5.,
        "activate_to_read_ns": 4., "activate_to_write_ns": 3., "write_to_precharge_ns": 4.})
    assert (config.minimum_active_ns, config.precharge_to_activate_ns, config.activate_cycle_ns) == (7., 3., 5.)
    for field in ("minimum_active_ns", "precharge_to_activate_ns", "activate_cycle_ns",
                  "activate_to_read_ns", "activate_to_write_ns", "write_to_precharge_ns"):
        assert config.metadata["parameter_provenance"][field]["status"] == "explicit_assumption"
        assert field in config.metadata["unknown_parameters"]
    assert config.metadata["vendor_verified"] is False


def test_unknown_refresh_duration_stays_unknown_instead_of_inheriting_a_table_value():
    payload = make_hbm_geometry_config(generation="HBM3", die_density_gbits=24,
        dies_per_stack=8, capacity_bytes=8 * 2**30,
        read_bandwidth_gbps=4096., write_bandwidth_gbps=4096.)
    config = parse_execution_physical_config(payload)
    assert config.refresh_cycle_ns == 0.
    assert config.refresh_policy == "disabled"
    assert "refresh_cycle_ns" in config.metadata["unknown_parameters"]
    assert config.maximum_active_ns == 35100.
    assert config.metadata["vendor_verified"] is False
    assert config.metadata["timing_constraint_validation"] == "known_jedec_constraints_checked"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("field,value", [("maximum_active_ns", 1.), ("refresh_cycle_ns", 4000.)])
def test_manual_coupled_violation_is_rejected_before_automatic_provenance_can_hide_it(path, field, value):
    config = reference("HBM4")
    assert config.refresh_interval_ns == 3900.
    with pytest.raises(ValueError, match=field):
        update(config, {field: value}, path)


@pytest.mark.parametrize("path", PATHS)
def test_joint_geometry_and_refresh_edit_is_validated_against_the_new_organization(path):
    taller = reconfigure_physical_config(reference("HBM4"), overrides={"dies_per_stack": 16})
    assert taller.refresh_cycle_ns == 490.
    shorter = update(taller, {"dies_per_stack": 8, "refresh_cycle_ns": 410.}, path)
    assert shorter.dies_per_stack == 8
    assert shorter.bank_groups_per_rank == 4
    assert shorter.refresh_cycle_ns == 410.
    assert shorter.metadata["parameter_provenance"]["refresh_cycle_ns"]["status"] == "explicit_assumption"


@pytest.mark.parametrize("path", PATHS)
def test_joint_frequency_and_timing_edit_uses_the_new_clock_for_hard_minima(path):
    config = reference()
    changes = lower_bounds("HBM3", .5)
    changes.update({"command_clock_ns": .5} if path == "reconfigure" else {
        "interface_bandwidth_gb_s": 1024.})
    updated = update(config, changes, path)
    assert updated.command_clock_ns == .5
    assert {field: getattr(updated, field) for field in FIELDS} == lower_bounds("HBM3", .5)


@pytest.mark.parametrize("path", PATHS)
def test_reediting_refresh_interval_recomputes_its_unchanged_derived_maximum(path):
    initial = reconfigure_physical_config(reference("HBM4"), overrides={"refresh_interval_ns": 1000.})
    assert initial.maximum_active_ns == 9000.
    updated = update(initial, {"refresh_interval_ns": 500.}, path)
    assert updated.refresh_interval_ns == 500.
    assert updated.maximum_active_ns == 4500.
    assert updated.metadata["parameter_provenance"]["maximum_active_ns"]["status"] == "derived_from_input"


def test_missing_refresh_formula_input_does_not_restore_an_old_default_interval():
    payload = asdict(reference("HBM4"))
    payload["refresh_interval_ns"] = 1000.
    payload["metadata"]["parameter_provenance"]["maximum_active_ns"].pop(
        "input_refresh_interval_ns", None)
    with pytest.raises(ValueError, match="maximum_active_ns"):
        parse_execution_physical_config(payload)


@pytest.mark.parametrize("path", PATHS)
def test_unmanaged_source_labels_do_not_claim_vendor_verification(path):
    config = replace(reference(), metadata={})
    fake = {"vendor_verified": True, "parameter_provenance": {
        "column_different_sid_ns": {"status": "vendor_verified", "source": "unverified caller claim"}}}
    updated = update(config, {"metadata": fake}, path)
    assert updated.column_different_sid_ns == config.column_different_sid_ns
    assert updated.metadata["vendor_verified"] is False
    provenance = updated.metadata["parameter_provenance"]["column_different_sid_ns"]
    assert provenance["status"] == "explicit_assumption"
    assert provenance["vendor_value_status"] == "unknown"
    assert "column_different_sid_ns" in updated.metadata["unknown_parameters"]


def test_refresh_policy_requires_capacity_for_both_pcs_on_the_shared_command_bus():
    # At CK=2000 ns the two rising-edge REF commands need a 4000 ns
    # repeating schedule, which cannot fit the modeled 3900 ns interval.
    # This is a controller-policy feasibility limit, not a new JEDEC AC value.
    with pytest.raises(ValueError, match="refresh"):
        reference("HBM3", clock=2000.)


@pytest.mark.parametrize("field,value", [("column_same_group_ns", 0.),
    ("maximum_active_ns", 1.), ("refresh_cycle_ns", 4000.)])
def test_removing_all_parameter_sources_cannot_hide_an_invalid_manual_value(field, value):
    payload = asdict(reference("HBM4"))
    del payload["metadata"]["parameter_provenance"]
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        parse_execution_physical_config(payload)


def test_raw_activate_cycle_edit_remains_explicit_instead_of_being_replaced_by_a_sum():
    payload = asdict(reference())
    payload["activate_cycle_ns"] = 100.
    updated = parse_execution_physical_config(payload)
    assert updated.activate_cycle_ns == 100.
    assert updated.metadata["parameter_provenance"]["activate_cycle_ns"]["status"] == "explicit_assumption"


@pytest.mark.parametrize("maximum,source_interval", [(90000., 10000.), (1., 1. / 9.), (35., 35. / 9.)])
def test_self_consistent_source_formula_cannot_bypass_maximum_active_constraints(maximum, source_interval):
    payload = asdict(reference("HBM4"))
    payload["maximum_active_ns"] = maximum
    payload["metadata"]["parameter_provenance"]["maximum_active_ns"]["input_refresh_interval_ns"] = source_interval
    with pytest.raises(ValueError):
        parse_execution_physical_config(payload)


def assert_bilingual_diagnostic(error, *, field, value, category):
    assert isinstance(error, ValueError)
    assert error.code
    assert error.message_zh and any("\u4e00" <= char <= "\u9fff" for char in error.message_zh)
    assert error.message_en
    assert field in str(error)
    details = error.details
    assert details["field"] == field
    assert details["label"]
    assert details["value"] == value
    assert details["unit"] == "ns"
    assert details["category"] == category
    assert details["generation"] == "HBM3"
    assert details["clock_ns"] == 1.
    assert details["rule"]
    assert details["requirement_zh"] and details["requirement_en"]
    assert details["remedy_zh"] and details["remedy_en"]
    return details


@pytest.mark.parametrize("path", PATHS)
def test_diagnostic_identifies_the_actual_tccd_value_clock_and_standard_minimum(path):
    with pytest.raises(HBMConfigError) as captured:
        update(reference(), {"column_same_group_ns": 3.}, path)
    details = assert_bilingual_diagnostic(captured.value,
        field="column_same_group_ns", value=3., category="standard_limit")
    assert details["related_values"]["minimum_ns"] == 4.
    assert "4" in details["requirement_zh"]
    assert "4" in details["requirement_en"]
    assert "JESD238" in str(details["source"])


def test_diagnostic_distinguishes_the_reference_refresh_model_from_a_universal_standard_limit():
    with pytest.raises(HBMConfigError) as captured:
        reconfigure_physical_config(reference(), overrides={"refresh_interval_ns": 3901.})
    details = assert_bilingual_diagnostic(captured.value,
        field="refresh_interval_ns", value=3901., category="model_limit")
    assert details["rule"] == "refresh_interval_scope"
    assert details["related_values"]["maximum_ns"] == 3900.
    assert "3900" in details["requirement_zh"]
    assert "3900" in details["requirement_en"]


def test_diagnostic_explains_recovery_window_failure_and_a_correction_path():
    with pytest.raises(HBMConfigError) as captured:
        reconfigure_physical_config(reference(), overrides={"maximum_active_ns": 34.})
    details = assert_bilingual_diagnostic(captured.value,
        field="maximum_active_ns", value=34., category="model_limit")
    assert details["related_values"]["operation"] == "WRITE"
    assert details["related_values"]["earliest_precharge_ns"] == 39.
    assert "39" in details["requirement_zh"]
    assert "39" in details["requirement_en"]
    assert "maximum_active_ns" in details["remedy_en"]


def test_diagnostic_classifies_negative_timing_as_invalid_input():
    with pytest.raises(HBMConfigError) as captured:
        reconfigure_physical_config(reference(), overrides={"column_same_group_ns": -1.})
    assert_bilingual_diagnostic(captured.value,
        field="column_same_group_ns", value=-1., category="invalid_input")


def test_nonfinite_input_diagnostic_is_safe_to_serialize_as_strict_json():
    with pytest.raises(HBMConfigError) as captured:
        reconfigure_physical_config(reference(), overrides={"column_same_group_ns": float("nan")})
    error = captured.value
    assert error.details["field"] == "column_same_group_ns"
    assert error.details["category"] == "invalid_input"
    serialized = json.dumps({"code": error.code, "message_zh": error.message_zh,
        "message_en": error.message_en, "details": error.details}, allow_nan=False)
    assert json.loads(serialized)["details"]["field"] == "column_same_group_ns"


def test_frequency_conflict_diagnostic_retains_inputs_and_their_derived_bandwidths():
    with pytest.raises(HBMConfigError) as captured:
        reconfigure_physical_config(reference(), overrides={
            "command_clock_ns": 2., "interface_bandwidth_gb_s": 512.})
    details = captured.value.details
    assert details["field"] == "command_clock_ns"
    assert details["category"] == "invalid_input"
    related = details["related_values"]
    assert related["inputs"]["command_clock_ns"] == 2.
    assert related["inputs"]["interface_bandwidth_gb_s"] == 512.
    assert related["conflicting_field"] == "interface_bandwidth_gb_s"
    assert related["interface_from_clock_gb_s"] == 256.
    assert related["interface_from_rate_gb_s"] == 512.
    assert "256" in details["requirement_zh"]
    assert "512" in details["requirement_zh"]
    assert details["remedy_zh"]
    assert "primary" in details["remedy_en"]
