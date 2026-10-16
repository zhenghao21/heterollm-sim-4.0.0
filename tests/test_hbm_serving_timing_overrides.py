"""Single timing experiments reach the same serving hardware as full packs."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_serving_study import build_study_scenario
from heterollm_sim.physical_contract import parse_execution_physical_config


def config(scenario, component="hbm0"):
    return parse_execution_physical_config(
        scenario.hardware.get_component(component).metadata["physical_memory_config"])


def test_serving_timing_override_preserves_inventory_and_recomputes_dependencies():
    args = dict(prompt=4, output=2, hbm_parameter_profile="reference")
    baseline = build_study_scenario("B_all_kv_hbf", **args)
    changed = build_study_scenario("B_all_kv_hbf", **args,
        hbm_timing_overrides={"minimum_active_ns": 28.125})
    before, after = config(baseline), config(changed)
    assert (after.minimum_active_ns, after.precharge_to_activate_ns, after.activate_cycle_ns) == (28.125, 15., 43.125)
    for field in ("capacity_bytes", "stacks", "command_clock_ns", "read_bandwidth_gb_s",
                  "write_bandwidth_gb_s", "address_mapping", "scheduler", "queue_scope"):
        assert getattr(after, field) == getattr(before, field)
    assert config(changed, "hbf0") == config(baseline, "hbf0")
    assert after.metadata["hbm_parameter_profile"]["overridden_fields"] == ["minimum_active_ns"]
    assert after.metadata["parameter_provenance"]["minimum_active_ns"]["source"] == "physical_config_override"
    assert after.metadata["hbm_parameter_profile"]["vendor_calibrated"] is False


@pytest.mark.parametrize("overrides", [{"command_clock_ns": .5}, {"capacity_bytes": 64}, "ramulator2"])
def test_serving_timing_override_rejects_non_timing_changes(overrides):
    with pytest.raises(ValueError, match="HBM timing"):
        build_study_scenario("A_hbm_only", hbm_timing_overrides=overrides)
