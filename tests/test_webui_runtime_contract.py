"""V4 report-to-frontend contract for runtime-owned, read-only placement."""
from dataclasses import replace

import pytest

from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import scenario_to_payload, validation_payload


@pytest.mark.parametrize("mode", ["static", "continuous"])
def test_report_exposes_actual_runtime_placement_without_mutating_authoring(mode):
    source = build_reference_scenario()
    source = replace(source, workload=replace(
        source.workload, scheduler=replace(source.workload.scheduler, mode=mode)))
    original = scenario_to_payload(source)
    result = run_scenario(source)
    report = report_dict(result)
    runtime = report["runtime_placement"]
    assert runtime["schema_version"] == "runtime-placement/v1"
    assert runtime["read_only"] is True
    assert runtime["parallel"] == to_primitive(result.scenario.placement.parallel)
    assert runtime["control_plane"] == to_primitive(
        result.scenario.placement.metadata["control_plane"])
    decision = runtime["control_plane"]["decision"]
    assert decision["fully_placed"] is True
    assert decision["operator_execution_targets"]
    assert decision["rank_weight_shards"]
    assert runtime["control_plane"]["evidence"]["input_fingerprint"]
    assert report["summary"]["makespan_ns"] > 0
    assert report["manifest"]["evidence"] == "analytical"
    assert scenario_to_payload(source) == original
    assert original["placement"]["op_to_component"] == {}
    assert validation_payload(original)["valid"] is True
