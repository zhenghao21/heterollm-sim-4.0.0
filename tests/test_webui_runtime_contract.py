"""V4 report-to-frontend contract for runtime-owned, read-only placement."""
from dataclasses import replace

import pytest

from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
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
    assert "memory_tiers" not in result.scenario.placement.metadata
    assert runtime["memory_tiers"] == {}
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


def test_report_memory_tiers_come_from_solved_mapping_without_mutating_authoring():
    source = build_reference_scenario()
    source = replace(source, placement=replace(
        source.placement, kv_policy=replace(
            source.placement.kv_policy, offload_component=None, offload_ratio=0.0)),
        workload=replace(source.workload, mtp=None))
    source = replace(source, workload=replace(
        source.workload, scheduler=replace(source.workload.scheduler, mode="static")))
    original = scenario_to_payload(source)
    assert "memory_tiers" not in source.placement.metadata
    assert source.placement.tensor_to_component == {}

    mapping = plan_runtime_placement(
        source, PlacementPolicy(kv_layer_targets={"moe1": "hbm2"}))
    assert mapping.fully_placed, mapping.unplaced
    result = run_scenario(mapping.apply(source))
    report = report_dict(result)
    runtime = report["runtime_placement"]
    expected = {"kv_layer_components": {"moe1": "hbm2"}}
    assert runtime["schema_version"] == "runtime-placement/v1"
    assert runtime["read_only"] is True
    assert runtime["memory_tiers"] == expected
    assert runtime["memory_tier_details"]["kv_cache"]["cache_component"] == "hbm0"
    assert runtime["memory_tier_details"]["kv_cache"]["component_bytes_per_page"]["hbm2"] > 0
    assert runtime["memory_tier_details"]["kv_cache"]["capacity_bytes"] > 0
    assert runtime["memory_tiers"] == to_primitive(
        result.scenario.placement.metadata["memory_tiers"])
    assert result.scenario.placement.tensor_to_component["moe1.kv_cache"] == "hbm2"
    assert result.scenario.placement.tensor_to_component["kv_cache"] == "hbm0"
    assert report["summary"]["makespan_ns"] > 0
    assert scenario_to_payload(source) == original

    # The serialized nested map must not alias the runtime-owned mapping either.
    runtime["memory_tiers"]["kv_layer_components"]["moe1"] = "hbm0"
    assert result.scenario.placement.metadata["memory_tiers"] == expected
    assert scenario_to_payload(source) == original
