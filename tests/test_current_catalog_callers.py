"""Architecture scan uses current payload fields and concrete parallel ranks."""
from dataclasses import replace
import json

from heterollm_sim.architecture_scan import (
    build_batched_gemm_candidates, scan_architecture_candidates,
)
from heterollm_sim.ir import RankMappingSpec
from heterollm_sim.reference import build_reference_scenario


def _current_payload(result):
    # These are the fields consumed by the architecture scan browser view.
    assert "statistics" not in result
    assert "backend" not in result
    assert result["backend_requested"] == "numpy"
    assert result["counts"]["top_results"] == len(result["top_results"])
    assert result["counts"]["ranks"] == len(result["rank_placement"])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_single_rank_scan_keeps_real_rank_and_current_result_fields():
    scenario = build_reference_scenario()
    result = scan_architecture_candidates(scenario, backend="numpy", top_n=3)
    _current_payload(result)
    assert result["backend_used"] == "numpy"
    assert result["counts"]["candidates"] == len(build_batched_gemm_candidates(scenario)) > 3
    rank, = result["rank_placement"]
    assert rank["rank"] == 0
    assert rank["component_id"] == "gpu0"
    for candidate in result["top_results"]:
        assert candidate["logical_rank"] == candidate["tp_rank"] == candidate["pp_rank"] == candidate["ep_rank"] == 0
        assert candidate["rank_component_id"] == "gpu0"
        assert candidate["rank_memory_component_id"] == "hbm0"
        assert candidate["rank_cim_component_id"] == "cim0"
        assert "|rank0000|" in candidate["candidate_id"]


def test_multiple_logical_ranks_keep_shards_and_optional_cim_binding():
    scenario = build_reference_scenario()
    parallel = replace(scenario.placement.parallel, tp_degree=2, rank_mapping=tuple(
        RankMappingSpec(rank=index, component_id="gpu0", tp_rank=index, pp_rank=0,
                        ep_rank=0, memory_component_id="hbm0") for index in range(2)))
    scenario = replace(scenario, placement=replace(scenario.placement, parallel=parallel))
    result = scan_architecture_candidates(scenario, backend="numpy", top_n=100)
    _current_payload(result)
    candidates = build_batched_gemm_candidates(scenario)
    assert result["counts"]["candidates"] == len(candidates) > 0
    assert {candidate.logical_rank for candidate in candidates} == {0, 1}
    assert {(row["logical_rank"], row["tp_rank"]) for row in result["top_results"]} == {(0, 0), (1, 1)}
    assert all(row["rank_cim_component_id"] is None for row in result["top_results"])


def test_zero_candidate_scan_still_reports_the_current_parallel_plan():
    scenario = build_reference_scenario()
    components = tuple(
        replace(component, peak_ops_per_s=0) if component.normalized_kind == "gpu"
        else replace(component, capacity_bytes=1) if component.component_id == "cim0"
        else component for component in scenario.hardware.components)
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=components))
    result = scan_architecture_candidates(scenario, backend="numpy")
    _current_payload(result)
    assert result["backend_used"] == "not_run"
    assert result["counts"]["candidates"] == 0
    assert result["top_results"] == []
    assert [rank["rank"] for rank in result["rank_placement"]] == [0]
