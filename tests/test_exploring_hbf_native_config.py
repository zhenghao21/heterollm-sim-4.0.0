from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from exploring_hbf_native_config import build_native_comparison_scenario, gpu_reference
from exploring_hbf_native_pilot import build_pilot_scenario
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.planner import _kv_components
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.serving import compile_serving_plan
from heterollm_sim.topology import assert_valid_topology


def test_public_peaks_use_a_structural_native_profile_with_explicit_research_geometry():
    profile = gpu_reference("gpu3")
    assert profile.tensor_core.peak_tops("bf16") == pytest.approx(4000.)
    assert profile.elementwise_gops == pytest.approx(130000.)
    assert profile.attainable_efficiency == profile.occupancy == 1.
    assert profile.tensor_core.resource_id == "gpu3.tensor_core"


@pytest.mark.parametrize("hardware", ["HBM4", "CONV", "HBF", "CONV+", "HBF+"])
def test_all_paper_hardware_has_private_native_endpoints_and_valid_8gpu_topology(hardware):
    scenario = build_native_comparison_scenario("llama3_405b", hardware, "SHORT")
    assert_valid_topology(scenario.hardware)
    stores = scenario.placement.metadata["rank_weight_components"]
    assert scenario.placement.metadata["rank_kv_components"] == stores
    assert len(set(stores.values())) == 8
    assert scenario.simulation.mode == "detailed"
    assert "external_kv_ingress" in scenario.workload.metadata
    assert scenario.model.metadata["paper_side_backend_used"] is False
    for rank in scenario.placement.parallel.rank_mapping:
        weight = scenario.hardware.get_component(stores[str(rank.rank)])
        active = scenario.hardware.get_component(rank.memory_component_id)
        if hardware.endswith("+"):
            assert active.kind == "sram"
            assert require_physical_memory_config(active).capacity_bytes == 320_000_000
            assert not any(component.kind == "hbm" for component in scenario.hardware.components)
        else:
            assert active.kind == "hbm"
        assert weight.kind == ("hbm" if hardware == "HBM4" else "hbf")


def test_native_control_plane_shards_flash_weights_independently_of_sram_activation_owner():
    model = build_pilot_scenario("hbm_tp1").model
    scenario = build_native_comparison_scenario("small-functional", "HBF+", "SHORT", tp=2,
        model_override=model, prompt_override=16, output_override=4, warmup_override=1, measurement_override=2)
    decision = plan_runtime_placement(scenario)
    assert decision.fully_placed, [(item.item_id, item.reason) for item in decision.unplaced]
    scenario = decision.apply(scenario)
    shards = scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]
    assert shards
    for rows in shards.values():
        for shard in rows:
            assert shard["storage_component_id"] == f"hbf{shard['rank_id']}"
    plan = build_parallel_plan(scenario)
    for rank in plan.ranks:
        assert rank.memory_component_id == f"sram{rank.rank}"
        assert _kv_components(scenario, rank, rank.component_id)[0] == f"hbf{rank.rank}"


def test_native_request_sorting_keeps_all_initial_aged_seeds_before_fresh_requests():
    model = build_pilot_scenario("hbm_tp1").model
    s = build_native_comparison_scenario("small-functional", "HBM4", "SHORT", tp=2, batch_size=8,
        model_override=model, prompt_override=16, output_override=4)
    plan = compile_serving_plan(s)
    assert all(request.metadata["warmup_seed"] for request in plan.requests[:8])
    assert all(not request.metadata["warmup_seed"] for request in plan.requests[8:])


@pytest.mark.parametrize("tp", (1, 2))
def test_pipeline_targets_select_the_first_native_rank_of_the_actual_stage(tp):
    model = build_pilot_scenario("hbm_tp1").model
    scenario = build_native_comparison_scenario("small-functional", "HBF+", "SHORT", tp=tp, pp=2,
        model_override=model, prompt_override=16, output_override=4)
    targets = scenario.placement.metadata["control_plane"]["policy"]["options"]["operator_targets"]
    assert targets["embedding"] == "gpu0"
    assert targets["lm_head"] == f"gpu{tp}"
    assert all(target == "gpu0" for name, target in targets.items() if name.startswith("dense0."))
    assert all(target == f"gpu{tp}" for name, target in targets.items() if name.startswith("dense1."))
    decision = plan_runtime_placement(scenario)
    assert decision.fully_placed, [(item.item_id, item.reason) for item in decision.unplaced]
