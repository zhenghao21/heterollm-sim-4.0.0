from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exploring_hbf_native_pilot import build_pilot_scenario, run_pilot
from heterollm_sim.ir import model_graph_execution_layers
from heterollm_sim.parallel import build_parallel_plan


@pytest.mark.parametrize("case", ("hbm_tp1", "hbf_kv_tp1", "hbm_tp2", "hbf_weights_kv_tp1"))
def test_native_pilot_builds_real_multirequest_bf16_scenario(case):
    scenario = build_pilot_scenario(case)
    assert scenario.simulation.mode == "detailed"
    assert scenario.workload.scheduler.mode == "continuous"
    assert len(scenario.workload.requests) == 8
    assert scenario.workload.scheduler.max_num_seqs == 2
    assert all((request.prompt_tokens, request.output_tokens) == (32, 4)
               for request in scenario.workload.requests)
    assert len(model_graph_execution_layers(scenario.model.graph)) == 2
    assert all(layer.dtype == "bf16" for layer in model_graph_execution_layers(scenario.model.graph))
    assert scenario.placement.kv_policy.dtype == "bf16"
    assert scenario.placement.kv_policy.cache_component == ("hbf0" if case.startswith("hbf_") else "hbm0")


def test_tp2_has_distinct_memory_owners_and_direct_gpu_links():
    scenario = build_pilot_scenario("hbm_tp2")
    plan = build_parallel_plan(scenario)
    assert [(rank.component_id, rank.memory_component_id) for rank in plan.ranks] == [
        ("gpu0", "hbm0"), ("gpu1", "hbm1")]
    links = {link.link_id: link for link in scenario.hardware.links}
    assert links["gpu-hbm0"].source_component == "gpu0"
    assert links["gpu-hbm1"].source_component == "gpu1"
    assert set(scenario.runtime_profile.gpu_controllers) == {"gpu0", "gpu1"}
    assert {links["pilot-peer"].source_component, links["pilot-peer"].target_component} == {"gpu0", "gpu1"}


def test_native_pilot_real_execution_admits_replacements_and_records_physical_cores(tmp_path):
    result = run_pilot("hbm_tp1", tmp_path)
    assert result["status"] == "completed"
    assert result["completed_requests"] == 8
    assert result["completed_visible_output_tokens"] == 32
    assert result["admitted_after_first_completion"] == 6
    assert result["batch_size_distribution"] == {2: 16}
    assert result["live_physical_cores"]["valid_for_committed_byte_totals"]
    assert result["dram_traffic"]["physical_bytes"] > 0
    assert result["nand_traffic"]["task_count"] == 0
    assert all(owner["core"] != "NandCore" for owner in result["live_physical_cores"]["owners"])
    assert result["mean_tpot_ns"] > 0


def test_native_tp2_executes_collectives_and_both_private_hbm_owners(tmp_path):
    result = run_pilot("hbm_tp2", tmp_path)
    assert result["completed_requests"] == 8
    assert result["admitted_after_first_completion"] == 6
    assert result["native_collective_graph"]["distinct_all_reduce_operations"] == 64
    assert result["native_collective_graph"]["all_reduce_peer_link_tasks"] > 0
    assert result["executed_collective_category_service_ns"] > 0
    assert all(value > 0 for value in result["executed_peer_link_service_ns"].values())
    owners = {owner["physical_owner"]: owner for owner in result["live_physical_cores"]["owners"]}
    assert owners["hbm0.hbm_fabric"]["physical_read_bytes"] > 0
    assert owners["hbm1.hbm_fabric"]["physical_read_bytes"] > 0


def test_native_mixed_placement_executes_hbf_weights_and_kv(tmp_path):
    result = run_pilot("hbf_weights_kv_tp1", tmp_path)
    assert result["completed_requests"] == 8
    assert set(result["resolved_placement"]["weight_tensor_owners"].values()) == {"hbf0"}
    assert result["resolved_placement"]["kv_owner"] == "hbf0"
    assert result["resolved_placement"]["rank_active_memory"] == ["hbm0"]
    assert result["nand_traffic"]["physical_read_bytes"] > 0
    assert result["nand_traffic"]["physical_write_bytes"] > 0
