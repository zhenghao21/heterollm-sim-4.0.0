"""The comparison changes placements explicitly and never impersonates the paper."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from exploring_hbf_compare import build_control_scenario, execute_control
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.planner import _direct_device_memory


def test_intermediate_placement_retains_the_same_flash_and_direct_weight_path():
    flash = build_control_scenario("scan_hbf", prompt_tokens=16, output_tokens=3)
    hybrid = build_control_scenario("scan_hbf_hbm_intermediates", prompt_tokens=16, output_tokens=3)
    assert asdict(require_physical_memory_config(flash.hardware.get_component("hbf0"))) == asdict(
        require_physical_memory_config(hybrid.hardware.get_component("hbf0")))
    assert flash.weights_resident and hybrid.weights_resident
    assert hybrid.placement.kv_policy == flash.placement.kv_policy
    assert hybrid.placement.metadata["control_plane"]["policy"]["options"]["weight_tensor_targets"] == (
        flash.placement.metadata["control_plane"]["policy"]["options"]["weight_tensor_targets"])
    assert _direct_device_memory(hybrid, "hbf0", "gpu0")
    assert hybrid.placement.parallel.rank_mapping[0].memory_component_id == "hbm0"
    assert flash.placement.parallel.rank_mapping[0].memory_component_id == "hbf0"


@pytest.mark.parametrize("case", ["paper_hbm4", "paper_hbf", "paper_conv"])
def test_paper_memory_controls_keep_compute_and_authored_short_request(case):
    scenario = build_control_scenario(case, prompt_tokens=16, output_tokens=3, batch_size=4)
    base = build_control_scenario("scan_hbm", prompt_tokens=16, output_tokens=3, batch_size=4)
    assert scenario.resolve_component_profile("gpu0") == base.resolve_component_profile("gpu0")
    assert scenario.model == base.model
    assert len(scenario.workload.requests) == 4
    assert scenario.workload.scheduler.slo_tpot_ns == 100e6
    assert scenario.simulation.mode == "detailed"
    if case != "paper_hbm4":
        assert require_physical_memory_config(scenario.hardware.get_component("hbf0")).protocol_mode == "generic_nand"
        assert _direct_device_memory(scenario, "hbf0", "gpu0")
        assert scenario.weights_resident


def test_real_mixed_memory_request_has_flash_weights_and_kv_but_hbm_workspace():
    row = execute_control(("scan_hbf_hbm_intermediates", 1, "qwen2_5-0_5b", 16, 3))
    assert row["resolved_placement"] == {
        "tensor_owners": ["hbf0"], "kv_owner": "hbf0", "rank_active_memory": ["hbm0"]}
    assert row["storage_traffic"]["logical_read_bytes"] > 0
    assert row["storage_traffic"]["logical_write_bytes"] > 0
    assert row["dram_traffic"]
    assert len(row["request_tpot_ns"]) == 1
    window = row["request_token_windows"]["control-0"]
    assert row["tpot_ns_mean"] == (window["last_token_ns"] - window["first_token_ns"]) / 2
    assert row["decode_window_tps"] == pytest.approx(1e9 / row["tpot_ns_mean"])
    assert row["paper_tps_reproduced"] is False


@pytest.mark.parametrize("case", ["paper_hbf", "paper_conv"])
def test_paper_page_service_executes_real_native_partial_kv_writes(case):
    row = execute_control((case, 1, "qwen2_5-0_5b", 16, 3))
    assert row["status"] == "completed"
    assert row["physical_configs"]["hbf0"]["partial_page_policy"] == "read_modify_write"
    assert row["storage_traffic"]["physical_write_bytes"] > 0
    assert row["resolved_placement"]["rank_active_memory"] == ["hbm0"]
    assert row["paper_tps_reproduced"] is False


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_is_not_silently_coerced(value):
    with pytest.raises(ValueError, match="batch_size"):
        build_control_scenario("scan_hbm", batch_size=value)
