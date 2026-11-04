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


@pytest.mark.parametrize("case", ["paper_hbf", "paper_conv"])
def test_paper_page_service_executes_real_native_partial_kv_writes(case):
    row = execute_control((case, 1, "qwen2_5-0_5b", 16, 3))
    assert row["status"] == "completed"
    assert row["physical_configs"]["hbf0"]["partial_page_policy"] == "host_merge"
    assert row["storage_traffic"]["physical_write_bytes"] > 0
    assert row["resolved_placement"]["rank_active_memory"] == ["hbm0"]
    assert row["paper_tps_reproduced"] is False


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_is_not_silently_coerced(value):
    with pytest.raises(ValueError, match="batch_size"):
        build_control_scenario("scan_hbm", batch_size=value)
