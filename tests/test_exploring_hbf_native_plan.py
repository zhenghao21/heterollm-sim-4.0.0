"""Native plan exports actual configuration, never invented formal metrics."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from exploring_hbf_native_plan import build_native_plan, export_native_plan
from heterollm_sim.config import model_from_dict
from heterollm_sim.physical_contract import parse_execution_physical_config


def test_native_matrix_is_separate_and_export_does_not_execute_prepared_cases():
    plan = build_native_plan()
    assert len(plan["cases"]) == len({item["case_id"] for item in plan["cases"]}) == 30
    assert plan["runtime_status_file"] == "status.json" and not plan["automatic_execution"]
    assert "formal_completed_count" not in plan  # Runtime owns live counts.
    assert all(case["metrics"] is None and case["execution_enabled"]
               and case["status"] == "prepared" for case in plan["cases"])
    for case in plan["cases"]:
        missing = case["required_native_extensions"]
        assert "external_prefill_admission" in missing
        assert "decode_measurement_boundary" in missing
        assert ("logic_die_sram" in missing) == case["hardware"].endswith("+")
        assert ("maverick_architecture" in missing) == (case["model"] == "llama4_maverick")
    assert plan["execution_policy"]["cancelled_5k_10k_restored"] is False
    assert plan["execution_policy"]["historical_results_reused_as_native"] is False
    assert [phase["case_count"] for phase in plan["phases"] if "case_count" in phase] == [9, 6, 15]
    for phase in plan["phases"][1:]:
        assert {"paper_gpu_profile", "external_prefill_admission", "decode_measurement_boundary"} <= set(phase["start_when"])
    assert "logic_die_sram" in plan["phases"][-1]["start_when"]
    search = plan["deployment_search"]
    assert search["first_control"] == {"tp_degree": 8, "pp_degree": 1, "ep_degree": 1, "dp_replicas": 1}
    assert search["control_batch_candidates"] == [1, 8] and "formal_batch_candidates" not in search
    assert "--search" in search["batch_search"] and "不外推" in search["batch_search"]
    for key in ("layout", "ep_scope", "dp_scope", "selection"):
        assert search[key]
    assert "单GPU吞吐" in search["dp_scope"] and "独立子进程" in search["implementation"]
    # Cases are hardware/workload points; the layout is searched, never fixed per case.
    assert all(not {"tp", "pp", "ep", "dp"} & case.keys() for case in plan["cases"])


def test_exported_inputs_roundtrip_through_native_parsers_with_real_sram(tmp_path):
    result = export_native_plan(tmp_path / "native")
    assert result["experiments_started_by_export"] == 0
    inputs = json.loads((tmp_path / "native/native_inputs.json").read_text(encoding="utf-8"))
    model = model_from_dict(inputs["models"]["llama3_405b"])
    assert model.total_declared_weight_bytes == 811_698_487_296
    maverick = model_from_dict(inputs["models"]["llama4_maverick"])
    assert maverick.total_declared_weight_bytes == 801_422_704_640
    for identity, hardware in inputs["hardware"].items():
        sram = hardware["native_intermediate_sram_config"]
        assert (sram is not None) == identity.endswith("+")
        if sram:
            parsed = parse_execution_physical_config(sram)
            assert parsed.kind == "SRAM" and parsed.capacity_bytes == 320_000_000
            assert parsed.ports == 8
        hbm = hardware["native_hbm_config"]
        nand = hardware["native_nand_config"]
        assert (hbm is None) == identity.endswith("+")
        if hbm:
            parsed = parse_execution_physical_config(hbm)
            assert parsed.capacity_bytes == hardware["per_gpu_table"]["hbm_capacity_bytes"]
        if nand:
            parsed = parse_execution_physical_config(nand)
            assert parsed.read_cache_pages_per_bank == 2
            assert parsed.metadata["execution_backend"] == "native_nand_core"
            assert parsed.capacity_bytes == hardware["per_gpu_table"]["hbf_capacity_bytes"]
    assert inputs["hardware"]["HBF+"]["logic_sram_status"] == "native_pooled_sram"


def test_retired_bypass_command_cannot_start_a_new_matrix(monkeypatch):
    import reproduce_exploring_hbf as historical
    monkeypatch.setattr(historical, "run_case", lambda *args: pytest.fail("retired runner executed"))
    with pytest.raises(SystemExit, match="入口已停用"):
        historical.main()


def test_historical_report_identifies_its_backend(tmp_path):
    import exploring_hbf_reproduction_report as historical
    text, _ = historical._markdown(tmp_path, {"cases": []}, [], [], [], historical.eight_gpu_comparison([]))
    assert text.startswith("# Exploring HBF历史旁路模型结果")
    assert "没有经过原生" in text
    assert "不能将其称为原生仿真器" in text
