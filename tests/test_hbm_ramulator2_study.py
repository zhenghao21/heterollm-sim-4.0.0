"""Comparison cases cover timing dependencies and verify actual execution inputs."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hbm_ramulator2_study as study
from heterollm_sim.memory_mapping import map_dram_address


def test_each_different_timing_is_covered_without_changing_inventory_or_controller():
    variants = study.variants()
    assert len(variants) == 18
    assert len(study.planned_serving_points()) == 48
    assert len(study.workloads()) == 14
    assert len(set(study.planned_serving_points())) == 48
    assert all(point[2] in study.LARGE_MODELS for point in study.planned_serving_points()[:8])
    items = study.parameter_comparison()
    assert sum(item["differs"] for item in items) == 17
    for item in items:
        assert not item["differs"] or item["covered_by_variants"]
    assert "single_activate_cycle_ns" not in variants
    assert "single_read_to_write_command_ns" in variants
    assert all(study.variant_descriptor(variant)["fixed_configuration_matches_reference"] for variant in variants)


def test_large_stage_is_the_complete_eight_case_70b_and_235b_comparison():
    points = study.planned_serving_points(stage="large")
    assert set(points) == {(variant, architecture, model, 128, 1, 4)
                          for model in ("llama3_1-70b", "qwen3-235b-a22b")
                          for variant in ("reference", "ramulator2")
                          for architecture in ("A_hbm_only", "B_all_kv_hbf")}
    assert set(points) <= set(study.planned_serving_points(stage="full-service"))
    assert study.planned_serving_points(["single_refresh_cycle_ns"], "large") == []
    filtered = study.planned_serving_points(["reference", "ramulator2"], "large", ["llama3_1-70b"])
    assert len(filtered) == 4
    assert all(point[2] == "llama3_1-70b" for point in filtered)


def test_large_model_aggregation_includes_latency_and_complete_request_throughput(tmp_path):
    for variant, architecture, model, prompt, batch, output in study.planned_serving_points(stage="large"):
        scale = (1.1 if variant == "ramulator2" else 1.) * (2 if architecture == "B_all_kv_hbf" else 1)
        row = {"id": model + variant + architecture, "variant": variant, "placement": architecture,
            "parameters": {"model_id": model, "prompt": prompt, "batch": batch, "output": output,
                "hbm_parameter_profile": variant, "hbm_timing_overrides": None, "hbm_capacity_bytes": 512 * study.GIB},
            "point_result_version": study.POINT_RESULT_VERSION,
            "timing_comparison": study.variant_descriptor(variant),
            "core_execution_audit": {"owners": [], "executed_physical_configurations": {
                "hbm": {"physical_config": asdict(study.configuration(variant))}}},
            "decode_core_checks": {"executed": True}, "placement_verification": {"correct": True},
            "decode_measurement": {"checks": {"executed": True}}, "wall_seconds": 1.,
            "tpot_ns": {"p50": 1e6 * scale}, "ttft_ns": {"p50": 4e6 * scale},
            "makespan_ns": 7e6 * scale, "node_tps": 4e9 / (7e6 * scale),
            "model_structure": {"layers": 80 if model == "llama3_1-70b" else 94, "weight_bytes": 140e9}}
        study.save(tmp_path / "serving" / (row["id"] + ".json"), row)
    result = study.aggregate(tmp_path)
    assert result["status"] == "incomplete"  # Earlier model/micro points are still required.
    assert result["checks"]["all_serving_cases_use_current_actual_inputs"]
    assert len(result["serving_comparisons"]) == 4
    compared = next(item for item in result["serving_comparisons"] if item["variant"] == "ramulator2"
                    and item["model"] == "qwen3-235b-a22b")
    assert compared["a_tpot_change_percent"] == pytest.approx(10.)
    assert compared["a_ttft_ns_change_percent"] == pytest.approx(10.)
    assert compared["a_node_tps_change_percent"] == pytest.approx(-100 / 11)
    assert compared["b_over_a"] == 2.
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "llama3_1-70b | 80" in report
    assert "qwen3-235b-a22b | 94" in report


@pytest.mark.parametrize("field,dependent", [
    ("minimum_active_ns", "activate_cycle_ns"),
    ("precharge_to_activate_ns", "activate_cycle_ns"),
    ("read_latency_ns", "read_to_write_command_ns"),
    ("write_latency_ns", "read_to_write_command_ns"),
])
def test_single_input_exposes_its_real_dependent_change(field, dependent):
    variant = "single_" + field
    descriptor = study.variant_descriptor(variant)
    assert descriptor["authored_fields"] == [field]
    assert descriptor["dependent_changed_fields"] == [dependent]
    assert study.configuration(variant).metadata["ramulator2_comparison"]["copied_parameter_sources"][field]["source"]


def test_direct_rtw_is_distinct_from_rl_wl_formula_dependency():
    direct = study.configuration("single_read_to_write_command_ns")
    reference = study.base_configuration()
    assert direct.read_to_write_command_ns == 19.375
    assert direct.read_latency_ns == reference.read_latency_ns
    assert direct.write_latency_ns == reference.write_latency_ns
    assert study.variant_descriptor("single_read_to_write_command_ns")["dependent_changed_fields"] == []
    rl_case = study.configuration("single_read_latency_ns")
    assert rl_case.read_latency_ns != direct.read_latency_ns
    assert rl_case.read_to_write_command_ns == 10.625
    assert study.comparison_source("single_read_to_write_command_ns").read_latency_ns == 20.
    assert study.comparison_source("single_read_to_write_command_ns").write_latency_ns == 5.
    assert rl_case.metadata["parameter_provenance"]["read_to_write_command_ns"]["rule"] == "jedec_rtw_with_assumed_phy"
    assert direct.metadata["parameter_provenance"]["read_to_write_command_ns"]["rule"] == "explicit_profile_override"


def test_targeted_probe_mapping_includes_different_sids_and_groups():
    config = study.base_configuration()
    requests = study.workloads()
    mapped = [map_dram_address(config, request.address) for request in requests["different_sid_reads"]]
    assert len({(item.stack, item.channel, item.pseudo_channel) for item in mapped}) == 1
    assert {(item.bank_group * config.banks_per_group + item.bank) // 16 for item in mapped} == {0, 1}
    groups = [map_dram_address(config, request.address).bank_group for request in requests["different_group_rw"]]
    assert set(groups) == {0, 1}


def test_actual_micro_records_show_rtp_effect_and_reject_stale_inputs():
    requests = study.workloads()["late_read_conflict"]
    baseline = study.micro("reference", "late_read_conflict", requests)
    changed = study.micro("single_read_to_precharge_ns", "late_read_conflict", requests)
    assert changed["request_completion_ns"][-1] > baseline["request_completion_ns"][-1]
    assert any(event["name"] == "PRECHARGE" for event in changed["command_events"])
    assert study.matches_current_inputs(changed)
    stale = deepcopy(changed)
    stale["physical_configuration"]["read_to_precharge_ns"] = 999.
    assert not study.matches_current_inputs(stale)
    stale = deepcopy(changed)
    stale["point_result_version"] -= 1
    assert not study.matches_current_inputs(stale)


def test_single_large_aggregate_keeps_compiled_execution_enabled():
    pytest.importorskip("numba")
    row = study.micro("reference", "aggregate", [study.AccessRequest("stream", "read", 0, 32768)])
    assert row["compiled_requests"] == 1
    assert row["command_events"] == []


def test_serving_checks_actual_core_configuration_instead_of_only_profile_name():
    variant = "single_refresh_cycle_ns"
    config = asdict(study.configuration(variant))
    row = {"variant": variant, "point_result_version": study.POINT_RESULT_VERSION,
        "timing_comparison": study.variant_descriptor(variant),
        "parameters": {"hbm_parameter_profile": "reference", "hbm_timing_overrides": {"refresh_cycle_ns": 550.},
                       "hbm_capacity_bytes": 512 * study.GIB},
        "core_execution_audit": {"executed_physical_configurations": {
            "hbm0.hbm_fabric": {"core": "DramCore", "physical_config": config}}}}
    assert study.matches_current_inputs(row, serving_case=True)
    row["core_execution_audit"]["executed_physical_configurations"]["hbm0.hbm_fabric"]["physical_config"]["refresh_cycle_ns"] = 500.
    assert not study.matches_current_inputs(row, serving_case=True)


def test_partial_matrix_cannot_be_reported_as_passed(tmp_path):
    row = study.micro("reference", "cold_read", study.workloads()["cold_read"])
    study.save(tmp_path / "micro/reference-cold_read.json", row)
    result = study.aggregate(tmp_path)
    assert result["status"] == "incomplete"
    assert result["planned_micro_points"] == 252
    assert result["planned_serving_points"] == 48
    assert result["checks"]["all_microbenchmarks_use_current_actual_inputs"]
    assert not result["checks"]["all_planned_microbenchmarks_executed"]
    row["physical_configuration"]["refresh_cycle_ns"] = 123.
    study.save(tmp_path / "micro/reference-cold_read.json", row)
    assert not study.aggregate(tmp_path)["checks"]["all_microbenchmarks_use_current_actual_inputs"]
