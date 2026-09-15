"""Low-cost regressions for repeatability statistics and timestamp evidence."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("native_repeatability_audit", Path(__file__).resolve().parents[1] / "tools/audit_native_repeatability.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def response(output=3):
    ts = [2000, 4000, 6000][:output]
    return {"tokens_predicted": output, "id_slot": 0, "timings": {
        "engine_clock": "ggml_time_us_monotonic", "engine_timepoints_scope": "non_speculative_generated_tokens",
        "engine_timepoints_complete": True, "engine_request_begin_us": 1000,
        "engine_prompt_last_us": ts[0], "engine_last_token_us": ts[-1],
        "engine_token_times_us": ts, "predicted_n": output, "prompt_ms": 1.0,
        "predicted_ms": (ts[-1] - ts[0]) / 1000, "cache_n": 0, "prompt_n": 5}}


def test_cv_below_five_does_not_pass_plus_minus_five():
    result = audit.deviation_summary([100, 100, 100, 106])
    assert result["cv_pct"] < 5
    assert result["max_abs_deviation_pct"] == pytest.approx(6)
    assert result["observed_within_band"] is False
    assert result["fraction_within_5pct"] == 0.75


def test_slow_run_is_preserved_in_signed_deviations_and_p90():
    result = audit.deviation_summary([90, 100, 130])
    assert result["values_ms"] == [90, 100, 130]
    assert result["signed_deviation_pct"] == pytest.approx([-10, 0, 30])
    assert result["p90_abs_deviation_pct"] == pytest.approx(26)


def test_single_repeat_is_not_a_repeatability_pass():
    result = audit.deviation_summary([100])
    assert result["observed_within_band"] is False
    assert result["statistical_guarantee"] is False


def test_timing_formula_matches_first_last_and_n_minus_one():
    value, violations = audit.timing_record(response())
    assert not violations
    assert {key: value[key] for key in audit.METRICS} == {"ttft": 1, "tpot": 2, "e2e": 5}


def test_single_token_tpot_is_not_applicable():
    value, violations = audit.timing_record(response(1))
    assert value["tpot"] is None
    assert not violations


def test_counter_or_timestamp_mutation_is_detected():
    item = response()
    item["timings"]["predicted_ms"] = 20
    _, violations = audit.timing_record(item)
    assert "counter_predicted_ms_mismatch" in violations
    item = response()
    item["timings"]["engine_token_times_us"] = [2000, 6000, 4000]
    _, violations = audit.timing_record(item)
    assert "nonmonotonic_timepoints" in violations


def test_missing_engine_tokens_never_falls_back_to_client_wall():
    item = response()
    del item["timings"]["engine_token_times_us"]
    item["client_e2e_ms"] = 5
    value, violations = audit.timing_record(item)
    assert not value
    assert "missing_or_noninteger_token_timestamps" in violations


def test_late_engine_start_identifies_stagger_without_causal_claim():
    first, _ = audit.timing_record(response())
    later = copy.deepcopy(first)
    later.update(slot_id=1, begin_us=3000, first_us=5000, last_us=7000)
    cohort = audit.cohort_description([first, later])
    assert cohort["split_before_all_engine_starts"] is True
    assert cohort["requests_started_after_first_token"] == 1
    assert cohort["engine_batch_makespan_ms"] == 6


def test_slot_matching_keeps_request_order_and_all_samples():
    cells = []
    for repeat, values in enumerate(([1, 2], [1.01, 2.02], [1.20, 2.00]), start=1):
        rows = [{"slot_id": slot, "engine_start_rank": 1-slot, "request_index": slot,
                 "ttft": value, "tpot": value, "e2e": value} for slot, value in enumerate(values)]
        cells.append({"cell_id": str(repeat), "repeat": repeat, "records": rows})
    result = audit.matching_diagnostics(cells, "slot_id")
    assert result["pooled"]["ttft"]["n"] == 6
    assert result["pooled"]["ttft"]["max_abs_deviation_pct"] > 18
    assert result["positions"][0]["metrics"]["ttft"]["n"] == 3


def test_configuration_discrepancy_is_detected():
    command = ["llama-server", "-c", "2048", "-np", "4", "-ngl", "-1", "-b", "64", "-ub", "64", "-t", "16", "-tb", "16", "-ctk", "f16", "-ctv", "f16"]
    config = {"ctx": 2048, "parallel": 2, "gpu_layers": -1, "batch": 64, "ubatch": 64, "threads": 16, "threads_batch": 16, "kv_type_k": "f16", "kv_type_v": "f16"}
    assert audit.command_config_issues(command, config) == ["command_config_mismatch:parallel"]
