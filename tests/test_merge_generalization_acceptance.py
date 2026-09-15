import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('merge_acceptance', Path(__file__).resolve().parents[1] / 'tools/merge_generalization_acceptance.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def record(native, simulator):
    return {'status': 'measured', 'native_ms': native, 'simulator_ms': simulator}


def test_group_statistics_use_scenario_medians_without_scale_weighting():
    scenarios = {'small': {'metrics': M.aggregate([{'metrics': {m: record(1, 1.2) for m in M.METRICS}}] * 3, 3)},
                 'large': {'metrics': M.aggregate([{'metrics': {m: record(1000, 1000) for m in M.METRICS}}] * 3, 3)}}
    groups = {'model': {'expected': list(range(6)), 'observed': [], 'eligible': [], 'scenarios': set(scenarios)}}
    result = M._group_stats(groups, 3, scenarios)['model']['metrics']['ttft_ms']
    assert abs(result['median_of_repeats_abs_pct'] - 10) < 1e-10
    assert abs(result['worst_abs_pct'] - 20) < 1e-10


def test_missing_repeats_do_not_make_not_applicable_complete():
    result = M._metric_summary([{'status': 'not_applicable'}], 3)
    assert result['status'] == 'evidence_insufficient'


def test_invalid_metrics_do_not_enter_coverage_or_groups(tmp_path):
    manifest = {'schema': M.EXPECTED_SCHEMA, 'matrix': {'repeats': 3},
                'scenarios': {'model': [{'prompt_band': 's', 'output_band': 's', 'parallel': 1, 'migration_type': 'same_model_new_shape'}]}}
    expected, repeats = M.expected_cells_from_manifest(manifest)
    cells = [{'cell_id': key, 'status': 'valid', 'metrics': {m: record(10, 10) for m in M.METRICS}} for key in expected]
    report = M._write_report(tmp_path / 'manifest.json', manifest, expected, cells, repeats, tmp_path / 'out.json')
    assert report['metric_denominators']['ttft_ms']['coverage'] == 0
    assert report['groups']['model']['model']['valid_cells'] == 0
    assert report['overall_acceptance_status'] == 'fail'
    assert report['repeat_evidence_status'] == 'screening_only'
    assert report['groups']['migration']['same_model_new_shape']['expected_cells'] == 3
    assert 'same_hardware_runtime_new_model' not in report['groups']['migration']

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest


CHECK_NAMES = (
    "schema", "geometry", "tokens", "parallel", "prompt", "model_sha", "binary_sha", "stream",
    "prediction_before_native", "output_policy", "configuration", "configuration_complete",
    "engine_evidence", "engine_contract", "measurement_status", "engine_semantic_proof",
    "parallel_support", "native_request_count", "simulator_request_count", "native_request_ids",
    "simulator_request_ids", "request_set", "proof_manifest_binding",
)


def strict_case(tmp_path, model_scenarios=None, repeats=4):
    model_scenarios = model_scenarios or {"model": 1}
    config = {"ctx": 64, "batch": 8, "ubatch": 8, "threads": 4, "threads_batch": 4,
              "gpu_layers": -1, "seed": 42, "request_timing": "stream",
              "output_mode": "fixed", "ignore_eos": True}
    native_config = {key: value for key, value in config.items() if key not in {"output_mode", "ignore_eos"}} | {"parallel": 1}
    identity_config = {key: native_config[key] for key in ("ctx", "parallel", "batch", "ubatch", "threads", "gpu_layers", "seed")} | {"flash_attn": False}
    runtime = [{"path": str(tmp_path / "llama-server.exe"), "sha256": "b" * 64}]
    proof = {"path": str(tmp_path / "proof.json"), "sha256": "c" * 64}
    manifest = {"schema": M.EXPECTED_SCHEMA, "matrix": {"models": list(model_scenarios), "repeats": repeats},
                "scenarios": {model: [{"prompt_band": "s", "output_band": str(index), "parallel": 1,
                                        "prompt": "synthetic", "requested_output_tokens": 5,
                                        "configuration": copy.deepcopy(config), "migration_type": "shape_holdout"}
                                       for index in range(count)] for model, count in model_scenarios.items()},
                "model_sha256": {model: hashlib.sha256(model.encode()).hexdigest() for model in model_scenarios},
                "binary_sha256": "b" * 64, "hardware_fingerprint": "d" * 64,
                "runtime_artifacts": runtime, "engine_semantic_proof": proof}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    freeze_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected, _ = M.expected_cells_from_manifest(manifest)
    cells = []
    for cell_id, (model, prompt, output, parallel, repeat, scenario) in expected.items():
        cells.append({
            "cell_id": cell_id, "model_key": model, "prompt_band": prompt, "output_band": output, "parallel": parallel, "repeat": repeat,
            "prompt": "synthetic", "requested_output_tokens": 5, "config": copy.deepcopy(config),
            "observed_configuration": copy.deepcopy(native_config), "status": "valid", "completed": True, "observed": True,
            "model_sha256": manifest["model_sha256"][model], "binary_sha256": manifest["binary_sha256"],
            "freeze_manifest": str(path), "freeze_sha256": freeze_sha,
            "prediction_artifact": str(tmp_path / (cell_id + ".prediction.json")), "prediction_sha256": "e" * 64,
            "engine_semantic_proof": copy.deepcopy(proof),
            "checks": {key: True for key in CHECK_NAMES},
            "request_counts": {"native": parallel, "simulator": parallel},
            "parallel_support": {"status": "modeled", "requested": parallel, "native_requests": parallel, "simulator_requests": parallel},
            "identity": {"gguf_sha256": manifest["model_sha256"][model], "hardware_fingerprint": manifest["hardware_fingerprint"],
                         "runtime_fingerprint": M._expected_runtime_fingerprint(config, parallel), "configuration": copy.deepcopy(identity_config)},
            "evidence": {"native_binary": copy.deepcopy(runtime[0]), "runtime_artifacts": copy.deepcopy(runtime), "runtime_stable": True,
                         "engine_semantic_proof": {**proof, "status": "verified"}},
            "metrics": {metric: {**record(10, 10.1), "boundary": "engine", "contract_id": "engine-boundary/v1"} for metric in M.METRICS},
        })
    return manifest, path, freeze_sha, expected, cells


def strict_report(tmp_path, case):
    manifest, path, freeze_sha, expected, cells = case
    return M._write_report(path, manifest, expected, cells, manifest["matrix"]["repeats"], tmp_path / "report.json",
                           require_manifest_identity=True, freeze_sha256=freeze_sha)


def test_manifest_bound_synthetic_cells_are_valid(tmp_path):
    report = strict_report(tmp_path, strict_case(tmp_path))
    assert report["valid_cell_count"] == 4
    assert report["coverage_pass"] is True
    assert report["all_cells_valid"] is True


@pytest.mark.parametrize("field", [
    ("identity", "hardware_fingerprint"), ("identity", "gguf_sha256"), ("identity", "runtime_fingerprint"),
    ("model_sha256",), ("binary_sha256",), ("evidence", "native_binary", "sha256"),
    ("engine_semantic_proof", "sha256"), ("engine_semantic_proof", "path"),
    ("evidence", "engine_semantic_proof", "sha256"), ("evidence", "engine_semantic_proof", "path"),
    ("observed_configuration", "threads"), ("identity", "configuration", "ctx"),
])
def test_manifest_identity_changes_fail_despite_all_producer_checks_true(tmp_path, field):
    case = strict_case(tmp_path)
    cell = case[-1][0]
    cursor = cell
    for key in field[:-1]:
        cursor = cursor[key]
    cursor[field[-1]] = "0" * 64
    report = strict_report(tmp_path, case)
    assert report["valid_cell_count"] == 3
    assert report["metric_denominators"]["ttft_ms"]["coverage"] == 0.75
    assert report["overall_acceptance_status"] == "fail"
    assert any("manifest identity mismatch" in reason for reason in report["invalid_cells"][0]["reasons"])


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("path", "different-library.dll")])
def test_loaded_runtime_path_and_sha_are_bound(tmp_path, field, value):
    case = strict_case(tmp_path)
    case[-1][0]["evidence"]["runtime_artifacts"][0][field] = value
    report = strict_report(tmp_path, case)
    assert report["valid_cell_count"] == 3
    assert "manifest identity mismatch: runtime_artifacts" in report["invalid_cells"][0]["reasons"]


def test_recorded_request_counts_cannot_be_overridden_by_true_checks(tmp_path):
    case = strict_case(tmp_path)
    case[-1][0]["request_counts"]["native"] = 0
    report = strict_report(tmp_path, case)
    assert "observed request counts mismatch" in report["invalid_cells"][0]["reasons"]


def test_every_major_group_must_reach_coverage_threshold(tmp_path):
    case = strict_case(tmp_path, {"large_group": 9, "small_group": 1}, repeats=10)
    case[-1].pop()
    report = strict_report(tmp_path, case)
    assert report["metric_denominators"]["ttft_ms"]["coverage"] == 0.99
    assert report["coverage_by_major_group"]["model"]["small_group"]["ttft_ms"] == 0.9
    assert report["coverage_pass"] is False
    assert {row["group"] for row in report["coverage_failures"]} == {"small_group"}


@pytest.mark.parametrize("field,value", [("schema", "generalization-acceptance-full/v3"),
                                         ("freeze_end_verified", False), ("freeze_end_verified", None),
                                         ("freeze_end_verified", 1), ("freeze_sha256", "0" * 64)])
def test_matrix_input_must_have_schema_verified_end_and_freeze_sha(tmp_path, field, value):
    path, freeze_sha = tmp_path / "freeze.json", "f" * 64
    payload = {"schema": M.MATRIX_SCHEMA, "freeze_end_verified": True, "freeze_sha256": freeze_sha,
               "freeze_manifest": str(path), "cells": []}
    payload[field] = value
    assert M._matrix_input_errors(payload, path, freeze_sha)


def test_formal_merge_rejects_unverified_matrix_input_before_writing_report(tmp_path, monkeypatch):
    manifest, path, freeze_sha, expected, cells = strict_case(tmp_path)
    input_path, output = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps({"schema": M.MATRIX_SCHEMA, "freeze_end_verified": False,
                                     "freeze_manifest": str(path), "freeze_sha256": freeze_sha, "cells": cells}), encoding="utf-8")
    monkeypatch.setattr(M.sys, "argv", ["merge", "--manifest", str(path), "--input", str(input_path), "--output", str(output)])
    monkeypatch.setattr(M.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(SystemExit, match="end freeze was not verified"):
        M.main()
    assert not output.exists()


def test_formal_merge_always_enforces_cell_manifest_identity(tmp_path, monkeypatch):
    manifest, path, freeze_sha, expected, cells = strict_case(tmp_path)
    cells[0]["identity"]["hardware_fingerprint"] = "0" * 64
    input_path, output = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps({"schema": M.MATRIX_SCHEMA, "freeze_end_verified": True,
                                     "freeze_manifest": str(path), "freeze_sha256": freeze_sha, "cells": cells}), encoding="utf-8")
    monkeypatch.setattr(M.sys, "argv", ["merge", "--manifest", str(path), "--input", str(input_path), "--output", str(output)])
    monkeypatch.setattr(M.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    M.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["manifest_identity_required"] is True
    assert report["valid_cell_count"] == 3


def test_formal_merge_rejects_duplicate_cell_inputs(tmp_path, monkeypatch):
    manifest, path, freeze_sha, expected, cells = strict_case(tmp_path)
    payload = {"schema": M.MATRIX_SCHEMA, "freeze_end_verified": True,
               "freeze_manifest": str(path), "freeze_sha256": freeze_sha, "cells": cells}
    inputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for input_path in inputs:
        input_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output.json"
    monkeypatch.setattr(M.sys, "argv", ["merge", "--manifest", str(path), "--input", str(inputs[0]), "--input", str(inputs[1]), "--output", str(output)])
    monkeypatch.setattr(M.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(SystemExit, match="duplicate cell input"):
        M.main()
    assert not output.exists()


def rewrite_freeze(case):
    manifest, path, _sha, expected, cells = case
    path.write_text(json.dumps(manifest), encoding="utf-8")
    freeze_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for cell in cells:
        cell["freeze_sha256"] = freeze_sha
    return manifest, path, freeze_sha, expected, cells


def formal_case(tmp_path, repeats=20):
    case = strict_case(tmp_path, repeats=repeats)
    manifest = case[0]
    protocol = {
        "schema": "generalization-formal-evidence-protocol/v1", "registered_utc": "2026-09-14T00:00:00Z",
        "sampling": {"method": "fixed_repeats", "unit": "independent_native_run", "repeats": repeats,
                     "optional_stopping": False, "additional_sampling": "new_freeze_required", "failure_policy": "retain_all_planned_cells"},
        "uncertainty": {"method": "exact_median_order_statistics", "multiple_comparisons": "bonferroni",
                        "family_confidence_level": 0.95, "max_relative_ci_width_pct": 10.0},
        "observer": {"method": "paired_relative_difference_median", "equivalence_margin_pct": 5.0,
                     "paired_repeats": 20, "order": "alternating_ab_ba", "collection_mode": "benchmark_only", "boundary": "engine"},
    }
    manifest["created_utc"] = "2026-09-14T02:00:00Z"
    manifest["formal_evidence_protocol"] = protocol
    manifest["formal_evidence_protocol_sha256"] = M._canonical_sha(protocol)
    baseline = tmp_path / "baseline.exe"
    baseline.write_bytes(b"synthetic baseline fixture only")
    source = tmp_path / "observer-raw.json"
    rows = {}
    for model, scenarios in manifest["scenarios"].items():
        for scenario in scenarios:
            key = f'{model}|{scenario["prompt_band"]}|{scenario["output_band"]}|{scenario["parallel"]}'
            rows[key] = {"configuration_sha256": M._scenario_binding(model, scenario),
                         "model_sha256": manifest["model_sha256"][model],
                         "runtime_fingerprint": M._expected_runtime_fingerprint(scenario["configuration"], scenario["parallel"]),
                         "pairs": [{"repeat": index, "order": "AB" if index % 2 else "BA",
                                    "baseline_ms": {metric: 100.0 for metric in M.METRICS},
                                    "instrumented_ms": {metric: 100.5 for metric in M.METRICS}} for index in range(1, 21)]}
    raw = {"schema": "engine-observer-pairs/v1", "hardware_fingerprint": manifest["hardware_fingerprint"],
           "instrumented_binary_sha256": manifest["binary_sha256"],
           "baseline_binary_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
           "engine_semantic_proof_sha256": manifest["engine_semantic_proof"]["sha256"],
           "pairs_by_scenario": {key: row["pairs"] for key, row in rows.items()}}
    source.write_text(json.dumps(raw), encoding="utf-8")
    evidence = {"schema": "engine-observer-equivalence/v1", "protocol_sha256": manifest["formal_evidence_protocol_sha256"],
                "created_utc": "2026-09-14T01:30:00Z", "first_observation_utc": "2026-09-14T01:00:00Z", "last_observation_utc": "2026-09-14T01:29:00Z",
                "collection_mode": "benchmark_only", "boundary": "engine", "hardware_fingerprint": manifest["hardware_fingerprint"],
                "instrumented_binary_sha256": manifest["binary_sha256"], "engine_semantic_proof_sha256": manifest["engine_semantic_proof"]["sha256"],
                "baseline_binary": {"path": str(baseline), "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()},
                "source_artifacts": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}], "scenarios": rows}
    evidence_path = tmp_path / "observer.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest["observer_equivalence_evidence"] = {"path": str(evidence_path), "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()}
    return rewrite_freeze(case)


def rewrite_observer(case, change):
    ref = case[0]["observer_equivalence_evidence"]
    path = Path(ref["path"])
    evidence = json.loads(path.read_text(encoding="utf-8"))
    change(evidence)
    path.write_text(json.dumps(evidence), encoding="utf-8")
    ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rewrite_freeze(case)


@pytest.mark.parametrize("repeats", [4, 20, 100])
def test_repeat_count_alone_never_enables_formal_acceptance(tmp_path, repeats):
    report = strict_report(tmp_path, strict_case(tmp_path, repeats=repeats))
    assert report["accuracy_pass"] and report["all_cells_valid"] and report["coverage_pass"]
    assert report["evidence_pass"] is False
    assert report["evidence_status"] == "evidence_insufficient"
    assert report["overall_acceptance_status"] == "fail"


def test_exact_median_interval_uses_finite_order_statistics_only():
    assert M._exact_median_interval([10.0] * 4, 0.95) is None
    interval = M._exact_median_interval(list(range(1, 11)), 0.95)
    assert interval["lower"] == 2.0 and interval["upper"] == 9.0
    assert interval["confidence_level"] == 1 - 22 / 1024


def test_formal_protocol_uncertainty_and_observer_equivalence_can_pass(tmp_path):
    report = strict_report(tmp_path, formal_case(tmp_path))
    assert report["evidence_pass"] is True
    assert report["overall_acceptance_status"] == "pass"
    assert report["formal_evidence"]["uncertainty"]["status"] == "sufficient"
    assert report["formal_evidence"]["observer_equivalence"]["status"] == "equivalent"


def test_four_identical_runs_still_lack_finite_formal_confidence_interval(tmp_path):
    report = strict_report(tmp_path, formal_case(tmp_path, repeats=4))
    assert report["all_cells_valid"] and report["accuracy_pass"]
    assert report["formal_evidence"]["observer_equivalence"]["status"] == "equivalent"
    assert report["formal_evidence"]["uncertainty"]["status"] == "evidence_insufficient"
    assert report["evidence_pass"] is False


@pytest.mark.parametrize("change", [
    lambda p: p["sampling"].update(optional_stopping=True),
    lambda p: p["sampling"].update(additional_sampling="until_pass"),
    lambda p: p["sampling"].update(repeats=21),
    lambda p: p["uncertainty"].update(family_confidence_level=0.90),
    lambda p: p["observer"].update(equivalence_margin_pct=6.0),
    lambda p: p.update(registered_utc="2026-09-14T03:00:00Z"),
])
def test_unregistered_or_relaxed_protocols_are_rejected(tmp_path, change):
    case = formal_case(tmp_path)
    change(case[0]["formal_evidence_protocol"])
    case[0]["formal_evidence_protocol_sha256"] = M._canonical_sha(case[0]["formal_evidence_protocol"])
    report = strict_report(tmp_path, rewrite_freeze(case))
    assert report["formal_evidence"]["sampling_status"] == "evidence_insufficient"
    assert report["evidence_pass"] is False


def test_protocol_sha_is_enforced(tmp_path):
    case = formal_case(tmp_path)
    case[0]["formal_evidence_protocol_sha256"] = "0" * 64
    report = strict_report(tmp_path, rewrite_freeze(case))
    assert "formal evidence protocol SHA mismatch" in report["formal_evidence"]["reasons"]


def test_missing_observer_proof_is_explicitly_insufficient(tmp_path):
    case = formal_case(tmp_path)
    del case[0]["observer_equivalence_evidence"]
    report = strict_report(tmp_path, rewrite_freeze(case))
    assert report["formal_evidence"]["uncertainty"]["status"] == "sufficient"
    assert report["formal_evidence"]["observer_equivalence"]["status"] == "evidence_insufficient"
    assert report["evidence_pass"] is False


@pytest.mark.parametrize("direction", [1, -1])
def test_observer_equivalent_label_cannot_override_confidence_interval(tmp_path, direction):
    case = formal_case(tmp_path)
    def change(evidence):
        evidence["status"] = "equivalent"
        for row in evidence["scenarios"].values():
            for index, pair in enumerate(row["pairs"]):
                pair["instrumented_ms"] = {metric: 100.0 + (direction * 10.0 if index >= 13 else 0.0) for metric in M.METRICS}
    report = strict_report(tmp_path, rewrite_observer(case, change))
    observer = report["formal_evidence"]["observer_equivalence"]
    assert observer["status"] == "evidence_insufficient"
    assert any("not contained within [-5%, +5%]" in reason for reason in observer["reasons"])
    assert report["evidence_pass"] is False


def test_observer_evidence_requires_complete_frozen_scenario_coverage(tmp_path):
    case = formal_case(tmp_path)
    report = strict_report(tmp_path, rewrite_observer(case, lambda evidence: evidence["scenarios"].clear()))
    assert "observer evidence must cover exactly all frozen scenarios" in report["formal_evidence"]["observer_equivalence"]["reasons"]


def test_observer_artifact_drift_is_not_accepted(tmp_path):
    case = formal_case(tmp_path)
    ref = case[0]["observer_equivalence_evidence"]
    Path(ref["path"]).write_text("{}", encoding="utf-8")
    report = strict_report(tmp_path, case)
    assert any("SHA mismatch" in reason for reason in report["formal_evidence"]["observer_equivalence"]["reasons"])


def test_observer_raw_source_drift_is_not_accepted(tmp_path):
    case = formal_case(tmp_path)
    evidence = json.loads(Path(case[0]["observer_equivalence_evidence"]["path"]).read_text(encoding="utf-8"))
    Path(evidence["source_artifacts"][0]["path"]).write_text("changed", encoding="utf-8")
    report = strict_report(tmp_path, case)
    assert any("observer source:" in reason for reason in report["formal_evidence"]["observer_equivalence"]["reasons"])


def test_excessive_measurement_uncertainty_cannot_pass_on_medians(tmp_path):
    case = formal_case(tmp_path)
    for index, cell in enumerate(case[-1]):
        for metric in M.METRICS:
            cell["metrics"][metric]["native_ms"] = 10.0 if index < 13 else 20.0
    report = strict_report(tmp_path, case)
    assert report["accuracy_pass"] is True
    assert report["formal_evidence"]["uncertainty"]["status"] == "evidence_insufficient"
    assert any("uncertainty width exceeded" in reason for reason in report["formal_evidence"]["uncertainty"]["reasons"])
    assert report["evidence_pass"] is False


def test_missing_captured_proof_sha_is_invalid_without_current_hash_fallback(tmp_path):
    case = strict_case(tmp_path)
    del case[-1][0]["evidence"]["engine_semantic_proof"]["sha256"]
    report = strict_report(tmp_path, case)
    assert report["valid_cell_count"] == 3
    assert "manifest identity mismatch: engine_semantic_proof" in report["invalid_cells"][0]["reasons"]


def test_observer_summary_cannot_substitute_measurements_for_raw_pairs(tmp_path):
    case = formal_case(tmp_path)
    def change(evidence):
        for row in evidence["scenarios"].values():
            for pair in row["pairs"]:
                pair["instrumented_ms"] = {metric: 100.0 for metric in M.METRICS}
    report = strict_report(tmp_path, rewrite_observer(case, change))
    assert report["formal_evidence"]["observer_equivalence"]["status"] == "evidence_insufficient"
    assert any("differ from frozen raw paired observations" in reason
               for reason in report["formal_evidence"]["observer_equivalence"]["reasons"])
