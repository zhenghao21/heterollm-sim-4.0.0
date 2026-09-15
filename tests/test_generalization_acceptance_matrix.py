import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generalization_acceptance_matrix", ROOT / "tools" / "generalization_acceptance_matrix.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_matrix_has_requested_dimensions_and_repeats():
    cells = list(MODULE.planned_cells(list(MODULE.MODELS), 3))
    assert len(MODULE.MODELS) == 5
    assert len(MODULE.PROMPTS) == 3
    assert len(MODULE.OUTPUTS) == 3
    assert MODULE.PARALLEL == (1, 2, 4)
    assert len(cells) == 405


def test_aggregate_record_uses_concurrent_batch_p50_and_absolute_delta():
    payload = {
        "native": {
            "aggregate": {
                "engine_ttft_ms": {"p50_ms": 10.0},
                "engine_tpot_ms": {"p50_ms": 2.0},
                "engine_e2e_ms": {"p50_ms": 20.0},
            }
        },
        "simulator": {
            "aggregate": {
                "engine_ttft_ms": {"p50_ms": 11.0},
                "engine_tpot_ms": {"p50_ms": 2.2},
                "engine_e2e_ms": {"p50_ms": 19.0},
            }
        },
    }
    records = MODULE.metric_records(payload)
    assert records["ttft_ms"]["signed_error_pct"] == 10.0
    assert abs(records["tpot_ms"]["absolute_delta_ms"] - 0.2) < 1e-9
    assert records["e2e_ms"]["signed_error_pct"] == -5.0

import copy
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def payload_case(tmp_path, monkeypatch):
    # The chronology checker has its own file tests. This fixture isolates
    # request geometry/proof checks without accessing any native evidence.
    monkeypatch.setattr(MODULE, "validate_prediction_before_native", lambda *args, **kwargs: [])
    proof = tmp_path / "proof.json"
    proof.write_text('{"schema":"synthetic-proof"}', encoding="utf-8")
    config = {"ctx": 64, "batch": 8, "ubatch": 8, "threads": 4, "threads_batch": 4,
              "gpu_layers": -1, "seed": 42, "request_timing": "stream",
              "output_mode": "fixed", "ignore_eos": True}
    records = [{"request_id": "request-0000"}, {"request_id": "request-0001"}]
    payload = {
        "schema": "native-simulator-comparison/v2",
        "configuration": {key: value for key, value in config.items() if key not in {"output_mode", "ignore_eos"}} | {"parallel": 2},
        "parity": {"geometry": {"ok": True}, "tokens": {"ok": True}},
        "request": {"prompt": "synthetic", "requested_output_tokens": 5, "output_mode": "fixed", "request_timing": "stream"},
        "output_policy": {"mode": "fixed", "ignore_eos": True},
        "gguf": {"gguf": {"sha256": "a" * 64}},
        "evidence": {"native_binary": {"sha256": "b" * 64},
                     "engine_timing": {"status": "counter_proven"},
                     "timing_contract": {"id": "engine-stage+client-real-token/v3"},
                     "engine_semantic_proof": {"status": "verified", "path": str(proof), "sha256": MODULE.sha(proof)}},
        "validity_status": "request_boundary_aligned",
        "native": {"requests": copy.deepcopy(records)}, "simulator": {"requests": copy.deepcopy(records)},
        "parallel_support": {"status": "modeled", "requested": 2, "native_requests": 2, "simulator_requests": 2},
    }
    kwargs = {"model": "synthetic", "prompt": "synthetic", "output": 5, "parallel": 2,
              "model_sha": "a" * 64, "binary_sha": "b" * 64, "expected_config": config,
              "engine_semantic_proof": proof, "engine_semantic_proof_sha256": MODULE.sha(proof)}
    return payload, kwargs


def test_complete_parallel_request_sets_validate(payload_case):
    payload, kwargs = payload_case
    checks, valid = MODULE.validate_payload(payload, **kwargs)
    assert valid, checks


@pytest.mark.parametrize("side", ["native", "simulator"])
def test_request_length_cannot_be_hidden_by_support_counter(payload_case, side):
    payload, kwargs = payload_case
    payload[side]["requests"].pop()
    checks, valid = MODULE.validate_payload(payload, **kwargs)
    assert not valid
    assert checks[side + "_request_count"] is False


@pytest.mark.parametrize("side", ["native", "simulator"])
def test_support_counter_must_match_observed_requests(payload_case, side):
    payload, kwargs = payload_case
    payload["parallel_support"][side + "_requests"] = 1
    checks, valid = MODULE.validate_payload(payload, **kwargs)
    assert not valid
    assert checks[side + "_request_count"] is False


def test_parallel_request_sets_must_have_matching_unique_ids(payload_case):
    payload, kwargs = payload_case
    payload["native"]["requests"][1]["request_id"] = "request-0000"
    checks, valid = MODULE.validate_payload(payload, **kwargs)
    assert not valid
    assert checks["request_set"] is False


def test_proof_override_requires_same_frozen_path_and_bytes(tmp_path):
    proof = tmp_path / "proof.json"
    proof.write_text("proof bytes", encoding="utf-8")
    freeze = {"engine_semantic_proof": {"path": str(proof), "sha256": MODULE.sha(proof)}}
    assert MODULE._resolve_engine_semantic_proof(freeze) == (proof.resolve(), MODULE.sha(proof))
    alias = tmp_path / "other-proof.json"
    alias.write_bytes(proof.read_bytes())
    with pytest.raises(ValueError, match="override path differs"):
        MODULE._resolve_engine_semantic_proof(freeze, alias)
    proof.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA differs"):
        MODULE._resolve_engine_semantic_proof(freeze, proof)


@pytest.mark.parametrize("mutation", ["path", "declared_sha", "missing_sha", "null_sha", "actual_sha"])
def test_verified_proof_label_cannot_bypass_manifest_binding(payload_case, tmp_path, mutation):
    payload, kwargs = payload_case
    if mutation == "path":
        payload["evidence"]["engine_semantic_proof"]["path"] = str(tmp_path / "different.json")
    elif mutation == "declared_sha":
        payload["evidence"]["engine_semantic_proof"]["sha256"] = "0" * 64
    elif mutation == "missing_sha":
        payload["evidence"]["engine_semantic_proof"].pop("sha256")
    elif mutation == "null_sha":
        payload["evidence"]["engine_semantic_proof"]["sha256"] = None
    else:
        kwargs["engine_semantic_proof"].write_text("changed", encoding="utf-8")
    checks, valid = MODULE.validate_payload(payload, **kwargs)
    assert not valid
    assert checks["proof_manifest_binding"] is False


def test_proof_capture_hashes_the_same_original_bytes_it_parses(tmp_path, monkeypatch):
    import hashlib
    from tools.native_llama_compare import _read_engine_semantic_proof

    # Shape validation only: no native result or frozen proof is read. Use
    # noncanonical JSON bytes, then replace the file immediately after read.
    # A second read for hashing would bind the capture to the wrong proof.
    original = b'{\r\n  "schema": "synthetic-proof", "revision": 1\r\n}\r\n'
    replacement = b'{"schema":"synthetic-proof","revision":2}'
    proof = tmp_path / "proof.json"
    proof.write_bytes(original)
    read_bytes = Path.read_bytes
    reads = []

    def read_then_replace(path):
        reads.append(path)
        captured = read_bytes(path)
        path.write_bytes(replacement)
        return captured

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    document, capture = _read_engine_semantic_proof(proof)
    assert reads == [proof]
    assert document == {"schema": "synthetic-proof", "revision": 1}
    assert capture == {"path": str(proof.resolve()), "sha256": hashlib.sha256(original).hexdigest()}
    assert capture["sha256"] != hashlib.sha256(replacement).hexdigest()


def test_aggregate_group_excludes_invalid_numeric_repeats():
    good = {"status": "valid", "observed": True, "completed": True,
            "metrics": {metric: {"status": "measured", "native_ms": 10.0, "simulator_ms": 11.0} for metric in MODULE.METRICS}}
    bad = copy.deepcopy(good)
    bad["status"] = "invalid"
    bad["metrics"]["ttft_ms"]["native_ms"] = 10000.0
    result = MODULE._aggregate_group([good, bad], 2)
    metric = result["metrics"]["ttft_ms"]
    assert result["observed_repeats"] == 2
    assert result["valid_repeats"] == 1
    assert result["metric_eligible_repeats"]["ttft_ms"] == 1
    assert metric["native_median_ms"] == 10.0
    assert metric["median_of_repeats_abs_pct"] == 10.0
    assert metric["status"] == "incomplete"


def test_partial_na_is_evidence_insufficient():
    cells = [{"status": "valid", "metrics": {metric: {"status": "not_applicable"} for metric in MODULE.METRICS}}]
    assert MODULE._aggregate_group(cells, 3)["metrics"]["ttft_ms"]["status"] == "evidence_insufficient"


def test_failed_end_gate_recomputes_counts_and_aggregation():
    planned = [("c1", "model", "short", "short", 1, 1)]
    cells = [{"cell_id": "c1", "model_key": "model", "prompt_band": "short", "output_band": "short", "parallel": 1,
              "status": "valid", "valid": True, "observed": True, "completed": True,
              "metrics": {metric: {"status": "measured", "native_ms": 10.0, "simulator_ms": 11.0} for metric in MODULE.METRICS}}]
    summary = MODULE._summarize_after_end_gate(cells, planned, False, "source drift")
    assert summary["valid_cell_count"] == 0
    assert summary["invalid_cell_count"] == 1
    assert summary["observed_cell_count"] == summary["completed_cell_count"] == 1
    assert summary["metric_denominators"]["ttft_ms"]["coverage"] == 0
    aggregate = summary["aggregation"]["model|short|short|1"]
    assert aggregate["valid_repeats"] == 0
    assert aggregate["metric_eligible_repeats"]["ttft_ms"] == 0
    assert aggregate["metrics"]["ttft_ms"]["median_of_repeats_abs_pct"] is None
    assert cells[0]["checks"]["freeze_end"] is False


def test_matrix_main_runs_end_gate_before_aggregation_and_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "MODELS", {"model": tmp_path / "model.gguf"})
    monkeypatch.setattr(MODULE, "PROMPTS", {"short": "synthetic"})
    monkeypatch.setattr(MODULE, "OUTPUTS", {"short": 5})
    monkeypatch.setattr(MODULE, "PARALLEL", (1,))
    proof = tmp_path / "proof.json"
    proof.write_text("synthetic", encoding="utf-8")
    freeze = {"schema": MODULE.FREEZE_SCHEMA, "status": "frozen_not_run",
              "matrix": {"models": ["model"], "repeats": 1, "prompt_bands": ["short"], "output_bands": ["short"], "parallel_values": [1]},
              "source_sha256": {"synthetic.py": "a" * 64},
              "engine_semantic_proof": {"path": str(proof), "sha256": MODULE.sha(proof)},
              "scenarios": {"model": [{"prompt_band": "short", "output_band": "short", "parallel": 1, "configuration": {}}]}}
    manifest_path, output = tmp_path / "freeze.json", tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(freeze), encoding="utf-8")
    monkeypatch.setattr(MODULE.sys, "argv", ["matrix", "--dry-run", "--models", "model", "--repeats", "1", "--manifest", str(manifest_path), "--output", str(output)])
    calls = []
    def verify(*args, **kwargs):
        command = args[0]
        assert "native_llama_compare.py" not in " ".join(command)
        phase = command[-1]
        calls.append(phase)
        return SimpleNamespace(returncode=1 if phase == "end" else 0, stdout="source drift" if phase == "end" else "", stderr="")
    monkeypatch.setattr(MODULE.subprocess, "run", verify)
    original_aggregate = MODULE._aggregate_group
    def aggregate(cells, repeats):
        assert calls[-1] == "end"
        assert all(cell["status"] == "invalid" for cell in cells)
        return original_aggregate(cells, repeats)
    monkeypatch.setattr(MODULE, "_aggregate_group", aggregate)
    assert MODULE.main() == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert calls == ["resume", "end"]
    assert result["freeze_end_verified"] is False
    assert result["valid_cell_count"] == 0
    assert result["invalid_cell_count"] == 1


@pytest.mark.parametrize("capture_state", ["captured", "missing", "mismatch"])
def test_resume_requires_the_original_captured_proof_sha(payload_case, tmp_path, monkeypatch, capture_state):
    from tools.evaluation_contract import ENGINE_CONTRACT_ID
    from tools.native_llama_compare import _aggregate_request_records

    # Exercise the actual resume branch with synthetic request sets and a
    # temporary manifest. No native process or existing experiment is used.
    payload, kwargs = payload_case
    for side, value in (("native", 10.0), ("simulator", 11.0)):
        for record in payload[side]["requests"]:
            record.update({"engine_timing_status": "counter_proven", "output_tokens": 5,
                           "timing_contract_id": ENGINE_CONTRACT_ID,
                           "engine_ttft_ms": value, "engine_tpot_ms": value, "engine_e2e_ms": value})
        payload[side]["aggregate"] = _aggregate_request_records(payload[side]["requests"])
    if capture_state == "missing":
        payload["evidence"]["engine_semantic_proof"].pop("sha256")
    elif capture_state == "mismatch":
        payload["evidence"]["engine_semantic_proof"]["sha256"] = "0" * 64
    monkeypatch.setattr(MODULE, "MODELS", {"synthetic": tmp_path / "model.gguf"})
    monkeypatch.setattr(MODULE, "PROMPTS", {"short": kwargs["prompt"]})
    monkeypatch.setattr(MODULE, "OUTPUTS", {"short": kwargs["output"]})
    monkeypatch.setattr(MODULE, "PARALLEL", (kwargs["parallel"],))
    manifest = {"schema": MODULE.FREEZE_SCHEMA, "status": "frozen_not_run",
                "matrix": {"models": ["synthetic"], "repeats": 1, "prompt_bands": ["short"],
                           "output_bands": ["short"], "parallel_values": [kwargs["parallel"]]},
                "source_sha256": {"synthetic.py": "c" * 64},
                "model_sha256": {"synthetic": kwargs["model_sha"]}, "binary_sha256": kwargs["binary_sha"],
                "engine_semantic_proof": {"path": str(kwargs["engine_semantic_proof"]),
                                          "sha256": kwargs["engine_semantic_proof_sha256"]},
                "scenarios": {"synthetic": [{"prompt_band": "short", "output_band": "short",
                                              "parallel": kwargs["parallel"], "configuration": kwargs["expected_config"]}]}}
    manifest_path, output = tmp_path / "manifest.json", tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    cell_id = next(MODULE.planned_cells(["synthetic"], 1, manifest["scenarios"]))[0]
    raw_path = cells_dir / (cell_id + ".json")
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    raw_sha = MODULE.sha(raw_path)
    (cells_dir / (cell_id + ".freeze.json")).write_text(
        json.dumps({"cell_id": cell_id, "freeze_sha256": MODULE.sha(manifest_path)}), encoding="utf-8")
    monkeypatch.setattr(MODULE.sys, "argv", ["matrix", "--models", "synthetic", "--repeats", "1",
                        "--manifest", str(manifest_path), "--output", str(output), "--cells-dir", str(cells_dir),
                        "--engine-semantic-proof", str(kwargs["engine_semantic_proof"])])
    phases = []

    def verify_only(command, **unused):
        assert Path(command[1]).name == "check_freeze_manifest.py"
        phases.append(command[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", verify_only)
    assert MODULE.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    expected_valid = int(capture_state == "captured")
    assert phases == ["resume", "end"]
    assert result["planned_cell_count"] == result["observed_cell_count"] == 1
    assert result["valid_cell_count"] == expected_valid
    assert result["invalid_cell_count"] == 1 - expected_valid
    assert result["cells"][0]["checks"]["proof_manifest_binding"] is bool(expected_valid)
    assert result["metric_denominators"]["ttft_ms"]["coverage"] == expected_valid
    assert MODULE.sha(raw_path) == raw_sha
