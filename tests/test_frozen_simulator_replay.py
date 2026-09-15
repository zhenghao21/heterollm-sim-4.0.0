"""Regression coverage for immutable development replay and honest aggregation."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import frozen_simulator_replay as fr


def test_file_binding_fails_on_empty_sha_missing_file_and_drift(tmp_path):
    path = tmp_path / "native.json"
    path.write_text('{"native": 1}', encoding="utf-8")
    refs = {}
    fr.bind_artifact(refs, fr.ref(path))
    fr.check_files(refs)
    with pytest.raises(ValueError, match="path or SHA"):
        fr.bind_artifact({}, {"path": str(path), "sha256": None})
    with pytest.raises(ValueError, match="empty frozen"):
        fr.check_files({})
    path.write_text('{"native": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        fr.check_files(refs)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        fr.check_files(refs)


def test_frozen_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "prediction.json"
    fr.save_new(path, {"prediction": 1})
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        fr.save_new(path, {"prediction": 2})
    assert path.read_bytes() == old


def _expected():
    expected = {}
    for model in ("a", "b"):
        for repeat in range(1, 4):
            key = f"{model}__short__short__p1__r{repeat}"
            expected[key] = (model, "short", "short", 1, repeat, {"migration_type": "shape"})
    return expected


def _records(expected):
    return [{"cell_id": key, "completed": True, "status": "valid", "metrics": {
        name: {"status": "measured", "native_ms": 10.0, "simulator_ms": 10.5}
        for name in fr.METRICS}} for key in expected]


def test_invalid_and_missing_replays_stay_in_group_denominators():
    expected = _expected()
    rows = _records(expected)
    rows[0]["status"] = "invalid"
    rows[0]["metrics"]["ttft_ms"]["simulator_ms"] = 999999.0
    rows.pop()
    groups = fr.summaries(expected, rows, {"hardware_fingerprint": "gpu"}, 3)
    assert groups["model"]["a"]["expected_cells"] == 3
    assert groups["model"]["a"]["valid_cells"] == 2
    assert groups["model"]["b"]["missing_cells"] == 1
    assert groups["model"]["a"]["metrics"]["ttft_ms"]["median_of_repeats_abs_pct"] is None
    assert groups["model"]["a"]["evidence_completeness"]["ttft_ms"] == pytest.approx(2 / 3)
    assert not fr.accuracy_gate(groups)


def test_medians_computed_before_scenario_error():
    expected = _expected()
    rows = _records(expected)
    for row, native in zip(rows[:3], (1.0, 10.0, 100.0)):
        for metric in row["metrics"].values():
            metric["native_ms"] = native
    groups = fr.summaries(expected, rows, {"hardware_fingerprint": "gpu"}, 3)
    stat = groups["model"]["a"]["metrics"]["ttft_ms"]
    assert stat["median_of_repeats_abs_pct"] == pytest.approx(5.0)
    assert stat["median_absolute_ms"] == pytest.approx(.5)
    assert fr.accuracy_gate(groups)


def test_freeze_rejects_source_set_drift_without_running_native(tmp_path, monkeypatch):
    original = tmp_path / "original.json"
    fr.save_new(original, {"schema": "blind-generalization-freeze/v5", "matrix": {"repeats": 1},
                            "scenarios": {"a": [{"prompt_band": "s", "output_band": "s", "parallel": 1}]}})
    parent = tmp_path / "matrix.json"
    fr.save_new(parent, {"cell": 1})
    manifest = tmp_path / "freeze.json"
    fr.save_new(manifest, {"schema": fr.SCHEMA, "calibration_mode": "analytical", "independent_blind": False,
        "parent_matrix": fr.ref(parent), "parent_freeze": fr.ref(original),
        "source_sha256": {"old.py": "a" * 64}, "repeats": 1, "planned_cell_count": 1,
        "expected_cells": [{"cell_id": "a__s__s__p1__r1"}]})
    monkeypatch.setattr(fr, "source_identity", lambda: {"new.py": "b" * 64})
    with pytest.raises(ValueError, match="source file set"):
        fr.verify_freeze(manifest, full=False)

def _native_fixture(tmp_path, *, suffix="1", prompt="p"):
    raw = tmp_path / ("raw" + suffix + ".json")
    raw.write_text(json.dumps({"run": suffix}), encoding="utf-8")
    original = {"hardware_fingerprint": "h", "model_sha256": {"a": "m"}, "binary_sha256": "b",
                "runtime_artifacts": [{"path": str(tmp_path / "runtime.dll"), "sha256": "a" * 64}]}
    scene = {"prompt": "p", "requested_output_tokens": 2, "configuration": {"batch": 64}}
    payload = {"request": {"prompt": prompt, "requested_output_tokens": 2, "output_mode": "fixed", "ignore_eos": True},
               "configuration": {"batch": 64, "parallel": 1},
               "identity": {"hardware_fingerprint": "h", "gguf_sha256": "m"},
               "evidence": {"native_binary": {"sha256": "b"}, "runtime_artifacts": original["runtime_artifacts"], "raw_native_capture": fr.ref(raw)},
               "native": {"requests": [{"request_id": "r0", "output_tokens": 2, "truncated": False}]},
               "parallel_support": {"requested": 1, "native_requests": 1}}
    source = tmp_path / ("cell" + suffix + ".json")
    fr.save_new(source, payload)
    return payload, original, ("a", "short", "short", 1, 1, scene), source


def test_freeze_mapping_rejects_one_native_source_as_two_repeats(tmp_path):
    _, original, spec, source = _native_fixture(tmp_path)
    expected = {"r1": spec, "r2": spec[:4] + (2,) + spec[5:]}
    specs = [{"cell_id": key, "model_key": "a", "source": fr.ref(source)} for key in expected]
    with pytest.raises(ValueError, match="one native source"):
        fr.validate_source_mapping(specs, expected, original)


def test_freeze_mapping_rejects_copied_payload_with_same_raw_capture(tmp_path):
    payload, original, spec, source = _native_fixture(tmp_path)
    source2 = tmp_path / "copied.json"
    fr.save_new(source2, {**payload, "copy": True})
    expected = {"r1": spec, "r2": spec[:4] + (2,) + spec[5:]}
    specs = [{"cell_id": key, "model_key": "a", "source": fr.ref(path)} for key, path in zip(expected, (source, source2))]
    with pytest.raises(ValueError, match="one raw native capture"):
        fr.validate_source_mapping(specs, expected, original)


@pytest.mark.parametrize("changed", ["prompt", "output", "parallel", "batch", "model", "request_ids"])
def test_scene_identity_checked_against_native(tmp_path, changed):
    payload, original, spec, _ = _native_fixture(tmp_path)
    fr.validate_native_cell(payload, spec, original)
    if changed == "prompt": payload["request"]["prompt"] = "another prompt"
    elif changed == "output": payload["request"]["requested_output_tokens"] = 99
    elif changed == "parallel": payload["configuration"]["parallel"] = 2
    elif changed == "batch": payload["configuration"]["batch"] = 32
    elif changed == "model": payload["identity"]["gguf_sha256"] = "wrong"
    elif changed == "request_ids": payload["native"]["requests"] *= 2
    with pytest.raises(ValueError):
        fr.validate_native_cell(payload, spec, original)

def _resume_fixture(tmp_path, monkeypatch):
    payload, original, expected, source = _native_fixture(tmp_path)
    payload["token_counts"] = {"prompt": 8, "output": 2}
    record = payload["native"]["requests"][0]
    record.update(prompt_tokens=8, engine_ttft_ms=10.0, engine_tpot_ms=2.0, engine_e2e_ms=12.0,
                  engine_timing_status="counter_proven", timing_contract_id="engine-boundary/v1")
    source.write_text(json.dumps(payload), encoding="utf-8")
    spec = {"cell_id": "r1", "model_key": "a", "source": fr.ref(source)}
    pred = {"schema": "simulator-replay-prediction/v1", "calibration_mode": "analytical",
            "independent_blind_prediction": False, "cell_id": "r1", "freeze_sha256": "f" * 64,
            "source_payload_sha256": spec["source"]["sha256"], "simulator_requests": [
                {**record, "visible_output_tokens": 2, "engine_timing_status": "measured",
                 "engine_ttft_ms": 20.0, "engine_tpot_ms": 4.0, "engine_e2e_ms": 24.0}]}
    path = tmp_path / "r1.prediction.json"
    fr.save_new(path, pred)
    cached = {"cell_id": "r1", "source": spec["source"], "freeze_sha256": "f" * 64,
              "prediction_artifact": fr.ref(path), "status": "valid", "metrics": {
                  k: {"status": "measured", "native_ms": 10.0, "simulator_ms": 10.0} for k in fr.METRICS}}
    monkeypatch.setattr(fr, "_validate_native_evidence", lambda *a: ([], {}))
    return cached, spec, path, expected, original, pred


def test_resume_recomputes_metrics_instead_of_trusting_cached_perfect_score(tmp_path, monkeypatch):
    cached, spec, path, expected, original, _ = _resume_fixture(tmp_path, monkeypatch)
    source_before = Path(spec["source"]["path"]).read_bytes()
    result = fr.rescore_bound_prediction(cached, spec, "f" * 64, path, expected, original)
    assert result["status"] == "valid"
    assert result["metrics"]["ttft_ms"]["absolute_error_pct"] == pytest.approx(100)
    assert result["metrics"]["e2e_ms"]["simulator_ms"] == pytest.approx(24)
    assert result["acceptance_eligible"] is False
    assert Path(spec["source"]["path"]).read_bytes() == source_before


@pytest.mark.parametrize("field,value", [("calibration_mode", "legacy"), ("cell_id", "r2"),
    ("source_payload_sha256", "b" * 64), ("freeze_sha256", "e" * 64)])
def test_resume_rejects_prediction_with_wrong_frozen_context(tmp_path, monkeypatch, field, value):
    cached, spec, path, expected, original, pred = _resume_fixture(tmp_path, monkeypatch)
    pred[field] = value
    path.write_text(json.dumps(pred), encoding="utf-8")
    cached["prediction_artifact"] = fr.ref(path)
    with pytest.raises(ValueError, match="prediction content"):
        fr.rescore_bound_prediction(cached, spec, "f" * 64, path, expected, original)


def test_resume_rejects_record_from_another_cell(tmp_path, monkeypatch):
    cached, spec, path, expected, original, _ = _resume_fixture(tmp_path, monkeypatch)
    cached["cell_id"] = "r2"
    with pytest.raises(ValueError, match="result identity"):
        fr.rescore_bound_prediction(cached, spec, "f" * 64, path, expected, original)
