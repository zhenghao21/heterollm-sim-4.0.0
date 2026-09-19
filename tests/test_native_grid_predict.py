"""Pure wrapper tests: no model inference, native processes, or measured answers."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools import native_grid_predict as grid


@dataclass(frozen=True)
class Workload:
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    workload: Workload = field(default_factory=Workload)


def inputs(tmp_path, monkeypatch, *, count=1, fail=False):
    model = tmp_path / "test.gguf"
    model.write_bytes(b"fixture model metadata only")
    runtime = tmp_path / grid.RUNTIME
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"fixture binary identity; never execute")
    ids = list(range(128))
    prompts = {"models": [{"model": str(model), "model_id": "test", "model_sha256": "gguf-sha",
        "prompts": {"128": {"ids": ids, "ids_sha256": grid.stable_hash(ids)}}}]}
    hardware = {"cpu": "fixture", "gpu": {"name": "fixture", "clocks": {"graphics_mhz": 2400}}}
    protocol = {"schema": "native-repeatability-protocol/v1", "exe": grid.RUNTIME,
        "environment": {"GGML_OP_OFFLOAD_MIN_BATCH": None, "OMP_NUM_THREADS": "16"},
        "defaults": {"batch": 64, "ubatch": 64, "threads": 16, "seed": 42, "kv_unified_per_slot": 2048},
        "jobs": [{"id": f"test_p128_o32_c2_{i}", "model": str(model), "expected_prompt_tokens": 128,
            "prompt_token_ids": ids, "output": 32, "parallel": 2, "gpu_layers": 0,
            "warmup_batches": 2, "measure_batches": 3, "process_blocks": 1} for i in range(count)]}
    for name, value in (("protocol.json", protocol), ("prompts.json", prompts), ("hardware.json", hardware)):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    calls = {"read": [], "model": [], "scenario": [], "run": [], "timing": []}
    def read(path):
        calls["read"].append(path)
        return SimpleNamespace(sha256="gguf-sha")
    def model_build(gguf):
        calls["model"].append(gguf)
        return "static-model"
    def scenario_build(*args, **kwargs):
        calls["scenario"].append((args, kwargs))
        return Scenario()
    def run(scenario, **kwargs):
        calls["run"].append((scenario, kwargs))
        if fail:
            raise RuntimeError("simulator intentionally unsupported")
        return SimpleNamespace(metrics=SimpleNamespace(request_metrics={
            f"r{i}": SimpleNamespace(visible_output_tokens=32) for i in range(2)}))
    def timing(result, metric):
        calls["timing"].append(metric)
        return {"engine_ttft_ms": 1., "engine_tpot_ms": 2., "engine_e2e_ms": 63.}
    monkeypatch.setattr(grid, "read_gguf_metadata", read)
    monkeypatch.setattr(grid, "build_model_from_gguf", model_build)
    monkeypatch.setattr(grid, "build_matching_scenario", scenario_build)
    monkeypatch.setattr(grid.reporting, "run_scenario", run)
    monkeypatch.setattr(grid, "_simulator_request_timing", timing)
    monkeypatch.setattr(grid, "source_identity", lambda: {"sha256": "source-sha", "files": [], "execution_source_root": str(grid.ROOT)})
    return protocol, prompts, hardware, calls


def test_one_cell_uses_only_frozen_static_inputs(tmp_path, monkeypatch):
    import subprocess
    from tools import native_llama_compare as compare
    def forbidden(*args, **kwargs):
        pytest.fail("native process, live hardware probe, or calibration is forbidden")
    for name in ("Popen", "run", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("probe_hardware", "load_native_calibration", "apply_native_calibration", "post_json"):
        monkeypatch.setattr(compare, name, forbidden)
    protocol, prompts, hardware, calls = inputs(tmp_path, monkeypatch)
    # Unexpected answer-like keys are never forwarded as simulation inputs.
    protocol["native"] = {"engine_ttft_ms": 999999}
    protocol["native_profile"] = "DO-NOT-READ.json"
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    manifest = grid.build_predictions(tmp_path / "protocol.json", tmp_path / "out", data_root=tmp_path)
    assert manifest["planned_cells"] == manifest["successful_cells"] == 1
    assert len(calls["read"]) == len(calls["model"]) == len(calls["run"]) == 1
    args, kw = calls["scenario"][0]
    assert args == (128, 32)
    assert kw == {"ctx": 2048, "parallel": 2, "batch": 64, "ubatch": 64, "threads": 16,
        "gpu_layers": 0, "seed": 42, "coherent_dma_mode": "pipelined", "op_offload": True,
        "model": "static-model", "hardware_snapshot": hardware,
        "runtime_binary": (tmp_path / grid.RUNTIME).resolve(),
        "runtime_environment": {"GGML_OP_OFFLOAD_MIN_BATCH": None, "OMP_NUM_THREADS": "16"}}
    scenario, run_kw = calls["run"][0]
    assert run_kw == {"retention_policy": "aggregate"}
    assert scenario.workload.metadata["serving_runtime"]["kv_slot_context_tokens"] == 2048
    ref = manifest["prediction_refs"][0]
    prediction = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    assert prediction["formal_prediction_eligible"] is False
    assert prediction["native_answers_used"] is False
    assert len(prediction["formal_prediction_ineligibility_reasons"]) == 5
    assert prediction["requests"][0]["engine_e2e_ms"] == 63
    assert prediction["aggregate"]["engine_ttft_ms"]["median_ms"] == 1
    assert grid.file_ref(ref["path"])["sha256"] == ref["sha256"]
    expected = prediction.pop("content_sha256")
    assert grid.stable_hash(prediction) == expected


def test_135_failures_keep_denominator_and_cache_model_once(tmp_path, monkeypatch):
    _, _, _, calls = inputs(tmp_path, monkeypatch, count=135, fail=True)
    manifest = grid.build_predictions(tmp_path / "protocol.json", tmp_path / "out", data_root=tmp_path)
    assert manifest["planned_cells"] == manifest["prediction_files"] == manifest["failed_or_incomplete_cells"] == 135
    assert manifest["successful_cells"] == 0
    assert len(calls["read"]) == len(calls["model"]) == 1
    assert len(manifest["prediction_refs"]) == 135
    record = json.loads(Path(manifest["prediction_refs"][0]["path"]).read_text(encoding="utf-8"))
    assert "intentionally unsupported" in record["reason"]
    assert all(row[m] is None for row in record["requests"] for m in grid.METRICS)
    assert all(v["median_ms"] is None and v["planned_requests"] == 2 for v in record["aggregate"].values())


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    inputs(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "retained.json").write_text("unchanged")
    with pytest.raises(FileExistsError):
        grid.build_predictions(tmp_path / "protocol.json", out, data_root=tmp_path)
    assert (out / "retained.json").read_text() == "unchanged"


def test_prompt_mismatch_fails_before_model_or_simulation(tmp_path, monkeypatch):
    protocol, _, _, calls = inputs(tmp_path, monkeypatch)
    protocol["jobs"][0]["prompt_token_ids"][1] = 999
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    manifest = grid.build_predictions(tmp_path / "protocol.json", tmp_path / "out", data_root=tmp_path)
    assert manifest["failed_or_incomplete_cells"] == 1
    assert not calls["read"] and not calls["scenario"]


def test_new_runtime_identity_is_required(tmp_path, monkeypatch):
    protocol, _, _, calls = inputs(tmp_path, monkeypatch)
    old = tmp_path / "old.exe"
    old.write_bytes(b"old")
    protocol["exe"] = str(old)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="actual new thread-control runtime"):
        grid.build_predictions(tmp_path / "protocol.json", tmp_path / "out", data_root=tmp_path)
    assert not calls["read"]

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_write_new_rejects_unserializable_evidence_without_partial_file(tmp_path, bad):
    target = tmp_path / "result.json"
    with pytest.raises((ValueError, TypeError)):
        grid.write_new(target, {"metric": bad})
    assert not target.exists()
    grid.write_new(target, {"metric": 1})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        grid.write_new(target, {"metric": 2})
    assert target.read_bytes() == before


def test_large_file_reference_streams_and_reuses_unchanged_digest(tmp_path, monkeypatch):
    import hashlib
    target = tmp_path / "model.gguf"
    payload = b"small stand-in for a large model"
    target.write_bytes(payload)
    monkeypatch.setattr(grid, "_MAX_CACHED_RAW_BYTES", 1)
    def no_full_read(*args):
        pytest.fail("large identity reads must stream")
    monkeypatch.setattr(Path, "read_bytes", no_full_read)
    expected = {"path": str(target.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload)}
    assert grid.file_ref(target) == expected
    monkeypatch.setattr(Path, "open", no_full_read)
    assert grid.file_ref(target) == expected
