"""Pure adapter tests. All simulator/native execution is stubbed; no GPU work."""
from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import threading

import pytest
from tools import native_162_dataset as selector
from tools import predict_stable_native_dataset as adapter


@dataclass(frozen=True)
class Tensor:
    frequency_ghz: float = 2.617
    sm_count: int = 84
    cycles_per_mma: float = 2.0


@dataclass(frozen=True)
class Profile:
    tensor_core: Tensor = field(default_factory=Tensor)


@dataclass(frozen=True)
class Metadata:
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    llama_cpp_config: object = None
    workload: Metadata = field(default_factory=Metadata)
    hardware: Metadata = field(default_factory=Metadata)
    component_profiles: dict = field(default_factory=lambda: {"gpu": {"analytical": Profile()}})


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return adapter.grid.file_ref(path)


def seal(selection):
    selection.pop("payload_sha256", None)
    selection["payload_sha256"] = selector._digest(selection)
    return selection


def fixture(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic metadata")
    runtime = tmp_path / adapter.grid.RUNTIME
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"never run native")
    cuda = runtime.parent / "ggml-cuda.dll"
    cuda.write_bytes(b"synthetic cuda build")
    build = tmp_path / "source/llama.cpp-semantic/build-semantic-direct"
    (build / "bin").mkdir(parents=True)
    (build / "bin/ggml-cuda.dll").write_bytes(cuda.read_bytes())
    (build / "CMakeCache.txt").write_text("GGML_CUDA_GRAPHS:BOOL=OFF\n")
    state = {"gpu_state": {"returncode": 0, "stdout": "GPU-test, P0, 50, 2392 MHz, 15201 MHz, 100, 0\n"}}
    raw_ref = document(tmp_path / "native.json", {"state_measurement_before": state, "state_after": state, "native_latency_ms": 123456})
    config = {"model": str(model), "prompt_token_ids": [1, 2, 3, 4], "expected_prompt_tokens": 4,
        "output": 2, "parallel": 2, "ctx": 4096, "gpu_layers": 66, "fit_params": False,
        "threads": 16, "threads_batch": 16, "worker_cpu_mask": "0x55555555", "poll": 50,
        "batch": 64, "ubatch": 64, "seed": 42, "environment": {"GGML_OP_OFFLOAD_MIN_BATCH": None},
        "flash_attention": False, "load_mode": "mmap"}
    hardware = {"cpu": "fixture", "gpu": {"uuid": "GPU-test", "name": "fixture", "clocks": {"sm_mhz": 2707}}}
    actuals = [{"block": 0, "repeat": repeat, "request_index": i,
        "metrics_ms": {"ttft": center, "tpot": center * 2, "e2e": center * 3}}
        for repeat, center in enumerate([10, 12, 14]) for i in range(2)]
    row = {"cell_id": "qwen38_gpu_p4_o2_c2__fixed_runtime", "model_key": "qwen38_gpu", "parallel": 2,
        "config": config, "model_ref": adapter.grid.file_ref(model),
        "native_runtime_refs": [adapter.grid.file_ref(runtime), adapter.grid.file_ref(cuda)],
        "static_hardware": {"schema": "native-selected-cell-hardware/v1", "frozen_hardware": hardware,
            "configured_clock": {"expected_gpu_sm_clock_mhz": 2400}, "state_refs": [{"raw_ref": raw_ref}]},
        "native_actuals": actuals, "metrics": {"ttft": {"native_median_ms": 12}, "tpot": {"native_median_ms": 24}, "e2e": {"native_median_ms": 36}}}
    selection = {"schema": selector.SCHEMA, "selected_count": 1, "planned_cells": 162,
        "selected_cell_ids": [row["cell_id"]], "selected_cells": [row],
        "coverage": [{"model_key": group, "placement_group": group, "planned_cells": 27,
            "selected_cells": int(group == "qwen38_gpu"),
            "excluded_cells": 27 - int(group == "qwen38_gpu")} for group in selector.GROUPS]}
    seal(selection)
    selection_path = tmp_path / "selection.json"
    document(selection_path, selection)
    calls = {"build": [], "run": [], "replan": []}
    monkeypatch.setattr(adapter.grid, "read_gguf_metadata", lambda path: SimpleNamespace(
        sha256=adapter.grid.file_ref(path)["sha256"], architecture="fixture_arch", n_layer=64,
        metadata={"fixture_arch.block_count": 65, "fixture_arch.nextn_predict_layers": 1}))
    monkeypatch.setattr(adapter.grid, "build_model_from_gguf", lambda gguf: SimpleNamespace(name="GGUF-qwen35", architecture="qwen35", num_layers=64))
    def build(*args, **kwargs):
        calls["build"].append((args, kwargs))
        return Scenario()
    def run(scenario, **kwargs):
        assert scenario.hardware.metadata.get("test_replanned_after_static_bindings") is True
        calls["run"].append(scenario)
        return SimpleNamespace(metrics=SimpleNamespace(request_metrics={f"r{i}": SimpleNamespace(visible_output_tokens=2) for i in range(2)}))
    def replan(scenario, *, recurrent_batching_contract=None, slot_order_contract=None):
        from dataclasses import replace
        calls["replan"].append(scenario)
        calls["recurrent_contract"] = recurrent_batching_contract
        calls["slot_order_contract"] = slot_order_contract
        updated = replace(scenario, hardware=replace(scenario.hardware, metadata={
            **scenario.hardware.metadata, "test_replanned_after_static_bindings": True}))
        return updated, {"normal_validation_passed": True, "test_stub": True}
    monkeypatch.setattr(adapter, "replan_final_static_scenario", replan)
    monkeypatch.setattr(adapter.grid, "build_matching_scenario", build)
    monkeypatch.setattr(adapter.grid.reporting, "run_scenario", run)
    monkeypatch.setattr(adapter.grid, "_simulator_request_timing", lambda result, metric: {"engine_ttft_ms": 18, "engine_tpot_ms": 30, "engine_e2e_ms": 48})
    monkeypatch.setattr(adapter, "source_freeze", lambda dest: {"root": str(dest), "sha256": "test-source", "files": []})
    return selection_path, selection, row, calls


def test_real_selector_schema_and_sampled_clock_map_without_answers(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("native execution/calibration forbidden")
    from tools import native_llama_compare
    for name in ("probe_hardware", "load_native_calibration", "apply_native_calibration", "post_json"):
        monkeypatch.setattr(native_llama_compare, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    assert "native_actuals" not in inputs and "metrics" not in inputs
    assert "native_latency_ms" not in json.dumps(inputs)
    result = adapter.predict_cell(inputs)
    kwargs = calls["build"][0][1]
    assert kwargs["gpu_layers"] == 65 and kwargs["ctx"] == 2048 and kwargs["parallel"] == 2
    assert kwargs["runtime_binary"] == Path(row["native_runtime_refs"][0]["path"])
    tensor = calls["run"][0].component_profiles["gpu"]["analytical"].tensor_core
    assert tensor.frequency_ghz == 2.392 and tensor.sm_count == 84 and tensor.cycles_per_mma == 2
    assert calls["run"][0].workload.metadata["serving_runtime"] == {"kv_slot_context_tokens": 2048, "compiled_cuda_graphs": False, "cuda_graph_replay_cost_applied": False}
    assert result["input_identity"]["native_total_context_tokens"] == 4096
    assert result["input_identity"]["native_configuration"]["gpu_layers"] == 66
    assert result["input_identity"]["native_configuration"]["fit_params"] is False
    mapping = result["input_identity"]["gpu_loading_unit_mapping"]
    assert mapping["native_gpu_layers"] == 66 and mapping["simulator_gpu_layers"] == 65
    assert mapping["model_executable_layers"] == 64 and mapping["excluded_mtp_loading_units"] == 1
    assert mapping["mapping_applied"] is True
    assert result["status"] == "predicted" and result["calibration_applied"] is False
    assert result["formal_prediction_eligible"] is False
    row["native_actuals"][0]["metrics_ms"]["ttft"] = 1e99
    assert adapter.predict_cell(adapter.static_inputs(row, selection, tmp_path))["aggregate"] == result["aggregate"]


def test_frozen_target_never_uses_prelock_clock(tmp_path, monkeypatch):
    _, selection, row, _ = fixture(tmp_path, monkeypatch)
    row["static_hardware"]["state_refs"] = []
    clock = adapter.gpu_clock(adapter.static_inputs(row, selection, tmp_path))
    assert clock["mhz"] == 2400 and "without_sampled_readback" in clock["source"]


def test_auto_gpu_layers_is_explicitly_conditional(tmp_path, monkeypatch):
    _, selection, row, _ = fixture(tmp_path, monkeypatch)
    row["config"]["gpu_layers"] = -1
    result = adapter.predict_cell(adapter.static_inputs(row, selection, tmp_path))
    assert "auto_gpu_layer_fit" in {r["dimension"] for r in result["unsupported_dimensions"]}


def test_bad_context_and_prompt_rejected(tmp_path, monkeypatch):
    _, selection, row, _ = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    inputs["config"]["ctx"] = 2048
    with pytest.raises(ValueError, match="2048 \\* parallel"):
        adapter.predict_cell(inputs)
    inputs["config"]["ctx"] = 4096
    inputs["config"]["prompt_token_ids"].pop()
    with pytest.raises(ValueError, match="token IDs"):
        adapter.predict_cell(inputs)


def test_freeze_precedes_prediction_and_worker_never_reads_answers(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    assert not calls["run"]
    freeze, _ = adapter.grid.read_document(out / "freeze.json")
    assert freeze["selected_denominator"] == 1 and freeze["native_grid_denominator"] == 162
    assert "native_actuals" not in json.dumps(freeze)
    result_path = out / "predictions" / (row["cell_id"] + ".prediction.json")
    adapter.worker_cell(out / "freeze.json", row["cell_id"], result_path)
    result, _ = adapter.grid.read_document(result_path)
    assert result["created_utc"] >= freeze["created_utc"]
    assert result["status"] == "predicted"
    report = adapter.score_predictions(out)
    metric = report["cells"][0]["metrics"]["engine_ttft_ms"]
    assert metric["native_median_ms"] == 12 and metric["signed_error_pct"] == 50
    assert metric["absolute_error_ms"] == 6
    assert report["overall"]["engine_ttft_ms"]["signed_error_pct"]["median"] == 50
    assert report["strict_gate"]["verdict"] == "insufficient_evidence"
    assert report["strict_gate"]["insufficient_evidence_cells"] == 1
    assert len(calls["run"]) == 1  # scoring never reruns or alters predictions


def test_soft_deadline_keeps_late_success_and_resume_preserves_result(tmp_path, monkeypatch):
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    launches=[]
    class NaturalWorker:
        pid=12345
        returncode=None
        def __init__(self, command, **kwargs):self.command=command;self.waits=0;launches.append(command)
        def wait(self, timeout=None):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired(self.command,timeout)
            command=self.command
            adapter.worker_cell(Path(command[command.index("--worker-freeze")+1]),command[command.index("--worker-cell")+1],Path(command[command.index("--worker-result")+1]))
            self.returncode=0;return 0
        def poll(self):return self.returncode
        def kill(self):pytest.fail("must never kill")
        def terminate(self):pytest.fail("must never terminate")
    monkeypatch.setattr(subprocess,"Popen",NaturalWorker)
    result = adapter.run_predictions(out, timeout_seconds=.01)
    assert result["selected_denominator"] == result["successful_cells"] == 1
    before = result["cells"][0]["prediction_ref"]
    record,_=adapter.grid.read_document(before["path"])
    evidence=adapter.verify_worker_execution(record,before["path"])
    assert evidence["late"] and evidence["natural_exit_observed"] and not evidence["hard_time_limit_enforced"]
    resumed = adapter.run_predictions(out, resume=True)
    assert resumed["cells"][0]["prediction_ref"] == before and len(launches)==1
    assert len(calls["run"])==1
    score = adapter.score_predictions(out)
    assert score["overall"]["engine_ttft_ms"]["selected_cells"] == 1
    assert score["overall"]["engine_ttft_ms"]["missing_cells"] == 0


def test_missing_static_inputs_remain_failed_not_excluded(tmp_path, monkeypatch):
    path, selection, row, _ = fixture(tmp_path, monkeypatch)
    del row["config"]["gpu_layers"]
    document(path, seal(selection))
    freeze = adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path)
    assert freeze["selected_denominator"] == 1
    assert "gpu_layers" in freeze["cells"][0]["preparation_error"]


def test_selection_tampering_refuses_resume(tmp_path, monkeypatch):
    path, selection, _, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    selection["selected_count"] = 0
    document(path, selection)
    with pytest.raises(ValueError, match="frozen reference changed"):
        adapter.run_predictions(out, resume=True)


def test_duplicate_ids_and_wrong_schema_are_rejected(tmp_path, monkeypatch):
    _, selection, row, _ = fixture(tmp_path, monkeypatch)
    selection["selected_cells"].append(row)
    selection["selected_count"] = 2
    seal(selection)
    with pytest.raises(ValueError, match="duplicate"):
        adapter.selected_rows(selection)
    selection["schema"] = "other"
    with pytest.raises(ValueError, match="native-stable-dataset"):
        adapter.selected_rows(selection)



def test_payload_tamper_rejected_even_when_file_sha_is_new(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    row["native_actuals"][0]["metrics_ms"]["ttft"] = 999
    document(path, selection)  # valid JSON/new file SHA cannot replace payload binding
    with pytest.raises(ValueError, match="selection payload SHA256"):
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path)
    assert not calls["run"]


def test_selected_ids_must_match_rows_even_with_valid_payload(tmp_path, monkeypatch):
    path, selection, _, _ = fixture(tmp_path, monkeypatch)
    selection["selected_cell_ids"] = ["some_other_same_count_cell"]
    document(path, seal(selection))
    with pytest.raises(ValueError, match="selected_cell_ids"):
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path)


def test_external_native_report_cannot_replace_frozen_truth(tmp_path, monkeypatch):
    path, selection, _, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    alternate = copy.deepcopy(selection)
    alternate["selected_cells"][0]["native_actuals"][0]["metrics_ms"]["ttft"] = 9000
    alternate_path = tmp_path / "alternate_truth.json"
    document(alternate_path, seal(alternate))
    with pytest.raises(ValueError, match="input SHA256 mismatch"):
        adapter.score_predictions(out, native_report=alternate_path)
    assert not calls["run"]


def test_identical_truth_copy_allowed_and_all_six_empty_groups_retained(tmp_path, monkeypatch):
    path, selection, row, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    freeze = adapter.freeze_selection(path, out, data_root=tmp_path)
    assert freeze["coverage"] == selection["coverage"]
    adapter.worker_cell(out / "freeze.json", row["cell_id"], out / "predictions" / (row["cell_id"] + ".prediction.json"))
    copied = tmp_path / "same_truth_bytes.json"
    copied.write_bytes(path.read_bytes())
    report = adapter.score_predictions(out, native_report=copied)
    assert set(report["by_model_deployment"]) == set(selector.GROUPS)
    assert set(report["by_model"]) == set(selector.GROUPS)
    empty = report["by_model_deployment"]["qwen25"]
    assert empty["native_grid_planned_cells"] == 27 and empty["native_selected_cells"] == 0
    assert empty["engine_ttft_ms"]["selected_cells"] == 0
    assert empty["engine_ttft_ms"]["absolute_error_ms"] == {"median": None, "p90": None, "max": None}
    assert report["by_model_deployment"]["qwen38_gpu"]["engine_ttft_ms"]["scored_cells"] == 1


def test_bounded_workers_keep_selection_order_and_separate_run_receipts(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    rows = []
    for index in range(5):
        item = copy.deepcopy(row)
        item["cell_id"] += f"_{index:02d}"
        rows.append(item)
    selection.update(selected_cells=rows, selected_count=len(rows), selected_cell_ids=[r["cell_id"] for r in rows])
    for group in selection["coverage"]:
        group["selected_cells"] = len(rows) if group["model_key"] == "qwen38_gpu" else 0
        group["excluded_cells"] = 27 - group["selected_cells"]
    document(path, seal(selection))
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    freeze_before = adapter.grid.file_ref(out / "freeze.json")
    lock, barrier, second_finished = threading.Lock(), threading.Barrier(2), threading.Event()
    tracker = {"live": 0, "peak": 0}
    def fake_run(command, **kwargs):
        cell_id = command[command.index("--worker-cell") + 1]
        index = int(cell_id.rsplit("_", 1)[1])
        with lock:
            tracker["live"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["live"])
        try:
            if index < 2:
                barrier.wait(timeout=5)
            if index == 0:
                assert second_finished.wait(timeout=5)
            adapter.worker_cell(Path(command[command.index("--worker-freeze") + 1]), cell_id,
                Path(command[command.index("--worker-result") + 1]))
            if index == 1:
                second_finished.set()
            return SimpleNamespace(returncode=0)
        finally:
            with lock:
                tracker["live"] -= 1
    class FakeWorker:
        pid=12345
        returncode=None
        def __init__(self,command,**kwargs):self.command=command;self.kwargs=kwargs
        def wait(self,timeout=None):self.returncode=fake_run(self.command,**self.kwargs).returncode;return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(subprocess, "Popen", FakeWorker)
    result = adapter.run_predictions(out, workers=2, timeout_seconds=17)
    assert tracker == {"live": 0, "peak": 2}
    assert result["successful_cells"] == result["selected_denominator"] == 5
    assert [r["cell_id"] for r in result["cells"]] == selection["selected_cell_ids"]
    start, _ = adapter.grid.read_document(out / "runs/run.0001.start.json")
    finish, _ = adapter.grid.read_document(out / "runs/run.0001.finish.json")
    assert start["execution_budget"]["workers"] == finish["execution_budget"]["workers"] == 2
    assert start["execution_budget"]["per_cell_soft_observation_seconds"] == 17
    assert [r["cell_id"] for r in finish["cells"]] == selection["selected_cell_ids"]
    assert adapter.grid.file_ref(out / "freeze.json") == freeze_before
    assert not (out / "runs/coordinator.lock").exists()
    assert len(calls["run"]) == 5


@pytest.mark.parametrize("workers", [0, -1, 9, True])
def test_worker_bound_rejected_before_any_execution(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        adapter.run_predictions(tmp_path, workers=workers)



def test_zero_selected_cells_still_report_all_native_groups(tmp_path, monkeypatch):
    path, selection, _, calls = fixture(tmp_path, monkeypatch)
    selection.update(selected_cells=[], selected_count=0, selected_cell_ids=[])
    for group in selection["coverage"]:
        group.update(selected_cells=0, excluded_cells=27)
    document(path, seal(selection))
    out = tmp_path / "out"
    freeze = adapter.freeze_selection(path, out, data_root=tmp_path)
    assert freeze["selected_denominator"] == 0 and freeze["native_grid_denominator"] == 162
    report = adapter.score_predictions(out)
    assert set(report["by_model_deployment"]) == set(selector.GROUPS)
    assert all(group["native_selected_cells"] == 0 for group in report["by_model_deployment"].values())
    assert not calls["run"]



@pytest.mark.parametrize("raw_blocks,mtp,imported,native,fit,reason", [
    (64, 0, 64, 66, False, "without positive GGUF MTP evidence"),
    (None, 0, 64, 66, False, "without positive GGUF MTP evidence"),
    (65, 1, 63, 66, False, "counts disagree"),
    (66, 1, 64, 67, False, "counts disagree"),
    (65, 1, 64, 67, False, "not the exact GGUF full-offload count"),
    (65, 1, 64, 66, True, "fit_params=false"),
    (65, 1, 64, 66, None, "fit_params=false"),
])
def test_loading_count_never_clamped_without_exact_mtp_evidence(raw_blocks, mtp, imported, native, fit, reason):
    gguf = SimpleNamespace(sha256="fixture-sha", architecture="unrelated_architecture", n_layer=imported,
        metadata={"unrelated_architecture.block_count": raw_blocks,
                  "unrelated_architecture.nextn_predict_layers": mtp})
    inputs = {"config": {"gpu_layers": native, "fit_params": fit}}
    model = SimpleNamespace(name="not-a-known-model-name", num_layers=64)
    with pytest.raises(ValueError, match=reason):
        adapter.gpu_layer_mapping(inputs, gguf, model)


def test_mtp_mapping_depends_on_structural_counts_not_27b_model_name():
    gguf = SimpleNamespace(sha256="fixture-sha", architecture="new_arch", n_layer=7,
        metadata={"new_arch.block_count": 9, "new_arch.nextn_predict_layers": 2})
    inputs = {"config": {"gpu_layers": 10, "fit_params": False}}
    result = adapter.gpu_layer_mapping(inputs, gguf, SimpleNamespace(name="other", num_layers=7))
    assert result["native_full_loading_units"] == 10 and result["simulator_gpu_layers"] == 8
    assert result["excluded_mtp_loading_units"] == 2
    assert result["gguf_sha256"] == "fixture-sha"


@pytest.mark.parametrize("native", [-1, 0, 1, 32, 65])
def test_non_mtp_valid_loading_counts_are_preserved(native):
    gguf = SimpleNamespace(sha256="fixture-sha", architecture="dense", n_layer=64,
        metadata={"dense.block_count": 64})
    inputs = {"config": {"gpu_layers": native, "fit_params": False}}
    result = adapter.gpu_layer_mapping(inputs, gguf, SimpleNamespace(num_layers=64))
    assert result["native_gpu_layers"] == result["simulator_gpu_layers"] == native
    assert result["excluded_mtp_loading_units"] == 0 and result["mapping_applied"] is False



@pytest.mark.parametrize("gpu_layers", [0, -1])
def test_real_static_scenario_replans_after_frequency_binding_without_bypassing_validation(gpu_layers):
    from dataclasses import replace
    from tools.native_llama_compare import build_matching_scenario
    from heterollm_sim.control_plane_state import mapping_fingerprint_status
    from heterollm_sim.planner import validate_scenario, ScenarioValidationError
    case = build_matching_scenario(4, 2, ctx=2048, parallel=1, batch=64, ubatch=64,
        threads=16, gpu_layers=gpu_layers,
        runtime_binary=adapter.ROOT / adapter.grid.RUNTIME)
    original = mapping_fingerprint_status(case)
    assert original["mapping_stale"] is False
    profiles = {kind: dict(values) for kind, values in case.component_profiles.items()}
    for key, profile in profiles["gpu"].items():
        profiles["gpu"][key] = replace(profile, tensor_core=replace(profile.tensor_core, frequency_ghz=2.392))
    changed = replace(case, component_profiles=profiles,
        hardware=replace(case.hardware, metadata={**case.hardware.metadata,
            "frozen_native_gpu_clock": {"mhz": 2392, "frequency_ghz": 2.392}}))
    assert mapping_fingerprint_status(changed)["mapping_stale"] is True
    with pytest.raises(ScenarioValidationError):
        validate_scenario(changed).raise_for_errors()
    refreshed, evidence = adapter.replan_final_static_scenario(changed)
    status = mapping_fingerprint_status(refreshed)
    assert evidence["previous_decision_stale"] is True
    assert evidence["normal_validation_passed"] is True
    assert status["mapping_stale"] is False
    assert status["input_fingerprint"] == status["current_input_fingerprint"]
    assert status["input_fingerprint"] != original["input_fingerprint"]
    assert validate_scenario(refreshed).is_valid
    assert refreshed.placement.metadata["control_plane"]["policy"] == case.placement.metadata["control_plane"]["policy"]
    assert refreshed.placement.metadata["control_plane"]["decision"]["generated_op_keys"]
    assert refreshed.placement.metadata["llama_cpp_kv_layer_components"] == case.placement.metadata["llama_cpp_kv_layer_components"]
    # The normal identity gate must still reject any later semantic edit.
    invalidated = replace(refreshed, weights_resident=not refreshed.weights_resident)
    assert mapping_fingerprint_status(invalidated)["mapping_stale"] is True
    with pytest.raises(ScenarioValidationError):
        validate_scenario(invalidated).raise_for_errors()


def test_gguf_hash_failure_records_actual_expected_and_both_paths(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    actual = "f" * 64
    monkeypatch.setattr(adapter.grid, "read_gguf_metadata", lambda path: SimpleNamespace(sha256=actual))
    with pytest.raises(ValueError, match="GGUF SHA256 mismatch") as error:
        adapter.predict_cell(inputs)
    message = str(error.value)
    assert actual in message and row["model_ref"]["sha256"] in message
    assert row["model_ref"]["path"] in message and "native_model_path=" in message
    assert not calls["run"]


def test_explicit_identical_model_snapshot_keeps_native_identity(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    native = Path(row["model_ref"]["path"])
    copied = tmp_path / "prediction-readonly.gguf"
    copied.write_bytes(native.read_bytes())
    selection_before = adapter.grid.file_ref(path)
    out = tmp_path / "out"
    freeze = adapter.freeze_selection(path, out, data_root=tmp_path,
        model_snapshot_map={str(native): str(copied)})
    inputs = freeze["cells"][0]["static_inputs"]
    assert inputs["config"]["model"] == str(native)
    assert inputs["native_model_ref"] == row["model_ref"]
    assert inputs["prediction_model_ref"]["path"] == str(copied)
    assert inputs["prediction_model_ref"]["sha256"] == row["model_ref"]["sha256"]
    observed, reader = [], adapter.grid.read_gguf_metadata
    def read_model(source):
        observed.append(str(source))
        return reader(source)
    monkeypatch.setattr(adapter.grid, "read_gguf_metadata", read_model)
    result = adapter.predict_cell(inputs)
    assert result["status"] == "predicted" and observed == [str(copied)]
    assert result["input_identity"]["native_configuration"]["model"] == str(native)
    assert result["input_identity"]["model"]["path"] == str(copied)
    assert adapter.grid.file_ref(path) == selection_before


def test_different_model_snapshot_is_rejected_before_any_simulation(tmp_path, monkeypatch):
    path, _, row, calls = fixture(tmp_path, monkeypatch)
    copied = tmp_path / "wrong-copy.gguf"
    copied.write_bytes(b"not identical model bytes")
    with pytest.raises(ValueError, match="model snapshot SHA256 mismatch") as error:
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path,
            model_snapshot_map={row["model_ref"]["path"]: str(copied)})
    assert row["model_ref"]["sha256"] in str(error.value)
    assert adapter.grid.file_ref(copied)["sha256"] in str(error.value)
    assert not calls["run"]


def test_cell_filter_runs_only_requested_pilots_then_resumes_full_denominator(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    rows = []
    for index in range(3):
        item = copy.deepcopy(row)
        item["cell_id"] += f"_pilot_{index}"
        rows.append(item)
    selection.update(selected_cells=rows, selected_count=3,
        selected_cell_ids=[r["cell_id"] for r in rows])
    for group in selection["coverage"]:
        group["selected_cells"] = 3 if group["model_key"] == "qwen38_gpu" else 0
        group["excluded_cells"] = 27 - group["selected_cells"]
    document(path, seal(selection))
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    frozen = adapter.grid.file_ref(out / "freeze.json")
    def fake_run(command, **kwargs):
        adapter.worker_cell(Path(command[command.index("--worker-freeze") + 1]),
            command[command.index("--worker-cell") + 1],
            Path(command[command.index("--worker-result") + 1]))
        return SimpleNamespace(returncode=0)
    class FakeWorker:
        pid=12345
        returncode=None
        def __init__(self,command,**kwargs):self.command=command;self.kwargs=kwargs
        def wait(self,timeout=None):self.returncode=fake_run(self.command,**self.kwargs).returncode;return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(subprocess, "Popen", FakeWorker)
    requested = [rows[2]["cell_id"], rows[0]["cell_id"]]
    first = adapter.run_predictions(out, workers=2, cell_ids=requested)
    assert first["selected_denominator"] == 3 and first["successful_cells"] == 2 and first["pending_cells"] == 1
    assert [r["status"] for r in first["cells"]] == ["predicted", "pending", "predicted"]
    start, _ = adapter.grid.read_document(out / "runs/run.0001.start.json")
    assert start["cell_id_filter"] == requested
    assert start["scheduled_cell_ids"] == [rows[0]["cell_id"], rows[2]["cell_id"]]
    final = adapter.run_predictions(out, resume=True, workers=1)
    assert final["successful_cells"] == final["selected_denominator"] == 3 and not final["pending_cells"]
    assert final["cells"][0]["prediction_ref"] == first["cells"][0]["prediction_ref"]
    assert final["cells"][2]["prediction_ref"] == first["cells"][2]["prediction_ref"]
    assert adapter.grid.file_ref(out / "freeze.json") == frozen and len(calls["run"]) == 3


def test_unknown_pilot_cell_rejected_before_worker_execution(tmp_path, monkeypatch):
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("unknown cell launched a worker"))
    with pytest.raises(ValueError, match="cell-id is outside frozen selection"):
        adapter.run_predictions(out, cell_ids=["outside_the_frozen_selection"])
    assert not calls["run"] and not (out / "runs").exists()



def timing_fixture(begin=0, first=20_000_000, last=100_000_000, request_id="r0"):
    metric = SimpleNamespace(request_id=request_id, arrival_ns=0, start_ns=1_000_000,
        first_token_ns=first, finish_ns=last + 1_000_000, visible_output_tokens=2)
    timing = {"engine_request_begin_ns": begin, "engine_last_token_ns": last,
        "engine_start_source": "first_prompt_batch_processing", "engine_last_token_source": "accumulator_with_partial_event_retention"}
    request = {"request_id": request_id, "visible_output_tokens": 2,
        "engine_ttft_ms": (first - begin) / 1e6, "engine_tpot_ms": (last - first) / 1e6,
        "engine_e2e_ms": (last - begin) / 1e6}
    events = [SimpleNamespace(timestamp_ns=begin, event_type="engine_request_begin", request_id=request_id,
        cohort_id="batch0", details={"boundary": "first_prompt_batch_processing"})]
    result = SimpleNamespace(serving=SimpleNamespace(events=events))
    return result, metric, timing, request


def test_absolute_engine_timepoints_are_observed_and_preserve_zero_origin():
    result, metric, timing, request = timing_fixture()
    fields = adapter.request_timepoints(result, metric, timing, request)
    assert fields["engine_request_begin_ns"] == fields["first_prompt_batch_processing_ns"] == 0
    assert fields["engine_first_token_ns"] == 20_000_000 and fields["engine_last_token_ns"] == 100_000_000
    assert fields["timepoint_validation"]["status"] == "verified"
    assert fields["timepoint_validation"]["engine_begin_matches_retained_event"] is True
    assert fields["service_finish_ns"] == 101_000_000  # not used as last engine token


def test_missing_or_inferred_timepoints_are_not_backfilled_from_latency():
    result, metric, timing, request = timing_fixture()
    timing.update(engine_request_begin_ns=None, engine_last_token_source="incomplete_token_evidence")
    fields = adapter.request_timepoints(result, metric, timing, request)
    assert fields["engine_request_begin_ns"] is None and fields["engine_last_token_ns"] is None
    assert fields["timepoint_validation"]["status"] == "incomplete"
    assert len(fields["timepoint_validation"]["missing_fields"]) == 2


def test_conflicting_retained_engine_begin_event_is_visible_and_span_rejected():
    result, metric, timing, request = timing_fixture()
    result.serving.events[0].timestamp_ns = 5_000_000
    request.update(adapter.request_timepoints(result, metric, timing, request))
    assert request["timepoint_validation"]["status"] == "inconsistent"
    assert "engine_begin_metric_and_retained_event_disagree" in request["timepoint_validation"]["issues"]
    assert adapter.engine_cohort_span([request])["cohort_engine_span_ms"] is None


def test_serial_requests_keep_whole_cohort_span_separate_from_per_request_latency():
    requests = []
    for i, begin in enumerate((0, 100_000_000)):
        result, metric, timing, request = timing_fixture(begin, begin + 20_000_000, begin + 100_000_000, f"r{i}")
        request.update(adapter.request_timepoints(result, metric, timing, request))
        requests.append(request)
    span = adapter.engine_cohort_span(requests)
    assert all(r["engine_e2e_ms"] == 100 for r in requests)
    assert span["cohort_engine_span_ms"] == 200
    assert span["engine_start_spread_ns"] == 100_000_000
    assert span["max_overlapping_engine_request_intervals"] == 1


def test_batch_schedule_exposes_membership_cost_path_and_prompt_processing_offset():
    result, metric, timing, request = timing_fixture()
    request.update(adapter.request_timepoints(result, metric, timing, request))
    cost = SimpleNamespace(duration_ns=25, metadata={"execution_path": "serial_fallback", "fallback_reason": "stateful batching unqualified"})
    result.serving.batches = [SimpleNamespace(cohort_id="batch0", kind="prefill", start_ns=10, end_ns=35,
        request_ids=("r0",), token_count=4, cost=cost,
        items=(SimpleNamespace(request_id="r0", phase="prefill", token_count=4, context_tokens=0),), metadata={})]
    result.serving.scheduler_metrics = SimpleNamespace(total_batches=1, scheduling_rounds=1)
    schedule = adapter.batch_schedule(result, [request])
    assert schedule["phase_batch_counts"] == {"prefill": 1}
    assert schedule["batches"][0]["cost_metadata"]["execution_path"] == "serial_fallback"
    assert schedule["per_request"]["r0"]["engine_begin_to_first_retained_prefill_batch_ns"] == 10
    assert schedule["scheduler_metrics"]["total_batches"] == 1


def test_diagnostic_events_are_bounded_by_count_and_serialized_bytes(tmp_path):
    events = [SimpleNamespace(timestamp_ns=i, event_type="engine_request_begin", request_id="r0", cohort_id="batch0",
        details={"large": "x" * 4000}) for i in range(100)]
    result = SimpleNamespace(serving=SimpleNamespace(events=events))
    trace = adapter.diagnostic_event_trace(result, limit=7, max_bytes=8192)
    assert trace["returned_events"] <= 7 and trace["truncated"] is True
    assert trace["history_complete"] is None
    ref = adapter.grid.write_new(tmp_path / "events.json", {"schema": "test-diagnostics", **trace})
    assert ref["size_bytes"] <= 8192


def test_diagnostic_event_attachment_does_not_change_primary_prediction(tmp_path, monkeypatch):
    path, _, row, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    plain_path, diagnostic_path = out / "plain.json", out / "diagnostic.json"
    adapter.worker_cell(out / "freeze.json", row["cell_id"], plain_path)
    adapter.worker_cell(out / "freeze.json", row["cell_id"], diagnostic_path, diagnostic_events=True, diagnostic_event_limit=3)
    plain, _ = adapter.grid.read_document(plain_path)
    diagnostic, _ = adapter.grid.read_document(diagnostic_path)
    assert plain["aggregate"] == diagnostic["aggregate"]
    assert plain["input_identity"] == diagnostic["input_identity"]
    assert "diagnostic_events_ref" not in plain and "_diagnostic_events" not in diagnostic
    events, _ = adapter.grid.read_document(diagnostic["diagnostic_events_ref"]["path"])
    assert events["status"] == "unavailable" and events["events"] == []


def test_auxiliary_native_cohort_span_uses_engine_clock_and_three_run_median():
    actuals = []
    for repeat in range(3):
        epoch = (repeat + 1) * 1_000_000
        for index, offset in enumerate((0, 100_000)):
            actuals.append({"block": 0, "repeat": repeat, "request_index": index,
                "engine_request_begin_us": epoch + offset,
                "engine_first_token_us": epoch + offset + 20_000,
                "engine_last_token_us": epoch + offset + 100_000,
                "client_e2e_ms": 999999})
    prediction = {"cohort_engine_timeline": {"status": "complete", "cohort_engine_span_ms": 250}}
    result = adapter.cohort_engine_comparison(prediction, {"parallel": 2, "native_actuals": actuals})
    assert result["native_run_engine_spans_ms"] == [200, 200, 200]
    assert result["native_median_cohort_engine_span_ms"] == 200
    assert result["signed_error_pct"] == 25 and result["primary_metrics_changed"] is False
    actuals[0].pop("engine_first_token_us")
    assert adapter.cohort_engine_comparison(prediction, {"parallel": 2, "native_actuals": actuals})["status"] == "unavailable"


def test_recurrent_hook_is_explicit_and_does_not_change_default_inputs(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    adapter.predict_cell(inputs)
    assert calls["recurrent_contract"] is None
    treatment = {"schema": "test-contract-forwarding-only"}
    inputs["recurrent_batching_contract"] = treatment
    adapter.predict_cell(inputs)
    assert calls["recurrent_contract"] is treatment


def test_existing_overlay_build_audit_is_verified_without_large_binary_hashes(monkeypatch):
    audit_path = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_simulation_v2/runtime_build_audit.json"
    if not audit_path.exists():
        pytest.skip("local recorded native build audit unavailable")
    original = adapter.grid.file_ref
    def small_only(path):
        assert Path(path).suffix.lower() not in {".dll", ".exe", ".gguf", ".obj", ".lib"}
        return original(path)
    monkeypatch.setattr(adapter.grid, "file_ref", small_only)
    proof = adapter.verified_runtime_build_audit(audit_path, adapter.ROOT)
    assert proof["compiled_cuda_graphs"] is False
    assert proof["binding"] == "verified_annotation_to_native_overlay_CUDA_build_chain"
    assert proof["runtime_dispatch_or_cost_parity_proven"] is False
    native_refs = [{"path": "ggml-cuda.dll", "sha256": proof["artifact_sha256"]}]
    assert adapter.compiled_graph_evidence(native_refs, adapter.ROOT, verified_audit=proof)["compiled_cuda_graphs"] is False
    native_refs[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match selected CUDA"):
        adapter.compiled_graph_evidence(native_refs, adapter.ROOT, verified_audit=proof)


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_existing_recurrent_contract_is_rederived_and_bound_to_captured_runtime(tmp_path):
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    contract_path = base / "optimization_loop/round_001/recurrent_source_contract.json"
    if not contract_path.exists():
        pytest.skip("local recorded recurrent source contract unavailable")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    proof = adapter.verified_recurrent_batching_contract(contract_path, adapter.selected_rows(selection), adapter.ROOT)
    assert proof["contract"]["status"] == "source_derived"
    assert proof["contract"]["native_latency_used"] is False
    assert "llama-server-impl.dll" in proof["native_runtime_sha256"]
    tampered = copy.deepcopy(proof["contract"])
    tampered["captured_sequence_capacity"] = 999
    changed = tmp_path / "changed-contract.json"
    document(changed, tampered)
    with pytest.raises(ValueError, match="does not equal re-derived"):
        adapter.verified_recurrent_batching_contract(changed, adapter.selected_rows(selection), adapter.ROOT)



def local_iq_panel_contract():
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    contract = base / "optimization_loop/round_002/iq_panel_source_contract.json"
    if not contract.exists():
        pytest.skip("local native IQ panel source contract unavailable")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    return contract, adapter.selected_rows(selection)


def test_iq_panel_default_is_off_and_no_global_dtype_layout_facts_are_injected(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    adapter.predict_cell(inputs)
    assert inputs["cpu_iq_panel_reuse"] is None
    assert "llama_cpp_cpu_iq_panel_reuse" not in calls["run"][-1].workload.metadata
    dispatch = {"enabled": True, "compiled_avx2": True,
        "no_iq_panel_environment_state": "unknown", "assume_default_unset": True,
        "source_sha256": "source-sha", "cpu_backend_sha256": "backend-sha",
        "source_refs": ["frozen-source-contract"], "native_dispatch_proven": False,
        "evaluation_scope": "conditional_default_unset_ablation"}
    inputs["cpu_iq_panel_reuse"] = dispatch
    candidate = adapter.predict_cell(inputs)
    applied = calls["run"][-1].workload.metadata["llama_cpp_cpu_iq_panel_reuse"]
    assert applied == dispatch
    for key in ("source_activation_dtype", "source_output_dtype", "source_weight_layout", "source_activation_ne3"):
        assert key not in applied
    assert candidate["input_identity"]["cpu_iq_panel_reuse"]["native_dispatch_proven"] is False
    reason = next(item for item in candidate["unsupported_dimensions"] if item["dimension"] == "cpu_iq_panel_historical_dispatch")
    assert reason["evaluation_scope"] == "conditional_default_unset_ablation"


def test_iq_panel_assumption_requires_explicit_contract(tmp_path, monkeypatch):
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires an explicit source contract"):
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path, iq_panel_assume_default_unset=True)
    assert not calls["run"]


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_iq_panel_unknown_history_never_becomes_unset_from_current_environment(monkeypatch):
    path, rows = local_iq_panel_contract()
    monkeypatch.setenv("GGML_NO_IQ_PANEL", "0")
    historical = adapter.verified_iq_panel_source_contract(path, rows, adapter.ROOT)
    assumed = adapter.verified_iq_panel_source_contract(path, rows, adapter.ROOT, assume_default_unset=True)
    assert historical["dispatch"]["no_iq_panel_environment_state"] == "unknown"
    assert historical["dispatch"]["assume_default_unset"] is False
    assert assumed["dispatch"]["no_iq_panel_environment_state"] == "unknown"
    assert assumed["dispatch"]["assume_default_unset"] is True
    assert assumed["evaluation_scope"] == "conditional_default_unset_ablation"
    assert historical["native_dispatch_proven"] is assumed["native_dispatch_proven"] is False
    assert assumed["contract"]["today_environment_read"] is False
    assert assumed["contract"]["native_latency_used"] is False


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_iq_panel_source_contract_rejects_promoted_history_even_with_new_content_hash(tmp_path):
    path, rows = local_iq_panel_contract()
    changed, _ = adapter.grid.read_document(path)
    changed.pop("content_sha256")
    changed["no_iq_panel_environment_state"] = "unset"
    forged = tmp_path / "forged-iq-contract.json"
    adapter.grid.write_new(forged, changed)
    with pytest.raises(ValueError, match="differs from re-derived"):
        adapter.verified_iq_panel_source_contract(forged, rows, adapter.ROOT, assume_default_unset=True)


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_iq_panel_source_contract_rejects_different_selected_cpu_backend():
    path, rows = local_iq_panel_contract()
    row = copy.deepcopy(rows[0])
    cpu = next(ref for ref in row["native_runtime_refs"] if Path(ref["path"]).name.lower() == "ggml-cpu.dll")
    cpu["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CPU backend differs"):
        adapter.verified_iq_panel_source_contract(path, [row], adapter.ROOT, assume_default_unset=True)


def fake_verified_iq_contract(tmp_path):
    reference = document(tmp_path / "test-iq-source.json", {"source": "test-only"})
    dispatch = {"enabled": True, "compiled_avx2": True, "no_iq_panel_environment_state": "unknown",
        "assume_default_unset": True, "source_sha256": "source-sha", "cpu_backend_sha256": "cpu-sha",
        "source_refs": [reference["path"]], "native_dispatch_proven": False,
        "evaluation_scope": "conditional_default_unset_ablation"}
    return {"contract": {"schema": adapter.IQ_PANEL_SOURCE_SCHEMA}, "contract_ref": reference,
        "dispatch": dispatch, "evidence_refs": [reference], "native_dispatch_proven": False,
        "evaluation_scope": "conditional_default_unset_ablation"}


def test_iq_panel_freeze_binds_explicit_assumption_and_source_separately_from_native_selection(tmp_path, monkeypatch):
    from heterollm_sim import cost_models
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    selection_before = adapter.grid.file_ref(path)
    proof = fake_verified_iq_contract(tmp_path)
    received = []
    def verify(*args, **kwargs):
        received.append(kwargs["assume_default_unset"])
        return proof
    monkeypatch.setattr(adapter, "verified_iq_panel_source_contract", verify)
    monkeypatch.setattr(cost_models, "CPUIQPanelDispatch", object, raising=False)
    monkeypatch.setattr(cost_models, "estimate_cpu_gemm", lambda cpu, memory, workload, iq_panel_dispatch=None: None)
    frozen = adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path,
        iq_panel_source_contract_path=Path(proof["contract_ref"]["path"]), iq_panel_assume_default_unset=True)
    assert received == [True]
    assert frozen["cpu_iq_panel_reuse"]["native_dispatch_proven"] is False
    assert frozen["cells"][0]["static_inputs"]["cpu_iq_panel_reuse"] == proof["dispatch"]
    assert adapter.grid.file_ref(path) == selection_before and not calls["run"]


def test_iq_panel_freeze_fails_if_candidate_cost_api_is_missing(tmp_path, monkeypatch):
    from heterollm_sim import cost_models
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    proof = fake_verified_iq_contract(tmp_path)
    monkeypatch.setattr(adapter, "verified_iq_panel_source_contract", lambda *args, **kwargs: proof)
    monkeypatch.delattr(cost_models, "CPUIQPanelDispatch", raising=False)
    with pytest.raises(ValueError, match="requires the opt-in CPU cost/planner implementation"):
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path,
            iq_panel_source_contract_path=Path(proof["contract_ref"]["path"]), iq_panel_assume_default_unset=True)
    assert not calls["run"]


def test_iq_panel_treatment_cannot_be_added_to_an_existing_freeze_by_cli(tmp_path, monkeypatch):
    path, _, _, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    before = adapter.grid.file_ref(out / "freeze.json")
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--output", str(out), "--resume", "--iq-panel-assume-default-unset"])
    assert adapter.grid.file_ref(out / "freeze.json") == before



def test_slot_order_default_off_and_explicit_contract_is_forwarded_without_forcing_policy(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    baseline = adapter.predict_cell(inputs)
    assert inputs["slot_order_contract"] is None and calls["slot_order_contract"] is None
    assert baseline["input_identity"]["slot_order_treatment"]["requested"] is False
    contract = {"schema": "test-slot-hook-only", "scope": "fresh_same_arrival_cohort_only"}
    inputs["slot_order_contract"] = contract
    candidate = adapter.predict_cell(inputs)
    assert calls["slot_order_contract"] is contract
    assert candidate["input_identity"]["slot_order_treatment"]["requested"] is True
    assert candidate["input_identity"]["slot_order_treatment"]["applied"] is False
    assert candidate["input_identity"]["slot_order_treatment"]["qualified"] is False
    assert candidate["input_identity"]["slot_order_treatment"]["phase_candidate_order_after_lowering"] is None
    assert "phase_candidate_order" not in calls["replan"][-1].workload.metadata
    assert baseline["aggregate"] == candidate["aggregate"]  # hook itself does not alter the timing extractor


def test_slot_order_treatment_cannot_be_enabled_during_resume(tmp_path, monkeypatch):
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    frozen = adapter.grid.file_ref(out / "freeze.json")
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--output", str(out), "--resume", "--slot-order-contract", str(tmp_path / "unread-contract.json")])
    assert adapter.grid.file_ref(out / "freeze.json") == frozen and not calls["run"]


def local_slot_order_contract():
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    path = base / "optimization_loop/round_003/slot_order_source_contract.json"
    if not path.exists():
        pytest.skip("local recorded slot-order source contract unavailable")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    return path, adapter.selected_rows(selection)


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_slot_order_source_contract_is_rederived_and_runtime_bound_without_enabling_recurrent():
    path, rows = local_slot_order_contract()
    proof = adapter.verified_slot_order_contract(path, rows, adapter.ROOT)
    assert proof["contract"]["phase_candidate_order"] == "stable_admission"
    assert proof["contract"]["preserves_engine_start_definition"] is True
    assert proof["native_latency_used"] is False
    assert proof["recurrent_treatment_enabled_by_this_validation"] is False
    assert proof["qualification_required_after_static_bindings"] is True
    assert "llama-server-impl.dll" in proof["native_runtime_sha256"]
    assert proof["source_chain_contract_ref"] in proof["evidence_refs"]


@pytest.mark.parametrize("mutation,reason", [
    ("source", "frozen reference changed"),
    ("canonical", "differs from re-derived"),
    ("chain", "differs from re-derived"),
    ("missing_chain", "chain binding is required"),
    ("digest", "content SHA256 mismatch"),
    ("runtime", "differs from selected native artifact"),
])
@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_slot_order_source_contract_rejects_tampered_identity_or_rule(tmp_path, mutation, reason):
    path, rows = local_slot_order_contract()
    contract, _ = adapter.grid.read_document(path)
    if mutation == "source":
        contract["source_sha256"][next(iter(contract["source_sha256"]))] = "0" * 64
    elif mutation == "canonical":
        contract["prompt_fill_rule"] = "rotate_between_prompts"
    elif mutation == "chain":
        contract["source_chain_binding"]["server_sha256"] = "0" * 64
    elif mutation == "missing_chain":
        contract["source_chain_binding"] = None
    elif mutation == "digest":
        contract["content_sha256"] = "0" * 64
    elif mutation == "runtime":
        rows = copy.deepcopy(rows[:1])
        next(ref for ref in rows[0]["native_runtime_refs"] if Path(ref["path"]).name.lower() == "llama-server-impl.dll")["sha256"] = "0" * 64
    changed = tmp_path / "changed-slot-contract.json"
    document(changed, contract)
    with pytest.raises(ValueError, match=reason):
        adapter.verified_slot_order_contract(changed, rows, adapter.ROOT)


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_slot_order_freeze_binds_source_evidence_and_fails_if_runtime_hook_is_missing(tmp_path, monkeypatch):
    from heterollm_sim import llama_scenario
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    source_path, rows = local_slot_order_contract()
    proof = adapter.verified_slot_order_contract(source_path, rows, adapter.ROOT)
    monkeypatch.setattr(adapter, "verified_slot_order_contract", lambda *args, **kwargs: proof)
    frozen = adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path,
        slot_order_contract_path=source_path)
    assert frozen["slot_order"] == proof
    assert frozen["recurrent_batching"] is None
    assert frozen["cells"][0]["static_inputs"]["slot_order_contract"] == proof["contract"]
    assert frozen["cells"][0]["static_inputs"]["recurrent_batching_contract"] is None
    assert not calls["run"]
    monkeypatch.undo()
    monkeypatch.setattr(llama_scenario, "apply_llama_runtime_config", lambda scenario, config: scenario)
    with pytest.raises(ValueError, match="requires the opt-in runtime adapter implementation"):
        adapter.verified_slot_order_contract(source_path, rows, adapter.ROOT)


@pytest.mark.parametrize("variation,reason", [
    ("fresh", None), ("staggered", "dynamic_or_staggered_arrivals_unproven"),
    ("reused", "nonfresh_or_reused_slots_unproven"),
    ("dynamic", "arrival_stream_unproven"),
])
@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_slot_order_final_static_replan_retains_core_qualification(variation, reason):
    from dataclasses import replace
    from heterollm_sim.llama_scenario import apply_llama_runtime_config
    from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
    from tests.test_llama_slot_order import cohort
    path, _ = local_slot_order_contract()
    contract, _ = adapter.grid.read_document(path)
    case = cohort()
    if variation == "staggered":
        requests = (case.workload.requests[0], replace(case.workload.requests[1], arrival_ns=1.0))
        case = replace(case, workload=replace(case.workload, requests=requests))
    elif variation == "reused":
        case = replace(case, workload=replace(case.workload, metadata={"slot_reuse": True}))
    elif variation == "dynamic":
        case = replace(case, workload=replace(case.workload, arrival_rate_rps=1.0))
    case = apply_llama_runtime_config(case,
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2))
    refreshed, evidence = adapter.replan_final_static_scenario(case, slot_order_contract=contract)
    qualification = refreshed.workload.metadata["llama_cpp_slot_order"]
    assert evidence["normal_validation_passed"] is True
    assert qualification["qualified"] is (reason is None)
    assert qualification["applied"] is (reason is None)
    assert qualification["preserves_engine_start_definition"] is True
    assert refreshed.workload.requests == case.workload.requests
    if reason:
        assert reason in qualification["reasons"]
        assert refreshed.workload.scheduler.phase_candidate_order == case.workload.scheduler.phase_candidate_order
    else:
        assert refreshed.workload.scheduler.phase_candidate_order == "stable_admission"


def dispatch_result(*, total=1, with_summary=True):
    metadata = {"unrelated_" + str(index): index for index in range(40)}
    if with_summary:
        metadata["cpu_iq_panel_reuse"] = {
            "schema": "heterollm.cpu-iq-panel-reuse-coverage/v1", "cpu_gemm_tasks": 6,
            "audited_tasks": 6, "applied_tasks": 2, "conditional_tasks": 2,
            "native_dispatch_proven_tasks": 0, "uncovered_tasks": 4,
            "uncovered_reason_counts": {"F32_hidden_storage_unproven": 4}}
        metadata["host_gemm_offload"] = {
            "schema": "test-host-offload-ledger", "audited_tasks": 4, "applied_tasks": 3,
            "reason_counts": {"physical_m_below_threshold": 1}}
    batch = SimpleNamespace(kind="prefill", cohort_id="c0", request_ids=(), items=(),
        token_count=64, start_ns=0, end_ns=10, metadata={},
        cost=SimpleNamespace(metadata=metadata, duration_ns=10))
    return SimpleNamespace(serving=SimpleNamespace(batches=(batch,),
        scheduler_metrics=SimpleNamespace(total_batches=total)), retention_policy="aggregate")


def test_named_dispatch_ledgers_survive_general_metadata_bounds():
    result = dispatch_result()
    scheduled = adapter.batch_schedule(result, [])
    row = scheduled["batches"][0]
    assert "cpu_iq_panel_reuse" not in row["cost_metadata"]
    assert row["cpu_iq_panel_reuse"]["applied_tasks"] == 2
    assert row["cpu_iq_panel_reuse"]["uncovered_reason_counts"] == {"F32_hidden_storage_unproven": 4}
    assert row["host_gemm_offload"]["applied_tasks"] == 3
    summary = adapter.retained_dispatch_summary(result)
    assert summary["history_complete"] is True
    assert summary["cpu_iq_panel_reuse"]["applied_tasks"] == 2
    assert summary["cpu_iq_panel_reuse"]["conditional_tasks"] == 2
    assert summary["cpu_iq_panel_reuse"]["native_dispatch_proven_tasks"] == 0
    assert summary["host_gemm_offload"]["applied_tasks"] == 3


def test_dispatch_ledger_absence_or_partial_history_is_not_reported_as_zero():
    old = adapter.retained_dispatch_summary(dispatch_result(with_summary=False))
    assert old["cpu_iq_panel_reuse"]["applied_tasks"] is None
    assert old["host_gemm_offload"]["observed_applied_tasks"] is None
    partial = adapter.retained_dispatch_summary(dispatch_result(total=2))
    assert partial["history_complete"] is False
    assert partial["cpu_iq_panel_reuse"]["applied_tasks"] is None
    assert partial["cpu_iq_panel_reuse"]["observed_applied_tasks"] == 2
    assert partial["host_gemm_offload"]["applied_tasks"] is None
    assert partial["host_gemm_offload"]["observed_applied_tasks"] == 3


def test_dispatch_qualification_retains_native_unknown_and_actual_storage():
    case = Scenario(workload=Metadata(metadata={"llama_cpp_f32_hidden_storage": True,
        "llama_cpp_cuda_op_offload": {"status": "enabled", "reason": "source_qualified_physical_m_dispatch", "minimum_m": 32},
        "llama_cpp_slot_order": {"qualified": False, "applied": False, "reasons": ["arrival_stream_unproven"]}}))
    inputs = {"cpu_iq_panel_reuse": {"enabled": True, "no_iq_panel_environment_state": "unknown",
        "assume_default_unset": True, "native_dispatch_proven": False,
        "evaluation_scope": "conditional_default_unset_ablation"}}
    facts = adapter.dispatch_qualification(case, inputs)
    assert facts["cpu_iq_panel_reuse"]["native_dispatch_proven"] is False
    assert facts["cpu_iq_panel_reuse"]["conditional"] is True
    assert facts["storage"]["f32_hidden_storage"] is True
    assert facts["host_gemm_offload"]["qualified"] is True
    assert facts["slot_order"]["applied"] is False
    assert facts["slot_order"]["reasons"] == ["arrival_stream_unproven"]
    assert adapter.dispatch_qualification(None)["storage"]["f32_hidden_storage"] is None


def test_host_offload_default_off_and_verified_apply_precedes_replan_without_forcing_storage(tmp_path, monkeypatch):
    from dataclasses import replace
    from heterollm_sim import runtime_adapters
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    baseline_inputs = adapter.static_inputs(row, selection, tmp_path)
    baseline = adapter.predict_cell(baseline_inputs)
    assert baseline_inputs["host_offload_source_contract"] is None
    assert baseline["dispatch_qualification"]["host_gemm_offload"]["requested"] is False
    contract = {"schema": "test-forwarded-source", "minimum_m": 32}
    proof = {"requested": True, "status": "verified", "source_contract": contract,
        "op_offload_enabled": True, "cuda_backend_available": True, "uncovered_reasons": []}
    inputs = {**baseline_inputs, "host_offload_source_contract": contract, "host_offload_evidence": proof}
    monkeypatch.setattr(adapter.grid, "build_matching_scenario", lambda *args, **kwargs: Scenario(
        llama_cpp_config=SimpleNamespace(op_offload=True)))
    def apply(case, config, *, source_contract, cuda_backend_available):
        assert source_contract is contract and cuda_backend_available is True
        return replace(case, workload=Metadata(metadata={"llama_cpp_cuda_op_offload": {
            "status": "enabled", "reason": "source_qualified_physical_m_dispatch", "minimum_m": 32}}))
    monkeypatch.setattr(runtime_adapters, "apply_llama_cuda_op_offload", apply)
    predicted = adapter.predict_cell(inputs)
    assert calls["replan"][-1].workload.metadata["llama_cpp_cuda_op_offload"]["status"] == "enabled"
    assert predicted["dispatch_qualification"]["host_gemm_offload"]["qualified"] is True
    assert predicted["dispatch_qualification"]["storage"]["f32_hidden_storage"] is None
    assert predicted["dispatch_summary"]["host_gemm_offload"]["applied_tasks"] is None
    assert predicted["aggregate"] == baseline["aggregate"]


def test_host_offload_uncovered_does_not_install_capability_and_mismatched_cli_fails(tmp_path, monkeypatch):
    from heterollm_sim import runtime_adapters
    case = Scenario(llama_cpp_config=SimpleNamespace(op_offload=False))
    monkeypatch.setattr(runtime_adapters, "apply_llama_cuda_op_offload", lambda *args, **kwargs: pytest.fail("must not apply"))
    proof = {"requested": True, "status": "uncovered", "source_contract": None,
        "uncovered_reasons": ["historical_GGML_OP_OFFLOAD_MIN_BATCH_not_captured"]}
    assert adapter.apply_host_offload_static_contract(case, {"host_offload_evidence": proof}) is case
    qualification = adapter.dispatch_qualification(case, {"host_offload_evidence": proof})
    assert qualification["host_gemm_offload"]["requested"] is True
    assert qualification["host_gemm_offload"]["binding_status"] == "uncovered"
    verified = {"status": "verified", "source_contract": {"test": True},
        "op_offload_enabled": True, "cuda_backend_available": True}
    with pytest.raises(ValueError, match="differs from captured CLI"):
        adapter.apply_host_offload_static_contract(case, {"host_offload_evidence": verified,
            "host_offload_source_contract": verified["source_contract"]})


@pytest.mark.parametrize("enabled,available", [(True, True), (False, True), (True, False)])
def test_host_offload_uses_existing_adapter_storage_and_capability_semantics(enabled, available):
    from tests.test_runtime_op_offload import host_case, contract
    case = host_case(enabled=enabled, available=False)
    source = contract()
    proof = {"status": "verified", "source_contract": source,
        "op_offload_enabled": enabled, "cuda_backend_available": available}
    result = adapter.apply_host_offload_static_contract(case, {
        "host_offload_source_contract": source, "host_offload_evidence": proof})
    assert result.placement == case.placement
    assert (result.gpu_profile.host_gemm_offload is not None) is (enabled and available)
    assert bool(result.workload.metadata.get("llama_cpp_f32_hidden_storage")) is (enabled and available)


def test_host_offload_cannot_be_added_during_resume(tmp_path, monkeypatch):
    path, _, _, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    adapter.freeze_selection(path, output, data_root=tmp_path)
    before = adapter.grid.file_ref(output / "freeze.json")
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--output", str(output), "--resume", "--host-offload-source-contract", "unread.json"])
    assert adapter.grid.file_ref(output / "freeze.json") == before


def fake_host_build_binding(tmp_path, monkeypatch, *, status="verified"):
    from tools import llama_runtime_source_binding as source_api
    audit = document(tmp_path / "runtime_build_audit.json", {"schema": "stable-native-runtime-build-posthoc-audit/v1"})
    receipt = document(tmp_path / "base_build_receipt.json", {"source_sha256_before": {},
        "source_sha256_after": {}, "source_unchanged": True, "returncode": 0})
    source = {"schema": "test-physical-offload", "minimum_m": 32,
        "environment": {"name": "GGML_OP_OFFLOAD_MIN_BATCH", "value": None, "status": "captured_absent"}}
    binding = {"schema": source_api.SCHEMA, "status": "verified_build_chain", "evidence_refs": [audit, receipt]}
    path = tmp_path / "binding.json"
    document(path, binding)
    monkeypatch.setattr(source_api, "verify_llama_runtime_source_binding", lambda *args, **kwargs: copy.deepcopy(binding))
    def bind(*args, **kwargs):
        return {"status": status, "source_contract": source if status == "verified" else None,
            "cuda_backend_available": True, "op_offload_enabled": True if status == "verified" else None,
            "op_offload_cli_basis": "test-only-captured", "uncovered_reasons": [] if status == "verified" else ["CLI_unbound"],
            "evidence_refs": [kwargs["native_record_ref"]], "native_dispatch_proven": False,
            "native_timing_must_not_leak": 123456, "required_workload_metadata": {"llama_cpp_f32_hidden_storage": True}}
    monkeypatch.setattr(source_api, "bind_llama_cuda_op_offload_contract", bind)
    return path, binding, source


def test_host_offload_freeze_rederives_binding_and_preserves_independent_treatments(tmp_path, monkeypatch):
    selection_path, _, row, calls = fixture(tmp_path, monkeypatch)
    source_path, binding, source = fake_host_build_binding(tmp_path, monkeypatch)
    proof = adapter.verified_host_offload_source_contract(source_path, [row], tmp_path)
    assert proof["verified_cell_count"] == 1 and proof["uncovered_cell_count"] == 0
    cell = proof["cells"][row["cell_id"]]
    assert "native_timing_must_not_leak" not in cell and "required_workload_metadata" not in cell
    frozen = adapter.freeze_selection(selection_path, tmp_path / "out", data_root=tmp_path,
        host_offload_source_contract_path=source_path)
    inputs = frozen["cells"][0]["static_inputs"]
    assert inputs["host_offload_source_contract"] == source
    assert inputs["config"]["op_offload"] is True
    assert inputs["cpu_iq_panel_reuse"] is None and inputs["slot_order_contract"] is None
    assert inputs["recurrent_batching_contract"] is None and not calls["run"]
    assert "native_timing_must_not_leak" not in json.dumps(inputs)
    adapter.verify_freeze_references(frozen)


def test_host_offload_binding_rejects_source_mutation_and_selected_environment_mismatch(tmp_path, monkeypatch):
    _, _, row, _ = fixture(tmp_path, monkeypatch)
    path, binding, _ = fake_host_build_binding(tmp_path, monkeypatch)
    tampered = copy.deepcopy(binding)
    tampered["fabricated"] = True
    document(path, tampered)
    with pytest.raises(ValueError, match="differs from re-derived"):
        adapter.verified_host_offload_source_contract(path, [row], tmp_path)
    document(path, binding)
    row["config"]["environment"]["GGML_OP_OFFLOAD_MIN_BATCH"] = "99"
    with pytest.raises(ValueError, match="captured environment differs"):
        adapter.verified_host_offload_source_contract(path, [row], tmp_path)


def test_uncovered_host_offload_binding_is_retained_but_never_enabled(tmp_path, monkeypatch):
    _, selection, row, _ = fixture(tmp_path, monkeypatch)
    path, _, _ = fake_host_build_binding(tmp_path, monkeypatch, status="uncovered")
    proof = adapter.verified_host_offload_source_contract(path, [row], tmp_path)
    assert proof["verified_cell_count"] == 0 and proof["uncovered_cell_count"] == 1
    inputs = adapter.static_inputs(row, selection, tmp_path, host_offload=proof)
    assert inputs["host_offload_source_contract"] is None
    assert inputs["host_offload_evidence"]["uncovered_reasons"] == ["CLI_unbound"]
    assert "op_offload" not in inputs["config"]


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_recorded_host_offload_binding_uses_current_runtime_and_captured_threshold_without_native_execution():
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    audit = base / "optimization_loop/round_004/runtime_source_binding_structural_audit.json"
    if not audit.is_file():
        pytest.skip("recorded host-offload build audit unavailable")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    rows = adapter.selected_rows(selection)
    selected = [next(row for row in rows if row["cell_id"] == "qwen38_p1536_o128_c1__fixed_runtime"),
        next(row for row in rows if row["model_key"] == "qwen38_gpu")]
    proof = adapter.verified_host_offload_source_contract(audit, selected, adapter.ROOT)
    assert proof["verified_cell_count"] == 2
    assert proof["uncovered_cell_count"] == 0
    for cell in proof["cells"].values():
        source = cell["source_contract"]
        assert source["minimum_m"] == 32
        assert source["environment"]["status"] == "captured_absent"
        assert source["native_dispatch_proven"] is False
        assert source["source_runtime_binding"] == "verified_annotation_cuda_and_original_base_inherited_by_selected_native_runtime"
        assert cell["cuda_backend_available"] is True
        assert cell["op_offload_enabled"] is True
        assert "llama_cpp_f32_hidden_storage" not in cell
        assert "execution_environment" not in cell
    assert proof["native_latency_used"] is False


def test_actual_host_dispatch_requires_observation_and_absence_is_unknown():
    observed = adapter.retained_dispatch_summary(dispatch_result())
    assert observed["host_gemm_offload"]["actual_cpu_to_gpu_offload"] is True
    unknown = adapter.retained_dispatch_summary(dispatch_result(with_summary=False))
    assert unknown["host_gemm_offload"]["actual_cpu_to_gpu_offload"] is None
    empty = dispatch_result()
    empty.serving.batches[0].cost.metadata["host_gemm_offload"]["applied_tasks"] = 0
    assert adapter.retained_dispatch_summary(empty)["host_gemm_offload"]["actual_cpu_to_gpu_offload"] is False
    empty.serving.scheduler_metrics.total_batches = 3
    assert adapter.retained_dispatch_summary(empty)["host_gemm_offload"]["actual_cpu_to_gpu_offload"] is None


def fake_tensor_storage_proof(tmp_path, *, f32=False):
    contract = {"schema": "test-tensor-storage-source", "get_rows_access": "selected_packed_rows_per_index"}
    ref = document(tmp_path / "tensor_storage_test_contract.json", contract)
    return {"contract": contract, "contract_ref": ref, "evidence_refs": [ref],
        "f32_hidden_storage_requested": f32, "native_latency_used": False,
        "limits": ["row conversion unpriced", "cache lines and write allocation unpriced"]}


@pytest.mark.parametrize("f32", [False, True])
def test_tensor_storage_freeze_requires_explicit_separate_hidden_flag(tmp_path, monkeypatch, f32):
    path, _, _, calls = fixture(tmp_path, monkeypatch)
    proof = fake_tensor_storage_proof(tmp_path, f32=f32)
    observed = []
    def verify(*args, **kwargs):
        observed.append(kwargs)
        return proof
    monkeypatch.setattr(adapter, "verified_tensor_storage_contract", verify)
    frozen = adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path,
        tensor_storage_contract_path=Path(proof["contract_ref"]["path"]), tensor_storage_f32_hidden=f32)
    inputs = frozen["cells"][0]["static_inputs"]
    assert inputs["tensor_storage_contract"] == proof["contract"]
    assert inputs["tensor_storage_f32_hidden"] is f32
    assert observed[0]["f32_hidden_storage"] is f32
    assert frozen["host_offload_source"] is None
    assert inputs["host_offload_evidence"] is None
    assert inputs["cpu_iq_panel_reuse"] is None and inputs["slot_order_contract"] is None
    assert not calls["run"]
    adapter.verify_freeze_references(frozen)


def test_tensor_storage_default_off_and_f32_alone_is_rejected_before_freeze(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    assert inputs["tensor_storage_contract"] is None and inputs["tensor_storage_f32_hidden"] is False
    assert adapter.apply_tensor_storage_static_contract(Scenario(), inputs) == Scenario()
    with pytest.raises(ValueError, match="requires its explicit source contract"):
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path, tensor_storage_f32_hidden=True)
    assert not (tmp_path / "out").exists() and not calls["run"]


@pytest.mark.parametrize("args", [["--tensor-storage-contract", "unread.json"],
    ["--tensor-storage-f32-hidden"], ["--no-tensor-storage-f32-hidden"]])
def test_tensor_storage_treatments_cannot_change_during_resume(tmp_path, monkeypatch, args):
    path, _, _, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    adapter.freeze_selection(path, output, data_root=tmp_path)
    before = adapter.grid.file_ref(output / "freeze.json")
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--output", str(output), "--resume", *args])
    assert adapter.grid.file_ref(output / "freeze.json") == before


@pytest.mark.parametrize("flag", ["--tensor-storage-f32-hidden", "--no-tensor-storage-f32-hidden"])
def test_explicit_hidden_cli_option_requires_tensor_storage_contract(tmp_path, monkeypatch, flag):
    path, _, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--selection", str(path), "--output", str(tmp_path / "out"), flag])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("f32", [False, True])
def test_tensor_storage_uses_core_apply_before_final_replanning(tmp_path, monkeypatch, f32):
    from dataclasses import replace
    from heterollm_sim import llama_tensor_storage
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    proof = fake_tensor_storage_proof(tmp_path, f32=f32)
    inputs = adapter.static_inputs(row, selection, tmp_path, tensor_storage=proof)
    def apply(case, contract, *, f32_hidden_storage):
        assert contract is proof["contract"] and f32_hidden_storage is f32
        return replace(case, workload=Metadata(metadata={"llama_cpp_tensor_storage": {
            "qualified": True, "status": "enabled", "reasons": [],
            "f32_hidden_storage_requested": f32, "timing_completeness": "row conversion unpriced"}}))
    monkeypatch.setattr(llama_tensor_storage, "apply_llama_tensor_storage_contract", apply)
    prediction = adapter.predict_cell(inputs)
    assert calls["replan"][-1].workload.metadata["llama_cpp_tensor_storage"]["qualified"] is True
    assert prediction["dispatch_qualification"]["tensor_storage"]["f32_hidden_storage_requested"] is f32
    assert prediction["dispatch_qualification"]["tensor_storage"]["qualified"] is True
    assert prediction["dispatch_qualification"]["storage"]["f32_hidden_storage"] is None
    assert prediction["dispatch_summary"]["tensor_storage"]["applied_tasks"] is None
    assert prediction["input_identity"]["tensor_storage_binding"] == proof


@pytest.mark.parametrize("f32", [False, True])
def test_tensor_storage_actual_row_bytes_and_capacity_remain_distinct(f32):
    from tests import test_llama_tensor_storage as storage_tests
    contract = storage_tests.contract.__wrapped__()
    case = storage_tests.scenario()
    proof = {"contract": contract, "f32_hidden_storage_requested": f32}
    result = adapter.apply_tensor_storage_static_contract(case, {"tensor_storage_contract": contract,
        "tensor_storage_evidence": proof, "tensor_storage_f32_hidden": f32})
    task = storage_tests.compute_task(storage_tests.embedding_tasks(result))
    assert task.metadata["rank_weight_capacity_bytes"] == 340000
    assert task.metadata["weight_read_bytes"] == 2176
    assert task.metadata["lookup_index_read_bytes"] == 256
    assert task.metadata["lookup_write_bytes"] == 8192
    assert task.metadata["embedding_dequant_compute"] == "unmodeled_selected_rows_only"
    assert bool(result.workload.metadata.get("llama_cpp_f32_hidden_storage")) is f32
    assert result.placement == case.placement


def test_tensor_storage_flag_mismatch_cannot_bypass_frozen_treatment(tmp_path):
    proof = fake_tensor_storage_proof(tmp_path)
    with pytest.raises(ValueError, match="flag differs from verified evidence"):
        adapter.apply_tensor_storage_static_contract(Scenario(), {"tensor_storage_evidence": proof,
            "tensor_storage_contract": proof["contract"], "tensor_storage_f32_hidden": True})


def test_tensor_storage_summary_preserves_explicit_bytes_and_does_not_sum_table_capacity():
    result = dispatch_result(total=2)
    ledger = {"embedding_tasks": 1, "audited_tasks": 1, "applied_tasks": 1,
        "selected_weight_read_bytes": 2176, "index_read_bytes": 256, "output_write_bytes": 8192,
        "rank_weight_capacity_bytes": 340000, "selected_rows": 64, "selected_elements": 2048,
        "full_table_staging_bytes": 0, "uncovered_reason_counts": {"dequantization_compute_unpriced": 1}}
    result.serving.batches[0].cost.metadata["tensor_storage"] = ledger
    result.serving.batches = result.serving.batches * 2
    batch = adapter.batch_schedule(result, [])["batches"][0]
    assert batch["tensor_storage"] == ledger
    assert "tensor_storage" not in batch["cost_metadata"]
    summary = adapter.retained_dispatch_summary(result)["tensor_storage"]
    assert summary["applied_tasks"] == 2
    assert summary["selected_weight_read_bytes"] == 4352
    assert summary["index_read_bytes"] == 512
    assert summary["output_write_bytes"] == 16384
    assert summary["maximum_observed_rank_weight_capacity_bytes"] == 340000
    assert "rank_weight_capacity_bytes" not in summary
    assert summary["uncovered_reason_counts"] == {"dequantization_compute_unpriced": 2}


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_recorded_tensor_storage_contract_rederives_all_four_current_module_paths():
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    source_path = base / "optimization_loop/round_005/tensor_storage_source_contract.json"
    if not source_path.exists():
        pytest.skip("recorded tensor storage contract unavailable")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    rows = adapter.selected_rows(selection)
    selected = [next(row for row in rows if row["cell_id"] == "qwen38_p1536_o128_c1__fixed_runtime"),
        next(row for row in rows if row["model_key"] == "qwen38_gpu")]
    proof = adapter.verified_tensor_storage_contract(source_path, selected, adapter.ROOT)
    roles = proof["source_compilation"]
    assert set(roles) == {"operators", "cpu_get_rows", "cuda_get_rows", "input_graph"}
    assert "recompiled" in roles["cpu_get_rows"]["binding"]
    assert "native-thread-control" in roles["cpu_get_rows"]["compiled_source"]
    assert "explicitly_reused" in roles["input_graph"]["binding"]
    assert "explicitly_reused" in roles["cuda_get_rows"]["binding"]
    assert proof["f32_hidden_storage_requested"] is False
    assert proof["host_offload_treatment_enabled_by_this_validation"] is False
    assert proof["native_dispatch_proven"] is False and proof["native_latency_used"] is False


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_recorded_tensor_storage_rejects_forged_rule_even_if_document_is_resealed(tmp_path):
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    source = base / "optimization_loop/round_005/tensor_storage_source_contract.json"
    if not source.exists():
        pytest.skip("recorded tensor storage contract unavailable")
    contract, _ = adapter.grid.read_document(source)
    contract["get_rows_access"] = "whole_table_always"
    contract["content_sha256"] = adapter.grid.stable_hash(contract)
    changed = tmp_path / "changed-storage-contract.json"
    document(changed, contract)
    with pytest.raises(ValueError, match="differs from re-derived source rules"):
        adapter.verified_tensor_storage_contract(changed, [], adapter.ROOT)


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
def test_recorded_tensor_storage_rejects_selected_cpu_module_mismatch():
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    source = base / "optimization_loop/round_005/tensor_storage_source_contract.json"
    if not source.exists():
        pytest.skip("recorded tensor storage contract unavailable")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    row = copy.deepcopy(next(r for r in adapter.selected_rows(selection) if r["cell_id"] == "qwen38_p1536_o128_c1__fixed_runtime"))
    cpu = next(ref for ref in row["native_runtime_refs"] if Path(ref["path"]).name.lower() == "ggml-cpu.dll")
    cpu["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity chain disagrees"):
        adapter.verified_tensor_storage_contract(source, [row], adapter.ROOT)


def fake_gpu_invocation_contract_api(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    api = ModuleType("heterollm_sim.llama_gpu_invocations")
    api.SCHEMA = "llama.cpp.gpu-native-invocations/v1"
    source_ref = document(tmp_path / "gpu-rule.json", {"source": "test-only"})
    def derive(binding, *, captured_kernel_environment, cuda_compute_capability, mmq_device_evidence=None):
        captured = "GGML_CUDA_DISABLE_FUSION" in captured_kernel_environment
        value = captured_kernel_environment.get("GGML_CUDA_DISABLE_FUSION")
        return {"schema": api.SCHEMA, "status": "conditional" if captured else "uncovered",
            "cuda_compute_capability": cuda_compute_capability,
            "runtime_binding_sha256": binding.get("content_sha256"),
            "kernel_environment": {"GGML_CUDA_DISABLE_FUSION": {"captured": captured, "value": value}},
            "fusion_enabled": (value is None or value == "0") if captured else None,
            "mmq_device_evidence": dict(mmq_device_evidence) if mmq_device_evidence else {"available": False, "reason": "missing_device_evidence"},
            "source_refs": [source_ref], "native_dispatch_proven": False,
            "uncovered_reasons": [] if captured else ["historical_cuda_fusion_environment_unknown"],
            "conditional_reasons": ["historical_model_body_unbound"], "unpriced_terms": ["vector_main_unpriced"]}
    api.derive_llama_gpu_invocation_contract = derive
    api.apply_llama_gpu_invocation_contract = lambda scenario, contract, *, enabled=False, enable_mmq_source_costs=False: scenario
    monkeypatch.setitem(sys.modules, api.__name__, api)
    binding_ref = document(tmp_path / "gpu-build-binding.json", {"content_sha256": "test-binding"})
    template = derive({"content_sha256": "test-binding"}, captured_kernel_environment={"GGML_CUDA_DISABLE_FUSION": None}, cuda_compute_capability=1200)
    path = tmp_path / "gpu-invocation-contract.json"
    document(path, template)
    monkeypatch.setattr(adapter, "verify_gpu_invocation_source_links", lambda *args, **kwargs: {"source_compilation": {}, "evidence_refs": [source_ref]})
    return api, path, template, binding_ref


def fake_gpu_runtime_binding(tmp_path, row, binding_ref, *, fusion_recorded=True, value=None):
    raw_ref = document(tmp_path / "gpu-record.json", {"execution_environment": {
        "GGML_CUDA_DISABLE_FUSION": {"is_set": value is not None, "value": value}} if fusion_recorded else {},
        "native_timing_must_not_propagate": 987654})
    return {"contract": {"content_sha256": "test-binding"}, "contract_ref": binding_ref,
        "evidence_refs": [binding_ref, raw_ref], "cells": {row["cell_id"]: {
            "cuda_backend_available": True, "device_evidence": {"compute_capability": "12.0", "gpu_uuid": "GPU-test"},
            "native_record_refs": [raw_ref]}}}


def test_kernel_environment_missing_stays_unknown_despite_today_or_selected_default(monkeypatch):
    monkeypatch.setenv("GGML_CUDA_DISABLE_FUSION", "1")
    values, facts = adapter.captured_kernel_environment({}, {"GGML_CUDA_DISABLE_FUSION": None})
    assert values == {}
    assert facts["GGML_CUDA_DISABLE_FUSION"] == {"captured": False, "state": "unknown", "value": None}
    values, facts = adapter.captured_kernel_environment({"execution_environment": {
        "GGML_CUDA_DISABLE_FUSION": {"is_set": False, "value": None}}})
    assert values == {"GGML_CUDA_DISABLE_FUSION": None}
    assert facts["GGML_CUDA_DISABLE_FUSION"]["state"] == "captured_absent"
    with pytest.raises(ValueError, match="presence/value disagree"):
        adapter.captured_kernel_environment({"execution_environment": {
            "GGML_CUDA_DISABLE_FUSION": {"is_set": False, "value": "0"}}})
    with pytest.raises(ValueError, match="differs from selected"):
        adapter.captured_kernel_environment({"execution_environment": {"GGML_CUDA_DISABLE_FUSION": "1"}}, {"GGML_CUDA_DISABLE_FUSION": None})


@pytest.mark.parametrize("captured", [False, True])
def test_gpu_invocation_each_cell_rederives_environment_instead_of_borrowing_template(tmp_path, monkeypatch, captured):
    _, _, row, _ = fixture(tmp_path, monkeypatch)
    _, source_path, template, binding_ref = fake_gpu_invocation_contract_api(tmp_path, monkeypatch)
    runtime = fake_gpu_runtime_binding(tmp_path, row, binding_ref, fusion_recorded=captured)
    monkeypatch.setenv("GGML_CUDA_DISABLE_FUSION", "0")
    proof = adapter.verified_gpu_invocation_contract(source_path, [row], tmp_path, runtime_binding=runtime)
    cell = proof["cells"][row["cell_id"]]
    assert proof["contract"] == template
    assert proof["template_environment_is_runtime_evidence"] is False
    assert cell["contract"]["status"] == ("conditional" if captured else "uncovered")
    assert cell["kernel_environment"]["GGML_CUDA_DISABLE_FUSION"]["captured"] is captured
    assert cell["native_dispatch_proven"] is False
    assert "native_timing_must_not_propagate" not in json.dumps(proof)
    assert proof["today_environment_read"] is False


def test_gpu_invocation_source_and_device_mismatch_fail_closed(tmp_path, monkeypatch):
    _, _, row, _ = fixture(tmp_path, monkeypatch)
    _, path, template, binding_ref = fake_gpu_invocation_contract_api(tmp_path, monkeypatch)
    runtime = fake_gpu_runtime_binding(tmp_path, row, binding_ref)
    document(path, {**template, "fusion_enabled": False})
    with pytest.raises(ValueError, match="differs from re-derived"):
        adapter.verified_gpu_invocation_contract(path, [row], tmp_path, runtime_binding=runtime)
    document(path, template)
    runtime["cells"][row["cell_id"]]["device_evidence"]["compute_capability"] = "8.0"
    with pytest.raises(ValueError, match="differs from captured native device"):
        adapter.verified_gpu_invocation_contract(path, [row], tmp_path, runtime_binding=runtime)


@pytest.mark.parametrize("costs", [False, True])
def test_gpu_invocation_freeze_and_apply_keep_source_cost_treatment_separate(tmp_path, monkeypatch, costs):
    from dataclasses import replace
    selection_path, selection, row, calls = fixture(tmp_path, monkeypatch)
    api, source_path, _, binding_ref = fake_gpu_invocation_contract_api(tmp_path, monkeypatch)
    runtime = fake_gpu_runtime_binding(tmp_path, row, binding_ref)
    monkeypatch.setattr(adapter, "verified_host_offload_source_contract", lambda *args, **kwargs: runtime)
    def apply(case, contract, *, enabled=False, enable_mmq_source_costs=False, enable_conversion_cta_costs=False):
        assert enabled is True and enable_mmq_source_costs is costs
        assert enable_conversion_cta_costs is False
        return replace(case, workload=Metadata(metadata={"llama_cpp_gpu_native_invocations": {
            "applied": True, "conditional": True, "status": "conditional", "native_dispatch_proven": False,
            "qualified_projection_groups": 4, "uncovered_group_reason_counts": {"layout_unbound": 1},
            "mmq_source_costs": {"requested": costs, "applied": False, "reason": "device_evidence_missing"}}}))
    api.apply_llama_gpu_invocation_contract = apply
    frozen = adapter.freeze_selection(selection_path, tmp_path / "out", data_root=tmp_path,
        gpu_invocation_contract_path=source_path, gpu_mmq_source_costs=costs)
    inputs = frozen["cells"][0]["static_inputs"]
    assert inputs["gpu_mmq_source_costs"] is costs
    assert inputs["host_offload_evidence"] is None and frozen["host_offload_source"] is None
    assert inputs["tensor_storage_contract"] is None and inputs["slot_order_contract"] is None
    prediction = adapter.predict_cell(inputs)
    audit = prediction["dispatch_qualification"]["gpu_invocations"]
    assert audit["applied"] is True and audit["mmq_source_costs_requested"] is costs
    assert audit["mmq_source_costs"]["applied"] is False
    assert audit["native_dispatch_proven"] is False and audit["conditional"] is True
    assert calls["replan"][-1].workload.metadata["llama_cpp_gpu_native_invocations"]["applied"] is True
    assert "llama_cpp_mmq_source_work" not in calls["replan"][-1].workload.metadata
    adapter.verify_freeze_references(frozen)


def test_gpu_invocation_default_off_and_costs_alone_are_rejected(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    assert inputs["gpu_invocation_contract"] is None and inputs["gpu_mmq_source_costs"] is False
    assert adapter.apply_gpu_invocation_static_contract(Scenario(), inputs) == Scenario()
    with pytest.raises(ValueError, match="require a GPU invocation source contract"):
        adapter.freeze_selection(path, tmp_path / "out", data_root=tmp_path, gpu_mmq_source_costs=True)
    assert not (tmp_path / "out").exists() and not calls["run"]


@pytest.mark.parametrize("switches", [["--gpu-invocation-contract", "unread.json"], ["--gpu-mmq-source-costs"], ["--no-gpu-mmq-source-costs"]])
def test_gpu_invocation_switches_cannot_be_added_during_resume(tmp_path, monkeypatch, switches):
    path, _, _, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    before = adapter.grid.file_ref(out / "freeze.json")
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--output", str(out), "--resume", *switches])
    assert adapter.grid.file_ref(out / "freeze.json") == before


@pytest.mark.parametrize("flag", ["--gpu-mmq-source-costs", "--no-gpu-mmq-source-costs"])
def test_gpu_cost_cli_option_requires_invocation_contract(tmp_path, monkeypatch, flag):
    path, _, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--selection", str(path), "--output", str(tmp_path / "out"), flag])
    assert not (tmp_path / "out").exists()


def test_gpu_geometry_and_mmq_cost_ledgers_remain_independent_and_not_truncated():
    result = dispatch_result()
    meta = result.serving.batches[0].cost.metadata
    meta["gpu_invocations"] = {"projection_tasks": 12, "audited_tasks": 12, "applied_tasks": 10,
        "conditional_tasks": 10, "native_dispatch_proven_tasks": 0,
        "uncovered_reason_counts": {"physical_alias_unproven": 2}}
    meta["mmq_source_work"] = {"audited_tasks": 8, "applied_tasks": 5,
        "matrix_tasks": 3, "conversion_tasks": 2,
        "uncovered_reason_counts": {"unsupported_k_geometry": 3}}
    batch = adapter.batch_schedule(result, [])["batches"][0]
    assert "gpu_invocations" not in batch["cost_metadata"]
    assert batch["gpu_invocations"] == meta["gpu_invocations"]
    assert batch["mmq_source_work"] == meta["mmq_source_work"]
    summary = adapter.retained_dispatch_summary(result)
    assert summary["gpu_invocations"]["applied_tasks"] == 10
    assert summary["gpu_invocations"]["native_dispatch_proven_tasks"] == 0
    assert summary["mmq_source_work"]["applied_tasks"] == 5
    assert summary["mmq_source_work"]["uncovered_reason_counts"] == {"unsupported_k_geometry": 3}
    missing = adapter.retained_dispatch_summary(dispatch_result(with_summary=False))
    assert missing["gpu_invocations"]["applied_tasks"] is None
    assert missing["mmq_source_work"]["applied_tasks"] is None


def test_gpu_invocation_identical_captures_share_one_derivation_but_preserve_each_identity(tmp_path, monkeypatch):
    _, _, row, _ = fixture(tmp_path, monkeypatch)
    api, source_path, _, binding_ref = fake_gpu_invocation_contract_api(tmp_path, monkeypatch)
    second = copy.deepcopy(row)
    second["cell_id"] = "second-cell"
    runtime = fake_gpu_runtime_binding(tmp_path, row, binding_ref)
    second_raw = document(tmp_path / "second-gpu-record.json", {"execution_environment": {"GGML_CUDA_DISABLE_FUSION": None}})
    runtime["cells"][second["cell_id"]] = {**runtime["cells"][row["cell_id"]], "native_record_refs": [second_raw]}
    calls = []
    derive = api.derive_llama_gpu_invocation_contract
    def observe(*args, **kwargs):
        calls.append(kwargs)
        return derive(*args, **kwargs)
    api.derive_llama_gpu_invocation_contract = observe
    proof = adapter.verified_gpu_invocation_contract(source_path, [row, second], tmp_path, runtime_binding=runtime)
    assert len(calls) == 1 and proof["source_derivation_variants"] == 1
    assert proof["conditional_cell_count"] == 2
    assert proof["cells"][second["cell_id"]]["native_record_refs"] == [second_raw]


def write_imported_projection_gguf(path):
    """A small real GGUF file exercises importer metadata and physical aliases."""
    import struct
    from tests.test_gguf_parity import _kv, _str
    items = [_kv("general.architecture", 8, _str("qwen2"))]
    for key, value in {"general.file_type": 7, "qwen2.block_count": 1,
        "qwen2.embedding_length": 256, "qwen2.feed_forward_length": 512,
        "qwen2.attention.head_count": 4, "qwen2.attention.head_count_kv": 2,
        "qwen2.context_length": 256, "general.vocab_size": 64}.items():
        items.append(_kv(key, 4, struct.pack("<I", value)))
    tensors = [("token_embd.weight", (256, 64), 8), ("output.weight", (256, 64), 8),
        ("output_norm.weight", (256,), 0), ("blk.0.attn_norm.weight", (256,), 0),
        ("blk.0.ffn_norm.weight", (256,), 0), ("blk.0.attn_q.weight", (256, 256), 8),
        ("blk.0.attn_k.weight", (256, 128), 8), ("blk.0.attn_v.weight", (256, 128), 8),
        ("blk.0.attn_output.weight", (256, 256), 8), ("blk.0.ffn_gate.weight", (256, 512), 8),
        ("blk.0.ffn_up.weight", (256, 512), 8), ("blk.0.ffn_down.weight", (512, 256), 8)]
    directory, offset = [], 0
    for name, shape, kind in tensors:
        directory.append(_str(name) + struct.pack("<I", len(shape)) + struct.pack("<" + "Q" * len(shape), *shape)
            + struct.pack("<IQ", kind, offset))
        elements = 1
        for extent in shape:
            elements *= extent
        size = elements * 4 if kind == 0 else elements // 32 * 34
        offset += (size + 31) // 32 * 32
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(items)) + b"".join(items) + b"".join(directory)
    path.write_bytes(header + b"\0" * ((-len(header)) % 32 + offset))
    return path


@pytest.mark.parametrize("costs", [False, True])
def test_gpu_invocation_real_gguf_import_nested_identity_and_final_replan(tmp_path, costs):
    from heterollm_sim.gguf_parity import read_gguf_metadata, build_model_from_gguf
    from heterollm_sim.control_plane_state import mapping_fingerprint_status
    from heterollm_sim import planner
    from tools.native_llama_compare import build_matching_scenario
    from tests.test_llama_gpu_invocations import contract
    gguf = read_gguf_metadata(write_imported_projection_gguf(tmp_path / "actual-import.gguf"))
    model = build_model_from_gguf(gguf)
    assert model.metadata.get("gguf_sha256") is None
    assert model.metadata["metadata"]["gguf_sha256"] == gguf.sha256
    case = build_matching_scenario(4, 2, model=model, ctx=128, parallel=1, batch=64, ubatch=64,
        threads=16, gpu_layers=-1, runtime_binary=adapter.ROOT / adapter.grid.RUNTIME)
    source = contract()
    proof = {"contract": source, "mmq_source_costs_requested": costs}
    inputs = {"gpu_invocation_contract": source, "gpu_invocation_evidence": proof, "gpu_mmq_source_costs": costs}
    lowered = adapter.apply_gpu_invocation_static_contract(case, inputs)
    audit = lowered.workload.metadata["llama_cpp_gpu_native_invocations"]
    assert audit["applied"] is True and audit["model_sha256"] == gguf.sha256
    assert audit["qualified_projection_groups"] == 2
    assert audit["native_dispatch_proven"] is False and audit["conditional"] is True
    assert audit["mmq_source_costs"]["requested"] is costs
    assert audit["mmq_source_costs"]["applied"] is False
    before = planner._execution_layers(case)[0].metadata["weight_projection_descriptors"]["projections"]
    after = planner._execution_layers(lowered)[0].metadata["weight_projection_descriptors"]["projections"]
    assert all(key not in before for key in ("attention.q", "attention.k", "attention.v"))
    assert all(len(after[key]["segments"]) == 1 for key in ("attention.q", "attention.k", "attention.v"))
    assert [after[key]["segments"][0]["physical_tensor_name"] for key in ("attention.q", "attention.k", "attention.v")] == ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"]
    final, evidence = adapter.replan_final_static_scenario(lowered)
    assert evidence["normal_validation_passed"] is True
    assert mapping_fingerprint_status(final)["mapping_stale"] is False
    assert final.model.embedding_weight_bytes == case.model.embedding_weight_bytes
    assert sum(layer.weight_bytes for layer in planner._execution_layers(final)) == sum(layer.weight_bytes for layer in planner._execution_layers(case))


def local_gpu_invocation_contract(*, mmq=False):
    base = adapter.ROOT / "artifacts/development/native_long_grid_135_20260915"
    name = "derived_source_contract_mmq_optional.json" if mmq else "derived_source_contract.json"
    source = base / "optimization_loop/round_006/candidate_src" / name
    if not source.is_file():
        pytest.skip("recorded GPU invocation source contract unavailable")
    return base, source


def import_actual_gguf_audited_header(model_key):
    """Validate the actual GGUF identity and audited header without inference."""
    import hashlib
    import inspect
    from heterollm_sim import gguf_parity
    base, _ = local_gpu_invocation_contract()
    audit, _ = adapter.grid.read_document(base / "optimization_loop/round_006/gpu_invocation_dispatch_audit.json")
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    evidence = next(item for item in audit["model_header_evidence"] if item["model_key"] == model_key)
    row = next(item for item in adapter.selected_rows(selection) if item["model_key"] == model_key)
    expected_sha = evidence["recorded_model_sha256_not_recomputed"]
    assert expected_sha == row["model_ref"]["sha256"]
    path = Path(evidence["path"])
    assert path.resolve() == Path(row["model_ref"]["path"]).resolve()
    if not path.is_file():
        pytest.skip("fixed real GGUF header artifact unavailable")
    # Run the production parser and its same-open-file identity checks. Old
    # test-only source rewriting bypassed those checks and broke when the
    # parser closed its historical two-open gap. This check reads the weights
    # but does not execute native inference or change any archived evidence.
    size = evidence["header_bytes"]
    assert 0 < size < 32 * 1024 * 1024
    with path.open("rb") as handle:
        header = handle.read(size)
    assert len(header) == size
    assert hashlib.sha256(header).hexdigest() == evidence["header_sha256"]
    metadata = gguf_parity.read_gguf_metadata(path)
    assert metadata.sha256 == expected_sha
    return gguf_parity.build_model_from_gguf(metadata), row


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
@pytest.mark.parametrize("model_key", ["qwen25", "smollm2", "tinyllama", "qwen35", "qwen38_gpu"])
def test_gpu_invocation_fixed_real_gguf_header_qualifies_after_import_and_replan(model_key):
    from heterollm_sim import planner
    from heterollm_sim.control_plane_state import mapping_fingerprint_status
    from tools.native_llama_compare import build_matching_scenario
    model, row = import_actual_gguf_audited_header(model_key)
    _, source = local_gpu_invocation_contract()
    contract, _ = adapter.grid.read_document(source)
    runtime = next(ref for ref in row["native_runtime_refs"] if Path(ref["path"]).name == "llama-server.exe")
    case = build_matching_scenario(4, 2, model=model, ctx=2048, parallel=1, batch=64, ubatch=64,
        threads=16, gpu_layers=-1, runtime_binary=Path(runtime["path"]), hardware_snapshot=row["static_hardware"]["frozen_hardware"])
    before_layers = planner._execution_layers(case)
    original_descriptors = copy.deepcopy([layer.metadata["weight_projection_descriptors"] for layer in before_layers])
    proof = {"contract": contract, "mmq_source_costs_requested": False}
    changed = adapter.apply_gpu_invocation_static_contract(case, {"gpu_invocation_contract": contract,
        "gpu_invocation_evidence": proof, "gpu_mmq_source_costs": False})
    audit = changed.workload.metadata["llama_cpp_gpu_native_invocations"]
    assert audit["applied"] is True
    assert audit["qualified_projection_groups"] > 0 and audit["model_sha256"] == row["model_ref"]["sha256"]
    assert audit["native_dispatch_proven"] is False and audit["conditional"] is True
    assert [layer.metadata["weight_projection_descriptors"] for layer in before_layers] == original_descriptors
    final, replanning = adapter.replan_final_static_scenario(changed)
    assert replanning["normal_validation_passed"] is True
    assert mapping_fingerprint_status(final)["mapping_stale"] is False
    assert sum(layer.weight_bytes for layer in planner._execution_layers(final)) == sum(layer.weight_bytes for layer in before_layers)
    assert final.model.embedding_weight_bytes == case.model.embedding_weight_bytes


@pytest.mark.skipif(not (adapter.ROOT / "artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json").is_file(), reason="requires the fixed local native evidence archive; not a portable unit fixture")
@pytest.mark.parametrize("costs", [False, True])
def test_gpu_invocation_real_contract_rederives_source_runtime_and_driver_probe(costs, tmp_path):
    base, source = local_gpu_invocation_contract(mmq=costs)
    selection, _ = adapter.grid.read_document(base / "stable_native_dataset.json")
    rows = adapter.selected_rows(selection)
    selected = [next(row for row in rows if row["model_key"] == "qwen25"),
        next(row for row in rows if row["model_key"] == "qwen38_gpu")]
    # Historical snapshots are immutable: reject their obsolete derivation,
    # then create a new contract from the currently locked native sources.
    from heterollm_sim.llama_gpu_invocations import derive_llama_gpu_invocation_contract
    previous, _ = adapter.grid.read_document(source)
    runtime_path = base / "optimization_loop/round_004/runtime_source_binding_structural_audit.json"
    runtime = adapter.verified_host_offload_source_contract(runtime_path, selected, adapter.ROOT)
    captured = {key: fact.get("value") for key, fact in previous["kernel_environment"].items() if fact["captured"]}
    device = previous["mmq_device_evidence"] if previous["mmq_device_evidence"]["available"] else None
    current = derive_llama_gpu_invocation_contract(runtime["contract"], captured_kernel_environment=captured,
        cuda_compute_capability=previous["cuda_compute_capability"], mmq_device_evidence=device)
    if current != previous:
        with pytest.raises(ValueError, match="differs from re-derived"):
            adapter.verified_gpu_invocation_contract(source, selected, adapter.ROOT,
                enable_mmq_source_costs=costs, runtime_binding=runtime)
    source = tmp_path / "current_gpu_invocation_contract.json"
    source.write_text(json.dumps(current), encoding="utf-8")
    proof = adapter.verified_gpu_invocation_contract(source, selected, adapter.ROOT,
        enable_mmq_source_costs=costs, runtime_binding=runtime)
    assert proof["conditional_cell_count"] == 2 and proof["uncovered_cell_count"] == 0
    assert proof["source_derivation_variants"] == 1
    assert set(proof["source_linkage"]["source_compilation"]) == {"graph", "model", "kv_cache", "mmvq", "mmq", "set_rows"}
    for cell in proof["cells"].values():
        assert cell["contract"]["cuda_compute_capability"] == 1200
        assert cell["contract"]["mmq_device_evidence"]["available"] is costs
        assert cell["kernel_environment"]["GGML_CUDA_DISABLE_FUSION"]["state"] == "captured_absent"
        assert cell["native_dispatch_proven"] is False
        assert cell["conditional_reasons"]
    assert proof["today_environment_read"] is False
    assert proof["host_offload_treatment_enabled_by_this_validation"] is False


def test_sampling_policy_static_binding_reaches_builder_without_native_answers(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    policy = {"mode": "greedy", "temperature": 0.0, "implementation": "llama_cpp_cpu_chain",
              "top_k": 1, "top_p": 0.95, "min_p": 0.05, "min_keep": 0}
    binding = {"typed_policy": policy, "backend_sampling": False,
               "effective_settings": {"ignore_eos": True}, "limitations": ["filter tail unpriced"]}
    inputs = adapter.static_inputs(row, selection, tmp_path, sampling={"cells": {row["cell_id"]: binding}})
    assert inputs["sampling_binding"] == binding
    assert "native_actuals" not in json.dumps(inputs)
    assert "native_latency_ms" not in json.dumps(inputs)
    result = adapter.predict_cell(inputs)
    actual = calls["build"][0][1]["sampling_policy"]
    assert actual.implementation == "llama_cpp_cpu_chain" and actual.min_keep == 0
    assert actual.top_k == 1 and actual.temperature == 0.0
    assert result["input_identity"]["sampling_binding"] == binding
    assert calls["run"][0].workload.metadata["native_sampling_binding"] == binding
    status = next(x for x in result["unsupported_dimensions"] if x["dimension"] == "cpu_sampling_chain")
    assert status["policy_bound"] and status["status"] == "conditional"


def test_missing_sampling_binding_remains_explicitly_unmodeled(tmp_path, monkeypatch):
    _, selection, row, calls = fixture(tmp_path, monkeypatch)
    inputs = adapter.static_inputs(row, selection, tmp_path)
    result = adapter.predict_cell(inputs)
    assert calls["build"][0][1]["sampling_policy"] is None
    status = next(x for x in result["unsupported_dimensions"] if x["dimension"] == "cpu_sampling_chain")
    assert status["status"] == "unmodeled" and not status["policy_bound"]


def test_sampling_contract_cannot_change_on_resume(tmp_path):
    with pytest.raises(SystemExit):
        adapter.main(["predict", "--output", str(tmp_path), "--resume", "--sampling-contract", str(tmp_path / "contract.json")])


@pytest.mark.parametrize("kind", ["StopIteration", "RuntimeError"])
def test_worker_failure_retains_same_attempt_traceback(tmp_path, monkeypatch, kind):
    path, _, row, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    seen = []
    def fail_at_consumer(inputs, **kwargs):
        seen.append(inputs["cell_id"])
        if kind == "StopIteration":
            next(iter(()))
        raise RuntimeError("consumer placement mismatch")
    monkeypatch.setattr(adapter, "predict_cell", fail_at_consumer)
    result_path = out / "predictions" / (row["cell_id"] + ".prediction.json")
    adapter.worker_cell(out / "freeze.json", row["cell_id"], result_path)
    result, _ = adapter.grid.read_document(result_path)
    assert result["status"] == "failed" and result["native_answers_used"] is False
    diagnostic = result["failure_diagnostic"]
    assert diagnostic["exception_type"] == kind
    assert "fail_at_consumer" in diagnostic["traceback"]
    assert kind in diagnostic["traceback"]
    assert diagnostic["freeze_ref"] == result["freeze_ref"]
    assert diagnostic["cell_id"] == row["cell_id"]
    assert diagnostic["worker_pid"] > 0 and diagnostic["automatic_retry"] is False
    assert seen == [row["cell_id"]] and not calls["run"]


def test_failed_cell_stops_new_launches_and_preserves_pending_for_diagnosis(tmp_path, monkeypatch):
    path, selection, row, calls = fixture(tmp_path, monkeypatch)
    rows = [dict(copy.deepcopy(row), cell_id=row["cell_id"] + "_failure_" + str(i)) for i in range(3)]
    selection.update(selected_cells=rows, selected_count=3, selected_cell_ids=[r["cell_id"] for r in rows])
    for group in selection["coverage"]:
        group["selected_cells"] = 3 if group["model_key"] == "qwen38_gpu" else 0
        group["excluded_cells"] = 27 - group["selected_cells"]
    document(path, seal(selection))
    out = tmp_path / "out"
    adapter.freeze_selection(path, out, data_root=tmp_path)
    before = adapter.grid.file_ref(out / "freeze.json")
    launched = []
    def failure(inputs, **kwargs):
        raise RuntimeError("deterministic output-index mismatch")
    monkeypatch.setattr(adapter, "predict_cell", failure)
    class FakeWorker:
        pid = 12345
        returncode = None
        def __init__(self, command, **kwargs):
            self.command = command
            launched.append(command[command.index("--worker-cell") + 1])
        def wait(self, timeout=None):
            command = self.command
            adapter.worker_cell(Path(command[command.index("--worker-freeze") + 1]),
                command[command.index("--worker-cell") + 1], Path(command[command.index("--worker-result") + 1]))
            self.returncode = 0
            return 0
        def poll(self):
            return self.returncode
        def terminate(self):
            raise AssertionError("started worker must finish naturally")
        kill = terminate
    monkeypatch.setattr(subprocess, "Popen", FakeWorker)
    result = adapter.run_predictions(out, workers=1, stop_on_cell_failure=True)
    assert launched == [rows[0]["cell_id"]]
    assert result["pending_cells"] == 2 and result["selected_denominator"] == 3
    assert [r["status"] for r in result["cells"]] == ["failed", "pending", "pending"]
    notice_paths = list((out / "runs").glob("run.0001.failed.*.json"))
    assert len(notice_paths) == 1
    notice, _ = adapter.grid.read_document(notice_paths[0])
    assert notice["cell_id"] == rows[0]["cell_id"]
    assert notice["automatic_retry"] is False and notice["termination_requested"] is False
    finish, _ = adapter.grid.read_document(out / "runs/run.0001.finish.json")
    assert finish["launches_stopped_after_cell_failure"] is True
    assert finish["all_started_workers_exited_and_sealed"] is True
    assert finish["launches_stopped_after_observation_interrupt"] is False
    assert not (out / "runs/coordinator.lock").exists()
    assert before == adapter.grid.file_ref(out / "freeze.json")
    assert not calls["run"]
