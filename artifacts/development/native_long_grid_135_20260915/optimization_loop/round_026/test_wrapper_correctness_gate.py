"""Host-only launcher gates. All GPU launches and hardware queries are mocked."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import pytest

spec = importlib.util.spec_from_file_location("r26_wrapper_gate", Path(__file__).with_name("run_wrapper_correctness.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_current_frozen_build_is_verified_without_gpu_or_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only build check must not create/query/launch GPU process")
    monkeypatch.setattr(d.subprocess, "run", forbidden)
    monkeypatch.setattr(d.subprocess, "Popen", forbidden)
    context = d.verify_build()
    assert context["frozen_refs"]["executable"]["sha256"] == d.PINS["mmvq_gpu_correctness.exe"]
    assert context["host_evidence_scope"]["external_host_process_returncode_verified"] is False
    assert context["static"]["runtime_equivalence_verified"] is False


def test_references_reject_absent_sha_false_length_and_mutation(tmp_path):
    p = tmp_path/"data"; p.write_bytes(b"evidence")
    good = d.ref(p); assert d.check_ref(good) == good
    slash_variant = dict(good); slash_variant["path"] = p.as_posix()
    assert d.check_ref(slash_variant) == good
    for item in [None, {}, {"path": str(p), "sha256": None, "bytes": 8},
                 {"path": str(p), "sha256": good["sha256"], "bytes": True},
                 {"path": "relative", "sha256": good["sha256"], "bytes": 8}]:
        with pytest.raises(ValueError): d.check_ref(item)
    p.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="identity changed"): d.check_ref(good)


def test_environment_is_explicit_and_does_not_claim_observation(monkeypatch):
    for key in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH", "GGML_CUDA_FORCE_MMQ",
                "CUDA_LAUNCH_BLOCKING", "CAPTURE_RUNTIME_AUTHORIZED", "UNRELATED_API_KEY"):
        monkeypatch.setenv(key, "must-not-inherit")
    env, dirs = d.environment_for({"expected_runtime_dlls": {
        "ggml-base.dll": {"path": "F:/locked/bin/ggml-base.dll"},
        "cudart64_12.dll": {"path": "E:/cuda/bin/cudart64_12.dll"}}})
    assert env["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert set(dirs) == {str(Path("F:/locked/bin")), str(Path("E:/cuda/bin"))}
    assert not any(key in env for key in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH", "GGML_CUDA_FORCE_MMQ",
                                          "CUDA_LAUNCH_BLOCKING", "CAPTURE_RUNTIME_AUTHORIZED", "UNRELATED_API_KEY"))


def test_live_native_simulation_unknown_python_and_wrapper_are_rejected():
    records = [
        {"pid": 1, "name": "llama-server.exe", "cmdline": []},
        {"pid": 2, "name": "python.exe", "cmdline": ["python", "predict_stable_native_dataset.py", "--worker-cell"]},
        {"pid": 3, "name": "mmvq_gpu_correctness.exe", "cmdline": []},
        {"pid": 4, "name": "python.exe", "cmdline": None},
        {"pid": 5, "name": "python.exe", "cmdline": ["python", "run_wrapper_correctness.py", "--check"]},
    ]
    assert [v["pid"] for v in d.process_conflicts(records)] == [1, 2, 3, 4]
    assert all("cmdline" not in v for v in d.process_conflicts(records))


def test_missing_campaign_closure_is_not_treated_as_idle(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "CLOSED_CAMPAIGN", tmp_path/"absent.json")
    with pytest.raises(ValueError, match="closure missing"): d.campaign_finished()


def test_active_campaign_prevents_even_output_creation(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(d, "P", tmp_path)
    def reject(): raise ValueError("campaign still active")
    monkeypatch.setattr(d, "campaign_finished", reject)
    monkeypatch.setattr(d, "verify_build", lambda: called.append("build"))
    monkeypatch.setattr(d, "launch_process", lambda *a: called.append("launch"))
    with pytest.raises(ValueError, match="still active"): d.execute()
    assert called == [] and not (tmp_path/"wrapper_correctness_run.0001").exists()


def mock_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "P", tmp_path)
    context = {"inputs": {}, "hardware": {}, "frozen_refs": {"identity": "same"},
               "prerequisite": {}, "host_evidence_scope": {"external_host_process_returncode_verified": False}}
    monkeypatch.setattr(d, "campaign_finished", lambda: {"closure": "same"})
    monkeypatch.setattr(d, "verify_build", lambda: context)
    monkeypatch.setattr(d, "assert_idle", lambda: None)
    monkeypatch.setattr(d, "environment_for", lambda _: ({"GGML_CUDA_DISABLE_GRAPHS": "1"}, [str(tmp_path)]))
    def hardware(label, output, *_):
        value = {"label": label}; d.write_new(output/(label+".json"), value); return value
    monkeypatch.setattr(d, "capture_hardware", hardware)
    monkeypatch.setattr(d, "compare_hardware", lambda *a: {"performance_stability_qualified": False})
    return context


def test_existing_run_directory_is_never_reused(monkeypatch, tmp_path):
    mock_execution(monkeypatch, tmp_path)
    output = tmp_path/"wrapper_correctness_run.0001"; output.mkdir(); (output/"old").write_bytes(b"preserve")
    monkeypatch.setattr(d, "launch_process", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError): d.execute()
    assert (output/"old").read_bytes() == b"preserve" and not (output/"start.json").exists()


def test_startup_exception_preserves_failed_terminal_and_both_hardware_records(monkeypatch, tmp_path):
    mock_execution(monkeypatch, tmp_path)
    def reject(*a, **k): raise OSError("cannot start wrapper")
    monkeypatch.setattr(d, "launch_process", reject)
    with pytest.raises(ValueError, match="saved failure terminal"): d.execute()
    output = tmp_path/"wrapper_correctness_run.0001"; finish = d.read_json(output/"finish.json")
    assert finish["status"] == "rejected" and finish["returncode"] is None
    assert "cannot start wrapper" in finish["execution_error"] and finish["identity_unchanged"] is True
    assert finish["numerical_qualification"] is False and finish["performance_parameters_admitted"] == 0
    assert (output/"hardware_before.json").is_file() and (output/"hardware_after.json").is_file()


def test_missing_raw_result_fails_even_with_zero_exit(monkeypatch, tmp_path):
    mock_execution(monkeypatch, tmp_path)
    monkeypatch.setattr(d, "launch_process", lambda *a, **k: 0)
    with pytest.raises(ValueError, match="saved failure terminal"): d.execute()
    finish = d.read_json(tmp_path/"wrapper_correctness_run.0001/finish.json")
    assert finish["returncode"] == 0 and finish["status"] == "rejected"
    assert "raw correctness result missing" in finish["execution_error"]


def test_numeric_pass_cannot_override_postrun_identity_failure(monkeypatch, tmp_path):
    context = mock_execution(monkeypatch, tmp_path)
    calls = 0
    def verify():
        nonlocal calls; calls += 1
        if calls > 1: raise ValueError("source drift")
        return context
    monkeypatch.setattr(d, "verify_build", verify)
    def launch(argv, env, cwd, stdout, stderr, output):
        dump(output/"correctness.json", {}); return 0
    monkeypatch.setattr(d, "launch_process", launch)
    monkeypatch.setattr(d, "qualify_raw", lambda *a: ({"device": {}}, {"numerical_qualification": True}))
    with pytest.raises(ValueError, match="saved failure terminal"): d.execute()
    finish = d.read_json(tmp_path/"wrapper_correctness_run.0001/finish.json")
    assert finish["raw_numerical_checks"]["numerical_qualification"] is True
    assert finish["identity_unchanged"] is False and finish["numerical_qualification"] is False
    assert finish["performance_qualified"] is False


def test_numeric_pass_remains_separate_from_performance(monkeypatch, tmp_path):
    mock_execution(monkeypatch, tmp_path)
    def launch(argv, env, cwd, stdout, stderr, output):
        dump(output/"correctness.json", {}); return 0
    monkeypatch.setattr(d, "launch_process", launch)
    monkeypatch.setattr(d, "qualify_raw", lambda *a: ({"device": {}}, {"numerical_qualification": True}))
    d.execute()
    finish = d.read_json(tmp_path/"wrapper_correctness_run.0001/finish.json")
    assert finish["status"] == "synthetic_wrapper_numerically_qualified"
    assert finish["numerical_qualification"] is True
    assert finish["runtime_equivalence_qualified"] is finish["performance_qualified"] is False
    assert finish["timed_runs"] == finish["performance_parameters_admitted"] == 0


def test_interrupted_child_observation_never_requests_termination(monkeypatch, tmp_path):
    class FakeChild:
        pid = 77
        def wait(self): raise KeyboardInterrupt("observer stopped")
        def poll(self): return None
        def kill(self): pytest.fail("must not kill child")
        def terminate(self): pytest.fail("must not terminate child")
    monkeypatch.setattr(d.subprocess, "Popen", lambda *a, **k: FakeChild())
    with pytest.raises(RuntimeError, match="no termination requested"):
        d.launch_process(["fake"], {}, str(tmp_path), None, None, tmp_path)
    failure = d.read_json(tmp_path/"child_observation_error.json")
    assert failure["pid"] == 77 and failure["child_may_be_live"] is True
    assert failure["observed_returncode"] is None and failure["termination_requested"] is False


def test_hardware_csv_rejects_missing_and_multiple_rows():
    valid = "0, GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b, Fixture GPU, 00000000:01:00.0, 581.00, 32768, P8, 40, 300, 400, 0, 256\n"
    assert d.parse_gpu_csv(valid)["memory.total"] == "32768"
    for invalid in ("", valid+valid, valid.replace("581.00", "N/A"), valid.replace("32768", "N/A")):
        with pytest.raises(ValueError): d.parse_gpu_csv(invalid)
    assert d.parse_compute_csv("") == []
    with pytest.raises(ValueError): d.parse_compute_csv("N/A, unknown\n")


def test_hardware_changes_are_not_ignored():
    expected = {"gpu_uuid": "GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b", "attributes": {
        "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR": 12, "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR": 0,
        "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT": 84, "CU_DEVICE_ATTRIBUTE_WARP_SIZE": 32,
        "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE": 67108864}}
    before = {"tool_ref": {"same": True}, "compute_processes": [], "gpu": dict(index="0", uuid=expected["gpu_uuid"], name="Fixture GPU", **{"pci.bus_id": "0000:01:00.0", "driver_version": "581.00", "memory.total": "32768"})}
    device = dict(index=0,uuid_hex=expected["gpu_uuid"].removeprefix("GPU-").replace("-", ""),name="Fixture GPU",
                  compute_major=12,compute_minor=0,sm_count=84,warp_size=32,l2_cache_bytes=67108864,
                  cuda_driver_version=13000,cuda_runtime_version=12080,runtime_header_version=12080)
    assert d.compare_hardware(before, copy.deepcopy(before), device, expected)["performance_stability_qualified"] is False
    for key, value in [("uuid", "different"), ("driver_version", "changed"), ("memory.total", "123")]:
        after = copy.deepcopy(before); after["gpu"][key] = value
        with pytest.raises(ValueError, match="changed during"): d.compare_hardware(before, after, device, expected)
    bad_device = dict(device); bad_device["sm_count"] = 83
    with pytest.raises(ValueError, match="property mismatch"): d.compare_hardware(before, before, bad_device, expected)


def numeric_fixture(tmp_path):
    result_path = tmp_path/"correctness.json"
    hashes = {}; artifacts = {}
    specs = {
        "packed_weight_q5_0": (".weights.q5_0.bin", bytes(8650752), "packed_q5_0"),
        "input_f32": (".input.f32.bin", bytes(16384), "input_f32"),
        "expected_q8_1": (".expected.q8_1.bin", bytes(4608), "expected_q8_1"),
        "actual_q8_1": (".actual.q8_1.bin", bytes(4608), None),
        "reference_f64": (".reference.f64.bin", struct.pack("<3072d", *([1.0]*3072)), "reference_f64"),
        "bounds_f64": (".bounds.f64.bin", struct.pack("<3072d", *([0.01]*3072)), "bounds_f64"),
        "actual_output_f32": (".actual.output.f32.bin", struct.pack("<3072f", *([1.0]*3072)), None),
    }
    for name, (suffix, data, key) in specs.items():
        path = Path(str(result_path)+suffix); path.write_bytes(data); artifacts[name] = d.ref(path)
        if key: hashes[key] = artifacts[name]["sha256"]
    refs = {}
    for name in ["program.exe", "inputs.json", "ggml-base.dll", "cudart64_12.dll"]:
        path = tmp_path/name; path.write_bytes(name.encode()); refs[name] = d.ref(path)
    tolerance = {"fixture_test_bound": 0.01}
    context = {"host": {"fixture_hashes": hashes, "tolerance": tolerance},
               "frozen_refs": {"executable": refs["program.exe"], "inputs": refs["inputs.json"]},
               "inputs": {"expected_runtime_dlls": {k: refs[k] for k in ("ggml-base.dll", "cudart64_12.dll")}}}
    raw = dict(schema="heterollm.mmvq-wrapper-correctness/v1", status="synthetic_correctness_passed_runtime_equivalence_unverified",
               cpu_only=False,cuda_api_calls_started=True,gpu_execution_performed=True,timed_runs=0,performance_parameters_admitted=0,
               target_llm_latency_used=False,runtime_equivalence_verified=False,fixture=d.FIXTURE,main_shim_called=True,
               allocation_guards_intact=True,cuda_errors_or_validation_failures=[],tolerance=tolerance,fixture_hashes=hashes,
               identity={"executable": refs["program.exe"], "build_inputs": refs["inputs.json"], "ggml_quantize_chunk_provider": refs["ggml-base.dll"],
                         "loaded_modules": [refs["ggml-base.dll"], refs["cudart64_12.dll"]]},raw_artifacts=artifacts,
               conversion=dict(bytes_compared=4608,byte_mismatches=0),
               main_output=dict(tested=3072,failed=0,worst_index=-1,max_absolute_error=0.0,max_bound_ratio=0.0))
    dump(result_path, raw)
    return result_path, raw, context


def test_independent_raw_byte_check_accepts_valid_fixture(tmp_path):
    path, raw, context = numeric_fixture(tmp_path)
    _, result = d.qualify_raw(path, context, tmp_path)
    assert result["output_rows_checked"] == 3072 and result["conversion_bytes_compared"] == 4608
    assert result["numerical_qualification"] is True and result["performance_parameters_admitted"] == 0


def test_reported_zero_error_cannot_hide_wrong_output_bytes(tmp_path):
    path, raw, context = numeric_fixture(tmp_path)
    p = Path(raw["raw_artifacts"]["actual_output_f32"]["path"])
    p.write_bytes(struct.pack("<3072f", *([2.0]+[1.0]*3071)))
    raw["raw_artifacts"]["actual_output_f32"] = d.ref(p); dump(path, raw)
    with pytest.raises(ValueError, match="independent output check exceeds"): d.qualify_raw(path, context, tmp_path)


def test_converted_bytes_must_equal_frozen_reference(tmp_path):
    path, raw, context = numeric_fixture(tmp_path)
    p = Path(raw["raw_artifacts"]["actual_q8_1"]["path"]); p.write_bytes(b"X"+bytes(4607))
    raw["raw_artifacts"]["actual_q8_1"] = d.ref(p); dump(path, raw)
    with pytest.raises(ValueError, match="differs from frozen CPU reference"): d.qualify_raw(path, context, tmp_path)


def test_unknown_execution_flags_are_not_promoted_to_pass(tmp_path):
    path, raw, context = numeric_fixture(tmp_path)
    del raw["allocation_guards_intact"]; dump(path, raw)
    with pytest.raises(ValueError, match="evidence incomplete"): d.qualify_raw(path, context, tmp_path)
    raw["allocation_guards_intact"] = True; raw["runtime_equivalence_verified"] = True; dump(path, raw)
    with pytest.raises(ValueError, match="incorrectly claims"): d.qualify_raw(path, context, tmp_path)
