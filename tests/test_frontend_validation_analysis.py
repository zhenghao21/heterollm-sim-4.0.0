from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import pytest


_SPEC = spec_from_file_location("frontend_validation_analysis", Path(__file__).parents[1] / "tools/analyze_frontend_validation.py")
assert _SPEC is not None and _SPEC.loader is not None
analysis = module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def physical_report():
    return {
        "summary": {
            "dram_traffic": {"task_count": 1, "physical_bytes": 192, "physical_read_bytes": 128,
                             "physical_write_bytes": 64, "logical_bytes": 192,
                             "logical_read_bytes": 128, "logical_write_bytes": 64,
                             "service_ns": 20, "burst_count": 3, "row_hits": 2,
                             "row_misses": 1, "row_conflicts": 0},
            "storage_traffic": {"schema_version": "heterollm.nand-traffic/v1", "task_count": 0},
        },
    }


def test_unused_nand_does_not_need_fabricated_zero_counters():
    assert analysis.physical_participation({}, physical_report())["errors"] == []


def test_actual_b200_memory_activity_uses_inline_profile_without_legacy_owner_metadata():
    evidence = Path(__file__).parents[1] / "docs/frontend_native_validation_2026-10-07"
    prefix = "ui_b200_3hbf_llama3_1_405b"
    scenario = analysis.read_json(evidence / f"{prefix}_submission.json")["scenario"]
    report = analysis.read_json(evidence / f"{prefix}_result.json")["report"]
    components = {row["component_id"]: row for row in analysis.hardware(scenario)["components"]}
    assert "memory_service_owner" not in components["hbm0"]["metadata"]
    result = analysis.physical_participation(scenario, report)
    assert result["errors"] == []
    memories = {row["component_id"]: row for row in result["configured_memories"]}
    assert memories["hbm0"]["owner"] == "hbm0.hbm_fabric"
    assert memories["hbm0"]["physical_activity_observed"] is True
    assert memories["hostmem0"]["physical_activity_observed"] is True
    assert memories["hbf2"]["physical_activity_observed"] is False
    components["hbm0"]["metadata"]["memory_service_owner"] = "incorrect-owner"
    assert any("physical owner conflicts" in error for error in analysis.physical_participation(scenario, report)["errors"])
    del components["hbm0"]["execution_profile"]["parameters"]["resource_id"]
    assert any("no explicit inline physical profile resource_id" in error
               for error in analysis.physical_participation(scenario, report)["errors"])


def test_uncollected_dram_statistics_stay_null_in_participation_evidence():
    report = physical_report()
    ledger = report["summary"]["dram_traffic"]
    for key in ("read_write_switches", "refresh_wait_ns", "turnaround_wait_ns"):
        ledger[key] = None
    ledger["metric_availability"] = {"turnaround_wait_ns": {"status": "not_recorded"}}
    result = analysis.physical_participation({}, report)
    assert result["errors"] == []
    assert result["physical_traffic"]["dram_traffic"]["turnaround_wait_ns"] is None
    assert result["physical_traffic"]["dram_traffic"]["metric_availability"] == ledger["metric_availability"]


def test_inconsistent_dram_traffic_is_rejected():
    report = physical_report()
    report["summary"]["dram_traffic"]["physical_bytes"] = 191
    result = analysis.physical_participation({}, report)
    assert result["status"] == "failed"
    assert any("differs from read plus write" in error for error in result["errors"])


def test_legacy_nand_missing_logical_directions_are_unrecorded_but_physical_totals_stay_required():
    report = physical_report()
    nand = {"task_count": 1, "logical_bytes": 17, "physical_bytes": 4096,
            "physical_read_bytes": 4096, "physical_write_bytes": 0, "service_ns": 20}
    report["summary"]["storage_traffic"] = nand
    result = analysis.physical_participation({}, report)
    assert result["errors"] == []
    saved = result["physical_traffic"]["storage_traffic"]
    for key in ("logical_read_bytes", "logical_write_bytes"):
        assert saved[key] is None
        assert saved["metric_availability"][key]["status"] == "not_recorded"
    nand["physical_read_bytes"] = None
    assert any("storage_traffic.physical_read_bytes" in error
               for error in analysis.physical_participation({}, report)["errors"])
    nand["physical_read_bytes"] = 4000
    assert any("physical_bytes differs from read plus write" in error
               for error in analysis.physical_participation({}, report)["errors"])


def test_nonfinite_dram_traffic_is_reported_without_crashing():
    report = physical_report()
    report["summary"]["dram_traffic"]["service_ns"] = None
    assert analysis.physical_participation({}, report)["status"] == "failed"


def test_error_comparison_uses_native_median_and_preserves_sign():
    result = analysis.compare_values(80, [95, 100, 105, 100, 500], "ns")
    assert result["native"]["median"] == 100
    assert result["signed_error_percent"] == -20
    assert result["absolute_error_percent"] == 20
    assert result["inside_native_observed_range"] is False


@pytest.mark.parametrize("live", [None, 0, 127, 128.0, True, 128])
def test_completed_report_requires_actual_persistent_execution_for_every_cohort(live):
    scenario = {"workload": {"requests": [{"prompt_tokens": 512, "output_tokens": 128}]}}
    report = {"requests": {"r": {"status": "finished", "visible_output_tokens": 128,
              "engine_ttft_ns": 1, "engine_tpot_ns": 2, "engine_e2e_ns": 255}},
              "summary": {"completed_requests": 1, "rejected_requests": 0, "batch_count": 128,
                          "physical_live_batch_count": live, "makespan_ns": 255,
                          "mtp": {"enabled": False}},
              "measurement_semantics": {"latency": {"primary_boundary": "engine"}}}
    errors = analysis.completed_checks(scenario, report)
    assert bool(errors) is (type(live) is not int or live != 128)
    if errors:
        assert errors == ["not every reported cohort executed through the persistent live physical kernel"]


def matched_pair():
    scenario = {
        "model": {"metadata": {"gguf_sha256": "a" * 64}},
        "hardware": {"components": [{"component_id": "gpu0", "kind": "gpu", "cost_profile_id": "gpu0"}]},
        "profiles": {"llama_cpp": {"context": 768, "flash_attn": False},
                     "components": {"gpu": {"gpu0": {"kernel_model": {"graph_enabled": False}}}}},
        "workload": {"metadata": {"llama_cpp_f32_hidden_storage": True}},
    }
    sample = {"status": "completed", "prompt_n": 512, "cache_n": 0, "visible_output_tokens": 128,
              "engine_ttft_ns": 1, "engine_tpot_ns": 2, "engine_e2e_ns": 255}
    native = {"configuration": {"context": 768, "flash_attn": "off", "cache_prompt": False,
                                 "speculative_decoding": False, "model_path": "model.gguf",
                                 "effective_env": {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "cuda_graphs_disabled": True},
              "actual_server_context": {"props_n_ctx": 768, "startup_n_ctx_slot": 768},
              "samples": [deepcopy(sample) for _ in range(5)], "warmups": [{}, {}]}
    expected = {"model": {"sha256": "a" * 64, "path": "model.gguf"}}
    return scenario, native, expected


def test_pair_context_mismatch_is_not_silently_compared():
    scenario, native, expected = matched_pair()
    assert analysis.pair_configuration_errors(scenario, scenario, native, expected) == []
    native["actual_server_context"]["props_n_ctx"] = 1024
    assert "actual native context differs from simulation" in analysis.pair_configuration_errors(
        scenario, scenario, native, expected)


@pytest.mark.parametrize("env,disabled", [({}, True), ({"GGML_CUDA_DISABLE_GRAPHS": "0"}, True),
    ({"GGML_CUDA_DISABLE_GRAPHS": "1"}, False), ({"GGML_CUDA_DISABLE_GRAPHS": 1}, True)])
def test_native_graph_disabled_requires_the_explicit_child_environment(env, disabled):
    scenario, native, expected = matched_pair()
    native["configuration"].update(effective_env=env, cuda_graphs_disabled=disabled)
    assert any("child-process environment" in error for error in
               analysis.pair_configuration_errors(scenario, scenario, native, expected))


@pytest.mark.parametrize("enabled", [None, True, 0])
def test_simulation_graph_setting_must_be_explicit_false_not_missing_or_truthy(enabled):
    scenario, native, expected = matched_pair()
    scenario["profiles"]["components"]["gpu"]["gpu0"]["kernel_model"]["graph_enabled"] = enabled
    assert any("graph_enabled=False" in error for error in
               analysis.pair_configuration_errors(scenario, scenario, native, expected))


@pytest.mark.parametrize("slug", ["qwen3_0_6b_f16", "qwen3_8_27b_mixed"])
def test_graph_gate_reads_actual_frontend_v4_inline_gpu_profile(slug):
    evidence = Path(__file__).parents[1] / "docs/frontend_native_validation_2026-10-07"
    scenario = analysis.read_json(evidence / f"ui_pair_{slug}_submission.json")["scenario"]
    gpus = [component for component in analysis.hardware(scenario)["components"]
            if component["kind"] == "gpu"]
    assert gpus and all("cost_profile_id" not in component for component in gpus)
    assert analysis.simulation_gpu_graphs_explicitly_disabled(scenario)
    # The inline input is authoritative even while redundant runtime profiles
    # still contain the old value. Check every GPU, including a later entry.
    gpus[-1]["execution_profile"]["parameters"]["kernel_model"]["graph_enabled"] = True
    assert not analysis.simulation_gpu_graphs_explicitly_disabled(scenario)


@pytest.mark.parametrize("binding", [None, {},
    {"profile_id": "", "profile_kind": "gpu", "parameters": {"kernel_model": {"graph_enabled": False}}},
    {"profile_id": "gpu0.matrix", "profile_kind": "cpu", "parameters": {"kernel_model": {"graph_enabled": False}}},
    {"profile_id": "gpu0.matrix", "profile_kind": "gpu", "parameters": {}},
])
def test_inline_gpu_profile_cannot_fall_back_to_redundant_runtime_profile(binding):
    scenario, _, _ = matched_pair()
    scenario["hardware_input"] = {"hardware": {"components": [
        {"component_id": "gpu0", "kind": "gpu", "execution_profile": binding}]}}
    assert not analysis.simulation_gpu_graphs_explicitly_disabled(scenario)


def full_pair_inputs():
    scenario, native, expected = matched_pair()
    scenario["profiles"].update({
        "host_output": {"target_component_id": "hostmem0", "vocabulary_size": 100,
                        "logits_dtype": "fp32", "logits_element_bytes": 4},
        "sampling": {"mode": "greedy", "temperature": 0.0, "implementation": "llama_cpp_cpu_chain",
                     "top_k": 40, "top_p": 0.95, "min_p": 0.05, "min_keep": 0},
        "components": {"gpu": {"gpu0": {"kernel_launch_ns": 1000.0, "kernel_model": {"graph_enabled": False}}}},
        "host_orchestration": {"submission_ns": 250.0},
        "fusion": {"flash_attention": False},
        "runtime": {"gpu_controllers": {"gpu0": {"launch_ns": 5}}},
        "cim_interconnect": {"latency_ns": 20},
    })
    scenario["workload"].update({
        "scheduler": {"max_num_seqs": 1},
        "requests": [{"request_id": "r0", "prompt_tokens": 512, "output_tokens": 128, "arrival_ns": 0,
                      "metadata": {"native_model_file": "model.gguf"}}],
    })
    scenario["workload"]["metadata"].update({
        "llama_cpp_tensor_storage_contract": {"schema": "llama.cpp.gguf.tensor-storage/v1",
                                               "embedding_output_storage_bits": 32},
        "llama_cpp_kernel_model_preset": "blackwell_analytical_v1",
        "native_rope_source_contract": {"schema": "llama.cpp.cuda-rope/v1", "strategy": "runtime_sin_cos"},
    })
    scenario["placement"] = {"kv_policy": {"dtype": "fp16", "cache_component": "gddr0"},
                             "parallel": {"tp_degree": 1}, "metadata": {"control_plane": {
                                 "policy": {"options": {"mode": "heuristic"}}}}}
    scenario["weights_resident"] = True
    return scenario, native, expected


@pytest.mark.parametrize("profile", ["host_output", "sampling", "components", "host_orchestration",
                                      "fusion", "runtime", "cim_interconnect", "llama_cpp"])
def test_pair_rejects_missing_cost_profile(profile):
    prepared, native, expected = full_pair_inputs()
    submitted = deepcopy(prepared)
    del submitted["profiles"][profile]
    errors = analysis.pair_configuration_errors(submitted, prepared, native, expected)
    assert f"submitted profiles.{profile} differs from prepared native contract" in errors


@pytest.mark.parametrize("field", ["llama_cpp_tensor_storage_contract", "llama_cpp_kernel_model_preset",
                                   "llama_cpp_f32_hidden_storage", "native_rope_source_contract"])
def test_pair_rejects_missing_explicit_execution_metadata(field):
    prepared, native, expected = full_pair_inputs()
    submitted = deepcopy(prepared)
    del submitted["workload"]["metadata"][field]
    errors = analysis.pair_configuration_errors(submitted, prepared, native, expected)
    assert f"submitted workload.metadata.{field} differs from prepared native contract" in errors


@pytest.mark.parametrize("path, value", [
    (("profiles", "sampling", "top_k"), 1),
    (("profiles", "components", "gpu", "gpu0", "kernel_launch_ns"), 0),
    (("workload", "metadata", "llama_cpp_tensor_storage_contract", "embedding_output_storage_bits"), 16),
    (("workload", "scheduler", "max_num_seqs"), 2),
    (("placement", "kv_policy", "dtype"), "q8_0"),
    (("placement", "metadata", "control_plane", "policy", "options", "mode"), "cp_sat"),
    (("weights_resident",), 1),
])
def test_pair_rejects_changed_cost_and_execution_inputs(path, value):
    prepared, native, expected = full_pair_inputs()
    submitted = deepcopy(prepared)
    target = submitted
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert analysis.pair_configuration_errors(submitted, prepared, native, expected)


def test_pair_accepts_only_known_ui_provenance_and_derived_metadata_differences():
    prepared, native, expected = full_pair_inputs()
    prepared["workload"]["metadata"]["llama_cpp_runtime_identity"] = {
        "schema": "llama.cpp.runtime-identity/v1", "status": "unbound", "source_derived": False}
    submitted = deepcopy(prepared)
    submitted["workload"]["metadata"].update({
        "ui": {"expanded": True}, "native_output_contract_evidence": {"note": "display only"},
        "llama_cpp_runtime_fingerprint": "new output",
        "llama_cpp_runtime_identity": {"source_sha256": {}, "binary_sha256": {}, "status": "partial"},
    })
    submitted["workload"]["requests"][0]["metadata"]["ui"] = {"selected": True}
    submitted["placement"]["metadata"]["control_plane"]["decision"] = {"generated": True}
    submitted["profiles"]["components"]["gpu"]["gpu0"]["kernel_launch_ns"] = 1000
    assert analysis.pair_configuration_errors(submitted, prepared, native, expected) == []
    submitted["workload"]["metadata"]["new_execution_policy"] = True
    assert any("new_execution_policy" in error for error in analysis.pair_configuration_errors(
        submitted, prepared, native, expected))


def test_completed_pair_missing_source_contract_is_not_marked_compared(tmp_path):
    prepared, native, expected = full_pair_inputs()
    submitted = deepcopy(prepared)
    del submitted["workload"]["metadata"]["llama_cpp_tensor_storage_contract"]
    slug = "qwen3_0_6b_f16"
    prefix = "ui_pair_" + slug
    expected.update(case_id=slug, scenario_file="prepared.json",
                    scenario_path="Z:/original-machine/evidence/prepared.json")
    native["status"] = "completed"
    job = {"report": {"requests": {"r0": {"engine_ttft_ns": 1, "engine_tpot_ns": 2, "engine_e2e_ns": 255}}}}
    for name, value in (("prepared.json", prepared), (prefix + "_submission.json", submitted),
                        (prefix + "_result.json", job), ("native_" + slug + ".json", native)):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    attempts = {prefix: ({"status": "completed", "validation_errors": []}, submitted, job)}
    result = analysis.analyze_pair(tmp_path, slug, attempts, {"cases": [expected]})
    assert result["status"] == "configuration_mismatch"
    assert "metrics" not in result


@pytest.mark.parametrize("name", [None, "../prepared.json", "C:/prepared.json", r"C:\prepared.json"])
def test_prepared_scenario_requires_explicit_bundled_file_not_original_machine_path(tmp_path, name):
    original = tmp_path / "original.json"
    original.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="scenario_file basename"):
        analysis.prepared_scenario_file(tmp_path, {"scenario_file": name, "scenario_path": str(original)})


def test_prepared_scenario_file_is_relative_to_evidence_directory(tmp_path):
    expected = {"scenario_file": "prepared.json", "scenario_path": "Z:/original-machine/prepared.json"}
    assert analysis.prepared_scenario_file(tmp_path, expected) == tmp_path / "prepared.json"


def captured_validation():
    from heterollm_sim.architecture_presets import materialize_architecture_payload
    return {
        "submission": {
            "name": "capacity case", "hardware": materialize_architecture_payload(
                "local-native-rtx5080-9950x3d-gddr7-ddr5"),
            "model": {"name": "Qwen3-14B", "metadata": {"preset_id": "qwen3-14b"}},
            "profiles": {"llama_cpp": dict(analysis.MATRIX_RUNTIME)},
            "workload": {"requests": [{"prompt_tokens": 512, "output_tokens": 128, "arrival_ns": 0}],
                         "mtp": None, "metadata": {"llama_cpp_kernel_model_preset": "blackwell_analytical_v1"}},
        },
        "response": {"valid": False, "errors": {"scenario": [{
            "message_en": "llama.cpp shared device-memory capacity exhausted for layer-024.attention_weights"}]}}
    }


def test_frontend_capacity_validation_has_no_invented_job(tmp_path):
    path = tmp_path / "ui_local_qwen3_14b_validation.json"
    path.write_text(json.dumps(captured_validation()), encoding="utf-8")
    row, _, job = analysis.analyze_validation(path)
    assert row["status"] == "capacity_rejected"
    assert row["scope"] == "preset_matrix"
    assert row["job_created"] is False
    assert "job_id" not in row
    assert job is None


def test_capacity_rejection_with_changed_workload_is_not_counted_as_default_case(tmp_path):
    capture = captured_validation()
    capture["submission"]["profiles"]["llama_cpp"]["context"] = 768
    path = tmp_path / "ui_local_qwen3_14b_validation.json"
    path.write_text(json.dumps(capture), encoding="utf-8")
    row, _, _ = analysis.analyze_validation(path)
    assert row["status"] == "input_contract_failed"
    assert any("context" in error for error in row["validation_errors"])


@pytest.mark.parametrize("valid", [False, True])
def test_response_only_validation_is_preserved_but_cannot_replace_run_or_claim_capacity(tmp_path, monkeypatch, valid):
    import os
    capture = captured_validation()
    prefix = "ui_local_qwen3_14b"
    raw_response = {**capture["response"], "valid": valid}
    validation = tmp_path / (prefix + "_validation.json")
    validation.write_text(json.dumps(raw_response), encoding="utf-8")
    submission = tmp_path / (prefix + "_submission.json")
    submission.write_text(json.dumps({"scenario": capture["submission"]}), encoding="utf-8")
    os.utime(submission, (100, 100))
    os.utime(validation, (200, 200))
    monkeypatch.setattr(analysis, "list_architecture_presets", lambda: [
        {"id": "local-native-rtx5080-9950x3d-gddr7-ddr5", "loadable": True}])
    monkeypatch.setattr(analysis, "list_model_presets", lambda: [{"id": "qwen3-14b"}])
    result = analysis.analyze(tmp_path)
    row = next(row for row in result["attempts"] if row["record_kind"] == "pre_run_validation")
    assert row["status"] == "validation_capture_incomplete"
    assert row["response"] == raw_response
    assert row["hardware_id"] is None and row["model_id"] is None
    assert result["matrix"][0]["latest_attempt"] == prefix
    assert result["matrix"][0]["status"] == "pending"
    assert json.loads(validation.read_text(encoding="utf-8")) == raw_response


@pytest.mark.parametrize("validation_latest", [False, True])
def test_matrix_references_unique_run_or_validation_record_for_same_prefix(tmp_path, monkeypatch, validation_latest):
    import os
    capture = captured_validation()
    prefix = "ui_local_qwen3_14b"
    validation = tmp_path / (prefix + "_validation.json")
    submission = tmp_path / (prefix + "_submission.json")
    validation.write_text(json.dumps(capture), encoding="utf-8")
    submission.write_text(json.dumps({"scenario": capture["submission"]}), encoding="utf-8")
    os.utime(validation, (300 if validation_latest else 100,) * 2)
    os.utime(submission, (200,) * 2)
    monkeypatch.setattr(analysis, "list_architecture_presets", lambda: [
        {"id": "local-native-rtx5080-9950x3d-gddr7-ddr5", "loadable": True}])
    monkeypatch.setattr(analysis, "list_model_presets", lambda: [{"id": "qwen3-14b"}])
    result = analysis.analyze(tmp_path)
    rows = {row["attempt_id"]: row for row in result["attempts"]}
    assert set(rows) == {prefix, prefix + "@validation"}
    matrix = result["matrix"][0]
    expected_id = prefix + "@validation" if validation_latest else prefix
    assert matrix["latest_attempt"] == expected_id
    assert matrix["status"] == rows[expected_id]["status"]
    renderer_spec = spec_from_file_location("validation_renderer", Path(__file__).parents[1] / "tools/render_frontend_validation_report.py")
    renderer = module_from_spec(renderer_spec)
    renderer_spec.loader.exec_module(renderer)
    assert renderer.index_attempts(result)[expected_id] is rows[expected_id]
    assert renderer.matrix_section(result, tmp_path)


def test_diagnostic_captures_are_not_matrix_attempts():
    assert analysis.attempt_scope("ui_diagnostic_native_to_preset") == "diagnostic"
    assert analysis.attempt_scope("ui_before_fix_local_qwen3_0_6b") == "diagnostic"


@pytest.mark.parametrize("tag,job_id,expected", [
    (None, "current", "not_revalidated"),
    ("physical_profile_energy_v2", "different", "not_revalidated"),
    ("physical_profile_energy_v2", "current", "physical_profile_energy_v2"),
])
def test_energy_version_requires_matching_actual_browser_job(tmp_path, monkeypatch, tag, job_id, expected):
    prefix = "ui_local_qwen3_14b"
    (tmp_path / (prefix + "_submission.json")).write_text(
        json.dumps({"scenario": captured_validation()["submission"]}), encoding="utf-8")
    (tmp_path / (prefix + "_job_created.json")).write_text(
        json.dumps({"job_id": "current", "status": "queued"}), encoding="utf-8")
    (tmp_path / "ui_jobs.json").write_text(json.dumps([
        {"prefix": prefix, "job_id": job_id, "energy_accounting": tag}]), encoding="utf-8")
    monkeypatch.setattr(analysis, "list_architecture_presets", lambda: [])
    monkeypatch.setattr(analysis, "list_model_presets", lambda: [])
    row = analysis.analyze(tmp_path)["attempts"][0]
    assert row["energy_accounting"] == expected


@pytest.mark.parametrize("tag", ["not_revalidated", "physical_profile_energy_v2"])
def test_report_shows_final_total_energy_only_for_explicit_v2_jobs(tag):
    renderer_spec = spec_from_file_location("validation_renderer", Path(__file__).parents[1] / "tools/render_frontend_validation_report.py")
    renderer = module_from_spec(renderer_spec)
    renderer_spec.loader.exec_module(renderer)
    row = {"prefix": "run", "attempt_id": "run", "hardware_id": "test", "model_id": "test",
           "energy_accounting": tag, "physical_execution": {"batch_count": 128, "physical_live_batch_count": 128},
           "participation": {"physical_traffic": {}, "total_energy_pj": 617000000000000}}
    data = {"attempts": [row], "matrix": [{"latest_attempt": "run", "status": "completed"}]}
    html = renderer.physical_participation_section(data)
    assert ('data-value="617000000000000"' in html) is (tag == "physical_profile_energy_v2")
    assert ('<td class=note>未按修复后代码复核</td>' in html) is (tag != "physical_profile_energy_v2")


@pytest.mark.parametrize("mutation", [None, "job_id", "original_total", "not_validated", "duplicate"])
def test_physical_record_repricing_is_separate_and_requires_same_job_and_original_total(mutation):
    attempt = {"prefix": "ui_test", "job_id": "j1", "status": "completed",
               "energy_accounting": "not_revalidated", "participation": {"total_energy_pj": 100}}
    entry = {"prefix": "ui_test", "job_id": "j1", "status": "repriced",
             "new_simulation_executed": False, "method": "same_execution_physical_record_repricing",
             "original_total_energy_pj": 100, "repriced_total_energy_pj": 140}
    evidence = {"schema": "frontend-physical-energy-repricing/v1", "entries": [entry],
                "source_generation_audit": {"status": "reviewed_current_producers_with_restricted_field_scope"},
                "validation": {"v2_execution_count": 1}}
    if mutation == "job_id":
        entry["job_id"] = "old"
    elif mutation == "original_total":
        entry["original_total_energy_pj"] = 99
    elif mutation == "not_validated":
        evidence["validation"]["v2_execution_count"] = 0
    elif mutation == "duplicate":
        evidence["entries"].append(deepcopy(entry))
    analysis.attach_postprocessed_energy(attempt, evidence)
    assert ("postprocessed_energy" in attempt) is (mutation is None)
    assert attempt["energy_accounting"] == "not_revalidated"
    if mutation is None:
        renderer_spec = spec_from_file_location("validation_renderer", Path(__file__).parents[1] / "tools/render_frontend_validation_report.py")
        renderer = module_from_spec(renderer_spec)
        renderer_spec.loader.exec_module(renderer)
        rendered = renderer.energy_cell(attempt)
        assert 'data-value="140"' in rendered
        assert "物理记录重计" in rendered
        assert "修复后前端运行" not in rendered


def test_capacity_error_code_is_recognized_without_english_message():
    assert analysis.capacity_rejection({"scenario": [{"code": "device_memory_capacity_exhausted"}]})


def test_model_comparison_ignores_only_ui_and_tensor_directory_order():
    original = {"graph": {"attributes": {"ui": {"x": 1}}, "tensors": [
        {"tensor_id": "b", "logical_bytes": 4}, {"tensor_id": "a", "logical_bytes": 8}]}}
    reordered = {"graph": {"attributes": {"ui": {"x": 9}}, "tensors": [
        {"tensor_id": "a", "logical_bytes": 8}, {"tensor_id": "b", "logical_bytes": 4}],
        "source_operators": [], "sub_operators": []}}
    assert analysis.comparable_model(original) == analysis.comparable_model(reordered)
    reordered["graph"]["tensors"][0]["logical_bytes"] = 16
    assert analysis.comparable_model(original) != analysis.comparable_model(reordered)
