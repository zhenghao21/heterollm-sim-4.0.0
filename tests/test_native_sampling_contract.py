"""Small source/config fixtures only: no model, GPU, latency model, or native process."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import native_sampling_contract as sampling


def file_ref(path):
    raw = Path(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return file_ref(path)


def canonical(settings):
    # Fixture encoding of the exported v1 profile, intentionally no validation.
    result = {key: copy.deepcopy(settings[key]) for key in sampling._REQUIRED}
    result["logit_bias"] = [{"token": entry["token"], "bias_kind": "negative_infinity", "wire_encoding": "json_null", "source_basis": "ignore_eos EOG list; common.cpp populates -INFINITY; server-schema.cpp appends it"} for entry in settings["logit_bias"]]
    result["generation_prompt_empty"] = True
    return result


def settings_fixture():
    return {
        "seed": 42, "temperature": 0.0, "dynatemp_range": 0.0, "dynatemp_exponent": 1.0,
        "top_k": 1, "top_p": 0.949999988079071, "min_p": 0.05000000074505806,
        "top_n_sigma": -1.0, "xtc_probability": 0.0, "xtc_threshold": 0.1, "typical_p": 1.0,
        "repeat_last_n": 64, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 64,
        "dry_sequence_breakers": ["\n", ":", '"', "*"], "mirostat": 0, "mirostat_tau": 5.0,
        "mirostat_eta": 0.1, "adaptive_target": -1.0, "adaptive_decay": 0.9,
        "ignore_eos": True, "stream": True, "n_probs": 0, "min_keep": 0, "grammar": "",
        "grammar_lazy": False, "grammar_triggers": [], "preserved_tokens": [],
        "samplers": list(sampling._ORDER), "speculative.types": "none,none", "post_sampling_probs": False,
        "backend_sampling": False, "lora": [], "logit_bias": [{"token": 5, "bias": None}], "generation_prompt": "",
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original_paths = {
        "common_defaults": "native-source/common/common.h", "common_sampling": "native-source/common/sampling.cpp",
        "common_initialization": "native-source/common/common.cpp", "native_samplers": "native-source/src/llama-sampler.cpp",
        "native_sampler_header": "native-source/include/llama.h", "server_schema": "native-source/tools/server/server-schema.cpp",
        "server_task": "native-source/tools/server/server-task.cpp", "server_context": "annotation-source/tools/server/server-context.cpp",
    }
    snapshots = {}; known = {}
    for key, relative in original_paths.items():
        original = write(tmp_path / relative, ("reviewed fixture native source " + key).encode())
        snapshot = write(tmp_path / "snapshots" / (key + ".txt"), Path(original["path"]).read_bytes())
        snapshots[key] = {"original": original, "snapshot": snapshot}; known[key] = original["sha256"]
    monkeypatch.setattr(sampling, "_REVIEWED_NATIVE_SOURCE_SHA256", known)
    for key in sampling._BASELINE_SOURCES:
        original = write(tmp_path / "changing-simulator" / (key + ".py"), ("baseline " + key).encode())
        snapshot = write(tmp_path / "snapshots" / (key + ".py"), Path(original["path"]).read_bytes())
        snapshots[key] = {"original": original, "snapshot": snapshot}
    sampling_header = write(tmp_path / "native-source/common/sampling.h", b"sampling header fixture")
    headers = [snapshots["common_defaults"]["original"], sampling_header, snapshots["native_sampler_header"]["original"]]
    header_snapshot = write(tmp_path / "annotation-evidence/header_snapshot.json", {"files": {ref["path"]: ref["sha256"] for ref in headers}})
    modules = []; ancestors = {}
    for name in sampling._CORE_MODULES:
        blob = ("fixture binary identity: " + name).encode()
        modules.append(write(tmp_path / "native-bin" / name, blob))
        ancestors[name] = write(tmp_path / "annotation-bin" / name, blob)
    am = write(tmp_path / "annotation-evidence/source_manifest.json", {"schema": "fixture annotation source"})
    nm = write(tmp_path / "native-evidence/source_manifest.json", {"schema": "fixture native source"})
    ar_data = {"schema": "llama_annotation_control_build_v1", "source_manifest_sha256": am["sha256"], "header_snapshot_sha256": header_snapshot["sha256"], "output_sha256": {ref["path"]: ref["sha256"] for ref in ancestors.values()}, "input_sha256": {snapshots["server_context"]["original"]["path"]: snapshots["server_context"]["original"]["sha256"]}}
    ar = write(tmp_path / "annotation-evidence/build_receipt.json", ar_data)
    nr_data = {"schema": "native_thread_control_build_v1", "source_manifest_sha256": nm["sha256"], "runtime_base": str(tmp_path / "annotation-bin"), "unchanged_runtime_sha256": {ref["path"]: ref["sha256"] for ref in modules}}
    nr = write(tmp_path / "native-evidence/build_receipt.json", nr_data)
    collector = write(tmp_path / "frozen-collector.py", b"raise RuntimeError('must never execute contract code')\n")
    freeze = write(tmp_path / "freeze.json", {"artifact_refs": [ar, nr, nm], "source_refs": [collector]})
    payload = {"prompt": [1, 2], "n_predict": 2, "ignore_eos": True, "cache_prompt": False, "temperature": 0, "top_k": 1, "seed": 42, "stream": True}
    settings = settings_fixture()
    response = {"generation_settings": settings, "timings": {"secret_latency": 987654321.123}, "content": "FORBIDDEN_GENERATED_TEXT", "tokens": [987654321]}
    raw = {"schema": "native-repeatability-block/v1", "key": "fixture__b00", "status": "complete", "config": {"seed": 42, "output": 2}, "payload": payload, "actual_argv": [modules[0]["path"], "--spec-type", "none"], "warmup": [], "runs": [{"requests": [{"response": response}], "engine_start_us": 987654321.123}], "native_actuals": {"secret_latency": 987654321.123}}
    raw_path = tmp_path / "raw.json"; raw_ref = write(raw_path, raw)
    row = {"cell_id": "fixture-cell", "config": {"seed": 42, "output": 2}, "source_per_cell": {"source_id": "fixture", "freeze_ref": freeze}, "native_runtime_refs": modules, "evidence_index": [{"key": raw["key"], "raw_ref": raw_ref}], "native_actuals": {"secret_latency": 987654321.123}}
    selection_path = tmp_path / "selection.json"
    selection = {"schema": "native-stable-dataset/v1", "selected_count": 1, "selected_cells": [row]}
    selection_ref = write(selection_path, selection)
    profile = canonical(settings); pid = sampling._profile_id(profile)
    cell = {"cell_id": row["cell_id"], "source_id": "fixture", "freeze_ref": freeze, "collector_source_ref": collector, "raw_configuration_evidence": [{"raw_ref": raw_ref}], "payload_projection": {k: v for k, v in payload.items() if k != "prompt"}, "sampling_profile_id": pid, "typed_policy_candidate": {"mode": "greedy", "temperature": 0.0, "implementation": "llama_cpp_cpu_chain", "top_k": 1, "top_p": settings["top_p"], "min_p": settings["min_p"], "min_keep": 0}, "backend_sampling": False}
    lineage = {"receipts": [am, ar, nm, nr], "historical_header_bindings": [{"source": ref, "historical_sha256": ref["sha256"], "snapshot_ref": header_snapshot} for ref in headers], "unchanged_binary_chain": [{"name": ref_name, "selected": ref, "annotation_ancestor": ancestors[ref_name], "native_receipt_ref": nr} for ref_name, ref in ((Path(ref["path"]).name, ref) for ref in modules)]}
    contract = {"schema": sampling._SCHEMA, "selection_ref": selection_ref, "effective_profiles": {pid: profile}, "cells": [cell], "runtime_refs": modules, "build_and_header_lineage": lineage, "source_snapshots": snapshots, "exporter_ref": collector, "status": "self-reported anything", "summary": {"selected_cells": 999999, "secret_latency": 987654321.123}}
    contract_path = tmp_path / "contract.json"; write(contract_path, contract)
    result = SimpleNamespace(root=tmp_path, contract_path=contract_path, contract=contract, selection_path=selection_path, selection=selection, selection_ref=selection_ref, row=row, rows=[row], raw_path=raw_path, raw=raw, settings=settings, snapshots=snapshots, modules=modules)
    result.save_contract = lambda: write(contract_path, contract)
    result.verify = lambda: sampling.verify_sampling_contract(contract_path, result.rows, result.selection_ref, tmp_path)
    return result


def rebind_changed_raw(f):
    new_raw = write(f.raw_path, f.raw)
    f.row["evidence_index"][0]["raw_ref"] = new_raw
    f.contract["cells"][0]["raw_configuration_evidence"][0]["raw_ref"] = new_raw
    f.selection_ref = write(f.selection_path, f.selection)
    f.contract["selection_ref"] = f.selection_ref
    profile = canonical(f.settings); pid = sampling._profile_id(profile)
    f.contract["effective_profiles"] = {pid: profile}
    f.contract["cells"][0]["sampling_profile_id"] = pid
    f.contract["cells"][0]["typed_policy_candidate"].update({key: f.settings[key] for key in ("temperature", "top_k", "top_p", "min_p", "min_keep")})
    f.save_contract()


def test_valid_contract_is_static_and_does_not_execute_exporter(fixture, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no subprocess allowed"))
    result = fixture.verify()
    cell = result["cells"]["fixture-cell"]
    assert cell["typed_policy"] == {"mode": "greedy", "temperature": 0.0, "implementation": "llama_cpp_cpu_chain", "top_k": 1, "top_p": 0.949999988079071, "min_p": 0.05000000074505806, "min_keep": 0}
    assert cell["backend_sampling"] is False
    assert result["summary"]["selected_cells"] == 1
    assert result["summary"]["resolved_request_settings_checked"] == {"warmup": 0, "runs": 1}
    text = json.dumps(result)
    assert "FORBIDDEN_GENERATED_TEXT" not in text and "987654321" not in text
    assert "native_actuals" not in text and '"timings"' not in text and "secret_latency" not in text
    assert cell["effective_settings"]["logit_bias"][0]["bias_kind"] == "negative_infinity"
    assert cell["effective_settings"]["dry_multiplier"] == 0.0


def test_baseline_simulator_originals_can_change_but_copies_are_verified(fixture):
    for key in sampling._BASELINE_SOURCES:
        Path(fixture.snapshots[key]["original"]["path"]).write_text("intentional core repair", encoding="utf-8")
    result = fixture.verify()
    evidence_paths = {ref["path"] for ref in result["evidence_refs"]}
    assert all(fixture.snapshots[key]["original"]["path"] not in evidence_paths for key in sampling._BASELINE_SOURCES)
    Path(fixture.snapshots["planner"]["snapshot"]["path"]).write_text("changed frozen copy", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        fixture.verify()


def test_profile_and_typed_policy_self_claims_cannot_bypass_raw(fixture):
    old = next(iter(fixture.contract["effective_profiles"].values()))
    old["top_p"] = 0.8
    pid = sampling._profile_id(old)
    fixture.contract["effective_profiles"] = {pid: old}
    fixture.contract["cells"][0]["sampling_profile_id"] = pid
    fixture.contract["cells"][0]["typed_policy_candidate"]["top_p"] = 0.8
    fixture.contract["status"] = "verified"
    fixture.contract["summary"] = {"all_passed": True}
    fixture.save_contract()
    with pytest.raises(ValueError, match="effective profile"):
        fixture.verify()


def test_typed_policy_alone_cannot_change_min_keep(fixture):
    fixture.contract["cells"][0]["typed_policy_candidate"]["min_keep"] = 1
    fixture.save_contract()
    with pytest.raises(ValueError, match="typed policy"):
        fixture.verify()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "empty-id"])
def test_complete_unique_cell_coverage(fixture, mutation):
    if mutation == "duplicate":
        fixture.contract["cells"].append(copy.deepcopy(fixture.contract["cells"][0]))
    elif mutation == "missing":
        fixture.contract["cells"] = []
    else:
        fixture.contract["cells"][0]["cell_id"] = ""
    fixture.save_contract()
    with pytest.raises(ValueError):
        fixture.verify()


def test_foreign_selection_and_empty_identity_rejected(fixture):
    fixture.contract["selection_ref"] = {**fixture.selection_ref, "sha256": "0" * 64}
    fixture.save_contract()
    with pytest.raises(ValueError, match="foreign selection"):
        fixture.verify()
    fixture.contract["selection_ref"]["sha256"] = ""
    fixture.save_contract()
    with pytest.raises(ValueError, match="SHA256"):
        fixture.verify()


def test_raw_bytes_and_raw_identity_are_independently_checked(fixture):
    fixture.raw["payload"]["top_k"] = 2
    write(fixture.raw_path, fixture.raw)
    with pytest.raises(ValueError, match="SHA256"):
        fixture.verify()
    new_ref = file_ref(fixture.raw_path)
    fixture.contract["cells"][0]["raw_configuration_evidence"][0]["raw_ref"] = new_ref
    fixture.save_contract()
    with pytest.raises(ValueError, match="raw refs"):
        fixture.verify()


def test_missing_or_null_profile_fields_rejected(fixture):
    profile = next(iter(fixture.contract["effective_profiles"].values()))
    profile["min_keep"] = None
    fixture.contract["effective_profiles"] = {"made-up": profile}
    fixture.save_contract()
    with pytest.raises(ValueError, match="null"):
        fixture.verify()


@pytest.mark.parametrize("field,value", [("dynatemp_range", 0.5), ("repeat_penalty", 1.2), ("presence_penalty", 0.1), ("dry_multiplier", 1.0), ("xtc_probability", 0.2), ("typical_p", 0.9), ("adaptive_target", 0.5), ("backend_sampling", True), ("top_k", True), ("seed", True), ("min_keep", None), ("grammar", "root ::= 'a'")])
def test_self_consistent_but_unqualified_native_modifiers_rejected(fixture, field, value):
    fixture.settings[field] = value
    rebind_changed_raw(fixture)
    with pytest.raises(ValueError):
        fixture.verify()


def test_changed_sampler_order_rejected_even_when_rebound(fixture):
    fixture.settings["samplers"][3], fixture.settings["samplers"][5] = fixture.settings["samplers"][5], fixture.settings["samplers"][3]
    rebind_changed_raw(fixture)
    with pytest.raises(ValueError, match="sampler order"):
        fixture.verify()


def test_every_request_is_rederived(fixture):
    second = copy.deepcopy(fixture.raw["runs"][0]["requests"][0])
    second["response"]["generation_settings"]["seed"] = 43
    fixture.raw["runs"][0]["requests"].append(second)
    rebind_changed_raw(fixture)
    with pytest.raises(ValueError, match="request/resolved"):
        fixture.verify()


def test_runtime_module_bytes_and_lineage_binding_rejected(fixture):
    Path(fixture.modules[2]["path"]).write_bytes(b"different runtime")
    with pytest.raises(ValueError, match="SHA256"):
        fixture.verify()


def test_foreign_runtime_ref_rejected(fixture):
    foreign = write(fixture.root / "foreign/llama-common.dll", b"foreign common")
    fixture.contract["runtime_refs"] = [foreign if Path(ref["path"]).name == "llama-common.dll" else ref for ref in fixture.contract["runtime_refs"]]
    fixture.save_contract()
    with pytest.raises(ValueError, match="runtime mismatch"):
        fixture.verify()


def test_caller_rows_cannot_replace_selected_raw_evidence(fixture):
    fixture.rows = copy.deepcopy(fixture.rows)
    fixture.rows[0]["evidence_index"][0]["raw_ref"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="caller raw refs"):
        fixture.verify()


def test_unreviewed_native_source_rejected_even_with_updated_claims(fixture):
    pair = fixture.contract["source_snapshots"]["native_samplers"]
    pair["original"] = write(Path(pair["original"]["path"]), b"forged source")
    pair["snapshot"] = write(Path(pair["snapshot"]["path"]), b"forged source")
    fixture.save_contract()
    with pytest.raises(ValueError, match="unreviewed native source"):
        fixture.verify()


def test_historical_header_must_match_selected_build_receipt(fixture):
    wrong = write(fixture.root / "foreign-header.json", {"files": {}})
    for item in fixture.contract["build_and_header_lineage"]["historical_header_bindings"]:
        item["snapshot_ref"] = wrong
    fixture.save_contract()
    with pytest.raises(ValueError, match="header snapshot"):
        fixture.verify()


def test_missing_native_source_proof_fails_closed(fixture):
    del fixture.contract["source_snapshots"]["common_defaults"]
    fixture.save_contract()
    with pytest.raises(ValueError, match="snapshots missing"):
        fixture.verify()
