"""Mechanistic evaluation of a frozen stable-native selection; never blind.

Selection interface: native-stable-dataset/v1 from native_162_dataset.py.
selected_cells[] supplies config, model_ref, native_runtime_refs, static_hardware
and native_actuals. Only explicit static fields enter the frozen worker inputs;
native_actuals and metric values are read only by the independent scoring step.

--selection FILE --output DIR --data-root ROOT --freeze-only creates the freeze.
--output DIR --resume runs bounded subprocesses from the copied source tree.
--output DIR --score reads answers afterwards, with optional --native-report.
"""
from __future__ import annotations
import argparse
import ast
import inspect
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from collections.abc import Mapping
import csv
import io
import json
from dataclasses import replace
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(import_root))
from tools import native_grid_predict as grid
from tools import native_162_dataset as selector
from heterollm_sim.config import SamplingPolicy

METRICS = grid.METRICS
ALIASES = dict(zip(("ttft", "tpot", "e2e"), METRICS))
PREDICTION_TYPE = "mechanistic_development_post_selection_conditional"
STATIC_KEYS = (
    "model", "model_sha256", "prompt_token_ids", "expected_prompt_tokens",
    "prompt_tokens", "output", "output_tokens", "parallel", "gpu_layers",
    "batch", "ubatch", "threads", "threads_batch", "seed", "environment",
    "worker_cpu_mask", "poll", "priority", "kv_unified_per_slot", "ctx",
    "context", "op_offload", "coherent_dma_mode", "gpu_sm_clock_mhz",
    "gpu_sm_clock_samples_mhz", "gpu_clock_source", "flash_attn", "kv_type_k",
    "kv_type_v", "kv_unified", "cont_batching", "warmup", "mmap", "mlock",
    "offload_kqv", "split_mode", "main_gpu", "cache_ram_mib", "fit_params",
    "compiled_cuda_graphs", "resolved_cpu_affinity", "poll_batch")


def now():
    return datetime.now(timezone.utc).isoformat()


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return value


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return value


def selected_rows(selection):
    if selection.get("schema") != "native-stable-dataset/v1":
        raise ValueError("native-stable-dataset/v1 required")
    if not selector.verify_selection_payload(selection):
        raise ValueError("selection payload SHA256 mismatch")
    rows = selection.get("selected_cells")
    if not isinstance(rows, list):
        raise ValueError("selected_cells[] required")
    if selection.get("selected_count", len(rows)) != len(rows):
        raise ValueError("selected denominator mismatch")
    seen = set()
    for row in rows:
        ident = row.get("cell_id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", ident) or ident in seen:
            raise ValueError("unsafe or duplicate cell_id")
        if not isinstance(row.get("config"), dict):
            raise ValueError("selected cell requires frozen config")
        if row.get("model_key") not in selector.GROUPS:
            raise ValueError("selected cell model/placement group is unknown")
        seen.add(ident)
    if selection.get("selected_cell_ids") != [row["cell_id"] for row in rows]:
        raise ValueError("selected_cell_ids differ from selected rows/order")
    coverage = selection.get("coverage")
    if not isinstance(coverage, list) or len(coverage) != len(selector.GROUPS) or {c.get("model_key") for c in coverage} != set(selector.GROUPS):
        raise ValueError("selection must retain all six model/placement coverage groups")
    for group in coverage:
        expected = sum(row.get("model_key") == group["model_key"] for row in rows)
        if group.get("selected_cells") != expected:
            raise ValueError("coverage selected count differs from selected rows")
    return rows


def measurement_clock_snapshot(static_hardware, gpu_uuid, data_root):
    """Read hardware fields only from bound native state records, not timings."""
    samples, refs = [], []
    for source in static_hardware.get("state_refs", []):
        reference = source["raw_ref"]
        document, ref = grid.read_document(grid.resolve_data(reference["path"], data_root), reference["sha256"])
        refs.append(ref)
        for key in ("state_measurement_before", "state_after"):
            state = document.get(key, {}).get("gpu_state", {})
            if state.get("returncode") != 0:
                raise ValueError("frozen measurement GPU state unavailable")
            matches = [r for r in csv.reader(io.StringIO(state.get("stdout", ""))) if len(r) >= 4 and r[0].strip() == gpu_uuid]
            if len(matches) != 1:
                raise ValueError("measurement GPU UUID missing or ambiguous")
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:MHz)?\s*", matches[0][3])
            if not match:
                raise ValueError("invalid frozen measurement SM clock")
            samples.append(float(match.group(1)))
    if samples:
        return samples, "native_state_measurement_before_and_after", refs
    # A declared locked target is better than pre-lock hardware capture clocks.
    target = static_hardware.get("configured_clock", {}).get("expected_gpu_sm_clock_mhz")
    return [positive(target, "configured locked GPU clock")], "configured_locked_target_without_sampled_readback", refs


_COMPILED_GRAPH_CACHE = {}


def compiled_graph_evidence(native_refs, data_root, *, verified_audit=None):
    """Bind the shared CUDA artifact to its own CMake setting, never old dispatch."""
    native_cuda = next((r for r in native_refs if Path(r["path"]).name.lower() == "ggml-cuda.dll"), None)
    if not native_cuda:
        return {"compiled_cuda_graphs": None, "reason": "CUDA artifact not recorded"}
    if verified_audit is not None:
        if native_cuda["sha256"] != verified_audit["artifact_sha256"]:
            raise ValueError("runtime build audit does not match selected CUDA artifact")
        return dict(verified_audit)
    key = (str(data_root), native_cuda["sha256"])
    if key in _COMPILED_GRAPH_CACHE:
        return _COMPILED_GRAPH_CACHE[key]
    base = data_root / "source/llama.cpp-semantic/build-semantic-direct"
    cmake, cuda = base / "CMakeCache.txt", base / "bin/ggml-cuda.dll"
    result = {"compiled_cuda_graphs": None, "reason": "matching CUDA build evidence unavailable"}
    if cmake.is_file() and cuda.is_file():
        cuda_ref = grid.file_ref(cuda)
        if cuda_ref["sha256"] == native_cuda["sha256"]:
            match = re.search(r"^GGML_CUDA_GRAPHS:BOOL=(ON|OFF)$", cmake.read_text(encoding="utf-8"), re.M)
            if match:
                result = {"compiled_cuda_graphs": match[1] == "ON", "cmake_ref": grid.file_ref(cmake), "cuda_ref": cuda_ref, "binding": "identical_shared_CUDA_binary_only_not_runtime_dispatch"}
    _COMPILED_GRAPH_CACHE[key] = result
    return result



def verified_runtime_build_audit(path, data_root):
    """Bind historical annotation-to-native build facts as NEW static inputs."""
    audit, audit_ref = grid.read_document(path)
    if audit.get("schema") != "stable-native-runtime-build-posthoc-audit/v1":
        raise ValueError("unsupported runtime build audit schema")
    expected_checks = (
        "all_131_selected_cells_same_cuda_digest", "selected_digest_matches_native_output",
        "native_unchanged_runtime_digest_matches", "native_build_complete_no_cuda_recompile",
        "native_baseline_pre_and_post_verified", "native_runtime_base_is_annotation",
        "native_protected_annotation_cuda_digest_matches", "annotation_output_digest_matches",
        "annotation_complete_link_success", "annotation_cuda_compile_success_no_graph_defines",
        "annotation_cuda_source_matches_manifest", "annotation_reused_inputs_and_runtime_postverified",
        "small_recorded_build_source_refs_match", "base_143_cuda_units_no_graph_defines", "base_cache_observed_off")
    if any(audit.get("checks", {}).get(key) is not True for key in expected_checks):
        raise ValueError("runtime build audit contains missing or failed evidence checks")
    conclusion = audit.get("conclusion", {})
    if conclusion.get("status") != "confirmed_off_from_recorded_build_provenance" or conclusion.get("post_hoc_compiled_cuda_graphs") is not False:
        raise ValueError("runtime audit does not establish compiled CUDA Graphs OFF")
    stages = {stage["stage"]: stage for stage in audit["provenance_chain"]}
    if len(stages) != len(audit["provenance_chain"]):
        raise ValueError("duplicate runtime provenance stages")
    required_stages = ("selection_to_native_output", "native_to_annotation", "annotation_compile_and_link", "base_configuration_and_preprocessor_guards")
    if any(key not in stages for key in required_stages):
        raise ValueError("runtime build provenance chain is incomplete")
    refs = audit.get("small_evidence_refs", [])
    if not refs:
        raise ValueError("runtime build audit requires source references")
    for ref in refs:
        grid.resolve_data(ref["path"], data_root)
        if Path(ref["path"]).suffix.lower() in {".dll", ".exe", ".gguf", ".obj", ".lib"}:
            raise ValueError("runtime audit reference must be recorded build/source evidence")
    verify_refs(refs)
    source, inheritance, overlay, base = [stages[name] for name in required_stages]
    def read_ref(ref):
        resolved = grid.resolve_data(ref["path"], data_root)
        return grid.read_document(resolved, ref["sha256"])[0]
    native = read_ref(source["native_build_receipt_ref"])
    native_source = read_ref(inheritance["native_source_manifest_ref"])
    annotation = read_ref(inheritance["annotation_build_receipt_ref"])
    artifact = audit["scope"]["selected_native_cuda_artifact"]["sha256"]
    def cuda_digest(receipt, field):
        values = [value for key, value in receipt.get(field, {}).items() if Path(key).name.lower() == "ggml-cuda.dll"]
        if len(values) != 1:
            raise ValueError("CUDA artifact missing or ambiguous in build receipt " + field)
        return values[0]
    if native.get("status") != "complete" or native.get("cuda_recompiled") is not False or any(native.get(key) is not True for key in ("baseline_preverified", "baseline_postverified")):
        raise ValueError("native overlay build was not fully verified")
    if annotation.get("status") != "complete" or any(annotation.get(key) is not True for key in ("old_runtime_preverified", "old_runtime_postverified", "old_link_inputs_postverified")):
        raise ValueError("annotation overlay build was not fully verified")
    inherited_cuda = (Path(native["runtime_base"]) / "ggml-cuda.dll").resolve()
    protected = [value for filename, value in native_source.get("protected_sha256", {}).items() if Path(filename).resolve() == inherited_cuda]
    if len(protected) != 1:
        raise ValueError("native source manifest does not bind the inherited annotation CUDA artifact")
    if any(value != artifact for value in (source["artifact_sha256"], inheritance["artifact_sha256"], overlay["output_cuda_sha256"],
        cuda_digest(native, "output_sha256"), cuda_digest(native, "unchanged_runtime_sha256"),
        protected[0], cuda_digest(annotation, "output_sha256"))):
        raise ValueError("annotation/native CUDA artifact identity chain disagrees")
    if Path(native["runtime_base"]).resolve() != Path(inheritance["runtime_base"]).resolve():
        raise ValueError("native runtime base differs from annotation inheritance")
    if native.get("source_manifest_sha256") != inheritance["native_source_manifest_ref"]["sha256"] or annotation.get("source_manifest_sha256") != overlay["source_manifest_ref"]["sha256"]:
        raise ValueError("source manifests do not match overlay receipts")
    for key, ref_key in (("compile_commands_sha256", "compile_commands_ref"), ("build_ninja_sha256", "build_ninja_ref"), ("header_snapshot_sha256", "header_snapshot_ref")):
        if annotation.get(key) != base[ref_key]["sha256"]:
            raise ValueError("annotation reused build input identity mismatch: " + key)
    steps = annotation.get("steps", [])
    compile_steps = [step for step in steps if step.get("label") == overlay["compile_step_label"]]
    link_steps = [step for step in steps if step.get("label") == overlay["link_step_label"]]
    if len(compile_steps) != 1 or len(link_steps) != 1 or compile_steps[0].get("returncode") != 0 or link_steps[0].get("returncode") != 0:
        raise ValueError("annotation CUDA compile/link success is not established")
    if compile_steps[0].get("argv") != overlay["compile_argv"]:
        raise ValueError("annotation compiler argv differs from bound audit")
    macro_pattern = re.compile(r'(?:-D|/D)\s*"?(?:GGML_CUDA_USE_GRAPHS|GGML_HIP_GRAPHS|GGML_MUSA_GRAPHS|USE_CUDA_GRAPH)\b')
    if macro_pattern.search(" ".join(compile_steps[0]["argv"])):
        raise ValueError("annotation compiler enables CUDA Graphs")
    commands_ref = base["compile_commands_ref"]
    commands = json.loads(Path(commands_ref["path"]).read_text(encoding="utf-8-sig"))
    cuda_commands = [item for item in commands if str(item.get("file", "")).lower().endswith(".cu")]
    if not cuda_commands or len(cuda_commands) != base["cuda_compile_units"]:
        raise ValueError("CUDA base compilation-unit coverage mismatch")
    for command in cuda_commands:
        text = command.get("command", " ".join(command.get("arguments", [])))
        if macro_pattern.search(text):
            raise ValueError("base CUDA compile command enables graph macros")
    cache_ref = base["cmake_cache_ref"]
    if not re.search(r"^GGML_CUDA_GRAPHS:BOOL=OFF$", Path(cache_ref["path"]).read_text(encoding="utf-8"), re.M):
        raise ValueError("runtime audit supporting CMake context differs")
    return {"compiled_cuda_graphs": False, "artifact_sha256": artifact,
        "binding": "verified_annotation_to_native_overlay_CUDA_build_chain",
        "audit_ref": audit_ref, "evidence_refs": refs,
        "evidence_role": "static_build_input_verified_before_this_new_prediction_freeze",
        "original_audit_role": audit.get("audit_role"), "prior_freeze_modified": False,
        "runtime_dispatch_or_cost_parity_proven": False,
        "reason": "Verified unchanged CUDA inheritance, successful annotation compile/link, and graph-disabled compiler/source evidence; no timing calibration"}



def verified_recurrent_batching_contract(path, rows, data_root):
    """Re-derive source rules and bind the static probe to selected runtimes."""
    from heterollm_sim.runtime_adapters import derive_llama_hybrid_batch_contract
    contract, contract_ref = grid.read_document(path)
    sources = contract.get("source_sha256", {})
    if not isinstance(sources, dict) or len(sources) != 4:
        raise ValueError("recurrent contract requires four source identities")
    resolved = {str(grid.resolve_data(name, data_root)): sha for name, sha in sources.items()}
    server = [Path(name) for name in resolved if Path(name).name in {"server-context.cpp", "server.cpp"}]
    hybrid = [Path(name) for name in resolved if Path(name).name == "llama-memory-hybrid.cpp"]
    if len(server) != 1 or len(hybrid) != 1:
        raise ValueError("recurrent source roots are ambiguous")
    log_ref = contract.get("runtime_log_ref", {})
    log = grid.resolve_data(log_ref["path"], data_root)
    expected_refs = [{"path": name, "sha256": sha} for name, sha in resolved.items()]
    expected_refs.append(log_ref)
    verify_refs(expected_refs)
    derived = derive_llama_hybrid_batch_contract(hybrid[0].parent.parent, server_source=server[0], runtime_log=log)
    if derived != contract:
        raise ValueError("recurrent batching contract does not equal re-derived source/runtime facts")
    receipt, receipt_ref = grid.read_document(log.with_suffix(".receipt.json"))
    if receipt.get("status") != "complete" or receipt.get("log_ref", {}).get("sha256") != log_ref["sha256"] or Path(receipt["log_ref"]["path"]).resolve() != log:
        raise ValueError("recurrent runtime log is not bound to a complete native receipt")
    raw_ref = receipt["raw_ref"]
    raw, actual_raw_ref = grid.read_document(grid.resolve_data(raw_ref["path"], data_root), raw_ref["sha256"])
    probe_freeze, probe_freeze_ref = grid.read_document(log.parent / "freeze.json")
    freeze_digest = selector.legacy.experiment.digest(probe_freeze)
    baseline_ref = raw.get("runtime_baseline_ref", {})
    baseline, actual_baseline_ref = grid.read_document(grid.resolve_data(baseline_ref["path"], data_root), baseline_ref["sha256"])
    if raw.get("status") != "complete" or raw.get("freeze_sha256") != freeze_digest or receipt.get("freeze_sha256") != freeze_digest or baseline.get("freeze_sha256") != freeze_digest:
        raise ValueError("recurrent probe raw/receipt/runtime freeze identities disagree")
    identity = baseline.get("module_identity_sha256")
    for key in ("runtime_before", "runtime_after"):
        if not identity or raw.get(key, {}).get("status") != "captured" or raw.get(key, {}).get("module_identity_sha256") != identity:
            raise ValueError("recurrent probe loaded runtime identity changed")
    modules = {Path(ref["path"]).name.lower(): ref["sha256"] for ref in baseline.get("artifacts", [])}
    selected_runtime, unloaded = {}, {}
    required_modules = {"llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll"}
    for row in rows:
        declared = {Path(ref["path"]).name.lower(): ref["sha256"] for ref in row["native_runtime_refs"]}
        if not required_modules.issubset(declared):
            raise ValueError("selected runtime is missing required server/compute modules")
        for name, sha in declared.items():
            if name in modules:
                if modules[name] != sha:
                    raise ValueError("recurrent probe runtime differs from selected native artifact: " + name)
                selected_runtime[name] = sha
            elif name in required_modules:
                raise ValueError("recurrent probe did not capture required module: " + name)
            else:
                unloaded[name] = sha
    native_receipt_path = data_root / "source/llama.cpp-native-thread-control/evidence/build_receipt.json"
    annotation_receipt_path = data_root / "source/llama.cpp-annotation-control/evidence/build_receipt.json"
    native_build, native_build_ref = grid.read_document(native_receipt_path)
    annotation_build, annotation_build_ref = grid.read_document(annotation_receipt_path)
    def named(mapping, name):
        values = [value for filename, value in mapping.items() if Path(filename).name.lower() == name]
        return values[0] if len(values) == 1 else None
    if native_build.get("status") != "complete" or native_build.get("baseline_postverified") is not True or annotation_build.get("status") != "complete" or annotation_build.get("old_link_inputs_postverified") is not True:
        raise ValueError("recurrent source/runtime build chain is not verified")
    implementation = "llama-server-impl.dll"
    if not selected_runtime.get(implementation) or any(value != selected_runtime[implementation] for value in (
        named(native_build.get("unchanged_runtime_sha256", {}), implementation),
        named(annotation_build.get("output_sha256", {}), implementation))):
        raise ValueError("recurrent server implementation is not the unchanged annotation build")
    if annotation_build.get("input_sha256", {}).get(str(server[0])) != resolved[str(server[0])]:
        raise ValueError("recurrent server source does not match its recorded compile input")
    refs = [contract_ref, *expected_refs, receipt_ref, actual_raw_ref, probe_freeze_ref, actual_baseline_ref,
            native_build_ref, annotation_build_ref]
    return {"contract": contract, "contract_ref": contract_ref, "evidence_refs": refs,
        "native_runtime_sha256": selected_runtime, "shipped_but_not_loaded_artifacts": unloaded, "validation": "source rules re-derived; captured native runtime and annotation server compilation bound",
        "limits": ["Base model/memory implementation source is checked against the declared source hashes; reused object/link build provenance remains conditional", "No native latency used in the scheduling contract"]}



def verified_slot_order_contract(path, rows, data_root, *, source_chain_contract_path=None):
    """Re-derive traversal and reuse the independently verified build/runtime chain.

    The saved slot contract names its probe, so an independently copied contract
    can locate the existing round-one chain through that probe's ancestors.
    Discover only this campaign's conventional chain path, never arbitrary JSON.
    Validating that chain does not enable recurrent scheduling.
    """
    from heterollm_sim.runtime_adapters import LLAMA_SLOT_ORDER_SCHEMA, derive_llama_slot_order_contract
    from heterollm_sim.llama_scenario import apply_llama_runtime_config
    if "slot_order_contract" not in inspect.signature(apply_llama_runtime_config).parameters:
        raise ValueError("slot-order treatment requires the opt-in runtime adapter implementation")
    contract, contract_ref = grid.read_document(path)
    payload = dict(contract)
    claimed = payload.pop("content_sha256", None)
    if claimed is not None and grid.stable_hash(payload) != claimed:
        raise ValueError("slot-order contract content SHA256 mismatch")
    if payload.get("schema") != LLAMA_SLOT_ORDER_SCHEMA:
        raise ValueError("slot-order source contract schema mismatch")
    sources = payload.get("source_sha256", {})
    if not isinstance(sources, dict) or len(sources) != 1:
        raise ValueError("slot-order contract requires one server source identity")
    server_name, server_sha = next(iter(sources.items()))
    server = grid.resolve_data(server_name, data_root)
    if server.name not in {"server-context.cpp", "server.cpp"}:
        raise ValueError("slot-order source is not an identified server implementation")
    source_ref = {"path": str(server), "sha256": server_sha}
    verify_refs([source_ref])
    binding = payload.get("source_chain_binding")
    if not isinstance(binding, dict) or binding.get("binding") != "same_server_source_as_existing_recurrent_contract":
        raise ValueError("slot-order source/build/runtime chain binding is required")
    log_ref = binding.get("runtime_log_ref")
    if not isinstance(log_ref, dict) or not log_ref.get("path") or not log_ref.get("sha256"):
        raise ValueError("slot-order source chain requires a captured runtime log identity")
    log = grid.resolve_data(log_ref["path"], data_root)
    verify_refs([{"path": str(log), "sha256": log_ref["sha256"]}])
    if source_chain_contract_path is None:
        candidates = {Path(path).resolve().parent.parent / "round_001/recurrent_source_contract.json"}
        candidates.update(parent / "optimization_loop/round_001/recurrent_source_contract.json" for parent in log.parents)
        existing = sorted({candidate.resolve() for candidate in candidates if candidate.is_file()}, key=str)
        if len(existing) != 1:
            raise ValueError("slot-order source chain contract is missing or ambiguous; supply its recurrent contract path")
        source_chain_contract_path = existing[0]
    chain = verified_recurrent_batching_contract(source_chain_contract_path, rows, data_root)
    derived = derive_llama_slot_order_contract(server, source_chain=chain["contract"])
    if derived != payload:
        raise ValueError("slot-order contract differs from re-derived source/build/runtime facts")
    refs = [contract_ref, source_ref, *chain["evidence_refs"]]
    verify_refs(refs)
    return {"contract": payload, "contract_ref": contract_ref, "evidence_refs": refs,
        "source_chain_contract_ref": chain["contract_ref"],
        "native_runtime_sha256": chain["native_runtime_sha256"],
        "validation": "source traversal re-derived; captured native runtime and annotation server compilation bound",
        "scope": payload["scope"], "native_latency_used": False,
        "qualification_required_after_static_bindings": True,
        "recurrent_treatment_enabled_by_this_validation": False,
        "limits": chain["limits"]}


def verified_host_offload_source_contract(path, rows, data_root):
    """Revalidate a saved build binding, then bind each selected native capture."""
    from tools import llama_runtime_source_binding as binding_api
    from heterollm_sim.runtime_adapters import apply_llama_cuda_op_offload
    if not {"source_contract", "cuda_backend_available"}.issubset(inspect.signature(apply_llama_cuda_op_offload).parameters):
        raise ValueError("host-offload treatment requires the source-qualified runtime adapter")
    saved_document, contract_ref = grid.read_document(path)
    saved = saved_document.get("build_binding") if saved_document.get("schema") == "llama-runtime-source-binding-validation/v1" else saved_document
    if not isinstance(saved, Mapping):
        raise ValueError("host-offload validation audit lacks its saved build binding")
    if saved.get("schema") != binding_api.SCHEMA or saved.get("status") != "verified_build_chain":
        raise ValueError("host-offload source contract must be a saved verified runtime build binding")
    references = saved.get("evidence_refs", [])
    if not isinstance(references, list) or not references:
        raise ValueError("host-offload source binding requires its input evidence references")
    audit_refs, base_refs = [], []
    for ref in references:
        if Path(ref["path"]).suffix.lower() != ".json":
            continue
        evidence_path = grid.resolve_data(ref["path"], data_root)
        actual_ref = grid.file_ref(evidence_path)
        if actual_ref["sha256"] != ref["sha256"]:
            raise ValueError("host-offload source evidence SHA256 mismatch: " + str(evidence_path))
        document = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, Mapping):
            continue
        if document.get("schema") == "stable-native-runtime-build-posthoc-audit/v1":
            audit_refs.append(actual_ref)
        if all(key in document for key in ("source_sha256_before", "source_sha256_after", "source_unchanged", "returncode")):
            base_refs.append(actual_ref)
    if len(audit_refs) != 1 or len(base_refs) != 1:
        raise ValueError("host-offload source binding has missing or ambiguous audit/base-build inputs")
    derived = binding_api.verify_llama_runtime_source_binding(audit_refs[0]["path"],
        base_build_receipt_path=base_refs[0]["path"], data_root=data_root)
    if json.loads(json.dumps(derived)) != saved:
        raise ValueError("host-offload binding differs from re-derived source/build/runtime facts")
    refs = {ref["path"]: ref for ref in [contract_ref, *derived["evidence_refs"]]}
    # CUDA availability consumes immutable device identity, not snapshot clocks.
    # A matching UUID/architecture capture already declared by another selected
    # cell can establish the same device; each run must still match that UUID.
    hardware_documents = {}
    for row in rows:
        for ref in row.get("static_hardware", {}).get("hardware_refs", []):
            if ref["path"] not in hardware_documents:
                hardware, actual_ref = grid.read_document(grid.resolve_data(ref["path"], data_root), ref["sha256"])
                hardware_documents[ref["path"]] = hardware, actual_ref
    cells = {}
    for row in rows:
        native_records = {ref["raw_ref"]["path"]: ref["raw_ref"]
            for ref in row.get("static_hardware", {}).get("state_refs", []) if isinstance(ref.get("raw_ref"), Mapping)}
        if not native_records:
            raise ValueError("host-offload source binding requires each cell's recorded runtime evidence")
        physical = row["static_hardware"].get("frozen_hardware", {}).get("gpu", {})
        hardware_refs = []
        for hardware, actual_ref in hardware_documents.values():
            gpu = hardware.get("gpu")
            if not isinstance(gpu, Mapping) or not physical.get("uuid") or gpu.get("uuid") != physical["uuid"]:
                continue
            if not all(gpu.get(key) == physical.get(key) for key in ("name", "compute_capability")):
                raise ValueError("host-offload device identity contradicts selected frozen hardware")
            hardware_refs.append(actual_ref)
        hardware_refs.sort(key=lambda ref: ref["path"])
        bound_records = [binding_api.bind_llama_cuda_op_offload_contract(derived,
            native_runtime_refs=row["native_runtime_refs"], native_record_ref=ref, data_root=data_root,
            hardware_evidence_ref=hardware_refs[0] if hardware_refs else None) for ref in native_records.values()]
        reference = bound_records[0]
        static_keys = ("status", "source_contract", "cuda_backend_available", "op_offload_enabled", "op_offload_cli_basis", "uncovered_reasons")
        if any(any(bound.get(key) != reference.get(key) for key in static_keys) for bound in bound_records[1:]):
            raise ValueError("host-offload source/environment/runtime binding differs across selected cell captures")
        contract = reference.get("source_contract")
        if reference["status"] == "verified":
            if not isinstance(contract, Mapping) or type(reference.get("op_offload_enabled")) is not bool or type(reference.get("cuda_backend_available")) is not bool:
                raise ValueError("verified host-offload binding lacks its explicit contract and runtime switches")
            environment = row.get("config", {}).get("environment", {})
            captured = contract.get("environment", {})
            name = captured.get("name")
            if name not in environment or environment[name] != captured.get("value"):
                raise ValueError("host-offload captured environment differs from selected static input")
            authored = row.get("config", {}).get("op_offload")
            if authored is not None and authored is not reference["op_offload_enabled"]:
                raise ValueError("host-offload captured CLI differs from selected static input")
        elif contract is not None:
            raise ValueError("uncovered host-offload binding must not provide an applicable source contract")
        evidence_refs = {ref["path"]: ref for bound in bound_records for ref in bound["evidence_refs"]}
        refs.update(evidence_refs)
        # Keep only static evidence. Do not carry native timing payloads into workers,
        # and let apply_llama_cuda_op_offload itself establish any F32 storage flag.
        cells[row["cell_id"]] = {key: reference.get(key) for key in (
            *static_keys, "device_evidence", "native_dispatch_proven", "performance_accuracy_validated",
            "per_operator_requirements", "host_weight_buffer_evidence", "staging_evidence", "limits")}
        cells[row["cell_id"]].update(requested=True, native_record_refs=list(native_records.values()),
            evidence_refs=list(evidence_refs.values()), storage_binding_method="existing_apply_llama_cuda_op_offload_semantics",
            hardware_evidence_scope="selected-dataset GPU identity/architecture matched to each run UUID; clocks remain from that run")
    verify_refs(list(refs.values()))
    return {"contract": saved, "contract_ref": contract_ref, "evidence_refs": list(refs.values()), "cells": cells,
        "verified_cell_count": sum(cell["status"] == "verified" for cell in cells.values()),
        "uncovered_cell_count": sum(cell["status"] != "verified" for cell in cells.values()),
        "native_latency_used": False, "runtime_build_audit_ref": audit_refs[0], "base_build_receipt_ref": base_refs[0]}


def apply_host_offload_static_contract(scenario, inputs):
    """Install a verified capability through the existing typed adapter only."""
    proof = inputs.get("host_offload_evidence")
    if not isinstance(proof, Mapping) or proof.get("status") != "verified":
        return scenario
    from heterollm_sim.runtime_adapters import apply_llama_cuda_op_offload
    contract = inputs.get("host_offload_source_contract")
    if not isinstance(contract, Mapping) or contract != proof.get("source_contract"):
        raise ValueError("host-offload static contract and verified evidence disagree")
    if type(proof.get("cuda_backend_available")) is not bool or type(proof.get("op_offload_enabled")) is not bool:
        raise ValueError("host-offload capability requires explicit captured runtime switches")
    if scenario.llama_cpp_config.op_offload is not proof["op_offload_enabled"]:
        raise ValueError("host-offload typed runtime config differs from captured CLI evidence")
    return apply_llama_cuda_op_offload(scenario, scenario.llama_cpp_config,
        source_contract=contract, cuda_backend_available=proof["cuda_backend_available"])


def verified_tensor_storage_contract(path, rows, data_root, *, f32_hidden_storage=False,
                                     runtime_binding=None, runtime_source_contract_path=None):
    """Bind four tensor rules to the actually inherited/rebuilt native modules."""
    from heterollm_sim.llama_tensor_storage import SCHEMA, derive_llama_tensor_storage_contract, apply_llama_tensor_storage_contract
    from tools import llama_runtime_source_binding as source_api
    if type(f32_hidden_storage) is not bool or "f32_hidden_storage" not in inspect.signature(apply_llama_tensor_storage_contract).parameters:
        raise ValueError("tensor-storage treatment requires an explicit boolean and its typed storage adapter")
    saved, contract_ref = grid.read_document(path)
    payload = dict(saved)
    digest = payload.pop("content_sha256", None)
    if digest is not None and digest != grid.stable_hash(payload):
        raise ValueError("tensor-storage contract content SHA256 mismatch")
    if payload.get("schema") != SCHEMA:
        raise ValueError("tensor-storage source contract schema mismatch")
    sources = payload.get("source_sha256", {})
    if not isinstance(sources, Mapping) or len(sources) != 4:
        raise ValueError("tensor-storage contract requires four source identities")
    source_files = {Path(name).name: grid.resolve_data(name, data_root) for name in sources}
    if set(source_files) != {"ggml.c", "ops.cpp", "getrows.cu", "llama-graph.cpp"}:
        raise ValueError("tensor-storage source roles are missing or ambiguous")
    source_root = source_files["ggml.c"].parents[2]
    source_refs = [{"path": str(grid.resolve_data(name, data_root)), "sha256": sha} for name, sha in sources.items()]
    verify_refs(source_refs)
    derived = derive_llama_tensor_storage_contract(source_root)
    if json.loads(json.dumps(derived)) != payload:
        raise ValueError("tensor-storage contract differs from re-derived source rules")
    if runtime_binding is None:
        runtime_source_contract_path = runtime_source_contract_path or Path(path).resolve().parent.parent / "round_004/runtime_source_binding_structural_audit.json"
        runtime_binding = verified_host_offload_source_contract(runtime_source_contract_path, rows, data_root)
    binding = runtime_binding["contract"]
    ev = source_api._Evidence(data_root)
    audit = ev.document(runtime_binding["runtime_build_audit_ref"])
    base_receipt = ev.document(runtime_binding["base_build_receipt_ref"])
    stages = {stage["stage"]: stage for stage in audit["provenance_chain"]}
    base = stages["base_configuration_and_preprocessor_guards"]
    inherited = stages["native_to_annotation"]
    compiled = stages["annotation_compile_and_link"]
    native_ref = stages["selection_to_native_output"]["native_build_receipt_ref"]
    native = ev.document(native_ref)
    native_manifest = ev.document(inherited["native_source_manifest_ref"])
    annotation = ev.document(inherited["annotation_build_receipt_ref"])
    annotation_manifest = ev.document(compiled["source_manifest_ref"])
    if source_api._identity(source_root) != source_api._identity(annotation_manifest["base_source"]):
        raise ValueError("tensor-storage source root is not the runtime's compiled base source")
    build_root = Path(annotation["base"]).resolve()
    annotation_build = Path(annotation["build"]).resolve()
    commands = ev.document(base["compile_commands_ref"])
    ninja = ev.text(base["build_ninja_ref"])
    modules = binding["runtime_modules"]
    roles = {"ggml.c": ("operators", "ggml-base.dll"), "ops.cpp": ("cpu_get_rows", "ggml-cpu.dll"),
             "getrows.cu": ("cuda_get_rows", "ggml-cuda.dll"), "llama-graph.cpp": ("input_graph", "llama.dll")}
    compilation = {}
    for name, (role, module) in roles.items():
        source = source_files[name]
        sha = sources[str(source)]
        source_api._equal_digests(sha, source_api._digest_at(base_receipt["source_sha256_before"], source),
                                 source_api._digest_at(base_receipt["source_sha256_after"], source))
        entry = source_api._compile_entry(commands, source, build_root)
        obj = Path(entry["output"]).resolve()
        source_api._ninja_source_link(ninja, source, obj, build_root, module)
        if role == "cpu_get_rows":
            continue
        if role == "operators":
            if source_api._identity(binding["source_compilation"]["operators"]["source"]) != source_api._identity(source):
                raise ValueError("tensor-storage operator source differs from verified base-module compilation")
            basis = "unchanged_ggml_base_module_from_verified_original_build"
        else:
            link = source_api._step(annotation, "link bin/" + module)
            rsp_args = [arg[1:] for arg in link["argv"] if arg.startswith("@")]
            if len(rsp_args) != 1:
                raise ValueError("tensor-storage annotation link response is ambiguous")
            rsp = Path(rsp_args[0]).resolve()
            rsp_text = ev.text({"path": str(rsp), "sha256": source_api._digest_at(annotation["input_sha256"], rsp)})
            args = source_api._tokens(rsp_text)
            if not any(source_api._identity(arg) == source_api._identity(obj) for arg in args):
                raise ValueError("tensor-storage original source object is absent from actual annotation link")
            source_api._digest_at(annotation["input_sha256"], obj)
            outputs = [arg[5:] for arg in args if arg.lower().startswith("/out:")]
            if len(outputs) != 1 or source_api._identity(outputs[0]) != source_api._identity(annotation_build / "bin" / module):
                raise ValueError("tensor-storage annotation link output does not match inherited module")
            basis = "original_source_object_explicitly_reused_by_annotation_link_then_inherited_by_native"
        compilation[role] = {"source": str(source), "sha256": sha, "object": str(obj),
            "module": modules[module], "binding": basis}
    # The CPU backend is rebuilt by native-thread-control, unlike ggml-base.
    source = source_files["ops.cpp"]
    units = [unit for unit in native_manifest["compile_units"]
             if source_api._identity(unit.get("base_file", "")) == source_api._identity(source)]
    if len(units) != 1:
        raise ValueError("tensor-storage CPU GET_ROWS native compile unit is missing or ambiguous")
    unit = units[0]
    source_api._equal_digests(unit["sha256"], sources[str(source)], source_api._digest_at(native["readonly_input_sha256"], source))
    overlay_source = Path(unit["file"]).resolve()
    ev.read({"path": str(overlay_source), "sha256": unit["sha256"]})
    step = source_api._step(native, "compile ops.cpp")
    argv = step["argv"]
    if "-c" not in argv or source_api._identity(argv[argv.index("-c")+1]) != source_api._identity(overlay_source):
        raise ValueError("tensor-storage CPU compile did not consume the byte-identical native source")
    outputs = [arg[3:] for arg in argv if arg.startswith("/Fo")]
    if len(outputs) != 1:
        raise ValueError("tensor-storage CPU compile output is ambiguous")
    cpu_build = Path(step["cwd"]).resolve()
    cpu_obj = (cpu_build / outputs[0]).resolve()
    cpu_sha = source_api._digest_at(native["output_sha256"], Path(native["runtime_output"]) / "ggml-cpu.dll")
    source_api._equal_digests(cpu_sha, native["new_cpu_dll_sha256"])
    source_api._digest_at(native["output_sha256"], cpu_obj)
    matching_commands = [command for command in native["commands"] if source_api._identity(command["file"]) == source_api._identity(overlay_source)]
    if len(matching_commands) != 1 or matching_commands[0]["argv"] != argv or source_api._identity(matching_commands[0]["output"]) != source_api._identity(cpu_obj):
        raise ValueError("tensor-storage CPU successful compile and saved command/object disagree")
    link = source_api._step(native, "link ggml-cpu.dll")
    rsp_args = [arg[1:] for arg in link["argv"] if arg.startswith("@")]
    if len(rsp_args) != 1:
        raise ValueError("tensor-storage CPU link response is ambiguous")
    script_path = Path(native_ref["path"]).resolve().parent.parent / "build_native_threads.py"
    script = ev.text({"path": str(script_path), "sha256": native["build_script_sha256"]})
    syntax = ast.parse(script)
    linker = [node.value for node in syntax.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "linkargs" for t in node.targets)]
    expected_first = ast.parse("[*(str(e['obj']) for e in entries)]", mode="eval").body.elts[0]
    if len(linker) != 1 or not isinstance(linker[0], ast.List) or ast.dump(linker[0].elts[0]) != ast.dump(expected_first):
        raise ValueError("tensor-storage recorded CPU build script does not link every compiled source object")
    expected_args = [command["output"] for command in native["commands"]] + [str(build_root / "ggml/src/ggml-base.lib"),
        "kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib", "shell32.lib", "ole32.lib", "oleaut32.lib", "uuid.lib", "comdlg32.lib", "advapi32.lib",
        "/machine:x64", "/INCREMENTAL:NO", "/dll", "/version:0.23", "/out:" + str(cpu_build / "bin/ggml-cpu.dll"),
        "/implib:" + str(cpu_build / "ggml/src/ggml-cpu.lib"), "/pdb:" + str(cpu_build / "bin/ggml-cpu.pdb")]
    response = ev.text(rsp_args[0])
    if response.splitlines() != [subprocess.list2cmdline([arg]) for arg in expected_args]:
        raise ValueError("tensor-storage CPU response differs from recorded build-script reconstruction")
    cpu_module = {"path": str(cpu_build / "bin/ggml-cpu.dll"), "sha256": cpu_sha}
    compilation["cpu_get_rows"] = {"source": str(source), "compiled_source": str(overlay_source),
        "sha256": unit["sha256"], "object": str(cpu_obj), "module": cpu_module,
        "binding": "byte_identical_source_recompiled_and_linked_by_recorded_native_thread_control_build",
        "link_response_binding": "reconstructed_from_digest_bound_build_script_and_recorded_commands"}
    for row in rows:
        selected = source_api._native_module_map(row["native_runtime_refs"])
        if "ggml-cpu.dll" not in selected or source_api._identity(selected["ggml-cpu.dll"]["path"]) != source_api._identity(cpu_module["path"]):
            raise ValueError("tensor-storage selected CPU module differs from native rebuilt output")
        source_api._equal_digests(selected["ggml-cpu.dll"]["sha256"], cpu_sha)
        for ref in runtime_binding["cells"][row["cell_id"]]["native_record_refs"]:
            record = ev.document(ref)
            for key in ("runtime_before", "runtime_after"):
                captured = source_api._native_module_map(record[key]["actual_modules"])
                module = captured.get("ggml-cpu.dll", {})
                if source_api._identity(module.get("path", "")) != source_api._identity(cpu_module["path"]):
                    raise ValueError("tensor-storage native CPU module was not captured at the compiled output path")
                source_api._equal_digests(module.get("sha256"), cpu_sha)
    refs = {ref["path"]: ref for ref in [contract_ref, *source_refs, *runtime_binding["evidence_refs"], *ev.refs.values()]}
    verify_refs(list(refs.values()))
    return {"contract": payload, "contract_ref": contract_ref, "evidence_refs": list(refs.values()),
        "runtime_source_binding_ref": runtime_binding["contract_ref"], "source_compilation": compilation,
        "f32_hidden_storage_requested": f32_hidden_storage, "native_latency_used": False,
        "host_offload_treatment_enabled_by_this_validation": False, "native_dispatch_proven": False,
        "qualification_required_after_static_bindings": True,
        "limits": ["Only the four listed generic tensor/input/GET_ROWS translation units are bound; architecture-specific qwen35.cpp source equivalence is not claimed",
                   "Row dequantization/conversion timing, repeated-index cache reuse, cache-line/page/write-allocation costs remain unmodeled",
                   "Full embedding table capacity and any existing cross-device staging remain separate from selected-row traffic"]}


def apply_tensor_storage_static_contract(scenario, inputs):
    proof = inputs.get("tensor_storage_evidence")
    if proof is None:
        return scenario
    contract = inputs.get("tensor_storage_contract")
    flag = inputs.get("tensor_storage_f32_hidden", False)
    if not isinstance(proof, Mapping) or contract != proof.get("contract") or type(flag) is not bool or flag is not proof.get("f32_hidden_storage_requested"):
        raise ValueError("tensor-storage static contract/flag differs from verified evidence")
    from heterollm_sim.llama_tensor_storage import apply_llama_tensor_storage_contract
    return apply_llama_tensor_storage_contract(scenario, contract, f32_hidden_storage=flag)


def captured_kernel_environment(record, selected_environment=None):
    """Normalize only historically recorded kernel controls; absence stays unknown."""
    observed = record.get("execution_environment", {})
    if not isinstance(observed, Mapping):
        raise ValueError("captured kernel environment must be an object")
    values, evidence = {}, {}
    for key in ("GGML_CUDA_DISABLE_FUSION",):
        if key not in observed:
            evidence[key] = {"captured": False, "state": "unknown", "value": None}
            continue
        item = observed[key]
        if isinstance(item, Mapping):
            present, value = item.get("is_set"), item.get("value")
            if type(present) is not bool or (present and not isinstance(value, str)) or (not present and value is not None):
                raise ValueError("captured kernel environment presence/value disagree")
        elif item is None or isinstance(item, str):
            value = item
            present = item is not None
        else:
            raise ValueError("captured kernel environment requires a string or observed absence")
        if isinstance(selected_environment, Mapping) and key in selected_environment and selected_environment[key] != value:
            raise ValueError("captured kernel environment differs from selected static input")
        values[key] = value
        evidence[key] = {"captured": True, "state": "captured_present" if present else "captured_absent", "value": value}
    return values, evidence


def verify_gpu_invocation_source_links(runtime_binding, data_root, *, include_context=False):
    """Bind added GPU/graph source objects to the inherited annotation modules."""
    from tools import llama_runtime_source_binding as source_api
    ev = source_api._Evidence(data_root)
    audit = ev.document(runtime_binding["runtime_build_audit_ref"])
    stages = {stage["stage"]: stage for stage in audit["provenance_chain"]}
    base = stages["base_configuration_and_preprocessor_guards"]
    annotation = ev.document(stages["native_to_annotation"]["annotation_build_receipt_ref"])
    base_receipt = ev.document(runtime_binding["base_build_receipt_ref"])
    binding = runtime_binding["contract"]
    root = Path(binding["source_paths"]["scheduler"]).parents[2]
    build = Path(annotation["base"]).resolve()
    commands, ninja = ev.document(base["compile_commands_ref"]), ev.text(base["build_ninja_ref"])
    result = {}
    for name, relative, module in (
        ("graph", "src/llama-graph.cpp", "llama.dll"),
        ("model", "src/llama-model.cpp", "llama.dll"),
        ("kv_cache", "src/llama-kv-cache.cpp", "llama.dll"),
        ("mmvq", "ggml/src/ggml-cuda/mmvq.cu", "ggml-cuda.dll"),
        ("mmq", "ggml/src/ggml-cuda/mmq.cu", "ggml-cuda.dll"),
        ("set_rows", "ggml/src/ggml-cuda/set-rows.cu", "ggml-cuda.dll"),
        *((("context", "src/llama-context.cpp", "llama.dll"),
           ("hybrid", "src/llama-memory-hybrid.cpp", "llama.dll")) if include_context else ())):
        source = root / relative
        sha = source_api._digest_at(base_receipt["source_sha256_before"], source)
        source_api._equal_digests(sha, source_api._digest_at(base_receipt["source_sha256_after"], source))
        ev.read({"path": str(source), "sha256": sha})
        entry = source_api._compile_entry(commands, source, build)
        obj = Path(entry["output"]).resolve()
        source_api._ninja_source_link(ninja, source, obj, build, module)
        if name == "context":
            # This unit is rebuilt by annotation-control, not inherited as the
            # original object. Bind the compiled overlay and its linked output.
            manifest_ref = stages["annotation_compile_and_link"]["source_manifest_ref"]
            manifest = ev.document(manifest_ref)
            source_api._equal_digests(manifest_ref["sha256"], annotation["source_manifest_sha256"])
            modified = manifest["modified_translation_units"][relative]
            source_api._equal_digests(sha, modified["before_sha256"])
            source = Path(manifest["overlay_source"]) / relative
            sha = modified["after_sha256"]
            ev.read({"path": str(source), "sha256": sha})
            source_api._equal_digests(sha, source_api._digest_at(annotation["input_sha256"], source))
            step = source_api._step(annotation, "compile " + relative)
            argv = step["argv"]
            if "-c" not in argv or source_api._identity(argv[argv.index("-c") + 1]) != source_api._identity(source):
                raise ValueError("nonflash context compile source differs from annotation overlay")
            outputs = [arg[3:] for arg in argv if arg.startswith("/Fo")]
            if len(outputs) != 1:
                raise ValueError("nonflash context compile output is ambiguous")
            obj = (Path(step["cwd"]) / outputs[0]).resolve()
        link = source_api._step(annotation, "link bin/" + module)
        responses = [arg[1:] for arg in link["argv"] if arg.startswith("@")]
        if len(responses) != 1:
            raise ValueError("GPU invocation source link response is ambiguous")
        response = Path(responses[0]).resolve()
        args = source_api._tokens(ev.text({"path": str(response), "sha256": source_api._digest_at(annotation["input_sha256"], response)}))
        if not any(source_api._identity(arg) == source_api._identity(obj) for arg in args):
            raise ValueError("GPU invocation original source object is not in the inherited module link")
        object_sha = source_api._digest_at(annotation["output_sha256"] if name == "context" else annotation["input_sha256"], obj)
        expected = Path(annotation["build"]) / "bin" / module
        outputs = [arg[5:] for arg in args if arg.lower().startswith("/out:")]
        if len(outputs) != 1 or source_api._identity(outputs[0]) != source_api._identity(expected):
            raise ValueError("GPU invocation source link output differs from inherited module")
        source_api._equal_digests(source_api._digest_at(annotation["output_sha256"], expected), binding["runtime_modules"][module]["sha256"])
        result[name] = {"source": str(source), "source_sha256": sha, "recorded_object_sha256": object_sha,
            "module": binding["runtime_modules"][module], "historical_source_content_bound": True}
    return {"source_compilation": result, "evidence_refs": list(ev.refs.values()),
        "architecture_specific_model_source_content": "conditional; historical unity include body hashes unavailable"}



def derive_nonflash_kv_view_contract(runtime_binding, data_root):
    """Derive a timing-free physical-view rule from linked native source objects."""
    from tools import llama_runtime_source_binding as source_api
    linkage = verify_gpu_invocation_source_links(runtime_binding, data_root, include_context=True)
    roles = ("kv_cache", "graph", "model", "context", "hybrid")
    texts = {role: Path(linkage["source_compilation"][role]["source"]).read_text(encoding="utf-8")
             for role in roles}
    compact = {role: re.sub(r"\s+", "", text) for role, text in texts.items()}
    # These are source assertions, never coefficients fitted to measurements.
    expected = {
        "kv_cache": ("n_stream(unified?1:n_seq_max)", "v_cells[s].resize(kv_size);",
            "constuint32_tn_pad_cur=std::max(n_pad,256u);",
            "std::min(cells.size(),std::max(n_pad_cur,GGML_PAD(cells.used_max_p1(),n_pad_cur)))",
            "kv->apply_ubatch(sinfos[i_cur],ubatches[i_cur]);n_kv=kv->get_n_kv(sinfos[i_cur]);"),
        "context": ("cparams.n_ctx=GGML_PAD(cparams.n_ctx,256);",
            "if(cparams.kv_unified){cparams.n_ctx_seq=cparams.n_ctx;"),
        "graph": ("constauton_stream=cparams.kv_unified?1:ubatch.n_seqs_unq;",
            "ggml_new_tensor_4d(ctx,type,n_kv,n_tokens/n_stream,1,n_stream)",
            "ggml_tensor*kq=ggml_mul_mat(ctx0,k,q);",
            "kq=ggml_soft_max_ext(ctx0,kq,kq_mask,kq_scale,hparams.f_max_alibi_bias);",
            "ggml_tensor*kqv=ggml_mul_mat(ctx0,v,kq);"),
        "hybrid": ("mem_attn(newllama_kv_cache(model,model.hparams,type_k,type_v,v_trans,offload,unified,kv_size,n_seq_max,n_pad,n_swa,swa_type,",),
        "model": ("res=newllama_kv_cache(*this,hparams,params.type_k,params.type_v,!cparams.flash_attn,cparams.offload_kqv,cparams.kv_unified,cparams.n_ctx_seq,cparams.n_seq_max,1,",),
    }
    for role, rules in expected.items():
        if any(rule not in compact[role] for rule in rules):
            raise ValueError("nonflash KV source rule differs: " + role)
    hybrid = re.sub(r"/\*.*?\*/|//[^\n]*", "", texts["model"], flags=re.S)
    if "res=newllama_memory_hybrid(*this,params.type_k,params.type_v,!cparams.flash_attn,cparams.n_ctx_seq,1,hparams.n_swa,hparams.swa_type," not in re.sub(r"\s+", "", hybrid):
        raise ValueError("hybrid attention cache n_pad=1 source rule missing")
    return {"schema": "heterollm.llama-nonflash-kv-view/v1", "n_pad": 1, "n_kv_padding": 256,
        "context_allocation_alignment": 256,
        "source_sha256": {linkage["source_compilation"][role]["source"]:
            linkage["source_compilation"][role]["source_sha256"] for role in roles},
        "source_compilation": {role: linkage["source_compilation"][role] for role in roles},
        "runtime_llama_module": runtime_binding["contract"]["runtime_modules"]["llama.dll"],
        "runtime_binding_sha256": runtime_binding["contract"].get("content_sha256"),
        "rules": {"extent": "min(cache_cells,max(256,pad(used_max_p1,256)))",
            "occupied_lower_bound": "largest_current_sequence_retained_context_only",
            "mask": "F32[n_kv,ubatch.n_tokens,1,1]", "nonflash": "KQ -> masked_softmax -> PV",
            "padding_source": "get_n_kv max(n_pad,256); cache constructor n_pad=1; no get_padding function in locked source"},
        "native_latency_used": False,
        "evidence_refs": linkage["evidence_refs"],
        "uncovered_reasons": ["unified_allocator_extent_holes_inactive_slots_and_shared_prefix_union_unknown"]}


def verified_nonflash_kv_view_contract(path, rows, data_root, *, runtime_binding=None, runtime_source_contract_path=None):
    saved, contract_ref = grid.read_document(path)
    if runtime_binding is None:
        if runtime_source_contract_path is None:
            raise ValueError("nonflash KV view requires --host-offload-source-contract for runtime binding")
        runtime_binding = verified_host_offload_source_contract(runtime_source_contract_path, rows, data_root)
    canonical = derive_nonflash_kv_view_contract(runtime_binding, data_root)
    if json.loads(json.dumps(canonical)) != saved:
        raise ValueError("nonflash KV view differs from re-derived source/build rules")
    cells = {}
    for row in rows:
        host = runtime_binding["cells"][row["cell_id"]]
        if host.get("status") != "verified":
            raise ValueError("nonflash KV view requires verified selected native runtime identity")
        raw = row["config"]
        slot, parallel = raw.get("kv_unified_per_slot", 2048), raw["parallel"]
        cfg = {"batch": raw.get("batch", 64), "ubatch": raw.get("ubatch", 64),
            "parallel": parallel, "simulator_slot_context_tokens": slot,
            "native_context_tokens": raw.get("context", raw.get("ctx", slot * parallel)),
            "flash_attn": raw.get("flash_attn", raw.get("flash_attention", False)),
            "kv_unified": raw.get("kv_unified", True), "kv_type_k": raw.get("kv_type_k", "f16"),
            "kv_type_v": raw.get("kv_type_v", "f16")}
        if (cfg["native_context_tokens"] != slot * parallel or cfg["native_context_tokens"] % 256
                or cfg["flash_attn"] is not False or cfg["kv_unified"] is not True
                or cfg["kv_type_k"] != "f16" or cfg["kv_type_v"] != "f16"):
            raise ValueError("nonflash KV view frozen native cache configuration unsupported")
        cells[row["cell_id"]] = {**canonical, "runtime_binding_status": "verified", "configuration": cfg}
    refs = {ref["path"]: ref for ref in [contract_ref, *runtime_binding["evidence_refs"], *canonical["evidence_refs"]]}
    return {"contract": canonical, "contract_ref": contract_ref, "cells": cells,
        "evidence_refs": list(refs.values()), "native_latency_used": False}


def apply_nonflash_kv_view_static_contract(scenario, inputs, *, gguf=None):
    contract = inputs.get("nonflash_kv_view_contract")
    if contract is None:
        return scenario
    if gguf is not None:
        sliding = {key: value for key, value in gguf.metadata.items()
                   if "sliding_window" in key and value not in (None, 0, False)}
        contract = {**contract, "model_cache_scope": {
            "ordinary_retained_prefix": gguf.architecture in {"llama", "qwen2", "qwen35"} and not sliding,
            "gguf_sha256": gguf.sha256, "sliding_window_metadata": sliding}}
    return replace(scenario, workload=replace(scenario.workload,
        metadata={**scenario.workload.metadata, "llama_cpp_nonflash_kv_view": contract}))

def verified_gpu_invocation_contract(path, rows, data_root, *, enable_mmq_source_costs=False,
                                     runtime_binding=None, runtime_source_contract_path=None):
    """Validate a source template, then derive every cell from captured controls."""
    from heterollm_sim.llama_gpu_invocations import SCHEMA, derive_llama_gpu_invocation_contract, apply_llama_gpu_invocation_contract
    if type(enable_mmq_source_costs) is not bool or not {"enabled", "enable_mmq_source_costs"}.issubset(inspect.signature(apply_llama_gpu_invocation_contract).parameters):
        raise ValueError("GPU invocation treatment requires separate explicit geometry and source-cost switches")
    saved, contract_ref = grid.read_document(path)
    if saved.get("schema") != SCHEMA:
        raise ValueError("GPU invocation source contract schema mismatch")
    if runtime_binding is None:
        if runtime_source_contract_path is None:
            candidates = {parent / "round_004/runtime_source_binding_structural_audit.json" for parent in Path(path).resolve().parents}
            present = sorted({candidate.resolve() for candidate in candidates if candidate.is_file()}, key=str)
            if len(present) > 1:
                raise ValueError("GPU invocation runtime source binding is ambiguous")
            runtime_source_contract_path = present[0] if present else Path(path).resolve().parent.parent / "round_004/runtime_source_binding_structural_audit.json"
        runtime_binding = verified_host_offload_source_contract(runtime_source_contract_path, rows, data_root)
    linkage = verify_gpu_invocation_source_links(runtime_binding, data_root)
    template_environment = {}
    for key, fact in saved.get("kernel_environment", {}).items():
        if not isinstance(fact, Mapping) or type(fact.get("captured")) is not bool:
            raise ValueError("GPU invocation template has malformed kernel environment evidence")
        if fact["captured"]:
            if fact.get("value") is not None and not isinstance(fact["value"], str):
                raise ValueError("GPU invocation template has malformed captured environment value")
            template_environment[key] = fact.get("value")
    mmq = saved.get("mmq_device_evidence", {})
    mmq_argument = mmq if isinstance(mmq, Mapping) and mmq.get("available") is True else None
    canonical = derive_llama_gpu_invocation_contract(runtime_binding["contract"],
        captured_kernel_environment=template_environment, cuda_compute_capability=saved.get("cuda_compute_capability"),
        mmq_device_evidence=mmq_argument)
    if json.loads(json.dumps(canonical)) != saved:
        raise ValueError("GPU invocation contract differs from re-derived source/build/device facts")
    refs = {ref["path"]: ref for ref in [contract_ref, *runtime_binding["evidence_refs"], *linkage["evidence_refs"], *canonical["source_refs"]]}
    def derivation_key(environment, cc):
        return grid.stable_hash({"environment": environment, "cuda_compute_capability": cc,
            "runtime_binding_sha256": runtime_binding["contract"].get("content_sha256"), "mmq_device_evidence": mmq_argument})
    # Identical static facts share one source derivation; every raw capture is
    # still checked and all source references are postverified before freezing.
    contract_cache = {derivation_key(template_environment, canonical["cuda_compute_capability"]): canonical}
    cells = {}
    for row in rows:
        runtime = runtime_binding["cells"][row["cell_id"]]
        evidence = runtime.get("device_evidence") or {}
        device_cc = evidence.get("compute_capability")
        match = re.fullmatch(r"(\d+)\.(\d+)", str(device_cc or ""))
        if runtime.get("cuda_backend_available") is not True or match is None:
            cells[row["cell_id"]] = {"status": "uncovered", "contract": None,
                "uncovered_reasons": ["captured_cuda_backend_or_architecture_unavailable"],
                "native_dispatch_proven": False, "mmq_source_costs_requested": enable_mmq_source_costs}
            continue
        cc = 100 * int(match[1]) + 10 * int(match[2])
        if cc != canonical["cuda_compute_capability"]:
            raise ValueError("GPU invocation contract architecture differs from captured native device")
        if mmq_argument and mmq_argument.get("gpu_uuid") != evidence.get("gpu_uuid"):
            raise ValueError("GPU invocation MMQ property probe differs from selected native GPU UUID")
        record_facts = []
        for native_ref in runtime["native_record_refs"]:
            raw, raw_ref = grid.read_document(grid.resolve_data(native_ref["path"], data_root), native_ref["sha256"])
            env, env_evidence = captured_kernel_environment(raw, row.get("config", {}).get("environment"))
            record_facts.append((env, env_evidence, raw_ref))
            refs[raw_ref["path"]] = raw_ref
        if not record_facts or any(value[:2] != record_facts[0][:2] for value in record_facts[1:]):
            raise ValueError("GPU invocation kernel environment differs across selected cell captures")
        env, env_evidence, _ = record_facts[0]
        key = derivation_key(env, cc)
        if key not in contract_cache:
            contract_cache[key] = derive_llama_gpu_invocation_contract(runtime_binding["contract"],
                captured_kernel_environment=env, cuda_compute_capability=cc, mmq_device_evidence=mmq_argument)
        contract = contract_cache[key]
        for ref in contract["source_refs"]:
            refs[ref["path"]] = ref
        cells[row["cell_id"]] = {"status": contract["status"], "contract": contract,
            "kernel_environment": env_evidence, "cuda_compute_capability": cc,
            "device_evidence": evidence, "native_record_refs": [value[2] for value in record_facts],
            "mmq_source_costs_requested": enable_mmq_source_costs, "native_dispatch_proven": False,
            "uncovered_reasons": contract.get("uncovered_reasons", []),
            "conditional_reasons": contract.get("conditional_reasons", []), "unpriced_terms": contract.get("unpriced_terms", [])}
    verify_refs(list(refs.values()))
    return {"contract": saved, "contract_ref": contract_ref, "evidence_refs": list(refs.values()), "cells": cells,
        "runtime_source_binding_ref": runtime_binding["contract_ref"], "source_linkage": linkage,
        "mmq_source_costs_requested": enable_mmq_source_costs,
        "source_derivation_variants": len(contract_cache),
        "conditional_cell_count": sum(cell["status"] == "conditional" for cell in cells.values()),
        "uncovered_cell_count": sum(cell["status"] == "uncovered" for cell in cells.values()),
        "template_environment_is_runtime_evidence": False, "per_cell_environment_source": "captured execution_environment only",
        "native_dispatch_proven": False, "native_latency_used": False, "today_environment_read": False,
        "host_offload_treatment_enabled_by_this_validation": False}


def apply_gpu_invocation_static_contract(scenario, inputs):
    proof = inputs.get("gpu_invocation_evidence")
    if proof is None:
        return scenario
    if not isinstance(proof, Mapping):
        raise ValueError("GPU invocation evidence must be a frozen mapping")
    if proof.get("contract") is None:
        return scenario
    contract = inputs.get("gpu_invocation_contract")
    flag = inputs.get("gpu_mmq_source_costs", False)
    if not isinstance(proof, Mapping) or contract != proof.get("contract") or type(flag) is not bool or flag is not proof.get("mmq_source_costs_requested"):
        raise ValueError("GPU invocation static contract or source-cost switch differs from frozen evidence")
    from heterollm_sim.llama_gpu_invocations import apply_llama_gpu_invocation_contract
    conversion_flag = inputs.get("gpu_conversion_cta_costs", False)
    if type(conversion_flag) is not bool or conversion_flag is not proof.get("conversion_cta_costs_requested", False):
        raise ValueError("conversion CTA source switch differs from frozen evidence")
    return apply_llama_gpu_invocation_contract(scenario, contract, enabled=True, enable_mmq_source_costs=flag,
                                               enable_conversion_cta_costs=conversion_flag)


MMVQ_HARDWARE_DOCUMENT_URL = "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"
MMVQ_DOCUMENT_MAX_BYTES = 16 * 1024 * 1024


def checked_mmvq_document_bytes(payload):
    """Verify actual immutable PDF bytes, never a supplied digest declaration."""
    import hashlib
    from heterollm_sim.mmvq_issue_bound import HARDWARE_DOCUMENT_SHA256
    if not payload.startswith(b"%PDF-") or len(payload) > MMVQ_DOCUMENT_MAX_BYTES:
        raise ValueError("MMVQ hardware document is not a bounded PDF")
    if hashlib.sha256(payload).hexdigest() != HARDWARE_DOCUMENT_SHA256:
        raise ValueError("MMVQ hardware document content SHA256 mismatch")
    return HARDWARE_DOCUMENT_SHA256


def freeze_mmvq_hardware_document(output, document_path=None):
    """Capture the actual official PDF; unavailable evidence disables treatment."""
    import urllib.request
    import urllib.error
    try:
        if document_path is None:
            with urllib.request.urlopen(MMVQ_HARDWARE_DOCUMENT_URL, timeout=30) as response:
                payload = response.read(MMVQ_DOCUMENT_MAX_BYTES + 1)
        else:
            with Path(document_path).open("rb") as source:
                payload = source.read(MMVQ_DOCUMENT_MAX_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__, "ref": None,
            "source_url": MMVQ_HARDWARE_DOCUMENT_URL, "content_bytes_verified": False}
    digest = checked_mmvq_document_bytes(payload)
    target = Path(output) / "evidence/nvidia-rtx-blackwell-gpu-architecture.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as destination:
        destination.write(payload)
    ref = grid.file_ref(target)
    if ref["sha256"] != digest:
        raise ValueError("MMVQ hardware document changed while freezing")
    return {"status": "verified", "ref": ref, "source_url": MMVQ_HARDWARE_DOCUMENT_URL,
        "content_bytes_verified": True, "size_bytes": len(payload),
        "revision": "RTX Blackwell v1.1", "figure": "Figure 5, printed page 11",
        "interpretation": "four 32-thread dispatch partitions; not measured DP4A throughput"}


def verify_mmvq_hardware_document(document):
    if not isinstance(document, Mapping):
        raise ValueError("MMVQ hardware document evidence must be a mapping")
    if document.get("status") == "unavailable":
        if document.get("ref") is not None or document.get("content_bytes_verified") is not False:
            raise ValueError("MMVQ unavailable hardware document claims verification")
        return
    if (document.get("status") != "verified" or document.get("content_bytes_verified") is not True
            or document.get("source_url") != MMVQ_HARDWARE_DOCUMENT_URL):
        raise ValueError("MMVQ hardware document has no verified content")
    ref = document["ref"]
    with Path(ref["path"]).open("rb") as source:
        payload = source.read(MMVQ_DOCUMENT_MAX_BYTES + 1)
    if checked_mmvq_document_bytes(payload) != ref["sha256"] or len(payload) != document.get("size_bytes"):
        raise ValueError("MMVQ frozen hardware document reference mismatch")


def derive_mmvq_issue_source_binding(runtime_binding_ref, data_root):
    """Bind the complete MMVQ source snapshot to the recorded module build chain.

    The inherited object has a verified link identity. Headers were captured by
    the later annotation receipt; their bytes at the original compile remain an
    explicit condition, not a claim that a dependency/preprocessor trace exists.
    """
    from tools import llama_runtime_source_binding as source_api
    from heterollm_sim.mmvq_work import SOURCE_SHA256
    grid.read_document(runtime_binding_ref["path"], runtime_binding_ref["sha256"])
    runtime = verified_host_offload_source_contract(runtime_binding_ref["path"], [], data_root)
    if runtime["contract_ref"]["sha256"] != runtime_binding_ref["sha256"]:
        raise ValueError("MMVQ runtime source binding changed during validation")
    linkage = verify_gpu_invocation_source_links(runtime, data_root)
    binding = runtime["contract"]
    ev = source_api._Evidence(data_root)
    audit = ev.document(runtime["runtime_build_audit_ref"])
    stages = {stage["stage"]: stage for stage in audit["provenance_chain"]}
    base_stage = stages["base_configuration_and_preprocessor_guards"]
    annotation = ev.document(stages["native_to_annotation"]["annotation_build_receipt_ref"])
    base_receipt = ev.document(runtime["base_build_receipt_ref"])
    header_ref = base_stage["header_snapshot_ref"]
    source_api._equal_digests(header_ref["sha256"], annotation["header_snapshot_sha256"])
    headers = ev.document(header_ref)
    root = Path(binding["source_paths"]["scheduler"]).parents[2]
    module = binding["runtime_modules"]["ggml-cuda.dll"]
    compilation = linkage["source_compilation"]["mmvq"]
    if compilation["module"] != module:
        raise ValueError("MMVQ object linkage does not reach the selected CUDA module")
    refs = {ref["path"]: ref for ref in [runtime["contract_ref"], *runtime["evidence_refs"], *linkage["evidence_refs"]]}
    result = {"status": "uncovered", "runtime_source_binding_ref": runtime["contract_ref"],
        "data_root": str(Path(data_root).resolve()), "runtime_binding_sha256": binding["content_sha256"],
        "runtime_module_ref": module, "source_compilation": compilation,
        "header_snapshot_ref": header_ref, "source_refs": [],
        "original_compile_header_bytes_proven": False, "native_instruction_mapping_proven": False,
        "native_latency_used": False,
        "conditions": ["captured_header_snapshot_matches_headers_consumed_by_original_mmvq_compilation"],
        "header_scope": "digest_bound_annotation_header_snapshot; no original compiler dependency/preprocessor trace"}
    source_refs, texts = [], {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / "ggml/src" / relative
        history = base_receipt["source_sha256_before"] if path.suffix == ".cu" else headers.get("files", {})
        matches = [digest for filename, digest in history.items()
            if source_api._identity(filename) == source_api._identity(path)]
        if len(matches) != 1:
            refs.update({ref["path"]: ref for ref in ev.refs.values()})
            return {**result, "uncovered_reasons": ["historical_mmvq_source_snapshot_missing:" + relative],
                "evidence_refs": list(refs.values())}
        source_api._equal_digests(matches[0], expected)
        if path.suffix == ".cu":
            source_api._equal_digests(expected, source_api._digest_at(base_receipt["source_sha256_after"], path),
                compilation["source_sha256"])
            if source_api._identity(path) != source_api._identity(compilation["source"]):
                raise ValueError("MMVQ compilation input differs from locked source path")
        texts[relative] = ev.text({"path": str(path), "sha256": expected})
        source_refs.append(ev.refs[source_api._identity(path)])
    includes = (("ggml-cuda/mmvq.cu", '"mmvq.cuh"'),
        ("ggml-cuda/mmvq.cu", '"vecdotq.cuh"'), ("ggml-cuda/mmvq.cuh", '"common.cuh"'),
        ("ggml-cuda/vecdotq.cuh", '"common.cuh"'), ("ggml-cuda/common.cuh", '"ggml-common.h"'))
    if any("#include " + target not in texts.get(relative, "") for relative, target in includes):
        raise ValueError("MMVQ captured header include chain differs from locked source")
    commands = ev.document(base_stage["compile_commands_ref"])
    build = Path(annotation["base"]).resolve()
    entry = source_api._compile_entry(commands, root / "ggml/src/ggml-cuda/mmvq.cu", build)
    argv = source_api._tokens(entry.get("arguments", entry.get("command")))
    include_dirs = [argument[2:] for argument in argv if argument.startswith("-I") and len(argument) > 2]
    if not any(source_api._identity(path) == source_api._identity(root / "ggml/src") for path in include_dirs):
        raise ValueError("MMVQ recorded include search does not reach the captured ggml-common header")
    refs.update({ref["path"]: ref for ref in ev.refs.values()})
    verify_refs(list(refs.values()))
    return {**result, "status": "conditional", "uncovered_reasons": [], "source_refs": source_refs,
        "evidence_refs": list(refs.values()), "snapshot_identity_verified": True}


def merged_mmvq_source_refs(primary, supplement):
    """Deduplicate identical paths but never conceal conflicting source hashes."""
    refs = {}
    for ref in [*primary, *supplement]:
        key = str(Path(ref["path"]).resolve()).casefold()
        if key in refs and refs[key]["sha256"] != ref["sha256"]:
            raise ValueError("MMVQ conflicting source identity across invocation and supplemental binding")
        refs[key] = ref
    return list(refs.values())


def mmvq_static_cell_evidence(gpu_binding, hardware, native_refs, document, *, checked_sources=None, source_binding=None):
    """Recheck only static source/device facts and historical module identities."""
    from heterollm_sim.mmvq_issue_bound import source_issue_contract, CLOCK_CONDITION
    from heterollm_sim.mmvq_work import SOURCE_SHA256 as MMVQ_SOURCES
    from heterollm_sim.conversion_work import SOURCE_SHA256 as CONVERSION_SOURCES
    if (not isinstance(gpu_binding, Mapping) or gpu_binding.get("mmq_source_costs_requested") is not True
            or gpu_binding.get("conversion_cta_costs_requested") is not True):
        raise ValueError("MMVQ issue bound requires frozen GPU invocation, MMQ and conversion source costs")
    evidence = {"requested": True, "status": "uncovered", "contract": None,
        "gpu_invocation_sha256": grid.stable_hash(gpu_binding), "hardware_document": document,
        "native_instruction_mapping_proven": False, "native_latency_used": False,
        "calibration_applied": False, "formal_prediction_eligible": False,
        "clock_condition": CLOCK_CONDITION,
        "unpriced_work": ["unpack", "constant_dot_correction", "float_scale", "warp_reduction",
            "barrier_latency", "register_pressure", "spill", "native_instruction_latency"],
        "hbm_accounting": "unchanged_legacy_unverified", "evidence_refs": []}
    if document["status"] != "verified":
        return {**evidence, "uncovered_reasons": ["hardware_document_content_unavailable"]}
    if source_binding is not None:
        evidence["mmvq_source_binding"] = source_binding
        evidence["evidence_refs"] = source_binding["evidence_refs"]
        if source_binding["status"] != "conditional":
            return {**evidence, "uncovered_reasons": source_binding["uncovered_reasons"]}
    contract = gpu_binding.get("contract")
    if contract is None:
        return {**evidence, "uncovered_reasons": ["gpu_invocation_contract_unavailable"]}
    device = contract.get("mmq_device_evidence", {})
    if device.get("available") is not True:
        return {**evidence, "uncovered_reasons": ["static_device_properties_unavailable"]}
    refs = [document["ref"]]
    probe_ref, hardware_ref = device["source_ref"], device["selected_hardware_ref"]
    probe, actual_probe_ref = grid.read_document(probe_ref["path"], probe_ref["sha256"])
    selected, actual_hardware_ref = grid.read_document(hardware_ref["path"], hardware_ref["sha256"])
    refs.extend([actual_probe_ref, actual_hardware_ref])
    attrs = probe.get("attributes", {})
    sm = attrs.get("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT")
    major, minor = attrs.get("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"), attrs.get("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR")
    warp = attrs.get("CU_DEVICE_ATTRIBUTE_WARP_SIZE")
    uuid = probe.get("gpu_uuid")
    if (probe.get("schema") != "cuda-driver-static-device-properties/v1"
            or probe.get("native_configuration_modified") is not False
            or any(type(value) is not int for value in (sm, major, minor, warp)) or sm < 1
            or (major, minor, warp) != (12, 0, 32)
            or device.get("sm_count") != sm or device.get("warp_size") != warp
            or device.get("cuda_compute_capability") != 1200
            or contract.get("cuda_compute_capability") != 1200
            or gpu_binding.get("cuda_compute_capability") != 1200
            or not uuid or any(gpu.get("uuid") != uuid or gpu.get("compute_capability") != "12.0"
                for gpu in (hardware.get("gpu", {}), selected.get("gpu", {})))
            or device.get("gpu_uuid") != uuid
            or gpu_binding.get("device_evidence", {}).get("gpu_uuid") != uuid):
        raise ValueError("MMVQ static device SM/UUID/architecture binding mismatch")
    module = contract.get("runtime_modules", {}).get("ggml-cuda.dll")
    actual = [ref for ref in native_refs if Path(ref["path"]).name.lower() == "ggml-cuda.dll"]
    if (not isinstance(module, Mapping) or len(actual) != 1
            or module.get("sha256") != actual[0].get("sha256")
            or Path(module["path"]).resolve() != Path(actual[0]["path"]).resolve()):
        raise ValueError("MMVQ runtime CUDA module differs from selected runtime identity")
    source_refs = contract.get("source_refs", [])
    if source_binding is not None:
        if (source_binding["runtime_binding_sha256"] != contract.get("runtime_binding_sha256")
                or source_binding["runtime_module_ref"] != module):
            raise ValueError("MMVQ supplemental source history differs from selected GPU invocation runtime")
        source_refs = merged_mmvq_source_refs(source_refs, source_binding["source_refs"])
        refs.extend(source_binding["evidence_refs"])
    cache = set() if checked_sources is None else checked_sources
    for relative, expected in {**CONVERSION_SOURCES, **MMVQ_SOURCES}.items():
        matches = [ref for ref in source_refs if str(ref["path"]).replace(chr(92), "/").endswith("/" + relative)]
        if len(matches) != 1 or matches[0].get("sha256") != expected:
            raise ValueError("MMVQ immutable source proof missing or changed: " + relative)
        ref = matches[0]
        identity = (ref["path"], ref["sha256"])
        if identity not in cache:
            verify_refs([ref])
            cache.add(identity)
        refs.append(ref)
    return {**evidence, "status": "conditional", "uncovered_reasons": [],
        "contract": source_issue_contract(runtime_binary_sha256=module["sha256"], sm_count=sm),
        "gpu_uuid": uuid, "runtime_module_ref": dict(actual[0]),
        "device_evidence": device, "evidence_refs": refs,
        "conditions": ["weight_dependent_source_dp4a_maps_to_native_integer_warp_issue",
            CLOCK_CONDITION, "conditional_dot_issue_lower_bound_only",
            *((source_binding or {}).get("conditions", []))]}


def verified_mmvq_issue_binding(rows, gpu_invocation, output, *, document_path=None, data_root=None):
    if (not isinstance(gpu_invocation, Mapping) or gpu_invocation.get("mmq_source_costs_requested") is not True
            or gpu_invocation.get("conversion_cta_costs_requested") is not True):
        raise ValueError("MMVQ issue bound requires GPU invocation, MMQ and conversion source costs")
    document = freeze_mmvq_hardware_document(output, document_path)
    verify_mmvq_hardware_document(document)
    cells, cache, refs = {}, set(), {}
    source_binding = None
    if document["status"] == "verified" and gpu_invocation.get("runtime_source_binding_ref") is not None:
        source_binding = derive_mmvq_issue_source_binding(gpu_invocation["runtime_source_binding_ref"], data_root or ROOT)
    if document.get("ref") is not None:
        refs[document["ref"]["path"]] = document["ref"]
    for row in rows:
        proof = mmvq_static_cell_evidence(gpu_invocation["cells"][row["cell_id"]],
            row["static_hardware"]["frozen_hardware"], row["native_runtime_refs"], document, checked_sources=cache, source_binding=source_binding)
        cells[row["cell_id"]] = proof
        refs.update({ref["path"]: ref for ref in proof["evidence_refs"]})
    return {"requested": True, "hardware_document": document, "cells": cells,
        **({"mmvq_source_binding": source_binding} if source_binding is not None else {}),
        "evidence_refs": list(refs.values()), "native_latency_used": False,
        "conditional_cell_count": sum(cell["status"] == "conditional" for cell in cells.values()),
        "uncovered_cell_count": sum(cell["status"] == "uncovered" for cell in cells.values())}


def apply_mmvq_issue_static_contract(scenario, inputs):
    flag = inputs.get("mmvq_vector_issue_bound", False)
    proof = inputs.get("mmvq_issue_evidence")
    if flag is False and proof is None and inputs.get("mmvq_issue_contract") is None:
        return scenario
    if type(flag) is not bool or flag is not True or not isinstance(proof, Mapping) or proof.get("requested") is not True:
        raise ValueError("MMVQ issue switch differs from frozen evidence")
    if inputs.get("gpu_mmq_source_costs") is not True or inputs.get("gpu_conversion_cta_costs") is not True:
        raise ValueError("MMVQ issue bound requires MMQ and conversion source costs")
    gpu_binding = inputs.get("gpu_invocation_evidence")
    if not isinstance(gpu_binding, Mapping) or inputs.get("gpu_invocation_contract") != gpu_binding.get("contract"):
        raise ValueError("MMVQ GPU invocation differs from frozen evidence")
    verify_mmvq_hardware_document(proof.get("hardware_document"))
    source_binding = proof.get("mmvq_source_binding")
    if source_binding is not None:
        rederived = derive_mmvq_issue_source_binding(source_binding["runtime_source_binding_ref"], source_binding["data_root"])
        if rederived != source_binding:
            raise ValueError("MMVQ supplemental source binding differs from recorded build/header history")
    expected = mmvq_static_cell_evidence(gpu_binding, inputs["hardware_snapshot"],
        inputs.get("runtime_module_refs", []), proof["hardware_document"], source_binding=source_binding)
    if proof != expected or inputs.get("mmvq_issue_contract") != proof.get("contract"):
        raise ValueError("MMVQ issue contract differs from re-derived frozen evidence")
    qualification = {**proof, "declared_clock": gpu_clock(inputs)}
    flags = {**scenario.workload.metadata, "llama_cpp_mmvq_vector_issue_qualification": qualification}
    if proof["status"] != "conditional":
        return replace(scenario, workload=replace(scenario.workload, metadata=flags))
    from heterollm_sim.mmvq_issue_bound import MMVQIssueContract
    from heterollm_sim.conversion_work import ConversionSourceContract
    contract = MMVQIssueContract.from_mapping(proof["contract"])
    gpus = [component for component in scenario.hardware.components if str(component.kind).lower() == "gpu"]
    if len(gpus) != 1:
        raise ValueError("MMVQ bound requires one selected GPU component")
    gpu = gpus[0]
    profile = scenario.resolve_component_profile(gpu)
    invocation_audit = scenario.workload.metadata.get("llama_cpp_gpu_native_invocations", {})
    if isinstance(invocation_audit, Mapping) and invocation_audit.get("applied") is False:
        flags["llama_cpp_mmvq_vector_issue_qualification"] = {**qualification, "status": "uncovered",
            "uncovered_reasons": ["scenario_gpu_invocation_not_qualified", *invocation_audit.get("reasons", [])]}
        return replace(scenario, workload=replace(scenario.workload, metadata=flags))
    conversion = gpu.metadata.get("llama_cpp_conversion_source_contract")
    if not isinstance(conversion, Mapping):
        raise ValueError("MMVQ issue bound requires installed conversion source contract")
    conversion = ConversionSourceContract(**conversion)
    if (conversion.runtime_binary_sha256 != contract.runtime_binary_sha256
            or profile.tensor_core.sm_count != contract.sm_count
            or gpu.metadata.get("cuda_compute_capability") != contract.compute_capability
            or scenario.workload.metadata.get("llama_cpp_mmq_source_work") is not True
            or scenario.workload.metadata.get("llama_cpp_conversion_cta_costs") is not True):
        raise ValueError("MMVQ runtime/profile/installed source treatment mismatch")
    if (gpu.metadata.get("llama_cpp_mmvq_prmt_partial_contract") or {}).get("enabled") is True:
        raise ValueError("MMVQ issue bound conflicts with PRMT partial contract")
    replacement = replace(gpu, metadata={**gpu.metadata, "llama_cpp_mmvq_vector_issue_contract": proof["contract"]})
    if source_binding is not None:
        flags["llama_cpp_gpu_native_invocations"] = {**invocation_audit,
            "source_refs": merged_mmvq_source_refs(invocation_audit.get("source_refs", []), source_binding["source_refs"]),
            "mmvq_source_binding": source_binding}
    flags["llama_cpp_mmvq_vector_issue_bound"] = True
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        replacement if component.component_id == gpu.component_id else component for component in scenario.hardware.components)),
        workload=replace(scenario.workload, metadata=flags))


IQ_PANEL_SOURCE_SCHEMA = "llama.cpp.cpu.iq-panel-source-contract/v1"
IQ_PANEL_VARIABLE = "GGML_NO_IQ_PANEL"


def derive_iq_panel_source_contract(historical_audit_path, data_root):
    """Read native source/build/history facts; never read today's environment."""
    from heterollm_sim.runtime_adapters import _source_function
    audit, audit_ref = grid.read_document(historical_audit_path)
    if audit.get("schema") != "cpu-iq-panel-historical-dispatch-audit/v1" or audit.get("variable") != IQ_PANEL_VARIABLE:
        raise ValueError("CPU IQ panel historical audit schema/variable mismatch")
    if audit.get("historical_state") != "unknown" or audit.get("today_environment_read") is not False or audit.get("new_native_run") is not False:
        raise ValueError("this contract requires explicitly unknown, unreconstructed historical environment")
    frozen_ref, raw_ref = audit["native_source_freeze_ref"], audit["sampled_raw_ref"]
    frozen, verified_freeze_ref = grid.read_document(grid.resolve_data(frozen_ref["path"], data_root), frozen_ref["sha256"])
    raw, verified_raw_ref = grid.read_document(grid.resolve_data(raw_ref["path"], data_root), raw_ref["sha256"])
    environments = frozen.get("environments", {})
    if not environments or any(IQ_PANEL_VARIABLE in values for values in environments.values()) or IQ_PANEL_VARIABLE in raw.get("execution_environment", {}):
        raise ValueError("historical variable is recorded; cannot label it unknown")
    capture_refs = [ref for ref in frozen["source_refs"] if Path(ref["path"]).name == "native_repeatability_experiment.py"]
    if len(capture_refs) != 1:
        raise ValueError("one frozen environment-capture source is required")
    verify_refs(capture_refs)
    capture = Path(capture_refs[0]["path"]).read_text(encoding="utf-8")
    syntax = ast.parse(capture)
    keys = [ast.literal_eval(node.value) for node in syntax.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ENV_KEYS" for target in node.targets)]
    functions = [node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "environment_for"]
    inherited = bool(functions) and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"
        and any(isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "os" and arg.attr == "environ" for arg in node.args)
        for node in ast.walk(functions[0]))
    if len(keys) != 1 or IQ_PANEL_VARIABLE in keys[0] or not inherited:
        raise ValueError("frozen capture source does not establish unknown inherited environment")
    facts = audit["native_iqp_compiled_avx2"]
    iqp_ref = grid.file_ref(grid.resolve_data(facts["iqp_source_ref"]["path"], data_root))
    if iqp_ref["sha256"] != facts["iqp_source_ref"]["sha256"]:
        raise ValueError("IQ panel source SHA256 mismatch")
    cpu_ref = grid.file_ref(grid.resolve_data(facts["cpu_backend_ref_as_frozen"]["path"], data_root))
    if cpu_ref["sha256"] != facts["cpu_backend_ref_as_frozen"]["sha256"]:
        raise ValueError("IQ panel CPU backend SHA256 differs from frozen native")
    build, build_ref = grid.read_document(grid.resolve_data(facts["build_receipt_ref"]["path"], data_root), facts["build_receipt_ref"]["sha256"])
    manifest, manifest_ref = grid.read_document(Path(build_ref["path"]).parent / "source_manifest.json", build.get("source_manifest_sha256"))
    units = [unit for unit in manifest.get("compile_units", []) if Path(unit.get("file", "")).resolve() == Path(iqp_ref["path"])]
    headers, headers_ref = grid.read_document(Path(build_ref["path"]).parent / "header_snapshot.json", build.get("header_snapshot_sha256"))
    common_headers = [(filename, sha) for filename, sha in headers.get("files", {}).items() if Path(filename).name == "ggml-common.h"]
    if len(common_headers) != 1:
        raise ValueError("IQ panel quantization block header is not bound")
    quant_ref = grid.file_ref(grid.resolve_data(common_headers[0][0], data_root))
    if quant_ref["sha256"] != common_headers[0][1] or not re.search(r"#define\s+QK_K\s+256\b", Path(quant_ref["path"]).read_text(encoding="utf-8")):
        raise ValueError("IQ panel quantization block geometry differs from compiled source")
    dispatcher_ref = grid.file_ref(Path(iqp_ref["path"]).parent / "ggml-cpu.c")
    dispatcher_units = [unit for unit in manifest.get("compile_units", []) if Path(unit.get("file", "")).resolve() == Path(dispatcher_ref["path"])]
    if len(dispatcher_units) != 1 or dispatcher_units[0].get("sha256") != dispatcher_ref["sha256"]:
        raise ValueError("IQ panel CPU matmul dispatch source is not compile-bound")
    dispatcher_source = Path(dispatcher_ref["path"]).read_text(encoding="utf-8")
    if "ggml_cpu_iqp_supports_mul_mat(dst) && !params->use_ref" not in dispatcher_source:
        raise ValueError("IQ panel reference-kernel exclusion is not source-bound")
    compiled = [step for step in build.get("steps", []) if step.get("label") == "compile iqp.cpp"]
    linked = [step for step in build.get("steps", []) if step.get("label") == "link ggml-cpu.dll"]
    output_shas = [sha for path, sha in build.get("output_sha256", {}).items() if Path(path).name.lower() == "ggml-cpu.dll"]
    if build.get("status") != "complete" or build.get("baseline_postverified") is not True or len(units) != 1 or units[0].get("sha256") != iqp_ref["sha256"]:
        raise ValueError("IQ panel source/CPU build receipt binding failed")
    if len(compiled) != 1 or len(linked) != 1 or compiled[0].get("returncode") != 0 or linked[0].get("returncode") != 0 or output_shas != [cpu_ref["sha256"]]:
        raise ValueError("IQ panel compile/link/output identity not established")
    argv = compiled[0].get("argv", [])
    if "-DGGML_AVX2" not in argv or "/arch:AVX2" not in argv or str(Path(iqp_ref["path"])) not in argv:
        raise ValueError("IQ panel compile command does not prove AVX2 for this source")
    source = Path(iqp_ref["path"]).read_text(encoding="utf-8")
    common = _source_function(source, "static bool iqp_supported_common(const struct ggml_tensor * dst)")
    support = _source_function(source, "bool ggml_cpu_iqp_supports_mul_mat(const struct ggml_tensor * dst)")
    execution = _source_function(source, "void ggml_compute_forward_mul_mat_iqp(const struct ggml_compute_params * params, struct ggml_tensor * dst)")
    common_fragments = ('getenv("GGML_NO_IQ_PANEL") != nullptr', 'src1->type != GGML_TYPE_F32',
        'dst->type != GGML_TYPE_F32', 'dst->nb[0] != sizeof(float)', '!ggml_is_contiguous(src0)',
        'src0->ne[3] != 1', 'src1->ne[3] != 1', 'src0->ne[0] % QK_K', 'src0->ne[1] % IQP_NB_ROWS',
        'ggml_get_type_traits_cpu(src0->type)->vec_dot_type != GGML_TYPE_Q8_K', '!ggml_cpu_has_avx2()')
    if any(fragment not in common for fragment in common_fragments) or 'src0->ne[2] != 1' not in support or 'src1->ne[1] < GGML_IQP_MIN_BATCH' not in support:
        raise ValueError("IQ panel eligibility source is not recognized")
    if not re.search(r"#define\s+GGML_IQP_MIN_BATCH\s+8\b", source) or not re.search(r"#define\s+IQP_NB_ROWS\s+8\b", source):
        raise ValueError("IQ panel source thresholds changed")
    decode, rows = execution.find('iqp_decode_panel_8('), execution.find('for (int64_t i12 = 0; i12 < ne12; i12++)')
    if decode < 0 or rows <= decode or 'const int64_t ngroups = ne01 / IQP_NB_ROWS' not in execution:
        raise ValueError("IQ panel source does not establish decode-before-row-reuse")
    evidence = [audit_ref, verified_freeze_ref, verified_raw_ref, *capture_refs, iqp_ref, dispatcher_ref, quant_ref, cpu_ref, build_ref, manifest_ref, headers_ref]
    return {"schema": IQ_PANEL_SOURCE_SCHEMA, "status": "source_derived",
        "historical_audit_ref": audit_ref, "compiled_avx2": True,
        "source_sha256": iqp_ref["sha256"], "cpu_backend_sha256": cpu_ref["sha256"],
        "no_iq_panel_environment_state": "unknown", "native_dispatch_proven": False,
        "today_environment_read": False, "native_latency_used": False,
        "candidate_weight_formats": ["IQ3_S", "IQ4_XS"],
        "required_native_predicates": {"activation_dtype": "F32", "output_dtype": "F32",
            "weight_layout": "ordinary_contiguous_2d", "activation_ne3": 1, "reference_kernel": False,
            "minimum_m": 8, "k_alignment": 256, "n_alignment": 8},
        "operator_fact_scope": "These are branch requirements, not per-operator facts; planner must independently qualify each physical weight projection and native graph dtype/layout",
        "source_rule": "decode each N/8 weight panel once before reuse across M rows and ne12; preserve multiply-add work and original byte/throughput inputs",
        "scope": "source-qualified CPU IQ weight-panel decode reuse only; no dynamic attention, experts, scatter or dtype-by-bitwidth inference",
        "evidence_refs": evidence,
        "limits": ["Historical GGML_NO_IQ_PANEL was not captured and cannot be treated as unset",
            "This is not a complete IQP kernel cost model; conversion, scratch traffic and integer-dot throughput remain separately unresolved"]}


def verified_iq_panel_source_contract(path, rows, data_root, *, assume_default_unset=False):
    """Validate the source contract and retain an explicit unknown-state ablation."""
    if type(assume_default_unset) is not bool:
        raise ValueError("IQ panel default-unset assumption must be an explicit boolean")
    contract, contract_ref = grid.read_document(path)
    payload = dict(contract)
    claimed = payload.pop("content_sha256", None)
    if claimed is not None and grid.stable_hash(payload) != claimed:
        raise ValueError("IQ panel contract content SHA256 mismatch")
    if payload.get("schema") != IQ_PANEL_SOURCE_SCHEMA:
        raise ValueError("IQ panel source contract schema mismatch")
    audit_ref = payload["historical_audit_ref"]
    grid.read_document(audit_ref["path"], audit_ref["sha256"])
    derived = derive_iq_panel_source_contract(audit_ref["path"], data_root)
    if derived != payload:
        raise ValueError("IQ panel contract differs from re-derived source/build/history facts")
    for row in rows:
        cpus = [ref for ref in row["native_runtime_refs"] if Path(ref["path"]).name.lower() == "ggml-cpu.dll"]
        if len(cpus) != 1 or cpus[0]["sha256"] != payload["cpu_backend_sha256"]:
            raise ValueError("IQ panel contract CPU backend differs from selected native runtime")
    dispatch = {"enabled": True, "compiled_avx2": True, "no_iq_panel_environment_state": "unknown",
        "assume_default_unset": assume_default_unset, "source_sha256": payload["source_sha256"],
        "cpu_backend_sha256": payload["cpu_backend_sha256"],
        "source_refs": [contract_ref["path"], *[ref["path"] for ref in payload["evidence_refs"]]],
        "native_dispatch_proven": False,
        "evaluation_scope": "conditional_default_unset_ablation" if assume_default_unset else "historical_unknown_without_dispatch_assumption"}
    return {"contract": payload, "contract_ref": contract_ref, "dispatch": dispatch,
        "evidence_refs": [contract_ref, *payload["evidence_refs"]], "native_dispatch_proven": False,
        "evaluation_scope": dispatch["evaluation_scope"]}


def static_inputs(row, selection, data_root, *, model_snapshot_map=None, runtime_build_audit=None, recurrent_batching=None, iq_panel=None, slot_order=None, host_offload=None, tensor_storage=None, gpu_invocation=None, sampling=None, nonflash_kv_view=None, mmvq_issue=None):
    """Static allowlist only: measured timing/profile fields are discarded."""
    raw = row["config"]
    config = {k: raw[k] for k in STATIC_KEYS if k in raw}
    model_ref = row["model_ref"]
    if Path(config["model"]).resolve() != grid.resolve_data(model_ref["path"], data_root):
        raise ValueError("model path differs from selected GGUF identity")
    config["model"] = str(grid.resolve_data(config["model"], data_root))
    config["model_sha256"] = model_ref["sha256"]
    prediction_model_ref = dict(model_ref)
    if model_snapshot_map is not None:
        prediction_model_ref = dict(model_snapshot_map.get(config["model"], model_ref))
        if prediction_model_ref["sha256"] != model_ref["sha256"]:
            raise ValueError("prediction model copy differs from native GGUF identity")
    if "flash_attention" in raw:
        config["flash_attn"] = raw["flash_attention"]
    if raw.get("load_mode", "mmap") != "mmap":
        raise ValueError("only frozen mmap load mode is currently supported")
    native_refs = row["native_runtime_refs"]
    server_refs = [r for r in native_refs if Path(r["path"]).name.lower() == "llama-server.exe"]
    if len(server_refs) != 1:
        raise ValueError("one selected native server identity required")
    runtime_ref = server_refs[0]
    grid.resolve_data(runtime_ref["path"], data_root)
    hardware = row["static_hardware"]["frozen_hardware"]
    gpu = hardware.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError("frozen_hardware.gpu required")
    physical = {k: hardware[k] for k in ("cpu", "cpu_topology", "cpu_cores", "cpu_threads", "host_memory", "pcie") if k in hardware}
    physical["gpu"] = {k: gpu[k] for k in ("name", "uuid", "memory_mib", "memory_free_mib", "memory_used_mib", "driver", "compute_capability", "clocks", "pcie", "bandwidth_gbps_one_way") if k in gpu}
    samples, basis, state_refs = measurement_clock_snapshot(row["static_hardware"], gpu.get("uuid"), data_root)
    config.update(gpu_sm_clock_samples_mhz=samples, gpu_clock_source=basis)
    graph_evidence = compiled_graph_evidence(native_refs, data_root, verified_audit=runtime_build_audit)
    if config.get("compiled_cuda_graphs") is not None and config["compiled_cuda_graphs"] != graph_evidence["compiled_cuda_graphs"]:
        raise ValueError("compiled CUDA Graphs flag lacks matching artifact evidence")
    config["compiled_cuda_graphs"] = graph_evidence["compiled_cuda_graphs"]
    host_binding = host_offload["cells"][row["cell_id"]] if host_offload else None
    gpu_binding = gpu_invocation["cells"][row["cell_id"]] if gpu_invocation else None
    mmvq_binding = mmvq_issue["cells"][row["cell_id"]] if mmvq_issue else None
    if host_binding and host_binding["status"] == "verified":
        config["op_offload"] = host_binding["op_offload_enabled"]
    return {"cell_id": row["cell_id"], "model_key": row["model_key"],
        "deployment": row.get("deployment", "explicit_gpu_layers_" + str(config.get("gpu_layers"))),
        "config": config, "hardware_snapshot": physical,
        "sampling_binding": sampling["cells"][row["cell_id"]] if sampling else None,
        "nonflash_kv_view_contract": nonflash_kv_view["cells"][row["cell_id"]] if nonflash_kv_view else None,
        "native_model_ref": dict(model_ref), "prediction_model_ref": prediction_model_ref,
        "recurrent_batching_contract": recurrent_batching["contract"] if recurrent_batching else None,
        "recurrent_batching_evidence": recurrent_batching,
        "slot_order_contract": slot_order["contract"] if slot_order else None,
        "slot_order_evidence": slot_order,
        "host_offload_source_contract": host_binding["source_contract"] if host_binding else None,
        "host_offload_evidence": host_binding,
        "tensor_storage_contract": tensor_storage["contract"] if tensor_storage else None,
        "tensor_storage_evidence": tensor_storage,
        "tensor_storage_f32_hidden": tensor_storage["f32_hidden_storage_requested"] if tensor_storage else False,
        "gpu_invocation_contract": gpu_binding["contract"] if gpu_binding else None,
        "gpu_invocation_evidence": gpu_binding,
        "gpu_mmq_source_costs": gpu_invocation["mmq_source_costs_requested"] if gpu_invocation else False,
        "gpu_conversion_cta_costs": gpu_invocation.get("conversion_cta_costs_requested", False) if gpu_invocation else False,
        **({"mmvq_vector_issue_bound": True, "mmvq_issue_contract": mmvq_binding["contract"],
            "mmvq_issue_evidence": mmvq_binding} if mmvq_binding is not None else {}),
        "cpu_iq_panel_reuse": iq_panel["dispatch"] if iq_panel else None,
        "cpu_iq_panel_evidence": iq_panel,
        "hardware_ref": row["static_hardware"].get("frozen_hardware_ref"),
        "measurement_state_refs": state_refs, "runtime_build_evidence": graph_evidence,
        "runtime_ref": runtime_ref, "runtime_module_refs": [r for r in native_refs if Path(r["path"]).suffix.lower() == ".dll"]}


def configuration(inputs):
    raw = inputs["config"]
    ids = raw.get("prompt_token_ids")
    prompt = integer(raw.get("expected_prompt_tokens", raw.get("prompt_tokens", len(ids) if isinstance(ids, list) else None)), "prompt_tokens")
    if not isinstance(ids, list) or len(ids) != prompt or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("exact frozen prompt token IDs/count required")
    output = integer(raw.get("output", raw.get("output_tokens")), "output_tokens")
    parallel = integer(raw.get("parallel"), "parallel")
    threads = integer(raw.get("threads", 16), "threads")
    if threads != 16 or raw.get("threads_batch", threads) != threads:
        raise ValueError("dataset requires 16 native worker and batch threads")
    gpu_layers = raw.get("gpu_layers")
    if type(gpu_layers) is not int or gpu_layers < -1:
        raise ValueError("actual native gpu_layers must be explicit")
    slot = integer(raw.get("kv_unified_per_slot", 2048), "slot context")
    total = integer(raw.get("context", raw.get("ctx", slot * parallel)), "native context")
    if slot != 2048 or total != slot * parallel or prompt + output > slot:
        raise ValueError("native context must be 2048 * parallel and hold prompt/output")
    for key, expected in {"flash_attn": False, "kv_type_k": "f16", "kv_type_v": "f16", "kv_unified": True, "cont_batching": True, "mmap": True, "mlock": False, "offload_kqv": True, "split_mode": "layer", "main_gpu": 0}.items():
        if key in raw and raw[key] != expected:
            raise ValueError("unsupported native configuration: " + key)
    config = {"ctx": slot, "parallel": parallel, "batch": integer(raw.get("batch", 64), "batch"), "ubatch": integer(raw.get("ubatch", 64), "ubatch"), "threads": threads, "gpu_layers": gpu_layers, "seed": integer(raw.get("seed", 42), "seed", 0), "coherent_dma_mode": raw.get("coherent_dma_mode", "pipelined"), "op_offload": raw.get("op_offload", True)}
    if type(config["op_offload"]) is not bool:
        raise ValueError("op_offload must be boolean")
    env = raw.get("environment", {})
    if not isinstance(env, dict) or any(v is not None and not isinstance(v, str) for v in env.values()):
        raise ValueError("runtime environment must contain strings or null")
    return prompt, output, config, total, env


def gpu_clock(inputs):
    raw = inputs["config"]
    samples = raw.get("gpu_sm_clock_samples_mhz")
    if samples is not None:
        if not isinstance(samples, list) or not samples:
            raise ValueError("GPU clock samples must be nonempty")
        mhz = statistics.median([positive(x, "GPU SM clock MHz") for x in samples])
        source = raw.get("gpu_clock_source", "frozen_native_clock_samples_median")
    elif raw.get("gpu_sm_clock_mhz") is not None:
        mhz = positive(raw["gpu_sm_clock_mhz"], "GPU SM clock MHz")
        source = raw.get("gpu_clock_source", "frozen_native_sm_clock")
    else:
        clocks = inputs["hardware_snapshot"]["gpu"].get("clocks", {})
        mhz = positive(clocks.get("sm_mhz", clocks.get("graphics_mhz")), "frozen GPU SM clock MHz")
        source = "frozen_hardware_snapshot.gpu.clocks"
    return {"mhz": float(mhz), "frequency_ghz": mhz / 1000, "clock_hz": mhz * 1e6, "source": source}


def unsupported_dimensions(inputs, model=None):
    raw = inputs["config"]
    rows = [
        {"dimension": "cpu_worker_binding", "status": "conditional", "native": {k: raw.get(k) for k in ("threads", "threads_batch", "worker_cpu_mask", "poll", "priority")}, "reason": "16 cores are modeled; physical worker mask, strict binding, polling and scheduling are not."},
        {"dimension": "kv_shared_physical_pool", "status": "conditional", "native_total_context_tokens": 2048 * raw.get("parallel", 1), "simulator_slot_context_tokens": 2048, "reason": "Logical slot capacity matches; native unified physical KV pool allocation/contention parity remains unproven."},
        {"dimension": "runtime_op_offload_contract", "status": "unsupported", "runtime_ref": inputs["runtime_ref"], "op_offload": raw.get("op_offload", True), "reason": "Actual new runtime identity is retained and does not inherit the old semantic-runtime CUDA op-offload contract."}]
    if inputs.get("mmvq_issue_evidence") is not None:
        proof = inputs["mmvq_issue_evidence"]
        rows.append({"dimension": "mmvq_vector_integer_issue_bound", "status": proof["status"],
            "reason": "Conditional source/PTX dot issue lower bound; native instruction mapping and wall-time clock bound are unproven.",
            "unpriced_work": proof["unpriced_work"], "calibration_applied": False,
            "conditions": proof.get("conditions", []),
            "original_compile_header_bytes_proven": (proof.get("mmvq_source_binding") or {}).get("original_compile_header_bytes_proven", False),
            "formal_prediction_eligible": False, "hbm_accounting": "unchanged_legacy_unverified"})
    if inputs.get("nonflash_kv_view_contract") is not None:
        rows.append({"dimension": "nonflash_physical_kv_view", "status": "conditional",
            "reason": "Source-bound padded lower bound from the longest current retained prefix; full unified allocation/high-water, inactive and cached slots remain unknown."})
    host_binding = inputs.get("host_offload_evidence")
    if isinstance(host_binding, Mapping):
        rows[2] = {"dimension": "runtime_op_offload_contract",
            "status": "conditional" if host_binding.get("status") == "verified" else "unsupported",
            "runtime_ref": inputs["runtime_ref"], "source_binding_status": host_binding.get("status"),
            "op_offload": host_binding.get("op_offload_enabled"),
            "uncovered_reasons": host_binding.get("uncovered_reasons", []),
            "native_dispatch_proven": False,
            "reason": "Verified source/build/runtime capability remains subject to per-invocation tensor/layout/buffer qualification and analytical cost limits" if host_binding.get("status") == "verified" else "Recorded source/runtime evidence does not yet qualify host CUDA offload"}
    if inputs.get("cpu_iq_panel_reuse") is not None:
        rows.append({"dimension": "cpu_iq_panel_historical_dispatch", "status": "conditional",
            "historical_state": "unknown", "assume_default_unset": inputs["cpu_iq_panel_reuse"]["assume_default_unset"],
            "native_dispatch_proven": False, "evaluation_scope": inputs["cpu_iq_panel_reuse"]["evaluation_scope"],
            "reason": "Historical GGML_NO_IQ_PANEL was not captured; candidate assumption never becomes observed native dispatch evidence"})
    tensor_storage = inputs.get("tensor_storage_evidence")
    if isinstance(tensor_storage, Mapping):
        rows.append({"dimension": "tensor_storage_timing_completeness", "status": "conditional",
            "f32_hidden_storage_requested": inputs.get("tensor_storage_f32_hidden", False),
            "native_dispatch_proven": False, "reason": "Source-bound logical GET_ROWS traffic and storage do not price row dequantization, repeated-index cache reuse, cache-line/page or write-allocation effects",
            "limits": tensor_storage.get("limits", [])})
    invocation = inputs.get("gpu_invocation_evidence")
    if isinstance(invocation, Mapping):
        rows.append({"dimension": "gpu_physical_invocation_and_source_costs", "status": "conditional",
            "binding_status": invocation.get("status"), "native_dispatch_proven": False,
            "mmq_source_costs_requested": inputs.get("gpu_mmq_source_costs", False),
        "conversion_cta_costs_requested": inputs.get("gpu_conversion_cta_costs", False),
            "conditional_reasons": invocation.get("conditional_reasons", []),
            "uncovered_reasons": invocation.get("uncovered_reasons", []), "unpriced_terms": invocation.get("unpriced_terms", []),
            "reason": "Physical GGUF tensor checks qualify simulation geometry; historical model-specific graph bodies and native per-operator dispatch remain unproven"})
    if raw.get("gpu_layers") == -1:
        rows.append({"dimension": "auto_gpu_layer_fit", "status": "conditional", "native": -1, "reason": "Native -ngl -1 is auto with fit; simulator treats it as all layers. Actual loaded layer count is not established for this cell."})
    sampling = inputs.get("sampling_binding")
    rows.append({"dimension": "cpu_sampling_chain", "status": "conditional" if sampling else "unmodeled",
        "policy_bound": sampling is not None,
        "reason": "Source-bound candidate materialization and top-k scan; bias/suppression, filter tail, RNG and accept remain partial" if sampling else "No source/config sampling policy bound; sampling cost is absent",
        "limitations": sampling.get("limitations", []) if isinstance(sampling, Mapping) else []})
    rows.append({"dimension": "cuda_graph_lifecycle", "status": "conditional", "compiled_cuda_graphs": raw.get("compiled_cuda_graphs"), "reason": "No CUDA Graph replay timing or prior native profile is applied; direct launch/synchronization parity remains unvalidated."})
    if model is not None and ("qwen35" in str(getattr(model, "architecture", "")) or "qwen3" in str(getattr(model, "name", "")).lower()):
        rows.append({"dimension": "hybrid_recurrent_invocation_geometry", "status": "conditional", "reason": "Native hybrid/recurrent ubatch invocation geometry is unvalidated for this cell."})
    return rows



def gpu_layer_mapping(inputs, gguf, model):
    """Translate only evidenced auxiliary-unit accounting, never clamp counts.

    GGUF block_count includes nextn/MTP blocks, while the imported executable
    model excludes them. The native full-offload loading count includes those
    auxiliary units plus the output head. No shape name or timing participates.
    """
    native = inputs["config"]["gpu_layers"]
    executable = integer(getattr(model, "num_layers", None), "model executable layer count")
    output_units = 1  # the builder/control-plane counts the lm_head as a unit
    sim_units = executable + output_units
    architecture = getattr(gguf, "architecture", None)
    metadata = getattr(gguf, "metadata", {})
    block_key = f"{architecture}.block_count" if architecture else None
    mtp_key = f"{architecture}.nextn_predict_layers" if architecture else None
    raw_blocks = metadata.get(block_key) if block_key else None
    mtp = metadata.get(mtp_key, 0) if mtp_key else 0
    imported = getattr(gguf, "n_layer", None)
    result = {"native_gpu_layers": native, "simulator_gpu_layers": native,
        "model_executable_layers": executable, "simulator_loading_units": sim_units,
        "gguf_declared_block_count": raw_blocks, "gguf_imported_executable_layers": imported,
        "gguf_mtp_layer_count": mtp, "excluded_mtp_loading_units": 0,
        "output_loading_units": output_units, "mapping_applied": False,
        "conversion_basis": "native count preserved; no loading-unit conversion required",
        "metadata_keys": {"block_count": block_key, "nextn_predict_layers": mtp_key},
        "gguf_sha256": gguf.sha256}
    if native <= sim_units:
        return result
    # A larger requested count can be mapped only for explicit full offload,
    # with auto-fit disabled, and mutually consistent GGUF/imported geometry.
    if type(raw_blocks) is not int or type(mtp) is not int or mtp <= 0:
        raise ValueError("native gpu_layers exceeds simulator units without positive GGUF MTP evidence; refusing to clamp")
    if type(imported) is not int or imported != executable or raw_blocks - mtp != executable:
        raise ValueError("GGUF MTP and executable layer counts disagree; refusing loading-unit conversion")
    native_units = raw_blocks + output_units
    if native != native_units:
        raise ValueError("native gpu_layers is not the exact GGUF full-offload count; refusing to clamp")
    if inputs["config"].get("fit_params") is not False:
        raise ValueError("MTP loading-unit conversion requires explicit fit_params=false")
    result.update(simulator_gpu_layers=sim_units, native_full_loading_units=native_units,
        excluded_mtp_loading_units=mtp, mapping_applied=True,
        conversion_basis="Exact explicit native full offload with fit disabled: GGUF block_count plus lm_head minus metadata-declared MTP units absent from executable trunk")
    return result



def replan_final_static_scenario(scenario, *, recurrent_batching_contract=None, slot_order_contract=None):
    """Recompute derived placement after physical-input edits, then validate.

    Keep the generated-output ledger and authored constraints. The existing
    typed planner replaces its own decision/evidence; fingerprints are never
    patched by hand and validation is never disabled.
    """
    from heterollm_sim.control_plane_state import mapping_fingerprint_status
    from heterollm_sim.llama_scenario import apply_llama_runtime_config
    from heterollm_sim.planner import validate_scenario
    before = mapping_fingerprint_status(scenario)
    options = {"recurrent_batching_contract": recurrent_batching_contract} if recurrent_batching_contract is not None else {}
    if slot_order_contract is not None:
        options["slot_order_contract"] = slot_order_contract
    refreshed = apply_llama_runtime_config(scenario, scenario.llama_cpp_config, materialize_placement=True, **options)
    validation = validate_scenario(refreshed)
    validation.raise_for_errors()
    after = mapping_fingerprint_status(refreshed)
    return refreshed, {"method": "typed_runtime_placement_replanned_after_all_static_bindings",
        "previous_input_fingerprint": before["input_fingerprint"],
        "previous_decision_stale": before["mapping_stale"],
        "input_fingerprint": after["input_fingerprint"],
        "current_input_fingerprint": after["current_input_fingerprint"],
        "mapping_stale": after["mapping_stale"], "normal_validation_passed": validation.is_valid}



DIAGNOSTIC_EVENT_LIMIT = 2000
DIAGNOSTIC_EVENT_MAX_BYTES = 2 * 1024 * 1024
BATCH_DETAIL_LIMIT = 512


def finite_time(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def bounded_value(value, depth=0):
    """Bound already-retained diagnostic metadata without serializing graphs."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"unavailable": "nonfinite"}
    if isinstance(value, str):
        return value if len(value) <= 2048 else {"prefix": value[:2048], "truncated_characters": len(value) - 2048}
    if depth >= 4:
        return {"truncated": "depth_limit", "type": type(value).__name__}
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 32:
                result["__omitted_keys__"] = len(value) - 32
                break
            result[str(key)] = bounded_value(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [bounded_value(item, depth + 1) for item in value[:32]]
        if len(value) > 32:
            return {"items": items, "omitted_items": len(value) - 32}
        return items
    return {"unavailable_type": type(value).__name__}


def request_timepoints(result, metric, timing, request):
    """Absolute simulation clock only; absent/inferred events stay unavailable."""
    begin = finite_time(timing.get("engine_request_begin_ns"))
    first = finite_time(getattr(metric, "first_token_ns", None))
    last_source = timing.get("engine_last_token_source")
    reliable_last = last_source in {"accumulator_complete", "accumulator_with_partial_event_retention", "complete_token_events"}
    last = finite_time(timing.get("engine_last_token_ns")) if reliable_last else None
    fields = {"engine_request_begin_ns": begin, "engine_first_token_ns": first,
        "engine_last_token_ns": last, "arrival_ns": finite_time(getattr(metric, "arrival_ns", None)),
        "service_start_ns": finite_time(getattr(metric, "start_ns", None)),
        "service_finish_ns": finite_time(getattr(metric, "finish_ns", None)),
        "metric_done_ns": finite_time(getattr(metric, "done_ns", None))}
    sources = {"engine_request_begin_ns": timing.get("engine_start_source") if begin is not None else "unavailable",
        "engine_first_token_ns": "request_metrics.first_token_ns" if first is not None else "unavailable",
        "engine_last_token_ns": last_source if last is not None else "unavailable_or_inferred"}
    events = getattr(getattr(result, "serving", None), "events", ()) or ()
    begin_events = [finite_time(event.timestamp_ns) for event in events
        if getattr(event, "request_id", None) == request["request_id"]
        and getattr(event, "event_type", None) == "engine_request_begin"]
    begin_events = [value for value in begin_events if value is not None]
    missing = [name for name in sources if fields[name] is None]
    issues = []
    if not missing and not (begin <= first <= last):
        issues.append("engine_timepoints_out_of_order")
    event_match = None
    if begin is not None and begin_events:
        event_match = all(math.isclose(begin, value, rel_tol=1e-12, abs_tol=1e-6) for value in begin_events)
        if not event_match:
            issues.append("engine_begin_metric_and_retained_event_disagree")
    derived = {}
    if not missing and not issues:
        derived = {"engine_ttft_ms": (first - begin) / 1e6,
            "engine_e2e_ms": (last - begin) / 1e6}
        tokens = request.get("visible_output_tokens")
        if isinstance(tokens, int) and tokens > 1:
            derived["engine_tpot_ms"] = (last - first) / (tokens - 1) / 1e6
        for key, value in derived.items():
            if request.get(key) is not None and not math.isclose(value, request[key], rel_tol=1e-9, abs_tol=1e-9):
                issues.append(key + "_does_not_match_timepoints")
    return {**fields, "first_prompt_batch_processing_ns": begin if sources["engine_request_begin_ns"] == "first_prompt_batch_processing" else None,
        "timepoint_sources": sources, "timepoint_validation": {
            "status": "inconsistent" if issues else "incomplete" if missing else "verified",
            "missing_fields": missing, "issues": issues, "derived_intervals_ms": derived,
            "retained_engine_begin_event_times_ns": begin_events,
            "engine_begin_matches_retained_event": event_match,
            "clock": "simulator_absolute_ns; not wall-clock or native clock"}}


def engine_cohort_span(requests):
    keys = ("engine_request_begin_ns", "engine_first_token_ns", "engine_last_token_ns")
    valid = [r for r in requests if all(finite_time(r.get(key)) is not None for key in keys)
             and r[keys[0]] <= r[keys[1]] <= r[keys[2]]
             and r.get("timepoint_validation", {}).get("status") != "inconsistent"]
    if not requests or len(valid) != len(requests):
        return {"status": "incomplete", "planned_requests": len(requests), "observed_requests": len(valid),
            "engine_begin_min_ns": None, "engine_last_token_max_ns": None,
            "cohort_engine_span_ns": None, "cohort_engine_span_ms": None}
    begin = min(r[keys[0]] for r in requests)
    finish = max(r[keys[2]] for r in requests)
    starts = {}
    boundaries = []
    for request in requests:
        starts.setdefault(request[keys[0]], []).append(request["request_id"])
        if request[keys[2]] > request[keys[0]]:
            boundaries.extend(((request[keys[0]], 1), (request[keys[2]], -1)))
    active = peak = 0
    for _, delta in sorted(boundaries):
        active += delta
        peak = max(peak, active)
    return {"status": "complete", "planned_requests": len(requests), "observed_requests": len(valid),
        "engine_begin_min_ns": begin, "engine_last_token_max_ns": finish,
        "cohort_engine_span_ns": finish - begin, "cohort_engine_span_ms": (finish - begin) / 1e6,
        "engine_start_spread_ns": max(starts) - min(starts),
        "engine_start_order": [{"timestamp_ns": stamp, "request_ids": starts[stamp]} for stamp in sorted(starts)],
        "max_overlapping_engine_request_intervals": peak,
        "overlap_semantics": "request engine-clock intervals, not proof of simultaneous kernel execution"}



def compact_dispatch_evidence(metadata):
    """Read named core ledgers before general metadata truncation can hide them."""
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {key: bounded_value(dict(value)) if isinstance(value, Mapping) else None
        for key in ("cpu_iq_panel_reuse", "host_gemm_offload", "tensor_storage", "gpu_invocations", "mmq_source_work")
        for value in (metadata.get(key),)}


def dispatch_qualification(scenario, inputs=None):
    """Keep actual adapter decisions distinct from source proof or dispatch counts."""
    metadata = getattr(getattr(scenario, "workload", None), "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    inputs = inputs or {}
    host = metadata.get("llama_cpp_cuda_op_offload")
    host = host if isinstance(host, Mapping) else {}
    storage = metadata.get("llama_cpp_tensor_storage")
    storage = storage if isinstance(storage, Mapping) else {}
    invocation = metadata.get("llama_cpp_gpu_native_invocations")
    invocation = invocation if isinstance(invocation, Mapping) else {}
    slot = metadata.get("llama_cpp_slot_order")
    slot = slot if isinstance(slot, Mapping) else {}
    iq = inputs.get("cpu_iq_panel_reuse")
    iq = iq if isinstance(iq, Mapping) else metadata.get("llama_cpp_cpu_iq_panel_reuse", {})
    iq = iq if isinstance(iq, Mapping) else {}
    return {
        "cpu_iq_panel_reuse": {"requested": bool(iq),
            "historical_state": iq.get("no_iq_panel_environment_state"),
            "conditional": iq.get("assume_default_unset"),
            "native_dispatch_proven": iq.get("native_dispatch_proven"),
            "evaluation_scope": iq.get("evaluation_scope")},
        "host_gemm_offload": {"requested": inputs.get("host_offload_evidence") is not None,
            "binding_status": (inputs.get("host_offload_evidence") or {}).get("status"),
            "binding_uncovered_reasons": (inputs.get("host_offload_evidence") or {}).get("uncovered_reasons"),
            "native_dispatch_proven": (inputs.get("host_offload_evidence") or {}).get("native_dispatch_proven"),
            "qualified": host.get("status") == "enabled" if host else None,
            "status": host.get("status"), "reason": host.get("reason"),
            "op_offload": host.get("op_offload"), "cuda_backend_available": host.get("cuda_backend_available"),
            "minimum_m": host.get("minimum_m"), "environment": bounded_value(host.get("environment")),
            "actual_dispatch_requires_physical_invocation_evidence": True},
        "tensor_storage": {"requested": inputs.get("tensor_storage_contract") is not None,
            "f32_hidden_storage_requested": inputs.get("tensor_storage_f32_hidden", False),
            "native_dispatch_proven": False if inputs.get("tensor_storage_contract") is not None else None,
            **{key: bounded_value(storage.get(key)) for key in ("qualified", "status", "reasons", "previous_f32_hidden_storage", "timing_completeness", "scope")}},
        "gpu_invocations": {"requested": inputs.get("gpu_invocation_evidence") is not None,
            "mmq_source_costs_requested": inputs.get("gpu_mmq_source_costs", False),
        "conversion_cta_costs_requested": inputs.get("gpu_conversion_cta_costs", False),
            "kernel_environment": bounded_value((inputs.get("gpu_invocation_evidence") or {}).get("kernel_environment")),
            "binding_uncovered_reasons": (inputs.get("gpu_invocation_evidence") or {}).get("uncovered_reasons"),
            **{key: bounded_value(invocation.get(key)) for key in ("applied", "status", "conditional", "native_dispatch_proven",
                "reasons", "qualified_projection_groups", "uncovered_group_reason_counts", "fusion_enabled",
                "mmq_source_costs", "cache_write_qualified", "cache_write_reason", "unpriced_terms")}},
        "storage": {"f32_hidden_storage": metadata.get("llama_cpp_f32_hidden_storage"),
            "evidence_source": "final_scenario.workload.metadata.llama_cpp_f32_hidden_storage"},
        "slot_order": {key: bounded_value(slot.get(key)) for key in (
            "qualified", "applied", "status", "reasons", "phase_candidate_order",
            "preserves_engine_start_definition")},
    }


def retained_dispatch_summary(result):
    """Aggregate retained core ledgers; absent observations remain unknown."""
    serving = getattr(result, "serving", None)
    batches = tuple(getattr(serving, "batches", ()) or ())
    total = getattr(getattr(serving, "scheduler_metrics", None), "total_batches", None)
    complete = total == len(batches) if type(total) is int else None
    result_summary = {"schema": "stable-native-retained-dispatch-summary/v1",
        "count_scope": "simulator physical tasks in retained serving batch ledgers",
        "retained_batch_count": len(batches), "total_batches": total,
        "history_complete": complete, "native_timing_used": False}
    for key in ("cpu_iq_panel_reuse", "host_gemm_offload", "tensor_storage", "gpu_invocations", "mmq_source_work"):
        ledgers = []
        for batch in batches:
            metadata = getattr(getattr(batch, "cost", None), "metadata", {})
            value = metadata.get(key) if isinstance(metadata, Mapping) else None
            if isinstance(value, Mapping):
                ledgers.append(value)
        summary = {"status": "observed" if ledgers else "unavailable",
            "summarized_batch_count": len(ledgers), "missing_batch_summaries": len(batches) - len(ledgers),
            "all_batches_summarized": bool(batches) and len(ledgers) == len(batches) and complete is True,
            "applied_tasks": None, "observed_applied_tasks": None,
            "reason": None if ledgers else "core ledger was not retained; no zero is inferred"}
        numeric_keys = {name for ledger in ledgers for name, value in ledger.items()
            if name.endswith("_tasks") and type(value) is int}
        if key == "tensor_storage":
            numeric_keys.update(name for ledger in ledgers for name, value in ledger.items()
                if type(value) is int and "capacity" not in name and name.endswith(("_bytes", "_rows", "_elements")))
        for name in sorted(numeric_keys):
            values = [ledger.get(name) for ledger in ledgers]
            observed = sum(values) if all(type(value) is int and value >= 0 for value in values) else None
            summary["observed_" + name] = observed
            summary[name] = observed if summary["all_batches_summarized"] else None
        count_keys = {name for ledger in ledgers for name, value in ledger.items()
            if name.endswith("_counts") and isinstance(value, Mapping)}
        for name in sorted(count_keys):
            counts = Counter()
            for ledger in ledgers:
                for label, value in ledger.get(name, {}).items():
                    if type(value) is int and value >= 0:
                        counts[str(label)] += value
            summary[name] = dict(sorted(counts.items()))
        if key == "gpu_invocations":
            query_counts = Counter()
            query_missing_reasons = Counter()
            query_ledgers = [ledger.get("kernel_query_ledger") for ledger in ledgers]
            valid_ledgers = [value for value in query_ledgers if isinstance(value, Mapping)]
            for value in valid_ledgers:
                query_missing_reasons.update(value.get("missing_reason_counts", {}))
                for row in value.get("signatures", ()):
                    if isinstance(row, Mapping) and type(row.get("task_count")) is int and row["task_count"] > 0:
                        query_counts[json.dumps(row["key"], sort_keys=True, separators=(",", ":"), allow_nan=False)] += row["task_count"]
            summary["kernel_query_ledger"] = {
                "schema": "stable-native-kernel-query-ledger/v1",
                "signatures": [{"key": json.loads(ident), "task_count": count} for ident, count in sorted(query_counts.items())],
                "summarized_batches": len(valid_ledgers),
                "complete": summary["all_batches_summarized"] and len(valid_ledgers) == len(ledgers)
                    and all(v.get("complete_geometry") is True for v in valid_ledgers),
                "represented_tasks": sum(query_counts.values()),
                "unrepresented_tasks": sum(value.get("unrepresented_tasks", 0) for value in valid_ledgers),
                "missing_reason_counts": dict(sorted(query_missing_reasons.items())),
                "missing_batch_ledgers": len(batches) - len(valid_ledgers),
                "native_dispatch_proven": False, "calibration_eligible": False,
                "scope": "simulated main physical geometry only; cache and native dispatch unknown"}
        if key == "tensor_storage":
            capacity_keys = {name for ledger in ledgers for name, value in ledger.items()
                if type(value) is int and "capacity" in name and name.endswith("_bytes")}
            for name in sorted(capacity_keys):
                summary["maximum_observed_" + name] = max(ledger.get(name, 0) for ledger in ledgers)
            summary["capacity_semantics"] = "maximum retained table footprint; never a sum of repeated batch capacities"
            summary["timing_completeness"] = "partial; selected-row conversion/dequantization, cache-line/page and write-allocation costs remain unpriced"
        if key == "host_gemm_offload":
            observed = summary.get("observed_applied_tasks")
            summary["actual_cpu_to_gpu_offload"] = True if type(observed) is int and observed > 0 else False if summary.get("applied_tasks") == 0 else None
            summary["dispatch_observation_domain"] = "simulation only; native per-operator dispatch remains unproven"
        result_summary[key] = summary
    return result_summary

def batch_schedule(result, requests, *, scenario=None, inputs=None):
    serving = getattr(result, "serving", None)
    batches = getattr(serving, "batches", None)
    if batches is None:
        return {"status": "unavailable", "reason": "result has no retained serving batch history", "cohort_engine_timeline": engine_cohort_span(requests)}
    batches = tuple(batches)
    rows, counts = [], Counter()
    per_request = {r["request_id"]: {"retained_batch_count": 0, "first_retained_prefill_batch_start_ns": None} for r in requests}
    max_requests = multi_request = 0
    for index, batch in enumerate(batches):
        kind = str(getattr(batch, "kind", "unknown"))
        counts[kind] += 1
        ids = list(getattr(batch, "request_ids", ()) or ())
        max_requests = max(max_requests, len(set(ids)))
        multi_request += int(len(set(ids)) > 1)
        items = tuple(getattr(batch, "items", ()) or ())
        for rid in set(ids):
            if rid not in per_request:
                continue
            per_request[rid]["retained_batch_count"] += 1
            is_prefill = kind == "prefill" or any(getattr(item, "request_id", None) == rid and getattr(item, "phase", None) == "prefill" for item in items)
            if is_prefill and finite_time(getattr(batch, "start_ns", None)) is not None:
                previous = per_request[rid]["first_retained_prefill_batch_start_ns"]
                per_request[rid]["first_retained_prefill_batch_start_ns"] = batch.start_ns if previous is None else min(previous, batch.start_ns)
        if index < BATCH_DETAIL_LIMIT:
            cost = getattr(batch, "cost", None)
            rows.append({"batch_index": index, "cohort_id": getattr(batch, "cohort_id", None), "kind": kind,
                "start_ns": finite_time(getattr(batch, "start_ns", None)), "end_ns": finite_time(getattr(batch, "end_ns", None)),
                "request_ids": ids[:32], "request_ids_truncated": len(ids) > 32,
                "token_count": getattr(batch, "token_count", None), "cost_duration_ns": finite_time(getattr(cost, "duration_ns", None)),
                "items": [{key: getattr(item, key, None) for key in ("request_id", "phase", "token_count", "context_tokens", "completion_cursor")} for item in items[:32]],
                "items_truncated": len(items) > 32, "metadata": bounded_value(getattr(batch, "metadata", {})),
                **compact_dispatch_evidence(getattr(cost, "metadata", {})),
                "dispatch_qualification": dispatch_qualification(scenario, inputs),
                "cost_metadata": bounded_value(getattr(cost, "metadata", {}))})
    for request in requests:
        info = per_request[request["request_id"]]
        start, begin = info["first_retained_prefill_batch_start_ns"], request.get("engine_request_begin_ns")
        info["engine_begin_to_first_retained_prefill_batch_ns"] = start - begin if start is not None and begin is not None else None
    scheduler = getattr(serving, "scheduler_metrics", None)
    totals = {key: getattr(scheduler, key, None) for key in ("scheduling_rounds", "total_batches", "prefill_batches", "decode_batches", "mtp_batches", "max_batch_sequences", "max_batch_tokens")}
    return {"status": "retained_history", "scope": "already-retained serving batches; no reconstruction of missing history",
        "result_type": type(result).__name__, "retention_policy": getattr(result, "retention_policy", None),
        "retained_batch_count": len(batches), "returned_batch_count": len(rows), "detail_limit": BATCH_DETAIL_LIMIT,
        "details_truncated": len(rows) < len(batches), "phase_batch_counts": dict(counts),
        "max_requests_in_retained_batch": max_requests, "multi_request_retained_batches": multi_request,
        "scheduler_metrics": totals, "runtime_kernel_metrics": bounded_value(getattr(serving, "runtime_kernel_metrics", {})),
        "per_request": per_request, "batches": rows, "cohort_engine_timeline": engine_cohort_span(requests)}


def diagnostic_event_trace(result, limit=DIAGNOSTIC_EVENT_LIMIT, max_bytes=DIAGNOSTIC_EVENT_MAX_BYTES):
    integer(limit, "diagnostic event limit")
    if limit > 20000 or max_bytes < 4096 or max_bytes > 8 * 1024 * 1024:
        raise ValueError("diagnostic event bounds exceed supported limits")
    events = getattr(getattr(result, "serving", None), "events", None)
    if events is None:
        return {"status": "unavailable", "reason": "no retained serving events", "events": []}
    rows, used = [], 0
    for index, event in enumerate(events):
        if index >= limit:
            break
        row = {"retained_event_index": index, "timestamp_ns": finite_time(getattr(event, "timestamp_ns", None)),
            "event_type": getattr(event, "event_type", None), "request_id": getattr(event, "request_id", None),
            "cohort_id": getattr(event, "cohort_id", None), "details": bounded_value(getattr(event, "details", {}))}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2)
        size = len(encoded.encode("utf-8")) + 8 * (encoded.count("\n") + 1)
        if used + size > max_bytes - 4096:
            break
        used += size
        rows.append(row)
    return {"status": "retained_events", "clock": "simulator_absolute_ns", "source": "run_scenario aggregate serving.events",
        "retained_events_available": len(events), "returned_events": len(rows), "event_limit": limit,
        "max_bytes": max_bytes, "truncated": len(rows) < len(events), "history_complete": None,
        "completeness_reason": "Core retention may have removed earlier events; no missing events are synthesized", "events": rows}


def predict_cell(inputs, *, model_cache=None, diagnostic_events=False, diagnostic_event_limit=DIAGNOSTIC_EVENT_LIMIT):
    """Accept static-only worker inputs. No native actuals/profile argument."""
    prompt, output, config, total, env = configuration(inputs)
    clock = gpu_clock(inputs)
    path = Path(inputs.get("prediction_model_ref", {}).get("path", inputs["config"]["model"]))
    cache = {} if model_cache is None else model_cache
    if str(path) not in cache:
        gguf = grid.read_gguf_metadata(path)
        cache[str(path)] = gguf, grid.build_model_from_gguf(gguf)
    gguf, model = cache[str(path)]
    expected = inputs["config"].get("model_sha256")
    if expected and gguf.sha256 != expected:
        raise ValueError(f"GGUF SHA256 mismatch: path={path}; expected={expected}; actual={gguf.sha256}; native_model_path={inputs['config']['model']}")
    loading_mapping = gpu_layer_mapping(inputs, gguf, model)
    simulator_config = {**config, "gpu_layers": loading_mapping["simulator_gpu_layers"]}
    sampling = inputs.get("sampling_binding")
    policy = SamplingPolicy(**sampling["typed_policy"]) if sampling is not None else None
    scenario = grid.build_matching_scenario(prompt, output, model=model, sampling_policy=policy,
        hardware_snapshot=inputs["hardware_snapshot"], runtime_binary=Path(inputs["runtime_ref"]["path"]),
        runtime_environment=dict(env), **simulator_config)
    scenario = apply_host_offload_static_contract(scenario, inputs)
    scenario = apply_tensor_storage_static_contract(scenario, inputs)
    scenario = apply_gpu_invocation_static_contract(scenario, inputs)
    scenario = apply_nonflash_kv_view_static_contract(scenario, inputs, gguf=gguf)
    profiles = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    changed = []
    for ident, profile in profiles.get("gpu", {}).items():
        tensor = getattr(profile, "tensor_core", None)
        if tensor is not None:
            profiles["gpu"][ident] = replace(profile, tensor_core=replace(tensor, frequency_ghz=clock["frequency_ghz"]))
            changed.append(ident)
    if not changed:
        raise ValueError("no tensor-core frequency input available")
    metadata = dict(scenario.workload.metadata)
    if sampling is not None:
        metadata["native_sampling_binding"] = sampling
    metadata["serving_runtime"] = {**metadata.get("serving_runtime", {}), "kv_slot_context_tokens": 2048, "compiled_cuda_graphs": inputs["config"].get("compiled_cuda_graphs"), "cuda_graph_replay_cost_applied": False}
    if inputs.get("cpu_iq_panel_reuse") is not None:
        metadata["llama_cpp_cpu_iq_panel_reuse"] = dict(inputs["cpu_iq_panel_reuse"])
    scenario = replace(scenario, component_profiles=profiles,
        workload=replace(scenario.workload, metadata=metadata),
        hardware=replace(scenario.hardware, metadata={**scenario.hardware.metadata, "frozen_native_gpu_clock": clock}))
    scenario = apply_mmvq_issue_static_contract(scenario, inputs)
    contract = inputs.get("recurrent_batching_contract")
    options = {"recurrent_batching_contract": contract} if contract is not None else {}
    if inputs.get("slot_order_contract") is not None:
        options["slot_order_contract"] = inputs["slot_order_contract"]
    scenario, placement_refresh = replan_final_static_scenario(scenario, **options)
    slot_qualification = scenario.workload.metadata.get("llama_cpp_slot_order", {})
    slot_qualification = slot_qualification if isinstance(slot_qualification, dict) else {}
    result = grid.reporting.run_scenario(scenario, retention_policy="aggregate")
    requests = []
    for index, (rid, metric) in enumerate(sorted(result.metrics.request_metrics.items())):
        timing = grid._simulator_request_timing(result, metric)
        request = {"request_id": str(rid), "request_index": index, "prompt_tokens": prompt, "requested_output_tokens": output, "visible_output_tokens": getattr(metric, "visible_output_tokens", None)}
        for key in METRICS:
            value = timing.get(key)
            request[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
        for key in ("engine_start_source", "engine_last_token_source"):
            request[key] = timing.get(key)
        request.update(request_timepoints(result, metric, timing, request))
        requests.append(request)
    complete = len(requests) == config["parallel"] and all(r["visible_output_tokens"] == output and all(r[k] is not None for k in METRICS) for r in requests)
    prediction = {"status": "predicted" if complete else "incomplete", "reason": None if complete else "Missing request, token or timing boundary",
        "prediction_type": PREDICTION_TYPE, "native_answers_used": False, "calibration_applied": False, "formal_prediction_eligible": False,
        "unsupported_dimensions": unsupported_dimensions(inputs, model), "input_identity": {
            "static_inputs_sha256": grid.stable_hash(inputs), "sampling_binding": sampling, "model": {"path": str(path), "sha256": gguf.sha256},
            "nonflash_kv_view_contract": inputs.get("nonflash_kv_view_contract"),
            "native_model_ref": inputs.get("native_model_ref"), "prediction_model_ref": inputs.get("prediction_model_ref"),
            "control_plane_replan": placement_refresh, "cpu_iq_panel_reuse": inputs.get("cpu_iq_panel_reuse"),
            "slot_order_contract": inputs.get("slot_order_contract"),
            "host_offload_source_contract": inputs.get("host_offload_source_contract"),
            "host_offload_binding": inputs.get("host_offload_evidence"),
            "tensor_storage_contract": inputs.get("tensor_storage_contract"),
            "tensor_storage_f32_hidden_requested": inputs.get("tensor_storage_f32_hidden", False),
            "tensor_storage_binding": inputs.get("tensor_storage_evidence"),
            "gpu_invocation_contract": inputs.get("gpu_invocation_contract"),
            "gpu_invocation_binding": inputs.get("gpu_invocation_evidence"),
            **({"mmvq_vector_issue_bound": inputs["mmvq_vector_issue_bound"],
                "mmvq_issue_contract": inputs["mmvq_issue_contract"],
                "mmvq_issue_evidence": inputs["mmvq_issue_evidence"]}
                if "mmvq_issue_evidence" in inputs else {}),
            "gpu_mmq_source_costs_requested": inputs.get("gpu_mmq_source_costs", False),
            "gpu_conversion_cta_costs_requested": inputs.get("gpu_conversion_cta_costs", False),
            "slot_order_treatment": {"requested": inputs.get("slot_order_contract") is not None,
                "qualified": slot_qualification.get("qualified", False), "applied": slot_qualification.get("applied", False),
                "status": slot_qualification.get("status", "qualification_not_reported"),
                "reasons": slot_qualification.get("reasons", []),
                "phase_candidate_order_after_lowering": getattr(getattr(scenario.workload, "scheduler", None), "phase_candidate_order", None),
                "lowering_evidence": {key: bounded_value(value) for key, value in scenario.workload.metadata.items() if key.startswith("llama_cpp_slot_order")}},
            "prompt_token_ids_sha256": grid.stable_hash(inputs["config"]["prompt_token_ids"]),
            "native_configuration": inputs["config"], "simulator_configuration": simulator_config,
            "gpu_loading_unit_mapping": loading_mapping,
            "native_total_context_tokens": total, "simulator_slot_context_tokens": 2048,
            "gpu_clock": clock, "frequency_mapped_profile_ids": changed,
            "runtime_ref": inputs["runtime_ref"], "hardware_snapshot": inputs["hardware_snapshot"], "runtime_build_evidence": inputs.get("runtime_build_evidence")},
        "requests": requests, "aggregate": grid.aggregate(requests, config["parallel"]),
        "cohort_engine_timeline": engine_cohort_span(requests),
        "dispatch_qualification": dispatch_qualification(scenario, inputs), "dispatch_summary": retained_dispatch_summary(result),
        "batch_schedule": batch_schedule(result, requests, scenario=scenario, inputs=inputs),
        "absolute_engine_timepoints_complete": bool(requests) and all(r["timepoint_validation"]["status"] == "verified" for r in requests)}
    if diagnostic_events:
        prediction["_diagnostic_events"] = diagnostic_event_trace(result, diagnostic_event_limit)
    return prediction


def source_freeze(destination):
    refs = []
    for base in (ROOT / "src", ROOT / "tools"):
        for source in sorted(base.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            before = grid.file_ref(source)
            shutil.copy2(source, target)
            after = grid.file_ref(target)
            if before["sha256"] != after["sha256"]:
                raise ValueError("source changed during freeze")
            refs.append(after)
    return {"root": str(destination), "sha256": grid.stable_hash(refs), "files": refs}


def verify_refs(refs):
    for ref in refs:
        if grid.file_ref(ref["path"])["sha256"] != ref["sha256"]:
            raise ValueError("frozen reference changed: " + ref["path"])



def verified_model_snapshot_map(rows, requested_map, data_root):
    """Verify each explicitly requested reading copy once before the freeze."""
    if requested_map is None:
        return {}
    if not isinstance(requested_map, dict):
        raise ValueError("model snapshot map must map native paths to copy paths")
    native_models = {}
    for row in rows:
        ref = row["model_ref"]
        native_path = str(grid.resolve_data(ref["path"], data_root))
        if native_path in native_models and native_models[native_path] != ref["sha256"]:
            raise ValueError("selection assigns conflicting SHA256 identities to a native model path")
        native_models[native_path] = ref["sha256"]
    result, copies = {}, {}
    for original, copy_path in requested_map.items():
        if not isinstance(original, str) or not isinstance(copy_path, str):
            raise ValueError("model snapshot map paths must be strings")
        native_path = str(grid.resolve_data(original, data_root))
        if native_path not in native_models:
            raise ValueError("model snapshot map contains a model outside frozen selection: " + native_path)
        source = str(grid.resolve_data(copy_path, data_root))
        if source not in copies:
            copies[source] = grid.file_ref(source)
        reference = copies[source]
        expected = native_models[native_path]
        if reference["sha256"] != expected:
            raise ValueError(f"model snapshot SHA256 mismatch: native_path={native_path}; prediction_path={source}; expected={expected}; actual={reference['sha256']}")
        result[native_path] = dict(reference)
    return result


def freeze_selection(selection_path, output, *, data_root=None, model_snapshot_map=None, runtime_build_audit_path=None, recurrent_batching_contract_path=None, iq_panel_source_contract_path=None, iq_panel_assume_default_unset=False, slot_order_contract_path=None, host_offload_source_contract_path=None, tensor_storage_contract_path=None, tensor_storage_f32_hidden=False, gpu_invocation_contract_path=None, gpu_mmq_source_costs=False, gpu_conversion_cta_costs=False, sampling_contract_path=None, nonflash_kv_view_source_contract_path=None, mmvq_vector_issue_bound=False, mmvq_issue_hardware_document_path=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite/mix prediction campaign")
    if type(mmvq_vector_issue_bound) is not bool:
        raise ValueError("MMVQ issue switch must be an explicit boolean")
    if mmvq_vector_issue_bound and (gpu_invocation_contract_path is None or gpu_mmq_source_costs is not True or gpu_conversion_cta_costs is not True):
        raise ValueError("MMVQ issue bound requires GPU invocation, MMQ and conversion source costs")
    if mmvq_issue_hardware_document_path is not None and not mmvq_vector_issue_bound:
        raise ValueError("MMVQ hardware document requires the explicit issue-bound switch")
    selection, selection_ref = grid.read_document(selection_path)
    data_root = Path(data_root or selection.get("data_root") or ROOT).resolve(strict=True)
    rows = selected_rows(selection)
    sampling = None
    if sampling_contract_path is not None:
        from tools.native_sampling_contract import verify_sampling_contract
        sampling = verify_sampling_contract(sampling_contract_path, rows, selection_ref, data_root)
    snapshots = verified_model_snapshot_map(rows, model_snapshot_map, data_root)
    build_audit = verified_runtime_build_audit(runtime_build_audit_path, data_root) if runtime_build_audit_path else None
    recurrent = verified_recurrent_batching_contract(recurrent_batching_contract_path, rows, data_root) if recurrent_batching_contract_path else None
    slot_order = verified_slot_order_contract(slot_order_contract_path, rows, data_root, source_chain_contract_path=recurrent_batching_contract_path) if slot_order_contract_path else None
    host_offload = verified_host_offload_source_contract(host_offload_source_contract_path, rows, data_root) if host_offload_source_contract_path else None
    nonflash_kv_view = verified_nonflash_kv_view_contract(nonflash_kv_view_source_contract_path, rows, data_root,
        runtime_binding=host_offload, runtime_source_contract_path=host_offload_source_contract_path) if nonflash_kv_view_source_contract_path else None
    if type(tensor_storage_f32_hidden) is not bool or (tensor_storage_f32_hidden and tensor_storage_contract_path is None):
        raise ValueError("tensor-storage F32 hidden treatment requires its explicit source contract and boolean flag")
    tensor_storage = verified_tensor_storage_contract(tensor_storage_contract_path, rows, data_root,
        f32_hidden_storage=tensor_storage_f32_hidden, runtime_binding=host_offload,
        runtime_source_contract_path=host_offload_source_contract_path) if tensor_storage_contract_path else None
    if type(gpu_mmq_source_costs) is not bool or (gpu_mmq_source_costs and gpu_invocation_contract_path is None):
        raise ValueError("GPU MMQ source costs require a GPU invocation source contract and explicit boolean")
    gpu_invocation = verified_gpu_invocation_contract(gpu_invocation_contract_path, rows, data_root,
        enable_mmq_source_costs=gpu_mmq_source_costs, runtime_binding=host_offload,
        runtime_source_contract_path=host_offload_source_contract_path) if gpu_invocation_contract_path else None
    if type(gpu_conversion_cta_costs) is not bool or (gpu_conversion_cta_costs and not gpu_mmq_source_costs):
        raise ValueError("conversion CTA source costs require source MMQ/MMVQ dispatch")
    if gpu_invocation is not None:
        gpu_invocation["conversion_cta_costs_requested"] = gpu_conversion_cta_costs
        for evidence in gpu_invocation["cells"].values():
            evidence["conversion_cta_costs_requested"] = gpu_conversion_cta_costs
    mmvq_issue = verified_mmvq_issue_binding(rows, gpu_invocation, output,
        document_path=mmvq_issue_hardware_document_path, data_root=data_root) if mmvq_vector_issue_bound else None
    if iq_panel_assume_default_unset and iq_panel_source_contract_path is None:
        raise ValueError("IQ panel default-unset assumption requires an explicit source contract")
    iq_panel = verified_iq_panel_source_contract(iq_panel_source_contract_path, rows, data_root,
        assume_default_unset=iq_panel_assume_default_unset) if iq_panel_source_contract_path else None
    if iq_panel is not None:
        from heterollm_sim import cost_models
        if not hasattr(cost_models, "CPUIQPanelDispatch") or "iq_panel_dispatch" not in inspect.signature(cost_models.estimate_cpu_gemm).parameters:
            raise ValueError("IQ panel source treatment requires the opt-in CPU cost/planner implementation")
    if build_audit:
        for row in rows:
            compiled_graph_evidence(row["native_runtime_refs"], data_root, verified_audit=build_audit)
    output.mkdir(parents=True, exist_ok=True)
    source = source_freeze(output / "source")
    entries = []
    for row in rows:
        error, inputs = None, None
        try:
            inputs = static_inputs(row, selection, data_root, model_snapshot_map=snapshots, runtime_build_audit=build_audit, recurrent_batching=recurrent, iq_panel=iq_panel, slot_order=slot_order, host_offload=host_offload, tensor_storage=tensor_storage, gpu_invocation=gpu_invocation, sampling=sampling, nonflash_kv_view=nonflash_kv_view, mmvq_issue=mmvq_issue)
            configuration(inputs)
            gpu_clock(inputs)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        entries.append({"cell_id": row.get("cell_id", row.get("id")), "model_key": (inputs or row).get("model_key"),
            "deployment": (inputs or row).get("deployment", row.get("deployment_key")), "static_inputs": inputs, "preparation_error": error})
    freeze = {"schema": "stable-native-simulation-freeze/v1", "created_utc": now(), "selection_ref": selection_ref,
        "selection_sha256": selection_ref["sha256"], "selection_created_utc": selection.get("created_utc"),
        "selected_denominator": len(entries), "native_grid_denominator": selection.get("native_grid_denominator", selection.get("planned_cells", 162)),
        "evaluation_type": "development_post_selection", "blind_evaluation": False, "calibration_applied": False,
        "source": source, "data_root": str(data_root), "model_snapshot_map": snapshots, "runtime_build_audit": build_audit, "recurrent_batching": recurrent, "cpu_iq_panel_reuse": iq_panel, "slot_order": slot_order, "host_offload_source": host_offload, "tensor_storage": tensor_storage, "gpu_invocation": gpu_invocation, "sampling": sampling, "nonflash_kv_view": nonflash_kv_view,
        **({"mmvq_vector_issue_bound": True, "mmvq_issue_bound": mmvq_issue} if mmvq_issue is not None else {}),
        "coverage": selection["coverage"], "cells": entries}
    grid.write_new(output / "freeze.json", freeze)
    verify_refs([selection_ref, *source["files"]])
    if build_audit:
        verify_refs([build_audit["audit_ref"], *build_audit["evidence_refs"]])
    if recurrent:
        verify_refs(recurrent["evidence_refs"])
    if iq_panel:
        verify_refs(iq_panel["evidence_refs"])
    if slot_order:
        verify_refs(slot_order["evidence_refs"])
    if host_offload:
        verify_refs(host_offload["evidence_refs"])
    if tensor_storage:
        verify_refs(tensor_storage["evidence_refs"])
    if gpu_invocation:
        verify_refs(gpu_invocation["evidence_refs"])
    if mmvq_issue:
        verify_refs(mmvq_issue["evidence_refs"])
    if nonflash_kv_view:
        verify_refs(nonflash_kv_view["evidence_refs"])
    if sampling:
        verify_refs(sampling["evidence_refs"])
    return freeze



def verify_freeze_references(freeze):
    refs = [freeze["selection_ref"], *freeze["source"]["files"]]
    build = freeze.get("runtime_build_audit")
    if build:
        refs += [build["audit_ref"], *build["evidence_refs"]]
    recurrent = freeze.get("recurrent_batching")
    if recurrent:
        refs += recurrent["evidence_refs"]
    iq_panel = freeze.get("cpu_iq_panel_reuse")
    if iq_panel:
        refs += iq_panel["evidence_refs"]
    slot_order = freeze.get("slot_order")
    if slot_order:
        refs += slot_order["evidence_refs"]
    host_offload = freeze.get("host_offload_source")
    if host_offload:
        refs += host_offload["evidence_refs"]
    tensor_storage = freeze.get("tensor_storage")
    if tensor_storage:
        refs += tensor_storage["evidence_refs"]
    gpu_invocation = freeze.get("gpu_invocation")
    if gpu_invocation:
        refs += gpu_invocation["evidence_refs"]
    nonflash_kv_view = freeze.get("nonflash_kv_view")
    if nonflash_kv_view:
        refs += nonflash_kv_view["evidence_refs"]
    sampling = freeze.get("sampling")
    if sampling:
        refs += sampling["evidence_refs"]
    verify_refs(refs)
    verify_mmvq_freeze_binding(freeze)


def verify_mmvq_freeze_binding(freeze, entry=None):
    binding = freeze.get("mmvq_issue_bound")
    selected = [entry] if entry is not None else freeze["cells"]
    if binding is None:
        if freeze.get("mmvq_vector_issue_bound", False) is not False or any(
                (cell.get("static_inputs") or {}).get("mmvq_issue_evidence") is not None
                or (cell.get("static_inputs") or {}).get("mmvq_vector_issue_bound", False) is not False
                for cell in selected):
            raise ValueError("MMVQ frozen switch has no campaign evidence")
        return
    if freeze.get("mmvq_vector_issue_bound") is not True or binding.get("requested") is not True:
        raise ValueError("MMVQ campaign switch differs from frozen evidence")
    verify_mmvq_hardware_document(binding["hardware_document"])
    verify_refs(binding["evidence_refs"])
    for cell in selected:
        inputs = cell.get("static_inputs")
        if inputs is None and cell.get("preparation_error"):
            continue
        proof = binding["cells"][cell["cell_id"]]
        if (inputs.get("mmvq_vector_issue_bound") is not True or inputs.get("mmvq_issue_evidence") != proof
                or inputs.get("mmvq_issue_contract") != proof.get("contract")
                or proof.get("hardware_document") != binding["hardware_document"]):
            raise ValueError("MMVQ cell switch or proof differs from frozen campaign")


def failure(entry, reason):
    inputs = entry.get("static_inputs")
    parallel = (inputs or {}).get("config", {}).get("parallel", 1)
    parallel = parallel if type(parallel) is int and parallel > 0 else 1
    return {"status": "failed", "reason": reason, "prediction_type": PREDICTION_TYPE,
        "native_answers_used": False, "calibration_applied": False, "formal_prediction_eligible": False,
        "unsupported_dimensions": unsupported_dimensions(inputs) if inputs else [{"dimension": "static_input_preparation", "status": "unsupported", "reason": reason}],
        "input_identity": inputs, "requests": [{"request_index": i, **dict.fromkeys(METRICS)} for i in range(parallel)], "aggregate": grid.aggregate([], parallel)}


def prediction_document(entry, prediction, freeze, freeze_ref, started):
    return {"schema": "stable-native-cell-prediction/v1", "cell_id": entry["cell_id"], "model_key": entry.get("model_key"), "deployment": entry.get("deployment"),
        "created_utc": started, "finished_utc": now(), "freeze_ref": freeze_ref, "selection_sha256": freeze["selection_sha256"], "source_sha256": freeze["source"]["sha256"], **prediction}


def worker_cell(freeze_path, cell_id, result_path, *, diagnostic_events=False, diagnostic_event_limit=DIAGNOSTIC_EVENT_LIMIT):
    freeze, freeze_ref = grid.read_document(freeze_path)
    entry = next(e for e in freeze["cells"] if e["cell_id"] == cell_id)
    started = now()
    try:
        if entry.get("preparation_error"):
            raise ValueError(entry["preparation_error"])
        inputs = entry["static_inputs"]
        verify_mmvq_freeze_binding(freeze, entry)
        verify_refs([inputs["runtime_ref"], *inputs.get("runtime_module_refs", [])])
        prediction = predict_cell(inputs, diagnostic_events=diagnostic_events, diagnostic_event_limit=diagnostic_event_limit)
    except Exception as exc:
        prediction = failure(entry, type(exc).__name__ + ": " + str(exc))
    trace = prediction.pop("_diagnostic_events", None)
    if trace is not None:
        trace_path = Path(result_path).with_name(entry["cell_id"] + ".diagnostic-events.json")
        prediction["diagnostic_events_ref"] = grid.write_new(trace_path, {
            "schema": "stable-native-simulation-diagnostic-events/v1", "cell_id": cell_id,
            "freeze_ref": freeze_ref, "selection_sha256": freeze["selection_sha256"], **trace})
    prediction["diagnostic_events_requested"] = diagnostic_events
    grid.write_new(result_path, prediction_document(entry, prediction, freeze, freeze_ref, started))


def run_predictions(output, *, timeout_seconds=600, resume=False, max_cells=None, workers=4, cell_ids=None, diagnostic_events=False, diagnostic_event_limit=DIAGNOSTIC_EVENT_LIMIT):
    """One coordinator owns a bounded queue of independent, disposable workers."""
    workers = integer(workers, "workers")
    integer(diagnostic_event_limit, "diagnostic event limit")
    if diagnostic_event_limit > 20000:
        raise ValueError("diagnostic event limit must not exceed 20000")
    if workers > 8:
        raise ValueError("workers must not exceed 8")
    timeout_seconds = positive(timeout_seconds, "per-cell timeout")
    if max_cells is not None:
        integer(max_cells, "max_cells")
    output = Path(output).resolve(strict=True)
    freeze_path = output / "freeze.json"
    freeze, freeze_ref = grid.read_document(freeze_path)
    verify_freeze_references(freeze)
    requested_ids = None
    if cell_ids is not None:
        if not isinstance(cell_ids, (list, tuple)) or any(not isinstance(cell, str) for cell in cell_ids):
            raise ValueError("cell_ids must be a list of frozen cell IDs")
        requested_ids = list(dict.fromkeys(cell_ids))
        unknown = set(requested_ids) - {entry["cell_id"] for entry in freeze["cells"]}
        if unknown:
            raise ValueError("cell-id is outside frozen selection: " + ", ".join(sorted(unknown)))
    result_dir, run_dir = output / "predictions", output / "runs"
    result_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)
    executable = Path(freeze["source"]["root"]) / "tools/predict_stable_native_dataset.py"
    lock_path = run_dir / "coordinator.lock"
    # This transient coordination lock is separate from immutable inputs/results.
    # It is intentionally never stolen: an interrupted parent must first stop
    # its child processes before the stale lock can be removed explicitly.
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    try:
        pending, retained = [], []
        for entry in freeze["cells"]:
            path = result_dir / (entry["cell_id"] + ".prediction.json")
            if path.exists():
                if not resume:
                    raise FileExistsError("existing cell output requires --resume")
                document, reference = grid.read_document(path)
                if document.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]:
                    raise ValueError("existing prediction belongs to another freeze")
                retained.append(reference)
            elif requested_ids is None or entry["cell_id"] in requested_ids:
                pending.append(entry)
        scheduled = pending if max_cells is None else pending[:max_cells]
        run_number = len(list(run_dir.glob("run.*.start.json"))) + 1
        run_id = f"run.{run_number:04d}"
        receipt = {"schema": "stable-native-prediction-run/v1", "phase": "start",
            "created_utc": now(), "run_id": run_id, "freeze_ref": freeze_ref,
            "source_sha256": freeze["source"]["sha256"], "selection_sha256": freeze["selection_sha256"],
            "execution_budget": {"workers": workers, "maximum_workers": 8,
                "per_cell_timeout_seconds": timeout_seconds, "max_cells": max_cells,
                "scheduled_cells": len(scheduled),
                "nominal_timeout_rounds_seconds": math.ceil(len(scheduled) / workers) * timeout_seconds},
            "resume": resume, "cell_id_filter": requested_ids,
            "diagnostics": {"events": diagnostic_events, "event_limit": diagnostic_event_limit, "event_max_bytes": DIAGNOSTIC_EVENT_MAX_BYTES}, "retained_prediction_refs": retained,
            "scheduled_cell_ids": [entry["cell_id"] for entry in scheduled],
            "selected_denominator": freeze["selected_denominator"],
            "python_executable": sys.executable, "coordinator_pid": os.getpid()}
        start_ref = grid.write_new(run_dir / (run_id + ".start.json"), receipt)

        def execute(entry):
            result_path = result_dir / (entry["cell_id"] + ".prediction.json")
            started = now()
            log_path = result_dir / (entry["cell_id"] + "." + run_id + ".worker.log")
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    command = [sys.executable, str(executable), "--worker-freeze", str(freeze_path),
                        "--worker-cell", entry["cell_id"], "--worker-result", str(result_path)]
                    if diagnostic_events:
                        command += ["--diagnostic-events", "--diagnostic-event-limit", str(diagnostic_event_limit)]
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False)
                if completed.returncode or not result_path.exists():
                    raise RuntimeError(f"worker exited {completed.returncode}; see {log_path.name}")
            except Exception as exc:
                if not result_path.exists():
                    grid.write_new(result_path, prediction_document(entry,
                        failure(entry, type(exc).__name__ + ": " + str(exc)), freeze, freeze_ref, started))
            record, reference = grid.read_document(result_path)
            if record.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]:
                raise ValueError("worker prediction belongs to another freeze")
            return {"cell_id": entry["cell_id"], "status": record["status"], "prediction_ref": reference}

        completed_by_id = {}
        iterator = iter(scheduled)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight = {}
            for _ in range(min(workers, len(scheduled))):
                entry = next(iterator)
                in_flight[pool.submit(execute, entry)] = entry["cell_id"]
            while in_flight:
                ready, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in ready:
                    cell_id = in_flight.pop(future)
                    record = future.result()
                    completed_by_id[cell_id] = record
                    print(f"[{len(completed_by_id)}/{len(scheduled)}] {cell_id}: {record['status']}", flush=True)
                    entry = next(iterator, None)
                    if entry is not None:
                        in_flight[pool.submit(execute, entry)] = entry["cell_id"]
        # Receipts/manifests use selection order, independent of completion order.
        ordered = [completed_by_id[entry["cell_id"]] for entry in scheduled]
        manifest = finish_manifest(output)
        grid.write_new(run_dir / (run_id + ".finish.json"), {
            "schema": "stable-native-prediction-run/v1", "phase": "finish", "created_utc": now(),
            "run_id": run_id, "start_ref": start_ref, "freeze_ref": freeze_ref,
            "execution_budget": receipt["execution_budget"], "cells": ordered,
            "successful_cells": sum(r["status"] == "predicted" for r in ordered),
            "failed_or_incomplete_cells": sum(r["status"] != "predicted" for r in ordered),
            "pending_cells": manifest["pending_cells"], "selected_denominator": freeze["selected_denominator"]})
        return manifest
    finally:
        lock_path.unlink(missing_ok=True)


def finish_manifest(output):
    output = Path(output)
    freeze, freeze_ref = grid.read_document(output / "freeze.json")
    verify_freeze_references(freeze)
    entries = []
    for entry in freeze["cells"]:
        path = output / "predictions" / (entry["cell_id"] + ".prediction.json")
        record, ref = grid.read_document(path) if path.exists() else ({"status": "pending"}, None)
        if ref and record.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]:
            raise ValueError("prediction freeze mismatch")
        entries.append({"cell_id": entry["cell_id"], "model_key": entry.get("model_key"), "deployment": entry.get("deployment"), "status": record["status"], "prediction_ref": ref})
    manifest = {"schema": "stable-native-prediction-manifest/v1", "created_utc": now(), "freeze_ref": freeze_ref,
        "selected_denominator": len(entries), "successful_cells": sum(e["status"] == "predicted" for e in entries),
        "failed_or_incomplete_cells": sum(e["status"] not in ("predicted", "pending") for e in entries),
        "pending_cells": sum(e["status"] == "pending" for e in entries), "coverage": freeze["coverage"], "cells": entries,
        "evaluation_type": "development_post_selection", "blind_evaluation": False, "formal_prediction_eligible": False,
        "native_answers_used_for_prediction": False, "verification": "selection and frozen execution source verified after predictions"}
    grid.write_new(output / f"manifest.{len(list(output.glob('manifest.*.json'))) + 1:04d}.json", manifest)
    return manifest


def native_run_medians(row, metric):
    """Score-side only: compare to exactly three native scenario runs."""
    alias = next(k for k, value in ALIASES.items() if value == metric)
    if isinstance(row.get("native_actuals"), list):
        groups = {}
        for actual in row["native_actuals"]:
            groups.setdefault((actual["block"], actual["repeat"]), []).append(actual)
        if len(groups) != 3:
            raise ValueError("exactly three native run groups required")
        values = []
        for actuals in groups.values():
            if len(actuals) != row["parallel"] or {a["request_index"] for a in actuals} != set(range(row["parallel"])):
                raise ValueError("native request coverage is incomplete")
            values.append(statistics.median(positive(a["metrics_ms"][alias], metric) for a in actuals))
        median = statistics.median(values)
        declared = row.get("metrics", {}).get(alias, {}).get("native_median_ms")
        if declared is not None and not math.isclose(median, declared, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("native run medians disagree with selected summary")
        return median, values
    runs = row.get("native_runs", row.get("runs"))
    if isinstance(runs, list):
        values = [run.get(metric, run.get(alias)) for run in runs]
    else:
        metrics = row.get("native_metrics", row.get("metrics", {}))
        evidence = metrics.get(metric, metrics.get(alias, {}))
        values = evidence.get("run_medians_ms", evidence.get("batch_medians_ms")) if isinstance(evidence, dict) else None
        if values is None:
            evidence = row.get("raw_summary", {}).get("metrics", {}).get(alias, {}).get("batch_medians", {})
            values = evidence.get("values_ms", evidence.get("samples_ms"))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("exactly three native scenario-run medians required for " + metric)
    return statistics.median([positive(v, metric) for v in values]), values


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def error_summary(rows):
    summary = {}
    for metric in METRICS:
        valid = [r["metrics"][metric] for r in rows if r.get("metrics", {}).get(metric, {}).get("status") == "scored"]
        data = {"selected_cells": len(rows), "scored_cells": len(valid), "missing_cells": len(rows) - len(valid)}
        for name in ("signed_error_pct", "absolute_percentage_error_pct", "absolute_error_ms", "signed_error_ms"):
            values = [r[name] for r in valid]
            data[name] = {"median": statistics.median(values) if values else None, "p90": percentile(values, .9), "max": max(values) if values else None}
        summary[metric] = data
    return summary



def cohort_engine_comparison(prediction, native_row):
    """Score-only native engine-clock spans; never compare unrelated epochs."""
    actuals = (native_row or {}).get("native_actuals", [])
    groups = {}
    for actual in actuals:
        groups.setdefault((actual.get("block"), actual.get("repeat")), []).append(actual)
    native_runs = []
    for key, records in sorted(groups.items(), key=lambda pair: tuple(str(value) for value in pair[0])):
        rows = []
        for record in records:
            row = {"request_id": str(record.get("request_index"))}
            for target, source in (("engine_request_begin_ns", "engine_request_begin_us"),
                    ("engine_first_token_ns", "engine_first_token_us"), ("engine_last_token_ns", "engine_last_token_us")):
                value = finite_time(record.get(source))
                row[target] = value * 1000 if value is not None else None
            rows.append(row)
        coverage = len(rows) == (native_row or {}).get("parallel") and len({r["request_id"] for r in rows}) == len(rows)
        timeline = engine_cohort_span(rows) if coverage else {"status": "incomplete", "cohort_engine_span_ms": None}
        native_runs.append({"block": key[0], "repeat": key[1], "timeline": timeline, "requests": rows})
    spans = [run["timeline"]["cohort_engine_span_ms"] for run in native_runs if run["timeline"]["status"] == "complete"]
    native_median = statistics.median(spans) if len(native_runs) == len(spans) == 3 else None
    simulator = prediction.get("cohort_engine_timeline", {})
    sim_span = simulator.get("cohort_engine_span_ms") if simulator.get("status") == "complete" else None
    result = {"status": "unavailable", "native_runs": native_runs, "native_run_engine_spans_ms": spans,
        "native_median_cohort_engine_span_ms": native_median, "simulator_cohort_engine_span_ms": sim_span,
        "definition": "max(engine_last_token)-min(engine_request_begin) within one scenario run; native uses engine microseconds only, not client E2E",
        "clock_semantics": "Absolute epochs differ; only per-run engine spans are compared", "primary_metrics_changed": False}
    if native_median is not None and native_median > 0 and finite_time(sim_span) is not None:
        signed = sim_span - native_median
        result.update(status="scored", signed_error_ms=signed, absolute_error_ms=abs(signed),
            signed_error_pct=100 * signed / native_median, absolute_percentage_error_pct=100 * abs(signed) / native_median)
    return result


def score_predictions(output, *, native_report=None):
    """Independent actual comparison; never invokes or alters predictions."""
    output = Path(output).resolve(strict=True)
    freeze, freeze_ref = grid.read_document(output / "freeze.json")
    # A second file is permitted only when it is byte-identical to the frozen
    # selection; matching cell IDs never authorize replacement ground truth.
    native_report = native_report or freeze["selection_ref"]["path"]
    native, native_ref = grid.read_document(native_report, freeze["selection_sha256"])
    actual_rows = selected_rows(native)
    actual = {r["cell_id"]: r for r in actual_rows}
    rows = []
    for entry in freeze["cells"]:
        ident = entry["cell_id"]
        path = output / "predictions" / (ident + ".prediction.json")
        prediction, prediction_ref = grid.read_document(path) if path.exists() else ({}, None)
        if prediction and prediction.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]:
            raise ValueError("prediction freeze mismatch at scoring")
        scored = {"cell_id": ident, "model_key": entry.get("model_key"), "deployment": entry.get("deployment"), "prediction_ref": prediction_ref, "metrics": {}}
        for metric in METRICS:
            try:
                if prediction.get("status") != "predicted":
                    raise ValueError("prediction unavailable or incomplete")
                value = positive(prediction["aggregate"][metric]["median_ms"], "simulator median")
                measured, run_values = native_run_medians(actual[ident], metric)
                signed = value - measured
                scored["metrics"][metric] = {"status": "scored", "simulator_median_ms": value, "native_median_ms": measured,
                    "native_run_medians_ms": run_values, "signed_error_ms": signed, "absolute_error_ms": abs(signed),
                    "signed_error_pct": 100 * signed / measured, "absolute_percentage_error_pct": 100 * abs(signed) / measured}
            except (ValueError, KeyError, TypeError) as exc:
                scored["metrics"][metric] = {"status": "unscored", "reason": str(exc)}
        scored["cohort_engine_span_diagnostic"] = cohort_engine_comparison(prediction, actual.get(ident))
        rows.append(scored)
    coverage = freeze["coverage"]
    grouped = {group["model_key"]: [] for group in coverage}
    by_model = {group["model_key"]: [] for group in coverage}
    by_deployment = {}
    for row in rows:
        grouped[row["model_key"]].append(row)
        by_model[row["model_key"]].append(row)
        by_deployment.setdefault(str(row["deployment"]), []).append(row)
    group_reports = {}
    for group in coverage:
        name = group["model_key"]
        group_reports[name] = {"model_key": name, "placement_group": group.get("placement_group", name),
            "native_grid_planned_cells": group["planned_cells"],
            "native_selected_cells": group["selected_cells"],
            "native_excluded_cells": group.get("excluded_cells", group["planned_cells"] - group["selected_cells"]),
            "deployment_configurations": sorted({row["deployment"] for row in grouped[name]}),
            **error_summary(grouped[name])}
    report = {"schema": "stable-native-simulation-errors/v1", "created_utc": now(), "evaluation_type": "development_post_selection",
        "blind_evaluation": False, "formal_prediction_eligible": False, "calibration_applied": False,
        "comparison_definition": "Median simulated request timing versus median of three native per-run scenario medians; signed error = sim - native; percentile uses linear interpolation.",
        "freeze_ref": freeze_ref, "native_report_ref": native_ref, "selected_denominator": len(rows), "overall": error_summary(rows),
        "coverage": coverage, "by_model_deployment": group_reports,
        "by_model": {k: error_summary(group) for k, group in by_model.items()},
        "by_deployment": {k: error_summary(group) for k, group in by_deployment.items()}, "cells": rows}
    verify_freeze_references(freeze)
    grid.write_new(output / f"errors.{len(list(output.glob('errors.*.json'))) + 1:04d}.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runtime-build-audit", type=Path, help="verified annotation/native CUDA build chain for a new freeze")
    parser.add_argument("--recurrent-batching-contract", type=Path, help="explicit source-derived hybrid batching treatment for a new freeze; baseline default is none")
    parser.add_argument("--slot-order-contract", type=Path, help="source-bound stable slot traversal for a qualified fresh same-arrival cohort; default off")
    parser.add_argument("--host-offload-source-contract", type=Path, help="verified source/build/native-runtime binding for host MUL_MAT CUDA dispatch; initial freeze only, default off")
    parser.add_argument("--tensor-storage-contract", type=Path, help="source/build-bound indexed GET_ROWS storage traffic; initial freeze only, default off")
    parser.add_argument("--sampling-contract", type=Path, help="source/config-bound native CPU sampling policy; initial freeze only, default off")
    parser.add_argument("--gpu-invocation-contract", type=Path, help="conditional source/build-bound physical GPU projection and fusion mapping; initial freeze only, default off")
    parser.add_argument("--gpu-mmq-source-costs", action=argparse.BooleanOptionalAction, default=None,
        help="separate source MMQ/MMVQ cost treatment; requires GPU invocation contract; default false")
    parser.add_argument("--nonflash-kv-view-source-contract", type=Path,
        help="source/build-bound non-Flash physical KV view lower bound; initial freeze only; default off")
    parser.add_argument("--gpu-conversion-cta-costs", action=argparse.BooleanOptionalAction, default=None,
        help="source conversion grid compute-resource cap; initial freeze only; requires GPU MMQ source costs")
    parser.add_argument("--mmvq-vector-issue-bound", action=argparse.BooleanOptionalAction, default=None,
        help="conditional source/PTX integer issue lower bound; initial freeze only; requires MMQ and conversion source costs")
    parser.add_argument("--mmvq-issue-hardware-document", type=Path,
        help="local official NVIDIA PDF to verify and freeze; otherwise fetch official bytes; issue-bound switch required")
    parser.add_argument("--tensor-storage-f32-hidden", action=argparse.BooleanOptionalAction, default=None,
        help="separate full F32 hidden-storage ablation; requires tensor-storage contract; default false")
    parser.add_argument("--iq-panel-source-contract", type=Path, help="explicit frozen CPU IQ panel source/build/history contract; default off")
    parser.add_argument("--iq-panel-assume-default-unset", action="store_true", help="explicit conditional ablation for unknown GGML_NO_IQ_PANEL; never proves native dispatch")
    parser.add_argument("--model-snapshot-map", type=Path, help="JSON object mapping native model paths to byte-identical prediction copy paths; initial freeze only")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--cell-id", action="append", help="run only these frozen IDs this time; repeat for multiple cells")
    parser.add_argument("--workers", type=int, default=4, help="independent cell processes; 1..8, default 4")
    parser.add_argument("--diagnostic-events", action="store_true")
    parser.add_argument("--diagnostic-event-limit", type=int, default=DIAGNOSTIC_EVENT_LIMIT)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--worker-freeze", type=Path)
    parser.add_argument("--worker-cell")
    parser.add_argument("--worker-result", type=Path)
    args = parser.parse_args(argv)
    if args.worker_freeze:
        worker_cell(args.worker_freeze, args.worker_cell, args.worker_result, diagnostic_events=args.diagnostic_events, diagnostic_event_limit=args.diagnostic_event_limit)
        return
    if not args.output:
        parser.error("--output required")
    if (args.mmvq_vector_issue_bound is not None or args.mmvq_issue_hardware_document or args.nonflash_kv_view_source_contract or args.sampling_contract or args.model_snapshot_map or args.runtime_build_audit or args.recurrent_batching_contract or args.iq_panel_source_contract or args.iq_panel_assume_default_unset or args.slot_order_contract or args.host_offload_source_contract or args.tensor_storage_contract or args.tensor_storage_f32_hidden is not None or args.gpu_invocation_contract or args.gpu_mmq_source_costs is not None or args.gpu_conversion_cta_costs is not None) and (not args.selection or args.resume):
        parser.error("model snapshots/build audit/recurrent/IQ panel/slot-order/host-offload/tensor-storage/GPU invocation contracts are only accepted for an initial selection freeze")
    if args.mmvq_vector_issue_bound and (args.gpu_invocation_contract is None or args.gpu_mmq_source_costs is not True or args.gpu_conversion_cta_costs is not True):
        parser.error("--mmvq-vector-issue-bound requires GPU invocation, MMQ and conversion source costs")
    if args.mmvq_issue_hardware_document and not args.mmvq_vector_issue_bound:
        parser.error("--mmvq-issue-hardware-document requires --mmvq-vector-issue-bound")
    if args.gpu_mmq_source_costs is not None and args.gpu_invocation_contract is None:
        parser.error("--gpu-mmq-source-costs requires --gpu-invocation-contract")
    if args.gpu_conversion_cta_costs and args.gpu_mmq_source_costs is not True:
        parser.error("--gpu-conversion-cta-costs requires --gpu-mmq-source-costs")
    if args.tensor_storage_f32_hidden is not None and args.tensor_storage_contract is None:
        parser.error("--tensor-storage-f32-hidden requires --tensor-storage-contract")
    if args.selection and not args.resume:
        snapshots = grid.read_document(args.model_snapshot_map)[0] if args.model_snapshot_map else None
        freeze_selection(args.selection, args.output, data_root=args.data_root, model_snapshot_map=snapshots, runtime_build_audit_path=args.runtime_build_audit, recurrent_batching_contract_path=args.recurrent_batching_contract, iq_panel_source_contract_path=args.iq_panel_source_contract, iq_panel_assume_default_unset=args.iq_panel_assume_default_unset, slot_order_contract_path=args.slot_order_contract, host_offload_source_contract_path=args.host_offload_source_contract, tensor_storage_contract_path=args.tensor_storage_contract, tensor_storage_f32_hidden=bool(args.tensor_storage_f32_hidden), gpu_invocation_contract_path=args.gpu_invocation_contract, gpu_mmq_source_costs=bool(args.gpu_mmq_source_costs), gpu_conversion_cta_costs=bool(args.gpu_conversion_cta_costs), sampling_contract_path=args.sampling_contract, nonflash_kv_view_source_contract_path=args.nonflash_kv_view_source_contract, mmvq_vector_issue_bound=bool(args.mmvq_vector_issue_bound), mmvq_issue_hardware_document_path=args.mmvq_issue_hardware_document)
    elif not (args.output / "freeze.json").is_file():
        parser.error("--selection required for initial freeze")
    if not args.freeze_only and not args.score:
        run_predictions(args.output, timeout_seconds=positive(args.timeout_seconds, "timeout"), resume=args.resume, max_cells=args.max_cells, workers=args.workers, cell_ids=args.cell_id, diagnostic_events=args.diagnostic_events, diagnostic_event_limit=args.diagnostic_event_limit)
    if args.score:
        score_predictions(args.output, native_report=args.native_report)


if __name__ == "__main__":
    main()
