"""Optional frozen final-output binding; zero new cost coefficients.

Consume only already-verified runtime/sampling identities and static source
files. Never read native timing records, load a contract's exporter, or infer
an architecture from a model display name. Architecture include body history
remains conditional even when the inherited OBJ/DLL identities are verified.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import hashlib
import re

from heterollm_sim.final_layer_output_selection import (
    SOURCE_KEY, source_declaration, resolve_declaration, model_declaration,
)
from heterollm_sim.serde import stable_hash

SCHEMA = "heterollm.final-output-binding/v1"
FLAG = "final_output_selection"
INPUT_KEY = "final_output_selection_binding"
AUDIT_KEY = "llama_cpp_final_output_binding"
# Current bytes independently reviewed in round_024/final_layer_binding_audit.
# These identify source semantics, not historical compilation of include bodies.
REVIEWED_SOURCES = {
    "qwen2": "1a9583580e9d753d60cfa45e5697a243d225aeaa0bf2d834946723aef0516287",
    "llama": "b005c4aaf126b4dd86c65ccbd39a978deb44a962d4539577376092aee1442503",
    "qwen35": "c277c0fbef003f268f4ebd539e08afbf3bbd539aa713f29c33319960d98abf4c",
    "graph": "a6a8241c2d149961801d0fdeaa68f1bb176297b4d5156b6138db0b3d169abb04",
    "model": "94ede4e7ac8119c5a4d2fad97e3432008ec7d30b42ab37395db6e2a8047d1984",
    "context": "e677c1e6e56fc08561fa56d9861501740405190e578d690e9efcad26fa48622e",
    "server_context": "99f7aead4dd6b190292db2a14b2586d4076871a3710a49a94c71424b6f05501e",
    "server_task_header": "8bed70ca9a719c82a1e83b5ea8f7423d2e863c3ce79a154df6d0ed5ca67d849e",
    "speculative": "4dbb7bee2619f2c6bc016f0304b526a4e07e3cfb8a22e664bf78ddcd8808284d",
}
# Both reviewed collectors call client.batch(base+'/completion', payload).
REVIEWED_COMPLETION_COLLECTORS = frozenset({
    "d14e77582047676f3bd9336fd5c2ac460e1eb654777b65b73027e9295a51fb6f",
    "44797d425e71fb6f2fccf011802f606b5d67a38b153aa472c69b048250c26e54",
})
GRAPH_ARCHITECTURES = {"qwen2": {"qwen2", "qwen2_decoder"},
                       "llama": {"llama", "llama_decoder"},
                       "qwen35": {"qwen3_5_hybrid_transformer"}}
CONFIG_KEYS = ("batch", "ubatch", "parallel", "ctx", "gpu_layers", "op_offload",
               "flash_attn", "seed", "output", "expected_prompt_tokens")
LIMITATIONS = [
    "Historical architecture Unity include body hashes are unavailable; source-to-inherited-object binding is conditional.",
    "Current reviewed source bytes plus fixed runtime lineage do not prove native GET_ROWS dispatch or physical placement.",
    "Completion parameters follow reviewed collector/server branches and verified effective non-speculative CPU sampling; no live runtime parameter probe.",
    "Existing row-selection cost models only; no fitted timing or new coefficients.",
]


def require(condition, message):
    if not condition:
        raise ValueError("final output binding: " + message)


def seal(value):
    value = dict(value)
    value["content_sha256"] = stable_hash(value)
    return value


def checked(value):
    require(isinstance(value, Mapping), "binding must be a mapping")
    raw = {k: v for k, v in value.items() if k != "content_sha256"}
    require(value.get("content_sha256") == stable_hash(raw), "content digest mismatch")
    require(value.get("schema") == SCHEMA, "unsupported schema")
    return value


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_ref(value):
    require(isinstance(value, Mapping) and isinstance(value.get("path"), str), "reference missing")
    require(isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "reference digest missing")
    require(ref(value["path"])["sha256"] == value["sha256"], "reference changed: " + value["path"])


def modules(refs):
    result = {}
    for value in refs:
        name = Path(value["path"]).name.lower()
        require(name not in result, "duplicate runtime module")
        require(re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))), "runtime digest missing")
        result[name] = value["sha256"]
    return result


def completion_qualification(sampling):
    """Project reviewed static sampling evidence; never open raw response files."""
    if not isinstance(sampling, Mapping):
        return ["verified_sampling_binding_missing"]
    effective = sampling.get("effective_settings", {})
    provenance = sampling.get("provenance", {})
    collector = provenance.get("collector_source_ref", {})
    reasons = []
    if collector.get("sha256") not in REVIEWED_COMPLETION_COLLECTORS:
        reasons.append("completion_endpoint_collector_unreviewed")
    if (sampling.get("backend_sampling") is not False or effective.get("backend_sampling") is not False
            or effective.get("speculative.types") != "none,none" or effective.get("lora") != []):
        reasons.append("ordinary_non_speculative_completion_not_qualified")
    counts = provenance.get("resolved_request_settings_checked", {})
    if (not provenance.get("native_lineage") or not isinstance(counts, Mapping)
            or not counts or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) <= 0):
        reasons.append("effective_request_or_runtime_lineage_missing")
    return reasons


def make_source_contract(runtime_binding, sampling_binding, linkage):
    """Called only after the predictor's existing verifiers have succeeded."""
    require(isinstance(runtime_binding, Mapping) and isinstance(sampling_binding, Mapping),
            "verified host-offload runtime and sampling contracts are required")
    compiled = linkage["source_compilation"]
    root = Path(compiled["model"]["source"]).parents[1]
    # Snapshot original paths from the already-verified sampling contract.
    import json
    verify_ref(sampling_binding["contract_ref"])
    sampling_document = json.loads(Path(sampling_binding["contract_ref"]["path"]).read_text(encoding="utf-8-sig"))
    server_ref = sampling_document["source_snapshots"]["server_context"]["original"]
    paths = {arch: root / "src" / "models" / (arch + ".cpp") for arch in GRAPH_ARCHITECTURES}
    paths.update({role: Path(compiled[role]["source"]) for role in ("graph", "model", "context")})
    paths.update(server_context=Path(server_ref["path"]),
                 server_task_header=root / "tools/server/server-task.h", speculative=root / "common/speculative.cpp")
    source_refs = {role: ref(path) for role, path in paths.items()}
    require(set(source_refs) == set(REVIEWED_SOURCES), "source role coverage differs")
    for role, value in source_refs.items():
        require(value["sha256"] == REVIEWED_SOURCES[role], "unreviewed source: " + role)
    runtime_refs = runtime_binding["contract"]["runtime_modules"]
    require("llama.dll" in runtime_refs, "llama runtime identity missing")
    for role in ("graph", "model", "context"):
        require(compiled[role]["module"]["sha256"] == runtime_refs["llama.dll"]["sha256"],
                "source object linked to another llama module")
        require(compiled[role]["source_sha256"] == source_refs[role]["sha256"], "compiled source identity differs")
    evidence = [runtime_binding["contract_ref"], runtime_binding["runtime_build_audit_ref"],
                runtime_binding["base_build_receipt_ref"], sampling_binding["contract_ref"],
                *linkage["evidence_refs"], *source_refs.values()]
    unique = {value["path"]: value for value in evidence}
    return seal({"schema": SCHEMA, "kind": "source_contract", "source_refs": source_refs,
        "runtime_modules": {name.lower(): value["sha256"] for name, value in runtime_refs.items()},
        "binding_inputs": {key: runtime_binding[key] for key in ("contract_ref", "runtime_build_audit_ref", "base_build_receipt_ref")},
        "sampling_contract_ref": sampling_binding["contract_ref"],
        "declaration": source_declaration(), "source_compilation": {role: compiled[role] for role in ("graph", "model", "context")},
        "evidence_refs": list(unique.values()), "historical_include_content_proven": False,
        "native_dispatch_proven": False, "qualification": "conditional", "limitations": list(LIMITATIONS)})


def rederive_source(source):
    """Recheck actual build/source identities, not a resealable status claim."""
    import json
    from tools import llama_runtime_source_binding as native
    from tools.predict_stable_native_dataset import verify_gpu_invocation_source_links
    refs = source["binding_inputs"]
    for value in (*refs.values(), source["sampling_contract_ref"]):
        verify_ref(value)
    document = json.loads(Path(refs["contract_ref"]["path"]).read_text(encoding="utf-8-sig"))
    saved = document.get("build_binding") if document.get("schema") == "llama-runtime-source-binding-validation/v1" else document
    # All references produced by existing verifiers are absolute. The source
    # root is only a resolver base; no new native/backend is selected here.
    data_root = Path(source["source_refs"]["model"]["path"]).parents[3]
    derived = native.verify_llama_runtime_source_binding(refs["runtime_build_audit_ref"]["path"],
        base_build_receipt_path=refs["base_build_receipt_ref"]["path"], data_root=data_root)
    require(json.loads(json.dumps(derived)) == saved, "runtime source binding no longer re-derives")
    runtime = {"contract": saved, **refs}
    linkage = verify_gpu_invocation_source_links(runtime, data_root, include_context=True)
    return make_source_contract(runtime, {"contract_ref": source["sampling_contract_ref"]}, linkage)


def derive_cell(source, *, cell_id, model_ref, model_scope, config, native_refs, sampling):
    checked(source)
    arch = model_scope.get("architecture")
    require(re.fullmatch(r"[0-9a-f]{64}", str(model_ref.get("sha256", ""))), "model digest missing")
    native = modules(native_refs)
    reasons = completion_qualification(sampling)
    if arch not in GRAPH_ARCHITECTURES:
        reasons.append("gguf_architecture_unsupported")
    if any(native.get(name) != sha for name, sha in source["runtime_modules"].items()):
        reasons.append("selected_runtime_identity_uncovered")
    return seal({"schema": SCHEMA, "kind": "cell_binding", "requested": True,
        "status": "uncovered" if reasons else "conditional", "reasons": reasons,
        "cell_id": cell_id, "model_sha256": model_ref["sha256"], "gguf_architecture": arch,
        "config": {key: config.get(key) for key in CONFIG_KEYS}, "runtime_modules": native,
        "sampling_sha256": stable_hash(sampling), "source_contract": source,
        "declaration": source_declaration() if not reasons else None,
        "historical_include_content_proven": False, "native_dispatch_proven": False,
        "completion_qualification": "reviewed_completion_task_need_embd_false_and_non_speculative_context_defaults" if not reasons else "uncovered",
        "limitations": list(LIMITATIONS)})


def freeze_binding(rows, *, runtime_binding, sampling_binding, source_linkage, model_scope_reader):
    source = make_source_contract(runtime_binding, sampling_binding, source_linkage)
    cache, cells = {}, {}
    for row in rows:
        mr = row["model_ref"]
        if mr["sha256"] not in cache:
            cache[mr["sha256"]] = model_scope_reader(mr)
        cells[row["cell_id"]] = derive_cell(source, cell_id=row["cell_id"], model_ref=mr,
            model_scope=cache[mr["sha256"]], config=row["config"], native_refs=row["native_runtime_refs"],
            sampling=sampling_binding["cells"].get(row["cell_id"]))
    return {"requested": True, "source_contract": source, "cells": cells,
            "evidence_refs": source["evidence_refs"], "new_cost_coefficients": 0}


def verify_sampling_projection(source, inputs):
    """Check against the freeze-verified sampler document, never its raw timings."""
    import json
    from tools.native_sampling_contract import _profile_id
    sampling = inputs.get("sampling_binding")
    if not isinstance(sampling, Mapping):
        return
    value = source["sampling_contract_ref"]
    verify_ref(value)
    document = json.loads(Path(value["path"]).read_text(encoding="utf-8-sig"))
    rows = [row for row in document.get("cells", []) if row.get("cell_id") == inputs["cell_id"]]
    require(len(rows) == 1, "sampling contract cell missing or duplicate")
    effective = sampling.get("effective_settings", {})
    profile = _profile_id(effective)
    require(rows[0].get("sampling_profile_id") == profile
            and document.get("effective_profiles", {}).get(profile) == effective
            and rows[0].get("typed_policy_candidate") == sampling.get("typed_policy")
            and rows[0].get("backend_sampling") is sampling.get("backend_sampling"),
            "sampling projection differs from freeze-verified contract")
    require(sampling.get("provenance", {}).get("selection_ref", {}).get("sha256")
            == document.get("selection_ref", {}).get("sha256"), "sampling selection identity differs")
    collector = sampling.get("provenance", {}).get("collector_source_ref")
    verify_ref(collector)


def verify_cell(inputs, *, verify_files=True):
    flag, proof = inputs.get(FLAG, False), inputs.get(INPUT_KEY)
    if flag is False and proof is None:
        return None
    require(flag is True, "switch must match frozen proof")
    checked(proof)
    source = checked(proof.get("source_contract"))
    require(source.get("kind") == "source_contract" and source.get("declaration") == source_declaration(), "source declaration differs")
    require(source.get("historical_include_content_proven") is False and source.get("native_dispatch_proven") is False,
            "source qualification cannot be upgraded")
    require(set(source.get("source_refs", {})) == set(REVIEWED_SOURCES), "source role coverage differs")
    for role, value in source["source_refs"].items():
        require(value.get("sha256") == REVIEWED_SOURCES[role], "unreviewed source: " + role)
    if verify_files:
        for value in source["evidence_refs"]:
            verify_ref(value)
        require(rederive_source(source) == source, "source contract differs from re-derived build facts")
    verify_sampling_projection(source, inputs)
    expected = derive_cell(source, cell_id=inputs["cell_id"], model_ref=inputs["prediction_model_ref"],
        model_scope={"architecture": proof.get("gguf_architecture")}, config=inputs["config"],
        native_refs=[inputs["runtime_ref"], *inputs.get("runtime_module_refs", [])], sampling=inputs.get("sampling_binding"))
    require(expected == proof, "cell proof differs from frozen static inputs")
    return proof


def apply_binding(scenario, inputs, *, gguf):
    proof = verify_cell(inputs)
    if proof is None:
        return scenario
    require(gguf.sha256 == proof["model_sha256"] and gguf.architecture == proof["gguf_architecture"], "worker GGUF identity differs")
    existing = model_declaration(scenario.model)
    require(existing is None or existing == source_declaration(), "existing declaration differs")
    reasons = list(proof["reasons"])
    identities = [owner.get("gguf_sha256") for owner in (
        scenario.model.metadata, scenario.model.metadata.get("metadata", {}),
        scenario.model.graph.attributes.get("metadata", {})) if isinstance(owner, Mapping) and "gguf_sha256" in owner]
    require(identities and all(value == gguf.sha256 for value in identities), "scenario model is not the bound GGUF")
    runtime = scenario.llama_cpp_config
    if runtime is None:
        reasons.append("typed_runtime_configuration_missing")
    else:
        cfg = proof["config"]
        if (any(getattr(runtime, key) != cfg[key] for key in ("batch", "ubatch", "parallel"))
                or runtime.context * runtime.parallel != cfg["ctx"]):
            reasons.append("scenario_runtime_shape_differs_from_frozen_configuration")
    if scenario.model.architecture not in GRAPH_ARCHITECTURES.get(gguf.architecture, ()):
        reasons.append("graph_and_gguf_architecture_mismatch")
    view = scenario.model._execution_view
    if scenario.workload.mtp is not None or view.mtp_descriptors:
        reasons.append("mtp_execution_uncovered")
    if scenario.placement.parallel.world_size != 1:
        reasons.append("multi_rank_uncovered")
    if scenario.workload.metadata.get("llama_cpp_f32_hidden_storage") is not True:
        reasons.append("source_qualified_f32_hidden_storage_missing")
    last = view.layer_instances[-1].layer
    if last.is_moe or (gguf.architecture in {"qwen2", "llama"} and last.is_linear_attention):
        reasons.append("tail_kind_uncovered")
    audit = {"requested": True, "applied": not reasons, "status": "uncovered" if reasons else "conditional",
             "reasons": reasons, "binding_sha256": proof["content_sha256"], "native_dispatch_proven": False,
             "historical_include_content_proven": False, "new_cost_coefficients": 0, "limitations": proof["limitations"]}
    workload = replace(scenario.workload, metadata={**scenario.workload.metadata, AUDIT_KEY: audit})
    if reasons:
        require(existing is None, "uncovered binding cannot retain an enabled declaration")
        return replace(scenario, workload=workload)
    resolve_declaration(proof["declaration"], scenario.model.architecture, mtp_present=False)
    # ModelSpec.metadata is the sole canonical declaration owner. No graph
    # nested mutation; downstream GPU binding and replan preserve ModelSpec.
    model = replace(scenario.model, metadata={**scenario.model.metadata, SOURCE_KEY: dict(proof["declaration"])})
    return replace(scenario, model=model, workload=workload)


def verify_freeze(freeze, *, entry=None):
    campaign = freeze.get(INPUT_KEY)
    selected = [entry] if entry is not None else freeze["cells"]
    if campaign is None:
        require(freeze.get(FLAG, False) is False and not any((cell.get("static_inputs") or {}).get(FLAG, False)
                or (cell.get("static_inputs") or {}).get(INPUT_KEY) is not None for cell in selected), "campaign evidence missing")
        return
    require(freeze.get(FLAG) is True and campaign.get("requested") is True, "campaign switch mismatch")
    source = checked(campaign["source_contract"])
    for value in campaign["evidence_refs"]:
        verify_ref(value)
    require(rederive_source(source) == source, "campaign source contract no longer re-derives")
    for cell in selected:
        inputs = cell.get("static_inputs")
        if inputs is None and cell.get("preparation_error"):
            continue
        proof = verify_cell(inputs, verify_files=False)
        require(proof == campaign["cells"][cell["cell_id"]] and proof["source_contract"] == source,
                "cell/campaign mismatch")
