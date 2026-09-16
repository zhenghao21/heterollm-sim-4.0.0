"""Freeze-time verification of the reviewed native CPU sampling contract.

Only payload/configuration and source identities are projected. Workers receive
static cells and never need to parse native results or run a contract's exporter.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

_SCHEMA = "source-bound-native-sampling-parity/v1"
_CORE_MODULES = ("llama-server.exe", "llama-server-impl.dll", "llama-common.dll", "llama.dll")
_BASELINE_SOURCES = {"sampling_policy", "planner", "matching_scenario", "static_predictor"}
# Reviewed source identities for this contract version, independent of its own
# claimed hashes. Future native implementations require a new source review.
_REVIEWED_NATIVE_SOURCE_SHA256 = {'common_defaults': '6e678df90075831853a3b509a7a3fe01e5aa4cbeb1b3ae4364c0ac51442b36c3', 'common_sampling': 'a38a6624109ce01d317d6c89f5d1ff27c1427fd5934bddcf71c77d922fcf15b2', 'common_initialization': '5e80bdf7336324a55942bd7a46aba652bf438ceebd2d14133fe5e8be455b530c', 'native_samplers': 'a2637a34869a95008c847461e6b10e5edb71ba4c6d19f99a3dbd1875d66dada8', 'native_sampler_header': 'bace72539125f3a2bcf7f71392cf7cc4ae1bd3988535fca16c04a97e531a27ed', 'server_schema': '4b419ef42556cf8dda5c0471ed9294388da613b62eeee56efa2e13ed34252238', 'server_task': '4ef9e2c0a78ee4480b795d5056a4f979f10529c89e2c3526c728e6d7f6107613', 'server_context': '99f7aead4dd6b190292db2a14b2586d4076871a3710a49a94c71424b6f05501e'}
_ORDER = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]
_REQUIRED = ("seed", "temperature", "dynatemp_range", "dynatemp_exponent", "top_k", "top_p", "min_p", "top_n_sigma", "xtc_probability", "xtc_threshold", "typical_p", "repeat_last_n", "repeat_penalty", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers", "mirostat", "mirostat_tau", "mirostat_eta", "adaptive_target", "adaptive_decay", "ignore_eos", "stream", "n_probs", "min_keep", "grammar", "grammar_lazy", "grammar_triggers", "preserved_tokens", "samplers", "speculative.types", "post_sampling_probs", "backend_sampling", "lora")
_PAYLOAD_KEYS = {"prompt", "n_predict", "ignore_eos", "cache_prompt", "temperature", "top_k", "seed", "stream"}
_CLI_PREFIXES = ("--temp", "--temperature", "--top-k", "--top-p", "--min-p", "--min-keep", "--samplers", "--sampling-seq", "--seed", "--mirostat", "--logit-bias", "--ignore-eos", "--backend-sampling", "--no-backend-sampling", "--grammar", "--repeat-penalty", "--presence-penalty", "--frequency-penalty", "--dry-", "--reasoning-budget")
_LIMITATIONS = [
    "Source-qualified sampling semantics only; no measured latency or fitted cost.",
    "Logit bias, additional vocabulary suppress tokens, chain traversal, accept/history and RNG work remain partial unless separately lowered.",
    "Neutral penalties/DRY constructors use empty samplers, not active penalty/history algorithms; common_sampler previous-token ring commit still remains.",
    "The singleton dist path still draws uniform RNG; temperature zero is not a direct greedy-sampler shortcut.",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("sampling contract: " + message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), label + " must be non-empty text")
    return value


def _nonnull(value: Any, label: str) -> None:
    _require(value is not None, label + " must not be null")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _nonnull(item, label + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _nonnull(item, label + "[" + str(index) + "]")
    elif isinstance(value, float):
        _require(math.isfinite(value), label + " must be finite")


class _Evidence:
    def __init__(self, data_root: str | Path):
        self.root = Path(data_root).resolve()
        self.checked: dict[str, dict] = {}

    def path(self, value: Any) -> Path:
        path = Path(_text(value, "reference path"))
        return (path if path.is_absolute() else self.root / path).resolve()

    def identity(self, value: Mapping[str, Any]) -> tuple[str, str]:
        _require(isinstance(value, Mapping), "file reference required")
        sha = value.get("sha256")
        _require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "non-empty SHA256 identity required")
        if "bytes" in value:
            _require(type(value["bytes"]) is int and value["bytes"] >= 0, "reference bytes must be a non-negative integer")
        return os.path.normcase(str(self.path(value.get("path")))), sha

    def same(self, left: Mapping, right: Mapping) -> bool:
        return self.identity(left) == self.identity(right)

    def verify(self, expected: Mapping[str, Any]) -> dict:
        key, sha = self.identity(expected)
        if key not in self.checked:
            path = self.path(expected["path"])
            _require(path.is_file(), "missing evidence file: " + str(path))
            with path.open("rb") as handle:
                actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            self.checked[key] = {"path": str(path), "sha256": actual_sha, "bytes": path.stat().st_size}
        actual = self.checked[key]
        _require(actual["sha256"] == sha, "evidence SHA256 mismatch: " + expected["path"])
        if "bytes" in expected:
            _require(actual["bytes"] == expected["bytes"], "evidence size mismatch: " + expected["path"])
        return dict(actual)

    def read(self, expected: Mapping) -> Any:
        key, sha = self.identity(expected)
        path = self.path(expected["path"])
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == sha, "JSON evidence SHA256 mismatch: " + str(path))
        if "bytes" in expected:
            _require(len(raw) == expected["bytes"], "JSON evidence size mismatch")
        self.checked[key] = {"path": str(path), "sha256": sha, "bytes": len(raw)}
        return json.loads(raw.decode("utf-8"))

    def capture(self, path: str | Path) -> dict:
        path = self.path(str(path))
        with path.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        return self.verify({"path": str(path), "sha256": sha, "bytes": path.stat().st_size})


def _unique(items: Sequence[Mapping], field: str, label: str) -> dict[str, Mapping]:
    _require(isinstance(items, list), label + " must be a list")
    result = {}
    for item in items:
        _require(isinstance(item, Mapping), label + " item must be an object")
        key = _text(item.get(field), label + " identity")
        _require(key not in result, "duplicate " + label + ": " + key)
        result[key] = item
    return result


def _raw_anchors(row: Mapping, evidence: _Evidence) -> dict[tuple[str, str], Mapping]:
    entries = row.get("evidence_index")
    _require(isinstance(entries, list) and bool(entries), "selected row requires raw evidence")
    result = {}
    for item in entries:
        ident = evidence.identity(item.get("raw_ref"))
        _require(ident not in result, "duplicate selected raw identity")
        _text(item.get("key"), "selected raw key")
        result[ident] = item
    return result


def _core_refs(row: Mapping, evidence: _Evidence) -> dict[str, Mapping]:
    refs = row.get("native_runtime_refs")
    _require(isinstance(refs, list), "selected native runtime refs required")
    result = {}
    for ref in refs:
        evidence.identity(ref)
        name = evidence.path(ref["path"]).name
        if name in _CORE_MODULES:
            _require(name not in result, "duplicate runtime module: " + name)
            result[name] = ref
    _require(set(result) == set(_CORE_MODULES), "complete native sampling runtime module set required")
    return result


def _hash_at(mapping: Mapping, path: str | Path, evidence: _Evidence) -> str:
    _require(isinstance(mapping, Mapping), "historical hash map required")
    key = os.path.normcase(str(evidence.path(str(path))))
    values = [value for candidate, value in mapping.items() if os.path.normcase(str(evidence.path(candidate))) == key]
    _require(len(values) == 1, "historical file identity missing: " + str(path))
    return values[0]


def _native_provenance(contract: Mapping, freezes: list[Mapping], modules: dict[str, Mapping], evidence: _Evidence) -> dict:
    anchored = {}
    for freeze in freezes:
        refs = freeze.get("artifact_refs")
        _require(isinstance(refs, list), "freeze artifact_refs required")
        for ref in refs:
            if evidence.path(ref["path"]).name == "build_receipt.json":
                value = evidence.read(ref)
                if value.get("schema") in ("llama_annotation_control_build_v1", "native_thread_control_build_v1"):
                    role = value["schema"]
                    if role in anchored:
                        _require(evidence.same(anchored[role][0], ref), "mixed native build lineage")
                    anchored[role] = (ref, value)
    _require(len(anchored) == 2, "selected freezes must anchor annotation and native build receipts")
    ar, annotation = anchored["llama_annotation_control_build_v1"]
    nr, native = anchored["native_thread_control_build_v1"]
    lineage = contract.get("build_and_header_lineage")
    _require(isinstance(lineage, Mapping), "native source/header lineage required")
    receipt_refs = lineage.get("receipts")
    _require(isinstance(receipt_refs, list), "lineage receipts required")
    expected_shas = {ar["sha256"], nr["sha256"], annotation.get("source_manifest_sha256"), native.get("source_manifest_sha256")}
    _require(None not in expected_shas and "" not in expected_shas, "historical source-manifest hashes required")
    _require(len(receipt_refs) == len(expected_shas), "complete unique source/build receipt set required")
    _require({evidence.verify(ref)["sha256"] for ref in receipt_refs} == expected_shas, "source/build receipts not bound to selected freeze")
    historical = lineage.get("historical_header_bindings")
    _require(isinstance(historical, list) and bool(historical), "historical native header bindings required")
    seen_headers = {}
    for binding in historical:
        header_ref, snapshot_ref = binding.get("source"), binding.get("snapshot_ref")
        header = evidence.verify(header_ref)
        _require(evidence.verify(snapshot_ref)["sha256"] == annotation.get("header_snapshot_sha256"), "header snapshot not bound to annotation build")
        snapshot = evidence.read(snapshot_ref)
        expected = _hash_at(snapshot.get("files"), header_ref["path"], evidence)
        _require(header["sha256"] == expected == binding.get("historical_sha256"), "historical header hash mismatch")
        basename = evidence.path(header["path"]).name
        _require(basename not in seen_headers, "duplicate historical header binding")
        seen_headers[basename] = header
    _require({"common.h", "sampling.h", "llama.h"} <= set(seen_headers), "common/sampling/llama header history required")
    chain = _unique(lineage.get("unchanged_binary_chain"), "name", "binary lineage")
    _require(set(chain) == set(_CORE_MODULES), "incomplete binary lineage")
    for name, module in modules.items():
        item = chain[name]
        _require(evidence.same(item.get("selected"), module), "selected runtime differs from lineage: " + name)
        selected = evidence.verify(module)
        _require(_hash_at(native.get("unchanged_runtime_sha256"), selected["path"], evidence) == selected["sha256"], "native unchanged-runtime receipt mismatch")
        _require(evidence.same(item.get("native_receipt_ref"), nr), "foreign native receipt in module lineage")
        ancestor = evidence.verify(item.get("annotation_ancestor"))
        expected_path = evidence.path(native.get("runtime_base")) / name
        _require(os.path.normcase(ancestor["path"]) == os.path.normcase(str(expected_path)), "wrong annotation runtime ancestor")
        _require(ancestor["sha256"] == selected["sha256"] == _hash_at(annotation.get("output_sha256"), ancestor["path"], evidence), "annotation-to-native module identity mismatch")
    sources = contract.get("source_snapshots")
    _require(isinstance(sources, Mapping), "reviewed source snapshots required")
    required_sources = _BASELINE_SOURCES | set(_REVIEWED_NATIVE_SOURCE_SHA256)
    _require(required_sources <= set(sources), "required native/simulator source snapshots missing")
    for name, pair in sources.items():
        _require(isinstance(pair, Mapping), "source snapshot pair required")
        original, snapshot = pair.get("original"), pair.get("snapshot")
        _, original_sha = evidence.identity(original)
        copied = evidence.verify(snapshot)
        _require(original_sha == copied["sha256"], "source snapshot differs from recorded original: " + name)
        if "bytes" in original:
            _require(original["bytes"] == copied["bytes"], "source snapshot size mismatch")
        if name in _REVIEWED_NATIVE_SOURCE_SHA256:
            _require(original_sha == _REVIEWED_NATIVE_SOURCE_SHA256[name], "unreviewed native source revision: " + name)
            evidence.verify(original)
        # Baseline simulator originals are intentionally changing. Only their
        # frozen copies belong in evidence_refs or pre/post/resume verification.
    _require(evidence.same(sources["common_defaults"]["original"], seen_headers["common.h"]), "default common.h is not the historical header")
    _require(evidence.same(sources["native_sampler_header"]["original"], seen_headers["llama.h"]), "native sampler header is not the historical header")
    context = sources["server_context"]["original"]
    _require(_hash_at(annotation.get("input_sha256"), context["path"], evidence) == context["sha256"], "server sample/accept source not bound to build input")
    return {"annotation_build_receipt": evidence.verify(ar), "native_build_receipt": evidence.verify(nr), "historical_headers": list(seen_headers.values()), "source_revision_qualification": "reviewed native source fingerprints plus selected build/header/runtime lineage"}


def _number(value: Any, label: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)), label + " must be finite numeric")
    return float(value)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, label + " must be an integer >= " + str(minimum))
    return value


def _project_settings(settings: Mapping, payload: Mapping) -> dict:
    _require(isinstance(settings, Mapping), "resolved generation_settings required")
    _require(set(_REQUIRED) <= set(settings), "resolved sampling fields missing")
    projected = {key: copy.deepcopy(settings[key]) for key in _REQUIRED}
    _nonnull(projected, "resolved sampling settings")
    for key in ("temperature", "dynatemp_range", "dynatemp_exponent", "top_p", "min_p", "top_n_sigma", "xtc_probability", "xtc_threshold", "typical_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base", "mirostat_tau", "mirostat_eta", "adaptive_target", "adaptive_decay"):
        _number(projected[key], key)
    for key in ("seed", "min_keep", "n_probs", "mirostat", "repeat_last_n", "dry_allowed_length", "dry_penalty_last_n"):
        _integer(projected[key], key)
    _integer(projected["top_k"], "top_k", 1)
    _require(projected["temperature"] == 0 and projected["top_k"] == 1 and projected["min_keep"] == 0, "unsupported frozen temperature/top-k/min-keep domain")
    _require(projected["seed"] < 0xffffffff, "fixed non-random-default seed required")
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    _require(projected["top_p"] == f32(.95) and projected["min_p"] == f32(.05), "unsupported native top-p/min-p defaults")
    neutral = {"dynatemp_range": 0, "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0, "dry_multiplier": 0, "xtc_probability": 0, "typical_p": 1, "mirostat": 0, "n_probs": 0}
    _require(all(projected[key] == expected for key, expected in neutral.items()), "non-neutral sampler modifier unsupported")
    _require(projected["top_n_sigma"] <= 0 and projected["adaptive_target"] < 0, "non-neutral sigma/adaptive modifier unsupported")
    _require(projected["samplers"] == _ORDER, "unreviewed sampler order")
    _require(projected["backend_sampling"] is False and projected["ignore_eos"] is True and projected["stream"] is True and projected["post_sampling_probs"] is False, "unsupported backend/EOS/probability path")
    _require(projected["grammar"] == "" and projected["grammar_lazy"] is False and projected["grammar_triggers"] == [] and projected["preserved_tokens"] == [] and projected["lora"] == [] and projected["speculative.types"] == "none,none" and settings.get("generation_prompt", "") == "", "unreviewed grammar/reasoning/speculative/LoRA modifier")
    for key in ("temperature", "top_k", "seed", "ignore_eos", "stream"):
        _require(projected[key] == payload[key], "request/resolved policy mismatch: " + key)
    biases = settings.get("logit_bias")
    _require(isinstance(biases, list), "resolved logit_bias required")
    projected["logit_bias"] = []
    for bias in biases:
        _require(isinstance(bias, Mapping) and set(bias) == {"token", "bias"}, "unsupported serialized bias record")
        token = _integer(bias["token"], "bias token")
        if bias["bias"] is None:
            # This normalization is reached only after selected source hashes,
            # historical defaults, request ignore_eos, and CLI path are checked.
            projected["logit_bias"].append({"token": token, "bias_kind": "negative_infinity", "wire_encoding": "json_null", "source_basis": "ignore_eos EOG list; common.cpp populates -INFINITY; server-schema.cpp appends it"})
        else:
            _number(bias["bias"], "logit bias")
            projected["logit_bias"].append({"token": token, "bias_kind": "finite", "value": bias["bias"], "wire_encoding": "json_number"})
    projected["generation_prompt_empty"] = True
    _nonnull(projected, "canonical sampling settings")
    return projected


def _typed(settings: Mapping) -> dict:
    return {"mode": "greedy", "temperature": settings["temperature"], "implementation": "llama_cpp_cpu_chain", "top_k": settings["top_k"], "top_p": settings["top_p"], "min_p": settings["min_p"], "min_keep": settings["min_keep"]}


def _profile_id(settings: Mapping) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def verify_sampling_contract(path, rows, selection_ref, data_root):
    """Return source-bound static cells; never execute an exporter or infer delays."""
    evidence = _Evidence(data_root)
    selection_identity = evidence.verify(selection_ref)
    selection = evidence.read(selection_ref)
    selected = _unique(selection.get("selected_cells"), "cell_id", "selected cells")
    supplied = _unique(list(rows), "cell_id", "caller rows")
    _require(set(selected) == set(supplied), "caller rows do not match selected IDs")
    contract_ref = evidence.capture(path)
    contract = evidence.read(contract_ref)
    _require(isinstance(contract, Mapping) and contract.get("schema") == _SCHEMA, "unsupported contract schema")
    _require(evidence.same(contract.get("selection_ref"), selection_identity), "foreign selection identity")
    cells = _unique(contract.get("cells"), "cell_id", "contract cells")
    _require(set(cells) == set(selected), "contract must cover exactly all selected IDs")
    profiles = contract.get("effective_profiles")
    _require(isinstance(profiles, Mapping) and bool(profiles), "effective profiles required")
    for profile_id, profile in profiles.items():
        _nonnull(profile, "claimed profile")
        _require(profile_id == _profile_id(profile), "claimed profile hash mismatch")
    frozen = {}; modules = {}; row_raw = {}
    for cell_id, row in selected.items():
        caller = supplied[cell_id]
        anchors = _raw_anchors(row, evidence)
        _require(set(anchors) == set(_raw_anchors(caller, evidence)), "caller raw refs differ from selection")
        _require(row.get("config", {}).get("seed") == caller.get("config", {}).get("seed") and row.get("config", {}).get("output") == caller.get("config", {}).get("output"), "caller sampling config differs from selection")
        source, caller_source = row.get("source_per_cell"), caller.get("source_per_cell")
        _require(isinstance(source, Mapping) and isinstance(caller_source, Mapping), "selected source binding required")
        _require(evidence.same(source.get("freeze_ref"), caller_source.get("freeze_ref")), "caller freeze differs from selection")
        freeze_key = evidence.identity(source.get("freeze_ref"))
        if freeze_key not in frozen:
            frozen[freeze_key] = evidence.read(source["freeze_ref"])
        native = _core_refs(row, evidence); supplied_native = _core_refs(caller, evidence)
        for name, native_ref in native.items():
            _require(evidence.same(native_ref, supplied_native[name]), "caller runtime differs from selection")
            if name in modules:
                _require(evidence.same(modules[name], native_ref), "mixed selected sampling runtime identity")
            modules[name] = native_ref
        row_raw[cell_id] = anchors
    claimed_modules = _core_refs({"native_runtime_refs": contract.get("runtime_refs")}, evidence)
    _require(len(contract["runtime_refs"]) == len(_CORE_MODULES), "extra claimed runtime modules")
    for name in modules:
        _require(evidence.same(modules[name], claimed_modules[name]), "contract runtime mismatch: " + name)
    provenance = _native_provenance(contract, list(frozen.values()), modules, evidence)
    output = {}; used_profiles = set(); requests = {"warmup": 0, "runs": 0}; raw_blocks = 0
    for cell_id, row in selected.items():
        cell = cells[cell_id]; source = row["source_per_cell"]
        _require(cell.get("source_id") == source.get("source_id"), "cell source identity mismatch")
        _require(evidence.same(cell.get("freeze_ref"), source["freeze_ref"]), "cell freeze identity mismatch")
        freeze = frozen[evidence.identity(source["freeze_ref"])]
        collector = cell.get("collector_source_ref")
        _require(any(evidence.same(collector, candidate) for candidate in freeze.get("source_refs", [])), "collector source not bound to selected freeze")
        evidence.verify(collector)  # Read as evidence bytes only; never import it.
        claimed_raw = cell.get("raw_configuration_evidence")
        _require(isinstance(claimed_raw, list) and len(claimed_raw) == len(row_raw[cell_id]), "incomplete raw configuration evidence")
        claimed_ids = [evidence.identity(item.get("raw_ref")) for item in claimed_raw]
        _require(len(set(claimed_ids)) == len(claimed_ids) and set(claimed_ids) == set(row_raw[cell_id]), "wrong or duplicate cell raw refs")
        derived = None; per_cell_counts = {"warmup": 0, "runs": 0}; checked_raw = []
        for raw_id, anchor in row_raw[cell_id].items():
            raw = evidence.read(anchor["raw_ref"]); raw_blocks += 1
            _require(raw.get("key") == anchor["key"] and raw.get("status") == "complete", "raw record identity/status mismatch")
            payload = raw.get("payload")
            _require(isinstance(payload, Mapping) and set(payload) == _PAYLOAD_KEYS, "unsupported or missing request fields")
            _nonnull(payload, "request payload")
            _integer(payload["seed"], "request seed"); _integer(payload["top_k"], "request top_k", 1); _integer(payload["n_predict"], "n_predict", 1); _number(payload["temperature"], "temperature")
            _require(payload["cache_prompt"] is False and payload["ignore_eos"] is True and payload["stream"] is True and payload["temperature"] == 0 and payload["top_k"] == 1, "unsupported frozen request policy")
            config = raw.get("config", {})
            _require(config.get("seed") == payload["seed"] == row.get("config", {}).get("seed") and config.get("output") == payload["n_predict"] == row.get("config", {}).get("output"), "selected config/raw payload mismatch")
            argv = raw.get("actual_argv")
            _require(isinstance(argv, list) and bool(argv), "native argv identity required")
            _require(os.path.normcase(str(evidence.path(argv[0]))) == os.path.normcase(str(evidence.path(modules["llama-server.exe"]["path"]))), "raw native executable mismatch")
            for item in argv[1:]:
                _require(isinstance(item, str), "native argv must be text")
                _require(not any(item == flag or item.startswith(flag + "=") or (flag.endswith("-") and item.startswith(flag)) for flag in _CLI_PREFIXES), "unreviewed sampling CLI override")
            projection = {key: payload[key] for key in _PAYLOAD_KEYS if key != "prompt"}
            _require(projection == cell.get("payload_projection"), "claimed request projection mismatch")
            count = 0
            for phase in ("warmup", "runs"):
                batches = raw.get(phase)
                _require(isinstance(batches, list), "raw request phase missing")
                for batch in batches:
                    batch_requests = batch.get("requests")
                    _require(isinstance(batch_requests, list) and bool(batch_requests), "raw request list required")
                    for request in batch_requests:
                        response = request.get("response")
                        _require(isinstance(response, Mapping), "raw response container required")
                        current = _project_settings(response.get("generation_settings"), payload)
                        if derived is None:
                            derived = current
                        _require(current == derived, "resolved settings differ across selected requests")
                        count += 1; per_cell_counts[phase] += 1; requests[phase] += 1
            _require(count > 0, "no resolved settings evidence")
            checked_raw.append(evidence.verify(anchor["raw_ref"]))
        _require(derived is not None, "sampling evidence missing")
        profile_id = _profile_id(derived); used_profiles.add(profile_id)
        _require(cell.get("sampling_profile_id") == profile_id and profiles.get(profile_id) == derived, "claimed effective profile differs from raw settings")
        typed = _typed(derived)
        _require(cell.get("typed_policy_candidate") == typed and cell.get("backend_sampling") is False, "claimed typed policy/backend mismatch")
        output[cell_id] = {"typed_policy": typed, "backend_sampling": False, "effective_settings": derived, "provenance": {"source_id": source["source_id"], "selection_ref": selection_identity, "raw_configuration_refs": checked_raw, "freeze_ref": evidence.verify(source["freeze_ref"]), "collector_source_ref": evidence.verify(collector), "resolved_request_settings_checked": per_cell_counts, "native_lineage": provenance}, "limitations": list(_LIMITATIONS)}
    _require(used_profiles == set(profiles), "unreferenced or missing effective profiles")
    if "exporter_ref" in contract:
        evidence.verify(contract["exporter_ref"])  # identity only, never execution
    result = {"contract_ref": contract_ref, "evidence_refs": sorted(evidence.checked.values(), key=lambda item: item["path"]), "cells": output, "summary": {"selected_cells": len(output), "raw_blocks_verified": raw_blocks, "resolved_request_settings_checked": requests, "distinct_sampling_profiles": len(used_profiles), "all_backend_sampling_false": True, "all_min_keep_zero": True, "latency_fields_projected_or_used": False, "self_reported_summary_used_for_acceptance": False, "qualification": "source_bound_host_visible_llama_cpp_cpu_chain"}}
    _nonnull(result, "verified sampling binding")
    return result
