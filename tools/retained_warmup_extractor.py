"""Derive timing-free retained-KV warmup qualification evidence.

The extractor deliberately reads only the raw record's warmup branch. It emits
configuration, token-count, slot-set, and identity evidence; it never emits
request duration, clock, or timestamp values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "heterollm.retained-warmup-static-qualification/v1"
CREATED_DATE = "2026-09-16"
ORDINARY_FAMILIES = frozenset({"qwen25", "smollm2", "tinyllama"})
HYBRID_FAMILIES = frozenset({"qwen35", "qwen38", "qwen38_gpu"})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _process_identity_digest(value: object) -> str:
    return _sha256_bytes(str(value).encode("utf-8"))


def _load_json(path: Path) -> Mapping[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON object required: {path}")
    return loaded


def _actual_file_ref(path: Path, *, declared: object | None = None) -> tuple[dict[str, object], list[str]]:
    """Return actual byte identity and fail-closed errors for a declared ref."""
    errors: list[str] = []
    if not path.is_file():
        return {"path": str(path)}, ["missing_file"]
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256_bytes(path.read_bytes())
    result: dict[str, object] = {"path": str(path), "sha256": actual_sha256, "bytes": actual_bytes}
    if isinstance(declared, Mapping):
        declared_sha = declared.get("sha256")
        declared_bytes = declared.get("bytes", declared.get("size_bytes"))
        if not isinstance(declared_sha, str) or not declared_sha:
            errors.append("declared_sha256_missing")
        elif declared_sha != actual_sha256:
            errors.append("declared_sha256_mismatch")
            result["declared_sha256"] = declared_sha
        if declared_bytes is not None:
            if type(declared_bytes) is not int:
                errors.append("declared_bytes_invalid")
            elif declared_bytes != actual_bytes:
                errors.append("declared_bytes_mismatch")
                result["declared_bytes"] = declared_bytes
    return result, errors


def _reference_from_declared(declared: object, label: str) -> tuple[dict[str, object] | None, list[str]]:
    if not isinstance(declared, Mapping) or not isinstance(declared.get("path"), str):
        return None, [f"{label}_reference_missing"]
    actual, errors = _actual_file_ref(Path(str(declared["path"])), declared=declared)
    return actual, [f"{label}_{error}" for error in errors]


def _arg_value(argv: Sequence[object], flag: str) -> object | None:
    values = [str(value) for value in argv]
    try:
        index = values.index(flag)
    except ValueError:
        return None
    return values[index + 1] if index + 1 < len(values) else None


def _family_scope(model_key: str) -> str:
    if model_key in ORDINARY_FAMILIES:
        return "ordinary_full_attention_candidate"
    if model_key in HYBRID_FAMILIES:
        return "hybrid_attention_cache_only"
    return "architecture_unclassified"


def _frozen_plan(cell: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
    plans = cell.get("plans")
    if not isinstance(plans, list) or len(plans) != 1 or not isinstance(plans[0], Mapping):
        return {}, ["single_frozen_plan_missing"]
    return plans[0], []


def _evidence_index(cell: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
    entries = cell.get("evidence_index")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], Mapping):
        return {}, ["single_verified_evidence_index_missing"]
    return entries[0], []


def _integer(value: object, name: str, errors: list[str]) -> int:
    if type(value) is not int:
        errors.append(f"{name}_invalid")
        return -1
    return int(value)


def _static_configuration(plan: Mapping[str, Any], errors: list[str]) -> dict[str, object]:
    prompt = _integer(plan.get("expected_prompt_tokens"), "expected_prompt_tokens", errors)
    output = _integer(plan.get("output"), "output", errors)
    parallel = _integer(plan.get("parallel"), "parallel", errors)
    slot_context = _integer(plan.get("kv_unified_per_slot"), "kv_unified_per_slot", errors)
    context = _integer(plan.get("ctx"), "ctx", errors)
    checks = {
        "two_warmup_batches": plan.get("warmup_batches") == 2,
        "three_measured_batches": plan.get("measure_batches") == 3,
        "one_process_block": plan.get("process_blocks") == 1,
        "nonflash": plan.get("flash_attention") is False,
        "unified_slot_context": slot_context == 2048 and context == parallel * slot_context,
        "fits_slot_context": prompt > 0 and output > 0 and prompt + output <= slot_context,
        "cache_ram_zero": plan.get("cache_ram_mib") == 0,
        "batch_ubatch_64": plan.get("batch") == 64 and plan.get("ubatch") == 64,
    }
    errors.extend(f"static_{name}" for name, passed in checks.items() if not passed)
    return {
        "prompt_tokens": prompt,
        "output_tokens": output,
        "parallel": parallel,
        "slot_context_tokens": slot_context,
        "unified_context_tokens": context,
        "warmup_batches": plan.get("warmup_batches"),
        "measure_batches": plan.get("measure_batches"),
        "process_blocks": plan.get("process_blocks"),
        "batch": plan.get("batch"),
        "ubatch": plan.get("ubatch"),
        "flash_attention": plan.get("flash_attention"),
        "cache_ram_mib": plan.get("cache_ram_mib"),
        "checks": checks,
    }


def _warmup_batches(
    raw: Mapping[str, Any], plan: Mapping[str, Any], config: Mapping[str, object], errors: list[str]
) -> list[dict[str, object]]:
    warmup = raw.get("warmup")
    if not isinstance(warmup, list) or len(warmup) != 2:
        errors.append("warmup_batch_count_not_two")
        return []
    expected_prompt = config["prompt_tokens"]
    expected_output = config["output_tokens"]
    parallel = config["parallel"]
    rows: list[dict[str, object]] = []
    for ordinal, batch in enumerate(warmup):
        if not isinstance(batch, Mapping):
            errors.append(f"warmup_{ordinal}_qualification_failed")
            rows.append({"warmup_ordinal": ordinal, "qualified": False})
            continue
        requests = batch.get("requests")
        if not isinstance(requests, list):
            requests = []
        slots: list[object] = []
        prompt_counts: list[object] = []
        output_counts: list[object] = []
        cache_counts: list[object] = []
        all_request_counts_match = True
        for request in requests:
            request = request if isinstance(request, Mapping) else {}
            response = request.get("response")
            response = response if isinstance(response, Mapping) else {}
            counts = response.get("timings")
            counts = counts if isinstance(counts, Mapping) else {}
            prompt_count = counts.get("prompt_n")
            output_count = counts.get("predicted_n")
            cache_count = counts.get("cache_n")
            prompt_counts.append(prompt_count)
            output_counts.append(output_count)
            cache_counts.append(cache_count)
            slots.append(request.get("slot"))
            all_request_counts_match = all_request_counts_match and (
                request.get("status") == "measured"
                and response.get("truncated") is not True
                and type(prompt_count) is int and prompt_count == expected_prompt
                and type(output_count) is int and output_count == expected_output
                and type(cache_count) is int and cache_count == 0
            )
        slots_present = all(type(slot) is int and slot >= 0 for slot in slots)
        slot_labels = sorted(slots, key=lambda value: (type(value).__name__, repr(value))) if slots_present else list(slots)
        qualified = (
            batch.get("status") == "complete"
            and batch.get("phase") == "warmup"
            and batch.get("process_block") == plan.get("block")
            and len(requests) == parallel
            and slots_present
            and len(set(slots)) == parallel
            and all_request_counts_match
        )
        if not qualified:
            errors.append(f"warmup_{ordinal}_qualification_failed")
        rows.append({
            "warmup_ordinal": ordinal,
            "complete": batch.get("status") == "complete",
            "phase_is_warmup": batch.get("phase") == "warmup",
            "server_block_matches_plan": batch.get("process_block") == plan.get("block"),
            "request_count": len(requests),
            "distinct_slot_count": len(set(slots)) if slots_present else 0,
            "slot_labels": slot_labels,
            "actual_prompt_token_counts": sorted(set(prompt_counts), key=lambda value: (value is None, repr(value))),
            "actual_output_token_counts": sorted(set(output_counts), key=lambda value: (value is None, repr(value))),
            "actual_cache_token_counts": sorted(set(cache_counts), key=lambda value: (value is None, repr(value))),
            "all_requests_match_frozen_counts": all_request_counts_match,
            "qualified": qualified,
        })
    return rows


def _payload_contract(raw: Mapping[str, Any], config: Mapping[str, object], errors: list[str]) -> dict[str, object]:
    payload = raw.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    checks = {
        "cache_prompt_false": payload.get("cache_prompt") is False,
        "n_predict_matches_frozen_output": payload.get("n_predict") == config["output_tokens"],
        "single_completion_no_n_override": "n" not in payload and "n_cmpl" not in payload,
        "no_explicit_slot_or_restore_payload": not any(
            key in payload for key in ("id_slot", "slot_save_path", "slot_restore_path", "slot_action")
        ),
        "no_lora_or_spec_payload": not any(
            key in payload for key in ("lora", "lora_adapters", "speculative", "spec_type")
        ),
    }
    errors.extend(f"payload_{name}" for name, passed in checks.items() if not passed)
    return {"checks": checks}


def _server_command_contract(raw: Mapping[str, Any], config: Mapping[str, object], errors: list[str]) -> dict[str, object]:
    argv = raw.get("actual_argv")
    argv = argv if isinstance(argv, list) else []
    checks = {
        "np_matches_parallel": _arg_value(argv, "-np") == str(config["parallel"]),
        "ctx_matches_unified_context": _arg_value(argv, "-c") == str(config["unified_context_tokens"]),
        "batch_matches": _arg_value(argv, "-b") == str(config["batch"]),
        "ubatch_matches": _arg_value(argv, "-ub") == str(config["ubatch"]),
        "flash_attention_disabled": _arg_value(argv, "-fa") == "off",
        "unified_kv_enabled": "-kvu" in [str(value) for value in argv],
        "cache_ram_zero": _arg_value(argv, "--cache-ram") == "0",
        "speculation_disabled": _arg_value(argv, "--spec-type") == "none",
    }
    errors.extend(f"server_command_{name}" for name, passed in checks.items() if not passed)
    return {"checks": checks}


def _server_identity(raw: Mapping[str, Any], plan: Mapping[str, Any], errors: list[str]) -> dict[str, object]:
    before = raw.get("runtime_before")
    after = raw.get("runtime_after")
    before = before if isinstance(before, Mapping) else {}
    after = after if isinstance(after, Mapping) else {}
    module_before = before.get("module_identity_sha256")
    module_after = after.get("module_identity_sha256")
    process_before = before.get("process_identity")
    process_after = after.get("process_identity")
    same_identity = bool(
        module_before and module_before == module_after and process_before is not None and process_before == process_after
    )
    if not same_identity:
        errors.append("single_server_process_identity_not_verified")
    key_matches = raw.get("key") == plan.get("key")
    if not key_matches:
        errors.append("raw_record_key_plan_key_mismatch")
    if raw.get("status") != "complete":
        errors.append("raw_record_not_complete")
    return {
        "record_key_matches_frozen_plan": key_matches,
        "server_block": plan.get("block"),
        "raw_status_complete": raw.get("status") == "complete",
        "module_identity_sha256": module_before if module_before == module_after else None,
        "process_identity_digest": _process_identity_digest(process_before) if same_identity else None,
        "same_server_process_identity_before_after": same_identity,
    }


def _source_refs(
    selection_path: Path,
    selection: Mapping[str, Any],
    cell: Mapping[str, Any],
    evidence: Mapping[str, Any],
    errors: list[str],
) -> dict[str, object]:
    selection_ref, selection_errors = _actual_file_ref(selection_path)
    errors.extend(f"selection_{error}" for error in selection_errors)
    source_per_cell = cell.get("source_per_cell")
    source_per_cell = source_per_cell if isinstance(source_per_cell, Mapping) else {}
    source_freeze, source_freeze_errors = _reference_from_declared(source_per_cell.get("freeze_ref"), "source_freeze")
    raw_record, raw_errors = _reference_from_declared(evidence.get("raw_ref"), "raw_record")
    receipt, receipt_errors = _reference_from_declared(evidence.get("receipt_ref"), "receipt")
    runtime_baseline, runtime_errors = _reference_from_declared(evidence.get("runtime_baseline_ref"), "runtime_baseline")
    selector, selector_errors = _reference_from_declared(selection.get("selector_source_ref"), "selector_source")
    errors.extend(source_freeze_errors + raw_errors + receipt_errors + runtime_errors + selector_errors)
    return {
        "selection": selection_ref,
        "source_freeze": source_freeze,
        "raw_record": raw_record,
        "receipt": receipt,
        "runtime_baseline": runtime_baseline,
        "frozen_plan_sha256": evidence.get("config_sha256"),
        "selector_source": selector,
        "selection_payload_sha256": selection.get("payload_sha256"),
    }


def _qualify_cell(selection_path: Path, selection: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, object]:
    errors: list[str] = []
    plan, plan_errors = _frozen_plan(cell)
    evidence, evidence_errors = _evidence_index(cell)
    errors.extend(plan_errors + evidence_errors)
    config = _static_configuration(plan, errors)
    plan_digest = _canonical_sha256(plan) if plan else None
    if plan and evidence.get("config_sha256") != plan_digest:
        errors.append("frozen_plan_sha256_mismatch")
    refs = _source_refs(selection_path, selection, cell, evidence, errors)
    raw_record_ref = refs.get("raw_record")
    raw: Mapping[str, Any] = {}
    if isinstance(raw_record_ref, Mapping) and isinstance(raw_record_ref.get("path"), str):
        raw_path = Path(str(raw_record_ref["path"]))
        if raw_path.is_file():
            raw = _load_json(raw_path)
        else:
            errors.append("raw_record_missing")
    else:
        errors.append("raw_record_missing")
    warmup = _warmup_batches(raw, plan, config, errors)
    payload = _payload_contract(raw, config, errors)
    command = _server_command_contract(raw, config, errors)
    identity = _server_identity(raw, plan, errors)
    model_key = str(cell.get("model_key", ""))
    scope = _family_scope(model_key)
    missing = sorted(set(errors))
    qualified = not missing
    template: dict[str, object] | None = None
    if qualified:
        retained = int(config["prompt_tokens"]) + int(config["output_tokens"]) - 1
        parallel = int(config["parallel"])
        template = {
            "kind": "symmetric_per_slot_completed_warmup_template",
            "slot_count": parallel,
            "retained_tokens_per_slot": retained,
            "total_distinct_occupied_tokens_lower_bound_at_final_warmup_boundary": parallel * retained,
            "slot_phase": "current_completed_retained",
            "requires_next_batch_slot_lifecycle_tracking": True,
        }
    if not qualified:
        ordinary_conclusion = "not_qualified"
    elif scope == "ordinary_full_attention_candidate":
        ordinary_conclusion = "qualified_if_model_cache_is_the_ordinary_attention_cache"
    else:
        ordinary_conclusion = "attention_cache_only_requires_separate_hybrid_cache_scope"
    return {
        "cell_id": str(cell.get("cell_id")),
        "model_key": model_key,
        "attention_family_scope": scope,
        "static_configuration": config,
        "source_refs": refs,
        "server_block_and_process": identity,
        "warmup_batches": warmup,
        "initial_retained_slot_template": template,
        "qualification": {
            "warmup_record_and_static_protocol": "qualified" if qualified else "not_qualified",
            "missing_or_failed": missing,
            "ordinary_attention_conclusion": ordinary_conclusion,
        },
        "payload_contract": payload,
        "server_command_contract": command,
    }


def _summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    qualified = [cell for cell in cells if cell["qualification"]["warmup_record_and_static_protocol"] == "qualified"]
    ordinary = [cell for cell in qualified if cell["attention_family_scope"] == "ordinary_full_attention_candidate"]
    hybrid = [cell for cell in qualified if cell["attention_family_scope"] == "hybrid_attention_cache_only"]
    reasons = Counter(
        reason for cell in cells for reason in cell["qualification"].get("missing_or_failed", [])
    )
    return {
        "selected_cells": len(cells),
        "warmup_record_and_static_protocol_qualified_cells": len(qualified),
        "warmup_record_and_static_protocol_unqualified_cells": len(cells) - len(qualified),
        "ordinary_full_attention_candidate_cells_with_qualified_warmup": len(ordinary),
        "hybrid_attention_cache_only_cells_with_qualified_warmup": len(hybrid),
        "architecture_unclassified_cells_with_qualified_warmup": len(qualified) - len(ordinary) - len(hybrid),
        "qualified_by_model_key": dict(sorted(Counter(cell["model_key"] for cell in qualified).items())),
        "unqualified_reasons": dict(sorted(reasons.items())),
    }


def derive_qualification(selection_path: Path, cell_ids: Iterable[str] | None = None) -> dict[str, object]:
    """Derive all or selected cell evidence without writing any file."""
    selection_path = selection_path.resolve()
    selection = _load_json(selection_path)
    selected = selection.get("selected_cells")
    selected_ids = selection.get("selected_cell_ids")
    if not isinstance(selected, list) or not isinstance(selected_ids, list):
        raise ValueError("selection must contain selected_cells and selected_cell_ids")
    by_id = {str(cell.get("cell_id")): cell for cell in selected if isinstance(cell, Mapping)}
    if len(by_id) != len(selected) or set(by_id) != set(str(value) for value in selected_ids):
        raise ValueError("selection cell IDs are not a one-to-one map")
    requested = list(cell_ids) if cell_ids is not None else list(selected_ids)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("requested cell IDs absent from selection: " + ", ".join(unknown))
    cells = [_qualify_cell(selection_path, selection, by_id[cell_id]) for cell_id in requested]
    selection_ref, selection_ref_errors = _actual_file_ref(selection_path)
    if selection_ref_errors:
        raise ValueError("selection byte identity cannot be established")
    return {
        "schema": SCHEMA,
        "created_date": CREATED_DATE,
        "scope": {
            "selected_cell_count": len(cells),
            "selection_total_cell_count": len(selected_ids),
            "raw_record_fields_extracted": [
                "record_status", "warmup.batch_status", "warmup.phase", "warmup.process_block",
                "warmup.request_status", "warmup.slot", "warmup.response.truncated",
                "warmup.response.counts.prompt_n", "warmup.response.counts.predicted_n",
                "warmup.response.counts.cache_n", "runtime_module_identity", "process_identity",
                "payload.count_and_cache_flags", "actual_argv.non_timing_runtime_flags",
            ],
            "non_count_measurement_data_retained": False,
        },
        "source_contract": {
            "selection": selection_ref,
            "selection_payload_sha256": selection.get("payload_sha256"),
            "static_protocol_rule": "2 HTTP warmup batches then 3 measured batches in one server process block; cache_ram=0; unified slot context=2048",
            "completed_slot_rule": "on the qualified ordinary singleton-owner path, a completed request retains prompt_tokens + output_tokens - 1 KV tokens",
            "initialization_boundary": "immediately after the final qualified HTTP warmup batch; no later purge/clear/lifecycle state is inferred",
            "not_a_high_water_claim": True,
        },
        "summary": _summary(cells),
        "limitations": [
            "The template is valid only at the final-warmup boundary. It does not survive an unobserved clear, purge, shift, restore, cancellation, failure, or slot reuse event.",
            "Hybrid rows apply only to their ordinary attention cache after an independent cache-scope qualification; recurrent or linear-attention state is not token KV occupancy.",
            "A later simulator must track per-slot lifecycle transitions. A fresh cohort and parallel count cannot recreate this history.",
        ],
        "cells": cells,
    }


def write_qualification(result: Mapping[str, Any], output_path: Path, *, overwrite: bool = False) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing evidence: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, output_path)


def _default_selection_path() -> Path:
    return Path(__file__).resolve().parents[2] / "stable_native_dataset.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=_default_selection_path())
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = derive_qualification(args.selection, args.cell_ids)
    write_qualification(result, args.output, overwrite=args.overwrite)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
