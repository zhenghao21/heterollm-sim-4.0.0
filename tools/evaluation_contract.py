"""Shared timing evaluation contract for native/simulator comparisons.

The evaluator is deliberately fail-closed.  Numeric fields are only usable
when the producer has declared the engine boundary, the canonical contract,
and a complete request aggregate.  Client/legacy counters remain diagnostic
and can never be relabelled as engine evidence.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any


ENGINE_CONTRACT_ID = "engine-boundary/v1"
CLIENT_CONTRACT_ID = "client-real-token/v1"
SUPPORTED_AGGREGATIONS = ("p50", "p90")
ENGINE_PROVEN_STATUSES = frozenset(("measured", "counter_proven", "marker_proven"))
FORMAL_AGGREGATE_SCOPE = "request_set"
METRIC_STATUSES = frozenset(("measured", "not_applicable", "evidence_insufficient", "incomplete"))
SEMANTIC_PROOF_SCHEMA = "engine-semantic-proof/v1"
ENGINE_COUNTER_UNPROVEN_STATUS = "counter_observed_unproven"
ENGINE_METRICS = {
    "ttft_ms": ("engine_ttft_ms", "engine_ttft_ms"),
    "tpot_ms": ("engine_tpot_ms", "engine_tpot_ms"),
    "e2e_ms": ("engine_e2e_ms", "engine_e2e_ms"),
}
CLIENT_METRICS = {
    "ttft_ms": ("client_ttft_ms", "client_ttft_ms"),
    "tpot_ms": ("client_tpot_ms", "client_tpot_ms"),
    "e2e_ms": ("client_e2e_ms", "client_e2e_ms"),
}


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _stat_value(value: object, aggregation: str) -> float | None:
    if isinstance(value, Mapping):
        value = value.get(f"{aggregation}_ms")
    return _finite_nonnegative(value)


def _metric_status(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    return str(status) if status is not None else None


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_engine_semantic_proof(
    proof: object, *, binary_artifacts: object,
    contract_id: str = "engine-stage+client-real-token/v3",
) -> tuple[bool, list[str]]:
    """Validate evidence that makes slot counters semantically usable.

    A producer supplied ``status=verified`` is only a claim.  This gate also
    requires the current executable/DLL SHA set, at least one source artifact
    whose bytes still match its declared SHA, and an explicit boundary
    declaration.  The function is intentionally read-only and can be reused by
    capture and replay tooling.
    """
    errors: list[str] = []
    if not isinstance(proof, Mapping):
        return False, ["semantic proof missing"]
    if proof.get("schema") not in {SEMANTIC_PROOF_SCHEMA, "engine-semantic-proof/v2"}:
        errors.append("semantic proof schema mismatch")
    if proof.get("status") != "verified":
        errors.append("semantic proof status is not verified")
    if proof.get("contract_id") != contract_id:
        errors.append("semantic proof contract mismatch")
    current = binary_artifacts if isinstance(binary_artifacts, list) else []
    current_hashes: set[str] = set()
    current_binary = None
    if not current:
        errors.append("current runtime artifacts missing")
    for index, item in enumerate(current):
        if not isinstance(item, Mapping):
            errors.append("current runtime artifact malformed")
            continue
        path_value, declared_sha = item.get("path"), item.get("sha256")
        if not isinstance(path_value, str) or not path_value.strip() or not _is_sha256(declared_sha):
            errors.append("current runtime artifact identity missing")
            continue
        path = Path(path_value).resolve()
        try:
            actual_sha = _sha256_file(path)
        except OSError:
            errors.append(f"current runtime artifact unreadable: {path}")
            continue
        current_hashes.add(actual_sha.lower())
        if index == 0:
            current_binary = actual_sha
        if actual_sha.lower() != str(declared_sha).lower():
            errors.append(("semantic proof executable SHA mismatch" if index == 0
                           else "runtime artifact SHA drift (runtime artifact SHA set mismatch)") + f": {path}")
    declared_binary = proof.get("binary_sha256")
    if not _is_sha256(declared_binary):
        errors.append("semantic proof binary_sha256 missing or malformed")
    elif current_binary is None or str(declared_binary).lower() != str(current_binary).lower():
        errors.append("semantic proof executable SHA mismatch")
    declared_runtime = proof.get("runtime_artifacts")
    if not isinstance(declared_runtime, list) or not declared_runtime:
        errors.append("semantic proof runtime_artifacts missing")
    else:
        runtime_hashes: set[str] = set()
        for item in declared_runtime:
            if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
                errors.append("semantic proof runtime artifact SHA missing or malformed")
                continue
            runtime_hashes.add(str(item.get("sha256")).lower())
        if runtime_hashes != current_hashes:
            errors.append("semantic proof runtime artifact SHA set mismatch")
    source_artifacts = proof.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        errors.append("semantic proof source_artifacts missing")
    else:
        matched_sources = 0
        for item in source_artifacts:
            if not isinstance(item, Mapping):
                errors.append("semantic proof source artifact malformed")
                continue
            path_value, declared_sha = item.get("path"), item.get("sha256")
            if not isinstance(path_value, str) or not path_value.strip() or not _is_sha256(declared_sha):
                errors.append("semantic proof source artifact identity missing")
                continue
            path = Path(path_value).resolve()
            if not path.exists() or not path.is_file():
                errors.append(f"semantic proof source artifact missing: {path}")
                continue
            if _sha256_file(path).lower() != str(declared_sha).lower():
                errors.append(f"semantic proof source artifact SHA mismatch: {path}")
                continue
            matched_sources += 1
        if matched_sources == 0:
            errors.append("semantic proof has no verified source artifact")
    boundary = proof.get("boundary")
    if not isinstance(boundary, Mapping):
        errors.append("semantic proof boundary declaration missing")
    else:
        source = str(boundary.get("source") or "")
        fields = boundary.get("fields")
        if not source or not isinstance(fields, list) or not fields:
            errors.append("semantic proof boundary declaration incomplete")
        if not ("server_slot_stats" in source or "engine_" in source
                or any("t_prompt_last" in str(field) or "t_gen_last" in str(field)
                       for field in fields if isinstance(field, str))):
            errors.append("semantic proof boundary source is not engine scoped")
    if proof.get("schema") == "engine-semantic-proof/v2":
        def read_ref(ref, label):
            if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str) or not _is_sha256(ref.get("sha256")):
                errors.append(label + " reference missing")
                return None
            try:
                path = Path(ref["path"])
                if _sha256_file(path) != ref["sha256"]:
                    errors.append(label + " SHA mismatch")
                    return None
                import json
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                errors.append(label + " unreadable")
                return None
        receipt = read_ref(proof.get("build_receipt"), "build receipt")
        if isinstance(receipt, Mapping):
            before, after = receipt.get("source_sha256_before"), receipt.get("source_sha256_after")
            if receipt.get("returncode") != 0 or not isinstance(before, Mapping) or not before or before != after:
                errors.append("build source receipt incomplete or drifted")
            else:
                for source_path, expected_sha in before.items():
                    try:
                        if _sha256_file(Path(source_path)) != expected_sha:
                            errors.append("build source drift: " + source_path)
                    except OSError:
                        errors.append("build source missing: " + source_path)
            built = receipt.get("binary_artifacts")
            if not isinstance(built, list) or not built or not any(item.get("sha256") == declared_binary for item in built):
                errors.append("build receipt does not bind executable")
            for item in built or []:
                try:
                    if _sha256_file(Path(item["path"])) != item["sha256"]:
                        errors.append("build output drift")
                except (KeyError, OSError, TypeError):
                    errors.append("build output missing")
            log = receipt.get("build_log") or {}
            try:
                if _sha256_file(Path(log["path"])) != log["sha256"]:
                    errors.append("build log drift")
            except (KeyError, OSError, TypeError):
                errors.append("build log missing")
        loaded = read_ref(proof.get("loaded_runtime_capture"), "loaded runtime capture")
        if not isinstance(loaded, Mapping) or loaded.get("status") != "captured":
            errors.append("actual loaded runtime evidence missing")
        elif {item.get("sha256") for item in loaded.get("artifacts", [])} != current_hashes:
            errors.append("actual loaded runtime set differs")
        identity = proof.get("extractor_sources")
        if not isinstance(identity, list) or not identity:
            errors.append("extractor sources missing")
        for item in identity or []:
            try:
                if _sha256_file(Path(item["path"])) != item["sha256"]:
                    errors.append("extractor source drift")
            except (KeyError, OSError, TypeError):
                errors.append("extractor source missing")
        if proof.get("supported_execution") != "non_speculative_fixed_tokens":
            errors.append("execution semantics unsupported")
    return not errors, errors


def _record_output_tokens(record: Mapping[str, object]) -> int | None:
    value = record.get("output_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def metric_status_for_records(records: object, metric: str) -> str:
    """Classify one metric without conflating NA with missing evidence.

    ``not_applicable`` is reserved for the mathematically undefined TPOT of
    records that each produced exactly one real token.  Unknown output counts,
    missing timing values, and missing records remain evidence failures.  A
    multi-request set with only some complete records is ``incomplete``.
    """
    if not isinstance(records, list) or not records:
        return "evidence_insufficient"
    is_tpot = "tpot" in str(metric).lower()
    field = str(metric)
    output_counts = [_record_output_tokens(item) if isinstance(item, Mapping) else None
                     for item in records]
    if any(value is None or value < 1 for value in output_counts):
        return "evidence_insufficient" if len(records) == 1 else "incomplete"
    if is_tpot:
        if all(value == 1 for value in output_counts):
            return "not_applicable"
        if any(value == 1 for value in output_counts):
            return "incomplete"
    missing = []
    for item, output_count in zip(records, output_counts):
        if is_tpot and output_count <= 1:
            continue
        if not isinstance(item, Mapping) or _finite_nonnegative(item.get(field)) is None:
            missing.append(item)
    if missing:
        return "evidence_insufficient" if len(records) == 1 else "incomplete"
    return "measured"


def metric_status_is_valid(status: object) -> bool:
    return isinstance(status, str) and status in METRIC_STATUSES


def _side_status(side: Mapping[str, object], aggregate: Mapping[str, object] | None) -> str | None:
    status = aggregate.get("engine_timing_status") if isinstance(aggregate, Mapping) else None
    if status is None:
        status = side.get("engine_timing_status")
    return str(status) if status is not None else None


def _side_contract(side: Mapping[str, object], aggregate: Mapping[str, object] | None) -> str | None:
    contract = aggregate.get("timing_contract_id") if isinstance(aggregate, Mapping) else None
    if contract is None:
        contract = side.get("timing_contract_id")
    return str(contract) if contract is not None else None


def _side_requests(side: Mapping[str, object], aggregate: Mapping[str, object] | None) -> tuple[int | None, list[str]]:
    """Return the declared request count and structural errors for one side."""
    errors: list[str] = []
    requests = side.get("requests")
    request_count = aggregate.get("request_count") if isinstance(aggregate, Mapping) else None
    if side.get("record_scope") == "request_0_legacy":
        if not isinstance(aggregate, Mapping) or aggregate.get("record_scope") != FORMAL_AGGREGATE_SCOPE:
            errors.append("request-0 legacy record cannot be used as formal aggregate")
    if requests is not None:
        if not isinstance(requests, list) or not requests:
            errors.append("requests missing or malformed")
        else:
            if request_count is not None and request_count != len(requests):
                errors.append("aggregate.request_count does not match requests")
            request_count = len(requests)
            ids = [item.get("request_id") if isinstance(item, Mapping) else None for item in requests]
            if any(not isinstance(item, str) or not item for item in ids) or len(set(map(str, ids))) != len(ids):
                errors.append("request_ids missing or duplicated")
            if isinstance(aggregate, Mapping) and aggregate.get("request_ids") != ids:
                errors.append("aggregate.request_ids does not match requests")
    if request_count is not None:
        if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count < 1:
            errors.append("aggregate.request_count invalid")
            request_count = None
        elif request_count > 1 and isinstance(aggregate, Mapping):
            if aggregate.get("record_scope") != FORMAL_AGGREGATE_SCOPE:
                errors.append("multi-request aggregate record_scope is not request_set")
            request_ids = aggregate.get("request_ids")
            if (not isinstance(request_ids, list) or len(request_ids) != request_count
                    or len(set(map(str, request_ids))) != request_count
                    or any(not isinstance(item, str) or not item for item in request_ids)):
                errors.append("multi-request aggregate request_ids incomplete")
    return request_count, errors


def _side_value(side: Mapping[str, object], field: str, aggregation: str,
                *, metric: str | None = None) -> tuple[float | None, list[str], str | None]:
    """Read one field and verify that its aggregate covers the request set."""
    aggregate = side.get("aggregate")
    aggregate = aggregate if isinstance(aggregate, Mapping) else None
    request_count, errors = _side_requests(side, aggregate)
    status = _side_status(side, aggregate)
    contract = _side_contract(side, aggregate)
    # Minimal one-request records used by the low-level evaluator may omit
    # producer proof metadata; full matrix cells are gated separately by
    # validate_payload/validate_cell.  When metadata is present, it must agree.
    if side.get("requests") is not None and status is None:
        errors.append("engine timing evidence is not proven")
    if side.get("requests") is not None and contract is None:
        errors.append("unsupported engine contract: missing")
    if status is not None and status not in ENGINE_PROVEN_STATUSES:
        errors.append("engine timing evidence is not proven")
    if contract is not None and contract != ENGINE_CONTRACT_ID:
        errors.append(f"unsupported engine contract: {contract}")
    if request_count is not None and request_count != 1 and status not in ENGINE_PROVEN_STATUSES:
        errors.append("engine timing evidence is not proven")

    if aggregate is not None:
        raw = aggregate.get(field)
        value = _stat_value(raw, aggregation)
        declared_status = _metric_status(raw)
        if declared_status is not None and not metric_status_is_valid(declared_status):
            errors.append(f"{field} status invalid")
        records = side.get("requests")
        if isinstance(records, list) and records:
            expected_status = metric_status_for_records(records, field)
            if declared_status != expected_status:
                errors.append(f"{field} status disagrees with request evidence")
            if expected_status != "measured" and value is not None:
                errors.append(f"{field} has a value outside its evidence status")
        if request_count is not None and isinstance(raw, Mapping):
            declared_count = raw.get("count")
            expected_count = 0 if declared_status == "not_applicable" else request_count
            if declared_count is not None and declared_count != expected_count:
                errors.append(f"{field}.count does not match applicable request_count")
        elif request_count is not None and raw is None:
            errors.append(f"{field} missing from aggregate")
        return value, errors, declared_status

    # Direct fields are a single-request compatibility form only.  A payload
    # that declares multiple requests without an aggregate cannot pass.
    if request_count is not None and request_count != 1:
        errors.append("multi-request side lacks aggregate")
    return _stat_value(side.get(field), aggregation), errors, _metric_status(side.get(field))


def _evaluate_specs(native: Mapping[str, object], prediction: Mapping[str, object],
                    specs: Mapping[str, tuple[str, str]], aggregation: str,
                    *, require_engine_contract: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "aggregation_policy": aggregation,
        "native_field": {metric: specs[metric][0] for metric in specs},
        "simulator_field": {metric: specs[metric][1] for metric in specs},
        "metrics": {},
        "evidence_reasons": [],
    }
    for metric, (native_field, simulator_field) in specs.items():
        n, n_errors, n_status = _side_value(native, native_field, aggregation, metric=metric)
        s, s_errors, s_status = _side_value(prediction, simulator_field, aggregation, metric=metric)
        reasons = [f"native:{item}" for item in n_errors] + [f"simulator:{item}" for item in s_errors]
        if not require_engine_contract:
            # Client diagnostics do not use engine evidence statuses, but the
            # same finite-number and request-aggregate checks still apply.
            reasons = [item for item in reasons if "engine timing evidence is not proven" not in item
                       and "unsupported engine contract" not in item]
        entry: dict[str, object] = {
            "native_ms": n, "simulator_ms": s,
            "signed_error_pct": None, "absolute_error_pct": None,
            "absolute_delta_ms": None, "status": "evidence_insufficient",
            "evidence_reasons": reasons,
        }
        if n_status == "incomplete" or s_status == "incomplete":
            entry["status"] = "incomplete"
        elif not reasons and metric == "tpot_ms" and n is None and s is None and n_status == s_status == "not_applicable":
            entry["status"] = "not_applicable"
        elif not reasons and n not in (None, 0) and s is not None:
            error = 100.0 * (s - n) / n
            entry.update(signed_error_pct=error, absolute_error_pct=abs(error),
                         absolute_delta_ms=abs(s - n), status="measured")
        result["metrics"][metric] = entry
        result["evidence_reasons"].extend(f"{metric}:{reason}" for reason in reasons)
    result["status"] = ("measured" if all(item["status"] in ("measured", "not_applicable")
                                           for item in result["metrics"].values())
                         else "incomplete" if any(item["status"] == "incomplete" for item in result["metrics"].values())
                         else "evidence_insufficient")
    return result


def evaluate_metrics(native: Mapping[str, object], prediction: Mapping[str, object],
                     *, boundary: str = "engine", aggregation: str = "p50",
                     contract_id: str | None = None) -> dict[str, object]:
    """Evaluate one payload using an explicit boundary and supported statistic.

    ``boundary='engine'`` is the only acceptance boundary.  ``p50`` and
    ``p90`` are the only supported aggregation policies and select the matching
    aggregate field; an unknown policy is rejected rather than silently treated
    as p50.
    """
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    if boundary == "engine":
        specs = ENGINE_METRICS
        expected_contract = ENGINE_CONTRACT_ID
        require_engine_contract = True
    elif boundary == "client":
        specs = CLIENT_METRICS
        expected_contract = CLIENT_CONTRACT_ID
        require_engine_contract = False
    else:
        raise ValueError(f"unsupported timing boundary: {boundary}")
    if contract_id is None:
        contract_id = expected_contract
    result = _evaluate_specs(native, prediction, specs, aggregation,
                             require_engine_contract=require_engine_contract)
    result.update({"boundary": boundary, "contract_id": contract_id})
    if contract_id != expected_contract:
        result["status"] = "evidence_insufficient"
        result["evidence_reasons"].append(
            f"contract_id mismatch: expected {expected_contract}, got {contract_id}")
    return result


__all__ = [
    "CLIENT_CONTRACT_ID", "ENGINE_CONTRACT_ID", "ENGINE_PROVEN_STATUSES",
    "ENGINE_COUNTER_UNPROVEN_STATUS", "FORMAL_AGGREGATE_SCOPE", "METRIC_STATUSES",
    "SEMANTIC_PROOF_SCHEMA", "SUPPORTED_AGGREGATIONS", "evaluate_metrics",
    "metric_status_for_records", "metric_status_is_valid", "validate_engine_semantic_proof",
]
