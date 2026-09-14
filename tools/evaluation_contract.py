"""Shared timing evaluation contract for native/simulator comparisons.

The evaluator is deliberately fail-closed.  Numeric fields are only usable
when the producer has declared the engine boundary, the canonical contract,
and a complete request aggregate.  Client/legacy counters remain diagnostic
and can never be relabelled as engine evidence.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


ENGINE_CONTRACT_ID = "engine-boundary/v1"
CLIENT_CONTRACT_ID = "client-real-token/v1"
SUPPORTED_AGGREGATIONS = ("p50", "p90")
ENGINE_PROVEN_STATUSES = frozenset(("measured", "counter_proven", "marker_proven"))
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


def _side_status(side: Mapping[str, object], aggregate: Mapping[str, object] | None) -> str | None:
    status = side.get("engine_timing_status")
    if status is None and isinstance(aggregate, Mapping):
        status = aggregate.get("engine_timing_status")
    return str(status) if status is not None else None


def _side_contract(side: Mapping[str, object], aggregate: Mapping[str, object] | None) -> str | None:
    contract = side.get("timing_contract_id")
    if contract is None and isinstance(aggregate, Mapping):
        contract = aggregate.get("timing_contract_id")
    return str(contract) if contract is not None else None


def _side_requests(side: Mapping[str, object], aggregate: Mapping[str, object] | None) -> tuple[int | None, list[str]]:
    """Return the declared request count and structural errors for one side."""
    errors: list[str] = []
    requests = side.get("requests")
    request_count = aggregate.get("request_count") if isinstance(aggregate, Mapping) else None
    if requests is not None:
        if not isinstance(requests, list) or not requests:
            errors.append("requests missing or malformed")
        else:
            if request_count is not None and request_count != len(requests):
                errors.append("aggregate.request_count does not match requests")
            request_count = len(requests)
    if request_count is not None:
        if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count < 1:
            errors.append("aggregate.request_count invalid")
            request_count = None
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
    if status is not None and status not in ENGINE_PROVEN_STATUSES:
        errors.append("engine timing evidence is not proven")
    if contract is not None and contract != ENGINE_CONTRACT_ID:
        errors.append(f"unsupported engine contract: {contract}")
    if request_count is not None and request_count != 1 and status not in ENGINE_PROVEN_STATUSES:
        errors.append("engine timing evidence is not proven")

    if aggregate is not None:
        raw = aggregate.get(field)
        value = _stat_value(raw, aggregation)
        if isinstance(raw, Mapping) and raw.get("status") == "not_applicable":
            return None, errors, "not_applicable"
        if isinstance(raw, Mapping) and raw.get("status") not in (None, "measured"):
            errors.append(f"{field} status incomplete")
        if request_count is not None and isinstance(raw, Mapping):
            declared_count = raw.get("count")
            if declared_count is not None and declared_count != request_count:
                # TPOT may legitimately have no observations for a one-token
                # request; its explicit not_applicable status is handled below.
                if not (metric == "tpot_ms" and _metric_status(raw) == "not_applicable"):
                    errors.append(f"{field}.count does not match request_count")
        elif request_count is not None and raw is None:
            errors.append(f"{field} missing from aggregate")
        return value, errors, _metric_status(raw)

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
        if metric == "tpot_ms" and n is None and s is None and n_status == s_status == "not_applicable":
            entry["status"] = "not_applicable"
        elif not reasons and n not in (None, 0) and s is not None:
            error = 100.0 * (s - n) / n
            entry.update(signed_error_pct=error, absolute_error_pct=abs(error),
                         absolute_delta_ms=abs(s - n), status="measured")
        result["metrics"][metric] = entry
        result["evidence_reasons"].extend(f"{metric}:{reason}" for reason in reasons)
    result["status"] = ("measured" if all(item["status"] in ("measured", "not_applicable")
                                           for item in result["metrics"].values())
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
    "SUPPORTED_AGGREGATIONS", "evaluate_metrics",
]
