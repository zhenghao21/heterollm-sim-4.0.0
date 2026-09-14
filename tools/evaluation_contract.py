"""Shared timing evaluation contract for native/simulator comparisons.

The contract keeps engine timing separate from client transport timing.  A
missing or unverified engine field is evidence-insufficient; it is never
silently replaced with a client or legacy counter field.
"""
from __future__ import annotations

import math
from typing import Mapping


ENGINE_CONTRACT_ID = "engine-boundary/v1"
ENGINE_METRICS = {
    "ttft_ms": ("engine_ttft_ms", "engine_ttft_ms"),
    "tpot_ms": ("engine_tpot_ms", "engine_tpot_ms"),
    "e2e_ms": ("engine_e2e_ms", "engine_e2e_ms"),
}


def _p50(value: object) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("p50_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _side_value(side: Mapping[str, object], field: str) -> float | None:
    aggregate = side.get("aggregate")
    if isinstance(aggregate, Mapping):
        value = _p50(aggregate.get(field))
        if value is not None:
            return value
    # Explicit per-request engine fields are allowed for a single request.
    return _p50(side.get(field))


def evaluate_metrics(native: Mapping[str, object], prediction: Mapping[str, object],
                     *, boundary: str = "engine", aggregation: str = "p50",
                     contract_id: str = ENGINE_CONTRACT_ID) -> dict[str, object]:
    """Evaluate one payload using one explicit boundary and aggregation.

    ``boundary='engine'`` is the acceptance contract.  Client stream fields
    are available only when callers explicitly request ``client``.
    """
    if boundary == "engine":
        specs = ENGINE_METRICS
    elif boundary == "client":
        specs = {
            "ttft_ms": ("client_ttft_ms", "client_ttft_ms"),
            "tpot_ms": ("client_tpot_ms", "client_tpot_ms"),
            "e2e_ms": ("client_e2e_ms", "client_e2e_ms"),
        }
    else:
        raise ValueError(f"unsupported timing boundary: {boundary}")
    result: dict[str, object] = {
        "boundary": boundary,
        "aggregation_policy": aggregation,
        "contract_id": contract_id,
        "native_field": {metric: specs[metric][0] for metric in specs},
        "simulator_field": {metric: specs[metric][1] for metric in specs},
        "metrics": {},
    }
    for metric, (native_field, simulator_field) in specs.items():
        n = _side_value(native, native_field)
        s = _side_value(prediction, simulator_field)
        entry = {"native_ms": n, "simulator_ms": s,
                 "signed_error_pct": None, "absolute_error_pct": None,
                 "absolute_delta_ms": None, "status": "evidence_insufficient"}
        if n not in (None, 0) and s is not None:
            error = 100.0 * (s - n) / n
            entry.update(signed_error_pct=error, absolute_error_pct=abs(error),
                         absolute_delta_ms=abs(s - n), status="measured")
        result["metrics"][metric] = entry
    result["status"] = ("measured" if all(v["status"] == "measured"
                                           for v in result["metrics"].values())
                         else "evidence_insufficient")
    return result


__all__ = ["ENGINE_CONTRACT_ID", "evaluate_metrics"]
