"""Small structural diff for native/simulator scheduler observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        value = value.get("schedule_trace", value.get("scheduler_trace", ()))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("scheduler trace must be an ordered sequence or result mapping")
    rows = tuple(row for row in value if isinstance(row, Mapping))
    if len(rows) != len(value):
        raise ValueError("scheduler trace contains a non-mapping row")
    return rows


def _category(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> str:
    if left.get("kind") != right.get("kind"):
        return "request_selection"
    if key in {"logical_batch", "states"}:
        return "request_selection" if key == "logical_batch" else "state_evolution"
    if key == "execution_plan":
        plan_left = left.get("execution_plan", {})
        plan_right = right.get("execution_plan", {})
        if isinstance(plan_left, Mapping) and isinstance(plan_right, Mapping):
            if plan_left.get("physical_ubatches") != plan_right.get("physical_ubatches"):
                return "ubatch"
            if plan_left.get("kv_ranges") != plan_right.get("kv_ranges"):
                return "kv"
            if plan_left.get("execution_dependencies") != plan_right.get("execution_dependencies"):
                return "execution_dependency"
        return "execution_work"
    if key in {"time_ns", "completion_ns"}:
        return "timing"
    return "configuration"


def diff_scheduler_traces(reference: Any, candidate: Any) -> dict[str, Any]:
    """Return the first structural divergence and a conservative category.

    This compares simulator observations or a native trace exported with the
    same schema.  It never treats native timestamps as prediction inputs.
    """
    left_rows = _rows(reference)
    right_rows = _rows(candidate)
    for index, (left, right) in enumerate(zip(left_rows, right_rows)):
        if left == right:
            continue
        keys = tuple(dict.fromkeys((*left.keys(), *right.keys())))
        key = next((name for name in keys if left.get(name) != right.get(name)), "kind")
        return {
            "schema": "heterollm.scheduler-trace-diff/v1",
            "equal": False,
            "first_divergence": index,
            "category": _category(left, right, key),
            "field": key,
            "reference": dict(left),
            "candidate": dict(right),
            "compared_rows": index + 1,
        }
    if len(left_rows) != len(right_rows):
        index = min(len(left_rows), len(right_rows))
        return {
            "schema": "heterollm.scheduler-trace-diff/v1",
            "equal": False,
            "first_divergence": index,
            "category": "request_selection",
            "field": "row_count",
            "reference": len(left_rows),
            "candidate": len(right_rows),
            "compared_rows": index,
        }
    return {
        "schema": "heterollm.scheduler-trace-diff/v1",
        "equal": True,
        "first_divergence": None,
        "category": None,
        "field": None,
        "compared_rows": len(left_rows),
    }


__all__ = ["diff_scheduler_traces"]
