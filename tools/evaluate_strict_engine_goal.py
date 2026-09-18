"""Strict fixed-native development gate with one read-only score pass."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools import predict_stable_native_dataset as predictor
from tools import render_optimization_loop_report as report
from tools.evaluation_contract import derive_engine_metrics_ms
from tools.verify_fixed_native import verify_lock

METRICS = ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
REQUIRED_CELLS = 131
THRESHOLD_PCT = 10


def _cell_result(issues, metrics):
    insufficient = bool(issues) or len(metrics) != len(METRICS)
    failed = any(not value["passed"] for value in metrics.values())
    passed = not insufficient and not failed
    return {
        "verdict": "insufficient_evidence" if insufficient else "passed" if passed else "accuracy_failed",
        "issues": issues,
        "metrics": metrics,
        "all3_below10": passed,
        "accuracy_failed": failed,
        "insufficient_evidence": insufficient,
    }


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def check_cell(prediction, native, scored):
    """Recompute one cell from request timestamps and fixed native medians."""
    issues = []
    metrics = {}
    if prediction.get("status") != "predicted":
        return _cell_result(["prediction not successful"], {})
    if prediction.get("native_answers_used") is not False:
        issues.append("native answer usage missing or enabled")
    requests = prediction.get("requests")
    parallel = native.get("parallel")
    if type(parallel) is not int or parallel < 1 or not isinstance(requests, list) or len(requests) != parallel:
        return _cell_result([*issues, "request coverage mismatch"], {})
    actuals = native.get("native_actuals")
    if isinstance(actuals, list) and any(
        not isinstance(item, dict) or type(item.get("request_index")) is not int for item in actuals
    ):
        return _cell_result([*issues, "native request identity invalid"], {})

    values = {metric: [] for metric in METRICS}
    ids = set()
    metric_issues = {metric: [] for metric in METRICS}
    for row in requests:
        if not isinstance(row, dict):
            issues.append("request must be an object")
            continue
        ident = row.get("request_index")
        count = row.get("visible_output_tokens")
        if type(ident) is not int or ident in ids or not 0 <= ident < parallel:
            issues.append("request identity missing/duplicate")
            continue
        ids.add(ident)
        if type(count) is not int or count <= 1 or count != native.get("output_tokens"):
            issues.append("output token count mismatch")
            continue
        if row.get("prompt_tokens") != native.get("prompt_tokens"):
            issues.append("prompt token count mismatch")
        try:
            derived = derive_engine_metrics_ms(
                row.get("engine_request_begin_ns"),
                row.get("engine_first_token_ns"),
                row.get("engine_last_token_ns"),
                count,
            )
        except ValueError:
            issues.append("invalid engine timestamps")
            continue
        for metric, value in derived.items():
            if not finite_positive(row.get(metric)) or not math.isclose(
                value, row[metric], rel_tol=1e-9, abs_tol=1e-8
            ):
                metric_issues[metric].append("request metric/timestamp mismatch")
            values[metric].append(value)

    for metric in METRICS:
        try:
            if metric_issues[metric]:
                raise ValueError(";".join(metric_issues[metric]))
            if len(values[metric]) != parallel:
                raise ValueError("incomplete derived metrics")
            simulator_median = statistics.median(values[metric])
            native_median, _ = predictor.native_run_medians(native, metric)
            if not finite_positive(native_median):
                raise ValueError("invalid native median")
            aggregate = prediction.get("aggregate", {}).get(metric, {})
            if (
                aggregate.get("planned_requests") != parallel
                or aggregate.get("observed_requests") != parallel
                or aggregate.get("missing_requests") != 0
            ):
                raise ValueError("aggregate request coverage")
            stored = aggregate.get("median_ms")
            if not finite_positive(stored) or not math.isclose(
                simulator_median, stored, rel_tol=1e-9, abs_tol=1e-8
            ):
                raise ValueError("aggregate differs from timestamps")
            item = scored.get("metrics", {}).get(metric, {})
            report.metric_values(item)
            if (
                item.get("status") != "scored"
                or not finite_positive(item.get("simulator_median_ms"))
                or not math.isclose(simulator_median, item["simulator_median_ms"], rel_tol=1e-9, abs_tol=1e-8)
                or not math.isclose(native_median, item["native_median_ms"], rel_tol=1e-9, abs_tol=1e-8)
            ):
                raise ValueError("score differs from native/request derivation")
            error = abs(simulator_median - native_median) / native_median * 100
            metrics[metric] = {
                "simulator_median_ms": simulator_median,
                "native_median_ms": native_median,
                "absolute_percentage_error_pct": error,
                "signed_error_ms": simulator_median - native_median,
                "absolute_error_ms": abs(simulator_median - native_median),
                "passed": error < THRESHOLD_PCT,
            }
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            issues.append(metric + ":" + str(exc))
    return _cell_result(issues, metrics)


def evaluate(state_path, directory, score_name):
    """Run the fixed-native gate once; no repeated implementation snapshots."""
    state, state_ref = report.read_json(state_path)
    lock = verify_lock(state_path)
    selection, selection_ref = report.read_json(state["native_selection_ref"]["path"])
    selected = report.rows_by_id(selection["selected_cells"])
    expected = set(selection["selected_cell_ids"])
    if set(selected) != expected or len(expected) != REQUIRED_CELLS:
        raise ValueError("fixed 131-cell scope mismatch")

    freeze, freeze_ref = report.read_json(Path(directory) / "freeze.json")
    if not freeze.get("source", {}).get("files"):
        raise ValueError("empty frozen source map")
    predictor.verify_freeze_references(freeze)
    if set(report.rows_by_id(freeze["cells"])) != expected:
        raise ValueError("freeze scope missing/extra cells")

    # One canonical read-only report pass checks score/prediction bindings and
    # native selection links.  The per-cell loop below only recomputes the
    # three arithmetic metrics from request timestamps.
    report.load_evaluation(
        "strict_A",
        directory,
        score_name,
        selection,
        selection_ref,
        selected,
        {cell: report.native_cell_raws(row) for cell, row in selected.items()},
    )
    scores, score_ref = report.read_json(Path(directory) / score_name)
    scored = report.rows_by_id(scores.get("cells"))
    if set(scored) != expected:
        raise ValueError("score scope missing/extra cells")

    rows = []
    for cell in selection["selected_cell_ids"]:
        row = scored[cell]
        ref = row.get("prediction_ref")
        try:
            pred, actual_ref = report.read_json(report.evidence_path(ref, directory))
            if (
                actual_ref["sha256"] != ref.get("sha256")
                or pred.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]
                or pred.get("selection_sha256") != selection_ref["sha256"]
                or pred.get("cell_id") != cell
            ):
                raise ValueError("prediction identity mismatch")
            result = check_cell(pred, selected[cell], row)
        except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
            result = _cell_result([str(exc)], {})
        rows.append({"cell_id": cell, **result})

    passed = sum(row["all3_below10"] for row in rows)
    missing = sum(row["insufficient_evidence"] for row in rows)
    failed = sum(row["accuracy_failed"] for row in rows)
    return {
        "schema": "strict-engine-acceptance/v1",
        "gate_A": {
            "verdict": "accuracy_failed" if failed else "insufficient_evidence" if missing else "passed",
            "passed_cells": passed,
            "accuracy_failed_cells": failed,
            "insufficient_evidence_cells": missing,
            "required_cells": REQUIRED_CELLS,
            "required_metrics": REQUIRED_CELLS * len(METRICS),
            "failure_counts_may_overlap": True,
            "passing_metrics": sum(
                value["passed"] for row in rows if not row["issues"] for value in row["metrics"].values()
            ),
            "threshold_pct_strict": THRESHOLD_PCT,
            "prediction_coverage_pct": 100 * (REQUIRED_CELLS - missing) / REQUIRED_CELLS,
        },
        "gate_B": {
            "verdict": "unvalidated",
            "reason": "No registered independent acceptance and joint uncertainty evidence supplied; development gate never promotes B.",
        },
        "task_complete": False,
        "next_action": "repair_evidence_and_next_mechanism_round" if missing else "next_mechanism_round" if failed else "freeze_and_prepare_independent_acceptance",
        "native_evaluable_coverage": {
            "selected": REQUIRED_CELLS,
            "planned": selection["planned_cells"],
            "pct": REQUIRED_CELLS / selection["planned_cells"] * 100,
        },
        "native_lock": lock,
        "sources": {"state": state_ref, "selection": selection_ref, "freeze": freeze_ref, "score": score_ref},
        "cells": rows,
        "scope": "Fixed disclosed development gate only; no inference run, refit, or native remeasurement.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing overwrite")
    try:
        result = evaluate(args.state, args.evaluation, args.score_file)
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        result = {
            "schema": "strict-engine-acceptance/v1",
            "gate_A": {"verdict": "insufficient_evidence", "reason": str(exc)},
            "gate_B": {"verdict": "unvalidated"},
            "task_complete": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("gate_A", "gate_B", "task_complete")}, ensure_ascii=False))
    return 0 if result["gate_A"]["verdict"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
