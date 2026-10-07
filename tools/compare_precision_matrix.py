"""Compare exact accelerated reports with a selectable DRAM reference path.

The default group is the first four native models. Supply --models and a new
--output file to reuse the runner for another group or architecture.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from heterollm_sim import _compiled_dram, _dram_numeric
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore
from heterollm_sim.reporting import report_dict, run_scenario
from validate_preset_matrix import LLAMA_DEFAULTS, report_checks, scenario_for


DEFAULT_HARDWARE = "local-native-rtx5080-9950x3d-gddr7-ddr5"
DEFAULT_MODELS = [
    "qwen2_5-0_5b", "qwen3-0_6b", "llama3_2-1b", "qwen2_5-1_5b",
]
DEFAULT_BASELINE = ROOT / "docs" / "exact_matrix_native_2026-10-07.json"
DEFAULT_OUTPUT = ROOT / "docs" / "precision_native_group1_2026-10-07.json"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _relative_error(reference: float, absolute: float) -> Optional[float]:
    if reference == 0:
        return 0.0 if absolute == 0 else None
    return absolute / abs(reference)


def numeric_differences(base: Any, other: Any, prefix: str = "") -> Dict[str, Any]:
    """Return compact absolute/relative deltas for numeric report fields."""
    result: Dict[str, Any] = {}
    if _is_number(base) and _is_number(other):
        left, right = float(base), float(other)
        absolute = abs(left - right)
        result[prefix or "$" ] = {
            "accelerated": base,
            "reference": other,
            "absolute_difference": absolute,
            "relative_difference": _relative_error(right, absolute),
            "relative_difference_status": (
                "undefined_reference_is_zero" if right == 0 and absolute != 0 else "defined"
            ),
        }
    elif isinstance(base, dict) and isinstance(other, dict):
        ignored = {"resource_totals", "organization_profiles", "owner_ids"}
        for key in sorted(set(base) & set(other)):
            if key in ignored:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(numeric_differences(base[key], other[key], path))
    elif isinstance(base, (list, tuple)) and isinstance(other, (list, tuple)):
        if len(base) == len(other) and base and all(
            _is_number(a) and _is_number(b) for a, b in zip(base, other)
        ):
            diffs = [abs(float(a) - float(b)) for a, b in zip(base, other)]
            rels = [
                _relative_error(float(b), diff)
                for a, b, diff in zip(base, other, diffs)
            ]
            finite_rels = [value for value in rels if value is not None]
            result[prefix or "$" ] = {
                "count": len(base),
                "max_absolute_difference": max(diffs),
                "mean_absolute_difference": sum(diffs) / len(diffs),
                "max_relative_difference": max(finite_rels) if finite_rels else None,
                "undefined_relative_count": sum(value is None for value in rels),
                "max_absolute_index": diffs.index(max(diffs)),
            }
        elif len(base) == len(other):
            for index, (left, right) in enumerate(zip(base, other)):
                path = f"{prefix}[{index}]"
                result.update(numeric_differences(left, right, path))
    return result


@contextmanager
def instrument_run(mode: str, counters: Dict[str, Any]) -> Iterator[None]:
    """Instrument only this sequential in-process run, restoring methods after."""
    original_compiled = _compiled_dram.execute_compiled
    original_numeric = _dram_numeric.run_numeric
    original_dram_accelerated = DramCore._execute_accelerated_accepted
    original_nand_accepted = NandCore._execute_accepted
    original_nand_detailed = NandCore._execute_detailed_accepted

    def compiled_wrapper(core, request):
        counters["dram_compiled_dispatches"] += 1
        result = original_compiled(core, request)
        if result is None:
            counters["dram_compiled_fallbacks"] += 1
        else:
            counters["dram_compiled_results"] += 1
            counters["dram_total_bursts"] += int(result.counters.get("burst_count", 0))
            counters["dram_skipped_row_hit_bursts"] += int(
                result.counters.get("accelerated_row_hit_bursts", 0)
            )
        return result

    def dram_accelerated_wrapper(self, request, first, stop):
        counters["dram_python_accelerated_calls"] += 1
        result = original_dram_accelerated(self, request, first, stop)
        counters["dram_total_bursts"] += int(result.counters.get("burst_count", 0))
        counters["dram_skipped_row_hit_bursts"] += int(
            result.counters.get("accelerated_row_hit_bursts", 0)
        )
        return result

    def nand_detailed_wrapper(self, request):
        counters["nand_detailed_calls"] += 1
        result = original_nand_detailed(self, request)
        counters["nand_detailed_pages"] += int(result.counters.get("page_count", 0))
        return result

    def nand_accelerated_wrapper(self, request):
        result = original_nand_accepted(self, request)
        pages = int(result.counters.get("accelerated_pages", 0))
        if pages:
            counters["nand_accelerated_calls"] += 1
            counters["nand_accelerated_pages"] += pages
        return result

    _compiled_dram.execute_compiled = compiled_wrapper
    DramCore._execute_accelerated_accepted = dram_accelerated_wrapper
    NandCore._execute_detailed_accepted = nand_detailed_wrapper

    if mode == "accelerated":
        counters["dram_reference_method"] = "production_accelerated_path"
        counters["dram_reference_full_burst_traversal"] = False
        NandCore._execute_accepted = nand_accelerated_wrapper
    elif mode == "serial_numeric":
        try:
            from precision_dram_reference import run_numeric_serial
        except ImportError as exc:
            _compiled_dram.execute_compiled = original_compiled
            DramCore._execute_accelerated_accepted = original_dram_accelerated
            NandCore._execute_detailed_accepted = original_nand_detailed
            raise RuntimeError(
                "serial_numeric reference is unavailable; wait for "
                "tools/precision_dram_reference.py"
            ) from exc
        _dram_numeric.run_numeric = run_numeric_serial
        counters["dram_reference_method"] = "serial_numeric_full_burst"
        counters["dram_reference_full_burst_traversal"] = True
        # Keep DRAM dispatch and replace its recurrence; force NAND page expansion.
        NandCore._execute_accepted = nand_detailed_wrapper
    elif mode == "exact_python":
        _compiled_dram.execute_compiled = lambda core, request: None
        counters["dram_reference_method"] = "exact_python_row_recurrence"
        counters["dram_reference_full_burst_traversal"] = False
        # Retain the Python repeated-row recurrence and force NAND page expansion.
        NandCore._execute_accepted = nand_detailed_wrapper
    else:
        raise ValueError(f"unsupported reference mode: {mode}")

    try:
        yield
    finally:
        _compiled_dram.execute_compiled = original_compiled
        _dram_numeric.run_numeric = original_numeric
        DramCore._execute_accelerated_accepted = original_dram_accelerated
        NandCore._execute_accepted = original_nand_accepted
        NandCore._execute_detailed_accepted = original_nand_detailed


def _new_counters() -> Dict[str, Any]:
    return {
        "dram_compiled_dispatches": 0,
        "dram_compiled_fallbacks": 0,
        "dram_compiled_results": 0,
        "dram_python_accelerated_calls": 0,
        "dram_total_bursts": 0,
        "dram_skipped_row_hit_bursts": 0,
        "nand_accelerated_calls": 0,
        "nand_accelerated_pages": 0,
        "nand_detailed_calls": 0,
        "nand_detailed_pages": 0,
        "dram_reference_full_burst_traversal": False,
    }


def _run_one(hardware: str, model: str, mode: str) -> Dict[str, Any]:
    counters = _new_counters()
    scenario = scenario_from_dict(scenario_for(hardware, model))
    started = time.perf_counter()
    with instrument_run(mode, counters):
        result = run_scenario(scenario, retention_policy="aggregate")
        report = report_dict(result)
    elapsed = time.perf_counter() - started
    # Round-trip checks JSON finiteness and converts extension containers.
    report = json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))
    failures = report_checks({"error": None, "report": report})
    if failures:
        raise ValueError(f"fresh {mode} report failed checks: {failures}")
    counters["host_elapsed_s"] = round(elapsed, 6)
    return {"summary": report["summary"], "requests": report["requests"], "counters": counters}


def _validate_saved_baseline(row: dict) -> None:
    if not row.get("report_available") or not row.get("summary") or not row.get("requests"):
        raise ValueError(f"accelerated baseline has no report for {row.get('model')}")
    failures = report_checks({
        "error": row.get("error"),
        "report": {"summary": row["summary"], "requests": row["requests"]},
    })
    if failures:
        raise ValueError(f"accelerated baseline is invalid for {row.get('model')}: {failures}")


def compare_case(baseline_row: dict, reference: Dict[str, Any], accelerated_probe: Dict[str, Any]) -> dict:
    _validate_saved_baseline(baseline_row)
    accelerated = {"summary": baseline_row["summary"], "requests": baseline_row["requests"]}
    fresh = {"summary": accelerated_probe["summary"], "requests": accelerated_probe["requests"]}
    return {
        "accelerated_call_counts": accelerated_probe["counters"],
        "reference_call_counts": reference["counters"],
        "reference_host_elapsed_s": reference["counters"]["host_elapsed_s"],
        "accelerated_probe_host_elapsed_s": accelerated_probe["counters"]["host_elapsed_s"],
        "baseline_vs_accelerated_probe_numeric_differences": numeric_differences(accelerated, fresh),
        "baseline_vs_reference_numeric_differences": numeric_differences(accelerated, {
            "summary": reference["summary"], "requests": reference["requests"]
        }),
        "accelerated": accelerated,
        "reference": {"summary": reference["summary"], "requests": reference["requests"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reference-mode", choices=("serial_numeric", "exact_python"), default="serial_numeric")
    args = parser.parse_args()

    baseline_data = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    if baseline_data.get("hardware") != args.hardware:
        raise ValueError(f"baseline hardware mismatch: {baseline_data.get('hardware')} != {args.hardware}")
    baseline_by_model = {row["model"]: row for row in baseline_data["results"]}
    output = {
        "schema": "heterollm.precision-matrix/v1",
        "hardware": args.hardware,
        "baseline_file": args.baseline_json.name,
        "reference_mode": args.reference_mode,
        "workload": {
            "requests": 1, "prompt_tokens": 512, "output_tokens": 128,
            "llama_cpp": LLAMA_DEFAULTS, "retention_policy": "aggregate",
        },
        "method_note": (
            "serial_numeric visits every DRAM burst with the exact transition kernel and no repeated-row fold; "
            "NAND reference expands pages. This is a software reference, not device calibration."
            if args.reference_mode == "serial_numeric" else
            "exact_python falls back to the existing Python row recurrence, which still folds repeated row-hit cycles; "
            "NAND reference expands pages. This is not a complete per-burst baseline."
        ),
        "relative_difference_convention": "absolute_difference / abs(reference); null with undefined_reference_is_zero when reference is 0 and the values differ",
        "results": [],
    }

    for model in args.models:
        if model not in baseline_by_model:
            raise ValueError(f"accelerated baseline lacks model: {model}")
        print(json.dumps({"event": "START", "hardware": args.hardware, "model": model}, ensure_ascii=False), flush=True)
        row = {
            "hardware": args.hardware,
            "model": model,
            "input": json.loads(json.dumps(baseline_by_model[model]["input"])),
        }
        try:
            _validate_saved_baseline(baseline_by_model[model])
            accelerated_probe = _run_one(args.hardware, model, "accelerated")
            reference = _run_one(args.hardware, model, args.reference_mode)
            row.update(status="completed", **compare_case(baseline_by_model[model], reference, accelerated_probe))
        except Exception as exc:
            row.update(status="failed", error={"exception_type": type(exc).__name__, "message": str(exc)})
        output["results"] = [item for item in output["results"] if item["model"] != model] + [row]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "RESULT", "hardware": args.hardware, "model": model,
                          "status": row["status"], "error": row.get("error"),
                          "reference_host_elapsed_s": row.get("reference_host_elapsed_s")}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "DONE", "hardware": args.hardware, "cases": len(output["results"]),
                      "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
