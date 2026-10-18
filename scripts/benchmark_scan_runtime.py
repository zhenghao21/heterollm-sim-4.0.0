"""Measure full serving execution with real scan geometry and explicit lengths."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import nullcontext
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument("--implementation-root", type=Path)
_source_args, _ = _source_parser.parse_known_args()
IMPLEMENTATION_ROOT = ((_source_args.implementation_root or (ROOT / "src"))
                       .resolve(strict=True))
if not (IMPLEMENTATION_ROOT / "heterollm_sim" / "__init__.py").is_file():
    raise ValueError("--implementation-root must contain the heterollm_sim package")
sys.path[:0] = [str(IMPLEMENTATION_ROOT), str(ROOT / "scripts")]

# Load the implementation before timing so concurrent development cannot
# replace lazily imported source halfway through a baseline measurement.
import heterollm_sim.dram_core
import heterollm_sim.nand_core
import heterollm_sim.hbf_staging
import heterollm_sim.hbf_staging_runtime
import heterollm_sim.data_motion
import heterollm_sim.event_kernel
import heterollm_sim.serving
import heterollm_sim.scalable_serving
import heterollm_sim._hbm_numeric
import heterollm_sim._compiled_nand
import heterollm_sim._compiled_dram
import heterollm_sim._compiled_hbm
import heterollm_sim._resource_ledger
import heterollm_sim._cost_ledger
import heterollm_sim._hbm_pack
import heterollm_sim._nand_pack
import heterollm_sim.planner


def preload_execution_modules():
    """Load available native execution modules outside measured simulation.

    Whole-package diagnostic baselines can predate these modules. Probe that
    package's actual import path rather than borrowing current implementations.
    """
    loaded = {}
    for suffix in ("_token_program", "_physical_task", "_io_execution", "_nand_execution"):
        name = "heterollm_sim." + suffix
        if importlib.util.find_spec(name) is not None:
            loaded[suffix] = importlib.import_module(name).__file__
    return loaded


PRELOADED_MODULES = preload_execution_modules()
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.reporting import run_scenario
from b200_hbm_hbf_scan import collect_result, json_default, scenario_gc_thresholds, utc_now, write_json
from b200_hbm_hbf_scan_config import build_scan_scenario


_RUN_METADATA = frozenset({
    "wall_seconds", "cpu_seconds", "finished_at", "python_executable",
    "gc_thresholds", "decode_unprofiled_steps", "decode_unprofiled_wall_seconds",
    "profiled", "purpose", "benchmark_timing", "source_paths", "implementation_root", "transaction_capture", "preloaded_modules",
})
_COHORT_TIMING_METADATA = frozenset({
    "wall_seconds", "cpu_seconds", "profiled", "elapsed_wall_seconds",
})


def load_sources(*, hbm_source=None):
    """Select the HBM numeric source before constructing runtime/core state.

    A complete NAND implementation must be selected with implementation-root;
    its page wrapper, packing and numeric state machine live in distinct modules.
    HBM packed states retain run_hbm_numeric after their first construction, so
    this replacement occurs before execution, never between measured cohorts.

    A dedicated Numba cache namespace prevents source-local environments from
    another diagnostic module name being deserialized into this dispatcher.
    """
    if hbm_source is None:
        return {}
    name = "heterollm_sim._hbm_numeric"
    path = Path(hbm_source).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Python source: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules[name]
    from numba import config as numba_config
    previous_cache_dir = numba_config.CACHE_DIR
    numba_config.CACHE_DIR = str(ROOT / "tmp" / "benchmark_source_cache" / name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules[name] = previous
        raise
    finally:
        numba_config.CACHE_DIR = previous_cache_dir
    package, attribute = name.rsplit(".", 1)
    setattr(sys.modules[package], attribute, module)
    return {"hbm": str(path)}

def timing_summary(row):
    """Separate actual elapsed measurements from profiler-instrumented work."""
    cohorts = row["cohort_timings"]
    prefill = []
    for item in cohorts:
        if item["kind"] == "decode":
            break
        if item["kind"] == "prefill":
            prefill.append(item)
    decode = [item for item in cohorts
              if item["kind"] == "decode" and item.get("profiled") is False]
    decode_wall = sum(item["wall_seconds"] for item in decode)
    return {
        "prefill_before_first_token_cohorts": [item["completed"] for item in prefill],
        "prefill_before_first_token_wall_seconds": sum(item["wall_seconds"] for item in prefill),
        "prefill_wall_scope": "sum of Prefill cohort intervals; excludes initial setup/control plane",
        # Earlier results only retained individual intervals. Do not infer
        # startup-to-first-token wall time from their instrumented total.
        "wall_to_first_token_seconds": prefill[-1].get("elapsed_wall_seconds") if prefill else None,
        "prefill_all_unprofiled": bool(prefill) and all(item.get("profiled") is False for item in prefill),
        "decode_unprofiled_cohorts": [item["completed"] for item in decode],
        "decode_unprofiled_steps": len(decode),
        "decode_unprofiled_wall_seconds": decode_wall,
        "decode_unprofiled_seconds_per_step": decode_wall / len(decode) if decode else None,
        "total_wall_seconds": row["wall_seconds"],
        "total_cpu_seconds": row.get("cpu_seconds"),
        "transaction_capture": bool(row.get("transaction_capture")),
        # Even when profiling starts after the last cohort, result reduction
        # can be instrumented. The whole-run flag remains conservative here.
        "total_contains_profiling": row.get("profiled") is not False or any(
            item.get("profiled") is not False for item in cohorts),
    }


def _simulation_result(row):
    result = {key: value for key, value in row.items() if key not in _RUN_METADATA}
    result["cohort_timings"] = [
        {key: value for key, value in item.items() if key not in _COHORT_TIMING_METADATA}
        for item in row["cohort_timings"]
    ]
    return result


def compare_results(baseline, candidate):
    """Compare every simulation field exactly, including cohort completion times.

    Only the named run/timing fields above are omitted. In particular,
    request finish_ns, resource counters, unknown fields and ordered cohorts
    remain part of the comparison. No numerical tolerance is applied.
    """
    differences = []
    difference_count = 0

    def changed(path, left, right):
        nonlocal difference_count
        difference_count += 1
        if len(differences) < 20:
            differences.append({"path": path, "baseline": left, "candidate": right})

    def compare(left, right, path):
        if type(left) is not type(right):
            changed(path, left, right)
        elif isinstance(left, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left:
                    changed(f"{path}.{key}", {"missing": True}, right[key])
                elif key not in right:
                    changed(f"{path}.{key}", left[key], {"missing": True})
                else:
                    compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                changed(f"{path}.length", len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{index}]")
        elif isinstance(left, float):
            if math.isnan(left) or math.isnan(right) or left.hex() != right.hex():
                changed(path, left, right)
        elif left != right:
            changed(path, left, right)

    compare(_simulation_result(baseline), _simulation_result(candidate), "$")
    before, after = timing_summary(baseline), timing_summary(candidate)
    equal = difference_count == 0
    notes = []
    if not equal:
        notes.append("Simulation results differ; no speedup is accepted.")
    complete = baseline.get("status") == candidate.get("status") == "completed"
    if not complete:
        notes.append("Both runs must be completed before any speedup is accepted.")

    def ratio(left, right):
        return (left / right if equal and complete and math.isfinite(left)
                and math.isfinite(right) and left > 0 and right > 0 else None)

    decode_speedup = None
    traced = before["transaction_capture"] or after["transaction_capture"]
    if (not traced and before["decode_unprofiled_cohorts"]
            and before["decode_unprofiled_cohorts"] == after["decode_unprofiled_cohorts"]):
        decode_speedup = ratio(before["decode_unprofiled_wall_seconds"], after["decode_unprofiled_wall_seconds"])
    else:
        notes.append("Decode timing is instrumented, absent or uses different cohort windows; Decode speedup is unavailable.")
    prefill_speedup = None
    if (not traced and before["prefill_all_unprofiled"] and after["prefill_all_unprofiled"]
            and before["prefill_before_first_token_cohorts"] == after["prefill_before_first_token_cohorts"]):
        prefill_speedup = ratio(before["prefill_before_first_token_wall_seconds"], after["prefill_before_first_token_wall_seconds"])
    else:
        notes.append("Prefill timing is instrumented or the cohort windows differ; Prefill speedup is unavailable.")
    total_speedup = None
    if not traced and not before["total_contains_profiling"] and not after["total_contains_profiling"]:
        total_speedup = ratio(before["total_wall_seconds"], after["total_wall_seconds"])
    else:
        notes.append("Total wall time includes instrumentation and must not establish speedup.")
    if traced:
        notes.append("Transaction capture instruments execution; no captured timing establishes speedup.")
    return {"exactly_equal": equal, "difference_count": difference_count,
            "differences": differences, "timing": {"baseline": before, "candidate": after},
            "speedup": {"prefill_unprofiled": prefill_speedup,
                        "decode_unprofiled": decode_speedup, "total_unprofiled": total_speedup},
            "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare saved results exactly; this does not run a simulation")
    parser.add_argument("--compare-transactions", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare complete diagnostic transaction traces without running simulation")
    parser.add_argument("--transactions", type=Path,
                        help="Observe physical transactions in a separate diagnostic run; disables speedup claims")
    parser.add_argument("--architecture", choices=("HBM", "HBF"))
    parser.add_argument("--implementation-root", type=Path,
                        help="Load the complete diagnostic heterollm_sim package from this source root")
    parser.add_argument("--prompt", type=int, default=32)
    parser.add_argument("--output", type=int, default=3)
    parser.add_argument("--model", default="qwen2_5-0_5b")
    parser.add_argument("--result", type=Path, help="Benchmark result or optional comparison output")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-decode-after", type=int,
                        help="Profile only after this many unprofiled Decode steps")
    parser.add_argument("--hbm-source", type=Path,
                        help="Use this diagnostic _hbm_numeric source before starting the run")
    args = parser.parse_args()
    if args.compare_transactions:
        from capture_physical_transactions import compare_transaction_files
        comparison = compare_transaction_files(*args.compare_transactions)
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.result, comparison)
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
        return 0 if comparison["exactly_equal"] else 1
    if args.compare:
        baseline, candidate = (json.loads(path.read_text(encoding="utf-8")) for path in args.compare)
        comparison = compare_results(baseline, candidate)
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.result, comparison)
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
        return 0 if comparison["exactly_equal"] else 1
    if args.architecture is None or args.result is None:
        parser.error("a benchmark run requires --architecture and --result")
    if args.profile_decode_after is not None and (not args.profile or args.profile_decode_after < 1):
        parser.error("--profile-decode-after requires --profile and a positive step count")
    source_paths = load_sources(hbm_source=args.hbm_source)
    point = {"architecture": args.architecture, "model_id": args.model,
             "prompt_tokens": args.prompt, "output_tokens": args.output}
    scenario = build_scan_scenario(args.model, args.architecture, args.prompt, args.output)
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    cpu_started = time.process_time()
    cohort_timings = []
    previous_cohort_at = started
    previous_cohort_cpu = cpu_started
    profile_active = False
    decode_steps = 0
    print("implementation loaded; starting full benchmark request", flush=True)
    def report(event):
        nonlocal previous_cohort_at, previous_cohort_cpu, profile_active, decode_steps
        if event.stage in {"control_plane", "serving_cohorts"}:
            now, cpu_now = time.perf_counter(), time.process_time()
            kind = event.metadata.get("cohort_kind")
            if event.stage == "serving_cohorts":
                if event.completed:
                    cohort_timings.append({"completed": event.completed, "kind": kind,
                        "wall_seconds": now - previous_cohort_at,
                        "elapsed_wall_seconds": now - started,
                        "cpu_seconds": cpu_now - previous_cohort_cpu, "profiled": profile_active,
                        "simulated_time_ns": event.simulated_time_ns})
                previous_cohort_at, previous_cohort_cpu = now, cpu_now
                if kind == "decode":
                    decode_steps += 1
                if (profiler and not profile_active and args.profile_decode_after is not None
                        and decode_steps >= args.profile_decode_after):
                    profiler.enable()
                    profile_active = True
            print(json.dumps({"stage": event.stage, "completed": event.completed,
                "kind": kind, "wall_seconds": now - started}), flush=True)
    if profiler and args.profile_decode_after is None:
        profiler.enable()
        profile_active = True
    capture = nullcontext()
    if args.transactions:
        from capture_physical_transactions import PhysicalTransactionCapture
        capture = PhysicalTransactionCapture(args.transactions)
    with scenario_gc_thresholds() as thresholds, capture:
        result = run_scenario(scenario, retention_policy="aggregate",
                              control=ExecutionControl(progress_callback=report))
    if profiler:
        profiler.disable()
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile))
    elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    row = {**point, **collect_result(result, point), "wall_seconds": elapsed,
           "cpu_seconds": cpu_seconds, "finished_at": utc_now(),
           "python_executable": sys.executable,
           "implementation_root": str(IMPLEMENTATION_ROOT),
           "preloaded_modules": PRELOADED_MODULES,
           "transaction_capture": str(args.transactions.resolve()) if args.transactions else None,
           "source_paths": source_paths,
           "gc_thresholds": thresholds,
           "cohort_timings": cohort_timings,
           "decode_unprofiled_steps": sum(item["kind"] == "decode" and not item["profiled"]
                                          for item in cohort_timings),
           "decode_unprofiled_wall_seconds": sum(item["wall_seconds"] for item in cohort_timings
                                                if item["kind"] == "decode" and not item["profiled"]),
           "profiled": bool(profiler), "purpose": "runtime optimization benchmark; not a formal scan point"}
    row["benchmark_timing"] = timing_summary(row)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.result, row)
    print(json.dumps({"wall_seconds": elapsed, "cpu_seconds": cpu_seconds,
                      "ttft_ns": row["ttft_ns"], "tpot_ns": row["tpot_ns"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
