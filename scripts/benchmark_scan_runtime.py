"""Measure full serving execution with real scan geometry and explicit lengths."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

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
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.reporting import run_scenario
from b200_hbm_hbf_scan import collect_result, json_default, scenario_gc_thresholds, utc_now, write_json
from b200_hbm_hbf_scan_config import build_scan_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("HBM", "HBF"), required=True)
    parser.add_argument("--prompt", type=int, default=32)
    parser.add_argument("--output", type=int, default=3)
    parser.add_argument("--model", default="qwen2_5-0_5b")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-decode-after", type=int,
                        help="Profile only after this many unprofiled Decode steps")
    args = parser.parse_args()
    if args.profile_decode_after is not None and (not args.profile or args.profile_decode_after < 1):
        parser.error("--profile-decode-after requires --profile and a positive step count")
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
    with scenario_gc_thresholds() as thresholds:
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
           "gc_thresholds": thresholds,
           "cohort_timings": cohort_timings,
           "decode_unprofiled_steps": sum(item["kind"] == "decode" and not item["profiled"]
                                          for item in cohort_timings),
           "decode_unprofiled_wall_seconds": sum(item["wall_seconds"] for item in cohort_timings
                                                if item["kind"] == "decode" and not item["profiled"]),
           "profiled": bool(profiler), "purpose": "runtime optimization benchmark; not a formal scan point"}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.result, row)
    print(json.dumps({"wall_seconds": elapsed, "cpu_seconds": cpu_seconds,
                      "ttft_ns": row["ttft_ns"], "tpot_ns": row["tpot_ns"]}), flush=True)


if __name__ == "__main__":
    main()
