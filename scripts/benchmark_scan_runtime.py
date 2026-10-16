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
    args = parser.parse_args()
    point = {"architecture": args.architecture, "model_id": args.model,
             "prompt_tokens": args.prompt, "output_tokens": args.output}
    scenario = build_scan_scenario(args.model, args.architecture, args.prompt, args.output)
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    cpu_started = time.process_time()
    print("implementation loaded; starting full benchmark request", flush=True)
    def report(event):
        if event.stage in {"control_plane", "serving_cohorts"}:
            print(json.dumps({"stage": event.stage, "completed": event.completed,
                "wall_seconds": time.perf_counter() - started}), flush=True)
    if profiler:
        profiler.enable()
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
           "profiled": bool(profiler), "purpose": "runtime optimization benchmark; not a formal scan point"}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.result, row)
    print(json.dumps({"wall_seconds": elapsed, "cpu_seconds": cpu_seconds,
                      "ttft_ns": row["ttft_ns"], "tpot_ns": row["tpot_ns"]}), flush=True)


if __name__ == "__main__":
    main()
