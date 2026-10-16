"""Time complete serving with optional temporary pre-optimization modules.

The baseline inputs are explicit test sources, never a simulator fallback.
Run before/after in separate processes to isolate imports and JIT dispatchers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def load_baseline(name, path, canonical):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.modules[canonical] = module
    import heterollm_sim
    setattr(heterollm_sim, canonical.rsplit(".", 1)[1], module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-numeric", type=Path)
    parser.add_argument("--baseline-packing", type=Path)
    parser.add_argument("--baseline-history-copy", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    baseline = (args.baseline_numeric, args.baseline_packing, args.baseline_history_copy)
    if any(baseline) and not all(baseline):
        parser.error("all three baseline sources must be provided together")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if all(baseline):
        load_baseline("heterollm_sim._hbm_numeric_baseline", args.baseline_numeric,
                      "heterollm_sim._hbm_numeric")
        load_baseline("heterollm_sim._compiled_hbm_baseline", args.baseline_packing,
                      "heterollm_sim._compiled_hbm")
        import heterollm_sim.data_motion as data_motion
        scope = {"__package__": "heterollm_sim"}
        exec(compile(args.baseline_history_copy.read_text(encoding="utf-8"),
                     str(args.baseline_history_copy), "exec"), scope)
        data_motion._copy_hbm_control_state = scope["_copy_hbm_control_state"]

    from hbm_ramulator2_study import base_configuration, serving
    from heterollm_sim.dram_core import DramCore
    from heterollm_sim.memory_types import AccessRequest

    # Exclude one-time Numba startup from both sets of timed serving runs.
    for profile in ("reference", "ramulator2"):
        DramCore(base_configuration(profile), capture_details=False).submit(
            AccessRequest("warmup", "read", 0, 4096))
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for profile in ("reference", "ramulator2"):
            for architecture in ("A_hbm_only", "B_all_kv_hbf"):
                wall, cpu = time.perf_counter(), time.process_time()
                row = serving(profile, architecture, "small", 128, 1, 4)
                cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                row.pop("wall_seconds")
                rows.append({"repeat": repeat, "profile": profile, "architecture": architecture,
                             "wall_seconds": wall, "cpu_seconds": cpu, "result": row})
                args.output.write_text(json.dumps({"baseline": bool(all(baseline)),
                    "planned_points": args.repeats * 4, "runs": rows},
                    ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                print(json.dumps({"repeat": repeat, "profile": profile, "architecture": architecture,
                                  "wall_seconds": wall, "cpu_seconds": cpu}), flush=True)


if __name__ == "__main__":
    main()
