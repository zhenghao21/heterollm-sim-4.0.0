"""Warm-JIT HBM numeric benchmarks, optionally checked against a saved module.

Example: python scripts/benchmark_hbm_numeric.py --mib 128 --repetitions 3
Use --baseline-source /temporary/path/_hbm_numeric.py for before/after checking.
The saved implementation is a diagnostic input, never a runtime fallback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim._hbm_numeric import run_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_presets import resolve_physical_config


def load_numeric(path, name="heterollm_sim._hbm_numeric_baseline"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_hbm_numeric


def execute(function, state, initial, requests):
    arrays = tuple(array.copy() for array in initial)
    started = time.perf_counter()
    cpu_started = time.process_time()
    results = [function(first, stop, arrival, read, state.g, state.t, *arrays)
               for first, stop, arrival, read in requests]
    cpu_seconds = time.process_time() - cpu_started
    return time.perf_counter() - started, results, arrays, cpu_seconds


def assert_same_bits(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def assert_equal(old, new):
    for expected, actual in zip(old[1], new[1], strict=True):
        for expected_value, actual_value in zip(expected, actual, strict=True):
            assert_same_bits(expected_value, actual_value)
    for expected, actual in zip(old[2], new[2], strict=True):
        assert_same_bits(expected, actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--arrival-ns", type=float, default=0.)
    parser.add_argument("--case", action="append", choices=("cold_read", "continued_read",
                        "cold_write", "continued_write", "alternating_row_conflicts"))
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.mib <= 8192 or args.repetitions < 1 or not math.isfinite(args.arrival_ns) or args.arrival_ns < 0:
        parser.error("mib must be 1..8192, repetitions positive, and arrival finite/nonnegative")
    baseline = load_numeric(args.baseline_source) if args.baseline_source else None
    numeric = load_numeric(args.candidate_source, "heterollm_sim._hbm_numeric_candidate") if args.candidate_source else run_hbm_numeric
    config = resolve_physical_config("hbm3e_timing_reference", overrides={
        "capacity_bytes": 512 * 1024 ** 3, "read_bandwidth_gb_s": 1600.,
        "write_bandwidth_gb_s": 1600., "interface_bandwidth_gb_s": 6553.6,
    })
    size = args.mib * 1024 ** 2
    state = _pack_hbm_numeric(DramCore(config, capture_details=False),
                              AccessRequest("benchmark", "read", 0, size))
    if state is None:
        raise RuntimeError("reference configuration did not enable the numeric controller")
    # Exclude first compilation/loading and array cloning from measured times.
    for function in (baseline, numeric):
        if function is not None:
            execute(function, state, state.arrays, [(0, 128, 0., True)])
    cases = []
    for read in (True, False):
        direction = "read" if read else "write"
        request = (0, size // 32, args.arrival_ns, read)
        if not args.case or "cold_" + direction in args.case:
            cases.append(("cold_" + direction, state.arrays, [request], size))
        if not args.case or "continued_" + direction in args.case:
            seed = execute(numeric, state, state.arrays, [request])
            if baseline:
                assert_equal(execute(baseline, state, state.arrays, [request]), seed)
            cases.append(("continued_" + direction, seed[2],
                          [(size // 32, 2 * size // 32, args.arrival_ns, read)], size))
    span = config.lane_count * config.bank_count * config.row_bytes // 32
    # Single bursts alternate two rows of one bank. The Python request loop is
    # included here; every request boundary and controller command is retained.
    conflicts = [((index % 2) * span, (index % 2) * span + 1, args.arrival_ns, index % 3 != 1)
                 for index in range(65536)]
    cases.append(("alternating_row_conflicts", state.arrays, conflicts, len(conflicts) * 32))
    report = {"profile": "hbm3e_timing_reference", "capacity_gib": 512,
              "clock_ns": config.command_clock_ns, "refresh_policy": config.refresh_policy,
              "repetitions": args.repetitions, "arrival_ns": args.arrival_ns,
              "baseline_source": str(args.baseline_source) if baseline else None,
              "candidate_source": str(args.candidate_source) if args.candidate_source else None, "cases": []}
    for name, initial, requests, byte_count in cases:
        if args.case and name not in args.case:
            continue
        old_times, new_times, old_cpu, new_cpu = [], [], [], []
        for repetition in range(args.repetitions):
            # Alternate order to reduce bias from CPU frequency/load changes.
            if baseline and repetition % 2:
                new = execute(numeric, state, initial, requests)
                old = execute(baseline, state, initial, requests)
            else:
                old = execute(baseline, state, initial, requests) if baseline else None
                new = execute(numeric, state, initial, requests)
            if old is not None:
                assert_equal(old, new)
                old_times.append(old[0])
                old_cpu.append(old[3])
            new_times.append(new[0])
            new_cpu.append(new[3])
        item = {"name": name, "physical_bytes": byte_count, "requests": len(requests),
                "numeric_seconds": new_times, "median_seconds": statistics.median(new_times),
                "numeric_cpu_seconds": new_cpu,
                "median_cpu_seconds": statistics.median(new_cpu),
                "row_hits": sum(result[1] for result in new[1]),
                "row_misses": sum(result[2] for result in new[1]),
                "row_conflicts": sum(result[3] for result in new[1])}
        if baseline:
            item.update(baseline_seconds=old_times,
                        speedup=statistics.median(old_times) / item["median_seconds"],
                        baseline_cpu_seconds=old_cpu,
                        cpu_speedup=statistics.median(old_cpu) / max(item["median_cpu_seconds"], 1e-12),
                        exact_request_results_and_all_arrays=True)
        report["cases"].append(item)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
