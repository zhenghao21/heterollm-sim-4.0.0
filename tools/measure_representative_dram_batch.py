"""Compare complete 1/2 model graphs with only ordered DRAM batching toggled."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import gzip
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

from heterollm_sim import data_motion, reporting
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.dram_core import DramCore
from measure_physical_capture_equivalence import Difference


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/frontend_native_validation_2026-10-07"
RAW = ROOT.parent.parent / "_scratch/37-native-validation-raw/dram-batch-representative"
CASES = ("qwen3_30b_a3b", "qwen2_5_7b", "llama3_1_8b")


def normalized(value):
    if is_dataclass(value):
        return normalized(asdict(value))
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items() if key != "compiled_batch_float64"}
    if isinstance(value, (tuple, list)):
        return [normalized(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((normalized(item) for item in value), key=repr)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported comparison field {type(value).__name__}")


def physical_state(kernel):
    runtime = kernel.physical_runtime
    return normalized({
        "owner_states": {owner: {"clock_ns": item.clock_ns,
            "core": {key: value for key, value in vars(item.core).items() if key != "timeline"}}
            for owner, item in runtime.runtimes.items()},
        "timeline": vars(runtime.timeline),
        "allocators": {owner: allocator.snapshot() for owner, allocator in runtime.allocators.items()},
        "committed_owners": runtime._committed_owners,
        "l2": {owner: {"profile": item[0], "snapshot": item[1].snapshot()}
            for owner, item in kernel._l2_states.items()},
    })


def compare(slug, *, coverage_only=False):
    submission = REPORT / f"ui_b200_2hbf_{slug}_submission.json"
    payload = json.loads(submission.read_text(encoding="utf-8"))
    scenario = scenario_from_dict(payload["scenario"])
    request = replace(scenario.workload.requests[0], prompt_tokens=1, output_tokens=2)
    scenario = replace(scenario, workload=replace(scenario.workload,
        requests=(request,), prompt_tokens=1, output_tokens=2))
    results, clocks, batches = {}, {}, {}
    path_counts = {}
    original_bootstrap = reporting.bootstrap_control_plane
    RAW.mkdir(parents=True, exist_ok=True)
    if coverage_only:
        with gzip.open(RAW / f"{slug}_batch.json.gz", "rt", encoding="utf-8") as source:
            results[False] = json.load(source)
    for enabled in ((True,) if coverage_only else (False, True)):
        captured = []
        counters = {"submit_many_calls": 0, "requests_submitted_to_many": 0, "compiled_batch_results": 0}
        original_many = DramCore.submit_many
        def counted_many(core, requests):
            requests = tuple(requests)
            counters["submit_many_calls"] += 1
            counters["requests_submitted_to_many"] += len(requests)
            values = original_many(core, requests)
            counters["compiled_batch_results"] += sum(bool(value.counters.get("compiled_batch_float64")) for value in values)
            return values
        def bootstrap(config, **kwargs):
            result = original_bootstrap(config, **kwargs)
            captured.append(result)
            return result
        start = time.perf_counter()
        with patch.object(data_motion, "_USE_COMPILED_DRAM_BATCH", enabled), patch.object(reporting, "bootstrap_control_plane", bootstrap), patch.object(DramCore, "submit_many", counted_many):
            result = reporting.run_scenario(scenario, retention_policy="aggregate")
            report = reporting.report_dict(result)
        clocks[enabled] = time.perf_counter() - start
        kernel = captured[-1].kernel
        assert not kernel.physical_runtime.capture_details
        results[enabled] = {"complete_report": normalized(report), "physical_final_state": physical_state(kernel)}
        batches[enabled] = len(result.serving.batches)
        path_counts[enabled] = counters
        mode = "batch" if enabled else "serial"
        if not coverage_only:
            with gzip.open(RAW / f"{slug}_{mode}.json.gz", "wt", encoding="utf-8") as output:
                json.dump(results[enabled], output, ensure_ascii=False, separators=(",", ":"))
        print(f"{slug} {mode}: {clocks[enabled]:.3f}s; {batches[enabled]} cohorts; {counters}", flush=True)
    diff = Difference()
    diff.compare(results[False], results[True])
    if coverage_only:
        return {"counters": path_counts[True], "replay_against_saved_batch_report_and_final_state": diff.result(),
                "verification_wall_seconds": clocks[True],
                "interpretation": "Zero calls means this short generic path verifies non-triggering compatibility only; its wall-time variation must not be attributed to batching. Positive compiled results establish actual batched execution."}
    return {"model": slug, "submission_file": submission.name,
        "scope": "Complete original model and B200-2HBF hardware; only workload length changed to input 1/output 2. Current identical code toggles _USE_COMPILED_DRAM_BATCH False/True.",
        "comparison": "All report fields and complete DRAM/NAND core, physical timeline, allocator and L2 final state; only compiled_batch_float64 implementation marker omitted.",
        "serial_wall_seconds": clocks[False], "batch_wall_seconds": clocks[True],
        "wall_speedup": clocks[False] / clocks[True],
        "cohort_count": batches[True], "difference": diff.result(), "path_counts": path_counts,
        "raw_results_directory": str(RAW)}


if __name__ == "__main__":
    output = REPORT / "dram_batch_representative_equivalence.json"
    if "--coverage-only" in sys.argv:
        record = json.loads(output.read_text(encoding="utf-8"))
        for case in record["cases"]:
            case["active_path_verification"] = compare(case["model"], coverage_only=True)
            output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(0)
    record = {"scope": "Separate coverage for ordered DRAM submit_many only, not the earlier 63-case proof of three different optimizations.",
        "timing_context": "One serial and one batched run per case while UI simulations continue; wall times are indicative, not isolated throughput measurements.",
        "status": "running", "cases": []}
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for slug in CASES:
        record["cases"].append(compare(slug))
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record["status"] = "completed"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
