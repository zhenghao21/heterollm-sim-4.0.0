"""Reduce an independent direct-launch/CUDA-Graph residual probe.

The probe input is intentionally a runtime microbenchmark record, never a
native LLM trace.  It is kept separate from ``native_llama_compare.py`` so a
TTFT/TPOT/E2E value cannot accidentally become a launch target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from heterollm_sim.runtime_residual import load_runtime_residual_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-id")
    parser.add_argument("--runtime-id")
    parser.add_argument("--architecture")
    parser.add_argument("--qualified", action="store_true",
                        help="only an independently predeclared holdout may set this")
    args = parser.parse_args()
    calibration = load_runtime_residual_calibration(
        args.input, hardware_id=args.hardware_id, runtime_id=args.runtime_id,
        architecture=args.architecture)
    if args.qualified and not calibration.qualified:
        raise SystemExit("input is not independently holdout-qualified")
    raw = json.loads(args.input.read_text(encoding="utf-8-sig"))
    cases = raw.get("samples", [])
    training = [row for row in cases if row.get("split", "training") != "holdout"]
    holdouts = [row for row in cases if row.get("split") == "holdout"]
    def median(values):
        return statistics.median(values) if values else None
    def cv(values):
        m = median(values)
        return (statistics.stdev(values) / m) if m and len(values) > 1 else None
    holdout_rows = []
    for row in holdouts:
        ordinary_actual = median(row["ordinary_durations_ns"])
        graph_actual = median(row["graph_durations_ns"])
        ordinary_pred = calibration.ordinary_launch_ns * row["kernel_count"]
        graph_pred = calibration.graph_replay_ns * row.get("replay_count", 1)
        holdout_rows.append({
            "kernel_count": row["kernel_count"],
            "ordinary_prediction_ns": ordinary_pred,
            "ordinary_actual_ns": ordinary_actual,
            "ordinary_ape_percent": 100.0 * abs(ordinary_pred - ordinary_actual) / ordinary_actual,
            "graph_prediction_ns": graph_pred,
            "graph_actual_ns": graph_actual,
            "graph_ape_percent": 100.0 * abs(graph_pred - graph_actual) / graph_actual,
        })
    source_path = Path(__file__).with_name("probe_runtime_residual.cu")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else None
    training_cv = {
        "ordinary_host_ns": [cv(row["ordinary_durations_ns"]) for row in training],
        "graph_host_ns": [cv(row["graph_durations_ns"]) for row in training],
    }
    module_sha_verified = bool(raw.get("module_sha256"))
    qualified = bool(
        source_sha
        and module_sha_verified
        and holdout_rows
        and all(row["ordinary_ape_percent"] <= 10 and row["graph_ape_percent"] <= 10 for row in holdout_rows)
        and all(value is not None and value <= 0.10 for values in training_cv.values() for value in values)
    )
    result = {
        "schema": "heterollm.runtime-residual-calibration/v1",
        "hardware_id": calibration.hardware_id,
        "runtime_id": calibration.runtime_id,
        "architecture": calibration.architecture,
        "ordinary_launch_ns": calibration.ordinary_launch_ns,
        "graph_replay_ns": calibration.graph_replay_ns,
        "qualified": qualified,
        "production_qualified": False,
        "acceptance": {
            "training_cases": len(training),
            "holdout_cases": len(holdouts),
            "training_host_cv": training_cv,
            "holdout": holdout_rows,
            "gate": "host CV <= 10% and ordinary/graph holdout APE <= 10%",
            "passed": qualified,
        },
        "validation_relative_error": calibration.validation_relative_error,
        "evidence": calibration.evidence,
        "source_kind": calibration.source_kind,
        "target_llm_latency_used": False,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "protocol": calibration.protocol,
        "source_sha256": source_sha,
        "module_identity": {
            "device": raw.get("hardware_id"),
            "runtime_version": raw.get("runtime_version"),
            "driver_version": raw.get("driver_version"),
            "sha256_verified": module_sha_verified,
        },
        "rejection_reason": None if qualified else "independent runtime residual holdout/CV or module identity gate failed",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
