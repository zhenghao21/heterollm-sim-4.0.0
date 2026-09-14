"""Build a stage-scoped train/holdout report from Nsight profile artifacts.

The report is deliberately conservative: only kernel-name stages with a
non-zero training count are calibrated.  Unknown kernels and missing KV
markers remain blocked instead of being distributed across simulator tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _num(value: object) -> float:
    return float(str(value or "0").replace(",", ""))


def _api(profile: dict, names: set[str]) -> tuple[float, int]:
    total = calls = 0.0
    for row in profile.get("stats", {}).get("api", {}).get("rows", []):
        if str(row.get("Name", "")) in names:
            total += _num(row.get("Total Time (ns)"))
            calls += _num(row.get("Num Calls"))
    return total, int(calls)


def _stage(profile: dict, stage: str) -> tuple[float, int]:
    aggregate = profile.get("kernel_stage_mapping", {}).get("aggregate", {})
    item = aggregate.get(stage, {})
    return _num(item.get("total_ns")), int(_num(item.get("instances")))


def _with_mapping(profile: dict, path: Path) -> dict:
    """Attach the sibling native-calibration mapping when input is raw profile."""
    if profile.get("kernel_stage_mapping"):
        return profile
    sibling = path.with_name(path.stem.replace("profile", "calibration") + ".json")
    if sibling.exists():
        calibration = json.loads(sibling.read_text(encoding="utf-8"))
        profile = dict(profile)
        profile["kernel_stage_mapping"] = calibration.get("kernel_stage_mapping", {})
        profile["_calibration_coverage"] = calibration.get("kernel_stage_mapping", {}).get("aggregate", {})
    return profile


def _holdout_stage(train: dict, holdout: dict, stage: str) -> dict:
    train_ns, train_n = _stage(train, stage)
    hold_ns, hold_n = _stage(holdout, stage)
    rate = train_ns / train_n if train_n else None
    predicted = rate * hold_n if rate is not None else None
    error = (predicted - hold_ns) / hold_ns * 100.0 if predicted is not None and hold_ns else None
    return {
        "status": "calibrated" if train_n and hold_n else "blocked_no_semantic_evidence",
        "train_total_ns": train_ns,
        "train_instances": train_n,
        "train_ns_per_instance": rate,
        "holdout_total_ns": hold_ns,
        "holdout_instances": hold_n,
        "holdout_predicted_ns": predicted,
        "holdout_relative_error_pct": error,
    }


def _holdout_api(train: dict, holdout: dict, name: str, names: set[str]) -> dict:
    train_ns, train_n = _api(train, names)
    hold_ns, hold_n = _api(holdout, names)
    rate = train_ns / train_n if train_n else None
    predicted = rate * hold_n if rate is not None else None
    error = (predicted - hold_ns) / hold_ns * 100.0 if predicted is not None and hold_ns else None
    return {
        "status": "calibrated" if train_n and hold_n else "blocked_no_api_evidence",
        "train_total_ns": train_ns,
        "train_calls": train_n,
        "train_ns_per_call": rate,
        "holdout_total_ns": hold_ns,
        "holdout_calls": hold_n,
        "holdout_predicted_ns": predicted,
        "holdout_relative_error_pct": error,
    }


def build(train_path: Path, holdout_path: Path) -> dict:
    train = _with_mapping(json.loads(train_path.read_text(encoding="utf-8")), train_path)
    holdout = _with_mapping(json.loads(holdout_path.read_text(encoding="utf-8")), holdout_path)
    stages = {stage: _holdout_stage(train, holdout, stage) for stage in ("attention_qkv", "ffn", "kv", "lm_head")}
    api = {
        "launch": _holdout_api(train, holdout, "launch", {"cudaLaunchKernel", "cudaLaunchKernelExC_v11060"}),
        "synchronize": _holdout_api(train, holdout, "synchronize", {"cudaStreamSynchronize"}),
    }
    unknown = train.get("stats", {}).get("kernel", {}).get("rows", [])
    return {
        "schema": "native-stage-holdout-calibration/v1",
        "train_profile": str(train_path.resolve()),
        "holdout_profile": str(holdout_path.resolve()),
        "model_sha256": (train.get("gguf", {}).get("gguf", {}) or {}).get("sha256"),
        "runtime": train.get("server_command"),
        "stages": stages,
        "api": api,
        "coverage": {
            "train_kernel_rows": len(train.get("stats", {}).get("kernel", {}).get("rows", [])),
            "holdout_kernel_rows": len(holdout.get("stats", {}).get("kernel", {}).get("rows", [])),
            "train_unknown_kernel_time_share": ((train.get("kernel_stage_mapping", {}).get("aggregate", {}) or {}).get("unknown", {}) or {}).get("share"),
            "holdout_unknown_kernel_time_share": ((holdout.get("kernel_stage_mapping", {}).get("aggregate", {}) or {}).get("unknown", {}) or {}).get("share"),
            "nvtx_operator_ranges": False,
            "note": "Current llama.cpp binary exposes CUPTI kernels/graph nodes but no NVTX activity; KV and most GEMM kernels remain unassigned.",
        },
        "calibration_policy": {
            "fit": "train stage total / train name-heuristic instance count",
            "holdout": "predicted rate * holdout name-heuristic instance count",
            "apply": "evidence-only until NVTX/CUPTI task-level markers cover the stage; do not apply unknown or KV rates",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train", type=Path)
    ap.add_argument("holdout", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = build(args.train, args.holdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
