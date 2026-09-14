"""Audit strict six-dimensional operator-key applicability (B-09).

This report is descriptive only.  It never fits a coefficient and never
converts a stage/phase aggregate into an exact-key entry.  Missing profile
entries are reported as analytical fallbacks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterollm_sim.calibration import NativeCalibrationProfile, audit_exact_operator_coverage, load_native_calibration


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_from(path: Path | None) -> NativeCalibrationProfile:
    if path is None:
        return NativeCalibrationProfile(kernel_stage_mapping={}, coverage_status="covered")
    return load_native_calibration(path)


def build_report(coverage: dict[str, Any], profile: NativeCalibrationProfile) -> dict[str, Any]:
    """Build a key-level gate report from an operator coverage artifact."""
    pairs = coverage.get("pairs") if isinstance(coverage.get("pairs"), list) else []
    exact_entries = {}
    mapping = profile.kernel_stage_mapping if isinstance(profile.kernel_stage_mapping, dict) else {}
    raw = mapping.get("exact_operator_keys") or mapping.get("_exact_operator_keys")
    if isinstance(raw, dict):
        exact_entries = raw
    pair_reports = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        train_keys = int(pair.get("train_joint_keys") or 0)
        holdout_keys = int(pair.get("holdout_joint_keys") or 0)
        intersection = int(pair.get("joint_intersection_keys") or 0)
        pair_reports.append({
            "train_joint_keys": train_keys,
            "holdout_joint_keys": holdout_keys,
            "holdout_keys_with_exact_train_evidence": intersection,
            "profile_exact_entries": len(exact_entries),
            "gate": "exact_only; all other keys analytical_fallback",
        })
    aggregate = coverage.get("aggregate") if isinstance(coverage.get("aggregate"), dict) else {}
    return {
        "schema": "b09-exact-operator-gate/v1",
        "scope": "development_only",
        "key_fields": ["stage", "phase", "shape", "dtype", "layout", "kernel_family"],
        "profile_exact_entries": len(exact_entries),
        "profile_coverage_status": profile.coverage_status,
        "pairs": pair_reports,
        "aggregate": {
            "union_joint_keys": int(aggregate.get("union_joint_keys") or 0),
            "keys_seen_in_train_and_holdout_pair": int(aggregate.get("keys_seen_in_train_and_holdout_pair") or 0),
        },
        "policy": {
            "missing_dimension": "analytical_fallback",
            "uncovered_key": "analytical_fallback",
            "cross_shape_average_rate": "forbidden",
            "interpolation": "forbidden_under_exact_policy",
        },
        "note": "No exact profile is enabled by this artifact; it documents the gate and sparse evidence only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(_load_json(args.coverage), _profile_from(args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
