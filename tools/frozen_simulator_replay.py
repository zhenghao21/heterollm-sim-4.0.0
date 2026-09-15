"""Freeze and replay a declared diagnostic matrix without executing native LLMs.

This is development evidence. A new source freeze cannot restore the blind
qualification of an already revealed native dataset. All planned cells, failed
replays, immutable source records and prediction artifacts remain identifiable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from check_freeze_manifest import _source_files, TOKENIZER
from merge_generalization_acceptance import expected_cells_from_manifest, _group_stats, METRICS
from native_llama_compare import _hardware_fingerprint, native_extractor_identity, probe_hardware
from replay_simulator_from_native import replay, _validate_native_evidence, _is_sha256, PROFILES, _profile_binary_gate

SCHEMA = "simulator-replay-freeze/v1"


def sha(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty frozen input: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def source_identity():
    return {str(p.resolve()): sha(p) for p in _source_files()}


def check_files(identities):
    if not isinstance(identities, dict) or not identities:
        raise ValueError("empty frozen identity map")
    for path, expected in identities.items():
        if not _is_sha256(expected) or sha(path) != expected:
            raise ValueError(f"frozen input drift: {path}")


def bind_artifact(identities, artifact):
    if not isinstance(artifact, dict) or not artifact.get("path") or not _is_sha256(artifact.get("sha256")):
        raise ValueError("captured artifact lacks path or SHA")
    path = str(Path(artifact["path"]).resolve())
    if path in identities and identities[path] != artifact["sha256"]:
        raise ValueError(f"conflicting historical identities: {path}")
    identities[path] = artifact["sha256"]


def validate_native_cell(payload, spec, original):
    """Check scene identity without retroactively inventing missing proof SHA."""
    model, _prompt_band, _output_band, parallel, _repeat, scene = spec
    request, config = payload.get("request", {}), payload.get("configuration", {})
    if request.get("prompt") != scene.get("prompt"):
        raise ValueError("native prompt differs from planned scene")
    requested = scene.get("requested_output_tokens")
    if (request.get("requested_output_tokens") != requested or request.get("output_mode") != "fixed"
            or request.get("ignore_eos") is not True):
        raise ValueError("native output policy differs from planned scene")
    if config.get("parallel") != parallel:
        raise ValueError("native parallel differs from planned scene")
    for name, value in scene.get("configuration", {}).items():
        observed = request.get(name) if name in {"output_mode", "ignore_eos"} else config.get(name)
        if observed != value:
            raise ValueError("native configuration mismatch: " + name)
    identity = payload.get("identity", {})
    if identity.get("hardware_fingerprint") != original.get("hardware_fingerprint"):
        raise ValueError("native hardware differs from parent freeze")
    if identity.get("gguf_sha256") != original.get("model_sha256", {}).get(model):
        raise ValueError("native model differs from parent freeze")
    if payload.get("evidence", {}).get("native_binary", {}).get("sha256") != original.get("binary_sha256"):
        raise ValueError("native executable differs from parent freeze")
    def modules(items):
        result = {}
        for artifact in items:
            if not artifact.get("path") or not _is_sha256(artifact.get("sha256")):
                raise ValueError("runtime module identity incomplete")
            key = str(Path(artifact["path"]).resolve()).lower()
            if key in result:
                raise ValueError("duplicate runtime module identity")
            result[key] = artifact["sha256"]
        return result
    observed_modules = modules(payload.get("evidence", {}).get("runtime_artifacts", []))
    if not observed_modules or observed_modules != modules(original.get("runtime_artifacts", [])):
        raise ValueError("native actual module set differs from parent freeze")
    requests = payload.get("native", {}).get("requests", [])
    ids = [r.get("request_id") for r in requests]
    if (len(requests) != parallel or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != parallel):
        raise ValueError("native request set differs from planned parallel")
    if any(r.get("output_tokens") != requested or r.get("truncated") is True for r in requests):
        raise ValueError("native token count or truncation violates scene")
    support = payload.get("parallel_support", {})
    if support.get("native_requests") != parallel or support.get("requested") != parallel:
        raise ValueError("native request count evidence mismatch")


def validate_source_mapping(specs, expected, original):
    """A repeated run must correspond to a distinct raw capture."""
    used_sources, used_raw_paths, used_raw_hashes = set(), set(), set()
    for item in specs:
        cell_id = item.get("cell_id")
        if cell_id not in expected or item.get("model_key") != expected[cell_id][0]:
            raise ValueError("replay scene binding mismatch")
        if item.get("source") is None:
            continue
        source = item["source"]
        path = str(Path(source["path"]).resolve()).lower()
        if path in used_sources:
            raise ValueError("one native source reused as multiple repeats")
        if sha(source["path"]) != source.get("sha256"):
            raise ValueError("native source SHA mismatch")
        used_sources.add(path)
        payload = load(source["path"])
        validate_native_cell(payload, expected[cell_id], original)
        raw = payload.get("evidence", {}).get("raw_native_capture", {})
        if not raw.get("path") or not _is_sha256(raw.get("sha256")):
            raise ValueError("raw native capture identity missing")
        raw_path = str(Path(raw["path"]).resolve()).lower()
        if raw_path in used_raw_paths or raw["sha256"] in used_raw_hashes:
            raise ValueError("one raw native capture reused as multiple repeats")
        if sha(raw["path"]) != raw["sha256"]:
            raise ValueError("raw native capture SHA mismatch")
        used_raw_paths.add(raw_path)
        used_raw_hashes.add(raw["sha256"])


def create_freeze(parent_matrix, manifest_path, *, baseline_report=None):
    if Path(manifest_path).exists():
        raise FileExistsError("refusing to overwrite a replay freeze")
    parent = load(parent_matrix)
    original = load(parent["freeze_manifest"])
    baseline = load(baseline_report) if baseline_report is not None else None
    if baseline is not None:
        if (baseline.get("schema") != "derived-simulator-screening/v1" or baseline.get("freeze_end_verified") is not True
                or baseline.get("parent_matrix") != ref(parent_matrix) or baseline.get("native_execution_count") != 0):
            raise ValueError("baseline must be a closed replay on this same native parent")
    if sha(parent["freeze_manifest"]) != parent.get("freeze_sha256"):
        raise ValueError("parent freeze content identity mismatch")
    expected, repeats = expected_cells_from_manifest(original)
    rows = parent.get("cells", [])
    by_id = {r["cell_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) - set(expected):
        raise ValueError("duplicate or unexpected parent cell")
    files = {}
    for path in (parent_matrix, parent["freeze_manifest"], *([baseline_report] if baseline_report else [])):
        bind_artifact(files, ref(path))
    bind_artifact(files, original["engine_semantic_proof"])
    proof = load(original["engine_semantic_proof"]["path"])
    bind_artifact(files, proof["build_receipt"])
    receipt = load(proof["build_receipt"]["path"])
    if receipt.get("source_unchanged") is not True or not receipt.get("source_sha256_after"):
        raise ValueError("native build receipt source identity incomplete")
    for path, digest in receipt["source_sha256_after"].items():
        bind_artifact(files, {"path": path, "sha256": digest})
    bind_artifact(files, receipt["build_log"])
    bind_artifact(files, {"path": str(TOKENIZER), "sha256": original["tokenizer_sha256"]})
    hardware = probe_hardware()
    if _hardware_fingerprint(hardware) != original.get("hardware_fingerprint"):
        raise ValueError("actual hardware differs from native capture")
    extractor_sha = native_extractor_identity()["implementation_sha256"]
    profile_controls = {}
    for model, (path, stage, memory, phase) in PROFILES.items():
        profile_controls[model] = {"path": str(path) if path else None, "apply": [stage, memory, phase], "sha256": sha(path) if path and path.is_file() else None}
        if path and path.is_file():
            bind_artifact(files, ref(path))
    specs = []
    for cell_id, spec in expected.items():
        row = by_id.get(cell_id)
        item = {"cell_id": cell_id, "model_key": spec[0], "source": None}
        if row is not None and row.get("output") and Path(row["output"]).is_file():
            payload_path = Path(row["output"])
            payload = load(payload_path)
            validate_native_cell(payload, spec, original)
            captured = payload.get("evidence", {})
            if captured.get("extractor", {}).get("implementation_sha256") != extractor_sha:
                raise ValueError(f"extractor implementation differs: {cell_id}")
            identity = payload.get("identity", {})
            if identity.get("hardware_fingerprint") != original["hardware_fingerprint"]:
                raise ValueError(f"native hardware mismatch: {cell_id}")
            if identity.get("gguf_sha256") != original["model_sha256"][spec[0]]:
                raise ValueError(f"native model mismatch: {cell_id}")
            bind_artifact(files, {"path": payload["model"], "sha256": identity["gguf_sha256"]})
            for artifact in captured.get("runtime_artifacts", []):
                bind_artifact(files, artifact)
            if not captured.get("runtime_artifacts"):
                raise ValueError(f"native runtime module manifest absent: {cell_id}")
            bind_artifact(files, captured.get("raw_native_capture"))
            bind_artifact(files, {"path": payload.get("prediction_artifact"), "sha256": payload.get("prediction_sha256")})
            for path in (payload_path, payload_path.with_suffix(".native_raw.json"), payload_path.with_suffix(".prediction.json"), payload_path.with_suffix(".llama.log")):
                bind_artifact(files, ref(path))
            if captured.get("native_binary", {}).get("sha256") != original["binary_sha256"]:
                raise ValueError(f"native executable mismatch: {cell_id}")
            item["source"] = ref(payload_path)
        specs.append(item)
    validate_source_mapping(specs, expected, original)
    check_files(files)
    frozen = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "derived_development_screening", "independent_blind": False,
              "comparison_baseline": ref(baseline_report) if baseline_report else None,
              "parent_matrix": ref(parent_matrix), "parent_freeze": ref(parent["freeze_manifest"]),
              "source_sha256": source_identity(), "input_sha256": files,
              "native_extractor_sha256": extractor_sha, "hardware_snapshot": hardware,
              "hardware_fingerprint": _hardware_fingerprint(hardware),
              "calibration_mode": "analytical", "calibration_policy": "all_profiles_and_stage_memory_phase_overrides_disabled",
              "legacy_profile_controls": profile_controls,
              "expected_cells": specs, "planned_cell_count": len(expected), "repeats": repeats,
              "limits": {"native_execution_count": 0, "passes": 1},
              "acceptance_thresholds": {"median_abs_pct_lt": 10.0, "p90_abs_pct_le": 20.0, "worst_abs_pct_le": 30.0,
                                         "coverage_ge": 0.95, "evidence_completeness": 1.0},
              "limitations": ["Already revealed native evidence; cannot establish independent generalization.",
                              "Parent v7 collection tools drifted and observer equivalence is unresolved.",
                              "GGML_OP_OFFLOAD_MIN_BATCH was not captured for v7; runtime lowering uses an explicit conditional backend-default assumption."]}
    save_new(manifest_path, frozen)
    return frozen


def verify_freeze(path, *, full=True):
    frozen = load(path)
    if frozen.get("schema") != SCHEMA or frozen.get("calibration_mode") != "analytical":
        raise ValueError("replay freeze schema or calibration policy invalid")
    if frozen.get("independent_blind") is not False or not frozen.get("expected_cells"):
        raise ValueError("invalid replay scope")
    for name in ("parent_matrix", "parent_freeze"):
        item = frozen.get(name, {})
        if not item.get("path") or not _is_sha256(item.get("sha256")):
            raise ValueError("missing parent artifact identity")
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("parent artifact drift")
    expected, repeats = expected_cells_from_manifest(load(frozen["parent_freeze"]["path"]))
    specs = frozen["expected_cells"]
    if (len(specs) != len(expected) or {s.get("cell_id") for s in specs} != set(expected)
            or frozen.get("planned_cell_count") != len(expected) or frozen.get("repeats") != repeats):
        raise ValueError("frozen planned scenario denominator mismatch")
    current = source_identity()
    if not frozen.get("source_sha256") or current != frozen["source_sha256"]:
        raise ValueError("source file set or bytes changed during frozen replay")
    if native_extractor_identity()["implementation_sha256"] != frozen.get("native_extractor_sha256"):
        raise ValueError("extractor identity drift")
    if full:
        validate_source_mapping(specs, expected, load(frozen["parent_freeze"]["path"]))
        check_files(frozen.get("input_sha256"))
        if _hardware_fingerprint(probe_hardware()) != frozen.get("hardware_fingerprint"):
            raise ValueError("actual hardware changed")
    return frozen


def summaries(expected, records, original, repeats):
    by_id = {r["cell_id"]: r for r in records}
    buckets = {kind: {} for kind in ("scenario", "model", "hardware", "migration")}
    for cell_id, spec in expected.items():
        model, prompt, output, parallel, _, scene = spec
        sk = f"{model}|{prompt}|{output}|{parallel}"
        keys = {"scenario": sk, "model": model, "hardware": original["hardware_fingerprint"],
                "migration": "development_replay_" + str(scene.get("migration_type", "unverified"))}
        for kind, key in keys.items():
            g = buckets[kind].setdefault(key, {"expected": [], "observed": [], "eligible": [], "scenarios": set()})
            g["expected"].append(cell_id)
            g["scenarios"].add(sk)
            if cell_id in by_id:
                g["observed"].append(by_id[cell_id])
                if by_id[cell_id].get("status") == "valid":
                    g["eligible"].append(by_id[cell_id])
    groups = {"scenario": _group_stats(buckets["scenario"], repeats)}
    for kind in ("model", "hardware", "migration"):
        groups[kind] = _group_stats(buckets[kind], repeats, groups["scenario"])
    return groups


def accuracy_gate(groups):
    measurements = [m for kind in ("model", "hardware", "migration")
                    for group in groups[kind].values() for m in group["metrics"].values()]
    return bool(measurements) and all(
        m.get("status") == "measured" and m.get("median_of_repeats_abs_pct") is not None
        and m["median_of_repeats_abs_pct"] < 10 and m.get("p90_abs_pct") is not None
        and m["p90_abs_pct"] <= 20 and m.get("worst_abs_pct") is not None and m["worst_abs_pct"] <= 30
        for m in measurements)


def rescore_bound_prediction(record, spec, freeze_sha, prediction_path, expected_spec, original):
    from replay_simulator_from_native import score_saved_prediction
    if (record.get("cell_id") != spec["cell_id"] or record.get("freeze_sha256") != freeze_sha
            or record.get("source") != spec["source"]):
        raise ValueError("resume result identity mismatch")
    artifact = record.get("prediction_artifact")
    if (not isinstance(artifact, dict) or not artifact.get("path")
            or Path(artifact["path"]).resolve() != Path(prediction_path).resolve()
            or sha(prediction_path) != artifact.get("sha256")):
        raise ValueError("resume prediction missing, misbound or drifted")
    prediction = load(prediction_path)
    required = {"schema": "simulator-replay-prediction/v1", "calibration_mode": "analytical",
                "independent_blind_prediction": False, "freeze_sha256": freeze_sha,
                "cell_id": spec["cell_id"], "source_payload_sha256": spec["source"]["sha256"]}
    if prediction.get("independent_blind_prediction") is not False or any(prediction.get(key) != value for key, value in required.items()):
        raise ValueError("prediction content differs from frozen replay context")
    payload = load(spec["source"]["path"])
    if sha(spec["source"]["path"]) != spec["source"]["sha256"]:
        raise ValueError("resume native source changed")
    validate_native_cell(payload, expected_spec, original)
    errors, _ = _validate_native_evidence(payload, spec["source"]["path"])
    if errors:
        raise ValueError("resume native evidence: " + "; ".join(errors))
    result = score_saved_prediction(payload, prediction)
    fresh = dict(record)
    fresh.update(result)
    fresh["metrics"] = result.get("engine_evaluation", {}).get("metrics", {})
    fresh["status"] = "valid" if result.get("engine_evaluation", {}).get("status") == "measured" else "invalid"
    fresh["acceptance_eligible"] = False
    fresh["completed"] = True
    fresh["resumed_from_bound_prediction"] = True
    return fresh


def run_frozen(manifest_path, output, *, resume=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("completed report is immutable; use a new run id")
    frozen = verify_freeze(manifest_path)
    freeze_sha = sha(manifest_path)
    original = load(frozen["parent_freeze"]["path"])
    parent = load(frozen["parent_matrix"]["path"])
    baseline_ref = frozen.get("comparison_baseline")
    comparison = load(baseline_ref["path"]) if baseline_ref else parent
    parent_cells = {r["cell_id"]: r for r in comparison["cells"]}
    if len(parent_cells) != len(comparison["cells"]):
        raise ValueError("duplicate baseline cell")
    expected, repeats = expected_cells_from_manifest(original)
    artifacts = output.with_suffix("")
    artifacts.mkdir(parents=True, exist_ok=resume)
    rows, baselines, calibration_controls = [], [], {}
    end_error = None
    for spec in frozen["expected_cells"]:
        verify_freeze(manifest_path, full=False)
        if sha(manifest_path) != freeze_sha:
            raise ValueError("replay manifest changed")
        cell_id = spec["cell_id"]
        result_path = artifacts / (cell_id + ".json")
        prediction_path = artifacts / (cell_id + ".prediction.json")
        if result_path.exists():
            if not resume:
                raise FileExistsError(result_path)
            record = load(result_path)
            record = rescore_bound_prediction(record, spec, freeze_sha, prediction_path, expected[cell_id], original)
        else:
            record = {"cell_id": cell_id, "freeze_sha256": freeze_sha, "source": spec["source"],
                      "status": "invalid", "completed": False, "acceptance_eligible": False}
            try:
                if spec["source"] is None:
                    raise ValueError("planned native evidence missing")
                source = Path(spec["source"]["path"])
                if sha(source) != spec["source"]["sha256"]:
                    raise ValueError("source payload drift")
                payload = load(source)
                profile = frozen.get("legacy_profile_controls", {}).get(spec["model_key"], {})
                if profile.get("path") and profile.get("sha256"):
                    calibration_controls[cell_id] = _profile_binary_gate(profile["path"], payload)
                else:
                    calibration_controls[cell_id] = {"status": "not_available", "reasons": ["no admissible existing calibration control"]}
                record["legacy_calibration_control"] = calibration_controls[cell_id]
                errors, _ = _validate_native_evidence(payload, source)
                if errors:
                    raise ValueError("; ".join(errors))
                # Fresh prediction reads only static model/configuration/shape
                # and frozen generic costs. Native latency never enters costs.
                if prediction_path.exists():
                    if not resume:
                        raise FileExistsError(prediction_path)
                    record["prediction_artifact"] = ref(prediction_path)
                    result = rescore_bound_prediction(record, spec, freeze_sha, prediction_path, expected[cell_id], original)
                else:
                    result = replay(payload, spec["model_key"], source,
                                    calibration_mode="analytical", prediction_output=prediction_path,
                                    prediction_identity={"freeze_sha256": freeze_sha, "cell_id": cell_id})
                record.update({k: result.get(k) for k in ("status", "engine_evaluation", "native", "simulator",
                              "simulator_requests", "prediction_artifact", "profile_gate", "reason")})
                record["metrics"] = result.get("engine_evaluation", {}).get("metrics", {})
                record["completed"] = bool(result.get("simulator"))
                record["native_measurements_sha256"] = payload["evidence"].get("native_measurements_sha256")
                record["prediction_source"] = "conditional_mechanism_analysis"
                record["within_validated_domain"] = False
                record["evidence_level"] = "conditional_development_replay"
            except Exception as exc:
                record.update(status="invalid", reason=str(exc))
            save_new(result_path, record)
        rows.append(record)
        calibration_controls[cell_id] = record.get("legacy_calibration_control", {"status": "unavailable"})
        old = parent_cells.get(cell_id, {})
        baselines.append({"cell_id": cell_id, "status": "valid" if old.get("valid") is True or old.get("status") == "valid" else "invalid",
                          "completed": old.get("completed", False), "metrics": old.get("metrics", {})})
        print(json.dumps({"cell_id": cell_id, "status": record["status"], "done": len(rows)}, ensure_ascii=False), flush=True)
    try:
        verify_freeze(manifest_path)
        if sha(manifest_path) != freeze_sha:
            raise ValueError("replay manifest changed")
    except Exception as exc:
        end_error = str(exc)
        for row in rows:
            row.update(status="invalid", validation_reason="end_freeze: " + end_error)
    groups = summaries(expected, rows, original, repeats)
    baseline_groups = summaries(expected, baselines, original, repeats)
    accuracy_pass = accuracy_gate(groups)
    coverage = {kind: {k: v["evidence_completeness"] for k, v in groups[kind].items()} for kind in ("model", "hardware", "migration")}
    coverage_pass = all(v >= .95 for b in coverage.values() for ms in b.values() for v in ms.values())
    report = {"schema": "derived-simulator-screening/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "development_mechanism_comparison", "freeze_manifest": str(Path(manifest_path).resolve()),
              "freeze_sha256": freeze_sha, "freeze_end_verified": end_error is None, "freeze_error": end_error,
              "comparison_baseline": baseline_ref,
              "parent_matrix": frozen["parent_matrix"], "parent_freeze": frozen["parent_freeze"],
              "raw_input_sha256": frozen["input_sha256"], "source_sha256": frozen["source_sha256"],
              "native_execution_count": 0, "native_actual_reused_count": sum(r.get("status") == "valid" for r in rows),
              "planned_cell_count": len(expected), "observed_cell_count": len(rows),
              "completed_cell_count": sum(r.get("completed") is True for r in rows),
              "valid_cell_count": sum(r.get("status") == "valid" for r in rows),
              "legacy_calibration_controls": calibration_controls, "ablation_status": "analytical_baseline_vs_mechanism; legacy_profiles_checked_but_not_force_enabled",
              "groups": groups, "aggregation": groups["scenario"], "baseline_groups": baseline_groups,
              "coverage_by_major_group": coverage, "coverage_pass": coverage_pass, "accuracy_pass": accuracy_pass,
              "evidence_pass": False, "validated_domain_coverage": 0.0, "coverage_interpretation": "structurally_valid_development_records_only", "overall_acceptance_status": "fail" if not accuracy_pass else "evidence_insufficient",
              "independent_blind": False, "limitations": frozen["limitations"], "cells": rows,
              "invalid_cells": [{"cell_id": r["cell_id"], "reason": r.get("reason", r.get("validation_reason"))} for r in rows if r.get("status") != "valid"]}
    save_new(output, report)
    lines = ["# 冻结仿真回放：开发筛查", "", "该报告复用已经揭盲的 native actual；新冻结不恢复独立盲测资格。",
             f"\n计划/完成/有效：{len(expected)}/{report['completed_cell_count']}/{report['valid_cell_count']}；native 重测 0 次。",
             f"\n冻结结束核验：{report['freeze_end_verified']}；误差门槛：{accuracy_pass}；正式证据通过：False。", "",
             "| 模型 | 指标 | 原预测中位误差% | 新预测中位误差% | P90% | 最坏% | 绝对ms中位 |", "|---|---|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "缺证据" if x is None else f"{x:.3f}"
    for model, group in groups["model"].items():
        for metric, stat in group["metrics"].items():
            old = baseline_groups["model"][model]["metrics"][metric]
            vals = [old.get("median_of_repeats_abs_pct"), stat.get("median_of_repeats_abs_pct"), stat.get("p90_abs_pct"), stat.get("worst_abs_pct"), stat.get("median_absolute_ms")]
            lines.append("| " + " | ".join([model, metric] + [fmt(v) for v in vals]) + " |")
    with output.with_suffix(".md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--parent-matrix", type=Path)
    ap.add_argument("--baseline-report", type=Path)
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.create:
        if args.parent_matrix is None:
            ap.error("--create requires --parent-matrix")
        create_freeze(args.parent_matrix, args.manifest, baseline_report=args.baseline_report)
    if args.output:
        result = run_frozen(args.manifest, args.output, resume=args.resume)
        print(json.dumps({k: result[k] for k in ("valid_cell_count", "planned_cell_count", "freeze_end_verified", "overall_acceptance_status", "native_execution_count")}, ensure_ascii=False))
        return 0 if result["freeze_end_verified"] and not result["invalid_cells"] else 2
    if not args.create:
        verify_freeze(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
