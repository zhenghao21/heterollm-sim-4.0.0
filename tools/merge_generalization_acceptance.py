"""Merge manifest-defined matrix outputs without hiding failures.

This module is intentionally independent of native execution.  It consumes
cell summaries produced by the matrix runner and keeps planned, observed,
completed, structurally valid, and metric-eligible counts separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

try:
    from .generalization_acceptance_matrix import _expected_runtime_fingerprint, _same_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generalization_acceptance_matrix import _expected_runtime_fingerprint, _same_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/multimodel_next/blind_generalization_freeze_v5.json"
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
EXPECTED_SCHEMA = "blind-generalization-freeze/v5"
MATRIX_SCHEMA = "generalization-acceptance-matrix/v3"
MAJOR_GROUPS = ("model", "hardware", "migration")


def percentile(values, p):
    vals = sorted(float(x) for x in values if _finite(x))
    if not vals:
        return None
    position = (len(vals) - 1) * p / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    return vals[lower] if lower == upper else vals[lower] + (vals[upper] - vals[lower]) * (position - lower)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _scenario_key(row):
    try:
        return str(row.get("prompt_band")), str(row.get("output_band")), int(row.get("parallel"))
    except (AttributeError, TypeError, ValueError):
        return None


def _load_manifest(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"freeze manifest unreadable: {exc}")


def expected_cells_from_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("legacy or unsupported freeze manifest")
    matrix = manifest.get("matrix") if isinstance(manifest.get("matrix"), dict) else {}
    repeats = matrix.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("freeze repeats missing or invalid")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("freeze scenarios missing")
    expected = {}
    for model, rows in scenarios.items():
        if not isinstance(rows, list):
            raise ValueError(f"freeze scenarios malformed: {model}")
        for row in rows:
            key = _scenario_key(row)
            if key is None:
                raise ValueError(f"freeze scenario malformed: {model}")
            prompt, output, parallel = key
            for repeat in range(1, repeats + 1):
                cell_id = f"{model}__{prompt}__{output}__p{parallel}__r{repeat}"
                if cell_id in expected:
                    raise ValueError(f"duplicate freeze cell: {cell_id}")
                expected[cell_id] = (model, prompt, output, parallel, repeat, row)
    return expected, repeats


def _identity_value(cell, key):
    identity = cell.get("identity") if isinstance(cell, dict) else None
    if isinstance(identity, dict) and key in identity:
        return identity.get(key)
    return cell.get(key) if isinstance(cell, dict) else None


def _identity_errors(cell):
    errors = []
    identity = cell.get("identity") if isinstance(cell, dict) else None
    if not isinstance(identity, dict):
        errors.append("identity missing")
        identity = {}
    for key in ("runtime_fingerprint", "hardware_fingerprint", "gguf_sha256"):
        if not isinstance(identity.get(key), str) or not identity.get(key).strip():
            errors.append(f"identity.{key} missing")
    for key in ("gguf_sha256",):
        if identity.get(key) is not None and not _is_sha256(identity.get(key)):
            errors.append(f"identity.{key} malformed")
    for key in ("model_sha256", "binary_sha256"):
        if not _is_sha256(_identity_value(cell, key)):
            errors.append(f"{key} missing or malformed")
    if not isinstance(cell.get("config"), dict) or not cell.get("config"):
        errors.append("config missing")
    return errors


def _artifact_identities(artifacts):
    if not isinstance(artifacts, list) or not artifacts:
        return None
    result = {}
    for item in artifacts:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]
                or not _is_sha256(item.get("sha256"))):
            return None
        try:
            path = Path(item["path"]).resolve()
        except (OSError, ValueError):
            return None
        if path in result:
            return None
        result[path] = item["sha256"].lower()
    return result


def _manifest_identity_errors(cell, manifest, expected, manifest_path, freeze_sha256):
    """Compare observed identities with the freeze, never producer booleans."""
    errors = []
    if expected is None:
        return ["cell not declared by freeze"]
    model, _prompt, _output, parallel, _repeat, scenario = expected
    identity = cell.get("identity")
    identity = identity if isinstance(identity, dict) else {}
    evidence = cell.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    expected_model = (manifest.get("model_sha256") or {}).get(model)
    for field, wanted in (("model_sha256", expected_model), ("binary_sha256", manifest.get("binary_sha256"))):
        if field in identity and identity[field] != wanted:
            errors.append("manifest identity mismatch: identity." + field)
    observed_config = cell.get("observed_configuration")
    expected_config = {key: value for key, value in (scenario.get("configuration") or {}).items()
                       if key not in {"output_mode", "ignore_eos"}}
    expected_config["parallel"] = parallel
    if not isinstance(observed_config, dict) or any(observed_config.get(key) != value or key not in observed_config for key, value in expected_config.items()):
        errors.append("manifest identity mismatch: observed_configuration")
    identity_config = identity.get("configuration")
    expected_identity_config = {key: expected_config[key] for key in ("ctx", "parallel", "batch", "ubatch", "threads", "gpu_layers", "seed") if key in expected_config}
    expected_identity_config["flash_attn"] = expected_config.get("flash_attn", False)
    if not isinstance(identity_config, dict) or any(identity_config.get(key) != value or key not in identity_config for key, value in expected_identity_config.items()):
        errors.append("manifest identity mismatch: identity.configuration")
    if cell.get("observed") is not True or cell.get("completed") is not True:
        errors.append("valid cell is not observed and completed")
    if (cell.get("parallel_support") or {}).get("requested") != parallel:
        errors.append("requested parallel support mismatch")
    for key, actual, wanted in (
            ("model_sha256", cell.get("model_sha256"), expected_model),
            ("identity.gguf_sha256", identity.get("gguf_sha256"), expected_model),
            ("binary_sha256", cell.get("binary_sha256"), manifest.get("binary_sha256")),
            ("native_binary.sha256", (evidence.get("native_binary") or {}).get("sha256"), manifest.get("binary_sha256")),
            ("identity.hardware_fingerprint", identity.get("hardware_fingerprint"), manifest.get("hardware_fingerprint"))):
        if not _is_sha256(wanted) or not _is_sha256(actual) or actual.lower() != wanted.lower():
            errors.append("manifest identity mismatch: " + key)
    try:
        runtime_fingerprint = _expected_runtime_fingerprint(scenario["configuration"], parallel)
    except (KeyError, TypeError, ValueError):
        runtime_fingerprint = None
    if runtime_fingerprint is None or identity.get("runtime_fingerprint") != runtime_fingerprint:
        errors.append("manifest identity mismatch: identity.runtime_fingerprint")
    frozen_artifacts = _artifact_identities(manifest.get("runtime_artifacts"))
    actual_artifacts = _artifact_identities(evidence.get("runtime_artifacts"))
    if frozen_artifacts is None or actual_artifacts != frozen_artifacts:
        errors.append("manifest identity mismatch: runtime_artifacts")
    if evidence.get("runtime_stable") is not True:
        errors.append("runtime artifacts were not stable across native capture")
    if not _same_path(cell.get("freeze_manifest"), manifest_path):
        errors.append("cell freeze path mismatch")
    if not _is_sha256(freeze_sha256) or cell.get("freeze_sha256") != freeze_sha256:
        errors.append("cell freeze SHA mismatch")
    declaration = manifest.get("engine_semantic_proof")
    declaration = declaration if isinstance(declaration, dict) else {}
    bound_proof = cell.get("engine_semantic_proof")
    bound_proof = bound_proof if isinstance(bound_proof, dict) else {}
    captured_proof = evidence.get("engine_semantic_proof")
    captured_proof = captured_proof if isinstance(captured_proof, dict) else {}
    if (not _same_path(bound_proof.get("path"), declaration.get("path"))
            or not _same_path(captured_proof.get("path"), declaration.get("path"))
            or not _is_sha256(declaration.get("sha256"))
            or bound_proof.get("sha256") != declaration.get("sha256")
            or captured_proof.get("sha256") != declaration.get("sha256")
            or captured_proof.get("status") != "verified"):
        errors.append("manifest identity mismatch: engine_semantic_proof")
    counts = cell.get("request_counts")
    if not isinstance(counts, dict) or any(type(counts.get(side)) is not int or counts[side] != parallel for side in ("native", "simulator")):
        errors.append("observed request counts mismatch")
    checks = cell.get("checks") or {}
    for key in ("parallel_support", "native_request_count", "simulator_request_count", "native_request_ids", "simulator_request_ids", "request_set", "proof_manifest_binding"):
        if checks.get(key) is not True:
            errors.append("check:" + key)
    return errors


def _matrix_input_errors(payload, manifest_path, freeze_sha256):
    if not isinstance(payload, dict):
        return ["matrix input malformed"]
    errors = []
    if payload.get("schema") != MATRIX_SCHEMA:
        errors.append("matrix input schema mismatch")
    if payload.get("freeze_end_verified") is not True:
        errors.append("matrix input end freeze was not verified")
    if not _is_sha256(freeze_sha256) or payload.get("freeze_sha256") != freeze_sha256:
        errors.append("matrix input freeze SHA mismatch")
    if not _same_path(payload.get("freeze_manifest"), manifest_path):
        errors.append("matrix input freeze path mismatch")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        errors.append("matrix input cells missing or malformed")
    else:
        for cell in cells:
            if not isinstance(cell, dict) or not isinstance(cell.get("cell_id"), str) or not cell.get("cell_id"):
                errors.append("matrix input cell malformed")
            elif cell.get("freeze_sha256") != freeze_sha256:
                errors.append("cell freeze SHA mismatch: " + cell["cell_id"])
    return errors


def validate_cell(cell, *, expected=None, expected_contract_id="engine-boundary/v1", expected_boundary="engine",
                  manifest=None, manifest_path=None, freeze_sha256=None):
    """Validate one result cell while preserving every failure reason."""
    reasons = []
    if not isinstance(cell, dict):
        return False, ["cell malformed"]
    if cell.get("status") != "valid":
        reasons.append(f"status={cell.get('status')}")
    for field in ("freeze_manifest", "prediction_artifact", "prediction_sha256"):
        if not isinstance(cell.get(field), str) or not cell.get(field).strip():
            reasons.append(f"{field} missing")
    if expected is not None:
        model, prompt, output, parallel, repeat, scenario = expected
        for field, want in (("model_key", model), ("prompt_band", prompt), ("output_band", output), ("parallel", parallel), ("repeat", repeat)):
            if cell.get(field) != want:
                reasons.append(f"geometry:{field}")
        if cell.get("prompt") != scenario.get("prompt"):
            reasons.append("scenario prompt mismatch")
        if cell.get("requested_output_tokens") != scenario.get("requested_output_tokens"):
            reasons.append("scenario output mismatch")
        declared_config = scenario.get("configuration")
        if not isinstance(declared_config, dict) or cell.get("config") != declared_config:
            reasons.append("scenario configuration mismatch")
    checks = cell.get("checks") or {}
    for name in ("schema", "geometry", "tokens", "parallel", "prompt", "model_sha", "binary_sha", "stream",
                 "prediction_before_native", "output_policy", "configuration", "configuration_complete",
                 "engine_evidence", "engine_contract", "measurement_status", "engine_semantic_proof"):
        if checks.get(name) is not True:
            reasons.append(f"check:{name}")
    support = cell.get("parallel_support") or {}
    if support.get("status") != "modeled":
        reasons.append(f"parallel_support={support.get('status')}")
    for field in ("native_requests", "simulator_requests"):
        if support.get(field) != cell.get("parallel"):
            reasons.append(f"{field}_count")
    reasons.extend(_identity_errors(cell))
    if manifest is not None:
        reasons.extend(_manifest_identity_errors(cell, manifest, expected, manifest_path, freeze_sha256))
    metrics = cell.get("metrics")
    if not isinstance(metrics, dict):
        reasons.append("metrics missing")
    else:
        for metric in METRICS:
            record = metrics.get(metric)
            if not isinstance(record, dict):
                reasons.append(f"metric:{metric} missing")
                continue
            if record.get("boundary") != expected_boundary:
                reasons.append(f"metric:{metric} boundary")
            if record.get("contract_id") != expected_contract_id:
                reasons.append(f"metric:{metric} contract")
            status = record.get("status")
            if status not in ("measured", "not_applicable"):
                reasons.append(f"metric:{metric} evidence")
            if status == "measured" and (not _finite(record.get("native_ms")) or not _finite(record.get("simulator_ms"))
                                         or record["native_ms"] <= 0 or record["simulator_ms"] < 0):
                reasons.append(f"metric:{metric} numeric values missing")
    return not reasons, reasons


def _metric_summary(records, expected_repeats):
    measured = [record for record in records if isinstance(record, dict) and record.get("status") == "measured"
                and _finite(record.get("native_ms")) and _finite(record.get("simulator_ms"))]
    statuses = [record.get("status") if isinstance(record, dict) else None for record in records]
    if not measured:
        status = "not_applicable" if len(statuses) == expected_repeats and all(item == "not_applicable" for item in statuses) else "evidence_insufficient"
        return {"status": status, "n": 0, "expected_repeats": expected_repeats,
                "median_of_repeats_signed_pct": None, "median_of_repeats_abs_pct": None,
                "p90_abs_pct": None, "worst_abs_pct": None,
                "median_absolute_ms": None, "p90_absolute_ms": None, "worst_absolute_ms": None,
                "metric_eligible_repeats": 0}
    signed = [100.0 * (float(item["simulator_ms"]) - float(item["native_ms"])) / float(item["native_ms"])
              for item in measured if float(item["native_ms"]) != 0]
    abs_pct = [abs(value) for value in signed]
    abs_ms = [abs(float(item["simulator_ms"]) - float(item["native_ms"])) for item in measured]
    native = [float(item["native_ms"]) for item in measured]
    simulator = [float(item["simulator_ms"]) for item in measured]
    native_center, simulator_center = statistics.median(native), statistics.median(simulator)
    center_error = 100.0 * (simulator_center - native_center) / native_center if native_center else None
    return {"status": "measured" if len(measured) == expected_repeats else "incomplete",
            "n": len(measured), "expected_repeats": expected_repeats,
            "metric_eligible_repeats": len(measured),
            "median_of_repeats_signed_pct": center_error,
            "median_of_repeats_abs_pct": abs(center_error) if center_error is not None else None,
            "median_signed_pct": statistics.median(signed) if signed else None,
            "median_abs_pct": statistics.median(abs_pct) if abs_pct else None,
            "p90_abs_pct": percentile(abs_pct, 90), "worst_abs_pct": max(abs_pct) if abs_pct else None,
            "median_absolute_ms": statistics.median(abs_ms) if abs_ms else None,
            "p90_absolute_ms": percentile(abs_ms, 90), "worst_absolute_ms": max(abs_ms) if abs_ms else None,
            "native_median_ms": native_center, "simulator_median_ms": simulator_center,
            "repeat_variability": {
                "native_min_ms": min(native), "native_max_ms": max(native),
                "native_mad_ms": statistics.median(abs(value - native_center) for value in native),
                "simulator_min_ms": min(simulator), "simulator_max_ms": max(simulator),
                "simulator_mad_ms": statistics.median(abs(value - simulator_center) for value in simulator),
            }}


def aggregate(cells, expected_repeats):
    return {metric: _metric_summary([((cell.get("metrics") or {}).get(metric) or {}) for cell in cells], expected_repeats)
            for metric in METRICS}


def _group_stats(groups, expected_repeats, scenario_stats=None):
    result = {}
    for name, cells in groups.items():
        eligible = cells["eligible"]
        if scenario_stats is None:
            metrics = aggregate(eligible, expected_repeats)
        else:
            metrics = {}
            for metric in METRICS:
                entries = [scenario_stats[key]["metrics"][metric] for key in sorted(cells["scenarios"])]
                complete = [item for item in entries if item["status"] == "measured"]
                signed = [item["median_of_repeats_signed_pct"] for item in complete
                          if item["median_of_repeats_signed_pct"] is not None]
                absolute = [abs(value) for value in signed]
                deltas = [abs(item["simulator_median_ms"] - item["native_median_ms"]) for item in complete]
                metrics[metric] = {
                    "status": "measured" if entries and len(complete) == len(entries) else "evidence_insufficient",
                    "expected_scenarios": len(entries), "eligible_scenarios": len(complete),
                    "metric_eligible_repeats": sum(item["metric_eligible_repeats"] for item in entries),
                    "median_of_repeats_signed_pct": statistics.median(signed) if signed else None,
                    "median_of_repeats_abs_pct": statistics.median(absolute) if absolute else None,
                    "p90_abs_pct": percentile(absolute, 90), "worst_abs_pct": max(absolute) if absolute else None,
                    "median_absolute_ms": statistics.median(deltas) if deltas else None,
                    "p90_absolute_ms": percentile(deltas, 90), "worst_absolute_ms": max(deltas) if deltas else None,
                }
        result[name] = {
            "expected_cells": len(cells["expected"]), "observed_cells": len(cells["observed"]),
            "completed_cells": sum(cell.get("completed") is True for cell in cells["observed"]),
            "valid_cells": len(eligible), "missing_cells": len(cells["expected"]) - len(cells["observed"]),
            "invalid_cells": len(cells["observed"]) - len(eligible), "metrics": metrics,
        }
        result[name]["evidence_completeness"] = {
            metric: metrics[metric]["metric_eligible_repeats"] / len(cells["expected"]) if cells["expected"] else 0.0
            for metric in METRICS
        }
    return result


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _utc(value):
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo is not None else None
    except ValueError:
        return None


def _read_bound_artifact(ref, *, as_json=False):
    if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]
            or not _is_sha256(ref.get("sha256"))):
        raise ValueError("artifact path/SHA missing")
    try:
        data = Path(ref["path"]).read_bytes()
    except OSError as exc:
        raise ValueError("artifact unreadable: " + ref["path"]) from exc
    if not data or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("artifact empty or SHA mismatch: " + ref["path"])
    if not as_json:
        return None
    try:
        result = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact JSON malformed: " + ref["path"]) from exc
    if not isinstance(result, dict):
        raise ValueError("artifact JSON must be an object: " + ref["path"])
    return result


def _formal_protocol_errors(manifest, repeats):
    protocol = manifest.get("formal_evidence_protocol")
    if not isinstance(protocol, dict):
        return ["frozen repeat sampling/uncertainty/observer protocol missing"]
    errors = []
    if protocol.get("schema") != "generalization-formal-evidence-protocol/v1":
        errors.append("formal evidence protocol schema unsupported")
    if manifest.get("formal_evidence_protocol_sha256") != _canonical_sha(protocol):
        errors.append("formal evidence protocol SHA mismatch")
    registered, frozen = _utc(protocol.get("registered_utc")), _utc(manifest.get("created_utc"))
    if registered is None or frozen is None or registered > frozen:
        errors.append("protocol must be registered no later than freeze creation")
    sampling = protocol.get("sampling") if isinstance(protocol.get("sampling"), dict) else {}
    required = {"method": "fixed_repeats", "unit": "independent_native_run", "repeats": repeats,
                "optional_stopping": False, "additional_sampling": "new_freeze_required",
                "failure_policy": "retain_all_planned_cells"}
    for key, value in required.items():
        if key not in sampling or type(sampling[key]) is not type(value) or sampling[key] != value:
            errors.append("unsupported or missing sampling rule: " + key)
    uncertainty = protocol.get("uncertainty") if isinstance(protocol.get("uncertainty"), dict) else {}
    if uncertainty.get("method") != "exact_median_order_statistics" or uncertainty.get("multiple_comparisons") != "bonferroni":
        errors.append("uncertainty method/multiple-comparison rule missing or unsupported")
    level = uncertainty.get("family_confidence_level")
    if not _finite(level) or not 0.95 <= level < 1:
        errors.append("family confidence level must be at least 0.95 and below 1")
    width = uncertainty.get("max_relative_ci_width_pct")
    if not _finite(width) or width <= 0:
        errors.append("prespecified positive uncertainty width limit missing")
    observer = protocol.get("observer") if isinstance(protocol.get("observer"), dict) else {}
    for key, value in {"method": "paired_relative_difference_median", "equivalence_margin_pct": 5.0,
                       "order": "alternating_ab_ba", "collection_mode": "benchmark_only", "boundary": "engine"}.items():
        if observer.get(key) != value:
            errors.append("unsupported or missing observer rule: " + key)
    if type(observer.get("paired_repeats")) is not int or observer["paired_repeats"] < 1:
        errors.append("observer paired repeat count missing")
    return errors


def _exact_median_interval(values, confidence):
    """Conservative exact two-sided order-statistic CI; no normal approximation.

    Coverage of [X_(k), X_(n-k+1)] is 1 - 2 * P(Bin(n, .5) < k).
    If even the full sample range lacks requested coverage, a finite interval
    is unavailable; identical point estimates cannot turn that into certainty.
    """
    ordered = sorted(float(value) for value in values if _finite(value))
    n = len(ordered)
    if n != len(values) or n == 0 or not _finite(confidence) or not 0 < confidence < 1:
        return None
    tails, selected = 0, None
    for k in range(1, (n + 1) // 2 + 1):
        tails += math.comb(n, k - 1)
        coverage = 1.0 - (2 * tails) / (1 << n)
        if coverage < confidence:
            break
        selected = {"lower": ordered[k - 1], "upper": ordered[n - k],
                    "confidence_level": coverage, "requested_confidence_level": confidence,
                    "n": n, "order_statistic_rank": k}
    return selected


def _scenario_binding(model, scenario):
    return _canonical_sha({"model_key": model, **{key: scenario.get(key)
                           for key in ("prompt", "requested_output_tokens", "parallel", "configuration")}})


def _uncertainty_evidence(expected, valid_cells, repeats, protocol, scenario_rows, confidence):
    errors, intervals, upper_errors = [], {}, {}
    cells_by_scenario = {key: [] for key in scenario_rows}
    for cell in valid_cells:
        spec = expected[cell["cell_id"]]
        key = "|".join(map(str, spec[:4]))
        cells_by_scenario[key].append(cell)
    max_width = protocol["uncertainty"]["max_relative_ci_width_pct"]
    for key, cells in cells_by_scenario.items():
        intervals[key], upper_errors[key] = {}, {}
        for metric in METRICS:
            measured = [(cell.get("metrics") or {}).get(metric) or {} for cell in cells]
            if len(measured) != repeats or any(row.get("status") != "measured" for row in measured):
                errors.append(f"{key}/{metric}: all planned measured repeats are required")
                continue
            sides = {}
            for side in ("native", "simulator"):
                values = [row.get(side + "_ms") for row in measured]
                interval = _exact_median_interval(values, confidence)
                if interval is None or interval["lower"] <= 0:
                    errors.append(f"{key}/{metric}/{side}: finite positive median CI unavailable")
                    continue
                center = statistics.median(values)
                interval["relative_width_pct"] = 100.0 * (interval["upper"] - interval["lower"]) / center
                if interval["relative_width_pct"] > max_width:
                    errors.append(f"{key}/{metric}/{side}: prespecified uncertainty width exceeded")
                sides[side] = interval
            intervals[key][metric] = sides
            if set(sides) == {"native", "simulator"}:
                low = 100.0 * (sides["simulator"]["lower"] / sides["native"]["upper"] - 1.0)
                high = 100.0 * (sides["simulator"]["upper"] / sides["native"]["lower"] - 1.0)
                upper_errors[key][metric] = max(abs(low), abs(high))
    group_scenarios = {kind: {} for kind in MAJOR_GROUPS}
    for key, (model, scenario) in scenario_rows.items():
        for kind, name in (("model", model), ("hardware", str(scenario.get("hardware_fingerprint") or protocol["_hardware_fingerprint"])),
                           ("migration", str(scenario.get("migration_type") or "unverified_migration"))):
            group_scenarios[kind].setdefault(name, []).append(key)
    bounds = {kind: {} for kind in MAJOR_GROUPS}
    for kind, groups in group_scenarios.items():
        for name, keys in groups.items():
            bounds[kind][name] = {}
            for metric in METRICS:
                values = [upper_errors[key][metric] for key in keys if metric in upper_errors[key]]
                if len(values) != len(keys):
                    continue
                item = {"median_abs_pct_upper": statistics.median(values),
                        "p90_abs_pct_upper": percentile(values, 90), "worst_abs_pct_upper": max(values)}
                bounds[kind][name][metric] = item
                if item["median_abs_pct_upper"] >= 10 or item["p90_abs_pct_upper"] > 20 or item["worst_abs_pct_upper"] > 30:
                    errors.append(f"{kind}/{name}/{metric}: uncertainty bound does not satisfy accuracy thresholds")
    return {"status": "sufficient" if not errors else "evidence_insufficient", "reasons": errors,
            "method": "exact_median_order_statistics", "per_interval_confidence_level": confidence,
            "scenario_intervals": intervals, "major_group_error_upper_bounds": bounds}


def _observer_evidence(manifest, protocol, scenario_rows, confidence):
    errors, intervals = [], {}
    ref = manifest.get("observer_equivalence_evidence")
    try:
        evidence = _read_bound_artifact(ref, as_json=True)
    except ValueError as exc:
        return {"status": "evidence_insufficient", "reasons": ["frozen observer evidence: " + str(exc)]}
    for key, wanted in {
            "schema": "engine-observer-equivalence/v1", "protocol_sha256": manifest["formal_evidence_protocol_sha256"],
            "hardware_fingerprint": manifest.get("hardware_fingerprint"), "instrumented_binary_sha256": manifest.get("binary_sha256"),
            "engine_semantic_proof_sha256": (manifest.get("engine_semantic_proof") or {}).get("sha256"),
            "collection_mode": "benchmark_only", "boundary": "engine"}.items():
        if evidence.get(key) != wanted or wanted is None:
            errors.append("observer evidence identity mismatch: " + key)
    registered, first, last, created, frozen = (_utc(protocol.get("registered_utc")), _utc(evidence.get("first_observation_utc")),
                                              _utc(evidence.get("last_observation_utc")), _utc(evidence.get("created_utc")), _utc(manifest.get("created_utc")))
    if any(value is None for value in (registered, first, last, created, frozen)) or not registered <= first <= last <= created <= frozen:
        errors.append("observer observations must follow registration and precede the acceptance freeze")
    refs = evidence.get("source_artifacts")
    raw_pairs = {}
    if not isinstance(refs, list) or not refs:
        errors.append("observer raw source artifact references missing")
    else:
        for artifact in refs:
            try:
                raw = _read_bound_artifact(artifact, as_json=True)
            except ValueError as exc:
                errors.append("observer source: " + str(exc))
                continue
            if raw.get("schema") != "engine-observer-pairs/v1" or not isinstance(raw.get("pairs_by_scenario"), dict):
                errors.append("observer raw paired-observation schema malformed")
                continue
            for field, wanted in {"hardware_fingerprint": evidence.get("hardware_fingerprint"),
                                  "instrumented_binary_sha256": evidence.get("instrumented_binary_sha256"),
                                  "baseline_binary_sha256": (evidence.get("baseline_binary") or {}).get("sha256"),
                                  "engine_semantic_proof_sha256": evidence.get("engine_semantic_proof_sha256")}.items():
                if not _is_sha256(wanted) or raw.get(field) != wanted:
                    errors.append("observer raw identity mismatch: " + field)
            for key, pairs in raw["pairs_by_scenario"].items():
                if key not in scenario_rows or not isinstance(pairs, list):
                    errors.append("observer raw scenario/pairs malformed: " + str(key))
                else:
                    raw_pairs.setdefault(key, []).extend(pairs)
    try:
        _read_bound_artifact(evidence.get("baseline_binary"))
    except ValueError as exc:
        errors.append("observer baseline binary: " + str(exc))
    rows = evidence.get("scenarios")
    if not isinstance(rows, dict) or set(rows) != set(scenario_rows):
        errors.append("observer evidence must cover exactly all frozen scenarios")
        rows = rows if isinstance(rows, dict) else {}
    count = protocol["observer"]["paired_repeats"]
    for key, (model, scenario) in scenario_rows.items():
        row = rows.get(key)
        if not isinstance(row, dict):
            continue
        if (row.get("configuration_sha256") != _scenario_binding(model, scenario)
                or row.get("model_sha256") != (manifest.get("model_sha256") or {}).get(model)
                or row.get("runtime_fingerprint") != _expected_runtime_fingerprint(scenario["configuration"], scenario["parallel"])):
            errors.append(key + ": observer scenario identity mismatch")
        pairs = row.get("pairs")
        if key not in raw_pairs or raw_pairs[key] != pairs:
            errors.append(key + ": observer measurements differ from frozen raw paired observations")
        if not isinstance(pairs, list) or len(pairs) != count:
            errors.append(key + ": planned observer pair count mismatch")
            continue
        deltas = {metric: [] for metric in METRICS}
        malformed = False
        for index, pair in enumerate(pairs, 1):
            if not isinstance(pair, dict) or pair.get("repeat") != index or pair.get("order") != ("AB" if index % 2 else "BA"):
                malformed = True
                continue
            baseline, instrumented = pair.get("baseline_ms"), pair.get("instrumented_ms")
            if not isinstance(baseline, dict) or not isinstance(instrumented, dict):
                malformed = True
                continue
            for metric in METRICS:
                base, observed = baseline.get(metric), instrumented.get(metric)
                if not _finite(base) or base <= 0 or not _finite(observed) or observed <= 0:
                    malformed = True
                else:
                    deltas[metric].append(100.0 * (observed - base) / base)
        if malformed:
            errors.append(key + ": observer pair geometry/order/values malformed")
        intervals[key] = {}
        for metric, values in deltas.items():
            interval = _exact_median_interval(values, confidence) if len(values) == count else None
            intervals[key][metric] = interval
            if interval is None:
                errors.append(f"{key}/{metric}: observer equivalence CI unavailable")
            elif interval["lower"] < -5.0 or interval["upper"] > 5.0:
                errors.append(f"{key}/{metric}: observer CI is not contained within [-5%, +5%]")
    return {"status": "equivalent" if not errors else "evidence_insufficient", "reasons": errors,
            "equivalence_margin_pct": 5.0, "per_interval_confidence_level": confidence,
            "artifact": ref, "scenario_intervals": intervals}


def _formal_evidence_gate(manifest, expected, valid_cells, repeats, *, require_manifest_identity):
    errors = _formal_protocol_errors(manifest, repeats)
    if not require_manifest_identity:
        errors.append("formal evidence requires manifest-bound cell validation")
    if not expected:
        errors.append("formal evidence requires a nonempty planned matrix")
    if errors:
        return {"status": "evidence_insufficient", "reasons": errors,
                "sampling_status": "evidence_insufficient", "uncertainty": {"status": "evidence_insufficient"},
                "observer_equivalence": {"status": "evidence_insufficient"}}
    protocol = manifest["formal_evidence_protocol"]
    scenario_rows = {"|".join(map(str, spec[:4])): (spec[0], spec[5]) for spec in expected.values()}
    # Allocate the one frozen family error budget across native/simulator CIs
    # and observer paired-difference CIs (three intervals per scenario/metric).
    confidence = 1.0 - (1.0 - protocol["uncertainty"]["family_confidence_level"]) / (3 * len(scenario_rows) * len(METRICS))
    uncertainty = _uncertainty_evidence(expected, valid_cells, repeats,
                                      {**protocol, "_hardware_fingerprint": manifest.get("hardware_fingerprint")}, scenario_rows, confidence)
    observer = _observer_evidence(manifest, protocol, scenario_rows, confidence)
    errors = uncertainty["reasons"] + observer["reasons"]
    return {"status": "sufficient" if not errors else "evidence_insufficient", "reasons": errors,
            "sampling_status": "preregistered_fixed_repeats", "uncertainty": uncertainty,
            "observer_equivalence": observer}


def _write_report(manifest_path, manifest, expected, cells, repeats, output, markdown_output=None,
                  *, require_manifest_identity=False, freeze_sha256=None):
    """Synthetic fixtures may omit file bindings; the CLI always requires them."""
    by_id = {cell.get("cell_id"): cell for cell in cells if isinstance(cell, dict) and cell.get("cell_id")}
    missing_ids = sorted(set(expected) - set(by_id))
    unexpected_ids = sorted(set(by_id) - set(expected))
    duplicate_ids = sorted(cell_id for cell_id, count in _counts([cell.get("cell_id") for cell in cells]).items() if count > 1 and cell_id)
    valid, invalid = [], []
    for cell_id, cell in by_id.items():
        expected_spec = expected.get(cell_id)
        ok, reasons = validate_cell(cell, expected=expected_spec,
                                    manifest=manifest if require_manifest_identity else None,
                                    manifest_path=manifest_path, freeze_sha256=freeze_sha256)
        if cell_id in duplicate_ids:
            reasons.append("duplicate cell input")
        if cell_id not in expected:
            reasons.append("unexpected cell input")
        ok = ok and not reasons
        entry = dict(cell)
        entry["validation_reasons"] = reasons
        if ok:
            valid.append(entry)
        else:
            invalid.append({"cell_id": cell_id, "model_key": cell.get("model_key"), "reasons": reasons})
    observed = [cell for cell_id, cell in by_id.items() if cell_id in expected]
    valid = [cell for cell in valid if cell["cell_id"] in expected and cell["cell_id"] not in duplicate_ids]
    eligible_by_id = {cell["cell_id"]: cell for cell in valid}
    scenario_groups, model_groups, hardware_groups, migration_groups = {}, {}, {}, {}
    for cell_id, spec in expected.items():
        model, prompt, output_band, parallel, _repeat, scenario = spec
        scenario_key = f"{model}|{prompt}|{output_band}|{parallel}"
        hardware = str(scenario.get("hardware_fingerprint") or manifest.get("hardware_fingerprint") or "unknown")
        migration = str(scenario.get("migration_type") or "unverified_migration")
        for groups, key in ((scenario_groups, scenario_key), (model_groups, model),
                            (hardware_groups, hardware), (migration_groups, migration)):
            group = groups.setdefault(key, {"expected": [], "observed": [], "eligible": [], "scenarios": set()})
            group["expected"].append(cell_id)
            group["scenarios"].add(scenario_key)
            if cell_id in by_id:
                group["observed"].append(by_id[cell_id])
            if cell_id in eligible_by_id:
                group["eligible"].append(eligible_by_id[cell_id])
    scenario_stats = _group_stats(scenario_groups, repeats)
    groups = {"scenario": scenario_stats,
              "model": _group_stats(model_groups, repeats, scenario_stats),
              "hardware": _group_stats(hardware_groups, repeats, scenario_stats),
              "migration": _group_stats(migration_groups, repeats, scenario_stats)}
    report = {
        "schema": "generalization-acceptance-full/v3", "freeze_manifest": str(manifest_path),
        "freeze_sha256": freeze_sha256, "manifest_identity_required": require_manifest_identity,
        "models": list((manifest.get("matrix") or {}).get("models") or []),
        "prompt_bands": list((manifest.get("matrix") or {}).get("prompt_bands") or []),
        "output_bands": list((manifest.get("matrix") or {}).get("output_bands") or []),
        "parallel_values": list((manifest.get("matrix") or {}).get("parallel_values") or []),
        "repeats": repeats, "expected_cells": len(expected), "planned_cell_count": len(expected),
        "observed_cell_count": len(observed), "completed_cell_count": sum(cell.get("completed") is True for cell in observed),
        "valid_cell_count": len(valid), "invalid_cell_count": len(invalid), "missing_cell_count": len(missing_ids),
        "missing_cell_ids": missing_ids, "unexpected_cell_ids": unexpected_ids, "duplicate_cell_ids": duplicate_ids,
        "matrix_complete": not missing_ids and not unexpected_ids and not duplicate_ids,
        "all_cells_valid": bool(expected) and len(valid) == len(expected) and not invalid and not duplicate_ids,
        "metric_denominators": {
            metric: {"expected_cells": len(expected), "observed_cells": len(observed),
                     "completed_cells": sum(cell.get("completed") is True for cell in observed),
                     "valid_cells": len(valid),
                     "eligible_cells": sum((((cell.get("metrics") or {}).get(metric) or {}).get("status") == "measured") for cell in valid),
                     "not_applicable_cells": sum((((cell.get("metrics") or {}).get(metric) or {}).get("status") == "not_applicable") for cell in valid),
                     "incomplete_or_insufficient_cells": sum((((cell.get("metrics") or {}).get(metric) or {}).get("status") in {"incomplete", "evidence_insufficient"}) for cell in observed),
                     "missing_cells": len(missing_ids), "coverage": (sum((((cell.get("metrics") or {}).get(metric) or {}).get("status") == "measured") for cell in valid) / len(expected) if expected else 0.0)}
            for metric in METRICS
        },
        "groups": groups, "aggregation": groups["scenario"], "invalid_cells": invalid,
        "evidence_completeness": {kind: {name: {metric: stats["metrics"][metric]["metric_eligible_repeats"] / stats["expected_cells"] if stats["expected_cells"] else 0.0 for metric in METRICS} for name, stats in values.items()} for kind, values in groups.items()},
        "accuracy_pass": False, "coverage_pass": False,
        "evidence_pass": False, "overall_acceptance_status": "fail",
        "policy": "Manifest-defined aggregation. Missing, failed, incomplete, not_applicable, and evidence-insufficient cells remain in denominators; no result-driven removal is allowed.",
    }
    report["coverage_by_major_group"] = {
        kind: {name: dict(group["evidence_completeness"]) for name, group in groups[kind].items()}
        for kind in MAJOR_GROUPS
    }
    report["coverage_failures"] = [
        {"group_type": kind, "group": name, "metric": metric, "coverage": coverage}
        for kind, group_rows in report["coverage_by_major_group"].items()
        for name, metrics in group_rows.items() for metric, coverage in metrics.items() if coverage < 0.95
    ]
    report["coverage_pass"] = (bool(expected) and not report["coverage_failures"]
                               and all(item["coverage"] >= 0.95 for item in report["metric_denominators"].values()))
    scenario_metrics = [metric for kind in MAJOR_GROUPS for group in groups[kind].values() for metric in group["metrics"].values()
                        if metric.get("status") not in {"not_applicable"}]
    report["accuracy_pass"] = bool(scenario_metrics) and all(
        metric.get("status") == "measured" and metric.get("median_of_repeats_abs_pct") is not None
        and metric["median_of_repeats_abs_pct"] < 10.0
        and (metric.get("p90_abs_pct") is None or metric["p90_abs_pct"] <= 20.0)
        and (metric.get("worst_abs_pct") is None or metric["worst_abs_pct"] <= 30.0)
        for metric in scenario_metrics)
    formal_evidence = _formal_evidence_gate(manifest, expected, valid, repeats, require_manifest_identity=require_manifest_identity)
    report["formal_evidence"] = formal_evidence
    report["evidence_status"] = formal_evidence["status"]
    report["repeat_evidence_status"] = "screening_only" if repeats <= 3 else formal_evidence["sampling_status"]
    report["trend_evidence"] = {"status": "unverified", "reason": "No independent resource-scaling sweep is supplied; matrix accuracy cannot establish trend validity."}
    report["evidence_pass"] = (formal_evidence["status"] == "sufficient" and report["matrix_complete"]
                               and report["all_cells_valid"] and report["coverage_pass"])
    report["overall_acceptance_status"] = "pass" if report["accuracy_pass"] and report["coverage_pass"] and report["evidence_pass"] else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if markdown_output is None:
        markdown_output = output.with_suffix(".md")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return report


def _counts(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def report_metric_denominators_placeholder(report=None):
    # Kept as a tiny internal compatibility hook so older imports do not fail;
    # report construction immediately replaces its value with actual data.
    return []


def render_markdown(report):
    lines = ["# 泛化盲测验收表", "",
             f"- 计划/观察/完成/有效：{report['planned_cell_count']}/{report['observed_cell_count']}/{report['completed_cell_count']}/{report['valid_cell_count']}",
             f"- 缺失/无效/重复：{report['missing_cell_count']}/{report['invalid_cell_count']}/{len(report.get('duplicate_cell_ids') or [])}",
             f"- 矩阵完整：`{report['matrix_complete']}`；全部有效：`{report['all_cells_valid']}`；总体：`{report['overall_acceptance_status']}`",
             f"- 正式证据：`{report.get('evidence_status', 'evidence_insufficient')}`；采样协议：`{report.get('repeat_evidence_status')}`",
             "- 正式证据缺口：" + ("；".join((report.get("formal_evidence") or {}).get("reasons", [])[:12]) or "无"), "",
             "| 分组 | 指标 | 计划 | 观测 | 完成 | 有效 | metric eligible | 状态 | p50 signed % | p50 abs % | p90 abs % | worst abs % | abs ms p50 | abs ms p90 | abs ms worst |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("model", "hardware", "migration", "scenario"):
        for name in sorted((report.get("groups") or {}).get(kind, {})):
            group = report["groups"][kind][name]
            for metric in METRICS:
                item = group["metrics"][metric]
                fmt = lambda value: "—" if value is None else f"{float(value):.3f}"
                lines.append("| " + " | ".join((kind + ":" + name, metric, str(group["expected_cells"]), str(group["observed_cells"]), str(group["completed_cells"]), str(group["valid_cells"]), str(item.get("metric_eligible_repeats", 0)), str(item.get("status")), fmt(item.get("median_of_repeats_signed_pct")), fmt(item.get("median_of_repeats_abs_pct")), fmt(item.get("p90_abs_pct")), fmt(item.get("worst_abs_pct")), fmt(item.get("median_absolute_ms")), fmt(item.get("p90_absolute_ms")), fmt(item.get("worst_absolute_ms")))) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multimodel_next/generalization_acceptance_full_v3.json")
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    freeze_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = _load_manifest(manifest_path)
    expected, repeats = expected_cells_from_manifest(manifest)
    checker = ROOT / "tools/check_freeze_manifest.py"
    check = subprocess.run([sys.executable, str(checker), "--manifest", str(manifest_path), "--verify", "--phase", "end"], capture_output=True, text=True)
    if check.returncode:
        raise SystemExit("freeze verification failed: " + (check.stdout + check.stderr)[-1200:])
    rows = []
    seen_paths, seen_cells = set(), set()
    for path in args.input:
        resolved = path.resolve()
        if resolved in seen_paths or resolved == args.output.resolve():
            raise SystemExit("duplicate or self-referencing matrix input: " + str(path))
        seen_paths.add(resolved)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit("matrix input unreadable: " + str(path)) from exc
        errors = _matrix_input_errors(payload, manifest_path, freeze_sha)
        if errors:
            raise SystemExit("matrix input rejected: " + str(path) + ": " + "; ".join(errors))
        for cell in payload["cells"]:
            if cell["cell_id"] in seen_cells:
                raise SystemExit("duplicate cell input: " + cell["cell_id"])
            seen_cells.add(cell["cell_id"])
        rows.append(payload)
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != freeze_sha:
        raise SystemExit("freeze manifest changed while reading matrix inputs")
    cells = [cell for row in rows for cell in row["cells"]]
    report = _write_report(manifest_path, manifest, expected, cells, repeats, args.output, args.markdown_output,
                           require_manifest_identity=True, freeze_sha256=freeze_sha)
    print(json.dumps({"output": str(args.output.resolve()), "planned": report["planned_cell_count"], "observed": report["observed_cell_count"], "completed": report["completed_cell_count"], "valid": report["valid_cell_count"], "missing": report["missing_cell_count"], "status": report["overall_acceptance_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
