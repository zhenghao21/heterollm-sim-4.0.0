"""Run a manifest-defined, fail-closed generalization matrix.

The matrix separates planned, scheduled, observed, completed, structurally
valid, and metric-eligible cells.  It never removes failed or incomplete cells
from the denominator.  New acceptance runs require the immutable v5 freeze
manifest; legacy v1-v4 manifests are diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    from .unified_evidence_manifest import validate_prediction_before_native
    from .evaluation_contract import evaluate_metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unified_evidence_manifest import validate_prediction_before_native
    from evaluation_contract import evaluate_metrics

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe"
MODELS = {
    "qwen25": ROOT / "artifacts/multimodel_20260913/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen35": ROOT / "artifacts/multimodel_20260913/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    "qwen38": ROOT / "artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf",
    "tinyllama": ROOT / "artifacts/multimodel_20260913/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "smollm2": ROOT / "artifacts/multimodel_20260913/models/smollm2-1.7b-instruct-q4_k_m.gguf",
}
PROMPTS = {
    "short": "Hi.",
    "medium": "Explain how deterministic benchmarking affects reproducibility in language model inference.",
    "long": " ".join(["Benchmarking a language model requires measuring prompt evaluation and steady state decode separately while preserving identical model, runtime, hardware, and scheduling configuration."] * 8),
}
OUTPUTS = {"short": 8, "medium": 32, "long": 128}
PARALLEL = (1, 2, 4)
PROFILE = {
    "qwen25": (ROOT / "artifacts/multimodel_next/qwen25_medium_prompt8_boundary_profile_v3.json", True, False, True),
    "qwen35": (ROOT / "artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_calibration_v1.json", True, False, True),
    "qwen38": (ROOT / "artifacts/multimodel_next/qwen38_cpu_semantic_calibration_prompt8_f9_v2.json", True, True, False),
    "tinyllama": (ROOT / "artifacts/multimodel_next/tinyllama_semantic_calibration_launch8us_locked_v1.json", False, False, False),
    "smollm2": (None, False, False, False),
}
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
FREEZE_SCHEMA = "blind-generalization-freeze/v5"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _measured_metric(record):
    return (isinstance(record, dict) and record.get("status") == "measured"
            and _finite(record.get("native_ms")) and record["native_ms"] > 0
            and _finite(record.get("simulator_ms")) and record["simulator_ms"] >= 0)


def _same_path(left, right):
    if not isinstance(left, (str, Path)) or not isinstance(right, (str, Path)) or not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except (OSError, ValueError):
        return False


def _resolve_engine_semantic_proof(freeze, override=None):
    """Resolve a CLI override only when its path and bytes match the freeze."""
    declaration = freeze.get("engine_semantic_proof")
    if not isinstance(declaration, dict) or not isinstance(declaration.get("path"), str) or not declaration["path"]:
        raise ValueError("freeze engine semantic proof path missing")
    expected_sha = declaration.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("freeze engine semantic proof SHA missing or malformed")
    path = Path(declaration["path"]).resolve()
    if override is not None and not _same_path(override, path):
        raise ValueError("engine semantic proof override path differs from freeze")
    try:
        actual_sha = sha(path)
    except OSError as exc:
        raise ValueError("freeze engine semantic proof unreadable") from exc
    if actual_sha.lower() != expected_sha.lower():
        raise ValueError("engine semantic proof SHA differs from freeze")
    return path, actual_sha


def _expected_runtime_fingerprint(configuration, parallel):
    # Use the same typed runtime identity as the native producer. Fixed driver
    # defaults are explicit here; configurable values come from the freeze.
    try:
        from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))
        from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
    return LlamaCppRuntimeConfig(
        threads=configuration["threads"],
        threads_batch=configuration.get("threads_batch", configuration["threads"]),
        batch=configuration["batch"], ubatch=configuration["ubatch"],
        context=configuration["ctx"], parallel=parallel, gpu_layers=configuration["gpu_layers"],
        flash_attn=configuration.get("flash_attn", False),
        kv_type_k=configuration.get("kv_type_k", "f16"), kv_type_v=configuration.get("kv_type_v", "f16"),
        kv_unified=configuration.get("kv_unified", True),
        cont_batching=configuration.get("continuous_batching", True), warmup=configuration.get("warmup", True),
        seed=configuration["seed"], mmap=configuration.get("mmap", True), mlock=configuration.get("mlock", False),
        offload_kqv=configuration.get("offload_kqv", True), op_offload=configuration.get("op_offload", True),
        split_mode=configuration.get("split_mode", "layer"), main_gpu=configuration.get("main_gpu", 0),
        tensor_split=configuration.get("tensor_split"), device=configuration.get("device"),
        cpu_range=configuration.get("cpu_range"), cpu_range_batch=configuration.get("cpu_range_batch"),
        numa=configuration.get("numa"),
    ).fingerprint


def _request_set_checks(payload, parallel):
    support = payload.get("parallel_support")
    support = support if isinstance(support, dict) else {}
    checks = {"parallel_support": support.get("status") == "modeled" and support.get("requested") == parallel}
    request_ids = {}
    for side in ("native", "simulator"):
        part = payload.get(side)
        records = part.get("requests") if isinstance(part, dict) else None
        count = support.get(side + "_requests")
        checks[side + "_request_count"] = (isinstance(records, list) and len(records) == parallel
                                             and isinstance(count, int) and not isinstance(count, bool)
                                             and count == parallel
                                             and all(isinstance(item, dict) for item in records))
        ids = [item.get("request_id") for item in records if isinstance(item, dict)] if isinstance(records, list) else []
        request_ids[side] = ids
        checks[side + "_request_ids"] = (len(ids) == parallel
                                           and all(isinstance(item, str) and item for item in ids)
                                           and len(set(item for item in ids if isinstance(item, str))) == parallel)
    checks["request_set"] = (checks["native_request_ids"] and checks["simulator_request_ids"]
                                and set(request_ids["native"]) == set(request_ids["simulator"]))
    return checks


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    pos = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _scenario_key(item):
    return (str(item.get("prompt_band")), str(item.get("output_band")), int(item.get("parallel")))


def planned_cells(models, repeats, scenarios=None):
    """Yield exactly the cells declared by the supplied freeze scenarios."""
    scenarios = scenarios or {}
    for model in models:
        declared = scenarios.get(model)
        allowed = {_scenario_key(item) for item in declared if isinstance(item, dict)} if declared is not None else None
        for prompt_band in PROMPTS:
            for output_band in OUTPUTS:
                for parallel in PARALLEL:
                    if allowed is not None and (prompt_band, output_band, parallel) not in allowed:
                        continue
                    for repeat in range(1, repeats + 1):
                        yield f"{model}__{prompt_band}__{output_band}__p{parallel}__r{repeat}", model, prompt_band, output_band, parallel, repeat


def _freeze_errors(freeze, *, selected_models, repeats):
    errors = []
    if not isinstance(freeze, dict):
        return ["freeze manifest missing or malformed"]
    if freeze.get("schema") != FREEZE_SCHEMA:
        errors.append("legacy or unsupported freeze schema")
    if freeze.get("status") not in {"frozen_not_run", "running"}:
        errors.append(f"freeze status is not runnable: {freeze.get('status')}")
    matrix = freeze.get("matrix") if isinstance(freeze.get("matrix"), dict) else {}
    if matrix.get("models") != list(MODELS):
        errors.append("freeze model declaration mismatch")
    if matrix.get("repeats") != repeats:
        errors.append("freeze repeats mismatch")
    if matrix.get("prompt_bands") != list(PROMPTS):
        errors.append("freeze prompt bands mismatch")
    if matrix.get("output_bands") != list(OUTPUTS):
        errors.append("freeze output bands mismatch")
    if matrix.get("parallel_values") != list(PARALLEL):
        errors.append("freeze parallel declaration mismatch")
    if set(selected_models) - set(MODELS):
        errors.append("unknown model selected")
    scenarios = freeze.get("scenarios")
    if not isinstance(scenarios, dict):
        errors.append("freeze scenarios missing")
    else:
        for model in selected_models:
            rows = scenarios.get(model)
            if not isinstance(rows, list) or not rows:
                errors.append(f"freeze scenarios missing: {model}")
            elif len({_scenario_key(row) for row in rows if isinstance(row, dict)}) != len(rows):
                errors.append(f"duplicate or malformed scenarios: {model}")
    source_sha = freeze.get("source_sha256")
    if not isinstance(source_sha, dict) or not source_sha:
        errors.append("freeze source_sha256 missing")
    return errors


def _load_freeze(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"freeze manifest unreadable: {exc}")


def _scenario_row(freeze, model, prompt_band, output_band, parallel):
    for row in (freeze.get("scenarios") or {}).get(model, []):
        if isinstance(row, dict) and _scenario_key(row) == (prompt_band, output_band, parallel):
            return row
    raise KeyError(f"scenario missing from freeze: {model}/{prompt_band}/{output_band}/p{parallel}")


def _metric_denominators(cells, expected_count):
    result = {}
    for metric in METRICS:
        statuses = []
        for cell in cells:
            record = (cell.get("metrics") or {}).get(metric) or {}
            if cell.get("status") != "valid":
                statuses.append("incomplete")
            elif _measured_metric(record):
                statuses.append("measured")
            else:
                statuses.append(record.get("status") if record.get("status") != "measured" else "evidence_insufficient")
        result[metric] = {
            "expected_cells": int(expected_count),
            "observed_cells": sum(cell.get("observed") is True for cell in cells),
            "completed_cells": sum(cell.get("completed") is True for cell in cells),
            "valid_cells": sum(cell.get("status") == "valid" for cell in cells),
            "metric_eligible_cells": sum(status == "measured" for status in statuses),
            "not_applicable_cells": sum(status == "not_applicable" for status in statuses),
            "incomplete_cells": sum(status in {"incomplete", "evidence_insufficient"} for status in statuses),
            "missing_cells": max(0, int(expected_count) - len(cells)),
            "coverage": (sum(status == "measured" for status in statuses) / expected_count if expected_count else 0.0),
        }
    return result


def metric_records(payload):
    native = payload.get("native") or {}
    simulator = payload.get("simulator") or {}
    evaluation = evaluate_metrics(native, simulator, boundary="engine", aggregation="p50")
    metrics = evaluation["metrics"]
    for metric, entry in metrics.items():
        entry.update(boundary=evaluation["boundary"], contract_id=evaluation["contract_id"],
                     aggregation_policy=evaluation["aggregation_policy"],
                     native_field=evaluation["native_field"][metric],
                     simulator_field=evaluation["simulator_field"][metric])
    return metrics


def _expected_output_policy(output_mode, output):
    return {"mode": output_mode, "ignore_eos": output_mode == "fixed", "requested_output_tokens": int(output)}


def _policy_matches(payload, *, output_mode, output):
    request = payload.get("request") or {}
    policy = payload.get("output_policy") or {}
    expected = _expected_output_policy(output_mode, output)
    return (request.get("requested_output_tokens") == expected["requested_output_tokens"]
            and request.get("output_mode") == expected["mode"]
            and policy.get("mode") == expected["mode"]
            and policy.get("ignore_eos") == expected["ignore_eos"])


def validate_payload(payload, *, model, prompt, output, parallel, model_sha, binary_sha,
                     output_mode="fixed", source_path=None, expected_config=None,
                     engine_semantic_proof=None, engine_semantic_proof_sha256=None):
    checks = {}
    config = payload.get("configuration") or {}
    checks["schema"] = payload.get("schema") == "native-simulator-comparison/v2"
    checks["geometry"] = bool((payload.get("parity") or {}).get("geometry", {}).get("ok"))
    checks["tokens"] = bool((payload.get("parity") or {}).get("tokens", {}).get("ok"))
    checks["parallel"] = config.get("parallel") == parallel
    checks.update(_request_set_checks(payload, parallel))
    checks["prompt"] = (payload.get("request") or {}).get("prompt") == prompt
    checks["model_sha"] = ((payload.get("gguf") or {}).get("gguf") or {}).get("sha256") == model_sha
    captured_binary = ((payload.get("evidence") or {}).get("native_binary") or {}).get("sha256")
    checks["binary_sha"] = isinstance(captured_binary, str) and captured_binary.lower() == str(binary_sha).lower()
    checks["output_policy"] = _policy_matches(payload, output_mode=output_mode, output=output)
    checks["stream"] = payload.get("request", {}).get("request_timing", "stream") == "stream" or payload.get("validity_status") == "request_boundary_aligned"
    checks["prediction_before_native"] = not validate_prediction_before_native(payload, source_path=source_path, require=True)
    evidence = payload.get("evidence") or {}
    timing = evidence.get("engine_timing") or {}
    checks["engine_evidence"] = timing.get("status") in {"counter_proven", "marker_proven"}
    checks["engine_contract"] = (evidence.get("timing_contract") or {}).get("id") == "engine-stage+client-real-token/v3"
    proof_evidence = evidence.get("engine_semantic_proof") or {}
    proof_declared = proof_evidence.get("status")
    checks["engine_semantic_proof"] = proof_declared == "verified" if engine_semantic_proof else proof_declared in {"verified", None} and timing.get("status") == "marker_proven"
    if engine_semantic_proof is not None:
        try:
            proof_sha = sha(Path(engine_semantic_proof))
        except OSError:
            proof_sha = None
        checks["proof_manifest_binding"] = (
            _same_path(proof_evidence.get("path"), engine_semantic_proof)
            and isinstance(engine_semantic_proof_sha256, str)
            and proof_sha == engine_semantic_proof_sha256
            and proof_evidence.get("sha256") == engine_semantic_proof_sha256)
    else:
        # Legacy synthetic fixtures can omit a proof, but main always supplies
        # the path and SHA resolved against the manifest before any execution.
        checks["proof_manifest_binding"] = engine_semantic_proof_sha256 is None
    checks["measurement_status"] = checks["engine_evidence"] and payload.get("validity_status") == "request_boundary_aligned"
    if expected_config is not None:
        policy_keys = {"output_mode", "ignore_eos"}
        config_expected = {key: value for key, value in expected_config.items() if key not in policy_keys}
        checks["configuration"] = all(config.get(key) == value for key, value in config_expected.items())
        checks["configuration_complete"] = all(key in config and config.get(key) == value for key, value in config_expected.items())
    else:
        checks["configuration"] = True
        checks["configuration_complete"] = True
    return checks, all(checks.values())


def _group_template(planned):
    groups = {}
    for _, model, prompt, output, parallel, _ in planned:
        groups.setdefault((model, prompt, output, parallel), {"expected_repeats": 0, "cells": []})["expected_repeats"] += 1
    return groups


def _aggregate_group(cells, expected_repeats):
    result = {
        "expected_repeats": expected_repeats,
        "observed_repeats": sum(cell.get("observed") is True for cell in cells),
        "completed_repeats": sum(c.get("completed") is True for c in cells),
        "valid_repeats": sum(c.get("status") == "valid" for c in cells),
        "metric_eligible_repeats": {},
        "metrics": {},
    }
    for metric in METRICS:
        records = [((cell.get("metrics") or {}).get(metric) or {}) for cell in cells if cell.get("status") == "valid"]
        eligible = [record for record in records if _measured_metric(record)]
        statuses = [record.get("status") for record in records]
        result["metric_eligible_repeats"][metric] = len(eligible)
        if not eligible:
            status = "not_applicable" if len(statuses) == expected_repeats and all(value == "not_applicable" for value in statuses) else "evidence_insufficient"
            result["metrics"][metric] = {"status": status, "n": 0, "expected_repeats": expected_repeats,
                                          "median_of_repeats_signed_pct": None, "median_of_repeats_abs_pct": None,
                                          "p90_abs_pct": None, "worst_abs_pct": None, "median_absolute_ms": None,
                                          "p90_absolute_ms": None, "worst_absolute_ms": None}
            continue
        signed = [100.0 * (float(record["simulator_ms"]) - float(record["native_ms"])) / float(record["native_ms"]) for record in eligible]
        absolute = [abs(value) for value in signed]
        delta = [abs(float(record["simulator_ms"]) - float(record["native_ms"])) for record in eligible]
        native = [float(record["native_ms"]) for record in eligible]
        simulator = [float(record["simulator_ms"]) for record in eligible]
        native_center, simulator_center = statistics.median(native), statistics.median(simulator)
        center_error = 100.0 * (simulator_center - native_center) / native_center if native_center else None
        result["metrics"][metric] = {
            "status": "measured" if len(eligible) == expected_repeats else "incomplete",
            "n": len(eligible), "expected_repeats": expected_repeats,
            "median_of_repeats_signed_pct": center_error,
            "median_of_repeats_abs_pct": abs(center_error) if center_error is not None else None,
            "median_signed_pct": statistics.median(signed), "median_abs_pct": statistics.median(absolute),
            "p90_abs_pct": percentile(absolute, 90), "worst_abs_pct": max(absolute),
            "median_absolute_ms": statistics.median(delta), "p90_absolute_ms": percentile(delta, 90),
            "worst_absolute_ms": max(delta), "native_median_ms": native_center,
            "simulator_median_ms": simulator_center,
        }
    return result


def _verify_end(verifier, freeze_path, freeze_sha):
    try:
        check = subprocess.run([sys.executable, str(verifier), "--manifest", str(freeze_path), "--verify", "--phase", "end"],
                               capture_output=True, text=True)
        verified = check.returncode == 0 and sha(freeze_path) == freeze_sha
        error = (check.stdout + check.stderr)[-2000:]
        if not verified and not error:
            error = "freeze end verification failed or manifest SHA changed"
        return verified, error
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)


def _summarize_after_end_gate(cells, all_planned, verified, error=""):
    if not verified:
        for cell in cells:
            cell["status"] = "invalid"
            cell["valid"] = False
            cell.setdefault("checks", {})["freeze_end"] = False
            cell["freeze_end_error"] = error
    aggregation = {}
    for key, template in _group_template(all_planned).items():
        selected_cells = [cell for cell in cells if (cell.get("model_key"), cell.get("prompt_band"), cell.get("output_band"), cell.get("parallel")) == key]
        aggregation["|".join(map(str, key))] = _aggregate_group(selected_cells, template["expected_repeats"])
    summary = {
        "freeze_end_verified": verified,
        "observed_cell_count": sum(cell.get("observed") is True for cell in cells),
        "completed_cell_count": sum(cell.get("completed") is True for cell in cells),
        "valid_cell_count": sum(cell.get("status") == "valid" for cell in cells),
        "invalid_cell_count": sum(cell.get("status") not in {"valid", "planned"} for cell in cells),
        "metric_denominators": _metric_denominators(cells, len(all_planned)),
        "aggregation": aggregation,
    }
    if not verified:
        summary["freeze_end_error"] = error
    return summary


def _build_command(model, prompt, output, parallel, configuration, profile, proof_path, path):
    command = [sys.executable, str(ROOT / "tools/native_llama_compare.py"), "--exe", str(BINARY),
               "--model", str(MODELS[model]), "--prompt", prompt, "--predict", str(output),
               "--ctx", str(configuration["ctx"]), "--parallel", str(parallel),
               "--batch", str(configuration["batch"]), "--ubatch", str(configuration["ubatch"]),
               "--threads", str(configuration["threads"]), "--gpu-layers", str(configuration["gpu_layers"]),
               "--seed", str(configuration["seed"]), "--temperature", str(configuration["temperature"]),
               "--top-k", str(configuration["top_k"]), "--warmup-predict", str(configuration["warmup_predict"]),
               "--request-timing", str(configuration["request_timing"]), "--output-mode", str(configuration["output_mode"]),
               "--output", str(path)]
    if proof_path:
        command += ["--engine-semantic-proof", str(proof_path)]
    if profile and profile.exists():
        command += ["--calibration-profile", str(profile)]
        if configuration.get("apply_stage_calibration"):
            command.append("--apply-stage-calibration")
        if configuration.get("apply_memory_calibration"):
            command.append("--apply-memory-calibration")
        if configuration.get("apply_phase_boundary_calibration"):
            command.append("--apply-phase-boundary-calibration")
        if configuration.get("apply_launch_calibration"):
            command.append("--apply-launch-calibration")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multimodel_next/generalization_acceptance_v5.json")
    parser.add_argument("--cells-dir", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--start-cell", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--freeze-manifest", "--manifest", dest="freeze_manifest", type=Path,
                        default=ROOT / "artifacts/multimodel_next/blind_generalization_freeze_v5.json")
    parser.add_argument("--engine-semantic-proof", type=Path)
    args = parser.parse_args()
    selected = [item for item in args.models.split(",") if item]
    freeze_path = args.freeze_manifest.resolve()
    freeze = _load_freeze(freeze_path)
    freeze_sha = sha(freeze_path)
    freeze_errors = _freeze_errors(freeze, selected_models=selected, repeats=args.repeats)
    if freeze_errors:
        raise SystemExit("freeze validation failed: " + "; ".join(freeze_errors))
    verifier = ROOT / "tools/check_freeze_manifest.py"
    check = subprocess.run([sys.executable, str(verifier), "--manifest", str(freeze_path), "--verify", "--phase", "resume"], capture_output=True, text=True)
    if check.returncode:
        raise SystemExit("freeze verification failed: " + (check.stdout + check.stderr)[-1200:])
    try:
        proof_path, proof_sha = _resolve_engine_semantic_proof(freeze, args.engine_semantic_proof)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    all_planned = list(planned_cells(selected, args.repeats, freeze.get("scenarios") or {}))
    scheduled = all_planned[args.start_cell:]
    if args.max_cells is not None:
        scheduled = scheduled[:args.max_cells]
    out = args.cells_dir.resolve() if args.cells_dir else args.output.with_suffix("").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell_id, model, prompt_band, output_band, parallel, repeat in scheduled:
        scenario = _scenario_row(freeze, model, prompt_band, output_band, parallel)
        configuration = dict(scenario.get("configuration") or {})
        prompt = str(scenario.get("prompt", PROMPTS[prompt_band]))
        requested_output = int(scenario.get("requested_output_tokens", OUTPUTS[output_band]))
        profile_entry = (freeze.get("calibration_profiles") or {}).get(model) or {}
        profile = Path(str(profile_entry.get("path"))).resolve() if profile_entry.get("path") else None
        for source_key, flag_key in (("apply_stage", "apply_stage_calibration"), ("apply_memory", "apply_memory_calibration"), ("apply_phase_boundary", "apply_phase_boundary_calibration")):
            if profile_entry.get(source_key):
                raise SystemExit("legacy calibrated profile flags require a separately validated profile; freeze them disabled for analytical validation")
        path = out / f"{cell_id}.json"
        binding_path = out / f"{cell_id}.freeze.json"
        if binding_path.exists():
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            if binding.get("freeze_sha256") != freeze_sha:
                raise SystemExit("cell belongs to a different freeze: " + cell_id)
        elif path.exists():
            raise SystemExit("existing cell has no freeze binding: " + cell_id)
        elif not args.dry_run:
            with binding_path.open("x", encoding="utf-8") as handle:
                json.dump({"freeze_sha256": freeze_sha, "cell_id": cell_id}, handle)

        meta = {"cell_id": cell_id, "model_key": model, "prompt_band": prompt_band, "output_band": output_band,
                "parallel": parallel, "repeat": repeat, "prompt": prompt, "requested_output_tokens": requested_output,
                "model_path": str(MODELS[model]), "model_sha256": (freeze.get("model_sha256") or {}).get(model),
                "binary_sha256": freeze.get("binary_sha256"), "freeze_manifest": str(freeze_path), "freeze_sha256": freeze_sha,
                "engine_semantic_proof": {"path": str(proof_path), "sha256": proof_sha},
                "config": configuration, "output": str(path), "planned": True, "observed": False,
                "completed": False, "status": "missing"}
        payload = None
        returncode = 0
        if path.exists() and not args.dry_run:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                meta["observed"] = True
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                payload = None
        elif args.dry_run:
            meta["status"] = "planned"
            cells.append(meta)
            continue
        if payload is None:
            command = _build_command(model, prompt, requested_output, parallel, configuration, profile, proof_path, path)
            meta["command"] = command
            started = time.perf_counter()
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                returncode = -9
                meta["timeout"] = True
                meta["stderr"] = str(exc)
            meta["wall_s"] = time.perf_counter() - started
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    meta["observed"] = True
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
        if payload is None:
            meta.update({"status": "failed", "completed": True, "returncode": returncode})
            cells.append(meta)
            continue
        meta["completed"] = True
        model_sha = (freeze.get("model_sha256") or {}).get(model)
        checks, valid = validate_payload(payload, model=model, prompt=prompt, output=requested_output,
                                         parallel=parallel, model_sha=model_sha,
                                         binary_sha=freeze.get("binary_sha256"), output_mode=str(configuration.get("output_mode", "fixed")),
                                         source_path=path, expected_config=configuration,
                                         engine_semantic_proof=proof_path, engine_semantic_proof_sha256=proof_sha)
        try:
            metrics = metric_records(payload)
        except Exception as exc:
            metrics = {metric: {"status": "evidence_insufficient", "evidence_reasons": [str(exc)]} for metric in METRICS}
            valid = False
            checks["metric_evaluation"] = False
        else:
            checks["metric_evaluation"] = all(item.get("status") in {"measured", "not_applicable"} for item in metrics.values())
            valid = valid and checks["metric_evaluation"]
        meta.update({"status": "valid" if returncode == 0 and valid else "invalid", "returncode": returncode,
                     "checks": checks, "metrics": metrics, "valid": bool(returncode == 0 and valid),
                     "validity_status": payload.get("validity_status"), "prediction_artifact": payload.get("prediction_artifact"),
                     "prediction_sha256": payload.get("prediction_sha256"), "parallel_support": payload.get("parallel_support"),
                     "output_policy": payload.get("output_policy"), "identity": payload.get("identity"),
                     "evidence": payload.get("evidence"), "observed_configuration": payload.get("configuration"),
                     "request_counts": {side: len((payload.get(side) or {}).get("requests") or [])
                                        for side in ("native", "simulator")}})
        cells.append(meta)
    # Invalidate cells before constructing any counts, denominators or numeric
    # aggregation so a failed end gate cannot leave a stale valid headline.
    freeze_end_verified, freeze_end_error = _verify_end(verifier, freeze_path, freeze_sha)
    summary = _summarize_after_end_gate(cells, all_planned, freeze_end_verified, freeze_end_error)
    result = {
        "schema": "generalization-acceptance-matrix/v3",
        "freeze_manifest": str(freeze_path), "freeze_sha256": freeze_sha, "models": selected,
        "prompt_bands": list(PROMPTS), "output_bands": list(OUTPUTS), "parallel_values": list(PARALLEL),
        "repeats": args.repeats, "planned_cell_count": len(all_planned), "scheduled_cell_count": len(scheduled),
        **summary, "cells": cells,
        "policy": "Manifest-defined frozen run. Planned, scheduled, observed, completed, valid, and metric-eligible counts are separate; missing, failed, incomplete, and not_applicable records remain in denominators. Prediction must precede native reveal.",
        "limitations": ["Client timing remains diagnostic; engine semantic proof is required for formal cells.", "Natural EOS is not an acceptance policy.", "Qwen3.8 reduced scenario budget is defined in the freeze manifest."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "planned": len(all_planned), "scheduled": len(scheduled), "observed": result["observed_cell_count"], "completed": result["completed_cell_count"], "valid": result["valid_cell_count"]}, ensure_ascii=False))
    return 0 if result["freeze_end_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
