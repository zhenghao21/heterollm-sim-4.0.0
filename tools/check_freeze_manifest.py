"""Create and verify an immutable v5 generalization freeze manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/multimodel_next/blind_generalization_freeze_v5.json"
BINARY = ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe"
TOKENIZER = ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-tokenize.exe"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from native_llama_compare import (  # noqa: E402
    _hardware_fingerprint,
    _runtime_artifact_refs,
    probe_hardware,
)
from generalization_acceptance_matrix import (  # noqa: E402
    MODELS,
    OUTPUTS,
    PARALLEL,
    PROMPTS,
    PROFILE,
)

from evaluation_contract import validate_engine_semantic_proof

SCHEMA = "blind-generalization-freeze/v5"
FREEZE_ID = "v5-five-model-nine-scenario"
REPEATS = 3
ENGINE_TIMING_CONTRACT = {
    "id": "engine-stage+client-real-token/v3",
    "engine_ttft": "first_engine_token-engine_request_begin;counter=t_prompt_last-t_start",
    "engine_e2e": "last_engine_token-engine_request_begin;counter=t_gen_last-t_start",
    "engine_tpot": "(last_engine_token-first_engine_token)/(output_tokens-1);counter=(t_gen_last-t_prompt_last)/(output_tokens-1);null_if_output_tokens<=1",
    "ttft": "first_real_token-request_start;client_diagnostic",
    "e2e": "last_real_token-request_start;DONE_excluded;client_diagnostic",
    "tpot": "(last_real_token-first_real_token)/(output_tokens-1);client_diagnostic",
    "unit": "milliseconds",
    "clock": "monotonic_ns",
}

# Nine combinations are shared by all five models.  Qwen3.8 is intentionally
# kept at the same nine-cell budget and this fact is explicit in the manifest.
SCENARIO_KEYS = (
    ("short", "short", 1), ("short", "medium", 4), ("short", "long", 2),
    ("medium", "short", 4), ("medium", "medium", 2), ("medium", "long", 1),
    ("long", "short", 2), ("long", "medium", 1), ("long", "long", 4),
)


VALIDATION_PROMPTS = {
    "short": "Name one warm color.",
    "medium": "Describe how request batching changes memory traffic and latency during autoregressive inference, using concise technical language.",
    "long": " ".join(["An inference scheduler tracks independent requests, key value cache growth, physical memory traffic, tensor shapes and output selection. Explain these interactions in a reproducible experiment."] * 6),
}
VALIDATION_OUTPUTS = {"short": 5, "medium": 17, "long": 49}
LARGE_MODEL_OUTPUTS = {"short": 2, "medium": 4, "long": 8}


def digest(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"empty freeze input: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_files() -> list[Path]:
    roots = [ROOT / "src", ROOT / "tools", ROOT / "source/llama.cpp-semantic-patches"]
    files: list[Path] = []
    for base in roots:
        if base.exists():
            files.extend(path for path in sorted(base.rglob("*")) if path.is_file())
    # Generated/temporary files under tools are not source inputs.
    files = [path for path in files if path.suffix.lower() not in {".pyc", ".log", ".csv", ".sqlite"}]
    return files


def _artifact_inputs() -> dict[str, list[Path]]:
    profiles = [PROFILE[name][0] for name in MODELS if name in PROFILE and PROFILE[name][0] is not None]
    runtime = [Path(item["path"]) for item in _runtime_artifact_refs(BINARY)
               if item.get("status") == "captured" and item.get("path")]
    return {
        "source": _source_files(),
        "models": [Path(path) for path in MODELS.values()],
        "profiles": [Path(path) for path in profiles],
        "runtime": runtime + [TOKENIZER],
    }


def _sha_map(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        out[str(resolved.relative_to(ROOT)).replace("\\", "/")] = digest(resolved)
    return out


def _scenario_manifest() -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for model in MODELS:
        rows = []
        for prompt_band, output_band, parallel in SCENARIO_KEYS:
            rows.append({
                "prompt_band": prompt_band,
                "prompt": VALIDATION_PROMPTS[prompt_band],
                "output_band": output_band,
                "requested_output_tokens": int((LARGE_MODEL_OUTPUTS if model == "qwen38" else VALIDATION_OUTPUTS)[output_band]),
                "parallel": int(parallel),
                "configuration": {
                    "ctx": 2048, "batch": 64, "ubatch": 64, "threads": 16,
                    "threads_batch": 16, "gpu_layers": 0 if model == "qwen38" else -1,
                    "request_timing": "stream", "output_mode": "fixed",
                    "ignore_eos": True, "seed": 42, "temperature": 0.0, "top_k": 1,
                    "warmup_predict": 2,
                },
                "migration_type": "same_hardware_shape_validation",
                "budget_note": "qwen38 uses the reduced nine-scenario budget; fixed output bands remain explicit",
            })
        result[model] = rows
    return result


def _hardware_errors(snapshot: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(snapshot, dict):
        return ["hardware snapshot missing"]
    cpu = str(snapshot.get("cpu") or "").strip()
    gpu = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), dict) else {}
    for name, value in (("cpu", cpu), ("gpu.name", gpu.get("name")),
                        ("gpu.uuid", gpu.get("uuid")),
                        ("gpu.compute_capability", gpu.get("compute_capability"))):
        if value in (None, "", "unknown"):
            errors.append(f"hardware.{name} missing")
    if not snapshot.get("cpu_topology"):
        errors.append("hardware.cpu_topology missing")
    if not gpu.get("pcie"):
        errors.append("hardware.gpu.pcie missing")
    return errors


def _manifest_path(value: Path | None) -> Path:
    return (value or DEFAULT_MANIFEST).resolve()


def create(manifest_path: Path = DEFAULT_MANIFEST, *, engine_semantic_proof: Path | None = None) -> dict[str, object]:
    manifest_path = _manifest_path(manifest_path)
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing freeze manifest: {manifest_path}")
    inputs = _artifact_inputs()
    missing = [str(path) for paths in inputs.values() for path in paths
               if not path.exists() or not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise SystemExit("missing or empty freeze input: " + ", ".join(missing))
    hardware = probe_hardware()
    errors = _hardware_errors(hardware)
    if errors:
        raise SystemExit("hardware identity incomplete: " + "; ".join(errors))
    scenarios = _scenario_manifest()
    proof_path = engine_semantic_proof
    if proof_path is None or not proof_path.is_file():
        raise SystemExit("required engine semantic proof missing")
    proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof_payload.get("schema") != "engine-semantic-proof/v2":
        raise SystemExit("strict freeze requires semantic proof v2")
    runtime_refs = proof_payload.get("runtime_artifacts")
    ok, proof_errors = validate_engine_semantic_proof(proof_payload, binary_artifacts=runtime_refs)
    if not ok:
        raise SystemExit("semantic proof invalid: " + "; ".join(proof_errors))
    proof = {"path": str(proof_path.resolve()), "sha256": digest(proof_path)}
    manifest = {
        "schema": SCHEMA,
        "freeze_id": FREEZE_ID,
        "status": "frozen_not_run",
        "purpose": "preregistered_shape_screening_observer_effect_unresolved",
        "calibration_policy": "analytical_baseline_all_stage_memory_phase_overrides_disabled",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Freeze before execution; prediction is saved before native reveal; no result-driven tuning; missing/failed/incomplete evidence remains in denominators.",
        "matrix": {
            "models": list(MODELS), "repeats": REPEATS,
            "prompt_bands": list(PROMPTS), "output_bands": list(OUTPUTS),
            "parallel_values": list(PARALLEL), "scenario_count_by_model": {m: len(v) for m, v in scenarios.items()},
            "execution_count": sum(len(v) for v in scenarios.values()) * REPEATS,
        },
        "scenarios": scenarios,
        "source_sha256": _sha_map(inputs["source"]),
        "model_sha256": {name: digest(path) for name, path in MODELS.items()},
        "profile_sha256": {name: digest(PROFILE[name][0]) for name in MODELS
                            if name in PROFILE and PROFILE[name][0] is not None},
        "calibration_profiles": {
            name: {"path": str(PROFILE[name][0].resolve()),
                   "apply_stage": False,
                   "apply_memory": False,
                   "apply_phase_boundary": False}
            for name in MODELS if name in PROFILE and PROFILE[name][0] is not None
        },
        "runtime_artifacts": runtime_refs,
        "tokenizer_sha256": digest(TOKENIZER),
        "binary_sha256": digest(BINARY),
        "hardware_snapshot": hardware,
        "hardware_fingerprint": _hardware_fingerprint(hardware),
        "engine_semantic_proof": proof,
        "measurement_contract": {
            "engine_timing_contract": ENGINE_TIMING_CONTRACT,
            "engine_timing_contract_sha256": hashlib.sha256(json.dumps(ENGINE_TIMING_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "engine_primary_metrics": ["engine_ttft", "engine_tpot", "engine_e2e"],
            "eos": "fixed ignore_eos; natural EOS is exploratory only",
        },
        "required_identity": ["source_sha256", "model_sha256", "profile_sha256", "runtime_artifacts", "binary_sha256", "tokenizer_sha256", "hardware_fingerprint"],
        "acceptance_status": "not_run",
        "limitations": ["cross-hardware and cross-model are unverified", "Qwen3.8 uses nine scenarios with output 2/4/8; these bands are relative to its reduced budget", "observer effect equivalence remains unproven; results cannot certify final accuracy"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"manifest": str(manifest_path), "status": manifest["status"], "executions": manifest["matrix"]["execution_count"]}


def verify(manifest_path: Path = DEFAULT_MANIFEST, *, phase: str = "before") -> dict[str, object]:
    manifest_path = _manifest_path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"freeze manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"freeze manifest unreadable: {exc}")
    errors: list[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append("freeze schema mismatch; legacy freeze is not accepted")
    if phase in {"before", "resume"} and manifest.get("status") not in {"frozen_not_run", "running"}:
        errors.append(f"freeze status invalid for {phase}: {manifest.get('status')}")
    if phase == "end" and manifest.get("status") not in {"frozen_not_run", "running", "completed", "failed"}:
        errors.append(f"freeze status invalid for end: {manifest.get('status')}")
    matrix = manifest.get("matrix") if isinstance(manifest.get("matrix"), dict) else {}
    if matrix.get("models") != list(MODELS) or matrix.get("repeats") != REPEATS:
        errors.append("freeze model/repeat declaration mismatch")
    if matrix.get("prompt_bands") != list(PROMPTS) or matrix.get("output_bands") != list(OUTPUTS):
        errors.append("freeze prompt/output declaration mismatch")
    if matrix.get("parallel_values") != list(PARALLEL):
        errors.append("freeze parallel declaration mismatch")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, dict):
        errors.append("freeze scenarios missing")
    else:
        for model in MODELS:
            rows = scenarios.get(model)
            if not isinstance(rows, list) or len(rows) != 9:
                errors.append(f"freeze scenario count invalid: {model}")
                continue
            keys = set()
            for row in rows:
                if not isinstance(row, dict):
                    errors.append(f"freeze scenario malformed: {model}")
                    continue
                key = (row.get("prompt_band"), row.get("output_band"), row.get("parallel"))
                if key in keys:
                    errors.append(f"duplicate freeze scenario: {model}:{key}")
                keys.add(key)
                if row.get("prompt") != VALIDATION_PROMPTS.get(row.get("prompt_band")):
                    errors.append(f"prompt declaration mismatch: {model}:{key}")
                if row.get("requested_output_tokens") != (LARGE_MODEL_OUTPUTS if model == "qwen38" else VALIDATION_OUTPUTS).get(row.get("output_band")):
                    errors.append(f"output declaration mismatch: {model}:{key}")
                if not isinstance(row.get("configuration"), dict) or not row.get("configuration"):
                    errors.append(f"scenario configuration missing: {model}:{key}")
    source_sha = manifest.get("source_sha256")
    if not isinstance(source_sha, dict) or not source_sha:
        errors.append("source_sha256 missing")
    else:
        expected_sources = {str(p.relative_to(ROOT)).replace("\\", "/") for p in _source_files()}
        if set(source_sha) != expected_sources:
            errors.append("source file set mismatch (added or omitted inputs)")
        for rel, wanted in source_sha.items():
            path = ROOT / str(rel)
            if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"source input missing or empty: {rel}")
            elif not isinstance(wanted, str) or len(wanted) != 64 or digest(path).lower() != wanted.lower():
                errors.append(f"source hash mismatch: {rel}")
    for section in ("model_sha256", "profile_sha256"):
        values = manifest.get(section)
        if not isinstance(values, dict) or not values:
            errors.append(f"{section} missing")
        else:
            expected_names = set(MODELS) if section == "model_sha256" else {name for name in MODELS if PROFILE.get(name, (None,))[0] is not None}
            if set(values) != expected_names:
                errors.append(section + " coverage mismatch")
            for name, wanted in values.items():
                path = MODELS.get(name) if section == "model_sha256" else PROFILE.get(name, (None,))[0]
                if path is None or not path.exists() or path.stat().st_size <= 0:
                    errors.append(f"{section} input missing: {name}")
                elif not isinstance(wanted, str) or len(wanted) != 64 or digest(path).lower() != wanted.lower():
                    errors.append(f"{section} hash mismatch: {name}")
    calibration_profiles = manifest.get("calibration_profiles")
    if not isinstance(calibration_profiles, dict):
        errors.append("calibration_profiles missing")
    else:
        for name, entry in calibration_profiles.items():
            if name not in MODELS or not isinstance(entry, dict):
                errors.append(f"calibration profile declaration malformed: {name}")
                continue
            path = Path(str(entry.get("path", "")))
            wanted = (manifest.get("profile_sha256") or {}).get(name)
            if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"calibration profile input missing: {name}")
            elif not isinstance(wanted, str) or digest(path).lower() != wanted.lower():
                errors.append(f"calibration profile hash mismatch: {name}")

    runtime = manifest.get("runtime_artifacts")
    if not isinstance(runtime, list) or not runtime:
        errors.append("runtime_artifacts missing")
    else:
        for item in runtime:
            try:
                if digest(Path(item["path"])) != item["sha256"]:
                    errors.append("runtime artifact identity mismatch")
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("runtime artifact missing or malformed")
    for key, path in (("binary_sha256", BINARY), ("tokenizer_sha256", TOKENIZER)):
        wanted = manifest.get(key)
        if not isinstance(wanted, str) or len(wanted) != 64 or not path.exists() or digest(path).lower() != wanted.lower():
            errors.append(f"{key} mismatch")
    hardware = manifest.get("hardware_snapshot")
    errors.extend(_hardware_errors(hardware))
    wanted_hw = manifest.get("hardware_fingerprint")
    if not isinstance(wanted_hw, str) or len(wanted_hw) != 64:
        errors.append("hardware_fingerprint missing or malformed")
    else:
        current_hw = probe_hardware()
        if _hardware_fingerprint(current_hw) != wanted_hw:
            errors.append("hardware fingerprint mismatch")
    contract = manifest.get("measurement_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("engine_timing_contract"), dict):
        errors.append("engine timing contract missing")
    if isinstance(contract, dict):
        expected_contract_sha = hashlib.sha256(json.dumps(ENGINE_TIMING_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if contract.get("engine_timing_contract") != ENGINE_TIMING_CONTRACT or contract.get("engine_timing_contract_sha256") != expected_contract_sha:
            errors.append("engine timing contract content or SHA mismatch")
    proof = manifest.get("engine_semantic_proof")
    if not isinstance(proof, dict) or not proof.get("path") or not proof.get("sha256"):
        errors.append("engine semantic proof declaration missing")
    else:
        path = Path(str(proof["path"]))
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            errors.append("engine semantic proof input missing")
        elif digest(path).lower() != str(proof["sha256"]).lower():
            errors.append("engine semantic proof hash mismatch")
        else:
            proof_payload = json.loads(path.read_text(encoding="utf-8"))
            if proof_payload.get("schema") != "engine-semantic-proof/v2":
                errors.append("strict freeze requires semantic proof v2")
            ok, proof_errors = validate_engine_semantic_proof(proof_payload, binary_artifacts=runtime)
            errors.extend(proof_errors)
    if errors:
        raise SystemExit("freeze verification failed: " + "; ".join(errors))
    return {"manifest": str(manifest_path), "verified": True, "phase": phase, "status": manifest.get("status")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--engine-semantic-proof", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--phase", choices=("before", "resume", "end"), default="before")
    args = parser.parse_args()
    if args.create == args.verify:
        parser.error("choose exactly one of --create or --verify")
    result = create(args.manifest, engine_semantic_proof=args.engine_semantic_proof) if args.create else verify(args.manifest, phase=args.phase)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
