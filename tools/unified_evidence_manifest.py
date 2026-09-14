"""Build and verify one immutable evidence manifest for native runs.

The native comparator already stores a per-run ``evidence`` object.  This
module adds a small, sidecar-friendly manifest that binds that object to the
raw NVTX/CUPTI/NSYS files used by B-06/B-08, the timing boundaries, and the
exact extractor/binary bytes.  It deliberately has no fitting or prediction
logic: a missing or changed artifact is an invalid manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


# The v3 contract is the only contract that can feed the L1 engine gate.  Keep
# this copy local to the manifest validator so a stale/hand-edited manifest
# cannot silently downgrade to a client-only timing interpretation.
ENGINE_TIMING_CONTRACT_ID = "engine-stage+client-real-token/v3"
ENGINE_TIMING_CONTRACT_FIELDS = (
    "engine_ttft", "engine_e2e", "engine_tpot", "ttft", "e2e", "tpot",
    "unit", "clock",
)
ENGINE_BOUNDARY_FIELDS = ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse the ISO-8601 timestamps emitted by the native comparator.

    A timestamp without an offset is deliberately rejected.  Comparing a
    local wall clock to a UTC/native timestamp would make the prediction
    ordering gate meaningless.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_prediction_before_native(payload: Mapping[str, Any], *,
                                      source_path: Path | None = None,
                                      require: bool = True) -> list[str]:
    """Fail closed unless the prediction sidecar predates native execution.

    The comparator writes a simulator prediction sidecar before starting the
    native server.  This check binds that sidecar to the payload, verifies its
    bytes, and compares its timezone-aware creation timestamp with the native
    reveal timestamp.  It intentionally does not inspect or fit native
    latencies.  ``require=False`` is useful for historical/client-only data.
    """
    errors: list[str] = []
    ref = payload.get("prediction_artifact")
    expected_sha = payload.get("prediction_sha256")
    status = payload.get("prediction_status")
    reveal = _parse_timestamp(payload.get("native_reveal_timestamp_utc"))
    if not ref or not expected_sha or status != "saved_before_native_reveal" or reveal is None:
        if require:
            if not ref:
                errors.append("prediction_artifact missing")
            if not _is_sha256(expected_sha):
                errors.append("prediction_sha256 missing or malformed")
            if status != "saved_before_native_reveal":
                errors.append("prediction_status not saved_before_native_reveal")
            if reveal is None:
                errors.append("native_reveal_timestamp_utc missing or malformed")
        return errors
    path = _resolve(ref, root=(source_path.parent if source_path else Path.cwd()),
                    source_path=source_path)
    if path is None or not path.exists():
        errors.append("prediction_artifact missing")
        return errors
    actual = sha256_file(path)
    if str(expected_sha).lower() != actual:
        errors.append("prediction_artifact sha256 mismatch")
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        sidecar = None
    if not isinstance(sidecar, Mapping):
        errors.append("prediction_artifact malformed")
        return errors
    if sidecar.get("status") != "saved_before_native_reveal":
        errors.append("prediction_artifact status invalid")
    created = _parse_timestamp(sidecar.get("created_utc"))
    if created is None:
        errors.append("prediction_artifact created_utc missing or malformed")
    elif created >= reveal:
        errors.append("prediction_artifact was not saved before native reveal")
    return errors


def _sha256_text(value: Mapping[str, Any]) -> str:
    """Hash the canonical timing contract, excluding its self-hash field."""
    body = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _native_measurements_digest(native: Mapping[str, Any]) -> str:
    """Hash the complete captured native result, excluding outer evidence."""
    # Keep this algorithm identical to replay_simulator_from_native without
    # importing it (that module imports the comparator and would cycle).
    from heterollm_sim.serde import stable_hash
    if not isinstance(native, Mapping):
        return stable_hash({})
    measured = dict(native)
    # Persisted payloads may mirror the evidence object under ``native``;
    # excluding it keeps this digest non-self-referential and matches both
    # producer and replay validators.
    measured.pop("evidence", None)
    return stable_hash(measured)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: Any, *, root: Path, source_path: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [root / path]
    if source_path is not None:
        candidates.append(source_path.resolve().parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _ref(value: Any, *, role: str, root: Path, source_path: Path | None = None,
         declared_sha: str | None = None) -> dict[str, Any] | None:
    path = _resolve(value, root=root, source_path=source_path)
    if path is None:
        return None
    item: dict[str, Any] = {"role": role, "path": str(path.resolve())}
    item["exists"] = path.exists()
    item["sha256"] = sha256_file(path) if path.exists() else None
    if declared_sha:
        item["declared_sha256"] = str(declared_sha)
    return item


def _add_ref(items: list[dict[str, Any]], ref: dict[str, Any] | None) -> None:
    if ref is None:
        return
    key = (ref.get("path"), ref.get("role"))
    if not any((old.get("path"), old.get("role")) == key for old in items):
        items.append(ref)


def _sidecar_refs(sidecar: Mapping[str, Any], *, root: Path,
                  source_path: Path | None) -> list[dict[str, Any]]:
    """Collect file references from B-06/B-08 style manifests.

    The two historical manifests use different shapes (``scenarios[].files``
    versus ``train_trace``/``profile_artifacts``), so collection is explicit
    and conservative.  Unknown values are retained as metadata by the caller
    but are never guessed as artifacts.
    """
    result: list[dict[str, Any]] = []
    def add(value: Any, role: str, declared: str | None = None) -> None:
        if isinstance(value, Mapping):
            add(value.get("path"), role, value.get("sha256"))
            return
        if isinstance(value, str):
            _add_ref(result, _ref(value, role=role, root=root, source_path=source_path,
                                   declared_sha=declared))

    for key in ("train_trace", "holdout_trace"):
        add(sidecar.get(key), f"{key}.trace")
    for key in ("profile_artifacts", "extractor_output_artifacts", "extractor_script_artifacts"):
        for value in sidecar.get(key) or []:
            add(value, key.rstrip("s"))
    for scenario in sidecar.get("scenarios") or []:
        if not isinstance(scenario, Mapping):
            continue
        for value in scenario.get("files") or []:
            add(value, f"scenario.{scenario.get('stem', 'unknown')}")
    for trace in sidecar.get("traces") or []:
        if isinstance(trace, Mapping):
            add(trace.get("trace"), "trace.json")
    # B-08 build manifest stores named artifacts with path/sha256.
    for name, value in (sidecar.get("artifacts") or {}).items():
        add(value, f"build.{name}")
    # B-10 host evidence has direct NSYS/SQLite references.
    for key, role in (("sqlite", "host.sqlite"), ("nsys_rep", "host.nsys-rep"),
                      ("binary", "host.binary")):
        add(sidecar.get(key), role, sidecar.get(f"{key}_sha256"))
    return result


def build_manifest(payload: Mapping[str, Any], *, sidecars: Iterable[Path] = (),
                   source_path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """Return a deterministic unified manifest without changing ``payload``."""
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    evidence = payload.get("evidence") or (payload.get("native") or {}).get("evidence") or {}
    identity = payload.get("identity") or (payload.get("native") or {}).get("identity") or {}
    artifacts: list[dict[str, Any]] = []
    for section, role in (("native_binary", "native.binary"), ("raw_trace_events", "raw.trace"),
                          ("extractor", "extractor"), ("extractor_output", "extractor.output")):
        value = evidence.get(section) or {}
        if isinstance(value, Mapping) and value.get("path"):
            declared = value.get("file_sha256") if section == "extractor" else value.get("sha256")
            _add_ref(artifacts, _ref(value["path"], role=role, root=root,
                                     source_path=source_path, declared_sha=declared))
    # A CUDA trace depends on the DLLs loaded beside the executable.  Bind
    # capture-time dependency hashes when the producer supplied them; never
    # synthesize this list from the current filesystem during verification.
    for dependency in evidence.get("runtime_artifacts") or evidence.get("native_binary_artifacts") or []:
        if isinstance(dependency, Mapping) and dependency.get("path"):
            _add_ref(artifacts, _ref(dependency["path"], role="native.dependency",
                                     root=root, source_path=source_path,
                                     declared_sha=dependency.get("sha256")))
    # Keep the pre-reveal simulator prediction in the same immutable binding
    # as native trace/extractor artifacts.  Without this reference a later
    # replay could silently replace the prediction while retaining the native
    # payload and its SHA.
    prediction_path = payload.get("prediction_artifact")
    if prediction_path:
        _add_ref(artifacts, _ref(prediction_path, role="prediction.pre_native",
                                 root=root, source_path=source_path,
                                 declared_sha=payload.get("prediction_sha256")))
    for section in ("extractor_output_artifacts", "extractor_script_artifacts"):
        for value in evidence.get(section) or []:
            if isinstance(value, Mapping):
                _add_ref(artifacts, _ref(value.get("path"), role=section, root=root,
                                         source_path=source_path, declared_sha=value.get("sha256")))
    raw = evidence.get("raw_trace_events") or {}
    for value in raw.get("supplemental_artifacts") or []:
        if isinstance(value, Mapping):
            _add_ref(artifacts, _ref(value.get("path"), role="raw.supplemental", root=root,
                                     source_path=source_path, declared_sha=value.get("sha256")))
    sidecar_records = []
    for sidecar_path in sidecars:
        path = _resolve(sidecar_path, root=root, source_path=source_path)
        if path is None or not path.exists():
            sidecar_records.append({"path": str(path or sidecar_path), "exists": False, "sha256": None})
            continue
        sidecar_json: Any = None
        try:
            sidecar_json = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sidecar_json = {}
        sidecar_records.append({"path": str(path.resolve()), "exists": True,
                                "sha256": sha256_file(path), "schema": sidecar_json.get("schema") if isinstance(sidecar_json, Mapping) else None})
        if isinstance(sidecar_json, Mapping):
            for item in _sidecar_refs(sidecar_json, root=root, source_path=path):
                _add_ref(artifacts, item)
    engine = evidence.get("engine_timing") or {}
    token = evidence.get("token_timestamps") or {}
    request = evidence.get("request_boundaries") or {}
    timing = evidence.get("timing_contract") or {}
    server = evidence.get("server_boundary") or evidence.get("host_boundary") or {"status": "unavailable"}
    prediction_present = bool(payload.get("prediction_artifact"))
    prediction_errors = validate_prediction_before_native(payload, source_path=source_path,
                                                          require=prediction_present)
    client = {"status": "captured" if request.get("status") == "captured" else "diagnostic",
              "timing_contract": {k: timing.get(k) for k in ("ttft", "tpot", "e2e") if k in timing},
              "request_boundaries": request}
    manifest: dict[str, Any] = {
        "schema": "native-evidence-manifest/v2",
        "purpose": "immutable_native_evidence_binding",
        "validation_scope": "live_capture",
        "source_payload": str(source_path.resolve()) if source_path else None,
        "identity": identity,
        "native_contract_sha256": evidence.get("native_contract_sha256"),
        "native_measurements_sha256": evidence.get("native_measurements_sha256"),
        "timing_contract": timing,
        "boundaries": {
            "engine": {**engine, "token_timestamps": token, "request_boundaries": request},
            "server": server,
            "client": client,
        },
        "artifacts": artifacts,
        "sidecar_manifests": sidecar_records,
        "evidence_level": "complete" if engine.get("status") not in (None, "unavailable") else "development_without_engine",
        "prediction_before_native": {
            "status": ("valid" if not prediction_errors else "invalid") if prediction_present else "unavailable",
            "errors": prediction_errors,
        },
        "validation": {"status": "unverified", "errors": []},
    }
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, require_engine: bool = True,
                      require_engine_contract: bool = False) -> list[str]:
    """Validate immutable evidence and (optionally) the L1 engine contract.

    ``require_engine`` preserves the historical, schema-level check.  The
    stricter ``require_engine_contract`` mode is used by replay/freeze gates:
    it rejects a stale v1/client contract, missing engine fields, and a
    missing or malformed server status.  ``server.status=unavailable`` is
    valid in this mode and deliberately means that L2/L3 decomposition is
    unavailable; it must never be interpreted as captured evidence.
    """
    errors: list[str] = []
    identity = manifest.get("identity") or {}
    for field in ("model_path", "gguf_sha256", "runtime_fingerprint", "hardware_fingerprint",
                  "prompt_fingerprint", "configuration"):
        if identity.get(field) in (None, "", {}, []):
            errors.append(f"identity.{field} missing")
    if not manifest.get("native_contract_sha256"):
        errors.append("native_contract_sha256 missing")
    timing = manifest.get("timing_contract") or {}
    if not timing or not timing.get("id"):
        errors.append("timing_contract missing")
    engine = (manifest.get("boundaries") or {}).get("engine") or {}
    if require_engine and engine.get("status") in (None, "unavailable"):
        errors.append("engine boundary evidence unavailable")
    if require_engine_contract:
        if timing.get("id") != ENGINE_TIMING_CONTRACT_ID:
            errors.append("timing_contract v3 required for L1 engine evidence")
        for field in ENGINE_TIMING_CONTRACT_FIELDS:
            if timing.get(field) in (None, ""):
                errors.append(f"timing_contract.{field} missing")
        declared_contract_sha = timing.get("sha256")
        if not _is_sha256(declared_contract_sha):
            errors.append("timing_contract.sha256 missing or malformed")
        elif declared_contract_sha.lower() != _sha256_text(timing):
            errors.append("timing_contract.sha256 mismatch")
        if engine.get("status") not in ("counter_proven", "marker_proven"):
            errors.append("engine boundary evidence not proven")
        measurements_sha = manifest.get("native_measurements_sha256")
        if not _is_sha256(measurements_sha):
            errors.append("native_measurements_sha256 missing or malformed")
        fields = engine.get("fields")
        if tuple(fields or ()) != ENGINE_BOUNDARY_FIELDS:
            errors.append("engine boundary fields incomplete")
        if not engine.get("source"):
            errors.append("engine boundary source missing")
        for name, section in (("token_timestamps", engine.get("token_timestamps")),
                              ("request_boundaries", engine.get("request_boundaries"))):
            if not isinstance(section, Mapping) or section.get("status") != "captured":
                errors.append(f"engine.{name} not captured")
        server = (manifest.get("boundaries") or {}).get("server")
        if not isinstance(server, Mapping) or server.get("status") not in ("captured", "unavailable"):
            errors.append("server boundary status missing or invalid")
    source_payload = manifest.get("source_payload")
    if require_engine_contract and (not source_payload or not Path(str(source_payload)).exists()):
        errors.append("source_payload missing")
    if require_engine_contract and source_payload and Path(str(source_payload)).exists():
        try:
            payload = json.loads(Path(str(source_payload)).read_text(encoding="utf-8"))
            payload_evidence = payload.get("evidence") or (payload.get("native") or {}).get("evidence") or {}
            payload_native = payload.get("native") or {}
            declared_native_sha = payload_evidence.get("native_measurements_sha256")
            if not _is_sha256(declared_native_sha):
                errors.append("source_payload.native_measurements_sha256 missing or malformed")
            elif declared_native_sha.lower() != _native_measurements_digest(payload_native):
                errors.append("source_payload.native_measurements_sha256 mismatch")
            elif manifest.get("native_measurements_sha256") != declared_native_sha:
                errors.append("native_measurements_sha256 manifest/source mismatch")
            if manifest.get("native_contract_sha256") != payload_evidence.get("native_contract_sha256"):
                errors.append("native_contract_sha256 manifest/source mismatch")
            errors.extend(validate_prediction_before_native(payload,
                                                           source_path=Path(str(source_payload)),
                                                           require=True))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append("source_payload unreadable for prediction gate")
        if not any(item.get("role") == "prediction.pre_native"
                   for item in manifest.get("artifacts") or []):
            errors.append("prediction artifact binding missing")
    historical_capture = manifest.get("validation_scope") == "historical_capture"
    for index, sidecar in enumerate(manifest.get("sidecar_manifests") or []):
        sidecar_path = Path(str(sidecar.get("path", "")))
        if historical_capture:
            if not sidecar.get("exists") or not _is_sha256(sidecar.get("sha256")):
                errors.append(f"sidecar_manifests[{index}] capture-time identity missing")
            continue
        if not sidecar.get("exists") or not sidecar.get("sha256") or not sidecar_path.exists():
            errors.append(f"sidecar_manifests[{index}] missing")
        elif not _is_sha256(sidecar.get("sha256")):
            errors.append(f"sidecar_manifests[{index}] sha256 mismatch: {sidecar_path}")
        elif sha256_file(sidecar_path) != str(sidecar.get("sha256")).lower():
            errors.append(f"sidecar_manifests[{index}] sha256 mismatch: {sidecar_path}")
    for index, artifact in enumerate(manifest.get("artifacts") or []):
        path = Path(str(artifact.get("path", "")))
        if historical_capture:
            # A historical manifest is intentionally checked against the
            # capture-time declaration, not the mutable current worktree.
            # This preserves an old native actual as evidence for its original
            # environment without mislabelling it as a live freeze pass.
            if not artifact.get("exists") or not _is_sha256(artifact.get("sha256")):
                errors.append(f"artifacts[{index}] capture-time identity missing: {path}")
            declared = artifact.get("declared_sha256")
            if declared is not None and not _is_sha256(declared):
                errors.append(f"artifacts[{index}] declared capture SHA malformed: {path}")
            continue
        if not artifact.get("exists") or not path.exists():
            errors.append(f"artifacts[{index}] missing: {path}")
            continue
        actual = sha256_file(path)
        if not artifact.get("sha256"):
            errors.append(f"artifacts[{index}] sha256 missing: {path}")
        elif not _is_sha256(artifact.get("sha256")):
            errors.append(f"artifacts[{index}] sha256 mismatch: {path}")
        elif str(artifact.get("sha256")).lower() != actual:
            errors.append(f"artifacts[{index}] sha256 mismatch: {path}")
        declared = artifact.get("declared_sha256")
        if declared and str(declared).lower() != actual:
            errors.append(f"artifacts[{index}] declared sha256 mismatch: {path}")
    if require_engine_contract and not manifest.get("artifacts"):
        errors.append("artifacts missing")
    return errors


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--sidecar", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-engine", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    manifest = build_manifest(payload, sidecars=args.sidecar, source_path=args.payload)
    errors = validate_manifest(manifest, require_engine=not args.allow_missing_engine,
                               require_engine_contract=not args.allow_missing_engine)
    manifest["validation"] = {"status": "valid" if not errors else "invalid", "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": manifest["validation"]["status"], "errors": errors}, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    _cli()
