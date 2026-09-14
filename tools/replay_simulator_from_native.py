"""Replay simulator timing from immutable native evidence, without native runs."""
from __future__ import annotations
import argparse, json, math, statistics, hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_llama_compare import build_matching_scenario
from heterollm_sim.gguf_parity import read_gguf_metadata, build_model_from_gguf
from heterollm_sim.calibration import load_native_calibration, apply_native_calibration
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serde import stable_hash

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "qwen25": ROOT / "artifacts/multimodel_20260913/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen35": ROOT / "artifacts/multimodel_20260913/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    "qwen38": ROOT / "artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf",
    "tinyllama": ROOT / "artifacts/multimodel_20260913/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "smollm2": ROOT / "artifacts/multimodel_20260913/models/smollm2-1.7b-instruct-q4_k_m.gguf",
}
PROFILES = {
    # The startup residual is derived only from independent Qwen2.5 GPU
    # semantic traces (B06/B07); it is charged once on the first decode
    # invocation and remains identity gated.  The target replay's native
    # final latency is never used to construct this profile.
    "qwen25": (ROOT / "artifacts/development/gpu_decode_startup_profile_qwen25_v1.json", True, False, True),
    # The Qwen3.5 profile currently has fully covered semantic train/holdout
    # traces for the prefill boundary, but its stage aggregates are not an
    # exact six-dimensional operator profile for this replay shape.  Applying
    # those aggregates would double-count/overstate decode work (and is
    # therefore fail-closed at the global cost-model level).  Keep only the
    # independently evidenced one-task prefill boundary calibration here;
    # operator costs remain analytical until an exact-key profile is built.
    "qwen35": (ROOT / "artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_calibration_v1.json", False, False, True),
    "qwen38": (ROOT / "artifacts/multimodel_next/qwen38_cpu_semantic_calibration_prompt8_f9_v2.json", True, True, False),
    # The TinyLlama phase candidate is retained for diagnosis only until its
    # capture-time binary identity is proven.  The legacy global 8 us launch
    # scalar also remains disabled because it lacks phase/exact provenance.
    "tinyllama": (ROOT / "artifacts/development/tinyllama_semantic_calibration_phase_v2.json", False, False, False),
    # SmolLM2 phase evidence is retained for diagnosis only until its engine
    # marker/cost boundary is separately validated; do not apply it in replay.
    "smollm2": (None, False, False, False),
}


def _resolve_evidence_path(value, source_path=None):
    """Resolve an evidence path without trusting the payload working directory."""
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path
    candidates = [ROOT / path]
    if source_path is not None:
        candidates.append(Path(source_path).resolve().parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_binary_gate(profile_path, payload):
    """Fail closed when a calibration profile lacks capture-time binary proof.

    A runtime fingerprint is not enough for CUDA calibration: the executable
    can load a different llama/ggml CUDA DLL while retaining the same CLI and
    hardware fingerprint.  Profiles therefore need an explicit executable
    hash *and* a dependency-module manifest, and the native payload must carry
    the corresponding capture-time manifest.  Never fill the latter from the
    current filesystem; doing so would manufacture provenance after capture.
    """
    result = {"status": "blocked", "profile": str(profile_path), "reasons": []}
    try:
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except Exception as exc:
        result["reasons"].append(f"profile unreadable: {exc}")
        return result
    if not isinstance(profile, dict):
        result["reasons"].append("profile is not an object")
        return result
    # Accept only explicit fields written at capture time.  A source_sha256
    # mapping is accepted for the executable for backwards compatibility, but
    # it still cannot satisfy the dependency-manifest requirement.
    declared_binary = profile.get("binary_sha256")
    profile_artifacts = profile.get("runtime_artifacts") or profile.get("native_binary_artifacts")
    if declared_binary is None and isinstance(profile_artifacts, list):
        # native_llama_profile.py stores the executable and loaded DLLs under
        # native_binary_artifacts; select the executable by its path.
        for item in profile_artifacts:
            if isinstance(item, dict) and str(item.get("path", "")).lower().endswith(("llama-server.exe", "llama-server")):
                declared_binary = item.get("sha256")
                break
    source_hashes = profile.get("source_sha256")
    if declared_binary is None and isinstance(source_hashes, dict):
        for path, value in source_hashes.items():
            if str(path).lower().endswith(("llama-server.exe", "llama-server")):
                declared_binary = value
                break
    if not _is_sha256(declared_binary):
        result["reasons"].append("profile binary_sha256 missing or malformed")
    evidence = payload.get("evidence") or {}
    native_binary = evidence.get("native_binary") if isinstance(evidence, dict) else None
    native_binary = native_binary if isinstance(native_binary, dict) else {}
    native_hash = native_binary.get("sha256")
    if not _is_sha256(native_hash):
        result["reasons"].append("native evidence binary SHA missing or malformed")
    elif _is_sha256(declared_binary) and str(declared_binary).lower() != str(native_hash).lower():
        result["reasons"].append("profile/native executable SHA mismatch")
    profile_modules = profile_artifacts or profile.get("dependency_artifacts")
    native_modules = evidence.get("runtime_artifacts") or evidence.get("dependency_artifacts")
    if not isinstance(profile_modules, list) or not profile_modules:
        result["reasons"].append("profile runtime_artifacts missing")
        profile_modules = []
    if not isinstance(native_modules, list) or not native_modules:
        result["reasons"].append("native evidence runtime_artifacts missing")
        native_modules = []
    def module_hashes(values):
        return {str(item.get("sha256")).lower() for item in values
                if isinstance(item, dict) and _is_sha256(item.get("sha256"))}
    profile_hashes = module_hashes(profile_modules)
    native_hashes = module_hashes(native_modules)
    if profile_modules and len(profile_hashes) != len(profile_modules):
        result["reasons"].append("profile runtime_artifacts contains missing/malformed SHA")
    if native_modules and len(native_hashes) != len(native_modules):
        result["reasons"].append("native evidence runtime_artifacts contains missing/malformed SHA")
    if profile_hashes and native_hashes and profile_hashes != native_hashes:
        result["reasons"].append("profile/native dependency SHA set mismatch")
    # A dependency manifest must include at least one non-executable module;
    # otherwise an EXE-only hash is being mislabeled as complete CUDA
    # provenance.
    if profile_modules and not any(
            isinstance(item, dict) and str(item.get("path", "")).lower().endswith(".dll")
            for item in profile_modules):
        result["reasons"].append("profile runtime_artifacts has no dependency module")
    if result["reasons"]:
        return result
    result["status"] = "pass"
    result["profile_binary_sha256"] = str(declared_binary).lower()
    result["native_binary_sha256"] = str(native_hash).lower()
    result["dependency_sha256"] = sorted(profile_hashes)
    return result


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

def _native_extractor_identity():
    """Return the comparator's selected-function implementation identity."""
    from native_llama_compare import native_extractor_identity
    return native_extractor_identity()


def _native_measurements_digest(native):
    """Hash the immutable native result object used by an L1 replay.

    ``native_contract_sha256`` intentionally covers configuration and request
    identity, but historically did not cover the measured counters.  Keeping
    this digest beside the evidence contract prevents a caller from changing
    ``native.aggregate``/``native.timings`` after capture and still presenting
    the result as a valid engine target.  The digest is over the complete
    native object (which also includes per-request boundaries and perf output)
    and never includes the outer evidence object, avoiding a self-reference.
    """
    if not isinstance(native, dict):
        return stable_hash({})
    measured = dict(native)
    # The evidence manifest lives inside ``native`` in persisted payloads;
    # exclude it so the digest covers only captured measurements and is not
    # self-referential.  This matches native_llama_compare's producer path.
    measured.pop("evidence", None)
    return stable_hash(measured)

def _timing_contract():
    return {
        "id": "engine-stage+client-real-token/v3",
        "engine_ttft": "first_engine_token-engine_request_begin;counter=t_prompt_last-t_start",
        "engine_e2e": "last_engine_token-engine_request_begin;counter=t_gen_last-t_start",
        "engine_tpot": "(last_engine_token-first_engine_token)/(output_tokens-1);counter=(t_gen_last-t_prompt_last)/(output_tokens-1);null_if_output_tokens<=1",
        "ttft": "first_real_token-client_request_start",
        "e2e": "last_real_token-client_request_start;DONE_excluded",
        "tpot": "(last_real_token-first_real_token)/(output_tokens-1);null_if_output_tokens<=1",
        "unit": "ms", "clock": "time.perf_counter",
    }


def _explicit_native_engine_values(native_aggregate):
    """Read only fields proven by the engine timing contract.

    Prompt/eval/total counters are intentionally excluded because their
    boundaries differ from engine request/first-token/last-token markers.
    """
    aggregate = native_aggregate if isinstance(native_aggregate, dict) else {}
    return {
        "engine_ttft_ms": (aggregate.get("engine_ttft_ms") or {}).get("p50_ms"),
        "engine_tpot_ms": (aggregate.get("engine_tpot_ms") or {}).get("p50_ms"),
        "engine_e2e_ms": (aggregate.get("engine_e2e_ms") or {}).get("p50_ms"),
    }

def _validate_native_evidence(payload, source_path=None):
    """Fail closed unless immutable native evidence is present and coherent."""
    errors = []
    native = payload.get("native") or {}
    identity = payload.get("identity") or native.get("identity") or {}
    required_identity = ("model_path", "gguf_sha256", "runtime_fingerprint",
                         "hardware_fingerprint", "prompt_fingerprint", "configuration")
    for field in required_identity:
        value = identity.get(field)
        if value in (None, "", {}, []):
            errors.append(f"identity.{field} missing")
    if payload.get("model") and identity.get("model_path") and Path(str(payload["model"])).resolve() != Path(str(identity["model_path"])).resolve():
        errors.append("model path identity mismatch")
    request = payload.get("request") or {}
    output_policy = payload.get("output_policy") or {}
    if request.get("prompt") in (None, ""):
        errors.append("request.prompt missing")
    elif stable_hash(request.get("prompt")) != identity.get("prompt_fingerprint"):
        errors.append("prompt fingerprint recomputation mismatch")
    if request.get("prompt_fingerprint") != identity.get("prompt_fingerprint"):
        errors.append("request.prompt_fingerprint mismatch")
    if request.get("output_mode") is None or output_policy.get("mode") != request.get("output_mode"):
        errors.append("output policy mismatch")
    if output_policy.get("ignore_eos") is None or output_policy.get("ignore_eos") != request.get("ignore_eos"):
        errors.append("output policy ignore_eos mismatch")
    config = payload.get("configuration") or {}
    identity_config = identity.get("configuration") or {}
    for field in ("ctx", "parallel", "batch", "ubatch", "threads", "gpu_layers"):
        if field not in config or field not in identity_config or config[field] != identity_config[field]:
            errors.append(f"configuration.{field} mismatch")
    counts = payload.get("token_counts") or {}
    output_tokens = counts.get("output")
    parallel = int(config.get("parallel", 1) or 1)
    requests = native.get("requests") or []
    if not requests:
        requests = [{"request_boundary": native.get("request_boundary")}]
    if len(requests) != parallel:
        errors.append(f"request_boundaries count {len(requests)} != parallel {parallel}")
    for index, request in enumerate(requests):
        boundary = request.get("request_boundary") or {}
        timestamps = boundary.get("token_chunk_times_ms")
        expected_output = request.get("output_tokens", output_tokens)
        if boundary.get("status") != "measured":
            errors.append(f"request[{index}] boundary not measured")
        if not isinstance(timestamps, list) or not timestamps:
            errors.append(f"request[{index}] token timestamps missing")
        elif isinstance(expected_output, int) and len(timestamps) != expected_output:
            errors.append(f"request[{index}] token timestamp count {len(timestamps)} != output {expected_output}")
        if boundary.get("request_to_first_token_ms") is None or boundary.get("request_to_end_ms") is None:
            errors.append(f"request[{index}] request boundary timestamps missing")
    manifest_present = "evidence" in payload or "evidence" in native
    evidence_raw = payload.get("evidence", native.get("evidence")) if manifest_present else None
    if manifest_present and not isinstance(evidence_raw, dict):
        errors.append("native evidence manifest missing")
    evidence = evidence_raw if isinstance(evidence_raw, dict) else {}
    # A v2 unified manifest is optional for legacy payloads, but when present
    # it is authoritative for sidecar/raw-trace binding.  Keep this check
    # fail-closed and independent from any timing fitting or replay logic.
    unified_ref = evidence.get("unified_manifest") or payload.get("evidence_manifest")
    if unified_ref:
        try:
            from unified_evidence_manifest import validate_manifest
            unified = unified_ref
            if isinstance(unified_ref, str):
                manifest_path = _resolve_evidence_path(unified_ref, source_path)
                if manifest_path is None or not manifest_path.exists():
                    errors.append("unified evidence manifest missing")
                    unified = None
                else:
                    unified = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(unified, dict):
                # A unified manifest used for simulator-only replay feeds the
                # L1 engine gate.  Require the frozen v3 contract here so a
                # stale client-only/v1 manifest cannot silently pass replay.
                errors.extend("unified_manifest." + item for item in validate_manifest(
                    unified, require_engine=True, require_engine_contract=True))
            elif unified is not None:
                errors.append("unified evidence manifest malformed")
        except Exception as exc:
            errors.append(f"unified evidence manifest unavailable: {exc}")
    legacy_contract = False
    if manifest_present:
        required_sections = ("native_binary", "extractor", "raw_trace_events", "extractor_output",
                             "timing_contract", "token_timestamps", "request_boundaries")
        for section in required_sections:
            value = evidence.get(section)
            if not isinstance(value, dict) or not value:
                errors.append(f"evidence.{section} missing")
        for section in ("native_binary", "extractor", "raw_trace_events"):
            value = evidence.get(section) or {}
            if value.get("status") != "captured":
                errors.append(f"evidence.{section}.status not captured")
            if not value.get("path") or not value.get("sha256"):
                errors.append(f"evidence.{section} path/sha256 missing")
        contract = evidence.get("timing_contract") or {}
        legacy_contract = contract.get("id") == "client-real-token/v1"
        engine_section = evidence.get("engine_timing")
        if engine_section is not None and (not isinstance(engine_section, dict)
                                           or engine_section.get("status") in (None, "unavailable")):
            errors.append("evidence.engine_timing status unavailable")
        engine_status = engine_section.get("status") if isinstance(engine_section, dict) else None
        # A v3 timing contract without a proven engine section is still a
        # client-only payload with hand-added labels. Keep it out of L1 even
        # when aggregate fields happen to be present.
        if not legacy_contract and engine_status not in ("counter_proven", "marker_proven"):
            errors.append("evidence.engine_timing not proven")
        if not legacy_contract and engine_status in ("counter_proven", "marker_proven"):
            fields = tuple(engine_section.get("fields") or ())
            if fields != ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"):
                errors.append("evidence.engine_timing fields incomplete")
            if not engine_section.get("source"):
                errors.append("evidence.engine_timing source missing")
            digest = evidence.get("native_measurements_sha256")
            if not _is_sha256(digest):
                errors.append("native measurements sha256 missing or malformed")
            elif digest.lower() != _native_measurements_digest(native):
                errors.append("native measurements sha256 mismatch")
        extractor_identity = evidence.get("extractor") or {}
        try:
            current_extractor = _native_extractor_identity()
            if extractor_identity.get("implementation_sha256") != current_extractor.get("implementation_sha256"):
                if not legacy_contract:
                    errors.append("extractor implementation sha256 mismatch")
            if extractor_identity.get("functions") != current_extractor.get("functions"):
                if not legacy_contract:
                    errors.append("extractor function list mismatch")
            if extractor_identity.get("sha256_basis") != current_extractor.get("sha256_basis"):
                if not legacy_contract:
                    errors.append("extractor sha256 basis mismatch")
        except Exception as exc:
            errors.append(f"extractor identity unavailable: {exc}")
        expected_contract = _timing_contract()
        if (not legacy_contract and (contract.get("id") != expected_contract["id"] or
              any(contract.get(k) != expected_contract[k] for k in expected_contract if k != "id"))):
            errors.append("timing contract mismatch")
        if (not legacy_contract and contract.get("sha256") != hashlib.sha256(json.dumps(expected_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
            errors.append("timing contract sha256 mismatch")
        native_contract_sha = evidence.get("native_contract_sha256")
        contract_input = {key: payload.get(key) for key in ("command", "configuration", "request", "output_policy", "token_counts", "runtime_config", "identity", "hardware")}
        if not native_contract_sha:
            errors.append("native contract sha256 missing")
        elif native_contract_sha != stable_hash(contract_input):
            errors.append("native contract sha256 mismatch")
    raw = evidence.get("raw_trace_events") or {}
    if evidence and raw and not raw.get("sha256"):
        errors.append("raw trace event sha256 missing")
    raw_path_value = raw.get("path") or payload.get("log")
    raw_path = _resolve_evidence_path(raw_path_value, source_path)
    if raw_path is None or not raw_path.exists():
        errors.append("raw trace event file missing")
    elif raw.get("sha256") and raw.get("sha256") != _sha256_file(raw_path):
        errors.append("raw trace event sha256 mismatch")
    if manifest_present:
        for section_name in ("native_binary", "extractor"):
            ref = evidence.get(section_name) or {}
            ref_path = _resolve_evidence_path(ref.get("path"), source_path)
            if ref_path is None or not ref_path.exists():
                errors.append(f"{section_name} artifact missing")
            elif section_name == "native_binary" and ref.get("sha256") != _sha256_file(ref_path):
                errors.append(f"{section_name} artifact sha256 mismatch")
            elif section_name == "extractor" and ref.get("file_sha256") and ref.get("file_sha256") != _sha256_file(ref_path):
                errors.append(f"{section_name} file sha256 mismatch")
        binary = evidence.get("native_binary") or {}
        command = payload.get("command") or []
        if binary.get("path") and command and Path(str(binary["path"])).resolve() != Path(str(command[0])).resolve():
            errors.append("native binary path/command mismatch")
    for index, artifact in enumerate(raw.get("supplemental_artifacts") or []):
        artifact_path = _resolve_evidence_path((artifact or {}).get("path"), source_path)
        if artifact_path is None or not artifact_path.exists():
            errors.append(f"supplemental raw trace[{index}] missing")
        elif not (artifact or {}).get("sha256"):
            errors.append(f"supplemental raw trace[{index}] sha256 missing")
        elif artifact["sha256"] != _sha256_file(artifact_path):
            errors.append(f"supplemental raw trace[{index}] sha256 mismatch")
    for section_name in ("extractor_output_artifacts", "extractor_script_artifacts"):
        for index, artifact in enumerate(evidence.get(section_name) or []):
            artifact_path = _resolve_evidence_path((artifact or {}).get("path"), source_path)
            if artifact_path is None or not artifact_path.exists():
                errors.append(f"{section_name}[{index}] missing")
            elif not (artifact or {}).get("sha256"):
                errors.append(f"{section_name}[{index}] sha256 missing")
            elif artifact["sha256"] != _sha256_file(artifact_path):
                errors.append(f"{section_name}[{index}] sha256 mismatch")
    extractor = evidence.get("extractor_output") or {}
    if evidence and extractor and not extractor.get("sha256"):
        errors.append("extractor output sha256 missing")
    perf_log = native.get("perf_log")
    if not isinstance(perf_log, dict) or not any(perf_log.get(key) for key in ("prompt", "decode", "total", "graphs_reused")):
        errors.append("extractor output missing")
    if extractor.get("sha256"):
        actual = hashlib.sha256(json.dumps(perf_log, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if actual != extractor["sha256"]:
            errors.append("extractor output sha256 mismatch")
    return errors, {"identity": identity, "raw_trace_path": str(raw_path) if raw_path else None,
                    "native_binary": evidence.get("native_binary"), "extractor": evidence.get("extractor"),
                    "timing_contract": evidence.get("timing_contract"),
                    "raw_trace_sha256": _sha256_file(raw_path) if raw_path and raw_path.exists() else None,
                    "extractor_output": perf_log, "status": "complete" if not errors else "invalid",
                    "evidence_level": "legacy_development" if legacy_contract else ("complete" if manifest_present and not errors else ("legacy_development" if not errors else "invalid")),
                    "timing_contract_legacy": legacy_contract,
                    "errors": errors}

def p50(values):
    return statistics.median([float(v) for v in values if v is not None]) if values else None

def error(native, simulated):
    if native in (None, 0) or simulated is None: return None
    return 100.0 * (float(simulated) - float(native)) / float(native)

def replay(payload, model_key, source_path=None):
    evidence_errors, evidence_summary = _validate_native_evidence(payload, source_path=source_path)
    if evidence_errors:
        return {"status": "invalid", "reason": "native evidence contract: " + "; ".join(evidence_errors), "evidence": evidence_summary}
    config = payload.get("configuration") or {}
    counts = payload.get("token_counts") or {}
    prompt_tokens = counts.get("prompt")
    output_tokens = counts.get("output")
    prompt = (payload.get("request") or {}).get("prompt")
    model_path = Path(payload.get("model") or MODEL_PATHS[model_key])
    if prompt is None or not isinstance(prompt_tokens, int) or not isinstance(output_tokens, int):
        return {"status": "invalid", "reason": "missing immutable prompt/token evidence"}
    requested_output = (payload.get("request") or {}).get("requested_output_tokens")
    requested_output = requested_output if isinstance(requested_output, int) else counts.get("requested_output")
    if not isinstance(requested_output, int) or requested_output < 1:
        return {"status": "invalid", "reason": "missing requested output shape"}
    if (payload.get("request") or {}).get("output_mode") == "natural":
        return {"status": "unsupported", "evidence_level": "conditional_replay",
                "reason": "natural-EOS replay cannot use actual output as prediction shape"}
    model = build_model_from_gguf(read_gguf_metadata(model_path))
    scenario = build_matching_scenario(prompt_tokens, requested_output, ctx=int(config.get("ctx", 2048)),
                                       parallel=int(config.get("parallel", 1)), batch=int(config.get("batch", 64)),
                                       ubatch=int(config.get("ubatch", 64)), threads=int(config.get("threads", 16)),
                                       gpu_layers=int(config.get("gpu_layers", -1)), model=model,
                                       hardware_snapshot=payload.get("hardware"))
    profile, stage, memory, phase = PROFILES[model_key]
    profile_gate = {"status": "not_requested", "profile": str(profile) if profile is not None else None,
                    "reasons": []}
    if profile is not None and profile.exists() and (stage or memory or phase):
        profile_gate = _profile_binary_gate(profile, payload)
        if profile_gate.get("status") == "pass":
            scenario = apply_native_calibration(scenario, load_native_calibration(profile), apply_stage=stage,
                                                apply_memory=memory, apply_phase_boundary=phase)
        else:
            # Keep the simulator's analytical mechanism and expose the
            # blocked calibration explicitly.  Silently applying an old
            # profile would mix binaries/DLLs and invalidate L1 evidence.
            metadata = dict(scenario.placement.metadata)
            metadata["native_calibration_profile_gate"] = profile_gate
            from dataclasses import replace as _replace
            scenario = _replace(scenario, placement=_replace(scenario.placement, metadata=metadata))
    elif profile is not None and not profile.exists() and (stage or memory or phase):
        profile_gate = {"status": "blocked", "profile": str(profile),
                        "reasons": ["profile file missing"]}
    sim = run_scenario(scenario, retention_policy="aggregate")
    native = payload.get("native") or {}
    native_agg = native.get("aggregate") or {}
    sim_requests = []
    for metric in sim.metrics.request_metrics.values():
        arrival = float(getattr(metric, "arrival_ns", 0.0) or 0.0)
        request_id = str(getattr(metric, "request_id", ""))
        # ServingRequestMetrics carries the post-admission engine start; the
        # aggregate metrics object used by replay does not.
        serving_metrics = getattr(getattr(sim, "serving", None), "request_metrics", {}) or {}
        serving_metric = serving_metrics.get(request_id) if isinstance(serving_metrics, dict) else None
        start = getattr(serving_metric, "start_ns", None)
        if start is None:
            start = getattr(metric, "start_ns", None)
        first = getattr(metric, "first_token_ns", None)
        finish = getattr(metric, "finish_ns", None)
        token_count = int(getattr(metric, "visible_output_tokens", 0) or 0)
        token_events = [event for event in getattr(sim.serving, "events", ()) or ()
                        if str(getattr(event, "request_id", "")) == request_id
                        and str(getattr(event, "event_type", "")) == "tokens_committed"
                        and getattr(event, "timestamp_ns", None) is not None]
        last_token = max((float(getattr(event, "timestamp_ns")) for event in token_events), default=None)
        if last_token is None and getattr(metric, "last_token_ns", None) is not None:
            last_token = float(getattr(metric, "last_token_ns"))
        client_ttft_ms = ((float(first) - arrival) / 1e6 if first is not None else None)
        tpot_ms = float(metric.tpot_ns) / 1e6 if getattr(metric, "tpot_ns", None) is not None else None
        engine_ttft_ms = ((float(first) - float(start)) / 1e6
                          if first is not None and start is not None else None)
        engine_e2e_ms = ((last_token - float(start)) / 1e6
                         if last_token is not None and start is not None else None)
        client_e2e_ms = (client_ttft_ms + tpot_ms * (token_count - 1)
                         if client_ttft_ms is not None and tpot_ms is not None and token_count > 1 else
                         ((float(finish) - arrival) / 1e6 if finish is not None else None))
        sim_requests.append({
            "ttft_ms": client_ttft_ms, "tpot_ms": tpot_ms,
            "client_ttft_ms": client_ttft_ms, "client_tpot_ms": tpot_ms,
            "engine_ttft_ms": engine_ttft_ms, "engine_tpot_ms": tpot_ms,
            "e2e_ms": client_e2e_ms, "client_e2e_ms": client_e2e_ms,
            "engine_e2e_ms": engine_e2e_ms,
            "engine_ttft_source": "simulator.first_engine_token-start" if engine_ttft_ms is not None else "unavailable",
            "visible_output_tokens": token_count,
        })
    sim_values = {m: [r[m] for r in sim_requests if r[m] is not None]
                  for m in ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")}
    # Only explicit engine fields are eligible for the一级 engine comparison.
    # Legacy prompt_eval/total counters have different boundaries and must not
    # be relabelled or backfilled into TTFT_engine/E2E_engine.
    explicit_native_engine = _explicit_native_engine_values(native_agg)
    native_engine_ttft = explicit_native_engine["engine_ttft_ms"]
    native_engine_tpot = explicit_native_engine["engine_tpot_ms"]
    native_engine_e2e = explicit_native_engine["engine_e2e_ms"]
    native_values = {
        "engine_ttft_ms": native_engine_ttft,
        "engine_tpot_ms": native_engine_tpot,
        "engine_e2e_ms": native_engine_e2e,
    }
    sim_values = {m: p50(v) for m, v in sim_values.items()}
    def _agg_metric(primary: str, *fallbacks: str):
        for name in (primary,) + fallbacks:
            value = (native_agg.get(name) or {}).get("p50_ms")
            if value is not None:
                return value
        return None
    sim_client_values = {
        "ttft_ms": p50([r.get("client_ttft_ms") for r in sim_requests]),
        "tpot_ms": p50([r.get("client_tpot_ms") for r in sim_requests]),
        "e2e_ms": p50([r.get("client_e2e_ms") for r in sim_requests]),
    }
    explicit_engine = all(native_agg.get(name, {}).get("p50_ms") is not None
                           for name in ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"))
    row_status = "valid" if evidence_summary.get("evidence_level") == "complete" and explicit_engine else "legacy_development"
    return {"status": row_status, "evidence_level": evidence_summary.get("evidence_level"), "acceptance_eligible": False,
            "model_key": model_key, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "parallel": int(config.get("parallel", 1)), "native": native_values, "simulator": sim_values,
            "relative_error_pct": {m: error(native_values[m], sim_values[m]) for m in native_values},
            "relative_error_client_pct": {
                "ttft_ms": error(_agg_metric("client_ttft_ms", "request_to_first_token_ms"), sim_client_values["ttft_ms"]),
                "tpot_ms": error(_agg_metric("client_tpot_ms", "tpot_ms"), sim_client_values["tpot_ms"]),
                "e2e_ms": error(_agg_metric("client_e2e_ms", "request_to_end_ms"), sim_client_values["e2e_ms"]),
            },
            "absolute_delta_ms": {m: (abs(sim_values[m] - native_values[m]) if native_values[m] is not None and sim_values[m] is not None else None) for m in native_values},
            "simulator_request_count": len(sim_requests), "simulator_output_tokens": [r["visible_output_tokens"] for r in sim_requests],
            "prefill_chunk_tokens": getattr(scenario.workload.scheduler, "prefill_chunk_tokens", None),
            "native_identity": (payload.get("identity") or payload.get("native", {}).get("identity")),
            "profile_gate": profile_gate,
            "engine_timing_status": "measured" if explicit_engine else "legacy_counter_diagnostic",
            "native_request_boundary": native.get("request_boundary"),
            "native_evidence": evidence_summary,
            "source_payload": str(payload.get("model", ""))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source-dir", action="append", required=True,
        help="directory containing immutable native payload JSON files; repeat per model")
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    rows=[]
    skipped_non_payload = 0
    skipped_unknown_model = 0
    for raw in args.source_dir:
        directory=Path(raw)
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".prediction.json"): continue
            try: payload=json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc: rows.append({"status":"invalid","source":str(path),"reason":f"json:{exc}"}); continue
            # A source directory may also contain calibration, coverage,
            # manifest, or freeze-summary artifacts. They are not native
            # payloads and must not be reported as invalid replay attempts.
            if (not isinstance(payload, dict)
                    or not isinstance(payload.get("native"), dict)
                    or not payload.get("model")
                    or not isinstance(payload.get("request"), dict)
                    or not isinstance(payload.get("identity"), dict)):
                skipped_non_payload += 1
                continue
            model_key=next((m for m in MODEL_PATHS if path.stem.startswith(m + "__")), None)
            if model_key is None:
                model_text = str(payload.get("model", "")).lower()
                model_aliases = {
                    "qwen25": ("qwen25", "qwen2.5"),
                    "qwen35": ("qwen35", "qwen3.5"),
                    "qwen38": ("qwen38", "qwen3.8"),
                    "tinyllama": ("tinyllama",),
                    "smollm2": ("smollm2", "smollm-2"),
                }
                model_key = next((key for key, aliases in model_aliases.items()
                                  if any(alias in model_text for alias in aliases)), None)
            if model_key is None:
                skipped_unknown_model += 1
                continue
            item=replay(payload, model_key, source_path=path); item["source"]=str(path); item["cell_id"]=path.stem; rows.append(item)
            print(json.dumps({"cell_id":path.stem,"status":item["status"],"done":len(rows)},ensure_ascii=False),flush=True)
    # This command only replays immutable payloads through the simulator.
    # Keep native execution explicitly zero; ``payload_scan_count`` records
    # how many saved payloads were inspected, which is a different quantity.
    replayed_count = sum(1 for row in rows if row.get("status") in ("valid", "legacy_development"))
    calibration_manifest = {}
    for model_key, (profile_path, apply_stage, apply_memory, apply_phase) in PROFILES.items():
        row = {"profile": str(profile_path) if profile_path is not None else None,
               "apply_stage": bool(apply_stage), "apply_memory": bool(apply_memory),
               "apply_phase_boundary": bool(apply_phase)}
        if profile_path is not None and profile_path.exists():
            row["profile_sha256"] = _sha256_file(profile_path)
        else:
            row["profile_sha256"] = None
        calibration_manifest[model_key] = row
    out={"schema":"simulator-replay-from-native/v1","purpose":"mechanism_validation","native_execution_count":0,
         "payload_scan_count":len(rows),
         "identity_reuse_count":replayed_count,
         "native_actual_reused_count":replayed_count,
         "simulator_execution_count":replayed_count,
         "valid_replay_count":sum(x.get("status")=="valid" for x in rows),
         "legacy_development_count":sum(x.get("status")=="legacy_development" for x in rows),
         "invalid_replay_count":sum(x.get("status") not in ("valid", "legacy_development") for x in rows),
         "rows":rows,"skipped_non_payload_count":skipped_non_payload,
         "skipped_unknown_model_count":skipped_unknown_model,
         "replay_calibration": calibration_manifest,
         "policy":"Native payloads, token timestamps and boundaries are immutable inputs; only simulator is rerun.",
         "limitations":["Replay is valid only while native binary/model/hardware/CLI/prompt-output policy/timing contract/extractor identity remains unchanged.","Natural-EOS output-length changes require a new native reveal."]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output.resolve()),"records":len(rows),"valid":out["valid_replay_count"]},ensure_ascii=False))
if __name__ == "__main__": main()
