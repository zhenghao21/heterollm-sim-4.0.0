import hashlib
import json
from pathlib import Path

from tools.unified_evidence_manifest import (
    build_manifest,
    validate_manifest,
    validate_prediction_before_native,
)


def _payload(tmp_path: Path):
    binary = tmp_path / "llama-server.exe"
    trace = tmp_path / "trace.sqlite"
    extractor = tmp_path / "extract.py"
    for path, data in ((binary, b"binary"), (trace, b"trace"), (extractor, b"extractor")):
        path.write_bytes(data)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {"model_path": str(tmp_path / "model.gguf"), "gguf_sha256": "m",
                "runtime_fingerprint": "r", "hardware_fingerprint": "h",
                "prompt_fingerprint": "p", "configuration": {"ctx": 8}}
    return {
        "identity": identity,
        "evidence": {
            "native_binary": {"path": str(binary), "sha256": digest(binary)},
            "raw_trace_events": {"path": str(trace), "sha256": digest(trace),
                                  "supplemental_artifacts": []},
            "extractor": {"path": str(extractor), "file_sha256": digest(extractor)},
            "timing_contract": {"id": "engine/v1", "engine_ttft": "start-first", "ttft": "client"},
            "engine_timing": {"status": "counter_proven", "source": "server_slot_stats"},
            "token_timestamps": {"status": "captured"},
            "request_boundaries": {"status": "captured"},
            "native_contract_sha256": "contract",
            "extractor_output": {"status": "captured"},
        },
    }


def test_unified_manifest_binds_engine_client_and_artifacts(tmp_path):
    payload = _payload(tmp_path)
    manifest = build_manifest(payload, source_path=tmp_path / "payload.json", root=tmp_path)
    assert manifest["schema"] == "native-evidence-manifest/v2"
    assert manifest["boundaries"]["engine"]["status"] == "counter_proven"
    assert manifest["boundaries"]["client"]["status"] == "captured"
    assert {item["role"] for item in manifest["artifacts"]} == {"native.binary", "raw.trace", "extractor"}
    assert validate_manifest(manifest) == []


def test_unified_manifest_recursively_binds_b06_b08_sidecars(tmp_path):
    payload = _payload(tmp_path)
    sidecar_file = tmp_path / "kernel.sqlite"
    sidecar_file.write_bytes(b"sqlite")
    sidecar = tmp_path / "b06.json"
    sidecar.write_text(json.dumps({"schema": "b06", "train_trace": str(sidecar_file),
                                   "profile_artifacts": [{"path": str(sidecar_file), "sha256": hashlib.sha256(sidecar_file.read_bytes()).hexdigest()}]}), encoding="utf-8")
    manifest = build_manifest(payload, sidecars=[sidecar], source_path=tmp_path / "payload.json", root=tmp_path)
    assert manifest["sidecar_manifests"][0]["exists"] is True
    assert sum(item["role"] == "train_trace.trace" for item in manifest["artifacts"]) == 1
    assert validate_manifest(manifest) == []


def test_unified_manifest_fails_closed_on_changed_artifact(tmp_path):
    payload = _payload(tmp_path)
    manifest = build_manifest(payload, source_path=tmp_path / "payload.json", root=tmp_path)
    Path(manifest["artifacts"][0]["path"]).write_bytes(b"changed")
    errors = validate_manifest(manifest)
    assert any("sha256 mismatch" in error for error in errors)


def test_unified_manifest_can_audit_development_sidecar_without_engine(tmp_path):
    payload = _payload(tmp_path)
    payload["evidence"]["engine_timing"] = {"status": "unavailable"}
    manifest = build_manifest(payload, source_path=tmp_path / "payload.json", root=tmp_path)
    assert validate_manifest(manifest, require_engine=False) == []
    assert validate_manifest(manifest, require_engine=True)


def test_replay_validator_checks_explicit_unified_manifest(tmp_path):
    from tools.replay_simulator_from_native import _validate_native_evidence
    payload = _payload(tmp_path)
    # Deliberately attach only the unified object to a minimal payload.  The
    # normal native contract errors are expected; the important assertion is
    # that a changed sidecar is surfaced through the unified namespace.
    manifest = build_manifest(payload, source_path=tmp_path / "payload.json", root=tmp_path)
    manifest["artifacts"][0]["sha256"] = "bad"
    payload["evidence"]["unified_manifest"] = manifest
    errors, _ = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert any(error.startswith("unified_manifest.artifacts[0] sha256 mismatch") for error in errors)


def test_engine_v3_manifest_is_strict_and_server_unavailable_is_explicit():
    """The development D-01 manifest is L1-valid, while L2 stays fail-closed."""
    manifest_path = (Path(__file__).resolve().parents[1] /
                     "artifacts/development/engine_contract_probe_evidence_manifest_v2.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validate_manifest(manifest, require_engine=True, require_engine_contract=True) == []
    assert manifest["boundaries"]["server"]["status"] == "unavailable"

    missing_server = json.loads(json.dumps(manifest))
    missing_server["boundaries"].pop("server")
    errors = validate_manifest(missing_server, require_engine=True, require_engine_contract=True)
    assert "server boundary status missing or invalid" in errors


def test_engine_v3_manifest_rejects_timing_contract_tampering():
    manifest_path = (Path(__file__).resolve().parents[1] /
                     "artifacts/development/engine_contract_probe_evidence_manifest_v2.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["timing_contract"]["engine_ttft"] = "client-only"
    errors = validate_manifest(manifest, require_engine=True, require_engine_contract=True)
    assert "timing_contract.sha256 mismatch" in errors


def test_engine_v3_manifest_binds_native_measurements_digest():
    manifest_path = (Path(__file__).resolve().parents[1] /
                     "artifacts/development/engine_contract_probe_evidence_manifest_v2.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validate_manifest(manifest, require_engine=True, require_engine_contract=True) == []
    tampered = json.loads(json.dumps(manifest))
    tampered["native_measurements_sha256"] = "0" * 64
    errors = validate_manifest(tampered, require_engine=True, require_engine_contract=True)
    assert "native_measurements_sha256 manifest/source mismatch" in errors


def test_prediction_artifact_must_precede_native_reveal(tmp_path):
    prediction = tmp_path / "run.prediction.json"
    prediction.write_text(json.dumps({
        "status": "saved_before_native_reveal",
        "created_utc": "2026-09-14T00:00:00+00:00",
    }), encoding="utf-8")
    payload = {
        "prediction_artifact": str(prediction),
        "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "prediction_status": "saved_before_native_reveal",
        "native_reveal_timestamp_utc": "2026-09-14T00:00:01+00:00",
    }
    assert validate_prediction_before_native(payload, source_path=tmp_path / "payload.json") == []
    late = dict(payload, native_reveal_timestamp_utc="2026-09-13T23:59:59+00:00")
    assert "prediction_artifact was not saved before native reveal" in validate_prediction_before_native(
        late, source_path=tmp_path / "payload.json"
    )


def test_prediction_artifact_hash_and_status_are_fail_closed(tmp_path):
    prediction = tmp_path / "run.prediction.json"
    prediction.write_text(json.dumps({
        "status": "saved_before_native_reveal",
        "created_utc": "2026-09-14T00:00:00+00:00",
    }), encoding="utf-8")
    payload = {
        "prediction_artifact": str(prediction),
        "prediction_sha256": "0" * 64,
        "prediction_status": "saved_before_native_reveal",
        "native_reveal_timestamp_utc": "2026-09-14T00:00:01+00:00",
    }
    assert "prediction_artifact sha256 mismatch" in validate_prediction_before_native(payload)
    payload["prediction_status"] = "missing"
    assert "prediction_status not saved_before_native_reveal" in validate_prediction_before_native(payload)
