import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from replay_simulator_from_native import (  # noqa: E402
    _profile_binary_gate,
    _validate_native_evidence,
)


def _payload(binary_sha="a" * 64, modules=None):
    return {"evidence": {"native_binary": {"sha256": binary_sha},
                          "runtime_artifacts": modules or []}}


def test_profile_gate_rejects_missing_capture_dependency_manifest(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"binary_sha256": "a" * 64}), encoding="utf-8")
    result = _profile_binary_gate(profile, _payload("a" * 64))
    assert result["status"] == "blocked"
    assert "profile runtime_artifacts missing" in result["reasons"]
    assert "native evidence runtime_artifacts missing" in result["reasons"]


def test_profile_gate_accepts_matching_exe_and_dll_manifest(tmp_path):
    modules = [{"path": "llama-server.exe", "sha256": "a" * 64},
               {"path": "ggml-cuda.dll", "sha256": "b" * 64}]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"native_binary_artifacts": modules}), encoding="utf-8")
    result = _profile_binary_gate(profile, _payload("a" * 64, modules))
    assert result["status"] == "pass"
    assert result["dependency_sha256"] == ["a" * 64, "b" * 64]


def test_profile_gate_rejects_executable_mismatch(tmp_path):
    modules = [{"path": "llama-server.exe", "sha256": "a" * 64},
               {"path": "ggml-cuda.dll", "sha256": "b" * 64}]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"native_binary_artifacts": modules}), encoding="utf-8")
    result = _profile_binary_gate(profile, _payload("c" * 64, modules))
    assert result["status"] == "blocked"
    assert "profile/native executable SHA mismatch" in result["reasons"]


def test_profile_gate_does_not_invalidate_native_evidence(tmp_path):
    """A blocked profile must leave the immutable native evidence valid.

    The gate controls whether a calibration profile may be applied.  It must
    not relabel a complete native measurement as invalid merely because the
    optional profile came from a different runtime identity.
    """
    source = (Path(__file__).resolve().parents[1] /
              "artifacts/development/engine_contract_probe_engine_v3.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    # The checked-in development payload intentionally points at a historical
    # binary.  Rebind only the test copy to an immutable temporary executable
    # so the evidence validator exercises the complete contract without
    # depending on the current worktree binary.
    import hashlib
    import shutil
    from replay_simulator_from_native import stable_hash
    executable = tmp_path / "llama-server.exe"
    shutil.copy2(sys.executable, executable)
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    payload["command"][0] = str(executable)
    payload["evidence"]["native_binary"].update({
        "path": str(executable), "sha256": executable_sha,
        "bytes": executable.stat().st_size,
    })
    raw_trace = tmp_path / "native.log"
    raw_trace.write_text("captured native event\n", encoding="utf-8")
    payload["evidence"]["raw_trace_events"].update({
        "path": str(raw_trace),
        "sha256": hashlib.sha256(raw_trace.read_bytes()).hexdigest(),
    })
    contract_input = {key: payload.get(key) for key in (
        "command", "configuration", "request", "output_policy", "token_counts",
        "runtime_config", "identity", "hardware")}
    payload["evidence"]["native_contract_sha256"] = stable_hash(contract_input)
    profile = tmp_path / "mismatched-profile.json"
    profile.write_text(json.dumps({
        "native_binary_artifacts": [
            {"path": "llama-server.exe", "sha256": "0" * 64},
            {"path": "ggml-cuda.dll", "sha256": "1" * 64},
        ]
    }), encoding="utf-8")

    errors, summary = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert errors == []
    assert summary["status"] == "complete"
    gate = _profile_binary_gate(profile, payload)
    assert gate["status"] == "blocked"
    assert payload["evidence"]["native_measurements_sha256"]
    # This is the contract boundary: the native payload remains reusable for
    # simulator-only replay, while the mismatched profile is not applied.
    assert payload.get("native_execution_count", 0) == 0
