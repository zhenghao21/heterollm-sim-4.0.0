from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from tools import native_runtime_evidence as evidence


def _fixture_modules(tmp_path: Path):
    exe = tmp_path / "llama-server.exe"
    dll = tmp_path / "ggml-cuda.dll"
    exe.write_bytes(b"executable identity")
    dll.write_bytes(b"loaded library identity")
    snapshot = {
        "method": "mock_actual_loader", "process_identity": "process-start-123",
        "modules": [
            {"path": str(dll), "base_address": 0x2000, "image_size_bytes": 4096},
            {"path": str(exe), "base_address": 0x1000, "image_size_bytes": 4096},
        ],
    }
    return exe, dll, snapshot


def test_actual_modules_exe_first_and_sha(tmp_path, monkeypatch):
    exe, dll, snapshot = _fixture_modules(tmp_path)
    (tmp_path / "unused.dll").write_bytes(b"not loaded")
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: copy.deepcopy(snapshot))
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "captured"
    assert result["errors"] == []
    assert result["method"] == "mock_actual_loader"
    assert result["artifacts"][0]["role"] == "executable"
    assert evidence._path_key(result["artifacts"][0]["path"]) == evidence._path_key(exe)
    assert {Path(row["path"]).name for row in result["artifacts"]} == {exe.name, dll.name}
    assert result["module_sha256_set"] == sorted([
        hashlib.sha256(exe.read_bytes()).hexdigest(), hashlib.sha256(dll.read_bytes()).hexdigest()])
    assert result["module_count"] == 2
    assert result["current_modules"] == result["actual_modules"]
    assert len(result["module_identity_sha256"]) == 64


@pytest.mark.parametrize("failure", [PermissionError("access denied"), ProcessLookupError("not running"), RuntimeError("enumeration failed")])
def test_backend_failure_never_falls_back_to_directory(tmp_path, monkeypatch, failure):
    exe, _, _ = _fixture_modules(tmp_path)
    def fail(pid):
        raise failure
    monkeypatch.setattr(evidence, "_capture_platform_modules", fail)
    monkeypatch.setattr(Path, "glob", lambda *args, **kwargs: pytest.fail("directory scan forbidden"))
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "failed"
    assert result["artifacts"] == result["current_modules"] == []
    assert str(failure) in result["errors"][0]


def test_expected_executable_must_be_loaded(tmp_path, monkeypatch):
    exe, _, snapshot = _fixture_modules(tmp_path)
    snapshot["modules"] = snapshot["modules"][:1]
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: snapshot)
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "failed"
    assert "not in the actual loaded" in result["errors"][0]


@pytest.mark.parametrize("field,value", [("modules", []), ("process_identity", None), ("method", "")])
def test_empty_or_missing_backend_evidence_rejected(tmp_path, monkeypatch, field, value):
    exe, _, snapshot = _fixture_modules(tmp_path)
    snapshot[field] = value
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: snapshot)
    assert evidence.capture_loaded_runtime(123, exe)["status"] == "failed"


@pytest.mark.parametrize("change", ["added", "removed", "pid_reused", "relocated"])
def test_runtime_identity_change_during_capture_rejected(tmp_path, monkeypatch, change):
    exe, _, snapshot = _fixture_modules(tmp_path)
    snapshots = [copy.deepcopy(snapshot), copy.deepcopy(snapshot)]
    if change == "added":
        other = tmp_path / "late.dll"
        other.write_bytes(b"late loaded")
        snapshots[1]["modules"].append({"path": str(other), "base_address": 0x3000, "image_size_bytes": 4096})
    elif change == "removed":
        snapshots[1]["modules"].pop(0)
    elif change == "pid_reused":
        snapshots[1]["process_identity"] = "other-process-start"
    else:
        snapshots[1]["modules"][0]["base_address"] = 0x5000
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: snapshots.pop(0))
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "failed"
    assert "changed during capture" in result["errors"][0]
    assert not result["artifacts"]


def test_module_mutation_during_capture_rejected(tmp_path, monkeypatch):
    exe, dll, snapshot = _fixture_modules(tmp_path)
    calls = 0
    def capture(pid):
        nonlocal calls
        calls += 1
        if calls == 2:
            dll.write_bytes(b"changed after its file hash")
        return copy.deepcopy(snapshot)
    monkeypatch.setattr(evidence, "_capture_platform_modules", capture)
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "failed"
    assert "file changed during capture" in result["errors"][0]


def test_unreadable_module_fail_closed(tmp_path, monkeypatch):
    exe, dll, snapshot = _fixture_modules(tmp_path)
    dll.unlink()
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: snapshot)
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "failed"
    assert not result["current_modules"]


def test_empty_module_fail_closed(tmp_path, monkeypatch):
    exe, dll, snapshot = _fixture_modules(tmp_path)
    dll.write_bytes(b"")
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: snapshot)
    result = evidence.capture_loaded_runtime(123, exe)
    assert result["status"] == "failed"


@pytest.mark.parametrize("pid", [0, -1, True, None, "123"])
def test_invalid_pid_rejected(tmp_path, monkeypatch, pid):
    exe, _, _ = _fixture_modules(tmp_path)
    monkeypatch.setattr(evidence, "_capture_platform_modules", lambda pid: pytest.fail("invalid pid must not reach backend"))
    assert evidence.capture_loaded_runtime(pid, exe)["status"] == "failed"


def test_linux_maps_tracks_executable_modules_excludes_model_and_preserves_spaces():
    rows = evidence._parse_linux_maps(
        "1000-2000 r--p 00000000 08:01 12 /opt/server\n"
        "2000-3000 r-xp 00001000 08:01 12 /opt/server\n"
        "3000-4000 r--p 00000000 08:01 13 /opt/lib space.so\n"
        "4000-5000 r-xp 00001000 08:01 13 /opt/lib space.so\n"
        "5000-9000 r--p 00000000 08:01 14 /models/weights.gguf\n"
        "9000-a000 r-xp 00000000 00:00 0 [vdso]\n")
    assert {row["path"] for row in rows} == {"/opt/server", "/opt/lib space.so"}
    assert rows[0]["image_size_bytes"] == 8192
    assert rows[0]["base_address"] == 0x1000


def test_linux_deleted_executable_mapping_rejected():
    with pytest.raises(RuntimeError, match="no longer has verifiable file"):
        evidence._parse_linux_maps("1000-2000 r-xp 00000000 08:01 12 /opt/lib.so (deleted)\n")