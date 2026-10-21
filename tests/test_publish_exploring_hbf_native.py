from pathlib import Path
import json
from types import SimpleNamespace as NS
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_exploring_hbf_native as publish


def snapshot():
    return {"schema": publish.SCHEMA, "scan_id": publish.SCAN_ID,
        "observed_at": "2026-10-20T08:00:00+08:00", "completed_results": []}


def test_node_subprocess_contains_only_nonsecret_cli_path_and_snapshot_uses_stdin(monkeypatch):
    def run(command, **options):
        assert command[1].endswith("netlify_native_upload.mjs")
        assert command[-2] == "--cli-root"
        assert "token" not in " ".join(command).lower()
        assert "env" not in options
        assert options["timeout"] == 30
        assert json.loads(options["input"]) == snapshot()
        return NS(returncode=0, stdout=json.dumps({"ok": True, "verified": True,
            "observed_at": snapshot()["observed_at"], "key": publish.UPLOAD_KEY, "http_status": 200}))
    monkeypatch.setattr(publish.subprocess, "run", run)
    result = publish.upload_snapshot(snapshot())
    assert result == {"verified": True, "observed_at": snapshot()["observed_at"], "http_status": 200}


def test_wrong_snapshot_never_starts_node(monkeypatch):
    monkeypatch.setattr(publish.subprocess, "run", lambda *_a, **_k: pytest.fail("unexpected upload"))
    with pytest.raises(publish.UploadError):
        publish.upload_snapshot({**snapshot(), "scan_id": "old-scan"})


def test_upload_timeout_is_sanitized_and_run_enforces_child_timeout(monkeypatch):
    def run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("node", 30, output="private stdout", stderr="private stderr")
    monkeypatch.setattr(publish.subprocess, "run", run)
    with pytest.raises(publish.UploadError) as error:
        publish.upload_snapshot(snapshot())
    assert error.value.safe_details == {"name": "UploadTimeout", "code": "UPLOAD_TIMEOUT", "status": None}
    assert "private" not in str(error.value)


def test_snapshot_is_refreshed_and_failure_retains_last_success_without_error_body(monkeypatch, tmp_path):
    refreshed = []
    monkeypatch.setattr(publish, "write_snapshot", lambda path: refreshed.append(path) or snapshot())
    def reject(*_a, **_k):
        raise publish.UploadError("FetchError", code="NETWORK_ERROR", status=403)
    monkeypatch.setattr(publish, "upload_snapshot", reject)
    result = publish.publish_once(tmp_path, previous={"last_success_at": "previous", "consecutive_failures": 2})
    assert refreshed == [tmp_path]
    assert result["state"] == "error" and result["consecutive_failures"] == 3
    assert result["last_success_at"] == "previous"
    assert result["http_status"] == 403
    assert result["error"] == {"name": "FetchError", "code": "NETWORK_ERROR", "status": 403}


def test_one_successful_iteration_writes_only_publisher_status(monkeypatch, tmp_path):
    monkeypatch.setattr(publish, "write_snapshot", lambda _path: snapshot())
    monkeypatch.setattr(publish, "upload_snapshot", lambda *_a, **_k: {"http_status": 200})
    status = tmp_path / "native_comparison_publisher.json"
    assert publish.run_publisher(tmp_path / "scan", once=True, status_path=status,
        lock_path=tmp_path / "publisher.lock") == 0
    result = json.loads(status.read_text(encoding="utf-8"))
    assert result["state"] == "healthy" and result["verified"] is True
    assert result["pid"] == publish.os.getpid()
    assert result["last_success_observed_at"] == snapshot()["observed_at"]
    assert result["last_success_at"]
    assert not (tmp_path / "scan/status.json").exists()


def test_os_lock_refuses_duplicate_and_releases_on_exit(tmp_path):
    lock = tmp_path / "publisher.lock"
    with publish.SinglePublisher(lock):
        with pytest.raises(publish.PublisherConflict):
            with publish.SinglePublisher(lock):
                pass
    with publish.SinglePublisher(lock):
        pass


def test_duplicate_publisher_does_not_overwrite_owner_status(monkeypatch, tmp_path):
    status, lock = tmp_path / "status.json", tmp_path / "publisher.lock"
    status.write_text("owner", encoding="utf-8")
    monkeypatch.setattr(publish, "publish_once", lambda *_a, **_k: pytest.fail("duplicate publish"))
    with publish.SinglePublisher(lock):
        with pytest.raises(publish.PublisherConflict):
            publish.run_publisher(tmp_path, once=True, status_path=status, lock_path=lock)
    assert status.read_text(encoding="utf-8") == "owner"


def test_arbitrary_node_failure_text_is_not_logged(monkeypatch):
    monkeypatch.setattr(publish.subprocess, "run", lambda *_a, **_k: NS(returncode=2,
        stdout=json.dumps({"ok": False, "status": 401, "code": "AUTH_ERROR", "name": "AuthError",
                           "message": "private response", "headers": "private credential"}), stderr="private stack"))
    with pytest.raises(publish.UploadError) as error:
        publish.upload_snapshot(snapshot())
    assert error.value.safe_details == {"name": "AuthError", "code": "AUTH_ERROR", "status": 401}
    assert "private" not in json.dumps(error.value.safe_details)


def test_background_publisher_is_hidden_and_has_no_credential_argument(monkeypatch, tmp_path):
    monkeypatch.setattr(publish, "LOG_PATH", tmp_path / "publisher.log")
    calls = []
    def popen(command, **options):
        calls.append((command, options))
        return NS(pid=987)
    monkeypatch.setattr(publish.subprocess, "Popen", popen)
    args = NS(directory=tmp_path, interval=15., node="node", cli_root=tmp_path, once=False)
    result = publish.start_background(args)
    command, options = calls[0]
    assert result["launcher_pid"] == 987
    assert "--background" not in command
    assert options["stdin"] == subprocess.DEVNULL
    assert "env" not in options
    if publish.os.name == "nt":
        assert options["creationflags"] & subprocess.CREATE_NO_WINDOW
