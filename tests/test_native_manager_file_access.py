"""Progress-file sharing conflicts must not terminate a running scan."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exploring_hbf_native as run


def test_progress_read_recovers_after_temporary_access_denial(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    run.write_json(path, {"state": "running", "watchdog_phase": "decode"})
    original = Path.read_text
    attempts, waits = [], []

    def shared_read(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) <= 2:
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", shared_read)
    monkeypatch.setattr(run.time, "sleep", waits.append)
    assert run.read_json(path, {}) == {"state": "running", "watchdog_phase": "decode"}
    assert len(waits) == 2


def test_persistent_access_denial_is_bounded_and_propagates(tmp_path, monkeypatch):
    waits = []

    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(run.time, "sleep", waits.append)
    with pytest.raises(PermissionError):
        run.read_json(tmp_path / "progress.json", {})
    assert sum(waits) < 1.
    assert waits


def test_missing_progress_keeps_the_callers_default(tmp_path):
    default = {"state": "pending"}
    assert run.read_json(tmp_path / "missing.json", default) is default


def test_corrupt_progress_is_not_treated_as_missing(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text("{", encoding="utf-8")
    waits = []
    monkeypatch.setattr(run.time, "sleep", waits.append)
    with pytest.raises(json.JSONDecodeError):
        run.read_json(path, {})
    assert not waits
