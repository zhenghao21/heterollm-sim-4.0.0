"""Live report reads tolerate temporary Windows access conflicts."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan_report.py"
    spec = importlib.util.spec_from_file_location("scan_report_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_read_retries_temporary_permission_conflict(report_module, tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"completed": 12}', encoding="utf-8-sig")
    original_read = Path.read_text
    attempts, delays = [], []

    def read(path, **kwargs):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("publication temporarily blocks the reader")
        return original_read(path, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(report_module.time, "sleep", delays.append)
    assert report_module._read_optional_json(target) == {"completed": 12}
    assert len(attempts) == 3 and delays == [0.1, 0.1]


def test_persistent_access_failure_is_reported(report_module, tmp_path, monkeypatch):
    attempts, delays = [], []

    def denied(path, **kwargs):
        attempts.append(path)
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(report_module.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="access denied"):
        report_module._read_optional_json(tmp_path / "progress.json")
    assert len(attempts) == 10 and delays == [0.1] * 9


def test_missing_progress_is_optional_but_invalid_json_is_not(report_module, tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(report_module.time, "sleep", delays.append)
    target = tmp_path / "progress.json"
    assert report_module._read_optional_json(target) is None
    target.write_text("invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        report_module._read_optional_json(target)
    assert delays == []
