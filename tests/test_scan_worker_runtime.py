"""Dedicated process tuning must restore the caller's collector settings."""
import gc
import importlib.util
import json
from pathlib import Path

import pytest


def load_scan_worker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan.py"
    spec = importlib.util.spec_from_file_location("scan_worker_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_publication_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    module = load_scan_worker()
    target = tmp_path / "progress.json"
    target.write_text('{"completed": 1}', encoding="utf-8")
    original_replace = Path.replace
    attempts, delays = [], []

    def replace(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 1}
        if len(attempts) < 3:
            raise PermissionError("reader temporarily holds the destination")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    module.write_json(target, {"completed": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 2}
    assert len(attempts) == 3 and delays == [0.1, 0.1]


def test_progress_publication_still_reports_persistent_permission_failure(tmp_path, monkeypatch):
    module = load_scan_worker()
    target = tmp_path / "progress.json"
    target.write_text('{"completed": 1}', encoding="utf-8")
    attempts, delays = [], []

    def denied(source, destination):
        attempts.append(source)
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="access denied"):
        module.write_json(target, {"completed": 2})
    assert len(attempts) == 10 and delays == [0.1] * 9
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 1}


@pytest.mark.parametrize("enabled", (True, False))
def test_worker_gc_context_preserves_enablement_and_restores_after_failure(enabled):
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan.py"
    spec = importlib.util.spec_from_file_location("scan_worker_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous, was_enabled = gc.get_threshold(), gc.isenabled()
    try:
        gc.set_threshold(12000, 7, 130)
        (gc.enable if enabled else gc.disable)()
        with pytest.raises(RuntimeError, match="cancelled simulation"):
            with module.scenario_gc_thresholds() as configured:
                assert configured == gc.get_threshold() == (12000, 100, 130)
                assert gc.isenabled() is enabled
                raise RuntimeError("cancelled simulation")
        assert gc.get_threshold() == (12000, 7, 130)
        assert gc.isenabled() is enabled
    finally:
        gc.set_threshold(*previous)
        (gc.enable if was_enabled else gc.disable)()
