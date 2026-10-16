"""Dedicated process tuning must restore the caller's collector settings."""
import gc
import importlib.util
from pathlib import Path

import pytest


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
