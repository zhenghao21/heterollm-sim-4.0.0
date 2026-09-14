import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation_contract", ROOT / "tools" / "evaluation_contract.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_engine_evaluation_ignores_client_values():
    native = {"aggregate": {"engine_ttft_ms": {"p50_ms": 10.0}, "engine_tpot_ms": {"p50_ms": 2.0}, "engine_e2e_ms": {"p50_ms": 20.0}, "ttft_ms": {"p50_ms": 100.0}}}
    sim = {"aggregate": {"engine_ttft_ms": {"p50_ms": 10.0}, "engine_tpot_ms": {"p50_ms": 2.0}, "engine_e2e_ms": {"p50_ms": 20.0}, "ttft_ms": {"p50_ms": 1.0}}}
    result = MODULE.evaluate_metrics(native, sim)
    assert result["status"] == "measured"
    assert result["metrics"]["ttft_ms"]["absolute_error_pct"] == 0.0


def test_engine_evaluation_does_not_fallback_to_client_boundary():
    result = MODULE.evaluate_metrics(
        {"aggregate": {"client_ttft_ms": {"p50_ms": 10.0}}},
        {"aggregate": {"client_ttft_ms": {"p50_ms": 11.0}}},
    )
    assert result["status"] == "evidence_insufficient"
    assert result["metrics"]["ttft_ms"]["native_ms"] is None


def test_engine_counter_rejects_invalid_values():
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.native_llama_compare import _engine_counter_timing
    assert _engine_counter_timing({"prompt_ms": -1.0, "predicted_ms": 2.0}, 3)["engine_timing_status"] == "unavailable"
    assert _engine_counter_timing({"prompt_ms": float("nan"), "predicted_ms": 2.0}, 3)["engine_timing_status"] == "unavailable"
