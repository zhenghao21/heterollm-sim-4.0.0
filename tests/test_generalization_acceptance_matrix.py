import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generalization_acceptance_matrix", ROOT / "tools" / "generalization_acceptance_matrix.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_matrix_has_requested_dimensions_and_repeats():
    cells = list(MODULE.planned_cells(list(MODULE.MODELS), 3))
    assert len(MODULE.MODELS) == 5
    assert len(MODULE.PROMPTS) == 3
    assert len(MODULE.OUTPUTS) == 3
    assert MODULE.PARALLEL == (1, 2, 4)
    assert len(cells) == 405


def test_aggregate_record_uses_concurrent_batch_p50_and_absolute_delta():
    payload = {
        "native": {
            "aggregate": {
                "engine_ttft_ms": {"p50_ms": 10.0},
                "engine_tpot_ms": {"p50_ms": 2.0},
                "engine_e2e_ms": {"p50_ms": 20.0},
            }
        },
        "simulator": {
            "aggregate": {
                "engine_ttft_ms": {"p50_ms": 11.0},
                "engine_tpot_ms": {"p50_ms": 2.2},
                "engine_e2e_ms": {"p50_ms": 19.0},
            }
        },
    }
    records = MODULE.metric_records(payload)
    assert records["ttft_ms"]["signed_error_pct"] == 10.0
    assert abs(records["tpot_ms"]["absolute_delta_ms"] - 0.2) < 1e-9
    assert records["e2e_ms"]["signed_error_pct"] == -5.0
