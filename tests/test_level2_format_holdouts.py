import json
from pathlib import Path

from heterollm_sim.kernel_level2_surfaces import LEVEL2_VALIDATION_ERROR_BY_FORMAT, level2_samples


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/development/level2_format_holdouts_20260930"


def test_format_protocol_is_independent_and_fail_closed():
    protocol = json.loads((ARTIFACT / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["formats"] == ["Q5_K", "Q8_0"]
    assert protocol["harness"] == "python_exact_q8"
    assert protocol["no_retry_on_failure"] is True
    assert protocol["runtime_binary_sha256"] == (
        "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
    )
    assert all(len(shapes) == 4 for shapes in protocol["training_shapes_by_format"].values())
    assert all(shapes == [[1, 3072, 3072]]
               for shapes in protocol["holdout_shapes_by_format"].values())
    for fmt in ("Q5_K", "Q8_0"):
        individual = json.loads((ARTIFACT / f"protocol_{fmt}.json").read_text(encoding="utf-8"))
        assert individual["format"] == fmt
        assert individual["training_shapes"] == protocol["training_shapes_by_format"][fmt]
        assert individual["holdout_shapes"] == [[1, 3072, 3072]]


def test_measurements_have_correctness_runtime_and_device_boundary():
    rows = json.loads((ARTIFACT / "measurements.json").read_text(encoding="utf-8"))
    assert len(rows) == 10
    assert {row["format"] for row in rows} == {"Q5_K", "Q8_0"}
    assert all(row["measurement_boundary"] == "cuda_device_kernel_interval" for row in rows)
    assert all(row["target_llm_timing_used"] is False for row in rows)
    assert all(row["runtime_binary_sha256"] == (
        "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
    ) for row in rows)
    assert all(row["correctness"]["passed"] for row in rows)


def test_production_acceptance_requires_all_gates_and_surface_is_installed():
    report = json.loads((ARTIFACT / "holdout_evaluation.json").read_text(encoding="utf-8"))
    assert report["native_llm_timing_used"] is False
    assert report["all_formats_accepted"] is True
    for item in report["formats"]:
        assert item["complete_four_corner_grid"] is True
        assert item["training_rows"] == item["required_training_rows"] == 4
        assert item["source_signature_same_domain"] is True
        assert item["resource_signature_same_domain"] is True
        assert item["accepted"] is True
        holdout = item["holdout_rows"]
        assert len(holdout) == 1 and holdout[0]["eligible"] is True
        assert holdout[0]["cv_pct"] < 10 and holdout[0]["ape_pct"] < 10
    assert len(level2_samples("q5_k")) == 4
    assert len(level2_samples("q8_0")) == 4
    assert LEVEL2_VALIDATION_ERROR_BY_FORMAT["q5_k"] < 0.1
    assert LEVEL2_VALIDATION_ERROR_BY_FORMAT["q8_0"] < 0.1
