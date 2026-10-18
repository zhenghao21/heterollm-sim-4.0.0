"""The long diagnostic is limited, persistent, and never reports partial success."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import validate_fast_long_context as validation


def record(case):
    model, arch, mode = case
    return {"model_id": model, "architecture": arch, "requested_mode": mode,
        "prompt_tokens": 65536, "output_tokens": 5120, "visible_output_tokens": 5120,
        "request_status": "forecast_completed" if mode == "fast" else "finished",
        "ttft_ns": 1000., "tpot_ns": 100. if arch == "HBM" else 200.,
        "latency_boundary": "arrival", "wall_seconds": 10. if mode == "fast" else 100.,
        "decode_wall_seconds": 8. if mode == "fast" else 80.,
        "instrumented": False, "affinity": None, "gc_thresholds": [7000, 100, 100],
        "approximation": {"effective_mode": mode}}


def prepare(directory):
    (directory / "progress").mkdir()


def test_queue_has_only_authorized_complete_requests():
    assert len(validation.CASES) == len(set(validation.CASES)) == 8
    assert set(case[0] for case in validation.CASES) == {"qwen2_5-0_5b", "qwen2_5-7b"}
    assert validation.PROMPT == 65536 and validation.OUTPUT == 5120
    row = record(validation.CASES[0])
    assert validation.completed(row, validation.CASES[0])
    for key, value in (("output_tokens", 9), ("visible_output_tokens", 9),
                       ("request_status", "running"), ("prompt_tokens", 32)):
        assert not validation.completed({**row, key: value}, validation.CASES[0])


def test_partial_report_has_no_fabricated_speedup_or_pass(tmp_path):
    prepare(tmp_path)
    case = validation.CASES[0]
    validation.write_json(tmp_path / (validation.case_name(case) + ".json"), record(case))
    state = validation.refresh(tmp_path, {"status": "running"})
    assert state["completed_runs"] == 1
    assert validation.read_json(tmp_path / "comparison.json")["all_within_target"] is None
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "已完成 1/8" in report and "待完成" in report
    assert not list(tmp_path.glob("*.tmp"))


def test_complete_report_compares_both_models_and_independent_ratios(tmp_path):
    prepare(tmp_path)
    for case in validation.CASES:
        validation.write_json(tmp_path / (validation.case_name(case) + ".json"), record(case))
    validation.refresh(tmp_path, {"status": "completed"})
    result = validation.read_json(tmp_path / "comparison.json")
    assert result["all_within_target"] is True
    assert len(result["comparisons"]) == 2
    for comparison in result["comparisons"].values():
        assert comparison["ratio_comparison"]["tpot_ns"]["relative_error"] == 0.
        assert comparison["cases"]["HBM"]["speedup"]["total_wall"] == 10.


def test_lock_rejects_duplicate_manager(tmp_path):
    with validation.manager_lock(tmp_path):
        with pytest.raises(OSError):
            with validation.manager_lock(tmp_path):
                pass


def test_existing_stop_marker_is_respected_without_running_any_worker(tmp_path, monkeypatch):
    (tmp_path / "stop.request").write_text("user paused", encoding="utf-8")
    monkeypatch.setattr(validation.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("started worker"))
    assert validation.run(tmp_path) == 2
    assert validation.read_json(tmp_path / "status.json")["status"] == "paused"
    assert (tmp_path / "stop.request").read_text(encoding="utf-8") == "user paused"
