"""The optimized queue is tested with fake processes, never a simulation."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import validate_fast_optimized as validation


def record(case, *, guard=False, kernel_error=False):
    model, arch, mode = case
    row = {"model_id": model, "architecture": arch, "requested_mode": mode,
        "prompt_tokens": 65536, "output_tokens": 5120,
        "wall_seconds": 10. if mode == "fast" else 100.,
        "decode_wall_seconds": 8. if mode == "fast" else 80.,
        "instrumented": False, "affinity": None, "gc_thresholds": [7000, 100, 100],
        "timing_breakdown": {"prefill_cohorts_wall_seconds": 4. if mode == "fast" else 10.,
            "decode_cohorts_wall_seconds": 5. if mode == "fast" else 89.},
        "simulation_config": {"fast_decode": {"sampling_strategy": "adaptive"}},
        "approximation": {"effective_mode": mode,
            "sampled_decode_steps": 12 if mode == "fast" else 5119,
            "predicted_decode_steps": 5107 if mode == "fast" else 0}}
    if guard or kernel_error:
        row.update(error_type=validation.GUARD_ERROR if guard else "ValueError",
                   reason="physical guard refused forecast" if guard else "physical execution failed")
    else:
        row.update(visible_output_tokens=5120,
            request_status="forecast_completed" if mode == "fast" else "finished",
            ttft_ns=1000., tpot_ns=100. if arch == "HBM" else 200.,
            latency_boundary="arrival")
    return row


def ready_baseline(tmp_path, status="completed"):
    baseline = tmp_path / "original"
    baseline.mkdir()
    (baseline / "progress").mkdir()
    for case in validation.BASELINE_CASES:
        validation.write_json(baseline / (validation.baseline_case_name(case) + ".json"), record(case))
    validation.write_json(baseline / "status.json", {"status": status})
    return baseline


def fake_processes(monkeypatch, *, guard_case=None, kernel_case=None):
    commands = []

    class Worker:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 990001
            model = command[command.index("--model") + 1]
            arch = command[command.index("--architecture") + 1]
            case = model, arch, "fast"
            row = record(case, guard=case == guard_case, kernel_error=case == kernel_case)
            validation.write_json(Path(command[command.index("--result") + 1]), row)
            self.returncode = int(case in {guard_case, kernel_case})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(validation.subprocess, "Popen", Worker)
    return commands


@pytest.fixture(autouse=True)
def no_real_pid_checks(monkeypatch):
    monkeypatch.setattr(validation, "process_alive", lambda pid: False)


def test_only_four_fast_candidates_reuse_original_results(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    original = {path.name: path.read_bytes() for path in baseline.glob("*.json")}
    commands = fake_processes(monkeypatch)
    output = tmp_path / "candidate"
    assert validation.run(output, baseline) == 0
    assert len(commands) == 4
    for command in commands:
        assert command[command.index("--mode") + 1] == "fast"
        assert command[command.index("--sampling-strategy") + 1] == "adaptive"
        assert "--require-forecast" in command
        assert command[command.index("--prompt") + 1] == "65536"
        assert command[command.index("--output") + 1] == "5120"
    state = validation.read_json(output / "status.json")
    assert state["completed_candidates"] == state["attempted_candidates"] == 4
    assert state["status"] == "completed"
    result = validation.read_json(output / "comparison.json")
    assert result["all_within_target"] is True
    assert "historical" in result["wall_comparison_scope"]
    assert "not an interleaved ABBA" in result["wall_comparison_scope"]
    assert {path.name: path.read_bytes() for path in baseline.glob("*.json")} == original
    assert "后台负载未受控，非 ABBA" in (output / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("baseline_status", ["paused", "failed", "stopping"])
def test_stopped_original_queue_never_starts_a_candidate(tmp_path, monkeypatch, baseline_status):
    baseline = ready_baseline(tmp_path, baseline_status)
    monkeypatch.setattr(validation.subprocess, "Popen", lambda *a, **k: pytest.fail("started a worker"))
    output = tmp_path / "candidate"
    assert validation.run(output, baseline) == 2
    state = validation.read_json(output / "status.json")
    assert state["status"] == "baseline_stopped"
    assert baseline_status in state["reason"]


def test_waits_for_original_completion_without_parallel_work(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path, "running")
    output = tmp_path / "candidate"
    commands = fake_processes(monkeypatch)
    waits = []

    def sleep(seconds):
        assert seconds == 10 and not commands
        assert validation.read_json(output / "status.json")["status"] == "waiting_baseline"
        waits.append(seconds)
        validation.write_json(baseline / "status.json", {"status": "completed"})

    monkeypatch.setattr(validation.time, "sleep", sleep)
    assert validation.run(output, baseline) == 0
    assert waits == [10] and len(commands) == 4


def test_completed_original_still_requires_all_eight_complete_requests(tmp_path):
    baseline = ready_baseline(tmp_path)
    case = validation.BASELINE_CASES[0]
    path = baseline / (validation.baseline_case_name(case) + ".json")
    row = validation.read_json(path)
    validation.write_json(path, {**row, "visible_output_tokens": 9})
    gate, reason = validation.baseline_gate(baseline)
    assert gate == "blocked" and "7/8" in reason


def test_original_real_worker_must_exit_even_when_status_is_completed(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    validation.write_json(baseline / "status.json", {"status": "completed",
        "worker_pid": 100, "worker_progress": {"process_id": 200}})
    monkeypatch.setattr(validation, "process_alive", lambda pid: pid == 200)
    gate, reason = validation.baseline_gate(baseline)
    assert gate == "waiting" and "200" in reason


def test_restarted_manager_refuses_live_real_child_after_launcher_exits(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "progress").mkdir()
    name = validation.case_name(validation.CANDIDATES[0])
    validation.write_json(output / "status.json", {"status": "running",
        "current_case": name, "worker_pid": 100})
    validation.write_json(output / "progress" / (name + ".json"), {
        "status": "running", "process_id": 200})
    monkeypatch.setattr(validation, "process_alive", lambda pid: pid == 200)
    monkeypatch.setattr(validation.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate worker"))
    with pytest.raises(RuntimeError, match="200"):
        validation.run(output, baseline)


def test_forecast_guard_is_failed_point_without_running_detailed_fallback(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    commands = fake_processes(monkeypatch, guard_case=validation.CANDIDATES[0])
    output = tmp_path / "candidate"
    assert validation.run(output, baseline) == 1
    assert len(commands) == 4
    state = validation.read_json(output / "status.json")
    assert state["status"] == "completed_with_failures"
    assert state["attempted_candidates"] == 4 and state["completed_candidates"] == 3
    assert validation.read_json(output / "comparison.json")["all_within_target"] is False
    comparison = validation.read_json(output / (validation.MODELS[0] + "_comparison.json"))
    assert comparison["cases"]["HBM"]["speedup"]["total_wall"] is None
    assert not list(output.glob("*detailed*.json"))
    commands.clear()
    assert validation.run(output, baseline) == 1
    assert not commands, "guard failures must not be retried or converted to detailed work"


def test_other_kernel_failure_stops_before_remaining_candidates(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    commands = fake_processes(monkeypatch, kernel_case=validation.CANDIDATES[0])
    output = tmp_path / "candidate"
    assert validation.run(output, baseline) == 1
    assert len(commands) == 1
    state = validation.read_json(output / "status.json")
    assert state["status"] == "failed" and state["reason"] == "physical execution failed"
    assert validation.read_json(output / "comparison.json")["all_within_target"] is None


def test_completed_forecast_above_ten_percent_is_reported_as_failure(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    original_record = record

    def inaccurate_record(case, **kwargs):
        row = original_record(case, **kwargs)
        if case == validation.CANDIDATES[0]:
            row["tpot_ns"] *= 1.2
        return row

    monkeypatch.setattr(sys.modules[__name__], "record", inaccurate_record)
    commands = fake_processes(monkeypatch)
    output = tmp_path / "candidate"
    assert validation.run(output, baseline) == 1
    assert len(commands) == 4
    state = validation.read_json(output / "status.json")
    assert state["status"] == "completed_with_failures" and state["completed_candidates"] == 4
    comparison = validation.read_json(output / (validation.MODELS[0] + "_comparison.json"))
    assert comparison["cases"]["HBM"]["relative_errors"]["tpot_ns"] == pytest.approx(.2)
    assert not comparison["ratio_comparison"]["tpot_ns"]["within_target"]


def test_previous_fast_comparison_reports_real_stage_wall_and_sample_counts(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    for case in validation.CANDIDATES:
        path = baseline / (validation.baseline_case_name(case) + ".json")
        row = validation.read_json(path)
        row.update(wall_seconds=30., timing_breakdown={
            "prefill_cohorts_wall_seconds": 20., "decode_cohorts_wall_seconds": 9.})
        row["approximation"].update(sampled_decode_steps=56, predicted_decode_steps=5063)
        validation.write_json(path, row)
    fake_processes(monkeypatch)
    output = tmp_path / "candidate"
    assert validation.run(output, baseline) == 0
    comparison = validation.read_json(output / (validation.MODELS[0] + "_comparison.json"))
    timing = comparison["cases"]["HBM"]["optimization_comparison"]
    assert timing["previous_fast"]["sampled_decode_steps"] == 56
    assert timing["adaptive_fast"]["sampled_decode_steps"] == 12
    assert timing["previous_over_adaptive"] == {
        "total_wall_seconds": 3., "prefill_wall_seconds": 5., "real_decode_wall_seconds": 1.8}
    assert "excludes forecast" in timing["real_decode_scope"]
    assert "not an interleaved ABBA" in timing["wall_comparison_scope"]
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "56/12" in report and "5.000×" in report
    assert "非 ABBA" in report and "实算 Decode 耗时只含真实 cohort 区间" in report


def test_stop_marker_is_preserved_and_never_launches_worker(tmp_path, monkeypatch):
    baseline = ready_baseline(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "stop.request").write_text("pause", encoding="utf-8")
    monkeypatch.setattr(validation.subprocess, "Popen", lambda *a, **k: pytest.fail("started worker"))
    assert validation.run(output, baseline) == 2
    assert validation.read_json(output / "status.json")["status"] == "paused"
    assert (output / "stop.request").read_text(encoding="utf-8") == "pause"


@pytest.mark.parametrize("pause_via_marker", [True, False])
def test_original_pause_during_candidate_cancels_worker_and_retains_markers(
        tmp_path, monkeypatch, pause_via_marker):
    baseline = ready_baseline(tmp_path)
    output = tmp_path / "candidate"
    commands = []

    class Worker:
        pid, returncode, polls = 990001, None, 0

        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                if pause_via_marker:
                    (baseline / "stop.request").write_text("user pause", encoding="utf-8")
                else:
                    validation.write_json(baseline / "status.json", {"status": "paused",
                        "reason": "user pause"})
                return None
            assert (output / "stop.request").exists(), "pause must reach the worker's cancellation marker"
            self.returncode = 2
            return self.returncode

    def sleep(seconds):
        assert seconds == 10
        assert validation.read_json(output / "status.json")["status"] == "stopping"
        assert (output / "stop.request").exists()

    monkeypatch.setattr(validation.subprocess, "Popen", Worker)
    monkeypatch.setattr(validation.time, "sleep", sleep)
    assert validation.run(output, baseline) == 2
    assert len(commands) == 1
    state = validation.read_json(output / "status.json")
    assert state["status"] == "paused" and "Original validation pause" in state["reason"]
    assert (output / "stop.request").exists()
    if pause_via_marker:
        assert (baseline / "stop.request").read_text(encoding="utf-8") == "user pause"


def test_candidate_output_cannot_modify_original_validation(tmp_path):
    baseline = ready_baseline(tmp_path)
    with pytest.raises(ValueError, match="must not write inside"):
        validation.run(baseline / "new", baseline)
    assert not (baseline / "new").exists()


def test_fixed_or_detailed_result_cannot_satisfy_adaptive_forecast(tmp_path):
    case = validation.CANDIDATES[0]
    row = record(case)
    row["simulation_config"]["fast_decode"]["sampling_strategy"] = "fixed"
    assert validation.candidate_status(row, case) is None
    row = record(case)
    row["approximation"]["effective_mode"] = "detailed"
    assert validation.candidate_status(row, case) is None
