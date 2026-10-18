"""Fast accuracy checks never weaken the detailed result comparator."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("fast_mode_benchmark_test",
        Path(__file__).parents[1] / "scripts" / "benchmark_fast_mode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(architecture="HBM", mode="detailed", **changes):
    result = {"architecture": architecture, "model_id": "model", "prompt_tokens": 2048,
        "output_tokens": 257, "requested_mode": mode, "request_status": "finished",
        "visible_output_tokens": 257, "ttft_ns": 100., "tpot_ns": 10.,
        "wall_seconds": 100., "decode_wall_seconds": 90., "instrumented": False,
        "affinity": {"logical_cpu": 2}, "gc_thresholds": [7000, 100, 100],
        "approximation": {"effective_mode": mode, "sampled_decode_steps": 56,
            "predicted_decode_steps": 200 if mode == "fast" else 0}}
    result.update(changes)
    return result


def test_forecast_metrics_and_speedup_are_separate_from_exactness(benchmark):
    result = benchmark.compare_case(row(), row(mode="fast", request_status="forecast_completed",
        ttft_ns=100., tpot_ns=10.5, wall_seconds=10., decode_wall_seconds=5.))
    assert result["within_target"]
    assert result["relative_errors"] == {"ttft_ns": 0., "tpot_ns": .05}
    assert result["speedup"] == {"total_wall": 10., "decode_wall": 18.}
    assert result["predicted_decode_steps"] == 200
    assert not result["transaction_equivalence_claimed"]


@pytest.mark.parametrize("changes", [
    {"output_tokens": 9}, {"visible_output_tokens": 9}, {"prompt_tokens": 32},
    {"model_id": "other"}, {"request_status": "running"}, {"requested_mode": "detailed"},
    {"instrumented": True}, {"affinity": {"logical_cpu": 3}},
    {"gc_thresholds": [1, 2, 3]}, {"tpot_ns": math.nan}, {"ttft_ns": 0},
])
def test_invalid_or_shortened_references_cannot_pass_or_claim_speedup(benchmark, changes):
    result = benchmark.compare_case(row(), row(mode="fast", **changes))
    assert not result["within_target"]
    assert not result["valid_comparison"]
    assert result["invalid_reasons"]
    assert result["speedup"] == {"total_wall": None, "decode_wall": None}


def test_over_target_error_is_real_failure_but_timing_remains_visible(benchmark):
    result = benchmark.compare_case(row(), row(mode="fast", tpot_ns=12.))
    assert result["valid_comparison"]
    assert not result["within_target"]
    assert result["relative_errors"]["tpot_ns"] == .2


def test_ratio_error_can_fail_even_if_both_storage_errors_pass(benchmark):
    rows = [row(), row(mode="fast", tpot_ns=9.1),
            row("HBF"), row("HBF", "fast", tpot_ns=10.9)]
    result = benchmark.compare_pair(rows)
    assert all(case["within_target"] for case in result["cases"].values())
    assert not result["ratio_comparison"]["tpot_ns"]["within_target"]
    assert not result["all_within_target"]


def test_detailed_fallback_is_reported_without_claiming_sampling_gain(benchmark):
    candidate = row(mode="fast")
    candidate["approximation"] = {"effective_mode": "detailed", "sampled_decode_steps": 256,
        "predicted_decode_steps": 0, "fallback_reasons": ["maintenance_guard"]}
    result = benchmark.compare_case(row(), candidate)
    assert result["within_target"]
    assert result["effective_mode"] == "detailed"
    assert result["predicted_decode_steps"] == 0
    assert result["fallback_reasons"] == ["maintenance_guard"]


def test_pair_rejects_missing_or_duplicate_case(benchmark):
    rows = [row(), row(mode="fast"), row("HBF"), row("HBF", "fast")]
    assert benchmark.compare_pair(rows)["all_within_target"]
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.compare_pair(rows[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.compare_pair(rows[:-1] + [deepcopy(rows[0])])


def test_hardware_ratio_requires_matching_full_request_between_storage(benchmark):
    rows = [row(), row(mode="fast"), row("HBF", prompt_tokens=4096),
            row("HBF", "fast", prompt_tokens=4096)]
    result = benchmark.compare_pair(rows)
    assert all(case["within_target"] for case in result["cases"].values())
    assert not result["all_within_target"]


class NeverVisit:
    def __iter__(self):
        raise AssertionError("batch/event report must not be traversed")


def detailed_endpoint(*, engine_start=None, events=None, status="finished"):
    metric = SimpleNamespace(request_id="request", status=SimpleNamespace(value=status),
        arrival_ns=0., first_token_ns=10., finish_ns=50., ttft_ns=10., tpot_ns=20.,
        visible_output_tokens=3, engine_start_ns=engine_start)
    return SimpleNamespace(serving=SimpleNamespace(request_metrics={"request": metric},
        events=NeverVisit() if events is None else events, batches=NeverVisit()))


def test_endpoint_arrival_reads_no_batches_or_events(benchmark):
    endpoint = benchmark.extract_request_endpoint(detailed_endpoint())
    assert endpoint["request"] == {"request_id": "request", "status": "finished",
        "arrival_ns": 0., "first_token_ns": 10., "finish_ns": 50.,
        "ttft_ns": 10., "tpot_ns": 20., "e2e_ns": 50.,
        "visible_output_tokens": 3, "latency_boundary": "arrival"}
    assert endpoint["approximation"] == {"effective_mode": "detailed"}
    assert "summary" not in endpoint["report"]


def test_endpoint_engine_uses_last_visible_commit_not_request_cleanup(benchmark):
    events = [SimpleNamespace(event_type="tokens_committed", request_id="request",
        timestamp_ns=timestamp, details={"visible_tokens": visible})
        for timestamp, visible in ((10., 1), (25., 1), (40., 1), (45., 0))]
    endpoint = benchmark.extract_request_endpoint(detailed_endpoint(engine_start=5., events=events))
    request = endpoint["request"]
    assert request["latency_boundary"] == "engine"
    assert (request["ttft_ns"], request["tpot_ns"], request["e2e_ns"]) == (5., 15., 35.)
    assert request["finish_ns"] == 50.


def test_unfinished_endpoint_uses_arrival_boundary_even_with_engine_marker(benchmark):
    endpoint = benchmark.extract_request_endpoint(detailed_endpoint(engine_start=5., status="running"))
    assert endpoint["request"]["latency_boundary"] == "arrival"


def test_fast_forecast_endpoint_does_not_materialize_a_detailed_report(benchmark):
    forecast = SimpleNamespace(request_id="request", arrival_ns=3., first_token_ns=13.,
        finish_ns=53., ttft_ns=10., tpot_ns=20., output_tokens=3)
    result = SimpleNamespace(forecast=forecast, detailed=None,
        approximation={"effective_mode": "fast", "predicted_decode_steps": 2})
    endpoint = benchmark.extract_request_endpoint(result)
    assert endpoint["request"]["status"] == "forecast_completed"
    assert endpoint["request"]["e2e_ns"] == 50.
    assert endpoint["request"]["latency_boundary"] == "arrival"
    assert endpoint["report"]["trace_fidelity"] == "approximate"
    assert endpoint["approximation"] == result.approximation


def test_fast_detailed_wrapper_preserves_reasons_and_metrics(benchmark):
    result = SimpleNamespace(forecast=None, detailed=detailed_endpoint(),
        approximation={"effective_mode": "detailed", "fallback_reasons": ["guard"]})
    endpoint = benchmark.extract_request_endpoint(result)
    assert endpoint["request"]["tpot_ns"] == 20.
    assert endpoint["report"]["simulation_mode"] == "fast"
    assert endpoint["report"]["trace_fidelity"] == "aggregate"
    assert endpoint["approximation"]["fallback_reasons"] == ["guard"]


def test_endpoint_rejects_non_single_online_request(benchmark):
    result = detailed_endpoint()
    result.serving.request_metrics = {}
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.extract_request_endpoint(result)
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.extract_request_endpoint(SimpleNamespace())


def test_mismatched_public_latency_boundary_invalidates_comparison(benchmark):
    result = benchmark.compare_case(row(latency_boundary="engine"),
        row(mode="fast", latency_boundary="arrival"))
    assert not result["valid_comparison"]
    assert "different_public_latency_boundary" in result["invalid_reasons"]


def progress_args(tmp_path):
    return SimpleNamespace(model="model", architecture="HBF", mode="fast",
        prompt=65536, output=5120, progress_file=tmp_path / "progress.json",
        result=tmp_path / "result.json")


def progress_event(stage="serving_cohorts", completed=1, kind="decode", **metadata):
    return SimpleNamespace(stage=stage, completed=completed,
        metadata={"cohort_kind": kind, **metadata})


def test_progress_is_atomic_throttled_and_retains_exact_recent_windows(benchmark, tmp_path):
    import json
    now = [0.]
    args = progress_args(tmp_path)
    monitor = benchmark.BenchmarkProgress(args, clock=lambda: now[0])
    monitor.begin_simulation()
    now[0] = .5
    monitor.observe(progress_event(completed=0))
    now[0] = 2.5
    monitor.observe(progress_event(kind="prefill"))
    first = args.progress_file.read_text(encoding="utf-8")
    now[0] = 3.
    monitor.observe(progress_event(stage="cohort_tasks"))
    assert args.progress_file.read_text(encoding="utf-8") == first
    for index in range(1, 33):
        now[0] = 2.5 + 2 * index
        monitor.observe(progress_event(completed=index + 1))
    monitor.publish(force=True)
    data = json.loads(args.progress_file.read_text(encoding="utf-8"))
    assert data["prefill_steps"] == 1
    assert data["decode_steps"] == 32
    assert data["full_output_tokens"] == 5120
    assert data["recent_16_decode_step_mean_seconds"] == 2.
    assert data["windows"] == [
        {"first_decode_step": 1, "last_decode_step": 16,
         "mean_step_wall_seconds": 2., "end_wall_seconds": 34.5},
        {"first_decode_step": 17, "last_decode_step": 32,
         "mean_step_wall_seconds": 2., "end_wall_seconds": 66.5}]
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_fallback_and_terminal_progress_are_published_without_waiting(benchmark, tmp_path):
    import json
    now = [0.]
    args = progress_args(tmp_path)
    monitor = benchmark.BenchmarkProgress(args, clock=lambda: now[0])
    monitor.begin_simulation()
    now[0] = .01
    monitor.observe(progress_event(stage="fast_mode_fallback", fallback_reasons=["gc_guard"]))
    assert json.loads(args.progress_file.read_text(encoding="utf-8"))["fallback_reasons"] == ["gc_guard"]
    monitor.finish("completed", predicted_decode_steps=5000)
    data = json.loads(args.progress_file.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["predicted_decode_steps"] == 5000


def test_required_forecast_rejects_before_fallback_and_preserves_reason(benchmark, tmp_path):
    import json
    args = progress_args(tmp_path)
    args.require_forecast = True
    monitor = benchmark.BenchmarkProgress(args, clock=lambda: 0.)
    monitor.begin_simulation()
    with pytest.raises(benchmark.FastForecastUnavailable, match="gc_guard"):
        monitor.observe(progress_event(stage="fast_mode_fallback", fallback_reasons=["gc_guard"]))
    assert json.loads(args.progress_file.read_text(encoding="utf-8"))["fallback_reasons"] == ["gc_guard"]


def test_calibration_budget_and_samples_survive_later_progress_and_rejection(benchmark, tmp_path):
    import json
    args = progress_args(tmp_path)
    args.require_forecast = True
    now = [0.]
    monitor = benchmark.BenchmarkProgress(args, clock=lambda: now[0])
    monitor.begin_simulation()
    attempt = {"sampled_decode_steps": 100, "conditional_relative_error_budget": .06,
        "error_budget": {"holdout": .01, "maintenance": .03, "context": .02}}
    diagnostics = {"effective_mode": "sampling", "sampling_attempts": [attempt],
        "calibration_samples": [{"decode_step": 100, "elapsed_ns": 123.}]}
    now[0] = .01
    monitor.observe(progress_event(stage="fast_calibration", approximation=diagnostics,
        calibration_attempt=attempt))
    # The simulator is allowed to mutate its working diagnostics after the
    # callback. The completed observation must retain its original budget.
    attempt["error_budget"]["maintenance"] = 99.
    diagnostics["calibration_samples"][0]["elapsed_ns"] = 999.
    now[0] = 11.
    monitor.observe(progress_event(stage="cohort_tasks"))
    ordinary = json.loads(args.progress_file.read_text(encoding="utf-8"))
    assert ordinary["calibration_attempts"][0]["error_budget"]["maintenance"] == .03
    assert ordinary["approximation"]["calibration_samples"][0]["elapsed_ns"] == 123.

    terminal = {**ordinary["approximation"], "effective_mode": "detailed",
        "fallback_reasons": ["conditional_error_budget_exceeded"]}
    with pytest.raises(benchmark.FastForecastUnavailable):
        monitor.observe(progress_event(stage="fast_mode_fallback", approximation=terminal,
            fallback_reasons=terminal["fallback_reasons"]))
    durable = json.loads(args.progress_file.read_text(encoding="utf-8"))
    assert durable["approximation"] == terminal
    assert durable["calibration_attempts"] == ordinary["calibration_attempts"]
    assert durable["status"] != "forecast_completed"


def test_failed_candidate_retains_budget_samples_and_actual_timing(benchmark, tmp_path, monkeypatch):
    import json
    args = progress_args(tmp_path)
    args.require_forecast = True
    attempt = {"sampled_decode_steps": 100, "error_budget": {"maintenance": .03}}
    terminal = {"effective_mode": "detailed", "sampling_attempts": [attempt],
        "calibration_samples": [{"decode_step": 100, "elapsed_ns": 123.}]}
    def reject(unused, monitor):
        monitor.begin_simulation()
        monitor.serving_start_wall = 1.
        monitor.last_cohort_wall = 5.
        monitor.last_prefill = 3.
        monitor.phase_wall = {"prefill": 2., "decode": 2.}
        monitor.decode_steps = 100
        monitor.observe(progress_event(stage="fast_calibration", approximation=terminal,
            calibration_attempt=attempt))
        monitor.observe(progress_event(stage="fast_mode_fallback", approximation=terminal,
            fallback_reasons=["conditional_error_budget_exceeded"]))
        raise AssertionError("detailed fallback must not execute")
    monkeypatch.setattr(benchmark, "_benchmark_case", reject)
    with pytest.raises(benchmark.FastForecastUnavailable):
        benchmark.benchmark_case(args)
    progress = json.loads(args.progress_file.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    for data in (progress, result):
        assert data["status"] == data["request_status"] == "failed"
        assert data["approximation"] == terminal
        assert data["calibration_attempts"] == [attempt]
        assert data["decode_steps"] == 100
        assert data["timing_breakdown"]["prefill_cohorts_wall_seconds"] == 2.
        assert data["timing_breakdown"]["decode_cohorts_wall_seconds"] == 2.
        assert data["timing_breakdown"]["control_plane"] == {}


def test_successful_result_and_terminal_progress_keep_all_calibration_attempts(
        benchmark, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from dataclasses import dataclass
    import json
    import sys
    @dataclass
    class Scenario:
        simulation: object = None
    @contextmanager
    def thresholds():
        yield [7000, 100, 100]
    attempt = {"sampled_decode_steps": 52, "error_budget": {"maintenance": .001}}
    approximation = {"effective_mode": "fast", "sampling_attempts": [attempt],
        "calibration_samples": [{"decode_step": 52, "elapsed_ns": 123.}],
        "predicted_decode_steps": 5067}
    def simulate(unused, *, retention_policy, control):
        assert retention_policy == "aggregate"
        control.progress_callback(progress_event(stage="fast_calibration",
            approximation=approximation, calibration_attempt=attempt))
        return SimpleNamespace()
    for name, module in {
        "benchmark_scan_runtime": SimpleNamespace(),
        "b200_hbm_hbf_scan": SimpleNamespace(scenario_gc_thresholds=thresholds),
        "b200_hbm_hbf_scan_config": SimpleNamespace(build_scan_scenario=lambda *unused: Scenario()),
        "heterollm_sim.execution_control": SimpleNamespace(ExecutionControl=lambda **values: SimpleNamespace(**values)),
        "heterollm_sim.reporting": SimpleNamespace(run_scenario=simulate),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(benchmark, "extract_request_endpoint", lambda unused: {
        "request": {"request_id": "request", "status": "forecast_completed",
            "visible_output_tokens": 5120, "ttft_ns": 10., "tpot_ns": 20.,
            "latency_boundary": "arrival", "e2e_ns": 102390.},
        "approximation": approximation, "report": {}})
    args = progress_args(tmp_path)
    args.cpu, args.target = None, .10
    row = benchmark.benchmark_case(args)
    saved = json.loads(args.result.read_text(encoding="utf-8"))
    progress = json.loads(args.progress_file.read_text(encoding="utf-8"))
    for data in (row, saved, progress):
        assert data["approximation"] == approximation
        assert data["calibration_attempts"] == [attempt]
        assert data["request_status"] == "forecast_completed"
    assert progress["status"] == "completed"


@pytest.mark.parametrize("strategy", ["adaptive", "fixed"])
def test_diagnostic_sampling_and_error_target_are_explicit(benchmark, strategy):
    options = benchmark.simulation_options(SimpleNamespace(mode="fast", target=.05,
        sampling_strategy=strategy))
    assert options.mode == "fast"
    assert options.fast_decode.sampling_strategy == strategy
    assert options.fast_decode.target_relative_error == .05


@pytest.mark.parametrize("limit", [None, 256, 512])
def test_diagnostic_sample_cap_is_explicit_without_changing_default(benchmark, limit):
    args = SimpleNamespace(mode="fast", target=.10, max_sample_tokens=limit)
    options = benchmark.simulation_options(args)
    assert options.fast_decode.max_sample_tokens == (128 if limit is None else limit)


def test_cli_passes_real_sample_cap_to_single_run_without_executing_kernel(
        benchmark, tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(benchmark, "benchmark_case", lambda args: captured.append(args))
    assert benchmark.main(["--mode", "fast", "--architecture", "HBF",
        "--result", str(tmp_path / "candidate.json"), "--max-sample-tokens", "256",
        "--require-forecast"]) == 0
    assert len(captured) == 1
    assert captured[0].max_sample_tokens == 256
    assert benchmark.simulation_options(captured[0]).fast_decode.max_sample_tokens == 256


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_cli_rejects_invalid_sample_cap_before_running(benchmark, tmp_path, monkeypatch, limit):
    monkeypatch.setattr(benchmark, "benchmark_case",
        lambda unused: pytest.fail("invalid CLI must not start a run"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--mode", "fast", "--architecture", "HBF",
            "--result", str(tmp_path / "candidate.json"), "--max-sample-tokens", limit])
    assert error.value.code == 2


def test_timing_summary_uses_existing_boundaries_only(benchmark, tmp_path):
    args = progress_args(tmp_path)
    monitor = benchmark.BenchmarkProgress(args, clock=lambda: 0.)
    monitor.serving_start_wall = 1.
    monitor.last_cohort_wall = 12.
    monitor.last_prefill = 4.
    monitor.phase_wall = {"prefill": 3., "decode": 8.}
    result = SimpleNamespace(manifest=SimpleNamespace(metadata={"control_plane": {"makespan_ns": 10.}}))
    timing = monitor.timing_summary(13., .001, result)
    assert timing["after_last_cohort_seconds"] == 1.
    assert timing["setup_to_serving_start_seconds"] + timing["prefill_cohorts_wall_seconds"] + timing["decode_cohorts_wall_seconds"] + timing["after_last_cohort_seconds"] == 13.
    assert timing["control_plane"] == {"makespan_ns": 10.}


def test_stop_marker_poll_is_throttled_and_never_deletes_marker(benchmark, tmp_path):
    now = [0.]
    args = progress_args(tmp_path)
    args.stop_file = tmp_path / "stop.request"
    monitor = benchmark.BenchmarkProgress(args, clock=lambda: now[0])
    assert not monitor.cancellation_requested()
    args.stop_file.write_text("stop", encoding="utf-8")
    now[0] = .49
    assert not monitor.cancellation_requested()
    now[0] = .5
    assert monitor.cancellation_requested()
    assert args.stop_file.exists()
    assert monitor.cancellation_requested()


@pytest.mark.parametrize("error,status", [(RuntimeError("failed simulation"), "failed"),
    (KeyboardInterrupt(), "cancelled")])
def test_benchmark_failure_is_durable_without_importing_or_running_kernel(
        benchmark, tmp_path, monkeypatch, error, status):
    import json
    args = progress_args(tmp_path)
    def fail(*unused):
        raise error
    monkeypatch.setattr(benchmark, "_benchmark_case", fail)
    with pytest.raises(type(error)):
        benchmark.benchmark_case(args)
    progress = json.loads(args.progress_file.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    assert progress["status"] == result["status"] == status
    assert progress["error_type"] == result["error_type"] == type(error).__name__
