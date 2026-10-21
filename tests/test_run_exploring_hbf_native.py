"""The native experiment runner schedules/observes; it never models latency."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exploring_hbf_native as run


def small_parameters():
    case = {**run.case_matrix()[0], "output_tokens": 4}
    return run.parameters(case, 2)


def fake_result():
    specs = [NS(request_id=key, metadata={"warmup_seed": key == "seed"})
             for key in ("seed", "fresh1", "fresh2", "unfinished")]
    times = {"seed": [10.], "fresh1": [20., 25., 30., 35.],
             "fresh2": [24., 29., 34., 39.], "unfinished": [40., 45., 50.]}
    events = [NS(event_type="tokens_committed", timestamp_ns=stamp, request_id=key,
                 details={"visible_tokens": 1}) for key, stamps in times.items() for stamp in stamps]
    metrics = {key: NS(request_id=key, status="finished" if key != "unfinished" else "running",
                       rejection_reason=None, ttft_ns=stamps[0]) for key, stamps in times.items()}
    window = {"complete": True, "start_ns": 10., "end_ns": 50., "measured_decode_cohorts": 6,
              "visible_output_tokens": 11, "complete_request_tpot_ns": {"fresh1": 5., "fresh2": 5.},
              "batch_sizes": [2] * 6, "mean_iteration_ns": 40/6, "max_iteration_ns": 10.}
    return NS(scenario=NS(workload=NS(requests=specs)),
              serving=NS(decode_measurement=window, events=events, request_metrics=metrics,
                external_kv_ingress_metrics={"protocol": "native_external_kv_v1", "imported_requests": 4,
                                            "imported_logical_bytes": 2048}))


def candidate(case, batch=1, state="completed"):
    return {"parameters": run.parameters(case, batch), "candidate_id": run.candidate_id(case, batch),
        "status": state, "backend": run.BACKEND, "runner_contract": run.CONTRACT, "forecast_tokens": 0,
        "metrics": {"cluster_tokens_per_second": 800./batch, "tokens_per_second_per_gpu": 100./batch,
                    "mean_complete_request_tpot_ns": 1e7} if state == "completed" else None}


def test_exact_9_6_15_order_and_60_default_batch_candidates():
    cases = run.case_matrix()
    assert len(cases) == 30
    assert [row["phase"] for row in cases] == ["dense_native"]*9 + ["dense_plus"]*6 + ["maverick_native"]*15
    assert len({run.candidate_id(case, batch) for case in cases for batch in (1, 8)}) == 60
    assert all((case["tp"], case["pp"], case["ep"], case["gpu_count"]) == (8, 1, 1, 8) for case in cases)
    assert all(case["output_tokens"] in (373, 499, 1100) for case in cases)


def _fast_result(upper=99e6, accepted=True):
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.reporting import FastScenarioResult
    from heterollm_sim.native_fast_window import NativeWindowForecast
    scenario = build_reference_scenario()
    forecast = NativeWindowForecast(100., 5, 1111, 6, 1111, {}, {},
        {"cluster_tokens_per_second": 80., "tokens_per_second_per_gpu": 10.,
         "mean_complete_request_tpot_ns": 95e6},
        {"mean_complete_request_tpot_ns": {"estimate": 95e6, "lower": 90e6, "upper": upper}},
        (), {"accepted": True}) if accepted else None
    return FastScenarioResult(scenario, None,
        {"effective_mode": "fast" if accepted else "forecast_rejected",
         "rejection_reasons": [] if accepted else ["future_cache_not_bounded"]},
        (), "aggregate", native_window_forecast=forecast)


def test_native_fast_runner_uses_upper_bound_and_keeps_full_traffic_unknown():
    params = run.parameters(run.case_matrix()[0], 1, "fast")
    row = run.native_fast_result(_fast_result(upper=101e6), params)
    assert row["status"] == "slo_rejected" and row["metrics"] is None
    assert row["diagnostics"]["observed_metrics"]["mean_complete_request_tpot_ns"] < run.SLO_NS
    assert row["traffic"] is None and row["forecast_tokens"] == 1111
    assert run.native_fast_result(_fast_result(upper=100e6), params)["status"] == "completed"


def test_native_fast_runner_rejection_is_terminal_with_empty_metrics():
    params = run.parameters(run.case_matrix()[0], 1, "fast")
    row = run.native_fast_result(_fast_result(accepted=False), params)
    assert row["status"] == "forecast_rejected" and row["metrics"] is None
    assert row["rejection_reason"] == "future_cache_not_bounded"
    assert row["status"] in run.CANDIDATE_TERMINAL
    assert params["runner_contract"] != run.CONTRACT


@pytest.mark.parametrize("unit", ["cohorts", "layer_windows"])
def test_native_fast_assumptions_do_not_claim_complete_physical_window(unit):
    from exploring_hbf_native_config import PHYSICAL_WINDOW_ASSUMPTION
    scenario = NS(assumptions=("authored hardware assumption", PHYSICAL_WINDOW_ASSUMPTION))
    assumptions = run.native_fast_assumptions(scenario, unit)
    assert PHYSICAL_WINDOW_ASSUMPTION not in assumptions
    assert "authored hardware assumption" in assumptions
    assert any("10%" in text and "conditional" in text for text in assumptions)
    assert run.native_fast_assumptions(NS(assumptions=assumptions), unit) == assumptions
    assert any("layer windows" in text for text in assumptions) == (unit == "layer_windows")


def test_fast_paired_reference_excludes_detailed_and_mismatched_candidates(tmp_path):
    case = run.case_matrix()[0]
    params = run.parameters(run.case_matrix()[6], 1, "fast")
    path = tmp_path / "candidates" / (run.candidate_id(case, 1) + ".json")
    row = candidate(case)
    row["approximation"] = {"error_budget": {"paired_reference": {"source_id": "matched"}}}
    run.write_json(path, row)
    assert run.matched_fast_reference(params, tmp_path) is None
    row.update(parameters=run.parameters(case, 1, "fast"), runner_contract=run.FAST_CONTRACT)
    run.write_json(path, row)
    assert run.matched_fast_reference(params, tmp_path) == {"source_id": "matched"}
    row["parameters"]["output_tokens"] = 1
    run.write_json(path, row)
    assert run.matched_fast_reference(params, tmp_path) is None


def test_native_layer_runner_contract_cannot_reuse_cohort_or_detailed_reference(tmp_path):
    case, target = run.case_matrix()[0], run.case_matrix()[6]
    params = run.parameters(target, 8, "fast", "layer_windows")
    assert params["runner_contract"] == run.FAST_LAYER_CONTRACT
    assert params["target_relative_error"] == .1
    assert params["candidate_wall_budget_seconds"] == 1800.
    assert params["candidate_wall_target_seconds"] == 600.
    path = tmp_path / "candidates" / (run.candidate_id(case, 8) + ".json")
    row = candidate(case, 8)
    reference = {"source_id": "same-native-layer-contract"}
    row["approximation"] = {"error_budget": {"paired_reference": reference}}
    for mode, unit in (("detailed", "cohorts"), ("fast", "cohorts")):
        row.update(parameters=run.parameters(case, 8, mode, unit),
                   runner_contract=run.runner_contract(mode, unit))
        run.write_json(path, row)
        assert run.matched_fast_reference(params, tmp_path) is None
    row.update(parameters=run.parameters(case, 8, "fast", "layer_windows"),
               runner_contract=run.FAST_LAYER_CONTRACT)
    run.write_json(path, row)
    assert run.matched_fast_reference(params, tmp_path) == reference


def test_layer_runner_preserves_sample_scope_and_rejects_mismatched_forecast():
    from test_native_fast_reporting import layer_sample

    params = run.parameters(run.case_matrix()[0], 1, "fast", "layer_windows")
    value = _fast_result()
    with pytest.raises(RuntimeError, match="sampling contract"):
        run.native_fast_result(value, params)
    value = replace(value, native_window_forecast=replace(value.native_window_forecast,
        sampling_unit="layer_windows", sampled_decode_cohorts=0, sampled_visible_output_tokens=0),
        layer_window_samples=(layer_sample(),))
    row = run.native_fast_result(value, params)
    assert row["status"] == "completed" and row["sampled_tokens"] == 0
    assert row["traffic"] is row["physical_prefix_traffic"] is None
    assert row["physical_layer_samples"][0]["visible_output_tokens_observed"] == 0
    assert not row["physical_layer_samples"][0]["complete_native_cohort"]
    with pytest.raises(RuntimeError, match="sampling contract"):
        run.native_fast_result(replace(value, sampled=object()), params)


@pytest.mark.parametrize("stop", [False, True])
def test_fast_total_wall_budget_includes_bootstrap_and_user_pause_wins(tmp_path, monkeypatch, stop):
    import heterollm_sim.execution_control as execution_control
    value = _fast_result()
    params = run.parameters(run.case_matrix()[0], 1, "fast")
    clock = [0.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(execution_control, "perf_counter", lambda: clock[0])
    seen = []
    def runner(scenario, **options):
        seen.append(scenario.simulation)
        clock[0] = 1801.
        if stop:
            (tmp_path / "stop.request").write_text("user_pause", encoding="utf-8")
        return value
    row = run.execute_candidate(params, tmp_path, NS(report=lambda p: None),
        builder=lambda *a, **kw: value.scenario, runner=runner)
    assert seen[0].mode == "fast" and seen[0].native_window is not None
    assert row["status"] == ("paused" if stop else "forecast_rejected")
    assert row["metrics"] is None


def test_real_progress_writer_uses_native_stop_deadline_and_flushes_queue(tmp_path, monkeypatch):
    import heterollm_sim.execution_control as execution_control
    clock = [0.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(execution_control, "perf_counter", lambda: clock[0])
    value = _fast_result()
    params = run.parameters(run.case_matrix()[0], 1, "fast")
    writer = run.ProgressWriter(tmp_path, params["case_id"])
    writer.begin_candidate(params)
    def runner(scenario, **options):
        control = options["control"]
        assert execution_control.native_callback_free_control(control)
        assert control.progress_queue is writer.queue
        assert control.progress_callback is control.cancellation_callback is None
        assert control.stop_request_path == str(tmp_path / "stop.request")
        assert control.deadline_monotonic == 1800.
        clock[0] = 2.
        control.report("serving_boundary", 0, metadata={"cumulative_progress": {"visible_output_tokens": 0}})
        clock[0] = 7.
        control.report("serving_cohorts", 1, metadata={"cumulative_progress": {"visible_output_tokens": 2}})
        clock[0] = 9.
        return value
    row = run.execute_candidate(params, tmp_path, writer,
        builder=lambda *a, **kw: value.scenario, runner=runner)
    assert row["status"] == "completed"
    assert row["progress"]["cumulative_progress"]["visible_output_tokens"] == 2
    assert row["progress"]["wall_breakdown_seconds"] == {
        "starting": 2., "serving_boundary": 5., "serving_cohorts": 2.}
    writer.publish()
    assert run.read_json(writer.path)["cumulative_progress"]["visible_output_tokens"] == 2


def test_fast_layer_runner_preserves_explicit_contiguous_window_search(tmp_path):
    value = _fast_result()
    value = replace(value, native_window_forecast=replace(value.native_window_forecast,
        sampling_unit="layer_windows", sampled_decode_cohorts=0, sampled_visible_output_tokens=0))
    params = {**run.parameters(run.case_matrix()[0], 1, "fast", "layer_windows"),
        "layer_window_size": 1, "max_layer_window_size": 4}
    seen = []

    def runner(scenario, **_options):
        seen.append(scenario.simulation.native_window)
        return value

    row = run.execute_candidate(params, tmp_path, NS(report=lambda _p: None),
        builder=lambda *a, **kw: value.scenario, runner=runner)
    assert row["status"] == "completed"
    assert row["parameters"] == params
    assert seen[0].layer_window_size == 1 and seen[0].max_layer_window_size == 4
    assert seen[0].target_relative_error == .1
    assert 0 < seen[0].max_wall_seconds <= params["candidate_wall_budget_seconds"] == 1800.


def test_measurement_uses_start_exclusive_actual_tokens_and_complete_fresh_requests():
    metrics, diagnostics = run.measure_result(fake_result(), small_parameters())
    assert metrics["measurement_visible_output_tokens"] == 11
    assert metrics["complete_fresh_requests"] == 2
    assert metrics["cluster_tokens_per_second"] == 11e9/40
    assert metrics["tokens_per_second_per_gpu"] == 11e9/40/8
    assert metrics["mean_complete_request_tpot_ns"] == 5.
    assert diagnostics["complete_request_tpot_ns"] == {"fresh1": 5., "fresh2": 5.}
    assert diagnostics["unfinished_requests_at_measurement_stop"] == 1


def test_aggregate_native_maverick_pd_sram_window_reaches_runner_metrics():
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.reporting import run_scenario
    from test_native_maverick_semantics import miniature_maverick
    model = miniature_maverick(prompt=7).model
    scenario = build_native_comparison_scenario("miniature-functional", "HBF+", "SHORT",
        batch_size=2, tp=2, model_override=model, prompt_override=7, output_override=4,
        warmup_override=3, measurement_override=6)
    from heterollm_sim.execution_control import ExecutionControl
    observations = []
    result = run_scenario(scenario, retention_policy="aggregate",
                          control=ExecutionControl(progress_callback=observations.append))
    case = {**run.case_matrix()[-3], "gpu_count": 2, "tp": 2, "input_tokens": 7, "output_tokens": 4}
    metrics, diagnostics = run.measure_result(result, run.parameters(case, 2))
    assert metrics["complete_fresh_requests"] >= 2 and diagnostics["slo_met"]
    assert metrics["cluster_tokens_per_second"] == metrics["tokens_per_second_per_gpu"] * 2
    assert not any(component.normalized_kind == "hbm" for component in result.scenario.hardware.components)
    traffic = run.traffic_result(result)
    assert traffic["external_kv_ingress"]["imported_logical_bytes"] > 0
    assert traffic["combined_physical_traffic"]["NAND"]["physical_write_bytes"] > 0
    assert traffic["combined_physical_traffic"]["SRAM"]["physical_read_bytes"] > 0
    final = next(row for row in reversed(observations) if row.stage == "serving_complete")
    counters = final.metadata["cumulative_progress"]
    assert counters["cohorts"] == len(result.serving.batches)
    assert counters["visible_output_tokens"] == sum(event.details.get("visible_tokens", 0)
        for event in result.serving.events if event.event_type == "tokens_committed")
    assert counters["prefill_compute_tokens"] == sum(event.details["chunk_tokens"]
        for event in result.serving.events if event.event_type == "prefill_chunk_complete")
    assert counters["measurement_decode_cohorts"] == 6
    assert counters["measurement_visible_output_tokens"] == metrics["measurement_visible_output_tokens"]
    assert counters["decode_cohorts"] == counters["target_decode_cohorts"] == 9
    assert counters["external_imported_requests"] > 0
    assert counters["prefill_compute_tokens"] < 7 * counters["external_imported_requests"]
    stages = {row.stage for row in observations}
    assert {"cohort_preparation", "cohort_lowering", "cohort_scheduling", "cohort_tasks", "cohort_commit"} <= stages
    unobserved = run_scenario(scenario, retention_policy="aggregate")
    assert unobserved.serving.events == result.serving.events
    assert [(batch.start_ns, batch.end_ns) for batch in unobserved.serving.batches] == [
        (batch.start_ns, batch.end_ns) for batch in result.serving.batches]
    assert run.traffic_result(unobserved) == traffic


def test_100ms_gate_uses_mean_request_tpot_instead_of_maximum_request_gap():
    result = fake_result()
    replacement = {"seed": [10.], "fresh1": [20. + 10e6*i for i in range(4)],
                   "fresh2": [24. + 190e6*i for i in range(4)],
                   "unfinished": [590e6, 595e6, 600e6]}
    result.serving.events = [NS(event_type="tokens_committed", timestamp_ns=stamp,
        request_id=key, details={"visible_tokens": 1}) for key, stamps in replacement.items() for stamp in stamps]
    result.serving.decode_measurement.update(end_ns=600e6,
        complete_request_tpot_ns={"fresh1": 10e6, "fresh2": 190e6})
    metrics, diagnostics = run.measure_result(result, small_parameters())
    assert metrics["mean_complete_request_tpot_ns"] == 100e6
    assert metrics["max_complete_request_tpot_ns"] == 190e6
    assert diagnostics["slo_met"] is True


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.serving.decode_measurement.update(complete=False), "did not complete"),
    (lambda r: r.serving.decode_measurement.update(visible_output_tokens=12), "visible-token"),
    (lambda r: r.serving.decode_measurement.update(complete_request_tpot_ns={}), "TPOT statistics"),
    (lambda r: r.serving.decode_measurement.update(batch_sizes=[1]*6), "full batch"),
    (lambda r: r.serving.external_kv_ingress_metrics.update(imported_logical_bytes=0), "real imported"),
    (lambda r: setattr(r.serving.request_metrics["fresh1"], "status", "rejected"), "rejected"),
])
def test_incomplete_or_unexplained_measurements_are_failures(mutation, reason):
    result = fake_result()
    mutation(result)
    with pytest.raises(RuntimeError, match=reason):
        run.measure_result(result, small_parameters())


def test_best_actual_passing_candidate_and_real_failed_candidate_are_distinct():
    case = run.case_matrix()[0]
    a, b = candidate(case), candidate(case, 8, "slo_rejected")
    selected = run.select_case(case, [1, 8], [a, b])
    assert selected["status"] == "completed"
    assert selected["selected"] == a
    assert "not a global optimum" in selected["selection_scope"]
    b["status"] = "failed"
    b["error"] = {"type": "MemoryError", "message": "host out of memory"}
    rejected = run.select_case(case, [1, 8], [a, b])
    assert rejected["status"] == "failed" and rejected["selected"] is None
    assert run.select_case(case, [1, 8], [candidate(case, 1, "capacity_rejected"),
        candidate(case, 8, "slo_rejected")])["status"] == "all_candidates_rejected"


def test_unknown_and_host_memory_errors_are_not_device_capacity_rejections():
    from heterollm_sim.memory_allocator import AllocationError
    assert run.capacity_error(AllocationError("unable to allocate 1024 bytes in capacity 512"))
    assert not run.capacity_error(AllocationError("allocation overlaps another without an explicit alias"))
    assert not run.capacity_error(MemoryError("capacity exhausted in host memory"))
    assert not run.capacity_error(RuntimeError("unexpected bug mentioning capacity"))


def test_traffic_adds_pd_once_and_missing_sram_is_unknown(monkeypatch):
    import heterollm_sim.reporting as reporting
    keys = ("logical_read_bytes", "logical_write_bytes", "physical_read_bytes", "physical_write_bytes", "physical_bytes")
    batch = dict(zip(keys, (10, 20, 64, 128, 192)))
    pd = dict(zip(keys, (1, 2, 8, 16, 24)))
    monkeypatch.setattr(reporting, "_sum_batch_dram_traffic", lambda _: batch)
    monkeypatch.setattr(reporting, "_sum_batch_storage_traffic", lambda _: batch)
    result = NS(scenario=NS(hardware=NS(components=[NS(normalized_kind="sram")])),
        serving=NS(batches=[NS(cost=NS(metadata={}))], external_kv_ingress_metrics={
            "cost_ledger": {"dram_traffic": pd, "storage_traffic": pd, "phases": ["large real details"]}}))
    traffic = run.traffic_result(result)
    assert traffic["combined_physical_traffic"]["DRAM"]["physical_bytes"] == 216
    assert traffic["combined_physical_traffic"]["NAND"]["physical_write_bytes"] == 144
    assert traffic["combined_physical_traffic"]["SRAM"]["physical_bytes"] is None
    assert "phases" not in traffic["external_kv_ingress"]["cost_ledger"]
    result.serving.batches[0].cost.metadata["sram_traffic"] = batch
    assert run.traffic_result(result)["combined_physical_traffic"]["SRAM"]["physical_bytes"] == 192


def test_rejected_candidates_keep_official_csv_metrics_blank(tmp_path):
    case = run.case_matrix()[0]
    row = candidate(case, 1, "slo_rejected")
    row["diagnostics"] = {"observed_metrics": {"mean_complete_request_tpot_ns": 101e6}}
    row["rejection_reason"] = "over 100ms"
    run.write_json(tmp_path / "candidates" / (row["candidate_id"] + ".json"), row)
    run.refresh(tmp_path, [case], [1])
    import csv
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        saved, = list(csv.DictReader(handle))
    assert saved["tokens_per_second_per_gpu"] == saved["mean_complete_request_tpot_ns"] == ""
    assert saved["status"] == "slo_rejected"


def test_final_rejections_and_live_cases_are_not_reported_as_all_success(tmp_path):
    cases = run.case_matrix()
    for case in cases:
        row = candidate(case, state="capacity_rejected" if case == cases[0] else "completed")
        run.write_json(tmp_path / "points" / (case["case_id"] + ".json"),
                       run.select_case(case, [1], [row]))
    assert run.run_manager(tmp_path, batches=[1], spawn=lambda *a: pytest.fail("terminal point rerun")) == 0
    saved = run.read_json(tmp_path / "status.json")
    assert saved["state"] == "completed_with_rejections"
    assert saved["counts"] == {"all_candidates_rejected": 1, "completed": 29}
    active = cases[0]
    (tmp_path / "points" / (active["case_id"] + ".json")).unlink()
    state = run.refresh(tmp_path, cases, [1], [{"case_id": active["case_id"], "launcher_pid": 1}])
    assert state["counts"]["running"] == 1 and state["counts"]["completed"] == 29


def test_worker_reuses_exact_terminal_candidates_without_executing_native(tmp_path, monkeypatch):
    case = run.case_matrix()[0]
    rows = [candidate(case, batch) for batch in (1, 8)]
    for row in rows:
        run.write_json(tmp_path / "candidates" / (row["candidate_id"] + ".json"), row)
    monkeypatch.setattr(run, "execute_candidate", lambda *a, **k: pytest.fail("completed candidate rerun"))
    assert run.worker_case(case, [1, 8], tmp_path) == 0
    saved = run.read_json(tmp_path / "points" / (case["case_id"] + ".json"))
    assert saved["status"] == "completed"
    assert saved["selected"]["parameters"]["batch_size"] == 1
    assert run.read_json(tmp_path / "progress" / (case["case_id"] + ".json"))["state"] == "completed"


def test_stop_marker_is_preserved_and_no_candidate_is_started(tmp_path):
    marker = tmp_path / "stop.request"
    marker.write_text("user paused", encoding="utf-8")
    assert run.run_manager(tmp_path, spawn=lambda *a: pytest.fail("stop ignored"), poll_seconds=0) == 0
    assert marker.read_text(encoding="utf-8") == "user paused"
    assert run.read_json(tmp_path / "status.json")["state"] == "paused"


def test_phase_barriers_concurrency_and_terminal_resume(tmp_path):
    cases = run.case_matrix()
    spawned = []
    active = []
    class Process:
        def __init__(self, case):
            self.case, self.pid = case, 123456 + len(spawned)
        def poll(self):
            active.remove(self.case["case_id"])
            run.write_json(tmp_path / "points" / (self.case["case_id"] + ".json"),
                           run.select_case(self.case, [1], [candidate(self.case)]))
            return 0
    def spawn(case, batches, directory):
        assert len(active) < 2
        index = cases.index(case)
        for prior in cases[:9 if index >= 9 else 0] + (cases[9:15] if index >= 15 else []):
            assert run.read_json(tmp_path / "points" / (prior["case_id"] + ".json"))["status"] == "completed"
        active.append(case["case_id"])
        spawned.append(case["case_id"])
        return Process(case)
    assert run.run_manager(tmp_path, batches=[1], spawn=spawn, poll_seconds=0) == 0
    assert spawned == [row["case_id"] for row in cases]
    assert run.read_json(tmp_path / "status.json")["state"] == "completed"
    assert run.run_manager(tmp_path, batches=[1], spawn=lambda *a: pytest.fail("terminal case rerun")) == 0


def test_failed_worker_stops_queue_and_preserves_empty_official_metrics(tmp_path):
    class Process:
        pid = 123456
        def poll(self):
            return 1
    assert run.run_manager(tmp_path, workers=1, batches=[1], spawn=lambda *a: Process(), poll_seconds=0) == 1
    status = run.read_json(tmp_path / "status.json")
    assert status["state"] == "failed" and status["counts"]["failed"] == 1
    assert (tmp_path / "stop.request").exists()
    point = run.read_json(tmp_path / "points" / (run.case_matrix()[0]["case_id"] + ".json"))
    assert point["selected"] is None and point["error"]["type"] == "WorkerExit"


def test_live_manager_or_worker_is_never_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "pid_alive", lambda pid: True)
    with run.ManagerLock(tmp_path):
        with pytest.raises(RuntimeError, match="lock is live"):
            run.run_manager(tmp_path)
    run.write_json(tmp_path / "progress" / "active.json", {"pid": 424242, "state": "running"})
    with pytest.raises(RuntimeError, match="worker is still alive"):
        run.run_manager(tmp_path)


def test_historical_backend_is_not_reused_as_native(tmp_path):
    case = run.case_matrix()[0]
    run.write_json(tmp_path / "points" / (case["case_id"] + ".json"),
        {"case": case, "status": "completed", "backend": "paper_bulk_tile_events", "runner_contract": run.CONTRACT})
    with pytest.raises(RuntimeError, match="historical reuse"):
        run.run_manager(tmp_path)


@pytest.mark.parametrize("workers", [0, 3, True])
def test_parallel_worker_limit_is_explicit(tmp_path, workers):
    with pytest.raises(ValueError, match="at most two"):
        run.run_manager(tmp_path, workers=workers)


def test_progress_keeps_committed_counters_when_next_task_count_resets(tmp_path, monkeypatch):
    from heterollm_sim.execution_control import ExecutionProgress
    clock = NS(now=0.)
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock.now)
    writer = run.ProgressWriter(tmp_path, run.case_matrix()[0]["case_id"])
    params = small_parameters()
    writer.begin_candidate(params)
    clock.now = 2.
    writer.report(ExecutionProgress("cohort_tasks", 25, 100,
        metadata={"current_cohort": {"cohort_id": "first", "index": 1}}))
    clock.now = 5.
    writer.report(ExecutionProgress("serving_cohorts", 1, metadata={"cumulative_progress": {
        "cohorts": 1, "visible_output_tokens": 2, "prefill_compute_tokens": 2}}))
    clock.now = 6.
    writer.report(ExecutionProgress("cohort_preparation", 1,
        metadata={"current_cohort": {"cohort_id": "second", "index": 2}}))
    clock.now = 7.
    writer.report(ExecutionProgress("cohort_tasks", 0, 150))
    clock.now = 9.
    writer.report(ExecutionProgress("cohort_tasks", 50, 150))
    value = writer.snapshot()
    assert value["cumulative_progress"]["cohorts"] == 1
    assert value["cumulative_progress"]["visible_output_tokens"] == 2
    assert value["current_cohort"] == {"cohort_id": "second", "index": 2,
        "completed_tasks": 50, "total_tasks": 150}
    assert value["wall_breakdown_seconds"] == {"starting": 2., "cohort_tasks": 5.,
        "serving_cohorts": 1., "cohort_preparation": 1.}
    assert sum(value["wall_breakdown_seconds"].values()) == value["candidate_wall_seconds"] == 9.
    writer.begin_candidate({**params, "batch_size": 8})
    reset = writer.snapshot()
    assert reset["cumulative_progress"]["visible_output_tokens"] == 0
    assert reset["current_cohort"] is None
    assert reset["candidate_wall_seconds"] == 0.
