"""The native experiment runner schedules/observes; it never models latency."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exploring_hbf_native as run


LAYOUT = run.CONTROL_LAYOUT


def small_parameters(layout=LAYOUT):
    case = {**run.case_matrix()[0], "output_tokens": 4}
    return run.parameters(case, layout, 2)


def fake_result():
    specs = [NS(request_id=key, metadata={"warmup_seed": key == "seed"})
             for key in ("seed", "fresh1", "fresh2", "unfinished")]
    times = {"seed": [10.], "fresh1": [20., 25., 30., 35.],
             "fresh2": [24., 29., 34., 39.], "unfinished": [40., 45., 50.]}
    events = [NS(event_type="external_kv_import_start", timestamp_ns=stamp, request_id=key, details={})
              for key, stamp in (("fresh1", 2.), ("fresh2", 4.), ("unfinished", 6.))]
    events += [NS(event_type="tokens_committed", timestamp_ns=stamp, request_id=key,
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


def candidate(case, batch=1, state="completed", layout=LAYOUT):
    return {"parameters": run.parameters(case, layout, batch), "candidate_id": run.candidate_id(case, layout, batch),
        "status": state, "backend": run.BACKEND, "runner_contract": run.CONTRACT, "forecast_tokens": 0,
        "metrics": {"cluster_tokens_per_second": 800./batch, "tokens_per_second_per_gpu": 100./batch,
                    "mean_complete_request_tpot_ns": 1e7} if state == "completed" else None}


def test_exact_9_6_15_order_and_60_default_batch_candidates():
    cases = run.case_matrix()
    assert len(cases) == 30
    assert [row["phase"] for row in cases] == ["dense_native"]*9 + ["dense_plus"]*6 + ["maverick_native"]*15
    assert len({run.candidate_id(case, LAYOUT, batch) for case in cases for batch in (1, 8)}) == 60
    # The layout is a searched candidate dimension, not part of the case.
    assert all(case["gpu_count"] == 8 and not {"dp", "tp", "pp", "ep"} & case.keys() for case in cases)
    assert all(case["output_tokens"] in (373, 499, 1100) for case in cases)


def test_pipeline_microbatch_layout_identity_reaches_worker_and_parameters(tmp_path):
    case = run.case_matrix()[0]
    plain = run.make_layout(1, 4, 2, 1)
    split = run.make_layout(1, 4, 2, 1, microbatches=2)
    params = run.parameters(case, split, 4, "fast")
    assert run.params_layout(params) == split
    assert run.params_candidate_id(params) != run.candidate_id(case, plain, 4)
    assert run.layout_label(split).endswith("_M2")
    command = run.candidate_command(case, split, 4, tmp_path, "fast")
    assert command[command.index("--microbatches") + 1] == "2"
    assert run.layout_rejection(case, split, 3) == "batch_size must be divisible by ep * microbatches"
    with pytest.raises(ValueError, match="positive integer"):
        run.make_layout(1, 4, 2, 1, microbatches=True)


@pytest.mark.parametrize("layout", [run.make_layout(1, 4, 2, 1), run.make_layout(1, 8, 1, 1, microbatches=2)])
def test_pipeline_contract_refuses_old_candidate_and_case_cache(tmp_path, layout):
    case = run.case_matrix()[0]
    params = run.parameters(case, layout, 4, "fast")
    assert params["pipeline_execution_contract"] == run.PIPELINE_FORECAST_CONTRACT
    assert "pipeline_execution_contract" not in run.parameters(case, LAYOUT, 4, "fast")
    legacy = {key: value for key, value in params.items() if key != "pipeline_execution_contract"}
    identity = run.params_candidate_id(params)
    run.save_candidate(tmp_path, {"parameters": legacy, "candidate_id": identity, "backend": run.BACKEND,
        "runner_contract": run.FAST_CONTRACT, "status": "forecast_rejected", "metrics": None})
    assert not run._saved_terminal(tmp_path, case, layout, 4, "fast")
    with pytest.raises(RuntimeError, match="saved candidate parameters/backend differ"):
        run._search_lookup(tmp_path, case, layout)(4)
    run.write_json(tmp_path / "points" / (case["case_id"] + ".json"), {
        "case": case, "backend": run.BACKEND, "runner_contract": run.FAST_CONTRACT,
        "status": "all_candidates_rejected", "planned_candidate_ids": [identity]})
    with pytest.raises(RuntimeError, match="different pipeline execution contract"):
        run._reusable_point(case, [identity], tmp_path, "fast", layout=layout)


def test_pp_prefill_cache_has_a_distinct_contract(tmp_path):
    from heterollm_sim.reference import build_reference_scenario
    params = run.parameters(run.case_matrix()[0], run.make_layout(1, 4, 2, 1), 4, "fast")
    expected = run.prefill_parameters(params)
    assert expected["prefill_contract"] == run.PIPELINE_PREFILL_CONTRACT != run.PREFILL_CONTRACT
    assert run.prefill_parameters(run.parameters(run.case_matrix()[0], LAYOUT, 4, "fast"))["prefill_contract"] == run.PREFILL_CONTRACT
    identity = run.prefill_id(expected)
    run.write_json(tmp_path / "prefill" / (identity + ".json"), {
        "parameters": {**expected, "prefill_contract": run.PREFILL_CONTRACT}, "backend": run.BACKEND,
        "status": "rejected", "rejection_reasons": ["old_pipeline_unsupported"]})
    observed = []
    def runner(scenario, **kwargs):
        observed.append(scenario)
        return NS(approximation={"prefill_pass": {"accepted": True}, "rejection_reasons": []},
                  layer_window_samples=())
    value = run.prefill_pass(params, tmp_path, builder=lambda *a, **k: build_reference_scenario(), runner=runner)
    assert len(observed) == 1 and value["status"] == "accepted"


def test_pipeline_manager_refuses_a_manifest_without_execution_contract(tmp_path, monkeypatch):
    case, layout = run.case_matrix()[0], run.make_layout(1, 4, 2, 1)
    monkeypatch.setattr(run, "selected_cases", lambda case_ids=None: [case])
    monkeypatch.setattr(run, "check_grid", lambda *args: None)
    run.write_json(tmp_path / "run_manifest.json", {"runner_contract": run.FAST_CONTRACT,
        "cases": [case], "layout": layout, "max_parallel_configurations": 1, "batch_sizes": [4]})
    with pytest.raises(RuntimeError, match="existing native manifest differs"):
        run.run_manager(tmp_path, layout=layout, batches=[4], workers=1, mode="fast")


def test_exact_pipeline_window_is_measured_without_a_forecast(monkeypatch):
    from heterollm_sim.reporting import FastScenarioResult
    from heterollm_sim.reference import build_reference_scenario
    scenario = build_reference_scenario()
    scenario = replace(scenario, placement=replace(scenario.placement,
        parallel=replace(scenario.placement.parallel, pp_degree=2, rank_mapping=())))
    detailed = fake_result()
    result = FastScenarioResult(scenario, None,
        {"method": "native_pipeline_full_window/v1", "effective_mode": "detailed"},
        (), "aggregate", detailed=detailed)
    monkeypatch.setattr(run, "traffic_result", lambda measured: {"native": measured is detailed})
    params = small_parameters(run.make_layout(2, 2, 2, 1, microbatches=2))
    row = run.native_fast_result(result, params)
    assert row["status"] == "completed"
    assert row["metrics"]["mean_complete_request_tpot_ns"] == 5.
    assert row["metrics"]["cluster_tokens_per_second"] == 550e6
    assert row["physical_tokens"] == 12
    assert row["forecast_tokens"] == 0 and "native_window_forecast" not in row
    assert row["traffic"] == {"native": True}
    for name, bound in row["diagnostics"]["metric_bounds"].items():
        assert bound["lower"] == bound["estimate"] == bound["upper"] == row["metrics"][name]
        assert bound["relative_error_budget"] == 0.
    assert not any("SAME_CLASS" in value or "same native work" in value for value in row["assumptions"])


def test_pp_prefill_uses_chunk_sampling_and_does_not_inherit_decode_microbatches(tmp_path):
    from heterollm_sim.reference import build_reference_scenario
    params = run.parameters(run.case_matrix()[0], run.make_layout(1, 4, 2, 1, microbatches=2), 4, "fast")
    seen = []
    def builder(*args, **kwargs):
        seen.append(kwargs)
        return build_reference_scenario()
    def runner(scenario, **kwargs):
        assert not scenario.simulation.native_prefill.full_pass
        assert scenario.simulation.native_prefill.chunked_sampling
        assert scenario.simulation.native_prefill.max_wall_seconds == run.PREFILL_WALL_BUDGET_SECONDS
        assert kwargs["control"].deadline_monotonic is not None
        return NS(approximation={"prefill_pass": {"accepted": True}, "rejection_reasons": []},
                  layer_window_samples=())
    result = run.prefill_pass(params, tmp_path, builder=builder, runner=runner)
    assert result["parameters"]["pass"] == "sampled"
    assert seen == [{"tp": 4, "pp": 2, "ep": 1, "full_pass": False}]


def _fast_result(upper=99e6, accepted=True):
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.reporting import FastScenarioResult
    from heterollm_sim.native_fast_window import NativeWindowForecast
    scenario = build_reference_scenario()
    forecast = NativeWindowForecast(100., 0, 1111, 0, 1111, {}, {},
        {"cluster_tokens_per_second": 80., "tokens_per_second_per_gpu": 10.,
         "mean_complete_request_tpot_ns": 95e6},
        {"mean_complete_request_tpot_ns": {"estimate": 95e6, "lower": 90e6, "upper": upper}},
        (), {"accepted": True}) if accepted else None
    return FastScenarioResult(scenario, None,
        {"effective_mode": "fast" if accepted else "forecast_rejected",
         "rejection_reasons": [] if accepted else ["future_cache_not_bounded"]},
        (), "aggregate", native_window_forecast=forecast)


def test_pipeline_forecast_result_is_terminal_for_worker_reuse():
    from dataclasses import replace
    from heterollm_sim.native_pipeline_forecast import NativePipelineForecast
    forecast = NativePipelineForecast(0., 1, 3, 0, 8,
        {"cluster_tokens_per_second": 80., "mean_complete_request_tpot_ns": 95e6},
        {"cluster_tokens_per_second": {"estimate": 80., "lower": 79., "upper": 81.},
         "mean_complete_request_tpot_ns": {"estimate": 95e6, "lower": 94e6, "upper": 96e6}},
        (), (), {"relative_error": .01})
    result = replace(_fast_result(), approximation={"method": "native_pipeline_stage_forecast/v2"},
                     native_window_forecast=None, native_pipeline_forecast=forecast)
    params = run.parameters(run.case_matrix()[0], run.make_layout(1, 4, 2, 1), 4, "fast")
    row = run.native_fast_result(result, params)
    assert row["status"] == "completed"


def test_pipeline_forecast_uses_effective_pipeline_error_target():
    from dataclasses import replace
    from heterollm_sim.native_pipeline_forecast import NativePipelineForecast
    forecast = NativePipelineForecast(0., 1, 3, 0, 8,
        {"cluster_tokens_per_second": 80., "mean_complete_request_tpot_ns": 95e6},
        {"mean_complete_request_tpot_ns": {"estimate": 90e6, "lower": 80e6, "upper": 99e6}},
        (), (), {"target_relative_error": .20, "relative_error": .12})
    result = replace(_fast_result(), approximation={"method": "native_pipeline_stage_forecast/v2"},
                     native_window_forecast=None, native_pipeline_forecast=forecast)
    params = run.parameters(run.case_matrix()[0], run.make_layout(1, 4, 2, 1), 4, "fast")
    row = run.native_fast_result(result, params)
    assert row["status"] == "completed"


PREFILL_LAYOUT = {"tp": 8, "pp": 1, "ep": 1}
REJECTED_PREFILL = {"prefill_id": "prefill", "status": "rejected", "prefill_pass": {"metric_bounds": {}},
                    "parameters": PREFILL_LAYOUT, "rejection_reasons": ["native_prefill_independent_budget_exceeded"], "wall_seconds": 1.}


def _prefill(lower, upper, observed=None):
    from heterollm_sim.native_stationary import _interval
    return {"prefill_id": "prefill", "status": "accepted", "rejection_reasons": [], "wall_seconds": 1.,
            "parameters": PREFILL_LAYOUT, "prefill_pass": {"metric_bounds": {"prefill_pass_ns": _interval(lower, upper)},
                             "error_budget": {"observed_pass_ns": observed}}}


def test_ttft_adds_the_prefill_pass_to_the_ingress_ttft():
    from heterollm_sim.native_stationary import _interval
    fast = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    ingress = _interval(40e6, 44e6)
    row = {"metrics": {"mean_complete_request_ingress_ttft_ns": ingress["estimate"]},
           "diagnostics": {"metric_bounds": {"mean_complete_request_ingress_ttft_ns": ingress}}}
    run.with_prefill_ttft(row, fast, _prefill(100e6, 108e6))
    total = row["diagnostics"]["metric_bounds"]["mean_complete_request_ttft_ns"]
    assert (total["lower"], total["upper"]) == (140e6, 152e6)
    # The sum's relative budget lies between its parts' (mediant).
    assert total["relative_error_budget"] <= max(ingress["relative_error_budget"], 8 / 208)
    assert row["metrics"]["mean_complete_request_ttft_ns"] == total["estimate"]
    assert row["metrics"]["prefill_pass_ns"] == _interval(100e6, 108e6)["estimate"]
    detailed = run.parameters(run.case_matrix()[0], LAYOUT, 1)
    row = {"metrics": {"mean_complete_request_ingress_ttft_ns": 42e6}}
    run.with_prefill_ttft(row, detailed, _prefill(99.9e6, 100.1e6, observed=100e6))
    assert row["metrics"]["mean_complete_request_ttft_ns"] == 142e6
    row = {"metrics": {"mean_complete_request_ingress_ttft_ns": 42e6}}
    run.with_prefill_ttft(row, detailed, REJECTED_PREFILL)
    assert row["metrics"]["mean_complete_request_ttft_ns"] is row["metrics"]["prefill_pass_ns"] is None
    assert row["prefill_pass"]["rejection_reasons"] == ["native_prefill_independent_budget_exceeded"]


def test_pipeline_prefill_without_ingress_bound_is_a_forecast_rejection():
    params = run.parameters(run.case_matrix()[0], run.make_layout(1, 4, 2, 1), 4, "fast")
    row = {"status": "completed", "metrics": {"mean_complete_request_tpot_ns": 1e6},
           "diagnostics": {"metric_bounds": {"mean_complete_request_tpot_ns":
               {"lower": 1e6, "upper": 1e6, "estimate": 1e6, "relative_error_budget": 0.}}}}
    run.with_prefill_ttft(row, params, _prefill(10e6, 11e6))
    assert row["status"] == "forecast_rejected"
    assert row["rejection_reason"] == "native_pipeline_ingress_ttft_unavailable"


def test_prefill_pass_runs_once_per_model_prompt_and_layout(tmp_path):
    seen = []

    def builder(model, workload, **options):
        seen.append((model, workload, options))
        return _fast_result().scenario

    def runner(scenario, **options):
        assert scenario.simulation.native_prefill.full_pass is False
        assert scenario.simulation.native_prefill.chunked_sampling is True
        assert scenario.simulation.native_prefill.max_wall_seconds == run.PREFILL_WALL_BUDGET_SECONDS
        return NS(approximation={"prefill_pass": {"accepted": True}, "rejection_reasons": []},
                  layer_window_samples=())

    hbm, hbf = run.case_matrix()[0], run.case_matrix()[6]
    assert hbm["hardware"] != hbf["hardware"] and hbm["workload"] == hbf["workload"]
    first = run.prefill_pass(run.parameters(hbm, LAYOUT, 1, "fast"), tmp_path, builder=builder, runner=runner)
    again = run.prefill_pass(run.parameters(hbf, LAYOUT, 8, "fast"), tmp_path, builder=builder,
                             runner=lambda *a, **k: pytest.fail("shared Prefill pass rerun"))
    assert first == again and first["status"] == "accepted"
    assert seen == [(hbm["model"], hbm["workload"], {"tp": 8, "pp": 1, "ep": 1, "full_pass": False})]
    assert first["parameters"]["prefill_hardware"] == "HBM4" and "batch_size" not in first["parameters"]
    def detailed(scenario, **options):
        # The detailed contract's full pass has no wall cap, like its Decode run.
        assert scenario.simulation.native_prefill.max_wall_seconds is None
        assert options["control"].deadline_monotonic is None
        return NS(approximation={"prefill_pass": {"accepted": scenario.simulation.native_prefill.full_pass},
                                 "rejection_reasons": []}, layer_window_samples=())

    assert run.prefill_pass(run.parameters(hbm, LAYOUT, 1), tmp_path, builder=builder,
                            runner=detailed)["status"] == "accepted"
    assert len(seen) == 2 and seen[-1][2]["full_pass"] is True


@pytest.mark.parametrize("previous_contract", [None, "native_prefill_whole_prompt_forward_v1"])
def test_prefill_contract_invalidates_old_rejection_without_changing_decode(tmp_path, monkeypatch, previous_contract):
    monkeypatch.setattr(run, "prefill_tp", lambda model, workload, tp, pp, ep: tp)
    case = next(row for row in run.case_matrix() if
        (row["model"], row["hardware"], row["workload"]) == ("llama4_maverick", "HBM4", "LONG"))
    params = run.parameters(case, run.make_layout(1, 4, 1, 2), 228, "fast")
    original_params = deepcopy(params)
    expected = run.prefill_parameters(params)
    legacy = {key: value for key, value in expected.items() if key != "prefill_contract"}
    if previous_contract is not None:
        legacy["prefill_contract"] = previous_contract
    path = tmp_path / "prefill" / (run.prefill_id(expected) + ".json")
    run.write_json(path, {"backend": run.BACKEND, "parameters": legacy, "status": "rejected",
        "prefill_pass": {"accepted": False},
        "rejection_reasons": ["native_prefill_pass_requires_one_whole_prompt_forward"]})
    runs = []

    def runner(scenario, **options):
        runs.append(scenario)
        return NS(approximation={"prefill_pass": {"accepted": True}, "rejection_reasons": []},
                  layer_window_samples=())

    result = run.prefill_pass(params, tmp_path, builder=lambda *a, **k: _fast_result().scenario, runner=runner)
    assert result["status"] == "accepted" and result["parameters"]["prefill_contract"] == run.PREFILL_CONTRACT
    again = run.prefill_pass(params, tmp_path,
        builder=lambda *a, **k: pytest.fail("current Prefill cache rebuilt"),
        runner=lambda *a, **k: pytest.fail("current Prefill cache rerun"))
    assert result == again == run.read_json(path) and len(runs) == 1
    assert params == original_params and "prefill_contract" not in params
    assert params["runner_contract"] == run.FAST_CONTRACT == "native_stationary_layout_fast_v21"


def test_native_fast_runner_uses_upper_bound_and_keeps_full_traffic_unknown():
    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    row = run.native_fast_result(_fast_result(upper=101e6), params)
    assert row["status"] == "slo_rejected" and row["metrics"] is None
    assert row["diagnostics"]["observed_metrics"]["mean_complete_request_tpot_ns"] < run.SLO_NS
    assert row["traffic"] is None and row["forecast_tokens"] == 1111
    assert run.native_fast_result(_fast_result(upper=100e6), params)["status"] == "completed"


def test_native_fast_runner_rejection_is_terminal_with_empty_metrics():
    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    row = run.native_fast_result(_fast_result(accepted=False), params)
    assert row["status"] == "forecast_rejected" and row["metrics"] is None
    assert row["rejection_reason"] == "future_cache_not_bounded"
    assert row["status"] in run.CANDIDATE_TERMINAL
    assert params["runner_contract"] != run.CONTRACT


def test_budget_rejected_point_reports_its_intervals_as_missing_the_target(tmp_path):
    from dataclasses import replace
    case = run.case_matrix()[0]
    params = run.parameters(case, LAYOUT, 1, "fast")
    bounds = {"cluster_tokens_per_second": {"estimate": 80., "lower": 70., "upper": 90., "relative_error_budget": .125},
              "mean_complete_request_tpot_ns": {"estimate": 95e6, "lower": 85e6, "upper": 105e6, "relative_error_budget": .11}}
    value = _fast_result(accepted=False)
    value = replace(value, approximation={**value.approximation, "metric_bounds": bounds,
        "rejection_reasons": ["native_stationary_independent_budget_exceeded"]})
    row = run.native_fast_result(value, params)
    assert row["status"] == "forecast_rejected" and row["metrics"] is None
    assert row["rejected_metric_bounds"] == {"target_relative_error": .1, "target_met": False,
        "label": "未达到10%目标的条件区间，不是通过验收的结果", "metrics": bounds}
    assert "rejected_metric_bounds" not in run.native_fast_result(_fast_result(accepted=False), params)
    identity = run.candidate_id(case, LAYOUT, 1)
    row = {**row, "parameters": params, "candidate_id": identity, "backend": run.BACKEND}
    run.write_json(tmp_path / "candidates" / (identity + ".json"), row)
    run.refresh(tmp_path, [case], {case["case_id"]: [(LAYOUT, 1)]}, mode="fast")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "未达到10%目标" in report and "70.000～90.000" in report and "12.50%（未达标）" in report
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "unaccepted_cluster_tokens_per_second_lower" in summary and ",70.0,90.0," in summary
    point = run.select_case(case, [identity], [row], "fast")
    assert point["status"] == "all_candidates_rejected" and point["selected"] is None


def test_native_fast_assumptions_do_not_claim_complete_physical_window():
    from exploring_hbf_native_config import PHYSICAL_WINDOW_ASSUMPTION
    scenario = NS(assumptions=("authored hardware assumption", PHYSICAL_WINDOW_ASSUMPTION),
                  placement=NS(parallel=NS(pp_degree=1, pipeline_microbatches=1)))
    assumptions = run.native_fast_assumptions(scenario)
    assert PHYSICAL_WINDOW_ASSUMPTION not in assumptions
    assert "authored hardware assumption" in assumptions
    assert any("10%" in text and "conditional" in text for text in assumptions)
    assert run.native_fast_assumptions(NS(assumptions=assumptions, placement=scenario.placement)) == assumptions
    assert any("continuous-refill" in text for text in assumptions)
    from heterollm_sim.native_stationary import SAME_CLASS_ADDRESS_ASSUMPTION
    assert SAME_CLASS_ADDRESS_ASSUMPTION in assumptions


def test_fast_runner_contract_keeps_the_budget_and_sample_scope():
    from test_native_fast_reporting import layer_sample

    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    assert params["runner_contract"] == run.FAST_CONTRACT != run.CONTRACT
    assert params["target_relative_error"] == .1
    assert params["candidate_wall_budget_seconds_per_worker"] == 3000.
    value = replace(_fast_result(), layer_window_samples=(layer_sample(),))
    row = run.native_fast_result(value, params)
    assert row["status"] == "completed" and row["sampled_tokens"] == 0
    assert row["traffic"] is None and "physical_prefix_traffic" not in row
    assert row["physical_layer_samples"][0]["visible_output_tokens_observed"] == 0
    assert not row["physical_layer_samples"][0]["complete_native_cohort"]
    with pytest.raises(RuntimeError, match="partial cohort"):
        run.native_fast_result(replace(value, sampled=object()), params)


@pytest.mark.parametrize("stop", [False, True])
def test_fast_total_wall_budget_includes_bootstrap_and_user_pause_wins(tmp_path, monkeypatch, stop):
    import heterollm_sim.execution_control as execution_control
    value = _fast_result()
    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    clock = [0.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(execution_control, "perf_counter", lambda: clock[0])
    seen = []
    def runner(scenario, **options):
        seen.append(scenario.simulation)
        clock[0] = 3001.
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
    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    writer = run.ProgressWriter(tmp_path, run.params_candidate_id(params))
    writer.begin_candidate(params)
    def runner(scenario, **options):
        control = options["control"]
        assert execution_control.native_callback_free_control(control)
        assert control.progress_queue is writer.queue
        assert control.progress_callback is control.cancellation_callback is None
        assert control.stop_request_path == str(tmp_path / "stop.request")
        assert control.deadline_monotonic == 3000.
        clock[0] = 2.
        control.report("serving_boundary", 0, metadata={"cumulative_progress": {"visible_output_tokens": 0}})
        clock[0] = 7.
        control.report("serving_cohorts", 1, metadata={"cumulative_progress": {"visible_output_tokens": 2}})
        clock[0] = 9.
        return value
    row = run.execute_candidate(params, tmp_path, writer,
        builder=lambda *a, **kw: value.scenario, runner=runner, prefill=lambda *a: REJECTED_PREFILL)
    assert row["status"] == "completed"
    assert row["metrics"]["mean_complete_request_ttft_ns"] is None and row["prefill_pass"]["status"] == "rejected"
    assert row["progress"]["cumulative_progress"]["visible_output_tokens"] == 2
    assert row["progress"]["wall_breakdown_seconds"] == {
        "starting": 2., "serving_boundary": 5., "serving_cohorts": 2.}
    writer.publish()
    assert run.read_json(writer.path)["cumulative_progress"]["visible_output_tokens"] == 2


def test_measurement_uses_start_exclusive_actual_tokens_and_complete_fresh_requests():
    metrics, diagnostics = run.measure_result(fake_result(), small_parameters())
    assert metrics["measurement_visible_output_tokens"] == 11
    assert metrics["complete_fresh_requests"] == 2
    assert metrics["cluster_tokens_per_second"] == 11e9/40
    assert metrics["tokens_per_second_per_gpu"] == 11e9/40/8
    assert metrics["mean_complete_request_tpot_ns"] == 5.
    # Ingress TTFT runs from each fresh request's paid import start.
    assert metrics["mean_complete_request_ingress_ttft_ns"] == 19.
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
    case = {**run.case_matrix()[-3], "gpu_count": 2, "input_tokens": 7, "output_tokens": 4}
    metrics, diagnostics = run.measure_result(result, run.parameters(case, run.make_layout(1, 2, 1, 1), 2))
    # Centered seed ages leave one fresh request complete inside this window.
    assert metrics["complete_fresh_requests"] >= 1 and diagnostics["slo_met"]
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
    result.serving.events = [event for event in result.serving.events
                             if event.event_type == "external_kv_import_start"]
    result.serving.events += [NS(event_type="tokens_committed", timestamp_ns=stamp,
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
    planned = [a["candidate_id"], b["candidate_id"]]
    selected = run.select_case(case, planned, [a, b])
    assert selected["status"] == "completed"
    assert selected["selected"] == a
    assert "not a global optimum" in selected["selection_scope"]
    b["status"] = "failed"
    b["error"] = {"type": "MemoryError", "message": "host out of memory"}
    rejected = run.select_case(case, planned, [a, b])
    assert rejected["status"] == "failed" and rejected["selected"] is None
    assert run.select_case(case, planned, [candidate(case, 1, "capacity_rejected"),
        candidate(case, 8, "slo_rejected")])["status"] == "all_candidates_rejected"


def test_unknown_and_host_memory_errors_are_not_device_capacity_rejections():
    from heterollm_sim.memory_allocator import AllocationError
    assert run.capacity_error(AllocationError("unable to allocate 1024 bytes in capacity 512"))
    assert not run.capacity_error(AllocationError("allocation overlaps another without an explicit alias"))
    assert not run.capacity_error(MemoryError("capacity exhausted in host memory"))
    assert not run.capacity_error(RuntimeError("unexpected bug mentioning capacity"))
    assert not run.capacity_error(ValueError("GDDR allocation 486127488 exceeds physical capacity 320000000"))


def test_real_physical_allocation_capacity_error_rejects_candidate_and_reduces_search_batch(tmp_path):
    from heterollm_sim.planner import ScenarioValidationError, _gddr_stable_address

    # This is the physical allocation branch the Maverick EP4 pilot reached;
    # classification must use its structured diagnostic, not its ValueError text.
    case = next(case for case in run.case_matrix()
                if case["case_id"] == "native__llama4_maverick__HBF_plus__SHORT__G8")
    layout = run.make_layout(2, 1, 1, 4)
    entry = {"layout": layout, "quantum": 4, "bypass": {
        "candidate_batch": 512, "capacity_max_batch": 8192,
        "tpot_ns": 80e6, "per_gpu_tokens_s": 100., "tpot_line_ns": (10e6, 1e5)}}
    initial = run.search_plan.plan_layout(entry, lambda batch: None, run.SLO_NS, 1800.)
    rejected_batch = initial["next"]

    def builder(*names, **options):
        _gddr_stable_address("pilot.lm_head.output", 0, 486127488, 320000000, 32,
            component_id="sram0", physical_kind="SRAM", physical_owner="sram0")

    row = run.execute_candidate(run.parameters(case, layout, rejected_batch, "fast"), tmp_path,
        NS(report=lambda progress: None), builder=builder)
    assert row["status"] == "capacity_rejected" and row["metrics"] is None
    assert row["error"]["type"] == ScenarioValidationError.__name__
    diagnostic, = row["error"]["details"]["diagnostics"]
    assert diagnostic["code"] == "physical_memory_capacity_exceeded"
    assert diagnostic["stage"] == "capacity_check"
    assert diagnostic["requested_bytes"] == 486127488
    assert diagnostic["capacity_bytes"] == 320000000
    assert (diagnostic["component_id"], diagnostic["physical_kind"], diagnostic["physical_owner"]) == (
        "sram0", "SRAM", "sram0")
    assert "SRAM" in row["error"]["message"] and "GDDR" not in row["error"]["message"]
    assert run.search_plan.classify(row) == "too_big"
    following = run.search_plan.plan_layout(entry,
        lambda batch: row if batch == rejected_batch else None, run.SLO_NS, 1800.)
    assert following["next"] == rejected_batch // 2

    # Invalid access geometry remains a correctness failure, even with enough
    # memory: it cannot be converted into a smaller-batch recommendation.
    def invalid_builder(*names, **options):
        _gddr_stable_address("pilot.lm_head.output", 16, 32, 320000000, 32, allocation_size=32)

    invalid = run.execute_candidate(run.parameters(case, layout, 4, "fast"), tmp_path,
        NS(report=lambda progress: None), builder=invalid_builder)
    assert invalid["status"] == "failed" and invalid["metrics"] is None
    assert invalid["error"]["type"] == "ValueError"


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
    run.refresh(tmp_path, [case], {case["case_id"]: [(LAYOUT, 1)]})
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
                       run.select_case(case, [row["candidate_id"]], [row]))
    assert run.run_manager(tmp_path, batches=[1], spawn=lambda *a: pytest.fail("terminal point rerun")) == 0
    saved = run.read_json(tmp_path / "status.json")
    assert saved["state"] == "completed_with_rejections"
    assert saved["counts"] == {"all_candidates_rejected": 1, "completed": 29}
    active = cases[0]
    (tmp_path / "points" / (active["case_id"] + ".json")).unlink()
    state = run.refresh(tmp_path, cases, {case["case_id"]: [(LAYOUT, 1)] for case in cases}, [{"case_id": active["case_id"],
        "candidate_id": run.candidate_id(active, LAYOUT, 1), "launcher_pid": 1}])
    assert state["counts"]["running"] == 1 and state["counts"]["completed"] == 29


def test_worker_reuses_an_exact_terminal_candidate_without_executing_native(tmp_path, monkeypatch):
    case = run.case_matrix()[0]
    rows = [candidate(case, batch) for batch in (1, 8)]
    for row in rows:
        run.write_json(tmp_path / "candidates" / (row["candidate_id"] + ".json"), row)
    monkeypatch.setattr(run, "execute_candidate", lambda *a, **k: pytest.fail("completed candidate rerun"))
    assert run.worker_candidate(case, LAYOUT, 8, tmp_path) == 0
    assert run.read_json(tmp_path / "progress" / (rows[1]["candidate_id"] + ".json"))["state"] == "completed"
    # The worker never writes the case point; the manager selects it.
    assert not (tmp_path / "points").exists()
    run.write_case_point(tmp_path, case, LAYOUT, [1, 8])
    saved = run.read_json(tmp_path / "points" / (case["case_id"] + ".json"))
    assert saved["status"] == "completed"
    assert saved["selected"]["parameters"]["batch_size"] == 1


def test_worker_reexecutes_one_retryable_fast_forecast_rejection(tmp_path, monkeypatch):
    case = run.case_matrix()[0]
    params = run.parameters(case, LAYOUT, 8, "fast")
    rejected = {"runner_contract": run.FAST_CONTRACT, "backend": run.BACKEND,
                "parameters": params, "candidate_id": run.params_candidate_id(params),
                "status": "forecast_rejected", "metrics": None,
                "rejection_reason": "native_pipeline_independent_budget_exceeded"}
    run.write_json(tmp_path / "candidates" / (rejected["candidate_id"] + ".json"), rejected)
    calls = []

    def execute(*args, **kwargs):
        calls.append(True)
        return {**rejected, "status": "completed", "metrics": {"tokens_per_second_per_gpu": 1.}}

    monkeypatch.setattr(run, "execute_candidate", execute)
    assert run.worker_candidate(case, LAYOUT, 8, tmp_path, mode="fast") == 0
    assert calls == [True]
    assert run.read_json(tmp_path / "candidates" / (rejected["candidate_id"] + ".json"))["status"] == "completed"


def test_search_reuses_layout_quality_evidence_after_two_real_rejections(tmp_path):
    case = run.case_matrix()[0]
    layout = run.make_layout(1, 4, 2, 1)
    for batch in (32, 64):
        params = run.parameters(case, layout, batch, "fast")
        run.save_candidate(tmp_path, {"runner_contract": run.FAST_CONTRACT, "backend": run.BACKEND,
            "parameters": params, "candidate_id": run.params_candidate_id(params),
            "status": "forecast_rejected", "metrics": None,
            "rejection_reason": "native_pipeline_independent_budget_exceeded"})
    reused = run._search_lookup(tmp_path, case, layout)(128)
    assert reused["status"] == "forecast_rejected"
    assert reused["rejection_reason"] == "native_pipeline_quality_evidence_reused"
    assert reused["wall_seconds"] == 0.
    assert run.read_candidate(tmp_path, reused["candidate_id"])["rejection_reason"] == reused["rejection_reason"]


def test_stop_marker_is_preserved_and_no_candidate_is_started(tmp_path):
    marker = tmp_path / "stop.request"
    marker.write_text("user paused", encoding="utf-8")
    assert run.run_manager(tmp_path, spawn=lambda *a: pytest.fail("stop ignored"), poll_seconds=0) == 0
    assert marker.read_text(encoding="utf-8") == "user paused"
    assert run.read_json(tmp_path / "status.json")["state"] == "paused"


def _manager_run(tmp_path, workers, mode="detailed", *, concurrent=None):
    cases = run.case_matrix()
    spawned, active = [], []
    concurrent = workers if concurrent is None else concurrent
    class Process:
        def __init__(self, case, layout, batch):
            self.case, self.layout, self.batch, self.pid = case, layout, batch, 123456 + len(spawned)
        def poll(self):
            active.remove(self.case["case_id"])
            row = candidate(self.case, self.batch, layout=self.layout)
            run.write_json(tmp_path / "candidates" / (row["candidate_id"] + ".json"), row)
            return 0
    def spawn(case, layout, batch, directory):
        assert len(active) < concurrent and layout == LAYOUT and batch == 1
        active.append(case["case_id"])
        spawned.append(case["case_id"])
        return Process(case, layout, batch)
    assert run.run_manager(tmp_path, workers=workers, batches=[1], spawn=spawn, poll_seconds=0,
                           mode=mode) == 0
    assert run.read_json(tmp_path / "status.json")["state"] == "completed"
    assert run.run_manager(tmp_path, workers=workers, batches=[1], mode=mode,
                           spawn=lambda *a: pytest.fail("terminal case rerun")) == 0
    return cases, spawned


def test_one_queue_keeps_phase_order_without_barriers_and_resumes_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 1 << 50)
    cases, spawned = _manager_run(tmp_path, 4)
    assert spawned == [row["case_id"] for row in cases]


def test_low_memory_admits_no_additional_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES-1)
    cases, spawned = _manager_run(tmp_path, 4, concurrent=1)
    assert len(spawned) == len(cases)


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


@pytest.mark.parametrize("workers", [0, run.MAX_WORKERS+1, True])
def test_parallel_worker_limit_is_explicit(tmp_path, workers):
    with pytest.raises(ValueError, match="workers must be an integer"):
        run.run_manager(tmp_path, workers=workers)


def test_progress_keeps_committed_counters_when_next_task_count_resets(tmp_path, monkeypatch):
    from heterollm_sim.execution_control import ExecutionProgress
    clock = NS(now=0.)
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock.now)
    params = small_parameters()
    writer = run.ProgressWriter(tmp_path, run.params_candidate_id(params))
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


# The approved scoped defaults require a separate fast cache contract.
def test_contracts_keep_detailed_v17_and_separate_scoped_ep_b4_fast_v21():
    assert run.CONTRACT.endswith("_v17") and run.FAST_CONTRACT.endswith("_v21")
    assert "tp8" not in run.CONTRACT and "tp8" not in run.FAST_CONTRACT
    assert run.parameters(run.case_matrix()[0], LAYOUT, 1)["runner_contract"] == run.CONTRACT
    assert run.DEFAULT_DIRECTORY.name != "exploring_hbf_native_20261019"
    assert run.DEFAULT_DIRECTORY.parent == run.RETIRED_DIRECTORIES[0].parent
    assert run.DEFAULT_DIRECTORY not in run.RETIRED_DIRECTORIES


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("legacy_contract", ["native_stationary_layout_fast_v17", "native_stationary_layout_fast_v18", "native_stationary_layout_fast_v19", "native_stationary_layout_fast_v20"])
def test_scoped_b4_does_not_resume_older_candidates_points_or_manifests(tmp_path, monkeypatch, search, legacy_contract):
    case = run.case_matrix()[0]
    params = run.parameters(case, LAYOUT, 1, "fast")
    legacy_params = {**params, "runner_contract": legacy_contract}
    identity = run.candidate_id(case, LAYOUT, 1)
    run.save_candidate(tmp_path, {"parameters": legacy_params, "candidate_id": identity,
        "backend": run.BACKEND, "runner_contract": legacy_contract, "status": "completed", "metrics": None})
    assert not run._saved_terminal(tmp_path, case, LAYOUT, 1, "fast")
    with pytest.raises(RuntimeError, match="saved candidate parameters/backend differ"):
        run._search_lookup(tmp_path, case, LAYOUT)(1)
    run.write_json(tmp_path / "points" / (case["case_id"] + ".json"), {
        "case": case, "backend": run.BACKEND, "runner_contract": legacy_contract,
        "status": "completed", "planned_candidate_ids": [identity]})
    with pytest.raises(RuntimeError, match="refusing historical reuse"):
        run._reusable_point(case, [identity], tmp_path, "fast")

    manifest = {"runner_contract": legacy_contract}
    run.write_json(tmp_path / "run_manifest.json", manifest)
    monkeypatch.setattr(run, "check_grid", lambda *args: None)
    never_spawn = lambda *args: pytest.fail("legacy result caused a worker to start")
    with pytest.raises(RuntimeError, match="manifest differs"):
        if search:
            run.run_search_manager(tmp_path, workers=1, case_ids=[case["case_id"]], spawn=never_spawn)
        else:
            run.run_manager(tmp_path, workers=1, batches=[1], mode="fast",
                            case_ids=[case["case_id"]], spawn=never_spawn)
    assert run.read_json(tmp_path / "run_manifest.json") == manifest
    assert not (tmp_path / "stop.request").exists()


def test_identity_and_parameters_carry_the_layout_and_must_fill_the_gpus():
    case = run.case_matrix()[-1]
    a, b = run.make_layout(1, 8, 1, 1), run.make_layout(2, 2, 1, 2)
    assert len({run.candidate_id(case, a, 8), run.candidate_id(case, b, 8), run.candidate_id(case, b, 4)}) == 3
    assert run.candidate_id(case, b, 4) == case["case_id"] + "__DP2_TP2_PP1_EP2__B4"
    params = run.parameters(case, b, 4)
    assert [params[key] for key in ("dp", "tp", "pp", "ep", "batch_size")] == [2, 2, 1, 2, 4]
    assert run.params_layout(params) == b and run.params_candidate_id(params) == run.candidate_id(case, b, 4)
    with pytest.raises(ValueError, match="exactly 8 GPUs"):
        run.parameters(case, run.make_layout(1, 4, 1, 1), 4)
    with pytest.raises(ValueError, match="positive"):
        run.make_layout(1, 8, 0, 1)


def test_layout_rejection_follows_the_model_graph(tmp_path):
    dense, maverick = run.case_matrix()[0], run.case_matrix()[-1]
    split = run.make_layout(1, 2, 1, 4)
    assert run.layout_rejection(dense, LAYOUT, 8) is None
    assert "MoE" in run.layout_rejection(dense, split, 8)
    assert run.layout_rejection(maverick, split, 8) is None
    assert "divisible by ep" in run.layout_rejection(maverick, split, 6)
    assert "exactly 8 GPUs" in run.layout_rejection(maverick, run.make_layout(1, 2, 1, 2), 8)
    # A grid with one unbuildable candidate starts nothing and opens no lock.
    with pytest.raises(ValueError, match="cannot be built"):
        run.run_manager(tmp_path, layout=split, batches=[8], spawn=lambda *a: pytest.fail("started"))
    assert not (tmp_path / "manager.lock").exists()


@pytest.mark.parametrize("layout,replica_gpus", [(run.make_layout(1, 8, 1, 1), 8), (run.make_layout(2, 4, 1, 1), 4),
    (run.make_layout(2, 2, 1, 2), 4), (run.make_layout(1, 2, 2, 2), 8), (run.make_layout(4, 1, 1, 2), 2)])
def test_throughput_is_replica_rate_over_replica_gpus_scaled_to_the_cluster(layout, replica_gpus):
    metrics, _ = run.measure_result(fake_result(), small_parameters(layout))
    replica = 11e9/40
    assert metrics["replica_tokens_per_second"] == replica
    assert metrics["tokens_per_second_per_gpu"] == pytest.approx(replica / replica_gpus)
    assert metrics["cluster_tokens_per_second"] == pytest.approx(replica * layout["dp"])
    assert metrics["cluster_tokens_per_second"] == pytest.approx(metrics["tokens_per_second_per_gpu"] * 8)


def test_selection_is_by_per_gpu_throughput_across_layouts():
    case = run.case_matrix()[-1]
    a = candidate(case, 1, layout=run.make_layout(1, 8, 1, 1))
    b = candidate(case, 8, layout=run.make_layout(2, 2, 1, 2))
    b["metrics"] = {**b["metrics"], "tokens_per_second_per_gpu": 140., "cluster_tokens_per_second": 1120.}
    point = run.select_case(case, [a["candidate_id"], b["candidate_id"]], [a, b])
    assert point["selected"] is b and point["planned_candidate_ids"] == [a["candidate_id"], b["candidate_id"]]
    # A larger cluster rate alone does not win when the per-GPU rate is lower.
    a["metrics"] = {**a["metrics"], "tokens_per_second_per_gpu": 100., "cluster_tokens_per_second": 9999.}
    assert run.select_case(case, [a["candidate_id"], b["candidate_id"]], [a, b])["selected"] is b


def test_builder_receives_the_candidate_layout_and_batch(tmp_path):
    case = run.case_matrix()[-1]
    layout = run.make_layout(2, 2, 1, 2)
    seen = {}
    def builder(*names, **options):
        seen.update(names=names, **options)
        raise RuntimeError("stop after construction")
    row = run.execute_candidate(run.parameters(case, layout, 4), tmp_path, NS(report=lambda p: None), builder=builder)
    assert seen == {"names": (case["model"], case["hardware"], case["workload"]), "batch_size": 4, "tp": 2, "pp": 1, "ep": 2}
    assert row["status"] == "failed" and row["candidate_id"] == run.candidate_id(case, layout, 4)


def test_placement_capacity_diagnostics_are_capacity_rejections_but_other_placement_failures_are_not(tmp_path):
    from heterollm_sim.planner import ScenarioValidationError
    capacity = {"code": "runtime_placement_unplaced", "stage": "capacity_check", "item_id": "dense0.mlp",
                "message_zh": "总容量约束排除了所有可行候选项", "message_en": "总容量约束排除了所有可行候选项"}
    error = ScenarioValidationError("运行时控制平面无法完成放置：总容量约束排除了所有可行候选项",
                                    [capacity, {**capacity, "item_id": "lm_head"}])
    assert run.capacity_error(error)
    other = {**capacity, "message_zh": "没有可用的放置候选项", "message_en": "没有可用的放置候选项"}
    assert not run.capacity_error(ScenarioValidationError("无法放置", [capacity, other]))
    assert not run.capacity_error(ScenarioValidationError("无法放置", [{**capacity, "stage": "model_check"}]))
    assert not run.capacity_error(ScenarioValidationError("无法放置", [{**capacity, "code": "runtime_placement_validation"}]))
    assert not run.capacity_error(ScenarioValidationError("无法放置"))
    assert run.capacity_error(ScenarioValidationError("x", [{"message_en": "KV demand exceeds available capacity"}]))
    case = run.case_matrix()[0]
    def builder(*names, **options):
        raise error
    row = run.execute_candidate(run.parameters(case, LAYOUT, 8), tmp_path, NS(report=lambda p: None), builder=builder)
    assert row["status"] == "capacity_rejected" and row["metrics"] is None


def test_each_candidate_gets_its_own_worker_command_that_main_parses_back(tmp_path, monkeypatch):
    case = run.case_matrix()[-1]
    layout = run.make_layout(2, 2, 1, 2)
    command = run.candidate_command(case, layout, 4, tmp_path, "fast", 3)
    assert Path(command[1]).name == "run_exploring_hbf_native.py"
    assert command[command.index("--layout") + 1:command.index("--layout") + 5] == ["2", "2", "1", "2"]
    seen = []
    monkeypatch.setattr(run, "worker_candidate", lambda *args: seen.append(args) or 0)
    assert run.main(command[2:]) == 0
    assert seen == [(case, layout, 4, tmp_path, "fast", 3)]


def test_manager_spawns_one_process_per_candidate_and_writes_the_per_gpu_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 1 << 50)
    cases, spawned = run.case_matrix(), []
    class Process:
        pid = 1
        def __init__(self, case, batch):
            self.case, self.batch = case, batch
        def poll(self):
            row = candidate(self.case, self.batch)
            run.write_json(tmp_path / "candidates" / (row["candidate_id"] + ".json"), row)
            return 0
    def spawn(case, layout, batch, directory):
        spawned.append((case["case_id"], batch))
        return Process(case, batch)
    assert run.run_manager(tmp_path, batches=[1, 8], workers=1, spawn=spawn, poll_seconds=0) == 0
    assert len(spawned) == 60 and spawned[:2] == [(cases[0]["case_id"], 1), (cases[0]["case_id"], 8)]
    point = run.read_json(tmp_path / "points" / (cases[0]["case_id"] + ".json"))
    assert point["status"] == "completed" and point["selected"]["parameters"]["batch_size"] == 1
    assert run.read_json(tmp_path / "run_manifest.json")["layout"] == LAYOUT
    import csv
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60 and {(r["dp"], r["tp"], r["pp"], r["ep"]) for r in rows} == {("1", "8", "1", "1")}
    with pytest.raises(RuntimeError, match="manifest differs"):
        run.run_manager(tmp_path, layout=run.make_layout(2, 4, 1, 1), batches=[1, 8],
                        spawn=lambda *a: pytest.fail("started"))


def test_the_paused_v15_scan_directory_is_never_opened(tmp_path, monkeypatch):
    retired = tmp_path / "paused"
    monkeypatch.setattr(run, "RETIRED_DIRECTORIES", (retired,))
    with pytest.raises(ValueError, match="paused v15 scan"):
        run.run_manager(retired)
    with pytest.raises(SystemExit):
        run.main(["--directory", str(retired)])
    with pytest.raises(ValueError, match="paused v15 scan"):
        run._spawn(run.case_matrix()[0], LAYOUT, 1, retired)
    assert not retired.exists()


def test_fast_forecast_is_scaled_from_one_replica_to_the_cluster():
    from dataclasses import replace
    from heterollm_sim.native_fast_window import NativeWindowForecast
    case = run.case_matrix()[-1]
    params = run.parameters(case, run.make_layout(2, 2, 1, 2), 4, "fast")
    bounds = {"cluster_tokens_per_second": {"estimate": 80., "lower": 70., "upper": 90., "relative_error_budget": .05},
              "mean_complete_request_tpot_ns": {"estimate": 95e6, "lower": 90e6, "upper": 99e6}}
    forecast = NativeWindowForecast(100., 0, 1111, 0, 1111, {}, {},
        {"cluster_tokens_per_second": 80., "tokens_per_second_per_gpu": 999., "mean_complete_request_tpot_ns": 95e6},
        bounds, (), {"accepted": True})
    row = run.native_fast_result(replace(_fast_result(), native_window_forecast=forecast), params)
    # One replica of 80 tokens/s on 4 GPUs is 20/GPU; two replicas fill the 8 GPUs.
    assert (row["metrics"]["replica_tokens_per_second"], row["metrics"]["tokens_per_second_per_gpu"],
            row["metrics"]["cluster_tokens_per_second"]) == (80., 20., 160.)
    scaled = row["diagnostics"]["metric_bounds"]
    assert scaled["cluster_tokens_per_second"] == {"estimate": 160., "lower": 140., "upper": 180., "relative_error_budget": .05}
    assert scaled["mean_complete_request_tpot_ns"] == bounds["mean_complete_request_tpot_ns"]
    assert row["replica_scaling"]["dp"] == 2 and row["replica_scaling"]["gpus_per_replica"] == 4
    assert row["native_window_forecast"]["metrics"]["cluster_tokens_per_second"] == 80.


def test_fast_wall_caps_are_absolute_across_concurrency(tmp_path, monkeypatch):
    import heterollm_sim.execution_control as execution_control
    clock = [0.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(execution_control, "perf_counter", lambda: clock[0])
    value = _fast_result()
    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    seen = []
    def runner(scenario, **options):
        seen.append((options["control"].deadline_monotonic, scenario.simulation.native_window.max_wall_seconds))
        clock[0] = 3001.
        return value
    row = run.execute_candidate(params, tmp_path, NS(report=lambda p: None), builder=lambda *a, **kw: value.scenario,
                                runner=runner, prefill=lambda *a: REJECTED_PREFILL, wall_factor=3)
    assert seen == [(3000., 3000.)]
    assert row["status"] == "forecast_rejected" and row["wall_factor"] == 3 and row["candidate_wall_budget_seconds"] == 3000.


def test_refill_rejection_reruns_the_batch_once_with_the_measured_pool(tmp_path, monkeypatch):
    import heterollm_sim.execution_control as execution_control
    clock = [0.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(execution_control, "perf_counter", lambda: clock[0])
    accepted = _fast_result()
    rejected = replace(_fast_result(accepted=False), approximation={"effective_mode": "forecast_rejected",
        "rejection_reasons": ["native_stationary_refill_not_always_ready"],
        "error_budget": {"import_pool_required": 6, "pd_import_span_ns": {"lower": 2e8, "upper": 3e8}}})
    pools = []

    def builder(*args, **options):
        pools.append(options.get("import_pool"))
        scenario = accepted.scenario
        return replace(scenario, workload=replace(scenario.workload, metadata={**scenario.workload.metadata,
            "external_kv_ingress": {"max_inflight_requests": options.get("import_pool") or 2}}))

    def runner(scenario, **options):
        clock[0] += 100. if len(pools) == 1 else 40.
        return rejected if len(pools) == 1 else accepted

    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    row = run.execute_candidate(params, tmp_path, NS(report=lambda p: None), builder=builder, runner=runner,
                                prefill=lambda *a: REJECTED_PREFILL)
    assert pools == [None, 6] and row["status"] == "completed" and "rejection_reason" not in row
    attempt, = row["native_import_pool_attempts"]
    assert (attempt["import_pool"], attempt["required_import_pool"], attempt["wall_seconds"]) == (2, 6, 100.)
    assert attempt["rejection_reason"] == "native_stationary_refill_not_always_ready"
    # The planner projects from the re-run alone.
    assert row["native_retry_wall_seconds"] == 40.
    assert run.search_plan._point(1, row)["native_wall_seconds"] == 40.


def test_prefill_replica_widens_tp_only_when_hbm4_cannot_hold_the_weights(tmp_path, monkeypatch):
    from heterollm_sim.planner import ScenarioValidationError
    weights = {"code": "runtime_placement_validation", "stage": "capacity_check", "item_id": "weights",
               "message_en": "remaining resident model weights require 235904434176 bytes across physical "
                             "TP/PP/EP placements, but hbm0, hbm1 provide 205946880 bytes"}
    assert run.capacity_error(ScenarioValidationError("无法放置", [weights]))
    fits = {1: False, 2: False, 4: True}
    import heterollm_sim.control_plane as control_plane
    import exploring_hbf_native_config as config
    monkeypatch.setattr(config, "build_native_prefill_scenario", lambda model, workload, *, tp, **_: tp)
    monkeypatch.setattr(control_plane, "bootstrap_control_plane",
                        lambda tp: None if fits[tp] else (_ for _ in ()).throw(ScenarioValidationError("x", [weights])))
    run.prefill_tp.cache_clear()
    try:
        assert run.prefill_tp("m", "SHORT", 2, 1, 1) == 4 and run.prefill_tp("m", "SHORT", 4, 1, 1) == 4
        fits[4] = fits[8] = False
        run.prefill_tp.cache_clear()
        assert run.prefill_tp("m", "SHORT", 2, 1, 1) is None
        case = {**run.case_matrix()[6], "model": "m"}
        monkeypatch.setattr(run, "_model_layers", lambda model: ())
        monkeypatch.setattr(config, "validate_native_layout", lambda *args: None)
        layout = run.make_layout(4, 2, 1, 1)
        assert run.layout_rejection(case, layout, 8) == "native_prefill_replica_does_not_fit_hbm4"
    finally:
        run.prefill_tp.cache_clear()
    monkeypatch.setattr(run, "prefill_tp", lambda model, workload, tp, pp, ep: 4)
    params = run.parameters(run.case_matrix()[6], run.make_layout(4, 2, 1, 1), 8, "fast")
    expected = run.prefill_parameters(params)
    assert expected["tp"] == 4 and "decode_tp" not in expected
    assert run.prefill_id(expected).endswith("TP4PP1EP1__sampled")
    row = {"metrics": {"mean_complete_request_ingress_ttft_ns": 42e6}}
    run.with_prefill_ttft(row, run.parameters(run.case_matrix()[6], run.make_layout(4, 2, 1, 1), 8),
                          {**_prefill(99e6, 101e6, observed=100e6), "parameters": expected})
    assert row["prefill_pass"]["layout"] == {"tp": 4, "pp": 1, "ep": 1, "decode_tp": 2,
                                              "tp_widened_for_hbm4_weights": True}


@pytest.mark.parametrize("mode", ["fast", "detailed"])
def test_widened_decode_layouts_share_prefill_without_reusing_decode_provenance(tmp_path, monkeypatch, mode):
    from heterollm_sim.native_stationary import _interval

    monkeypatch.setattr(run, "prefill_tp", lambda model, workload, tp, pp, ep: 4)
    case = next(row for row in run.case_matrix() if
        (row["model"], row["hardware"], row["workload"]) == ("llama4_maverick", "HBF", "SHORT"))
    runs = []
    expected_full = mode == "detailed"
    pass_bounds, ingress_bounds = _interval(99e6, 101e6), _interval(40e6, 44e6)

    def runner(scenario, **options):
        runs.append(scenario)
        assert scenario.simulation.native_prefill.full_pass is expected_full
        return NS(approximation={"prefill_pass": {"accepted": True,
            "metric_bounds": {"prefill_pass_ns": pass_bounds},
            "error_budget": {"observed_pass_ns": 100e6}},
            "rejection_reasons": []}, layer_window_samples=())

    saved = None
    for tp in (1, 2, 4, 1):
        params = run.parameters(case, run.make_layout(8 // tp, tp, 1, 1), 8, mode)
        prefill = run.prefill_pass(params, tmp_path,
            builder=lambda *a, **k: _fast_result().scenario, runner=runner)
        if saved is not None:
            assert prefill == saved
        saved = prefill
        assert "decode_tp" not in prefill["parameters"]
        row = {"metrics": {"mean_complete_request_ingress_ttft_ns": 42e6}}
        if mode == "fast":
            row["metrics"]["mean_complete_request_ingress_ttft_ns"] = ingress_bounds["estimate"]
            row["diagnostics"] = {"metric_bounds": {
                "mean_complete_request_ingress_ttft_ns": dict(ingress_bounds)}}
        run.with_prefill_ttft(row, params, prefill)
        assert row["prefill_pass"]["layout"] == {
            "tp": 4, "pp": 1, "ep": 1, "decode_tp": tp,
            "tp_widened_for_hbm4_weights": tp != 4}
        if mode == "fast":
            total = _interval(139e6, 145e6)
            assert row["diagnostics"]["metric_bounds"]["prefill_pass_ns"] == pass_bounds
            assert row["diagnostics"]["metric_bounds"]["mean_complete_request_ttft_ns"] == total
            assert row["metrics"]["prefill_pass_ns"] == pass_bounds["estimate"]
            assert row["metrics"]["mean_complete_request_ttft_ns"] == total["estimate"]
        else:
            assert row["metrics"]["prefill_pass_ns"] == 100e6
            assert row["metrics"]["mean_complete_request_ttft_ns"] == 142e6
    assert len(runs) == 1
    # PP and EP remain physical Prefill identity, even when TP widens alike.
    identities = {saved["prefill_id"]}
    for pp, ep in ((2, 1), (1, 2)):
        expected_full = mode == "detailed"
        params = run.parameters(case, run.make_layout(8 // (pp * ep), 1, pp, ep), 8, mode)
        prefill = run.prefill_pass(params, tmp_path,
            builder=lambda *a, **k: _fast_result().scenario, runner=runner)
        assert (prefill["parameters"]["tp"], prefill["parameters"]["pp"], prefill["parameters"]["ep"]) == (4, pp, ep)
        assert prefill["prefill_id"] not in identities
        identities.add(prefill["prefill_id"])
    assert len(runs) == 3


def test_a_prefill_placement_failure_ends_the_layout_not_the_scan(tmp_path):
    from heterollm_sim.planner import ScenarioValidationError
    weights = {"code": "runtime_placement_validation", "stage": "capacity_check", "item_id": "weights",
               "message_en": "remaining resident model weights require 2 bytes across placements, but hbm0 provide 1 bytes"}

    def prefill(*args):
        raise ScenarioValidationError("无法放置", [weights])

    params = run.parameters(run.case_matrix()[0], LAYOUT, 1, "fast")
    row = run.execute_candidate(params, tmp_path, NS(report=lambda p: None),
                                builder=lambda *a, **k: _fast_result().scenario,
                                runner=lambda *a, **k: _fast_result(), prefill=prefill)
    assert row["status"] == "forecast_rejected" and row["metrics"] is None
    assert row["rejection_reason"] == "native_prefill_placement_infeasible"
    assert run.search_plan.classify(row) == "unusable"


def _job(directory, identity, *, factor, peak, working_set=None):
    if working_set is not None:
        run.write_json(directory / "progress" / (identity + ".json"),
                       {"candidate_id": identity, "working_set_bytes": working_set})
    return {"process": NS(pid=1), "factor": factor, "peak": peak}


def test_admission_fills_predicted_memory_and_uses_the_run_worker_limit(tmp_path, monkeypatch):
    GiB = 2**30
    case = run.case_matrix()[0]
    queue = [(case, LAYOUT, batch) for batch in (100, 101, 102, 103)]
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 8 * GiB + 40 * GiB)
    # Without measured peaks each candidate counts DEFAULT_CANDIDATE_PEAK_BYTES (16 GiB).
    chosen, factor = run.admit(queue, {}, [], 32, tmp_path)
    assert [item for item, _peak in chosen] == queue[:2] and factor == 32
    # Nothing running: one candidate always starts, however large.
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 0)
    assert run.admit(queue, {}, [], 32, tmp_path) == ([(queue[0], 16 * GiB)], 32)
    # A running candidate's remaining predicted growth is reserved; what it
    # already uses is already missing from the available memory.
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 8 * GiB + 20 * GiB)
    known = [(run.parameters(case, LAYOUT, 100, "fast"), 4 * GiB)]
    active = {"a": _job(tmp_path, "a", factor=4, peak=10 * GiB, working_set=8 * GiB)}
    chosen, factor = run.admit(queue, active, known, 32, tmp_path)
    assert len(chosen) == 4 and factor == 32
    active["b"] = _job(tmp_path, "b", factor=2, peak=1)
    chosen, factor = run.admit(queue, active, known, 32, tmp_path)
    assert len(chosen) == 4 and factor == 32
    assert run.admit(queue, {"a": active["a"]}, known, 1, tmp_path) == ([], 1)


def test_peak_prediction_prefers_the_same_layout_and_is_linear_in_the_batch():
    GiB = 2**30
    case, other = run.case_matrix()[0], run.make_layout(2, 4, 1, 1)
    known = [(run.parameters(case, LAYOUT, 100, "fast"), 6 * GiB), (run.parameters(case, LAYOUT, 300, "fast"), 8 * GiB),
             (run.parameters(case, other, 100, "fast"), 3 * GiB)]
    assert run.predicted_peak_bytes(known, case, LAYOUT, 500) == pytest.approx(10 * GiB)
    assert run.predicted_peak_bytes(known, case, LAYOUT, 1) == 6 * GiB - .99 * GiB
    assert run.predicted_peak_bytes(known[2:], case, other, 200) == 6 * GiB
    foreign = run.case_matrix()[-1]
    assert run.predicted_peak_bytes(known, foreign, LAYOUT, 1) == run.DEFAULT_CANDIDATE_PEAK_BYTES


def test_summary_rows_drop_the_physical_ledgers_and_follow_newer_full_rows(tmp_path):
    import os
    row = {**candidate(run.case_matrix()[0]), "physical_layer_samples": [{"executed_task_count": 7}] * 2,
           "physical_boundary_accounting": {"ledger": [1] * 100},
           "approximation": {"metric_bounds": {"x": 1}, "physical_boundary_accounting": {"ledger": [1]}}}
    run.save_candidate(tmp_path, row)
    summary = run.read_candidate(tmp_path, row["candidate_id"])
    assert "physical_layer_samples" not in summary and "physical_boundary_accounting" not in summary
    assert summary["approximation"] == {"metric_bounds": {"x": 1}}
    assert summary["physical_layer_summary"] == {"completed_windows": 2, "executed_tasks": 14,
                                                 "complete_cohorts": 0, "visible_output_tokens": 0}
    full = tmp_path / "candidates" / (row["candidate_id"] + ".json")
    run.write_json(full, {**row, "status": "failed"})
    stamp = (tmp_path / "summaries" / full.name).stat().st_mtime_ns
    os.utime(full, ns=(stamp + 10**9, stamp + 10**9))
    assert run.read_candidate(tmp_path, row["candidate_id"])["status"] == "failed"
    assert run.read_candidate(tmp_path, "missing") is None


def test_capacity_rejection_records_the_largest_batch_the_native_kv_holds():
    params = run.parameters(run.case_matrix()[0], LAYOUT, 353, "fast")
    scenario = NS(workload=NS(metadata={"external_kv_ingress": {"max_inflight_requests": 2}}))
    held = lambda counts: {"approximation": {"kv_requests_held_by_group": counts}}
    # 405B SHORT: 373 outputs, two pool requests per freed slot: 338 seeds + 2 imports fill 340.
    assert run.native_capacity_max_batch(held([340]), params, scenario) == 338
    assert run.native_capacity_max_batch({"approximation": {}}, params, scenario) is None
    assert run.native_capacity_max_batch(held([400]), params, scenario) is None


@pytest.mark.parametrize("hardware,workload,expected", [
    ("HBM4", "SHORT", "all"), ("HBM4", "MID", "quiet"), ("HBF", "SHORT", "quiet"),
])
def test_fast_runner_uses_and_records_the_shared_scenario_options(tmp_path, hardware, workload, expected):
    from dataclasses import asdict
    from exploring_hbf_native_config import native_fast_window_options

    case = next(row for row in run.case_matrix() if
        (row["model"], row["hardware"], row["workload"]) == ("llama3_405b", hardware, workload))
    params = run.parameters(case, LAYOUT, 16, "fast")
    rejected = _fast_result(accepted=False)
    seen = []

    def runner(scenario, **kwargs):
        options = scenario.simulation.native_window
        seen.append(options)
        assert options == native_fast_window_options(params["model"], hardware, workload,
            tp=params["tp"], pp=params["pp"], ep=params["ep"], max_wall_seconds=options.max_wall_seconds)
        assert options.tp_rank_symmetry == expected
        return replace(rejected, scenario=scenario, approximation={**rejected.approximation,
            "sampling_options": asdict(options)})

    row = run.execute_candidate(params, tmp_path, NS(report=lambda p: None),
        builder=lambda *args, **kwargs: rejected.scenario, runner=runner)
    assert len(seen) == 1
    assert row["status"] == "forecast_rejected"
    assert row["sampling_options"] == asdict(seen[0])
    assert row["approximation"]["sampling_options"] == row["sampling_options"]
    saved = run.read_json(tmp_path / "configurations" / (row["candidate_id"] + ".json"))
    assert saved["simulation"]["native_window"] == row["sampling_options"]
