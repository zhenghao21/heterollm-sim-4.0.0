"""Native window samples never become a fabricated complete physical run."""
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import sys

import pytest

from heterollm_sim import native_fast_window as native
from heterollm_sim import fast_simulation
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim import execution_control as control_module
from heterollm_sim.simulation_options import (NativeWindowFastConfig, SimulationConfig,
                                             simulation_from_dict)


def harness(monkeypatch, budgets=None):
    budgets = budgets or [{"accepted": True}]
    options = NativeWindowFastConfig()
    request = NS(request_id="r0", prompt_tokens=1330, output_tokens=373)
    scenario = NS(simulation=SimulationConfig(mode="fast", native_window=options),
        workload=NS(metadata={"external_kv_ingress": {
            "warmup_decode_cohorts": 372, "measurement_decode_cohorts": 744,
            "require_full_batch": True}}))
    plan = NS(requests=(request,), mtp=NS(enabled=False),
        scheduler=NS(preemption_enabled=False), kv_policy=NS(offload_component=None, prefetch_distance=0))
    runtimes, attempts = [], []

    class Runtime:
        def __init__(self, plan, lowerer, **kwargs):
            self.plan, self.control = plan, kwargs["execution_control"]
            self.now = self._runtime_origin_ns = kwargs["runtime_origin_ns"]
            self._completed_decode_steps, self.events, self.batches = 0, [], []
            self._decode_window = {"complete": False}
            self.run_calls = 0
            runtimes.append(self)

        def run_until_decode_steps(self, target, *, strict_cohort_cutoff=False):
            assert strict_cohort_cutoff
            while self._completed_decode_steps < target:
                self.control.raise_if_cancelled()
                index = self._completed_decode_steps
                start = self.now + (10. if index == 0 else 0.)
                self.now = start + 100.
                item = NS(request_id="r0", context_tokens=1331 + index,
                          committed_tokens=1, phase="decode", token_count=1)
                self.batches.append(NS(cohort_id="c" + str(index), kind="decode", start_ns=start,
                    end_ns=self.now, request_ids=("r0",), items=(item,),
                    cost=NS(metadata={"task_count": 11})))
                self.events.append(NS(cohort_id="c" + str(index), request_id="r0", event_type="tokens_committed",
                    details={"visible_tokens": 1}, timestamp_ns=self.now))
                self._completed_decode_steps += 1
            return False

        def _result(self):
            return NS(request_states={"r0": NS(status="running")},
                      decode_measurement={"complete": False}, events=tuple(self.events),
                      batches=tuple(self.batches))

        def run(self):
            self.run_calls += 1
            pytest.fail("native forecast rejection cannot run a detailed suffix")

    def budget(runtime, records, options, **kwargs):
        attempts.append((runtime._completed_decode_steps, records))
        result = dict(budgets[min(len(attempts) - 1, len(budgets) - 1)])
        if result["accepted"]:
            result.update(metrics={"tokens_per_second_per_gpu": 10.,
                "cluster_tokens_per_second": 80., "mean_complete_request_tpot_ns": 1.e8,
                "mean_complete_request_ingress_ttft_ns": 2.e8, "measurement_duration_ns": 1.e9,
                "complete_fresh_requests": 2, "measurement_visible_output_tokens": 80},
                metric_bounds={"tpot": [1.e8, 1.e8]},
                error_budget={"accepted": True}, window={"scope": "forecast_only"},
                request_forecasts=(), predicted_visible_output_tokens=1111)
        return result

    monkeypatch.setattr(native, "_compilation_scope", lambda *_args: nullcontext())
    monkeypatch.setattr(native, "compile_serving_plan", lambda _scenario: plan)
    monkeypatch.setattr(native, "_OnlineRuntime", Runtime)
    monkeypatch.setitem(sys.modules, "heterollm_sim.native_window_budget", NS(evaluate_native_window=budget))
    return scenario, plan, runtimes, attempts


def test_native_dispatch_samples_full_cohorts_and_keeps_cutoff_state(monkeypatch):
    scenario, plan, runtimes, attempts = harness(monkeypatch)
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    runtime, = runtimes
    assert runtime._completed_decode_steps == 5 and runtime.run_calls == 0
    assert plan.requests[0].output_tokens == 373
    assert outcome.detailed is outcome.forecast is None
    assert outcome.sampled.decode_measurement == {"complete": False}
    forecast = outcome.native_window_forecast
    assert forecast.sampled_decode_cohorts == 5
    assert forecast.predicted_decode_cohorts == 1111
    assert forecast.cutoff_ns == runtime.now == 510.
    assert forecast.sampled_visible_output_tokens == 5
    assert forecast.predicted_visible_output_tokens == 1111
    assert forecast.to_metadata()["forecast_has_physical_trace"] is False
    assert attempts[0][1][0]["boundary_gap_ns"] == 10.
    assert attempts[0][1][0]["commit_interval_ns"] == 110.
    assert attempts[0][1][1]["commit_interval_ns"] == 100.
    assert not outcome.diagnostics["future_physical_state_advanced"]


def test_cohort_sampling_forwards_native_queue_stop_and_total_deadline(monkeypatch, tmp_path):
    from queue import SimpleQueue
    scenario, _, runtimes, _ = harness(monkeypatch)
    progress = SimpleQueue()
    control = ExecutionControl(progress_queue=progress, stop_request_path=tmp_path / "stop.request",
        deadline_monotonic=native.time.perf_counter() + 600.)
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object(), execution_control=control)
    sampled = runtimes[0].control
    assert control_module.native_callback_free_control(sampled)
    assert sampled.progress_queue is progress
    assert sampled.stop_request_path == control.stop_request_path
    assert sampled.deadline_monotonic <= control.deadline_monotonic
    assert sampled.cancellation_callback is None
    assert outcome.native_window_forecast is not None
    assert progress.get_nowait().stage == "native_fast_sampling"


def test_rejected_budget_preserves_real_sample_and_never_falls_back(monkeypatch):
    scenario, _, runtimes, _ = harness(monkeypatch, [{"accepted": False,
        "reasons": ["future_pd_interference_not_bounded"],
        "error_budget": {"future_pd_ns": None}}])
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    assert runtimes[0].run_calls == 0 and runtimes[0]._completed_decode_steps == 5
    assert outcome.native_window_forecast is outcome.detailed is None
    assert outcome.sampled.request_states["r0"].status == "running"
    assert outcome.diagnostics["effective_mode"] == "forecast_rejected"
    assert outcome.diagnostics["error_budget"]["future_pd_ns"] is None
    assert outcome.diagnostics["rejection_reasons"] == ["future_pd_interference_not_bounded"]


def test_budget_can_request_more_real_samples_without_rewinding(monkeypatch):
    scenario, _, runtimes, attempts = harness(monkeypatch, [
        {"accepted": False, "can_expand_sampling": True, "reasons": ["holdout_drift"]},
        {"accepted": True}])
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    assert [count for count, _ in attempts] == [5, 7]
    assert runtimes[0].run_calls == 0
    assert outcome.native_window_forecast.sampled_decode_cohorts == 7
    assert attempts[1][1][:5] == attempts[0][1]


def test_unknown_lowerer_rejects_before_physical_sampling(monkeypatch):
    scenario, _, runtimes, attempts = harness(monkeypatch)
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object(), custom_lowerer=True)
    assert not runtimes and not attempts
    assert outcome.sampled is None and outcome.native_window_forecast is None
    assert outcome.diagnostics["rejection_reasons"] == ["native_window_requires_standard_native_lowerer"]


def test_wall_limit_rejection_does_not_expose_a_partial_cohort_as_sample(monkeypatch):
    scenario, _, runtimes, _ = harness(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(control_module, "perf_counter", lambda: clock[0])
    original = native._OnlineRuntime.run_until_decode_steps
    def over_budget(self, *args, **kwargs):
        clock[0] = 1801.
        return original(self, *args, **kwargs)
    monkeypatch.setattr(native._OnlineRuntime, "run_until_decode_steps", over_budget)
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    assert runtimes[0].run_calls == runtimes[0]._completed_decode_steps == 0
    assert outcome.sampled is outcome.native_window_forecast is None
    assert outcome.diagnostics["partial_physical_cohort_discarded"] is True
    assert outcome.diagnostics["rejection_reasons"] == ["native_sample_wall_budget_exhausted"]


def test_user_cancel_propagates_instead_of_becoming_forecast_rejection(monkeypatch):
    scenario, _, _, _ = harness(monkeypatch)
    with pytest.raises(ExecutionCancelledError):
        fast_simulation.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(cancellation_callback=lambda: True))


def test_user_pause_during_projection_wins_over_an_accepted_budget(monkeypatch):
    scenario, _, runtimes, _ = harness(monkeypatch)
    cancelled = [False]
    budget_module = sys.modules["heterollm_sim.native_window_budget"]
    original = budget_module.evaluate_native_window
    def pause(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled[0] = True
        return result
    budget_module.evaluate_native_window = pause
    with pytest.raises(ExecutionCancelledError):
        fast_simulation.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(cancellation_callback=lambda: cancelled[0]))
    assert runtimes[0]._completed_decode_steps == 5 and runtimes[0].run_calls == 0


def test_result_materialization_time_is_included_in_wall_budget(monkeypatch):
    scenario, _, runtimes, _ = harness(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(control_module, "perf_counter", lambda: clock[0])
    original = native._OnlineRuntime._result

    def slow_result(self):
        clock[0] = 1801.
        return original(self)

    monkeypatch.setattr(native._OnlineRuntime, "_result", slow_result)
    outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    assert outcome.native_window_forecast is None
    assert outcome.sampled is not None and runtimes[0].run_calls == 0
    assert outcome.diagnostics["rejection_reasons"] == ["native_result_wall_budget_exhausted"]


@pytest.mark.parametrize("pause", (False, True))
def test_completion_observer_cannot_bypass_pause_or_wall_budget(monkeypatch, pause):
    scenario, _, runtimes, _ = harness(monkeypatch)
    clock, cancelled = [0.], [False]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(control_module, "perf_counter", lambda: clock[0])

    def observer(progress):
        if progress.stage == "native_fast_forecast_complete":
            if pause:
                cancelled[0] = True
            else:
                clock[0] = 1801.

    control = ExecutionControl(progress_callback=observer,
        cancellation_callback=lambda: cancelled[0])
    if pause:
        with pytest.raises(ExecutionCancelledError):
            fast_simulation.simulate_fast(scenario, batch_lowerer=object(), execution_control=control)
    else:
        outcome = fast_simulation.simulate_fast(scenario, batch_lowerer=object(), execution_control=control)
        assert outcome.native_window_forecast is None and outcome.sampled is not None
        assert outcome.diagnostics["effective_mode"] == "forecast_rejected"
        assert outcome.diagnostics["rejection_reasons"] == ["native_publication_wall_budget_exhausted"]
    assert runtimes[0].run_calls == 0


def test_sample_commits_retain_token_timestamp_before_cohort_cleanup(monkeypatch):
    scenario, _, runtimes, _ = harness(monkeypatch)
    fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    runtime, = runtimes
    runtime.events[0].timestamp_ns -= 7.
    record = native.cohort_samples(runtime)[0]
    assert record["end_ns"] == 110.
    assert record["visible_commits"][0]["timestamp_ns"] == 103.
    assert record["visible_output_tokens"] == 1


def test_budget_samples_do_not_duplicate_per_resource_physical_ledger(monkeypatch):
    scenario, _, runtimes, _ = harness(monkeypatch)
    fast_simulation.simulate_fast(scenario, batch_lowerer=object())
    runtime, = runtimes
    totals = {"bank0": {"bytes_moved": 64}, "bank1": {"bytes_moved": 128}}
    runtime.batches[0].cost.metadata["dram_traffic"] = {"logical_bytes": 191,
        "physical_bytes": 192, "resource_totals": totals, "refresh_wait_ns": None}
    records = native.cohort_samples(runtime)
    assert records[0]["physical_observations"]["dram_traffic"] == {
        "logical_bytes": 191, "physical_bytes": 192, "refresh_wait_ns": None}
    assert runtime.batches[0].cost.metadata["dram_traffic"]["resource_totals"] is totals


def test_native_window_configuration_roundtrip_keeps_original_schema_strict():
    value = SimulationConfig(mode="fast", native_window=NativeWindowFastConfig())
    assert simulation_from_dict(asdict(value)) == value
    assert simulation_from_dict({}).native_window is None
    with pytest.raises(ValueError, match="unknown simulation.native_window"):
        simulation_from_dict({"mode": "fast", "native_window": {"sample_tokens": 5}})
    with pytest.raises(ValueError, match="requires fast mode"):
        SimulationConfig(native_window=NativeWindowFastConfig())


def test_layer_window_configuration_is_explicit_and_preserves_disjoint_sample_roles():
    options = NativeWindowFastConfig(sampling_unit="layer_windows", layer_window_size=2,
        calibration_layer_windows=3, validation_layer_windows=2, max_sample_layer_windows=6)
    assert options.sample_layer_windows == 5
    value = SimulationConfig(mode="fast", native_window=options)
    assert simulation_from_dict(asdict(value)) == value
    assert NativeWindowFastConfig().sampling_unit == "cohorts"


@pytest.mark.parametrize("fields", [{"sampling_unit": "tokens"}, {"sampling_unit": []},
    {"layer_window_size": 0}, {"layer_window_size": True}, {"validation_layer_windows": 1},
    {"calibration_layer_windows": 1}, {"max_sample_layer_windows": 3},
    {"max_seed_anchor_layer_windows": 3}, {"max_seed_anchor_layer_windows": True},
    {"calibration_layer_windows": 3, "validation_layer_windows": 3, "max_sample_layer_windows": 5}])
def test_layer_window_configuration_cannot_remove_holdout_or_exceed_sample_cap(fields):
    with pytest.raises(ValueError):
        NativeWindowFastConfig(**fields)


@pytest.mark.parametrize("fields", [{"max_wall_seconds": 1801}, {"target_relative_error": .10000000000000002},
    {"calibration_cohorts": 1}, {"validation_cohorts": True}, {"max_sample_cohorts": 4}])
def test_native_window_configuration_rejects_weakened_or_incomplete_budgets(fields):
    with pytest.raises(ValueError):
        NativeWindowFastConfig(**fields)


def test_real_native_samples_include_paid_pd_gap_and_actual_prefill_first_tokens():
    from test_native_external_kv import ingress_scenario, native_runtime

    runtime = native_runtime(ingress_scenario(warmup=3, measured=6, count=4))
    runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
    records = native.cohort_samples(runtime)
    assert sum(row["kind"] == "decode" for row in records) == 1
    visible = sum(event.details.get("visible_tokens", 0) for event in runtime.events
                  if event.event_type == "tokens_committed")
    assert sum(row["visible_output_tokens"] for row in records) == visible
    assert any(row["kind"] == "prefill" and row["visible_output_tokens"] > 0 for row in records)
    imported = [event.timestamp_ns for event in runtime.events
                if event.event_type == "external_kv_import_complete"]
    assert imported and records[0]["end_ns"] >= max(imported) > runtime._runtime_origin_ns
    assert runtime._external_executor.counters["read_bytes"] > 0
    assert sum(row["commit_interval_ns"] for row in records) == pytest.approx(
        runtime.batches[-1].end_ns - runtime._runtime_origin_ns)
    assert not runtime._result().decode_measurement["complete"]


def test_native_fast_rejection_matches_the_same_real_prefix_without_completing_it(monkeypatch):
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.planner import TopologyAwareBatchCostProvider
    from heterollm_sim.serving import _OnlineRuntime, compile_serving_plan
    from test_native_external_kv import ingress_scenario

    options = NativeWindowFastConfig(warmup_cohorts=0, max_sample_cohorts=4)
    authored = ingress_scenario("hbm_tp2", warmup=3, measured=6, count=8)
    scenario = replace(authored, simulation=SimulationConfig(mode="fast", native_window=options),
        workload=replace(authored.workload, scheduler=replace(authored.workload.scheduler,
            preemption_enabled=False)))

    def prepared():
        bootstrap = bootstrap_control_plane(scenario, capture_physical_details=False)
        return bootstrap, TopologyAwareBatchCostProvider(bootstrap.scenario,
                                                         defer_physical_execution=True)

    bootstrap, lowerer = prepared()
    expected = _OnlineRuntime(compile_serving_plan(bootstrap.scenario), lowerer,
        execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
    expected.run_until_decode_steps(4, strict_cohort_cutoff=True)
    expected_result = expected._result()
    bootstrap, lowerer = prepared()
    monkeypatch.setitem(sys.modules, "heterollm_sim.native_window_budget", NS(
        evaluate_native_window=lambda *_args, **_kwargs: {"accepted": False,
            "reasons": ["unproven_future_phase"]}))
    outcome = fast_simulation.simulate_fast(bootstrap.scenario, batch_lowerer=lowerer,
        execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
    assert outcome.detailed is outcome.forecast is outcome.native_window_forecast is None
    assert outcome.sampled.events == expected_result.events
    assert outcome.sampled.request_states == expected_result.request_states
    assert outcome.sampled.request_metrics == expected_result.request_metrics
    assert outcome.sampled.external_kv_ingress_metrics == expected_result.external_kv_ingress_metrics
    assert [(row.start_ns, row.end_ns) for row in outcome.sampled.batches] == [
        (row.start_ns, row.end_ns) for row in expected_result.batches]
    assert outcome.diagnostics["sampled_decode_cohorts"] == 4
    assert not outcome.sampled.decode_measurement["complete"]


def compute_bound_native_scenario():
    """A new synthetic TP2 fixture, not a commercial-device performance claim.

    Its slow GPU makes the conservative phase bounds smaller than its
    constant native compute path. The ordinary fast GPU fixture correctly
    rejects when row/refresh/PD uncertainty consumes its timing budget.
    """
    from heterollm_sim.control_plane_planner import _derive_requirements
    from heterollm_sim.ir import RequestSpec
    from test_native_external_kv import ingress_scenario

    scenario = ingress_scenario("hbm_tp2", warmup=7, measured=14, count=4)
    profile = scenario.component_profiles["gpu"]["legacy-gpu"]
    profile = replace(profile, tensor_core=replace(profile.tensor_core, frequency_ghz=.0001))
    targets = {item.mapping_key: "gpu0" for item in _derive_requirements(scenario)
               if item.mapping_key and not item.state_tensor}
    return replace(scenario,
        component_profiles={**scenario.component_profiles, "gpu": {"legacy-gpu": profile}},
        placement=replace(scenario.placement, metadata={**scenario.placement.metadata,
            "control_plane": {"policy": {"options": {"operator_targets": targets}}}}),
        workload=replace(scenario.workload,
            requests=tuple(RequestSpec(f"r{index}", 0., 128, 8) for index in range(4)),
            scheduler=replace(scenario.workload.scheduler, max_num_seqs=1,
                preemption_enabled=False, mixed_phase_batching=False)))


def test_real_tp2_window_keeps_native_prefix_and_respects_current_phase_budget():
    from statistics import fmean
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.planner import TopologyAwareBatchCostProvider
    from heterollm_sim.serving import _OnlineRuntime, compile_serving_plan

    authored = compute_bound_native_scenario()
    fast = replace(authored, simulation=SimulationConfig(mode="fast", native_window=NativeWindowFastConfig()))
    bootstrap = bootstrap_control_plane(fast, capture_physical_details=False)
    outcome = fast_simulation.simulate_fast(bootstrap.scenario,
        batch_lowerer=TopologyAwareBatchCostProvider(bootstrap.scenario, defer_physical_execution=True),
        execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
    forecast = outcome.native_window_forecast
    assert outcome.detailed is outcome.forecast is None
    assert not outcome.sampled.decode_measurement["complete"]

    bootstrap = bootstrap_control_plane(authored, capture_physical_details=False)
    runtime = _OnlineRuntime(compile_serving_plan(bootstrap.scenario),
        TopologyAwareBatchCostProvider(bootstrap.scenario, defer_physical_execution=True),
        execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
    runtime.run_until_decode_steps(outcome.diagnostics["sampled_decode_cohorts"], strict_cohort_cutoff=True)
    prefix = runtime._result()
    assert prefix.events == outcome.sampled.events
    assert prefix.request_metrics == outcome.sampled.request_metrics
    assert prefix.external_kv_ingress_metrics == outcome.sampled.external_kv_ingress_metrics
    assert [(row.start_ns, row.end_ns) for row in prefix.batches] == [
        (row.start_ns, row.end_ns) for row in outcome.sampled.batches]
    if forecast is None:
        # A tighter physical phase model may correctly reject this synthetic
        # request. Acceptance must not be preserved by deleting its guard or
        # forcing an unexecuted detailed suffix merely to satisfy this test.
        assert outcome.diagnostics["rejection_reasons"] == [
            "native_window_single_point_or_independent_pair_budget_exceeded"]
        assert outcome.diagnostics["effective_mode"] == "forecast_rejected"
        assert not runtime._result().decode_measurement["complete"]
        return
    cutoff = forecast.cutoff_ns
    full = runtime.run()
    window = full.decode_measurement
    complete = window["complete_request_tpot_ns"]
    observed = {"mean_complete_request_tpot_ns": fmean(complete.values()),
        "mean_complete_request_ingress_ttft_ns": fmean(full.request_metrics[key].ttft_ns for key in complete),
        "cluster_tokens_per_second": window["visible_tokens_per_second"],
        "measurement_duration_ns": window["duration_ns"]}
    assert forecast.cutoff_ns == cutoff < full.makespan_ns
    assert forecast.metrics["measurement_visible_output_tokens"] == window["visible_output_tokens"]
    assert forecast.metrics["complete_fresh_requests"] == len(complete)
    for name, measured in observed.items():
        bounds = forecast.metric_bounds[name]
        assert bounds["lower"] <= measured <= bounds["upper"], name
        assert abs(forecast.metrics[name]-measured)/measured <= .1, name
