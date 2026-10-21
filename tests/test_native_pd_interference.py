"""Measured PD interference replaces the per-import worst-case span (B=1)."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import native_layer_sampling as layer
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.native_pd_interference import interference_model, single_slot_import_timeline
from heterollm_sim.native_window_budget import single_slot_skeleton
from test_native_concurrent_imports import _executor_state
from test_native_external_kv import ingress_scenario, native_runtime
from test_native_window_lifecycle import capture_first_native_query
from test_token_owned_history import bits


def _run(origin, layers, steps, imports=None):
    times, now = {}, origin
    for identity, step in zip(layers, steps):
        now += step
        times[identity] = now
    return {"start_ns": origin, "end_ns": now, "duration_ns": now-origin, "boundary_origin_ns": origin,
        "elapsed_from_boundary_ns": now-origin, "task_count": 10, "layer_completion_ns": times,
        "concurrent_imports": imports, "wall_seconds": 0.}


def _imports(writes, written, final=()):
    return {"request_ids": ["r2"], "submitted_phases": writes, "written_bytes_in_window": written,
        "write_completions_ns": [], "write_phases_in_window": writes, "final_reached": list(final),
        "scope": "test"}


def _window(role, layers, origin, busy, writes, written, final=()):
    return {"role": role, "layer_ids": layers, "origin_ns": origin,
        "quiet": _run(origin, layers, [100.]*len(layers)),
        "concurrent": _run(origin, layers, busy, _imports(writes, written, final))}


def _samples(**changes):
    calibration = _window("calibration", [f"L{index}" for index in range(5)], 1000.,
                          [150., 160., 150., 170., 150.], 4, 4000.)
    holdout = _window("holdout", ["L6", "L7", "L8"], 5000., [150., 160., 150.], 3, 3000.)
    non_layer = {"position": "before_first_layer", "origin_ns": 1000.,
        "quiet": _run(1000., ["x"], [200.]), "concurrent": _run(1000., ["x"], [240.], _imports(1, 100.))}
    samples = {"qualified": True, "continued_request_id": "r2", "layer_count": 9,
        "layer_windows": [calibration, holdout], "non_layer_windows": [non_layer]}
    samples.update(changes)
    return samples


def test_model_uses_post_transient_slope_quantization_hull_and_quiet_weights():
    model = interference_model(_samples())
    assert model["qualified"]
    calibration, holdout = model["windows"]
    # Completion slopes after the first layer: 630/400 and 310/200.
    assert calibration["slowdown"] == pytest.approx(.575) and holdout["slowdown"] == pytest.approx(.55)
    # Per-layer excess {60,50,70,50} and {60,50}: quantum is the largest layer excess.
    assert calibration["quantization"] == pytest.approx(.05) and holdout["quantization"] == pytest.approx(.05)
    assert model["quantum_ns"] == 70. and calibration["transient_ns"] == 50.
    # Hull of s +- Q, widened on both sides by the 0.025 calibration/holdout drift.
    assert model["calibration_holdout_drift"] == pytest.approx(.025)
    assert model["layer_slowdown"] == pytest.approx({"lower": .475, "upper": .65})
    # Nine quiet 100 ns layers plus a 200 ns prefix at 0.2 +- 70/200.
    assert model["slowdown"]["lower"] == pytest.approx((900*.475+200*(.2-.35))/1100)
    assert model["slowdown"]["upper"] == pytest.approx((900*.65+200*(.2+.35))/1100)
    assert model["write_rate_bytes_per_ns"] == pytest.approx({
        "lower": 4000/780*.75, "upper": 3000/460*4/3})


@pytest.mark.parametrize("change,reason", (
    ({"qualified": False, "reason": "native_PD_import_not_in_flight_at_cutoff"},
     "native_PD_import_not_in_flight_at_cutoff"),
    ({"layer_windows": [_window("calibration", ["L0", "L1", "L2"], 0., [150., 160., 150.], 3, 3000.)]},
     "native_PD_interference_requires_calibration_and_holdout"),
    ({"layer_windows": [_window("calibration", ["L0", "L1", "L2"], 0., [150., 160., 150.], 3, 3000., ("r2",)),
                        _window("holdout", ["L6", "L7", "L8"], 0., [150., 160., 150.], 3, 3000.)]},
     "native_PD_interference_window_not_single_unfinished_import"),
    ({"layer_windows": [_window("calibration", ["L0", "L1", "L2"], 0., [150., 160., 150.], 1, 1000.),
                        _window("holdout", ["L6", "L7", "L8"], 0., [150., 160., 150.], 3, 3000.)]},
     "native_PD_interference_window_has_too_few_import_writes")))
def test_model_fails_closed(change, reason):
    result = interference_model(_samples(**change))
    assert (result["qualified"], result["reason"]) == (False, reason)


def test_model_widens_a_phase_locked_drift_instead_of_refusing_it():
    # The holdout locks into a slower pattern (s=1.55) than calibration (0.55).
    model = interference_model(_samples(layer_windows=[
        _window("calibration", ["L0", "L1", "L2"], 0., [150., 160., 150.], 3, 3000.),
        _window("holdout", ["L6", "L7", "L8"], 0., [150., 260., 250.], 3, 3000.)]))
    assert model["qualified"] and model["calibration_holdout_drift"] == pytest.approx(1.)
    assert model["layer_slowdown"] == pytest.approx({"lower": .55-.05-1., "upper": 1.55+.05+1.})


MODEL = {"qualified": True, "slowdown": {"lower": .5, "upper": .6},
         "write_rate_bytes_per_ns": {"lower": 4., "upper": 5.}, "quantum_ns": 2.}


def _fake_runtime(limit=2, pages=(5, 5, 5, 0), pending=("r1", "r2")):
    # r1's last 50-byte chunk is in flight; r2 sits between chunks.
    states = {1: {"request_id": "r1", "destinations": [{"size_bytes": 1000}], "row": 0, "offset": 950,
                  "logical_bytes": 1000, "actions": ("memory",), "amount": 50},
              2: {"request_id": "r2", "destinations": [{"size_bytes": 1000}], "row": 0, "offset": 0,
                  "logical_bytes": 1000, "actions": (), "amount": 0}}
    specs = [SimpleNamespace(request_id=f"r{index}", prompt_tokens=8, output_tokens=3) for index in range(4)]
    return SimpleNamespace(_external_ingress={"max_inflight_requests": limit, "chunk_bytes": 100,
                                              "buffer_bytes": 400},
        _external_executor=SimpleNamespace(_async_plan_states=states),
        _external_pending={identity: {} for identity in pending}, _external_ready=set(),
        _state_values=[SimpleNamespace(spec=spec, kv_pages=count) for spec, count in zip(specs, pages)],
        plan=SimpleNamespace(requests=specs, scheduler=SimpleNamespace(max_num_seqs=1)),
        _retained_kv_state=None, _prompt_cache_save_enabled=lambda: False)


def _pd(future=("r1", "r2", "r3")):
    return {"future_requests": [{"request_id": identity, "logical_bytes": 1000, "chunk_count": 10,
                                 "initial_state_phase_serial_upper_ns": 10.} for identity in future]}


SKELETON = {"events": [{"kind": kind, "request_id": identity} for identity in ("r0", "r1", "r2")
                       for kind in ("prefill", "decode", "decode")]}


def test_timeline_anchors_imports_at_admissions_and_charges_measured_extra():
    timeline = single_slot_import_timeline(_fake_runtime(), SKELETON, _pd(), [(200., 220.)]*9, MODEL)
    rows = {row["request_id"]: row for row in timeline["imports"]}
    # r1/r2 are in flight at the cutoff; r3 starts when r1 frees a slot.
    assert {key: row["anchor_boundary"] for key, row in rows.items()} == {"r1": 0, "r2": 0, "r3": 3}
    # r1's 50-byte tail may overlap r2: one quantum per chunk, and its span extends r2.
    assert rows["r1"]["tail"] and rows["r1"]["extra_ns"] == (0., 2.)
    # The in-flight chunk may be partly done: it leaves only the lower span.
    assert rows["r1"]["span_lower_ns"] == 0. and rows["r1"]["span_upper_ns"] == pytest.approx(50/4+1.)
    assert rows["r2"]["span_upper_ns"] == pytest.approx(1000/4+10.+50/4+1.)
    assert rows["r2"]["extra_ns"] == pytest.approx((200*.5/1.5-4., rows["r2"]["span_upper_ns"]*.6/1.6+4.))
    assert rows["r3"]["extra_ns"] == pytest.approx((200*.5/1.5-4., (1000/4+10.)*.6/1.6+4.))
    span = timeline["span_extra"]
    # The window (events 3..8) contains r3 completely, and neither earlier import.
    assert span(3, 9) == pytest.approx(rows["r3"]["extra_ns"])
    # r1's TTFT contains r1's tail and r2 entirely; r3 may only have started.
    assert span(0, 4) == pytest.approx((rows["r2"]["extra_ns"][0],
        2.+rows["r2"]["extra_ns"][1]+rows["r3"]["extra_ns"][1]))
    # r3 starts one event before r1's life, so its lower part may lie before it.
    assert span(4, 6) == pytest.approx((0., rows["r3"]["extra_ns"][1]))
    assert timeline["total_upper_ns"] == pytest.approx(sum(row["extra_ns"][1] for row in rows.values()))


@pytest.mark.parametrize("runtime,pd,services,reason", (
    (_fake_runtime(), _pd(), [(10., 11.)]*9, "native_PD_admission_may_wait_for_import"),
    (_fake_runtime(), _pd(("r1", "r3")), [(200., 220.)]*9, "native_PD_in_flight_plan_state_unavailable"),
    (_fake_runtime(pending=("r1",)), _pd(("r1", "r3")), [(200., 220.)]*9,
     "native_PD_admission_import_not_started"),
    (_fake_runtime(pages=(5, 0, 5, 0)), _pd(), [(200., 220.)]*9, "native_PD_future_import_capacity_not_proven")))
def test_timeline_fails_closed(runtime, pd, services, reason):
    assert single_slot_import_timeline(runtime, SKELETON, pd, services, MODEL) == {
        "qualified": False, "reason": reason}


def test_timeline_without_future_imports_adds_nothing_and_needs_no_samples():
    timeline = single_slot_import_timeline(_fake_runtime(), SKELETON, {"future_requests": []},
        [(200., 220.)]*9, {"qualified": False, "reason": "unused"})
    assert timeline["qualified"] and timeline["imports"] == [] and timeline["span_extra"](0, 9) == (0., 0.)
    assert single_slot_import_timeline(_fake_runtime(), SKELETON, _pd(), [(200., 220.)]*9,
        {"qualified": False, "reason": "native_PD_interference_window_has_too_few_import_writes"})["reason"] == (
        "native_PD_interference_window_has_too_few_import_writes")


def _b1_scenario():
    scenario = ingress_scenario("hbm_tp1", warmup=3, measured=6, count=4)
    return replace(scenario, workload=replace(scenario.workload,
        scheduler=replace(scenario.workload.scheduler, max_num_seqs=1)))


@pytest.fixture(scope="module")
def b1_cutoff():
    return capture_first_native_query(_b1_scenario())


def test_timeline_anchors_match_live_serving_import_starts(b1_cutoff):
    from heterollm_sim.native_window_producer import _native_pd_evidence
    runtime, _prepared = b1_cutoff
    skeleton = single_slot_skeleton(runtime.plan.requests, 3, 6)
    model = {**MODEL, "write_rate_bytes_per_ns": {"lower": 1., "upper": 2.}}
    timeline = single_slot_import_timeline(runtime, skeleton, _native_pd_evidence(runtime),
        [(1e7, 2e7)]*len(skeleton["events"]), model)
    assert timeline["qualified"], timeline
    anchors = {row["request_id"]: row["anchor_boundary"] for row in timeline["imports"]}
    live = native_runtime(_b1_scenario())
    live.run()
    starts = {event.request_id: event.timestamp_ns for event in live.events
              if event.event_type == "external_kv_import_start"}
    admitted = {event.request_id: event.timestamp_ns for event in live.events
                if event.event_type == "request_admitted"}
    assert set(anchors) == set(starts)-runtime._external_initialized
    for identity, anchor in anchors.items():
        admission = skeleton["events"][anchor]["request_id"]
        # In-flight imports started no later than the seed's admission.
        assert starts[identity] <= admitted[admission] if anchor == 0 else starts[identity] == admitted[admission]
    assert anchors["r3"] > 0


def test_real_pairs_continue_only_the_longest_import_and_leave_source_unchanged(b1_cutoff):
    runtime, captured = b1_cutoff
    prepared = layer.prepare_native_layer_windows(runtime, captured["cohort"], captured["cost"],
        captured["stages"], captured["ready"], ExecutionControl())
    source, executor = prepared["source"], runtime._external_executor
    saved = layer._independent_kernel(source)[0].physical_runtime.snapshot()
    before = (bits(saved["timeline"]), _executor_state(executor), source.makespan_ns)
    # The two-layer miniature cannot hold disjoint calibration/holdout runs.
    assert len(prepared["inventory"]) == 2
    assert layer.sample_native_pd_interference(runtime, prepared, ExecutionControl(),
        calibration_layers=2, holdout_layers=2)["reason"] == "native_PD_interference_layer_windows_do_not_fit"
    remaining = {identity: layer.remaining_import_bytes(state)
                 for state in executor._async_plan_states.values()
                 if (identity := state.get("request_id")) in runtime._external_pending}
    longest = max(remaining, key=lambda key: (remaining[key], key))
    layers = [row["layer_id"] for row in prepared["inventory"]]
    ids = [identity for layer_id in layers for identity in prepared["by_layer"][layer_id]]
    windows = []
    for role, shift in (("calibration", 0.), ("holdout", 5e3)):
        origin = prepared["origin"]+shift
        pair = layer._pd_pair(source, prepared["specs"], ids, origin_ns=origin, control=ExecutionControl(),
            sample_id="pd-test-"+role, imports=(executor, [longest]), progress_metadata=None)
        assert pair["quiet"]["concurrent_imports"] is None
        assert pair["concurrent"]["concurrent_imports"]["request_ids"] == [longest]
        assert pair["quiet"]["task_count"] == pair["concurrent"]["task_count"]
        assert set(pair["quiet"]["layer_completion_ns"]) == set(pair["concurrent"]["layer_completion_ns"]) == set(layers)
        windows.append({"role": role, "layer_ids": layers, "origin_ns": origin, **pair})
    after = layer._independent_kernel(source)[0].physical_runtime.snapshot()
    assert (bits(after["timeline"]), _executor_state(executor), source.makespan_ns) == before
    # The miniature import is far shorter than its layers: it finishes, so the
    # k=1 model refuses instead of extrapolating a finished import.
    samples = {"qualified": True, "continued_request_id": longest, "layer_count": 2,
               "layer_windows": windows, "non_layer_windows": []}
    assert interference_model(samples)["reason"] == "native_PD_interference_window_not_single_unfinished_import"


@pytest.fixture(scope="module")
def seeded_cutoff():
    from heterollm_sim.config import RequestSpec
    from heterollm_sim.simulation_options import NativeWindowFastConfig
    scenario = _b1_scenario()
    requests = tuple(RequestSpec(f"r{index}", 0., 7, 4, metadata={"warmup_seed": index == 0})
                     for index in range(4))
    runtime, captured = capture_first_native_query(replace(scenario,
        workload=replace(scenario.workload, requests=requests)))
    options = NativeWindowFastConfig(sampling_unit="layer_windows")
    payload = layer.sample_native_layer_windows(runtime, captured["cohort"], captured["cost"],
        captured["stages"], captured["ready"], options, ExecutionControl())
    return runtime, captured, payload, options


def _evaluate(seeded, monkeypatch, *, model=None, zero_extra=False):
    from heterollm_sim import native_layer_budget as budget, native_pd_interference as interference
    from heterollm_sim import native_window_producer as producer
    runtime, captured, payload, options = seeded
    phase = {"future_context_phase_growth_serial_upper_ns": 0.,
             "future_nonphysical_service_growth_serial_upper_ns": 0.,
             "future_nonphysical_service_decrease_serial_upper_ns": 0.}
    # Only the PD plumbing is under test: the layer and context contracts are fixed.
    monkeypatch.setattr(budget, "evaluate_native_layer_windows", lambda *_args, **_kwargs: {
        "accepted": True, "reasons": [], "cohort_interval": {"lower": 2e7, "upper": 2.1e7}, "error_budget": {}})
    monkeypatch.setattr(budget, "_query_decode_contract", lambda *_args, **_kwargs: {
        "nonphysical_service_difference_serial_guard_ns": 0., "visible_commit_lag_upper_ns": 0.,
        "future_context_phase_contract": dict(phase), "future_query_context_phase_contract": dict(phase)})
    monkeypatch.setattr(producer, "_allocation_evidence", lambda *_args, **_kwargs: {"qualified": True})
    if model is not None:
        monkeypatch.setattr(interference, "interference_model", lambda samples: model)
    if zero_extra:
        real = interference.single_slot_import_timeline
        def without_extra(*args):
            return {**real(*args), "span_extra": lambda first, last: (0., 0.)}
        monkeypatch.setattr(interference, "single_slot_import_timeline", without_extra)
    return budget.evaluate_native_layer_window(runtime, captured["cohort"], captured["cost"],
        captured["stages"], payload, options)


def test_window_forecast_fails_closed_without_measured_interference(seeded_cutoff, monkeypatch):
    assert seeded_cutoff[2]["pd_interference"]["reason"] == "native_PD_interference_layer_windows_do_not_fit"
    result = _evaluate(seeded_cutoff, monkeypatch)
    assert result["reasons"] == ["native_PD_interference_layer_windows_do_not_fit"]
    assert not result["accepted"] and not result["metric_bounds"]


def test_window_forecast_adds_measured_extra_to_both_bounds_and_no_serial_PD(seeded_cutoff, monkeypatch):
    model = {**MODEL, "write_rate_bytes_per_ns": {"lower": 1., "upper": 2.}}
    with_pd = _evaluate(seeded_cutoff, monkeypatch, model=model)
    budget = with_pd["error_budget"]
    assert "future_PD_serial_upper_ns" not in budget and "future_PD_life_exposure" not in budget
    measured = budget["future_PD_measured_interference"]
    assert measured["model"] is model and "span_extra" not in measured["timeline"]
    assert budget["future_PD_outstanding_request_ids"] == ["r1", "r2", "r3"]
    monkeypatch.undo()
    without = _evaluate(seeded_cutoff, monkeypatch, model=model, zero_extra=True)
    window = measured["window_extra_ns"]
    assert window["upper"] > window["lower"] > 0.
    bounds, base = with_pd["metric_bounds"], without["metric_bounds"]
    duration, base_duration = bounds["measurement_duration_ns"], base["measurement_duration_ns"]
    assert duration["lower"]-base_duration["lower"] == pytest.approx(window["lower"])
    assert duration["upper"]-base_duration["upper"] == pytest.approx(window["upper"])
    extras = measured["request_extras"]
    for name, key in (("mean_complete_request_tpot_ns", "life_extra_ns"),
                      ("mean_complete_request_ingress_ttft_ns", "ingress_extra_ns")):
        scale = 3 if name.endswith("tpot_ns") else 1
        for side in ("lower", "upper"):
            expected = sum(row[key][side] for row in extras)/len(extras)/scale
            assert bounds[name][side]-base[name][side] == pytest.approx(expected)


def _long_import_toy(factor=1000.):
    """56 contended layers whose real PD imports outlast every sample window.

    PD links run at 25 Mb/s, so each 68 KB import stays active ~22-38 ms; the
    HBM interface is `factor` times slower than the synthetic controller, so
    Decode reads and import writes contend on it. Layers take ~0.66 ms and a
    request lives ~111 ms, so every import still ends before its admission.
    """
    from heterollm_sim.physical_contract import parse_execution_physical_config
    from test_native_layer_budget import toy_scenario
    scenario = toy_scenario(56, gpu_ghz=.0001)
    components = []
    for component in scenario.hardware.components:
        if component.kind == "hbm":
            config = parse_execution_physical_config(component.metadata["physical_memory_config"])
            config = replace(config, burst_interval_ns=config.burst_interval_ns*factor,
                read_bandwidth_gb_s=config.read_bandwidth_gb_s/factor,
                write_bandwidth_gb_s=config.write_bandwidth_gb_s/factor,
                interface_bandwidth_gb_s=config.interface_bandwidth_gb_s/factor)
            component = replace(component, bandwidth_gbps=component.bandwidth_gbps/factor,
                read_bandwidth_gbps=component.read_bandwidth_gbps/factor,
                write_bandwidth_gbps=component.write_bandwidth_gbps/factor,
                ports=tuple(replace(port, bandwidth_gbps=port.bandwidth_gbps/factor) for port in component.ports),
                metadata={**component.metadata, "physical_memory_config": config})
        components.append(component)
    hbm = {component.component_id for component in components if component.kind == "hbm"}
    links = tuple(replace(link, bandwidth_gbps=.025) if link.link_id.startswith("pd-ingress")
                  else replace(link, bandwidth_gbps=link.bandwidth_gbps/factor)
                  if {link.source_component, link.target_component} & hbm else link
                  for link in scenario.hardware.links)
    profiles = {key: replace(profile, bandwidth_gb_s=profile.bandwidth_gb_s/factor,
                             read_bandwidth_gb_s=profile.read_bandwidth_gb_s/factor,
                             write_bandwidth_gb_s=profile.write_bandwidth_gb_s/factor)
                for key, profile in scenario.component_profiles["hbm"].items()}
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(components), links=links),
                   component_profiles={**scenario.component_profiles, "hbm": profiles})


def test_long_import_toy_forecast_contains_its_complete_simulation():
    """The measured PD model on real contention, checked against a full run.

    Decode and the import phase-lock into a start-phase dependent pattern, so
    calibration and holdout drift beyond write quantization; the interval is
    widened by that drift. The full simulation must lie inside every window
    metric, and every non-tail import span inside its forecast span.
    """
    from statistics import fmean
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import SimulationConfig, NativeWindowFastConfig
    scenario = _long_import_toy()
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(sampling_unit="layer_windows"))), retention_policy="aggregate")
    forecast = fast.native_window_forecast
    assert forecast is not None, fast.approximation["rejection_reasons"]
    assert fast.sampled is None and fast.detailed is None
    measured = forecast.error_budget["future_PD_measured_interference"]
    model = measured["model"]
    calibration, holdout = model["windows"]
    # Each window holds one uniform per-layer excess (no write quantization),
    # yet the two start phases lock into different patterns.
    assert calibration["quantization"] == pytest.approx(0., abs=1e-9)
    assert holdout["quantization"] == pytest.approx(0., abs=1e-9)
    drift = model["calibration_holdout_drift"]
    assert drift == pytest.approx(abs(calibration["slowdown"]-holdout["slowdown"])) and drift > 0.
    assert model["layer_slowdown"]["lower"] == pytest.approx(
        min(calibration["slowdown"], holdout["slowdown"])-drift)
    full = run_scenario(scenario, retention_policy="aggregate").serving
    starts = {event.request_id: event.timestamp_ns for event in full.events
              if event.event_type == "external_kv_import_start"}
    done = {event.request_id: event.timestamp_ns for event in full.events
            if event.event_type == "external_kv_import_complete"}
    imports = [row for row in measured["timeline"]["imports"] if not row["tail"]]
    assert imports
    for row in imports:
        span = done[row["request_id"]]-starts[row["request_id"]]
        assert row["span_lower_ns"] <= span <= row["span_upper_ns"], row["request_id"]
    window = full.decode_measurement
    complete = window["complete_request_tpot_ns"]
    observed = {"mean_complete_request_tpot_ns": fmean(complete.values()),
        "mean_complete_request_ingress_ttft_ns": fmean(full.request_metrics[key].ttft_ns for key in complete),
        "cluster_tokens_per_second": window["visible_tokens_per_second"],
        "measurement_duration_ns": window["duration_ns"]}
    assert forecast.metrics["complete_fresh_requests"] == len(complete)
    assert forecast.metrics["measurement_visible_output_tokens"] == window["visible_output_tokens"]
    for key, value in observed.items():
        interval = forecast.metric_bounds[key]
        assert interval["lower"] <= value <= interval["upper"], key
        assert abs(interval["estimate"]-value)/value <= .1, key
