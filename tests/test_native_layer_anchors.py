"""A seed-shape anchor is another real disjoint layer set, never a Token."""
from dataclasses import fields, replace
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim.native_layer_anchors import _monotone_native_work, add_native_seed_shape_anchor
from heterollm_sim.native_layer_sampling import sample_native_layer_windows
from heterollm_sim.simulation_options import NativeWindowFastConfig
from test_native_multislot_budget import bandwidth_only_native_scenario
from test_native_window_lifecycle import capture_first_native_query


def four_layer_native_scenario():
    from heterollm_sim.ir import ModelSpec, build_model_graph_from_layer_specs

    scenario = bandwidth_only_native_scenario()
    template = scenario.model._execution_view.layer_instances[0].layer
    layers = tuple(replace(template, layer_id="dense"+str(index)) for index in range(4))
    model = ModelSpec(name="native-four-layer-anchor-test", graph=build_model_graph_from_layer_specs(
        "native-four-layer-anchor-test", layers, vocabulary_size=1024,
        max_sequence_length=256, embedding_weight_bytes=256*1024*2,
        output_weight_bytes=256*1024*2, output_head_dtype="bf16"))
    return replace(scenario, model=model, placement=replace(scenario.placement,
        model_name=model.name, parallel=replace(scenario.placement.parallel,
            layer_to_stage={layer.layer_id: 0 for layer in layers})))


def nand_memory_bound_native_scenario():
    """Synthetic NAND keeps weight payload dominant and all real Core IO."""
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.memory_types import DramConfig, NandConfig
    from heterollm_sim.physical_contract import parse_execution_physical_config

    scenario = build_native_comparison_scenario("llama3_405b", "HBF", "SHORT", tp=2, batch_size=8,
        model_override=build_pilot_scenario("hbm_tp1").model, prompt_override=7, output_override=8,
        warmup_override=7, measurement_override=14)
    components = []
    for component in scenario.hardware.components:
        raw = component.metadata.get("physical_memory_config")
        if raw:
            config = parse_execution_physical_config(raw)
            if isinstance(config, DramConfig):
                config = replace(config, control_mode="analytical", refresh_policy="disabled", metadata={},
                    **{row.name: 0. for row in fields(config) if row.name.endswith("_ns")})
            elif isinstance(config, NandConfig):
                config = replace(config, host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=.1,
                    host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=1000.,
                    page_read_ns=0., page_program_ns=0., front_ns=0., read_cache_pages_per_bank=0,
                    host_full_duplex=True)
                component = replace(component, bandwidth_gbps=800., read_bandwidth_gbps=.8,
                    write_bandwidth_gbps=800.)
            component = replace(component, metadata={**component.metadata, "physical_memory_config": config})
        components.append(component)
    profiles = {**scenario.component_profiles, "host_memory": {key: (replace(profile,
        bandwidth_gb_s=100., read_bandwidth_gb_s=.1, write_bandwidth_gb_s=100., efficiency=1.)
        if key.startswith("native-hbf") else profile)
        for key, profile in scenario.component_profiles["host_memory"].items()}}
    return replace(scenario, component_profiles=profiles,
        hardware=replace(scenario.hardware, components=tuple(components)))


def test_real_b8_and_b1_anchor_keep_distinct_complete_calibration_and_holdout_windows():
    from test_token_owned_history import bits

    runtime, prepared = capture_first_native_query(four_layer_native_scenario())
    progress = []
    options, control = NativeWindowFastConfig(sampling_unit="layer_windows"), ExecutionControl(progress_callback=progress.append)
    payload = sample_native_layer_windows(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], prepared["ready"], options, control)
    before = (runtime.now, tuple(runtime.events), runtime._completed_decode_steps)
    def physical_state():
        saved = runtime._execution_resource_kernel.physical_runtime.snapshot()
        return bits({key: saved[key] for key in ("timeline", "allocator_state", "committed_owners", "hbf_staging")})
    physical_before = physical_state()
    result = add_native_seed_shape_anchor(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], prepared["ready"], payload, options, control)
    assert len(result["records"]) == 8
    assert len({row["window_id"] for row in result["records"]}) == 8
    anchor, = result["batch_shape_anchors"]
    assert anchor["native_monotone_work_contract"]["qualified"]
    assert anchor["query_decode_native_IR"]["qualified"]
    for rows in (result["layer_window_samples"], anchor["sample_payload"]["layer_window_samples"]):
        assert [row["role"] for row in rows] == ["calibration"]*2+["validation"]*2
        assert len({identity for row in rows for identity in row["layer_ids"]}) == 4
        assert all(row["executed_task_count"] > 0 and row["visible_output_tokens"] == 0 for row in rows)
    assert all(len(row["context_tokens_by_request"]) == 1
               for row in anchor["sample_payload"]["records"])
    assert all(len(row["context_tokens_by_request"]) == 8 for row in result["layer_window_samples"])
    assert runtime.batches == []
    assert before == (runtime.now, tuple(runtime.events), runtime._completed_decode_steps)
    assert physical_before == physical_state()
    assert anchor["physical_complete_cohort_count"] == anchor["physical_visible_output_token_count"] == 0
    rows = [value.metadata["physical_layer_progress"] for value in progress
            if value.stage == "native_layer_sample_completed"]
    assert [row["completed_windows"] for row in rows] == list(range(1, 9))
    assert rows[-1]["total_windows"] == 8 and rows[-1]["complete_cohorts"] == rows[-1]["visible_output_tokens"] == 0


def test_real_native_nand_b8_forecast_matches_new_small_complete_window():
    from statistics import fmean
    from heterollm_sim.native_layer_budget import evaluate_native_layer_window

    runtime, prepared = capture_first_native_query(nand_memory_bound_native_scenario())
    options, control = NativeWindowFastConfig(sampling_unit="layer_windows"), ExecutionControl()
    payload = sample_native_layer_windows(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], prepared["ready"], options, control)
    payload = add_native_seed_shape_anchor(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], prepared["ready"], payload, options, control)
    forecast = evaluate_native_layer_window(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], payload, options)
    assert forecast["accepted"], forecast["reasons"]
    assert not runtime.batches
    assert all(row["relative_error_budget"] <= .1/2.1 for row in forecast["metric_bounds"].values())
    del runtime._native_layer_sampler
    full = runtime.run()
    window = full.decode_measurement
    complete = window["complete_request_tpot_ns"]
    observed = {"measurement_duration_ns": window["duration_ns"],
        "cluster_tokens_per_second": window["visible_tokens_per_second"],
        "mean_complete_request_tpot_ns": fmean(complete.values()),
        "mean_complete_request_ingress_ttft_ns": fmean(full.request_metrics[key].ttft_ns for key in complete)}
    assert forecast["metrics"]["complete_fresh_requests"] == len(complete) == 16
    assert forecast["metrics"]["measurement_visible_output_tokens"] == window["visible_output_tokens"]
    for metric, value in observed.items():
        bound = forecast["metric_bounds"][metric]
        assert bound["lower"] <= value <= bound["upper"], (metric, value, bound)
        assert abs(bound["estimate"]-value)/value <= .1


def test_cancelled_seed_anchor_retains_base_and_only_completed_anchor_windows():
    runtime, prepared = capture_first_native_query(bandwidth_only_native_scenario())
    options = NativeWindowFastConfig(sampling_unit="layer_windows")
    payload = sample_native_layer_windows(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], prepared["ready"], options, ExecutionControl())
    cancelled = [False]
    def progress(value):
        if value.stage == "native_layer_sample_completed":
            cancelled[0] = True
    control = ExecutionControl(progress_callback=progress, cancellation_callback=lambda: cancelled[0])
    with pytest.raises(ExecutionCancelledError):
        add_native_seed_shape_anchor(runtime, prepared["cohort"], prepared["cost"],
            prepared["stages"], prepared["ready"], payload, options, control)
    rows = runtime._native_layer_sample_records
    assert len(rows) == len(payload["records"])+1
    assert rows[-1]["window_id"].startswith("seed-shape-B1.")
    assert all(row["executed_task_count"] > 0 for row in rows)
    assert not runtime.batches


def test_native_anchor_rejects_nonmonotone_resource_work_or_changed_task_dependencies():
    full, anchor = NS(cohort_id="full"), NS(cohort_id="anchor")
    task = TaskSpec("full.00001.full.prefill.op", "r", "op", TaskCategory.COMPUTE,
        demands=(ResourceDemand("gpu", 10., work_units=4.),))
    original = replace(task, task_id="anchor.00001.anchor.prefill.op",
        demands=(ResourceDemand("gpu", 5., work_units=1.),))
    stage = lambda row: (NS(execution_tasks=(row,)),)
    assert _monotone_native_work(full, stage(task), anchor, stage(original))["qualified"]
    larger = replace(original, demands=(ResourceDemand("gpu", 11., work_units=1.),))
    assert not _monotone_native_work(full, stage(task), anchor, stage(larger))["qualified"]
    changed = replace(original, dependencies=("anchor.00000.anchor.prefill.unknown",))
    assert not _monotone_native_work(full, stage(task), anchor, stage(changed))["qualified"]
