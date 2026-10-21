"""Independent layer traces need both calibration and future-state guards."""
from copy import deepcopy

import pytest

from heterollm_sim.native_layer_budget import evaluate_native_layer_windows


def summary(phase=1.):
    return {"schema": "native-window-ir-observations/v1", "unsupported": [],
        "stateful_gpu_cache": False, "ordered_resource_phase_per_cohort_ns": phase}


def sample(identity, role, *, phase=1., carry=0.):
    return {"sample_id": identity, "role": role, "class_key": "dense",
        "layer_ids": (identity,), "start_ns": 0., "end_ns": 10000., "duration_ns": 10000.,
        "boundary_wait_ns": 0., "sample_state": "independent_native_kernel_copy_not_full_cohort_prefix",
        "native_ir_summary": summary(phase),
        "input_boundary": {"qualified": True, "initial_carry_upper_ns": carry},
        "output_boundary": {"qualified": True}}


def payload():
    windows = [sample(f"L{i}", role) for i, role in enumerate(
        ("calibration", "calibration", "validation", "validation"))]
    nonlayer = sample("head", "nonlayer", phase=0.)
    return {"schema": "native-layer-physical-samples/v1",
        "layer_inventory": [{"layer_id": f"L{i}", "class_key": "dense", "signature": {},
            "native_ir_summary": summary()} for i in range(8)],
        "layer_window_samples": windows, "non_layer_samples": [nonlayer],
        "non_layer_coverage_complete": True,
        "layer_composition": {"qualified": True, "basis": "native DAG proof",
            "serial_layer_boundaries_proven": True, "boundary_guard_ns": 0.,
            "source_pending_carry_upper_ns": 0.}}


def test_disjoint_validation_composes_values_without_claiming_full_cohort_execution():
    result = evaluate_native_layer_windows(payload())
    assert result["accepted"] and not result["reasons"]
    assert result["physical_sampled_layer_count"] == result["forecast_layer_count"] == 4
    assert result["cohort_interval"]["lower"] == 90000.-8*2.
    assert result["cohort_interval"]["upper"] == 90000.+8*2.
    assert result["physical_visible_output_tokens"] == 0
    assert all(row["forecast_only_in_full_cohort"] for row in result["layer_forecasts"])


@pytest.mark.parametrize("forecast_end,sample_end,accepted", [
    (100000., 10000., True), (100001., 10000., False), (90000., 100001., False)])
def test_restricted_hbm_clock_profile_requires_full_lifecycle_and_samples_inside_domain(
        forecast_end, sample_end, accepted):
    from heterollm_sim.native_layer_budget import _close_hbm_clock_domain
    data = payload()
    data["layer_inventory"][0]["native_ir_summary"]["hbm_clock_domain_limit_ns"] = 200000.
    data["non_layer_samples"][0]["native_ir_summary"]["hbm_clock_domain_limit_ns"] = 100000.
    layer = evaluate_native_layer_windows(data)
    assert layer["accepted"]
    budget = layer["error_budget"]
    assert budget["hbm_clock_domain_limit_ns"] == 100000.
    budget["physical_sample_completion_upper_ns"] = sample_end
    result = {"reasons": [], "error_budget": {}}
    _close_hbm_clock_domain(result, (budget,), forecast_end)
    assert result["error_budget"]["native_hbm_clock_domain"]["qualified"] is accepted
    assert bool(result["reasons"]) is not accepted


def test_seed_anchor_clock_domain_must_also_close_and_old_unrestricted_bounds_need_no_limit():
    from heterollm_sim.native_layer_budget import _close_hbm_clock_domain
    result = {"reasons": [], "error_budget": {}}
    _close_hbm_clock_domain(result, ({},), 1e20)
    assert not result["reasons"] and not result["error_budget"]
    main = {"hbm_clock_domain_limit_ns": 1e6, "physical_sample_completion_upper_ns": 1e4}
    anchor = {"hbm_clock_domain_limit_ns": 1e5, "physical_sample_completion_upper_ns": 1e4}
    _close_hbm_clock_domain(result, (main, anchor), 2e5)
    assert result["reasons"] == ["native_layer_full_lifecycle_exceeds_proven_hbm_clock_domain"]
    result = {"reasons": [], "error_budget": {}}
    _close_hbm_clock_domain(result, (main,), 2e5,
        phase_proofs=({"hbm_clock_domain_limit_ns": 1e5},))
    assert result["reasons"] == ["native_layer_full_lifecycle_exceeds_proven_hbm_clock_domain"]


@pytest.mark.parametrize("change,reason", [
    ("old_role", "native_layer_sample_role_invalid"),
    ("overlap", "native_layer_calibration_and_holdout_layers_overlap"),
    ("missing_class", "native_layer_disjoint_semantic_class_samples_insufficient"),
    ("nonlayer", "native_layer_embedding_head_host_coverage_incomplete"),
    ("carry", "native_layer_source_carry_bound_unavailable"),
    ("commit", "native_layer_samples_must_not_commit_full_cohort_tokens"),
    ("serial", "native_layer_serial_composition_not_proven"),
])
def test_unproven_state_or_composition_is_rejected(change, reason):
    data = payload()
    if change == "old_role":
        data["layer_window_samples"][2]["role"] = "holdout"
    elif change == "overlap":
        data["layer_window_samples"][2]["layer_ids"] = ("L0",)
    elif change == "missing_class":
        data["layer_inventory"][-1]["class_key"] = "other"
    elif change == "nonlayer":
        data["non_layer_coverage_complete"] = False
    elif change == "carry":
        del data["layer_window_samples"][0]["input_boundary"]["initial_carry_upper_ns"]
    elif change == "commit":
        data["visible_output_tokens"] = 1
    else:
        data["layer_composition"]["serial_layer_boundaries_proven"] = False
    result = evaluate_native_layer_windows(data)
    assert not result["accepted"] and reason in result["reasons"]


def test_calibration_carry_is_a_lower_bias_and_full_cohort_carry_is_charged_once():
    base = evaluate_native_layer_windows(payload())
    data = payload()
    for row in data["layer_window_samples"]:
        row["input_boundary"]["initial_carry_upper_ns"] = 100.
    data["layer_composition"]["source_pending_carry_upper_ns"] = 100.
    result = evaluate_native_layer_windows(data)
    assert result["accepted"]
    assert result["cohort_interval"]["lower"] == base["cohort_interval"]["lower"]-8*100.
    assert result["cohort_interval"]["upper"] == base["cohort_interval"]["upper"]+100.
    assert result["error_budget"]["classes"][0]["base_service_per_layer_ns"] == 10000.


def test_row_cache_refresh_phase_is_charged_on_both_sides_of_independent_copy():
    data = payload()
    for row in data["layer_window_samples"]:
        row["native_ir_summary"] = summary(200.)
    for row in data["layer_inventory"]:
        row["native_ir_summary"] = summary(200.)
    result = evaluate_native_layer_windows(data)
    assert result["accepted"]
    assert result["cohort_interval"]["lower"] == 90000.-8*400.
    assert result["cohort_interval"]["upper"] == 90000.+8*400.
    assert result["error_budget"]["full_cohort_physical_phase_guard_ns"] == 8*400.


def test_large_phase_is_a_real_rejection_without_adjusting_ten_percent_target():
    data = payload()
    for row in data["layer_window_samples"]:
        row["native_ir_summary"] = summary(2000.)
    result = evaluate_native_layer_windows(data)
    assert not result["accepted"]
    assert "native_layer_conditional_single_or_independent_pair_budget_exceeded" in result["reasons"]


def test_known_pair_can_defer_to_real_directional_gate_without_relaxing_ten_percent_point_target():
    data = payload()
    for row in data["layer_window_samples"]:
        row["native_ir_summary"] = summary(400.)
    for row in data["layer_inventory"]:
        row["native_ir_summary"] = summary(400.)
    independent = evaluate_native_layer_windows(data)
    paired_point = evaluate_native_layer_windows(data, independent_sufficient=False)
    assert not independent["accepted"] and paired_point["accepted"]
    assert .1/2.1 < paired_point["cohort_interval"]["relative_error_budget"] <= .1
    assert paired_point["error_budget"]["target_relative_error"] == .1


def test_exhaustive_small_class_still_pays_independent_state_phase():
    data = payload()
    data["layer_inventory"] = data["layer_inventory"][:2]
    data["layer_window_samples"] = data["layer_window_samples"][:2]
    for row in data["layer_window_samples"]:
        row["role"] = "exhaustive"
    result = evaluate_native_layer_windows(data)
    assert result["accepted"]
    assert result["cohort_interval"]["lower"] == 30000.-2*2.
    assert result["physical_sampled_layer_count"] == 2 and result["forecast_layer_count"] == 0


@pytest.mark.parametrize("layer_count,window_size", ((6, 2), (12, 4)))
def test_expanded_window_exhaustion_is_actual_class_coverage_with_full_phase(
        layer_count, window_size):
    from heterollm_sim.native_layer_sampling import _select_windows
    from heterollm_sim.simulation_options import NativeWindowFastConfig
    data = payload()
    data["layer_inventory"] = [{"layer_id": f"L{i}", "layer_index": i,
        "class_key": "dense", "signature": {}, "native_ir_summary": summary()}
        for i in range(layer_count)]
    selected = _select_windows(data["layer_inventory"], NativeWindowFastConfig(
        sampling_unit="layer_windows", layer_window_size=window_size))
    data["layer_window_samples"] = [
        sample(window["layers"][0]["layer_id"], window["role"]) for window in selected]
    assert all(window["role"] == "exhaustive" and len(window["layers"]) == 1
               for window in selected)
    result = evaluate_native_layer_windows(data)
    assert result["accepted"]
    assert result["error_budget"]["classes"][0]["exhaustive_class"]
    assert result["physical_sampled_layer_count"] == layer_count
    assert result["forecast_layer_count"] == 0
    assert result["error_budget"]["full_cohort_physical_phase_guard_ns"] == layer_count*2.
    assert result["cohort_interval"]["lower"] == (layer_count+1)*10000.-layer_count*2.
    assert result["cohort_interval"]["upper"] == (layer_count+1)*10000.+layer_count*2.


@pytest.mark.parametrize("change,reason", (
    ("missing", "native_layer_exhaustive_class_coverage_or_roles_invalid"),
    ("mixed", "native_layer_exhaustive_class_coverage_or_roles_invalid"),
    ("duplicate", "native_layer_calibration_and_holdout_layers_overlap"),
    ("calibration_only", "native_layer_disjoint_semantic_class_samples_insufficient"),
))
def test_exhaustive_coverage_cannot_be_inferred_from_class_size_or_partial_roles(change, reason):
    data = payload()
    data["layer_inventory"] = data["layer_inventory"][:3]
    data["layer_window_samples"] = [sample(f"L{i}", "exhaustive") for i in range(3)]
    if change == "missing":
        data["layer_window_samples"].pop()
    elif change == "mixed":
        data["layer_window_samples"][0]["role"] = "calibration"
    elif change == "duplicate":
        data["layer_window_samples"][-1]["layer_ids"] = ("L0",)
    else:
        for row in data["layer_window_samples"]:
            row["role"] = "calibration"
    result = evaluate_native_layer_windows(data)
    assert not result["accepted"] and reason in result["reasons"]


def test_exhaustive_class_still_rejects_excessive_independent_state_budget():
    data = payload()
    data["layer_window_samples"] = [sample(f"L{i}", "exhaustive", phase=2000.)
        for i in range(len(data["layer_inventory"]))]
    result = evaluate_native_layer_windows(data)
    assert not result["accepted"]
    assert result["error_budget"]["classes"][0]["exhaustive_class"]
    assert "native_layer_conditional_single_or_independent_pair_budget_exceeded" in result["reasons"]


def test_compact_native_cost_retains_formula_and_KV_conversion_evidence():
    from heterollm_sim.scalable_serving import compact_execution_metadata
    raw = {"cost_model": {"model": "gpu_memory_roofline", "profile": "native_gpu",
        "cache": {"logical_read_bytes": 128, "logical_write_bytes": 128,
                  "cross_invocation_cache_state": False, "expanded_trace": [1, 2]}},
        "cache_input_bits": 16, "cache_output_bits": 16, "cache_artifact_format": None,
        "conversion_instructions_priced": True, "timing_completeness": "complete_copy",
        "unpriced_terms": (), "event_kind": "kv_materialize", "rank": 0}
    compact = compact_execution_metadata(raw)
    assert compact["cost_model"] is True  # Existing native launch/floor interpretation.
    assert compact["native_cost_model"]["model"] == "gpu_memory_roofline"
    assert compact["native_cost_model"]["cache"] == {
        "logical_read_bytes": 128, "logical_write_bytes": 128,
        "cross_invocation_cache_state": False}
    assert compact_execution_metadata(compact) == compact
    assert compact["timing_completeness"] == "complete_copy"


def test_future_context_rejects_duplicate_native_resource_demands_before_projection():
    from dataclasses import replace
    from heterollm_sim.native_layer_budget import _future_context_phase_contract
    from test_native_window_lifecycle import capture_first_native_query, lifecycle_scenario
    runtime, prepared = capture_first_native_query(lifecycle_scenario(1))
    stages = list(prepared["stages"])
    index = next(i for i, stage in enumerate(stages) if any(task.demands for task in stage.execution_tasks))
    tasks = list(stages[index].execution_tasks)
    position = next(i for i, task in enumerate(tasks) if task.demands)
    task = tasks[position]
    tasks[position] = replace(task, demands=(*task.demands, task.demands[0]))
    stages[index] = replace(stages[index], execution_tasks=tuple(tasks))
    with pytest.raises(ValueError, match="native_layer_future_duplicate_resource_demand"):
        _future_context_phase_contract(runtime, prepared["cohort"], tuple(stages))


def toy_scenario(layer_count=2, gpu_ghz=.00001):
    """Small native HBM4 B=1 PD scenario on a scalar analytical controller.

    Both sides share a declared analytical no-refresh synthetic controller;
    independent REF and aliased-resource tests retain those guards in
    production models. Neither side is a 405B/device speed claim.
    """
    from dataclasses import replace
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    from heterollm_sim.memory_types import DramConfig
    from heterollm_sim.physical_contract import parse_execution_physical_config

    layer_bytes = (32*(32+16+16+32)+3*32*64)*2
    layers = tuple(LayerSpec("dense"+str(i), "dense", 32, 64, 4,
        kv_heads=2, dtype="bf16", weight_bytes=layer_bytes) for i in range(layer_count))
    model = ModelSpec(name="native-layer-budget-toy", graph=build_model_graph_from_layer_specs(
        "native-layer-budget-toy", layers, vocabulary_size=64, max_sequence_length=256,
        embedding_weight_bytes=32*64*2, output_weight_bytes=32*64*2,
        output_head_dtype="bf16"))
    scenario = build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", tp=2,
        batch_size=1, model_override=model,
        prompt_override=20, output_override=3, warmup_override=2, measurement_override=4)
    components = []
    for component in scenario.hardware.components:
        raw = component.metadata.get("physical_memory_config")
        if raw is not None and not isinstance(raw, DramConfig):
            raw = parse_execution_physical_config(raw)
        if isinstance(raw, DramConfig):
            component = replace(component, capacity_bytes=8*1024**2, metadata={**component.metadata,
                "physical_memory_config": replace(raw, control_mode="analytical",
                    refresh_policy="disabled", channels=1, pseudo_channels_per_channel=1,
                    subchannels_per_channel=1, stacks=1, dies_per_stack=1, ranks_per_channel=1,
                    bank_groups_per_rank=1, banks_per_group=1, rows_per_bank=8192,
                    data_lanes=1, row_bytes=1024, capacity_bytes=8*1024**2,
                    metadata={**raw.metadata,
                        "control_status": "synthetic_analytical",
                        "geometry_status": "explicit_assumption"})})
        components.append(component)
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)),
        component_profiles={**scenario.component_profiles, "gpu": {
            key: replace(profile, tensor_core=replace(profile.tensor_core, frequency_ghz=gpu_ghz))
            for key, profile in scenario.component_profiles["gpu"].items()}})


def test_real_toy_layer_forecast_refuses_PD_imports_it_cannot_measure():
    """Future B=1 PD imports are budgeted only from measured interference.

    The two-layer toy cannot hold disjoint calibration/holdout layer runs, and
    each of its imports (~5 us) ends long before one sample layer (~3 ms), so
    no k=1 steady state exists to measure. The forecast must refuse instead of
    adding a per-import worst-case span.
    """
    from dataclasses import replace
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import SimulationConfig, NativeWindowFastConfig

    scenario = toy_scenario()
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(sampling_unit="layer_windows"))), retention_policy="aggregate")
    assert fast.native_window_forecast is None
    assert fast.approximation["rejection_reasons"] == ["native_PD_interference_layer_windows_do_not_fit"]
    assert fast.sampled is None and fast.detailed is None
    assert all(row.to_metadata()["visible_output_tokens_observed"] == 0 for row in fast.layer_window_samples)


def _prove(row, qualified=True):
    row["native_ir_summary"]["native_issue_order_proof"] = {"qualified": qualified}


def test_proven_issue_order_composes_guards_one_sided():
    data = payload()
    for row in (*data["layer_inventory"], *data["layer_window_samples"], *data["non_layer_samples"]):
        _prove(row)
    result = evaluate_native_layer_windows(data)
    assert result["accepted"]
    assert result["cohort_interval"]["lower"] == 90000.-8*1.
    assert result["cohort_interval"]["upper"] == 90000.+8*1.
    composition = result["error_budget"]["guard_composition"]
    assert composition["one_sided_layers"] == 8 and composition["one_sided_non_layer_samples"] == 1
    assert composition["symmetric_fallback_components"] == []


def test_unproven_window_or_target_keeps_symmetric_guards():
    data = payload()
    for row in (*data["layer_inventory"], *data["layer_window_samples"], *data["non_layer_samples"]):
        _prove(row)
    _prove(data["layer_inventory"][3], False)
    result = evaluate_native_layer_windows(data)
    assert result["cohort_interval"]["lower"] == 90000.-7*1.-2.
    assert result["cohort_interval"]["upper"] == 90000.+7*1.+2.
    assert result["error_budget"]["guard_composition"]["symmetric_fallback_components"] == ["L3"]
    # One holdout window without the proof removes one-sided calibration
    # for its whole semantic class.
    data = payload()
    for row in (*data["layer_inventory"], *data["layer_window_samples"], *data["non_layer_samples"]):
        _prove(row)
    del data["layer_window_samples"][2]["native_ir_summary"]["native_issue_order_proof"]
    result = evaluate_native_layer_windows(data)
    assert result["cohort_interval"]["lower"] == 90000.-8*2.
    assert result["error_budget"]["classes"][0]["issue_order_proven_for_all_windows"] is False


def test_non_layer_samples_pay_one_guard_per_side_only_when_proven():
    for qualified, width in ((True, 3.), (False, 6.)):
        data = payload()
        data["non_layer_samples"][0]["native_ir_summary"]["ordered_resource_phase_per_cohort_ns"] = 3.
        _prove(data["non_layer_samples"][0], qualified)
        result = evaluate_native_layer_windows(data)
        assert result["cohort_interval"]["lower"] == 90000.-8*2.-width
        assert result["cohort_interval"]["upper"] == 90000.+8*2.+width
        assert result["error_budget"]["embedding_head_host_phase_guard_ns"] == width
