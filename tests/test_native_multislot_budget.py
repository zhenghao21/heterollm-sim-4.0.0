"""Native full-batch samples do not silently cover unsampled seed shapes."""
from dataclasses import fields, replace
from statistics import fmean

import pytest

from heterollm_sim.native_fast_window import cohort_samples
from heterollm_sim.native_window_budget import evaluate_native_window
from heterollm_sim.simulation_options import NativeWindowFastConfig
from test_native_external_kv import native_runtime
from test_native_window_lifecycle import capture_first_native_query, lifecycle_scenario


def test_real_b2_waiting_seed_refuses_future_wave_projection_without_extra_execution():
    runtime = native_runtime(lifecycle_scenario(2))
    runtime.run_until_decode_steps(5, strict_cohort_cutoff=True)
    records = cohort_samples(runtime)
    cutoff, batches = runtime.now, tuple(runtime.batches)
    result = evaluate_native_window(runtime, records, NativeWindowFastConfig())
    assert not result["accepted"]
    assert result["reasons"] == ["native_multislot_seed_washout_not_proven"]
    assert not result["metrics"]
    assert runtime.now == cutoff and tuple(runtime.batches) == batches
    assert "warmup_seed_still_waiting_or_prefilling" in result["error_budget"]["native_lifecycle"][
        "seed_washout"]["reasons"]


def test_real_b8_short_seed_lifetimes_need_enough_disjoint_full_batch_samples():
    runtime = native_runtime(lifecycle_scenario(8))
    runtime.run_until_decode_steps(5, strict_cohort_cutoff=True)
    records = cohort_samples(runtime)
    result = evaluate_native_window(runtime, records, NativeWindowFastConfig())
    assert not result["accepted"]
    assert result["reasons"] == ["native_multislot_disjoint_full_batch_samples_insufficient"]
    assert result["can_expand_sampling"]
    assert result["error_budget"]["native_lifecycle"]["seed_washout"]["qualified"]
    assert not result["metrics"]


def bandwidth_only_native_scenario(prompt_tokens=7):
    """New synthetic timing fixture, not a claim about formal HBM devices.

    It keeps the original native graph, Core, addresses, TP and paid PD.
    Explicit zero analytical phase makes lifecycle/composition assertions
    independent of the commercial-preset phase budget's expected rejection.
    No production configuration or ten-percent guard is changed.
    """
    from heterollm_sim.memory_types import DramConfig
    from heterollm_sim.physical_contract import parse_execution_physical_config

    scenario = lifecycle_scenario(8)
    if prompt_tokens != 7:
        scenario = replace(scenario,
            workload=replace(scenario.workload, requests=tuple(
                replace(row, prompt_tokens=row.prompt_tokens+prompt_tokens-7)
                for row in scenario.workload.requests)))
    components = []
    for component in scenario.hardware.components:
        raw = component.metadata.get("physical_memory_config")
        if raw:
            config = parse_execution_physical_config(raw)
            if isinstance(config, DramConfig):
                config = replace(config, control_mode="analytical", refresh_policy="disabled",
                    metadata={}, **{row.name: 0. for row in fields(config) if row.name.endswith("_ns")})
                component = replace(component,
                    metadata={**component.metadata, "physical_memory_config": config})
        components.append(component)
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)))


@pytest.mark.parametrize("prompt_tokens", (7, 32))
def test_real_b8_layer_sampling_reaches_window_budget_without_fabricated_prefix(prompt_tokens):
    from heterollm_sim.execution_control import ExecutionControl
    from heterollm_sim.native_layer_budget import evaluate_native_layer_window
    from heterollm_sim.native_layer_sampling import sample_native_layer_windows

    runtime, prepared = capture_first_native_query(bandwidth_only_native_scenario(prompt_tokens))
    options = NativeWindowFastConfig(sampling_unit="layer_windows")
    before = (runtime.now, runtime._completed_decode_steps, tuple(runtime.events))
    payload = sample_native_layer_windows(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], prepared["ready"], options, ExecutionControl())
    result = evaluate_native_layer_window(runtime, prepared["cohort"], prepared["cost"],
        prepared["stages"], payload, options)
    assert len(payload["records"]) == 2
    assert all(row["executed_task_count"] > 0 for row in payload["records"])
    assert payload["full_cohorts_committed"] == payload["visible_output_tokens"] == 0
    assert before == (runtime.now, runtime._completed_decode_steps, tuple(runtime.events))
    assert runtime.batches == []
    budget = result["error_budget"]
    assert budget["native_layer_sampling"]["accepted"]
    assert "query_decode_native_IR" in budget, result["reasons"]
    assert budget["query_decode_native_IR"]["qualified"]
    assert budget["native_lifecycle"]["physical_complete_cohort_samples"] == 0
    assert budget["native_lifecycle"]["seed_washout"]["qualified"]
    exposure = budget["native_wave_PD_exposure"]
    assert exposure["qualified"] and exposure["PD_cost_not_removed"]
    assert exposure["measurement_import_count_upper"] == 18
    assert {row["within_wave_import_count_upper"] for row in exposure["per_request"].values()} == {2}
    assert budget["visible_commit_cleanup_upper_ns"] == 0.
    assert budget["unsampled_seed_native_invariant_floor_ns"] > budget["native_full_batch_fixed_path_floor_ns"]/2
    assert budget["per_point_target"] == .1
    assert budget["effective_independent_target"] == .1/2.1
    assert set(result["metric_bounds"]) == {"measurement_duration_ns", "cluster_tokens_per_second",
        "mean_complete_request_tpot_ns", "mean_complete_request_ingress_ttft_ns"}
    assert len(result["request_forecasts"]) == 16
    assert all(row["request_id"] is None and not row["per_request_label_order_proven"]
               for row in result["request_forecasts"])
    assert not result["semantic_guards"]["physical_state_advanced"]
    if result["accepted"]:
        assert all(row["relative_error_budget"] <= .1/2.1
                   for row in result["metric_bounds"].values())
    else:
        assert result["reasons"] == [
            "native_layer_multislot_conditional_single_or_independent_pair_budget_exceeded"]
        assert not result["metrics"]
    # The fresh final-query batches reset the mixed seed contexts. Their
    # elapsed time and token lifetimes must stay in the same interval as the
    # later Decode batches, even when a wide interval honestly rejects.
    del runtime._native_layer_sampler
    actual = runtime.run()
    window = actual.decode_measurement
    complete = window["complete_request_tpot_ns"]
    observed = {
        "measurement_duration_ns": window["duration_ns"],
        "cluster_tokens_per_second": window["visible_tokens_per_second"],
        "mean_complete_request_tpot_ns": fmean(complete.values()),
        "mean_complete_request_ingress_ttft_ns": fmean(
            actual.request_metrics[key].ttft_ns for key in complete),
    }
    for key, value in observed.items():
        interval = result["metric_bounds"][key]
        assert interval["lower"] <= value <= interval["upper"], (key, value, interval)


def test_native_mmu_bound_keeps_per_task_extrema_when_context_returns_to_lower_corner():
    from heterollm_sim.native_layer_bounds import native_mmu_context_bounds
    from heterollm_sim.serving import _lower_cost

    runtime, prepared = capture_first_native_query(bandwidth_only_native_scenario())
    initial = prepared["cohort"]
    upper = replace(initial, cohort_id=initial.cohort_id+".upper",
        items=tuple(replace(item, context_tokens=13) for item in initial.items))
    cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, upper)
    stages, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
    assert not fallback
    stages = runtime._with_gpu_controller_stages(stages, upper)
    accumulator = {}
    native_mmu_context_bounds(runtime, initial, prepared["stages"], payload_maxima=accumulator)
    high = native_mmu_context_bounds(runtime, upper, stages, payload_maxima=accumulator)
    returned = native_mmu_context_bounds(
        runtime, initial, prepared["stages"], payload_maxima=accumulator)
    assert high["qualified"] and returned["qualified"]
    assert [row["full_payload_translation_bytes_upper"] for row in high["gpu_bounds"]] == [
        row["full_payload_translation_bytes_upper"] for row in returned["gpu_bounds"]]
    assert all(row["positive_candidate_stage_count"] > 0 for row in returned["gpu_bounds"])
    assert not runtime.batches


@pytest.mark.parametrize("change", ("unknown_L2_source", "changed_MMU_service"))
def test_native_mmu_bound_rejects_unknown_payload_or_non_native_controller(change):
    from collections.abc import Mapping
    from heterollm_sim.native_layer_bounds import native_mmu_context_bounds

    runtime, prepared = capture_first_native_query(bandwidth_only_native_scenario())
    stages, changed = [], False
    for stage in prepared["stages"]:
        tasks = []
        for task in stage.execution_tasks:
            md = task.metadata
            model = md.get("cost_model")
            if not isinstance(model, Mapping):
                model = md.get("native_cost_model", {})
            if not changed and change == "unknown_L2_source" and any(
                    demand.resource_id == "gpu0.l2" for demand in task.demands):
                task = replace(task, metadata={**md,
                    "cost_model": {**model, "model": "opaque_user_cost"}})
                changed = True
            elif (not changed and change == "changed_MMU_service"
                    and md.get("event_kind") == "gpu_mmu_tlb_batch"):
                task = replace(task, demands=(replace(task.demands[0],
                    service_ns=task.demands[0].service_ns+1.),))
                changed = True
            tasks.append(task)
        stages.append(replace(stage, execution_tasks=tuple(tasks)))
    assert changed
    reason = ("payload_source_not_bounded" if change == "unknown_L2_source"
              else "root_service_not_native")
    with pytest.raises(ValueError, match=reason):
        native_mmu_context_bounds(runtime, prepared["cohort"], tuple(stages))


@pytest.mark.parametrize("count,quanta,expected", (
    (193, (96, 128), 288), (96, (96, 128), 128), (0, (96, 128), 0),
))
def test_native_backing_payload_rounds_each_nondivisible_profile_quantum(count, quanta, expected):
    from heterollm_sim.native_layer_bounds import _native_backing_payload_upper

    assert _native_backing_payload_upper(count, quanta) == expected


def _mmu_corners():
    from heterollm_sim.native_layer_bounds import native_mmu_context_bounds
    from heterollm_sim.serving import _lower_cost

    runtime, prepared = capture_first_native_query(bandwidth_only_native_scenario())
    initial = prepared["cohort"]
    upper = replace(initial, cohort_id=initial.cohort_id+".upper",
        items=tuple(replace(item, context_tokens=13) for item in initial.items))
    cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, upper)
    stages, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
    assert not fallback
    stages = runtime._with_gpu_controller_stages(stages, upper)
    return native_mmu_context_bounds, runtime, prepared, (initial, upper), stages


def test_native_mmu_lower_bound_is_positive_and_contains_every_native_root():
    bounds, runtime, prepared, (initial, upper), stages = _mmu_corners()
    accumulator = {}
    low = bounds(runtime, initial, prepared["stages"], payload_maxima=accumulator)
    high = bounds(runtime, upper, stages, payload_maxima=accumulator)
    for proof, rows in ((low, prepared["stages"]), (high, stages)):
        assert proof["qualified"]
        services = {task.task_id: task.demands[0].service_ns for stage in rows
                    for task in stage.execution_tasks if task.metadata.get("event_kind") == "gpu_mmu_tlb_batch"}
        for task_id, bound in proof["task_bounds"].items():
            assert bound["lower_ns"] <= services[task_id] <= bound["upper_ns"]
    assert all(row["per_root_service_lower_ns"] > 0 for row in high["gpu_bounds"]
               if row["positive_candidate_stage_count"])
    assert all(row["full_payload_translation_bytes_lower"] <= row["observed_translation_bytes"]
               <= row["full_payload_translation_bytes_upper"] for row in high["gpu_bounds"])


def test_native_mmu_fixed_payload_that_changes_between_corners_loses_its_lower():
    bounds, runtime, prepared, (initial, _upper), _stages = _mmu_corners()
    accumulator = {}
    before = bounds(runtime, initial, prepared["stages"], payload_maxima=accumulator)
    fixed = next(key for key, values in accumulator["fixed_input_bytes"].items()
                 if accumulator["input_lower_bytes"].get(key))
    accumulator["fixed_input_bytes"][fixed].add(-1)
    after = bounds(runtime, initial, prepared["stages"], payload_maxima=accumulator)
    gpu = fixed[0]
    lower = {row["gpu_id"]: row["full_payload_translation_bytes_lower"] for row in after["gpu_bounds"]}
    old = {row["gpu_id"]: row["full_payload_translation_bytes_lower"] for row in before["gpu_bounds"]}
    assert accumulator["input_lower_bytes"][fixed] == 0 and accumulator["input_exact_upper_bytes"][fixed] is None
    assert lower[gpu] < old[gpu]
