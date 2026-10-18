"""K/V de-duplication cannot bypass unqualified workspace rounding bounds."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.fast_state import (
    _decode_program_budget, _fixed_decode_write_geometry, _project_program_allocations,
    project_fast_tail,
)
from heterollm_sim.hbf_staging import HbfStagingController, HbfStagingIO
from test_fast_state_projection import config, projected_runtime


def rows_for(widths, current, final):
    return [{"bytes_per_token": width,
             "completed_page_appends": final * width // 4096 - current * width // 4096}
            for width in widths]


def cohort(context, pages, **item_updates):
    item = dict(request_id="r0", phase="decode", token_count=1,
                context_tokens=context, kv_append_tokens=1, kv_materialized_tokens=1)
    item.update(item_updates)
    return NS(items=(NS(**item),), cost=NS(metadata={
        "physical_execution_scope": "persistent_live_kernel",
        "cost_duration_semantics": "live_cohort_span",
        "storage_traffic": {"pages_programmed": pages}}))


def budget_for(widths, *, prompt=13, prefix=20, remaining=37, workspace=3):
    current, final = prompt + prefix, prompt + prefix + remaining
    batches = [cohort(context, workspace + sum(
        (context + 1) * width // 4096 - context * width // 4096 for width in widths))
        for context in range(prompt, current)]
    result = _decode_program_budget(batches, rows_for(widths, current, final),
        request_id="r0", prompt_tokens=prompt, current_tokens=current,
        final_tokens=final, remaining=remaining)
    return result, batches


@pytest.mark.parametrize("widths", [(256, 256), (1024, 1024), (4096, 4096),
                                    (8192, 8192), (256, 1024, 8192)])
def test_phase_exact_kv_candidate_is_diagnostic_without_workspace_geometry(widths):
    result, _ = budget_for(widths)
    assert result["kv_program_lower_bound_verified"]
    assert not result["kv_program_deduction_available"]
    assert result["kv_program_deduction_unavailable_reason"] == "non_kv_write_geometry_unverified"
    assert result["observed_max_non_kv_program_pages_per_decode"] == 3
    future = sum(70 * width // 4096 - 33 * width // 4096 for width in widths)
    assert result["future_kv_completed_page_program_budget"] == future
    assert result["candidate_deduplicated_program_page_budget"] == 37 * 7 + future
    assert result["candidate_deduplicated_program_page_budget"] < result["legacy_program_page_guard_budget"]
    assert result["program_page_guard_budget"] == result["legacy_program_page_guard_budget"]
    assert result["observed_kv_completed_page_programs_deducted"] == 0
    assert not result["candidate_deduplicated_budget_used_for_acceptance"]


@pytest.mark.parametrize("phase", range(16))
def test_every_partial_page_phase_keeps_exact_whole_future_boundary_total(phase):
    result, _ = budget_for((256, 256), prompt=32 + phase, prefix=19, remaining=35)
    current = 51 + phase
    assert result["future_kv_completed_page_program_budget"] == 2 * (
        (current + 35) // 16 - current // 16)


@pytest.mark.parametrize("mutation,reason", [
    (lambda batches: setattr(batches[0].items[0], "phase", "prefill"), "decode_kv_phase_not_contiguous"),
    (lambda batches: batches.reverse(), "decode_kv_phase_not_contiguous"),
    (lambda batches: setattr(batches[0].items[0], "request_id", "other"), "decode_kv_phase_not_contiguous"),
    (lambda batches: setattr(batches[0].items[0], "kv_append_tokens", 0), "decode_kv_phase_not_contiguous"),
    (lambda batches: setattr(batches[0].items[0], "kv_materialized_tokens", 2), "decode_kv_phase_not_contiguous"),
    (lambda batches: batches[0].cost.metadata.pop("physical_execution_scope"), "program_counter_not_cohort_local"),
    (lambda batches: batches[0].cost.metadata.update(cost_duration_semantics="commit_gap"), "program_counter_not_cohort_local"),
    (lambda batches: batches[0].cost.metadata["storage_traffic"].update(pages_programmed=0), "cohort_program_count_below_kv_lower_bound"),
])
def test_unverified_observations_keep_original_guard(mutation, reason):
    _, batches = budget_for((8192, 8192))
    mutation(batches)
    result = _decode_program_budget(batches, rows_for((8192, 8192), 33, 70),
        request_id="r0", prompt_tokens=13, current_tokens=33, final_tokens=70, remaining=37)
    assert not result["kv_program_deduction_available"]
    assert result["kv_program_deduction_unavailable_reason"] == reason
    assert result["observed_kv_completed_page_programs_deducted"] == 0
    assert result["program_page_guard_budget"] == result["legacy_program_page_guard_budget"]


def test_missing_counter_and_empty_observations_are_not_measured_zero():
    _, batches = budget_for((256, 256))
    batches[0].cost.metadata["storage_traffic"].pop("pages_programmed")
    result = _decode_program_budget(batches, rows_for((256, 256), 33, 70),
        request_id="r0", prompt_tokens=13, current_tokens=33, final_tokens=70, remaining=37)
    assert not result["observed_program_counters_available"]
    assert not result["kv_program_deduction_available"]
    empty = _decode_program_budget((), rows_for((256, 256), 13, 70),
        request_id="r0", prompt_tokens=13, current_tokens=13, final_tokens=70, remaining=57)
    assert empty["observed_max_program_pages_per_decode"] == 0
    assert not empty["kv_program_deduction_available"]
    assert not empty["observed_program_counters_available"]


def test_multiple_owner_aggregate_subtracts_only_this_owners_kv():
    widths, other_widths = (256, 256), (8192, 8192)
    current, final = 33, 70
    batches = [cohort(context, 3 + sum(
        (context + 1) * width // 4096 - context * width // 4096
        for width in widths + other_widths)) for context in range(13, current)]
    result = _decode_program_budget(batches, rows_for(widths, current, final),
        request_id="r0", prompt_tokens=13, current_tokens=current,
        final_tokens=final, remaining=37)
    assert result["kv_program_lower_bound_verified"]
    assert result["observed_max_non_kv_program_pages_per_decode"] == 7
    assert result["observed_kv_completed_page_programs_verified"] == 4
    assert result["observed_kv_completed_page_programs_deducted"] == 0


@pytest.mark.parametrize("width", [256, 1024, 4096, 8192])
def test_real_staging_partial_completion_programs_are_exact_kv_lower_bound(width):
    configured = replace(config(), blocks_per_plane=64, capacity_bytes=8 * 1024**2)
    controller = HbfStagingController(configured, 64 * 4096)
    programs = []
    def write(address, count, timestamp):
        assert count == 4096
        programs.append(address)
        return timestamp
    identity = lambda address, count, timestamp: timestamp
    io = HbfStagingIO(identity, identity, identity, write,
                      lambda direction, count, timestamp: timestamp)
    prompt, prefix, remaining = 13, 20, 37
    for part in ("k", "v"):
        controller.write(part, 0, 0, prompt * width, 0., io)
    observations = []
    for context in range(prompt, prompt + prefix):
        before = len(programs)
        for part in ("k", "v"):
            controller.write(part, 0, context * width, width, 0., io)
        actual = len(programs) - before
        expected = 2 * ((context + 1) * width // 4096 - context * width // 4096)
        assert actual == expected
        observations.append(cohort(context, actual))
    current, final = prompt + prefix, prompt + prefix + remaining
    result = _decode_program_budget(observations, rows_for((width, width), current, final),
        request_id="r0", prompt_tokens=prompt, current_tokens=current,
        final_tokens=final, remaining=remaining)
    assert result["observed_max_non_kv_program_pages_per_decode"] == 0
    state = controller.snapshot()
    projection = _project_program_allocations(controller, result["candidate_deduplicated_program_page_budget"])
    assert controller.snapshot() == state
    before = len(programs)
    for context in range(current, final):
        for part in ("k", "v"):
            controller.write(part, 0, context * width, width, 0., io)
    actual_tail = programs[before:]
    assert len(actual_tail) == result["future_kv_completed_page_program_budget"]
    page_zero = [0] * configured.array_units
    from heterollm_sim.memory_mapping import map_nand_address
    for address in actual_tail:
        mapping = map_nand_address(configured, address)
        page_zero[mapping.unit] += int(mapping.page == 0)
    assert projection["page_zero_programs_by_bank"] == tuple(page_zero)
    assert controller.stats.get("gc_cycles", 0) == 0
    assert controller.stats.get("pressure_flushes", 0) == 0
    assert controller.stats.get("rmw_pages", 0) == 0


def test_projection_retains_gc_rmw_pressure_and_host_guards():
    runner, controller = projected_runtime()
    for stat in ("gc_cycles", "pressure_flushes", "rmw_pages"):
        controller.stats[stat] = 1
    controller.host_capacity_bytes = 2 * 4096
    before = controller.snapshot()
    result = project_fast_tail(runner)
    assert controller.snapshot() == before
    for reason in ("hbf_observed_gc_cycles", "hbf_observed_pressure_flushes",
                   "hbf_observed_rmw_pages", "hbf_host_tail_slots_may_require_pressure_flush"):
        assert reason + ":hbf0.memory" in result.safety_reasons
    assert "hbf_decode_program_observations_unavailable:hbf0.memory" in result.safety_reasons


def test_partial_workspace_cleanup_is_never_deducted_as_kv_programs():
    configured = replace(config(), blocks_per_plane=64, capacity_bytes=8 * 1024**2)
    controller = HbfStagingController(configured, 64 * 4096)
    programs = []
    identity = lambda address, count, timestamp: timestamp
    def write(address, count, timestamp):
        programs.append(address)
        return timestamp
    io = HbfStagingIO(identity, identity, identity, write,
                     lambda direction, count, timestamp: timestamp)
    for part in ("k", "v"):
        controller.write(part, 0, 0, 13 * 256, 0., io)
    observations = []
    for context in range(13, 33):
        before = len(programs)
        for part in ("k", "v"):
            controller.write(part, 0, context * 256, 256, 0., io)
        controller.write("temporary", context, 0, 128, 0., io)
        controller.flush("temporary", context, 0., io)
        controller.release("temporary", context)
        observations.append(cohort(context, len(programs) - before))
    result = _decode_program_budget(observations, rows_for((256, 256), 33, 70),
        request_id="r0", prompt_tokens=13, current_tokens=33, final_tokens=70, remaining=37)
    assert result["kv_program_lower_bound_verified"]
    assert result["observed_max_non_kv_program_pages_per_decode"] == 1
    assert result["observed_kv_completed_page_programs_verified"] == 4
    assert result["observed_kv_completed_page_programs_deducted"] == 0
    # The future bound is at least the actual allocation count even when
    # each future cleanup also explicitly flushes an incomplete workspace.
    expected = _project_program_allocations(controller, result["candidate_deduplicated_program_page_budget"])
    before = len(programs)
    for context in range(33, 70):
        for part in ("k", "v"):
            controller.write(part, 0, context * 256, 256, 0., io)
        controller.write("temporary", context, 0, 128, 0., io)
        controller.flush("temporary", context, 0., io)
        controller.release("temporary", context)
    from heterollm_sim.memory_mapping import map_nand_address
    actual_zero = sum(map_nand_address(configured, address).page == 0
                      for address in programs[before:])
    assert expected["page_zero_programs"] >= actual_zero


def test_projection_preserves_free_block_reuse_rejection():
    runner, controller = projected_runtime()
    physical = runner._execution_resource_kernel.physical_runtime
    core = physical.runtimes["hbf0.memory"].core
    from heterollm_sim.memory_mapping import map_nand_address
    block = controller._free_blocks.ranges[0].start
    mapping = map_nand_address(controller.config, controller._address(block, 0))
    core._set_program_frontier(core._block_key(mapping), controller.config.pages_per_block)
    before, frontiers = controller.snapshot(), dict(core._next_program_page)
    result = project_fast_tail(runner)
    assert "hbf_free_block_reuse_may_require_erase:hbf0.memory" in result.safety_reasons
    assert controller.snapshot() == before
    assert core._next_program_page == frontiers


def test_actual_physical_cohort_counter_verifies_kv_but_not_all_workspaces():
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.planner import TopologyAwareBatchCostProvider, _compilation_scope
    from heterollm_sim.serving import _OnlineRuntime, compile_serving_plan
    from test_fast_prefix_runtime import runtime
    bootstrap = bootstrap_control_plane(runtime(output=35).plan.scenario,
                                         capture_physical_details=False)
    scenario = bootstrap.scenario
    lowerer = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    with _compilation_scope(scenario, lowerer._compilation_context):
        runner = _OnlineRuntime(compile_serving_plan(scenario), lowerer,
            execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
        assert not runner.run_until_decode_steps(2)
        projection = project_fast_tail(runner, detailed_decode_steps=2)
    assert projection.hbf_devices
    for device in projection.hbf_devices:
        assert device["kv_program_lower_bound_verified"]
        assert not device["kv_program_deduction_available"]
        assert device["kv_program_deduction_unavailable_reason"] == "non_kv_write_geometry_unverified"
        assert device["observed_program_counters_available"]
        assert device["ordinary_erase_serial_time_guard_ns"] == device["legacy_ordinary_erase_serial_time_guard_ns"]
        assert device["program_page_guard_budget"] == device["legacy_program_page_guard_budget"]


def test_independently_rounded_context_workspaces_cannot_use_aggregate_ceil():
    # Observed two separate full 100-page buffers. A 0.1% context increase
    # takes each to 101 pages, not ceil(200 * 1.001) == 201 in aggregate.
    from math import ceil
    assert 2 * ceil(100 * 1.001) == 202
    assert ceil(200 * 1.001) == 201
    prompt, prefix, remaining = 2999, 1, 3
    result = _decode_program_budget((cohort(prompt, 200),),
        rows_for((256, 256), 3000, 3003), request_id="r0",
        prompt_tokens=prompt, current_tokens=3000, final_tokens=3003,
        remaining=remaining)
    assert result["context_scaled_non_kv_program_pages_per_decode"] == 201
    assert not result["kv_program_deduction_available"]
    assert result["kv_program_deduction_unavailable_reason"] == "non_kv_write_geometry_unverified"
    assert result["program_page_guard_budget"] == result["legacy_program_page_guard_budget"]
    assert not result["candidate_deduplicated_budget_used_for_acceptance"]


def test_qualified_fixed_workspace_geometry_uses_constant_residual_not_context_ceil():
    _, batches = budget_for((256, 256))
    result = _decode_program_budget(batches, rows_for((256, 256), 33, 70),
        request_id="r0", prompt_tokens=13, current_tokens=33, final_tokens=70,
        remaining=37, write_geometry={"qualified": True})
    assert result["kv_program_deduction_available"]
    assert result["fixed_non_kv_program_pages_per_decode"] == 3
    assert result["program_page_guard_budget"] == 37 * 3 + 4
    assert result["program_page_guard_budget"] < result["legacy_program_page_guard_budget"]


def formal_blueprint_runtime(model="qwen2_5-0_5b"):
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.control_plane_planner import plan_runtime_placement
    from heterollm_sim.planner import TopologyAwareBatchCostProvider, _execution_layers
    scenario = build_scan_scenario(model, "HBF", 65536, 10240)
    scenario = plan_runtime_placement(scenario).apply(scenario)
    state = NS(cached_tokens=65664, spec=scenario.workload.requests[0])
    runner = NS(lowerer=TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True),
                plan=NS(scenario=scenario), _state_values=(state,))
    rows = [{"buffer_id": f"kv:scan-request:{layer.layer_id}:rank0:tp0:pp0:{part}",
             "physical_owner": "hbf0.memory", "component_id": "hbf0"}
            for layer in _execution_layers(scenario) for part in ("k", "v")]
    return runner, rows


@pytest.mark.parametrize("model", ["qwen2_5-0_5b", "qwen2_5-1_5b", "qwen2_5-3b", "qwen2_5-7b"])
def test_all_four_formal_models_have_read_only_fixed_geometry_certificate(model, monkeypatch):
    from heterollm_sim import planner
    runner, rows = formal_blueprint_runtime(model)
    before = runner._state_values[0].cached_tokens
    def forbidden(*args, **kwargs):
        pytest.fail("geometry qualification must never execute a schedule or physical preview")
    monkeypatch.setattr(planner, "execute_cost_schedule", forbidden)
    result = _fixed_decode_write_geometry(runner, rows, 75775)
    assert result["qualified"], result
    assert result["backend_kind"] == "standard_topology_analytic_dense"
    assert result["non_kv_write_span_count"] > 0
    assert result["non_kv_write_span_signature"]
    assert result["partial_host_slots_by_owner"]["hbf0.memory"] == 2 * result["non_kv_write_span_count"]
    assert not result["physical_state_advanced"] and not result["physical_io_executed"]
    assert runner._state_values[0].cached_tokens == before
    assert not hasattr(runner, "_execution_resource_kernel")


def test_unadmitted_flash_and_changed_workspace_geometry_keep_original_guard(monkeypatch):
    from heterollm_sim import planner
    runner, rows = formal_blueprint_runtime()
    original = planner._lower_serving_cohort
    def unadmitted(*args, **kwargs):
        result = original(*args, **kwargs)
        tasks = tuple(replace(task, metadata={**task.metadata, "fusion_enabled": False})
            if task.metadata.get("fusion_group") == "flash_attention" else task
            for task in result.schedule.tasks)
        return replace(result, schedule=replace(result.schedule, tasks=tasks))
    monkeypatch.setattr(planner, "_lower_serving_cohort", unadmitted)
    result = _fixed_decode_write_geometry(runner, rows, 75775)
    assert not result["qualified"]
    assert result["reason"] == "terminal_flash_attention_not_admitted_for_every_layer"
    def variable(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1].items[0].context_tokens > 70000:
            tasks = []
            changed = False
            for task in result.schedule.tasks:
                accesses = []
                for access in task.metadata.get("memory_accesses", ()):
                    if not changed and access.get("operation") == "write" and access.get("physical_owner") == "hbf0.memory":
                        access = {**access, "byte_count": access["byte_count"] + 1}
                        changed = True
                    accesses.append(access)
                tasks.append(replace(task, metadata={**task.metadata, "memory_accesses": tuple(accesses)})
                             if accesses else task)
            result = replace(result, schedule=replace(result.schedule, tasks=tuple(tasks)))
        return result
    monkeypatch.setattr(planner, "_lower_serving_cohort", variable)
    result = _fixed_decode_write_geometry(runner, rows, 75775)
    assert not result["qualified"]
    assert result["reason"] == "non_kv_write_or_alias_geometry_changes_with_context"


def test_real_small_bf16_prefix_uses_new_bound_without_advancing_any_physical_state():
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.ir import build_model_graph_from_layer_specs
    from heterollm_sim.planner import (TopologyAwareBatchCostProvider, _compilation_scope,
                                      _execution_layers)
    from heterollm_sim.serving import _OnlineRuntime, compile_serving_plan
    from test_fast_prefix_runtime import runtime
    source = runtime(output=35).plan.scenario
    # The rebuilt BF16 graph needs twice the INT8 matrix storage, too.
    layers = tuple(replace(layer, dtype="bf16", quantization=None, weight_bytes=1179648)
                   for layer in _execution_layers(source))
    graph = build_model_graph_from_layer_specs(source.model.name, layers,
        vocabulary_size=1024, max_sequence_length=262144)
    source = replace(source, model=replace(source.model, graph=graph),
        placement=replace(source.placement, kv_policy=replace(source.placement.kv_policy, dtype="bf16")))
    bootstrap = bootstrap_control_plane(source, capture_physical_details=False)
    scenario = bootstrap.scenario
    lowerer = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    with _compilation_scope(scenario, lowerer._compilation_context):
        runner = _OnlineRuntime(compile_serving_plan(scenario), lowerer,
            execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
        assert not runner.run_until_decode_steps(20)
        physical = runner._execution_resource_kernel.physical_runtime
        controllers = {owner: value["controller"].snapshot()
                       for owner, value in physical.hbf_staging.items()}
        allocations = {owner: dict(allocator._allocations)
                       for owner, allocator in physical.allocators.items()}
        clocks = {owner: device.clock_ns for owner, device in physical.runtimes.items()}
        cached, batches = runner._state_values[0].cached_tokens, tuple(runner.batches)
        result = project_fast_tail(runner, detailed_decode_steps=20)
    assert result.hbf_devices
    for device in result.hbf_devices:
        assert device["non_kv_write_geometry_qualified"]
        assert device["kv_program_deduction_available"]
        assert device["program_page_guard_budget"] < device["legacy_program_page_guard_budget"]
        assert device["ordinary_erase_serial_time_guard_ns"] <= device["legacy_ordinary_erase_serial_time_guard_ns"]
        assert device["fixed_non_kv_program_pages_per_decode"] is not None
    assert controllers == {owner: value["controller"].snapshot()
                           for owner, value in physical.hbf_staging.items()}
    assert allocations == {owner: dict(allocator._allocations)
                           for owner, allocator in physical.allocators.items()}
    assert clocks == {owner: device.clock_ns for owner, device in physical.runtimes.items()}
    assert runner._state_values[0].cached_tokens == cached
    assert tuple(runner.batches) == batches
