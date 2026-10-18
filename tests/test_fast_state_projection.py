"""Forecast guards inspect actual FTL state and never advance it."""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from heterollm_sim import fast_state
from heterollm_sim.fast_state import _project_program_allocations, project_fast_tail
from heterollm_sim.hbf_staging import HbfStagingController, HbfStagingIO
from heterollm_sim.memory_allocator import PhysicalAddressAllocator
from heterollm_sim.memory_types import make_hbf_config
from heterollm_sim.nand_core import NandCore
from test_fast_prefix_runtime import runtime


def config(**updates):
    return replace(make_hbf_config(channels=2, host_channels=2, planes_per_lun=2,
        blocks_per_plane=8, pages_per_block=8, capacity_bytes=1024**2,
        protocol_mode="ocp_hbf_0_7", queue_scope="host_channel", planes_independent=True,
        host_full_duplex=True, read_cache_pages_per_bank=2,
        page_bytes=4096, host_granularity_bytes=64), **updates)


def io():
    return HbfStagingIO(*(lambda _address, _count, timestamp: timestamp for _ in range(4)),
                        lambda _direction, _count, timestamp: timestamp)


@pytest.mark.parametrize("prefix,count", [(0, 0), (0, 1), (0, 3), (0, 4), (0, 5),
                                         (0, 127), (1, 1), (1, 4),
                                         (7, 79), (31, 129), (95, 1000)])
def test_scalar_allocator_round_jumps_match_every_real_allocation(prefix, count):
    configured = config()
    controller = HbfStagingController(configured, 8 * 4096)
    callbacks = io()
    for index in range(prefix):
        controller._allocate(0., callbacks, ("buffer", 0, index))
    before = controller.snapshot()
    result = _project_program_allocations(controller, count)
    assert controller.snapshot() == before
    expected = [0] * configured.array_units
    expected_zero = [0] * configured.array_units
    allocated = 0
    # Stay before collection; the guard reports the rest as unallocated.
    available = (len(controller._free_blocks) * configured.pages_per_block
        + sum(configured.pages_per_block - controller._used[block] for block in controller._active))
    for index in range(min(count, available)):
        (block, page), _ = controller._allocate(0., callbacks, ("tail", 0, index))
        expected[block % configured.array_units] += 1
        expected_zero[block % configured.array_units] += int(page == 0)
        allocated += 1
    assert result["program_pages_by_bank"] == tuple(expected)
    assert result["page_zero_programs_by_bank"] == tuple(expected_zero)
    assert result["page_zero_programs"] == sum(expected_zero)
    assert result["unallocated_program_pages"] == count - allocated


def projected_runtime():
    runner = runtime(output=35)
    runner.run_until_decode_steps(2)
    physical = runner._execution_resource_kernel.physical_runtime
    component = runner.plan.scenario.hardware.get_component("hbf0")
    configured = component.metadata["physical_memory_config"]
    controller = HbfStagingController(configured, 65536)
    physical.hbf_staging["hbf0.memory"] = {"controller": controller}
    physical.runtimes["hbf0.memory"] = SimpleNamespace(core=NandCore(configured, capture_details=False))
    allocator = PhysicalAddressAllocator(configured.effective_capacity_bytes)
    physical.allocators["hbf0.memory"] = allocator
    first = project_fast_tail(runner)
    for row in first.kv_buffers:
        allocator.allocate(row["buffer_id"], row["final_bytes"])
    return runner, controller


def test_projection_counts_k_and_v_separately_and_keeps_ftl_unchanged():
    runner, controller = projected_runtime()
    before = controller.snapshot()
    allocations = dict(runner._execution_resource_kernel.physical_runtime.allocators["hbf0.memory"]._allocations)
    result = project_fast_tail(runner)
    assert result.remaining_decode_steps == 32
    assert result.current_cached_tokens == 9 and result.final_cached_tokens == 41
    assert len(result.kv_buffers) == 2
    for row in result.kv_buffers:
        assert row["completed_page_appends"] == row["final_bytes"] // 4096 - row["current_bytes"] // 4096
        assert row["final_tail_bytes"] == row["final_bytes"] % 4096
    assert controller.snapshot() == before
    assert runner._execution_resource_kernel.physical_runtime.allocators["hbf0.memory"]._allocations == allocations


def test_default_offload_ratio_without_target_is_inert():
    runner, _ = projected_runtime()
    scenario = runner.plan.scenario
    scenario = replace(scenario, placement=replace(scenario.placement,
        kv_policy=replace(scenario.placement.kv_policy,
                          offload_component=None, offload_ratio=1.)))
    runner.plan = replace(runner.plan, scenario=scenario,
        kv_policy=replace(runner.plan.kv_policy, offload_component=None, offload_ratio=1.))
    result = project_fast_tail(runner)
    assert len(result.kv_buffers) == 2
    assert not any("offload" in reason for reason in result.safety_reasons)
    assert result.to_metadata()["physical_state_advanced"] is False
    assert result.to_metadata()["physical_tail_transactions_recorded"] is False


@pytest.mark.parametrize("stat", ["gc_cycles", "pressure_flushes", "rmw_pages"])
def test_existing_hbf_discontinuities_are_reported_as_guards(stat):
    runner, controller = projected_runtime()
    controller.stats[stat] = 1
    result = project_fast_tail(runner)
    assert "hbf_observed_{}:hbf0.memory".format(stat) in result.safety_reasons
    assert result.safe is False


def test_insufficient_host_tail_slots_require_detailed_execution():
    runner, controller = projected_runtime()
    controller.host_capacity_bytes = 2 * 4096
    result = project_fast_tail(runner)
    assert "hbf_host_tail_slots_may_require_pressure_flush:hbf0.memory" in result.safety_reasons


def test_reused_free_block_is_detected_without_erasing_it():
    configured = config()
    controller = HbfStagingController(configured, 8 * 4096)
    core = NandCore(configured, capture_details=False)
    from heterollm_sim.memory_mapping import map_nand_address
    mapping = map_nand_address(configured, controller._address(0, 0))
    core._set_program_frontier(core._block_key(mapping), configured.pages_per_block)
    frontiers = dict(core._next_program_page)
    result = _project_program_allocations(controller, 1, core)
    assert result["reused_blocks_requiring_erase"] == 1
    assert core._next_program_page == frontiers


def test_fresh_page_zero_programs_also_erase_under_ocp():
    configured = config()
    controller = HbfStagingController(configured, 8 * 4096)
    core = NandCore(configured, capture_details=False)
    result = _project_program_allocations(controller, 1, core)
    assert result["new_blocks_opened"] == configured.array_units
    assert result["page_zero_programs"] == 1
    assert result["reused_blocks_requiring_erase"] == 0


def test_projection_charges_normal_page_zero_erase_service_without_claiming_stall():
    runner, controller = projected_runtime()
    result = project_fast_tail(runner)
    row = result.hbf_devices[0]
    expected = row["program_allocation_guard"]["page_zero_programs"]
    assert expected > 0
    assert row["ordinary_erase_operations_guard_budget"] == expected
    assert row["ordinary_erase_serial_time_guard_ns"] == expected * controller.config.block_erase_ns
    assert row["future_physical_state"] == "not_advanced"


def test_no_detailed_prefix_cannot_be_reported_as_calibrated_tail():
    runner = runtime(output=35)
    result = project_fast_tail(runner)
    assert "exact_prefill_and_first_token_required" in result.safety_reasons
    assert result.safe is False


def test_caller_cannot_label_an_unexecuted_step_as_detailed():
    runner, _ = projected_runtime()
    result = project_fast_tail(runner, detailed_decode_steps=3)
    assert "detailed_decode_step_count_does_not_match_runtime" in result.safety_reasons


@pytest.mark.parametrize("rows,expected", [((256,), 16), ((1024,), 4),
                                        ((4096,), 1), ((1024, 256, 128), 32)])
def test_static_page_period_uses_joint_geometry_before_execution(monkeypatch, rows, expected):
    rank = SimpleNamespace(component_id="gpu0")
    parallel = SimpleNamespace(tp_degree=1, ranks_for_layer=lambda layer: (rank,))
    scenario = SimpleNamespace(hardware=SimpleNamespace(get_component=lambda name:
        SimpleNamespace(normalized_kind="hbf")))
    monkeypatch.setattr(fast_state, "_parallel_plan", lambda scenario: parallel)
    monkeypatch.setattr(fast_state, "_execution_layers", lambda scenario: rows)
    monkeypatch.setattr(fast_state, "_kv_tensor_bytes", lambda scenario, layer, tp, count: layer)
    monkeypatch.setattr(fast_state, "_kv_components", lambda *args: ("hbf0", None, 0.))
    assert fast_state.hbf_kv_page_period(scenario) == expected


def program_observations(*, copies=(6, 6, 6, 6, 6, 6, 4, 4), workspace=688):
    owner, prompt = "hbf0.memory", 262144
    rows = [{"physical_owner": owner, "bytes_per_token": 512,
             "completed_page_appends": 10} for _ in range(72)]
    batches = []
    for index, copied in enumerate(copies):
        context = prompt + index
        kv = 72 * ((context + 1) // 8 - context // 8)
        metadata = {
            "physical_execution_scope": "persistent_live_kernel",
            "cost_duration_semantics": "live_cohort_span",
            "storage_traffic": {"pages_programmed": workspace + copied + kv},
            "storage_gc_observations": {owner: {
                "available": True, "semantics": "cohort_local_staging_counter_deltas",
                "gc_cycles": int(copied > 0), "gc_copied_pages": copied,
                "pressure_flushes": 0, "rmw_pages": 0,
                "recorded_task_count": 2, "expected_task_count": 2}}}
        item = SimpleNamespace(request_id="r0", phase="decode", token_count=1,
            context_tokens=context, kv_append_tokens=1, kv_materialized_tokens=1)
        batches.append(SimpleNamespace(items=(item,), cost=SimpleNamespace(metadata=metadata)))
    return batches, rows


def program_projection(batches, rows):
    return fast_state._decode_program_budget(batches, rows, request_id="r0",
        prompt_tokens=262144, current_tokens=262144 + len(batches),
        final_tokens=262144 + len(batches) + 80, remaining=80,
        write_geometry={"qualified": True})


def test_verified_gc_copies_are_not_charged_again_as_future_ordinary_programs():
    batches, rows = program_observations()
    before = deepcopy((batches, rows))
    result = program_projection(batches, rows)
    parts = result["program_page_budget_breakdown"]
    assert parts["observed_raw_program_pages"] == 5620
    assert parts["verified_gc_copy_program_pages"] == 44
    assert parts["gc_copy_program_pages_deducted"] == 44
    assert parts["compulsory_kv_program_pages_deducted"] == 72
    assert parts["residual_non_kv_program_pages_upper"] == 5504
    assert result["observed_residual_non_kv_program_pages_upper_by_decode"] == (688,) * 8
    assert result["fixed_non_kv_program_pages_per_decode"] == 688
    assert result["program_page_guard_budget"] == 80 * 688 + 720
    assert result["gc_copy_program_deduction_available"]
    assert result["gc_copy_program_deduction_unavailable_reason"] is None
    assert parts["future_gc_copy_allocation_and_service"] == "charged_separately_by_gc_projection"
    assert parts["physical_state_advanced"] is False
    assert [batch.cost.metadata for batch in batches] == [batch.cost.metadata for batch in before[0]]
    assert rows == before[1]
    assert program_projection(batches, rows) == result


@pytest.mark.parametrize("mutation", [
    lambda metadata: metadata.pop("storage_gc_observations"),
    lambda metadata: metadata.update(storage_gc_observations=[]),
    lambda metadata: metadata["storage_gc_observations"].update(other={}),
    lambda metadata: metadata["storage_gc_observations"].update(
        other=metadata["storage_gc_observations"].pop("hbf0.memory")),
    lambda metadata: metadata["storage_gc_observations"].update({"hbf0.memory": []}),
    *(lambda metadata, key=key, value=value:
        metadata["storage_gc_observations"]["hbf0.memory"].update({key: value})
      for key, value in (("available", False), ("semantics", "lifetime_totals"),
        ("gc_copied_pages", None), ("gc_copied_pages", True), ("gc_copied_pages", -1),
        ("gc_copied_pages", 100000), ("gc_cycles", 0), ("gc_cycles", True),
        ("recorded_task_count", 0), ("recorded_task_count", 1),
        ("expected_task_count", None), ("pressure_flushes", 1), ("rmw_pages", 1))),
])
def test_unavailable_or_ambiguous_gc_ledger_retains_conservative_program_upper(mutation):
    batches, rows = program_observations()
    mutation(batches[0].cost.metadata)
    before = deepcopy([batch.cost.metadata for batch in batches])
    result = program_projection(batches, rows)
    assert result["kv_program_deduction_available"]  # Existing IR proof is unchanged.
    assert not result["gc_copy_program_deduction_available"]
    assert result["gc_copy_program_deduction_unavailable_reason"]
    assert result["observed_gc_copied_pages_verified"] is None
    assert result["observed_gc_copied_pages_deducted"] == 0
    assert result["fixed_non_kv_program_pages_per_decode"] == 694
    assert result["program_page_guard_budget"] == 80 * 694 + 720
    assert result["program_page_budget_breakdown"]["verified_gc_copy_program_pages"] is None
    assert result["program_page_budget_breakdown"]["residual_scope"] == (
        "ordinary_non_kv_plus_undeducted_unknown_gc_programs")
    assert [batch.cost.metadata for batch in batches] == before


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].pop("physical_owner"),
    lambda rows: rows[0].update(physical_owner="other"),
    lambda rows: rows[0].update(physical_owner=[]),
])
def test_gc_deduction_requires_one_known_physical_owner(mutation):
    batches, rows = program_observations()
    mutation(rows)
    result = program_projection(batches, rows)
    assert not result["gc_copy_program_deduction_available"]
    assert result["gc_copy_program_deduction_unavailable_reason"] == "program_gc_physical_owner_unverified"
    assert result["fixed_non_kv_program_pages_per_decode"] == 694


def test_gc_and_compulsory_kv_cannot_exceed_raw_physical_programs():
    batches, rows = program_observations()
    metadata = batches[-1].cost.metadata
    metadata["storage_gc_observations"]["hbf0.memory"]["gc_copied_pages"] = 700
    result = program_projection(batches, rows)
    assert result["gc_copy_program_deduction_unavailable_reason"] == "program_gc_and_kv_exceed_cohort_programs"
    assert not result["gc_copy_program_deduction_available"]
    assert result["fixed_non_kv_program_pages_per_decode"] == 694


def test_verified_zero_gc_changes_no_existing_program_budget():
    batches, rows = program_observations(copies=(0,) * 8)
    zero = program_projection(batches, rows)
    for batch in batches:
        batch.cost.metadata.pop("storage_gc_observations")
    unknown = program_projection(batches, rows)
    for field in ("program_page_guard_budget", "fixed_non_kv_program_pages_per_decode",
                  "future_kv_completed_page_program_budget", "legacy_program_page_guard_budget"):
        assert zero[field] == unknown[field]
    assert zero["observed_gc_copied_pages_verified"] == 0
    assert unknown["observed_gc_copied_pages_verified"] is None


def test_real_live_gc_copy_programs_are_deducted_once_without_advancing_ftl():
    from test_hbf_gc_write_segments import _fill_fragmented
    from test_hbf_page_state import PageIO, config as page_config

    controller = HbfStagingController(page_config(channels=1, host_channels=1,
        pages_per_block=32, blocks_per_plane=12), 4 * 4096)
    hardware = PageIO()
    _fill_fragmented(controller, hardware)
    batches, _ = program_observations(copies=(0,) * 8, workspace=80)
    prompt = controller._block_count - 1
    for index, batch in enumerate(batches):
        metadata = batch.cost.metadata
        batch.items[0].context_tokens = prompt + index
        before = dict(controller.stats)
        controller.write("workspace", batch.items[0].context_tokens, 0, 80 * 4096,
                         0., hardware.io)
        controller.release("workspace", batch.items[0].context_tokens)
        controller.write("K", 0, (prompt + index) * 4096, 4096, 0., hardware.io)
        copied = controller.stats["gc_copied_pages"] - before.get("gc_copied_pages", 0)
        cycles = controller.stats["gc_cycles"] - before.get("gc_cycles", 0)
        metadata["storage_gc_observations"]["hbf0.memory"].update(
            gc_cycles=cycles, gc_copied_pages=copied)
        metadata["storage_traffic"]["pages_programmed"] = (
            controller.stats["flash_write_bytes"] - before["flash_write_bytes"]) // 4096
    state, events = controller.snapshot(), list(hardware.events)
    result = fast_state._decode_program_budget(batches,
        [{"physical_owner": "hbf0.memory", "bytes_per_token": 4096,
          "completed_page_appends": 80}], request_id="r0", prompt_tokens=prompt,
        current_tokens=prompt + 8, final_tokens=prompt + 88, remaining=80,
        write_geometry={"qualified": True})
    assert result["observed_gc_copied_pages_verified"] > 0
    assert result["fixed_non_kv_program_pages_per_decode"] == 80
    assert result["program_page_guard_budget"] == 80 * 80 + 80
    assert controller.snapshot() == state
    assert hardware.events == events
