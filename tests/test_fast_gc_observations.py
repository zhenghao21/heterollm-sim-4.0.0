"""Forecast maintenance observations are cohort deltas, not lifetime totals."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.event_kernel import KernelEvent
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.planner import _summarize_hbf_gc_observations

from test_hbf_staging_runtime import setup, task


def record(name, staging=None, *, owners=("flash",)):
    metadata = {"physical_execution_by_owner": {
        owner: {"kind": "HBF", "pages_programmed": 1} for owner in owners}}
    if staging is not None:
        metadata["hbf_host_staging"] = staging
    return SimpleNamespace(task_id=name, metadata=metadata)


def test_observations_sum_each_task_owner_once_without_host_or_link_duplication():
    first = record("a", {"flash": {"gc_cycles": 2, "gc_copied_pages": 3}})
    first.metadata["physical_execution_by_owner"].update(
        host={"kind": "DDR"}, wire={"kind": "LINK"})
    second = record("b", {"flash": {"gc_cycles": 1, "pressure_flushes": 2},
                           "flash2": {"gc_copied_pages": 4, "rmw_pages": 5}},
                    owners=("flash", "flash2"))
    observed = _summarize_hbf_gc_observations((first, second, first))
    assert observed["flash"] == {"gc_cycles": 3, "gc_copied_pages": 3,
        "pressure_flushes": 2, "rmw_pages": 0, "available": True,
        "recorded_task_count": 2, "expected_task_count": 2,
        "semantics": "cohort_local_staging_counter_deltas"}
    assert observed["flash2"]["gc_copied_pages"] == 4
    assert observed["flash2"]["rmw_pages"] == 5
    assert set(observed) == {"flash", "flash2"}


def test_recorded_sparse_zero_is_explicit_but_missing_hbf_ledger_is_unavailable():
    observed = _summarize_hbf_gc_observations((
        record("zero", {"flash": {"logical_read_bytes": 4096}}),
        record("unknown", {"other": {}}, owners=("unknown",)),
    ))
    assert observed["flash"]["gc_cycles"] == observed["flash"]["gc_copied_pages"] == 0
    assert observed["flash"]["available"] is True
    assert observed["unknown"]["available"] is False
    assert observed["unknown"]["gc_cycles"] is None
    assert observed["unknown"]["recorded_task_count"] == 0
    assert observed["other"]["available"] is True  # A HOST-only staged tail still has a ledger.


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "2"])
def test_invalid_counter_does_not_become_a_safe_zero(value):
    observed = _summarize_hbf_gc_observations((
        record("good", {"flash": {"gc_cycles": 2}}),
        record("bad", {"flash": {"gc_copied_pages": value}}),
    ))["flash"]
    assert observed["available"] is False
    assert observed["gc_cycles"] is observed["gc_copied_pages"] is None
    assert observed["recorded_task_count"] == 1
    assert observed["expected_task_count"] == 2


def test_live_metadata_retains_staging_delta_without_observing_new_physical_work():
    demand = ResourceDemand("wire", 2.0)
    authored = TaskSpec("a", "request", "a", TaskCategory.MEMORY, demands=(demand,),
        metadata=record("a", {"flash": {"gc_cycles": 3, "gc_copied_pages": 7}}).metadata)
    live = LiveCohortMetrics()
    live.observe(KernelEvent(authored, 4., 6., 4., 4., (demand,), {}, {"wire": 0}))
    first = live.metadata()
    second = live.metadata()
    assert first["storage_gc_observations"] == second["storage_gc_observations"]
    assert first["storage_gc_observations"]["flash"]["gc_cycles"] == 3
    assert first["storage_gc_observations"]["flash"]["gc_copied_pages"] == 7
    assert first["storage_traffic"]["pages_programmed"] == 1
    assert first["physical_execution_scope"] == "persistent_live_kernel"


def test_previous_prefill_gc_is_not_recharged_as_the_decode_cohort_observation():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    now = 0.

    def write(name, pages, generation):
        nonlocal now
        result = resolve_physical_task(replace(task(config, binding, "write", count=pages * 4096,
            buffer=name, generation=generation, size=pages * 4096), task_id=name), runtime, now)
        now = result.metadata["physical_completion_ns"]
        return result

    def release(name, generation):
        runtime.allocators["flash"].release(name, generation)
        runtime.release_staging_buffer("flash", name, generation, arrival_ns=now)

    write("fill", 60, 1)
    release("fill", 1)
    previous = write("prefill", 4, 2)
    release("prefill", 2)
    current = write("decode", 4, 3)
    controller = runtime.hbf_staging["flash"]["controller"]
    assert controller.stats["gc_cycles"] == 2
    assert _summarize_hbf_gc_observations((previous,))["flash"]["gc_cycles"] == 1
    observed = _summarize_hbf_gc_observations((current,))["flash"]
    assert observed["gc_cycles"] == 1
    assert observed["gc_copied_pages"] == 0
    assert observed["pressure_flushes"] == observed["rmw_pages"] == 0
    assert observed["available"] is True


def test_hbm_only_cohort_does_not_invent_hbf_observations():
    hbm = record("hbm", owners=())
    hbm.metadata["physical_execution_by_owner"] = {"hbm": {"kind": "HBM"}}
    assert _summarize_hbf_gc_observations((hbm,)) == {}


def test_real_bf16_prefix_connects_empty_gc_projection_observations_and_budget(monkeypatch):
    """Exercise the production chain on a tiny model, with no physical suffix."""
    import math
    import pickle

    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.data_motion import lower_physical_allocations
    from heterollm_sim.dram_core import DramCore
    from heterollm_sim.fast_simulation import (
        SamplingWindow, conditional_error_budget, maintenance_error_envelope,
    )
    from heterollm_sim.fast_state import hbf_kv_page_period, project_fast_tail
    from heterollm_sim.hbf_resident_init import _inventory
    from heterollm_sim.ir import build_model_graph_from_layer_specs
    from heterollm_sim.nand_core import NandCore
    from heterollm_sim.planner import (
        TopologyAwareBatchCostProvider, _compilation_scope, _execution_layers,
    )
    from heterollm_sim.serving import _OnlineRuntime, compile_serving_plan
    from test_fast_prefix_runtime import runtime

    source = runtime(output=257).plan.scenario
    layers = tuple(replace(layer, dtype="bf16", quantization=None, weight_bytes=1179648)
                   for layer in _execution_layers(source))
    graph = build_model_graph_from_layer_specs(source.model.name, layers,
        vocabulary_size=1024, max_sequence_length=262144)
    hbf = source.hardware.get_component("hbf0")
    hbf = replace(hbf, metadata={**hbf.metadata, "host_write_staging": {
        **hbf.metadata["host_write_staging"], "capacity_bytes": 4 * 1024**2}})
    source = replace(source, model=replace(source.model, graph=graph),
        hardware=replace(source.hardware, components=tuple(
            hbf if component.component_id == "hbf0" else component
            for component in source.hardware.components)),
        workload=replace(source.workload, requests=tuple(
            replace(request, prompt_tokens=2048) for request in source.workload.requests)),
        placement=replace(source.placement, kv_policy=replace(source.placement.kv_policy, dtype="bf16")))
    bootstrap = bootstrap_control_plane(source, capture_physical_details=False)
    scenario, kernel = bootstrap.scenario, bootstrap.kernel
    physical = kernel.physical_runtime
    owner = "hbf0.memory"
    controller = physical.hbf_staging[owner]["controller"]
    _, _, bindings, _ = _inventory(scenario)
    binding = bindings[owner]
    pages = len(controller._free_blocks) * controller.config.pages_per_block
    # Submit and then release real garbage before the measured request.
    # Neither the free-pool pointer nor the NAND program frontiers are reset.
    seed = lower_physical_allocations(TaskSpec("gc-seed", "seed", "initial garbage",
        TaskCategory.MEMORY, metadata={
            "physical_memory_config": controller.config, "physical_owner": owner,
            "physical_memory_configs": {owner: controller.config,
                binding["host_owner"]: binding["host_config"]},
            "hbf_host_staging_by_owner": {owner: binding},
            "memory_accesses": ({"operation": "write", "physical_owner": owner,
                "buffer_id": "gc-seed", "byte_count": pages * 4096, "offset_bytes": 0,
                "allocation_generation": 1, "allocation_size_bytes": pages * 4096},)}))
    seed = resolve_physical_task(seed, physical, bootstrap.makespan_ns)
    origin = seed.metadata["physical_completion_ns"]
    physical.allocators[owner].release("gc-seed", 1)
    physical.release_staging_buffer(owner, "gc-seed", 1, arrival_ns=origin)
    assert not controller._free_blocks and not controller._active
    assert not controller._owners or all(not owners for owners in controller._owners.values())

    lowerer = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    with _compilation_scope(scenario, lowerer._compilation_context):
        runner = _OnlineRuntime(compile_serving_plan(scenario), lowerer,
            execution_kernel=kernel, runtime_origin_ns=origin)
        period = hbf_kv_page_period(scenario)
        window = SamplingWindow(4, 2 * period, period, period)
        assert window.total < 257
        assert not runner.run_until_decode_steps(window.total)
        assert controller.stats["gc_cycles"] > 0
        decode = [batch for batch in runner.batches if batch.kind == "decode"]
        assert len(decode) == window.total
        assert all(batch.cost.metadata["storage_gc_observations"][owner]["available"]
                   for batch in decode)

        def physical_state():
            # Inspect raw Python/native state without publishing numeric
            # ownership or invoking a transactional snapshot of the device.
            return pickle.dumps((controller.snapshot(), vars(physical.timeline),
                tuple((name, device.clock_ns, vars(device.core))
                      for name, device in physical.runtimes.items()),
                tuple((name, dict(allocator._allocations))
                      for name, allocator in physical.allocators.items())))

        before = physical_state()
        events, batches = tuple(runner.events), tuple(runner.batches)
        cutoff = runner._completed_decode_steps
        live_core_ids = {id(device.core) for device in physical.runtimes.values()}
        original_nand, original_dram = NandCore.submit, DramCore.submit
        def guard_live_submit(original):
            def submit(core, *args, **kwargs):
                if id(core) in live_core_ids:
                    pytest.fail("a read-only GC budget must not execute live physical work or the suffix")
                return original(core, *args, **kwargs)
            return submit
        monkeypatch.setattr(NandCore, "submit", guard_live_submit(original_nand))
        monkeypatch.setattr(DramCore, "submit", guard_live_submit(original_dram))
        projected = project_fast_tail(runner, detailed_decode_steps=window.total)
        assert projected.safe, projected.safety_reasons
        device = projected.hbf_devices[0]
        guard = device["gc_projection"]
        assert guard["qualified"], guard
        assert guard["zero_copy_proven"]
        assert guard["empty_victim_reservoir"] >= guard["cycles_upper"] > 0
        assert guard["copied_pages_upper"] == 0
        media = guard["temporary_read_media_proof"]
        assert media["applied"], media
        assert media["physical_state_advanced"] is False
        assert media["potential_nand_read_pages_per_decode"] + media[
            "proven_host_only_read_pages_per_decode"] == media["rounded_read_pages_per_decode"]
        assert media["charged_nand_read_pages_per_decode"] == media["potential_nand_read_pages_per_decode"]
        remaining = projected.remaining_decode_steps
        envelope = maintenance_error_envelope(runner, projected, window, remaining,
            projected.final_cached_tokens / projected.current_cached_tokens)
        assert envelope["gc_budget_required"]
        assert envelope["gc_observations_available"], envelope
        assert envelope["gc_prefix_mapping_stable"]
        terms = ("prefix_gc_bias_guard_ns", "prefix_write_placement_bias_guard_ns",
            "prefix_read_placement_bias_guard_ns", "future_gc_erase_serial_guard_ns",
            "future_gc_non_erase_serial_guard_ns", "future_write_placement_serial_guard_ns",
            "future_read_placement_serial_guard_ns", "future_gc_carry_in_serial_guard_ns")
        assert all(math.isfinite(envelope[name]) and envelope[name] >= 0 for name in terms)
        assert envelope["future_gc_erase_serial_guard_ns"] > 0
        assert envelope["future_write_placement_serial_guard_ns"] > 0
        assert envelope["future_read_placement_serial_guard_ns"] > 0
        # Deliberately impossible timing endpoints reject rather than
        # manufacturing a suffix or relaxing the conditional interval.
        budget = conditional_error_budget(1., 1., 1., 0., envelope)
        assert budget["conditional_relative_error_budget"] is None
        assert budget["budget_unavailable_reason"] is not None
        assert physical_state() == before
        assert tuple(runner.events) == events and tuple(runner.batches) == batches
        assert runner._completed_decode_steps == cutoff < 256
        assert runner._state_values[0].finished_ns is None
