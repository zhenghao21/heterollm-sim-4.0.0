"""PP owns layer-local weights/KV and the real transmitted hidden edge."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

import native_toy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exploring_hbf_native_config import build_native_comparison_scenario
from heterollm_sim import planner
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_allocator import AllocationError
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.serving import BatchCohort, BatchItem, compile_serving_plan


def pp_scenario(hardware="HBM4", *, layers=4, tp=2, ep=1, batch=4, microbatches=1):
    scenario = build_native_comparison_scenario(
        "pp-capacity-toy", hardware, "SHORT", tp=tp, pp=2, ep=ep,
        batch_size=batch, model_override=native_toy._moe_toy_model(layers),
        prompt_override=7, output_override=4, warmup_override=2, measurement_override=4)
    scenario = replace(scenario, placement=replace(scenario.placement,
        parallel=replace(scenario.placement.parallel, pipeline_microbatches=microbatches)))
    scenario = native_toy._analytical_toy_hardware(scenario, .00001)
    decision = plan_runtime_placement(scenario)
    assert decision.fully_placed
    return decision.apply(scenario)


def pp_cohort(scenario):
    ep = scenario.placement.parallel.ep_degree
    return BatchCohort("cohort-pp-capacity", "decode", 0., tuple(
        BatchItem(row.request_id, "decode", 1, 7, main_tokens=1,
                  draft_tokens=0, verifier_tokens=0, committed_tokens=1,
                  logit_tokens=1, ep_group=index % ep)
        for index, row in enumerate(scenario.workload.requests[:scenario.workload.scheduler.max_num_seqs])))


@pytest.mark.parametrize("hardware,ep", [("HBM4", 1), ("HBF+", 1), ("HBM4", 2)])
def test_weights_and_kv_live_only_on_their_actual_pipeline_stage(hardware, ep):
    scenario = pp_scenario(hardware, layers=5, ep=ep)
    parallel = planner._parallel_plan(scenario)
    policy = compile_serving_plan(scenario).kv_policy
    shards = scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]
    expected_bytes = {}
    for layer in planner._execution_layers(scenario):
        ranks = parallel.ranks_for_layer(layer)
        allowed = {rank.rank for rank in ranks}
        for tensor, rows in shards.items():
            if tensor.startswith(layer.layer_id + "."):
                assert {row["rank_id"] for row in rows} == allowed
        expected_owners = tuple(
            scenario.placement.metadata["rank_kv_components"][str(rank.rank)]
            for rank in ranks)
        assert policy.kv_layer_component_shards[layer.layer_id] == expected_owners
        local_bytes = planner._kv_bytes_per_token(scenario, layer, parallel.tp_degree)
        for owner in expected_owners:
            expected_bytes[owner] = expected_bytes.get(owner, 0) + local_bytes * policy.tokens_per_page
    assert policy.component_bytes_per_page == expected_bytes
    assert {row["rank_id"] for row in shards["embedding_weights"]} == {
        rank.rank for rank in parallel.ranks_for_stage(0)}
    assert {row["rank_id"] for row in shards["lm_head_weights"]} == {
        rank.rank for rank in parallel.ranks_for_stage(1)}
    assert policy.bytes_per_page == sum(expected_bytes[owner] for owner in policy.ep_group_components[0])
    if hardware == "HBF+":
        assert all(owner.startswith("hbf") for owner in expected_bytes)
        assert all(rank.memory_component_id.startswith("sram") for rank in parallel.ranks)
    else:
        weights = {}
        for rows in shards.values():
            for row in rows:
                owner = row["storage_component_id"]
                weights[owner] = weights.get(owner, 0) + row["physical_bytes"]
        assert policy.capacity_bytes_by_component == {
            owner: scenario.hardware.get_component(owner).capacity_bytes - weights[owner]
            for owner in expected_bytes}


def test_one_small_stage_limits_admission_instead_of_summing_device_capacity():
    scenario = pp_scenario(layers=5)
    first = compile_serving_plan(scenario).kv_policy
    owner = "hbm3"
    component = scenario.hardware.get_component(owner)
    weights = component.capacity_bytes - first.capacity_bytes_by_component[owner]
    capacity = weights + 3 * first.component_bytes_per_page[owner]
    physical = require_physical_memory_config(component)
    component = replace(component, capacity_bytes=capacity, metadata={**component.metadata,
        "physical_memory_config": replace(physical, capacity_bytes=capacity,
            rows_per_bank=(capacity + physical.row_bytes - 1) // physical.row_bytes)})
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        component if row.component_id == owner else row for row in scenario.hardware.components)))
    policy = compile_serving_plan(scenario).kv_policy
    assert policy.capacity_pages == 3
    assert policy.bottleneck_component == owner
    assert policy.capacity_bytes_by_component[owner] == 3 * policy.component_bytes_per_page[owner]


@pytest.mark.parametrize("hardware", ["HBM4", "HBF+"])
@pytest.mark.parametrize("layers", [2, 4])
def test_boundary_moves_complete_tp_hidden_and_reuses_producer_and_consumers(hardware, layers):
    scenario = pp_scenario(hardware, batch=2, layers=layers)
    tasks = planner.compile_serving_cohort_schedule(scenario, pp_cohort(scenario)).tasks
    boundary = [task for task in tasks if task.metadata.get("event_kind") == "pipeline_activation"
                and task.metadata.get("memory_accesses")]
    assert len(boundary) == 4
    for task in boundary:
        access, = task.metadata["memory_accesses"]
        assert access["byte_count"] == 2 * 32 * 2
        key = access["buffer_id"], access["allocation_generation"]
        related = [other for other in tasks if other not in boundary
                   for row in other.metadata.get("memory_accesses", ())
                   if (row.get("buffer_id"), row.get("allocation_generation")) == key]
        assert related
        if access["operation"] == "read":
            assert any(other.metadata.get("event_kind") in ("moe_residual", "mlp_residual")
                       for other in related)
        else:
            assert any(other.metadata.get("event_kind") == "input_norm_reduce" for other in related)
            assert any(other.metadata.get("event_kind") in ("attention_residual", "fused_residual_norm")
                       for other in related)


@pytest.mark.parametrize("hardware", ["HBM4", "HBF+"])
@pytest.mark.parametrize("microbatches", [1, 2])
def test_native_allocator_holds_received_hidden_until_real_consumers_then_releases(hardware, microbatches):
    scenario = pp_scenario(hardware, batch=2 * microbatches, microbatches=microbatches)
    schedule = planner.compile_serving_cohort_schedule(scenario, pp_cohort(scenario))
    kernel = UnifiedEventKernel.from_closed_graph(schedule.tasks, capture_physical_details=False,
        resource_capacities=schedule.resource_capacities, resource_owners=schedule.resource_owners)
    destinations = []
    active_peak = {}
    while (event := kernel.step()) is not None:
        task = event.task
        for owner, allocator in kernel.physical_runtime.allocators.items():
            if owner.startswith("sram"):
                active_peak[owner] = max(active_peak.get(owner, 0),
                    sum(row.size_bytes for row in allocator.allocations() if row.alias_of is None))
        if task.metadata.get("event_kind") != "pipeline_activation":
            continue
        for access in task.metadata.get("memory_accesses", ()):
            if access["operation"] != "write":
                continue
            owner = task.metadata["physical_owner"]
            key = owner, access["buffer_id"], access["allocation_generation"]
            # A completed receive is still owned by real next-stage consumers.
            assert kernel._physical_allocation_uses[key] > 0
            assert kernel.physical_runtime.allocators[owner].get_allocation(key[1], key[2]) is not None
            destinations.append(key)
    assert len(destinations) == 2 * microbatches
    by_storage = {}
    for owner, buffer, generation in destinations:
        by_storage.setdefault((owner, buffer), set()).add(generation)
    assert all(len(generations) == microbatches for generations in by_storage.values())
    kernel._reclaim_physical_allocations(kernel.makespan_ns)
    assert all(kernel.physical_runtime.allocators[owner].get_allocation(buffer, generation) is None
               for owner, buffer, generation in destinations)
    if hardware == "HBF+":
        assert len(active_peak) == 4
        assert all(128 <= peak <= 320_000_000 for peak in active_peak.values())


def test_pipeline_boundary_cannot_overrun_a_rank_sram():
    scenario = pp_scenario("HBF+", batch=2)
    tasks = planner.compile_serving_cohort_schedule(scenario, pp_cohort(scenario)).tasks
    boundary = [task for task in tasks if task.metadata.get("event_kind") == "pipeline_activation"]
    ids = {task.task_id for task in boundary}
    short = []
    for task in boundary:
        metadata = task.metadata
        if metadata.get("physical_memory_config") is not None:
            raw = metadata["physical_memory_config"]
            raw = replace(raw, capacity_bytes=64) if not isinstance(raw, dict) else {**raw, "capacity_bytes": 64}
            metadata = {**metadata, "physical_memory_config": raw}
        short.append(replace(task, dependencies=tuple(dep for dep in task.dependencies if dep in ids),
                             metadata=metadata))
    kernel = UnifiedEventKernel.from_closed_graph(short, capture_physical_details=False)
    with pytest.raises((AllocationError, ValueError), match="capacity|allocate|exceeds"):
        while kernel.step() is not None:
            pass


@pytest.mark.parametrize("hardware", ["HBM4", "HBF+"])
def test_tp_dense_residual_stays_live_through_ffn_until_final_residual_read(hardware):
    scenario = pp_scenario(hardware, batch=4, microbatches=2)
    dense_layers = {layer.layer_id for layer in planner._execution_layers(scenario) if not layer.is_moe}
    schedule = planner.compile_serving_cohort_schedule(scenario, pp_cohort(scenario))
    kernel = UnifiedEventKernel.from_closed_graph(schedule.tasks, capture_physical_details=False,
        resource_capacities=schedule.resource_capacities, resource_owners=schedule.resource_owners)
    residuals, consumed = {}, set()
    ffn_observed = set()
    while (event := kernel.step()) is not None:
        metadata = event.task.metadata
        if metadata.get("layer_id") not in dense_layers:
            continue
        owner = metadata.get("physical_owner")
        for row in metadata.get("memory_accesses", ()):
            if ".activation.attention_residual:" not in row.get("buffer_id", ""):
                continue
            key = owner, row["buffer_id"], row["allocation_generation"]
            if row["operation"] == "write":
                assert metadata["event_kind"] == "fused_residual_norm"
                assert kernel._physical_allocation_uses[key] > 0
                residuals[key] = metadata["layer_id"]
            elif metadata.get("event_kind") == "mlp_residual":
                assert key in residuals
                assert kernel.physical_runtime.allocators[owner].get_allocation(key[1], key[2]) is not None
                consumed.add(key)
        if ".mlp_down" in event.task.name:
            for key, layer in residuals.items():
                if key[0] == owner and layer == metadata["layer_id"] and key not in consumed:
                    assert kernel.physical_runtime.allocators[owner].get_allocation(key[1], key[2]) is not None
                    ffn_observed.add(key)
    assert len(residuals) == 8
    assert consumed == ffn_observed == set(residuals)
    kernel._reclaim_physical_allocations(kernel.makespan_ns)
    assert all(kernel.physical_runtime.allocators[owner].get_allocation(buffer, generation) is None
               for owner, buffer, generation in residuals)
