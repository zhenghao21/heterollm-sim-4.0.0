"""Streaming lowering declares complete buffer bounds before dispatch."""
from dataclasses import replace

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext, register_physical_allocations
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.planner import (
    compile_scenario,
    compile_streaming_scenario,
    iter_request_task_chunks,
)
from heterollm_sim.reference import build_reference_scenario


@pytest.mark.parametrize("mtp_enabled", (False, True))
def test_streaming_allocations_cover_later_chunks_and_requests(mtp_enabled):
    scenario = build_reference_scenario()
    requests = tuple(
        replace(scenario.workload.requests[0], request_id=request_id,
                prompt_tokens=prompt_tokens, output_tokens=5)
        for request_id, prompt_tokens in (("first-1", 1), ("second-1", 8))
    )
    scenario = replace(scenario, workload=replace(
        scenario.workload, requests=requests,
        mtp=scenario.workload.mtp if mtp_enabled else None,
    ))
    schedule = compile_streaming_scenario(scenario)
    tasks = tuple(task for request in requests
                  for chunk in iter_request_task_chunks(schedule, request)
                  for task in chunk.tasks)
    assert tasks == compile_scenario(scenario).tasks

    # Registration uses one runtime across every chunk and request. In MTP
    # mode a later verifier has a larger activation extent than prefill.
    runtime = PhysicalRuntimeContext(capture_details=False)
    physical_tasks = tuple(task for task in tasks
                           if task.metadata.get("physical_memory_config") is not None)
    for task in physical_tasks:
        assert task.metadata["physical_allocations"]
        register_physical_allocations(task, runtime)
    assert runtime.allocators
    declarations = {
        request.request_id: {
            row["buffer_id"]: row["generation"]
            for task in physical_tasks if task.request_id == request.request_id
            for row in task.metadata["physical_allocations"]
        }
        for request in requests
    }
    first_rows, second_rows = (declarations[request.request_id] for request in requests)
    temporary = "read:operator=host_cohort_pack:ordinal=0"
    assert first_rows[temporary] > 0
    assert first_rows[temporary] != second_rows[temporary]
    shared_weights = {
        row["buffer_id"] for task in physical_tasks
        for row in task.metadata["physical_allocations"]
        if row.get("allocation_arena") == "resident" and "attention_weights" in row["buffer_id"]
    }
    assert shared_weights
    assert all(first_rows[buffer_id] == second_rows[buffer_id] == 0
               for buffer_id in shared_weights)

    # Exercise the actual dispatch boundary that previously failed because
    # the streamed CPU memory task had no explicit allocation declarations.
    first = replace(physical_tasks[0], dependencies=())
    kernel = UnifiedEventKernel.from_closed_graph((first,))
    event = kernel.step()
    assert event is not None
    assert event.task.metadata["physical_execution_by_owner"]
    assert not kernel.has_active_tasks


def test_full_and_streamed_physical_dispatch_have_identical_event_times():
    from heterollm_sim.control_plane_planner import plan_runtime_placement
    from heterollm_sim.engine import simulate_schedule
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    from heterollm_sim.streaming_des import execute_incremental_schedule

    scenario = build_reference_scenario()
    model = ModelSpec(name="stream-dispatch", graph=build_model_graph_from_layer_specs(
        "stream-dispatch", (LayerSpec("dense0", "dense", 32, 64, 4,
                                     kv_heads=2, dtype="fp16"),),
        vocabulary_size=64, max_sequence_length=128,
    ))
    requests = tuple(replace(scenario.workload.requests[0], request_id=f"stream-{index}",
                             prompt_tokens=prompt, output_tokens=3)
                     for index, prompt in enumerate((1, 2)))
    scenario = replace(scenario, model=model,
        placement=replace(scenario.placement, model_name=model.name,
                          parallel=replace(scenario.placement.parallel,
                                           layer_to_stage={"dense0": 0})),
        workload=replace(scenario.workload, requests=requests, mtp=None))
    scenario = plan_runtime_placement(scenario).apply(scenario)

    full = simulate_schedule(compile_scenario(scenario))
    streamed = execute_incremental_schedule(compile_streaming_scenario(scenario)).trace
    assert [(task.task_id, task.start_ns, task.end_ns) for task in full.tasks] == [
        (task.task_id, task.start_ns, task.end_ns) for task in streamed.tasks]
    assert full.makespan_ns == streamed.makespan_ns
