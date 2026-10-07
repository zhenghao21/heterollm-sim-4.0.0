"""Live stages retain source accesses rather than cached physical results."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import planner as p
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.kernel_memory import bind_l2_invocation
from heterollm_sim.scalable_serving import execute_cost_schedule
from heterollm_sim.serving import BatchCohort, BatchItem, _OnlineRuntime, _live_task_metadata
from test_dram_traffic_summary import _config
from test_embedding_traffic_audit import embedding_scenario
from test_source_f32_attention_storage import case as source_case, layer_tasks


def test_stage_retains_unresolved_physical_input_and_zero_demand_tasks():
    task = TaskSpec(task_id="physical", request_id="r", name="physical", category=TaskCategory.MEMORY,
        demands=(ResourceDemand("dram", 0),), metadata={
            "physical_memory_config": _config(), "physical_owner": "dram",
            "memory_access": {"operation": "read", "address": 0, "byte_count": 64,
                              "physical_owner": "dram"}})
    summary = execute_cost_schedule(SimpleNamespace(tasks=(task,), resource_capacities={}, resource_owners={}),
                                    retain_task_metadata=False)
    record = summary.execution_records[0]
    assert record.original_task is task
    assert any(d.service_ns > 0 for d in record.demands)
    rows = p._execution_task_facts(summary.execution_records, ("r",))
    assert len(rows) == 1
    assert rows.prepared_tasks[0].demands == task.demands
    assert rows[0]["resource_demands"][0]["service_ns"] == 0
    assert rows[0]["metadata"]["memory_access"] == task.metadata["memory_access"]
    assert "physical_execution" not in rows[0]["metadata"]
    assert rows[0]["category"] == TaskCategory.MEMORY.value


def test_l2_invocation_rebinds_physical_descriptor_and_alias_together():
    metadata = {"stateful_l2": {"invocation_buffers": ("@invocation:x",),
                              "accesses": ({"buffer_id": "@invocation:x"},)},
                "memory_accesses": ({"buffer_id": "@invocation:x", "operation": "write"},),
                "physical_allocations": ({"buffer_id": "slice", "alias_of": "@invocation:x"},),
                "output_buffer_id": "@invocation:x"}
    bound = bind_l2_invocation(metadata, "cohort-1")
    expected = "cohort-1:@invocation:x"
    assert bound["stateful_l2"]["accesses"][0]["buffer_id"] == expected
    assert bound["memory_accesses"][0]["buffer_id"] == expected
    assert bound["physical_allocations"][0]["alias_of"] == expected
    assert bound["output_buffer_id"] == expected
    assert metadata["memory_accesses"][0]["buffer_id"] == "@invocation:x"


def _drain(kernel, tasks):
    kernel.add_tasks(tasks)
    kernel._index_physical_allocation_uses(tasks)
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
    for event in events:
        kernel.release_completed(event.task.task_id)
    return events


@pytest.mark.parametrize("source", [None, "qwen3_0_6b_f16", "qwen3_8_27b_mixed", "qwen3_0_6b_f16_complete"])
def test_original_dag_and_stages_match_across_prefill_and_two_decodes(source):
    complete_source = source == "qwen3_0_6b_f16_complete"
    scenario = source_case(source.removesuffix("_complete")) if source else embedding_scenario()
    if complete_source:
        from heterollm_sim.ir import build_model_graph_from_layer_specs
        # Keep actual source tensor shapes, full vocabulary and host output;
        # only the decoder stack is reduced to its first real layer.
        model = scenario.model
        graph = build_model_graph_from_layer_specs(model.name, (p._execution_layers(scenario)[0],),
            architecture=model.architecture, vocabulary_size=model.vocabulary_size,
            max_sequence_length=model.max_sequence_length, embedding_weight_bytes=model.embedding_weight_bytes,
            output_weight_bytes=model.output_weight_bytes,
            tie_word_embeddings=model.graph.attributes["tie_word_embeddings"],
            output_head_dtype="fp16", metadata=model.graph.attributes["metadata"])
        scenario = replace(scenario, model=replace(model, graph=graph))
    kernels = [UnifiedEventKernel(resource_capacities=p._scenario_resource_capacities(scenario),
                                 resource_owners=p._scenario_resource_owners(scenario),
                                 capture_physical_details=False) for _ in range(2)]
    for index, (phase, tokens, prior) in enumerate((("prefill", 4, 0), ("decode", 1, 4), ("decode", 1, 5))):
        cohort = BatchCohort("cohort-" + str(index), phase, 0,
            (BatchItem(scenario.workload.requests[0].request_id, phase, tokens, prior,
                       kv_append_tokens=tokens, kv_materialized_tokens=tokens, logit_tokens=1),))
        if source and not complete_source:
            _, tasks = layer_tasks(scenario, tokens, prior, request_id=cohort.cohort_id)
            group_id = "source.group.0"
            tasks = tuple(replace(task, metadata={**task.metadata,
                "operator_invocation_group_id": group_id}) for task in tasks)
            schedule = SimpleNamespace(tasks=tasks, resource_capacities=p._scenario_resource_capacities(scenario),
                                       resource_owners=p._scenario_resource_owners(scenario))
            groups = ({"group_id": group_id, "request_ids": cohort.request_ids},)
        else:
            with p._compilation_scope(scenario):
                lowering = p._lower_serving_cohort(scenario, cohort)
            schedule = replace(lowering.schedule, tasks=p._promote_physical_allocation_extents(lowering.schedule.tasks))
            groups = lowering.extra_metadata["operator_invocation_groups"]
        preview = execute_cost_schedule(schedule, retain_task_metadata=False)
        rows, error = p._compact_execution_stages(scenario, preview.execution_records,
                                                  groups)
        assert error is None
        stages = p._trusted_execution_stages(rows)
        assert stages is not None
        namespace = "serving.test." + str(index)
        direct_tasks = tuple(replace(task, earliest_start_ns=kernels[0].makespan_ns,
                            metadata=_live_task_metadata(task.metadata, namespace)) for task in schedule.tasks)
        stage_tasks, _ = _OnlineRuntime._stage_task_specs(stages,
                            {stage.stage_id: kernels[1].makespan_ns for stage in stages}, namespace)
        staged_names = {task.name for task in stage_tasks}
        omitted = [task for task in direct_tasks if task.task_id not in staged_names]
        assert all(not task.demands and task.metadata.get("physical_memory_config") is None for task in omitted)
        direct = _drain(kernels[0], direct_tasks)
        replayed = _drain(kernels[1], stage_tasks)
        assert kernels[0].makespan_ns == pytest.approx(kernels[1].makespan_ns, rel=0, abs=1e-6)
        direct_traffic = p._summarize_dram_task_traffic(tuple(e.task for e in direct))
        stage_traffic = p._summarize_dram_task_traffic(tuple(e.task for e in replayed))
        for key in ("logical_read_bytes", "logical_write_bytes", "physical_read_bytes", "physical_write_bytes"):
            assert direct_traffic[key] == stage_traffic[key]
