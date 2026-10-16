"""Deferred cohorts publish one exact allocation and stage metadata contract."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim import planner
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.scalable_serving import TaskExecutionRecord, execute_cost_schedule
from heterollm_sim.serving import BatchCohort, BatchItem


@pytest.mark.parametrize("medium", ["HBM", "HBF"])
def test_deferred_metadata_matches_materialized_physical_graph(medium, monkeypatch):
    from test_deferred_physical_preview import scenario, strict_staging_scenario

    configured = (scenario(prompt=4, count=1, output=3) if medium == "HBM" else
                  strict_staging_scenario(prompt=4, output=3, all_hbf=True, capacity=65536))
    item = BatchItem("request", "decode", 1, 4, main_tokens=1, draft_tokens=0,
                     verifier_tokens=1, committed_tokens=1, logit_tokens=1)
    cohort = BatchCohort("decode", "decode", 0., (item,))
    with planner._compilation_scope(configured):
        lowering = planner._lower_serving_cohort(configured, cohort)
        schedule = lowering.schedule
        original_metadata = deepcopy(tuple(task.metadata for task in schedule.tasks))
        promoted = planner._promote_physical_allocation_extents(schedule.tasks)
        materialized = execute_cost_schedule(replace(schedule, tasks=promoted),
            defer_physical_execution=True, retain_task_metadata=False)
        groups = lowering.extra_metadata["operator_invocation_groups"]
        expected = planner._compact_execution_stages(configured, materialized.execution_records, groups)
        metadata = dict(zip((task.task_id for task in schedule.tasks),
                            planner._physical_task_metadata(schedule.tasks)))
        deferred = execute_cost_schedule(schedule, defer_physical_execution=True,
                                         retain_task_metadata=False)
        serialized = []
        serialize = planner._serialize_execution_stage_tasks
        def counted(records, *args, **kwargs):
            serialized.extend(record.task_id for record in records)
            return serialize(records, *args, **kwargs)
        monkeypatch.setattr(planner, "_serialize_execution_stage_tasks", counted)
        actual = planner._compact_execution_stages(configured, deferred.execution_records, groups,
                                                    physical_metadata=metadata)
    assert actual == expected
    assert planner._prepare_execution_stage_rows(actual[0]) == planner._prepare_execution_stage_rows(expected[0])
    assert serialized and len(serialized) == len(set(serialized))
    assert tuple(task.metadata for task in schedule.tasks) == original_metadata
    assert replace(materialized, execution_records=()) == replace(deferred, execution_records=())


def test_stage_record_order_retains_zero_time_dependencies_and_rejects_invalid_graphs():
    def record(name, dependencies=()):
        return TaskExecutionRecord(name, 0., 0., TaskCategory.COMPUTE, dependencies,
                                   {"event_kind": "marker"}, ())
    first, second = record("z"), record("a", ("z", "external"))
    ordered = planner._ordered_execution_stage_records((second, first))
    assert ordered == (first, second)
    facts = planner._serialize_execution_stage_tasks(ordered, ("request", "request"))
    assert tuple(row["task_id"] for row in facts) == ("z", "a")
    assert facts[1]["dependencies"] == ("z",)
    assert facts[1]["request_ids"] == ("request",)
    with pytest.raises(ValueError, match="duplicate"):
        planner._ordered_execution_stage_records((first, first))
    with pytest.raises(ValueError, match="cycle"):
        planner._ordered_execution_stage_records((record("a", ("b",)), record("b", ("a",))))


def test_fused_allocation_metadata_keeps_aliases_generations_and_inputs():
    def task(name, offset, size, generation=3, **fields):
        return TaskSpec(name, "request", name, TaskCategory.MEMORY,
            demands=(ResourceDemand("memory", 0., bytes_moved=size),), metadata={
                "physical_memory_config": {"capacity_bytes": 4096, "burst_bytes": 64},
                "physical_owner": "memory", "memory_accesses": ({
                    "operation": "read", "buffer_id": "shared", "offset_bytes": offset,
                    "byte_count": size, "allocation_generation": generation, **fields},)})
    tasks = (task("short", 0, 64, alias_of="base", alias_generation=2, alias_offset_bytes=64),
             task("long", 64, 128), task("new_generation", 0, 32, generation=4))
    before = deepcopy(tasks)
    metadata = planner._physical_task_metadata(tasks)
    promoted = planner._promote_physical_allocation_extents(tasks)
    assert tuple(task.metadata for task in promoted) == metadata
    assert tasks == before
    declarations = tuple(row["physical_allocations"][0] for row in metadata)
    assert tuple(row["size_bytes"] for row in declarations) == (192, 192, 32)
    assert declarations[0]["alias_of"] == "base"
    assert declarations[0]["alias_generation"] == 2
    assert declarations[0]["alias_offset_bytes"] == 64
