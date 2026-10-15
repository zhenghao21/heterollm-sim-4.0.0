"""Memory reports consume the explicit fields emitted by current producers."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.memory_types import Operation, TransactionResult
from heterollm_sim.contracts import (
    ResourceDemand, RunManifest, SIMULATION_SCHEMA_VERSION, TaskCategory, TaskSpec,
)
from heterollm_sim.engine import ScheduleIR
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.planner import _serving_invocation_groups
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import _memory_direction_rows
from heterollm_sim.scalable_serving import compact_execution_metadata, execute_cost_schedule
from heterollm_sim.serving import BatchCohort, BatchItem, _OnlineRuntime
from heterollm_sim.streaming_des import _CriticalPathAccumulator


@pytest.fixture
def invocation():
    cohort = BatchCohort("cohort", "prefill", 0.0, (
        BatchItem("request", "prefill", 3, 0, main_tokens=3,
                  draft_tokens=0, verifier_tokens=0, committed_tokens=0,
                  logit_tokens=1),
    ))
    groups = _serving_invocation_groups(build_reference_scenario(), cohort)
    return cohort, tuple(group.audit_metadata() for group in groups)


@pytest.mark.parametrize("serialized", [False, True])
def test_execution_plan_uses_current_group_rows_and_kv_ranges(invocation, serialized):
    cohort, groups = invocation
    if serialized:
        groups = json.loads(json.dumps(groups))
    plan = _OnlineRuntime._execution_plan_for(cohort, {"operator_invocation_groups": groups})
    assert plan["operator_invocation_group_count"] == len(groups) > 0
    assert sum(row["rows"] for row in plan["physical_ubatches"]) == 3
    for group, physical, kv_range in zip(groups, plan["physical_ubatches"], plan["kv_ranges"]):
        assert physical["group_id"] == group["group_id"]
        assert physical["rows"] == group["physical_ubatch_rows"]
        assert physical["logit_rows"] == group["logit_token_batch"]
        assert kv_range["materialized_tokens"] == group["kv_materialized_tokens"]


@pytest.mark.parametrize("missing", [
    "group_id", "lanes", "physical_ubatch_rows", "token_batch",
    "logit_token_batch", "kv_read_tokens", "kv_append_tokens",
    "kv_materialized_tokens",
])
def test_execution_plan_rejects_incomplete_old_group_metadata(invocation, missing):
    cohort, groups = invocation
    old = dict(groups[0])
    old.pop(missing)
    with pytest.raises(KeyError, match=missing):
        _OnlineRuntime._execution_plan_for(cohort, {"operator_invocation_groups": (old,)})


@pytest.mark.parametrize("groups", [None, {}, "group", ("group",)])
def test_execution_plan_does_not_hide_invalid_group_shapes(invocation, groups):
    cohort, _ = invocation
    with pytest.raises(ValueError, match="operator_invocation_groups"):
        _OnlineRuntime._execution_plan_for(cohort, {"operator_invocation_groups": groups})


def test_cost_provider_without_group_metadata_remains_supported(invocation):
    cohort, _ = invocation
    plan = _OnlineRuntime._execution_plan_for(cohort, {})
    assert plan["physical_ubatches"] == plan["kv_ranges"] == ()


def test_memory_direction_report_uses_mapping_and_explicit_compact_marker():
    metadata = {"cost_model": {"read_bytes": 48, "write_bytes": 16},
                "modeled_memory_write_bytes": 16, "memory_direction": "read"}
    for payload in (metadata, compact_execution_metadata(metadata)):
        task = SimpleNamespace(metadata=payload)
        assert _memory_direction_rows(task, "hbm.memory", 64) == [
            ("write", 16.0), ("read", 48.0),
        ]
    task = SimpleNamespace(metadata={"cost_model": True, "memory_direction": "read"})
    assert _memory_direction_rows(task, "hbm.memory", 64) == [("read", 64)]
    task = SimpleNamespace(metadata={"cost_model": {"write_bytes": 16}})
    assert _memory_direction_rows(task, "hbm.memory", 64) == [("write", 16.0), ("read", 48.0)]
    task = SimpleNamespace(metadata={"cost_model": SimpleNamespace(write_bytes=16)})
    with pytest.raises(ValueError, match="cost_model must be a mapping"):
        _memory_direction_rows(task, "hbm.memory", 64)


def test_transaction_program_count_has_one_current_property():
    result = TransactionResult("write", Operation.WRITE, 0.0, 10.0, 64, 64,
                               counters={"pages_programmed": 2})
    assert result.pages_programmed == 2
    with pytest.raises(AttributeError):
        result.physical_programs


def memory_tasks():
    return tuple(TaskSpec(name, "request", name, TaskCategory.MEMORY,
        demands=(ResourceDemand("wire", duration, bytes_moved=64),))
        for name, duration in (("a", 5.), ("b", 7.), ("c", 11.)))


def memory_schedule(tasks, lanes=1):
    manifest = RunManifest(schema_version=SIMULATION_SCHEMA_VERSION,
        run_id="memory-inputs", random_seed=0, simulator_version="test",
        model_name="test", hardware_name="test", workload_name="test")
    return ScheduleIR(manifest, tasks, resource_capacities={"wire": lanes})


def observe_event(collector, event):
    return collector.observe(event.task, start_ns=event.start_ns, end_ns=event.end_ns,
        demands=event.demands, resource_predecessors=event.resource_predecessors,
        resource_lanes=event.resource_lanes)


@pytest.mark.parametrize("lanes, expected", [(1, 23.), (2, 16.)])
def test_memory_critical_paths_use_current_kernel_predecessors(lanes, expected):
    tasks = memory_tasks()
    kernel = UnifiedEventKernel.from_closed_graph(tasks, resource_capacities={"wire": lanes})
    collector = _CriticalPathAccumulator()
    while kernel.has_active_tasks:
        event = kernel.step()
        observe_event(collector, event)
    assert ("lane" in event.resource_predecessors["wire"]) == (lanes > 1)
    assert collector.global_categories() == {TaskCategory.MEMORY: expected}
    result = execute_cost_schedule(memory_schedule(tasks, lanes))
    assert result.makespan_ns == expected
    assert result.critical_path_category_ns == collector.global_categories()


@pytest.mark.parametrize("missing", ["end_ns", "resource_id"])
def test_memory_critical_paths_reject_incomplete_predecessors(monkeypatch, missing):
    original_step = UnifiedEventKernel.step

    def old_event(kernel):
        event = original_step(kernel)
        if event is not None and event.resource_predecessors:
            info = dict(event.resource_predecessors["wire"])
            info.pop(missing)
            return replace(event, resource_predecessors={"wire": info})
        return event

    monkeypatch.setattr(UnifiedEventKernel, "step", old_event)
    collector = _CriticalPathAccumulator()
    kernel = UnifiedEventKernel.from_closed_graph(memory_tasks())
    observe_event(collector, kernel.step())
    with pytest.raises(KeyError, match=missing):
        observe_event(collector, kernel.step())
    with pytest.raises(KeyError, match=missing):
        execute_cost_schedule(memory_schedule(memory_tasks()))
