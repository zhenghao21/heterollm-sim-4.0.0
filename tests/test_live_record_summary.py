"""Live summaries consume immutable records with identical ordered reductions."""
from collections import UserDict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import KernelEvent
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.scalable_serving import TaskExecutionRecord
from heterollm_sim import planner


@pytest.fixture(params=("python", "compiled"))
def ledger(request):
    if request.param == "compiled":
        from heterollm_sim import _cost_ledger
        return _cost_ledger
    source = Path(__file__).resolve().parents[1] / "src" / "heterollm_sim" / "_cost_ledger.py"
    spec = importlib.util.spec_from_file_location("python_live_record_cost_ledger", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(value):
    if type(value) is dict:
        return (dict, tuple((key, exact(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(exact(item) for item in value))
    if type(value) is float:
        return (float, value.hex())
    return (type(value), value)


def records():
    rows, names = [], {}
    for index, duration in enumerate((1e16, 1., 1.)):
        task_id = "task-" + str(index)
        names[task_id] = "hidden.output" if index == 1 else "weights"
        raw = {"logical_bytes": 2**55 + index, "logical_read_bytes": 4,
               "logical_write_bytes": 2, "service_ns": duration, "queue_wait_ns": -0.,
               "host_admission_depth_scope": "host channel", "pages_read": 3,
               "resource_busy_ns": {"wire": duration}, "resource_bytes": {"wire": 2**55 + index},
               "resource_owners": {"wire": "first" if index == 0 else "second"}}
        metadata = {"physical_execution_by_owner": {"memory": {"kind": "HBF", **raw},
                    "host": {"kind": "DDR", **raw}},
                    "modeled_memory_write_bytes": 1 if index == 2 else 0}
        if index == 2:
            metadata["resource_directions"] = {"wire": "WrItE"}
        rows.append(TaskExecutionRecord(task_id, float(index), float(index + 1), TaskCategory.MEMORY,
            (), metadata, (ResourceDemand("wire", duration, 4), ResourceDemand("unused", -0.))))
    return tuple(rows), names


def test_existing_direction_and_physical_summaries_accept_records_without_proxies(ledger):
    rows, names = records()
    tasks = tuple(SimpleNamespace(task_id=row.task_id, name=names[row.task_id],
                                  metadata=row.metadata, demands=row.demands) for row in rows)
    expected = {"direction": ledger._resource_busy_by_direction(tasks),
                "dram": planner._summarize_dram_task_traffic(tasks),
                "nand": planner._summarize_nand_task_traffic(tasks)}
    actual = {"direction": ledger._resource_busy_by_direction(rows, task_names=names),
              "dram": planner._summarize_dram_task_traffic(rows),
              "nand": planner._summarize_nand_task_traffic(rows)}
    assert exact(actual) == exact(expected)
    assert actual["nand"]["service_ns"] == 1e16
    assert actual["dram"]["resource_totals"]["wire"]["owner"] == "wire"


def test_direction_name_mapping_is_lazy_and_retains_original_metadata_protocol(ledger):
    calls = []
    class Names(UserDict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)
    rows = (
        SimpleNamespace(task_id="explicit", metadata=UserDict({"memory_direction": "write"}),
                        demands=(ResourceDemand("first", 1.),)),
        SimpleNamespace(task_id="infer", metadata=UserDict({"event_kind": "ordinary"}),
                        demands=(ResourceDemand("second", 1.),)),
    )
    assert ledger._resource_busy_by_direction(rows, task_names=Names({"infer": "append.result"})) == {
        "write": {"first": 1., "second": 1.}}
    assert calls == ["infer"]


def test_live_metadata_keeps_every_field_order_type_and_original_summary_records(ledger, monkeypatch):
    monkeypatch.setattr(planner, "_resource_busy_by_direction", ledger._resource_busy_by_direction)
    live = LiveCohortMetrics()
    for index, name in enumerate(("weights", "output.store", "host")):
        metadata = {"orchestration_stage": "host_prefix", "submission_count": "2"} if index == 2 else {}
        demand = ResourceDemand("wire-" + str(index), float(index + 1), bytes_moved=index + 1)
        task = TaskSpec(str(index), "request", name, TaskCategory.COMPUTE,
                        demands=(demand,), metadata=metadata)
        live.observe(KernelEvent(task, float(index), float(index + index + 1), float(index),
                                 float(index), (demand,), {}, {}))
    summary = live.summary()
    tasks = tuple(SimpleNamespace(task_id=row.task_id, name=live._names[row.task_id],
                                  metadata=row.metadata, demands=row.demands)
                  for row in summary.execution_records)
    expected = {"resource_busy_by_direction_ns": ledger._resource_busy_by_direction(tasks),
                "dram_traffic": planner._summarize_dram_task_traffic(tasks, resource_owners=live.resource_owners),
                "storage_traffic": planner._summarize_nand_task_traffic(tasks, resource_owners=live.resource_owners)}
    # Observe that all three reducers receive the exact record tuple, not a
    # new wrapper collection, without changing their reduction code.
    observed = []
    for name in ("_resource_busy_by_direction", "_summarize_dram_task_traffic", "_summarize_nand_task_traffic"):
        original = getattr(planner, name)
        def wrapper(rows, _original=original, **kwargs):
            observed.append(rows)
            return _original(rows, **kwargs)
        monkeypatch.setattr(planner, name, wrapper)
    report = live.metadata()
    assert observed == [summary.execution_records] * 3
    assert all(rows is summary.execution_records for rows in observed)
    assert exact({key: report[key] for key in expected}) == exact(expected)
    assert report["host_orchestration_task_count"] == 1
    assert report["host_submission_count"] == 2
    assert list(report) == ["task_count", "resource_accounted_bytes", "resource_busy_ns",
        "resource_busy_by_direction_ns", "category_time_ns", "critical_path_category_ns", "coverage",
        "linear_state_bytes", "dram_traffic", "storage_traffic", "execution_stage_makespan_ns",
        "host_orchestration_ns", "device_execution_ns", "host_task_time_ns", "host_device_overlap_ns",
        "execution_idle_ns", "execution_time_semantics", "host_orchestration_task_count",
        "host_submission_count", "physical_execution_scope", "cost_duration_semantics"]
