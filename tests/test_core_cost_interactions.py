"""Cost conservation from owner ledgers through resource traces and reports."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.engine import _task_resource_intervals
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config
from heterollm_sim.planner import _summarize_dram_task_traffic, _summarize_nand_task_traffic


def config(medium, **changes):
    if medium == "dram":
        return make_hbm_config(capacity_bytes=1048576, data_lanes=1,
                              metadata={"data_resource_prefix": "shared"}, **changes)
    return make_hbf_config(capacity_bytes=1048576,
                          metadata={"host_resource_id": "shared:0"}, **changes)


def shared_task(media):
    configurations = {owner: config(medium) for owner, medium in zip(("a", "b"), media)}
    accesses = tuple({"physical_owner": owner, "operation": "read", "address": 0,
                      "byte_count": 64 if medium == "dram" else 16384}
                     for owner, medium in zip(("a", "b"), media))
    return TaskSpec("shared", "r", "shared wire", TaskCategory.MEMORY,
        metadata={"physical_memory_config": configurations["a"],
                  "physical_memory_configs": configurations,
                  "memory_accesses": accesses})


@pytest.mark.parametrize("media", [("dram", "dram"), ("nand", "nand"), ("dram", "nand")])
def test_shared_resource_is_counted_once_across_owner_and_medium_reports(media):
    resolved = resolve_physical_task(shared_task(media), PhysicalRuntimeContext(), 0.)
    rows = [_summarize_dram_task_traffic([resolved]), _summarize_nand_task_traffic([resolved])]
    resource_totals = [row.get("resource_totals", {}).get("shared:0", {}) for row in rows]
    demand = next(d for d in resolved.demands if d.resource_id == "shared:0")
    for key, expected in (("bytes_moved", demand.bytes_moved),
                          ("service_ns", demand.service_ns)):
        assert sum(row.get(key, 0) for row in resource_totals) == pytest.approx(expected)
    if media[0] == media[1]:
        row = next(row for row in resource_totals if row)
        assert row["owner"] == "shared:0"


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_detailed_bytes_match_demands_across_owners(medium):
    task = replace(shared_task((medium, medium)), demands=(ResourceDemand("compute", 5.),))
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    intervals = _task_resource_intervals(resolved, resolved.demands, 0.)
    for demand in resolved.demands:
        retained = [item for item in intervals if item.resource_id == demand.resource_id]
        assert sum(item.bytes_moved for item in retained) == demand.bytes_moved


def test_multi_access_observations_and_detail_bounds_describe_the_whole_task():
    c = config("nand", detailed_observability=True, max_expanded_segments=1)
    task = TaskSpec("bounded", "r", "bounded details", TaskCategory.MEMORY,
        metadata={"physical_memory_config": c, "physical_owner": "nand",
                  "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 32768},
                                      {"operation": "read", "address": 0, "byte_count": 16384})})
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    physical = next(iter(resolved.metadata["physical_execution_by_owner"].values()))
    assert len(physical["nand_observations"]["events"]) == 3
    assert physical["details_truncated"]
    assert physical["intervals_truncated"]
    # Later complete request stages remain valid despite an earlier truncation.
    intervals = _task_resource_intervals(resolved, resolved.demands, 0.)
    assert any(item.resource_id.startswith("nand:nand:array") for item in intervals)


def test_alias_of_replaced_physical_preview_does_not_survive_as_second_bill():
    runtime = PhysicalRuntimeContext()
    runtime.timeline.resource_aliases = {"preview-view": "shared:0"}
    c = config("dram")
    task = TaskSpec("alias", "r", "preview replacement", TaskCategory.MEMORY,
        demands=(ResourceDemand("preview-view", 999., bytes_moved=64),
                 ResourceDemand("compute", 5.)),
        metadata={"physical_memory_config": c, "physical_owner": "dram",
                  "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 64},)})
    resolved = resolve_physical_task(task, runtime, 0.)
    assert "preview-view" not in {d.resource_id for d in resolved.demands}
    assert sum(d.bytes_moved for d in resolved.demands) == 64
    assert resolved.metadata["physical_nonmemory_demand_ids"] == ("compute",)


def test_complete_stages_keep_owner_bytes_after_payload_recorder_is_full():
    task = shared_task(("dram", "dram"))
    configs = {owner: replace(c, max_expanded_segments=2)
               for owner, c in task.metadata["physical_memory_configs"].items()}
    task = replace(task, metadata={**task.metadata, "physical_memory_configs": configs,
                                  "physical_memory_config": configs["a"]})
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    physical = tuple(resolved.metadata["physical_execution_by_owner"].values())
    assert any(row["intervals_truncated"] for row in physical)
    assert not any(row["details_truncated"] for row in physical)
    intervals = _task_resource_intervals(resolved, resolved.demands, 0.)
    shared = [item for item in intervals if item.resource_id == "shared:0"]
    assert [item.bytes_moved for item in shared] == [64, 64]


def test_parallel_owners_with_different_payload_bounds_keep_their_own_bytes():
    task = shared_task(("dram", "dram"))
    configs = {owner: replace(c, max_expanded_segments=limit)
               for (owner, c), limit in zip(task.metadata["physical_memory_configs"].items(), (2, 3))}
    task = replace(task, metadata={**task.metadata, "physical_memory_configs": configs,
                                  "physical_memory_config": configs["a"]})
    runtime = PhysicalRuntimeContext()
    runtime.timeline.lane_available["shared:0"] = [0., 0.]
    resolved = resolve_physical_task(task, runtime, 0.)
    payloads = [item for row in resolved.metadata["physical_execution_by_owner"].values()
                for item in row["resource_interval_payloads"].get("shared:0", ())]
    assert len(payloads) == 1 and payloads[0][2] == 64
    intervals = _task_resource_intervals(resolved, resolved.demands, 0.)
    shared = [item for item in intervals if item.resource_id == "shared:0"]
    assert len(shared) == 2
    assert shared[0].start_ns == shared[1].start_ns
    assert sum(item.bytes_moved for item in shared) == 128


def test_shared_owner_bill_requires_attributable_byte_counts():
    resolved = resolve_physical_task(shared_task(("dram", "dram")), PhysicalRuntimeContext(), 0.)
    owners = {owner: {key: value for key, value in row.items()
                      if key != "resource_bytes"}
              for owner, row in resolved.metadata["physical_execution_by_owner"].items()}
    resolved = replace(resolved, metadata={**resolved.metadata, "physical_execution_by_owner": owners})
    with pytest.raises(ValueError, match="require resource_bytes"):
        _summarize_dram_task_traffic([resolved])


@pytest.mark.parametrize("count", [1, 2])
def test_owner_payload_intervals_keep_exact_byte_counts(count):
    demand = ResourceDemand("wire", float(count), bytes_moved=128)
    task = TaskSpec("payloads", "r", "owner payload intervals", TaskCategory.MEMORY,
        demands=(demand,), metadata={
            "physical_execution_by_owner": {"memory": {
                "resource_interval_payloads": {"wire": [(2.*i, 2.*i+1, 128 // count) for i in range(count)]},
                "resource_bytes": {"wire": 128}}},
            "physical_demands_resource_ids": ("wire",),
            "physical_nonmemory_demand_ids": ()})
    intervals = _task_resource_intervals(task, task.demands, 0.)
    assert len(intervals) == count
    assert sum(item.bytes_moved for item in intervals) == 128


@pytest.mark.parametrize("detail_limit", [2, 100])
def test_simultaneous_reservations_retain_multiplicity_with_partial_or_full_payloads(detail_limit):
    c = make_hbm_config(capacity_bytes=1048576, channels=2,
                        pseudo_channels_per_channel=1, max_expanded_segments=detail_limit)
    kernel = UnifiedEventKernel(
        resource_owners={"memory:dram:command:0": "ca", "memory:dram:command:1": "ca",
                         "memory:dram:data:0": "wire", "memory:dram:data:1": "wire"},
        resource_capacities={"ca": 2, "wire": 2}, capture_physical_details=True)
    kernel.submit([TaskSpec("parallel", "r", "parallel physical", TaskCategory.MEMORY,
        metadata={"physical_memory_config": c, "physical_owner": "memory",
                  "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 128},)})])
    event = kernel.step()
    intervals = _task_resource_intervals(event.task, event.demands, event.start_ns)
    commands = [item for item in intervals if item.resource_id == "ca"]
    assert len(commands) == 2
    assert commands[0].start_ns == commands[1].start_ns
    assert commands[0].end_ns == commands[1].end_ns
    for demand in event.demands:
        matching = [item for item in intervals if item.resource_id == demand.resource_id]
        assert sum(item.end_ns - item.start_ns for item in matching) == pytest.approx(demand.service_ns)
        assert sum(item.bytes_moved for item in matching) == demand.bytes_moved
