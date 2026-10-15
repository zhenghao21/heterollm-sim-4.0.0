"""Shared Core calendars and event failure transactions."""
import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config


def physical_task(config, *, task_id="physical", owner="memory", **kwargs):
    return TaskSpec(task_id, "request", task_id, TaskCategory.MEMORY,
                    metadata={"physical_memory_config": config,
                              "physical_owner": owner,
                              "memory_accesses": [{"operation": "read", "address": 0,
                                                   "byte_count": 4096}]}, **kwargs)


@pytest.mark.parametrize("physical", [False, True])
def test_timing_rejection_keeps_task_and_reverts_core(physical):
    demands = (ResourceDemand("compute", 1e308),)
    task = (physical_task(make_hbm_config(capacity_bytes=1048576),
                          demands=demands, earliest_start_ns=1e308) if physical else
            TaskSpec("plain", "request", "plain", TaskCategory.COMPUTE,
                     demands=demands, earliest_start_ns=1e308))
    kernel = UnifiedEventKernel()
    kernel.submit([task])
    for _ in range(2):
        with pytest.raises(ValueError, match="timing exceeds finite"):
            kernel.step()
        assert task.task_id in kernel._tasks
        assert len(kernel._ready_heap) == 1
        assert kernel.completed_count == 0
        assert not kernel.physical_runtime.runtimes
        assert not kernel.physical_runtime.timeline.ready_ns
        assert not kernel.physical_runtime.timeline.bytes_moved
        assert not kernel._resource_lane_available
        assert not kernel.resource_available


@pytest.mark.parametrize("capture_details", [False, True])
@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_normal_and_core_demands_share_alias_calendar(medium, capture_details):
    if medium == "dram":
        config = make_hbm_config(capacity_bytes=1048576)
        physical_resource = "memory:dram:data:0"
    else:
        config = make_hbf_config(capacity_bytes=1048576)
        physical_resource = "memory:nand:host"
    kernel = UnifiedEventKernel(
        resource_owners={physical_resource: "wire"},
        capture_physical_details=capture_details)
    kernel.submit([TaskSpec("before", "request", "before", TaskCategory.MEMORY,
                            demands=(ResourceDemand("wire", 100000),))])
    assert kernel.step().end_ns == 100000
    kernel.submit([physical_task(config)])
    event = kernel.step()
    execution = next(iter(event.task.metadata["physical_execution_by_owner"].values()))
    assert "wire" in execution["resource_busy_ns"]
    assert physical_resource not in kernel._resource_lane_available
    wire_end = kernel._resource_lane_available["wire"][0]
    assert wire_end > 100000
    if capture_details:
        assert all(start >= 100000 for start, _end
                   in execution["resource_intervals"]["wire"])
    kernel.submit([TaskSpec("after", "request", "after", TaskCategory.MEMORY,
                            demands=(ResourceDemand("wire", 10),))])
    after = kernel.step()
    assert after.start_ns == pytest.approx(wire_end)
    assert after.end_ns == pytest.approx(wire_end + 10)


def test_shared_wire_collects_each_device_admission_wait_once():
    config = make_hbm_config(capacity_bytes=1048576,
        channels=1, pseudo_channels_per_channel=1, interface_bandwidth_gb_s=64,
        open_ns=0, close_ns=0, read_latency_ns=0, write_latency_ns=0,
        burst_interval_ns=0, read_recovery_ns=0, write_recovery_ns=0,
        read_to_write_ns=0, write_to_read_ns=0, max_outstanding_requests=1,
        metadata={"data_resource_prefix": "shared-data"})
    task = TaskSpec("shared", "request", "shared", TaskCategory.MEMORY,
        metadata={"physical_memory_config": config,
                  "physical_memory_configs": {"a": config, "b": config},
                  "memory_accesses": [
                      {"physical_owner": owner, "operation": "read",
                       "address": 64 * index, "byte_count": 64}
                      for owner in ("a", "b") for index in range(2)]})
    kernel = UnifiedEventKernel()
    kernel.submit([task])
    event = kernel.step()
    by_owner = event.task.metadata["physical_execution_by_owner"]
    assert all(row["queue_wait_ns"] > 0 for row in by_owner.values())
    expected = sum(row["queue_wait_ns"] for row in by_owner.values())
    assert sum(kernel.resource_queue_wait_ns.values()) == pytest.approx(expected)
    assert kernel.resource_queue_wait_ns["shared-data:0"] == pytest.approx(expected)
