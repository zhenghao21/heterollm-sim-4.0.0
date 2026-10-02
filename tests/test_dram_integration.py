from dataclasses import asdict

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import endpoint_service
from heterollm_sim.dram import DramProfile
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec, PortSpec


def profile(**overrides):
    values = dict(
        channels=2,
        banks_per_channel=2,
        burst_bytes=16,
        row_bytes=64,
        t_rcd_ns=2.0,
        t_rp_ns=3.0,
        t_ras_ns=4.0,
        read_latency_ns=1.0,
        write_latency_ns=2.0,
        read_to_write_ns=5.0,
        write_to_read_ns=6.0,
        read_recovery_ns=1.0,
        write_recovery_ns=2.0,
        refresh_interval_ns=0.0,
        refresh_duration_ns=0.0,
        evidence="integration analytical contract",
        aggregate_policy="cold_contiguous",
    )
    values.update(overrides)
    return DramProfile(**values)


def component(address):
    return ComponentSpec(
        "dram0",
        "hbm",
        ports=(PortSpec("host", "HBM", "device", bandwidth_gbps=800.0),),
        read_bandwidth_gbps=800.0,
        write_bandwidth_gbps=800.0,
        metadata={
            "memory_service_owner": "dram0.controller",
            "dram_profile": asdict(profile()),
            "dram_address_bytes": address,
        },
    )


def dram_task(task_id, offset, operation="read"):
    direction = "read" if operation == "read" else "write"
    service = endpoint_service(
        component(offset), 16, read=direction == "read", name=task_id
    )
    contract = service.metadata["dram_access"]
    return TaskSpec(
        task_id,
        "request",
        task_id,
        TaskCategory.MEMORY,
        demands=service.demands,
        metadata={"dram_access": contract},
    )


def test_endpoint_contract_reaches_kernel_and_reports_resolved_dram_metrics():
    task = dram_task("read0", 0)
    kernel = UnifiedEventKernel(resource_capacities={"hbm.channel": 1})
    kernel.submit((task,))
    event = kernel.step()
    assert event is not None
    report = event.task.metadata["dram_execution"]
    assert report["model"] == "dram_addressed_burst_v1"
    assert report["logical_read_bytes"] == 16
    assert report["physical_bytes"] == 16
    assert report["row_misses"] == 1
    assert event.demands[0].bytes_moved == 16
    assert event.demands[0].service_ns == pytest.approx(report["service_ns"])


def test_kernel_persists_rows_between_endpoint_tasks_without_planner_mutation():
    first = dram_task("read0", 0)
    hit = dram_task("read1", 0)
    conflict = dram_task("read2", 256)
    kernel = UnifiedEventKernel(resource_capacities={"hbm.channel": 1})
    kernel.submit((first, hit, conflict))
    events = tuple(kernel.step() for _ in range(3))
    assert events[1].task.metadata["dram_execution"]["row_hits"] == 1
    assert events[2].task.metadata["dram_execution"]["row_conflicts"] == 1
    assert events[0].task.metadata["dram_access"] == first.metadata["dram_access"]


def test_failed_dram_preview_does_not_commit_kernel_state():
    task = dram_task("read0", 0)
    task = TaskSpec(
        task.task_id, task.request_id, task.name, task.category,
        demands=task.demands,
        metadata={
            "dram_access": {
                **task.metadata["dram_access"],
                "profile": {**task.metadata["dram_access"]["profile"], "max_bursts_per_access": 0},
            }
        },
    )
    kernel = UnifiedEventKernel()
    kernel.submit((task,))
    with pytest.raises(ValueError):
        kernel.step()
    assert kernel._dram_states == {}
    assert kernel.peek_ready_key() is not None
