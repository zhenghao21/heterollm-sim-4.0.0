from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

from heterollm_sim.contracts import ResourceDemand, RunManifest, TaskCategory, TaskSpec, RetentionPolicy
from heterollm_sim.engine import ScheduleIR, _task_resource_intervals, simulate_schedule
from heterollm_sim.memory_types import DramConfig
from heterollm_sim.planner import RequestTaskChunk, StreamingScheduleIR
from heterollm_sim.schema_v4 import SIMULATION_SCHEMA_VERSION
from heterollm_sim.streaming_des import execute_incremental_schedule


def _dram(**kwargs):
    return DramConfig(
        banks_per_group=1,
        rows_per_bank=4,
        row_bytes=128,
        burst_bytes=64,
        lane_bandwidth_gb_s=64,
        open_ns=10,
        read_latency_ns=5,
        write_latency_ns=5,
        burst_interval_ns=1,
        write_recovery_ns=0,
        **kwargs,
    )


def _task(*, detail=True):
    config = _dram(max_expanded_segments=100 if detail else 1)
    access = {"operation": "read", "address": 0, "byte_count": 64,
              "physical_owner": "A", "resource_id": "A"}
    return TaskSpec(
        "t", "r", "mixed", TaskCategory.MEMORY,
        demands=(
            ResourceDemand("A", 16, bytes_moved=64),
            ResourceDemand("gpu.compute", 1000, work_units=1234),
        ),
        metadata={"physical_memory_config": asdict(config), "memory_access": access},
    )


def _manifest():
    return RunManifest(
        schema_version=SIMULATION_SCHEMA_VERSION, run_id="r", random_seed=0,
        simulator_version="test", model_name="m", hardware_name="h", workload_name="w",
    )


def test_simulate_schedule_keeps_compute_and_physical_payload():
    result = simulate_schedule(ScheduleIR(_manifest(), (_task(),)))
    intervals = result.tasks[0].resource_intervals
    compute = next(item for item in intervals if item.resource_id == "gpu.compute")
    data = next(item for item in intervals if item.resource_id == "A:dram:data:0")
    assert compute.end_ns - compute.start_ns == 1000
    assert compute.bytes_moved == 0
    assert compute.energy_pj == 0
    assert (data.start_ns, data.end_ns, data.bytes_moved, data.energy_pj) == (15, 16, 64, 0)
    bank = next(item for item in intervals if item.resource_id == "A:dram:bank:0:0:0:0:0:0")
    command = next(item for item in intervals if item.resource_id == "A:dram:command:0")
    assert (bank.start_ns, bank.end_ns) == (0, 10)
    assert (command.start_ns, command.end_ns) == (10, 11)
    assert len(intervals) == 4


def test_streaming_full_history_uses_same_projection(monkeypatch):
    task = _task()
    schedule = StreamingScheduleIR(_manifest(), object(), ())
    monkeypatch.setattr("heterollm_sim.streaming_des._compilation_scope", lambda _: nullcontext())
    monkeypatch.setattr(
        "heterollm_sim.streaming_des._request_iterator",
        lambda _: iter((SimpleNamespace(request_id="r"),)),
    )
    monkeypatch.setattr(
        "heterollm_sim.streaming_des._request_task_chunks",
        lambda _schedule, _request: iter((RequestTaskChunk((task,), "t", True),)),
    )
    result = execute_incremental_schedule(schedule, retention_policy=RetentionPolicy.EXACT)
    intervals = result.trace.tasks[0].resource_intervals
    assert any(item.resource_id == "gpu.compute" for item in intervals)
    assert any(item.resource_id == "A:dram:data:0" and item.bytes_moved == 64 for item in intervals)
    assert any(item.resource_id == "A:dram:bank:0:0:0:0:0:0" and (item.start_ns, item.end_ns) == (0, 10)
               for item in intervals)
    assert any(item.resource_id == "A:dram:command:0" and (item.start_ns, item.end_ns) == (10, 11)
               for item in intervals)
    assert len(intervals) == 4


def test_multi_burst_report_merges_reservations_without_duplicate_bytes():
    task = _task()
    task = replace(task, metadata={
        **task.metadata,
        "memory_access": {**task.metadata["memory_access"], "byte_count": 128},
    })
    result = simulate_schedule(ScheduleIR(_manifest(), (task,)))
    intervals = result.tasks[0].resource_intervals
    data = [item for item in intervals if item.resource_id == "A:dram:data:0"]
    commands = [item for item in intervals if item.resource_id == "A:dram:command:0"]
    assert len(data) == len(commands) == 2
    assert sum(item.bytes_moved for item in data) == 128
    assert len({(item.resource_id, item.start_ns, item.end_ns) for item in intervals}) == len(intervals)


def test_low_detail_does_not_fabricate_physical_intervals_but_keeps_compute():
    config = _dram(max_expanded_segments=1)
    access = {"operation": "read", "address": 0, "byte_count": 256,
              "physical_owner": "A", "resource_id": "A"}
    task = TaskSpec(
        "t", "r", "low", TaskCategory.MEMORY,
        demands=(ResourceDemand("A", 16, bytes_moved=256), ResourceDemand("gpu.compute", 1000, work_units=1234)),
        metadata={"physical_memory_config": asdict(config), "memory_access": access},
    )
    result = simulate_schedule(ScheduleIR(_manifest(), (task,)))
    intervals = result.tasks[0].resource_intervals
    assert any(item.resource_id == "gpu.compute" for item in intervals)
    assert not any(item.resource_id.startswith("A:dram:") for item in intervals)


def test_payload_projection_keeps_bytes_and_energy():
    task = TaskSpec(
        "p", "r", "payload", TaskCategory.MEMORY,
        demands=(ResourceDemand("gpu.compute", 1000, work_units=1234),),
        metadata={
            "physical_execution": {
                "resource_interval_payloads": {"A:dram:data:0": ((15, 16, 64, 2.5),)},
            },
            "physical_nonmemory_demand_ids": ("gpu.compute",),
        },
    )
    intervals = _task_resource_intervals(task, task.demands, 0)
    physical = next(item for item in intervals if item.resource_id == "A:dram:data:0")
    assert (physical.start_ns, physical.end_ns, physical.bytes_moved, physical.energy_pj) == (15, 16, 64, 2.5)
