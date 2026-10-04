from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import DramConfig


def _config(channels=1):
    return DramConfig(
        channels=channels,
        banks_per_group=1,
        rows_per_bank=4,
        row_bytes=128,
        burst_bytes=64,
        open_ns=10,
        read_latency_ns=5,
        burst_interval_ns=1,
        lane_bandwidth_gb_s=64,
    )


def _task(config, task_id, address, dependencies=(), owner="dram0"):
    return TaskSpec(
        task_id=task_id,
        request_id=task_id,
        name=task_id,
        category=TaskCategory.COMMUNICATION,
        dependencies=tuple(dependencies),
        # The planner preview demand is deliberately a placeholder.  The
        # kernel replaces it with core-owned resource reservations at step().
        demands=(ResourceDemand(owner, 1.0),),
        metadata={
            "physical_memory_config": config,
            "physical_owner": owner,
            "memory_access": {
                "operation": "read",
                "address": address,
                "byte_count": 64,
                "physical_owner": owner,
                "resource_id": owner,
            },
            "physical_execution": {"service_ns": 0.0},
        },
    )


def test_event_kernel_commits_row_state_at_actual_task_start():
    config = _config()
    kernel = UnifiedEventKernel()
    kernel.add_tasks([_task(config, "first", 0), _task(config, "second", 0)])
    first = kernel.step()
    second = kernel.step()
    assert first is not None and second is not None
    assert first.task.metadata["physical_execution"]["row_misses"] == 1
    assert second.task.metadata["physical_execution"]["row_hits"] == 1
    assert second.task.metadata["physical_execution"]["row_misses"] == 0
    assert second.end_ns > first.end_ns


def test_event_kernel_keeps_independent_channels_parallel():
    config = _config(channels=2)
    kernel = UnifiedEventKernel()
    kernel.add_tasks([_task(config, "left", 0), _task(config, "right", 64)])
    first = kernel.step()
    second = kernel.step()
    assert first is not None and second is not None
    assert first.start_ns == second.start_ns == 0.0
    assert first.end_ns == second.end_ns


def test_event_kernel_propagates_physical_completion_to_dependencies():
    config = _config()
    kernel = UnifiedEventKernel()
    kernel.add_tasks([_task(config, "first", 0), _task(config, "dependent", 0, ("first",))])
    first = kernel.step()
    dependent = kernel.step()
    assert first is not None and dependent is not None
    assert dependent.start_ns == first.end_ns
    assert dependent.task.metadata["physical_arrival_ns"] == first.end_ns
