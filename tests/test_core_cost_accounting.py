"""Physical task admission, rollback and endpoint byte accounting."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_types import AccessRequest, make_hbm_config, make_hbf_config
from heterollm_sim.nand_core import NandCore


def task(config, accesses, **metadata):
    return TaskSpec("audit", "request", "physical audit", TaskCategory.MEMORY,
                    metadata={"physical_memory_config": config,
                              "physical_owner": "memory",
                              "memory_accesses": accesses, **metadata})


@pytest.mark.parametrize("existing_owner", [False, True])
def test_rejected_task_does_not_leave_allocations(existing_owner):
    config = make_hbm_config(capacity_bytes=1024 * 1024)
    runtime = PhysicalRuntimeContext()
    if existing_owner:
        runtime.runtime(config, "memory")
        runtime.allocators["memory"].allocate("keep", 64)
    rejected = task(config, [{"operation": "read", "address": 0,
                             "byte_count": config.capacity_bytes + 1}],
                    physical_allocations=[{"buffer_id": "leaked", "size_bytes": 1024}])
    with pytest.raises(ValueError, match="exceeds owner"):
        resolve_physical_task(rejected, runtime, 0.0)
    assert not runtime.timeline.ready_ns
    if existing_owner:
        assert [a.buffer_id for a in runtime.allocators["memory"].allocations()] == ["keep"]
    else:
        assert not runtime.runtimes
        assert not runtime.allocators


def test_late_rejected_access_rolls_back_executed_core_and_allocation():
    config = make_hbf_config(capacity_bytes=4 * 16384, partial_page_policy="reject")
    runtime = PhysicalRuntimeContext()
    core = runtime.runtime(config, "memory").core
    rejected = task(config, [
        {"operation": "read", "address": 0, "byte_count": 16384},
        {"operation": "write", "address": 16384, "byte_count": 1},
    ], physical_allocations=[{"buffer_id": "leaked", "size_bytes": 16384}])
    with pytest.raises(ValueError, match="partial NAND page write"):
        resolve_physical_task(rejected, runtime, 0.0)
    assert not core._array_ready
    assert not core._inflight
    assert not runtime.timeline.ready_ns
    assert not runtime.timeline.bytes_moved
    assert not runtime.allocators["memory"].allocations()


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_restore_point_remains_valid_after_resumed_execution(medium):
    config = (make_hbm_config(capacity_bytes=1024 * 1024) if medium == "dram"
              else make_hbf_config(capacity_bytes=4 * 16384, write_buffer_bytes=16384))
    runtime = PhysicalRuntimeContext(config=config, physical_owner="memory")
    core = runtime.runtimes["memory"].core
    saved = runtime.snapshot()
    for _ in range(2):
        core.submit(AccessRequest("write", "write", 0, 64))
        runtime.restore(saved)
        assert not core._inflight
        if medium == "dram":
            assert not core._banks
        else:
            assert not core._staged_pages
            assert core.pending_write_buffer_bytes == 0
        assert not runtime.timeline.ready_ns
        assert not runtime.timeline.bytes_moved


def test_task_aggregates_buffer_waits_and_merges_instead_of_last_request():
    config = make_hbf_config(capacity_bytes=65536, page_bytes=4096,
                             host_granularity_bytes=1, write_buffer_bytes=2048)
    addresses = (0, 1024, 8192, 9216)
    accesses = [{"operation": "write", "address": address, "byte_count": 1024}
                for address in addresses]
    core = NandCore(config)
    expected = [core.submit(AccessRequest(str(i), "write", address, 1024))
                for i, address in enumerate(addresses)]
    resolved = resolve_physical_task(task(config, accesses), PhysicalRuntimeContext(), 0.0)
    expected_wait = sum(r.write_buffer_wait_ns for r in expected)
    assert expected_wait > 0
    for row in (next(iter(resolved.metadata["physical_execution_by_owner"].values())),
                resolved.metadata["physical_execution_by_owner"]["memory"]):
        assert row["write_buffer_wait_ns"] == pytest.approx(expected_wait)
        assert row["write_buffer_bytes"] == 4096
        assert row["write_buffer_peak_bytes"] == 2048
        assert row["merge_bytes"] == 2048
        assert row["staging_write_pages"] == 4
    assert resolved.metadata["physical_completion_ns"] == expected[-1].completion_ns


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_memory_bytes_are_charged_once_and_compute_demand_preserved(medium):
    if medium == "dram":
        config = make_hbm_config(capacity_bytes=1024 * 1024, burst_bytes=64)
        expected_media_bytes = 64
    else:
        config = make_hbf_config(capacity_bytes=4 * 16384, host_granularity_bytes=1)
        # A one-byte partial page write reads and programs one whole page.
        expected_media_bytes = 2 * 16384
    original = task(config, [{"operation": "write", "address": 0, "byte_count": 1,
                              "resource_id": "memory"}])
    original = replace(original, demands=(ResourceDemand("gpu:compute", 200),
                                          ResourceDemand("memory", 999999)))
    resolved = resolve_physical_task(original, PhysicalRuntimeContext(), 0.0)
    execution = next(iter(resolved.metadata["physical_execution_by_owner"].values()))
    assert execution["physical_bytes"] == expected_media_bytes
    assert resolved.demands[0] == ResourceDemand("gpu:compute", 200)
    assert all(d.resource_id != "memory" for d in resolved.demands)
