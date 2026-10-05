import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_types import AccessRequest, DramConfig, MemoryKind, Operation, parse_physical_memory_config
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel


def _config():
    return DramConfig(
        kind=MemoryKind.GDDR,
        generation="GDDR7",
        channels=1,
        data_lanes=2,
        data_width_bits=32,
        data_rate_mt_s=10000.0,
        stacks=1,
        dies_per_stack=1,
        ranks_per_channel=1,
        bank_groups_per_rank=2,
        banks_per_group=2,
        rows_per_bank=64,
        row_bytes=256,
        burst_bytes=64,
        interleave_bytes=64,
        interface_bandwidth_gb_s=80.0,
        capacity_bytes=131072,
    )


def test_gddr_alias_parsing_and_shared_dram_core():
    config = parse_physical_memory_config({**_config().__dict__, "kind": "GDDR7"})
    assert config.kind is MemoryKind.GDDR
    assert config.generation == "GDDR7"
    core = DramCore(config)
    result = core.submit(AccessRequest("read", Operation.READ, 0, 1))
    assert result.transfer_bytes == 64
    assert result.counters["burst_count"] == 1
    with pytest.raises(ValueError, match="ERASE"):
        core.submit(AccessRequest("erase", Operation.ERASE, 0, 64))


def test_gddr_physical_task_uses_owner_and_keeps_nonmemory_demand():
    raw = _config().__dict__
    task = TaskSpec(
        task_id="gemm.gddr",
        request_id="request-0",
        name="GEMM local GDDR read",
        category=TaskCategory.MEMORY,
        demands=(
            ResourceDemand("gpu0.compute", 3.0),
            ResourceDemand("gddr0.gddr_fabric", 1.0, bytes_moved=64),
        ),
        metadata={
            "physical_memory_config": raw,
            "memory_access": {
                "operation": "read",
                "address": 0,
                "byte_count": 1,
                "physical_owner": "gddr0.gddr_fabric",
                "resource_id": "gddr0.gddr_fabric",
            },
        },
    )
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.0)
    assert "gpu0.compute" in {d.resource_id for d in resolved.demands}
    assert "gddr0.gddr_fabric" not in {d.resource_id for d in resolved.demands}
    assert resolved.metadata["physical_execution"]["physical_bytes"] == 64


def test_unified_event_kernel_commits_gddr_at_dispatch():
    raw = _config().__dict__
    task = TaskSpec(
        task_id="event.gddr",
        request_id="request-1",
        name="event GDDR read",
        category=TaskCategory.MEMORY,
        demands=(ResourceDemand("gddr0.gddr_fabric", 1.0, bytes_moved=64),),
        metadata={
            "physical_memory_config": raw,
            "memory_access": {
                "operation": "read",
                "address": 0,
                "byte_count": 1,
                "physical_owner": "gddr0.gddr_fabric",
                "resource_id": "gddr0.gddr_fabric",
            },
        },
    )
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    event = kernel.step()
    assert event is not None
    assert event.task.metadata["physical_execution"]["physical_bytes"] == 64
