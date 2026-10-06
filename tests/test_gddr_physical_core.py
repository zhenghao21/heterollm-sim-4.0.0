import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    PhysicalRuntimeContext,
    register_physical_allocations,
    resolve_physical_task,
)
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("offset_bytes", 1.5, "offset_bytes"),
        ("generation", 2.5, "generation"),
        ("alias_offset_bytes", 3.5, "alias_offset_bytes"),
    ),
)
def test_physical_descriptor_rejects_fractional_integer_fields(
    field, value, message
):
    """Malformed descriptor integers must not be silently truncated."""

    raw = _config().__dict__
    access = {
        "operation": "read",
        "buffer_id": "activation",
        "byte_count": 1,
        "physical_owner": "gddr0",
        "resource_id": "gddr0",
        "offset_bytes": 0,
    }
    if field == "alias_offset_bytes":
        declarations = (
            {
                "buffer_id": "base",
                "size_bytes": 64,
                "physical_owner": "gddr0",
            },
            {
                "buffer_id": "activation",
                "size_bytes": 1,
                "physical_owner": "gddr0",
                "alias_of": "base",
                "alias_offset_bytes": value,
            },
        )
    else:
        declarations = ()
        access[field] = value
    task = TaskSpec(
        task_id="malformed.descriptor",
        request_id="request-0",
        name="malformed physical descriptor",
        category=TaskCategory.MEMORY,
        metadata={
            "physical_memory_config": raw,
            "physical_owner": "gddr0",
            "physical_allocations": declarations,
            "memory_access": access,
        },
    )
    with pytest.raises(ValueError, match=message):
        register_physical_allocations(task, PhysicalRuntimeContext())


def test_physical_allocation_accepts_allocation_generation_alias():
    runtime = PhysicalRuntimeContext()
    task = TaskSpec(
        task_id="generation.alias",
        request_id="request-0",
        name="allocation generation alias",
        category=TaskCategory.MEMORY,
        metadata={
            "physical_memory_config": _config().__dict__,
            "physical_allocations": ({
                "buffer_id": "activation",
                "size_bytes": 64,
                "allocation_generation": 7,
                "physical_owner": "gddr0",
            },),
        },
    )
    register_physical_allocations(task, runtime)
    allocation = runtime.allocators["gddr0"].lookup("activation", 7)
    assert allocation.generation == 7


@pytest.mark.parametrize("offset", (0, 64, 128))
def test_runtime_preserves_buffer_offsets(offset):
    task = TaskSpec(
        task_id="offset.gddr", request_id="offset-0", name="offset GDDR read",
        category=TaskCategory.MEMORY,
        metadata={
            "physical_memory_config": _config().__dict__,
            "physical_owner": "gddr0",
            "memory_accesses": ({
                "operation": "read", "address": offset, "byte_count": 64,
                "buffer_id": "activation", "offset_bytes": offset,
                "allocation_size_bytes": 256,
                "address_source": "stable_buffer_tensor_offset",
                "physical_owner": "gddr0", "resource_id": "gddr0",
            },),
        },
        demands=(ResourceDemand("gddr0", 0, bytes_moved=64),),
    )
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.0)
    access = resolved.metadata["memory_accesses"][0]
    assert access["offset_bytes"] == offset
    assert access["address"] == offset


def _multi_access_task(accesses):
    return TaskSpec(
        task_id="multi.gddr", request_id="multi-0", name="multi GDDR access",
        category=TaskCategory.MEMORY,
        metadata={"physical_memory_config": _config().__dict__,
                  "memory_accesses": tuple({"physical_owner": "gddr0", "resource_id": "gddr0", **item}
                                            for item in accesses)},
    )


def test_physical_batch_independent_lanes_share_arrival():
    resolved = resolve_physical_task(_multi_access_task((
        {"operation": "read", "address": 0, "byte_count": 1},
        {"operation": "read", "address": 64, "byte_count": 1},
    )), PhysicalRuntimeContext(), 0.0)
    contiguous = resolve_physical_task(_multi_access_task((
        {"operation": "read", "address": 0, "byte_count": 65},
    )), PhysicalRuntimeContext(), 0.0)
    execution = resolved.metadata["physical_execution"]
    assert execution["operation_count"] == 2
    assert execution["completion_ns"] == resolved.metadata["physical_completion_ns"]
    assert execution["completion_ns"] == contiguous.metadata["physical_completion_ns"]
    assert execution["queue_wait_ns"] >= 0


def test_physical_batch_same_lane_contends_and_overlap_write_reads_wait():
    runtime = PhysicalRuntimeContext()
    same_lane = resolve_physical_task(_multi_access_task((
        {"operation": "read", "address": 0, "byte_count": 1},
        {"operation": "read", "address": 128, "byte_count": 1},
    )), runtime, 0.0)
    independent = resolve_physical_task(_multi_access_task((
        {"operation": "read", "address": 0, "byte_count": 1},
        {"operation": "read", "address": 64, "byte_count": 1},
    )), PhysicalRuntimeContext(), 0.0)
    assert same_lane.metadata["physical_completion_ns"] > independent.metadata["physical_completion_ns"]

    overlap = resolve_physical_task(_multi_access_task((
        {"operation": "write", "address": 0, "byte_count": 1},
        {"operation": "read", "address": 0, "byte_count": 1},
    )), PhysicalRuntimeContext(), 0.0)
    intervals = overlap.metadata["physical_resource_intervals"]
    assert overlap.metadata["physical_completion_ns"] >= max(item.end_ns for item in intervals)
