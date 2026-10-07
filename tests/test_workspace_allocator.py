from dataclasses import asdict

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_allocator import AllocationError, PhysicalAddressAllocator
from heterollm_sim.memory_mapping import map_dram_address
from heterollm_sim.memory_types import AccessRequest, DramConfig, Operation


def test_partition_prevents_resident_weight_from_splitting_reusable_workspace():
    legacy = PhysicalAddressAllocator(384, 64)
    legacy.allocate("temporary-a", 128, 1)
    legacy.allocate("resident", 128, 0)
    legacy.release("temporary-a", 1)
    with pytest.raises(AllocationError, match="unable to allocate"):
        legacy.allocate("temporary-b", 192, 1)
    # Total capacity is unchanged. Only this explicitly configured allocator
    # keeps resident objects from splitting the transient free interval.
    partitioned = PhysicalAddressAllocator(384, 64, workspace_capacity_bytes=192)
    first = partitioned.allocate("temporary-a", 128, 1)
    resident = partitioned.allocate("resident", 128, 0)
    partitioned.release("temporary-a", 1)
    second = partitioned.allocate("temporary-b", 192, 1)
    assert resident.base_address == 0
    assert first.base_address == second.base_address == 192
    assert second.base_address + second.size_bytes == partitioned.capacity_bytes == legacy.capacity_bytes


@pytest.mark.parametrize("generation,name", [(0, "resident"), (1, "workspace")])
def test_partition_never_borrows_capacity_from_the_other_arena(generation, name):
    allocator = PhysicalAddressAllocator(512, 64, workspace_capacity_bytes=256)
    with pytest.raises(AllocationError, match=name):
        allocator.allocate("too-large", 257, generation)
    address = 256 if generation == 0 else 0
    with pytest.raises(AllocationError, match=name):
        allocator.allocate("wrong-arena", 64, generation, address=address)
    root = allocator.allocate("root", 128, generation, inferred=True)
    with pytest.raises(AllocationError, match="exceeds"):
        allocator.allocate("root", 257, generation, inferred=True)
    assert allocator.lookup("root", generation) == root


def test_partition_release_reuses_adjacent_space_and_alias_does_not_consume_capacity():
    allocator = PhysicalAddressAllocator(512, 64, workspace_capacity_bytes=256)
    root = allocator.allocate("root", 128, 1)
    allocator.allocate("second", 128, 2)
    # Alias generations are identities, not a second allocation. A resident
    # namespace alias of a transient root follows the target's real address.
    view = allocator.allocate("view", 64, 0, alias_of="root", alias_generation=1, alias_offset_bytes=32)
    assert view.base_address == root.base_address + 32
    with pytest.raises(AllocationError, match="live aliases"):
        allocator.release("root", 1)
    allocator.release("view", 0)
    allocator.release("root", 1)
    allocator.release("second", 2)
    assert allocator.allocate("all-workspace", 256, 3).base_address == 256
    restored = PhysicalAddressAllocator.from_snapshot(allocator.snapshot())
    assert restored.workspace_capacity_bytes == 256
    assert restored.snapshot() == allocator.snapshot()
    with pytest.raises(AllocationError, match="workspace"):
        restored.allocate("excess", 1, 4)


@pytest.mark.parametrize("workspace", [-1, 512, 513, 1.5, "256", True, 255])
def test_partition_rejects_invalid_or_unaligned_configuration(workspace):
    with pytest.raises(AllocationError):
        PhysicalAddressAllocator(512, 64, workspace_capacity_bytes=workspace)


@pytest.mark.parametrize("generation", [None, -1, 1.5, "1", True])
def test_partition_rejects_unknown_generation(generation):
    with pytest.raises(AllocationError, match="generation"):
        PhysicalAddressAllocator(512, 64, workspace_capacity_bytes=256).allocate("invalid", 64, generation)


def test_zero_partition_preserves_legacy_addresses_and_snapshot_shape():
    legacy = PhysicalAddressAllocator(512, 64)
    explicit_zero = PhysicalAddressAllocator(512, 64, workspace_capacity_bytes=0)
    for name, size, generation in (("transient", 80, 1), ("weight", 120, 0), ("other", 16, 2)):
        assert legacy.allocate(name, size, generation) == explicit_zero.allocate(name, size, generation)
    assert legacy.snapshot() == explicit_zero.snapshot()
    assert "workspace_capacity_bytes" not in legacy.snapshot()


def test_context_partition_preserves_real_dram_addresses_bank_mapping_and_timing():
    config = DramConfig(data_lanes=2, banks_per_group=2, row_bytes=1024,
        rows_per_bank=8, capacity_bytes=3072, metadata={"workspace_capacity_bytes": 1536})
    context = PhysicalRuntimeContext(capture_details=False)
    task = TaskSpec("addresses", "cohort-addresses", "addresses", TaskCategory.MEMORY,
        metadata={"physical_owner": "memory", "physical_memory_config": asdict(config),
        "memory_accesses": (
            {"buffer_id": "weight", "byte_count": 64, "buffer_size_bytes": 64,
             "allocation_generation": 0, "operation": "read"},
            {"buffer_id": "activation", "byte_count": 64, "buffer_size_bytes": 64,
             "allocation_generation": 1, "operation": "write"})})
    resolve_physical_task(task, context, 0.0)
    allocator, core = context.allocators["memory"], context.runtimes["memory"].core
    addresses = [allocator.lookup("weight", 0).base_address, allocator.lookup("activation", 1).base_address]
    assert addresses == [0, 1536]
    assert core.config.effective_capacity_bytes == 3072
    reference = DramCore(core.config, capture_details=False)
    for index, (address, operation) in enumerate(zip(addresses, (Operation.READ, Operation.WRITE))):
        reference.submit(AccessRequest(str(index), operation, address, 64, 0.0))
        mapping = map_dram_address(core.config, address)
        assert core._bank_id(mapping) in core._banks
    assert core._banks == reference._banks
    assert core.timeline.ready_ns == reference.timeline.ready_ns
    assert core.timeline.busy_ns == reference.timeline.busy_ns
    assert core.timeline.bytes_moved == reference.timeline.bytes_moved
    assert core._inflight == reference._inflight
    assert core._acceptance_ns == reference._acceptance_ns
