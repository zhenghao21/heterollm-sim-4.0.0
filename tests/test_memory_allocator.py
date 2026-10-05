import pytest

from heterollm_sim.memory_allocator import AllocationError, PhysicalAddressAllocator


def test_allocations_are_stable_and_independent():
    allocator = PhysicalAddressAllocator(512, alignment_bytes=64)
    first = allocator.allocate("activation", 80)
    same = allocator.allocate("activation", 32)
    second = allocator.allocate("weight", 80)

    assert same.base_address == first.base_address
    assert second.base_address >= first.base_address + first.size_bytes
    assert allocator.address("activation", 16, 8) == first.base_address + 16
    expanded_activation = allocator.allocate("activation", 128)
    assert expanded_activation.base_address == first.base_address
    with pytest.raises(AllocationError, match="cannot grow"):
        allocator.allocate("activation", 160)
    expandable = allocator.allocate("scratch", 32)
    expanded = allocator.allocate("scratch", 96)
    assert expanded.base_address == expandable.base_address
    assert expanded.size_bytes == 96


def test_overlap_requires_explicit_alias_and_generation_isolated():
    allocator = PhysicalAddressAllocator(512, alignment_bytes=64)
    original = allocator.allocate("kv", 128, generation=1)
    with pytest.raises(AllocationError, match="overlaps"):
        allocator.allocate("other", 64, address=original.base_address)

    alias = allocator.allocate(
        "kv-view", 64, alias_of="kv", alias_offset_bytes=32, generation=0
    )
    assert alias.base_address == original.base_address + 32
    replacement = allocator.allocate("kv", 128, generation=2)
    assert replacement.base_address != original.base_address
    allocator.release("kv", generation=1)
    assert allocator.address("kv", 0, 1, generation=2) == replacement.base_address


def test_snapshot_restore_preserves_addresses():
    allocator = PhysicalAddressAllocator(256)
    first = allocator.allocate("a", 64)
    snapshot = allocator.snapshot()
    allocator.allocate("b", 64)
    restored = PhysicalAddressAllocator.from_snapshot(snapshot)
    assert restored.address("a", 0, 64) == first.base_address
    with pytest.raises(AllocationError, match="not registered"):
        restored.address("b", 0, 1)
