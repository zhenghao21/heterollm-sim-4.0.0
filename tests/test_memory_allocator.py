import pytest

from heterollm_sim.memory_allocator import AllocationError, PhysicalAddressAllocator


def test_allocations_are_stable_and_independent():
    allocator = PhysicalAddressAllocator(512, alignment_bytes=64)
    first = allocator.allocate("activation", 80)
    same = allocator.allocate("activation", 80)
    second = allocator.allocate("weight", 80)

    assert same.base_address == first.base_address
    assert second.base_address >= first.base_address + first.size_bytes
    assert allocator.address("activation", 16, 8) == first.base_address + 16
    with pytest.raises(AllocationError, match="cannot grow"):
        allocator.allocate("activation", 128)
    expandable = allocator.allocate("scratch", 32, inferred=True)
    expanded = allocator.allocate("scratch", 96, inferred=True)
    assert expanded.base_address == expandable.base_address
    assert expanded.size_bytes == 96
    allocator.allocate("guard", 64)
    with pytest.raises(AllocationError, match="cannot grow"):
        allocator.allocate("scratch", 160, inferred=True)


def test_overlap_requires_explicit_alias_and_generation_isolated():
    allocator = PhysicalAddressAllocator(512, alignment_bytes=64)
    original = allocator.allocate("kv", 128, generation=1)
    with pytest.raises(AllocationError, match="overlaps"):
        allocator.allocate("other", 64, address=original.base_address)
    with pytest.raises(AllocationError, match="generation 0"):
        allocator.allocate("implicit-view", 16, alias_of="kv", generation=0)

    alias = allocator.allocate(
        "kv-view", 64, alias_of="kv", alias_generation=1, alias_offset_bytes=32, generation=0
    )
    assert alias.base_address == original.base_address + 32
    replacement = allocator.allocate("kv", 128, generation=2)
    assert replacement.base_address != original.base_address
    with pytest.raises(AllocationError, match="live aliases"):
        allocator.release("kv", generation=1)
    allocator.release("kv-view", generation=0)
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


def test_alias_generation_and_canonical_range_are_explicit_and_transitive():
    allocator = PhysicalAddressAllocator(512)
    root = allocator.allocate("buffer", 128, generation=3)
    view = allocator.allocate(
        "view", 64, generation=7, alias_of="buffer", alias_generation=3, alias_offset_bytes=16
    )
    nested = allocator.allocate(
        "nested", 16, generation=2, alias_of="view", alias_generation=7, alias_offset_bytes=8
    )
    assert allocator.lookup("view", 7) == view
    assert allocator.canonical_range("nested", 2, 4, 2) == ("buffer", 3, 26)
    with pytest.raises(AllocationError, match="alias changed"):
        allocator.allocate("view", 64, generation=7, alias_of="buffer", alias_generation=4, alias_offset_bytes=16)
    with pytest.raises(AllocationError, match="live aliases"):
        allocator.release("buffer", generation=3)
    allocator.release("nested", generation=2)
    with pytest.raises(AllocationError, match="live aliases"):
        allocator.release("buffer", generation=3)
    allocator.release("view", generation=7)
    allocator.release("buffer", generation=3)


@pytest.mark.parametrize(
    "field,value",
    (
        ("size_bytes", 64.5),
        ("size_bytes", "64"),
        ("generation", 1.5),
        ("generation", "1"),
        ("alias_offset_bytes", 1.5),
        ("alias_offset_bytes", "0"),
    ),
)
def test_register_rejects_non_integer_descriptor_values(field, value):
    allocator = PhysicalAddressAllocator(256)
    declaration = {"buffer_id": "buffer", "size_bytes": 64}
    declaration[field] = value
    with pytest.raises(AllocationError):
        allocator.register(declaration)
