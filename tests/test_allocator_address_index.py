"""Address ordering survives extent changes, imported overlap, and rollback."""
import pytest

from heterollm_sim.memory_allocator import PhysicalAddressAllocator, PhysicalAllocation


def test_workspace_holes_keep_first_fit_and_snapshot_order():
    allocator = PhysicalAddressAllocator(16384, 64, workspace_capacity_bytes=4096)
    for index in range(80):
        allocator.allocate(f"weight{index}", 128)
    a = allocator.allocate("a", 128, 1)
    b = allocator.allocate("b", 192, 1)
    c = allocator.allocate("c", 64, 1)
    assert (a.base_address, b.base_address, c.base_address) == (12288, 12416, 12608)
    allocator.release("b", 1)
    assert allocator.allocate("hole", 64, 2).base_address == 12416
    restored = PhysicalAddressAllocator.from_snapshot(allocator.snapshot())
    assert restored.snapshot() == allocator.snapshot()
    assert restored.allocate("tail", 128, 3).base_address == 12480


def test_imported_cross_boundary_extent_is_not_hidden_by_later_short_row():
    allocator = PhysicalAddressAllocator.from_snapshot({
        "capacity_bytes": 16384, "alignment_bytes": 64, "workspace_capacity_bytes": 4096,
        "allocations": (
            PhysicalAllocation("crossing", 0, 13000),
            PhysicalAllocation("short", 12000, 16),
        ),
    })
    # The long earlier interval crosses the workspace boundary. Looking at
    # just the immediate predecessor of that boundary would miss it.
    assert allocator.allocate("workspace", 64, 1).base_address == 13056


def test_growth_release_and_rollback_refresh_address_index_without_losing_alias():
    allocator = PhysicalAddressAllocator(1 << 100, 64)
    original = allocator.allocate("grow", 64, inferred=True)
    saved = dict(allocator._allocations)
    public_map = allocator._allocations
    grown = allocator.allocate("grow", 192, inferred=True)
    assert grown.size_bytes == 192
    assert allocator.allocate("following", 64).base_address == 192
    allocator.release("following")
    allocator._restore_allocations(saved)
    assert allocator._allocations is public_map
    assert allocator.allocations() == (original,)
    assert allocator.allocate("restored-next", 64).base_address == 64


def test_native_snapshot_preserves_stable_order_for_unordered_values():
    # Native snapshots historically accept dataclass records directly. Their
    # public order must stay equal to sorted(dict.values()), including when
    # a caller supplied an address that is not a total-order key.
    first = PhysicalAllocation("first", float("nan"), 64)
    second = PhysicalAllocation("second", 0, 64)
    allocator = PhysicalAddressAllocator.from_snapshot({
        "capacity_bytes": 1024, "alignment_bytes": 64,
        "allocations": (first, second),
    })
    assert allocator.allocations() == (first, second)


def test_nested_alias_release_and_reuse_preserve_live_target_protection():
    allocator = PhysicalAddressAllocator(16384, 64)
    allocator.allocate("base", 1024)
    allocator.allocate("view", 512, 3, alias_of="base", alias_generation=0,
                       alias_offset_bytes=128)
    allocator.allocate("nested", 128, 8, alias_of="view", alias_generation=3,
                       alias_offset_bytes=64)
    for buffer_id, generation in (("base", 0), ("view", 3)):
        with pytest.raises(ValueError, match="with live aliases"):
            allocator.release(buffer_id, generation)
    allocator.release("nested", 8)
    allocator.release("view", 3)
    allocator.release("base")
    assert allocator.allocate("reuse", 1024).base_address == 0


def test_alias_target_protection_is_rebuilt_after_rollback_and_native_snapshot():
    allocator = PhysicalAddressAllocator(16384, 64)
    allocator.allocate("base", 1024)
    allocator.allocate("view", 128, alias_of="base")
    saved = dict(allocator._allocations)
    allocator.release("view")
    allocator.release("base")
    allocator._restore_allocations(saved)
    restored = PhysicalAddressAllocator.from_snapshot(allocator.snapshot())
    for candidate in (allocator, restored):
        with pytest.raises(ValueError, match="with live aliases"):
            candidate.release("base")
        candidate.release("view")
        candidate.release("base")
        assert candidate.allocations() == ()


def test_reregistering_alias_arena_does_not_leave_target_pinned_after_release():
    allocator = PhysicalAddressAllocator(16384, 64, workspace_capacity_bytes=4096)
    allocator.allocate("base", 1024)
    first = allocator.allocate("view", 128, alias_of="base")
    assert allocator.allocate("view", 128, alias_of="base",
                              allocation_arena="resident") is first
    allocator.release("view")
    allocator.release("base")
    assert allocator.allocations() == ()
