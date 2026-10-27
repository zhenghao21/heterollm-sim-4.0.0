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


def test_free_extent_first_fit_matches_original_scan_through_fragmentation_and_restore():
    import random
    from heterollm_sim.memory_allocator import AllocationError
    rng = random.Random(9042)
    fast = PhysicalAddressAllocator(32768, 64, workspace_capacity_bytes=8192)
    reference = PhysicalAddressAllocator(32768, 64, workspace_capacity_bytes=8192)
    reference._free_extents = None
    for step in range(500):
        if step % 71 == 0:
            fast = PhysicalAddressAllocator.from_snapshot(fast.snapshot())
            reference = PhysicalAddressAllocator.from_snapshot(reference.snapshot())
            reference._free_extents = None
        saved = dict(fast._allocations)
        if fast._allocations and rng.random() < .45:
            key = rng.choice(list(fast._allocations))
            for allocator in (fast, reference):
                allocator.release(*key)
        else:
            generation = rng.randrange(2)
            size = rng.randrange(1, 17) * 31
            answers = []
            for allocator in (fast, reference):
                try:
                    answers.append(allocator.allocate("item" + str(step), size, generation))
                except AllocationError as error:
                    answers.append(str(error))
            assert answers[0] == answers[1]
        if step % 11 == 0:
            fast._restore_allocations(saved)
            reference._restore_allocations(saved)
            reference._free_extents = None
        assert fast.snapshot() == reference.snapshot()
        assert fast._fit_extents == [(start, end) for start, end in fast._free_extents
                                     if -(-start // 64) * 64 < end]


def test_aliases_never_create_holes_and_growth_consumes_adjacent_free_bytes():
    allocator = PhysicalAddressAllocator(4096, 64)
    allocator.allocate("base", 128, inferred=True)
    allocator.allocate("view", 64, alias_of="base")
    allocator.release("view")
    assert allocator.allocate("next", 64).base_address == 128
    allocator.release("next")
    allocator.allocate("base", 192, inferred=True)
    assert allocator.allocate("after-growth", 64).base_address == 192
    allocator.release("after-growth")
    allocator.release("base")
    assert allocator._free_extents == [(0, 4096)]


class _FullMapSave:
    def __init__(self):
        self.allocations = None


class _FullMapAllocator(PhysicalAddressAllocator):
    """Test oracle: the pre-journal full-map checkpoint algorithm."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unchanged = None

    def begin_transaction(self):
        save = _FullMapSave()
        self._open_saves.append(save)
        return save

    def _current_copy(self):
        if self._unchanged is None:
            self._unchanged = dict(self._allocations)
        return self._unchanged

    def _before_mutation(self, key):
        if self._open_saves and self._open_saves[-1].allocations is None:
            current = self._current_copy()
            for save in reversed(self._open_saves):
                if save.allocations is not None:
                    break
                save.allocations = current
        self._unchanged = None

    def _close(self, save):
        for index in range(len(self._open_saves) - 1, -1, -1):
            if self._open_saves[index] is save:
                for closed in self._open_saves[index:]:
                    if closed.allocations is None:
                        closed.allocations = self._current_copy()
                del self._open_saves[index:]
                return

    def _restore_allocations(self, allocations):
        self._before_mutation(None)
        super()._restore_allocations(allocations)

    def rollback_transaction(self, save):
        self._close(save)
        if save.allocations is not self._unchanged:
            self._restore_allocations(save.allocations)
            self._unchanged = save.allocations


@pytest.mark.parametrize("seed", range(8))
def test_journal_matches_full_map_transactions_through_random_branches(seed):
    import copy
    import random
    from heterollm_sim.memory_allocator import AllocationError

    rng = random.Random(seed)
    pair = [cls(32768, 64, workspace_capacity_bytes=8192)
            for cls in (PhysicalAddressAllocator, _FullMapAllocator)]
    saves = [[], []]
    snapshots = [[], []]
    counts = dict.fromkeys(("begin", "commit", "rollback", "copy", "snapshot", "restore", "from_snapshot"), 0)
    for step in range(600):
        action = rng.randrange(12)
        if action < 2:
            counts["begin"] += 1
            for allocator, checkpoints in zip(pair, saves):
                checkpoints.append(allocator.begin_transaction())
        elif action < 4 and saves[0]:
            index = rng.randrange(len(saves[0]))
            method = "commit_transaction" if action == 2 else "rollback_transaction"
            counts["commit" if action == 2 else "rollback"] += 1
            for allocator, checkpoints in zip(pair, saves):
                getattr(allocator, method)(checkpoints[index])
            # Keep closed saves, and visit them again after branch edits.
        elif action == 4:
            counts["copy"] += 1
            pair, saves, snapshots = copy.deepcopy((pair, saves, snapshots))
        elif action == 5:
            counts["snapshot"] += 1
            for allocator, stored in zip(pair, snapshots):
                stored.append(copy.deepcopy(allocator.snapshot()))
        elif action == 6 and snapshots[0]:
            index = rng.randrange(len(snapshots[0]))
            counts["restore"] += 1
            for allocator, stored in zip(pair, snapshots):
                restored = type(allocator).from_snapshot(stored[index])
                allocator._restore_allocations(dict(restored._allocations))
        elif action == 7 and snapshots[0]:
            counts["from_snapshot"] += 1
            index = rng.randrange(len(snapshots[0]))
            pair = [type(allocator).from_snapshot(stored[index])
                    for allocator, stored in zip(pair, snapshots)]
            saves = [[], []]
        else:
            generation = rng.randrange(2)
            key = "item" + str(rng.randrange(28))
            size = rng.randrange(1, 10) * 31
            alias = rng.choice(pair[0].allocations()) if pair[0]._allocations and rng.random() < .2 else None
            release = rng.random() < .4
            answers = []
            for allocator in pair:
                try:
                    if release:
                        answer = allocator.release(key, generation)
                    elif alias is not None:
                        answer = allocator.allocate(key, 1, generation, alias_of=alias.buffer_id,
                                                    alias_generation=alias.generation)
                    else:
                        answer = allocator.allocate(key, size, generation, inferred=True)
                    answers.append(answer)
                except AllocationError as error:
                    answers.append((type(error), str(error)))
            assert answers[0] == answers[1], (seed, step)
        assert pair[0].snapshot() == pair[1].snapshot(), (seed, step)
        assert list(pair[0]._allocations) == list(pair[1]._allocations), (seed, step)
        assert pair[0]._address_keys == pair[1]._address_keys
        assert pair[0]._free_extents == pair[1]._free_extents
        assert pair[0]._fit_extents == pair[1]._fit_extents
        assert pair[0]._aliases_by_target == pair[1]._aliases_by_target
    assert all(counts.values()), counts


def test_closed_unchanged_nested_saves_are_reusable_after_copy_and_other_rollbacks():
    import copy
    allocator = PhysicalAddressAllocator(4096)
    allocator.allocate("first", 64)
    outer = allocator.begin_transaction()
    inner = allocator.begin_transaction()
    allocator.commit_transaction(outer)  # Also closes the unchanged inner.
    allocator.release("first")
    allocator.allocate("second", 128)
    cloned, saved = copy.deepcopy((allocator, inner))
    for current, checkpoint in ((allocator, inner), (cloned, saved)):
        current.rollback_transaction(checkpoint)
        assert [row.buffer_id for row in current.allocations()] == ["first"]
        current.release("first")
        current.allocate("third", 192)
        current.rollback_transaction(checkpoint)
        assert current.allocate("next", 64).base_address == 64


def test_pickled_allocator_and_closed_checkpoints_keep_their_weak_registry():
    import pickle
    allocator = PhysicalAddressAllocator(4096)
    allocator.allocate("first", 64)
    outer = allocator.begin_transaction()
    allocator.allocate("second", 128)
    inner = allocator.begin_transaction()
    allocator.commit_transaction(outer)
    restored, first, second = pickle.loads(pickle.dumps((allocator, outer, inner)))
    restored.release("first")
    restored.allocate("third", 192)
    restored.rollback_transaction(second)
    assert [row.buffer_id for row in restored.allocations()] == ["first", "second"]
    restored.rollback_transaction(first)
    assert [row.buffer_id for row in restored.allocations()] == ["first"]
