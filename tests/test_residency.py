import random
import unittest
from unittest.mock import patch

import heterollm_sim.residency as residency_module

from heterollm_sim.residency import (
    AccessOperation,
    AllocationLifecycle,
    AllocationResidencyManager,
    CapacityExceededError,
    MemoryAccess,
    MigrationKind,
    PhysicalPool,
    ResidencyError,
    ResidencyValidationError,
)


def _reference_make_room_plan(
    manager,
    additional_bytes,
    *,
    protected_owner=None,
    protected_ranges=(),
):
    """Reproduce the former per-backing-unit ``_make_room`` selector."""

    need = max(
        0,
        manager.pool.resident_bytes
        + additional_bytes
        - manager.pool.capacity_bytes,
    )
    candidates = []
    for owner in manager._allocations.values():
        for start, end, sequence in manager._candidate_recency_ranges(
            owner, protected_owner, protected_ranges
        ):
            residency_module._ensure_backing_unit_ranges(
                "reference eviction candidate",
                [(start, end)],
                owner.committed_bytes,
                owner.residency_granule_bytes,
            )
            candidates.append(
                (
                    sequence,
                    owner._registration_order,
                    owner.allocation_id,
                    start,
                    end,
                    owner,
                )
            )
    candidates.sort(key=lambda item: item[:4])

    plan_order = []
    planned_ranges = {}
    remaining = need
    for _, _, owner_id, start, end, owner in candidates:
        if remaining <= 0:
            break
        for unit_start, unit_end in residency_module._iter_backing_unit_ranges(
            owner.committed_bytes,
            owner.residency_granule_bytes,
            [(start, end)],
        ):
            if remaining <= 0:
                break
            if owner_id not in planned_ranges:
                plan_order.append(owner_id)
                planned_ranges[owner_id] = []
            planned_ranges[owner_id].append((unit_start, unit_end))
            remaining -= unit_end - unit_start
    return plan_order, planned_ranges, remaining


class AllocationResidencyManagerTests(unittest.TestCase):
    def _assert_make_room_matches_reference(
        self,
        manager,
        additional_bytes,
        *,
        protected_owner=None,
        protected_ranges=(),
    ):
        before_snapshot = manager.snapshot()
        before_migrations = manager.migrations
        before_owner_state = {
            owner_id: (
                tuple(owner._resident_ranges),
                tuple(owner._dirty_ranges),
                tuple(owner._resident_recency_ranges),
                owner.resident_bytes,
                owner.dirty_bytes,
                owner.dirty,
                owner.last_access,
            )
            for owner_id, owner in manager._allocations.items()
        }
        plan_order, planned_ranges, remaining = _reference_make_room_plan(
            manager,
            additional_bytes,
            protected_owner=protected_owner,
            protected_ranges=protected_ranges,
        )

        if remaining > 0:
            with self.assertRaises(CapacityExceededError):
                manager._make_room(
                    additional_bytes,
                    protected_owner=protected_owner,
                    protected_ranges=protected_ranges,
                )
            self.assertEqual(manager.snapshot(), before_snapshot)
            self.assertEqual(manager.migrations, before_migrations)
            self.assertEqual(
                {
                    owner_id: (
                        tuple(owner._resident_ranges),
                        tuple(owner._dirty_ranges),
                        tuple(owner._resident_recency_ranges),
                        owner.resident_bytes,
                        owner.dirty_bytes,
                        owner.dirty,
                        owner.last_access,
                    )
                    for owner_id, owner in manager._allocations.items()
                },
                before_owner_state,
            )
            return plan_order, planned_ranges, remaining

        expected_migrations = []
        expected_resident_bytes = before_snapshot.resident_bytes
        for owner_id in plan_order:
            owner = manager.get_allocation(owner_id)
            selected = residency_module._merge_ranges(planned_ranges[owner_id])
            selected_bytes = residency_module._range_bytes(selected)
            dirty_bytes = residency_module._range_bytes(
                residency_module._intersect_ranges(
                    selected, before_owner_state[owner_id][1]
                )
            )
            clean_bytes = selected_bytes - dirty_bytes
            expected_resident_bytes -= selected_bytes
            if dirty_bytes:
                expected_migrations.append(
                    (
                        owner_id,
                        MigrationKind.DIRTY_WRITEBACK,
                        dirty_bytes,
                        manager.pool.component_id,
                        owner.backing,
                        owner.residency_granule_bytes,
                        residency_module._ceil_div(
                            dirty_bytes, owner.residency_granule_bytes
                        ),
                    )
                )
            if clean_bytes:
                expected_migrations.append(
                    (
                        owner_id,
                        MigrationKind.CLEAN_DISCARD,
                        clean_bytes,
                        manager.pool.component_id,
                        owner.backing,
                        owner.residency_granule_bytes,
                        residency_module._ceil_div(
                            clean_bytes, owner.residency_granule_bytes
                        ),
                    )
                )

        manager._make_room(
            additional_bytes,
            protected_owner=protected_owner,
            protected_ranges=protected_ranges,
        )

        for owner_id, owner in manager._allocations.items():
            selected = residency_module._merge_ranges(
                planned_ranges.get(owner_id, ())
            )
            self.assertEqual(
                owner.resident_ranges,
                tuple(
                    residency_module._subtract_ranges(
                        before_owner_state[owner_id][0], selected
                    )
                ),
            )
            self.assertEqual(
                owner.resident_bytes,
                residency_module._range_bytes(owner.resident_ranges),
            )
            self.assertEqual(
                tuple(owner._dirty_ranges),
                tuple(
                    residency_module._subtract_ranges(
                        before_owner_state[owner_id][1], selected
                    )
                ),
            )
            self.assertEqual(
                owner.dirty_bytes,
                residency_module._range_bytes(owner._dirty_ranges),
            )
            self.assertEqual(owner.dirty, bool(owner._dirty_ranges))
            self.assertEqual(
                tuple(owner._resident_recency_ranges),
                tuple(
                    residency_module._subtract_recency_ranges(
                        before_owner_state[owner_id][2], selected
                    )
                ),
            )
        self.assertEqual(manager.resident_bytes, expected_resident_bytes)
        self.assertEqual(
            manager.pool.peak_resident_bytes,
            before_snapshot.peak_resident_bytes,
        )
        new_migrations = manager.migrations[len(before_migrations) :]
        self.assertEqual(
            [
                (
                    migration.allocation_id,
                    migration.kind,
                    migration.byte_count,
                    migration.source,
                    migration.destination,
                    migration.residency_granule_bytes,
                    migration.granule_count,
                )
                for migration in new_migrations
            ],
            expected_migrations,
        )
        self.assertEqual(
            [migration.sequence for migration in new_migrations],
            list(
                range(
                    len(before_migrations) + 1,
                    len(before_migrations) + len(new_migrations) + 1,
                )
            ),
        )
        manager.assert_consistent()
        return plan_order, planned_ranges, remaining

    def test_migrations_since_returns_only_suffix_and_validates_cursor(self):
        manager = AllocationResidencyManager(
            "hbm0", 16, page_size_bytes=8
        )
        manager.register(
            "weights",
            16,
            backing="hostmem0",
            committed_bytes=16,
            residency_granule_bytes=8,
        )
        self.assertEqual(manager.migration_count, 0)
        self.assertEqual(manager.migrations_since(0), ())

        first = manager.read("weights", size_bytes=1)
        cursor = manager.migration_count
        second = manager.read("weights", offset_bytes=8, size_bytes=1)

        self.assertEqual(manager.migrations_since(0), manager.migrations)
        self.assertEqual(manager.migrations_since(cursor), second.migrations)
        self.assertEqual(manager.migrations_since(cursor)[0].sequence, 2)
        self.assertEqual(first.migrations[0].sequence, 1)
        with self.assertRaises(TypeError):
            manager.migrations_since(True)
        with self.assertRaises(ValueError):
            manager.migrations_since(-1)
        with self.assertRaises(ValueError):
            manager.migrations_since(manager.migration_count + 1)

    def test_migration_totals_do_not_rescan_append_only_history(self):
        manager = AllocationResidencyManager("gpu0", 8)
        manager.register(
            "weights",
            8,
            backing="host0",
            resident_bytes=4,
            read_only=True,
        )
        manager.read("weights", offset_bytes=4, size_bytes=4)
        expected = manager.migration_totals

        class NonIterableHistory(list):
            def __iter__(self):
                raise AssertionError("migration totals rescanned history")

        manager._migrations = NonIterableHistory(manager._migrations)
        self.assertEqual(manager.migration_totals, expected)
        self.assertEqual(manager.snapshot().migrations, expected)

    def test_leased_range_is_excluded_until_consumer_release(self):
        manager = AllocationResidencyManager(
            "hbm0", 16, page_size_bytes=8
        )
        manager.register(
            "weights-a",
            16,
            backing="hostmem0",
            committed_bytes=16,
            resident_bytes=16,
            residency_granule_bytes=8,
        )
        manager.read("weights-a", offset_bytes=0, size_bytes=1)
        lease = manager.lease(
            "weights-a",
            "consumer-a",
            lease_id="lease-a",
            offset_bytes=0,
            size_bytes=1,
        )

        manager.register(
            "weights-b",
            8,
            backing="hostmem0",
            committed_bytes=8,
            residency_granule_bytes=8,
        )
        fault = manager.read("weights-b", size_bytes=1)

        self.assertEqual(lease.residency_ranges, ((0, 8),))
        self.assertEqual(manager.allocations["weights-a"].resident_ranges, ((0, 8),))
        self.assertEqual(fault.migrations[0].allocation_id, "weights-a")
        self.assertEqual(fault.migrations[0].consumer_task_ids, ())

        manager.release_lease("lease-a")
        manager.read("weights-a", offset_bytes=8, size_bytes=1)
        eviction = next(
            item
            for item in manager.migrations_since(fault.migrations[-1].sequence)
            if item.kind is not MigrationKind.PAGE_IN
        )
        self.assertEqual(eviction.consumer_task_ids, ("consumer-a",))

    def test_released_lease_anchor_is_latest_and_consumed_on_eviction(self):
        manager = AllocationResidencyManager(
            "hbm0", 8, page_size_bytes=8
        )
        manager.register(
            "weights",
            8,
            backing="hostmem0",
            committed_bytes=8,
            resident_bytes=8,
            residency_granule_bytes=8,
        )
        for index in range(20):
            lease_id = "lease-{}".format(index)
            task_id = "consumer-{}".format(index)
            manager.lease(
                "weights",
                task_id,
                lease_id=lease_id,
                size_bytes=1,
            )
            manager.release_lease(lease_id)
        self.assertEqual(
            manager._released_consumer_ranges,
            {"weights": [(0, 8, "consumer-19")]},
        )

        eviction = manager.evict("weights")

        self.assertEqual(eviction[0].consumer_task_ids, ("consumer-19",))
        self.assertEqual(manager._released_consumer_ranges, {})

    def test_released_lease_interval_updates_match_reference_without_rescanning(self):
        manager = AllocationResidencyManager(
            "hbm0", 4096, page_size_bytes=4
        )
        manager.register(
            "weights",
            4096,
            backing="hostmem0",
            committed_bytes=4096,
            resident_bytes=4096,
            residency_granule_bytes=4,
        )
        reference = []
        randomizer = random.Random(20260901)

        with patch.object(
            residency_module,
            "_subtract_ranges",
            wraps=residency_module._subtract_ranges,
        ) as subtract_ranges:
            for index in range(512):
                start = randomizer.randrange(0, 1024) * 4
                size = randomizer.randrange(1, 65) * 4
                end = min(4096, start + size)
                lease_id = "lease-{}".format(index)
                task_id = "consumer-{}".format(index)
                manager.lease(
                    "weights",
                    task_id,
                    lease_id=lease_id,
                    offset_bytes=start,
                    size_bytes=end - start,
                )

                retained = []
                for anchor_start, anchor_end, anchor_task in reference:
                    retained.extend(
                        (fragment_start, fragment_end, anchor_task)
                        for fragment_start, fragment_end in (
                            residency_module._subtract_ranges(
                                [(anchor_start, anchor_end)],
                                ((start, end),),
                            )
                        )
                    )
                retained.append((start, end, task_id))
                reference = sorted(
                    retained, key=lambda item: (item[0], item[1], item[2])
                )

                before_release_calls = subtract_ranges.call_count
                manager.release_lease(lease_id)
                self.assertEqual(
                    subtract_ranges.call_count,
                    before_release_calls,
                    "single-range lease release must not rescan anchors",
                )
                self.assertEqual(
                    manager._released_consumer_ranges["weights"],
                    reference,
                )

    def test_system_claim_is_pinned_and_cannot_be_evicted(self):
        manager = AllocationResidencyManager(PhysicalPool("hbm0", 10))

        claim = manager.claim_system("driver", 6)

        self.assertTrue(claim.pinned)
        self.assertFalse(claim.evictable)
        self.assertEqual(claim.lifecycle, AllocationLifecycle.SYSTEM)
        self.assertEqual(manager.resident_bytes, 6)
        with self.assertRaises(ResidencyValidationError):
            manager.evict("driver")

    def test_weights_kv_and_state_share_one_physical_pool(self):
        manager = AllocationResidencyManager("memory0", 100)
        manager.claim_system("runtime", 20)
        manager.register(
            "weights", 40, kind="weight", backing="storage", resident_bytes=40
        )
        manager.register(
            "kv", 20, kind="kv", backing="host", resident_bytes=20, dirty=True
        )
        manager.register(
            "state", 10, kind="state", backing="host", resident_bytes=10, dirty=True
        )

        self.assertEqual(manager.resident_bytes, 90)
        self.assertEqual(manager.committed_bytes, 90)
        self.assertEqual(
            {item.kind for item in manager.allocations.values()},
            {"system", "weight", "kv", "state"},
        )
        self.assertEqual(manager.snapshot().available_bytes, 10)

    def test_clean_weight_is_discarded_and_refaulted(self):
        manager = AllocationResidencyManager("hbm0", 8)
        weight = manager.register(
            "weight", 8, kind="weight", backing="host", resident_bytes=8
        )
        manager.register(
            "other", 8, kind="weight", backing="host", resident_bytes=8
        )

        self.assertEqual(weight.resident_bytes, 0)
        self.assertEqual(manager.migrations[-1].kind, MigrationKind.CLEAN_DISCARD)
        result = manager.read("weight")

        self.assertEqual(result.page_in_bytes, 8)
        self.assertEqual(weight.resident_bytes, 8)
        self.assertEqual(result.migrations[-1].kind, MigrationKind.PAGE_IN)

    def test_dirty_kv_and_state_are_written_back_on_eviction(self):
        manager = AllocationResidencyManager("hbm0", 8)
        kv = manager.register(
            "kv", 4, kind="kv", backing="host", resident_bytes=4, dirty=True
        )
        state = manager.register(
            "state", 4, kind="state", backing="host", resident_bytes=4, dirty=True
        )

        manager.register(
            "weights", 8, kind="weight", backing="host", resident_bytes=8
        )

        self.assertEqual(kv.resident_bytes, 0)
        self.assertEqual(state.resident_bytes, 0)
        self.assertFalse(kv.dirty)
        self.assertFalse(state.dirty)
        self.assertEqual(manager.migration_totals.dirty_writeback_bytes, 8)
        self.assertEqual(manager.migration_totals.page_out_bytes, 8)

    def test_alias_has_zero_capacity_and_refreshes_only_its_owner_range(self):
        manager = AllocationResidencyManager("hbm0", 20)
        owner = manager.register(
            "a", 10, backing="host", resident_bytes=10
        )
        victim = manager.register(
            "b", 10, backing="host", resident_bytes=10
        )
        view = manager.create_view("a.head", "a", offset_bytes=0, size_bytes=5)
        before = manager.resident_bytes

        result = manager.read("a.head")
        manager.register("c", 10, backing="host", resident_bytes=10)

        self.assertEqual(view.capacity_bytes, 0)
        self.assertEqual(view.resident_bytes, 0)
        self.assertEqual(manager.resident_bytes, before)
        self.assertEqual(result.owner_id, owner.allocation_id)
        self.assertEqual(owner.resident_ranges, ((0, 5),))
        self.assertEqual(victim.resident_ranges, ((5, 10),))
        self.assertGreater(owner.last_access, victim.last_access)

    def test_two_views_keep_the_touched_view_hot_within_one_owner(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 8, page_size_bytes=4)
        )
        owner = manager.register(
            "backend_weights",
            8,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )
        manager.create_view(
            "weights.hot", "backend_weights", offset_bytes=0, size_bytes=4
        )
        manager.create_view(
            "weights.cold", "backend_weights", offset_bytes=4, size_bytes=4
        )

        hot_access = manager.read("weights.hot")
        manager.register(
            "pressure",
            4,
            backing="host",
            resident_bytes=4,
            residency_granule_bytes=4,
        )

        self.assertEqual(owner.resident_ranges, ((0, 4),))
        self.assertEqual(
            owner.resident_recency_ranges,
            ((0, 4, hot_access.sequence),),
        )
        self.assertEqual(manager.migrations[-1].allocation_id, "backend_weights")

    def test_single_range_recency_fast_path_matches_reference(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 256, page_size_bytes=4)
        )
        owner = manager.register(
            "owner",
            256,
            backing="host",
            resident_bytes=256,
            residency_granule_bytes=4,
        )
        accesses = [
            (0, 256),
            (0, 1),
            (255, 1),
            (64, 64),
            (60, 72),
            (128, 4),
        ]
        randomizer = random.Random(20260831)
        for _ in range(256):
            offset = randomizer.randrange(256)
            accesses.append((offset, randomizer.randrange(1, 257 - offset)))

        for offset, size in accesses:
            before = tuple(owner._resident_recency_ranges)
            result = manager.read(
                owner.allocation_id,
                offset_bytes=offset,
                size_bytes=size,
            )
            retained = residency_module._subtract_recency_ranges(
                before, result.residency_ranges
            )
            expected = residency_module._merge_recency_ranges(
                [
                    *retained,
                    *(
                        (start, end, result.sequence)
                        for start, end in result.residency_ranges
                    ),
                ]
            )
            self.assertEqual(owner._resident_recency_ranges, expected)
            manager.assert_consistent()

        before = tuple(owner._resident_recency_ranges)
        sequence = owner.last_access
        AllocationResidencyManager._refresh_recency(
            owner, ((0, 128),), sequence
        )
        expected = residency_module._merge_recency_ranges(
            [
                *residency_module._subtract_recency_ranges(
                    before, ((0, 128),)
                ),
                (0, 128, sequence),
            ]
        )
        self.assertEqual(owner._resident_recency_ranges, expected)
        manager.assert_consistent()

    def test_owner_and_intra_owner_ranges_share_one_lru_order(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 12, page_size_bytes=4)
        )
        first = manager.register(
            "first",
            8,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )
        second = manager.register(
            "second",
            4,
            backing="host",
            resident_bytes=4,
            residency_granule_bytes=4,
        )
        manager.create_view("first.hot", "first", offset_bytes=0, size_bytes=4)
        manager.read("first.hot")

        manager.register(
            "pressure",
            8,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )

        # first[4:8] is oldest, then all of second, while first[0:4]
        # remains protected by its newer access even though it shares an owner.
        self.assertEqual(first.resident_ranges, ((0, 4),))
        self.assertEqual(second.resident_bytes, 0)
        self.assertEqual(
            [migration.allocation_id for migration in manager.migrations[-2:]],
            ["first", "second"],
        )

    def test_refault_protects_already_resident_bytes_in_accessed_range(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 8, page_size_bytes=4)
        )
        owner = manager.register(
            "owner",
            12,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )

        result = manager.read("owner", offset_bytes=4, size_bytes=8)

        self.assertEqual(result.page_in_bytes, 4)
        self.assertEqual(owner.resident_ranges, ((4, 12),))
        self.assertEqual(manager.migration_totals.clean_discard_bytes, 4)
        manager.assert_consistent()

    def test_dirty_tail_partial_granule_preserves_exact_byte_totals(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 10, page_size_bytes=4)
        )
        owner = manager.register(
            "tail_owner",
            10,
            backing="host",
            resident_bytes=10,
            residency_granule_bytes=4,
        )
        manager.write("tail_owner", offset_bytes=8, size_bytes=2)
        manager.touch("tail_owner", offset_bytes=0, size_bytes=4)

        manager.register(
            "pressure",
            6,
            backing="host",
            resident_bytes=6,
            residency_granule_bytes=4,
        )

        self.assertEqual(owner.resident_ranges, ((0, 4),))
        self.assertEqual(owner.resident_bytes, 4)
        self.assertEqual(owner.dirty_bytes, 0)
        self.assertEqual(manager.resident_bytes, 10)
        self.assertEqual(manager.migration_totals.page_out_bytes, 6)
        self.assertEqual(manager.migration_totals.clean_discard_bytes, 4)
        self.assertEqual(manager.migration_totals.dirty_writeback_bytes, 2)
        manager.assert_consistent()

    def test_temporary_workspace_releases_at_lifecycle_boundary(self):
        manager = AllocationResidencyManager("sram0", 16)
        workspace = manager.register_temporary("workspace", 16)

        self.assertEqual(workspace.lifecycle, AllocationLifecycle.TEMPORARY)
        self.assertEqual(manager.resident_bytes, 16)
        released = manager.release("workspace")

        self.assertEqual(released.lifecycle, AllocationLifecycle.RELEASED)
        self.assertEqual(released.resident_bytes, 0)
        self.assertEqual(manager.resident_bytes, 0)
        with self.assertRaises(KeyError):
            manager.get_allocation("workspace")

    def test_global_lru_is_deterministic_across_kinds(self):
        manager = AllocationResidencyManager("memory0", 20)
        first = manager.register(
            "first", 10, kind="weight", backing="host", resident_bytes=10
        )
        second = manager.register(
            "second", 10, kind="state", backing="host", resident_bytes=10
        )
        manager.touch("first")

        manager.register(
            "third", 10, kind="kv", backing="host", resident_bytes=10
        )

        self.assertEqual(first.resident_bytes, 10)
        self.assertEqual(second.resident_bytes, 0)
        self.assertEqual(manager.migrations[-1].allocation_id, "second")

    def test_capacity_failure_is_atomic(self):
        manager = AllocationResidencyManager("hbm0", 10)
        manager.claim_system("system", 8)
        cold = manager.register("cold", 4, backing="host", resident_bytes=0)

        with self.assertRaises(CapacityExceededError):
            manager.read("cold")

        self.assertEqual(cold.resident_bytes, 0)
        self.assertEqual(manager.resident_bytes, 8)
        self.assertEqual(manager.migrations, ())

    def test_migration_totals_distinguish_all_categories(self):
        manager = AllocationResidencyManager("hbm0", 10)
        weight = manager.register(
            "weight", 10, kind="weight", backing="host", resident_bytes=10
        )
        manager.register(
            "kv", 10, kind="kv", backing="host", resident_bytes=10, dirty=True
        )
        manager.read(weight.allocation_id)

        totals = manager.migration_totals
        self.assertEqual(totals.page_in_bytes, 10)
        self.assertEqual(totals.page_out_bytes, 20)
        self.assertEqual(totals.clean_discard_bytes, 10)
        self.assertEqual(totals.dirty_writeback_bytes, 10)

    def test_partial_residency_and_memory_access_order(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 8, page_size_bytes=4)
        )
        owner = manager.register("tensor", 12, backing="host")

        results = manager.access_many(
            (
                MemoryAccess("tensor", AccessOperation.READ, 8, 4),
                MemoryAccess("tensor", AccessOperation.READ, 0, 4),
                MemoryAccess("tensor", AccessOperation.WRITE, 4, 4),
            )
        )

        self.assertEqual([item.sequence for item in results], sorted(item.sequence for item in results))
        self.assertEqual(owner.resident_bytes, 8)
        self.assertTrue(owner.dirty)
        self.assertEqual(owner.dirty_bytes, 4)
        self.assertEqual(manager.migration_totals.page_in_bytes, 8)
        self.assertEqual(manager.migration_totals.clean_discard_bytes, 4)

    def test_full_granule_write_miss_skips_refault_but_partial_write_refaults(self):
        full_manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 10, page_size_bytes=4)
        )
        full_owner = full_manager.register(
            "full",
            10,
            backing="host",
            residency_granule_bytes=4,
        )

        full_result = full_manager.write("full")

        self.assertEqual(full_result.page_in_bytes, 0)
        self.assertEqual(full_result.migrations, ())
        self.assertEqual(full_owner.resident_ranges, ((0, 10),))
        self.assertEqual(full_owner.dirty_bytes, 10)

        partial_manager = AllocationResidencyManager(
            PhysicalPool("hbm1", 4, page_size_bytes=4)
        )
        partial_owner = partial_manager.register(
            "partial",
            4,
            backing="host",
            residency_granule_bytes=4,
        )

        partial_result = partial_manager.write(
            "partial", offset_bytes=1, size_bytes=1
        )

        self.assertEqual(partial_result.page_in_bytes, 4)
        self.assertEqual(
            [migration.kind for migration in partial_result.migrations],
            [MigrationKind.PAGE_IN],
        )
        self.assertEqual(partial_owner.resident_ranges, ((0, 4),))
        self.assertEqual(partial_owner.dirty_bytes, 4)

    def test_allocations_in_same_pool_can_use_different_residency_granules(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 64, page_size_bytes=4)
        )
        fine = manager.register(
            "fine", 32, backing="host", residency_granule_bytes=4
        )
        coarse = manager.register(
            "coarse", 32, backing="host", residency_granule_bytes=16
        )

        fine_result = manager.read("fine", offset_bytes=5, size_bytes=1)
        coarse_result = manager.read("coarse", offset_bytes=5, size_bytes=1)

        self.assertEqual(fine.residency_granule_bytes, 4)
        self.assertEqual(coarse.residency_granule_bytes, 16)
        self.assertEqual(fine.resident_ranges, ((4, 8),))
        self.assertEqual(coarse.resident_ranges, ((0, 16),))
        self.assertEqual(fine_result.page_in_bytes, 4)
        self.assertEqual(coarse_result.page_in_bytes, 16)

    def test_view_access_uses_owner_residency_granule(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 32, page_size_bytes=4)
        )
        owner = manager.register(
            "owner", 32, backing="host", residency_granule_bytes=16
        )
        view = manager.create_view(
            "owner.window", "owner", offset_bytes=8, size_bytes=4
        )

        result = manager.read("owner.window", offset_bytes=0, size_bytes=1)

        self.assertEqual(view.residency_granule_bytes, owner.residency_granule_bytes)
        self.assertEqual(result.offset_bytes, 8)
        self.assertEqual(result.page_in_bytes, 16)
        self.assertEqual(owner.resident_ranges, ((0, 16),))
        self.assertEqual(result.migrations[0].residency_granule_bytes, 16)
        self.assertEqual(result.migrations[0].granule_count, 1)

    def test_migration_records_residency_granule_and_ceil_granule_count(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 64, page_size_bytes=4)
        )
        manager.register(
            "tail", 20, backing="host", residency_granule_bytes=16
        )

        result = manager.read("tail", offset_bytes=0, size_bytes=17)
        page_in = result.migrations[0]
        clean_discard = manager.evict("tail", 5)[0]

        self.assertEqual(page_in.kind, MigrationKind.PAGE_IN)
        self.assertEqual(page_in.byte_count, 20)
        self.assertEqual(page_in.residency_granule_bytes, 16)
        self.assertEqual(page_in.granule_count, 2)
        self.assertEqual(clean_discard.kind, MigrationKind.CLEAN_DISCARD)
        self.assertEqual(clean_discard.byte_count, 20)
        self.assertEqual(clean_discard.residency_granule_bytes, 16)
        self.assertEqual(clean_discard.granule_count, 2)
        self.assertEqual(manager.get_allocation("tail").resident_bytes, 0)

    def test_initial_prefix_residency_rounds_to_backing_units(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 16, page_size_bytes=4)
        )

        owner = manager.register(
            "prefix",
            10,
            backing="host",
            resident_bytes=5,
            residency_granule_bytes=4,
        )

        self.assertEqual(owner.resident_ranges, ((0, 8),))
        self.assertEqual(owner.resident_bytes, 8)
        self.assertEqual(manager.resident_bytes, 8)
        manager.assert_consistent()

    def test_tail_backing_unit_is_atomic_for_access_and_eviction(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 10, page_size_bytes=4)
        )
        owner = manager.register(
            "tail",
            10,
            backing="host",
            residency_granule_bytes=4,
        )

        result = manager.read("tail", offset_bytes=9, size_bytes=1)
        clean_discard = manager.evict("tail", 1)[0]

        self.assertEqual(result.size_bytes, 1)
        self.assertEqual(result.physical_residency_bytes, 2)
        self.assertEqual(result.residency_ranges, ((8, 10),))
        self.assertEqual(result.page_in_bytes, 2)
        self.assertEqual(result.migrations[0].granule_count, 1)
        self.assertEqual(clean_discard.byte_count, 2)
        self.assertEqual(clean_discard.granule_count, 1)
        self.assertEqual(owner.resident_bytes, 0)

    def test_dirty_and_clean_eviction_categories_keep_full_units(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 8, page_size_bytes=4)
        )
        owner = manager.register(
            "mixed",
            8,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )
        manager.write("mixed", offset_bytes=1, size_bytes=1)

        migrations = manager.evict("mixed", 5)

        self.assertEqual(owner.resident_bytes, 0)
        self.assertEqual(owner.dirty_bytes, 0)
        self.assertEqual(
            [(migration.kind, migration.byte_count) for migration in migrations],
            [
                (MigrationKind.DIRTY_WRITEBACK, 4),
                (MigrationKind.CLEAN_DISCARD, 4),
            ],
        )
        self.assertEqual(manager.migration_totals.page_out_bytes, 8)

    def test_make_room_can_over_release_complete_backing_units(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 10, page_size_bytes=4)
        )
        victim = manager.register(
            "victim",
            8,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )

        pressure = manager.register(
            "pressure",
            4,
            backing="host",
            resident_bytes=4,
            residency_granule_bytes=4,
        )

        self.assertEqual(victim.resident_ranges, ((4, 8),))
        self.assertEqual(pressure.resident_ranges, ((0, 4),))
        self.assertEqual(manager.resident_bytes, 8)
        self.assertEqual(manager.available_bytes, 2)
        self.assertEqual(manager.migrations[-1].byte_count, 4)

    def test_make_room_arithmetic_matches_reference_for_rounding_and_tail(self):
        with self.subTest("remaining smaller than granule"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm0", 10, page_size_bytes=4)
            )
            manager.register(
                "owner",
                8,
                backing="host",
                resident_bytes=8,
                residency_granule_bytes=4,
            )

            _, planned_ranges, remaining = self._assert_make_room_matches_reference(
                manager, 3
            )

            self.assertEqual(planned_ranges, {"owner": [(0, 4)]})
            self.assertEqual(remaining, -3)

        with self.subTest("remaining is not a granule multiple"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm1", 10, page_size_bytes=4)
            )
            manager.register(
                "owner",
                8,
                backing="host",
                resident_bytes=8,
                residency_granule_bytes=4,
            )

            _, planned_ranges, remaining = self._assert_make_room_matches_reference(
                manager, 7
            )

            self.assertEqual(planned_ranges, {"owner": [(0, 4), (4, 8)]})
            self.assertEqual(remaining, -3)

        with self.subTest("committed short tail"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm2", 10, page_size_bytes=4)
            )
            manager.register(
                "owner",
                10,
                backing="host",
                resident_bytes=10,
                residency_granule_bytes=4,
            )
            manager.touch("owner", offset_bytes=0, size_bytes=8)

            _, planned_ranges, remaining = self._assert_make_room_matches_reference(
                manager, 1
            )

            self.assertEqual(planned_ranges, {"owner": [(8, 10)]})
            self.assertEqual(remaining, -1)

    def test_make_room_arithmetic_matches_reference_across_candidates(self):
        with self.subTest("multiple candidates in one owner"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm0", 12, page_size_bytes=4)
            )
            manager.register(
                "owner",
                12,
                backing="host",
                resident_bytes=12,
                residency_granule_bytes=4,
            )
            manager.touch("owner", offset_bytes=4, size_bytes=4)

            plan_order, planned_ranges, remaining = (
                self._assert_make_room_matches_reference(manager, 6)
            )

            self.assertEqual(plan_order, ["owner"])
            self.assertEqual(planned_ranges, {"owner": [(0, 4), (8, 12)]})
            self.assertEqual(remaining, -2)

        with self.subTest("multiple owners with different granules and recency"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm1", 24, page_size_bytes=4)
            )
            manager.register(
                "coarse",
                16,
                backing="host",
                resident_bytes=16,
                residency_granule_bytes=8,
            )
            manager.register(
                "fine",
                8,
                backing="host",
                resident_bytes=8,
                residency_granule_bytes=4,
            )
            manager.touch("coarse")

            plan_order, planned_ranges, remaining = (
                self._assert_make_room_matches_reference(manager, 10)
            )

            self.assertEqual(plan_order, ["fine", "coarse"])
            self.assertEqual(
                planned_ranges,
                {
                    "fine": [(0, 4), (4, 8)],
                    "coarse": [(0, 8)],
                },
            )
            self.assertEqual(remaining, -6)

        with self.subTest("protected range splits a candidate"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm2", 12, page_size_bytes=4)
            )
            manager.register(
                "owner",
                12,
                backing="host",
                resident_bytes=12,
                residency_granule_bytes=4,
            )

            _, planned_ranges, remaining = self._assert_make_room_matches_reference(
                manager,
                5,
                protected_owner="owner",
                protected_ranges=((4, 8),),
            )

            self.assertEqual(planned_ranges, {"owner": [(0, 4), (8, 12)]})
            self.assertEqual(remaining, -3)
            self.assertEqual(
                manager.get_allocation("owner").resident_ranges,
                ((4, 8),),
            )

    def test_make_room_arithmetic_matches_reference_migrations_and_failure(self):
        with self.subTest("dirty and clean selected ranges"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm0", 12, page_size_bytes=4)
            )
            manager.register(
                "owner",
                12,
                backing="host",
                resident_bytes=12,
                residency_granule_bytes=4,
            )
            manager.write("owner", offset_bytes=0, size_bytes=1)

            plan_order, planned_ranges, remaining = (
                self._assert_make_room_matches_reference(manager, 9)
            )

            self.assertEqual(plan_order, ["owner"])
            self.assertEqual(
                planned_ranges,
                {"owner": [(4, 8), (8, 12), (0, 4)]},
            )
            self.assertEqual(remaining, -3)
            self.assertEqual(
                [
                    (migration.kind, migration.byte_count)
                    for migration in manager.migrations
                ],
                [
                    (MigrationKind.DIRTY_WRITEBACK, 4),
                    (MigrationKind.CLEAN_DISCARD, 8),
                ],
            )

        with self.subTest("insufficient eligible capacity is atomic"):
            manager = AllocationResidencyManager(
                PhysicalPool("hbm1", 12, page_size_bytes=4)
            )
            manager.claim_system("system", 8, residency_granule_bytes=4)
            victim = manager.register(
                "victim",
                4,
                backing="host",
                resident_bytes=4,
                residency_granule_bytes=4,
            )

            plan_order, planned_ranges, remaining = (
                self._assert_make_room_matches_reference(manager, 8)
            )

            self.assertEqual(plan_order, ["victim"])
            self.assertEqual(planned_ranges, {"victim": [(0, 4)]})
            self.assertEqual(remaining, 4)
            self.assertEqual(victim.resident_ranges, ((0, 4),))
            self.assertEqual(manager.migrations, ())

    def test_make_room_large_range_does_not_iterate_backing_units(self):
        granule = 64 * 1024
        committed = 4 * 1024**4
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", committed, page_size_bytes=granule)
        )
        owner = manager.register(
            "huge",
            committed,
            backing="host",
            resident_bytes=committed,
            residency_granule_bytes=granule,
        )

        with patch.object(
            residency_module,
            "_iter_backing_unit_ranges",
            side_effect=AssertionError("_make_room enumerated backing units"),
        ) as iterator:
            manager._make_room(committed - 1)

        iterator.assert_not_called()
        self.assertEqual(owner.resident_bytes, 0)
        self.assertEqual(manager.resident_bytes, 0)
        self.assertEqual(len(manager.migrations), 1)
        self.assertEqual(manager.migrations[0].kind, MigrationKind.CLEAN_DISCARD)
        self.assertEqual(manager.migrations[0].byte_count, committed)
        self.assertEqual(manager.migrations[0].granule_count, committed // granule)

    def test_access_result_reports_expanded_physical_residency(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 32, page_size_bytes=4)
        )
        manager.register(
            "owner",
            32,
            backing="host",
            residency_granule_bytes=16,
        )

        result = manager.read("owner", offset_bytes=5, size_bytes=1)

        self.assertEqual(result.size_bytes, 1)
        self.assertEqual(result.offset_bytes, 5)
        self.assertEqual(result.physical_residency_bytes, 16)
        self.assertEqual(result.residency_ranges, ((0, 16),))
        self.assertEqual(result.page_in_bytes, 16)

    def test_deferred_validation_checks_once_at_outer_batch_boundary(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 16, page_size_bytes=4)
        )
        owner = manager.register(
            "owner",
            16,
            backing="host",
            residency_granule_bytes=4,
        )

        manager.begin_deferred_validation()
        manager.begin_deferred_validation()
        manager.read("owner", offset_bytes=1, size_bytes=1)
        manager.write("owner", offset_bytes=8, size_bytes=1)
        manager.end_deferred_validation()
        manager.end_deferred_validation()

        self.assertEqual(owner.resident_ranges, ((0, 4), (8, 12)))
        manager.assert_consistent()
        with self.assertRaisesRegex(ResidencyError, "not active"):
            manager.end_deferred_validation()

        manager.begin_deferred_validation()
        owner._resident_ranges = [(0, 5)]
        owner._resident_recency_ranges = [(0, 5, owner.last_access)]
        with self.assertRaisesRegex(ResidencyError, "backing granule"):
            manager.end_deferred_validation()

    def test_internal_half_granule_state_fails_closed(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 8, page_size_bytes=4)
        )
        owner = manager.register(
            "owner",
            8,
            backing="host",
            resident_bytes=8,
            residency_granule_bytes=4,
        )
        owner._resident_ranges = [(0, 5)]
        owner._resident_recency_ranges = [(0, 5, owner.last_access)]

        with self.assertRaisesRegex(ResidencyError, "backing granule"):
            manager.assert_consistent()

    def test_residency_granule_defaults_to_pool_page_size(self):
        manager = AllocationResidencyManager(
            PhysicalPool("hbm0", 64, page_size_bytes=8)
        )

        owner = manager.register("default", 16, backing="host")
        system = manager.claim_system("system", 4)
        temporary = manager.register_temporary("temporary", 4)
        result = manager.read("default", offset_bytes=1, size_bytes=1)

        self.assertEqual(owner.residency_granule_bytes, 8)
        self.assertEqual(system.residency_granule_bytes, 8)
        self.assertEqual(temporary.residency_granule_bytes, 8)
        self.assertEqual(result.page_in_bytes, 8)
        self.assertEqual(result.migrations[0].residency_granule_bytes, 8)
        self.assertEqual(result.migrations[0].granule_count, 1)

    def test_residency_granule_must_be_positive(self):
        manager = AllocationResidencyManager("hbm0", 16)

        with self.assertRaises(ResidencyValidationError):
            manager.register("bad", 4, residency_granule_bytes=0)
        with self.assertRaises(ResidencyValidationError):
            manager.register_temporary("bad_temporary", 4, residency_granule_bytes=0)
        with self.assertRaises(ResidencyValidationError):
            manager.claim_system("bad_system", 4, residency_granule_bytes=0)

    def test_alias_and_underflow_validation_fail_closed(self):
        manager = AllocationResidencyManager("hbm0", 10)
        manager.register("owner", 10, backing="host", resident_bytes=5)
        manager.create_view("alias", "owner", size_bytes=5)

        with self.assertRaises(ResidencyValidationError):
            manager.create_view("nested", "alias", size_bytes=1)
        with self.assertRaises(ResidencyValidationError):
            manager.create_view("outside", "owner", offset_bytes=9, size_bytes=2)
        with self.assertRaises(ResidencyValidationError):
            manager.evict("owner", 6)
        with self.assertRaises(ResidencyValidationError):
            manager.release("owner")

        self.assertEqual(manager.resident_bytes, 5)
        self.assertEqual(manager.get_allocation("owner").resident_bytes, 5)

    def test_fully_resident_read_only_temporary_discards_without_page_in(self):
        manager = AllocationResidencyManager(
            PhysicalPool("vram0", 128, page_size_bytes=8)
        )
        manager.claim_system("activation-envelope", 16)
        migration_start = len(manager.migrations)
        manager.register(
            "staged-weight",
            37,
            kind="model_weight",
            backing="hostmem0",
            committed_bytes=37,
            resident_bytes=37,
            evictable=True,
            lifecycle="temporary",
            read_only=True,
        )

        read = manager.read("staged-weight")
        manager.evict("staged-weight")
        manager.release("staged-weight")
        migrations = manager.migrations[migration_start:]

        self.assertEqual(read.page_in_bytes, 0)
        self.assertEqual(
            sum(item.page_in_bytes for item in migrations),
            0,
        )
        self.assertEqual(
            sum(item.clean_discard_bytes for item in migrations),
            37,
        )
        self.assertEqual(
            sum(item.dirty_writeback_bytes for item in migrations),
            0,
        )
        self.assertEqual(manager.resident_bytes, 16)
        self.assertEqual(manager.snapshot().allocation_count, 1)


if __name__ == "__main__":
    unittest.main()
