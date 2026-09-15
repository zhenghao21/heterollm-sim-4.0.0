import unittest

from heterollm_sim.cache_state import (
    CacheAccess,
    CacheStateError,
    ExplicitCacheState,
)


class ExplicitCacheStateTests(unittest.TestCase):
    def test_read_miss_then_read_hit_across_invocations(self):
        cache = ExplicitCacheState(capacity_bytes=32, line_bytes=16)
        first = cache.read("weights", 0, 16, buffer_size_bytes=32)
        second = cache.read("weights", 0, 16, buffer_size_bytes=32)

        self.assertEqual((first.touched_lines, first.miss_lines, first.hit_lines), (1, 1, 0))
        self.assertEqual(first.read_fill_bytes, 16)
        self.assertEqual((second.miss_lines, second.hit_lines, second.read_fill_bytes), (0, 1, 0))
        self.assertEqual(cache.snapshot().access_count, 2)
        self.assertEqual(cache.snapshot().hit_lines, 1)

    def test_write_allocate_partial_miss_reads_line_and_dirties_it(self):
        cache = ExplicitCacheState(
            capacity_bytes=16, line_bytes=16,
            write_back=True, write_allocate=True,
        )
        result = cache.write("kv", 4, 4, buffer_size_bytes=16)

        self.assertEqual(result.miss_lines, 1)
        self.assertEqual(result.allocated_lines, 1)
        self.assertEqual(result.read_for_ownership_bytes, 16)
        self.assertEqual(result.backing_read_bytes, 16)
        self.assertEqual(result.backing_write_bytes, 0)
        self.assertEqual(cache.snapshot().dirty_bytes, 16)

    def test_full_line_write_allocate_does_not_need_rfo(self):
        cache = ExplicitCacheState(
            capacity_bytes=16, line_bytes=16,
            write_back=True, write_allocate=True,
        )
        result = cache.write("kv", 0, 16, buffer_size_bytes=16)
        self.assertEqual(result.read_fill_bytes, 0)
        self.assertEqual(result.read_for_ownership_bytes, 0)
        self.assertEqual(result.backing_write_bytes, 0)
        self.assertEqual(cache.snapshot().dirty_bytes, 16)

    def test_no_write_allocate_bypasses_miss_without_residency(self):
        cache = ExplicitCacheState(
            capacity_bytes=16, line_bytes=16,
            write_back=True, write_allocate=False,
        )
        result = cache.write("activation", 0, 8, buffer_size_bytes=16)
        self.assertEqual(result.miss_lines, 1)
        self.assertEqual(result.allocated_lines, 0)
        self.assertEqual(result.read_fill_bytes, 0)
        self.assertEqual(result.bypass_write_bytes, 8)
        self.assertEqual(result.backing_write_bytes, 8)
        self.assertEqual(cache.snapshot().resident_lines, 0)

    def test_write_through_writes_back_on_each_write(self):
        cache = ExplicitCacheState(
            capacity_bytes=16, line_bytes=16,
            write_back=False, write_allocate=True,
        )
        miss = cache.write("out", 0, 4, buffer_size_bytes=16)
        hit = cache.write("out", 0, 4, buffer_size_bytes=16)
        self.assertEqual(miss.write_through_bytes, 4)
        self.assertEqual(hit.write_through_bytes, 4)
        self.assertEqual(cache.snapshot().dirty_lines, 0)
        self.assertEqual(cache.snapshot().write_through_bytes, 8)

    def test_dirty_lru_eviction_is_reported_and_flush_is_conserved(self):
        cache = ExplicitCacheState(capacity_bytes=32, line_bytes=16)
        cache.write("a", 0, 16, buffer_size_bytes=16)
        cache.write("b", 0, 16, buffer_size_bytes=16)
        cache.read("b", 0, 16, buffer_size_bytes=16)  # make a the LRU victim
        eviction = cache.read("c", 0, 16, buffer_size_bytes=16)

        self.assertEqual(eviction.dirty_eviction_lines, 1)
        self.assertEqual(eviction.dirty_eviction_bytes, 16)
        self.assertEqual(eviction.backing_write_bytes, 16)
        self.assertEqual(cache.snapshot().resident_lines, 2)
        flushed = cache.flush()
        self.assertEqual(flushed.flushed_lines, 1)
        self.assertEqual(flushed.writeback_bytes, 16)
        self.assertEqual(cache.snapshot().dirty_lines, 0)

    def test_flush_can_be_scoped_to_one_buffer(self):
        cache = ExplicitCacheState(capacity_bytes=32, line_bytes=16)
        cache.write("a", 0, 16)
        cache.write("b", 0, 16)
        flushed = cache.flush("a")
        self.assertEqual(flushed.buffer_id, "a")
        self.assertEqual(flushed.writeback_bytes, 16)
        self.assertEqual(cache.snapshot().dirty_lines, 1)
        self.assertEqual(cache.snapshot().buffers, ("a", "b"))

    def test_different_buffers_do_not_alias_same_local_offset(self):
        cache = ExplicitCacheState(capacity_bytes=32, line_bytes=16)
        first = cache.read("weights", 0, 16)
        other = cache.read("kv", 0, 16)
        self.assertEqual(first.miss_lines, 1)
        self.assertEqual(other.miss_lines, 1)
        self.assertEqual(cache.snapshot().resident_lines, 2)
        self.assertEqual({line.key for line in cache.resident_lines()}, {("weights", 0), ("kv", 0)})

    def test_capacity_is_line_granular_and_lru_is_deterministic(self):
        cache = ExplicitCacheState(capacity_bytes=32, line_bytes=16)
        cache.read("x", 0, 16)
        cache.read("x", 16, 16)
        cache.read("x", 0, 16)
        evicted = cache.read("x", 32, 16)
        self.assertEqual(evicted.miss_lines, 1)
        self.assertEqual([line.line_index for line in cache.resident_lines()], [0, 2])

    def test_remembered_buffer_bound_is_checked_without_resupplying_size(self):
        cache = ExplicitCacheState(32, 16)
        cache.read("x", 0, 16, buffer_size_bytes=24)
        before = cache.snapshot(), cache.resident_lines()
        with self.assertRaises(CacheStateError):
            cache.read("x", 20, 8)
        self.assertEqual((cache.snapshot(), cache.resident_lines()), before)

    def test_unknown_buffer_extent_cannot_change_existing_line_geometry(self):
        cache = ExplicitCacheState(16, 16)
        cache.read("x", 0, 4)
        before = cache.snapshot(), cache.resident_lines()
        with self.assertRaises(CacheStateError):
            cache.read("x", 0, 4, buffer_size_bytes=4)
        self.assertEqual((cache.snapshot(), cache.resident_lines()), before)

    def test_no_allocate_policy_still_writes_a_resident_read_line(self):
        cache = ExplicitCacheState(16, 16, write_allocate=False)
        cache.read("x", 0, 16)
        write = cache.write("x", 0, 4)
        self.assertEqual((write.hit_lines, write.bypass_write_bytes), (1, 0))
        self.assertEqual(cache.snapshot().dirty_bytes, 16)
        self.assertEqual(cache.flush().writeback_bytes, 16)

    def test_partial_final_line_capacity_and_complete_writeback_accounting(self):
        cache = ExplicitCacheState(32, 16)
        first = cache.write("x", 0, 20, buffer_size_bytes=20)
        self.assertEqual(first.read_for_ownership_bytes, 0)
        self.assertEqual(cache.snapshot().resident_lines, 2)
        self.assertEqual(cache.snapshot().resident_bytes, 20)
        eviction = cache.read("y", 0, 16)
        flush = cache.flush()
        self.assertEqual(eviction.dirty_eviction_bytes + flush.writeback_bytes, 20)
        self.assertEqual(cache.snapshot().backing_write_bytes, 20)
        self.assertEqual(cache.snapshot().flush_writeback_bytes, 4)
        self.assertEqual(cache.flush().writeback_bytes, 0)
        self.assertEqual(cache.snapshot().backing_write_bytes, 20)

    def test_explicit_ranges_and_policy_inputs_fail_closed(self):
        with self.assertRaises(CacheStateError):
            CacheAccess.read("", 0, 1)
        with self.assertRaises(CacheStateError):
            CacheAccess.read("x", -1, 1)
        with self.assertRaises(CacheStateError):
            CacheAccess.read("x", 0, 17, buffer_size_bytes=16)
        with self.assertRaises(CacheStateError):
            ExplicitCacheState(16, 8, write_back="yes")
        cache = ExplicitCacheState(16, 16)
        cache.read("x", 0, 16, buffer_size_bytes=16)
        with self.assertRaises(CacheStateError):
            cache.read("x", 0, 16, buffer_size_bytes=32)


if __name__ == "__main__":
    unittest.main()
