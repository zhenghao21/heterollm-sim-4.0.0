from heterollm_sim.cache_state import CacheAccess, ExplicitCacheState


def _ops(result):
    return tuple(
        (item.operation, item.buffer_id, item.offset_bytes, item.size_bytes, item.allocation_generation)
        for item in result.backing_accesses
    )


def test_partial_hit_reports_only_the_cold_line_fill():
    cache = ExplicitCacheState(128, 64)
    cold = cache.read("activation", 0, 64, buffer_size_bytes=128)
    partial = cache.read("activation", 1, 1, buffer_size_bytes=128)

    assert cold.read_fill_bytes == 64
    assert _ops(cold) == (("read", "activation", 0, 64, 0),)
    assert partial.hit_lines == 1
    assert partial.miss_lines == 0
    assert partial.backing_accesses == ()


def test_one_byte_write_miss_reports_read_for_ownership_and_dirty_victim():
    cache = ExplicitCacheState(64, 64, write_back=True, write_allocate=True)
    first = cache.write("weight", 0, 1, buffer_size_bytes=64, allocation_generation=7)
    second = cache.read("other", 0, 1, buffer_size_bytes=64, allocation_generation=2)

    assert first.read_for_ownership_bytes == 64
    assert _ops(first) == (("read", "weight", 0, 64, 7),)
    assert second.dirty_eviction_bytes == 64
    assert _ops(second) == (
        ("write", "weight", 0, 64, 7),
        ("read", "other", 0, 64, 2),
    )


def test_write_through_and_write_no_allocate_emit_only_actual_writes():
    through = ExplicitCacheState(64, 64, write_back=False, write_allocate=True)
    result = through.write("output", 3, 1, buffer_size_bytes=64, allocation_generation=4)
    assert result.read_fill_bytes == 64
    assert result.write_through_bytes == 1
    assert _ops(result) == (
        ("read", "output", 0, 64, 4),
        ("write", "output", 3, 1, 4),
    )

    bypass = ExplicitCacheState(64, 64, write_back=False, write_allocate=False)
    result = bypass.write("output", 3, 1, buffer_size_bytes=64, allocation_generation=5)
    assert result.read_fill_bytes == 0
    assert result.bypass_write_bytes == 1
    assert _ops(result) == (("write", "output", 3, 1, 5),)


def test_allocation_generation_prevents_stale_line_hits():
    cache = ExplicitCacheState(64, 64)
    cache.read("kv", 0, 1, buffer_size_bytes=64, allocation_generation=1)
    reused = cache.read("kv", 0, 1, buffer_size_bytes=64, allocation_generation=2)

    assert reused.miss_lines == 1
    assert reused.read_fill_bytes == 64
    assert reused.backing_accesses[0].allocation_generation == 2


def test_transaction_rollback_restores_lru_and_counters_without_full_copy():
    cache = ExplicitCacheState(128, 64)
    cache.read("a", 0, 64, buffer_size_bytes=64)
    cache.read("b", 0, 64, buffer_size_bytes=64)
    before = tuple(line.key for line in cache.resident_lines())
    before_snapshot = cache.snapshot()

    transaction = cache.begin_transaction()
    cache.write("c", 0, 64, buffer_size_bytes=64)
    cache.rollback_transaction(transaction)

    assert tuple(line.key for line in cache.resident_lines()) == before
    assert cache.snapshot() == before_snapshot
