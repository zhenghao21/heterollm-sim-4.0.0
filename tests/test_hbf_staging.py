"""HOST DRAM residency, legal NAND remapping, and initialized-byte boundaries."""
from copy import deepcopy

import pytest

from heterollm_sim.hbf_staging import HbfStagingController, HbfStagingIO
from heterollm_sim.memory_mapping import map_nand_address
from heterollm_sim.memory_types import AccessRequest, NandConfig
from heterollm_sim.nand_core import NandCore


PAGE = 4096


@pytest.mark.parametrize("lanes,pages_per_block", [(1, 4), (3, 4), (8, 7)])
def test_immutable_preload_plan_matches_fragmented_scalar_page_mapping(lanes, pages_per_block):
    import numpy as np
    from heterollm_sim.hbf_staging import _Preload, _preload_read_plan

    preload = _Preload(11 * pages_per_block * PAGE - 64,
                       (range(2, 6), range(17, 24)))
    for first, stop in ((0, 16), (3, 24), (9, 11 * pages_per_block - 1)):
        addresses, blocks, inverse = _preload_read_plan(preload, lanes, pages_per_block, first, stop)
        expected = [preload.page_location(index, lanes, pages_per_block)
                    for index in range(first, stop)]
        assert addresses.tolist() == [
            (block // lanes * pages_per_block * lanes + page * lanes + block % lanes) * PAGE
            for block, page in expected]
        assert [blocks[index] for index in inverse] == [block for block, _ in expected]
        assert _preload_read_plan(preload, lanes, pages_per_block, first, stop)[0] is addresses
        with pytest.raises(ValueError):
            addresses.flags.writeable = True
        ready = np.zeros(len(blocks))
        completions = np.arange(stop - first, dtype=float)[::-1]  # not monotone by block
        np.maximum.at(ready, inverse, completions)
        assert dict(zip(blocks, ready.tolist())) == {
            block: max(time for (actual, _), time in zip(expected, completions) if actual == block)
            for block in blocks}


def config(**updates):
    values = dict(kind="HBF", protocol_mode="ocp_hbf_0_7", channels=1,
                  queue_scope="host_channel", planes_independent=True,
                  planes_per_lun=1, parallel_units=None,
                  blocks_per_plane=8, pages_per_block=4, page_bytes=PAGE,
                  host_granularity_bytes=64, host_full_duplex=True,
                  read_cache_pages_per_bank=2, host_bandwidth_gb_s=64.,
                  host_read_bandwidth_gb_s=64., host_write_bandwidth_gb_s=64.,
                  internal_bandwidth_gb_s=64., page_read_ns=100.,
                  page_program_ns=1000., block_erase_ns=2000.)
    values.update(updates)
    return NandConfig(**values)


class Hardware:
    def __init__(self, cfg, host_capacity=3 * PAGE):
        self.core = NandCore(cfg)
        self.host_capacity = host_capacity
        self.events = []
        self.results = []
        self.io = HbfStagingIO(self.read_host, self.write_host, self.read_flash,
                              self.write_flash, self.transfer)

    def host(self, kind, address, count, time):
        assert count > 0 and 0 <= address < address + count <= self.host_capacity
        self.events.append((kind, address, count, time))
        return time + count / 64.

    def read_host(self, address, count, time):
        return self.host("host_read", address, count, time)

    def write_host(self, address, count, time):
        return self.host("host_write", address, count, time)

    def flash(self, kind, address, count, time):
        self.events.append((kind, address, count, time))
        result = self.core.submit(AccessRequest(str(len(self.events)), kind.removeprefix("flash_"),
                                               address, count, time))
        self.results.append(result)
        return result.completion_ns

    def read_flash(self, address, count, time):
        return self.flash("flash_read", address, count, time)

    def write_flash(self, address, count, time):
        assert address % PAGE == count % PAGE == 0
        return self.flash("flash_write", address, count, time)

    def transfer(self, route, count, time):
        self.events.append((route, 0, count, time))
        return time + count / 32.

    def bytes(self, kind):
        return sum(count for event, _, count, _ in self.events if event == kind)


class StreamHardware(Hardware):
    """Real scalar NAND oracle presented through the numeric stream API."""

    def __init__(self, cfg, host_capacity=3 * PAGE):
        from types import SimpleNamespace
        super().__init__(cfg, host_capacity)
        self.io = SimpleNamespace(**vars(self.io), read_flash_pages=self.read_pages)

    def read_pages(self, addresses, arrivals):
        import numpy as np
        if np.isscalar(arrivals):
            arrivals = [arrivals] * len(addresses)
        return np.asarray([self.transfer("read_hbf", PAGE,
            self.read_flash(int(address), PAGE, float(arrival)))
            for address, arrival in zip(addresses, arrivals)])


def test_dense_kv_reads_keep_exact_dependencies_across_tail_growth_and_sparse_append():
    cfg = config(blocks_per_plane=64)
    controllers, hardware, completions = [], [], []
    for hardware_type in (Hardware, StreamHardware):
        staging, hw = HbfStagingController(cfg, 3 * PAGE), hardware_type(cfg)
        ends = [staging.write("K", 0, 0, 40 * PAGE, 0., hw.io)]
        ends.append(staging.read("K", 0, 0, 40 * PAGE, 0., hw.io))
        for token in range(32):
            offset = 40 * PAGE + token * 256
            ends.append(staging.write("K", 0, offset, 256, 0., hw.io))
            ends.append(staging.read("K", 0, 0, offset + 256, 0., hw.io))
        if hardware_type is StreamHardware:
            state = staging._read_states[("K", 0)]
            assert len(state.plans) <= 4
            assert all(page.ready_ns == state.ready[index]
                       for (owner, generation, index), page in staging._pages.items())
        # Large sparse offsets must not expand a cached dense page array.
        ends.append(staging.write("K", 0, 100_000 * PAGE, 64, 0., hw.io))
        assert not staging._read_states
        controllers.append(staging.snapshot())
        hardware.append(hw)
        completions.append(ends)
    assert completions[0] == completions[1]
    assert controllers[0] == controllers[1]
    assert hardware[0].events == hardware[1].events


def test_dense_ready_journal_survives_nested_rollback_rebind_and_release():
    cfg = config(blocks_per_plane=64)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), StreamHardware(cfg)
    staging.write("K", 0, 0, 32 * PAGE, 0., hw.io)
    staging.read("K", 0, 0, 32 * PAGE, 0., hw.io)
    before = staging.snapshot()
    outer = staging.begin_transaction()
    staging.read("K", 0, 0, 32 * PAGE, 0., hw.io)
    assert len(outer["read_ready"]) == 1 and not outer["page_ready"]
    after_read = staging.snapshot()
    inner = staging.begin_transaction()
    staging.write("K", 0, 32 * PAGE, PAGE, 0., hw.io)  # grow ready vector
    staging.write("K", 0, 4 * PAGE + 64, 64, 0., hw.io)
    staging.read("K", 0, 0, 33 * PAGE, 0., hw.io)
    staging.rollback_transaction(inner)
    assert staging.snapshot() == after_read
    staging.read("K", 0, 0, 32 * PAGE, 0., hw.io)  # rebind after rollback
    staging.release("K", 0)
    staging.rollback_transaction(outer)
    assert staging.snapshot() == before
    saved = staging.snapshot()
    staging.read("K", 0, 0, 32 * PAGE, 0., hw.io)
    staging.restore(saved)
    assert staging.snapshot() == saved


def test_dense_validity_checks_holes_before_any_hardware_submission():
    cfg = config(blocks_per_plane=64)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), StreamHardware(cfg)
    staging.write("K", 0, 0, 32 * PAGE, 0., hw.io)
    staging.write("K", 0, 33 * PAGE, 256, 0., hw.io)
    staging.read("K", 0, 0, 32 * PAGE, 0., hw.io)
    before = len(hw.events)
    with pytest.raises(ValueError, match="uninitialized"):
        staging.read("K", 0, 0, 33 * PAGE + 256, 0., hw.io)
    assert len(hw.events) == before


def test_scalar_frontier_journal_keeps_nested_delete_order_without_full_copy():
    from heterollm_sim.hbf_staging import _JournalDict
    values = _JournalDict({2: 4, 7: 8, 3: 9})
    outer = values.begin()
    values[2] = 5
    values[10] = 1
    inner = values.begin()
    values.pop(7)
    values[2] = 6
    values.update_max((2, 3, 11), (4.0, 17.0, 3.0))
    values.rollback(inner)
    assert tuple(values.items()) == ((2, 5), (7, 8), (3, 9), (10, 1))
    values.pop(3)
    values.rollback(outer)
    assert tuple(values.items()) == ((2, 4), (7, 8), (3, 9))


def test_gc_delete_journal_does_not_rescan_frontiers_after_first_collection():
    from heterollm_sim.hbf_staging import _JournalDict

    class CountedMap(_JournalDict):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    values = CountedMap({block: 256 for block in range(8191)})
    for block in range(128):
        token = values.begin()
        values.pop(block)
        values[8191 + block] = 1
        assert len(token["values"]) == 2
        assert len(token["deleted_positions"]) == 1
        values.commit(token)
    assert values.scans == 1
    assert tuple(values) == tuple(range(128, 8319))


@pytest.mark.parametrize("seed", range(8))
def test_delete_order_journal_matches_plain_dict_at_nested_transaction_boundaries(seed):
    import random
    from heterollm_sim.hbf_staging import _JournalDict

    rng = random.Random(seed)
    values = _JournalDict({key: -key for key in range(16)})
    expected = dict(values)
    stack = []
    for _ in range(800):
        choice = rng.randrange(12)
        key, value = rng.randrange(24), rng.randrange(100)
        if choice < 2 and len(stack) < 4:
            stack.append((values.begin(), expected.copy()))
        elif choice == 2 and stack:
            token, saved = stack.pop()
            values.rollback(token)
            expected = saved
        elif choice == 3 and stack:
            token, _ = stack.pop()
            values.commit(token)
        elif choice in (4, 5):
            assert values.pop(key, None) == expected.pop(key, None)
        elif choice == 6:
            updates = {key: value, (key + 1) % 24: value + 1}
            values.update(updates)
            expected.update(updates)
        elif choice == 7:
            keys = (key, (key + 1) % 24)
            updates = (float(value), float(value + 1))
            values.update_max(keys, updates)
            for changed, ready in zip(keys, updates):
                expected[changed] = max(ready, expected.get(changed, 0.))
        else:
            values[key] = value
            expected[key] = value
        assert tuple(values.items()) == tuple(expected.items())
    while stack:
        token, expected = stack.pop()
        values.rollback(token)
        assert tuple(values.items()) == tuple(expected.items())


def test_delete_order_cache_is_reset_by_clear_and_rebuilt_only_when_needed():
    from heterollm_sim.hbf_staging import _JournalDict
    values = _JournalDict({0: 1, 1: 2})
    token = values.begin()
    values.pop(0)
    values.commit(token)
    values.clear()
    assert values._positions is None
    values.update({6: 9, 2: 3})
    token = values.begin()
    values.clear()
    values[1] = 3
    values.rollback(token)
    assert tuple(values.items()) == ((6, 9), (2, 3))


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("dirty_tail", [False, True])
def test_clean_numeric_release_preserves_partial_blocks_and_all_remaining_state(fragmented, dirty_tail):
    cfg = config(channels=3, blocks_per_plane=64)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), StreamHardware(cfg)
    for index in range(37):
        if fragmented and index % 3 == 0:
            staging.write("other", 0, index * PAGE, PAGE, 0., hw.io)
        staging.write("K", 7, index * PAGE, PAGE, 0., hw.io)
    if dirty_tail:
        staging.write("K", 7, 37 * PAGE, 256, 0., hw.io)
    staging.read("K", 7, 0, 37 * PAGE, 0., hw.io)
    saved = staging.snapshot()
    reference = HbfStagingController(cfg, 3 * PAGE)
    reference.restore(saved)  # No numeric read state: independent scalar release.
    kept_owner_maps = tuple(staging._owners.items())
    assert staging._read_states[("K", 7)]
    assert staging.release("K", 7, 19.) == reference.release("K", 7, 19.)
    assert staging.snapshot() == reference.snapshot()
    assert all(staging._owners[block] is owners for block, owners in kept_owner_maps)
    # Reuse the logical generation after release, with exactly the same
    # available slots, physical frontiers, and dependency clocks.
    assert staging._slots == reference._slots
    assert staging._block_ready == reference._block_ready


def test_sixteen_small_kv_writes_reach_flash_only_when_page_is_full():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    time = staging.write("K", 0, 0, 256, 0, hw.io)
    time = staging.read("K", 0, 0, 256, time, hw.io)
    assert hw.bytes("host_read") == 256
    assert hw.bytes("from_host") == 256
    assert hw.bytes("flash_write") == hw.bytes("flash_read") == 0
    for token in range(1, 16):
        time = staging.write("K", 0, token * 256, 256, time, hw.io)
    assert hw.bytes("to_host") == PAGE
    assert hw.bytes("flash_write") == PAGE
    assert staging.host_occupied_bytes == 0
    assert staging.stats.get("padding_bytes", 0) == 0
    staging.read("K", 0, 0, PAGE, time, hw.io)
    assert hw.bytes("flash_read") == PAGE


def test_partial_overwrite_reads_host_and_flash_and_flushes_out_of_place():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    time = staging.write("K", 0, 0, PAGE, 0, hw.io)
    first_address = next(e[1] for e in hw.events if e[0] == "flash_write")
    time = staging.write("K", 0, 256, 256, time, hw.io)
    hw.events.clear()
    time = staging.read("K", 0, 0, PAGE, time, hw.io)
    assert hw.bytes("from_host") == 256
    assert hw.bytes("flash_read") == PAGE - 256
    assert hw.bytes("read_hbf") == PAGE - 256
    time = staging.flush("K", 0, time, hw.io)
    assert staging.stats["rmw_read_bytes"] == PAGE - 256
    assert next(e[1] for e in hw.events if e[0] == "flash_write") != first_address
    assert staging.stats["flash_write_bytes"] == 2 * PAGE


def test_host_pressure_pads_tail_but_never_initializes_unwritten_kv():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 2 * PAGE), Hardware(cfg, 2 * PAGE)
    time = staging.write("K", 0, 0, 256, 0, hw.io)
    time = staging.write("V", 0, 0, 256, time, hw.io)
    assert staging.host_occupied_bytes == PAGE
    assert staging.host_reserved_bytes == PAGE
    assert staging.stats["pressure_flushes"] == 1
    assert staging.stats["padding_bytes"] == PAGE - 256
    assert hw.bytes("host_write") == PAGE + 256
    staging.read("K", 0, 0, 256, time, hw.io)
    before = list(hw.events)
    with pytest.raises(ValueError, match="uninitialized"):
        staging.read("K", 0, 0, 512, time, hw.io)
    assert hw.events == before


def test_cross_page_read_and_write_keep_buffer_and_generation_separate():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    time = staging.write("K", 7, PAGE - 128, 256, 0, hw.io)
    assert staging.host_occupied_bytes == 2 * PAGE
    staging.read("K", 7, PAGE - 128, 256, time, hw.io)
    assert hw.bytes("from_host") == 256
    for name, gen in [("K", 8), ("V", 7)]:
        with pytest.raises(ValueError, match="uninitialized"):
            staging.read(name, gen, PAGE - 128, 64, time, hw.io)
    staging.release("K", 7, time)
    assert staging.occupied_bytes == staging.host_occupied_bytes == 0
    with pytest.raises(ValueError, match="uninitialized"):
        staging.read("K", 7, PAGE - 128, 64, time, hw.io)


def test_gc_at_full_logical_capacity_preserves_live_pages_and_program_order():
    cfg = config(blocks_per_plane=3, pages_per_block=2)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    time = staging.write("K", 0, 0, 4 * PAGE, 0, hw.io)
    assert staging.occupied_bytes == staging.usable_capacity_bytes == 4 * PAGE
    for _ in range(8):
        time = staging.write("K", 0, 0, PAGE, time, hw.io)
        time = staging.read("K", 0, 0, 4 * PAGE, time, hw.io)
    assert staging.stats["gc_cycles"] == 8
    assert staging.stats["gc_copied_pages"] == 8
    assert hw.bytes("from_hbf") == 8 * PAGE
    assert sum(r.counters.get("erase_operations", 0) for r in hw.results) >= 8
    with pytest.raises(MemoryError, match="logical capacity"):
        staging.write("V", 0, 0, 64, time, hw.io)


def test_release_reclaims_physical_pages_without_durable_tail_write():
    cfg = config(blocks_per_plane=3, pages_per_block=2)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    time = staging.write("old", 0, 0, 3 * PAGE + 64, 0, hw.io)
    writes = hw.bytes("flash_write")
    time = staging.release("old", 0, time)
    assert hw.bytes("flash_write") == writes
    time = staging.write("new", 1, 0, 4 * PAGE, time, hw.io)
    staging.read("new", 1, 0, 4 * PAGE, time, hw.io)
    assert staging.stats["gc_cycles"] > 0


def test_preload_is_immutable_compressed_and_charges_block_padding():
    cfg = config(channels=2, host_channels=2)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    staging.preload("weights", 0, PAGE + 64)
    assert staging.occupied_bytes == cfg.block_bytes
    assert len(staging._pages) == 0
    assert staging.stats["preload_padding_bytes"] == cfg.block_bytes - PAGE - 64
    assert not hw.events
    for address, frontier in staging.programmed_page_counts.items():
        hw.core._next_program_page[hw.core._block_key(map_nand_address(cfg, address))] = frontier
    time = staging.read("weights", 0, 0, PAGE + 64, 0, hw.io)
    with pytest.raises(ValueError, match="uninitialized"):
        staging.read("weights", 0, PAGE + 64, 64, time, hw.io)
    with pytest.raises(ValueError, match="immutable"):
        staging.write("weights", 0, 0, 64, time, hw.io)
    staging.write("K", 0, 0, PAGE, time, hw.io)
    loaded_block = staging._preloads[("weights", 0)].block_at(0)
    dynamic_block = staging._pages[("K", 0, 0)].flash[0]
    assert dynamic_block != loaded_block


def test_snapshot_restores_dirty_ranges_frontiers_and_slot_dependencies():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    time = staging.write("K", 0, 0, 256, 0, hw.io)
    state = staging.snapshot()
    core_state = deepcopy(hw.core.__dict__)
    completion = staging.flush("K", 0, time, hw.io)
    stats = deepcopy(staging.stats)
    staging.restore(state)
    hw.core.__dict__ = deepcopy(core_state)
    assert staging.host_occupied_bytes == PAGE
    assert staging.flush("K", 0, time, hw.io) == completion
    assert staging.stats == stats


def test_transaction_rollback_restores_gc_mappings_pressure_flush_and_release():
    cfg = config(blocks_per_plane=3, pages_per_block=2)
    staging, hw = HbfStagingController(cfg, 2 * PAGE), Hardware(cfg, 2 * PAGE)
    now = staging.write("K", 0, 0, 4 * PAGE, 0, hw.io)
    before = staging.snapshot()
    token = staging.begin_transaction()
    now = staging.write("K", 0, 0, 64, now, hw.io)
    now = staging.write("K", 0, PAGE, 64, now, hw.io)
    now = staging.flush("K", 0, now, hw.io)
    now = staging.read("K", 0, 0, 4 * PAGE, now, hw.io)
    staging.release("K", 0, now)
    assert staging.stats["gc_cycles"] > before["stats"].get("gc_cycles", 0)
    assert staging.stats["pressure_flushes"] > 0
    staging.rollback_transaction(token)
    assert staging.snapshot() == before


def test_nested_transactions_keep_outer_undo_and_inner_rollback_independent():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    before = staging.snapshot()
    outer = staging.begin_transaction()
    now = staging.write("K", 0, 0, 64, 0, hw.io)
    after_outer = staging.snapshot()
    inner = staging.begin_transaction()
    staging.release("K", 0, now)
    staging.preload("weights", 0, PAGE)
    staging.rollback_transaction(inner)
    assert staging.snapshot() == after_outer
    inner = staging.begin_transaction()
    staging.write("V", 0, 0, 64, now, hw.io)
    staging.commit_transaction(inner)
    staging.rollback_transaction(outer)
    assert staging.snapshot() == before
    with pytest.raises(ValueError, match="finish once"):
        staging.commit_transaction(outer)


def test_transaction_records_only_pages_and_blocks_it_changes():
    cfg = config(blocks_per_plane=64)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    now = staging.write("K", 0, 0, 128 * PAGE, 0, hw.io)
    token = staging.begin_transaction()
    assert not token["pages"] and not token["owners"]
    staging.write("K", 0, 127 * PAGE, 64, now, hw.io)
    assert tuple(token["pages"]) == (("K", 0, 127),)
    assert not token["owners"]
    staging.commit_transaction(token)
    saved = staging.snapshot()
    # Public snapshots remain repeatable and independent of live transactions.
    staging.release("K", 0, now)
    staging.restore(saved)
    staging.release("K", 0, now)
    staging.restore(saved)
    assert staging.snapshot() == saved


def test_read_scalar_journal_promotes_to_full_page_before_nested_mutation():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    now = staging.write("K", 0, 0, PAGE, 0, hw.io)
    before = staging.snapshot()
    outer = staging.begin_transaction()
    now = staging.read("K", 0, 0, PAGE, now, hw.io)
    assert not outer["pages"]
    assert outer["page_ready"] == {("K", 0, 0): before["_pages"][("K", 0, 0)].ready_ns}
    after_read = staging.snapshot()
    inner = staging.begin_transaction()
    staging.write("K", 0, 0, 64, now, hw.io)
    assert not outer["page_ready"] and ("K", 0, 0) in outer["pages"]
    staging.rollback_transaction(inner)
    assert staging.snapshot() == after_read
    staging.release("K", 0, now)
    staging.rollback_transaction(outer)
    assert staging.snapshot() == before


def test_buffer_page_index_rollback_preserves_sparse_insertion_order():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 4 * PAGE), Hardware(cfg, 4 * PAGE)
    for index in (9, 1, 4):
        staging.write("K", 0, index * PAGE, 64, 0, hw.io)
    before = staging.snapshot()
    token = staging.begin_transaction()
    staging.write("K", 0, 4 * PAGE, 64, 0, hw.io)
    staging.release("K", 0)
    staging.write("K", 0, 8 * PAGE, 64, 0, hw.io)
    staging.rollback_transaction(token)
    assert staging.snapshot() == before
    assert tuple(staging._buffer_pages[("K", 0)]) == (9, 1, 4)


def test_append_after_pressure_flush_uses_rmw_and_finishes_full_page():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 2 * PAGE), Hardware(cfg, 2 * PAGE)
    time = staging.write("K", 0, 0, 256, 0, hw.io)
    time = staging.write("V", 0, 0, 256, time, hw.io)
    time = staging.write("K", 0, 256, PAGE - 256, time, hw.io)
    assert staging._pages[("K", 0, 0)].slot is None
    assert staging.stats["rmw_read_bytes"] == 256
    staging.read("K", 0, 0, PAGE, time, hw.io)


def test_zero_length_operations_are_noops_and_bad_capacity_is_rejected():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    assert staging.write("K", 0, 1, 0, 12., hw.io) == 12.
    assert staging.read("K", 0, 1, 0, 12., hw.io) == 12.
    assert not hw.events and not staging._pages
    with pytest.raises(ValueError, match="two aligned"):
        HbfStagingController(cfg, PAGE)
    with pytest.raises(ValueError, match="two complete"):
        HbfStagingController(config(blocks_per_plane=1), 2 * PAGE)


def test_preloaded_pages_stripe_over_banks_and_read_concurrently():
    cfg = config(channels=4, host_channels=4)
    staging, hw = HbfStagingController(cfg, 5 * PAGE), Hardware(cfg, 5 * PAGE)
    staging.preload("weights", 0, 4 * cfg.block_bytes)
    completion = staging.read("weights", 0, 0, 4 * PAGE, 0, hw.io)
    reads = [event for event in hw.events if event[0] == "flash_read"]
    assert [address for _, address, _, _ in reads] == [0, PAGE, 2 * PAGE, 3 * PAGE]
    assert {arrival for _, _, _, arrival in reads} == {0.0}
    assert len({result.completion_ns for result in hw.results}) == 1
    assert completion == hw.results[0].completion_ns + PAGE / 32.


def test_complete_pages_stripe_concurrently_without_using_host_staging():
    cfg = config(channels=4, host_channels=4)
    times = []
    for host_pages in (2, 5):
        staging = HbfStagingController(cfg, host_pages * PAGE)
        hw = Hardware(cfg, host_pages * PAGE)
        times.append(staging.write("K", 0, 0, 4 * PAGE, 0, hw.io))
        writes = [event for event in hw.events if event[0] == "flash_write"]
        assert [address for _, address, _, _ in writes] == [0, PAGE, 2 * PAGE, 3 * PAGE]
        assert len({arrival for _, _, _, arrival in writes}) == 1
        assert len({result.completion_ns for result in hw.results}) == 1
        assert hw.bytes("host_read") == hw.bytes("host_write") == 0
        assert hw.bytes("write_hbf") == 4 * PAGE
    assert times[0] == times[1]


def test_large_device_and_preload_have_constant_size_static_snapshot_state():
    cfg = config(channels=16, planes_per_lun=16, host_channels=16, blocks_per_plane=2048,
                 pages_per_block=256)
    staging = HbfStagingController(cfg, 4 * PAGE)
    assert cfg.effective_capacity_bytes == 512 * 1024 ** 3
    assert staging.reserved_bytes == cfg.block_bytes
    staging.preload("weights", 0, 140_000_000_000)
    state = staging.snapshot()
    assert state["_preloads"][("weights", 0)] is staging._preloads[("weights", 0)]
    assert len(state["_free_blocks"].ranges) == 1
    assert len(state["_used"]) == len(state["_pages"]) == 0


def test_preload_cannot_consume_capacity_promised_to_host_resident_pages():
    cfg = config(blocks_per_plane=3, pages_per_block=2)
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    staging.write("K", 0, 0, 64, 0, hw.io)
    with pytest.raises(MemoryError, match="preload"):
        staging.preload("weights", 0, 3 * PAGE)


def test_read_transaction_keeps_free_host_queue_without_copying_it():
    cfg = config()
    staging, hw = HbfStagingController(cfg, 3 * PAGE), Hardware(cfg)
    now = staging.write("K", 0, 0, PAGE, 0, hw.io)
    before = staging.snapshot()
    slots = staging._slots
    outer = staging.begin_transaction()
    now = staging.read("K", 0, 0, PAGE, now, hw.io)
    assert "_slots" not in outer["state"]
    assert staging._slots is slots
    # A nested mutation must save both transaction boundaries before taking
    # a slot, even though the outer transaction started with a read.
    inner = staging.begin_transaction()
    staging.write("V", 0, 0, 64, now, hw.io)
    assert outer["state"]["_slots"] == before["_slots"]
    assert inner["state"]["_slots"] == before["_slots"]
    assert outer["state"]["_slots"] is not inner["state"]["_slots"]
    staging.commit_transaction(inner)
    staging.rollback_transaction(outer)
    assert staging.snapshot() == before
