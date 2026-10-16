"""HOST DRAM residency, legal NAND remapping, and initialized-byte boundaries."""
from copy import deepcopy

import pytest

from heterollm_sim.hbf_staging import HbfStagingController, HbfStagingIO
from heterollm_sim.memory_mapping import map_nand_address
from heterollm_sim.memory_types import AccessRequest, NandConfig
from heterollm_sim.nand_core import NandCore


PAGE = 4096


def config(**updates):
    values = dict(kind="HBF", protocol_mode="ocp_hbf_0_7", channels=1,
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
    cfg = config(channels=256, host_channels=16, blocks_per_plane=2048,
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
