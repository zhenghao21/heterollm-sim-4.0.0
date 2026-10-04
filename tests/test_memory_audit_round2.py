from heterollm_sim.memory_types import AccessRequest, DramConfig, NandConfig
from heterollm_sim.memory_mapping import map_dram_address
from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore


def test_multibank_mapping_is_injective():
    cfg = DramConfig(channels=1, banks_per_group=2, rows_per_bank=2,
                     row_bytes=128, burst_bytes=64, interleave_bytes=64)
    locations = [map_dram_address(cfg, a) for a in range(0, cfg.capacity_bytes, cfg.burst_bytes)]
    assert len({(m.stack, m.die, m.lane, m.rank, m.bank_group, m.bank, m.row, m.column)
                for m in locations}) == len(locations)


def test_parallel_units_limit_array_stages_without_remapping():
    cfg = NandConfig(luns_per_die=2, parallel_units=1, page_bytes=1024,
                     host_granularity_bytes=1024, page_read_ns=100,
                     host_bandwidth_gb_s=1024, internal_bandwidth_gb_s=1024)
    core = NandCore(cfg)
    result = core.run([AccessRequest("a", "read", 0, 1024),
                       AccessRequest("b", "read", 1024, 1024)])
    stages = [s for r in result for s in r.stages if s.name == "PAGE_READ"]
    assert [r.mapping[0].lun for r in result] == [0, 1]
    assert stages[1].start_ns >= stages[0].end_ns


def test_host_granularity_counts_aligned_transactions():
    cfg = NandConfig(page_bytes=16384, host_granularity_bytes=4096)
    result = NandCore(cfg).execute(AccessRequest("u", "read", 4095, 2))
    assert result.host_transfer_bytes == 8192


def test_large_request_keeps_execution_bounded_in_detail():
    cfg = DramConfig(channels=1, banks_per_group=1, rows_per_bank=8,
                     row_bytes=512, burst_bytes=64, max_expanded_segments=2)
    result = DramCore(cfg).execute(AccessRequest("large", "read", 0, 512))
    assert result.counters["burst_count"] == 8
    assert result.counters["details_truncated"]
    assert not result.stages and not result.mapping
