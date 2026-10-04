from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, make_ddr_config, make_hbm_config, make_ssd_config
from heterollm_sim.nand_core import NandCore


def test_physical_memory_smoke() -> None:
    dram = make_ddr_config(
        banks_per_group=1, rows_per_bank=4, row_bytes=128, burst_bytes=64,
        data_rate_mt_s=3200,
    )
    core = DramCore(dram)
    hit = core.execute(AccessRequest("hit", "read", 0, 128))
    assert hit.counters["row_misses"] == 1
    assert hit.counters["row_hits"] == 1
    conflict = core.execute(AccessRequest("conflict", "read", 128, 64))
    assert conflict.counters["row_conflicts"] == 1
    pipeline = DramCore(make_ddr_config(
        banks_per_group=1, rows_per_bank=4, row_bytes=256, burst_bytes=64,
        lane_bandwidth_gb_s=64, read_latency_ns=20, burst_interval_ns=2,
    )).execute(AccessRequest("pipeline", "read", 0, 256))
    read_starts = [s.start_ns for s in pipeline.stages if s.name == "READ_PIPELINE"]
    first_data_end = next(s.end_ns for s in pipeline.stages if s.name == "BURST_TRANSFER")
    assert len(read_starts) == 4 and read_starts[-1] < first_data_end

    hbm1 = make_hbm_config(channels=1, pseudo_channels_per_channel=1, stacks=1)
    hbm2 = make_hbm_config(channels=1, pseudo_channels_per_channel=1, stacks=2)
    assert hbm2.capacity_bytes == 2 * hbm1.capacity_bytes
    assert hbm2.bandwidth_gb_s == hbm1.bandwidth_gb_s

    nand = make_ssd_config(
        luns_per_die=2, planes_per_lun=1, blocks_per_plane=2,
        pages_per_block=4, page_bytes=1024, host_granularity_bytes=256,
        page_read_ns=10, page_program_ns=20, block_erase_ns=30,
    )
    ncore = NandCore(nand)
    read = ncore.execute(AccessRequest("pages", "read", 100, 1500))
    assert read.pages_read == 2
    assert read.host_transfer_bytes == 1792
    write = ncore.execute(AccessRequest("write", "write", 0, 1024))
    assert write.pages_programmed == 1
    assert write.pages_read == 0
    erase = ncore.execute(AccessRequest("erase", "erase", 0, nand.block_bytes))
    assert erase.erase_operations == 1
    assert erase.transfer_bytes == 0
