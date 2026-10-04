"""Correctness regressions for the physical-memory model.

Audited commit: b8e3eaad2200d65e8c3c7e80adab0c2436be784b.
These tests assert intended physical invariants, not the current buggy outputs.
Run from the project root with the project importable:
    python -m pytest -q path/to/test_memory_audit_regressions.py

No repository modifications, hardware measurements, or network access are made.
Synthetic parameters keep examples small; they are NOT real device profiles.
"""
from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore
from heterollm_sim.memory_types import AccessRequest, DramConfig, NandConfig
from heterollm_sim.memory_mapping import (
    map_dram_address, map_nand_address, split_nand_request,
)


def _dram_location(mapping):
    return tuple(getattr(mapping, name) for name in (
        "stack", "die", "lane", "rank", "bank_group", "bank", "row", "column",
    ))


def _nand_block(mapping):
    return tuple(getattr(mapping, name) for name in (
        "channel", "target", "die", "lun", "plane", "block",
    ))


def test_independent_planes_do_not_create_undeclared_channels():
    cfg = NandConfig(channels=1, planes_per_lun=2,
                     planes_independent=True, page_bytes=1024)
    mappings = [map_nand_address(cfg, index * cfg.page_bytes)
                for index in range(8)]
    assert all(0 <= m.channel < cfg.channels for m in mappings), mappings


def test_interleave_preserves_distinct_dram_burst_locations():
    cfg = DramConfig(channels=1, banks_per_group=1, rows_per_bank=4,
                     row_bytes=128, burst_bytes=64, interleave_bytes=128)
    a, b = (map_dram_address(cfg, address) for address in (0, 64))
    assert _dram_location(a) != _dram_location(b), (a, b)


def test_nand_single_page_buffer_is_not_overwritten_during_program():
    cfg = NandConfig(channels=1, page_bytes=1024,
                     host_granularity_bytes=1024,
                     host_bandwidth_gb_s=1024,
                     internal_bandwidth_gb_s=1024, page_program_ns=100)
    core = NandCore(cfg)
    first = core.execute(AccessRequest("first", "write", 0, 1024))
    second = core.execute(AccessRequest("second", "write", 1024, 1024))
    program = next(s for s in first.stages if s.name == "PAGE_PROGRAM")
    data_in = next(s for s in second.stages if s.name == "INTERNAL_TRANSFER")
    assert data_in.start_ns >= program.end_ns, (program, data_in)


def test_rmw_internal_traffic_equals_sum_of_internal_transfers():
    cfg = NandConfig(page_bytes=1024, host_granularity_bytes=256,
                     host_bandwidth_gb_s=1024,
                     internal_bandwidth_gb_s=1024,
                     page_read_ns=10, page_program_ns=100)
    result = NandCore(cfg).execute(AccessRequest("partial", "write", 0, 256))
    actual = sum(s.bytes for s in result.stages
                 if s.name in {"INTERNAL_TRANSFER", "RMW_INTERNAL_TRANSFER"})
    assert result.internal_transfer_bytes == actual, (result.counters, actual)
    assert result.transfer_bytes == actual


def test_erase_covers_physical_blocks_touched_by_its_address_range():
    # The byte range intersects block 0 in each of the two interleaved LUNs.
    cfg = NandConfig(channels=1, luns_per_die=2, planes_per_lun=1,
                     blocks_per_plane=2, pages_per_block=4, page_bytes=1024)
    touched = {_nand_block(map_nand_address(cfg, address))
               for address in range(0, cfg.block_bytes, cfg.page_bytes)}
    request = AccessRequest("erase", "erase", 0, cfg.block_bytes)
    erased = {_nand_block(s.mapping) for s in split_nand_request(request, cfg)}
    assert touched == erased, {"touched": touched, "erased": erased}


def test_dram_max_outstanding_survives_simultaneous_completions():
    cfg = DramConfig(channels=4, banks_per_group=1, rows_per_bank=8,
                     row_bytes=256, burst_bytes=64,
                     lane_bandwidth_gb_s=64, open_ns=10,
                     read_latency_ns=10, burst_interval_ns=1,
                     max_outstanding_requests=2)
    requests = [AccessRequest(str(i), "read", i * 64, 64) for i in range(4)]
    results = DramCore(cfg).execute_batch(requests)
    intervals = [(min(s.start_ns for s in r.stages), r.completion_ns)
                 for r in results]
    # End events sort before start events at the same timestamp.
    events = sorted([(start, 1) for start, _ in intervals]
                    + [(end, -1) for _, end in intervals])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak <= cfg.max_outstanding_requests, (intervals, peak)


def test_directional_rate_cannot_exceed_declared_physical_interface():
    # Accept either rejecting inconsistent input or enforcing its physical cap.
    try:
        cfg = DramConfig(interface_bandwidth_gb_s=1,
                         read_bandwidth_gb_s=100, open_ns=0,
                         read_latency_ns=0, burst_interval_ns=0)
    except ValueError:
        return
    result = DramCore(cfg).execute(AccessRequest("read", "read", 0, 6400))
    assert result.actual_bandwidth_gb_s <= cfg.interface_bandwidth_gb_s + 1e-9


def test_explicit_capacity_cannot_create_undeclared_dram_stacks():
    # A usable-capacity cap may be smaller than geometry, not larger.
    try:
        cfg = DramConfig(stacks=1, banks_per_group=1, rows_per_bank=1,
                         row_bytes=64, burst_bytes=64, capacity_bytes=128)
        mapping = map_dram_address(cfg, 64)
    except ValueError:
        return
    assert mapping.stack < cfg.stacks, mapping
