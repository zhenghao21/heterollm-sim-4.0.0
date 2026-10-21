"""Finite HOST DRAM, real PD write traffic, and exact chunk-event compression."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.paper_inference import PaperExecutionAssumptions, PaperNetwork
from heterollm_sim.paper_memory import PaperMemoryAssumptions, PaperMemorySystem
from heterollm_sim.paper_pd import MIB, PaperPDHostAssumptions, PaperPDImporter


def importer(name="HBF+", *, accelerate=True, trace=False, memory_trace=False,
             memory_duplex=True, host_duplex=True, shared_duplex=True,
             nic_scope="gpu", rank=0):
    hardware = hardware_by_id(name)
    assumptions = PaperMemoryAssumptions(
        sram_read_gb_s_per_stack=10000., sram_write_gb_s_per_stack=10000., sram_access_ns=1.,
        host_read_gb_s=hardware.hbm_read_gb_s+hardware.hbf_read_gb_s,
        host_write_gb_s=hardware.hbm_write_gb_s+hardware.hbf_read_gb_s,
        host_full_duplex=host_duplex, memory_full_duplex=memory_duplex,
        shared_io_gb_s=100000., shared_io_full_duplex=shared_duplex,
        hbm_access_ns=20., basis="Explicit fast ingress semantic test, not measured hardware")
    memory = PaperMemorySystem(hardware, assumptions=assumptions, retain_trace=memory_trace)
    network = PaperNetwork(PaperExecutionAssumptions(nic_scope=nic_scope))
    return PaperPDImporter(network, memory, rank, accelerate=accelerate, retain_trace=trace)


def close_nested(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            close_nested(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            close_nested(left, right)
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected, rel=1e-11, abs=1e-6)
    else:
        assert actual == expected


def observer_summary(instance):
    summary = instance.summary()
    summary.pop("compressed_chunks")
    summary.pop("explicit_chunks")
    return dict(host=summary, memory=instance.memory.summary(),
        network=dict(calendars=instance.network.calendars, busy=instance.network.busy,
                     bytes=instance.network.bytes, operations=instance.network.operations))


def test_default_host_dram_is_two_finite_slots_and_not_hidden_hbm():
    instance = importer(trace=True, memory_trace=True)
    chunk = instance.chunk_bytes("hbf")
    result = instance.import_kv(chunk*4+1)
    summary = instance.summary()
    assert summary["host_buffer_capacity_bytes"] == 64*MIB
    assert summary["host_buffer_slot_bytes"] == 32*MIB
    assert result.host_buffer_peak_bytes == 2*chunk <= 64*MIB
    assert summary["hidden_HBM_staging_bytes"] == 0
    assert not any(key.startswith("hbm:") for key in instance.memory.bytes_by_operation)
    assert result.logical_bytes == result.transfer_bytes == chunk*4+1
    assert result.counters["host_write_bytes"] == result.counters["host_read_bytes"] == result.logical_bytes
    assert result.chunk_count == result.counters["network_operations"] == 5
    assert result.counters["host_write_operations"] == result.counters["host_read_operations"] == 5
    assert max(item.logical_bytes for item in instance.trace) <= 32*MIB


def test_chunks_follow_receive_host_write_host_read_then_program_before_reuse():
    instance = importer(trace=True, memory_trace=True)
    chunk = instance.chunk_bytes("hbf")
    result = instance.import_kv(3*chunk+17, now_ns=100.)
    assert result.explicit_chunks == 4
    assert [row.slot for row in instance.trace] == [0, 1, 0, 1]
    previous = {}
    for record in instance.trace:
        assert record.network_end_ns >= record.network_start_ns >= 100
        assert record.host_write_start_ns >= record.network_end_ns
        assert record.host_read_start_ns >= record.host_write_end_ns
        assert record.end_ns > record.host_read_end_ns
        if record.slot in previous:
            assert record.network_start_ns >= previous[record.slot].end_ns
        previous[record.slot] = record
    # Receiving the next chunk overlaps the previous chunk's device write.
    assert instance.trace[1].network_start_ns < instance.trace[0].end_ns
    events = [event for event in instance.memory.events if event.operation == "write"]
    assert sum(event.size_bytes for event in events) == result.physical_program_bytes
    assert all(event.tier == "hbf" and event.size_bytes % 4096 == 0 for event in events)


@pytest.mark.parametrize("size,pages", [(1, 1), (128, 1), (4096, 1), (4097, 2), (8191, 2)])
def test_tiny_tail_packs_complete_pages_before_stack_interleave(size, pages):
    instance = importer(memory_trace=True)
    result = instance.import_kv(size)
    writes = [event for event in instance.memory.events if event.operation == "write"]
    assert result.physical_program_bytes == pages*4096
    assert sum(event.size_bytes for event in writes) == pages*4096
    assert [event.stack for event in writes] == list(range(pages))
    assert len(writes) == pages  # Never invent eight padded pages for one tiny KV.
    assert result.transfer_bytes == result.logical_bytes == size


@pytest.mark.parametrize("name", ["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
@pytest.mark.parametrize("duplex", [True, False])
def test_max_plus_middle_matches_real_chunk_time_bytes_counts_and_slot_state(name, duplex):
    tier = "hbm" if name == "HBM4" else "hbf"
    fast = importer(name, accelerate=True, memory_duplex=duplex,
                    host_duplex=duplex, shared_duplex=duplex)
    slow = importer(name, accelerate=False, memory_duplex=duplex,
                    host_duplex=duplex, shared_duplex=duplex)
    chunk = fast.chunk_bytes(tier)
    # Odd full chunk count and partial tail preserve nontrivial slot rotation.
    result_fast = fast.import_kv(chunk*43+5003, tier=tier, now_ns=1234.)
    result_slow = slow.import_kv(chunk*43+5003, tier=tier, now_ns=1234.)
    assert result_fast.compressed_chunks >= 32
    assert result_fast.explicit_chunks <= 6
    assert result_slow.compressed_chunks == 0
    assert result_fast.end_ns == pytest.approx(result_slow.end_ns, rel=1e-11)
    assert result_fast.physical_program_bytes == result_slow.physical_program_bytes
    assert result_fast.transfer_bytes == result_slow.transfer_bytes
    assert result_fast.chunk_count == result_slow.chunk_count
    close_nested(result_fast.counters, result_slow.counters)
    close_nested(observer_summary(fast), observer_summary(slow))


def test_import_reuses_two_slots_across_requests_and_respects_existing_shared_work():
    fast = importer(accelerate=True, memory_duplex=False, host_duplex=False,
                    shared_duplex=False, nic_scope="node", rank=7)
    slow = importer(accelerate=False, memory_duplex=False, host_duplex=False,
                    shared_duplex=False, nic_scope="node", rank=7)
    chunk = fast.chunk_bytes("hbf")
    for instance in (fast, slow):
        instance.import_kv(chunk+123, now_ns=0.)
        # Another GPU on this node owns the same incoming NIC. Array reads
        # compete with half-duplex writes; GPU compute remains independent.
        instance.network.receive_pd(0, 100_000_000, 9e6)
        instance.memory.access("hbf", "read", 4096, stack=0, ready_ns=8e6)
        instance.memory.compute(1000., ready_ns=3e6)
    first = fast.import_kv(chunk*32+13, now_ns=50.)
    second = slow.import_kv(chunk*32+13, now_ns=50.)
    assert first.compressed_chunks > 0
    assert first.end_ns == pytest.approx(second.end_ns, rel=1e-11)
    close_nested(observer_summary(fast), observer_summary(slow))


def test_retained_trace_requires_every_real_chunk_instead_of_omitted_events():
    for trace, memory_trace in ((True, False), (False, True)):
        instance = importer(trace=trace, memory_trace=memory_trace)
        result = instance.import_kv(instance.chunk_bytes("hbf")*20)
        assert result.compressed_chunks == 0
        assert result.explicit_chunks == result.chunk_count == 20


def test_hbm_destination_uses_host_slots_but_no_nand_programs():
    instance = importer("HBM4")
    result = instance.import_kv(40*instance.chunk_bytes("hbm")+1, tier="hbm")
    assert result.compressed_chunks > 0
    assert result.physical_program_bytes == 0
    assert result.counters["memory_write_bytes"] == result.logical_bytes
    assert "hbf:write" not in instance.memory.bytes_by_operation


def test_hundreds_of_gigabytes_do_not_expand_to_per_chunk_or_page_work():
    instance = importer()
    result = instance.import_kv(500_000_000_123)
    assert result.chunk_count > 30_000
    assert result.explicit_chunks <= 2
    assert result.compressed_chunks + result.explicit_chunks == result.chunk_count
    assert result.physical_program_bytes == (result.logical_bytes+4095)//4096*4096
    assert result.counters["network_operations"] == result.chunk_count
    assert result.host_buffer_peak_bytes <= 64*MIB


@pytest.mark.parametrize("name", ["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
@pytest.mark.parametrize("duplex", [True, False])
@pytest.mark.parametrize("full_chunks", [1, 2, 3, 4, 5])
def test_first_full_pair_algebra_handles_short_imports_without_warmup(name, duplex, full_chunks):
    tier = "hbm" if name == "HBM4" else "hbf"
    fast = importer(name, accelerate=True, memory_duplex=duplex,
                    host_duplex=duplex, shared_duplex=duplex)
    slow = importer(name, accelerate=False, memory_duplex=duplex,
                    host_duplex=duplex, shared_duplex=duplex)
    chunk = fast.chunk_bytes(tier)
    logical = full_chunks*chunk + 5003
    a = fast.import_kv(logical, tier=tier)
    b = slow.import_kv(logical, tier=tier)
    assert a.compressed_chunks == (full_chunks//2)*2
    assert a.explicit_chunks == full_chunks % 2 + 1
    assert a.chunk_count == b.chunk_count
    assert a.logical_bytes == b.logical_bytes == logical
    assert a.physical_program_bytes == b.physical_program_bytes
    assert a.end_ns == pytest.approx(b.end_ns, rel=1e-11)
    close_nested(a.counters, b.counters)
    close_nested(observer_summary(fast), observer_summary(slow))


@pytest.mark.parametrize("duplex", [True, False])
def test_first_pair_preserves_large_old_slot_overlap_with_smaller_new_chunks(duplex):
    fast = importer("HBF", accelerate=True, memory_duplex=duplex,
                    host_duplex=duplex, shared_duplex=duplex)
    slow = importer("HBF", accelerate=False, memory_duplex=duplex,
                    host_duplex=duplex, shared_duplex=duplex)
    # A legitimate prior import leaves one 32 MiB HBM-directed HOST slot
    # occupied while the first, smaller HBF chunk starts in the other slot.
    for instance in (fast, slow):
        instance.import_kv(32*MIB, tier="hbm")
    chunk = fast.chunk_bytes("hbf")
    a = fast.import_kv(2*chunk, tier="hbf")
    b = slow.import_kv(2*chunk, tier="hbf")
    assert a.compressed_chunks == 2
    assert a.explicit_chunks == 0
    assert a.host_buffer_peak_bytes == b.host_buffer_peak_bytes == 32*MIB+chunk
    assert a.host_buffer_peak_bytes > 2*chunk
    close_nested(observer_summary(fast), observer_summary(slow))


@pytest.mark.parametrize("full_chunks", [2, 3, 4, 5])
def test_first_pair_handles_future_nic_and_nonempty_slot_frontiers(full_chunks):
    fast = importer("HBF+", accelerate=True, nic_scope="node", rank=7)
    slow = importer("HBF+", accelerate=False, nic_scope="node", rank=7)
    chunk = fast.chunk_bytes("hbf")
    for instance in (fast, slow):
        instance.import_kv(chunk+37, now_ns=0.)
        instance.network.receive_pd(0, 999_999_999, 9e6)
        instance.memory.access("hbf", "write", 4096, stack=0, ready_ns=99e6)
    a = fast.import_kv(full_chunks*chunk+117, now_ns=3.)
    b = slow.import_kv(full_chunks*chunk+117, now_ns=3.)
    assert a.compressed_chunks == (full_chunks//2)*2
    close_nested(a.counters, b.counters)
    close_nested(observer_summary(fast), observer_summary(slow))


@pytest.mark.parametrize("changes", [{"buffer_bytes": 0}, {"buffer_bytes": 3},
    {"read_gb_s": 0}, {"write_gb_s": float("inf")}, {"basis": ""}])
def test_host_research_parameters_are_explicit_and_valid(changes):
    with pytest.raises(ValueError):
        replace(PaperPDHostAssumptions(), **changes)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_import_rejects_invalid_logical_bytes(value):
    with pytest.raises(ValueError):
        importer().import_kv(value)


def test_zero_bandwidth_or_nonexistent_tier_cannot_create_free_import():
    with pytest.raises(ValueError):
        importer("HBM4").import_kv(100, tier="hbf")
    with pytest.raises(ValueError):
        importer().import_kv(100, now_ns=float("nan"))
