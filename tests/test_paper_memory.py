"""Executable capacity, dependency and resource tests for paper SRAM control."""
from dataclasses import asdict
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.paper_memory import (
    PaperBufferCreditError, PaperMemoryAssumptions, PaperMemoryCapacityError,
    PaperMemorySystem, _STREAM_TEMPLATES,
)


def assumptions(**changes):
    values = dict(sram_read_gb_s_per_stack=10000.0, sram_write_gb_s_per_stack=10000.0,
                  sram_access_ns=0.0, host_read_gb_s=12800.0, host_write_gb_s=12800.0,
                  host_full_duplex=True, memory_full_duplex=True, shared_io_gb_s=12800.0,
                  shared_io_full_duplex=True, hbm_access_ns=0.0,
                  basis="Explicit ideal fast-SRAM and aggregate link test profile; not paper measurements")
    values.update(changes)
    return PaperMemoryAssumptions(**values)


def system(name="HBF+", **changes):
    return PaperMemorySystem(hardware_by_id(name), assumptions=assumptions(**changes), retain_trace=True)


@pytest.mark.parametrize("name", ["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
def test_exact_stack_presence_and_no_hidden_hbm(name):
    memory = system(name)
    hardware = hardware_by_id(name)
    assert memory.hbm_stacks == hardware.hbm_stacks
    assert memory.hbf_stacks == hardware.hbf_stacks
    summary = memory.summary()
    assert summary["intermediate_capacity_bytes_per_stack"] == hardware.intermediate_sram_bytes_per_stack
    assert summary["staging_capacity_bytes_per_stack"] == hardware.read_staging_bytes_per_hbf_stack
    memory.stream_read_compute(10_000_000, 1000.0)
    assert ("hbm:read" in memory.bytes_by_operation) == (name == "HBM4")
    assert ("hbf:read" in memory.bytes_by_operation) == (name != "HBM4")
    assert "not_OCP" in summary["abstraction"]


@pytest.mark.parametrize("field,value", [("sram_read_gb_s_per_stack", 0),
    ("host_read_gb_s", float("nan")), ("host_write_gb_s", -1),
    ("shared_io_gb_s", float("inf")), ("sram_access_ns", -1),
    ("hbm_access_ns", True), ("basis", ""), ("host_full_duplex", 1),
    ("memory_full_duplex", "yes"), ("shared_io_full_duplex", None)])
def test_undisclosed_assumptions_must_be_valid_and_explicit(field, value):
    with pytest.raises(ValueError):
        assumptions(**{field: value})


def test_missing_assumptions_and_zero_sram_plus_are_rejected():
    with pytest.raises(TypeError):
        PaperMemorySystem(hardware_by_id("HBF+"), assumptions=None)
    hardware = asdict(hardware_by_id("HBF+"))
    hardware["intermediate_sram_bytes_per_stack"] = 0
    with pytest.raises(ValueError, match="40 MB"):
        PaperMemorySystem(hardware, assumptions=assumptions())


def test_per_stack_intermediate_capacity_is_not_a_fungible_320mb_pool():
    memory = system()
    first = memory.allocate_intermediate("activation-a", 30_000_000, stack=0)
    memory.allocate_intermediate("activation-b", 40_000_000, stack=1)
    with pytest.raises(PaperMemoryCapacityError):
        memory.allocate_intermediate("would-overflow-stack-zero", 10_000_001, stack=0)
    memory.release_intermediate(first, at_ns=100)
    with pytest.raises(PaperMemoryCapacityError):
        memory.allocate_intermediate("too-early", 20_000_000, stack=0, ready_ns=99)
    memory.allocate_intermediate("reused-after-release", 40_000_000, stack=0, ready_ns=100)
    assert memory.summary()["intermediate_peak_bytes_per_stack"][:2] == [40_000_000, 40_000_000]


def test_intermediate_lifetimes_reject_time_travel_double_free_and_absent_sram():
    memory = system()
    lease = memory.allocate_intermediate("a", 100, stack=0, ready_ns=50)
    with pytest.raises(ValueError, match="before allocation"):
        memory.release_intermediate(lease, at_ns=49)
    with pytest.raises(ValueError, match="nondecreasing"):
        memory.allocate_intermediate("b", 100, stack=0, ready_ns=49)
    memory.release_intermediate(lease, at_ns=60)
    with pytest.raises(ValueError, match="already released"):
        memory.release_intermediate(lease, at_ns=60)
    with pytest.raises(ValueError, match="absent"):
        system("HBF").allocate_intermediate("a", 100, stack=0)


def test_staging_is_separate_from_intermediate_capacity():
    memory = system()
    memory.allocate_intermediate("full-intermediate", 40_000_000, stack=0)
    memory.prefetch(0, memory.staging_slot_bytes)
    memory.prefetch(0, memory.staging_slot_bytes)
    assert memory.summary()["staging_peak_bytes_per_stack"][0] == 3_130_000
    with pytest.raises(PaperBufferCreditError):
        memory.prefetch(0, 1)
    with pytest.raises(PaperMemoryCapacityError):
        system().prefetch(0, 1_565_001)


def test_double_buffer_cannot_overwrite_until_gpu_consumption_finishes():
    memory = system()
    first = memory.prefetch(0, 1000)
    second = memory.prefetch(0, 1000)
    consume = memory.consume(first, 100_000, ready_ns=5000)
    replacement = memory.prefetch(0, 1000)
    assert consume.start_ns >= first.read.end_ns
    assert consume.end_ns == first.released_at_ns
    assert replacement.read.ready_ns >= consume.end_ns
    assert replacement.slot == first.slot
    assert second.slot != replacement.slot
    with pytest.raises(ValueError):
        memory.consume(first, 1)


def test_gpu_waits_for_all_stack_reads_and_compute_is_not_sram_compute():
    memory = system()
    first = memory.prefetch(0, 1000)
    second = memory.prefetch(1, 1000, ready_ns=1_000_000)
    consume = memory.consume_many([first, second], 100)
    assert consume.start_ns >= second.read.end_ns
    assert consume.resources == ("gpu:compute",)
    assert first.released_at_ns == second.released_at_ns == consume.end_ns
    assert all(event.operation != "compute" for event in memory.events if event.tier != "gpu")


def test_cancelled_read_does_not_refund_already_reserved_io_or_overwrite_in_flight():
    memory = system()
    ticket = memory.prefetch(0, 1000)
    end = memory.cancel_prefetch(ticket)
    assert end == ticket.read.end_ns
    again = memory.prefetch(0, 1000)
    assert again.read.ready_ns >= end
    assert memory.summary()["bytes_by_operation"]["hbf:read"] == 2000
    assert memory.summary()["cancelled_reads"] == 1
    with pytest.raises(ValueError):
        memory.consume(ticket, 10)


@pytest.mark.parametrize("change", [dict(host_full_duplex=False), dict(shared_io_full_duplex=False)])
def test_shared_read_write_bus_contention_is_paid(change):
    memory = system("HBM4", host_read_gb_s=1, host_write_gb_s=1,
                    shared_io_gb_s=1, **change)
    read = memory.access("hbm", "read", 100, stack=0)
    write = memory.access("hbm", "write", 100, stack=1)
    assert read.end_ns == 100
    assert write.start_ns == 100
    assert write.end_ns == 200


def test_full_duplex_can_overlap_and_medium_half_duplex_can_serialize():
    memory = system("HBM4", host_read_gb_s=1, host_write_gb_s=1, shared_io_gb_s=1)
    assert memory.access("hbm", "read", 100, stack=0).end_ns == 100
    assert memory.access("hbm", "write", 100, stack=0).end_ns == 100
    serialized = system("HBM4", memory_full_duplex=False,
                        host_read_gb_s=1, host_write_gb_s=1, shared_io_gb_s=1)
    serialized.access("hbm", "read", 100, stack=0)
    assert serialized.access("hbm", "write", 100, stack=0).start_ns > 0


def test_stack_parallelism_is_not_destroyed_by_shared_bus_ownership():
    memory = system("HBM4")
    events = [memory.access("hbm", "read", 1_600_000, stack=stack) for stack in range(8)]
    # Stack service is 1000 ns, shared bus byte service is only 125 ns.
    assert events[0].end_ns == 1000
    assert events[-1].start_ns == 875
    assert events[-1].end_ns == 1875
    assert memory.summary()["resources"]["host:read"]["busy_ns"] == 1000


def test_real_read_and_write_startup_asymmetry_is_preserved():
    hbf = system("HBF")
    conv = system("CONV")
    hbf_read = hbf.access("hbf", "read", 1600, stack=0)
    conv_read = conv.access("hbf", "read", 350, stack=0)
    assert hbf_read.end_ns == 1001
    assert conv_read.end_ns == 3001
    write = hbf.access("hbf", "write", 16, stack=0)
    assert write.end_ns - write.start_ns == pytest.approx(100000 + 16 / 12800)
    assert hbf.summary()["resources"]["hbf:0:write"]["busy_ns"] == 100000


@pytest.mark.parametrize("name", ["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
def test_streaming_executes_exact_bytes_with_bounded_credits_and_gpu_work(name):
    memory = system(name)
    size = 51_234_567
    result = memory.stream_read_compute(size, 123_456, label="one-decoder-op")
    summary = memory.summary()
    tier = "hbm" if name == "HBM4" else "hbf"
    assert summary["bytes_by_operation"][tier + ":read"] == size
    assert summary["resources"]["gpu:compute"]["busy_ns"] == pytest.approx(123_456)
    assert result.end_ns > result.start_ns
    assert all(value <= 3_130_000 for value in summary["staging_peak_bytes_per_stack"])
    assert not memory._tickets
    assert summary["events"] < 200  # tile events, never one event per 4-KiB page


def test_faster_sram_and_prefetch_dependencies_change_elapsed_execution():
    slow = system(sram_read_gb_s_per_stack=1, sram_write_gb_s_per_stack=1)
    fast = system()
    assert slow.stream_read_compute(10_000_000, 10000).end_ns > fast.stream_read_compute(10_000_000, 10000).end_ns


@pytest.mark.parametrize("method,args", [("prefetch", (0, 0)), ("prefetch", (0, True)),
    ("prefetch", (8, 100)), ("stream_read_compute", (0, 100)),
    ("stream_read_compute", (100, -1))])
def test_invalid_operations_are_rejected(method, args):
    with pytest.raises(ValueError):
        getattr(system(), method)(*args)


def _equivalent_memory_state(slow, fast, *, float_relative=2e-12):
    original, accelerated = slow.summary(), fast.summary()
    assert accelerated.keys() == original.keys()
    for key in original:
        if key in {"end_ns", "resources"}:
            continue
        assert accelerated[key] == original[key], key
    assert accelerated["end_ns"] == pytest.approx(original["end_ns"], rel=float_relative, abs=1e-6)
    assert accelerated["resources"].keys() == original["resources"].keys()
    for key, values in original["resources"].items():
        assert accelerated["resources"][key]["operations"] == values["operations"]
        for field in ("available_ns", "busy_ns"):
            assert accelerated["resources"][key][field] == pytest.approx(values[field], rel=float_relative, abs=1e-6), (key, field)
    assert slow._next_ticket == fast._next_ticket
    assert slow._tickets == fast._tickets == {}
    for left, right in zip(slow._slots, fast._slots):
        for (owner_a, time_a), (owner_b, time_b) in zip(left, right):
            assert owner_a == owner_b
            assert time_b == pytest.approx(time_a, rel=float_relative, abs=1e-6)


@pytest.mark.parametrize("name", ["HBF", "HBF+", "CONV", "CONV+"])
@pytest.mark.parametrize("profile", [dict(), dict(memory_full_duplex=False),
    dict(host_full_duplex=False, shared_io_full_duplex=False),
    dict(host_read_gb_s=50, shared_io_gb_s=100),
    dict(shared_io_gb_s=50, sram_read_gb_s_per_stack=33, sram_write_gb_s_per_stack=77)])
def test_bulk_max_plus_matches_each_finite_buffer_wave_and_calendar(name, profile):
    rng = random.Random(107)
    for _ in range(5):
        slow = PaperMemorySystem(hardware_by_id(name), assumptions=assumptions(**profile),
                                  bulk_wave_acceleration=False)
        fast = PaperMemorySystem(hardware_by_id(name), assumptions=assumptions(**profile))
        # Existing calendars can include asymmetric read/write traffic and
        # slot reuse; powers must not assume a pristine or saturated device.
        for memory in (slow, fast):
            ticket = memory.prefetch(0, 1337)
            memory.consume(ticket, 500)
            memory.access("hbf", "write", 3003, stack=1, ready_ns=500)
        wave_bytes = slow.hbf_stacks * slow.staging_slot_bytes
        total_bytes = rng.randint(35, 110) * wave_bytes + rng.randint(0, wave_bytes - 1)
        compute = rng.choice([0, 3, 100_000, 10_000_000]) + rng.random()
        ready = rng.random() * 200_000
        before = slow.stream_read_compute(total_bytes, compute, ready_ns=ready)
        after = fast.stream_read_compute(total_bytes, compute, ready_ns=ready)
        assert fast._bulk_advanced_waves > 0
        assert after.end_ns == pytest.approx(before.end_ns, rel=2e-12, abs=1e-6)
        _equivalent_memory_state(slow, fast)


def test_bulk_preserves_slot_parity_and_later_calls_not_just_one_request():
    slow = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions(), bulk_wave_acceleration=False)
    fast = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    wave_bytes = slow.hbf_stacks * slow.staging_slot_bytes
    for waves in (35, 36, 101, 34):
        total = waves * wave_bytes + (waves % 2) * 51
        slow.stream_read_compute(total, 1_234_567.89, ready_ns=123)
        fast.stream_read_compute(total, 1_234_567.89, ready_ns=123)
        _equivalent_memory_state(slow, fast)
    for memory in (slow, fast):
        ticket = memory.prefetch(0, 10_000)
        memory.cancel_prefetch(ticket)
        ticket = memory.prefetch(1, 1000)
        memory.consume(ticket, 100)
    _equivalent_memory_state(slow, fast)


def test_retained_trace_and_nonhomogeneous_small_requests_use_real_wave_path():
    traced = system()
    wave_bytes = traced.hbf_stacks * traced.staging_slot_bytes
    traced.stream_read_compute(40 * wave_bytes, 100)
    assert traced._bulk_advanced_waves == 0
    assert len(traced.events) == traced.event_count
    small = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    small.stream_read_compute(33 * wave_bytes, 100)
    assert small._bulk_advanced_waves == 0


def test_ten_thousand_full_waves_preserve_all_resources_and_integer_observers():
    profile = assumptions(memory_full_duplex=False, host_full_duplex=False,
                          shared_io_full_duplex=False, host_read_gb_s=100,
                          shared_io_gb_s=150)
    slow = PaperMemorySystem(hardware_by_id("CONV"), assumptions=profile,
                             bulk_wave_acceleration=False)
    fast = PaperMemorySystem(hardware_by_id("CONV"), assumptions=profile)
    total = 10_000 * slow.hbf_stacks * slow._read_slot_bytes
    slow.stream_read_compute(total, 123_456_789.123)
    fast.stream_read_compute(total, 123_456_789.123)
    assert fast._bulk_advanced_waves == 9998
    assert fast.event_count == 220_000
    # Powers regroup floating-point additions.  Repeated scalar additions
    # have their own cumulative rounding; 1e-10 is a roundoff-only tolerance
    # for this long accumulation, not the experiment's 10% error constraint.
    _equivalent_memory_state(slow, fast, float_relative=1e-10)


def _quiescent_layer(memory, start, *, wave_count=2):
    lease = memory.allocate_intermediate("short-lived-activation", 10000, stack=0, ready_ns=start)
    write = memory.access("sram", "write", 10000, stack=0, ready_ns=start)
    memory.stream_read_compute(wave_count * memory.hbf_stacks * memory.staging_slot_bytes,
                               123456.789, ready_ns=write.end_ns)
    end = memory.summary()["end_ns"]
    memory.release_intermediate(lease, at_ns=end)
    return end


def test_quiescent_period_exactly_repeats_used_resources_counts_and_lifetimes():
    profile = assumptions()
    slow = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    fast = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    # This old, unused stack-write resource must retain its actual frontier.
    for memory in (slow, fast):
        memory.access("hbf", "write", 1000, stack=7)
    start = fast.summary()["end_ns"]
    signature = ("two-wave-layer", 10000, 123456.789)
    before = fast.snapshot_counters(signature=signature)
    first_end = _quiescent_layer(fast, start)
    assert fast.repeat_quiescent_period(before, start, first_end, 20, signature=signature)
    end = start
    for _ in range(21):
        end = _quiescent_layer(slow, end)
    _equivalent_memory_state(slow, fast)
    assert fast.summary()["resources"]["hbf:7:write"]["available_ns"] == start
    assert fast._slot_releases == slow._slot_releases
    assert not fast._leases and not fast._release_heap
    assert fast._intermediate_live == [0] * 8
    assert fast._allocation_frontier_ns == pytest.approx(slow._allocation_frontier_ns)


@pytest.mark.parametrize("case", ["future_at_begin", "future_at_end", "lease_not_released",
                                  "pending_read", "signature", "trace", "events_before_start"])
def test_quiescent_period_refuses_unproven_cases_without_changing_results(case):
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions(), retain_trace=case == "trace")
    signature = ("fixed", 100)
    if case == "future_at_begin":
        memory.access("hbf", "write", 1000, stack=0, ready_ns=1_000_000)
    before = memory.snapshot_counters(signature=signature)
    start = 100 if case == "events_before_start" else 0
    if case == "future_at_end":
        memory.access("hbf", "write", 1000, stack=0, ready_ns=1_000_000)
    elif case == "lease_not_released":
        memory.allocate_intermediate("still-live", 100, stack=0)
    elif case == "pending_read":
        memory.prefetch(0, 100)
    memory.compute(100, ready_ns=0)
    state = memory.summary()
    assert not memory.repeat_quiescent_period(before, start, 200, 5,
                                               signature=("different",) if case == "signature" else signature)
    assert memory.summary() == state


def test_quiescent_period_rejects_missing_foreign_or_stale_signature_snapshot():
    a = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions())
    b = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions())
    signature = ("same-gpu-work",)
    before = a.snapshot_counters(signature=signature)
    a.compute(100)
    b.compute(100)
    assert not b.repeat_quiescent_period(before, 0, 100, 5, signature=signature)
    assert not a.repeat_quiescent_period(before, 0, 100, 5)
    a.snapshot_counters(signature=signature)
    assert not a.repeat_quiescent_period(before, 0, 100, 5, signature=signature)
    with pytest.raises(ValueError, match="immutable"):
        a.snapshot_counters(signature={"mutable": 1})


def test_odd_buffer_permutation_is_not_silently_reset_or_repeated():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("one-wave-layer",)
    before = memory.snapshot_counters(signature=signature)
    end = _quiescent_layer(memory, 0, wave_count=1)
    state = memory.summary()
    assert not memory.repeat_quiescent_period(before, 0, end, 5, signature=signature)
    assert memory.summary() == state


def test_hbm_idle_compute_period_can_repeat_without_hbf_buffers():
    memory = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions())
    signature = ("kernel", 125)
    before = memory.snapshot_counters(signature=signature)
    memory.compute(125, ready_ns=500)
    assert memory.repeat_quiescent_period(before, 500, 625, 125, signature=signature)
    assert memory.summary()["resources"]["gpu:compute"] == {
        "available_ns": 500 + 126 * 125, "busy_ns": 126 * 125, "operations": 126}


def test_hbm_overlap_requires_explicit_capacity_and_never_gains_hbf_resources():
    serial = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions())
    with pytest.raises(ValueError, match="explicit finite HBM"):
        serial.prefetch(0, 1000)
    profile = assumptions(hbm_read_buffer_bytes_per_stack=2 * 1024 * 1024)
    pipelined = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=profile, retain_trace=True)
    first = pipelined.prefetch(0, 1024 * 1024)
    second = pipelined.prefetch(0, 1024 * 1024)
    with pytest.raises(PaperBufferCreditError):
        pipelined.prefetch(0, 1)
    consumed = pipelined.consume(first, 10000)
    replaced = pipelined.prefetch(0, 1000)
    assert replaced.read.ready_ns >= consumed.end_ns
    assert replaced.read.tier == "hbm_buffer"
    summary = pipelined.summary()
    assert summary["hbm_read_buffer_capacity_bytes_per_stack"] == 2 * 1024 * 1024
    assert summary["hbm_read_buffer_peak_bytes_per_stack"][0] == 2 * 1024 * 1024
    assert summary["staging_capacity_bytes_per_stack"] == summary["intermediate_capacity_bytes_per_stack"] == 0
    assert summary["staging_peak_bytes_per_stack"] == []
    assert not any(key.startswith(("staging:", "hbf:", "sram:")) for key in summary["resources"])
    assert summary["research_assumptions"]["hbm_read_buffer_bytes_per_stack"] == 2 * 1024 * 1024


@pytest.mark.parametrize("profile", [dict(), dict(memory_full_duplex=False),
    dict(host_full_duplex=False, shared_io_full_duplex=False, host_read_gb_s=100),
    dict(shared_io_gb_s=125, sram_read_gb_s_per_stack=50, sram_write_gb_s_per_stack=33)])
def test_hbm_paid_double_buffer_and_max_plus_have_the_same_calendar_semantics(profile):
    configured = assumptions(hbm_read_buffer_bytes_per_stack=2 * 1024 * 1024, **profile)
    slow = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=configured,
                             bulk_wave_acceleration=False)
    fast = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=configured)
    total = 71 * 8 * 1024 * 1024 + 731
    before = slow.stream_read_compute(total, 123456.789, ready_ns=100)
    after = fast.stream_read_compute(total, 123456.789, ready_ns=100)
    assert after.end_ns == pytest.approx(before.end_ns)
    assert fast._bulk_advanced_waves == 69
    _equivalent_memory_state(slow, fast)
    assert fast._slot_releases == slow._slot_releases


def test_hbm_pipeline_overlap_is_real_and_buffer_port_bandwidth_remains_paid():
    profile = assumptions(hbm_read_buffer_bytes_per_stack=2 * 1024 * 1024)
    pipelined = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=profile)
    serial = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions())
    size = 40 * 8 * 1024 * 1024
    compute = 100000
    overlap = pipelined.stream_read_compute(size, compute)
    no_overlap = serial.stream_read_compute(size, compute)
    assert overlap.end_ns < no_overlap.end_ns
    assert pipelined.summary()["bytes_by_operation"]["hbm_buffer:hbm_buffer_fill"] == size
    assert pipelined.summary()["bytes_by_operation"]["hbm_buffer:hbm_buffer_read"] == size
    slow_ports = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions(
        hbm_read_buffer_bytes_per_stack=2 * 1024 * 1024,
        sram_read_gb_s_per_stack=1, sram_write_gb_s_per_stack=1))
    assert slow_ports.stream_read_compute(size, compute).end_ns > overlap.end_ns


def test_hbm_buffer_period_repetition_preserves_finite_slot_state():
    profile = assumptions(hbm_read_buffer_bytes_per_stack=2 * 1024 * 1024)
    slow = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=profile)
    fast = PaperMemorySystem(hardware_by_id("HBM4"), assumptions=profile)
    signature = ("hbm-gpu-buffer-layer", 2 * 8 * 1024 * 1024, 1200)
    before = fast.snapshot_counters(signature=signature)
    end = fast.stream_read_compute(signature[1], signature[2]).end_ns
    assert fast.repeat_quiescent_period(before, 0, end, 125, signature=signature)
    now = 0
    for _ in range(126):
        now = slow.stream_read_compute(signature[1], signature[2], ready_ns=now).end_ns
    _equivalent_memory_state(slow, fast)
    assert slow._slot_releases == fast._slot_releases


@pytest.mark.parametrize("capacity", [-1, True, 1, 3, 2.0])
def test_hbm_read_buffer_capacity_must_define_two_finite_equal_slots(capacity):
    with pytest.raises(ValueError):
        assumptions(hbm_read_buffer_bytes_per_stack=capacity)


def test_period_qualification_is_side_effect_free_and_supports_multi_gpu_preflight():
    memories = [PaperMemorySystem(hardware_by_id("HBM4"), assumptions=assumptions()) for _ in range(2)]
    signature = ("rank-work", 100)
    snapshots = [memory.snapshot_counters(signature=signature) for memory in memories]
    for rank, memory in enumerate(memories):
        memory.compute(100 * (rank + 1))
    summaries = [memory.summary() for memory in memories]
    serials = [memory._snapshot_serial for memory in memories]
    for _ in range(3):
        assert all(memory.can_repeat_quiescent_period(before, 0, 200, signature=signature)
                   for memory, before in zip(memories, snapshots))
        assert [memory.summary() for memory in memories] == summaries
        assert [memory._snapshot_serial for memory in memories] == serials
    assert not memories[0].can_repeat_quiescent_period(snapshots[1], 0, 200, signature=signature)
    for memory, before in zip(memories, snapshots):
        assert memory.repeat_quiescent_period(before, 0, 200, 4, signature=signature)
    assert memories[0].summary()["resources"]["gpu:compute"]["available_ns"] == 900
    assert memories[1].summary()["resources"]["gpu:compute"]["available_ns"] == 1000


def test_failed_period_preflight_does_not_release_leases_or_future_pd():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("fixed-work",)
    before = memory.snapshot_counters(signature=signature)
    lease = memory.allocate_intermediate("not-retired", 100, stack=0)
    memory.access("hbf", "write", 4096, stack=0, ready_ns=1_000_000)
    state = memory.summary()
    assert not memory.can_repeat_quiescent_period(before, 0, 100, signature=signature)
    assert memory.summary() == state
    assert memory._leases[lease.allocation_id] is lease
    assert lease.released_at_ns is None


def test_nand_bulk_service_byte_count_is_explicitly_not_implicit_page_padding():
    memory = system("HBF")
    write = memory.access("hbf", "write", 17, stack=0)
    assert write.size_bytes == 17
    assert memory.summary()["bytes_by_operation"]["hbf:write"] == 17
    assert "no_implicit_page_rounding" in memory.summary()["transfer_byte_semantics"]


@pytest.mark.parametrize("name", ["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
@pytest.mark.parametrize("waves", [1, 2, 3, 40])
def test_idle_stream_template_replays_real_prime_tail_calendars_and_slot_lifetimes(name, waves):
    _STREAM_TEMPLATES.clear()
    profile = assumptions(hbm_read_buffer_bytes_per_stack=2 * 1024 * 1024)
    warm = PaperMemorySystem(hardware_by_id(name), assumptions=profile)
    total = waves * warm._buffer_stacks * warm._read_slot_bytes + 37
    warm.stream_read_compute(total, 123456.789)
    baseline = PaperMemorySystem(hardware_by_id(name), assumptions=profile, quiescent_stream_reuse=False)
    reused = PaperMemorySystem(hardware_by_id(name), assumptions=profile)
    # Alter initial physical slot order on one stack.  Template slots are
    # mapped oldest-first, not hard-wired to the source GPU's slot labels.
    for memory in (baseline, reused):
        ticket = memory.prefetch(0, 111)
        memory.consume(ticket, 333)
    ready = baseline.summary()["end_ns"] + 100
    for _ in range(5):
        before = baseline.stream_read_compute(total, 123456.789, ready_ns=ready)
        after = reused.stream_read_compute(total, 123456.789, ready_ns=ready)
        assert after.end_ns == pytest.approx(before.end_ns, rel=2e-12, abs=1e-6)
        _equivalent_memory_state(baseline, reused)
        assert baseline._slot_releases == reused._slot_releases
        ready = before.end_ns + 100
    assert reused._stream_cache_hits == 5


@pytest.mark.parametrize("change", [dict(memory_full_duplex=False), dict(host_full_duplex=False),
                                   dict(shared_io_full_duplex=False)])
def test_future_pd_shared_resource_prevents_idle_stream_reuse(change):
    _STREAM_TEMPLATES.clear()
    profile = assumptions(**change)
    warm = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    warm.stream_read_compute(50_000_000, 1000)
    baseline = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile, quiescent_stream_reuse=False)
    reused = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    for memory in (baseline, reused):
        memory.access("hbf", "write", 4096, stack=0, ready_ns=1_000_000_000)
    baseline.stream_read_compute(50_000_000, 1000)
    reused.stream_read_compute(50_000_000, 1000)
    assert reused._stream_cache_hits == 0
    _equivalent_memory_state(baseline, reused)


def test_unrelated_full_duplex_write_and_live_intermediate_are_preserved_by_reuse():
    _STREAM_TEMPLATES.clear()
    profile = assumptions()
    warm = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    warm.stream_read_compute(50_000_000, 1000)
    baseline = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile, quiescent_stream_reuse=False)
    reused = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    leases = []
    for memory in (baseline, reused):
        leases.append(memory.allocate_intermediate("still-needed", 40_000_000, stack=0))
        memory.access("hbf", "write", 4096, stack=0, ready_ns=1_000_000_000)
    old_write = reused._calendars["hbf:0:write"].available_ns
    baseline.stream_read_compute(50_000_000, 1000)
    reused.stream_read_compute(50_000_000, 1000)
    assert reused._stream_cache_hits == 1
    assert reused._calendars["hbf:0:write"].available_ns == old_write
    assert reused._leases[leases[1].allocation_id] is leases[1]
    assert leases[1].released_at_ns is None
    _equivalent_memory_state(baseline, reused)


def test_stream_template_does_not_import_another_gpu_historical_peak():
    _STREAM_TEMPLATES.clear()
    profile = assumptions()
    warm = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    ready = warm.stream_read_compute(50_000_000, 1000).end_ns
    warm.stream_read_compute(8, 100, ready_ns=ready)
    baseline = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile, quiescent_stream_reuse=False)
    reused = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    baseline.stream_read_compute(8, 100)
    reused.stream_read_compute(8, 100)
    assert reused._stream_cache_hits == 1
    assert reused.staging_peak_bytes == [1] * 8
    _equivalent_memory_state(baseline, reused)


@pytest.mark.parametrize("name,arrays,read_ns", [("HBF+", 400, 1000), ("CONV+", 256, 3000)])
def test_array_preparation_really_overlaps_previous_data_transfer(name, arrays, read_ns):
    memory = system(name, host_read_gb_s=1e9, shared_io_gb_s=1e9)
    size = arrays * 4096
    first = memory.access("hbf", "read", size, stack=0)
    first_port_end = memory._calendars["hbf:0:read"].available_ns
    second = memory.access("hbf", "read", size, stack=0)
    assert first.start_ns == 0
    assert second.start_ns == read_ns
    assert second.start_ns < first_port_end
    assert memory._calendars["hbf_array:0:read"].busy_ns == 2 * read_ns
    assert memory._calendars["hbf:0:read"].busy_ns == pytest.approx(2 * size / (
        hardware_by_id(name).hbf_read_gb_s / memory.hbf_stacks))
    assert first.end_ns >= read_ns + size / (hardware_by_id(name).hbf_read_gb_s / memory.hbf_stacks)


def test_conventional_array_count_caps_read_burst_without_increasing_staging():
    memory = system("CONV+")
    assert memory.hbf_array_units_per_stack == 16 * 16
    assert memory._read_slot_bytes == 16 * 16 * 4096
    assert memory.summary()["staging_capacity_bytes_per_stack"] == 3_130_000
    with pytest.raises(PaperMemoryCapacityError):
        memory.prefetch(0, 16 * 16 * 4096 + 1)
    with pytest.raises(PaperMemoryCapacityError, match="finite array group"):
        memory.access("hbf", "read", 16 * 16 * 4096 + 1, stack=0)
    assert system("HBF+")._read_slot_bytes == 1_565_000


def test_logic_die_prefetch_does_not_cross_gpu_link_before_consumption():
    memory = system(sram_access_ns=1., shared_io_gb_s=100_000.)
    ticket = memory.prefetch(0, 1_565_000)
    assert ticket.read.end_ns == 1157.5  # tREAD + paid local SRAM fill
    assert "host:read" not in memory._calendars
    assert "gpu_io:read" not in memory._calendars
    assert "hbf:0:read" not in memory._calendars
    consumed = memory.consume(ticket, 0., ready_ns=1_000_000.)
    assert consumed.end_ns == pytest.approx(1_000_979.125)
    drain = next(event for event in memory.events if event.operation == "staging_read")
    assert drain.start_ns == 1_000_000.
    assert {"staging:0:read", "hbf:0:read", "host:read", "gpu_io:read"} <= set(drain.resources)
    assert memory._calendars["host:read"].available_ns > 1_000_000.
    assert memory._calendars["gpu_io:read"].available_ns > 1_000_000.
    assert all("host:read" not in event.resources for event in memory.events
               if event.operation in {"read", "staging_fill"})
    assert memory.summary()["hbm_read_buffer_capacity_bytes_per_stack"] == 0
    assert ticket.released_at_ns == consumed.end_ns


def test_logic_staging_consume_pays_existing_gpu_link_reservation_and_fifo_order():
    memory = system(host_read_gb_s=100., host_write_gb_s=100., shared_io_gb_s=100.,
                    host_full_duplex=False, shared_io_full_duplex=False)
    first, second = memory.prefetch(0, 4096), memory.prefetch(1, 4096)
    # PD ingress legitimately owns the shared host/GPU link in the future.
    programmed = memory.access("hbf", "write", 4096, stack=2, ready_ns=1_000_000.)
    link_end = memory._calendars["host:shared"].available_ns
    consume = memory.consume_many((first, second), 500.)
    drains = [event for event in memory.events if event.operation == "staging_read"]
    assert drains[0].start_ns >= link_end
    assert drains[1].start_ns >= memory.hardware["page_read_ns"]
    assert consume.start_ns >= max(event.end_ns for event in drains)
    assert first.released_at_ns == second.released_at_ns == consume.end_ns
    assert memory._calendars["host:shared"].operations == 3
    assert memory._calendars["host:shared"].busy_ns == pytest.approx(3*4096/100.)
    assert memory._calendars["hbf:2:write"].available_ns == programmed.end_ns


def test_cancelled_logic_prefetch_pays_array_and_fill_but_no_unused_gpu_transfer():
    memory = system()
    ticket = memory.prefetch(0, 4096)
    memory.cancel_prefetch(ticket)
    assert memory._calendars["hbf_array:0:read"].operations == 1
    assert memory._calendars["staging:0:write"].operations == 1
    assert not any(key.startswith(("host:", "gpu_io:", "hbf:0:")) for key in memory._calendars)
    assert memory.bytes_by_operation == {"hbf:read": 4096, "staging:staging_fill": 4096}


@pytest.mark.parametrize("profile", [dict(), dict(sram_access_ns=17.3),
    dict(host_read_gb_s=13., shared_io_gb_s=21., sram_read_gb_s_per_stack=44.),
    dict(memory_full_duplex=False, host_full_duplex=False, shared_io_full_duplex=False)])
def test_corrected_logic_staging_bulk_and_both_replay_levels_match_real_events(profile):
    slow = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions(**profile),
                             bulk_wave_acceleration=False, quiescent_stream_reuse=False)
    fast = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions(**profile))
    total = 70*8*slow._read_slot_bytes + 1337
    signature = ("logic-staging-repeat", total)
    for memory in (slow, fast):
        memory.access("hbf", "write", 4096, stack=0, ready_ns=20_000.)
    start = max(slow.summary()["end_ns"], fast.summary()["end_ns"])
    before = fast.snapshot_counters(signature=signature)
    end = fast.stream_read_compute(total, 55_000.7, ready_ns=start).end_ns
    reference = slow.stream_read_compute(total, 55_000.7, ready_ns=start).end_ns
    assert end == pytest.approx(reference)
    _equivalent_memory_state(slow, fast)
    template = fast.make_quiescent_transition(before, start, end, signature=signature)
    assert template is not None
    ready = end + 100.
    assert fast.can_apply_quiescent_transition(template, ready, signature=signature)
    repeated = fast.apply_quiescent_transition(template, ready, signature=signature)
    expected = slow.stream_read_compute(total, 55_000.7, ready_ns=ready).end_ns
    assert repeated == pytest.approx(expected)
    _equivalent_memory_state(slow, fast)
    # An ordinary repeated stream exercises its separate same-work cache.
    ready = max(repeated, expected) + 100.
    slow.stream_read_compute(total, 55_000.7, ready_ns=ready)
    fast.stream_read_compute(total, 55_000.7, ready_ns=ready)
    assert fast._stream_cache_hits > 0
    _equivalent_memory_state(slow, fast)


def test_long_gpu_compute_cannot_build_an_unbounded_queue_of_prepared_array_results():
    memory = system()
    first = memory.prefetch(0, memory._read_slot_bytes)
    second = memory.prefetch(0, memory._read_slot_bytes)
    consumed = memory.consume(first, 1_000_000)
    third = memory.prefetch(0, memory._read_slot_bytes)
    third_data_start = memory._calendars["hbf_read_credit:0"].available_ns
    assert third_data_start >= consumed.end_ns
    # The next new read may prepare while third's data transfer runs, but
    # cannot prepare before third has a data-stage handoff credit.
    memory.consume(second, 1_000_000)
    fourth = memory.prefetch(0, memory._read_slot_bytes)
    reads = [event for event in memory.events if event.operation == "read" and event.tier == "hbf"]
    assert reads[-1].start_ns >= third_data_start
    assert fourth.read.end_ns > third.read.end_ns
    with pytest.raises(PaperBufferCreditError):
        memory.prefetch(0, 1)


@pytest.mark.parametrize("name", ["HBF+", "CONV+"])
def test_large_pure_read_reaches_finite_array_or_table_service_ceiling_without_erasing_tread(name):
    hardware = hardware_by_id(name)
    profile = assumptions(sram_read_gb_s_per_stack=1e9, sram_write_gb_s_per_stack=1e9,
                          host_read_gb_s=1e9, shared_io_gb_s=1e9)
    memory = PaperMemorySystem(hardware, assumptions=profile)
    size = 100_000_000_000
    end = memory.stream_read_compute(size, 0).end_ns
    expected = hardware.hbf_stacks * min(memory._read_slot_bytes / hardware.page_read_ns,
                                         hardware.hbf_read_gb_s / hardware.hbf_stacks)
    observed = size / end
    assert 0.999 * expected < observed <= expected
    assert all(memory.summary()["resources"][f"hbf_array:{stack}:read"]["busy_ns"] > 0
               for stack in range(hardware.hbf_stacks))


@pytest.mark.parametrize("name", ["HBF", "HBF+", "CONV", "CONV+"])
def test_program_service_bandwidth_includes_tprog_and_small_pages_keep_the_floor(name):
    memory = system(name)
    small = memory.access("hbf", "write", 4096, stack=0)
    assert memory._calendars["hbf:0:write"].busy_ns == memory.hardware["page_program_ns"]
    rate = memory.hardware["hbf_write_gb_s"] / memory.hbf_stacks
    large_bytes = 10_000_000
    large = memory.access("hbf", "write", large_bytes, stack=0)
    assert memory._calendars["hbf:0:write"].busy_ns == pytest.approx(
        memory.hardware["page_program_ns"] + large_bytes / rate)
    assert large.end_ns - small.end_ns == pytest.approx(large_bytes / rate)
    assert memory.summary()["bytes_by_operation"]["hbf:write"] == 4096 + large_bytes


def test_half_duplex_program_service_blocks_the_real_read_array():
    memory = system(memory_full_duplex=False)
    written = memory.access("hbf", "write", 4096, stack=0)
    read = memory.access("hbf", "read", 4096, stack=0)
    assert read.start_ns >= written.end_ns
    assert "hbf_array:0:shared" in written.resources
    assert "hbf_array:0:shared" in read.resources


def test_whole_step_transition_reuses_a_then_b_then_a_with_exact_slot_mapping():
    slow = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions(), quiescent_stream_reuse=False)
    fast = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("whole-step-A", 1, 123456.789)
    before = fast.snapshot_counters(signature=signature)
    a_end = _quiescent_layer(fast, 0, wave_count=1)
    slow_end = _quiescent_layer(slow, 0, wave_count=1)
    state = fast.summary()
    template = fast.make_quiescent_transition(before, 0, a_end, signature=signature)
    assert template is not None  # odd slot rotation is represented, not reset
    assert fast.summary() == state
    for _ in range(5):
        # Different whole-step B returns to a different physical initial
        # slot order than A's source.  Subsequent A maps oldest-first slots.
        fast_end = _quiescent_layer(fast, a_end, wave_count=2)
        slow_end = _quiescent_layer(slow, slow_end, wave_count=2)
        lease_ids = tuple(fast._leases)
        release_heap = tuple(fast._release_heap)
        state = fast.summary()
        serial = fast._snapshot_serial
        for _ in range(3):
            assert fast.can_apply_quiescent_transition(template, fast_end, signature=signature)
            assert fast.summary() == state
            assert tuple(fast._leases) == lease_ids and tuple(fast._release_heap) == release_heap
            assert fast._snapshot_serial == serial
        a_end = fast.apply_quiescent_transition(template, fast_end, signature=signature)
        slow_end = _quiescent_layer(slow, slow_end, wave_count=1)
        assert a_end == pytest.approx(slow_end, rel=2e-12, abs=1e-6)
        _equivalent_memory_state(slow, fast)
        assert fast._slot_releases == slow._slot_releases
        assert fast._next_lease == slow._next_lease
        assert fast._allocation_frontier_ns == pytest.approx(slow._allocation_frontier_ns)


@pytest.mark.parametrize("reason", ["future-pd", "live-lease", "signature", "geometry", "assumptions", "trace", "foreign"])
def test_whole_step_preflight_rejects_without_mutating_any_resource(reason):
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("step",)
    before = memory.snapshot_counters(signature=signature)
    end = _quiescent_layer(memory, 0)
    template = memory.make_quiescent_transition(before, 0, end, signature=signature)
    assert template is not None
    if reason == "future-pd":
        memory.access("hbf", "write", 4096, stack=0, ready_ns=end + 1_000_000)
    elif reason == "live-lease":
        memory.allocate_intermediate("still-in-use", 100, stack=0, ready_ns=end)
    elif reason == "geometry":
        memory.hardware["planes_per_hbf_die"] += 1
    elif reason == "assumptions":
        memory.assumptions = assumptions(host_write_gb_s=1000)
    elif reason == "trace":
        memory.retain_trace = True
    elif reason == "foreign":
        memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    supplied = ("different-step",) if reason == "signature" else signature
    state = memory.summary()
    serial = memory._snapshot_serial
    leases = tuple(memory._leases)
    assert not memory.can_apply_quiescent_transition(template, end, signature=supplied)
    assert memory.apply_quiescent_transition(template, end, signature=supplied) is None
    assert memory.summary() == state and memory._snapshot_serial == serial
    assert tuple(memory._leases) == leases


def test_whole_transition_leaves_unused_future_pd_calendars_at_actual_times():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("compute-only-step", 100)
    before = memory.snapshot_counters(signature=signature)
    memory.compute(100)
    template = memory.make_quiescent_transition(before, 0, 100, signature=signature)
    assert template is not None
    pd = memory.access("hbf", "write", 4096, stack=0, ready_ns=1_000_000)
    write_before = memory._calendars["hbf:0:write"].available_ns
    assert memory.can_apply_quiescent_transition(template, 100, signature=signature)
    assert memory.apply_quiescent_transition(template, 100, signature=signature) == 200
    assert memory._calendars["hbf:0:write"].available_ns == write_before == pd.end_ns


def test_outer_whole_step_snapshot_survives_inner_layer_period_snapshots():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    outer_signature, inner_signature = ("whole", 4), ("layer", 2)
    outer = memory.snapshot_counters(signature=outer_signature)
    inner = memory.snapshot_counters(signature=inner_signature)
    end = _quiescent_layer(memory, 0, wave_count=2)
    assert memory.repeat_quiescent_period(inner, 0, end, 3, signature=inner_signature)
    final = end * 4
    template = memory.make_quiescent_transition(outer, 0, final, signature=outer_signature)
    assert template is not None
    assert memory.can_apply_quiescent_transition(template, final, signature=outer_signature)
    assert memory.apply_quiescent_transition(template, final, signature=outer_signature) == final * 2


def test_whole_transition_refuses_index_dependent_in_period_slot_ties():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("tie-sensitive",)
    before = memory.snapshot_counters(signature=signature)
    first, second = memory.prefetch(0, 100), memory.prefetch(0, 100)
    end = memory.consume_many((first, second), 100).end_ns
    ticket = memory.prefetch(0, 100, ready_ns=end)
    end = memory.consume(ticket, 100).end_ns
    state = memory.summary()
    assert memory.make_quiescent_transition(before, 0, end, signature=signature) is None
    assert memory.summary() == state


def _reserved_decode_body(memory, start, waves=1):
    lease = memory.allocate_intermediate("decode-live", 10000, stack=0, ready_ns=start)
    workspace = memory.access("sram", "write", 10000, stack=0, ready_ns=start)
    read = memory.stream_read_compute(waves * memory.hbf_stacks * memory._read_slot_bytes,
                                      123456.789, ready_ns=workspace.end_ns)
    memory.release_intermediate(lease, at_ns=read.end_ns)
    return memory.access("hbf", "write", 4096, stack=0, ready_ns=read.end_ns).end_ns


def test_reserved_frontier_transition_matches_a_pd_a_after_different_b_without_copying_pd_costs():
    profile = assumptions()
    slow = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile, quiescent_stream_reuse=False)
    fast = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=profile)
    signature = ("decode-A-with-PD-frontier", 1)
    for memory in (slow, fast):
        memory.access("hbf", "write", 8192, stack=0, ready_ns=1_000_000)
    before = fast.snapshot_counters(signature=signature)
    now = _reserved_decode_body(fast, 0)
    slow_now = _reserved_decode_body(slow, 0)
    assert fast.make_quiescent_transition(before, 0, now, signature=signature) is None
    state = fast.summary()
    template = fast.make_quiescent_transition(before, 0, now, signature=signature,
                                               allow_reserved_frontiers=True)
    assert template is not None and template.allow_reserved_frontiers
    assert any(offset > 0 for _, offset, _ in template.input_frontiers)
    assert fast.summary() == state
    for difference in (0, 4096, 0):
        # B has a different request workload and a different reservation.
        for memory in (slow, fast):
            memory.access("hbf", "write", 8192, stack=0, ready_ns=now + 2_000_000)
        now = _reserved_decode_body(fast, now, waves=2)
        slow_now = _reserved_decode_body(slow, slow_now, waves=2)
        for memory in (slow, fast):
            memory.access("hbf", "write", 8192, stack=0, ready_ns=now + 1_000_000 + difference)
        state = fast.summary()
        eligible = fast.can_apply_quiescent_transition(template, now, signature=signature)
        assert eligible == (difference == 0)
        assert fast.summary() == state
        if eligible:
            now = fast.apply_quiescent_transition(template, now, signature=signature)
        else:
            assert fast.apply_quiescent_transition(template, now, signature=signature) is None
            assert fast.summary() == state
            now = _reserved_decode_body(fast, now)
        slow_now = _reserved_decode_body(slow, slow_now)
        assert now == pytest.approx(slow_now, rel=2e-12, abs=1e-6)
        _equivalent_memory_state(slow, fast)
        assert fast._next_lease == slow._next_lease
        assert fast._slot_releases == slow._slot_releases


def test_reserved_frontier_template_preserves_unused_source_and_target_future_pd():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("A-independent-of-stack-seven-program",)
    memory.access("hbf", "write", 1_000_000_000, stack=7)
    before = memory.snapshot_counters(signature=signature)
    end = _reserved_decode_body(memory, 0)
    assert memory._calendars["hbf:7:write"].available_ns > end
    assert memory.make_quiescent_transition(before, 0, end, signature=signature) is None
    template = memory.make_quiescent_transition(before, 0, end, signature=signature,
                                               allow_reserved_frontiers=True)
    assert template is not None
    memory.access("hbf", "write", 1_000_000_000, stack=7, ready_ns=end)
    unused = memory._calendars["hbf:7:write"]
    state = (unused.available_ns, unused.busy_ns, unused.operations)
    assert memory.can_apply_quiescent_transition(template, end, signature=signature)
    assert memory.apply_quiescent_transition(template, end, signature=signature) is not None
    assert (unused.available_ns, unused.busy_ns, unused.operations) == state


def test_reserved_frontier_matching_never_ignores_positive_or_adds_positive_to_idle():
    memory = PaperMemorySystem(hardware_by_id("HBF+"), assumptions=assumptions())
    signature = ("positive-state",)
    memory.access("hbf", "write", 4096, stack=0, ready_ns=1_000_000)
    before = memory.snapshot_counters(signature=signature)
    end = _reserved_decode_body(memory, 0)
    template = memory.make_quiescent_transition(before, 0, end, signature=signature,
                                               allow_reserved_frontiers=True)
    assert template is not None
    # No replacement reservation: positive source program state is now idle.
    state = memory.summary()
    assert not memory.can_apply_quiescent_transition(template, end, signature=signature)
    assert memory.apply_quiescent_transition(template, end, signature=signature) is None
    assert memory.summary() == state
