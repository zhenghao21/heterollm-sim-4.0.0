"""Protocol boundaries and physical data-buffer lifetimes for HBF execution."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.memory_types import AccessRequest, NandConfig
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore


PAGE = 4096


def config(**updates):
    values = dict(kind="HBF", channels=1, blocks_per_plane=32, pages_per_block=16,
                  page_bytes=PAGE, host_granularity_bytes=64,
                  host_bandwidth_gb_s=64., host_read_bandwidth_gb_s=64.,
                  host_write_bandwidth_gb_s=64., internal_bandwidth_gb_s=4096.,
                  page_read_ns=100., page_program_ns=1000., block_erase_ns=2000.,
                  host_full_duplex=True, read_cache_pages_per_bank=2)
    if updates.get("protocol_mode") == "ocp_hbf_0_7":
        values.update(queue_scope="host_channel", planes_independent=True,
                      planes_per_lun=1, parallel_units=None)
    values.update(updates)
    return NandConfig(**values)


def request(name, op="read", address=0, count=64, arrival=0.):
    return AccessRequest(name, op, address, count, arrival)


def stage(result, name):
    return next(item for item in result.stages if item.name == name)


@pytest.mark.parametrize("update", [
    {"host_channels": 2}, {"host_channels": True},
    {"read_cache_pages_per_bank": -1}, {"host_full_duplex": 1},
    {"protocol_mode": "legacy_hbf"},
    {"queue_scope": "channel"}, {"max_outstanding_write_requests": 0},
    {"max_outstanding_write_requests": True},
    {"max_outstanding_requests": 2, "max_outstanding_write_requests": 3},
])
def test_invalid_transport_configuration_is_rejected(update):
    with pytest.raises(ValueError):
        config(**update)


def test_independent_host_channels_preserve_aggregate_bandwidth():
    core = NandCore(config(channels=2, host_channels=2,
                          host_bandwidth_gb_s=2., host_read_bandwidth_gb_s=2.,
                          host_write_bandwidth_gb_s=2.))
    left = core.submit(request("left", count=PAGE))
    right = core.submit(request("right", address=PAGE, count=PAGE))
    a, b = stage(left, "HOST_TRANSFER"), stage(right, "HOST_TRANSFER")
    assert a.resource_id == "nand:host:0:read"
    assert b.resource_id == "nand:host:1:read"
    assert a.start_ns == b.start_ns
    assert a.end_ns - a.start_ns == b.end_ns - b.start_ns == PAGE
    assert left.bandwidth_ceiling_gb_s == 2.


def test_full_duplex_allows_overlapping_read_and_write_on_one_host_channel():
    core = NandCore(config(channels=2, host_channels=1,
                          host_bandwidth_gb_s=1., host_read_bandwidth_gb_s=1.,
                          host_write_bandwidth_gb_s=1.))
    read = core.submit(request("read", count=PAGE))
    write = core.submit(request("write", "write", PAGE, PAGE))
    r, w = stage(read, "HOST_TRANSFER"), stage(write, "HOST_TRANSFER")
    assert r.resource_id == "nand:host:read"
    assert w.resource_id == "nand:host:write"
    assert max(r.start_ns, w.start_ns) < min(r.end_ns, w.end_ns)


def test_cache_hit_waits_for_future_fill_and_charges_only_requested_host_bytes():
    timeline = ResourceTimeline(lane_available={"nand:host:read": [0., 0.]})
    core = NandCore(config(internal_bandwidth_gb_s=1.), timeline)
    miss = core.submit(request("miss"))
    hit = core.submit(request("hit"))
    assert miss.host_transfer_bytes == hit.host_transfer_bytes == 64
    assert miss.internal_transfer_bytes == PAGE
    assert miss.pages_read == 1
    assert hit.pages_read == hit.internal_transfer_bytes == 0
    assert hit.counters["read_cache_hits"] == 1
    assert stage(hit, "CACHE_READ").start_ns >= stage(miss, "INTERNAL_TRANSFER").end_ns
    assert hit.completion_ns >= stage(miss, "INTERNAL_TRANSFER").end_ns


def test_third_page_reuses_buffer_only_after_inflight_reader_finishes():
    timeline = ResourceTimeline(lane_available={"nand:host:read": [0., 0., 0.]})
    core = NandCore(config(page_read_ns=1., host_bandwidth_gb_s=.01,
                          host_read_bandwidth_gb_s=.01, host_write_bandwidth_gb_s=.01), timeline)
    first = core.submit(request("first", count=PAGE))
    second = core.submit(request("second", address=PAGE, count=PAGE))
    third = core.submit(request("third", address=2 * PAGE, count=PAGE))
    assert stage(second, "INTERNAL_TRANSFER").end_ns < first.completion_ns
    assert stage(third, "INTERNAL_TRANSFER").start_ns >= first.completion_ns
    assert len(next(iter(core._read_cache.values()))) == 2
    old = core.submit(request("evicted", address=0))
    assert old.pages_read == 1


@pytest.mark.parametrize("operation,count", [("write", 64), ("write", PAGE), ("erase", PAGE)])
def test_write_rmw_and_erase_invalidate_cached_page(operation, count):
    core = NandCore(config())
    core.submit(request("fill"))
    mutation = core.submit(request("mutate", operation, count=count))
    read = core.submit(request("read-after"))
    assert read.pages_read == 1
    assert stage(read, "PAGE_READ").start_ns >= mutation.completion_ns
    if operation == "write" and count == 64:
        assert mutation.counters["rmw_read_operations"] == 1


def test_invalidated_buffer_retains_live_reader_dependency():
    timeline = ResourceTimeline(lane_available={"nand:host:read": [0., 0., 0.]})
    core = NandCore(config(page_read_ns=1., page_program_ns=1.,
                          host_bandwidth_gb_s=64., host_read_bandwidth_gb_s=.01), timeline)
    reader = core.submit(request("long-read", count=PAGE))
    core.submit(request("other-buffer", address=PAGE, count=PAGE))
    core.submit(request("write", "write", count=PAGE))
    fresh = core.submit(request("fresh", address=0))
    assert fresh.pages_read == 1
    # Invalidating a tag must not lose the physical slot's reader lifetime.
    assert stage(fresh, "INTERNAL_TRANSFER").start_ns >= reader.completion_ns


def test_flush_invalidates_read_cache_created_while_partial_write_is_staged():
    core = NandCore(config(write_buffer_bytes=PAGE))
    core.submit(request("write-part", "write"))
    overlay = core.submit(request("overlay", count=128))
    assert overlay.counters["staging_read_merges"] == 1
    assert overlay.pages_read == 1
    drained = core.flush_write_buffer()
    read = core.submit(request("after-drain", count=128))
    assert read.pages_read == 1
    assert stage(read, "PAGE_READ").start_ns >= drained["media_drain_completion_ns"]


@pytest.mark.parametrize("updates", [
    {"page_bytes": 16384}, {"host_granularity_bytes": 4096},
    {"host_full_duplex": False}, {"read_cache_pages_per_bank": 0},
    {"write_buffer_bytes": PAGE},
    {"parallel_units": 1}, {"planes_independent": False},
])
def test_ocp_contract_rejects_inconsistent_physical_settings(updates):
    with pytest.raises(ValueError, match="ocp_hbf_0_7"):
        config(protocol_mode="ocp_hbf_0_7", **updates)


def test_ocp_rejects_explicit_erase_before_mutation():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7"))
    before = deepcopy(vars(core))
    with pytest.raises(ValueError, match="ocp_hbf_0_7"):
        core.submit(request("explicit-erase", "erase", count=PAGE))
    assert vars(core) == before


@pytest.mark.parametrize("address,count,moved", [(0, 32, 64), (1, 64, 128), (96, 32, 64), (0, 65, 128)])
def test_host_covers_a_sub_64_byte_access_with_whole_64_byte_units(address, count, moved):
    core = NandCore(config(protocol_mode="ocp_hbf_0_7"))
    assert core.submit(request("tiny", address=address, count=count)).host_transfer_bytes == moved


def test_channel_write_pointer_rejects_skips_and_advances():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7", channels=2, host_channels=2,
                           planes_per_lun=2, planes_independent=True))
    assert core._write_pointer == [0, 0]
    core.submit(request("sequential", "write", count=4 * PAGE))
    assert core._write_pointer == [2, 2]
    with pytest.raises(ValueError, match="write order violation"):
        core.submit(request("skip", "write", address=0, count=PAGE))
    assert core._write_pointer == [2, 2]
    core.submit(request("reads-take-no-page", count=4 * PAGE))
    assert core._write_pointer == [2, 2]


def test_generic_nand_keeps_no_log():
    core = NandCore(config())
    core.submit(request("program", "write", count=PAGE))
    assert core._bank_log is None


def test_programs_on_one_bank_run_one_after_another_in_log_order():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7", channels=2, host_channels=2))
    programs = {}
    for index in range(16):
        result = core.submit(request(f"page{index}", "write", address=index * PAGE, count=PAGE))
        program = stage(result, "PAGE_PROGRAM")
        programs.setdefault(program.resource_id, []).append((program.start_ns, program.end_ns))
    assert core._write_pointer == [8, 8]
    for rows in programs.values():
        assert all(before[1] <= after[0] for before, after in zip(rows, rows[1:]))


def test_summary_and_detail_execute_same_host_cache_and_protocol_state():
    cfg = config(protocol_mode="ocp_hbf_0_7", channels=2, host_channels=2)
    detailed, summary = NandCore(cfg), NandCore(cfg, capture_details=False)
    trace = [request("program", "write", count=256 * PAGE),
             request("read", count=256 * PAGE),
             request("tail-hit", address=255 * PAGE),
             request("next-page", "write", address=256 * PAGE, count=PAGE)]
    for req in trace:
        a, b = detailed.submit(req), summary.submit(req)
        assert a.completion_ns == b.completion_ns
        for name in ("host_transfer_bytes", "internal_transfer_bytes", "pages_read",
                     "pages_programmed", "erase_operations", "read_cache_hits", "read_cache_misses"):
            assert a.counters[name] == b.counters[name]
    assert detailed._read_cache == summary._read_cache
    assert detailed._write_pointer == summary._write_pointer
    assert detailed._host_inflight == summary._host_inflight
    assert detailed._host_acceptance_ns == summary._host_acceptance_ns


def test_page_zero_auto_erase_precedes_program_and_blocks_same_bank_read():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7", channels=1,
                           host_channels=1, planes_per_lun=1,
                           block_erase_ns=2000., page_program_ns=1000.))
    write = core.submit(request("page-zero", "write", count=PAGE))
    erase, program = stage(write, "BLOCK_ERASE"), stage(write, "PAGE_PROGRAM")
    assert erase.end_ns <= program.start_ns
    read = core.submit(request("after-erase", "read", count=PAGE))
    assert stage(read, "PAGE_READ").start_ns >= program.end_ns
    assert write.counters["erase_operations"] == 1


def test_phase_replay_maps_logical_rewrites_to_sequential_physical_pages():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7", channels=1,
                           host_channels=1, planes_per_lun=2,
                           metadata={"write_pointer_phase": "first_write"}))
    first = core.submit(request("first", "write", address=0, count=PAGE))
    second = core.submit(request("rewrite", "write", address=0, count=PAGE))
    third = core.submit(request("next", "write", address=PAGE, count=PAGE))

    # The logical page is rewritten at the next physical DLU.  OCP's block
    # spans both planes, so plane 1/page 0 shares the plane-0 erase window.
    assert [first.counters["erase_operations"], second.counters["erase_operations"],
            third.counters["erase_operations"]] == [1, 0, 0]
    first_erase = stage(first, "BLOCK_ERASE")
    assert stage(second, "PAGE_PROGRAM").start_ns >= first_erase.end_ns
    assert core._write_pointer == [3]
    assert core._logical_page_table[0].address == PAGE
    assert core._logical_page_table[1].address == 2 * PAGE

    read = core.submit(request("read-latest", "read", address=0, count=PAGE))
    assert stage(read, "PAGE_READ").resource_id == "nand:array:1"


def test_logic_die_scratchpad_is_per_channel_and_high_priority_lane():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7", channels=2,
                           host_channels=2, planes_per_lun=1,
                           page_staging="logic_die",
                           page_staging_bytes_per_host_channel=2_500_000))
    result = core.submit(request("partial", "write", address=0, count=64))
    stage_write = stage(result, "STAGING_WRITE")
    assert stage_write.resource_id == "nand:scratchpad:0:write"
    assert core.page_staging_state()["capacity_bytes"] == 2_500_000


def _ocp_runtime(**updates):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    values = dict(protocol_mode="ocp_hbf_0_7", channels=1, host_channels=1,
                  metadata={"write_pointer_phase": "first_write"})
    values.update(updates)
    context = PhysicalRuntimeContext(config(**values), "flash", capture_details=False)
    return context.runtimes["flash"].core


def _billed(result):
    # The runtime context prefixes its owner to every resource.
    return {name.removeprefix("flash:") for name in result.counters["resource_busy_ns"]}


def _arrays(result):
    return sorted(name for name in _billed(result) if name.startswith("nand:array"))


def test_read_window_reads_pages_where_the_page_table_put_them():
    core = _ocp_runtime(planes_per_lun=4)
    for name, address in (("first", 0), ("rewrite", 0), ("one", PAGE), ("two", 2 * PAGE)):
        core.submit(request(name, "write", address=address, count=PAGE))
    # Logical pages 0..2 sit one DLU (one plane) later; page 3 was never written.
    assert core._read_pages(0, 4).tolist() == [1, 2, 3, 3]
    results = core.submit_many([request(str(page), address=page * PAGE, count=PAGE, arrival=1e6)
                                for page in range(3)])
    assert all("native_fullpage_pages" in result.counters for result in results)
    assert [_arrays(result) for result in results] == [["nand:array:1"], ["nand:array:2"], ["nand:array:3"]]


def test_read_window_serves_every_host_channel_at_once():
    # A request is admitted on its first page's host channel, but its pages
    # lie on every channel. With credits to spare, the host-channel window
    # schedules exactly as one device-wide window: no host channel's
    # requests queue behind another channel's requests on the NAND channels.
    def window(scope):
        core = _ocp_runtime(channels=4, host_channels=4, planes_per_lun=2, queue_scope=scope)
        return core.submit_many([request(str(index), address=index * 6 * PAGE, count=6 * PAGE, arrival=1e6)
                                 for index in range(8)])
    ocp, device = window("host_channel"), window("device")
    assert {index * 6 % 4 for index in range(8)} == {0, 2}
    assert all("native_fullpage_pages" in result.counters for result in ocp)
    assert [result.completion_ns for result in ocp] == [result.completion_ns for result in device]


def test_read_window_takes_credits_per_host_channel():
    core = _ocp_runtime(channels=2, host_channels=2, max_outstanding_requests=1,
                        max_outstanding_write_requests=1)
    first, second, other = core.submit_many([request("first", address=0, count=PAGE, arrival=1e6),
                                             request("second", address=2 * PAGE, count=PAGE, arrival=1e6),
                                             request("other", address=PAGE, count=PAGE, arrival=1e6)])
    assert all("native_fullpage_pages" in result.counters for result in (first, second, other))
    # Host channel 0 has one credit, so its second request waits for the
    # first; host channel 1 admits its request at once.
    assert first.counters["queue_wait_ns"] == other.counters["queue_wait_ns"] == 0
    assert second.arrival_ns + second.counters["queue_wait_ns"] == first.completion_ns


def test_read_window_moves_staged_pages_over_the_scratchpad_lane():
    core = _ocp_runtime(planes_per_lun=1, page_staging="logic_die",
                        page_staging_bytes_per_host_channel=2_500_000)
    core.submit(request("partial", "write", address=0, count=64))
    staged, media = core.submit_many([request("staged", address=0, count=64, arrival=1e6),
                                      request("media", address=5 * PAGE, count=PAGE, arrival=1e6)])
    assert "native_fullpage_pages" in staged.counters
    assert staged.counters["staging_read_hits"] == 1
    assert _billed(staged) == {"nand:scratchpad:0:read"}
    assert "nand:host:read" in _billed(media)
    assert "nand:scratchpad:0:read" not in _billed(media)
