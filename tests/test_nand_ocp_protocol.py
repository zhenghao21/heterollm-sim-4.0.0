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
    {"write_buffer_bytes": PAGE}, {"channels": 32, "host_channels": 32},
])
def test_ocp_contract_rejects_inconsistent_physical_settings(updates):
    with pytest.raises(ValueError, match="ocp_hbf_0_7"):
        config(protocol_mode="ocp_hbf_0_7", **updates)


@pytest.mark.parametrize("req", [
    request("fragment", "write", count=64),
    request("unaligned-read", address=1),
    request("unaligned-size", count=65),
    request("explicit-erase", "erase", count=PAGE),
    request("skip-page", "write", address=PAGE, count=PAGE),
])
def test_ocp_rejects_unsupported_or_illegal_requests_before_mutation(req):
    core = NandCore(config(protocol_mode="ocp_hbf_0_7"))
    before = deepcopy(vars(core))
    with pytest.raises(ValueError, match="ocp_hbf_0_7"):
        core.submit(req)
    assert vars(core) == before


def test_ocp_batch_program_order_autoerase_and_media_completion():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7"))
    results = core.execute_batch([
        request("page0", "write", count=PAGE),
        request("page1", "write", address=PAGE, count=PAGE),
    ])
    first, second = results
    assert first.erase_operations == 1
    assert second.erase_operations == 0
    assert first.counters["rmw_read_operations"] == second.counters["rmw_read_operations"] == 0
    assert stage(first, "PAGE_PROGRAM").start_ns >= stage(first, "BLOCK_ERASE").end_ns
    assert first.completion_ns == stage(first, "PAGE_PROGRAM").end_ns
    assert first.counters["write_completion"] == "media_program_complete"
    assert stage(second, "PAGE_PROGRAM").start_ns >= first.completion_ns
    with pytest.raises(ValueError, match="write order violation"):
        core.submit(request("rewrite1", "write", address=PAGE, count=PAGE))
    core.submit(request("cache-page1", address=PAGE))
    restart = core.submit(request("restart0", "write", count=PAGE))
    assert restart.erase_operations == 1
    assert all(slot.page_key is None for slots in core._read_cache.values() for slot in slots)


def test_ocp_invalid_batch_is_rejected_without_programming_valid_prefix():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7"))
    with pytest.raises(ValueError, match="write order violation"):
        core.execute_batch([request("page0", "write", count=PAGE),
                            request("skip2", "write", address=2 * PAGE, count=PAGE)])
    assert not core.timeline.ready_ns
    assert not core._next_program_page


def test_ocp_striped_aggregate_write_tracks_order_per_physical_block():
    core = NandCore(config(protocol_mode="ocp_hbf_0_7", channels=2, host_channels=2))
    result = core.submit(request("stripe", "write", count=4 * PAGE))
    assert result.pages_programmed == 4
    assert result.erase_operations == 2
    assert set(core._next_program_page.values()) == {2}


def test_summary_and_detail_execute_same_host_cache_and_protocol_state():
    cfg = config(protocol_mode="ocp_hbf_0_7", channels=2, host_channels=2)
    detailed, summary = NandCore(cfg), NandCore(cfg, capture_details=False)
    trace = [request("program", "write", count=256 * PAGE),
             request("read", count=256 * PAGE),
             request("tail-hit", address=255 * PAGE),
             request("repeat-page0", "write", count=PAGE)]
    for req in trace:
        a, b = detailed.submit(req), summary.submit(req)
        assert a.completion_ns == b.completion_ns
        for name in ("host_transfer_bytes", "internal_transfer_bytes", "pages_read",
                     "pages_programmed", "erase_operations", "read_cache_hits", "read_cache_misses"):
            assert a.counters[name] == b.counters[name]
    assert detailed._read_cache == summary._read_cache
    assert detailed._next_program_page == summary._next_program_page
