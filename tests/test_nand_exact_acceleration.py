from dataclasses import replace

import pytest

from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore


class PageReference(NandCore):
    def _execute_accepted(self, request):
        return self._execute_detailed_accepted(request)


def config(**changes):
    return replace(NandConfig(
        kind="HBF", channels=4, planes_per_lun=2, blocks_per_plane=1024,
        page_bytes=64, host_granularity_bytes=64, pages_per_block=8,
        host_bandwidth_gb_s=8, internal_bandwidth_gb_s=16,
        page_read_ns=50, page_program_ns=400, block_erase_ns=70, front_ns=7,
    ), capacity_bytes=None, **changes)


def assert_times(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=2e-11, abs=0.0001)


def assert_state(actual, expected):
    assert_times(actual._array_ready, expected._array_ready)
    assert_times(actual._buffer_ready, expected._buffer_ready)
    for field in ("ready_ns", "lane_available", "busy_ns", "last_intervals"):
        assert_times(getattr(actual.timeline, field), getattr(expected.timeline, field))
    assert actual.timeline.bytes_moved == expected.timeline.bytes_moved
    assert actual.timeline.directions == expected.timeline.directions
    assert actual.timeline._touched == expected.timeline._touched
    assert actual._inflight == pytest.approx(expected._inflight, rel=2e-11, abs=0.0001)


def compare(actual, expected, request):
    a, e = actual.submit(request), expected.submit(request)
    assert a.completion_ns == pytest.approx(e.completion_ns, rel=2e-11, abs=0.0001)
    assert a.logical_bytes == e.logical_bytes
    assert a.transfer_bytes == e.transfer_bytes
    for key in ("host_transfer_bytes", "internal_transfer_bytes", "physical_read_bytes",
                "physical_write_bytes", "pages_read", "pages_programmed", "erase_operations", "page_count"):
        assert a.counters[key] == e.counters[key]
    assert_times(a.counters["resource_busy_ns"], e.counters["resource_busy_ns"])
    assert a.counters["resource_bytes"] == e.counters["resource_bytes"]
    assert_state(actual, expected)
    return a


@pytest.mark.parametrize("cfg", [
    config(channels=1), config(), config(planes_independent=True),
    config(channels=2, targets_per_channel=2, dies_per_target=2, luns_per_die=2),
    config(parallel_units=3),
    config(host_bandwidth_gb_s=0.25), config(internal_bandwidth_gb_s=0.125),
    config(page_read_ns=0, page_program_ns=0, front_ns=0),
    config(internal_transfer_bytes=100), config(page_bytes=6, host_granularity_bytes=4),
    config(metadata={"host_resource_id": "nand:array:0"}),
    config(metadata={"channel_resource_prefix": "nand:array"}),
])
def test_cycle_acceleration_preserves_all_states_and_followup_requests(cfg):
    actual, expected = NandCore(cfg, capture_details=False), PageReference(cfg, capture_details=False)
    page = cfg.page_bytes
    first = 3 * page + 1
    for op in (Operation.READ, Operation.WRITE, Operation.READ):
        result = compare(actual, expected, AccessRequest("large", op, first, 5000 * page - 2, 0))
        assert result.counters["accelerated_pages"] > 0
    for op in (Operation.READ, Operation.WRITE, Operation.ERASE, Operation.READ):
        compare(actual, expected, AccessRequest("probe", op, first + 4999 * page, page - 2, 10000000))


def test_existing_resource_contention_and_later_bottleneck_switch():
    cfg = config(parallel_units=3)
    actual, expected = NandCore(cfg, capture_details=False), PageReference(cfg, capture_details=False)
    for core in (actual, expected):
        # A far-future busy channel becomes critical after a different rate
        # dominates initially; equal consecutive deltas alone are insufficient.
        core.timeline.reserve("nand:channel:2", 0, 600000)
        core.timeline.reserve("nand:host", 0, 80000)
        core.timeline.reserve("nand:parallel:1", 0, 500000)
        core._array_ready["nand:array:0"] = 40000
        core._buffer_ready["nand:array:3"] = 100000
    for op in (Operation.READ, Operation.WRITE, Operation.READ):
        compare(actual, expected, AccessRequest("busy", op, 64, 15000 * 64, 0))


def test_shared_host_between_cores_keeps_exact_contention():
    cfg = config(metadata={"host_resource_id": "shared:host"})
    actual, expected = NandCore(cfg, capture_details=False), PageReference(cfg, capture_details=False)
    for op in (Operation.READ, Operation.WRITE):
        compare(actual, expected, AccessRequest("first", op, 0, 5000 * 64))
        other_cfg = replace(cfg, metadata={"host_resource_id": "shared:host", "array_resource_prefix": "other:array", "channel_resource_prefix": "other:channel"})
        other_a = NandCore(other_cfg, actual.timeline, capture_details=False)
        other_e = PageReference(other_cfg, expected.timeline, capture_details=False)
        compare(other_a, other_e, AccessRequest("second", op, 0, 5000 * 64))


def test_multilane_and_huge_host_phase_fall_back_to_page_execution():
    for cfg in (config(), config(host_granularity_bytes=65537)):
        actual, expected = NandCore(cfg, capture_details=False), PageReference(cfg, capture_details=False)
        for core in (actual, expected):
            core.timeline.lane_available["nand:channel:0"] = [500, 1000]
            core.timeline.ready_ns["nand:channel:0"] = 500
        result = compare(actual, expected, AccessRequest("fallback", Operation.READ, 17, 5000 * 64 - 19))
        assert result.counters["accelerated_pages"] == 0


def test_threshold_and_detailed_trace_do_not_change_timing():
    cfg = config(page_bytes=4096, host_granularity_bytes=4096, pages_per_block=256,
                 host_bandwidth_gb_s=10, internal_bandwidth_gb_s=20, page_read_ns=50000)
    actual, expected = NandCore(cfg, capture_details=False), PageReference(cfg, capture_details=False)
    for pages in (127, 128, 8192, 8193, 9000):
        compare(actual, expected, AccessRequest("threshold", Operation.READ, 0, pages * 4096))
    detailed = NandCore(cfg, capture_details=True).submit(AccessRequest("trace", Operation.READ, 0, 9000 * 4096))
    reference = PageReference(cfg).submit(AccessRequest("trace", Operation.READ, 0, 9000 * 4096))
    assert detailed == reference


def test_erase_retains_real_block_deduplication_even_above_detail_limit():
    cfg = config(max_expanded_segments=1)
    actual, expected = NandCore(cfg, capture_details=False), PageReference(cfg, capture_details=False)
    result = compare(actual, expected, AccessRequest("erase", Operation.ERASE, 65, 5000 * 64 - 2))
    assert result.counters["host_transfer_bytes"] == 0
    assert result.counters["erase_operations"] == result.counters["page_count"]


def test_empty_resource_id_keeps_original_validation_error():
    cfg = config(metadata={"host_resource_id": ""})
    for cls in (NandCore, PageReference):
        with pytest.raises(ValueError, match="resource_id must be a non-empty string"):
            cls(cfg, capture_details=False).submit(AccessRequest("invalid", Operation.READ, 0, 5000 * 64))


def test_custom_timeline_hooks_are_not_bypassed():
    class InstrumentedTimeline(ResourceTimeline):
        calls = 0

        def reserve(self, *args, **kwargs):
            self.calls += 1
            return super().reserve(*args, **kwargs)

    timeline = InstrumentedTimeline()
    result = NandCore(config(), timeline, capture_details=False).submit(
        AccessRequest("hooks", Operation.READ, 0, 5000 * 64))
    assert timeline.calls == 15000
    assert result.counters["accelerated_pages"] == 0
