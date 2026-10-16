"""Exact OCP aggregate execution against the ordinary page state machine."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_presets import resolve_physical_config


def _reference_submit(core, request):
    with patch.object(NandCore, "_submit_ocp_single_page_summary", return_value=None):
        return core.submit(request)


def _sum_metric(results, name):
    values = {}
    for result in results:
        for key, value in result.counters[name].items():
            values[key] = values.get(key, 0) + value
    return values


def _links(timeline, program, arrival):
    completion = arrival
    bills = []
    for _, direction, demands in program:
        started = completion
        start = max(completion, *(timeline.available_ns(resource) for resource, _, _ in demands))
        busy, moved, last = {}, {}, {}
        for resource, duration, size in demands:
            transfer = timeline.reserve(resource, start, duration, direction=direction)
            timeline.bytes_moved[resource] = timeline.bytes_moved.get(resource, 0) + size
            completion = max(completion, transfer.end_ns)
            busy[resource] = busy.get(resource, 0.) + duration
            moved[resource] = moved.get(resource, 0) + size
            last[resource] = (transfer.start_ns, transfer.end_ns)
        bills.append((started, completion, busy, moved, last))
    return completion, bills


@pytest.mark.parametrize("host_lanes", [1, 3])
def test_single_page_summary_preserves_cache_writes_admission_and_complete_state(host_lanes):
    config = replace(resolve_physical_config("hbf_ocp_0_7"), max_outstanding_requests=3)
    timeline = ResourceTimeline(lane_available={"nand:host:0:read": [0.] * host_lanes})
    expected = NandCore(config, deepcopy(timeline), capture_details=False)
    actual = NandCore(config, deepcopy(timeline), capture_details=False)
    requests = [AccessRequest(f"write-{i}", "write", i * 4096, 4096, 0.) for i in range(140)]
    requests += [AccessRequest(f"read-{i}", "read", (i % 140) * 4096 + (i % 8) * 64, 64, 0.)
                 for i in range(320)]
    requests += [AccessRequest("reuse-block", "write", 0, 4096, 900_000.),
                 AccessRequest("read-rewritten", "read", 0, 4096, 900_000.)]
    for request in requests:
        assert actual.submit(request) == _reference_submit(expected, request)
    assert vars(actual) == vars(expected)


@pytest.mark.parametrize("independent,parallel", [(False, None), (True, 3), (True, None)])
def test_compiled_pages_preserve_each_completion_cache_slots_and_multilane_links(independent, parallel):
    config = NandConfig(kind="HBF", protocol_mode="ocp_hbf_0_7", channels=2, page_bytes=4096,
        host_granularity_bytes=64,
        targets_per_channel=2, dies_per_target=2, luns_per_die=1, planes_per_lun=2,
        blocks_per_plane=8, pages_per_block=8, host_channels=2, host_full_duplex=True,
        read_cache_pages_per_bank=2, host_bandwidth_gb_s=128., host_read_bandwidth_gb_s=128.,
        host_write_bandwidth_gb_s=32., internal_bandwidth_gb_s=2048., page_read_ns=100.,
        page_program_ns=1000., block_erase_ns=2000., front_ns=7., max_outstanding_requests=3,
        planes_independent=independent, parallel_units=parallel)
    program = (("package", "hbf->gpu", (("wire", 4096 / 13., 4096), ("credit", 4096 / 13. + 71., 4096))),
               ("fabric", "switch->gpu", (("wire2", 4096 / 19., 4096), ("credit2", 4096 / 19. + 30., 0))))
    timeline = ResourceTimeline(lane_available={"wire": [4.], "credit": [20., 5., 2.],
        "wire2": [0.], "credit2": [12., 3.], "nand:host:0:read": [0., 10.],
        "nand:host:1:read": [0., 10.]})
    timeline.directions["credit"] = "earlier-direction"
    expected = NandCore(config, deepcopy(timeline), capture_details=False)
    actual = NandCore(config, deepcopy(timeline), capture_details=False)
    for core in (expected, actual):
        for index in range(16):
            _reference_submit(core, AccessRequest(str(index), "write", index * 4096, 4096, 0.))
    addresses = np.array([*(i * 4096 for i in range(48)), *(i * 4096 for i in range(16, 48)), 0, 4096] * 3)
    expected_pages, completions, expected_links = [], [], [[] for _ in program]
    for index, address in enumerate(addresses):
        result = _reference_submit(expected, AccessRequest(str(index), "read", int(address), 4096, 11.))
        expected_pages.append(result)
        completion, bills = _links(expected.timeline, program, result.completion_ns)
        completions.append(completion)
        for group, bill in zip(expected_links, bills):
            group.append(bill)
    result = actual.submit_read_pages(addresses, 11., link_program=program)
    assert result is not None
    completed, nand, link_results = result
    assert completed.tolist() == completions
    assert vars(actual) == vars(expected)
    for name in ("resource_busy_ns", "resource_bytes"):
        assert nand.counters[name] == _sum_metric(expected_pages, name)
    for name in ("pages_read", "read_cache_hits", "read_cache_misses", "host_transfer_bytes",
                 "internal_transfer_bytes", "host_admission_samples", "queue_wait_ns"):
        assert nand.counters[name] == sum(page.counters[name] for page in expected_pages)
    for name in ("host_outstanding_at_admission_peak", "host_read_outstanding_at_admission_peak",
                 "host_write_outstanding_at_admission_peak"):
        assert nand.counters[name] == max(page.counters[name] for page in expected_pages)
    for (owner, bill), (expected_owner, _, _), expected_bills in zip(link_results, program, expected_links):
        assert owner == expected_owner
        assert bill.arrival_ns == min(row[0] for row in expected_bills)
        assert bill.completion_ns == max(row[1] for row in expected_bills)
        for field, position in (("resource_busy_ns", 2), ("resource_bytes", 3)):
            totals = {}
            for row in expected_bills:
                for key, value in row[position].items():
                    totals[key] = totals.get(key, 0) + value
            assert bill.counters[field] == totals
        assert bill.counters["resource_last_intervals"] == expected_bills[-1][4]


def test_compiled_pages_decline_aliased_external_resources_without_mutation():
    core = NandCore(resolve_physical_config("hbf_ocp_0_7"), capture_details=False)
    before = deepcopy(vars(core))
    assert core.submit_read_pages([0, 4096], 0., link_program=(
        ("bad-link", "read", (("nand:host:0:read", 1., 4096),)),)) is None
    assert vars(core) == before
    assert core.submit_read_pages([0, 4096], 0., link_program=(
        ("hop-a", "read", (("shared-wire", .1, 4096),)),
        ("hop-b", "read", (("shared-wire", .3, 4096),)))) is None
    assert vars(core) == before


def test_compiled_zero_array_latency_still_materializes_the_same_calendar():
    config = replace(resolve_physical_config("hbf_ocp_0_7"), page_read_ns=0.)
    actual, expected = NandCore(config, capture_details=False), NandCore(config, capture_details=False)
    addresses = [i * 4096 for i in range(128)]
    last = {}
    for index, address in enumerate(addresses):
        result = _reference_submit(expected, AccessRequest(str(index), "read", address, 4096, 0.))
        last.update(result.counters["resource_last_intervals"])
    _, bill, _ = actual.submit_read_pages(addresses, 0.)
    assert bill.counters["resource_last_intervals"] == last
    assert vars(actual) == vars(expected)


@pytest.mark.parametrize("independent", [False, True])
def test_compiled_programs_preserve_incoming_links_erases_invalidation_and_page_order(independent):
    config = NandConfig(kind="HBF", protocol_mode="ocp_hbf_0_7", page_bytes=4096,
        host_granularity_bytes=64, channels=2, host_channels=2, planes_per_lun=2,
        planes_independent=independent, parallel_units=1, pages_per_block=4, blocks_per_plane=32,
        host_full_duplex=True, read_cache_pages_per_bank=2, max_outstanding_requests=3,
        page_read_ns=100., page_program_ns=1000., block_erase_ns=2000.)
    program = (("incoming", "gpu->hbf", (("write-wire", 4096 / 31., 4096),
                ("write-credit", 4096 / 31. + 79., 4096))),)
    timeline = ResourceTimeline(lane_available={"write-wire": [0.], "write-credit": [10., 30., 0.]})
    expected = NandCore(config, deepcopy(timeline), capture_details=False)
    actual = NandCore(config, deepcopy(timeline), capture_details=False)
    for core in (expected, actual):
        for index in range(12):
            _reference_submit(core, AccessRequest(f"warm-{index}", "read", index * 4096, 4096, 0.))
    addresses = [i * 4096 for i in range(64)] + [i * 4096 for i in range(8)]
    arrivals = [(i % 7) * 20.5 for i in range(len(addresses))]
    completions, expected_pages = [], []
    for index, (address, arrival) in enumerate(zip(addresses, arrivals)):
        link_end, _ = _links(expected.timeline, program, arrival)
        result = _reference_submit(expected, AccessRequest(str(index), "write", address, 4096, link_end))
        expected_pages.append(result)
        completions.append(result.completion_ns)
    completed, bill, links = actual.submit_write_pages(addresses, arrivals, link_program=program)
    assert completed.tolist() == completions
    assert vars(actual) == vars(expected)
    assert bill.operation is Operation.WRITE
    assert bill.arrival_ns == min(row.arrival_ns for row in expected_pages)
    for name in ("resource_busy_ns", "resource_bytes"):
        assert bill.counters[name] == _sum_metric(expected_pages, name)
    for name in ("pages_programmed", "erase_operations", "host_transfer_bytes", "internal_transfer_bytes",
                 "host_admission_samples", "queue_wait_ns"):
        expected_total = 0
        for page in expected_pages:
            expected_total += page.counters[name]
        assert bill.counters[name] == expected_total
    assert bill.counters["read_cache_hits"] == bill.counters["read_cache_misses"] == 0
    assert links[0][1].counters["physical_operation_count"] == len(addresses)


def test_compiled_program_order_rejection_keeps_the_entire_live_state():
    core = NandCore(resolve_physical_config("hbf_ocp_0_7"), capture_details=False)
    before = deepcopy(vars(core))
    with pytest.raises(ValueError, match="write order violation"):
        core.submit_write_pages([64 * 4096, 65 * 4096], [0., 10.])
    assert vars(core) == before


def test_compiled_reads_accept_independent_ready_times_for_clean_dynamic_pages():
    config = resolve_physical_config("hbf_ocp_0_7")
    actual, expected = NandCore(config, capture_details=False), NandCore(config, capture_details=False)
    addresses = [i * 4096 for i in range(128)]
    arrivals = [(i % 11) * 1000. for i in range(128)]
    results = [_reference_submit(expected, AccessRequest(str(i), "read", address, 4096, arrival))
               for i, (address, arrival) in enumerate(zip(addresses, arrivals))]
    completions, bill, _ = actual.submit_read_pages(addresses, arrivals)
    assert completions.tolist() == [row.completion_ns for row in results]
    assert vars(actual) == vars(expected)
    assert bill.arrival_ns == min(arrivals)


def test_compiled_geometry_cache_tracks_mutable_resource_prefixes():
    config = replace(resolve_physical_config("hbf_ocp_0_7"), metadata={})
    actual, expected = NandCore(config, capture_details=False), NandCore(config, capture_details=False)
    for offset in (0, 128):
        if offset:
            config.metadata.update(array_resource_prefix="new-array", channel_resource_prefix="new-channel",
                                   host_resource_id="new-host", parallel_resource_prefix="new-parallel")
        addresses = [(offset + i) * 4096 for i in range(128)]
        results = [_reference_submit(expected, AccessRequest(str(i), "read", address, 4096, 0.))
                   for i, address in enumerate(addresses)]
        completions, _, _ = actual.submit_read_pages(addresses, 0.)
        assert completions.tolist() == [row.completion_ns for row in results]
        assert vars(actual) == vars(expected)
