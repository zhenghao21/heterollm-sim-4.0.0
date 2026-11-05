"""OCP command credits belong to each independent host channel."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_presets import resolve_physical_config


PAGE = 4096


def settings(**changes):
    values = dict(channels=2, host_channels=2, planes_per_lun=4,
                  blocks_per_plane=2, pages_per_block=4,
                  capacity_bytes=2 * 4 * 2 * 4 * PAGE,
                  host_bandwidth_gb_s=1000., host_read_bandwidth_gb_s=1000.,
                  host_write_bandwidth_gb_s=1000., internal_bandwidth_gb_s=1000.,
                  page_read_ns=100., page_program_ns=1000., block_erase_ns=2000.,
                  max_outstanding_requests=2, max_outstanding_write_requests=1)
    values.update(changes)
    return replace(resolve_physical_config("hbf_ocp_0_7"), **values)


def request(name, op, page=0, count=1, arrival=0.):
    return AccessRequest(name, op, page * PAGE, count * PAGE, arrival)


@pytest.mark.parametrize("details", [False, True])
def test_full_host_does_not_delay_a_different_host(details):
    core = NandCore(settings(max_outstanding_requests=1), capture_details=details)
    first = core.submit(request("first", "read"))
    blocked = core.submit(request("same-host", "read", 2))
    other = core.submit(request("other-host", "read", 1))
    assert blocked.queue_wait_ns == first.completion_ns
    assert other.queue_wait_ns == 0.
    assert other.completion_ns < blocked.completion_ns
    assert core._host_acceptance_ns == {0: first.completion_ns, 1: 0.}
    assert core._inflight == core._inflight_operations == []
    assert core._acceptance_ns == 0.


@pytest.mark.parametrize("details", [False, True])
def test_write_subcap_waits_for_write_and_keeps_shared_read_write_total(details):
    core = NandCore(settings(max_outstanding_requests=3), capture_details=details)
    first = core.submit(request("program", "write"))
    read = core.submit(request("independent-read", "read", 4))
    write = core.submit(request("blocked-program", "write", 2))
    assert read.completion_ns < first.completion_ns
    assert read.queue_wait_ns == 0.
    assert write.queue_wait_ns == first.completion_ns
    assert write.counters["host_write_outstanding_at_admission_peak"] == 1
    assert read.counters["host_outstanding_at_admission_peak"] == 2
    assert read.counters["host_read_outstanding_at_admission_peak"] == 1
    assert read.counters["host_write_outstanding_at_admission_peak"] == 1
    combined = NandCore(settings(max_outstanding_requests=1), capture_details=details)
    outstanding = combined.submit(request("write", "write"))
    next_read = combined.submit(request("read", "read", 4))
    assert next_read.queue_wait_ns == outstanding.completion_ns


@pytest.mark.parametrize("details", [False, True])
def test_preset_write_limit_is_64_per_host_and_reads_use_remaining_total_credits(details):
    config = resolve_physical_config("hbf_ocp_0_7")
    core = NandCore(config, capture_details=details)
    writes = [core.submit(request(str(index), "write", config.channels * index))
              for index in range(64)]
    assert all(row.queue_wait_ns == 0. for row in writes)
    assert len(core._host_inflight[0]) == 64
    read = core.submit(request("read-under-writes", "read", 0))
    assert read.queue_wait_ns == 0.
    assert read.counters["host_outstanding_at_admission_peak"] == 65
    assert read.counters["host_write_outstanding_at_admission_peak"] == 64
    next_write = core.submit(request("write-over-cap", "write", config.channels * 64))
    assert next_write.queue_wait_ns == min(row.completion_ns for row in writes)
    # Parallel page-0 programs can finish at the same instant, returning more
    # than one credit together; the next population remains below the cap.
    assert next_write.counters["host_write_outstanding_at_admission_peak"] <= 64


def test_bulk_retains_each_page_observation_and_bounded_trace_details():
    core = NandCore(settings(detailed_observability=True, max_expanded_segments=2))
    result = core.submit(request("observed-bulk", "read", count=4))
    assert len(result.counters["nand_observations"]["events"]) == 4
    assert result.counters["details_truncated"]
    assert result.counters["intervals_truncated"]
    assert not result.mapping and not result.stages
    for key in ("resource_intervals", "resource_interval_payloads"):
        assert sum(len(rows) for rows in result.counters[key].values()) <= 2


def test_explicit_device_clock_advance_applies_to_each_host():
    core = NandCore(settings(), capture_details=False)
    core.flush_write_buffer(now_ns=1000.)
    for page in (0, 1):
        result = core.submit(request("read", "read", page))
        assert result.queue_wait_ns == 1000.


@pytest.mark.parametrize("transactional", [False, True])
def test_runtime_snapshot_detaches_host_command_lists_and_restores(transactional):
    runtime = PhysicalRuntimeContext(config=settings(), physical_owner="hbf", capture_details=False)
    core = runtime.runtimes["hbf"].core
    core.submit(request("seed", "read"))
    saved = runtime.snapshot(transactional=transactional)
    original = deepcopy(core._host_inflight)
    saved_host = saved["runtime_state"]["hbf"]["core"]["_host_inflight"]
    assert saved_host[0] is not core._host_inflight[0]
    core.submit(request("other", "read", 1))
    core._host_inflight[0].append((1.e9, Operation.WRITE))
    assert saved_host == original
    runtime.restore(saved)
    assert core._host_inflight == original
    assert core._host_acceptance_ns == {0: 0.}
    assert core._host_inflight[0] is not saved_host[0]
