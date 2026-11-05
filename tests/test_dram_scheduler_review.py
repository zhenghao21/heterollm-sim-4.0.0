"""Regression checks for scheduled physical metadata and tiny HBM execution."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_dram_controller_policies import config as analytical_config
from test_hbm_compiled_control import configuration


def _task(config, addresses, ident):
    return TaskSpec(ident, "request", ident, TaskCategory.MEMORY,
        metadata={"physical_owner": "hbm", "physical_memory_config": config,
            "memory_accesses": tuple({"operation": "read", "address": address, "byte_count": 32}
                                     for address in addresses)})


@pytest.mark.parametrize("details", [False, True])
@pytest.mark.parametrize("hbm", [False, True])
def test_reordered_results_keep_latest_global_and_owner_resource_intervals(details, hbm):
    config = (configuration(four_activate_window_ns=31.)
              if hbm else replace(analytical_config(), scheduler="row_hit_first"))
    runtime = PhysicalRuntimeContext(capture_details=details)
    resolve_physical_task(_task(config, (0,), "warm"), runtime, 0.)
    row_span = config.lane_count * config.bank_count * config.row_bytes
    result = resolve_physical_task(_task(config, (row_span, 32), "reordered"), runtime, 100.)
    owner = result.metadata["physical_execution_by_owner"]["hbm"]
    # FR-FCFS reorders bursts, not requests.
    assert owner["scheduler_reordered_requests"] == (0 if hbm else 1)
    observed = result.metadata["physical_resource_last_intervals"]
    assert observed == owner["resource_last_intervals"]
    assert observed == runtime.timeline.last_intervals


def _large_config(**changes):
    return configuration(channels=16, stacks=8, four_activate_window_ns=31.,
        refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.,
        scheduler_window=4, max_outstanding_requests=2, **changes)


def _without_execution_marker(results):
    return tuple(replace(row, counters={k: v for k, v in row.counters.items()
                                       if k != "compiled_hbm_float64"}) for row in results)


def test_tiny_reference_and_large_numeric_requests_share_identical_state():
    config = _large_config()
    direct, numeric = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    span = config.lane_count * config.bank_count * config.row_bytes
    requests = tuple(AccessRequest(str(i), op, address, count, arrival)
        for i, (op, address, count, arrival) in enumerate((
            ("read", 0, 64, 0.), ("write", span, 64, 0.), ("read", 0, 64, 0.),
            ("write", 64, 96, 0.), ("read", 65536, 4096, 1.),
            ("write", 0, 129, 10000.), ("read", 0, 1024, 10000.))))
    numeric._hbm_numeric_enabled = True
    expected = tuple(numeric.submit(request) for request in requests)
    numeric._hbm_numeric_enabled = False
    actual = tuple(direct.submit(request) for request in requests)
    assert all(row.counters["compiled_hbm_float64"] for row in expected)
    # Cold scalar execution on this 8192-bank device keeps up to 64 bursts
    # on the reference controller, including the final 1024-byte/32-burst read.
    assert [bool(row.counters.get("compiled_hbm_float64")) for row in actual] == [False] * 4 + [True, False, False]
    assert _without_execution_marker(actual) == _without_execution_marker(expected)
    following = AccessRequest("after", "read", 0, 4096, 20000.)
    assert direct.submit(following) == numeric.submit(following)


def test_tiny_calls_skip_packing_but_a_larger_batch_packs_once(monkeypatch):
    from heterollm_sim import _compiled_hbm
    core = DramCore(_large_config(), capture_details=False)
    calls = []
    original = _compiled_hbm._pack_hbm_numeric
    def observe(core, request, *, refresh_requests=None):
        calls.append((request.request_id, refresh_requests))
        return original(core, request, refresh_requests=refresh_requests)
    monkeypatch.setattr(_compiled_hbm, "_pack_hbm_numeric", observe)
    core.submit(AccessRequest("scalar", "read", 0, 64))
    core.submit_many((AccessRequest("tiny-1", "read", 64, 64), AccessRequest("tiny-2", "read", 128, 64)))
    assert calls == []
    requests = (AccessRequest("batch-1", "read", 4096, 2048),
                AccessRequest("batch-2", "read", 8192, 2048))
    result = core.submit_many(requests)
    assert calls == [("batch-1", requests)]
    assert all(row.counters["compiled_hbm_float64"] for row in result)
