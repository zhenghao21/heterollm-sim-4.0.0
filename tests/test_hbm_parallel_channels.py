"""Channel workers retain serial commands, resource clocks and rollback."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import threading

import numpy as np
import pytest

from heterollm_sim import _hbm_numeric
from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


@pytest.mark.parametrize("channels,mapping,interleave,refresh,clock", [
    (3, "bank_interleaved", 32, False, 1.),
    (5, "bank_interleaved", 128, False, 1.),
    (9, "row_local", None, False, 1.),
    (3, "row_local", None, True, 1.),
    (5, "bank_interleaved", 64, False, 1.28),
])
def test_parallel_request_matches_serial_all_live_arrays(channels, mapping, interleave, refresh, clock, monkeypatch):
    config = configuration(channels=channels, address_mapping=mapping, interleave_bytes=interleave,
        command_clock_ns=clock, data_rate_mt_s=4000. / clock,
        read_latency_ns=8 * clock, write_latency_ns=4 * clock,
        column_same_group_ns=4 * clock, column_different_group_ns=2 * clock,
        column_different_sid_ns=3 * clock, precharge_to_precharge_ns=2 * clock,
        activate_same_group_ns=5 * clock, activate_different_group_ns=2 * clock,
        rows_per_bank=128, activated_row_msb=6, refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=3900. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    requests = [AccessRequest("read", "read", 193, 2 ** 21 + 37),
                AccessRequest("write", "write", 8193, 2 ** 21 + 115, 100.),
                AccessRequest("after-maintenance", "read", 31, 2 ** 21 + 71, 200000.)]
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False), requests[0])
    before = tuple(array.copy() for array in packed.arrays)
    for request in requests:
        packed.arrays = tuple(array.copy() for array in before)
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 1)
        expected = packed.execute(request)
        expected_arrays = tuple(array.copy() for array in packed.arrays)
        packed.arrays = tuple(array.copy() for array in before)
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 8)
        actual = packed.execute(request)
        assert replace(actual, counters={}) == replace(expected, counters={})
        # Independent-channel results merge in a fixed channel order. Only
        # these summary sums may differ in the last float bits; command and
        # resource clocks, physical counts and all live arrays remain exact.
        for key, value in expected.counters.items():
            if key in {"row_command_bus_busy_ns", "column_command_bus_busy_ns"}:
                assert math.isclose(actual.counters[key], value, rel_tol=1e-10, abs_tol=1e-8)
            else:
                assert actual.counters[key] == value
        for actual, saved in zip(packed.arrays, expected_arrays):
            assert np.array_equal(actual.view(np.uint8), saved.view(np.uint8))
        before = expected_arrays


def test_parallel_failure_waits_for_every_worker_before_runtime_rollback(monkeypatch):
    config = configuration(channels=3, rows_per_bank=128, activated_row_msb=6)
    runtime = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    packed = core._hbm_numeric_state
    expected = tuple(array.copy() for array in packed.arrays)
    original = _hbm_numeric._run_hbm_channel_range
    released = threading.Event()
    waiting = threading.Event()
    lock = threading.Lock()
    started = []
    completed = []

    def delayed_worker(*args):
        first_channel, stop_channel = args[-2:]
        if first_channel == stop_channel:
            return original(*args)
        if first_channel == 0:
            raise RuntimeError("injected channel failure")
        with lock:
            started.append(first_channel)
            if len(started) == 2:
                waiting.set()
        assert released.wait(5.), "test did not release channel worker"
        result = original(*args)
        with lock:
            completed.append(first_channel)
        return result

    monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 8)
    monkeypatch.setattr(_hbm_numeric, "_run_hbm_channel_range", delayed_worker)
    transaction = runtime.begin_transaction()
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(core.submit, AccessRequest("failed", "write", 193, 2 ** 21 + 37))
        try:
            assert waiting.wait(5.), "channel workers did not start"
            assert not future.done()
        finally:
            released.set()
        with pytest.raises(RuntimeError, match="injected channel failure"):
            future.result(timeout=10.)
    assert sorted(completed) == [1, 2]
    runtime.restore(transaction)
    for actual, saved in zip(core._hbm_numeric_state.arrays, expected):
        assert np.array_equal(actual.view(np.uint8), saved.view(np.uint8))


@pytest.mark.parametrize("failure", ["submit", "submit_after_enqueue", "wait"])
def test_dispatch_or_wait_failure_drains_already_started_workers(failure, monkeypatch):
    runtime = PhysicalRuntimeContext(configuration(channels=3, rows_per_bank=128, activated_row_msb=6),
                                     "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    expected = tuple(array.copy() for array in core._hbm_numeric_state.arrays)
    original = _hbm_numeric._run_hbm_channel_range
    original_wait = _hbm_numeric.wait
    release = threading.Event()
    failed = threading.Event()
    returned = threading.Event()
    started_any = threading.Event()
    started = []
    completed = []

    def delayed_worker(*args):
        if args[-2] == args[-1]:
            return original(*args)
        started.append(args[-2])
        started_any.set()
        assert release.wait(5.), "test did not release the accepted worker"
        result = original(*args)
        completed.append(args[-2])
        return result

    with ThreadPoolExecutor(max_workers=3) as workers:
        class FailingDispatcher:
            submitted = 0

            def submit(self, *args):
                self.submitted += 1
                if failure in {"submit", "submit_after_enqueue"} and self.submitted == 2:
                    assert started_any.wait(5.), "first worker did not enter its kernel"
                    if failure == "submit_after_enqueue":
                        workers.submit(*args)  # Executor accepts the job but loses its Future.
                    failed.set()
                    raise RuntimeError("injected executor submission failure")
                return workers.submit(*args)

        def interrupted_wait(jobs):
            if not failed.is_set():
                assert started_any.wait(5.), "no accepted worker entered its kernel"
                failed.set()
                raise KeyboardInterrupt("injected wait interruption")
            return original_wait(jobs)

        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 8)
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_EXECUTOR", FailingDispatcher())
        monkeypatch.setattr(_hbm_numeric, "_run_hbm_channel_range", delayed_worker)
        if failure == "wait":
            monkeypatch.setattr(_hbm_numeric, "wait", interrupted_wait)

        def submit():
            try:
                return core.submit(AccessRequest("failed", "write", 193, 2 ** 21 + 37))
            finally:
                returned.set()

        transaction = runtime.begin_transaction()
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(submit)
            try:
                assert failed.wait(5.), "injected failure was not reached"
                assert not returned.wait(.05), "request escaped while an accepted worker could still mutate arrays"
            finally:
                release.set()
            with pytest.raises(KeyboardInterrupt if failure == "wait" else RuntimeError):
                future.result(timeout=10.)
        assert started
        assert sorted(completed) == sorted(started)
        runtime.restore(transaction)
        for actual, saved in zip(core._hbm_numeric_state.arrays, expected):
            assert np.array_equal(actual.view(np.uint8), saved.view(np.uint8))


def test_address_stripe_over_int64_declines_before_compiled_traversal():
    from heterollm_sim._hbm_control import _Controller

    config = configuration(channels=1024, rows_per_bank=2 ** 50, activated_row_msb=49,
                           interleave_bytes=2 ** 60, capacity_bytes=2 ** 30)
    core = DramCore(config, capture_details=False)
    reference = DramCore(config, capture_details=False)
    request = AccessRequest("small-valid-read", "read", 0, 32)
    assert _pack_hbm_numeric(core, request) is None
    reference.timeline.metrics_snapshot()
    core.timeline.metrics_snapshot()
    expected = _Controller(reference, request).execute()
    from heterollm_sim._hbm_control import execute_hbm
    assert execute_hbm(core, request) == expected
    assert core._banks == reference._banks
    assert core._hbm_control_state == reference._hbm_control_state
