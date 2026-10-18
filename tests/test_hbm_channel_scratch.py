"""Channel workspaces never become inputs to later physical requests."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from heterollm_sim import _hbm_numeric
from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def same_bits(expected, actual):
    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert expected.tobytes() == actual.tobytes()


@pytest.mark.parametrize("mapping,interleave", [("bank_interleaved", 32),
                                               ("bank_interleaved", 128),
                                               ("row_local", None)])
def test_reused_poisoned_scratch_matches_fresh_workspaces(mapping, interleave):
    config = configuration(channels=5, address_mapping=mapping, interleave_bytes=interleave,
                           rows_per_bank=128, activated_row_msb=6)
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False),
                              AccessRequest("pack", "read", 0, 1024))
    initial = tuple(array.copy() for array in packed.arrays)
    for index, (first, stop, arrival, read) in enumerate((
            (0, 8192, 0., True), (3, 37, 100., False),
            (257, 8321, 200000., False), (0, 4097, 400000., True))):
        expected_arrays = tuple(array.copy() for array in initial)
        expected = _hbm_numeric._run_hbm_numeric(
            first, stop, arrival, read, packed.g, packed.t, expected_arrays,
            _hbm_numeric._new_channel_scratch(packed.g, expected_arrays))
        actual_arrays = tuple(array.copy() for array in initial)
        lease = getattr(_hbm_numeric._CHANNEL_SCRATCH, "lease", None)
        if lease is not None:
            for array in lease[1]:
                array.fill(np.nan if array.dtype.kind == "f" else -37)
        actual = _hbm_numeric.run_hbm_numeric(
            first, stop, arrival, read, packed.g, packed.t, *actual_arrays)
        for left, right in zip(expected, actual, strict=True):
            same_bits(np.asarray(left), np.asarray(right))
        for left, right in zip(expected_arrays, actual_arrays, strict=True):
            same_bits(left, right)
        initial = expected_arrays


def test_empty_compile_probe_never_reads_or_changes_workspaces():
    packed = _pack_hbm_numeric(DramCore(configuration(), capture_details=False),
                              AccessRequest("pack", "read", 0, 1024))
    scratch = _hbm_numeric._new_channel_scratch(packed.g, packed.arrays)
    for array in scratch:
        array.fill(np.nan if array.dtype.kind == "f" else -37)
    saved = tuple(array.copy() for array in scratch)
    result = _hbm_numeric._run_hbm_channel_range(
        17, 17, 12.5, True, packed.g, packed.t, *packed.arrays, scratch, 0, 0)
    assert result[:4] == (12.5, 0, 0, 0)
    assert np.array_equal(result[4], np.zeros(7))
    for before, after in zip(saved, scratch, strict=True):
        same_bits(before, after)


def test_reentrant_and_failed_calls_release_only_their_own_workspace(monkeypatch):
    packed = _pack_hbm_numeric(DramCore(configuration(), capture_details=False),
                              AccessRequest("pack", "read", 0, 1024))
    entered = []

    def execute(first, stop, arrival, read, g, t, arrays, scratch):
        entered.append(scratch)
        if first == 0:
            with pytest.raises(RuntimeError, match="nested"):
                _hbm_numeric.run_hbm_numeric(1, stop, arrival, read, g, t, *arrays)
            assert _hbm_numeric._CHANNEL_SCRATCH.lease[2]
        else:
            raise RuntimeError("nested")
        return 1

    monkeypatch.setattr(_hbm_numeric, "_run_hbm_numeric", execute)
    assert _hbm_numeric.run_hbm_numeric(0, 3, 0., True, packed.g, packed.t, *packed.arrays) == 1
    assert entered[0] is not entered[1]
    assert not _hbm_numeric._CHANNEL_SCRATCH.lease[2]
    assert _hbm_numeric.run_hbm_numeric(0, 3, 0., True, packed.g, packed.t, *packed.arrays) == 1
    assert entered[0] is entered[2]


def test_concurrent_callers_have_distinct_scratch(monkeypatch):
    import threading

    packed = _pack_hbm_numeric(DramCore(configuration(), capture_details=False),
                              AccessRequest("pack", "read", 0, 1024))
    barrier = threading.Barrier(2)

    def execute(first, stop, arrival, read, g, t, arrays, scratch):
        barrier.wait(timeout=5.)
        return scratch

    monkeypatch.setattr(_hbm_numeric, "_run_hbm_numeric", execute)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = [workers.submit(_hbm_numeric.run_hbm_numeric,
                                  0, 3, 0., True, packed.g, packed.t, *packed.arrays)
                   for _ in range(2)]
        assert results[0].result() is not results[1].result()
