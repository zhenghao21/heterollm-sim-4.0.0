"""Sparse undo covers both PCs, maintenance, wrapping and accepted prefixes."""
import numpy as np
import pytest

from heterollm_sim import _compiled_hbm
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration, assert_core_state_equal


def pair(mapping):
    config = configuration(channels=8, address_mapping=mapping,
        interleave_bytes=128 if mapping == "bank_interleaved" else None,
        refresh_policy="all_bank", refresh_interval_ns=400.0, refresh_cycle_ns=40.0)
    contexts = [PhysicalRuntimeContext(config, "hbm", capture_details=False) for _ in range(2)]
    cores = [context.runtimes["hbm"].core for context in contexts]
    lane_span = config.row_bytes * config.bank_count if mapping == "row_local" else config.interleave_bytes
    warm = AccessRequest("warm", "read", 0, lane_span * config.lane_count)
    assert cores[0].submit(warm) == cores[1].submit(warm)
    cores[1]._hbm_numeric_state.request_selection = lambda request: None
    return config, lane_span, contexts, cores


def assert_arrays_equal(cores):
    for actual, expected in zip(cores[0]._hbm_numeric_state.arrays, cores[1]._hbm_numeric_state.arrays):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def assert_published_equal(cores):
    # Exposing the public bank mapping materializes resident private state.
    assert cores[0]._banks == cores[1]._banks
    assert_core_state_equal(*cores)


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
def test_sparse_checkpoints_match_full_arrays_and_public_metrics(mapping):
    config, span, contexts, cores = pair(mapping)
    requests = (
        AccessRequest("one", "write", span + 3, 65, 1500.),
        AccessRequest("wrap", "read", span * config.lane_count - 31, 65, 2000.),
        AccessRequest("sibling-maintenance", "read", 2 * span + 3, 65, 5000.),
        AccessRequest("dense", "write", 0, span * config.lane_count, 6000.),
    )
    for request in requests:
        assert cores[0].submit(request) == cores[1].submit(request)
        assert_arrays_equal(cores)
    assert_published_equal(cores)


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
@pytest.mark.parametrize("decline", (False, True))
def test_sparse_failure_keeps_prefix_and_restores_global_touched(mapping, decline, monkeypatch):
    config, span, contexts, cores = pair(mapping)
    requests = (AccessRequest("prefix", "read", span + 3, 65, 1500.),
                AccessRequest("stop", "write", span * config.lane_count - 31, 65, 3000.))
    original = _compiled_hbm._HBMNumericState.execute
    def stop_after_mutation(state, request):
        result = original(state, request)
        if request.request_id == "stop":
            if decline:
                return None
            raise RuntimeError("sparse request stopped")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm._HBMNumericState, "execute", stop_after_mutation)
        if decline:
            assert cores[0].submit_many(requests) == cores[1].submit_many(requests)
        else:
            for core in cores:
                with pytest.raises(RuntimeError, match="sparse request stopped"):
                    core.submit_many(requests)
            assert_arrays_equal(cores)
    assert_published_equal(cores)


def test_sparse_checkpoint_copies_fewer_bank_rows_and_is_independent():
    config, span, contexts, cores = pair("bank_interleaved")
    packed = cores[0]._hbm_numeric_state
    request = AccessRequest("one", "write", span + 3, 65)
    before = tuple(array.copy() for array in packed.arrays)
    selection, saved = packed.checkpoint(request)
    assert selection is not None
    assert saved[1].shape[0] == 2 * config.bank_count
    assert sum(array.nbytes for array in saved) < sum(array.nbytes for array in before) / 3
    for array, index in zip(packed.arrays, selection):
        array[index] = 0
    packed.restore_checkpoint((selection, saved))
    for actual, expected in zip(packed.arrays, before):
        assert actual.tobytes() == expected.tobytes()


def test_sparse_nonfinite_result_declines_and_restores_every_touched_array(monkeypatch):
    config, span, contexts, cores = pair("bank_interleaved")
    packed = cores[0]._hbm_numeric_state
    request = AccessRequest("overflow", "write", span + 3, 65, 5000.)
    before = tuple(array.copy() for array in packed.arrays)
    original = packed.run_numeric
    def nonfinite(*args):
        result = original(*args)
        packed.arrays[1][0, 0] = np.inf
        return result
    monkeypatch.setattr(packed, "run_numeric", nonfinite)
    assert _compiled_hbm.execute_hbm_numeric(cores[0], request) is None
    for actual, expected in zip(packed.arrays, before):
        assert actual.tobytes() == expected.tobytes()
