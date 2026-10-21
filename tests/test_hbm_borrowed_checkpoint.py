"""A first dense request can share immutable outer controller undo rows."""

import pytest

from heterollm_sim import _compiled_hbm
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def prepared():
    runtime = PhysicalRuntimeContext(configuration(channels=8), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    return runtime, core


def image(core):
    packed = core._hbm_numeric_state
    if packed is None:
        # A normal reference fallback publishes arrays into Python objects.
        # Repack those objects without submitting another physical access.
        packed = _compiled_hbm._pack_hbm_numeric(core, AccessRequest("inspect", "read", 0, 32))
    return tuple(array.tobytes() for array in packed.arrays)


def test_first_dense_checkpoint_borrows_only_unchanged_controller_rows():
    _, core = prepared()
    before = image(core)
    token = core.begin_transaction()
    request = AccessRequest("weight", "read", 65536, 4096, 5000.)
    packed = _compiled_hbm._pack_hbm_numeric(core, request)
    _, checkpoint = packed.checkpoint(request)
    assert isinstance(checkpoint, _compiled_hbm._BorrowedControllerCheckpoint)
    for index, array in enumerate(checkpoint):
        if index in (8, 9, 10, 11):
            assert array is not token.numeric.whole[index]
        else:
            assert array is token.numeric.whole[index]
    core.submit(request)
    _, following = packed.checkpoint(request)
    assert not isinstance(following, _compiled_hbm._BorrowedControllerCheckpoint)
    core.rollback_transaction(token)
    assert image(core) == before


@pytest.mark.parametrize("failure", ("exception", "decline"))
def test_failed_or_declined_request_does_not_make_undo_rows_live(failure, monkeypatch):
    runtime, core = prepared()
    before = runtime.snapshot()
    core.submit(AccessRequest("restore-residency", "read", 0, 4096))
    transaction_start = image(core)
    touches_at_start = set(runtime.timeline._touched)
    token = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
    original = _compiled_hbm._HBMNumericState.execute
    def reject(packed, request):
        result = original(packed, request)
        if request.request_id == "rejected":
            if failure == "decline":
                return None
            raise RuntimeError("injected dense request failure")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm._HBMNumericState, "execute", reject)
        request = AccessRequest("rejected", "write", 65536, 4096, 1000.)
        if failure == "exception":
            with pytest.raises(RuntimeError, match="dense request failure"):
                core.submit(request)
        else:
            core.submit(request)
    # A retry occurs while the outer token remains alive. It must not write
    # into either the borrowed backup or the outer transaction start image.
    core.submit(AccessRequest("retry", "write", 131072, 4096, 2000.))
    runtime.restore(token)
    if core._hbm_numeric_state is None:
        # Reference fallback discards the numeric recorder bitmap. Repacking
        # initializes that derived scratch to zero; the real touched resource
        # recorder and every controller/resource column still must restore.
        restored = image(core)
        assert restored[:11] + restored[12:] == transaction_start[:11] + transaction_start[12:]
        assert runtime.timeline._touched == touches_at_start
    else:
        assert image(core) == transaction_start
    # Restoring the original public point remains independent and reusable.
    runtime.restore(before)
    assert runtime.snapshot() == before


def test_nested_dense_rollback_keeps_both_immutable_start_images():
    runtime, core = prepared()
    before = image(core)
    outer = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
    core.submit(AccessRequest("outer", "write", 65536, 4096, 1000.))
    middle = image(core)
    inner = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
    core.submit(AccessRequest("inner", "write", 131072, 4096, 2000.))
    runtime.restore(inner)
    assert image(core) == middle
    core.submit(AccessRequest("retry", "read", 196608, 4096, 3000.))
    runtime.restore(outer)
    assert image(core) == before


def test_first_sparse_request_borrows_both_pc_controller_rows_but_not_resources():
    runtime, core = prepared()
    before = image(core)
    token = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
    request = AccessRequest("sparse", "write", 0, 32, 2000.)
    packed = _compiled_hbm._pack_hbm_numeric(core, request)
    selection, saved = packed.checkpoint(request)
    assert selection is not None and isinstance(saved, _compiled_hbm._BorrowedControllerCheckpoint)
    channel = selection[6].start
    outer = core._hbm_transactions[-1]
    assert outer.numeric.whole is None
    for index in range(len(saved)):
        if index not in (8, 9, 10, 11):
            assert saved[index] is outer.numeric.channels[channel][1][index]
    core.submit(request)
    # A later request sees the new controller state and must own its image.
    _, later = packed.checkpoint(request)
    assert not isinstance(later, _compiled_hbm._BorrowedControllerCheckpoint)
    runtime.restore(token)
    assert image(core) == before


@pytest.mark.parametrize("failure", ["exception", "decline"])
def test_sparse_borrowed_failure_retry_and_nested_rollback_are_independent(monkeypatch, failure):
    runtime, core = prepared()
    before = image(core)
    before_touched = set(runtime.timeline._touched)
    outer = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
    core.submit(AccessRequest("outer", "read", 0, 32, 1000.))
    middle = image(core)
    middle_touched = set(runtime.timeline._touched)
    inner = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
    original = _compiled_hbm._HBMNumericState.execute
    def execute(packed, request):
        result = original(packed, request)
        if request.request_id == "bad":
            if failure == "exception":
                raise RuntimeError("sparse checkpoint failure")
            return None
        return result
    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm._HBMNumericState, "execute", execute)
        request = AccessRequest("bad", "write", 32, 32, 2000.)
        if failure == "exception":
            with pytest.raises(RuntimeError, match="sparse checkpoint failure"):
                core.submit(request)
        else:
            core.submit(request)
    core.submit(AccessRequest("retry", "read", 64, 32, 3000.))
    runtime.restore(inner)
    restored = image(core)
    # Scalar decline discards only the derived numeric touched scratch; the
    # real timeline recorder and every physical controller value must agree.
    assert restored[:11] + restored[12:] == middle[:11] + middle[12:]
    assert runtime.timeline._touched == middle_touched
    core.submit(AccessRequest("after", "write", 96, 32, 4000.))
    runtime.restore(outer)
    restored = image(core)
    assert restored[:11] + restored[12:] == before[:11] + before[12:]
    assert runtime.timeline._touched == before_touched
