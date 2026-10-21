"""Transaction copies follow written channels, with exact nested rollback."""

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def prepared(mapping="bank_interleaved"):
    config = configuration(channels=16, address_mapping=mapping,
        interleave_bytes=128 if mapping == "bank_interleaved" else None,
        refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.)
    runtime = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    span = (config.interleave_bytes if mapping == "bank_interleaved"
            else config.row_bytes * config.bank_count)
    core.submit(AccessRequest("warm", "read", 0, span * config.lane_count))
    return core, span


def image(core):
    return tuple(array.tobytes() for array in core._hbm_numeric_state.arrays)


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
def test_begin_is_lazy_and_repeated_sparse_channel_is_saved_once(mapping):
    core, span = prepared(mapping)
    before = image(core)
    token = core.begin_transaction()
    undo = token.numeric
    assert undo.whole is None and undo.touched is None and not undo.channels
    core.submit(AccessRequest("sparse", "write", span + 3, 65, 1500.))
    assert undo.whole is None
    assert set(undo.channels) == {0}
    saved_size = sum(array.nbytes for _, values in undo.channels.values()
                     for array in values if array is not None) + undo.touched.nbytes
    assert saved_size < sum(array.nbytes for array in core._hbm_numeric_state.arrays) / 4
    old_saved = undo.channels[0]
    core.submit(AccessRequest("same-channel", "read", 3, 65, 5000.))
    assert undo.channels[0] is old_saved
    assert set(undo.channels) == {0}
    core.rollback_transaction(token)
    assert image(core) == before


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
@pytest.mark.parametrize("dense", (False, True))
def test_nested_disjoint_wrapped_and_dense_requests_restore_original_arrays(mapping, dense):
    core, span = prepared(mapping)
    before = image(core)
    outer = core.begin_transaction()
    core.submit(AccessRequest("outer", "write", 2 * span + 3, 65, 1500.))
    middle = image(core)
    inner = core.begin_transaction()
    core.submit(AccessRequest("wrap", "read", core.config.lane_count * span - 31, 65, 3000.))
    core.submit(AccessRequest("disjoint", "write", 16 * span + 3, 65, 5000.))
    if dense:
        core.submit(AccessRequest("dense", "write", 0, core.config.lane_count * span, 6000.))
        assert inner.numeric.whole is not None
        assert outer.numeric.whole is not None
    core.rollback_transaction(inner)
    assert image(core) == middle
    core.rollback_transaction(outer)
    assert image(core) == before


def test_sparse_journal_saves_resources_before_external_calendar_refresh():
    core, span = prepared()
    packed = core._hbm_numeric_state
    before = image(core)
    token = core.begin_transaction()
    resource = packed.ids[0]
    core.timeline.ready_ns[resource] += 2000.
    core.timeline.lane_available[resource][0] += 2000.
    core.submit(AccessRequest("external-calendar", "read", 0, 65, 5000.))
    core.rollback_transaction(token)
    assert image(core) == before


def test_no_request_rollback_keeps_arrays_and_numeric_identity():
    core, _ = prepared()
    packed = core._hbm_numeric_state
    arrays = packed.arrays
    before = image(core)
    token = core.begin_transaction()
    core.rollback_transaction(token)
    assert core._hbm_numeric_state is packed
    assert packed.arrays is arrays
    assert image(core) == before


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
def test_nested_same_channel_undo_keeps_outer_original(mapping):
    core, span = prepared(mapping)
    before = image(core)
    outer = core.begin_transaction()
    core.submit(AccessRequest("outer", "read", span + 3, 65, 1500.))
    middle = image(core)
    outer_saved = outer.numeric.channels[0]
    inner = core.begin_transaction()
    core.submit(AccessRequest("inner", "write", 3, 65, 3000.))
    assert outer.numeric.channels[0] is outer_saved
    core.rollback_transaction(inner)
    assert image(core) == middle
    core.rollback_transaction(outer)
    assert image(core) == before
