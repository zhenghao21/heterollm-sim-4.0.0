"""Numeric controller undo journals preserve ordinary rollback semantics."""
import copy

import pytest

from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim._hbm_control import _Controller
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def state(core):
    return copy.deepcopy((core._banks, core._hbm_control_state))


def prepared_core():
    core = DramCore(configuration(), capture_details=False)
    execute_hbm_numeric(core, AccessRequest("warm", "read", 0, 4096))
    pc = next(iter(core._hbm_control_state["pcs"].values()))
    bank = next(iter(core._hbm_control_state["banks"].values()))
    core._hbm_control_state["extension"] = {"pc": pc, "history": pc["column"], "bank": bank}
    return core


def test_numeric_rollback_preserves_original_objects_and_extension_aliases():
    core = prepared_core()
    before = state(core)
    extension = core._hbm_control_state["extension"]
    token = core.begin_transaction()
    execute_hbm_numeric(core, AccessRequest("write", "write", 65536, 8192))
    execute_hbm_numeric(core, AccessRequest("read", "read", 0, 512))
    assert token.full is None
    assert token.objects
    core.rollback_transaction(token)
    assert state(core) == before
    assert extension is core._hbm_control_state["extension"]
    assert extension["pc"]["column"] is extension["history"]
    assert extension["bank"] in core._hbm_control_state["banks"].values()


@pytest.mark.parametrize("details", [False, True])
def test_reference_fallback_after_numeric_prefix_restores_transaction_start(details):
    core = prepared_core()
    core.capture_details = details
    before = state(core)
    token = core.begin_transaction()
    if not details:
        execute_hbm_numeric(core, AccessRequest("numeric-prefix", "write", 131072, 512))
        assert token.full is None
    _Controller(core, AccessRequest("reference", "write", 0, 512, 50000.)).execute()
    assert token.full is not None
    core.rollback_transaction(token)
    assert state(core) == before
    extension = core._hbm_control_state["extension"]
    assert extension["pc"]["column"] is extension["history"]


@pytest.mark.parametrize("fallback", [False, True])
def test_nested_transactions_keep_outer_rollback_after_inner_rollback(fallback):
    core = prepared_core()
    before = state(core)
    outer = core.begin_transaction()
    execute_hbm_numeric(core, AccessRequest("outer", "write", 65536, 512))
    middle = state(core)
    inner = core.begin_transaction()
    if fallback:
        _Controller(core, AccessRequest("inner", "read", 131072, 512)).execute()
    else:
        execute_hbm_numeric(core, AccessRequest("inner", "read", 131072, 512))
    core.rollback_transaction(inner)
    assert state(core) == middle
    execute_hbm_numeric(core, AccessRequest("after-inner", "write", 262144, 512))
    core.rollback_transaction(outer)
    assert state(core) == before


def test_empty_controller_rollback_removes_new_maps_and_commit_keeps_them():
    core = DramCore(configuration(), capture_details=False)
    before = state(core)
    token = core.begin_transaction()
    execute_hbm_numeric(core, AccessRequest("new", "write", 0, 4096))
    core.rollback_transaction(token)
    assert state(core) == before
    token = core.begin_transaction()
    execute_hbm_numeric(core, AccessRequest("committed", "read", 0, 4096))
    after = state(core)
    core.commit_transaction(token)
    assert state(core) == after
    with pytest.raises(ValueError, match="reverse begin order"):
        core.rollback_transaction(token)


@pytest.mark.parametrize("fallback", [False, True])
def test_runtime_rollback_restores_timeline_queues_and_controller_aliases(fallback, monkeypatch):
    from heterollm_sim import _compiled_hbm
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    runtime = PhysicalRuntimeContext(configuration(), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    pc = next(iter(core._hbm_control_state["pcs"].values()))
    core._hbm_control_state["extension"] = {"pc": pc, "history": pc["column"]}
    saved = runtime.snapshot()
    transaction = runtime.begin_transaction()
    core.submit(AccessRequest("numeric", "write", 65536, 4096))
    with monkeypatch.context() as patch:
        if fallback:
            patch.setattr(_compiled_hbm, "_pack_hbm_numeric", lambda *_: None)
        core.submit(AccessRequest("following", "read", 131072, 1024, 50000.))
    runtime.restore(transaction)
    assert runtime.snapshot() == saved
    extension = core._hbm_control_state["extension"]
    assert extension["pc"]["column"] is extension["history"]
    assert not core._hbm_transactions
    request = AccessRequest("retry", "write", 65536, 4096)
    expected = core.submit(request)
    runtime.restore(saved)
    transaction = runtime.begin_transaction()
    assert core.submit(request) == expected
    runtime.commit_transaction(transaction)
    assert not core._hbm_transactions


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_public_snapshot_preserves_active_transaction_token(finish):
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    runtime = PhysicalRuntimeContext(configuration(), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    before = state(core)
    token = core.begin_transaction()
    core.submit(AccessRequest("prefix", "write", 65536, 4096))
    middle = state(core)
    saved = runtime.snapshot()
    core.submit(AccessRequest("after-snapshot", "write", 131072, 4096))
    runtime.restore(saved)
    assert core._hbm_transactions[-1] is token
    assert state(core) == middle
    getattr(core, finish + "_transaction")(token)
    assert state(core) == (before if finish == "rollback" else middle)
