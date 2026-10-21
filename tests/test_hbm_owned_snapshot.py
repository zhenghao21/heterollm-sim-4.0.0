"""Owned snapshots preserve residency without exporting live bank aliases."""

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration
from test_hbm_resident_controller import visible


def pair():
    contexts = [PhysicalRuntimeContext(configuration(channels=8), "hbm",
                                       capture_details=False) for _ in range(2)]
    contexts[1].runtimes["hbm"].core._hbm_numeric_enabled = False
    return contexts, [context.runtimes["hbm"].core for context in contexts]


def test_owned_snapshot_does_not_disable_later_io_and_has_no_live_bank_alias():
    contexts, cores = pair()
    warm = AccessRequest("warm", "read", 0, 4096)
    assert cores[0].submit(warm) == cores[1].submit(warm)
    saved = contexts[0].snapshot()
    assert cores[0]._hbm_numeric_enabled
    assert cores[0]._hbm_numeric_state is None
    bank_id, saved_bank = next(iter(saved["runtime_state"]["hbm"]["core"]["_banks"].items()))
    assert saved_bank is not cores[0].__dict__["_banks"][bank_id]
    original_time = saved_bank.ready_ns
    for index in range(3):
        request = AccessRequest(str(index), "write", index * 65536, 4096, index * 1000.)
        assert cores[0].submit(request) == cores[1].submit(request)
        assert cores[0]._hbm_numeric_state is not None
    assert saved_bank.ready_ns == original_time
    assert visible(contexts[0].snapshot()) == visible(contexts[1].snapshot())


@pytest.mark.parametrize("alias_kind", ("banks", "control"))
def test_preexisting_public_alias_keeps_residency_disabled(alias_kind):
    contexts, cores = pair()
    warm = AccessRequest("warm", "read", 0, 4096)
    for core in cores:
        core.submit(warm)
    aliases = [core._banks if alias_kind == "banks" else core._hbm_control_state
               for core in cores]
    contexts[0].snapshot()
    assert not cores[0]._hbm_numeric_enabled
    for alias in aliases:
        if alias_kind == "banks":
            next(iter(alias.values())).ready_ns += 10000.
        else:
            next(iter(alias["banks"].values()))["activate_ready"] += 10000.
    assert cores[0].submit(warm) == cores[1].submit(warm)
    assert cores[0]._hbm_numeric_state is None
    assert visible(contexts[0].snapshot()) == visible(contexts[1].snapshot())


def test_restore_owned_snapshot_reuses_same_snapshot_and_reopens_numeric():
    contexts, cores = pair()
    warm = AccessRequest("warm", "read", 0, 4096)
    assert cores[0].submit(warm) == cores[1].submit(warm)
    saved = contexts[0].snapshot()
    oracle_saved = contexts[1].snapshot()
    request = AccessRequest("after-restore", "write", 65536, 4096, 5000.)
    for _ in range(3):
        contexts[0].restore(saved)
        contexts[1].restore(oracle_saved)
        assert cores[0]._hbm_numeric_enabled
        assert cores[0].submit(request) == cores[1].submit(request)
        assert cores[0]._hbm_numeric_state is not None
        assert visible(contexts[0].snapshot()) == visible(contexts[1].snapshot())
