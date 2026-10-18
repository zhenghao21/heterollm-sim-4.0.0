"""Program-order/wear journals preserve exact rollback without full copies."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest, NandWearConfig
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_presets import resolve_physical_config


@pytest.mark.parametrize("executor", ["summary", "detailed", "compiled"])
def test_frontier_journal_covers_all_write_executors(executor):
    config = resolve_physical_config("hbf_ocp_0_7")
    core = NandCore(config, capture_details=executor == "detailed")
    core.submit(AccessRequest("seed", "write", 0, 4096))
    initial = dict(core._next_program_page)
    token = core.begin_transaction()
    assert not token.program_frontiers
    addresses = [config.array_units * 4096, 4096]
    if executor == "compiled":
        assert core.submit_write_pages(addresses, 0.) is not None
    else:
        for address in addresses:
            core.submit(AccessRequest("write", "write", address, 4096))
    assert len(token.program_frontiers) == 2
    assert core._next_program_page != initial
    core.rollback_transaction(token)
    assert core._next_program_page == initial
    assert not core._nand_transactions


@pytest.mark.parametrize("finish_inner", ["commit", "rollback"])
def test_nested_frontier_journals_keep_outer_original_values(finish_inner):
    core = NandCore(resolve_physical_config("hbf_ocp_0_7"))
    old, added = (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)
    core._set_program_frontier(old, 1)
    outer = core.begin_transaction()
    core._set_program_frontier(old, 2)
    inner = core.begin_transaction()
    core._update_program_frontiers({old: 3, added: 1})
    with pytest.raises(ValueError, match="reverse begin order"):
        core.commit_transaction(outer)
    getattr(core, finish_inner + "_transaction")(inner)
    assert core._next_program_page[old] == (3 if finish_inner == "commit" else 2)
    core.rollback_transaction(outer)
    assert core._next_program_page == {old: 1}
    with pytest.raises(ValueError, match="reverse begin order"):
        core.rollback_transaction(outer)


def test_wear_journal_restores_existing_rows_and_removes_new_blocks():
    config = replace(resolve_physical_config("hbf_ocp_0_7"),
                     wear_model=NandWearConfig(assumption="test explicit wear"))
    core = NandCore(config, capture_details=False)
    core.submit(AccessRequest("seed", "write", 0, 4096))
    initial = deepcopy(core._block_wear_state)
    original_rows = dict(core._block_wear_state)
    token = core.begin_transaction()
    core.submit(AccessRequest("rewrite", "write", 0, 4096))
    core.submit(AccessRequest("new-block", "write", config.array_units * config.pages_per_block * 4096, 4096))
    assert len(token.block_wear) == 2
    core.rollback_transaction(token)
    assert core._block_wear_state == initial
    assert all(core._block_wear_state[key] is row for key, row in original_rows.items())


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_public_snapshot_restores_active_nand_token_identity(finish):
    config = resolve_physical_config("hbf_ocp_0_7")
    runtime = PhysicalRuntimeContext(config=config, physical_owner="hbf", capture_details=False)
    core = runtime.runtimes["hbf"].core
    core.submit(AccessRequest("seed", "write", 0, 4096))
    initial = dict(core._next_program_page)
    token = core.begin_transaction()
    core.submit(AccessRequest("page-1", "write", config.array_units * 4096, 4096))
    saved = runtime.snapshot()
    middle = dict(core._next_program_page)
    core.submit(AccessRequest("page-2", "write", config.array_units * 8192, 4096))
    getattr(core, finish + "_transaction")(token)
    runtime.restore(saved)
    assert core._nand_transactions[-1] is token
    assert core._next_program_page == middle
    core.rollback_transaction(token)
    assert core._next_program_page == initial


@pytest.mark.parametrize("executor", ["summary", "detailed", "compiled"])
def test_cache_journal_restores_hits_fills_invalidation_and_new_banks(executor):
    config = resolve_physical_config("hbf_ocp_0_7")
    runtime = PhysicalRuntimeContext(config=config, physical_owner="hbf",
                                     capture_details=executor == "detailed")
    core = runtime.runtimes["hbf"].core
    for address in (0, 4096, 8192):
        core.submit(AccessRequest("seed", "read", address, 4096))
    before = runtime.snapshot()
    saved = runtime.begin_transaction()
    token = core._nand_transactions[-1]
    assert not token.cache_banks and "_read_cache" not in saved["runtime_state"]["hbf"]["core"]
    if executor == "compiled":
        assert core.submit_read_pages([0, 12288], 0.) is not None
        assert core.submit_write_pages([0, 12288], 0.) is not None
    else:
        for operation in ("read", "write"):
            for address in (0, 12288):
                core.submit(AccessRequest("access", operation, address, 4096))
    # Resident execution leaves Python slots untouched; scalar execution
    # journals only the two banks it actually changes.
    assert len(token.cache_banks) == (0 if executor == "compiled" else 2)
    assert token.cache_snapshot is None
    runtime.restore(saved)
    assert runtime.snapshot() == before


@pytest.mark.parametrize("finish_inner", ["commit", "rollback"])
def test_nested_cache_journals_preserve_original_slots_and_scalar_lifetimes(finish_inner):
    config = resolve_physical_config("hbf_ocp_0_7")
    core = NandCore(config, capture_details=False)
    core.submit(AccessRequest("seed", "read", 0, 4096))
    initial = deepcopy(core.__dict__["_read_cache"])
    original = next(iter(core.__dict__["_read_cache"].values()))
    outer = core.begin_transaction()
    core.submit(AccessRequest("hit", "read", 0, 4096))
    middle = deepcopy(core.__dict__["_read_cache"])
    inner = core.begin_transaction()
    core.submit_write_pages([0, 4096], 0.)
    core.submit_read_pages([0, 4096], 0.)
    getattr(core, finish_inner + "_transaction")(inner)
    if finish_inner == "rollback":
        assert core.__dict__["_read_cache"] == middle
    core.rollback_transaction(outer)
    assert core.__dict__["_read_cache"] == initial
    assert next(iter(core.__dict__["_read_cache"].values())) is original


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_public_snapshot_reuses_active_cache_undo_with_cross_bank_aliases(finish):
    from heterollm_sim.nand_core import _ReadCacheSlot

    runtime = PhysicalRuntimeContext(config=resolve_physical_config("hbf_ocp_0_7"),
                                     physical_owner="hbf", capture_details=False)
    core = runtime.runtimes["hbf"].core
    first, second = _ReadCacheSlot(None, 10., 20., 3), _ReadCacheSlot(None, 30., 40., 4)
    row = [first, second, first]
    core._read_cache = {(0,): row, (1,): row}
    extension = {"slot": second, "row": row, "cache": core._read_cache, "values": [7]}
    extension["cycle"] = extension
    first.extension = second.extension = extension
    first.dictionary = vars(first)
    token = core.begin_transaction()
    core._touch_cache_bank((0,))
    first.ready_ns = 11.
    saved = runtime.snapshot()  # Upgrade undo before live objects can be replaced.
    assert token.cache_snapshot is not None
    for _ in range(2):
        first.ready_ns = 99.
        getattr(core, finish + "_transaction")(token)
        runtime.restore(saved)
        assert core._nand_transactions[-1] is token
        assert core._read_cache[(0,)][0].ready_ns == 11.
        core.rollback_transaction(token)
        cache, current = core._read_cache, core._read_cache[(0,)]
        assert current is cache[(1,)] and current[0] is current[2]
        assert current[0].ready_ns == 10.
        assert current[0].extension is current[1].extension
        assert current[0].extension["slot"] is current[1]
        assert current[0].extension["row"] is current
        assert current[0].extension["cache"] is cache
        assert current[0].extension["cycle"] is current[0].extension
        assert current[0].dictionary is vars(current[0])
        runtime.restore(saved)
