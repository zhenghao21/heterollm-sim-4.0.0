"""Static names and selective writeback preserve live HBM controller state."""
import copy
from dataclasses import replace

import pytest

from heterollm_sim import _compiled_hbm
from heterollm_sim._compiled_hbm import _pack_hbm_numeric, execute_hbm_numeric
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration, assert_same, assert_core_state_equal


def test_static_layout_reuses_names_but_rechecks_live_aliases_and_configuration():
    config = configuration()
    core = DramCore(config, capture_details=False)
    request = AccessRequest("r", "read", 0, 512)
    first = _pack_hbm_numeric(core, request)
    again = _pack_hbm_numeric(core, request)
    assert first.bank_ids is again.bank_ids
    assert first.ids is again.ids
    assert first.arrays[0] is not again.arrays[0]
    core.timeline.resource_aliases = {first.ids[-1]: first.ids[-2]}
    assert _pack_hbm_numeric(core, request) is None
    core.timeline.resource_aliases = {}
    core.config = replace(config, metadata={
        "bank_resource_prefix": "other:bank", "row_command_resource_prefix": "other:row",
        "column_command_resource_prefix": "other:column", "data_resource_prefix": "other:data"})
    renamed = _pack_hbm_numeric(core, request)
    assert all(name.startswith("other:") for name in renamed.bank_ids + renamed.ids)
    core.config = replace(config, stacks=2)
    wider = _pack_hbm_numeric(core, request)
    assert len(wider.bank_ids) == 2 * len(first.bank_ids)
    assert len(wider.pc_keys) == 2 * len(first.pc_keys)
    assert wider.channel_keys[-1][0] == 1


def test_static_layout_lookup_is_immutable():
    layout = _compiled_hbm._numeric_resource_layout(4, 32, 2, 4, 4, "b", "r", "c", "d")
    with pytest.raises(TypeError):
        layout[4][layout[3][0]] = 99
    with pytest.raises(TypeError):
        layout[5][layout[1][0]] = 99


def test_sparse_commit_preserves_extension_fields_and_rebuilds_unmodified_pcs():
    config = configuration()
    reference, numeric = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    assert_same(reference, numeric, AccessRequest("warm-all-banks", "read", 0, 4096))
    for core in (reference, numeric):
        core._hbm_control_state["extra"] = {"opaque": [1, 2]}
        for bank in core._hbm_control_state["banks"].values():
            bank["extra"] = [3]
        for pc in core._hbm_control_state["pcs"].values():
            pc["extra"] = {"value": 4}
    untouched_pc = numeric._hbm_control_state["pcs"][(0, 1, 0)]
    unchanged_history = untouched_pc["column"]
    assert_same(reference, numeric, AccessRequest("one-bank", "write", 0, 32))
    assert untouched_pc["column"] is not unchanged_history
    assert numeric._hbm_control_state["extra"] == {"opaque": [1, 2]}


def test_unmodified_pc_extension_alias_is_detached_before_following_access():
    config = configuration()
    reference, numeric = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    for core in (reference, numeric):
        execute_hbm_numeric(core, AccessRequest("warm-all", "read", 0, 4096))
        pc = core._hbm_control_state["pcs"][(0, 1, 0)]
        core._hbm_control_state["extension"] = {
            name: pc[name] for name in ("activate_window", "activate", "column", "write_end", "next_data_start")}
        execute_hbm_numeric(core, AccessRequest("other-pc", "write", 0, 32))
        assert all(value is not pc[name] for name, value in core._hbm_control_state["extension"].items())
    # An extension writing its former history reference must not change the
    # timing of the next request to the PC whose histories were rebuilt.
    numeric._hbm_control_state["extension"]["column"]["read"][(0, 0)] = 1.e6
    following = AccessRequest("following", "read", 2 * config.burst_bytes, 32)
    expected = execute_hbm_numeric(reference, following)
    actual = execute_hbm_numeric(numeric, following)
    assert actual == expected
    del reference._hbm_control_state["extension"]
    del numeric._hbm_control_state["extension"]
    assert_core_state_equal(numeric, reference)


def test_refresh_commits_changes_to_banks_outside_requested_address():
    config = configuration(refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.)
    reference, numeric = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    assert_same(reference, numeric, AccessRequest("warm", "read", 0, 4096))
    other_bank_id = numeric._bank_id(_compiled_hbm.map_dram_address(config, 32))
    previous = copy.copy(numeric._banks[other_bank_id])
    assert_same(reference, numeric, AccessRequest("refresh-one-address", "read", 0, 32, 1000.))
    assert numeric._banks[other_bank_id] != previous


def test_refresh_materializes_first_seen_pseudo_channel_without_a_bank():
    config = configuration(channels=1, refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.)
    reference, numeric = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    assert_same(reference, numeric, AccessRequest("first-pc", "read", 0, 32))
    assert (0, 0, 1) not in numeric._hbm_control_state["pcs"]
    assert_same(reference, numeric, AccessRequest("refresh", "read", 0, 32, 1000.))
    assert (0, 0, 1) in numeric._hbm_control_state["pcs"]


def test_seen_flag_materializes_default_pc_even_without_changed_timing_values():
    core = DramCore(configuration(), capture_details=False)
    packed = _pack_hbm_numeric(core, AccessRequest("r", "read", 0, 32))
    packed.arrays[5][0] = True
    packed.commit_controller()
    pc = core._hbm_control_state["pcs"][(0, 0, 0)]
    assert pc == {"activate_window": [], "activate": {}, "column": {"read": {}, "write": {}},
                  "write_end": {}, "last_precharge": None, "blocked_until": 0.,
                  "next_data_start": {"read": 0., "write": 0.}}


def test_commit_preserves_materialization_of_control_only_bank():
    core = DramCore(configuration(), capture_details=False)
    execute_hbm_numeric(core, AccessRequest("warm", "read", 0, 32))
    bank_id = next(iter(core._banks))
    core._banks.clear()
    packed = _pack_hbm_numeric(core, AccessRequest("r", "read", 0, 32))
    packed.commit_controller()
    assert bank_id in core._banks
    assert core._banks[bank_id].open_row is None
    assert core._hbm_control_state["banks"][bank_id]["mapping"].address == 0


def test_repeated_runtime_restore_does_not_reuse_numeric_state():
    context = PhysicalRuntimeContext(configuration(refresh_policy="all_bank",
        refresh_interval_ns=400., refresh_cycle_ns=40.), "hbm", capture_details=False)
    core = context.runtimes["hbm"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    core._hbm_control_state["extra"] = {"values": [1, 2]}
    saved = context.snapshot()
    request = AccessRequest("after-restore", "write", 8192, 1024, 1000.)
    expected = core.submit(request)
    expected_state = copy.deepcopy(core)
    for _ in range(2):
        context.restore(saved)
        assert core.submit(request) == expected
        assert_core_state_equal(core, expected_state)
        assert core._hbm_control_state["extra"] == {"values": [1, 2]}


@pytest.mark.parametrize("raises", [False, True])
def test_sparse_batch_rollback_keeps_prefix_and_unrequested_refresh_state(monkeypatch, raises):
    config = configuration(refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.)
    scalar, batched = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    for core in (scalar, batched):
        core.submit(AccessRequest("warm", "read", 0, 4096))
    requests = (AccessRequest("prefix", "read", 0, 1024, 1000.),
                AccessRequest("stop", "write", 8192, 512, 1000.))
    original = _compiled_hbm._HBMNumericState.execute

    def fail_after_mutation(packed, request):
        result = original(packed, request)
        if request.request_id == "stop":
            if raises:
                raise ValueError("injected stop")
            return None
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm._HBMNumericState, "execute", fail_after_mutation)
        if raises:
            with pytest.raises(ValueError, match="injected stop"):
                tuple(scalar.submit(request) for request in requests)
            with pytest.raises(ValueError, match="injected stop"):
                batched.submit_many(requests)
        else:
            tuple(scalar.submit(request) for request in requests)
            batched.submit_many(requests)
    assert_core_state_equal(scalar, batched)


def test_writeback_tracks_signed_zero_changes_exactly():
    core = DramCore(configuration(), capture_details=False)
    execute_hbm_numeric(core, AccessRequest("warm", "read", 0, 32))
    packed = _pack_hbm_numeric(core, AccessRequest("r", "read", 0, 32))
    index = packed.bank_ids.index(next(iter(core._banks)))
    packed.arrays[1][index, 1] = -0.0
    packed.commit_controller()
    import math
    assert math.copysign(1., core._hbm_control_state["banks"][packed.bank_ids[index]]["activate_ready"]) == -1.
    packed.arrays[1][index, 1] = 0.0
    packed.commit_controller()
    assert math.copysign(1., core._hbm_control_state["banks"][packed.bank_ids[index]]["activate_ready"]) == 1.
