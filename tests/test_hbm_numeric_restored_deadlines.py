"""Mapping materialization keeps the original active-deadline scan points."""
import copy

import pytest

from heterollm_sim._compiled_hbm import _pack_hbm_numeric, execute_hbm_numeric
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.dram_core import DramCore, _BankState
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import assert_core_state_equal, configuration


def _seed_open_row_without_control_mapping(core):
    packed = _pack_hbm_numeric(core, AccessRequest("seed", "read", 0, 32))
    assert packed is not None
    core._banks[packed.bank_ids[0]] = _BankState(0, 0., 0.)
    assert not core._hbm_control_state


@pytest.mark.parametrize("deadline,completion,activates,precharges,hits,misses", [
    (40., 8907., 2047, 2040, 1, 2047),
    (100., 8407., 2047, 2021, 1, 2047),
    (200., 5125., 1023, 960, 1025, 1023),
])
def test_open_row_mapping_is_included_at_original_next_activate_scan(
    deadline, completion, activates, precharges, hits, misses,
):
    core = DramCore(configuration(channels=1, maximum_active_ns=deadline),
                    capture_details=False)
    _seed_open_row_without_control_mapping(core)
    result = execute_hbm_numeric(core, AccessRequest("read", "read", 0, 2048 * 32))
    assert result is not None
    # These values come from the original complete scan after each ACT.
    # A newly materialized mapping must neither vanish from that scan nor
    # cause a maintenance command earlier than the original scan point.
    assert result.completion_ns == completion
    assert result.counters["activate_commands"] == activates
    assert result.counters["precharge_commands"] == precharges
    assert result.counters["row_hits"] == hits
    assert result.counters["row_misses"] == misses
    assert result.counters["row_conflicts"] == 0


def test_mapping_only_row_hit_does_not_move_deadline_scan_before_an_activate():
    core = DramCore(configuration(channels=1, maximum_active_ns=40.),
                    capture_details=False)
    _seed_open_row_without_control_mapping(core)
    result = execute_hbm_numeric(core, AccessRequest("hit", "read", 0, 32, 100.))
    assert result is not None
    assert result.completion_ns == 110.
    assert result.counters["row_hits"] == 1
    assert result.counters["activate_commands"] == 0
    assert result.counters["precharge_commands"] == 0


def test_restoring_mapping_gap_repeats_the_same_deadline_behavior():
    runtime = PhysicalRuntimeContext(
        configuration(channels=1, maximum_active_ns=40.), "hbm",
        capture_details=False,
    )
    core = runtime.runtimes["hbm"].core
    _seed_open_row_without_control_mapping(core)
    saved = runtime.snapshot()
    request = AccessRequest("read", "read", 0, 2048 * 32)
    expected = execute_hbm_numeric(core, request)
    assert expected is not None
    assert expected.completion_ns == 8907.
    expected_core = copy.deepcopy(core)
    for _ in range(2):
        runtime.restore(saved)
        actual = execute_hbm_numeric(core, request)
        assert actual == expected
        assert_core_state_equal(core, expected_core)
