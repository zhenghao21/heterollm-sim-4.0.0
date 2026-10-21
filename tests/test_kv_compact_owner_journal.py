"""External-PD compact statistics undo protects only groups that can change."""
from copy import deepcopy

import pytest

from heterollm_sim.data_motion import AccessKind
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import SramConfig


def executor():
    config = SramConfig(65536, 64, 100., 100., 100., 1., 1., 2, 8)
    components = [ComponentSpec(name, "sram", capacity_bytes=65536,
        metadata={"physical_memory_config": config}) for name in ("a", "b")]
    result = PhysicalKVTransferExecutor(components, retain_phase_details=False)
    for component in ("a", "b"):
        result._price_chunks(component, AccessKind.WRITE, 0, 64, arrival_ns=0.,
                             caller_managed_transaction=True)
    return result


def write(executor, component, *, address=64):
    executor._price_chunks(component, AccessKind.WRITE, address, 64, arrival_ns=10.,
                           caller_managed_transaction=True)


def test_owner_undo_keeps_other_groups_by_reference_and_restores_submitted_prefix():
    value = executor()
    original = value.snapshot()
    owner = value.services["a"].physical_owner
    other = next(group for key, group in value._cost_groups.items() if key[1] != owner)
    with pytest.raises(ValueError, match="last statistics"):
        with value._io_transaction(owners=(owner,)):
            write(value, "a")
            raise ValueError("last statistics")
    assert value.snapshot() == original
    assert next(group for key, group in value._cost_groups.items() if key[1] != owner) is other
    assert not value._compact_cost_savepoints


def test_outer_undo_protects_a_new_owner_first_touched_in_nested_phase():
    value = executor()
    original = deepcopy(value._cost_groups)
    owner_a, owner_b = (value.services[name].physical_owner for name in ("a", "b"))
    with pytest.raises(ValueError, match="outer"):
        with value._io_transaction(owners=(owner_a,)):
            write(value, "a")
            with value._io_transaction(owners=(owner_b,)):
                write(value, "b")
            raise ValueError("outer")
    assert value._cost_groups == original
    assert not value._compact_cost_savepoints


def test_inner_failure_preserves_outer_statistics_prefix_then_outer_can_commit():
    value = executor()
    owner_a, owner_b = (value.services[name].physical_owner for name in ("a", "b"))
    with value._io_transaction(owners=(owner_a,)):
        write(value, "a")
        prefix = deepcopy(value._cost_groups)
        with pytest.raises(ValueError, match="inner"):
            with value._io_transaction(owners=(owner_b,)):
                write(value, "b")
                raise ValueError("inner")
        assert value._cost_groups == prefix
    assert not value._compact_cost_savepoints


def test_unknown_statistics_hook_retains_full_ledger_rollback(monkeypatch):
    value = executor()
    original = value.snapshot()
    owner = value.services["a"].physical_owner
    record = value._record_memory_cost
    def injected(*args, **kwargs):
        record(*args, **kwargs)
        for group in value._cost_groups.values():
            group["custom_statistics"] = ["partial"]
        raise ValueError("unknown hook")
    monkeypatch.setattr(value, "_record_memory_cost", injected)
    with pytest.raises(ValueError, match="unknown hook"):
        with value._io_transaction(owners=(owner,)):
            write(value, "a")
    assert value.snapshot() == original
    assert not value._compact_cost_savepoints


def test_native_compact_commit_keeps_earlier_group_values_immutable_for_outer_rollback():
    value = executor()
    owner = value.services["a"].physical_owner
    key = next(key for key in value._cost_groups if key[1] == owner)
    previous = value._cost_groups[key]
    before = deepcopy(previous)
    original = value.snapshot()
    with pytest.raises(ValueError, match="outer rollback"):
        with value._io_transaction(owners=(owner,), _owned_timeline=True):
            with value._io_transaction(owners=(owner,), _owned_timeline=True):
                write(value, "a")
            assert previous == before
            assert value._cost_groups[key]["submitted_phase_count"] == previous["submitted_phase_count"]+1
            raise ValueError("outer rollback")
    assert value.snapshot() == original
