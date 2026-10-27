"""Independent allocator checkpoints and immutable NAND footprint reuse."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim import _nand_acceleration as acceleration
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.data_motion import PhysicalService
from heterollm_sim.nand_core import NandCore
from test_owned_nand_sram_timeline import settings


def test_footprint_cache_reuses_geometry_but_rereads_aliases_config_and_live_calendars(monkeypatch):
    context = PhysicalRuntimeContext(settings("nand"), "flash", capture_details=False)
    core = context.runtimes["flash"].core
    original_config = core.config
    acceleration._FOOTPRINT_CACHE.clear()
    first = acceleration.owned_resource_footprint(core)
    assert len(acceleration._FOOTPRINT_CACHE) == 1
    assert acceleration.owned_resource_footprint(core) is first
    assert context._owners_have_native_hooks(("flash",))
    rid = first[0]
    context.timeline.ready_ns[rid] = object()
    assert not context._owners_have_native_hooks(("flash",))
    context.timeline.ready_ns.pop(rid)
    assert context._owners_have_native_hooks(("flash",))
    context.timeline.resource_aliases[rid] = "renamed-resource"
    changed = acceleration.owned_resource_footprint(core)
    assert changed != first and "renamed-resource" in changed
    core.config = replace(core.config, channels=1, capacity_bytes=None)
    assert acceleration.owned_resource_footprint(core) != changed
    context.timeline.resource_aliases.clear()
    core.config = original_config
    original = NandCore._array_id
    with monkeypatch.context() as patch:
        patch.setattr(NandCore, "_array_id", lambda self, mapping: original(self, mapping) + ":patched")
        assert acceleration._footprint_cache_key(core) is None
        assert not context._owners_have_native_hooks(("flash",))
    assert acceleration.owned_resource_footprint(core) is first


@pytest.mark.parametrize("hook", ("mapping", "instance", "config_property"))
def test_footprint_cache_declines_replaced_callbacks(monkeypatch, hook):
    context = PhysicalRuntimeContext(settings("nand"), "flash", capture_details=False)
    core = context.runtimes["flash"].core
    assert acceleration._footprint_cache_key(core) is not None
    if hook == "mapping":
        original = acceleration._owned_execution_module.map_nand_address
        monkeypatch.setattr(acceleration._owned_execution_module, "map_nand_address",
                            lambda *args: original(*args))
    elif hook == "instance":
        monkeypatch.setattr(core, "_array_id", lambda mapping: "custom")
    else:
        cls = type(core.config)
        original = cls.effective_capacity_bytes
        monkeypatch.setattr(cls, "effective_capacity_bytes", property(lambda self: original.__get__(self)))
    assert acceleration._footprint_cache_key(core) is None


def test_runtime_public_restore_and_transactions_keep_allocator_checkpoints_independent():
    # With no Core dispatch, runtime.restore can deliberately revisit a
    # committed allocator token. Production uses this interface for public
    # snapshots and transaction rollback; allocator saves themselves outlive
    # their close boundary by contract.
    context = PhysicalRuntimeContext(settings("nand"), "flash", capture_details=False)
    allocator = context.allocators["flash"]
    allocator.allocate("a", 64)
    outer = context.begin_transaction()
    saved = context.snapshot()
    allocator.allocate("b", 128)
    context.restore(saved)
    allocator.allocate("c", 192)
    cloned, token = deepcopy((context, outer))
    for current, checkpoint in ((context, outer), (cloned, token)):
        current.restore(checkpoint)
        assert [row.buffer_id for row in current.allocators["flash"].allocations()] == ["a"]
        assert current.allocators["flash"]._fit_extents == allocator._fit_extents


def test_price_unknown_configuration_callback_restores_private_sibling_state(monkeypatch):
    from heterollm_sim import physical_contract
    context = PhysicalRuntimeContext(settings("nand"), "flash", capture_details=False)
    context.runtime(settings("nand"), "sibling")
    sibling = context.runtimes["sibling"].core
    service = PhysicalService("service", "flash", "wire", 1., 1., component=object())

    def rejected(component):
        sibling.__dict__["_array_ready"]["callback-array"] = 99.
        raise RuntimeError("unknown configuration callback")

    monkeypatch.setattr(physical_contract, "_borrow_physical_memory_config", rejected)
    with pytest.raises(RuntimeError, match="unknown configuration callback"):
        service.price("READ", 4096, page_offset_bytes=0, runtime=context)
    assert "callback-array" not in sibling.__dict__["_array_ready"]
