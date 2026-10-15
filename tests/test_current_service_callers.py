"""Service identity and KV pool callers use the current typed contracts."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.data_motion import resolve_service
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_pool import (
    DynamicKVPool, KvPoolComponent, KvPoolError, KvPoolTransferPlan, KvPoolUnsupported,
)
from heterollm_sim.memory_types import make_hbm_config


def memory(**metadata):
    config = make_hbm_config(capacity_bytes=4096)
    return ComponentSpec("hbm", "hbm", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config, **metadata})


@pytest.mark.parametrize("obsolete", ["physical_owner", "resource_id", "memory_resource_id", "service_id"])
def test_component_service_identity_rejects_retired_sources(obsolete):
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        resolve_service(memory(**{obsolete: "old", "memory_service_owner": "current"}))


@pytest.mark.parametrize("derived", [{"resource_id": "old"}, {"service_id": "old"}])
def test_derived_service_rejects_retired_identity(derived):
    with pytest.raises(ValueError, match="memory_service"):
        resolve_service(memory(memory_service=derived))


def test_service_identity_is_component_access_and_explicit_physical_owner():
    component = memory(memory_service_owner="shared.memory")
    service = resolve_service(component)
    assert service.service_id == "hbm.access"
    assert service.physical_owner == service.resource_id == "shared.memory"
    with pytest.raises(TypeError, match="service_id"):
        resolve_service(component, service_id="old")


@pytest.mark.parametrize("profile_as_mapping", [False, True])
def test_current_profile_and_derived_service_layers_keep_directional_parameters(profile_as_mapping):
    profile = HBMProfile(bandwidth_gb_s=80, read_bandwidth_gb_s=60,
        write_bandwidth_gb_s=40, efficiency=.5, transaction_bytes=128,
        max_outstanding_requests=7)
    raw_profile = asdict(profile) if profile_as_mapping else profile
    service = resolve_service(memory(), raw_profile)
    assert (service.read_bandwidth_gb_s, service.write_bandwidth_gb_s) == (30, 20)
    assert service.transaction_granularity == 128 and service.queue_depth == 7
    bound = resolve_service(memory(memory_service={"service_id": "hbm.access",
        "physical_owner": "bound.memory", "bandwidth_gb_s": 25,
        "read_bandwidth_gb_s": 20, "write_bandwidth_gb_s": 10,
        "transaction_bytes": 64, "max_outstanding_requests": 3}), raw_profile)
    assert bound.physical_owner == bound.resource_id == "bound.memory"
    assert (bound.read_bandwidth_gb_s, bound.write_bandwidth_gb_s) == (20, 10)
    assert bound.transaction_granularity == 64 and bound.queue_depth == 3


@pytest.mark.parametrize("value", [True, False])
def test_component_rejects_read_only_even_beside_current_writable(value):
    with pytest.raises(ValueError, match="read_only is unsupported"):
        memory(read_only=value, writable=True)


def test_component_writable_controls_service_and_pool_projection():
    component = memory(writable=False)
    assert not component.is_writable
    assert resolve_service(component).write_bandwidth_gb_s == 0
    projected = KvPoolComponent.from_component(component)
    assert projected.active and not projected.writable
    with pytest.raises(ValueError, match="writable must be a boolean"):
        memory(writable="false")


@pytest.mark.parametrize("raw", [{"component_id": "hbm", "capacity_bytes": 4096},
    SimpleNamespace(component_id="hbm", capacity_bytes=4096, kind="hbm")])
def test_pool_rejects_untyped_component_adapters(raw):
    with pytest.raises(KvPoolError, match="ComponentSpec or KvPoolComponent"):
        DynamicKVPool((raw,))


def test_pool_requires_component_sequence_and_keeps_explicit_logical_budget():
    component = memory()
    with pytest.raises(KvPoolError, match="iterable"):
        DynamicKVPool({"hbm": component})
    descriptor = replace(KvPoolComponent.from_component(component), capacity_bytes=128)
    pool = DynamicKVPool((descriptor,), page_bytes=64)
    assert pool.resize_detailed("r", 2).success
    assert not pool.resize_detailed("r", 3).success


@pytest.mark.parametrize("field", ["active", "writable"])
def test_pool_descriptor_rejects_string_boolean(field):
    with pytest.raises(KvPoolError, match="must be booleans"):
        KvPoolComponent("memory", 4096, **{field: "false"})


def pool_components():
    return (KvPoolComponent("active", 256),
            KvPoolComponent("backing", 256, active=False, tier="offload"))


@pytest.mark.parametrize("raw", [12, "12", True])
def test_transfer_executor_must_return_plan_without_mutating_capacity(raw):
    pool = DynamicKVPool(pool_components(), page_bytes=64, transfer_executor=lambda request: raw)
    assert pool.resize_detailed("r", 1).success
    (page,) = pool.request_pages("r")
    before = dict(pool.used_bytes_by_component)
    with pytest.raises(KvPoolUnsupported, match="must return KvPoolTransferPlan"):
        pool.offload_page(page.logical_page_id, "backing")
    assert page.owner_component == "active" and pool.used_bytes_by_component == before


def test_plan_only_executor_and_minimal_ledger_support_current_async_admission():
    used = {"active": 0, "backing": 0}
    def can_adjust(component, delta):
        return 0 <= used[component] + delta <= 256
    def adjust(component, delta):
        if not can_adjust(component, delta):
            return False
        used[component] += delta
        return True
    # Only these two methods are required; native batch/transfer/usage and
    # staged physical scheduling are optional current capabilities.
    ledger = SimpleNamespace(can_adjust=can_adjust, adjust=adjust)
    requests = []
    def execute(request):
        requests.append(request)
        return KvPoolTransferPlan(10., start_ns=20., completion_ns=30.)
    pool = DynamicKVPool(pool_components(), page_bytes=64, ledger=ledger, transfer_executor=execute)
    assert pool.resize_detailed("r", 1).success
    (page,) = pool.request_pages("r")
    assert pool.used_bytes_by_component == used == {"active": 64, "backing": 0}
    assert pool.offload_page(page.logical_page_id, "backing")
    assert pool.used_bytes_by_component == used == {"active": 0, "backing": 64}
    assert pool.snapshot_transfer_executor() is None
    plan = pool.prepare_restore_page(page.logical_page_id, "active")
    assert plan is not None and used == {"active": 64, "backing": 64}
    assert page.owner_component == "backing"
    pool.complete_prepared_restore(page.logical_page_id, plan, now_ns=30.)
    assert page.owner_component == "active"
    assert pool.used_bytes_by_component == used == {"active": 64, "backing": 0}
    assert len(requests) == 2
    pool.release("r")
    assert pool.used_bytes_by_component == used == {"active": 0, "backing": 0}


@pytest.mark.parametrize("obsolete", ["ledger_can_adjust", "ledger_adjust", "ledger_can_transfer", "ledger_transfer"])
def test_pool_rejects_scattered_ledger_callback_interface(obsolete):
    with pytest.raises(TypeError, match=obsolete):
        DynamicKVPool(pool_components(), **{obsolete: lambda *args: True})


def test_pool_rejects_incomplete_ledger_instead_of_silent_success():
    with pytest.raises(KvPoolError, match="must implement can_adjust and adjust"):
        DynamicKVPool(pool_components(), ledger=SimpleNamespace(can_adjust=lambda *args: True))


def test_pool_uses_only_page_methods_and_one_ledger_property():
    pool = DynamicKVPool(pool_components())
    for obsolete in ("offload", "restore", "physical_ledger"):
        assert not hasattr(pool, obsolete)
