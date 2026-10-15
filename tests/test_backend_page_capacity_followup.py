"""Logical KV payloads and per-media allocation footprints are distinct."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent, KvPoolError
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config
from heterollm_sim.serving import _KVLedger, _PhysicalCapacityLedger


def fixture(*, payload=4096, hbf_capacity=32768, active_hbf=False, ledger=None,
            attach=True, declared_config=True):
    configs = {"hbm": make_hbm_config(capacity_bytes=131072),
               "hbf": make_hbf_config(capacity_bytes=hbf_capacity, page_bytes=16384,
                    write_buffer_bytes=min(32768, hbf_capacity), partial_page_policy="read_modify_write")}
    hardware = [ComponentSpec(name, name, capacity_bytes=config.capacity_bytes,
        read_bandwidth_gbps=800, write_bandwidth_gbps=80,
        metadata={"physical_memory_config": config}) for name, config in configs.items()]
    components = [replace(KvPoolComponent.from_component(component),
        active=(component.component_id == "hbf") if active_hbf else component.component_id == "hbm",
        metadata=component.metadata if declared_config else {}) for component in hardware]
    executor = PhysicalKVTransferExecutor(hardware)
    physical = ledger or _PhysicalCapacityLedger({name: config.capacity_bytes for name, config in configs.items()})
    pool = DynamicKVPool(components, page_bytes=payload, tokens_per_page=16, ledger=physical,
        offload_components=("hbf",), transfer_executor=executor if attach else None)
    return pool, executor, physical


def allocate(pool, executor, count):
    assert pool.resize_detailed("request", count).success
    executor.sync_pages(pool.pages())
    return pool.request_pages("request")


def finish(plan):
    while plan.advance is not None:
        plan = plan.advance()
    return plan


def test_small_hbf_capacity_rejects_third_page_before_physical_submission():
    pool, executor, ledger = fixture()
    pages = allocate(pool, executor, 3)
    assert pool.offload_page(pages[0].logical_page_id, "hbf")
    assert pool.offload_page(pages[1].logical_page_id, "hbf")
    assert ledger.used_bytes == {"hbm": 4096, "hbf": 32768}
    before = executor.snapshot()
    assert not pool.offload_page(pages[2].logical_page_id, "hbf")
    assert executor.snapshot() == before
    assert all(page.bytes == 4096 for page in pages)
    assert [event.bytes for event in pool.events] == [4096, 4096]
    bindings = [executor._binding(page) for page in pages[:2]]
    assert [(b.allocation.base_address, b.allocation.size_bytes) for b in bindings] == [(0, 16384), (16384, 16384)]
    assert pool.component_stats()["hbf"]["logical_pool_bytes"] == 8192
    assert pool.restore_page(pages[0].logical_page_id, "hbm")
    assert ledger.used_bytes == {"hbm": 8192, "hbf": 16384}
    assert pool.offload_page(pages[2].logical_page_id, "hbf")
    assert pool.resize_detailed("request", 0).success
    executor.sync_pages(pool.pages())
    assert ledger.used_bytes == {"hbm": 0, "hbf": 0}
    assert all(not allocator.allocations() for allocator in executor.runtime.allocators.values())


def test_active_nand_resize_and_probe_reserve_whole_pages_atomically():
    pool, executor, ledger = fixture(active_hbf=True)
    assert not pool.can_resize("request", 3)
    assert not pool.resize_detailed("request", 3).success
    assert not pool.pages() and ledger.used_bytes == {"hbm": 0, "hbf": 0}
    pages = allocate(pool, executor, 2)
    assert ledger.used_bytes["hbf"] == 32768
    with pytest.raises(ValueError, match="logical page payload"):
        executor.read_page(pages[0], offset_bytes=4095, size_bytes=2)
    assert executor._binding(pages[0]).allocation.size_bytes == 16384


def test_prefix_eviction_stops_after_freeing_one_physical_page():
    pool, executor, ledger = fixture(active_hbf=True)
    pages = allocate(pool, executor, 2)
    pool.register_prefix("prefix", pages)
    pool.release("request")
    assert pool.evict(16384, component_ids=("hbf",)) == 16384
    executor.sync_pages(pool.pages())
    assert len(pool.pages()) == 1
    assert ledger.used_bytes["hbf"] == 16384


@pytest.mark.parametrize("action", ["complete", "abort", "cancel"])
def test_async_reserves_two_distinct_footprints_and_releases_correct_side(action):
    # A non-burst-aligned payload checks both DRAM and NAND rounding.
    pool, executor, ledger = fixture(payload=4097)
    (page,) = allocate(pool, executor, 1)
    assert ledger.used_bytes["hbm"] == 4160
    assert pool.offload_page(page.logical_page_id, "hbf")
    plan = pool.prepare_restore_page(page.logical_page_id, "hbm")
    assert plan is not None
    assert ledger.used_bytes == {"hbm": 4160, "hbf": 16384}
    assert page.owner_component == "hbf"
    assert executor._binding(page).allocation.size_bytes == 16384
    if action == "complete":
        plan = finish(plan)
        pool.complete_prepared_restore(page.logical_page_id, plan, now_ns=plan.completion_ns)
        assert ledger.used_bytes == {"hbm": 4160, "hbf": 0}
        assert executor._binding(page).allocation.size_bytes == 4160
        assert page.owner_component == "hbm"
    elif action == "abort":
        pool.abort_prepared_restore(page.logical_page_id, plan)
        assert ledger.used_bytes == {"hbm": 0, "hbf": 16384}
    else:
        pool.cancel_prepared_restore(page.logical_page_id, plan, now_ns=plan.completion_ns)
        assert ledger.used_bytes == {"hbm": 0, "hbf": 16384}
    assert ledger.peak_used_bytes == {"hbm": 4160, "hbf": 16384}
    assert not pool._pending_moves
    assert pool.resize_detailed("request", 0).success
    executor.sync_pages(pool.pages())
    assert ledger.used_bytes == {"hbm": 0, "hbf": 0}


def test_sync_commit_failure_reverses_asymmetric_ledger_and_allocation(monkeypatch):
    pool, executor, ledger = fixture()
    (page,) = allocate(pool, executor, 1)
    before = executor.snapshot()
    def fail(*args, **kwargs):
        raise KvPoolError("injected commit failure")
    monkeypatch.setattr(executor, "_commit_transfer", fail)
    with pytest.raises(KvPoolError, match="injected commit failure"):
        pool.offload_page(page.logical_page_id, "hbf")
    assert ledger.used_bytes == {"hbm": 4096, "hbf": 0}
    assert executor.snapshot() == before
    assert page.owner_component == "hbm"


def test_rejected_asymmetric_ledger_transfer_reverses_source_and_physical_plan():
    used = {"hbm": 0, "hbf": 0}
    def adjust(component, delta):
        if component == "hbf" and delta > 0:
            return False
        used[component] += delta
        return True
    ledger = SimpleNamespace(used_bytes=used, limits={"hbm": 131072, "hbf": 32768},
        can_adjust=lambda component, delta: True, adjust=adjust)
    pool, executor, _ = fixture(ledger=ledger)
    (page,) = allocate(pool, executor, 1)
    before = executor.snapshot()
    assert not pool.offload_page(page.logical_page_id, "hbf")
    assert used == {"hbm": 4096, "hbf": 0}
    assert executor.snapshot() == before


@pytest.mark.parametrize("capacity,success", [(32768, True), (16384, False)])
def test_late_executor_attachment_reconciles_existing_logical_pages(capacity, success):
    pool, executor, ledger = fixture(hbf_capacity=capacity, active_hbf=True,
                                    attach=False, declared_config=False)
    assert pool.resize_detailed("request", 2).success
    assert ledger.used_bytes["hbf"] == 8192
    if success:
        pool.set_transfer_executor(executor)
        executor.sync_pages(pool.pages())
        assert ledger.used_bytes["hbf"] == 32768
        assert all(executor._binding(page).allocation.size_bytes == 16384 for page in pool.pages())
    else:
        with pytest.raises(KvPoolError, match="existing pages exceed capacity"):
            pool.set_transfer_executor(executor)
        assert pool.transfer_executor is None
        assert ledger.used_bytes["hbf"] == 8192
        assert pool.allocation_bytes("hbf", 4096) == 4096


def serving_ledger(*, payload=4096, quota=32768):
    pool, executor, physical = fixture(payload=payload)
    policy = SimpleNamespace(component_bytes_per_page={}, bytes_per_page=payload,
        offload_capacity_bytes=quota, cache_component="hbm", offload_component="hbf",
        capacity_pages=32)
    ledger = _KVLedger(policy, physical)
    ledger.paged_pool = pool
    request = SimpleNamespace(spec=SimpleNamespace(request_id="request"), kv_pages=0,
        temporary_kv_pages=0, peak_kv_pages=0, swapped_pages=0, swap_bytes=0)
    return ledger, executor, physical, request


def test_serving_offload_role_quota_is_physical_without_changing_payload_counters():
    ledger, executor, physical, request = serving_ledger(quota=16384)
    assert ledger.resize(request, 2)
    before = executor.snapshot()
    assert not ledger.offload(request)  # 8 KiB payload needs 32 KiB media.
    assert executor.snapshot() == before
    assert ledger.resize(request, 1)
    assert ledger.offload(request)
    assert physical.used_bytes == {"hbm": 0, "hbf": 16384}
    assert request.swap_bytes == ledger.offload_used_bytes == 4096
    ledger.discard_offload(request)
    assert physical.used_bytes == {"hbm": 0, "hbf": 0}


def test_serving_restore_and_prefetch_preflight_use_target_burst_footprint():
    ledger, executor, physical, request = serving_ledger(payload=4097)
    assert ledger.resize(request, 1)
    assert ledger.offload(request)
    assert physical.adjust("hbm", 131072 - 4100)
    before = executor.snapshot()
    assert not ledger.can_restore(request)
    assert ledger.prefetch(request, 1) == 0
    assert ledger.prepare_prefetch(request, 1) == ()
    assert executor.snapshot() == before
    assert physical.adjust("hbm", -(131072 - 4100))
    assert ledger.restore(request)
    assert physical.used_bytes == {"hbm": 4160, "hbf": 0}
