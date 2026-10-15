"""State ownership, failure boundaries and physical capacity regressions."""
import pytest

from heterollm_sim.cache_state import ExplicitCacheState
from heterollm_sim.communication import RouteHop
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent, KvPoolError
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbf_config
from heterollm_sim.residency import AllocationResidencyManager, ResidencyValidationError
from heterollm_sim.serde import to_primitive
from heterollm_sim.serving import BatchCost
from heterollm_sim.workspace_memory import physical_workspace_bound
from test_async_kv_prefetch import offloaded_runtime, runtime as serving_runtime


@pytest.mark.parametrize("buffer_id", [None, "a"])
def test_cache_flush_rollback_restores_dirty_lines_and_writeback_obligation(buffer_id):
    cache = ExplicitCacheState(128, 64)
    cache.write("a", 0, 64)
    cache.write("b", 0, 64)
    before, lines = cache.snapshot(), cache.resident_lines()
    transaction = cache.begin_transaction()
    assert cache.flush(buffer_id).writeback_bytes > 0
    cache.rollback_transaction(transaction)
    assert cache.snapshot() == before
    assert cache.resident_lines() == lines
    # Retrying the flush must bill the same dirty data, exactly once.
    assert cache.flush().writeback_bytes == 128
    assert cache.flush().writeback_bytes == 0


def test_released_prefix_allows_exclusive_handoff_only_after_last_prefix():
    pool = DynamicKVPool([KvPoolComponent("hbm", 4096)], page_bytes=1024)
    assert pool.resize_detailed("first", 1).success
    page = pool.request_pages("first")[0]
    pool.register_prefix("prefix-a", (page,))
    pool.register_prefix("prefix-b", (page,))
    pool.release_prefix("prefix-b")
    assert page.prefix_key == "prefix-a" and page.ref_count == 2
    with pytest.raises(KvPoolError, match="exclusive"):
        pool.handoff_request("first", "second")
    pool.release_prefix("prefix-a")
    assert page.prefix_key is None and page.ref_count == 1
    assert pool.handoff_request("first", "second") == (page.logical_page_id,)
    assert pool.request_pages("second")[0] is page
    pool.release("second")
    assert not pool.pages()
    assert pool.used_bytes_by_component["hbm"] == 0


def test_duplicate_prefix_page_rejected_without_reference_or_capacity_change():
    pool = DynamicKVPool([KvPoolComponent("hbm", 4096)], page_bytes=1024)
    assert pool.resize_detailed("request", 1).success
    page = pool.request_pages("request")[0]
    with pytest.raises(KvPoolError, match="same physical page"):
        pool.register_prefix("invalid", (page, page))
    assert page.ref_count == 1 and page.prefix_key is None
    assert pool.prefix_pages("invalid") == ()
    pool.release("request")
    assert not pool.pages()


def test_shared_prefix_cannot_silently_overwrite_another_request_kv():
    pool = DynamicKVPool([KvPoolComponent("hbm", 4096)], page_bytes=1024)
    assert pool.resize_detailed("first", 1).success
    pool.register_prefix("prefix", pool.request_pages("first"))
    assert pool.resize_detailed("second", 2, prefix_key="prefix").success
    first = pool.cache_accesses("first", first_token=0, token_count=16)
    second = pool.cache_accesses("second", first_token=0, token_count=16)
    assert first == second
    with pytest.raises(KvPoolError, match="copy-on-write"):
        pool.cache_accesses("second", first_token=15, token_count=2, operation="write")
    # Appending to a private suffix is supported without charging a fake copy.
    assert pool.cache_accesses("second", first_token=16, token_count=1, operation="write")


def prepared_restore():
    runner, _ = offloaded_runtime(distance=1)
    pool = runner.ledger.paged_pool
    executor = pool.transfer_executor
    page = pool.request_pages("low")[0]
    plan = pool.prepare_restore_page(page.logical_page_id, "hbm0")
    while plan.advance is not None:
        plan = plan.advance()
    return runner, pool, executor, page, plan


@pytest.mark.parametrize("cancel", [False, True])
def test_async_publish_rejected_ledger_leaves_physical_identity_and_cost_for_retry(monkeypatch, cancel):
    runner, pool, executor, page, plan = prepared_restore()
    physical_before = executor.snapshot()
    ledger_before = dict(runner.physical_ledger.used_bytes)
    operation = pool.cancel_prepared_restore if cancel else pool.complete_prepared_restore
    owner = "hbm0" if cancel else "hbf0"
    adjust = pool._ledger.adjust
    with monkeypatch.context() as change:
        change.setattr(pool._ledger, "adjust", lambda component, delta:
            False if component == owner and delta < 0 else adjust(component, delta))
        with pytest.raises(KvPoolError, match="could not release"):
            operation(page.logical_page_id, plan, now_ns=plan.completion_ns)
    assert executor.snapshot() == physical_before
    assert runner.physical_ledger.used_bytes == ledger_before
    assert page.owner_component == executor._bindings[page.logical_page_id].component_id == "hbf0"
    operation(page.logical_page_id, plan, now_ns=plan.completion_ns)
    assert page.owner_component == ("hbf0" if cancel else "hbm0")
    assert page.logical_page_id not in pool._pending_moves
    for key in ("read_bytes", "write_bytes", "service_ns", "link_bytes", "link_ns"):
        assert executor.counters[key] == physical_before["counters"][key]


@pytest.mark.parametrize("cancel", [False, True])
def test_async_publish_physical_release_failure_is_atomic_and_retryable(monkeypatch, cancel):
    runner, pool, executor, page, plan = prepared_restore()
    before = executor.snapshot()
    ledger_before = dict(runner.physical_ledger.used_bytes)
    release = executor._release_allocation
    operation = pool.cancel_prepared_restore if cancel else pool.complete_prepared_restore

    def fail_after_release(*args):
        release(*args)
        raise RuntimeError("injected physical release failure")

    with monkeypatch.context() as change:
        change.setattr(executor, "_release_allocation", fail_after_release)
        with pytest.raises((RuntimeError, KvPoolError), match="physical release failure"):
            operation(page.logical_page_id, plan, now_ns=plan.completion_ns)
    assert executor.snapshot() == before
    assert runner.physical_ledger.used_bytes == ledger_before
    operation(page.logical_page_id, plan, now_ns=plan.completion_ns)
    assert page.logical_page_id not in pool._pending_moves


@pytest.mark.parametrize("now", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("cancel", [False, True])
def test_async_publish_rejects_nonfinite_completion_evidence(now, cancel):
    _, pool, executor, page, plan = prepared_restore()
    before = executor.snapshot()
    operation = pool.cancel_prepared_restore if cancel else pool.complete_prepared_restore
    with pytest.raises(KvPoolError):
        operation(page.logical_page_id, plan, now_ns=now)
    assert executor.snapshot() == before


def test_nand_workspace_bound_covers_real_page_aligned_simultaneous_allocations():
    config = make_hbf_config(capacity_bytes=65536, page_bytes=16384)
    tasks = tuple(TaskSpec(str(i), "request", "write", TaskCategory.COMPUTE,
        metadata={"physical_memory_config": to_primitive(config), "physical_owner": "hbf",
                  "memory_accesses": ({"buffer_id": str(i), "byte_count": 64,
                                    "allocation_generation": 1},)}) for i in range(2))
    bound = physical_workspace_bound(tasks)["bytes_by_owner"]["hbf"]
    assert bound == 2 * config.page_bytes
    runtime = PhysicalRuntimeContext()
    runtime.runtime(config, "hbf")
    allocator = runtime.allocators["hbf"]
    allocations = [allocator.allocate(str(i), 64, generation=1) for i in range(2)]
    assert bound >= max(item.base_address + item.size_bytes for item in allocations)


@pytest.mark.parametrize("cancel", [False, True])
def test_multi_page_prefetch_publication_failure_preserves_whole_request(monkeypatch, cancel):
    runner, state = offloaded_runtime(distance=17)
    assert runner._prefetch_waiting()
    while not runner._pending_prefetches["low"]["details"]["completion_is_final"]:
        runner._advance_prefetches_until(runner._next_prefetch_completion())
    entry = runner._pending_prefetches["low"]
    runner.now = entry["completion_ns"]
    if cancel:
        entry.update(cancel_requested_ns=runner.now, cancel_reason="test")
    pool = runner.ledger.paged_pool
    before = runner.ledger._snapshot_paged_pool_state()
    before_executor = pool.transfer_executor.snapshot()
    live_cores = {owner: active.core for owner, active in pool.transfer_executor.runtime.runtimes.items()}
    method = "cancel_prepared_restore" if cancel else "complete_prepared_restore"
    original = getattr(pool, method)
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise RuntimeError("second page publication failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as change:
        change.setattr(pool, method, fail_second)
        with pytest.raises(RuntimeError, match="second page"):
            runner._complete_prefetches()
    after = runner.ledger._snapshot_paged_pool_state()
    assert {k: v for k, v in after.items() if k != "executor"} == {
        k: v for k, v in before.items() if k != "executor"}
    assert pool.transfer_executor.snapshot() == before_executor
    assert all(pool.transfer_executor.runtime.runtimes[owner].core is core
               for owner, core in live_cores.items())
    assert state.kv_pages == 0 and state.swapped_pages == 2
    runner._complete_prefetches()
    assert "low" not in runner._pending_prefetches
    assert not pool._pending_moves
    assert state.kv_pages == (0 if cancel else 2)
    assert state.swapped_pages == (2 if cancel else 0)


def leased_residency():
    manager = AllocationResidencyManager("hbm", 128, page_size_bytes=64)
    manager.register("data", 128, backing="hbf", resident_bytes=128, dirty=True)
    manager.acquire_lease("slow", "data", "slow-reader", offset_bytes=0, size_bytes=128)
    manager.acquire_lease("fast", "data", "fast-reader", offset_bytes=64, size_bytes=64)
    manager.release_lease("slow")
    manager.release_lease("fast")
    return manager


def test_eviction_waits_for_all_parallel_consumers_of_overlapping_ranges():
    manager = leased_residency()
    manager.register("next", 128, backing="hbf", resident_bytes=128)
    assert set(manager.migrations[-1].consumer_task_ids) == {"slow-reader", "fast-reader"}
    assert manager.migrations[-1].dirty_writeback_bytes == 128


def test_retiring_completed_consumer_preserves_other_overlapping_dependencies():
    manager = leased_residency()
    manager.retire_completed_consumers(("fast-reader",))
    manager.register("next", 128, backing="hbf", resident_bytes=128)
    assert manager.migrations[-1].consumer_task_ids == ("slow-reader",)
    manager.assert_consistent()


def test_active_consumer_cannot_be_retired_as_completed():
    manager = leased_residency()
    manager.acquire_lease("active", "data", "not-completed")
    before = dict(manager._released_consumer_ranges)
    with pytest.raises(ResidencyValidationError, match="active residency leases"):
        manager.retire_completed_consumers(("fast-reader", "not-completed"))
    assert manager._released_consumer_ranges == before


@pytest.mark.parametrize("fail", [False, True])
def test_serving_retires_only_successfully_completed_cohort_anchors(monkeypatch, fail):
    runner = serving_runtime()
    runner._arrive()
    runner._fill_slots()
    cohort = runner._select_cohort()
    manager = AllocationResidencyManager("hbm", 64, page_size_bytes=64)
    manager.register("data", 64, backing="hbf", resident_bytes=64)
    runner.residency_manager = manager
    monkeypatch.setattr(runner, "_apply_owner_residency", lambda _cohort, cost: cost)

    def lower(*_args):
        manager.acquire_lease("lease", "data", "cohort-reader")
        manager.release_lease("lease")
        runner._residency_cohort_consumer_ids.add("cohort-reader")
        if fail:
            raise RuntimeError("cohort failed before completion")
        return BatchCost(100.)

    runner.lowerer = lower
    if fail:
        with pytest.raises(RuntimeError, match="cohort failed"):
            runner._execute(cohort)
        assert manager._released_consumer_ranges["data"] == [(0, 64, "cohort-reader")]
    else:
        runner._execute(cohort)
        assert runner.batches[-1].end_ns >= 100.
        assert not manager._released_consumer_ranges


def test_migration_cost_ledger_survives_cancel_and_event_retention():
    runner, _ = offloaded_runtime(distance=1)
    pool = runner.ledger.paged_pool
    executor = pool.transfer_executor
    before = executor.cost_ledger()
    assert before["storage_traffic"]["physical_write_bytes"] == 32768
    assert before["dram_traffic"]["physical_read_bytes"] == 32768
    page = pool.request_pages("low")[0]
    plan = pool.prepare_restore_page(page.logical_page_id, "hbm0")
    # Only the source NAND read has been submitted; cancelling must bill it,
    # and must not charge the unsubmitted destination write or fabric hops.
    submitted = executor.cost_ledger()
    assert submitted["phase_count"] == before["phase_count"] + 1
    pool.cancel_prepared_restore(page.logical_page_id, plan, now_ns=plan.completion_ns)
    executor.events.clear()
    assert executor.cost_ledger() == submitted
    assert submitted["storage_traffic"]["physical_read_bytes"] == 16384
    assert submitted["dram_traffic"]["physical_write_bytes"] == 0
    assert submitted["resource_accounted_bytes"] == sum(submitted["resource_bytes"].values())
    assert submitted["storage_traffic"]["nand_read_queue_depth"] is None
    assert all(page.owner_component == "hbf0" for page in pool.request_pages("low"))


def test_migration_immediate_rollback_removes_only_unpublished_cost():
    runner, _ = offloaded_runtime(distance=1)
    pool = runner.ledger.paged_pool
    executor = pool.transfer_executor
    before = executor.cost_ledger()
    page = pool.request_pages("low")[0]
    plan = pool.prepare_restore_page(page.logical_page_id, "hbm0")
    assert executor.cost_ledger()["phase_count"] > before["phase_count"]
    pool.abort_prepared_restore(page.logical_page_id, plan)
    assert executor.cost_ledger() == before


def test_rejected_abort_keeps_physical_reservation_retryable(monkeypatch):
    runner, _ = offloaded_runtime(distance=1)
    pool = runner.ledger.paged_pool
    executor = pool.transfer_executor
    page = pool.request_pages("low")[0]
    plan = pool.prepare_restore_page(page.logical_page_id, "hbm0")
    before = executor.snapshot()
    adjust = pool._ledger.adjust
    with monkeypatch.context() as change:
        change.setattr(pool._ledger, "adjust", lambda component, delta:
            False if component == "hbm0" and delta < 0 else adjust(component, delta))
        with pytest.raises(KvPoolError, match="could not release"):
            pool.abort_prepared_restore(page.logical_page_id, plan)
    assert executor.snapshot() == before
    pool.abort_prepared_restore(page.logical_page_id, plan)
    assert page.logical_page_id not in pool._pending_moves


def test_memory_phase_reports_actual_core_bytes_and_detailed_payloads():
    from test_kv_transfer_deep_semantics import component, page
    config = make_hbf_config(capacity_bytes=1048576, page_bytes=16384,
                             partial_page_policy="read_modify_write")
    executor = PhysicalKVTransferExecutor([component("hbf", config)])
    item = page(size=16384)
    executor.bind_page(item)
    bill = executor.write_page(item, size_bytes=64)
    ledger = executor.cost_ledger()
    # A partial NAND write incurs an entire page read and page program.
    assert bill["physical_bytes"] == 32768
    assert ledger["storage_traffic"]["physical_read_bytes"] == 16384
    assert ledger["storage_traffic"]["physical_write_bytes"] == 16384
    phase = ledger["phases"][0]
    assert phase["resource_interval_payloads"]
    assert sum(size for values in phase["resource_interval_payloads"].values()
               for _start, _end, size in values) == ledger["resource_accounted_bytes"]


@pytest.mark.parametrize("queue_depth, second_end", [(1, 240.), (2, 220.)])
def test_migration_link_wire_and_inflight_have_same_contract_as_regular_tasks(queue_depth, second_end):
    hop = RouteHop("link", "a", "b", "wire", 8., 20., queue_depth=queue_depth)
    executor = PhysicalKVTransferExecutor([])
    first = executor._submit_link_phase(hop, 100, 0.)
    second = executor._submit_link_phase(hop, 100, 0.)
    assert first["completion_ns"] == 120.
    assert second["completion_ns"] == second_end
    ledger = executor.cost_ledger()
    assert ledger["resource_busy_ns"] == {"wire": 200., "wire.inflight": 240.}
    assert ledger["resource_bytes"] == {"wire": 200, "wire.inflight": 0}


def test_serving_exposes_same_always_on_cost_ledger_as_executor():
    runner = serving_runtime()
    result = runner.run()
    ledger = result.runtime_kernel_metrics["paged_kv_transfers"]["cost_ledger"]
    assert ledger == runner._paged_pool_transfer_executor.cost_ledger()
    assert ledger["phase_count"] > 0
