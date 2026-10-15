"""Resource-lane and asynchronous KV lifecycle interaction regressions."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.kv_pool import KvPoolError
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbm_config
from test_kv_transfer_deep_semantics import page, request, transfer_executor


@pytest.mark.parametrize("details", [False, True])
@pytest.mark.parametrize("lanes", [1, 2])
def test_physical_stages_preserve_lane_readiness_and_real_predecessors(details, lanes):
    config = make_hbm_config(capacity_bytes=1048576, channels=1,
                             pseudo_channels_per_channel=1)
    kernel = UnifiedEventKernel(resource_capacities={"wire": lanes},
        resource_owners={"memory:dram:data:0": "wire"}, capture_physical_details=details)
    kernel.submit([TaskSpec("physical", "request", "read", TaskCategory.MEMORY,
        metadata={"physical_memory_config": config, "physical_owner": "memory",
                  "memory_accesses": [{"operation": "read", "address": 0, "byte_count": 64}]})])
    physical = kernel.step()
    kernel.submit([TaskSpec("ordinary", "request", "ordinary", TaskCategory.MEMORY,
                            demands=(ResourceDemand("wire", 100),))])
    ordinary = kernel.step()
    if lanes == 1:
        assert ordinary.start_ns == physical.end_ns
        assert ordinary.resource_predecessors["wire"]["task_id"] == "physical"
    else:
        assert ordinary.start_ns == 0
        assert "wire" not in ordinary.resource_predecessors
        kernel.submit([TaskSpec("third", "request", "third", TaskCategory.MEMORY,
                                demands=(ResourceDemand("wire", 10),))])
        third = kernel.step()
        assert third.start_ns == physical.end_ns
        assert third.resource_predecessors["wire"]["task_id"] == "physical"


def test_old_async_phase_cannot_advance_cancel_or_roll_back_again():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    initial = e.prepare_async(request(p, "hbf"))
    final = initial.advance()
    before = e.snapshot()
    for hook in (initial.advance, initial.cancel, initial.rollback):
        with pytest.raises(KvPoolError):
            hook()
        assert e.snapshot() == before
    final.cancel()
    after_cancel = e.snapshot()
    with pytest.raises(KvPoolError, match="already committed or cancelled"):
        initial.advance()
    assert e.snapshot() == after_cancel


def test_async_phase_token_rewinds_with_executor_snapshot():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    initial = e.prepare_async(request(p, "hbf"))
    saved = e.snapshot()
    old_final = initial.advance()
    expected = e.snapshot()
    e.restore(saved)
    with pytest.raises(KvPoolError, match="every physical transfer phase"):
        old_final.commit()
    final = initial.advance()
    replayed = e.snapshot()
    for key in ("runtime", "bindings", "events", "counters"):
        assert replayed[key] == expected[key]
    assert replayed["async_plan_states"][0]["phase_token"] != expected["async_plan_states"][0]["phase_token"]
    final.commit()
    committed = e.snapshot()
    with pytest.raises(KvPoolError, match="already committed or cancelled"):
        initial.advance()
    assert e.snapshot() == committed


def test_restored_same_index_phase_rejects_discarded_future_bills():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    initial = e.prepare_async(request(p, "hbf"))
    saved = e.snapshot()
    old_final = initial.advance()
    e.restore(saved)
    # Real intervening traffic changes this replay's downstream completion.
    owner = e.services["hbf"].physical_owner
    e.runtime.timeline.reserve(owner + ":nand:host", 0, 1_000_000)
    final = initial.advance()
    assert final.completion_ns > old_final.completion_ns
    before = e.snapshot()
    for hook in (old_final.commit, old_final.cancel):
        with pytest.raises(KvPoolError, match="discarded future"):
            hook()
        assert e.snapshot() == before
    final.commit()
    assert e.events[-1]["completion_ns"] == final.completion_ns
    final.commit()  # Current-branch terminal commit stays idempotent.
    assert len(e.events) == 1
    with pytest.raises(KvPoolError, match="discarded future"):
        old_final.commit()


def test_active_snapshot_can_be_reused_without_reviving_discarded_plans():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    before_prepare = e.snapshot()
    initial = e.prepare_async(request(p, "hbf"))
    final = initial.advance()
    saved = e.snapshot()
    for _ in range(3):
        e.restore(saved)
        final.commit()
        final.commit()
        assert e._async_plan_states == {}
        assert len(e.events) == 1
    e.restore(before_prepare)
    with pytest.raises(KvPoolError, match="no longer active"):
        final.commit()
    new_initial = e.prepare_async(request(p, "hbf"))
    assert tuple(e._async_plan_states) == (1,)
    with pytest.raises(KvPoolError, match="no longer active"):
        initial.advance()
    new_initial.cancel()


def test_terminal_async_states_do_not_accumulate_in_executor_snapshots():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    retained_callbacks = []
    for i in range(12):
        target = "hbf" if p.owner_component == "hbm" else "hbm"
        final = e.prepare_async(request(p, target)).advance()
        retained_callbacks.append(final)
        if i % 2:
            final.cancel()
            with pytest.raises(KvPoolError, match="already committed or cancelled"):
                final.cancel()
        else:
            final.commit()
            final.commit()
            p.owner_component = target
            p.allocation_generation += 1
        assert e._async_plan_states == {}
        assert e.snapshot()["async_plan_states"] == {}
    assert e.counters["transfer_count"] == 6
    assert e.counters["cancelled_transfer_count"] == 6


def test_synchronous_plan_recommits_after_reusing_prepared_snapshot():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    plan = e(request(p, "hbf"))
    saved = e.snapshot()
    plan.commit()
    expected = e.snapshot()
    for _ in range(3):
        e.restore(saved)
        assert e._bindings[p.logical_page_id].component_id == "hbm"
        plan.commit()
        plan.commit()
        assert e.snapshot() == expected
        assert e._bindings[p.logical_page_id].component_id == "hbf"
        assert e.counters["transfer_count"] == 1
        assert e.events[-1]["completion_ns"] == plan.completion_ns


def test_synchronous_plan_from_discarded_future_cannot_commit():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    saved = e.snapshot()
    old_plan = e(request(p, "hbf"))
    old_plan.commit()
    e.restore(saved)
    for hook in (old_plan.commit, old_plan.rollback):
        with pytest.raises(KvPoolError, match="no longer active"):
            hook()
        assert e.snapshot() == saved
    new_plan = e(request(p, "hbf"))
    before = e.snapshot()
    with pytest.raises(KvPoolError, match="no longer active"):
        old_plan.commit()
    assert e.snapshot() == before
    new_plan.commit()
    assert e._bindings[p.logical_page_id].component_id == "hbf"


def test_terminal_synchronous_states_do_not_accumulate_in_snapshots():
    p = page("hbm", size=16384)
    e = transfer_executor(p)
    retained_callbacks = []
    for i in range(12):
        target = "hbf" if p.owner_component == "hbm" else "hbm"
        plan = e(request(p, target))
        retained_callbacks.append(plan)
        if i % 2:
            plan.rollback()
            plan.rollback()
        else:
            plan.commit()
            plan.commit()
            p.owner_component = target
            p.allocation_generation += 1
        assert e._async_plan_states == {}
        assert e.snapshot()["async_plan_states"] == {}
    assert e.counters["transfer_count"] == 6
    assert len(e.events) == 6


def buffered_executor(p, *, same_owner=False, relay=False):
    original = transfer_executor(p)
    components = []
    for name, component in original.components.items():
        if name == "hbf":
            config = replace(component.metadata["physical_memory_config"],
                write_buffer_bytes=65536)
            component = replace(component, metadata={**component.metadata,
                                                     "physical_memory_config": config})
        components.append(component)
    if same_owner:
        hbf = next(c for c in components if c.component_id == "hbf")
        components.append(replace(hbf, component_id="hbf-view",
                                  metadata={**hbf.metadata, "memory_service_owner": "hbf.memory"}))
    e = PhysicalKVTransferExecutor(components, relay_components=("hbf",) if relay else ())
    e.bind_page(p)
    return e


def nand_core(e):
    return e.runtime.runtimes[e.services["hbf"].physical_owner].core


def test_cancel_buffered_target_discards_only_its_unprogrammed_pages():
    p = page("hbm", size=4096)
    e = buffered_executor(p)
    adjacent = SimpleNamespace(**{**vars(page("hbf", size=4096)),
                                 "logical_page_id": 1, "request_id": "neighbor"})
    e.bind_page(adjacent)
    e.write_page(adjacent)
    plan = e.prepare_async(request(p, "hbf")).advance()
    core = nand_core(e)
    assert core.pending_write_buffer_bytes == 8192
    before = dict(e.runtime.timeline.bytes_moved)
    billed = plan.cancel()
    assert billed["phases"][-1]["write_completion"] == "host_buffer_ack_pending_media_drain"
    assert core.pending_write_buffer_bytes == 4096
    assert e.runtime.timeline.bytes_moved == before
    assert e._bindings[p.logical_page_id].component_id == "hbm"
    assert core.flush_write_buffer()["physical_programs"] == 1


def test_successful_migration_releases_source_staging_and_keeps_read_cost():
    p = page("hbf", size=4096)
    e = buffered_executor(p)
    e.write_page(p)
    before_host = dict(e.runtime.timeline.bytes_moved)
    plan = e(request(p, "hbm"))
    plan.commit()
    core = nand_core(e)
    assert core.pending_write_buffer_bytes == 0
    assert core.flush_write_buffer()["physical_programs"] == 0
    assert e._bindings[p.logical_page_id].component_id == "hbm"
    assert e.counters["read_bytes"] == 4096
    assert all(e.runtime.timeline.bytes_moved[key] >= value for key, value in before_host.items())


def test_same_owner_migration_discards_source_page_without_target_page():
    p = page("hbf", size=4096)
    e = buffered_executor(p, same_owner=True)
    e.write_page(p)
    source_address = e.address(p)
    plan = e(request(p, "hbf-view"))
    plan.commit()
    assert e.address(p) != source_address
    assert nand_core(e).pending_write_buffer_bytes == 4096
    assert nand_core(e).flush_write_buffer()["physical_programs"] == 1


def test_plain_release_discards_staging_before_address_reuse():
    p = page("hbf", size=4096)
    e = buffered_executor(p)
    e.write_page(p)
    e.release_page(p)
    assert nand_core(e).pending_write_buffer_bytes == 0
    assert nand_core(e).flush_write_buffer()["physical_programs"] == 0


def test_relay_release_keeps_read_traffic_but_discards_unused_dirty_copy():
    p = page("hbm", size=4096)
    e = buffered_executor(p, relay=True)
    plan = e(request(p, "hbm"))
    plan.commit()
    assert e.counters["relay_bytes"] == 8192
    assert e._bindings[p.logical_page_id].component_id == "hbm"
    assert nand_core(e).pending_write_buffer_bytes == 0
    assert nand_core(e).flush_write_buffer()["physical_programs"] == 0
