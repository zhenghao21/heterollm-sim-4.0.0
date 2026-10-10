from dataclasses import asdict, replace

import pytest

from heterollm_sim.cache_state import ExplicitCacheState
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import DramConfig, MemoryKind


def _config():
    return DramConfig(kind=MemoryKind.GDDR, channels=1, data_lanes=2,
                      banks_per_group=2, rows_per_bank=32, row_bytes=256,
                      capacity_bytes=32768, interface_bandwidth_gb_s=64)


def _task(*, accesses=None, **metadata):
    if accesses is None:
        accesses = ({"operation": "read", "address": 0, "byte_count": 64},
                    {"operation": "read", "address": 64, "byte_count": 64})
    return TaskSpec(
        task_id="transaction", request_id="request", name="transaction",
        category=TaskCategory.MEMORY,
        demands=(ResourceDemand("dram0", 0, bytes_moved=128),),
        metadata={"physical_memory_config": asdict(_config()),
                  "physical_owner": "dram0",
                  "memory_accesses": tuple({"physical_owner": "dram0", "resource_id": "dram0", **a}
                                           for a in accesses), **metadata},
    )


def test_kernel_and_standalone_each_take_one_snapshot(monkeypatch):
    original = PhysicalRuntimeContext.snapshot
    calls = []

    def counted(runtime):
        calls.append(runtime)
        return original(runtime)

    monkeypatch.setattr(PhysicalRuntimeContext, "snapshot", counted)
    kernel = UnifiedEventKernel.from_closed_graph((_task(),))
    assert kernel.step() is not None
    assert calls == [kernel.physical_runtime]
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(_task(), runtime, 0)
    assert calls == [kernel.physical_runtime, runtime]


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
@pytest.mark.parametrize("standalone", [False, True])
def test_second_submit_failure_restores_physical_state_and_kernel_retry(monkeypatch, exception_type, standalone):
    task = _task()
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    runtime = kernel.physical_runtime
    core = runtime.runtime(_config(), "dram0").core
    submit = core.submit
    calls = 0

    def fail_second(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception_type("second physical submission")
        return submit(request)

    monkeypatch.setattr(core, "submit", fail_second)
    before = runtime.snapshot()
    with pytest.raises(exception_type, match="second physical submission"):
        resolve_physical_task(task, runtime, 0) if standalone else kernel.step()
    assert runtime.snapshot() == before
    monkeypatch.setattr(core, "submit", submit)
    if not standalone:
        assert kernel.has_active_tasks
        actual = kernel.step()
        expected = UnifiedEventKernel.from_closed_graph((task,)).step()
        assert actual.start_ns == expected.start_ns
        assert actual.end_ns == expected.end_ns
        assert actual.demands == expected.demands
        assert not kernel.has_active_tasks


def test_invalid_new_allocation_access_removes_new_owner_and_allocation():
    task = _task(
        accesses=({"operation": "read", "buffer_id": "new", "byte_count": 65,
                   "address_source": "stable_buffer_tensor_offset"},),
        physical_allocations=({"physical_owner": "dram0", "buffer_id": "new", "size_bytes": 64},),
    )
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    before = kernel.physical_runtime.snapshot()
    with pytest.raises(ValueError):
        kernel.step()
    assert kernel.physical_runtime.snapshot() == before
    assert kernel.has_active_tasks
    assert len(kernel._ready_heap) == 1


@pytest.mark.parametrize("existing_cache", [False, True])
def test_failure_after_l2_mutation_restores_cache_allocations_and_queue(monkeypatch, existing_cache):
    task = _task(
        accesses=({"operation": "read", "buffer_id": "new", "byte_count": 64,
                   "address_source": "stable_buffer_tensor_offset"},),
        physical_allocations=({"physical_owner": "dram0", "buffer_id": "new", "size_bytes": 64},),
        stateful_l2={
            "owner": "gpu0.l2", "capacity_bytes": 64, "line_bytes": 64,
            "write_back": True, "write_allocate": True,
            "memory_resource": "dram0", "cache_resource": "gpu0.l2",
            "cache_bandwidth_gb_s": 64, "cache_latency_ns": 1,
            "cache_energy_pj_per_byte": 0.0, "cache_parallelism": 1,
            "bandwidth_gb_s": 64, "hbm": asdict(HBMProfile(bandwidth_gb_s=64, resource_id="dram0")),
            "accesses": ({"buffer_id": "new", "offset_bytes": 0, "size_bytes": 64,
                          "operation": "read", "buffer_size_bytes": 64, "allocation_generation": 0},),
            "envelope_resources": ("gpu0.l2.access_order",), "order_resource": "gpu0.l2.access_order",
        },
    )
    task = replace(task, demands=task.demands + (ResourceDemand("gpu0.l2", 0),
                                                ResourceDemand("gpu0.l2.access_order", 0)))
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    runtime = kernel.physical_runtime
    core = runtime.runtime(_config(), "dram0").core
    if existing_cache:
        runtime.allocators["dram0"].allocate("old", 64, 0)
        cache = ExplicitCacheState(64, 64)
        cache.write("old", 0, 64, buffer_size_bytes=64)
        kernel._l2_states["gpu0.l2"] = ((64, 64, True, True), cache)
        before_cache = cache.snapshot()
        before_lines = cache.resident_lines()

    def fail_after_cache_update(request):
        active_cache = kernel._l2_states["gpu0.l2"][1]
        assert any(line.key[0] == "new" for line in active_cache.resident_lines())
        raise RuntimeError("physical failure after L2 mutation")

    monkeypatch.setattr(core, "submit", fail_after_cache_update)
    before = runtime.snapshot()
    with pytest.raises(RuntimeError, match="physical failure after L2 mutation"):
        kernel.step()
    assert runtime.snapshot() == before
    if existing_cache:
        assert kernel._l2_states["gpu0.l2"][1] is cache
        assert cache.snapshot() == before_cache
        assert cache.resident_lines() == before_lines
    else:
        assert kernel._l2_states == {}
    assert kernel.has_active_tasks
    assert len(kernel._ready_heap) == 1
