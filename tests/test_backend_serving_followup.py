"""Cross-layer serving identity and accounting regressions."""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hbf_hbm_agent_study import build_agent_scenario
from heterollm_sim.serving import _OnlineRuntime, compile_serving_plan, TopologyAwareBatchCostProvider
from heterollm_sim.cache_state import ExplicitCacheState
from test_kv_transfer_deep_semantics import page, request, transfer_executor

from test_async_kv_prefetch import runtime as serving_runtime


def stateful_l2_agent_scenario():
    from dataclasses import replace
    from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile
    scenario = build_agent_scenario(background=False)
    gpu = scenario.resolve_component_profile("gpu0")
    kernels = tuple(KernelCapability(f"{phase}-{operator}-{dtype}-{bits}",
        ("int8", "fp16", "fp32"), dtype, phase, "tensor", "analytical test declaration",
        operator=operator, internal_dtype=dtype, output_bits=bits)
        for phase in ("prefill", "decode") for operator in ("gemm", "attention")
        for dtype in ("int8", "fp16", "fp32") for bits in (8, 16, 32))
    gpu = replace(gpu, kernel_model=KernelModelProfile("test-device", "test-runtime",
        "test-architecture", kernels=kernels, stateful_l2=True))
    return replace(scenario, component_profiles={**scenario.component_profiles,
        "gpu": {**scenario.component_profiles["gpu"], "legacy-gpu": gpu}})


def test_full_agent_serving_with_stateful_l2_uses_live_cache_for_migration(monkeypatch):
    import heterollm_sim._token_program as token_program_module
    scenario = stateful_l2_agent_scenario()
    runner = _OnlineRuntime(compile_serving_plan(scenario), TopologyAwareBatchCostProvider(scenario))
    executor = runner._paged_pool_transfer_executor
    original = token_program_module.resolve_l2_task
    page_accesses, caches = [], {}

    def observe(task, states, physical_runtime=None):
        resolved = original(task, states, physical_runtime)
        if states is executor.cache_states:
            contract = task.metadata.get("stateful_l2", {})
            if contract:
                cache = states[contract["owner"]][1]
                previous = caches.setdefault(contract["owner"], cache)
                assert previous is cache
            for access in contract.get("accesses", ()):
                if not access["buffer_id"].startswith(executor.namespace + ":"):
                    continue
                bindings = [binding for binding in executor._bindings.values()
                    if (binding.allocation.buffer_id, binding.generation)
                    == (access["buffer_id"], access["allocation_generation"])]
                assert len(bindings) == 1
                assert cache in executor._owner_caches(bindings[0].owner)
                page_accesses.append(access)
        return resolved

    monkeypatch.setattr(token_program_module, "resolve_l2_task", observe)
    result = runner.run()
    assert result.agent_lifecycle_metrics["handoffs"] == 2
    assert {access["operation"] for access in page_accesses} == {"read", "write"}
    assert any(access["allocation_generation"] >= 2 for access in page_accesses)
    assert caches
    writebacks = []
    for event in executor.events:
        phases = event.get("phases", ())
        dirty = [phase for phase in phases if phase["kind"] == "source_cache_writeback"]
        reads = [phase for phase in phases if phase["kind"] == "source_read"]
        if dirty:
            assert reads
            assert min(phase["arrival_ns"] for phase in reads) >= max(
                phase["completion_ns"] for phase in dirty)
            writebacks.extend(dirty)
    assert sum(phase["physical_write_bytes"] for phase in writebacks) > 0
    assert all(not line.buffer_id.startswith(executor.namespace + ":")
        for cache in caches.values() for line in cache.resident_lines())
    snapshot = executor.snapshot()
    executor.restore(snapshot)
    assert all(executor.cache_states[owner][1] is cache for owner, cache in caches.items())


@pytest.mark.parametrize("cold", [False, True])
def test_gpu_kv_uses_actual_pool_pages_across_session_handoff_and_migration(monkeypatch, cold):
    import heterollm_sim.data_motion as kernel_module
    scenario = build_agent_scenario(background=False, write_budget_bytes=None if cold else 0)
    runner = _OnlineRuntime(compile_serving_plan(scenario), TopologyAwareBatchCostProvider(scenario))
    executor = runner._paged_pool_transfer_executor
    original = kernel_module.resolve_physical_task
    accesses = []

    def observe(task, runtime, arrival_ns, **kwargs):
        result = original(task, runtime, arrival_ns, **kwargs)
        if runtime is executor.runtime:
            borrowed = set(result.metadata.get("physical_external_allocations", ()))
            for access in result.metadata.get("memory_accesses", ()):
                name = access.get("buffer_id", "")
                assert not name.startswith("kv:agent-"), "GPU must not create a second KV allocation"
                if not name.startswith(executor.namespace + ":"):
                    continue
                generation = access.get("allocation_generation", 0)
                key = (access["physical_owner"], name, generation)
                assert key in borrowed
                bound = [binding for binding in executor._bindings.values()
                         if (binding.owner, binding.allocation.buffer_id, binding.generation) == key]
                assert len(bound) == 1
                offset = access.get("offset_bytes", 0)
                assert access["address"] == bound[0].allocation.address(offset, access["byte_count"])
                accesses.append((bound[0].request_id, generation, name, access["operation"]))
        return result

    monkeypatch.setattr(kernel_module, "resolve_physical_task", observe)
    result = runner.run()
    assert result.agent_lifecycle_metrics["handoffs"] == 2
    assert {row[0] for row in accesses} == {"agent-0", "agent-1", "agent-2"}
    assert {row[3] for row in accesses} == {"read", "write"}
    if cold:
        assert any(generation >= 2 for _request, generation, _name, _operation in accesses)
    else:
        assert any(request != "agent-0" and ":agent-0:page:" in name
                   for request, _generation, name, _operation in accesses)
        assert not result.runtime_kernel_metrics["paged_kv_transfers"]["cost_ledger"]["phase_count"]
    assert all(not allocation.buffer_id.startswith(("kv:agent-", executor.namespace + ":"))
               for allocator in executor.runtime.allocators.values() for allocation in allocator.allocations())


def dirty_source_executor():
    item = page("hbm", 16384)
    executor = transfer_executor(item)
    binding = executor._binding(item)
    cache = ExplicitCacheState(4096, 64)
    cache.write(binding.allocation.buffer_id, 0, 64, buffer_size_bytes=16384,
                allocation_generation=0)
    cache.write(binding.allocation.buffer_id, 0, 64, buffer_size_bytes=16384,
                allocation_generation=1)
    cache.write("unrelated", 0, 64)
    executor.cache_states["gpu"] = ("test-cache", cache)
    executor.cache_owners[binding.owner] = {"gpu"}
    return executor, item, cache


@pytest.mark.parametrize("cancel", [False, True])
def test_migration_flushes_only_source_generation_and_keeps_real_writeback_cost(cancel):
    executor, item, cache = dirty_source_executor()
    binding = executor._binding(item)
    plan = executor.prepare_async(request(item, "hbf"))
    before = executor.cost_ledger()
    assert before["dram_traffic"]["physical_write_bytes"] == 64
    assert before["dram_traffic"]["physical_read_bytes"] == 16384
    writeback, read = before["phases"]
    assert read["arrival_ns"] >= writeback["completion_ns"]
    assert cache.snapshot().dirty_bytes == 128
    if cancel:
        plan.cancel()
        assert executor.cost_ledger() == before
        assert any(line.buffer_id == binding.allocation.buffer_id and line.allocation_generation == 0
                   and not line.dirty for line in cache.resident_lines())
    else:
        while plan.advance is not None:
            plan = plan.advance()
        plan.commit()
        assert all(not (line.buffer_id == binding.allocation.buffer_id and line.allocation_generation == 0)
                   for line in cache.resident_lines())
    # Releasing generation 0 must not invalidate any other buffer/generation.
    assert sum(line.dirty for line in cache.resident_lines()) == 2
    assert executor.cache_states["gpu"][1] is cache


def test_immediate_migration_abort_restores_dirty_cache_and_cost_for_retry():
    executor, item, cache = dirty_source_executor()
    before = executor.snapshot()
    plan = executor.prepare_async(request(item, "hbf"))
    assert cache.snapshot().flush_writeback_bytes == 64
    plan.rollback()
    assert executor.snapshot() == before
    assert cache.snapshot().dirty_bytes == 192
    assert executor.cache_states["gpu"][1] is cache
    retry = executor.prepare_async(request(item, "hbf"))
    assert executor.cost_ledger()["dram_traffic"]["physical_write_bytes"] == 64
    retry.cancel()


def test_generation_scoped_flush_emits_actual_dirty_ranges_and_is_transactional():
    cache = ExplicitCacheState(256, 64)
    cache.write("page", 0, 64, allocation_generation=1)
    cache.write("page", 64, 64, allocation_generation=2)
    before = cache.snapshot(), cache.resident_lines()
    transaction = cache.begin_transaction()
    flushed = cache.flush("page", allocation_generation=1)
    assert flushed.writeback_bytes == 64
    assert [(access.offset_bytes, access.size_bytes, access.allocation_generation)
            for access in flushed.backing_accesses] == [(0, 64, 1)]
    cache.rollback_transaction(transaction)
    assert (cache.snapshot(), cache.resident_lines()) == before
