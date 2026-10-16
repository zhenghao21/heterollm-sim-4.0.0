"""Host page assembly must charge actual DRAM, links and strict NAND."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    PhysicalRuntimeContext, lower_physical_allocations, resolve_physical_task,
)
from heterollm_sim.memory_types import make_ddr_config
from heterollm_sim.physical_presets import resolve_physical_config


def setup(*, capacity=8192, host_capacity=65536, bandwidth=32.0):
    flash = resolve_physical_config("hbf_ocp_0_7", overrides={
        "channels": 1, "parallel_units": 1, "host_channels": 1,
        "pages_per_block": 4, "blocks_per_plane": 16, "capacity_bytes": 262144})
    host = make_ddr_config(capacity_bytes=host_capacity)
    def hop(name):
        return ({"link_id": name, "resource_id": name,
                 "bandwidth_gb_s": bandwidth, "latency_ns": 80., "queue_depth": 2},)
    binding = {"host_owner": "host", "host_component_id": "hostmem0",
        "host_config": host, "capacity_bytes": capacity,
        "routes": {name: hop(name) for name in
                   ("to_host", "from_host", "to_hbf", "from_hbf", "read_hbf", "write_hbf")}}
    return flash, binding


def task(config, binding, operation, offset=0, count=256, *, buffer="kv:k", generation=0,
         size=8192, preloaded=False):
    return lower_physical_allocations(TaskSpec("io", "request", "logical HBF access",
        TaskCategory.MEMORY, metadata={
            "physical_memory_config": config, "physical_owner": "flash",
            "physical_memory_configs": {"flash": config, "host": binding["host_config"]},
            "hbf_host_staging_by_owner": {"flash": binding},
            "hbf_preloaded_buffers": (("flash", buffer, generation),) if preloaded else (),
            "memory_accesses": ({"operation": operation, "buffer_id": buffer,
                "byte_count": count, "offset_bytes": offset,
                "allocation_generation": generation, "allocation_size_bytes": size},)}))


def test_small_kv_writes_charge_host_then_program_exactly_one_page():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    now, programs = 0., 0
    for token in range(16):
        result = resolve_physical_task(task(config, binding, "write", token * 256), runtime, now)
        costs = result.metadata["physical_execution_by_owner"]
        programs += costs.get("flash", {}).get("pages_programmed", 0)
        assert costs["host"]["physical_write_bytes"] >= 256
        assert costs["to_host"]["resource_bytes"]["to_host"] == 256
        if token < 15:
            assert "flash" not in costs
        now = result.metadata["physical_completion_ns"]
    assert programs == 1
    assert costs["flash"]["erase_operations"] == 1
    assert costs["host"]["physical_read_bytes"] >= 4096
    assert costs["to_hbf"]["resource_bytes"]["to_hbf"] == 4096
    assert runtime.hbf_staging["flash"]["controller"].host_occupied_bytes == 0


def test_attention_reads_unflushed_tail_and_flushed_prefix_from_different_owners():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    first = resolve_physical_task(task(config, binding, "write", count=4352), runtime, 0.)
    result = resolve_physical_task(task(config, binding, "read", count=4352), runtime,
                                   first.metadata["physical_completion_ns"])
    costs = result.metadata["physical_execution_by_owner"]
    assert costs["host"]["logical_read_bytes"] == 256
    assert costs["flash"]["logical_read_bytes"] == 4096
    assert costs["from_host"]["resource_bytes"]["from_host"] == 256
    assert costs["read_hbf"]["resource_bytes"]["read_hbf"] == 4096


def test_complete_page_uses_direct_hbf_route_without_host_bounce():
    config, binding = setup()
    result = resolve_physical_task(task(config, binding, "write", count=4096),
                                   PhysicalRuntimeContext(), 0.)
    costs = result.metadata["physical_execution_by_owner"]
    assert costs["flash"]["pages_programmed"] == 1
    assert costs["write_hbf"]["resource_bytes"]["write_hbf"] == 4096
    assert "host" not in costs
    assert "to_host" not in costs
    assert "to_hbf" not in costs


def test_invalid_read_rolls_back_media_host_and_staging_state():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(task(config, binding, "write"), runtime, 0.)
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="uninitialized|initialized|unwritten"):
        resolve_physical_task(task(config, binding, "read", count=512), runtime, 1000.)
    assert runtime.snapshot() == before


def test_host_staging_reservation_competes_with_other_host_allocations():
    config, binding = setup(capacity=8192, host_capacity=12288)
    runtime = PhysicalRuntimeContext()
    runtime.runtime(binding["host_config"], "host")
    runtime.allocators["host"].allocate("host-request-data", 8192)
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="allocate|capacity"):
        resolve_physical_task(task(config, binding, "write"), runtime, 0.)
    assert runtime.snapshot() == before


def test_slower_actual_host_link_delays_kv_acceptance():
    ends = []
    for bandwidth in (32., .5):
        config, binding = setup(bandwidth=bandwidth)
        result = resolve_physical_task(task(config, binding, "write"), PhysicalRuntimeContext(), 0.)
        ends.append(result.metadata["physical_completion_ns"])
    assert ends[1] > ends[0]


def test_raw_physical_batch_cannot_bypass_logical_host_staging():
    from heterollm_sim.data_motion import resolve_service
    from heterollm_sim.ir import ComponentSpec
    config, binding = setup()
    component = ComponentSpec("hbf", "hbf", capacity_bytes=config.capacity_bytes,
        read_bandwidth_gbps=800., write_bandwidth_gbps=80., metadata={
            "physical_memory_config": config,
            "host_write_staging": {"host_component_id": "hostmem0",
                                   "capacity_bytes": binding["capacity_bytes"]}})
    runtime = PhysicalRuntimeContext()
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="logical buffer descriptors"):
        resolve_service(component).price_batch(
            [{"operation": "write", "address": 0, "byte_count": 4096}], runtime=runtime)
    assert runtime.snapshot() == before


def test_release_and_allocator_alias_preserve_live_root_only():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(task(config, binding, "write"), runtime, 0.)
    runtime.allocators["flash"].allocate("alias", 256, alias_of="kv:k")
    alias = task(config, binding, "read", buffer="alias", size=256)
    # The alias has already been registered and must not become a new allocation.
    alias = replace(alias, metadata={**alias.metadata, "physical_allocations": ()})
    result = resolve_physical_task(alias, runtime, 1000.)
    assert result.metadata["physical_execution_by_owner"]["host"]["logical_read_bytes"] == 256
    runtime.allocators["flash"].release("alias")
    runtime.release_staging_buffer("flash", "alias", 0)
    assert runtime.hbf_staging["flash"]["controller"].host_occupied_bytes == 4096
    runtime.allocators["flash"].release("kv:k")
    runtime.release_staging_buffer("flash", "kv:k", 0)
    assert runtime.hbf_staging["flash"]["controller"].host_occupied_bytes == 0


def test_preloaded_weights_are_read_without_host_or_program_cost():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    result = resolve_physical_task(task(config, binding, "read", buffer="weights:w",
        size=8192, preloaded=True), runtime, 0.)
    costs = result.metadata["physical_execution_by_owner"]
    assert "host" not in costs
    assert costs["flash"]["pages_programmed"] == 0
    assert costs["flash"]["logical_read_bytes"] == 256
    # Allocating and programming later KV must not overwrite a weight block.
    result = resolve_physical_task(task(config, binding, "write", count=4096), runtime,
                                   result.metadata["physical_completion_ns"])
    assert result.metadata["physical_execution_by_owner"]["flash"]["pages_programmed"] == 1


@pytest.mark.parametrize("unrelated_aliases", [False, True])
def test_compiled_preload_keeps_page_link_order_counters_and_live_state(monkeypatch, unrelated_aliases):
    from heterollm_sim.hbf_staging_runtime import _PhysicalIO

    config, binding = setup()
    accesses = (
        task(config, binding, "write", count=4096),
        task(config, binding, "read", count=16 * 4096, buffer="weights:w", size=16 * 4096, preloaded=True),
        task(config, binding, "read", count=16 * 4096, buffer="weights:w", size=16 * 4096, preloaded=True),
        task(config, binding, "read", offset=15 * 4096, count=4096, buffer="weights:w", size=16 * 4096, preloaded=True),
        task(config, binding, "write", offset=256, count=256),
        task(config, binding, "read", count=4096),
    )
    original = _PhysicalIO.read_flash_pages
    compiled_calls = []

    def observed(io, addresses, arrival):
        value = original(io, addresses, arrival)
        if value is not None:
            compiled_calls.append(len(addresses))
        return value

    runtimes, results = [], []
    for compiled in (False, True):
        monkeypatch.setattr(_PhysicalIO, "read_flash_pages", observed if compiled else lambda *_: None)
        runtime = PhysicalRuntimeContext(capture_details=False)
        if unrelated_aliases:
            # The live serving kernel always supplies resource_owners, even
            # when those aliases do not overlap this NAND/fabric path.
            runtime.timeline.resource_aliases = {"gpu.compute": "gpu.compute", "cpu.logical": "cpu.physical"}
        rows = []
        for access in accesses:
            # An in-flight write precedes the first stream. Repeated streams
            # and a later scalar cache hit then reuse its resulting state.
            rows.append(resolve_physical_task(access, runtime, 0.))
        runtimes.append(runtime)
        results.append(rows)
    assert compiled_calls == [16, 16]
    for reference, compiled in zip(*results):
        assert compiled.metadata["physical_completion_ns"] == reference.metadata["physical_completion_ns"]
        assert compiled.metadata["hbf_host_staging"] == reference.metadata["hbf_host_staging"]
        assert compiled.demands == reference.demands
        for owner, expected in reference.metadata["physical_execution_by_owner"].items():
            actual = compiled.metadata["physical_execution_by_owner"][owner]
            for key, value in expected.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert actual.get(key) == pytest.approx(value), (owner, key, actual.get(key), value)
                elif value is None:
                    assert actual.get(key) is None, (owner, key, actual.get(key))
    reference, compiled = runtimes
    assert compiled.hbf_staging["flash"]["controller"].snapshot() == reference.hbf_staging["flash"]["controller"].snapshot()
    for key in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "lane_available", "lane_directions", "directions"):
        assert getattr(compiled.timeline, key) == getattr(reference.timeline, key)
    for owner in reference.runtimes:
        if owner != "flash":
            continue
        for key in ("_array_ready", "_buffer_ready", "_next_program_page", "_acceptance_ns",
                    "_inflight", "_inflight_operations", "_read_cache", "_cache_use"):
            assert getattr(compiled.runtimes[owner].core, key) == getattr(reference.runtimes[owner].core, key)


def test_compiled_dynamic_pages_preserve_dirty_tail_overwrite_and_per_page_ready(monkeypatch):
    from heterollm_sim.hbf_staging_runtime import _PhysicalIO

    config, binding = setup(capacity=16384)
    config = replace(config, blocks_per_plane=128, capacity_bytes=2 * 1024 * 1024)
    size, page = 41 * 4096, 4096
    accesses = (
        task(config, binding, "write", count=256, size=size),
        task(config, binding, "write", count=40 * page, size=size),
        task(config, binding, "read", count=40 * page, size=size),
        task(config, binding, "write", offset=17 * page + 128, count=256, size=size),
        task(config, binding, "write", offset=40 * page, count=256, size=size),
        task(config, binding, "read", count=40 * page + 256, size=size),
        task(config, binding, "read", offset=64, count=40 * page + 128, size=size),
        task(config, binding, "write", count=40 * page, size=size),
        task(config, binding, "read", count=40 * page, size=size),
    )
    originals = (_PhysicalIO.read_flash_pages, _PhysicalIO.write_flash_pages)
    calls = [[], []]
    def observed(which):
        def submit(io, addresses, arrivals):
            result = originals[which](io, addresses, arrivals)
            if result is not None:
                calls[which].append(len(addresses))
            return result
        return submit

    results, runtimes = [], []
    for compiled in (False, True):
        for which, name in enumerate(("read_flash_pages", "write_flash_pages")):
            monkeypatch.setattr(_PhysicalIO, name, observed(which) if compiled else lambda *_: None)
        runtime = PhysicalRuntimeContext(capture_details=False)
        # Request arrival deliberately precedes already scheduled page work.
        results.append([resolve_physical_task(access, runtime, 0.) for access in accesses])
        runtimes.append(runtime)
    assert calls == [[40, 17, 22, 16, 22, 40], [40, 40]]
    for expected, actual in zip(*results):
        assert actual.metadata["physical_completion_ns"] == expected.metadata["physical_completion_ns"]
        assert actual.metadata["hbf_host_staging"] == expected.metadata["hbf_host_staging"]
        assert actual.demands == expected.demands
        for owner, row in expected.metadata["physical_execution_by_owner"].items():
            actual_row = actual.metadata["physical_execution_by_owner"][owner]
            for name, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert actual_row.get(name) == pytest.approx(value), (owner, name)
                elif value is None:
                    assert actual_row.get(name) is None, (owner, name)
    expected, actual = runtimes
    assert actual.hbf_staging["flash"]["controller"].snapshot() == expected.hbf_staging["flash"]["controller"].snapshot()
    for name in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "lane_available", "lane_directions", "directions"):
        assert getattr(actual.timeline, name) == getattr(expected.timeline, name), name
    for name in ("_array_ready", "_buffer_ready", "_next_program_page", "_acceptance_ns",
                 "_inflight", "_inflight_operations", "_read_cache", "_cache_use"):
        assert getattr(actual.runtimes["flash"].core, name) == getattr(expected.runtimes["flash"].core, name), name


def test_compiled_write_failure_restores_core_and_controller_allocation(monkeypatch):
    from heterollm_sim.hbf_staging_runtime import _PhysicalIO

    config, binding = setup()
    runtime = PhysicalRuntimeContext(capture_details=False)
    size = 40 * 4096
    resolve_physical_task(task(config, binding, "write", count=256, size=size), runtime, 0.)
    before = runtime.snapshot()
    original = _PhysicalIO.write_flash_pages
    def fail_after_media(io, addresses, arrivals):
        assert original(io, addresses, arrivals) is not None
        raise RuntimeError("injected after compiled program")
    monkeypatch.setattr(_PhysicalIO, "write_flash_pages", fail_after_media)
    with pytest.raises(RuntimeError, match="injected after compiled program"):
        resolve_physical_task(task(config, binding, "write", count=size, size=size), runtime, 0.)
    assert runtime.snapshot() == before


def test_gc_after_compiled_full_pages_preserves_live_mapping_and_timing(monkeypatch):
    from heterollm_sim.hbf_staging_runtime import _PhysicalIO

    config, binding = setup()
    original = _PhysicalIO.write_flash_pages
    calls, snapshots, ends = [], [], []
    def observed(io, addresses, arrivals):
        value = original(io, addresses, arrivals)
        if value is not None:
            calls.append(len(addresses))
        return value
    for compiled in (False, True):
        monkeypatch.setattr(_PhysicalIO, "write_flash_pages", observed if compiled else lambda *_: None)
        runtime = PhysicalRuntimeContext(capture_details=False)
        size = 40 * 4096  # 60 allocatable pages: the overwrite must collect.
        rows = [resolve_physical_task(task(config, binding, "write", count=size, size=size), runtime, 0.)
                for _ in range(2)]
        assert runtime.hbf_staging["flash"]["controller"].stats["gc_cycles"] > 0
        ends.append([row.metadata["physical_completion_ns"] for row in rows])
        snapshots.append(runtime.snapshot())
    assert calls == [40]
    assert ends[1] == ends[0]
    # Saved snapshots contain their runtime objects as well as copied state;
    # compare the exact controller, core and resource calendars separately.
    a, b = snapshots
    assert a["timeline"] == b["timeline"]
    assert a["hbf_staging"]["flash"]["saved_controller"] == b["hbf_staging"]["flash"]["saved_controller"]
    for owner in a["runtime_state"]:
        assert a["runtime_state"][owner]["core"] == b["runtime_state"][owner]["core"]
