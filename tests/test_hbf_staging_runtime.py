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


def test_cached_link_demands_recheck_parameters_aliases_and_actual_lane_capacity():
    from types import SimpleNamespace
    from heterollm_sim.hbf_staging_runtime import _PhysicalIO
    from heterollm_sim.memory_transfer import ResourceTimeline

    _, binding = setup()
    runtime = SimpleNamespace(timeline=ResourceTimeline())
    def route():
        return _PhysicalIO(runtime, "flash", {}, binding, [])._route("to_host", 256)

    first = route()
    assert first == (("to_host", "to_host", (("to_host", 8., 256),
                                            ("to_host.inflight", 88., 0))),)
    assert route() == first
    binding["routes"]["to_host"][0]["bandwidth_gb_s"] = 16.
    assert route()[0][2][0][1] == 16.
    runtime.timeline.resource_aliases["to_host"] = "shared-wire"
    assert route()[0][2][0] == ("shared-wire", 16., 256)
    runtime.timeline.lane_available["to_host.inflight"] = [0.]
    with pytest.raises(ValueError, match="queue conflicts"):
        route()
    binding["routes"]["to_host"][0]["queue_depth"] = 1
    route()
    # bool and int compare equal as cache keys, but only int is a valid depth.
    binding["routes"]["to_host"][0]["queue_depth"] = True
    with pytest.raises(ValueError, match="queue_depth"):
        route()


def test_summary_link_retention_keeps_all_counters_and_calendar_state(monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    import heterollm_sim.hbf_staging_runtime as staging
    from heterollm_sim.memory_transfer import ResourceTimeline

    _, binding = setup()
    results, states = [], []
    for detailed in (True, False):
        runtime = SimpleNamespace(timeline=ResourceTimeline(), capture_details=detailed)
        records = []
        io = staging._PhysicalIO(runtime, "flash", {}, binding, records)
        if not detailed:
            def forbidden_stage(*_args, **_kwargs):
                raise AssertionError("summary constructed a detailed link stage")
            monkeypatch.setattr(staging, "StageTiming", forbidden_stage)
        for route, size, arrival in (("to_host", 256, 0.), ("to_host", 512, 2.),
                                     ("from_host", 4096, 3.), ("to_host", 0, 7.)):
            io.transfer(route, size, arrival)
        results.append([row[-1] for row in records])
        states.append(deepcopy(vars(runtime.timeline)))
    assert states[0] == states[1]
    for detailed, summary in zip(*results):
        assert not summary.stages
        assert summary.counters["resource_intervals"] == {}
        assert summary.counters["resource_interval_payloads"] == {}
        assert summary.counters["intervals_truncated"] == bool(detailed.stages)
        excluded = {"resource_intervals", "resource_interval_payloads", "intervals_truncated"}
        assert replace(detailed, stages=(), counters={
            key: value for key, value in detailed.counters.items() if key not in excluded}) == replace(
            summary, counters={key: value for key, value in summary.counters.items() if key not in excluded})


def test_summary_link_retention_preserves_kernel_events_and_physical_state():
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from heterollm_sim.live_execution_metrics import LiveCohortMetrics

    config, binding = setup()
    tasks = tuple(replace(task(config, binding, op, offset=offset, count=count),
        task_id=str(index), dependencies=(str(index - 1),) if index else ())
        for index, (op, offset, count) in enumerate((
            ("write", 0, 256), ("read", 0, 256), ("write", 256, 256),
            ("read", 0, 512), ("write", 0, 4096), ("read", 0, 4096))))
    kernels, event_sets, summaries, metadata = [], [], [], []
    for detailed in (True, False):
        kernel = UnifiedEventKernel.from_closed_graph(tasks, capture_physical_details=detailed)
        metric = LiveCohortMetrics()
        events = []
        while kernel.has_active_tasks:
            item = kernel.step()
            metric.observe(item)
            events.append(item)
        kernels.append(kernel)
        event_sets.append(events)
        summaries.append(metric.summary())
        metadata.append(metric.metadata())
    for detailed, summary in zip(*event_sets):
        assert not summary.task.metadata["physical_resource_intervals"]
        assert summary.start_ns == detailed.start_ns
        assert summary.end_ns == detailed.end_ns
        assert summary.demands == detailed.demands
        assert summary.resource_predecessors == detailed.resource_predecessors
        assert summary.resource_lanes == detailed.resource_lanes
        assert summary.task.metadata["physical_resource_last_intervals"] == detailed.task.metadata["physical_resource_last_intervals"]
        for owner, row in detailed.task.metadata["physical_execution_by_owner"].items():
            ignored = {"resource_intervals", "resource_interval_payloads", "details_truncated", "intervals_truncated"}
            assert {key: value for key, value in row.items() if key not in ignored} == {
                key: value for key, value in summary.task.metadata["physical_execution_by_owner"][owner].items()
                if key not in ignored}
    assert replace(summaries[0], execution_records=()) == replace(summaries[1], execution_records=())
    assert metadata[0] == metadata[1]
    old, new = (kernel.physical_runtime for kernel in kernels)
    assert old.hbf_staging["flash"]["controller"].snapshot() == new.hbf_staging["flash"]["controller"].snapshot()
    for key in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "lane_available", "lane_directions", "directions"):
        assert getattr(old.timeline, key) == getattr(new.timeline, key)
    assert old.runtimes["host"].core._banks == new.runtimes["host"].core._banks
    for key in ("_array_ready", "_buffer_ready", "_next_program_page", "_acceptance_ns",
                "_inflight", "_inflight_operations", "_read_cache", "_cache_use"):
        assert getattr(old.runtimes["flash"].core, key) == getattr(new.runtimes["flash"].core, key)


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
    original_read = _PhysicalIO.read_flash_pages
    calls, snapshots, ends = [], [], []
    def observed(io, addresses, arrivals):
        value = original(io, addresses, arrivals)
        if value is not None:
            calls.append(len(addresses))
        return value
    for compiled in (False, True):
        monkeypatch.setattr(_PhysicalIO, "write_flash_pages", observed if compiled else lambda *_: None)
        monkeypatch.setattr(_PhysicalIO, "read_flash_pages", original_read if compiled else lambda *_: None)
        runtime = PhysicalRuntimeContext(capture_details=False)
        size = 40 * 4096  # 60 allocatable pages: the overwrite must collect.
        rows = [resolve_physical_task(task(config, binding, operation, count=size, size=size), runtime, 0.)
                for operation in ("write", "read", "write", "read")]
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


def test_numeric_ready_read_failure_restores_media_and_cached_controller(monkeypatch):
    from heterollm_sim.hbf_staging import HbfStagingController

    config, binding = setup()
    runtime = PhysicalRuntimeContext(capture_details=False)
    size = 40 * 4096
    for operation in ("write", "read"):
        resolve_physical_task(task(config, binding, operation, count=size, size=size), runtime, 0.)
    assert runtime.hbf_staging["flash"]["controller"]._read_states
    before = runtime.snapshot()
    original = HbfStagingController._read_full_pages
    def fail_after_controller(self, *args):
        assert original(self, *args) is not None
        raise RuntimeError("injected after numeric ready update")
    monkeypatch.setattr(HbfStagingController, "_read_full_pages", fail_after_controller)
    with pytest.raises(RuntimeError, match="numeric ready update"):
        resolve_physical_task(task(config, binding, "read", count=size, size=size), runtime, 0.)
    assert runtime.snapshot() == before


@pytest.mark.parametrize("overlap", [False, True])
def test_host_task_scope_keeps_interleaved_link_aliases_and_all_operation_records(overlap):
    from heterollm_sim import hbf_staging_runtime as staging

    config, binding = setup()
    old, new = (PhysicalRuntimeContext(capture_details=False) for _ in range(2))
    for runtime in (old, new):
        staging.configure(runtime, "flash", config, binding)
        if overlap:
            host = runtime.runtimes["host"].core
            prefix = host.config.metadata.get("data_resource_prefix", "dram:data")
            runtime.timeline.resource_aliases["from_host"] = prefix + ":0"
    records = [[], []]
    ios = [staging._PhysicalIO(runtime, "flash", runtime.hbf_staging["flash"], binding, rows)
           for runtime, rows in zip((old, new), records)]
    token = staging.begin_task(new)
    for index in range(6):
        expected = ios[0].write_host(index * 256, 256, index * .25)
        assert ios[1].write_host(index * 256, 256, index * .25) == expected
        if index == 3:
            expected = ios[0].transfer("from_host", 256, 7.)
            assert ios[1].transfer("from_host", 256, 7.) == expected
        expected = ios[0].read_host(index * 256, 256, index * .125)
        assert ios[1].read_host(index * 256, 256, index * .125) == expected
    staging.end_task(new, token)
    assert records[0] == records[1]
    expected, actual = old.snapshot(), new.snapshot()
    assert actual["timeline"] == expected["timeline"]
    assert actual["allocator_state"] == expected["allocator_state"]
    for owner in old.runtimes:
        assert actual["runtime_state"][owner]["core"] == expected["runtime_state"][owner]["core"]
        assert actual["runtime_state"][owner]["clock_ns"] == expected["runtime_state"][owner]["clock_ns"]
    assert actual["hbf_staging"]["flash"]["saved_controller"] == expected["hbf_staging"]["flash"]["saved_controller"]
    # Scope publication updates the original lane lists in place.
    slots = new.timeline.lane_available
    retained = {key: value for key, value in slots.items()}
    token = staging.begin_task(new)
    ios[1].write_host(0, 256, 1000.)
    staging.end_task(new, token)
    assert all(slots[key] is value for key, value in retained.items())


def test_failed_host_task_discards_pending_arrays_before_existing_transaction_rollback(monkeypatch):
    from heterollm_sim import hbf_staging_runtime as staging, _dram_numeric
    config, binding = setup()
    runtime = PhysicalRuntimeContext(capture_details=False)
    state = staging.configure(runtime, "flash", config, binding)
    io = staging._PhysicalIO(runtime, "flash", state, binding, [])
    io.write_host(0, 256, 0.)
    before = runtime.snapshot()
    transaction = runtime.begin_transaction()
    token = staging.begin_task(runtime)
    io.write_host(256, 256, 0.)
    def fail(*args):
        raise RuntimeError("injected HOST failure")
    monkeypatch.setattr(_dram_numeric, "run_numeric_short", fail)
    with pytest.raises(RuntimeError, match="injected HOST failure"):
        io.read_host(0, 256, 1.)
    staging.end_task(runtime, token, success=False)
    runtime.restore(transaction)
    assert runtime.snapshot() == before

@pytest.mark.parametrize("bulk", (False, True))
@pytest.mark.parametrize("overlap", ("none", "media_alias", "link_alias", "shared_name", "lane_alias"))
def test_host_scope_survives_only_disjoint_media_and_preserves_every_record(bulk, overlap):
    from heterollm_sim import hbf_staging_runtime as staging
    from heterollm_sim.memory_mapping import map_nand_address
    from heterollm_sim.memory_types import Operation

    config, binding = setup()
    if overlap == "shared_name":
        config = replace(config, metadata={**config.metadata, "host_resource_id": "shared-wire"})
        host = binding["host_config"]
        binding = {**binding, "host_config": replace(host, metadata={**host.metadata,
            "shared_command_resource": config.metadata["host_resource_id"] + ":read"})}
    runtimes = [PhysicalRuntimeContext(capture_details=False) for _ in range(2)]
    records, ios = [[], []], []
    for runtime, rows in zip(runtimes, records):
        state = staging.configure(runtime, "flash", config, binding)
        io = staging._PhysicalIO(runtime, "flash", state, binding, rows)
        ios.append(io)
        host = runtime.runtimes["host"].core
        media = runtime.runtimes["flash"].core
        host_data = str(host.config.metadata.get("data_resource_prefix", "dram:data")) + ":0"
        media_host = media._host_id(map_nand_address(config, 0), Operation.READ)
        io.write_host(0, 256, 0.)
        if overlap == "media_alias":
            runtime.timeline.resource_aliases[media_host] = host_data
        elif overlap == "link_alias":
            runtime.timeline.resource_aliases["read_hbf"] = host_data
        elif overlap == "lane_alias":
            runtime.timeline.lane_available[media_host] = runtime.timeline.lane_available[host_data]
    old, new = runtimes
    token = staging.begin_task(new)
    assert ios[1].write_host(256, 256, 1.) == ios[0].write_host(256, 256, 1.)
    host_scope = token.cores[new.runtimes["host"].core]
    pending = host_scope.workspace
    assert pending is not None
    completions = []
    for io in ios:
        if bulk:
            value = io.read_flash_pages([0, 4096], 2.)
            assert value is not None
            completions.append(value.tolist())
        else:
            end = io.read_flash(0, 4096, 2.)
            completions.append(io.transfer("read_hbf", 4096, end))
    assert completions[0] == completions[1]
    assert (host_scope.workspace is pending) if overlap == "none" else (host_scope.workspace is None)
    assert ios[1].read_host(0, 256, 3.) == ios[0].read_host(0, 256, 3.)
    staging.end_task(new, token)
    assert records[0] == records[1]
    expected, actual = old.snapshot(), new.snapshot()
    assert actual["timeline"] == expected["timeline"]
    for owner in old.runtimes:
        assert actual["runtime_state"][owner]["core"] == expected["runtime_state"][owner]["core"]
        assert actual["runtime_state"][owner]["clock_ns"] == expected["runtime_state"][owner]["clock_ns"]
    if overlap == "lane_alias":
        assert new.timeline.lane_available[media_host] is new.timeline.lane_available[host_data]


def test_disjoint_page_program_keeps_pending_host_and_failure_still_rolls_back():
    from heterollm_sim import hbf_staging_runtime as staging

    config, binding = setup()
    runtime = PhysicalRuntimeContext(capture_details=False)
    state = staging.configure(runtime, "flash", config, binding)
    io = staging._PhysicalIO(runtime, "flash", state, binding, [])
    io.write_host(0, 256, 0.)
    before = runtime.snapshot()
    transaction = runtime.begin_transaction()
    token = staging.begin_task(runtime)
    io.write_host(256, 256, 1.)
    host_scope = token.cores[runtime.runtimes["host"].core]
    pending = host_scope.workspace
    assert io.write_flash_pages([0, 4096], 2.) is not None
    assert host_scope.workspace is pending
    io.write_flash(8192, 4096, 3.)
    assert host_scope.workspace is pending
    staging.end_task(runtime, token, success=False)
    runtime.restore(transaction)
    assert runtime.snapshot() == before
