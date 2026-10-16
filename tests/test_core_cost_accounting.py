"""Physical task admission, rollback and endpoint byte accounting."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_types import AccessRequest, make_hbm_config, make_hbf_config
from heterollm_sim.nand_core import NandCore


def task(config, accesses, **metadata):
    return TaskSpec("audit", "request", "physical audit", TaskCategory.MEMORY,
                    metadata={"physical_memory_config": config,
                              "physical_owner": "memory",
                              "memory_accesses": accesses, **metadata})


@pytest.mark.parametrize("existing_owner", [False, True])
def test_rejected_task_does_not_leave_allocations(existing_owner):
    config = make_hbm_config(capacity_bytes=1024 * 1024)
    runtime = PhysicalRuntimeContext()
    if existing_owner:
        runtime.runtime(config, "memory")
        runtime.allocators["memory"].allocate("keep", 64)
    rejected = task(config, [{"operation": "read", "address": 0,
                             "byte_count": config.capacity_bytes + 1}],
                    physical_allocations=[{"buffer_id": "leaked", "size_bytes": 1024}])
    with pytest.raises(ValueError, match="exceeds owner"):
        resolve_physical_task(rejected, runtime, 0.0)
    assert not runtime.timeline.ready_ns
    if existing_owner:
        assert [a.buffer_id for a in runtime.allocators["memory"].allocations()] == ["keep"]
    else:
        assert not runtime.runtimes
        assert not runtime.allocators


def test_late_rejected_access_rolls_back_executed_core_and_allocation():
    config = make_hbf_config(capacity_bytes=4 * 16384, partial_page_policy="reject")
    runtime = PhysicalRuntimeContext()
    core = runtime.runtime(config, "memory").core
    rejected = task(config, [
        {"operation": "read", "address": 0, "byte_count": 16384},
        {"operation": "write", "address": 16384, "byte_count": 1},
    ], physical_allocations=[{"buffer_id": "leaked", "size_bytes": 16384}])
    with pytest.raises(ValueError, match="partial NAND page write"):
        resolve_physical_task(rejected, runtime, 0.0)
    assert not core._array_ready
    assert not core._inflight
    assert not runtime.timeline.ready_ns
    assert not runtime.timeline.bytes_moved
    assert not runtime.allocators["memory"].allocations()


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_restore_point_remains_valid_after_resumed_execution(medium):
    config = (make_hbm_config(capacity_bytes=1024 * 1024) if medium == "dram"
              else make_hbf_config(capacity_bytes=4 * 16384, write_buffer_bytes=16384))
    runtime = PhysicalRuntimeContext(config=config, physical_owner="memory")
    core = runtime.runtimes["memory"].core
    saved = runtime.snapshot()
    for _ in range(2):
        core.submit(AccessRequest("write", "write", 0, 64))
        runtime.restore(saved)
        assert not core._inflight
        if medium == "dram":
            assert not core._banks
        else:
            assert not core._staged_pages
            assert core.pending_write_buffer_bytes == 0
        assert not runtime.timeline.ready_ns
        assert not runtime.timeline.bytes_moved


def test_task_aggregates_buffer_waits_and_merges_instead_of_last_request():
    config = make_hbf_config(capacity_bytes=65536, page_bytes=4096,
                             host_granularity_bytes=1, write_buffer_bytes=2048)
    addresses = (0, 1024, 8192, 9216)
    accesses = [{"operation": "write", "address": address, "byte_count": 1024}
                for address in addresses]
    core = NandCore(config)
    expected = [core.submit(AccessRequest(str(i), "write", address, 1024))
                for i, address in enumerate(addresses)]
    resolved = resolve_physical_task(task(config, accesses), PhysicalRuntimeContext(), 0.0)
    expected_wait = sum(r.write_buffer_wait_ns for r in expected)
    assert expected_wait > 0
    for row in (next(iter(resolved.metadata["physical_execution_by_owner"].values())),
                resolved.metadata["physical_execution_by_owner"]["memory"]):
        assert row["write_buffer_wait_ns"] == pytest.approx(expected_wait)
        assert row["write_buffer_bytes"] == 4096
        assert row["write_buffer_peak_bytes"] == 2048
        assert row["merge_bytes"] == 2048
        assert row["staging_write_pages"] == 4
    assert resolved.metadata["physical_completion_ns"] == expected[-1].completion_ns


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_memory_bytes_are_charged_once_and_compute_demand_preserved(medium):
    if medium == "dram":
        config = make_hbm_config(capacity_bytes=1024 * 1024, burst_bytes=64)
        expected_media_bytes = 64
    else:
        config = make_hbf_config(capacity_bytes=4 * 16384, host_granularity_bytes=1)
        # A one-byte partial page write reads and programs one whole page.
        expected_media_bytes = 2 * 16384
    original = task(config, [{"operation": "write", "address": 0, "byte_count": 1,
                              "resource_id": "memory"}])
    original = replace(original, demands=(ResourceDemand("gpu:compute", 200),
                                          ResourceDemand("memory", 999999)))
    resolved = resolve_physical_task(original, PhysicalRuntimeContext(), 0.0)
    execution = next(iter(resolved.metadata["physical_execution_by_owner"].values()))
    assert execution["physical_bytes"] == expected_media_bytes
    assert resolved.demands[0] == ResourceDemand("gpu:compute", 200)
    assert all(d.resource_id != "memory" for d in resolved.demands)


def test_owner_config_cache_detects_nested_authored_metadata_changes():
    config = make_hbm_config(capacity_bytes=1024 * 1024,
                             metadata={"authored": {"value": 1}})
    runtime = PhysicalRuntimeContext(capture_details=False)
    active = runtime.runtime(config, "memory")
    assert runtime.runtime(config, "memory") is active
    config.metadata["authored"]["value"] = 2
    with pytest.raises(ValueError, match="multiple physical geometries"):
        runtime.runtime(config, "memory")
    assert active.core.config.metadata["authored"]["value"] == 1


def test_execution_config_cache_detects_mutated_input_metadata():
    raw = asdict(make_hbf_config(capacity_bytes=65536, metadata={"nested": {"value": 1}}))
    runtime = PhysicalRuntimeContext()
    first = runtime.parse_config(raw)
    assert runtime.parse_config(raw) is first
    raw["metadata"]["nested"]["value"] = 2
    updated = runtime.parse_config(raw)
    assert updated.metadata["nested"]["value"] == 2
    assert first.metadata["nested"]["value"] == 1


def test_execution_config_cache_does_not_conflate_boolean_and_integer():
    raw = asdict(make_hbf_config(capacity_bytes=65536, channels=1))
    runtime = PhysicalRuntimeContext()
    runtime.parse_config(raw)
    raw["channels"] = True
    with pytest.raises(ValueError, match="channels.*positive integer"):
        runtime.parse_config(raw)


def test_execution_config_cache_reparses_modified_returned_metadata():
    raw = asdict(make_hbf_config(capacity_bytes=65536, metadata={"nested": {"value": 1}}))
    runtime = PhysicalRuntimeContext()
    first = runtime.parse_config(raw)
    first.metadata["nested"]["value"] = 99
    assert raw["metadata"]["nested"]["value"] == 1
    restored = runtime.parse_config(raw)
    assert restored is not first
    assert restored.metadata["nested"]["value"] == 1
    assert runtime.parse_config(raw) is restored


def test_nand_cache_restore_preserves_slot_list_aliases_and_extensions():
    from heterollm_sim.nand_core import _ReadCacheSlot

    runtime = PhysicalRuntimeContext(config=make_hbf_config(capacity_bytes=65536), physical_owner="memory")
    core = runtime.runtimes["memory"].core
    first, second = _ReadCacheSlot((0, 1), 10., 20., 3), _ReadCacheSlot((2, 3), 30., 40., 4)
    shared = [first, second, first]
    core._read_cache = {(0,): shared, (1,): shared}
    extension = {"slot": second, "row": shared, "cache": core._read_cache, "values": [7]}
    extension["cycle"] = extension
    first.extension = second.extension = extension
    first.dictionary = vars(first)
    saved = runtime.snapshot()
    for _ in range(2):
        core._read_cache[(0,)][0].ready_ns = 99.
        core._read_cache[(0,)][0].extension["values"].append(8)
        runtime.restore(saved)
        cache = core._read_cache
        row = cache[(0,)]
        assert row is cache[(1,)] and row[0] is row[2]
        assert row[0].ready_ns == 10.
        assert row[0].extension is row[1].extension
        assert row[0].extension["slot"] is row[1]
        assert row[0].extension["row"] is row
        assert row[0].extension["cache"] is cache
        assert row[0].extension["cycle"] is row[0].extension
        assert row[0].extension["values"] == [7]
        assert row[0].dictionary is vars(row[0])


def test_small_physical_batches_preserve_overlap_dependencies(monkeypatch):
    import heterollm_sim.data_motion as module
    from test_hbm_column_pack_equivalence import b200_memory

    config = b200_memory()
    accesses = [
        {"operation": op, "address": addr, "byte_count": count}
        for op, addr, count in (
            ("read", 0, 65536), ("read", 65536, 65536),
            ("write", 0, 32768), ("write", 32768, 32768),
            ("read", 0, 65536), ("read", 16, 65536),
            ("write", 0, 65536), ("write", 16, 65536))]
    serial = PhysicalRuntimeContext(capture_details=False)
    batched = PhysicalRuntimeContext(capture_details=False)
    monkeypatch.setattr(module, "_USE_COMPILED_DRAM_BATCH", False)
    expected = resolve_physical_task(task(config, accesses), serial, 0.0)
    monkeypatch.setattr(module, "_USE_COMPILED_DRAM_BATCH", True)
    actual = resolve_physical_task(task(config, accesses), batched, 0.0)
    # This diagnostic counts submissions to submit_many, rather than a
    # physical command or a service duration. Three safe groups are batched.
    actual_bill = actual.metadata["physical_execution_by_owner"]["memory"]
    expected_bill = expected.metadata["physical_execution_by_owner"]["memory"]
    assert actual_bill["scheduler_batches"] == 3
    assert expected_bill["scheduler_batches"] == 0
    expected_bill["scheduler_batches"] = actual_bill["scheduler_batches"]
    assert actual == expected
    assert batched.snapshot()["timeline"] == serial.snapshot()["timeline"]
