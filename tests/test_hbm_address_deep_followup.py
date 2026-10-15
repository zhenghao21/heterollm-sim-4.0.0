"""HBM physical identity, allocation admission and partition regressions."""
from dataclasses import asdict

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import lower_physical_allocations, PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_allocator import PhysicalAddressAllocator
from heterollm_sim.memory_types import make_hbm_config


def physical_task(access, declarations=(), **metadata):
    return TaskSpec("access", "request", "HBM address", TaskCategory.MEMORY, metadata={
        "physical_memory_config": make_hbm_config(capacity_bytes=1048576),
        "physical_owner": "hbm", "physical_allocations": declarations,
        "memory_accesses": ({"operation": "write", "byte_count": 64, **access},), **metadata})


@pytest.mark.parametrize("source", ["", "explicit", "explicit_buffer"])
@pytest.mark.parametrize("fault", ["address", "range", "generation"])
def test_explicit_buffer_access_cannot_bypass_identity_or_extent(source, fault):
    access = {"buffer_id": "kv", "address": 0, "address_source": source}
    if fault == "address":
        access["address"] = 8192
    elif fault == "range":
        access.update(offset_bytes=64, address=64)
    else:
        access["allocation_generation"] = 2
    runtime = PhysicalRuntimeContext()
    task = physical_task(access, ({"buffer_id": "kv", "size_bytes": 64, "address": 0},))
    with pytest.raises(ValueError):
        resolve_physical_task(lower_physical_allocations(task), runtime, 0.)
    assert not runtime.allocators and not runtime.runtimes
    assert not runtime.timeline.bytes_moved


@pytest.mark.parametrize("source", ["", "explicit"])
def test_explicit_address_derives_allocation_base_from_buffer_offset(source):
    runtime = PhysicalRuntimeContext()
    task = physical_task({"buffer_id": "kv", "address": 8320, "offset_bytes": 128,
                          "allocation_size_bytes": 256, "address_source": source})
    result = resolve_physical_task(lower_physical_allocations(task), runtime, 0.)
    allocation = runtime.allocators["hbm"].lookup("kv")
    assert allocation.base_address == 8192
    assert result.metadata["memory_accesses"][0]["address"] == allocation.address(128, 64)
    assert next(iter(result.metadata["physical_execution_by_owner"].values()))["physical_write_bytes"] == 64


def test_planner_preview_address_is_rebound_and_explicit_alias_is_preserved():
    runtime = PhysicalRuntimeContext()
    task = physical_task({"buffer_id": "view", "address": 100000, "address_source": "run_allocator"}, (
        {"buffer_id": "root", "size_bytes": 256, "address": 8192},
        {"buffer_id": "view", "size_bytes": 64, "alias_of": "root", "alias_offset_bytes": 128}))
    result = resolve_physical_task(lower_physical_allocations(task), runtime, 0.)
    assert result.metadata["memory_accesses"][0]["address"] == 8320


def test_failed_inferred_growth_does_not_mutate_allocation():
    allocator = PhysicalAddressAllocator(4096)
    before = allocator.allocate("kv", 64, inferred=True)
    with pytest.raises(ValueError, match="address changed"):
        allocator.allocate("kv", 128, inferred=True, address=512)
    assert allocator.lookup("kv") == before
    assert allocator.allocate("next", 64).base_address == 64


@pytest.mark.parametrize("json_snapshot", [False, True])
def test_resident_generation_is_independent_of_workspace_arena(json_snapshot):
    allocator = PhysicalAddressAllocator(32768, workspace_capacity_bytes=4096)
    kv = allocator.allocate("kv", 8192, generation=4, allocation_arena="resident")
    activation = allocator.allocate("activation", 4096, generation=1)
    alias = allocator.allocate("view", 64, generation=5, alias_of="kv", alias_generation=4)
    assert kv.base_address == 0 and activation.base_address == 28672
    assert alias.allocation_arena == "resident"
    saved = allocator.snapshot()
    if json_snapshot:
        saved["allocations"] = tuple(asdict(item) for item in saved["allocations"])
    restored = PhysicalAddressAllocator.from_snapshot(saved)
    assert restored.snapshot() == allocator.snapshot()
    assert restored.allocate("kv", 8192, generation=4) == kv
    with pytest.raises(ValueError, match="arena changed"):
        restored.allocate("kv", 8192, generation=4, allocation_arena="workspace")


def test_external_page_registration_stays_resident_after_generation_change():
    config = make_hbm_config(capacity_bytes=32768,
                             metadata={"workspace_capacity_bytes": 4096})
    task = physical_task({"buffer_id": "kv", "allocation_generation": 4,
        "allocation_size_bytes": 8192}, physical_memory_config=config,
        physical_external_allocations=[["hbm", "kv", 4]])
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(lower_physical_allocations(task), runtime, 0.)
    allocation = runtime.allocators["hbm"].lookup("kv", 4)
    assert allocation.base_address == 0
    assert allocation.allocation_arena == "resident"


@pytest.mark.parametrize("request_owned", [False, True])
def test_versioned_resident_buffer_survives_cohort_but_request_can_release_it(request_owned):
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from dataclasses import replace
    from heterollm_sim.workspace_memory import physical_workspace_bound

    task = physical_task({"buffer_id": "persistent", "allocation_generation": 5,
        "allocation_size_bytes": 8192, "allocation_arena": "resident"},
        physical_memory_config=asdict(make_hbm_config(capacity_bytes=32768,
            metadata={"workspace_capacity_bytes": 4096})),
        persistent_request_buffers={"request": ("persistent",)} if request_owned else {})
    task = replace(task, request_id="cohort-1")
    bound = physical_workspace_bound((task,))
    assert bound["bytes_by_owner"] == {}
    assert bound["resident_bytes_by_owner"] == {"hbm": 8192}
    kernel = UnifiedEventKernel.from_closed_graph((lower_physical_allocations(task),))
    event = kernel.step()
    allocator = kernel.physical_runtime.allocators["hbm"]
    before = allocator.lookup("persistent", 5)
    assert before.base_address == 0
    assert event.task.metadata["memory_accesses"][0]["address"] == 0
    kernel.release_request_physical_allocations("request")
    assert (allocator.get_allocation("persistent", 5) is None) == request_owned


def test_explicit_workspace_version_zero_is_released_after_last_consumer():
    from dataclasses import replace
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from heterollm_sim.workspace_memory import physical_workspace_bound

    task = replace(physical_task({"buffer_id": "temporary", "allocation_generation": 0,
        "allocation_size_bytes": 64, "allocation_arena": "workspace"},
        physical_memory_config=asdict(make_hbm_config(capacity_bytes=32768,
            metadata={"workspace_capacity_bytes": 4096}))), request_id="cohort-1")
    bound = physical_workspace_bound((task,))
    assert bound["bytes_by_owner"] == {"hbm": 64}
    assert bound["resident_bytes_by_owner"] == {}
    kernel = UnifiedEventKernel.from_closed_graph((lower_physical_allocations(task),))
    event = kernel.step()
    assert event.task.metadata["memory_accesses"][0]["address"] == 28672
    assert kernel.physical_runtime.allocators["hbm"].get_allocation("temporary", 0) is None


@pytest.mark.parametrize("resident_first", [False, True])
def test_duplicate_ranges_merge_resident_declaration_independent_of_order(resident_first):
    access = {"buffer_id": "kv", "operation": "read", "byte_count": 64,
              "allocation_generation": 1, "allocation_size_bytes": 8192}
    resident = {**access, "allocation_arena": "resident"}
    task = physical_task({}, physical_memory_config=make_hbm_config(capacity_bytes=16384,
        metadata={"workspace_capacity_bytes": 4096}),
        memory_accesses=(resident, access) if resident_first else (access, resident))
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(lower_physical_allocations(task), runtime, 0.)
    allocation = runtime.allocators["hbm"].lookup("kv", 1)
    assert allocation.allocation_arena == "resident" and allocation.base_address == 0


@pytest.mark.parametrize("field,values", [("allocation_arena", ("resident", "workspace")),
                                        ("address", (0, 8192))])
def test_duplicate_physical_identity_conflict_is_rejected_atomically(field, values):
    access = {"buffer_id": "kv", "operation": "read", "byte_count": 64,
              "allocation_generation": 1, "allocation_size_bytes": 128}
    task = physical_task({}, memory_accesses=tuple({**access, field: value} for value in values))
    runtime = PhysicalRuntimeContext()
    with pytest.raises(ValueError, match="conflicting physical allocation"):
        resolve_physical_task(lower_physical_allocations(task), runtime, 0.)
    assert not runtime.allocators and not runtime.runtimes


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_l2_ranges_keep_one_allocation_base(reverse):
    from dataclasses import replace
    from heterollm_sim.cache_state import CacheAccess
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from test_native_kernel_physical_cost import native_task

    task, _ = native_task(1000., l2=True)
    accesses = tuple({"buffer_id": "kv", "operation": "read", "offset_bytes": offset,
        "byte_count": 64, "allocation_size_bytes": 128, "address": 8192 + offset,
        "address_source": "explicit_physical_address"} for offset in (0, 64))
    if reverse:
        accesses = tuple(reversed(accesses))
    contract = {**task.metadata["stateful_l2"], "accesses": tuple(
        asdict(CacheAccess.read("kv", offset, 64, buffer_size_bytes=128)) for offset in (0, 64))}
    task = replace(task, metadata={**task.metadata, "physical_allocations": (), "memory_accesses": accesses,
                                   "stateful_l2": contract})
    kernel = UnifiedEventKernel.from_closed_graph((lower_physical_allocations(task),))
    event = kernel.step()
    allocation = kernel.physical_runtime.allocators[task.metadata["physical_owner"]].lookup("kv")
    assert allocation.base_address == 8192
    assert {row["address"] for row in event.task.metadata["memory_accesses"]} == {8192, 8256}
    assert next(iter(event.task.metadata["physical_execution_by_owner"].values()))["physical_read_bytes"] == 128
