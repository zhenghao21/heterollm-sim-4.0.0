"""Physical pricing keeps explicit state and rejects retired boundary values."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim import data_motion
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    AccessKind, PhysicalRuntimeContext, lower_physical_allocations,
    resolve_physical_task, resolve_service,
)
from heterollm_sim.ir import ComponentSpec, PortSpec
from heterollm_sim.memory_allocator import PhysicalAddressAllocator
from heterollm_sim.memory_types import make_hbm_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.topology import validate_topology
from heterollm_sim.workspace_memory import physical_workspace_bound


def service():
    config = make_hbm_config(capacity_bytes=4096)
    return resolve_service(ComponentSpec("memory", "hbm", capacity_bytes=4096,
        metadata={"physical_memory_config": config}))


@pytest.mark.parametrize("entry", ["single", "batch"])
def test_preview_is_independent_of_explicit_runtime_history(entry):
    endpoint = service()
    runtime = PhysicalRuntimeContext()
    def price(context=None):
        if entry == "single":
            result = endpoint.price(AccessKind.READ, 64, page_offset_bytes=0, runtime=context)
            return result["completion_ns"]
        result = endpoint.price_batch([{"operation": "read", "address": 0, "byte_count": 64}], runtime=context)
        return result["end_ns"]
    first_preview = price()
    first = price(runtime)
    second = price(runtime)
    saved = runtime.snapshot()
    assert price() == price() == first_preview == first
    assert second > first
    assert runtime.snapshot() == saved


@pytest.mark.parametrize("obsolete", ["page_offset_bytes", "offset_bytes", "offset", "size_bytes", "bytes"])
def test_batch_rejects_obsolete_fields_even_beside_canonical_address(obsolete):
    runtime = PhysicalRuntimeContext()
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        service().price_batch([{"operation": "read", "address": 0, "byte_count": 64,
                               obsolete: 128}], runtime=runtime)
    assert not runtime.runtimes and not runtime.allocators
    assert not runtime.timeline.bytes_moved


def buffer_task(**access_updates):
    return TaskSpec("read", "request", "read", TaskCategory.MEMORY, metadata={
        "physical_memory_config": make_hbm_config(capacity_bytes=4096),
        "physical_owner": "memory", "memory_accesses": ({
            "operation": "read", "buffer_id": "buffer", "byte_count": 64,
            "allocation_generation": 3, "offset_bytes": 0, **access_updates,
        },),
    })


@pytest.mark.parametrize("field", ["allocation_generation", "offset_bytes", "byte_count"])
def test_explicit_null_descriptor_does_not_become_generation_or_offset_zero(field):
    task = buffer_task(**{field: None})
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        lower_physical_allocations(task)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        physical_workspace_bound((task,))
    declarations = lower_physical_allocations(buffer_task()).metadata["physical_allocations"]
    task = replace(task, metadata={**task.metadata, "physical_allocations": declarations})
    runtime = PhysicalRuntimeContext()
    with pytest.raises(ValueError, match="must be a (non-negative|positive) integer"):
        resolve_physical_task(task, runtime, 0.)
    assert not runtime.allocators and not runtime.runtimes


def test_omitted_descriptor_defaults_and_large_generation_remain_valid():
    generation = 2**160 + 7
    task = lower_physical_allocations(buffer_task(allocation_generation=generation))
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(task, runtime, 0.)
    assert runtime.allocators["memory"].lookup("buffer", generation).size_bytes == 64
    zero = buffer_task()
    row = dict(zero.metadata["memory_accesses"][0])
    row.pop("allocation_generation")
    row.pop("offset_bytes")
    zero = lower_physical_allocations(replace(zero, metadata={**zero.metadata, "memory_accesses": (row,)}))
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(zero, runtime, 0.)
    assert runtime.allocators["memory"].lookup("buffer", 0).base_address == 0


def test_allocator_typed_and_serialized_snapshots_keep_real_alias_storage():
    allocator = PhysicalAddressAllocator(4096)
    allocator.allocate("root", 128, 4)
    allocator.allocate("view", 64, 8, alias_of="root", alias_generation=4, alias_offset_bytes=64)
    snapshot = allocator.snapshot()
    for rows in (snapshot["allocations"], tuple(asdict(row) for row in snapshot["allocations"])):
        restored = PhysicalAddressAllocator.from_snapshot({**snapshot, "allocations": rows})
        assert restored.canonical_range("view", 0, 64, 8) == ("root", 4, 64)
        assert restored.address("view", 0, 64, 8) == 64


@pytest.mark.parametrize("direction", ["in", "rx", "out", "tx", "bi", "bidir", "inout"])
def test_port_boundary_rejects_retired_direction_spellings(direction):
    with pytest.raises(ValueError, match="direction must be input, output, or bidirectional"):
        PortSpec("port", "PCIe", "endpoint", direction=direction)


def test_current_topology_and_physical_api_have_single_public_names():
    report = validate_topology(build_reference_scenario().hardware)
    assert report.is_valid, report.format()
    assert not hasattr(report, "ok")
    for name in ("READ", "WRITE", "COPY", "current_physical_runtime_context", "reset_physical_runtimes"):
        assert not hasattr(data_motion, name)
    assert not hasattr(service(), "bill")
