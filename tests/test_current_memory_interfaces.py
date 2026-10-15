"""Physical access, allocation and transfer boundaries reject retired shapes."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    AccessKind, PhysicalRuntimeContext, lower_physical_allocations,
    resolve_physical_task, resolve_service,
)
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_allocator import AllocationError, PhysicalAddressAllocator
from heterollm_sim.memory_types import make_hbm_config


def physical_task(access, declarations=()):
    return TaskSpec("read", "request", "read", TaskCategory.MEMORY, metadata={
        "physical_memory_config": make_hbm_config(capacity_bytes=4096),
        "physical_owner": "hbm", "memory_accesses": (access,),
        "physical_allocations": declarations,
    })


@pytest.mark.parametrize("obsolete", ["generation", "size_bytes", "offset", "tensor_id", "buffer_size_bytes"])
def test_physical_access_rejects_aliases_even_beside_canonical_fields(obsolete):
    access = {"operation": "read", "buffer_id": "buffer", "byte_count": 64,
              "offset_bytes": 0, "allocation_generation": 7, "allocation_size_bytes": 64,
              obsolete: "other" if obsolete == "tensor_id" else 1}
    task = physical_task(access, ({"buffer_id": "buffer", "size_bytes": 64, "generation": 7},))
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        lower_physical_allocations(task)
    runtime = PhysicalRuntimeContext()
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        resolve_physical_task(task, runtime, 0.)
    assert not runtime.allocators and not runtime.runtimes


@pytest.mark.parametrize("obsolete", ["allocation_generation", "allocation_size_bytes", "buffer_size_bytes", "base_address"])
def test_allocation_declaration_rejects_old_fields(obsolete):
    task = physical_task({"operation": "read", "address": 0, "byte_count": 64},
        ({"buffer_id": "buffer", "size_bytes": 64, "generation": 7, "address": 0, obsolete: 1},))
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        lower_physical_allocations(task)


@pytest.mark.parametrize("obsolete", ["allocation_generation", "allocation_size_bytes"])
def test_allocator_serialized_records_keep_base_address_but_reject_aliases(obsolete):
    allocator = PhysicalAddressAllocator(4096)
    allocation = allocator.register({"buffer_id": "buffer", "size_bytes": 64,
                                     "generation": 7, "base_address": 128})
    assert allocation.base_address == 128 and allocation.generation == 7
    row = {**asdict(allocation), obsolete: 1}
    with pytest.raises(AllocationError, match="unsupported fields: " + obsolete):
        allocator.register(row)
    with pytest.raises(AllocationError, match="unsupported fields: " + obsolete):
        PhysicalAddressAllocator.from_snapshot({**allocator.snapshot(), "allocations": (row,)})


def test_transfer_requires_component_sequence_and_does_not_retry_router_errors():
    with pytest.raises(ValueError, match="sequence of ComponentSpec"):
        PhysicalKVTransferExecutor({"components": ()})
    with pytest.raises(ValueError, match="sequence of ComponentSpec"):
        PhysicalKVTransferExecutor(SimpleNamespace(hardware=SimpleNamespace(components=())))
    calls = []
    def route(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("router execution failed")
    executor = PhysicalKVTransferExecutor((), topology=SimpleNamespace(route=route))
    with pytest.raises(TypeError, match="router execution failed"):
        executor._route("source", "target", 64)
    assert calls == [(("source", "target", 64), {})]


@pytest.mark.parametrize("obsolete,value", [("transaction_bytes", 64), ("service_model", "serialized")])
def test_physical_price_rejects_unused_aggregate_cost_arguments(obsolete, value):
    config = make_hbm_config(capacity_bytes=4096)
    component = ComponentSpec("hbm", "hbm", capacity_bytes=config.capacity_bytes,
                              metadata={"physical_memory_config": config})
    service = resolve_service(component)
    with pytest.raises(TypeError, match=obsolete):
        service.price(AccessKind.READ, 64, page_offset_bytes=0, **{obsolete: value})
