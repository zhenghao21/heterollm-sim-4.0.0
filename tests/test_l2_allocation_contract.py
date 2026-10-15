"""L2 dispatch consumes compiled allocations, including prior dirty aliases."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.event_kernel import UnifiedEventKernel
from test_native_kernel_physical_cost import native_task


def test_physical_l2_does_not_invent_missing_allocations():
    task, _ = native_task(1000., l2=True)
    task = replace(task, metadata={key: value for key, value in task.metadata.items()
                                   if key != "physical_allocations"})
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    with pytest.raises(ValueError, match="no physical allocator"):
        kernel.step()
    assert not kernel.physical_runtime.allocators
    assert not kernel._l2_states


def test_alias_dirty_eviction_uses_existing_root_without_new_declarations():
    template, _ = native_task(1000., l2=True)
    generation = 2**120 + 4
    declarations = (
        {"buffer_id": "root", "size_bytes": 256, "generation": generation},
        {"buffer_id": "view", "size_bytes": 64, "generation": generation + 1,
         "alias_of": "root", "alias_generation": generation, "alias_offset_bytes": 128},
    )

    def task(identity, access, allocations, dependencies=()):
        contract = {**template.metadata["stateful_l2"], "capacity_bytes": 64,
                    "accesses": (asdict(access),)}
        return replace(template, task_id=identity, dependencies=dependencies,
            metadata={**template.metadata, "stateful_l2": contract,
                "physical_allocations": allocations,
                "memory_accesses": ({"buffer_id": access.buffer_id,
                    "operation": access.operation, "byte_count": access.size_bytes,
                    "offset_bytes": access.offset_bytes,
                    "allocation_generation": access.allocation_generation},)})

    first = task("write-alias", CacheAccess.write("view", 0, 64,
        allocation_generation=generation + 1), declarations)
    second_declarations = ({"buffer_id": "other", "size_bytes": 64, "generation": 0},)
    second = task("evict-alias", CacheAccess.read("other", 0, 64),
                  second_declarations, (first.task_id,))
    kernel = UnifiedEventKernel.from_closed_graph((first, second))
    write, eviction = kernel.step(), kernel.step()
    canonical = write.task.metadata["l2_execution"]["accesses"][0]
    assert (canonical["buffer_id"], canonical["allocation_generation"], canonical["offset_bytes"]) == (
        "root", generation, 128)
    execution = eviction.task.metadata["l2_execution"]
    assert execution["dirty_eviction_bytes"] == execution["hbm_write_bytes"] == 64
    assert any(row["buffer_id"] == "root" and row["operation"] == "write"
               and row["allocation_generation"] == generation for row in execution["backing_accesses"])
    assert eviction.task.metadata["physical_allocations"] == second_declarations
    physical = next(iter(eviction.task.metadata["physical_execution_by_owner"].values()))
    assert physical["physical_write_bytes"] == 64
