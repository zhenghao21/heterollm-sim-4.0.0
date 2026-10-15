"""Execution accounting consumes only the current physical record contracts."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import physical_task_demands
from heterollm_sim.engine import _task_resource_intervals
from heterollm_sim.memory_transfer import summarize_batch
from heterollm_sim.memory_types import BatchResult, Operation, StageTiming, TransactionResult, make_hbm_config
from heterollm_sim.workspace_memory import physical_workspace_bound


def task(metadata, demands=()):
    return TaskSpec("task", "request", "memory", TaskCategory.MEMORY,
                    metadata=metadata, demands=demands)


def execution_metadata(stages):
    return {
        "physical_execution_by_owner": {"memory": {
            "resource_interval_payloads": {"wire": ((1., 2., 64),)},
        }},
        "physical_resource_intervals": stages,
        "physical_nonmemory_demand_ids": ("compute",),
    }


def test_stage_records_deduplicate_payload_copies_but_keep_parallel_work():
    stage = StageTiming("data", 1., 2., "wire", 64)
    current = task(execution_metadata((stage, stage, StageTiming("wait", 0., 1.))),
                   (ResourceDemand("wire", 2., bytes_moved=128),
                    ResourceDemand("compute", 3.)))
    intervals = _task_resource_intervals(current, current.demands, 0.)
    assert [(row.resource_id, row.start_ns, row.end_ns, row.bytes_moved)
            for row in intervals] == [
                ("wire", 1., 2., 64), ("wire", 1., 2., 64), ("compute", 0., 3., 0)]


@pytest.mark.parametrize("old", [
    ("wire", 1., 2.), ("wire", 1., 2., 64),
    SimpleNamespace(resource_id="wire", start_ns=1., end_ns=2.),
    SimpleNamespace(resource_id="wire", start_ns=1., end_ns=2., bytes=64),
])
def test_stage_records_reject_retired_sequences_and_partial_objects(old):
    current = task(execution_metadata((old,)))
    with pytest.raises(ValueError, match="requires StageTiming"):
        _task_resource_intervals(current, current.demands, 0.)


def test_owner_records_require_explicit_interval_payloads():
    current = task({"physical_execution_by_owner": {"memory": {
        "resource_intervals": {"wire": ((1., 2.),)}, "resource_bytes": {"wire": 64}}}})
    with pytest.raises(ValueError, match="require resource_interval_payloads"):
        _task_resource_intervals(current, current.demands, 0.)


def workspace_metadata():
    return {"physical_owner": "memory", "physical_memory_config": make_hbm_config(capacity_bytes=4096)}


@pytest.mark.parametrize("obsolete", ["tensor_id", "generation", "offset", "size_bytes", "buffer_size_bytes", "physical_resource_id"])
def test_capacity_and_demand_projection_reject_old_access_fields(obsolete):
    current = task({**workspace_metadata(), "memory_accesses": ({
        "buffer_id": "buffer", "byte_count": 64, "allocation_generation": 1,
        obsolete: "old" if obsolete in {"tensor_id", "physical_resource_id"} else 1,
    },)})
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        physical_workspace_bound((current,))
    with pytest.raises(ValueError, match="unsupported fields: " + obsolete):
        physical_task_demands(current)


@pytest.mark.parametrize("field,record", [
    ("memory_accesses", {"buffer_id": "buffer", "byte_count": 64}),
    ("physical_allocations", {"buffer_id": "buffer", "size_bytes": 64}),
    ("stateful_l2", {"accesses": {"buffer_id": "buffer", "size_bytes": 64}}),
])
def test_capacity_rejects_single_mapping_descriptors(field, record):
    with pytest.raises(ValueError, match="must be a sequence"):
        physical_workspace_bound((task({**workspace_metadata(), field: record}),))


def test_capacity_explicitly_projects_cache_access_and_allocation_sizes():
    generation = 2 ** 120
    current = task({**workspace_metadata(),
        "physical_allocations": ({"buffer_id": "buffer", "size_bytes": 256, "generation": generation},),
        "memory_accesses": (
            {"buffer_id": "buffer", "byte_count": 64, "allocation_generation": generation},
            {"buffer_id": "buffer", "byte_count": 64, "allocation_size_bytes": 64,
             "allocation_generation": generation + 1},
        ),
        "stateful_l2": {"line_bytes": 128, "memory_resource": "memory", "accesses": (
            asdict(CacheAccess.read("buffer", 0, 64, allocation_generation=generation)),
        )},
    })
    bound = physical_workspace_bound((current,))
    assert bound["bytes_by_owner"] == {"memory": 320}
    assert bound["buffer_count"] == 2


def test_capacity_does_not_borrow_another_owners_geometry():
    current = task({**workspace_metadata(),
        "physical_memory_configs": {"other": make_hbm_config(capacity_bytes=4096)},
        "memory_accesses": ({"buffer_id": "buffer", "byte_count": 64, "allocation_generation": 1},),
    })
    with pytest.raises(ValueError, match="memory has no declared configuration"):
        physical_workspace_bound((current,))


def test_internal_transfer_bytes_require_their_own_counter():
    transaction = TransactionResult("request", Operation.READ, 0., 1., 64, 128)
    batch = BatchResult((transaction,), 0., 1., 64, 128)
    for result in (transaction, batch):
        with pytest.raises(KeyError, match="internal_transfer_bytes"):
            _ = result.internal_transfer_bytes
    assert summarize_batch(()).internal_transfer_bytes == 0
