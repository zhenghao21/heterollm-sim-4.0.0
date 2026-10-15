"""Public physical pricing is atomic and charges shared wires by owner."""
from unittest.mock import patch

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import AccessKind, PhysicalRuntimeContext, resolve_service, resolve_physical_task
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config
from heterollm_sim.nand_core import NandCore


def service():
    config = make_hbf_config(capacity_bytes=65536, partial_page_policy="reject")
    component = ComponentSpec("hbf", "hbf", capacity_bytes=config.capacity_bytes,
        read_bandwidth_gbps=800, write_bandwidth_gbps=80,
        metadata={"physical_memory_config": config})
    return resolve_service(component)


@pytest.mark.parametrize("entry", ["price", "price_batch"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_public_submission_restores_runtime_even_after_core_execution(entry, existing):
    endpoint = service()
    runtime = PhysicalRuntimeContext()
    valid = {"operation": "read", "address": 0, "byte_count": 16384}
    if existing:
        endpoint.price_batch([valid], runtime=runtime)
    saved = runtime.snapshot()
    original = NandCore.submit
    def fail_after_submit(core, request):
        original(core, request)
        raise ValueError("injected post-execution failure")
    with patch.object(NandCore, "submit", fail_after_submit):
        with pytest.raises(ValueError, match="post-execution"):
            if entry == "price":
                endpoint.price(AccessKind.READ, 16384, page_offset_bytes=0, runtime=runtime)
            else:
                endpoint.price_batch([valid], runtime=runtime)
    assert runtime.snapshot() == saved
    # A retry must observe precisely the original queue, without phantom work.
    first = endpoint.price_batch([valid], runtime=runtime)
    runtime.restore(saved)
    retry = endpoint.price_batch([valid], runtime=runtime)
    assert first == retry


def test_invalid_batch_request_does_not_register_a_new_runtime():
    runtime = PhysicalRuntimeContext()
    with pytest.raises(ValueError, match="partial NAND page write"):
        service().price_batch([
            {"operation": "read", "address": 0, "byte_count": 16384},
            {"operation": "write", "address": 16384, "byte_count": 1},
        ], runtime=runtime)
    assert not runtime.runtimes and not runtime.allocators
    assert not runtime.timeline.bytes_moved


def test_batch_requires_runtime_and_address_without_aliases():
    runtime = PhysicalRuntimeContext()
    with pytest.raises(TypeError, match="state"):
        service().price_batch([], state=runtime)
    with pytest.raises(ValueError, match="unsupported fields: page_offset_bytes"):
        service().price_batch([{"operation": "read", "page_offset_bytes": 0,
                               "byte_count": 16384}], runtime=runtime)
    assert not runtime.runtimes


@pytest.mark.parametrize("front_ns", [0., 800.])
def test_explicit_zero_cost_erase_preserves_operation_without_inventing_bytes(front_ns):
    config = make_hbf_config(capacity_bytes=65536, block_erase_ns=0., front_ns=front_ns)
    task = TaskSpec("erase", "request", "zero-cost erase", TaskCategory.MEMORY,
        metadata={"physical_memory_config": config, "physical_owner": "nand",
            "memory_accesses": ({"operation": "erase", "address": 0, "byte_count": 16384},)})
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    assert resolved.demands == ()
    result = next(iter(resolved.metadata["physical_execution_by_owner"].values()))
    assert result["erase_operations"] == 1
    assert result["physical_bytes"] == 0
    assert result["service_ns"] == front_ns


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_shared_wire_preserves_each_owner_and_total_traffic(medium):
    if medium == "dram":
        config = make_hbm_config(capacity_bytes=1048576, data_lanes=1,
                                 metadata={"data_resource_prefix": "shared"})
        amount, physical, moved = 64, 64, 64
        resource = "shared:0"
    else:
        config = make_hbf_config(capacity_bytes=65536,
                                 metadata={"host_resource_id": "shared"})
        amount, physical, moved = 16384, 16384, 32768
        resource = "shared"
    task = TaskSpec("shared", "request", "shared wire audit", TaskCategory.MEMORY,
        metadata={"physical_memory_config": config,
            "physical_memory_configs": {"a": config, "b": config},
            "memory_accesses": tuple({"physical_owner": owner, "operation": "read",
                "address": 0, "byte_count": amount} for owner in ("a", "b"))})
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    owners = resolved.metadata["physical_execution_by_owner"]
    assert owners["a"]["physical_read_bytes"] == physical
    assert owners["b"]["physical_read_bytes"] == physical
    assert sum(d.bytes_moved for d in resolved.demands) == moved * 2
    assert {owner: row["resource_owners"][resource] for owner, row in owners.items()} == {"a": "a", "b": "b"}
    assert owners["b"]["completion_ns"] >= owners["a"]["completion_ns"]
