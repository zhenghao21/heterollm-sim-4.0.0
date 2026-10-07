from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_types import DramConfig, MemoryKind


def _config(capacity):
    return DramConfig(
        kind=MemoryKind.GDDR, generation="GDDR7", channels=1,
        data_lanes=2, data_width_bits=32, data_rate_mt_s=10000.0,
        stacks=1, dies_per_stack=1, ranks_per_channel=1,
        bank_groups_per_rank=2, banks_per_group=2,
        rows_per_bank=64, row_bytes=256, burst_bytes=64,
        interleave_bytes=64, interface_bandwidth_gb_s=80.0,
        capacity_bytes=capacity,
    )


def _task(accesses, configs):
    return TaskSpec(
        task_id="two-physical-owners", request_id="request", name="physical accesses",
        category=TaskCategory.MEMORY,
        metadata={
            "physical_owner": "hbm0.memory",
            "physical_memory_config": configs["hbm0.memory"],
            "physical_memory_configs": configs,
            "physical_energy_pj_per_byte_by_owner": {
                "hbm0.memory": 2.0,
                "hbm1.memory": 3.0,
            },
            "memory_accesses": tuple(accesses),
        },
    )


def test_owner_specific_geometry_and_same_numeric_address_do_not_cross_order():
    configs = {
        "hbm0.memory": _config(131072).__dict__,
        "hbm1.memory": _config(65536).__dict__,
    }
    remote_read = {
        "physical_owner": "hbm1.memory", "resource_id": "hbm1.memory",
        "operation": "read", "address": 0, "byte_count": 64,
    }
    isolated = resolve_physical_task(
        _task((remote_read,), configs), PhysicalRuntimeContext(), 0.0,
    )
    combined = resolve_physical_task(
        _task((
            {"physical_owner": "hbm0.memory", "resource_id": "hbm0.memory",
             "operation": "write", "address": 0, "byte_count": 64},
            remote_read,
        ), configs),
        PhysicalRuntimeContext(), 0.0,
    )
    by_owner = combined.metadata["physical_execution_by_owner"]
    assert set(by_owner) == {"hbm0.memory", "hbm1.memory"}
    assert by_owner["hbm1.memory"]["completion_ns"] == isolated.metadata["physical_completion_ns"]
    assert by_owner["hbm0.memory"]["physical_write_bytes"] > 0
    assert by_owner["hbm1.memory"]["physical_read_bytes"] > 0
    assert by_owner["hbm0.memory"]["energy_pj"] == 2.0 * by_owner["hbm0.memory"]["physical_bytes"]
    assert by_owner["hbm1.memory"]["energy_pj"] == 3.0 * by_owner["hbm1.memory"]["physical_bytes"]

    out_of_bounds = {**remote_read, "address": 65536}
    try:
        resolve_physical_task(
            _task((out_of_bounds,), configs), PhysicalRuntimeContext(), 0.0,
        )
    except ValueError as exc:
        assert "hbm1.memory capacity" in str(exc)
    else:
        raise AssertionError("remote owner capacity must be checked")
