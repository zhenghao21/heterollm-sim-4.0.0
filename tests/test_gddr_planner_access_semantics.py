from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import GDDRProfile
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import DramConfig, MemoryKind
from heterollm_sim.planner import _attach_gddr_physical_task


def _scenario(capacity=4096):
    physical = DramConfig(
        kind=MemoryKind.GDDR,
        generation="GDDR7",
        channels=1,
        ranks_per_channel=1,
        bank_groups_per_rank=1,
        banks_per_group=8,
        rows_per_bank=1,
        row_bytes=8192,
        burst_bytes=64,
        data_width_bits=64,
        data_rate_mt_s=64000,
        interface_bandwidth_gb_s=512,
        capacity_bytes=capacity,
    )
    gpu = ComponentSpec("gpu0", "gpu")
    gddr = ComponentSpec(
        "gddr0",
        "gddr",
        cost_profile_id="gddr-profile",
        capacity_bytes=capacity,
        metadata={
            "physical_memory_config": physical,
            "memory_service": {"physical_owner": "gddr0.gddr_fabric"},
        },
    )
    hardware = SimpleNamespace(
        components=(gpu, gddr),
        component_map=lambda: {"gpu0": gpu, "gddr0": gddr},
    )
    return SimpleNamespace(
        hardware=hardware,
        placement=SimpleNamespace(tensor_bytes={"weight-W": 1024}),
        component_profile_kind=lambda _component: "gddr",
        resolve_component_profile=lambda _component, _expected_type=None: GDDRProfile(
            bandwidth_gb_s=512,
            resource_id="gddr0.gddr_fabric",
            generation="GDDR7",
        ),
    )


def _task(task_id, physical_bytes, cost, **metadata):
    return TaskSpec(
        task_id=task_id,
        request_id="request-0",
        name=task_id,
        category=TaskCategory.COMPUTE,
        demands=(
            ResourceDemand(
                "gddr0.gddr_fabric",
                1.0,
                bytes_moved=physical_bytes,
            ),
        ),
        metadata={
            "target_component": "gpu0",
            "cost_model": cost,
            "input_tensor_id": "activation-A",
            "weight_tensor_id": "weight-W",
            "output_tensor_id": "output-O",
            **metadata,
        },
    )


def test_gddr_planner_adds_gemm_reads_and_output_write():
    scenario = _scenario()
    cost = {"activation_bytes": 128, "weight_bytes": 256, "output_bytes": 64}
    first = _attach_gddr_physical_task(
        _task("request-0.1.gemm", 448, cost), scenario
    )
    second = _attach_gddr_physical_task(
        _task("request-0.2.gemm", 448, cost), scenario
    )

    first_accesses = first.metadata["memory_accesses"]
    second_accesses = second.metadata["memory_accesses"]
    assert [(item["operation"], item["byte_count"]) for item in first_accesses] == [
        ("read", 128),
        ("read", 256),
        ("write", 64),
    ]
    assert [item["address"] for item in first_accesses] == [
        item["address"] for item in second_accesses
    ]
    assert all(
        item["address"] + item["byte_count"] <= 4096 for item in first_accesses
    )


def test_gddr_planner_rejects_unresolved_directional_sharding():
    scenario = _scenario()
    with pytest.raises(ValueError, match="exact physical directional contract"):
        _attach_gddr_physical_task(
            _task(
                "sharded-gemm",
                400,
                {"activation_bytes": 128, "weight_bytes": 256, "output_bytes": 64},
            ),
            scenario,
        )


def test_gddr_planner_consumes_exact_cache_direction_after_sharding():
    scenario = _scenario()
    task = _attach_gddr_physical_task(
        _task(
            "sharded-gemm",
            400,
            {
                "activation_bytes": 128,
                "weight_bytes": 256,
                "output_bytes": 64,
                "cache": {
                    "physical_read_bytes": 336,
                    "physical_write_bytes": 64,
                },
            },
        ),
        scenario,
    )
    assert sum(item["byte_count"] for item in task.metadata["memory_accesses"] if item["operation"] == "read") == 336
    assert sum(item["byte_count"] for item in task.metadata["memory_accesses"] if item["operation"] == "write") == 64
    assert "gddr_directional_reconstruction" not in task.metadata


def test_gddr_planner_rejects_out_of_range_buffer_offset():
    scenario = _scenario()
    with pytest.raises(ValueError, match="capacity"):
        _attach_gddr_physical_task(
            _task(
                "bad-offset",
                448,
                {"activation_bytes": 128, "weight_bytes": 256, "output_bytes": 64},
                input_offset_bytes=4000,
            ),
            scenario,
        )


def test_gddr_planner_uses_explicit_buffer_range_and_checks_it():
    scenario = _scenario()
    with pytest.raises(ValueError, match="exceeds capacity"):
        _attach_gddr_physical_task(
            _task(
                "bad-explicit-range",
                128,
                {"read_bytes": 64, "write_bytes": 0},
                memory_accesses=(
                    {
                        "operation": "read",
                        "address": 4032,
                        "byte_count": 128,
                        "buffer_id": "activation-A",
                    },
                ),
            ),
            scenario,
        )


def test_gddr_planner_leaves_access_derived_extent_inferred():
    scenario = _scenario()
    first = _attach_gddr_physical_task(
        _task(
            "range-128",
            128,
            {"read_bytes": 128, "write_bytes": 0},
            memory_accesses=(
                {
                    "operation": "read",
                    "buffer_id": "activation-A",
                    "offset_bytes": 0,
                    "byte_count": 128,
                },
            ),
        ),
        scenario,
    )
    second = _attach_gddr_physical_task(
        _task(
            "range-64",
            64,
            {"read_bytes": 64, "write_bytes": 0},
            memory_accesses=(
                {
                    "operation": "read",
                    "buffer_id": "activation-A",
                    "offset_bytes": 0,
                    "byte_count": 64,
                },
            ),
        ),
        scenario,
    )
    assert "allocation_size_bytes" not in first.metadata["memory_accesses"][0]
    assert "allocation_size_bytes" not in second.metadata["memory_accesses"][0]


def test_gddr_planner_preserves_declared_fixed_extent():
    scenario = _scenario()
    task = _attach_gddr_physical_task(
        _task(
            "fixed-range",
            128,
            {"read_bytes": 128, "write_bytes": 0},
            memory_accesses=(
                {
                    "operation": "read",
                    "buffer_id": "activation-A",
                    "offset_bytes": 0,
                    "byte_count": 128,
                    "buffer_size_bytes": 256,
                },
            ),
        ),
        scenario,
    )
    assert task.metadata["memory_accesses"][0]["allocation_size_bytes"] == 256
