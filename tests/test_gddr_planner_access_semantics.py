from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.planner import _attach_gddr_physical_task


def _scenario(capacity=4096):
    raw = {
        "kind": "GDDR",
        "generation": "GDDR7",
        "capacity_bytes": capacity,
        "burst_bytes": 64,
    }
    gpu = ComponentSpec("gpu0", "gpu")
    gddr = ComponentSpec(
        "gddr0",
        "gddr",
        capacity_bytes=capacity,
        metadata={
            "physical_memory_config": raw,
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


def test_gddr_planner_records_directional_reconstruction_on_sharded_demand():
    scenario = _scenario()
    task = _attach_gddr_physical_task(
        _task(
            "sharded-gemm",
            400,
            {"activation_bytes": 128, "weight_bytes": 256, "output_bytes": 64},
        ),
        scenario,
    )
    reconstruction = task.metadata["gddr_directional_reconstruction"]
    assert reconstruction["declared_read_bytes"] == 384
    assert reconstruction["declared_write_bytes"] == 64
    assert reconstruction["physical_demand_bytes"] == 400
    assert reconstruction["reconstructed_read_bytes"] + reconstruction["reconstructed_write_bytes"] == 400


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
