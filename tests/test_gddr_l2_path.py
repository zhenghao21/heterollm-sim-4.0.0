from dataclasses import asdict

from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import DramConfig, MemoryKind


def _config():
    return DramConfig(
        kind=MemoryKind.GDDR,
        generation="GDDR7",
        channels=1,
        data_lanes=2,
        data_width_bits=32,
        data_rate_mt_s=10000.0,
        stacks=1,
        dies_per_stack=1,
        ranks_per_channel=1,
        bank_groups_per_rank=2,
        banks_per_group=2,
        rows_per_bank=64,
        row_bytes=256,
        burst_bytes=64,
        interleave_bytes=64,
        interface_bandwidth_gb_s=80.0,
        capacity_bytes=131072,
    )


def _task(task_id, *, config, access):
    hbm = HBMProfile(
        bandwidth_gb_s=config.interface_bandwidth_gb_s,
        read_bandwidth_gb_s=config.interface_bandwidth_gb_s,
        write_bandwidth_gb_s=config.interface_bandwidth_gb_s,
        resource_id="gddr0.gddr_fabric",
        transaction_bytes=64,
    )
    contract = {
        "owner": "gpu0.l2",
        "capacity_bytes": 128,
        "line_bytes": 64,
        "write_back": True,
        "write_allocate": True,
        "cache_resource": "gpu0.l2",
        "memory_resource": "gddr0.gddr_fabric",
        "cache_bandwidth_gb_s": 1000.0,
        "cache_latency_ns": 1.0,
        "cache_parallelism": 1,
        "bandwidth_gb_s": config.interface_bandwidth_gb_s,
        "hbm": asdict(hbm),
        "accesses": ({
            "buffer_id": "weight-W",
            "offset_bytes": 0,
            "size_bytes": 64,
            "operation": "read",
        },),
        "invocation_buffers": (),
        "envelope_resources": ("gpu0.l2.access_order",),
        "order_resource": "gpu0.l2.access_order",
        "dependency_ns": 0.0,
    }
    return TaskSpec(
        task_id=task_id,
        request_id="request-0",
        name="GDDR stateful read",
        category=TaskCategory.MEMORY,
        demands=(
            ResourceDemand("gddr0.gddr_fabric", 1.0, bytes_moved=64),
            ResourceDemand("gpu0.l2", 0.0),
            ResourceDemand("gpu0.l2.access_order", 1.0),
        ),
        metadata={
            "physical_memory_config": asdict(config),
            "memory_access": dict(access),
            "stateful_l2": contract,
        },
    )


def test_stateful_l2_runs_before_gddr_and_keeps_cold_miss_physical():
    config = _config()
    access = {
        "operation": "read",
        "address": 0,
        "byte_count": 64,
        "physical_owner": "gddr0.gddr_fabric",
        "resource_id": "gddr0.gddr_fabric",
    }
    task = _task("cold", config=config, access=access)
    event = UnifiedEventKernel.from_closed_graph(
        (task,),
        resource_capacities={"gddr0.gddr_fabric": 1, "gpu0.l2": 1, "gpu0.l2.access_order": 1},
    ).step()
    assert event is not None
    assert event.task.metadata["l2_execution"]["hbm_read_bytes"] == 64
    assert event.task.metadata["physical_execution"]["physical_read_bytes"] == 64
    assert event.task.metadata["physical_execution"]["physical_write_bytes"] == 0


def test_stateful_l2_hit_does_not_submit_an_empty_gddr_transaction():
    config = _config()
    access = {
        "operation": "read",
        "address": 0,
        "byte_count": 64,
        "physical_owner": "gddr0.gddr_fabric",
        "resource_id": "gddr0.gddr_fabric",
    }
    tasks = (
        _task("cold", config=config, access=access),
        _task("hot", config=config, access=access),
    )
    kernel = UnifiedEventKernel.from_closed_graph(
        tasks,
        resource_capacities={"gddr0.gddr_fabric": 1, "gpu0.l2": 1, "gpu0.l2.access_order": 1},
    )
    first = kernel.step()
    second = kernel.step()
    assert first is not None and second is not None
    assert first.task.metadata["physical_execution"]["physical_read_bytes"] == 64
    assert second.task.metadata["l2_execution"]["hbm_read_bytes"] == 0
    assert "physical_execution" not in second.task.metadata


def test_gddr_presets_derive_command_interval_from_lane_payload_budget():
    expected = {
        "gddr6-16gb-20_0-256bit": 0.8,
        "gddr6x-16gb-21_0-256bit": 64.0 / 84.0,
        "gddr7-16gb-30_0-256bit": 64.0 / 120.0,
    }
    for preset_id, interval in expected.items():
        preset = get_component_preset(preset_id)
        config = preset.component.metadata["physical_memory_config"]
        assert config["data_lanes"] == 8
        assert config["burst_bytes"] == 64
        assert config["burst_interval_ns"] == interval
        command_ceiling = config["data_lanes"] * config["burst_bytes"] / interval
        assert command_ceiling >= config["interface_bandwidth_gb_s"]
        assert config["metadata"]["command_interval_model"] == "per_lane_burst_payload_budget"
