from dataclasses import asdict, replace

import pytest

from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import CacheHierarchyProfile, CacheLevelProfile, GPUProfile, HBMProfile, TensorCoreProfile
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.kernel_memory import attach_l2_contract, resolve_l2_task
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
        "cache_energy_pj_per_byte": 0.0,
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
    assert event.task.metadata["l2_execution"]["backing_accesses"][0]["operation"] == "read"
    assert event.task.metadata["l2_execution"]["backing_accesses"][0]["size_bytes"] == 64
    allocation = event.task.metadata["l2_execution"]["physical_allocations"][0]
    assert allocation["buffer_id"] == "weight-W"
    assert allocation["size_bytes"] == 64
    assert allocation["generation"] == 0
    physical_allocation = event.task.metadata["physical_allocations"][0]
    assert physical_allocation["buffer_id"] == "weight-W"
    assert physical_allocation["size_bytes"] == 64
    assert physical_allocation["generation"] == 0
    assert physical_allocation["physical_owner"] == "gddr0.gddr_fabric"
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
    assert second.start_ns >= first.task.metadata["physical_completion_ns"]


def test_l2_profile_energy_prices_both_cold_and_hot_accesses():
    level = CacheLevelProfile(name="L2", capacity_bytes=128, line_bytes=64,
                              bandwidth_gb_s=1000, hit_latency_ns=1,
                              energy_pj_per_byte=0.6, resource_id="gpu0.l2")
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1), CacheHierarchyProfile((level,)),
                     32, 1, 1, 1, 1)
    hbm = HBMProfile(bandwidth_gb_s=80, energy_pj_per_byte=4,
                     resource_id="hbm0")
    task = TaskSpec("read", request_id="request", name="read", category=TaskCategory.MEMORY,
                    demands=(ResourceDemand("hbm0", 1, bytes_moved=64),))
    task = attach_l2_contract(task, gpu=gpu, hbm=hbm, memory_resource="hbm0",
                              cache_resource="gpu0.l2", owner="gpu0.l2",
                              accesses=(CacheAccess("W", 0, 64, "read", 64),))
    states = {}
    cold = resolve_l2_task(task, states)
    hot = resolve_l2_task(replace(task, task_id="hot"), states)
    for resolved, expected_memory_energy in ((cold, 256), (hot, 0)):
        demands = {d.resource_id: d for d in resolved.demands}
        assert demands["gpu0.l2"].energy_pj == pytest.approx(38.4)
        assert demands["hbm0"].energy_pj == expected_memory_energy
        assert demands["gpu0.l2.access_order"].service_ns == pytest.approx(
            1.8 if expected_memory_energy else 1.0)


@pytest.mark.parametrize("energy", (None, -1, float("inf"), float("nan"), True))
def test_l2_requires_explicit_valid_energy_before_mutating_state(energy):
    task = _task("invalid-energy", config=_config(), access={})
    contract = dict(task.metadata["stateful_l2"])
    if energy is None:
        contract.pop("cache_energy_pj_per_byte")
    else:
        contract["cache_energy_pj_per_byte"] = energy
    states = {}
    with pytest.raises(ValueError, match="cache_energy_pj_per_byte"):
        resolve_l2_task(replace(task, metadata={**task.metadata, "stateful_l2": contract}), states)
    assert states == {}


@pytest.mark.parametrize("dirty", (False, True))
def test_l2_backing_waits_for_lookup_and_updates_device_envelope(dirty):
    config = _config()
    owner = "gddr0.gddr_fabric"
    access = {"operation": "read", "address": 0, "byte_count": 64,
              "physical_owner": owner, "resource_id": owner}
    task = _task("lookup", config=config, access=access)
    contract = {**task.metadata["stateful_l2"], "capacity_bytes": 64,
                "cache_latency_ns": 100,
                "envelope_resources": ("gpu0.l2.access_order", "gpu0.device_stream")}
    old_prediction = {"prediction": {"prediction_ns": 9999}, "analytical_service_ns": 9999}
    task = replace(task, demands=task.demands + (ResourceDemand("gpu0.device_stream", 9999),),
                   metadata={**task.metadata, "stateful_l2": contract,
                             "kernel_prediction": old_prediction,
                             "phase_metadata": {"kernel_model": old_prediction},
                             "cost_model": {"kernel_model": old_prediction}})
    tasks = []
    if dirty:
        seed_contract = {**contract, "accesses": ({"buffer_id": "dirty-W", "offset_bytes": 0,
                         "size_bytes": 64, "operation": "write", "buffer_size_bytes": 64},)}
        seed = replace(task, task_id="seed", metadata={**task.metadata, "stateful_l2": seed_contract})
        tasks.append(seed)
        task = replace(task, dependencies=("seed",))
    tasks.append(task)
    kernel = UnifiedEventKernel.from_closed_graph(tuple(tasks))
    if dirty:
        assert kernel.step() is not None
    event = kernel.step()
    assert event is not None
    physical = event.task.metadata["physical_execution"]
    assert physical["arrival_ns"] == event.start_ns + 100
    assert event.task.metadata["l2_execution"]["dirty_eviction_bytes"] == (64 if dirty else 0)
    assert physical["physical_write_bytes"] == (64 if dirty else 0)
    demands = {d.resource_id: d for d in event.demands}
    assert demands["gpu0.device_stream"].service_ns == pytest.approx(event.service_ns)
    assert event.end_ns == physical["completion_ns"]
    assert event.service_ns == pytest.approx(100 + physical["service_ns"])
    assert event.task.metadata["l2_execution"]["memory_completion_offset_ns"] == pytest.approx(event.service_ns)
    for audit in (event.task.metadata["kernel_prediction"],
                  event.task.metadata["phase_metadata"]["kernel_model"],
                  event.task.metadata["cost_model"]["kernel_model"]):
        assert audit["prediction"]["prediction_ns"] == pytest.approx(event.service_ns)
    assert old_prediction["prediction"]["prediction_ns"] == 9999


def test_stateful_l2_preserves_declared_fixed_base():
    config = _config()
    access = {
        "operation": "read", "address": 0, "byte_count": 64,
        "physical_owner": "gddr0.gddr_fabric", "resource_id": "gddr0.gddr_fabric",
    }
    task = _task("fixed-base", config=config, access=access)
    task = replace(task, metadata={
        **task.metadata,
        "physical_allocations": [{
            "buffer_id": "weight-W", "size_bytes": 64, "generation": 0,
            "address": 1024, "physical_owner": "gddr0.gddr_fabric",
        }],
    })
    event = UnifiedEventKernel.from_closed_graph(
        (task,),
        resource_capacities={"gddr0.gddr_fabric": 1, "gpu0.l2": 1, "gpu0.l2.access_order": 1},
    ).step()
    assert event is not None
    assert event.task.metadata["physical_execution"]["physical_read_bytes"] == 64
    assert event.task.metadata["memory_accesses"][0]["address"] == 1024


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


def test_freed_serving_buffer_discards_l2_dirty_lines_before_address_reuse():
    config = _config()
    owner = "gddr0.gddr_fabric"

    def temporary(task_id, buffer_id, operation, dependencies=()):
        access = {"operation": operation, "byte_count": 64,
                  "buffer_id": buffer_id, "offset_bytes": 0, "generation": 1,
                  "allocation_generation": 1, "allocation_size_bytes": 64,
                  "physical_owner": owner, "resource_id": owner,
                  "address_source": "stable_buffer_tensor_offset"}
        task = _task(task_id, config=config, access=access)
        contract = {**task.metadata["stateful_l2"], "capacity_bytes": 64,
                    "accesses": ({"buffer_id": buffer_id, "offset_bytes": 0,
                                  "size_bytes": 64, "operation": operation,
                                  "buffer_size_bytes": 64, "allocation_generation": 1},)}
        return replace(task, request_id="cohort-000000", dependencies=tuple(dependencies),
                       metadata={**task.metadata, "physical_owner": owner, "stateful_l2": contract})

    kernel = UnifiedEventKernel.from_closed_graph((
        temporary("write", "dead", "write"),
        temporary("read", "next", "read", ("write",)),
    ))
    assert kernel.step() is not None
    second = kernel.step()
    assert second is not None
    assert second.task.metadata["l2_execution"]["dirty_eviction_bytes"] == 0
    assert kernel.physical_runtime.allocators[owner].allocations() == ()
    assert kernel._l2_states["gpu0.l2"][1].resident_lines() == ()
