"""Explicit HOST DRAM staging survives planner and serving boundaries."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
from heterollm_sim.planner import _attach_gddr_physical_task, _attach_hbf_host_staging


def strict_staging_scenario(*, prompt=31, output=3, capacity=8192, all_hbf=False, layers=1):
    base = scenario(hbf=True, count=1, prompt=prompt, output=output)
    if layers > 1:
        layer_specs = tuple(LayerSpec("dense{}".format(index), "dense", 256, 512, 4,
            kv_heads=2, dtype="int8", quantization="w8a8", weight_bytes=655360)
            for index in range(layers))
        model = ModelSpec(name="staged-repeat-dense", graph=build_model_graph_from_layer_specs(
            "staged-repeat-dense", layer_specs, vocabulary_size=1024,
            max_sequence_length=262144))
        base = replace(base, model=model, placement=replace(base.placement,
            model_name=model.name, parallel=replace(base.placement.parallel,
                layer_to_stage={layer.layer_id: 0 for layer in layer_specs})))
    hbf = base.hardware.get_component("hbf0")
    physical = replace(hbf.metadata["physical_memory_config"], protocol_mode="ocp_hbf_0_7",
        channels=1, host_channels=1, parallel_units=None, planes_per_lun=1,
        planes_independent=True, queue_scope="host_channel", blocks_per_plane=128 if all_hbf else 32,
        pages_per_block=16, capacity_bytes=(8 if all_hbf else 2) * 1024**2,
        page_read_ns=100., page_program_ns=1000., block_erase_ns=2000.)
    hbf = replace(hbf, capacity_bytes=physical.capacity_bytes, metadata={**hbf.metadata,
        "physical_memory_config": physical,
        "parallel_lanes": 1, "read_latency_ns": 100., "write_latency_ns": 1000.,
        "memory_service": {**hbf.metadata["memory_service"], "parallel_lanes": 1,
                           "read_latency_ns": 100., "write_latency_ns": 1000.},
        "host_write_staging": {"host_component_id": "hostmem0", "capacity_bytes": capacity}})
    profiles = {kind: dict(values) for kind, values in base.component_profiles.items()}
    profiles["host_memory"][hbf.cost_profile_id] = replace(
        profiles["host_memory"][hbf.cost_profile_id], parallel_lanes=1,
        read_latency_ns=100., write_latency_ns=1000.)
    configured = replace(base, component_profiles=profiles, hardware=replace(base.hardware, components=tuple(
        hbf if component.component_id == hbf.component_id else component
        for component in base.hardware.components)))
    if all_hbf:
        gpu_operators = {key: "gpu0" for key in (
            "embedding", "lm_head",
            "dense0.attention", "dense0.mlp", "dense0.input_norm.reduce",
            "dense0.input_norm.apply", "dense0.post_attention_norm.reduce",
            "dense0.post_attention_norm.apply", "dense0.attention.rope",
            "dense0.attention.qk", "dense0.attention.pv",
            "dense0.attention.softmax.reduce", "dense0.attention.softmax.normalize",
            "dense0.attention.residual", "dense0.mlp.activation", "dense0.mlp.residual")}
        for index in range(1, layers):
            gpu_operators.update({key.replace("dense0.", "dense{}.".format(index)): "gpu0"
                                  for key in tuple(gpu_operators) if key.startswith("dense0.")})
        removed = {component.component_id for component in configured.hardware.components
                   if component.normalized_kind == "hbm"}
        configured = replace(configured, hardware=replace(configured.hardware,
            components=tuple(component for component in configured.hardware.components
                             if component.component_id not in removed),
            links=tuple(link for link in configured.hardware.links
                        if link.source_component not in removed and link.target_component not in removed)),
            placement=replace(configured.placement, metadata={**configured.placement.metadata,
                "control_plane": {"policy": {"options": {"operator_targets": gpu_operators}}}},
                parallel=replace(configured.placement.parallel,
                rank_mapping=tuple(replace(rank, memory_component_id="hbf0")
                                   for rank in configured.placement.parallel.rank_mapping))))
    return configured


def small_kv_task():
    return TaskSpec("kv", "request", "small KV append", TaskCategory.MEMORY,
        demands=(ResourceDemand("hbf0.memory", 0., bytes_moved=128),), metadata={
            "target_component": "gpu0", "rank": 0,
            "persistent_buffer_ids": ("kv:request:k",),
            "persistent_request_buffers": {"request": ("kv:request:k",)},
            "memory_accesses": ({"buffer_id": "kv:request:k", "operation": "write",
                "byte_count": 128, "offset_bytes": 0, "allocation_size_bytes": 4096},)})


def test_active_hbf_lowering_preserves_kind_and_binds_host_fabric():
    configured = strict_staging_scenario()
    lowered = _attach_hbf_host_staging(
        _attach_gddr_physical_task(small_kv_task(), configured), configured)
    assert lowered.metadata["physical_memory_config"].protocol_mode == "ocp_hbf_0_7"
    assert lowered.metadata["physical_memory_config"].kind.value == "HBF"
    assert lowered.metadata["memory_accesses"][0]["byte_count"] == 128
    contract = lowered.metadata["hbf_host_staging_by_owner"]["hbf0.memory"]
    assert contract["host_component_id"] == "hostmem0"
    assert contract["capacity_bytes"] == 8192
    assert contract["host_owner"] in lowered.metadata["physical_memory_configs"]
    assert contract["routes"]["to_host"]
    assert any(hop["link_id"] == "cpu-gpu-pcie" for hop in contract["routes"]["to_host"])
    assert any(hop["link_id"] == "gpu-hbf-ucie" for hop in contract["routes"]["to_hbf"])
    assert all(hop["resource_capacities"][hop["inflight_resource_id"]] == hop["queue_depth"]
               for path in contract["routes"].values() for hop in path)
    assert not lowered.metadata["hbf_preloaded_buffers"]


def test_staging_preloads_only_declared_resident_weights():
    configured = strict_staging_scenario()
    kv = small_kv_task()
    access = {**kv.metadata["memory_accesses"][0], "buffer_id": "weights:dense0", "operation": "read"}
    weights = replace(kv, metadata={"target_component": "gpu0", "rank": 0,
                                   "memory_accesses": (access,)})
    lowered = _attach_hbf_host_staging(_attach_gddr_physical_task(weights, configured), configured)
    assert lowered.metadata["hbf_preloaded_buffers"] == (("hbf0.memory", "weights:dense0", 0),)


@pytest.mark.parametrize("capacity", [0, 1024, 4097, True])
def test_invalid_staging_capacity_is_not_silently_rounded(capacity):
    configured = strict_staging_scenario(capacity=capacity)
    with pytest.raises(ValueError, match="capacity_bytes"):
        _attach_hbf_host_staging(_attach_gddr_physical_task(small_kv_task(), configured), configured)


def test_staging_requires_an_actual_host_route():
    configured = strict_staging_scenario()
    configured = replace(configured, hardware=replace(configured.hardware,
        links=tuple(link for link in configured.hardware.links if link.link_id != "cpu-gpu-pcie")))
    with pytest.raises(ValueError, match="route"):
        _attach_hbf_host_staging(_attach_gddr_physical_task(small_kv_task(), configured), configured)


def test_zero_hbm_two_layer_serving_keeps_partial_kv_readable_and_charges_real_resources(monkeypatch):
    from heterollm_sim.hbf_staging import HbfStagingController
    from heterollm_sim.memory_types import Operation
    from heterollm_sim.nand_core import NandCore
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.serving import RequestStatus

    configured = strict_staging_scenario(all_hbf=True, layers=2, capacity=65536)
    assert not any(component.normalized_kind == "hbm" for component in configured.hardware.components)
    assert {"cpu", "host_memory"} <= {component.normalized_kind for component in configured.hardware.components}
    logical_writes, nand_programs = [], []
    original_write, original_submit = HbfStagingController.write, NandCore.submit

    def write(controller, buffer_id, generation, offset, size, *args, **kwargs):
        logical_writes.append((buffer_id, offset, size))
        return original_write(controller, buffer_id, generation, offset, size, *args, **kwargs)

    def submit(core, request):
        if request.operation is Operation.WRITE:
            nand_programs.append(request)
        return original_submit(core, request)

    monkeypatch.setattr(HbfStagingController, "write", write)
    monkeypatch.setattr(NandCore, "submit", submit)
    result = run_scenario(configured)
    metrics = result.serving.request_metrics["r0"]
    assert metrics.status is RequestStatus.FINISHED
    assert metrics.visible_output_tokens == 3
    assert metrics.ttft_ns > 0 and metrics.tpot_ns > 0
    assert all(target == "gpu0" for target in result.scenario.placement.op_to_component.values())
    assert any("kv" in identity and 0 < size < 4096 for identity, _offset, size in logical_writes)
    assert nand_programs
    assert all(request.byte_count == 4096 and request.address % 4096 == 0 for request in nand_programs)
    ledgers = [batch.cost.metadata for batch in result.serving.batches]
    assert len(ledgers) == 3
    assert all(row["physical_execution_scope"] == "persistent_live_kernel" for row in ledgers)
    assert all(row["preview_physical_cost_scope"] == "deferred_to_live_persistent_runtime" for row in ledgers)
    assert sum(row["storage_traffic"]["pages_programmed"] for row in ledgers) > 0
    assert all(row["dram_traffic"]["physical_write_bytes"] > 0 for row in ledgers)
    assert all(any(resource["owner"] == "cpu0.memory" for resource in
                   row["dram_traffic"]["resource_totals"].values()) for row in ledgers)
    assert all(any("cpu-gpu-pcie" in resource and duration > 0
                   for resource, duration in row["resource_busy_ns"].items()) for row in ledgers)
