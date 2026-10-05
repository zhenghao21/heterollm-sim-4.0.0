"""Small end-to-end checks for the real GPU/GDDR lowering path."""

from dataclasses import replace
import json
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import GDDRProfile
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import (
    LayerSpec,
    RequestSpec,
    build_model_graph_from_layer_specs,
)
from heterollm_sim.kernel_model import KernelModelProfile
from heterollm_sim.planner import compile_scenario, validate_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.web import build_server, scenario_to_payload


def _gpu_gddr_scenario():
    """Replace one reference HBM stack with a connected GDDR7 device."""

    reference = build_reference_scenario()
    gpu = reference.hardware.get_component("gpu0")
    old_memory = reference.hardware.get_component("hbm0")
    gddr = get_component_preset("gddr7-16gb-30_0-256bit").component
    metadata = dict(gddr.metadata)
    # The preset id is intentionally independent of this test's component id;
    # let ComponentSpec/config bind the physical owner to hbm0 at construction.
    metadata.pop("memory_service_owner", None)
    memory = replace(
        gddr,
        component_id="hbm0",
        package_id=old_memory.package_id,
        die_id=old_memory.die_id,
        metadata=metadata,
    )
    gpu = replace(
        gpu,
        ports=tuple(
            replace(
                port,
                protocol="GDDR",
                version="GDDR7",
                lanes=256,
                bandwidth_gbps=7680.0,
            )
            if port.port_id == "hbm0"
            else port
            for port in gpu.ports
        ),
        metadata={"supported_gddr_generations": ["GDDR7"]},
    )
    links = tuple(
        replace(
            link,
            protocol="GDDR",
            version="GDDR7",
            lanes=256,
            bandwidth_gbps=7680.0,
        )
        if link.link_id == "gpu-hbm0"
        else link
        for link in reference.hardware.links
    )
    components = tuple(
        gpu if component.component_id == "gpu0" else
        memory if component.component_id == "hbm0" else component
        for component in reference.hardware.components
    )
    profiles = {kind: dict(registry) for kind, registry in reference.component_profiles.items()}
    profiles["gddr"] = {
        "legacy-gddr": GDDRProfile(
            bandwidth_gb_s=960.0,
            efficiency=0.75,
            energy_pj_per_byte=4.0,
            resource_id="gddr7_16gb_30_0_256bit.memory",
            read_latency_ns=35.0,
            write_latency_ns=35.0,
            transaction_bytes=256,
            max_outstanding_requests=64,
            parallel_lanes=8,
            generation="GDDR7",
        )
    }
    return replace(
        reference,
        hardware=replace(
            reference.hardware,
            components=components,
            links=links,
        ),
        component_profiles=profiles,
    )


def _post_json(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_real_gpu_gddr_compile_carries_gemm_attention_and_directional_physical_accesses():
    scenario = _gpu_gddr_scenario()
    assert validate_scenario(scenario).is_valid
    schedule = compile_scenario(scenario)

    physical = [
        task
        for task in schedule.tasks
        if task.metadata.get("physical_memory_config")
        and task.metadata.get("memory_accesses")
    ]
    assert physical, "the real reference workload must lower GPU work to GDDR descriptors"
    assert any(task.metadata.get("phase") == "gpu_gemm" for task in physical)
    assert any("attention_flash.gpu_fused_attention" in task.name for task in physical)

    gemm = next(task for task in physical if task.name.endswith("qkv.gpu_gemm"))
    cost = gemm.metadata["cost_model"]
    assert cost["activation_bytes"] + cost["weight_bytes"] > 0
    assert cost["output_bytes"] > 0
    accesses = tuple(gemm.metadata["memory_accesses"])
    assert sum(item["byte_count"] for item in accesses if item["operation"] == "read") > 0
    assert sum(item["byte_count"] for item in accesses if item["operation"] == "write") == cost["output_bytes"]
    capacity = gemm.metadata["physical_memory_config"]["capacity_bytes"]
    assert all(
        0 <= item["address"] < capacity
        and item["address"] + item["byte_count"] <= capacity
        for item in accesses
    )


def test_frontend_transport_round_trip_accepts_default_gddr_component(tmp_path):
    """Exercise the same normalize/validate transport used by the Web UI."""

    scenario = _gpu_gddr_scenario()
    payload = scenario_to_payload(scenario)
    # The UI saves JSON, then reloads it and asks /normalize for canonical
    # memory-service metadata.  Keep this test at the actual HTTP boundary.
    server = build_server("127.0.0.1", 0, component_preset_cache_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = "http://127.0.0.1:{}".format(server.server_address[1])
        normalized = _post_json(base_url + "/api/normalize", payload)
        reloaded = scenario_from_dict(normalized)
        component = reloaded.hardware.get_component("hbm0")
        physical = component.metadata["physical_memory_config"]
        assert component.kind == "gddr"
        assert physical["kind"] == "GDDR"
        assert physical["generation"] == "GDDR7"
        assert component.capacity_bytes == physical["capacity_bytes"]
        assert component.capacity_bytes == 16_000_000_000
        validation = _post_json(base_url + "/api/validate", normalized)
        assert validation["valid"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _tiny_dense_gddr_scenario(prompt_tokens, *, stateful_l2):
    """Keep the reference topology, replacing its workload with tiny real ops."""

    scenario = _gpu_gddr_scenario()
    graph = build_model_graph_from_layer_specs(
        "tiny-dense-transformer",
        (
            LayerSpec(
                layer_id="tiny0",
                kind="dense",
                hidden_size=8,
                intermediate_size=16,
                attention_heads=2,
                kv_heads=1,
                dtype="int8",
                quantization="w8a8",
                weight_bytes=512,
            ),
        ),
        architecture="decoder_only_transformer",
        vocabulary_size=16,
        max_sequence_length=16,
        embedding_weight_bytes=128,
        output_weight_bytes=128,
    )
    model = replace(scenario.model, name="tiny-dense-transformer", graph=graph)

    gpu_profile = scenario.component_profiles["gpu"]["legacy-gpu"]
    # The tiny cache deliberately fits two lines.  This makes natural reuse,
    # eviction, and dirty writeback observable without synthetic cache probes.
    levels = list(gpu_profile.cache_hierarchy.levels)
    levels[0] = replace(
        levels[0], capacity_bytes=128, associativity=1, banks=1,
        read_ports=1, write_ports=1, max_outstanding=2,
    )
    levels[-1] = replace(
        levels[-1], capacity_bytes=256, associativity=1, banks=1,
        read_ports=1, write_ports=1, max_outstanding=2,
    )
    tiny_gpu = replace(
        gpu_profile,
        cache_hierarchy=replace(gpu_profile.cache_hierarchy, levels=tuple(levels)),
        kernel_model=KernelModelProfile(
            hardware_id="tiny-gddr-test",
            runtime_id="analytical",
            architecture="tiny-dense-transformer",
            stateful_l2=stateful_l2,
        ),
    )
    profiles = {kind: dict(registry)
                for kind, registry in scenario.component_profiles.items()}
    profiles["gpu"]["legacy-gpu"] = tiny_gpu

    placement = replace(
        scenario.placement,
        model_name=model.name,
        parallel=replace(
            scenario.placement.parallel,
            layer_to_stage={"tiny0": 0},
        ),
        kv_policy=replace(
            scenario.placement.kv_policy,
            tokens_per_page=4,
            dtype="int8",
            offload_ratio=0.0,
        ),
    )
    workload = replace(
        scenario.workload,
        name="tiny-dense-workload",
        requests=(RequestSpec(
            request_id="tiny-request",
            arrival_ns=0.0,
            prompt_tokens=prompt_tokens,
            output_tokens=3,
        ),),
        request_count=1,
        prompt_tokens=prompt_tokens,
        output_tokens=3,
        mtp=None,
        scheduler=replace(
            scenario.workload.scheduler,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            prefill_chunk_tokens=prompt_tokens,
        ),
    )
    return replace(
        scenario,
        name="tiny-dense-gpu-gddr",
        model=model,
        placement=placement,
        workload=workload,
        component_profiles=profiles,
    )


@pytest.mark.parametrize("prompt_tokens", (2, 5))
@pytest.mark.parametrize("stateful_l2", (False, True))
def test_tiny_dense_gddr_runs_prefill_and_two_decode_steps(
    prompt_tokens, stateful_l2,
):
    scenario = _tiny_dense_gddr_scenario(
        prompt_tokens, stateful_l2=stateful_l2,
    )
    payload = scenario_to_payload(scenario)
    reloaded = scenario_from_dict(payload)
    assert validate_scenario(reloaded).is_valid
    schedule = compile_scenario(reloaded)
    physical_tasks = [
        task for task in schedule.tasks
        if task.metadata.get("physical_memory_config")
    ]
    assert physical_tasks
    assert all(task.metadata.get("target_component") == "gpu0"
               for task in physical_tasks)

    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks,
        resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
    assert not kernel.has_active_tasks
    phases = {event.task.metadata.get("phase") for event in events}
    assert "prefill" in phases
    assert {"decode0001", "decode0002"}.issubset(phases)

    physical_events = [
        event for event in events
        if "physical_execution" in event.task.metadata
    ]
    assert physical_events
    for event in physical_events:
        metadata = event.task.metadata
        config = metadata["physical_memory_config"]
        assert config["kind"] == "GDDR"
        assert config["generation"] == "GDDR7"
        capacity = config["capacity_bytes"]
        accesses = metadata["memory_accesses"]
        assert accesses
        assert any(item["operation"] == "read" for item in accesses)
        for access in accesses:
            assert access["physical_owner"] == "hbm0.memory"
            assert access["resource_id"] == "hbm0.memory"
            assert access["operation"] in {"read", "write"}
            assert access["offset_bytes"] >= 0
            assert access["generation"] == 0
            assert 0 <= access["address"]
            assert access["address"] + access["byte_count"] <= capacity
        assert event.task.dependencies
        assert 0 <= event.dependency_ready_ns <= event.start_ns

    if stateful_l2:
        l2 = [event.task.metadata["l2_execution"]
              for event in physical_events]
        assert any(item["miss_lines"] for item in l2)
        assert any(item["hit_lines"] for item in l2)
        assert any(item["dirty_eviction_bytes"] for item in l2)
        assert any(
            any(access["operation"] == "write"
                for access in event.task.metadata["memory_accesses"])
            and
            event.task.metadata["physical_execution"]["physical_write_bytes"] > 0
            for event in physical_events
        )
    else:
        assert all("l2_execution" not in event.task.metadata
                   for event in physical_events)
