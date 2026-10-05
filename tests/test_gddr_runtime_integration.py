"""Small end-to-end checks for the real GPU/GDDR lowering path."""

from dataclasses import replace
import json
from threading import Thread
from urllib.request import Request, urlopen

from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import GDDRProfile
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
