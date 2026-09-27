from dataclasses import replace

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.config import _bind_local_rtx5080_hardware_presets, scenario_from_dict
from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.ir import ComponentSpec, PortSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.web import scenario_to_payload
from tools.native_llama_compare import build_matching_scenario


def test_memory_profile_and_local_protocol_link_share_component_source():
    scenario = build_reference_scenario()
    hbm = replace(scenario.hardware.get_component("hbm0"), bandwidth_gbps=16_000.0)
    link = next(item for item in scenario.hardware.links if item.link_id == "gpu-hbm0")
    link = replace(link, bandwidth_gbps=12_000.0)
    scenario = replace(
        scenario,
        hardware=replace(
            scenario.hardware,
            components=tuple(hbm if item.component_id == "hbm0" else item for item in scenario.hardware.components),
            links=tuple(link if item.link_id == "gpu-hbm0" else item for item in scenario.hardware.links),
        ),
    )

    payload = scenario_to_payload(scenario)
    assert payload["profiles"]["components"]["hbm"]["legacy-hbm"]["bandwidth_gb_s"] == 4096.0
    parsed = scenario_from_dict(payload)
    profile = parsed.resolve_component_profile("hbm0")
    hop = TopologyRouter(parsed.hardware).route("gpu0", "hbm0", 1024)[0]

    # The profile is a calibrated aggregate service model and remains intact;
    # the physical link still limits the topology hop independently.
    assert profile.bandwidth_gb_s == 4096.0
    assert hop.bandwidth_gbps == 12_000.0
    # Without an explicit memory_service_owner on the component, the link is
    # intentionally kept as its own contention resource.
    assert hop.resource_id == "link.gpu-hbm0.gpu0->hbm0"


def test_storage_without_write_capability_does_not_invent_a_write_budget():
    storage = ComponentSpec(
        "hbf0",
        "hbf",
        ports=(PortSpec("p", "UCIe", "endpoint", bandwidth_gbps=2048),),
        read_bandwidth_gbps=128.0,
        write_bandwidth_gbps=0.0,
    )
    assert storage.directional_bandwidth_gbps("write") == 0.0


def test_local_memory_transfer_uses_one_shared_resource_without_endpoint_recharge():
    scenario = build_matching_scenario(
        8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=16, gpu_layers=0
    )
    profile = scenario.resolve_component_profile("hostmem0")
    phases = TopologyRouter(scenario.hardware).transfer_phases(
        "cpu0", "hostmem0", 1024
    )
    assert [phase.name for phase in phases] == ["transfer.link00"]
    assert phases[0].demands[0].resource_id == profile.resource_id


def test_complete_native_payload_roundtrip_keeps_curated_profile_calibration():
    """A frontend save/load must not rebind an already materialized native scene."""
    scenario = build_matching_scenario(
        512, 256, ctx=2048, parallel=1, batch=64, ubatch=64, threads=16,
        gpu_layers=-1,
    )
    payload = scenario_to_payload(scenario)
    # The browser authoring form omits runtime-generated placement mirrors.
    for field in ("op_to_component", "tensor_bytes", "tensor_to_component"):
        payload["placement"].pop(field, None)
    parsed = scenario_from_dict(payload)
    assert parsed.hardware.metadata["native_hardware_preset_id"] == "local-rtx5080-9950x3d"
    assert parsed.resolve_component_profile("hbm0").resource_id == "gpu0.hbm_fabric"
    assert parsed.resolve_component_profile("hbm0").read_latency_ns == 0.0
    assert parsed.resolve_component_profile("hostmem0").resource_id == "cpu0.memory"
    assert parsed.resolve_component_profile("hostmem0").write_latency_ns == 0.0


def test_local_rtx5080_scene_binds_curated_hbm_and_local_ddr5_presets():
    payload = {
        "name": "native-llama-parity-rtx5080",
        "hardware": {
                "name": "RTX5080-local",
                "metadata": {"native_hardware_preset_id": "local-rtx5080-9950x3d"},
            "components": [
                {"component_id": "gpu0", "kind": "gpu", "ports": [{"port_id": "hbm0", "protocol": "HBM", "version": "3.0", "role": "controller", "direction": "bidirectional", "lanes": 16, "bandwidth_gbps": 7680.0, "max_links": 1}]},
                {"component_id": "cpu0", "kind": "cpu", "ports": [{"port_id": "ddr0", "protocol": "DDR", "version": "5.0", "role": "controller", "direction": "bidirectional", "lanes": 64, "bandwidth_gbps": 716.8, "max_links": 1}]},
                {"component_id": "hbm0", "kind": "hbm", "cost_profile_id": "legacy-hbm", "ports": [{"port_id": "host", "protocol": "HBM", "version": "3.0", "role": "device", "direction": "bidirectional", "lanes": 16, "bandwidth_gbps": 7680.0, "max_links": 1}], "capacity_bytes": 16303 * 1024 ** 2, "read_bandwidth_gbps": 0.0, "write_bandwidth_gbps": 0.0, "metadata": {"memory_type": "GDDR7"}},
                {"component_id": "hostmem0", "kind": "host_memory", "cost_profile_id": "legacy-host-memory", "ports": [{"port_id": "ddr0", "protocol": "DDR", "version": "5.0", "role": "device", "direction": "bidirectional", "lanes": 64, "bandwidth_gbps": 716.8, "max_links": 1}], "capacity_bytes": 1, "read_bandwidth_gbps": 716.8, "write_bandwidth_gbps": 716.8},
            ],
            "links": [
                {"link_id": "cpu-hostmem-ddr", "source_component": "cpu0", "source_port": "ddr0", "target_component": "hostmem0", "target_port": "ddr0", "protocol": "DDR", "version": "5.0", "lanes": 64, "bandwidth_gbps": 716.8},
                {"link_id": "gpu-hbm0", "source_component": "gpu0", "source_port": "hbm0", "target_component": "hbm0", "target_port": "host", "protocol": "HBM", "version": "3.0", "lanes": 16, "bandwidth_gbps": 7680.0},
            ],
        },
        "profiles": {"components": {"hbm": {"legacy-hbm": {}}, "host_memory": {"legacy-host-memory": {}}}},
    }
    # A previously constructed scene can still contain the illustrative
    # machine's shared totals even when its ports already look correct.
    payload["hardware"]["components"][2]["bandwidth_gbps"] = 4096.0
    payload["hardware"]["components"][3]["bandwidth_gbps"] = 3276.8
    bound = _bind_local_rtx5080_hardware_presets(payload)
    components = {item["component_id"]: item for item in bound["hardware"]["components"]}
    assert "component_preset_id" not in components["hbm0"]["metadata"]
    assert components["hbm0"]["metadata"]["attached_memory_preset_id"] == "gddr7-16gb-30_0-256bit"
    assert components["hbm0"]["metadata"]["memory_type"] == "GDDR7"
    assert components["hbm0"]["ports"][0]["protocol"] == "GDDR7"
    assert components["hbm0"]["capacity_bytes"] == 16_000_000_000
    assert components["hbm0"]["bandwidth_gbps"] == 7680.0
    assert components["hostmem0"]["metadata"]["component_preset_id"] == "acer-local-ddr5-128gb-5600-dual-channel"
    assert components["hostmem0"]["capacity_bytes"] == 4 * 32 * 1024 ** 3
    assert components["hostmem0"]["metadata"]["resident_access_path"] == "topology"
    assert components["hostmem0"]["bandwidth_gbps"] == 716.8
    assert components["hostmem0"]["ports"][0]["protocol"] == "DDR5"
    links = {item["link_id"]: item for item in bound["hardware"]["links"]}
    assert links["cpu-hostmem-ddr"]["protocol"] == "DDR5"
    assert links["gpu-hbm0"]["bandwidth_gbps"] == 7680.0
    assert links["gpu-hbm0"]["protocol"] == "GDDR7"


def test_architecture_preset_is_not_rebound_by_legacy_rtx5080_scene_name():
    hardware = materialize_architecture_payload(
        "local-native-rtx5080-9950x3d-gddr7-ddr5"
    )
    payload = {"name": "native-llama-parity-rtx5080", "hardware": hardware}
    bound = _bind_local_rtx5080_hardware_presets(payload)
    assert bound is payload
    assert bound["hardware"]["metadata"]["architecture_preset"]["id"] == (
        "local-native-rtx5080-9950x3d-gddr7-ddr5"
    )
    assert bound["hardware"]["name"] == "本机 Native RTX 5080 + Ryzen 9 9950X3D"
