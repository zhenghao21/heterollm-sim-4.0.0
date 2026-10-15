"""GDDR generation belongs to version, with current physical service intact."""
from dataclasses import replace

import pytest

from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.data_motion import endpoint_service, resolve_service
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from heterollm_sim.protocol_presets import protocol_preset_detail
from heterollm_sim.topology import validate_topology


PRESETS = (
    ("gddr6-16gb-20_0-256bit", "gddr6-20_0-256", "GDDR6"),
    ("gddr6x-16gb-21_0-256bit", "gddr6x-21_0-256", "GDDR6X"),
    ("gddr7-16gb-30_0-256bit", "gddr7-30_0-256", "GDDR7"),
)


def _hardware(component_preset, protocol_preset):
    detail = protocol_preset_detail(protocol_preset)
    defaults = detail["simulation_defaults"]
    memory = replace(get_component_preset(component_preset).component,
                     component_id="memory", ports=(PortSpec("memory", **defaults["target_port"]),))
    controller = ComponentSpec("gpu", "gpu", ports=(PortSpec("memory", **defaults["source_port"]),))
    link = LinkSpec("local", "gpu", "memory", "memory", "memory", **{
        **defaults["link"], "metadata": {**defaults["link"]["metadata"], "service_ref": "memory.access"}})
    return HardwareSpec("current-gddr", (controller, memory), (link,))


@pytest.mark.parametrize("component_preset,protocol_preset,generation", PRESETS)
def test_current_gddr_presets_validate_and_share_one_physical_service(component_preset, protocol_preset, generation):
    hardware = _hardware(component_preset, protocol_preset)
    report = validate_topology(hardware)
    assert report.is_valid, report.format_en()
    detail = protocol_preset_detail(protocol_preset)
    assert detail["preset"]["protocol"] == "GDDR"
    assert detail["preset"]["generation"] == generation
    memory = hardware.get_component("memory")
    service = resolve_service(memory)
    router = TopologyRouter(hardware)
    for source, target in (("gpu", "memory"), ("memory", "gpu")):
        hop, = router.route(source, target, 256)
        assert hop.service_ref == "memory.access"
        assert hop.resource_id == service.resource_id
        assert hop.demands(256) == ()
    for operation in ("read", "write"):
        endpoint = endpoint_service(memory, 256, operation=operation, name="gddr", page_offset_bytes=0)
        assert endpoint is not None
        assert endpoint.metadata["physical_memory_config"]["generation"] == generation
        physical = endpoint.metadata["physical_execution_by_owner"][service.resource_id]
        assert physical["kind"] == "GDDR"
        assert physical["service_ns"] > 0
        assert physical["physical_bytes"] >= 256


@pytest.mark.parametrize("obsolete", ("GDDR6", "gddr_6", "GDDR6X", "gddr_6x", "GDDR7", "gddr_7"))
@pytest.mark.parametrize("location", ("link", "port", "both", "unconnected_port"))
def test_generation_shaped_protocols_fail_actual_topology_validation(obsolete, location):
    hardware = _hardware(*PRESETS[2][:2])
    if location in {"link", "both"}:
        hardware = replace(hardware, links=(replace(hardware.links[0], protocol=obsolete),))
    if location in {"port", "both", "unconnected_port"}:
        hardware = replace(hardware, components=tuple(replace(component, ports=tuple(
            replace(port, protocol=obsolete) for port in component.ports)) for component in hardware.components))
    if location == "unconnected_port":
        hardware = replace(hardware, links=(), require_connected=False)
    report = validate_topology(hardware)
    assert not report.is_valid
    if location in {"link", "both"}:
        assert "obsolete_gddr_link_protocol" in report.error_codes
    if location in {"port", "both", "unconnected_port"}:
        assert "obsolete_gddr_port_protocol" in report.error_codes


def test_current_native_gddr7_and_ddr5_architecture_remains_legal():
    hardware = hardware_from_dict(materialize_architecture_payload("local-native-rtx5080-9950x3d-gddr7-ddr5"))
    report = validate_topology(hardware)
    assert report.is_valid, report.format_en()


def test_gddr_generation_mismatch_is_checked_from_version():
    hardware = _hardware(*PRESETS[2][:2])
    hardware = replace(hardware, links=(replace(hardware.links[0], version="GDDR6"),))
    report = validate_topology(hardware)
    assert "gddr_generation_mismatch" in report.error_codes
    assert "gddr_memory_port_generation_mismatch" in report.error_codes
