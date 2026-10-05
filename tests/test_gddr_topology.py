from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from heterollm_sim.topology import validate_topology


def _hardware(memory_generation="GDDR7", link_generation="GDDR7"):
    gpu = ComponentSpec(
        component_id="gpu0",
        kind="gpu",
        metadata={"supported_gddr_generations": ["GDDR6", "GDDR6X", "GDDR7"]},
        ports=(PortSpec("gddr", "GDDR", "controller", version=link_generation, lanes=256, bandwidth_gbps=7680.0),),
    )
    memory = ComponentSpec(
        component_id="gddr0",
        kind="gddr",
        generation=memory_generation,
        metadata={"generation": memory_generation},
        ports=(PortSpec("host", "GDDR", "device", version=memory_generation, lanes=256, bandwidth_gbps=7680.0),),
    )
    link = LinkSpec(
        link_id="gpu-gddr",
        source_component="gpu0",
        source_port="gddr",
        target_component="gddr0",
        target_port="host",
        protocol="GDDR",
        version=link_generation,
        lanes=256,
        bandwidth_gbps=7680.0,
    )
    return HardwareSpec("gddr-topology", (gpu, memory), (link,))


def test_gddr_same_generation_link_is_valid():
    report = validate_topology(_hardware())
    assert report.is_valid, report.format_en()


def test_gddr_generation_mismatch_is_rejected():
    report = validate_topology(_hardware(memory_generation="GDDR7", link_generation="GDDR6"))
    assert not report.is_valid
    assert any(issue.code == "gddr_generation_mismatch" for issue in report.errors)
