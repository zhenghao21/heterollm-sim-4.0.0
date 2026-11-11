"""HBM preset geometry comes from its declared interface, not a generation guess."""
from dataclasses import asdict

import pytest

from heterollm_sim.architecture_presets import (
    ANALYTICAL_APPROXIMATION,
    _component,
    _gh200_nvl2,
    _gh200_superchip,
    _hbm_accelerator_cluster,
    _port,
)
from heterollm_sim.memory_types import DramConfig, parse_physical_memory_config
from heterollm_sim.reference import build_reference_scenario


def component(ports, **metadata):
    return _component("memory", "hbm", ports,
        capacity_bytes=64_000_000_000, read_bandwidth_gbps=16384.,
        write_bandwidth_gbps=16384., role="active_memory", extra_metadata=metadata)


@pytest.mark.parametrize("generation,channels,bandwidth,rate", [
    ("HBM3", 16, 6553.6, 6400.), ("HBM4", 32, 16384., 8000.),
])
def test_hbm_builder_retains_declared_channels_width_and_rate(generation, channels, bandwidth, rate):
    port = _port("host", "HBM", "device", version=generation,
                 lanes=channels, bandwidth_gbps=bandwidth)
    item = _component("memory", "hbm", (port,), capacity_bytes=64_000_000_000,
        read_bandwidth_gbps=bandwidth, write_bandwidth_gbps=bandwidth, role="active_memory")
    config = parse_physical_memory_config(item.metadata["physical_memory_config"])
    assert config.channels == channels
    assert config.lane_count == channels * 2
    assert config.generation == generation
    assert config.lane_count * config.data_width_bits == channels * 64
    assert config.data_rate_mt_s == rate
    assert config.physical_interface_bandwidth_gb_s == bandwidth / 8
    assert config.capacity_bytes == item.capacity_bytes


def test_hbm_source_is_memory_endpoint_not_first_auxiliary_port():
    item = component((
        _port("control", "PCIe", "endpoint", version="5.0", lanes=4, bandwidth_gbps=32.),
        _port("memory", "HBM", "device", version="HBM4", lanes=32, bandwidth_gbps=16384.),
    ))
    config = parse_physical_memory_config(item.metadata["physical_memory_config"])
    assert config.generation == "HBM4"
    assert config.channels == 32


@pytest.mark.parametrize("ports", [
    (),
    (_port("wrong-role", "HBM", "controller", version="HBM4", lanes=32),),
    (_port("first", "HBM", "device", version="HBM4", lanes=32),
     _port("second", "HBM", "device", version="HBM4", lanes=32)),
])
def test_ambiguous_or_missing_hbm_endpoint_requires_explicit_config(ports):
    with pytest.raises(ValueError, match="exactly one HBM memory endpoint"):
        component(ports)


def test_explicit_physical_configuration_remains_authoritative():
    declared = asdict(DramConfig(kind="HBM", channels=5, banks_per_group=2,
        rows_per_bank=131072, row_bytes=65536, capacity_bytes=64_000_000_000,
        interface_bandwidth_gb_s=2048.))
    item = component((), physical_memory_config=declared)
    assert item.metadata["physical_memory_config"] == declared
    assert parse_physical_memory_config(item.metadata["physical_memory_config"]).channels == 5


def test_bundled_reference_hbm_uses_standard_organization():
    scenario = build_reference_scenario()
    hbm = scenario.hardware.get_component("hbm0")
    config = parse_physical_memory_config(hbm.metadata["physical_memory_config"])
    assert config.generation == "HBM3"
    assert (config.stacks, config.dies_per_stack, config.channels,
            config.subchannels_per_channel, config.pseudo_channels_per_channel) == (1, 8, 16, 1, 2)
    assert (config.bank_count, config.rows_per_bank, config.row_bytes) == (32, 16384, 1024)
    assert config.capacity_bytes == config.computed_capacity_bytes == 16 * 1024**3
    assert config.physical_interface_bandwidth_gb_s == 512.
    assert (config.data_width_bits, config.data_rate_mt_s, config.burst_interval_ns) == (32, 4000., 2.)
    assert config.metadata["geometry_status"] == "standard_organization"
    assert "dies_per_stack" in config.metadata["unknown_parameters"]


@pytest.mark.parametrize("generation,hbm4", [("HBM3E", False), ("HBM4", True)])
def test_controller_policy_notes_reach_the_returned_and_parsed_config(generation, hbm4):
    from heterollm_sim.hbm_geometry import make_hbm_geometry_config
    payload = make_hbm_geometry_config(generation=generation, capacity_bytes=16 * 1024**3,
                                       read_bandwidth_gbps=8192., write_bandwidth_gbps=8192.)
    for metadata in (payload["metadata"], parse_physical_memory_config(payload).metadata):
        assert {"address_mapping_basis", "controller_policy", "queue_policy"} <= set(metadata)
        assert metadata["queue_policy"]["read_depth_per_pc"] == payload["read_queue_depth_per_pc"] == 128
        assert ("timing_assumptions" in metadata) is hbm4


@pytest.mark.parametrize("generation,channels", [("HBM3", 16), ("HBM4", 32)])
@pytest.mark.parametrize("builder", ["cluster", "superchip", "nvl2"])
def test_physical_stack_channels_reach_controller_link_endpoint_and_core(builder, generation, channels):
    # Synthetic geometry exercises every generation-parameterized wrapper;
    # it does not assert that these products ship with an HBM4 variant.
    capacity_gb, bandwidth_gbps, stack_count = 64, 32768., 2
    if builder == "cluster":
        definition = _hbm_accelerator_cluster(
            preset_id="synthetic-geometry", name="Synthetic geometry", gpu_model="Synthetic",
            memory_kind=generation, channels_per_stack=channels,
            memory_capacity_gb=capacity_gb, memory_bandwidth_gbps=bandwidth_gbps,
            peak_ops_per_s=1e12, count=1, fabric_protocol="NVLink", fabric_version="4.0",
            fabric_bandwidth_gbps=1000., vendor="Synthetic", family="Geometry", sources=(),
            support_level=ANALYTICAL_APPROXIMATION, memory_physical_unit_count=stack_count,
        )
        controller_count = 1
    elif builder == "superchip":
        definition = _gh200_superchip(
            channels_per_stack=channels, hbm_generation=generation, hbm_stack_count=stack_count,
            hbm_capacity_gb=capacity_gb, hbm_bandwidth_gbps=bandwidth_gbps,
        )
        controller_count = 1
    else:
        definition = _gh200_nvl2(
            channels_per_stack=channels, hbm_generation=generation, hbm_stack_count=stack_count,
            hbm_capacity_gb_per_superchip=capacity_gb,
            hbm_bandwidth_gbps_per_superchip=bandwidth_gbps,
        )
        controller_count = 2

    hardware = definition.hardware
    links = [link for link in hardware.links if link.protocol == "HBM"]
    assert len(links) == stack_count * controller_count
    memory_components = []
    for link in links:
        controller = hardware.get_component(link.source_component)
        memory = hardware.get_component(link.target_component)
        source_port = next(port for port in controller.ports if port.port_id == link.source_port)
        target_port = next(port for port in memory.ports if port.port_id == link.target_port)
        config = parse_physical_memory_config(memory.metadata["physical_memory_config"])
        assert source_port.lanes == link.lanes == target_port.lanes == config.channels == channels
        assert source_port.version == link.version == target_port.version == config.generation == generation
        for port in (source_port, target_port):
            assert port.metadata["channels_per_stack"] == channels
            assert port.metadata["physical_channel_total"] == channels
            assert port.bandwidth_gbps == bandwidth_gbps / stack_count
        assert link.bandwidth_gbps == bandwidth_gbps / stack_count
        assert config.physical_interface_bandwidth_gb_s == link.bandwidth_gbps / 8
        assert config.lane_count * config.data_width_bits == channels * 64
        assert config.data_rate_mt_s == 8000. * 32 / channels
        memory_components.append(memory)
    assert sum(item.capacity_bytes for item in memory_components) == capacity_gb * 10**9 * controller_count
    assert sum(link.bandwidth_gbps for link in links) == bandwidth_gbps * controller_count
