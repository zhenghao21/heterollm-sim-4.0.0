import json
import math

from heterollm_sim.component_presets import materialize_component_payload
from heterollm_sim.memory_types import DramConfig, parse_physical_memory_config


PRESET_ID = "acer-local-ddr5-128gb-5600-dual-channel"


def test_local_ddr5_preset_physical_config_roundtrips_with_matching_capacity_and_bandwidth():
    component = materialize_component_payload(PRESET_ID)
    raw = component["metadata"]["physical_memory_config"]
    restored = json.loads(json.dumps(raw))
    config = parse_physical_memory_config(restored)

    assert isinstance(config, DramConfig)
    assert config.kind.value == "DDR"
    assert config.generation == "DDR5"
    assert config.channels == 2
    assert config.data_width_bits == 64
    assert config.data_rate_mt_s == 5600
    assert config.physical_interface_bandwidth_gb_s == 89.6
    assert config.capacity_bytes == 128 * 1024**3
    assert config.computed_capacity_bytes == config.capacity_bytes

    # The 100 ns host-visible cost profile remains independent of the DRAM
    # core's representative command-to-data timing assumptions.
    assert component["metadata"]["cost_profile_template"]["read_latency_ns"] == 100.0
    assert config.read_latency_ns != 100.0
    assert config.metadata["parameter_basis"] == "analytical_equivalent_assumptions"
    assert math.isclose(config.burst_interval_ns, 64 / 44.8)
