"""Bundled storage presets must execute with an explicit physical core."""

import pytest

from heterollm_sim.architecture_presets import _LEGACY_ARCHITECTURE_PRESETS, _PRESETS
from heterollm_sim.component_presets import _PRESETS as COMPONENT_PRESETS
from heterollm_sim.memory_types import DramConfig, NandConfig
from heterollm_sim.physical_contract import (
    PHYSICAL_MEMORY_COMPONENT_KINDS,
    require_physical_memory_config,
)
from heterollm_sim.reference import build_reference_scenario


_STORAGE_COMPONENTS = [
    component
    for component in (
        *(preset.component for preset in COMPONENT_PRESETS),
        *(component for preset in (*_LEGACY_ARCHITECTURE_PRESETS, *_PRESETS)
          for component in preset.hardware.components),
        *build_reference_scenario().hardware.components,
    )
    if component.normalized_kind in PHYSICAL_MEMORY_COMPONENT_KINDS
]


@pytest.mark.parametrize(
    "component",
    _STORAGE_COMPONENTS,
    ids=lambda component: component.component_id,
)
def test_bundled_storage_has_matching_physical_capacity_and_bandwidth(component):
    config = require_physical_memory_config(component)
    assert config.capacity_bytes == component.capacity_bytes
    assert config.metadata["parameter_basis"] == "analytical_equivalent_assumptions" or (
        config.metadata.get("parameter_basis") is not None
    )
    if isinstance(config, DramConfig):
        assert config.physical_interface_bandwidth_gb_s * 8 >= component.shared_bandwidth_gbps
    else:
        assert isinstance(config, NandConfig)
        assert config.host_bandwidth_gb_s * 8 >= component.read_bandwidth_gbps


def test_reference_hbm_profiles_bind_one_stack_each():
    scenario = build_reference_scenario()
    hbm = [component for component in scenario.hardware.components if component.kind == "hbm"]
    profiles = scenario.component_profiles["hbm"]
    assert len(hbm) == len(profiles) == 8
    assert len({profiles[component.cost_profile_id].resource_id for component in hbm}) == 8
    for component in hbm:
        profile = profiles[component.cost_profile_id]
        assert component.metadata["memory_service_owner"] == profile.resource_id
        assert profile.bandwidth_gb_s == component.read_bandwidth_gbps / 8.0
