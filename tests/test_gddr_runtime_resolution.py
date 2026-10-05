from types import SimpleNamespace

import pytest

from heterollm_sim.cost_models import GDDRProfile, HBMProfile, HostMemoryProfile
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec
from heterollm_sim import planner


class _RuntimeScenario:
    """Small real HardwareSpec backed scenario for the runtime resolver."""

    def __init__(self, components, links, profiles):
        self.hardware = HardwareSpec(
            "runtime-memory-resolution",
            tuple(components),
            tuple(links),
            require_connected=False,
        )
        self.placement = SimpleNamespace(metadata={})
        self._profiles = profiles

    def resolve_component_profile(self, component, expected_type=None):
        component_id = component if isinstance(component, str) else component.component_id
        profile = self._profiles[component_id]
        if expected_type is not None and not isinstance(profile, expected_type):
            raise TypeError("unexpected profile type")
        return profile


def _component(component_id, kind, profile_id=None, *, bandwidth_gbps=0.0):
    return ComponentSpec(
        component_id,
        kind,
        cost_profile_id=profile_id,
        bandwidth_gbps=bandwidth_gbps,
    )


def _link(link_id, source, target, protocol, bandwidth_gbps=1000.0):
    return LinkSpec(
        link_id,
        source,
        "port",
        target,
        "port",
        protocol,
        bandwidth_gbps=bandwidth_gbps,
    )


def _scenario_for_memory(memory_kind):
    gpu = _component("gpu0", "gpu", "gpu")
    memory = _component("memory0", memory_kind, "memory", bandwidth_gbps=1000.0)
    return _RuntimeScenario(
        (gpu, memory),
        (_link("gpu-memory", "gpu0", "memory0", memory_kind.upper()),),
        {
            "gpu0": object(),
            "memory0": (
                GDDRProfile(generation="GDDR7")
                if memory_kind == "gddr"
                else HBMProfile()
            ),
        },
    )


def test_runtime_memory_resolver_accepts_gpu_gddr_and_hbm(monkeypatch):
    # The helper consults the parallel plan for an optional explicit rank
    # binding.  An empty real plan is sufficient to exercise fallback lookup.
    monkeypatch.setattr(planner, "_parallel_plan", lambda scenario: SimpleNamespace(ranks=()))

    assert planner._compute_local_runtime_memory_component_id(
        _scenario_for_memory("gddr"), "gpu0"
    ) == "memory0"
    assert planner._compute_local_runtime_memory_component_id(
        _scenario_for_memory("hbm"), "gpu0"
    ) == "memory0"


def test_runtime_memory_resolver_accepts_cpu_host_memory(monkeypatch):
    monkeypatch.setattr(planner, "_parallel_plan", lambda scenario: SimpleNamespace(ranks=()))
    cpu = _component("cpu0", "cpu", "cpu")
    host = _component("host0", "host_memory", "host")
    scenario = _RuntimeScenario(
        (cpu, host),
        (_link("cpu-host", "cpu0", "host0", "DDR"),),
        {
            "cpu0": object(),
            "host0": HostMemoryProfile(),
        },
    )

    assert planner._compute_local_runtime_memory_component_id(scenario, "cpu0") == "host0"


def test_runtime_memory_resolver_rejects_non_compute_target(monkeypatch):
    monkeypatch.setattr(planner, "_parallel_plan", lambda scenario: SimpleNamespace(ranks=()))
    scenario = _scenario_for_memory("gddr")
    with pytest.raises(ValueError, match="must be a CPU or GPU"):
        planner._compute_local_runtime_memory_component_id(scenario, "memory0")


def test_gddr_component_capacity_matches_physical_config_capacity():
    capacity = 16 * 1000**3
    component = ComponentSpec(
        "gddr0",
        "gddr",
        cost_profile_id="gddr",
        capacity_bytes=capacity,
        bandwidth_gbps=5120.0,
        metadata={
            "generation": "GDDR6",
            "physical_memory_config": {
                "kind": "GDDR",
                "generation": "GDDR6",
                "channels": 1,
                "data_lanes": 8,
                "data_width_bits": 32,
                "effective_pin_data_rate_gbps": 20,
                "bandwidth_input_mode": "data_rate",
                "stacks": 1,
                "dies_per_stack": 1,
                "ranks_per_channel": 1,
                "bank_groups_per_rank": 4,
                "banks_per_group": 4,
                "rows_per_bank": 131072,
                "row_bytes": 8192,
                "burst_bytes": 64,
                "capacity_bytes": capacity,
                "metadata": {"capacity_unit": "GB"},
            },
        },
    )
    assert component.capacity_bytes == capacity
    assert component.metadata["physical_memory_config"]["capacity_bytes"] == capacity
