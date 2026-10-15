"""The physical config remains authoritative across scenario and direct entry points."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim.config import _seed_omitted_memory_profile_bandwidths
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_service
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import AccessRequest, make_hbm_config, make_hbf_config
from heterollm_sim.physical_contract import (
    parse_execution_physical_config, physical_memory_bandwidth_caps_gbps,
    require_physical_memory_config,
)
from heterollm_sim.reference import build_reference_scenario


def component(kind="hbm", **updates):
    config = (make_hbm_config(interface_bandwidth_gb_s=512.,
                             read_bandwidth_gb_s=256., write_bandwidth_gb_s=64.)
              if kind == "hbm" else make_hbf_config(host_bandwidth_gb_s=512.,
                  host_read_bandwidth_gb_s=256., host_write_bandwidth_gb_s=64.))
    base = ComponentSpec("memory0", kind, capacity_bytes=config.capacity_bytes,
        bandwidth_gbps=4096., read_bandwidth_gbps=2048., write_bandwidth_gbps=512.,
        metadata={"physical_memory_config": asdict(config)})
    return replace(base, **updates)


@pytest.mark.parametrize("kind", ["hbm", "hbf"])
def test_shared_core_interface_is_not_the_largest_directional_rate(kind):
    memory = component(kind)
    assert physical_memory_bandwidth_caps_gbps(memory) == (4096., 2048., 512.)
    assert require_physical_memory_config(memory).capacity_bytes == memory.capacity_bytes


@pytest.mark.parametrize("kind", ["hbm", "hbf"])
@pytest.mark.parametrize("direction", ["read", "write"])
def test_explicit_direction_above_core_is_rejected_even_in_shared_mode(kind, direction):
    memory = component(kind, **{direction + "_bandwidth_gbps": 4096.})
    # Legacy ComponentSpec.directional_bandwidth_gbps() masks these fields
    # when bandwidth_mode is omitted; the physical boundary must not.
    with pytest.raises(ValueError, match="component " + direction + " bandwidth exceeds"):
        require_physical_memory_config(memory)


@pytest.mark.parametrize("kind", ["hbm", "hbf"])
def test_smaller_component_rates_do_not_masquerade_as_core_configuration(kind):
    memory = component(kind, bandwidth_gbps=256., read_bandwidth_gbps=128.,
                       write_bandwidth_gbps=64.)
    assert physical_memory_bandwidth_caps_gbps(memory) == (4096., 2048., 512.)
    core = require_physical_memory_config(memory)
    assert physical_memory_bandwidth_caps_gbps(memory, physical_config=core) == (
        4096., 2048., 512.)


def scenario_with_asymmetric_hbm(*, profile_read=128., profile_write=64.):
    scenario = build_reference_scenario()
    original = scenario.hardware.get_component("hbm0")
    physical = {**original.metadata["physical_memory_config"],
                "read_bandwidth_gb_s": 128., "write_bandwidth_gb_s": 64.}
    updated = replace(original, read_bandwidth_gbps=0., write_bandwidth_gbps=0.,
        metadata={**original.metadata, "physical_memory_config": physical})
    hardware = replace(scenario.hardware, components=tuple(
        updated if c.component_id == "hbm0" else c for c in scenario.hardware.components))
    registry = dict(scenario.component_profiles["hbm"])
    registry[original.cost_profile_id] = replace(registry[original.cost_profile_id],
        bandwidth_gb_s=128., efficiency=1., read_bandwidth_gb_s=profile_read,
        write_bandwidth_gb_s=profile_write)
    return replace(scenario, hardware=hardware,
                   component_profiles={**scenario.component_profiles, "hbm": registry})


def test_scenario_reports_core_caps_separately_from_effective_profile():
    scenario = scenario_with_asymmetric_hbm()
    memory = scenario.hardware.get_component("hbm0")
    service = memory.metadata["memory_service"]
    assert (service["physical_bandwidth_gb_s"], service["physical_read_bandwidth_gb_s"],
            service["physical_write_bandwidth_gb_s"]) == (512., 128., 64.)
    assert (service["bandwidth_gb_s"], service["read_bandwidth_gb_s"],
            service["write_bandwidth_gb_s"]) == (128., 128., 64.)
    assert resolve_service(memory).write_bandwidth_gb_s == 64.


def test_omitted_profile_uses_core_direction_caps_and_keeps_authored_override():
    scenario = scenario_with_asymmetric_hbm()
    memory = scenario.hardware.get_component("hbm0")
    seeded = _seed_omitted_memory_profile_bandwidths(
        {"hbm": {memory.cost_profile_id: {"write_bandwidth_gb_s": 32.}}}, scenario.hardware)
    profile = seeded["hbm"][memory.cost_profile_id]
    assert profile["bandwidth_gb_s"] == 512.
    assert profile["read_bandwidth_gb_s"] == 128.
    assert profile["write_bandwidth_gb_s"] == 32.


def test_shared_template_does_not_inherit_one_members_explicit_owner():
    scenario = build_reference_scenario()
    first = scenario.hardware.get_component("hbm0")
    second = scenario.hardware.get_component("hbm1")
    metadata = dict(second.metadata)
    metadata.pop("memory_service_owner")
    second = replace(second, cost_profile_id=first.cost_profile_id, metadata=metadata)
    hardware = replace(scenario.hardware, components=tuple(
        second if c.component_id == "hbm1" else c for c in scenario.hardware.components))
    seeded = _seed_omitted_memory_profile_bandwidths(
        {"hbm": {first.cost_profile_id: {}}}, hardware)
    assert "resource_id" not in seeded["hbm"][first.cost_profile_id]


@pytest.mark.parametrize("direction", ["read", "write"])
def test_scenario_rejects_profile_above_core_even_without_directional_mode(direction):
    with pytest.raises(ValueError, match="profile " + direction + " bandwidth exceeds"):
        scenario_with_asymmetric_hbm(**{"profile_" + direction: 200.})


@pytest.mark.parametrize("kind", ["hbm", "hbf"])
def test_direct_service_ignores_forged_physical_caps_and_rejects_effective_overrun(kind):
    memory = component(kind)
    memory = replace(memory, metadata={**memory.metadata, "memory_service": {
        "physical_bandwidth_gb_s": 1024., "physical_write_bandwidth_gb_s": 128.,
        "bandwidth_gb_s": 256., "read_bandwidth_gb_s": 256.,
        "write_bandwidth_gb_s": 128.}})
    with pytest.raises(ValueError, match="effective write bandwidth cannot exceed"):
        resolve_service(memory)


@pytest.mark.parametrize("kind", ["hbm", "hbf"])
def test_execution_parser_preserves_typed_configs_and_rejects_partial_serialized_geometry(kind):
    physical = require_physical_memory_config(component(kind))
    assert parse_execution_physical_config(physical) is physical
    assert parse_execution_physical_config(asdict(physical)) == physical
    raw = asdict(physical)
    raw.pop("channels")
    with pytest.raises(ValueError, match="missing explicit fields: channels"):
        parse_execution_physical_config(raw)
    raw.pop("kind")
    with pytest.raises(ValueError, match="missing explicit fields: kind"):
        parse_execution_physical_config(raw)


def test_template_profile_owner_creates_independent_core_and_allocator_per_component():
    scenario = build_reference_scenario()
    registry = dict(scenario.component_profiles["hbm"])
    components = []
    for memory in scenario.hardware.components:
        if memory.component_id in {"hbm0", "hbm1"}:
            registry[memory.cost_profile_id] = replace(registry[memory.cost_profile_id],
                                                      resource_id="hbm.channel")
            metadata = dict(memory.metadata)
            metadata.pop("memory_service_owner")
            memory = replace(memory, metadata=metadata)
        components.append(memory)
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)),
                       component_profiles={**scenario.component_profiles, "hbm": registry})
    context = PhysicalRuntimeContext()
    completions = []
    for identity in ("hbm0", "hbm1"):
        memory = scenario.hardware.get_component(identity)
        owner = resolve_service(memory).physical_owner
        assert owner == identity + ".hbm_fabric"
        core = context.runtime(require_physical_memory_config(memory), owner).core
        completions.append(core.submit(AccessRequest(identity, "read", 0, 64)).completion_ns)
    assert completions[0] == completions[1]
    assert set(context.runtimes) == set(context.allocators) == {"hbm0.hbm_fabric", "hbm1.hbm_fabric"}


@pytest.mark.parametrize("explicit_owner", [None, "physical-hbm"])
def test_explicit_custom_profile_or_component_owner_is_preserved(explicit_owner):
    scenario = build_reference_scenario()
    memory = scenario.hardware.get_component("hbm0")
    metadata = dict(memory.metadata)
    metadata.pop("memory_service_owner")
    if explicit_owner:
        metadata["memory_service_owner"] = explicit_owner
    hardware = replace(scenario.hardware, components=tuple(
        replace(c, metadata=metadata) if c.component_id == "hbm0" else c
        for c in scenario.hardware.components))
    registry = dict(scenario.component_profiles["hbm"])
    registry[memory.cost_profile_id] = replace(registry[memory.cost_profile_id], resource_id="authored-owner")
    scenario = replace(scenario, hardware=hardware,
                       component_profiles={**scenario.component_profiles, "hbm": registry})
    assert resolve_service(scenario.hardware.get_component("hbm0")).physical_owner == (
        explicit_owner or "authored-owner")
