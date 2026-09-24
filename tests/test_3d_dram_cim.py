"""3D topology, shared transport causality and static thermal operating points."""

from dataclasses import replace
import json

import pytest

from heterollm_sim.architecture_presets import (
    architecture_preset_detail, materialize_architecture_payload,
)
from heterollm_sim.communication import TopologyRouter, declared_resource_owners
from heterollm_sim.config import hardware_from_dict, scenario_from_dict
from heterollm_sim.cost_models import GemmWorkload, estimate_cim_gemm, estimate_gpu_gemm
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive
from heterollm_sim.thermal import ThermalOperatingPoint, apply_thermal_operating_point
from heterollm_sim.topology import validate_topology
from heterollm_sim.web import scenario_to_payload


PRESET = "soc-2x-dram-sram-cim"
SHARED = PRESET + "-shared-phy-noc"


def hardware(preset=PRESET):
    return hardware_from_dict(materialize_architecture_payload(preset))


def scenario():
    base = build_reference_scenario()
    hw = hardware()
    # Reference profiles are scaffolds, not measured SoC/DRAM characteristics.
    # Separate controller identities are essential even at equal numeric rates.
    registry = {kind: dict(profiles) for kind, profiles in base.component_profiles.items()}
    components = []
    for component in hw.components:
        if component.component_id in {"dram0", "dram1"}:
            profile_id = component.component_id + "-analytical"
            registry["host_memory"][profile_id] = replace(
                registry["host_memory"]["legacy-host-memory"], bandwidth_gb_s=256.0,
                resource_id=component.metadata["memory_service_owner"],
            )
            component = replace(component, cost_profile_id=profile_id)
        components.append(component)
    hw = replace(hw, components=tuple(components))
    placement = replace(
        base.placement, hardware_name=hw.name,
        parallel=replace(base.placement.parallel, rank_mapping=(replace(
            base.placement.parallel.rank_mapping[0], component_id="soc0", memory_component_id="dram0",
        ),)),
        kv_policy=replace(base.placement.kv_policy, cache_component="dram0", offload_component="dram1", offload_ratio=0.0),
    )
    return replace(base, hardware=hw, placement=placement, component_profiles=registry,
                   host_orchestration_profile=replace(base.host_orchestration_profile, gpu_component_id="soc0"),
                   cim_interconnect=None)


def transfer_elapsed(hw, sources=("dram0", "dram1")):
    router = TopologyRouter(hw)
    tasks = []
    for source in sources:
        tasks.extend(router.transfer_pipeline(
            source, "soc0", 1 << 20, chunk_size_bytes=1 << 20,
            buffer_capacity_bytes=1 << 20, max_inflight_chunks=1,
            name=source, buffer_id=source + ".buffer",
        ).tasks)
    kernel = UnifiedEventKernel.from_closed_graph(tuple(tasks), resource_owners=router.resource_owners)
    while kernel.has_active_tasks:
        assert kernel.step() is not None
    return kernel.makespan_ns


@pytest.mark.parametrize("preset", [PRESET, SHARED])
def test_3d_presets_load_and_round_trip_without_calibration_claim(preset):
    payload = materialize_architecture_payload(preset)
    hw = hardware_from_dict(json.loads(json.dumps(payload, allow_nan=False)))
    assert to_primitive(hw) == payload
    report = validate_topology(hw)
    assert report.is_valid, report.format_en()
    components = hw.component_map()
    assert set(components) == {"soc0", "dram0", "dram1", "cim0", "cpu0", "host_memory0"}
    assert components["soc0"].kind == "gpu"
    assert components["soc0"].metadata["performance_model"] == "analytical_gpu_proxy"
    assert components["soc0"].die_id == components["cim0"].die_id
    assert [components[name].metadata["stack_layer"] for name in ("soc0", "dram0", "dram1")] == [0, 1, 2]
    assert "stack_id" not in components["host_memory0"].metadata
    assert all(not component.metadata["calibrated"] for component in components.values())
    assert all(components[name].metadata["resident_access_path"] == "topology" for name in ("dram0", "dram1"))
    assert not hw.metadata["thermal_operating_point"]["enabled"]
    assert hw.metadata["topology_evidence"]["scope"] == "user_authored_analytical_topology"
    assert not hw.metadata["topology_evidence"]["physical_wiring_claimed"]
    detail = architecture_preset_detail(preset)
    assert detail["compatibility"]["planner_executable"]
    assert detail["compatibility"]["requires_profile_review"]
    assert not detail["compatibility"]["requires_cpu_attachment"]
    payload["components"][0]["metadata"]["stack_layer"] = 99
    assert materialize_architecture_payload(preset)["components"][0]["metadata"]["stack_layer"] == 0


def test_vertical_paths_and_on_die_noc_are_real_bidirectional_routes():
    hw = hardware()
    router = TopologyRouter(hw)
    owners = declared_resource_owners(hw)
    for index in range(2):
        dram = "dram{}".format(index)
        read = router.route(dram, "soc0", 4096)
        write = router.route("soc0", dram, 4096)
        assert len(read) == len(write) == 1
        protocols = {link.link_id: link.protocol for link in hw.links}
        assert protocols[read[0].link_id] == protocols[write[0].link_id] == "TSV"
        assert read[0].resource_id == write[0].resource_id == "link.vertical_" + dram
        assert owners["component." + dram + ".read"] == owners["component." + dram + ".write"] == dram + ".controller"
    assert [protocols[hop.link_id] for hop in router.route("dram1", "cim0", 4096)] == ["TSV", "NoC"]
    assert all(link.protocol != "UCIe" for link in hw.links)


def test_shared_phy_noc_serializes_transfers_but_controllers_remain_independent():
    independent, shared = hardware(), hardware(SHARED)
    owners = declared_resource_owners(shared)
    link_ids = ["link.vertical_dram0", "link.vertical_dram1", "link.soc_cim_noc"]
    assert len({owners[link_id] for link_id in link_ids}) == 1
    assert owners["component.dram0.read"] != owners["component.dram1.read"]
    assert {link.bandwidth_gbps for link in shared.links if link.link_id.startswith("vertical") or link.protocol == "NoC"} == {2048.0}
    # Same rates and per-transfer service; only simultaneous contention differs.
    assert transfer_elapsed(shared, ("dram0",)) == transfer_elapsed(independent, ("dram0",))
    assert transfer_elapsed(shared) > transfer_elapsed(independent)


def test_unused_dram_timing_does_not_change_single_dram_transfer():
    base = hardware()
    changed = replace(base, links=tuple(
        replace(link, bandwidth_gbps=link.bandwidth_gbps / 100, latency_ns=1e6)
        if link.link_id == "vertical_dram1" else link for link in base.links
    ))
    assert transfer_elapsed(changed, ("dram0",)) == transfer_elapsed(base, ("dram0",))
    assert transfer_elapsed(changed, ("dram1",)) > transfer_elapsed(base, ("dram1",))


@pytest.mark.parametrize("updates,code", [
    ({"stack_layer": -1}, "invalid_stack_location"),
    ({"stack_layer": True}, "invalid_stack_location"),
    ({"stack_layer": 1.5}, "invalid_stack_location"),
    ({"stack_layer": 0}, "stack_layer_collision"),
    ({"stack_id": ""}, "invalid_stack_id"),
    ({"stack_id": "other"}, "vertical_link_stack_mismatch"),
    ({"thermal_domain_id": ""}, "invalid_thermal_domain_id"),
    ({"vertical_link_id": "missing"}, "invalid_vertical_link_reference"),
])
def test_stack_metadata_rejects_invalid_annotations(updates, code):
    base = hardware()
    mutated = replace(base, components=tuple(
        replace(component, metadata={**component.metadata, **updates})
        if component.component_id == "dram0" else component for component in base.components
    ))
    assert code in {issue.code for issue in validate_topology(mutated).errors}


@pytest.mark.parametrize("updates,code", [
    ({"read_path": ["soc0", "dram0"]}, "vertical_link_read_path"),
    ({"write_path": ["dram0", "soc0"]}, "vertical_link_write_path"),
    ({"vertical_link": "true"}, "invalid_vertical_link"),
    ({"stack_id": "other"}, "vertical_link_stack_mismatch"),
])
def test_vertical_metadata_requires_explicit_correct_read_write_paths(updates, code):
    base = hardware()
    mutated = replace(base, links=tuple(
        replace(link, metadata={**link.metadata, **updates})
        if link.link_id == "vertical_dram0" else link for link in base.links
    ))
    assert code in {issue.code for issue in validate_topology(mutated).errors}


def test_same_die_ucie_remains_illegal_and_on_die_links_cannot_cross_dies():
    base = hardware()
    ucie = replace(base, links=tuple(
        replace(link, protocol="UCIe", payload="streaming") if link.link_id == "soc_cim_noc" else link
        for link in base.links
    ))
    assert "ucie_same_die" in {issue.code for issue in validate_topology(ucie).errors}
    cross_die = replace(base, components=tuple(
        replace(component, die_id="separate_cim_die") if component.component_id == "cim0" else component
        for component in base.components
    ))
    assert "on_die_link_cross_die" in {issue.code for issue in validate_topology(cross_die).errors}


def test_scenario_scaffold_with_profiles_loads_for_p8():
    original = scenario()
    loaded = scenario_from_dict(json.loads(json.dumps(scenario_to_payload(original), allow_nan=False)))
    assert loaded.placement.parallel.rank_mapping[0].memory_component_id == "dram0"
    assert loaded.host_orchestration_profile.gpu_component_id == "soc0"
    assert loaded.resolve_component_profile("dram0").resource_id == "dram0.controller"
    assert loaded.resolve_component_profile("dram1").resource_id == "dram1.controller"
    assert loaded.resolve_component_profile("cim0")
    assert validate_topology(loaded.hardware).is_valid


def test_thermal_disabled_is_identity_and_unknown_enabled_domain_fails():
    base = scenario()
    assert apply_thermal_operating_point(base, ThermalOperatingPoint("stack0.thermal")) is base
    with pytest.raises(ValueError, match="unknown thermal domain"):
        apply_thermal_operating_point(base, ThermalOperatingPoint("unknown", enabled=True, evidence="test sensitivity"))


@pytest.mark.parametrize("overrides", [
    {"enabled": 1}, {"frequency_scale": 0}, {"frequency_scale": 1.1},
    {"memory_bandwidth_scale": float("nan")}, {"link_bandwidth_scale": float("inf")},
    {"frequency_scale": True}, {"latency_scale": 0.5}, {"enabled": True},
])
def test_thermal_inputs_are_explicit_finite_deratings(overrides):
    with pytest.raises(ValueError):
        ThermalOperatingPoint("stack0.thermal", **overrides)


def test_thermal_changes_costs_links_and_profiles_without_mutating_baseline():
    base = scenario()
    before = scenario_to_payload(base)
    point = ThermalOperatingPoint("stack0.thermal", enabled=True, frequency_scale=0.5,
                                  memory_bandwidth_scale=0.5, link_bandwidth_scale=0.5,
                                  latency_scale=2.0, evidence="analytical sensitivity, not measured throttling")
    derated = apply_thermal_operating_point(base, point)
    assert scenario_to_payload(base) == before
    assert derated.placement == base.placement
    assert derated.resolve_component_profile("cpu0") is base.resolve_component_profile("cpu0")
    assert derated.resolve_component_profile("host_memory0") is base.resolve_component_profile("host_memory0")
    assert derated.resolve_component_profile("soc0").tensor_core.frequency_ghz == base.resolve_component_profile("soc0").tensor_core.frequency_ghz / 2
    assert derated.resolve_component_profile("dram0").effective_bandwidth_gb_s == base.resolve_component_profile("dram0").effective_bandwidth_gb_s / 2
    assert derated.resolve_component_profile("cim0").frequency_ghz == base.resolve_component_profile("cim0").frequency_ghz / 2
    assert [component.capacity_bytes for component in derated.hardware.components] == [component.capacity_bytes for component in base.hardware.components]
    assert declared_resource_owners(derated.hardware) == declared_resource_owners(base.hardware)
    assert validate_topology(derated.hardware).is_valid
    assert transfer_elapsed(derated.hardware) > transfer_elapsed(base.hardware)
    workload = GemmWorkload(m=512, k=1024, n=1024, activation_bits=8, weight_bits=8, output_bits=16)
    assert estimate_gpu_gemm(derated.resolve_component_profile("soc0"), derated.resolve_component_profile("dram0"), workload).service_ns > estimate_gpu_gemm(base.resolve_component_profile("soc0"), base.resolve_component_profile("dram0"), workload).service_ns
    assert estimate_cim_gemm(derated.resolve_component_profile("cim0"), workload).service_ns > estimate_cim_gemm(base.resolve_component_profile("cim0"), workload).service_ns
    loaded = scenario_from_dict(scenario_to_payload(derated))
    assert loaded.hardware.metadata["thermal_operating_point"]["mode"] == "static_derating_only"
    assert not loaded.hardware.metadata["thermal_operating_point"]["dynamic_temperature_model"]
    assert not loaded.hardware.metadata["thermal_operating_point"]["calibrated"]
    with pytest.raises(ValueError, match="unmodified baseline"):
        apply_thermal_operating_point(derated, point)


def test_identity_operating_point_does_not_change_transfer_or_compute_cost():
    base = scenario()
    unchanged = apply_thermal_operating_point(base, ThermalOperatingPoint("stack0.thermal", enabled=True, evidence="identity sensitivity check"))
    assert transfer_elapsed(unchanged.hardware) == transfer_elapsed(base.hardware)
    assert unchanged.resolve_component_profile("soc0") == base.resolve_component_profile("soc0")
    assert unchanged.resolve_component_profile("dram0") == base.resolve_component_profile("dram0")
    assert unchanged.resolve_component_profile("cim0") == base.resolve_component_profile("cim0")


@pytest.mark.parametrize("technology", ["TSV", "HybridBonding"])
def test_vertical_technology_is_explicit_not_an_alias_for_ucie(technology):
    base = hardware()
    changed = replace(base, components=tuple(
        replace(component, ports=tuple(replace(port, protocol=technology) if port.protocol == "TSV" else port for port in component.ports))
        for component in base.components
    ), links=tuple(replace(link, protocol=technology) if link.protocol == "TSV" else link for link in base.links))
    assert validate_topology(changed).is_valid
    invalid = replace(base, links=tuple(
        replace(link, protocol="UCIe", payload="streaming") if link.link_id == "vertical_dram0" else link
        for link in base.links
    ))
    assert "vertical_link_protocol" in {issue.code for issue in validate_topology(invalid).errors}


def test_vertical_direction_and_package_contracts_cannot_be_silently_changed():
    base = hardware()
    one_way = replace(base, links=tuple(replace(link, bidirectional=False) if link.link_id == "vertical_dram0" else link for link in base.links))
    assert "vertical_link_directions" in {issue.code for issue in validate_topology(one_way).errors}
    cross_package = replace(base, components=tuple(replace(component, package_id="other_package") if component.component_id == "dram0" else component for component in base.components))
    assert "stack_cross_package" in {issue.code for issue in validate_topology(cross_package).errors}


def test_thermal_derates_memory_addressable_flash_profile_and_endpoint_together():
    from tests.test_memory_tier_placement import _scenario
    base = _scenario()
    base = replace(base, hardware=replace(base.hardware, components=tuple(
        replace(c, metadata={**c.metadata, "thermal_domain_id": "flash-domain"})
        if c.component_id == "hbf0" else c for c in base.hardware.components)))
    point = ThermalOperatingPoint("flash-domain", enabled=True, memory_bandwidth_scale=0.5,
                                  latency_scale=2.0, evidence="analytical sensitivity test")
    slowed = apply_thermal_operating_point(base, point)
    before = base.resolve_component_profile("hbf0")
    after = slowed.resolve_component_profile("hbf0")
    assert after.bandwidth_gb_s == before.bandwidth_gb_s * 0.5
    assert after.read_latency_ns == before.read_latency_ns * 2
    endpoint = slowed.hardware.get_component("hbf0")
    assert endpoint.metadata["read_latency_ns"] == after.read_latency_ns
    assert endpoint.read_bandwidth_gbps == base.hardware.get_component("hbf0").read_bandwidth_gbps * 0.5
