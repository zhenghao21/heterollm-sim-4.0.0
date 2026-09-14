import json
import re
import unittest
from urllib.parse import urlparse

from heterollm_sim.architecture_presets import (
    ANALYTICAL_APPROXIMATION,
    CATALOG_VERSION,
    EXACT_PUBLIC_TOPOLOGY,
    EXPERIMENTAL_REFERENCE,
    REPLACEMENT_POLICY,
    SUPPORT_LEVELS,
    architecture_preset_detail,
    architecture_preset_page,
    get_architecture_preset,
    list_architecture_presets,
    materialize_architecture_payload,
)
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.protocol_presets import protocol_preset_detail
from heterollm_sim.topology import validate_topology


REQUIRED = {
    "nvidia-h100-sxm-8-nvswitch",
    "nvidia-h200-sxm-8-nvswitch",
    "nvidia-gh200-superchip",
    "nvidia-gh200-superchip-144gb-hbm3e",
    "nvidia-gh200-nvl2",
    "nvidia-gh200-nvl2-96gb-hbm3",
    "nvidia-gb200-nvl4",
    "nvidia-gb200-nvl72",
    "amd-mi300a-apu",
    "amd-mi300x",
    "amd-mi300x-8-infinity-fabric",
    "intel-gaudi3-8-roce",
    "cxl-type3-memory-expander",
    "cxl-memory-pool",
    "ucie-chiplet-package",
    "gpu-hbm-cim",
    "hbm-pim",
    "gpu-nvme-gds",
    "gpu-hbf",
}

PRIMARY_DOMAINS = {
    "www.nvidia.com",
    "docs.nvidia.com",
    "www.amd.com",
    "www.intel.com",
    "computeexpresslink.org",
    "www.uciexpress.org",
    "www.jedec.org",
    "semiconductor.samsung.com",
    "download.semiconductor.samsung.com",
    "www.opencompute.org",
}


class ArchitecturePresetCatalogTests(unittest.TestCase):
    def test_structurally_executable_presets_have_both_cpu_and_gpu(self):
        expected = {
            "amd-mi300a-apu",
            "gpu-nvme-gds",
            "nvidia-gb200-nvl4",
            "nvidia-gb200-nvl72",
            "nvidia-gh200-nvl2",
            "nvidia-gh200-nvl2-96gb-hbm3",
            "nvidia-gh200-superchip",
            "nvidia-gh200-superchip-144gb-hbm3e",
        }
        actual = {
            item["id"]
            for item in list_architecture_presets()
            if item["compatibility"]["planner_executable"]
        }
        self.assertEqual(actual, expected)

    def test_catalog_is_separate_compact_hardware_only_and_complete(self):
        items = list_architecture_presets()
        ids = [item["id"] for item in items]

        self.assertEqual(CATALOG_VERSION, "1.3.0")
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), REQUIRED)
        self.assertEqual({item["support_level"] for item in items}, SUPPORT_LEVELS)
        for item in items:
            self.assertTrue(item["hardware_only"])
            self.assertTrue(item["loadable"])
            self.assertNotIn("hardware", item)
            self.assertNotIn("components", item)
            self.assertNotIn("model", item)
            self.assertNotIn("workload", item)
            self.assertGreater(item["component_count"], 1)
            self.assertGreater(item["link_count"], 0)
            self.assertGreater(item["group_count"], 0)
            self.assertTrue(item["capacity_display_value"] or item["capacity_bytes"] == 0)
            self.assertTrue(item["sources"])
            self.assertTrue(item["limitations"])
            compatibility = item["compatibility"]
            self.assertEqual(item["catalog_version"], CATALOG_VERSION)
            self.assertEqual(compatibility["schema_version"], "1.1")
            self.assertTrue(compatibility["recommended_model_classes"])
            self.assertTrue(compatibility["recommended_parallelism"])
            self.assertIn("supported_storage_roles", compatibility)
            self.assertTrue(compatibility["feature_tags"])
            self.assertIn("placement_must_be_regenerated", compatibility["constraints"])
            self.assertTrue(all(source["primary_source"] for source in item["sources"]))
            for source in item["sources"]:
                self.assertIn(urlparse(source["url"]).hostname, PRIMARY_DOMAINS)

    def test_evidence_parameters_and_planner_compatibility_are_separate_layers(self):
        gpu_attachment_required = {
            "cxl-type3-memory-expander",
            "cxl-memory-pool",
            "ucie-chiplet-package",
        }
        for item in list_architecture_presets():
            with self.subTest(preset_id=item["id"]):
                evidence = item["topology_evidence"]
                parameters = item["parameter_basis"]
                compatibility = item["compatibility"]

                self.assertEqual(evidence["level"], item["support_level"])
                self.assertFalse(evidence["physical_wiring_claimed"])
                self.assertIn("source_scope", evidence)
                self.assertEqual(
                    parameters["policy"], "per_field_metadata_is_authoritative"
                )
                self.assertEqual(parameters["bandwidth_storage_unit"], "Gb/s_decimal_one_way")
                self.assertEqual(
                    compatibility["requires_gpu_attachment"],
                    item["id"] in gpu_attachment_required,
                )
                self.assertEqual(
                    compatibility["requires_cpu_attachment"],
                    "cpu" not in item["component_kinds"],
                )
                self.assertEqual(
                    compatibility["planner_executable"],
                    "gpu" in item["component_kinds"]
                    and "cpu" in item["component_kinds"],
                )
                self.assertTrue(compatibility["requires_profile_review"])

                detail = architecture_preset_detail(item["id"])
                self.assertEqual(
                    compatibility["planner_gpu_count"],
                    sum(
                        component["kind"] == "gpu"
                        for component in detail["components"]
                    ),
                )
                if compatibility["planner_gpu_count"]:
                    self.assertIn(
                        "world_size_must_not_exceed_{}_planner_gpu_endpoints".format(
                            compatibility["planner_gpu_count"]
                        ),
                        compatibility["constraints"],
                    )

                hardware_metadata = detail["hardware"]["metadata"]
                self.assertEqual(hardware_metadata["topology_evidence"], evidence)
                self.assertEqual(hardware_metadata["parameter_basis"], parameters)
                self.assertEqual(
                    hardware_metadata["architecture_preset"]["planner_executable"],
                    compatibility["planner_executable"],
                )
                for field in (
                    "requires_gpu_attachment",
                    "requires_cpu_attachment",
                    "requires_profile_review",
                    "planner_gpu_count",
                    "planner_cpu_count",
                ):
                    self.assertEqual(
                        hardware_metadata["architecture_preset"][field],
                        compatibility[field],
                    )

                for component in detail["components"]:
                    if component["peak_ops_per_s"] <= 0:
                        continue
                    basis = component["metadata"]["peak_ops_basis"]
                    self.assertTrue(basis["precision"])
                    self.assertTrue(basis["sparsity"])
                    self.assertTrue(basis["source_basis"])
                    self.assertIn("not a cross-precision maximum", basis["ir_contract"])

    def test_every_hbm_node_declares_physical_to_simulator_composition(self):
        for preset_id in REQUIRED:
            detail = architecture_preset_detail(preset_id)
            for component in detail["components"]:
                if component["kind"] != "hbm":
                    continue
                with self.subTest(
                    preset_id=preset_id, component_id=component["component_id"]
                ):
                    composition = component["metadata"]["physical_composition"]
                    self.assertEqual(
                        composition["simulator_representation"],
                        "single_physical_unit_node",
                    )
                    self.assertEqual(composition["simulator_node_count"], 1)
                    self.assertEqual(composition["physical_unit_kind"], "HBM_stack")
                    self.assertEqual(composition["physical_unit_count"], 1)
                    self.assertEqual(
                        composition["physical_unit_count_status"],
                        "explicit_physical_node",
                    )
                    self.assertEqual(
                        composition["unit_capacity_bytes"],
                        component["capacity_bytes"],
                    )
                    self.assertEqual(
                        composition["unit_bandwidth_gbps"],
                        component["read_bandwidth_gbps"],
                    )
                    self.assertTrue(composition["unit_count_formula"])
                    self.assertTrue(composition["unit_count_status"])
                    self.assertTrue(composition["source_basis"])
                    self.assertTrue(composition["controller_component_id"])
                    self.assertEqual(
                        composition["memory_subsystem_id"],
                        composition["controller_component_id"],
                    )
                    self.assertIn("parameter_basis", component["metadata"])
                    self.assertIn("provenance", component["metadata"])

    def test_product_hbm_is_physically_expanded_and_totals_are_conserved(self):
        cases = {
            "nvidia-h100-sxm-8-nvswitch": (40, 640_000_000_000, 214_400.0),
            "nvidia-h200-sxm-8-nvswitch": (48, 1_128_000_000_000, 307_200.0),
            "nvidia-gh200-superchip": (6, 96_000_000_000, 32_000.0),
            "nvidia-gh200-superchip-144gb-hbm3e": (6, 144_000_000_000, 40_000.0),
            "nvidia-gh200-nvl2": (12, 288_000_000_000, 80_000.0),
            "nvidia-gh200-nvl2-96gb-hbm3": (12, 192_000_000_000, 64_000.0),
            "nvidia-gb200-nvl4": (32, 744_000_000_000, 256_000.0),
            "nvidia-gb200-nvl72": (576, 13_392_000_000_000, 4_608_000.0),
            "amd-mi300a-apu": (8, 128_000_000_000, 42_400.0),
            "amd-mi300x": (8, 192_000_000_000, 42_600.0),
            "amd-mi300x-8-infinity-fabric": (64, 1_536_000_000_000, 340_800.0),
            "intel-gaudi3-8-roce": (64, 1_024_000_000_000, 236_800.0),
        }
        for preset_id, (node_count, capacity, bandwidth) in cases.items():
            detail = architecture_preset_detail(preset_id)
            hbm_components = [
                component for component in detail["components"] if component["kind"] == "hbm"
            ]
            hbm_links = [link for link in detail["links"] if link["protocol"] == "HBM"]
            self.assertEqual(len(hbm_components), node_count)
            self.assertEqual(len(hbm_links), node_count)
            self.assertEqual(sum(component["capacity_bytes"] for component in hbm_components), capacity)
            self.assertAlmostEqual(
                sum(component["read_bandwidth_gbps"] for component in hbm_components),
                bandwidth,
            )
            self.assertAlmostEqual(
                sum(link["bandwidth_gbps"] for link in hbm_links),
                bandwidth,
            )

    def test_hbm_group_membership_roots_and_collapse_policy(self):
        cases = {
            "nvidia-h100-sxm-8-nvswitch": ("module0", "accelerator0", 6, True),
            "nvidia-h200-sxm-8-nvswitch": ("module0", "accelerator0", 7, True),
            "amd-mi300x-8-infinity-fabric": ("module0", "accelerator0", 9, True),
            "intel-gaudi3-8-roce": ("module0", "accelerator0", 9, True),
            "nvidia-gh200-superchip": ("superchip0", "hopper0", 9, False),
            "nvidia-gh200-nvl2": ("superchip0", "hopper0", 9, False),
            "nvidia-gb200-nvl4": ("gpu0_hbm", "gpu0", 9, True),
            "nvidia-gb200-nvl72": ("gpu0_hbm", "gpu0", 9, True),
            "amd-mi300a-apu": ("package0", "cdna_gpu0", 10, False),
            "amd-mi300x": ("package0", "mi300x0", 9, False),
        }
        for preset_id, (group_id, root, member_count, collapsed) in cases.items():
            groups = architecture_preset_detail(preset_id)["groups"]
            group = next(item for item in groups if item["group_id"] == group_id)
            self.assertEqual(group["root"], root)
            self.assertEqual(len(group["members"]), member_count)
            self.assertEqual(group["collapsed"], collapsed)
            self.assertIn(root, group["members"])

    def test_gb200_group_boundaries_follow_memory_subsystems(self):
        cases = {
            "nvidia-gb200-nvl4": (4, 2, False),
            "nvidia-gb200-nvl72": (72, 36, True),
        }
        for preset_id, (gpu_count, grace_count, grace_collapsed) in cases.items():
            detail = architecture_preset_detail(preset_id)
            components = {
                component["component_id"]: component
                for component in detail["components"]
            }
            groups = {group["group_id"]: group for group in detail["groups"]}
            member_to_group = {}
            for group in detail["groups"]:
                for member in group["members"]:
                    self.assertNotIn(member, member_to_group)
                    member_to_group[member] = group["group_id"]

            self.assertEqual(len(groups), gpu_count + grace_count + 1)
            self.assertEqual(groups["fabric"]["members"], ["fabric0"])
            self.assertEqual(groups["fabric"]["root"], "fabric0")
            self.assertTrue(groups["fabric"]["collapsed"])

            for grace_index in range(grace_count):
                grace = "grace{}".format(grace_index)
                lpddr = "lpddr{}".format(grace_index)
                group = groups["grace{}_lpddr".format(grace_index)]
                self.assertEqual(group["members"], [grace, lpddr])
                self.assertEqual(group["root"], grace)
                self.assertEqual(group["collapsed"], grace_collapsed)

            for gpu_index in range(gpu_count):
                gpu = "gpu{}".format(gpu_index)
                group_id = "gpu{}_hbm".format(gpu_index)
                group = groups[group_id]
                self.assertEqual(group["root"], gpu)
                self.assertTrue(group["collapsed"])
                self.assertEqual(len(group["members"]), 9)
                hbm_members = [
                    member
                    for member in group["members"]
                    if components[member]["kind"] == "hbm"
                ]
                self.assertEqual(len(hbm_members), 8)
                self.assertEqual(set(group["members"]), {gpu, *hbm_members})
                for hbm_id in hbm_members:
                    composition = components[hbm_id]["metadata"][
                        "physical_composition"
                    ]
                    self.assertEqual(composition["memory_subsystem_id"], gpu)
                    self.assertEqual(member_to_group[hbm_id], group_id)

    def test_visible_and_raw_capacity_provenance_are_separate(self):
        cases = {
            "nvidia-h200-sxm-8-nvswitch": (23_500_000_000, 24_000_000_000, "derived_from_product_total_and_24GB_stack_class"),
            "nvidia-gb200-nvl4": (23_250_000_000, 24_000_000_000, "vendor_documented_stack_sites"),
        }
        for preset_id, (visible, raw, count_status) in cases.items():
            component = next(
                item
                for item in architecture_preset_detail(preset_id)["components"]
                if item["kind"] == "hbm"
            )
            composition = component["metadata"]["physical_composition"]
            self.assertEqual(component["capacity_bytes"], visible)
            self.assertEqual(composition["unit_raw_capacity_bytes"], raw)
            self.assertEqual(composition["unit_count_status"], count_status)
            self.assertEqual(
                composition["capacity_accounting"],
                "product_visible_and_raw_capacity_recorded_separately",
            )

    def test_hbm_nodes_reference_matching_component_presets(self):
        for preset_id in REQUIRED:
            detail = architecture_preset_detail(preset_id)
            for component in detail["components"]:
                if component["kind"] != "hbm":
                    continue
                composition = component["metadata"]["physical_composition"]
                component_preset_id = composition["component_preset_id"]
                self.assertTrue(component_preset_id)
                preset_component = get_component_preset(component_preset_id).component
                self.assertEqual(preset_component.kind, "hbm")
                self.assertEqual(preset_component.capacity_bytes, component["capacity_bytes"])
                self.assertAlmostEqual(
                    preset_component.read_bandwidth_gbps,
                    component["read_bandwidth_gbps"],
                )

    def test_grace_lpddr_aggregate_count_and_system_scopes(self):
        cases = {
            "nvidia-gh200-superchip": (1, 480_000_000_000, 4_000.0),
            "nvidia-gh200-superchip-144gb-hbm3e": (1, 480_000_000_000, 4_000.0),
            "nvidia-gh200-nvl2": (2, 960_000_000_000, 8_000.0),
            "nvidia-gh200-nvl2-96gb-hbm3": (2, 960_000_000_000, 8_000.0),
            "nvidia-gb200-nvl4": (2, 960_000_000_000, 8_192.0),
            "nvidia-gb200-nvl72": (36, 17_280_000_000_000, 112_000.0),
        }
        for preset_id, (count, capacity, bandwidth) in cases.items():
            detail = architecture_preset_detail(preset_id)
            memories = [
                component
                for component in detail["components"]
                if component["kind"] == "host_memory"
            ]
            self.assertEqual(len(memories), count)
            self.assertEqual(sum(item["capacity_bytes"] for item in memories), capacity)
            self.assertAlmostEqual(
                sum(item["read_bandwidth_gbps"] for item in memories),
                bandwidth,
            )
            for memory in memories:
                composition = memory["metadata"]["physical_composition"]
                self.assertEqual(composition["simulator_representation"], "aggregate_node")
                self.assertIsNone(composition["physical_unit_count"])
                self.assertEqual(
                    composition["physical_unit_count_status"],
                    "not_reliably_disclosed",
                )
                expected_component_preset_id = (
                    "lpddr5x-gb200-480gb-512gbs-aggregate"
                    if preset_id.startswith("nvidia-gb200")
                    else "lpddr5x-gh200-480gb-500gbs-aggregate"
                )
                self.assertEqual(
                    composition["component_preset_id"],
                    expected_component_preset_id,
                )
                expected_component_preset_status = (
                    "catalog_reference_with_system_bandwidth_override"
                    if preset_id == "nvidia-gb200-nvl72"
                    else "catalog_reference"
                )
                self.assertEqual(
                    composition["component_preset_status"],
                    expected_component_preset_status,
                )
        nvl72 = architecture_preset_detail("nvidia-gb200-nvl72")
        lpddr = next(
            component for component in nvl72["components"] if component["kind"] == "host_memory"
        )
        self.assertEqual(
            lpddr["metadata"]["bandwidth_scope"],
            "per_Grace_analysis_share_of_NVL72_rack_14TBps_envelope",
        )
        self.assertIn(
            "rack 14 TB/s",
            lpddr["metadata"]["parameter_basis"]["bandwidth_gbps"],
        )

    def test_gaudi3_roce_endpoint_explains_selected_ports_and_rate(self):
        detail = architecture_preset_detail("intel-gaudi3-8-roce")
        accelerators = [
            component for component in detail["components"] if component["kind"] == "gpu"
        ]
        roce_links = [link for link in detail["links"] if link["protocol"] == "RoCE"]

        self.assertEqual(len(accelerators), 8)
        self.assertEqual(len(roce_links), 8)
        for accelerator in accelerators:
            port = next(port for port in accelerator["ports"] if port["protocol"] == "RoCE")
            self.assertEqual(port["lanes"], 8)
            self.assertEqual(port["bandwidth_gbps"], 1_600.0)
            self.assertEqual(port["metadata"]["aggregate_member_count"], 8)
            self.assertEqual(port["metadata"]["aggregate_member_rate_gbps"], 200.0)
            self.assertEqual(port["metadata"]["device_total_physical_port_count"], 24)
            self.assertEqual(
                port["metadata"]["modeled_port_subset_status"],
                "analytical_endpoint_envelope",
            )
        for link in roce_links:
            self.assertEqual(link["metadata"]["aggregation_formula"], "member_count_x_member_rate")
            self.assertFalse(link["metadata"]["protocol_payload_overhead_included"])

    def test_pcie_cxl_and_ucie_publish_distinct_bandwidth_accounting(self):
        gds = architecture_preset_detail("gpu-nvme-gds")
        pcie_ports = [
            port
            for component in gds["components"]
            for port in component["ports"]
            if port["protocol"] == "PCIe"
        ]
        self.assertTrue(pcie_ports)
        for port in pcie_ports:
            self.assertEqual(port["metadata"]["bandwidth_basis"], "decoded_phy_line_rate")
            self.assertEqual(port["metadata"]["encoding"], "128b/130b")
            self.assertTrue(port["metadata"]["encoding_overhead_included"])
            self.assertFalse(port["metadata"]["transaction_overhead_included"])
        ssd_link = next(link for link in gds["links"] if link["link_id"] == "ssd_pcie")
        self.assertAlmostEqual(ssd_link["bandwidth_gbps"], 126.03076923076924)
        self.assertEqual(ssd_link["metadata"]["bandwidth_basis"], "decoded_phy_line_rate")
        self.assertAlmostEqual(ssd_link["metadata"]["decoded_phy_capacity_gbps"], 126.03076923076924)
        self.assertEqual(ssd_link["metadata"]["device_read_envelope_gbps"], 112.0)
        self.assertTrue(ssd_link["metadata"]["media_bandwidth_modeled_separately"])

        ports_by_component = {
            component["component_id"]: {
                port["port_id"]: port for port in component["ports"]
            }
            for component in gds["components"]
        }
        for link in gds["links"]:
            source_port = ports_by_component[link["source_component"]][link["source_port"]]
            target_port = ports_by_component[link["target_component"]][link["target_port"]]
            self.assertLessEqual(
                link["bandwidth_gbps"],
                min(source_port["bandwidth_gbps"], target_port["bandwidth_gbps"]),
            )

        for preset_id in ("cxl-type3-memory-expander", "cxl-memory-pool"):
            detail = architecture_preset_detail(preset_id)
            for link in detail["links"]:
                metadata = link["metadata"]
                self.assertEqual(metadata["bandwidth_basis"], "raw_phy_envelope")
                self.assertEqual(metadata["line_rate_gtps_per_lane"], 64.0)
                self.assertFalse(metadata["flit_fec_overhead_included"])

        for preset_id in ("ucie-chiplet-package", "gpu-hbm-cim", "hbm-pim", "gpu-hbf"):
            detail = architecture_preset_detail(preset_id)
            for link in (link for link in detail["links"] if link["protocol"] == "UCIe"):
                metadata = link["metadata"]
                self.assertEqual(
                    metadata["bandwidth_basis"],
                    "analytical_raw_phy_operating_point",
                )
                self.assertEqual(metadata["line_rate_gtps_per_lane"], 32.0)
                self.assertFalse(metadata["protocol_payload_overhead_included"])

    def test_every_architecture_capability_field_publishes_a_unit(self):
        for preset_id in REQUIRED:
            with self.subTest(preset_id=preset_id):
                detail = architecture_preset_detail(preset_id)
                for component in detail["components"]:
                    units = component["metadata"]["capability_units"]
                    self.assertEqual(units["capacity_bytes"], "B")
                    self.assertEqual(units["peak_ops_per_s"], "op/s")
                    self.assertEqual(
                        units["read_bandwidth_gbps"],
                        "Gb/s_decimal_one_way",
                    )
                    for port in component["ports"]:
                        port_units = port["metadata"]["capability_units"]
                        self.assertEqual(port_units["lanes"], "lane_count")
                        self.assertEqual(
                            port_units["bandwidth_gbps"],
                            "Gb/s_decimal_one_way",
                        )
                    if component["kind"] == "cpu" and component["peak_ops_per_s"] == 0:
                        self.assertIn(
                            "peak_ops_per_s",
                            component["metadata"]["unknown_value_sentinels"],
                        )
                        self.assertIn(
                            "profiles.components.cpu",
                            component["metadata"]["cpu_profile_contract"],
                        )
                        self.assertIn(
                            "profiles.components.host_memory",
                            component["metadata"]["cpu_profile_contract"],
                        )
                for link in detail["links"]:
                    link_units = link["metadata"]["capability_units"]
                    self.assertEqual(link_units["lanes"], "lane_count")
                    self.assertEqual(link_units["latency_ns"], "ns")
                    self.assertEqual(
                        link_units["bandwidth_gbps"],
                        "Gb/s_decimal_one_way",
                    )

    def test_gb200_uses_vendor_dense_bf16_and_physical_stack_basis(self):
        detail = architecture_preset_detail("nvidia-gb200-nvl72")
        gpus = [component for component in detail["components"] if component["kind"] == "gpu"]
        hbm = [component for component in detail["components"] if component["kind"] == "hbm"]
        self.assertEqual(len(gpus), 72)
        self.assertEqual(len(hbm), 576)
        self.assertTrue(all(component["peak_ops_per_s"] == 2_500_000_000_000_000.0 for component in gpus))
        self.assertTrue(all(component["capacity_bytes"] == 23_250_000_000 for component in hbm))
        self.assertTrue(all(component["read_bandwidth_gbps"] == 8_000.0 for component in hbm))
        self.assertEqual(sum(component["capacity_bytes"] for component in hbm), 13_392_000_000_000)
        self.assertEqual(sum(component["read_bandwidth_gbps"] for component in hbm), 4_608_000.0)
        self.assertTrue(
            all(
                component["metadata"]["physical_composition"]["unit_count_in_product"]
                == 8
                for component in hbm
            )
        )
        self.assertEqual(
            {
                component["metadata"]["physical_composition"]["memory_subsystem_id"]
                for component in hbm
            },
            {"gpu{}".format(index) for index in range(72)},
        )
        self.assertTrue(
            all(
                component["metadata"]["peak_ops_basis"]["sparsity"]
                == "dense_no_structured_sparsity_credit"
                for component in gpus
            )
        )

    def test_every_payload_round_trips_and_topology_validates(self):
        for preset_id in REQUIRED:
            with self.subTest(preset_id=preset_id):
                payload = materialize_architecture_payload(preset_id)
                json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
                hardware = hardware_from_dict(payload)
                report = validate_topology(hardware)

                self.assertTrue(report.is_valid, report.format())
                self.assertTrue(hardware.require_connected)
                self.assertEqual(
                    hardware.metadata["architecture_preset"]["id"], preset_id
                )
                self.assertTrue(
                    hardware.metadata["architecture_preset"]["hardware_only"]
                )
                components = hardware.component_map()
                for link in hardware.links:
                    source_port = components[link.source_component].port_map()[link.source_port]
                    target_port = components[link.target_component].port_map()[link.target_port]
                    if source_port.bandwidth_gbps:
                        self.assertLessEqual(link.bandwidth_gbps, source_port.bandwidth_gbps)
                    if target_port.bandwidth_gbps:
                        self.assertLessEqual(link.bandwidth_gbps, target_port.bandwidth_gbps)
                storage = [
                    component for component in components.values()
                    if component.kind in {"ssd", "high_io_ssd", "hbf"}
                ]
                for component in storage:
                    self.assertGreater(component.metadata["dma_bandwidth_gbps"], 0)
                    self.assertGreater(component.metadata["dma_latency_ns"], 0)
                    self.assertGreaterEqual(component.metadata["max_outstanding_requests"], 1)
                    self.assertIn(
                        "analytical",
                        component.metadata["storage_transport_parameter_basis"],
                    )

    def test_topology_views_have_complete_layout_and_flat_disjoint_groups(self):
        for preset_id in REQUIRED:
            with self.subTest(preset_id=preset_id):
                detail = architecture_preset_detail(preset_id)
                component_ids = {
                    component["component_id"] for component in detail["components"]
                }
                view = detail["topology_view"]
                positions = view["layout"]["positions"]
                groups = view["groups"]

                self.assertEqual(set(positions), component_ids)
                self.assertEqual(detail["groups"], groups)
                self.assertEqual(
                    {member for group in groups for member in group["members"]},
                    component_ids,
                )
                members = [member for group in groups for member in group["members"]]
                self.assertEqual(len(members), len(set(members)))
                for group in groups:
                    self.assertIn(group["root"], group["members"])
                    self.assertFalse(
                        any(isinstance(member, (dict, list)) for member in group["members"])
                    )

    def test_nvl72_is_collapsed_linear_size_not_a_gpu_full_mesh(self):
        detail = architecture_preset_detail("nvidia-gb200-nvl72")
        gpus = [component for component in detail["components"] if component["kind"] == "gpu"]
        fabric = [
            component
            for component in detail["components"]
            if component["component_id"] == "fabric0"
        ]
        gpu_to_gpu = [
            link
            for link in detail["links"]
            if link["source_component"].startswith("gpu")
            and link["target_component"].startswith("gpu")
        ]

        self.assertEqual(len(gpus), 72)
        self.assertEqual(len(fabric), 1)
        self.assertEqual(len(gpu_to_gpu), 0)
        # 72 C2C + 72 GPU-Fabric + 72×8 physical HBM + 36 LPDDR links:
        # still O(GPU),
        # with no quadratic GPU-to-GPU mesh.
        self.assertEqual(len(detail["links"]), 756)
        self.assertEqual(
            detail["preset"]["support_level"], ANALYTICAL_APPROXIMATION
        )
        limitations = " ".join(detail["preset"]["limitations"])
        self.assertIn("不声称", limitations)
        self.assertIn("72 条 GPU-Fabric 链路", limitations)

    def test_new_interconnect_links_do_not_exceed_catalog_one_way_envelopes(self):
        protocol_limits = {
            ("NVLink-C2C", "GH200"): protocol_preset_detail("nvlink-c2c-gh200")["simulation_defaults"]["link"]["bandwidth_gbps"],
            ("InfinityFabric", "MI300X"): protocol_preset_detail("infinity-fabric-mi300x-envelope")["simulation_defaults"]["link"]["bandwidth_gbps"],
            ("RoCE", "v2"): protocol_preset_detail("roce-v2-gaudi3-8x200gbe-envelope")["simulation_defaults"]["link"]["bandwidth_gbps"],
            ("LPDDR5X", "GH200"): protocol_preset_detail("lpddr5x-gh200-aggregate")["simulation_defaults"]["link"]["bandwidth_gbps"],
        }
        for preset_id in REQUIRED:
            detail = architecture_preset_detail(preset_id)
            for link in detail["links"]:
                limit = protocol_limits.get((link["protocol"], link["version"]))
                if limit is not None:
                    self.assertLessEqual(
                        link["bandwidth_gbps"],
                        limit,
                        "{}:{} exceeds protocol envelope".format(preset_id, link["link_id"]),
                    )

    def test_detail_declares_atomic_replace_and_mapping_invalidation(self):
        detail = architecture_preset_detail("nvidia-gh200-superchip")

        self.assertEqual(detail["replacement_policy"], REPLACEMENT_POLICY)
        self.assertEqual(detail["compatibility"], detail["preset"]["compatibility"])
        self.assertEqual(detail["replacement_policy"]["mode"], "replace_hardware")
        self.assertEqual(detail["replacement_policy"]["preserve"], ["model", "workload"])
        self.assertEqual(
            detail["replacement_policy"]["invalidate"],
            [
                "placement",
                "rank_mapping",
                "hardware_cost_profiles",
                "host_orchestration",
            ],
        )
        self.assertTrue(detail["replacement_policy"]["confirmation_required"])
        self.assertEqual(detail["replacement_policy"]["undo_scope"], "single_operation")
        self.assertNotIn("placement", detail["hardware"])
        self.assertNotIn("workload", detail["hardware"])

    def test_filters_cover_vendor_protocol_and_support_level(self):
        page = architecture_preset_page(
            vendor="NVIDIA",
            protocol="NVLink",
            support_level=ANALYTICAL_APPROXIMATION,
        )

        self.assertEqual(page["total"], len(page["items"]))
        self.assertGreaterEqual(page["total"], 4)
        self.assertTrue(all(item["vendor"] == "NVIDIA" for item in page["items"]))
        self.assertTrue(all("NVLink" in item["protocols"] for item in page["items"]))
        self.assertIn(EXACT_PUBLIC_TOPOLOGY, page["filters"]["support_level"])
        self.assertIn(EXPERIMENTAL_REFERENCE, page["filters"]["support_level"])
        self.assertIn("CXL", page["filters"]["protocol"])
        self.assertEqual(
            architecture_preset_page(query="memory pool")["items"][0]["id"],
            "cxl-memory-pool",
        )

    def test_user_visible_catalog_copy_uses_display_units_not_ir_bit_units(self):
        forbidden = re.compile(r"Gb/s|Gbps|\bbit\b", re.IGNORECASE)
        capacity_units = re.compile(r"(?:KiB|MiB|GiB|TiB)")
        for item in list_architecture_presets():
            with self.subTest(preset_id=item["id"]):
                visible = [
                    item["name"],
                    item["notes"],
                    item["capacity_display_value"],
                    *item["limitations"],
                    *(source["title"] for source in item["sources"]),
                ]
                self.assertFalse(any(forbidden.search(text) for text in visible if text))
                if item["capacity_display_value"]:
                    self.assertRegex(item["capacity_display_value"], capacity_units)
                    self.assertRegex(item["capacity_display_value"], "口径")
                    self.assertIn("IEC", item["capacity_source_basis"])

                detail = architecture_preset_detail(item["id"])
                for component in detail["components"]:
                    display = component["metadata"].get("capacity_display_value", "")
                    if display:
                        self.assertRegex(display, capacity_units)
                        self.assertRegex(display, "口径")
                        self.assertNotRegex(display, forbidden)
                        self.assertIn("IEC", component["metadata"]["capacity_source_basis"])

    def test_unknown_preset_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_architecture_preset("does-not-exist")


if __name__ == "__main__":
    unittest.main()
