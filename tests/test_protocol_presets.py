import re
import unittest
from urllib.parse import urlparse

from heterollm_sim.config import hardware_from_dict
from heterollm_sim.protocol_presets import (
    get_protocol_preset,
    list_protocol_presets,
    protocol_preset_detail,
    protocol_preset_page,
)
from heterollm_sim.topology import validate_topology


REQUIRED = {
    "hbm3-6_4-1024",
    "hbm2e-3_2-1024",
    "hbm3e-h200-stack-slice",
    "hbm4-8_0-2048",
    "pcie-5_0-x16",
    "cxl-3_0-x16",
    "ucie-2_0-standard-x64",
    "nvlink-4-h100-18",
    "nvlink-c2c-gh200",
    "infinity-fabric-mi300x-envelope",
    "roce-v2-gaudi3-8x200gbe-envelope",
    "lpddr5x-gh200-aggregate",
}

NEW_HETEROGENEOUS_PROTOCOLS = {
    "hbm2e-3_2-1024",
    "nvlink-c2c-gh200",
    "infinity-fabric-mi300x-envelope",
    "roce-v2-gaudi3-8x200gbe-envelope",
    "lpddr5x-gh200-aggregate",
}

PRIMARY_DOMAINS = {
    "www.jedec.org",
    "www.nvidia.com",
    "pcisig.com",
    "computeexpresslink.org",
    "www.uciexpress.org",
    "www.amd.com",
    "www.intel.com",
}


class ProtocolPresetCatalogTests(unittest.TestCase):
    def test_catalog_is_offline_stable_and_uses_only_primary_sources(self):
        items = list_protocol_presets()
        ids = [item["id"] for item in items]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(REQUIRED.issubset(ids))
        for item in items:
            self.assertIn(item["protocol"], {"HBM", "PCIe", "CXL", "UCIe", "NVLink", "NVLink-C2C", "InfinityFabric", "RoCE", "LPDDR5X"})
            self.assertTrue(item["version"])
            self.assertGreater(item["transfer_unit_count"], 0)
            self.assertRegex(item["transfer_unit_semantics"], r"[\u3400-\u9fff]")
            self.assertRegex(item["displayed_bandwidth_scope"], r"[\u3400-\u9fff]")
            self.assertTrue(item["sources"])
            for source in item["sources"]:
                self.assertIn(urlparse(source["url"]).hostname, PRIMARY_DOMAINS)
                self.assertIn(source["publisher"], {"JEDEC", "NVIDIA", "AMD", "Intel", "PCI-SIG", "CXL Consortium", "UCIe Consortium"})

    def test_bandwidth_scopes_do_not_confuse_bits_bytes_or_directions(self):
        pcie = protocol_preset_detail("pcie-5_0-x16")["preset"]["bandwidth"]
        self.assertEqual(pcie["raw_gbps"], 512.0)
        self.assertAlmostEqual(pcie["raw_gbs"], 64.0)
        self.assertAlmostEqual(pcie["effective_one_way_gbps"], 512.0 * 128.0 / 130.0)
        self.assertAlmostEqual(pcie["aggregate_bidirectional_gbps"], 2.0 * pcie["effective_one_way_gbps"])

        nvlink = protocol_preset_detail("nvlink-4-h100-18")["preset"]["bandwidth"]
        self.assertIsNone(nvlink["raw_gbps"])
        self.assertEqual(nvlink["effective_one_way_gbs"], 450.0)
        self.assertEqual(nvlink["aggregate_bidirectional_gbs"], 900.0)

        hbm = protocol_preset_detail("hbm3-6_4-1024")["preset"]["bandwidth"]
        self.assertEqual(hbm["effective_one_way_gbs"], 819.2)
        self.assertIsNone(hbm["aggregate_bidirectional_gbps"])

        for preset_id in ("cxl-3_0-x16", "ucie-2_0-standard-x64"):
            bandwidth = protocol_preset_detail(preset_id)["preset"]["bandwidth"]
            self.assertIsNone(bandwidth["effective_one_way_gbps"])
            self.assertEqual(bandwidth["aggregate_bidirectional_gbps"], 2 * bandwidth["raw_gbps"])

    def test_every_catalog_bandwidth_preserves_one_way_and_aggregate_invariants(self):
        """Guard the complete catalog against bit/byte and direction regressions."""

        for preset_id in (item["id"] for item in list_protocol_presets()):
            with self.subTest(preset_id=preset_id):
                detail = protocol_preset_detail(preset_id)
                bandwidth = detail["preset"]["bandwidth"]
                defaults = detail["simulation_defaults"]
                one_way = bandwidth["effective_one_way_gbps"]
                raw = bandwidth["raw_gbps"]
                aggregate = bandwidth["aggregate_bidirectional_gbps"]

                self.assertGreater(defaults["link"]["bandwidth_gbps"], 0.0)
                self.assertEqual(
                    defaults["link"]["bandwidth_gbps"],
                    defaults["source_port"]["bandwidth_gbps"],
                )
                self.assertEqual(
                    defaults["link"]["bandwidth_gbps"],
                    defaults["target_port"]["bandwidth_gbps"],
                )
                if one_way is not None and aggregate is not None:
                    self.assertAlmostEqual(aggregate, 2.0 * one_way)
                elif raw is not None and aggregate is not None:
                    self.assertAlmostEqual(aggregate, 2.0 * raw)
                if one_way is not None:
                    self.assertLessEqual(defaults["link"]["bandwidth_gbps"], one_way)

    def test_simulation_defaults_create_valid_ports_and_links(self):
        for preset_id in REQUIRED:
            with self.subTest(preset_id=preset_id):
                detail = protocol_preset_detail(preset_id)
                defaults = detail["simulation_defaults"]
                source_port = {"port_id": "p", **defaults["source_port"]}
                target_port = {"port_id": "p", **defaults["target_port"]}
                protocol = detail["preset"]["protocol"]
                target_kind = "hbm" if protocol == "HBM" else "gpu"
                link = {
                    "link_id": "l0",
                    "source_component": "left",
                    "source_port": "p",
                    "target_component": "right",
                    "target_port": "p",
                    **defaults["link"],
                }
                hardware = hardware_from_dict(
                    {
                        "name": preset_id,
                        "components": [
                            {"component_id": "left", "kind": "gpu", "package_id": "pkg0", "die_id": "die0", "ports": [source_port]},
                            {"component_id": "right", "kind": target_kind, "package_id": "pkg0", "die_id": "die1", "ports": [target_port]},
                        ],
                        "links": [link],
                        "require_connected": True,
                    }
                )
                report = validate_topology(hardware)
                self.assertTrue(report.is_valid, report.format())
                self.assertEqual(defaults["link"]["metadata"]["bandwidth_semantics"], "one_way_capacity")
                self.assertTrue(defaults["manual_override"]["allowed"])
                self.assertIn("bandwidth_gbps", defaults["manual_override"]["fields"])

    def test_new_heterogeneous_protocols_use_one_way_defaults_and_visible_byte_units(self):
        forbidden = re.compile(r"Gb/s|Gbps|\bbit\b")
        for preset_id in NEW_HETEROGENEOUS_PROTOCOLS:
            with self.subTest(preset_id=preset_id):
                detail = protocol_preset_detail(preset_id)
                preset = detail["preset"]
                defaults = detail["simulation_defaults"]
                visible = [
                    preset["name"],
                    preset["transfer_unit_semantics"],
                    preset["displayed_bandwidth_scope"],
                    detail["derivation"],
                    *preset["limitations"],
                ]
                self.assertFalse(any(forbidden.search(text) for text in visible))
                self.assertIn("GB/s", " ".join(visible) + " " + preset["io_speed"]["unit"])
                self.assertEqual(
                    defaults["link"]["bandwidth_gbps"],
                    defaults["source_port"]["bandwidth_gbps"],
                )
                self.assertEqual(
                    defaults["link"]["bandwidth_gbps"],
                    defaults["target_port"]["bandwidth_gbps"],
                )
                self.assertEqual(
                    defaults["link"]["metadata"]["bandwidth_semantics"],
                    "one_way_capacity",
                )

        infinity = protocol_preset_detail("infinity-fabric-mi300x-envelope")
        self.assertEqual(infinity["preset"]["evidence_level"], "A_ANALYTICAL")
        self.assertIn("可覆盖", " ".join(infinity["preset"]["limitations"]))
        roce = protocol_preset_detail("roce-v2-gaudi3-8x200gbe-envelope")
        self.assertEqual(roce["preset"]["evidence_level"], "A_ANALYTICAL")

    def test_all_user_visible_protocol_copy_avoids_internal_bit_rate_units(self):
        forbidden = re.compile(r"Gb/s|Gbps|\bbit\b")
        for preset_id in REQUIRED:
            with self.subTest(preset_id=preset_id):
                detail = protocol_preset_detail(preset_id)
                preset = detail["preset"]
                visible = [
                    preset["name"],
                    preset["transfer_unit_semantics"],
                    preset["displayed_bandwidth_scope"],
                    detail["derivation"],
                    *preset["limitations"],
                ]
                self.assertFalse(any(forbidden.search(text) for text in visible))

    def test_filters_and_unknown_id(self):
        page = protocol_preset_page(protocol="HBM", organization="JEDEC")
        self.assertEqual(page["total"], 3)
        self.assertTrue(all(item["protocol"] == "HBM" for item in page["items"]))
        self.assertIn("NVLink", page["filters"]["protocol"])
        with self.assertRaises(KeyError):
            get_protocol_preset("missing")


if __name__ == "__main__":
    unittest.main()
