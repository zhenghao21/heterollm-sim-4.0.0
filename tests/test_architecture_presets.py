"""Contract tests for the curated, loadable architecture presets."""

import unittest

from heterollm_sim.architecture_presets import (
    architecture_preset_detail,
    architecture_preset_page,
    get_architecture_preset,
    list_architecture_presets,
    materialize_architecture_payload,
)
from heterollm_sim.topology import validate_topology
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.component_presets import materialize_component_payload


PUBLIC_ID = "nvidia-b200-1gpu-2hbf-2hbm"
DEEPSEEK_CAPACITY_ID = "nvidia-b200-1gpu-3hbf-2hbm"
NATIVE_ID = "local-native-rtx5080-9950x3d-gddr7-ddr5"
REMOVED_IDS = {
    "nvidia-h100-sxm-8-nvswitch",
    "nvidia-h200-sxm-8-nvswitch",
    "nvidia-gh200-superchip",
    "gpu-hbm-cim",
    "soc-2x-dram-sram-cim",
}


class CuratedArchitectureCatalogTests(unittest.TestCase):
    def test_curated_presets_are_public(self):
        rows = list_architecture_presets()
        self.assertEqual({row["id"] for row in rows}, {PUBLIC_ID, DEEPSEEK_CAPACITY_ID, NATIVE_ID})
        self.assertEqual(architecture_preset_page()["total"], 3)
        filtered = architecture_preset_page(vendor="NVIDIA", protocol="UCIe", loadable=True)
        self.assertEqual([row["id"] for row in filtered["items"]], [PUBLIC_ID, DEEPSEEK_CAPACITY_ID])

    def test_native_rtx5080_payload_uses_curated_memory_and_protocol_limits(self):
        detail = architecture_preset_detail(NATIVE_ID)
        hardware = hardware_from_dict(detail["hardware"])
        report = validate_topology(hardware)
        self.assertTrue(report.is_valid, report.format_en())

        components = {item.component_id: item for item in hardware.components}
        self.assertEqual(set(components), {"gpu0", "cpu0", "hostmem0", "hbm0"})
        self.assertEqual(components["gpu0"].metadata["component_preset_id"], "nvidia-rtx-5080")
        self.assertEqual(components["cpu0"].metadata["component_preset_id"], "amd-ryzen-9-9950x3d")
        self.assertEqual(components["hostmem0"].metadata["component_preset_id"], "acer-local-ddr5-128gb-5600-dual-channel")
        self.assertEqual(components["hbm0"].metadata["component_preset_id"], "gddr7-16gb-30_0-256bit")
        self.assertEqual(components["hbm0"].metadata["component_preset_status"], "logical_gpu_attached_memory")
        self.assertEqual(components["hbm0"].metadata["attached_memory_preset_id"], "gddr7-16gb-30_0-256bit")
        self.assertEqual(components["hbm0"].metadata["memory_type"], "GDDR7")
        self.assertEqual(components["hbm0"].bandwidth_gbps, 7680.0)
        self.assertEqual(components["hostmem0"].bandwidth_gbps, 716.8)
        self.assertNotIn("hbm1", components)

        links = {item.link_id: item for item in hardware.links}
        self.assertEqual(links["gpu-gddr7"].protocol, "GDDR7")
        self.assertEqual(links["gpu-gddr7"].bandwidth_gbps, 7680.0)
        self.assertEqual(links["cpu-hostmem-ddr"].protocol, "DDR5")
        self.assertEqual(links["cpu-hostmem-ddr"].bandwidth_gbps, 716.8)
        self.assertEqual(links["cpu-gpu-pcie"].protocol, "PCIe")
        self.assertEqual(links["cpu-gpu-pcie"].bandwidth_gbps, 252.032)
        self.assertTrue(detail["hardware"]["metadata"]["native_input"]["hardware_only_projection"])

    def test_removed_architectures_are_not_addressable(self):
        for preset_id in REMOVED_IDS:
            with self.subTest(preset_id=preset_id):
                with self.assertRaises(KeyError):
                    get_architecture_preset(preset_id)
                with self.assertRaises(KeyError):
                    materialize_architecture_payload(preset_id)
                with self.assertRaises(KeyError):
                    architecture_preset_detail(preset_id)

    def test_b200_payload_is_loadable_and_has_units(self):
        detail = architecture_preset_detail(PUBLIC_ID)
        self.assertEqual(detail["preset"]["id"], PUBLIC_ID)
        self.assertTrue(detail["hardware"]["components"])
        self.assertTrue(detail["hardware"]["links"])
        report = validate_topology(hardware_from_dict(detail["hardware"]))
        self.assertTrue(report.is_valid, report.format_en())
        self.assertIn("capacity_bytes", detail["preset"]["capability_units"])
        self.assertIn("read_bandwidth_gbps", detail["preset"]["capability_units"])

    def test_b200_is_composed_from_curated_component_presets(self):
        detail = architecture_preset_detail(PUBLIC_ID)
        components = {item["component_id"]: item for item in detail["components"]}
        expected = {
            "gpu0": "nvidia-b200-sxm-gpu",
            "hbf0": "sk-hynix-hbf-512gb",
            "hbf1": "sk-hynix-hbf-512gb",
            "hbm0": "samsung-hbm3e-36gb-9_2",
            "hbm1": "samsung-hbm3e-36gb-9_2",
            "hostmem0": "samsung-ddr5-32gb-udimm-5600",
        }
        self.assertEqual(detail["hardware"]["metadata"]["component_preset_ids"], expected)
        for component_id, preset_id in expected.items():
            payload = materialize_component_payload(preset_id)
            component = components[component_id]
            self.assertEqual(component["capacity_bytes"], payload["capacity_bytes"])
            self.assertEqual(component["read_bandwidth_gbps"], payload["read_bandwidth_gbps"])
            self.assertEqual(component["write_bandwidth_gbps"], payload["write_bandwidth_gbps"])


if __name__ == "__main__":
    unittest.main()
