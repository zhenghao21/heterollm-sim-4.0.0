"""Contract tests for the curated hardware component catalog."""

import unittest

from heterollm_sim.component_presets import (
    COMPONENT_KINDS,
    component_preset_detail,
    component_preset_page,
    get_component_preset,
    list_component_presets,
    materialize_component_payload,
)


PUBLIC_IDS = {
    "amd-ryzen-9-9950x3d",
    "samsung-hbm3e-36gb-9_2",
    "gddr7-16gb-30_0-256bit",
    "sk-hynix-hbf-512gb",
    "ymtc-zhitai-ti-pro9100",
    "samsung-ddr5-32gb-udimm-5600",
    "acer-local-ddr5-128gb-5600-dual-channel",
    "sram-cim-analytical-tile",
    "nvidia-b200-sxm-gpu",
    "nvidia-rtx-5080",
}
REMOVED_IDS = {
    "hbm3e-24gb-8_0",
    "nvidia-h100-sxm-gpu",
    "ocp-hbf-2026-512gb",
    "nvidia-h100-sxm-bundle",
}


class CuratedComponentCatalogTests(unittest.TestCase):
    def test_public_catalog_contains_only_curated_components(self):
        rows = list_component_presets()
        self.assertEqual({row["id"] for row in rows}, PUBLIC_IDS)
        self.assertEqual([row["id"] for row in rows], sorted(PUBLIC_IDS))
        self.assertTrue(all(row["preset_type"] == "component" for row in rows))
        self.assertTrue(all(row["component_kind"] in COMPONENT_KINDS for row in rows))

    def test_legacy_ids_are_removed_from_all_component_accessors(self):
        for preset_id in REMOVED_IDS:
            with self.subTest(preset_id=preset_id):
                with self.assertRaises(KeyError):
                    get_component_preset(preset_id)
                with self.assertRaises(KeyError):
                    materialize_component_payload(preset_id)
                with self.assertRaises(KeyError):
                    component_preset_detail(preset_id)

    def test_vendor_values_and_provenance(self):
        samsung = component_preset_detail("samsung-hbm3e-36gb-9_2")["component"]
        self.assertEqual(samsung["capacity_bytes"], 36_000_000_000)
        self.assertAlmostEqual(samsung["read_bandwidth_gbps"], 9420.8)
        self.assertEqual(samsung["metadata"]["vendor_parameter_provenance"]["pin_speed_gbps"]["unit"], "Gb/s_per_pin")

        hbf = component_preset_detail("sk-hynix-hbf-512gb")["component"]
        self.assertEqual(hbf["capacity_bytes"], 512_000_000_000)
        self.assertEqual(hbf["read_bandwidth_gbps"], 3904.0)
        self.assertEqual(hbf["write_bandwidth_gbps"], 217.6)
        self.assertEqual(hbf["metadata"]["access_mode"], "memory")
        self.assertEqual(hbf["metadata"]["read_latency_ns"], 4000.0)
        self.assertEqual(hbf["metadata"]["write_latency_ns"], 75000.0)
        self.assertEqual(hbf["metadata"]["capability_status"]["write_bandwidth_gbps"], "analytical_user_configured")

        tip = component_preset_detail("ymtc-zhitai-ti-pro9100")["component"]
        self.assertEqual(tip["metadata"]["vendor_parameter_provenance"]["mpn"]["value"], "ZTSS3CB08D6CMC")
        self.assertEqual(tip["read_bandwidth_gbps"], 96.0)
        self.assertEqual(tip["write_bandwidth_gbps"], 85.6)
        self.assertEqual(tip["ports"][0]["bandwidth_gbps"], 126.03076923076924)
        self.assertIn("96 Gb/s", tip["metadata"]["vendor_parameter_provenance"]["read_bandwidth_gbps"]["formula"])

        b200 = component_preset_detail("nvidia-b200-sxm-gpu")["component"]
        self.assertEqual(b200["capacity_bytes"], 0)
        self.assertEqual(b200["read_bandwidth_gbps"], 0.0)
        self.assertEqual(b200["write_bandwidth_gbps"], 0.0)
        self.assertIsNone(b200["metadata"]["technology"]["memory_stack_count"])
        self.assertTrue(b200["metadata"]["memory_requires_explicit_component"])

        cpu = component_preset_detail("amd-ryzen-9-9950x3d")["component"]
        self.assertEqual(cpu["metadata"]["technology"]["core_count"], 16)
        self.assertEqual(cpu["metadata"]["technology"]["thread_count"], 32)
        self.assertEqual(cpu["metadata"]["sources"][0]["publisher"], "AMD")
        self.assertEqual(cpu["metadata"]["technology"]["l2_capacity_bytes"], 16 * 1024 * 1024)
        self.assertEqual(cpu["metadata"]["technology"]["l3_capacity_bytes"], 128 * 1024 * 1024)
        cpu_levels = cpu["metadata"]["cost_profile_template"]["cache_hierarchy"]["levels"]
        self.assertEqual(cpu_levels[0]["capacity_bytes"], 1280 * 1024)
        self.assertEqual(cpu_levels[1]["capacity_bytes"], 16 * 1024 * 1024)
        self.assertEqual(cpu_levels[2]["capacity_bytes"], 128 * 1024 * 1024)
        self.assertNotIn("Grace", cpu["metadata"]["cost_profile_parameter_basis"]["cache_hierarchy.levels[2].capacity_bytes"])
        self.assertNotIn("Grace", " ".join(cpu["metadata"]["cost_profile_parameter_basis"].values()))

        rtx = component_preset_detail("nvidia-rtx-5080")["component"]
        self.assertEqual(rtx["metadata"]["technology"]["cuda_cores"], 10752)
        self.assertEqual(rtx["metadata"]["technology"]["device_memory_gb"], 16.0)
        self.assertEqual(rtx["metadata"]["technology"]["device_memory_bandwidth_gb_s"], 960.0)
        self.assertEqual(rtx["capacity_bytes"], 0)
        self.assertEqual(rtx["read_bandwidth_gbps"], 0.0)
        self.assertEqual(rtx["write_bandwidth_gbps"], 0.0)
        self.assertAlmostEqual(rtx["peak_ops_per_s"], 112.6e12)
        self.assertFalse(rtx["metadata"]["external_memory_modeled_in_metadata"])
        self.assertTrue(rtx["metadata"]["memory_requires_explicit_component"])
        rtx_levels = rtx["metadata"]["cost_profile_template"]["cache_hierarchy"]["levels"]
        self.assertEqual(rtx_levels[0]["capacity_bytes"], 10752 * 1024)
        self.assertEqual(rtx_levels[-1]["capacity_bytes"], 65536 * 1024)
        self.assertNotIn("Grace", " ".join(rtx["metadata"]["cost_profile_parameter_basis"].values()))

        gddr7 = component_preset_detail("gddr7-16gb-30_0-256bit")["component"]
        self.assertEqual(gddr7["metadata"]["technology"]["memory_type"], "GDDR7")
        self.assertEqual(gddr7["ports"][0]["protocol"], "GDDR7")
        self.assertEqual(gddr7["ports"][0]["bandwidth_gbps"], 7680.0)
        self.assertEqual(gddr7["capacity_bytes"], 16_000_000_000)

    def test_page_filters_the_curated_catalog(self):
        page = component_preset_page(component_kind="hbf")
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["id"], "sk-hynix-hbf-512gb")


if __name__ == "__main__":
    unittest.main()
