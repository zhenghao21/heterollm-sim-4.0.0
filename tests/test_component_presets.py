import http.client
import json
import threading
import unittest

from heterollm_sim.component_presets import (
    COMPONENT_KINDS,
    EVIDENCE_LEVELS,
    component_preset_detail,
    component_preset_page,
    list_component_presets,
    materialize_component_payload,
)
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.config import _component_profile_registries_from_dict
from heterollm_sim.topology import validate_topology
from heterollm_sim.web import build_server


REQUIRED_PRESETS = {
    "nvidia-grace-cpu-gh200",
    "nvidia-grace-cpu-gb200",
    "hbm2e-16gb-3_2",
    "hbm2e-16gb-0_4625tbs-gaudi3-slice",
    "hbm3-16gb-0_6625tbs-mi300a-slice",
    "hbm3-16gb-0_667tbs-product-slice",
    "hbm3-16gb-0_670tbs-h100-slice",
    "hbm3-24gb-0_665625tbs-mi300x-slice",
    "hbm3e-24gb-0_800tbs-h200-slice",
    "hbm3e-24gb-0_833tbs-gh200-slice",
    "hbm3e-24gb-1_000tbs-gb200-slice",
    "hbm-pim-16gb-3_2-analysis",
    "lpddr5x-gh200-480gb-500gbs-aggregate",
    "lpddr5x-gb200-480gb-512gbs-aggregate",
    "jedec-hbm3-24gb-6_4",
    "hbm3e-24gb-8_0",
    "hbm3e-36gb-9_2",
    "jedec-hbm4-64gb-8_0",
    "nvidia-h100-sxm-gpu",
    "nvidia-h200-sxm-gpu",
    "samsung-pm1743-15_36tb",
    "solidigm-d7-p5810-800gb",
    "ocp-hbf-2026-512gb",
    "digital-sram-cim-analysis",
}
REQUIRED_BUNDLES = {
    "nvidia-h100-sxm-bundle",
    "nvidia-h200-sxm-bundle",
}


class ComponentPresetCatalogTests(unittest.TestCase):
    def test_runtime_profile_templates_are_complete_and_parseable(self):
        """A newly imported preset must not fall back to a legacy profile."""

        cases = {
            "jedec-hbm3-24gb-6_4": "hbm",
            "hbm3e-36gb-9_2": "hbm",
            "nvidia-grace-cpu-gb200": "cpu",
            "nvidia-h100-sxm-gpu": "gpu",
            "lpddr5x-gb200-480gb-512gbs-aggregate": "host_memory",
            "digital-sram-cim-analysis": "cim",
        }
        registry = {kind: {} for kind in set(cases.values())}
        for preset_id, expected_kind in cases.items():
            component = materialize_component_payload(preset_id)
            metadata = component["metadata"]
            self.assertEqual(metadata["cost_profile_key"], expected_kind)
            template = metadata["cost_profile_template"]
            basis = metadata["cost_profile_parameter_basis"]
            self.assertIsInstance(template, dict)
            self.assertIsInstance(basis, dict)
            self.assertTrue(template)
            self.assertTrue(basis)
            registry[expected_kind][preset_id] = template

        parsed = _component_profile_registries_from_dict(registry)
        self.assertEqual(set(parsed), set(registry))
        self.assertEqual(
            set(parsed["hbm"]),
            {"jedec-hbm3-24gb-6_4", "hbm3e-36gb-9_2"},
        )
        self.assertGreater(
            parsed["hbm"]["jedec-hbm3-24gb-6_4"].read_latency_ns,
            0.0,
        )
        self.assertEqual(parsed["cpu"]["nvidia-grace-cpu-gb200"].pipeline.core_count, 72)
        self.assertEqual(parsed["cpu"]["nvidia-grace-cpu-gb200"].pipeline.simd_width_bits, 128)
        self.assertAlmostEqual(
            parsed["gpu"]["nvidia-h100-sxm-gpu"].tensor_core.peak_tops("bf16"),
            989.5,
            places=3,
        )
        self.assertAlmostEqual(
            parsed["gpu"]["nvidia-h100-sxm-gpu"].tensor_core.peak_tops("fp16"),
            989.5,
            places=3,
        )
        self.assertAlmostEqual(
            parsed["gpu"]["nvidia-h100-sxm-gpu"].tensor_core.peak_tops("int8"),
            1979.0,
            places=3,
        )
        self.assertEqual(
            parsed["cpu"]["nvidia-grace-cpu-gb200"].cache_hierarchy.levels[0].capacity_bytes,
            72 * 64 * 1024,
        )

    def test_every_hbm_and_memory_preset_exposes_latency_and_profile_basis(self):
        for item in list_component_presets():
            if item["component_kind"] not in {"hbm", "host_memory"}:
                continue
            component = materialize_component_payload(item["id"])
            metadata = component["metadata"]
            self.assertGreater(metadata["read_latency_ns"], 0.0)
            self.assertGreater(metadata["write_latency_ns"], 0.0)
            self.assertGreater(metadata["transfer_granularity_bytes"], 0)
            self.assertGreater(metadata["max_outstanding_requests"], 0)
            self.assertIn("read_latency_ns", metadata["cost_profile_parameter_basis"])
            self.assertIn("write_latency_ns", metadata["cost_profile_parameter_basis"])

    def test_bundle_components_carry_the_same_runtime_template_contract(self):
        for preset_id in REQUIRED_BUNDLES:
            detail = component_preset_detail(preset_id)
            for component in detail["components"]:
                metadata = component["metadata"]
                if component["kind"] not in {"gpu", "hbm"}:
                    continue
                self.assertIn("cost_profile_template", metadata)
                self.assertIn("cost_profile_parameter_basis", metadata)
                self.assertEqual(
                    metadata["cost_profile_key"],
                    "gpu" if component["kind"] == "gpu" else "hbm",
                )
                if component["kind"] == "hbm":
                    self.assertGreater(metadata["read_latency_ns"], 0.0)

    def test_catalog_contains_required_metadata_without_full_components(self):
        presets = list_component_presets()
        ids = [item["id"] for item in presets]

        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue((REQUIRED_PRESETS | REQUIRED_BUNDLES).issubset(set(ids)))

        for item in presets:
            self.assertRegex(item["id"], r"^[a-z0-9][a-z0-9_-]*$")
            self.assertIn(item["component_kind"], COMPONENT_KINDS | {"topology_bundle"})
            self.assertIn(item["preset_type"], {"component", "topology_bundle"})
            self.assertIn(item["evidence_level"], EVIDENCE_LEVELS)
            if item["component_kind"] == "cpu":
                self.assertEqual(item["capacity_bytes"], 0)
            else:
                self.assertGreater(item["capacity_bytes"], 0)
            self.assertIsInstance(item["sources"], list)
            self.assertTrue(item["sources"])
            self.assertTrue(all(source["title"] for source in item["sources"]))
            self.assertTrue(all(source["publisher"] for source in item["sources"]))
            self.assertTrue(all(source["accessed_at"] for source in item["sources"]))
            self.assertIsInstance(item["limitations"], list)
            self.assertTrue(item["limitations"])
            self.assertIn("value_scope", item)
            self.assertIn("conditions", item)
            self.assertTrue(
                "组件模板" in item["usage_hint"]
                or "组合拓扑" in item["usage_hint"]
            )
            self.assertNotIn("component", item)

    def test_page_envelope_supports_filters(self):
        page = component_preset_page(
            component_kind="hbm",
            evidence_level="S2_VENDOR_DECLARED",
        )

        self.assertEqual(page["total"], len(page["items"]))
        self.assertIn("items", page)
        self.assertIn("filters", page)
        self.assertIn("catalog", page)
        self.assertIn("hbm", page["filters"]["component_kind"])
        self.assertIn("cpu", page["filters"]["component_kind"])
        self.assertIn("ssd", page["filters"]["component_kind"])
        self.assertIn("high_io_ssd", page["filters"]["component_kind"])
        self.assertTrue(page["items"])
        self.assertTrue(all(item["component_kind"] == "hbm" for item in page["items"]))
        self.assertTrue(
            all(item["evidence_level"] == "S2_VENDOR_DECLARED" for item in page["items"])
        )

    def test_detail_returns_one_parseable_component_template(self):
        for preset_id in REQUIRED_PRESETS:
            with self.subTest(preset_id=preset_id):
                detail = component_preset_detail(preset_id)

                self.assertEqual(detail["preset"]["id"], preset_id)
                self.assertEqual(detail["links"], [])
                self.assertIn("添加到当前拓扑", detail["usage_hint"])

                component = detail["component"]
                self.assertEqual(component["component_id"], preset_id.replace("-", "_"))
                self.assertEqual(component["kind"], detail["preset"]["component_kind"])
                self.assertEqual(component["metadata"]["preset"]["id"], preset_id)
                self.assertIn("sources", component["metadata"])

                hardware = hardware_from_dict(
                    {
                        "name": "single-component-template",
                        "components": [component],
                        "links": [],
                        "require_connected": False,
                    }
                )
                self.assertEqual(len(hardware.components), 1)
                self.assertEqual(hardware.links, ())

    def test_hopper_sxm_bundles_are_valid_expanded_gpu_root_groups(self):
        cases = {
            "nvidia-h100-sxm-bundle": (5, "HBM3", 80_000_000_000, 26_800.0, 16_000_000_000, "vendor_documented_active_stacks"),
            "nvidia-h200-sxm-bundle": (6, "HBM3E", 141_000_000_000, 38_400.0, 24_000_000_000, "derived_from_product_total_and_24GB_stack_class"),
        }
        for preset_id, (stack_count, generation, capacity, bandwidth, raw_capacity, count_status) in cases.items():
            with self.subTest(preset_id=preset_id):
                detail = component_preset_detail(preset_id)
                self.assertEqual(detail["preset"]["preset_type"], "topology_bundle")
                self.assertEqual(len(detail["components"]), stack_count + 1)
                self.assertEqual(len(detail["links"]), stack_count)
                self.assertEqual(detail["group"]["root"], "gpu0")
                self.assertFalse(detail["group"]["collapsed"])
                self.assertEqual(detail["group"]["members"][0], "gpu0")
                memories = [item for item in detail["components"] if item["kind"] == "hbm"]
                self.assertEqual(len(memories), stack_count)
                self.assertEqual(sum(item["capacity_bytes"] for item in memories), capacity)
                self.assertAlmostEqual(sum(item["read_bandwidth_gbps"] for item in memories), bandwidth)
                self.assertTrue(all(item["metadata"]["technology"]["generation"] == generation for item in memories))
                for index, memory in enumerate(memories):
                    composition = memory["metadata"]["physical_composition"]
                    self.assertEqual(composition["simulator_representation"], "single_physical_unit_node")
                    self.assertEqual(composition["physical_unit_kind"], "HBM_stack")
                    self.assertEqual(composition["physical_unit_count"], 1)
                    self.assertEqual(composition["unit_index"], index)
                    self.assertEqual(composition["unit_count_in_product"], stack_count)
                    self.assertEqual(composition["unit_count_status"], count_status)
                    self.assertEqual(composition["controller_component_id"], "gpu0")
                    self.assertEqual(composition["memory_subsystem_id"], "gpu0")
                    self.assertEqual(composition["unit_capacity_bytes"], memory["capacity_bytes"])
                    self.assertEqual(composition["unit_raw_capacity_bytes"], raw_capacity)
                    self.assertEqual(composition["product_total_capacity_bytes"], capacity)
                    self.assertAlmostEqual(composition["unit_bandwidth_gbps"], bandwidth / stack_count)
                    self.assertTrue(composition["unit_count_formula"])
                    self.assertTrue(composition["source_basis"])
                    self.assertIn("parameter_basis", memory["metadata"])
                    self.assertIn("provenance", memory["metadata"])

                hardware = hardware_from_dict(
                    {
                        "name": preset_id,
                        "components": detail["components"],
                        "links": detail["links"],
                        "require_connected": True,
                    }
                )
                report = validate_topology(hardware)
                self.assertTrue(report.is_valid, report.format())

    def test_hbm_generations_stay_metadata_not_kind(self):
        cases = {
            "jedec-hbm3-24gb-6_4": ("HBM3", "JESD238B.01"),
            "hbm3e-24gb-8_0": ("HBM3E", "HBM3E-1.2.0"),
            "hbm3e-36gb-9_2": ("HBM3E", "HBM3E-1.2.0"),
            "jedec-hbm4-64gb-8_0": ("HBM4", "JESD270-4A"),
        }

        for preset_id, (generation, revision) in cases.items():
            component = materialize_component_payload(preset_id)

            self.assertEqual(component["kind"], "hbm")
            self.assertEqual(
                component["metadata"]["technology"]["generation"],
                generation,
            )
            self.assertEqual(component["metadata"]["revision"], revision)
            self.assertNotIn(generation.lower(), COMPONENT_KINDS)

        hbm3e_8 = component_preset_detail("hbm3e-24gb-8_0")
        self.assertEqual(hbm3e_8["preset"]["name"], "HBM3E 24GB 1.024 TB/s Stack")
        self.assertTrue(
            any(
                source["url"]
                == "https://semiconductor.samsung.com/dram/hbm/hbm3e/"
                for source in hbm3e_8["preset"]["sources"]
            )
        )

    def test_product_slice_presets_publish_physical_and_derived_provenance(self):
        preset_ids = {
            "hbm3-16gb-0_670tbs-h100-slice",
            "hbm3-16gb-0_667tbs-product-slice",
            "hbm3-16gb-0_6625tbs-mi300a-slice",
            "hbm3-24gb-0_665625tbs-mi300x-slice",
            "hbm3e-24gb-0_800tbs-h200-slice",
            "hbm3e-24gb-0_833tbs-gh200-slice",
            "hbm3e-24gb-1_000tbs-gb200-slice",
            "hbm2e-16gb-0_4625tbs-gaudi3-slice",
            "hbm-pim-16gb-3_2-analysis",
        }
        for preset_id in preset_ids:
            component = materialize_component_payload(preset_id)
            composition = component["metadata"]["physical_composition"]
            self.assertEqual(composition["simulator_representation"], "single_physical_unit_node")
            self.assertEqual(composition["physical_unit_kind"], "HBM_stack")
            self.assertEqual(composition["component_preset_id"], preset_id)
            self.assertEqual(composition["unit_capacity_bytes"], component["capacity_bytes"])
            self.assertEqual(composition["unit_bandwidth_gbps"], component["read_bandwidth_gbps"])
            self.assertTrue(composition["unit_count_formula"])
            self.assertTrue(component["metadata"]["provenance"]["source_basis"])

    def test_grace_lpddr_preset_is_explicit_unknown_count_aggregate(self):
        component = materialize_component_payload(
            "lpddr5x-gh200-480gb-500gbs-aggregate"
        )
        composition = component["metadata"]["physical_composition"]
        self.assertEqual(component["kind"], "host_memory")
        self.assertEqual(component["capacity_bytes"], 480_000_000_000)
        self.assertEqual(component["read_bandwidth_gbps"], 4_000.0)
        self.assertEqual(composition["simulator_representation"], "aggregate_node")
        self.assertIsNone(composition["physical_unit_count"])
        self.assertEqual(
            composition["physical_unit_count_status"],
            "not_reliably_disclosed",
        )
        gb200 = materialize_component_payload(
            "lpddr5x-gb200-480gb-512gbs-aggregate"
        )
        self.assertEqual(gb200["capacity_bytes"], 480_000_000_000)
        self.assertEqual(gb200["read_bandwidth_gbps"], 4_096.0)
        self.assertEqual(gb200["ports"][0]["bandwidth_gbps"], 4_096.0)

    def test_selected_semantics_are_explicit(self):
        grace = materialize_component_payload("nvidia-grace-cpu-gb200")
        self.assertEqual(grace["kind"], "cpu")
        self.assertEqual(grace["metadata"]["technology"]["core_count"], 72)
        self.assertEqual(grace["peak_ops_per_s"], 0.0)
        self.assertTrue(grace["metadata"]["cpu_profile_required"])
        self.assertIn(
            "peak_ops_per_s",
            grace["metadata"]["unknown_value_sentinels"],
        )
        self.assertEqual(grace["ports"][-1]["bandwidth_gbps"], 4096.0)

        hbf = materialize_component_payload("ocp-hbf-2026-512gb")
        self.assertEqual(hbf["kind"], "hbf")
        self.assertEqual(hbf["read_bandwidth_gbps"], 24000.0)
        self.assertEqual(hbf["ports"][0]["bandwidth_gbps"], 2048.0)
        self.assertEqual(hbf["write_bandwidth_gbps"], 0.0)
        self.assertEqual(hbf["metadata"]["evidence_level"], "S3_VENDOR_PREPRODUCTION")
        self.assertEqual(
            hbf["metadata"]["physical_composition"]["physical_unit_kind"],
            "HBF_stack",
        )
        self.assertEqual(hbf["metadata"]["physical_composition"]["physical_unit_count"], 1)
        self.assertEqual(hbf["metadata"]["capacity_scope"], "up_to_512GB")
        self.assertEqual(hbf["metadata"]["bandwidth_scope"], "grade3_up_to_3TB_per_s")
        self.assertGreater(hbf["metadata"]["dma_bandwidth_gbps"], 0)
        self.assertGreater(hbf["metadata"]["dma_latency_ns"], 0)
        self.assertGreaterEqual(hbf["metadata"]["max_outstanding_requests"], 1)
        self.assertIn("analytical", hbf["metadata"]["storage_transport_parameter_basis"])
        self.assertIn("write_bandwidth_gbps", hbf["metadata"]["unknown_value_sentinels"])
        self.assertIn(
            "IR 写带宽字段保持 0.0",
            " ".join(hbf["metadata"]["applicability_limitations"]),
        )

        gpu = materialize_component_payload("nvidia-h200-sxm-gpu")
        self.assertEqual(gpu["kind"], "gpu")
        self.assertEqual(gpu["metadata"]["component_scope"], "compute_side_only")
        self.assertTrue(gpu["metadata"]["external_memory_modeled_in_metadata"])

        cim = materialize_component_payload("digital-sram-cim-analysis")
        self.assertEqual(cim["kind"], "digital_sram_cim")
        self.assertEqual(
            cim["metadata"]["evidence_level"],
            "A_ANALYTICAL",
        )

    def test_bandwidth_values_are_ir_gbps_not_display_gbs(self):
        expected = {
            "jedec-hbm3-24gb-6_4": (6553.6, 6553.6),
            "hbm3e-24gb-8_0": (8192.0, 8192.0),
            "hbm3e-36gb-9_2": (9420.8, 9420.8),
            "jedec-hbm4-64gb-8_0": (16384.0, 16384.0),
            "ocp-hbf-2026-512gb": (24000.0, 2048.0),
        }
        for preset_id, (media_bandwidth, port_bandwidth) in expected.items():
            with self.subTest(preset_id=preset_id):
                component = materialize_component_payload(preset_id)
                self.assertAlmostEqual(component["read_bandwidth_gbps"], media_bandwidth)
                self.assertAlmostEqual(component["ports"][0]["bandwidth_gbps"], port_bandwidth)

        h100 = materialize_component_payload("nvidia-h100-sxm-gpu")
        self.assertEqual(h100["peak_ops_per_s"], 989_500_000_000_000.0)
        self.assertEqual(h100["metadata"]["technology"]["bf16_tensor_dense_tflops"], 989.5)
        self.assertEqual(h100["metadata"]["technology"]["device_memory_bandwidth_gbps"], 26800.0)
        self.assertAlmostEqual(h100["ports"][0]["bandwidth_gbps"], 5360.0)
        self.assertEqual(h100["ports"][-2]["bandwidth_gbps"], 3600.0)
        self.assertEqual(
            h100["ports"][-2]["metadata"]["aggregate_bidirectional_gbps"],
            7200.0,
        )
        self.assertEqual(
            h100["ports"][-2]["metadata"]["aggregate_bidirectional_display_value"],
            "900 GB/s",
        )
        self.assertEqual(h100["ports"][-1]["bandwidth_gbps"], 512.0)
        self.assertEqual(
            h100["ports"][-1]["metadata"]["aggregate_bidirectional_display_value"],
            "128 GB/s",
        )

        h200 = materialize_component_payload("nvidia-h200-sxm-gpu")
        self.assertEqual(h200["peak_ops_per_s"], 989_500_000_000_000.0)
        self.assertEqual(h200["metadata"]["technology"]["bf16_tensor_dense_tflops"], 989.5)
        self.assertEqual(h200["metadata"]["technology"]["device_memory_bandwidth_gbps"], 38400.0)
        self.assertAlmostEqual(h200["ports"][0]["bandwidth_gbps"], 6400.0)

        pm1743 = materialize_component_payload("samsung-pm1743-15_36tb")
        self.assertEqual(pm1743["kind"], "high_io_ssd")
        self.assertEqual(pm1743["read_bandwidth_gbps"], 112.0)
        self.assertEqual(pm1743["write_bandwidth_gbps"], 56.8)
        self.assertAlmostEqual(pm1743["ports"][0]["bandwidth_gbps"], 126.03076923076924)
        self.assertEqual(pm1743["metadata"]["dma_bandwidth_gbps"], 112.0)
        self.assertGreater(pm1743["metadata"]["dma_latency_ns"], 0)
        self.assertEqual(pm1743["metadata"]["transfer_granularity_bytes"], 4096)
        self.assertEqual(
            pm1743["metadata"]["conditions"],
            ["128KiB sequential", "FIO", "Ubuntu 18.04.2", "QD256", "1 worker"],
        )
        pm1743_sources = {
            source["url"]
            for source in component_preset_detail("samsung-pm1743-15_36tb")["preset"]["sources"]
        }
        self.assertEqual(
            pm1743_sources,
            {
                "https://download.semiconductor.samsung.com/resources/white-paper/PM1743_White_Paper_240510.pdf",
                "https://semiconductor.samsung.com/ssd/enterprise-ssd/pm1743/",
            },
        )

        p5810 = materialize_component_payload("solidigm-d7-p5810-800gb")
        self.assertEqual(p5810["kind"], "ssd")
        self.assertEqual(p5810["read_bandwidth_gbps"], 51.2)
        self.assertEqual(p5810["write_bandwidth_gbps"], 35.2)
        self.assertAlmostEqual(p5810["ports"][0]["bandwidth_gbps"], 63.01538461538462)
        self.assertEqual(p5810["metadata"]["dma_bandwidth_gbps"], 51.2)
        self.assertGreater(p5810["metadata"]["dma_latency_ns"], 0)

    def test_every_capability_field_publishes_a_machine_readable_unit(self):
        required_component_units = {
            "capacity_bytes": "B",
            "peak_ops_per_s": "op/s",
            "read_bandwidth_gbps": "Gb/s_decimal_one_way",
            "write_bandwidth_gbps": "Gb/s_decimal_one_way",
            "dma_energy_pj_per_byte": "pJ/B",
        }
        for preset_id in REQUIRED_PRESETS:
            with self.subTest(preset_id=preset_id):
                component = materialize_component_payload(preset_id)
                units = component["metadata"]["capability_units"]
                for field, unit in required_component_units.items():
                    self.assertEqual(units[field], unit)
                for port in component["ports"]:
                    self.assertEqual(
                        port["metadata"]["capability_units"]["lanes"],
                        "lane_count",
                    )
                    self.assertEqual(
                        port["metadata"]["capability_units"]["bandwidth_gbps"],
                        "Gb/s_decimal_one_way",
                    )

    def test_user_visible_explanations_are_chinese_and_display_units(self):
        forbidden = r"Gb/s|Gbps|\bbit\b"
        for preset_id in REQUIRED_PRESETS:
            with self.subTest(preset_id=preset_id):
                detail = component_preset_detail(preset_id)
                visible_texts = [
                    detail["preset"]["notes"],
                    detail["component"]["metadata"]["measurement_basis"],
                    detail["component"]["metadata"]["technology"].get("kind_policy", ""),
                    *detail["preset"]["limitations"],
                    *detail["component"]["metadata"]["applicability_limitations"],
                ]
                for text in visible_texts:
                    if not text:
                        continue
                    self.assertRegex(text, r"[\u3400-\u9fff]")
                    self.assertNotRegex(text, forbidden)

    def test_unknown_preset_raises_key_error(self):
        with self.assertRaises(KeyError):
            component_preset_detail("does-not-exist")


class ComponentPresetApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = build_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5.0)
        cls.server.server_close()

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            body = None
            headers = {}
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def test_list_detail_unknown_and_wrong_method(self):
        status, page = self.request("GET", "/api/component-presets")
        self.assertEqual(status, 200)
        self.assertIn("items", page)
        self.assertIn("total", page)
        self.assertIn("filters", page)
        self.assertTrue(REQUIRED_PRESETS.issubset({item["id"] for item in page["items"]}))

        status, filtered = self.request(
            "GET", "/api/component-presets?kind=hbf&tag=flash"
        )
        self.assertEqual(status, 200)
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["id"], "ocp-hbf-2026-512gb")

        status, detail = self.request(
            "GET", "/api/component-presets/hbm3e-36gb-9_2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["id"], "hbm3e-36gb-9_2")
        self.assertEqual(detail["component"]["kind"], "hbm")
        self.assertEqual(detail["links"], [])

        status, payload = self.request("GET", "/api/component-presets/missing")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")
        self.assertEqual(payload["error"]["message"], "未找到指定的组件预设")

        status, payload = self.request("POST", "/api/component-presets", {})
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")
        self.assertEqual(payload["error"]["message"], "此端点要求使用 GET 方法")


if __name__ == "__main__":
    unittest.main()
