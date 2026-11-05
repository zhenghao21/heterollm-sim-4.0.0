"""HBM address-capacity and evidence-boundary regressions."""

from dataclasses import fields
import unittest

from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.memory_types import DramConfig


GIB = 1 << 30


class HbmStandardOrganizationTests(unittest.TestCase):
    def make_config(self, **overrides):
        arguments = dict(capacity_bytes=36 * GIB,
                         read_bandwidth_gbps=9600.0, write_bandwidth_gbps=8000.0)
        arguments.update(overrides)
        return DramConfig.from_mapping(make_hbm_geometry_config(**arguments))

    def test_24gbit_12high_geometry_matches_valid_address_bits(self):
        config = self.make_config(die_density_gbits=24, dies_per_stack=12)
        # BA[3:0] has 16 values; SID=11 and RA[14:13]=11 are invalid.
        self.assertEqual(config.bank_count, 16 * 3)
        self.assertEqual(config.rows_per_bank, 3 * (1 << 13))
        self.assertEqual(config.bank_groups_per_rank, 12)
        self.assertEqual(config.banks_per_group, 4)
        self.assertEqual(config.row_bytes, (1 << 5) * 32)
        self.assertEqual(config.burst_bytes, 32)
        self.assertEqual(config.lane_count, 32)
        self.assertEqual(config.lane_count * config.data_width_bits, 1024)
        self.assertEqual(config.computed_capacity_bytes, 38_654_705_664)
        self.assertEqual(config.capacity_bytes, 36 * GIB)
        self.assertEqual(config.bank_geometry_scope, "per_stack")
        self.assertEqual(config.addressable_die_count, 1)
        self.assertEqual(config.dies_per_stack, 12)

    def test_stacks_add_external_lanes_without_repeating_aggregate_bandwidth(self):
        single = self.make_config(die_density_gbits=24, dies_per_stack=12)
        two = self.make_config(capacity_bytes=72 * GIB, stacks=2,
                               die_density_gbits=24, dies_per_stack=12)
        self.assertEqual(two.lane_count, 64)
        self.assertEqual(two.computed_capacity_bytes, 2 * single.computed_capacity_bytes)
        self.assertEqual(two.physical_interface_bandwidth_gb_s, 1200.0)
        self.assertEqual(two.read_bandwidth_gb_s, 1200.0)
        self.assertEqual(two.write_bandwidth_gb_s, 1000.0)
        self.assertEqual(two.effective_lane_bandwidth_gb_s,
                         single.effective_lane_bandwidth_gb_s / 2)

    def test_every_supported_hbm3_organization_has_exact_binary_capacity(self):
        for density in (8, 16, 24, 32):
            for height in (4, 8, 12, 16):
                with self.subTest(density=density, height=height):
                    raw = density * height * GIB // 8
                    config = self.make_config(capacity_bytes=raw,
                                              die_density_gbits=density,
                                              dies_per_stack=height)
                    self.assertEqual(config.computed_capacity_bytes, raw)
                    self.assertEqual(config.metadata["raw_capacity_bytes"], raw)
                    self.assertEqual(config.metadata["unexposed_capacity_bytes"], 0)

    def test_visible_capacity_does_not_grow_rows_or_change_raw_geometry(self):
        config = self.make_config(capacity_bytes=36_000_000_000,
                                  die_density_gbits=24, dies_per_stack=12)
        self.assertEqual(config.rows_per_bank, 24576)
        self.assertEqual(config.capacity_bytes, 36_000_000_000)
        self.assertEqual(config.computed_capacity_bytes, 36 * GIB)
        self.assertEqual(config.metadata["visible_capacity_bytes"], 36_000_000_000)
        self.assertEqual(config.metadata["unexposed_capacity_bytes"], 2_654_705_664)

    def test_reference_selection_is_capacity_minimal_and_never_vendor_verified(self):
        for visible, density, height in ((8, 8, 8), (12, 8, 12), (16, 16, 8),
                                         (24, 24, 8), (36, 24, 12), (48, 32, 12),
                                         (64, 32, 16)):
            with self.subTest(visible=visible):
                config = self.make_config(capacity_bytes=visible * GIB)
                self.assertEqual(config.metadata["die_density_gbits"], density)
                self.assertEqual(config.dies_per_stack, height)
                self.assertFalse(config.metadata["vendor_verified"])
                for name in ("die_density_gbits", "dies_per_stack"):
                    self.assertIn(name, config.metadata["unknown_parameters"])
                    self.assertEqual(config.metadata["parameter_provenance"][name]["status"],
                                     "selected_reference")

    def test_supplied_density_constrains_reference_selection(self):
        config = self.make_config(capacity_bytes=12 * GIB, die_density_gbits=24)
        self.assertEqual(config.dies_per_stack, 4)
        self.assertNotIn("die_density_gbits", config.metadata["unknown_parameters"])
        self.assertIn("dies_per_stack", config.metadata["unknown_parameters"])
        self.assertTrue(any("vendor-specific" in text
                            for text in config.metadata["geometry_limitations"]))

    def test_jedec_hbm2e_organizations_keep_64bit_pseudochannels(self):
        for height, raw in ((4, 8 * GIB), (8, 16 * GIB)):
            with self.subTest(height=height):
                config = self.make_config(generation="HBM2E", capacity_bytes=raw,
                                          die_density_gbits=16, dies_per_stack=height)
                self.assertEqual(config.channels, 8)
                self.assertEqual(config.lane_count, 16)
                self.assertEqual(config.data_width_bits, 64)
                self.assertEqual(config.burst_bytes, 4 * (64 // 8))
                self.assertEqual(config.rows_per_bank, 32768)
                self.assertEqual(config.bank_count, 16 if height == 4 else 32)
                self.assertEqual(config.computed_capacity_bytes, raw)
                self.assertEqual(config.metadata["jedec_standard"], "JESD235B")

    def test_hbm4_table4_and_table5_organizations(self):
        for density in (24, 32):
            for height in (4, 8, 12, 16):
                with self.subTest(density=density, height=height):
                    raw = density * height * GIB // 8
                    config = self.make_config(generation="HBM4", capacity_bytes=raw,
                                              die_density_gbits=density, dies_per_stack=height)
                    self.assertEqual(config.channels, 32)
                    self.assertEqual(config.lane_count, 64)
                    self.assertEqual(config.lane_count * config.data_width_bits, 2048)
                    self.assertEqual(config.rows_per_bank, 12288 if density == 24 else 16384)
                    self.assertEqual(config.bank_groups_per_rank, height // 2)
                    self.assertEqual(config.banks_per_group, 8)
                    self.assertEqual(config.bank_count, height * 4)
                    self.assertEqual(config.computed_capacity_bytes, raw)
                    self.assertEqual(config.metadata["jedec_standard"], "JESD270-4A")
                    self.assertFalse(any("vendor-specific" in item for item in
                                         config.metadata["geometry_limitations"]))

    def test_effective_page_is_distinguished_from_activated_page(self):
        for generation in ("HBM3", "HBM3E", "HBM4"):
            with self.subTest(generation=generation):
                config = self.make_config(generation=generation,
                                          die_density_gbits=24, dies_per_stack=12)
                self.assertEqual(config.row_bytes, 1024)
                self.assertEqual(config.metadata["effective_addressed_page_bytes"], 1024)
                self.assertEqual(config.metadata["activated_page_bytes"], 2048)
                self.assertEqual(config.control_mode, "jedec_hbm")

    def test_geometry_evidence_uses_original_protocol_files(self):
        for generation in ("HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4"):
            with self.subTest(generation=generation):
                config = self.make_config(generation=generation, capacity_bytes=16 * GIB)
                self.assertTrue(config.metadata["jedec_reference_file"].endswith(".pdf"))
                self.assertEqual(len(config.metadata["source_documents"]), 1)
                source = config.metadata["source_documents"][0]
                self.assertEqual(source["document_author"], "JEDEC")
                self.assertNotIn("studylib", source["source"])
                self.assertEqual(config.metadata["channel_operation_mode"], "pseudo_channel")
                mode_basis = config.metadata["channel_operation_mode_basis"]
                if generation in ("HBM2", "HBM2E"):
                    self.assertIn("PC-only", mode_basis)
                    self.assertIn("not a runtime", mode_basis)
                else:
                    self.assertIn("no alternative Legacy mode", mode_basis)

    def test_complete_payload_records_geometry_and_unknown_timing_separately(self):
        payload = make_hbm_geometry_config(capacity_bytes=36 * GIB,
                    read_bandwidth_gbps=9600, write_bandwidth_gbps=8000)
        self.assertEqual(set(payload), {item.name for item in fields(DramConfig)})
        provenance = payload["metadata"]["parameter_provenance"]
        self.assertTrue((set(payload) - {"metadata"}).issubset(provenance))
        self.assertEqual(payload["metadata"]["geometry_status"], "standard_organization")
        self.assertEqual(provenance["read_latency_ns"]["status"], "selected_reference")
        self.assertEqual(provenance["column_same_group_ns"]["status"], "standard_minimum")
        self.assertIsNone(payload["effective_pin_data_rate_gbps"])
        self.assertIn("array_timing_ns", payload["metadata"]["unknown_parameters"])

    def test_unsupported_or_inconsistent_inputs_fail_without_synthetic_fallback(self):
        bad = (
            {"generation": "HBM"},
            {"channels": 8}, {"stacks": 0}, {"stacks": True},
            {"capacity_bytes": 64 * GIB + 32},
            {"capacity_bytes": 36 * GIB + 1},
            {"die_density_gbits": 24, "dies_per_stack": 8},
            {"die_density_gbits": 12}, {"dies_per_stack": 6},
            {"generation": "HBM2E", "capacity_bytes": 24 * GIB},
            {"generation": "HBM2E", "die_density_gbits": 8},
            {"generation": "HBM2E", "dies_per_stack": 12},
            {"generation": "HBM4", "die_density_gbits": 16},
            {"generation": "HBM4", "channels": 16},
            {"generation": "HBM4", "capacity_bytes": 64 * GIB + 32},
            {"read_bandwidth_gbps": float("nan")}, {"write_bandwidth_gbps": 0},
        )
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_config(**overrides)


if __name__ == "__main__":
    unittest.main()
