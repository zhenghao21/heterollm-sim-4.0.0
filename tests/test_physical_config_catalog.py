from copy import deepcopy
from dataclasses import asdict, fields
import unittest

from heterollm_sim.memory_types import (
    DRAM_BANK_GROUP_TIMING_FIELDS, DramConfig, NandConfig, make_hbm_config,
)
from heterollm_sim.physical_contract import (
    parse_execution_physical_config, reconfigure_physical_config,
)
from heterollm_sim.physical_presets import (
    _PHYSICAL_CONFIG_CATALOG, physical_config_profile_ids,
    resolve_physical_config,
)


class PhysicalConfigCatalogTests(unittest.TestCase):
    def test_every_catalog_entry_is_complete_and_strictly_loadable(self):
        for profile_id in physical_config_profile_ids():
            with self.subTest(profile_id=profile_id):
                config = resolve_physical_config(profile_id)
                self.assertIsInstance(config, (DramConfig, NandConfig))
                self.assertEqual(set(_PHYSICAL_CONFIG_CATALOG[profile_id]),
                                 {field.name for field in fields(config)})
                self.assertEqual(asdict(parse_execution_physical_config(asdict(config))), asdict(config))
                self.assertEqual(config.metadata["physical_config_profile_id"], profile_id)

    def test_analytical_profile_has_explicit_study_physical_parameters(self):
        analytical = asdict(resolve_physical_config("hbm_analytical_zero"))
        expected = asdict(make_hbm_config(rows_per_bank=524288, burst_interval_ns=0.0))
        analytical.pop("metadata")
        expected.pop("metadata")
        self.assertEqual(analytical, expected)

    def test_geometry_control_only_changes_bank_partition(self):
        analytical = asdict(resolve_physical_config("hbm_analytical_zero"))
        control = asdict(resolve_physical_config("hbm_group_geometry_control"))
        for field in ("metadata", "bank_groups_per_rank", "banks_per_group"):
            analytical.pop(field)
            control.pop(field)
        self.assertEqual(analytical, control)

    def test_obsolete_profile_names_are_not_resolved(self):
        for profile_id in ("hbm_legacy_zero", "hbm_group_geometry_zero",
                           "hbm3e_reference_flat_zero", "hbm4_reference_flat_zero",
                           "published_hbm", "published_hbm_serving"):
            with self.subTest(profile_id=profile_id), self.assertRaisesRegex(ValueError, "unknown physical config profile"):
                resolve_physical_config(profile_id)

    def test_nand_execution_requires_explicit_directional_bandwidths(self):
        for name in ("host_read_bandwidth_gb_s", "host_write_bandwidth_gb_s"):
            raw = asdict(resolve_physical_config("hbf_synthetic"))
            raw.pop(name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                parse_execution_physical_config(raw)

    def test_jedec_hbm_rejects_analytical_timing_overrides(self):
        original = resolve_physical_config("hbm3e_reference")
        for name in ("open_ns", "close_ns", "read_recovery_ns", "write_recovery_ns", "burst_interval_ns"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "command-anchored"):
                reconfigure_physical_config(original, overrides={name: 20.0})

    def test_synthetic_profile_only_adds_declared_group_timings(self):
        control = resolve_physical_config("hbm_group_geometry_control")
        synthetic = resolve_physical_config("hbm_synthetic_bank_groups")
        self.assertEqual(control.bank_count, 8)
        self.assertEqual(synthetic.bank_count, 8)
        self.assertFalse(control.bank_group_timing_enabled)
        self.assertEqual(tuple(getattr(synthetic, name) for name in DRAM_BANK_GROUP_TIMING_FIELDS),
                         (4.0, 1.0, 8.0, 2.0, 16.0, 4.0, 2.0))
        self.assertIn("not JEDEC", synthetic.metadata["timing_basis"])
        self.assertEqual(len(synthetic.metadata["bank_group_timing_assumptions"]), 4)
        a, b = asdict(control), asdict(synthetic)
        for field in ("metadata", *DRAM_BANK_GROUP_TIMING_FIELDS):
            a.pop(field)
            b.pop(field)
        self.assertEqual(a, b)

    def test_nonideal_micropresets_use_one_standard_stack_and_retain_controls(self):
        bandwidths = {"research_hbm_acceptance": 488.0, "research_hbm_sessions": 512.0,
                      "research_hbm_relay": 256.0, "research_hbm_package": 256.0,
                      "research_hbm_path_relay": 128.0}
        for profile_id, bandwidth in bandwidths.items():
            with self.subTest(profile_id=profile_id):
                config = resolve_physical_config(profile_id)
                self.assertEqual((config.generation, config.stacks, config.channels,
                                  config.pseudo_channels_per_channel, config.lane_count),
                                 ("HBM3E", 1, 16, 2, 32))
                self.assertEqual((config.row_bytes, config.burst_bytes, config.rows_per_bank,
                                  config.bank_groups_per_rank, config.banks_per_group),
                                 (1024, 32, 8192, 4, 4))
                self.assertEqual(config.metadata["geometry_status"], "standard_organization")
                self.assertEqual(config.metadata["reference_profile_id"], "hbm3e_reference")
                self.assertFalse(config.metadata["vendor_verified"])
                self.assertGreater(config.computed_capacity_bytes, config.capacity_bytes)
                self.assertEqual(config.physical_interface_bandwidth_gb_s, bandwidth)
                self.assertEqual(config.directional_bandwidth_gb_s("read"), bandwidth)
                self.assertEqual(config.directional_bandwidth_gb_s("write"), bandwidth)
                self.assertEqual(config.control_mode, "jedec_hbm")
                self.assertAlmostEqual(config.command_clock_ns, 512 / bandwidth)
                self.assertEqual((config.read_latency_ns / config.command_clock_ns,
                                  config.write_latency_ns / config.command_clock_ns,
                                  config.burst_interval_ns / config.command_clock_ns,
                                  config.max_outstanding_requests, config.max_expanded_segments),
                                 (32, 8, 2, 64, 1000000))
                self.assertGreaterEqual(config.column_same_group_ns, 4 * config.command_clock_ns)
                self.assertEqual(config.metadata["parameter_provenance"]["burst_interval_ns"]["status"],
                                 "derived_from_input")

    def test_micro_capacity_selects_a_covering_standard_part_without_virtual_rows(self):
        config = resolve_physical_config("research_hbm_sessions",
            overrides={"capacity_bytes": 4 * 1024 ** 3 + 32}, expand_geometry=True)
        self.assertEqual(config.capacity_bytes, 4 * 1024 ** 3 + 32)
        self.assertEqual(config.computed_capacity_bytes, 8 * 1024 ** 3)
        self.assertEqual(config.stacks, 1)
        self.assertEqual(config.rows_per_bank, 8192)
        self.assertEqual(config.metadata["geometry_status"], "standard_organization")
        with self.assertRaises(ValueError):
            resolve_physical_config("research_hbm_sessions",
                overrides={"capacity_bytes": 64 * 1024 ** 3 + 32}, expand_geometry=True)

    def test_micro_profiles_share_standard_height_and_stack_dependencies(self):
        base = resolve_physical_config("research_hbm_package")
        taller = resolve_physical_config("research_hbm_package", overrides={"dies_per_stack": 8})
        self.assertEqual((base.dies_per_stack, taller.dies_per_stack), (4, 8))
        self.assertEqual(taller.bank_count, 32)
        self.assertEqual(taller.bank_groups_per_rank, 8)
        self.assertEqual(taller.rows_per_bank, base.rows_per_bank)
        self.assertEqual(taller.refresh_cycle_ns, 260.)
        self.assertEqual(taller.metadata["geometry_status"], "standard_organization")
        stacked = resolve_physical_config("research_hbm_package", overrides={"stacks": 2})
        self.assertEqual(stacked.stacks, 2)
        self.assertEqual(stacked.lane_count, 2 * base.lane_count)
        self.assertEqual(stacked.computed_capacity_bytes, 2 * base.computed_capacity_bytes)
        self.assertEqual(stacked.command_clock_ns, 2 * base.command_clock_ns)
        self.assertEqual(stacked.capacity_bytes, base.capacity_bytes)

    def test_micro_profiles_reject_invalid_protocol_geometry(self):
        for name, value in (("row_bytes", 8192), ("burst_bytes", 64), ("data_lanes", 4),
                            ("pseudo_channels_per_channel", 1)):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    resolve_physical_config("research_hbm_package", overrides={name: value})

    def test_micro_timing_bandwidth_and_queue_axes_remain_configurable(self):
        config = resolve_physical_config("research_hbm_relay", overrides={
            "capacity_bytes": 16384, "interface_bandwidth_gb_s": 256.0,
            "read_bandwidth_gb_s": 128.0, "write_bandwidth_gb_s": 64.0,
            "read_latency_ns": 10.0, "max_outstanding_requests": 7})
        self.assertEqual(config.capacity_bytes, 16384)
        self.assertEqual((config.directional_bandwidth_gb_s("read"),
                          config.directional_bandwidth_gb_s("write")), (128.0, 64.0))
        self.assertEqual((config.read_latency_ns, config.max_outstanding_requests), (10.0, 7))
        self.assertEqual(config.metadata["geometry_status"], "standard_organization")
        self.assertEqual(config.metadata["parameter_provenance"]["read_latency_ns"]["source"],
                         "physical_config_override")

    def test_catalog_and_caller_metadata_are_isolated(self):
        overrides = {"metadata": {"nested": ["authored"]}}
        a = resolve_physical_config("hbf_synthetic", overrides=overrides)
        a.metadata["nested"].append("changed")
        self.assertEqual(overrides["metadata"]["nested"], ["authored"])
        self.assertNotIn("nested", resolve_physical_config("hbf_synthetic").metadata)
        self.assertEqual(a.metadata["physical_config_override_fields"], ["metadata"])

    def test_unknown_or_malformed_updates_are_rejected(self):
        for profile_id, overrides in (("missing", {}), ("hbm_analytical_zero", {"row_byets": 128}),
                                      ("hbm_analytical_zero", {"kind": "HBF"}),
                                      ("hbm_analytical_zero", {"column_same_group_ns": -1}),
                                      ("hbm_analytical_zero", {"metadata": None})):
            with self.subTest(profile_id=profile_id, overrides=overrides):
                with self.assertRaises(ValueError):
                    resolve_physical_config(profile_id, overrides=overrides)

    def test_reconfiguration_does_not_accept_incomplete_execution_mapping(self):
        with self.assertRaisesRegex(ValueError, "missing explicit fields"):
            reconfigure_physical_config({"kind": "HBM"}, overrides={"capacity_bytes": 4096})

    def test_dram_capacity_growth_is_explicit_and_does_not_round_capacity(self):
        base = resolve_physical_config("hbm_analytical_zero")
        capacity = base.capacity_bytes + 64
        with self.assertRaisesRegex(ValueError, "physical geometry"):
            reconfigure_physical_config(base, overrides={"capacity_bytes": capacity})
        grown = reconfigure_physical_config(base, overrides={"capacity_bytes": capacity}, expand_geometry=True)
        self.assertEqual(grown.capacity_bytes, capacity)
        self.assertEqual(grown.rows_per_bank, base.rows_per_bank + 1)
        self.assertEqual(base.rows_per_bank, 524288)
        with self.assertRaisesRegex(ValueError, "multiple of burst_bytes"):
            reconfigure_physical_config(base, overrides={"capacity_bytes": capacity + 1}, expand_geometry=True)

    def test_geometry_growth_respects_stacks_dies_and_explicit_data_lanes(self):
        base = resolve_physical_config("research_hbm_ideal")
        grown = reconfigure_physical_config(base, overrides={
            "capacity_bytes": 8 * 4096, "stacks": 2, "dies_per_stack": 2,
            "data_lanes": 2, "rows_per_bank": 1}, expand_geometry=True)
        self.assertEqual(grown.rows_per_bank, 1)
        self.assertEqual(grown.computed_capacity_bytes, grown.capacity_bytes)

    def test_geometry_growth_retains_valid_interleave(self):
        base = resolve_physical_config("research_hbm_ideal")
        grown = reconfigure_physical_config(base, overrides={"rows_per_bank": 1,
            "capacity_bytes": 5 * 4096, "interleave_bytes": 2 * 4096}, expand_geometry=True)
        self.assertEqual(grown.rows_per_bank, 6)

    def test_nand_geometry_growth_uses_all_physical_dimensions(self):
        base = resolve_physical_config("research_hbf_wear_trace")
        capacity = base.computed_capacity_bytes + base.page_bytes
        with self.assertRaisesRegex(ValueError, "physical geometry"):
            reconfigure_physical_config(base, overrides={"capacity_bytes": capacity})
        grown = reconfigure_physical_config(base, overrides={"capacity_bytes": capacity}, expand_geometry=True)
        self.assertEqual(grown.blocks_per_plane, base.blocks_per_plane + 1)
        self.assertEqual(grown.capacity_bytes, capacity)
        self.assertEqual(grown.parallel_units, base.parallel_units)

    def test_published_hbf_parallelism_is_fixed_not_fitted_to_interface(self):
        config = resolve_physical_config("published_hbf", overrides={
            "capacity_bytes": 128 * 1024 ** 2, "host_bandwidth_gb_s": 100.0,
            "host_read_bandwidth_gb_s": 100.0, "host_write_bandwidth_gb_s": 10.0,
            "page_read_ns": 4000.0, "page_bytes": 4096})
        self.assertEqual(config.parallel_units, 128)
        self.assertEqual(config.channels, 128)
        self.assertEqual(config.blocks_per_plane, 1)
        self.assertEqual(config.host_granularity_bytes, 64)
        self.assertEqual(config.internal_bandwidth_gb_s, 100000.0)
        self.assertEqual(config.front_ns, 0.0)

    def test_explicit_published_hbf_experiment_variables_take_precedence(self):
        config = resolve_physical_config("published_hbf", overrides={
            "parallel_units": 8, "capacity_bytes": 8 * 1024 ** 2,
            "page_bytes": 64, "internal_bandwidth_gb_s": 900.0})
        self.assertEqual(config.channels, 8)
        self.assertEqual(config.blocks_per_plane, 64)
        self.assertEqual(config.host_granularity_bytes, 64)
        self.assertEqual(config.internal_bandwidth_gb_s, 900.0)

    def test_bad_derived_inputs_raise_validation_errors(self):
        for updates in ({"page_bytes": 0}, {"page_bytes": True}, {"page_read_ns": float("inf")},
                        {"page_read_ns": "4000"}, {"parallel_units": 0}, {"parallel_units": True},
                        {"host_read_bandwidth_gb_s": float("inf")}):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    resolve_physical_config("published_hbf", overrides=updates)

    def test_reconfigure_metadata_preserves_original_without_aliasing(self):
        original = resolve_physical_config("hbf_synthetic", overrides={"metadata": {"a": [1]}})
        before = deepcopy(asdict(original))
        updated = reconfigure_physical_config(original, overrides={"metadata": {"b": [2]}})
        updated.metadata["a"].append(3)
        self.assertEqual(asdict(original), before)
        self.assertEqual(updated.metadata["b"], [2])


if __name__ == "__main__":
    unittest.main()
