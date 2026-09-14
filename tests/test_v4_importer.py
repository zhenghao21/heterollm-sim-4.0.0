from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import heterollm_sim
from heterollm_sim import schema_v4
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.compiler_ir import compile_canonical_scenario
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.contracts import SIMULATION_SCHEMA_VERSION
from heterollm_sim.ir import SCHEMA_VERSION
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v1 import SCHEMA_V1_VERSION
from heterollm_sim.schema_v4 import (
    AUTHORING_SCHEMA_VERSION,
    CONTROL_PLANE_FINGERPRINT_SCHEMA,
    ControllerProfile,
    GPUControllerProfile,
    LEGACY_MAPPING_FINGERPRINT_SCHEMA,
    _migrate_v3_scenario_payload,
    _import_v3_scenario_file,
    runtime_profile_from_dict,
)
from heterollm_sim.serde import read_json, to_primitive, write_json


def _v4_payload(*, include_runtime=False):
    scenario = build_reference_scenario()
    profiles = {
        "components": to_primitive(scenario.component_profiles),
        "host_orchestration": to_primitive(
            scenario.host_orchestration_profile
        ),
        "fusion": to_primitive(scenario.fusion_policy),
        "cim_interconnect": to_primitive(scenario.cim_interconnect),
    }
    if include_runtime:
        profiles["runtime"] = to_primitive(scenario.runtime_profile)
    return {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "name": scenario.name,
        "hardware": to_primitive(scenario.hardware),
        "model": to_primitive(scenario.model),
        "placement": to_primitive(scenario.placement),
        "workload": to_primitive(scenario.workload),
        "profiles": profiles,
        "weights_resident": scenario.weights_resident,
        "assumptions": list(scenario.assumptions),
    }


def _replace_authoring_version(value, source, target):
    if isinstance(value, dict):
        return {
            key: (
                target
                if key == "schema_version" and item == source
                else _replace_authoring_version(item, source, target)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_authoring_version(item, source, target) for item in value]
    return value


def _v3_payload(*, auto_mapping=None):
    payload = _replace_authoring_version(
        _v4_payload(), AUTHORING_SCHEMA_VERSION, "3.0.0"
    )
    metadata = payload["placement"].setdefault("metadata", {})
    metadata["independent_v1_artifact"] = {
        "schema_version": "heterollm.weight-projections/v1",
        "value": 7,
    }
    if auto_mapping is not None:
        metadata["auto_mapping"] = deepcopy(auto_mapping)
    return payload


class V4VersionBoundaryTests(unittest.TestCase):
    def test_product_authoring_and_simulation_are_v4_but_canonical_stays_v11(self):
        self.assertEqual(heterollm_sim.__version__, "4.0.0")
        self.assertEqual(AUTHORING_SCHEMA_VERSION, "4.0.0")
        self.assertEqual(SCHEMA_VERSION, "4.0.0")
        self.assertEqual(SIMULATION_SCHEMA_VERSION, "4.0.0")
        self.assertEqual(SCHEMA_V1_VERSION, "1.1")

    def test_v3_importer_is_not_a_public_runtime_api(self):
        public_names = (
            "migrate_v3_scenario_payload",
            "import_v3_scenario_payload",
            "import_v3_scenario_file",
            "import_v3_to_v4",
        )
        for name in public_names:
            with self.subTest(name=name):
                self.assertNotIn(name, heterollm_sim.__all__)
                self.assertFalse(hasattr(heterollm_sim, name))
                self.assertNotIn(name, schema_v4.__all__)
        self.assertFalse(hasattr(schema_v4, "import_v3_to_v4"))

    def test_normal_parser_accepts_only_v4_and_never_imports_v3(self):
        parsed = scenario_from_dict(_v4_payload())
        self.assertEqual(parsed.schema_version, "4.0.0")
        with self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
            scenario_from_dict(_v3_payload())

        manually_reversioned = _v4_payload()
        manually_reversioned["placement"]["metadata"]["auto_mapping"] = {
            "fingerprint_schema": LEGACY_MAPPING_FINGERPRINT_SCHEMA
        }
        with self.assertRaisesRegex(ValueError, "explicit V3-to-V4 importer"):
            scenario_from_dict(manually_reversioned)

    def test_v4_authoring_rejects_manual_placement_and_locks(self):
        for field, value in (
            ("op_to_component", {"dense0.mlp": "gpu0"}),
            ("tensor_to_component", {"model_weights": "hbm0"}),
            ("tensor_bytes", {"model_weights": 1}),
        ):
            payload = _v4_payload()
            payload["placement"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "manual placement fields"
            ):
                scenario_from_dict(payload)

        payload = _v4_payload()
        payload["placement"]["metadata"]["control_plane"] = {
            "policy": {"locked_op_keys": []},
            "decision": {},
            "evidence": {},
        }
        with self.assertRaisesRegex(ValueError, "manual control-plane locks"):
            scenario_from_dict(payload)

    def test_runtime_control_plane_can_materialize_internal_placement(self):
        authoring = scenario_from_dict(_v4_payload())
        self.assertFalse(authoring.placement.op_to_component)
        result = plan_runtime_placement(authoring)
        mapped = result.apply(authoring)
        self.assertTrue(mapped.placement.op_to_component)
        self.assertIn("control_plane", mapped.placement.metadata)

        runtime_payload = _v4_payload()
        runtime_payload["placement"] = to_primitive(mapped.placement)
        with self.assertRaisesRegex(ValueError, "manual placement fields"):
            scenario_from_dict(runtime_payload)


class V4RuntimeProfileTests(unittest.TestCase):
    def test_missing_runtime_profile_gets_architecture_default(self):
        parsed = scenario_from_dict(_v4_payload())
        self.assertIsInstance(parsed.runtime_profile, ControllerProfile)
        self.assertEqual(
            set(parsed.runtime_profile.gpu_controllers), {"gpu0"}
        )
        self.assertIsInstance(
            parsed.runtime_profile.gpu_controllers["gpu0"],
            GPUControllerProfile,
        )
        self.assertGreater(
            parsed.runtime_profile.pcie_dma_iommu.aggregate_pcie_bandwidth_gb_s,
            0,
        )

    def test_explicit_runtime_profile_round_trips_and_is_strict(self):
        payload = _v4_payload(include_runtime=True)
        expected = runtime_profile_from_dict(payload["profiles"]["runtime"])
        parsed = scenario_from_dict(payload)
        self.assertEqual(parsed.runtime_profile, expected)
        self.assertEqual(to_primitive(parsed.runtime_profile), payload["profiles"]["runtime"])

        invalid = deepcopy(payload)
        invalid["profiles"]["runtime"]["pcie_dma_iommu"][
            "pcie_lane_count"
        ] = 0
        with self.assertRaisesRegex(ValueError, "pcie_lane_count"):
            scenario_from_dict(invalid)

        unknown = deepcopy(payload)
        unknown["profiles"]["runtime"]["mystery_controller"] = {}
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            scenario_from_dict(unknown)


class OfflineV3ImporterTests(unittest.TestCase):
    def test_importer_emits_explicit_runtime_and_converts_old_mapping_metadata(self):
        old_mapping = {
            "fingerprint_algorithm": "sha256",
            "fingerprint_schema": LEGACY_MAPPING_FINGERPRINT_SCHEMA,
            "input_fingerprint": "old-fingerprint",
            "surrogate": "legacy-solver",
            "locked_op_keys": ["dense0.attention"],
            "locked_tensor_ids": ["model_weights"],
            "options": {"objective": "balanced", "time_limit_s": 60.0},
            "operator_execution_targets": {
                "dense0.attention": [
                    {"rank_id": 0, "component_id": "gpu0"}
                ]
            },
            "rank_weight_shards": {
                "embedding_weights": [
                    {
                        "rank_id": 0,
                        "component_id": "hbm0",
                        "logical_bytes": 1,
                        "physical_bytes": 1,
                    }
                ]
            },
            "resident_cim_replicas": {"cim0": 2},
            "derived_tensor_bytes": {"must_not_be_carried": 99},
        }

        source = _v3_payload(auto_mapping=old_mapping)
        source["placement"]["op_to_component"] = {"dense0.mlp": "gpu0"}
        source["placement"]["tensor_to_component"] = {"model_weights": "hbm0"}
        source["placement"]["tensor_bytes"] = {"model_weights": 1}
        migrated = _migrate_v3_scenario_payload(source)
        self.assertEqual(migrated["schema_version"], "4.0.0")
        self.assertEqual(
            migrated["hardware"]["components"][0]["schema_version"],
            "4.0.0",
        )
        self.assertIn("runtime", migrated["profiles"])
        self.assertEqual(
            migrated["profiles"]["runtime"]["schema_version"], "4.0.0"
        )

        metadata = migrated["placement"]["metadata"]
        self.assertNotIn("auto_mapping", metadata)
        control_plane = metadata["control_plane"]
        self.assertEqual(control_plane["policy"], {})
        self.assertEqual(control_plane["decision"], {})
        legacy = control_plane["evidence"]["legacy_v3_auto_mapping"]
        self.assertEqual(legacy["policy"]["locked_op_keys"], ["dense0.attention"])
        self.assertEqual(legacy["policy"]["locked_tensor_ids"], ["model_weights"])
        self.assertIn("operator_execution_targets", legacy["decision"])
        self.assertIn("rank_weight_shards", legacy["decision"])
        self.assertIn("resident_cim_replicas", legacy["decision"])
        self.assertNotIn("derived_tensor_bytes", legacy["decision"])
        self.assertEqual(
            control_plane["evidence"]["legacy_v3_explicit_placement"],
            {
                "op_to_component": {"dense0.mlp": "gpu0"},
                "tensor_to_component": {"model_weights": "hbm0"},
                "tensor_bytes": {"model_weights": 1},
            },
        )
        self.assertNotIn("input_fingerprint", control_plane["evidence"])
        self.assertEqual(control_plane["evidence"]["status"], "migrated")
        self.assertEqual(
            control_plane["evidence"]["migrated_from"],
            {
                "schema_version": "3.0.0",
                "artifact": "placement.metadata.auto_mapping",
                "fingerprint_schema": LEGACY_MAPPING_FINGERPRINT_SCHEMA,
            },
        )
        self.assertNotEqual(
            control_plane["evidence"].get("fingerprint_schema"),
            LEGACY_MAPPING_FINGERPRINT_SCHEMA,
        )
        self.assertEqual(migrated["placement"]["op_to_component"], {})
        self.assertEqual(migrated["placement"]["tensor_to_component"], {})
        self.assertEqual(migrated["placement"]["tensor_bytes"], {})
        self.assertEqual(
            metadata["independent_v1_artifact"]["schema_version"],
            "heterollm.weight-projections/v1",
        )
        self.assertEqual(
            CONTROL_PLANE_FINGERPRINT_SCHEMA, "runtime-control-plane-v4"
        )

    def test_importer_does_not_rewrite_third_party_metadata_versions(self):
        payload = _v3_payload()
        payload["placement"].setdefault("metadata", {})["third_party"] = {
            "schema_version": "3.0.0",
            "value": 7,
        }

        migrated = _migrate_v3_scenario_payload(payload)

        self.assertEqual(
            migrated["placement"]["metadata"]["third_party"],
            {"schema_version": "3.0.0", "value": 7},
        )
        scenario_from_dict(migrated)

    def test_importer_without_old_mapping_still_emits_explicit_runtime(self):
        migrated = _migrate_v3_scenario_payload(_v3_payload())
        self.assertIn("runtime", migrated["profiles"])
        self.assertNotIn("control_plane", migrated["placement"]["metadata"])
        canonical = compile_canonical_scenario(scenario_from_dict(migrated))
        self.assertEqual(canonical.schema_version, "1.1")
        self.assertEqual(canonical.attributes["source_schema_version"], "4.0.0")

    def test_importer_fails_closed_on_wrong_source_or_unknown_fingerprint(self):
        with self.assertRaisesRegex(ValueError, "requires scenario.schema_version"):
            _migrate_v3_scenario_payload(_v4_payload())

        bad = _v3_payload(
            auto_mapping={"fingerprint_schema": "unknown-mapping-schema"}
        )
        with self.assertRaisesRegex(ValueError, "unsupported V3"):
            _migrate_v3_scenario_payload(bad)

        conflict = _v3_payload(auto_mapping={"locked_op_keys": []})
        conflict["placement"]["metadata"]["control_plane"] = {}
        with self.assertRaisesRegex(ValueError, "both auto_mapping and control_plane"):
            _migrate_v3_scenario_payload(conflict)

    def test_file_import_is_explicit_validated_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "scenario-v3.json"
            destination = directory / "scenario-v4.json"
            write_json(source, _v3_payload(auto_mapping={"locked_op_keys": []}))

            self.assertEqual(
                _import_v3_scenario_file(source, destination), destination
            )
            scenario_from_dict(read_json(destination))
            with self.assertRaises(FileExistsError):
                _import_v3_scenario_file(source, destination)
            self.assertEqual(
                _import_v3_scenario_file(
                    source, destination, overwrite=True
                ),
                destination,
            )


if __name__ == "__main__":
    unittest.main()
