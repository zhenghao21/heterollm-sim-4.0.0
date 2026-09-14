import unittest

from heterollm_sim.contracts import (
    EvidenceStatus,
    ResourceDemand,
    RunManifest,
    SIMULATION_SCHEMA_VERSION,
    TaskCategory,
    TaskSpec,
    TraceFidelity,
    VisualizationTraceOptions,
)
from heterollm_sim.serde import canonical_json, stable_hash


class ContractTests(unittest.TestCase):
    def test_negative_demand_is_rejected(self):
        with self.assertRaises(ValueError):
            ResourceDemand("hbm.channel0", -1.0)

    def test_duplicate_resource_in_task_is_rejected(self):
        demand = ResourceDemand("gpu.compute", 10.0)
        with self.assertRaises(ValueError):
            TaskSpec(
                task_id="op",
                request_id="r0",
                name="op",
                category=TaskCategory.COMPUTE,
                demands=(demand, demand),
            )

    def test_non_finite_earliest_start_is_rejected_at_contract_boundary(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "earliest_start_ns must be finite"
            ):
                TaskSpec(
                    task_id="op",
                    request_id="r0",
                    name="op",
                    category=TaskCategory.COMPUTE,
                    earliest_start_ns=value,
                )

    def test_manifest_serialization_and_hash_are_deterministic(self):
        manifest = RunManifest(
            schema_version=SIMULATION_SCHEMA_VERSION,
            run_id="run-0",
            random_seed=7,
            simulator_version="0.1.0",
            model_name="generic-dense",
            hardware_name="gpu-cim",
            workload_name="single-request",
            evidence=EvidenceStatus.ANALYTICAL,
        )
        first = canonical_json(manifest)
        second = canonical_json(manifest)
        self.assertEqual(first, second)
        self.assertEqual(stable_hash(manifest), stable_hash(manifest))
        self.assertIn('"evidence": "analytical"', first)

    def test_manifest_is_v3_only_and_keyword_only(self):
        with self.assertRaisesRegex(TypeError, "positional"):
            RunManifest(SIMULATION_SCHEMA_VERSION)
        with self.assertRaisesRegex(ValueError, "must be exactly 4.0.0"):
            RunManifest(
                schema_version="2.0.0",
                run_id="old-schema",
                random_seed=7,
                simulator_version="3.0.0",
                model_name="model",
                hardware_name="hardware",
                workload_name="workload",
            )

    def test_canonical_json_rejects_non_finite_numbers(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "有限值"):
                    canonical_json({"value": value})

    def test_visualization_trace_contract_bounds_pages_and_fidelity(self):
        options = VisualizationTraceOptions(event_offset=3, event_limit=7)
        self.assertEqual(options.event_offset, 3)
        self.assertEqual(options.event_limit, 7)
        self.assertEqual(TraceFidelity.EXACT.value, "exact")
        self.assertEqual(TraceFidelity.AGGREGATE.value, "aggregate")
        with self.assertRaisesRegex(ValueError, "event_offset"):
            VisualizationTraceOptions(event_offset=-1)
        with self.assertRaisesRegex(ValueError, "event_limit"):
            VisualizationTraceOptions(event_limit=0)


if __name__ == "__main__":
    unittest.main()
