import unittest
from dataclasses import replace
from unittest.mock import patch

import heterollm_sim.planner as planner
from heterollm_sim.reference import build_reference_scenario


class PlannerArtifactWorkloadCacheTests(unittest.TestCase):
    def test_cache_hit_skips_repeated_artifact_label_discovery(self):
        scenario = build_reference_scenario()
        layer = replace(
            planner._execution_layers(scenario)[0],
            quantization="IQ3_S-FFN-IQ4_XS",
        )
        context = planner.CompilationContext(scenario)

        with planner._compilation_scope(scenario, context), patch.object(
            planner,
            "_artifact_label_from_metadata",
            wraps=planner._artifact_label_from_metadata,
        ) as resolve_label:
            first_spec, first = planner._artifact_workload_metadata(
                layer,
                4,
                512,
                1024,
                "dense0.ffn_up",
                explicit_metadata={"op_name": "first-task"},
                scenario=scenario,
            )
            first["caller_owned_probe"] = True
            second_spec, second = planner._artifact_workload_metadata(
                layer,
                4,
                512,
                1024,
                "dense0.ffn_up",
                explicit_metadata={"op_name": "second-task"},
                scenario=scenario,
            )

        self.assertEqual(resolve_label.call_count, 1)
        self.assertEqual(first_spec, second_spec)
        self.assertEqual(first_spec.name, "IQ4_XS")
        self.assertNotIn("caller_owned_probe", second)
        self.assertEqual(
            context.artifact_workload_cache_stats,
            {"hits": 1, "misses": 1, "bypasses": 0, "entries": 1},
        )

    def test_mixed_artifact_operator_kinds_keep_distinct_cache_entries(self):
        scenario = build_reference_scenario()
        layer = replace(
            planner._execution_layers(scenario)[0],
            quantization="IQ3_S-FFN-IQ4_XS",
        )
        context = planner.CompilationContext(scenario)

        with planner._compilation_scope(scenario, context):
            attention_spec, _attention = planner._artifact_workload_metadata(
                layer,
                4,
                512,
                1024,
                "dense0.qkv",
                scenario=scenario,
            )
            ffn_spec, _ffn = planner._artifact_workload_metadata(
                layer,
                4,
                512,
                1024,
                "dense0.ffn_up",
                scenario=scenario,
            )

        self.assertEqual(attention_spec.name, "IQ3_S")
        self.assertEqual(ffn_spec.name, "IQ4_XS")
        self.assertEqual(context.artifact_workload_cache_stats["entries"], 2)

    def test_nested_explicit_metadata_keeps_fail_safe_bypass(self):
        scenario = build_reference_scenario()
        layer = planner._execution_layers(scenario)[0]
        context = planner.CompilationContext(scenario)
        nested = {"contract": {"artifact_quantization": "IQ3_S"}}

        with planner._compilation_scope(scenario, context), patch.object(
            planner,
            "_artifact_label_from_metadata",
            wraps=planner._artifact_label_from_metadata,
        ) as resolve_label:
            for _index in range(2):
                spec, _metadata = planner._artifact_workload_metadata(
                    layer,
                    4,
                    512,
                    1024,
                    "dense0.qkv",
                    explicit_metadata=nested,
                    scenario=scenario,
                )
                self.assertEqual(spec.name, "IQ3_S")

        self.assertEqual(resolve_label.call_count, 2)
        self.assertEqual(
            context.artifact_workload_cache_stats,
            {"hits": 0, "misses": 0, "bypasses": 2, "entries": 0},
        )


if __name__ == "__main__":
    unittest.main()
