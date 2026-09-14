import json
import unittest
from dataclasses import replace
from unittest import mock

from heterollm_sim import architecture_scan
from heterollm_sim.architecture_scan import (
    BatchedGemmCandidate,
    build_batched_gemm_candidates,
    scan_architecture_candidates,
)
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.ir import HardwareSpec
from heterollm_sim.reference import build_reference_scenario


class ArchitectureScanTests(unittest.TestCase):
    def test_reference_scan_is_json_safe_and_explicitly_analytical(self):
        result = scan_architecture_candidates(
            build_reference_scenario(), backend="numpy", top_n=7
        )

        self.assertEqual(result["analysis_kind"], "analytical_architecture_scan")
        self.assertFalse(result["is_event_simulation"])
        self.assertEqual(result["backend_requested"], "numpy")
        self.assertEqual(result["backend_used"], "numpy")
        self.assertEqual(result["counts"]["components_total"], 12)
        self.assertEqual(result["counts"]["components_eligible"], 2)
        self.assertEqual(result["counts"]["operators"], 8)
        self.assertEqual(result["counts"]["ranks"], 1)
        self.assertEqual(result["counts"]["candidates"], 16)
        self.assertEqual(result["counts"]["top_results"], 7)
        self.assertTrue(any("不是有序事件仿真" in item for item in result["diagnostics"]))
        self.assertTrue(any("NumPy" in item for item in result["diagnostics"]))
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_candidates_carry_real_rank_placement_and_route_information(self):
        authored = build_reference_scenario()
        scenario = plan_runtime_placement(authored).apply(authored)
        candidates = build_batched_gemm_candidates(scenario)

        self.assertTrue(candidates)
        self.assertTrue(all(isinstance(item, BatchedGemmCandidate) for item in candidates))
        self.assertEqual(
            tuple(item.candidate_id for item in candidates),
            tuple(sorted(item.candidate_id for item in candidates)),
        )
        cim = next(
            item
            for item in candidates
            if item.operator_id == "dense0.mlp_up_gate"
            and item.component_id == "cim0"
        )
        gpu = next(
            item
            for item in candidates
            if item.operator_id == "dense0.mlp_up_gate"
            and item.component_id == "gpu0"
        )
        self.assertEqual(cim.logical_rank, 0)
        self.assertEqual(cim.rank_component_id, "gpu0")
        self.assertEqual(cim.rank_memory_component_id, "hbm0")
        self.assertEqual(cim.rank_cim_component_id, "cim0")
        self.assertEqual(cim.current_placement_component_id, "cim0")
        self.assertTrue(cim.is_current_placement)
        self.assertGreater(cim.communication_bytes, 0)
        self.assertEqual(cim.communication_path, ("gpu-cim", "gpu-cim"))
        self.assertEqual(gpu.communication_bytes, 0)
        self.assertFalse(gpu.is_current_placement)

    def test_one_vectorized_backend_call_and_stable_top_n(self):
        scenario = build_reference_scenario()
        real = architecture_scan.evaluate_batched_gemm
        with mock.patch.object(
            architecture_scan, "evaluate_batched_gemm", wraps=real
        ) as evaluate:
            first = scan_architecture_candidates(
                scenario, backend="numpy", top_n=4
            )
        second = scan_architecture_candidates(
            scenario, backend="numpy", top_n=4
        )

        evaluate.assert_called_once()
        submitted = evaluate.call_args.args[0]
        self.assertEqual(len(submitted.candidate_ids), first["counts"]["candidates"])
        self.assertEqual(first["top_results"], second["top_results"])
        self.assertEqual(
            [item["position"] for item in first["top_results"]], [1, 2, 3, 4]
        )

    def test_missing_capabilities_are_skipped_without_inventing_values(self):
        scenario = build_reference_scenario()
        components = tuple(
            replace(component, peak_ops_per_s=0.0)
            if component.component_id == "gpu0"
            else component
            for component in scenario.hardware.components
        )
        scenario = replace(
            scenario,
            hardware=HardwareSpec(
                name=scenario.hardware.name,
                components=components,
                links=scenario.hardware.links,
                require_connected=scenario.hardware.require_connected,
                metadata=scenario.hardware.metadata,
                schema_version=scenario.hardware.schema_version,
            ),
        )

        result = scan_architecture_candidates(
            scenario, backend="numpy", top_n=3
        )

        self.assertEqual(result["backend_used"], "numpy")
        self.assertEqual(result["counts"]["components_eligible"], 1)
        self.assertGreater(result["counts"]["candidates"], 0)
        text = "\n".join(result["diagnostics"])
        self.assertIn("gpu0", text)
        self.assertIn("不能虚构", text)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_candidates_resolve_profiles_from_their_actual_components(self):
        scenario = build_reference_scenario()
        gpu_profile = scenario.resolve_component_profile("gpu0")
        hbm_profile = scenario.resolve_component_profile("hbm0")
        cim_profile = scenario.resolve_component_profile("cim0")
        registries = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        registries["gpu"]["scan-gpu"] = replace(
            gpu_profile, attainable_efficiency=0.25, kernel_launch_ns=321.0
        )
        registries["hbm"]["scan-hbm"] = replace(
            hbm_profile, efficiency=0.25
        )
        registries["cim"]["unused-cim"] = replace(
            cim_profile, peripheral_latency_ns=999.0
        )
        components = tuple(
            replace(component, cost_profile_id="scan-gpu")
            if component.component_id == "gpu0"
            else replace(component, cost_profile_id="scan-hbm")
            if component.component_id == "hbm0"
            else component
            for component in scenario.hardware.components
        )
        scenario = replace(
            scenario,
            hardware=replace(scenario.hardware, components=components),
            component_profiles=registries,
        )

        candidates = build_batched_gemm_candidates(scenario)
        gpu = next(item for item in candidates if item.component_id == "gpu0")
        cim = next(item for item in candidates if item.component_id == "cim0")

        self.assertEqual(gpu.compute_efficiency, 0.25)
        self.assertEqual(gpu.kernel_launch_ns, 321.0)
        self.assertAlmostEqual(
            gpu.memory_efficiency,
            (0.25 + 7.0 * hbm_profile.efficiency) / 8.0,
        )
        self.assertNotEqual(cim.kernel_launch_ns, 999.0)

    def test_invalid_public_options_fail_with_chinese_diagnostics(self):
        scenario = build_reference_scenario()
        with self.assertRaisesRegex(ValueError, "top_n"):
            scan_architecture_candidates(scenario, top_n=0)
        with self.assertRaisesRegex(ValueError, "backend"):
            scan_architecture_candidates(scenario, backend="tpu")


if __name__ == "__main__":
    unittest.main()
