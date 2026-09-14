import json
import unittest
from dataclasses import replace

from heterollm_sim.config import model_from_dict
from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.planner import compile_scenario, validate_scenario
from heterollm_sim.reporting import (
    compare_with_gpu_baseline,
    format_comparison,
    report_dict,
    run_scenario,
)
from heterollm_sim.ir import ParallelSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive


class EndToEndTests(unittest.TestCase):
    def test_reference_scenario_is_valid_and_runs(self):
        scenario = build_reference_scenario()
        validation = validate_scenario(scenario)
        self.assertTrue(validation.is_valid, validation.errors)
        self.assertEqual(len(scenario.hardware.components), 12)
        self.assertEqual(len(scenario.hardware.links), 11)

        result = run_scenario(scenario)
        request = result.metrics.request_metrics["request-0000"]

        self.assertGreater(result.trace.makespan_ns, 0)
        self.assertEqual(request.visible_output_tokens, 4)
        self.assertGreater(request.ttft_ns, 0)
        self.assertIsNotNone(request.tpot_ns)

        payload = report_dict(result)
        json.dumps(payload)
        self.assertEqual(payload["execution_mode"], "continuous_batching")
        self.assertEqual(payload["manifest"]["evidence"], "analytical")
        self.assertIn("cim", payload["category_time_ns"])
        self.assertTrue(
            any(resource_id.startswith("cim0.") for resource_id in payload["resource_utilization"])
        )
        self.assertIn("scheduler", payload)
        self.assertIn("kv_cache", payload)
        self.assertIn("mtp", payload["summary"])
        self.assertGreater(payload["summary"]["mtp"]["proposed_tokens"], 0)

    def test_schedule_and_trace_are_deterministic(self):
        scenario = build_reference_scenario()
        first_schedule = compile_scenario(scenario)
        second_schedule = compile_scenario(scenario)
        self.assertEqual(first_schedule.manifest.run_id, second_schedule.manifest.run_id)
        self.assertEqual(first_schedule.tasks, second_schedule.tasks)

        first = simulate_schedule(first_schedule)
        second = simulate_schedule(second_schedule)
        first_rows = [(task.task_id, task.start_ns, task.end_ns) for task in first.tasks]
        second_rows = [(task.task_id, task.start_ns, task.end_ns) for task in second.tasks]
        self.assertEqual(first_rows, second_rows)

    def test_cold_cim_inserts_weight_loads_and_is_slower(self):
        base = build_reference_scenario()
        hardware = replace(
            base.hardware,
            components=tuple(
                replace(component, capacity_bytes=1 << 20)
                if component.component_id == "hbm0"
                else component
                for component in base.hardware.components
            ),
        )
        warm_authoring = replace(base, hardware=hardware)
        warm_mapping = plan_runtime_placement(warm_authoring)
        self.assertTrue(warm_mapping.fully_placed, warm_mapping.unplaced)
        warm = warm_mapping.apply(warm_authoring)

        cold_authoring = replace(
            warm_authoring,
            name="cold-reference",
            weights_resident=False,
        )
        cold_mapping = plan_runtime_placement(
            cold_authoring,
            PlacementPolicy(allow_cold_cim_streaming=True),
        )
        self.assertTrue(cold_mapping.fully_placed, cold_mapping.unplaced)
        cold = cold_mapping.apply(cold_authoring)
        warm_schedule = compile_scenario(warm)
        cold_schedule = compile_scenario(cold)
        warm_trace = simulate_schedule(warm_schedule)
        cold_trace = simulate_schedule(cold_schedule)

        self.assertFalse(any("weight_load" in task.name for task in warm_schedule.tasks))
        self.assertTrue(any("weight_load" in task.name for task in cold_schedule.tasks))
        self.assertGreater(cold_trace.makespan_ns, warm_trace.makespan_ns)

    def test_tensor_parallel_requires_enough_compute_components(self):
        scenario = build_reference_scenario()
        placement = replace(scenario.placement, parallel=ParallelSpec(tp_degree=2))
        unsupported = replace(scenario, placement=placement)
        report = validate_scenario(unsupported)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("并行域需要 2 个计算组件" in error for error in report.errors)
        )

    def test_unsupported_graph_precision_is_rejected_before_lowering(self):
        scenario = build_reference_scenario()
        source = to_primitive(scenario.model)
        model_payload = {
            key: source[key]
            for key in (
                "schema_version",
                "name",
                "vocabulary_size",
                "max_sequence_length",
                "embedding_weight_bytes",
                "text_backbone_only",
                "supported_modalities",
                "excluded_subgraphs",
                "architecture",
                "metadata",
                "graph",
            )
            if key in source
        }
        group = next(
            item
            for item in model_payload["graph"]["operators"]
            if item["op_kind"] == "layer_group"
        )
        group["parameters"]["dtype"] = "mystery"
        parent_id = group["operator_id"]
        feed_forward = next(
            item
            for item in model_payload["graph"]["operators"]
            if item.get("attributes", {}).get("parent_group_id") == parent_id
            and item["op_kind"] in {"dense_mlp", "moe_router", "moe_experts"}
        )
        feed_forward["parameters"]["quantization"] = None

        with self.assertRaisesRegex(ValueError, "无损覆盖.*dtype|dtype.*静默降级"):
            model_from_dict(model_payload)

    def test_reference_cim_mapping_can_be_compared_with_gpu_baseline(self):
        scenario = build_reference_scenario()
        comparison = compare_with_gpu_baseline(scenario)
        self.assertGreater(comparison["comparison"]["latency_speedup"], 1.0)
        self.assertEqual(comparison["candidate"]["execution_mode"], "continuous_batching")
        self.assertGreater(comparison["candidate"]["summary"]["task_count"], 0)
        self.assertGreater(comparison["candidate"]["summary"]["mtp"]["proposed_tokens"], 0)
        self.assertNotEqual(
            comparison["candidate"]["manifest"]["run_id"],
            comparison["gpu_baseline"]["manifest"]["run_id"],
        )

    def test_zero_output_comparison_reports_na_throughput_speedup(self):
        scenario = build_reference_scenario()
        request = replace(scenario.workload.requests[0], output_tokens=0)
        workload = replace(scenario.workload, requests=(request,))
        zero_output = replace(scenario, name="zero-output", workload=workload)

        comparison = compare_with_gpu_baseline(zero_output)

        self.assertIsNone(comparison["comparison"]["throughput_speedup"])
        self.assertIn("吞吐加速比：NA", format_comparison(zero_output))

    def test_two_requests_contend_deterministically(self):
        scenario = build_reference_scenario()
        first = scenario.workload.requests[0]
        second = replace(first, request_id="request-0001")
        workload = replace(scenario.workload, requests=(first, second))
        concurrent = replace(scenario, name="two-requests", workload=workload)

        result = run_scenario(concurrent)
        first_metrics = result.metrics.request_metrics[first.request_id]
        second_metrics = result.metrics.request_metrics[second.request_id]

        self.assertTrue(
            any(batch.request_ids == (first.request_id, second.request_id) for batch in result.serving.batches)
        )
        self.assertEqual(second_metrics.ttft_ns, first_metrics.ttft_ns)
        self.assertEqual(first_metrics.visible_output_tokens, 4)
        self.assertEqual(second_metrics.visible_output_tokens, 4)

    def test_invalid_op_target_and_kv_offload_fail_validation(self):
        scenario = build_reference_scenario()
        bad_mapping = dict(scenario.placement.op_to_component)
        bad_mapping["dense0.mlp"] = "hbm0"
        placement = replace(
            scenario.placement,
            op_to_component=bad_mapping,
            kv_policy=replace(
                scenario.placement.kv_policy,
                offload_component="hbm1",
            ),
        )
        report = validate_scenario(replace(scenario, placement=placement))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("不支持的组件类型" in item for item in report.errors))
        self.assertFalse(any("KV 卸载" in item for item in report.errors))


if __name__ == "__main__":
    unittest.main()
