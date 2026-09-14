import unittest
from dataclasses import replace

from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.control_plane_state import (
    MAPPING_FINGERPRINT_SCHEMA,
    mapping_fingerprint_status,
    mapping_input_fingerprint,
    mapping_input_payload,
)
from heterollm_sim.ir import LinearAttentionSpec
from heterollm_sim.planner import validate_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive
from heterollm_sim.serving import compile_serving_plan
from heterollm_sim.web import validation_payload_for_scenario
from tests.model_helpers import execution_layers, replace_model_layer_specs


class MappingFingerprintContracts(unittest.TestCase):
    def setUp(self):
        self.original = build_reference_scenario()
        self.result = plan_runtime_placement(self.original)
        self.mapped = self.result.apply(self.original)

    def test_control_plane_changes_only_placement_and_is_immediately_current(self):
        before = to_primitive(self.original)
        after = to_primitive(self.mapped)
        before.pop("placement")
        after.pop("placement")

        self.assertEqual(after, before)
        self.assertNotIn("weights_resident", self.result.placement_patch)
        status = mapping_fingerprint_status(self.mapped)
        self.assertEqual(
            status["input_fingerprint"], status["current_input_fingerprint"]
        )
        self.assertFalse(status["mapping_stale"])
        self.assertFalse(validate_scenario(self.mapped).mapping_stale)

        cold = replace(self.original, weights_resident=False)
        cold_result = plan_runtime_placement(
            cold, PlacementPolicy(allow_cold_cim_streaming=True)
        )
        self.assertFalse(cold_result.apply(cold).weights_resident)

    def test_browser_json_round_trip_keeps_integral_time_limit_current(self):
        options = PlacementPolicy(
            mode="optimal",
            objective="balanced",
            time_limit_s=60.0,
            solver="builtin",
        )
        float_options = to_primitive(options)
        integer_options = dict(float_options, time_limit_s=60)
        self.assertEqual(
            mapping_input_fingerprint(self.original, options=float_options),
            mapping_input_fingerprint(self.original, options=integer_options),
        )

        result = plan_runtime_placement(self.original, options)
        mapped = result.apply(self.original)
        metadata = dict(mapped.placement.metadata)
        control_plane = dict(metadata["control_plane"])
        policy = dict(control_plane["policy"])
        self.assertEqual(policy["options"]["time_limit_s"], 60.0)
        policy["options"] = dict(
            policy["options"],
            time_limit_s=60,
        )
        control_plane["policy"] = policy
        metadata["control_plane"] = control_plane
        browser_round_trip = replace(
            mapped,
            placement=replace(mapped.placement, metadata=metadata),
        )

        status = mapping_fingerprint_status(browser_round_trip)
        self.assertEqual(
            status["input_fingerprint"], status["current_input_fingerprint"]
        )
        self.assertFalse(status["mapping_stale"])
        self.assertTrue(validate_scenario(browser_round_trip).is_valid)

    def test_fingerprint_payload_contains_the_complete_component_registry(self):
        component_profiles = {
            kind: dict(registry)
            for kind, registry in self.original.component_profiles.items()
        }
        component_profiles["hbm"]["unbound-hbm"] = replace(
            self.original.resolve_component_profile("hbm0"),
            efficiency=0.5,
        )
        heterogeneous = replace(
            self.original, component_profiles=component_profiles
        )

        payload = mapping_input_payload(heterogeneous)

        self.assertIn("unbound-hbm", payload["profiles"]["components"]["hbm"])
        self.assertNotEqual(
            mapping_input_fingerprint(self.original),
            mapping_input_fingerprint(heterogeneous),
        )

    def test_missing_or_old_fingerprint_schema_fails_closed(self):
        for fingerprint_schema in (None, "automatic-mapping-deployment-v2"):
            with self.subTest(fingerprint_schema=fingerprint_schema):
                metadata = dict(self.mapped.placement.metadata)
                control_plane = dict(metadata["control_plane"])
                evidence = dict(control_plane["evidence"])
                if fingerprint_schema is None:
                    evidence.pop("fingerprint_schema", None)
                else:
                    evidence["fingerprint_schema"] = fingerprint_schema
                control_plane["evidence"] = evidence
                metadata["control_plane"] = control_plane
                non_v3 = replace(
                    self.mapped,
                    placement=replace(self.mapped.placement, metadata=metadata),
                )

                status = mapping_fingerprint_status(non_v3)
                report = validate_scenario(non_v3)

                self.assertEqual(
                    status["current_fingerprint_schema"],
                    MAPPING_FINGERPRINT_SCHEMA,
                )
                self.assertEqual(status["fingerprint_schema"], fingerprint_schema)
                self.assertTrue(status["mapping_stale"])
                self.assertTrue(report.mapping_stale)
                self.assertFalse(report.is_valid)
                self.assertTrue(
                    any(
                        "control-plane decision is stale" in error
                        for error in report.errors_en
                    ),
                    report.errors_en,
                )

    def test_deployment_inputs_stale_mapping_but_runtime_workload_does_not(self):
        metadata = dict(self.mapped.hardware.metadata)
        metadata["topology_view"] = {
            "positions": {"gpu0": {"x": 999, "y": -20}},
            "viewport": {"scale": 1.5},
        }
        view_only = replace(
            self.mapped,
            hardware=replace(self.mapped.hardware, metadata=metadata),
        )
        self.assertFalse(mapping_fingerprint_status(view_only)["mapping_stale"])

        components = tuple(
            replace(component, peak_ops_per_s=component.peak_ops_per_s + 1)
            if component.component_id == "gpu0"
            else component
            for component in self.mapped.hardware.components
        )
        changed_hardware = replace(
            self.mapped,
            hardware=replace(self.mapped.hardware, components=components),
        )
        self.assertTrue(
            mapping_fingerprint_status(changed_hardware)["mapping_stale"]
        )

        request = replace(
            self.mapped.workload.requests[0],
            output_tokens=self.mapped.workload.requests[0].output_tokens + 1,
        )
        changed_workload = replace(
            self.mapped,
            workload=replace(self.mapped.workload, requests=(request,)),
        )
        workload_status = mapping_fingerprint_status(changed_workload)
        self.assertFalse(workload_status["mapping_stale"])
        self.assertEqual(
            workload_status["input_fingerprint"],
            workload_status["current_input_fingerprint"],
        )

        changed_parallel = replace(
            self.mapped,
            placement=replace(
                self.mapped.placement,
                parallel=replace(
                    self.mapped.placement.parallel,
                    collective_algorithm="tree",
                ),
            ),
        )
        self.assertTrue(
            mapping_fingerprint_status(changed_parallel)["mapping_stale"]
        )

        changed_kv = replace(
            self.mapped,
            placement=replace(
                self.mapped.placement,
                kv_policy=replace(
                    self.mapped.placement.kv_policy,
                    dtype="fp16",
                ),
            ),
        )
        self.assertTrue(mapping_fingerprint_status(changed_kv)["mapping_stale"])

        changed_workload_mtp = replace(
            self.mapped,
            workload=replace(
                self.mapped.workload,
                mtp=replace(
                    self.mapped.workload.mtp,
                    candidate_tokens=(
                        self.mapped.workload.mtp.candidate_tokens + 1
                    ),
                ),
            ),
        )
        self.assertFalse(
            mapping_fingerprint_status(changed_workload_mtp)["mapping_stale"]
        )

        # ModelGraph is the execution authority.  Exercise an actual typed MTP
        # weight change and synchronize its formal execution projection.
        mtp_operator_id = "mtp.prediction_layer.000"
        mtp_weight_id = mtp_operator_id + ".weights"
        updated_operators = []
        for operator in self.mapped.model.graph.operators:
            if operator.operator_id == mtp_operator_id:
                parameters = dict(operator.parameters)
                parameters["weight_bytes"] += 1
                operator = replace(operator, parameters=parameters)
            updated_operators.append(operator)
        updated_tensors = tuple(
            replace(tensor, logical_bytes=tensor.logical_bytes + 1)
            if tensor.tensor_id == mtp_weight_id
            else tensor
            for tensor in self.mapped.model.graph.tensors
        )
        updated_graph = replace(
            self.mapped.model.graph,
            operators=tuple(updated_operators),
            tensors=updated_tensors,
        )
        changed_model_mtp = replace(
            self.mapped,
            model=replace(
                self.mapped.model,
                graph=updated_graph,
            ),
        )
        self.assertTrue(
            mapping_fingerprint_status(changed_model_mtp)["mapping_stale"]
        )

    def test_backing_component_is_input_but_generated_output_is_not(self):
        tensor_mapping = dict(self.mapped.placement.tensor_to_component)
        tensor_mapping["model_weights"] = "hbm1"
        changed_backing = replace(
            self.mapped,
            placement=replace(
                self.mapped.placement,
                tensor_to_component=tensor_mapping,
            ),
        )
        self.assertTrue(
            mapping_fingerprint_status(changed_backing)["mapping_stale"]
        )

        op_mapping = dict(self.mapped.placement.op_to_component)
        op_mapping["unlocked.generated.output"] = "gpu0"
        output_only = replace(
            self.mapped,
            placement=replace(
                self.mapped.placement,
                op_to_component=op_mapping,
            ),
        )
        self.assertFalse(mapping_fingerprint_status(output_only)["mapping_stale"])

    def test_stale_fields_are_exposed_to_validation_api(self):
        changed = replace(
            self.mapped,
            model=replace(self.mapped.model, name=self.mapped.model.name + "-v2"),
        )
        payload = validation_payload_for_scenario(changed)

        self.assertTrue(payload["mapping_stale"])
        self.assertEqual(payload["fingerprint_algorithm"], "sha256")
        self.assertEqual(
            payload["mapping"]["input_fingerprint"],
            self.result.input_fingerprint,
        )
        self.assertFalse(payload["valid"])
        self.assertTrue(
            any(
                "控制平面决策已过期" in item["message"]
                for item in payload["errors"]["scenario"]
            )
        )


class AdmissionAndWorkloadContracts(unittest.TestCase):
    def test_schema_v4_prefetch_distance_fails_closed(self):
        base = build_reference_scenario()
        scenario = replace(
            base,
            placement=replace(
                base.placement,
                kv_policy=replace(base.placement.kv_policy, prefetch_distance=2),
            ),
        )

        report = validate_scenario(scenario)
        prefetch_errors = tuple(
            error
            for error in report.errors_en
            if "prefetch_distance" in error
        )

        self.assertEqual(
            prefetch_errors,
            (
                "V4 placement does not support non-zero prefetch_distance; "
                "proactive KV lookahead is not modeled",
            ),
        )
        with self.assertRaisesRegex(ValueError, "prefetch_distance"):
            compile_serving_plan(scenario)

    def test_old_scenario_schema_versions_are_rejected_by_typed_ir(self):
        base = build_reference_scenario()
        for version in ("0.1", "0.2", "0.3", "0.4"):
            with self.subTest(schema_version=version):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^scenario schema_version must be exactly 4\.0\.0$",
                ):
                    replace(base, schema_version=version)

    def test_full_sequence_limit_is_reported_once(self):
        base = build_reference_scenario()
        graph_attributes = dict(base.model.graph.attributes)
        graph_attributes["max_sequence_length"] = 10
        graph_operators = tuple(
            replace(
                operator,
                parameters=dict(operator.parameters, max_sequence_length=10),
            )
            if operator.op_kind == "model_input"
            else operator
            for operator in base.model.graph.operators
        )
        bounded_model = replace(
            base.model,
            graph=replace(
                base.model.graph,
                operators=graph_operators,
                attributes=graph_attributes,
            ),
        )
        scenario = replace(
            base,
            model=bounded_model,
            placement=replace(
                base.placement,
                kv_policy=replace(base.placement.kv_policy, prefetch_distance=0),
            ),
            workload=replace(
                base.workload,
                requests=(
                    replace(
                        base.workload.requests[0],
                        prompt_tokens=5,
                        output_tokens=6,
                    ),
                ),
            ),
        )

        report = validate_scenario(scenario)
        sequence_errors = tuple(
            error
            for error in report.errors_en
            if "sequence has" in error
        )

        self.assertEqual(
            sequence_errors,
            (
                "request request-0000 sequence has 11 tokens, exceeding "
                "model max_sequence_length 10",
            ),
        )

    def test_generated_linear_state_ledger_does_not_cap_runtime_state(self):
        scenario = build_reference_scenario()
        layers = execution_layers(scenario.model)
        linear_layer = replace(
            layers[0],
            sequence_mixer="linear_attention",
            linear_attention=LinearAttentionSpec(
                key_heads=4,
                value_heads=8,
                key_head_dim=16,
                value_head_dim=16,
                conv_kernel_size=4,
                state_dtype="fp32",
                output_gate=True,
                gate_activation="silu",
            ),
        )
        authored = replace(
            scenario,
            model=replace_model_layer_specs(
                scenario.model,
                (linear_layer,) + layers[1:],
            ),
        )
        materialized = plan_runtime_placement(authored).apply(authored)
        self.assertIn(
            "linear_state",
            materialized.placement.metadata["control_plane"]["decision"][
                "generated_tensor_ids"
            ],
        )
        tensor_bytes = dict(materialized.placement.tensor_bytes)
        tensor_bytes["linear_state"] = 1
        configured = replace(
            materialized,
            placement=replace(materialized.placement, tensor_bytes=tensor_bytes),
        )

        report = validate_scenario(configured)
        plan = compile_serving_plan(configured)

        self.assertFalse(report.mapping_stale, report.errors_en)
        self.assertTrue(report.is_valid, report.errors_en)
        self.assertFalse(
            any("tensor_bytes[linear_state] declares" in error for error in report.errors_en),
            report.errors_en,
        )
        self.assertGreaterEqual(plan.linear_state_policy.capacity_requests, 1)

    def test_generated_kv_ledger_does_not_cap_a_changed_workload(self):
        scenario = build_reference_scenario()
        mapped = plan_runtime_placement(scenario).apply(scenario)
        tensor_bytes = dict(mapped.placement.tensor_bytes)
        tensor_bytes["kv_cache"] = 16 * 1024
        request = replace(
            mapped.workload.requests[0],
            prompt_tokens=20_000,
            output_tokens=1,
        )
        configured = replace(
            mapped,
            workload=replace(mapped.workload, requests=(request,)),
            placement=replace(mapped.placement, tensor_bytes=tensor_bytes),
        )

        status = mapping_fingerprint_status(configured)
        report = validate_scenario(configured)
        plan = compile_serving_plan(configured)

        self.assertFalse(status["mapping_stale"])
        self.assertGreater(plan.kv_policy.capacity_bytes, 16 * 1024)
        self.assertFalse(
            any("KV working set requires" in error for error in report.errors_en),
            report.errors_en,
        )

    def test_changed_workload_over_physical_capacity_is_rejected_not_stale(self):
        scenario = build_reference_scenario()
        components = tuple(
            replace(component, capacity_bytes=256 * 1024)
            if component.kind.lower() in {"gpu", "hbm"}
            else component
            for component in scenario.hardware.components
        )
        constrained = replace(
            scenario,
            hardware=replace(scenario.hardware, components=components),
        )
        mapped = plan_runtime_placement(constrained).apply(constrained)
        request = replace(
            mapped.workload.requests[0],
            prompt_tokens=20_000,
            output_tokens=1,
        )
        configured = replace(
            mapped,
            workload=replace(mapped.workload, requests=(request,)),
        )

        report = validate_scenario(configured)

        self.assertFalse(report.mapping_stale)
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("KV working set requires" in error for error in report.errors_en),
            report.errors_en,
        )

    def test_all_requests_over_kv_capacity_fail_preflight(self):
        scenario = build_reference_scenario()
        request = replace(
            scenario.workload.requests[0],
            prompt_tokens=20_000,
            output_tokens=1,
        )
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes["kv_cache"] = 16 * 1024
        configured = replace(
            scenario,
            workload=replace(scenario.workload, requests=(request,)),
            placement=replace(
                scenario.placement,
                tensor_bytes=tensor_bytes,
            ),
        )

        report = validate_scenario(configured)

        self.assertFalse(report.is_valid)
        message = next(
            error for error in report.errors if "KV 工作集需要" in error
        )
        self.assertIn("KiB", message)
        self.assertNotRegex(message, r"\d{5,}\s*字节")

    def test_materialized_capacity_messages_use_iec_units(self):
        scenario = build_reference_scenario()
        mapped = plan_runtime_placement(scenario).apply(scenario)
        report = validate_scenario(mapped)

        messages = [
            message
            for message in report.warnings + report.information
            if "显式详细放置" in message or "显式覆盖声明" in message
        ]
        self.assertTrue(messages)
        self.assertTrue(any("MiB" in message for message in messages))
        self.assertTrue(all(" 字节" not in message for message in messages))

    def test_conflicting_synthetic_fields_warn_when_requests_are_explicit(self):
        scenario = build_reference_scenario()
        workload = replace(
            scenario.workload,
            request_count=8,
            prompt_tokens=1,
            output_tokens=99,
            arrival_rate_rps=123.0,
        )

        report = validate_scenario(replace(scenario, workload=workload))

        self.assertTrue(
            any(
                "显式 workload.requests 优先" in warning
                for warning in report.warnings
            )
        )


if __name__ == "__main__":
    unittest.main()
