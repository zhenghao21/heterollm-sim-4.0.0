import unittest
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.cost_models import HostMemoryProfile
from heterollm_sim.planner import (
    MappingResolutionError,
    ScenarioValidationError,
    compile_scenario,
    compile_serving_cohort_schedule,
    validate_scenario,
)
from heterollm_sim.reference import build_reference_scenario


def _cpu_scenario():
    scenario = build_reference_scenario()
    component_profiles = {
        kind: dict(registry)
        for kind, registry in scenario.component_profiles.items()
    }
    cpu_profile_id = scenario.hardware.get_component("cpu0").cost_profile_id
    host_memory_profile_id = scenario.hardware.get_component(
        "hostmem0"
    ).cost_profile_id
    assert cpu_profile_id is not None
    assert host_memory_profile_id is not None
    cpu_profile = component_profiles["cpu"][cpu_profile_id]
    component_profiles["cpu"][cpu_profile_id] = replace(
        cpu_profile,
        pipeline=replace(
            cpu_profile.pipeline,
            core_count=1000,
            frequency_ghz=1000.0,
            simd_width_bits=1024,
            vector_fma_units_per_core=1000,
            vector_alu_units_per_core=1000,
            load_units_per_core=1000,
            store_units_per_core=1000,
            resource_id="cpu0.pipeline",
        ),
        attainable_efficiency=1.0,
        dispatch_ns=0.0,
    )
    component_profiles["host_memory"][
        host_memory_profile_id
    ] = HostMemoryProfile(
        bandwidth_gb_s=1.0e12,
        resource_id="cpu0.memory",
    )
    return replace(
        scenario,
        component_profiles=component_profiles,
    )


class OperatorClassMappingTests(unittest.TestCase):
    def test_cpu_target_requires_cpu_and_host_memory_profiles(self):
        scenario = _cpu_scenario()
        missing_cpu = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        missing_cpu["cpu"] = {}
        with self.assertRaisesRegex(ValueError, "unknown cpu cost profile"):
            replace(scenario, component_profiles=missing_cpu)
        missing_host_memory = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        missing_host_memory["host_memory"] = {}
        with self.assertRaisesRegex(
            ValueError, "unknown host_memory cost profile"
        ):
            replace(scenario, component_profiles=missing_host_memory)

    def test_control_plane_keeps_typed_primitive_on_declared_rank(self):
        scenario = _cpu_scenario()
        mapped = plan_runtime_placement(scenario)

        self.assertEqual(
            mapped.placement.op_to_component["dense0.input_norm.reduce"],
            "gpu0",
        )
        decision = next(
            item
            for item in mapped.decisions
            if item.item_id == "dense0.input_norm.reduce"
        )
        self.assertIn("rank", decision.reason)
        self.assertGreater(decision.analytical_cost, 0.0)

    def test_control_plane_generated_keys_are_consumed_by_reference_lowering(self):
        scenario = build_reference_scenario()
        mapped = plan_runtime_placement(scenario)

        generated = set(
            mapped.placement.metadata["control_plane"]["decision"][
                "generated_op_keys"
            ]
        )
        report = validate_scenario(mapped.apply(scenario))
        unused_warnings = tuple(
            warning
            for warning in report.warnings_en
            if "not consumed by the reference lowering" in warning
        )

        self.assertTrue(generated)
        self.assertFalse(unused_warnings, unused_warnings)

    def test_unknown_manual_operator_key_still_warns(self):
        scenario = build_reference_scenario()
        unknown_key = "dense0.not_a_reference_operator"
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                op_to_component={
                    **scenario.placement.op_to_component,
                    unknown_key: "gpu0",
                },
            ),
        )

        report = validate_scenario(configured)

        self.assertTrue(
            any(
                "op placement key {} is not consumed".format(unknown_key)
                in warning
                for warning in report.warnings_en
            ),
            report.warnings_en,
        )

    def test_absent_aggregate_operator_group_still_warns(self):
        scenario = build_reference_scenario()
        unused_keys = {"linear_attention", "*.linear_attention"}
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                op_to_component={
                    **scenario.placement.op_to_component,
                    **{key: "gpu0" for key in unused_keys},
                },
            ),
        )

        warnings = validate_scenario(configured).warnings_en

        for key in unused_keys:
            with self.subTest(key=key):
                self.assertTrue(
                    any(
                        "op placement key {} is not consumed".format(key)
                        in warning
                        for warning in warnings
                    ),
                    warnings,
                )

    def test_explicit_cpu_chain_moves_once_and_cpu_gemm_is_supported(self):
        scenario = _cpu_scenario()
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                "dense0.input_norm.reduce": "cpu0",
                "dense0.input_norm.apply": "cpu0",
                "dense0.mlp": "cpu0",
            },
        )
        schedule = compile_scenario(replace(scenario, placement=placement))

        reduce_tasks = tuple(
            task
            for task in schedule.tasks
            if task.metadata.get("operator_id")
            == "dense0.input_norm.reduce"
        )
        apply_tasks = tuple(
            task
            for task in schedule.tasks
            if task.metadata.get("operator_id")
            == "dense0.input_norm.apply"
        )
        self.assertTrue(reduce_tasks)
        self.assertTrue(apply_tasks)
        self.assertTrue(
            any(
                task.metadata.get("event_kind")
                == "operator_input_transfer"
                for task in reduce_tasks
            )
        )
        self.assertFalse(
            any(
                task.metadata.get("event_kind")
                == "operator_input_transfer"
                for task in apply_tasks
            )
        )
        self.assertTrue(
            any(
                task.metadata.get("operator_class") == "gemm"
                and task.metadata.get("target_component") == "cpu0"
                and ".mlp_" in task.metadata.get("op_name", "")
                for task in schedule.tasks
            )
        )

    def test_cpu_attached_host_memory_weight_read_is_roofline_local(self):
        scenario = _cpu_scenario()
        configured = replace(
            scenario,
            weights_resident=False,
            placement=replace(
                scenario.placement,
                op_to_component={
                    **scenario.placement.op_to_component,
                    "dense0.mlp": "cpu0",
                },
                tensor_to_component={
                    **scenario.placement.tensor_to_component,
                    "dense0.mlp_weights": "hostmem0",
                },
            ),
        )

        schedule = compile_scenario(configured)
        local_reads = tuple(
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "model_weight_access"
            and task.metadata.get("weight_source_component") == "hostmem0"
            and task.metadata.get("weight_target_component") == "cpu0"
        )
        cpu_gemms = tuple(
            task
            for task in schedule.tasks
            if task.metadata.get("phase") == "cpu_gemm"
            and task.metadata.get("weight_tensor_id")
            == "dense0.mlp_weights"
        )

        self.assertTrue(local_reads)
        self.assertTrue(cpu_gemms)
        self.assertTrue(all(not task.demands for task in local_reads))
        self.assertTrue(
            all(
                task.metadata.get("resource_accounting")
                == "included_in_gemm_roofline"
                and task.metadata.get("compute_local_backing")
                == "cpu_attached_host_memory"
                and task.metadata.get("weight_source_transfer_emitted")
                is False
                and task.metadata.get("weight_backing_read_gate")
                == "source_is_cpu_attached_host_memory"
                for task in local_reads
            )
        )
        self.assertTrue(
            all(
                task.metadata["cost_model"]["read_bytes"]
                >= task.metadata["weight_read_bytes"]
                for task in cpu_gemms
            )
        )
        # ``estimate_cpu_gemm`` already routes activation + physical weight
        # reads through the local backing demand.  The planner must not add
        # the resident weight a second time merely because the lifecycle
        # marker records a model-weight access.
        self.assertTrue(
            all(
                next(
                    demand
                    for demand in task.demands
                    if demand.resource_id == "cpu0.memory"
                ).bytes_moved
                == task.metadata["cost_model"]["read_bytes"]
                + task.metadata["cost_model"]["write_bytes"]
                for task in cpu_gemms
            )
        )

    def test_non_gemm_cim_mapping_fails_closed_in_validation_and_lowering(self):
        scenario = _cpu_scenario()
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                "dense0.input_norm.reduce": "cim0",
            },
        )
        configured = replace(scenario, placement=placement)
        report = validate_scenario(configured)
        self.assertFalse(report.is_valid)
        diagnostic = next(
            item
            for item in report.diagnostics
            if item.get("operator_id") == "dense0.input_norm.reduce"
        )
        self.assertEqual(diagnostic["operator_class"], "reduction")
        self.assertEqual(diagnostic["requested_target"], "cim0")
        self.assertIn(diagnostic["resolved_target"], {"cpu0", "gpu0"})
        self.assertFalse(diagnostic["resolution_applied"])

        with self.assertRaises(ScenarioValidationError) as detailed_error:
            compile_scenario(configured)
        detailed_diagnostic = next(
            item
            for item in detailed_error.exception.details["diagnostics"]
            if item.get("operator_id") == "dense0.input_norm.reduce"
        )
        self.assertEqual(
            detailed_diagnostic["requested_target"], "cim0"
        )
        self.assertFalse(detailed_diagnostic["resolution_applied"])

        cohort = SimpleNamespace(
            cohort_id="typed-serving",
            kind="prefill",
            items=(
                SimpleNamespace(
                    request_id="request-0",
                    token_count=2,
                    context_tokens=0,
                    phase="prefill",
                    kv_append_tokens=2,
                ),
            ),
        )
        with self.assertRaises(MappingResolutionError) as serving_error:
            compile_serving_cohort_schedule(configured, cohort)
        self.assertEqual(
            serving_error.exception.details["operator_class"], "reduction"
        )
        self.assertEqual(
            serving_error.exception.details["requested_target"], "cim0"
        )
        self.assertFalse(serving_error.exception.details["resolution_applied"])

    def test_dynamic_qk_pv_rhs_is_transferred_and_never_treated_as_resident_weight(self):
        scenario = build_reference_scenario()
        configured = replace(
            scenario,
            weights_resident=True,
            fusion_policy=replace(scenario.fusion_policy, flash_attention=False),
            placement=replace(
                scenario.placement,
                op_to_component={
                    **scenario.placement.op_to_component,
                    "dense0.attention.qk": "cim0",
                    "dense0.attention.pv": "cim0",
                },
            ),
        )

        schedule = compile_scenario(configured)
        dynamic_names = ("attention_qk", "attention_pv")
        dynamic_tasks = tuple(
            task
            for task in schedule.tasks
            if any(marker in task.name for marker in dynamic_names)
            and ".dense0." in task.name
        )
        rhs_transfers = tuple(
            task
            for task in dynamic_tasks
            if task.metadata.get("transport") == "dynamic_rhs"
        )
        weight_loads = tuple(
            task
            for task in dynamic_tasks
            if task.metadata.get("phase") == "weight_load"
        )

        self.assertTrue(rhs_transfers)
        self.assertTrue(weight_loads)
        self.assertTrue(
            all(task.metadata.get("rhs_operand_kind") == "activation" for task in rhs_transfers)
        )
        self.assertTrue(all(task.metadata.get("bytes", 0) > 0 for task in rhs_transfers))
        self.assertFalse(
            any(task.metadata.get("event_kind") == "model_weight_read" for task in dynamic_tasks)
        )
        self.assertFalse(
            any(task.metadata.get("weight_tensor_id") for task in dynamic_tasks)
        )


if __name__ == "__main__":
    unittest.main()
