import unittest
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.contracts import TaskCategory
from heterollm_sim.ir import ComponentSpec, LinkSpec, PortSpec
from heterollm_sim.planner import (
    ScenarioValidationError,
    compile_scenario,
    compile_serving_cohort_schedule,
    validate_scenario,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import compile_serving_plan, serving_admission_diagnostics


def _with_hbf(
    *,
    capacity_bytes,
    read_bandwidth_gbps=64.0,
    write_bandwidth_gbps=32.0,
    scenario=None,
):
    scenario = scenario or build_reference_scenario()
    gpu = scenario.hardware.get_component("gpu0")
    gpu_port = PortSpec(
        "hbf-port",
        "UCIe",
        "endpoint",
        bandwidth_gbps=64.0,
        payload="streaming",
    )
    hbf_port = replace(gpu_port, port_id="host")
    hbf = ComponentSpec(
        component_id="hbf0",
        kind="hbf",
        ports=(hbf_port,),
        package_id=gpu.package_id,
        die_id="hbf_die",
        capacity_bytes=capacity_bytes,
        read_bandwidth_gbps=read_bandwidth_gbps,
        write_bandwidth_gbps=write_bandwidth_gbps,
        metadata={
            "read_latency_ns": 100.0,
            "write_latency_ns": 200.0,
            "transfer_granularity_bytes": 4096,
        },
    )
    hardware = replace(
        scenario.hardware,
        components=tuple(
            replace(component, ports=component.ports + (gpu_port,))
            if component.component_id == gpu.component_id
            else component
            for component in scenario.hardware.components
        )
        + (hbf,),
        links=scenario.hardware.links
        + (
            LinkSpec(
                "gpu-hbf0",
                "gpu0",
                "hbf-port",
                "hbf0",
                "host",
                "UCIe",
                bandwidth_gbps=64.0,
                latency_ns=20.0,
                payload="streaming",
            ),
        ),
    )
    return replace(scenario, hardware=hardware)


def _with_no_unallocated_active_weight_capacity(scenario):
    occupied = {
        component_id: sum(
            int(byte_count)
            for tensor_name, byte_count in scenario.placement.tensor_bytes.items()
            if scenario.placement.tensor_to_component.get(tensor_name)
            == component_id
        )
        for component_id in ("hbm0", "cim0")
    }
    return replace(
        scenario,
        hardware=replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=occupied[component.component_id])
                if component.component_id in occupied
                else component
                for component in scenario.hardware.components
            ),
        ),
    )


def _with_component_capacity(scenario, component_id, capacity_bytes):
    return replace(
        scenario,
        hardware=replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=capacity_bytes)
                if component.component_id == component_id
                else component
                for component in scenario.hardware.components
            ),
        ),
    )


class PlannerMappingFailClosedTests(unittest.TestCase):
    def test_legacy_norm_cim_mapping_is_diagnostic_only_and_fails_closed(self):
        scenario = build_reference_scenario()
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                op_to_component={
                    **scenario.placement.op_to_component,
                    "dense0.norm": "cim0",
                },
            ),
        )

        report = validate_scenario(configured)

        self.assertFalse(report.is_valid)
        diagnostics = tuple(
            diagnostic
            for diagnostic in report.diagnostics
            if diagnostic.get("requested_mapping_key") == "dense0.norm"
        )
        self.assertTrue(diagnostics)
        self.assertEqual(
            {diagnostic["requested_target"] for diagnostic in diagnostics},
            {"cim0"},
        )
        self.assertTrue(
            all(
                diagnostic["resolved_target"] in {"gpu0", "cpu0"}
                and diagnostic["resolution_applied"] is False
                for diagnostic in diagnostics
            )
        )
        with self.assertRaises(ScenarioValidationError) as raised:
            compile_scenario(configured)
        self.assertGreaterEqual(
            len(raised.exception.details["diagnostics"]),
            len(diagnostics),
        )

    def test_v4_zero_active_memory_capacity_fails_closed(self):
        scenario = _with_component_capacity(build_reference_scenario(), "hbm0", 0)

        report = validate_scenario(scenario)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any(
                "V4 placement requires positive capacity_bytes" in error
                and "hbm0" in error
                for error in report.errors_en
            ),
            report.errors_en,
        )
        self.assertTrue(
            any(
                "request-0000 KV working set" in error
                and "cache capacity 0 bytes" in error
                for error in report.errors_en
            ),
            report.errors_en,
        )
        self.assertTrue(
            any(
                "request-0000 KV working set" in failure
                and "cache capacity 0 bytes" in failure
                for failure in serving_admission_diagnostics(scenario)
            )
        )

        plan = compile_serving_plan(scenario)
        self.assertEqual(plan.kv_policy.capacity_bytes, 0)
        self.assertEqual(plan.kv_policy.capacity_pages, 0)

    def test_prior_schema_versions_are_rejected_at_construction(self):
        for version in ("0.1", "0.2", "0.3", "0.4"):
            with self.subTest(schema_version=version):
                with self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
                    replace(build_reference_scenario(), schema_version=version)

    def test_control_plane_rejects_zero_active_memory(self):
        configured = _with_component_capacity(build_reference_scenario(), "hbm0", 0)

        mapped = plan_runtime_placement(configured)

        self.assertFalse(mapped.fully_placed)
        self.assertTrue(
            any(
                item.item_id == "scenario_validation"
                for item in mapped.unplaced
            ),
            mapped.unplaced,
        )
        self.assertTrue(
            any(
                "hbm0" in warning
                and "capacity_bytes=0" in warning
                and "按不可用处理" in warning
                for warning in mapped.warnings
            ),
            mapped.warnings,
        )

    def test_zero_capacity_offload_is_rejected_by_mapping_and_validation(self):
        authored = build_reference_scenario()
        materialized = plan_runtime_placement(authored).apply(authored)
        scenario = _with_hbf(capacity_bytes=0, scenario=materialized)
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                "dense0.mlp": "gpu0",
            },
            tensor_to_component={
                **scenario.placement.tensor_to_component,
                "dense0.mlp_weights": "hbf0",
            },
            metadata={
                **scenario.placement.metadata,
                "control_plane": {
                    "policy": {
                        "locked_op_keys": ["dense0.mlp"],
                        "locked_tensor_ids": ["dense0.mlp_weights"],
                    },
                    "decision": {},
                    "evidence": {},
                },
            },
        )
        configured = replace(scenario, placement=placement)

        report = validate_scenario(configured)
        self.assertTrue(
            any(
                "hbf0" in error
                and "capacity_bytes=0" in error
                and "正容量" in error
                for error in report.errors
            ),
            report.errors,
        )

        mapped = plan_runtime_placement(scenario)
        self.assertTrue(
            any(
                "hbf0" in warning
                and "不会将其作为无限容量" in warning
                for warning in mapped.warnings
            ),
            mapped.warnings,
        )

    def test_hbf_missing_write_bandwidth_fails_serving_swap_closed(self):
        scenario = _with_hbf(
            capacity_bytes=1 << 30,
            write_bandwidth_gbps=0.0,
        )
        cohort = SimpleNamespace(
            cohort_id="swap-out",
            kind="kv_swap_out",
            metadata={
                "source_component": "hbm0",
                "target_component": "hbf0",
                "byte_count": 4096,
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            r"storage component hbf0 requires a positive write bandwidth",
        ):
            compile_serving_cohort_schedule(scenario, cohort)

    def test_cold_weights_require_backing(self):
        scenario = build_reference_scenario()
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                tensor_to_component={
                    key: value
                    for key, value in scenario.placement.tensor_to_component.items()
                    if "weight" not in key
                },
                tensor_bytes={
                    key: value
                    for key, value in scenario.placement.tensor_bytes.items()
                    if "weight" not in key
                },
            ),
            weights_resident=False,
        )

        report = validate_scenario(scenario)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any(
                "cold/streamed weights require" in error
                for error in report.errors_en
            ),
            report.errors_en,
        )

    def test_resident_offload_backing_and_active_capacity_fail_independently(self):
        authored = build_reference_scenario()
        base = plan_runtime_placement(authored).apply(authored)
        total_bytes = base.model.total_declared_weight_bytes
        scenario = _with_hbf(
            capacity_bytes=total_bytes - 1,
            scenario=base,
        )
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                tensor_to_component={
                    **scenario.placement.tensor_to_component,
                    "model_weights": "hbf0",
                },
                tensor_bytes={
                    **scenario.placement.tensor_bytes,
                    "model_weights": total_bytes,
                },
            ),
            weights_resident=True,
        )
        scenario = _with_no_unallocated_active_weight_capacity(scenario)
        scenario = _with_component_capacity(
            scenario,
            "cim0",
            scenario.hardware.get_component("cim0").capacity_bytes - 1,
        )
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                metadata={
                    key: value
                    for key, value in scenario.placement.metadata.items()
                    if key != "control_plane"
                },
            ),
        )

        report = validate_scenario(scenario)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any(
                "declared tensors require {} bytes on hbf0".format(total_bytes)
                in error
                for error in report.errors_en
            ),
            report.errors_en,
        )
        self.assertTrue(
            any(
                "declared tensors require" in error and "cim0" in error
                for error in report.errors_en
            ),
            report.errors_en,
        )

    def test_offload_capacity_cannot_substitute_for_active_residency(self):
        authored = build_reference_scenario()
        base = plan_runtime_placement(authored).apply(authored)
        total_bytes = base.model.total_declared_weight_bytes
        scenario = _with_hbf(
            capacity_bytes=total_bytes * 4,
            scenario=base,
        )
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                tensor_to_component={
                    **scenario.placement.tensor_to_component,
                    "model_weights": "hbf0",
                },
                tensor_bytes={
                    **scenario.placement.tensor_bytes,
                    "model_weights": total_bytes,
                },
            ),
            weights_resident=True,
        )
        scenario = _with_no_unallocated_active_weight_capacity(scenario)
        scenario = _with_component_capacity(
            scenario,
            "cim0",
            scenario.hardware.get_component("cim0").capacity_bytes - 1,
        )
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                metadata={
                    key: value
                    for key, value in scenario.placement.metadata.items()
                    if key != "control_plane"
                },
            ),
        )

        report = validate_scenario(scenario)

        self.assertTrue(
            any(
                "declared tensors require" in error and "cim0" in error
                for error in report.errors_en
            ),
            report.errors_en,
        )
        self.assertFalse(
            any(
                "declared tensors require" in error and "hbf0" in error
                for error in report.errors_en
            ),
            report.errors_en,
        )

    def test_transfer_and_collective_tasks_have_operator_classes(self):
        scenario = _with_hbf(capacity_bytes=1 << 30)
        placement = replace(
            scenario.placement,
            tensor_to_component={
                **scenario.placement.tensor_to_component,
                "model_weights": "hbf0",
            },
            tensor_bytes={
                **scenario.placement.tensor_bytes,
                "model_weights": scenario.model.total_declared_weight_bytes,
            },
        )
        schedule = compile_scenario(
            replace(
                scenario,
                placement=placement,
                weights_resident=False,
            )
        )

        transfers = tuple(
            task
            for task in schedule.tasks
            if task.category == TaskCategory.COMMUNICATION and task.demands
        )
        self.assertTrue(transfers)
        self.assertTrue(
            all(
                task.metadata.get("operator_class") == "communication"
                for task in transfers
            )
        )
        self.assertTrue(
            any(
                task.metadata.get("source_operator_class") == "gemm"
                for task in transfers
            )
        )

        collectives = tuple(
            task
            for task in schedule.tasks
            if "collective" in str(task.metadata.get("event_kind", ""))
        )
        self.assertTrue(collectives)
        self.assertTrue(
            all(
                task.metadata.get("operator_class") == "communication"
                for task in collectives
            )
        )


if __name__ == "__main__":
    unittest.main()
