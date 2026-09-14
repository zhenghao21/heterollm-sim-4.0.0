from __future__ import annotations

import unittest
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
from types import SimpleNamespace
from unittest import mock

import heterollm_sim.scalable_serving as scalable_serving
import heterollm_sim.streaming_des as streaming_des_module
from heterollm_sim.contracts import (
    ResourceDemand,
    RetentionPolicy,
    RunManifest,
    SIMULATION_SCHEMA_VERSION,
    TaskCategory,
    TaskSpec,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.planner import RequestTaskChunk, compile_streaming_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import run_scenario
from heterollm_sim.streaming_des import execute_incremental_schedule


def _static_scenario():
    scenario = build_reference_scenario()
    return replace(
        scenario,
        workload=replace(
            scenario.workload,
            scheduler=replace(
                scenario.workload.scheduler,
                mode="static",
            ),
        ),
    )


def _aggregate_projection(result):
    return {
        "task_count": result.task_count,
        "makespan_ns": result.trace.makespan_ns,
        "resource_busy_ns": result.trace.resource_busy_ns,
        "metrics": result.metrics,
        "total_energy_pj": result.total_energy_pj,
        "resource_accounted_bytes": result.resource_accounted_bytes,
        "proposed_tokens": result.proposed_tokens,
        "accepted_tokens": result.accepted_tokens,
        "kv_event_counts": result.kv_event_counts,
        "kv_logical_event_bytes": result.kv_logical_event_bytes,
        "kv_physical_event_bytes": result.kv_physical_event_bytes,
        "kv_phase_bytes": result.kv_phase_bytes,
        "state_event_counts": result.state_event_counts,
        "state_event_bytes": result.state_event_bytes,
        "analytical_coverage": result.analytical_coverage,
    }


class UnifiedKernelRetentionTests(unittest.TestCase):
    def test_retention_observers_preserve_tied_resource_critical_path(self):
        tasks = (
            TaskSpec(
                task_id="resource_path",
                request_id="r0",
                name="resource_path",
                category=TaskCategory.CIM,
                demands=(ResourceDemand("r", 10),),
            ),
            TaskSpec(
                task_id="dependency_path",
                request_id="r0",
                name="dependency_path",
                category=TaskCategory.COMMUNICATION,
                earliest_start_ns=5,
                demands=(ResourceDemand("s", 5),),
            ),
            TaskSpec(
                task_id="join",
                request_id="r0",
                name="join",
                category=TaskCategory.COMPUTE,
                dependencies=("dependency_path",),
                demands=(ResourceDemand("r", 5),),
            ),
        )
        manifest = RunManifest(
            schema_version=SIMULATION_SCHEMA_VERSION,
            run_id="retention-observer-test",
            random_seed=1,
            simulator_version="test",
            model_name="toy",
            hardware_name="toy-hw",
            workload_name="toy-workload",
        )

        def execute(retention_policy):
            schedule = SimpleNamespace(scenario=object(), manifest=manifest)
            request = SimpleNamespace(request_id="r0")
            chunk = RequestTaskChunk(tasks, "join", final=True)
            with mock.patch.object(
                streaming_des_module,
                "_request_iterator",
                return_value=iter((request,)),
            ), mock.patch.object(
                streaming_des_module,
                "_request_task_chunks",
                return_value=iter((chunk,)),
            ), mock.patch.object(
                streaming_des_module,
                "_compilation_scope",
                return_value=nullcontext(),
            ):
                return execute_incremental_schedule(
                    schedule,
                    retention_policy=retention_policy,
                    retained_task_limit=2,
                )

        exact = execute(RetentionPolicy.EXACT)
        streaming = execute(RetentionPolicy.STREAMING)
        aggregate = execute(RetentionPolicy.AGGREGATE)

        self.assertEqual(_aggregate_projection(streaming), _aggregate_projection(exact))
        self.assertEqual(_aggregate_projection(aggregate), _aggregate_projection(exact))
        self.assertEqual(
            exact.metrics.critical_path_category_ns,
            {TaskCategory.CIM: 10, TaskCategory.COMPUTE: 5},
        )

    def test_retention_modes_are_strictly_equivalent_before_retention(self):
        schedule = compile_streaming_scenario(_static_scenario())
        exact = execute_incremental_schedule(
            schedule,
            retention_policy=RetentionPolicy.EXACT,
        )
        streaming = execute_incremental_schedule(
            schedule,
            retention_policy=RetentionPolicy.STREAMING,
            retained_task_limit=7,
        )
        aggregate = execute_incremental_schedule(
            schedule,
            retention_policy=RetentionPolicy.AGGREGATE,
        )

        expected = _aggregate_projection(exact)
        self.assertEqual(_aggregate_projection(streaming), expected)
        self.assertEqual(_aggregate_projection(aggregate), expected)
        self.assertEqual(len(exact.trace.tasks), exact.task_count)
        self.assertEqual(len(streaming.trace.tasks), 7)
        self.assertEqual(len(aggregate.trace.tasks), 0)

    def test_all_modes_declare_one_kernel_and_v3_schema(self):
        self.assertFalse(hasattr(UnifiedEventKernel(), "retention"))
        schedule = compile_streaming_scenario(_static_scenario())
        for retention_policy in RetentionPolicy:
            with self.subTest(retention_policy=retention_policy.value):
                result = execute_incremental_schedule(
                    schedule,
                    retention_policy=retention_policy,
                    retained_task_limit=3,
                )
                manifest = result.trace.manifest
                self.assertEqual(
                    manifest.schema_version, SIMULATION_SCHEMA_VERSION
                )
                self.assertEqual(
                    manifest.metadata["event_kernel"], "UnifiedEventKernel"
                )
                self.assertEqual(
                    manifest.metadata["scheduling_semantics_version"],
                    SIMULATION_SCHEMA_VERSION,
                )
                self.assertEqual(
                    manifest.metadata["retention_policy"],
                    retention_policy.value,
                )

    def test_removed_backend_module_and_objects_are_absent(self):
        self.assertIsNone(
            importlib.util.find_spec("heterollm_sim.execution_backends")
        )
        self.assertFalse(hasattr(scalable_serving, "ValidatedAffineCostCache"))
        self.assertFalse(hasattr(scalable_serving, "simulate_online_scalable"))

    def test_run_scenario_rejects_every_removed_backend_name(self):
        scenario = _static_scenario()
        for removed_backend_name in (
            "auto",
            "detailed",
            "detailed_des",
            "scalable",
            "scalable_serving",
            "streaming_des",
        ):
            with self.subTest(
                removed_backend_name=removed_backend_name
            ), self.assertRaises(ValueError):
                run_scenario(
                    scenario, retention_policy=removed_backend_name
                )


if __name__ == "__main__":
    unittest.main()
