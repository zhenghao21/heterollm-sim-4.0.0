import math
import unittest
from dataclasses import replace

from heterollm_sim.contracts import (
    ChangePointInterval,
    ResourceInterval,
    RunManifest,
    RetentionPolicy,
    SimulationTrace,
    TaskCategory,
    TaskResult,
)
from heterollm_sim.metrics import bound_change_point_intervals, summarize_metrics
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import ScenarioResult, report_dict, run_scenario
from heterollm_sim.serving import compile_serving_plan
from heterollm_sim.streaming_des import ScheduleExecutionResult


def _all_series(payload):
    contract = payload["component_timeseries"]
    return [
        series
        for component in contract["components"]
        for series in component["series"]
    ] + [
        series
        for link in contract["links"]
        for series in link["series"]
    ]


def _scenario_result(scenario, trace):
    execution = ScheduleExecutionResult(
        trace=trace,
        metrics=summarize_metrics(trace),
        task_count=len(trace.tasks),
        total_energy_pj=0.0,
        resource_accounted_bytes=0,
        proposed_tokens=0,
        accepted_tokens=0,
        kv_event_counts={},
        kv_logical_event_bytes={},
        kv_physical_event_bytes={},
        kv_phase_bytes={},
        state_event_counts={},
        state_event_bytes={},
        analytical_coverage={},
        retention_policy=RetentionPolicy.EXACT,
        retained_task_limit=None,
    )
    return ScenarioResult(
        scenario=scenario,
        execution=execution,
        validation_warnings=(),
        retention_policy=RetentionPolicy.EXACT.value,
    )


class ComponentTimeseriesContractTests(unittest.TestCase):
    def _static_hbm_payload_for_memory_metadata(self, metadata, total_bytes=100):
        scenario = build_reference_scenario()
        components = tuple(
            replace(
                component,
                read_bandwidth_gbps=8.0,
                write_bandwidth_gbps=8.0,
            )
            if component.component_id == "hbm0"
            else component
            for component in scenario.hardware.components
        )
        scenario = replace(
            scenario,
            hardware=replace(scenario.hardware, components=components),
        )
        manifest = RunManifest(
            schema_version=scenario.schema_version,
            run_id="hbm-output-byte-series",
            random_seed=0,
            simulator_version="test",
            model_name=scenario.model.name,
            hardware_name=scenario.hardware.name,
            workload_name=scenario.workload.name,
        )
        task = TaskResult(
            task_id="gemm-memory",
            request_id="request",
            name="rank0.gemm.tensor",
            category=TaskCategory.COMPUTE,
            start_ns=0.0,
            end_ns=1_000_000_000.0,
            dependency_ready_ns=0.0,
            resource_intervals=(
                ResourceInterval(
                    "component.hbm0.tensor",
                    0.0,
                    1_000_000_000.0,
                    total_bytes,
                ),
            ),
            metadata={
                "rank": 0,
                "operator_class": "gemm",
                "event_kind": "gemm",
                **metadata,
            },
        )
        trace = SimulationTrace(
            manifest=manifest,
            tasks=(task,),
            resource_busy_ns={
                "component.hbm0.tensor": task.duration_ns,
            },
            makespan_ns=task.duration_ns,
        )
        return report_dict(_scenario_result(scenario, trace))

    def _active_component_metric_value(self, payload, component_id, metric):
        component = next(
            row
            for row in payload["component_timeseries"]["components"]
            if row["component_id"] == component_id
        )
        series = next(row for row in component["series"] if row["metric"] == metric)
        return next(point["value"] for point in series["points"] if point["value"])

    def test_static_memory_timeseries_splits_output_bytes_as_writes(self):
        payload = self._static_hbm_payload_for_memory_metadata(
            {"output_bytes": 30}
        )

        self.assertAlmostEqual(
            self._active_component_metric_value(
                payload, "hbm0", "memory_write_bandwidth_utilization"
            ),
            30.0 / 1.0e9,
        )
        self.assertAlmostEqual(
            self._active_component_metric_value(
                payload, "hbm0", "memory_read_bandwidth_utilization"
            ),
            70.0 / 1.0e9,
        )

    def test_static_memory_timeseries_prefers_modeled_write_bytes(self):
        payload = self._static_hbm_payload_for_memory_metadata(
            {
                "output_bytes": 80,
                "modeled_memory_write_bytes": 25,
            }
        )

        self.assertAlmostEqual(
            self._active_component_metric_value(
                payload, "hbm0", "memory_write_bandwidth_utilization"
            ),
            25.0 / 1.0e9,
        )
        self.assertAlmostEqual(
            self._active_component_metric_value(
                payload, "hbm0", "memory_read_bandwidth_utilization"
            ),
            75.0 / 1.0e9,
        )

    def test_static_memory_timeseries_uses_resource_specific_direction(self):
        payload = self._static_hbm_payload_for_memory_metadata(
            {
                "memory_direction": "read",
                "resource_directions": {
                    "component.hbm0.tensor": "write",
                },
            }
        )

        self.assertAlmostEqual(
            self._active_component_metric_value(
                payload, "hbm0", "memory_write_bandwidth_utilization"
            ),
            100.0 / 1.0e9,
        )
        component = next(
            row
            for row in payload["component_timeseries"]["components"]
            if row["component_id"] == "hbm0"
        )
        read_series = next(
            row
            for row in component["series"]
            if row["metric"] == "memory_read_bandwidth_utilization"
        )
        self.assertTrue(all(point["value"] == 0.0 for point in read_series["points"]))

    def test_cpu_utilization_uses_operator_class_profile_not_generic_peak_ops(self):
        scenario = build_reference_scenario()
        cpu = ComponentSpec(
            component_id="cpu-report",
            kind="cpu",
            cost_profile_id="report-cpu",
            peak_ops_per_s=1.0e15,
        )
        cpu_profile = replace(
            scenario.resolve_component_profile("cpu0"),
            pipeline=replace(
                scenario.resolve_component_profile("cpu0").pipeline,
                core_count=1,
                frequency_ghz=1.25,
                simd_width_bits=64,
                vector_fma_units_per_core=50,
                vector_alu_units_per_core=20,
                resource_id="cpu.compute",
            ),
            attainable_efficiency=0.5,
        )
        component_profiles = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        component_profiles["cpu"]["report-cpu"] = cpu_profile
        scenario = replace(
            scenario,
            hardware=replace(
                scenario.hardware,
                components=scenario.hardware.components + (cpu,),
            ),
            component_profiles=component_profiles,
        )
        manifest = RunManifest(
            schema_version=scenario.schema_version,
            run_id="cpu-class-series",
            random_seed=0,
            simulator_version="test",
            model_name=scenario.model.name,
            hardware_name=scenario.hardware.name,
            workload_name=scenario.workload.name,
        )
        task = TaskResult(
            "cpu-elementwise",
            "request",
            "cpu.elementwise",
            TaskCategory.COMPUTE,
            0.0,
            4.0,
            0.0,
            resource_intervals=(
                ResourceInterval("cpu-report.compute", 0.0, 4.0),
            ),
            metadata={
                "operator_class": "elementwise",
                "analytical_ops": 100.0,
            },
        )
        trace = SimulationTrace(
            manifest=manifest,
            tasks=(task,),
            resource_busy_ns={"cpu-report.compute": 4.0},
            makespan_ns=4.0,
        )
        payload = report_dict(_scenario_result(scenario, trace))
        component = next(
            row
            for row in payload["component_timeseries"]["components"]
            if row["component_id"] == "cpu-report"
        )
        utilization = next(
            row
            for row in component["series"]
            if row["metric"] == "modeled_compute_utilization"
        )
        active = next(point for point in utilization["points"] if point["value"])
        self.assertAlmostEqual(active["value"], 0.5)

    def test_storage_media_dma_and_link_roles_are_not_collapsed(self):
        scenario = build_reference_scenario()
        storage = ComponentSpec(
            component_id="hbf-report",
            kind="hbf",
            capacity_bytes=1 << 30,
            read_bandwidth_gbps=80.0,
            write_bandwidth_gbps=40.0,
            metadata={"dma_bandwidth_gbps": 20.0},
        )
        scenario = replace(
            scenario,
            hardware=replace(
                scenario.hardware,
                components=scenario.hardware.components + (storage,),
            ),
        )
        manifest = RunManifest(
            schema_version=scenario.schema_version,
            run_id="storage-role-series",
            random_seed=0,
            simulator_version="test",
            model_name=scenario.model.name,
            hardware_name=scenario.hardware.name,
            workload_name=scenario.workload.name,
        )
        tasks = (
            TaskResult(
                "media-read",
                "request",
                "storage.read",
                TaskCategory.MEMORY,
                0.0,
                2.0,
                0.0,
                resource_intervals=(
                    ResourceInterval(
                        "component.hbf-report.read", 0.0, 2.0, 20
                    ),
                ),
                metadata={"event_kind": "memory_read"},
            ),
            TaskResult(
                "dma",
                "request",
                "storage.dma",
                TaskCategory.COMMUNICATION,
                2.0,
                4.0,
                2.0,
                resource_intervals=(
                    ResourceInterval("component.hbf-report.dma", 2.0, 4.0, 5),
                ),
                metadata={"event_kind": "dma"},
            ),
            TaskResult(
                "media-write",
                "request",
                "storage.write",
                TaskCategory.MEMORY,
                4.0,
                6.0,
                4.0,
                resource_intervals=(
                    ResourceInterval(
                        "component.hbf-report.write", 4.0, 6.0, 10
                    ),
                ),
                metadata={"event_kind": "memory_write"},
            ),
        )
        trace = SimulationTrace(
            manifest=manifest,
            tasks=tasks,
            resource_busy_ns={
                interval.resource_id: interval.end_ns - interval.start_ns
                for task in tasks
                for interval in task.resource_intervals
            },
            makespan_ns=6.0,
        )
        payload = report_dict(_scenario_result(scenario, trace))
        component = next(
            row
            for row in payload["component_timeseries"]["components"]
            if row["component_id"] == "hbf-report"
        )
        by_metric = {row["metric"]: row for row in component["series"]}
        for metric in (
            "storage_read_bandwidth_utilization",
            "storage_write_bandwidth_utilization",
            "dma_engine_utilization",
            "storage_io_utilization",
        ):
            self.assertIn(metric, by_metric)
            self.assertTrue(
                any(point["value"] > 0.0 for point in by_metric[metric]["points"])
            )

    def test_kv_endpoint_phases_count_once_in_static_and_online_traffic(self):
        scenario = build_reference_scenario()
        components = tuple(
            replace(
                component,
                read_bandwidth_gbps=256.0,
                write_bandwidth_gbps=256.0,
            )
            if component.component_id in {"gpu0", "hbm1"}
            else component
            for component in scenario.hardware.components
        )
        placement = scenario.placement
        placement = replace(
            placement,
            kv_policy=replace(
                placement.kv_policy,
                cache_component="hbm1",
                offload_component=None,
            ),
        )
        request = replace(
            scenario.workload.requests[0], prompt_tokens=4, output_tokens=2
        )
        workload = replace(
            scenario.workload,
            requests=(request,),
            scheduler=replace(
                scenario.workload.scheduler, mode="static"
            ),
            mtp=None,
        )
        static = replace(
            scenario,
            hardware=replace(scenario.hardware, components=components),
            placement=placement,
            workload=workload,
        )
        serving_plan = compile_serving_plan(static)
        page_tokens = serving_plan.kv_policy.tokens_per_page
        page_bytes = serving_plan.kv_policy.bytes_per_page
        per_token = (page_bytes + page_tokens - 1) // page_tokens
        expected_prefill_write = request.prompt_tokens * per_token

        static_result = run_scenario(static, retention_policy="exact")
        static_payload = report_dict(static_result)
        access_phase_counts = {}
        for task in static_result.trace.tasks:
            access_id = task.metadata.get("kv_access_id")
            if access_id and task.metadata.get("event_kind") == "kv_append":
                access_phase_counts[access_id] = (
                    access_phase_counts.get(access_id, 0) + 1
                )
        self.assertTrue(access_phase_counts)
        self.assertGreaterEqual(max(access_phase_counts.values()), 3)
        self.assertEqual(
            static_payload["kv_cache"]["physical_prefill_write_bytes"],
            expected_prefill_write,
        )

        continuous = replace(
            static,
            workload=replace(
                workload,
                scheduler=replace(
                    scenario.workload.scheduler,
                    mode="continuous",
                ),
            ),
        )
        online_result = run_scenario(
            continuous, retention_policy="aggregate"
        )
        online_payload = report_dict(online_result)
        prefill_batch = next(
            batch
            for batch in online_result.serving.batches
            if batch.kind == "prefill"
        )
        self.assertEqual(
            prefill_batch.cost.metadata["kv_append_physical_bytes"],
            expected_prefill_write,
        )
        self.assertEqual(
            online_payload["kv_cache"]["physical_prefill_write_bytes"],
            expected_prefill_write,
        )

    def test_static_report_exposes_kv_traffic_and_prompt_plus_output_minus_one_peak(self):
        scenario = build_reference_scenario()
        scheduler = replace(
            scenario.workload.scheduler, mode="static"
        )
        scenario = replace(
            scenario,
            workload=replace(scenario.workload, scheduler=scheduler),
        )

        payload = report_dict(
            run_scenario(scenario, retention_policy="exact")
        )
        kv = payload["kv_cache"]
        request = scenario.workload.requests[0]
        self.assertEqual(kv["logical_prefill_read_bytes"], 0)
        self.assertGreater(kv["logical_prefill_write_bytes"], 0)
        self.assertGreater(kv["logical_decode_append_bytes"], 0)
        self.assertEqual(kv["offload_events"], 0)
        self.assertEqual(kv["prefetch_events"], 0)
        self.assertEqual(
            kv["max_live_tokens_per_request"],
            request.prompt_tokens + request.output_tokens - 1,
        )
        self.assertFalse(kv["prefetch_distance_modeled"])

    def test_change_point_contract_rejects_non_finite_and_invalid_bounds(self):
        with self.assertRaises(ValueError):
            ChangePointInterval(0.0, 1.0, math.nan)
        with self.assertRaises(ValueError):
            ChangePointInterval(-1.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            ChangePointInterval(2.0, 1.0, 0.0)

    def test_large_series_is_bounded_and_preserves_time_weighted_mean(self):
        intervals = (
            ChangePointInterval(float(index), float(index + 1), float(index % 2))
            for index in range(20_000)
        )
        bounded = bound_change_point_intervals(intervals)

        self.assertEqual(bounded.raw_point_count, 20_000)
        self.assertEqual(bounded.point_count, 500)
        self.assertTrue(bounded.merged)
        weighted_total = sum(
            point.value * (point.end_ns - point.start_ns)
            for point in bounded.points
        )
        self.assertAlmostEqual(weighted_total / 20_000.0, 0.5)
        self.assertTrue(
            all(
                math.isfinite(value) and 0.0 <= point.value <= 1.0
                for point in bounded.points
                for value in (point.start_ns, point.end_ns, point.value)
            )
        )

    def test_static_report_uses_exact_change_points_and_v3_report_sections(self):
        scenario = build_reference_scenario()
        manifest = RunManifest(
            schema_version=scenario.schema_version,
            run_id="component-series-test",
            random_seed=0,
            simulator_version="test",
            model_name=scenario.model.name,
            hardware_name=scenario.hardware.name,
            workload_name=scenario.workload.name,
        )
        task = TaskResult(
            task_id="compute",
            request_id="request",
            name="已知计算",
            category=TaskCategory.COMPUTE,
            start_ns=2.0,
            end_ns=6.0,
            dependency_ready_ns=2.0,
            resource_intervals=(
                ResourceInterval("gpu0.compute", 2.0, 6.0),
            ),
            metadata={"rank": 0, "analytical_ops": 240_000.0},
        )
        trace = SimulationTrace(
            manifest=manifest,
            tasks=(task,),
            resource_busy_ns={"gpu0.compute": 4.0},
            makespan_ns=10.0,
        )
        result = _scenario_result(scenario, trace)
        payload = report_dict(result)

        self.assertIn("resource_utilization", payload)
        self.assertIn("visualization", payload)
        contract = payload["component_timeseries"]
        self.assertEqual(contract["schema_version"], "1.0")
        self.assertEqual(contract["fidelity"], "exact")
        gpu = next(
            row for row in contract["components"] if row["component_id"] == "gpu0"
        )
        busy = next(
            row for row in gpu["series"] if row["metric"] == "busy_fraction"
        )
        self.assertEqual(
            busy["points"],
            [
                {"start_ns": 0.0, "end_ns": 2.0, "value": 0.0},
                {"start_ns": 2.0, "end_ns": 6.0, "value": 1.0},
                {"start_ns": 6.0, "end_ns": 10.0, "value": 0.0},
            ],
        )
        compute = next(
            row
            for row in gpu["series"]
            if row["metric"] == "modeled_compute_utilization"
        )
        active = next(point for point in compute["points"] if point["value"] > 0.0)
        self.assertAlmostEqual(active["value"], 0.5)
        unknown = {
            row["metric"]: row
            for row in gpu["series"]
            if row["metric"]
            in {"activation_residency_bytes", "temporary_residency_bytes"}
        }
        self.assertEqual(unknown["activation_residency_bytes"]["quality"], "unknown")
        self.assertEqual(unknown["activation_residency_bytes"]["points"], [])
        self.assertEqual(unknown["temporary_residency_bytes"]["points"], [])

    def test_continuous_report_is_aggregate_finite_and_bounded(self):
        scenario = build_reference_scenario()
        result = run_scenario(scenario)
        payload = report_dict(result)
        contract = payload["component_timeseries"]

        self.assertEqual(
            payload["kv_cache"]["max_live_tokens_per_request"],
            max(
                request.prompt_tokens + max(request.output_tokens - 1, 0)
                for request in scenario.workload.requests
            ),
        )

        self.assertEqual(contract["execution_mode"], "continuous_batching")
        self.assertEqual(contract["fidelity"], "aggregate")
        self.assertTrue(contract["components"])
        self.assertTrue(contract["links"])
        series_rows = _all_series(payload)
        self.assertTrue(series_rows)
        for series in series_rows:
            self.assertLessEqual(series["point_count"], 500)
            self.assertEqual(series["point_count"], len(series["points"]))
            if series["unit"] == "ratio":
                self.assertTrue(
                    all(0.0 <= point["value"] <= 1.0 for point in series["points"])
                )
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for point in series["points"]
                    for value in point.values()
                )
            )
        self.assertTrue(
            any(
                series["metric"] == "kv_cache_residency_bytes"
                and series["fidelity"] == "aggregate"
                for series in series_rows
            )
        )
        kv_cache = payload["kv_cache"]
        self.assertGreater(kv_cache["physical_prefill_write_bytes"], 0)
        self.assertGreater(kv_cache["physical_decode_append_bytes"], 0)
        self.assertTrue(
            any(
                "not added to resource-accounted bytes" in limitation
                for limitation in kv_cache["modeling_limits"]
            )
        )
        cache_component = next(
            component
            for component in contract["components"]
            if component["component_id"]
            == result.scenario.placement.kv_policy.cache_component
        )
        self.assertTrue(
            any(
                series["metric"] == "busy_fraction"
                and any(point["value"] > 0.0 for point in series["points"])
                for series in cache_component["series"]
            )
        )


if __name__ == "__main__":
    unittest.main()
