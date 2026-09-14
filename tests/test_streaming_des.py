from __future__ import annotations

import math
import random
import sys
import tracemalloc
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterollm_sim.contracts import (  # noqa: E402
    RunManifest,
    RetentionPolicy,
    SimulationTrace,
    TaskCategory,
)
from heterollm_sim.engine import simulate_schedule  # noqa: E402
from heterollm_sim.ir import KVCachePolicy, MTPPolicy, RequestSpec, WorkloadSpec  # noqa: E402
from heterollm_sim.metrics import MetricsSummary, RequestMetrics, summarize_metrics  # noqa: E402
from heterollm_sim.planner import (  # noqa: E402
    compile_scenario,
    compile_streaming_scenario,
)
from heterollm_sim.reference import build_reference_scenario  # noqa: E402
from heterollm_sim.reporting import ScenarioResult, report_dict, run_scenario  # noqa: E402
import heterollm_sim.reporting as reporting_module  # noqa: E402
import heterollm_sim.streaming_des as streaming_des_module  # noqa: E402
from heterollm_sim.run_estimation import estimate_scenario  # noqa: E402
from heterollm_sim.streaming_des import (  # noqa: E402
    ScheduleExecutionResult,
    execute_incremental_schedule,
)


def _strict_reference_scenario():
    """Keep reference fixtures inside the supported V4 KV contract."""

    base = build_reference_scenario()
    return replace(
        base,
        placement=replace(
            base.placement,
            kv_policy=replace(base.placement.kv_policy, prefetch_distance=0),
        ),
    )


class IncrementalScheduleTests(unittest.TestCase):
    def test_persistent_path_comparison_matches_tuple_lexicographic_order(
        self,
    ) -> None:
        def path(keys):
            state = streaming_des_module._PathState(17.0)
            for key in keys:
                state = streaming_des_module._PathState(
                    17.0,
                    previous=state,
                    node_key=key,
                )
            return state

        cases = (
            ((), ()),
            ((), ("a",)),
            (("a",), ()),
            (("a",), ("a",)),
            (("a",), ("a", "tail")),
            (("a", "tail"), ("a",)),
            (("a", "b", "c"), ("a", "b", "d")),
            (("a", "z", "tail"), ("b", "a")),
            (("", "same", "left"), ("", "same", "right")),
            (("aa",), ("a", "z")),
            (("á",), ("😀",)),
        )
        for left_keys, right_keys in cases:
            with self.subTest(left=left_keys, right=right_keys):
                left = path(left_keys)
                right = path(right_keys)
                expected = left if left_keys < right_keys else right
                self.assertIs(
                    streaming_des_module._better_path(left, right),
                    expected,
                )
                expected_sign = (left_keys > right_keys) - (
                    left_keys < right_keys
                )
                self.assertEqual(
                    streaming_des_module._compare_path_keys(left, right),
                    expected_sign,
                )

    def test_persistent_path_comparison_handles_zero_duration_and_no_materialize(
        self,
    ) -> None:
        empty = streaming_des_module._PathState(0.0)
        zero = streaming_des_module._extend_path(
            empty,
            0.0,
            TaskCategory.POLICY,
            "discarded-zero-key",
            3.0,
        )
        self.assertEqual(streaming_des_module._path_keys(zero), ())

        shared = streaming_des_module._extend_path(
            zero,
            1.0,
            TaskCategory.COMPUTE,
            "shared",
            4.0,
        )
        left = shared
        right = shared
        for index in range(2_048):
            left = streaming_des_module._extend_path(
                left,
                0.0,
                TaskCategory.COMPUTE,
                "left-{:04d}".format(index),
                4.0,
            )
            right = streaming_des_module._extend_path(
                right,
                0.0,
                TaskCategory.COMPUTE,
                "right-{:04d}".format(index),
                4.0,
            )

        with mock.patch.object(
            streaming_des_module,
            "_path_keys",
            side_effect=AssertionError("comparison materialized a path"),
        ):
            self.assertIs(
                streaming_des_module._better_path(left, right),
                left,
            )

    def test_persistent_path_random_differential_and_weak_branch_release(
        self,
    ) -> None:
        rng = random.Random(0)
        keys = ("", "a", "a", "aa", "b", "same", "á", "😀")

        def random_path():
            state = streaming_des_module._PathState(0.0)
            expected = ()
            for _ in range(rng.randrange(0, 80)):
                key = rng.choice(keys)
                duration = rng.choice((0.0, 0.0, 0.25, 1.0))
                state = streaming_des_module._extend_path(
                    state,
                    duration,
                    TaskCategory.COMPUTE,
                    key,
                    0.0,
                )
                if state.distance == 0.0:
                    expected = ()
                else:
                    expected += (key,)
            self.assertEqual(streaming_des_module._path_keys(state), expected)
            return state, expected

        paths = [random_path() for _ in range(200)]
        for _ in range(2_000):
            (left, left_keys), (right, right_keys) = rng.sample(paths, 2)
            expected_sign = (left_keys > right_keys) - (
                left_keys < right_keys
            )
            self.assertEqual(
                streaming_des_module._compare_path_keys(left, right),
                expected_sign,
            )

        import gc
        import weakref

        index = streaming_des_module._PathKeyIndex()
        state = streaming_des_module._PathState(
            0.0,
            key_path=index.root,
        )
        for value in range(256):
            state = streaming_des_module._PathState(
                1.0,
                previous=state,
                node_key="node-{}".format(value),
            )
        leaf = weakref.ref(state.key_path)
        del state
        gc.collect()
        self.assertIsNone(leaf())
        self.assertEqual(len(index._nodes), 0)
        index_reference = weakref.ref(index)
        del index
        gc.collect()
        self.assertIsNone(index_reference())

    def test_request_lowering_yields_planner_chunks_lazily(self) -> None:
        first_chunk = object()

        def chunks(_scenario, _request):
            yield first_chunk
            raise AssertionError("requested the next chunk eagerly")

        schedule = mock.Mock(scenario=object())
        request = object()
        with mock.patch.object(
            streaming_des_module,
            "iter_request_task_chunks",
            side_effect=chunks,
        ):
            iterator = streaming_des_module._request_task_chunks(
                schedule, request
            )
            self.assertIs(next(iterator), first_chunk)

    def test_streaming_retention_matches_exact_engine_metrics(self) -> None:
        for name, scenario in _parity_scenarios():
            with self.subTest(name=name):
                detailed_schedule = compile_scenario(scenario)
                detailed_trace = simulate_schedule(detailed_schedule)
                detailed_metrics = summarize_metrics(detailed_trace)

                streaming_schedule = compile_streaming_scenario(scenario)
                streaming = execute_incremental_schedule(
                    streaming_schedule,
                    retention_policy=RetentionPolicy.STREAMING,
                    retained_task_limit=len(detailed_schedule.tasks) + 1,
                )

                self.assertEqual(streaming.task_count, len(detailed_trace.tasks))
                self.assertAlmostEqual(
                    streaming.trace.makespan_ns, detailed_trace.makespan_ns
                )
                self.assertAlmostEqual(
                    streaming.metrics.makespan_ns, detailed_metrics.makespan_ns
                )
                self.assertMappingAlmostEqual(
                    streaming.trace.resource_busy_ns, detailed_trace.resource_busy_ns
                )
                self.assertMappingAlmostEqual(
                    streaming.metrics.resource_utilization,
                    detailed_metrics.resource_utilization,
                )
                detailed_total_energy_pj = sum(
                    interval.energy_pj
                    for task in detailed_trace.tasks
                    for interval in task.resource_intervals
                )
                self.assertTrue(
                    math.isclose(
                        streaming.total_energy_pj,
                        detailed_total_energy_pj,
                        rel_tol=1e-15,
                        abs_tol=1e-9,
                    ),
                    f"streaming energy {streaming.total_energy_pj!r} differs from "
                    f"detailed energy {detailed_total_energy_pj!r}",
                )
                self.assertEqual(
                    streaming.resource_accounted_bytes,
                    sum(
                        interval.bytes_moved
                        for task in detailed_trace.tasks
                        for interval in task.resource_intervals
                    ),
                )
                self.assertMetricsAlmostEqual(streaming.metrics, detailed_metrics)

    def test_streaming_retention_counts_kv_and_mtp_events_exactly(self) -> None:
        scenario = _kv_mtp_scenario()
        schedule = compile_streaming_scenario(scenario)
        detailed_task_count = len(compile_scenario(scenario).tasks)

        result = execute_incremental_schedule(
            schedule,
            retention_policy=RetentionPolicy.STREAMING,
            retained_task_limit=detailed_task_count + 1,
        )

        self.assertEqual(result.task_count, detailed_task_count)
        self.assertEqual(result.kv_event_counts, {"kv_append": 4, "kv_read": 2})
        self.assertEqual(
            result.kv_logical_event_bytes,
            {"kv_append": 4096, "kv_read": 4096},
        )
        self.assertEqual(
            result.kv_physical_event_bytes,
            {"kv_append": 4096, "kv_read": 4096},
        )
        self.assertEqual(result.proposed_tokens, 2)
        self.assertEqual(result.accepted_tokens, 2)

    def test_representative_task_history_is_strictly_bounded(self) -> None:
        scenario = _single_request_scenario(prompt_tokens=2, output_tokens=8)
        detailed_schedule = compile_scenario(scenario)
        detailed_trace = simulate_schedule(detailed_schedule)
        limit = 11

        result = execute_incremental_schedule(
            compile_streaming_scenario(scenario),
            retention_policy=RetentionPolicy.STREAMING,
            retained_task_limit=limit,
        )

        retained_ids = [task.task_id for task in result.trace.tasks]
        sorted_detailed = sorted(
            detailed_trace.tasks,
            key=lambda item: (item.start_ns, item.end_ns, item.task_id),
        )

        self.assertGreater(result.task_count, limit)
        self.assertEqual(result.task_count, len(detailed_trace.tasks))
        self.assertEqual(result.retained_task_limit, limit)
        self.assertLessEqual(len(result.trace.tasks), limit)
        self.assertEqual(len(result.trace.tasks), limit)
        self.assertEqual(len(retained_ids), len(set(retained_ids)))
        self.assertEqual(
            result.trace.tasks,
            tuple(
                sorted(
                    result.trace.tasks,
                    key=lambda item: (item.start_ns, item.end_ns, item.task_id),
                )
            ),
        )
        self.assertEqual(result.trace.tasks[0].task_id, sorted_detailed[0].task_id)
        self.assertEqual(result.trace.tasks[-1].task_id, sorted_detailed[-1].task_id)
        self.assertIn(
            "streaming retention keeps exact aggregates and a bounded task trace",
            result.trace.warnings,
        )

    def test_explicit_streaming_report_uses_exact_aggregates_when_trace_is_bounded(
        self,
    ) -> None:
        scenario = _kv_mtp_scenario(output_tokens=30)

        result = run_scenario(scenario, retention_policy="streaming")
        payload = report_dict(result, visualization_limit=3)

        self.assertIsInstance(result, ScenarioResult)
        self.assertEqual(result.retention_policy, "streaming")
        self.assertEqual(payload["retention_policy"], "streaming")
        self.assertEqual(payload["trace_fidelity"], "representative")
        self.assertGreater(result.execution.task_count, len(result.trace.tasks))
        self.assertEqual(payload["summary"]["task_count"], result.execution.task_count)
        self.assertAlmostEqual(
            payload["summary"]["total_energy_pj"],
            result.execution.total_energy_pj,
        )
        self.assertEqual(
            payload["summary"]["resource_accounted_bytes"],
            result.execution.resource_accounted_bytes,
        )
        self.assertEqual(
            payload["kv_cache"]["event_task_counts"],
            result.execution.kv_event_counts,
        )
        self.assertEqual(
            payload["kv_cache"]["logical_event_bytes"],
            result.execution.kv_logical_event_bytes,
        )
        self.assertEqual(
            payload["kv_cache"]["physical_event_bytes"],
            result.execution.kv_physical_event_bytes,
        )
        self.assertEqual(
            payload["summary"]["mtp"]["proposed_tokens"],
            result.execution.proposed_tokens,
        )
        self.assertEqual(
            payload["summary"]["mtp"]["accepted_tokens"],
            result.execution.accepted_tokens,
        )
        self.assertEqual(
            payload["analytical_coverage"]["runtime"],
            result.execution.analytical_coverage,
        )
        self.assertEqual(
            payload["report_limits"]["retained_tasks"],
            {
                "total": result.execution.task_count,
                "returned": len(result.trace.tasks),
                "limit": result.execution.retained_task_limit,
                "truncated": True,
            },
        )

        visualization = payload["visualization"]
        self.assertEqual(visualization["fidelity"], "representative")
        self.assertEqual(
            visualization["logical_event_total"], result.execution.task_count
        )
        self.assertEqual(
            visualization["retained_event_total"], len(result.trace.tasks)
        )
        self.assertEqual(visualization["pagination"]["total"], len(result.trace.tasks))
        self.assertTrue(visualization["truncated"])

        timeseries = payload["component_timeseries"]
        self.assertEqual(timeseries["execution_mode"], "static")
        self.assertEqual(timeseries["fidelity"], "representative")
        activity_series = [
            series
            for component in timeseries["components"]
            for series in component["series"]
            if series["metric"] == "busy_fraction"
        ] + [
            series
            for link in timeseries["links"]
            for series in link["series"]
            if series["metric"] == "link_bandwidth_utilization"
        ]
        self.assertTrue(activity_series)
        self.assertTrue(
            any(series["fidelity"] == "representative" for series in activity_series)
        )

    def test_synthetic_static_streaming_kv_report_uses_constant_memory(
        self,
    ) -> None:
        scenario = _strict_reference_scenario()
        scenario = replace(
            scenario,
            workload=replace(
                scenario.workload,
                requests=(),
                request_count=1_000_000,
                prompt_tokens=33,
                output_tokens=4,
            ),
        )
        execution = mock.Mock(
            kv_event_counts={},
            kv_logical_event_bytes={},
            kv_physical_event_bytes={},
            kv_phase_bytes={},
        )
        result = mock.Mock(scenario=scenario, execution=execution)
        kv_policy = mock.Mock(
            tokens_per_page=16,
            bytes_per_page=4096,
            logical_bytes_per_token=256,
            capacity_pages=1024,
            capacity_bytes=1024 * 4096,
        )

        with mock.patch.object(
            reporting_module,
            "compile_serving_plan",
            return_value=mock.Mock(kv_policy=kv_policy),
        ):
            tracemalloc.start()
            try:
                payload = reporting_module._static_kv_report(result)
                _, peak_traced_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        self.assertEqual(payload["peak_used_pages"], 1024)
        self.assertEqual(payload["peak_used_bytes"], 1024 * 4096)
        self.assertEqual(payload["max_live_tokens_per_request"], 36)
        self.assertLess(peak_traced_bytes, 2 * 1024 * 1024)

    def test_explicit_streaming_uses_incremental_executor_without_full_graph(
        self,
    ) -> None:
        scenario = _critical_static_scenario()
        self.assertEqual(
            estimate_scenario(scenario)["recommended_retention_policy"],
            "streaming",
        )
        fake_schedule = object()
        fake_execution = _empty_execution_result(
            scenario, RetentionPolicy.STREAMING
        )

        with mock.patch.object(
            reporting_module,
            "compile_streaming_scenario",
            return_value=fake_schedule,
        ) as compile_schedule, mock.patch.object(
            reporting_module,
            "execute_incremental_schedule",
            return_value=fake_execution,
        ) as execute:
            result = run_scenario(scenario, retention_policy="streaming")

        self.assertEqual(result.retention_policy, "streaming")
        compile_schedule.assert_called_once()
        compiled_scenario = compile_schedule.call_args.args[0]
        self.assertEqual(compiled_scenario.model, scenario.model)
        self.assertEqual(compiled_scenario.workload, scenario.workload)
        self.assertTrue(compiled_scenario.placement.tensor_to_component)
        self.assertIn("control_plane", compiled_scenario.placement.metadata)
        execute.assert_called_once_with(
            fake_schedule,
            retention_policy=RetentionPolicy.STREAMING,
            control=mock.ANY,
            execution_kernel=mock.ANY,
            runtime_origin_ns=mock.ANY,
        )
        self.assertGreater(execute.call_args.kwargs["runtime_origin_ns"], 0.0)

    def test_explicit_exact_is_not_overridden_by_estimator_risk(self) -> None:
        scenario = _critical_static_scenario()
        self.assertEqual(
            estimate_scenario(scenario)["recommended_retention_policy"],
            "streaming",
        )
        fake_schedule = object()
        fake_execution = _empty_execution_result(
            scenario, RetentionPolicy.EXACT
        )

        with mock.patch.object(
            reporting_module,
            "compile_streaming_scenario",
            return_value=fake_schedule,
        ), mock.patch.object(
            reporting_module,
            "execute_incremental_schedule",
            return_value=fake_execution,
        ) as execute:
            result = run_scenario(scenario, retention_policy="exact")

        self.assertEqual(result.retention_policy, "exact")
        execute.assert_called_once_with(
            fake_schedule,
            retention_policy=RetentionPolicy.EXACT,
            control=mock.ANY,
            execution_kernel=mock.ANY,
            runtime_origin_ns=mock.ANY,
        )
        self.assertGreater(execute.call_args.kwargs["runtime_origin_ns"], 0.0)

    def assertMetricsAlmostEqual(
        self, actual: MetricsSummary, expected: MetricsSummary
    ) -> None:
        self.assertAlmostEqual(actual.makespan_ns, expected.makespan_ns)
        self.assertMappingAlmostEqual(actual.throughput, expected.throughput)
        self.assertMappingAlmostEqual(actual.category_time_ns, expected.category_time_ns)
        self.assertMappingAlmostEqual(
            actual.critical_path_category_ns, expected.critical_path_category_ns
        )
        self.assertMappingAlmostEqual(
            actual.resource_utilization, expected.resource_utilization
        )
        self.assertEqual(set(actual.request_metrics), set(expected.request_metrics))
        for request_id in expected.request_metrics:
            self.assertRequestMetricsAlmostEqual(
                actual.request_metrics[request_id],
                expected.request_metrics[request_id],
            )

    def assertRequestMetricsAlmostEqual(
        self, actual: RequestMetrics, expected: RequestMetrics
    ) -> None:
        self.assertEqual(actual.request_id, expected.request_id)
        self.assertOptionalAlmostEqual(actual.arrival_ns, expected.arrival_ns)
        self.assertOptionalAlmostEqual(actual.first_token_ns, expected.first_token_ns)
        self.assertOptionalAlmostEqual(actual.done_ns, expected.done_ns)
        self.assertOptionalAlmostEqual(actual.ttft_ns, expected.ttft_ns)
        self.assertSequenceAlmostEqual(actual.tbt_ns, expected.tbt_ns)
        self.assertOptionalAlmostEqual(actual.tpot_ns, expected.tpot_ns)
        self.assertOptionalAlmostEqual(actual.e2e_ns, expected.e2e_ns)
        self.assertEqual(actual.visible_output_tokens, expected.visible_output_tokens)
        self.assertMappingAlmostEqual(actual.category_time_ns, expected.category_time_ns)
        self.assertMappingAlmostEqual(
            actual.critical_path_category_ns, expected.critical_path_category_ns
        )

    def assertMappingAlmostEqual(
        self,
        actual: Mapping[object, float],
        expected: Mapping[object, float],
    ) -> None:
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])

    def assertSequenceAlmostEqual(
        self, actual: Sequence[float], expected: Sequence[float]
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right)

    def assertOptionalAlmostEqual(
        self, actual: Optional[float], expected: Optional[float]
    ) -> None:
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertIsNotNone(actual)
            self.assertAlmostEqual(actual, expected)


def _single_request_scenario(
    *,
    prompt_tokens: int,
    output_tokens: int,
):
    base = _strict_reference_scenario()
    return replace(
        base,
        workload=WorkloadSpec(
            name="single-request",
            requests=(
                RequestSpec(
                    request_id="r0",
                    arrival_ns=0.0,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                ),
            ),
        ),
    )


def _multi_request_scenario(*, future_arrival: bool):
    base = _strict_reference_scenario()
    second_arrival_ns = 100_000.0 if future_arrival else 0.0
    return replace(
        base,
        workload=WorkloadSpec(
            name="future-arrivals" if future_arrival else "shared-resource-arrivals",
            requests=(
                RequestSpec(
                    request_id="r0",
                    arrival_ns=0.0,
                    prompt_tokens=2,
                    output_tokens=2,
                ),
                RequestSpec(
                    request_id="r1",
                    arrival_ns=second_arrival_ns,
                    prompt_tokens=2,
                    output_tokens=2,
                ),
            ),
        ),
    )


def _kv_mtp_scenario(*, output_tokens: int = 3):
    base = _strict_reference_scenario()
    return replace(
        base,
        placement=replace(
            base.placement,
            kv_policy=KVCachePolicy(
                cache_component="hbm0",
                offload_component="hbm1",
                tokens_per_page=2,
                dtype="int8",
                prefetch_distance=0,
            ),
        ),
        workload=WorkloadSpec(
            name="kv-mtp",
            requests=(
                RequestSpec(
                    request_id="r0",
                    arrival_ns=0.0,
                    prompt_tokens=2,
                    output_tokens=output_tokens,
                ),
            ),
            mtp=MTPPolicy(
                candidate_tokens=2,
                acceptance_rate=0.5,
                proposal_cost_scale=0.1,
            ),
        ),
    )


def _parity_scenarios():
    return (
        ("single_request", _single_request_scenario(prompt_tokens=2, output_tokens=2)),
        ("shared_resource_competition", _multi_request_scenario(future_arrival=False)),
        ("future_arrival", _multi_request_scenario(future_arrival=True)),
        ("kv_mtp", _kv_mtp_scenario()),
    )


def _critical_static_scenario():
    base = _strict_reference_scenario()
    return replace(
        base,
        workload=WorkloadSpec(
            name="critical-static-estimate",
            request_count=1_000_000,
            prompt_tokens=1,
            output_tokens=0,
        ),
    )


def _manifest_for(scenario):
    return RunManifest(
        schema_version=scenario.schema_version,
        run_id="streaming-test",
        random_seed=scenario.workload.random_seed,
        simulator_version="test",
        model_name=scenario.model.name,
        hardware_name=scenario.hardware.name,
        workload_name=scenario.workload.name,
    )


def _empty_metrics() -> MetricsSummary:
    return MetricsSummary(
        makespan_ns=0.0,
        request_metrics={},
        throughput={
            "requests_per_s": 0.0,
            "visible_output_tokens_per_s": 0.0,
        },
        category_time_ns={},
        critical_path_category_ns={},
        resource_utilization={},
    )


def _empty_execution_result(
    scenario,
    retention_policy: RetentionPolicy,
) -> ScheduleExecutionResult:
    trace = SimulationTrace(
        manifest=_manifest_for(scenario),
        tasks=(),
        resource_busy_ns={},
        makespan_ns=0.0,
    )
    return ScheduleExecutionResult(
        trace=trace,
        metrics=_empty_metrics(),
        task_count=0,
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
        retention_policy=retention_policy,
        retained_task_limit=(
            1 if retention_policy is RetentionPolicy.STREAMING else None
        ),
    )


if __name__ == "__main__":
    unittest.main()
