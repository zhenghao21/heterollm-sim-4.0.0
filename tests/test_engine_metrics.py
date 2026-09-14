from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterollm_sim.contracts import (  # noqa: E402
    ResourceDemand,
    RunManifest,
    SIMULATION_SCHEMA_VERSION,
    TaskCategory,
    TaskSpec,
    TraceMarker,
)
from heterollm_sim.engine import ScheduleIR, simulate_schedule  # noqa: E402
import heterollm_sim.event_kernel as event_kernel_module  # noqa: E402
from heterollm_sim.execution_control import (  # noqa: E402
    ExecutionCancelledError,
    ExecutionControl,
)
from heterollm_sim.metrics import summarize_cost_metrics, summarize_metrics  # noqa: E402
import heterollm_sim.metrics as metrics_module  # noqa: E402


def manifest() -> RunManifest:
    return RunManifest(
        schema_version=SIMULATION_SCHEMA_VERSION,
        run_id="run-test",
        random_seed=1,
        simulator_version="test",
        model_name="toy",
        hardware_name="toy-hw",
        workload_name="toy-workload",
    )


def demand(resource_id: str, service_ns: float) -> ResourceDemand:
    return ResourceDemand(resource_id=resource_id, service_ns=service_ns)


def task(
    task_id: str,
    request_id: str = "r0",
    category: TaskCategory = TaskCategory.COMPUTE,
    dependencies=(),
    demands=(),
    earliest_start_ns: float = 0.0,
    marker=None,
    token_index=None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        request_id=request_id,
        name=task_id,
        category=category,
        dependencies=tuple(dependencies),
        demands=tuple(demands),
        earliest_start_ns=earliest_start_ns,
        marker=marker,
        token_index=token_index,
    )


class EngineMetricsTests(unittest.TestCase):
    def test_empty_and_zero_duration_graphs_are_deterministic(self) -> None:
        empty = simulate_schedule(ScheduleIR(manifest(), ()))
        self.assertEqual(empty.tasks, ())
        self.assertEqual(empty.resource_busy_ns, {})
        self.assertEqual(empty.makespan_ns, 0)

        tasks = (
            task("b", demands=(demand("r", 0),)),
            task("a", demands=(demand("r", 0),)),
            task("c", dependencies=("b",), demands=(demand("r", 0),)),
        )
        forward = simulate_schedule(ScheduleIR(manifest(), tasks))
        reverse = simulate_schedule(ScheduleIR(manifest(), tuple(reversed(tasks))))
        self.assertEqual(forward.tasks, reverse.tasks)
        self.assertEqual(
            [(item.task_id, item.start_ns, item.end_ns) for item in forward.tasks],
            [("a", 0, 0), ("b", 0, 0), ("c", 0, 0)],
        )
        self.assertEqual(forward.resource_busy_ns, {"r": 0})
        self.assertEqual(forward.makespan_ns, 0)

    def test_ready_heap_rekeys_one_shared_resource_group_linearly(self) -> None:
        total_tasks = 250
        schedule = ScheduleIR(
            manifest(),
            tuple(
                task(
                    "task-{:04d}".format(index),
                    demands=(demand("gpu0", 1),),
                )
                for index in range(total_tasks)
            ),
        )

        with mock.patch.object(
            event_kernel_module,
            "task_start_ns",
            wraps=event_kernel_module.task_start_ns,
        ) as task_start:
            trace = simulate_schedule(schedule)

        self.assertEqual(len(trace.tasks), total_tasks)
        self.assertEqual(trace.makespan_ns, total_tasks)
        # One initial candidate plus one replacement candidate per task is
        # linear.  The old per-task stale rekey loop made roughly N^2 calls.
        self.assertLessEqual(task_start.call_count, 3 * total_tasks)

    def test_multi_request_metrics_share_one_critical_path_index(self) -> None:
        request_shapes = (
            ("r0", 5, 3, 2),
            ("r1", 7, 4, 1),
            ("r2", 2, 6, 3),
        )
        tasks = []
        for request_id, prefill_ns, first_ns, second_ns in request_shapes:
            tasks.extend(
                (
                    task(
                        "{}-arrival".format(request_id),
                        request_id=request_id,
                        category=TaskCategory.POLICY,
                        marker=TraceMarker.REQUEST_ARRIVAL,
                    ),
                    task(
                        "{}-prefill".format(request_id),
                        request_id=request_id,
                        dependencies=("{}-arrival".format(request_id),),
                        demands=(demand("gpu0", prefill_ns),),
                    ),
                    task(
                        "{}-tok0".format(request_id),
                        request_id=request_id,
                        dependencies=("{}-prefill".format(request_id),),
                        demands=(demand("gpu0", first_ns),),
                        marker=TraceMarker.FIRST_TOKEN,
                        token_index=0,
                    ),
                    task(
                        "{}-tok1".format(request_id),
                        request_id=request_id,
                        dependencies=("{}-tok0".format(request_id),),
                        demands=(demand("gpu0", second_ns),),
                        marker=TraceMarker.TOKEN_EMIT,
                        token_index=1,
                    ),
                    task(
                        "{}-done".format(request_id),
                        request_id=request_id,
                        category=TaskCategory.OUTPUT,
                        dependencies=("{}-tok1".format(request_id),),
                        marker=TraceMarker.REQUEST_DONE,
                    ),
                )
            )

        trace = simulate_schedule(ScheduleIR(manifest(), tuple(tasks)))
        reversed_trace = simulate_schedule(
            ScheduleIR(manifest(), tuple(reversed(tasks)))
        )
        self.assertEqual(trace.tasks, reversed_trace.tasks)
        self.assertEqual(
            trace.by_task_id()["r1-prefill"].metadata[
                "_engine_resource_predecessors"
            ]["gpu0"]["task_id"],
            "r0-prefill",
        )

        with mock.patch.object(
            metrics_module,
            "_build_critical_path_index",
            wraps=metrics_module._build_critical_path_index,
        ) as build_index:
            metrics = summarize_metrics(trace)

        self.assertEqual(build_index.call_count, 1)
        self.assertEqual(trace.makespan_ns, 33)
        self.assertEqual(
            {
                request_id: (
                    item.ttft_ns,
                    item.tbt_ns,
                    item.tpot_ns,
                    item.e2e_ns,
                )
                for request_id, item in metrics.request_metrics.items()
            },
            {
                "r0": (17, (12,), 12, 29),
                "r1": (21, (9,), 9, 30),
                "r2": (27, (6,), 6, 33),
            },
        )
        self.assertEqual(
            {
                request_id: item.critical_path_category_ns[
                    TaskCategory.COMPUTE
                ]
                for request_id, item in metrics.request_metrics.items()
            },
            {"r0": 29, "r1": 30, "r2": 33},
        )
        self.assertEqual(
            metrics.critical_path_category_ns,
            {TaskCategory.COMPUTE: 33},
        )

    def test_cooperative_cancellation_stops_inside_detailed_event_loop(self) -> None:
        total_tasks = 600
        updates = []
        cancelled = {"value": False}

        def progress(update) -> None:
            updates.append(update)
            if 0 < update.completed < total_tasks:
                cancelled["value"] = True

        control = ExecutionControl(
            progress_callback=progress,
            cancellation_callback=lambda: cancelled["value"],
        )
        schedule = ScheduleIR(
            manifest=manifest(),
            tasks=tuple(task("task-{0:04d}".format(index)) for index in range(total_tasks)),
        )

        with self.assertRaisesRegex(ExecutionCancelledError, "已取消"):
            simulate_schedule(schedule, control=control)

        in_flight = [
            update
            for update in updates
            if 0 < update.completed < total_tasks
        ]
        self.assertEqual(len(in_flight), 1)
        self.assertLessEqual(in_flight[0].completed, 256)
        self.assertFalse(
            any(update.completed == total_tasks for update in updates)
        )

    def test_cost_metrics_match_general_global_summary(self) -> None:
        schedule = ScheduleIR(
            manifest=manifest(),
            tasks=(
                task(
                    "compute",
                    demands=(demand("gpu0", 10),),
                ),
                task(
                    "communicate",
                    category=TaskCategory.COMMUNICATION,
                    dependencies=("compute",),
                    demands=(demand("link0", 7),),
                ),
            ),
        )
        trace = simulate_schedule(schedule)

        general = summarize_metrics(trace)
        cost = summarize_cost_metrics(trace)

        self.assertEqual(cost.category_time_ns, general.category_time_ns)
        self.assertEqual(
            cost.critical_path_category_ns,
            general.critical_path_category_ns,
        )

    def test_request_metrics_are_hand_calculated(self) -> None:
        schedule = ScheduleIR(
            manifest=manifest(),
            tasks=(
                task(
                    "arrival",
                    category=TaskCategory.POLICY,
                    marker=TraceMarker.REQUEST_ARRIVAL,
                ),
                task("prefill", dependencies=("arrival",), demands=(demand("gpu0", 10),)),
                task(
                    "tok0",
                    dependencies=("prefill",),
                    demands=(demand("gpu0", 5),),
                    marker=TraceMarker.FIRST_TOKEN,
                    token_index=0,
                ),
                task(
                    "tok1",
                    dependencies=("tok0",),
                    demands=(demand("gpu0", 7),),
                    marker=TraceMarker.TOKEN_EMIT,
                    token_index=1,
                ),
                task(
                    "tok2",
                    dependencies=("tok1",),
                    demands=(demand("gpu0", 3),),
                    marker=TraceMarker.TOKEN_EMIT,
                    token_index=2,
                ),
                task(
                    "done",
                    category=TaskCategory.OUTPUT,
                    dependencies=("tok2",),
                    marker=TraceMarker.REQUEST_DONE,
                ),
            ),
        )

        trace = simulate_schedule(schedule)
        metrics = summarize_metrics(trace)
        request = metrics.request_metrics["r0"]

        self.assertEqual(trace.makespan_ns, 25)
        self.assertEqual(trace.resource_busy_ns, {"gpu0": 25.0})
        self.assertEqual(request.ttft_ns, 15)
        self.assertEqual(request.tbt_ns, (7, 3))
        self.assertEqual(request.tpot_ns, 5)
        self.assertEqual(request.e2e_ns, 25)
        self.assertEqual(request.visible_output_tokens, 3)
        self.assertEqual(metrics.category_time_ns[TaskCategory.COMPUTE], 25)
        self.assertEqual(metrics.critical_path_category_ns[TaskCategory.COMPUTE], 25)
        self.assertAlmostEqual(metrics.throughput["requests_per_s"], 40_000_000.0)
        self.assertAlmostEqual(
            metrics.throughput["visible_output_tokens_per_s"], 120_000_000.0
        )

    def test_shared_resources_queue_and_separate_resources_overlap(self) -> None:
        schedule = ScheduleIR(
            manifest=manifest(),
            tasks=(
                task("c", category=TaskCategory.COMMUNICATION, demands=(demand("link0", 7),)),
                task("b", demands=(demand("gpu0", 5),)),
                task("a", demands=(demand("gpu0", 10),)),
            ),
        )

        trace = simulate_schedule(schedule)
        by_task = trace.by_task_id()
        metrics = summarize_metrics(trace)

        self.assertEqual((by_task["a"].start_ns, by_task["a"].end_ns), (0.0, 10.0))
        self.assertEqual((by_task["b"].start_ns, by_task["b"].end_ns), (10.0, 15.0))
        self.assertEqual((by_task["c"].start_ns, by_task["c"].end_ns), (0.0, 7.0))
        self.assertEqual(trace.makespan_ns, 15)
        self.assertEqual(trace.resource_busy_ns, {"gpu0": 15.0, "link0": 7.0})
        self.assertEqual(metrics.resource_utilization["gpu0"], 1.0)
        self.assertAlmostEqual(metrics.resource_utilization["link0"], 7.0 / 15.0)
        self.assertEqual(metrics.category_time_ns[TaskCategory.COMPUTE], 15)
        self.assertEqual(metrics.category_time_ns[TaskCategory.COMMUNICATION], 7)
        self.assertEqual(metrics.critical_path_category_ns[TaskCategory.COMPUTE], 15)
        self.assertNotIn(TaskCategory.COMMUNICATION, metrics.critical_path_category_ns)

    def test_multiple_demands_start_together_and_task_ends_at_longest_demand(self) -> None:
        trace = simulate_schedule(
            ScheduleIR(
                manifest=manifest(),
                tasks=(
                    task(
                        "hybrid",
                        category=TaskCategory.CIM,
                        demands=(demand("gpu0", 10), demand("ucie0", 20)),
                    ),
                ),
            )
        )
        result = trace.by_task_id()["hybrid"]

        self.assertEqual(result.start_ns, 0)
        self.assertEqual(result.end_ns, 20)
        self.assertEqual(
            {interval.resource_id: interval.end_ns for interval in result.resource_intervals},
            {"gpu0": 10.0, "ucie0": 20.0},
        )
        self.assertEqual(trace.resource_busy_ns, {"gpu0": 10.0, "ucie0": 20.0})

    def test_schedule_is_deterministic_across_input_order(self) -> None:
        tasks = (
            task("b", demands=(demand("gpu0", 2),)),
            task("a", demands=(demand("gpu0", 1),)),
            task("c", dependencies=("b",), demands=(demand("gpu0", 3),)),
        )
        trace_a = simulate_schedule(ScheduleIR(manifest(), tasks))
        trace_b = simulate_schedule(ScheduleIR(manifest(), tuple(reversed(tasks))))

        self.assertEqual(
            [(item.task_id, item.start_ns, item.end_ns) for item in trace_a.tasks],
            [(item.task_id, item.start_ns, item.end_ns) for item in trace_b.tasks],
        )
        self.assertEqual(
            [(item.task_id, item.start_ns, item.end_ns) for item in trace_a.tasks],
            [("a", 0.0, 1.0), ("b", 1.0, 3.0), ("c", 3.0, 6.0)],
        )

    def test_missing_dependency_duplicate_id_and_cycle_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing dependencies"):
            simulate_schedule(
                ScheduleIR(manifest(), (task("a", dependencies=("missing",)),))
            )

        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            simulate_schedule(ScheduleIR(manifest(), (task("a"), task("a"))))

        with self.assertRaisesRegex(ValueError, "cycle"):
            simulate_schedule(
                ScheduleIR(
                    manifest(),
                    (
                        task("a", dependencies=("b",)),
                        task("b", dependencies=("a",)),
                    ),
                )
            )

    def test_dependency_chain_beyond_recursion_limit_is_valid(self) -> None:
        chain_length = sys.getrecursionlimit() + 10
        tasks = tuple(
            task(
                "task-{:04d}".format(index),
                dependencies=("task-{:04d}".format(index - 1),) if index else (),
            )
            for index in range(chain_length)
        )

        trace = simulate_schedule(ScheduleIR(manifest(), tasks))

        self.assertEqual(len(trace.tasks), chain_length)

    def test_tpot_is_none_for_single_visible_token(self) -> None:
        trace = simulate_schedule(
            ScheduleIR(
                manifest=manifest(),
                tasks=(
                    task(
                        "arrival",
                        category=TaskCategory.POLICY,
                        marker=TraceMarker.REQUEST_ARRIVAL,
                    ),
                    task(
                        "tok0",
                        dependencies=("arrival",),
                        demands=(demand("gpu0", 10),),
                        marker=TraceMarker.FIRST_TOKEN,
                        token_index=0,
                    ),
                    task(
                        "done",
                        category=TaskCategory.OUTPUT,
                        dependencies=("tok0",),
                        marker=TraceMarker.REQUEST_DONE,
                    ),
                ),
            )
        )
        request = summarize_metrics(trace).request_metrics["r0"]

        self.assertEqual(request.ttft_ns, 10)
        self.assertEqual(request.tbt_ns, ())
        self.assertIsNone(request.tpot_ns)
        self.assertEqual(request.e2e_ns, 10)

    def test_future_multi_resource_reservation_does_not_block_earlier_work(self) -> None:
        trace = simulate_schedule(
            ScheduleIR(
                manifest(),
                (
                    task("a_block_s", demands=(demand("s", 100),)),
                    task(
                        "b_multi",
                        category=TaskCategory.CIM,
                        demands=(demand("s", 10), demand("r", 10)),
                    ),
                    task(
                        "c_single_r",
                        earliest_start_ns=10,
                        demands=(demand("r", 20),),
                    ),
                ),
            )
        )
        by_task = trace.by_task_id()
        self.assertEqual(
            (by_task["c_single_r"].start_ns, by_task["c_single_r"].end_ns),
            (10.0, 30.0),
        )
        self.assertEqual(
            (by_task["b_multi"].start_ns, by_task["b_multi"].end_ns),
            (100.0, 110.0),
        )

    def test_critical_path_uses_blocking_resource_interval_not_whole_task(self) -> None:
        trace = simulate_schedule(
            ScheduleIR(
                manifest(),
                (
                    task(
                        "a_multi",
                        category=TaskCategory.CIM,
                        demands=(demand("r", 5), demand("s", 100)),
                    ),
                    task("b_r", demands=(demand("r", 150),)),
                ),
            )
        )
        metrics = summarize_metrics(trace)
        self.assertEqual(trace.makespan_ns, 155)
        self.assertEqual(metrics.critical_path_category_ns[TaskCategory.CIM], 5)
        self.assertEqual(metrics.critical_path_category_ns[TaskCategory.COMPUTE], 150)
        self.assertEqual(sum(metrics.critical_path_category_ns.values()), 155)

    def test_tied_dependency_and_resource_constraints_keep_longest_path(self) -> None:
        trace = simulate_schedule(
            ScheduleIR(
                manifest(),
                (
                    task(
                        "resource_path",
                        category=TaskCategory.CIM,
                        demands=(demand("r", 10),),
                    ),
                    task(
                        "dependency_path",
                        category=TaskCategory.COMMUNICATION,
                        earliest_start_ns=5,
                        demands=(demand("s", 5),),
                    ),
                    task(
                        "join",
                        dependencies=("dependency_path",),
                        demands=(demand("r", 5),),
                    ),
                ),
            )
        )

        join = trace.by_task_id()["join"]
        metrics = summarize_metrics(trace)
        self.assertEqual(
            join.metadata["_engine_resource_predecessors"]["r"]["task_id"],
            "resource_path",
        )
        self.assertEqual(
            metrics.critical_path_category_ns,
            {TaskCategory.CIM: 10, TaskCategory.COMPUTE: 5},
        )
        self.assertEqual(sum(metrics.critical_path_category_ns.values()), 15)

    def test_non_finite_task_times_and_realized_overflow_are_rejected(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "earliest_start_ns must be finite"
            ):
                simulate_schedule(
                    ScheduleIR(
                        manifest(),
                        (task("invalid", earliest_start_ns=value),),
                    )
                )

        with self.assertRaisesRegex(ValueError, "finite simulation range"):
            simulate_schedule(
                ScheduleIR(
                    manifest(),
                    (
                        task("first", demands=(demand("r", 1e308),)),
                        task("second", demands=(demand("r", 1e308),)),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
