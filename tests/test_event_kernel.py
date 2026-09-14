from __future__ import annotations

import math
import random
import unittest
from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import (
    CompiledGraphLayout,
    KernelEvent,
    ReadyKey,
    UnifiedEventKernel,
    task_start_ns,
)


def demand(resource_id: str, service_ns: float) -> ResourceDemand:
    return ResourceDemand(resource_id=resource_id, service_ns=service_ns)


def task(
    task_id: str,
    *,
    dependencies: Iterable[str] = (),
    demands: Iterable[ResourceDemand] = (),
    earliest_start_ns: float = 0.0,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        request_id="request",
        name=task_id,
        category=TaskCategory.COMPUTE,
        dependencies=tuple(dependencies),
        demands=tuple(demands),
        earliest_start_ns=earliest_start_ns,
    )


def reference_task_start_ns(
    demands: Sequence[ResourceDemand],
    effective_ready_ns: float,
    resource_available: Mapping[str, float],
) -> float:
    """Pre-optimization expression retained as an independent oracle."""

    resources_ready_ns = max(
        (resource_available.get(item.resource_id, 0.0) for item in demands),
        default=0.0,
    )
    return max(effective_ready_ns, resources_ready_ns)


def reference_schedule(
    tasks: Sequence[TaskSpec],
) -> Tuple[
    List[KernelEvent],
    Dict[str, float],
    Dict[str, Dict[str, object]],
    Dict[str, float],
    Dict[str, float],
]:
    """Brute-force the original deterministic event semantics."""

    remaining = {item.task_id: item for item in tasks}
    completed_end: Dict[str, float] = {}
    resource_available: Dict[str, float] = {}
    resource_last_interval: Dict[str, Dict[str, object]] = {}
    resource_busy_ns: Dict[str, float] = {}
    events: List[KernelEvent] = []

    while remaining:
        ready: List[Tuple[ReadyKey, TaskSpec]] = []
        for item in remaining.values():
            if not all(
                dependency_id in completed_end
                for dependency_id in item.dependencies
            ):
                continue
            dependency_ready_ns = max(
                (completed_end[dependency_id] for dependency_id in item.dependencies),
                default=0.0,
            )
            effective_ready_ns = max(
                dependency_ready_ns,
                item.earliest_start_ns,
            )
            start_ns = reference_task_start_ns(
                item.demands,
                effective_ready_ns,
                resource_available,
            )
            ready.append(
                (
                    (
                        start_ns,
                        effective_ready_ns,
                        dependency_ready_ns,
                        item.earliest_start_ns,
                        item.task_id,
                    ),
                    item,
                )
            )
        if not ready:
            raise AssertionError("reference graph is not drainable")

        ready_key, selected = min(ready, key=lambda entry: entry[0])
        start_ns, effective_ready_ns, dependency_ready_ns, _, task_id = ready_key
        sorted_demands = tuple(
            sorted(selected.demands, key=lambda item: item.resource_id)
        )
        interval_end_ns = tuple(
            start_ns + item.service_ns for item in sorted_demands
        )
        if not math.isfinite(start_ns) or any(
            not math.isfinite(end_ns) for end_ns in interval_end_ns
        ):
            raise ValueError("reference timing exceeds finite range")

        resource_predecessors: Dict[str, Dict[str, object]] = {}
        end_ns = start_ns
        for item, demand_end_ns in zip(sorted_demands, interval_end_ns):
            previous_interval = resource_last_interval.get(item.resource_id)
            available_ns = resource_available.get(item.resource_id, 0.0)
            if previous_interval is not None and available_ns == start_ns:
                resource_predecessors[item.resource_id] = dict(previous_interval)
            end_ns = max(end_ns, demand_end_ns)
            resource_available[item.resource_id] = demand_end_ns
            resource_last_interval[item.resource_id] = {
                "task_id": selected.task_id,
                "resource_id": item.resource_id,
                "start_ns": start_ns,
                "end_ns": demand_end_ns,
            }
            resource_busy_ns[item.resource_id] = (
                resource_busy_ns.get(item.resource_id, 0.0) + item.service_ns
            )

        completed_end[task_id] = end_ns
        remaining.pop(task_id)
        events.append(
            KernelEvent(
                task=selected,
                start_ns=start_ns,
                end_ns=end_ns,
                dependency_ready_ns=dependency_ready_ns,
                effective_ready_ns=effective_ready_ns,
                demands=sorted_demands,
                resource_predecessors=resource_predecessors,
            )
        )

    return (
        events,
        completed_end,
        resource_available,
        resource_last_interval,
        resource_busy_ns,
    )


class EventKernelHotLoopTests(unittest.TestCase):
    def test_prevalidated_bulk_drain_matches_incremental_persistent_state(self) -> None:
        rng = random.Random(0xC4A128)
        capacities = {"dma": 3, "gpu": 2, "memory": 1, "pcie": 2}

        def drain_incrementally(
            kernel: UnifiedEventKernel,
            items: Sequence[TaskSpec],
        ) -> Tuple[KernelEvent, ...]:
            events = []
            while kernel.has_active_tasks:
                event = kernel.step()
                self.assertIsNotNone(event)
                events.append(event)
            return tuple(events)

        def normalized(value: object) -> object:
            if isinstance(value, float):
                return value.hex()
            if isinstance(value, Mapping):
                return tuple(
                    (key, normalized(item))
                    for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
                )
            if isinstance(value, (list, tuple)):
                return tuple(normalized(item) for item in value)
            return value

        def event_snapshot(events: Sequence[KernelEvent]) -> object:
            return normalized(
                tuple(
                    {
                        "task_id": event.task.task_id,
                        "start_ns": event.start_ns,
                        "end_ns": event.end_ns,
                        "dependency_ready_ns": event.dependency_ready_ns,
                        "effective_ready_ns": event.effective_ready_ns,
                        "demands": tuple(
                            (item.resource_id, item.service_ns)
                            for item in event.demands
                        ),
                        "resource_predecessors": event.resource_predecessors,
                    }
                    for event in events
                )
            )

        def persistent_snapshot(kernel: UnifiedEventKernel) -> object:
            return normalized(
                {
                    "active_task_count": kernel.active_task_count,
                    "completed_end_ns": kernel.completed_end_ns,
                    "completion_leases": kernel.completion_leases,
                    "dependents": {
                        task_id: tuple(dependents)
                        for task_id, dependents in kernel._dependents.items()
                    },
                    "metrics": kernel.metrics,
                    "resource_available": kernel.resource_available,
                    "resource_last_interval": kernel.resource_last_interval,
                    "resource_lane_available": kernel._resource_lane_available,
                    "resource_lane_last_interval": (
                        kernel._resource_lane_last_interval
                    ),
                    "resource_busy_ns": kernel.resource_busy_ns,
                    "resource_queue_wait_ns": kernel.resource_queue_wait_ns,
                    "resource_task_count": kernel.resource_task_count,
                    "seen_ids": tuple(sorted(kernel._seen_ids)),
                }
            )

        warmup = (
            task("warm.dma-a", demands=(demand("dma", 1.25),)),
            task("warm.dma-b", demands=(demand("dma", 3.75),)),
            task("warm.gpu", demands=(demand("gpu", 2.5),)),
            task(
                "warm.join",
                dependencies=("warm.dma-a", "warm.dma-b", "warm.gpu"),
                demands=(demand("pcie", 0.625),),
            ),
        )
        ordinary = UnifiedEventKernel(resource_capacities=capacities)
        bulk = UnifiedEventKernel(resource_capacities=capacities)
        for kernel in (ordinary, bulk):
            kernel.submit(warmup)
            drain_incrementally(kernel, warmup)
            kernel.retain_completed("warm.join")
        self.assertEqual(persistent_snapshot(bulk), persistent_snapshot(ordinary))

        resources = tuple(capacities)
        generated = []
        for position in range(48):
            task_id = "live.{:02d}".format(position)
            if position < 5:
                dependencies = ()
            else:
                dependency_count = rng.randint(0, min(3, position))
                dependencies = tuple(
                    sorted(
                        "live.{:02d}".format(index)
                        for index in rng.sample(
                            range(position), dependency_count
                        )
                    )
                )
            demand_count = rng.randint(0, 3)
            demand_resources = rng.sample(resources, demand_count)
            rng.shuffle(demand_resources)
            generated.append(
                task(
                    task_id,
                    dependencies=dependencies,
                    demands=tuple(
                        demand(
                            resource_id,
                            (rng.randint(1, 41) / 8.0),
                        )
                        for resource_id in demand_resources
                    ),
                    earliest_start_ns=rng.randint(0, 31) / 8.0,
                )
            )
        rng.shuffle(generated)
        live = tuple(generated)
        layout = CompiledGraphLayout.compile(live)

        ordinary.submit_compiled(live, layout)
        ordinary_events = drain_incrementally(ordinary, live)
        bulk_events = bulk._drain_prevalidated_compiled(live, layout)

        self.assertEqual(event_snapshot(bulk_events), event_snapshot(ordinary_events))
        self.assertEqual(persistent_snapshot(bulk), persistent_snapshot(ordinary))

        release_ids = tuple(event.task.task_id for event in ordinary_events)
        for task_id in release_ids:
            self.assertEqual(
                bulk.release_completed(task_id),
                ordinary.release_completed(task_id),
            )
        self.assertEqual(
            bulk.release_completed("warm.join"),
            ordinary.release_completed("warm.join"),
        )
        self.assertEqual(persistent_snapshot(bulk), persistent_snapshot(ordinary))

        followup = (
            task(
                "followup",
                demands=(demand("dma", 0.375), demand("gpu", 0.875)),
                earliest_start_ns=4.25,
            ),
        )
        for kernel in (ordinary, bulk):
            kernel.submit(followup)
        ordinary_followup = drain_incrementally(ordinary, followup)
        bulk_followup = drain_incrementally(bulk, followup)
        self.assertEqual(
            event_snapshot(bulk_followup),
            event_snapshot(ordinary_followup),
        )
        self.assertEqual(persistent_snapshot(bulk), persistent_snapshot(ordinary))

    def test_compiled_append_matches_ordinary_dynamic_submission(self) -> None:
        template = (
            task("root", demands=(demand("dma", 1.0),)),
            task(
                "tail",
                dependencies=("root",),
                demands=(demand("gpu", 1.0),),
            ),
        )
        layout = CompiledGraphLayout.compile(template)
        live = (
            task(
                "cohort.root",
                demands=(demand("dma", 5.0),),
                earliest_start_ns=7.0,
            ),
            task(
                "cohort.tail",
                dependencies=("cohort.root",),
                demands=(demand("gpu", 3.0),),
                earliest_start_ns=7.0,
            ),
        )
        ordinary = UnifiedEventKernel(resource_capacities={"dma": 2})
        ordinary.add_tasks(live)
        compiled = UnifiedEventKernel(resource_capacities={"dma": 2})
        receipt = compiled.submit_compiled(live, layout)

        ordinary_events = tuple(ordinary.step() for _task in live)
        compiled_events = tuple(compiled.step() for _task in live)
        self.assertEqual(compiled_events, ordinary_events)
        self.assertEqual(compiled.metrics, ordinary.metrics)
        self.assertEqual(receipt.task_ids, tuple(item.task_id for item in live))

    def test_compiled_append_rejects_structure_mismatch_atomically(self) -> None:
        template = (task("root", demands=(demand("dma", 1.0),)),)
        layout = CompiledGraphLayout.compile(template)
        kernel = UnifiedEventKernel()
        kernel.add_tasks((task("existing"),))
        before_metrics = dict(kernel.metrics)

        with self.assertRaisesRegex(
            ValueError, "task graph does not match compiled structure"
        ):
            kernel.submit_compiled(
                (task("root", demands=(demand("other", 1.0),)),),
                layout,
            )

        self.assertEqual(dict(kernel.metrics), before_metrics)
        self.assertEqual(kernel.active_task_count, 1)
        self.assertEqual(kernel.step().task.task_id, "existing")

    def test_compiled_append_rejects_malformed_layout_atomically(self) -> None:
        template = (
            task("root", demands=(demand("dma", 1.0),)),
            task(
                "tail",
                dependencies=("root",),
                demands=(demand("gpu", 1.0),),
            ),
        )
        layout = CompiledGraphLayout.compile(template)
        malformed_layouts = (
            replace(layout, resource_groups=(layout.resource_groups[0],)),
            replace(layout, demand_order=((1,), (0,))),
            replace(layout, root_positions=(0, 2)),
            replace(layout, dependent_positions=((), (99,))),
        )
        live = (
            task("cohort.root", demands=(demand("dma", 5.0),)),
            task(
                "cohort.tail",
                dependencies=("cohort.root",),
                demands=(demand("gpu", 3.0),),
            ),
        )
        kernel = UnifiedEventKernel()
        kernel.add_tasks((task("existing"),))
        before_seen = set(kernel._seen_ids)
        before_dependents = {
            task_id: list(dependents)
            for task_id, dependents in kernel._dependents.items()
        }

        for malformed_layout in malformed_layouts:
            with self.subTest(layout=malformed_layout):
                with self.assertRaisesRegex(
                    ValueError, "task graph does not match compiled structure"
                ):
                    kernel.submit_compiled(live, malformed_layout)
                self.assertEqual(kernel.active_task_count, 1)
                self.assertEqual(kernel._seen_ids, before_seen)
                self.assertEqual(
                    kernel._dependents,
                    before_dependents,
                )

    def test_compact_seen_namespace_rejects_public_resubmit_and_allows_fresh(
        self,
    ) -> None:
        kernel = UnifiedEventKernel()
        base_task_ids = ("root", "tail")
        kernel._register_compact_seen_namespace(
            "history.",
            base_task_ids,
        )

        self.assertIs(kernel._compact_seen_namespaces["history."], base_task_ids)
        self.assertEqual(kernel._seen_ids, set())
        before_seen = set(kernel._seen_ids)
        before_namespaces = dict(kernel._compact_seen_namespaces)
        before_metrics = dict(kernel.metrics)

        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id: history\\.root",
        ):
            kernel.add_tasks((task("history.root"),))

        self.assertEqual(kernel.active_task_count, 0)
        self.assertEqual(kernel._seen_ids, before_seen)
        self.assertEqual(kernel._compact_seen_namespaces, before_namespaces)
        self.assertEqual(dict(kernel.metrics), before_metrics)

        receipt = kernel.submit((task("history.fresh"),))
        self.assertEqual(receipt.task_ids, ("history.fresh",))
        self.assertEqual(kernel.step().task.task_id, "history.fresh")
        self.assertIsNone(kernel.step())

    def test_compact_seen_namespace_reaches_compiled_and_prevalidated_paths(
        self,
    ) -> None:
        template = (
            task("root", demands=(demand("dma", 1.0),)),
            task(
                "tail",
                dependencies=("root",),
                demands=(demand("gpu", 1.0),),
            ),
        )
        layout = CompiledGraphLayout.compile(template)

        compiled = UnifiedEventKernel()
        compiled._register_compact_seen_namespace(
            "compiled.",
            ("root", "tail"),
        )
        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id: compiled\\.root",
        ):
            compiled.submit_compiled(
                (
                    task("compiled.root", demands=(demand("dma", 1.0),)),
                    task(
                        "compiled.tail",
                        dependencies=("compiled.root",),
                        demands=(demand("gpu", 1.0),),
                    ),
                ),
                layout,
            )
        self.assertEqual(compiled.active_task_count, 0)
        self.assertEqual(compiled._seen_ids, set())

        prevalidated = UnifiedEventKernel()
        prevalidated._register_compact_seen_namespace(
            "prevalidated.",
            ("root", "tail"),
        )
        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id: prevalidated\\.root",
        ):
            prevalidated._drain_prevalidated_compiled(
                (
                    task("prevalidated.root", demands=(demand("dma", 1.0),)),
                    task(
                        "prevalidated.tail",
                        dependencies=("prevalidated.root",),
                        demands=(demand("gpu", 1.0),),
                    ),
                ),
                layout,
            )
        self.assertEqual(prevalidated.active_task_count, 0)
        self.assertEqual(prevalidated._seen_ids, set())

    def test_compact_seen_namespace_registration_is_fail_closed(self) -> None:
        kernel = UnifiedEventKernel()
        kernel.submit((task("plain.known"),))
        before_seen = set(kernel._seen_ids)
        before_namespaces = dict(kernel._compact_seen_namespaces)
        before_metrics = dict(kernel.metrics)

        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id: plain\\.known",
        ):
            kernel._register_compact_seen_namespace(
                "plain.",
                ("known", "fresh"),
            )

        self.assertEqual(kernel._seen_ids, before_seen)
        self.assertEqual(kernel._compact_seen_namespaces, before_namespaces)
        self.assertEqual(dict(kernel.metrics), before_metrics)
        self.assertEqual(kernel.active_task_count, 1)

        kernel._register_compact_seen_namespace(
            "compact.",
            ("one", "two"),
        )
        registered_namespaces = dict(kernel._compact_seen_namespaces)

        with self.assertRaisesRegex(
            ValueError,
            "compact seen namespace already registered: compact\\.",
        ):
            kernel._register_compact_seen_namespace(
                "compact.",
                ("three",),
            )
        self.assertEqual(kernel._compact_seen_namespaces, registered_namespaces)

        with self.assertRaisesRegex(
            ValueError,
            "compact seen namespace contains duplicate task ids",
        ):
            kernel._register_compact_seen_namespace(
                "other.",
                ("same", "same"),
            )
        self.assertNotIn("other.", kernel._compact_seen_namespaces)

        shared = UnifiedEventKernel()
        shared_base_ids = ("root", "tail")
        shared._register_compact_seen_namespace(
            "shared.a.",
            shared_base_ids,
        )
        shared._register_compact_seen_namespace(
            "shared.b.",
            shared_base_ids,
        )
        self.assertIs(
            shared._compact_seen_namespaces["shared.a."],
            shared_base_ids,
        )
        self.assertIs(
            shared._compact_seen_namespaces["shared.b."],
            shared_base_ids,
        )
        self.assertIs(
            shared._compact_seen_base_id_sets["shared.a."],
            shared._compact_seen_base_id_sets["shared.b."],
        )

        parent_prefix = UnifiedEventKernel()
        parent_prefix._register_compact_seen_namespace(
            "overlap.",
            ("root.tail",),
        )
        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id: overlap\\.root\\.tail",
        ):
            parent_prefix._register_compact_seen_namespace(
                "overlap.root.",
                ("tail",),
            )
        self.assertNotIn("overlap.root.", parent_prefix._compact_seen_namespaces)

        child_prefix = UnifiedEventKernel()
        child_prefix._register_compact_seen_namespace(
            "overlap.root.",
            ("tail",),
        )
        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id: overlap\\.root\\.tail",
        ):
            child_prefix._register_compact_seen_namespace(
                "overlap.",
                ("root.tail",),
            )
        self.assertNotIn("overlap.", child_prefix._compact_seen_namespaces)

    def test_coherent_dma_demands_finish_at_the_slowest_resource(self) -> None:
        copy = task(
            "copy",
            demands=(
                demand("host.ddr.read", 8.0),
                demand("pcie", 12.0),
                demand("gpu.vram.write", 3.0),
            ),
        )
        gemm = task(
            "gemm",
            dependencies=("copy",),
            demands=(demand("gpu.compute", 5.0),),
        )

        kernel = UnifiedEventKernel.from_closed_graph((gemm, copy))
        copy_event = kernel.step()
        gemm_event = kernel.step()

        self.assertEqual(copy_event.start_ns, 0.0)
        self.assertEqual(copy_event.end_ns, 12.0)
        self.assertEqual(kernel.resource_available["host.ddr.read"], 8.0)
        self.assertEqual(kernel.resource_available["pcie"], 12.0)
        self.assertEqual(kernel.resource_available["gpu.vram.write"], 3.0)
        self.assertEqual(gemm_event.start_ns, copy_event.end_ns)
        self.assertEqual(gemm_event.end_ns, 17.0)

    def test_task_start_ns_matches_original_expression(self) -> None:
        cases = (
            ((), 7.0, {}),
            ((demand("missing", 1.0),), 2.0, {}),
            (
                (demand("first", 1.0), demand("second", 1.0)),
                3.0,
                {"first": 11.0, "second": 5.0},
            ),
            (
                (demand("first", 1.0), demand("second", 1.0)),
                3.0,
                {"first": 5.0, "second": 11.0},
            ),
        )
        for demands, effective_ready_ns, resource_available in cases:
            with self.subTest(
                demands=tuple(item.resource_id for item in demands),
                resource_available=resource_available,
            ):
                self.assertEqual(
                    task_start_ns(
                        demands,
                        effective_ready_ns,
                        resource_available,
                    ),
                    reference_task_start_ns(
                        demands,
                        effective_ready_ns,
                        resource_available,
                    ),
                )

    def test_hot_loop_preserves_full_event_trace_and_timing(self) -> None:
        tasks = (
            task("a-gpu", demands=(demand("gpu", 2.0),)),
            task(
                "b-gpu-later",
                demands=(demand("gpu", 1.0),),
                earliest_start_ns=1.0,
            ),
            task("c-link", demands=(demand("link", 3.0),)),
            task("d-memory", demands=(demand("memory", 5.0),)),
            task("e-release", earliest_start_ns=1.5),
            task(
                "f-zero",
                dependencies=("e-release",),
                demands=(demand("gpu", 0.0),),
            ),
            task(
                "g-multi",
                dependencies=("b-gpu-later", "e-release"),
                demands=(demand("memory", 4.0), demand("link", 2.0)),
                earliest_start_ns=2.0,
            ),
            task("h-join", dependencies=("f-zero", "g-multi")),
            task(
                "i-tail",
                dependencies=("h-join",),
                demands=(demand("memory", 1.0),),
            ),
        )
        (
            expected_events,
            expected_completed_end,
            expected_resource_available,
            expected_last_interval,
            expected_resource_busy_ns,
        ) = reference_schedule(tasks)

        kernel = UnifiedEventKernel.from_closed_graph(tuple(reversed(tasks)))
        actual_events = []
        for expected_event in expected_events:
            expected_ready_key = (
                expected_event.start_ns,
                expected_event.effective_ready_ns,
                expected_event.dependency_ready_ns,
                expected_event.task.earliest_start_ns,
                expected_event.task.task_id,
            )
            self.assertEqual(kernel.peek_ready_key(), expected_ready_key)
            actual_event = kernel.step()
            self.assertEqual(actual_event, expected_event)
            actual_events.append(actual_event)

        self.assertEqual(actual_events, expected_events)
        self.assertIsNone(kernel.peek_ready_key())
        self.assertIsNone(kernel.step())
        kernel.assert_drained()
        self.assertEqual(kernel.completed_count, len(expected_events))
        self.assertEqual(kernel.completed_end_ns, expected_completed_end)
        self.assertEqual(kernel.resource_available, expected_resource_available)
        self.assertEqual(kernel.resource_last_interval, expected_last_interval)
        self.assertEqual(kernel.resource_busy_ns, expected_resource_busy_ns)
        self.assertEqual(
            kernel.makespan_ns,
            max(expected_completed_end.values(), default=0.0),
        )


if __name__ == "__main__":
    unittest.main()
