"""Acceptance tests for V4 KV prefetch distance semantics."""

import importlib.util
from dataclasses import replace
from pathlib import Path
import unittest

from heterollm_sim.ir import RequestSpec
from heterollm_sim.planner import compile_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import _prefetch_semantics
from heterollm_sim.serving import BatchCost, _OnlineRuntime, compile_serving_plan


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accept_hbf_hbm.py"
_SPEC = importlib.util.spec_from_file_location("accept_hbf_hbm_prefetch", _SCRIPT)
_acceptance = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_acceptance)

_PAGE = 16_384


def _scenario(distance: int):
    requests = (
        RequestSpec(
            request_id="prefetch-low",
            arrival_ns=0.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=0,
        ),
        RequestSpec(
            request_id="prefetch-high",
            arrival_ns=1.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=10,
        ),
    )
    scenario = _acceptance._serving_paged_scenario(
        hbf_capacity=2 * _PAGE, requests=requests
    )
    policy = replace(scenario.placement.kv_policy, prefetch_distance=distance)
    return replace(
        scenario,
        placement=replace(scenario.placement, kv_policy=policy),
    )


def _run(distance: int):
    scenario = _scenario(distance)
    plan = compile_serving_plan(scenario)
    runtime = _OnlineRuntime(
        plan, lambda *_: BatchCost(1.0)
    )
    return plan, runtime, runtime.run()


class PrefetchDistanceAcceptanceTests(unittest.TestCase):
    def test_nonzero_distance_is_accepted_and_modeled(self):
        plan, _runtime, result = _run(17)
        self.assertEqual(plan.kv_policy.prefetch_distance, 17)
        self.assertTrue(result.kv_metrics.prefetch_distance_modeled)
        self.assertGreaterEqual(result.kv_metrics.prefetch_events, 1)
        self.assertGreater(result.kv_metrics.prefetch_bytes, 0)
        self.assertGreater(result.kv_metrics.swap_transfer_time_ns, 0.0)
        self.assertGreater(result.kv_metrics.prefetch_transfer_time_ns, 0.0)
        self.assertEqual(result.kv_metrics.prefetch_distance_tokens, 17)
        self.assertEqual(result.kv_metrics.prefetch_lookahead_pages, 2)
        events = [
            event
            for event in result.events
            if event.event_type == "kv_prefetch"
            or event.event_type.startswith("kv_prefetch_")
        ]
        self.assertTrue(events)
        details = events[0].details
        self.assertEqual(details.get("distance_tokens"), 17)
        self.assertEqual(details.get("lookahead_pages"), 2)
        self.assertEqual(details.get("bytes"), 2 * _PAGE)

    def test_zero_distance_keeps_prefetch_disabled(self):
        plan, _runtime, result = _run(0)
        self.assertEqual(plan.kv_policy.prefetch_distance, 0)
        self.assertFalse(result.kv_metrics.prefetch_distance_modeled)
        self.assertEqual(result.kv_metrics.prefetch_events, 0)
        self.assertEqual(result.kv_metrics.prefetch_bytes, 0)
        self.assertFalse(
            any(
                event.event_type == "kv_prefetch"
                or event.event_type.startswith("kv_prefetch_")
                for event in result.events
            )
        )

    def test_negative_distance_is_rejected(self):
        with self.assertRaises(ValueError):
            compile_serving_plan(_scenario(-1))

    def test_static_planner_emits_prefetch_tasks(self):
        # The offline V4 planner has no page identities, so it records a
        # bounded token lookahead on the generated transfer task. This keeps
        # static and serving paths on the same prefetch contract.
        scenario = build_reference_scenario()
        policy = replace(scenario.placement.kv_policy, prefetch_distance=17)
        scenario = replace(
            scenario, placement=replace(scenario.placement, kv_policy=policy)
        )
        schedule = compile_scenario(scenario)
        tasks = [
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "kv_prefetch"
        ]
        self.assertTrue(tasks)
        self.assertTrue(
            all(task.metadata.get("kv_prefetch_distance") == 17 for task in tasks)
        )
        self.assertTrue(
            all(task.metadata.get("lookahead_tokens") == 17 for task in tasks)
        )
        self.assertTrue(all(int(task.metadata.get("bytes", 0)) > 0 for task in tasks))

    def test_capacity_failure_is_atomic(self):
        # Build a fully offloaded two-page victim, then occupy all active HBM
        # pages with another request. A distance-based prefetch must fail as
        # one operation and leave owners, request counters, and physical bytes
        # unchanged.
        scenario = _acceptance._serving_paged_scenario(
            hbf_capacity=2 * _PAGE,
            hbm_capacity=2 * _PAGE,
            requests=(
                RequestSpec(
                    request_id="prefetch-victim",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
                RequestSpec(
                    request_id="prefetch-owner",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
            ),
            prefetch_distance=17,
        )
        plan = compile_serving_plan(scenario)
        runtime = _OnlineRuntime(
            plan, lambda *_: BatchCost(1.0)
        )
        victim = runtime.states["prefetch-victim"]
        owner = runtime.states["prefetch-owner"]
        pool = runtime.ledger.paged_pool
        self.assertIsNotNone(pool)
        self.assertTrue(runtime.ledger.resize(victim, 2))
        self.assertTrue(runtime.ledger.offload(victim))
        self.assertTrue(runtime.ledger.resize(owner, 2))
        before_owners = tuple(
            page.owner_component for page in pool.request_pages(victim.spec.request_id)
        )
        before_physical = dict(runtime.physical_ledger.used_bytes)
        moved = runtime.ledger.prefetch(victim, 2)
        self.assertEqual(moved, 0)
        self.assertEqual(
            tuple(page.owner_component for page in pool.request_pages(victim.spec.request_id)),
            before_owners,
        )
        self.assertEqual(victim.kv_pages, 0)
        self.assertEqual(victim.swapped_pages, 2)
        self.assertEqual(victim.swap_bytes, 2 * _PAGE)
        self.assertEqual(runtime.physical_ledger.used_bytes, before_physical)

    def test_reporting_distinguishes_online_and_static_prefetch_contracts(self):
        paged = _scenario(17)
        online = _prefetch_semantics(paged, execution_mode="online")
        static = _prefetch_semantics(paged, execution_mode="static")
        self.assertTrue(online["modeled"])
        self.assertEqual(online["mode"], "page_accurate")
        self.assertTrue(static["modeled"])
        self.assertEqual(static["mode"], "analytical_only")

        legacy = build_reference_scenario()
        policy = replace(legacy.placement.kv_policy, prefetch_distance=17)
        legacy = replace(
            legacy, placement=replace(legacy.placement, kv_policy=policy)
        )
        legacy_online = _prefetch_semantics(legacy, execution_mode="online")
        self.assertFalse(legacy_online["modeled"])
        self.assertEqual(legacy_online["mode"], "unsupported_layout")


if __name__ == "__main__":
    unittest.main()
