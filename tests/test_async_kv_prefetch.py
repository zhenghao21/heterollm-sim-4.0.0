"""Asynchronous physical KV copies, reservation safety and consumer readiness."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from heterollm_sim.ir import RequestSpec
from heterollm_sim.kv_pool import KvPoolError
from heterollm_sim.serving import BatchCost, RequestStatus, _OnlineRuntime, compile_serving_plan


_SPEC = importlib.util.spec_from_file_location(
    "async_prefetch_fixture", Path(__file__).resolve().parents[1] / "scripts" / "accept_hbf_hbm.py")
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)
PAGE = 16384


def runtime(*, distance=17, hbm_pages=32, high_output=8, cohort_ns=1000.0):
    scenario = _FIXTURE._serving_paged_scenario(
        hbf_capacity=16 * PAGE, hbm_capacity=hbm_pages * PAGE,
        prefetch_distance=distance,
        requests=(RequestSpec("low", 0.0, 32, 8, priority=0),
                  RequestSpec("high", 1.0, 32, high_output, priority=10)))
    return _OnlineRuntime(compile_serving_plan(scenario),
                          lambda *_: BatchCost(cohort_ns))


def offloaded_runtime(**kwargs):
    runner = runtime(**kwargs)
    low = runner.states["low"]
    assert runner.ledger.resize(low, 2)
    assert runner.ledger.offload(low)
    runner._sync_paged_pool_events()
    runner._set_status(low, RequestStatus.SWAPPED)
    low.preemption_strategy = "swap"
    return runner, low


def finish_prefetch(runner):
    while runner._pending_prefetches:
        runner.now = runner._next_prefetch_completion()
        runner._complete_prefetches()


class AsyncKVPrefetchTests(unittest.TestCase):
    def test_submit_keeps_source_and_reserves_destination_until_ready(self):
        runner, low = offloaded_runtime()
        pool = runner.ledger.paged_pool
        before = dict(runner.physical_ledger.used_bytes)
        self.assertEqual(runner.physical_ledger.peak_used_bytes["hbf0"], 2 * PAGE)
        clocks = (runner.now, runner._host_available_ns, runner._gpu_available_ns)
        executor = runner._paged_pool_transfer_executor
        target = executor.runtime.runtime(executor._configs["hbm0"], executor.services["hbm0"].physical_owner)
        target_clock = target.core._acceptance_ns
        self.assertTrue(runner._prefetch_waiting())
        self.assertEqual(target.core._acceptance_ns, target_clock)
        self.assertEqual(clocks, (runner.now, runner._host_available_ns, runner._gpu_available_ns))
        self.assertEqual(runner.physical_ledger.used_bytes["hbf0"], before["hbf0"])
        self.assertEqual(runner.physical_ledger.used_bytes["hbm0"], before["hbm0"] + 2 * PAGE)
        self.assertGreaterEqual(runner.physical_ledger.peak_used_bytes["hbm0"], 2 * PAGE)
        self.assertTrue(all(page.owner_component == "hbf0" for page in pool.request_pages("low")))
        self.assertFalse(runner._resume_swap(low))
        with self.assertRaises(KvPoolError):
            pool.cache_accesses("low", first_token=0, token_count=16)
        self.assertFalse(pool.resize_detailed("low", 0).success)
        entry = runner._pending_prefetches["low"]
        page_id, plan = entry["prepared"][0]
        with self.assertRaises(KvPoolError):
            pool.complete_prepared_restore(page_id, plan, now_ns=plan.completion_ns - 1)
        runner.now = entry["completion_ns"]
        runner._complete_prefetches()
        self.assertTrue(runner._pending_prefetches)
        self.assertFalse(runner._resume_swap(low))
        finish_prefetch(runner)
        self.assertTrue(all(page.owner_component == "hbm0" for page in pool.request_pages("low")))
        self.assertEqual(runner.physical_ledger.used_bytes["hbf0"], before["hbf0"] - 2 * PAGE)
        self.assertEqual(low.kv_pages, 2)
        self.assertEqual(low.swapped_pages, 0)
        self.assertTrue(pool.cache_accesses("low", first_token=0, token_count=16))

    def test_copy_overlaps_independent_decode_and_consumer_waits(self):
        runner = runtime()
        result = runner.run()
        start = next(e for e in result.events if e.event_type == "kv_prefetch_start")
        done = next(e for e in result.events if e.event_type == "kv_prefetch")
        overlapping = [b for b in result.batches if "high" in b.request_ids
                       and any(item.phase == "decode" for item in b.items)
                       and b.start_ns < done.timestamp_ns and b.end_ns > start.timestamp_ns]
        self.assertTrue(overlapping)
        resumed = next(e for e in result.events if e.event_type == "request_resumed" and e.request_id == "low")
        self.assertGreaterEqual(resumed.timestamp_ns, done.timestamp_ns)
        self.assertTrue(all(b.start_ns >= done.timestamp_ns for b in result.batches
                            if "low" in b.request_ids and b.start_ns > start.timestamp_ns))
        self.assertTrue(all(m.status == RequestStatus.FINISHED for m in result.request_metrics.values()))
        self.assertEqual(result.kv_metrics.prefetch_completed_bytes, 2 * PAGE)
        self.assertEqual(result.kv_metrics.prefetch_useful_bytes, 2 * PAGE)
        self.assertEqual(result.kv_metrics.prefetch_inflight_bytes, 0)
        self.assertFalse(runner.ledger.paged_pool._pending_moves)
        self.assertTrue(all(value == 0 for value in runner.physical_ledger.used_bytes.values()))
        baseline = runtime(distance=0).run()
        self.assertLess(result.makespan_ns, baseline.makespan_ns)

    def test_short_independent_work_wakes_at_copy_completion(self):
        result = runtime(high_output=2).run()
        done = next(e for e in result.events if e.event_type == "kv_prefetch")
        high = result.request_metrics["high"]
        self.assertLess(high.finish_ns, done.timestamp_ns)
        resumed = next(e for e in result.events if e.event_type == "request_resumed" and e.request_id == "low")
        self.assertGreaterEqual(resumed.timestamp_ns, done.timestamp_ns)
        self.assertEqual(result.request_metrics["low"].status, RequestStatus.FINISHED)

    def test_partial_lookahead_restores_remaining_pages_on_demand(self):
        result = runtime(distance=1).run()
        starts = [e for e in result.events if e.event_type == "kv_prefetch_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].details["pages"], 1)
        self.assertEqual(result.kv_metrics.prefetch_completed_bytes, PAGE)
        self.assertGreater(result.kv_metrics.swap_in_bytes, 0)
        self.assertTrue(all(m.status == RequestStatus.FINISHED for m in result.request_metrics.values()))

    def test_full_target_does_not_start_or_leak_a_copy(self):
        runner, low = offloaded_runtime(hbm_pages=2)
        owner = runner.states["high"]
        self.assertTrue(runner.ledger.resize(owner, 2))
        before = dict(runner.physical_ledger.used_bytes)
        self.assertFalse(runner._prefetch_waiting())
        self.assertEqual(before, runner.physical_ledger.used_bytes)
        self.assertFalse(runner._pending_prefetches)
        self.assertEqual(low.swapped_pages, 2)

    def test_failed_second_page_rolls_back_reservations_in_reverse_order(self):
        runner, low = offloaded_runtime()
        pool = runner.ledger.paged_pool
        before = dict(runner.physical_ledger.used_bytes)
        physical_before = runner._paged_pool_transfer_executor.snapshot()
        original = pool.prepare_restore_page
        calls = []
        def prepare(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs) if len(calls) == 1 else None
        with patch.object(pool, "prepare_restore_page", side_effect=prepare):
            self.assertFalse(runner._prefetch_waiting())
        self.assertEqual(before, runner.physical_ledger.used_bytes)
        self.assertEqual(physical_before, runner._paged_pool_transfer_executor.snapshot())
        self.assertFalse(pool._pending_moves)
        self.assertEqual(low.swapped_pages, 2)

    def test_prefetch_preserves_running_request_growth_headroom(self):
        runner, _low = offloaded_runtime(hbm_pages=4)
        owner = runner.states["high"]
        self.assertTrue(runner.ledger.resize(owner, 2))
        runner._set_status(owner, RequestStatus.RUNNING)
        # The owner needs a third page as output grows; a two-page copy would
        # fit now but would prevent its next page allocation.
        self.assertFalse(runner._prefetch_waiting())
        self.assertTrue(runner.ledger.resize(owner, 3))

    def test_later_capacity_pressure_reclaims_completed_unused_copy(self):
        runner, low = offloaded_runtime(hbm_pages=4)
        self.assertTrue(runner._prefetch_waiting())
        finish_prefetch(runner)
        owner = runner.states["high"]
        runner._set_status(owner, RequestStatus.RUNNING)
        self.assertTrue(runner._reserve_with_pressure(owner, 3))
        self.assertEqual(low.kv_pages, 0)
        self.assertEqual(low.swapped_pages, 2)
        self.assertEqual(runner.prefetch_reclaim_bytes, 2 * PAGE)
        self.assertEqual(runner._prefetch_discarded_bytes, 2 * PAGE)
        self.assertEqual(runner.physical_ledger.used_bytes["hbm0"], 3 * PAGE)
        self.assertEqual(runner.physical_ledger.used_bytes["hbf0"], 2 * PAGE)
        self.assertFalse(runner._prefetch_waiting())

    def test_real_full_serving_overlaps_physical_copy_with_independent_kernel(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from hbf_hbm_serving_study import build_study_scenario
        from heterollm_sim.reporting import run_scenario
        from hbf_hbm_metrics import prefetch_overlap
        scenario = build_study_scenario("C_hot_cold", prompt=128, output=8, batch=1,
                                        request_count=2, interval_ns=30000.0,
                                        priority_preemption=True, hbf_page_read_ns=75000.)
        scenario = replace(scenario, placement=replace(scenario.placement,
            kv_policy=replace(scenario.placement.kv_policy, prefetch_distance=17)))
        result = run_scenario(scenario).serving
        start = next(e for e in result.events if e.event_type == "kv_prefetch_start")
        done = next(e for e in result.events if e.event_type == "kv_prefetch")
        overlap = prefetch_overlap(result, {"gpu0"})
        self.assertGreater(overlap["operator_overlap_ns"], 0)
        self.assertGreater(overlap["decode_operator_overlap_ns"], 0)
        self.assertEqual(overlap["premature_consumer_overlap_ns"], 0)
        resumed = next(e for e in result.events if e.event_type == "request_resumed"
                       and e.request_id == start.request_id)
        self.assertGreaterEqual(resumed.timestamp_ns, done.timestamp_ns)
        self.assertTrue(all(m.status == RequestStatus.FINISHED for m in result.request_metrics.values()))
        self.assertGreater(result.kv_metrics.prefetch_useful_bytes, 0)
        self.assertGreater(result.kv_metrics.peak_bytes_by_component["hbf0"], 0)

    def test_deferred_target_write_still_waits_for_shared_hbm_traffic(self):
        from heterollm_sim.data_motion import AccessKind
        runner, _low = offloaded_runtime(distance=1)
        self.assertTrue(runner._prefetch_waiting())
        executor = runner._paged_pool_transfer_executor
        entry = runner._pending_prefetches["low"]
        hops = [hop for hop in executor._route("hbf0", "hbm0", PAGE)
                if not getattr(hop, "service_ref", None)]
        for _hop in hops:
            runner._advance_prefetches_until(entry["completion_ns"])
        target_arrival = entry["completion_ns"]
        target = executor.services["hbm0"]
        # An independent request is admitted before the copy reaches HBM.
        # Its real shared channel reservations must survive copy advancement.
        competing = target.price(AccessKind.READ, 16 * PAGE, page_offset_bytes=0,
                                  runtime=executor.runtime, arrival_ns=target_arrival - 1.)
        finish_prefetch(runner)
        copied = executor.events[-1]
        final = copied["phases"][-1]
        self.assertEqual(final["kind"], "target_write")
        self.assertGreater(final["completion_ns"], competing["completion_ns"])
        baseline, _ = offloaded_runtime(distance=1)
        self.assertTrue(baseline._prefetch_waiting())
        finish_prefetch(baseline)
        self.assertGreater(final["completion_ns"],
            baseline._paged_pool_transfer_executor.events[-1]["completion_ns"])

    def test_ordinary_link_task_observes_copy_resource_reservation(self):
        from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
        runner, _low = offloaded_runtime(distance=1)
        self.assertTrue(runner._prefetch_waiting())
        executor = runner._paged_pool_transfer_executor
        entry = runner._pending_prefetches["low"]
        first_hop = next(hop for hop in executor._route("hbf0", "hbm0", PAGE)
                         if not getattr(hop, "service_ref", None))
        source_end = entry["completion_ns"]
        kernel = runner._execution_resource_kernel
        kernel.submit((TaskSpec(task_id="independent-link", request_id="high", name="competing link",
            category=TaskCategory.COMMUNICATION, earliest_start_ns=source_end,
            demands=(ResourceDemand(first_hop.resource_id, 123.),)),))
        runner._advance_prefetches_before_kernel_step(kernel)
        event = kernel.step()
        wire_busy = next(d.service_ns for d in first_hop.demands(PAGE)
                         if d.resource_id == first_hop.resource_id)
        self.assertGreaterEqual(event.start_ns, source_end + wire_busy)
        finish_prefetch(runner)


if __name__ == "__main__":
    unittest.main()
