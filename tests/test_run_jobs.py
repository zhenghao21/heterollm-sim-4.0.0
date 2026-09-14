import json
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import replay_online_batch_trace
from heterollm_sim.planner import (
    compile_scenario,
    materialize_requests,
    validate_scenario,
)
from heterollm_sim.run_estimation import estimate_scenario
from heterollm_sim.run_jobs import (
    RunJobCapacityError,
    RunJobManager,
    RunJobTraceError,
)
from tests.model_helpers import execution_layers


def wait_for(manager, job_id, statuses, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.get(job_id)
        if snapshot is not None and snapshot["status"] in statuses:
            return snapshot
        time.sleep(0.005)
    raise AssertionError("job did not reach {}".format(sorted(statuses)))


class RunEstimationTests(unittest.TestCase):
    def test_estimate_is_json_safe_and_recommends_continuous_backend(self):
        scenario = build_reference_scenario()
        estimate = estimate_scenario(scenario)

        json.dumps(estimate, ensure_ascii=False)
        self.assertEqual(estimate["request_count"], 1)
        expected_prompt = sum(
            request.prompt_tokens for request in scenario.workload.requests
        )
        expected_output = sum(
            request.output_tokens for request in scenario.workload.requests
        )
        self.assertEqual(estimate["prompt_tokens"], expected_prompt)
        self.assertEqual(estimate["output_tokens"], expected_output)
        self.assertEqual(
            estimate["total_tokens"], expected_prompt + expected_output
        )
        self.assertEqual(
            estimate["layer_count"],
            len(execution_layers(scenario.model)),
        )
        self.assertEqual(
            estimate["world_size"], scenario.placement.parallel.world_size
        )
        self.assertGreater(estimate["estimated_cohort_count"], 0)
        self.assertGreater(estimate["estimated_event_task_count"], 0)
        self.assertEqual(
            estimate["recommended_retention_policy"], "aggregate"
        )
        self.assertNotIn("estimated_seconds", estimate)
        self.assertTrue(any("墙钟秒数" in item for item in estimate["warnings"]))

    def test_static_scenario_recommends_exact(self):
        scenario = build_reference_scenario()
        scheduler = replace(scenario.workload.scheduler, mode="static")
        workload = replace(scenario.workload, scheduler=scheduler)
        estimate = estimate_scenario(replace(scenario, workload=workload))
        self.assertEqual(
            estimate["recommended_retention_policy"], "exact"
        )

    def test_mtp_estimate_matches_serving_rounds_and_labels_no_preemption_baseline(self):
        scenario = build_reference_scenario()
        scheduler = replace(
            scenario.workload.scheduler,
            max_num_seqs=4,
            max_num_batched_tokens=4096,
            prefill_chunk_tokens=32,
            preemption_enabled=True,
        )
        workload = replace(
            scenario.workload,
            requests=(),
            request_count=8,
            prompt_tokens=1024,
            output_tokens=8192,
            scheduler=scheduler,
            mtp=replace(
                scenario.workload.mtp,
                candidate_tokens=4,
                acceptance_rate=0.65,
            ),
        )

        estimate = estimate_scenario(replace(scenario, workload=workload))

        # candidate_tokens is four drafts, so a full verifier has width five.
        # Draft-prefix expectations accumulate request-wide instead of being
        # rounded independently every round.  Eight requests form two groups.
        self.assertEqual(estimate["estimated_cohort_count"], 6552)
        self.assertEqual(estimate["estimated_mtp_round_count"], 6488)
        self.assertTrue(estimate["mtp_enabled"])
        self.assertEqual(estimate["mtp_candidate_tokens"], 4)
        self.assertEqual(estimate["estimate_basis"], "no_preemption_baseline")
        self.assertEqual(estimate["estimate_basis_zh"], "无抢占基线")
        self.assertFalse(estimate["dynamic_preemption_modeled"])
        self.assertTrue(
            any("estimated_cohort_count 是无抢占基线" in item for item in estimate["explanation"])
        )
        self.assertTrue(
            any("实际batch可因抢占/重算显著放大" in item for item in estimate["warnings"])
        )
        self.assertNotIn("estimated_seconds", estimate)

    def test_mtp_estimate_obeys_explicit_acceptance_model(self):
        scenario = build_reference_scenario()
        scheduler = replace(
            scenario.workload.scheduler,
            max_num_seqs=1,
            max_num_batched_tokens=4096,
            prefill_chunk_tokens=32,
        )
        common_mtp = replace(
            scenario.workload.mtp,
            candidate_tokens=4,
            acceptance_rate=1.0,
            acceptance_trace=(0.0,),
        )
        common_workload = replace(
            scenario.workload,
            requests=(),
            request_count=1,
            prompt_tokens=1,
            output_tokens=8,
            scheduler=scheduler,
        )

        expected = estimate_scenario(
            replace(
                scenario,
                workload=replace(
                    common_workload,
                    mtp=replace(common_mtp, acceptance_model="expected"),
                ),
            )
        )
        traced = estimate_scenario(
            replace(
                scenario,
                workload=replace(
                    common_workload,
                    mtp=replace(common_mtp, acceptance_model="trace"),
                ),
            )
        )

        self.assertEqual(expected["estimated_mtp_round_count"], 2)
        self.assertEqual(traced["estimated_mtp_round_count"], 7)

    def test_decode_estimate_excludes_prefill_first_token_for_all_workload_forms(self):
        base = build_reference_scenario()
        template = base.workload.requests[0]
        enabled_mtp = replace(
            base.workload.mtp,
            candidate_tokens=4,
            acceptance_model="expected",
            acceptance_rate=0.8,
        )

        for scheduler_mode in ("static", "continuous"):
            for workload_form in ("explicit", "synthetic"):
                for mtp_enabled in (False, True):
                    for output_tokens in (0, 1, 2):
                        with self.subTest(
                            scheduler_mode=scheduler_mode,
                            workload_form=workload_form,
                            mtp_enabled=mtp_enabled,
                            output_tokens=output_tokens,
                        ):
                            scheduler = replace(
                                base.workload.scheduler,
                                mode=scheduler_mode,
                                max_num_seqs=1,
                                max_num_batched_tokens=32,
                                prefill_chunk_tokens=32,
                            )
                            workload_values = {
                                "scheduler": scheduler,
                                "mtp": enabled_mtp if mtp_enabled else None,
                            }
                            if workload_form == "explicit":
                                workload_values.update(
                                    requests=(
                                        replace(
                                            template,
                                            prompt_tokens=4,
                                            output_tokens=output_tokens,
                                        ),
                                    ),
                                    request_count=1,
                                    prompt_tokens=4,
                                    output_tokens=output_tokens,
                                )
                            else:
                                workload_values.update(
                                    requests=(),
                                    request_count=1,
                                    prompt_tokens=4,
                                    output_tokens=output_tokens,
                                )
                            workload = replace(
                                base.workload, **workload_values
                            )

                            estimate = estimate_scenario(
                                replace(base, workload=workload)
                            )
                            remaining = max(0, output_tokens - 1)

                            self.assertEqual(
                                estimate["estimated_cohort_count"],
                                1 + remaining,
                            )
                            self.assertEqual(
                                estimate["estimated_mtp_round_count"],
                                remaining if mtp_enabled else 0,
                            )

    def test_continuous_promptless_estimate_keeps_first_decode_token(self):
        base = build_reference_scenario()
        template = base.workload.requests[0]
        enabled_mtp = replace(
            base.workload.mtp,
            candidate_tokens=4,
            acceptance_model="expected",
            acceptance_rate=1.0,
        )

        for scheduler_mode in ("static", "continuous"):
            for workload_form in ("explicit", "synthetic"):
                for mtp_enabled in (False, True):
                    for output_tokens in (0, 1, 2):
                        with self.subTest(
                            scheduler_mode=scheduler_mode,
                            workload_form=workload_form,
                            mtp_enabled=mtp_enabled,
                            output_tokens=output_tokens,
                        ):
                            scheduler = replace(
                                base.workload.scheduler,
                                mode=scheduler_mode,
                                max_num_seqs=1,
                                max_num_batched_tokens=32,
                                prefill_chunk_tokens=32,
                            )
                            values = {
                                "scheduler": scheduler,
                                "mtp": enabled_mtp if mtp_enabled else None,
                                "request_count": 1,
                                "prompt_tokens": 0,
                                "output_tokens": output_tokens,
                            }
                            if workload_form == "explicit":
                                values["requests"] = (
                                    replace(
                                        template,
                                        prompt_tokens=0,
                                        output_tokens=output_tokens,
                                    ),
                                )
                            else:
                                values["requests"] = ()
                            estimate = estimate_scenario(
                                replace(
                                    base,
                                    workload=replace(base.workload, **values),
                                )
                            )

                            remaining = (
                                max(0, output_tokens - 1)
                                if scheduler_mode == "static"
                                else output_tokens
                            )
                            decode_rounds = (
                                min(1, remaining)
                                if mtp_enabled
                                else remaining
                            )
                            prefill_rounds = (
                                1 if scheduler_mode == "static" else 0
                            )
                            self.assertEqual(
                                estimate["estimated_cohort_count"],
                                prefill_rounds + decode_rounds,
                            )
                            self.assertEqual(
                                estimate["estimated_mtp_round_count"],
                                decode_rounds if mtp_enabled else 0,
                            )

    def test_large_synthetic_workload_is_estimated_without_materializing_requests(self):
        scenario = build_reference_scenario()
        workload = replace(
            scenario.workload,
            requests=(),
            request_count=1_000_000_000,
            prompt_tokens=8,
            output_tokens=2,
        )

        estimate = estimate_scenario(replace(scenario, workload=workload))

        self.assertEqual(estimate["request_count"], 1_000_000_000)
        self.assertEqual(estimate["prompt_tokens"], 8_000_000_000)
        self.assertEqual(estimate["output_tokens"], 2_000_000_000)
        self.assertGreater(estimate["estimated_cohort_count"], 0)

    def test_large_synthetic_workload_uses_lazy_planner_requests(self):
        scenario = build_reference_scenario()
        large = replace(
            scenario,
            workload=replace(
                scenario.workload,
                requests=(),
                request_count=1_000_000_000,
                prompt_tokens=8,
                output_tokens=2,
                arrival_rate_rps=2.0,
            ),
        )

        requests = materialize_requests(large)

        self.assertNotIsInstance(requests, tuple)
        self.assertEqual(len(requests), 1_000_000_000)
        self.assertEqual(requests[3].request_id, "request-0003")
        self.assertEqual(requests[3].arrival_ns, 1_500_000_000.0)
        self.assertEqual(requests[-1].request_id, "request-999999999")
        self.assertFalse(validate_scenario(large).errors)

        with patch(
            "heterollm_sim.planner._compile_parallel_request",
            side_effect=RuntimeError("stop after first synthetic request"),
        ) as lowered:
            with self.assertRaisesRegex(RuntimeError, "stop after first"):
                compile_scenario(large)

        self.assertEqual(lowered.call_args.args[1].request_id, "request-0000")


class RunJobManagerTests(unittest.TestCase):
    def setUp(self):
        self.scenario = build_reference_scenario()
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown(wait=True, cancel_pending=True, cancel_running=True)

    def manager(self, **kwargs):
        manager = RunJobManager(**kwargs)
        self.managers.append(manager)
        return manager

    def test_job_completes_saves_report_and_progress(self):
        manager = self.manager(max_workers=1, max_history=4)
        observed_progress = threading.Event()

        def fake_run(scenario, *, retention_policy, control):
            control.report(
                "cohort", 2, 4, message="已完成两个 cohort", metadata={"batch": 2}
            )
            observed_progress.set()
            time.sleep(0.02)
            control.report("cohort", 4, 4, message="cohort 完成")
            return {
                "scenario": scenario.name,
                "retention_policy": retention_policy,
            }

        with patch("heterollm_sim.run_jobs.run_scenario", side_effect=fake_run), patch(
            "heterollm_sim.run_jobs.report_dict", side_effect=lambda result: result
        ):
            job_id = manager.submit(self.scenario)
            self.assertTrue(observed_progress.wait(1.0))
            running = manager.get(job_id)
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["progress"]["stage"], "cohort")
            self.assertGreaterEqual(running["progress"]["ratio"], 0.5)
            completed = wait_for(manager, job_id, {"completed"})

        self.assertEqual(completed["report"]["scenario"], self.scenario.name)
        self.assertEqual(completed["report"]["retention_policy"], "aggregate")
        self.assertEqual(completed["progress"]["ratio"], 1.0)
        self.assertIsNotNone(completed["started_at"])
        self.assertIsNotNone(completed["finished_at"])
        json.dumps(completed, ensure_ascii=False)

    def test_completed_online_job_replays_paged_batch_trace_without_snapshot_result(self):
        manager = self.manager(max_workers=1, max_history=4)
        job_id = manager.submit(self.scenario)
        completed = wait_for(manager, job_id, {"completed"})

        self.assertNotIn("result", completed)
        trace_index = completed["report"]["batch_trace_index"]
        batch = trace_index["batches"][1]
        first = manager.trace_page(
            job_id,
            batch["batch_id"],
            limit=2,
        )
        second = manager.trace_page(
            job_id,
            batch["batch_id"],
            offset=2,
            limit=2,
        )

        self.assertEqual(first["pagination"]["offset"], 0)
        self.assertEqual(first["pagination"]["returned"], 2)
        self.assertEqual(first["pagination"]["next_offset"], 2)
        self.assertEqual(second["pagination"]["offset"], 2)
        self.assertNotEqual(
            first["events"][0]["event_id"],
            second["events"][0]["event_id"],
        )
        self.assertAlmostEqual(
            batch["start_ns"], first["scope"]["device_start_ns"]
        )
        self.assertTrue(
            all(
                event["start_ns"]
                >= first["scope"]["start_ns"] - 1.0e-6
                for event in first["events"] + second["events"]
            )
        )
        null_rank = first["events"][0]["rank"]
        self.assertIsNone(null_rank["rank"])
        self.assertIsNone(null_rank["component_id"])
        self.assertTrue(
            any(
                "有界完成历史" in limitation
                for limitation in first["limitations"]
            )
        )
        json.dumps(first, ensure_ascii=False)

    def test_trace_replay_cache_is_single_flight_lru_and_prune_bounded(self):
        manager = self.manager(
            max_workers=1,
            max_history=4,
            max_trace_cache_entries=1,
        )
        job_id = manager.submit(self.scenario)
        completed = wait_for(manager, job_id, {"completed"})
        batches = completed["report"]["batch_trace_index"]["batches"]
        first_batch = batches[0]["batch_id"]
        second_batch = batches[1]["batch_id"]
        replay_started = threading.Event()
        release_replay = threading.Event()

        def blocking_replay(result, batch_id):
            replay_started.set()
            release_replay.wait(1.0)
            return replay_online_batch_trace(result, batch_id)

        pages = []
        errors = []

        def read_page(offset):
            try:
                pages.append(
                    manager.trace_page(
                        job_id,
                        first_batch,
                        offset=offset,
                        limit=2,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        with patch(
            "heterollm_sim.run_jobs.replay_online_batch_trace",
            side_effect=blocking_replay,
        ) as replay:
            threads = [
                threading.Thread(target=read_page, args=(offset,))
                for offset in (0, 2)
            ]
            threads[0].start()
            self.assertTrue(replay_started.wait(1.0))
            threads[1].start()
            time.sleep(0.02)
            release_replay.set()
            for thread in threads:
                thread.join(1.0)

            self.assertFalse(errors)
            self.assertEqual(len(pages), 2)
            self.assertEqual(replay.call_count, 1)
            manager.trace_page(job_id, first_batch, offset=4, limit=2)
            self.assertEqual(replay.call_count, 1)
            manager.trace_page(job_id, second_batch, limit=1)
            self.assertEqual(replay.call_count, 2)
            self.assertEqual(manager.trace_cache_size, 1)
            manager.trace_page(job_id, first_batch, limit=1)
            self.assertEqual(replay.call_count, 3)
            self.assertEqual(manager.trace_cache_size, 1)

        self.assertEqual(manager.prune(job_id), 1)
        self.assertEqual(manager.trace_cache_size, 0)

    def test_shutdown_clears_materialized_trace_cache(self):
        manager = self.manager(max_workers=1, max_history=2)
        job_id = manager.submit(self.scenario)
        completed = wait_for(manager, job_id, {"completed"})
        batch_id = completed["report"]["batch_trace_index"]["batches"][0][
            "batch_id"
        ]
        manager.trace_page(job_id, batch_id, limit=1)
        self.assertEqual(manager.trace_cache_size, 1)

        manager.shutdown(wait=True)

        self.assertEqual(manager.trace_cache_size, 0)

    def test_trace_page_rejects_invalid_page_state_and_batch(self):
        manager = self.manager(max_workers=1, max_history=4)
        release = threading.Event()
        started = threading.Event()

        def blocking_run(scenario, *, retention_policy, control):
            started.set()
            release.wait(1.0)
            return {}

        with patch("heterollm_sim.run_jobs.run_scenario", side_effect=blocking_run), patch(
            "heterollm_sim.run_jobs.report_dict", return_value={}
        ):
            job_id = manager.submit(self.scenario)
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(RunJobTraceError) as pending:
                manager.trace_page(job_id, "cohort-000000")
            self.assertEqual(pending.exception.code, "run_job_trace_not_ready")
            release.set()
            wait_for(manager, job_id, {"completed"})

        for offset, limit, code in (
            (-1, 1, "invalid_trace_offset"),
            (0, 0, "invalid_trace_limit"),
            (0, 5001, "invalid_trace_limit"),
        ):
            with self.subTest(offset=offset, limit=limit):
                with self.assertRaises(RunJobTraceError) as invalid:
                    manager.trace_page(
                        job_id,
                        "cohort-000000",
                        offset=offset,
                        limit=limit,
                    )
                self.assertEqual(invalid.exception.code, code)

        with self.assertRaises(RunJobTraceError) as unavailable:
            manager.trace_page(job_id, "cohort-000000")
        self.assertEqual(
            unavailable.exception.code,
            "run_job_trace_unavailable",
        )

        real_job = manager.submit(self.scenario)
        completed = wait_for(manager, real_job, {"completed"})
        with self.assertRaises(RunJobTraceError) as missing_batch:
            manager.trace_page(real_job, "missing-batch")
        self.assertEqual(
            missing_batch.exception.code,
            "run_job_batch_not_found",
        )
        manager.prune(real_job)
        with self.assertRaises(RunJobTraceError) as evicted:
            manager.trace_page(real_job, completed["report"]["batch_history"][0]["cohort_id"])
        self.assertEqual(evicted.exception.code, "run_job_not_found")

    def test_serving_progress_keeps_batch_primary_and_current_task_detail(self):
        manager = self.manager(max_workers=1, max_history=4)
        detail_observed = threading.Event()
        release = threading.Event()

        def fake_run(scenario, *, retention_policy, control):
            control.report(
                "serving_cohorts",
                3,
                message="正在推进在线批次",
                metadata={"finished_requests": 1, "request_count": 4},
            )
            control.report("cohort_tasks", 2, 5, message="正在执行批次拓扑任务")
            detail_observed.set()
            release.wait(1.0)
            return {
                "scenario": scenario.name,
                "retention_policy": retention_policy,
            }

        with patch("heterollm_sim.run_jobs.run_scenario", side_effect=fake_run), patch(
            "heterollm_sim.run_jobs.report_dict", side_effect=lambda result: result
        ):
            job_id = manager.submit(self.scenario)
            self.assertTrue(detail_observed.wait(1.0))
            running = manager.get(job_id)
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["progress"]["stage"], "serving_cohorts")
            self.assertEqual(running["progress"]["completed"], 3)
            self.assertIsNone(running["progress"]["total"])
            self.assertIsNone(running["progress"]["ratio"])
            self.assertEqual(running["progress"]["unit"], "serving_batches")
            self.assertEqual(running["progress"]["unit_label_zh"], "在线批次")
            detail = running["progress"]["detail"]
            self.assertEqual(detail["scope"], "nested")
            self.assertEqual(detail["stage"], "cohort_tasks")
            self.assertEqual(detail["completed"], 2)
            self.assertEqual(detail["total"], 5)
            self.assertEqual(detail["unit"], "schedule_tasks")
            self.assertEqual(detail["unit_label_zh"], "拓扑任务")
            self.assertNotEqual(
                running["progress"]["total"],
                running["estimate"]["estimated_cohort_count"],
            )
            json.dumps(running, ensure_ascii=False)
            release.set()
            completed = wait_for(manager, job_id, {"completed"})

        self.assertEqual(completed["progress"]["stage"], "completed")
        self.assertEqual(completed["progress"]["ratio"], 1.0)
        self.assertNotIn("detail", completed["progress"])

    def test_running_job_can_be_cancelled_cooperatively(self):
        manager = self.manager(max_workers=1, max_history=4)
        started = threading.Event()

        def cancellable_run(scenario, *, retention_policy, control):
            started.set()
            index = 0
            while True:
                control.raise_if_cancelled()
                control.report("cohort", index, 100, message="运行中")
                index += 1
                time.sleep(0.002)

        with patch(
            "heterollm_sim.run_jobs.run_scenario", side_effect=cancellable_run
        ):
            job_id = manager.submit(self.scenario)
            self.assertTrue(started.wait(1.0))
            self.assertTrue(manager.cancel(job_id))
            cancelled = wait_for(manager, job_id, {"cancelled"})

        self.assertTrue(cancelled["cancellation_requested"])
        self.assertIsNone(cancelled["report"])
        self.assertIsNone(cancelled["error"])
        self.assertEqual(cancelled["progress"]["stage"], "cancelled")

    def test_running_exact_job_cancels_at_an_in_loop_checkpoint(self):
        checkpoint = threading.Event()
        release_checkpoint = threading.Event()

        class CheckpointManager(RunJobManager):
            def _update_progress(self, job_id, progress):
                super()._update_progress(job_id, progress)
                if (
                    progress.stage == "schedule"
                    and progress.completed > 0
                    and progress.total is None
                ):
                    checkpoint.set()
                    release_checkpoint.wait(1.0)

        manager = CheckpointManager(max_workers=1, max_history=4)
        self.managers.append(manager)
        scheduler = replace(
            self.scenario.workload.scheduler,
            mode="static",
        )
        scenario = replace(
            self.scenario,
            workload=replace(self.scenario.workload, scheduler=scheduler),
        )

        try:
            job_id = manager.submit(
                scenario, retention_policy="exact"
            )
            self.assertTrue(checkpoint.wait(2.0))
            running = manager.get(job_id)
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["progress"]["stage"], "schedule")
            self.assertGreater(running["progress"]["completed"], 0)
            self.assertIsNone(running["progress"]["total"])

            cancel_started = time.monotonic()
            self.assertTrue(manager.cancel(job_id))
            release_checkpoint.set()
            cancelled = wait_for(manager, job_id, {"cancelled"}, timeout=1.0)
            self.assertLess(time.monotonic() - cancel_started, 1.0)
        finally:
            release_checkpoint.set()

        self.assertTrue(cancelled["cancellation_requested"])
        self.assertIsNone(cancelled["report"])
        self.assertEqual(cancelled["progress"]["stage"], "cancelled")

    def test_queued_job_can_be_cancelled(self):
        manager = self.manager(max_workers=1, max_history=4)
        release = threading.Event()
        first_started = threading.Event()

        def blocking_run(scenario, *, retention_policy, control):
            first_started.set()
            while not release.wait(0.005):
                control.raise_if_cancelled()
            return {}

        with patch("heterollm_sim.run_jobs.run_scenario", side_effect=blocking_run), patch(
            "heterollm_sim.run_jobs.report_dict", return_value={}
        ):
            first_id = manager.submit(self.scenario)
            self.assertTrue(first_started.wait(1.0))
            queued_id = manager.submit(self.scenario)
            self.assertEqual(manager.get(queued_id)["status"], "queued")
            self.assertTrue(manager.cancel(queued_id))
            self.assertEqual(manager.get(queued_id)["status"], "cancelled")
            release.set()
            wait_for(manager, first_id, {"completed"})

    def test_unknown_job_id_is_safe(self):
        manager = self.manager(max_workers=1, max_history=2)
        self.assertIsNone(manager.get("missing"))
        self.assertFalse(manager.cancel("missing"))
        self.assertEqual(manager.prune("missing"), 0)

    def test_active_job_queue_is_bounded(self):
        manager = self.manager(max_workers=1, max_history=2, max_active_jobs=1)
        started = threading.Event()

        def blocking_run(scenario, *, retention_policy, control):
            started.set()
            while True:
                control.raise_if_cancelled()
                time.sleep(0.002)

        with patch("heterollm_sim.run_jobs.run_scenario", side_effect=blocking_run):
            job_id = manager.submit(self.scenario)
            self.assertTrue(started.wait(1.0))
            with patch(
                "heterollm_sim.run_jobs.estimate_scenario",
                side_effect=AssertionError("队列已满时不应再次估算"),
            ):
                with self.assertRaisesRegex(RunJobCapacityError, "队列已满"):
                    manager.submit(self.scenario)
            self.assertTrue(manager.cancel(job_id))
            wait_for(manager, job_id, {"cancelled"})

    def test_terminal_history_is_bounded(self):
        manager = self.manager(max_workers=1, max_history=2)

        with patch("heterollm_sim.run_jobs.run_scenario", return_value={}), patch(
            "heterollm_sim.run_jobs.report_dict", return_value={"ok": True}
        ):
            ids = []
            for _ in range(4):
                job_id = manager.submit(self.scenario)
                ids.append(job_id)
                wait_for(manager, job_id, {"completed"})

        snapshots = manager.list()
        self.assertEqual(len(snapshots), 2)
        self.assertIsNone(manager.get(ids[0]))
        self.assertEqual({item["job_id"] for item in snapshots}, set(ids[-2:]))

    def test_terminal_history_keeps_latest_completion_not_latest_submission(self):
        manager = self.manager(max_workers=2, max_history=1)
        slow_started = threading.Event()
        release_slow = threading.Event()
        static_scenario = replace(
            self.scenario,
            workload=replace(
                self.scenario.workload,
                scheduler=replace(
                    self.scenario.workload.scheduler,
                    mode="static",
                ),
            ),
        )

        def out_of_order_run(_scenario, *, retention_policy, control):
            if retention_policy == "streaming":
                slow_started.set()
                release_slow.wait(1.0)
            return {"retention_policy": retention_policy}

        with patch(
            "heterollm_sim.run_jobs.run_scenario",
            side_effect=out_of_order_run,
        ), patch(
            "heterollm_sim.run_jobs.report_dict",
            side_effect=lambda result: result,
        ):
            slow_id = manager.submit(
                static_scenario, retention_policy="streaming"
            )
            self.assertTrue(slow_started.wait(1.0))
            fast_id = manager.submit(
                static_scenario, retention_policy="aggregate"
            )
            wait_for(manager, fast_id, {"completed"})
            release_slow.set()
            slow = wait_for(manager, slow_id, {"completed"})

        self.assertEqual(
            slow["report"]["retention_policy"], "streaming"
        )
        self.assertIsNone(manager.get(fast_id))
        self.assertEqual(
            [snapshot["job_id"] for snapshot in manager.list()],
            [slow_id],
        )

    def test_failure_exposes_only_safe_error(self):
        manager = self.manager(max_workers=1, max_history=2)

        with patch(
            "heterollm_sim.run_jobs.run_scenario",
            side_effect=ValueError("secret path D:/private/config.json"),
        ):
            job_id = manager.submit(self.scenario)
            failed = wait_for(manager, job_id, {"failed"})

        self.assertEqual(
            failed["error"],
            {
                "message": "仿真任务失败，请检查场景配置后重试。",
                "exception_type": "ValueError",
            },
        )
        self.assertNotIn("private", json.dumps(failed, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
