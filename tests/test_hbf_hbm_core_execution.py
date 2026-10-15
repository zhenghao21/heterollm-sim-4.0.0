"""HBM operators and KV copies must execute on the persistent physical Core."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from hbf_hbm_core_audit import CoreExecutionAudit
from hbf_hbm_experiments import scenario
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest, Operation, make_hbm_config, parse_physical_memory_config
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import RequestStatus
import test_paged_pool_serving as paged_fixtures


def _rows(summary, kind):
    return [row for row in summary["owners"] if row["memory_kind"] == kind]


class CoreAuditTests(unittest.TestCase):
    def test_identical_owner_preview_and_unregistered_runtime_are_excluded(self):
        config = make_hbm_config(capacity_bytes=1 << 20)
        live = PhysicalRuntimeContext(capture_details=False)
        other = PhysicalRuntimeContext(capture_details=False)
        with CoreExecutionAudit() as audit:
            audit._register_runtime(live)
            actual = live.runtime(config, "same-owner")
            preview = live.preview_runtime(config, "same-owner", capture_details=False)
            unrelated = other.runtime(config, "same-owner")
            actual.core.submit(AccessRequest("live", Operation.READ, 0, 64))
            preview.core.submit(AccessRequest("preview", Operation.READ, 0, 4096))
            unrelated.core.submit(AccessRequest("other", Operation.WRITE, 0, 8192))
        owners = audit.summary()["owners"]
        self.assertEqual(len(owners), 1)
        self.assertEqual(owners[0]["transactions"], 1)
        self.assertEqual(owners[0]["logical_read_bytes"], 64)
        self.assertEqual(owners[0]["logical_write_bytes"], 0)

    def test_submit_many_counts_each_transaction_once_in_both_execution_paths(self):
        # Detailed execution falls through submit; summary execution can use
        # the compiled row/burst engine without calling submit. Both must be
        # observed once, and both must retain actual read/write/burst counts.
        requests = (AccessRequest("read", Operation.READ, 0, 8192),
                    AccessRequest("write", Operation.WRITE, 8192, 4096))
        observations = []
        for detailed in (True, False):
            with self.subTest(capture_details=detailed):
                context = PhysicalRuntimeContext(capture_details=detailed)
                with CoreExecutionAudit() as audit:
                    audit._register_runtime(context)
                    core = context.runtime(make_hbm_config(capacity_bytes=1 << 20), "hbm").core
                    self.assertEqual(core._can_accelerate(), not detailed)
                    results = core.submit_many(iter(requests))
                row = audit.summary()["owners"][0]
                self.assertEqual(row["entrypoint_calls"], {"submit_many": 1})
                self.assertEqual(row["transactions"], 2)
                self.assertEqual(row["logical_read_bytes"], 8192)
                self.assertEqual(row["logical_write_bytes"], 4096)
                self.assertEqual(row["burst_count"], 12288 // core.config.burst_bytes)
                self.assertEqual(row["last_completion_ns"], max(r.completion_ns for r in results))
                observations.append((results, row))
        for detailed, compact in zip(observations[0][0], observations[1][0]):
            self.assertAlmostEqual(detailed.completion_ns, compact.completion_ns, places=8)
            self.assertEqual(detailed.transfer_bytes, compact.transfer_bytes)


class PhysicalHbmServingTests(unittest.TestCase):
    def test_default_hbm_and_hbf_paths_reach_their_live_cores(self):
        for hbf in (False, True):
            with self.subTest(hbf=hbf):
                with CoreExecutionAudit() as audit:
                    result = run_scenario(scenario(hbf=hbf, prompt=32, count=1, output=4))
                summary = audit.summary()
                self.assertTrue(summary["valid_for_committed_byte_totals"])
                self.assertTrue(all(m.status is RequestStatus.FINISHED
                                    for m in result.serving.request_metrics.values()))
                for kind, core_name in (("HBM", "DramCore"), ("HBF", "NandCore")):
                    rows = _rows(summary, kind)
                    if kind == "HBF" and not hbf:
                        self.assertFalse(rows)
                        continue
                    self.assertTrue(rows, f"No actual {kind} Core submissions")
                    self.assertTrue(all(row["core"] == core_name for row in rows))
                    self.assertGreater(sum(row["logical_read_bytes"] for row in rows), 0)
                    self.assertGreater(sum(row["logical_write_bytes"] for row in rows), 0)
                decode = [cohort for cohort in summary["cohorts"] if cohort["phase"] == "decode"]
                self.assertEqual(len(decode), 3)
                for cohort in decode:
                    self.assertGreater(sum(row["logical_read_bytes"] for row in _rows(cohort, "HBM")), 0)
                    if hbf:
                        self.assertGreater(sum(row["pages_read"] for row in _rows(cohort, "HBF")), 0)
                        self.assertGreater(sum(row["logical_write_bytes"] for row in _rows(cohort, "HBF")), 0)

    def test_dedicated_hbm_kv_reads_and_appends_execute_separately_from_weights(self):
        config = scenario(prompt=32, count=1, output=4, kv_budget=65536)
        kv_owner = config.hardware.get_component("hbm_kv").metadata["memory_service_owner"]
        with CoreExecutionAudit() as audit:
            result = run_scenario(config)
        summary = audit.summary()
        self.assertTrue(summary["valid_for_committed_byte_totals"])
        rows = [row for row in summary["owners"] if row["physical_owner"] == kv_owner]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["core"], "DramCore")
        self.assertGreater(rows[0]["logical_read_bytes"], 0)
        self.assertGreater(rows[0]["logical_write_bytes"], 0)
        self.assertGreater(result.serving.kv_metrics.logical_decode_append_bytes, 0)
        for cohort in summary["cohorts"]:
            if cohort["phase"] != "decode":
                continue
            kv = [row for row in cohort["owners"] if row["physical_owner"] == kv_owner]
            self.assertTrue(kv)
            self.assertGreater(sum(row["logical_read_bytes"] for row in kv), 0)
            self.assertGreater(sum(row["logical_write_bytes"] for row in kv), 0)

    def test_hbm_encoded_read_latency_changes_actual_decode_not_just_reported_metadata(self):
        base = scenario(prompt=32, count=1, output=3)
        components = []
        for component in base.hardware.components:
            if component.component_id == "hbm0":
                physical = parse_physical_memory_config(component.metadata["physical_memory_config"])
                component = replace(component, metadata={**component.metadata,
                    "physical_memory_config": replace(physical,
                        read_latency_ns=63 * physical.command_clock_ns)})
            components.append(component)
        slow = replace(base, hardware=replace(base.hardware, components=tuple(components)))
        normal = next(iter(run_scenario(base).serving.request_metrics.values()))
        delayed = next(iter(run_scenario(slow).serving.request_metrics.values()))
        self.assertEqual(normal.visible_output_tokens, delayed.visible_output_tokens)
        self.assertGreater(delayed.tpot_ns, normal.tpot_ns)

    def test_removed_aggregate_metadata_cannot_silently_bypass_dram_core(self):
        base = scenario(prompt=32, count=1, output=2)
        components = tuple(replace(c, metadata={**c.metadata, "memory_execution_mode": "aggregate"})
                           if c.component_id == "hbm0" else c for c in base.hardware.components)
        with self.assertRaisesRegex(ValueError, "memory_execution_mode.*no longer configurable"):
            obsolete = replace(base, hardware=replace(base.hardware, components=components))
            run_scenario(obsolete)

    def test_analytical_hbm_preview_timing_cannot_set_live_memory_timing(self):
        from hbf_hbm_serving_study import build_study_scenario
        # Hold the operator-to-device mapping fixed: otherwise this large
        # preview change legitimately makes the auto-placer select the CPU.
        base = build_study_scenario("A_hbm_only", prompt=32, batch=1, output=3)
        component = base.hardware.get_component("hbm0")
        registry = {kind:dict(items) for kind,items in base.component_profiles.items()}
        profile = registry["hbm"][component.cost_profile_id]
        registry["hbm"][component.cost_profile_id] = replace(profile,
            bandwidth_gb_s=1., read_bandwidth_gb_s=1., write_bandwidth_gb_s=1.,
            read_latency_ns=1e9, write_latency_ns=1e9)
        changed_preview = replace(base, component_profiles=registry)
        self.assertEqual(component.metadata["physical_memory_config"],
            changed_preview.hardware.get_component("hbm0").metadata["physical_memory_config"])
        normal = next(iter(run_scenario(base).serving.request_metrics.values()))
        observed = next(iter(run_scenario(changed_preview).serving.request_metrics.values()))
        self.assertEqual(normal.tpot_ns, observed.tpot_ns)
        self.assertEqual(normal.finish_ns, observed.finish_ns)

    def test_offload_and_restore_pay_both_dram_and_nand_endpoints(self):
        runtime, request = paged_fixtures.PagedPoolServingTests()._runtime()
        executor = runtime.ledger.paged_pool.transfer_executor
        page_bytes = runtime.plan.kv_policy.bytes_per_page
        with CoreExecutionAudit(max_call_records=32) as audit:
            audit._register_runtime(executor.runtime)
            self.assertTrue(runtime.ledger.resize(request, 1))
            self.assertTrue(runtime.ledger.offload(request))
            runtime._sync_paged_pool_events()
            offload_end = runtime.now
            first = audit.summary()
            self.assertEqual(sum(row["logical_read_bytes"] for row in _rows(first, "HBM")), page_bytes)
            self.assertEqual(sum(row["logical_write_bytes"] for row in _rows(first, "HBF")), page_bytes)
            self.assertTrue(runtime._resume_swap(request))
        final = audit.summary()
        self.assertEqual(sum(row["logical_write_bytes"] for row in _rows(final, "HBM")), page_bytes)
        self.assertEqual(sum(row["logical_read_bytes"] for row in _rows(final, "HBF")), page_bytes)
        self.assertGreater(sum(row["burst_count"] for row in _rows(final, "HBM")), 0)
        self.assertGreater(sum(row["pages_read"] for row in _rows(final, "HBF")), 0)
        self.assertGreater(sum(row["pages_programmed"] for row in _rows(final, "HBF")), 0)
        self.assertGreater(runtime.now, offload_end)
        reads = [r for r in final["call_records"] if r["memory_kind"] == "HBF"
                 and r["logical_read_bytes"]]
        writes = [r for r in final["call_records"] if r["memory_kind"] == "HBM"
                  and r["logical_write_bytes"]]
        self.assertGreaterEqual(min(r["first_arrival_ns"] for r in writes),
                                max(r["last_completion_ns"] for r in reads))


if __name__ == "__main__":
    unittest.main()
