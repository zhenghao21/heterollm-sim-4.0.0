"""Bounded host admission depth and per-access page utilization accounting."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from heterollm_sim.memory_types import AccessRequest, Operation, make_hbf_config
from heterollm_sim.nand_core import NandCore
from heterollm_sim.planner import _summarize_nand_task_traffic
from heterollm_sim.reporting import _sum_batch_storage_traffic, report_dict, run_scenario


PAGE = 4096


def config(**kwargs):
    values = dict(channels=1, targets_per_channel=1, dies_per_target=1,
                  luns_per_die=1, planes_per_lun=1, blocks_per_plane=2,
                  pages_per_block=256, page_bytes=PAGE, host_granularity_bytes=1,
                  host_bandwidth_gb_s=10.0, host_read_bandwidth_gb_s=10.0, host_write_bandwidth_gb_s=10.0, internal_bandwidth_gb_s=10.0,
                  page_read_ns=100.0, page_program_ns=200.0,
                  max_outstanding_requests=3)
    values.update(kwargs)
    return make_hbf_config(**values)


def task(name, counters):
    return SimpleNamespace(task_id=name, demands=(), metadata={
        "physical_execution_by_owner": {"hbf0": {"kind": "HBF", **counters}}})


class NandAdmissionMetricsTests(unittest.TestCase):
    def test_mixed_requests_sample_admission_population_and_remain_bounded(self):
        core = NandCore(config())
        rows = [core.submit(AccessRequest(str(i), operation, i * PAGE, PAGE))
                for i, operation in enumerate((Operation.READ, Operation.WRITE, Operation.READ))]
        self.assertEqual([r.counters["host_outstanding_at_admission_peak"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[-1].counters["host_read_outstanding_at_admission_peak"], 2)
        self.assertEqual(rows[-1].counters["host_write_outstanding_at_admission_peak"], 1)
        for index in range(3, 40):
            row = core.submit(AccessRequest(str(index), Operation.WRITE, index * PAGE, PAGE))
            self.assertLessEqual(row.counters["host_outstanding_at_admission_peak"], 3)
            self.assertLessEqual(len(core._inflight_operations), 3)
        after = core.submit(AccessRequest("after", Operation.READ, 0, PAGE,
                                         max(end for end, _op in core._inflight_operations)))
        self.assertEqual(after.counters["host_outstanding_at_admission_peak"], 1)

    def test_unaligned_access_counts_touched_pages_instead_of_unique_residency(self):
        core = NandCore(config())
        rows = [core.submit(AccessRequest(str(i), Operation.READ, PAGE - 16, 32))
                for i in range(2)]
        summary = _summarize_nand_task_traffic([task(str(i), r.counters) for i, r in enumerate(rows)])
        self.assertEqual(summary["page_access_payload_bytes"], 64)
        self.assertEqual(summary["page_access_capacity_bytes"], 4 * PAGE)
        self.assertEqual(summary["page_access_payload_utilization"], 32 / (2 * PAGE))
        self.assertEqual(summary["host_outstanding_at_admission_peak"], 2)
        self.assertIsNone(summary["nand_read_queue_depth"])
        self.assertEqual(summary["metric_availability"]["nand_program_queue_depth"]["status"], "not_recorded")

    def test_accelerated_large_access_preserves_counters_without_page_trace(self):
        request = AccessRequest("large", Operation.READ, PAGE - 8, 128 * PAGE)
        detailed = NandCore(config()).submit(request)
        accelerated = NandCore(config(), capture_details=False).submit(request)
        for key in ("page_access_payload_bytes", "page_access_capacity_bytes", "host_admission_samples",
                    "host_outstanding_at_admission_peak", "host_read_outstanding_at_admission_peak"):
            self.assertEqual(detailed.counters[key], accelerated.counters[key])
        self.assertEqual(accelerated.counters["page_access_capacity_bytes"], 129 * PAGE)
        self.assertFalse(accelerated.stages)
        self.assertFalse(accelerated.mapping)

    def test_buffered_host_write_leaves_population_at_ack_before_media_drain(self):
        core = NandCore(config(write_buffer_bytes=2 * PAGE, metadata={}))
        write = core.submit(AccessRequest("write", Operation.WRITE, 0, PAGE))
        self.assertEqual(write.pages_programmed, 0)
        read = core.submit(AccessRequest("read", Operation.READ, 0, PAGE, write.completion_ns))
        self.assertEqual(read.counters["host_write_outstanding_at_admission_peak"], 0)
        self.assertEqual(read.counters["host_outstanding_at_admission_peak"], 1)
        self.assertEqual(core.buffered_page_count, 1)
        self.assertEqual(read.staging_read_hits, 1)

    def test_batch_aggregation_takes_max_depth_and_sums_access_denominators(self):
        core = NandCore(config())
        ledgers = [_summarize_nand_task_traffic([task(str(i), core.submit(
            AccessRequest(str(i), Operation.READ, i * PAGE, PAGE // (i + 1))).counters)])
            for i in range(2)]
        batches = [SimpleNamespace(cost=SimpleNamespace(metadata={"storage_traffic": row}))
                   for row in ledgers]
        result = SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=batches))
        summary = _sum_batch_storage_traffic(result)
        self.assertEqual(summary["host_outstanding_at_admission_peak"], 2)
        self.assertEqual(summary["page_access_payload_utilization"], 0.75)
        self.assertEqual(summary["host_admission_samples"], 2)
        del ledgers[1]["host_outstanding_at_admission_peak"]
        del ledgers[1]["page_access_capacity_bytes"]
        partial = _sum_batch_storage_traffic(result)
        self.assertIsNone(partial["host_outstanding_at_admission_peak"])
        self.assertIsNone(partial["page_access_payload_utilization"])
        self.assertEqual(partial["metric_availability"]["page_access_capacity_bytes"]["status"], "partial")

    def test_full_serving_retains_exact_scope_in_public_report(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from hbf_hbm_serving_study import build_study_scenario
        result = run_scenario(build_study_scenario("B_all_kv_hbf", prompt=32, output=3))
        summary = report_dict(result, include_visualization=False)["storage_traffic"]
        self.assertGreater(summary["host_admission_samples"], 0)
        self.assertGreaterEqual(summary["host_outstanding_at_admission_peak"], 1)
        self.assertGreater(summary["page_access_payload_utilization"], 0)
        self.assertLessEqual(summary["page_access_payload_utilization"], 1)
        self.assertIsNone(summary["nand_program_queue_depth"])
        self.assertEqual(summary["metric_availability"]["host_read_outstanding_at_admission_peak"]["status"], "recorded")
        self.assertIn("not NAND page/program queues", summary["queue_depth_scope"])


if __name__ == "__main__":
    unittest.main()
