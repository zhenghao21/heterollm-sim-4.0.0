"""Small, explicit DRAM semantics and accelerated-path equivalence checks."""
from dataclasses import replace
from pathlib import Path
import builtins
import copy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import map_dram_address, split_dram_request
from heterollm_sim.memory_types import AccessRequest, DramConfig


def config(**changes):
    base = DramConfig(banks_per_group=1, rows_per_bank=4, row_bytes=256,
                      burst_bytes=64, interface_bandwidth_gb_s=64,
                      open_ns=10, close_ns=7, read_latency_ns=5,
                      write_latency_ns=3, burst_interval_ns=2,
                      write_recovery_ns=9, read_to_write_ns=4,
                      write_to_read_ns=6)
    return replace(base, capacity_bytes=None, **changes)


def state(core):
    return copy.deepcopy((core._banks, core._inflight, core._acceptance_ns,
                          vars(core.timeline)))


class DramCoreSemanticsTests(unittest.TestCase):
    def assert_matches(self, expected_core, expected, actual_core, actual):
        self.assertEqual(len(expected), len(actual))
        scalar = ("row_hits", "row_misses", "row_conflicts", "burst_count",
                  "physical_read_bytes", "physical_write_bytes", "queue_wait_ns")
        for left, right in zip(expected, actual):
            self.assertAlmostEqual(left.completion_ns, right.completion_ns, delta=1e-8)
            self.assertEqual(left.logical_bytes, right.logical_bytes)
            self.assertEqual(left.transfer_bytes, right.transfer_bytes)
            for key in scalar:
                self.assertAlmostEqual(left.counters[key], right.counters[key], delta=1e-8)
            for key in ("resource_busy_ns", "resource_bytes", "resource_last_intervals"):
                self.assertEqual(set(left.counters[key]), set(right.counters[key]))
                for resource, value in left.counters[key].items():
                    other = right.counters[key][resource]
                    for a, b in zip(value if isinstance(value, tuple) else (value,),
                                    other if isinstance(other, tuple) else (other,)):
                        self.assertAlmostEqual(a, b, delta=1e-8)
        self.assertEqual(expected_core.timeline.bytes_moved, actual_core.timeline.bytes_moved)
        self.assertEqual(expected_core.timeline.directions, actual_core.timeline.directions)
        self.assertEqual(set(expected_core._banks), set(actual_core._banks))
        for key, bank in expected_core._banks.items():
            self.assertEqual(bank.open_row, actual_core._banks[key].open_row)
            self.assertAlmostEqual(bank.ready_ns, actual_core._banks[key].ready_ns, delta=1e-8)
        for key, ready in expected_core.timeline.ready_ns.items():
            self.assertAlmostEqual(ready, actual_core.timeline.ready_ns[key], delta=1e-8)
        self.assertEqual(expected_core._acceptance_ns, actual_core._acceptance_ns)
        for left, right in zip(expected_core._inflight, actual_core._inflight):
            self.assertAlmostEqual(left, right, delta=1e-8)

    def test_hand_calculated_row_hit_conflict_turnaround_and_accounting(self):
        core = DramCore(config())
        rows = core.submit_many((AccessRequest("r0", "read", 0, 64),
                                 AccessRequest("w0", "write", 64, 64),
                                 AccessRequest("r1", "read", 128, 64),
                                 AccessRequest("conflict", "read", 256, 64)))
        self.assertEqual([r.completion_ns for r in rows], [16, 30, 28, 53])
        self.assertEqual([(r.counters["row_hits"], r.counters["row_misses"],
                           r.counters["row_conflicts"]) for r in rows],
                         [(0, 1, 0), (1, 0, 0), (1, 0, 0), (0, 0, 1)])
        # Turnarounds and pipeline/recovery waits are latency, not bus busy.
        self.assertEqual(core.timeline.busy_ns["dram:data:0"], 4)
        self.assertEqual(core.timeline.busy_ns["dram:command:0"], 8)
        self.assertEqual(core.timeline.busy_ns["dram:bank:0:0:0:0:0:0"], 27)
        self.assertEqual(core.timeline.bytes_moved, {"dram:data:0": 256})
        self.assertEqual(sum(r.physical_read_bytes for r in rows), 192)
        self.assertEqual(sum(r.physical_write_bytes for r in rows), 64)
        # Same-row data may complete before a preceding write's recovery.
        # A conflicting row's PRE must wait until that recovery is finished.
        pre = next(s for s in rows[-1].stages if s.name == "PRECHARGE")
        self.assertEqual(pre.start_ns, 30)

    def test_independent_channels_and_banks_with_shared_lane_bandwidth(self):
        two_lanes = config(channels=2, interface_bandwidth_gb_s=128)
        independent = DramCore(two_lanes)
        rows = independent.submit_many((AccessRequest("c0", "read", 0, 64),
                                        AccessRequest("c1", "read", 64, 64)))
        self.assertEqual([r.completion_ns for r in rows], [16, 16])
        shared = DramCore(replace(two_lanes, metadata={"shared_command_resource": "ca"}))
        rows = shared.submit_many((AccessRequest("c0", "read", 0, 64),
                                   AccessRequest("c1", "read", 64, 64)))
        self.assertEqual([r.completion_ns for r in rows], [16, 18])
        banks = DramCore(config(banks_per_group=2))
        rows = banks.submit_many((AccessRequest("b0", "read", 0, 64),
                                  AccessRequest("b1", "read", 64, 64)))
        self.assertEqual([r.completion_ns for r in rows], [16, 18])
        self.assertEqual(len(banks._banks), 2)
        self.assertEqual(len(banks.timeline.bytes_moved), 1)
        stacked = config(stacks=2, dies_per_stack=2)
        self.assertEqual(stacked.directional_bandwidth_gb_s(rows[0].operation), 64)
        self.assertEqual(stacked.capacity_bytes, 4 * config().capacity_bytes)

    def test_admission_queue_is_request_ordered_and_bounded(self):
        core = DramCore(config(max_outstanding_requests=1))
        rows = core.submit_many((AccessRequest("first", "read", 0, 64, 100),
                                 AccessRequest("early", "read", 64, 64, 0)))
        self.assertEqual([r.completion_ns for r in rows], [116, 122])
        self.assertEqual(rows[1].queue_wait_ns, 116)
        self.assertEqual(rows[1].arrival_ns, 0)

    def test_all_burst_addresses_map_uniquely_in_small_geometries(self):
        # Includes interleave larger than one row: valid but not accelerable.
        for interleave in (64, 128, 512):
            c = config(channels=2, ranks_per_channel=2, bank_groups_per_rank=2,
                       banks_per_group=2, stacks=2, dies_per_stack=2,
                       interleave_bytes=interleave)
            seen = set()
            for address in range(0, c.capacity_bytes, c.burst_bytes):
                m = map_dram_address(c, address)
                coord = (m.stack, m.die, m.lane, m.rank, m.bank_group,
                         m.bank, m.row, m.column)
                self.assertNotIn(coord, seen)
                self.assertLess(m.row, c.rows_per_bank)
                self.assertLess(m.column, c.row_bytes // c.burst_bytes)
                seen.add(coord)
            self.assertEqual(len(seen), c.capacity_bytes // c.burst_bytes)
            with self.assertRaises(ValueError):
                map_dram_address(c, c.capacity_bytes)

    def test_non_aligned_logical_bytes_and_atomic_invalid_requests(self):
        c = config()
        core = DramCore(c)
        segments = tuple(split_dram_request(AccessRequest("split", "read", 63, 2), c))
        self.assertEqual([(s.address, s.logical_bytes, s.transfer_bytes) for s in segments],
                         [(0, 1, 64), (64, 1, 64)])
        result = core.submit(AccessRequest("last", "read", c.capacity_bytes - 1, 1))
        self.assertEqual((result.logical_bytes, result.transfer_bytes), (1, 64))
        before = state(core)
        for invalid in (AccessRequest("end", "read", c.capacity_bytes, 1),
                        AccessRequest("cross", "write", c.capacity_bytes - 1, 2),
                        AccessRequest("erase", "erase", 0, 64)):
            with self.subTest(request=invalid.request_id), self.assertRaises(ValueError):
                core.submit_many((AccessRequest("valid", "read", 0, 64), invalid))
            self.assertEqual(before, state(core))
        with self.assertRaises(ValueError):
            AccessRequest("zero", "read", 0, 0)
        for entry in (core.submit, core.submit_many):
            with self.assertRaisesRegex(TypeError, "AccessRequest"):
                entry((object(),) if entry == core.submit_many else object())
        self.assertEqual(before, state(core))

    def test_compiled_single_batch_and_python_fold_match_burst_execution(self):
        for shared in (False, True):
            for outstanding in (1, 3):
                with self.subTest(shared=shared, outstanding=outstanding):
                    c = config(channels=2, banks_per_group=2, stacks=2, dies_per_stack=2,
                               row_bytes=4096, interleave_bytes=128,
                               interface_bandwidth_gb_s=73.25, burst_interval_ns=2.3,
                               max_outstanding_requests=outstanding,
                               metadata={"shared_command_resource": "ca"} if shared else {})
                    requests = (AccessRequest("long", "read", 3, 33001),
                                AccessRequest("other_die", "write", c.capacity_bytes // 2 + 13, 19411, 3),
                                AccessRequest("short", "write", 0, 65, 0),
                                AccessRequest("old_row", "read", 128, 12400, 7),
                                AccessRequest("late", "read", 192, 1, 100000))
                    reference = DramCore(c)
                    expected = reference.submit_many(requests)
                    compiled = DramCore(c, capture_details=False)
                    self.assert_matches(reference, expected, compiled,
                                        tuple(compiled.submit(r) for r in requests))
                    batch = DramCore(c, capture_details=False)
                    result = batch.submit_many(requests)
                    self.assertTrue(all(r.counters.get("compiled_batch_float64") for r in result))
                    self.assert_matches(reference, expected, batch, result)
                    fallback = DramCore(c, capture_details=False)
                    with patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None):
                        result = tuple(fallback.submit(r) for r in requests)
                    self.assert_matches(reference, expected, fallback, result)

    def test_accelerator_unavailable_keeps_batch_fallback_and_state(self):
        c = config(row_bytes=8192)
        requests = (AccessRequest("read", "read", 1, 4096),
                    AccessRequest("write", "write", 4097, 4096))
        reference = DramCore(c)
        expected = reference.submit_many(requests)
        core = DramCore(c, capture_details=False)
        original_import = builtins.__import__
        def unavailable(name, *args, **kwargs):
            if name in ("_dram_numeric", "heterollm_sim._dram_numeric"):
                raise ImportError("simulated unavailable numeric accelerator")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=unavailable):
            result = core.submit_many(requests)
        self.assert_matches(reference, expected, core, result)
        self.assertTrue(all(not r.counters.get("compiled_batch_float64") for r in result))

    def test_details_cap_does_not_change_service_or_physical_accounting(self):
        c = config(row_bytes=8192)
        request = AccessRequest("cap", "write", 63, 4097)
        full, capped = DramCore(c), DramCore(replace(c, max_expanded_segments=2))
        expected, actual = full.submit(request), capped.submit(request)
        self.assert_matches(full, (expected,), capped, (actual,))
        self.assertTrue(actual.counters["details_truncated"])
        self.assertEqual(actual.stages, ())
        self.assertEqual(actual.mapping, ())


if __name__ == "__main__":
    unittest.main()
