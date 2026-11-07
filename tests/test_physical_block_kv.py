"""Physical KV in per-rank, per-layer block pools addressed through request block tables."""
import copy
from dataclasses import replace
from pathlib import Path
import itertools
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.kv_placement import kv_sequential_layout, sequential_page_table_runs
from heterollm_sim.native_stationary import kv_block_peak
from heterollm_sim.memory_types import KVRuns
from heterollm_sim.planner import (_generic_kv_accesses, _generic_kv_read_rows, _kv_block_runs, kv_block_geometry,
                                   _parallel_plan, _execution_layers, _subtract_direct_buffer_accesses)
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import RequestStatus, _KVBlockTables, _settled_free_list


class KVPlacementTests(unittest.TestCase):
    def test_sequential_layout_uses_compact_arithmetic_page_runs(self):
        layout = kv_sequential_layout(1660, 373, 2, channels=16, planes=25, blocks=64)
        self.assertEqual(layout["placement_mode"], "sequential_channel")
        self.assertEqual(layout["page_table_runs"], ((0, 64 * 256, 0, 0, 16),))
        self.assertEqual(sequential_page_table_runs(32, 4), ((0, 32, 0, 0, 4),))

    def test_block_tables_expose_compact_runs_for_sequential_reads(self):
        blocks = _KVBlockTables(1, {"full": 8})
        blocks.resize(0, "a", 3, 0)
        blocks.resize(0, "b", 2, 0)
        self.assertEqual(blocks.page_table_runs("a", "full"), ((0, 3),))
        self.assertEqual(blocks.page_table_runs("b", "full"), ((3, 2),))


class KVBlockTableTests(unittest.TestCase):
    def test_free_list_is_fifo_and_releases_append_in_logical_order(self):
        blocks = _KVBlockTables(1, {"full": 6})
        blocks.resize(0, "a", 2, 0)
        blocks.resize(0, "b", 2, 0)
        self.assertEqual(blocks.table("a")["full"], (0, 1))
        self.assertEqual(blocks.table("b")["full"], (2, 3))
        freed = blocks.resize(0, "a", 0, 2)
        self.assertEqual(freed, {"full": 1})
        self.assertNotIn("a", blocks.tables)
        # Fresh blocks come first, then the released ones in a's order.
        blocks.resize(0, "c", 4, 0)
        self.assertEqual(blocks.table("c")["full"], (4, 5, 0, 1))
        self.assertFalse(blocks.can_resize(0, "b", 3))

    def test_lazy_growth_takes_one_block_per_crossed_boundary(self):
        blocks = _KVBlockTables(2, {"full": 4})
        for pages in (1, 1, 2, 3):
            blocks.resize(1, "a", pages, len(blocks.table("a").get("full", ())))
        self.assertEqual(blocks.table("a")["full"], (0, 1, 2))
        # Each EP group allocates from its own pools.
        blocks.resize(0, "b", 2, 0)
        self.assertEqual(blocks.table("b")["full"], (0, 1))

    def test_chunk_ring_holds_at_most_its_ring_and_frees_the_newest_row_block(self):
        blocks = _KVBlockTables(1, {"full": 8, "chunk": 4}, ring=2)
        for pages in range(1, 6):
            blocks.resize(0, "a", pages, pages - 1)
        self.assertEqual(len(blocks.table("a")["full"]), 5)
        self.assertEqual(blocks.table("a")["chunk"], (0, 1))
        freed = blocks.resize(0, "a", 0, 5)
        # Page 5 (index 4) wrapped onto ring slot 0.
        self.assertEqual(freed, {"full": 4, "chunk": 0})


class KVBlockReleaseTests(unittest.TestCase):
    def test_a_release_hands_the_newest_row_block_to_the_hook(self):
        from types import SimpleNamespace as NS
        from heterollm_sim.serving import _KVLedger
        ledger = object.__new__(_KVLedger)
        ledger.groups, ledger.blocks, calls = 1, _KVBlockTables(1, {"full": 8}), []
        ledger.release_hook = lambda request_id, group, freed: calls.append((request_id, group, freed))
        state = NS(spec=NS(request_id="a"), ep_group=None)
        ledger._resize_blocks(state, 3, 0)
        ledger._resize_blocks(state, 0, 3)
        self.assertEqual(calls, [("a", 0, {"full": 2})])

    def test_a_freed_block_discards_its_staged_tail_page_inside_the_pool(self):
        from types import SimpleNamespace as NS
        from heterollm_sim.data_motion import PhysicalRuntimeContext
        calls = []
        runtime = object.__new__(PhysicalRuntimeContext)
        runtime.runtimes = {"hbf0": NS(core=NS(config=NS(page_staging="logic_die"),
                                               discard_tail_pages=lambda *args: calls.append(args)))}
        runtime.allocators = {"hbf0": NS(get_allocation=lambda buffer, generation: NS(base_address=1 << 20))}
        runtime.discard_staged_range("hbf0", "kv:pool:l0:rank0:tp0:pp0:k", 3 * 4096, 4096)
        self.assertEqual(calls, [((1 << 20) + 3 * 4096, 4096)])


class KVBlockAddressTests(unittest.TestCase):
    def test_spans_map_through_the_table_and_coalesce_consecutive_blocks(self):
        contract = {"tokens_per_block": 4, "row_bytes": 8, "block_bytes": 32}
        owner = {"blocks": (5, 6, 2)}
        self.assertEqual(_kv_block_runs(contract, owner, ((0, 10),)), [[160, 64], [64, 16]])
        self.assertEqual(_kv_block_runs(contract, owner, ((7, 2),)), [[216, 8], [64, 8]])
        with self.assertRaises(ValueError):
            _kv_block_runs(contract, owner, ((11, 2),))

    def test_a_nand_page_never_holds_rows_of_two_blocks(self):
        config = scenario(hbf=True, kv_budget=131072, count=1, slots=1)
        plan = _parallel_plan(config)
        layer = next(layer for layer in _execution_layers(config) if not layer.is_linear_attention)
        row, stride, per_page = kv_block_geometry(config, plan, plan.rank_at(0, 0, 0), layer, 16)
        self.assertEqual(stride % 4096, 0)
        self.assertGreaterEqual(stride, 16 * row)
        self.assertEqual(per_page, 16)


class KVBlockPeakTests(unittest.TestCase):
    def test_refill_cycle_peak_matches_stepping_every_slot(self):
        ages, prompt, output, pool, per_block = [[0, 3, 5], [1, 6]], 13, 7, 2, 4
        peaks = kv_block_peak(ages, prompt, output, pool, per_block, ring=3)
        for group, slots in enumerate(ages):
            for kind, limit in (("full", None), ("chunk", 3)):
                def blocks(tokens):
                    count = -(-tokens // per_block)
                    return count if limit is None else min(count, limit)
                brute = max(sum(blocks(prompt + (age + step) % output) for age in slots)
                            for step in range(output)) + pool * blocks(prompt - 1)
                self.assertEqual(peaks[kind][group], brute)


class PhysicalBlockKVServingTests(unittest.TestCase):
    @staticmethod
    def _lazy_scenario(budget, policy="lazy"):
        config = scenario(hbf=True, kv_budget=budget, count=4, slots=4)
        return replace(config, placement=replace(config.placement,
            kv_policy=replace(config.placement.kv_policy, allocation_policy=policy)),
            workload=replace(config.workload, scheduler=replace(config.workload.scheduler,
                                                                preemption_enabled=policy == "lazy")))

    def test_lazy_admission_grows_blocks_and_preempts_when_the_pool_is_full(self):
        # 2 KiB blocks on 4 KiB NAND pages: the 128 KiB budget holds 16 blocks
        # per part. Two 8-block prompts fill it; their first output block
        # forces a preemption, as in vLLM.
        result = run_scenario(self._lazy_scenario(131072)).serving
        self.assertTrue(all(item.status is RequestStatus.FINISHED
                            for item in result.request_metrics.values()))
        self.assertEqual(result.kv_metrics.physical_block_pool_blocks, {"full": 16})
        self.assertEqual(result.kv_metrics.peak_used_pages, 16)

    def test_eager_admission_reserves_the_whole_live_extent(self):
        result = run_scenario(self._lazy_scenario(131072, "eager")).serving
        self.assertTrue(all(item.status is RequestStatus.FINISHED
                            for item in result.request_metrics.values()))
        self.assertLessEqual(result.kv_metrics.peak_used_pages, 9)

    def test_a_request_longer_than_the_pool_is_rejected(self):
        result = run_scenario(self._lazy_scenario(65536)).serving
        self.assertTrue(all(item.status is RequestStatus.REJECTED
                            for item in result.request_metrics.values()))


class SettledLayoutTest(unittest.TestCase):
    def test_settled_list_is_the_pool_and_the_seeds_take_their_histories(self):
        prompt, output, ages, pool = 40, 32, [1, 9, 17, 25], 2
        history = lambda age: -(-(prompt + age - 1) // 16)
        wanted = kv_block_peak([ages], prompt, output, pool, 16)["full"][0]
        settled = _settled_free_list(range(wanted), ages, prompt, output, pool, 16, None)
        self.assertEqual(sorted(settled), list(range(wanted)))
        # Fragmented, not the ascending order the pool starts in.
        self.assertNotEqual(settled, list(range(wanted)))
        tables = _KVBlockTables(1, {"full": wanted})
        tables.settle([ages], prompt, output, pool, 16)
        for index, age in enumerate(ages):
            tables.resize(0, str(index), history(age), 0)
        cursor = 0
        for index, age in enumerate(ages):
            self.assertEqual(tables.table(str(index))["full"], tuple(settled[cursor:cursor + history(age)]))
            cursor += history(age)

    def test_a_cycle_that_overflows_the_pool_keeps_the_ascending_list(self):
        ages = [1, 9, 17, 25]
        too_small = kv_block_peak([ages], 40, 32, 2, 16)["full"][0] - 1
        self.assertIsNone(_settled_free_list(range(too_small), ages, 40, 32, 2, 16, None))
        tables = _KVBlockTables(1, {"full": too_small})
        tables.settle([ages], 40, 32, 2, 16)
        self.assertEqual(list(tables.free[0, "full"]), list(range(too_small)))

    def test_chunk_rings_hold_at_most_their_blocks(self):
        ages = [1, 9, 17, 25]
        wanted = kv_block_peak([ages], 40, 32, 2, 16, 2)["chunk"][0]
        settled = _settled_free_list(range(wanted), ages, 40, 32, 2, 16, 2)
        self.assertEqual(sorted(settled), list(range(wanted)))


class CompactRowsTest(unittest.TestCase):
    CONTRACT = {"schema": "layer-block-kv/v1", "pool_ids": {"k": "pool:k", "v": "pool:v"}, "pool_bytes": 1 << 20,
                "tokens_per_block": 16, "row_bytes": 64, "block_bytes": 1024,
                "owners": [{"blocks": (7, 2, 3, 9), "read_ranges": [(0, 40)], "write_positions": ()},
                           {"blocks": (5, 6), "read_ranges": [(0, 20)], "write_positions": ()}]}

    def test_scattered_rows_hold_exactly_the_per_run_rows(self):
        rows = _generic_kv_read_rows(self.CONTRACT, "k")
        plain = _generic_kv_accesses(self.CONTRACT, "k")
        self.assertEqual(len(rows), 2)
        expanded = [{**rows[0], "offset_bytes": offset, "size_bytes": size} for offset, size in rows[0]["kv_runs"]]
        for row in expanded:
            del row["kv_runs"]
        self.assertEqual(expanded + rows[1:], plain)
        # A contiguous request stays an ordinary row.
        self.assertNotIn("kv_runs", rows[1])
        self.assertEqual(rows[0]["size_bytes"], 40 * 64)

    def test_run_lists_are_shared_and_survive_a_deep_copy(self):
        from copy import deepcopy
        first = _generic_kv_read_rows(self.CONTRACT, "k")[0]["kv_runs"]
        self.assertIs(_generic_kv_read_rows(self.CONTRACT, "v")[0]["kv_runs"], first)
        self.assertIs(deepcopy(first), first)
        self.assertEqual(KVRuns(list(first)), first)
        self.assertEqual(hash(KVRuns(list(first))), hash(first))

    def test_direct_subtraction_removes_a_scattered_row_by_its_runs(self):
        rows = _generic_kv_read_rows(self.CONTRACT, "k")
        other = {**rows[0], "kv_runs": KVRuns([(0, 8), (4096, 8)]), "size_bytes": 16}
        remote = {key: value for key, value in rows[0].items() if key != "size_bytes"}
        remote["byte_count"] = rows[0]["size_bytes"]
        self.assertEqual(_subtract_direct_buffer_accesses([rows[0], other, rows[1]], [remote]), (other, rows[1]))
        with self.assertRaises(ValueError):
            _subtract_direct_buffer_accesses([other, rows[1]], [remote])


if __name__ == "__main__":
    unittest.main()
