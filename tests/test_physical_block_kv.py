"""Physical KV in per-rank, per-layer block pools addressed through request block tables."""
from collections import deque
import copy
from dataclasses import replace
import math
from pathlib import Path
import itertools
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.kv_placement import KVTailRing, kv_stripe_layout
from heterollm_sim.native_stationary import kv_block_peak
from heterollm_sim.memory_types import KVRuns
from heterollm_sim.planner import (_generic_kv_accesses, _generic_kv_read_rows, _kv_block_runs, kv_block_geometry,
                                   _parallel_plan, _execution_layers, _subtract_direct_buffer_accesses)
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import RequestStatus, _KVBlockTables, _settled_free_list


class KVPlacementTests(unittest.TestCase):
    # (group's requests, its prepared imports, prompt, output, tokens per page, banks), from the native comparison.
    POINTS = ((701, 4, 1660, 373, 2, 3200), (815, 6, 1660, 373, 2, 2800), (386, 2, 5900, 499, 2, 3200),
              (323, 2, 1660, 373, 4, 3200), (550, 4, 1660, 373, 16, 2048), (66, 2, 1660, 373, 2, 2800))

    @staticmethod
    def geometry(layout, per_page, banks, chunk=0):
        shift = (layout["pool_blocks"] * layout["stride_pages"] + layout["phase_pages"]) % banks
        return banks, per_page, layout["stride_pages"], shift, chunk

    def test_the_stride_holds_a_block_spreads_the_pool_and_needs_no_ages(self):
        for live, prepared, prompt, output, per_page, banks in self.POINTS:
            layout = kv_stripe_layout(prompt, output, per_page, banks, live + prepared)
            stride, pool = layout["stride_pages"], layout["pool_blocks"]
            self.assertGreater(pool, live + prepared)
            self.assertGreaterEqual(stride, -(-(prompt + output - 1) // per_page))
            self.assertEqual(math.gcd(stride, banks), 1)
            self.assertGreaterEqual(len({block * stride % banks for block in range(pool)}), min(pool, banks))
            # V's tail starts the history's pages after K's, so an import rolls round the banks.
            self.assertEqual(self.geometry(layout, per_page, banks)[3], -(-(prompt - 1) // per_page) % banks)

    def test_a_block_picked_at_join_keeps_unequal_lengths_off_each_others_banks(self):
        """No request's length is used, and lengths vary by half either way."""
        rng = random.Random(1)
        for live, prepared, prompt, output, per_page, banks in self.POINTS[:2]:
            longest = 2 * (prompt + output)
            layout = kv_stripe_layout(longest, 0, per_page, banks, live + prepared)
            ring = KVTailRing(*self.geometry(layout, per_page, banks))
            shift = ring.shift
            free = deque(rng.sample(range(layout["pool_blocks"]), layout["pool_blocks"]))
            tails, ends = {}, {}

            def admit(seat, step):
                rows, life = round(prompt * rng.uniform(.5, 1.5)), round(output * rng.uniform(.5, 1.5))
                options = np.fromiter(free, dtype=np.int64, count=len(free))
                block = int(options[ring.pick(options, rows - step)])
                free.remove(block)
                tails[seat] = block, ring.position(block, rows - step)
                ring.add(tails[seat][1])
                ends[seat] = step + life

            for seat in range(live):
                admit(seat, -rng.randrange(output))
            for step in range(2 * output):
                for seat in [seat for seat, end in ends.items() if end <= step]:
                    block, position = tails[seat]
                    ring.add(position, -1)
                    free.append(block)
                    admit(seat, step)
                pages = (np.array([position for _block, position in tails.values()]) + step) % ring.length // per_page
                banks_used = np.concatenate((pages, (pages + shift) % banks))
                self.assertEqual(len(np.unique(banks_used)), 2 * live)

    def test_block_tables_keep_every_tail_on_its_own_bank_through_refills(self):
        live, prepared, prompt, output, per_page, banks = 323, 2, 1660, 373, 4, 3200
        layout = kv_stripe_layout(prompt, output, per_page, banks, live + prepared)
        geometry = self.geometry(layout, per_page, banks)
        stride, shift = geometry[2], geometry[3]
        ages = [(2 * index + 1) * output // (2 * live) for index in range(live)]
        blocks = _KVBlockTables(1, {"full": layout["pool_blocks"]})
        blocks.place_seeds([ages], prompt, output, prepared, {"full": geometry})
        requests = {f"seed{index}": prompt + age for index, age in enumerate(ages)}
        for name in requests:
            blocks.resize(0, name, 1, 0)
        self.assertEqual(blocks.tail_sharing()["full"], (1, 0, 2 * live))
        # Oldest first; the two imports land in the two slots that free next.
        order, created = deque(sorted(requests, key=requests.get, reverse=True)), 0
        waiting = deque()

        def start_import():
            nonlocal created
            name = f"fresh{created}"
            created += 1
            blocks.resize(0, name, 1, 0)
            waiting.append(name)

        for _ in range(prepared):
            start_import()
        ages_now = {name: requests[name] - prompt for name in requests}
        for step in range(2 * output):
            running = {name: age + step for name, age in ages_now.items()}
            pages = [blocks.table(name)["full"][0] * stride + (prompt + age) // per_page for name, age in running.items()]
            used = [page % banks for page in pages] + [(page + shift) % banks for page in pages]
            self.assertEqual(len(set(used)), 2 * live)
            for name, age in tuple(running.items()):
                if age == output - 1:
                    blocks.resize(0, name, 0, 1)
                    del ages_now[name]
                    ages_now[waiting.popleft()] = -step - 1
                    start_import()
        self.assertEqual(len(ages_now), live)

    def test_seeds_out_of_order_are_refused_and_a_release_clears_the_tail(self):
        layout = kv_stripe_layout(1660, 373, 2, 3200, 8)
        geometry = {"full": self.geometry(layout, 2, 3200)}
        with self.assertRaises(ValueError):
            _KVBlockTables(1, {"full": layout["pool_blocks"]}).place_seeds([[4, 2, 0]], 1660, 373, 1, geometry)
        blocks = _KVBlockTables(1, {"full": layout["pool_blocks"]})
        blocks.place_seeds([[0, 2, 4]], 1660, 373, 1, geometry)
        for name in ("young", "middle", "old"):
            blocks.resize(0, name, 1, 0)
        copied = copy.deepcopy(blocks)
        blocks.resize(0, "old", 0, 1)
        self.assertEqual(blocks.tail_sharing()["full"][2], 4)
        self.assertEqual(copied.tail_sharing()["full"][2], 6)


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
