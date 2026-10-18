"""Exact victim selection after the free physical-block pool is exhausted."""
import pytest

from heterollm_sim import _hbf_page_state
from heterollm_sim.hbf_staging import HbfStagingController, _Page
from test_hbf_page_state import PageIO, config


PAGE = 4096


def scanned_victim(controller, replacing):
    candidates = []
    for block in controller._used:
        if block == controller._spare:
            continue
        owners = controller._owners.get(block, {})
        live = len(owners) - int(replacing in owners.values())
        if live < controller.config.pages_per_block:
            candidates.append((live, block))
    if not candidates:
        raise MemoryError("HBF has no reclaimable block after reserving GC space")
    return min(candidates)[1]


@pytest.mark.parametrize("native", [False, True])
def test_index_matches_full_scan_events_state_release_and_nested_rollback(monkeypatch, native):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    indexed_victim = HbfStagingController._gc_victim
    outcomes = []
    for selector in (scanned_victim, indexed_victim):
        monkeypatch.setattr(HbfStagingController, "_gc_victim", selector)
        staging = HbfStagingController(config(channels=1, host_channels=1,
                                             blocks_per_plane=12), 4 * PAGE)
        hardware = PageIO()
        ends = []
        # Forty-four free pages. A released temporary plus repeated KV
        # overwrites exhaust them, exercising empty and partially live victims.
        ends.append(staging.write("temporary", 0, 0, 16 * PAGE, 0., hardware.io))
        ends.append(staging.write("K", 0, 0, 20 * PAGE, 0., hardware.io))
        ends.append(staging.release("temporary", 0))
        snapshots = [staging.snapshot()]
        for iteration in range(4):
            ends.append(staging.write("K", 0, iteration * PAGE, 16 * PAGE,
                                      iteration * 100., hardware.io))
            ends.append(staging.read("K", 0, 0, 20 * PAGE, 0., hardware.io))
            snapshots.append(staging.snapshot())
        assert staging.stats["gc_cycles"] > 0
        outer = staging.begin_transaction()
        ends.append(staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io))
        after_outer = staging.snapshot()
        inner = staging.begin_transaction()
        ends.append(staging.write("K", 0, PAGE, 16 * PAGE, 0., hardware.io))
        staging.release("K", 0)
        staging.rollback_transaction(inner)
        assert staging.snapshot() == after_outer
        ends.append(staging.write("K", 0, 2 * PAGE, 16 * PAGE, 0., hardware.io))
        staging.rollback_transaction(outer)
        assert staging.snapshot() == snapshots[-1]
        # Restore must invalidate an already-built index, too.
        saved = staging.snapshot()
        ends.append(staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io))
        staging.restore(saved)
        ends.append(staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io))
        snapshots.append(staging.snapshot())
        # Numeric release can bypass scalar _touch_owners.
        ends.append(staging.read("K", 0, 0, 20 * PAGE, 0., hardware.io))
        ends.append(staging.release("K", 0))
        ends.append(staging.write("new", 0, 0, 16 * PAGE, 0., hardware.io))
        snapshots.append(staging.snapshot())
        outcomes.append((snapshots, ends, hardware.events))
    assert outcomes[0] == outcomes[1]


def test_full_block_becomes_candidate_only_for_the_replacing_page():
    staging = HbfStagingController(config(channels=1, host_channels=1,
                                         blocks_per_plane=4), 4 * PAGE)
    staging._used.update({0: 4, 1: 4, 2: 4})
    for block in range(3):
        staging._owners[block] = {index: ("K", 0, block * 4 + index) for index in range(4)}
        for index, key in staging._owners[block].items():
            staging._pages[key] = _Page(flash=(block, index))
    key = ("K", 0, 9)
    assert staging._gc_victim(key) == scanned_victim(staging, key) == 2
    with pytest.raises(MemoryError):
        staging._gc_victim(("absent", 0, 0))
    staging._touch_owners(1)
    staging._owners[1].pop(0)
    assert staging._gc_victim(key) == scanned_victim(staging, key) == 1


def test_index_updates_only_changed_blocks_after_initial_build():
    staging = HbfStagingController(config(channels=1, host_channels=1,
                                         blocks_per_plane=4096), 4 * PAGE)
    staging._used.update({block: 4 for block in range(4095)})
    assert staging._gc_victim(("absent", 0, 0)) == 0
    # Once initialized, victim lookup must never iterate the device-wide map.
    class NoIteration(dict):
        def __iter__(self):
            raise AssertionError("rescanned all physical blocks")
    staging._used = NoIteration(staging._used)
    for block in range(100):
        staging._touch_owners(block)
        staging._used.pop(block)
        assert staging._gc_victim(("absent", 0, 0)) == block + 1
    assert len(staging._gc_heap) <= 2 * len(staging._gc_entries) + 1024


@pytest.mark.parametrize("native", [False, True])
def test_full_page_native_commit_invalidates_old_and_new_block_counts(monkeypatch, native):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    staging = HbfStagingController(config(channels=1, host_channels=1,
                                         pages_per_block=32, blocks_per_plane=8), 4 * PAGE)
    hardware = PageIO()
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert staging._gc_victim(("absent", 0, 0)) == scanned_victim(staging, ("absent", 0, 0))
    for first in (0, 8, 0, 16, 0):
        staging.write("K", 0, first * PAGE, 16 * PAGE, 0., hardware.io)
        for key in (("absent", 0, 0), ("K", 0, first)):
            assert staging._gc_victim(key) == scanned_victim(staging, key)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("lanes,pages", [(1, 4), (3, 17), (4, 32)])
def test_empty_collection_bulk_matches_original_scalar_gc_events(monkeypatch, native, lanes, pages):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    bulk = HbfStagingController._write_full_pages
    indexed_victim = HbfStagingController._gc_victim

    def original_bulk(controller, *args):
        first, stop = args[2:4]
        available = len(controller._free_blocks) * pages + sum(
            pages - controller._used[block] for block in controller._active)
        return None if stop - first > available else bulk(controller, *args)

    outcomes = []
    for scalar in (True, False):
        monkeypatch.setattr(HbfStagingController, "_gc_victim",
                            scanned_victim if scalar else indexed_victim)
        monkeypatch.setattr(HbfStagingController, "_write_full_pages",
                            original_bulk if scalar else bulk)
        staging = HbfStagingController(config(channels=lanes, host_channels=lanes,
            pages_per_block=pages, blocks_per_plane=24), 4 * PAGE)
        hardware = PageIO()
        ends = []
        size = 24 * PAGE
        for iteration in range(lanes * pages * 24 // 56 + 8):
            ends.append(staging.write("K", 0, 0, size, iteration * 100., hardware.io))
            ends.append(staging.write("temporary", iteration, 0, 32 * PAGE, 0., hardware.io))
            ends.append(staging.release("temporary", iteration))
        assert staging.stats["gc_cycles"] > 0
        outcomes.append((staging.snapshot(), ends, hardware.events))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("failure", ["decline", "raise"])
def test_empty_collection_plan_rolls_back_before_media_decline_or_failure(failure):
    staging = HbfStagingController(config(channels=1, host_channels=1,
                                         blocks_per_plane=12), 4 * PAGE)
    hardware = PageIO()
    staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io)
    staging.write("temporary", 0, 0, 28 * PAGE, 0., hardware.io)
    staging.release("temporary", 0)
    before = staging.snapshot()
    events = list(hardware.events)
    hardware.decline, hardware.fail = failure == "decline", failure == "raise"
    if failure == "raise":
        with pytest.raises(RuntimeError, match="media failure"):
            staging._write_full_pages("K", 0, 0, 16, 0., hardware.io, hardware.write_pages)
    else:
        assert staging._write_full_pages("K", 0, 0, 16, 0., hardware.io, hardware.write_pages) is None
    assert staging.snapshot() == before
    assert hardware.events == events
