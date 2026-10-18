"""Native metadata loops against the authored full-page Python state machine."""
from types import SimpleNamespace

import numpy as np
import pytest

from heterollm_sim import _hbf_page_state
from heterollm_sim.hbf_staging import HbfStagingController, _JournalDict, _Page, _use_page_helpers
from heterollm_sim.memory_types import NandConfig


PAGE = 4096


def config(**changes):
    values = dict(kind="HBF", protocol_mode="ocp_hbf_0_7", channels=3,
                  host_channels=3, queue_scope="host_channel", planes_independent=True,
                  planes_per_lun=1, parallel_units=None, blocks_per_plane=64,
                  pages_per_block=4, page_bytes=PAGE, host_granularity_bytes=64,
                  host_full_duplex=True, read_cache_pages_per_bank=2,
                  host_bandwidth_gb_s=64., internal_bandwidth_gb_s=64.,
                  page_read_ns=100., page_program_ns=1000., block_erase_ns=2000.)
    values.update(changes)
    return NandConfig(**values)


class PageIO:
    """Deterministic completion oracle; every scalar page/link stays observable."""

    def __init__(self, *, completion_type="array", decline=False, fail=False):
        self.events = []
        self.completion_type = completion_type
        self.decline, self.fail = decline, fail
        self.io = SimpleNamespace(
            read_host=lambda a, n, t: self.access("host_read", a, n, t),
            write_host=lambda a, n, t: self.access("host_write", a, n, t),
            read_flash=lambda a, n, t: self.access("flash_read", a, n, t),
            write_flash=lambda a, n, t: self.access("flash_write", a, n, t),
            transfer=self.transfer, read_flash_pages=self.read_pages,
            write_flash_pages=self.write_pages)

    def access(self, operation, address, count, arrival):
        self.events.append((operation, address, count, arrival))
        # Vary completions within a stripe; do not accidentally test only
        # monotone arrivals or one identical per-bank dependency.
        return arrival + count / 64. + (address // PAGE % 5) * 0.25

    def transfer(self, route, count, arrival):
        self.events.append((route, 0, count, arrival))
        return arrival + count / 32.

    def write_pages(self, addresses, arrivals):
        if self.fail:
            raise RuntimeError("media failure")
        if self.decline:
            return None
        values = [self.access("flash_write", address, PAGE,
                             self.transfer("write_hbf", PAGE, arrival))
                  for address, arrival in zip(addresses, arrivals)]
        if self.completion_type == "tuple":
            return tuple(values)
        return np.asarray(values) if self.completion_type == "array" else values

    def read_pages(self, addresses, arrivals):
        if np.isscalar(arrivals):
            arrivals = [arrivals] * len(addresses)
        return np.asarray([self.transfer("read_hbf", PAGE,
            self.access("flash_read", int(address), PAGE, float(arrival)))
            for address, arrival in zip(addresses, arrivals)])


def exercise_pages(monkeypatch, native, completion_type):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    staging = HbfStagingController(config(), 4 * PAGE)
    hardware = PageIO(completion_type=completion_type)
    assert _use_page_helpers(staging, "K", 7, 0, 24) is native
    snapshots, completions = [], []
    for operation, offset, size in (
            ("write", 0, 24 * PAGE), ("read", 0, 24 * PAGE),
            ("write", 5 * PAGE + 64, 128),
            ("write", 0, 24 * PAGE), ("read", 0, 24 * PAGE),
            ("write", 24 * PAGE, 16 * PAGE), ("read", 0, 40 * PAGE),
            ("write", 50_000 * PAGE, 64)):
        completions.append(getattr(staging, operation)("K", 7, offset, size, 0., hardware.io))
        snapshots.append(staging.snapshot())
    assert not staging._read_states  # Sparse growth discarded its dense cache.
    completions.append(staging.release("K", 7))
    snapshots.append(staging.snapshot())
    return snapshots, completions, hardware.events


@pytest.mark.parametrize("completion_type", ["array", "list", "tuple"])
def test_full_page_metadata_matches_python_across_overwrite_tail_and_sparse_growth(
        monkeypatch, completion_type):
    reference = exercise_pages(monkeypatch, False, completion_type)
    candidate = exercise_pages(monkeypatch, True, completion_type)
    assert candidate == reference


def test_full_page_nested_journals_restore_existing_dense_pages_and_owner_order(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 40 * PAGE, 0., hardware.io)
        staging.read("K", 0, 0, 40 * PAGE, 0., hardware.io)
        before = staging.snapshot()
        outer = staging.begin_transaction()
        staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io)
        staging.read("K", 0, 0, 40 * PAGE, 0., hardware.io)
        after_outer = staging.snapshot()
        inner = staging.begin_transaction()
        staging.write("K", 0, 40 * PAGE, 16 * PAGE, 0., hardware.io)
        staging.read("K", 0, 0, 56 * PAGE, 0., hardware.io)
        staging.release("K", 0)
        staging.rollback_transaction(inner)
        assert staging.snapshot() == after_outer
        inner = staging.begin_transaction()
        staging.write("V", 0, 0, 16 * PAGE, 0., hardware.io)
        staging.commit_transaction(inner)
        staging.rollback_transaction(outer)
        assert staging.snapshot() == before
        staging.restore(after_outer)
        staging.release("K", 0)
        staging.restore(after_outer)
        assert staging.snapshot() == after_outer
        outcomes.append((staging.snapshot(), hardware.events))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("failure", ["decline", "raise"])
def test_bulk_prepare_rolls_back_every_frontier_when_media_declines_or_raises(monkeypatch, failure):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging = HbfStagingController(config(), 4 * PAGE)
        hardware = PageIO(decline=failure == "decline", fail=failure == "raise")
        before = staging.snapshot()
        if failure == "raise":
            with pytest.raises(RuntimeError, match="media failure"):
                staging._write_full_pages("K", 0, 0, 24, 0., hardware.io,
                                          hardware.write_pages)
        else:
            assert staging._write_full_pages("K", 0, 0, 24, 0., hardware.io,
                                              hardware.write_pages) is None
        assert staging.snapshot() == before
        assert not hardware.events
        outcomes.append(staging.snapshot())
    assert outcomes[0] == outcomes[1]


def test_gc_capacity_boundary_retains_scalar_collection_and_page_order(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging = HbfStagingController(config(channels=1, host_channels=1,
                                             blocks_per_plane=8), 4 * PAGE)
        hardware = PageIO()
        ends = [staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io)]
        # Only twelve pages remain free: this full run must enter the
        # original scalar allocation/collection path despite native support.
        ends.append(staging.write("K", 0, 0, 16 * PAGE, 0., hardware.io))
        ends.append(staging.read("K", 0, 0, 16 * PAGE, 0., hardware.io))
        assert staging.stats["gc_cycles"] > 0
        outcomes.append((staging.snapshot(), hardware.events, ends))
    assert outcomes[0] == outcomes[1]


def test_native_helpers_preserve_overridden_controller_callbacks(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    original, calls = staging._allocate, []

    def observe(time, io, key):
        calls.append(key)
        return original(time, io, key)

    staging._allocate = observe
    assert not _use_page_helpers(staging, "K", 0, 0, 24)
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert calls == [("K", 0, index) for index in range(24)]


def test_custom_nested_index_and_journal_reads_keep_python_semantics(monkeypatch):
    from collections import OrderedDict

    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    staging._buffer_pages[("K", 0)] = OrderedDict(staging._buffer_pages[("K", 0)])
    assert not _use_page_helpers(staging, "K", 0, 0, 24)
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert isinstance(staging._buffer_pages[("K", 0)], OrderedDict)
    staging._buffer_pages[("K", 0)] = dict(staging._buffer_pages[("K", 0)])
    original, calls = _JournalDict.get, []

    def observe(mapping, key, default=None):
        calls.append(key)
        return original(mapping, key, default)

    monkeypatch.setattr(_JournalDict, "get", observe)
    assert not _use_page_helpers(staging, "K", 0, 0, 24)
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert calls


def test_full_page_helpers_retain_page_ready_property_and_mapping_aliases(monkeypatch):
    original = _Page.ready_ns
    observations = []

    def read_ready(page):
        value = original.fget(page)
        observations.append(("read", value))
        return value

    def write_ready(page, value):
        observations.append(("write", value))
        original.fset(page, value)

    monkeypatch.setattr(_Page, "ready_ns", property(read_ready, write_ready))
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
        # Existing callers retaining these containers must see the same
        # in-place mutations rather than a replaced metadata representation.
        pages, owners, indexes = staging._pages, staging._owners, staging._buffer_pages[("K", 0)]
        before = len(observations)
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        assert any(kind == "write" for kind, value in observations[before:])
        assert staging._pages is pages and staging._owners is owners
        assert staging._buffer_pages[("K", 0)] is indexes
        outcomes.append((staging.snapshot(), hardware.events))
    assert outcomes[0] == outcomes[1]


def test_source_simulation_uses_original_loops_without_per_page_helpers(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("source simulation entered native metadata helper")

    for name in ("count_missing_pages", "prepare_full_write", "commit_full_write",
                 "initialize_read_state"):
        monkeypatch.setattr(_hbf_page_state, name, forbidden)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert len(staging._pages) == 24
