"""Native metadata loops against the authored full-page Python state machine."""
from types import SimpleNamespace

import numpy as np
import pytest

from heterollm_sim import _hbf_page_state
from heterollm_sim.hbf_staging import (HbfStagingController, _BufferReadState,
    _JournalDict, _Page, _PageRangeCode, _use_page_helpers)
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


def test_authored_page_attribute_access_uses_original_metadata_loops(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    names = []

    def observe(page, name):
        names.append(name)
        return object.__getattribute__(page, name)

    monkeypatch.setattr(_Page, "__getattribute__", observe)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    assert not _use_page_helpers(staging, "K", 0, 0, 24)
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert "valid" in names and "dirty" in names


@pytest.mark.parametrize("name", ["_full_write_page_journals", "_prepare_empty_gc_write"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_authored_full_write_evidence_cannot_enable_range_undo(monkeypatch, name, scope):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()

    def forbidden(*args, **kwargs):
        raise AssertionError("authored full-write evidence reached the native path")

    monkeypatch.setattr(staging if scope == "instance" else HbfStagingController,
                        name, forbidden)
    assert not _use_page_helpers(staging, "K", 0, 0, 24)
    outer = staging.begin_transaction()
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert outer["new_page_ranges"] == {}
    assert len(outer["pages"]) == 24
    staging.rollback_transaction(outer)
    assert not staging._pages


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


def test_compact_clean_page_ranges_materialize_as_independent_mutable_lists(monkeypatch):
    from heterollm_sim.hbf_staging import _PageRangeCode

    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    first, second = staging._pages[("K", 0, 0)], staging._pages[("K", 0, 1)]
    assert first._valid is second._valid is _PageRangeCode.FULL
    assert first._dirty is second._dirty is _PageRangeCode.EMPTY
    staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert first._valid is second._valid is _PageRangeCode.FULL
    assert first._dirty is second._dirty is _PageRangeCode.EMPTY
    first.valid.append((12, 16))
    first.dirty.append((12, 16))
    assert second.valid == [(0, PAGE)]
    assert second.dirty == []
    assert first.valid == [(0, PAGE), (12, 16)]
    assert first.dirty == [(12, 16)]


def test_explicit_range_lists_and_public_snapshots_keep_independent_ownership():
    from dataclasses import asdict, replace

    valid, dirty = [(16, 32)], [(16, 24)]
    page = _Page(valid, dirty)
    valid.append((48, 64))
    assert page.valid is valid and page.dirty is dirty
    first, second = _Page(), _Page()
    first.valid.append((1, 2))
    first.dirty.append((1, 2))
    assert second.valid == second.dirty == []
    assert asdict(page) == {"valid": [(16, 32), (48, 64)], "dirty": [(16, 24)],
        "slot": None, "flash": None, "_ready_ns": 0.0, "_read_state": None, "_read_index": 0}
    assert replace(page, slot=3) == _Page(valid, dirty, 3)


def test_new_dense_state_is_initialized_at_first_read_with_exact_page_values(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert not staging._read_states
    initialize, initialized = _hbf_page_state.initialize_read_state, []
    def observe(*args):
        initialized.append(args[1])
        return initialize(*args)
    monkeypatch.setattr(_hbf_page_state, "initialize_read_state", observe)
    staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert initialized == [("K", 0)]
    state = staging._read_states[("K", 0)]
    assert np.all(state.valid[:24])
    for index in range(24):
        page = staging._pages[("K", 0, index)]
        assert page._read_state is state
        assert page._read_index == index
        assert state.addresses[index] == staging._address(*page.flash)
        assert state.ready[index] == page.ready_ns

    staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert initialized == [("K", 0)]


def test_custom_completion_scalars_keep_dense_state_construction_at_first_read(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()

    class Completion(float):
        pass

    def write_pages(addresses, arrivals):
        return [Completion(value) for value in hardware.write_pages(addresses, arrivals)]

    hardware.io.write_flash_pages = write_pages
    staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert not staging._read_states
    assert type(staging._pages[("K", 0, 0)].ready_ns) is Completion
    staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
    assert ("K", 0) in staging._read_states


def test_dense_read_addresses_do_not_alias_authored_bulk_callback_input(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        def write_pages(addresses, arrivals):
            result = hardware.write_pages(addresses, arrivals)
            addresses[:] = [0] * len(addresses)
            return result
        hardware.io.write_flash_pages = write_pages
        ends = [staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io),
                staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)]
        state = staging._read_states[("K", 0)]
        assert list(state.addresses[:24]) == [
            staging._address(*staging._pages[("K", 0, index)].flash) for index in range(24)]
        outcomes.append((staging.snapshot(), hardware.events, ends))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("large_integer", [False, True])
def test_scalar_completion_type_before_first_read_is_never_coerced_early(monkeypatch, large_integer):
    outcomes = []
    class Completion(float):
        pass
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        assert not staging._read_states
        def scalar_write(address, size, arrival):
            value = hardware.access("flash_write", address, size, arrival)
            return 10**400 if large_integer else Completion(value)
        hardware.io.write_flash = scalar_write
        end = staging.write("K", 0, 0, PAGE, 0., hardware.io)
        assert type(end) is (int if large_integer else Completion)
        assert type(staging._pages[("K", 0, 0)].ready_ns) is type(end)
        before_read = staging.snapshot()
        if large_integer:
            with pytest.raises(OverflowError):
                staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
        else:
            staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
        outcomes.append((before_read, staging.snapshot(), hardware.events, end))
    assert outcomes[0] == outcomes[1]


def _page_mapping_order(staging):
    return (tuple(staging._pages),
        tuple((owner, tuple(indexes)) for owner, indexes in staging._buffer_pages.items()),
        tuple((block, tuple(owners.items())) for block, owners in staging._owners.items()))


def test_new_page_range_nested_partial_tail_release_and_rewrite_match_individual_undo(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        for index in (9, 1, 4):
            staging.write("old-sparse", 0, index * PAGE, 64, 0., hardware.io)
        before, before_order = staging.snapshot(), _page_mapping_order(staging)
        outer = staging.begin_transaction()
        staging.write("new", 0, 0, 24 * PAGE, 0., hardware.io)
        if native:
            assert outer["new_page_ranges"] == {("new", 0): (0, 24)}
            assert not outer["pages"]
        staging.read("new", 0, PAGE + 64, 64, 0., hardware.io)
        staging.write("new", 0, 5 * PAGE + 64, 64, 0., hardware.io)
        # This run intersects the compressed prefix and appends sixteen
        # uncompressed pages. Only the latter keep ordinary None preimages.
        staging.write("new", 0, 8 * PAGE, 32 * PAGE, 0., hardware.io)
        staging.read("new", 0, 0, 40 * PAGE, 0., hardware.io)
        if native:
            new_undo = {key: value for key, value in outer["pages"].items() if key[:2] == ("new", 0)}
            assert tuple(new_undo) == tuple(("new", 0, i) for i in range(24, 40))
            assert all(value is None for value in new_undo.values())
            assert not any(key[:2] == ("new", 0) for key in outer["page_ready"])
        after_outer, after_order = staging.snapshot(), _page_mapping_order(staging)
        inner = staging.begin_transaction()
        staging.release("new", 0)
        staging.write("new", 0, 0, 24 * PAGE, 0., hardware.io)
        assert ("new", 0) not in inner["new_page_ranges"]  # owner existed at inner begin
        staging.rollback_transaction(inner)
        assert staging.snapshot() == after_outer
        assert _page_mapping_order(staging) == after_order
        inner = staging.begin_transaction()
        staging.write("child", 0, 0, 16 * PAGE, 0., hardware.io)
        staging.commit_transaction(inner)
        # Sparse growth lies outside the proven new prefix and exercises the
        # ordinary dense-read-state detachment and per-page undo path.
        staging.write("new", 0, 50_000 * PAGE, 64, 0., hardware.io)
        staging.rollback_transaction(outer)
        assert staging.snapshot() == before
        assert _page_mapping_order(staging) == before_order
        outcomes.append((staging.snapshot(), hardware.events, _page_mapping_order(staging)))
    assert outcomes[0] == outcomes[1]


def test_released_old_owner_is_not_new_to_outer_token_but_is_new_to_inner_token(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        staging.read("K", 0, 0, 24 * PAGE, 0., hardware.io)
        before, before_order = staging.snapshot(), _page_mapping_order(staging)
        outer = staging.begin_transaction()
        staging.release("K", 0)
        released = staging.snapshot()
        inner = staging.begin_transaction()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        assert not outer["new_page_ranges"]
        assert all(page is not None for page in outer["pages"].values())
        if native:
            assert inner["new_page_ranges"] == {("K", 0): (0, 24)}
            assert not inner["pages"]
        staging.rollback_transaction(inner)
        assert staging.snapshot() == released
        inner = staging.begin_transaction()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        staging.commit_transaction(inner)
        staging.rollback_transaction(outer)
        assert staging.snapshot() == before
        assert _page_mapping_order(staging) == before_order
        outcomes.append((staging.snapshot(), hardware.events, _page_mapping_order(staging)))
    assert outcomes[0] == outcomes[1]


def test_real_live_page_gc_inside_new_range_needs_no_individual_page_preimages(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging = HbfStagingController(config(channels=1, host_channels=1,
            blocks_per_plane=8), 4 * PAGE)
        hardware = PageIO()
        before = staging.snapshot()
        outer = staging.begin_transaction()
        ends = [staging.write("new", 0, 0, 24 * PAGE, 0., hardware.io)]
        # Use the final four free pages, invalidating one live page in each
        # of four old blocks. The fifth overwrite must copy three live pages.
        for index in (0, 4, 8, 12, 16):
            ends.append(staging.write("new", 0, index * PAGE, PAGE, 0., hardware.io))
        assert staging.stats["gc_cycles"] > 0 and staging.stats["gc_copied_pages"] > 0
        ends.append(staging.read("new", 0, 0, 24 * PAGE, 0., hardware.io))
        after_gc = staging.snapshot()
        after_order = _page_mapping_order(staging)
        if native:
            assert outer["new_page_ranges"] == {("new", 0): (0, 24)}
            assert not outer["pages"] and not outer["page_ready"]
        staging.release("new", 0)
        staging.rollback_transaction(outer)
        assert staging.snapshot() == before
        outcomes.append((after_gc, after_order, ends, hardware.events, staging.snapshot()))
    assert outcomes[0] == outcomes[1]


def test_range_qualification_never_rehashes_or_compares_authored_owner_keys(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    calls = []
    class AuthoredOwner:
        def __hash__(self):
            calls.append("hash")
            return hash("new")  # Deliberately collide with the standard new owner.
        def __eq__(self, other):
            calls.append("eq")
            return self is other
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    authored = AuthoredOwner()
    staging.write(authored, 0, 0, PAGE, 0., hardware.io)
    calls.clear()
    token = staging.begin_transaction()
    assert not calls  # The new qualification adds no authored hash/eq callback.
    assert token["buffer_owners"] is None
    staging.write("new", 0, 0, 24 * PAGE, 0., hardware.io)
    assert not token["new_page_ranges"]
    assert tuple(key for key in token["pages"] if key[0] == "new") == tuple(
        ("new", 0, i) for i in range(24))
    staging.rollback_transaction(token)


def test_custom_outer_owner_mapping_keeps_ordinary_page_evidence_without_iteration(monkeypatch):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
    class AuthoredOwnerMap(dict):
        def __iter__(self):
            raise AssertionError("new range qualification iterated an authored mapping")
    staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
    staging._buffer_pages = AuthoredOwnerMap()
    token = staging.begin_transaction()
    assert token["buffer_owners"] is None
    staging.write("new", 0, 0, 24 * PAGE, 0., hardware.io)
    assert not token["new_page_ranges"]
    assert len(token["pages"]) == 24
    staging.rollback_transaction(token)


@pytest.mark.parametrize("full,dirty", [(True, False), (True, True), (False, True)])
def test_dense_initialization_keeps_authored_list_equality_truth_and_aliases(monkeypatch, full, dirty):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        calls = []
        class ValidRanges(list):
            def __eq__(self, other):
                calls.append(("eq", tuple(other)))
                return list.__eq__(self, other)
        class DirtyRanges(list):
            def __bool__(self):
                calls.append(("bool",))
                return len(self) != 0
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        page = staging._pages[("K", 0, 4)]
        valid = ValidRanges([(0, PAGE)] if full else [])
        modified = DirtyRanges([(0, 64)] if dirty else [])
        page.valid, page.dirty = valid, modified
        state = staging._read_state(("K", 0))
        assert page.valid is valid and page.dirty is modified
        assert calls == [("eq", ((0, PAGE),))] + ([("bool",)] if full else [])
        outcomes.append((state.ready.tolist(), state.valid.tolist(), state.addresses.tolist(), calls))
    assert outcomes[0] == outcomes[1]


def test_dense_initialization_observes_page_subclass_properties(monkeypatch):
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        calls = []
        class AuthoredPage(_Page):
            @property
            def valid(self):
                calls.append("valid")
                return _Page.valid.fget(self)
            @property
            def dirty(self):
                calls.append("dirty")
                return _Page.dirty.fget(self)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        old = staging._pages[("K", 0, 4)]
        staging._pages[("K", 0, 4)] = AuthoredPage(
            [(0, PAGE)], [], old.slot, old.flash, old.ready_ns)
        state = staging._read_state(("K", 0))
        assert calls == ["valid", "dirty"]
        outcomes.append((state.ready.tolist(), state.valid.tolist(), state.addresses.tolist(), calls))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("name", ["valid", "dirty"])
def test_replaced_page_range_property_retains_the_observable_python_path(monkeypatch, name):
    descriptor = getattr(_Page, name)
    calls = []
    def read(page):
        calls.append(name)
        return descriptor.fget(page)
    monkeypatch.setattr(_Page, name, property(read, descriptor.fset))
    outcomes = []
    for native in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        calls.clear()
        assert not _use_page_helpers(staging, "K", 0, 0, 24)
        state = staging._read_state(("K", 0))
        assert calls == [name] * 24
        outcomes.append((state.ready.tolist(), state.valid.tolist(), state.addresses.tolist(), tuple(calls)))
    assert outcomes[0] == outcomes[1]


def _initialize_arrays(staging, state, typed):
    _hbf_page_state.initialize_read_state(staging, ("K", 0),
        staging._buffer_pages[("K", 0)], state,
        _PageRangeCode.FULL, _PageRangeCode.EMPTY, _Page,
        _BufferReadState if typed else object)


@pytest.mark.parametrize("arrays", ["ordinary", "strided", "unaligned", "int32", "float32", "uint8",
    "readonly_addresses", "readonly_valid", "readonly_ready", "subclass"])
def test_dense_array_views_keep_numpy_dtypes_strides_flags_and_write_order(monkeypatch, arrays):
    outcomes = []
    for typed in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        state = _BufferReadState(24)
        calls = []
        class AuthoredArray(np.ndarray):
            def __setitem__(self, key, value):
                calls.append((self.tag, key))
                return super().__setitem__(key, value)
        if arrays == "strided":
            state.addresses = np.full(64, -1, dtype=np.int64)[::2]
        elif arrays == "unaligned":
            state.addresses = np.ndarray((32,), np.int64, np.zeros(257, dtype=np.uint8), offset=1)
            state.addresses.fill(-1)
        elif arrays == "int32":
            state.addresses = state.addresses.astype(np.int32)
        elif arrays == "float32":
            state.ready = state.ready.astype(np.float32)
        elif arrays == "uint8":
            state.valid = state.valid.astype(np.uint8)
        elif arrays.startswith("readonly_"):
            getattr(state, arrays.removeprefix("readonly_")).flags.writeable = False
        elif arrays == "subclass":
            for name in ("addresses", "valid", "ready"):
                value = getattr(state, name).view(AuthoredArray)
                value.tag = name
                setattr(state, name, value)
        error = None
        try:
            _initialize_arrays(staging, state, typed)
        except ValueError as exc:
            error = (type(exc), str(exc))
        if arrays == "subclass":
            assert calls == [(name, i) for i in range(24)
                             for name in ("addresses", "valid", "ready")]
        if arrays.startswith("readonly_"):
            assert error is not None
            assert state.ready[0] == 0.
            assert bool(state.valid[0]) is (arrays == "readonly_ready")
            assert bool(state.addresses[0] >= 0) is (arrays != "readonly_addresses")
        else:
            assert error is None
        outcomes.append((state.addresses.tolist(), state.valid.tolist(),
                         state.ready.tolist(), calls, error))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("hook", ["range_equality", "dirty_truth", "page_property",
                                  "flash_divmod", "ready_conversion"])
def test_dense_array_views_release_before_author_callbacks_and_remain_numpy(monkeypatch, hook):
    outcomes = []
    for typed in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        state = _BufferReadState(24)
        calls = []
        def make_readonly():
            calls.append("author")
            # Buffer exports must be gone before arbitrary authored code can
            # change writeability, shape or storage. Later writes stay NumPy.
            state.addresses.flags.writeable = False
        page = staging._pages[("K", 0, 4)]
        class ValidRanges(list):
            def __eq__(self, other):
                make_readonly()
                return list.__eq__(self, other)
        class DirtyRanges(list):
            def __bool__(self):
                make_readonly()
                return False
        class AuthoredPage(_Page):
            @property
            def valid(self):
                make_readonly()
                return _Page.valid.fget(self)
        class FlashBlock(int):
            def __divmod__(self, divisor):
                make_readonly()
                return divmod(int(self), divisor)
        class Ready:
            def __float__(self):
                make_readonly()
                return 900.
        if hook == "range_equality":
            page.valid = ValidRanges([(0, PAGE)])
        elif hook == "dirty_truth":
            page.dirty = DirtyRanges()
        elif hook == "page_property":
            staging._pages[("K", 0, 4)] = AuthoredPage(
                [(0, PAGE)], [], page.slot, page.flash, page.ready_ns)
        elif hook == "flash_divmod":
            page.flash = (FlashBlock(page.flash[0]), page.flash[1])
        else:
            page._ready_ns = Ready()
        with pytest.raises(ValueError, match="read-only"):
            _initialize_arrays(staging, state, typed)
        assert calls == ["author"]
        # ready conversion is the last action of this page, after its two
        # metadata stores; every other callback precedes the address store.
        assert bool(state.valid[4]) is (hook == "ready_conversion")
        assert state.valid[5] == False
        outcomes.append((state.addresses.tolist(), state.valid.tolist(),
                         state.ready.tolist(), calls))
    assert outcomes[0] == outcomes[1]


def test_dense_array_views_release_before_resize_and_keep_partial_numpy_failure(monkeypatch):
    outcomes = []
    for typed in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        state = _BufferReadState(24)
        class ValidRanges(list):
            def __eq__(self, other):
                for values in (state.addresses, state.valid, state.ready):
                    values.resize(10, refcheck=False)
                return list.__eq__(self, other)
        staging._pages[("K", 0, 4)].valid = ValidRanges([(0, PAGE)])
        with pytest.raises(IndexError, match="index 10"):
            _initialize_arrays(staging, state, typed)
        assert state.valid.all() and state.ready.shape == (10,)
        outcomes.append((state.addresses.tolist(), state.valid.tolist(), state.ready.tolist()))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("field,value", [("flash", (1 << 70, 0)),
                                        ("_ready_ns", 10 ** 400)])
def test_dense_array_views_keep_numpy_overflow_and_preceding_assignments(monkeypatch, field, value):
    outcomes = []
    for typed in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        state = _BufferReadState(24)
        page = staging._pages[("K", 0, 4)]
        setattr(page, field, value)
        with pytest.raises(OverflowError):
            _initialize_arrays(staging, state, typed)
        assert bool(state.valid[4]) is (field == "_ready_ns")
        assert page._read_state is None
        outcomes.append((state.addresses.tolist(), state.valid.tolist(), state.ready.tolist()))
    assert outcomes[0] == outcomes[1]


def test_dense_array_views_keep_numpy_bounds_failure_after_the_completed_prefix(monkeypatch):
    outcomes = []
    for typed in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        state = _BufferReadState(4)
        with pytest.raises(IndexError, match="index 4"):
            _initialize_arrays(staging, state, typed)
        assert state.valid.all()
        assert staging._pages[("K", 0, 3)]._read_state is state
        assert staging._pages[("K", 0, 4)]._read_state is None
        outcomes.append((state.addresses.tolist(), state.valid.tolist(), state.ready.tolist()))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("kind", ["owner", "index"])
def test_dense_array_views_reject_colliding_author_keys_before_any_lookup(monkeypatch, kind):
    outcomes = []
    for typed in (False, True):
        monkeypatch.setattr(_hbf_page_state, "COMPILED", True)
        staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
        staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
        state = _BufferReadState(24)
        calls = []
        armed = False
        class AuthoredKey:
            def __hash__(self):
                if armed:
                    calls.append("hash")
                return hash("K" if kind == "owner" else 0)
            def __eq__(self, other):
                if armed:
                    calls.append("eq")
                    state.addresses.flags.writeable = False
                return self is other
        authored = AuthoredKey()
        if kind == "owner":
            staging._buffer_pages[(authored, 0)] = {0: None}
            first = (authored, 0, 0)
        else:
            indexes = staging._buffer_pages[("K", 0)]
            indexes[authored] = None
            first = ("K", 0, authored)
        # Put the colliding authored key before the ordinary key, so even
        # the first builtin-index lookup can execute its equality callback.
        staging._pages = {first: _Page([(0, PAGE)], [], None, (0, 0), 0.),
                          **staging._pages}
        calls.clear()
        armed = True
        with pytest.raises(ValueError, match="read-only"):
            _initialize_arrays(staging, state, typed)
        assert "eq" in calls
        assert not state.valid.any()
        outcomes.append((state.addresses.tolist(), state.valid.tolist(),
                         state.ready.tolist(), calls))
    assert outcomes[0] == outcomes[1]


def test_replaced_page_setattr_disables_native_views_before_the_binding_hook(monkeypatch):
    outcomes = []
    for native in (False, True):
        with monkeypatch.context() as patch:
            patch.setattr(_hbf_page_state, "COMPILED", native)
            staging, hardware = HbfStagingController(config(), 4 * PAGE), PageIO()
            staging.write("K", 0, 0, 24 * PAGE, 0., hardware.io)
            calls = []
            def assign(page, name, value):
                if name == "_read_state":
                    calls.append(name)
                    # This is supported before initialization; it must not
                    # encounter newly exported typed views from that call.
                    value.addresses.resize(64, refcheck=True)
                object.__setattr__(page, name, value)
            patch.setattr(_Page, "__setattr__", assign)
            assert not _use_page_helpers(staging, "K", 0, 0, 24)
            state = staging._read_state(("K", 0))
            assert calls == ["_read_state"] * 24
            outcomes.append((state.addresses.tolist(), state.valid.tolist(), state.ready.tolist()))
    assert outcomes[0] == outcomes[1]
