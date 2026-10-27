"""Native batch journaling preserves request state, identity and rollback."""
import numpy as np
import pytest

from heterollm_sim import _nand_pack
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_transfer import ResourceTimeline, _TIMELINE_MAPS
from heterollm_sim.memory_types import AccessRequest
from test_nand_dense_metrics import ordered_bits
from test_nand_generic_buffered_acceleration import PAGE, config


def journal_timeline(count=96):
    names = tuple(f"r{index}" for index in range(count))
    row = ResourceTimeline(
        ready_ns={rid: index + .125 for index, rid in enumerate(names)},
        busy_ns={rid: index + .375 for index, rid in enumerate(names)},
        bytes_moved={rid: 2**54 + index for index, rid in enumerate(names)},
        last_intervals={rid: (-0., index + .125) for index, rid in enumerate(names)},
        lane_available={rid: [index + .125] for index, rid in enumerate(names)},
    )
    return row, names


def public_bits(row):
    return tuple((name, ordered_bits(getattr(row, name))) for name in _TIMELINE_MAPS)


def remember(row, names, active, scope):
    _nand_pack.remember_generic_resources(row, active, scope, names)


@pytest.mark.parametrize("depth", [1, 3])
def test_complete_native_journal_reuses_proof_and_restores_original_roots(depth):
    row, names = journal_timeline()
    before = public_bits(row)
    roots = {name: getattr(row, name) for name in _TIMELINE_MAPS}
    tokens = [row._begin_owned_transaction() for _ in range(depth)]
    row._reset_metric_recorder(0)
    scope = {"pack_pages": 1024}
    remember(row, names, set(names), scope)
    proof = scope["remembered_layout"]
    for token in tokens:
        for field, root in token.roots.items():
            if getattr(row, field) is root:
                assert set(token.entries[field]) == set(names)
            else:
                assert token.entries[field] == {}  # Replaced recorder owns its before-image.
    row._reset_metric_recorder(0)
    remember(row, names, set(reversed(names)), scope)
    assert scope["remembered_layout"] is proof
    for rid in names:
        row.ready_ns[rid] += 7.
        row.lane_available[rid][0] += 7.
    for token in reversed(tokens):
        token.rollback()
    assert public_bits(row) == before
    assert all(getattr(row, name) is root for name, root in roots.items())


def test_partial_footprint_does_not_journal_untouched_layout_or_establish_proof():
    row, names = journal_timeline()
    token = row._begin_owned_transaction()
    scope = {"pack_pages": 1024}
    active = set(names[:64])
    remember(row, names, active, scope)
    assert "remembered_layout" not in scope
    assert set(token.entries["ready_ns"]) == active
    remember(row, names, active, scope)
    assert "remembered_layout" not in scope
    assert set(token.entries["ready_ns"]) == active
    token.rollback()


@pytest.mark.parametrize("count, expected", [(63, False), (64, True)])
def test_full_layout_proof_threshold_keeps_small_requests_on_original_loop(count, expected):
    row, names = journal_timeline(count)
    token = row._begin_owned_transaction()
    scope = {"pack_pages": 1024}
    remember(row, names, set(names), scope)
    assert ("remembered_layout" in scope) is expected
    token.rollback()


@pytest.mark.parametrize("change", ["token", "roots", "entries", "root_map", "entry_map", "names"])
def test_large_request_rechecks_every_proof_binding(change):
    row, names = journal_timeline()
    token = row._begin_owned_transaction()
    scope = {"pack_pages": 1024}
    remember(row, names, set(names), scope)
    old = scope["remembered_layout"]
    if change == "token":
        token.finish()
        token = row._begin_owned_transaction()
    elif change == "roots":
        token.roots = dict(token.roots)
    elif change == "entries":
        token.entries = dict(token.entries)
    elif change == "root_map":
        token.roots["busy_ns"] = dict(token.roots["busy_ns"])
    elif change == "entry_map":
        token.entries["ready_ns"] = dict(token.entries["ready_ns"])
    else:
        names = tuple(list(names))
    remember(row, names, set(names), scope)
    assert scope["remembered_layout"] is not old
    assert scope["remembered_layout"][0] is row
    assert scope["remembered_layout"][1] is names
    token.finish()


def test_different_timeline_cannot_reuse_a_complete_proof():
    first, names = journal_timeline()
    first_token = first._begin_owned_transaction()
    scope = {"pack_pages": 1024}
    remember(first, names, set(names), scope)
    old = scope["remembered_layout"]
    second, _ = journal_timeline()
    second_token = second._begin_owned_transaction()
    remember(second, names, set(names), scope)
    assert scope["remembered_layout"] is not old
    assert set(second_token.entries["ready_ns"]) == set(names)
    second_token.rollback()
    first_token.rollback()


def test_small_request_journals_new_token_without_spending_or_reusing_old_proof():
    row, names = journal_timeline()
    token = row._begin_owned_transaction()
    scope = {"pack_pages": 1024}
    remember(row, names, set(names), scope)
    old = scope["remembered_layout"]
    token.finish()
    replacement = row._begin_owned_transaction()
    remember(row, names, {names[0]}, scope)
    assert set(replacement.entries["ready_ns"]) == {names[0]}
    assert scope["remembered_layout"] is old
    remember(row, names, set(names), scope)
    assert scope["remembered_layout"] is not old
    assert set(replacement.entries["ready_ns"]) == set(names)
    replacement.rollback()


def test_failed_original_remember_never_establishes_a_proof(monkeypatch):
    row, names = journal_timeline()
    before = public_bits(row)
    token = row._begin_owned_transaction()
    original = row._remember_resources

    def failing(resources):
        original(resources)
        raise RuntimeError("after journal prefix")

    monkeypatch.setattr(row, "_remember_resources", failing)
    scope = {"pack_pages": 1024}
    with pytest.raises(RuntimeError, match="after journal prefix"):
        remember(row, names, set(names), scope)
    assert "remembered_layout" not in scope
    token.rollback()
    assert public_bits(row) == before


def test_authored_fallback_clears_proof_before_entering_the_callback(monkeypatch):
    row, names = journal_timeline()
    token = row._begin_owned_transaction()
    scope = {"pack_pages": 1024}
    remember(row, names, set(names), scope)
    original = token.remember
    called = []

    def authored(rid):
        assert "remembered_layout" not in scope
        called.append(rid)
        original(rid)

    monkeypatch.setattr(token, "remember", authored)
    # Small requests still clear a stale proof before an unknown callback.
    remember(row, names, {names[0]}, scope)
    assert called == [names[0]]
    assert "remembered_layout" not in scope
    token.rollback()


@pytest.mark.parametrize("mode", ["single_empty", "single_nonempty", "multi"])
def test_publication_keeps_bits_order_original_lists_and_rollback(mode):
    names = ("array", "channel", "host")
    multi = mode == "multi"
    histories = {} if mode == "single_empty" else {"channel": ["write", "read"] if multi else ["write"]}
    rows = [ResourceTimeline(
        ready_ns={"host": 7., "array": -0., "channel": 5.},
        busy_ns={"channel": .125, "array": .375, "host": .625},
        bytes_moved={"host": 2**54 + 1, "channel": 2**54 + 3},
        lane_available={"array": [-0.], "channel": [5., 9.] if multi else [5.], "host": [7.]},
        lane_directions={key: list(value) for key, value in histories.items()},
    ) for _ in range(2)]
    before = [public_bits(row) for row in rows]
    originals = [[row.lane_available[rid] for rid in names] for row in rows]
    tokens = [row._begin_owned_transaction() for row in rows]
    state = np.array([[-0., .1 + .2, .125, .875], [5., 7., 8., 9.], [6., 10., 11., 12.]])
    moved = np.array([0, 2**54 + 3, 1024], dtype=np.int64)
    directions = np.array([0, 2, 1], dtype=np.int64)
    lanes = np.array([-0., 5., 13., 6.] if multi else [-0., 5., 6.])
    lane_directions = np.array([0, 1, 2, 1] if multi else [0, 2, 1], dtype=np.int64)
    starts = np.array([0, 1, 3] if multi else [0, 1, 2], dtype=np.int64)
    counts = np.array([1, 2, 1] if multi else [1, 1, 1], dtype=np.int64)
    indexes = np.arange(3, dtype=np.int64)
    active, bus, labels = set(names), (0, 1, 1), [None, "read", "write"]
    scope = {"pack_pages": 1024}
    for _ in range(2):
        for row in rows:
            row._reset_metric_recorder(0)
        _nand_pack.publish_active_timeline(rows[0], names, state, moved, directions, lanes,
            lane_directions, starts, counts, indexes, active, bus, labels, not multi, True, scope)
        _nand_pack.publish_timeline(rows[1], names, state.tolist(), moved.tolist(), directions.tolist(),
            lanes.tolist(), lane_directions.tolist(), starts.tolist(), counts.tolist(), active, (),
            list(bus), labels, not multi)
        assert public_bits(rows[0]) == public_bits(rows[1])
        for row, old in zip(rows, originals):
            assert all(row.lane_available[rid] is value for rid, value in zip(names, old))
    for row, token, saved in zip(rows, tokens, before):
        token.rollback()
        assert public_bits(row) == saved


def test_shared_pack_preserves_foreign_resources_and_each_request_bits():
    from test_token_owned_history import bits
    settings = config(channels=64, targets_per_channel=1, dies_per_target=1, planes_per_lun=2)
    pair = [PhysicalRuntimeContext(settings, "flash", capture_details=False) for _ in range(2)]
    for runtime in pair:
        runtime.runtimes["flash"].core.submit(AccessRequest("seed", "read", 0, 128 * PAGE))
        runtime.timeline.busy_ns["foreign-owner"] = .125
        runtime.timeline._touched.add("foreign-owner")
    before = [public_bits(runtime.timeline) for runtime in pair]
    tokens = [runtime.begin_transaction(owners=("flash",), _owned_timeline=True) for runtime in pair]
    requests = [AccessRequest(str(index), "read", (index + 1) * 128 * PAGE, 128 * PAGE)
                for index in range(8)]
    fast, sequential = [runtime.runtimes["flash"].core for runtime in pair]
    actual = fast.submit_many(requests)
    expected = tuple(sequential.submit(request) for request in requests)
    assert bits(actual) == bits(expected)
    assert all(row.counters.get("native_fullpage_pages") == 128 for row in actual)
    assert all("foreign-owner" not in row.counters["resource_busy_ns"] for row in actual)
    assert public_bits(pair[0].timeline) == public_bits(pair[1].timeline)
    assert "_nand_request_metric_orders" not in fast.__dict__
    for runtime, token, saved in zip(pair, tokens, before):
        assert runtime.timeline.busy_ns["foreign-owner"] == .125
        runtime.restore(token)
        assert public_bits(runtime.timeline) == saved
