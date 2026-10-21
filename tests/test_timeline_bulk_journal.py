"""Compiled sparse journaling retains the original Python undo protocol."""
import copy
import pytest

from heterollm_sim.memory_transfer import ResourceTimeline, _OwnedTimelineTransaction, _TIMELINE_MISSING
from test_resource_ledger import ledger, _same


def timeline():
    result = ResourceTimeline()
    shared = [-0.0, 1.25]
    result.lane_available = {"x": shared, "y": shared, "z": [2.]}
    result.lane_directions = {"x": ["read"], "y": ["write"], "z": [None]}
    result.ready_ns = {"x": -0.0, "y": .1 + .2}
    result.busy_ns = {"x": -0.0, "y": 3.}
    result.bytes_moved = {"x": 2**180, "y": 7}
    result.last_intervals = {"x": (-0.0, 1.), "y": (1., 2.)}
    result.directions = {"x": "read", "y": "write"}
    result._intervals = {"x": [(0., 1.)]}
    result._interval_payloads = {"x": [(0., 1., 32)]}
    result._touched = {"x"}
    return result


def visible(value):
    return copy.deepcopy({key: item for key, item in vars(value).items()
                          if key != "_owned_transactions"})


def remember(ledger, value, ids):
    ledger.remember_owned_timeline_resources(value, ids, _TIMELINE_MISSING,
        _OwnedTimelineTransaction, _OwnedTimelineTransaction.remember, ResourceTimeline._remember_resource)


def scalar(value, ids):
    for key in ids:
        value._remember_resource(key)


def same_journal(actual, expected):
    _same(actual.entries, expected.entries)
    _same(list(actual.objects.values()), list(expected.objects.values()))
    _same(actual.deleted_orders, expected.deleted_orders)
    _same(actual.added_touches, expected.added_touches)


def test_bulk_remember_keeps_replaced_roots_shared_lists_directions_and_signed_zero(ledger):
    values = [timeline(), timeline()]
    initial = [visible(value) for value in values]
    tokens = [value._begin_owned_transaction() for value in values]
    original_recorders = [value._intervals for value in values]
    for value in values:
        value._intervals = {"transient": [(7., 8.)]}
    remember(ledger, values[0], iter(("y", "new", "x", "y")))
    scalar(values[1], iter(("y", "new", "x", "y")))
    same_journal(tokens[0], tokens[1])
    for value, token, recorder in zip(values, tokens, original_recorders):
        assert not token.entries["_intervals"]
        assert token.roots["_intervals"] is recorder
        value.lane_available["x"][0] = 9.
        value.lane_directions.pop("y")
        value.busy_ns["new"] = 4.
        value._touched.add("new")
        token.rollback()
        assert value.lane_available["x"] is value.lane_available["y"]
        assert value._intervals is recorder
    _same(visible(values[0]), initial[0])
    _same(visible(values[1]), initial[1])


def test_bulk_remember_preserves_each_nested_transaction_first_image(ledger):
    values = [timeline(), timeline()]
    initial = visible(values[0])
    outer = [value._begin_owned_transaction() for value in values]
    remember(ledger, values[0], ("x",))
    scalar(values[1], ("x",))
    for value in values:
        value.lane_available["x"][0] = 3.
    inner = [value._begin_owned_transaction() for value in values]
    remember(ledger, values[0], ("y", "z", "new"))
    scalar(values[1], ("y", "z", "new"))
    same_journal(outer[0], outer[1])
    same_journal(inner[0], inner[1])
    for value, token in zip(values, inner):
        value.lane_available["y"][0] = 7.
        value.lane_directions.pop("z")
        value.busy_ns["new"] = 2.
        token.rollback()
        assert value.lane_available["x"][0] == 3.
    _same(visible(values[0]), visible(values[1]))
    for token in outer:
        token.rollback()
    _same(visible(values[0]), initial)
    _same(visible(values[1]), initial)


def test_core_accounting_keeps_metric_before_images_and_outer_physical_undo(ledger):
    value = timeline()
    initial = visible(value)
    outer = value._begin_owned_transaction()
    before = value._core_metrics_snapshot()
    accounting = before["_owned_metric_transaction"]
    assert set(accounting.roots) == {"busy_ns", "bytes_moved", "last_intervals"}
    remember(ledger, value, ("x", "new"))
    value.lane_available["x"][0] = 42.
    value.busy_ns["x"] = 7.
    value.bytes_moved["x"] += 64
    value.last_intervals["x"] = (2., 9.)
    value._touched.update(("x", "new"))
    assert before["busy_ns"]["x"] == -0.
    assert before["bytes_moved"]["x"] == 2**180
    assert before["last_intervals"]["x"] == (-0., 1.)
    with pytest.raises(ValueError, match="do not own physical rollback"):
        accounting.rollback()
    accounting.finish()
    outer.rollback()
    _same(visible(value), initial)


def test_bulk_remember_keeps_dynamic_generator_and_unknown_callback_boundaries(ledger):
    values = [timeline(), timeline()]
    events = [[], []]
    outer = [value._begin_owned_transaction() for value in values]
    def resources(value, log, token):
        log.append("yield x")
        yield "x"
        original = token.remember
        def transaction_callback(key):
            log.append(("transaction", key))
            original(key)
        token.remember = transaction_callback
        log.append("yield y")
        yield "y"
        original_resource = value._remember_resource
        def timeline_callback(key):
            log.append(("timeline", key))
            original_resource(key)
        value._remember_resource = timeline_callback
        log.append("yield z")
        yield "z"
    remember(ledger, values[0], resources(values[0], events[0], outer[0]))
    scalar(values[1], resources(values[1], events[1], outer[1]))
    _same(events[0], events[1])
    same_journal(outer[0], outer[1])


def test_bulk_remember_keeps_completed_prefix_if_resource_generator_raises(ledger):
    values = [timeline(), timeline()]
    tokens = [value._begin_owned_transaction() for value in values]
    def resources():
        yield "x"
        yield "y"
        raise RuntimeError("authored generator")
    import pytest
    with pytest.raises(RuntimeError, match="authored generator"):
        remember(ledger, values[0], resources())
    with pytest.raises(RuntimeError, match="authored generator"):
        scalar(values[1], resources())
    same_journal(tokens[0], tokens[1])


def test_bulk_remember_observes_transaction_timeline_changed_by_resource_generator(ledger):
    values = [timeline(), timeline()]
    tokens = [value._begin_owned_transaction() for value in values]
    foreign = [timeline(), timeline()]
    def resources(token, replacement):
        yield "x"
        token.timeline = replacement
        yield "y"
    remember(ledger, values[0], resources(tokens[0], foreign[0]))
    scalar(values[1], resources(tokens[1], foreign[1]))
    same_journal(tokens[0], tokens[1])
    assert "x" in tokens[0].entries["ready_ns"]
    assert "y" not in tokens[0].entries["ready_ns"]


def test_native_before_get_keeps_list_preimage_defaults_and_authored_getitem(monkeypatch):
    from heterollm_sim.memory_transfer import _TimelineBefore
    value = timeline()
    token = value._begin_owned_transaction()
    before = token.before_lanes()
    assert before.get("missing", "default") == "default"
    value._remember_resource("x")
    value.lane_available["x"][0] = 12.
    _same(before.get("x"), (-0.0, 1.25))
    original = _TimelineBefore.__getitem__
    events = []
    def authored(self, key):
        events.append(key)
        if key == "custom":
            return "authored"
        return original(self, key)
    monkeypatch.setattr(_TimelineBefore, "__getitem__", authored)
    assert before.get("custom") == "authored"
    assert before.get("absent", "fallback") == "fallback"
    assert events == ["custom", "absent"]
