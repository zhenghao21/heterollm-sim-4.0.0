"""Interval merging preserves index order, comparison callbacks and float bits."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import pytest

from heterollm_sim import _resource_ledger as installed_ledger
from heterollm_sim import data_motion


ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "src/heterollm_sim/_resource_ledger.py").is_file())


def reference(target, incoming):
    for resource_id, interval in incoming.items():
        previous = target.get(resource_id)
        if previous is None or (interval[1], interval[0]) > (previous[1], previous[0]):
            target[resource_id] = interval


@pytest.fixture(params=("python", "compiled"))
def ledger(request):
    if request.param == "compiled":
        if Path(installed_ledger.__file__).suffix.lower() not in (".pyd", ".so"):
            pytest.skip("resource ledger extension is not built")
        return installed_ledger
    spec = importlib.util.spec_from_file_location(
        "pending_interval_ledger_python", ROOT / "src/heterollm_sim/_resource_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=("helper", "wrapper"))
def merge(request, ledger, monkeypatch):
    if request.param == "helper":
        return ledger.merge_latest_intervals
    monkeypatch.setattr(installed_ledger, "merge_latest_intervals", ledger.merge_latest_intervals)
    return data_motion._merge_latest_resource_intervals


class Interval:
    def __init__(self, name, values, events, fail=None):
        self.name, self.values, self.events, self.fail = name, list(values), events, fail

    def __getitem__(self, index):
        self.events.append(("get", self.name, index))
        if index == self.fail:
            raise IndexError(f"{self.name}[{index}]")
        return self.values[index]


class Number:
    def __init__(self, name, value, events, *, eq=None, gt=None, fail=None, on_eq=None):
        self.name, self.value, self.events = name, value, events
        self.eq, self.gt, self.fail, self.on_eq = eq, gt, fail, on_eq

    def __eq__(self, other):
        self.events.append(("eq", self.name, other.name))
        if self.fail == "eq":
            raise ValueError(f"{self.name}.eq")
        if self.on_eq is not None:
            self.on_eq()
        return self.value == other.value if self.eq is None else self.eq

    def __gt__(self, other):
        self.events.append(("gt", self.name, other.name))
        if self.fail == "gt":
            raise ValueError(f"{self.name}.gt")
        return self.value > other.value if self.gt is None else self.gt


def callback_case(left_end=5, right_end=5, *, index_failure=None,
                  comparison_failure=None, equal_starts=False, mutate_start=False):
    events = []
    left_start = Number("left.start", 3, events, eq=True if equal_starts else None,
                        gt=True if equal_starts else None,
                        fail=(comparison_failure or {}).get("left.start"))
    right_start = Number("right.start", 1, events)
    left_stop = Number("left.end", left_end, events,
                       fail=(comparison_failure or {}).get("left.end"))
    right_stop = Number("right.end", right_end, events)
    incoming_interval = Interval("left", (left_start, left_stop), events,
                                 (index_failure or {}).get("left"))
    previous = Interval("right", (right_start, right_stop), events,
                        (index_failure or {}).get("right"))
    if mutate_start:
        left_stop.on_eq = lambda: incoming_interval.values.__setitem__(
            0, Number("replacement.start", -100, events))
    prefix, suffix = object(), object()
    target = {"existing": (0.0, 0.0), "main": previous}
    incoming = {"prefix": prefix, "main": incoming_interval, "suffix": suffix}
    return target, incoming, events, previous, incoming_interval


def observe(merger, **kwargs):
    target, incoming, events, previous, current = callback_case(**kwargs)
    error = None
    try:
        merger(target, incoming)
    except Exception as exc:
        error = (type(exc), str(exc))
    return (events, error, tuple(target), target["main"] is current,
            target.get("prefix") is incoming["prefix"], "suffix" in target)


@pytest.mark.parametrize("left_end,right_end", [(5, 5), (9, 5), (1, 5)])
def test_all_four_index_reads_precede_comparisons(merge, left_end, right_end):
    actual = observe(merge, left_end=left_end, right_end=right_end)
    assert actual == observe(reference, left_end=left_end, right_end=right_end)
    assert actual[0][:4] == [("get", "left", 1), ("get", "left", 0),
                             ("get", "right", 1), ("get", "right", 0)]


def test_equal_start_still_observes_equality_and_skips_greater_than(merge):
    actual = observe(merge, equal_starts=True)
    assert actual == observe(reference, equal_starts=True)
    assert ("eq", "left.start", "right.start") in actual[0]
    assert ("gt", "left.start", "right.start") not in actual[0]
    assert actual[3] is False


def test_comparison_uses_start_captured_before_end_callback(merge):
    actual = observe(merge, mutate_start=True)
    assert actual == observe(reference, mutate_start=True)
    assert actual[3] is True
    assert ("gt", "left.start", "right.start") in actual[0]


@pytest.mark.parametrize("name,index", [("left", 1), ("left", 0), ("right", 1), ("right", 0)])
def test_index_failure_keeps_original_callbacks_and_successful_prefix(merge, name, index):
    actual = observe(merge, index_failure={name: index})
    assert actual == observe(reference, index_failure={name: index})
    assert actual[1] == (IndexError, f"{name}[{index}]")
    assert actual[4] is True and actual[5] is False


@pytest.mark.parametrize("field,operation,left_end", [
    ("left.end", "eq", 5), ("left.end", "gt", 9),
    ("left.start", "eq", 5), ("left.start", "gt", 5),
])
def test_comparison_failure_keeps_original_callbacks_and_prefix(merge, field, operation, left_end):
    kwargs = {"left_end": left_end, "comparison_failure": {field: operation}}
    actual = observe(merge, **kwargs)
    assert actual == observe(reference, **kwargs)
    assert actual[1] == (ValueError, f"{field}.{operation}")
    assert actual[4] is True and actual[5] is False


def test_shared_objects_use_identity_before_eq_and_gt(merge):
    class NeverCompare:
        def __eq__(self, other):
            raise AssertionError("identity equality must bypass __eq__")

        def __gt__(self, other):
            raise AssertionError("equal tuple must bypass __gt__")

    shared_end, shared_start = NeverCompare(), NeverCompare()
    previous = (shared_start, shared_end)
    current = (shared_start, shared_end)
    target = {"r": previous}
    merge(target, {"r": current})
    assert target["r"] is previous


@pytest.mark.parametrize("kind", ["shared_nan_end", "distinct_nan_end", "shared_nan_start", "signed_zero"])
@pytest.mark.parametrize("sequence", [tuple, list])
def test_nan_identity_and_float_bits_match_original(merge, kind, sequence):
    nan = float("nan")
    if kind == "shared_nan_end":
        previous, current = sequence((1.0, nan)), sequence((2.0, nan))
    elif kind == "distinct_nan_end":
        previous, current = sequence((1.0, float("nan"))), sequence((2.0, nan))
    elif kind == "shared_nan_start":
        previous, current = sequence((nan, 1.0)), sequence((nan, 1.0))
    else:
        previous, current = sequence((-0.0, 0.0)), sequence((0.0, -0.0))
    expected, actual = {"r": previous}, {"r": previous}
    reference(expected, {"r": current})
    merge(actual, {"r": current})
    assert actual["r"] is expected["r"]
    assert tuple(struct.pack("!d", value) for value in actual["r"]) == tuple(
        struct.pack("!d", value) for value in expected["r"])


def test_native_request_sequence_preserves_key_order_bits_and_interval_ownership(merge):
    keys = tuple(f"r{index}" for index in range(73))
    expected = {key: (-0.0, float(index)) for index, key in enumerate(keys[::2])}
    actual = dict(expected)
    for step in range(5):
        selected = keys[step::3][::-1]
        incoming = {key: (float(step) + .125, float(step * 100 + index) + .375)
                    for index, key in enumerate(selected)}
        reference(expected, incoming)
        merge(actual, incoming)
        assert tuple(actual) == tuple(expected)
        for key, interval in expected.items():
            assert actual[key] is interval
            assert tuple(struct.pack("!d", value) for value in actual[key]) == tuple(
                struct.pack("!d", value) for value in interval)


def test_custom_mapping_protocol_and_duplicate_items_are_preserved(merge):
    def run(merger):
        events = []

        class Target(dict):
            def get(self, key, default=None):
                events.append(("get", key))
                return super().get(key, default)

            def __setitem__(self, key, value):
                events.append(("set", key, value))
                return super().__setitem__(key, value)

        class Incoming:
            def items(self):
                events.append(("items",))
                return [("b", (1.0, 2.0)), ("a", (1.0, 4.0)), ("b", (2.0, 3.0))]

        target = Target(a=(0.0, 3.0))
        merger(target, Incoming())
        return events, tuple(target.items())

    assert run(merge) == run(reference)


def test_data_motion_wrapper_routes_to_existing_ledger(monkeypatch):
    target, incoming, calls = {}, {}, []

    def record(actual_target, actual_incoming):
        calls.append((actual_target is target, actual_incoming is incoming))

    monkeypatch.setattr(installed_ledger, "merge_latest_intervals", record)
    assert data_motion._merge_latest_resource_intervals(target, incoming) is None
    assert calls == [(True, True)]
