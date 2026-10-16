"""Resource reducers retain mapping and ordered Python numeric semantics."""
from collections import UserDict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from heterollm_sim import _cost_ledger


@pytest.fixture(params=("imported", "python"))
def ledger(request):
    if request.param == "imported":
        return _cost_ledger
    path = Path(__file__).resolve().parents[1] / "src/heterollm_sim/_cost_ledger.py"
    spec = importlib.util.spec_from_file_location("python_cost_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TracedDict(dict):
    def __init__(self, name, events, *args):
        super().__init__(*args)
        self.name, self.events = name, events

    def get(self, key, default=None):
        self.events.append((self.name, "get", key))
        return super().get(key, default)

    def __getitem__(self, key):
        self.events.append((self.name, "item", key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.events.append((self.name, "put", key))
        return super().__setitem__(key, value)

    def __contains__(self, key):
        self.events.append((self.name, "contains", key))
        return super().__contains__(key)


def test_accumulator_preserves_mapping_protocol_and_eager_owner_default(ledger):
    events = []
    row = TracedDict("row", events, {"owner": "first", "bytes_moved": 2**80, "service_ns": 1.})
    totals = TracedDict("totals", events, {"wire": row})
    busy = TracedDict("busy", events, {"wire": 2.})
    moved = TracedDict("moved", events, {"wire": 2**80})
    owners = TracedDict("owners", events, {"wire": "second"})
    resolved = TracedDict("resolved", events, {"wire": "ignored"})
    raw = TracedDict("raw", events, {"resource_busy_ns": busy,
        "resource_bytes": moved, "resource_owners": resolved})
    task = SimpleNamespace(demands=(SimpleNamespace(resource_id="wire"),))
    ledger._accumulate_physical_resource_totals(totals, task, raw, owners)
    assert events == [
        ("raw", "get", "resource_busy_ns"), ("raw", "get", "resource_bytes"),
        ("raw", "get", "resource_owners"), ("busy", "contains", "wire"),
        ("resolved", "get", "wire"), ("owners", "get", "wire"),
        ("totals", "get", "wire"), ("row", "item", "owner"),
        ("row", "put", "owner"), ("row", "item", "bytes_moved"),
        ("moved", "get", "wire"), ("row", "put", "bytes_moved"),
        ("row", "item", "service_ns"), ("busy", "item", "wire"),
        ("row", "put", "service_ns"),
    ]
    assert row == {"owner": "wire", "bytes_moved": 2**81, "service_ns": 3.}


def test_accumulator_preserves_in_place_numeric_protocol(ledger):
    class Accumulator:
        def __init__(self, value):
            self.value = value
        def __iadd__(self, other):
            self.value += other
            return self
        def __add__(self, other):
            raise AssertionError("in-place protocol was replaced")

    byte_count, service = Accumulator(3), Accumulator(1.)
    totals = UserDict({"wire": UserDict({"owner": "gpu", "bytes_moved": byte_count,
                                        "service_ns": service})})
    ledger._accumulate_physical_resource_totals(totals,
        SimpleNamespace(demands=(SimpleNamespace(resource_id="wire"),)),
        UserDict({"resource_busy_ns": UserDict({"wire": 2.}),
                  "resource_bytes": UserDict({"wire": 4})}), UserDict({"wire": "gpu"}))
    assert totals["wire"]["bytes_moved"] is byte_count and byte_count.value == 7
    assert totals["wire"]["service_ns"] is service and service.value == 3.


def test_mapping_get_argument_count_is_preserved(ledger):
    calls = []
    class Mapping(UserDict):
        def get(self, *args):
            calls.append(args)
            return super().get(*args)

    totals = Mapping()
    raw = Mapping({"resource_busy_ns": {"wire": 2.}, "resource_bytes": {"wire": 3}})
    ledger._accumulate_physical_resource_totals(totals,
        SimpleNamespace(demands=(SimpleNamespace(resource_id="wire"),)), raw, {})
    assert calls == [("resource_busy_ns", {}), ("resource_bytes",),
                     ("resource_owners", {}), ("wire",)]


def test_direction_uses_float_conversion_and_original_addition_order(ledger):
    class ExplicitFloat(float):
        def __float__(self):
            return 0.1

    def task(duration, *, moved=0, write=0, overrides=None):
        return SimpleNamespace(name="ordinary", metadata=UserDict({
            "memory_direction": "read", "modeled_memory_write_bytes": write,
            "resource_directions": overrides}), demands=(SimpleNamespace(
                resource_id="wire", service_ns=duration, bytes_moved=moved),))

    assert ledger._resource_busy_by_direction((task(ExplicitFloat(99.)),)) == {"read": {"wire": 0.1}}
    assert ledger._resource_busy_by_direction(tuple(task(v) for v in (1e16, 1., 1.))) == {
        "read": {"wire": 1e16}}
    assert ledger._resource_busy_by_direction((task(float("nan")), task(-0.), task(-1.))) == {}
    assert ledger._resource_busy_by_direction((task(7., moved=2**100, write=2**98),)) == {
        "read": {"wire": 5.25}, "write": {"wire": 1.75}}
    assert ledger._resource_busy_by_direction((task(7., overrides=UserDict({"wire": "WRITE"})),)) == {
        "write": {"wire": 7.}}


def test_accumulator_validates_bytes_only_for_matching_demands(ledger):
    task = SimpleNamespace(demands=(SimpleNamespace(resource_id="wire"),))
    totals = {}
    ledger._accumulate_physical_resource_totals(totals, task,
        UserDict({"resource_busy_ns": {"other": 1.}, "resource_bytes": None}), {})
    assert totals == {}
    with pytest.raises(ValueError, match="resource_bytes"):
        ledger._accumulate_physical_resource_totals(totals, task,
            UserDict({"resource_busy_ns": {"wire": 1.}, "resource_bytes": None}), {})
