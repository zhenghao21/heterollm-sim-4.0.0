"""Ordered accounting equivalence, also run against an explicitly built module."""
from copy import deepcopy
import builtins
import importlib.util
import math
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand
from heterollm_sim.memory_types import Operation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/heterollm_sim/_resource_ledger.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=("python", "compiled"))
def ledger(request):
    if request.param == "python":
        return _load(SOURCE, "resource_ledger_python_test")
    folder = Path(os.environ.get("HETEROLLM_LEDGER_EXTENSION_DIR", SOURCE.parent))
    binaries = tuple(folder.glob("_resource_ledger*.pyd")) + tuple(folder.glob("_resource_ledger*.so"))
    if not binaries:
        pytest.skip("optional resource ledger has not been built")
    return _load(binaries[0], "heterollm_sim._resource_ledger")


def _same(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    elif type(left) is float:
        assert left.hex() == right.hex()
    else:
        assert left == right


@pytest.mark.parametrize("proxy", [False, True])
def test_owner_frontiers_preserve_alias_chain_order_and_float_types(ledger, proxy):
    from types import MappingProxyType
    owners = {"late": "source", "chain": "late", "absent": "missing",
              "zero": "negative_zero", "source": "chain"}
    if proxy:
        owners = MappingProxyType(owners)
    before = {"source": .1 + .2, "late": 7, "negative_zero": -0.0}
    expected = dict(before)
    for resource, owner in owners.items():
        if owner in expected:
            expected[resource] = expected[owner]
    actual = dict(before)
    ledger.publish_owner_frontiers(owners, actual)
    _same(expected, actual)


@pytest.mark.parametrize("count,touched_count", [(8, 8), (64, 16), (64, 2)])
def test_timeline_delta_preserves_counter_types_and_resource_order(ledger, count, touched_count):
    from heterollm_sim.memory_transfer import ResourceTimeline
    timeline = ResourceTimeline()
    timeline.busy_ns = {f"r{i}": float(i) for i in range(count)}
    timeline.bytes_moved = {f"r{i}": 2 ** 180 + i for i in range(count)}
    before = timeline.metrics_snapshot()
    for i in range(touched_count):
        rid = f"r{count - i - 1}"
        timeline.busy_ns[rid] += 0.125
        timeline.bytes_moved[rid] += 2 ** 70
        timeline.last_intervals[rid] = (1.0, 2.0)
        timeline._touched.add(rid)
    expected = {
        "resource_busy_ns": {rid: value - before["busy_ns"].get(rid, 0.0)
            for rid, value in timeline.busy_ns.items()
            if value != before["busy_ns"].get(rid, 0.0)},
        "resource_bytes": {rid: value - before["bytes_moved"].get(rid, 0)
            for rid, value in timeline.bytes_moved.items()
            if value != before["bytes_moved"].get(rid, 0)},
        "resource_last_intervals": {rid: timeline.last_intervals[rid] for rid in timeline._touched},
        "resource_intervals": {}, "resource_interval_payloads": {},
        "intervals_truncated": False,
    }
    _same(ledger.timeline_metrics_delta(timeline, before), expected)
    _same(ledger.core_timeline_metrics_delta(timeline, before), expected)


def test_public_timeline_delta_observes_custom_mapping_lookups(ledger):
    from heterollm_sim.memory_transfer import ResourceTimeline
    calls = []

    class Before(dict):
        def get(self, key, default=None):
            calls.append((key, default))
            return super().get(key, default)

    class Counter(float):
        def __sub__(self, other):
            return ("difference", float(self), other)

    timeline = ResourceTimeline()
    timeline.busy_ns = {"a": Counter(2), "b": 0.0}
    timeline.bytes_moved = {"c": True}
    result = ledger.timeline_metrics_delta(timeline, {
        "busy_ns": Before(), "bytes_moved": Before()})
    assert result["resource_busy_ns"] == {"a": ("difference", 2.0, 0.0)}
    assert type(result["resource_bytes"]["c"]) is int
    assert calls == [("a", 0.0), ("a", 0.0), ("b", 0.0), ("c", 0), ("c", 0)]


def _merge(busy, byte_count, last, incoming_busy, incoming_bytes, incoming_last, owners, owner):
    for resource_id, interval in incoming_last.items():
        previous = last.get(resource_id)
        if previous is None or (interval[1], interval[0]) > (previous[1], previous[0]):
            last[resource_id] = interval
    for resource_id, value in incoming_bytes.items():
        byte_count[resource_id] = byte_count.get(resource_id, 0) + value
    for resource_id, value in incoming_busy.items():
        busy[resource_id] = busy.get(resource_id, 0.0) + value
        if owners is not None:
            owners[resource_id] = owner


def test_merge_preserves_order_float_steps_large_ints_and_types(ledger):
    state = ({"old": -0.0, "a": 1e16}, {"a": 2**200},
             {"a": (3.0, 5.0), "b": [1, 4]}, {"old": "earlier"})
    expected, actual = deepcopy(state), deepcopy(state)
    inputs = (
        ({"c": 7, "a": 1.0, "old": -0.0}, {"c": True, "a": 2**400},
         {"b": [2, 4], "a": (2.0, 5.0), "c": (0.0, 0.0)}),
        ({"a": -1e16, "b": True}, {"b": 3, "a": 1},
         {"a": (4.0, 5.0), "b": (0.0, 3.0)}),
    )
    for busy, byte_count, last in inputs:
        _merge(*expected[:3], busy, byte_count, last, expected[3], "gpu0")
        ledger.merge_resource_bill(*actual[:3], busy, byte_count, last, actual[3], "gpu0")
        _same(actual, expected)
    assert actual[0]["a"] == 0.0


def test_merge_preserves_shared_nan_tuple_comparison_and_numeric_subclasses(ledger):
    class AddedFloat(float):
        def __radd__(self, previous):
            return ("custom", previous, float(self))

    nan = float("nan")
    for previous, incoming in (((1.0, nan), (2.0, nan)),
                               ((1.0, float("nan")), (2.0, nan))):
        expected = ({}, {}, {"a": previous}, {})
        actual = ({}, {}, {"a": previous}, {})
        args = ({"a": AddedFloat(3)}, {"a": 2**200}, {"a": incoming})
        _merge(*expected[:3], *args, expected[3], "gpu0")
        ledger.merge_resource_bill(*actual[:3], *args, actual[3], "gpu0")
        _same(actual, expected)


def test_merge_aliases_and_rejected_mapping_leave_same_prefix(ledger):
    def state():
        shared = {"a": 2}
        return shared, shared, {}, {}
    expected, actual = state(), state()
    args = ({"a": 0.5}, {"a": 2**100}, {"a": (1.0, 2.0)})
    _merge(*expected[:3], *args, expected[3], "gpu0")
    ledger.merge_resource_bill(*actual[:3], *args, actual[3], "gpu0")
    _same(actual, expected)
    class CustomDict(dict):
        pass
    unchanged = deepcopy(actual)
    with pytest.raises(TypeError):
        ledger.merge_resource_bill(*actual[:3], CustomDict(), {}, {}, actual[3], "gpu0")
    _same(actual, unchanged)


def _publish(task_id, ids, available, before, times, resource_last, lane_last, ready, predecessors):
    completions = []
    for resource_id in ids:
        lanes = available.get(resource_id, ())
        before_lanes = before.get(resource_id, ())
        pair = times.get(resource_id)
        previous = resource_last.get(resource_id)
        interval = None
        if pair is not None:
            start, end = pair
            interval = {"task_id": task_id, "resource_id": resource_id, "start_ns": start, "end_ns": end}
            if len(lanes) == 1 and previous is not None and previous["end_ns"] == start:
                predecessors.setdefault(resource_id, dict(previous))
            if previous is None or end >= previous["end_ns"]:
                resource_last[resource_id] = interval
        for lane, end in enumerate(lanes):
            prior = before_lanes[lane] if lane < len(before_lanes) else 0.0
            if end <= prior:
                continue
            completions.append((resource_id, lane, end))
            if interval is not None and interval["end_ns"] == end and interval["start_ns"] >= prior:
                last = {**interval, "lane": lane}
            else:
                last = {"task_id": task_id, "resource_id": resource_id, "end_ns": end, "lane": lane}
            lane_last[(resource_id, lane)] = last
        if lanes:
            ready[resource_id] = lanes[0] if len(lanes) == 1 else min(lanes)
    return tuple(completions)


def _publication_state():
    return ("task", ("multi", "zero", "single", "empty", "missing"),
            {"multi": [3.0, 8.0, 9.0], "zero": [5.0], "single": [8.0], "empty": []},
            {"multi": [3.0, 6.0], "zero": [5.0], "single": [4.0]},
            {"multi": (7.0, 8.0), "zero": (5.0, 5.0), "single": (4.0, 8.0), "empty": (0, 0)},
            {"single": {"task_id": "previous", "resource_id": "single", "end_ns": 4.0}},
            {}, {"old": 1.0}, {})


def test_publication_multilane_zero_duration_missing_and_public_order(ledger):
    expected, actual = deepcopy(_publication_state()), deepcopy(_publication_state())
    _same(ledger.publish_physical_summary(*actual), _publish(*expected))
    _same(actual, expected)
    assert actual[-1]["single"] is not actual[-4]["single"]


def test_publication_preserves_mapping_overrides(ledger):
    class ObservedDict(dict):
        def get(self, key, default=None):
            self.calls.append(("get", key))
            return super().get(key, default)
        def __setitem__(self, key, value):
            self.calls.append(("set", key))
            return super().__setitem__(key, value)
        def setdefault(self, key, value):
            self.calls.append(("setdefault", key))
            return super().setdefault(key, value)
    def state():
        args = list(_publication_state())
        for i in range(2, len(args)):
            args[i] = ObservedDict(args[i])
            args[i].calls = []
        return args
    expected, actual = state(), state()
    _same(ledger.publish_physical_summary(*actual), _publish(*expected))
    _same(actual, expected)
    for a, b in zip(actual[2:], expected[2:]):
        assert a.calls == b.calls


def test_publication_exception_preserves_successful_prefix(ledger):
    expected, actual = list(_publication_state()), list(_publication_state())
    expected[4] = {**expected[4], "single": (1,)}
    actual[4] = {**actual[4], "single": (1,)}
    for function, args in ((_publish, expected), (ledger.publish_physical_summary, actual)):
        with pytest.raises(ValueError):
            function(*args)
    _same(actual, expected)


def _account(demands, owner, waits, busy, queue, count, capacities, lanes):
    for demand in demands:
        rid = demand.resource_id
        busy[rid] = busy.get(rid, 0.0) + demand.service_ns
        owner_wait = waits.pop(owner, 0.0)
        queue[rid] = queue.get(rid, 0.0) + owner_wait
        count[rid] = count.get(rid, 0) + 1
        capacities.setdefault(rid, 1)
        lanes[rid] = 0


@pytest.mark.parametrize("empty", (False, True))
def test_single_owner_keeps_wait_once_order_and_integer_counts(ledger, empty):
    demands = () if empty else (ResourceDemand("z", 1, 2**200), ResourceDemand("a", 0.5), ResourceDemand("z", 2))
    state = ({"gpu": 5.5, "other": 12.0}, {"z": 1e16}, {"z": -0.0},
             {"z": 2**400}, {"z": 7}, {})
    expected, actual = deepcopy(state), deepcopy(state)
    _account(demands, "gpu", *expected)
    ledger.account_single_owner_demands(demands, "gpu", *actual)
    _same(actual, expected)
    busy = {}
    assert ledger.accumulate_demands(busy, demands) == (2**200 if demands else 0)


@pytest.mark.parametrize("field,value", (("service_ns", True), ("bytes_moved", True), ("bytes_moved", 1.0)))
def test_demand_validation_remains_at_contract(field, value):
    with pytest.raises(ValueError):
        ResourceDemand(**{"resource_id": "r", "service_ns": 1.0, field: value})


def test_interpreted_module_does_not_require_cython(monkeypatch):
    original = builtins.__import__
    def import_without_cython(name, *args, **kwargs):
        if name == "cython":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", import_without_cython)
    functions = runpy.run_path(str(SOURCE))
    busy = {}
    assert functions["accumulate_demands"](busy, (ResourceDemand("r", 3.0, 2**100),)) == 2**100
    assert busy == {"r": 3.0}


def test_reducer_first_result_keeps_arithmetic_types_and_custom_items(ledger):
    class AddedInt(int):
        def __radd__(self, previous):
            return previous + int(self) + 100

    class AddedFloat(float):
        def __radd__(self, previous):
            return previous + float(self) + 100.0

    class ItemsDict(dict):
        def __init__(self, rows):
            super().__init__(ignored="must not copy backing storage")
            self.rows = rows
            self.calls = 0

        def items(self):
            self.calls += 1
            return iter(self.rows)

    busy = ItemsDict((("bool", True), ("custom", AddedFloat(2)), ("zero", -0.0)))
    byte_count = ItemsDict((("bool", True), ("custom", AddedInt(7)), ("huge", 2**200)))
    last = ItemsDict((("r", (4.0, 9.0)), ("r", (2.0, 8.0)), ("s", [1.0, 3.0])))
    expected = ({}, {}, {}, {})
    _merge(*expected[:3], busy, byte_count, last, expected[3], "gpu")
    item = SimpleNamespace(logical_bytes=64, transfer_bytes=128, completion_ns=12.0,
        counters={"resource_busy_ns": busy, "resource_bytes": byte_count,
                  "resource_last_intervals": last, "physical_operation_count": 3,
                  "physical_read_bytes": 128, "row_hits": 3, "peak": 5.0})
    rows = ledger.reduce_physical_results(
        [(Operation.READ, "gpu", SimpleNamespace(kind="dram"), item)], 2,
        ("row_hits", "row_misses"), frozenset(("row_hits", "row_misses")),
        frozenset(("peak",)), Operation.READ)
    row = rows["gpu"]
    _same((row["resource_busy_ns"], row["resource_bytes"], row["resource_last_intervals"],
           row["resource_owners"]), expected)
    assert (busy.calls, byte_count.calls, last.calls) == (2, 2, 2)
    assert type(row["resource_bytes"]["bool"]) is int
    assert row["resource_bytes"]["custom"] == 107
    assert row["operation_count"] == 3
    assert row["row_hits"] == 3 and row["row_misses"] is None
    assert row["peak"] == 5.0 and row["service_ns"] == 10.0
    assert (row["logical_read_bytes"], row["physical_read_bytes"], row["operation"]) == (64, 128, "read")


class _ObservedLedgerDict(dict):
    def __init__(self, values=()):
        super().__init__(values)
        self.calls = []

    def get(self, key, default=None):
        self.calls.append(("get", key))
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.calls.append(("set", key))
        return super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        self.calls.append(("setdefault", key))
        return super().setdefault(key, default)


class _DemandKernel:
    def __init__(self):
        self.resource_last_interval = {}
        self.resource_available = {}
        self.resource_busy_ns = _ObservedLedgerDict({"wire": 1e16})
        self.resource_queue_wait_ns = _ObservedLedgerDict()
        self.resource_task_count = _ObservedLedgerDict({"wire": 2**200})
        self.resource_capacities = _ObservedLedgerDict({"ordinary": 2})
        self.resource_owners = {"ordinary": "engine", "alias": "engine"}
        self._resource_lane_last_interval = {
            ("engine", 0): {"task_id": "previous", "resource_id": "ordinary", "end_ns": 10.0}}
        self.lanes = {"engine": [10.0, 20.0]}
        self.calls = []

    def _physical_owner(self, resource_id):
        self.calls.append(("owner", resource_id))
        return self.resource_owners.get(resource_id, resource_id)

    def _select_lane(self, resource_id):
        self.calls.append(("select", resource_id))
        if resource_id == "fail":
            raise RuntimeError("injected lane failure")
        lanes = self.lanes[self.resource_owners[resource_id]]
        index = min(range(len(lanes)), key=lanes.__getitem__)
        return lanes, index, lanes[index]


@pytest.mark.parametrize("fail", (False, True))
def test_task_accounting_mixed_shared_wire_compensated_sum_and_exception_prefix(ledger, fail):
    task = SimpleNamespace(task_id="task", metadata={
        "physical_demands_resource_ids": ("wire",), "physical_completion_ns": 12.0,
        # Intentionally unsorted insertion: the original loop sums sorted owners.
        "physical_execution_by_owner": {
            owner: {"resource_owners": {"wire": owner}, "queue_wait_ns": wait}
            for owner, wait in (("c", 1.0), ("a", 1e16), ("b", 1.0))}})
    demands = (ResourceDemand("ordinary", 3.0), ResourceDemand("wire", 1.0),
               ResourceDemand("wire", 1.0))
    if fail:
        demands += (ResourceDemand("fail", 1.0), ResourceDemand("after", 1.0))
    source = _load(SOURCE, "resource_ledger_account_reference")
    expected, actual = _DemandKernel(), _DemandKernel()
    results = []
    actual_lanes = actual.lanes["engine"]
    for function, kernel in ((source.account_task_demands, expected), (ledger.account_task_demands, actual)):
        if fail:
            with pytest.raises(RuntimeError, match="injected lane failure"):
                function(kernel, task, demands, True, 10.0, 7.0)
        else:
            results.append(function(kernel, task, demands, True, 10.0, 7.0))
    _same(actual.__dict__, expected.__dict__)
    for field in ("resource_busy_ns", "resource_queue_wait_ns", "resource_task_count", "resource_capacities"):
        assert getattr(actual, field).calls == getattr(expected, field).calls
    assert actual.lanes["engine"] is actual_lanes and actual_lanes == [13.0, 20.0]
    assert actual.resource_available == {"ordinary": 13.0, "engine": 13.0, "alias": 13.0}
    assert actual.resource_queue_wait_ns == {"ordinary": 3.0, "wire": sum((1e16, 1.0, 1.0))}
    assert actual.resource_queue_wait_ns["wire"] == 10000000000000002.0
    assert actual.resource_busy_ns == {"wire": 1e16, "ordinary": 3.0}
    assert actual.resource_task_count == {"wire": 2**200 + 2, "ordinary": 1}
    assert "fail" not in actual.resource_last_interval and "after" not in actual.resource_busy_ns
    if not fail:
        _same(results[0], results[1])
        assert results[1][0] == 13.0
        assert results[1][2]["ordinary"]["task_id"] == "previous"


def test_physical_demands_keeps_conversions_nan_filter_and_cache_identity(ledger):
    class FloatValues:
        def __init__(self):
            self.calls = 0

        def __float__(self):
            self.calls += 1
            return 1.0 if self.calls % 2 else 2.5

    class IntValue:
        def __init__(self):
            self.calls = 0

        def __int__(self):
            self.calls += 1
            return 2**200

    changing, byte_count = FloatValues(), IntValue()
    busy = {"z": True, "nan": float("nan"), "a": changing, "negative": -1, "zero": -0.0}
    cache = {}
    first = ledger.physical_demands(busy, {"a": byte_count, "z": True}, cache, ResourceDemand)
    second = ledger.physical_demands(busy, {"a": byte_count, "z": True}, cache, ResourceDemand)
    expected = (ResourceDemand("a", 2.5, 2**200), ResourceDemand("z", 1.0, 1))
    _same(first, expected)
    _same(second, expected)
    assert (changing.calls, byte_count.calls) == (4, 2)
    assert all(left is right for left, right in zip(first, second))
    # A second conversion may itself become invalid; filtering only once must
    # still let the normal ResourceDemand contract reject it.
    class BecomesNaN:
        calls = 0

        def __float__(self):
            self.calls += 1
            return 1.0 if self.calls == 1 else float("nan")

    before = dict(cache)
    with pytest.raises(ValueError):
        ledger.physical_demands({"bad": BecomesNaN()}, {}, cache, ResourceDemand)
    _same(cache, before)


def test_native_physical_demand_keeps_complete_dataclass_fields_and_order(ledger):
    actual = ledger.physical_demands({"wire": .1 + .2}, {"wire": 2**180}, {}, ResourceDemand)[0]
    expected = ResourceDemand("wire", .1 + .2, 2**180)
    assert type(actual) is type(expected)
    _same(vars(actual), vars(expected))
    assert actual == expected and hash(actual) == hash(expected)


@pytest.mark.parametrize("hook", ("init", "post_init", "setattr", "getattribute"))
def test_native_demand_callbacks_still_observe_regular_constructor(ledger, monkeypatch, hook):
    events = []
    if hook == "init":
        original = ResourceDemand.__init__
        def init(self, *args, **kwargs):
            events.append(("init", args, kwargs))
            original(self, *args, **kwargs)
        monkeypatch.setattr(ResourceDemand, "__init__", init)
    elif hook == "post_init":
        original = ResourceDemand.__post_init__
        def post_init(self):
            events.append("post_init")
            original(self)
        monkeypatch.setattr(ResourceDemand, "__post_init__", post_init)
    elif hook == "setattr":
        def set_attribute(self, name, value):
            events.append(("setattr", name, value))
            object.__setattr__(self, name, value)
        monkeypatch.setattr(ResourceDemand, "__setattr__", set_attribute)
    else:
        original = ResourceDemand.__getattribute__
        def get_attribute(self, name):
            if name == "__post_init__":
                events.append("getattribute")
            return original(self, name)
        monkeypatch.setattr(ResourceDemand, "__getattribute__", get_attribute)
    result = ledger.physical_demands({"wire": 3.}, {"wire": 11}, {}, ResourceDemand)[0]
    before = list(events)
    expected = ResourceDemand("wire", 3., bytes_moved=11)
    # A frozen dataclass initializer uses object.__setattr__, so a replaced
    # class __setattr__ does not receive calls on either ordinary construction.
    assert before == events[len(before):]
    assert result == expected


@pytest.mark.parametrize("busy,moved", (({"": 1.}, {}), ({"wire": float("inf")}, {}),
                                        ({"wire": 1.}, {"wire": -1})))
def test_native_demand_invalid_fields_keep_original_error(ledger, busy, moved):
    key, value = next(iter(busy.items()))
    with pytest.raises(ValueError) as reference:
        ResourceDemand(str(key), float(value), bytes_moved=int(moved.get(key, 0)))
    with pytest.raises(ValueError, match=str(reference.value)):
        ledger.physical_demands(busy, moved, {}, ResourceDemand)


def test_native_demand_rechecks_callbacks_after_authored_numeric_conversion(ledger, monkeypatch):
    events = []
    original = ResourceDemand.__post_init__
    def post_init(self):
        events.append("post_init")
        original(self)
    class Duration:
        def __float__(self):
            monkeypatch.setattr(ResourceDemand, "__post_init__", post_init)
            return 3.
    result = ledger.physical_demands({"wire": Duration()}, {"wire": 5}, {}, ResourceDemand)
    assert events == ["post_init"]
    assert result[0].service_ns == 3.


def test_native_demand_keeps_authored_finite_validator_calls(ledger, monkeypatch):
    import math
    original = math.isfinite
    events = []
    def finite(value):
        events.append(value)
        return original(value)
    monkeypatch.setattr(math, "isfinite", finite)
    result = ledger.physical_demands({"wire": 3.}, {"wire": 5}, {}, ResourceDemand)
    assert events == [3., 0.]
    assert result[0].bytes_moved == 5
