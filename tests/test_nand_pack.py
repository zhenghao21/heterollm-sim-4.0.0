"""Compiled packing must preserve the same source's public-object behavior."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from heterollm_sim import _nand_pack
from heterollm_sim.memory_transfer import ResourceTimeline


def python_source():
    path = Path(__file__).resolve().parents[1] / "src/heterollm_sim/_nand_pack.py"
    spec = importlib.util.spec_from_file_location("nand_pack_python_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pack_reloads_public_values_with_identical_float64_bits():
    source = python_source()
    names = ("a", "b", "c")
    timeline = ResourceTimeline(lane_available={"a": [-0.], "b": [17], "c": [12.5]},
        ready_ns={"a": -0., "b": "12.125", "c": np.float32(.1)},
        busy_ns={"a": 2**53 + 1, "b": .1, "c": 9.75},
        bytes_moved={"a": 11, "b": 2**62, "c": 0},
        directions={"a": "read", "b": "write"},
        last_intervals={"a": (-0., 3.5), "b": (2**53 + 1, 2**53 + 7)})
    for wave in range(3):
        actual = _nand_pack.pack_single_lane(timeline, names, 16, 4096)
        expected = source.pack_single_lane(timeline, names, 16, 4096)
        assert actual is not None
        for value, reference in zip(actual, expected):
            assert value.dtype == reference.dtype
            assert value.shape == reference.shape
            assert value.tobytes() == reference.tobytes()
        timeline.lane_available["a"][0] += .1
        timeline.busy_ns["b"] += .125
        timeline.bytes_moved["a"] += 13


@pytest.mark.parametrize("single_lane", [False, True])
def test_publication_retains_aliases_custom_mapping_overrides_and_large_integer_counts(single_lane):
    source = python_source()
    class Observed(dict):
        def __setitem__(self, key, value):
            self.events.append((key, value))
            super().__setitem__(key, value)
    outcomes = []
    for module in (source, _nand_pack):
        timeline = ResourceTimeline(lane_available={"a": [0.] if single_lane else [0., 1.], "b": [0.]})
        timeline.busy_ns = Observed()
        timeline.busy_ns.events = []
        references = dict(timeline.lane_available)
        lanes = [17., 20.] if single_lane else [17., 13., 20.]
        module.publish_timeline(timeline, ("a", "b"),
            [[17., .125, 7., 17.], [20., 19.5, 18., 20.]], [2**80, 9], [1, 2],
            lanes, [1, 2] if single_lane else [1, 2, 2],
            [0, 1] if single_lane else [0, 2], [1, 1] if single_lane else [2, 1],
            {"a", "b"}, ("a",), [4096, 0], [None, "read", "write"], single_lane)
        assert all(timeline.lane_available[name] is value for name, value in references.items())
        assert timeline.bytes_moved["a"] == 2**80
        assert timeline.busy_ns.events == [("a", .125), ("b", 19.5)]
        outcomes.append(deepcopy(vars(timeline)))
    assert outcomes[0] == outcomes[1]


def _multi_lane_reference(timeline, names, count, transfer_page_bytes,
                          direction_names, direction_lookup):
    """Original execute_pages loop, independent of the extracted helper."""
    def direction_id(value):
        if value not in direction_lookup:
            direction_lookup[value] = len(direction_names)
            direction_names.append(value)
        return direction_lookup[value]
    starts, lane_counts, lanes, lane_directions = [], [], [], []
    state_values, moved_values, direction_values = [], [], []
    seen_lanes = set()
    for name in names:
        existing = timeline.lane_available.get(name)
        if existing is not None:
            if not existing or id(existing) in seen_lanes:
                return None
            seen_lanes.add(id(existing))
        values = existing if existing is not None else [timeline.ready_ns.get(name, 0.)]
        if any(not math.isfinite(value) or value < 0 for value in values):
            return None
        starts.append(len(lanes)); lane_counts.append(len(values)); lanes.extend(values)
        previous = timeline.directions.get(name)
        histories = timeline.lane_directions.get(name)
        if len(values) == 1 or histories is None or len(histories) != len(values):
            histories = [previous] * len(values)
        lane_directions.extend(direction_id(value) for value in histories)
        state_values.append([timeline.ready_ns.get(name, 0.), timeline.busy_ns.get(name, 0.),
                             *timeline.last_intervals.get(name, (0., 0.))])
        byte_count = timeline.bytes_moved.get(name, 0)
        if not isinstance(byte_count, int) or not 0 <= byte_count <= 2**63 - 1 - count * max(4096, transfer_page_bytes):
            return None
        moved_values.append(byte_count)
        direction_values.append(direction_id(previous))
    return (starts, lane_counts, lanes, lane_directions,
            np.array(state_values, dtype=np.float64), np.array(moved_values, dtype=np.int64),
            np.array(direction_values, dtype=np.int64))


@pytest.mark.parametrize("case", ("normal", "shared_lane", "empty", "negative", "nan",
                                  "oversized_bytes", "malformed_interval"))
def test_multi_lane_pack_matches_original_protocol_order_and_array_bits(case):
    class Observed(dict):
        def get(self, key, default=None):
            self.events.append((self.label, key))
            return super().get(key, default)

    def state():
        timeline = ResourceTimeline(lane_available={"one": [-0.0], "multi": [3., 7.25, 1.],
            "odd": (1., 2.)}, ready_ns={"one": "3.5", "missing": np.float32(1.25)},
            busy_ns={"multi": 2**53 + 1}, bytes_moved={"one": True, "multi": 2**62},
            directions={"one": "read", "multi": "write", "odd": "custom"},
            lane_directions={"multi": ["read", "from-host", None], "odd": ["too-short"]},
            last_intervals={"one": (-0., 3.5), "multi": (2**53 + 1, 2**53 + 7)})
        if case == "shared_lane":
            timeline.lane_available["odd"] = timeline.lane_available["one"]
        elif case == "empty":
            timeline.lane_available["odd"] = []
        elif case in ("negative", "nan"):
            timeline.lane_available["odd"] = [-1. if case == "negative" else float("nan")]
        elif case == "oversized_bytes":
            timeline.bytes_moved["odd"] = 2**63
        elif case == "malformed_interval":
            timeline.last_intervals["odd"] = (1., 2., 3.)
        events = []
        for name in ("lane_available", "directions", "lane_directions", "ready_ns", "busy_ns", "bytes_moved", "last_intervals"):
            mapping = Observed(getattr(timeline, name))
            mapping.events, mapping.label = events, name
            setattr(timeline, name, mapping)
        return timeline, events

    outcomes = []
    for function in (_multi_lane_reference, python_source().pack_multi_lane, _nand_pack.pack_multi_lane):
        timeline, events = state()
        directions = [None, "read", "write"]
        lookup = {name: index for index, name in enumerate(directions)}
        if case == "malformed_interval":
            with pytest.raises(ValueError):
                function(timeline, ("one", "multi", "missing", "odd"), 16, 4096, directions, lookup)
            result = "raised"
        else:
            result = function(timeline, ("one", "multi", "missing", "odd"), 16, 4096, directions, lookup)
            if result is not None:
                result = (result[:4], tuple((value.dtype.str, value.shape, value.tobytes()) for value in result[4:]))
        outcomes.append((result, directions, lookup, events))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("module", [python_source(), _nand_pack])
def test_program_frontiers_keep_interleaved_order_and_page_zero_reset(module):
    addresses = np.array([0, 1, 2, 3, 0, 2], dtype=np.int64) * 4096
    current = {("a", 0): 4, ("b", 0): 3}
    expected = {("a", 0): 2, ("b", 0): 2}
    actual = module.validate_program_frontiers(addresses, 2, 4, (("a",), ("b",)), current)
    assert list(actual.items()) == list(expected.items())
    assert current == {("a", 0): 4, ("b", 0): 3}


@pytest.mark.parametrize("module", [python_source(), _nand_pack])
@pytest.mark.parametrize("span", [17, 2**63 - 1, 2**65])
def test_program_frontiers_validate_int64_limit_without_rounding(module, span):
    addresses = np.array([2**63 - 8192, 2**63 - 4096], dtype=np.int64)
    physical_page = (2**63 - 8192) // 4096 // 2
    block, page = divmod(physical_page, span)
    current = {("a", block): page, ("b", block): page}
    actual = module.validate_program_frontiers(addresses, 2, span, (("a",), ("b",)), current)
    assert list(actual.items()) == [(("a", block), page + 1), (("b", block), page + 1)]


@pytest.mark.parametrize("module", [python_source(), _nand_pack])
def test_program_frontier_error_preserves_lookup_order_and_does_not_mutate(module):
    class Observed(dict):
        def get(self, key, default=None):
            visited.append(key)
            return super().get(key, default)

    visited = []
    current = Observed({("a", 0): 0})
    with pytest.raises(ValueError, match=r"requires page 2, received 3"):
        module.validate_program_frontiers(np.array([0, 4096, 12288], dtype=np.int64),
                                         1, 4, (("a",),), current)
    assert visited == [("a", 0)] * 3
    assert current == {("a", 0): 0}
