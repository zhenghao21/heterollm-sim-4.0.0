"""Numeric/public resource conversion retains phases, types and aliases."""
from copy import deepcopy
import importlib.util
from itertools import repeat
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/heterollm_sim/_hbm_pack.py"


@pytest.fixture(params=("python", "compiled"))
def pack(request):
    path, name = SOURCE, "hbm_pack_python_test"
    if request.param == "compiled":
        folder = Path(os.environ.get("HETEROLLM_HBM_PACK_DIR", SOURCE.parent))
        binaries = tuple(folder.glob("_hbm_pack*.pyd")) + tuple(folder.glob("_hbm_pack*.so"))
        if not binaries:
            pytest.skip("optional HBM pack module has not been built")
        path, name = binaries[0], "heterollm_sim._hbm_pack"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def refresh_reference(resources, moved, directions, indices, ids, slots,
                      ready, busy, last, byte_count, direction, physical, workspace=None):
    selected = slice(None) if indices is None else indices
    previous_bytes = list(map(byte_count.get, ids, repeat(0)))
    limit = (1 << 63) - 1 - physical
    if any(not isinstance(value, (int, np.integer)) or not 0 <= value <= limit for value in previous_bytes):
        return False
    resources[selected, 0] = np.fromiter((float(slot[0]) if slot is not None else float(ready.get(rid, 0.0))
                                        for rid, slot in zip(ids, slots)), dtype=np.float64, count=len(ids))
    resources[selected, 1] = np.fromiter(map(busy.get, ids, repeat(0.0)), dtype=np.float64, count=len(ids))
    resources[selected, 2:] = list(map(last.get, ids, repeat((0.0, 0.0))))
    moved[selected] = previous_bytes
    values = list(map(direction.get, ids))
    directions[selected] = [1 if value == "read" else 2 if value == "write" else 3 if value else 0 for value in values]
    return True


def publish_reference(ids, resources, moved, indices, lanes, ready_map, busy_map, last_map,
                      byte_map, direction_map, lane_directions, touched, channels, operation):
    keys = list(map(ids.__getitem__, indices.tolist()))
    ready, busy, starts, stops = resources[indices].T.tolist()
    for rid, value in zip(keys, ready):
        lanes.setdefault(rid, [0.0])[0] = value
    ready_map.update(zip(keys, ready))
    busy_map.update(zip(keys, busy))
    last_map.update(zip(keys, zip(starts, stops)))
    if lane_directions:
        for rid in keys:
            lane_directions.pop(rid, None)
    touched.update(keys)
    data_indices = indices[indices >= 2 * channels]
    data_keys = list(map(ids.__getitem__, data_indices.tolist()))
    byte_map.update(zip(data_keys, moved[data_indices].tolist()))
    direction_map.update(dict.fromkeys(data_keys, operation))


def check_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            check_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            check_equal(a, b)
    elif type(left) is float:
        assert left.hex() == right.hex()
    else:
        assert left == right


def refresh_args():
    return (np.zeros((8, 4)), np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int8),
            np.array([1, 3, 6], dtype=np.intp), ("b", "a", "c"), ([1.25], None, [-0.0]),
            {"a": 2.5}, {"b": 7, "a": "4.5", "c": np.float64(0.5)},
            {"b": (1.0, 2.0), "a": [3, 4], "c": (-0.0, 0.0)},
            {"b": True, "a": np.int64(19), "c": 2**62},
            {"b": "read", "a": "write", "c": "mixed"}, 32)


def test_refresh_keeps_bits_shapes_types_and_live_calendar_values(pack):
    expected, actual = deepcopy(refresh_args()), deepcopy(refresh_args())
    assert pack.refresh_resource_columns(*actual) == refresh_reference(*expected)
    check_equal(actual, expected)
    actual[5][0][0] = expected[5][0][0] = 991.25
    assert pack.refresh_resource_columns(*actual) == refresh_reference(*expected)
    check_equal(actual, expected)


@pytest.mark.parametrize("field", ("ready", "busy", "interval", "bytes", "direction"))
def test_refresh_failures_match_completed_phase_prefix(pack, field):
    class BadDirection:
        def __eq__(self, other):
            raise ValueError("bad direction")
    def state():
        values = list(refresh_args())
        if field == "ready":
            values[5][0][0] = "bad"
        elif field == "busy":
            values[7]["a"] = "bad"
        elif field == "interval":
            values[8]["a"] = ("bad", 4)
        elif field == "bytes":
            values[9]["a"] = 2**80
        else:
            values[10]["a"] = BadDirection()
        return values
    expected, actual = state(), state()
    if field == "bytes":
        assert not refresh_reference(*expected)
        assert not pack.refresh_resource_columns(*actual)
    else:
        for function, args in ((refresh_reference, expected), (pack.refresh_resource_columns, actual)):
            with pytest.raises(ValueError):
                function(*args)
    for a, b in zip(actual[:3], expected[:3]):
        check_equal(a, b)


@pytest.mark.parametrize("intervals", ((2.0, 3.0), ((2.0,), (3.0,))))
def test_refresh_preserves_authored_numpy_broadcasting(pack, intervals):
    def state():
        values = list(refresh_args())
        values[3] = np.array([1, 3], dtype=np.intp)
        values[4], values[5] = ("b", "a"), ([1.0], [2.0])
        values[8] = dict(zip(values[4], intervals))
        return values
    expected, actual = state(), state()
    assert pack.refresh_resource_columns(*actual) == refresh_reference(*expected)
    check_equal(actual, expected)


@pytest.mark.parametrize("aliased", (False, True))
def test_publish_preserves_calendar_identity_dict_order_integer_and_float_types(pack, aliased):
    def state():
        times = np.arange(32, dtype=float).reshape(8, 4) / 4
        times[1, 0] = -0.0
        shared = {"old": 0.0}
        return (tuple("abcdefgh"), times, np.arange(8, dtype=np.int64) + 2**60,
                np.array([1, 3, 4, 7], dtype=np.intp), {"b": [123.0], "d": [4.0]},
                shared, shared if aliased else {"old": 0.0}, {}, {}, {}, {"b": "read"}, set(), 2, "write")
    expected, actual = state(), state()
    existing = actual[4]["b"]
    publish_reference(*expected)
    pack.publish_resource_columns(*actual)
    check_equal(actual, expected)
    assert actual[4]["b"] is existing


def test_publish_snapshot_precedes_custom_lane_setter_side_effects(pack):
    def state():
        resources = np.arange(16, dtype=float).reshape(4, 4)
        class EditingList(list):
            def __setitem__(self, key, value):
                resources[1:, :] = 1000.0
                return super().__setitem__(key, value)
        return (("a", "b", "c", "d"), resources, np.zeros(4, dtype=np.int64),
                np.arange(4, dtype=np.intp), {"a": EditingList([0.0])}, {}, {}, {}, {}, {}, {}, set(), 1, "read")
    expected, actual = state(), state()
    publish_reference(*expected)
    pack.publish_resource_columns(*actual)
    # The local subclass classes differ between independently constructed runs.
    for index in (1, 5, 6, 7, 8, 9):
        check_equal(actual[index], expected[index])


def test_publish_existing_none_calendar_keeps_exception_prefix(pack):
    def state():
        return (("a", "b"), np.arange(8, dtype=float).reshape(2, 4), np.zeros(2, dtype=np.int64),
                np.arange(2, dtype=np.intp), {"a": [0.0], "b": None}, {}, {}, {}, {}, {}, {}, set(), 1, "read")
    expected, actual = state(), state()
    for function, args in ((publish_reference, expected), (pack.publish_resource_columns, actual)):
        with pytest.raises(TypeError):
            function(*args)
    check_equal(actual, expected)
