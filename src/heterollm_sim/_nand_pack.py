"""Typed preparation/publication of the unchanged NAND calendar arrays.

The same source runs as Python or Cython. Public dictionaries and lane lists
are read on every call; no numeric state is cached here and no reservation
or resource accounting is omitted.
"""
from __future__ import annotations

import numpy as np
import math

try:
    import cython
except ImportError:
    class _PythonMode:
        compiled = False

        @staticmethod
        def cfunc(function):
            return function

        inline = cfunc

    cython = _PythonMode()


@cython.cfunc
@cython.inline
def _get(mapping: object, key: object, default: object = None) -> object:
    if cython.compiled and type(mapping) is dict:
        return cython.cast(dict, mapping).get(key, default)
    return mapping.get(key, default)


@cython.cfunc
@cython.inline
def _put(mapping: object, key: object, value: object) -> object:
    if cython.compiled and type(mapping) is dict:
        cython.cast(dict, mapping)[key] = value
    else:
        mapping[key] = value


@cython.cfunc
@cython.inline
def _item(mapping: object, key: object) -> object:
    if cython.compiled and type(mapping) is dict:
        return cython.cast(dict, mapping)[key]
    return mapping[key]


@cython.cfunc
@cython.inline
def _double(value: object) -> cython.double:
    if type(value) is float or type(value) is int:
        return value
    # Preserve NumPy's existing coercion for unusual public scalar values,
    # such as NumPy numbers, numeric strings and None; ordinary clocks never
    # pay for this conversion.
    return float(np.float64(value))


def pack_single_lane(timeline, names, count, transfer_page_bytes):
    """Read ordinary single-slot calendars, or decline before mutation."""
    lane_map: object = timeline.lane_available
    direction_map: object = timeline.directions
    ready_map: object = timeline.ready_ns
    busy_map: object = timeline.busy_ns
    moved_map: object = timeline.bytes_moved
    interval_map: object = timeline.last_intervals
    seen: set = set()
    rows: list = []
    previous: list = []
    moved_values: list = []
    name: object
    row: object
    value: object
    direction: object
    interval: object
    index: cython.Py_ssize_t
    size: cython.Py_ssize_t = len(names)
    limit: object = 2**63 - 1 - count * max(4096, transfer_page_bytes)
    for name in names:
        rows.append(_get(lane_map, name))
    for row in rows:
        if (type(row) is not list or len(row) != 1 or type(row[0]) not in (float, int)
                or id(row) in seen):
            return None
        seen.add(id(row))
    for name in names:
        previous.append(_get(direction_map, name))
    for direction in previous:
        if direction is not None and direction != "read" and direction != "write":
            return None
    lanes = np.empty(size, dtype=np.float64)
    lane_view: cython.double[::1] = lanes
    for index in range(size):
        lane_view[index] = rows[index][0]
    if not np.all(np.isfinite(lanes)) or np.any(lanes < 0):
        return None
    for name in names:
        moved_values.append(_get(moved_map, name, 0))
    for value in moved_values:
        if not isinstance(value, int) or not 0 <= value <= limit:
            return None
    directions = np.empty(size, dtype=np.int64)
    direction_view: cython.longlong[::1] = directions
    state = np.empty((size, 4), dtype=np.float64)
    state_view: cython.double[:, ::1] = state
    # Built-in dictionaries have no callbacks. The helper still uses the
    # mapping protocol for custom mappings instead of bypassing overrides.
    for index in range(size):
        direction = previous[index]
        direction_view[index] = 0 if direction is None else 1 if direction == "read" else 2
    for index, name in enumerate(names):
        state_view[index, 0] = _double(_get(ready_map, name, 0.))
    for index, name in enumerate(names):
        state_view[index, 1] = _double(_get(busy_map, name, 0.))
    for index, name in enumerate(names):
        interval = _get(interval_map, name, (0., 0.))
        state_view[index, 2], state_view[index, 3] = _double(interval[0]), _double(interval[1])
    return (np.arange(size, dtype=np.int64), np.ones(size, dtype=np.int64),
            lanes, directions.copy(), state, np.array(moved_values, dtype=np.int64), directions)


@cython.cfunc
@cython.inline
def _direction_id(value: object, lookup: dict, names: list) -> object:
    if value not in lookup:
        lookup[value] = len(names)
        names.append(value)
    return lookup[value]


def pack_multi_lane(timeline, names, count, transfer_page_bytes,
                    direction_names: list, direction_lookup: dict):
    """Pack ragged calendars in the original per-resource observation order.

    One multi-lane fabric credit resource must not move every NAND resource
    back into a Python loop. Lists stay lists until the caller's existing
    numeric conversion boundary; malformed public objects fail in the same
    order as the original loop.
    """
    lane_map: object = timeline.lane_available
    direction_map: object = timeline.directions
    lane_direction_map: object = timeline.lane_directions
    ready_map: object = timeline.ready_ns
    busy_map: object = timeline.busy_ns
    moved_map: object = timeline.bytes_moved
    interval_map: object = timeline.last_intervals
    seen_lanes: set = set()
    starts: list = []
    lane_counts: list = []
    lanes: list = []
    lane_directions: list = []
    state_values: list = []
    moved_values: list = []
    direction_values: list = []
    name: object
    existing: object
    values: object
    value: object
    previous: object
    histories: object
    byte_count: object
    limit: object = 2**63 - 1 - count * max(4096, transfer_page_bytes)
    for name in names:
        existing = _get(lane_map, name)
        if existing is not None:
            if not existing or id(existing) in seen_lanes:
                return None
            seen_lanes.add(id(existing))
        values = existing if existing is not None else [_get(ready_map, name, 0.)]
        for value in values:
            if not math.isfinite(value) or value < 0:
                return None
        starts.append(len(lanes))
        lane_counts.append(len(values))
        lanes.extend(values)
        previous = _get(direction_map, name)
        histories = _get(lane_direction_map, name)
        if len(values) == 1 or histories is None or len(histories) != len(values):
            histories = [previous] * len(values)
        for value in histories:
            lane_directions.append(_direction_id(value, direction_lookup, direction_names))
        state_values.append([_get(ready_map, name, 0.), _get(busy_map, name, 0.),
                             *_get(interval_map, name, (0., 0.))])
        byte_count = _get(moved_map, name, 0)
        if not isinstance(byte_count, int) or not 0 <= byte_count <= limit:
            return None
        moved_values.append(byte_count)
        direction_values.append(_direction_id(previous, direction_lookup, direction_names))
    return (starts, lane_counts, lanes, lane_directions,
            np.array(state_values, dtype=np.float64),
            np.array(moved_values, dtype=np.int64),
            np.array(direction_values, dtype=np.int64))


def publish_timeline(timeline, names, state_values: list, moved_values: list,
                     direction_values: list, lane_values: list, lane_direction_values: list,
                     start_values: list, lane_count_values: list, active_resources,
                     link_names, byte_values: list, direction_names: list, single_lane):
    """Publish each affected resource once, retaining mutable lane aliases."""
    lane_map: object = timeline.lane_available
    direction_map: object = timeline.directions
    lane_direction_map: object = timeline.lane_directions
    ready_map: object = timeline.ready_ns
    busy_map: object = timeline.busy_ns
    moved_map: object = timeline.bytes_moved
    interval_map: object = timeline.last_intervals
    index: cython.Py_ssize_t
    first: cython.Py_ssize_t
    lane_count: cython.Py_ssize_t
    name: object
    row: list
    slots: object
    values: list
    for index, name in enumerate(names):
        if name not in active_resources:
            continue
        first, lane_count = start_values[index], lane_count_values[index]
        if single_lane:
            slots = _item(lane_map, name)
            if cython.compiled and type(slots) is list:
                cython.cast(list, slots)[0] = lane_values[first]
            else:
                slots[0] = lane_values[first]
        else:
            values = lane_values[first:first + lane_count]
            if name in lane_map:
                slots = _item(lane_map, name)
                slots[:] = values
            else:
                _put(lane_map, name, values)
        row = state_values[index]
        _put(ready_map, name, row[0])
        _put(busy_map, name, row[1])
        _put(interval_map, name, (row[2], row[3]))
        if name in link_names or byte_values[index] or name in moved_map:
            _put(moved_map, name, moved_values[index])
        if direction_values[index]:
            _put(direction_map, name, direction_names[direction_values[index]])
        if lane_count > 1:
            _put(lane_direction_map, name,
                 [direction_names[value] for value in lane_direction_values[first:first + lane_count]])
        elif cython.compiled and type(lane_direction_map) is dict:
            cython.cast(dict, lane_direction_map).pop(name, None)
        else:
            lane_direction_map.pop(name, None)
