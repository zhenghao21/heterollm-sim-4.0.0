"""One-source HBM resource conversion, optionally compiled with Cython.

The caller has established ordinary timeline dictionaries and unique single
lane calendars.  Public calendar lists remain live; no value cache replaces
their reads.  The typed loops preserve each conversion/publication phase.
"""
from __future__ import annotations

import numpy as np

try:
    import cython
except ImportError:
    class _PythonMode:
        compiled = False
    cython = _PythonMode()

def refresh_resource_columns(resources, moved, directions, indices, ids, slots,
                             ready: dict, busy: dict, last: dict,
                             byte_count: dict, direction: dict, physical, workspace=None):
    """Read the public resource bill into selected numeric resource rows."""
    view: cython.double[:, ::1] = resources
    if workspace is None:
        workspace = np.empty((len(ids), 2), dtype=np.float64)
    scratch: cython.double[:, ::1] = workspace
    size: cython.Py_ssize_t = len(ids)
    position: cython.Py_ssize_t
    index: cython.Py_ssize_t
    rid: object
    slot: object
    value: object
    values: list = [byte_count.get(rid, 0) for rid in ids]
    intervals: list = []
    direction_values: list = []
    normal_intervals: cython.bint = True
    limit: object = (1 << 63) - 1 - physical
    for value in values:
        if (type(value) is not int and not isinstance(value, (int, np.integer))) or not 0 <= value <= limit:
            return False
    for position in range(size):
        slot = slots[position]
        scratch[position, 0] = float(slot[0]) if slot is not None else float(ready.get(ids[position], 0.0))
    for position in range(size):
        index = position if indices is None else indices[position]
        view[index, 0] = scratch[position, 0]
    for position in range(size):
        scratch[position, 0] = float(busy.get(ids[position], 0.0))
    for position in range(size):
        index = position if indices is None else indices[position]
        view[index, 1] = scratch[position, 0]
    # Preserve NumPy's existing shape/broadcast behavior for authored interval
    # values outside the normal two-element tuples produced by the timeline.
    for rid in ids:
        value = last.get(rid, (0.0, 0.0))
        intervals.append(value)
        if (type(value) is not tuple and type(value) is not list) or len(value) != 2:
            normal_intervals = False
    if normal_intervals:
        for position in range(size):
            value = intervals[position]
            scratch[position, 0] = float(value[0])
            scratch[position, 1] = float(value[1])
        for position in range(size):
            index = position if indices is None else indices[position]
            view[index, 2] = scratch[position, 0]
            view[index, 3] = scratch[position, 1]
    else:
        resources[slice(None) if indices is None else indices, 2:] = intervals
    moved[slice(None) if indices is None else indices] = values
    values = [direction.get(rid) for rid in ids]
    for value in values:
        direction_values.append(1 if value == "read" else 2 if value == "write" else 3 if value else 0)
    directions[slice(None) if indices is None else indices] = direction_values
    return True


def publish_resource_columns(ids, resources, moved, indices,
                             lane_available: dict, ready: dict, busy: dict,
                             last: dict, byte_count: dict, direction: dict,
                             lane_directions: dict, touched, channels, operation):
    """Publish selected numeric rows in the timeline's original phase order."""
    # The previous publication also captured all four numeric columns before
    # mutating any public lane list.  A custom list setter can observe or
    # mutate other state, so retain that boundary without Python float lists.
    current = resources[indices]
    view: cython.double[:, ::1] = current
    moved_view: cython.longlong[::1] = moved
    selected: cython.Py_ssize_t[::1] = indices
    count: cython.Py_ssize_t = len(indices)
    position: cython.Py_ssize_t
    index: cython.Py_ssize_t
    data_begin: cython.Py_ssize_t = 2 * channels
    keys: list = []
    data_keys: list = []
    rid: object
    slots: object
    for position in range(count):
        index = selected[position]
        rid = ids[index]
        keys.append(rid)
        slots = lane_available.get(rid)
        if slots is None and rid not in lane_available:
            slots = [0.0]
            lane_available[rid] = slots
        slots[0] = float(view[position, 0])
    for position in range(count):
        ready[keys[position]] = float(view[position, 0])
    for position in range(count):
        busy[keys[position]] = float(view[position, 1])
    for position in range(count):
        index = selected[position]
        last[keys[position]] = (float(view[position, 2]), float(view[position, 3]))
    if lane_directions:
        for rid in keys:
            lane_directions.pop(rid, None)
    touched.update(keys)
    for position in range(count):
        index = selected[position]
        if index >= data_begin:
            rid = keys[position]
            data_keys.append(rid)
            byte_count[rid] = int(moved_view[index])
    for rid in data_keys:
        direction[rid] = operation


class HBMResourceAccumulator:
    """One task's resource bill, consumed in physical request order.

    This is an internal summary sink.  Public Core calls do not use it and
    continue to return independent per-request resource dictionaries.
    """

    def __init__(self):
        self.resource_busy_ns = {}
        self.resource_bytes = {}
        self.resource_last_intervals = {}

    def consume_metrics(self, metrics):
        busy: dict = self.resource_busy_ns
        byte_count: dict = self.resource_bytes
        last: dict = self.resource_last_intervals
        rid: object
        value: object
        previous: object
        for rid, value in metrics.get("resource_last_intervals", {}).items():
            previous = last.get(rid)
            if previous is None or (value[1], value[0]) > (previous[1], previous[0]):
                last[rid] = value
        for rid, value in metrics.get("resource_bytes", {}).items():
            byte_count[rid] = byte_count.get(rid, 0) + value
        for rid, value in metrics.get("resource_busy_ns", {}).items():
            busy[rid] = busy.get(rid, 0.0) + value


def can_capture_resource_columns(timeline, ids):
    """Prove numeric deltas also reproduce the public accounting operands.

    In particular a large authored Python int can compare unequal to its
    rounded float64 representation even when the numeric column is unchanged.
    Custom calendars and aliased dictionaries retain public accounting.
    """
    mappings: tuple = (timeline.lane_available, timeline.ready_ns, timeline.busy_ns,
        timeline.last_intervals, timeline.bytes_moved, timeline.directions,
        timeline.lane_directions)
    if len({id(value) for value in mappings}) != len(mappings):
        return False
    busy: dict = timeline.busy_ns
    byte_count: dict = timeline.bytes_moved
    lanes: dict = timeline.lane_available
    rid: object
    value: object
    for rid in ids:
        value = busy.get(rid)
        if value is not None and type(value) is not float:
            return False
        value = byte_count.get(rid)
        if value is not None and type(value) is not int:
            return False
        value = lanes.get(rid)
        if value is not None and type(value) is not list:
            return False
    return True


def consume_resource_columns(accumulator, lookup: dict, resources, moved,
                             before_resources, before_moved, positions,
                             timeline):
    """Fuse exact public deltas with the task bill without temporary maps.

    Numeric checkpoint rows are immutable for this call.  Last intervals
    follow the actual touched set, exactly as ResourceTimeline's per-request
    accounting does. Byte and busy deltas visit only touched resources:
    updating a key already in the bill keeps its place, so only keys new to
    the bill need the public dictionary's order, taken in a second pass.
    """
    current: cython.double[:, ::1] = resources
    previous_rows: cython.double[:, ::1] = before_resources
    moved_view: cython.longlong[::1] = moved
    previous_bytes: cython.longlong[::1] = before_moved
    busy: dict = accumulator.resource_busy_ns
    byte_count: dict = accumulator.resource_bytes
    last: dict = accumulator.resource_last_intervals
    public_busy: dict = timeline.busy_ns
    public_bytes: dict = timeline.bytes_moved
    public_last: dict = timeline.last_intervals
    touched: set = timeline._touched
    rid: object
    interval: object
    prior_interval: object
    byte_delta: object
    index: cython.Py_ssize_t
    before_index: cython.Py_ssize_t
    value: cython.double
    before_value: cython.double
    total: cython.double
    for rid in touched:
        interval = public_last[rid]
        prior_interval = last.get(rid)
        if prior_interval is None or (interval[1], interval[0]) > (prior_interval[1], prior_interval[0]):
            last[rid] = interval
    fresh_bytes: list = []
    fresh_busy: list = []
    pending: set
    for rid in touched:
        if rid in public_bytes:
            index = lookup[rid]
            before_index = index if positions is None else positions[index]
            if moved_view[index] != previous_bytes[before_index]:
                if rid in byte_count:
                    byte_delta = int(moved_view[index]) - int(previous_bytes[before_index])
                    byte_count[rid] = byte_count[rid] + byte_delta
                else:
                    fresh_bytes.append(rid)
        if rid in public_busy:
            index = lookup[rid]
            before_index = index if positions is None else positions[index]
            value = current[index, 1]
            before_value = previous_rows[before_index, 1]
            if value != before_value:
                if rid in busy:
                    # Keep subtraction followed by exactly one addition per
                    # request; never replace this sequence with a batch subtotal.
                    total = busy[rid]
                    busy[rid] = total + (value - before_value)
                else:
                    fresh_busy.append(rid)
    if fresh_bytes:
        pending = set(fresh_bytes)
        for rid in public_bytes:
            if rid in pending:
                index = lookup[rid]
                before_index = index if positions is None else positions[index]
                byte_delta = int(moved_view[index]) - int(previous_bytes[before_index])
                byte_count[rid] = byte_count.get(rid, 0) + byte_delta
    if fresh_busy:
        pending = set(fresh_busy)
        for rid in public_busy:
            if rid in pending:
                index = lookup[rid]
                before_index = index if positions is None else positions[index]
                value = current[index, 1]
                before_value = previous_rows[before_index, 1]
                total = busy.get(rid, 0.0)
                busy[rid] = total + (value - before_value)
