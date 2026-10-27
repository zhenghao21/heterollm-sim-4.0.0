"""One-source HBM resource conversion, optionally compiled with Cython.

The caller has established ordinary timeline dictionaries and unique single
lane calendars.  Public calendar lists remain live; no value cache replaces
their reads.  The typed loops preserve each conversion/publication phase.
"""
from __future__ import annotations

import numpy as np
from heapq import heapify, heappop, heappush

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


class HBMAdmission:
    """Batch-local device credits with read-only O(log n) eligibility.

    Keep insertion order separately from completion order. Only successful
    requests prune credits, and publishing restores the exact public list.
    """

    def __init__(self, core):
        self.core = core
        self.active = dict(enumerate(core._inflight))
        self.heap = [(end, index) for index, end in self.active.items()]
        heapify(self.heap)
        self.next_index = len(self.active)
        self.expired = []
        self.published = False

    def admit(self, request):
        ready = max(request.arrival_ns, self.core._acceptance_ns, 0.0)
        self.expired = []
        while self.heap and self.heap[0][0] <= ready:
            self.expired.append(heappop(self.heap)[1])
        if len(self.heap) >= self.core.config.max_outstanding_requests:
            ready = max(ready, self.heap[0][0])
        return ready

    def record(self, completion, ready):
        for index in self.expired:
            del self.active[index]
        self.expired = []
        while self.heap and self.heap[0][0] <= ready:
            _end, index = heappop(self.heap)
            del self.active[index]
        self.active[self.next_index] = completion
        heappush(self.heap, (completion, self.next_index))
        self.next_index += 1

    def publish(self):
        if not self.published:
            self.core._inflight = list(self.active.values())
            self.published = True


class HBMResourceBatch:
    """Retain numeric resources inside one closed, owned submission batch.

    Public dictionaries are published once. The task bill still subtracts
    and adds once per request, with the original dictionary insertion order.
    """

    def __init__(self, packed, accumulator):
        self.packed = packed
        self.accumulator = accumulator
        lookup = packed._resource_lookup
        timeline = packed.core.timeline
        self.busy_order = {rid: None for rid in timeline.busy_ns if rid in lookup}
        self.byte_order = {rid: None for rid in timeline.bytes_moved if rid in lookup}
        self.pending = {}
        size = len(packed.ids)
        self.seen = np.zeros(size, dtype=np.uint8)
        self.busy_totals = np.zeros(size, dtype=np.float64)
        self.last_values = np.zeros((size, 2), dtype=np.float64)
        self.has_busy = np.zeros(size, dtype=np.uint8)
        self.has_bytes = np.zeros(size, dtype=np.uint8)
        self.has_last = np.zeros(size, dtype=np.uint8)
        self.changed_busy = np.zeros(size, dtype=np.uint8)
        self.changed_bytes = np.zeros(size, dtype=np.uint8)
        self.changed_last = np.zeros(size, dtype=np.uint8)
        self.before_moved = packed.arrays[9].copy()
        self.bill_busy_order = {rid: lookup[rid] for rid in accumulator.resource_busy_ns if rid in lookup}
        self.bill_byte_order = {rid: lookup[rid] for rid in accumulator.resource_bytes if rid in lookup}
        self.bill_last_order = {rid: lookup[rid] for rid in accumulator.resource_last_intervals if rid in lookup}
        for rid, index in self.bill_busy_order.items():
            self.busy_totals[index] = accumulator.resource_busy_ns[rid]
            self.has_busy[index] = 1
        for index in self.bill_byte_order.values():
            self.has_bytes[index] = 1
        for rid, index in self.bill_last_order.items():
            self.last_values[index] = accumulator.resource_last_intervals[rid]
            self.has_last[index] = 1

    def consume(self, before_resources, before_moved, positions):
        packed = self.packed
        timeline = packed.core.timeline
        indices = np.flatnonzero(packed.arrays[11])
        selected: cython.Py_ssize_t[::1] = indices
        pending: dict = self.pending
        busy_order: dict = self.busy_order
        byte_order: dict = self.byte_order
        lookup: dict = packed._resource_lookup
        current: cython.double[:, ::1] = packed.arrays[8]
        previous: cython.double[:, ::1] = before_resources
        moved: cython.longlong[::1] = packed.arrays[9]
        previous_moved: cython.longlong[::1] = before_moved
        totals: cython.double[::1] = self.busy_totals
        last_values: cython.double[:, ::1] = self.last_values
        seen: cython.uchar[::1] = self.seen
        has_busy: cython.uchar[::1] = self.has_busy
        has_bytes: cython.uchar[::1] = self.has_bytes
        has_last: cython.uchar[::1] = self.has_last
        changed_busy: cython.uchar[::1] = self.changed_busy
        changed_bytes: cython.uchar[::1] = self.changed_bytes
        changed_last: cython.uchar[::1] = self.changed_last
        fresh_busy: set = set()
        fresh_bytes: set = set()
        fresh_last: set = set()
        rid: object
        index: cython.Py_ssize_t
        before_index: cython.Py_ssize_t
        position: cython.Py_ssize_t
        data_begin: cython.Py_ssize_t = 2 * len(packed.channel_keys)
        value: cython.double
        before_value: cython.double
        total: cython.double
        start: cython.double
        end: cython.double
        for position in range(len(indices)):
            index = selected[position]
            rid = packed.ids[index]
            if not seen[index]:
                pending[rid] = index
                busy_order[rid] = None
                if index >= data_begin:
                    byte_order[rid] = None
                seen[index] = 1
            start = float(current[index, 2])
            end = float(current[index, 3])
            if (not has_last[index] or end > last_values[index, 1]
                    or (end == last_values[index, 1] and start > last_values[index, 0])):
                if not has_last[index]:
                    fresh_last.add(rid)
                    has_last[index] = 1
                last_values[index, 0] = start
                last_values[index, 1] = end
                changed_last[index] = 1
            before_index = index if positions is None else positions[index]
            if index >= data_begin and moved[index] != previous_moved[before_index]:
                if not has_bytes[index]:
                    fresh_bytes.add(rid)
                    has_bytes[index] = 1
                changed_bytes[index] = 1
            value = float(current[index, 1])
            before_value = float(previous[before_index, 1])
            if value != before_value:
                total = float(totals[index])
                # Preserve every subtraction and addition in request order.
                totals[index] = total + (value - before_value)
                changed_busy[index] = 1
                if not has_busy[index]:
                    fresh_busy.add(rid)
                    has_busy[index] = 1
        if fresh_last:
            # Public interval keys follow the complete request's touched set,
            # including resources already present in the task bill.
            for rid in set(packed._id_objects[indices].tolist()):
                if rid in fresh_last:
                    self.bill_last_order[rid] = lookup[rid]
        if fresh_bytes:
            for rid in byte_order:
                if rid in fresh_bytes:
                    self.bill_byte_order[rid] = lookup[rid]
        if fresh_busy:
            for rid in busy_order:
                if rid in fresh_busy:
                    self.bill_busy_order[rid] = lookup[rid]
        timeline._intervals_truncated = True

    def publish(self):
        if not self.pending:
            return
        packed = self.packed
        timeline = packed.core.timeline
        # Publication order is first touch across requests, rather than a
        # sorted union; new public keys retain their original insertion order.
        indices = np.fromiter(self.pending.values(), dtype=np.intp)
        timeline._remember_resources(tuple(self.pending))
        publish_resource_columns(packed.ids, packed.arrays[8], packed.arrays[9], indices,
            timeline.lane_available, timeline.ready_ns, timeline.busy_ns,
            timeline.last_intervals, timeline.bytes_moved, timeline.directions,
            timeline.lane_directions, set(), len(packed.channel_keys), "")
        index: cython.Py_ssize_t
        data_begin: cython.Py_ssize_t = 2 * len(packed.channel_keys)
        totals: cython.double[::1] = self.busy_totals
        last_values: cython.double[:, ::1] = self.last_values
        moved: cython.longlong[::1] = packed.arrays[9]
        previous_moved: cython.longlong[::1] = self.before_moved
        directions: cython.char[::1] = packed.arrays[10]
        rid: object
        for rid, index in self.pending.items():
            if index >= data_begin:
                timeline.directions[rid] = "read" if directions[index] == 1 else "write"
        busy: dict = self.accumulator.resource_busy_ns
        byte_count: dict = self.accumulator.resource_bytes
        last: dict = self.accumulator.resource_last_intervals
        for rid, index in self.bill_busy_order.items():
            if self.changed_busy[index]:
                busy[rid] = float(totals[index])
        for rid, index in self.bill_byte_order.items():
            if self.changed_bytes[index]:
                byte_count[rid] = byte_count.get(rid, 0) + (int(moved[index]) - int(previous_moved[index]))
        for rid, index in self.bill_last_order.items():
            if self.changed_last[index]:
                last[rid] = (float(last_values[index, 0]), float(last_values[index, 1]))
        # A failed executor has reset the recorder; a checkpoint failure has
        # not. Restore exactly that recorder boundary, never the batch union.
        if timeline._intervals_truncated:
            timeline._touched.update(packed._id_objects[np.flatnonzero(packed.arrays[11])].tolist())
        self.pending.clear()


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
