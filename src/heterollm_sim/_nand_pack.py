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


def validate_program_frontiers(addresses, banks, pages_per_block, bank_keys, current):
    """Validate every PROGRAM in order without NumPy scalar/Python division.

    execute_pages has already checked a nonnegative int64 page-address array.
    The clipped divisor is exact even for a larger authored block: all page
    indices are at most (2**63-1)//4096, hence remain in its first block.
    No live frontier is mutated until the numeric command batch succeeds.
    """
    frontiers: dict = {}
    bank_count: cython.longlong = banks
    block_span: cython.longlong = min(pages_per_block, 2**63 - 1)
    address: cython.longlong
    page_index: cython.longlong
    bank: cython.longlong
    physical_page: cython.longlong
    block: cython.longlong
    page: cython.longlong
    for address in addresses.tolist():
        page_index = address // 4096
        bank, physical_page = page_index % bank_count, page_index // bank_count
        block, page = physical_page // block_span, physical_page % block_span
        key = (*bank_keys[bank], block)
        # Evaluate the authored current lookup even on an intra-batch hit,
        # matching the existing mapping protocol and exception ordering.
        expected = frontiers.get(key, current.get(key, 0))
        if page != 0 and page != expected:
            raise ValueError("ocp_hbf_0_7 write order violation: block {} requires page {}, received {}"
                             .format(key, expected, page))
        frontiers[key] = page + 1
    return frontiers


def ordered_indexed_maximum(target, indexes, values):
    """Apply the ordinary maximum.at operation to finite contiguous pages.

    Validate the complete plain-array case before its first write. Other
    NumPy protocols, including exceptional/partially mutating cases, still
    enter the original ufunc with the original arguments.
    """
    if (cython.compiled and type(target) is np.ndarray and type(indexes) is np.ndarray
            and type(values) is np.ndarray
            and target.ndim == indexes.ndim == values.ndim == 1
            and target.dtype == values.dtype == np.dtype(np.float64)
            and indexes.dtype == np.dtype(np.int64)
            and target.flags.c_contiguous and indexes.flags.c_contiguous
            and values.flags.c_contiguous and target.flags.aligned
            and indexes.flags.aligned and values.flags.aligned
            and target.flags.writeable and len(indexes) == len(values)
            and not np.shares_memory(target, indexes)
            and not np.shares_memory(target, values)):
        output: cython.double[::1] = target
        positions: cython.const[cython.longlong][::1] = indexes
        incoming: cython.const[cython.double][::1] = values
        # Inspect exponent bits without floating-point operations: declining
        # a signalling NaN must not quiet it or change its payload first.
        target_bits: cython.const[cython.ulonglong][::1] = target.view(np.uint64)
        value_bits: cython.const[cython.ulonglong][::1] = values.view(np.uint64)
        exponent: cython.ulonglong = 0x7ff0000000000000
        negative_infinity: cython.ulonglong = 0xfff0000000000000
        bits: cython.ulonglong
        index: cython.Py_ssize_t
        position: cython.longlong
        size: cython.Py_ssize_t = len(target)
        count: cython.Py_ssize_t = len(indexes)
        valid: cython.bint = True
        for index in range(size):
            bits = target_bits[index]
            if bits & exponent == exponent and bits != negative_infinity:
                valid = False
                break
        if valid:
            for index in range(count):
                position = positions[index]
                if (position < 0 or position >= size
                        or value_bits[index] & exponent == exponent):
                    valid = False
                    break
        if valid:
            for index in range(count):
                position = positions[index]
                # maximum selects the later operand on a finite tie. Store
                # it as-is so +0/-0 follow the original input order.
                if incoming[index] >= output[position]:
                    output[position] = incoming[index]
            return None
    return np.maximum.at(target, indexes, values)


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


@cython.cfunc
@cython.inline
def _admission_clock(value: object) -> cython.double:
    if type(value) is float or type(value) is int:
        return value
    # Single-cell ndarray assignment rejects Python sequences which np.float64
    # construction can accept. Retain that original public-state conversion.
    scalar = np.empty(1, dtype=np.float64)
    scalar[0] = value
    return float(scalar[0])


def pack_host_admission(queues, acceptances, acceptance, host_channels,
                        outstanding_limit, read_operation):
    """Pack every public FIFO in order without NumPy scalar assignments.

    The queue remains public controller state. No cached copy is trusted
    across physical submissions, including a caller replacing a queue.
    """
    width = max(outstanding_limit,
                max((len(queue) for queue in queues.values()), default=0)) + 1
    admission = np.zeros((host_channels, width), dtype=np.float64)
    operations = np.zeros((host_channels, width), dtype=np.int64)
    counts = np.zeros(host_channels, dtype=np.int64)
    frontiers = np.zeros(host_channels, dtype=np.float64)
    times: cython.double[:, ::1] = admission
    kinds: cython.longlong[:, ::1] = operations
    sizes: cython.longlong[::1] = counts
    ready: cython.double[::1] = frontiers
    host: cython.Py_ssize_t
    index: cython.Py_ssize_t
    queue: object
    end: object
    operation: object
    value: object
    for host in range(host_channels):
        queue = _get(queues, host, ())
        sizes[host] = len(queue)
        value = max(_get(acceptances, host, 0.), acceptance)
        ready[host] = _admission_clock(value)
        for index, (end, operation) in enumerate(queue):
            times[host, index] = _admission_clock(end)
            kinds[host, index] = 1 if operation is read_operation else 2
    return admission, operations, counts, frontiers


def host_admission_mask(banks_touched, arrays_touched,
                        base_units: cython.Py_ssize_t,
                        per_channel: cython.Py_ssize_t,
                        host_channels: cython.Py_ssize_t):
    """Reduce the existing READ-bank and WRITE-unit masks to their hosts.

    Every READ marks its bank, including a page-cache hit; every WRITE marks
    its array unit. Shared-plane unit IDs retain the same base-unit modulo.
    Keeping this reduction outside the page machine leaves its numeric loop
    unchanged and touches each controller bank at most once here.
    """
    result = np.zeros(host_channels, dtype=np.bool_)
    banks: cython.uchar[::1] = banks_touched.view(np.uint8)
    units: cython.uchar[::1] = arrays_touched.view(np.uint8)
    active: cython.uchar[::1] = result.view(np.uint8)
    index: cython.Py_ssize_t
    for index in range(len(banks_touched)):
        if banks[index]:
            active[(index % base_units) // per_channel % host_channels] = 1
    for index in range(len(arrays_touched)):
        if units[index]:
            active[(index % base_units) // per_channel % host_channels] = 1
    return result


def publish_host_admission(queues, acceptances, admission, operations,
                           counts, frontiers, touched, read_operation,
                           write_operation):
    """Publish independent FIFO lists, including all per-command ends."""
    times: cython.double[:, ::1] = admission
    kinds: cython.longlong[:, ::1] = operations
    sizes: cython.longlong[::1] = counts
    ready: cython.double[::1] = frontiers
    active: cython.uchar[::1] = touched.view(np.uint8)
    host: cython.Py_ssize_t
    index: cython.Py_ssize_t
    size: cython.Py_ssize_t
    row: list
    for host in range(len(counts)):
        if not active[host]:
            continue
        size = sizes[host]
        row = []
        for index in range(size):
            row.append((float(times[host, index]),
                        read_operation if kinds[host, index] == 1 else write_operation))
        _put(queues, host, row)
        _put(acceptances, host, float(ready[host]))
