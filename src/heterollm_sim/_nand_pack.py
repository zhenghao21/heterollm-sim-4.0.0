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


def validate_buffered_cache(cache, bank_keys, prefixes, slot_type, pages_per_block, finite):
    """Read the ordinary two-slot cache in the scalar validation order."""
    if type(cache) is not dict:
        return None
    rows: list = []
    seen: set = set()
    bank: object
    prefix: object
    slots: object
    slot: object
    tag: object
    for bank, prefix in zip(bank_keys, prefixes):
        slots = cache.get(bank, ())
        if bank in cache and (type(slots) is not list or len(slots) != 2):
            return None
        for slot in slots:
            if (type(slot) is not slot_type or id(slot) in seen
                    or vars(slot).keys() != {"page_key", "ready_ns", "reader_completion_ns", "last_use"}
                    or type(slot.last_use) is not int or not 0 <= slot.last_use <= 2**63 - 1
                    or any(type(value) not in (int, float) or not finite(value) or value < 0
                           for value in (slot.ready_ns, slot.reader_completion_ns))):
                return None
            seen.add(id(slot))
            tag = slot.page_key
            if tag is not None and (type(tag) is not tuple or len(tag) != len(prefix) + 2
                    or tag[:-2] != prefix or any(type(value) is not int or value < 0 for value in tag)
                    or tag[-1] >= pages_per_block
                    or tag[-2] * pages_per_block + tag[-1] > 2**63 - 1):
                return None
        rows.append(slots)
    return rows


def pack_buffered_cache(rows, banks, pages_per_block):
    """Copy every validated slot to fresh arrays without retaining numeric state."""
    tags = np.full((banks, 2), -1, dtype=np.int64)
    ready, readers, uses = np.zeros((banks, 2)), np.zeros((banks, 2)), np.zeros((banks, 2), dtype=np.int64)
    present = np.zeros(banks, dtype=np.bool_)
    bank: cython.Py_ssize_t
    index: cython.Py_ssize_t
    slots: object
    slot: object
    for bank, slots in enumerate(rows):
        present[bank] = bool(slots)
        for index, slot in enumerate(slots):
            if slot.page_key is not None:
                tags[bank, index] = slot.page_key[-2] * pages_per_block + slot.page_key[-1]
            ready[bank, index], readers[bank, index], uses[bank, index] = (
                slot.ready_ns, slot.reader_completion_ns, slot.last_use)
    return tags, ready, readers, uses, present


def publish_buffered_cache(core, cache, bank_keys, prefixes, pages_per_block,
                           tags, ready, readers, uses, active_banks, slot_type):
    """Write the same touched slots, including transaction callbacks, in order."""
    tag_values, ready_values, reader_values, use_values = tags.tolist(), ready.tolist(), readers.tolist(), uses.tolist()
    bank: object
    key: object
    slots: object
    slot: object
    tag: object
    index: cython.Py_ssize_t
    for bank in np.flatnonzero(active_banks).tolist():
        key = bank_keys[bank]
        if core._nand_transactions:
            core._touch_cache_bank(key)
        slots = cache.get(key)
        if slots is None:
            slots = [slot_type(), slot_type()]
            cache[key] = slots
        for index, slot in enumerate(slots):
            tag = tag_values[bank][index]
            slot.page_key = None if tag < 0 else (*prefixes[bank], *divmod(tag, pages_per_block))
            slot.ready_ns, slot.reader_completion_ns, slot.last_use = (
                ready_values[bank][index], reader_values[bank][index], use_values[bank][index])


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
    timeline._remember_resources(active_resources)
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


def can_own_resource_aliases(aliases):
    """Inspect plain entries before a geometry key can hash caller objects."""
    if type(aliases) is not dict:
        return False
    name: object
    target: object
    values: dict = aliases
    for name, target in values.items():
        if type(name) is not str or type(target) is not str:
            return False
    return True


def can_own_timeline(timeline, resource_ids):
    """Reject authored conversion/setter callbacks on this owner's footprint."""
    columns = _native_timeline_columns(timeline)
    if columns is not None and type(resource_ids) is tuple:
        accepted = _can_own_native_columns(resource_ids, columns)
        if accepted is not None:
            return accepted
    rid: object
    value: object
    item: object
    mapping: object
    for rid in resource_ids:
        for mapping in (timeline.ready_ns, timeline.busy_ns):
            value = mapping.get(rid)
            if value is not None and type(value) is not int and type(value) is not float:
                return False
        value = timeline.bytes_moved.get(rid)
        if value is not None and type(value) is not int:
            return False
        value = timeline.last_intervals.get(rid)
        if value is not None:
            if type(value) is not tuple and type(value) is not list:
                return False
            if len(value) != 2:
                return False
            for item in value:
                if type(item) is not int and type(item) is not float:
                    return False
        value = timeline.lane_available.get(rid)
        if value is not None:
            if type(value) is not list:
                return False
            for item in value:
                if type(item) is not int and type(item) is not float:
                    return False
        value = timeline.directions.get(rid)
        if value is not None and type(value) is not str:
            return False
        value = timeline.lane_directions.get(rid)
        if value is not None:
            if type(value) is not list:
                return False
            for item in value:
                if item is not None and type(item) is not str:
                    return False
        for mapping in (timeline._intervals, timeline._interval_payloads):
            value = mapping.get(rid)
            if value is not None and type(value) is not list:
                return False
    return True


def _native_timeline_columns(timeline):
    """Read plain native maps once; public protocols retain the old walk."""
    if type(timeline) is not _NATIVE_TIMELINE_TYPE:
        return None
    namespace = vars(_NATIVE_TIMELINE_TYPE)
    slotnames = namespace.get("__slotnames__")
    if (slotnames is not None and (type(slotnames) is not list or slotnames)
            or len(namespace) - ("__slotnames__" in namespace) != len(_NATIVE_TIMELINE_NAMESPACE)
            or any(namespace.get(name) is not value for name, value in _NATIVE_TIMELINE_NAMESPACE)):
        return None
    state = vars(timeline)
    columns = tuple(state.get(name) for name in _NATIVE_TIMELINE_COLUMN_NAMES)
    return columns if all(type(column) is dict for column in columns) else None


def _can_own_native_columns(resource_ids, columns):
    # Cython's exact-dict locals avoid nine bound attribute/method lookups
    # per NAND bank. Values and shared mutable lanes are reread each call;
    # this is never a cached ownership verdict across physical phases.
    ready: dict = columns[0]
    busy: dict = columns[1]
    moved: dict = columns[2]
    last: dict = columns[3]
    lanes: dict = columns[4]
    directions: dict = columns[5]
    lane_directions: dict = columns[6]
    intervals: dict = columns[7]
    payloads: dict = columns[8]
    rid: object
    value: object
    item: object
    for rid in resource_ids:
        if type(rid) is not str:
            return None
        value = ready.get(rid)
        if value is not None and type(value) is not int and type(value) is not float:
            return False
        value = busy.get(rid)
        if value is not None and type(value) is not int and type(value) is not float:
            return False
        value = moved.get(rid)
        if value is not None and type(value) is not int:
            return False
        value = last.get(rid)
        if value is not None:
            if (type(value) is not tuple and type(value) is not list) or len(value) != 2:
                return False
            for item in value:
                if type(item) is not int and type(item) is not float:
                    return False
        value = lanes.get(rid)
        if value is not None:
            if type(value) is not list:
                return False
            for item in value:
                if type(item) is not int and type(item) is not float:
                    return False
        value = directions.get(rid)
        if value is not None and type(value) is not str:
            return False
        value = lane_directions.get(rid)
        if value is not None:
            if type(value) is not list:
                return False
            for item in value:
                if item is not None and type(item) is not str:
                    return False
        value = intervals.get(rid)
        if value is not None and type(value) is not list:
            return False
        value = payloads.get(rid)
        if value is not None and type(value) is not list:
            return False
    return True


from heterollm_sim.memory_transfer import ResourceTimeline as _NATIVE_TIMELINE_TYPE
_NATIVE_TIMELINE_NAMESPACE = tuple((name, value) for name, value in vars(_NATIVE_TIMELINE_TYPE).items()
                                    if name != "__slotnames__")
_NATIVE_TIMELINE_COLUMN_NAMES = ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "lane_available",
                               "directions", "lane_directions", "_intervals", "_interval_payloads")


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


def publish_active_timeline(timeline, names, state, moved, directions,
                            lane_values, lane_direction_values, starts, counts,
                            indexes, active_resources, bus_mask, direction_names, single_lane):
    """Publish only the native machine's touched columns, in original order.

    The page executor, its float64 calculations and the live calendar reads
    are unchanged. Reading typed active columns avoids converting thousands
    of unrelated resource rows to Python lists for a 256-page PD chunk.
    """
    rows: cython.double[:, ::1] = state
    byte_counts: cython.longlong[::1] = moved
    direction_values: cython.longlong[::1] = directions
    lanes: cython.double[::1] = lane_values
    lane_directions: cython.longlong[::1] = lane_direction_values
    firsts: cython.longlong[::1] = starts
    sizes: cython.longlong[::1] = counts
    positions: cython.longlong[::1] = indexes
    timeline._remember_resources(active_resources)
    lane_map: object = timeline.lane_available
    direction_map: object = timeline.directions
    lane_direction_map: object = timeline.lane_directions
    ready_map: object = timeline.ready_ns
    busy_map: object = timeline.busy_ns
    moved_map: object = timeline.bytes_moved
    interval_map: object = timeline.last_intervals
    offset: cython.Py_ssize_t
    index: cython.Py_ssize_t
    first: cython.Py_ssize_t
    size: cython.Py_ssize_t
    lane: cython.Py_ssize_t
    name: object
    slots: object
    values: list
    for offset in range(len(indexes)):
        index = positions[offset]
        name = names[index]
        first, size = firsts[index], sizes[index]
        if single_lane:
            slots = _item(lane_map, name)
            if cython.compiled and type(slots) is list:
                cython.cast(list, slots)[0] = float(lanes[first])
            else:
                slots[0] = float(lanes[first])
        else:
            values = []
            for lane in range(first, first + size):
                values.append(float(lanes[lane]))
            if name in lane_map:
                slots = _item(lane_map, name)
                slots[:] = values
            else:
                _put(lane_map, name, values)
        _put(ready_map, name, float(rows[index, 0]))
        _put(busy_map, name, float(rows[index, 1]))
        _put(interval_map, name, (float(rows[index, 2]), float(rows[index, 3])))
        if bus_mask[index] or name in moved_map:
            _put(moved_map, name, int(byte_counts[index]))
        if direction_values[index]:
            _put(direction_map, name, direction_names[direction_values[index]])
        if size > 1:
            values = []
            for lane in range(first, first + size):
                values.append(direction_names[lane_directions[lane]])
            _put(lane_direction_map, name, values)
        elif cython.compiled and type(lane_direction_map) is dict:
            cython.cast(dict, lane_direction_map).pop(name, None)
        else:
            lane_direction_map.pop(name, None)


def generic_request_resource_indexes(first_page, count, base_units, planes, independent,
                                     per_channel, host_channels, array_ids, channel_ids, host_ids):
    """Exact potential resource footprint of the unchanged sequential page loop."""
    first: cython.longlong = first_page
    base_count: cython.longlong = base_units
    bank_count: cython.longlong = base_units * planes
    plane: cython.longlong
    base: cython.longlong
    bank: cython.longlong
    unit: cython.longlong
    channel: cython.longlong
    page: cython.longlong
    offset: cython.Py_ssize_t
    arrays: cython.const[cython.longlong][::1] = array_ids
    channels: cython.const[cython.longlong][::1] = channel_ids
    hosts: cython.const[cython.longlong][::1] = host_ids
    selected: set = set()
    for offset in range(min(count, bank_count)):
        page = first + offset
        base = page % base_count
        plane = (page // base_count) % planes
        bank = base + plane * base_count
        unit = bank if independent else base
        channel = base // per_channel
        selected.add(int(arrays[unit]))
        selected.add(int(channels[channel]))
        selected.add(int(hosts[channel % host_channels]))
    return np.asarray(sorted(selected), dtype=np.int64)


def pack_selected_single_lane(timeline, names, indexes, count, transfer_page_bytes):
    """Pack the proven native request footprint without observing other banks.

    The outer owner transaction has already validated the full dynamic
    public state. Only the exact sequential page footprint is read here;
    selected aliases, multi-lane clocks and unusual values still decline to
    the complete original packer before any state is changed.
    """
    positions: cython.const[cython.longlong][::1] = indexes
    lane_map: object = timeline.lane_available
    direction_map: object = timeline.directions
    ready_map: object = timeline.ready_ns
    busy_map: object = timeline.busy_ns
    moved_map: object = timeline.bytes_moved
    interval_map: object = timeline.last_intervals
    rows: list = []
    previous: list = []
    moved_values: list = []
    seen: set = set()
    name: object
    row: object
    value: object
    direction: object
    interval: object
    offset: cython.Py_ssize_t
    index: cython.Py_ssize_t
    size: cython.Py_ssize_t = len(names)
    active_size: cython.Py_ssize_t = len(indexes)
    limit: object = 2**63 - 1 - count * max(4096, transfer_page_bytes)
    for offset in range(active_size):
        rows.append(_get(lane_map, names[positions[offset]]))
    for row in rows:
        if (type(row) is not list or len(row) != 1 or type(row[0]) not in (float, int)
                or id(row) in seen):
            return None
        seen.add(id(row))
    for offset in range(active_size):
        previous.append(_get(direction_map, names[positions[offset]]))
    for direction in previous:
        if direction is not None and direction != "read" and direction != "write":
            return None
    lanes = np.zeros(size, dtype=np.float64)
    lane_view: cython.double[::1] = lanes
    for offset in range(active_size):
        value = rows[offset][0]
        if not math.isfinite(value) or value < 0:
            return None
        lane_view[positions[offset]] = value
    for offset in range(active_size):
        moved_values.append(_get(moved_map, names[positions[offset]], 0))
    for value in moved_values:
        if not isinstance(value, int) or not 0 <= value <= limit:
            return None
    directions = np.zeros(size, dtype=np.int64)
    direction_view: cython.longlong[::1] = directions
    state = np.zeros((size, 4), dtype=np.float64)
    state_view: cython.double[:, ::1] = state
    moved = np.zeros(size, dtype=np.int64)
    moved_view: cython.longlong[::1] = moved
    for offset in range(active_size):
        index = positions[offset]
        direction = previous[offset]
        direction_view[index] = 0 if direction is None else 1 if direction == "read" else 2
        moved_view[index] = moved_values[offset]
    for offset in range(active_size):
        index = positions[offset]
        state_view[index, 0] = _double(_get(ready_map, names[index], 0.))
    for offset in range(active_size):
        index = positions[offset]
        state_view[index, 1] = _double(_get(busy_map, names[index], 0.))
    for offset in range(active_size):
        index = positions[offset]
        interval = _get(interval_map, names[index], (0., 0.))
        state_view[index, 2], state_view[index, 3] = _double(interval[0]), _double(interval[1])
    return (np.arange(size, dtype=np.int64), np.ones(size, dtype=np.int64),
            lanes, directions.copy(), state, moved, directions)
