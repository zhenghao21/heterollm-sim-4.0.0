"""Optional, ordered C reducers for built-in resource accounting dictionaries.

These functions do not schedule work or approximate counters.  Each addition
and publication has the same order as its Python loop.  Exact float operands
use a C double addition; arbitrary-size integers and numeric subclasses retain
Python arithmetic.  Custom mappings retain their mapping protocol; helpers requiring owned
built-in dictionaries explicitly validate that narrower contract.
"""
from __future__ import annotations

# These are compile-time annotations, not a runtime dependency.  The small
# shadow below lets the same source run when Cython is not installed.
try:
    import cython
except ImportError:
    class _PythonMode:
        compiled = False

        @staticmethod
        def cfunc(function):
            return function

        inline = cfunc

        @staticmethod
        def exceptval(*args, **kwargs):
            return lambda function: function

    cython = _PythonMode()

if cython.compiled:
    from cython.cimports.cpython.float import PyFloat_CheckExact, PyFloat_AS_DOUBLE
    from cython.cimports.cpython.object import PyObject_RichCompareBool, Py_EQ, Py_GT
    from cython.cimports.cpython.tuple import PyTuple_CheckExact, PyTuple_GET_ITEM, PyTuple_GET_SIZE
    from cython.cimports.cpython.ref import PyObject


@cython.cfunc
@cython.inline
def _add(previous: object, value: object) -> object:
    left: cython.double
    right: cython.double
    if cython.compiled:
        if PyFloat_CheckExact(previous) and PyFloat_CheckExact(value):
            left = PyFloat_AS_DOUBLE(previous)
            right = PyFloat_AS_DOUBLE(value)
            return left + right
    return previous + value


@cython.cfunc
@cython.inline
def _subtract(value: object, previous: object) -> object:
    left: cython.double
    right: cython.double
    if cython.compiled:
        if PyFloat_CheckExact(value) and PyFloat_CheckExact(previous):
            left = PyFloat_AS_DOUBLE(value)
            right = PyFloat_AS_DOUBLE(previous)
            return left - right
    return value - previous


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
def _setdefault(mapping: object, key: object, value: object) -> object:
    if cython.compiled and type(mapping) is dict:
        return cython.cast(dict, mapping).setdefault(key, value)
    return mapping.setdefault(key, value)


def timeline_metrics_delta(timeline, before):
    """Full public reducer, including externally edited resource dictionaries."""
    busy: dict = {}
    moved: dict = {}
    rid: object
    value: object
    # Keep both lookups: custom mappings can observe each original operation.
    for rid, value in timeline.busy_ns.items():
        if value != _get(before["busy_ns"], rid, 0.0):
            busy[rid] = _subtract(value, _get(before["busy_ns"], rid, 0.0))
    for rid, value in timeline.bytes_moved.items():
        if value != _get(before["bytes_moved"], rid, 0):
            moved[rid] = value - _get(before["bytes_moved"], rid, 0)
    intervals = {rid: timeline.last_intervals[rid] for rid in timeline._touched}
    payloads = {rid: tuple(items) for rid, items in timeline._interval_payloads.items()}
    return {"resource_busy_ns": busy, "resource_bytes": moved,
            "resource_last_intervals": intervals,
            "resource_intervals": {rid: tuple(items) for rid, items in timeline._intervals.items()},
            "resource_interval_payloads": payloads,
            "intervals_truncated": timeline._intervals_truncated}


def core_timeline_metrics_delta(timeline, before):
    """Sparse Core reducer with the original public dictionary iteration order."""
    touched = timeline._touched
    before_busy, before_bytes = before["busy_ns"], before["bytes_moved"]
    busy: dict = {}
    moved: dict = {}
    busy_values: dict = timeline.busy_ns
    byte_values: dict = timeline.bytes_moved
    rid: object
    value: object
    if len(touched) * 8 <= len(busy_values):
        for rid in timeline._ordered_metric_touches("busy_ns", busy_values, touched):
            value = busy_values[rid]
            if value != _get(before_busy, rid, 0.0):
                busy[rid] = _subtract(value, _get(before_busy, rid, 0.0))
    else:
        for rid, value in busy_values.items():
            if rid in touched and value != _get(before_busy, rid, 0.0):
                busy[rid] = _subtract(value, _get(before_busy, rid, 0.0))
    if len(touched) * 8 <= len(byte_values):
        for rid in timeline._ordered_metric_touches("bytes_moved", byte_values, touched):
            value = byte_values[rid]
            if value != _get(before_bytes, rid, 0):
                moved[rid] = value - _get(before_bytes, rid, 0)
    else:
        for rid, value in byte_values.items():
            if rid in touched and value != _get(before_bytes, rid, 0):
                moved[rid] = value - _get(before_bytes, rid, 0)
    intervals = {rid: timeline.last_intervals[rid] for rid in touched}
    return {"resource_busy_ns": busy, "resource_bytes": moved,
            "resource_last_intervals": intervals,
            "resource_intervals": {rid: tuple(items) for rid, items in timeline._intervals.items()},
            "resource_interval_payloads": {rid: tuple(items) for rid, items in timeline._interval_payloads.items()},
            "intervals_truncated": timeline._intervals_truncated}


def publish_timeline_frontiers(lane_available, resource_available):
    """Include zero-busy reservations when publishing physical readiness."""
    resource_id: object
    lanes: object
    for resource_id, lanes in lane_available.items():
        if lanes:
            _put(resource_available, resource_id, lanes[0] if len(lanes) == 1 else min(lanes))


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=True)
def _later(interval: object, previous: object) -> cython.bint:
    left_start: cython.pointer(PyObject)
    left_end: cython.pointer(PyObject)
    right_start: cython.pointer(PyObject)
    right_end: cython.pointer(PyObject)
    if cython.compiled:
        if (PyTuple_CheckExact(interval) and PyTuple_CheckExact(previous)
                and PyTuple_GET_SIZE(interval) >= 2 and PyTuple_GET_SIZE(previous) >= 2):
            left_start = PyTuple_GET_ITEM(interval, 0)
            left_end = PyTuple_GET_ITEM(interval, 1)
            right_start = PyTuple_GET_ITEM(previous, 0)
            right_end = PyTuple_GET_ITEM(previous, 1)
            if (PyFloat_CheckExact(cython.cast(object, left_start)) and PyFloat_CheckExact(cython.cast(object, left_end))
                    and PyFloat_CheckExact(cython.cast(object, right_start)) and PyFloat_CheckExact(cython.cast(object, right_end))):
                if left_end == right_end or PyFloat_AS_DOUBLE(cython.cast(object, left_end)) == PyFloat_AS_DOUBLE(cython.cast(object, right_end)):
                    return PyFloat_AS_DOUBLE(cython.cast(object, left_start)) > PyFloat_AS_DOUBLE(cython.cast(object, right_start))
                return PyFloat_AS_DOUBLE(cython.cast(object, left_end)) > PyFloat_AS_DOUBLE(cython.cast(object, right_end))
        # Tuple comparison uses identity-or-equality, including a shared NaN.
        if PyObject_RichCompareBool(interval[1], previous[1], Py_EQ):
            return PyObject_RichCompareBool(interval[0], previous[0], Py_GT)
        return PyObject_RichCompareBool(interval[1], previous[1], Py_GT)
    else:
        return (interval[1], interval[0]) > (previous[1], previous[0])


def merge_resource_bill(busy: dict, byte_count: dict, last: dict,
                        incoming_busy: dict, incoming_bytes: dict,
                        incoming_last: dict, owners=None, owner=None):
    """Accumulate one subsequent result into an existing resource bill.

    The caller retains its initial-result copies, so incoming integer/object
    identity on that first result is unchanged.  Optional owner publication
    has the same position relative to each busy addition as the Python loop.
    """
    resource_id: object
    value: object
    previous: object
    if (type(busy) is not dict or type(byte_count) is not dict
            or type(last) is not dict or type(incoming_busy) is not dict
            or type(incoming_bytes) is not dict or type(incoming_last) is not dict
            or (owners is not None and type(owners) is not dict)):
        raise TypeError("resource ledger requires built-in dictionaries")
    # data_motion's owner loop publishes intervals, bytes, then busy/owner.
    for resource_id, value in incoming_last.items():
        previous = last.get(resource_id)
        if previous is None or _later(value, previous):
            last[resource_id] = value
    for resource_id, value in incoming_bytes.items():
        byte_count[resource_id] = byte_count.get(resource_id, 0) + value
    for resource_id, value in incoming_busy.items():
        busy[resource_id] = _add(busy.get(resource_id, 0.0), value)
        if owners is not None:
            owners[resource_id] = owner


def publish_physical_summary(task_id, physical_ids, lane_available,
                             before_by_resource, last_by_resource,
                             resource_last_interval, lane_last_interval,
                             resource_available, predecessors):
    """Publish retained intervals and lane completions in resource order."""
    completions: list = []
    resource_id: object
    lanes: object
    before_lanes: object
    times: object
    previous: object
    interval: object
    start: object
    end: object
    before: object
    last: object
    lane_index: cython.Py_ssize_t
    for resource_id in physical_ids:
        lanes = _get(lane_available, resource_id, ())
        before_lanes = _get(before_by_resource, resource_id, ())
        times = _get(last_by_resource, resource_id)
        previous = _get(resource_last_interval, resource_id)
        interval = None
        if times is not None:
            start, end = times
            interval = {"task_id": task_id, "resource_id": resource_id,
                        "start_ns": start, "end_ns": end}
            if len(lanes) == 1 and previous is not None and previous["end_ns"] == start:
                _setdefault(predecessors, resource_id, dict(previous))
            if previous is None or end >= previous["end_ns"]:
                _put(resource_last_interval, resource_id, interval)
        # enumerate, rather than indexing, retains arbitrary lane sequences'
        # iteration behavior.  The index is C-sized; published indices remain
        # ordinary Python ints.
        for lane_index, end in enumerate(lanes):
            before = before_lanes[lane_index] if lane_index < len(before_lanes) else 0.0
            if end <= before:
                continue
            completions.append((resource_id, lane_index, end))
            if interval is not None and interval["end_ns"] == end and interval["start_ns"] >= before:
                last = {**interval, "lane": lane_index}
            else:
                last = {"task_id": task_id, "resource_id": resource_id,
                        "end_ns": end, "lane": lane_index}
            _put(lane_last_interval, (resource_id, lane_index), last)
        if lanes:
            _put(resource_available, resource_id, lanes[0] if len(lanes) == 1 else min(lanes))
    return tuple(completions)


def accumulate_demands(busy: dict, demands):
    """Add validated ResourceDemands and return their exact integer byte sum."""
    demand: object
    resource_id: object
    task_bytes: object = 0
    if type(busy) is not dict:
        raise TypeError("resource ledger requires a built-in dictionary")
    for demand in demands:
        resource_id = demand.resource_id
        busy[resource_id] = _add(busy.get(resource_id, 0.0), demand.service_ns)
        task_bytes += demand.bytes_moved
    return task_bytes


def account_single_owner_demands(demands, owner, owner_waits: dict,
                                busy: dict, queue_wait: dict, task_counts: dict,
                                capacities: dict, lanes: dict):
    """Account an already qualified, all-physical single-owner demand sequence.

    The event kernel must establish that every demand belongs to ``owner``.
    This function consumes exactly its input order, independently of summary
    interval publication order, and leaves other owners' waits untouched.
    ResourceDemand construction/validation remains with the caller.
    """
    demand: object
    resource_id: object
    owner_wait: object
    first: cython.bint = True
    if (type(owner_waits) is not dict or type(busy) is not dict
            or type(queue_wait) is not dict or type(task_counts) is not dict
            or type(capacities) is not dict or type(lanes) is not dict):
        raise TypeError("resource ledger requires built-in dictionaries")
    for demand in demands:
        resource_id = demand.resource_id
        busy[resource_id] = _add(busy.get(resource_id, 0.0), demand.service_ns)
        if first:
            owner_wait = owner_waits.pop(owner, 0.0)
            first = False
        else:
            owner_wait = 0.0
        queue_wait[resource_id] = _add(queue_wait.get(resource_id, 0.0), owner_wait)
        task_counts[resource_id] = task_counts.get(resource_id, 0) + 1
        capacities.setdefault(resource_id, 1)
        lanes[resource_id] = 0


def merge_latest_intervals(target, incoming):
    for resource_id, interval in incoming.items():
        previous = target.get(resource_id)
        if previous is None or _later(interval, previous):
            target[resource_id] = interval


def reduce_physical_results(physical_results: list, arrival_ns, optional_fields,
                            additive_fields: frozenset, peak_fields: frozenset,
                            read_operation):
    """Reduce every operation in its original order, retaining all counters."""
    row: dict
    optional_counts: dict
    operations: set
    execution_by_owner: dict = {}
    operations_by_owner: dict = {}
    optional_observed_by_owner: dict = {}
    for operation, owner, access_config, item in physical_results:
        row = execution_by_owner.get(owner)
        first_owner_result = row is None
        if row is None:
            row = {
                "kind": str(getattr(access_config.kind, "value", access_config.kind)),
                "operation_count": 0,
                "logical_bytes": 0,
                "logical_read_bytes": 0,
                "logical_write_bytes": 0,
                "physical_bytes": 0,
                "physical_read_bytes": 0,
                "physical_write_bytes": 0,
                "resource_busy_ns": {},
                "resource_bytes": {},
                "resource_owners": {},
                "resource_intervals": {}, "resource_interval_payloads": {},
                "resource_last_intervals": {},
                "details_truncated": False, "intervals_truncated": False,
                "arrival_ns": float(arrival_ns),
                "completion_ns": float(arrival_ns),
            }
            execution_by_owner[owner] = row
            operations_by_owner[owner] = set()
            optional_observed_by_owner[owner] = {key: 0 for key in optional_fields}
        operations_by_owner[owner].add(operation.value)
        operation_count = item.counters.get("physical_operation_count", 1)
        if type(operation_count) is not int or operation_count <= 0:
            raise ValueError("physical_operation_count must be a positive integer")
        row["operation_count"] += operation_count
        row["logical_bytes"] += item.logical_bytes
        row["logical_read_bytes" if operation is read_operation else "logical_write_bytes"] += item.logical_bytes
        row["physical_bytes"] += item.transfer_bytes
        row["physical_read_bytes"] += int(item.counters.get("physical_read_bytes", 0))
        row["physical_write_bytes"] += int(item.counters.get("physical_write_bytes", 0))
        row["completion_ns"] = max(row["completion_ns"], item.completion_ns)
        for key in ("details_truncated", "intervals_truncated"):
            row[key] = row[key] or bool(item.counters.get(key, False))
        for key in ("resource_intervals", "resource_interval_payloads"):
            for resource_id, intervals in item.counters.get(key, {}).items():
                row[key].setdefault(resource_id, []).extend(intervals)
        if first_owner_result:
            incoming_last = item.counters.get("resource_last_intervals", {})
            if type(incoming_last) is dict:
                row["resource_last_intervals"] = dict(incoming_last)
            else:
                merge_latest_intervals(row["resource_last_intervals"], incoming_last)
            row["resource_bytes"] = {rid: 0 + value
                for rid, value in item.counters.get("resource_bytes", {}).items()}
            row["resource_busy_ns"] = {rid: 0.0 + value
                for rid, value in item.counters.get("resource_busy_ns", {}).items()}
            row["resource_owners"] = dict.fromkeys(row["resource_busy_ns"], owner)
        elif all(type(item.counters.get(key, {})) is dict for key in (
                "resource_busy_ns", "resource_bytes", "resource_last_intervals")):
            merge_resource_bill(
                row["resource_busy_ns"], row["resource_bytes"], row["resource_last_intervals"],
                item.counters.get("resource_busy_ns", {}), item.counters.get("resource_bytes", {}),
                item.counters.get("resource_last_intervals", {}), row["resource_owners"], owner)
        else:
            merge_latest_intervals(row["resource_last_intervals"],
                                            item.counters.get("resource_last_intervals", {}))
            for resource_id, value in item.counters.get("resource_bytes", {}).items():
                row["resource_bytes"][resource_id] = row["resource_bytes"].get(resource_id, 0) + value
            for resource_id, value in item.counters.get("resource_busy_ns", {}).items():
                row["resource_busy_ns"][resource_id] = (
                    row["resource_busy_ns"].get(resource_id, 0.0) + value
                )
                row["resource_owners"][resource_id] = owner
        if item.counters.get("nand_observations") is not None:
            row.setdefault("_nand_observations", []).append(item.counters["nand_observations"])
        # Link bills carry only a few fields. Inspect fields that actually
        # exist rather than probing every NAND/DRAM counter on every hop.
        # Results still arrive in submission order, preserving each field's
        # floating-point addition order and weighted observation coverage.
        optional_counts = optional_observed_by_owner[owner]
        for key, value in item.counters.items():
            if key in additive_fields:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[key] = row.get(key, 0) + value
                    if key in optional_counts:
                        optional_counts[key] += operation_count
            elif key in peak_fields:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[key] = max(row.get(key, 0), value)
    for owner, row in execution_by_owner.items():
        for key, count in optional_observed_by_owner[owner].items():
            if count != row["operation_count"]:
                row[key] = None
        if "_nand_observations" in row:
            from .nand_core import summarize_nand_observations
            row["nand_observations"] = summarize_nand_observations(row.pop("_nand_observations"))
        row["service_ns"] = row["completion_ns"] - float(arrival_ns)
        operations = operations_by_owner[owner]
        row["operation"] = next(iter(operations)) if len(operations) == 1 else "read_write"
    return execution_by_owner


def account_task_demands(self, task, demands, is_physical, start_ns, effective_ready_ns):
    """Apply the kernel's ordered resource accounting without changing dispatch."""
    from collections.abc import Mapping
    resource_predecessors: dict = {}
    resource_lanes: dict = {}
    physical_ids = set(str(item) for item in task.metadata.get("physical_demands_resource_ids", ())) if is_physical else set()
    end_ns = float(task.metadata["physical_completion_ns"]) if is_physical else start_ns
    resource_last_interval = self.resource_last_interval
    resource_available = self.resource_available
    resource_busy_ns = self.resource_busy_ns
    resource_queue_wait_ns = self.resource_queue_wait_ns
    resource_task_count = self.resource_task_count
    queue_wait_ns = start_ns - effective_ready_ns
    physical_queue_wait_by_owner: dict = {}
    physical_resource_owners: dict = {}
    physical_resource_owner_sets: dict = {}
    single_physical_owner = None
    single_owner_resources = ()
    if is_physical:
        raw_by_owner = task.metadata.get("physical_execution_by_owner")
        if isinstance(raw_by_owner, Mapping):
            for raw_owner, raw_execution in raw_by_owner.items():
                if isinstance(raw_execution, Mapping):
                    raw_owners = raw_execution.get("resource_owners", {})
                    if len(raw_by_owner) == 1 and isinstance(raw_owners, Mapping):
                        single_physical_owner = str(raw_owner)
                        single_owner_resources = raw_owners
                    elif isinstance(raw_owners, Mapping):
                        physical_resource_owners.update({
                            str(resource_id): str(owner)
                            for resource_id, owner in raw_owners.items()
                        })
                        for resource_id in raw_owners:
                            rid, owner = str(resource_id), str(raw_owner)
                            previous_owner = physical_resource_owner_sets.get(rid)
                            if previous_owner is None:
                                physical_resource_owner_sets[rid] = owner
                            elif isinstance(previous_owner, set):
                                previous_owner.add(owner)
                            elif previous_owner != owner:
                                physical_resource_owner_sets[rid] = {previous_owner, owner}
                    try:
                        value = float(raw_execution.get("queue_wait_ns", 0.0))
                    except (TypeError, ValueError, OverflowError):
                        value = 0.0
                    physical_queue_wait_by_owner[str(raw_owner)] = value
        else:
            raise ValueError("physical tasks require physical_execution_by_owner")
    scheduled_demands = demands
    if (single_physical_owner is not None
            and all(type(values) is dict for values in (
                resource_busy_ns, resource_queue_wait_ns, resource_task_count,
                self.resource_capacities))
            and all(d.resource_id in physical_ids and d.resource_id in single_owner_resources
                    for d in demands)):
        account_single_owner_demands(
            demands, single_physical_owner, physical_queue_wait_by_owner,
            resource_busy_ns, resource_queue_wait_ns, resource_task_count,
            self.resource_capacities, resource_lanes)
        scheduled_demands = ()
    for demand in scheduled_demands:
        if is_physical and str(demand.resource_id) in physical_ids:
            # The core already reserved the real stages. Charging these
            # durations as another exclusive device demand would both
            # serialize independent channels and count contention twice.
            resource_busy_ns[demand.resource_id] = (
                _get(resource_busy_ns, demand.resource_id, 0.0) + demand.service_ns
            )
            owners = (single_physical_owner
                if demand.resource_id in single_owner_resources else
                physical_resource_owner_sets.get(demand.resource_id))
            if owners is None:
                owners = physical_resource_owners.get(
                    demand.resource_id, self._physical_owner(demand.resource_id)
                )
            # Several devices may use one explicit shared wire. Attribute
            # each device's host admission wait once, even when its only
            # positive-duration demand is on that shared resource.
            owner_wait = (physical_queue_wait_by_owner.pop(owners, 0.0)
                if isinstance(owners, str) else
                sum(physical_queue_wait_by_owner.pop(owner, 0.0) for owner in sorted(owners)))
            resource_queue_wait_ns[demand.resource_id] = (
                _get(resource_queue_wait_ns, demand.resource_id, 0.0)
                + owner_wait
            )
            resource_task_count[demand.resource_id] = _get(resource_task_count, demand.resource_id, 0) + 1
            self.resource_capacities.setdefault(demand.resource_id, 1)
            resource_lanes[demand.resource_id] = 0
            continue
        demand_end_ns = start_ns + demand.service_ns
        lanes, lane_index, available_ns = self._select_lane(
            demand.resource_id
        )
        owner_id = self._physical_owner(demand.resource_id)
        previous_interval = self._resource_lane_last_interval.get(
            (owner_id, lane_index)
        )
        resource_lanes[demand.resource_id] = lane_index
        if previous_interval is not None and available_ns == start_ns:
            resource_predecessors[demand.resource_id] = dict(
                previous_interval
            )
        if demand_end_ns > end_ns:
            end_ns = demand_end_ns
        interval: dict = {
            "task_id": task.task_id,
            "resource_id": demand.resource_id,
            "start_ns": start_ns,
            "end_ns": demand_end_ns,
        }
        if len(lanes) > 1:
            interval["lane"] = lane_index
        lanes[lane_index] = demand_end_ns
        available_ns = demand_end_ns if len(lanes) == 1 else min(lanes)
        resource_available[demand.resource_id] = available_ns
        resource_available[owner_id] = available_ns
        for logical_id, physical_id in self.resource_owners.items():
            if physical_id == owner_id:
                resource_available[logical_id] = available_ns
        resource_last_interval[demand.resource_id] = interval
        self._resource_lane_last_interval[(owner_id, lane_index)] = interval
        resource_busy_ns[demand.resource_id] = (
            _get(resource_busy_ns, demand.resource_id, 0.0)
            + demand.service_ns
        )
        resource_queue_wait_ns[demand.resource_id] = (
            _get(resource_queue_wait_ns, demand.resource_id, 0.0)
            + queue_wait_ns
        )
        resource_task_count[demand.resource_id] = (
            _get(resource_task_count, demand.resource_id, 0) + 1
        )

    return end_ns, physical_ids, resource_predecessors, resource_lanes



def physical_demands(resource_busy: dict, resource_bytes: dict, cache: dict, demand_type):
    """Materialize the complete exact bill, reusing identical immutable rows."""
    demands: list = []
    for resource_id, duration in sorted(resource_busy.items()):
        if not float(duration) > 0:
            continue
        # Keep both float conversions: custom numeric objects may implement
        # conversion, and the original filter and constructor each used it.
        key = (str(resource_id), float(duration), int(resource_bytes.get(resource_id, 0)))
        demand = cache.get(key)
        if demand is None:
            demand = demand_type(key[0], key[1], bytes_moved=key[2])
            if len(cache) >= 16384:
                cache.clear()
            cache[key] = demand
        demands.append(demand)
    return tuple(demands)


def observe_live_metrics(self, event, _better_path, _extend_path, TaskExecutionRecord, compact_execution_metadata):
    """Consume the complete event and all lane paths in original order."""
    task = event.task
    if task.task_id in self._names:
        raise ValueError("live cohort observed a duplicate task: {}".format(task.task_id))
    start_path = None
    for dependency_id in sorted(task.dependencies):
        if dependency_id in self._end_paths:
            start_path = _better_path(self._end_paths[dependency_id], start_path)
    for demand in event.demands:
        info = event.resource_predecessors.get(demand.resource_id)
        if info is None:
            continue
        if info["end_ns"] > event.start_ns:
            continue  # Internal Core wait is already in this task's span.
        lane = int(info.get("lane", event.resource_lanes.get(demand.resource_id, 0)))
        resource_id = info["resource_id"]
        previous = self._resource_paths.get((resource_id, lane))
        # Other cohorts can interleave on the same lane. An older task
        # from this cohort must not replace the actual external predecessor.
        if previous is not None and info["task_id"] == previous[0]:
            # Most physical lane releases precede the explicit task
            # dependency. Keep their exact distance but only allocate a
            # path node/key when it can win, including lexicographic ties.
            distance = previous[3]
            if distance > 0.0 and distance >= (start_path.distance if start_path is not None else 0.0):
                candidate = _extend_path(previous[1], previous[2], previous[4],
                                         "{}@{}".format(previous[0], resource_id))
                start_path = _better_path(candidate, start_path)
    duration = event.end_ns - event.start_ns
    self._end_paths[task.task_id] = _extend_path(
        start_path, duration, task.category, task.task_id)
    start_distance = start_path.distance if start_path is not None else 0.0
    record_resource_paths(self._resource_paths, task, event.demands,
                          event.start_ns, event.resource_lanes, start_path, start_distance)
    task_bytes = accumulate_demands(self._resource_busy, event.demands)
    self._task_bytes[task.task_id] = task_bytes
    self._records.append(TaskExecutionRecord(
        task_id=task.task_id, start_ns=event.start_ns, end_ns=event.end_ns,
        category=task.category, dependencies=tuple(task.dependencies),
        metadata=compact_execution_metadata(task.metadata), demands=event.demands))
    self._names[task.task_id] = task.name
    self._summary_cache = None


def resource_release_points(task, demands, start_ns, resource_lanes):
    """Yield actual per-lane releases, separately from summed resource busy time."""
    physical_ids = set(task.metadata.get("physical_demands_resource_ids", ()))
    for resource_id, lane, end_ns in task.metadata.get("physical_resource_lane_completions", ()):
        yield str(resource_id), int(lane), float(end_ns)
    for demand in demands:
        if demand.resource_id not in physical_ids:
            yield demand.resource_id, int(resource_lanes.get(demand.resource_id, 0)), start_ns + demand.service_ns



def record_resource_paths(paths, task, demands, start_ns, resource_lanes,
                          start_path, start_distance):
    """Fuse release iteration with its sole live observer, retaining every lane."""
    physical_ids = set(task.metadata.get("physical_demands_resource_ids", ()))
    resource_id: object
    lane: object
    end_ns: object
    duration: object
    for resource_id, lane, end_ns in task.metadata.get("physical_resource_lane_completions", ()):
        resource_id, lane, end_ns = str(resource_id), int(lane), float(end_ns)
        duration = end_ns - start_ns
        _put(paths, (resource_id, lane), (task.task_id, start_path, duration,
                                        start_distance + duration, task.category))
    for demand in demands:
        if demand.resource_id not in physical_ids:
            resource_id = demand.resource_id
            lane = int(_get(resource_lanes, demand.resource_id, 0))
            end_ns = start_ns + demand.service_ns
            duration = end_ns - start_ns
            _put(paths, (resource_id, lane), (task.task_id, start_path, duration,
                                            start_distance + duration, task.category))
