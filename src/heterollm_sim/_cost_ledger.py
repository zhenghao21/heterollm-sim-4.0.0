"""Ordered resource-summary reducers, executable as Python or strict Cython.

Each demand is visited in its original order. Native dictionary access is
limited to exact built-in dicts; custom mappings retain their protocol calls.
Only exact float additions use C doubles, without contraction or fast-math.
"""
from __future__ import annotations

from collections.abc import Mapping

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

if cython.compiled:
    from cython.cimports.cpython.float import PyFloat_CheckExact, PyFloat_AS_DOUBLE


@cython.cfunc
@cython.inline
def _get(mapping: object, key: object, default: object = None) -> object:
    if cython.compiled and type(mapping) is dict:
        return cython.cast(dict, mapping).get(key, default)
    return mapping.get(key, default)


@cython.cfunc
@cython.inline
def _get_optional(mapping: object, key: object) -> object:
    if cython.compiled and type(mapping) is dict:
        return cython.cast(dict, mapping).get(key)
    return mapping.get(key)


@cython.cfunc
@cython.inline
def _item(mapping: object, key: object) -> object:
    if cython.compiled and type(mapping) is dict:
        return cython.cast(dict, mapping)[key]
    return mapping[key]


@cython.cfunc
@cython.inline
def _put(mapping: object, key: object, value: object) -> object:
    if cython.compiled and type(mapping) is dict:
        cython.cast(dict, mapping)[key] = value
    else:
        mapping[key] = value


@cython.cfunc
@cython.inline
def _add(previous: object, value: object) -> object:
    left: cython.double
    right: cython.double
    if cython.compiled and PyFloat_CheckExact(previous) and PyFloat_CheckExact(value):
        left = PyFloat_AS_DOUBLE(previous)
        right = PyFloat_AS_DOUBLE(value)
        return left + right
    return previous + value


@cython.cfunc
@cython.inline
def _iadd(previous: object, value: object) -> object:
    left: cython.double
    right: cython.double
    if cython.compiled and PyFloat_CheckExact(previous) and PyFloat_CheckExact(value):
        left = PyFloat_AS_DOUBLE(previous)
        right = PyFloat_AS_DOUBLE(value)
        return left + right
    previous += value
    return previous


def _task_memory_direction(task, task_names=None):
    """Classify authored tasks or records without materializing task proxies."""
    explicit = str(task.metadata.get("memory_direction", "")).lower()
    if explicit in {"read", "write"}:
        return explicit
    text = " ".join(
        (
            str(task.metadata.get("event_kind", "")),
            str(task.metadata.get("phase", "")),
            str(task.name if task_names is None else task_names[task.task_id]),
        )
    ).lower()
    return "write" if any(
        token in text for token in ("append", "write", "store", "output")
    ) else "read"


def _resource_busy_by_direction(
    tasks,
    *,
    task_names=None,
):
    rows: dict = {"read": {}, "write": {}}
    reads: dict = rows["read"]
    writes: dict = rows["write"]
    default_rows: dict
    service_ns: cython.double
    write_share: cython.double
    resource_id: object
    demand: object
    resource_directions: object
    modeled_write: object
    for task in tasks:
        direction = _task_memory_direction(task, task_names)
        default_rows = rows[direction]
        raw_resource_directions = task.metadata.get("resource_directions", {})
        resource_directions = (
            raw_resource_directions
            if isinstance(raw_resource_directions, Mapping)
            else {}
        )
        modeled_write = max(
            0, int(task.metadata.get("modeled_memory_write_bytes", 0))
        )
        for demand in task.demands:
            resource_id = str(demand.resource_id)
            explicit_resource_direction = _get(resource_directions, resource_id, "")
            if type(explicit_resource_direction) is not str or explicit_resource_direction not in ("read", "write", ""):
                explicit_resource_direction = str(explicit_resource_direction).lower()
            service_ns = max(0.0, float(demand.service_ns))
            if service_ns <= 0.0:
                continue
            if explicit_resource_direction == "read":
                reads[resource_id] = _add(reads.get(resource_id, 0.0), service_ns)
                continue
            if explicit_resource_direction == "write":
                writes[resource_id] = _add(writes.get(resource_id, 0.0), service_ns)
                continue
            write_share = 0.0
            if modeled_write > 0 and demand.bytes_moved > 0:
                write_share = min(
                    1.0, modeled_write / float(demand.bytes_moved)
                )
            if write_share > 0.0:
                writes[resource_id] = _add(writes.get(resource_id, 0.0), service_ns * write_share)
                reads[resource_id] = _add(reads.get(resource_id, 0.0), service_ns * (1.0 - write_share))
            else:
                default_rows[resource_id] = _add(default_rows.get(resource_id, 0.0), service_ns)
    return {
        direction: dict(sorted(values.items()))
        for direction, values in rows.items()
        if values
    }


def _accumulate_physical_resource_totals(resources, task, raw, resource_owners):
    """Add only this owner row's contribution to a possibly shared wire."""
    busy = _get(raw, "resource_busy_ns", {})
    moved = _get_optional(raw, "resource_bytes")
    resolved_owners = _get(raw, "resource_owners", {})
    checked_moved = False
    owner_map = resource_owners or {}
    for demand in task.demands:
        resource_id = str(demand.resource_id)
        if resource_id not in busy:
            continue
        if not checked_moved:
            if not isinstance(moved, Mapping):
                raise ValueError("physical owner rows require resource_bytes")
            checked_moved = True
        owner = _get(owner_map, resource_id, _get(resolved_owners, resource_id, "unknown"))
        row = _get_optional(resources, resource_id)
        if row is None:
            row = {"owner": owner, "bytes_moved": 0, "service_ns": 0.0}
            _put(resources, resource_id, row)
        if _item(row, "owner") != owner:
            _put(row, "owner", resource_id)
        _put(row, "bytes_moved", _iadd(_item(row, "bytes_moved"), int(_get(moved, resource_id, 0))))
        _put(row, "service_ns", _iadd(_item(row, "service_ns"), float(_item(busy, resource_id))))
