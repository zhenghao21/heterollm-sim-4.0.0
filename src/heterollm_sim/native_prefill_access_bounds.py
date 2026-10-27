"""Experimental DRAM row-state envelope reset at each contiguous access.

Unlike the window's first-bank bound, every access may inherit a different
row state here, even if an earlier access already reached that bank.  This
allows omitted access history to change later row outcomes.  It does not
claim that a declared allocation has the address of a native continuation,
or that row state is the only omitted controller/cache state.

The propagation retains native_row_bounds' fixed executed resource order
and per-term serial-barrier conditions.  It is an optional comparison
experiment, not a replacement for the existing window bound or proof of a
full physical error interval.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .memory_types import DramConfig
from .native_row_bounds import _Device


PER_ACCESS_BOUND_ASSUMPTION = (
    "Every contiguous access may inherit an arbitrary DRAM row outcome under the existing "
    "first_access_penalty_ns model. Each access's declared address mapping and the executed "
    "dependency/resource order are retained, and terms are separated by serial barriers. "
    "This does not bound address-placement changes within an access, L2 traffic changes, "
    "refresh phase differences or every other controller/cache state.")


def _fresh_accesses(device, address, byte_count):
    """Equivalent to a fresh _Device.first_accesses for this one range.

    A contiguous range visits each die once. Banks can repeat inside that
    die segment, so numpy.unique counts them once; later die segments have
    disjoint bank identities. Reusing geometry avoids allocating/clearing a
    whole-device seen-bank bitmap for every small access. Work scales with
    touched dies and banks, never with every burst of a large transfer.
    """
    end = address + byte_count
    while address < end:
        stop = min(end, (address // device.per_die + 1) * device.per_die)
        stack = address // device.per_stack
        mapped = device._lanes_banks(address % device.per_die, stop - address)
        if mapped is None:
            lanes = np.repeat(np.arange(device.lanes, dtype=np.int64), device.banks)
            banks = np.tile(np.arange(device.banks, dtype=np.int64), device.lanes)
        else:
            lanes, banks = mapped
        # Stack/die are constant inside this segment, so lane*bank_count+bank
        # is sufficient to identify its unique banks. The path mapping still
        # receives the actual stack, as in native_row_bounds._Device.
        _, first = np.unique(lanes * device.banks + banks, return_index=True)
        paths, counts = np.unique(device._paths(stack, lanes[first]), return_counts=True)
        yield from zip(paths.tolist(), (count * device.penalty for count in counts.tolist()))
        address = stop


def per_access_ns(facts, configs, terms):
    """Return an experimental row-state width per term, in nanoseconds.

    Inputs match native_row_bounds.first_access_ns: each fact is
    ``(task_id, start_ns, end_ns, dependencies, resource_ids, accesses)``;
    each contiguous access is ``(owner, address, byte_count)``. ``configs``
    maps physical owners to configs and ``terms`` maps task ids to term ids.
    Owners without DRAM rows are ignored. The source facts/configs are read
    only. No simulator, allocator, cache or clock is advanced.
    """
    devices = {owner: _Device(config) for owner, config in configs.items()
               if isinstance(config, DramConfig)}
    ordered = sorted(range(len(facts)), key=lambda index: (facts[index][1], facts[index][2], index))
    delay, last_user, result = {}, {}, {}
    for index in ordered:
        task_id, _start, _end, dependencies, resources, accesses = facts[index]
        sums = defaultdict(float)
        for owner, address, byte_count in accesses:
            device = devices.get(owner)
            if device is not None and byte_count > 0:
                for path, penalty in _fresh_accesses(device, address, byte_count):
                    sums[owner, path] += penalty
        term = terms.get(task_id)
        if term is None:
            for resource in resources:
                last_user[resource] = task_id
            continue
        before = [delay[parent] for parent in dependencies if terms.get(parent) == term]
        for resource in resources:
            previous = last_user.get(resource)
            if previous is not None and terms.get(previous) == term:
                before.append(delay[previous])
            last_user[resource] = task_id
        delay[task_id] = max(sums.values(), default=0.0) + max(before, default=0.0)
        result[term] = max(result.get(term, 0.0), delay[task_id])
    return result


__all__ = ["per_access_ns", "PER_ACCESS_BOUND_ASSUMPTION"]
