"""Direct operand indexing must preserve the serial subtraction contract."""
import random
from collections.abc import Mapping
from copy import deepcopy

import pytest

from heterollm_sim.planner import (
    _gddr_non_negative_int,
    _subtract_direct_buffer_accesses,
)


def _serial_subtract(explicit, remote):
    """Reference: visit all remaining rows for each remote interval."""
    if not isinstance(explicit, (tuple, list)) or not all(isinstance(row, Mapping) for row in explicit):
        raise ValueError("buffer_accesses must be a sequence of mappings")
    local = [dict(row) for row in explicit]
    for access in remote:
        start = int(access["offset_bytes"])
        pending = [(start, start + int(access["byte_count"]))]
        remaining = []
        for row in local:
            if row.get("buffer_id") != access["buffer_id"] or row.get("operation") != access["operation"]:
                remaining.append(row)
                continue
            generation = row.get("allocation_generation")
            if generation is not None and int(generation) != int(access["allocation_generation"]):
                remaining.append(row)
                continue
            lo = _gddr_non_negative_int(row.get("offset_bytes", 0), "offset_bytes")
            hi = lo + _gddr_non_negative_int(row.get("size_bytes"), "size_bytes")
            pieces, unresolved = [(lo, hi)], []
            for begin, end in pending:
                overlap_lo, overlap_hi = max(begin, lo), min(end, hi)
                if overlap_lo >= overlap_hi:
                    unresolved.append((begin, end))
                    continue
                if begin < overlap_lo:
                    unresolved.append((begin, overlap_lo))
                if overlap_hi < end:
                    unresolved.append((overlap_hi, end))
                pieces = [(a, b) for left, right in pieces
                          for a, b in ((left, min(right, overlap_lo)), (max(left, overlap_hi), right)) if a < b]
            pending = unresolved
            for begin, end in pieces:
                residual = {**row, "offset_bytes": begin}
                residual["size_bytes"] = end - begin
                remaining.append(residual)
        if pending:
            raise ValueError("direct RHS is not represented by the full buffer_accesses contract")
        local = remaining
    return tuple(local)


def _remote(identity="kv", operation="read", start=0, size=1, generation=0):
    return dict(buffer_id=identity, operation=operation, offset_bytes=start,
                byte_count=size, allocation_generation=generation)


def _row(identity="kv", operation="read", start=0, size=16, **extra):
    return dict(buffer_id=identity, operation=operation, offset_bytes=start,
                size_bytes=size, **extra)


def _exact(value):
    if isinstance(value, dict):
        return (type(value), tuple((key, _exact(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return type(value), tuple(map(_exact, value))
    return type(value), value.hex() if isinstance(value, float) else value


def _compare(explicit, remote):
    before = deepcopy((explicit, remote))
    try:
        expected = _serial_subtract(explicit, remote)
    except Exception as error:
        with pytest.raises(type(error)) as caught:
            _subtract_direct_buffer_accesses(explicit, remote)
        assert str(caught.value) == str(error)
    else:
        result = _subtract_direct_buffer_accesses(explicit, remote)
        assert _exact(result) == _exact(expected)
        assert all(row is not source for row in result for source in explicit)
    assert _exact((explicit, remote)) == _exact(before)


@pytest.mark.parametrize("explicit,remote", [
    ([], []),
    ([_row()], []),
    ([_row()], [_remote(start=4, size=4), _remote(start=10, size=2)]),
    ([_row(start=0, size=4), _row("other"), _row(start=4, size=12)],
     [_remote(start=2, size=6)]),
    ([_row(), _row(), _row(operation="write")], [_remote(size=8)]),
    ([_row(allocation_generation=2), _row(allocation_generation=None),
      _row(allocation_generation="3")], [_remote(size=8, generation=3)]),
    ([_row(), _row("local"), _row("kv2")],
     [_remote("kv2", start=4, size=8), _remote(start=4, size=8)]),
    ([_row(["unhashable"]), _row()], [_remote(size=8)]),
    ([_row(["unhashable"])], [_remote(["unhashable"], size=8)]),
    ([_row(operation=["unhashable"]), _row()], [_remote(size=8)]),
    ([_row(), _row(size=-1)], [_remote(size=16)]),
    ([_row(allocation_generation=1)], [_remote()]),
    ([_row()], [_remote(size=17)]),
    ([_row()], [_remote(size=16), _remote()]),
    ([_row()], [_remote(size=0)]),
    ([_row()], [_remote(start=4, size=-1)]),
    ([{"buffer_id": "kv", "operation": "read", "size_bytes": True}], [_remote()]),
    ([{"buffer_id": "other", "operation": "read", "size_bytes": True}, _row()], [_remote()]),
    ([None], []),
    ({}, []),
])
def test_indexed_subtraction_matches_serial_order_types_and_errors(explicit, remote):
    _compare(explicit, remote)


@pytest.mark.parametrize("seed", range(40))
def test_indexed_subtraction_matches_serial_for_overlapping_and_repeated_ranges(seed):
    rng = random.Random(seed)
    explicit = [_row("kv" + str(rng.randrange(4)), rng.choice(("read", "write")),
                     rng.randrange(20), rng.randrange(1, 30),
                     allocation_generation=rng.choice((None, 0, 1, "2")),
                     marker=-0.0 if index % 2 else 1.25)
                for index in range(30)]
    remote = [_remote("kv" + str(rng.randrange(4)), rng.choice(("read", "write")),
                      rng.randrange(20), rng.randrange(1, 15), rng.randrange(3))
              for _ in range(10)]
    _compare(explicit, remote)


@pytest.mark.parametrize("batch", (256, 1024))
def test_batched_distinct_kv_owners_leave_local_rows_in_original_order(batch):
    explicit = [_row("kv" + str(index), size=32) for index in range(batch)]
    explicit[batch // 2:batch // 2] = [_row("query"), _row("output", "write")]
    remote = [_remote("kv" + str(index), size=32) for index in reversed(range(batch))]
    _compare(explicit, remote)
