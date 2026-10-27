"""Dense NAND accounting preserves ordered counter deltas and owned rollback."""
import numpy as np
import pytest

from heterollm_sim import _nand_pack
from heterollm_sim.memory_transfer import ResourceTimeline


KEYS = tuple(f"r{index}" for index in range(32))
BUSY_ORDER = KEYS[1::2] + KEYS[::2]
BYTE_ORDER = KEYS[::2] + KEYS[1::2]
NEW_BUSY = float.fromhex("0x1.0000000000001p-3")


def timeline():
    # Distinct insertion orders must survive independently. Byte counters
    # exceed 2**53 so an accidental float conversion loses integer precision.
    return ResourceTimeline(
        busy_ns={key: 2.**53 + index * 2. for index, key in enumerate(BUSY_ORDER)},
        bytes_moved={key: 2**53 + index for index, key in enumerate(BYTE_ORDER)},
        last_intervals={key: (index + .125, index + .375) for index, key in enumerate(KEYS)},
    )


def publish(row, keys, *, unchanged=()):
    """Publish known native-style counter changes, recording their before-images."""
    names = list(row.busy_ns)
    names += [key for key in keys if key not in row.busy_ns]
    previous_busy = np.array([row.busy_ns.get(key, 0.) for key in names], dtype=np.float64)
    previous_moved = np.array([row.bytes_moved.get(key, 0) for key in names], dtype=np.int64)
    row._touched = set()
    for key in keys:
        row._remember_resource(key)
        if key not in unchanged:
            row.busy_ns[key] = float(np.nextafter(row.busy_ns.get(key, .125), np.inf))
            row.bytes_moved[key] = row.bytes_moved.get(key, 0) + 4097
        row.last_intervals[key] = (.125, row.busy_ns[key])
        row._touched.add(key)
    indexes = np.array([names.index(key) for key in keys], dtype=np.int64)
    return names, indexes, previous_busy, previous_moved


def account(row, keys, scope, **kwargs):
    return _nand_pack.generic_metrics_delta(row, *publish(row, keys, **kwargs), scope)


def assert_metrics(result, row, busy_keys, byte_keys, busy_values):
    assert list(result) == ["resource_busy_ns", "resource_bytes", "resource_last_intervals",
                            "resource_intervals", "resource_interval_payloads", "intervals_truncated"]
    assert list(result["resource_busy_ns"]) == list(busy_keys)
    assert [value.hex() for value in result["resource_busy_ns"].values()] == [value.hex() for value in busy_values]
    assert list(result["resource_bytes"].items()) == [(key, 4097) for key in byte_keys]
    assert all(type(value) is int for value in result["resource_bytes"].values())
    assert result["resource_last_intervals"] == {key: row.last_intervals[key] for key in row._touched}
    assert list(result["resource_last_intervals"]) == list(row._touched)
    assert result["resource_intervals"] == result["resource_interval_payloads"] == {}
    assert result["intervals_truncated"] is row._intervals_truncated


def ordered_bits(value):
    if isinstance(value, float):
        return float, value.hex()
    if isinstance(value, dict):
        return dict, tuple((key, ordered_bits(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(ordered_bits(item) for item in value)
    if isinstance(value, set):
        return set, frozenset(value)
    return type(value), value


def test_dense_append_invalidates_ordinals_before_reverse_sparse_publication():
    row = timeline()
    scope = {"pack_pages": 1024}
    first = account(row, ["r3", "r2"], scope)
    assert_metrics(first, row, ["r3", "r2"], ["r2", "r3"], [2., 2.])
    assert "busy_ns" in scope and "bytes_moved" in scope

    dense = account(row, [*BUSY_ORDER, "z", "a"], scope)
    assert_metrics(dense, row, [*BUSY_ORDER, "z", "a"], [*BYTE_ORDER, "z", "a"],
                   [2.] * 32 + [NEW_BUSY, NEW_BUSY])
    assert scope == {"pack_pages": 1024}

    # Dense publication appended z then a. The next sparse publication's
    # a,z order must not replace that public-map order in rebuilt ordinals.
    sparse = account(row, ["a", "z"], scope)
    assert_metrics(sparse, row, ["z", "a"], ["z", "a"], [2.**-55, 2.**-55])
    for name, values in (("busy_ns", row.busy_ns), ("bytes_moved", row.bytes_moved)):
        assert scope[name][0] is values
        assert (scope[name][1]["z"], scope[name][1]["a"]) == (32, 33)
    assert scope["pack_pages"] == 1024


@pytest.mark.parametrize("with_scope", [False, True])
@pytest.mark.parametrize("count", [1, 4, 5, 32])
def test_existing_and_unchanged_counters_across_density_boundary(with_scope, count):
    row = timeline()
    scope = {} if with_scope else None
    selected = set(KEYS[:count]) - {"r0"}
    busy_keys = [key for key in BUSY_ORDER if key in selected]
    byte_keys = [key for key in BYTE_ORDER if key in selected]
    result = account(row, list(reversed(KEYS[:count])), scope, unchanged=("r0",))

    assert_metrics(result, row, busy_keys, byte_keys, [2.] * len(busy_keys))
    if with_scope:
        assert set(scope) == ({"busy_ns", "bytes_moved"} if count <= 4 else set())


def test_busy_and_byte_maps_apply_density_to_their_own_key_counts():
    row = timeline()
    row.bytes_moved = {key: row.bytes_moved[key] for key in ("r0", "r2", "r1", "r3")}
    scope = {"pack_pages": 1024}
    result = account(row, ["r3", "r1", "r2", "r0"], scope)

    assert_metrics(result, row, ["r1", "r3", "r0", "r2"], ["r0", "r2", "r1", "r3"], [2.] * 4)
    # Four busy keys out of 32 are sparse; all four byte keys are dense.
    assert set(scope) == {"pack_pages", "busy_ns"}


@pytest.mark.parametrize("fail", [False, True])
def test_dense_metrics_and_exception_leave_owned_rollback_intact(fail):
    row = timeline()
    before = ordered_bits(vars(row))
    roots = (row.busy_ns, row.bytes_moved, row.last_intervals)
    scope = {"pack_pages": 1024}
    token = row._begin_owned_transaction()
    try:
        arguments = publish(row, [*BUSY_ORDER, "z", "a"])
        if fail:
            row.last_intervals.pop("a")
            published = ordered_bits(vars(row))
            with pytest.raises(KeyError, match="a"):
                _nand_pack.generic_metrics_delta(row, *arguments, scope)
            assert ordered_bits(vars(row)) == published
        else:
            result = _nand_pack.generic_metrics_delta(row, *arguments, scope)
            assert_metrics(result, row, [*BUSY_ORDER, "z", "a"], [*BYTE_ORDER, "z", "a"],
                           [2.] * 32 + [NEW_BUSY, NEW_BUSY])
    finally:
        # submit_many discards this private scope on either outcome. The
        # original owned journal still restores all public values and roots.
        scope.clear()
        token.rollback()

    assert all(actual is previous for actual, previous in zip(
        (row.busy_ns, row.bytes_moved, row.last_intervals), roots))
    assert ordered_bits(vars(row)) == before
