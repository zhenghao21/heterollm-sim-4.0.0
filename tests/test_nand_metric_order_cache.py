"""Append-only NAND request ordering retains every public counter bit."""
import pytest

from heterollm_sim import _nand_pack
from heterollm_sim.memory_transfer import ResourceTimeline
from test_nand_dense_metrics import account, ordered_bits, publish, timeline


def ordered(row, values, touched, scope, published=None):
    return _nand_pack.generic_ordered_metric_touches(
        row, "busy_ns", values, touched,
        dict.fromkeys(touched) if published is None else published, scope)


def test_identical_footprint_reuses_order_but_returns_an_independent_list():
    row = ResourceTimeline()
    values = {"c": 3., "a": 1., "b": 2.}
    scope = {}
    first = ordered(row, values, {"b", "c"}, scope)
    cached_order = scope["busy_ns"][3]
    first.clear()
    second = ordered(row, values, {"c", "b"}, scope)
    assert second == ["c", "b"]
    assert second is not first
    assert scope["busy_ns"][3] is cached_order
    assert type(second) is list


def test_mutating_the_same_touched_set_changes_the_cached_footprint():
    row = ResourceTimeline()
    values = {"c": 3., "a": 1., "b": 2.}
    scope = {}
    touched = {"b", "c"}
    assert ordered(row, values, touched, scope) == ["c", "b"]
    touched.remove("c")
    touched.add("a")
    assert ordered(row, values, touched, scope) == ["a", "b"]


def test_appended_key_invalidates_even_when_the_touched_set_is_unchanged():
    row = ResourceTimeline()
    values = {"c": 3., "b": 2.}
    touched = {"b", "a"}
    scope = {}
    assert ordered(row, values, touched, scope) == ["b"]
    values["a"] = 1.
    assert ordered(row, values, touched, scope) == ["b", "a"]


def test_replaced_mapping_root_uses_its_new_public_order():
    row = ResourceTimeline()
    scope = {}
    assert ordered(row, {"c": 3., "b": 2.}, {"b", "c"}, scope) == ["c", "b"]
    assert ordered(row, {"b": 2., "c": 3.}, {"b", "c"}, scope) == ["b", "c"]


@pytest.mark.parametrize("sequence", [
    [("r3", "r2")] * 3,
    [("r3", "r2"), ("r1", "r0"), ("r3", "r2")],
    [("r3", "r2"), ("a", "z"), ("z", "a")],
])
def test_entire_per_request_metrics_match_uncached_order_and_float_bits(sequence):
    actual, expected = timeline(), timeline()
    scope = {"pack_pages": 1024}
    for keys in sequence:
        assert ordered_bits(account(actual, keys, scope)) == ordered_bits(account(expected, keys, None))


def test_cache_hit_keeps_last_interval_failure_after_the_same_public_prefix():
    row = timeline()
    scope = {"pack_pages": 1024}
    account(row, ("r3", "r2"), scope)
    arguments = publish(row, ("r3", "r2"))
    row.last_intervals.pop("r2")
    before = ordered_bits(vars(row))
    with pytest.raises(KeyError, match="r2"):
        _nand_pack.generic_metrics_delta(row, *arguments, scope)
    assert ordered_bits(vars(row)) == before
