"""The cohort resource index is exactly the original ordered owner scan."""
import pytest

from heterollm_sim import planner
from heterollm_sim.contracts import ResourceDemand
from heterollm_sim.reference import build_reference_scenario


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("resources", [(), (("unknown", 64),),
    (("shared", 0),), (("shared", 32),), (("third", 1), ("first", 1)),
    (("second-alias", 4), ("second", 8), ("shared", 16))])
def test_resource_index_preserves_order_deduplication_ambiguity_and_zero_bytes(bound, resources, monkeypatch):
    scenario = build_reference_scenario()
    candidates = tuple((name, False, "owner-" + name, frozenset(aliases)) for name, aliases in (
        ("first", ("first", "shared")), ("second", ("second", "second-alias", "shared")),
        ("third", ("third",))))
    monkeypatch.setattr(planner, "_physical_backing_candidates", lambda *args: candidates)
    demands = tuple(ResourceDemand(resource, 1., bytes_moved=size) for resource, size in resources)
    expected = [row for row in candidates if any(
        str(demand.resource_id) in row[3] and demand.bytes_moved > 0 for demand in demands)]
    def check():
        actual = planner._physical_backing_demand_candidates(scenario, None, None, demands)
        assert actual == expected
        assert all(left is right for left, right in zip(actual, expected))
    if bound:
        with planner._compilation_scope(scenario):
            check()
            check()
    else:
        check()
