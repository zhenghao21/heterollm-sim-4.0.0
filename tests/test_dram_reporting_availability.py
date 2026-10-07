"""An uncollected physical statistic must not become a measured zero."""
import json
from types import SimpleNamespace

import pytest

from heterollm_sim.reporting import _sum_batch_dram_traffic


def aggregate(*ledgers):
    batches = [SimpleNamespace(cost=SimpleNamespace(metadata={"dram_traffic": ledger}))
               for ledger in ledgers]
    return _sum_batch_dram_traffic(SimpleNamespace(serving=SimpleNamespace(batches=batches)))


@pytest.mark.parametrize("ledgers", [(), ({"task_count": 1},), ({"task_count": 1}, {"task_count": 2})])
def test_uncollected_statistics_are_null_and_profiles_are_not_claimed_complete(ledgers):
    result = json.loads(json.dumps(aggregate(*ledgers)))
    for key in ("read_write_switches", "refresh_wait_ns", "turnaround_wait_ns"):
        assert result[key] is None
        assert result["metric_availability"][key]["status"] == "not_recorded"
    assert result["organization_profiles"] == []
    assert result["metric_availability"]["organization_profiles"]["status"] == "not_recorded"


def test_partial_statistic_is_not_published_as_a_complete_total():
    result = aggregate({"task_count": 1, "read_write_switches": 5,
                        "organization_profiles": [{"banks": 8}]}, {"task_count": 1})
    assert result["read_write_switches"] is None
    assert result["metric_availability"]["read_write_switches"] == {
        "status": "partial", "recorded_batch_count": 1, "physical_batch_count": 2}
    assert result["metric_availability"]["organization_profiles"]["status"] == "partial"
    assert result["organization_profiles"] == [{"banks": 8}]


def test_real_zero_and_measured_totals_remain_distinct_from_missing_statistics():
    result = aggregate({"task_count": 1, "read_write_switches": 0,
                        "refresh_wait_ns": 0.0, "turnaround_wait_ns": 3.0,
                        "physical_read_bytes": 64, "service_ns": 20.0},
                       {"task_count": 2, "read_write_switches": 2,
                        "refresh_wait_ns": 0.0, "turnaround_wait_ns": 4.0,
                        "physical_read_bytes": 128, "service_ns": 40.0})
    assert result["read_write_switches"] == 2
    assert result["refresh_wait_ns"] == 0.0
    assert result["turnaround_wait_ns"] == 7.0
    assert result["physical_read_bytes"] == 192
    assert result["service_ns"] == 60.0
    for key in ("read_write_switches", "refresh_wait_ns", "turnaround_wait_ns"):
        assert result["metric_availability"][key]["status"] == "recorded"
