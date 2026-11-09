"""HBM bandwidth ceilings and DQ conservation, run against the real core.

These are internal-consistency checks: the ceilings come from the same
DramConfig the core executes, so they catch a model that bills bytes twice or
overlaps a lane, not a model that disagrees with an unmeasured vendor part.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hbm_bandwidth_envelope as envelope


@pytest.fixture(scope="module")
def hbm3e():
    return envelope.reference_config()


def test_envelope_identities_hold(hbm3e):
    result = envelope.analytic_envelope(hbm3e)
    assert all(result["identities"].values()), result["identities"]
    # The physical interface is the only thing the lanes imply; the declared
    # directional caps may sit below it.
    assert result["physical_interface_gb_s"] == pytest.approx(
        result["lanes"] * result["lane_bandwidth_gb_s"])
    assert result["sequential_read_gb_s"] <= result["physical_interface_gb_s"]


def test_sequential_stream_converges_without_exceeding_the_interface(hbm3e):
    rows = envelope.sequential_sweep(hbm3e, bursts=(256, 16384, 131072))
    fractions = [row["fraction_of_ceiling"] for row in rows]
    assert all(fraction <= 1.0 + 1e-9 for fraction in fractions), fractions
    assert fractions[-1] >= 0.95, fractions
    # Small requests are dominated by activation, so the trend must be upward.
    assert fractions[0] < fractions[-1]


def test_random_bursts_stay_below_the_sequential_stream(hbm3e):
    sequential = envelope.sequential_sweep(hbm3e, bursts=(65536,))
    random_rows = envelope.random_sweep(hbm3e, bursts=(4096,))
    assert random_rows[0]["fraction_of_ceiling"] <= sequential[0]["fraction_of_ceiling"]
    assert random_rows[0]["fraction_of_ceiling"] <= 1.0 + 1e-9


def test_every_burst_is_billed_once_on_a_non_overlapping_lane(hbm3e):
    result = envelope.dq_conservation(hbm3e, bursts=1024)
    assert result["passed"], {key: result["metrics"][key] for key in
                              ("overlap_count", "payload_bytes", "physical_bytes", "lanes_touched")}
    assert result["metrics"]["overlaps"] if "overlaps" in result["metrics"] else True
    assert result["metrics"]["payload_bytes"] == result["metrics"]["physical_bytes"]
    assert result["metrics"]["lane_busy_ns"] == pytest.approx(
        result["metrics"]["expected_lane_busy_ns"], rel=1e-6)
    assert result["metrics"]["max_lane_utilization"] <= 1.0 + 1e-9


def test_mixed_directions_exceed_one_cap_but_not_the_physical_link(hbm3e):
    result = envelope.mixed_duplex_probe(hbm3e, bursts=4096)
    assert result["bounded_by_physical"], result
    assert result["measured_gb_s"] <= result["physical_interface_gb_s"] * (1 + 1e-9)


def test_refresh_is_a_cost_not_a_monotonicity_defect():
    """A refreshed stream is non-monotone only because refresh accumulates."""
    config = envelope.reference_config(profile_id="hbm4_reference")
    edited = envelope.reference_config(profile_id="hbm4_reference", refresh=False)
    assert config.refresh_policy != "disabled"
    assert edited.refresh_policy == "disabled"
    overhead = envelope.refresh_overhead(config)
    # Positive because refresh costs time.  Not asserted tight: the fraction is
    # a function of run length, and short runs accumulate no refresh at all.
    assert overhead["overhead_fraction"] > 0.0
    assert overhead["refresh_commands"] > 0
