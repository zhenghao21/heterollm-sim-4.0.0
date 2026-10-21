"""Fixed-lookahead capacity refinement never executes or fractionally cuts a batch."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import exploring_hbf_search as search
from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.paper_inference import PaperExecutionAssumptions, PaperInferenceEngine, PaperParallelism
from heterollm_sim.paper_models import LLAMA3_405B, LLAMA4_MAVERICK


def artificial_capacity(monkeypatch, limit):
    seen = []

    def probe(model, hardware, parallel, batch, input_tokens, output_tokens, assumptions, precision):
        seen.append((batch, assumptions.prepared_lookahead))
        return {"accepted": batch <= limit, "batch": batch,
                "lookahead": assumptions.prepared_lookahead,
                "reason": "synthetic one-dimensional private-capacity boundary"}

    monkeypatch.setattr(search, "_capacity_probe", probe)
    return seen


def refine(**changes):
    arguments = dict(upper_batch=1555, input_tokens=1660, output_tokens=373,
                     assumptions=PaperExecutionAssumptions(prepared_lookahead=5))
    arguments.update(changes)
    return search.maximum_capacity_batch(LLAMA3_405B, hardware_by_id("HBM4"),
                                        PaperParallelism(tp=8), **arguments)


def test_two_request_capacity_excess_does_not_discard_fifteen_percent(monkeypatch):
    seen = artificial_capacity(monkeypatch, 1553)
    result = refine()
    assert result["batch_size"] == 1553
    assert result["next_rejected_batch"] == 1554
    assert result["capacity_at_batch"] == {
        "accepted": True, "batch": 1553, "lookahead": 5,
        "reason": "synthetic one-dimensional private-capacity boundary"}
    assert result["next_rejected_capacity"]["accepted"] is False
    assert result["prepared_lookahead"] == 5
    assert all(lookahead == 5 for _, lookahead in seen)
    assert len(seen) == len({batch for batch, _ in seen})
    assert len(seen) <= 14


def test_ep_microbatch_quantum_is_respected_without_fractional_layout(monkeypatch):
    seen = artificial_capacity(monkeypatch, 1553)
    result = search.maximum_capacity_batch(LLAMA4_MAVERICK, hardware_by_id("HBF+"),
        PaperParallelism(ep=2, microbatches=2), upper_batch=1555,
        input_tokens=1660, output_tokens=373,
        assumptions=PaperExecutionAssumptions(prepared_lookahead=9))
    assert result["batch_size"] == result["rounded_upper_batch"] == 1552
    assert result["batch_quantum"] == 4
    assert result["next_rejected_batch"] is None
    assert all(batch % 4 == 0 and lookahead == 9 for batch, lookahead in seen)


def test_minimum_capacity_rejection_returns_zero_with_real_rejection(monkeypatch):
    artificial_capacity(monkeypatch, 0)
    result = refine()
    assert result["batch_size"] == 0
    assert result["capacity_at_batch"] is None
    assert result["next_rejected_batch"] == 1
    assert result["capacity_at_minimum"]["accepted"] is False
    assert result["checked_batches"] == [1]


def test_large_upper_boundary_uses_logarithmic_probes(monkeypatch):
    seen = artificial_capacity(monkeypatch, 123_456_789)
    result = refine(upper_batch=1_000_000_000_000)
    assert result["batch_size"] == 123_456_789
    assert result["next_rejected_batch"] == 123_456_790
    assert len(seen) <= 43


def test_upper_smaller_than_quantum_does_not_probe_an_illegal_batch(monkeypatch):
    seen = artificial_capacity(monkeypatch, 100)
    result = search.maximum_capacity_batch(LLAMA4_MAVERICK, hardware_by_id("HBF+"),
        PaperParallelism(ep=2, microbatches=2), upper_batch=3,
        input_tokens=1660, output_tokens=373,
        assumptions=PaperExecutionAssumptions())
    assert result["batch_size"] == 0
    assert not seen


def test_real_paged_capacity_refinement_never_constructs_or_executes_actors(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("capacity-only refinement cannot execute actors")

    monkeypatch.setattr(PaperInferenceEngine, "__init__", forbidden)
    monkeypatch.setattr(PaperInferenceEngine, "execute_decode", forbidden)
    monkeypatch.setattr(PaperInferenceEngine, "run", forbidden)
    result = refine(upper_batch=1600)
    assert 0 < result["batch_size"] <= 1600
    assert result["capacity_at_batch"]["accepted"]
    if result["next_rejected_capacity"] is not None:
        assert not result["next_rejected_capacity"]["accepted"]
    assert result["execution_scope"] == "capacity_only_no_actors_or_timing_execution"


def test_probe_failure_is_not_reinterpreted_as_capacity_rejection(monkeypatch):
    def broken(*args):
        raise RuntimeError("real capacity implementation bug")

    monkeypatch.setattr(search, "_capacity_probe", broken)
    with pytest.raises(RuntimeError, match="implementation bug"):
        refine()


@pytest.mark.parametrize("changes", [{"upper_batch": 0}, {"upper_batch": 1.5},
    {"upper_batch": True}, {"input_tokens": 0}, {"output_tokens": 1},
    {"assumptions": None}, {"input_tokens": 131_072, "output_tokens": 3}])
def test_invalid_refinement_bounds_and_workload_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        refine(**changes)
