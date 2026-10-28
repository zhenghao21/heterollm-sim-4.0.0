"""Search control tests use tiny models and synthetic timing, not paper fits."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import exploring_hbf_search as search
from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.paper_inference import PaperParallelism, enumerate_parallelism
from heterollm_sim.paper_models import PaperModelSpec, PaperPrecision


MODEL = PaperModelSpec("tiny", "Tiny search control", 2, 32, 64, 4, 2, 8, 64, 10000, False)
WORKLOAD = SimpleNamespace(input_tokens=20, output_tokens=17)


def _fake_search(monkeypatch, *, strategies=None, capacities=None, slopes=None, failure=None):
    strategies = strategies or (PaperParallelism(),)
    capacities = capacities or {}
    slopes = slopes or {}
    initialized = []

    def key(parallel):
        return parallel.dp, parallel.pp, parallel.tp, parallel.ep, parallel.microbatches

    def capacity(model, hardware, parallel, batch, input_tokens, output_tokens, assumptions, precision):
        limit = capacities.get(key(parallel), 24)
        maximum = input_tokens + output_tokens - 1
        coefficient = model.kv_bytes_per_token(tp=parallel.tp) * maximum
        weights = 100
        return {"accepted": batch <= limit, "reason": "synthetic private memory capacity",
                "stages": [{"stage": 0, "resident_weight_bytes_per_gpu": weights,
                            "device_capacity_bytes": weights + coefficient * limit // parallel.ep,
                            "active_and_prepared_kv_bytes_per_gpu": coefficient * batch // parallel.ep,
                            "workspace_peak_bytes_per_gpu": batch // parallel.ep * 64,
                            "workspace_capacity_bytes": 1_000_000}]}

    class Engine:
        def __init__(self, model, hardware, parallel, *, batch_size, **kwargs):
            self.parallel, self.batch = parallel, batch_size
            initialized.append((key(parallel), batch_size))

        def execute_decode(self, requests, pending, ready):
            assert not pending and ready == 0
            assert len(requests) == self.batch
            assert [row.slot_id for row in requests] == list(range(self.batch))
            assert {row.context_tokens for row in requests} == {28}
            assert {row.generated_decode_tokens for row in requests} == {8}
            if failure and failure(self.parallel, self.batch):
                raise RuntimeError("synthetic executor fault")
            slope = slopes.get(key(self.parallel), 1_000_000.)
            return SimpleNamespace(end_ns=2_000_000. + self.batch * slope)

    monkeypatch.setattr(search, "enumerate_parallelism", lambda model, count: strategies)
    monkeypatch.setattr(search, "_capacity_probe", capacity)
    monkeypatch.setattr(search, "PaperInferenceEngine", Engine)
    return initialized


def test_capacity_binary_search_then_real_event_slo_screen(monkeypatch):
    initialized = _fake_search(monkeypatch)
    rows = search.search_candidates(MODEL, None, 1, WORKLOAD, slo_seconds=.015)
    row = rows[0]
    assert row["capacity_max_batch"] == 24
    assert row["capacity_upper_bound_batch"] >= 24
    assert not row["capacity_next_rejected"]["accepted"]
    assert row["candidate_batch"] == 13
    assert row["screen_tpot_seconds"] == .015
    assert row["status"] == "screened_candidate"
    assert row["screen_per_gpu_tokens_s"] == pytest.approx(13 / .015)
    assert row["final_closed_loop_validated"] is False
    assert "without_PD" in row["screening_scope"]
    assert len(initialized) == len({batch for _, batch in initialized})
    assert len(initialized) <= 8
    assert any(not trial["slo_accepted"] for trial in row["trials"])


def test_parallelism_enumeration_keeps_intermediate_legal_microbatch_degrees():
    rows = enumerate_parallelism(MODEL, 8)
    micro = {row.microbatches for row in rows if row.pp == 1}
    assert micro == {1}
    assert {1, 2}.issubset({row.microbatches for row in rows if row.pp == 2})


def test_every_strategy_is_kept_and_ranked_by_per_gpu_rate(monkeypatch):
    strategies = (PaperParallelism(dp=2), PaperParallelism(tp=2), PaperParallelism(pp=2))
    capacities = {(2, 1, 1, 1, 1): 4, (1, 1, 2, 1, 1): 24, (1, 2, 1, 1, 1): 0}
    slopes = {(2, 1, 1, 1, 1): 5_000_000., (1, 1, 2, 1, 1): 500_000.}
    _fake_search(monkeypatch, strategies=strategies, capacities=capacities, slopes=slopes)
    rows = search.search_candidates(MODEL, None, 2, WORKLOAD, slo_seconds=.015)
    assert len(rows) == 3
    assert rows[0]["parallelism"]["tp"] == 2
    assert rows[1]["parallelism"]["dp"] == 2
    assert rows[1]["screen_cluster_tokens_s"] == rows[1]["screen_per_gpu_tokens_s"] * 2
    assert rows[2]["status"] == "capacity_rejected"
    assert rows[2]["trials"] == []
    assert "private memory" in rows[2]["reason"]


def test_ep_and_microbatch_divisibility_is_preserved(monkeypatch):
    model = replace(MODEL, routed_experts=4, expert_intermediate_size=32, top_k=1,
                    shared_experts=1, moe_layer_step=2)
    initialized = _fake_search(monkeypatch, strategies=(PaperParallelism(ep=2, microbatches=2),))
    row = search.search_candidates(model, None, 2, WORKLOAD, slo_seconds=.015)[0]
    assert row["batch_quantum"] == 4 and row["candidate_batch"] == 12
    assert row["screen_per_gpu_tokens_s"] == pytest.approx(12 / (.014 * 2))
    assert all(batch % 4 == 0 for _, batch in initialized)
    assert all(batch % 4 == 0 for batch in row["capacity_checked_batches"])


def test_minimum_batch_slo_rejection_is_distinct_from_capacity(monkeypatch):
    _fake_search(monkeypatch)
    row = search.search_candidates(MODEL, None, 1, WORKLOAD, slo_seconds=.002)[0]
    assert row["status"] == "slo_rejected"
    assert row["capacity_max_batch"] == 24 and row["candidate_batch"] == 0
    assert len(row["trials"]) == 1 and row["trials"][0]["batch_size"] == 1


def test_execution_fault_is_recorded_and_not_searched_around(monkeypatch):
    initialized = _fake_search(monkeypatch, failure=lambda parallel, batch: batch == 1)
    row = search.search_candidates(MODEL, None, 1, WORKLOAD)[0]
    assert row["status"] == "execution_failed"
    assert row["failure_phase"] == "decode_screening"
    assert row["reason"] == "RuntimeError: synthetic executor fault"
    assert initialized == [((1, 1, 1, 1, 1), 1)]
    assert row["candidate_batch"] == 0


def test_offline_is_24_hours_divided_by_total_output_not_decode_tokens(monkeypatch):
    _fake_search(monkeypatch)
    row = search.search_candidates(MODEL, None, 1, WORKLOAD, slo_seconds="offline")[0]
    assert row["slo_seconds"] == 86400 / 17
    assert row["slo_basis"] == "offline_24_hours_divided_by_L_OUT"
    assert row["candidate_batch"] == row["capacity_max_batch"] == 24


def test_capacity_probe_uses_real_capacity_method_without_constructing_actors(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("capacity probe must not initialize engine actors")
    monkeypatch.setattr(search.PaperInferenceEngine, "__init__", forbidden)
    result = search._capacity_probe(MODEL, hardware_by_id("HBM4"), PaperParallelism(),
        2, 20, 17, search.PaperExecutionAssumptions(), PaperPrecision())
    assert result["accepted"]
    assert result["prepared_slots_per_replica"] == 2
    assert result["stages"][0]["active_and_prepared_kv_bytes_per_gpu"] > 0


def test_capacity_upper_bound_does_not_clip_the_staggered_paged_pool():
    model = replace(MODEL, attention_chunk_size=4, full_attention_interval=2)
    maximum = 20 + 17 - 1
    unit = 2 * model.local_kv_head_count() * model.head_dim * 2
    kv = unit * (maximum + 4)
    cap = {"stages": [{"stage": 0, "resident_weight_bytes_per_gpu": 100,
                       "device_capacity_bytes": 100 + kv * 7,
                       "workspace_peak_bytes_per_gpu": 64,
                       "workspace_capacity_bytes": 1_000_000}]}
    bound = search._capacity_upper_bound(model, PaperParallelism(), 20, 17, PaperPrecision(), cap)
    assert bound > 7  # Seven is only the overly conservative full-length reservation bound.
    assert bound * unit * (21+1) <= cap["stages"][0]["device_capacity_bytes"]-100


def test_tiny_model_runs_actual_event_screen_without_a_serving_or_formal_scan(monkeypatch):
    # Bound this unit test to four tiny requests. The actual capacity method
    # and actual inference event backend execute; no production model or
    # closed-loop matrix is launched here.
    monkeypatch.setattr(search, "enumerate_parallelism", lambda model, count: (PaperParallelism(),))
    monkeypatch.setattr(search, "_capacity_upper_bound", lambda *args: 4)
    rows = search.search_candidates(MODEL, hardware_by_id("HBM4"), 1, WORKLOAD)
    row = rows[0]
    assert row["status"] == "screened_candidate", row["reason"]
    assert row["candidate_batch"] == 4 and 0 < row["screen_tpot_seconds"] < .1
    assert row["representative_context_tokens"] == 28
    assert row["final_closed_loop_validated"] is False


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "day"])
def test_invalid_slo_is_rejected(value):
    with pytest.raises(ValueError):
        search.search_candidates(MODEL, None, 1, WORKLOAD, slo_seconds=value)


def test_context_limit_and_short_output_cannot_be_silently_changed():
    with pytest.raises(ValueError, match="context"):
        search.search_candidates(MODEL, None, 1, SimpleNamespace(input_tokens=9999, output_tokens=17))
    with pytest.raises(ValueError, match="three"):
        search.search_candidates(MODEL, None, 1, SimpleNamespace(input_tokens=20, output_tokens=2))
