"""Continue search from all verified scientific history without rerunning it."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exploring_hbf_native as run


LAYOUT = {"dp": 1, "tp": 8, "pp": 1, "ep": 1}


def _row(case, batch, tpot_ms, *, wall=1.):
    params = run.parameters(case, LAYOUT, batch, "fast")
    return {"candidate_id": run.params_candidate_id(params), "parameters": params,
        "backend": run.BACKEND, "runner_contract": run.FAST_CONTRACT,
        "status": "completed" if tpot_ms <= 100 else "slo_rejected",
        "metrics": {"tokens_per_second_per_gpu": float(batch),
            "cluster_tokens_per_second": float(batch * 8), "mean_complete_request_tpot_ns": tpot_ms * 1e6},
        "diagnostics": {"metric_bounds": {"mean_complete_request_tpot_ns": {
            "estimate": tpot_ms * 1e6, "lower": tpot_ms * 1e6, "upper": tpot_ms * 1e6}}},
        "wall_seconds": wall, "finished_at": "2026-10-26T10:00:00Z"}


def _screen(capacity=5000):
    return {"layouts": [{"layout": LAYOUT, "quantum": 1,
        "bypass": {"candidate_batch": capacity, "capacity_max_batch": capacity,
            "tpot_ns": 90e6, "per_gpu_tokens_s": 100., "tpot_line_ns": (1e6, 1e4)}}],
        "skipped_layouts": []}


def test_complete_history_preserves_the_larger_passing_batch_under_new_wall_projection(tmp_path):
    case = run.case_matrix()[0]
    rows = [_row(case, 1000, 50, wall=900), _row(case, 2292, 90, wall=2061.9)]
    for row in rows:
        run.save_candidate(tmp_path, row)
    original = {row["candidate_id"]: (tmp_path / "candidates" / (row["candidate_id"] + ".json")).read_bytes()
                for row in rows}
    plan, planned, point = run.search_case(tmp_path, case, _screen())
    assert not plan["next"]
    assert set(point["candidate_ids"]) == set(original)
    assert point["selected"]["parameters"]["batch_size"] == 2292
    assert set(plan["historical_candidate_ids"]) == set(original)
    assert {(tuple(layout.values()), batch) for layout, batch in planned} == {
        (tuple(LAYOUT.values()), 1000), (tuple(LAYOUT.values()), 2292)}
    for identity, data in original.items():
        assert (tmp_path / "candidates" / (identity + ".json")).read_bytes() == data


def test_refinement_keeps_old_acceptance_and_negative_evidence_without_scheduling_old_points(tmp_path):
    case = run.case_matrix()[0]
    rows = [_row(case, 100, 30), _row(case, 200, 60), _row(case, 300, 99), _row(case, 310, 102)]
    for row in rows:
        run.save_candidate(tmp_path, row)
    plan, planned, point = run.search_case(tmp_path, case, _screen())
    assert len(plan["next"]) == 1
    layout, batch = plan["next"][0]
    assert layout == LAYOUT and 300 < batch < 310
    assert batch not in {100, 200, 300, 310}
    assert set(point["candidate_ids"]) == {row["candidate_id"] for row in rows}
    assert {n for _, n in planned} == {100, 200, 300, 310, batch}
    assert {row["batch"] for row in plan["layouts"][0]["runs"]} == {100, 200, 300, 310}


def test_search_history_rejects_foreign_backend_instead_of_relabeling_it(tmp_path):
    case = run.case_matrix()[0]
    row = _row(case, 100, 30)
    row["backend"] = "another_backend"
    run.save_candidate(tmp_path, row)
    with pytest.raises(RuntimeError, match="parameters/backend differ"):
        run._search_history(tmp_path, [case])
