"""Continue a search from its saved rows without rerunning them."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exploring_hbf_native as run


LAYOUT = {"dp": 1, "tp": 8, "pp": 1, "ep": 1}


def _tpot_ms(batch):
    return 20. + .035 * batch


def _rate(batch):
    return batch / 8 / (_tpot_ms(batch) / 1e3)


def _row(case, batch):
    params = run.parameters(case, LAYOUT, batch, "fast")
    tpot_ms = _tpot_ms(batch)
    return {"candidate_id": run.params_candidate_id(params), "parameters": params,
        "backend": run.BACKEND, "runner_contract": run.FAST_CONTRACT,
        "status": "completed" if tpot_ms <= 100 else "slo_rejected",
        "metrics": {"tokens_per_second_per_gpu": _rate(batch),
            "cluster_tokens_per_second": _rate(batch) * 8, "mean_complete_request_tpot_ns": tpot_ms * 1e6},
        "diagnostics": {"metric_bounds": {"mean_complete_request_tpot_ns": {
            "estimate": tpot_ms * 1e6, "lower": tpot_ms * 1e6, "upper": tpot_ms * 1e6}}},
        "wall_seconds": 1., "finished_at": "2026-10-26T10:00:00Z"}


def _screen(capacity=5000):
    return {"layouts": [{"layout": LAYOUT, "quantum": 1,
        "bypass": {"candidate_batch": capacity, "capacity_max_batch": capacity,
            "tpot_ns": 90e6, "per_gpu_tokens_s": 100., "tpot_line_ns": (1e6, 1e4)}}],
        "skipped_layouts": []}


def _finish_search(directory, case):
    """Save what the search asks for, as its workers would, until it asks for nothing."""
    ran = []
    while True:
        plan, _planned, _point = run.search_case(directory, case, _screen())
        if not plan["next"]:
            return ran
        for layout, batch in plan["next"]:
            assert layout == LAYOUT
            assert batch not in ran, f"B{batch} was asked for again after it was saved"
            run.save_candidate(directory, _row(case, batch))
            ran.append(batch)


def test_a_finished_search_replays_from_its_rows_and_keeps_them_unchanged(tmp_path):
    case = run.case_matrix()[0]
    ran = _finish_search(tmp_path, case)
    saved = {path.name: path.read_bytes() for path in (tmp_path / "candidates").iterdir()}
    plan, planned, point = run.search_case(tmp_path, case, _screen())
    assert not plan["next"]
    assert {batch for _layout, batch in planned} == set(ran)
    passing = [batch for batch in ran if _tpot_ms(batch) <= 100]
    assert point["selected"]["parameters"]["batch_size"] == max(passing, key=_rate)
    assert {path.name: path.read_bytes() for path in (tmp_path / "candidates").iterdir()} == saved


def test_a_search_resumes_where_its_rows_end_without_scheduling_them_again(tmp_path):
    case = run.case_matrix()[0]
    ran = _finish_search(tmp_path / "whole", case)
    assert len(ran) >= 3
    resumed = tmp_path / "resumed"
    for batch in ran[:-1]:
        run.save_candidate(resumed, _row(case, batch))
    plan, planned, _point = run.search_case(resumed, case, _screen())
    assert plan["next"] == [(LAYOUT, ran[-1])]
    assert {batch for _layout, batch in planned} == set(ran)


def test_a_saved_row_of_another_backend_is_refused_instead_of_relabeled(tmp_path):
    case = run.case_matrix()[0]
    row = _row(case, run.search_plan.default_first_batch(_screen()["layouts"][0]))
    row["backend"] = "another_backend"
    run.save_candidate(tmp_path, row)
    with pytest.raises(RuntimeError, match="parameters/backend differ"):
        run.search_case(tmp_path, case, _screen())
