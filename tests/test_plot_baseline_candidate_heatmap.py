import json
from pathlib import Path

import pytest

from tools.plot_baseline_candidate_heatmap import generate


def _score(path: Path, values: dict[str, float], *, cell_ids=("a", "b")) -> None:
    cells = []
    for index, cell_id in enumerate(cell_ids):
        cells.append({"cell_id": cell_id, "model_key": "model", "metrics": {
            metric: {"absolute_percentage_error_pct": value + index}
            for metric, value in values.items()
        }})
    path.write_text(json.dumps({"schema": "stable-native-simulation-errors/v2", "cells": cells}), encoding="utf-8")


def test_baseline_only_heatmap_marks_candidate_absent(tmp_path):
    baseline = tmp_path / "baseline.json"
    _score(baseline, {"engine_ttft_ms": 10, "engine_tpot_ms": 30, "engine_e2e_ms": 50})
    manifest = generate(baseline, tmp_path / "heatmap")
    assert manifest["candidate_status"] == "absent"
    assert manifest["rows"] == 2
    assert len(manifest["artifacts"]) == 3
    assert all(Path(ref["path"]).is_file() for ref in manifest["artifacts"])
    assert json.loads((tmp_path / "heatmap" / "manifest.json").read_text(encoding="utf-8"))["candidate_status"] == "absent"


def test_candidate_delta_is_recorded(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _score(baseline, {"engine_ttft_ms": 20, "engine_tpot_ms": 20, "engine_e2e_ms": 20})
    _score(candidate, {"engine_ttft_ms": 10, "engine_tpot_ms": 25, "engine_e2e_ms": 30})
    manifest = generate(baseline, tmp_path / "heatmap", candidate)
    assert manifest["candidate_status"] == "present"
    assert manifest["candidate_score_ref"]["path"] == str(candidate.resolve())


def test_candidate_cell_mismatch_is_rejected(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _score(baseline, {"engine_ttft_ms": 20, "engine_tpot_ms": 20, "engine_e2e_ms": 20})
    _score(candidate, {"engine_ttft_ms": 20, "engine_tpot_ms": 20, "engine_e2e_ms": 20}, cell_ids=("a", "c"))
    with pytest.raises(ValueError, match="cell set"):
        generate(baseline, tmp_path / "heatmap", candidate)


def test_bad_schema_is_rejected(tmp_path):
    score = tmp_path / "bad.json"
    score.write_text(json.dumps({"schema": "wrong", "cells": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        generate(score, tmp_path / "heatmap")
