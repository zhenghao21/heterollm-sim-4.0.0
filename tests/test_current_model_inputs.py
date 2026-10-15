"""Current model identity and context fields reach the study plot unchanged."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hbf_hbm_experiments as experiments
import hbf_hbm_serving_study as study
import plot_hbf_hbm_results as plots


def _study_rows(monkeypatch):
    def recorded_row(scenario, row_id, *, placement, result_observer=None):
        return {"id": row_id, "mode": "full_operator_serving", "model": scenario.model.name,
                "placement": placement, "node_tps": 100.0, "prompt_tokens": 2048,
                "output_tokens_per_request": 4,
                "resolved_placement": {"tensor_to_component": {}, "op_to_component": {}}}

    monkeypatch.setattr(experiments, "serving_row", recorded_row)
    return [study.run_case(placement, row_id="model/" + placement,
                           model_id="llama3_1-8b", prompt=2048, output=4)
            for placement in (plots.A, plots.B)]


def test_current_study_producer_plots_model_id_separately_from_display_name(monkeypatch, tmp_path):
    rows = _study_rows(monkeypatch)
    assert rows[0]["model"] != rows[0]["parameters"]["model_id"]

    def capture(fig, output, stem, **result):
        plots.plt.close(fig)
        return result

    monkeypatch.setattr(plots, "save", capture)
    plotted = plots.named_context_figure({"09_serving_study": {"rows": rows}}, tmp_path)
    observation, = plotted["observations"]
    assert observation["model_id"] == "llama3_1-8b"
    assert observation["B_over_A_tps"] == 1.0
    assert observation["context_extrapolation"] is None


def test_named_model_plot_does_not_treat_display_name_as_missing_model_id(monkeypatch, tmp_path):
    rows = _study_rows(monkeypatch)
    for row in rows:
        del row["parameters"]["model_id"]
    with pytest.raises(KeyError, match="model_id"):
        plots.named_context_figure({"09_serving_study": {"rows": rows}}, tmp_path)


def test_context_plot_requires_the_current_model_structure_field(monkeypatch):
    row = _study_rows(monkeypatch)[0]
    current = {"native_max_tokens": 131072, "accuracy_claim": False}
    row["model_structure"]["context_extrapolation"] = current
    assert plots.context_extrapolation(row) == current
    obsolete = deepcopy(row)
    del obsolete["model_structure"]["context_extrapolation"]
    obsolete["context_extrapolation"] = current
    with pytest.raises(KeyError, match="context_extrapolation"):
        plots.context_extrapolation(obsolete)
