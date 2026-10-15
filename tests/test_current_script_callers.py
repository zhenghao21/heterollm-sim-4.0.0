"""Experiment aggregation and report readers require the current data shapes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hbf_hbm_bank_group_study as bank_study
import hbf_hbm_experiments as experiments
import hbf_hbm_report as reports
import hbf_hbm_session_experiments as sessions


def _main_arguments(monkeypatch, directory):
    monkeypatch.setattr(sys, "argv", ["hbf_hbm_experiments.py", "--groups", "4", "--output-dir", str(directory)])
    monkeypatch.setattr(experiments, "render", lambda report: "current report")


def test_current_session_producer_reaches_combined_results_as_one_group(monkeypatch, tmp_path):
    # Exercise the real group producer while omitting unrelated physical work.
    monkeypatch.setattr(sessions.SessionExperiment, "run", lambda self: {"checks": {"fixture": True}})
    _main_arguments(monkeypatch, tmp_path)
    assert experiments.main() == 0
    combined = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    group, = combined["groups"]
    assert group["id"] == "04_hot_cold_sessions"
    assert len(group["rows"]) == 30
    assert group["status"] == "passed"
    assert "groups" not in group


def test_session_entrypoint_does_not_unwrap_the_retired_groups_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "run_session_experiments", lambda: {
        "groups": [{"id": "04_hot_cold_sessions", "status": "passed", "rows": []}]})
    _main_arguments(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="id"):
        experiments.main()


def _bank_point(monkeypatch):
    config = {"kind": "HBM", "generation": "HBM3E", "bank_groups_per_rank": 4,
              "banks_per_group": 16, "metadata": {"geometry_status": "standard_organization"},
              **{name: 0.0 for name in reports._BANK_GROUP_TIMING_FIELDS}}

    def run_case(architecture, *, row_id, result_observer, **parameters):
        result_observer(SimpleNamespace(serving=None))
        return {"id": row_id, "executed_serving": True,
                "core_execution_audit": {"executed_physical_configurations": {
                    "memory": {"physical_config": config},
                    "flash": {"physical_config": {"kind": "HBF"}},
                }}}

    monkeypatch.setattr(bank_study, "run_case", run_case)
    monkeypatch.setattr(bank_study, "decode_measurement", lambda serving: {"decode_tokens": 2})
    monkeypatch.setattr(bank_study, "core_checks", lambda row: {"fixture": True})
    return bank_study.run_point((bank_study.PROFILES[1], 16, bank_study.ARCHITECTURES[0], 3))


def test_current_bank_group_producer_supplies_report_parameters(monkeypatch):
    row = _bank_point(monkeypatch)
    variant, = reports.bank_group_variants({"hardware_profiles": [row["hardware_profile"]], "rows": [row]})
    record, = variant["records"]
    assert record["owner"] == "memory"
    assert record["config"] == row["executed_hbm_parameters"]["memory"]
    assert "4×16" in variant["label_zh"]


def test_bank_group_report_does_not_reconstruct_a_missing_current_field(monkeypatch):
    row = deepcopy(_bank_point(monkeypatch))
    del row["executed_hbm_parameters"]
    with pytest.raises(KeyError, match="executed_hbm_parameters"):
        reports.bank_group_variants({"rows": [row]})


def test_bank_group_report_preserves_explicitly_empty_parameters(monkeypatch):
    row = _bank_point(monkeypatch)
    row["executed_hbm_parameters"] = {}
    variant, = reports.bank_group_variants({"rows": [row]})
    assert variant["records"] == []
    assert "配置未记录" in variant["label_zh"]
