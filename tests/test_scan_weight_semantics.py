"""Explicit result migration prevents old physical weights from entering new pairs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scan():
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan.py"
    spec = importlib.util.spec_from_file_location("scan_weight_semantics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_requires_explicit_migration_before_starting_new_semantics(scan, tmp_path):
    manifest = scan.initialize(tmp_path)
    assert manifest["physical_weight_semantics"] == scan.PHYSICAL_WEIGHT_SEMANTICS
    del manifest["physical_weight_semantics"]
    save(tmp_path / "manifest.json", manifest)
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="parameters differ"):
        scan.initialize(tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("tag", (None, "legacy_duplicated_weight_buffers"))
def test_report_rejects_stale_active_weights_instead_of_plotting_them(scan, tmp_path, tag):
    import b200_hbm_hbf_scan_report as report
    manifest = scan.initialize(tmp_path)
    point = manifest["points"][0]
    row = {**point, "status": "forecast_completed", "ttft_ns": 20., "tpot_ns": 2.}
    if tag is not None:
        row["physical_weight_semantics"] = tag
    save(tmp_path / "points" / (point["point_id"] + ".json"), row)
    with pytest.raises(ValueError, match="physical weight semantics"):
        report.write_report(tmp_path)
    assert not (tmp_path / "results.json").exists()


def test_retry_gate_accepts_new_hbm_independent_budget_but_never_legacy_rows(scan):
    identifiers = ("hbm", "hbf")
    rows = [{"point_id": identifier, "status": "forecast_completed",
        "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS,
        "approximation": {"paired_budget_decision": {"accepted": True,
            "strategy": "independent_symmetric_sufficient_bound" if identifier == "hbm"
                        else "matched_directional_pair"}}} for identifier in identifiers]
    assert scan.retry_group_counts(rows, identifiers)["retry_group_accepted"] is True
    del rows[0]["physical_weight_semantics"]
    assert scan.retry_group_counts(rows, identifiers)["retry_group_accepted"] is False


def test_archived_semantic_revalidation_reuses_original_priority_gate_and_only_reviewed_rows(scan, tmp_path, monkeypatch):
    import b200_hbm_hbf_scan_report as report
    manifest = scan.initialize(tmp_path)
    points = manifest["points"]
    original = points[:28]
    reusable = {point["point_id"] for point in original
                if point["model_id"] == "qwen2_5-7b" and point["architecture"] == "HBM"}
    affected = [point for point in original if point["point_id"] not in reusable]
    first_pair = [point["point_id"] for point in affected
                  if point["model_id"] == "qwen2_5-1_5b" and point["prompt_tokens"] == 262144]
    assert [point["architecture"] for point in affected if point["point_id"] in first_pair] == ["HBM", "HBF"]
    retry = tuple(first_pair + [point["point_id"] for point in affected if point["point_id"] not in first_pair])
    old = {}
    for point in original:
        identifier = point["point_id"]
        row = {**point, "status": "capacity_rejected" if identifier == first_pair[1]
               else "forecast_completed", "ttft_ns": 50., "tpot_ns": 5.}
        old[identifier] = deepcopy(row)
        if identifier in reusable:
            # This explicit, reviewed migration is performed by the operator,
            # never inferred by the scheduler from a model name or old timing.
            row.update(physical_weight_semantics=scan.PHYSICAL_WEIGHT_SEMANTICS,
                semantic_reuse_review="untied_7b_hbm_physically_unchanged")
            save(tmp_path / "points" / (identifier + ".json"), row)
        else:
            save(tmp_path / "points_history" / (identifier + "__old.json"), row)
    assert len(retry) == 25 and len(reusable) == 3
    reviewed_bytes = {identifier: (tmp_path / "points" / (identifier + ".json")).read_bytes()
                      for identifier in reusable}
    monkeypatch.setattr(report, "write_report", lambda *_args: None)
    monkeypatch.setattr(report, "refresh_markdown", lambda *_args: None)
    monkeypatch.setattr(scan.time, "sleep", lambda _seconds: None)
    launches = []

    def popen(command, **_kwargs):
        identifier = command[command.index("--point-id") + 1]
        point = next(point for point in points if point["point_id"] == identifier)
        if point["architecture"] == "HBF":
            hbm_id = scan.point_id(point["model_id"], "HBM", point["prompt_tokens"], point["output_tokens"])
            hbm = read(tmp_path / "points" / (hbm_id + ".json"))
            assert hbm["physical_weight_semantics"] == scan.PHYSICAL_WEIGHT_SEMANTICS
        launches.append(identifier)
        save(tmp_path / "points" / (identifier + ".json"), {**point,
            "status": "forecast_completed", "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS,
            "approximation": {"paired_budget_decision": {"accepted": True}}})
        return SimpleNamespace(pid=500000 + len(launches), poll=lambda: 0)

    monkeypatch.setattr(scan.subprocess, "Popen", popen)
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 0
    assert launches[:25] == list(retry)
    assert len(launches) == len(set(launches)) == 69
    assert not reusable.intersection(launches)
    assert read(tmp_path / "status.json")["retry_group_accepted"] is True
    assert read(tmp_path / "status.json")["completed"] == 72
    assert len(list((tmp_path / "points_history").glob("*.json"))) == 25
    for path in (tmp_path / "points_history").glob("*.json"):
        assert read(path) == old[read(path)["point_id"]]
    for identifier, value in reviewed_bytes.items():
        assert (tmp_path / "points" / (identifier + ".json")).read_bytes() == value
    launches.clear()
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 0 and not launches


def test_queue_does_not_automatically_relabel_or_reuse_legacy_success(scan, tmp_path, monkeypatch):
    import b200_hbm_hbf_scan_report as report
    manifest = scan.initialize(tmp_path)
    point = manifest["points"][0]
    path = tmp_path / "points" / (point["point_id"] + ".json")
    save(path, {**point, "status": "forecast_completed", "ttft_ns": 2., "tpot_ns": 1.})
    before = path.read_bytes()
    monkeypatch.setattr(report, "write_report", lambda *_args: pytest.fail("stale data reached report"))
    monkeypatch.setattr(scan.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("started simulation"))
    with pytest.raises(ValueError, match="physical weight semantics"):
        scan.run_queue(tmp_path)
    assert path.read_bytes() == before


def test_stale_active_point_blocks_queue_before_any_retry_archive(scan, tmp_path, monkeypatch):
    manifest = scan.initialize(tmp_path)
    selected, stale = manifest["points"][:2]
    path = tmp_path / "points" / (selected["point_id"] + ".json")
    save(path, {**selected, "status": "forecast_rejected",
        "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS})
    save(tmp_path / "points" / (stale["point_id"] + ".json"),
         {**stale, "status": "forecast_completed"})
    before = path.read_bytes()
    monkeypatch.setattr(scan.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("started simulation"))
    with pytest.raises(ValueError, match="physical weight semantics"):
        scan.run_queue(tmp_path, retry_point_ids=(selected["point_id"],))
    assert path.read_bytes() == before and not (tmp_path / "points_history").exists()
