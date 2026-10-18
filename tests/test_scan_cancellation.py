"""Persistent queue exclusions and takeover must not rerun physical work."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import b200_hbm_hbf_scan as scan
from b200_scan_plan import load_cancellations


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def plan(directory, points):
    rows = [{**point, "status": "cancelled", "reason": "user_cancelled_unstarted_long_output",
             "cancelled_at": "2026-10-18T15:45:00+00:00"} for point in points]
    save(directory / "queue_plan.json", {"schema": "b200-scan-plan/v1", "cancelled_points": rows})
    return rows


def test_cancellations_cannot_override_success_or_one_k(tmp_path):
    manifest = scan.initialize(tmp_path)
    long = next(point for point in manifest["points"] if point["output_tokens"] == 5120)
    plan(tmp_path, [long])
    assert list(load_cancellations(tmp_path, manifest)) == [long["point_id"]]
    save(tmp_path / "points" / (long["point_id"] + ".json"), {**long, "status": "forecast_completed"})
    with pytest.raises(ValueError, match="existing result"):
        load_cancellations(tmp_path, manifest)
    plan(tmp_path, [manifest["points"][0]])
    with pytest.raises(ValueError, match="long-output"):
        load_cancellations(tmp_path, manifest)


def test_cancelled_worker_does_not_enter_physical_runtime(tmp_path, monkeypatch):
    manifest = scan.initialize(tmp_path)
    point = next(point for point in manifest["points"] if point["output_tokens"] == 10240)
    plan(tmp_path, [point])
    import b200_hbm_hbf_scan_config as config
    monkeypatch.setattr(config, "build_scan_scenario", lambda *args: pytest.fail("cancelled simulation started"))
    assert scan.worker(tmp_path, point) == 0
    row = json.loads((tmp_path / "points" / (point["point_id"] + ".json")).read_text())
    assert row["status"] == "cancelled" and row["ttft_ns"] is row["tpot_ns"] is None


def test_resuming_full_manifest_never_launches_cancelled_points(tmp_path, monkeypatch):
    manifest = scan.initialize(tmp_path)
    plan(tmp_path, [point for point in manifest["points"] if point["output_tokens"] != 1024])
    import b200_hbm_hbf_scan_report as report
    monkeypatch.setattr(report, "write_report", lambda *args: None)
    monkeypatch.setattr(report, "refresh_markdown", lambda *args: None)
    launches = []
    def launch(command, **kwargs):
        identifier = command[command.index("--point-id") + 1]
        launches.append(identifier)
        point = next(point for point in manifest["points"] if point["point_id"] == identifier)
        assert point["output_tokens"] == 1024
        save(tmp_path / "points" / (identifier + ".json"), {**point,
             "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS, "status": "forecast_completed"})
        return SimpleNamespace(pid=123456, poll=lambda: 0)
    monkeypatch.setattr(scan.subprocess, "Popen", launch)
    assert scan.run_queue(tmp_path) == 0
    assert len(launches) == 24
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "completed"
    assert (status["completed"], status["cancelled"], status["planned_total"], status["pending"]) == (24, 48, 24, 0)
    launches.clear()
    assert scan.run_queue(tmp_path) == 0 and launches == []


@pytest.mark.parametrize("pause_after_current", (False, True))
def test_takeover_keeps_existing_worker_and_only_launches_remaining_point(tmp_path, monkeypatch, pause_after_current):
    manifest = scan.initialize(tmp_path)
    point = manifest["points"][0]
    other = manifest["points"][1]
    # All other points are saved successes, so the live worker must be adopted,
    # and only its next unrun 1K pair may be launched.
    for done in manifest["points"][2:]:
        save(tmp_path / "points" / (done["point_id"] + ".json"), {**done,
             "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS, "status": "forecast_completed"})
    save(tmp_path / "status.json", {"manager_pid": 100, "running": [{**point, "started_at": "original"}]})
    save(tmp_path / "progress" / (point["point_id"] + ".json"), {**point,
         "status": "running", "pid": 200, "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS})
    alive = {200}
    monkeypatch.setattr(scan, "process_alive", lambda pid: pid in alive)
    import b200_hbm_hbf_scan_report as report
    monkeypatch.setattr(report, "write_report", lambda *args: None)
    monkeypatch.setattr(report, "refresh_markdown", lambda *args: None)
    def finish(_delay):
        alive.clear()
        save(tmp_path / "points" / (point["point_id"] + ".json"), {**point,
             "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS, "status": "forecast_completed"})
    monkeypatch.setattr(scan.time, "sleep", finish)
    launches = []
    def launch(command, **kwargs):
        identifier = command[command.index("--point-id") + 1]
        launches.append(identifier)
        save(tmp_path / "points" / (identifier + ".json"), {**other,
             "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS, "status": "forecast_completed"})
        return SimpleNamespace(pid=300, poll=lambda: 0)
    monkeypatch.setattr(scan.subprocess, "Popen", launch)
    assert scan.run_queue(tmp_path, adopt_running=True, pause_after_current=pause_after_current) == int(pause_after_current)
    assert launches == ([] if pause_after_current else [other["point_id"]])
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["completed"] == (71 if pause_after_current else 72)
    if pause_after_current:
        assert status["status"] == "paused" and status["pending"] == 1
        assert (tmp_path / "stop.request").read_text() == "configuration_boundary"


def test_takeover_refuses_live_previous_manager(tmp_path, monkeypatch):
    manifest = scan.initialize(tmp_path)
    save(tmp_path / "status.json", {"manager_pid": 100, "running": []})
    monkeypatch.setattr(scan, "process_alive", lambda pid: pid == 100)
    with pytest.raises(ValueError, match="manager is still alive"):
        scan.existing_workers(tmp_path, manifest)


@pytest.mark.parametrize("progress_status", ("running", "forecast_completed"))
def test_takeover_counts_already_published_live_worker_once(tmp_path, monkeypatch, progress_status):
    manifest = scan.initialize(tmp_path)
    point = manifest["points"][0]
    # Publication and interpreter exit are separate: the adopted point can
    # already have a complete result while its original process remains alive.
    for done in manifest["points"]:
        save(tmp_path / "points" / (done["point_id"] + ".json"), {**done,
             "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS, "status": "forecast_completed"})
    save(tmp_path / "status.json", {"manager_pid": 100, "running": [{**point, "started_at": "original"}]})
    save(tmp_path / "progress" / (point["point_id"] + ".json"), {**point,
         "status": progress_status, "pid": 200, "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS})
    alive = {200}
    monkeypatch.setattr(scan, "process_alive", lambda pid: pid in alive)
    monkeypatch.setattr(scan.time, "sleep", lambda _delay: alive.clear())
    monkeypatch.setattr(scan.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("successful point relaunched"))
    import b200_hbm_hbf_scan_report as report
    monkeypatch.setattr(report, "write_report", lambda *args: None)
    monkeypatch.setattr(report, "refresh_markdown", lambda *args: None)

    assert scan.run_queue(tmp_path, adopt_running=True) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert (status["completed"], status["forecast_completed"], status["pending"], status["running"]) == (72, 72, 0, [])
