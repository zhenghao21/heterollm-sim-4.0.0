"""Dedicated process tuning must restore the caller's collector settings."""
import gc
import importlib.util
import json
from pathlib import Path
from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace

import pytest


def load_scan_worker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan.py"
    spec = importlib.util.spec_from_file_location("scan_worker_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_publication_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    module = load_scan_worker()
    target = tmp_path / "progress.json"
    target.write_text('{"completed": 1}', encoding="utf-8")
    original_replace = Path.replace
    attempts, delays = [], []

    def replace(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 1}
        if len(attempts) < 3:
            raise PermissionError("reader temporarily holds the destination")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    module.write_json(target, {"completed": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 2}
    assert len(attempts) == 3 and delays == [0.1, 0.1]


def test_progress_publication_still_reports_persistent_permission_failure(tmp_path, monkeypatch):
    module = load_scan_worker()
    target = tmp_path / "progress.json"
    target.write_text('{"completed": 1}', encoding="utf-8")
    attempts, delays = [], []

    def denied(source, destination):
        attempts.append(source)
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="access denied"):
        module.write_json(target, {"completed": 2})
    assert len(attempts) == 10 and delays == [0.1] * 9
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 1}


@pytest.mark.parametrize("enabled", (True, False))
def test_worker_gc_context_preserves_enablement_and_restores_after_failure(enabled):
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan.py"
    spec = importlib.util.spec_from_file_location("scan_worker_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous, was_enabled = gc.get_threshold(), gc.isenabled()
    try:
        gc.set_threshold(12000, 7, 130)
        (gc.enable if enabled else gc.disable)()
        with pytest.raises(RuntimeError, match="cancelled simulation"):
            with module.scenario_gc_thresholds() as configured:
                assert configured == gc.get_threshold() == (12000, 100, 130)
                assert gc.isenabled() is enabled
                raise RuntimeError("cancelled simulation")
        assert gc.get_threshold() == (12000, 7, 130)
        assert gc.isenabled() is enabled
    finally:
        gc.set_threshold(*previous)
        (gc.enable if was_enabled else gc.disable)()


@pytest.fixture
def scan():
    return load_scan_worker()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_directories(directory):
    for name in ("points", "progress", "logs", "configurations"):
        (directory / name).mkdir(exist_ok=True)


def test_queue_contains_all_72_authorized_combinations(scan):
    points = scan.all_points()
    expected = set(product((model["model_id"] for model in scan.MODELS),
                           ("HBM", "HBF"), (65536, 262144, 524288), (1024, 5120, 10240)))
    actual = {(p["model_id"], p["architecture"], p["prompt_tokens"], p["output_tokens"])
              for p in points}
    assert actual == expected and len(points) == len(actual) == 72
    assert len({point["point_id"] for point in points}) == 72


def test_manifest_defaults_to_fast_and_rejects_mode_change(scan, tmp_path):
    manifest = scan.initialize(tmp_path)
    assert manifest["point_count"] == len(manifest["points"]) == 72
    assert manifest["simulation"]["mode"] == "fast"
    assert manifest["forecast_refusal_policy"] == "record_and_continue_without_detailed_suffix"
    assert scan.initialize(tmp_path) == manifest
    with pytest.raises(ValueError, match="parameters differ"):
        scan.initialize(tmp_path, "detailed")
    assert read_json(tmp_path / "manifest.json") == manifest


def fast_result(point):
    expected = "hbf0" if point["architecture"] == "HBF" else "hbm0"
    placement = SimpleNamespace(op_to_component={"layer": "gpu0"},
        tensor_to_component={"weight": expected, "activation": expected, "workspace": expected},
        kv_policy=SimpleNamespace(cache_component=expected))
    scenario = SimpleNamespace(placement=placement, hardware=SimpleNamespace(components=[
        SimpleNamespace(normalized_kind="gpu"), SimpleNamespace(normalized_kind=point["architecture"].lower())]))
    forecast = SimpleNamespace(output_tokens=point["output_tokens"], prompt_tokens=point["prompt_tokens"],
        sampled_decode_steps=16, predicted_decode_steps=point["output_tokens"] - 17,
        ttft_ns=2000., tpot_ns=100.)
    return SimpleNamespace(scenario=scenario, forecast=forecast,
        approximation={"effective_mode": "fast", "sampling_attempts": [{"accepted": True}],
                       "sampled_decode_steps": 16})


@pytest.mark.parametrize("architecture", ("HBM", "HBF"))
def test_collect_forecast_keeps_real_prefix_fidelity_without_full_counters(scan, architecture):
    point = {**scan.all_points()[0], "architecture": architecture}
    result = fast_result(point)
    row = scan.collect_result(result, point)
    assert row["status"] == "forecast_completed" and row["simulation_mode"] == "fast"
    assert (row["ttft_ns"], row["tpot_ns"], row["visible_output_tokens"]) == (2000., 100., 1024)
    assert row["metric_fidelity"] == {"ttft": "detailed", "tpot": "estimated"}
    assert row["physical_execution_scope"] == "prefix_only"
    assert row["approximation"] == result.approximation
    assert row["approximation"] is not result.approximation
    assert {"resource_busy_ns", "storage_traffic", "dram_traffic", "batch_count"}.isdisjoint(row)


@pytest.mark.parametrize("field_name,value", (("output_tokens", 9), ("prompt_tokens", 32),
    ("predicted_decode_steps", 1), ("ttft_ns", float("nan")), ("tpot_ns", 0.)))
def test_collect_rejects_incomplete_or_invalid_forecast(scan, field_name, value):
    point = scan.all_points()[0]
    result = fast_result(point)
    setattr(result.forecast, field_name, value)
    with pytest.raises(ValueError, match="complete request"):
        scan.collect_result(result, point)


@pytest.mark.parametrize("violation", ("operator", "kv", "tensor", "hbm_present"))
def test_collect_rejects_wrong_hbf_placement(scan, violation):
    point = scan.all_points()[1]
    result = fast_result(point)
    placement = result.scenario.placement
    if violation == "operator":
        placement.op_to_component["layer"] = "cpu0"
    elif violation == "kv":
        placement.kv_policy.cache_component = "hostmem0"
    elif violation == "tensor":
        placement.tensor_to_component["weight"] = "hostmem0"
    else:
        result.scenario.hardware.components.append(SimpleNamespace(normalized_kind="hbm"))
    with pytest.raises(ValueError):
        scan.collect_result(result, point)


@dataclass
class FakeModel:
    name: str = "fake model"
    num_layers: int = 1
    total_declared_weight_bytes: int = 128
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeScenario:
    model: FakeModel = field(default_factory=FakeModel)
    simulation: object = None


def prepare_worker(scan, tmp_path, monkeypatch):
    import b200_hbm_hbf_scan_config as config
    import heterollm_sim.reporting as reporting
    scan.initialize(tmp_path)
    monkeypatch.setattr(config, "build_scan_scenario", lambda *args: FakeScenario())
    return reporting, scan.all_points()[1]


def test_worker_forecast_refusal_is_terminal_without_detailed_suffix(scan, tmp_path, monkeypatch):
    reporting, point = prepare_worker(scan, tmp_path, monkeypatch)
    approximation = {"effective_mode": "detailed", "sampling_attempts": [{"accepted": False,
        "budget": {"local": .02, "maintenance": .03, "total": .05}}], "raw_samples": [1., 2.]}
    after_fallback = []

    def run(scenario, *, retention_policy, control):
        assert scenario.simulation.mode == "fast" and retention_policy == "aggregate"
        control.report("serving_cohorts", 1, 32, metadata={"cohort_kind": "prefill"})
        control.report("serving_cohorts", 2, 32, metadata={"cohort_kind": "decode"})
        control.report("fast_mode_fallback", 80, 128,
            metadata={"approximation": approximation, "fallback_reasons": ["budget above target"]})
        after_fallback.append("detailed suffix executed")
        pytest.fail("forecast refusal did not interrupt the detailed suffix")

    monkeypatch.setattr(reporting, "run_scenario", run)
    assert scan.worker(tmp_path, point) == 0
    assert not after_fallback
    row = read_json(tmp_path / "points" / (point["point_id"] + ".json"))
    assert row["status"] == "forecast_rejected" and row["reason"] == "budget above target"
    assert row["ttft_ns"] is None and row["tpot_ns"] is None
    assert row["approximation"] == approximation and row["physical_execution_scope"] == "prefix_only"
    assert row["real_prefill_steps"] == row["real_decode_steps"] == 1
    assert {"resource_busy_ns", "storage_traffic", "dram_traffic"}.isdisjoint(row)
    assert read_json(tmp_path / "progress" / (point["point_id"] + ".json"))["status"] == "forecast_rejected"


@pytest.mark.parametrize("error_type,message,status,exit_code", (
    ("allocation", "capacity exhausted", "capacity_rejected", 0),
    ("memory", "HBF capacity exhausted", "capacity_rejected", 0),
    ("memory", "insufficient HBF pages", "capacity_rejected", 0),
    ("memory", "host Python allocation failed", "failed", 1)))
def test_worker_distinguishes_modeled_capacity_from_host_memory_failure(
        scan, tmp_path, monkeypatch, error_type, message, status, exit_code):
    from heterollm_sim.memory_allocator import AllocationError
    reporting, point = prepare_worker(scan, tmp_path, monkeypatch)

    def run(*args, **kwargs):
        raise (AllocationError if error_type == "allocation" else MemoryError)(message)

    monkeypatch.setattr(reporting, "run_scenario", run)
    assert scan.worker(tmp_path, point) == exit_code
    row = read_json(tmp_path / "points" / (point["point_id"] + ".json"))
    assert row["status"] == status and message in row["reason"]
    assert row["ttft_ns"] is None and row["tpot_ns"] is None


def mock_queue(scan, tmp_path, monkeypatch, points, statuses):
    import b200_hbm_hbf_scan_report as report
    make_directories(tmp_path)
    monkeypatch.setattr(scan, "initialize", lambda *args: {"points": points})
    monkeypatch.setattr(report, "write_report", lambda *args: None)
    monkeypatch.setattr(report, "refresh_markdown", lambda *args: None)
    monkeypatch.setattr(scan.time, "sleep", lambda seconds: None)
    launches = []

    def popen(command, **kwargs):
        identifier = command[command.index("--point-id") + 1]
        point = next(point for point in points if point["point_id"] == identifier)
        status = statuses[len(launches) % len(statuses)]
        launches.append(identifier)
        scan.write_json(tmp_path / "points" / (identifier + ".json"), {**point, "status": status,
            "ttft_ns": 10. if status == "forecast_completed" else None,
            "tpot_ns": 1. if status == "forecast_completed" else None})
        return SimpleNamespace(pid=500000 + len(launches), poll=lambda: 1 if status == "failed" else 0)

    monkeypatch.setattr(scan.subprocess, "Popen", popen)
    return launches


def test_queue_processes_all_72_points_and_keeps_rejections_terminal(scan, tmp_path, monkeypatch):
    points = scan.all_points()
    launches = mock_queue(scan, tmp_path, monkeypatch, points,
        ("forecast_completed", "forecast_rejected", "capacity_rejected"))
    assert scan.run_queue(tmp_path, workers=2) == 0
    assert len(launches) == len(set(launches)) == 72
    status = read_json(tmp_path / "status.json")
    assert status["status"] == "completed_with_rejections"
    assert status["completed"] == status["forecast_completed"] == 24
    assert status["forecast_rejected"] == status["capacity_rejected"] == 24
    assert status["pending"] == status["failed"] == 0 and status["running"] == []
    launches.clear()
    assert scan.run_queue(tmp_path) == 0
    assert launches == []


def test_queue_stops_on_real_failure(scan, tmp_path, monkeypatch):
    points = scan.all_points()[:4]
    launches = mock_queue(scan, tmp_path, monkeypatch, points, ("failed",))
    assert scan.run_queue(tmp_path, workers=1) == 1
    assert launches == [points[0]["point_id"]]
    status = read_json(tmp_path / "status.json")
    assert status["status"] == "needs_attention" and status["reason"] == "point_failed"
    assert status["failed"] == 1 and status["pending"] == 3


def test_queue_preserves_existing_stop_marker_without_launching(scan, tmp_path, monkeypatch):
    launches = mock_queue(scan, tmp_path, monkeypatch, scan.all_points(), ("forecast_completed",))
    marker = tmp_path / "stop.request"
    marker.write_text("user pause", encoding="utf-8")
    assert scan.run_queue(tmp_path) == 1 and launches == []
    assert marker.read_text(encoding="utf-8") == "user pause"
    assert read_json(tmp_path / "status.json")["status"] == "paused"


def test_actual_worker_pid_in_progress_blocks_duplicate_queue(scan, tmp_path, monkeypatch):
    make_directories(tmp_path)
    scan.write_json(tmp_path / "progress" / "actual.json", {"status": "running", "pid": 123456})
    checked = []
    monkeypatch.setattr(scan, "process_alive", lambda pid: checked.append(pid) or True)
    with pytest.raises(RuntimeError, match="worker 123456.*still alive"):
        scan.assert_no_live_workers(tmp_path)
    assert checked == [123456]


@pytest.mark.parametrize("status,alive", (("completed", True), ("forecast_rejected", True), ("running", False)))
def test_completed_or_dead_progress_does_not_block_queue(scan, tmp_path, monkeypatch, status, alive):
    make_directories(tmp_path)
    scan.write_json(tmp_path / "progress" / "done.json", {"status": status, "pid": 123456})
    monkeypatch.setattr(scan, "process_alive", lambda pid: alive)
    scan.assert_no_live_workers(tmp_path)


def test_process_alive_detects_current_interpreter(scan):
    import os
    assert scan.process_alive(os.getpid())
    assert not scan.process_alive(None)
    assert not scan.process_alive(0)


def test_counts_do_not_count_rejections_as_completed(scan):
    counts = scan.queue_counts([{"status": status} for status in (
        "completed", "forecast_completed", "capacity_rejected", "forecast_rejected", "failed")])
    assert counts == {"completed": 2, "forecast_completed": 1, "capacity_rejected": 1,
                      "forecast_rejected": 1, "failed": 1}
