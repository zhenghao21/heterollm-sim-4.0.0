"""Matched HBM reference reuse and bounded formal retry orchestration."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scan():
    path = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan.py"
    spec = importlib.util.spec_from_file_location("scan_paired_retry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def reference_fixture(scan, tmp_path):
    from heterollm_sim.simulation_options import SimulationConfig
    point = scan.all_points()[1]
    scenario = SimpleNamespace(model=SimpleNamespace(name="Qwen2.5-0.5B"),
                               simulation=SimulationConfig(mode="fast"))
    source = {**point, "architecture": "HBM", "point_id": scan.point_id(
        point["model_id"], "HBM", point["prompt_tokens"], point["output_tokens"]),
        "status": "forecast_completed", "simulation_mode": "fast",
        "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS,
        "model": {"name": scenario.model.name}, "metric_fidelity": {"tpot": "estimated"},
        "resolved_placement": {"active_memory": "hbm0"}, "tpot_ns": 100.,
        "approximation": {"effective_mode": "fast", "target_relative_error": .1,
            "sampled_decode_steps": 16, "predicted_decode_steps": 1007,
            "error_budget_breakdown": {"forecast_tail_ns": 90000.,
                "combined": {"lower_ns": 89000., "upper_ns": 91000.}}}}
    manifest = {"parameters": {"weights_dtype": "bf16", "kv_dtype": "bf16"},
                "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS}
    path = tmp_path / "points" / (source["point_id"] + ".json")
    write_json(path, source)
    return point, scenario, source, manifest, path


def test_paired_reference_adds_exact_prefix_to_tail_interval(scan, tmp_path):
    point, scenario, source, manifest, _path = reference_fixture(scan, tmp_path)
    reference = scan.load_paired_reference(tmp_path, point, scenario, manifest)
    assert reference.source_id == source["point_id"] and reference.memory_kind == "hbm"
    assert reference.model_name == scenario.model.name
    assert reference.decode_estimate_ns == 102300.
    assert reference.decode_lower_ns == 101300.
    assert reference.decode_upper_ns == 103300.
    assert (reference.weight_dtype, reference.kv_dtype) == ("bf16", "bf16")


@pytest.mark.parametrize("violation", (
    "id", "model_id", "model_name", "prompt", "output", "architecture", "mode",
    "effective_mode", "fidelity", "precision", "placement", "target", "nan_tpot",
    "nan_interval", "negative_prefix", "tail_order", "count_sum", "negative_count", "bool_count",
    "source_semantics", "manifest_semantics"))
def test_paired_reference_rejects_incompatible_saved_forecast(scan, tmp_path, violation):
    point, scenario, source, manifest, path = reference_fixture(scan, tmp_path)
    changes = {"id": (source, "point_id", "other"), "model_id": (source, "model_id", "other"),
        "model_name": (source["model"], "name", "other"), "prompt": (source, "prompt_tokens", 7),
        "output": (source, "output_tokens", 7), "architecture": (source, "architecture", "HBF"),
        "mode": (source, "simulation_mode", "detailed"),
        "effective_mode": (source["approximation"], "effective_mode", "detailed"),
        "fidelity": (source["metric_fidelity"], "tpot", "detailed"),
        "precision": (manifest["parameters"], "kv_dtype", "fp8"),
        "placement": (source["resolved_placement"], "active_memory", "hostmem0"),
        "target": (source["approximation"], "target_relative_error", .2),
        "nan_tpot": (source, "tpot_ns", float("nan")),
        "nan_interval": (source["approximation"]["error_budget_breakdown"]["combined"], "lower_ns", float("nan")),
        "negative_prefix": (source, "tpot_ns", 1.),
        "tail_order": (source["approximation"]["error_budget_breakdown"]["combined"], "lower_ns", 95000.),
        "count_sum": (source["approximation"], "predicted_decode_steps", 1),
        "negative_count": (source["approximation"], "sampled_decode_steps", -1),
        "bool_count": (source["approximation"], "sampled_decode_steps", True),
        "source_semantics": (source, "physical_weight_semantics", "legacy_duplicated_weights"),
        "manifest_semantics": (manifest, "physical_weight_semantics", "legacy_duplicated_weights")}
    mapping, key, value = changes[violation]
    mapping[key] = value
    write_json(path, source)
    with pytest.raises(ValueError):
        scan.load_paired_reference(tmp_path, point, scenario, manifest)


def test_paired_reference_never_silently_accepts_missing_weight_semantics(scan, tmp_path):
    point, scenario, source, manifest, path = reference_fixture(scan, tmp_path)
    del source["physical_weight_semantics"]
    write_json(path, source)
    with pytest.raises(ValueError, match="physical weight semantics"):
        scan.load_paired_reference(tmp_path, point, scenario, manifest)


@pytest.mark.parametrize("source_status", (None, "forecast_rejected", "completed", "failed"))
def test_missing_or_non_forecast_reference_is_not_fabricated(scan, tmp_path, source_status):
    point, scenario, source, manifest, path = reference_fixture(scan, tmp_path)
    if source_status is None:
        path.unlink()
    else:
        source["status"] = source_status
        write_json(path, source)
    assert scan.load_paired_reference(tmp_path, point, scenario, manifest) is None


@dataclass
class WorkerModel:
    name: str = "fake"
    num_layers: int = 1
    total_declared_weight_bytes: int = 128
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkerScenario:
    model: object = field(default_factory=WorkerModel)
    simulation: object = None


def test_worker_forwards_reference_and_interrupts_rejected_tail(scan, tmp_path, monkeypatch):
    import b200_hbm_hbf_scan_config as config
    import heterollm_sim.reporting as reporting
    from heterollm_sim.fast_budget import PairedTimingReference
    scan.initialize(tmp_path)
    point = scan.all_points()[1]
    reference = PairedTimingReference("fake", 65536, 1024, "hbm", .1, 100., 99., 101., "hbm-reference")
    monkeypatch.setattr(config, "build_scan_scenario", lambda *args: WorkerScenario())
    monkeypatch.setattr(scan, "load_paired_reference", lambda *args: reference)
    observed = []
    approximation = {"paired_budget_decision": {"accepted": False, "reasons": ["ratio"]}}

    def run(scenario, *, paired_reference, retention_policy, control):
        assert paired_reference is reference
        assert scenario.simulation.mode == "fast" and retention_policy == "aggregate"
        observed.append("prefix")
        control.report("fast_mode_fallback", 128, 128, metadata={"approximation": approximation,
            "fallback_reasons": ["directional_pair_conditional_error_exceeds_target"]})
        pytest.fail("detailed suffix was executed after a rejected paired forecast")

    monkeypatch.setattr(reporting, "run_scenario", run)
    assert scan.worker(tmp_path, point) == 0 and observed == ["prefix"]
    saved = read_json(tmp_path / "points" / (point["point_id"] + ".json"))
    assert saved["status"] == "forecast_rejected"
    assert saved["ttft_ns"] is saved["tpot_ns"] is None
    assert saved["approximation"] == approximation


def queue_fixture(scan, tmp_path, monkeypatch, outcomes=None):
    import b200_hbm_hbf_scan_report as report
    points = scan.all_points()
    retry = tuple(point["point_id"] for point in points if point["architecture"] == "HBF"
                  and point["prompt_tokens"] == 65536 and point["output_tokens"] == 10240)
    for name in ("points", "progress", "logs", "configurations"):
        (tmp_path / name).mkdir(exist_ok=True)
    write_json(tmp_path / "manifest.json", {"points": points, "simulation": {"mode": "fast"},
        "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS})
    initial = {}
    for point in points[:27]:
        status = "forecast_rejected" if point["point_id"] in retry else "forecast_completed"
        row = {**point, "status": status, "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS, "finished_at": "2026-10-18T01:02:03+00:00",
               "reason": "old budget refusal" if status == "forecast_rejected" else None}
        write_json(tmp_path / "points" / (point["point_id"] + ".json"), row)
        initial[point["point_id"]] = deepcopy(row)
    monkeypatch.setattr(scan, "initialize", lambda *args: {"points": points,
        "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS})
    monkeypatch.setattr(report, "write_report", lambda *args: None)
    monkeypatch.setattr(report, "refresh_markdown", lambda *args: None)
    monkeypatch.setattr(scan.time, "sleep", lambda seconds: None)
    launches = []

    def popen(command, **kwargs):
        identifier = command[command.index("--point-id") + 1]
        launches.append(identifier)
        point = next(item for item in points if item["point_id"] == identifier)
        status, accepted = (outcomes or {}).get(identifier, ("forecast_completed", True))
        write_json(tmp_path / "points" / (identifier + ".json"), {**point, "status": status,
            "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS,
            "approximation": {"paired_budget_decision": {"accepted": accepted}}})
        return SimpleNamespace(pid=500000 + len(launches), poll=lambda: 1 if status == "failed" else 0)

    monkeypatch.setattr(scan.subprocess, "Popen", popen)
    return points, retry, initial, launches


def test_retry_preserves_four_refusals_prioritizes_group_and_reuses_23_successes(scan, tmp_path, monkeypatch):
    points, retry, initial, launches = queue_fixture(scan, tmp_path, monkeypatch)
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 0
    assert launches[:4] == list(retry)
    assert len(launches) == len(set(launches)) == 49
    assert all(identifier not in launches for identifier, row in initial.items() if row["status"] == "forecast_completed")
    archives = list((tmp_path / "points_history").glob("*.json"))
    assert len(archives) == 4
    assert {read_json(path)["point_id"] for path in archives} == set(retry)
    for path in archives:
        row = read_json(path)
        assert row == initial[row["point_id"]]
    status = read_json(tmp_path / "status.json")
    assert status["status"] == "completed" and status["completed"] == len(points)
    assert status["retry_group_accepted"] is True
    launches.clear()
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 0
    assert not launches


@pytest.mark.parametrize("bad_status", ("forecast_rejected", "forecast_completed"))
def test_retry_finishes_group_then_pauses_when_any_paired_decision_is_rejected(scan, tmp_path, monkeypatch, bad_status):
    points, retry, initial, launches = queue_fixture(scan, tmp_path, monkeypatch)
    # A superficially successful status cannot bypass the joint budget gate.
    bad_id = retry[0]
    outcomes = {bad_id: (bad_status, False)}
    # Reinstall only the process producer with the desired outcome.
    def popen(command, **kwargs):
        identifier = command[command.index("--point-id") + 1]
        launches.append(identifier)
        point = next(item for item in points if item["point_id"] == identifier)
        status, accepted = outcomes.get(identifier, ("forecast_completed", True))
        write_json(tmp_path / "points" / (identifier + ".json"), {**point, "status": status,
            "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS,
            "approximation": {"paired_budget_decision": {"accepted": accepted}}})
        return SimpleNamespace(pid=500000 + len(launches), poll=lambda: 0)
    monkeypatch.setattr(scan.subprocess, "Popen", popen)
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 1
    assert launches == list(retry)
    status = read_json(tmp_path / "status.json")
    assert status["status"] == "needs_attention" and status["reason"] == "retry_group_not_accepted"
    assert status["retry_group_completed"] == 4 and status["retry_group_accepted"] is False
    assert status["pending"] == 45
    assert (tmp_path / "stop.request").read_text(encoding="utf-8") == "retry_group_not_accepted"


def test_real_failure_during_retry_stops_immediately(scan, tmp_path, monkeypatch):
    points, retry, initial, launches = queue_fixture(scan, tmp_path, monkeypatch)
    def failed(command, **kwargs):
        identifier = command[command.index("--point-id") + 1]
        launches.append(identifier)
        point = next(item for item in points if item["point_id"] == identifier)
        write_json(tmp_path / "points" / (identifier + ".json"), {**point, "status": "failed",
            "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS})
        return SimpleNamespace(pid=500001, poll=lambda: 1)
    monkeypatch.setattr(scan.subprocess, "Popen", failed)
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 1
    assert launches == [retry[0]]
    assert read_json(tmp_path / "status.json")["reason"] == "point_failed"


def test_resume_does_not_repeat_an_already_accepted_retry(scan, tmp_path, monkeypatch):
    points, retry, initial, launches = queue_fixture(scan, tmp_path, monkeypatch)
    point = next(item for item in points if item["point_id"] == retry[0])
    scan.archive_rejected_point(tmp_path, point, initial[retry[0]])
    write_json(tmp_path / "points" / (retry[0] + ".json"), {**point,
        "status": "forecast_completed", "physical_weight_semantics": scan.PHYSICAL_WEIGHT_SEMANTICS,
        "approximation": {"paired_budget_decision": {"accepted": True}}})
    assert scan.run_queue(tmp_path, retry_point_ids=retry) == 0
    assert launches[:3] == list(retry[1:])
    assert retry[0] not in launches and len(launches) == 48
    assert len(list((tmp_path / "points_history").glob("*.json"))) == 4


def test_invalid_retry_in_new_directory_does_not_initialize_or_create_it(scan, tmp_path):
    output_dir = tmp_path / "missing_queue"
    with pytest.raises(ValueError):
        scan.run_queue(output_dir, retry_point_ids=("unknown-id",))
    assert not output_dir.exists()


@pytest.mark.parametrize("selection", ("unknown", "missing_source", "duplicate", "already_successful", "stop"))
def test_invalid_retry_prevalidation_changes_no_scan_files(scan, tmp_path, monkeypatch, selection):
    points, retry, initial, launches = queue_fixture(scan, tmp_path, monkeypatch)
    selected = retry
    if selection == "unknown":
        selected = ("unknown-id",)
    elif selection == "missing_source":
        selected = (points[27]["point_id"],)
    elif selection == "duplicate":
        selected = (retry[0], retry[0])
    elif selection == "already_successful":
        selected = (points[0]["point_id"],)
    else:
        (tmp_path / "stop.request").write_text("user pause", encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        scan.run_queue(tmp_path, retry_point_ids=selected)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before and not launches


@pytest.mark.parametrize("mode,scheduler", (("detailed", "continuous"), ("fast", "static")))
def test_reporting_rejects_reference_outside_continuous_fast_mode(monkeypatch, mode, scheduler):
    import heterollm_sim.llama_scenario as llama
    import heterollm_sim.reporting as reporting
    scenario = SimpleNamespace(simulation=SimpleNamespace(mode=mode),
        workload=SimpleNamespace(scheduler=SimpleNamespace(mode=scheduler)))
    monkeypatch.setattr(llama, "prepare_llama_scenario", lambda item: item)
    monkeypatch.setattr(reporting, "bootstrap_control_plane", lambda *args, **kwargs: pytest.fail("bootstrapped invalid mode"))
    with pytest.raises(ValueError, match="continuous fast"):
        reporting.run_scenario(scenario, paired_reference=object())


def test_reporting_forwards_reference_to_fast_runner_without_detailed_simulation(monkeypatch):
    import heterollm_sim.llama_scenario as llama
    import heterollm_sim.reporting as reporting
    scenario = SimpleNamespace(simulation=SimpleNamespace(mode="fast"),
        workload=SimpleNamespace(scheduler=SimpleNamespace(mode="continuous")), weights_resident=False)
    reference, lowerer = object(), object()
    control_plane = SimpleNamespace(scenario=scenario, makespan_ns=0., kernel=object(),
        result=SimpleNamespace(completed_count=0), decision=SimpleNamespace(status="ready"))
    monkeypatch.setattr(llama, "prepare_llama_scenario", lambda item: item)
    monkeypatch.setattr(reporting, "bootstrap_control_plane", lambda *args, **kwargs: control_plane)
    monkeypatch.setattr(reporting, "_compilation_scope", lambda item: nullcontext())
    monkeypatch.setattr(reporting, "validate_scenario", lambda item: SimpleNamespace(raise_for_errors=lambda: None))
    class ReachedFastRunner(Exception):
        pass
    def fast(item, **kwargs):
        assert item is scenario and kwargs["paired_reference"] is reference
        assert kwargs["batch_lowerer"] is lowerer and kwargs["custom_lowerer"] is True
        raise ReachedFastRunner
    monkeypatch.setattr(reporting, "simulate_fast", fast)
    monkeypatch.setattr(reporting, "simulate_online", lambda *args, **kwargs: pytest.fail("ran detailed mode"))
    with pytest.raises(ReachedFastRunner):
        reporting.run_scenario(scenario, batch_lowerer=lowerer, paired_reference=reference)
