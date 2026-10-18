"""The website adapter is read-only and never invents unavailable measurements."""
import importlib.util
import io
import json
import os
from pathlib import Path
import urllib.error

import pytest


@pytest.fixture
def dashboard():
    source = Path(__file__).resolve().parents[1] / "scripts/b200_scan_dashboard.py"
    spec = importlib.util.spec_from_file_location("scan_dashboard_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def directory(tmp_path):
    points = [{"point_id": f"small__{arch}", "model_id": "small", "architecture": arch,
               "prompt_tokens": 65536, "output_tokens": 1024} for arch in ("HBM", "HBF")]
    manifest = {"physical_weight_semantics": "canonical_shared_resident_weights_v1", "points": points,
                "models": [{"model_id": "small", "label": "Small", "parameters_b": .5}],
                "parameters": {"prefill_chunk_tokens": 2048},
                "simulation": {"fast_decode": {"max_sample_tokens": 128}}}
    write(tmp_path / "manifest.json", manifest)
    write(tmp_path / "status.json", {"physical_weight_semantics": manifest["physical_weight_semantics"],
                                     "status": "running", "manager_pid": 100, "running": []})
    return tmp_path


def row(directory, architecture="HBF", **updates):
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = next(p for p in manifest["points"] if p["architecture"] == architecture)
    value = {**spec, "physical_weight_semantics": manifest["physical_weight_semantics"],
             "status": "forecast_completed", "ttft_ns": 2e9, "tpot_ns": 3e6, **updates}
    write(directory / "points" / f"{spec['point_id']}.json", value)
    return value


def test_pending_is_null_and_complete_manifest_is_retained(dashboard, directory):
    snapshot = dashboard.ScanReader(directory, alive=lambda pid: pid == 100).snapshot()
    assert len(snapshot["points"]) == 2
    assert snapshot["status"]["total"] == snapshot["status"]["pending"] == 2
    assert snapshot["status"]["manager_alive"] is True
    assert all(p["ttft_ms"] is None and p["tpot_ms"] is None for p in snapshot["points"])
    assert all(p["sampled_decode_steps"] is None for p in snapshot["points"])


@pytest.mark.parametrize("worker_record", [False, True])
def test_cancelled_rows_are_terminal_empty_and_retained_results_survive(dashboard, directory, worker_record):
    manifest = json.loads((directory / "manifest.json").read_text())
    retained = {**manifest["points"][0], "point_id": "small__HBM__D5120", "output_tokens": 5120}
    cancelled = {**manifest["points"][1], "point_id": "small__HBF__D10240", "output_tokens": 10240}
    manifest["points"].extend([retained, cancelled])
    write(directory / "manifest.json", manifest)
    original = row(directory)
    long_result = {**retained, "status": "forecast_completed", "ttft_ns": 2e9, "tpot_ns": 3e6,
                   "physical_weight_semantics": manifest["physical_weight_semantics"]}
    write(directory / "points" / f"{retained['point_id']}.json", long_result)
    cancelled.update(status="cancelled", reason="user_cancelled_unrun_long_outputs", cancelled_at="2026-10-18T16:00:00+00:00")
    write(directory / "queue_plan.json", {"schema": "b200-scan-plan/v1", "cancelled_points": [cancelled]})
    if worker_record:
        write(directory / "points" / f"{cancelled['point_id']}.json", {
            **cancelled, "physical_weight_semantics": dashboard.SEMANTICS, "ttft_ns": None, "tpot_ns": None})
    snapshot = dashboard.ScanReader(directory).snapshot()
    assert len(snapshot["points"]) == 4
    status = snapshot["status"]
    assert (status["total"], status["planned_total"], status["completed"], status["cancelled"], status["pending"]) == (4, 3, 2, 1, 1)
    assert status["one_k_total"] == 2 and status["one_k_completed"] == status["long_output_completed"] == 1
    result = snapshot["points"][-1]
    assert result["status"] == "cancelled" and result["cancelled_at"] == cancelled["cancelled_at"]
    assert result["ttft_ms"] is None and result["tpot_ms"] is None
    assert snapshot["points"][1]["tpot_ms"] == original["tpot_ns"] / 1e6
    assert any("不自动恢复取消项" in setting["value"] for setting in snapshot["settings"])


def test_cancelled_plan_cannot_hide_an_existing_success(dashboard, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["points"][1]["output_tokens"] = 5120
    write(directory / "manifest.json", manifest)
    result = row(directory)
    cancelled = {key: result[key] for key in ("point_id", "model_id", "architecture", "prompt_tokens", "output_tokens")}
    cancelled.update(status="cancelled", reason="cancelled", cancelled_at="2026-10-18T16:00:00+00:00")
    write(directory / "queue_plan.json", {"schema": "b200-scan-plan/v1", "cancelled_points": [cancelled]})
    with pytest.raises(ValueError, match="existing result|active result"):
        dashboard.ScanReader(directory).snapshot()


@pytest.mark.parametrize("status", ["forecast_rejected", "capacity_rejected", "failed", "interrupted"])
def test_rejected_metrics_never_leak_even_if_saved(dashboard, directory, status):
    row(directory, status=status, reason="guard refused")
    result = dashboard.ScanReader(directory).snapshot()["points"][1]
    assert result["ttft_ms"] is None and result["tpot_ms"] is None
    assert result["reason"] == "guard refused"


def test_only_current_semantics_and_manifest_matching_points_count(dashboard, directory):
    row(directory, physical_weight_semantics="old_weights")
    write(directory / "points_history/other.json", {"status": "forecast_completed", "ttft_ns": 1})
    write(directory / "points/foreign.json", {"status": "forecast_completed", "ttft_ns": 1})
    snapshot = dashboard.ScanReader(directory).snapshot()
    assert snapshot["status"]["completed"] == 0
    assert snapshot["points"][1]["status"] == "pending"
    assert len(snapshot["warnings"]) == 1
    row(directory, output_tokens=10240)
    assert dashboard.ScanReader(directory).snapshot()["status"]["completed"] == 0


def test_real_worker_pid_and_whole_cohort_progress_not_launcher(dashboard, directory):
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text())
    status["running"] = [{"point_id": "small__HBF", "model_id": "small", "pid": 200,
                          "started_at": "2026-10-18T12:00:00+00:00"}]
    write(status_path, status)
    progress = {"physical_weight_semantics": dashboard.SEMANTICS, "pid": 300,
                "updated_at": "2026-10-18T12:01:00+00:00", "real_prefill_steps": 17,
                "real_decode_steps": 0, "wall_seconds": 20,
                "progress": {"stage": "cohort_tasks", "completed": 123, "total": 456}}
    write(directory / "progress/small__HBF.json", progress)
    observed = []
    snapshot = dashboard.ScanReader(directory, alive=lambda pid: observed.append(pid) or pid in (100, 300)).snapshot()
    active = snapshot["status"]["running"][0]
    assert active["pid"] == 300 and active["worker_alive"] is True
    assert active["prefill_completed"] == 17 and active["prefill_total"] == 32
    assert active["detail_completed"] == 123 and active["detail_total"] == 456
    assert active["decode_limit"] == 128
    assert 200 not in observed
    assert snapshot["status"]["pending"] == 1
    assert snapshot["points"][1]["status"] == "running" and snapshot["points"][1]["tpot_ms"] is None


def test_stale_progress_is_not_a_new_worker(dashboard, directory):
    status = json.loads((directory / "status.json").read_text())
    status["running"] = [{"point_id": "small__HBF", "pid": 200, "started_at": "2026-10-18T12:00:00+00:00"}]
    write(directory / "status.json", status)
    write(directory / "progress/small__HBF.json", {"physical_weight_semantics": dashboard.SEMANTICS,
                                                  "pid": 300, "updated_at": "2026-10-18T11:00:00+00:00"})
    snapshot = dashboard.ScanReader(directory, alive=lambda pid: True).snapshot()
    assert snapshot["status"]["running"][0]["pid"] is None
    assert snapshot["status"]["running"][0]["worker_alive"] is False
    assert snapshot["warnings"]


def test_mtime_cache_avoids_large_json_reparse_and_deletion_is_not_reused(dashboard, directory):
    row(directory, approximation={"unused_huge_array": list(range(10000))})
    reader = dashboard.ScanReader(directory)
    first = reader.snapshot()
    count = reader.cache.parse_count
    reader.snapshot()
    assert reader.cache.parse_count == count
    row(directory, ttft_ns=3e9)
    assert reader.snapshot()["points"][1]["ttft_ms"] == 3000
    assert reader.cache.parse_count == count + 1
    (directory / "points/small__HBF.json").unlink()
    assert reader.snapshot()["points"][1]["ttft_ms"] is None
    assert first["points"][1]["ttft_ms"] == 2000


def test_publication_race_uses_previous_document_without_writing_results(dashboard, directory):
    row(directory)
    reader = dashboard.ScanReader(directory)
    reader.snapshot()
    result_path = directory / "points/small__HBF.json"
    result_path.write_text('{"partial":', encoding="utf-8")
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.rglob("*") if path.is_file()}
    snapshot = reader.snapshot()
    assert snapshot["points"][1]["ttft_ms"] == 2000
    assert snapshot["warnings"]
    after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.rglob("*") if path.is_file()}
    assert before == after
    row(directory, ttft_ns=5e9)
    assert reader.snapshot()["points"][1]["ttft_ms"] == 5000


def test_budget_preserves_numeric_zero_terms_and_conditional_pair_bounds(dashboard, directory):
    row(directory, approximation={
        "paired_budget_decision": {"accepted": False, "candidate_relative_error_budget": .09,
                                   "reference_relative_error_budget": .001, "pair_relative_error_budget": .101},
        "sampled_decode_steps": 28, "predicted_decode_steps": 995,
        "error_budget_breakdown": {"future_gc_erase_serial_guard_ns": 0, "prefix_gc_bias_guard_ns": 3e6,
                                    "bad_ns": True, "invalid_ns": "unknown", "nested": {"lower_ns": 4},
                                    "relative_error": .09},
        "model_assumptions": ["conditional, not measured"],
    })
    value = dashboard.ScanReader(directory).snapshot()["points"][1]
    assert value["single_error_bound"] == .09 and value["reference_error_bound"] == .001
    assert value["pair_error_bound"] == .101 and value["paired_accepted"] is False
    assert value["sampled_decode_steps"] == 28 and value["predicted_decode_steps"] == 995
    assert {x["key"]: x["ns"] for x in value["budget"]} == {
        "future_gc_erase_serial_guard_ns": 0, "prefix_gc_bias_guard_ns": 3e6}


def test_cli_snapshot_never_reads_stdin_or_writes_status(dashboard, directory, monkeypatch, capsys):
    class NoRead:
        def readline(self):
            raise AssertionError("snapshot must not read credentials")
    monkeypatch.setattr(dashboard.sys, "stdin", NoRead())
    assert dashboard.main(["--output-dir", str(directory), "--snapshot"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "b200-scan-dashboard/v1"


@pytest.mark.parametrize("url", ["http://example.com", "https://token@example.com", "https://example.com/?token=secret"])
def test_endpoint_rejects_credential_or_insecure_url(dashboard, url):
    with pytest.raises(ValueError):
        dashboard.endpoint(url)


def test_endpoint_accepts_base_or_full_path(dashboard):
    assert dashboard.endpoint("https://example.com/") == "https://example.com/api/scan"
    assert dashboard.endpoint("http://127.0.0.1:8000/api/scan") == "http://127.0.0.1:8000/api/scan"


def test_private_upload_header_json_ack_and_no_redirect(dashboard, monkeypatch):
    observed = []
    class Headers:
        def get_content_type(self):
            return "application/json"
    class Response:
        status = 200
        headers = Headers()
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, size): return b'{"ok":true}'
    class Opener:
        def open(self, request, timeout):
            observed.append((request, timeout))
            return Response()
    monkeypatch.setattr(dashboard.urllib.request, "build_opener", lambda *handlers: Opener())
    assert dashboard.send_snapshot("https://example.com/api/scan", "testing-secret", {"schema": "test"}) == 200
    request, timeout = observed[0]
    assert request.get_method() == "PUT" and timeout == 20
    assert request.get_header("Oai-sites-authorization") == "Bearer testing-secret"
    assert request.get_header("User-agent") == "B200-Memory-Lab/1.0"
    assert json.loads(request.data) == {"schema": "test"}


def test_html_login_is_not_reported_as_success(dashboard, monkeypatch):
    class Headers:
        def get_content_type(self): return "text/html"
    class Response:
        status = 200
        headers = Headers()
        def __enter__(self): return self
        def __exit__(self, *_): pass
    class Opener:
        def open(self, *_args, **_kwargs): return Response()
    monkeypatch.setattr(dashboard.urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(ValueError, match="JSON"):
        dashboard.send_snapshot("https://example.com/api/scan", "testing-secret", {})


def test_local_qa_never_uses_a_remote_proxy(dashboard, monkeypatch):
    def unexpected():
        raise AssertionError("local QA should not inspect a proxy")
    monkeypatch.setattr(dashboard.urllib.request, "getproxies", unexpected)
    assert dashboard.transport_proxies("http://127.0.0.1:5179/api/scan") == {}


def test_explicit_environment_proxy_is_preserved(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard.urllib.request, "getproxies", lambda: {
        "https": "http://explicit:8000", "node_use_env": "true", "no": "localhost"})
    def unexpected():
        raise AssertionError("explicit routing must be preserved")
    monkeypatch.setattr(dashboard.urllib.request, "getproxies_registry", unexpected, raising=False)
    assert dashboard.transport_proxies("https://example.com/api/scan") == {
        "https": "http://explicit:8000", "no": "localhost"}


@pytest.mark.skipif(os.name != "nt", reason="Windows registry proxy selection")
def test_node_env_flag_does_not_hide_existing_windows_proxy(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard.urllib.request, "getproxies", lambda: {
        "node_use_env": "true", "no": "localhost"})
    monkeypatch.setattr(dashboard.urllib.request, "getproxies_registry", lambda: {
        "http": "http://system:9000", "https": "http://system:9000"})
    assert dashboard.transport_proxies("https://example.com/api/scan") == {
        "http": "http://system:9000", "https": "http://system:9000", "no": "localhost"}


def test_401_status_does_not_include_token(dashboard, directory, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dashboard.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(dashboard.sys, "stdin", io.StringIO('{"token":"secret-token"}\n'))
    def unauthorized(*_):
        raise urllib.error.HTTPError("https://example.com", 401, "secret-token", {}, None)
    monkeypatch.setattr(dashboard, "send_snapshot", unauthorized)
    status_path = tmp_path.parent / "publisher-state.json"
    assert dashboard.main(["--output-dir", str(directory), "--url", "https://example.com", "--once",
                           "--status-file", str(status_path)]) == 2
    raw = status_path.read_text()
    assert "secret-token" not in raw + capsys.readouterr().out
    assert json.loads(raw)["http_status"] == 401


def test_single_publisher_lock_rejects_second_instance(dashboard, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.tempfile, "gettempdir", lambda: str(tmp_path))
    with dashboard.SinglePublisher():
        with pytest.raises(RuntimeError, match="already running"):
            with dashboard.SinglePublisher():
                pass


def test_background_passes_credential_only_through_stdin_pipe(dashboard, directory, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path.with_name(tmp_path.name + "_publisher"))
    monkeypatch.setattr(dashboard.sys, "stdin", io.StringIO('{"token":"background-secret"}\n'))
    captured = {}
    class Pipe:
        def write(self, value): captured["payload"] = value
        def flush(self): captured["flushed"] = True
        def close(self): captured["closed"] = True
    class Process:
        pid = 12345
        stdin = Pipe()
    def popen(command, **options):
        captured["command"] = command
        captured["options"] = options
        return Process()
    monkeypatch.setattr(dashboard.subprocess, "Popen", popen)
    status_path = dashboard.ROOT / "uploader.json"
    assert dashboard.main(["--output-dir", str(directory), "--url", "https://example.com",
                           "--background", "--status-file", str(status_path)]) == 0
    assert captured["command"][0] == dashboard.sys.executable
    assert "--background" not in captured["command"] and "--snapshot" not in captured["command"]
    assert "background-secret" not in str(captured["command"]) + str(captured["options"])
    assert "env" not in captured["options"]
    assert captured["options"]["stdin"] == dashboard.subprocess.PIPE
    assert captured["options"]["stderr"] == dashboard.subprocess.STDOUT
    assert captured["closed"] and captured["flushed"]
    assert json.loads(captured["payload"]) == {"token": "background-secret"}
    if dashboard.os.name == "nt":
        assert captured["options"]["creationflags"] == dashboard.subprocess.CREATE_NO_WINDOW | dashboard.subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        assert captured["options"]["start_new_session"] is True
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["launcher_pid"] == 12345
    assert "background-secret" not in stdout
    log_path = dashboard.ROOT / "tmp/b200_scan_dashboard_publisher.log"
    assert log_path.read_bytes() == b"" and not status_path.exists()
