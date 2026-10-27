"""Manager-owned hard deadlines: these tests never launch a simulator."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exploring_hbf_native as run


class Process:
    pid = 123456

    def __init__(self):
        self.code = None
        self.killed = False

    def poll(self):
        return self.code


def job(started=100., factor=1):
    return {"process": Process(), "case": run.case_matrix()[0], "layout": run.CONTROL_LAYOUT,
            "batch": 1, "factor": factor, "peak": 0, "started_monotonic": started,
            "candidate_wall_budget_seconds": 3000.,
            "watchdog_phase": "decode", "watchdog_phase_started": started}


def identity(value):
    return run.candidate_id(value["case"], value["layout"], value["batch"])


def fake_kill(process):
    process.killed = True
    process.code = 137
    return process.code, True


def progress(directory, value, phase="prefill", started=None, phase_started=200.):
    run.write_json(directory / "progress" / (identity(value) + ".json"), {
        "candidate_id": identity(value), "pid": value["process"].pid,
        "candidate_started_monotonic": value["started_monotonic"] + .1 if started is None else started,
        "watchdog_phase": phase, "watchdog_phase_started_monotonic": phase_started, "state": "running"})


def test_hard_deadline_is_absolute_and_preserves_rejection(tmp_path, monkeypatch):
    value = job(factor=2)
    clock = [100. + 3000. - .001]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(run, "_kill_worker_tree", fake_kill)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") is None
    assert not value["process"].killed
    clock[0] += .002
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") == 137
    row = run._settle_candidate(tmp_path, value["case"], value["layout"], 1, 137, "fast")
    assert row["status"] == "forecast_rejected" and row["metrics"] is None
    assert row["process_watchdog"]["budget_seconds"] == 3000.
    assert row["process_watchdog"]["hard_limit_seconds"] == 3000.
    assert run.search_plan.classify(row) == "too_big"
    assert run.read_json(tmp_path / "progress" / (identity(value) + ".json"))["state"] == "forecast_rejected"
    assert not (tmp_path / "stop.request").exists()


def test_prefill_gets_one_separate_manager_timed_budget(tmp_path, monkeypatch):
    value = job()
    clock = [200.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(run, "_kill_worker_tree", fake_kill)
    progress(tmp_path, value)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") is None
    assert value["watchdog_phase"] == "prefill" and value["watchdog_phase_started"] == 200.
    clock[0] = 2000.
    progress(tmp_path, value, phase="decode", started=1999.)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") is None
    progress(tmp_path, value, started=1999., phase_started=1999.)
    clock[0] = 200. + 3000. + .001
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") == 137
    assert value["watchdog_phase_started"] == 200.
    row = run.read_candidate(tmp_path, identity(value))
    assert row["process_watchdog"]["phase"] == "prefill"
    assert row["prefill_wall_seconds"] == pytest.approx(3000.001)


@pytest.mark.parametrize("started", [99., 1000., None])
def test_stale_future_or_missing_start_cannot_grant_prefill_budget(tmp_path, monkeypatch, started):
    value = job()
    monkeypatch.setattr(run.time, "perf_counter", lambda: 200.)
    progress(tmp_path, value, started=started)
    if started is None:
        path = tmp_path / "progress" / (identity(value) + ".json")
        observed = run.read_json(path)
        del observed["candidate_started_monotonic"]
        run.write_json(path, observed)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") is None
    assert value["watchdog_phase"] == "decode"


def test_late_prefill_transition_does_not_hide_expired_decode(tmp_path, monkeypatch):
    value = job()
    monkeypatch.setattr(run.time, "perf_counter", lambda: 3101.)
    monkeypatch.setattr(run, "_kill_worker_tree", fake_kill)
    progress(tmp_path, value, phase_started=3201.)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") == 137
    assert value["process_watchdog"]["phase"] == "decode"


def test_delayed_poll_after_decode_deadline_accepts_timely_prefill_without_extra_time(tmp_path, monkeypatch):
    value = job()
    clock = [2261.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(run, "_kill_worker_tree", fake_kill)
    progress(tmp_path, value, phase_started=200.)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") is None
    assert not value["process"].killed
    assert value["watchdog_phase"] == "prefill" and value["watchdog_phase_started"] == 200.
    clock[0] = 3200.001
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") == 137
    assert value["process_watchdog"]["phase_wall_seconds"] == pytest.approx(3000.001)


def test_watchdog_uses_the_jobs_actual_candidate_budget(tmp_path, monkeypatch):
    value = job()
    value["candidate_wall_budget_seconds"] = 10.
    monkeypatch.setattr(run.time, "perf_counter", lambda: 112.001)
    monkeypatch.setattr(run, "_kill_worker_tree", fake_kill)
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") == 137
    row = run.read_candidate(tmp_path, identity(value))
    assert row["candidate_wall_budget_seconds"] == 10.
    assert row["process_watchdog"]["hard_limit_seconds"] == 10.


def test_finished_and_detailed_workers_are_not_killed(tmp_path, monkeypatch):
    value = job()
    monkeypatch.setattr(run.time, "perf_counter", lambda: 999999.)
    monkeypatch.setattr(run, "_kill_worker_tree", lambda p: pytest.fail("unexpected kill"))
    assert run._poll_candidate(tmp_path, identity(value), value, "detailed") is None
    value["process"].code = 0
    assert run._poll_candidate(tmp_path, identity(value), value, "fast") == 0


@pytest.mark.parametrize("search", [False, True])
def test_managers_continue_after_watchdog_kill_without_failure_marker(tmp_path, monkeypatch, search):
    case = run.case_matrix()[0]
    clock, spawned, states = [100.], [], []
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(run, "selected_cases", lambda _: [case])
    monkeypatch.setattr(run, "check_grid", lambda *a: None)
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 1 << 50)
    monkeypatch.setattr(run, "_kill_worker_tree", fake_kill)
    monkeypatch.setattr(run, "refresh", lambda *a, **kw: states.append(kw.get("state")))

    def spawn(*args):
        process = Process()
        spawned.append(process)
        clock[0] += 3001.
        return process

    if search:
        monkeypatch.setattr(run, "case_screen", lambda *a, **kw: {})

        def replay(directory, selected, screen, **_kwargs):
            saved = run.read_candidate(directory, run.candidate_id(case, run.CONTROL_LAYOUT, 1))
            state = "all_candidates_rejected" if saved else "running"
            point = {"status": state}
            run.write_json(directory / "points" / (case["case_id"] + ".json"), point)
            return ({"next": [] if saved else [(run.CONTROL_LAYOUT, 1)]}, [(run.CONTROL_LAYOUT, 1)], point)

        monkeypatch.setattr(run, "search_case", replay)
        assert run.run_search_manager(tmp_path, workers=1, spawn=spawn, poll_seconds=0) == 0
    else:
        assert run.run_manager(tmp_path, workers=1, batches=[1, 2], mode="fast", spawn=spawn, poll_seconds=0) == 0
    assert len(spawned) == (1 if search else 2) and all(p.killed for p in spawned)
    assert states[-1] == "completed_with_rejections"
    assert not (tmp_path / "stop.request").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows venv launcher tree")
def test_windows_kill_targets_the_launcher_tree_and_waits(monkeypatch):
    process = Process()
    seen = []

    def taskkill(command, **kwargs):
        seen.append(command)
        assert kwargs["timeout"] == 10. and kwargs["creationflags"] == run.subprocess.CREATE_NO_WINDOW
        return NS(returncode=0, stderr="", stdout="")

    process.wait = lambda timeout: 137
    monkeypatch.setattr(run.subprocess, "run", taskkill)
    assert run._kill_worker_tree(process) == (137, True)
    assert seen == [["taskkill", "/PID", str(process.pid), "/T", "/F"]]


def test_progress_publishes_current_start_and_explicit_prefill_phase(tmp_path, monkeypatch):
    params = run.parameters(run.case_matrix()[0], run.CONTROL_LAYOUT, 1, "fast")
    clock = [100.]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    writer = run.ProgressWriter(tmp_path, run.params_candidate_id(params))
    writer.begin_candidate(params)
    assert writer.snapshot()["candidate_started_monotonic"] == 100.
    assert writer.snapshot()["watchdog_phase"] == "decode"
    clock[0] = 200.
    writer.begin_prefill()
    observed = run.read_json(writer.path)
    assert observed["watchdog_phase"] == "prefill"
    assert observed["candidate_started_monotonic"] == 100.
    assert observed["watchdog_phase_started_monotonic"] == 200.
    clock[0] = 300.
    writer.begin_prefill()
    assert run.read_json(writer.path)["watchdog_phase_started_monotonic"] == 200.


@pytest.mark.skipif(sys.platform != "win32", reason="Windows venv launcher tree")
def test_real_windows_worker_tree_is_killed_and_saved_as_wall_rejection(tmp_path):
    import subprocess
    import time
    import psutil

    process, owned = None, []
    try:
        # sys.executable is the project's venv Python used to run pytest.
        # Its redirector and sleeping Python child belong only to this test.
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        parent = psutil.Process(process.pid)
        owned.append(parent)
        deadline = time.perf_counter() + 5.
        children = []
        while time.perf_counter() < deadline:
            children = parent.children(recursive=True)
            if children or process.poll() is not None:
                break
            time.sleep(.02)
        owned.extend(children)
        assert process.poll() is None, "the sleeping venv launcher exited before the watchdog"
        assert children, "expected the Windows venv launcher to have a real Python child"

        value = job(started=time.perf_counter() - 1.)
        value["process"] = process
        value["candidate_wall_budget_seconds"] = .01
        code = run._poll_candidate(tmp_path, identity(value), value, "fast")
        assert code is not None and process.poll() == code
        row = run._settle_candidate(tmp_path, value["case"], value["layout"], 1, code, "fast")
        assert row["status"] == "forecast_rejected" and row["metrics"] is None
        assert "wall_budget" in row["rejection_reason"]
        assert row["process_watchdog"]["phase"] == "decode"
        assert row["process_watchdog"]["worker_exit_code"] == code
        assert run.search_plan.classify(row) == "too_big"
        _, alive = psutil.wait_procs(owned, timeout=5.)
        assert not alive, f"owned worker descendants survived: {[item.pid for item in alive]}"
        assert not (tmp_path / "stop.request").exists()
        assert not (tmp_path / "manager.lock").exists()
    finally:
        # Preserve ownership even when the launcher exited before a failure:
        # saved psutil.Process objects still identify its observed children.
        if process is not None and process.poll() is None:
            try:
                for member in psutil.Process(process.pid).children(recursive=True):
                    if member not in owned:
                        owned.append(member)
            except psutil.NoSuchProcess:
                pass
        for member in reversed(owned):
            try:
                member.kill()
            except psutil.NoSuchProcess:
                pass
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.)
        _, alive = psutil.wait_procs(owned, timeout=5.)
        assert not alive, f"test cleanup left owned processes alive: {[item.pid for item in alive]}"
