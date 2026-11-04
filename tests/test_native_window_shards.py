"""A point's windows shared over processes: same result, crashes, stops, the reservation restart.

Run as a script this file is a helper process of the fake window list below.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from heterollm_sim.execution_control import (ExecutionCancelledError, ExecutionControl, WindowCancelledError,
                                             native_callback_free_control)
from heterollm_sim.native_window_shards import WindowShards

WINDOWS = 6


class Underestimated(Exception):
    pass


def _wait_for(path, seconds=60.):
    deadline = time.perf_counter() + seconds
    while not path.exists():
        assert time.perf_counter() < deadline, f"{path} never appeared"
        time.sleep(.02)


def fake_list(shards, control, prepared, plan):
    """``sample(records)`` of a window list whose runs depend on their position and the reservation."""
    pass_directory = lambda: shards.directory / f"pass{shards.passes}"

    def execute(position):
        def run(window_control):
            act = plan.get(f"{'helper' if shards.helper else 'primary'}:{position}", "")
            if act == "crash":
                os._exit(3)
            if act == "raise" or plan.get(f"all:{position}") == "raise":
                raise ValueError(f"window {position}")
            if act == "until_stopped" or (act == "until_cancelled" and prepared["page_staging_occupancy"] == "small"):
                while True:
                    window_control.raise_if_cancelled()
                    time.sleep(.02)
            if act.startswith("after:"):
                _wait_for(pass_directory() / act[6:])
            return {"sample_id": f"w{position}", "wall_seconds": 1., "pid": os.getpid(),
                    "value": (position, prepared.get("page_staging_occupancy"))}
        return run

    def exceeded(record):
        return record["value"] == (plan.get("exceed_at"), "small")

    def sample(records):
        if shards.executing and not shards.helper and "primary_after" in plan:
            _wait_for(pass_directory() / plan["primary_after"])
        for position in range(WINDOWS):
            record = shards.window(f"w{position}", control, execute(position), exceeded)
            if record is not None:
                records.append(record)
                if exceeded(record):
                    raise Underestimated
        return {"windows": len(records)}
    return sample


def start_helper(store, plan, stop=None):
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(store), json.dumps(plan), str(stop or "")],
                            stdin=subprocess.PIPE, cwd=ROOT)


def end_helper(helper):
    helper.stdin.close()
    return helper.wait(timeout=30.)


def values(records):
    return [record["value"] for record in records]


def primary(tmp_path, plan, stop=None):
    import run_exploring_hbf_native as run
    return WindowShards(tmp_path, alive=run.pid_alive), ExecutionControl(stop_request_path=stop), {
        "page_staging_occupancy": "small"}


def test_every_window_runs_once_and_comes_back_in_list_order(tmp_path):
    plan = {}
    helpers = [start_helper(tmp_path, plan) for _ in range(2)]
    shards, control, prepared = primary(tmp_path, plan)
    records = []
    samples = shards.sample(prepared, control, fake_list(shards, control, prepared, plan), records)
    assert all(end_helper(helper) == 0 for helper in helpers)
    assert samples["windows"] == WINDOWS and shards.lost == 0
    assert values(records) == [(position, "small") for position in range(WINDOWS)]
    assert shards.window_seconds == WINDOWS


def test_a_killed_helper_window_is_run_again_by_the_primary(tmp_path):
    plan = {"helper:0": "crash", "primary_after": "0.claim"}
    helper = start_helper(tmp_path, plan)
    shards, control, prepared = primary(tmp_path, plan)
    records = []
    shards.sample(prepared, control, fake_list(shards, control, prepared, plan), records)
    assert helper.wait(timeout=30.) == 3
    assert values(records) == [(position, "small") for position in range(WINDOWS)]
    assert shards.lost == 1 and records[0]["pid"] == os.getpid()


def test_a_window_error_is_raised_at_its_place_in_the_list(tmp_path):
    plan = {"all:2": "raise"}
    helper = start_helper(tmp_path, plan)
    shards, control, prepared = primary(tmp_path, plan)
    records = []
    try:
        shards.sample(prepared, control, fake_list(shards, control, prepared, plan), records)
    except ValueError as error:
        assert str(error) == "window 2"
    else:
        raise AssertionError("the window's error was lost")
    assert end_helper(helper) == 0
    assert values(records) == [(0, "small"), (1, "small")]


def test_stop_request_ends_the_primary_and_its_helpers(tmp_path):
    stop = tmp_path / "stop.request"
    plan = {"helper:0": "until_stopped", "primary_after": "0.claim"}
    store = tmp_path / "store"
    helpers = [start_helper(store, plan, stop) for _ in range(2)]
    shards, control, prepared = primary(store, plan, stop)
    timer = threading.Timer(1., stop.touch)
    timer.start()
    try:
        shards.sample(prepared, control, fake_list(shards, control, prepared, plan), [])
    except ExecutionCancelledError:
        pass
    else:
        raise AssertionError("the primary did not stop")
    timer.join()
    # Without closing their stdin: the stop itself ends them.
    assert all(helper.wait(timeout=30.) == 0 for helper in helpers)


def test_a_too_small_reservation_stops_the_later_windows_and_restarts_every_process(tmp_path):
    # The primary's window 0 outlasts the helper's start on windows 1, 2 and 3,
    # the last one until it is cancelled; window 0 then proves the reservation small.
    plan = {"primary:0": "after:3.claim", "helper:3": "until_cancelled", "exceed_at": 0}
    helper = start_helper(tmp_path, plan)
    shards, control, prepared = primary(tmp_path, plan)
    records = []
    sample = fake_list(shards, control, prepared, plan)
    try:
        shards.sample(prepared, control, sample, records)
    except Underestimated:
        pass
    else:
        raise AssertionError("the first pass went on past its reservation")
    assert values(records) == [(0, "small")]
    first = tmp_path / "pass1"
    _wait_for(first / "3.cancel")
    deadline = time.perf_counter() + 30.
    while (first / "3.claim").exists():
        assert time.perf_counter() < deadline, "the running window was not cancelled"
        time.sleep(.02)
    assert not (first / "3.record").exists() and not (first / "5.record").exists()

    # As simulate_native_stationary does: the next pass runs with the new reservation, everywhere.
    prepared["page_staging_occupancy"] = "large"
    records.clear()
    plan["primary_after"] = "1.claim"
    shards.sample(prepared, control, sample, records)
    assert end_helper(helper) == 0
    assert values(records) == [(position, "large") for position in range(WINDOWS)]
    assert {record["pid"] for record in records} != {os.getpid()}


def test_a_window_cancel_keeps_the_native_control_and_a_stop_wins(tmp_path):
    cancel, stop = tmp_path / "cancel", tmp_path / "stop"
    control = ExecutionControl(stop_request_path=stop, window_cancel_path=str(cancel), window_shards=object())
    assert native_callback_free_control(control)
    control.raise_if_cancelled()
    cancel.touch()
    try:
        control.raise_if_cancelled()
    except WindowCancelledError:
        pass
    else:
        raise AssertionError("the window was not cancelled")
    stop.touch()
    try:
        control.raise_if_cancelled()
    except WindowCancelledError:
        raise AssertionError("a user's stop must win") from None
    except ExecutionCancelledError:
        pass


def _run_with_store_check(monkeypatch, sample):
    from heterollm_sim import native_stationary as stationary

    def execute(source, specs, ids, *, origin_ns, control, sample_id, concurrent_imports=None, row_terms=None):
        return {"sample_id": sample_id, "start_ns": origin_ns, "end_ns": origin_ns + 10., "duration_ns": 10.,
                "boundary_origin_ns": origin_ns, "elapsed_from_boundary_ns": 10., "task_count": len(ids),
                "layer_completion_ns": {}, "first_access_ns": {}, "stage_completion_ns": {"s": 1.},
                "concurrent_imports": None, "wall_seconds": 0.,
                "physical_observations": {"allocator_peak_end_bytes": {"owner": 100}}}

    monkeypatch.setattr(stationary, "_execute_window", execute)
    monkeypatch.setattr(stationary, "page_staging_occupancy", lambda runtime, cohort, peaks: {
        "rank": {"stored_fraction": .5, "shared_bytes_per_stack": peaks["owner"]}})
    common = {"cohort": NS(cohort_id="c", kind="decode"), "ranks": {"t": 0}, "context": {"r": 1}, "source": None,
              "staging_runtime": None, "specs": [NS(task_id="t")],
              "page_staging_occupancy": {"rank": {"stored_fraction": 1., "shared_bytes_per_stack": 10,
                                                  "kv_layers": [], "shared_owner": None}}}
    records = []
    try:
        sample(stationary, common, records)
    except stationary._StagingUnderestimated:
        assert len(records) == 1
    else:
        raise AssertionError("the run past the reservation did not stop the samples")


def test_any_replica_stops_at_the_first_run_past_the_reservation(monkeypatch):
    def plain(stationary, common, records):
        stationary.run_window({**common, "by_layer": {"l": ["t"]}}, ExecutionControl(), records, "calibration",
                              ["l"], (), 0., "w")

    def pipeline(stationary, common, records):
        block = {"roles": {"holdout": ["l"]}, "by_layer": {"l": ["t"]}, "suffix": [], "detached": {},
                 "stage_id": "s", "transfer_stage": None, "commit_stage": None}
        stationary._pipeline_window({**common, "pipeline": {"blocks": {(0, 0): block}}}, ExecutionControl(), records,
                                    "holdout", [(0, 0)], 0., "w")

    _run_with_store_check(monkeypatch, plain)
    _run_with_store_check(monkeypatch, pipeline)


def test_the_exit_run_goes_first_and_the_records_keep_their_order(monkeypatch):
    from heterollm_sim import native_stationary as stationary
    order, records = [], []

    def run_window(prepared, control, windows, role, layer_ids, groups, origin, sample_id, **options):
        order.append(role)
        windows.append(role)
        return {"role": role}

    monkeypatch.setattr(stationary, "run_window", run_window)
    monkeypatch.setattr(stationary, "window_terms", lambda *args, **kwargs: None)
    monkeypatch.setattr(stationary, "_sample_roles", lambda inventory, layers: dict.fromkeys(
        ("calibration", "holdout", "entry", "exit", "pd_import"), ["l"]))
    monkeypatch.setattr(stationary, "_import_sets", lambda runtime, prepared, envelope: (
        None, {"samples": []}, [(["r"], {"request_ids": ["r"]})]))
    prepared = {"inventory": [], "boundary": {}, "record_layers": None, "origin": 0.}
    samples = stationary._sample_windows(None, prepared, None, (1, 1), records, holdout_shift_ns=1.,
                                         pd_concurrency_envelope=True)
    assert order == ["exit", "calibration", "holdout", "entry", "pd_import"]
    assert records == ["calibration", "holdout", "entry", "exit", "pd_import"]
    assert list(samples["runs"]) == ["calibration", "holdout", "entry_at_holdout_origin", "exit_at_calibration_origin"]
    assert samples["runs"]["exit_at_calibration_origin"] == {"role": "exit"}


def test_the_sample_deadline_keeps_the_shared_windows():
    from heterollm_sim import native_stationary as stationary
    shards = object()
    control = stationary.deadline_control(ExecutionControl(window_shards=shards), 5.)
    assert control.window_shards is shards and control.deadline_monotonic == 5.


# The manager's rule.

def test_window_processes_follow_the_wall_the_slots_and_the_memory():
    import run_exploring_hbf_native as run
    rule = run.window_processes
    assert rule(599., 30, 1e12, 1.) == 1
    assert [rule(wall, 30, 1e12, 1.) for wall in (600., 1800., 3600., 86400.)] == [2, 3, 4, 4]
    # Half of the free slots and half of the free memory stay for the next candidates.
    assert rule(3600., 4, 1e12, 1.) == 3 and rule(3600., 1, 1e12, 1.) == 1
    assert rule(3600., 30, 6., 1.) == 4 and rule(3600., 30, 4., 1.) == 3
    assert rule(3600., 30, 1.9, 1.) == 1 and rule(3600., 30, -5., 1.) == 1


def test_admission_counts_all_the_processes_of_a_candidate(tmp_path, monkeypatch):
    import run_exploring_hbf_native as run
    GiB = 2**30
    case = run.case_matrix()[0]
    layout = run.make_layout(*(run.CONTROL_LAYOUT[key] for key in run.LAYOUT_KEYS))
    item = (case, layout, 100)
    known = [(run.parameters(case, layout, 100, "fast"), 4 * GiB, 4000.)]
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES + 40 * GiB)
    assert run.admit([item], {}, known, 32, tmp_path) == ([(item, 16 * GiB, 4)], 32)
    # The detailed mode has no windows; the prediction of a new layout starts it alone.
    assert run.admit([item], {}, known, 32, tmp_path, "detailed") == ([(item, 4 * GiB, 1)], 32)
    assert run.admit([item], {}, [], 32, tmp_path)[0][0][2] == 1
    # Running processes count against the limit, their growth against the memory.
    run.write_json(tmp_path / "progress" / "a.json", {"candidate_id": "a", "working_set_bytes": 2 * GiB})
    active = {"a": {"process": NS(pid=1), "peak": 12 * GiB, "processes": 3}}
    assert run.admit([item], active, known, 3, tmp_path) == ([], 3)
    assert run.admit([item], active, known, 32, tmp_path) == ([(item, 16 * GiB, 4)], 32)
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES + 18 * GiB)
    # 18 - (12 - 3 * 2) leaves 12: 4 for its first process, half of the other 8 for helpers.
    assert run.admit([item], active, known, 32, tmp_path) == ([(item, 8 * GiB, 2)], 32)
    # A candidate the memory or the worker limit holds back keeps everything at one process.
    next_item = (case, layout, 101)
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES + 5 * GiB)
    assert run.admit([item, next_item], {}, known, 32, tmp_path) == ([(item, 4 * GiB, 1)], 32)
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES + 90 * GiB)
    assert run.admit([item, next_item], {}, known, 1, tmp_path) == ([(item, 4 * GiB, 1)], 1)
    # Several starting together share what is left in turn.
    chosen, _factor = run.admit([item, next_item], {}, known, 32, tmp_path)
    assert [processes for _item, _peak, processes in chosen] == [4, 4]
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES + 36 * GiB)
    chosen, _factor = run.admit([item, next_item], {}, known, 32, tmp_path)
    assert [processes for _item, _peak, processes in chosen] == [4, 2]


def test_a_worker_in_its_windows_reserves_only_what_the_windows_move(tmp_path):
    import run_exploring_hbf_native as run
    GiB = 2**30
    job = {"peak": 17 * GiB, "processes": 1}
    path = tmp_path / "progress" / "a.json"

    def growth(job=job, **progress):
        run.write_json(path, {"candidate_id": "a", **progress})
        return run._remaining_growth(tmp_path, "a", job)

    assert run._remaining_growth(tmp_path, "a", job) == 17 * GiB
    assert growth(stage="cohort_lowering", working_set_bytes=6 * GiB) == 11 * GiB
    assert growth(stage="native_layer_tasks", working_set_bytes=6 * GiB) == 2 * GiB
    assert growth(stage="native_stationary_sample", working_set_bytes=16 * GiB) == 1 * GiB
    assert growth(stage="native_layer_tasks", working_set_bytes=18 * GiB) == 0
    # A re-sampled candidate prepares again and takes its whole reservation back.
    assert growth(stage="cohort_tasks", working_set_bytes=6 * GiB) == 11 * GiB
    # A report of the previous candidate of this path says nothing about this one.
    run.write_json(path, {"candidate_id": "old", "stage": "native_layer_tasks", "working_set_bytes": 6 * GiB})
    assert run._remaining_growth(tmp_path, "a", job) == 17 * GiB
    helpers = {"peak": 30 * GiB, "processes": 3}
    assert growth(helpers, stage="native_layer_tasks", working_set_bytes=5 * GiB) == 6 * GiB


def test_a_candidate_starts_with_its_processes_and_shares_the_cores_over_them(tmp_path, monkeypatch):
    import run_exploring_hbf_native as run
    GiB = 2**30
    case = run.case_matrix()[0]
    layout = run.make_layout(*(run.CONTROL_LAYOUT[key] for key in run.LAYOUT_KEYS))
    known = [(run.parameters(case, layout, 100, "fast"), GiB, 4000.)]
    spawned = []
    monkeypatch.setattr(run, "_spawn", lambda *args: spawned.append(args) or NS(pid=1))
    monkeypatch.setattr(run, "MAX_WORKERS", 16)
    monkeypatch.setattr(run, "available_memory_bytes", lambda: run.WORKER_MEMORY_RESERVE_BYTES + 100 * GiB)
    queue, active = [(case, layout, 100)], {"a": {"process": NS(pid=2), "peak": 0, "processes": 4,
                                                  "case": {"hardware": "HBF"}, "hbm_channel_threads": 1}}
    run._start(queue, active, known, 32, tmp_path, "fast", None)
    job = active[run.candidate_id(case, layout, 100)]
    assert not queue and job["processes"] == 4 and job["peak"] == 4 * GiB
    assert job["hbm_channel_threads"] == 3 and spawned[0][-2:] == (3, 4)
    command = run.candidate_command(case, layout, 100, tmp_path, "fast", 1, 4)
    assert command[command.index("--window-processes") + 1] == "4"


def test_hbm4_points_share_the_cores_others_leave_and_other_points_take_their_share(monkeypatch):
    import run_exploring_hbf_native as run
    monkeypatch.setattr(run, "MAX_WORKERS", 32)
    hbm4, hbf = {"hardware": "HBM4"}, {"hardware": "HBF"}
    others = {str(index): {"processes": 1, "case": hbf} for index in range(9)}
    assert run.channel_threads(hbm4, others, 1) == 8
    assert run.channel_threads(hbf, others, 1) == 3
    earlier = {**others, "x": {"processes": 1, "case": hbm4}, "y": {"processes": 1, "case": hbm4}}
    assert run.channel_threads(hbm4, earlier, 1) == 7
    # Small HBM4 points given many threads do not leave a large one with a single thread.
    small = {str(index): {"processes": 1, "case": hbm4, "hbm_channel_threads": 8} for index in range(12)}
    assert run.channel_threads(hbm4, small, 1) == 2
    assert run.channel_threads(hbm4, {str(index): {"processes": 1, "case": hbm4} for index in range(24)}, 1) == 1
    assert run.channel_threads(hbm4, {}, 4) == 8


def test_a_saved_row_gives_the_wall_of_its_windows_in_one_process():
    import run_exploring_hbf_native as run
    case = run.case_matrix()[0]
    layout = run.make_layout(*(run.CONTROL_LAYOUT[key] for key in run.LAYOUT_KEYS))
    params = run.parameters(case, layout, 100, "fast")
    alone = {"parameters": params, "peak_memory_bytes": 1, "wall_seconds": 900., "prefill_wall_seconds": 100.}
    assert run.known_row(alone) == (params, 1, 800.)
    shared = {**alone, "window_shards": {"processes": 3, "lost_windows": 0, "window_seconds": 2000.}}
    assert run.known_row(shared) == (params, 1, 2000.)
    assert run.predicted_wall_seconds([run.known_row(alone), run.known_row(shared)], case, layout, 200) == 4000.
    assert run.predicted_wall_seconds([], case, layout, 200) == 0.
    other = run.make_layout(layout["dp"], layout["tp"] * 2, 1, 1)
    assert run.predicted_wall_seconds([run.known_row(alone)], case, other, 300) == 2400.


# Real processes and a real point.

_PRIMARY = """
import sys, time
sys.path[:0] = [sys.argv[1] + "/scripts", sys.argv[1] + "/src"]
import run_exploring_hbf_native as run
from heterollm_sim.execution_control import ExecutionControl
run.candidate_command = lambda *args, **kwargs: [sys.executable, "-c", "import os; os.read(0, 1)"]
case = run.case_matrix()[0]
layout = run.make_layout(*(run.CONTROL_LAYOUT[key] for key in run.LAYOUT_KEYS))
params = run.parameters(case, layout, 8, "fast")
with run.window_helpers(params, sys.argv[2], None, 3, ExecutionControl(), {}) as control:
    print("ready", flush=True)
    time.sleep(120)
"""


def _primary_tree(tmp_path):
    import psutil
    process = subprocess.Popen([sys.executable, "-c", _PRIMARY, str(ROOT), str(tmp_path)], stdout=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert process.stdout.readline().strip() == b"ready"
    return process, psutil.Process(process.pid).children(recursive=True)


def test_killing_a_worker_tree_leaves_no_helper(tmp_path):
    import psutil
    import run_exploring_hbf_native as run
    process, tree = _primary_tree(tmp_path)
    assert len(tree) >= 3
    _code, killed = run._kill_worker_tree(process)
    assert killed
    _gone, alive = psutil.wait_procs(tree, timeout=10.)
    assert not alive


def test_helpers_end_when_only_the_primary_dies(tmp_path):
    import psutil
    process, tree = _primary_tree(tmp_path)
    # The launcher's real interpreter child alone, not its tree; the helpers below it lose their stdin.
    next(member for member in tree if member.ppid() == process.pid and member.name().startswith("python")).kill()
    process.wait(timeout=10.)
    _gone, alive = psutil.wait_procs(tree, timeout=20.)
    assert not alive


def test_the_helpers_end_with_the_attempt_and_the_row_keeps_their_cost(tmp_path, monkeypatch):
    import run_exploring_hbf_native as run
    monkeypatch.setattr(run, "candidate_command", lambda *args, **kwargs: [
        sys.executable, "-c", "import os; os.read(0, 1)"])
    case = run.case_matrix()[0]
    layout = run.make_layout(*(run.CONTROL_LAYOUT[key] for key in run.LAYOUT_KEYS))
    params = run.parameters(case, layout, 8, "fast")
    row = {}
    base = ExecutionControl()
    with run.window_helpers(params, tmp_path, None, 1, base, row) as control:
        assert control is base
    assert row == {}
    with run.window_helpers(params, tmp_path, None, 3, base, row) as control:
        store = run.window_store(tmp_path, run.params_candidate_id(params))
        assert control.window_shards.directory == store and store.exists()
        assert replace(control, window_shards=None) == base
    assert not store.exists()
    assert row["window_shards"] == {"processes": 3, "lost_windows": 0, "window_seconds": 0.,
                                    "helper_exit_codes": [0, 0]}


def _physical(row):
    """All a row says about the point, without what the wall clock says."""
    def walk(value):
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()
                    if "wall" not in str(key)}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return walk({key: row.get(key) for key in ("status", "rejection_reason", "metrics", "approximation",
                                               "native_window_forecast", "physical_layer_samples")})


def test_the_number_of_processes_changes_nothing_of_a_real_point(tmp_path, monkeypatch):
    """KV staging, attention data parallelism and every window on the HBM and NAND paths."""
    import run_exploring_hbf_native as run
    case = next(row for row in run.case_matrix() if row["case_id"] == "native__llama4_maverick__CONV_plus__LONG__G8")
    params = run.parameters(case, run.make_layout(4, 1, 1, 2), 12, "fast")
    monkeypatch.setenv("HETEROLLM_HBM_CHANNEL_THREADS", "2")

    def no_prefill(params, directory):
        raise ValueError("native_prefill_replica_does_not_fit_hbm4")

    rows = [run.execute_candidate(params, tmp_path / str(processes), NS(report=lambda progress: None),
                                  prefill=no_prefill, window_processes=processes) for processes in (1, 3)]
    assert rows[0]["rejection_reason"] == "native_prefill_placement_infeasible"
    assert rows[0]["physical_layer_samples"]
    assert _physical(rows[0]) == _physical(rows[1])
    assert rows[1]["window_shards"]["processes"] == 3 and rows[1]["window_shards"]["lost_windows"] == 0
    assert not (tmp_path / "3" / "window_store" / rows[1]["candidate_id"]).exists()


if __name__ == "__main__":
    store, plan, stop = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3] or None

    def exit_with_primary():
        try:
            os.read(sys.stdin.fileno(), 1)
        finally:
            os._exit(0)

    threading.Thread(target=exit_with_primary, daemon=True).start()
    shards = WindowShards(store, helper=True)
    control = ExecutionControl(stop_request_path=stop)
    prepared = {}
    try:
        shards.sample(prepared, control, fake_list(shards, control, prepared, plan), [])
    except ExecutionCancelledError:
        pass
