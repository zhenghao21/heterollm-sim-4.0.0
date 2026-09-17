import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("r33_runner_tests", Path(__file__).with_name("run_candidate.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture(autouse=True)
def frozen_observation_budget(monkeypatch):
    monkeypatch.setattr(d.f,"read_protocol",lambda:({"soft_observation_seconds":600},{}))


def test_score_cannot_invoke_scorer_before_full262_barrier(monkeypatch):
    calls = []
    def absent(): raise ValueError("missing full262 barrier")
    monkeypatch.setattr(d, "barrier", absent)
    monkeypatch.setattr(d, "command", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="missing full262"): d.score()
    assert calls == []


def test_full_does_not_run_off_if_on_lock_preflight_failed(monkeypatch):
    calls = []
    def failed(): raise ValueError("on lock preflight failed")
    monkeypatch.setattr(d, "guard", failed)
    monkeypatch.setattr(d, "command", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="on lock preflight failed"): d.full(4, 600)
    assert calls == []


def test_lock_runs_both_frozen_preflights_before_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    monkeypatch.setattr(d, "read_freeze_receipt", lambda: ({"native_lock": "unchanged"}, {}, {}, {}, {}))
    monkeypatch.setattr(d, "verify_native_runtime", lambda _: None)
    monkeypatch.setattr(d.f, "native_lock", lambda: "unchanged")
    calls = []
    def preflight(stage):
        assert not (tmp_path / "controls.json").exists()
        calls.append(stage)
        return {"off": {"status": "passed"}, "on": {"status": "passed"}}
    monkeypatch.setattr(d.f, "run_preflights", preflight)
    monkeypatch.setattr(d, "guard", lambda: calls.append("guard"))
    d.lock()
    controls = json.loads((tmp_path / "controls.json").read_text())
    assert set(controls["lock_preflights"]) == {"off", "on"} and calls == ["lock", "guard"]


def test_failed_lock_preflight_never_writes_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    monkeypatch.setattr(d, "read_freeze_receipt", lambda: ({}, {}, {}, {}, {}))
    def failed(stage): raise ValueError("on failed")
    monkeypatch.setattr(d.f, "run_preflights", failed)
    with pytest.raises(ValueError, match="on failed"): d.lock()
    assert not (tmp_path / "controls.json").exists()


def test_scores_prevent_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    (tmp_path / "on").mkdir(); (tmp_path / "on/errors.0001.json").write_text("{}")
    monkeypatch.setattr(d, "guard", lambda: "valid")
    calls = []; monkeypatch.setattr(d, "command", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="scores exist"): d.full(4, 600)
    assert calls == []


def test_full_preserves_old_terminal_and_waits_for_both131(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    monkeypatch.setattr(d, "guard", lambda: "locked")
    old = tmp_path / "old_failure.json"; old.write_text("original failed result")
    oldref = d.s.reference(old)
    calls = []; finished = set()
    monkeypatch.setattr(d, "retained_terminals", lambda arm: {"old-failed": oldref})
    monkeypatch.setattr(d.s,"read_json",lambda _:({},{}))
    monkeypatch.setattr(d.s,"unique_cells",lambda _:dict.fromkeys(["old-failed", *["new-%03d"%i for i in range(130)]]))
    def command(arm, *args):
        assert "--resume" in args and "--score" not in args
        requested=[args[i+1] for i,v in enumerate(args) if v=="--cell-id"]
        assert len(requested)==130 and "old-failed" not in requested
        assert not (tmp_path / "predictions_complete.json").exists()
        calls.append(arm); finished.add(arm)
    def terminal(arm):
        assert arm in finished
        return {"prediction_refs": {str(i): {} for i in range(131)}}
    monkeypatch.setattr(d, "command", command)
    monkeypatch.setattr(d, "terminal", terminal)
    monkeypatch.setattr(d, "barrier", lambda: None)
    d.full(4, 600)
    barrier = json.loads((tmp_path / "predictions_complete.json").read_text())
    assert barrier["terminal_count"] == 262 and set(barrier["arms"]) == {"off", "on"}
    assert calls == ["off", "on"] and old.read_text() == "original failed result"


def test_on_run_failure_preserves_off_and_cannot_create_barrier(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path); monkeypatch.setattr(d, "guard", lambda: "locked")
    monkeypatch.setattr(d, "retained_terminals", lambda arm: {})
    monkeypatch.setattr(d, "terminal", lambda arm: {})
    monkeypatch.setattr(d.s,"read_json",lambda _:({},{}))
    monkeypatch.setattr(d.s,"unique_cells",lambda _:dict.fromkeys(str(i) for i in range(131)))
    calls = []
    def command(arm, *args):
        calls.append(arm)
        if arm == "on": raise ValueError("worker-start failure")
    monkeypatch.setattr(d, "command", command)
    with pytest.raises(ValueError, match="worker-start"): d.full(4, 600)
    assert calls == ["off", "on"] and not (tmp_path / "predictions_complete.json").exists()


def test_bad_barrier_count_cannot_unlock_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    (tmp_path / "predictions_complete.json").write_text(json.dumps({"schema": "r33-full262-terminal-barrier/v1", "terminal_count": 131, "arms": {"off": {}}}))
    with pytest.raises(ValueError, match="complete262"): d.barrier()


def test_partial_terminal_membership_rejected(monkeypatch):
    monkeypatch.setattr(d.s, "read_json", lambda _: ({}, {}))
    monkeypatch.setattr(d.s, "unique_cells", lambda _: {str(i): {} for i in range(131)})
    monkeypatch.setattr(d, "retained_terminals", lambda _: {str(i): {} for i in range(130)})
    with pytest.raises(ValueError, match="all131"): d.terminal("on")


def test_complete_arm_never_launches_automatic_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(d,"P",tmp_path);monkeypatch.setattr(d,"guard",lambda:"locked")
    ref=d.s.reference(__file__);allcells={str(i):ref for i in range(131)}
    monkeypatch.setattr(d,"retained_terminals",lambda arm:allcells)
    monkeypatch.setattr(d.s,"read_json",lambda _:({},{}));monkeypatch.setattr(d.s,"unique_cells",lambda _:allcells)
    monkeypatch.setattr(d,"terminal",lambda arm:{"prediction_refs":allcells});monkeypatch.setattr(d,"barrier",lambda:None)
    monkeypatch.setattr(d,"command",lambda *args:pytest.fail("completed arm must never launch"))
    d.full(4,600)
    assert json.loads((tmp_path/"predictions_complete.json").read_text())["terminal_count"]==262


def test_existing_terminal_mutation_is_detected_before_barrier(tmp_path,monkeypatch):
    monkeypatch.setattr(d,"P",tmp_path);monkeypatch.setattr(d,"guard",lambda:"locked")
    old=tmp_path/"failed.json";old.write_text("failed terminal");ref=d.s.reference(old)
    monkeypatch.setattr(d,"retained_terminals",lambda arm:{"old":ref})
    monkeypatch.setattr(d.s,"read_json",lambda _:({},{}));monkeypatch.setattr(d.s,"unique_cells",lambda _:{"old":{},"missing":{}})
    monkeypatch.setattr(d,"command",lambda *args:old.write_text("incorrect retry rewrote old terminal"))
    with pytest.raises(ValueError):d.full(4,600)
    assert not (tmp_path/"predictions_complete.json").exists()
