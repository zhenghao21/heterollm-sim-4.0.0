"""R32 two-arm full262 terminal barrier; no scoring before both arms finish."""
from pathlib import Path
import argparse
import importlib.util
import subprocess
import sys

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r32_freezer", P / "freeze_candidate.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
s, ARMS = f.s, f.ARMS
now = f.now


def command(arm, *args):
    f.natural_process([sys.executable, str(P / arm / "source/tools/predict_stable_native_dataset.py"),
        "--output", str(P / arm), *args], check=True, purpose="coordinator-"+arm)


def no_scores():
    s.require(not any(list((P / arm).glob("errors.*.json")) for arm in ARMS), "scores exist; cannot predict again")


def read_freeze_receipt():
    protocol, pr = f.read_protocol()
    receipt, rr = s.read_json(P / "freeze_receipt.json")
    s.require(receipt.get("schema") == "r32-paired-freeze-receipt/v1" and receipt.get("compared_cells") == 131
              and receipt.get("baseline_static_compared_cells") == 131,
        "paired freeze receipt missing or invalid")
    s.require(s.verify_reference(receipt["protocol_ref"]) == pr, "freeze protocol differs")
    s.require(set(receipt["frozen_source_preflights"]) == set(ARMS), "both freeze preflights required")
    baseline, _ = s.read_json(f.BASE)
    freezes = {}
    for arm in ARMS:
        f.validate_preflight("freeze", arm, receipt["frozen_source_preflights"][arm], pr)
        freezes[arm], actual = s.read_json(P / arm / "freeze.json")
        s.require(s.verify_reference(receipt[arm]) == actual, "freeze receipt differs")
        f.check_source_inheritance(baseline, freezes[arm], protocol, arm)
    f.compare_inputs(freezes["off"], freezes["on"])
    f.compare_baseline_off(baseline, freezes["off"])
    return protocol, pr, receipt, rr, freezes


def verify_native_runtime(freezes):
    refs = {}
    for freeze in freezes.values():
        for row in freeze["cells"]:
            inputs = row["static_inputs"]
            for ref in [inputs["runtime_ref"], *inputs.get("runtime_module_refs", [])]:
                value = s.normalized_ref(ref)
                key = value["path"]
                s.require(key not in refs or refs[key] == value, "runtime identity conflict")
                refs[key] = value
    for ref in refs.values():
        s.verify_reference(ref)


def guard():
    controls, cr = s.read_json(P / "controls.json")
    s.require(controls.get("schema") == "r32-two-arm-controls/v1" and controls.get("arms") == list(ARMS)
        and controls.get("denominator") == 131 and controls.get("strict_threshold_pct") == 10, "invalid controls")
    protocol, pr, receipt, rr, freezes = read_freeze_receipt()
    s.require(s.verify_reference(controls["protocol_ref"]) == pr and s.verify_reference(controls["freeze_receipt_ref"]) == rr, "locked protocol/freeze differs")
    s.require(set(controls["lock_preflights"]) == set(ARMS), "both lock preflights required")
    for arm in ARMS:
        f.validate_preflight("lock", arm, controls["lock_preflights"][arm], pr)
        for ref in s.evidence_closure(freezes[arm]):
            s.verify_source_evidence_reference(ref)
        for ref in freezes[arm]["final_output_selection_binding"]["evidence_refs"]:
            s.verify_source_evidence_reference(ref)
    verify_native_runtime(freezes)
    s.require(f.native_lock() == protocol["native_lock"], "fixed native changed")
    return cr


def lock():
    no_scores()
    s.require(not (P / "controls.json").exists(), "controls already exist")
    s.require(not any(list((P / arm / "predictions").glob("*.prediction.json")) for arm in ARMS), "lock must precede predictions")
    protocol, pr, receipt, rr, freezes = read_freeze_receipt()
    preflights = f.run_preflights("lock")  # Fresh frozen-source execution in each arm.
    verify_native_runtime(freezes)
    s.require(f.native_lock() == protocol["native_lock"], "native changed during lock")
    f.write_new(P / "controls.json", {"schema": "r32-two-arm-controls/v1", "created_utc": now(), "arms": list(ARMS),
        "denominator": 131, "strict_threshold_pct": 10, "protocol_ref": pr, "freeze_receipt_ref": rr,
        "lock_preflights": preflights})
    guard()


def check_attempts_closed(arm):
    run_dir = P / arm / "runs"
    s.require(not (run_dir / "coordinator.lock").exists(), "coordinator/live observation lock blocks barrier or resume")
    for attempt in (run_dir / "attempts").glob("*"):
        s.require((attempt / "sealed.json").is_file(), "live/unresolved worker attempt blocks resume")
        start, _ = s.read_json(attempt / "start.json")
        record, _ = s.read_json(P / arm / "predictions" / (start["cell_id"] + ".prediction.json"))
        verify_terminal_execution(arm, record)


def verify_terminal_execution(arm, record):
    ref = s.verify_reference(record.get("worker_execution_ref"))
    location = Path(ref["path"]).resolve()
    s.require(location.name == "execution.json" and location.parent.parent == (P / arm / "runs/attempts").resolve(),
              "execution receipt outside the unique arm attempt")
    execution, actual = s.read_json(location)
    target = P / arm / "predictions" / (record["cell_id"] + ".prediction.json")
    s.require(actual == ref and execution.get("schema") == "stable-native-worker-execution/v1"
              and execution.get("cell_id") == record["cell_id"] and execution.get("freeze_ref") == record.get("freeze_ref")
              and Path(execution["official_result_path"]).resolve() == target.resolve()
              and execution.get("published_status") == record.get("status"), "execution/terminal binding differs")
    s.require(execution.get("wait_policy") == "natural_exit_soft_observation" and execution.get("hard_time_limit_enforced") is False
              and type(execution.get("observation_seconds")) in (int, float) and execution["observation_seconds"] == 600,
              "worker observation policy differs")
    if execution.get("spawned") is True:
        s.require(execution.get("natural_exit_observed") is True and type(execution.get("returncode")) is int,
                  "worker natural exit unresolved")
    else:
        s.require(execution.get("spawned") is False and record.get("status") == "failed", "unspawned worker cannot qualify")
    if record.get("status") == "predicted":
        s.require(execution.get("returncode") == 0 and execution.get("result_identity_valid") is True, "nonzero/invalid worker is not scoreable")
    for key in ("attempt_ref", "child_ref", "raw_result_ref", "soft_deadline_ref"):
        if execution.get(key) is not None:s.verify_reference(execution[key])
    if record.get("status") == "predicted":
        s.require(execution.get("raw_result_ref") is not None and execution.get("child_ref") is not None,
                  "successful terminal lacks raw/child evidence")
        raw, _ = s.read_json(execution["raw_result_ref"]["path"])
        s.require({k:v for k,v in record.items() if k not in ("worker_execution_ref", "content_sha256")} == {k:v for k,v in raw.items() if k != "content_sha256"},
                  "published prediction differs from immutable raw")
    seal, _ = s.read_json(location.with_name("sealed.json"))
    s.require(s.normalized_ref(seal.get("execution_ref")) == ref and s.verify_reference(seal.get("prediction_ref")) == s.reference(target), "terminal was not sealed or changed")
    return execution


def retained_terminals(arm):
    check_attempts_closed(arm)
    freeze, fr = s.read_json(P / arm / "freeze.json")
    cells = s.unique_cells(freeze)
    refs = {}
    for path in (P / arm / "predictions").glob("*.prediction.json"):
        ident = path.name.removesuffix(".prediction.json")
        s.require(ident in cells, "prediction outside frozen131")
        pred, ref = s.read_json(path)
        s.require(pred.get("schema") == "stable-native-cell-prediction/v1" and pred.get("cell_id") == ident
            and pred.get("status") in ("predicted", "failed", "incomplete"), "existing result not terminal; do not retry it")
        s.require(s.normalized_ref(pred["freeze_ref"]) == fr and pred.get("source_sha256") == freeze["source"]["sha256"]
            and pred.get("selection_sha256") == s.SELECTION_SHA, "existing terminal identity mismatch")
        verify_terminal_execution(arm, pred)
        refs[ident] = ref
    return refs


def terminal(arm):
    freeze, fr = s.read_json(P / arm / "freeze.json")
    cells = s.unique_cells(freeze)
    refs = retained_terminals(arm)
    s.require(len(cells) == 131 and set(refs) == set(cells), "all131 terminal predictions required")
    inputs = {"freeze_ref": fr, "prediction_refs": refs}
    s.load_bundle(P / arm, "retained", inputs, full=True)
    return inputs


def barrier():
    record, ref = s.read_json(P / "predictions_complete.json")
    s.require(record.get("schema") == "r32-full262-terminal-barrier/v1" and record.get("terminal_count") == 262
        and set(record.get("arms", {})) == set(ARMS), "complete262 barrier required")
    s.require(s.verify_reference(record["controls_ref"]) == guard(), "barrier controls differ")
    for arm in ARMS:
        check_attempts_closed(arm)
        loaded = s.load_bundle(P / arm, "retained", record["arms"][arm], full=True)
        for pred in loaded["predictions"].values():verify_terminal_execution(arm, pred)
        s.require(all(s.timestamp(pred["finished_utc"]) <= s.timestamp(record["created_utc"]) for pred in loaded["predictions"].values()), "barrier predates terminal prediction")
    return record, ref


def full(workers, timeout):
    before = guard(); no_scores()
    protocol, _ = f.read_protocol()
    s.require(type(timeout) in (int, float) and timeout == protocol["soft_observation_seconds"],
              "full observation deadline differs from frozen600s")
    if (P / "predictions_complete.json").exists():
        barrier(); return
    # guard requires both off/on frozen-source preflights at both freeze and lock.
    for arm in ARMS:
        s.require(guard() == before, "controls changed before arm")
        old = retained_terminals(arm)
        freeze, _ = s.read_json(P / arm / "freeze.json")
        missing = sorted(set(s.unique_cells(freeze)) - set(old))
        if missing:
            # Explicitly constrain the inherited coordinator; existing failed or
            # incomplete terminals are retained, never selected for another try.
            filters = [part for ident in missing for part in ("--cell-id", ident)]
            command(arm, "--resume", "--workers", str(workers), "--timeout-seconds", str(timeout), *filters)
        for ref in old.values():
            s.verify_reference(ref)
        terminal(arm)
        s.require(guard() == before, "controls/native changed after arm")
    inputs = {arm: terminal(arm) for arm in ARMS}
    f.write_new(P / "predictions_complete.json", {"schema": "r32-full262-terminal-barrier/v1", "created_utc": now(),
        "controls_ref": before, "arms": inputs, "terminal_count": 262, "failures_preserved": True,
        "target_errors_used_for_prediction": False, "prior_campaigns": "R30 and earlier not overwritten or retroactively accepted"})
    barrier()


def score():
    record, bref = barrier()  # No scorer can run before this full262 validation.
    results = {}
    for arm in ARMS:
        barrier()
        files = list((P / arm).glob("errors.*.json"))
        s.require(all(path.name == "errors.0001.json" for path in files), "unexpected score sequence")
        if not files:
            command(arm, "--score")
        loaded = s.load_bundle(P / arm, "retained", record["arms"][arm], full=True)
        checked = s.load_score(loaded, "errors.0001.json", loaded["cells"], s.reference(P / arm / "errors.0001.json"))
        s.require(s.timestamp(checked["created_utc"]) >= s.timestamp(record["created_utc"]), "score predates barrier")
        results[arm] = {"score_ref": checked["ref"], "summary": s.summarize(checked["rows"])}
    barrier()
    target = P / "report.json"
    if target.exists():
        prior, _ = s.read_json(target)
        s.require(prior["barrier_ref"] == bref and prior["arms"] == results, "saved report differs")
        return
    f.write_new(target, {"schema": "r32-two-arm-scoring-receipt/v1", "created_utc": now(), "barrier_ref": bref,
        "arms": results, "denominator_per_arm": 131, "gate_B": "unvalidated", "blind": False,
        "formal_success": False, "old_R25_failed_attempt": "preserved and not retroactively accepted"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lock", "full", "score"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=600, help="frozen600s soft observation only; never terminate workers")
    args = parser.parse_args()
    s.require(1 <= args.workers <= 8 and args.timeout_seconds > 0, "invalid execution budget")
    try:
        {"lock": lock, "full": lambda: full(args.workers, args.timeout_seconds), "score": score}[args.phase]()
    except BaseException as error:
        f.write_new(P / (args.phase + "_rejected.%04d.json" % (len(list(P.glob(args.phase + "_rejected.*.json"))) + 1)),
            {"schema":"r32-phase-failure/v1","created_utc":now(),"phase":args.phase,"status":"failed",
             "reason":type(error).__name__+": "+str(error),"partial_results_preserved":True,"automatic_retry":False})
        raise
