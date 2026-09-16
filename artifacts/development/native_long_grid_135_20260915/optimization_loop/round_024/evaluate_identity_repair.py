"""R24 single-arm proof-identity repair. No native execution; no score before full barrier.
Prepare only after root commits reviewed sources. Commands: freeze, lock, full, score, summary.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[4]
LOOP = P.parent
BASELINE = LOOP / "round_023/retained/freeze.json"
OLD_R22 = LOOP / "round_022/cta_issue"
OUTPUT = P / "repaired"
ALLOWED_SOURCE_CHANGES = {"tools/predict_stable_native_dataset.py"}


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = module_at("r24_saved_evidence_helpers", LOOP / "round_023/summarize_ablation.py")
recipe = module_at("r24_r23_recipe", LOOP / "round_023/evaluate_candidate.py")
METRICS = s.METRICS


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return s.reference(path)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def committed_source_gate(commit):
    s.require(isinstance(commit, str) and len(commit) == 40
              and all(c in "0123456789abcdef" for c in commit), "explicit reviewed 40-character commit required")
    s.require(git("rev-parse", "--verify", commit + "^{commit}").decode().strip() == commit, "reviewed commit cannot be resolved")
    # Only the reviewed replacement and driver/tests come from this commit.
    # All other execution Python is copied byte-for-byte from R23 frozen source,
    # so unrelated worktree edits and checkout newline conversion cannot leak in.
    paths = [ROOT / "tools/predict_stable_native_dataset.py", Path(__file__), P / "test_evaluate_identity_repair.py"]
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        try:
            committed = git("show", commit + ":" + relative)
        except subprocess.CalledProcessError as exc:
            raise s.EvidenceError("source not in reviewed commit: " + relative) from exc
        s.require(committed == path.read_bytes(), "reviewed source byte mismatch: " + relative)
    return commit


def build_reviewed_source_snapshot(destination, old, commit):
    """Inherit exact R23 source bytes and replace only the committed P0 file."""
    before = s.source_content(old["freeze"])
    replacement = "tools/predict_stable_native_dataset.py"
    payload = git("show", commit + ":" + replacement)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    inherited, replaced = [], []
    origin = Path(old["freeze"]["source"]["root"])
    for relative in sorted(before):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = origin / relative
        raw = payload if relative == replacement else source.read_bytes()
        with target.open("xb") as stream:
            stream.write(raw)
        actual = s.reference(target)
        if relative != replacement:
            s.require({k: actual[k] for k in ("sha256", "bytes")} == before[relative], "inherited frozen source changed")
            inherited.append({"relative": relative, "source_ref": s.reference(source), "snapshot_ref": actual})
        else:
            replaced.append({"relative": relative, "reviewed_commit": commit, "snapshot_ref": actual})
    s.require(len(replaced) == 1 and set(before) == {p.relative_to(destination).as_posix() for p in destination.rglob("*.py")}, "source snapshot membership differs")
    return {"root": str(destination.resolve()), "provenance": "R23 frozen manifest plus exactly one committed replacement",
            "baseline_freeze_ref": old["freeze_ref"], "inherited": inherited, "replaced": replaced}


def native_lock():
    verifier = module_at("r24_native_lock_verifier", ROOT / "tools/verify_fixed_native.py")
    result = verifier.verify_lock(LOOP / "state.json")
    s.require(result["selection_sha256"] == s.SELECTION_SHA and result["selected_cells"] == 131, "fixed native selection changed")
    return result


def bundle(freeze_path):
    freeze, ref = s.read_json(freeze_path)
    cells = s.unique_cells(freeze)
    s.require(freeze.get("selected_denominator") == 131 and freeze.get("selection_sha256") == s.SELECTION_SHA, "fixed 131 selection differs")
    return {"variant": "retained", "directory": Path(freeze_path).parent, "freeze": freeze,
            "freeze_ref": ref, "cells": cells, "ids": s.anchors(cells)}


def recipe_arguments(baseline):
    return recipe.freeze_arguments(baseline, "retained")


def normalized_proof(value, extractor_ref):
    """Normalize only the three reviewed representation-only proof fields.

    No recursive reference rewriting is permitted here: an unknown nested object,
    even if it looks like a file reference, is evidence and must compare exactly.
    """
    value = deepcopy(value)
    expected_extractor = s.normalized_ref(extractor_ref)

    # The extractor is copied into the candidate source snapshot.  Its location
    # may change, but only when its verified digest and exact byte count agree.
    current_extractor = value.get("extractor_ref")
    if isinstance(current_extractor, dict):
        normalized = s.normalized_ref(current_extractor)
        if (normalized["sha256"], normalized["bytes"]) == (expected_extractor["sha256"], expected_extractor["bytes"]):
            normalized["path"] = "copied-reviewed-extractor"
        value["extractor_ref"] = normalized

    # This is the sole model-reference location in the retained proof schema.
    # It permits only bytes/size_bytes spelling normalization; its path remains
    # evidence and cannot be substituted from a matching digest.
    model_scope = value.get("model_scope")
    if isinstance(model_scope, dict) and isinstance(model_scope.get("model_ref"), dict):
        model_scope["model_ref"] = s.normalized_ref(model_scope["model_ref"])

    # The contract digest is derived from the preceding representation-only
    # changes.  Do not remove similarly named fields from nested/unknown schemas.
    contract = value.get("contract")
    if isinstance(contract, dict) and contract.get("schema") == "heterollm.retained-kv-state/v1":
        contract.pop("evidence_sha256", None)
    return value

def verify_semantics(old, new):
    old_sources, new_sources = s.source_content(old["freeze"]), s.source_content(new["freeze"])
    s.require(set(old_sources) == set(new_sources), "source file membership changed")
    changed = {key for key in old_sources if old_sources[key] != new_sources[key]}
    s.require(changed == ALLOWED_SOURCE_CHANGES, "source changes exceed reviewed proof repair: " + str(sorted(changed)))
    s.require(s.static_variant_semantics(old) == s.static_variant_semantics(new), "cost/static inputs changed")
    counts = {"conditional": 0, "uncovered": 0}
    for ident in old["cells"]:
        a = old["cells"][ident]["static_inputs"]["retained_kv_warmup_evidence"]
        b = new["cells"][ident]["static_inputs"]["retained_kv_warmup_evidence"]
        s.require(a["extractor_ref"]["sha256"] == b["extractor_ref"]["sha256"], "extractor bytes changed")
        s.require(normalized_proof(a, a["extractor_ref"]) == normalized_proof(b, b["extractor_ref"]), "retained semantics changed: " + ident)
        counts[b["status"]] += 1
    s.require(counts == {"conditional": 62, "uncovered": 69}, "62/69 qualification changed")
    return {"changed_source_paths": sorted(changed), "source_content_sha256": s.stable_hash(new_sources), "source_file_count": len(new_sources), "scope_counts": counts}


def verify_frozen_closure(item, *, whole_models=False):
    s.verify_reference(item["freeze_ref"])
    s.source_content(item["freeze"])
    s.verify_reference(item["freeze"]["selection_ref"])
    for ref in s.evidence_closure(item["freeze"]):
        s.verify_source_evidence_reference(ref)
    native = {}
    for cell in item["cells"].values():
        inputs = cell["static_inputs"]
        for ref in [inputs["runtime_ref"], *inputs.get("runtime_module_refs", [])]:
            key = s.normalized_ref(ref)["path"]
            if key in native:
                s.require(native[key] == s.normalized_ref(ref), "conflicting native module identity")
            native[key] = s.normalized_ref(ref)
    for ref in native.values():
        s.verify_reference(ref)
    if whole_models:
        refs = item["freeze"].get("retained_kv_warmup", {}).get("model_identity_refs")
        s.require(isinstance(refs, list) and bool(refs), "repaired freeze lacks full-model identity closure")
        identities = {}
        for ref in refs:
            actual = s.normalized_ref(ref)
            s.require(actual["path"] not in identities, "duplicate model identity reference")
            identities[actual["path"]] = actual
        expected = {s.normalized_ref(cell["static_inputs"]["prediction_model_ref"])["path"]:
                    s.normalized_ref(cell["static_inputs"]["prediction_model_ref"]) for cell in item["cells"].values()}
        s.require(identities == expected, "full-model identity coverage differs")
        for ref in identities.values():
            print("R24 phase gate: full model SHA256 " + ref["path"], flush=True)
            s.verify_reference(ref)


def historical_inputs(directory):
    old = bundle(Path(directory) / "freeze.json")
    ids = set(old["cells"])
    files = {p.stem.removesuffix(".prediction"): p for p in (Path(directory) / "predictions").glob("*.prediction.json")}
    s.require(set(files) == ids, "historical terminal set incomplete")
    # Byte references only: historical target errors never enter prediction inputs.
    return {"freeze_ref": old["freeze_ref"], "prediction_refs": {ident: s.reference(files[ident]) for ident in sorted(ids)}}


def script_paths():
    return [Path(__file__), P / "test_evaluate_identity_repair.py", LOOP / "round_023/evaluate_candidate.py",
            LOOP / "round_023/summarize_ablation.py", ROOT / "tools/verify_fixed_native.py"]


def protocol_only():
    protocol, ref = s.read_json(P / "identity_repair_protocol.json")
    s.require(protocol.get("schema") == "retained-identity-repair-protocol/v1" and protocol.get("full_denominator") == 131
              and protocol.get("threshold_pct_strict") == 10.0 and protocol.get("metrics") == list(METRICS), "protocol invariants changed")
    s.require(protocol.get("coefficient_changes") == 0 and protocol.get("numeric_semantics_changes") == 0
              and protocol.get("gate_B") == "unvalidated" and protocol.get("candidate") == "repaired", "repair scope changed")
    for reference in protocol["scripts"]:
        s.verify_reference(reference)
    s.require({r["path"] for r in protocol["scripts"]} == {str(p.resolve()) for p in script_paths()}, "script closure changed")
    s.verify_reference(protocol["baseline_freeze_ref"])
    return protocol, ref


def freeze(reviewed_commit):
    committed_source_gate(reviewed_commit)
    s.require(not OUTPUT.exists() and not (P / "identity_repair_protocol.json").exists(), "new repair freeze only; existing evidence is never adopted or overwritten")
    start = native_lock()
    old = bundle(BASELINE)
    verify_frozen_closure(old)
    history = {"r23": historical_inputs(BASELINE.parent), "r22": historical_inputs(OLD_R22)}
    prior = [LOOP / rel for rel in ("round_022/pure/freeze.json", "round_022/current/freeze.json", "round_022/cta_issue/freeze.json", "round_023/current/freeze.json")]
    protocol = {"schema": "retained-identity-repair-protocol/v1", "created_utc": now(), "reviewed_commit": reviewed_commit,
        "candidate": "repaired", "variants": ["repaired"], "full_denominator": 131, "threshold_pct_strict": 10.0,
        "metrics": list(METRICS), "treatments": s.TREATMENTS["retained"], "coefficient_changes": 0, "numeric_semantics_changes": 0,
        "only_change": "canonical retained model reference and source-aware cache identity", "gate_B": "unvalidated",
        "blind_evaluation": False, "formal_acceptance": False, "native_remeasurement": False, "calibration_added": False,
        "baseline_freeze_ref": old["freeze_ref"], "native_lock": start, "scripts": [s.reference(p) for p in script_paths()],
        "historical_predictions": history, "prior_control_freezes": [s.reference(p) for p in prior],
        "paired_checks": {"r23_predicted_expected": 104, "r23_failed_expected": 27, "r22_common_gpu_predicted_expected": 25,
            "r22_previous_sha_failures_expected": 2, "numeric_comparison": "exact saved aggregate and request numeric fields; no tolerances or timing calibration"},
        "execution_source_provenance": "inherit exact R23 frozen source manifest; replace only committed tools/predict_stable_native_dataset.py; unrelated worktree is excluded",
        "workflow": "one repaired full131; preserve terminal failures; durable terminal barrier before score; immutable receipts; no terminal retry"}
    write_new(P / "identity_repair_protocol.json", protocol)
    support = P / "identity_repair_support"
    support.mkdir()
    copies = []
    for index, path in enumerate(script_paths()):
        destination = support / (str(index) + "_" + path.name)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        copies.append(s.reference(destination))
    execution_source = build_reviewed_source_snapshot(support / "execution_source", old, reviewed_commit)
    write_new(support / "manifest.json", {"scripts": copies, "originals": protocol["scripts"], "execution_source": execution_source})
    execution_root = Path(execution_source["root"])
    # This module's ROOT is the isolated reviewed snapshot, not the worktree.
    # Its normal source_freeze therefore copies only those inherited/reviewed bytes.
    frozen_api = module_at("r24_reviewed_prediction_api", execution_root / "tools/predict_stable_native_dataset.py")
    for name, imported in list(sys.modules.items()):
        if name == "tools" or name.startswith("tools.") or name == "heterollm_sim" or name.startswith("heterollm_sim."):
            location = getattr(imported, "__file__", None)
            if location is not None:
                s.require(Path(location).resolve().is_relative_to(execution_root.resolve()), "import outside reviewed snapshot: " + name)
    frozen_api.freeze_selection(Path(old["freeze"]["selection_ref"]["path"]), OUTPUT, **recipe_arguments(old["freeze"]))
    verify_semantics(old, bundle(OUTPUT / "freeze.json"))
    s.require(native_lock() == start, "native lock changed during freeze")


def reject_scores():
    s.require(not list(OUTPUT.glob("errors.*.json")), "scores exist; prediction phase cannot resume")


def lock():
    reject_scores()
    s.require(not list((OUTPUT / "predictions").glob("*.prediction.json")), "controls must lock before predictions")
    protocol, pr = protocol_only()
    old, new = bundle(BASELINE), bundle(OUTPUT / "freeze.json")
    verify_frozen_closure(old); verify_frozen_closure(new)
    checked = verify_semantics(old, new)
    copies, cr = s.read_json(P / "identity_repair_support/manifest.json")
    s.require(copies["originals"] == protocol["scripts"] and len(copies["scripts"]) == len(copies["originals"]), "support originals/copy coverage changed")
    for original, copied in zip(copies["originals"], copies["scripts"]):
        actual = s.verify_reference(copied)
        s.require((actual["sha256"], actual["bytes"]) == (original["sha256"], original["bytes"]), "support copy differs")
    source_snapshot_refs = [row["snapshot_ref"] for row in copies["execution_source"]["inherited"] + copies["execution_source"]["replaced"]]
    for ref in source_snapshot_refs:
        s.verify_reference(ref)
    payload = {"schema": "retained-identity-repair-controls/v1", "created_utc": now(), "protocol_ref": pr,
        "freeze_ref": new["freeze_ref"], "support_manifest_ref": cr,
        "support_refs": copies["scripts"] + source_snapshot_refs, **checked}
    write_new(P / "identity_repair_controls.json", payload)
    guard()


def guard():
    protocol, pr = protocol_only()
    controls, cref = s.read_json(P / "identity_repair_controls.json")
    s.require(controls.get("schema") == "retained-identity-repair-controls/v1" and s.verify_reference(controls["protocol_ref"]) == pr, "control protocol differs")
    s.verify_reference(controls["support_manifest_ref"])
    for ref in controls["support_refs"]:
        s.verify_reference(ref)
    new, old = bundle(OUTPUT / "freeze.json"), bundle(BASELINE)
    s.require(new["freeze_ref"] == s.verify_reference(controls["freeze_ref"]), "frozen candidate changed")
    verify_frozen_closure(new, whole_models=True); verify_frozen_closure(old)
    s.require(verify_semantics(old, new)["source_content_sha256"] == controls["source_content_sha256"], "source closure differs")
    for data in protocol["historical_predictions"].values():
        s.verify_reference(data["freeze_ref"])
        for ref in data["prediction_refs"].values():
            s.verify_reference(ref)
    for ref in protocol["prior_control_freezes"]:
        s.verify_reference(ref)
    current_native = native_lock()
    s.require(current_native == protocol["native_lock"], "native identity changed")
    return {"protocol": protocol, "controls": controls, "controls_ref": cref, "bundle": new, "native_lock": current_native}


def check_end(before):
    after = guard()
    s.require(after["controls_ref"] == before["controls_ref"] and after["native_lock"] == before["native_lock"], "phase control/native identity changed")


def run(args):
    result = subprocess.call([str(arg) for arg in args])
    s.require(result == 0, "frozen command failed with exit " + str(result))


def terminal_refs(item, *, complete):
    files = {p.stem.removesuffix(".prediction"): p for p in (item["directory"] / "predictions").glob("*.prediction.json")}
    expected = set(item["cells"])
    s.require(set(files) <= expected, "unexpected prediction outside fixed selection")
    if complete:
        s.require(set(files) == expected, "131 terminal predictions required before barrier")
    refs = {}
    for ident, path in files.items():
        value, ref = s.read_json(path)
        s.require(value.get("cell_id") == ident and value.get("status") in {"predicted", "failed", "incomplete"}, "existing nonterminal/corrupt result cannot be retried")
        s.require(s.normalized_ref(value["freeze_ref"]) == item["freeze_ref"] and value.get("source_sha256") == item["freeze"]["source"]["sha256"], "prediction freeze/source identity differs")
        refs[ident] = ref
    return refs


def load_barrier(before):
    barrier, bref = s.read_json(P / "full_predictions.json")
    s.require(barrier.get("schema") == "r24-terminal-prediction-barrier/v1" and barrier.get("terminal_count") == 131, "invalid terminal barrier")
    s.require(s.verify_reference(barrier["controls_ref"]) == before["controls_ref"], "barrier controls differ")
    loaded = s.load_bundle(OUTPUT, "retained", barrier["inputs"], full=True)
    s.require(all(s.timestamp(p["finished_utc"]) <= s.timestamp(barrier["created_utc"]) for p in loaded["predictions"].values()), "barrier preceded terminal prediction")
    return barrier, bref, loaded


def full(workers=4, timeout=600):
    before = guard(); reject_scores()
    if (P / "full_predictions.json").exists():
        load_barrier(before); check_end(before); return
    retained = terminal_refs(before["bundle"], complete=False)
    run([sys.executable, OUTPUT / "source/tools/predict_stable_native_dataset.py", "--output", OUTPUT,
         "--resume", "--workers", workers, "--timeout-seconds", timeout])
    for ref in retained.values():
        s.verify_reference(ref)
    refs = terminal_refs(before["bundle"], complete=True)
    inputs = {"freeze_ref": before["bundle"]["freeze_ref"], "prediction_refs": refs}
    s.load_bundle(OUTPUT, "retained", inputs, full=True)
    check_end(before)
    write_new(P / "full_predictions.json", {"schema": "r24-terminal-prediction-barrier/v1", "created_utc": now(),
        "terminal_count": 131, "controls_ref": before["controls_ref"], "inputs": inputs,
        "failures_preserved": True, "native_answers_used_for_prediction": False})
    load_barrier(before)


def score():
    before = guard()
    barrier, bref, loaded = load_barrier(before)  # Must happen before invoking scorer.
    s.require(not any(p.name != "errors.0001.json" for p in OUTPUT.glob("errors.*.json")), "unexpected score sequence")
    path = OUTPUT / "errors.0001.json"
    if not path.exists():
        run([sys.executable, OUTPUT / "source/tools/predict_stable_native_dataset.py", "--output", OUTPUT, "--score"])
    checked = s.load_score(loaded, path.name, loaded["cells"], s.reference(path))
    s.require(s.timestamp(checked["created_utc"]) >= s.timestamp(barrier["created_utc"]), "score predates durable full barrier")
    receipt = P / "full_scores.json"
    payload = {"schema": "r24-score-receipt/v1", "created_utc": now(), "controls_ref": before["controls_ref"],
               "barrier_ref": bref, "score_ref": checked["ref"]}
    if receipt.exists():
        old, _ = s.read_json(receipt)
        for key in ("controls_ref", "barrier_ref", "score_ref"):
            s.require(old[key] == payload[key], "score receipt differs")
    else:
        write_new(receipt, payload)
    load_barrier(before); check_end(before)


def numeric_projection(prediction):
    def numbers(node):
        if isinstance(node, dict):
            return {key: numbers(value) for key, value in node.items()
                    if isinstance(value, (dict, list, int, float)) and not isinstance(value, bool)}
        if isinstance(node, list):
            return [numbers(value) for value in node]
        return node
    return {"aggregate": numbers(prediction.get("aggregate", {})),
            "requests": [{"request_id": r.get("request_id"), **numbers(r)} for r in prediction.get("requests", [])]}


def paired_checks(new, r23, r22):
    expected104 = {i for i, p in r23.items() if p["status"] == "predicted"}
    old27 = set(r23) - expected104
    s.require(len(expected104) == 104 and len(old27) == 27, "R23 historical 104/27 changed")
    common25 = {i for i in old27 if r22[i]["status"] == "predicted"}
    old2 = old27 - common25
    s.require(len(common25) == 25 and len(old2) == 2 and all(r23[i]["model_key"] == "qwen38_gpu" for i in old27), "R22 GPU comparison membership changed")
    def compare(ids, old):
        rows = [{"cell_id": i, "new_status": new[i]["status"], "numeric_exact_match":
                 new[i]["status"] == "predicted" and numeric_projection(new[i]) == numeric_projection(old[i])} for i in sorted(ids)]
        return {"fixed_comparison_count": len(ids), "exact_matches": sum(r["numeric_exact_match"] for r in rows), "rows": rows}
    return {"r23_previous_104": compare(expected104, r23), "r22_common_gpu_25": compare(common25, r22),
            "r23_failed_27_now_predicted": sum(new[i]["status"] == "predicted" for i in old27),
            "r22_sha_failures_not_retroactively_accepted": sorted(old2)}


def summary():
    before = guard(); barrier, bref, loaded = load_barrier(before)
    receipt, _ = s.read_json(P / "full_scores.json")
    s.require(s.verify_reference(receipt["barrier_ref"]) == bref and s.verify_reference(receipt["controls_ref"]) == before["controls_ref"], "score receipt identity differs")
    checked = s.load_score(loaded, "errors.0001.json", loaded["cells"], receipt["score_ref"])
    s.require(s.timestamp(checked["created_utc"]) >= s.timestamp(barrier["created_utc"]), "score predates barrier")
    history = before["protocol"]["historical_predictions"]
    r23 = s.load_bundle(BASELINE.parent, "retained", history["r23"], full=True)
    r22 = s.load_bundle(OLD_R22, "retained", history["r22"], full=True)
    pairs = paired_checks(loaded["predictions"], r23["predictions"], r22["predictions"])
    totals = s.summarize(checked["rows"])
    report = {"schema": "retained-identity-repair-report/v1", "created_utc": now(), "controls_ref": before["controls_ref"],
        "barrier_ref": bref, "score_ref": checked["ref"], "fixed_denominator": 131, "candidate": totals,
        "paired_identity_repair_checks": pairs, "gate_A": "passed" if totals["strict_all3_below10_cells"] == 131 else "not_passed",
        "gate_B": "unvalidated", "formal_success": False, "blind_evaluation": False, "coefficient_changes": 0,
        "numeric_semantics_changes": 0, "prior_controls": before["protocol"]["prior_control_freezes"],
        "interpretation": "Execution/proof repair only; old failures preserved, no causal cost improvement or independent validation claim."}
    check_end(before)
    write_new(P / "identity_repair_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "lock", "full", "score", "summary"))
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    s.require(1 <= args.workers <= 8 and args.timeout_seconds > 0, "invalid execution budget")
    if args.phase == "freeze":
        freeze(args.reviewed_commit)
    elif args.phase == "lock":
        lock()
    elif args.phase == "full":
        full(args.workers, args.timeout_seconds)
    elif args.phase == "score":
        score()
    else:
        print(json.dumps(summary(), ensure_ascii=False))


if __name__ == "__main__":
    main()
