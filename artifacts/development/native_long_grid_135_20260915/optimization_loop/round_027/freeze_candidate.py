"""R27 same-source off/on freeze; replaces two committed adapters, never predicts."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import subprocess
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[4]
LOOP = P.parent
BASE = LOOP / "round_025/off/freeze.json"
ARMS = ("off", "on")
REPLACEMENTS = {"tools/predict_stable_native_dataset.py", "tools/native_final_output_binding.py"}
SOURCE_COUNT = 115
CONFIG_KEYS = ("batch", "ubatch", "parallel", "ctx", "gpu_layers", "op_offload", "flash_attn", "seed", "output", "expected_prompt_tokens")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = load("r27_saved_evidence", LOOP / "round_023/summarize_ablation.py")
r = load("r27_reviewed_recipe", LOOP / "round_023/evaluate_candidate.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return s.reference(path)


def blob(commit, relative):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + relative])


def helper_paths():
    return [*(P / name for name in ("freeze_candidate.py", "run_candidate.py", "summarize_groups.py", "verify_frozen_preflight.py",
            "test_freeze_candidate.py", "test_run_candidate.py", "test_summarize_groups.py")),
            LOOP / "round_023/summarize_ablation.py", LOOP / "round_023/evaluate_candidate.py", ROOT / "tools/verify_fixed_native.py"]


def check_commit(commit):
    s.require(isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "full reviewed commit required")
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--verify", commit + "^{commit}"], text=True).strip()
    s.require(actual == commit, "reviewed commit unresolved")
    for path in helper_paths():
        s.require(blob(commit, path.relative_to(ROOT).as_posix()) == path.read_bytes(), "helper differs from reviewed commit: " + str(path))


def snapshot(commit, baseline, destination):
    before = s.source_content(baseline)
    s.require(len(before) == SOURCE_COUNT and REPLACEMENTS <= set(before), "baseline must contain exact 115-file source membership")
    destination.mkdir(parents=True, exist_ok=False)
    refs = []
    for name in sorted(before):
        data = blob(commit, name) if name in REPLACEMENTS else (Path(baseline["source"]["root"]) / name).read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        actual = s.reference(target)
        if name not in REPLACEMENTS:
            s.require({key: actual[key] for key in ("sha256", "bytes")} == before[name], "inherited source differs")
        refs.append({"relative": name, "origin": "reviewed_commit" if name in REPLACEMENTS else "R25_off_frozen", "ref": actual})
    return refs


def normalize_known_copies(inputs, freeze):
    """Only three reviewed arm-local differences; no recursive exclusions."""
    result = deepcopy(inputs)
    retained = result["retained_kv_warmup_evidence"]
    extractor = s.verify_reference(retained["extractor_ref"])
    expected = Path(freeze["source"]["root"]) / "tools/retained_warmup_extractor.py"
    s.require(extractor == s.reference(expected), "extractor must be exact reviewed arm-source copy")
    contract = retained.get("contract")
    s.require(result.get("retained_kv_warmup_contract") == contract, "retained contract aliases differ")
    if contract is not None:
        evidence = {key: value for key, value in retained.items() if key != "contract"}
        s.require(contract.get("evidence_sha256") == s.stable_hash(evidence), "retained derived evidence digest invalid")
        contract["evidence_sha256"] = "verified_arm_evidence_digest"
        result["retained_kv_warmup_contract"]["evidence_sha256"] = "verified_arm_evidence_digest"
    retained["extractor_ref"]["path"] = "verified_arm_extractor_copy"
    issue = result["mmvq_issue_evidence"]
    document = issue["hardware_document"]["ref"]
    actual = s.verify_reference(document)
    s.require(actual == s.verify_reference(freeze["mmvq_issue_bound"]["hardware_document"]["ref"]), "hardware PDF differs from campaign")
    matches = [ref for ref in issue["evidence_refs"] if ref.get("path") == document["path"]]
    s.require(len(matches) == 1 and s.verify_reference(matches[0]) == actual, "hardware PDF evidence alias differs")
    document["path"] = "verified_arm_hardware_document_copy"
    matches[0]["path"] = "verified_arm_hardware_document_copy"
    return result


def compare_inputs(off, on):
    a, b = s.unique_cells(off), s.unique_cells(on)
    s.require(len(a) == 131 and set(a) == set(b), "fixed131 cell membership differs")
    s.require(s.source_content(off) == s.source_content(on), "two arms must share all source bytes")
    for ident in a:
        s.require(a[ident].get("preparation_error") is None and b[ident].get("preparation_error") is None, "preparation error blocks paired preflight: " + ident)
        x, y = deepcopy(a[ident]["static_inputs"]), deepcopy(b[ident]["static_inputs"])
        s.require("final_output_selection" not in x and "final_output_selection_binding" not in x, "off must preserve default inputs")
        s.require(y.pop("final_output_selection", None) is True, "on switch missing")
        proof = y.pop("final_output_selection_binding", None)
        s.require(isinstance(proof, dict) and proof.get("status") in ("conditional", "uncovered"), "on proof missing")
        s.require(proof.get("config") == {key: y["config"].get(key) for key in CONFIG_KEYS}, "proof was not bound to actual normalized inputs: " + ident)
        s.require(normalize_known_copies(x, off) == normalize_known_copies(y, on), "non-treatment static difference: " + ident)
    return len(a)


def compare_baseline_off(baseline, off):
    """R25/off actual static inputs remain exact after the three known copies."""
    before, after = s.unique_cells(baseline), s.unique_cells(off)
    s.require(len(before) == 131 and set(before) == set(after), "R25/off fixed131 membership changed")
    for ident in before:
        a, b = before[ident], after[ident]
        s.require(a.get("preparation_error") is None and b.get("preparation_error") is None, "baseline/off preparation changed")
        s.require(a.get("model_key") == b.get("model_key") and a.get("deployment") == b.get("deployment"), "baseline grouping changed")
        for cell in (a, b):
            s.require("final_output_selection" not in cell["static_inputs"]
                      and "final_output_selection_binding" not in cell["static_inputs"], "baseline/new off treatment must be absent")
        s.require(normalize_known_copies(a["static_inputs"], baseline) == normalize_known_copies(b["static_inputs"], off),
                  "R25/off static input changed: " + ident)
    return len(before)


def check_source_inheritance(baseline, candidate, protocol):
    before, after = s.source_content(baseline), s.source_content(candidate)
    expected = {row["relative"]: {k: row["ref"][k] for k in ("sha256", "bytes")} for row in protocol["source_refs"]}
    s.require(len(expected) == SOURCE_COUNT and set(before) == set(after) == set(expected), "source membership differs")
    s.require(after == expected, "candidate differs from reviewed snapshot")
    s.require(all(before[name] == after[name] for name in set(before) - REPLACEMENTS), "unreviewed source change")


def native_lock():
    verifier = load("r27_native_lock", ROOT / "tools/verify_fixed_native.py")
    result = verifier.verify_lock(LOOP / "state.json")
    s.require(result["selected_cells"] == 131 and result["selection_sha256"] == s.SELECTION_SHA, "fixed native changed")
    return result


def read_protocol():
    protocol, protocol_ref = s.read_json(P / "protocol.json")
    s.require(protocol.get("schema") == "r27-final-selection-normalized-inputs/v1" and protocol.get("arms") == list(ARMS)
              and protocol.get("denominator") == 131 and protocol.get("strict_threshold_pct") == 10
              and protocol.get("gate_B") == "unvalidated" and protocol.get("new_cost_coefficients") == 0, "protocol invariants differ")
    s.require(protocol["baseline_ref"]["path"] == str(BASE.resolve()), "baseline path differs")
    s.verify_reference(protocol["baseline_ref"])
    s.require({ref["path"] for ref in protocol["helpers"]} == {str(path.resolve()) for path in helper_paths()}, "helper membership differs")
    for ref in protocol["helpers"]:
        s.verify_reference(ref)
    for row in protocol["source_refs"]:
        s.verify_reference(row["ref"])
    s.require(len(protocol["helper_copies"]) == len(protocol["helpers"]), "helper copy coverage differs")
    for original, copied in zip(protocol["helpers"], protocol["helper_copies"]):
        actual = s.verify_reference(copied)
        s.require((actual["sha256"], actual["bytes"]) == (original["sha256"], original["bytes"]), "helper copy bytes differ")
    return protocol, protocol_ref


def validate_preflight(stage, arm, reference, protocol_ref):
    receipt, actual = s.read_json(reference["path"])
    s.require(actual == s.verify_reference(reference), "preflight receipt changed")
    s.require(receipt.get("schema") == "r27-frozen-source-preflight/v1" and receipt.get("stage") == stage
              and receipt.get("arm") == arm and receipt.get("status") == "passed" and receipt.get("verified_cells") == 131,
              "both131 frozen-source preflights must pass")
    s.require(s.verify_reference(receipt["freeze_ref"]) == s.reference(P / arm / "freeze.json"), "preflight freeze identity differs")
    s.require(s.verify_reference(receipt["protocol_ref"]) == protocol_ref, "preflight protocol differs")
    s.require(s.verify_reference(receipt["verifier_ref"]) == s.reference(P / "verify_frozen_preflight.py"), "preflight helper changed")
    for key, relative in (("actual_api", "tools/predict_stable_native_dataset.py"), ("actual_binding_helper", "tools/native_final_output_binding.py")):
        s.require(s.verify_reference(receipt[key]) == s.reference(P / arm / "source" / relative), "preflight used different frozen helper")
    return receipt


def run_preflights(stage):
    _, protocol_ref = read_protocol()
    references, failed = {}, []
    directory = P / "preflight"
    directory.mkdir(exist_ok=True)
    for arm in ARMS:
        receipt = directory / (stage + "_" + arm + ".json")
        log = directory / (stage + "_" + arm + ".log")
        s.require(not receipt.exists(), "preflight attempt already exists; preserve it")
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, str(P / "verify_frozen_preflight.py"), "--freeze", str(P / arm / "freeze.json"),
                "--protocol", str(P / "protocol.json"), "--stage", stage, "--arm", arm, "--receipt", str(receipt)],
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode or not receipt.exists():
            failed.append(arm)
        else:
            references[arm] = s.reference(receipt)
            validate_preflight(stage, arm, references[arm], protocol_ref)
    s.require(not failed and set(references) == set(ARMS), "frozen-source preflight failed: " + ",".join(failed))
    return references


def freeze(commit):
    check_commit(commit)
    s.require(not (P / "protocol.json").exists() and not any((P / arm).exists() for arm in ARMS), "new freeze only; old attempts are never adopted")
    baseline, base_ref = s.read_json(BASE)
    s.require(baseline.get("selection_sha256") == s.SELECTION_SHA and baseline.get("final_output_selection", False) is False, "baseline must be fixed R25/off")
    native = native_lock()
    source = P / "execution_source"
    refs = snapshot(commit, baseline, source)
    copies = P / "control_source"
    copies.mkdir()
    helper_refs, copy_refs = [], []
    for index, path in enumerate(helper_paths()):
        helper_refs.append(s.reference(path))
        target = copies / (str(index) + "_" + path.name)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        copy_refs.append(s.reference(target))
    protocol = {"schema": "r27-final-selection-normalized-inputs/v1", "created_utc": now(), "reviewed_commit": commit,
        "baseline_ref": base_ref, "native_lock": native, "helpers": helper_refs, "helper_copies": copy_refs, "source_refs": refs,
        "arms": list(ARMS), "denominator": 131, "strict_threshold_pct": 10, "blind": False, "gate_B": "unvalidated",
        "new_cost_coefficients": 0, "native_remeasurement": False, "target_latency_fitting": False,
        "source_policy": "R25/off115 files; only two reviewed-commit adapter replacements; same source both arms",
        "treatment": "final output row selection off/on; all inherited R25 costs and flags unchanged",
        "fix": "derive proof only from normalized actual static_inputs; preserve exact comparison and preparation failures",
        "expected_direction": "ordinary final FFN rows decrease when R<B; hybrid final norm rows increase; gather/index adds work; latency direction not guaranteed",
        "workflow": "both frozen-source full131 preflights at freeze and lock, then262 terminal barrier before scoring; no terminal retry",
        "old_R25_failed_attempt": "preserved, not retroactively accepted"}
    write_new(P / "protocol.json", protocol)
    api = load("r27_snapshot_api", source / "tools/predict_stable_native_dataset.py")
    for name, module in list(sys.modules.items()):
        if name == "tools" or name.startswith("tools.") or name == "heterollm_sim" or name.startswith("heterollm_sim."):
            path = getattr(module, "__file__", None)
            s.require(path is None or Path(path).resolve().is_relative_to(source.resolve()), "import escaped snapshot: " + name)
    for arm in ARMS:
        args = r.freeze_arguments(baseline, "retained")
        args["final_output_selection"] = arm == "on"
        api.freeze_selection(Path(baseline["selection_ref"]["path"]), P / arm, **args)
    preflights = run_preflights("freeze")
    off, off_ref = s.read_json(P / "off/freeze.json")
    on, on_ref = s.read_json(P / "on/freeze.json")
    for candidate in (off, on):
        check_source_inheritance(baseline, candidate, protocol)
    count = compare_inputs(off, on)
    baseline_count = compare_baseline_off(baseline, off)
    s.require(native_lock() == native, "native changed during freeze")
    write_new(P / "freeze_receipt.json", {"schema": "r27-paired-freeze-receipt/v1", "created_utc": now(),
        "protocol_ref": s.reference(P / "protocol.json"), "off": off_ref, "on": on_ref,
        "compared_cells": count, "baseline_static_compared_cells": baseline_count, "frozen_source_preflights": preflights})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-commit", required=True)
    freeze(parser.parse_args().reviewed_commit)
