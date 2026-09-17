"""R33 clean-Git frontend comparison with a shared identity-diagnostics patch; fixed native."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import hashlib
import importlib.metadata
import platform
import time
import subprocess
import sys
import uuid
from types import SimpleNamespace

P = Path(__file__).resolve().parent
ROOT = P.parents[4]
LOOP = P.parent
BASE = LOOP / "round_030/off/freeze.json"
BASELINE_COMMIT = "8b715dc80ded2a35d7ab52b46339320149a33da4"
ARMS = ("off", "on")
ALLOWED_SOURCE_DIFF = {"src/heterollm_sim/planner.py"}
COMMON_SOURCE_OVERRIDE_PATHS = {"tools/predict_stable_native_dataset.py"}
MODES = {arm: "legacy_mma_output_wave" for arm in ARMS}
SOURCE_COUNT = 117
CONFIG_KEYS = ("batch", "ubatch", "parallel", "ctx", "gpu_layers", "op_offload", "flash_attn", "seed", "output", "expected_prompt_tokens")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = load("r33_saved_evidence", LOOP / "round_023/summarize_ablation.py")
r = load("r33_reviewed_recipe", LOOP / "round_023/evaluate_candidate.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return s.reference(path)


def natural_process(argv, *, capture_output=False, text=False, stdout=None, stderr=None, check=False, purpose="helper", input=None):
    """Once spawned, every observer/journal failure waits for the same natural exit."""
    journal = P / "process_observations" / (purpose + "." + uuid.uuid4().hex)
    journal.mkdir(parents=True, exist_ok=False)
    # A failed pre-spawn receipt cannot strand a child: none has been launched.
    start_ref = write_new(journal / "start.json", {"schema": "r33-process-start/v1", "created_utc": now(),
        "argv": [str(x) for x in argv], "wait_policy": "natural_exit", "hard_time_limit_enforced": False})
    first_error = None; first_stage = None; error_count = 0; errors = []
    child = None; out = err = None; returncode = None; interruptions = 0

    def remember(error, stage):
        nonlocal first_error, first_stage, error_count, interruptions
        error_count += 1
        if isinstance(error, KeyboardInterrupt): interruptions += 1
        if first_error is None: first_error, first_stage = error, stage
        # Persistent I/O failure must not grow the journal or memory forever.
        if len(errors) < 16:
            errors.append({"stage": stage, "exception": type(error).__name__, "message": str(error)})

    def note(name, payload):
        try:
            return write_new(journal / name, {"created_utc": now(), **payload})
        except BaseException as error:
            remember(error, "journal:" + name)
            return None

    try:
        child = subprocess.Popen([str(x) for x in argv], text=text,
            stdin=subprocess.PIPE if input is not None else None,
            stdout=subprocess.PIPE if capture_output else stdout,
            stderr=subprocess.PIPE if capture_output else stderr)
    except BaseException as error:
        remember(error, "spawn")
        note("unresolved.json", {"reason": type(error).__name__ + ": " + str(error),
            "child_may_be_live": False, "automatic_retry": False, "termination_requested": False})
        raise first_error

    note("child.json", {"pid": child.pid, "start_ref": start_ref})
    retry_delay = 0.05; observation_noted = False
    while True:
        try:
            out, err = child.communicate(input=input)
            # communicate() retains pending input after an interruption. It must
            # never be submitted twice to the same child's stdin.
            input = None
            returncode = child.poll()
            if type(returncode) is not int:
                raise RuntimeError("helper natural exit remains unresolved")
            break
        except BaseException as error:
            input = None
            remember(error, "observe")
            if not observation_noted:
                observation_noted = True
                note("observation-interrupted.0001.json", {"pid": child.pid,
                    "exception": type(error).__name__, "message": str(error),
                    "action": "wait_for_natural_exit_then_reject_parent_phase", "automatic_retry": False,
                    "termination_requested": False, "evidence_eligible": False})
            # A read error can outlive the child. Observe exit independently,
            # while still rejecting the phase and never consuming partial output.
            try:
                returncode = child.poll()
                if type(returncode) is int: break
            except BaseException as poll_error:
                remember(poll_error, "poll")
            try:
                time.sleep(retry_delay)
            except BaseException as sleep_error:
                remember(sleep_error, "retry_wait")
            retry_delay = min(1.0, retry_delay * 2)

    note("finish.json", {"schema": "r33-process-finish/v1", "start_ref": start_ref,
        "pid": child.pid, "returncode": returncode, "natural_exit_observed": True,
        "observation_interruptions": interruptions, "observation_error_count": error_count,
        "observation_errors": list(errors), "errors_truncated": error_count > len(errors),
        "first_error_stage": first_stage, "parent_phase_rejected": first_error is not None,
        "automatic_retry": False, "termination_requested": False, "hard_time_limit_enforced": False})
    if first_error is not None:
        # Preserve the first exception, including SystemExit/KeyboardInterrupt;
        # only now is it safe to reject the caller and stop new submissions.
        raise first_error
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, argv, out, err)
    return SimpleNamespace(returncode=returncode, stdout=out, stderr=err)


def git_blobs(commit, names):
    names = list(names)
    requests = "".join(commit + ":" + name + "\n" for name in names).encode("utf-8")
    raw = natural_process(["git", "-C", str(ROOT), "cat-file", "--batch"], input=requests,
                          capture_output=True, check=True, purpose="git-blobs").stdout
    position = 0; result = {}
    for name in names:
        end = raw.find(b"\n", position)
        s.require(end >= position, "missing Git blob header")
        header = raw[position:end].decode("ascii").split()
        s.require(len(header) == 3 and header[1] == "blob" and header[2].isdigit(), "reviewed Git blob missing: " + name)
        start = end + 1; size = int(header[2]); finish = start + size
        s.require(finish < len(raw) and raw[finish:finish + 1] == b"\n", "truncated Git blob")
        data = raw[start:finish]
        s.require(hashlib.sha1(("blob %d\0" % size).encode() + data).hexdigest() == header[0], "Git object bytes disagree")
        result[name] = data; position = finish + 1
    s.require(position == len(raw), "unexpected extra Git blob data")
    return result


def helper_paths():
    names = ("freeze_candidate.py", "run_candidate.py", "summarize_groups.py", "verify_frozen_preflight.py",
             "test_freeze_candidate.py", "test_run_candidate.py", "test_summarize_groups.py",
             "postprocess/postprocess.py", "postprocess/archive_verified.py", "postprocess/test_postprocess.py")
    return [*(P / name for name in names), LOOP / "round_023/summarize_ablation.py",
            LOOP / "round_023/evaluate_candidate.py", ROOT / "tools/verify_fixed_native.py"]


def check_commit(commit):
    s.require(isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "full reviewed commit required")
    actual = natural_process(["git", "-C", str(ROOT), "rev-parse", "--verify", commit + "^{commit}"],
                             capture_output=True, text=True, check=True, purpose="git-commit").stdout.strip()
    s.require(actual == commit, "reviewed commit unresolved")
    paths = helper_paths(); expected = git_blobs(commit, [path.relative_to(ROOT).as_posix() for path in paths])
    for path in paths:
        s.require(expected[path.relative_to(ROOT).as_posix()] == path.read_bytes(), "helper differs from reviewed commit: " + str(path))


def git_members(commit):
    raw = natural_process(["git", "-C", str(ROOT), "ls-tree", "-r", "-z", commit, "--", "src", "tools"],
                          capture_output=True, check=True, purpose="git-tree").stdout
    names = []
    for record in raw.split(b"\0"):
        if not record: continue
        header, encoded = record.split(b"\t", 1)
        name = encoded.decode("utf-8")
        if not name.endswith(".py"): continue
        mode, kind, oid = header.decode("ascii").split()
        s.require(mode in ("100644", "100755") and kind == "blob", "non-file Python tree entry")
        s.require(name.startswith(("src/", "tools/")) and ".." not in Path(name).parts
                  and "__pycache__" not in Path(name).parts, "unexpected Python tree member")
        names.append(name)
    s.require(len(names) == SOURCE_COUNT and len({n.casefold() for n in names}) == SOURCE_COUNT,
              "reviewed tree must contain exact117 Python files")
    return sorted(names)


def source_blobs(commit, names, common_source_overrides):
    """Read each file from its declared Git commit, with one shared diagnostic override."""
    s.require(isinstance(common_source_overrides, dict)
              and set(common_source_overrides) == COMMON_SOURCE_OVERRIDE_PATHS,
              "common source overrides must be exactly the identity-diagnostics helper")
    s.require(COMMON_SOURCE_OVERRIDE_PATHS <= set(names), "common override missing from arm source membership")
    s.require(all(isinstance(value, str) and len(value) == 40
                  and all(char in "0123456789abcdef" for char in value)
                  for value in common_source_overrides.values()), "common override requires a full Git commit")
    content = git_blobs(commit, names)
    origins = {name: commit for name in names}
    for name, origin in common_source_overrides.items():
        content[name] = git_blobs(origin, [name])[name]
        origins[name] = origin
    return content, origins


def snapshot(commit, destination, common_source_overrides):
    names = git_members(commit)
    content, origins = source_blobs(commit, names, common_source_overrides)
    destination.mkdir(parents=True, exist_ok=False)
    refs = []
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream: stream.write(content[name])
        refs.append({"relative": name, "origin": "reviewed_git_blob", "commit": origins[name], "ref": s.reference(target)})
    return refs


def environment_identity():
    """Small runtime/build identity, not a replacement for target hardware evidence."""
    dependencies = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name"); version = dist.version
        s.require(isinstance(name, str) and bool(name.strip()) and isinstance(version, str) and bool(version),
                  "installed dependency has incomplete identity")
        key = name.lower().replace("_", "-").replace(".", "-")
        metadata = dist.read_text("METADATA") or dist.read_text("PKG-INFO")
        s.require(isinstance(metadata, str) and bool(metadata), "dependency metadata missing: " + key)
        record = dist.read_text("RECORD")
        item = {"version": version, "location": str(Path(dist.locate_file("")).resolve(strict=True)),
                "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
                "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record is not None else None}
        s.require(key not in dependencies or dependencies[key] == item, "conflicting installed dependency: " + key)
        dependencies[key] = item
    s.require({"numpy", "ortools"} <= set(dependencies), "required simulator dependencies are missing")
    return {"schema": "r33-python-environment/v1", "executable": s.reference(sys.executable),
            "version": sys.version, "implementation": platform.python_implementation(),
            "platform": sys.platform, "machine": platform.machine(),
            "dependencies": dict(sorted(dependencies.items())),
            "dependency_scope": "installed distribution metadata and RECORD identity; not a full installed-file integrity scan"}


def check_source_difference(before, after):
    s.require(len(before) == SOURCE_COUNT and set(before) == set(after), "paired117 source membership differs")
    changed = {name for name in before if before[name] != after[name]}
    s.require(changed == ALLOWED_SOURCE_DIFF, "source diff must be exactly reviewed frontend planner change")
    return changed


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


def check_mode_and_final_selection(freeze, inputs, arm):
    expected = MODES[arm]
    s.require(freeze.get("mmvq_hbm_mode", MODES["off"]) == expected, "campaign HBM mode differs")
    s.require(inputs.get("mmvq_hbm_mode", MODES["off"]) == expected, "cell HBM mode differs")
    s.require(freeze.get("final_output_selection") is True and inputs.get("final_output_selection") is True,
              "both arms must retain final output selection on")
    proof = inputs.get("final_output_selection_binding")
    s.require(isinstance(proof, dict) and proof.get("status") in ("conditional", "uncovered"), "inherited output selection proof missing")
    s.require(proof.get("config") == {key: inputs["config"].get(key) for key in CONFIG_KEYS},
              "output selection proof differs from actual normalized configuration")


def comparison_inputs(inputs, freeze):
    result = normalize_known_copies(inputs, freeze)
    result.pop("mmvq_hbm_mode", None)  # Omitted and explicit legacy defaults are identical after validation.
    return result


def compare_inputs(off, on):
    a, b = s.unique_cells(off), s.unique_cells(on)
    s.require(len(a) == 131 and set(a) == set(b), "fixed131 cell membership differs")
    check_source_difference(s.source_content(off), s.source_content(on))
    for ident in a:
        s.require(a[ident].get("preparation_error") is None and b[ident].get("preparation_error") is None,
                  "preparation error blocks paired preflight: " + ident)
        x, y = a[ident]["static_inputs"], b[ident]["static_inputs"]
        check_mode_and_final_selection(off, x, "off")
        check_mode_and_final_selection(on, y, "on")
        s.require(a[ident].get("model_key") == b[ident].get("model_key")
                  and a[ident].get("deployment") == b[ident].get("deployment"), "paired grouping changed")
        s.require(comparison_inputs(x, off) == comparison_inputs(y, on), "non-treatment static difference: " + ident)
    return len(a)


def compare_baseline_off(baseline, off):
    """R30/off actual inputs and final selection stay exact apart from known copies."""
    before, after = s.unique_cells(baseline), s.unique_cells(off)
    s.require(len(before) == 131 and set(before) == set(after), "R30/off fixed131 membership changed")
    for ident in before:
        a, b = before[ident], after[ident]
        s.require(a.get("preparation_error") is None and b.get("preparation_error") is None, "baseline/off preparation changed")
        s.require(a.get("model_key") == b.get("model_key") and a.get("deployment") == b.get("deployment"), "baseline grouping changed")
        check_mode_and_final_selection(baseline, a["static_inputs"], "off")
        check_mode_and_final_selection(off, b["static_inputs"], "off")
        s.require(comparison_inputs(a["static_inputs"], baseline) == comparison_inputs(b["static_inputs"], off),
                  "R30/off static input changed: " + ident)
    return len(before)


def check_source_inheritance(baseline, candidate, protocol, arm):
    after = s.source_content(candidate)
    copied_root = Path(candidate["source"]["root"]).resolve(strict=True)
    s.require(copied_root == (P / arm / "source").resolve(), "arm source root differs")
    actual = {path.relative_to(copied_root).as_posix() for path in copied_root.rglob("*.py") if "__pycache__" not in path.parts}
    expected = {row["relative"]: {k: row["ref"][k] for k in ("sha256", "bytes")}
                for row in protocol["source_refs"][arm]}
    s.require(len(expected) == SOURCE_COUNT and set(after) == actual == set(expected), "exact117 source membership differs")
    s.require(after == expected, "arm source differs from its reviewed Git snapshot")


def native_lock():
    verifier = load("r33_native_lock", ROOT / "tools/verify_fixed_native.py")
    result = verifier.verify_lock(LOOP / "state.json")
    s.require(result["selected_cells"] == 131 and result["raw_files"] == 131 and result["formal_requests"] == 894 and result["selection_sha256"] == s.SELECTION_SHA, "fixed native changed")
    return result


def read_protocol():
    protocol, protocol_ref = s.read_json(P / "protocol.json")
    s.require(protocol.get("schema") == "r33-clean-git-frontend-inputs/v1" and protocol.get("arms") == list(ARMS)
              and protocol.get("denominator") == 131 and protocol.get("strict_threshold_pct") == 10
              and protocol.get("gate_B") == "unvalidated" and protocol.get("new_cost_coefficients") == 0,
              "protocol invariants differ")
    s.require(protocol.get("modes") == MODES and protocol.get("final_output_selection_both_arms") is True,
              "both arms must retain legacy HBM mode and final output selection")
    s.require(protocol.get("worker_wait_policy") == "natural_exit_soft_observation"
              and type(protocol.get("soft_observation_seconds")) is int and protocol["soft_observation_seconds"] == 600
              and protocol.get("hard_time_limit_enforced") is False, "natural observation policy differs")
    check_commit(protocol.get("reviewed_commit"))
    commits = {"off": BASELINE_COMMIT, "on": protocol["reviewed_commit"]}
    s.require(protocol.get("source_commits") == commits and protocol.get("allowed_source_diff") == sorted(ALLOWED_SOURCE_DIFF),
              "reviewed source commit/whitelist differs")
    common_source_overrides = {name: protocol["reviewed_commit"] for name in COMMON_SOURCE_OVERRIDE_PATHS}
    s.require(protocol.get("common_source_overrides") == common_source_overrides,
              "common source override whitelist/commit differs")
    s.require(protocol.get("python_environment") == environment_identity(), "Python interpreter/dependency identity changed")
    s.require(protocol["baseline_ref"]["path"] == str(BASE.resolve()), "static baseline path differs")
    s.verify_reference(protocol["baseline_ref"])
    paths = helper_paths()
    s.require([ref["path"] for ref in protocol["helpers"]] == [str(path.resolve()) for path in paths], "helper membership/order differs")
    for ref in protocol["helpers"]: s.verify_reference(ref)
    s.require(set(protocol["source_refs"]) == set(ARMS), "both source snapshots required")
    summaries = {}
    for arm in ARMS:
        rows = protocol["source_refs"][arm]; names = [row["relative"] for row in rows]
        s.require(names == git_members(commits[arm]), "snapshot differs from reviewed Git tree membership")
        root = (P / "execution_source" / arm).resolve(strict=True)
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*.py") if "__pycache__" not in path.parts}
        s.require(actual == set(names), "unlisted Python source in execution copy")
        summaries[arm] = {}
        content, origins = source_blobs(commits[arm], names, common_source_overrides)
        for row in rows:
            found = s.verify_reference(row["ref"])
            s.require(Path(found["path"]).resolve() == root / row["relative"], "source escaped exact execution root")
            s.require(row.get("origin") == "reviewed_git_blob" and row.get("commit") == origins[row["relative"]], "source origin differs")
            s.require(Path(found["path"]).read_bytes() == content[row["relative"]], "snapshot differs from reviewed blob")
            summaries[arm][row["relative"]] = {k: found[k] for k in ("sha256", "bytes")}
    check_source_difference(summaries["off"], summaries["on"])
    s.require(len(protocol["helper_copies"]) == len(paths), "helper copy coverage differs")
    for index, (original, copied) in enumerate(zip(protocol["helpers"], protocol["helper_copies"])):
        actual = s.verify_reference(copied)
        s.require(Path(actual["path"]).resolve() == (P / "control_source" / (str(index) + "_" + Path(original["path"]).name)).resolve(),
                  "helper copy escaped exact path")
        s.require((actual["sha256"], actual["bytes"]) == (original["sha256"], original["bytes"]), "helper copy changed")
    return protocol, protocol_ref


def validate_preflight(stage, arm, reference, protocol_ref):
    receipt, actual = s.read_json(reference["path"])
    s.require(actual == s.verify_reference(reference), "preflight receipt changed")
    s.require(receipt.get("schema") == "r33-frozen-source-preflight/v1" and receipt.get("stage") == stage
              and receipt.get("arm") == arm and receipt.get("status") == "passed" and receipt.get("verified_cells") == 131,
              "both131 frozen-source preflights must pass")
    s.require(receipt.get("mmvq_hbm_mode") == MODES[arm] and receipt.get("final_output_selection") is True,
              "preflight treatment/final selection differs")
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
            result = natural_process([sys.executable, str(P / "verify_frozen_preflight.py"), "--freeze", str(P / arm / "freeze.json"),
                "--protocol", str(P / "protocol.json"), "--stage", stage, "--arm", arm, "--receipt", str(receipt)],
                stdout=stream, stderr=subprocess.STDOUT, check=False, purpose="preflight-"+stage+"-"+arm)
        if result.returncode or not receipt.exists():
            failed.append(arm)
        else:
            references[arm] = s.reference(receipt)
            validate_preflight(stage, arm, references[arm], protocol_ref)
    s.require(not failed and set(references) == set(ARMS), "frozen-source preflight failed: " + ",".join(failed))
    return references


def freeze_arm(arm):
    """Runs in a fresh interpreter so neither arm can reuse the other's imports."""
    protocol, _ = read_protocol()
    source = (P / "execution_source" / arm).resolve(strict=True)
    sys.path[:0] = [str(source / "src"), str(source)]
    api = load("r33_" + arm + "_snapshot_api", source / "tools/predict_stable_native_dataset.py")
    origins = load("r33_origin_check", P / "verify_frozen_preflight.py")
    origins.assert_imports_confined(source)
    baseline, _ = s.read_json(BASE)
    args = r.freeze_arguments(baseline, "retained")
    args.update(final_output_selection=True, mmvq_hbm_mode=MODES[arm])
    api.freeze_selection(Path(baseline["selection_ref"]["path"]), P / arm, **args)
    origins.assert_imports_confined(source)
    read_protocol()


def freeze(commit):
    check_commit(commit)
    s.require(not any((P / name).exists() for name in (*ARMS, "protocol.json", "execution_source", "control_source", "freeze_receipt.json", "controls.json")),
              "new freeze only; never adopt partial/old preparation")
    baseline, base_ref = s.read_json(BASE)
    s.require(baseline.get("selection_sha256") == s.SELECTION_SHA and baseline.get("final_output_selection") is True,
              "static baseline must be fixed R30/off")
    native = native_lock(); environment = environment_identity()
    commits = {"off": BASELINE_COMMIT, "on": commit}
    common_source_overrides = {name: commit for name in COMMON_SOURCE_OVERRIDE_PATHS}
    refs = {arm: snapshot(commits[arm], P / "execution_source" / arm, common_source_overrides) for arm in ARMS}
    by_arm = {arm: {row["relative"]: {key: row["ref"][key] for key in ("sha256", "bytes")} for row in refs[arm]} for arm in ARMS}
    check_source_difference(by_arm["off"], by_arm["on"])
    copies = P / "control_source"; copies.mkdir()
    helper_refs, copy_refs = [], []
    for index, path in enumerate(helper_paths()):
        helper_refs.append(s.reference(path)); target = copies / (str(index) + "_" + path.name)
        with target.open("xb") as stream: stream.write(path.read_bytes())
        copy_refs.append(s.reference(target))
    protocol = {"schema": "r33-clean-git-frontend-inputs/v1", "created_utc": now(), "reviewed_commit": commit,
        "baseline_ref": base_ref, "native_lock": native, "helpers": helper_refs, "helper_copies": copy_refs,
        "source_refs": refs, "source_commits": commits, "allowed_source_diff": sorted(ALLOWED_SOURCE_DIFF),
        "common_source_overrides": common_source_overrides,
        "python_environment": environment, "arms": list(ARMS), "denominator": 131, "strict_threshold_pct": 10,
        "blind": False, "gate_B": "unvalidated", "new_cost_coefficients": 0, "native_remeasurement": False,
        "target_latency_fitting": False, "source_policy": "117 raw Git blobs per arm; the exact common identity-diagnostics helper comes from the reviewed commit in both arms, all other files from each arm commit; per-file provenance; planner-only paired diff; no worktree source",
        "modes": MODES, "final_output_selection_both_arms": True,
        "worker_wait_policy": "natural_exit_soft_observation", "soft_observation_seconds": 600,
        "hard_time_limit_enforced": False, "treatment": "GPU consumer ownership in frontend transfer; all costs and static configurations equal",
        "expected_direction": {
            "scope": "non-MTP online target invocations",
            "no_gpu_consumer": "remove GPU control/transfer ownership while preserving CPU preparation",
            "gpu_consumers": "one control group per actual consumer device; preserve real input transfer and predecessor dependencies",
            "readiness": "only related GPU entry waits for device readiness; independent CPU prefix may execute",
            "invariants": "MTP, static request compilation and every operator-cost parameter remain unchanged",
            "validation": "compare same-input structures and paired off/on errors; do not select cells using actuals",
            "monotonicity": "no promised total-time or APE improvement; shared-resource competition may change the critical path"},
        "workflow": "separate-process freeze and preflight; full262 sealed terminals before scores",
        "shared_patch_scope": "tools/predict_stable_native_dataset.py records failed full-file identity checks in the same read; acceptance predicate and single 4MiB streaming read remain unchanged; this is diagnostics, not a claimed identity-root-cause fix",
        "prior_campaigns": "R32 sealed after static identity preparation failure, no predictions; later one-pass diagnostic matched only at its own observation time, never retroactively accepts or retries R32; R30 and earlier preserved"}
    write_new(P / "protocol.json", protocol)
    for arm in ARMS:
        natural_process([sys.executable, str(P / "freeze_candidate.py"), "--freeze-arm", arm],
                        check=True, purpose="freeze-arm-" + arm)
    preflights = run_preflights("freeze")
    off, off_ref = s.read_json(P / "off/freeze.json"); on, on_ref = s.read_json(P / "on/freeze.json")
    for arm, candidate in (("off", off), ("on", on)):
        check_source_inheritance(baseline, candidate, protocol, arm)
    count = compare_inputs(off, on); baseline_count = compare_baseline_off(baseline, off)
    s.require(native_lock() == native and environment_identity() == environment, "native/runtime changed during freeze")
    write_new(P / "freeze_receipt.json", {"schema": "r33-paired-freeze-receipt/v1", "created_utc": now(),
        "protocol_ref": s.reference(P / "protocol.json"), "off": off_ref, "on": on_ref,
        "compared_cells": count, "baseline_static_compared_cells": baseline_count, "frozen_source_preflights": preflights})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reviewed-commit")
    mode.add_argument("--freeze-arm", choices=ARMS)
    args = parser.parse_args()
    try:
        freeze_arm(args.freeze_arm) if args.freeze_arm else freeze(args.reviewed_commit)
    except BaseException as error:
        write_new(P / ("preparation_rejected.%04d.json" % (len(list(P.glob("preparation_rejected.*.json"))) + 1)),
            {"schema": "r33-preparation-failure/v1", "created_utc": now(), "reviewed_commit": args.reviewed_commit,
             "arm": args.freeze_arm, "status": "failed", "reason": type(error).__name__ + ": " + str(error),
             "partial_files_preserved": True, "automatic_retry": False, "native_run": False, "prediction_run": False})
        raise
