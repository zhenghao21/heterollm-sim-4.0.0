"""Post-collection, read-only evidence gate for frozen stream-event probes.

This supplement deliberately lives outside the probe freeze. It is an added
sanity/integrity check, not a preregistered calibration or blind-acceptance gate.
No GPU, model actual, fitted coefficient, or frozen file is accessed for writing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from uuid import uuid4

DIGEST = re.compile(r"[0-9a-f]{64}\Z")
TIMER_TOLERANCE_NS = 10_000


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            count += len(data)
            digest.update(data)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": count}


def load_json(path):
    def reject_constant(value):
        raise ValueError("non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def same_path(a, b):
    return isinstance(a, str) and Path(a).resolve() == Path(b).resolve()


def containment_issues(document):
    """Coarse timer sanity: an event interval must fit its host envelope.

    Ten microseconds or two QPC ticks (whichever is larger) is diagnostic
    timer tolerance, not a claim of ten-microsecond timing accuracy.
    """
    issues = []
    frequency = document.get("qpc_frequency")
    if not finite_positive(frequency):
        return ["invalid QPC frequency"], 0
    tolerance_ns = max(TIMER_TOLERANCE_NS, 2e9 / frequency)
    rows = document.get("runs")
    if not isinstance(rows, list) or not rows:
        return ["event rows missing"], 0
    checked = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(f"row {index}: invalid event row")
            continue
        event_ms, host_ns = row.get("event_envelope_ms"), row.get("host_wall_ns")
        if not finite_positive(event_ms) or not finite_positive(host_ns):
            issues.append(f"row {index}: invalid positive event/host duration")
            continue
        checked += 1
        if event_ms * 1e6 > host_ns + tolerance_ns:
            issues.append(f"row {index}: event exceeds enclosing host wall")
    return issues, checked


def check_raw_references(directory, protocol, audit_report):
    issues, records = [], []
    expected = {(cfg["id"], mode) for cfg in protocol["configs"] for mode in ("event", "control")}
    rows = audit_report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        issues.append("raw audit mapping count mismatch")
        rows = rows if isinstance(rows, list) else []
    keys = [(row.get("config"), row.get("mode")) for row in rows if isinstance(row, dict)]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        issues.append("raw audit mapping missing, duplicate, or unexpected")
    for config, mode in sorted(expected):
        path = directory / f"{config}.{mode}.json"
        try:
            current = fingerprint(path)
            records.append({"config": config, "mode": mode, **current})
            matches = [r for r in rows if isinstance(r, dict) and (r.get("config"), r.get("mode")) == (config, mode)]
            if len(matches) != 1:
                continue
            stored = matches[0]
            if not same_path(stored.get("path"), path):
                issues.append(f"raw audit path mismatch:{config}:{mode}")
            if stored.get("sha256") != current["sha256"]:
                issues.append(f"raw changed after audit:{config}:{mode}")
        except Exception as exc:
            issues.append(f"raw unavailable:{config}:{mode}:{exc}")
    return issues, records


def verify_manifest(probe, manifest):
    issues, records = [], []
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return ["empty manifest mapping"], []
    seen = set()
    for entry in entries:
        try:
            path = Path(entry["path"]).resolve()
            if path in seen:
                issues.append("duplicate manifest path:" + str(path))
            seen.add(path)
            current = fingerprint(path)
            records.append(current)
            if not isinstance(entry.get("sha256"), str) or not DIGEST.fullmatch(entry["sha256"]) or entry["sha256"] != current["sha256"] or entry.get("bytes") != current["bytes"]:
                issues.append("frozen file changed:" + str(path))
        except Exception as exc:
            issues.append("frozen file unavailable:" + str(exc))
    for name in ("protocol.json", "full_raw_audit.py", "assess.py", "summarize_matrix.py", "invoke.ps1", "run_frozen_matrix.ps1"):
        if (probe / name).resolve() not in seen:
            issues.append("mandatory frozen file absent:" + name)
    executable = manifest.get("executable", {})
    try:
        matching = [r for r in records if same_path(executable.get("path"), r["path"])]
        if len(matching) != 1 or matching[0]["sha256"] != executable.get("sha256") or matching[0]["bytes"] != executable.get("bytes"):
            issues.append("executable mapping missing/mismatch")
    except Exception:
        issues.append("invalid executable mapping")
    return issues, records


@contextmanager
def frozen_tools(probe):
    """Use exact already-verified sources without creating a frozen pycache."""
    names = ("full_raw_audit", "assess", "summarize_matrix")
    saved = {name: sys.modules.get(name) for name in names}
    old_no_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, probe / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        sys.dont_write_bytecode = old_no_bytecode
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def audit(probe, directory):
    probe, directory = Path(probe).resolve(), Path(directory).resolve()
    integrity, containment, inputs, assessments = [], [], [], []
    event_rows_checked = 0
    raw_passed = False
    summary_passed = False
    configs = []
    result = {"schema": "supplemental-stream-probe-gate/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "probe_directory": str(probe), "run_directory": str(directory),
              "gate_registration": "post-collection independent supplement; not preregistered blind acceptance",
              "event_host_tolerance_ns": TIMER_TOLERANCE_NS, "tolerance_rule": "max(10000 ns, two QPC ticks)",
              "calibration_eligible": False, "accuracy_promotion": False, "llm_actual_read": False, "gpu_access": False}
    try:
        for path in (probe / "protocol.json", probe / "build_manifest.json", directory / "full_raw_audit.json", directory / "execution.json", directory / "matrix_summary.json", directory / "identity_before.json", directory / "identity_after.json"):
            inputs.append(fingerprint(path))
        protocol = load_json(probe / "protocol.json")
        manifest = load_json(probe / "build_manifest.json")
        configs = protocol["configs"]
        ids = [c["id"] for c in configs]
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
            raise ValueError("empty/duplicate/unsafe configuration IDs")
        manifest_issues, frozen_records = verify_manifest(probe, manifest)
        integrity.extend(manifest_issues)
        inputs.extend(frozen_records)
        manifest_sha = fingerprint(probe / "build_manifest.json")["sha256"]
        for name in ("identity_before.json", "identity_after.json"):
            if load_json(directory / name).get("manifest_sha256") != manifest_sha:
                integrity.append("manifest differs from execution:" + name)
        stored_audit = load_json(directory / "full_raw_audit.json")
        issues, raw_records = check_raw_references(directory, protocol, stored_audit)
        integrity.extend(issues)
        inputs.extend(raw_records)
        execution, summary = load_json(directory / "execution.json"), load_json(directory / "matrix_summary.json")
        summary_refs, execution_refs = summary.get("assessments", []), execution.get("assessments", [])
        for label, refs in (("summary", summary_refs), ("execution", execution_refs)):
            if not isinstance(refs, list) or [r.get("config") for r in refs] != ids:
                integrity.append(label + " assessment count/order mismatch")
        for cfg in configs:
            config = cfg["id"]
            problems, checked = containment_issues(load_json(directory / f"{config}.event.json"))
            containment.extend(config + ":" + p for p in problems)
            event_rows_checked += checked
        # Refuse importing code whose frozen digest failed validation.
        if manifest_issues:
            integrity.append("current extraction not executed because frozen source identity failed")
        else:
            with frozen_tools(probe) as tools:
                fresh_audit = tools["full_raw_audit"].audit(directory, protocol, manifest)
                raw_passed = fresh_audit.get("complete_raw_passed") is True
                if fresh_audit != stored_audit:
                    integrity.append("stored raw audit differs from current re-audit")
                for cfg in configs:
                    config = cfg["id"]
                    path = directory / f"{config}.assessment.json"
                    current = fingerprint(path)
                    inputs.append(current)
                    stored = load_json(path)
                    fresh = tools["assess"].assess(load_json(directory / f"{config}.event.json"), load_json(directory / f"{config}.control.json"), cfg, protocol, manifest)
                    if stored != fresh:
                        integrity.append("stored assessment differs from current extraction:" + config)
                    refs = [r for r in summary_refs if r.get("config") == config]
                    if len(refs) != 1 or refs[0].get("sha256") != current["sha256"]:
                        integrity.append("summary assessment digest mismatch:" + config)
                    elif any(refs[0].get(k) != fresh.get(k) for k in ("accepted", "problems", "statistics")):
                        integrity.append("summary assessment result mismatch:" + config)
                    refs = [r for r in execution_refs if r.get("config") == config]
                    if len(refs) != 1 or not same_path(refs[0].get("path"), path) or refs[0].get("exists") is not True or refs[0].get("exit_code") != (0 if fresh.get("accepted") is True else 4):
                        integrity.append("execution assessment reference mismatch:" + config)
                    assessments.append({"config": config, "sha256": current["sha256"], "accepted": fresh.get("accepted") is True, "problems": fresh.get("problems", [])})
                fresh_summary = tools["summarize_matrix"].summarize(directory)
                if fresh_summary != summary:
                    integrity.append("stored summary differs from current extraction")
                summary_passed = fresh_summary.get("all_diagnostic_gates_passed") is True
    except Exception as exc:
        integrity.append("incomplete/invalid evidence:" + type(exc).__name__ + ":" + str(exc))
    # Re-read each file to detect edits during this supplemental audit, including
    # source/manifest/protocol changes and read/re-extraction races.
    bound = {}
    for item in inputs:
        path = item["path"]
        if path in bound and bound[path] != {k: item[k] for k in ("path", "sha256", "bytes")}:
            integrity.append("input changed between reads:" + path)
        bound[path] = {k: item[k] for k in ("path", "sha256", "bytes")}
    for path, before in bound.items():
        try:
            if fingerprint(path) != before:
                integrity.append("input changed during supplement:" + path)
        except Exception as exc:
            integrity.append("input unavailable during supplement:" + str(exc))
    integrity, containment = sorted(set(integrity)), sorted(set(containment))
    accepted = len(assessments) == len(configs) and bool(configs) and all(r["accepted"] for r in assessments)
    result.update(integrity_passed=not integrity, containment_passed=not containment and event_rows_checked > 0,
                  integrity_issues=integrity, containment_issues=containment, event_rows_checked=event_rows_checked,
                  fresh_raw_audit_passed=raw_passed, fresh_assessments_passed=accepted,
                  fresh_summary_passed=summary_passed, expected_configs=len(configs), assessments=assessments,
                  bound_files=sorted(bound.values(), key=lambda x: x["path"]))
    result["all_diagnostic_gates_passed"] = not integrity and result["containment_passed"] and raw_passed and accepted and summary_passed
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path(__file__).resolve().parent / ("supplemental_gate_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:8] + ".json")
    output = output.resolve()
    if output == args.probe.resolve() or args.probe.resolve() in output.parents:
        parser.error("supplement output must be outside frozen probe directory")
    if output.exists():
        parser.error("refusing to overwrite existing evidence")
    result = audit(args.probe, args.run)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({"output": str(output), **{k: result[k] for k in ("all_diagnostic_gates_passed", "integrity_passed", "containment_passed", "event_rows_checked")}, "integrity_issue_count": len(result["integrity_issues"]), "containment_issue_count": len(result["containment_issues"])}))
    return 0 if result["all_diagnostic_gates_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
