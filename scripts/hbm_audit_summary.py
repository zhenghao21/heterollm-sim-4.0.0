"""Sum a full-stream HBM command audit and compare its TPOT with a plain run of the same candidate.

Run a candidate with the observer on, the channel mirror and idle-refresh fold off:

    HETEROLLM_HBM_AUDIT_DISABLE_EQUIV=1 HETEROLLM_HBM_TRACE_LIMIT=0 \
    HETEROLLM_HBM_TRACE_SUMMARY_FILE=<run dir>/commands.summary <runner for one candidate into <run dir>>

then

    python scripts/hbm_audit_summary.py <run dir> <plain candidate json> <out json>

Every HBM component writes one committed ``commands.summary.component_*`` file.
The audit passes when every command the kernel issued was checked against the
35 JEDEC rules without a violation, no batch bypassed the kernel (those run in
the Python reference controller, which the observer does not see), and the
audited TPOT equals the plain run's bit for bit.
"""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def rule_names():
    source = (ROOT / "src/heterollm_sim/_hbm_fr_core.c").read_text(encoding="utf-8")
    names = re.findall(r'"([^"]+)"', source.split("fr_audit_rule_names[AUDIT_RULE_COUNT] = {", 1)[1]
                       .split("};", 1)[0])
    assert len(names) == 35
    return names


def summarize(run, plain_candidate):
    names = rule_names()
    totals, checks, violations, first = Counter(), Counter(), Counter(), []
    files = sorted(Path(run).glob("commands.summary*"))
    summaries = audits = 0
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines and lines[0].startswith("#COMMITTED_COMPONENT"), path
        for line in lines[1:]:
            fields = line.strip().split("\t")
            data = dict(part.split("=", 1) for part in fields[1:] if "=" in part)
            if fields[0] == "SUMMARY":
                summaries += 1
                totals.update({key: int(value) for key, value in data.items()})
            elif fields[0] == "AUDIT_DELTA":
                audits += 1
                totals["audit_commands"] += int(data["commands"])
                counts = list(map(int, data["checks"].split(",")))
                fails = list(map(int, data["violations"].split(",")))
                assert len(counts) == len(fails) == 35
                checks.update(dict(zip(names, counts)))
                violations.update(dict(zip(names, fails)))
                if data.get("first"):
                    first.append({"file": path.name, "first": data["first"]})
    found = list((Path(run) / "candidates").glob("*.json"))
    assert len(found) == 1, run
    audit = json.loads(found[0].read_text(encoding="utf-8"))
    plain = json.loads(Path(plain_candidate).read_text(encoding="utf-8"))
    audit_tpot = audit["metrics"]["mean_complete_request_tpot_ns"]
    plain_tpot = plain["metrics"]["mean_complete_request_tpot_ns"]
    payload = {
        "candidate_id": audit["candidate_id"], "component_files": len(files), "summary_lines": summaries,
        "audit_lines": audits, **totals, "total_violations": sum(violations.values()),
        "rules": [{"name": name, "checks": checks[name], "violations": violations[name]} for name in names],
        "rules_without_checks": [name for name in names if not checks[name]],
        "first_violations": first[:5],
        "audit_status": audit["status"], "plain_status": plain["status"],
        "audit_contract": audit["runner_contract"], "plain_contract": plain["runner_contract"],
        "audit_tpot_ns": audit_tpot, "plain_tpot_ns": plain_tpot,
        "tpot_bit_identical": audit_tpot.hex() == plain_tpot.hex(),
        "audit_wall_seconds": audit["wall_seconds"], "plain_wall_seconds": plain["wall_seconds"],
    }
    payload["passed"] = bool(files and summaries == audits == len(files) and not payload["total_violations"]
                             and totals["total"] == totals["audit_commands"] > 0
                             and not totals["unaudited_batches"] and payload["tpot_bit_identical"]
                             and audit["status"] == plain["status"] == "completed")
    return payload


def main(argv=None):
    run, plain, out = (Path(value) for value in (argv or sys.argv[1:]))
    payload = summarize(run, plain)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("rules", "first_violations")},
                     ensure_ascii=False))
    for row in payload["rules"]:
        print(f"  {row['name']:24s} {row['checks']:>14d} / {row['violations']}")
    if payload["first_violations"]:
        print(json.dumps(payload["first_violations"][:3]))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
