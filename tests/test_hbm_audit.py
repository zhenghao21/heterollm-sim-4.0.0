"""The HBM command observer: every kernel command checked against the JEDEC rules, results unchanged."""
from __future__ import annotations

from pathlib import Path
import random
import sys

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.physical_presets import resolve_physical_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hbm_audit_summary  # noqa: E402

GB, KIB = 10 ** 9, 1024
CONFIG = resolve_physical_config("hbm4_reference", overrides={
    "stacks": 1, "dies_per_stack": 12, "die_density_gbits": 24, "capacity_bytes": 36 * GB,
    "interface_bandwidth_gb_s": 1600., "read_bandwidth_gb_s": 1600., "write_bandwidth_gb_s": 1600.,
    "activate_different_group_ns": 4., "activate_same_group_ns": 6., "four_activate_window_ns": 20.})
ENV = ("HETEROLLM_HBM_TRACE_SUMMARY_FILE", "HETEROLLM_HBM_TRACE_FILE", "HETEROLLM_HBM_TRACE_LIMIT",
       "HETEROLLM_HBM_AUDIT_DISABLE_EQUIV")


def batches():
    """Scattered 64 KiB reads and writes over several refresh intervals."""
    rng = random.Random(3)
    slots = rng.sample(range(30 * GB // (64 * KIB)), 240)
    requests = [AccessRequest(f"r{index}", Operation.WRITE if index % 5 == 0 else Operation.READ,
                              slot * 64 * KIB, 64 * KIB, index * 150.) for index, slot in enumerate(slots)]
    return [requests[start:start + 40] for start in range(0, len(requests), 40)]


def run(monkeypatch, summary=None):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    if summary is not None:
        monkeypatch.setenv("HETEROLLM_HBM_TRACE_SUMMARY_FILE", str(summary))
        monkeypatch.setenv("HETEROLLM_HBM_TRACE_LIMIT", "0")
        monkeypatch.setenv("HETEROLLM_HBM_AUDIT_DISABLE_EQUIV", "1")
    core = DramCore(CONFIG, ResourceTimeline(), capture_details=False)
    core._hbm_numeric_enabled = True
    completions = [(result.request_id, result.completion_ns.hex())
                   for batch in batches() for result in core.submit_many(batch)]
    return core, completions


def totals(directory):
    names = hbm_audit_summary.rule_names()
    counts, checks, violations = {}, dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    files = sorted(directory.glob("commands.summary*"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split("\t")
            data = dict(part.split("=", 1) for part in fields[1:] if "=" in part)
            if fields[0] == "SUMMARY":
                for key, value in data.items():
                    counts[key] = counts.get(key, 0) + int(value)
            elif fields[0] == "AUDIT_DELTA":
                counts["audit_commands"] = counts.get("audit_commands", 0) + int(data["commands"])
                for name, check, fail in zip(names, data["checks"].split(","), data["violations"].split(",")):
                    checks[name] += int(check)
                    violations[name] += int(fail)
    return files, counts, checks, violations


def test_the_observer_checks_every_kernel_command_and_changes_nothing(tmp_path, monkeypatch):
    plain, plain_completions = run(monkeypatch)
    assert getattr(plain, "_hbm_audit", None) is None
    _audited, audited_completions = run(monkeypatch, tmp_path / "commands.summary")
    # The observer only reads: every completion is bit-identical.
    assert audited_completions == plain_completions
    files, counts, checks, violations = totals(tmp_path)
    assert len(files) == 1
    assert counts["total"] == counts["audit_commands"] > 0
    assert counts["RD"] > 0 and counts["WR"] > 0 and counts["ACT"] > 0 and counts["REFpb"] > 0
    assert counts["unaudited_batches"] == 0
    assert not any(violations.values()), {name: count for name, count in violations.items() if count}
    for rule in ("1.tRP", "2.tRCD", "3.tFAW", "3.tRRD", "4.WTR", "4.RTW", "6.tCCDL", "6.tCCDS",
                 "8.act_to_refpb", "10.refresh_order", "11.refresh_rate"):
        assert checks[rule] > 0, rule


def test_a_batch_that_bypasses_the_kernel_is_counted(tmp_path, monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HETEROLLM_HBM_TRACE_SUMMARY_FILE", str(tmp_path / "commands.summary"))
    monkeypatch.setenv("HETEROLLM_HBM_TRACE_LIMIT", "0")
    core = DramCore(CONFIG, ResourceTimeline(), capture_details=False)
    # A few bursts on a cold device run in the Python reference controller.
    core.submit_many([AccessRequest("tiny", Operation.READ, 0, 64)])
    _files, counts, _checks, _violations = totals(tmp_path)
    assert counts["unaudited_batches"] == 1 and counts["unaudited_requests"] == 1
