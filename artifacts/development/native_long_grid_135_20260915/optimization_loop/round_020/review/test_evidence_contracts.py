"""Static evidence-contract checks for the R20 independent review.

These tests read frozen summaries and protocols only.  They do not load native
libraries, launch benchmarks, or inspect per-call CPU timing samples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


LOOP = Path(__file__).resolve().parents[2]
R19 = LOOP / "round_019"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r6_rejected_quality_cannot_be_promoted_to_timing_evidence():
    quality = _json(R19 / "cpu_sampling_probe_r6/sampling_series_0001/quality.json")

    assert quality["stage_records_total"] == 36
    assert quality["steady_dispersion_pass_stages"] == 34
    assert quality["frequency_changed_stages"] == 0
    assert quality["timing_usable"] is False
    assert quality["diagnostic_only"] is True
    assert quality["accepted_for_timing_evidence"] is False
    assert "steady_dispersion" in quality["timing_quality_failure_reasons"]


def test_r6_run_freeze_and_quality_bind_the_same_actual_cpu_identity():
    series = R19 / "cpu_sampling_probe_r6/sampling_series_0001"
    freeze = _json(series / "run_freeze.json")
    quality = _json(series / "quality.json")
    identity = _json(R19 / "cpu_sampling_probe_r6/cpu_identity_r6_cpu0/actual_cpu_identity_freeze.json")

    identity_ref = freeze["actual_cpu_identity_freeze_ref"]
    assert _digest(Path(identity_ref["path"])) == identity_ref["sha256"]
    assert _digest(Path(quality["identity_freeze_ref"]["path"])) == quality["identity_freeze_ref"]["sha256"]
    assert Path(quality["identity_freeze_ref"]["path"]).resolve() == Path(identity_ref["path"]).resolve()
    assert identity["GPU_executed"] is False
    assert identity["model_loaded"] is False
    assert identity["timing_values_produced"] is False


def test_r6_build_manifest_binds_final_local_control_files():
    probe = R19 / "cpu_sampling_probe_r6"
    manifest = _json(probe / "build_manifest.json")
    inputs = {Path(item["path"]).resolve(): item for item in manifest["inputs"]}

    for name in ("cpu_sampling_probe.cpp", "entry.py", "protocol.json", "source_identity_r4.json"):
        path = (probe / name).resolve()
        assert path in inputs
        assert _digest(path) == inputs[path]["sha256"]
        assert path.stat().st_size == inputs[path]["bytes"]


def test_graph_pilot_counts_and_buffered_coverage_are_explicit():
    protocol = _json(R19 / "graph_gap_probe/protocol.json")
    execution = protocol["execution"]
    pilot = execution["pilot"]
    buffered = protocol["arms"]["buffered"]

    assert execution["calls_per_process"] == execution["first"] + execution["warmup"] + execution["formal"]
    assert pilot["native_processes"] == len(pilot["arms"]) * pilot["pairs_per_arm"] * len(pilot["modes"])
    assert pilot["expected_exports"] == len(pilot["arms"]) * pilot["pairs_per_arm"]
    assert buffered["write_cadence"] == "buffered_after_postformal"
    assert "does not prove every call" in buffered["numeric_coverage"]
