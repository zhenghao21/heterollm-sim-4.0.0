"""R27 structure tests; no actual freeze, prediction, native, or target-error access."""
from pathlib import Path
import copy
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("r27_freeze_tests", Path(__file__).with_name("freeze_candidate.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return d.s.reference(path)


def pair(tmp_path, monkeypatch):
    monkeypatch.setattr(d.s, "unique_cells", lambda value: value["cells"])
    monkeypatch.setattr(d.s, "source_content", lambda value: value["source_bytes"])
    docs = []
    config = {key: 1 for key in d.CONFIG_KEYS}
    config.update(flash_attn=False, op_offload=True)
    for arm in ("off", "on"):
        source = tmp_path / arm / "source"
        extractor = write(source / "tools/retained_warmup_extractor.py", b"reviewed exact helper")
        pdf = write(tmp_path / arm / "evidence/hardware.pdf", b"reviewed exact PDF")
        retained = {"extractor_ref": extractor, "scope": "fixed"}
        contract = {"rows": 543, "evidence_sha256": d.s.stable_hash(retained)}
        retained["contract"] = contract
        inputs = {"config": config, "retained_kv_warmup_evidence": retained, "retained_kv_warmup_contract": copy.deepcopy(contract),
                  "mmvq_issue_evidence": {"hardware_document": {"ref": pdf}, "evidence_refs": [copy.deepcopy(pdf)]}}
        if arm == "on":
            inputs.update(final_output_selection=True, final_output_selection_binding={"status": "conditional", "config": config.copy()})
        docs.append({"source": {"root": str(source)}, "source_bytes": {"same": "sha"},
            "mmvq_issue_bound": {"hardware_document": {"ref": pdf}},
            "cells": {str(i): {"static_inputs": copy.deepcopy(inputs), "preparation_error": None} for i in range(131)}})
    return docs


def test_three_exact_arm_copy_differences_are_allowed(tmp_path, monkeypatch):
    off, on = pair(tmp_path, monkeypatch)
    assert d.compare_inputs(off, on) == 131


@pytest.mark.parametrize("change", ["raw_config", "unknown", "source", "proof_none", "bad_digest", "extractor_sha", "pdf_alias", "preparation"])
def test_other_differences_fail_closed(tmp_path, monkeypatch, change):
    off, on = pair(tmp_path, monkeypatch)
    inputs = on["cells"]["0"]["static_inputs"]
    if change == "raw_config": inputs["config"]["parallel"] = 4
    elif change == "unknown": inputs["not_ignored"] = True
    elif change == "source": on["source_bytes"]["same"] = "different"
    elif change == "proof_none": inputs["final_output_selection_binding"]["config"]["flash_attn"] = None
    elif change == "bad_digest": inputs["retained_kv_warmup_evidence"]["contract"]["evidence_sha256"] = "forged"
    elif change == "extractor_sha": inputs["retained_kv_warmup_evidence"]["extractor_ref"]["sha256"] = "a" * 64
    elif change == "pdf_alias": inputs["mmvq_issue_evidence"]["evidence_refs"][0]["sha256"] = "a" * 64
    elif change == "preparation": on["cells"]["0"]["preparation_error"] = "preserved preparation failure"
    with pytest.raises(ValueError): d.compare_inputs(off, on)


def test_no_recursive_path_normalization(tmp_path, monkeypatch):
    off, on = pair(tmp_path, monkeypatch)
    off["cells"]["0"]["static_inputs"]["unrelated"] = {"path": "a"}
    on["cells"]["0"]["static_inputs"]["unrelated"] = {"path": "b"}
    with pytest.raises(ValueError, match="non-treatment"):
        d.compare_inputs(off, on)


def test_snapshot_inherits113_exact_files_and_only_replaces_two(tmp_path, monkeypatch):
    source = tmp_path / "baseline/source"
    names = sorted(d.REPLACEMENTS) + ["src/file%03d.py" % i for i in range(113)]
    before = {}
    for name in names:
        ref = write(source / name, ("frozen " + name + "\r\n").encode())
        before[name] = {key: ref[key] for key in ("sha256", "bytes")}
    monkeypatch.setattr(d.s, "source_content", lambda _: before)
    read_blobs = []
    def replacement(commit, name):
        read_blobs.append((commit, name)); return ("committed " + name).encode()
    monkeypatch.setattr(d, "blob", replacement)
    refs = d.snapshot("a" * 40, {"source": {"root": str(source)}}, tmp_path / "new")
    assert len(refs) == 115 and {name for _, name in read_blobs} == d.REPLACEMENTS
    assert sum(row["origin"] == "R25_off_frozen" for row in refs) == 113
    assert (tmp_path / "new/src/file000.py").read_bytes() == (source / "src/file000.py").read_bytes()


def fake_arm(tmp_path, stale):
    directory = tmp_path / "on"
    source = directory / "source"
    api = '''
import json
from pathlib import Path
from tools.native_final_output_binding import CONFIG_KEYS
class Grid:
    pass
grid=Grid()
def configuration(inputs):return None
def gpu_clock(inputs):return None
def verify_freeze_references(frozen):
    Path(frozen['call_marker']).write_text('actual frozen verifier was called')
    for c in frozen['cells']:
        i=c['static_inputs']
        if i['final_output_selection_binding']['config']!={k:i['config'].get(k) for k in CONFIG_KEYS}:
            raise ValueError('cell proof differs from frozen static inputs')
'''
    helper = '''
CONFIG_KEYS=('flash_attn','op_offload')
def verify_cell(inputs,verify_files=True):return inputs['final_output_selection_binding']
'''
    write(source / "tools/__init__.py", b"")
    write(source / "tools/predict_stable_native_dataset.py", api.encode())
    write(source / "tools/native_final_output_binding.py", helper.encode())
    config = {"flash_attn": False, "op_offload": True}
    proof_config = {"flash_attn": None, "op_offload": None} if stale else config.copy()
    cells = [{"cell_id": str(i), "preparation_error": None, "static_inputs": {"config": config,
        "final_output_selection_binding": {"config": proof_config, "status": "conditional"}}} for i in range(131)]
    marker = tmp_path / "verifier_called.txt"
    frozen = {"source": {"root": str(source), "sha256": "synthetic"}, "selected_denominator": 131,
        "selection_sha256": "synthetic", "cells": cells, "call_marker": str(marker)}
    write(directory / "freeze.json", json.dumps(frozen).encode())
    write(tmp_path / "protocol.json", b"{}")
    return directory, marker


@pytest.mark.parametrize("stale", [False, True])
def test_fresh_preflight_executes_arm_verifier_and_rejects_old_none_proof(tmp_path, stale):
    directory, marker = fake_arm(tmp_path, stale)
    receipt = tmp_path / "preflight.json"
    process = subprocess.run([sys.executable, str(d.P / "verify_frozen_preflight.py"), "--freeze", str(directory / "freeze.json"),
        "--protocol", str(tmp_path / "protocol.json"), "--stage", "freeze", "--arm", "on", "--receipt", str(receipt)],
        capture_output=True, text=True)
    result = json.loads(receipt.read_text())
    assert marker.read_text() == "actual frozen verifier was called"
    assert (process.returncode != 0) is stale
    assert result["status"] == ("failed" if stale else "passed")
    assert result["verified_cells"] == (0 if stale else 131)
    if stale: assert "cell proof differs" in result["reason"]
    else: assert result["actual_api"]["path"] == str((directory / "source/tools/predict_stable_native_dataset.py").resolve())


def test_preflight_failure_attempts_both_arms_and_prevents_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    monkeypatch.setattr(d, "read_protocol", lambda: ({}, {}))
    calls = []
    def failure(args, **kwargs):
        calls.append(args[args.index("--arm") + 1]); return SimpleNamespace(returncode=1)
    monkeypatch.setattr(d.subprocess, "run", failure)
    with pytest.raises(ValueError, match="preflight failed"):
        d.run_preflights("lock")
    assert calls == ["off", "on"]
    assert not (tmp_path / "controls.json").exists()


def test_reference_only_preflight_receipt_cannot_claim_other_source(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    protocol_ref = write(tmp_path / "protocol.json", b"{}")
    freeze_ref = write(tmp_path / "on/freeze.json", b"{}")
    verifier = write(tmp_path / "verify_frozen_preflight.py", b"helper")
    actual_api = write(tmp_path / "on/source/tools/predict_stable_native_dataset.py", b"reviewed API")
    actual_binding = write(tmp_path / "on/source/tools/native_final_output_binding.py", b"reviewed binding")
    foreign = write(tmp_path / "foreign.py", b"another API")
    receipt = {"schema": "r27-frozen-source-preflight/v1", "stage": "lock", "arm": "on", "status": "passed", "verified_cells": 131,
        "protocol_ref": protocol_ref, "freeze_ref": freeze_ref, "verifier_ref": verifier,
        "actual_api": foreign, "actual_binding_helper": actual_binding}
    ref = write(tmp_path / "receipt.json", json.dumps(receipt).encode())
    with pytest.raises(ValueError, match="different frozen helper"):
        d.validate_preflight("lock", "on", ref, protocol_ref)


def test_reviewed_commit_required_before_snapshot_or_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(d, "snapshot", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="full reviewed commit"):
        d.freeze("3d0020a")
    assert calls == []


def test_protocol_ref_return_not_overwritten_by_helper_or_source_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "P", tmp_path)
    baseline = tmp_path / "baseline.json"; base_ref = write(baseline, b"{}")
    monkeypatch.setattr(d, "BASE", baseline)
    helper = tmp_path / "helper.py"; helper_ref = write(helper, b"helper bytes")
    copy_ref = write(tmp_path / "copied_helper.py", helper.read_bytes())
    source_ref = write(tmp_path / "source.py", b"source bytes")
    monkeypatch.setattr(d, "helper_paths", lambda: [helper])
    protocol = {"schema": "r27-final-selection-normalized-inputs/v1", "arms": ["off", "on"], "denominator": 131,
        "strict_threshold_pct": 10, "gate_B": "unvalidated", "new_cost_coefficients": 0, "baseline_ref": base_ref,
        "helpers": [helper_ref], "helper_copies": [copy_ref], "source_refs": [{"ref": source_ref}]}
    pr = write(tmp_path / "protocol.json", json.dumps(protocol).encode())
    assert d.read_protocol()[1] == pr


def test_new_off_must_match_each_R25_off_static_input(tmp_path, monkeypatch):
    baseline, off = pair(tmp_path, monkeypatch)
    for row in off["cells"].values():
        row["static_inputs"].pop("final_output_selection")
        row["static_inputs"].pop("final_output_selection_binding")
    assert d.compare_baseline_off(baseline, off) == 131
    off["cells"]["7"]["static_inputs"]["hidden_new_setting"] = True
    with pytest.raises(ValueError, match="R25/off static input changed"):
        d.compare_baseline_off(baseline, off)


def test_both_arms_agree_does_not_allow_common_baseline_config_drift(tmp_path, monkeypatch):
    baseline, unused = pair(tmp_path, monkeypatch)
    off = copy.deepcopy(baseline)
    off["cells"]["4"]["static_inputs"]["config"]["batch"] = 128
    with pytest.raises(ValueError, match="R25/off static input changed"):
        d.compare_baseline_off(baseline, off)
