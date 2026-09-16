"""Structure-only tests: no freeze of real data, scenario simulation or native work."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r24_identity_driver_tests", HERE / "evaluate_identity_repair.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return d.s.reference(path)


def fake_prediction(ident, status="predicted", gpu=False):
    return {"cell_id": ident, "model_key": "qwen38_gpu" if gpu else "ordinary", "status": status,
        "aggregate": {m: {"median_ms": 10.0} for m in d.METRICS},
        "requests": [{"request_id": "request-0000", "request_index": 0, "engine_ttft_ms": 10.0,
                      "engine_tpot_ms": 1.0, "engine_e2e_ms": 17.0, "visible_output_tokens": 8}]}


def history():
    ids = ["cell-%03d" % i for i in range(131)]
    old23 = {ident: fake_prediction(ident, "predicted" if i < 104 else "failed", i >= 104) for i, ident in enumerate(ids)}
    old22 = {ident: fake_prediction(ident, "predicted" if i < 129 else "failed", i >= 104) for i, ident in enumerate(ids)}
    new = {ident: fake_prediction(ident, gpu=i >= 104) for i, ident in enumerate(ids)}
    return new, old23, old22


def test_freeze_requires_explicit_committed_source_before_any_native_or_freeze(monkeypatch):
    called = []
    monkeypatch.setattr(d, "native_lock", lambda: called.append("native"))
    with pytest.raises(ValueError, match="reviewed 40-character"):
        d.freeze(None)
    assert called == []


def test_normalized_proof_allows_alias_and_extractor_copy_only(tmp_path):
    one = put(tmp_path / "one.py", {"helper": 1})
    two = {**one, "path": str(tmp_path / "two.py")}
    a = {"extractor_ref": one, "model_scope": {"model_ref": {"path": str(tmp_path / "m.gguf"), "sha256": "a" * 64, "size_bytes": 5}},
         "contract": {"schema": "heterollm.retained-kv-state/v1", "evidence_sha256": "old", "rows": 7}}
    b = copy.deepcopy(a); b["extractor_ref"] = two
    b["model_scope"]["model_ref"]["bytes"] = b["model_scope"]["model_ref"].pop("size_bytes")
    b["contract"]["evidence_sha256"] = "new"
    assert d.normalized_proof(a, one) == d.normalized_proof(b, two)
    b["contract"]["rows"] = 8
    assert d.normalized_proof(a, one) != d.normalized_proof(b, two)


def test_normalized_proof_keeps_unknown_reference_aliases_and_paths(tmp_path):
    extractor = put(tmp_path / "extractor.py", {"helper": 1})
    unknown = put(tmp_path / "unknown.json", {"payload": 1})
    model = {"path": str(tmp_path / "model.gguf"), "sha256": "a" * 64, "size_bytes": 5}
    unknown_size_alias = {key: value for key, value in unknown.items() if key != "bytes"}
    unknown_size_alias["size_bytes"] = unknown["bytes"]
    a = {"extractor_ref": extractor, "model_scope": {"model_ref": model},
         "unknown_ref": unknown_size_alias, "contract": {"schema": "heterollm.retained-kv-state/v1", "evidence_sha256": "old"}}
    b = copy.deepcopy(a)
    b["unknown_ref"] = dict(unknown)
    # The model alias is approved; the unrelated reference alias is not.
    b["model_scope"]["model_ref"]["bytes"] = b["model_scope"]["model_ref"].pop("size_bytes")
    assert d.normalized_proof(a, extractor) != d.normalized_proof(b, extractor)


def test_normalized_proof_keeps_nested_schema_digest(tmp_path):
    extractor = put(tmp_path / "extractor.py", {"helper": 1})
    model = {"path": str(tmp_path / "model.gguf"), "sha256": "a" * 64, "bytes": 5}
    a = {"extractor_ref": extractor, "model_scope": {"model_ref": model},
         "contract": {"schema": "heterollm.retained-kv-state/v1", "evidence_sha256": "allowed"},
         "nested": {"schema": "heterollm.retained-kv-state/v1", "evidence_sha256": "must-remain"}}
    b = copy.deepcopy(a)
    b["nested"]["evidence_sha256"] = "changed"
    assert d.normalized_proof(a, extractor) != d.normalized_proof(b, extractor)


def test_source_guard_rejects_cost_changes(monkeypatch):
    old = {"freeze": {"tag": "old"}}
    new = {"freeze": {"tag": "new"}}
    monkeypatch.setattr(d.s, "source_content", lambda x: {"tools/predict_stable_native_dataset.py": x["tag"], "src/heterollm_sim/cost_models.py": x["tag"]})
    with pytest.raises(ValueError, match="source changes exceed"):
        d.verify_semantics(old, new)


def test_source_guard_rejects_source_membership_changes(monkeypatch):
    monkeypatch.setattr(d.s, "source_content", lambda x: {x["tag"]: "sha"})
    with pytest.raises(ValueError, match="membership"):
        d.verify_semantics({"freeze": {"tag": "old"}}, {"freeze": {"tag": "new"}})


def test_exact104_and_common25_comparison_keeps_original_two_failures():
    new, old23, old22 = history()
    result = d.paired_checks(new, old23, old22)
    assert result["r23_previous_104"]["exact_matches"] == 104
    assert result["r22_common_gpu_25"]["exact_matches"] == 25
    assert result["r23_failed_27_now_predicted"] == 27
    assert result["r22_sha_failures_not_retroactively_accepted"] == ["cell-129", "cell-130"]
    assert old22["cell-129"]["status"] == "failed"


def test_numeric_comparison_is_exact_not_toleranced_or_average():
    new, old23, old22 = history()
    new["cell-000"]["requests"][0]["engine_ttft_ms"] += 1e-12
    new["cell-105"]["status"] = "failed"
    result = d.paired_checks(new, old23, old22)
    assert result["r23_previous_104"]["exact_matches"] == 103
    assert result["r22_common_gpu_25"]["exact_matches"] == 24
    assert result["r23_failed_27_now_predicted"] == 26


def test_history_membership_cannot_be_redefined():
    new, old23, old22 = history()
    old23["cell-104"]["status"] = "predicted"
    with pytest.raises(ValueError, match="104/27"):
        d.paired_checks(new, old23, old22)


def test_score_cannot_run_without_durable_full_barrier(monkeypatch):
    calls = []
    monkeypatch.setattr(d, "guard", lambda: {})
    def missing(before):
        raise ValueError("missing full barrier")
    monkeypatch.setattr(d, "load_barrier", missing)
    monkeypatch.setattr(d, "run", lambda args: calls.append(args))
    with pytest.raises(ValueError, match="missing full barrier"):
        d.score()
    assert calls == []


def test_terminal_set_requires_exact_membership_and_rejects_pending(tmp_path):
    fr = put(tmp_path / "freeze.json", {"freeze": True})
    item = {"directory": tmp_path, "cells": {"a": {}, "b": {}}, "freeze_ref": fr, "freeze": {"source": {"sha256": "source"}}}
    p = {"cell_id": "a", "status": "failed", "freeze_ref": fr, "source_sha256": "source"}
    ref = put(tmp_path / "predictions/a.prediction.json", p)
    assert d.terminal_refs(item, complete=False) == {"a": ref}
    with pytest.raises(ValueError, match="131 terminal"):
        d.terminal_refs(item, complete=True)
    put(tmp_path / "predictions/b.prediction.json", {**p, "cell_id": "b", "status": "pending"})
    with pytest.raises(ValueError, match="nonterminal/corrupt"):
        d.terminal_refs(item, complete=True)


def test_full_preserves_existing_failed_terminal_and_writes_barrier_after_run(tmp_path, monkeypatch):
    output = tmp_path / "repaired"; output.mkdir()
    fr = put(output / "freeze.json", {"freeze": 1})
    cref = put(tmp_path / "controls.json", {"control": 1})
    ids = {"c%03d" % i: {} for i in range(131)}
    item = {"directory": output, "cells": ids, "freeze_ref": fr, "freeze": {"source": {"sha256": "source"}}}
    before = {"bundle": item, "controls_ref": cref}
    first = {"cell_id": "c000", "status": "failed", "freeze_ref": fr, "source_sha256": "source"}
    retained = put(output / "predictions/c000.prediction.json", first)
    calls = []
    monkeypatch.setattr(d, "P", tmp_path); monkeypatch.setattr(d, "OUTPUT", output)
    monkeypatch.setattr(d, "guard", lambda: before)
    monkeypatch.setattr(d, "check_end", lambda state: calls.append("checked-end"))
    monkeypatch.setattr(d.s, "load_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(d, "load_barrier", lambda state: calls.append("barrier-validated"))
    def simulate_command(args):
        assert "--resume" in args and "--score" not in args
        assert not (tmp_path / "full_predictions.json").exists()
        calls.append("predict")
        for ident in sorted(set(ids) - {"c000"}):
            put(output / "predictions" / (ident + ".prediction.json"), {**first, "cell_id": ident, "status": "predicted"})
    monkeypatch.setattr(d, "run", simulate_command)
    d.full()
    assert d.s.reference(output / "predictions/c000.prediction.json") == retained
    barrier = json.loads((tmp_path / "full_predictions.json").read_text())
    assert barrier["terminal_count"] == 131 and len(barrier["inputs"]["prediction_refs"]) == 131
    assert calls == ["predict", "checked-end", "barrier-validated"]


def test_existing_score_prevents_prediction_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "OUTPUT", tmp_path)
    put(tmp_path / "errors.0001.json", {})
    with pytest.raises(ValueError, match="scores exist"):
        d.reject_scores()


def test_phase_model_hashes_are_unique_not_per_cell(tmp_path, monkeypatch):
    model = put(tmp_path / "model.gguf", {"weights": 1})
    shared = {"prediction_model_ref": model, "runtime_ref": model}
    item = {"freeze_ref": model, "freeze": {"selection_ref": model, "retained_kv_warmup": {"model_identity_refs": [model]}},
        "cells": {str(i): {"static_inputs": shared} for i in range(131)}}
    checked = []
    monkeypatch.setattr(d.s, "source_content", lambda value: {})
    monkeypatch.setattr(d.s, "evidence_closure", lambda value: [])
    monkeypatch.setattr(d.s, "verify_reference", lambda ref: checked.append(ref) or d.s.normalized_ref(ref))
    d.verify_frozen_closure(item, whole_models=True)
    # freeze + selection + one unique runtime + one unique model, not 131 model hashes.
    assert len(checked) == 4


def test_recipe_retains_all_r23_mechanism_switches():
    ref = {"path": "/fixture/source.json"}
    baseline = {"data_root": "/fixture", "model_snapshot_map": {}, "runtime_build_audit": {"audit_ref": ref},
        "recurrent_batching": {"contract_ref": ref}, "slot_order": {"contract_ref": ref}, "host_offload_source": {"contract_ref": ref},
        "tensor_storage": {"contract_ref": ref}, "mmvq_issue_bound": {"hardware_document": {"ref": ref}}}
    args = d.recipe_arguments(baseline)
    for flag in ("tensor_storage_f32_hidden", "gpu_mmq_source_costs", "gpu_conversion_cta_costs", "mmvq_vector_issue_bound", "retained_kv_warmup_state"):
        assert args[flag] is True
    assert args["sampling_contract_path"] and args["nonflash_kv_view_source_contract_path"]


def test_commit_gate_checks_only_reviewed_replacement_and_driver_tests(monkeypatch):
    calls = []
    commit = "a" * 40
    def fake_git(*args):
        calls.append(args)
        if args[0] == "rev-parse":
            return (commit + "\n").encode()
        assert args[0] == "show"
        relative = args[1].split(":", 1)[1]
        return (d.ROOT / relative).read_bytes()
    monkeypatch.setattr(d, "git", fake_git)
    assert d.committed_source_gate(commit) == commit
    checked = [args[1].split(":", 1)[1] for args in calls if args[0] == "show"]
    assert checked == ["tools/predict_stable_native_dataset.py",
        Path(d.__file__).relative_to(d.ROOT).as_posix(), (d.P / "test_evaluate_identity_repair.py").relative_to(d.ROOT).as_posix()]
    assert "tools/render_stable_native_evaluation.py" not in checked


def test_snapshot_inherits_frozen_bytes_not_worktree_or_git_for_other_files(tmp_path, monkeypatch):
    frozen = tmp_path / "r23/source"
    working = tmp_path / "worktree"
    files = {"tools/predict_stable_native_dataset.py": b"old api\n",
        "tools/render_stable_native_evaluation.py": b"frozen renderer\r\n",
        "src/heterollm_sim/cost_models.py": b"original cost\n"}
    metadata = {}
    for relative, raw in files.items():
        path = frozen / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        actual = d.s.reference(path)
        metadata[relative] = {k: actual[k] for k in ("sha256", "bytes")}
        live = working / relative; live.parent.mkdir(parents=True, exist_ok=True); live.write_bytes(b"unrelated WIP must survive\n")
    extra = working / "src/untracked.py"; extra.write_text("not in frozen manifest")
    monkeypatch.setattr(d.s, "source_content", lambda value: metadata)
    monkeypatch.setattr(d, "git", lambda *args: b"committed repaired api\n")
    old = {"freeze": {"source": {"root": str(frozen)}}, "freeze_ref": {"sha256": "old"}}
    destination = tmp_path / "isolated"
    report = d.build_reviewed_source_snapshot(destination, old, "a" * 40)
    assert (destination / "tools/predict_stable_native_dataset.py").read_bytes() == b"committed repaired api\n"
    assert (destination / "tools/render_stable_native_evaluation.py").read_bytes() == files["tools/render_stable_native_evaluation.py"]
    assert (destination / "src/heterollm_sim/cost_models.py").read_bytes() == files["src/heterollm_sim/cost_models.py"]
    assert not (destination / "src/untracked.py").exists()
    assert (working / "tools/render_stable_native_evaluation.py").read_bytes() == b"unrelated WIP must survive\n"
    assert len(report["inherited"]) == 2 and len(report["replaced"]) == 1
    assert report["provenance"].startswith("R23 frozen")
