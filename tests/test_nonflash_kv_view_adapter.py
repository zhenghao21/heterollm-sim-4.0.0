"""Frozen opt-in and source/build binding checks without native/GPU execution."""
import copy
import json
from pathlib import Path

import pytest
from tools import predict_stable_native_dataset as adapter
from tests.test_predict_stable_native_dataset import fixture, document

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "artifacts/development/native_long_grid_135_20260915/optimization_loop"


def test_real_source_contract_rederives_from_locked_build_chain():
    binding_path = ROUND / "round_000/dependencies/runtime_source_binding_structural_audit.json"
    contract_path = ROUND / "round_000/dependencies/nonflash_kv_view_source_contract.json"
    if not binding_path.is_file() or not contract_path.is_file():
        pytest.skip("baseline source/build receipts are local evidence")
    binding = adapter.verified_host_offload_source_contract(binding_path, [], ROOT)
    canonical = adapter.derive_nonflash_kv_view_contract(binding, ROOT)
    assert canonical == json.loads(contract_path.read_text(encoding="utf-8"))
    assert canonical["n_pad"] == 1 and canonical["n_kv_padding"] == 256
    assert canonical["source_compilation"]["context"]["source"].endswith("llama-context.cpp")
    assert "annotation-control" in canonical["source_compilation"]["context"]["source"]
    assert canonical["native_latency_used"] is False


def test_verifier_binds_selected_config_and_rejects_changed_contract(tmp_path, monkeypatch):
    _, selection, row, _ = fixture(tmp_path, monkeypatch)
    canonical = {"schema": "heterollm.llama-nonflash-kv-view/v1", "n_pad": 1,
        "n_kv_padding": 256, "source_sha256": {"cache": "a" * 64}, "evidence_refs": []}
    path = tmp_path / "contract.json"
    document(path, canonical)
    binding = {"cells": {row["cell_id"]: {"status": "verified"}}, "evidence_refs": []}
    monkeypatch.setattr(adapter, "derive_nonflash_kv_view_contract", lambda *args: copy.deepcopy(canonical))
    result = adapter.verified_nonflash_kv_view_contract(path, [row], tmp_path, runtime_binding=binding)
    cell = result["cells"][row["cell_id"]]
    assert cell["configuration"]["native_context_tokens"] == 4096
    assert cell["configuration"]["simulator_slot_context_tokens"] == 2048
    assert cell["runtime_binding_status"] == "verified"
    changed = copy.deepcopy(canonical)
    changed["n_kv_padding"] = 128
    document(path, changed)
    with pytest.raises(ValueError, match="re-derived"):
        adapter.verified_nonflash_kv_view_contract(path, [row], tmp_path, runtime_binding=binding)
    document(path, canonical)
    binding["cells"][row["cell_id"]]["status"] = "conditional"
    with pytest.raises(ValueError, match="verified selected native runtime"):
        adapter.verified_nonflash_kv_view_contract(path, [row], tmp_path, runtime_binding=binding)


def test_freeze_persists_opt_in_and_worker_uses_frozen_value(tmp_path, monkeypatch):
    selection_path, selection, row, calls = fixture(tmp_path, monkeypatch)
    contract = {"schema": "heterollm.llama-nonflash-kv-view/v1", "configuration": {"batch": 64}}
    evidence = {"contract": contract, "cells": {row["cell_id"]: contract}, "evidence_refs": []}
    monkeypatch.setattr(adapter, "verified_nonflash_kv_view_contract", lambda *args, **kwargs: evidence)
    freeze = adapter.freeze_selection(selection_path, tmp_path / "freeze", data_root=tmp_path,
        nonflash_kv_view_source_contract_path=tmp_path / "contract.json")
    assert freeze["nonflash_kv_view"] == evidence
    frozen = freeze["cells"][0]["static_inputs"]
    assert frozen["nonflash_kv_view_contract"] == contract
    adapter.verify_freeze_references(freeze)
    adapter.predict_cell(frozen)
    installed = calls["run"][0].workload.metadata["llama_cpp_nonflash_kv_view"]
    assert installed["schema"] == contract["schema"] and installed["configuration"] == contract["configuration"]
    baseline = adapter.static_inputs(row, selection, tmp_path)
    assert baseline["nonflash_kv_view_contract"] is None
    assert adapter.grid.stable_hash(baseline) != adapter.grid.stable_hash(frozen)


def test_resume_cannot_change_physical_view_switch(tmp_path):
    with pytest.raises(SystemExit) as exc:
        adapter.main(["predict", "--output", str(tmp_path), "--resume", "--nonflash-kv-view-source-contract", "new.json"])
    assert exc.value.code == 2
