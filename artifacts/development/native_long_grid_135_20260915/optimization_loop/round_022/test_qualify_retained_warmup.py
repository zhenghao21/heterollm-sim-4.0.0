from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("qualify_retained_warmup.py")
SPEC = importlib.util.spec_from_file_location("qualify_retained_warmup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _fixture(tmp_path: Path, *, incomplete: bool = False, duplicate_slot: bool = False, wrong_count: bool = False, fake_raw_hash: bool = False) -> tuple[Path, str]:
    selector = tmp_path / "selector.py"
    selector.write_text("# static selector\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    _write_json(freeze, {"kind": "freeze"})
    receipt = tmp_path / "receipt.json"
    _write_json(receipt, {"kind": "receipt"})
    runtime = tmp_path / "runtime.json"
    _write_json(runtime, {"kind": "runtime"})
    plan = {
        "key": "cell__fixed__b00", "block": 0, "expected_prompt_tokens": 128,
        "output": 32, "parallel": 2, "kv_unified_per_slot": 2048, "ctx": 4096,
        "warmup_batches": 2, "measure_batches": 3, "process_blocks": 1,
        "flash_attention": False, "cache_ram_mib": 0, "batch": 64, "ubatch": 64,
    }
    def request(slot: int, prompt: int = 128) -> dict[str, object]:
        return {
            "status": "measured", "slot": slot,
            "response": {"truncated": False, "timings": {"prompt_n": prompt, "predicted_n": 32, "cache_n": 0}},
        }
    warmup = [
        {"status": "complete", "phase": "warmup", "process_block": 0, "requests": [request(0), request(1)]},
        {"status": "complete", "phase": "warmup", "process_block": 0, "requests": [request(0), request(1)]},
    ]
    if incomplete:
        warmup.pop()
    if duplicate_slot:
        warmup[0]["requests"][1]["slot"] = 0
    if wrong_count:
        warmup[0]["requests"][0]["response"]["timings"]["prompt_n"] = 127
    raw = {
        "key": plan["key"], "status": "complete", "warmup": warmup,
        "payload": {"cache_prompt": False, "n_predict": 32},
        "actual_argv": ["-np", "2", "-c", "4096", "-b", "64", "-ub", "64", "-fa", "off", "-kvu", "--cache-ram", "0", "--spec-type", "none"],
        "runtime_before": {"module_identity_sha256": "module", "process_identity": "process"},
        "runtime_after": {"module_identity_sha256": "module", "process_identity": "process"},
    }
    raw_path = tmp_path / "raw.json"
    _write_json(raw_path, raw)
    raw_ref = _ref(raw_path)
    if fake_raw_hash:
        raw_ref["sha256"] = "0" * 64
    cell_id = "qwen25_p128_o32_c2__fixed_runtime"
    cell = {
        "cell_id": cell_id, "model_key": "qwen25", "plans": [plan],
        "source_per_cell": {"freeze_ref": _ref(freeze)},
        "evidence_index": [{
            "raw_ref": raw_ref, "receipt_ref": _ref(receipt), "runtime_baseline_ref": _ref(runtime),
            "config_sha256": qualifier._canonical_sha256(plan),
        }],
    }
    selection = {
        "selected_cell_ids": [cell_id], "selected_cells": [cell], "payload_sha256": "payload-digest",
        "selector_source_ref": _ref(selector),
    }
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    return selection_path, cell_id


def _one_cell(selection_path: Path) -> dict[str, object]:
    return qualifier.derive_qualification(selection_path)["cells"][0]


def test_qualifies_complete_warmup_and_uses_actual_selection_bytes(tmp_path: Path) -> None:
    selection_path, _ = _fixture(tmp_path)
    result = qualifier.derive_qualification(selection_path)
    cell = result["cells"][0]
    assert result["source_contract"]["selection"]["sha256"] == _sha(selection_path)
    assert result["source_contract"]["selection"]["bytes"] == selection_path.stat().st_size
    assert cell["qualification"]["warmup_record_and_static_protocol"] == "qualified"
    assert cell["initial_retained_slot_template"]["retained_tokens_per_slot"] == 159
    assert cell["warmup_batches"][1]["slot_labels"] == [0, 1]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"incomplete": True}, "warmup_batch_count_not_two"),
        ({"duplicate_slot": True}, "warmup_0_qualification_failed"),
        ({"wrong_count": True}, "warmup_0_qualification_failed"),
        ({"fake_raw_hash": True}, "raw_record_declared_sha256_mismatch"),
    ],
)
def test_rejects_invalid_warmup_or_reference(tmp_path: Path, kwargs: dict[str, bool], reason: str) -> None:
    selection_path, _ = _fixture(tmp_path, **kwargs)
    cell = _one_cell(selection_path)
    assert cell["qualification"]["warmup_record_and_static_protocol"] == "not_qualified"
    assert cell["initial_retained_slot_template"] is None
    assert reason in cell["qualification"]["missing_or_failed"]


def test_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    selection_path, _ = _fixture(tmp_path)
    result = qualifier.derive_qualification(selection_path)
    output = tmp_path / "evidence.json"
    qualifier.write_qualification(result, output)
    with pytest.raises(FileExistsError):
        qualifier.write_qualification(result, output)


def test_result_excludes_latency_clock_and_timestamp_fields(tmp_path: Path) -> None:
    selection_path, _ = _fixture(tmp_path)
    result = qualifier.derive_qualification(selection_path)
    forbidden = ("metric", "duration", "clock", "timestamp", "engine_start", "wall_ms")

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert not any(token in str(key).casefold() for token in forbidden)
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(result)
