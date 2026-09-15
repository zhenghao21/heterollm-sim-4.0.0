"""Verify that an optimization loop still uses its original native truth."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

SHA256 = re.compile(r"[0-9a-f]{64}")


def verify_ref(ref):
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]:
        raise ValueError("missing native evidence path")
    if not isinstance(ref.get("sha256"), str) or SHA256.fullmatch(ref["sha256"]) is None:
        raise ValueError("missing or invalid native evidence SHA256")
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_file():
        raise ValueError("native evidence is not a file")
    expected_size = ref.get("bytes")
    if type(expected_size) is not int or expected_size <= 0 or path.stat().st_size != expected_size:
        raise ValueError("native evidence size mismatch: " + str(path))
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != ref["sha256"]:
        raise ValueError("native evidence changed: " + str(path))
    return path


def verify_lock(state_path):
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if state.get("schema") != "fixed-native-optimization-loop/v1":
        raise ValueError("unexpected optimization state schema")
    selection_path = verify_ref(state.get("native_selection_ref"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = selection.get("selected_cells")
    expected_count = state.get("native_selected_cells")
    if type(expected_count) is not int or expected_count <= 0 or not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("native selection membership/count changed")
    if state.get("native_remeasurement_allowed_in_this_loop") is not False:
        raise ValueError("fixed-native loop must prohibit remeasurement")
    refs = state.get("native_raw_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("native raw reference map is empty")
    bound = {}
    for ref in refs:
        path = verify_ref(ref)
        if str(path) in bound and bound[str(path)] != ref["sha256"]:
            raise ValueError("conflicting native raw identity")
        bound[str(path)] = ref["sha256"]
    request_count = 0
    used = set()
    ids = set()
    for row in rows:
        ident = row.get("cell_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("invalid or duplicate native cell identity")
        ids.add(ident)
        actuals = row.get("native_actuals")
        if not isinstance(actuals, list) or not actuals:
            raise ValueError("missing native actuals")
        for actual in actuals:
            ref = actual.get("raw_ref", {})
            path = str(Path(ref.get("path", "")).resolve())
            if not ref.get("sha256") or bound.get(path) != ref["sha256"]:
                raise ValueError("native actual is outside fixed raw reference map")
            used.add(path)
            request_count += 1
    if used != set(bound):
        raise ValueError("unused or missing native raw identity")
    return {"native_locked": True, "selection_sha256": state["native_selection_ref"]["sha256"],
            "selected_cells": expected_count, "raw_files": len(bound), "formal_requests": request_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_lock(args.state), ensure_ascii=False))


if __name__ == "__main__":
    main()
