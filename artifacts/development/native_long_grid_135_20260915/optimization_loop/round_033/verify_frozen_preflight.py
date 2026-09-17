"""Fresh-process static preflight using an arm's frozen Python, never prediction/native."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import importlib.machinery
from types import ModuleType
import json
import sys
import traceback


def reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def assert_imports_confined(root, modules=None):
    """Only real namespace packages may omit __file__; every search path is bound."""
    root = Path(root).resolve(strict=True)
    for name, module in list((sys.modules if modules is None else modules).items()):
        if not (name == "tools" or name.startswith("tools.") or name == "heterollm_sim" or name.startswith("heterollm_sim.")):
            continue
        if not isinstance(module, ModuleType):
            raise ValueError("unresolved frozen module origin: " + name)
        origin = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        if origin is not None:
            if not isinstance(origin, str) or not origin or not Path(origin).resolve(strict=True).is_relative_to(root):
                raise ValueError("import escaped arm source: " + name)
            if spec is None or spec.name != name or spec.origin is None or Path(spec.origin).resolve(strict=True) != Path(origin).resolve(strict=True):
                raise ValueError("module spec/file origin mismatch: " + name)
            continue
        # tools is normally a PEP420 namespace package in the Git snapshot.
        # A missing file on a normal module, an empty namespace, or even one
        # extra external namespace search location is never accepted.
        locations = list(getattr(spec, "submodule_search_locations", ()) or ())
        paths = list(getattr(module, "__path__", ()) or ())
        if (spec is None or spec.name != name or spec.origin is not None
                or not isinstance(spec.loader, importlib.machinery.NamespaceLoader)
                or not locations or locations != paths
                or any(not isinstance(path, str) or not path or not Path(path).resolve(strict=True).is_relative_to(root) for path in paths)):
            raise ValueError("unproven or escaped namespace origin: " + name)


def check(freeze_path, arm):
    freeze_path = Path(freeze_path).resolve(strict=True)
    before = reference(freeze_path)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    root = Path(frozen["source"]["root"]).resolve(strict=True)
    if root != (freeze_path.parent / "source").resolve():
        raise ValueError("frozen source root must be this arm's source copy")
    sys.path[:0] = [str(root / "src"), str(root)]
    spec = importlib.util.spec_from_file_location("r33_frozen_api", root / "tools/predict_stable_native_dataset.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    assert_imports_confined(root)
    # Do not mock or replace this call: this is the actual resume-start verifier.
    api.verify_freeze_references(frozen)
    from tools import native_final_output_binding as binding_helper
    verify_cell, CONFIG_KEYS = binding_helper.verify_cell, binding_helper.CONFIG_KEYS
    cells = frozen["cells"]
    if frozen.get("selected_denominator") != 131 or len(cells) != 131 or len({c["cell_id"] for c in cells}) != 131:
        raise ValueError("preflight requires exact131 cells")
    statuses = {}
    for cell in cells:
        if cell.get("preparation_error") is not None:
            raise ValueError("static preparation failed: " + cell["cell_id"] + ": " + str(cell["preparation_error"]))
        inputs = cell["static_inputs"]
        api.configuration(inputs)
        api.gpu_clock(inputs)
        expected_mode = {"off": "legacy_mma_output_wave", "on": "legacy_mma_output_wave"}[arm]
        if frozen.get("mmvq_hbm_mode", "legacy_mma_output_wave") != expected_mode or inputs.get("mmvq_hbm_mode", "legacy_mma_output_wave") != expected_mode:
            raise ValueError("frozen HBM mode differs: " + cell["cell_id"])
        if frozen.get("final_output_selection") is not True or inputs.get("final_output_selection") is not True:
            raise ValueError("final output selection must stay on in both arms")
        proof = verify_cell(inputs, verify_files=False)
        if proof is None or proof["config"] != {k: inputs["config"].get(k) for k in CONFIG_KEYS}:
            raise ValueError("normalized actual configuration differs: " + cell["cell_id"])
        statuses[proof["status"]] = statuses.get(proof["status"], 0) + 1
    assert_imports_confined(root)
    if reference(freeze_path) != before:
        raise ValueError("freeze changed during preflight")
    return {"verified_cells": len(cells), "mmvq_hbm_mode": expected_mode, "final_output_selection": True, "source_sha256": frozen["source"]["sha256"],
        "selection_sha256": frozen["selection_sha256"], "qualification_counts": statuses,
        "actual_api": reference(api.__file__),
        "actual_binding_helper": reference(binding_helper.__file__)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("freeze", "lock"))
    parser.add_argument("--arm", required=True, choices=("off", "on"))
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        raise FileExistsError("preflight receipt already exists")
    result = {"schema": "r33-frozen-source-preflight/v1", "stage": args.stage, "arm": args.arm,
        "started_utc": datetime.now(timezone.utc).isoformat(), "freeze_ref": reference(args.freeze),
        "protocol_ref": reference(args.protocol), "verifier_ref": reference(__file__),
        "native_run": False, "prediction_run": False, "target_errors_read": False}
    exit_code = 0
    try:
        result.update(check(args.freeze, args.arm), status="passed")
    except Exception as exc:
        result.update(status="failed", verified_cells=0, reason=type(exc).__name__ + ": " + str(exc))
        traceback.print_exc()
        exit_code = 1
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
