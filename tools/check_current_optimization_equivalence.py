"""Compare only this revision's three performance changes, keeping all current semantics.

This is a serial, in-process diagnostic, never a frontend submission or native benchmark.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import ctypes
import gc
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heterollm_sim import control_plane_planner, data_motion, ir, planner
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.llama_memory import LlamaDeviceMemoryCapacityError
from heterollm_sim.model_presets import list_model_presets, materialize_model_payload
from heterollm_sim.reporting import report_dict, run_scenario

DIRECTORY = ROOT / "docs/frontend_native_validation_2026-10-07"
TEMPLATES = {
    "local-native-rtx5080-9950x3d-gddr7-ddr5": "ui_before_fix_physical_io_local_qwen3_0_6b_submission.json",
    "nvidia-b200-1gpu-2hbf-2hbm": "ui_before_fix_physical_io_b200_2hbf_qwen3_0_6b_submission.json",
    "nvidia-b200-1gpu-3hbf-2hbm": "ui_before_fix_physical_io_b200_3hbf_qwen3_0_6b_submission.json",
}
VIEW_REPLACEMENTS = {
    "_direct_memory_address": [("len(_execution_layers(scenario))", "int(scenario.model.num_layers)", 1)],
    "_host_recurrent_offload_decision": [("_execution_view(scenario).architecture", "scenario.model.architecture", 1)],
    "_compile_parallel_embedding": [("_execution_view(scenario).vocabulary_size", "scenario.model.vocabulary_size", 2)],
    "_final_output_selection": [("_execution_view(scenario).architecture", "scenario.model.architecture", 1)],
    "_nonflash_kv_view_audit": [("_execution_view(scenario).architecture", "scenario.model.architecture", 1)],
}


def ast_replacement(function, replacements):
    """Change exact expression trees only; retain all current surrounding code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    rules = [(ast.dump(ast.parse(old, mode="eval").body, include_attributes=False),
              ast.parse(new, mode="eval").body, count, old, new)
             for old, new, count in replacements]
    counts = [0] * len(rules)

    class Transform(ast.NodeTransformer):
        def visit(self, node):
            for index, (expected, replacement, _, _, _) in enumerate(rules):
                if ast.dump(node, include_attributes=False) == expected:
                    counts[index] += 1
                    return ast.copy_location(deepcopy(replacement), node)
            return super().visit(node)

    changed = ast.fix_missing_locations(Transform().visit(tree))
    if counts != [row[2] for row in rules]:
        raise RuntimeError(f"Unrecognized source shape in {function.__qualname__}: {counts}")
    namespace = dict(function.__globals__)
    exec(compile(changed, f"<performance-only-baseline:{function.__qualname__}>", "exec"), namespace)
    return namespace[function.__name__], [
        {"function": function.__qualname__, "current": row[3], "baseline": row[4], "replacements": count}
        for row, count in zip(rules, counts)]


def build_baseline():
    functions, audit = {}, []
    for name, replacements in VIEW_REPLACEMENTS.items():
        result, changes = ast_replacement(getattr(planner, name), replacements)
        functions[(planner, name)] = result
        audit.extend(changes)
    result, changes = ast_replacement(control_plane_planner._derive_requirements, [
        ("execution_view.embedding_weight_bytes", "model.embedding_weight_bytes", 1),
        ("execution_view.output_weight_bytes", "model.output_weight_bytes", 1),
        ("execution_view.vocabulary_size", "model.vocabulary_size", 6),
    ])
    functions[(control_plane_planner, "_derive_requirements")] = result
    audit.extend(changes)
    snapshot, changes = ast_replacement(data_motion.PhysicalRuntimeContext.snapshot, [
        ("_BankState(bank.open_row, bank.ready_ns) if type(bank) is _BankState else copy.copy(bank)",
         "copy.copy(bank)", 1),
    ])
    audit.extend(changes)
    return functions, snapshot, audit


@contextmanager
def performance_mode(baseline, functions, old_snapshot, counters):
    original_snapshot = data_motion.PhysicalRuntimeContext.snapshot
    original_resolve = data_motion.resolve_physical_task
    original_guard = ir._model_graph_execution_payload

    def snapshot(runtime):
        counters["snapshot_calls"] += 1
        counters["bank_states_copied"] += sum(len(getattr(active.core, "_banks", {})) for active in runtime.runtimes.values())
        return (old_snapshot if baseline else original_snapshot)(runtime)

    def resolve(*args, **kwargs):
        if kwargs.get("_caller_managed_transaction") is True:
            counters["caller_managed_physical_resolves"] += 1
            if baseline:
                kwargs["_caller_managed_transaction"] = False
                counters["restored_duplicate_inner_snapshots"] += 1
        return original_resolve(*args, **kwargs)

    def guard(*args, **kwargs):
        counters["execution_graph_guard_calls"] += 1
        return original_guard(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(data_motion.PhysicalRuntimeContext, "snapshot", snapshot))
        stack.enter_context(patch.object(data_motion, "resolve_physical_task", resolve))
        stack.enter_context(patch.object(ir, "_model_graph_execution_payload", guard))
        if baseline:
            for (module, name), function in functions.items():
                stack.enter_context(patch.object(module, name, function))
        yield


def load_templates(directory):
    templates = {}
    for hardware_id, filename in TEMPLATES.items():
        payload = json.loads((directory / filename).read_text(encoding="utf-8"))["scenario"]
        hardware = payload["hardware_input"]["hardware"]
        if hardware["metadata"]["architecture_preset"]["id"] != hardware_id:
            raise ValueError(f"Actual frontend hardware template does not match {hardware_id}")
        if payload["profiles"]["llama_cpp"]["context"] != 640:
            raise ValueError("Expected the actual 640-context default llama frontend template")
        if len(payload["workload"]["requests"]) != 1 or payload["workload"].get("mtp") is not None:
            raise ValueError("Expected one non-speculative frontend request")
        templates[hardware_id] = payload
    return templates


def short_payload(template, model_id):
    payload = deepcopy(template)
    payload["model"] = materialize_model_payload(model_id)
    # Same model-dependent clearing as applyPresetDetailToScenario in app.js.
    placement = payload["placement"]
    placement["model_name"] = payload["model"]["name"]
    for key in ("op_to_component", "tensor_to_component", "tensor_bytes"):
        placement[key] = {}
    placement["metadata"].pop("control_plane", None)
    placement["parallel"]["layer_to_stage"] = {}
    workload = payload["workload"]
    workload["prompt_tokens"], workload["output_tokens"] = 1, 2
    workload["requests"][0]["prompt_tokens"] = 1
    workload["requests"][0]["output_tokens"] = 2
    return payload


def run_one(payload, baseline, functions, snapshot):
    counters = dict(snapshot_calls=0, bank_states_copied=0, caller_managed_physical_resolves=0,
                    restored_duplicate_inner_snapshots=0, execution_graph_guard_calls=0)
    started = time.perf_counter()
    try:
        with performance_mode(baseline, functions, snapshot, counters):
            report = report_dict(run_scenario(scenario_from_dict(deepcopy(payload)), retention_policy="aggregate"))
        summary = report["summary"]
        if summary.get("completed_requests") != 1 or summary.get("batch_count") != 2:
            raise RuntimeError("Short non-speculative request did not finish both cohorts")
        status = {"status": "completed", "summary": {
            key: summary.get(key) for key in ("engine_ttft_ns", "engine_tpot_ns", "engine_e2e_ns", "total_energy_pj", "task_count", "batch_count")},
            "physical_totals": {kind: {k: v for k, v in summary.get(kind, {}).items() if isinstance(v, (int, float))}
                                for kind in ("dram_traffic", "storage_traffic")}}
    except LlamaDeviceMemoryCapacityError as error:
        report = None
        status = {"status": "capacity_rejected", "exception_type": type(error).__name__, "message": str(error)}
    except Exception as error:
        report = None
        status = {"status": "execution_error", "exception_type": type(error).__name__, "message": str(error)}
    status["elapsed_seconds"] = time.perf_counter() - started
    status["performance_counters"] = counters
    return report, status


def compare_reports(before, after):
    result = {"numeric_fields_compared": 0, "nonnumeric_fields_compared": 0, "numeric_mismatch_count": 0,
              "other_mismatch_count": 0, "max_absolute_difference": 0, "max_relative_difference": 0,
              "undefined_relative_difference_count": 0, "mismatches": [], "nonnumeric_mismatches": []}

    def difference(target, row):
        if len(result[target]) < 30:
            result[target].append(row)

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                result["other_mismatch_count"] += 1
                difference("nonnumeric_mismatches", {"path": path, "only_before": sorted(set(left)-set(right)), "only_after": sorted(set(right)-set(left))})
            for key in sorted(set(left) & set(right)):
                compare(left[key], right[key], path + "." + str(key))
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                result["other_mismatch_count"] += 1
                difference("nonnumeric_mismatches", {"path": path, "before_length": len(left), "after_length": len(right)})
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, path + f"[{index}]")
        elif type(left) in (int, float) and type(right) in (int, float):
            result["numeric_fields_compared"] += 1
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"Nonfinite numeric report field: {path}")
            if left != right:
                delta = abs(right-left)
                relative = delta / abs(left) if left else None
                result["numeric_mismatch_count"] += 1
                result["max_absolute_difference"] = max(result["max_absolute_difference"], delta)
                if relative is None:
                    result["undefined_relative_difference_count"] += 1
                else:
                    result["max_relative_difference"] = max(result["max_relative_difference"], relative)
                difference("mismatches", {"path": path, "before": left, "after": right, "absolute_difference": delta, "relative_difference": relative})
        else:
            result["nonnumeric_fields_compared"] += 1
            if type(left) != type(right) or left != right:
                result["other_mismatch_count"] += 1
                difference("nonnumeric_mismatches", {"path": path, "before": left, "after": right})

    compare(before, after, "report")
    result["all_report_fields_identical"] = result["numeric_mismatch_count"] == result["other_mismatch_count"] == 0
    return result


def set_background_priority():
    if sys.platform != "win32":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.SetPriorityClass.restype = ctypes.c_int
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "optimization_matrix_equivalence.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--semantic-stage", default="unspecified",
                        help="Describe the shared semantic implementation used by both variants")
    args = parser.parse_args()
    below_normal = set_background_priority()
    functions, snapshot, audit = build_baseline()
    templates = load_templates(args.directory)
    models = [row["id"] for row in sorted(list_model_presets(), key=lambda row: (float(row["parameter_scale"].split("B")[0]), row["id"]))]
    if len(models) != 21 or len(templates) != 3:
        raise ValueError("Expected this validation's 21 by 3 preset catalog")
    data = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.is_file() else {
        "schema": "current-optimization-matrix-equivalence/v1", "status": "running",
        "semantic_stage": args.semantic_stage,
        "scope": "Current semantic code, with only three performance implementations toggled together; serial in-process diagnostics, not frontend validation",
        "workload": {"prompt_tokens": 1, "output_tokens": 2, "requests": 1, "llama_context": 640, "batch": 512, "ubatch": 512},
        "hardware_templates": TEMPLATES, "models": models, "expected_cases": 63,
        "baseline_expression_changes": audit,
        "baseline_transaction_change": "Only force _caller_managed_transaction=False at data_motion.resolve_physical_task; retain the current outer rollback and every current semantic operation",
        "comparison_exclusions": [],
        "limitations": ["All three performance changes are toggled together; prior isolated evidence remains separate.",
                        "The complete returned report_dict is compared, including its default bounded visualization; this is not an unbounded event trace.",
                        "One input/two output tokens cover both prefill and decode, not the formal 512/128 workload or longer queue/refresh interactions.",
                        "Capacity rejection is consistent rejection, never a successful prediction equivalence case.",
                        "Wall-clock execution time and instrumentation counters are outside the report and never treated as model prediction or as a speedup measurement."],
        "process_below_normal_priority": below_normal, "cases": [],
    }
    if data["baseline_expression_changes"] != audit:
        raise ValueError("Cannot resume across a different performance-only baseline definition")
    if data.get("semantic_stage", "unspecified") != args.semantic_stage:
        raise ValueError("Cannot resume across a different semantic stage")
    if args.resume:
        for row in data["cases"]:
            row.setdefault("process_below_normal_priority", data["process_below_normal_priority"])
    done = {(row["hardware_id"], row["model_id"]) for row in data["cases"]}
    count = 0
    for hardware_id, template in templates.items():
        for model_id in models:
            if (hardware_id, model_id) in done:
                continue
            print(json.dumps({"event": "start", "hardware_id": hardware_id, "model_id": model_id}), flush=True)
            payload = short_payload(template, model_id)
            before, before_status = run_one(payload, True, functions, snapshot)
            after, after_status = run_one(payload, False, functions, snapshot)
            row = {"hardware_id": hardware_id, "model_id": model_id, "before": before_status, "after": after_status,
                   "process_below_normal_priority": below_normal}
            if before is not None and after is not None:
                row.update(compare_reports(before, after))
                row["status"] = "identical" if row["all_report_fields_identical"] else "difference"
            elif before_status["status"] == after_status["status"] == "capacity_rejected" and before_status["message"] == after_status["message"]:
                row["status"] = "consistent_capacity_rejection"
            else:
                row["status"] = "execution_error_or_status_difference"
            data["cases"].append(row)
            data["status"] = "completed" if len(data["cases"]) == data["expected_cases"] else "running"
            data["summary"] = {state: sum(r["status"] == state for r in data["cases"]) for state in sorted({r["status"] for r in data["cases"]})}
            data["numeric_fields_compared"] = sum(r.get("numeric_fields_compared", 0) for r in data["cases"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"event": "result", "hardware_id": hardware_id, "model_id": model_id, "status": row["status"],
                              "numeric_fields_compared": row.get("numeric_fields_compared"), "mismatches": row.get("mismatches", []),
                              "errors": [s.get("message") for s in (before_status, after_status) if s.get("message")]}), flush=True)
            del before, after, payload
            gc.collect()
            count += 1
            if row["status"] not in {"identical", "consistent_capacity_rejection"}:
                raise SystemExit(1)
            if args.limit and count >= args.limit:
                return


if __name__ == "__main__":
    main()
