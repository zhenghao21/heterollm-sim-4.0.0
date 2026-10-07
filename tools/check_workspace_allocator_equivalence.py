"""Isolate arena candidate filtering; preserve every capacity/cost rule."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import textwrap
import time

from check_current_optimization_equivalence import (
    DIRECTORY, compare_reports, load_templates, short_payload,
)
from heterollm_sim import memory_allocator as allocator_module
from heterollm_sim.memory_allocator import PhysicalAddressAllocator, AllocationError
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.web import scenario_or_http_error


def previous_first_fit():
    tree = ast.parse(textwrap.dedent(inspect.getsource(PhysicalAddressAllocator._find_first_fit)))
    matches = [index for index, node in enumerate(tree.body[0].body)
               if isinstance(node, ast.If) and ast.unparse(node.test) == "self.workspace_capacity_bytes"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one arena filtering branch")
    tree.body[0].body[matches[0]] = ast.parse("candidates = self.allocations()").body[0]
    ast.fix_missing_locations(tree)
    namespace = dict(vars(allocator_module))
    exec(compile(tree, "<previous arena first-fit>", "exec"), namespace)
    return namespace["_find_first_fit"]


def allocation_sequence(method):
    PhysicalAddressAllocator._find_first_fit = method
    allocator = PhysicalAddressAllocator(36_000_000_000, 64,
                                        workspace_capacity_bytes=128 * 1024 * 1024)
    for index in range(2300):
        allocator.allocate("resident-{}".format(index), 1024 * 1024, 0)
    events = []
    started = time.perf_counter()
    for iteration in range(600):
        identities = []
        for index, size in enumerate((32 * 1024 * 1024, 64, 16 * 1024 * 1024, 8 * 1024 * 1024)):
            identity = "activation-{}-{}".format(iteration, index)
            item = allocator.allocate(identity, size, iteration + 1)
            identities.append(identity)
            events.append(("allocate", item.buffer_id, item.base_address, item.size_bytes, item.generation))
        alias = allocator.allocate("view", 1024, iteration + 1,
            alias_of=identities[0], alias_offset_bytes=128)
        events.append(("alias", alias.base_address))
        try:
            allocator.allocate("over-capacity", 128 * 1024 * 1024, iteration + 1)
        except AllocationError as exc:
            events.append(("rejection", str(exc)))
        else:
            raise AssertionError("Over-capacity operation unexpectedly succeeded")
        allocator.release("view", iteration + 1)
        for index in (1, 3, 0, 2):
            allocator.release(identities[index], iteration + 1)
    return events, time.perf_counter() - started


def main():
    current = PhysicalAddressAllocator._find_first_fit
    previous = previous_first_fit()
    output = DIRECTORY / "workspace_allocator_equivalence.json"
    data = {"schema": "workspace-allocator-equivalence/v1",
        "scope": "Only arena candidate filtering before sorting; all workspace placement, arena boundaries, physical costs and HBF energy fixes held identical.",
        "comparison_exclusions": [], "models": []}
    try:
        before, before_seconds = allocation_sequence(previous)
        after, after_seconds = allocation_sequence(current)
        if before != after:
            raise AssertionError("Allocator request sequence differs")
        data["allocator_sequence"] = {"resident_allocations": 2300,
            "allocation_alias_rejection_events_compared": len(before),
            "identical": True, "address_difference_bytes": 0,
            "before_seconds": before_seconds, "after_seconds": after_seconds,
            "speedup": before_seconds / after_seconds,
            "scope": "Fixed synthetic lifetime sequence; wall time is a microbenchmark, not a model prediction."}
        template = load_templates(DIRECTORY)["nvidia-b200-1gpu-2hbf-2hbm"]
        for model in ("qwen3-32b", "qwen2_5-32b"):
            PhysicalAddressAllocator._find_first_fit = current
            payload = short_payload(template, model)
            payload["workload"]["prompt_tokens"] = 512
            payload["workload"]["output_tokens"] = 1
            payload["workload"]["requests"][0].update(prompt_tokens=512, output_tokens=1)
            scenario = scenario_or_http_error(payload)
            reports = []
            wall = []
            for method in (previous, current):
                PhysicalAddressAllocator._find_first_fit = method
                started = time.perf_counter()
                report = report_dict(run_scenario(scenario, retention_policy="aggregate"))
                wall.append(time.perf_counter() - started)
                if report["summary"]["completed_requests"] != 1:
                    raise AssertionError("Whole-model request did not finish")
                reports.append(report)
            result = compare_reports(*reports)
            result.update(model_id=model, before_seconds=wall[0], after_seconds=wall[1],
                workload={"prompt_tokens": 512, "output_tokens": 1, "context": 640})
            data["models"].append(result)
            output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"model": model, "identical": result["all_report_fields_identical"],
                "numeric_fields": result["numeric_fields_compared"], "wall_seconds": wall}), flush=True)
            if not result["all_report_fields_identical"]:
                raise AssertionError("Whole-model reports differ")
        data["status"] = "completed"
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        PhysicalAddressAllocator._find_first_fit = current


if __name__ == "__main__":
    main()
