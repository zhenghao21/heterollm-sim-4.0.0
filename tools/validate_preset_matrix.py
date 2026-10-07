"""Run unchanged frontend preset combinations through the background-job API."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from heterollm_sim.architecture_presets import architecture_preset_detail
from heterollm_sim.model_presets import list_model_presets, materialize_model_payload
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.web import scenario_to_payload


LLAMA_DEFAULTS = {
    "policy": "llama_cpp", "gpu_layers": -1, "batch": 512, "ubatch": 512,
    "context": 640, "parallel": 1, "offload_kqv": True, "device_memory_tiering": True,
}


def scenario_for(hardware_id: str, model_id: str) -> dict:
    base = scenario_to_payload(build_llama_default_scenario())
    detail = architecture_preset_detail(hardware_id)
    hardware = copy.deepcopy(detail["hardware"])
    payload = copy.deepcopy(base)
    payload["hardware"] = copy.deepcopy(hardware)
    payload["model"] = materialize_model_payload(model_id)
    parameters = {
        key: copy.deepcopy(base["profiles"][key])
        for key in ("host_orchestration", "fusion", "runtime")
    }
    parameters["cim_interconnect"] = copy.deepcopy(base["profiles"].get("cim_interconnect"))
    runtime = parameters["runtime"]
    old_gpu = next(iter(runtime.get("gpu_controllers", {}).values()), {})
    runtime["gpu_controllers"] = {
        item["component_id"]: copy.deepcopy(old_gpu)
        for item in hardware["components"] if item.get("kind") == "gpu"
    }
    profile_kinds = {
        "gpu": "gpu", "cpu": "cpu", "host_memory": "host_memory",
        "hbm": "hbm", "gddr": "gddr", "hbf": "host_memory",
    }
    for component in hardware["components"]:
        metadata = component.setdefault("metadata", {})
        template = metadata.pop("cost_profile_template", None)
        profile_kind = profile_kinds.get(component.get("kind"), component.get("kind"))
        if template is None:
            source = base["profiles"]["components"][profile_kind]
            template = copy.deepcopy(source.get(component.get("cost_profile_id"), next(iter(source.values()))))
        component.pop("cost_profile_id", None)
        component["execution_profile"] = {
            "profile_id": f"{component['component_id']}.matrix",
            "profile_kind": profile_kind, "parameters": template,
        }
    hardware["parameters"] = parameters
    payload["hardware_input"] = {
        "schema_version": "4.0", "kind": "hardware_input", "contract_version": "2", "hardware": hardware,
    }
    payload["profiles"]["llama_cpp"] = copy.deepcopy(LLAMA_DEFAULTS)
    if hardware_id.startswith("local-native-"):
        payload["workload"].setdefault("metadata", {})["llama_cpp_kernel_model_preset"] = "blackwell_analytical_v1"
    payload["name"] = f"exact-matrix/{hardware_id}/{model_id}"
    placement = payload["placement"]
    placement["hardware_name"], placement["model_name"] = hardware["name"], payload["model"]["name"]
    for key in ("op_to_component", "tensor_to_component", "tensor_bytes"):
        placement[key] = {}
    parallel = placement.setdefault("parallel", {})
    parallel.update(rank_mapping=[], layer_to_stage={}, tp_degree=1, pp_degree=1, ep_degree=1)
    placement.setdefault("kv_policy", {}).update(cache_component=None, offload_component=None, pool_components=[])
    assert len(payload["workload"]["requests"]) == 1
    request = payload["workload"]["requests"][0]
    assert request["prompt_tokens"] == 512 and request["output_tokens"] == 128
    return payload


def api(base_url: str, path: str, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            result = {"error": {"message": body}}
        return error.code, result


def report_checks(snapshot: dict) -> list[str]:
    problems = []
    report = snapshot.get("report") or {}
    summary = report.get("summary") or {}
    if not snapshot.get("report"):
        problems.append("completed job has no report")
    if snapshot.get("error") is not None:
        problems.append("completed job has an error")
    if summary.get("completed_requests") != 1:
        problems.append("completed_requests must be 1")
    if summary.get("rejected_requests") != 0:
        problems.append("rejected_requests must be 0")
    requests = report.get("requests") or {}
    rows = list(requests.values()) if isinstance(requests, dict) else requests
    if len(rows) != 1 or rows[0].get("visible_output_tokens") != 128 or rows[0].get("status") != "finished":
        problems.append("the single request must finish with 128 visible output tokens")
    if summary.get("batch_count") != 128:
        problems.append("default non-speculative workload must execute 128 cohorts")
    if not isinstance(summary.get("task_count"), int) or summary["task_count"] <= 0:
        problems.append("task_count must be a positive integer")
    for key in ("makespan_ns",):
        value = summary.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            problems.append(f"{key} must be finite and positive")
    for key in ("dram_traffic", "storage_traffic"):
        traffic = summary.get(key) or {}
        for metric in ("physical_bytes", "service_ns"):
            value = traffic.get(metric)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                problems.append(f"{key}.{metric} must be finite and nonnegative")
        if traffic.get("task_count", 0) > 0:
            for metric in ("physical_bytes", "service_ns"):
                if metric not in traffic:
                    problems.append(f"{key}.{metric} is missing for executed physical tasks")
        for total, read, write in (("physical_bytes", "physical_read_bytes", "physical_write_bytes"),
                                   ("logical_bytes", "logical_read_bytes", "logical_write_bytes")):
            if all(metric in traffic for metric in (total, read, write)) and traffic[total] != traffic[read] + traffic[write]:
                problems.append(f"{key}.{total} disagrees with read/write totals")
    dram = summary.get("dram_traffic") or {}
    row_keys = ("row_hits", "row_misses", "row_conflicts")
    if "burst_count" in dram and all(key in dram for key in row_keys):
        if dram["burst_count"] != sum(dram[key] for key in row_keys):
            problems.append("DRAM row classifications disagree with burst_count")
    mtp = summary.get("mtp") or {}
    if mtp.get("enabled") is not False or mtp.get("proposed_tokens", 0) != 0:
        problems.append("default workload must not enable speculative drafting")
    return problems


def run_case(base_url: str, hardware_id: str, model_id: str) -> dict:
    started = time.monotonic()
    row = {"hardware": hardware_id, "model": model_id, "execution_method": "frontend_run_job_api"}
    try:
        payload = scenario_for(hardware_id, model_id)
        row["input"] = {"requests": 1, "prompt_tokens": 512, "output_tokens": 128,
                        "llama_cpp": copy.deepcopy(LLAMA_DEFAULTS), "retention_policy": "aggregate"}
        code, snapshot = api(base_url, "/api/run-jobs", {"scenario": payload, "retention_policy": "aggregate"})
        if code != 202:
            row.update(status="submission_failed", http_status=code, error=snapshot)
        else:
            row["job_id"] = snapshot["job_id"]
            last_log = started
            while snapshot["status"] not in {"completed", "failed", "cancelled"}:
                now = time.monotonic()
                if now - last_log >= 40:
                    print("PROGRESS " + json.dumps({"hardware": hardware_id, "model": model_id,
                          "elapsed_s": round(now - started, 1), "status": snapshot["status"],
                          "progress": snapshot.get("progress"), "detail": snapshot.get("progress_detail")},
                          ensure_ascii=False), flush=True)
                    last_log = now
                time.sleep(2)
                code, snapshot = api(base_url, "/api/run-jobs/" + row["job_id"])
                if code != 200:
                    raise RuntimeError(f"job polling returned {code}: {snapshot}")
            report = snapshot.get("report") or {}
            row.update(status=snapshot["status"], error=snapshot.get("error"),
                       report_available=bool(report), summary=report.get("summary"),
                       requests=report.get("requests"), validation_warnings=report.get("validation_warnings", []),
                       validation_information=report.get("validation_information", []),
                       started_at=snapshot.get("started_at"), finished_at=snapshot.get("finished_at"))
            if row["status"] == "completed":
                row["validation_failures"] = report_checks(snapshot)
                if row["validation_failures"]:
                    row["status"] = "validation_failed"
        if row["status"] != "completed" and "llama.cpp shared device-memory capacity exhausted" in json.dumps(row.get("error"), ensure_ascii=False):
            row["status"] = "capacity_rejected"
    except Exception as error:
        row.update(status="runner_failed", error={"exception_type": type(error).__name__, "message": str(error)})
    row["elapsed_s"] = round(time.monotonic() - started, 3)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    catalog = list_model_presets()
    models = args.models or [row["id"] for row in sorted(catalog, key=lambda row: (float(row["parameter_scale"].split("B")[0]), row["id"]))]
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists() else None
    data = existing or {"hardware": args.hardware, "base_url": args.base_url, "implementation": "exact_state_transition_acceleration",
                        "workload": {"requests": 1, "prompt_tokens": 512, "output_tokens": 128,
                                     "llama_cpp": LLAMA_DEFAULTS, "retention_policy": "aggregate"},
                        "results": []}
    completed = {row["model"] for row in data["results"] if row["status"] in {"completed", "capacity_rejected"}}
    for model in models:
        if model in completed:
            continue
        print("START " + json.dumps({"hardware": args.hardware, "model": model}, ensure_ascii=False), flush=True)
        result = run_case(args.base_url.rstrip("/"), args.hardware, model)
        data["results"] = [row for row in data["results"] if row["model"] != model] + [result]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("RESULT " + json.dumps({key: result.get(key) for key in
              ("hardware", "model", "status", "elapsed_s", "job_id", "error", "validation_failures")}, ensure_ascii=False), flush=True)
    print("DONE " + json.dumps({"hardware": args.hardware, "cases": len(data["results"]),
          "totals": {state: sum(row["status"] == state for row in data["results"])
                     for state in sorted({row["status"] for row in data["results"]})}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
