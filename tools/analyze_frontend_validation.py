"""Analyze saved browser submissions and job reports without running simulation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from heterollm_sim.architecture_presets import list_architecture_presets
from heterollm_sim.model_presets import list_model_presets
from heterollm_sim.physical_contract import PHYSICAL_MEMORY_COMPONENT_KINDS, parse_required_physical_config
from heterollm_sim.runtime_adapters import normalize_llama_runtime_identity


ROOT = Path(__file__).resolve().parents[1]
PAIRED_CASES = ("qwen3_0_6b_f16", "qwen3_8_27b_mixed")
LATENCIES = ("ttft", "tpot", "e2e")
MATRIX_RUNTIME = {"policy": "llama_cpp", "gpu_layers": -1, "batch": 512, "ubatch": 512,
                  "context": 640, "parallel": 1, "offload_kqv": True, "device_memory_tiering": True}
_MISSING = object()
_DERIVED_METADATA = frozenset({
    "llama_cpp_runtime", "llama_cpp_runtime_fingerprint", "llama_cpp_tensor_storage",
    "llama_cpp_slot_order", "llama_cpp_final_norm_static", "llama_cpp_mixed_phase_batching",
    "llama_cpp_effective_scheduler_policy", "llama_cpp_capabilities", "context_limit_semantics",
    "llama_cpp_kv_capacity_contract", "llama_cpp_gpu_layer_mapping", "llama_cpp_kv_layer_components",
    "llama_cpp_kv_layer_ranks", "llama_cpp_kv_contract", "logical_weight_aliases",
})
_DISPLAY_METADATA = frozenset({
    "ui", "workload_preset_id", "workload_preset_source", "native_model_path", "native_model_file",
    "native_runtime_context", "native_output_contract_evidence",
})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepared_scenario_file(base: Path, case):
    """Resolve the explicitly bundled file; original machine paths are audit only."""
    name = case.get("scenario_file")
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(char in name for char in ("/", "\\", ":"))):
        raise ValueError("prepared case requires a scenario_file basename in the evidence directory")
    return base / name


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value):
    return finite(value) and value > 0


def hardware(scenario):
    if "hardware_input" in scenario:
        return scenario["hardware_input"]["hardware"]
    return scenario.get("hardware", {})


def simulation_gpu_graphs_explicitly_disabled(scenario):
    """Read the same explicit hardware authority as the scenario parser.

    V4 hardware inputs carry inline execution profiles; their components do
    not have runtime cost_profile_id fields. The parser uses these profiles
    to replace the redundant runtime hardware/profile sections.
    """
    gpu_profiles = scenario.get("profiles", {}).get("components", {}).get("gpu", {})
    gpu_components = [component for component in hardware(scenario).get("components", [])
                      if component.get("kind") in {"gpu", "cuda"}]
    if not gpu_components:
        return False
    for component in gpu_components:
        if "hardware_input" in scenario:
            binding = component.get("execution_profile")
            if (not isinstance(binding, dict) or binding.get("profile_kind") != "gpu"
                    or not isinstance(binding.get("profile_id"), str)
                    or not binding["profile_id"].strip()):
                return False
            profile = binding.get("parameters")
        else:
            profile = gpu_profiles.get(component.get("cost_profile_id"))
        kernel = profile.get("kernel_model") if isinstance(profile, dict) else None
        if not isinstance(kernel, dict) or kernel.get("graph_enabled") is not False:
            return False
    return True


def model_identity(scenario):
    model = scenario.get("model", {})
    metadata = model.get("metadata", {})
    nested = metadata.get("metadata", {})
    attributes = model.get("graph", {}).get("attributes", {})
    graph_nested = attributes.get("metadata", {})
    identities = {owner["gguf_sha256"] for owner in (metadata, nested, attributes, graph_nested)
                  if owner.get("gguf_sha256")}
    return {
        "preset_id": metadata.get("preset_id"), "name": model.get("name"),
        "gguf_identities": sorted(identities),
    }


def attempt_scope(prefix):
    if prefix.startswith(("ui_before_fix_", "ui_diagnostic_")):
        return "diagnostic"
    return "native_pair" if prefix.startswith("ui_pair_") else "preset_matrix"


def submission_contract_errors(scenario, scope):
    """Inspect the captured input without preparing placement or executing work."""
    errors = []
    workload = scenario.get("workload", {})
    submitted = workload.get("requests", [])
    if not isinstance(submitted, list) or len(submitted) != 1:
        errors.append("submission must explicitly contain one request")
    else:
        request = submitted[0]
        if (request.get("prompt_tokens"), request.get("output_tokens"), request.get("arrival_ns")) != (512, 128, 0):
            errors.append("submission is not the default 512 input / 128 output request at time zero")
    if workload.get("mtp") is not None:
        errors.append("submission must disable speculative decoding")
    hw = hardware(scenario)
    hardware_id = hw.get("metadata", {}).get("architecture_preset", {}).get("id")
    if not hardware_id:
        errors.append("submission has no hardware preset identity")
    runtime = scenario.get("profiles", {}).get("llama_cpp", {})
    if scope == "preset_matrix":
        if model_identity(scenario)["preset_id"] is None:
            errors.append("matrix submission has no model preset identity")
        for key, value in MATRIX_RUNTIME.items():
            if runtime.get(key) != value:
                errors.append(f"matrix submission llama_cpp.{key} differs from default {value!r}")
    elif runtime.get("policy") != "llama_cpp":
        errors.append("submission does not select llama.cpp scheduling")
    if hardware_id and hardware_id.startswith("local-native-"):
        if workload.get("metadata", {}).get("llama_cpp_kernel_model_preset") != "blackwell_analytical_v1":
            errors.append("local hardware submission lost the selected Blackwell kernel model")
    for component in hw.get("components", []):
        kind = str(component.get("kind", "")).lower()
        if kind not in PHYSICAL_MEMORY_COMPONENT_KINDS:
            continue
        try:
            config = parse_required_physical_config(component.get("component_id"), kind,
                component.get("metadata", {}).get("physical_memory_config"))
            if component.get("capacity_bytes") != config.capacity_bytes:
                errors.append(f"{component.get('component_id')} capacity differs from physical geometry")
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    return errors


def capacity_rejection(errors):
    text = json.dumps(errors, ensure_ascii=False).lower()
    return ("device_memory_capacity_exhausted" in text
            or "capacity exhausted" in text or "insufficient capacity" in text)


def comparable_model(model):
    result = deepcopy(model)
    result.get("metadata", {}).pop("ui", None)
    graph = result.get("graph", {})
    graph.get("attributes", {}).pop("ui", None)
    # These two optional empty registries and tensor-directory ordering have
    # no executable semantics. Operator order and every tensor value remain.
    graph.setdefault("source_operators", [])
    graph.setdefault("sub_operators", [])
    if "tensors" in graph:
        graph["tensors"] = sorted(graph["tensors"], key=lambda tensor: tensor["tensor_id"])
    return result


def comparable_hardware(value):
    result = deepcopy(value)
    result.get("metadata", {}).pop("topology_view", None)
    return result


def same_contract(actual, expected):
    """Keep JSON booleans distinct from 0/1 while allowing 1 and 1.0."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_contract(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(same_contract(a, b) for a, b in zip(actual, expected))
    return actual == expected


def comparable_execution_metadata(value):
    """Remove known UI/provenance and regenerated outputs, not source contracts.

    Unknown fields stay in the comparison, so adding a new execution input
    cannot silently bypass the pairing gate. Runtime identity is normalized
    because an unbound record becomes an empty partial record during UI import.
    """
    result = deepcopy(value)
    for key in _DISPLAY_METADATA | _DERIVED_METADATA:
        result.pop(key, None)
    identity = result.pop("llama_cpp_runtime_identity", None)
    result["llama_cpp_runtime_identity"] = normalize_llama_runtime_identity({} if identity is None else identity)
    control = result.get("control_plane")
    if isinstance(control, dict):
        control.pop("decision", None)
        control.pop("evidence", None)
    return result


def paired_input_contract_errors(scenario, prepared):
    """Compare actual submitted cost inputs before accepting native metrics."""
    errors = []

    def compare_fields(actual, expected, prefix):
        for key in sorted(actual.keys() | expected.keys()):
            if not same_contract(actual.get(key, _MISSING), expected.get(key, _MISSING)):
                errors.append(f"submitted {prefix}.{key} differs from prepared native contract")

    compare_fields(scenario.get("profiles", {}), prepared.get("profiles", {}), "profiles")
    for section in ("workload", "placement"):
        actual = deepcopy(scenario.get(section, {}))
        expected = deepcopy(prepared.get(section, {}))
        for item in (actual, expected):
            # Names are UI labels; all execution fields including scheduler,
            # KV policy, rank maps and explicit requests remain authoritative.
            for key in ("name", "model_name", "hardware_name", "schema_version"):
                item.pop(key, None)
        a_meta = comparable_execution_metadata(actual.pop("metadata", {}))
        b_meta = comparable_execution_metadata(expected.pop("metadata", {}))
        compare_fields(a_meta, b_meta, section + ".metadata")
        if section == "workload":
            for item in (actual, expected):
                for request in item.get("requests", []):
                    request["metadata"] = comparable_execution_metadata(request.get("metadata", {}))
        compare_fields(actual, expected, section)
    if not same_contract(scenario.get("weights_resident", _MISSING), prepared.get("weights_resident", _MISSING)):
        errors.append("submitted weights_resident differs from prepared native contract")
    return errors


def requests(report):
    value = report.get("requests", {})
    return list(value.values()) if isinstance(value, dict) else list(value)


def physical_participation(scenario, report):
    summary = report.get("summary", {})
    errors, traffic = [], {}
    for name in ("dram_traffic", "storage_traffic"):
        data = summary.get(name)
        if not isinstance(data, dict):
            errors.append(f"missing summary.{name}")
            continue
        for key in ("task_count", "physical_bytes", "physical_read_bytes", "physical_write_bytes",
                    "logical_bytes", "logical_read_bytes", "logical_write_bytes", "service_ns"):
            if key != "task_count" and data.get("task_count") == 0 and key not in data:
                continue
            if (name == "storage_traffic" and key in {"logical_read_bytes", "logical_write_bytes"}
                    and data.get(key) is None):
                continue  # Older NAND reports omit these independently measured directions.
            if not finite(data.get(key)) or data[key] < 0:
                errors.append(f"{name}.{key} is missing, negative, or nonfinite")
        for total, read, write in (("physical_bytes", "physical_read_bytes", "physical_write_bytes"),
                                   ("logical_bytes", "logical_read_bytes", "logical_write_bytes")):
            if all(finite(data.get(key)) for key in (total, read, write)):
                if data[total] != data[read] + data[write]:
                    errors.append(f"{name}.{total} differs from read plus write")
        if positive(data.get("task_count")):
            if not positive(data.get("physical_bytes")) or not positive(data.get("service_ns")):
                errors.append(f"{name} physical tasks have no bytes or service time")
        elif data.get("task_count") == 0 and positive(data.get("physical_bytes")):
            errors.append(f"{name} reports physical bytes with zero physical tasks")
        if name == "dram_traffic":
            fields = ("burst_count", "row_hits", "row_misses", "row_conflicts")
            if all(finite(data.get(key)) for key in fields):
                if data["burst_count"] != sum(data[key] for key in fields[1:]):
                    errors.append("DRAM row classes do not sum to burst_count")
        selected = {key: value for key, value in data.items()
                    if key not in {"resource_totals", "organization_profiles"}}
        if name == "storage_traffic" and positive(data.get("task_count")):
            availability = dict(selected.get("metric_availability", {}))
            for key in ("logical_read_bytes", "logical_write_bytes"):
                if data.get(key) is None:
                    selected[key] = None
                    availability[key] = {"status": "not_recorded", "source": "saved_report"}
            if availability:
                selected["metric_availability"] = availability
        owners = defaultdict(lambda: {"resource_count": 0, "service_ns": 0.0,
                                      "bytes_moved": 0, "energy_pj": 0.0})
        for resource in data.get("resource_totals", {}).values():
            owner = resource.get("owner")
            if owner is None:
                continue
            row = owners[owner]
            row["resource_count"] += 1
            for key in ("service_ns", "bytes_moved", "energy_pj"):
                if finite(resource.get(key)):
                    row[key] += resource[key]
        selected["owner_resources"] = dict(owners)
        traffic[name] = selected
    dram = summary.get("dram_traffic", {})
    if not all(positive(dram.get(key)) for key in ("task_count", "physical_read_bytes", "physical_write_bytes")):
        errors.append("completed LLM run lacks positive physical DRAM task/read/write evidence")
    components = hardware(scenario).get("components", [])
    used_owners = set()
    for name in ("dram_traffic", "storage_traffic"):
        used_owners.update(summary.get(name, {}).get("owner_ids", []))
    memories = []
    for component in components:
        metadata = component.get("metadata", {})
        physical = metadata.get("physical_memory_config")
        if physical is None:
            continue
        binding = component.get("execution_profile")
        profile = binding.get("parameters") if isinstance(binding, dict) else None
        owner = profile.get("resource_id") if isinstance(profile, dict) else None
        if not isinstance(owner, str) or not owner:
            errors.append(f"{component['component_id']} has no explicit inline physical profile resource_id")
            owner = None
        else:
            declared_owner = metadata.get("memory_service_owner")
            if declared_owner is not None and declared_owner != owner:
                errors.append(f"{component['component_id']} physical owner conflicts with inline profile resource_id")
        memories.append({"component_id": component["component_id"], "kind": physical.get("kind"),
                         "owner": owner, "physical_activity_observed": owner in used_owners})
    resource_use = report.get("resource_utilization", {})
    groups = {}
    for group, kinds in (("cpu", {"cpu"}), ("gpu", {"gpu"})):
        prefixes = tuple(c["component_id"] + "." for c in components if c.get("kind") in kinds)
        active = {key: value for key, value in resource_use.items()
                  if key.startswith(prefixes) and finite(value) and value > 0}
        groups[group] = {"active_resource_count": len(active), "active_resources": active}
        if prefixes and not active:
            errors.append(f"no positive {group} resource activity reported")
    category = report.get("category_time_ns", {})
    runtime = report.get("analytical_coverage", {}).get("runtime", {})
    return {
        "status": "pass" if not errors else "failed", "errors": errors,
        "physical_traffic": traffic, "configured_memories": memories,
        "compute_resource_activity": groups, "category_time_ns": category,
        "critical_path_category_ns": report.get("critical_path_category_ns", {}),
        "model_operator_activity": runtime,
        "total_energy_pj": summary.get("total_energy_pj"),
        "semantics": [
            "Participation is based on actual report counters and resource use, not configured capabilities.",
            "A configured but unused memory is not automatically an error.",
            "Service and category durations may overlap; they are not additive wall latency.",
            "Unrecorded read/write switches and refresh/turnaround subtotals are null; legacy zeros without availability do not prove measurement.",
            "Empty organization profiles do not establish complete physical geometry reporting.",
            "Historical NAND logical directions may be unrecorded; physical directions and totals remain required.",
            "These checks do not establish hardware prediction accuracy or completeness of every cost model.",
        ],
    }


def completed_checks(scenario, report):
    errors = []
    summary = report.get("summary", {})
    submitted_requests = scenario.get("workload", {}).get("requests", [])
    actual = requests(report)
    if len(submitted_requests) != 1 or len(actual) != 1:
        errors.append("expected exactly one submitted and reported request")
        return errors
    expected = submitted_requests[0]
    row = actual[0]
    if expected.get("prompt_tokens") != 512 or expected.get("output_tokens") != 128:
        errors.append("submitted workload is not the default 512 input / 128 output")
    if row.get("status") != "finished" or row.get("visible_output_tokens") != expected.get("output_tokens"):
        errors.append("request did not finish with the requested visible output count")
    if summary.get("completed_requests") != 1 or summary.get("rejected_requests") != 0:
        errors.append("completed/rejected request counts do not describe one successful request")
    for metric in LATENCIES:
        if not finite(row.get(f"engine_{metric}_ns")) or row[f"engine_{metric}_ns"] <= 0:
            errors.append(f"missing positive engine_{metric}_ns")
    if report.get("measurement_semantics", {}).get("latency", {}).get("primary_boundary") != "engine":
        errors.append("report primary latency boundary is not explicitly engine")
    if summary.get("batch_count") != 128:
        errors.append("non-speculative default workload did not execute 128 cohorts")
    live = summary.get("physical_live_batch_count")
    if type(live) is not int or live != summary.get("batch_count") or live <= 0:
        errors.append("not every reported cohort executed through the persistent live physical kernel")
    if not finite(summary.get("makespan_ns")) or summary["makespan_ns"] <= 0:
        errors.append("makespan is not positive and finite")
    mtp = summary.get("mtp", {})
    if mtp.get("enabled") is not False or mtp.get("proposed_tokens", 0) != 0:
        errors.append("unexpected speculative decoding")
    return errors


def analyze_attempt(submission_path: Path):
    prefix = submission_path.name.removesuffix("_submission.json")
    submission = read_json(submission_path)
    scenario = submission.get("scenario", {})
    hw = hardware(scenario)
    identity = model_identity(scenario)
    result_path = submission_path.with_name(prefix + "_result.json")
    created_path = submission_path.with_name(prefix + "_job_created.json")
    row = {
        "prefix": prefix, "attempt_id": prefix, "submission_file": submission_path.name,
        "result_file": result_path.name if result_path.exists() else None,
        "scope": attempt_scope(prefix), "record_kind": "run_submission",
        "hardware_id": hw.get("metadata", {}).get("architecture_preset", {}).get("id"),
        "model_id": identity["preset_id"], "model_identity": identity,
        "scenario_name": scenario.get("name"),
        "submission_retention_policy": submission.get("retention_policy"),
        "status": "pending", "validation_errors": submission_contract_errors(scenario, attempt_scope(prefix)),
        "sort_time": submission_path.stat().st_mtime,
    }
    if not result_path.exists():
        if created_path.exists():
            created = read_json(created_path)
            row.update(job_id=created.get("job_id"), job_status=created.get("status"),
                       created_at=created.get("created_at"), started_at=created.get("started_at"))
            if created.get("status") in {"queued", "running", "cancelling"}:
                row["status"] = "running"
        return row, scenario, None
    job = read_json(result_path)
    recording = job.get("recording")
    if recording is not None:
        row["result_recording"] = recording
        if (not isinstance(recording, dict)
                or recording.get("schema") != "frontend-job-compact/v1"
                or not isinstance(recording.get("omitted_fields"), list)
                or any(field not in ("report.visualization", "report.batch_history")
                       for field in recording.get("omitted_fields", ()))
                or not isinstance(recording.get("raw_archive_path"), str)):
            row["validation_errors"].append("unsupported compact result omissions")
        else:
            row["raw_archive_available_locally"] = Path(recording.get("raw_archive_path", "")).is_file()
    row.update(job_id=job.get("job_id"), job_status=job.get("status"),
               created_at=job.get("created_at"), started_at=job.get("started_at"),
               finished_at=job.get("finished_at"), error=job.get("error"),
               sort_time=result_path.stat().st_mtime)
    if created_path.exists():
        created = read_json(created_path)
        if created.get("job_id") != job.get("job_id"):
            row["validation_errors"].append("result job_id differs from browser submission response")
    if job.get("status") in {"queued", "running", "cancelling"}:
        row["status"] = "running"
    elif job.get("status") == "completed":
        report = job.get("report")
        if not isinstance(report, dict):
            row["validation_errors"].append("completed job has no report")
        else:
            row["validation_errors"].extend(completed_checks(scenario, report))
            participation = physical_participation(scenario, report)
            row["participation"] = participation
            row["validation_errors"].extend(participation["errors"])
            if report.get("scenario") != scenario.get("name"):
                row["validation_errors"].append("report scenario name differs from submitted scenario")
            row["requests"] = [{key: value for key, value in request.items() if key not in {"tbt_ns"}}
                               for request in requests(report)]
            row["validation_warnings"] = report.get("validation_warnings", [])
            row["physical_execution"] = {
                "batch_count": report.get("summary", {}).get("batch_count"),
                "physical_live_batch_count": report.get("summary", {}).get("physical_live_batch_count"),
            }
            row["analytical_limits"] = {
                "host_output_contract_status": report.get("summary", {}).get("host_output_contract_status"),
                "host_output_contract_partial_reason": report.get("summary", {}).get("host_output_contract_partial_reason"),
                "tensor_storage": scenario.get("workload", {}).get("metadata", {}).get("llama_cpp_tensor_storage"),
            }
        row["status"] = "validation_failed" if row["validation_errors"] else "completed"
    elif job.get("status") == "cancelled":
        row["status"] = "cancelled"
    else:
        if capacity_rejection(job.get("error", job)):
            row["status"] = "capacity_rejected"
        else:
            row["status"] = "program_error" if job.get("status") == "failed" else "submission_rejected"
    if row["validation_errors"] and row["status"] == "capacity_rejected":
        row["status"] = "input_contract_failed"
    return row, scenario, job


def analyze_validation(validation_path):
    prefix = validation_path.name.removesuffix("_validation.json")
    record = read_json(validation_path)
    if not isinstance(record.get("submission"), dict):
        return ({
            "prefix": prefix, "attempt_id": prefix + "@validation", "record_kind": "pre_run_validation",
            "scope": attempt_scope(prefix), "validation_file": validation_path.name, "job_created": False,
            "hardware_id": None, "model_id": None, "status": "validation_capture_incomplete",
            "validation_errors": ["validation capture lacks the submitted scenario; original input cannot be established"],
            "response": record.get("response", record), "sort_time": validation_path.stat().st_mtime,
        }, {}, None)
    submission = record.get("submission", {})
    scenario = submission.get("scenario", submission)
    response = record.get("response", {})
    scope = attempt_scope(prefix)
    identity = model_identity(scenario)
    errors = submission_contract_errors(scenario, scope)
    row = {
        "prefix": prefix, "attempt_id": prefix + "@validation", "record_kind": "pre_run_validation", "scope": scope,
        "validation_file": validation_path.name, "job_created": False,
        "hardware_id": hardware(scenario).get("metadata", {}).get("architecture_preset", {}).get("id"),
        "model_id": identity["preset_id"], "model_identity": identity,
        "scenario_name": scenario.get("name"), "validation_errors": errors,
        "response": response, "sort_time": validation_path.stat().st_mtime,
    }
    if errors:
        row["status"] = "input_contract_failed"
    elif response.get("valid") is False:
        row["status"] = "capacity_rejected" if capacity_rejection(response.get("errors")) else "validation_rejected"
    elif response.get("valid") is True:
        row["status"] = "validated_not_run"
    else:
        row["status"] = "validation_response_invalid"
    return row, scenario, None


def sample_statistics(values):
    mean = statistics.fmean(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "mean": mean, "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "sample_count": len(values)}


def compare_values(simulated, native_values, unit):
    stats = sample_statistics(native_values)
    delta = simulated - stats["median"]
    return {"unit": unit, "simulation": simulated, "native": stats,
            "signed_error": delta, "absolute_error": abs(delta),
            "signed_error_percent": delta / stats["median"] * 100,
            "absolute_error_percent": abs(delta) / stats["median"] * 100,
            "inside_native_observed_range": stats["min"] <= simulated <= stats["max"]}


def pair_configuration_errors(scenario, prepared, native, expected_case):
    errors = paired_input_contract_errors(scenario, prepared)
    if model_identity(scenario)["gguf_identities"] != [expected_case["model"]["sha256"]]:
        errors.append("submitted GGUF identity does not match prepared native model")
    if comparable_model(scenario.get("model", {})) != comparable_model(prepared.get("model", {})):
        errors.append("submitted model graph/metadata differs from the prepared GGUF model")
    if comparable_hardware(hardware(scenario)) != comparable_hardware(hardware(prepared)):
        errors.append("submitted hardware contract differs from the prepared physical preset")
    a = scenario.get("profiles", {}).get("llama_cpp", {})
    config = native.get("configuration", {})
    mapping = {"context": "context", "batch": "batch", "ubatch": "ubatch", "threads": "threads",
               "threads_batch": "threads_batch", "gpu_layers": "gpu_layers_requested", "parallel": "parallel",
               "kv_type_k": "cache_type_k", "kv_type_v": "cache_type_v"}
    for simulation_key, native_key in mapping.items():
        if a.get(simulation_key) != config.get(native_key):
            errors.append(f"simulation {simulation_key} differs from native {native_key}")
    if a.get("flash_attn") != (config.get("flash_attn") == "on"):
        errors.append("Flash Attention differs between simulation and native")
    if not simulation_gpu_graphs_explicitly_disabled(scenario):
        errors.append("native pairing requires explicit graph_enabled=False on every simulation GPU kernel profile")
    effective_env = config.get("effective_env", {})
    if (not isinstance(effective_env, dict) or effective_env.get("GGML_CUDA_DISABLE_GRAPHS") != "1"
            or config.get("cuda_graphs_disabled") is not True):
        errors.append("native CUDA Graph disabling is not established by the recorded child-process environment")
    observed = native.get("actual_server_context", {})
    if observed.get("props_n_ctx") != a.get("context") or observed.get("startup_n_ctx_slot") != a.get("context"):
        errors.append("actual native context differs from simulation")
    metadata = scenario.get("workload", {}).get("metadata", {})
    if metadata.get("llama_cpp_f32_hidden_storage") is not True:
        errors.append("native F32 hidden tensor storage contract is not enabled")
    if config.get("cache_prompt") is not False or config.get("speculative_decoding") is not False:
        errors.append("native prompt reuse/speculative configuration is not explicitly disabled")
    if config.get("model_path") != expected_case["model"]["path"]:
        errors.append("native model path differs from prepared GGUF")
    samples = native.get("samples", [])
    if len(samples) != 5 or len(native.get("warmups", [])) != 2:
        errors.append("native did not collect exactly two warmups and five formal samples")
    for index, sample in enumerate(samples):
        if (sample.get("status") != "completed" or sample.get("prompt_n") != 512
                or sample.get("cache_n") != 0 or sample.get("visible_output_tokens") != 128):
            errors.append(f"native sample {index + 1} violates the token/cache contract")
        for metric in LATENCIES:
            if not finite(sample.get(f"engine_{metric}_ns")) or sample[f"engine_{metric}_ns"] <= 0:
                errors.append(f"native sample {index + 1} has invalid engine_{metric}_ns")
    return errors


def analyze_pair(base, slug, attempts, preparation):
    prefix = "ui_pair_" + slug
    names = (prefix + "_submission.json", prefix + "_result.json", "native_" + slug + ".json")
    missing = [name for name in names if not (base / name).exists()]
    row = {"case_id": slug, "status": "pending", "missing_files": missing}
    if missing:
        return row
    attempt, scenario, job = attempts[prefix]
    row["simulation_status"] = attempt["status"]
    if attempt["status"] != "completed":
        row["status"] = attempt["status"]
        row["validation_errors"] = attempt["validation_errors"]
        return row
    native = read_json(base / ("native_" + slug + ".json"))
    expected_case = next((case for case in preparation["cases"] if case["case_id"] == slug), None)
    if expected_case is None:
        row["missing_files"].append("preparation.json case " + slug)
        return row
    try:
        prepared_file = prepared_scenario_file(base, expected_case)
    except ValueError as error:
        row["status"] = "configuration_mismatch"
        row["validation_errors"] = [str(error)]
        return row
    if not prepared_file.is_file():
        row["missing_files"].append(prepared_file.name)
        return row
    prepared = read_json(prepared_file)
    errors = pair_configuration_errors(scenario, prepared, native, expected_case)
    if native.get("status") != "completed":
        errors.append("native benchmark did not complete")
    row["validation_errors"] = errors
    if errors:
        row["status"] = "configuration_mismatch"
        return row
    simulated = requests(job["report"])[0]
    samples = native["samples"]
    row["status"] = "compared"
    row["metrics"] = {metric: compare_values(simulated[f"engine_{metric}_ns"],
        [sample[f"engine_{metric}_ns"] for sample in samples], "ns") for metric in LATENCIES}
    row["metrics"]["decode_tokens_per_second"] = compare_values(
        1e9 / simulated["engine_tpot_ns"], [1e9 / sample["engine_tpot_ns"] for sample in samples], "token/s")
    row["metrics"]["output_tokens_per_engine_second"] = compare_values(
        128e9 / simulated["engine_e2e_ns"], [128e9 / sample["engine_e2e_ns"] for sample in samples], "token/s")
    row["measurement_boundary"] = "engine_request_begin to first/last visible token; TPOT=(last-first)/127"
    row["cuda_graphs_contract"] = "simulation GPU graph_enabled=False; native child GGML_CUDA_DISABLE_GRAPHS=1"
    row["simulation_file"] = names[1]
    row["native_file"] = names[2]
    row["analytical_limits"] = attempt.get("analytical_limits")
    row["physical_execution"] = attempt.get("physical_execution")
    row["energy_accounting"] = attempt.get("energy_accounting", "not_revalidated")
    row["native_client_metrics"] = native.get("summary", {}).get("client")
    row["interpretation"] = "Raw out-of-sample discrepancy; no native timing was used to tune these scenario parameters."
    return row


def attach_postprocessed_energy(attempt, evidence):
    """Keep postprocessing separate from the actual job/version evidence."""
    if (attempt.get("status") != "completed"
            or evidence.get("schema") != "frontend-physical-energy-repricing/v1"
            or evidence.get("validation", {}).get("v2_execution_count", 0) <= 0
            or evidence.get("source_generation_audit", {}).get("status")
            != "reviewed_current_producers_with_restricted_field_scope"):
        return
    matches = [row for row in evidence.get("entries", []) if row.get("prefix") == attempt["prefix"]
               and row.get("job_id") == attempt.get("job_id") and row.get("status") == "repriced"]
    if len(matches) != 1:
        return
    row = matches[0]
    if (row.get("new_simulation_executed") is not False
            or row.get("method") != "same_execution_physical_record_repricing"
            or row.get("original_total_energy_pj") != attempt.get("participation", {}).get("total_energy_pj")
            or not finite(row.get("repriced_total_energy_pj")) or row["repriced_total_energy_pj"] < 0):
        return
    attempt["postprocessed_energy"] = {**row, "evidence_file": "energy_repricing.json"}


def analyze(base):
    attempts = {}
    repricing_path = base / "energy_repricing.json"
    repricing = read_json(repricing_path) if repricing_path.exists() else {}
    browser_index_path = base / "ui_jobs.json"
    browser_index = {row["prefix"]: row for row in read_json(browser_index_path)} if browser_index_path.exists() else {}
    for path in sorted(base.glob("ui_*_submission.json")):
        result = analyze_attempt(path)
        result[0]["energy_accounting"] = "not_revalidated"
        index_entry = browser_index.get(result[0]["prefix"])
        if index_entry is not None:
            result[0]["browser_submission_index"] = index_entry
            if index_entry.get("job_id") != result[0].get("job_id"):
                result[0]["validation_errors"].append("job_id differs from browser ui_jobs.json index")
                if result[0]["status"] == "completed":
                    result[0]["status"] = "validation_failed"
            elif index_entry.get("energy_accounting") == "physical_profile_energy_v2":
                result[0]["energy_accounting"] = "physical_profile_energy_v2"
        attach_postprocessed_energy(result[0], repricing)
        attempts[result[0]["prefix"]] = result
    for path in sorted(base.glob("ui_*_validation.json")):
        result = analyze_validation(path)
        attempts[result[0]["prefix"] + "@validation"] = result
    rows = [value[0] for value in attempts.values()]
    hardware_ids = [item["id"] for item in list_architecture_presets() if item["loadable"]]
    model_ids = [item["id"] for item in list_model_presets()]
    latest = {}
    for row in rows:
        if row["scope"] != "preset_matrix" or row["status"] == "validation_capture_incomplete":
            continue
        key = row["hardware_id"], row["model_id"]
        if key not in latest or latest[key]["sort_time"] < row["sort_time"]:
            latest[key] = row
    matrix = []
    for hw in hardware_ids:
        for model in model_ids:
            row = latest.get((hw, model))
            matrix.append({"hardware_id": hw, "model_id": model,
                           "status": row["status"] if row else "pending",
                           "latest_attempt": row["attempt_id"] if row else None})
    preparation_path = base / "preparation.json"
    preparation = read_json(preparation_path) if preparation_path.exists() else {"cases": []}
    pairs = [analyze_pair(base, slug, attempts, preparation) for slug in PAIRED_CASES]
    result = {
        "schema": "heterollm.frontend-native-validation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_saved_browser_submissions_and_reports_only",
        "expected_scope": {"hardware_presets": len(hardware_ids), "model_presets": len(model_ids),
                           "matrix_combinations": len(matrix), "native_pairs": len(pairs)},
        "matrix_status_counts": dict(Counter(row["status"] for row in matrix)),
        "matrix_status_by_hardware": {
            hw: dict(Counter(row["status"] for row in matrix if row["hardware_id"] == hw))
            for hw in hardware_ids},
        "matrix": matrix, "attempts": [row for row in rows if row["scope"] != "diagnostic"],
        "diagnostic_attempts": [row for row in rows if row["scope"] == "diagnostic"],
        "native_comparisons": pairs,
        "limitations": [
            "Only saved browser submissions and final job reports are analyzed; no run is initiated.",
            "Missing files are pending, not successful tests.",
            "Physical participation checks establish reported activity, not real-hardware accuracy.",
            "Native observed min/max is measurement variation, not a statistical confidence interval.",
            "B200/HBF has no paired physical-machine measurement in this local run.",
        ],
    }
    for row in rows:
        row.pop("sort_time", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "docs/frontend_native_validation_2026-10-07")
    args = parser.parse_args()
    result = analyze(args.directory)
    target = args.directory / "comparison.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Matrix:", json.dumps(result["matrix_status_counts"], ensure_ascii=False))
    for pair in result["native_comparisons"]:
        print(pair["case_id"], pair["status"])
        for name, metric in pair.get("metrics", {}).items():
            print(f"  {name}: {metric['simulation']:.6g} vs native {metric['native']['median']:.6g} "
                  f"{metric['unit']}; error {metric['signed_error_percent']:+.3f}%")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
