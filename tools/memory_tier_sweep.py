"""Small, one-factor memory-tier sweep and pure mechanism checks.

PASS here means internal mechanism consistency, never hardware accuracy.
Missing coverage is NOT_COVERED, not a passing zero or invented measurement.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.qwen38_memory_scenario import (DEFAULT_MODEL, LIMITATIONS, OUTPUT_ROOT,
    SCENARIOS, UNVALIDATED, UnsupportedScenario, baseline_manifest, build_scenario,
    check_output_path, execute_scenario, load_model, scenario_payload, source_identity, write_output)
from heterollm_sim.serde import stable_hash

# Requested CIM and explicit DRAM-only comparisons remain separate; never replace blocked rows.
PRIMARY_SCENARIOS = tuple(s for s in SCENARIOS if "_converted" not in s and "_tiled" not in s)


def check(name, status, detail):
    return {"check": name, "status": status, "detail": detail}


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def close(left, right):
    return (finite_nonnegative(left) and finite_nonnegative(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-5))


def same_experiment(left, right):
    return all(left.get(key) and left.get(key) == right.get(key)
               for key in ("model_sha256", "workload_sha256", "placement_sha256"))


def component_activity(observation, component_id):
    """Read execution-ledger counters, including typed controller owners."""
    if (not observation.get("complete_resource_accounting")
            or component_id not in observation.get("component_ids", [])):
        return None
    owner = observation.get("profile_owners", {}).get(component_id)
    resources = set(observation.get("resource_bytes", {})) | set(observation.get("resource_service_ns", {}))
    selected = {r for r in resources if r == owner or r.startswith(component_id + ".")
                or r.startswith("component." + component_id + ".")}
    # Exact endpoint resources are preferred over link-name substring matching.
    values = [observation.get("resource_bytes", {}).get(r, 0) for r in selected]
    services = [observation.get("resource_service_ns", {}).get(r, 0) for r in selected]
    if not all(finite_nonnegative(v) for v in values + services):
        return None
    return {"bytes": sum(values), "service_ns": sum(services), "resources": sorted(selected)}


def zero_traffic_invariance(before, after, component_id):
    name = "zero_traffic_invariance:" + component_id
    if not same_experiment(before, after):
        return check(name, "NOT_COVERED", "model/workload/placement changed or identity missing")
    activity = [component_activity(o, component_id) for o in (before, after)]
    if any(a is None for a in activity):
        return check(name, "NOT_COVERED", "missing complete resource accounting")
    if any(a["bytes"] != 0 or a["service_ns"] != 0 for a in activity):
        return check(name, "NOT_COVERED", "component has traffic/service; this is not a zero-traffic experiment")
    metrics = ("ttft_ns", "tpot_ns", "e2e_ns")
    if any(not finite_nonnegative(o.get("metrics", {}).get(m)) for o in (before, after) for m in metrics):
        return check(name, "FAIL", "missing or non-finite timing")
    passed = all(close(before["metrics"][m], after["metrics"][m]) for m in metrics)
    return check(name, "PASS" if passed else "FAIL", "zero-byte/zero-service endpoint; fixed-placement timings must remain unchanged")


def monotonicity(points, *, parameter="read_latency_ns", direction="nondecreasing"):
    """points=[{'value': scalar, 'observation': actual_engine_observation}, ...]."""
    name = "monotonicity:" + parameter
    if direction not in {"nondecreasing", "nonincreasing"}:
        raise ValueError("unknown monotonicity direction")
    if len(points) < 2:
        return check(name, "NOT_COVERED", "at least two points required")
    if not all(finite_nonnegative(p.get("value")) for p in points):
        return check(name, "FAIL", "invalid parameter values")
    ordered = sorted(points, key=lambda p: p["value"])
    if len({p["value"] for p in ordered}) != len(ordered):
        return check(name, "NOT_COVERED", "parameter values must be distinct")
    if any(not p["observation"].get("complete_resource_accounting") for p in ordered):
        return check(name, "NOT_COVERED", "incomplete resource ledger cannot support trend acceptance")
    if any(not same_experiment(ordered[0]["observation"], p["observation"]) for p in ordered[1:]):
        return check(name, "NOT_COVERED", "placement changed: optimizer effects are not a fixed-placement monotonicity test")
    violations, changed = [], False
    for left, right in zip(ordered, ordered[1:]):
        for metric in ("ttft_ns", "tpot_ns", "e2e_ns"):
            a = left["observation"].get("metrics", {}).get(metric)
            b = right["observation"].get("metrics", {}).get(metric)
            if not finite_nonnegative(a) or not finite_nonnegative(b):
                return check(name, "FAIL", "missing or non-finite timing")
            if close(a, b):
                continue
            changed = True
            if (b < a if direction == "nondecreasing" else b > a):
                violations.append({"metric": metric, "from": left["value"], "to": right["value"],
                                   "before": a, "after": b})
    result = check(name, "FAIL" if violations else "PASS",
                  "fixed-placement weak monotonicity; a flat result does not prove latency sensitivity")
    result.update(violations=violations, observed_sensitivity=changed)
    return result


def validate_observation(observation):
    checks = []
    counters = list(observation.get("resource_bytes", {}).values()) + list(observation.get("resource_service_ns", {}).values())
    complete = observation.get("complete_resource_accounting", False)
    checks.append(check("resource_attribution", "PASS" if complete and counters and all(
        finite_nonnegative(v) for v in counters) else "FAIL", observation.get("accounting_errors", [])))
    totals = [observation.get(k) for k in ("counted_resource_bytes", "batch_resource_bytes", "reported_resource_bytes")]
    checks.append(check("resource_byte_conservation", "PASS" if complete and close(totals[0], totals[1])
        and close(totals[1], totals[2]) else "FAIL", "execution demands = batch sums = report; hop/cache bytes deliberately counted separately"))
    metrics = observation.get("metrics", {})
    checks.append(check("timing_finite", "PASS" if all(finite_nonnegative(metrics.get(k)) and metrics[k] > 0
        for k in ("ttft_ns", "tpot_ns", "e2e_ns")) else "FAIL", "no missing, NaN, negative or zero latency accepted"))
    summary = observation.get("summary", {})
    checks.append(check("request_completion", "PASS" if summary.get("completed_requests", 0) > 0
        and summary.get("completed_requests") == observation.get("expected_requests")
        and observation.get("actual_output_tokens") == observation.get("expected_output_tokens")
        and observation.get("expected_output_tokens", 0) > 0
        and summary.get("rejected_requests") == 0 else "FAIL", "requests must finish, not be rejected to make latency look shorter"))
    kv, state = observation.get("kv_cache"), observation.get("linear_state")
    checks.append(check("kv_linear_state_separation", "PASS" if isinstance(kv, dict) and isinstance(state, dict)
        and finite_nonnegative(kv.get("logical_bytes_per_token")) and finite_nonnegative(state.get("bytes_per_request"))
        and kv["logical_bytes_per_token"] > 0 and state["bytes_per_request"] > 0 else "FAIL",
        "full-attention KV grows with context; recurrent state is reported independently"))
    placement = observation.get("placement", {})
    mapping, options = placement.get("tensor_to_component", {}), placement.get("options", {})
    components = set(observation.get("component_ids", ()))
    mismatches = []
    for op, target in options.get("operator_targets", {}).items():
        if placement.get("op_to_component", {}).get(op) != target:
            mismatches.append(op)
    for tensor, target in options.get("weight_tensor_targets", {}).items():
        if mapping.get(tensor) != target:
            mismatches.append(tensor)
    for field, tensor in (("kv_cache_target", "kv_cache"), ("linear_state_target", "linear_state"),
                          ("linear_state_offload_target", "linear_state_offload")):
        if options.get(field) and mapping.get(tensor) != options[field]:
            mismatches.append(tensor)
    for field, state_type, map_key in (("kv_layer_targets", "kv_cache", "kv_layer_components"),
            ("linear_state_layer_targets", "linear_state", "linear_state_layer_components")):
        for layer, target in options.get(field, {}).items():
            if (mapping.get(f"{layer}.{state_type}") != target
                    or placement.get("memory_tiers", {}).get(map_key, {}).get(layer) != target):
                mismatches.append(f"{layer}.{state_type}")
    unknown = sorted({v for v in mapping.values() if v not in components})
    checks.append(check("placement_constraints", "FAIL" if mismatches or unknown or not mapping else "PASS",
                        {"mismatched": mismatches, "unknown_components": unknown}))
    weights, shards = placement.get("weight_tensor_details", {}), placement.get("rank_weight_shards", {})
    errors = []
    for tensor, detail in weights.items():
        rows = shards.get(tensor)
        # CIM replicas/backing are different capacity domains; never equate
        # their totals to unique model bytes. Only compare rank-sharded tensors.
        if detail.get("residency") != "rank_sharded_storage":
            continue
        if not rows or any(r.get("storage_component_id") not in components
                           or not finite_nonnegative(r.get("physical_bytes")) for r in rows):
            errors.append(tensor)
        elif not close(sum(r["physical_bytes"] for r in rows), detail.get("total_physical_bytes")):
            errors.append(tensor)
    checks.append(check("weight_shard_attribution", "FAIL" if errors or not weights else "PASS",
                        {"invalid_tensors": errors, "tensor_count": len(weights)}))
    return checks


def scenario_coverage(scenario, observation):
    """Never claim exercising a medium because it merely appears in a topology."""
    scenario_mode = ("hbf_" + scenario[len("hbf_media_"):]) if scenario.startswith("hbf_media_") else scenario
    resources = observation.get("resource_bytes", {})
    kv = observation.get("kv_cache", {})
    activity = component_activity(observation, "hbf0")
    coverage = {"validation_status": UNVALIDATED, "hardware_accuracy": "NOT_VALIDATED"}
    if scenario_mode == "hbf_remote_flash":
        reads = resources.get("component.hbf0.read", 0)
        writes = resources.get("component.hbf0.write", 0)
        exercised = (kv.get("swap_events", 0) > 0 and kv.get("swap_in_bytes", 0) > 0
            and kv.get("swap_out_bytes", 0) > 0 and reads > 0 and writes > 0)
        coverage.update(remote_flash_kv="COVERED" if exercised else "NOT_COVERED",
            hbf_traffic_semantics="combined KV and linear-state physical endpoint traffic, NOT pure KV",
            linear_state_swap_in_bytes=observation.get("linear_state", {}).get("swap_in_bytes"),
            linear_state_offload_bytes=observation.get("linear_state", {}).get("offload_bytes"),
            kv_swap_events=kv.get("swap_events"), kv_swap_in_bytes=kv.get("swap_in_bytes"),
            kv_swap_out_bytes=kv.get("swap_out_bytes"), hbf_read_bytes=reads, hbf_write_bytes=writes,
            remote_flash_weights="NOT_COVERED: this case exercises KV swap, not weight cold streaming")
    elif scenario_mode == "hbf_remote_weights":
        reads = resources.get("component.hbf0.read", 0)
        coverage.update(remote_flash_weights="COVERED" if reads > 0 else "NOT_COVERED",
                        hbf_read_bytes=reads, remote_flash_kv="NOT_COVERED: weight streaming case")
    elif scenario_mode.startswith("hbf_active"):
        coverage.update(active_memory="COVERED" if activity and activity["bytes"] > 0 else "NOT_COVERED",
                        hbf_activity=activity, page_level_migration="NOT_IMPLEMENTED")
    elif scenario.startswith("dual_dram"):
        activity0, activity1 = [component_activity(observation, c) for c in ("dram0", "dram1")]
        coverage["dram_controllers"] = "COVERED" if all(a and a["bytes"] > 0 for a in (activity0, activity1)) else "NOT_COVERED"
        paths = {r: b for r, b in resources.items() if r.startswith("link.vertical_dram") or r in {
            "stack0.phy0", "stack0.phy1", "stack0.shared_phy_noc"}}
        coverage["vertical_link_bytes"] = paths
        coverage["vertical_traffic"] = "COVERED" if all(any(
            (r == "link.vertical_" + dram or r.startswith("link.vertical_" + dram + ".")) and b > 0 for r, b in paths.items())
            for dram in ("dram0", "dram1")) else "NOT_COVERED"
        if scenario.endswith("shared"):
            owners = observation.get("physical_resource_owners", {})
            shared_bytes = {r: b for r, b in resources.items()
                if owners.get(r, r) == "stack0.shared_phy_noc" and b > 0}
            coverage["shared_fabric_logical_bytes"] = shared_bytes
            coverage["shared_fabric"] = "COVERED" if all(
                owners.get("link.vertical_" + dram) == "stack0.shared_phy_noc"
                and resources.get("link.vertical_" + dram, 0) > 0
                for dram in ("dram0", "dram1")) else "NOT_COVERED"
        coverage["cim_execution"] = "COVERED" if any(v == "cim0" for v in observation.get(
            "placement", {}).get("op_to_component", {}).values()) and observation.get("resource_service_ns", {}).get("cim0.array", 0) > 0 else "NOT_COVERED"
    return coverage


def coverage_check(scenario, coverage):
    scenario_mode = ("hbf_" + scenario[len("hbf_media_"):]) if scenario.startswith("hbf_media_") else scenario
    if scenario not in SCENARIOS:
        return check("scenario_mechanism_coverage", "NOT_COVERED", "unknown scenario: " + scenario)
    required = {
        "hbf_remote_flash": ("remote_flash_kv",),
        "hbf_remote_weights": ("remote_flash_weights",),
        "dual_dram_only": ("dram_controllers", "vertical_traffic"),
        "dual_dram_only_shared": ("dram_controllers", "vertical_traffic", "shared_fabric"),
        "dual_dram_cim": ("dram_controllers", "vertical_traffic", "cim_execution"),
        "dual_dram_cim_shared": ("dram_controllers", "vertical_traffic", "cim_execution", "shared_fabric"),
        "dual_dram_cim_converted": ("dram_controllers", "vertical_traffic", "cim_execution"),
        "dual_dram_cim_converted_shared": ("dram_controllers", "vertical_traffic", "cim_execution", "shared_fabric"),
        "dual_dram_cim_tiled": ("dram_controllers", "vertical_traffic", "cim_execution"),
        "dual_dram_cim_tiled_shared": ("dram_controllers", "vertical_traffic", "cim_execution", "shared_fabric"),
    }.get(scenario_mode, ("active_memory",) if scenario_mode.startswith("hbf_active") else ())
    missing = [key for key in required if coverage.get(key) != "COVERED"]
    return check("scenario_mechanism_coverage", "NOT_COVERED" if missing else "PASS",
                 {"required": list(required), "missing": missing,
                  "hardware_accuracy": "UNVALIDATED"})


def set_read_latency(scenario, component_id, latency_ns):
    if not finite_nonnegative(latency_ns):
        raise ValueError("latency must be finite and nonnegative")
    component = scenario.hardware.get_component(component_id)
    profiles = {kind: dict(items) for kind, items in scenario.component_profiles.items()}
    if component.normalized_kind == "hbf" and not component.is_active_memory:
        replacement = replace(component, metadata={**component.metadata, "read_latency_ns": latency_ns})
    else:
        kind = scenario.component_profile_kind(component)
        profile = scenario.resolve_component_profile(component_id)
        if not hasattr(profile, "read_latency_ns"):
            raise UnsupportedScenario("TODO typed read_latency_ns: " + component_id)
        profile_id = f"sweep-{component_id}"
        profiles[kind][profile_id] = replace(profile, read_latency_ns=latency_ns)
        replacement = replace(component, cost_profile_id=profile_id, metadata={**component.metadata,
            **({"read_latency_ns": latency_ns} if component.normalized_kind == "hbf" else {})})
    return replace(scenario, component_profiles=profiles, hardware=replace(scenario.hardware,
        components=tuple(replacement if c.component_id == component_id else c for c in scenario.hardware.components)))


def placement_mapping(placement):
    """Only actual mappings/physical allocation, never policy or signed evidence."""
    if isinstance(placement, dict):  # compact execution observation
        return {key: placement.get(key, {}) for key in (
            "op_to_component", "tensor_to_component", "tensor_bytes", "memory_tiers",
            "weight_tensor_details", "rank_weight_shards")}
    decision = placement.metadata.get("control_plane", {}).get("decision", {})
    return {"op_to_component": dict(placement.op_to_component),
            "tensor_to_component": dict(placement.tensor_to_component),
            "tensor_bytes": dict(placement.tensor_bytes),
            "memory_tiers": placement.metadata.get("memory_tiers", {}),
            "weight_tensor_details": decision.get("weight_tensor_details", {}),
            "rank_weight_shards": decision.get("rank_weight_shards", {})}


def solve_placement(scenario):
    from heterollm_sim.config import scenario_from_dict
    from heterollm_sim.control_plane_planner import plan_runtime_placement
    decision = plan_runtime_placement(scenario_from_dict(scenario_payload(scenario)))
    if not decision.fully_placed:
        raise UnsupportedScenario("placement unavailable: " + str(decision.unplaced))
    return decision.placement


def fixed_reference(scenario, component_id):
    """Solve at 100ns; express the result as V4 policy, not hand-authored maps."""
    from heterollm_sim.control_plane_planner import PlacementPolicy
    if "operator_targets" not in {f.name for f in fields(PlacementPolicy)}:
        raise UnsupportedScenario("fixed placement requires PlacementPolicy.operator_targets")
    reference = solve_placement(set_read_latency(scenario, component_id, 100.0))
    mapping = placement_mapping(reference)
    tensors = mapping["tensor_to_component"]
    weights = mapping["weight_tensor_details"]
    if not mapping["op_to_component"] or not weights:
        raise UnsupportedScenario("fixed placement requires explicit operator and weight mappings")
    for tensor in weights:
        owners = {s.get("storage_component_id") for s in
                  mapping["rank_weight_shards"].get(tensor, [])}
        if owners and owners != {tensors.get(tensor)}:
            raise UnsupportedScenario("fixed single-target policy cannot freeze multi-owner weight: " + tensor)
    options = dict(scenario.placement.metadata.get("control_plane", {}).get("policy", {}).get("options", {}))
    options.update(operator_targets=dict(mapping["op_to_component"]),
                   weight_tensor_targets={tensor: tensors[tensor] for tensor in weights})
    for tensor, layer_key, option_key in (
            ("kv_cache", "kv_layer_components", "kv_layer_targets"),
            ("linear_state", "linear_state_layer_components", "linear_state_layer_targets")):
        layer_map = mapping["memory_tiers"].get(layer_key)
        if layer_map:
            options[option_key] = dict(layer_map)
        if tensor in tensors:
            options[tensor + "_target"] = tensors[tensor]
    if "linear_state_offload" in tensors:
        options["linear_state_offload_target"] = tensors["linear_state_offload"]
    # Keep the original (unsolved) placement and all non-policy configuration.
    # Each point obtains fresh evidence through the ordinary control-plane solve.
    metadata = {**scenario.placement.metadata,
                "control_plane": {"policy": {"options": options}}}
    return replace(scenario, placement=replace(scenario.placement, metadata=metadata)), mapping


def require_fixed_mapping(expected, actual):
    if stable_hash(expected) != stable_hash(actual):
        changed = [key for key in expected if stable_hash(expected[key]) != stable_hash(actual.get(key))]
        raise UnsupportedScenario("fixed placement mapping changed: " + ", ".join(changed))


def plan_cells(scenarios, *, include_long=False):
    """Three short single-factor points plus at most one long shape per scenario."""
    cells = []
    for scenario in scenarios:
        component = "hbf0" if scenario.startswith("hbf") else "dram0" if scenario.startswith("dual_dram") else "hbm0"
        for latency in (50.0, 100.0, 200.0):
            cells.append({"id": f"{scenario}-p8-o4-b1-lat{int(latency)}", "scenario": scenario,
                          "prompt_tokens": 8, "output_tokens": 4, "batch": 1,
                          "component": component, "read_latency_ns": latency, "shape": "short"})
        if include_long:
            cells.append({"id": f"{scenario}-p512-o128-b1-lat100", "scenario": scenario,
                          "prompt_tokens": 512, "output_tokens": 128, "batch": 1,
                          "component": component, "read_latency_ns": 100.0, "shape": "long"})
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--placement-mode", choices=("adaptive", "fixed"), default="adaptive",
                        help="fixed: solve at 100ns, constrain and verify every point")
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "sweep.json")
    args = parser.parse_args(argv)
    try:
        check_output_path(args.output)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    scenarios = PRIMARY_SCENARIOS if args.scenario == "all" else (args.scenario,)
    payload = {"schema": "memory-tier-sweep/v1", "validation_status": UNVALIDATED,
               "native_execution": False, "limitations": LIMITATIONS, "cells": [], "checks": [],
               "placement_mode": args.placement_mode}
    try:
        source = source_identity()
        model, identity = load_model(args.model)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        payload.update(status="BLOCKED", error_type=type(exc).__name__, error=str(exc))
        write_output(args.output, payload)
        print(str(exc), file=sys.stderr)
        return 2
    payload["model"] = identity
    payload["source"] = source
    references = {}
    for cell in plan_cells(scenarios, include_long=args.include_long):
        row = dict(cell)
        try:
            scenario = build_scenario(model, cell["scenario"], prompt_tokens=cell["prompt_tokens"],
                output_tokens=cell["output_tokens"], batch=cell["batch"])
            expected_mapping = None
            if args.placement_mode == "fixed":
                key = (cell["scenario"], cell["prompt_tokens"], cell["output_tokens"], cell["batch"], cell["component"])
                if key not in references:
                    references[key] = fixed_reference(scenario, cell["component"])
                scenario, expected_mapping = references[key]
                row["reference_read_latency_ns"] = 100.0
                row["reference_mapping_sha256"] = stable_hash(expected_mapping)
            scenario = set_read_latency(scenario, cell["component"], cell["read_latency_ns"])
            if expected_mapping is not None:
                actual_mapping = placement_mapping(solve_placement(scenario))
                require_fixed_mapping(expected_mapping, actual_mapping)
                row["solved_mapping_sha256"] = stable_hash(actual_mapping)
                row["fixed_mapping_check"] = "PASS"
            authored = scenario_payload(scenario)
            from heterollm_sim.config import scenario_from_dict
            if scenario_payload(scenario_from_dict(authored)) != authored:
                raise ValueError("config round trip changed experiment")
            row["config_roundtrip"] = "PASS"
            row["scenario_sha256"] = stable_hash(authored)
            row["experiment_config"] = {k: v for k, v in authored.items() if k != "model"}
            row["model_config_sha256"] = stable_hash(authored["model"])
            row["actual_batch"] = len(scenario.workload.requests)
            if args.dry_run:
                if expected_mapping is None:
                    solve_placement(scenario)
                row["status"] = "PLANNED"
                row["scenario_config"] = scenario_payload(scenario)
            else:
                row["observation"] = execute_scenario(scenario)
                if expected_mapping is not None:
                    actual_mapping = placement_mapping(row["observation"]["placement"])
                    require_fixed_mapping(expected_mapping, actual_mapping)
                    row["executed_mapping_sha256"] = stable_hash(actual_mapping)
                row["checks"] = validate_observation(row["observation"])
                row["coverage"] = scenario_coverage(cell["scenario"], row["observation"])
                row["checks"].append(coverage_check(cell["scenario"], row["coverage"]))
                row["status"] = "FAILED_CHECKS" if any(c["status"] != "PASS" for c in row["checks"]) else "SIMULATED"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(status="BLOCKED", error_type=type(exc).__name__, error=str(exc))
        if source != source_identity():
            row.update(status="INVALIDATED", error="source drift; remaining cells not executed")
        payload["cells"].append(row)
        print(json.dumps({"cell": row["id"], "status": row["status"], "error": row.get("error")}, ensure_ascii=False), flush=True)
        if row["status"] == "INVALIDATED":
            break
    if not args.dry_run:
        for scenario in scenarios:
            rows = [r for r in payload["cells"] if r["scenario"] == scenario and r["shape"] == "short"]
            points = [{"value": r["read_latency_ns"], "observation": r["observation"]}
                      for r in rows if r["status"] == "SIMULATED"]
            result = (monotonicity(points) if len(points) == len(rows) else
                check("monotonicity:read_latency_ns", "NOT_COVERED",
                      "one or more planned points blocked or failed; do not validate a selected subset"))
            result["scenario"] = scenario
            payload["checks"].append(result)
            if scenario == "hbf_idle" and len(points) >= 2:
                payload["checks"].append(zero_traffic_invariance(points[0]["observation"], points[-1]["observation"], "hbf0"))
    payload["source_stable"] = source == source_identity()
    payload["status"] = ("INVALIDATED" if not payload["source_stable"] else "INCOMPLETE" if
        any(r["status"] not in {"PLANNED", "SIMULATED"} for r in payload["cells"]) or
        any(c["status"] != "PASS" for c in payload["checks"]) else "PLANNED" if args.dry_run else "SIMULATED")
    try:
        write_output(args.output, payload)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if payload["status"] in {"PLANNED", "SIMULATED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
