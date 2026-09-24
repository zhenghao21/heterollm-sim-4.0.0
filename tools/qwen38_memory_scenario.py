"""Read-only GGUF-sidecar Qwen3.8 memory experiments; never run native hardware.

All hardware values are analytical examples, NOT measured HBF/3D-SoC data.
Outputs are exclusive-create and confined to the new memory-tier experiment folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from heterollm_sim.config import hardware_from_dict, scenario_from_dict
from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.gguf_parity import (build_model_from_gguf, gguf_metadata_digest,
                                      read_gguf_metadata_cache)
from heterollm_sim.ir import (ComponentSpec, KVCachePolicy, LinkSpec, ParallelSpec,
                             PlacementSpec, PortSpec, RankMappingSpec, RequestSpec,
                             SchedulerSpec, WorkloadSpec)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.serde import stable_hash, to_primitive
from tools.native_grid_predict import file_ref, write_new

OUTPUT_ROOT = ROOT / "artifacts/memory_tier_scenarios_20260922"
DEFAULT_MODEL = ROOT / "artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf"
SCENARIOS = ("baseline", "hbf_idle", "hbf_remote_flash", "hbf_remote_weights", "hbf_active_memory",
             "hbf_active_weights", "hbf_active_kv", "hbf_active_split",
             "dual_dram_cim", "dual_dram_cim_shared", "dual_dram_only", "dual_dram_only_shared", "dual_dram_cim_converted", "dual_dram_cim_converted_shared",
             "dual_dram_cim_tiled", "dual_dram_cim_tiled_shared",
             "hbf_media_remote_flash", "hbf_media_remote_weights")
UNVALIDATED = "UNVALIDATED"
LIMITATIONS = [
    "Analytical example parameters only; no HBF, stacked-DRAM or CIM measurements.",
    "No new native acquisition, calibration, accuracy score or holdout acceptance.",
    "Layer-static KV/linear-state partitions; no page-level direct-addressing claim.",
    "GPU reference structural profile is an uncalibrated SoC proxy, not a source-kernel calibration.",
    "Resource-accounted bytes include every hop/cache; they are NOT unique payload bytes.",
    "Sidecar SHA binding and file stat checked; GGUF payload is not reread or rehashed.",
]


class UnsupportedScenario(ValueError):
    """Missing core capability must not be replaced by a fabricated prediction."""


def source_identity():
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((ROOT / "src/heterollm_sim").rglob("*.py"))}
    for name in ("qwen38_memory_scenario.py", "memory_tier_sweep.py"):
        path = ROOT / "tools" / name
        if path.exists():
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": files, "sha256": stable_hash(files)}


def load_model(model_path=DEFAULT_MODEL):
    # No fallback to read_gguf_metadata: that rehashes the entire 14.8 GB payload.
    path = Path(model_path).resolve(strict=True)
    sidecar = Path(str(path) + ".metadata.json")
    gguf = read_gguf_metadata_cache(path, sidecar, strict=False)
    model = build_model_from_gguf(gguf)
    groups = defaultdict(list)
    for item in model._execution_view.layer_instances:
        groups[item.layer.sequence_mixer].append(item.layer_id)
    if (gguf.architecture != "qwen35" or gguf.n_embd != 5120
            or len(groups["full_attention"]) != 16 or len(groups["linear_attention"]) != 48):
        raise ValueError("Expected the project's real Qwen3.8-27B 48-linear/16-full backbone")
    identity = {"gguf_path": str(path), "gguf_sha256": gguf.sha256,
                "size_bytes": path.stat().st_size, "sidecar": file_ref(sidecar),
                "metadata_digest": gguf_metadata_digest(gguf),
                "architecture": gguf.architecture, "hidden_size": gguf.n_embd,
                "quantization_label": gguf.quantization,
                "tensor_types": dict(sorted(Counter(t.type_name for t in gguf.tensor_directory).items())),
                "tensor_count": gguf.tensor_count, "layers": dict(groups),
                "dense_f16_matrix_count": sum(t.type_name == "F16" and len(t.shape) == 2 for t in gguf.tensor_directory),
                "nextn_predict_layers": gguf.metadata.get("qwen35.nextn_predict_layers"),
                "mtp_enabled": False, "gguf_payload_rehashed": False}
    return model, identity


def baseline_manifest(model_identity, source, scenario):
    return {"schema": "memory-tier-baseline/v1", "validation_status": UNVALIDATED,
            "source": source, "model": model_identity,
            "scenario_sha256": stable_hash(scenario_payload(scenario)),
            "workload": to_primitive(scenario.workload), "native_execution": False,
            "limitations": LIMITATIONS, "protected_artifacts_modified": False}


def scenario_payload(scenario):
    """Use the existing V4 authoring schema, not the internal dataclass layout."""
    profiles = {"components": to_primitive(scenario.component_profiles),
                "host_orchestration": to_primitive(scenario.host_orchestration_profile),
                "fusion": to_primitive(scenario.fusion_policy),
                "cim_interconnect": to_primitive(scenario.cim_interconnect),
                "runtime": to_primitive(scenario.runtime_profile)}
    return {"schema_version": scenario.schema_version, "name": scenario.name,
            "hardware": to_primitive(scenario.hardware), "model": to_primitive(scenario.model),
            "placement": to_primitive(scenario.placement), "workload": to_primitive(scenario.workload),
            "profiles": profiles, "weights_resident": scenario.weights_resident,
            "assumptions": list(scenario.assumptions)}


def _profile_fields(profile, **values):
    missing = set(values) - {field.name for field in fields(profile)}
    if missing:
        raise UnsupportedScenario("TODO core memory service fields: " + ", ".join(sorted(missing)))
    return replace(profile, **values)


def _constraints(scenario, options):
    missing = set(options) - {field.name for field in fields(PlacementPolicy)}
    if missing:
        raise UnsupportedScenario("TODO core placement constraints: " + ", ".join(sorted(missing)))
    metadata = {**scenario.placement.metadata,
                "control_plane": {"policy": {"options": {"solver": "builtin", **options}}}}
    return replace(scenario, placement=replace(scenario.placement, metadata=metadata))


def _add_hbf(scenario, *, active, latency_ns, media=False):
    gpu = scenario.hardware.get_component("gpu0")
    # These are the existing storage-test example values, not HBF specifications.
    port = PortSpec("hbf", "UCIe", "endpoint", bandwidth_gbps=(12288.0 if media else 128.0), payload="streaming")
    metadata = {"validation_status": UNVALIDATED, "access_mode": "memory" if active else "remote_flash",
                "read_latency_ns": latency_ns, "write_latency_ns": 200.0,
                "transfer_granularity_bytes": 4096, "max_outstanding_requests": 1}
    if media:
        # OCP-HBF host/page granularity plus a FlashAccel-style cold-page
        # latency coordinate. These are analytical references, not a device
        # calibration or a claim that all HBF implementations share them.
        metadata["hbf_media"] = {
            "version": "cold_page_v1", "host_transaction_bytes": 64,
            "host_max_request_bytes": 4096, "media_page_bytes": 4096,
            "command_queue_depth": 1024, "media_parallelism": 96,
            "physical_planes": 96, "page_read_latency_ns": 4000.0,
            "page_program_latency_ns": 75000.0,
            "access_pattern": "contiguous_page_aligned",
        }
    profiles = {kind: dict(items) for kind, items in scenario.component_profiles.items()}
    profile_id = None
    if active:
        profile_id = "example-hbf-memory"
        profiles["host_memory"][profile_id] = _profile_fields(
            scenario.host_memory_profile, name=profile_id, resource_id="hbf0.controller",
            bandwidth_gb_s=4.0, efficiency=1.0, read_latency_ns=latency_ns, write_latency_ns=200.0,
            transaction_bytes=4096, max_outstanding_requests=32)
        metadata.update(write_buffer_bytes=0, resident_access_path="topology",
                        transfer_granularity_bytes=4096, max_outstanding_requests=32,
                        memory_service_owner="hbf0.controller")
    hbf = ComponentSpec("hbf0", "hbf", (replace(port, port_id="host"),),
                        package_id=gpu.package_id, die_id="hbf_die", capacity_bytes=(512 * 10**9 if media else 32 * 1024**3),
                        read_bandwidth_gbps=(12288.0 if media else 64.0),
                        write_bandwidth_gbps=(1024.0 if media else 32.0),
                        cost_profile_id=profile_id, metadata=metadata)
    link = LinkSpec("gpu-hbf0", "gpu0", "hbf", "hbf0", "host", "UCIe",
                    bandwidth_gbps=(12288.0 if media else 32.0), latency_ns=20.0, payload="streaming")
    hardware = replace(scenario.hardware, components=tuple(
        replace(c, ports=c.ports + (port,)) if c.component_id == "gpu0" else c
        for c in scenario.hardware.components) + (hbf,), links=scenario.hardware.links + (link,))
    return replace(scenario, hardware=hardware, component_profiles=profiles)


def build_scenario(model, scenario="baseline", *, prompt_tokens=8, output_tokens=4,
                   batch=1, latency_ns=100.0):
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario: " + scenario)
    if type(latency_ns) not in (int, float) or not math.isfinite(latency_ns) or latency_ns < 0:
        raise ValueError("latency must be finite and nonnegative")
    if any(type(value) is not int or value <= 0 for value in (prompt_tokens, output_tokens, batch)):
        raise ValueError("prompt/output/batch must be positive integers")
    if output_tokens < 2:
        raise ValueError("at least two output tokens are needed to test TPOT")
    base = build_reference_scenario()
    media_variant = scenario.startswith("hbf_media_")
    scenario_mode = ("hbf_" + scenario[len("hbf_media_"):]) if media_variant else scenario
    # Remote KV needs actual pressure, not an unused offload label.
    actual_batch = max(2, batch) if scenario_mode == "hbf_remote_flash" else batch
    workload = WorkloadSpec(name=f"p{prompt_tokens}-o{output_tokens}-b{actual_batch}",
        requests=tuple(RequestSpec(f"request-{i:04d}", 0.0, prompt_tokens, output_tokens)
                       for i in range(actual_batch)), random_seed=7,
        scheduler=SchedulerSpec(mode="continuous", max_num_seqs=actual_batch,
            max_num_batched_tokens=max(8, prompt_tokens * actual_batch),
            prefill_chunk_tokens=max(8, prompt_tokens), preemption_enabled=False), mtp=None)
    configured = replace(base, name=f"qwen38-{scenario}-{workload.name}", model=model,
        placement=PlacementSpec(model.name, base.hardware.name, parallel=ParallelSpec(),
                                kv_policy=KVCachePolicy(dtype="fp16")), workload=workload,
        assumptions=tuple(base.assumptions) + tuple(LIMITATIONS))
    if scenario == "baseline":
        return configured
    # Ask the existing planner for its canonical weight IDs rather than guessing
    # graph-group tensor IDs (which differ from executable layer IDs).
    decision = plan_runtime_placement(configured)
    if not decision.fully_placed:
        raise UnsupportedScenario("reference model cannot be placed: " + str(decision.unplaced))
    weights = sorted(decision.placement.metadata["control_plane"]["decision"]["weight_tensor_details"])
    layers = model._execution_view.layer_instances
    if scenario.startswith("hbf"):
        active = scenario_mode.startswith("hbf_active")
        configured = _add_hbf(configured, active=active, latency_ns=latency_ns, media=media_variant)
        if scenario_mode == "hbf_idle":
            return configured
        if active and not configured.hardware.get_component("hbf0").is_active_memory:
            raise UnsupportedScenario("TODO core HBF metadata.access_mode=memory support")
        primary = "hbf0" if scenario == "hbf_active_memory" else "hbm0"
        configured = replace(configured, placement=replace(configured.placement,
            parallel=ParallelSpec(rank_mapping=(RankMappingSpec(0, "gpu0", 0, 0, 0, memory_component_id=primary),))))
        options = {"weight_tensor_targets": {t: "hbm1" for t in weights},
                   "kv_cache_target": "hbm0", "linear_state_target": "hbm2"}
        if scenario_mode in {"hbf_active_weights", "hbf_active_memory"}:
            options["weight_tensor_targets"] = {t: "hbf0" if ".mlp_weights" in t else "hbm1" for t in weights}
        if scenario_mode == "hbf_remote_weights":
            options["weight_tensor_targets"] = {t: "hbf0" if ".mlp_weights" in t else "hbm1" for t in weights}
            configured = replace(configured, weights_resident=False)
        if scenario_mode == "hbf_active_kv":
            options["kv_cache_target"] = "hbf0"
        if scenario_mode == "hbf_active_split":
            options["kv_layer_targets"] = {item.layer_id: ("hbf0" if i % 2 else "hbm0")
                for i, item in enumerate(item for item in layers if not item.layer.is_linear_attention)}
            options["linear_state_layer_targets"] = {item.layer_id: ("hbf0" if i % 2 else "hbm2")
                for i, item in enumerate(item for item in layers if item.layer.is_linear_attention)}
        if scenario_mode == "hbf_remote_flash":
            options["linear_state_offload_target"] = "hbf0"
            from heterollm_sim.serving import _kv_bytes_per_token
            kv_token_bytes = _kv_bytes_per_token(configured, "fp16")
            page_tokens = prompt_tokens
            # Two sequences initially fit, but cannot both grow another page.
            capacity = (actual_batch + 1) * page_tokens * kv_token_bytes
            configured = replace(configured, hardware=replace(configured.hardware, components=tuple(
                replace(c, capacity_bytes=capacity) if c.component_id == "hbm0" else c
                for c in configured.hardware.components)),
                placement=replace(configured.placement,
                    metadata={**configured.placement.metadata, "linear_state_offload_mode": "pressure"},
                    kv_policy=KVCachePolicy(
                    cache_component="hbm0", offload_component="hbf0", dtype="fp16",
                    tokens_per_page=page_tokens, preemption_mode="swap")),
                workload=replace(workload, scheduler=replace(workload.scheduler,
                    preemption_enabled=True, preemption_policy="swap")))
        return _constraints(configured, options)
    from heterollm_sim.architecture_presets import materialize_architecture_payload
    preset_id = ("soc-2x-dram-sram-cim-shared-phy-noc" if scenario.endswith("shared")
                 else "soc-2x-dram-sram-cim")
    try:
        hardware = hardware_from_dict(materialize_architecture_payload(preset_id))
    except KeyError as exc:
        raise UnsupportedScenario("TODO core topology preset: " + preset_id) from exc
    profiles = {kind: dict(items) for kind, items in configured.component_profiles.items()}
    if "_only" not in scenario:
        profiles["cim"]["legacy-cim"] = _profile_fields(
            configured.cim_profile, arithmetic_mode="fp16_fp32_analytical",
            float_cycles_per_eval=7.0, float_accumulator_outputs_per_cycle=2.0,
            float_contract_basis="UNVALIDATED user-defined FP16 tile / FP32 rounded reduction hypothesis; "
                "does NOT provide packed IQ/Q conversion, numerical equivalence, or measured silicon support")
    if "_converted" in scenario or "_tiled" in scenario:
        # Explicit sensitivity coordinates, not a fitted/measured decoder.
        # Packed artifacts remain unchanged and each invocation is cold.
        tiled = "_tiled" in scenario
        profiles["cim"]["legacy-cim"] = _profile_fields(
            profiles["cim"]["legacy-cim"],
            weight_conversion_mode=("packed_to_fp16_tiled_cold" if tiled else "packed_to_fp16_cold"),
            weight_decode_elements_per_ns=16.0,
            activation_fp32_to_fp16_elements_per_ns=16.0,
            conversion_scratch_capacity_bytes=(1 * 1024**2 if tiled else 512 * 1024**2),
            weight_capacity_bytes=(2 * 1024**2 if tiled else 512 * 1024**2),
            **({"tile_m": 8, "tile_k": 256, "tile_n": 256, "max_m_replication": 1} if tiled else {}),
            conversion_contract_basis=(
                "UNVALIDATED sensitivity: CIM-local tiled IQ3_S/IQ4_XS decode; 8x256x256 tiles, "
                "16 elements/ns assumed, serial tile stream, no cross-invocation reuse or accuracy claim"
                if tiled else
                "UNVALIDATED sensitivity: CIM-local full-matrix IQ3_S/IQ4_XS decode; "
                "16 elements/ns assumed, FP16 materialization, no cross-invocation reuse or accuracy claim"))
        hardware = replace(hardware, components=tuple(
            replace(c, capacity_bytes=(3 * 1024**2 if tiled else 1024 * 1024**2))
            if c.component_id == "cim0" else c
            for c in hardware.components))
        configured = replace(configured, weights_resident=False)
    for component_id in ("dram0", "dram1"):
        profiles["host_memory"][f"example-{component_id}"] = _profile_fields(
            configured.host_memory_profile, name=f"example-{component_id}",
            resource_id=f"{component_id}.controller", read_latency_ns=latency_ns,
            write_latency_ns=100.0, transaction_bytes=256, max_outstanding_requests=32)
    hardware = replace(hardware, components=tuple(
        replace(c, cost_profile_id=f"example-{c.component_id}", metadata={**c.metadata,
            "resident_access_path": "topology", "validation_status": UNVALIDATED})
        if c.component_id in {"dram0", "dram1"} else c for c in hardware.components))
    configured = replace(configured, hardware=hardware, component_profiles=profiles,
        host_orchestration_profile=replace(configured.host_orchestration_profile,
            gpu_component_id="soc0", submission_resource_id="soc0.command_queue"),
        runtime_profile=replace(configured.runtime_profile, gpu_controllers={
            "soc0": configured.runtime_profile.gpu_controllers["gpu0"]}),
        placement=replace(configured.placement, hardware_name=hardware.name,
            parallel=ParallelSpec(rank_mapping=(RankMappingSpec(0, "soc0", 0, 0, 0,
                memory_component_id="dram0", cim_component_id="cim0"),))))
    return _constraints(configured, {"weight_tensor_targets": {
        t: ("cim0" if t == "layer-000.mlp_weights" and "_only" not in scenario else
            "dram0" if i % 2 == 0 else "dram1") for i, t in enumerate(weights)},
        "kv_cache_target": "dram0", "linear_state_target": "dram1",
        **({"allow_cold_cim_streaming": True} if ("_converted" in scenario or "_tiled" in scenario) else {})})


def batch_demands(metadata):
    """Read one authoritative ledger only; never infer bytes from busy time."""
    stages = metadata.get("execution_stages")
    if stages:
        tasks = [task for stage in stages for task in stage.get("execution_tasks", ())]
        ids = [task.get("task_id") for task in tasks]
        if not tasks or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("missing/duplicate execution task id")
        if any("resource_demands" not in task for task in tasks):
            raise ValueError("execution task missing resource_demands")
        return [demand for task in tasks for demand in task["resource_demands"]], "execution_stages"
    demands = metadata.get("resource_demands")
    if isinstance(demands, (list, tuple)) and demands:
        if not all(isinstance(d, dict) and {"resource_id", "bytes_moved", "service_ns"} <= d.keys()
                   for d in demands):
            raise ValueError("malformed resource_demands ledger")
        return list(demands), "resource_demands"
    raise ValueError("no complete resource demand ledger")


def compact_observation(result, report):
    """Aggregate real execution demands, not guessed traffic from utilization."""
    resource_bytes, resource_service = defaultdict(float), defaultdict(float)
    linear_bytes = Counter()
    ledger_sources = Counter()
    conversion_batches = []
    errors = []
    counted_bytes = 0.0
    batch_bytes = 0.0
    observed_batches = 0
    for batch in result.serving.batches:
        observed_batches += 1
        meta = batch.cost.metadata
        batch_bytes += meta.get("resource_accounted_bytes", 0)
        linear_bytes.update(meta.get("linear_state_bytes", {}))
        if meta.get("cim_weight_conversion"):
            conversion_batches.append({"batch_id": batch.cohort_id, "kind": batch.kind,
                                       "audit": to_primitive(meta["cim_weight_conversion"])})
        try:
            demands, ledger_source = batch_demands(meta)
        except ValueError as exc:
            errors.append(f"{batch.kind}: {exc}")
            continue
        ledger_sources[ledger_source] += 1
        this_batch_bytes = 0
        for demand in demands:
            resource = demand["resource_id"]
            moved = demand["bytes_moved"]
            if not resource or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in (moved, demand["service_ns"])):
                errors.append("invalid raw resource demand: " + str(resource))
                continue
            resource_bytes[resource] += moved
            resource_service[resource] += demand["service_ns"]
            counted_bytes += moved
            this_batch_bytes += moved
        if not math.isclose(this_batch_bytes, meta.get("resource_accounted_bytes", -1), rel_tol=1e-10, abs_tol=1e-5):
            errors.append(f"{batch.kind}: per-batch demand bytes differ from aggregate")
    if observed_batches != report["summary"].get("batch_count"):
        errors.append("incomplete batch history")
    placement = result.scenario.placement
    constraints = placement.metadata.get("control_plane", {}).get("policy", {}).get("options", {})
    decision = placement.metadata.get("control_plane", {}).get("decision", {})
    weights = decision.get("weight_tensor_details", {})
    state_maps = placement.metadata.get("memory_tiers", {})
    from heterollm_sim.communication import declared_resource_owners
    physical_owners = dict(declared_resource_owners(result.scenario.hardware))
    components = result.scenario.hardware.component_map()
    profile_owners = {}
    for component in components.values():
        try:
            profile = result.scenario.resolve_component_profile(component.component_id)
        except ValueError:
            continue
        if getattr(profile, "resource_id", None):
            profile_owners[component.component_id] = profile.resource_id
    storage_distribution = {}
    for component_id, component in components.items():
        if not component.is_storage and "cim" not in component.normalized_kind:
            continue
        shards = [shard for rows in decision.get("rank_weight_shards", {}).values()
                  for shard in rows if shard.get("storage_component_id", shard.get("component_id")) == component_id]
        physical_weight_bytes = sum(shard.get("physical_bytes", 0) for shard in shards)
        tensor_ids = sorted(tensor for tensor, target in placement.tensor_to_component.items()
                            if target == component_id)
        owner = profile_owners.get(component_id)
        matching_resources = {r for r in resource_bytes if r == owner
                              or r.startswith("component." + component_id + ".")
                              or r.startswith(component_id + ".")}
        storage_distribution[component_id] = {
            "physical_kind": component.normalized_kind,
            "access_mode": component.metadata.get("access_mode", "default"),
            "capacity_bytes": component.capacity_bytes,
            "persistent_weight_shard_bytes": physical_weight_bytes,
            "tensor_ids": tensor_ids,
            "kv_layer_ids": sorted(layer for layer, target in state_maps.get("kv_layer_components", {}).items()
                                   if target == component_id),
            "linear_state_layer_ids": sorted(layer for layer, target in state_maps.get("linear_state_layer_components", {}).items()
                                             if target == component_id),
            "kv_default_owner": placement.tensor_to_component.get("kv_cache") == component_id,
            "linear_state_default_owner": placement.tensor_to_component.get("linear_state") == component_id,
            "resource_bytes": {r: resource_bytes[r] for r in sorted(matching_resources)},
            "resource_service_ns": {r: resource_service[r] for r in sorted(matching_resources)},
            "traffic_semantics": "per-resource diagnostic; do not sum these as unique payload or peak occupancy",
        }
    metrics = {key: report["summary"][key].get("p50") for key in ("ttft_ns", "tpot_ns", "e2e_ns")}
    return {"schema": "memory-tier-observation/v1", "validation_status": UNVALIDATED,
        "metrics": metrics, "summary": {k: v for k, v in report["summary"].items()
                                        if k != "simulated_subtargets"},
        "resource_bytes": dict(sorted(resource_bytes.items())),
        "resource_service_ns": dict(sorted(resource_service.items())),
        "counted_resource_bytes": counted_bytes, "batch_resource_bytes": batch_bytes,
        "reported_resource_bytes": report["summary"]["resource_accounted_bytes"],
        "accounting_errors": errors, "complete_resource_accounting": not errors,
        "resource_ledger_sources": dict(ledger_sources),
        "kv_cache": report["kv_cache"], "linear_state": report["linear_state"],
        "linear_state_traffic_bytes": dict(linear_bytes),
        "placement": {"tensor_to_component": dict(placement.tensor_to_component),
            "tensor_bytes": dict(placement.tensor_bytes), "memory_tiers": to_primitive(state_maps),
            "weight_tensor_details": to_primitive(weights),
            "rank_weight_shards": to_primitive(decision.get("rank_weight_shards", {})),
            "options": to_primitive(constraints),
            "op_to_component": dict(placement.op_to_component)},
        "cim_weight_conversion_batches": conversion_batches,
        "storage_distribution": storage_distribution,
        "component_ids": sorted(components), "profile_owners": profile_owners,
        "physical_resource_owners": physical_owners,
        "expected_requests": len(result.scenario.workload.requests),
        "expected_output_tokens": sum(r.output_tokens for r in result.scenario.workload.requests),
        "actual_output_tokens": sum(r.visible_output_tokens for r in result.serving.request_metrics.values()),
        "workload_sha256": stable_hash(to_primitive(result.scenario.workload)),
        "placement_sha256": stable_hash({"weights": weights, "tensors": placement.tensor_to_component,
                                         "ops": placement.op_to_component, "states": state_maps}),
        "model_sha256": stable_hash(to_primitive(result.scenario.model)),
        "limits": report.get("response_limits", {}),
        "warnings": report.get("validation_warnings", [])}


def execute_scenario(scenario):
    result = run_scenario(scenario, retention_policy="aggregate")
    report = report_dict(result, visualization_limit=1, visualization_memory_segment_limit=1)
    return compact_observation(result, report)


def check_output_path(path):
    target = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("output must be a NEW file inside " + str(root))
    if target.exists():
        raise FileExistsError("refusing to overwrite existing experiment: " + str(target))
    return target


def write_output(path, payload):
    return write_new(check_output_path(path), payload)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--latency-ns", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true", help="validate/build only, never execute the engine")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "scenario.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        check_output_path(args.output)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = {"schema": "qwen38-memory-scenario/v1", "scenario": args.scenario,
               "validation_status": UNVALIDATED, "native_execution": False, "limitations": LIMITATIONS}
    try:
        source = source_identity()
        model, identity = load_model(args.model)
        scenario = build_scenario(model, args.scenario, prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens, batch=args.batch, latency_ns=args.latency_ns)
        payload["baseline_manifest"] = baseline_manifest(identity, source, scenario)
        # Also verify the ordinary config reader accepts this exact authoring payload.
        authored = scenario_payload(scenario)
        payload["experiment_config"] = {k: v for k, v in authored.items() if k != "model"}
        payload["model_config_sha256"] = stable_hash(authored["model"])
        parsed = scenario_from_dict(authored)
        if scenario_payload(parsed) != authored:
            raise ValueError("scenario config round trip changed the declared experiment")
        if args.dry_run:
            decision = plan_runtime_placement(parsed)
            if not decision.fully_placed:
                raise UnsupportedScenario("placement unavailable: " + str(decision.unplaced))
            payload.update(status="PLANNED", scenario_config=authored)
        else:
            from tools.memory_tier_sweep import validate_observation, scenario_coverage, coverage_check
            observation = execute_scenario(scenario)
            payload.update(status="SIMULATED", observation=observation,
                           checks=validate_observation(observation),
                           coverage=scenario_coverage(args.scenario, observation))
            payload["checks"].append(coverage_check(args.scenario, payload["coverage"]))
        after = source_identity()
        payload["source_stable"] = source == after
        if source != after:
            payload.update(status="INVALIDATED", error="Source changed during this experiment; rerun against stable sources")
        elif not args.dry_run and any(c["status"] != "PASS" for c in payload["checks"]):
            payload["status"] = "FAILED_CHECKS"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        payload.update(status="BLOCKED", error_type=type(exc).__name__, error=str(exc))
    try:
        write_output(args.output, payload)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "validation_status": UNVALIDATED,
                      "output": str(args.output.resolve()), "error": payload.get("error")}, ensure_ascii=False))
    return 0 if payload["status"] in {"PLANNED", "SIMULATED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
