"""Bounded, full-operator HBM/HBF placement and named-model study.

Every timing row uses run_scenario through the shared serving_row collector.
The fixed hardware inventory is hypothetical; it is not a silicon calibration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Version 9 separates clock, directional budgets and fixed stack geometry;
# HBF uses 4 KiB pages, 64 B host transfers and explicit host/cache resources.
# Older execution results cannot establish this contract.
POINT_RESULT_VERSION = 10

from heterollm_sim.config import model_from_dict
from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.ir import (KVCachePolicy, RequestSpec, LinkSpec, PortSpec,
    MTPPolicy, MTPBranchSpec, build_model_graph_from_layer_specs, model_graph_execution_layers)
from heterollm_sim.physical_contract import reconfigure_physical_config, parse_execution_physical_config
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.hbm_timing_profiles import timing_profile_catalog
from heterollm_sim.memory_allocator import AllocationError
from heterollm_sim.model_presets import materialize_model_payload, list_model_presets, get_model_preset
from heterollm_sim.planner import ScenarioValidationError

GIB = 1024 ** 3
ARCHITECTURES = ("A_hbm_only", "B_all_kv_hbf", "C_hot_cold", "D_hbf_weights")
OFFLOAD_BASELINES = ("E_host_dram", "F_nvme", "G_cxl_dram")
DEFAULT_SECTIONS = ("placement", "models", "named_extended", "long_context", "baselines",
                    "slo", "tail_sample", "prefetch", "speculative")


def build_study_scenario(architecture: str, *, model_id: str = "small",
                         prompt: int = 128, batch: int = 1, output: int = 8,
                         request_count: int | None = None,
                         interval_ns: float = 0.0, priority_preemption: bool = False,
                         prefetch_distance: int = 0,
                         mtp_branch: bool = False, speculative_acceptance: float | None = None,
                         hbm_capacity_bytes: int = 512 * GIB,
                         hbf_capacity_bytes: int = 1024 * GIB,
                         hbf_page_read_ns: float | None = None,
                         hbm_timing_profile: str = "hbm3e_reference",
                         hbm_parameter_profile: str | None = None,
                         hbm_timing_overrides: dict | None = None,
                         hbm_controller: dict | None = None,
                         hbm_clock_ns: float | None = None,
                         slo_ttft_ns: float = 10e9, slo_tbt_ns: float = 100e6):
    """Build a real serving scenario, retaining identical device inventory.

priority_preemption is a reproducible pause/resume surrogate: a higher-priority
request interrupts a lower-priority request, which is later resumed. C uses
physical paged swaps; the other policies recompute. It is explicitly not an
agent-session API or a static hot-hit probability.
    """
    if architecture not in ARCHITECTURES + OFFLOAD_BASELINES:
        raise ValueError(f"unknown architecture: {architecture}")
    if hbm_timing_overrides is not None:
        allowed_timings = {field for pack in timing_profile_catalog().values()
                           for field in pack.get("parameters", {})}
        if not isinstance(hbm_timing_overrides, dict):
            raise ValueError("HBM timing overrides must be a dictionary of timing fields")
        unknown = set(hbm_timing_overrides) - allowed_timings
        if unknown:
            raise ValueError("unknown HBM timing fields: " + ", ".join(sorted(unknown)))
    tiered = architecture in {"C_hot_cold", "F_nvme"}
    cold_owner = "hostmem0" if architecture == "E_host_dram" else "hbf0"
    import hbf_hbm_experiments as experiments
    count = batch if request_count is None else request_count
    s = experiments.scenario(hbf=architecture == "B_all_kv_hbf", prompt=prompt,
                             count=count, slots=batch, output=output,
                             hbm_timing_profile=hbm_timing_profile)
    model = s.model
    if model_id != "small":
        payload = materialize_model_payload(model_id)
        # Preserve the preset's real architecture. Context extension changes
        # the declared bound only and is labelled as extrapolation in output.
        graph = payload["graph"]
        native = int(graph["attributes"].get("max_sequence_length", 0))
        if prompt + output > native:
            graph["attributes"]["max_sequence_length"] = prompt + output
            for op in graph["operators"]:
                if op["op_kind"] == "model_input":
                    op["parameters"]["max_sequence_length"] = prompt + output
            payload.setdefault("metadata", {})["study_context_extrapolation"] = {
                "native_max_tokens": native, "requested_tokens": prompt + output,
                "accuracy_claim": False,
            }
        model = model_from_dict(payload)
    layers = model_graph_execution_layers(model.graph)
    if mtp_branch or speculative_acceptance is not None:
        if model_id != "small":
            raise ValueError("The study's synthetic MTP branch may only be attached to its synthetic model")
        model = replace(model,name="synthetic-small-dense-explicit-mtp",graph=build_model_graph_from_layer_specs(
            "synthetic-small-dense-explicit-mtp",layers,vocabulary_size=model.vocabulary_size,
            max_sequence_length=model.max_sequence_length,embedding_weight_bytes=model.embedding_weight_bytes,
            output_weight_bytes=model.output_weight_bytes,
            mtp=MTPBranchSpec(prediction_layers=1,auxiliary_head=True)),
            metadata={**model.metadata,"study_mtp_branch":{
                "kind":"synthetic_prediction_layer_and_auxiliary_head",
                "trained_draft_model":False,"accuracy_claim":False}})
    components = []
    for c in s.hardware.components:
        if c.kind == "hbm" and c.component_id != "hbm0":
            continue
        if c.component_id in {"hbm0", "hbf0"}:
            capacity = hbm_capacity_bytes if c.component_id == "hbm0" else hbf_capacity_bytes
            metadata = {k: v for k, v in c.metadata.items() if k != "memory_service"}
            overrides = {"capacity_bytes": capacity}
            if c.component_id == "hbf0" and hbf_page_read_ns is not None:
                overrides["page_read_ns"] = hbf_page_read_ns
            if c.component_id == "hbm0":
                source = parse_execution_physical_config(metadata["physical_memory_config"])
                # A changed capacity budget selects a covering reference part
                # and stack count; it must not invent extra rows in one part.
                physical = experiments.profiles._hbm_physical_config(
                    capacity_bytes=capacity, read_gb_s=source.directional_bandwidth_gb_s("read"),
                    write_gb_s=source.directional_bandwidth_gb_s("write"),
                    clock_ns=source.command_clock_ns if hbm_clock_ns is None else hbm_clock_ns,
                    interface_input_scope=source.metadata.get("interface_input_scope"),
                    parameter_profile=hbm_parameter_profile,
                    profile_id=hbm_timing_profile)
                if hbm_timing_overrides:
                    physical = reconfigure_physical_config(physical, overrides=hbm_timing_overrides)
                if hbm_controller:
                    allowed = {"address_mapping", "scheduler", "scheduler_window", "queue_scope",
                               "max_outstanding_requests", "interleave_bytes"}
                    unknown = set(hbm_controller) - allowed
                    if unknown:
                        raise ValueError("unknown HBM controller fields: " + ", ".join(sorted(unknown)))
                    physical = reconfigure_physical_config(physical, overrides=hbm_controller)
            else:
                physical = reconfigure_physical_config(metadata["physical_memory_config"],
                    overrides=overrides, expand_geometry=True)
            metadata["physical_memory_config"] = physical
            if c.component_id == "hbf0":
                metadata["access_mode"] = "remote_flash" if tiered else "memory"
                metadata["write_buffer_bytes"] = physical.write_buffer_bytes
            c = replace(c, capacity_bytes=capacity, metadata=metadata)
            if c.component_id == "hbm0":
                interface_gbps = physical.physical_interface_bandwidth_gb_s * 8.0
                c = replace(c, bandwidth_gbps=interface_gbps,
                    ports=tuple(replace(port, bandwidth_gbps=interface_gbps) for port in c.ports))
            if c.component_id == "hbf0" and tiered:
                c = replace(c, cost_profile_id=None)
            if c.component_id == "hbf0" and architecture == "F_nvme":
                physical = resolve_physical_config("study_nvme_baseline",
                    overrides={"capacity_bytes":hbf_capacity_bytes},expand_geometry=True)
                c = replace(c, kind="ssd", bandwidth_gbps=56.,read_bandwidth_gbps=56.,write_bandwidth_gbps=32.,
                    ports=tuple(replace(p,protocol="PCIe",version="5.0",lanes=16,bandwidth_gbps=56.) for p in c.ports),
                    metadata={"physical_memory_config":physical,"memory_service_owner":"far_nvme"})
        if c.component_id == "gpu0" and architecture == "F_nvme":
            far_port = next(l.source_port for l in s.hardware.links if l.target_component=="hbf0")
            c = replace(c, ports=tuple(replace(p,protocol="PCIe",role="root_complex",version="5.0",lanes=16,bandwidth_gbps=56.)
                if p.port_id==far_port else p for p in c.ports))
        components.append(c)
    component_ids = {c.component_id for c in components}
    hardware = replace(s.hardware, components=tuple(components), links=tuple(
        l for l in s.hardware.links
        if l.source_component in component_ids and l.target_component in component_ids))
    hbm_interface_gbps = hardware.get_component("hbm0").bandwidth_gbps
    hbm_links = tuple(replace(link, bandwidth_gbps=hbm_interface_gbps)
        if "hbm0" in (link.source_component, link.target_component) else link
        for link in hardware.links)
    gpu_hbm_ports = {link.source_port if link.source_component == "gpu0" else link.target_port
                    for link in hbm_links if {link.source_component, link.target_component} == {"gpu0", "hbm0"}}
    hardware = replace(hardware, links=hbm_links, components=tuple(
        replace(component, ports=tuple(replace(port, bandwidth_gbps=hbm_interface_gbps)
            if port.port_id in gpu_hbm_ports else port for port in component.ports))
        if component.component_id == "gpu0" else component for component in hardware.components))
    if architecture == "F_nvme":
        hardware = replace(hardware,links=tuple(replace(l,protocol="PCIe",version="5.0",lanes=16,
            bandwidth_gbps=56.,latency_ns=800.) if "hbf0" in (l.source_component,l.target_component) else l
            for l in hardware.links))
    if architecture == "G_cxl_dram":
        hardware=replace(hardware,components=tuple(replace(c,ports=c.ports+(PortSpec(
            port_id="study-cxl",protocol="CXL",role="host" if c.component_id=="gpu0" else "device",version="3.0",lanes=16,
            bandwidth_gbps=512.,payload="coherent_dma"),)) if c.component_id in {"gpu0","hostmem0"}
            else c for c in hardware.components), links=hardware.links+(LinkSpec(
                link_id="study-cxl-dram",source_component="gpu0",source_port="study-cxl",
                target_component="hostmem0",target_port="study-cxl",protocol="CXL",version="3.0",
                lanes=16,bandwidth_gbps=512.,latency_ns=200.,payload="coherent_dma"),))
    dtype = "int8" if model_id == "small" else "bf16"
    kv_owner = ("hbf0" if architecture == "B_all_kv_hbf" else "hostmem0"
                if architecture in {"E_host_dram","G_cxl_dram"} else "hbm0")
    policy = KVCachePolicy(cache_component=kv_owner, dtype=dtype, tokens_per_page=16,
        prefetch_distance=prefetch_distance,
        allocation_policy="eager", preemption_mode="swap" if tiered else "recompute",
        layout_mode="paged_pool" if tiered else "legacy_single",
        pool_components=("hbm0", cold_owner) if tiered else (),
        offload_component=cold_owner if tiered else None)
    weight_owner = "hbf0" if architecture == "D_hbf_weights" else "hbm0"
    weight_map = {"embedding_weights": weight_owner, "lm_head_weights": weight_owner}
    operator_map = {"embedding":"gpu0", "lm_head":"gpu0"}
    for layer in layers:
        kinds = ("attention", "experts", "router") if layer.is_moe else ("attention", "mlp")
        for kind in kinds:
            tensor_kind = "expert" if kind == "experts" else kind
            weight_map[f"{layer.layer_id}.{tensor_kind}_weights"] = weight_owner
            operator_map[f"{layer.layer_id}.{kind}"] = "gpu0"
    metadata = dict(s.placement.metadata)
    if architecture == "D_hbf_weights":
        metadata["llama_backend_memory"] = {"hbf0":{"device_id":"gpu0", "buffer_kind":"HBF", "access":"direct", "priority":1}}
    metadata["control_plane"] = {"policy":{"options":{
        "weight_tensor_targets":weight_map,"operator_targets":operator_map,
        "kv_cache_target":kv_owner,"design_prefill_tokens":prompt,
        "design_decode_batch_size":batch}}}
    requests = tuple(RequestSpec(f"study-{i}", i * interval_ns, prompt, output,
        priority=10 if priority_preemption and i else 0) for i in range(count))
    scheduler = replace(s.workload.scheduler, max_num_seqs=batch,
        max_num_batched_tokens=max(batch * prompt, batch),
        max_num_ubatch_tokens=max(batch * prompt, batch), prefill_chunk_tokens=prompt,
        preemption_enabled=priority_preemption,
        preemption_policy="swap" if tiered else "recompute",
        slo_ttft_ns=slo_ttft_ns, slo_tbt_ns=slo_tbt_ns)
    hbm_component = hardware.get_component("hbm0")
    hbm_config = parse_execution_physical_config(hbm_component.metadata["physical_memory_config"])
    component_profiles = {kind:dict(registry) for kind,registry in s.component_profiles.items()}
    component_profiles["hbm"][hbm_component.cost_profile_id] = replace(
        s.resolve_component_profile("hbm0"),
        bandwidth_gb_s=hbm_config.physical_interface_bandwidth_gb_s,
        read_bandwidth_gb_s=hbm_config.directional_bandwidth_gb_s("read"),
        write_bandwidth_gb_s=hbm_config.directional_bandwidth_gb_s("write"),
        parallel_lanes=hbm_config.lane_count,
        transaction_bytes=hbm_config.burst_bytes,
        max_outstanding_requests=hbm_config.max_outstanding_requests)
    scenario = replace(s, name=f"study-{architecture}-{model_id}", hardware=hardware, model=model,
        component_profiles=component_profiles,
        placement=replace(s.placement, model_name=model.name, tensor_to_component={},
            tensor_bytes={}, metadata=metadata,
            parallel=replace(s.placement.parallel, layer_to_stage={l.layer_id:0 for l in layers}),
            kv_policy=policy),
        workload=replace(s.workload, requests=requests, request_count=count, scheduler=scheduler,
            mtp=MTPPolicy(candidate_tokens=4,acceptance_model="expected_prefix",
                acceptance_rate=speculative_acceptance) if speculative_acceptance is not None else None))
    # The control plane exposes primitive requirements separately from GEMMs.
    # Pin all of them, including normalization/reduction/activation, so adding
    # a CXL route cannot silently change this fixed-GPU placement comparison.
    requirements = _derive_requirements(scenario)
    options = metadata["control_plane"]["policy"]["options"]
    options["operator_targets"] = {r.mapping_key:"gpu0" for r in requirements
                                   if r.mapping_key and not r.state_tensor}
    options["weight_tensor_targets"] = {r.tensor_id:weight_owner for r in requirements
                                        if r.tensor_id and r.tensor_bytes>0 and not r.state_tensor}
    return scenario


def physical_capacity_failure(error):
    """Describe only a real first-fit capacity failure; other allocator bugs raise.

    The failed request size is not the node's peak requirement. When the local
    traceback is available, retain occupancy and fragmentation at that failure
    without changing allocator or Core state.
    """
    if not isinstance(error, AllocationError):
        return None
    match = re.fullmatch(r"unable to allocate (\d+) bytes in capacity (\d+)(?: \(.* arena \[\d+, \d+\)\))?", str(error))
    if match is None:
        return None
    detail = {"exception_type": "AllocationError", "message": str(error),
              "requested_allocation_bytes": int(match[1]), "capacity_bytes": int(match[2]),
              "classification": "physical_allocator_no_contiguous_extent",
              "failure_stage": "physical_allocation", "physical_owner": None,
              "occupied_bytes_at_failure": None, "largest_free_extent_bytes": None,
              "same_layout_minimum_capacity_bytes": None,
              "scope": "Physical allocator feasibility failure, not token throughput. The failing request size alone is not total peak demand; fragmentation may contribute. The same-layout bound only extends the current non-moving layout and is not a minimum hardware capacity."}
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        local = frame.f_locals
        if frame.f_code.co_name == "execute_cost_schedule":
            detail["failure_stage"] = "physical_planning_preview"
        if frame.f_code.co_name == "_register_physical_allocation":
            detail["physical_owner"] = local.get("owner")
            declaration = local.get("declaration", {})
            detail["allocation"] = {k: declaration[k] for k in
                ("buffer_id", "size_bytes", "allocation_size_bytes", "generation", "allocation_generation")
                if k in declaration}
        if frame.f_code.co_name == "_allocation_exhausted":
            candidates = [a for a in local["failure_allocations"] if a.alias_of is None]
            lower, upper = local["lower"], local["upper"]
            extent = [(max(lower,a.base_address), min(upper,a.base_address+a.size_bytes))
                      for a in candidates if a.base_address < upper and a.base_address+a.size_bytes > lower]
            used = sum(end-start for start,end in extent)
            cursor, largest = lower, 0
            alignment = local["self"].alignment_bytes
            for start,end in sorted(extent):
                cursor = ((cursor+alignment-1)//alignment)*alignment
                largest = max(largest,start-cursor)
                cursor = max(cursor,end)
            cursor = ((cursor+alignment-1)//alignment)*alignment
            largest = max(largest,upper-cursor)
            detail.update(occupied_bytes_at_failure=used, free_bytes_at_failure=upper-lower-used,
                largest_free_extent_bytes=largest, arena_start_bytes=lower, arena_end_bytes=upper,
                    same_layout_minimum_capacity_bytes=int(cursor)+int(local["size"]),
                compacted_occupancy_plus_request_bytes=used+int(local["size"]),
                occupancy_observation="non-alias extents in the failed allocator arena, captured before transaction rollback; not a measured hardware peak")
        trace = trace.tb_next
    return detail


def run_case(architecture: str, *, row_id: str, result_observer=None, **parameters) -> dict[str, Any]:
    import hbf_hbm_experiments as experiments
    s = build_study_scenario(architecture, **parameters)
    try:
        row = experiments.serving_row(s, row_id, placement=architecture,
                                      result_observer=result_observer)
    except ScenarioValidationError as error:
        return {"id":row_id, "placement":architecture, "mode":"configuration_rejected",
                "parameters":parameters, "reason":str(error), "node_tps":None,
                "goodput_tps":None, "executed_serving":False}
    except AllocationError as error:
        failure = physical_capacity_failure(error)
        if failure is None:
            raise
        return {"id":row_id,"placement":architecture,"mode":"physical_capacity_rejected",
                "model":s.model.name,"parameters":parameters,"reason":str(error),
                "node_tps":None,"goodput_tps":None,"executed_serving":False,
                "attempted_serving":True,"capacity_failure":failure,
                "fixed_inventory":{c.component_id:{"kind":c.kind,"capacity_bytes":c.capacity_bytes}
                    for c in s.hardware.components if c.is_storage},
                "model_structure":{"model_id":parameters.get("model_id","small"),
                    "layers":s.model.num_layers,"weight_bytes":s.model.total_declared_weight_bytes,
                    "context_extrapolation":s.model.metadata.get("study_context_extrapolation")}}
    row["parameters"] = parameters
    row["executed_serving"] = True
    row["model_structure"] = {"model_id":parameters.get("model_id", "small"),
        "layers":s.model.num_layers, "weight_bytes":s.model.total_declared_weight_bytes,
        "context_extrapolation":s.model.metadata.get("study_context_extrapolation"),
        "mtp_branch":s.model.metadata.get("study_mtp_branch"),
        "mtp_descriptor_count":len(s.model._execution_view.mtp_descriptors)}
    if parameters.get("model_id","small") != "small":
        preset=get_model_preset(parameters["model_id"])
        row["model_structure"]["preset_support_level"]=preset.support_level
        row["model_structure"]["preset_limitations"]=list(preset.limitations)
        row["model_structure"]["preset_notes"]=preset.notes
    row["fixed_inventory"] = {c.component_id: {"kind":c.kind,"capacity_bytes":c.capacity_bytes}
                              for c in s.hardware.components if c.is_storage}
    row["placement_contract"] = {"weights":"hbf0" if architecture == "D_hbf_weights" else "hbm0",
        "kv":s.placement.kv_policy.cache_component,"layout":s.placement.kv_policy.layout_mode,
        "cold":s.placement.kv_policy.offload_component}
    resolved = row["resolved_placement"]["tensor_to_component"]
    weight_targets = {key:value for key,value in resolved.items() if key.endswith(("_weights",".weights"))}
    row["placement_verification"] = {
        "all_resolved_weights_on_requested_tier":bool(weight_targets) and all(
            value==row["placement_contract"]["weights"] for value in weight_targets.values()),
        "resolved_kv_on_requested_tier":resolved.get("kv_cache")==row["placement_contract"]["kv"],
        "resolved_weight_tensor_count":len(weight_targets),
        "all_resolved_model_ops_on_gpu":bool(row["resolved_placement"].get("op_to_component")) and all(
            value=="gpu0" for value in row["resolved_placement"].get("op_to_component",{}).values()),
    }
    return row


def _qualified(row):
    return (row.get("executed_serving") and row.get("completed_requests") == row.get("request_count")
            and row.get("qualified_requests") == row.get("request_count"))


class StudyCasesError(RuntimeError):
    """One or more independent points failed; successful points are retained."""

    def __init__(self, failures, completed_rows):
        self.failures = tuple(failures)
        self.completed_rows = tuple(completed_rows)
        super().__init__("serving study point failures: " + "; ".join(
            f"{row_id}: {type(error).__name__}: {error}" for row_id, error in failures))


def study_cases(cases, *, workers=1, progress=True, point_cache_dir=None, resume=False):
    """Run independent (architecture, row_id, parameters) points in input order.

    Workers only simulate. The parent writes each completed point immediately,
    including when another point fails. Exceptions are raised after harvesting
    the remaining results, so a partial batch can never be reported as passed.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    cases = tuple(cases)
    identifiers = [row_id for _, row_id, _ in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("independent study point IDs must be unique")
    results = [None] * len(cases)
    pending = []
    failures = []
    cache_dir = Path(point_cache_dir) if point_cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for index, (architecture, row_id, parameters) in enumerate(cases):
        path = cache_dir / (row_id.replace('/', '__') + '.json') if cache_dir is not None else None
        cached = None
        if resume and path is not None and path.exists():
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                # Interrupted or unreadable point output is recomputed.
                cached = None
        if (isinstance(cached, dict) and cached.get('point_result_version') == POINT_RESULT_VERSION
                and cached.get('id') == row_id and cached.get('placement') == architecture
                and cached.get('parameters') == parameters and (cached.get('executed_serving')
                    or cached.get('mode') == 'physical_capacity_rejected' and cached.get('capacity_failure'))):
            results[index] = cached
            if progress:
                print('[09_serving_study] resume ' + row_id, flush=True)
        else:
            pending.append((index, architecture, row_id, parameters, path))

    def retain(index, row, path):
        row['point_result_version'] = POINT_RESULT_VERSION
        if path is not None:
            # Atomic replacement leaves the last complete point readable if
            # the process is interrupted while writing its successor.
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(path)
        results[index] = row

    if workers == 1:
        for index, architecture, row_id, parameters, path in pending:
            if progress:
                print('[09_serving_study] ' + row_id, flush=True)
            try:
                retain(index, run_case(architecture, row_id=row_id, **parameters), path)
            except Exception as error:
                failures.append((row_id, error))
    elif pending:
        # Spawn avoids inheriting mutable physical-runtime state on platforms
        # where the multiprocessing default would otherwise use fork.
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {}
            for index, architecture, row_id, parameters, path in pending:
                if progress:
                    print('[09_serving_study] queued ' + row_id, flush=True)
                try:
                    future = executor.submit(run_case, architecture, row_id=row_id, **parameters)
                    futures[future] = (index, row_id, path)
                except Exception as error:
                    failures.append((row_id, error))
            for future in as_completed(futures):
                index, row_id, path = futures[future]
                try:
                    retain(index, future.result(), path)
                    if progress:
                        print('[09_serving_study] completed ' + row_id, flush=True)
                except Exception as error:
                    failures.append((row_id, error))
    completed = [row for row in results if row is not None]
    if failures:
        raise StudyCasesError(failures, completed) from failures[0][1]
    return completed


def run_serving_study(*, sections=DEFAULT_SECTIONS, progress=True,
                      point_cache_dir=None, resume=False, workers=1) -> dict[str, Any]:
    """Run a bounded matrix; each section can also be persisted independently."""
    allowed=set(DEFAULT_SECTIONS)
    if not sections or not set(sections)<=allowed:
        raise ValueError("sections must be a nonempty subset of " + ", ".join(sorted(allowed)))
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    def study_case(architecture, *, row_id, **parameters):
        return study_cases(((architecture, row_id, parameters),), workers=1,
            progress=progress, point_cache_dir=point_cache_dir, resume=resume)[0]
    rows = []
    if "placement" in sections:
        for architecture in ARCHITECTURES:
            rows.append(study_case(architecture, row_id=f"placement/{architecture}",
                prompt=128, batch=1, output=12, request_count=2,
                interval_ns=50_000., priority_preemption=True))
    if "baselines" in sections:
        for architecture in OFFLOAD_BASELINES:
            rows.append(study_case(architecture, row_id=f"baseline/{architecture}",
                prompt=128, batch=1, output=12, request_count=2,
                interval_ns=50_000., priority_preemption=True))
    if "prefetch" in sections:
        for distance in (0,16,32):
            rows.append(study_case("C_hot_cold",row_id=f"prefetch/C_hot_cold/{distance}",
                prompt=128,batch=1,output=12,request_count=2,interval_ns=50_000.,
                priority_preemption=True,prefetch_distance=distance))
        for distance in (0,16):
            rows.append(study_case("C_hot_cold",row_id=f"prefetch-stress/C_hot_cold/{distance}",
                prompt=128,batch=1,output=12,request_count=2,interval_ns=50_000.,
                priority_preemption=True,prefetch_distance=distance,hbf_page_read_ns=25_000.))
        for distance in (0,16):
            rows.append(study_case("C_hot_cold",row_id=f"prefetch-stress/C_hot_cold/75000ns/{distance}",
                prompt=128,batch=1,output=12,request_count=2,interval_ns=50_000.,
                priority_preemption=True,prefetch_distance=distance,hbf_page_read_ns=75_000.))
    named_cases = []
    if "models" in sections or "long_context" in sections:
        for model in ("llama3_1-8b", "llama3_1-70b"):
            prompts = ((2048,) if "models" in sections else ()) + ((131072,1048576) if "long_context" in sections else ())
            for prompt in prompts:
                for architecture in ("A_hbm_only", "B_all_kv_hbf", "D_hbf_weights"):
                    named_cases.append((architecture, f"model/{model}/{prompt}/{architecture}",
                        dict(model_id=model, prompt=prompt, batch=1, output=32 if prompt==2048 else 4)))
    if "named_extended" in sections:
        for model in ("llama3_1-405b","qwen3-30b-a3b","qwen3-235b-a22b"):
            hbm_capacity=(1024 if model=="llama3_1-405b" else 512)*GIB
            for architecture in ("A_hbm_only","D_hbf_weights"):
                named_cases.append((architecture, f"named-extended/{model}/{architecture}",
                    dict(model_id=model,prompt=2048,batch=1,output=4,hbm_capacity_bytes=hbm_capacity)))
    if "long_context" in sections:
        # The physical allocator exposes the 512 GiB long-context limit.
        # Keep all three original-inventory observations above, and give all
        # A/B/D comparison points the same explicitly larger inventory.
        for architecture in ("A_hbm_only", "B_all_kv_hbf", "D_hbf_weights"):
            named_cases.append((architecture,f"model-capacity/llama3_1-70b/1048576/1tib/{architecture}",
                dict(model_id="llama3_1-70b",prompt=1048576,batch=1,output=4,
                     hbm_capacity_bytes=1024*GIB)))
    for row in study_cases(named_cases, workers=workers, progress=progress,
                           point_cache_dir=point_cache_dir, resume=resume):
        if row['id'].startswith('named-extended/'):
            hbm_capacity=row['parameters']['hbm_capacity_bytes']
            model=row['parameters']['model_id']
            row["inventory_comparison_scope"]={"paired_architectures":["A_hbm_only","D_hbf_weights"],
                    "hbm_capacity_bytes":hbm_capacity,"hbf_capacity_bytes":1024*GIB,
                    "capacity_expansion_reason":"405B BF16 weights exceed the base 512 GiB HBM; both paired architectures receive 1 TiB HBM" if model=="llama3_1-405b" else None,
                    "scope":"same physical capacity within each model pair; not equal package area or a production single-GPU topology"}
        if row['id'].startswith('model/llama3_1-70b/1048576/'):
            row["inventory_role"] = "capacity_control_512gib"
            row["inventory_comparison_scope"] = {"hbm_capacity_bytes":512*GIB,
                "hbf_capacity_bytes":1024*GIB,
                "scope":"Original inventory capacity control. Failed A/D have no throughput; the completed B is not paired against expanded-inventory A/D."}
        if row['id'].startswith('model-capacity/llama3_1-70b/1048576/1tib/'):
            row["inventory_role"] = "capacity_expanded_comparison_1tib"
            row["inventory_comparison_scope"] = {"paired_architectures":["A_hbm_only","B_all_kv_hbf","D_hbf_weights"],
                "hbm_capacity_bytes":1024*GIB,"hbf_capacity_bytes":1024*GIB,
                "capacity_expansion_reason":"The original 512 GiB physical allocator cannot admit a 56 GiB long-context buffer for A/D. Every compared architecture receives the same 1 TiB HBM inventory; the original failure remains a separate capacity control."}
        rows.append(row)
    if "speculative" in sections:
        for architecture in ("A_hbm_only","B_all_kv_hbf"):
            for acceptance in (None,0.,0.5,0.95):
                row=study_case(architecture,row_id=f"speculative/{architecture}/{acceptance}",
                    prompt=128,batch=4,output=32,mtp_branch=True,speculative_acceptance=acceptance)
                row["speculative_contract"]={"enabled":acceptance is not None,
                    "maximum_draft_tokens":4,"per_position_continuation_probability":acceptance,
                    "acceptance_model":"deterministic cumulative expected_prefix",
                    "scope":"Explicit synthetic prediction-layer and auxiliary-head GPU execution, verifier execution, acceptance and rejected draft KV handling; no trained draft or task-quality claim"}
                rows.append(row)
    if "tail_sample" in sections:
        for architecture in ("A_hbm_only","B_all_kv_hbf"):
            row=study_case(architecture,row_id=f"tail/{architecture}",prompt=128,batch=64,output=8,
                request_count=1024,interval_ns=10_000.,slo_ttft_ns=5e6,slo_tbt_ns=1e6)
            row["arrival_trace"]={"kind":"deterministic_uniform","count":1024,"interval_ns":10_000.,
                "semantics":"finite offered load; no random seeds, production trace or steady-state assumption"}
            rows.append(row)
    boundaries = []
    if "slo" in sections:
        # All runs use 32 requests, finite horizon, and a fixed 5ms/1ms SLO.
        # Search bounds are explicit; an unbracketed result remains a bound.
        for architecture in ("A_hbm_only", "B_all_kv_hbf"):
            samples = []
            for interval in (0., 10_000., 100_000., 1_000_000.):
                row = study_case(architecture, row_id=f"slo/{architecture}/{interval}",
                    prompt=128, batch=4, output=8, request_count=32,
                    interval_ns=interval, slo_ttft_ns=5e6, slo_tbt_ns=1e6)
                row["all_requests_meet_slo"] = bool(_qualified(row))
                rows.append(row); samples.append(row)
            feasible = [r["parameters"]["interval_ns"] for r in samples if _qualified(r)]
            lower = max((r["parameters"]["interval_ns"] for r in samples if not _qualified(r)
                         and feasible and r["parameters"]["interval_ns"] < min(feasible)), default=None)
            upper = min(feasible, default=None)
            if lower is not None and upper is not None:
                # Refinement is conditional on the observed bracket. We retain
                # every observation; this is no proof of global monotonicity.
                for _ in range(4):
                    interval=(lower+upper)/2
                    row=study_case(architecture,row_id=f"slo-refine/{architecture}/{interval}",
                        prompt=128,batch=4,output=8,request_count=32,interval_ns=interval,
                        slo_ttft_ns=5e6,slo_tbt_ns=1e6)
                    row["all_requests_meet_slo"]=bool(_qualified(row))
                    rows.append(row); samples.append(row)
                    if _qualified(row): upper=interval
                    else: lower=interval
            # Keep the original bounded study's B refinement inputs when a
            # changed physical backend produces a different adaptive bracket.
            # These remain individual finite observations, not a fitted curve.
            reference_intervals = (55_000.,77_500.,88_750.,94_375.) if architecture=="B_all_kv_hbf" else ()
            observed_intervals = {r["parameters"]["interval_ns"] for r in samples}
            for interval in reference_intervals:
                if interval in observed_intervals:
                    continue
                row=study_case(architecture,row_id=f"slo-refine/{architecture}/{interval}",
                    prompt=128,batch=4,output=8,request_count=32,interval_ns=interval,
                    slo_ttft_ns=5e6,slo_tbt_ns=1e6)
                row["all_requests_meet_slo"]=bool(_qualified(row))
                rows.append(row); samples.append(row)
            upper=min((r["parameters"]["interval_ns"] for r in samples if _qualified(r)),default=None)
            lower=max((r["parameters"]["interval_ns"] for r in samples if not _qualified(r)
                       and upper is not None and r["parameters"]["interval_ns"]<upper),default=None)
            boundaries.append({"architecture":architecture,
                "kind":"arrival_interval",
                "tested_arrival_intervals_ns":[r["parameters"]["interval_ns"] for r in samples],
                "minimum_tested_feasible_interval_ns":min((r["parameters"]["interval_ns"] for r in samples if _qualified(r)),default=None),
                "last_observed_infeasible_interval_ns":lower,
                "first_observed_feasible_interval_ns":upper,
                "refined_bracket_width_ns":upper-lower if lower is not None and upper is not None else None,
                "scope":"finite 32-request trace; not a proven sustainable arrival rate",
                "feasible_and_infeasible_observed":any(_qualified(r) for r in samples) and any(not _qualified(r) for r in samples)})
            batch_samples=[]
            for concurrency in (1,4,16,64):
                row=study_case(architecture,row_id=f"slo-batch/{architecture}/{concurrency}",
                    prompt=2048,batch=concurrency,output=8,slo_ttft_ns=5e6,slo_tbt_ns=1e6)
                row["all_requests_meet_slo"]=bool(_qualified(row))
                rows.append(row);batch_samples.append(row)
            boundaries.append({"architecture":architecture,"kind":"batch",
                "tested_batches":[1,4,16,64],
                "actual_batches":[r.get("max_observed_batch_requests") for r in batch_samples],
                "maximum_tested_feasible_batch":max((r["parameters"]["batch"] for r in batch_samples if _qualified(r)),default=None),
                "minimum_tested_infeasible_batch":min((r["parameters"]["batch"] for r in batch_samples if not _qualified(r)),default=None),
                "scope":"finite simultaneous cohort; values beyond 64 untested; no unbounded maximum claim"})
    placement_rows = [r for r in rows if r["id"].startswith("placement/")]
    baseline_rows = [r for r in rows if r["id"].startswith("baseline/")]
    prefetch_rows = [r for r in rows if r["id"].startswith(("prefetch/","prefetch-stress/"))]
    speculative_rows=[r for r in rows if r["id"].startswith("speculative/")]
    extended_rows=[r for r in rows if r["id"].startswith("named-extended/")]
    capacity_controls=[r for r in rows if r.get("inventory_role")=="capacity_control_512gib"]
    expanded=[r for r in rows if r.get("inventory_role")=="capacity_expanded_comparison_1tib"]
    checks = {"at_least_one_full_operator_run":any(r["executed_serving"] for r in rows),
        "all_requested_points_accounted_for":all(r["executed_serving"] or
            r in capacity_controls and r.get("mode")=="physical_capacity_rejected"
            and r.get("capacity_failure",{}).get("capacity_bytes")==512*GIB for r in rows),
        "all_performance_points_executed":all(r["executed_serving"] for r in rows if r not in capacity_controls),
        "all_completed_points_have_live_hbm_core_evidence":all(
            r.get("core_execution_audit",{}).get("valid_for_committed_byte_totals") and
            any(o.get("core")=="DramCore" and o.get("memory_kind")=="HBM" and o.get("transactions",0)>0
                for o in r["core_execution_audit"].get("owners",())) for r in rows if r["executed_serving"]),
        "all_executed_requests_terminal": all(r.get("unfinished_requests",0)==0 for r in rows),
        "all_executed_requests_complete":all(r.get("completed_requests")==r.get("request_count") for r in rows if r["executed_serving"]),
        "every_rejection_has_reason": all(r.get("reason") for r in rows if not r["executed_serving"]),
        "resolved_placement_matches_contract":all(r["placement_verification"]["all_resolved_weights_on_requested_tier"] and r["placement_verification"]["resolved_kv_on_requested_tier"] and r["placement_verification"]["all_resolved_model_ops_on_gpu"] for r in rows if r["executed_serving"]),
        "serving_rows_have_positive_tps": all(r["node_tps"]>0 for r in rows if r["executed_serving"])}
    if "long_context" in sections:
        checks.update(original_512gib_capacity_controls_retained=len(capacity_controls)==3,
            expanded_1tib_comparison_all_execute=len(expanded)==3 and all(r["executed_serving"] for r in expanded),
            expanded_1tib_comparison_same_inventory=len(expanded)==3 and all(
                r.get("fixed_inventory")==expanded[0].get("fixed_inventory") for r in expanded),
            capacity_failures_have_physical_allocator_evidence=all(
                r.get("capacity_failure",{}).get("exception_type")=="AllocationError" and
                r["capacity_failure"].get("requested_allocation_bytes",0)>0 and r["node_tps"] is None
                for r in capacity_controls if not r["executed_serving"]))
    if baseline_rows:
        checks["all_baselines_execute"] = all(r["executed_serving"] for r in baseline_rows)
        checks["host_and_cxl_really_place_kv_in_dram"] = all(r["kv"]["peak_bytes_by_component"].get("hostmem0",0)>0
            for r in baseline_rows if r["placement"] in {"E_host_dram","G_cxl_dram"} and r["executed_serving"])
    if prefetch_rows:
        checks["prefetch_nonzero_completed_and_used"] = all(r["kv"].get("prefetch_completed_bytes",0)>0
            and r["kv"].get("prefetch_useful_bytes",0)>0 for r in prefetch_rows if r["parameters"]["prefetch_distance"]>0)
        checks["prefetch_zero_keeps_prefetch_disabled"] = all(r["kv"].get("prefetch_completed_bytes",0)==0
            for r in prefetch_rows if r["parameters"]["prefetch_distance"]==0)
        checks["prefetch_never_consumed_before_copy_complete"] = all(
            r["prefetch_overlap"]["premature_consumer_overlap_ns"]==0 for r in prefetch_rows)
        for row in prefetch_rows:
            row["prefetch_observation"]={
                "overlaps_independent_gpu_operator":row["prefetch_overlap"]["operator_overlap_ns"]>0,
                "overlaps_independent_decode_operator":row["prefetch_overlap"]["decode_operator_overlap_ns"]>0,
                "interpretation":"Copy completion/usefulness does not establish hidden latency; use realized operator overlap and end-to-end baseline comparison."}
    if placement_rows:
        checks.update({"all_four_placements_executed":len(placement_rows)==4 and all(r["executed_serving"] for r in placement_rows),
            "common_memory_inventory":all(r.get("fixed_inventory")==placement_rows[0].get("fixed_inventory") for r in placement_rows),
            "cold_policy_really_swaps_and_restores":all(r["kv"].get("offload_bytes",0)>0 and r["kv"].get("swap_in_bytes",0)>0 for r in placement_rows if r["placement"]=="C_hot_cold"),
            "D_hbf_weights_read_and_not_written":all(r["resources"]["storage_traffic"].get("logical_read_bytes",0)>0 and r["resources"]["storage_traffic"].get("logical_write_bytes",0)==0 and r["kv"]["peak_bytes_by_component"].get("hbm0",0)>0 for r in placement_rows if r["placement"]=="D_hbf_weights"),
            "B_hbf_kv_really_read_and_written":all(r["resources"]["storage_traffic"].get("logical_read_bytes",0)>0 and r["resources"]["storage_traffic"].get("logical_write_bytes",0)>0 and r["kv"]["peak_bytes_by_component"].get("hbf0",0)>0 for r in placement_rows if r["placement"]=="B_all_kv_hbf")})
    if speculative_rows:
        checks["speculative_same_model_and_memory_budget"]=all(
            r.get("model_structure")==speculative_rows[0].get("model_structure")
            and r.get("fixed_inventory")==speculative_rows[0].get("fixed_inventory") for r in speculative_rows)
        checks["speculative_enabled_really_drafts"]=all(r["speculative"]["drafted_tokens"]>0
            and r["model_structure"]["mtp_descriptor_count"]==2 for r in speculative_rows if r["executed_serving"] and r["speculative_contract"]["enabled"])
        checks["speculative_accept_reject_conserve_drafts"]=all(r["speculative"]["drafted_tokens"]==
            r["speculative"]["accepted_draft_tokens"]+r["speculative"]["rejected_draft_tokens"] for r in speculative_rows if r["executed_serving"])
        checks["speculative_counts_reconcile_to_commit_events"]=all(
            all(r["speculative"][total]==sum(int(event.get(detail,0)) for event in r["speculative"].get("rounds",()))
                for total,detail in (("drafted_tokens","draft_tokens"),("accepted_draft_tokens","accepted_draft_tokens"),
                                     ("rejected_draft_tokens","rejected_draft_tokens")))
            for r in speculative_rows if r["executed_serving"])
        checks["zero_acceptance_rejects_all_drafts"]=all(r["speculative"]["accepted_draft_tokens"]==0
            and r["speculative"]["rejected_draft_tokens"]>0 for r in speculative_rows
            if r["executed_serving"] and r["parameters"]["speculative_acceptance"]==0.)
        checks["positive_acceptance_exercises_accept_and_reject"]=all(r["speculative"]["accepted_draft_tokens"]>0
            and r["speculative"]["rejected_draft_tokens"]>0 for r in speculative_rows
            if r["executed_serving"] and r["parameters"]["speculative_acceptance"] in (0.5,0.95))
        checks["disabled_speculation_has_no_drafts"]=all(r["speculative"]["drafted_tokens"]==0
            for r in speculative_rows if r["executed_serving"] and not r["speculative_contract"]["enabled"])
    if extended_rows:
        paired_models={r["parameters"]["model_id"] for r in extended_rows}
        checks["extended_models_have_same_capacity_within_pair"]=all(
            len(pair:= [r for r in extended_rows if r["parameters"]["model_id"]==model])==2
            and pair[0].get("fixed_inventory")==pair[1].get("fixed_inventory") for model in paired_models)
    target_models=["Llama 3.1 8B","Llama 3.1 70B","Llama 3.1 405B","Qwen3-30B-A3B","Qwen3-235B-A22B",
        "Qwen3-Coder","DeepSeek-V3/V4","Llama4 Scout/Maverick","1T MoE"]
    executed_names={r.get("model") for r in rows if r["executed_serving"]}
    catalog=list_model_presets()
    requested_preset_families={
        "Qwen3-Coder":("qwen3-coder-30b-a3b",),
        "DeepSeek-V3/V4":("deepseek-v3","deepseek-v4-flash-0731"),
        "Llama4 Scout/Maverick":("llama4-scout","llama4-maverick"),
        "1T MoE":("synthetic-1t-gqa-moe",),
    }
    materializable_ids={p["id"] for p in catalog if p["coverage"]!="metadata_only"}
    coverage={"scope":"group09_only; group17 reports additional full serving and structural probes",
        "sections":list(sections),"full_operator_runs":sum(r["executed_serving"] for r in rows),
        "configuration_rejections":sum(r.get("mode")=="configuration_rejected" for r in rows),
        "physical_capacity_rejections":sum(r.get("mode")=="physical_capacity_rejected" for r in rows),
        "executed_model_names":sorted(x for x in executed_names if x),
        "unexecuted_requested_model_families":[m for m in target_models if m not in executed_names],
        "available_model_presets":[{"name":p["name"],"preset_id":p["id"],"support_level":p["support_level"]} for p in catalog],
        "missing_requested_preset_families":[{"family":name,"reason":"one or more requested variants lack a materializable full-serving preset; metadata-only entries use explicitly bounded structural probes in group17"}
            for name,ids in requested_preset_families.items() if not set(ids)<=materializable_ids],
        "points":[{"id":r["id"],"placement":r["placement"],"model":r.get("model"),
            "context_tokens":r.get("prompt_tokens"),"requested_batch":r.get("requested_concurrency"),
            "actual_batch":r.get("max_observed_batch_requests"),"executed_serving":r["executed_serving"]} for r in rows]}
    passed=all(checks.values())
    return {"id":"09_serving_study", "title":"完整算子放置与命名模型研究",
        "status":"passed" if passed else "failed", "rows":rows,
        "checks":checks, "slo_boundaries":boundaries,"coverage":coverage,
        "capability":{"supported":["完整GPU/算子/服务调度时间线", "显式权重与KV放置", "有限轨迹SLO可行点"] if passed else [],
            "unsupported":["真实器件校准", "稳态最大到达率证明", "等物理封装面积/成本", "1M语义准确率"]},
        "limitations":["Hardware is hypothetical: the base inventory is one GPU, 512 GiB HBM and 1 TiB HBF; 405B pairs explicitly expand HBM to 1 TiB. Unused capacity is retained in each comparison budget.",
            "70B/1M retains its original 512 GiB HBM capacity controls including physical contiguous-allocation failures, and adds a separate A/B/D comparison with 1 TiB HBM for every architecture. The non-moving first-fit allocator may reject an extent despite enough total free bytes because free space is fragmented. This is not proof that the model's theoretical aggregate demand exceeds 512 GiB, nor that 1 TiB is the minimum hardware capacity. Rejections have no TPS and must not be paired against a successful run with different capacity.",
            "C uses priority preemption, physical KV swap and resume, not an agent pause/resume API.",
            "Named model architecture presets retain their true layers/weights; GPU compute remains an uncalibrated analytical model.",
            "Contexts beyond each preset native limit are architecture extrapolation stress tests, without model quality claims.",
            "E uses host DRAM over the declared PCIe/DDR path; G adds a local CXL-attached DRAM path. Neither is a rack-scale shared multi-GPU memory experiment.",
            "F models cold KV on a hypothetical 7 GB/s read, 4 GB/s write NVMe tier with 50 us read and 800 us program; component hbf0 is replaced by kind=ssd for this baseline.",
            "Migration accounting is reported independently from operator storage_traffic; it must not be interpreted as zero migration when storage task_count is zero.",
            "Prefetch overlap uses realized GPU operator envelopes, not whole batches or host setup. Zero overlap remains a negative finding; no latency-hiding success is inferred from useful bytes alone.",
            "Extended named-model pairs use the repository presets without fabricated missing models. Llama 405B pairs expand HBM to 1 TiB for both architectures; Qwen MoE presets retain their declared analytical approximation limitations.",
            "Speculative decoding uses a synthetic explicit MTP branch and deterministic expected-prefix acceptance; probability parameters are not empirical model acceptance or quality measurements.",
            "8B/70B 2K rows generate 32 output tokens; extended named-model 2K and long-context rows generate only four and are prefill-dominated. These finite traces do not establish production tail latency or stable service capacity."]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections",default=",".join(DEFAULT_SECTIONS))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--point-cache-dir",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    report=run_serving_study(sections=tuple(args.sections.split(",")),workers=args.workers,
        point_cache_dir=args.point_cache_dir,resume=args.resume)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"rows":len(report["rows"])}))
