"""Bounded physical placement controls for the Exploring HBF comparison.

This is a separate experiment, never a scan-queue entry point. It executes
short native-model requests on the existing B200/operator/memory runtime.
Paper memory budgets do not turn B200/Qwen into Rubin/Llama or closed-loop
decode serving. No full-length detailed reference is rerun or extrapolated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from b200_hbm_hbf_scan_config import build_scan_scenario, _hardware_and_profiles
from b200_hbm_hbf_scan import json_default, scenario_gc_thresholds
from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.ir import RequestSpec
from heterollm_sim.reporting import run_scenario, _sum_batch_storage_traffic, _sum_batch_dram_traffic
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.serving import RequestStatus
from heterollm_sim.simulation_options import SimulationConfig


CASES = {
    "scan_hbm": "现有 HBM3E 基线",
    "scan_hbf": "现有全 HBF 部署",
    "scan_hbf_hbm_intermediates": "现有 HBF 介质，激活与临时区放 HBM",
    "paper_hbm4": "论文 HBM4 容量与带宽，B200/Qwen 控制",
    "paper_hbf": "论文 HBF 混合存储参数，B200/Qwen 控制",
    "paper_conv": "论文 CONV 混合存储参数，B200/Qwen 控制",
}


def _set_memory(scenario, component_id, physical):
    """Rebind physical, interface and cost-model inputs as one authored unit."""
    is_hbm = component_id == "hbm0"
    if is_hbm:
        shared = physical.physical_interface_bandwidth_gb_s
        read = physical.directional_bandwidth_gb_s("read")
        write = physical.directional_bandwidth_gb_s("write")
        lanes, quantum = physical.lane_count, physical.burst_bytes
        read_latency, write_latency = physical.read_latency_ns, physical.write_latency_ns
    else:
        shared, read, write = (physical.host_bandwidth_gb_s,
            physical.host_read_bandwidth_gb_s, physical.host_write_bandwidth_gb_s)
        lanes, quantum = physical.array_units, physical.host_granularity_bytes
        read_latency, write_latency = physical.page_read_ns, physical.page_program_ns
    source = scenario.hardware.get_component(component_id)
    owner = component_id + (".hbm_fabric" if is_hbm else ".memory")
    outstanding = physical.max_outstanding_requests * (
        physical.host_channels if physical.queue_scope == "host_channel" and not is_hbm else 1)
    profile = replace(scenario.resolve_component_profile(source), bandwidth_gb_s=shared,
        read_bandwidth_gb_s=read, write_bandwidth_gb_s=write,
        read_latency_ns=read_latency, write_latency_ns=write_latency,
        parallel_lanes=lanes, transaction_bytes=quantum,
        max_outstanding_requests=outstanding, resource_id=owner)
    metadata = {key: value for key, value in source.metadata.items()
        if key not in {"memory_service", "scan_memory_contract"}}
    metadata.update(physical_memory_config=physical, memory_service_owner=owner,
        transfer_granularity_bytes=quantum, read_latency_ns=read_latency,
        write_latency_ns=write_latency, parallel_lanes=lanes,
        max_outstanding_requests=outstanding,
        cost_profile_template=asdict(profile))
    if not is_hbm:
        metadata.update(queue_scope=physical.queue_scope, access_mode="memory", writable=True,
                        write_buffer_bytes=physical.write_buffer_bytes)
        if physical.protocol_mode != "ocp_hbf_0_7":
            for key in ("host_write_staging", "max_outstanding_requests_per_host_channel",
                        "max_outstanding_write_requests_per_host_channel"):
                metadata.pop(key, None)
    version = "4.0" if is_hbm and physical.generation == "HBM4" else None
    memory = replace(source, capacity_bytes=physical.capacity_bytes,
        bandwidth_gbps=shared * 8, read_bandwidth_gbps=read * 8,
        write_bandwidth_gbps=write * 8, metadata=metadata,
        ports=tuple(replace(port, bandwidth_gbps=shared * 8,
                    **({"version": version} if version else {})) for port in source.ports))
    links = tuple(replace(link, bandwidth_gbps=shared * 8,
        **({"version": version} if version else {}))
        if component_id in (link.source_component, link.target_component) else link
        for link in scenario.hardware.links)
    gpu_ports = {link.source_port if link.source_component == "gpu0" else link.target_port
        for link in links if {link.source_component, link.target_component} == {"gpu0", component_id}}
    components = tuple(memory if item.component_id == component_id else replace(item,
        ports=tuple(replace(port, bandwidth_gbps=shared * 8,
            **({"version": version} if version else {})) if port.port_id in gpu_ports else port
            for port in item.ports)) if item.component_id == "gpu0" else item
        for item in scenario.hardware.components)
    registries = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    registries[scenario.component_profile_kind(source)][source.cost_profile_id] = profile
    return replace(scenario, hardware=replace(scenario.hardware, components=components, links=links),
                   component_profiles=registries)


def _add_hbm_for_intermediates(scenario):
    hardware, profiles, _ = _hardware_and_profiles("HBM")
    hbm = hardware.get_component("hbm0")
    gpu_hbm_port = next(port for port in hardware.get_component("gpu0").ports if port.port_id == "hbm0")
    components = tuple(replace(item, ports=item.ports + (gpu_hbm_port,),
        metadata={**item.metadata, "scan_active_memory": "hbm0"}) if item.component_id == "gpu0"
        else item for item in scenario.hardware.components) + (hbm,)
    links = scenario.hardware.links + tuple(link for link in hardware.links
        if "hbm0" in (link.source_component, link.target_component))
    registries = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    registries.setdefault("hbm", {}).update(profiles["hbm"])
    parallel = scenario.placement.parallel
    parallel = replace(parallel, rank_mapping=tuple(replace(rank, memory_component_id="hbm0")
                                                   for rank in parallel.rank_mapping))
    metadata = {**scenario.placement.metadata,
        "llama_backend_memory": {"hbf0": {"device_id": "gpu0", "buffer_kind": "HBF",
                                        "access": "direct", "priority": 1}}}
    return replace(scenario, hardware=replace(scenario.hardware, components=components, links=links),
        component_profiles=registries,
        placement=replace(scenario.placement, parallel=parallel, metadata=metadata))


def build_control_scenario(case, *, model_id="qwen2_5-0_5b", prompt_tokens=1660,
                           output_tokens=17, batch_size=1):
    if case not in CASES:
        raise ValueError("unknown control case: " + str(case))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    architecture = "HBM" if case in {"scan_hbm", "paper_hbm4"} else "HBF"
    scenario = build_scan_scenario(model_id, architecture, prompt_tokens, output_tokens)
    if case in {"scan_hbf_hbm_intermediates", "paper_hbf", "paper_conv"}:
        scenario = _add_hbm_for_intermediates(scenario)
    if case.startswith("paper_"):
        from exploring_hbf_paper_config import (build_paper_hbm_config, build_paper_nand_config,
                                               default_research_assumptions)
        identifier = {"paper_hbm4": "HBM4", "paper_hbf": "HBF", "paper_conv": "CONV"}[case]
        scenario = _set_memory(scenario, "hbm0", build_paper_hbm_config(identifier))
        if architecture == "HBF":
            scenario = _set_memory(scenario, "hbf0", build_paper_nand_config(identifier,
                assumptions=default_research_assumptions(identifier)))
    scheduler = replace(scenario.workload.scheduler, max_num_seqs=batch_size,
        max_num_batched_tokens=prompt_tokens * batch_size,
        max_num_ubatch_tokens=prompt_tokens * batch_size,
        prefill_chunk_tokens=prompt_tokens, slo_ttft_ns=None, slo_tbt_ns=None,
        slo_tpot_ns=100_000_000.0)
    metadata = deepcopy(scenario.placement.metadata)
    options = metadata["control_plane"]["policy"]["options"]
    options.update(design_decode_batch_size=batch_size, design_prefill_tokens=prompt_tokens)
    requirements = _derive_requirements(scenario)
    options["operator_targets"] = {item.mapping_key: "gpu0" for item in requirements
        if item.mapping_key and not item.state_tensor}
    # Only resident model weights are controlled here; active/workspace buffers
    # follow the rank memory_component_id, not the flash weight owner.
    options["weight_tensor_targets"] = {item.tensor_id: "hbm0" if architecture == "HBM" else "hbf0"
        for item in requirements if item.tensor_id and item.tensor_bytes > 0 and not item.state_tensor}
    return replace(scenario, name="exploring-hbf-control-" + case,
        simulation=SimulationConfig(mode="detailed"),
        placement=replace(scenario.placement, metadata=metadata),
        workload=replace(scenario.workload, name="bounded-closed-batch-physical-control",
            requests=tuple(RequestSpec(f"control-{index}", 0.0, prompt_tokens, output_tokens)
                           for index in range(batch_size)), request_count=batch_size, scheduler=scheduler),
        assumptions=("B200 and Qwen are controlled substitutions, not the paper Rubin/Llama workload.",
            "Finite closed batch: no departure-driven refill, no disaggregated KV reception, no paper staging SRAM.",
            "Short real execution; no forecast and no full-length detailed-reference rerun.",
            "Paper-memory cases use generic page service with explicit research controls; not OCP/GC parity with scan cases."))


def execute_control(parameters):
    case, batch_size, model_id, prompt_tokens, output_tokens = parameters
    started = time.perf_counter()
    scenario = build_control_scenario(case, model_id=model_id, batch_size=batch_size,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens)
    with scenario_gc_thresholds():
        result = run_scenario(scenario, retention_policy="aggregate")
    metrics = list(result.serving.request_metrics.values())
    if len(metrics) != batch_size or any(item.status is not RequestStatus.FINISHED
        or item.visible_output_tokens != output_tokens for item in metrics):
        raise RuntimeError("bounded control did not finish its authored requests")
    token_times = {item.request_id: [] for item in metrics}
    for event in result.serving.events:
        if event.event_type == "tokens_committed" and event.request_id in token_times:
            token_times[event.request_id].extend([float(event.timestamp_ns)] * int(event.details.get("visible_tokens", 0)))
    if any(len(times) != output_tokens for times in token_times.values()):
        raise RuntimeError("missing visible-token timing evidence")
    tpots = [(times[-1] - times[0]) / (len(times) - 1) for times in token_times.values() if len(times) >= 2]
    first = min(times[0] for times in token_times.values())
    last = max(times[-1] for times in token_times.values())
    decode_tokens = batch_size * (output_tokens - 1)
    scenario = result.scenario
    placement = scenario.placement
    memory = {component.component_id: asdict(require_physical_memory_config(component))
        for component in scenario.hardware.components if component.component_id in {"hbm0", "hbf0"}}
    phases = []
    for phase in ("prefill", "decode"):
        batches = [batch for batch in result.serving.batches if any(item.phase == phase for item in batch.items)]
        phases.append({"phase": phase, "batches": len(batches),
            "batch_occupied_ns": sum(batch.end_ns-batch.start_ns for batch in batches),
            "scope": "real physical cohorts; initialization excluded; not an exclusive hardware attribution"})
    return {"case": case, "label": CASES[case], "status": "completed", "batch_size": batch_size,
        "model_id": model_id, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "wall_seconds": time.perf_counter()-started, "ttft_ns_mean": statistics.mean(item.ttft_ns for item in metrics),
        "tpot_ns_mean": statistics.mean(tpots) if tpots else None,
        "tpot_slo_ns": 100_000_000.0, "tpot_slo_met": bool(tpots) and all(value <= 100_000_000.0 for value in tpots),
        "decode_window_tps": decode_tokens*1e9/(last-first) if last > first else None,
        "request_tpot_ns": tpots, "phase_metrics": phases,
        "request_token_windows": {key: {"first_token_ns": times[0], "last_token_ns": times[-1]}
                                  for key, times in token_times.items()},
        "storage_traffic": _sum_batch_storage_traffic(result), "dram_traffic": _sum_batch_dram_traffic(result),
        "physical_configs": memory, "assumptions": scenario.assumptions,
        "resolved_placement": {"tensor_owners": sorted(set(placement.tensor_to_component.values())),
            "kv_owner": placement.kv_policy.cache_component,
            "rank_active_memory": [rank.memory_component_id for rank in placement.parallel.rank_mapping]},
        "fidelity": "short_native_physical_execution_no_forecast",
        "paper_tps_reproduced": False,
        "throughput_scope": "finite closed-batch decode window after first visible token; not continuous steady-state TPS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--model-id", default="qwen2_5-0_5b")
    parser.add_argument("--prompt-tokens", type=int, default=1660)
    parser.add_argument("--output-tokens", type=int, default=17)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.output_tokens < 2:
        parser.error("at least two visible tokens are required for TPOT")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [(case, batch, args.model_id, args.prompt_tokens, args.output_tokens)
                  for case in args.cases for batch in args.batch_sizes]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(execute_control, item): item for item in parameters}
        for future in as_completed(pending):
            item = pending[future]
            try:
                row = future.result()
            except Exception as error:
                row = {"case": item[0], "batch_size": item[1], "status": "failed",
                       "error": type(error).__name__ + ": " + str(error)}
            rows.append(row)
            (args.output_dir / f"{item[0]}__B{item[1]}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False, default=json_default), encoding="utf-8")
            print(json.dumps({key: row.get(key) for key in
                ("case", "batch_size", "status", "wall_seconds", "tpot_ns_mean", "error")}, ensure_ascii=False), flush=True)
    payload = {"schema": "exploring-hbf-bounded-controls-v1", "rows": sorted(rows, key=lambda row: (row["case"], row["batch_size"])),
        "paper_reproduction": False, "formal_scan_touched": False,
        "scope": "B200/Qwen short physical controls; existing 24 full-request results are separate"}
    (args.output_dir / "controls.json").write_text(json.dumps(payload, ensure_ascii=False,
        indent=2, allow_nan=False, default=json_default), encoding="utf-8")
    return 1 if any(row["status"] != "completed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
