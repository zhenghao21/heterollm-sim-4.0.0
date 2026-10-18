"""Question-driven HBM/HBF experiments, with measured scope per group.

python -B scripts/hbf_hbm_experiments.py --output-dir artifacts/hbf_hbm

All rates use decimal GB/s. Synthetic parameters are explicit assumptions,
not commercial-device calibration. No third-party Python packages are needed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import html
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import accept_hbf_hbm as acceptance
import calibrated_hbf_profiles as profiles
import hbf_hbm_invariants as invariants
import hbf_hbm_metrics as study_metrics
from hbf_hbm_core_audit import CoreExecutionAudit
from hbf_hbm_serving_study import POINT_RESULT_VERSION
from heterollm_sim.ir import LayerSpec, ModelSpec, RequestSpec, build_model_graph_from_layer_specs
from heterollm_sim.memory_types import AccessRequest, Operation, parse_physical_memory_config
from heterollm_sim.physical_contract import reconfigure_physical_config
from heterollm_sim.nand_core import NandCore
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.serving import compile_serving_plan, RequestStatus
from heterollm_sim.planner import ScenarioValidationError

GIB = 1024 ** 3
MIB = 1024 ** 2
PAGE = 4096


def group(identifier, title, rows, checks, supported, limitations, **extra):
    passed = bool(checks) and all(checks.values())
    return dict(id=identifier, title=title, status="passed" if passed else "failed",
                point_result_version=POINT_RESULT_VERSION,
                rows=rows, checks=checks, capability={"supported": supported if passed else [],
                "unsupported": limitations}, limitations=limitations, **extra)


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    def at(p):
        if not values:
            return None
        pos = (len(values) - 1) * p
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (pos-lo)
    return {"count": len(values), **{k: at(p) for k, p in
            (("p50", .5), ("p95", .95), ("p99", .99), ("p999", .999))}}


def synthetic_profile():
    return profiles.CalibratedProfile(
        profile_id="synthetic_interface_budget", paper="Explicit analytical assumptions", source_url="",
        source_version="local experiment v2", hardware={"hbm_capacity_gb": 512 * GIB / 1e9,
        "hbf_capacity_gb": 512, "hbm_stacks": 8, "hbf_stacks": 1,
        "effective_parallel_units": 64,
        "effective_parallel_units_source": "fixed sensitivity assumption, not inferred from interface bandwidth or disclosed by a vendor",
        "hbf_host_channels": 16, "hbf_host_full_duplex": True},
        bandwidth={"hbm_read_gb_s": 1600., "hbm_write_gb_s": 1600.,
        "hbm_command_clock_ns": profiles.HBM_REFERENCE_CLOCK_NS,
        "hbf_read_gb_s": 1600., "hbf_write_gb_s": 16.,
        "interconnect_bandwidth_gbps": 1600. * 8., "interconnect_latency_ns": 100.},
        latency={"hbf_read_us": 4., "hbf_write_us": 75.},
        granularity={"page_bytes": PAGE, "read_bytes": 64},
        buffering={"read_cache_pages_per_bank": 2}, slo={}, workload={},
        calibration_notes=(
            "HBM uses a fixed 0.625 ns analytical clock and 64 GiB reference stacks; directional service budgets are independent of physical interface peak.",
            "HBF is generic analytical page service for logical model buffers, not OCP physical sequential-write execution; no FTL cost is invented.",
            "HBF 64 units x 4 KiB / 4 us gives a 65.536 GB/s ideal cold-array read ceiling; 75 us program gives 3.495253 GB/s, below the 1600/16 GB/s interface budgets.",))


def scenario(*, hbf=False, prompt=128, count=4, output=8, slots=None,
             interval_ns=0., kv_budget=None, read_gb_s=1600., write_gb_s=16.,
             units=64, read_us=4., program_us=75., page_bytes=PAGE,
             link_gb_s=1600., hbm_timing_profile="hbm3e_reference"):
    p = synthetic_profile()
    p = replace(p, bandwidth={**p.bandwidth, "hbf_read_gb_s": read_gb_s,
                "hbf_write_gb_s": write_gb_s, "interconnect_bandwidth_gbps": link_gb_s*8}, hardware={**p.hardware,
                "effective_parallel_units": units}, latency={**p.latency,"hbf_read_us":read_us,
                "hbf_write_us":program_us}, granularity={**p.granularity,"page_bytes":page_bytes})
    s = profiles._full_backend_scenario(p, use_hbf=hbf, request_count=count,
        prompt_tokens=prompt, output_tokens=output, hbm_timing_profile=hbm_timing_profile)
    # INT8 GQA: Q/K/V/O = 196608 B; gated MLP = 393216 B.
    layers = (LayerSpec("dense0", "dense", 256, 512, 4, kv_heads=2,
              dtype="int8", quantization="w8a8", weight_bytes=589824),)
    model = ModelSpec(name="synthetic-small-dense", graph=build_model_graph_from_layer_specs(
        "synthetic-small-dense", layers, vocabulary_size=1024,
        max_sequence_length=max(262144, prompt+output)))
    scheduler = replace(s.workload.scheduler, max_num_seqs=slots or count,
        max_num_batched_tokens=min(2048, prompt) * (slots or count),
        max_num_ubatch_tokens=min(2048, prompt) * (slots or count),
        prefill_chunk_tokens=min(2048, prompt), slo_ttft_ns=5_000_000., slo_tbt_ns=1_000_000.)
    requests = tuple(RequestSpec(f"r{i}", i * interval_ns, prompt, output) for i in range(count))
    if kv_budget is not None:
        # Reserve an explicit KV-only device, so weights/workspace on hbm0
        # cannot silently consume or enlarge the experiment's KV budget.
        source = s.hardware.get_component("hbf0" if hbf else "hbm0")
        physical = parse_physical_memory_config(source.metadata["physical_memory_config"])
        device_id = "hbf0" if hbf else "hbm_kv"
        metadata = {k:v for k,v in source.metadata.items() if k != "memory_service"}
        metadata["physical_memory_config"] = (
            reconfigure_physical_config(physical,
                overrides={"capacity_bytes":kv_budget},expand_geometry=True) if hbf else
            profiles._hbm_physical_config(capacity_bytes=kv_budget,
                stacks=physical.stacks,
                read_gb_s=physical.directional_bandwidth_gb_s("read"),
                write_gb_s=physical.directional_bandwidth_gb_s("write"),
                clock_ns=physical.command_clock_ns,
                interface_input_scope=physical.metadata.get("interface_input_scope"),
                profile_id=hbm_timing_profile))
        if not hbf:
            metadata["memory_service_owner"]="hbm_kv.hbm_fabric"
        device = replace(source, component_id=device_id, capacity_bytes=kv_budget,
            cost_profile_id=source.cost_profile_id if hbf else "experiment-kv-hbm",
            metadata=metadata)
        components = [device if c.component_id == device_id else c for c in s.hardware.components]
        links = list(s.hardware.links)
        registry = {k:dict(v) for k,v in s.component_profiles.items()}
        if not hbf:
            components.append(device)
            gpu = next(c for c in components if c.component_id=="gpu0")
            port = replace(next(p for p in gpu.ports if p.port_id=="hbm0"),port_id="hbm_kv")
            components=[replace(c,ports=c.ports+(port,)) if c.component_id=="gpu0" else c for c in components]
            link=next(l for l in links if l.source_component=="gpu0" and l.target_component=="hbm0")
            links.append(replace(link,link_id="gpu-hbm-kv",source_port="hbm_kv",target_component="hbm_kv"))
            kv_physical = metadata["physical_memory_config"]
            registry["hbm"]["experiment-kv-hbm"]=replace(registry["hbm"][source.cost_profile_id],
                resource_id="hbm_kv.hbm_fabric", parallel_lanes=kv_physical.lane_count,
                transaction_bytes=kv_physical.burst_bytes,
                max_outstanding_requests=kv_physical.max_outstanding_requests)
        s=replace(s,hardware=replace(s.hardware,components=tuple(components),links=tuple(links)),
            component_profiles=registry,placement=replace(s.placement,
            kv_policy=replace(s.placement.kv_policy,cache_component=device_id,allocation_policy="eager")))
    return replace(s, model=model, name="synthetic-dense-"+("B" if hbf else "A"),
        placement=replace(s.placement, model_name=model.name,
            parallel=replace(s.placement.parallel, layer_to_stage={"dense0": 0})),
        workload=replace(s.workload, requests=requests, scheduler=scheduler))


def serving_row(s, row_id, *, placement=None, result_observer=None):
    started = time.perf_counter()
    with CoreExecutionAudit() as core_audit:
        result = run_scenario(s)
    core_execution_audit = core_audit.summary()
    if result_observer is not None:
        result_observer(result)
    r = report_dict(result, visualization_limit=1, include_visualization=False)
    summary = r["summary"]
    metrics = list(result.serving.request_metrics.values())
    completed = [m for m in metrics if m.status is RequestStatus.FINISHED]
    events = result.serving.events
    token_times = {}
    for e in events:
        if e.event_type == "tokens_committed" and e.request_id:
            token_times.setdefault(e.request_id, []).extend([float(e.timestamp_ns)] * int(e.details.get("visible_tokens", 0)))
    tbts = [b-a for times in token_times.values() for a,b in zip(times, times[1:])]
    windows = study_metrics.measurement_windows(result.serving)
    duration = windows["request_window"]["duration_ns"]
    runtime_duration = windows["runtime_window"]["duration_ns"]
    token_count = sum(m.visible_output_tokens for m in completed)
    good = []
    for m in completed:
        times = token_times.get(m.request_id, [])
        gaps = [b-a for a,b in zip(times,times[1:])]
        sch = s.workload.scheduler
        if (m.deadline_met is not False and (sch.slo_ttft_ns is None or
            m.ttft_ns is not None and m.ttft_ns <= sch.slo_ttft_ns) and
            (sch.slo_tbt_ns is None or max(gaps, default=0.) <= sch.slo_tbt_ns)):
            good.append(m)
    # The primary window includes request waiting during initialization.
    # Keep runtime-only rates separately labelled, without altering the backend.
    tps = token_count * 1e9 / duration if duration else 0.
    good_tps = sum(m.visible_output_tokens for m in good) * 1e9 / duration if duration else 0.
    kv = r.get("kv_cache", {})
    hbf_component = next((c for c in s.hardware.components if c.component_id == "hbf0"), None)
    actual_hbf = parse_physical_memory_config(hbf_component.metadata["physical_memory_config"]) if hbf_component else None
    links = [{"id":l.link_id,"bandwidth_gbps":l.bandwidth_gbps,"latency_ns":l.latency_ns}
             for l in s.hardware.links if "hbf0" in (l.source_component,l.target_component)]
    phase_rates = study_metrics.phase_metrics(result.serving, duration, {m.request_id for m in completed})
    gpu_count = sum(str(c.normalized_kind).lower() == "gpu" for c in s.hardware.components)
    if not gpu_count:
        raise ValueError("serving study requires at least one GPU")
    migrations = [{"completion_ns":float(e.timestamp_ns),**dict(e.details)}
                  for e in events if e.event_type == "kv_pool_transfer_complete"]
    return {
        "id": row_id, "mode": "full_operator_serving", "model": s.model.name,
        "point_result_version": POINT_RESULT_VERSION,
        "core_execution_audit": core_execution_audit,
        "hbf_service": profiles.hbf_service_summary(actual_hbf) if actual_hbf else None,
        "hbm_protocol": {owner: profiles.hbm_protocol_summary(item["physical_config"])
                         for owner, item in core_execution_audit.get("executed_physical_configurations", {}).items()
                         if item["physical_config"].get("kind") == "HBM"},
        "placement": placement or ("B_all_kv_hbf" if s.placement.kv_policy.cache_component == "hbf0" else "A_hbm_only"),
        "resolved_placement": {"tensor_to_component":dict(result.scenario.placement.tensor_to_component),
            "op_to_component":dict(result.scenario.placement.op_to_component),
            "kv_cache_component":result.scenario.placement.kv_policy.cache_component,
            "kv_policy":{key:getattr(result.scenario.placement.kv_policy,key,None) for key in
                ("layout_mode","allocation_policy","offload_component","prefetch_distance")}},
        "prompt_tokens": s.workload.requests[0].prompt_tokens,
        "output_tokens_per_request":s.workload.requests[0].output_tokens,
        "requested_concurrency": s.workload.scheduler.max_num_seqs,
        "request_count": len(metrics), "completed_requests": len(completed),
        "rejected_requests": int(summary.get("rejected_requests", 0)),
        "unfinished_requests": len(metrics)-len(completed)-int(summary.get("rejected_requests", 0)),
        "rejection_reasons": [m.rejection_reason for m in metrics if m.rejection_reason],
        "rejection_rate": int(summary.get("rejected_requests", 0))/max(1,len(metrics)),
        "node_tps": tps, "per_gpu_tps": tps/gpu_count, "gpu_count": gpu_count,
        "visible_output_tokens": token_count,
        "runtime_tps": token_count*1e9/runtime_duration if runtime_duration else 0.,
        "runtime_goodput_tps": sum(m.visible_output_tokens for m in good)*1e9/runtime_duration if runtime_duration else 0.,
        "goodput_tps": good_tps, "qualified_requests": len(good),
        "requests_per_s": len(completed)*1e9/duration if duration else 0.,
        "makespan_ns": duration, "measurement_window": "earliest request arrival through simulation end, including initialization wait",
        "measurement_windows": windows,
        "ttft_ns": percentiles(m.ttft_ns for m in completed),
        "tpot_ns": percentiles(m.tpot_ns for m in completed), "tbt_ns": percentiles(tbts),
        "e2e_ns":percentiles(m.finish_ns-m.arrival_ns for m in completed),
        "phase_service_rates":phase_rates,
        "speculative": study_metrics.speculative_metrics(result.serving),
        "memory_metrics": study_metrics.memory_metrics(r, duration),
        "migration_ledger":{"events":migrations,
            "bytes_by_kind":{kind:sum(int(e.get('bytes',0)) for e in migrations if e.get('kind')==kind)
                             for kind in sorted({e.get('kind','unknown') for e in migrations})},
            "scope":"physical KV pool transfers, separate from operator NAND traffic; explicit endpoints and completion times"},
        "prefetch":{"submitted_bytes":sum(int(e.details.get('bytes',0)) for e in events if e.event_type=='kv_prefetch_start'),
            "completed_bytes":kv.get('prefetch_completed_bytes',0),
            "useful_bytes":kv.get('prefetch_useful_bytes',0),
            "unused_bytes":kv.get('prefetch_unused_bytes',0),
            "inflight_bytes":kv.get('prefetch_inflight_bytes',0),
            "cancelled_bytes":kv.get('prefetch_cancelled_bytes') or 0,
            "useful_byte_fraction":study_metrics.ratio(kv.get('prefetch_useful_bytes',0),kv.get('prefetch_completed_bytes',0)),
            "semantics":"fraction of completed lookahead bytes consumed by resumed requests; not learned-router prediction accuracy"},
        "prefetch_overlap":study_metrics.prefetch_overlap(result.serving,
            {c.component_id for c in s.hardware.components if str(c.normalized_kind).lower()=='gpu'}),
        "request_metrics":[{"id":m.request_id,"arrival_ns":m.arrival_ns,"finish_ns":m.finish_ns,
            "first_token_ns":m.first_token_ns,"status":m.status.value,"output_tokens":m.visible_output_tokens,
            "ttft_ns":m.ttft_ns,"tpot_ns":m.tpot_ns,"qualified":m in good} for m in metrics],
        "quantile_scope":"empirical interpolation over reported sample count; no population tail-latency guarantee",
        "address_assumptions":{"embedding":"No token-ID trace is supplied. Analytical remote embedding row traffic uses a bounded cyclic address model when reads exceed the resident table; repeated traffic is retained and the allocation is not enlarged. This is not a measured token locality trace."},
        "max_observed_batch_requests":max((len(b.request_ids) for b in result.serving.batches),default=0),
        "kv": kv, "resources": {k:r.get(k) for k in ("dram_traffic","storage_traffic","resource_utilization")},
        "scheduler":r.get("scheduler"), "owner_residency":r.get("owner_residency"),
        "applied_hardware":{"hbf_read_gb_s":getattr(actual_hbf,"host_read_bandwidth_gb_s",None),
            "memory_kinds":{c.component_id:c.normalized_kind for c in s.hardware.components if c.is_storage},
            "hbf_write_gb_s":getattr(actual_hbf,"host_write_bandwidth_gb_s",None),
            "hbf_parallel_units":getattr(actual_hbf,"parallel_units",None),"hbf_page_bytes":getattr(actual_hbf,"page_bytes",None),
            "hbf_read_ns":getattr(actual_hbf,"page_read_ns",None),"hbf_program_ns":getattr(actual_hbf,"page_program_ns",None),
            "hbm_peak_read_gb_s":s.resolve_component_profile("hbm0").read_bandwidth_gb_s,
            "hbm_efficiency":s.resolve_component_profile("hbm0").efficiency,
            "links":links,"kv_allocation_policy":s.placement.kv_policy.allocation_policy},
        "slo": {"ttft_ns":s.workload.scheduler.slo_ttft_ns,"tbt_ns":s.workload.scheduler.slo_tbt_ns},
        "wall_seconds": time.perf_counter()-started,
        "unavailable": {"lifetime":"not a stateful wear experiment"},
    }


def ideal_group():
    report = invariants.run_invariants()
    rows = report["checks"]
    return group("01_ideal", "① 理想隔离", rows, {r["id"]:r["passed"] for r in rows},
        ["同读带宽、去额外延迟时的HBM/HBF等价性；聚合与细粒度执行一致性"],
        ["这是内存不变量，不能据此预测完整LLM吞吐。"])


def demand_matrix():
    rows = []
    # BF16 80-layer GQA demand: independent of an unverified named GPU profile.
    bkv = 2*80*8*128*2
    for context in (2048,32768,131072,1048576):
        for batch in (1,4,16,64):
            read = bkv*context*batch
            write = bkv*batch
            rows.append({"context_tokens":context,"batch":batch,"kv_bytes_per_token":bkv,
                "read_bytes":read,"write_bytes":write,"kv_resident_bytes":read+write,
                "hbm_shared_bus_lower_bound_ns":(read+write)/1600.,
                "hbf_shared_bus_lower_bound_ns":read/1600.+write/16.,
                "mode":"analytical_memory_bound_not_serving_tps"})
    return rows


def fixed_workload_group():
    rows = []
    # Representative full-backend points, with a separate complete demand grid.
    for prompt,batch in ((2048,1),(2048,4),(2048,16),(2048,64),
                         (32768,1),(131072,1),(1048576,1)):
        for hbf in (False,True):
            rows.append(serving_row(scenario(hbf=hbf,prompt=prompt,count=batch,output=4),f"fixed-{prompt}-{batch}-{hbf}"))
    # One-factor sweeps only: avoid confusing bandwidth, concurrency and latency.
    for key,values in (("read_gb_s",(400.,3000.)),("write_gb_s",(4.,64.)),
                       ("units",(8,128)),("read_us",(1.,10.,25.)),
                       ("program_us",(100.,200.)),("page_bytes",(4096,65536,262144))):
        for value in values:
            row = serving_row(scenario(hbf=True,prompt=2048,count=1,output=4,**{key:value}),f"sweep-{key}-{value}")
            row["scan"]={key:value}
            rows.append(row)
    checks = {"all_requests_completed":all(r["completed_requests"] == r["request_count"] for r in rows),
              "real_serving_metrics":all(r["node_tps"]>0 and r["tpot_ns"]["p50"] is not None for r in rows),
              "requested_fixed_batches_observed":all(r['max_observed_batch_requests']==r['requested_concurrency'] for r in rows if r['id'].startswith('fixed-'))}
    baseline=rows[1]["makespan_ns"]
    checks["slower_write_bandwidth_changes_execution"]=next(r['makespan_ns'] for r in rows if r.get('scan')=={'write_gb_s':4.})>baseline
    checks["higher_read_latency_changes_execution"]=next(r['makespan_ns'] for r in rows if r.get('scan')=={'read_us':25.})>baseline
    return group("02_fixed_workload","② 固定工作量",rows,checks,
        ["小Dense模型固定工作量下A/B的吞吐、延迟及读写带宽/并行度/读延迟敏感性",
         "70B级BF16 GQA在2K至1M、batch 1/4/16/64的KV容量与带宽下界"],
        ["1600 GB/s相等的是端口峰值；HBF页延迟、64并行单元、互联仍会降低有效带宽。",
         "读带宽扫描至3000 GB/s时互联仍为1600 GB/s，必须一并考虑上游限制。",
         "70B表是字节需求/解析下界，不是70B完整算子或商用设备TPS。",
         "1M是小模型外推压力场景；有限请求分位数不代表生产p99尾延迟。",
         "检查通过表示执行/记账正确，不表示SLO达标；goodput为0的点保留。"],demand_matrix=demand_matrix())


def capacity_group():
    rows=[]
    for hbf in (False, True):
        for budget in (32*1024,64*1024,256*1024):
            for slots in (1,4):
                for interval in (0., 1_000_000., 10_000_000.):
                    s=scenario(hbf=hbf,kv_budget=budget,count=8,slots=slots,interval_ns=interval)
                    try:
                        row=serving_row(s,f"capacity-{hbf}-{budget}-{slots}-{interval}")
                    except ScenarioValidationError as error:
                        if budget != 32*1024 or 'KV' not in str(error):
                            raise
                        row={"id":f"capacity-{hbf}-{budget}-{slots}-{interval}",
                            "mode":"configuration_rejected", "placement":"B_all_kv_hbf" if hbf else "A_hbm_only",
                            "request_count":8,"completed_requests":0,"rejected_requests":8,
                            "rejection_stage":"pre_execution_validation","rejection_reasons":[str(error)],
                            "rejection_rate":1.,"unfinished_requests":0,"node_tps":0.,"goodput_tps":0.,
                            "per_gpu_tps":0.,"ttft_ns":None,"tpot_ns":None,"tbt_ns":None,
                            "unavailable":"single request cannot fit; no serving timeline was executed"}
                    row.update(kv_budget_bytes=budget,arrival_interval_ns=interval)
                    row["compiled_kv_capacity_bytes"]=compile_serving_plan(s).kv_policy.capacity_bytes
                    rows.append(row)
    checks={"capacity_budget_applied":all(r["compiled_kv_capacity_bytes"]<=r["kv_budget_bytes"] for r in rows),
            "over_capacity_is_rejected":any(r["rejected_requests"]>0 for r in rows),
            "sufficient_capacity_completes":any(r["completed_requests"]==8 for r in rows),
            "goodput_bounded_by_tps":all(r["goodput_tps"]<=r["node_tps"]+1e-8 for r in rows),
            "finished_or_explicitly_rejected":all(r["unfinished_requests"]==0 for r in rows)}
    return group("03_capacity_slo","③ 容量与SLO",rows,checks,
        ["给定小Dense请求流、容量和到达间隔下的拒绝、排队与合格吞吐；扫描范围内的可行点"],
        ["8请求有限轨迹不能证明无限期稳定服务的最大到达率；最大值仅指已扫描点。"])


def weight_trace(*, hbf_weights, moe):
    p=synthetic_profile()
    hbm,hbf,_=profiles._profile_hardware(p)
    cores={"hbm":hbm,"hbf":hbf}
    weight_tier="hbf" if hbf_weights else "hbm"
    # A/D consume identical explicit routing, without an oracle prefetch/cache.
    routes=((0,1),(0,2),(3,4),(0,1)) if moe else ((0,),(0,),(0,),(0,))
    expert_bytes=8*PAGE
    clock=0.; stages=[]; traffic={"hbm":{"read":0,"write":0},"hbf":{"read":0,"write":0}}
    for step,experts in enumerate(routes):
        end=[]
        for expert in experts:
            result=cores[weight_tier].submit(AccessRequest(f"w-{step}-{expert}",Operation.READ,expert*expert_bytes,expert_bytes,clock))
            end.append(result.completion_ns); traffic[weight_tier]["read"]+=expert_bytes
        # Full active KV read every step; no static hot-hit probability.
        kv_size=(2048+step)*512
        for op,address,size in ((Operation.READ,16*MIB,kv_size),(Operation.WRITE,16*MIB+kv_size,512)):
            result=hbm.submit(AccessRequest(f"kv-{step}-{op.value}",op,address,size,clock))
            end.append(result.completion_ns); traffic["hbm"]["read" if op is Operation.READ else "write"]+=size
        done=max(end); stages.append(done-clock); clock=done
    return {"id":f"{'D' if hbf_weights else 'A'}-{'moe' if moe else 'dense'}",
            "mode":"physical_memory_trace", "service_ns":clock,"step_ns":stages,
            "routing_trace":[list(x) for x in routes], "weight_bytes_per_expert":expert_bytes,
            "traffic":traffic,"node_tps":None,"per_gpu_tps":None,"goodput_tps":None,
            "ttft_ns":None,"tpot_ns":None,"unavailable":"fixed memory requests do not execute an LLM",
            "resources":{tier:{"busy_ns":dict(core.timeline.busy_ns),"bytes":dict(core.timeline.bytes_moved)} for tier,core in cores.items()}}


def weights_group():
    rows=[weight_trace(hbf_weights=hbf,moe=moe) for moe in (False,True) for hbf in (False,True)]
    checks={"same_routing_between_A_D":all(rows[i]["routing_trace"]==rows[i+1]["routing_trace"] for i in (0,2)),
            "hbf_weights_actually_read":all(r["traffic"]["hbf"]["read"]>0 for r in rows if r["id"].startswith("D")),
            "active_kv_stays_hbm":all(r["traffic"]["hbf"]["write"]==0 and r["traffic"]["hbm"]["write"]>0 for r in rows)}
    return group("08_weights","⑧ Dense/MoE权重",rows,checks,
        ["给定Dense/明确expert访问序列，A/D权重供给与HBM活跃KV的资源争用和服务成本"],
        ["未校准的合成expert轨迹；不能回答路由预测器精度或真实MoE端到端加速。"])


TITLES={1:"① 理想隔离",2:"② 固定工作量",3:"③ 容量与SLO",4:"④ 冷热会话与共享前缀",
        5:"⑤ 写缓冲与写放大",6:"⑥ Direct/relay/双路径",7:"⑦ 固定封装预算",8:"⑧ Dense/MoE权重",
        9:"⑨ 完整服务、命名模型与远端存储对照",10:"⑩ 精确热度、容量与读写比",
        11:"⑪ 预算约束下重新执行服务",12:"⑫ 完整服务消融与双路径QoS",
        13:"⑬ 多轮agent与取消",14:"⑭ 磨损与物理页队列",
        15:"⑮ 多GPU与共享内存池",16:"⑯ 持续负载SLO与会话边界",
        17:"⑰ 扩展模型规格与工作负载",18:"⑱ 目标有效带宽与物理页扫描",
        19:"⑲ DRAM/NAND Core下的独立Decode对照",
        20:"⑳ 统一参数入口下的Bank group Decode对照"}
AUDIT_FIXES=[
    "物理硬件使用统一目录与重配置入口，完整服务采用已核对的HBM参考组织；微基准中的理想化几何明确作为消融控制，第20组比较平坦分组、参考分组及合成组时序，实际参数以执行Core记录为准。",
    "修复迁移完成时刻重复累加、分块访问累加绝对时间、swap-in迁移漏计。",
    "修复写缓冲自动排空漏计、介质编程早于主机确认、独立通道被串行化，以及未写入区域误命中缓冲。",
    "修复迁移失败回滚、临时页预留破坏持久页、多页共享前缀释放，以及物理状态浅拷贝泄漏。",
    "修复通用KV按模型最大上下文过度预留；连续物理KV采用与真实缓冲一致的准入预留，并报告请求/有效策略。",
    "修复独立实验共用旧排队状态、把多步总时长当TPOT、未设置SLO仍生成合格吞吐。",
    "修正文献配置中HBF-2/B200的HBM带宽、HBF+的SRAM总量、物理路径与服务路径参数不一致。",
    "验收阶段状态由真实依赖检查计算；失败不会继续标记通过。论文参数化入口不再自行宣布整个计划完成。",
    "主TPS/goodput改用最早请求到达至结束窗口，另列初始化后TPS、窗口起止及初始化等待。",
    "修复实际batch达不到目标的问题；新增实际批量检查、1M压力点和缺失参数扫描。",
    "异步预取保留源副本并预占目标，完成后才可消费；迁移目标容量峰值按实际占用更新。",
    "远端NAND不再被要求绑定DRAM typed profile；NAND显式传输使用页对齐。",
    "完整服务报告新增分向带宽、页访问/读写操作、迁移账本、阶段墙钟吞吐及所有延迟分位数。",
    "前缀25%扫描改为20会话并区分请求/准入分母；能力声明随检查状态变化。",
    "批量研究可跳过UI轨迹序列化；摘要、KV、存储及调度统计保持相同。",
    "异步迁移按source/link/target阶段的真实就绪时刻提交，避免提前预订未来HBM写造成虚假的串行化。",
    "长prompt的解析embedding读取允许重复访问既有权重表；采用明确标注的循环地址假设，不扩大驻留权重容量。",
    "修复任务metadata重放使用过期对象标识引起的deepcopy缓存碰撞，避免长实验中的物理地址绑定污染。",
    "多轮agent在同一服务时间线继承实际KV页；TTL过期重新计算，取消迁移保留已执行成本并等待安全释放。",
    "补充完整服务的等效A/B消融、同链路预算direct/relay/dual及实际混合流优先级比较。",
    "NAND显式擦除计数与磨损影响后续服务，记录真实物理页队列和包含空闲plane的负载失衡。",
    "修复多GPU输出层集中到单GPU和服务汇总拒绝多GPU的问题；加入明确的多GPU共享内存暴露和端点路由约束。",
    "预取提交字节按实际提交事件计数，取消操作不再错误地报告为零提交。",
    "移除HBM按聚合带宽直接计时的旁路；算子访存和KV迁移均通过DRAM Core，GPU计算与启动仍保留原有简化模型。",
    "完整服务逐点记录持久运行时实际DRAM/NAND提交；另跑固定64输出的独立Decode窗口，避免prefill稀释Decode影响。",
]


def completion_matrix(groups):
    present={int(g['id'][:2]):g for g in groups}
    specs=[
        ('基础机制与参数扫描',(1,2,3,4,5,6,7,8), '八组理想、固定负载、容量、会话、写缓冲、路径、封装与权重实验全部存在并通过。'),
        ('理想A/B与逐项完整服务消融',(12,), '第12组独立读延迟/写限制以及并行度交互；零延迟下并行度可能无影响，保留零结果。'),
        ('direct/relay双路径与QoS',(12,), '同一64GB/s链路总预算；实际中继读写、静态带宽份额及请求优先级，不声称动态最优调度。'),
        ('多轮agent、预测放置和在途取消',(13,), '实际KV页继承、think time、TTL、写预算、预测失误和安全取消；采用可检查启发式，无训练精度声明。'),
        ('磨损及真实物理队列',(14,), '已用P/E条件影响后续NAND服务；显式擦除计数、read/program队列、物理plane失衡。'),
        ('多GPU与rack共享池',(15,), '1/2/4 GPU显式张量并行、实际通信和共享端点；机架带宽与时延进入实际路由。'),
        ('持续负载和session/GPU边界',(16,), '去除预热并延长轨迹，以SLO、完成率和积压变化搜索有限实验边界；不把有限实验当作无限期数学证明。'),
        ('原方案模型范围',(9,17), '完整算子服务与结构专属物理工作负载分别列出；复杂注意力未由GQA更名替代，逐模型公开执行范围。'),
        ('目标有效带宽与物理页扫描',(18,), '实际内存与链路服务后计算有效值；固定负载比较物理页和读延迟，保留带宽受限负对照。'),
        ('预算与SLO下配置选择',(11,), '枚举明确的九个固定4槽候选，重跑完整服务、共同预算上限和SLO；无可行解也是有效结果。'),
        ('购置成本、面积与3/5年摊销',(11,14), '参数作为内部假设；条件节点成本与实际候选栈数一致，均匀寿命推导与动态物理轨迹分别报告。'),
        ('投机解码、热点、共享前缀和读写混合',(4,9,10,13), '分别保存真实接受/拒绝、精确物理访问比例和多轮生命周期；不伪造模型token或训练路由轨迹。'),
        ('独立Decode与实际Core执行',(19,), '8B/70B、2K/32K/128K/1M、A/B共16点，统一64输出；实际执行prefill后单独统计63个Decode token。'),
        ('Bank group合成时序Decode对照',(20,), '小Dense完整服务的原参数、仅重分组和合成组时序三种统一目录配置；A/B分别对比，真实Core执行，不声称器件校准。')]
    return [{'item':item,'status':'已完成所述实验' if all(n in present and present[n]['status']=='passed' for n in numbers)
             and (17 not in numbers or extended_model_matrix_complete(groups))
             and (19 not in numbers or decode_matrix_complete(groups))
             and (20 not in numbers or bank_group_matrix_complete(groups)) else '本次未完成',
        'groups':list(numbers),'reason':reason} for item,numbers,reason in specs]


def bank_group_matrix_complete(groups):
    from hbf_hbm_bank_group_study import PROFILES, CONTEXTS, ARCHITECTURES
    study=next((g for g in groups if g['id']=='20_bank_group_decode'),{})
    rows=study.get('rows',[])
    expected={(p,c,a) for p in PROFILES for c in CONTEXTS for a in ARCHITECTURES}
    actual={(r.get('hardware_profile'),r.get('prompt_tokens'),r.get('placement')) for r in rows}
    return (study.get('status')=='passed' and actual==expected and len(rows)==len(expected)
            and all(r.get('visible_output_tokens')==8 and r.get('decode_measurement',{}).get('decode_tokens')==7
                    for r in rows))


def decode_matrix_complete(groups):
    decode=next((g for g in groups if g['id']=='19_decode_physical'),{})
    rows=decode.get('rows',[])
    expected={(m,c,a) for m in ('llama3_1-8b','llama3_1-70b')
              for c in (2048,32768,131072,1048576) for a in ('A_hbm_only','B_all_kv_hbf')}
    actual={(r.get('parameters',{}).get('model_id'),r.get('prompt_tokens'),r.get('placement')) for r in rows}
    return (decode.get('status')=='passed' and len(rows)==len(expected) and actual==expected
            and all(r.get('visible_output_tokens')==64 and r.get('decode_measurement',{}).get('decode_tokens')==63
                    and r.get('decode_core_checks') and all(r['decode_core_checks'].values()) for r in rows))


def memory_core_coverage(groups):
    rows=[r for g in groups for r in g.get('rows',[]) if r.get('mode')=='full_operator_serving']
    verified=[]; invalid=[]; nand_rows=[]; wrong_nand=[]
    for row in rows:
        audit=row.get('core_execution_audit',{})
        owners=audit.get('owners',[])
        hbm=[o for o in owners if o.get('memory_kind')=='HBM' and o.get('transactions',0)>0]
        nand=[o for o in owners if o.get('memory_kind') in {'HBF','SSD'} and o.get('transactions',0)>0]
        if nand and audit.get('valid_for_committed_byte_totals') and all(o.get('core')=='NandCore' for o in nand):
            nand_rows.append(row['id'])
        if any(o.get('core')!='NandCore' for o in nand):
            wrong_nand.append(row['id'])
        if (audit.get('valid_for_committed_byte_totals') and hbm
                and all(o.get('core')=='DramCore' for o in hbm)):
            verified.append(row['id'])
        else:
            invalid.append(row['id'])
    return {'full_serving_rows':len(rows),'verified_hbm_dram_core_rows':len(verified),
            'all_full_serving_hbm_verified':bool(rows) and not invalid,'unverified_row_ids':invalid,
            'verified_nand_core_rows':len(nand_rows),'wrong_nand_core_row_ids':wrong_nand,
            'execution_contract':'HBM memory timing uses DramCore; HBF memory timing uses NandCore; GPU compute and launch retain their analytical approximation.',
            'evidence_scope':'Actual successful submissions on each persistent online-serving runtime; excludes planning previews and pre-serving initialization.'}


def extended_model_matrix_complete(groups):
    extended=next((g for g in groups if g['id']=='17_extended_models'),{})
    original=next((g for g in groups if g['id']=='09_serving_study'),{})
    if not {'full','probes'}<=set(extended.get('sections',[])): return False
    if not {'placement','models','named_extended','long_context','baselines','slo','tail_sample','prefetch','speculative'}<=set(original.get('coverage',{}).get('sections',[])): return False
    full={(r.get('parameters',{}).get('model_id'),r.get('prompt_tokens'),r.get('placement'))
          for r in extended.get('rows',[]) if r.get('mode')=='full_operator_serving' and r.get('completed_requests',0)>0}
    probes={(r.get('model_id'),r.get('context'),r.get('placement'))
          for r in extended.get('rows',[]) if r.get('mode')=='structural_physical_probe' and r.get('node_tps') is None}
    architectures=('A_hbm_only','B_all_kv_hbf','D_hbf_weights')
    return all((m,c,a) in full for m in ('qwen3-coder-30b-a3b','synthetic-1t-gqa-moe') for c in (2048,32768) for a in architectures) and all(
        (m,c,a) in probes for m in ('deepseek-v3','deepseek-v4-flash-0731','llama4-scout','llama4-maverick') for c in (2048,32768) for a in architectures)


def model_execution_coverage(groups):
    rows=[r for g in groups if g['id'] in {'09_serving_study','17_extended_models'} for r in g.get('rows',[])]
    result=[]
    for model in ('llama3_1-8b','llama3_1-70b','llama3_1-405b','qwen3-30b-a3b','qwen3-235b-a22b',
                  'qwen3-coder-30b-a3b','synthetic-1t-gqa-moe','deepseek-v3','deepseek-v4-flash-0731','llama4-scout','llama4-maverick'):
        matches=[r for r in rows if r.get('model_id',r.get('parameters',{}).get('model_id'))==model]
        modes=sorted({r['mode'] for r in matches})
        result.append({'model_id':model,'modes':modes,'result_points':len(matches),
            'executed_points':sum(r.get('executed_serving') is True or r['mode']=='structural_physical_probe' for r in matches),
            'rejected_points':sum(r['mode'] in {'configuration_rejected','physical_capacity_rejected'} for r in matches),
            'contexts':sorted({context for r in matches if (context:=r.get('context',r.get('prompt_tokens',r.get('parameters',{}).get('prompt')))) is not None}),
            'native_full_serving':any(r['mode']=='full_operator_serving' for r in matches),
            'support_boundary':next((r['support_boundary'] for r in matches if 'support_boundary' in r),None)})
    return result


EXCLUDED_SCOPE=[
    {'item':'真实外部数据校准','status':'依用户指示暂不纳入','reason':'不要求真实GPU/HBM/HBF测量或独立精度误差。所有设备数字仍标注为仿真假设。'},
    {'item':'原方案长期研究方向','status':'与本次可复现内部实验区分','reason':'完整FTL/ECC、故障恢复、加密隔离、PIM及学习式路由器属于原文长期研究方向；不冒充已实现产品能力。'},
    {'item':'全模型全参数笛卡尔积与全局最优证明','status':'不作此声明','reason':'原方案的扫描与模型覆盖按明确实验矩阵输出；有界参数空间与有限轨迹不等于所有架构和无限期运行。'},
]


def checklist_progress(groups, economics):
    rows=[r for g in groups for r in g.get('rows',[]) if r.get('mode')=='full_operator_serving']
    study=next((g for g in groups if g['id']=='09_serving_study'),None)
    fixed=[r for r in rows if r['id'].startswith('fixed-')]
    return [
        {'item':'统计窗口及可读指标','status':'已修正并输出' if rows else '本次未执行',
         'reason':'TPS/goodput共同请求窗口；初始化后TPS、TTFT/TPOT/TBT/E2E四分位数和样本数、分向带宽均保留。'},
        {'item':'真实batch 1/4/16/64','status':'已覆盖' if {1,4,16,64}<={r['max_observed_batch_requests'] for r in fixed} else '部分或未执行',
         'reason':'按实际max_observed_batch_requests判断，不按场景名称计数。'},
        {'item':'1M完整服务','status':'已执行' if any(r['prompt_tokens']==1048576 for r in rows) else '未执行',
         'reason':'超过模型原生上下文的结果只作架构压力外推，不作语义准确率声明。'},
        {'item':'A/B/C/D与命名模型','status':study['status'] if study else '本次未执行',
         'reason':'实际算子执行与物理放置检查见第9组；第13组另运行真正的多轮agent生命周期。'},
        {'item':'购置成本/面积/寿命','status':'条件敏感性已输出' if economics else '本次未生成',
         'reason':'明确内部假设；第14组另将磨损直接反馈到服务时间线。'},
    ]


def render(report):
    from hbf_hbm_report import render_report
    return render_report(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'hbf_hbm')
    parser.add_argument('--groups',default=','.join(str(n) for n in range(1,21)),help='comma-separated group numbers')
    parser.add_argument('--render-only',action='store_true',help='render already saved group results without rerunning simulations')
    parser.add_argument('--reuse-groups',default='',help='reuse selected saved groups only when their physical-execution version matches')
    parser.add_argument('--resume-study',action='store_true',help='reuse completed study points with matching parameters from this output directory; only use if their code remains applicable')
    parser.add_argument('--workers',type=int,default=1,help='independent named-model worker processes; adaptive SLO search stays serial')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.workers<1 or args.workers>8: parser.error('--workers must be in [1, 8]')
    if args.render_only:
        report=json.loads((args.output_dir/'results.json').read_text(encoding='utf-8'))
        (args.output_dir/'report.html').write_text(render(report),encoding='utf-8')
        return 0
    selected={int(x) for x in args.groups.split(',')}
    if not selected or not selected <= set(range(1,21)):
        parser.error('--groups must contain numbers from 1 through 20')
    reused={int(x) for x in args.reuse_groups.split(',') if x.strip()}
    if not reused<=selected: parser.error('--reuse-groups must be selected in --groups')
    groups=[]
    for number in sorted(reused):
        files=list(args.output_dir.glob(f'{number:02d}_*.json'))
        # Partial/supplemental point collections are not canonical group files.
        # Extended model groups legitimately contain "models" in their name.
        files=[p for p in files if 'partial' not in p.stem and
               isinstance((saved:=json.loads(p.read_text(encoding='utf-8'))),dict) and saved.get('id')==p.stem]
        if len(files)!=1: parser.error(f'expected one saved result for group {number}, found {len(files)}')
        cached=json.loads(files[0].read_text(encoding='utf-8'))
        if int(cached['id'][:2])!=number: parser.error(f'saved group {number} has wrong id')
        if cached.get('point_result_version') != POINT_RESULT_VERSION:
            parser.error(f'saved group {number} uses an older or unknown physical-execution version; rerun it without --reuse-groups')
        groups.append(cached)
    active=selected-reused
    for number,func in ((1,ideal_group),(2,fixed_workload_group),(3,capacity_group),(8,weights_group)):
        if number in active:
            print(f'Running group {number}',flush=True)
            g=func(); groups.append(g)
            (args.output_dir/(g['id']+'.json')).write_text(json.dumps(g,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if 4 in active:
        import hbf_hbm_session_experiments as sessions
        session_report=sessions.run_session_experiments()
        groups.append(session_report)
    if active & {5,6,7}:
        import hbf_hbm_media_experiments as media
        groups.extend(g for g in media.run_media_experiments()['groups'] if int(g['id'][:2]) in active)
    if 9 in active:
        print('Running full operator placement/model/long-context/tier/SLO study',flush=True)
        from hbf_hbm_serving_study import run_serving_study
        groups.append(run_serving_study(point_cache_dir=args.output_dir/'study_points',resume=args.resume_study,workers=args.workers))
    if 10 in active:
        from hbf_hbm_traffic_study import run_traffic_study
        groups.append(run_traffic_study())
    if 11 in active:
        from hbf_hbm_provisioning import run_provisioning_study
        groups.append(run_provisioning_study())
    if 19 in active:
        from hbf_hbm_decode_study import run_decode_study
        groups.append(run_decode_study(workers=args.workers,point_dir=args.output_dir/'decode_points',resume=args.resume_study))
    extensions={12:('hbf_hbm_ablation_paths','run_group'),13:('hbf_hbm_agent_study','run_agent_study'),
        14:('hbf_hbm_wear_study','run_wear_study'),15:('hbf_hbm_system_study','run_system_study'),
        16:('hbf_hbm_sustained_study','run_sustained_study'),17:('hbf_hbm_extended_models','run_extended_models_study'),
        18:('hbf_hbm_bandwidth_study','run_bandwidth_study'),
        20:('hbf_hbm_bank_group_study','run_bank_group_study')}
    import importlib
    for number,(module,function) in extensions.items():
        if number in active:
            print(f'Running group {number}',flush=True)
            kwargs=({'workers':args.workers,'point_cache_dir':args.output_dir/'extended_model_points',
                     'resume':args.resume_study} if number==17 else {})
            if number==20:
                kwargs={'workers':args.workers,'point_dir':args.output_dir/'bank_group_points'}
            g=getattr(importlib.import_module(module),function)(**kwargs); groups.append(g)
            (args.output_dir/(g['id']+'.json')).write_text(json.dumps(g,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    groups.sort(key=lambda g:g['id'])
    for g in groups:g['title']=TITLES[int(g['id'][:2])]
    report={'schema_version':2,'point_result_version':POINT_RESULT_VERSION,
        'scope':'mechanism microbenchmarks plus full-operator placement/model/tier studies and explicitly conditional economics',
        'all_selected_checks_passed':all(g['status']=='passed' for g in groups),'groups':groups,
        'external_prediction_accuracy_validated':False,'overall_research_program_complete':False,
        'external_calibration_required':False,'output_directory':args.output_dir.as_posix(),
        'audit_fixes':AUDIT_FIXES,'audit_completion':completion_matrix(groups),'remaining_research_scope':EXCLUDED_SCOPE}
    report['memory_core_coverage']=memory_core_coverage(groups)
    if report['memory_core_coverage']['full_serving_rows']:
        report['all_selected_checks_passed'] &= report['memory_core_coverage']['all_full_serving_hbm_verified']
        report['all_selected_checks_passed'] &= not report['memory_core_coverage']['wrong_nand_core_row_ids']
    report['internal_experiment_matrix_complete']=(selected==set(range(1,21)) and
        report['all_selected_checks_passed'] and all(r['status']=='已完成所述实验' for r in report['audit_completion']))
    report['model_execution_coverage']=model_execution_coverage(groups)
    from hbf_hbm_economics import run_economics_sensitivity
    sources=[r for g in groups for r in g.get('rows',[]) if r.get('mode')=='full_operator_serving'
             and (r['id'].startswith('placement/') or r['id'].startswith('fixed-2048-1-'))]
    report['economics']=run_economics_sensitivity(sources)
    report['checklist_progress']=checklist_progress(groups,report['economics'])
    (args.output_dir/'economics.json').write_text(json.dumps(report['economics'],ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if set(range(1,9)) <= selected:
        if set(range(1,9))<=reused and all((args.output_dir/name).exists() for name in ('acceptance.json','profiles.json')):
            a=json.loads((args.output_dir/'acceptance.json').read_text(encoding='utf-8'))
            p=json.loads((args.output_dir/'profiles.json').read_text(encoding='utf-8'))
            if any(saved.get('point_result_version') != POINT_RESULT_VERSION for saved in (a, p)):
                parser.error('saved acceptance/profiles use an older or unknown physical-execution version; rerun groups 1-8 without --reuse-groups')
        else:
            print('Running backend acceptance and paper-parameterized profiles',flush=True)
            a=acceptance.run_acceptance()
            a['point_result_version']=POINT_RESULT_VERSION
            (args.output_dir/'acceptance.json').write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            p=profiles.build_report()
            p['point_result_version']=POINT_RESULT_VERSION
            (args.output_dir/'profiles.json').write_text(json.dumps(p,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        report['backend_acceptance_passed']=a['passed']
        report['all_selected_checks_passed'] &= a['passed']
    report['internal_experiment_matrix_complete'] &= report['all_selected_checks_passed']
    for g in groups:
        g['point_result_version'] = POINT_RESULT_VERSION
        (args.output_dir/(g['id']+'.json')).write_text(json.dumps(g,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Keep the combined machine-readable bundle compact; individual group
    # files above retain indentation for inspection and review.
    (args.output_dir/'results.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    (args.output_dir/'report.html').write_text(render(report),encoding='utf-8')
    print(json.dumps({'passed':report['all_selected_checks_passed'],'groups':[(g['id'],g['status'],len(g.get('rows',[]))) for g in groups]}))
    return 0 if report['all_selected_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
