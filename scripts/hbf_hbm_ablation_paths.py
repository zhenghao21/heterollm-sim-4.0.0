"""Internal full-serving ablations and fixed aggregate-link-budget path study.

Run with .venv/Scripts/python.exe -B scripts/hbf_hbm_ablation_paths.py.
Every timing row is produced by serving_row/run_scenario, never a scaled TPS.
The intentionally ideal media and static QoS partitions are analytical inputs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.cost_models import HBMProfile, HostMemoryProfile
from heterollm_sim.ir import LinkSpec, PortSpec
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.physical_contract import require_physical_memory_config, reconfigure_physical_config
from hbf_hbm_serving_study import build_study_scenario
from hbf_hbm_experiments import group, serving_row


MIB = 1024 ** 2
ABLATIONS = {
    "ideal": {},
    "read_latency": {"page_read_ns": 4000.},
    "write_restriction": {"host_write_bandwidth_gb_s": 1., "page_program_ns": 75000.},
    "finite_concurrency": {"parallel_units": 1, "max_outstanding_requests": 1},
    "read_write": {"page_read_ns": 4000., "host_write_bandwidth_gb_s": 1., "page_program_ns": 75000.},
    "read_write_finite": {"page_read_ns": 4000., "host_write_bandwidth_gb_s": 1.,
                          "page_program_ns": 75000., "parallel_units": 1, "max_outstanding_requests": 1},
}


def build_ablation_scenario(medium="hbf", ablation="ideal", *, batch=4, prompt=128, output=8):
    """A/B differ only in their dedicated KV medium, with common local HBM.

    Both media are GPU-addressable, equal-capacity, 64-byte-granularity,
    64-GB/s endpoints. The link is a view of that endpoint, not a second bus.
    HBM has zero bank/burst latency; HBF has non-bottleneck internal bandwidth.
    The finite-only ablation may have no effect at zero media latency; its
    interaction with nonzero media latency is measured separately.
    """
    if medium not in {"hbm", "hbf"} or ablation not in ABLATIONS:
        raise ValueError("unsupported medium or ablation")
    if medium == "hbm" and ablation != "ideal":
        raise ValueError("HBM is the unchanged ideal comparator")
    s = build_study_scenario("B_all_kv_hbf", batch=batch, prompt=prompt, output=output)
    ident = "hbf0" if medium == "hbf" else "hbm_kv"
    physical = resolve_physical_config("research_" + medium + "_ablation",
                                       overrides=ABLATIONS[ablation])
    capacity = physical.capacity_bytes
    bandwidth = (physical.host_bandwidth_gb_s if medium == "hbf"
                 else physical.physical_interface_bandwidth_gb_s)
    if medium == "hbf":
        profile = HostMemoryProfile(bandwidth_gb_s=bandwidth, efficiency=1.,
            read_bandwidth_gb_s=bandwidth, write_bandwidth_gb_s=physical.host_write_bandwidth_gb_s,
            resource_id=ident+".memory", transaction_bytes=64,
            # Typed HBF profile requires positive declarations; physical
            # execution below owns timing and permits the ideal zero latency.
            read_latency_ns=max(1e-12,physical.page_read_ns), write_latency_ns=max(1e-12,physical.page_program_ns),
            parallel_lanes=physical.parallel_units, max_outstanding_requests=physical.max_outstanding_requests)
        profile_kind = "host_memory"
    else:
        profile = HBMProfile(bandwidth_gb_s=bandwidth, efficiency=1., read_bandwidth_gb_s=bandwidth,
            write_bandwidth_gb_s=bandwidth, resource_id=ident+".memory", transaction_bytes=64,
            max_outstanding_requests=4096, parallel_lanes=64)
        profile_kind = "hbm"
    source = s.hardware.get_component("hbf0")
    endpoint = replace(source, component_id=ident, kind=medium, cost_profile_id="ablation-kv",
        capacity_bytes=capacity, bandwidth_gbps=bandwidth*8, read_bandwidth_gbps=bandwidth*8,
        write_bandwidth_gbps=profile.write_bandwidth_gb_s*8,
        ports=tuple(replace(p,bandwidth_gbps=bandwidth*8) for p in source.ports),
        metadata={"physical_memory_config":physical,"memory_service_owner":ident+".memory",
            "access_mode":"memory","writable":True,
            "read_latency_ns":profile.read_latency_ns,"write_latency_ns":profile.write_latency_ns,
            "transfer_granularity_bytes":64,"max_outstanding_requests":profile.max_outstanding_requests,
            "parallel_lanes":profile.parallel_lanes,
            "write_buffer_bytes":0})
    if medium == "hbm":
        endpoint=replace(endpoint,ports=tuple(replace(p,protocol="HBM",role="device",version="3.0") for p in endpoint.ports))
    components = tuple(endpoint if c.component_id=="hbf0" else
        replace(c,ports=tuple(replace(p,protocol="HBM",version="3.0") if p.port_id=="ucie_hbf" else p
                             for p in c.ports)) if medium=="hbm" and c.component_id=="gpu0" else c
        for c in s.hardware.components)
    links = tuple(replace(l,target_component=ident,bandwidth_gbps=bandwidth*8,latency_ns=0.,
                         protocol="HBM" if medium=="hbm" else l.protocol,
                         version="3.0" if medium=="hbm" else l.version,
                         metadata={"service_ref":ident+".access"}) if l.target_component=="hbf0" else l
                  for l in s.hardware.links)
    profiles = {kind:dict(registry) for kind,registry in s.component_profiles.items()}
    profiles.setdefault(profile_kind,{})["ablation-kv"] = profile
    metadata = copy.deepcopy(s.placement.metadata)
    metadata["llama_backend_memory"] = {ident:{"device_id":"gpu0","buffer_kind":medium.upper(),"access":"direct"}}
    metadata["control_plane"]["policy"]["options"]["kv_cache_target"] = ident
    return replace(s,name=f"full-ablation-{medium}-{ablation}",
        hardware=replace(s.hardware,components=components,links=links),component_profiles=profiles,
        placement=replace(s.placement,metadata=metadata,kv_policy=replace(s.placement.kv_policy,cache_component=ident)))


def run_ablation_case(medium, ablation, **parameters):
    s = build_ablation_scenario(medium,ablation,**parameters)
    row = serving_row(s,f"ablation/{medium}/{ablation}",placement="A_hbm_ideal" if medium=="hbm" else "B_hbf_"+ablation)
    endpoint = s.hardware.get_component(s.placement.kv_policy.cache_component)
    row["parameters"] = {"medium":medium,"ablation":ablation,**parameters}
    row["ablation_contract"] = {"kv_capacity_bytes":endpoint.capacity_bytes,
        "physical_config":asdict(require_physical_memory_config(endpoint)),
        "changed_fields_from_hbf_ideal":ABLATIONS[ablation],
        "common_weights_and_workspace":"hbm0", "extra_link_serialization":False,
        "scope":"Ideal 64-byte KV media equivalence, not realistic NAND pages or production HBM calibration"}
    return row


def build_path_scenario(path_mode="direct", *, total_link_gb_s=64., weight_share=.5,
                        mixed_weights=True, prompt=128, output=16, interval_ns=500_000.,
                        prefetch_distance=16, priority_preemption=True,
                        slo_ttft_ns=200_000., slo_tbt_ns=1_000_000.):
    """Execute real KV copies with a fixed sum of independent link rates.

    Direct uses one link; relay uses two links and a real staging-memory
    write/read per KV copy. Dual dedicates one path to direct HBF weight
    reads and the two-hop staging path to KV copies. This is static spatial
    bandwidth partitioning, not adaptive routing, packet QoS or striping.
    """
    if path_mode not in {"direct", "relay", "dual"}:
        raise ValueError("path_mode must be direct, relay or dual")
    if not math.isfinite(total_link_gb_s) or total_link_gb_s <= 0:
        raise ValueError("total_link_gb_s must be finite and positive")
    if not 0 < weight_share < 1:
        raise ValueError("weight_share must be strictly between zero and one")
    s=build_study_scenario("C_hot_cold",prompt=prompt,batch=1,output=output,request_count=2,
        interval_ns=interval_ns,priority_preemption=priority_preemption,prefetch_distance=prefetch_distance,
        slo_ttft_ns=slo_ttft_ns,slo_tbt_ns=slo_tbt_ns)
    profiles={kind:dict(registry) for kind,registry in s.component_profiles.items()}
    source=s.hardware.get_component("hostmem0")
    relay_profile=replace(s.resolve_component_profile("hostmem0"),resource_id="relaymem.memory",
        bandwidth_gb_s=128.,read_bandwidth_gb_s=128.,write_bandwidth_gb_s=128.,efficiency=1.)
    profiles["host_memory"]["path-relay"] = relay_profile
    relay=replace(source,component_id="relaymem",cost_profile_id="path-relay",capacity_bytes=16*MIB,
        package_id="package0",die_id="path-relay-die",
        bandwidth_gbps=1024.,read_bandwidth_gbps=1024.,write_bandwidth_gbps=1024.,
        ports=(),metadata={"memory_service_owner":"relaymem.memory",
            "physical_memory_config":resolve_physical_config("research_hbm_path_relay")})
    components={c.component_id:c for c in (*s.hardware.components,relay)}
    metadata=copy.deepcopy(s.placement.metadata)
    if mixed_weights:
        # Paged offload and directly addressable weights have distinct IR
        # memory classes. Use two explicit fixed physical partitions rather
        # than silently assigning one component incompatible roles.
        hbf=components["hbf0"]
        physical=require_physical_memory_config(hbf)
        partition=reconfigure_physical_config(physical, overrides={
            "capacity_bytes": hbf.capacity_bytes//2, "channels": physical.channels//2,
            "parallel_units": physical.parallel_units//2,
            "host_bandwidth_gb_s": physical.host_bandwidth_gb_s/2,
            "host_read_bandwidth_gb_s": physical.host_read_bandwidth_gb_s/2,
            "host_write_bandwidth_gb_s": physical.host_write_bandwidth_gb_s/2,
            "internal_bandwidth_gb_s": physical.internal_bandwidth_gb_s/2})
        hbf=replace(hbf,capacity_bytes=partition.capacity_bytes,bandwidth_gbps=hbf.bandwidth_gbps/2,
            read_bandwidth_gbps=hbf.read_bandwidth_gbps/2,write_bandwidth_gbps=hbf.write_bandwidth_gbps/2,
            metadata={**{k:v for k,v in hbf.metadata.items() if k!="memory_service"},
                      "physical_memory_config":partition})
        components["hbf0"]=hbf
        weights_profile=replace(profiles["host_memory"]["calibrated-hbf"],resource_id="hbf_weights.memory",
            bandwidth_gb_s=partition.host_bandwidth_gb_s,read_bandwidth_gb_s=partition.host_read_bandwidth_gb_s,
            write_bandwidth_gb_s=partition.host_write_bandwidth_gb_s,parallel_lanes=partition.parallel_units)
        profiles["host_memory"]["path-weights"]=weights_profile
        components["hbf_weights"]=replace(hbf,component_id="hbf_weights",cost_profile_id="path-weights",ports=(),
            metadata={**hbf.metadata,"memory_service_owner":"hbf_weights.memory","access_mode":"memory",
                "parallel_lanes":weights_profile.parallel_lanes})
        options=metadata["control_plane"]["policy"]["options"]
        options["weight_tensor_targets"]={key:"hbf_weights" for key in options["weight_tensor_targets"]}
        metadata["llama_backend_memory"]={"hbf_weights":{"device_id":"gpu0","buffer_kind":"HBF","access":"direct"}}
    # Remove the inherited HBF edge; no hidden high-rate alternative survives.
    links=[replace(l,latency_ns=0.,metadata={**l.metadata,"service_ref":"hbm0.access"}) if l.link_id=="gpu-hbm0" else l
           for l in s.hardware.links if "hbf0" not in (l.source_component,l.target_component)]
    edges=[]
    if path_mode=="direct":
        edges=[("path-direct","hbf0","gpu0",total_link_gb_s)]
    elif path_mode=="relay":
        edges=[("path-relay-in","hbf0","relaymem",total_link_gb_s/2),
               ("path-relay-out","relaymem","gpu0",total_link_gb_s/2)]
    else:
        edges=[("path-direct","hbf0","gpu0",total_link_gb_s*weight_share),
               ("path-relay-in","hbf0","relaymem",total_link_gb_s*(1-weight_share)/2),
               ("path-relay-out","relaymem","gpu0",total_link_gb_s*(1-weight_share)/2)]
    logical_edges=list(edges)
    if mixed_weights:
        # A second logical endpoint can share one physical fabric queue.
        # Count this resource once in the fixed bandwidth budget.
        weight_edge=next(edge for edge in edges if edge[1]=="hbf0")
        logical_edges.append((weight_edge[0]+"-weights","hbf_weights",weight_edge[2],weight_edge[3]))
    for name,source_id,target_id,bandwidth in logical_edges:
        for ident in (source_id,target_id):
            component=components[ident]
            components[ident]=replace(component,ports=component.ports+(PortSpec(name,"UCIe","endpoint",
                version="1.1",lanes=64,bandwidth_gbps=8*bandwidth,payload="streaming"),))
        links.append(LinkSpec(name,source_id,name,target_id,name,"UCIe",version="1.1",lanes=64,
            bandwidth_gbps=8*bandwidth,latency_ns=100.,payload="streaming",
            metadata={"shared_bidirectional":True,"bandwidth_resource_id":"link."+name.removesuffix("-weights")}))
    policy=replace(s.placement.kv_policy,relay_components=("relaymem",) if path_mode!="direct" else ())
    contract={"path_mode":path_mode,"total_link_budget_gb_s":total_link_gb_s,
        "links":[{"id":name,"source":source_id,"target":target_id,"bandwidth_gb_s":bw}
                 for name,source_id,target_id,bw in edges],
        "applied_sum_link_gb_s":sum(edge[3] for edge in edges),
        "weight_share":weight_share if path_mode=="dual" else None,
        "request_priority_policy":"later request priority=10, swap preemption" if priority_preemption else "both priority=0, FIFO without preemption",
        "requests":[{"id":r.request_id,"arrival_ns":r.arrival_ns,"priority":r.priority} for r in s.workload.requests],
        "mixed_hbf_weights":mixed_weights,"kv_relay_components":list(policy.relay_components),
        "hbf_partition_scope":"When mixed, HBF weights/KV own fixed equal capacity, bandwidth and effective-lane partitions; aggregate media inventory is unchanged across paths. Separate physical controllers; not dynamically shared NAND queues.",
        "relay_capacity_bytes":relay.capacity_bytes,
        "budget_scope":"Sum of distinct bidirectional path-link bandwidths; both directions share each resource. Common GPU-HBM and host-control links are unchanged and excluded.",
        "route_scope":"KV swaps physically write then read relay memory. GPU HBF weight reads use the topology path as a direct remote-memory pipelined resource envelope, with no host staging copy.",
        "qos_scope":"Static physical link partition between direct weights and relay KV; request priority preemption is explicit. No dynamic arbitration or packet-level QoS claim."}
    return replace(s,name=f"full-path-{path_mode}",hardware=replace(s.hardware,components=tuple(components.values()),links=tuple(links)),
        component_profiles=profiles,placement=replace(s.placement,metadata=metadata,kv_policy=policy)),contract


def run_path_case(path_mode, **parameters):
    s,contract=build_path_scenario(path_mode,**parameters)
    suffix=f"/w{parameters.get('weight_share',.5):g}" if path_mode=="dual" else ""
    if not parameters.get("priority_preemption",True):
        suffix+="/fifo"
    observed={}
    def capture(result):
        transfer=result.serving.runtime_kernel_metrics.get("paged_kv_transfers",{})
        observed["physical_transfers"]=list(transfer.get("events",()))
        observed["physical_counters"]=dict(transfer.get("counters",{}))
        # Actual GPU model stages, not compiler estimates or an envelope-only
        # proxy. The row's utilization supplies the executed link demands.
        observed["gpu_operator_stage_count"]=sum(1 for b in result.serving.batches
            for stage in b.cost.metadata.get("execution_stage_schedule",())
            if stage.get("component_id")=="gpu0" and stage.get("invocation_group_id"))
    row=serving_row(s,f"paths/{path_mode}{suffix}",placement="C_paged_"+path_mode,result_observer=capture)
    row["parameters"]={"path_mode":path_mode,**parameters}
    row["path_contract"]=contract
    router=TopologyRouter(s.hardware)
    row["resolved_routes"]={"weight_read":[h.link_id for h in router.route("hbf_weights" if parameters.get("mixed_weights",True) else "hbm0","gpu0",65536)],
        "kv_restore_segments":[[h.link_id for h in router.route(a,b,8192)] for a,b in
            zip(("hbf0",)+s.placement.kv_policy.relay_components,
                s.placement.kv_policy.relay_components+("hbm0",))]}
    events=observed["physical_transfers"]
    phases=[phase for event in events for phase in event.get("phases",())]
    row["path_observations"]={**observed,"migration_events":len(events),
        "migration_phase_kinds":sorted({phase.get("kind") for phase in phases}),
        "migration_link_bytes":{link_id:sum(int(p.get("bytes",0)) for p in phases
            if p.get("kind")=="link" and p.get("link_id")==link_id)
            for link_id in sorted({p.get("link_id") for p in phases if p.get("kind")=="link"})},
        "relay_write_bytes":sum(int(p.get("logical_bytes",0)) for p in phases if p.get("kind")=="relay_write"),
        "relay_read_bytes":sum(int(p.get("logical_bytes",0)) for p in phases if p.get("kind")=="relay_read"),
        "executed_path_resources":{key:value for key,value in row["resources"]["resource_utilization"].items()
            if key.startswith("link.path-") and value>0}}
    row["qos_observations"]={"later_request_ttft_ns":row["request_metrics"][1]["ttft_ns"],
        "later_request_e2e_ns":row["request_metrics"][1]["finish_ns"]-row["request_metrics"][1]["arrival_ns"],
        "earlier_request_e2e_ns":row["request_metrics"][0]["finish_ns"]-row["request_metrics"][0]["arrival_ns"],
        "qualified_requests":row["qualified_requests"],"slo":row["slo"],
        "scope":"Two-request deterministic contention probe; per-request latency and configured SLO, not statistical fairness or sustainable capacity"}
    return row


def run_group(*, progress=False):
    rows=[]
    for medium,name in [("hbm","ideal")]+[("hbf",name) for name in ABLATIONS]:
        if progress: print("[12] ablation",medium,name,flush=True)
        rows.append(run_ablation_case(medium,name))
    baseline=rows[0]
    for row in rows:
        row["relative_to_ideal_hbm"]={"tps_ratio":row["node_tps"]/baseline["node_tps"],
            "makespan_ratio":row["makespan_ns"]/baseline["makespan_ns"]}
    path_rows=[]
    matrix=[("direct",{}),("relay",{}),
            *( ("dual",{"weight_share":share}) for share in (.25,.5,.75)),
            ("dual",{"weight_share":.5,"priority_preemption":False})]
    for mode,parameters in matrix:
        if progress: print("[12] path",mode,parameters,flush=True)
        row=run_path_case(mode,**parameters)
        path_rows.append(row);rows.append(row)
    by_ablation={r["parameters"]["ablation"]:r for r in rows if r["parameters"].get("medium")=="hbf"}
    ideal=by_ablation["ideal"]
    relative_error=abs(ideal["makespan_ns"]-baseline["makespan_ns"])/baseline["makespan_ns"]
    for row in path_rows:
        row["relative_to_shared_direct"]={"node_tps_ratio":row["node_tps"]/path_rows[0]["node_tps"],
            "tbt_p99_ratio":row["tbt_ns"]["p99"]/path_rows[0]["tbt_ns"]["p99"]}
    checks={
        "all_rows_execute_full_serving":all(r["mode"]=="full_operator_serving" for r in rows),
        "all_requests_complete":all(r["completed_requests"]==r["request_count"] and not r["rejected_requests"] for r in rows),
        "ideal_complete_service_equivalence":relative_error<1e-7,
        "read_latency_changes_service":by_ablation["read_latency"]["makespan_ns"]>ideal["makespan_ns"],
        "write_restriction_changes_service":by_ablation["write_restriction"]["makespan_ns"]>ideal["makespan_ns"],
        "finite_concurrency_interaction_changes_service":by_ablation["read_write_finite"]["makespan_ns"]>by_ablation["read_write"]["makespan_ns"],
        "same_aggregate_link_budget":all(math.isclose(r["path_contract"]["applied_sum_link_gb_s"],64.) for r in path_rows),
        "gpu_model_operators_executed":all(r["path_observations"]["gpu_operator_stage_count"]>0 for r in path_rows),
        "priority_cases_execute_kv_migration":all(r["path_observations"]["migration_events"]>0 for r in path_rows if r["parameters"].get("priority_preemption",True)),
        "relay_paid_real_writes_and_reads":all(r["path_observations"]["relay_write_bytes"]>0 and
            r["path_observations"]["relay_read_bytes"]>0 for r in path_rows
            if r["parameters"]["path_mode"]!="direct" and r["parameters"].get("priority_preemption",True)),
        "direct_has_no_relay_memory_work":path_rows[0]["path_observations"]["relay_write_bytes"]==0,
        "dual_both_physical_paths_execute":all(r["path_observations"]["executed_path_resources"].get("link.path-direct",0)>0
            and r["path_observations"]["migration_link_bytes"].get("path-relay-in",0)>0
            and r["path_observations"]["migration_link_bytes"].get("path-relay-out",0)>0
            for r in path_rows if r["parameters"]["path_mode"]=="dual" and r["parameters"].get("priority_preemption",True)),
        "no_premature_prefetch_consumers":all(r["prefetch_overlap"]["premature_consumer_overlap_ns"]==0 for r in path_rows),
        "static_link_partition_changes_real_service":path_rows[2]["makespan_ns"]!=path_rows[4]["makespan_ns"],
        "priority_policy_changes_later_request_latency":path_rows[3]["qos_observations"]["later_request_ttft_ns"]<
            path_rows[5]["qos_observations"]["later_request_ttft_ns"],
    }
    return group("12_ablation_paths","⑫ 完整服务消融与链路预算路径",rows,checks,
        ["等效KV介质的完整服务基线；独立及累积读延迟/写限制/并发限制消融",
         "相同链路总带宽预算下direct/relay/dual完整服务；实际KV中继写读与双路径执行",
         "固定权重/KV物理分区与静态链路份额QoS，显式请求优先级对照"],
        ["有限小模型合成轨迹；解析理想参数，不声称外部硬件校准。",
         "理想消融固定64字节粒度，不把现实NAND页大小效应混入独立因素；finite-only零延迟时可以无收益变化。",
         "路径预算是独立链路带宽之和，不等于封装成本；双跳每段均计入，方向共享每个链路。",
         "relay的KV搬移真实落地中继；权重仍是经拓扑的远端直读资源包络，没有假设权重中继缓存命中。",
         "QoS为静态空间分流与请求优先级，不是动态队列公平性或全局最优路由证明。"],
        ideal_equivalence={"relative_makespan_error":relative_error,"tolerance":1e-7},
        experiment_scope="full_operator_serving",external_calibration_required=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/hbf_hbm/12_ablation_paths.json")
    args=parser.parse_args()
    result=run_group(progress=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({"status":result["status"],"checks":result["checks"],"rows":len(result["rows"])},ensure_ascii=False))
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
