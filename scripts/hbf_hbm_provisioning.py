"""Re-execute full serving for a bounded, explicitly hypothetical package search."""
from __future__ import annotations

from dataclasses import replace
import math

from hbf_hbm_economics import conditional_node_cost
from hbf_hbm_serving_study import build_study_scenario
from calibrated_hbf_profiles import _hbm_physical_config
from heterollm_sim.physical_contract import require_physical_memory_config, reconfigure_physical_config
from heterollm_sim.planner import ScenarioValidationError

HBM_CAPACITY = 36_000_000_000
HBF_CAPACITY = 512_000_000_000
BUDGETS = {"gpu_count": 1, "memory_area_mm2": 400.,
           "node_capex_usd": 15500., "total_memory_capacity_bytes": 1_600_000_000_000}


def build_package_scenario(hbm_packages, hbf_packages, architecture, *, model_id="llama3_1-70b", output=8):
    """One physical-core device per tier; scale capacity/bandwidth/concurrency together."""
    if not isinstance(hbm_packages, int) or not isinstance(hbf_packages, int) or min(hbm_packages, hbf_packages)<0:
        raise ValueError("package counts must be nonnegative integers")
    if not hbm_packages:
        raise ValueError("this policy requires HBM for activation/workspace and active KV")
    if not hbf_packages and architecture != "A_hbm_only":
        raise ValueError("the requested placement requires at least one HBF package")
    s = build_study_scenario(architecture, model_id=model_id, prompt=2048, batch=1, output=output,
                             hbm_capacity_bytes=HBM_CAPACITY*hbm_packages,
                             hbf_capacity_bytes=HBF_CAPACITY*max(1,hbf_packages),
                             hbm_clock_ns=0.32)
    counts={"hbm0":hbm_packages,"hbf0":hbf_packages}
    components=[]
    package_profiles={}
    for c in s.hardware.components:
        if c.component_id not in counts:
            components.append(c); continue
        count=counts[c.component_id]
        if not count:
            continue
        physical=require_physical_memory_config(c)
        profile=s.resolve_component_profile(c)
        if c.component_id=="hbm0":
            outstanding = physical.max_outstanding_requests * count
            physical = _hbm_physical_config(capacity_bytes=c.capacity_bytes,
                read_gb_s=physical.directional_bandwidth_gb_s("read")*count,
                write_gb_s=physical.directional_bandwidth_gb_s("write")*count,
                clock_ns=physical.command_clock_ns,
                interface_input_scope=physical.metadata.get("interface_input_scope"),
                stacks=count, profile_id=physical.metadata["physical_config_profile_id"])
            physical=reconfigure_physical_config(physical,
                overrides={"max_outstanding_requests": outstanding})
            shared_bw=physical.physical_interface_bandwidth_gb_s
            read_bw=physical.directional_bandwidth_gb_s("read")
            write_bw=physical.directional_bandwidth_gb_s("write")
        else:
            physical=reconfigure_physical_config(physical, overrides={
                "channels": physical.channels*count, "parallel_units": physical.parallel_units*count,
                "host_channels": physical.host_channels*count,
                "host_bandwidth_gb_s": physical.host_bandwidth_gb_s*count,
                "host_read_bandwidth_gb_s": physical.host_bandwidth_for("read")*count,
                "host_write_bandwidth_gb_s": physical.host_bandwidth_for("write")*count,
                "max_outstanding_requests": physical.max_outstanding_requests*count})
            shared_bw=physical.host_bandwidth_gb_s
            read_bw=physical.host_bandwidth_for("read")
            write_bw=physical.host_bandwidth_for("write")
        service_lanes = physical.lane_count if c.component_id=="hbm0" else physical.parallel_units
        package_profiles[c.component_id]=replace(profile, bandwidth_gb_s=shared_bw,
            read_bandwidth_gb_s=read_bw, write_bandwidth_gb_s=write_bw,
            max_outstanding_requests=physical.max_outstanding_requests,
            parallel_lanes=service_lanes,
            transaction_bytes=physical.burst_bytes if c.component_id=="hbm0" else profile.transaction_bytes)
        components.append(replace(c, bandwidth_gbps=shared_bw*8,
            read_bandwidth_gbps=read_bw*8, write_bandwidth_gbps=write_bw*8,
            ports=tuple(replace(p,bandwidth_gbps=shared_bw*8) for p in c.ports),
            metadata={**{k:v for k,v in c.metadata.items() if k!='memory_service'},
                "physical_memory_config":physical,"max_outstanding_requests":physical.max_outstanding_requests,
                "parallel_lanes":service_lanes}))
    ids={c.component_id for c in components}
    links=[]; gpu_ports={}
    for link in s.hardware.links:
        if link.source_component not in ids or link.target_component not in ids:
            continue
        memory=next((x for x in (link.source_component,link.target_component) if x in counts),None)
        if memory:
            link=replace(link,bandwidth_gbps=package_profiles[memory].bandwidth_gb_s*8)
            gpu_ports[link.source_port if link.source_component=="gpu0" else link.target_port]=link.bandwidth_gbps
        links.append(link)
    components=[replace(c,ports=tuple(replace(p,bandwidth_gbps=gpu_ports[p.port_id]) if p.port_id in gpu_ports else p
        for p in c.ports)) if c.component_id=="gpu0" else c for c in components]
    registry={k:dict(v) for k,v in s.component_profiles.items()}
    hbm=s.hardware.get_component("hbm0")
    registry["hbm"][hbm.cost_profile_id]=package_profiles["hbm0"]
    if hbf_packages:
        flash=s.hardware.get_component("hbf0")
        registry[s.component_profile_kind(flash)][flash.cost_profile_id]=package_profiles["hbf0"]
    return replace(s,hardware=replace(s.hardware,components=tuple(components),links=tuple(links)),component_profiles=registry)


def budget_feasibility(cost):
    return {"fixed_gpu_count":cost["gpu_count"]==BUDGETS["gpu_count"],
        "area_ceiling":cost["memory_package_area_mm2"]<=BUDGETS["memory_area_mm2"],
        "cost_ceiling":cost["capex_usd"]<=BUDGETS["node_capex_usd"],
        "capacity_ceiling":cost["hbm_capacity_bytes"]+cost["hbf_capacity_bytes"]<=BUDGETS["total_memory_capacity_bytes"]}


def run_provisioning_study(*, model_id="llama3_1-70b", progress=True):
    from hbf_hbm_experiments import serving_row, group
    rows=[]
    for hbm in (4,3,2,1,0):
        hbf=4-hbm
        architectures=("A_hbm_only",) if not hbf else ("B_all_kv_hbf","D_hbf_weights")
        for architecture in architectures:
            row_id=f"provision/{hbm}HBM-{hbf}HBF/{architecture}"
            if progress: print("[11_provisioning] "+row_id,flush=True)
            cost=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,active_output_tps=0.)
            configuration={"hbm_packages":hbm,"hbf_packages":hbf,"model_id":model_id,
                "hbm_capacity_bytes":hbm*HBM_CAPACITY,"hbf_capacity_bytes":hbf*HBF_CAPACITY,
                "hbm_peak_read_gb_s":1600.*hbm,"hbf_peak_read_gb_s":1600.*hbf,
                "hbf_peak_write_gb_s":16.*hbf,"hbf_parallel_units":64*hbf}
            if not hbm:
                row={"id":row_id,"mode":"configuration_rejected","placement":architecture,
                    "reason":"This policy requires HBM staging/activation/workspace; this is not a proof that every zero-HBM architecture is impossible."}
            else:
                s=build_package_scenario(hbm,hbf,architecture,model_id=model_id)
                try:
                    row=serving_row(s,row_id,placement=architecture)
                except ScenarioValidationError as error:
                    row={"id":row_id,"mode":"configuration_rejected","placement":architecture,"reason":str(error)}
            if row["mode"]=="full_operator_serving":
                cost=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,active_output_tps=row["node_tps"])
                cost["formula_scope"]="Package counts match this re-executed serving row; price/area remain hypothetical inputs, not measured node characteristics."
                row["conditional_5yr_cost"]=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,years=5.,active_output_tps=row["node_tps"])
                row["conditional_slo_cost"]=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,active_output_tps=row["goodput_tps"])
                for field in ("conditional_5yr_cost","conditional_slo_cost"):
                    row[field]["formula_scope"]=cost["formula_scope"]
            row.update(package_configuration=configuration,conditional_cost=cost,budget_checks=budget_feasibility(cost))
            row["jointly_feasible"]=all(row["budget_checks"].values()) and row["mode"]=="full_operator_serving" and row.get("qualified_requests")==row.get("request_count")
            rows.append(row)
    candidates=[r for r in rows if r["jointly_feasible"] and r.get("goodput_tps",0)>0]
    selected=min(candidates,key=lambda r:r["conditional_slo_cost"]["conditional_usd_per_million_tokens"],default=None)
    full=[r for r in rows if r["mode"]=="full_operator_serving"]
    checks={"all_candidates_executed_or_explicitly_rejected":len(rows)==9 and all(r["mode"] in {"full_operator_serving","configuration_rejected"} for r in rows),
        "executed_requests_completed":bool(full) and all(r["completed_requests"]==r["request_count"] for r in full),
        "cost_counts_match_execution":all(r["conditional_cost"]["hbm_packages"]==r["package_configuration"]["hbm_packages"] and r["conditional_cost"]["hbf_packages"]==r["package_configuration"]["hbf_packages"] for r in rows),
        "all_reported_rates_finite":all(math.isfinite(r["node_tps"]) and r["node_tps"]>0 for r in full)}
    return group("11_provisioning","⑪ 预算约束下重新执行服务",rows,checks,
        ["固定4槽、GPU和SLO，重新执行70B代表工作量后进行有限候选预算筛选与条件成本比较"],
        ["容量、面积、购置成本是共同上限而非每点消耗完全相等；价格和面积是假设。",
         "每种内存使用一个物理Core设备，容量、带宽和并行度同时随栈数改变；不是实际多栈布线模型。",
         "单请求2K/8输出、有限候选不构成全工作负载全局最优；磨损和购置成本尚未闭环。",
         "无可行候选时保留null，不放宽SLO或预算以制造赢家。"],
        budgets=BUDGETS,selected_candidate_id=selected["id"] if selected else None,
        selection_objective="minimum conditional purchase dollars per million SLO-qualified tokens among actually executed feasible candidates",
        feasible_candidate_ids=[r["id"] for r in candidates])
