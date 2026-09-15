"""Apply preregistered quality gates to v2 batches; never fit costs."""
import argparse,json,statistics
from pathlib import Path
from full_raw_audit import HERE,load,audit_run,positive

def percentile(values,p):
    values=sorted(values);index=(len(values)-1)*p;i=int(index)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(index-i)

def assess(event,control,config,protocol,manifest):
    problems=[];stats={}
    limit=protocol["quality_gates"]["dispersion_p90_p10_max"]
    for mode,data in [("event",event),("control",control)]:
        errors=audit_run(data,config,mode,protocol,manifest)
        problems.extend(mode+":"+x for x in errors)
        rows=[x for x in data.get("runs",[]) if x.get("phase")=="formal"]
        values=[x.get("host_wall_ns") for x in rows]
        if len(values)!=30 or not all(positive(x) for x in values):continue
        med=statistics.median(values);spread=percentile(values,.9)/percentile(values,.1)
        stats[mode]={"formal_batches":len(values),"graph_computations_per_batch":64,"host_batch_median_ns":med,"host_per_graph_median_ns":med/64,"host_p90_p10":spread,"host_max_deviation_from_median_fraction":max(abs(x/med-1) for x in values)}
        if spread>limit:problems.append(mode+":wall dispersion >1.50")
        if mode=="event":
            values=[x.get("event_envelope_ms") for x in rows]
            if all(positive(x) for x in values):
                med=statistics.median(values);spread=percentile(values,.9)/percentile(values,.1)
                stats[mode].update(event_batch_median_ms=med,event_per_graph_median_ns=med*1e6/64,event_p90_p10=spread,event_max_deviation_from_median_fraction=max(abs(x/med-1) for x in values))
                if spread>limit:problems.append("event:envelope dispersion >1.50")
    if len(stats)==2:
        ratio=stats["event"]["host_batch_median_ns"]/stats["control"]["host_batch_median_ns"]
        stats["instrumented_to_control_wall_ratio"]=ratio
        if abs(ratio-1)>protocol["quality_gates"]["event_control_wall_relative_difference_max"]:problems.append("event/control method perturbation >20%")
    for key in ("M","N","K","weight_format","seed","group","device_description","gpu_name","driver_version","runtime_version","pci_bus_id","compute_capability_major","compute_capability_minor","cache_policy","wait_api","graph_computations_per_batch"):
        if key not in event or key not in control or event[key] is None or control[key] is None or event[key]!=control[key]:problems.append("pair identity missing/mismatch:"+key)
    for key in ("packed_weight_sha256","input_sha256"):
        if event.get("quantization",{}).get(key)!=control.get("quantization",{}).get(key):problems.append("pair synthetic data mismatch:"+key)
    return {"schema":"stream-event-quality/v2","config":config["id"],"accepted":not problems,"problems":sorted(set(problems)),"statistics":stats,"calibration_eligible":False,"accuracy_promotion":False,"interpretation":"diagnostic batch event envelope; accepted does not establish native +/-5%, a kernel model, or LLM accuracy"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--event",required=True,type=Path);p.add_argument("--control",required=True,type=Path);p.add_argument("--config",required=True);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise SystemExit("output exists")
    protocol=load(HERE/"protocol.json");matches=[x for x in protocol["configs"] if x["id"]==a.config]
    if len(matches)!=1:raise SystemExit("unknown frozen config")
    try:result=assess(load(a.event),load(a.control),matches[0],protocol,load(HERE/"build_manifest.json"))
    except Exception as exc:result={"schema":"stream-event-quality/v2","config":a.config,"accepted":False,"problems":["assessment exception:"+str(exc)],"calibration_eligible":False}
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result));raise SystemExit(0 if result["accepted"] else 4)
