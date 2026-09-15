"""Validate all raw batches from the frozen synthetic v2 protocol; no fitting."""
from pathlib import Path
import argparse, hashlib, json, math, re

HERE=Path(__file__).resolve().parent

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0

def digest(value):
    return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None

def norm(path):
    return str(path).replace("/","\\").lower()

def audit_run(doc,config,mode,protocol,manifest=None):
    issues=[]
    if not isinstance(doc,dict):return ["raw is not an object"]
    for key in ("backend_name","device_name","device_description","gpu_name","pci_bus_id"):
        if not isinstance(doc.get(key),str) or not doc[key].strip():issues.append("missing identity:"+key)
    for key in ("driver_version","runtime_version","qpc_frequency"):
        if not positive(doc.get(key)):issues.append("invalid identity:"+key)
    for key in ("compute_capability_major","compute_capability_minor"):
        if type(doc.get(key)) is not int or doc[key]<0:issues.append("invalid identity:"+key)
    for key in ("M","N","K","seed","group"):
        if doc.get(key)!=config[key]:issues.append("configuration mismatch:"+key)
    for key,want in {"schema":"backend-stream-event-probe/v3","weight_format":config["quant"],"input_dtype":"F32","output_dtype":"F32","layout":"ordinary_contiguous_2d","device":"cuda","threads":1,"cuda_index":0,"status":"measured","modules_stable":True,"supported":True,"wait_api":"ggml_backend_synchronize","cache_policy":"same_buffers_repeated_hot_cache_no_flush","graph_computations_per_batch":1024,"graph_compute_calls":36864,"warmup_requested":5,"formal_repeats_requested":30,"control_mode":mode=="control"}.items():
        if doc.get(key)!=want or (type(want) is bool and doc.get(key) is not want):issues.append("required field mismatch:"+key)
    contract=doc.get("timing_contract",{})
    if contract.get("id")!="actual-backend-stream-batch-envelope/v2" or contract.get("host_device_times_additive") is not False:issues.append("timing contract mismatch")
    env=doc.get("environment",{})
    if env.get("GGML_CUDA_DISABLE_GRAPHS")!="1":issues.append("graphs not disabled")
    for key in ("GGML_CUDA_FORCE_MMQ","GGML_CUDA_FORCE_CUBLAS","CUDA_VISIBLE_DEVICES","LLAMA_TRACE_ANNOTATIONS","GGML_CUDA_DISABLE_FUSION","GGML_CUDA_CUBLAS_COMPUTE_TYPE"):
        if key not in env or env[key] is not None:issues.append("environment override:"+key)
    quant=doc.get("quantization",{})
    for key in ("packed_weight_sha256","input_sha256"):
        if not digest(quant.get(key)):issues.append("missing input identity:"+key)
    if not positive(quant.get("bytes")):issues.append("invalid quantized byte count")
    expected_modules={}
    if manifest:
        expected_modules={norm(x["path"]):x["sha256"] for x in manifest["files"] if Path(x["path"]).name.lower() in ("ggml-base.dll","ggml-cpu.dll","ggml-cuda.dll")}
        expected_modules[norm(manifest["executable"]["path"])]=manifest["executable"]["sha256"]
    before,after=doc.get("loaded_modules_before"),doc.get("loaded_modules_after")
    if not isinstance(before,list) or not before or before!=after:issues.append("module records absent/changed")
    else:
        actual={}
        for item in before:
            if not isinstance(item,dict) or not isinstance(item.get("path"),str) or not digest(item.get("sha256")) or not positive(item.get("bytes")):issues.append("invalid loaded module record");continue
            key=norm(item["path"])
            if key in actual:issues.append("duplicate loaded module")
            actual[key]=item["sha256"]
        for path,expected_hash in expected_modules.items():
            if actual.get(path)!=expected_hash:issues.append("loaded identity mismatch:"+path)
    cc=doc.get("correctness_contract",{})
    if cc.get("absolute_tolerance")!=0.05 or cc.get("relative_tolerance")!=0.03:issues.append("numerical threshold mismatch")
    if cc.get("reference_mode")!="dual_math_and_source_path" or cc.get("path_absolute_tolerance")!=.0001 or cc.get("path_relative_tolerance")!=.00001 or cc.get("source_runtime_equivalence_proven") is not False:issues.append("dual reference contract mismatch")
    expected_sample_count=min(4096,config["M"]*config["N"])
    def check_numerics(check,label):
        if not isinstance(check,dict) or check.get("passed") is not True or check.get("finite_all_outputs") is not True:issues.append(label+":numerical failure");return
        samples=check.get("samples",[])
        if check.get("sample_count")!=expected_sample_count or len(samples)!=expected_sample_count:issues.append(label+":numerical sample count");return
        for j,sample in enumerate(samples):
            index=j*(config["M"]*config["N"]-1)//max(expected_sample_count-1,1)
            if sample.get("n_index")!=index%config["N"] or sample.get("m_index")!=index//config["N"]:issues.append(label+":reference index mismatch");break
            a,r,p=sample.get("actual"),sample.get("reference"),sample.get("path_reference")
            vals=[a,r,p,sample.get("math_reference"),sample.get("math_absolute_error"),sample.get("path_absolute_error")]
            if not all(type(v) in (int,float) and math.isfinite(v) for v in vals):issues.append(label+":numerical sample rejected");break
            math_pass=abs(a-r)<=.05+.03*abs(r)
            path_pass=abs(a-p)<=.0001+.00001*abs(p) if config['quant']=='Q5_0' else math_pass
            if r!=sample['math_reference'] or not math.isclose(abs(a-r),sample['math_absolute_error'],abs_tol=1e-12) or not math.isclose(abs(a-p),sample['path_absolute_error'],abs_tol=1e-12) or sample.get('math_pass') is not math_pass or sample.get('path_pass') is not path_pass or sample.get('pass') is not True or not path_pass:issues.append(label+":numerical sample rejected");break
    check_numerics(doc.get("first_call_correctness"),"first check")
    check_numerics(doc.get("final_correctness"),"final check")
    rows=doc.get("runs",[])
    if not isinstance(rows,list):return issues+["runs absent"]
    sequence=[("first_call",0)]+[("warmup",i) for i in range(5)]+[("formal",i) for i in range(30)]
    if [(r.get("phase"),r.get("index")) for r in rows]!=sequence:issues.append("complete batch sequence mismatch")
    frequency=doc.get("qpc_frequency")
    for i,row in enumerate(rows):
        label=f"row {i}"
        if row.get("graph_computations")!=1024:issues.append(label+":submission count")
        check_numerics(row.get("correctness"),label)
        keys=["qpc_start","qpc_submit_start","qpc_submit_end","qpc_wait_start","qpc_wait_end","qpc_end"]
        if mode=="event":keys=["qpc_start","qpc_record_begin_start","qpc_record_begin_end","qpc_submit_start","qpc_submit_end","qpc_record_end_start","qpc_record_end_end","qpc_wait_start","qpc_wait_end","qpc_end"]
        times=[row.get(key) for key in keys]
        valid=all(type(t) is int and t>=0 for t in times)
        if not valid or times!=sorted(times):issues.append(label+":QPC ordering")
        if positive(frequency) and valid:
            wall=(row["qpc_end"]-row["qpc_start"])*1e9/frequency
            for key,want in [("host_wall_ns",wall),("host_per_graph_ns",wall/1024)]:
                if not positive(row.get(key)) or not math.isclose(row[key],want,rel_tol=1e-5,abs_tol=1):issues.append(label+":QPC derivation:"+key)
        statuses=["ggml_status","cuda_submit_status","cuda_wait_status"]
        if mode=="event":statuses += ["cuda_begin_record_status","cuda_end_record_status","cuda_query_after_wait","cuda_elapsed_status"]
        if any(type(row.get(k)) is not int or row[k]!=0 for k in statuses):issues.append(label+":CUDA/ggml status")
        if mode=="event":
            if row.get("cuda_query_before_wait") not in (0,600):issues.append(label+":prewait query")
            if not positive(row.get("event_envelope_ms")):issues.append(label+":event duration")
            elif positive(row.get("host_wall_ns")) and positive(frequency):
                tolerance=max(protocol["quality_gates"]["event_host_containment_tolerance_ns"],2e9/frequency)
                if row["event_envelope_ms"]*1e6 > row["host_wall_ns"]+tolerance:issues.append(label+":event exceeds enclosing host wall")
        elif row.get("event_envelope_ms") is not None:issues.append(label+":control contains event duration")
    return sorted(set(issues))

def audit(directory,protocol,manifest):
    rows=[];issues=[]
    for config in protocol["configs"]:
        for mode in ("event","control"):
            path=directory/f"{config['id']}.{mode}.json"
            try:
                problems=audit_run(load(path),config,mode,protocol,manifest)
                identity=hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception as exc:problems=["raw unavailable:"+str(exc)];identity=None
            rows.append({"config":config["id"],"mode":mode,"path":str(path),"sha256":identity,"issues":problems})
            issues.extend(config["id"]+":"+mode+":"+x for x in problems)
    return {"schema":"full-probe-raw-audit/v2","complete_raw_passed":not issues,"issues":issues,"rows":rows,"calibration_eligible":False,"interpretation":"Complete raw batches, input identities, native modules, QPC and final-output numerics only; quality and execution checks required separately."}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("directory",type=Path);a=ap.parse_args();out=a.directory/"full_raw_audit.json"
    if out.exists():raise SystemExit("refusing overwrite")
    result=audit(a.directory,load(HERE/"protocol.json"),load(HERE/"build_manifest.json"));out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({"complete_raw_passed":result["complete_raw_passed"],"issue_count":len(result["issues"])}));raise SystemExit(0 if result["complete_raw_passed"] else 1)
