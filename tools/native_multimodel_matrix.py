"""Run a reproducible multi-model/input/runtime native-vs-simulator matrix."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time, statistics
from pathlib import Path

EXPECTED = {
    "qwen25_0p5b": {"architecture":"qwen2", "n_layer":24, "n_embd":896, "n_head":14, "n_head_kv":2, "vocab_size":151936},
    "tinyllama_1p1b": {"architecture":"llama", "n_layer":22, "n_embd":2048, "n_head":32, "n_head_kv":4, "vocab_size":32000},
    # GGUF block_count includes one nextn/MTP head; n_layer is the executable
    # trunk exposed by llama.cpp (64), while n_layer_all is retained in the
    # per-cell GGUF metadata.
    "qwen27b": {"architecture":"qwen35", "n_layer":64, "n_embd":5120, "n_head":24, "n_head_kv":4, "vocab_size":248320},
}
INPUTS = {
    "short": "Hi.",
    "medium": "Explain deterministic benchmarking for inference systems.",
    "long": "Benchmarking long-context inference requires measuring prompt evaluation separately from steady-state decode, while preserving identical model and runtime configuration.",
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def load_geometry(root: Path, model: Path) -> dict:
    sys.path.insert(0, str(root / "src"))
    from heterollm_sim.gguf_parity import read_gguf_metadata
    g=read_gguf_metadata(model)
    return {"architecture":g.architecture,"n_layer":g.n_layer,"n_layer_all":g.n_layer_all,"n_layer_nextn":g.n_layer_nextn,"n_embd":g.n_embd,"n_head":g.n_head,"n_head_kv":g.n_head_kv,"vocab_size":g.vocab_size,"context_length":g.context_length,"quantization":g.quantization}

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", action="append", required=True, help="key=GGUF path; repeat")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--predict", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=1)
    args=ap.parse_args(); root=Path(__file__).resolve().parents[1]
    models={}
    for item in args.model:
        key,path=item.split("=",1); p=Path(path).resolve(); geom=load_geometry(root,p)
        expected=EXPECTED.get(key); mismatch={k:(expected[k],geom.get(k)) for k in expected or {} if expected[k]!=geom.get(k)}
        models[key]={"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size,"geometry":geom,"expected_geometry":expected,"geometry_mismatch":mismatch}
    out_dir=args.output.parent/(args.output.stem+"_cells"); out_dir.mkdir(parents=True,exist_ok=True)
    cells=[]
    for key,mi in models.items():
        if mi["geometry_mismatch"]:
            cells.append({"model_key":key,"status":"unsupported_model","reason":"expected_geometry_mismatch","model":mi}); continue
        n_layer=mi["geometry"]["n_layer"]
        runtimes={"cpu_only":0,"partial":min(13,n_layer),"full":-1}
        for inp,prompt in INPUTS.items():
            for rt,gl in runtimes.items():
                for rep in range(1,args.repeat+1):
                    cid=f"{key}__{inp}__{rt}__r{rep}"; op=out_dir/(cid+".json")
                    cmd=[sys.executable,str(root/"tools/native_llama_compare.py"),"--exe",args.exe,"--model",mi["path"],"--prompt",prompt,"--predict",str(args.predict),"--ctx",str(args.ctx),"--parallel","1","--batch","64","--ubatch","64","--threads","16","--gpu-layers",str(gl),"--output",str(op)]
                    t=time.perf_counter(); cp=subprocess.run(cmd,capture_output=True,text=True,timeout=600); wall=time.perf_counter()-t
                    cell={"cell_id":cid,"model_key":key,"input_id":inp,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"runtime_id":rt,"runtime":{"gpu_layers":gl,"ctx":args.ctx,"parallel":1,"batch":64,"ubatch":64,"threads":16},"repeat":rep,"model":mi,"returncode":cp.returncode,"wall_s":wall,"output":str(op)}
                    if op.exists():
                        try:
                            pld=json.loads(op.read_text(encoding="utf-8")); cell["status"]="valid" if pld.get("parity",{}).get("geometry",{}).get("ok") else "invalid"; cell["native"] = pld.get("native"); cell["simulator"]=pld.get("simulator"); cell["relative_error_pct"]=pld.get("relative_error_pct"); cell["parity"]=pld.get("parity"); cell["runtime_fingerprint"]=pld.get("runtime_fingerprint"); cell["hardware"]=pld.get("hardware"); cell["declared_token_counts"]={"requested_output":args.predict,"simulator":pld.get("token_counts")}; cell["native_actual_token_counts"]={"prompt":(pld.get("native") or {}).get("timings",{}).get("prompt_n"),"output":(pld.get("native") or {}).get("timings",{}).get("predicted_n")}
                        except Exception as e: cell.update(status="invalid",error=str(e))
                    else: cell.update(status="error",stderr=cp.stderr[-2000:])
                    cells.append(cell)
    summary={"schema":"native-multimodel-matrix/v1","models":models,"inputs":INPUTS,"cell_count":len(cells),"cells":cells,"aggregation":{}}
    groups={}
    for c in cells:
        if c.get("status")!="valid": continue
        k=(c["model"]["sha256"],c["input_id"],c["runtime_id"])
        groups.setdefault(k,[]).append(c)
    for k,arr in groups.items():
        metrics={}
        for m in ("ttft_ms","tpot_ms","e2e_ms"):
            vals=[(c.get("relative_error_pct") or {}).get(m) for c in arr]; vals=[float(v) for v in vals if v is not None]
            if vals: metrics[m]={"n":len(vals),"median_signed_pct":statistics.median(vals),"median_abs_pct":statistics.median(abs(v) for v in vals),"p95_abs_pct":sorted(abs(v) for v in vals)[min(len(vals)-1,max(0,int(0.95*len(vals))-1))]}
            else: metrics[m]={"n":0}
        summary["aggregation"]["|".join(k)]=metrics
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"output":str(args.output.resolve()),"cell_count":len(cells),"valid":sum(c.get('status')=='valid' for c in cells)},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
