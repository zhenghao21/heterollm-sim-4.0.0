"""Blocked observer-effect diagnostic; never calibrates simulator latency."""
from __future__ import annotations
import argparse, hashlib, json, os, random, socket, statistics, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_llama_compare import wait_health, post_json

def summarize(runs):
    blocks = {}
    for run in runs:
        t = run["timings"]
        blocks.setdefault((run["block"], run["enabled"]), []).append({
            "ttft_ms": t["prompt_ms"], "tpot_ms": t["predicted_ms"] / (t["predicted_n"] - 1),
            "e2e_ms": t["prompt_ms"] + t["predicted_ms"]})
    summary = {}
    rng = random.Random(20260914)
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = {mode: [statistics.median(r[metric] for r in rows) for (_, enabled), rows in blocks.items() if enabled == mode] for mode in (0, 1)}
        ratios = []
        for _ in range(2000):
            off = statistics.median(rng.choices(values[0], k=len(values[0])))
            on = statistics.median(rng.choices(values[1], k=len(values[1])))
            ratios.append(100 * (on / off - 1))
        ratios.sort()
        summary[metric] = {"off_block_medians_ms": values[0], "on_block_medians_ms": values[1],
                           "median_change_pct": 100*(statistics.median(values[1])/statistics.median(values[0])-1),
                           "bootstrap_95_interval_pct": [ratios[49], ratios[1949]]}
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=12)
    args = ap.parse_args()
    if args.output.exists(): raise SystemExit("refusing overwrite")
    if args.blocks < 4 or args.blocks % 4: raise SystemExit("blocks must be multiple of four")
    exe=ROOT/"source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe"
    model=ROOT/"artifacts/multimodel_20260913/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    order = ([0,1,1,0] * (args.blocks//4)); runs=[]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for block, enabled in enumerate(order):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
        cmd=[str(exe),"-m",str(model),"--host","127.0.0.1","--port",str(port),"-c","512","-ngl","-1","-np","1","-b","64","-ub","64","-t","16","-tb","16","-fa","off","--spec-type","none"]
        log=args.output.with_suffix(f".block{block}.log")
        with log.open("w",encoding="utf-8") as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=f,env=dict(os.environ,LLAMA_ENGINE_TOKEN_TIMES=str(enabled)))
        try:
            base=f"http://127.0.0.1:{port}"; wait_health(base,proc)
            payload={"prompt":"Timing overhead development check.","n_predict":8,"ignore_eos":True,"cache_prompt":False,"temperature":0,"stream":False}
            for _ in range(4): post_json(base+"/completion",payload)
            for repeat in range(4):
                response=post_json(base+"/completion",payload)
                runs.append({"block":block,"enabled":enabled,"repeat":repeat,"timings":response.get("timings"),"raw_response":response})
        finally:
            proc.terminate(); proc.wait(timeout=15)
    summary=summarize(runs)
    adequate=all(-5 <= v["bootstrap_95_interval_pct"][0] and v["bootstrap_95_interval_pct"][1] <= 5 for v in summary.values())
    result={"schema":"observer-effect-check/v2","order":order,"warmup_per_block":4,"runs_per_block":4,"runs":runs,"summary":summary,
            "preregistered_equivalence_margin_pct":5,"status":"within_margin_small_model_only" if adequate else "evidence_insufficient",
            "binary_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),
            "limitations":["independent process-block bootstrap; not cross-model evidence","same binary switch; no simulator fitting","do not run concurrently with benchmarks or tests"]}
    with args.output.open("x",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps({"status":result["status"],"summary":summary}))
    return 0
if __name__ == "__main__": raise SystemExit(main())
