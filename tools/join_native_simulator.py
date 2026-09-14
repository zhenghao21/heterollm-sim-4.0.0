"""Join a native Nsight trace with the corresponding simulator task graph."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'tools'))
from native_llama_compare import build_matching_scenario
from heterollm_sim.planner import compile_scenario
from heterollm_sim.task_mapping import map_native_events_to_tasks
from heterollm_sim.gguf_parity import read_gguf_metadata, build_model_from_gguf
from heterollm_sim.serde import stable_hash

def _int_flag(command, *flags, default):
    for i, value in enumerate(command):
        if value in flags and i + 1 < len(command):
            try: return int(command[i + 1])
            except ValueError: pass
    return default

def main():
    p=argparse.ArgumentParser(); p.add_argument('trace',type=Path); p.add_argument('--profile',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    trace=json.loads(a.trace.read_text(encoding='utf-8')); profile=json.loads(a.profile.read_text(encoding='utf-8')) if a.profile else {}
    formal=profile.get('formal',{}).get('timings',{}); prompt=int(formal.get('prompt_n',8)); output=int(formal.get('predicted_n',1))
    command = profile.get('server_command') or profile.get('command') or []
    ctx = _int_flag(command, '-c', '--ctx-size', default=512)
    parallel = _int_flag(command, '-np', '--parallel', default=1)
    batch = _int_flag(command, '-b', '--batch-size', default=64)
    ubatch = _int_flag(command, '-ub', '--ubatch-size', default=64)
    threads = _int_flag(command, '-t', '--threads', default=16)
    gpu_layers = _int_flag(command, '-ngl', '--gpu-layers', default=-1)
    model_path = profile.get('model') or trace.get('model')
    if not model_path:
        raise ValueError('native profile must identify the GGUF model')
    model = build_model_from_gguf(read_gguf_metadata(model_path))
    expected_sha = profile.get('gguf', {}).get('gguf', {}).get('sha256')
    actual_sha = model.metadata.get('metadata', {}).get('gguf_sha256') if isinstance(getattr(model, 'metadata', None), dict) else None
    if expected_sha and actual_sha and expected_sha != actual_sha:
        raise ValueError('native profile GGUF SHA does not match trace/model input')
    scenario=build_matching_scenario(prompt,output,ctx=ctx,parallel=parallel,batch=batch,ubatch=ubatch,threads=threads,gpu_layers=gpu_layers,model=model)
    scheduler=replace(scenario.workload.scheduler, mode='static')
    scenario=replace(scenario, workload=replace(scenario.workload,scheduler=scheduler))
    schedule=compile_scenario(scenario)
    report=map_native_events_to_tasks(trace.get('events',[]), schedule.tasks)
    report.update({'native_trace':str(a.trace.resolve()),'simulator_task_count':len(schedule.tasks),'prompt_tokens':prompt,'output_tokens':output,'scenario_fingerprint':stable_hash(scenario),'runtime_config':{'ctx':ctx,'parallel':parallel,'batch':batch,'ubatch':ubatch,'threads':threads,'gpu_layers':gpu_layers},'task_graph_view':'static replay of a single request; native continuous scheduler timing is not reproduced','native_event_kind_counts':{k:sum(1 for e in trace.get('events',[]) if e.get('kind')==k) for k in sorted({e.get('kind') for e in trace.get('events',[])})}})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'schema':report['schema'],'matched_count':report['matched_count'],'unmatched_count':report['unmatched_count'],'output':str(a.output.resolve())},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
