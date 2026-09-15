"""Frozen, bounded variance diagnostics. No simulator fitting or test-set claims."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_llama_compare import (wait_health, post_parallel_stream_json, probe_hardware,
                                       _native_execution_environment, native_extractor_identity)
from tools.native_runtime_evidence import capture_loaded_runtime
from tools.native_http_client import PersistentClient


def file_ref(path):
    path=Path(path).resolve(strict=True)
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''): digest.update(chunk)
    return {'path':str(path),'sha256':digest.hexdigest(),'bytes':path.stat().st_size}


def verify(refs):
    if not refs: raise ValueError('empty source freeze')
    for ref in refs:
        if file_ref(ref['path']) != ref: raise RuntimeError('freeze drift: '+ref['path'])


def write_new(path, value):
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)


def state(pid):
    import psutil
    proc=psutil.Process(pid)
    info={'captured_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,
          'process_priority':int(proc.nice()),'cpu_affinity':proc.cpu_affinity(),
          'cpu_percent_snapshot':psutil.cpu_percent(interval=.1),'memory_available':psutil.virtual_memory().available}
    cp=subprocess.run(['nvidia-smi','--query-gpu=uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    info['gpu_state']={'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
    return info


def summarize_runs(runs):
    metrics={k:[] for k in ('ttft','tpot','e2e')};all_requests={k:[] for k in metrics};splits=0;spreads=[]
    for run in runs:
        values={k:[] for k in metrics};begins=[];firsts=[]
        for response,boundary in run['pairs']:
            t=response['timings'];ts=t['engine_token_times_us'];begin=t['engine_request_begin_us']
            begins.append(begin);firsts.append(ts[0])
            values['ttft'].append((ts[0]-begin)/1000)
            values['tpot'].append((ts[-1]-ts[0])/1000/(len(ts)-1))
            values['e2e'].append((ts[-1]-begin)/1000)
        splits+=max(begins)>min(firsts)
        spreads.append((max(begins)-min(begins))/1000)
        for k in metrics:
            metrics[k].append(statistics.median(values[k]));all_requests[k].extend(values[k])
    return {'batches':len(runs),'split_batches':splits,'engine_start_spread_ms':spreads,
            'metrics':{k:{'batch_medians_ms':v,'median_ms':statistics.median(v),
                          'sample_cv_pct':100*statistics.stdev(v)/statistics.mean(v) if len(v)>1 else None,
                          'all_request_cv_pct':100*statistics.stdev(all_requests[k])/statistics.mean(all_requests[k]) if len(all_requests[k])>1 else None}
                       for k,v in metrics.items()}} if runs else {'batches':0,'metrics':{}}


def validate_pairs(pairs, parallel, output, prompt_tokens=None):
    if len(pairs)!=parallel: raise ValueError('request coverage mismatch')
    slots=[]
    for response,boundary in pairs:
        t=response['timings'];ts=t['engine_token_times_us'];slots.append(response['id_slot'])
        if len(ts)!=output or t['predicted_n']!=output or t['cache_n']!=0 or response.get('truncated'):
            raise ValueError('token/cache/truncation contract mismatch')
        if not t['engine_timepoints_complete'] or t['engine_request_begin_us']>ts[0] or ts!=sorted(ts):
            raise ValueError('incomplete/non-monotonic engine times')
        if ts[0]!=t['engine_prompt_last_us'] or ts[-1]!=t['engine_last_token_us']:
            raise ValueError('engine endpoints mismatch')
        if prompt_tokens is not None and t['prompt_n']!=prompt_tokens:raise ValueError('tokenizer count mismatch')
        if boundary.get('request_start_monotonic_s') is None or boundary.get('last_token_monotonic_s') is None:
            raise ValueError('missing client boundaries')
    if len(set(slots))!=parallel:raise ValueError('concurrent slot set mismatch')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',action='append',type=Path,required=True)
    ap.add_argument('--prompt',default='Hello, explain AI.')
    ap.add_argument('--predict',type=int,default=17)
    ap.add_argument('--parallel',type=int,default=4)
    ap.add_argument('--repeats',type=int,default=20)
    ap.add_argument('--warmup-batches',type=int,default=8)
    ap.add_argument('--conditions',nargs='+',choices=('default','quiet','preconnect_quiet','preconnect_priority_quiet'),default=['quiet','preconnect_quiet','preconnect_quiet','quiet'])
    args=ap.parse_args()
    if args.predict<2 or min(args.repeats,args.warmup_batches,args.parallel)<1:raise ValueError('positive counts; output must exceed one')
    args.output.mkdir(parents=True,exist_ok=False)
    exe=ROOT/'source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe'
    source_paths=[Path(__file__),ROOT/'tools/native_llama_compare.py',ROOT/'tools/native_http_client.py',ROOT/'tools/native_runtime_evidence.py',ROOT/'tools/evaluation_contract.py']
    source_refs=[file_ref(p) for p in source_paths]
    refs=[file_ref(exe)]+[file_ref(p) for p in args.model]
    protocol={'schema':'native-variance-protocol/v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'source_freeze':source_refs,'artifacts':refs,'hardware':probe_hardware(),
              'environment':_native_execution_environment(),'extractor':native_extractor_identity(),
              'models':[str(p.resolve()) for p in args.model],'prompt':args.prompt,'output':args.predict,'parallel':args.parallel,
              'conditions':args.conditions,'repeats':args.repeats,'warmup_batches':args.warmup_batches,
              'metric':'sample stdev / mean of per-batch request medians; pooled across all process blocks',
              'sampling':'fixed budget; no discarded runs, no retry, no variance-based extension',
              'target_cv_pct':10,'scope':'development variance diagnostic; not prediction-error acceptance; observer equivalence unresolved',
              'warmup_stream':True,'counter_queries_between_warmup_and_measurement':False,'native_cli':'ctx2048,b64,ub64,t16,tb16,FAoff,mmap,kvo,op-offload,kvu,cb,perf,metrics,warmup,spec-none; quiet adds --log-verbosity 0; priority explicitly adds --prio 1 --prio-batch 1'}
    write_new(args.output/'protocol.json',protocol)
    summaries=[];all_groups={};failures=[]
    for model_index,model in enumerate(args.model):
      for block,condition in enumerate(args.conditions):
        verify(source_refs)
        key=f'm{model_index}_b{block}_{condition}'
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        cmd=[str(exe),'-m',str(model.resolve()),'--host','127.0.0.1','--port',str(port),'-c','2048','-ngl','-1','-np',str(args.parallel),'-b','64','-ub','64','-t','16','-tb','16','-fa','off','--load-mode','mmap','-kvo','--op-offload','-sm','layer','-mg','0','-ctk','f16','-ctv','f16','-kvu','-cb','--perf','--metrics','--warmup','--spec-type','none']
        if condition!='default':cmd+=['--log-verbosity','0']
        if condition=='preconnect_priority_quiet':cmd+=['--prio','1','--prio-batch','1']
        record={'model':str(model.resolve()),'condition':condition,'block':block,'command':cmd,'warmup':[],'runs':[],'status':'incomplete'}
        client=PersistentClient(parallel=args.parallel,connection_policy='preconnect_each_batch') if condition.startswith('preconnect_') else None
        proc=None
        try:
          with (args.output/(key+'.log')).open('x') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=log)
          base=f'http://127.0.0.1:{port}';wait_health(base,proc)
          record['runtime_before']=capture_loaded_runtime(proc.pid,exe);record['state_before']=state(proc.pid)
          if condition=='preconnect_priority_quiet' and sys.platform=='win32' and record['state_before']['process_priority']!=0x8000:
            raise ValueError('requested ABOVE_NORMAL priority not observed')
          from tools.native_llama_compare import post_json
          tokenizer=post_json(base+'/tokenize',{'content':args.prompt,'add_special':True})
          record['tokenizer']=tokenizer;prompt_n=len(tokenizer['tokens'])
          payload={'prompt':args.prompt,'n_predict':args.predict,'ignore_eos':True,'cache_prompt':False,'temperature':0,'top_k':1,'seed':42,'stream':True}
          run_batch=lambda: client.batch(base+'/completion',payload) if client else post_parallel_stream_json(base+'/completion',payload,args.parallel)
          for repeat in range(args.warmup_batches):
            pairs=run_batch();record['warmup'].append({'repeat':repeat,'pairs':pairs})
            validate_pairs(pairs,args.parallel,args.predict,prompt_n)
          for repeat in range(args.repeats):
            pairs=run_batch();record['runs'].append({'repeat':repeat,'pairs':pairs})
            validate_pairs(pairs,args.parallel,args.predict,prompt_n)
          record['state_after']=state(proc.pid);record['runtime_after']=capture_loaded_runtime(proc.pid,exe)
          if record['runtime_before']['status']!='captured' or record['runtime_after']['status']!='captured' or record['runtime_before']['module_identity_sha256']!=record['runtime_after']['module_identity_sha256']:
            raise ValueError('runtime identity drift')
          record['status']='complete';verify(source_refs)
        except Exception as error:
          record['status']='failed';record['error']=repr(error);failures.append({'key':key,'error':repr(error)})
        finally:
          if client:client.close()
          if proc is not None:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
          write_new(args.output/(key+'.json'),record)
        qualified_runs=record['runs'] if record['status']=='complete' else []
        row={'key':key,'model':str(model.resolve()),'condition':condition,'status':record['status'],'captured_batches':len(record['runs']),**summarize_runs(qualified_runs)}
        summaries.append(row)
        all_groups.setdefault((str(model.resolve()),condition),[]).extend(qualified_runs)
        print(json.dumps({'key':key,'status':row['status'],'cv':{k:round(v['sample_cv_pct'],3) if v['sample_cv_pct'] is not None else None for k,v in row['metrics'].items()}}),flush=True)
    verify(source_refs);verify(refs)
    pooled=[{'model':m,'condition':c,**summarize_runs(runs)} for (m,c),runs in all_groups.items()]
    write_new(args.output/'summary.json',{'schema':'native-variance-results/v1','end_verified':True,'protocol_sha256':file_ref(args.output/'protocol.json')['sha256'],
              'planned_blocks':len(args.model)*len(args.conditions),'blocks':summaries,'pooled':pooled,'failures':failures,'formal_acceptance':False})
    return 1 if failures else 0
if __name__=='__main__':raise SystemExit(main())
