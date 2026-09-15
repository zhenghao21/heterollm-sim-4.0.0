"""Pre-registered native repeatability experiment; no fitting, no discarded slow samples.

Schema native-repeatability-protocol/v1: exe (optional), defaults, jobs.
Job: id, model, prompt OR prompt_token_ids, expected_prompt_tokens (optional),
kv_unified_per_slot (optional; sets ctx=parallel*capacity), output, parallel, gpu_layers, threads, threads_batch,
conditions=[{id, client, log_verbosity, priority, poll, cont_batching,
            warmup_batches, measure_batches, process_blocks,
            process_affinity_mask OR cpu_affinity_list,
            environment:{LLAMA_TRACE_ANNOTATIONS:'0'}, expected_power_scheme}].
Every condition may override job/default fields. Native runs are serial.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import types
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# An isolated source copy may read frozen models/native artifacts in a separate root.
# Never add the data root to Python imports.
DATA_ROOT = None
for import_root in (ROOT, ROOT / 'src'):
    if str(import_root) in sys.path: sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))
SCHEMA = 'native-repeatability-protocol/v1'
DEFAULTS = dict(client='preconnect_http', log_verbosity=0, priority=0,
 cont_batching=True, warmup_batches=8, measure_batches=12, process_blocks=2,
 request_timeout_seconds=900, ctx=2048, batch=64, ubatch=64, threads=16,
 threads_batch=16, output=17, parallel=4, gpu_layers=-1, flash_attention=False,
 seed=42, load_mode='mmap', telemetry=True, cache_ram_mib=8192)
ENV_KEYS = ('GGML_OP_OFFLOAD_MIN_BATCH', 'LLAMA_ENGINE_TOKEN_TIMES',
 'CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'OMP_NUM_THREADS',
 'GGML_CUDA_DISABLE_GRAPHS', 'LLAMA_TRACE_ANNOTATIONS', 'LLAMA_GRAPH_REUSE_DISABLE',
 'GGML_CPU_OPERATOR_TRACE', 'GGML_CPU_DISABLE_FUSION', 'GGML_CUDA_DISABLE_FUSION',
 'GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_NO_PINNED', 'GGML_CUDA_REGISTER_HOST',
 'GGML_CUDA_ENABLE_UNIFIED_MEMORY', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE')

def now(): return datetime.now(timezone.utc).isoformat()
def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()).hexdigest()
def file_ref(path):
    path=Path(path).resolve(strict=True); sha=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): sha.update(chunk)
    return {'path':str(path),'sha256':sha.hexdigest(),'bytes':path.stat().st_size}
def verify_refs(refs):
    if not isinstance(refs,list) or not refs: raise ValueError('empty frozen references')
    for ref in refs:
        if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256') or ref.get('bytes') is None:
            raise ValueError('incomplete frozen reference')
        if file_ref(ref['path']) != ref: raise RuntimeError('freeze drift: '+ref['path'])
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def write_batch_journal(output,freeze,c,phase,repeat,prompt_tokens,captures,batch_wall_ms):
    # Only write after all requests in this batch have returned. Keep every raw
    # response/SSE line/timepoint, without adding disk IO to the measured request.
    path=output/f'{c["key"]}.{phase}.{repeat:04d}.batch.json'
    value={'schema':'native-repeatability-batch-journal/v1','key':c['key'],
      'freeze_sha256':digest(freeze),'config_sha256':digest(c),'phase':phase,'repeat':repeat,
      'prompt_token_count':prompt_tokens,'completed_utc':now(),'batch_wall_ms':batch_wall_ms,'captures':captures}
    encoded=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    with path.open('x',encoding='utf-8') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    return file_ref(path)


def bind_runtime_identity(output,freeze,key,runtime):
    identity=runtime.get('module_identity_sha256')
    if runtime.get('status')!='captured' or not identity:raise ValueError('loaded native runtime missing')
    path=output/'loaded_runtime_identity.json'
    if path.exists():
        baseline=json.loads(path.read_text(encoding='utf-8'))
        if baseline.get('schema')!='native-repeatability-runtime-identity/v1' or baseline.get('freeze_sha256')!=digest(freeze):
            raise ValueError('loaded runtime baseline freeze mismatch')
        if baseline.get('module_identity_sha256')!=identity:raise ValueError('loaded runtime identity differs across process blocks')
    else:
        write_new(path,{'schema':'native-repeatability-runtime-identity/v1','freeze_sha256':digest(freeze),
          'first_block_key':key,'module_identity_sha256':identity,'artifacts':runtime.get('artifacts',[])})
    return file_ref(path)


class BatchBoundaryStop(Exception):pass


class BatchBoundaryControl:
    def __init__(self,output,deadline):self.output,self.deadline,self.interrupted=output,deadline,False
    def request_interrupt(self,*_):self.interrupted=True
    def reason(self):
        if self.interrupted:return 'keyboard_interrupt_at_batch_boundary'
        if (self.output/'STOP').exists():return 'operator_stop_at_batch_boundary'
        if time.monotonic()>=self.deadline:return 'budget_exhausted_at_batch_boundary'
        return None
    def check(self):
        reason=self.reason()
        if reason:raise BatchBoundaryStop(reason)


@contextmanager
def defer_keyboard_interrupts(control):
    # SIGINT becomes a cooperative stop. Do not interrupt an ordinary native
    # request or lose the returned captures while waiting for a batch to finish.
    previous=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,control.request_interrupt)
    try:yield
    finally:signal.signal(signal.SIGINT,previous)


def data_root():return Path(DATA_ROOT if DATA_ROOT is not None else ROOT).resolve(strict=True)
def project_file(path):
    root=data_root();path=Path(path)
    path=(root/path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    if not path.is_relative_to(root): raise ValueError('native/model must be inside project37 data root: '+str(path))
    return path
def positive_integer(value):return not isinstance(value,bool) and isinstance(value,int) and value>0
def valid_token_ids(ids):
    return isinstance(ids,list) and bool(ids) and all(not isinstance(i,bool) and isinstance(i,int) and i>=0 for i in ids)
def validate_prompt(c):
    ids=c.get('prompt_token_ids');expected=c.get('expected_prompt_tokens')
    if ids is not None:
        if not valid_token_ids(ids):raise ValueError('prompt_token_ids must be a nonempty array of nonnegative integers')
        if expected is not None and expected!=len(ids):raise ValueError('expected_prompt_tokens does not match prompt_token_ids length')
    elif not isinstance(c.get('prompt'),str) or not c['prompt']:raise ValueError('nonempty prompt or prompt_token_ids required')
    if expected is not None and not positive_integer(expected):raise ValueError('expected_prompt_tokens must be positive integer')
    count=len(ids) if ids is not None else expected
    if count is not None:validate_prompt_capacity(c,count)
def validate_prompt_capacity(c,count):
    capacity=c.get('kv_unified_per_slot',c['ctx'])
    if count+c['output']>capacity:raise ValueError('prompt tokens + output exceeds per-slot capacity')
def resolve_prompt(c,post,base,evidence=None):
    exact=c.get('prompt_token_ids');content=exact if exact is not None else c['prompt']
    tokenizer=post(base+'/tokenize',{'content':content,'add_special':exact is None})
    if evidence is not None:evidence['tokenizer']=tokenizer
    ids=tokenizer.get('tokens')
    if not valid_token_ids(ids):raise ValueError('missing or invalid tokenizer ids')
    if exact is not None and ids!=exact:raise ValueError('tokenizer ids differ from exact prompt_token_ids')
    if c.get('expected_prompt_tokens') is not None and len(ids)!=c['expected_prompt_tokens']:
        raise ValueError('tokenizer count differs from expected_prompt_tokens')
    validate_prompt_capacity(c,len(ids))
    return tokenizer,ids,content
def affinity_for(c):
    values,mask=c.get('cpu_affinity_list'),c.get('process_affinity_mask')
    if values is not None and mask is not None: raise ValueError('choose affinity list or mask')
    if mask is not None:
        mask=int(mask,0) if isinstance(mask,str) else int(mask)
        if mask<1: raise ValueError('empty affinity mask')
        values=[i for i in range(mask.bit_length()) if mask & (1<<i)]
    if values is None:return None
    if not isinstance(values,list) or not values or any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in values) or len(set(values))!=len(values):
        raise ValueError('invalid cpu_affinity_list')
    return sorted(values)
def environment_for(overrides=None):
    overrides=overrides or {}
    if not isinstance(overrides,dict) or any(k not in ENV_KEYS for k in overrides):raise ValueError('environment variable not allowlisted')
    env=dict(os.environ)
    for key,value in overrides.items():
        if value is None:env.pop(key,None)
        else:env[key]=str(value)
    return env,{key:{'is_set':key in env,'value':env.get(key)} for key in ENV_KEYS}
def expand_protocol(p):
    if p.get('schema')!=SCHEMA or not isinstance(p.get('jobs'),list) or not p['jobs']:raise ValueError('protocol schema/nonempty jobs required')
    defaults=dict(DEFAULTS,**p.get('defaults',{}))
    if 'expected_power_scheme' in p:defaults['expected_power_scheme']=p['expected_power_scheme']
    plans=[];keys=set()
    for ji,job in enumerate(p['jobs']):
        base=dict(defaults,**{k:v for k,v in job.items() if k!='conditions'})
        jid=str(job.get('id',f'job{ji:03d}'))
        conditions=job.get('conditions',p.get('conditions',[{'id':'baseline'}]))
        if not isinstance(conditions,list) or not conditions:raise ValueError('nonempty conditions required')
        prepared=[]
        for ci,condition in enumerate(conditions):
            c=dict(base,**condition);c['job_id']=jid;c['condition_id']=str(condition.get('id',f'condition{ci:02d}'))
            c['model']=str(project_file(c['model']))
            for k in ('output','parallel','threads','threads_batch','warmup_batches','measure_batches','process_blocks','ctx','batch','ubatch'):
                if isinstance(c[k],bool) or not isinstance(c[k],int) or c[k]<1:raise ValueError(k+' must be positive integer')
            if 'kv_unified_per_slot' in c:
                if not positive_integer(c['kv_unified_per_slot']):raise ValueError('kv_unified_per_slot must be positive integer')
                required_ctx=c['parallel']*c['kv_unified_per_slot']
                explicit_ctx=condition.get('ctx',job.get('ctx',p.get('defaults',{}).get('ctx')))
                if explicit_ctx is not None and explicit_ctx!=required_ctx:raise ValueError('ctx must equal parallel * kv_unified_per_slot')
                c['ctx']=required_ctx
            validate_prompt(c)
            if c['client'] not in ('preconnect_http','independent_http','atomic_prompt_list'):raise ValueError('unsupported client')
            if c.get('prompt_token_ids') is not None and c['client']=='atomic_prompt_list':
                raise ValueError('exact token IDs require preconnect_http or independent_http client')
            if c['priority'] not in (0,1,2):raise ValueError('priority 0/1/2 only; no realtime')
            if not isinstance(c['cont_batching'],bool):raise ValueError('cont_batching must be boolean')
            if 'fit_params' in c and not isinstance(c['fit_params'],bool):raise ValueError('fit_params must be boolean')
            if c['request_timeout_seconds']<=0:raise ValueError('positive timeout required')
            if 'poll' in c and (isinstance(c['poll'],bool) or not isinstance(c['poll'],int) or not 0<=c['poll']<=100):raise ValueError('poll 0..100 required')
            if c.get('worker_cpu_mask') is not None:
                worker_mask=int(str(c['worker_cpu_mask']),0)
                if worker_mask<=0 or worker_mask.bit_count()!=c['threads'] or c['threads_batch']!=c['threads']:
                    raise ValueError('strict worker mask must assign one logical processor per worker, same decode/batch pool')
            c['resolved_cpu_affinity']=affinity_for(c)
            c['environment']={**p.get('environment',{}),**base.get('environment',{}),**condition.get('environment',{})}
            environment_for(c['environment'])
            prepared.append(c)
        # Explicit alternating process-block order is frozen before measuring.
        for block in range(max(c['process_blocks'] for c in prepared)):
            for c in (prepared if block % 2 == 0 else reversed(prepared)):
                if block>=c['process_blocks']:continue
                key=f'{jid}__{c["condition_id"]}__b{block:02d}'
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',key) or key in keys:raise ValueError('duplicate/unsafe block key '+key)
                keys.add(key);plans.append(dict(c,key=key,block=block))
    return plans

def percentile(values,q):
    if not values:return None
    v=sorted(values);pos=(len(v)-1)*q;lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo]+(v[hi]-v[lo])*(pos-lo)
def distribution(values,planned):
    center=statistics.median(values) if values else None
    deltas=[100*(x-center)/center for x in values] if center is not None and center>0 else []
    absolute=[abs(x) for x in deltas];inside=sum(x<=5 for x in absolute)
    return {'planned_samples':planned,'observed_samples':len(values),'missing_or_invalid_samples':max(0,planned-len(values)),
      'values_ms':values,'median_ms':center,'signed_deviation_pct':deltas,
      'worst_abs_deviation_pct':max(absolute) if absolute else None,'p90_abs_deviation_pct':percentile(absolute,.9),
      'fraction_within_5pct':inside/planned if planned else None,'observed_fraction_within_5pct':inside/len(values) if values else None,
      'sample_cv_pct':100*statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values)>0 else None,
      'strict_within_5pct':bool(planned>1 and len(values)==planned and deltas and max(absolute)<=5)}
def inspect_request(response,boundary,output,prompt_tokens):
    errors=[];t=response.get('timings',{});ts=t.get('engine_token_times_us');begin=t.get('engine_request_begin_us')
    if not isinstance(ts,list) or len(ts)!=output or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in ts):errors.append('token_times_count_or_type')
    elif isinstance(begin,bool) or not isinstance(begin,(int,float)) or not math.isfinite(begin) or begin>ts[0] or ts!=sorted(ts):errors.append('nonmonotonic_engine_times')
    elif not t.get('engine_timepoints_complete') or t.get('engine_prompt_last_us')!=ts[0] or t.get('engine_last_token_us')!=ts[-1]:errors.append('engine_endpoint_mismatch')
    if not positive_integer(t.get('predicted_n')) or not positive_integer(t.get('prompt_n')) or t['predicted_n']!=output or t['prompt_n']!=prompt_tokens:errors.append('tokenizer_or_output_count_mismatch')
    if t.get('cache_n')!=0 or response.get('truncated'):errors.append('cache_or_truncation')
    if boundary.get('request_start_monotonic_s') is None or boundary.get('last_token_monotonic_s') is None:errors.append('missing_client_boundaries')
    values={}
    if not errors:
        values={'ttft':(ts[0]-begin)/1000,'e2e':(ts[-1]-begin)/1000,'tpot':(ts[-1]-ts[0])/1000/(output-1) if output>1 else None}
        if any(v is not None and v<=0 for v in values.values()):errors.append('nonpositive_engine_duration')
    return {'status':'measured' if not errors else 'invalid','errors':errors,'engine_start_us':begin,'slot':response.get('id_slot'),
      'engine_first_us':ts[0] if isinstance(ts,list) and ts else None,'metrics_ms':values,
      'tpot_status':'not_applicable' if output==1 else 'measured' if not errors else 'invalid'}
def inspect_batch(captures,c,prompt_tokens):
    rows=[]
    for i,capture in enumerate(captures):
        row=dict(capture,request_index=i)
        if capture.get('status')=='complete':row.update(inspect_request(capture['response'],capture['boundary'],c['output'],prompt_tokens))
        else:row.update(status='failed',metrics_ms={},errors=[capture.get('error','transport_failure')])
        rows.append(row)
    good=[r for r in rows if r['status']=='measured']
    complete=len(good)==c['parallel'] and len(rows)==c['parallel']
    if complete and (any(r['slot'] is None for r in good) or len({r['slot'] for r in good})!=c['parallel']):
        complete=False
        for r in good:r['status']='invalid';r['errors'].append('invalid_concurrent_slot_set')
    if complete:
        for rank,r in enumerate(sorted(good,key=lambda r:(r['engine_start_us'],r['request_index']))):r['engine_start_rank']=rank
    starts=[r['engine_start_us'] for r in good];firsts=[r['engine_first_us'] for r in good]
    return {'status':'complete' if complete else 'failed','requests':rows,
      'engine_start_spread_ms':(max(starts)-min(starts))/1000 if starts else None,
      'split_admission':max(starts)>min(firsts) if starts and firsts else None,
      'batch_medians_ms':{m:statistics.median(r['metrics_ms'][m] for r in good) for m in ('ttft','tpot','e2e')
        if complete and all(r['metrics_ms'].get(m) is not None for r in good)}}
def summarize_group(configs,records):
    planned=sum(c['measure_batches'] for c in configs);parallel=configs[0]['parallel'];output=configs[0]['output']
    batches=[b for r in records for b in r.get('runs',[])]
    result={'job_id':configs[0]['job_id'],'condition_id':configs[0]['condition_id'],'model':configs[0]['model'],
      'planned_blocks':len(configs),'captured_blocks':len(records),'complete_blocks':sum(r.get('status')=='complete' for r in records),
      'planned_batches':planned,'captured_batches':len(batches),'failed_batches':sum(b.get('status')!='complete' for b in batches),
      'split_admission_batches':sum(b.get('split_admission') is True for b in batches),
      'rank_definition':'engine_request_begin_us order within complete batch, ties by request index',
      'engine_start_spread_ms':[b.get('engine_start_spread_ms') for b in batches],'metrics':{}}
    for m in ('ttft','tpot','e2e'):
        if m=='tpot' and output==1:result['metrics'][m]={'status':'not_applicable','reason':'single_output_token'};continue
        bv=[b['batch_medians_ms'][m] for b in batches if m in b.get('batch_medians_ms',{})]
        ranks=[]
        for rank in range(parallel):
            values=[r['metrics_ms'][m] for b in batches for r in b['requests'] if r.get('status')=='measured' and r.get('engine_start_rank')==rank and r.get('metrics_ms',{}).get(m) is not None]
            ranks.append(dict(rank=rank,**distribution(values,planned)))
        deviations=[abs(d) for r in ranks for d in r['signed_deviation_pct']]
        bs=distribution(bv,planned);complete=len(records)==len(configs) and all(r.get('status')=='complete' for r in records)
        result['metrics'][m]={'status':'measured','batch_medians':bs,'request_ranks':ranks,
          'all_request_deviations':{'worst_abs_deviation_pct':max(deviations) if deviations else None,
          'p90_abs_deviation_pct':percentile(deviations,.9),'fraction_within_5pct':sum(d<=5 for d in deviations)/(planned*parallel)},
          'strict_within_5pct':bool(complete and len(configs)>=2 and bs['strict_within_5pct'] and all(r['strict_within_5pct'] for r in ranks))}
    metrics=[m for m in result['metrics'].values() if m['status']!='not_applicable']
    result['strict_within_5pct']=bool(metrics and all(m['strict_within_5pct'] for m in metrics));result['formal_prediction_acceptance']=False
    return result

class _CapturedResponse:
    def __init__(self,response,capture,drain=False):self.response,self.capture,self.drain=response,capture,drain
    def __enter__(self):return self
    def __iter__(self):
        for line in self.response:
            self.capture['raw_received_lines'].append({'received_monotonic_s':time.perf_counter(),'line':line.decode('utf-8',errors='replace')})
            yield line
    def __exit__(self,typ,exc,tb):
        try:
            if self.drain and typ is None:
                trailing=self.response.read()
                if trailing:self.capture['trailing_body']=trailing.decode('utf-8',errors='replace')
        finally:self.response.close()
        return False
class TrialClient:
    """Fixed workers; complete successes and partial failed wire responses retained."""
    def __init__(self,c):
        from tools.native_llama_compare import post_stream_json
        from tools.native_cohort_client import post_cohort_stream_json
        self.config,self.parser,self.cohort=c,post_stream_json,post_cohort_stream_json
        self.pool=ThreadPoolExecutor(max_workers=c['parallel'],thread_name_prefix='repeatability-request')
    def batch(self,url,payload):
        c=self.config
        if c['client']=='atomic_prompt_list':
            capture={'raw_received_lines':[]}
            def opened(req,timeout=None):return _CapturedResponse(urlopen(req,timeout=c['request_timeout_seconds']),capture)
            parser=types.FunctionType(self.cohort.__code__,dict(self.cohort.__globals__,urlopen=opened))
            try:
                pairs=parser(url,payload,c['parallel'])
                result=[{'status':'complete','response':r,'boundary':b,'shared_transport_record':True} for r,b in pairs]
                result[0]['raw_received_lines']=capture['raw_received_lines'];return result
            except Exception as exc:
                return [dict(capture if i==0 else {},status='failed',error=repr(exc),shared_transport_record=True) for i in range(c['parallel'])]
        barrier=threading.Barrier(c['parallel'])
        def one(index):
            capture={'request_index':index,'raw_received_lines':[],'post_retry_count':0};connection=None
            try:
                if c['client']=='preconnect_http':
                    parsed=urlsplit(url);connection=http.client.HTTPConnection(parsed.hostname,parsed.port or 80,timeout=c['request_timeout_seconds']);connection.connect()
                def opened(req,timeout=None):
                    if connection is None:response=urlopen(req,timeout=c['request_timeout_seconds'])
                    else:
                        parsed=urlsplit(req.full_url);target=(parsed.path or '/')+(('?'+parsed.query) if parsed.query else '')
                        connection.request('POST',target,body=req.data,headers=dict(req.header_items()));response=connection.getresponse()
                        if not 200<=response.status<300:
                            raw=response.read();response.close();raise HTTPError(req.full_url,response.status,response.reason,response.headers,io.BytesIO(raw))
                    return _CapturedResponse(response,capture,drain=connection is not None)
                parser=types.FunctionType(self.parser.__code__,dict(self.parser.__globals__,urlopen=opened))
                barrier.wait(timeout=min(60,c['request_timeout_seconds']))
                r,b=parser(url,payload);b['submission_mode']=c['client'];b['post_retry_count']=0
                capture.update(status='complete',response=r,boundary=b)
            except Exception as exc:barrier.abort();capture.update(status='failed',error=repr(exc))
            finally:
                if connection is not None:connection.close()
            return capture
        futures=[self.pool.submit(one,i) for i in range(c['parallel'])]
        return [f.result() for f in futures]
    def close(self):self.pool.shutdown(wait=True)

def parse_power_scheme_output(stdout,stderr,returncode):
    # GUID is ASCII even when the localized powercfg text changes code page.
    match=re.search(rb"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",stdout)
    return {"returncode":returncode,"stdout":stdout.decode("utf-8",errors="replace"),
            "stderr":stderr.decode("utf-8",errors="replace"),"stdout_raw_hex":stdout.hex(),
            "guid":match.group(0).decode("ascii").lower() if match else None}

def thread_affinities(pid):
    if sys.platform != 'win32':return {"status":"not_applicable","threads":[]}
    import ctypes,psutil
    class Group(ctypes.Structure):
        _fields_=[("mask",ctypes.c_size_t),("group",ctypes.c_ushort),("reserved",ctypes.c_ushort*3)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenThread.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenThread.restype=ctypes.c_void_p
    kernel.GetThreadGroupAffinity.argtypes=[ctypes.c_void_p,ctypes.POINTER(Group)];kernel.GetThreadGroupAffinity.restype=ctypes.c_int
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    rows=[]
    for thread in psutil.Process(pid).threads():
        handle=kernel.OpenThread(0x40,False,thread.id)
        try:
            group=Group();ok=bool(handle and kernel.GetThreadGroupAffinity(handle,ctypes.byref(group)))
            rows.append({"tid":thread.id,"ok":ok,"mask":hex(group.mask) if ok else None,
                         "group":group.group if ok else None,"error":None if ok else ctypes.get_last_error()})
        finally:
            if handle:kernel.CloseHandle(handle)
    return {"status":"captured","threads":rows}

def process_state(pid,telemetry=True):
    import psutil
    p=psutil.Process(pid)
    result={'thread_affinities':thread_affinities(pid),'captured_utc':now(),'pid':pid,'create_time':p.create_time(),'priority':int(p.nice()),'cpu_affinity':p.cpu_affinity(),'memory_available_bytes':psutil.virtual_memory().available}
    if sys.platform=='win32':
        cp=subprocess.run(['powercfg','/getactivescheme'],capture_output=True,timeout=10)
        result['power_scheme']=parse_power_scheme_output(cp.stdout,cp.stderr,cp.returncode)
    if telemetry:
        try:
            cp=subprocess.run(['nvidia-smi','--query-gpu=uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            result['gpu_state']={'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
        except (OSError,subprocess.TimeoutExpired) as exc:result['gpu_state']={'status':'unavailable','error':repr(exc)}
    return result
def validate_state(state,c,check_workers=True,check_gpu=True):
    if c.get('resolved_cpu_affinity') is not None and sorted(state['cpu_affinity'])!=c['resolved_cpu_affinity']:raise ValueError('requested affinity not observed')
    if sys.platform=='win32' and state['priority']!={0:0x20,1:0x8000,2:0x80}[c['priority']]:raise ValueError('requested native priority not observed')
    if check_workers and c.get('worker_cpu_mask') is not None and sys.platform=='win32':
        mask=int(str(c['worker_cpu_mask']),0)
        expected={hex(1<<i) for i in range(mask.bit_length()) if mask&(1<<i)}
        observed={r['mask'] for r in state.get('thread_affinities',{}).get('threads',[]) if r.get('ok') and r.get('group')==0}
        if not expected.issubset(observed):raise ValueError('strict worker affinity masks not observed')
    if c.get('expected_power_scheme') and state.get('power_scheme',{}).get('guid')!=c['expected_power_scheme'].lower():raise ValueError('expected power scheme not observed')
    if check_gpu and c.get('expected_gpu_sm_clock_mhz') is not None:
        target=float(c['expected_gpu_sm_clock_mhz']);tolerance=float(c.get('gpu_sm_clock_tolerance_mhz',30))
        if not math.isfinite(target) or target<=0 or not math.isfinite(tolerance) or tolerance<0:raise ValueError('invalid expected GPU SM clock/tolerance')
        gpu=state.get('gpu_state',{});rows=list(csv.reader(io.StringIO(gpu.get('stdout',''))))
        if gpu.get('returncode')!=0 or not rows:raise ValueError('expected GPU SM clock unavailable')
        for row in rows:
            match=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:MHz)?\s*',row[3]) if len(row)>3 else None
            if not match:raise ValueError('expected GPU SM clock unavailable')
            if abs(float(match.group(1))-target)>tolerance:raise ValueError('expected GPU SM clock not observed: '+row[3])
def command_for(exe,c,port):
    cmd=[str(exe),'-m',c['model'],'--host','127.0.0.1','--port',str(port),'-c',str(c['ctx']),'-ngl',str(c['gpu_layers']),'-np',str(c['parallel']),
      '-b',str(c['batch']),'-ub',str(c['ubatch']),'-t',str(c['threads']),'-tb',str(c['threads_batch']),'-fa','on' if c['flash_attention'] else 'off',
      '--load-mode',c['load_mode'],'-kvo','--op-offload','-sm','layer','-mg','0','-ctk','f16','-ctv','f16','-kvu','-cb' if c['cont_batching'] else '-nocb',
      '--cache-ram',str(c['cache_ram_mib']),'--perf','--metrics','--warmup','--spec-type','none','--log-verbosity',str(c['log_verbosity']),'--prio',str(c['priority']),'--prio-batch',str(c['priority'])]
    if 'fit_params' in c:cmd+=['--fit','on' if c['fit_params'] else 'off']
    if 'kv_unified_per_slot' in c:cmd+=['--kv-unified-per-slot',str(c['kv_unified_per_slot'])]
    if 'poll' in c:cmd+=['--poll',str(c['poll']),'--poll-batch',str(c.get('poll_batch',c['poll']))]
    if c.get('worker_cpu_mask') is not None:
        mask=hex(int(str(c['worker_cpu_mask']),0))
        cmd+=['--cpu-mask',mask,'--cpu-strict','1','--cpu-mask-batch',mask,'--cpu-strict-batch','1']
    return cmd
def helpers():
    from tools.native_llama_compare import probe_hardware,_hardware_fingerprint,native_extractor_identity,wait_health,post_json
    from tools.native_runtime_evidence import capture_loaded_runtime
    return probe_hardware,_hardware_fingerprint,native_extractor_identity,wait_health,post_json,capture_loaded_runtime

def run_block(c,exe,output,freeze,control=None):
    import psutil
    _,_,_,health,post,capture=helpers();key=c['key'];child_env,env_identity=environment_for(c['environment'])
    if env_identity!=freeze['environments'][key]:raise ValueError('runtime environment drift')
    record={'schema':'native-repeatability-block/v1','key':key,'config':c,'freeze_sha256':digest(freeze),'started_utc':now(),'status':'incomplete',
      'warmup':[],'runs':[],'batch_journal_refs':[],'execution_environment':env_identity,'hardware_fingerprint':freeze['hardware_fingerprint'],
      'artifact_refs':freeze['artifact_refs'],'timing_contract':'engine_request_begin_to_sample_accept_token;fixed_output_no_prompt_cache',
      'raw_wire_observer':'received line timestamps in client memory; no trace/CUPTI profiler','post_retry_count':0}
    proc=client=None;log_path=output/(key+'.log')
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        command=command_for(exe,c,port);record['actual_argv']=command
        with log_path.open('x',encoding='utf-8') as log:proc=subprocess.Popen(command,stdout=log,stderr=log,env=child_env)
        if c.get('resolved_cpu_affinity') is not None:psutil.Process(proc.pid).cpu_affinity(c['resolved_cpu_affinity'])
        base=f'http://127.0.0.1:{port}';health(base,proc)
        record['runtime_before']=capture(proc.pid,exe)
        record['runtime_baseline_ref']=bind_runtime_identity(output,freeze,key,record['runtime_before'])
        record['state_before']=process_state(proc.pid,c['telemetry']);validate_state(record['state_before'],c,check_workers=False,check_gpu=False)
        record['tokenizer'],ids,content=resolve_prompt(c,post,base,record)
        record['prompt_token_count']=len(ids)
        payload={'prompt':content,'n_predict':c['output'],'ignore_eos':True,'cache_prompt':False,'temperature':0,'top_k':1,'seed':c['seed'],'stream':True}
        record['payload']=payload;client=TrialClient(c)
        for phase,count in (('warmup',c['warmup_batches']),('runs',c['measure_batches'])):
            if phase=='runs':
                if control is not None:control.check()
                record['state_measurement_before']=process_state(proc.pid,c['telemetry'])
                validate_state(record['state_measurement_before'],c)
            for repeat in range(count):
                if control is not None:control.check()
                before=time.perf_counter();responses=client.batch(base+'/completion',payload)
                wall_ms=(time.perf_counter()-before)*1000
                record['batch_journal_refs'].append(write_batch_journal(output,freeze,c,phase,repeat,len(ids),responses,wall_ms))
                batch=inspect_batch(responses,c,len(ids));batch.update(repeat=repeat,process_block=c['block'],phase=phase,batch_wall_ms=wall_ms)
                record[phase].append(batch)
                if batch['status']!='complete':raise ValueError(f'{phase} batch {repeat} failed; partial responses preserved')
        record['state_after']=process_state(proc.pid,c['telemetry']);validate_state(record['state_after'],c)
        record['runtime_after']=capture(proc.pid,exe)
        if record['runtime_after'].get('status')!='captured' or record['runtime_before'].get('module_identity_sha256')!=record['runtime_after'].get('module_identity_sha256'):raise ValueError('loaded runtime module identity drift')
        verify_refs(freeze['source_refs']);verify_refs([record['runtime_baseline_ref']]);record['status']='complete'
    except BatchBoundaryStop as exc:record.update(status='incomplete',stop_reason=str(exc))
    except Exception as exc:record.update(status='failed',error=repr(exc))
    finally:
        if client is not None:client.close()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            record['process_returncode']=proc.returncode
        record['completed_utc']=now();record['log_ref']=file_ref(log_path) if log_path.exists() else None
        raw=output/(key+'.json');write_new(raw,record)
        write_new(output/(key+'.receipt.json'),{'key':key,'status':record['status'],'freeze_sha256':digest(freeze),'raw_ref':file_ref(raw),'log_ref':record['log_ref']})
    return record

def source_files():
    return sorted(set(ROOT.joinpath('tools').glob('*.py'))|set(ROOT.joinpath('src').rglob('*.py'))|{Path(__file__).resolve()})
def build_freeze(protocol_path,p,plans,exe,output):
    probe,fingerprint,extractor,_,_,_=helpers();hardware=probe();fp=fingerprint(hardware)
    if not fp:raise ValueError('actual hardware fingerprint missing')
    sources=[file_ref(path) for path in source_files()]
    paths={exe,*exe.parent.glob('*.dll'),*(Path(c['model']) for c in plans)}
    paths.update(project_file(path) for path in p.get('evidence_refs',[]))
    artifacts=[file_ref(path) for path in sorted(paths)]
    snapshot=output/'sources.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for ref in sources:archive.write(ref['path'],Path(ref['path']).relative_to(ROOT).as_posix())
    freeze={'schema':'native-repeatability-freeze/v1','created_utc':now(),'protocol_ref':file_ref(protocol_path),
      'source_refs':sources,'source_snapshot_ref':file_ref(snapshot),'artifact_refs':artifacts,'plans':plans,
      'data_root':str(data_root()),'execution_source_root':str(ROOT.resolve()),
      'hardware':hardware,'hardware_fingerprint':fp,'extractor_identity':extractor(),
      'environments':{c['key']:environment_for(c['environment'])[1] for c in plans},
      'block_order':'even process blocks forward; odd process blocks reverse (two conditions ABBA)',
      'criterion':'ALL engine request-by-rank AND batch-median deviations within +/-5%; full coverage; >=2 independent process blocks',
      'scope':'native repeatability development; not simulator accuracy or independent acceptance'}
    verify_refs(sources);return freeze

def verify_freeze(freeze,protocol_path,plans,verify_artifacts=True):
    if freeze.get('data_root',str(ROOT.resolve()))!=str(data_root()) or freeze.get('execution_source_root',str(ROOT.resolve()))!=str(ROOT.resolve()):
        raise ValueError('data/source root freeze mismatch')
    if freeze.get('schema')!='native-repeatability-freeze/v1' or not freeze.get('hardware_fingerprint') or not freeze.get('extractor_identity',{}).get('sha256'):raise ValueError('missing required freeze identity')
    verify_refs(freeze.get('source_refs'));verify_refs([freeze.get('source_snapshot_ref'),freeze.get('protocol_ref')])
    if freeze['protocol_ref']['path']!=str(Path(protocol_path).resolve()) or freeze.get('plans')!=plans:raise ValueError('protocol/plan freeze mismatch')
    envs={c['key']:environment_for(c['environment'])[1] for c in plans}
    if freeze.get('environments')!=envs:raise ValueError('environment freeze mismatch')
    if verify_artifacts:verify_refs(freeze.get('artifact_refs'))
    probe,fingerprint,extractor,_,_,_=helpers()
    if fingerprint(probe())!=freeze['hardware_fingerprint']:raise ValueError('actual hardware fingerprint mismatch')
    if extractor().get('sha256')!=freeze['extractor_identity']['sha256']:raise ValueError('native extractor identity mismatch')

def load_completed(output,freeze,plan):
    receipt_path=output/(plan['key']+'.receipt.json');raw_path=output/(plan['key']+'.json')
    if not receipt_path.exists():
        if raw_path.exists() or (output/(plan['key']+'.log')).exists():raise ValueError('orphan partial block; preserve evidence and use new output: '+plan['key'])
        return None
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('key')!=plan['key'] or receipt.get('freeze_sha256')!=digest(freeze):raise ValueError('receipt identity mismatch')
    refs=[receipt.get('raw_ref')]
    if receipt.get('log_ref'):refs.append(receipt['log_ref'])
    verify_refs(refs)
    if receipt['raw_ref']['path']!=str(raw_path.resolve()):raise ValueError('raw points to wrong block')
    record=json.loads(raw_path.read_text(encoding='utf-8'))
    if record.get('config')!=plan or record.get('freeze_sha256')!=digest(freeze) or record.get('status')!=receipt.get('status'):raise ValueError('raw source/config/status mismatch')
    expected=len(plan['prompt_token_ids']) if plan.get('prompt_token_ids') is not None else plan.get('expected_prompt_tokens')
    if expected is not None and record.get('prompt_token_count')!=expected and (record.get('warmup') or record.get('runs')):
        raise ValueError('saved prompt count differs from frozen protocol')
    extra_refs=record.get('batch_journal_refs',[])+([record['runtime_baseline_ref']] if record.get('runtime_baseline_ref') else [])
    if extra_refs:verify_refs(extra_refs)
    # Hashes bind immutable raw evidence; still recompute derived metrics from
    # the saved raw request responses so summary contents are never trusted.
    for phase in ('warmup','runs'):
        recomputed=[]
        for batch in record.get(phase,[]):
            captures=[]
            for row in batch['requests']:
                capture=dict(row)
                if row.get('response') is not None and row.get('boundary') is not None:capture['status']='complete'
                captures.append(capture)
            computed=inspect_batch(captures,plan,record.get('prompt_token_count'))
            computed.update({k:v for k,v in batch.items() if k not in computed})
            recomputed.append(computed)
        record[phase]=recomputed
    return record

def main(argv=None):
    global DATA_ROOT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-seconds',type=float,default=5400);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--data-root',type=Path,help='read-only model/native/evidence root; imports and source freeze stay beside this script')
    args=ap.parse_args(argv)
    DATA_ROOT=args.data_root.resolve(strict=True) if args.data_root is not None else None
    if not data_root().is_dir():raise ValueError('data root must be directory')
    if args.max_seconds<=0:raise ValueError('positive budget required')
    protocol_path=args.protocol.resolve(strict=True);p=json.loads(protocol_path.read_text(encoding='utf-8-sig'));plans=expand_protocol(p)
    exe=project_file(p.get('exe','source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe'));output=args.output.resolve()
    if args.resume:
        if not output.is_dir():raise ValueError('resume output does not exist')
        freeze=json.loads((output/'freeze.json').read_text(encoding='utf-8'))
    else:
        output.mkdir(parents=True,exist_ok=False);freeze=build_freeze(protocol_path,p,plans,exe,output);write_new(output/'freeze.json',freeze)
    verify_freeze(freeze,protocol_path,plans)
    records={}
    for plan in plans:
        record=load_completed(output,freeze,plan)
        if record is not None:records[plan['key']]=record
    started=time.monotonic();budget_hit=False;stop_reason=None;premeasurement_failures=0
    preflight=process_state(os.getpid(),False)
    if sys.platform=='win32' and not preflight.get('power_scheme',{}).get('guid'):raise ValueError('power scheme preflight failed')
    control=BatchBoundaryControl(output,started+args.max_seconds)
    with defer_keyboard_interrupts(control):
        for plan in plans:
            if plan['key'] in records:continue
            stop_reason=control.reason()
            if stop_reason:
                budget_hit=stop_reason=='budget_exhausted_at_batch_boundary';break
            verify_refs(freeze['source_refs'])
            record=run_block(plan,exe,output,freeze,control=control);records[plan['key']]=record
            print(json.dumps({'key':plan['key'],'status':record['status'],'measured_batches':len(record['runs']),'error':record.get('error'),'stop_reason':record.get('stop_reason')}),flush=True)
            if record.get('stop_reason'):
                stop_reason=record['stop_reason'];budget_hit=stop_reason=='budget_exhausted_at_batch_boundary';break
            premeasurement_failures=premeasurement_failures+1 if record['status']=='failed' and not record['runs'] else 0
            if premeasurement_failures>=2:stop_reason='repeated_premeasurement_failure';break
    end_verified=True;end_error=None
    try:verify_freeze(freeze,protocol_path,plans)
    except Exception as exc:end_verified=False;end_error=repr(exc)
    groups={}
    for plan in plans:groups.setdefault((plan['job_id'],plan['condition_id']),[]).append(plan)
    summaries=[summarize_group(cs,[records[c['key']] for c in cs if c['key'] in records]) for cs in groups.values()]
    if not end_verified:
        for group in summaries:
            group['strict_within_5pct']=False;group['freeze_invalid']=True
            for metric in group['metrics'].values():
                if metric.get('status')!='not_applicable':metric['strict_within_5pct']=False
    failures=[{'key':key,'error':r.get('error')} for key,r in records.items() if r['status']=='failed']
    incomplete_blocks=[key for key,r in records.items() if r['status'] not in ('complete','failed')]
    result={'schema':'native-repeatability-results/v1','completed_utc':now(),'freeze_sha256':digest(freeze),
      'end_verified':end_verified,'end_verification_error':end_error,'budget_seconds':args.max_seconds,
      'budget_elapsed_seconds':time.monotonic()-started,'budget_exhausted':budget_hit,'stop_reason':stop_reason,'planned_blocks':len(plans),'captured_blocks':len(records),
      'missing_blocks':[c['key'] for c in plans if c['key'] not in records],'incomplete_blocks':incomplete_blocks,'failures':failures,'groups':summaries,
      'strict_within_5pct':bool(end_verified and summaries and all(g['strict_within_5pct'] for g in summaries)),'formal_prediction_acceptance':False}
    result['status']='failed' if not end_verified or failures else 'incomplete' if incomplete_blocks or len(records)<len(plans) else 'complete'
    index=1
    while (output/f'summary_{index:03d}.json').exists():index+=1
    target=output/f'summary_{index:03d}.json';write_new(target,result)
    print(json.dumps({'summary':str(target),'status':result['status'],'strict_within_5pct':result['strict_within_5pct']}),flush=True)
    return 1 if result['status']=='failed' else 2 if result['status']=='incomplete' else 0
if __name__=='__main__':raise SystemExit(main())
