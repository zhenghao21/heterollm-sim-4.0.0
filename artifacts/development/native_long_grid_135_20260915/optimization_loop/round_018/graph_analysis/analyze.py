"""Read-only R18 microbenchmark analysis. No GPU access, no fit, no target LLM actuals."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median
import datetime, hashlib, json, math, re
P=Path(__file__).resolve().parent;R=P.parent;C=R/'collection';ROOT=next(p for p in P.parents if (p/'pyproject.toml').is_file())
REFS={}
def ref(p):
    p=Path(p).resolve();b=p.read_bytes();r={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};REFS[str(p)]=r;return r

def read(p):
    p=Path(p).resolve();ref(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def check(r):
    found=ref(r['path'])
    assert found['sha256']==r['sha256']
    assert found['bytes']==r.get('bytes',r.get('size_bytes'))

def q(v,f):
    v=sorted(v);i=(len(v)-1)*f;lo=int(i);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(i-lo)
def stats(values):
    v=list(values)
    if not v:return {'count':0}
    out={'count':len(v),'min':min(v),'median':median(v),'p10':q(v,.1),'p90':q(v,.9),'max':max(v)}
    out['p90_div_p10']=out['p90']/out['p10'] if out['p10']>0 else None;return out

def merged(intervals):
    out=[]
    for a,b in sorted(intervals):
        assert type(a) is int and type(b) is int and a<=b
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out

def duration(intervals):return sum(b-a for a,b in merged(intervals))
def intersection(a,b):return duration((max(x,z),min(y,w)) for x,y in merged(a) for z,w in merged(b) if max(x,z)<min(y,w))
def ranks(v):
    ordered=sorted(enumerate(v),key=lambda t:t[1]);r=[0.0]*len(v);i=0
    while i<len(v):
        j=i+1
        while j<len(v) and ordered[j][1]==ordered[i][1]:j+=1
        rank=(i+j-1)/2
        for k in range(i,j):r[ordered[k][0]]=rank
        i=j
    return r

def spearman(x,y):
    if len(x)<3:return None
    x,y=ranks(x),ranks(y);mx=sum(x)/len(x);my=sum(y)/len(y)
    xx=sum((a-mx)**2 for a in x);yy=sum((b-my)**2 for b in y)
    if not xx or not yy:return None
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy)

def app_rows(path):
    ref(path);records=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    header=records[0];footer=records[-1];assert header['record']=='header' and footer['status']=='complete'
    f=header['qpc_frequency'];setup=next(r for r in records if r['record']=='setup')
    validations={r['label']:r for r in records if r['record']=='validation'}
    out=[];previous=None
    for c in (r for r in records if r['record']=='graph_call'):
        v=validations[c['label']];r={'phase':v['phase'],'index':v['index'],'label':c['label'],
            'wall_ns':(c['qpc_sync_end']-c['qpc_submit_start'])*1e9/f,
            'submit_ns':(c['qpc_submit_end']-c['qpc_submit_start'])*1e9/f,
            'sync_ns':(c['qpc_sync_end']-c['qpc_sync_start'])*1e9/f,
            'instrumentation_gap_ns':(c['qpc_sync_start']-c['qpc_submit_end'])*1e9/f,
            'final_validation_ns':(v['qpc_validation_end']-v['qpc_validation_start'])*1e9/f,
            'preceding_outside_graph_gap_ns':None if previous is None else (c['qpc_submit_start']-previous['qpc_sync_end'])*1e9/f}
        assert abs(r['wall_ns']-r['submit_ns']-r['sync_ns']-r['instrumentation_gap_ns'])<.01
        out.append(r);previous=c
    assert len(out)==36
    return out,setup

def grouped(rows,field,phase):return stats(r[field] for r in rows if r['phase']==phase and r.get(field) is not None)

def trace_rows(mapped,raw):
    labels=defaultdict(list)
    for n in raw['tables']['NVTX_EVENTS']:
        if n.get('end') is not None:labels[n['resolved_text']].append(n)
    out=[]
    for c in mapped['calls']:
        label=c['nvtx_verbatim']['resolved_text'];host=labels[label+'/host_submit'];sync=labels[label+'/final_sync'];assert len(host)==len(sync)==1
        host,sync=host[0],sync[0];ks=[p['kernel'] for p in c['kernel_launch_pairs']]
        assert len({k['evidence_rowid'] for k in ks})==len(ks)==c['actual_kernel_count']
        ki=[(k['start'],k['end']) for k in ks];api=c['api_events_verbatim'];launch=[a for a in api if 'Launch' in a['name_text']]
        li=[(a['start'],a['end']) for a in launch];si=[(sync['start'],sync['end'])];hi=[(host['start'],host['end'])]
        ku=duration(ki);assert ku==c['kernel_union_ns'];assert sum(c['exclusive_partition_ns'].values())==c['nvtx_full_call_duration_ns']
        kernel_sum=sum(b-a for a,b in ki);last=max(b for a,b in ki);first=min(a for a,b in ki)
        pairs=sorted(c['kernel_launch_pairs'],key=lambda v:v['kernel']['start'])
        previous_overlap=0;future_submitted_early=0
        for left,right in zip(pairs,pairs[1:]):
            previous=left['kernel'];next_api=right['api']
            previous_overlap+=intersection([(previous['start'],previous['end'])],[(next_api['start'],next_api['end'])])>0
            future_submitted_early+=next_api['start']<previous['end']
        calls={'phase':c['phase'],'index':c['index'],'label':label,'kernel_count':len(ks),
            'graph_launch_API_records':c['graph_launch_API_count'],'graph_capture_update_API_records':c['graph_capture_update_API_count'],
            'kernel_union_ns':ku,'kernel_sum_ns':kernel_sum,'kernel_temporal_overlap_ns':kernel_sum-ku,
            'kernel_span_ns':c['kernel_span_ns'],'inter_kernel_gap_ns':c['inter_kernel_gap_ns'],
            'GPU_during_host_submit_ns':intersection(ki,hi),'GPU_during_final_sync_ns':intersection(ki,si),
            'GPU_overlap_host_launch_API_ns':intersection(ki,li),'first_kernel_before_submit_returns':first<host['end'],
            'last_kernel_complete_before_final_sync':last<=sync['start'],
            'subsequent_API_overlaps_previous_kernel_count':previous_overlap,
            'subsequent_API_begins_before_previous_kernel_end_count':future_submitted_early,
            'potential_successor_pairs':len(pairs)-1,
            'launch_API_union_ns':duration(li),'host_submit_NVTX_ns':host['end']-host['start'],
            'final_sync_NVTX_ns':sync['end']-sync['start'],'post_last_kernel_to_sync_return_ns':max(0,sync['end']-max(last,sync['start'])),
            'nvtx_wall_ns':c['nvtx_full_call_duration_ns'],'partition_ns':c['exclusive_partition_ns'],
            'API_counts':dict(Counter(a['name_text'] for a in api)),
            'API_tables':dict(Counter(a['evidence_table'] for a in api))}
        assert not c['issues'];out.append(calls)
    return out

def main():
    summary=read(C/'analysis_final_v1/summary.json');check(summary['freeze_ref']);freeze=read(C/'freeze.json')
    assert summary['diagnostic_accepted_configs']==0
    lifecycle=read(C/'graph_clock_control_finish.json');assert lifecycle['complete_stage_receipts']==54 and lifecycle['clock_reset']['returncode']==0
    for k in ('stdout_ref','stderr_ref'):check(lifecycle['clock_reset'][k])
    protocol=read(C/'protocol.json');results=[];all_trace=[];quality_reasons=Counter()
    for cfg in summary['configs']:
        config=cfg['config'];pairs=[]
        for pi,pair in enumerate(cfg['pairs']):
            for r in [pair['profile_raw']['source'],pair['direct_raw']['source'],*pair['trace_sources'].values()]:check(r)
            mapped=read(pair['trace_sources']['mapped']['path']);raw=read(pair['trace_sources']['raw']['path']);trace=trace_rows(mapped,raw);all_trace.extend(trace)
            arms={};origins=[]
            for mode in ('direct','profile'):
                rows,setup=app_rows(pair[mode+'_raw']['source']['path']);origins.extend(setup['loaded_modules_before'])
                per_phase={phase:{field:grouped(rows,field,phase) for field in ['wall_ns','submit_ns','sync_ns','instrumentation_gap_ns','final_validation_ns','preceding_outside_graph_gap_ns']} for phase in ['first','warmup','formal']}
                formal=[r for r in rows if r['phase']=='formal']
                corr={field:spearman([r['preceding_outside_graph_gap_ns'] for r in formal],[r[field] for r in formal]) for field in ['wall_ns','submit_ns','sync_ns']}
                arms[mode]={'phase_stats_ns':per_phase,'formal_gap_spearman':corr,'calls':rows}
            dll=[r for r in origins if r['name'].lower()=='ggml-cuda.dll'];assert len(dll)==2 and len({d['sha256'] for d in dll})==1
            trace_stats={phase:{field:grouped(trace,field,phase) for field in ['kernel_union_ns','kernel_sum_ns','kernel_span_ns','inter_kernel_gap_ns','GPU_during_host_submit_ns','GPU_during_final_sync_ns','GPU_overlap_host_launch_API_ns','launch_API_union_ns','host_submit_NVTX_ns','final_sync_NVTX_ns','post_last_kernel_to_sync_return_ns','nvtx_wall_ns']} for phase in ['first','warmup','formal']}
            pairs.append({'pair':pi,'quality_issues':[s for s in cfg['issues'] if s.startswith(f'pair_{pi}:')],
                'native_cuda_sha256':dll[0]['sha256'],'clock_pass':pair['clock_domain_validated'],'numerics_pass':pair['numerics_all_rows'],
                'trace_complete':pair['trace_chain_complete'],'direct':arms['direct'],'profile':arms['profile'],
                'trace_phase_stats_ns':trace_stats,'trace_calls':trace,
                'profile_direct_formal_wall_difference':abs(pair['profile_host']['median_ns']-pair['direct_host']['median_ns'])/pair['direct_host']['median_ns']})
        for s in cfg['issues']:quality_reasons[s.split(':')[-1]]+=1
        aggregated={}
        for phase in ['first','warmup','formal']:
            t=[row for pair in pairs for row in pair['trace_calls'] if row['phase']==phase]
            part={k:stats(r['partition_ns'][k] for r in t) for k in t[0]['partition_ns']}
            aggregated[phase]={'trace_call_count':len(t),'observed_kernels':sum(r['kernel_count'] for r in t),
                'GraphLaunch_API_records':sum(r['graph_launch_API_records'] for r in t),
                'graph_capture_or_update_API_records':sum(r['graph_capture_update_API_records'] for r in t),
                'trace_kernel_union_ns':stats(r['kernel_union_ns'] for r in t),
                'trace_kernel_sum_minus_union_ns':stats(r['kernel_temporal_overlap_ns'] for r in t),
                'GPU_during_submit_fraction':sum(r['GPU_during_host_submit_ns'] for r in t)/sum(r['kernel_union_ns'] for r in t),
                'GPU_during_sync_fraction':sum(r['GPU_during_final_sync_ns'] for r in t)/sum(r['kernel_union_ns'] for r in t),
                'GPU_overlap_launch_API_fraction':sum(r['GPU_overlap_host_launch_API_ns'] for r in t)/sum(r['kernel_union_ns'] for r in t),
                'kernel_starts_before_submit_returns':sum(r['first_kernel_before_submit_returns'] for r in t),
                'last_kernel_complete_before_sync_starts':sum(r['last_kernel_complete_before_final_sync'] for r in t),
                'successor_API_overlaps_previous_kernel_count':sum(r['subsequent_API_overlaps_previous_kernel_count'] for r in t),
                'successor_API_begins_before_previous_kernel_end_count':sum(r['subsequent_API_begins_before_previous_kernel_end_count'] for r in t),
                'potential_successor_pairs':sum(r['potential_successor_pairs'] for r in t),
                'exclusive_partition_ns':part,
                'direct_pair_wall_medians_ns':[a['direct']['phase_stats_ns'][phase]['wall_ns']['median'] for a in pairs],
                'profile_pair_wall_medians_ns':[a['profile']['phase_stats_ns'][phase]['wall_ns']['median'] for a in pairs],
                'direct_pair_submit_medians_ns':[a['direct']['phase_stats_ns'][phase]['submit_ns']['median'] for a in pairs],
                'direct_pair_sync_medians_ns':[a['direct']['phase_stats_ns'][phase]['sync_ns']['median'] for a in pairs],
                'direct_pair_validation_medians_ns':[a['direct']['phase_stats_ns'][phase]['final_validation_ns']['median'] for a in pairs],
                'direct_pair_preceding_gap_medians_ns':[a['direct']['phase_stats_ns'][phase]['preceding_outside_graph_gap_ns'].get('median') for a in pairs]}
        results.append({'config':config,'verdict':cfg['status'],'quality_issues':cfg['issues'],'frozen_quality_policy':cfg['fixed_policy'],
            'profile_process_median_max_relative_deviation':cfg['profile_process_median_max_relative_deviation'],
            'phase_summary':aggregated,'pairs':pairs})
    source_evidence={}
    cache=ROOT/'source/llama.cpp-semantic/build-semantic-direct/CMakeCache.txt';ref(cache);assert 'GGML_CUDA_GRAPHS:BOOL=OFF' in cache.read_text()
    annotation=read(ROOT/'source/llama.cpp-annotation-control/evidence/build_receipt.json')
    source_evidence['cuda_compile_steps']=[{'label':s['label'],'returncode':s['returncode'],'cuda_graph_macro_args':[a for a in s['argv'] if 'CUDA_GRAPH' in a]} for s in annotation['steps'] if 'ggml-cuda' in s['label']]
    source_evidence['cuda_output_SHA']={k:v for k,v in annotation['output_sha256'].items() if k.endswith('ggml-cuda.dll')}
    source_evidence['cmake_cuda_graphs']=False;source_evidence['prior_prediction_static_contracts']={}
    for name in ('pure','current','conversion_cta'):
        old=read(R.parent/'round_017'/name/'freeze.json')
        source_evidence['prior_prediction_static_contracts'][name]={'count':len(old['cells']),'compiled_cuda_graphs_counts':dict(Counter(str(c['static_inputs']['config'].get('compiled_cuda_graphs')) for c in old['cells'])),'binding':old['runtime_build_audit']['binding'],'artifact_sha256':old['runtime_build_audit']['artifact_sha256']}
        check(old['runtime_build_audit']['audit_ref'])
    paths=['source/llama.cpp-annotation-control/ggml/src/ggml-cuda/ggml-cuda.cu','source/llama.cpp-semantic/ggml/src/ggml-cuda/scale.cu','source/llama.cpp-semantic/ggml/src/ggml-cuda/scale.cuh','source/llama.cpp-semantic/ggml/src/ggml-cuda/CMakeLists.txt',
        'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_017/current/source/src/heterollm_sim/reference.py',
        'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_017/current/source/src/heterollm_sim/planner.py',
        'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_017/current/source/src/heterollm_sim/cost_models.py',
        'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_017/current/source/tools/native_llama_compare.py',
        'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_017/current/source/src/heterollm_sim/llama_gpu_invocations.py']
    for path in paths:ref(ROOT/path)
    source_evidence['per_kernel_launch_static_default_ns']=1000.0
    source_evidence['zero_launch_hypothesis']='contradicted by reference default1000; native builder replaces tensor fields only, not launch field; no zero-value assumption permitted'
    totals={'stage_receipts':54,'pairs':18,'configs':6,'profile_calls':len(all_trace),'direct_calls':648,
        'profile_kernels_observed':sum(r['kernel_count'] for r in all_trace),
        'GraphLaunch_API_records':sum(r['graph_launch_API_records'] for r in all_trace),
        'graph_capture_or_update_API_records':sum(r['graph_capture_update_API_records'] for r in all_trace),
        'runtime_API_counts':dict(sum((Counter(r['API_counts']) for r in all_trace),Counter())),
        'API_table_counts':dict(sum((Counter(r['API_tables']) for r in all_trace),Counter())),
        'configuration_quality_accepted':0,'configuration_quality_rejected':6,'quality_rejection_counts':dict(quality_reasons),
        'all_clock_numeric_trace_identity_passed':all(p['clock_pass'] and p['numerics_pass'] and p['trace_complete'] for c in results for p in c['pairs'])}
    assert totals['profile_calls']==648 and totals['profile_kernels_observed']==8856 and totals['GraphLaunch_API_records']==0 and totals['graph_capture_or_update_API_records']==0
    result={'schema':'R18-source-bound-graph-overlap-analysis/v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'verdict':'0_of_6_remain_rejected_not_calibration_ready','totals':totals,'source_evidence':source_evidence,'configs':results,
        'policy':{'no_fit':True,'coefficients_emitted':0,'target_LLM_actuals_read':False,'GPU_access':False,
            'clock_arithmetic':'QPC only for native wall/gaps; Nsight only for kernel/API/range overlap; never cross-domain subtraction',
            'correlations':'within-process formal descriptive Spearman only; no model fit or parameter use',
            'direct_device_timing':'unobserved, no profile kernel duration copied into direct result',
            'use':'qualitative mechanism and workload provenance only; no transfer proof or relaxed acceptance'},
        'sources':list(REFS.values())}
    for r in result['sources']:check(r)
    result['source_identities_rechecked_after']=True
    with (P/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(totals,ensure_ascii=False))
if __name__=='__main__':main()
