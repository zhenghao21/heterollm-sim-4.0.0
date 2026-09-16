"""R20 raw evidence contract. No native calls, no cost fitting."""
from pathlib import Path
import json,hashlib,re,math
from native_trace import distribution,expected_calls,ref

def load(path):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key: '+k)
            d[k]=v
        return d
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)

def require(condition, message):
    if not condition:raise ValueError(message)

def audit(path, stage, manifest, protocol, process_pid=None, extra_runtime_refs=()):
    path=Path(path).resolve(); rows=[json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    def one(kind):
        x=[r for r in rows if r.get('record')==kind];require(len(x)==1,'one '+kind+' required');return x[0]
    h,s,f,a=one('header'),one('setup'),one('footer'),one('arm_metadata')
    arm=stage['arm']; config=stage['config']; pair=stage['pair_id']; n,e=config['nodes'],config['elements']
    require(h.get('schema')=='graph-gap-probe/v1' and h.get('protocol_sha256')==manifest['protocol']['sha256'],'header schema/protocol')
    require(type(h.get('pid')) is int and 0<h['pid']<2**24,'actual PID')
    if process_pid is not None:require(h['pid']==process_pid,'direct PID differs from launcher')
    require(type(h.get('qpc_frequency')) is int and h['qpc_frequency']>0,'QPC frequency')
    require(type(s.get('scheduling',{}).get('caller_thread_id')) is int and s['scheduling']['caller_thread_id']>0,'actual caller TID')
    require(s.get('config')==config['id'] and s.get('arm')==arm and s.get('pair_id')==pair,'config/arm/pair binding')
    require(h.get('argv')==stage['app_argv'] and a.get('actual_argv')==h['argv'],'exact actual argv')
    require(a.get('arm')==arm and a.get('config')==config['id'] and a.get('freeze',{}).get('protocol_sha256')==manifest['protocol']['sha256'],'arm metadata identity')
    env={**protocol['runtime']['environment'],**{k:None for k in protocol['runtime']['extra_clear_environment']}}
    require(s.get('environment')==env,'explicit runtime environment')
    require(s.get('requested_nodes')==n and s.get('actual_ggml_nodes')==n and s.get('elements')==e,'planned GGML shape')
    require(s.get('tensor_payload_bytes')==4*e and s.get('allocated_payload_bytes')==(n+1)*4*e and s.get('logical_read_bytes')==n*4*e and s.get('logical_write_bytes')==n*4*e,'logical payload')
    require(type(s.get('allocated_buffer_bytes')) is int and s['allocated_buffer_bytes']>=s['allocated_payload_bytes'],'buffer allocation')
    nodes=s.get('graph_nodes',[]);require(len(nodes)==n,'dependency chain length')
    for i,node in enumerate(nodes):
        require(node=={'index':i,'name':f'scale_{i+1}','src0':('graph_input' if i==0 else f'scale_{i}'),'operator':'SCALE','dtype':'F32','scale':0.5,'ne':[e,1,1,1]},'source dependency chain')
    require(s.get('loaded_modules_before')==f.get('loaded_modules_after') and f.get('modules_stable') is True,'module lifetime drift')
    bypath={str(Path(x['path']).resolve()).casefold():x for x in [*manifest['files'],*extra_runtime_refs]}
    mods=s.get('loaded_modules_before',[]);require(bool(mods),'loaded module evidence missing')
    for item in mods:
        expected=bypath.get(str(Path(item['path']).resolve()).casefold());require(expected is not None and expected['sha256']==item['sha256'],'actual module not frozen')
    calls=[r for r in rows if r.get('record')=='graph_call'];require(len(calls)==36,'full36 call denominator')
    expected_labels=[f"graph_submit/{config['id']}/{p}/{i}" for p,i in expected_calls()]
    require([r.get('label') for r in calls]==expected_labels and [(r.get('phase'),r.get('index')) for r in calls]==expected_calls(),'ordered labels')
    keys=['qpc_outer_push_start','qpc_outer_push_end','qpc_submit_push_start','qpc_submit_push_end','qpc_submit_start','qpc_submit_end','qpc_submit_pop_start','qpc_submit_pop_end','qpc_sync_push_start','qpc_sync_push_end','qpc_sync_start','qpc_sync_end','qpc_sync_pop_start','qpc_sync_pop_end','qpc_outer_pop_start','qpc_outer_pop_end']
    previous=h.get('qpc_start');require(type(previous) is int and previous>0,'actual start QPC')
    for row in calls:
        require(row.get('arm')==arm and row.get('graph_status')==0 and row.get('observed_device_kernel_count') is None and row.get('observed_cuda_graph_launch_count') is None,'call status/direct observation scope')
        times=[row.get(k) for k in keys];require(all(type(x) is int for x in times) and times==sorted(times) and previous<=times[0]<times[-1],'QPC ordering/lifetime')
        previous=times[-1]
    require(type(f.get('qpc_end')) is int and previous<=f['qpc_end'],'footer QPC')
    blocks=[x for x in rows if x.get('record')=='numeric_block']
    if arm=='control':
        expected=[(name,ordinal,list(range(1,n)) if name=='first_or_postformal_intermediate_stages' else [n]) for ordinal in range(36) for name in (['per_call_final','first_or_postformal_intermediate_stages'] if ordinal in (0,35) else ['per_call_final'])]
    else:
        expected=[(name,ordinal,list(range(1,n+1))) for name,ordinal in [('first',0),('post_warmup',5),('post_formal',35)]]
        started=one('buffered_started');require(started.get('arm')=='buffered' and rows.index(started)<rows.index(calls[0]),'durable buffered start marker')
    require(len(blocks)==len(expected),'numeric block denominator');checked=mismatches=nonfinite=0
    for block,(name,ordinal,stages) in zip(blocks,expected):
        require(block.get('arm')==arm and block.get('name')==name and block.get('after_ordinal')==ordinal and [x.get('stage') for x in block.get('stages',[])]==stages,'numeric cadence/stage coverage')
        for sr in block['stages']:
            v=sr['result'];require(v.get('checked')==e and type(v.get('mismatches')) is int and type(v.get('nonfinite')) is int,'numeric count fields')
            require(v['mismatches']>=0 and v['nonfinite']>=0 and re.fullmatch('[0-9a-f]{16}',v.get('raw_fnv1a64','')),'numeric hash/count')
            require(v.get('bitwise_pass')==(v['mismatches']==0),'numeric pass flag inconsistency')
            checked+=v['checked'];mismatches+=v['mismatches'];nonfinite+=v['nonfinite']
    require((f.get('graph_calls'),f.get('numeric_blocks'),f.get('checked_values'),f.get('mismatches'),f.get('nonfinite'))==(36,len(blocks),checked,mismatches,nonfinite),'footer count closure')
    math_pass=mismatches==0 and nonfinite==0
    require(f.get('math_pass')==math_pass and f.get('graph_status_pass') is True and f.get('status')==('complete' if math_pass else 'quality_failed'),'footer status closure')
    side=load(str(path)+'.receipt.json');actual=ref(path)
    require(side.get('raw_sha256')==actual['sha256'] and side.get('raw_bytes')==actual['bytes'] and side.get('raw_path')==str(path) and side.get('protocol_sha256')==manifest['protocol']['sha256'] and side.get('arm')==arm and side.get('pair_id')==pair and side.get('config')==config['id'],'closed raw sidecar identity')
    formal=[x for x in calls if x['phase']=='formal'];host=distribution([(x['qpc_sync_end']-x['qpc_submit_start'])*1e9/h['qpc_frequency'] for x in formal])
    return {'raw_ref':actual,'sidecar_ref':ref(str(path)+'.receipt.json'),'header':h,'setup':s,'footer':f,'calls':calls,'numeric_blocks':blocks,'math_pass':math_pass,'per_call_final_validated':arm=='control','host_formal':host,'formal_clock_intervals':[{'index':x['index'],'qpc_start':x['qpc_outer_push_start'],'qpc_end':x['qpc_outer_pop_end']} for x in formal]}