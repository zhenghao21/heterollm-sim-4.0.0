"""Mock raw evidence in the exact r2 buffered emission order; no native calls."""
import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import strict_raw
from native_trace import expected_calls,ref
from contract import CONFIG
PROBE=Path(__file__).parent.parent/'host_settle_probe_r2'

def fixture(tmp_path,condition='short'):
    manifest=strict_raw.load(PROBE/'build_manifest.json');protocol=strict_raw.load(PROBE/'protocol.json')
    mod=tmp_path/'m.dll';mod.write_bytes(b'm');manifest['files']=[ref(mod)]
    ms=0 if condition=='short' else 1000;arm='buffered';n=CONFIG['nodes'];e=CONFIG['elements'];cid=CONFIG['id']
    raw=tmp_path/f'{condition}.jsonl';pair=f'r21_host_settle.{condition}.pair1'
    stage={'arm':arm,'condition':condition,'settle_ms':ms,'pair':1,'pair_id':pair,'config':dict(CONFIG),'raw':str(raw)}
    stage['app_argv']=[manifest['executable']['path'],'--run','--config',cid,'--arm',arm,'--settle-ms',str(ms),'--pair-id',pair,'--output',str(raw)]
    h={'record':'header','schema':'graph-gap-probe/v1','protocol_sha256':manifest['protocol']['sha256'],'pid':12,'qpc_start':1,'qpc_frequency':1000,'argv':stage['app_argv']}
    s={'record':'setup','config':cid,'arm':arm,'pair_id':pair,'scheduling':{'caller_thread_id':2},'environment':{**protocol['runtime']['environment'],**{k:None for k in protocol['runtime']['extra_clear_environment']}},'requested_nodes':n,'actual_ggml_nodes':n,'elements':e,'tensor_payload_bytes':4*e,'allocated_payload_bytes':(n+1)*4*e,'logical_read_bytes':n*4*e,'logical_write_bytes':n*4*e,'allocated_buffer_bytes':(n+1)*4*e,'graph_nodes':[{'index':i,'name':f'scale_{i+1}','src0':'graph_input' if i==0 else f'scale_{i}','operator':'SCALE','dtype':'F32','scale':.5,'ne':[e,1,1,1]} for i in range(n)],'loaded_modules_before':[ref(mod)]}
    start={'record':'buffered_started','arm':arm,'config':cid,'condition_settle_ms':ms,'first_timed_ordinal':0}
    settle={'record':'settle_metadata','arm':arm,'condition_settle_ms':ms,'qpc_begin':18,'qpc_end':18+ms,'qpc_frequency':1000,'iterations':0 if ms==0 else 100,'graph_status_pass':True,'estimator_time_used':False}
    meta={'record':'arm_metadata','arm':arm,'config':cid,'write_cadence':f'buffered_settle_{ms}ms','actual_argv':stage['app_argv'],'freeze':{'protocol_sha256':manifest['protocol']['sha256']}}
    calls=[];q=2
    keys=['qpc_outer_push_start','qpc_outer_push_end','qpc_submit_push_start','qpc_submit_push_end','qpc_submit_start','qpc_submit_end','qpc_submit_pop_start','qpc_submit_pop_end','qpc_sync_push_start','qpc_sync_push_end','qpc_sync_start','qpc_sync_end','qpc_sync_pop_start','qpc_sync_pop_end','qpc_outer_pop_start','qpc_outer_pop_end']
    for ordinal,(phase,i) in enumerate(expected_calls()):
        if ordinal==1:q=20+ms
        times=list(range(q,q+16));q+=16
        calls.append({'record':'graph_call','label':f'graph_submit/{cid}/{phase}/{i}','phase':phase,'index':i,'arm':arm,'graph_status':0,'observed_device_kernel_count':None,'observed_cuda_graph_launch_count':None,**dict(zip(keys,times))})
    blocks=[{'record':'numeric_block','arm':arm,'name':name,'after_ordinal':o,'stages':[{'stage':i,'result':{'checked':e,'mismatches':0,'nonfinite':0,'raw_fnv1a64':'0'*16,'bitwise_pass':True}} for i in range(1,n+1)]} for name,o in [('first',0),('post_warmup',5),('post_formal',35)]]
    footer={'record':'footer','status':'complete','arm':arm,'graph_calls':36,'numeric_blocks':3,'checked_values':3*n*e,'mismatches':0,'nonfinite':0,'graph_status_pass':True,'math_pass':True,'loaded_modules_after':[ref(mod)],'modules_stable':True,'qpc_end':q}
    rows=[h,s,start,settle,meta,*calls,*blocks,footer]
    side={'schema':'graph-gap-run-receipt/v1','status':'complete','raw_path':str(raw.resolve()),'protocol_sha256':manifest['protocol']['sha256'],'arm':arm,'pair_id':pair,'config':cid,'calibration_ready':False}
    Path(str(raw)+'.receipt.json').write_text(json.dumps(side));save(raw,rows)
    return raw,stage,manifest,protocol,rows

def save(raw,rows):
    raw.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    path=Path(str(raw)+'.receipt.json');side=json.loads(path.read_text());r=ref(raw);side.update(raw_sha256=r['sha256'],raw_bytes=r['bytes']);path.write_text(json.dumps(side))

def row(rows,kind):return next(x for x in rows if x['record']==kind)

@pytest.mark.parametrize('condition',['short','settled'])
def test_real_emission_order_and_formal_estimator(tmp_path,condition):
    raw,stage,m,p,rows=fixture(tmp_path,condition);result=strict_raw.audit(raw,stage,m,p,12)
    assert rows.index(row(rows,'settle_metadata'))<rows.index(row(rows,'graph_call'))
    assert result['math_pass'] and result['host_formal']['count']==30 and result['host_formal']['median_ns']==7_000_000
    assert len(result['formal_clock_intervals'])==30 and len(result['calls'])==36

@pytest.mark.parametrize('kind,msg',[
    ('missing','one settle_metadata'),('early','settle QPC placement'),('late','settle QPC placement'),('short','settle iteration/work duration'),('zero_iterations','settle iteration/work duration'),('float_iterations','settle iteration/work duration'),('wrong_qpf','settle identity'),('wrong_arm','settle identity'),('failed_work','settle identity'),('estimator','settle identity'),('forged_condition','settle identity'),('forged_stage','stage argv condition'),('missing_argv','stage argv condition'),('wrong_cadence','settle write cadence'),('call_count','full36'),('pair','config/arm/pair'),('pid','direct PID'),('qpc','QPC ordering'),('numeric_stage','numeric cadence'),('footer','footer count'),('math','footer status'),('module','module lifetime')])
def test_rejects_critical_tamper(tmp_path,kind,msg):
    raw,stage,m,p,rows=fixture(tmp_path,'settled');settle=row(rows,'settle_metadata')
    if kind=='missing':rows.remove(settle)
    elif kind=='early':settle['qpc_begin']=1
    elif kind=='late':settle['qpc_end']=5000
    elif kind=='short':settle['qpc_end']=1017
    elif kind=='zero_iterations':settle['iterations']=0
    elif kind=='float_iterations':settle['iterations']=1.0
    elif kind=='wrong_qpf':settle['qpc_frequency']=999
    elif kind=='wrong_arm':settle['arm']='control'
    elif kind=='failed_work':settle['graph_status_pass']=False
    elif kind=='estimator':settle['estimator_time_used']=True
    elif kind=='forged_condition':settle['condition_settle_ms']=0
    elif kind=='forged_stage':
        stage.update(condition='short',settle_ms=0,pair_id='r21_host_settle.short.pair1')
    elif kind=='missing_argv':stage['app_argv']=stage['app_argv'][:6]+stage['app_argv'][8:]
    elif kind=='wrong_cadence':row(rows,'arm_metadata')['write_cadence']='buffered_settle_0ms'
    elif kind=='call_count':rows.remove(row(rows,'graph_call'))
    elif kind=='pair':rows[1]['pair_id']='x'
    elif kind=='pid':rows[0]['pid']=99
    elif kind=='qpc':row(rows,'graph_call')['qpc_submit_end']=0
    elif kind=='numeric_stage':row(rows,'numeric_block')['stages'][0]['stage']=99
    elif kind=='footer':rows[-1]['graph_calls']=35
    elif kind=='math':rows[-1]['math_pass']=False
    elif kind=='module':rows[1]['loaded_modules_before'][0]['sha256']='f'*64
    save(raw,rows)
    with pytest.raises(ValueError,match=msg):strict_raw.audit(raw,stage,m,p,12)

def test_short_forbids_settle_iterations(tmp_path):
    raw,s,m,p,rows=fixture(tmp_path);row(rows,'settle_metadata')['iterations']=1;save(raw,rows)
    with pytest.raises(ValueError,match='settle iteration'):strict_raw.audit(raw,s,m,p)

def test_actual_module_requires_extra_runtime_ref(tmp_path):
    raw,s,m,p,rows=fixture(tmp_path);extra=tmp_path/'cuda.dll';extra.write_bytes(b'cuda');r=ref(extra)
    rows[1]['loaded_modules_before'].append(r);rows[-1]['loaded_modules_after'].append(r);save(raw,rows)
    with pytest.raises(ValueError,match='actual module not frozen'):strict_raw.audit(raw,s,m,p)
    assert strict_raw.audit(raw,s,m,p,extra_runtime_refs=[r])['math_pass']
