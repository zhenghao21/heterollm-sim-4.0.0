"""R22 mock-only controller/raw/trace regression tests. No native/GPU launch."""
import importlib.util,json,sys,copy
from pathlib import Path
from types import SimpleNamespace
import pytest
P=Path(__file__).parent;sys.path.insert(0,str(P))
import collect as c
import strict_raw as raw
import native_trace as nt
from contract import CONFIGS
PROBE=P.parent/'long_graph_probe'
def manifest():return raw.load(PROBE/'build_manifest.json')
def protocol():return raw.load(PROBE/'protocol.json')

def raw_fixture(tmp_path,nodes):
    m=manifest();p=protocol();cfg=next(x for x in CONFIGS if x['nodes']==nodes);cid=cfg['id'];e=cfg['elements'];mod=tmp_path/'dummy.dll';mod.write_bytes(b'dummy');m['files']=[nt.ref(mod)]
    path=tmp_path/'raw.jsonl';pair=f'r22_long_graph.{cid}.pair1';stage={'condition':cid,'config':cfg,'arm':'buffered','pair':1,'pair_id':pair,'raw':str(path)}
    stage['app_argv']=[m['executable']['path'],'--run','--config',cid,'--arm','buffered','--pair-id',pair,'--output',str(path)]
    header={'record':'header','schema':'graph-gap-probe/v1','protocol_sha256':m['protocol']['sha256'],'pid':12,'qpc_frequency':1000,'qpc_start':1,'argv':stage['app_argv']}
    setup={'record':'setup','config':cid,'arm':'buffered','pair_id':pair,'scheduling':{'caller_thread_id':2},'environment':{**p['runtime']['environment'],**{k:None for k in p['runtime']['extra_clear_environment']}},'requested_nodes':nodes,'actual_ggml_nodes':nodes,'elements':e,'tensor_payload_bytes':4*e,'allocated_payload_bytes':(nodes+1)*4*e,'logical_read_bytes':nodes*4*e,'logical_write_bytes':nodes*4*e,'allocated_buffer_bytes':(nodes+1)*4*e,'loaded_modules_before':[nt.ref(mod)],'graph_nodes':[{'index':i,'name':f'scale_{i+1}','src0':'graph_input' if i==0 else f'scale_{i}','operator':'SCALE','dtype':'F32','scale':.5 if (i+1)%2 else 2.0,'ne':[e,1,1,1]} for i in range(nodes)],'submission_entry':'ggml_backend_sched_graph_compute_async','synchronization_entry':'ggml_backend_sched_synchronize','scheduler_backends':1,'scheduler_parallel':False,'scheduler_op_offload':False}
    meta={'record':'arm_metadata','arm':'buffered','config':cid,'actual_argv':stage['app_argv'],'freeze':{'protocol_sha256':m['protocol']['sha256']},'write_cadence':'buffered_after_postformal'}
    keys=['qpc_outer_push_start','qpc_outer_push_end','qpc_submit_push_start','qpc_submit_push_end','qpc_submit_start','qpc_submit_end','qpc_submit_pop_start','qpc_submit_pop_end','qpc_sync_push_start','qpc_sync_push_end','qpc_sync_start','qpc_sync_end','qpc_sync_pop_start','qpc_sync_pop_end','qpc_outer_pop_start','qpc_outer_pop_end']
    calls=[{'record':'graph_call','arm':'buffered','phase':phase,'index':i,'label':f'graph_submit/{cid}/{phase}/{i}','graph_status':0,'observed_device_kernel_count':None,'observed_cuda_graph_launch_count':None,**dict(zip(keys,range(2+ordinal*20,18+ordinal*20)))} for ordinal,(phase,i) in enumerate(nt.expected_calls())]
    blocks=[{'record':'numeric_block','arm':'buffered','name':name,'after_ordinal':o,'stages':[{'stage':i,'result':{'checked':e,'mismatches':0,'nonfinite':0,'raw_fnv1a64':'0'*16,'bitwise_pass':True}} for i in range(1,nodes+1)]} for name,o in [('first',0),('post_warmup',5),('post_formal',35)]]
    footer={'record':'footer','arm':'buffered','status':'complete','graph_calls':36,'numeric_blocks':3,'checked_values':3*nodes*e,'mismatches':0,'nonfinite':0,'math_pass':True,'graph_status_pass':True,'loaded_modules_after':[nt.ref(mod)],'modules_stable':True,'qpc_end':1000}
    rows=[header,setup,{'record':'buffered_started','arm':'buffered','config':cid,'first_timed_ordinal':0},meta,*calls,*blocks,footer]
    side={'schema':'graph-gap-run-receipt/v1','status':'complete','raw_path':str(path.resolve()),'protocol_sha256':m['protocol']['sha256'],'arm':'buffered','pair_id':pair,'config':cid,'calibration_ready':False}
    Path(str(path)+'.receipt.json').write_text(json.dumps(side));save(path,rows);return path,stage,m,p,rows

def save(path,rows):
    path.write_text('\n'.join(json.dumps(x) for x in rows)+'\n');q=Path(str(path)+'.receipt.json');d=json.loads(q.read_text());r=nt.ref(path);d.update(raw_sha256=r['sha256'],raw_bytes=r['bytes']);q.write_text(json.dumps(d))

@pytest.mark.parametrize('nodes',[64,256])
def test_buffered_raw_full_node_math_coverage(tmp_path,nodes):
    path,stage,m,p,rows=raw_fixture(tmp_path,nodes);a=raw.audit(path,stage,m,p,12)
    assert len(a['calls'])==36 and a['host_formal']['count']==30 and a['math_pass'] and a['footer']['checked_values']==3*nodes*262144

@pytest.mark.parametrize('mutation',['scale','missing_node','scheduler','settle','per_call','count','argv','numeric_missing','numeric_mismatch','module','qpc'])
def test_raw_failures_retained_not_accepted(tmp_path,mutation):
    path,stage,m,p,rows=raw_fixture(tmp_path,64)
    if mutation=='scale':rows[1]['graph_nodes'][1]['scale']=.5
    elif mutation=='missing_node':rows[1]['graph_nodes'].pop()
    elif mutation=='scheduler':rows[1]['submission_entry']='ggml_backend_graph_compute_async'
    elif mutation=='settle':rows.append({'record':'settle_metadata'})
    elif mutation=='per_call':rows[3]['write_cadence']='per_call_validate_and_write'
    elif mutation=='count':rows.pop(4)
    elif mutation=='argv':stage['app_argv']+=['--settle-ms','1000']
    elif mutation=='numeric_missing':next(x for x in rows if x['record']=='numeric_block')['stages'].pop()
    elif mutation=='numeric_mismatch':next(x for x in rows if x['record']=='numeric_block')['stages'][0]['result']['mismatches']=1
    elif mutation=='module':rows[-1]['modules_stable']=False
    elif mutation=='qpc':rows[4]['qpc_submit_end']=0
    save(path,rows)
    with pytest.raises(ValueError):raw.audit(path,stage,m,p,12)

def test_fixed_plan_has12_native6export_no_settle():
    plan=c.plan(protocol(),manifest());assert len(plan)==18 and sum(x['mode']!='export' for x in plan)==12
    assert {x['config']['nodes'] for x in plan}=={64,256}
    assert all('--settle-ms' not in x['app_argv'] and x['arm']=='buffered' for x in plan)
    assert {x['config']['nodes'] for x in plan[:6]}=={64,256}
    assert all('--kill=false' in x['argv'] for x in plan if x['mode']=='profile')

@pytest.mark.parametrize('mutation',['configs','exe','settle','formal'])
def test_domain_tamper_rejected(mutation):
    m=manifest();p=protocol()
    if mutation=='configs':m['configs']=[]
    elif mutation=='exe':m['executable']['sha256']='a'*64
    elif mutation=='settle':p['execution']['settle_loop_present']=True
    else:p['execution']['formal']=29
    with pytest.raises(ValueError):c.plan(p,m)

def mock_run(tmp_path,monkeypatch,fail=False,stop_after=None,reset_fail=False):
    monkeypatch.setattr(c,'HERE',tmp_path);monkeypatch.setattr(c.time,'sleep',lambda _:None)
    f={'manifest':nt.ref(PROBE/'build_manifest.json'),'protocol':nt.ref(PROBE/'protocol.json'),'smi':{'path':'fake-smi'},'plan':c.plan(protocol(),manifest())};monkeypatch.setattr(c,'verified_freeze',lambda _:f)
    seen=[];commands=[]
    class Proc:
        def __init__(self,args,stdout,stderr,**kw):
            commands.append(args);self.args=args
            if args[1].startswith('--query-gpu'):
                gpu=protocol()['runtime']['gpu_expected'];stdout.write(f"{gpu['uuid']},{gpu['driver']},2400".encode())
        def wait(self):return 1 if reset_fail and '-rgc' in self.args else 0
    monkeypatch.setattr(c.subprocess,'Popen',Proc)
    def stage(x,*args):
        seen.append(x['ordinal'])
        if stop_after==len(seen):c.write_new(tmp_path/'stop.json',{'freeze_sha256':'a'*64,'stop':True})
        if fail and x['ordinal']==1:raise ValueError('mock stage failure')
    monkeypatch.setattr(c,'stage_run',stage);monkeypatch.setattr(c,'summarize_pairs',lambda *a:{'mock':True})
    return seen,commands

def test_nonblocking_checkpoint_continues_all18_without_consent(tmp_path,monkeypatch):
    seen,commands=mock_run(tmp_path,monkeypatch);assert c.run('a'*64)==0 and seen==list(range(18))
    checkpoint=raw.load(tmp_path/'first_pair_checkpoint.json');assert checkpoint['nonblocking'] and checkpoint['review_required'] is False
    assert not (tmp_path/'continue.json').exists() and commands[-1]==['fake-smi','-rgc']

def test_explicit_stop_only_after_natural_stage_exit(tmp_path,monkeypatch):
    seen,commands=mock_run(tmp_path,monkeypatch,stop_after=2);assert c.run('a'*64)==1 and seen==[0,1]
    result=raw.load(tmp_path/'controller_result.json');assert result['status']=='stopped_by_explicit_request' and len(result['stage_status'])==18 and commands[-1]==['fake-smi','-rgc']

def test_stage_failure_does_not_cancel_authorized_remaining_stages(tmp_path,monkeypatch):
    seen,commands=mock_run(tmp_path,monkeypatch,fail=True);assert c.run('a'*64)==1 and len(seen)==18
    result=raw.load(tmp_path/'controller_result.json');assert result['attempted_stages']==18 and result['completed_stages']==17 and result['status']=='completed_with_failures' and result['stage_errors'][0]['ordinal']==1

def test_reset_failure_persists_complete_denominator(tmp_path,monkeypatch):
    seen,commands=mock_run(tmp_path,monkeypatch,reset_fail=True);assert c.run('a'*64)==1
    result=raw.load(tmp_path/'controller_result.json');assert result['status']=='failed_cleanup' and len(result['stage_status'])==18

def test_wait_interruption_does_not_kill_owned_child(monkeypatch):
    monkeypatch.setattr(c.time,'sleep',lambda _:None)
    class Proc:
        n=0
        def wait(self):
            self.n+=1
            if self.n==1:raise KeyboardInterrupt()
            return 0
    assert c.wait_same(Proc())==0

def trace_fixture(actual_nodes=2):
    config=CONFIGS[0];tid=(1<<48)|(12<<24)|2;pid=tid&nt.PROCESS_MASK;markers=[];apis=[];kernels=[];calls=[]
    for ordinal,(phase,i) in enumerate(nt.expected_calls()):
        begin=ordinal*1000;label=f"graph_submit/{config['id']}/{phase}/{i}";calls.append({'label':label,'phase':phase,'index':i});markers.append({'start':begin,'end':begin+900,'globalTid':tid,'resolved_text':label})
        apis+=[{'start':begin+1,'end':begin+10,'globalTid':tid,'name_text':'cudaGraphLaunch','correlationId':ordinal,'returnValue':0},{'start':begin+800,'end':begin+890,'globalTid':tid,'name_text':'cudaStreamSynchronize','correlationId':1000+ordinal,'returnValue':0}]
        for n in range(actual_nodes):kernels.append({'evidence_rowid':ordinal*actual_nodes+n,'start':begin+20+n*20,'end':begin+30+n*20,'globalPid':pid,'correlationId':ordinal,'deviceId':0,'streamId':3,'short_name_text':'scale_f32','gridX':1024,'gridY':1,'gridZ':1,'blockX':256,'blockY':1,'blockZ':1,'dynamicSharedMemory':0})
    return {'tables':{'NVTX_EVENTS':markers,'CUPTI_ACTIVITY_KIND_RUNTIME':apis,'CUPTI_ACTIVITY_KIND_KERNEL':kernels}},{'header':{'pid':12},'setup':{'config':config['id'],'scheduling':{'caller_thread_id':2}},'footer':{'graph_calls':36},'calls':calls}

def test_trace_observes_count_not_one_kernel_per_node(monkeypatch):
    trace,doc=trace_fixture(2);monkeypatch.setattr(nt,'read_trace',lambda _:trace);result=nt.analyze_trace('mock',doc,CONFIGS[0])
    assert result['actual_kernels_mapped']==72 and result['formal_denominator']==30
    call=result['calls'][0];assert call['actual_kernel_count']==2 and call['planned_GGML_nodes']==64 and call['chain']['fusion_proven'] is False
    assert call['launch_api_events_verbatim'][0]['start']==1 and call['graph_end_sync_api_events_verbatim'][0]['start']==800
    pair=call['kernel_launch_pairs'][0];assert pair['api']['globalTid'] and pair['kernel']['streamId']==3 and pair['api']['correlationId']==pair['kernel']['correlationId']
    assert sum(call['exclusive_partition_ns'].values())==900

@pytest.mark.parametrize('mutation',['thread','sync','correlation','geometry'])
def test_trace_rejects_unbound_evidence(monkeypatch,mutation):
    trace,doc=trace_fixture()
    if mutation=='thread':doc['header']['pid']=99
    elif mutation=='sync':trace['tables']['CUPTI_ACTIVITY_KIND_RUNTIME']=[x for x in trace['tables']['CUPTI_ACTIVITY_KIND_RUNTIME'] if 'Synchronize' not in x['name_text']]
    elif mutation=='correlation':trace['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]['correlationId']=88888
    else:trace['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]['blockX']=128
    monkeypatch.setattr(nt,'read_trace',lambda _:trace)
    if mutation=='thread':
        with pytest.raises(ValueError):nt.analyze_trace('mock',doc,CONFIGS[0])
    else:assert nt.analyze_trace('mock',doc,CONFIGS[0])['all36_chain_complete'] is False

def test_frozen_probe_has_no_settle_and_single_scheduler_end_sync():
    text=(PROBE/'long_graph_probe.cpp').read_text();main=(PROBE/'long_graph_main.cpp').read_text()
    assert 'settle' not in text and 'ggml_backend_sched_graph_compute_async(harness.scheduler, harness.graph)' in text
    assert text.count('ggml_backend_sched_synchronize(harness.scheduler)')==1 and 'constexpr int capacity = 1024' in main
    controller=(P/'collect.py').read_text();assert 'continue.json' not in controller and '.kill(' not in controller and '.terminate(' not in controller

def test_exact_closed_form_never_underflows_at256():
    spec=importlib.util.spec_from_file_location('r22_reference',PROBE/'reference_check.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);assert mod.check()['pass']
