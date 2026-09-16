"""Portable fixed-matrix tests; no GPU or native execution."""
import copy
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import common
import extract
import quality
import runner


def config(mmvq=True,group='training'):
    return {'id':'unit','group':group,'quant':'Q5_0','M':4 if mmvq else 64,'N':896,'K':896,'expected_source_path':'MMVQ_Q8_1_HALF' if mmvq else 'MMQ_Q8_1_D4_F32'}


def kernel(name,ident=1,start=10,end=20,pid=0x1000000,quant=6):
    demangled=(f'void {name}<(ggml_type){quant}, (int)64, (bool)0>()' if name.startswith('mul_mat') else f'void {name}<(mmq_q8_1_ds_layout)0, (bool)0>()')
    return {'evidence_rowid':ident,'start':start,'end':end,'globalPid':pid,'correlationId':ident,'deviceId':0,'streamId':14,
        'short_name_text':name,'demangled_name_text':demangled,'gridX':1,'gridY':1,'gridZ':1,'blockX':128,'blockY':1,'blockZ':1,'staticSharedMemory':0,'dynamicSharedMemory':0}


def pair(index=0):
    dist={'count':30,'median_ns':100.,'p90_div_p10':1.2}
    return {'pair':index,'numerics_all_rows':True,'trace_chain_complete':True,'source_path_matches_observed_family':True,'trace_warning_free':True,'clock_domain_validated':True,
        'profile_kernel':dict(dist),'direct_host':dict(dist),'profile_host':dict(dist)}


def test_fixed_plan_denominators_and_alternating_order():
    configs=[{'id':f'c{i}'} for i in range(26)];plan=runner.plan(configs)
    assert len(plan)==234
    assert [s['mode'] for s in plan[:9]]==['profile','direct','export','direct','profile','export','profile','direct','export']
    assert sum(s['mode']=='profile' for s in plan)==78
    assert sum(s['mode']=='direct' for s in plan)==78
    assert sum(s['mode']=='export' for s in plan)==78


def test_direct_profile_application_args_are_identical_except_path():
    c=config();a=runner.app_argv(c,Path('profile/raw.json'));b=runner.app_argv(c,Path('direct/raw.json'))
    assert a[:-1]==b[:-1]
    assert '--control' not in a and a[a.index('--repeats')+1]=='30' and a[a.index('--warmup')+1]=='5'


def test_checkpoint_does_not_kill_or_continue():
    class Process:
        pid=123
        def wait(self,timeout):raise subprocess.TimeoutExpired('fake',timeout)
        def kill(self):raise AssertionError('must never kill')
        def terminate(self):raise AssertionError('must never terminate')
    r=runner.wait_checkpoint(Process(),180)
    assert r['status']=='still_running' and r['pid']==123 and r['kill_on_deadline'] is False


def test_execution_requires_both_root_gates():
    for review,idle in [(False,False),(True,False),(False,True)]:
        with pytest.raises(ValueError,match='root review'):runner.run(review,idle)


@pytest.mark.parametrize('names,mmvq', [(['quantize_q8_1','mul_mat_vec_q'],True),(['quantize_mmq_q8_1','mul_mat_q'],False),(['quantize_mmq_q8_1','mul_mat_q','mul_mat_q_stream_k_fixup'],False)])
def test_observed_supported_chains(names,mmvq):
    kernels=[kernel(name,i+1,10+i*20,20+i*20) for i,name in enumerate(names)]
    assert extract.validate_chain(kernels,config(mmvq))['complete'] is True


@pytest.mark.parametrize('change',['unknown','wrong_quant','wrong_layout','missing_conversion','extra_fixup','bad_grid'])
def test_unsupported_chains_retained_not_guessed(change):
    ks=[kernel('quantize_mmq_q8_1',1),kernel('mul_mat_q',2,30,40)]
    if change=='unknown':ks[1]['short_name_text']='unknown_kernel'
    if change=='wrong_quant':ks[1]['demangled_name_text']=ks[1]['demangled_name_text'].replace('(ggml_type)6','(ggml_type)8')
    if change=='wrong_layout':ks[0]['demangled_name_text']=ks[0]['demangled_name_text'].replace('(mmq_q8_1_ds_layout)0','(mmq_q8_1_ds_layout)1')
    if change=='missing_conversion':ks.pop(0)
    if change=='extra_fixup':ks.extend([kernel('mul_mat_q_stream_k_fixup',3,50,60),kernel('mul_mat_q_stream_k_fixup',4,70,80)])
    if change=='bad_grid':ks[1]['gridX']=0
    r=extract.validate_chain(ks,config(False));assert not r['complete'] and r['issues']


def trace_fixture():
    calls=common.expected_calls();markers=[];apis=[];kernels=[];app={'runs':[]};tid=0x1000000|17
    for i,(phase,index) in enumerate(calls):
        start=i*1000;label=f'label_{phase}_{index}';app['runs'].append({'phase':phase,'index':index,'nvtx_label':label})
        markers.append({'evidence_rowid':i+1,'start':start,'end':start+900,'globalTid':tid,'resolved_text':label})
        for j,name in enumerate(('quantize_q8_1','mul_mat_vec_q')):
            n=i*2+j+1;apis.append({'evidence_rowid':n,'start':start+20+j*10,'end':start+25+j*10,'globalTid':tid,'correlationId':n,'name_text':'cudaLaunchKernel_v7000'})
            kernels.append(kernel(name,n,start+100+j*30,start+120+j*30))
    return {'tables':{'NVTX_EVENTS':markers,'CUPTI_ACTIVITY_KIND_RUNTIME':apis,'CUPTI_ACTIVITY_KIND_KERNEL':kernels}},app


def test_exact36_correlation_counts_and_overlap_safe_union():
    raw,app=trace_fixture();r=extract.correlate(raw,app,config())
    assert r['all36_chain_complete'] and r['mapped_kernels']==72 and r['formal_kernel_union']['count']==30
    assert r['formal_kernel_union']['median_ns']==40
    assert len(r['calls'])==36


@pytest.mark.parametrize('change',['other_pid','outside_nvtx','missing_launch','missing_marker','duplicate_marker'])
def test_correlation_identity_and_full_call_boundaries_fail_closed(change):
    raw,app=trace_fixture()
    if change=='other_pid':raw['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]['globalPid']=0x2000000
    if change=='outside_nvtx':raw['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]['end']=950
    if change=='missing_launch':raw['tables']['CUPTI_ACTIVITY_KIND_RUNTIME'].pop(0)
    if change=='missing_marker':raw['tables']['NVTX_EVENTS'].pop(0)
    if change=='duplicate_marker':raw['tables']['NVTX_EVENTS'][-1]=raw['tables']['NVTX_EVENTS'][0]
    if change.endswith('marker'):
        with pytest.raises(ValueError,match='36 semantic'):extract.correlate(raw,app,config())
    else:assert extract.correlate(raw,app,config())['all36_chain_complete'] is False


def test_actual_known_nsight_schema_read_only():
    path=common.PILOT/'profile/gemm.sqlite';before=common.ref(path)
    r=extract.read_trace(path)
    assert len(r['tables']['CUPTI_ACTIVITY_KIND_KERNEL'])==96
    assert len(r['tables']['NVTX_EVENTS'])==24
    assert 'eventClass' in r['actual_schema']['CUPTI_ACTIVITY_KIND_RUNTIME']
    assert common.ref(path)==before


def test_unknown_sqlite_schema_rejected(tmp_path):
    path=tmp_path/'bad.sqlite'
    with sqlite3.connect(path) as c:c.execute('CREATE TABLE other(x)')
    with pytest.raises(ValueError,match='actual SQLite schema'):extract.read_trace(path)


def test_union_does_not_add_overlapping_gpu_intervals():
    assert common.union_ns([(0,20),(10,30),(30,40)])==40
    assert common.union_ns([])==0


def test_quality_holds_validation_out_and_emits_no_fit():
    r=quality.quality(config(group='validation'),[pair(i) for i in range(3)])
    assert r['measurement_cost_eligible'] is True
    assert r['calibration_eligible'] is False and r['fit_performed'] is False and r['holdout_not_for_fit'] is True


@pytest.mark.parametrize('mutation',['missing_pair','numeric','chain','path','warning','kernel_dispersion','direct_dispersion','profile_direct','cross_process','missing_samples'])
def test_quality_all_pairs_fixed_thresholds(mutation):
    pairs=[pair(i) for i in range(3)]
    if mutation=='missing_pair':pairs.pop()
    elif mutation=='numeric':pairs[1]['numerics_all_rows']=False
    elif mutation=='chain':pairs[1]['trace_chain_complete']=False
    elif mutation=='path':pairs[1]['source_path_matches_observed_family']=False
    elif mutation=='warning':pairs[1]['trace_warning_free']=False
    elif mutation=='kernel_dispersion':pairs[1]['profile_kernel']['p90_div_p10']=1.500001
    elif mutation=='direct_dispersion':pairs[1]['direct_host']['p90_div_p10']=1.500001
    elif mutation=='profile_direct':pairs[1]['profile_host']['median_ns']=120.0001
    elif mutation=='cross_process':pairs[1]['profile_kernel']['median_ns']=105.0001
    elif mutation=='missing_samples':pairs[1]['profile_kernel']['count']=29
    r=quality.quality(config(),pairs);assert not r['measurement_cost_eligible'] and r['issues']


def test_quality_exact_thresholds_are_inclusive():
    pairs=[pair(i) for i in range(3)]
    for p in pairs:p['profile_kernel']['p90_div_p10']=1.5;p['direct_host']['p90_div_p10']=1.5;p['profile_host']['median_ns']=120
    pairs[0]['profile_kernel']['median_ns']=95;pairs[2]['profile_kernel']['median_ns']=105
    assert quality.quality(config(),pairs)['measurement_cost_eligible'] is True


def test_probe_r2_path_and_no_import_side_effects(tmp_path):
    assert common.PROBE.name=='r3'
    source=tmp_path/'helper.py';source.write_text('value=42')
    m=common.module(source,'unit_helper')
    assert m.value==42 and not (tmp_path/'__pycache__').exists()


def test_unexpected_gpu_memory_activity_retained_as_unsupported():
    raw,app=trace_fixture()
    raw['tables']['CUPTI_ACTIVITY_KIND_MEMSET']=[{'globalPid':0x1000000,'correlationId':1,'start':90,'end':95,'evidence_rowid':1}]
    result=extract.correlate(raw,app,config())
    assert not result['all36_chain_complete']
    assert result['calls'][0]['additional_gpu_activity_verbatim']


def test_unknown_diagnostic_severity_not_silently_accepted():
    raw,app=trace_fixture();raw['tables']['DIAGNOSTIC_EVENT']=[{'severity':123,'text':'unknown'}]
    assert extract.correlate(raw,app,config())['warning_diagnostics_verbatim']


import json
import worker


def overwrite_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return common.ref(path)


@pytest.fixture
def frozen_fixture(tmp_path):
    local=tmp_path/'collection';probe=tmp_path/'probe';tools=tmp_path/'tools'
    local.mkdir();probe.mkdir();tools.mkdir()
    for name in ('common.py','runner.py','worker.py','extract.py','quality.py','test_collection.py','README.md'):(local/name).write_text('# synthetic fixture')
    for name in ('protocol.json','stream_event_probe.cpp','source_reference.h','identity_lock.h','identity_lock.json','invoke.ps1','full_raw_audit.py','operator-surface-probe.exe'):(probe/name).write_text('fixture')
    executable=common.ref(probe/'operator-surface-probe.exe');probe_files=[common.ref(p) for p in probe.iterdir()]
    build_ref=overwrite_json(probe/'build_manifest.json',{'schema':'operator-surface-probe-build/v1','files':probe_files,'executable':executable})
    for name in ('nsys.exe','cupti64.dll','other.bin'):(tools/name).write_text(name)
    tool_files=[common.ref(p) for p in tools.iterdir()]
    inventory_ref=overwrite_json(tmp_path/'tool_inventory.json',{'root':str(tools),'files':tool_files})
    signatures_ref=overwrite_json(tmp_path/'tool_signatures.json',{'critical_nvidia_binary_signatures_valid':True,'critical_names':['nsys.exe']})
    protocol_ref=overwrite_json(local/'protocol.json',{'schema':'operator-matrix-collection-protocol/v2','quality_policy':common.POLICY,'probe_root':str(probe),
        'probe_manifest_ref':build_ref,'executable':executable,'nsys':{'executable':common.ref(tools/'nsys.exe')}})
    python_ref=common.ref(sys.executable)
    files=[common.ref(p) for p in local.iterdir()]+[build_ref,inventory_ref,signatures_ref,python_ref]
    freeze={'schema':'operator-matrix-collection-freeze/v2','files':files,'probe_files':probe_files,'tool_files':tool_files,
        'critical_tool_files':[r for r in tool_files if Path(r['path']).name in ('nsys.exe','cupti64.dll')],
        'protocol_ref':protocol_ref,'probe_manifest_ref':build_ref,'tool_inventory_ref':inventory_ref,'tool_signatures_ref':signatures_ref,'python':python_ref}
    path=local/'freeze.json';fref=overwrite_json(path,freeze);overwrite_json(local/'ready.json',{'freeze_ref':fref})
    return path,freeze,fref['sha256']


def rewrite_approval_fixture(path,freeze):
    fref=overwrite_json(path,freeze);overwrite_json(path.with_name('ready.json'),{'freeze_ref':fref});return fref['sha256']


def test_required_freeze_sets_and_external_anchor_pass(frozen_fixture):
    path,freeze,expected=frozen_fixture
    result=common.verify_freeze(path,expected)
    assert result['passed'] and result['required_sets_complete'] and result['external_approved_sha256']==expected
    assert common.verify_freeze(path,expected,False)['passed']


@pytest.mark.parametrize('mutation',['empty_files','empty_probe','empty_tools','empty_critical','missing_probe','missing_tools','missing_critical',
    'omit_collector','omit_probe_exe','omit_tool','omit_cupti','wrong_protocol_sha','missing_python','wrong_python_sha','swap_freeze','wrong_ready'])
def test_freeze_missing_identity_rejected_even_with_fresh_manifest_hash(frozen_fixture,mutation):
    path,freeze,expected=frozen_fixture
    key={'empty_files':'files','empty_probe':'probe_files','empty_tools':'tool_files','empty_critical':'critical_tool_files',
         'missing_probe':'probe_files','missing_tools':'tool_files','missing_critical':'critical_tool_files'}.get(mutation)
    if mutation.startswith('empty_'):freeze[key]=[]
    elif mutation.startswith('missing_') and key:freeze.pop(key)
    elif mutation=='omit_collector':freeze['files']=[r for r in freeze['files'] if Path(r['path']).name!='worker.py']
    elif mutation=='omit_probe_exe':freeze['probe_files']=[r for r in freeze['probe_files'] if Path(r['path']).name!='operator-surface-probe.exe']
    elif mutation=='omit_tool':freeze['tool_files'].pop()
    elif mutation=='omit_cupti':freeze['critical_tool_files']=[r for r in freeze['critical_tool_files'] if 'cupti' not in r['path']]
    elif mutation=='wrong_protocol_sha':freeze['protocol_ref']={**freeze['protocol_ref'],'sha256':'f'*64}
    elif mutation=='missing_python':freeze.pop('python')
    elif mutation=='wrong_python_sha':freeze['python']={**freeze['python'],'sha256':'f'*64}
    elif mutation=='swap_freeze':freeze['added']='changed';overwrite_json(path,freeze)
    elif mutation=='wrong_ready':overwrite_json(path.with_name('ready.json'),{'freeze_ref':{'sha256':'f'*64}})
    if mutation not in ('swap_freeze','wrong_ready'):expected=rewrite_approval_fixture(path,freeze)
    with pytest.raises(common.IdentityError):common.verify_freeze(path,expected)


@pytest.mark.parametrize('expected',[None,'','WRONG','F'*64])
def test_external_approval_digest_required(frozen_fixture,expected):
    path,_,_=frozen_fixture
    with pytest.raises(common.IdentityError):common.verify_freeze(path,expected)


def test_added_tool_file_invalidates_membership(frozen_fixture):
    path,freeze,expected=frozen_fixture
    tools=Path(freeze['tool_files'][0]['path']).parent;(tools/'injected.dll').write_text('new')
    with pytest.raises(common.IdentityError,match='membership'):common.verify_freeze(path,expected)


def test_identity_failed_retained_receipt_blocks_resume_before_any_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'HERE',tmp_path)
    monkeypatch.setattr(runner,'checked_identity',lambda *args,**kwargs:{'passed':True})
    monkeypatch.setattr(runner,'verify_clock_receipt',lambda *args,**kwargs:{'receipt_ref':{'path':'clock','sha256':'a'*64,'bytes':1}})
    overwrite_json(tmp_path/'protocol.json',{'gpu_identity':{'uuid':'unit'},'execution_plan':[]})
    overwrite_json(tmp_path/'runs/config/pair_01/profile/complete.json',{'status':'failed_identity'})
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('must not launch'))
    result=runner.run(True,True,expected_freeze_sha256='a'*64,clock_receipt_path=Path('clock'),clock_receipt_sha256='a'*64)
    assert result['status']=='identity_terminal_stop' and (tmp_path/'identity-stop.json').exists()
    assert runner.run(True,True,expected_freeze_sha256='a'*64)['status']=='identity_terminal_stop'


def test_post_popen_record_failure_waits_for_child_real_exit(tmp_path,monkeypatch):
    order=[]
    class Process:
        pid=123
        def wait(self):order.append('wait');return 0
    class Support:
        def qpc(self):return 42
    def fail_launched(path,value):order.append('record_fail');raise OSError('disk full')
    monkeypatch.setattr(worker,'write_new',fail_launched)
    receipt={};spec={'argv':['unit'],'cwd':str(tmp_path),'_environment':{}}
    worker.supervise_child(spec,tmp_path,receipt,Support(),popen=lambda *args,**kwargs:Process())
    assert order==['record_fail','wait']
    assert receipt['child_process_exited'] is True and receipt['returncode']==0 and receipt['status']=='failed'
    assert not (tmp_path/'complete.json').exists()


def test_post_popen_qpc_failure_waits_and_wait_errors_do_not_release_ownership(tmp_path,monkeypatch):
    class Process:
        pid=123;calls=0
        def wait(self):
            self.calls+=1
            if self.calls==1:raise OSError('transient wait observer error')
            return 7
    class Support:
        def qpc(self):raise OSError('QPC failure')
    proc=Process();receipt={};monkeypatch.setattr(worker.time,'sleep',lambda n:None)
    worker.supervise_child({'argv':['unit'],'cwd':str(tmp_path),'_environment':{}},tmp_path,receipt,Support(),popen=lambda *a,**kw:proc)
    assert proc.calls==2 and receipt['child_process_exited'] and receipt['returncode']==7
    assert receipt['wait_observation_errors']


@pytest.mark.parametrize('table,different_process',[('CUPTI_ACTIVITY_KIND_KERNEL',True),('CUPTI_ACTIVITY_KIND_KERNEL',False),('CUPTI_ACTIVITY_KIND_MEMCPY',True),('CUPTI_ACTIVITY_KIND_MEMSET',False)])
def test_all_captured_same_device_external_gpu_overlap_rejected(table,different_process):
    raw,app=trace_fixture();event=kernel('unrelated',999,100,170,pid=0x2000000 if different_process else 0x1000000)
    raw['tables'].setdefault(table,[]).append(event)
    result=extract.correlate(raw,app,config())
    assert not result['all36_chain_complete']
    overlap=result['calls'][0]['captured_external_gpu_overlap_verbatim']
    assert overlap and overlap[0]['different_process'] is different_process


def test_eviction_outside_measured_intervals_is_retained_not_rejected():
    raw,app=trace_fixture();raw['tables']['CUPTI_ACTIVITY_KIND_KERNEL'].append(kernel('scale_f32',999,-100,-10))
    result=extract.correlate(raw,app,config());assert result['all36_chain_complete'] and 999 in result['unmapped_kernel_rowids']


def test_known_wrong_family_keeps_observation_separate_from_expectation():
    ks=[kernel('quantize_mmq_q8_1'),kernel('mul_mat_q',2,30,40)]
    r=extract.validate_chain(ks,config(True))
    assert r['observed_family']=='MMQ' and not r['source_path_matches_observed_family'] and not r['complete']


def clock_fixture():
    app={'qpc_frequency':1000000,'runs':[{'phase':'formal','index':i,'qpc_start':10000+i*10000,'qpc_end':10500+i*10000} for i in range(30)]}
    samples=[{'qpc_ticks':x,'sm_mhz':{'status':0,'value':2400}} for x in range(5000,320000,5000)]
    return app,samples


def test_formal_clock_gate_uses_actual_bracketing_readbacks():
    app,samples=clock_fixture();r=common.clock_readback_gate(app,samples)
    assert r['passed'] and r['bracketed_intervals']==30 and r['clock_locked_inferred'] is False


@pytest.mark.parametrize('mutation',['2730MHz','no_samples','missing_bracket','wide_bracket','nonfinite_readback'])
def test_formal_clock_gate_rejects_wrong_or_insufficient_domain(mutation):
    app,samples=clock_fixture()
    if mutation=='2730MHz':samples[10]['sm_mhz']['value']=2730
    if mutation=='no_samples':samples=[]
    if mutation=='missing_bracket':samples=samples[10:]
    if mutation=='wide_bracket':samples=[samples[0],dict(samples[-1],qpc_ticks=400000)]
    if mutation=='nonfinite_readback':samples[10]['sm_mhz']={'status':1,'value':None}
    assert not common.clock_readback_gate(app,samples)['passed']


def test_clock_receipt_is_external_hashed_evidence_not_lock_claim(tmp_path):
    stdout=tmp_path/'stdout.txt';stderr=tmp_path/'stderr.txt';stdout.write_text('successful');stderr.write_text('')
    path=tmp_path/'clock.json'
    document={'schema':'operator-clock-control-receipt/v1','gpu_uuid':'unit','target_sm_clock_mhz':2400,'sm_clock_tolerance_mhz':30,
        'requested_lock_min_mhz':2400,'requested_lock_max_mhz':2400,'lock_command_returncode':0,
        'command':['nvidia-smi','-lgc','2400,2400'],'created_utc':'2026-09-16T02:00:00Z','restore_on_exit_planned':True,
        'stdout_ref':common.ref(stdout),'stderr_ref':common.ref(stderr)}
    receipt=overwrite_json(path,document)
    assert common.verify_clock_receipt(path,receipt['sha256'],'unit')['receipt_ref']==receipt
    with pytest.raises(common.IdentityError):common.verify_clock_receipt(path,'a'*64,'unit')
    document['target_sm_clock_mhz']=2730;receipt=overwrite_json(path,document)
    with pytest.raises(common.IdentityError):common.verify_clock_receipt(path,receipt['sha256'],'unit')
