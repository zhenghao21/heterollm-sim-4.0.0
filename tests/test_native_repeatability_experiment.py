"""Pure checks for the bounded native experiment; never starts llama-server."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tools import native_repeatability_experiment as exp


def protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    model=tmp_path/'model.gguf';model.write_bytes(b'model')
    return {'schema':exp.SCHEMA,'defaults':{'measure_batches':2,'process_blocks':2},
            'jobs':[{'id':'test','model':str(model),'prompt':'hello','output':2,'parallel':2,
                     'conditions':[{'id':'a'},{'id':'b','process_affinity_mask':'0x5555','threads':8}]}]}


def capture(begin, first, last, slot):
    return {'status':'complete','response':{'id_slot':slot,'truncated':False,
      'timings':{'engine_token_times_us':[first,last],'engine_request_begin_us':begin,
                'engine_prompt_last_us':first,'engine_last_token_us':last,'engine_timepoints_complete':True,
                'prompt_n':1,'predicted_n':2,'cache_n':0}},
      'boundary':{'request_start_monotonic_s':1.,'last_token_monotonic_s':2.}}


def test_protocol_abba_and_condition_overrides(tmp_path,monkeypatch):
    plans=exp.expand_protocol(protocol(tmp_path,monkeypatch))
    assert [p['condition_id'] for p in plans]==['a','b','b','a']
    assert plans[1]['threads']==8
    assert plans[1]['resolved_cpu_affinity']==list(range(0,16,2))
    assert plans[0]['cache_ram_mib']==8192


def test_environment_override_and_secret_rejection(tmp_path,monkeypatch):
    p=protocol(tmp_path,monkeypatch)
    p['environment']={'LLAMA_TRACE_ANNOTATIONS':'1'}
    p['jobs'][0]['conditions'][1]['environment']={'LLAMA_TRACE_ANNOTATIONS':'0'}
    plans=exp.expand_protocol(p)
    assert plans[0]['environment']['LLAMA_TRACE_ANNOTATIONS']=='1'
    assert plans[1]['environment']['LLAMA_TRACE_ANNOTATIONS']=='0'
    with pytest.raises(ValueError,match='allowlisted'):
        exp.environment_for({'SECRET_TOKEN':'bad'})


def test_minus_plus_five_is_not_cv_threshold():
    result=exp.distribution([100.]*19+[106.],20)
    assert result['sample_cv_pct']<5
    assert result['worst_abs_deviation_pct']==6
    assert not result['strict_within_5pct']
    assert result['fraction_within_5pct']==.95


def test_missing_samples_stay_in_denominator():
    result=exp.distribution([100.,100.],4)
    assert result['fraction_within_5pct']==.5
    assert result['missing_or_invalid_samples']==2
    assert not result['strict_within_5pct']


def test_raw_times_generate_ranked_metrics():
    c={'parallel':2,'output':2}
    b=exp.inspect_batch([capture(2000,5000,9000,1),capture(1000,2000,4000,0)],c,1)
    assert b['status']=='complete'
    assert [r['engine_start_rank'] for r in b['requests']]==[1,0]
    assert b['batch_medians_ms']=={'ttft':2.,'tpot':3.,'e2e':5.}


def test_invalid_token_times_and_partial_requests_retained():
    c={'parallel':2,'output':2}
    good=capture(0,1000,2000,0)
    bad={'status':'failed','error':'timeout','raw_received_lines':[{'line':'data: half'}]}
    b=exp.inspect_batch([good,bad],c,1)
    assert b['status']=='failed' and len(b['requests'])==2
    assert b['requests'][1]['raw_received_lines']
    assert not b['batch_medians_ms']
    assert all('engine_start_rank' not in r for r in b['requests'])
    wrong=capture(0,2000,1000,0)
    assert exp.inspect_request(wrong['response'],wrong['boundary'],2,1)['status']=='invalid'


def test_unique_slots_required():
    b=exp.inspect_batch([capture(0,1000,2000,0),capture(0,1000,2000,0)],{'parallel':2,'output':2},1)
    assert b['status']=='failed'


def test_stable_one_block_insufficient_across_process_evidence():
    c=dict(job_id='x',condition_id='c',model='m',parallel=2,output=2,measure_batches=2)
    b=exp.inspect_batch([capture(0,1000,2000,0),capture(0,1000,2000,1)],c,1)
    r={'status':'complete','runs':[b,deepcopy(b)]}
    one=exp.summarize_group([c],[r])
    assert one['metrics']['ttft']['batch_medians']['strict_within_5pct']
    assert not one['strict_within_5pct']
    assert exp.summarize_group([c,c],[r,deepcopy(r)])['strict_within_5pct']


def test_failed_block_does_not_disappear_from_pass_criteria():
    c=dict(job_id='x',condition_id='c',model='m',parallel=2,output=2,measure_batches=2)
    b=exp.inspect_batch([capture(0,1000,2000,0),capture(0,1000,2000,1)],c,1)
    s=exp.summarize_group([c,c],[{'status':'complete','runs':[b,b]},{'status':'failed','runs':[]}])
    assert not s['strict_within_5pct']
    assert s['metrics']['ttft']['batch_medians']['fraction_within_5pct']==.5


def test_command_preserves_cache_ram_and_threads(tmp_path,monkeypatch):
    c=exp.expand_protocol(protocol(tmp_path,monkeypatch))[0]
    c.update(cache_ram_mib=0,poll=100)
    cmd=exp.command_for('llama.exe',c,9123)
    assert cmd[cmd.index('--cache-ram')+1]=='0'
    assert cmd[cmd.index('--poll')+1]=='100'
    assert cmd[cmd.index('-t')+1]=='16'


def test_strict_freeze_references_and_exclusive_writes(tmp_path):
    path=tmp_path/'raw.json';exp.write_new(path,{'a':1})
    ref=exp.file_ref(path);exp.verify_refs([ref])
    with pytest.raises(FileExistsError):exp.write_new(path,{'a':2})
    with pytest.raises(ValueError):exp.verify_refs([])
    path.write_text('changed')
    with pytest.raises(RuntimeError):exp.verify_refs([ref])


def test_resume_tampered_raw_rejected(tmp_path):
    plan={'key':'a','parallel':1,'output':2};freeze={'some':'source'}
    record={'status':'complete','config':plan,'freeze_sha256':exp.digest(freeze),'runs':[],'warmup':[]}
    raw=tmp_path/'a.json';exp.write_new(raw,record)
    receipt={'key':'a','status':'complete','freeze_sha256':exp.digest(freeze),'raw_ref':exp.file_ref(raw),'log_ref':None}
    exp.write_new(tmp_path/'a.receipt.json',receipt)
    assert exp.load_completed(tmp_path,freeze,plan)['status']=='complete'
    raw.write_text(json.dumps(dict(record,status='failed')))
    with pytest.raises(RuntimeError,match='freeze drift'):exp.load_completed(tmp_path,freeze,plan)


def test_power_guid_is_independent_of_localized_console_encoding():
    from tools.native_repeatability_experiment import parse_power_scheme_output
    guid='381b4222-f694-41f0-9685-ff5bb260df2e'
    for prefix in (bytes.fromhex('e794b5e6ba90'), bytes.fromhex('b5e7d4b4')):
        raw=prefix+b' GUID: '+guid.encode()
        result=parse_power_scheme_output(raw,b'',0)
        assert result['guid']==guid
        assert bytes.fromhex(result['stdout_raw_hex'])==raw


def test_runtime_identity_is_campaign_wide_and_immutable(tmp_path):
    freeze={'identity':'frozen'}
    runtime={'status':'captured','module_identity_sha256':'first','artifacts':[]}
    ref=exp.bind_runtime_identity(tmp_path,freeze,'block0',runtime)
    original=Path(ref['path']).read_bytes()
    assert exp.bind_runtime_identity(tmp_path,freeze,'block1',runtime)==ref
    with pytest.raises(ValueError,match='across process blocks'):
        exp.bind_runtime_identity(tmp_path,freeze,'block2',dict(runtime,module_identity_sha256='other'))
    with pytest.raises(ValueError,match='freeze mismatch'):
        exp.bind_runtime_identity(tmp_path,{'identity':'other'},'block3',runtime)
    assert Path(ref['path']).read_bytes()==original


def test_batch_journal_keeps_raw_evidence_and_never_overwrites(tmp_path):
    captures=[capture(0,1000,2000,0)]
    captures[0]['raw_received_lines']=[{'received_monotonic_s':1.25,'line':'data: {"content":"red"}\n'}]
    captures[0]['response']['raw_stream_events']=[{'content':'red','tokens':[12]}]
    ref=exp.write_batch_journal(tmp_path,{'freeze':'x'},{'key':'a'},'runs',0,1,captures,2.5)
    raw=Path(ref['path']).read_text(encoding='utf-8')
    saved=json.loads(raw)
    assert saved['captures']==captures
    assert saved['batch_wall_ms']==2.5
    assert '\n' not in raw
    with pytest.raises(FileExistsError):
        exp.write_batch_journal(tmp_path,{'freeze':'x'},{'key':'a'},'runs',0,1,captures,2.5)
    exp.verify_refs([ref])


def fake_block(tmp_path,monkeypatch,*,fail_workers=False,interrupt_after_warmup=False,config_overrides=None,tokenizer_ids=None,actual_prompt_count=1):
    events=[]
    c=dict(exp.DEFAULTS,key='a',block=0,model='fake-model',prompt='hello',environment={},
           resolved_cpu_affinity=None,output=2,parallel=2,warmup_batches=1,measure_batches=2)
    c.update(config_overrides or {})
    exe=tmp_path/'fake.exe';exe.write_bytes(b'fake')
    freeze={'hardware_fingerprint':'hw','artifact_refs':[exp.file_ref(exe)],
            'source_refs':[exp.file_ref(exe)],'environments':{'a':exp.environment_for({})[1]}}
    runtime={'status':'captured','module_identity_sha256':'locked','artifacts':[]}
    monkeypatch.setattr(exp,'helpers',lambda:(None,None,None,lambda *_:None,
        lambda *_:{'tokens':tokenizer_ids if tokenizer_ids is not None else [1]},lambda *_:deepcopy(runtime)))
    class Process:
        pid=123;returncode=None
        def poll(self):return self.returncode
        def terminate(self):self.returncode=0
        def wait(self,timeout=None):return self.returncode
    monkeypatch.setattr(exp.subprocess,'Popen',lambda *_,**__:Process())
    monkeypatch.setattr(exp,'process_state',lambda *_:{'state':'observed'})
    def validate(state,config,check_workers=True,check_gpu=True):
        events.append(('state',check_workers,check_gpu))
        if fail_workers and check_workers:raise ValueError('requested workers absent')
    monkeypatch.setattr(exp,'validate_state',validate)
    control=exp.BatchBoundaryControl(tmp_path,float('inf'))
    class Client:
        def __init__(self,_):self.count=0
        def batch(self,*_):
            self.count+=1;events.append(('batch',self.count))
            if interrupt_after_warmup and self.count==1:control.request_interrupt()
            rows=[capture(0,1000,2000,0),capture(0,1000,2000,1)]
            for row in rows:row['response']['timings']['prompt_n']=actual_prompt_count
            rows[0]['raw_received_lines']=[{'line':'data: complete\n','received_monotonic_s':2.0}]
            return rows
        def close(self):events.append(('close',))
    monkeypatch.setattr(exp,'TrialClient',Client)
    return exp.run_block(c,exe,tmp_path,freeze,control=control),events,c,freeze


def test_worker_gate_runs_after_warmup_and_before_measurement(tmp_path,monkeypatch):
    record,events,_,_=fake_block(tmp_path,monkeypatch,fail_workers=True)
    assert record['status']=='failed' and not record['runs']
    assert len(record['warmup'])==len(record['batch_journal_refs'])==1
    assert events[:3]==[('state',False,False),('batch',1),('state',True,True)]
    assert 'state_measurement_before' in record


def test_complete_block_keeps_journals_and_checks_workers_on_both_sides(tmp_path,monkeypatch):
    record,events,plan,freeze=fake_block(tmp_path,monkeypatch)
    assert record['status']=='complete'
    assert len(record['batch_journal_refs'])==3
    assert events==[('state',False,False),('batch',1),('state',True,True),('batch',2),('batch',3),('state',True,True),('close',)]
    assert exp.load_completed(tmp_path,freeze,plan)['status']=='complete'
    Path(record['batch_journal_refs'][0]['path']).write_text('changed',encoding='utf-8')
    with pytest.raises(RuntimeError,match='freeze drift'):
        exp.load_completed(tmp_path,freeze,plan)


def test_cooperative_interrupt_preserves_finished_batch_and_incomplete_receipt(tmp_path,monkeypatch):
    record,events,plan,freeze=fake_block(tmp_path,monkeypatch,interrupt_after_warmup=True)
    assert record['status']=='incomplete' and record['stop_reason']=='keyboard_interrupt_at_batch_boundary'
    assert len(record['warmup'])==1 and not record['runs']
    assert [event for event in events if event[0]=='batch']==[('batch',1)]
    restored=exp.load_completed(tmp_path,freeze,plan)
    assert restored['status']=='incomplete' and len(restored['warmup'])==1
    exp.verify_refs(restored['batch_journal_refs'])


def test_boundary_control_budget_stop_and_sigint_restoration(tmp_path,monkeypatch):
    monkeypatch.setattr(exp.time,'monotonic',lambda:10.)
    control=exp.BatchBoundaryControl(tmp_path,11.)
    assert control.reason() is None
    (tmp_path/'STOP').write_text('stop',encoding='utf-8')
    with pytest.raises(exp.BatchBoundaryStop,match='operator_stop'):
        control.check()
    other=exp.BatchBoundaryControl(tmp_path/'no-stop',10.)
    assert other.reason()=='budget_exhausted_at_batch_boundary'
    previous=object();handlers=[]
    monkeypatch.setattr(exp.signal,'getsignal',lambda _:previous)
    monkeypatch.setattr(exp.signal,'signal',lambda _,handler:handlers.append(handler))
    with exp.defer_keyboard_interrupts(other):
        handlers[-1](None,None)
        assert other.reason()=='keyboard_interrupt_at_batch_boundary'
    assert handlers[-1] is previous


def test_orphan_log_resume_still_fails_closed(tmp_path):
    (tmp_path/'a.log').write_text('partial',encoding='utf-8')
    with pytest.raises(ValueError,match='orphan partial block'):
        exp.load_completed(tmp_path,{}, {'key':'a'})


def test_expected_gpu_clock_uses_sm_column_and_fails_closed(monkeypatch):
    monkeypatch.setattr(exp.sys,'platform','linux')
    config={'expected_gpu_sm_clock_mhz':2400,'gpu_sm_clock_tolerance_mhz':30}
    def state(value):return {'gpu_state':{'returncode':0,'stdout':value}}
    exp.validate_state(state('GPU-0, P0, 55, 2425 MHz, 15000 MHz, 80 W, 5 %'),config)
    with pytest.raises(ValueError,match='GPU SM clock'):
        exp.validate_state(state('GPU-0, P0, 55, 2692 MHz, 2400 MHz, 80 W, 5 %'),config)
    for value in ('','GPU-0, P0, 55, N/A, 2400 MHz'):
        with pytest.raises(ValueError,match='GPU SM clock'):
            exp.validate_state(state(value),config)
    exp.validate_state({},config,check_gpu=False)


@pytest.mark.parametrize('parallel', [1,2,4])
def test_exact_tokens_and_per_slot_capacity_are_frozen(tmp_path,monkeypatch,parallel):
    p=protocol(tmp_path,monkeypatch)
    p['jobs'][0].pop('prompt')
    p['jobs'][0].update(prompt_token_ids=list(range(1536)),expected_prompt_tokens=1536,
                        output=256,parallel=parallel,kv_unified_per_slot=2048)
    c=exp.expand_protocol(p)[0]
    assert c['ctx']==2048*parallel
    assert len(c['prompt_token_ids'])==c['expected_prompt_tokens']==1536
    cmd=exp.command_for('fake.exe',c,1234)
    assert cmd[cmd.index('--kv-unified-per-slot')+1]=='2048'
    assert cmd[cmd.index('-c')+1]==str(2048*parallel)


@pytest.mark.parametrize('ids', [[],[True],[1.0],[-1],'1',[[1]]])
def test_exact_token_ids_must_be_nonempty_integer_array(tmp_path,monkeypatch,ids):
    p=protocol(tmp_path,monkeypatch);p['jobs'][0]['prompt_token_ids']=ids
    with pytest.raises(ValueError,match='prompt_token_ids'):
        exp.expand_protocol(p)


def test_exact_count_and_context_must_match(tmp_path,monkeypatch):
    p=protocol(tmp_path,monkeypatch)
    job=p['jobs'][0];job.update(prompt_token_ids=[1,2],expected_prompt_tokens=3)
    with pytest.raises(ValueError,match='expected_prompt_tokens'):
        exp.expand_protocol(p)
    job.update(expected_prompt_tokens=2,kv_unified_per_slot=2048,ctx=2048)
    with pytest.raises(ValueError,match='ctx must equal'):
        exp.expand_protocol(p)
    job['ctx']=4096;job['output']=2047
    with pytest.raises(ValueError,match='per-slot capacity'):
        exp.expand_protocol(p)
    job['output']=2046
    assert exp.expand_protocol(p)[0]['output']==2046


@pytest.mark.parametrize('expected', [True,0,-1,1.5])
def test_text_expected_count_is_positive_integer(tmp_path,monkeypatch,expected):
    p=protocol(tmp_path,monkeypatch);p['jobs'][0]['expected_prompt_tokens']=expected
    with pytest.raises(ValueError,match='expected_prompt_tokens'):
        exp.expand_protocol(p)


def test_tokenizer_confirms_exact_ids_without_adding_special_tokens():
    c=dict(exp.DEFAULTS,prompt_token_ids=[1,7,9],expected_prompt_tokens=3)
    seen=[]
    def post(url,payload):seen.append((url,payload));return {'tokens':[1,7,9]}
    tokenizer,ids,content=exp.resolve_prompt(c,post,'http://local')
    assert ids==content==c['prompt_token_ids']
    assert seen==[('http://local/tokenize',{'content':[1,7,9],'add_special':False})]
    for wrong in ([0,1,7,9],[1,8,9],[True,7,9]):
        with pytest.raises(ValueError,match='tokenizer'):
            exp.resolve_prompt(c,lambda *_:{'tokens':wrong},'http://local')


def test_text_prompt_keeps_special_token_behavior_and_checks_actual_capacity():
    c=dict(exp.DEFAULTS,prompt='hello',expected_prompt_tokens=1)
    seen=[]
    def post(url,payload):seen.append(payload);return {'tokens':[8]}
    assert exp.resolve_prompt(c,post,'http://local')[2]=='hello'
    assert seen==[{'content':'hello','add_special':True}]
    with pytest.raises(ValueError,match='expected_prompt_tokens'):
        exp.resolve_prompt(c,lambda *_:{'tokens':[1,2]},'http://local')
    c.pop('expected_prompt_tokens');c.update(kv_unified_per_slot=2,output=2)
    with pytest.raises(ValueError,match='per-slot capacity'):
        exp.resolve_prompt(c,post,'http://local')


def test_exact_payload_and_actual_prompt_count_are_preserved(tmp_path,monkeypatch):
    record,_,plan,freeze=fake_block(tmp_path,monkeypatch,
        config_overrides={'prompt_token_ids':[1,7,9],'expected_prompt_tokens':3,
                          'kv_unified_per_slot':2048,'ctx':4096},
        tokenizer_ids=[1,7,9],actual_prompt_count=3)
    assert record['status']=='complete'
    assert record['payload']['prompt']==[1,7,9]
    assert record['prompt_token_count']==3
    assert len(record['batch_journal_refs'])==3
    assert exp.load_completed(tmp_path,freeze,plan)['status']=='complete'


def test_exact_tokenizer_mismatch_fails_before_warmup_with_evidence(tmp_path,monkeypatch):
    record,events,_,_=fake_block(tmp_path,monkeypatch,
        config_overrides={'prompt_token_ids':[1,7,9],'expected_prompt_tokens':3},
        tokenizer_ids=[1,7,8])
    assert record['status']=='failed'
    assert not record['warmup'] and not record['runs']
    assert not [event for event in events if event[0]=='batch']
    assert record['tokenizer']=={'tokens':[1,7,8]}


def test_request_prompt_mismatch_is_journaled_and_fails_closed(tmp_path,monkeypatch):
    record,_,_,_=fake_block(tmp_path,monkeypatch,
        config_overrides={'prompt_token_ids':[1,7,9],'expected_prompt_tokens':3},
        tokenizer_ids=[1,7,9],actual_prompt_count=4)
    assert record['status']=='failed' and not record['runs']
    assert len(record['batch_journal_refs'])==1
    assert 'tokenizer_or_output_count_mismatch' in record['warmup'][0]['requests'][0]['errors']


@pytest.mark.parametrize('counter', [True,1.0,'1',0])
def test_server_prompt_counter_is_strict_integer(counter):
    row=capture(0,1000,2000,0);row['response']['timings']['prompt_n']=counter
    assert exp.inspect_request(row['response'],row['boundary'],2,1)['status']=='invalid'


def test_data_root_paths_stay_separate_from_execution_sources(tmp_path,monkeypatch):
    source=tmp_path/'execution';source.mkdir()
    data=tmp_path/'project';data.mkdir()
    (data/'model.gguf').write_bytes(b'model')
    monkeypatch.setattr(exp,'ROOT',source);monkeypatch.setattr(exp,'DATA_ROOT',data)
    assert exp.project_file('model.gguf')==data/'model.gguf'
    (source/'other.gguf').write_bytes(b'other')
    with pytest.raises(ValueError,match='data root'):
        exp.project_file(source/'other.gguf')
    for field,value in (('data_root',str(source)),('execution_source_root',str(data))):
        freeze={'data_root':str(data),'execution_source_root':str(source),field:value}
        with pytest.raises(ValueError,match='root freeze mismatch'):
            exp.verify_freeze(freeze,data/'protocol.json',[])


def test_snapshot_freeze_ignores_live_reporting_but_detects_snapshot_drift(tmp_path,monkeypatch):
    source=tmp_path/'execution';source.mkdir()
    data=tmp_path/'project';data.mkdir()
    model=data/'model.gguf';model.write_bytes(b'model')
    exe=data/'fake.exe';exe.write_bytes(b'exe')
    live=data/'reporting.py';live.write_text('version=1')
    frozen_source=source/'reporting.py';frozen_source.write_bytes(live.read_bytes())
    protocol_path=data/'protocol.json';protocol_path.write_text('{}')
    output=tmp_path/'output';output.mkdir()
    monkeypatch.setattr(exp,'ROOT',source);monkeypatch.setattr(exp,'DATA_ROOT',data)
    monkeypatch.setattr(exp,'source_files',lambda:[frozen_source])
    monkeypatch.setattr(exp,'helpers',lambda:(lambda:{'host':'test'},lambda _:'hw',
        lambda:{'sha256':'extractor'},None,None,None))
    plans=[{'model':str(model),'key':'a','environment':{}}]
    freeze=exp.build_freeze(protocol_path,{},plans,exe,output)
    assert freeze['data_root']==str(data)
    assert freeze['execution_source_root']==str(source)
    live.write_text('version=2')
    exp.verify_freeze(freeze,protocol_path,plans)
    frozen_source.write_text('version=3')
    with pytest.raises(RuntimeError,match='freeze drift'):
        exp.verify_freeze(freeze,protocol_path,plans)
