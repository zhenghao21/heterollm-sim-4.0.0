import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import dual_hash as h
import diagnostic as d

ABC='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


def digest(data):
    value=h.openssl_sha256();value.update(data);return value.hexdigest()


def test_actual_independent_engines_known_short_vectors():
    result=h.self_test()
    assert result['known_vectors_passed']==2 and result['million_a_control'] is False
    assert result['fallback_used'] is False and result['CNG_provider']=='Microsoft Primitive Provider'


def test_actual_CNG_binary_NUL_and_multi_update():
    data=bytes(range(256))*7
    with h.CNGSHA256() as value:
        for p in range(0,len(data),31):value.update(data[p:p+31])
        assert value.hexdigest()==digest(data)
        with pytest.raises(ValueError):value.update(b'late')


def test_small_file_chunk_logs_full_hash_and_stat(tmp_path):
    data=bytes(range(256))*9
    path=tmp_path/'fixture.bin';path.write_bytes(data)
    r=h.scan_file(path,tmp_path/'out',len(data),digest(data),chunk_bytes=127)
    assert r['status']=='agreed_expected_identity' and r['bytes_read']==len(data)
    assert r['stat_unchanged'] and r['file_open_count']==1
    rows=[json.loads(s) for s in (tmp_path/'out/chunks.jsonl').read_text().splitlines()]
    assert sum(r['bytes'] for r in rows)==len(data)
    assert all(row['equal'] for row in rows)
    assert r['full_openssl_sha256']==r['full_cng_sha256']==digest(data)


class BadCNG:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def update(self,data):pass
    def hexdigest(self):return '0'*64


def test_disagreement_never_selects_matching_openssl(tmp_path):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abc')
    r=h.scan_file(path,tmp_path/'out',3,ABC,cng_factory=BadCNG)
    assert r['status']=='rejected' and r['chunk_digest_disagreements']==[0]
    assert r['full_openssl_sha256']==ABC and r['full_cng_sha256']!=ABC
    assert r['algorithm_winner_selected'] is False


def test_both_algorithms_agree_on_wrong_identity_still_rejected(tmp_path):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abc')
    r=h.scan_file(path,tmp_path/'out',3,'1'*64)
    assert r['full_cng_sha256']==r['full_openssl_sha256']==ABC and r['status']=='rejected'


def test_CNG_unavailable_is_failure_before_file_open(tmp_path):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abc')
    def unavailable():raise OSError('BCrypt unavailable')
    r=h.scan_file(path,tmp_path/'out',3,ABC,cng_factory=unavailable)
    assert r['status']=='rejected' and r['bytes_read']==0 and r['file_open_count']==0
    assert any('BCrypt unavailable' in s for s in r['errors'])


def test_size_mismatch_zero_input_bytes(tmp_path):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abc')
    r=h.scan_file(path,tmp_path/'out',4,ABC)
    assert r['status']=='rejected' and r['bytes_read']==0


def test_file_growth_during_read_is_rejected_without_extra_EOF_read(tmp_path):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abc')
    class Mutating(h.CNGSHA256):
        def update(self,data):
            super().update(data)
            if path.stat().st_size==3:
                with path.open('ab') as f:f.write(b'x')
    r=h.scan_file(path,tmp_path/'out',3,ABC,cng_factory=Mutating)
    assert r['status']=='rejected' and not r['stat_unchanged'] and r['bytes_read']==3
    assert r['extra_EOF_probe_bytes']==0


@pytest.mark.parametrize('short',[False,True])
def test_early_EOF_or_short_read_is_preserved_not_retried(tmp_path,monkeypatch,short):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abcdef')
    original=Path.open;calls=[]
    class Stream:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*args):self.f.close()
        def fileno(self):return self.f.fileno()
        def read(self,size):
            calls.append(size)
            return self.f.read(1) if short else b''
    def open_file(p,*args,**kwargs):
        f=original(p,*args,**kwargs)
        return Stream(f) if p==path and args and args[0]=='rb' else f
    monkeypatch.setattr(Path,'open',open_file)
    r=h.scan_file(path,tmp_path/'out',6,digest(b'abcdef'),chunk_bytes=3)
    assert r['status']=='rejected' and len(calls)==1
    assert r['bytes_read']==(1 if short else 0)


def test_append_only_pass_refuses_overwrite(tmp_path):
    path=tmp_path/'fixture.bin';path.write_bytes(b'abc')
    h.scan_file(path,tmp_path/'out',3,ABC)
    with pytest.raises(FileExistsError):h.scan_file(path,tmp_path/'out',3,ABC)


def test_small_memory_control_preserves_both_results(tmp_path):
    r=d.memory_control(tmp_path/'control.json',b'abc',ABC)
    assert r['status']=='both_match_known_vector' and r['disk_input_bytes']==0
    assert r['update_calls_per_engine']==1


def test_bad_memory_control_retains_divergence(monkeypatch,tmp_path):
    monkeypatch.setattr(h,'CNGSHA256',BadCNG)
    with pytest.raises(RuntimeError):d.memory_control(tmp_path/'control.json',b'abc',ABC)
    r=json.loads((tmp_path/'control.json').read_text())
    assert r['status']=='rejected' and r['openssl_sha256']==ABC and r['cng_sha256']=='0'*64


def test_auxiliary_budget_rejects_before_read(tmp_path,monkeypatch):
    p=tmp_path/'fixture';p.write_bytes(b'abc')
    budget=d.ReadBudget(2)
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('must not read over budget'))
    with pytest.raises(ValueError,match='budget'):budget.read(p)
    assert budget.used==0


def test_chunk_comparison_localizes_read_difference(tmp_path):
    for label,value in [('serial','a'),('parallel','b')]:
        folder=tmp_path/label;folder.mkdir()
        row={'index':0,'offset':0,'bytes':3,'openssl_sha256':value,'cng_sha256':value,'equal':True}
        (folder/'chunks.jsonl').write_text(json.dumps(row)+'\n')
    r=d.compare_chunks(tmp_path,['serial','parallel'],d.ReadBudget())
    assert not r['full_pass_consistency'] and r['differences'][0]['index']==0


@pytest.fixture
def terminal_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'R27',tmp_path)
    controls=tmp_path/'controls.json';controls.write_text('{}')
    budget=d.ReadBudget();cr=d.ref(controls,budget)
    protocol={'campaign_refs':[cr],'arms':{}}
    barrier={'schema':'r27-full262-terminal-barrier/v1','terminal_count':262,'arms':{},
             'controls_ref':cr,'failures_preserved':True,'created_utc':'2026-09-17T02:00:00+00:00'}
    for arm in ['off','on']:
        folder=tmp_path/arm/'predictions';folder.mkdir(parents=True)
        freeze=tmp_path/arm/'freeze.json';freeze.write_text('{}');fr=d.ref(freeze,budget)
        protocol['campaign_refs'].append(fr)
        ids=['cell_%03d'%i for i in range(131)]
        protocol['arms'][arm]={'freeze_ref':fr,'cell_ids':ids,'source_sha256':'source','selection_sha256':'selection'}
        refs={}
        for i,cell in enumerate(ids):
            p=folder/(cell+'.prediction.json')
            p.write_text(json.dumps({'schema':'stable-native-cell-prediction/v1','cell_id':cell,
                'status':'failed' if i==0 else 'predicted','freeze_ref':fr,'source_sha256':'source',
                'selection_sha256':'selection','finished_utc':'2026-09-17T01:00:00+00:00'}))
            refs[cell]=d.ref(p,budget)
        barrier['arms'][arm]={'freeze_ref':fr,'prediction_refs':refs}
    (tmp_path/'predictions_complete.json').write_text(json.dumps(barrier))
    return protocol,barrier,tmp_path


def test_all262_terminal_accepts_preserved_failure_without_scoring(terminal_fixture):
    p,b,root=terminal_fixture
    r=d.terminal_gate(p,d.ReadBudget())
    assert r['terminal_count']==262 and r['status_counts_only']['off']['failed']==1
    assert r['model_metrics_read_for_scoring'] is False


@pytest.mark.parametrize('mutation',['missing_barrier','missing_cell','pending','reference_changed','early_barrier'])
def test_incomplete_or_changed_terminal_gate_rejected(terminal_fixture,mutation):
    p,b,root=terminal_fixture
    barrier=root/'predictions_complete.json'
    if mutation=='missing_barrier':barrier.unlink()
    elif mutation=='missing_cell':
        b['arms']['on']['prediction_refs'].pop('cell_001');barrier.write_text(json.dumps(b))
    elif mutation=='early_barrier':
        b['created_utc']='2026-09-16T00:00:00+00:00';barrier.write_text(json.dumps(b))
    else:
        path=root/'off/predictions/cell_001.prediction.json'
        row=json.loads(path.read_text());row['status']='pending';path.write_text(json.dumps(row))
        if mutation=='pending':
            b['arms']['off']['prediction_refs']['cell_001']=d.ref(path,d.ReadBudget());barrier.write_text(json.dumps(b))
    with pytest.raises((ValueError,FileNotFoundError)):d.terminal_gate(p,d.ReadBudget())


def test_idle_detector_blocks_R27_and_unknown_python_but_allows_compiler():
    rows=[{'pid':1,'name':'python.exe','cmdline':['python','run_candidate.py','full']},
          {'pid':2,'name':'python.exe','cmdline':None},
          {'pid':3,'name':'cl.exe','cmdline':[]},
          {'pid':4,'name':'explorer.exe','cmdline':[]},
          {'pid':5,'name':'python.exe','cmdline':['python','diagnostic.py','run']}]
    assert [r['pid'] for r in d.conflicts(rows,current_pid=5)]==[1,2]


def test_missing_execution_switch_prevents_any_runner(monkeypatch):
    monkeypatch.setattr(d.sys,'argv',['diagnostic.py','run'])
    monkeypatch.setattr(d,'execute',lambda *a:pytest.fail('large read must not begin'))
    with pytest.raises(ValueError,match='switch'):d.main()


@pytest.mark.parametrize('gate',['terminal','idle'])
def test_busy_or_unfinished_R27_leaves_failed_receipt_with_zero_model_reads(tmp_path,monkeypatch,gate):
    monkeypatch.setattr(d,'HERE',tmp_path)
    protocol={'target':{'bytes':99},'passes':['serial_01','parallel_01','parallel_02','parallel_03','parallel_04'],'run_directory':'read_comparison_0002','maximum_model_read_bytes':495,'prior_attempt':{'reported_model_read_bytes':99}}
    monkeypatch.setattr(d,'configure_execution_budget',lambda *a:{'mock_formula':True})
    monkeypatch.setattr(d,'verify_prepared',lambda *a:(protocol,{'sha256':'frozen'}))
    monkeypatch.setattr(h,'self_test',lambda:{'known_vectors_passed':2})
    def reject(*a):raise ValueError('R27 still active')
    monkeypatch.setattr(d,'terminal_gate',reject if gate=='terminal' else lambda *a:{'barrier_ref':{},'terminal_count':262})
    monkeypatch.setattr(d,'idle_snapshot',reject if gate=='idle' else lambda:{})
    monkeypatch.setattr(d,'worker',lambda *a:pytest.fail('must not read target'))
    with pytest.raises(RuntimeError):d.execute(tmp_path/'manifest.json')
    r=json.loads((tmp_path/'runs/read_comparison_0002/finish.json').read_text())
    assert r['status']=='rejected' and r['model_read_upper_bound_bytes']==0 and r['actual_model_read_bytes_reported']==0


@pytest.mark.parametrize('bad_label',[None,'serial_01','parallel_03'])
def test_fixed_serial_then_four_mock_workers_never_retries(tmp_path,monkeypatch,bad_label):
    monkeypatch.setattr(d,'HERE',tmp_path)
    labels=['serial_01','parallel_01','parallel_02','parallel_03','parallel_04']
    protocol={'target':{'bytes':3},'passes':labels,'run_directory':'read_comparison_0002','maximum_model_read_bytes':15,'prior_attempt':{'reported_model_read_bytes':3}}
    monkeypatch.setattr(d,'configure_execution_budget',lambda *a:{'mock_formula':True})
    verify_calls=[];terminal_calls=[]
    def verify(*a):verify_calls.append('full_program_runtime');return protocol,{'sha256':'fixed'}
    def terminal(*a):terminal_calls.append('full262');return {'barrier_ref':{'sha256':'terminal'},'terminal_count':262}
    monkeypatch.setattr(d,'verify_prepared',verify)
    monkeypatch.setattr(d,'terminal_gate',terminal)
    monkeypatch.setattr(d,'idle_snapshot',lambda:{'conflicts':[]})
    monkeypatch.setattr(h,'self_test',lambda:{'known_vectors_passed':2})
    calls=[];started=[];joined=[]
    def worker(protocol,output,event=None):
        output=Path(output);label=output.name
        if event is not None:assert event.released and len(started)==4
        calls.append(label);output.mkdir()
        h.write_new(output/'finish.json',{'bytes_read':3,'stat_unchanged':True,
          'status':'rejected' if label==bad_label else 'agreed_expected_identity'})
        h.write_new(output.parent/(label+'.memory_control.json'),{'status':'both_match_known_vector'})
    monkeypatch.setattr(d,'worker',worker)
    monkeypatch.setattr(d,'compare_chunks',lambda *a,**k:{'full_pass_consistency':True,'compared_chunks':1,'differences':[]})
    class Event:
        released=False
        def set(self):self.released=True
    class Child:
        exitcode=0
        def __init__(self,target,args,name):self.target=target;self.args=args;self.name=name;self.pid=100+len(started)
        def start(self):started.append(self.name)
        def join(self):self.target(*self.args);joined.append(self.name)
    monkeypatch.setattr(d.mp,'get_context',lambda value:SimpleNamespace(Event=Event,Process=Child))
    if bad_label:
        with pytest.raises(RuntimeError):d.execute(tmp_path/'manifest.json')
    else:assert d.execute(tmp_path/'manifest.json')['status']=='five_pass_agreement_diagnostic_only'
    r=json.loads((tmp_path/'runs/read_comparison_0002/finish.json').read_text())
    assert calls==labels and started==labels[1:] and joined==labels[1:]
    assert len(verify_calls)==len(terminal_calls)==3
    assert r['model_read_upper_bound_bytes']==15 and r['actual_model_read_bytes_reported']==15
    assert r['cumulative_model_read_upper_bound_bytes']==18 and r['cumulative_actual_model_read_bytes_reported']==18
    assert r['prior_attempt_remains_rejected'] and r['prior_serial_reused'] is False
    assert r['repairs_or_retries']==0 and r['old_failures_modified'] is False
    if bad_label:assert r['status']=='rejected'
    with pytest.raises(FileExistsError):d.execute(tmp_path/'manifest.json')



def test_prepared_runtime_paths_are_case_insensitive_but_location_sensitive(monkeypatch,tmp_path):
    protocol={'passes':['serial_01','parallel_01','parallel_02','parallel_03','parallel_04'],
              'parallel_workers':4,'target':{'bytes':3},'maximum_model_read_bytes':15,'chunk_bytes':h.MAX_CHUNK}
    m={'schema':'dual-hash-preparation/v1','files':[],
       'runtime_files':[{'path':r'C:\Windows\System32\bcrypt.dll'}],'crypto_environment':{}}
    monkeypatch.setattr(d,'document',lambda path,budget,**k:protocol if Path(path).name=='protocol.json' else m)
    monkeypatch.setattr(d,'check_ref',lambda *a:None)
    monkeypatch.setattr(d,'ref',lambda *a:{'sha256':'fixed'})
    monkeypatch.setattr(d,'crypto_environment',lambda:{})
    monkeypatch.setattr(d,'runtime_files',lambda:[Path(r'C:\WINDOWS\System32\bcrypt.dll')])
    assert d.verify_prepared(tmp_path/'manifest.json',d.ReadBudget())[0]==protocol
    monkeypatch.setattr(d,'runtime_files',lambda:[Path(r'C:\elsewhere\bcrypt.dll')])
    with pytest.raises(ValueError,match='runtime path'):d.verify_prepared(tmp_path/'manifest.json',d.ReadBudget())



def test_windows_path_handle_ctime_semantics_and_same_API_changes():
    a={'st_dev':1,'st_ino':2,'st_size':3,'st_mtime_ns':4,'st_birthtime_ns':5,'st_ctime_ns':5}
    b={**a,'st_ctime_ns':9}
    assert h.same_file_stat(a,b,cross_api=True)
    assert not h.same_file_stat(a,b)
    for field in ('st_dev','st_ino','st_size','st_mtime_ns','st_birthtime_ns'):
        assert not h.same_file_stat(a,{**b,field:999},cross_api=True)


def test_three_byte_actual_Windows_stat_and_handle(tmp_path):
    path=tmp_path/'three.bin';path.write_bytes(b'abc')
    r=h.scan_file(path,tmp_path/'out',3,ABC)
    assert r['status']=='agreed_expected_identity'
    assert r['stat_unchanged'] and all(r['stat_checks'].values())
    assert 'st_birthtime_ns' in r['handle_stat_before']
    assert 'st_ctime_ns' in r['path_stat_before']


def test_handle_ctime_change_not_hidden_by_cross_API_rule(monkeypatch,tmp_path):
    path=tmp_path/'three.bin';path.write_bytes(b'abc')
    original=h.stat_record;count=[]
    def record(st):
        r=original(st);count.append(1)
        if len(count)==3:r['st_ctime_ns']+=100
        return r
    monkeypatch.setattr(h,'stat_record',record)
    r=h.scan_file(path,tmp_path/'out',3,ABC)
    assert r['status']=='rejected' and not r['stat_checks']['handle_before_after']


def fake_cleanup(destroy_status=0,close_status=0):
    calls=[]
    obj=h.CNGSHA256.__new__(h.CNGSHA256)
    obj.handle=h.C.c_void_p(1);obj.alg=h.C.c_void_p(2)
    obj.dll=SimpleNamespace(
        BCryptDestroyHash=lambda value:(calls.append('destroy') or destroy_status),
        BCryptCloseAlgorithmProvider=lambda value,flags:(calls.append('close') or close_status))
    return obj,calls


@pytest.mark.parametrize('statuses',[(1,0),(0,-1),(1,-1)])
def test_CNG_cleanup_checks_both_statuses(statuses):
    obj,calls=fake_cleanup(*statuses)
    with pytest.raises(RuntimeError,match='BCrypt'):obj.close()
    assert calls==['destroy','close']
    assert not obj.handle.value and not obj.alg.value


def test_CNG_cleanup_retains_original_exception_and_all_errors():
    obj,calls=fake_cleanup(1,-1)
    original=ValueError('original hash failure')
    with pytest.raises(ValueError) as got:
        with obj:raise original
    assert got.value is original and calls==['destroy','close']
    evidence=h.error_record(got.value)
    assert evidence['message']=='original hash failure' and len(evidence['CNG_cleanup_errors'])==2


def test_outer_CNG_cleanup_failure_rejects_already_computed_file(tmp_path):
    path=tmp_path/'fixture';path.write_bytes(b'abc');count=[]
    class BadClose:
        def __init__(self):count.append(1);self.full=len(count)==1;self.value=h.openssl_sha256()
        def __enter__(self):return self
        def update(self,data):self.value.update(data)
        def hexdigest(self):return self.value.hexdigest()
        def __exit__(self,*args):
            if self.full:raise RuntimeError('BCryptDestroyHash failed injected')
    r=h.scan_file(path,tmp_path/'out',3,ABC,cng_factory=BadClose)
    assert r['full_openssl_sha256']==r['full_cng_sha256']==ABC
    assert r['status']=='rejected' and r['exception_evidence'][0]['type']=='RuntimeError'


def test_exact_gpu_collector_and_relative_runner_commands_blocked():
    rows=[{'pid':1,'name':'gpu-operator-timing.exe','cmdline':None},
          {'pid':2,'name':'python.exe','cmdline':'python runner.py run --run-dir run.0001'},
          {'pid':3,'name':'python.exe','cmdline':['python','runner.py','resume']},
          {'pid':4,'name':'python.exe','cmdline':['python','runner.py','extend']}]
    assert [r['pid'] for r in d.conflicts(rows,current_pid=99)]==[1,2,3,4]


def test_spawn_parent_ownership_or_unknown_fails_closed():
    rows=[{'pid':1,'ppid':100,'name':'python.exe','cmdline':['python','runner.py','run']},
          {'pid':2,'ppid':1,'name':'python.exe','cmdline':['python','-c','from multiprocessing.spawn import spawn_main; spawn_main()','--multiprocessing-fork']},
          {'pid':3,'ppid':888,'name':'python.exe','cmdline':['python','--multiprocessing-fork']},
          {'pid':4,'ppid':5,'name':'python.exe','cmdline':['python','--multiprocessing-fork']},
          {'pid':5,'ppid':4,'name':'python.exe','cmdline':['python','--multiprocessing-fork']},
          {'pid':6,'name':'python.exe','cmdline':['python','-m','pytest','test_identity_diagnostic.py']},
          {'pid':7,'ppid':6,'name':'python.exe','cmdline':['python','--multiprocessing-fork']},
          {'pid':8,'name':'python.exe','cmdline':['python','diagnostic.py','self-test']},
          {'pid':9,'name':'python.exe','cmdline':['python','build.py','host-test']}]
    bad=d.conflicts(rows,current_pid=99)
    assert [r['pid'] for r in bad]==[1,2,3,4,5]
    assert all('cmdline' not in r for r in bad)


def test_fake_host_label_in_code_does_not_allow_unknown_spawn():
    rows=[{'pid':1,'name':'python.exe','cmdline':['python','-c','pytest test_identity_diagnostic.py']},
          {'pid':2,'ppid':1,'name':'python.exe','cmdline':['python','--multiprocessing-fork']}]
    assert [r['pid'] for r in d.conflicts(rows,current_pid=99)]==[2]



def test_scan_records_original_update_and_cleanup_errors(tmp_path):
    path=tmp_path/'fixture';path.write_bytes(b'abc')
    class FailBoth:
        def __enter__(self):return self
        def update(self,data):raise ValueError('original update error')
        def __exit__(self,kind,error,tb):
            obj,_=fake_cleanup(1,-1);obj.close(error)
    r=h.scan_file(path,tmp_path/'out',3,ABC,cng_factory=FailBoth)
    assert r['status']=='rejected'
    evidence=r['exception_evidence'][0]
    assert evidence['type']=='ValueError' and evidence['message']=='original update error'
    assert len(evidence['CNG_cleanup_errors'])==2



def large_length_plan():
    total=1846642198
    lengths=[total//262]*262
    lengths[-1]+=total-sum(lengths)
    return dict(campaign_lengths=[1438,1706,59805,47875629,53021323],barrier_bytes=108182,
        prediction_lengths=lengths,source_lengths=[12482,30000,8000,28000,5000],
        runtime_lengths=[104208,62736,5297992,183376,716408],
        historical_lengths=[4483,19823,99202,3000,3409120,479],
        target_bytes=14865116128,chunk_bytes=h.MAX_CHUNK)


def test_actual_scale_formula_accounts_all_three_full262_checks_without_allocating_data():
    args=large_length_plan();plan=d.metadata_budget_formula(**args)
    assert plan['exact_frozen_length_totals']['predictions']==1846642198
    assert plan['per_terminal_bytes']==1947710281
    assert plan['terms']['three_full262_terminal_checks']==5843130843
    assert plan['declared_metadata_read_limit_bytes']>2*1024**3
    assert plan['required_metadata_read_upper_bound_bytes']<=plan['declared_metadata_read_limit_bytes']
    assert plan['declared_metadata_read_limit_bytes']-plan['required_metadata_read_upper_bound_bytes']<1024**2
    assert plan['all_required_full_content_checks_retained'] and not plan['stat_substituted_for_full_hash']
    # Charge the complete successful execution's upper bounds in true call counts.
    charges=[]
    for phase in range(3):
        charges.append(sum(args['campaign_lengths'])+args['barrier_bytes']+sum(args['prediction_lengths']))
        charges.append(sum(args['source_lengths'])+sum(args['runtime_lengths'])+sum(args['historical_lengths'])+2*d.PROTOCOL_MAX_BYTES+2*d.MANIFEST_MAX_BYTES)
    charges.append(d.MANIFEST_MAX_BYTES)
    charges += [plan['chunk_log_bound']['maximum_log_bytes_per_pass']+d.PASS_RECEIPT_MAX_BYTES+d.MEMORY_RECEIPT_MAX_BYTES]*5
    assert sum(charges)==plan['required_metadata_read_upper_bound_bytes']
    assert sum(charges)<=plan['declared_metadata_read_limit_bytes']


def test_chunk_bound_covers_every_line_of_complete_large_file_without_reading_it():
    size=14865116128;bound=d.chunk_log_upper_bound(size,h.MAX_CHUNK)
    total=0;count=0
    for offset in range(0,size,h.MAX_CHUNK):
        record={'index':count,'offset':offset,'bytes':min(h.MAX_CHUNK,size-offset),
            'openssl_sha256':'f'*64,'cng_sha256':'0'*64,'equal':False}
        encoded=(json.dumps(record)+'\n').encode('utf8')
        assert len(encoded)<=bound['maximum_chunk_line_bytes']
        total+=len(encoded);count+=1
    assert count==14177 and total<=bound['maximum_log_bytes_per_pass']


@pytest.mark.parametrize('mutation',['missing_prediction','negative','bool'])
def test_metadata_formula_rejects_incomplete_or_invalid_lengths(mutation):
    args=large_length_plan()
    if mutation=='missing_prediction':args['prediction_lengths'].pop()
    elif mutation=='negative':args['runtime_lengths'][0]=-1
    else:args['prediction_lengths'][0]=True
    with pytest.raises(ValueError):d.metadata_budget_formula(**args)


def synthetic_preflight(tmp_path,monkeypatch):
    target=tmp_path/'never_open.gguf';sizes={}
    def ref(name,size):
        path=(tmp_path/name).resolve();sizes[str(path)]=size
        return {'path':str(path),'bytes':size,'sha256':'a'*64}
    args=large_length_plan()
    snapshots={'off':{'prediction_refs':{}},'on':{'prediction_refs':{}}}
    for i,size in enumerate(args['prediction_lengths']):snapshots['off' if i<131 else 'on']['prediction_refs'][str(i)]=ref(f'prediction_{i}.json',size)
    selected=ref('protocol.0004.json',200000);manifest_path=tmp_path/'manifest.json';sizes[str(manifest_path.resolve())]=8000
    manifest={'files':[ref(f'source_{i}.py',size) for i,size in enumerate(args['source_lengths'])]+[selected],
        'runtime_files':[ref(f'runtime_{i}.dll',size) for i,size in enumerate(args['runtime_lengths'])]}
    p={'schema':'dual-hash-diagnostic-protocol/v2','protocol_filename':'protocol.0004.json',
        'frozen_terminal_snapshot':{'arms':snapshots,'barrier_ref':ref('barrier.json',args['barrier_bytes'])},
        'campaign_refs':[ref(f'campaign_{i}.json',size) for i,size in enumerate(args['campaign_lengths'])],
        'prior_attempt':{'evidence_refs':[ref(f'history_{i}.json',size) for i,size in enumerate(args['historical_lengths'])],
            'status':'rejected','reported_passes':1,'reported_model_read_bytes':args['target_bytes'],'serial_reused_in_new_batch':False},
        'target':{'path':str(target),'bytes':args['target_bytes']},'chunk_bytes':h.MAX_CHUNK,
        'maximum_model_read_bytes':5*args['target_bytes'],'cumulative_model_read_upper_bound_bytes':6*args['target_bytes']}
    p['metadata_budget_plan']=d.planned_metadata_budget(p,manifest);p['maximum_metadata_read_bytes']=p['metadata_budget_plan']['declared_metadata_read_limit_bytes']
    monkeypatch.setattr(d,'HERE',tmp_path)
    monkeypatch.setattr(d,'document',lambda *a,**k:manifest)
    original=Path.stat
    def stat(path,*a,**k):
        value=sizes.get(str(path.absolute()))
        return SimpleNamespace(st_size=value) if value is not None else original(path,*a,**k)
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('precheck must not read model or prediction bodies'))
    return p,manifest_path,sizes


def test_preflight_accepts_closed_arithmetic_budget_then_retains_full_hash_requirement(tmp_path,monkeypatch):
    p,path,_=synthetic_preflight(tmp_path,monkeypatch);budget=d.ReadBudget();budget.used=12000000
    plan=d.configure_execution_budget(p,path,budget)
    assert budget.limit==plan['declared_metadata_read_limit_bytes']>2*1024**3
    assert budget.used==12000000 and plan['length_precheck_is_identity_evidence'] is False


@pytest.mark.parametrize('mutation',['old_limit','formula_count','size_drift','reuse_serial','cumulative'])
def test_preflight_rejects_inconsistent_budget_and_history_before_model_read(tmp_path,monkeypatch,mutation):
    p,path,sizes=synthetic_preflight(tmp_path,monkeypatch)
    if mutation=='old_limit':p['maximum_metadata_read_bytes']=2*1024**3
    if mutation=='formula_count':p['metadata_budget_plan']['terminal_check_count']=2
    if mutation=='size_drift':sizes[p['frozen_terminal_snapshot']['arms']['off']['prediction_refs']['0']['path']]+=1
    if mutation=='reuse_serial':p['prior_attempt']['serial_reused_in_new_batch']=True
    if mutation=='cumulative':p['cumulative_model_read_upper_bound_bytes']=5*p['target']['bytes']
    with pytest.raises(ValueError):d.configure_execution_budget(p,path,d.ReadBudget())


def test_receipt_caps_are_enforced_before_allocating_or_reading(tmp_path,monkeypatch):
    path=tmp_path/'oversized_receipt.json'
    monkeypatch.setattr(Path,'stat',lambda *a,**k:SimpleNamespace(st_size=d.PASS_RECEIPT_MAX_BYTES+1))
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('must reject before read'))
    with pytest.raises(ValueError):d.document(path,d.ReadBudget(10**12),max_bytes=d.PASS_RECEIPT_MAX_BYTES)


def test_budget_revision_never_overwrites_or_relabels_prior_attempt_source():
    source=(Path(d.__file__).parent/'prepare.py').read_text(encoding='utf8')
    assert "protocol_path=HERE/'protocol.0004.json'" in source
    assert "manifest_path=HERE/'preparation_manifest.0004.json'" in source
    assert "run_directory='read_comparison_0002'" in source
    assert "serial_reused_in_new_batch':False" in source
    assert "old_failure_reclassified':False" in source
