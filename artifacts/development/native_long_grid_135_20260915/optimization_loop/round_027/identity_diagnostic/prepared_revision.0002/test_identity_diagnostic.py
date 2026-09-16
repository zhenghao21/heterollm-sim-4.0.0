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
    protocol={'target':{'bytes':99},'passes':['serial_01','parallel_01','parallel_02','parallel_03','parallel_04']}
    monkeypatch.setattr(d,'verify_prepared',lambda *a:(protocol,{'sha256':'frozen'}))
    monkeypatch.setattr(h,'self_test',lambda:{'known_vectors_passed':2})
    def reject(*a):raise ValueError('R27 still active')
    monkeypatch.setattr(d,'terminal_gate',reject if gate=='terminal' else lambda *a:{'barrier_ref':{},'terminal_count':262})
    monkeypatch.setattr(d,'idle_snapshot',reject if gate=='idle' else lambda:{})
    monkeypatch.setattr(d,'worker',lambda *a:pytest.fail('must not read target'))
    with pytest.raises(RuntimeError):d.execute(tmp_path/'manifest.json')
    r=json.loads((tmp_path/'runs/read_comparison_0001/finish.json').read_text())
    assert r['status']=='rejected' and r['model_read_upper_bound_bytes']==0 and r['actual_model_read_bytes_reported']==0


@pytest.mark.parametrize('bad_label',[None,'serial_01','parallel_03'])
def test_fixed_serial_then_four_mock_workers_never_retries(tmp_path,monkeypatch,bad_label):
    monkeypatch.setattr(d,'HERE',tmp_path)
    labels=['serial_01','parallel_01','parallel_02','parallel_03','parallel_04']
    protocol={'target':{'bytes':3},'passes':labels}
    monkeypatch.setattr(d,'verify_prepared',lambda *a:(protocol,{'sha256':'fixed'}))
    monkeypatch.setattr(d,'terminal_gate',lambda *a:{'barrier_ref':{'sha256':'terminal'},'terminal_count':262})
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
    monkeypatch.setattr(d,'compare_chunks',lambda *a:{'full_pass_consistency':True,'compared_chunks':1,'differences':[]})
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
    r=json.loads((tmp_path/'runs/read_comparison_0001/finish.json').read_text())
    assert calls==labels and started==labels[1:] and joined==labels[1:]
    assert r['model_read_upper_bound_bytes']==15 and r['actual_model_read_bytes_reported']==15
    assert r['repairs_or_retries']==0 and r['old_failures_modified'] is False
    if bad_label:assert r['status']=='rejected'
    with pytest.raises(FileExistsError):d.execute(tmp_path/'manifest.json')



def test_prepared_runtime_paths_are_case_insensitive_but_location_sensitive(monkeypatch,tmp_path):
    protocol={'passes':['serial_01','parallel_01','parallel_02','parallel_03','parallel_04'],
              'parallel_workers':4,'target':{'bytes':3},'maximum_model_read_bytes':15,'chunk_bytes':h.MAX_CHUNK}
    m={'schema':'dual-hash-preparation/v1','files':[],
       'runtime_files':[{'path':r'C:\Windows\System32\bcrypt.dll'}],'crypto_environment':{}}
    monkeypatch.setattr(d,'document',lambda path,budget:protocol if Path(path).name=='protocol.json' else m)
    monkeypatch.setattr(d,'check_ref',lambda *a:None)
    monkeypatch.setattr(d,'ref',lambda *a:{'sha256':'fixed'})
    monkeypatch.setattr(d,'crypto_environment',lambda:{})
    monkeypatch.setattr(d,'runtime_files',lambda:[Path(r'C:\WINDOWS\System32\bcrypt.dll')])
    assert d.verify_prepared(tmp_path/'manifest.json',d.ReadBudget())[0]==protocol
    monkeypatch.setattr(d,'runtime_files',lambda:[Path(r'C:\elsewhere\bcrypt.dll')])
    with pytest.raises(ValueError,match='runtime path'):d.verify_prepared(tmp_path/'manifest.json',d.ReadBudget())
