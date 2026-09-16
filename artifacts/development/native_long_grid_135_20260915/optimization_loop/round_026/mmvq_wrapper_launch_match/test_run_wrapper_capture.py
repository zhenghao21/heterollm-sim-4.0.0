"""Host-only failure tests. subprocess.run is always mocked before execute_attempt."""
from pathlib import Path
import hashlib, importlib.util, json, struct
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('wrapper_runtime_preparation',P/'run_wrapper_capture.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def rec(pid,name,*args):return {'pid':pid,'name':name,'cmdline':list(args)}

@pytest.mark.parametrize('args',[['freeze_candidate.py','freeze'],['verify_frozen_preflight.py'],['run_candidate.py','full'],
    ['predict_stable_native_dataset.py','--resume'],['run_wrapper_capture.py','--execute']])
def test_active_simulation_freeze_and_capture_rejected(args):
    assert r.process_conflicts([rec(300,'python.exe',*args)],100)

def test_desktop_GPU_and_own_wrapper_are_not_blockers():
    assert not r.process_conflicts([rec(100,'python.exe','run_wrapper_capture.py','--execute'),rec(200,'dwm.exe'),
        rec(201,'powershell.exe','python run_wrapper_capture.py --execute')],100)

def test_uninspectable_python_is_rejected():
    assert r.process_conflicts([{'pid':200,'name':'python.exe','cmdline':None}],100)

def prepare_mock_attempt(monkeypatch,tmp_path):
    prepared={'sealed':True};closure={'closed':True}
    monkeypatch.setattr(r,'P',tmp_path)
    monkeypatch.setattr(r,'environment_for',lambda _:dict(GGML_CUDA_DISABLE_GRAPHS='1',CAPTURE_CUPTI_DLL='mock_host_only',CAPTURE_RUNTIME_AUTHORIZED='1'))
    monkeypatch.setattr(r,'assert_idle',lambda:None)
    monkeypatch.setattr(r,'campaign_closed',lambda:closure)
    monkeypatch.setattr(r,'verify_prepared',lambda:(None,None,prepared))
    manifest={'executable':{'path':'HOST_ONLY_MOCK_NEVER_RUN'}}
    return tmp_path/'wrapper_capture_run.0001',prepared,{},manifest,closure

def test_start_failure_keeps_rejected_terminal_and_no_retry_overwrite(monkeypatch,tmp_path):
    args=prepare_mock_attempt(monkeypatch,tmp_path)
    def fail(*a,**k):raise OSError('synthetic host-only start failure')
    monkeypatch.setattr(r.subprocess,'run',fail)
    with pytest.raises(ValueError,match='terminal evidence retained'):r.execute_attempt(*args)
    finish=json.loads((args[0]/'finish.json').read_text(encoding='utf8'))
    assert finish['status']=='rejected' and finish['returncode'] is None and 'start failure' in finish['execution_error']
    initial=(args[0]/'finish.json').read_bytes()
    with pytest.raises(FileExistsError):r.execute_attempt(*args)
    assert (args[0]/'finish.json').read_bytes()==initial

def test_live_process_gate_after_preparation_keeps_terminal(monkeypatch,tmp_path):
    args=prepare_mock_attempt(monkeypatch,tmp_path)
    monkeypatch.setattr(r,'assert_idle',lambda:(_ for _ in ()).throw(ValueError('active simulation')))
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**k:pytest.fail('MUST NOT LAUNCH'))
    with pytest.raises(ValueError,match='terminal evidence retained'):r.execute_attempt(*args)
    finish=json.loads((args[0]/'finish.json').read_text(encoding='utf8'))
    assert finish['status']=='rejected' and finish['identity_unchanged'] is False and 'active simulation' in finish['execution_error']

def test_invalid_raw_json_keeps_terminal(monkeypatch,tmp_path):
    args=prepare_mock_attempt(monkeypatch,tmp_path)
    def fake(argv,**kw):
        Path(argv[-1]).write_text('not JSON',encoding='utf8')
        return type('Exit',(),{'returncode':0})()
    monkeypatch.setattr(r.subprocess,'run',fake)
    with pytest.raises(ValueError,match='terminal evidence retained'):r.execute_attempt(*args)
    finish=json.loads((args[0]/'finish.json').read_text(encoding='utf8'))
    assert finish['status']=='rejected' and 'JSONDecodeError' in finish['execution_error'] and finish['launches_ref'] is not None

def raw_numeric(tmp_path,first=0.25):
    data={'actual_q8':bytes(4608),'expected_q8':bytes(4608),'actual_output':struct.pack('<3072f',first,*([0.0]*3071)),
        'reference_output':bytes(3072*8),'bounds':struct.pack('<3072d',*([1.0]*3072))}
    refs={}
    for name,blob in data.items():
        path=tmp_path/(name+'.bin');path.write_bytes(blob);refs[name]=r.build.ref(path)
    return {'numerical_check':{'raw_refs':refs,'max_absolute_error':first,'max_bound_ratio':first}}

def test_new_numeric_raw_files_independently_recomputed(tmp_path):
    raw=raw_numeric(tmp_path);value=r.compare.verify_numerical_files(raw,r.check_ref)
    assert value['output_values']==3072 and value['max_absolute_error']==0.25

@pytest.mark.parametrize('damage',['byte','bound','summary','SHA'])
def test_numeric_raw_disagreement_is_rejected(tmp_path,damage):
    raw=raw_numeric(tmp_path);numeric=raw['numerical_check'];refs=numeric['raw_refs']
    if damage=='summary':numeric['max_bound_ratio']=0.125
    else:
        key='actual_q8' if damage in ('byte','SHA') else 'bounds';path=Path(refs[key]['path'])
        path.write_bytes(bytes([1])+bytes(4607) if key=='actual_q8' else struct.pack('<3072d',*([0.125]*3072)))
        if damage!='SHA':refs[key]=r.build.ref(path)
    with pytest.raises(ValueError):r.compare.verify_numerical_files(raw,r.check_ref)
