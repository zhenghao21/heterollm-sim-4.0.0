import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('r26_gate_test',Path(__file__).with_name('run_target_capture_ex.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def test_live_process_filter_catches_both_arms_and_native_but_not_reader():
    rows=[{'pid':1,'name':'python.exe','cmdline':['python','F:/x/round_025/on/source/tools/predict_stable_native_dataset.py','--worker-cell']},
          {'pid':2,'name':'llama-server.exe','cmdline':[]},
          {'pid':3,'name':'python.exe','cmdline':['python','F:/x/round_026/run_target_capture.py','--check']}]
    assert [x['pid'] for x in d.process_conflicts(rows)]==[1,2]

def test_execute_is_blocked_before_build_or_launch(monkeypatch):
    called=[]
    def reject():raise ValueError('campaign active')
    monkeypatch.setattr(d,'campaign_finished',reject)
    monkeypatch.setattr(d,'verify_build',lambda:called.append('build'))
    monkeypatch.setattr(d.subprocess,'run',lambda *a,**k:called.append('launch'))
    with pytest.raises(ValueError,match='campaign active'):d.execute()
    assert called==[]

def test_absent_barrier_is_not_read_as_stopped_process(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'CLOSED_CAMPAIGN',tmp_path/'missing.json')
    with pytest.raises(ValueError,match='closure not yet present'):d.campaign_finished()


def test_additional_actual_process_classes_are_blocked():
    rows=[{'pid':n,'name':name,'cmdline':cmd} for n,(name,cmd) in enumerate([
        ('llama-cli.exe',[]),('mmvq-target-recorder.0004.exe',[]),
        ('python.exe',['python','F:/other/predict_stable_native_dataset.py','--worker-cell']),
        ('host-settle-probe.exe',['--run'])])]
    assert len(d.process_conflicts(rows))==4

def test_environment_drops_injection_force_and_unrelated_secrets(monkeypatch):
    for key in ('CUDA_INJECTION64_PATH','NVTX_INJECTION64_PATH','GGML_CUDA_FORCE_MMQ','UNRELATED_API_KEY'):
        monkeypatch.setenv(key,'must-not-pass')
    env,_=d.environment_for({'target_modules':[{'path':'F:/locked/ggml-cuda.dll'}],'CUPTI_runtime':{'path':'F:/locked/cupti.dll'}})
    assert not any(key in env for key in ('CUDA_INJECTION64_PATH','NVTX_INJECTION64_PATH','GGML_CUDA_FORCE_MMQ','UNRELATED_API_KEY'))
    assert env['CAPTURE_RUNTIME_AUTHORIZED']=='1'

def test_startup_error_preserves_failed_terminal(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'P',tmp_path)
    monkeypatch.setattr(d,'campaign_finished',lambda:{'stable':'receipt'})
    manifest={'target_executable':{'path':'missing.exe'}}
    contract={'target_modules':[{'path':'F:/locked/ggml-cuda.dll'}],'CUPTI_runtime':{'path':'F:/locked/cupti.dll'}}
    monkeypatch.setattr(d,'verify_build',lambda:(manifest,contract))
    monkeypatch.setattr(d,'assert_idle',lambda:None)
    def reject(*args,**kwargs):raise OSError('cannot start')
    monkeypatch.setattr(d.subprocess,'run',reject)
    with pytest.raises(ValueError,match='saved failure terminal'):d.execute()
    import json
    finish=json.loads((tmp_path/'target_capture_ex_run.0001/finish.json').read_text())
    assert finish['status']=='rejected' and finish['returncode'] is None
    assert 'cannot start' in finish['execution_error'] and finish['identity_unchanged']
    assert finish['performance_parameter_admitted'] is False


def test_existing_recorder_and_other_campaign_entrypoints_are_blocked():
    rows=[{'pid':n,'name':name,'cmdline':cmd} for n,(name,cmd) in enumerate([
        ('launch-recorder.attempt0004.exe',['--run-recorder-only','out.json']),
        ('python.exe',['python','evaluate_identity_repair.py','full']),
        ('python.exe',['python','-m','heterollm_sim','run']),
        ('test-backend-ops.exe',['test']),('other.exe',['other.exe','--run-correctness'])])]
    assert len(d.process_conflicts(rows))==5
