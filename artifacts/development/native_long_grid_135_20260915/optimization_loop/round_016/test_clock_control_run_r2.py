"""Host-only controller transitions; no actual subprocess or clock changes."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('controller',Path(__file__).with_name('clock_control_run_r2.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_checkpoint_wait_same_exit_then_continue():
    events=[];responses=iter([{'status':'still_running','pid':7},{'status':'matrix_success'}])
    def invoke(limit):events.append(('invoke',limit));return next(responses)
    def wait(out):events.append(('wait',out['pid']));return {'status':'completed','child_process_exited':True}
    result=c.drive_matrix(invoke,wait,lambda:0,lambda:events.append(('verify',)))
    assert result['status']=='matrix_success'
    assert [e for e in events if e[0]!='verify']==[('invoke',None),('wait',7),('invoke',None)]


def test_first_pair_remaining_budget_after_checkpoint():
    count=[0];limits=[]
    def invoke(limit):
        limits.append(limit)
        if len(limits)==1:count[0]=1;return {'status':'still_running','pid':7}
        count[0]=3;return {'status':'bounded_stages_complete'}
    def wait(out):count[0]=2;return {'status':'completed','child_process_exited':True}
    assert c.drive_matrix(invoke,wait,lambda:count[0],lambda:None,first_pair_only=True)['status']=='first_pair_complete'
    assert limits==[3,1]


@pytest.mark.parametrize('status',['identity_terminal_stop','unresolved_missing_exit_receipt','unresolved_running','bounded_stages_complete'])
def test_nonterminal_or_identity_status_never_finishes_matrix(status):
    with pytest.raises(RuntimeError):c.drive_matrix(lambda _: {'status':status},lambda _:pytest.fail('unexpected wait'),lambda:0,lambda:None)


def test_failed_identity_after_checkpoint_never_reinvokes():
    calls=[]
    def invoke(_):calls.append(1);return {'status':'still_running'}
    with pytest.raises(RuntimeError,match='identity'):c.drive_matrix(invoke,lambda _:{'status':'failed_identity','child_process_exited':True},lambda:0,lambda:None)
    assert len(calls)==1


def test_unconfirmed_child_never_reinvokes():
    with pytest.raises(RuntimeError,match='exit'):c.drive_matrix(lambda _:{'status':'still_running'},lambda _:{'status':'completed','child_process_exited':False},lambda:0,lambda:None)


def test_failure_complete_matrix_is_not_success():
    r=c.drive_matrix(lambda _:{'status':'matrix_receipts_complete_with_failures'},lambda _:None,lambda:0,lambda:None)
    assert r['status']=='matrix_receipts_complete_with_failures'


def test_checkpoint_receipt_checked_after_same_supervisor(tmp_path,monkeypatch):
    directory=tmp_path/'stage';directory.mkdir()
    c.write(directory/'supervisor.json',{'pid':7});c.write(directory/'complete.json',{'unit':True})
    order=[];monkeypatch.setattr(c,'stage_directory',lambda _:directory)
    monkeypatch.setattr(c,'wait_pid',lambda pid:order.append(('wait',pid)))
    monkeypatch.setattr(c,'verify_completed_stage',lambda *args:order.append(('verify',)) or {'status':'completed'})
    assert c.wait_checkpoint({'status':'still_running','pid':7,'stage':{}},'sha',{})['status']=='completed'
    assert order==[('wait',7),('verify',)]


def test_missing_completion_waits_child_then_fails(tmp_path,monkeypatch):
    directory=tmp_path/'stage';directory.mkdir();c.write(directory/'supervisor.json',{'pid':7});c.write(directory/'launched.json',{'pid':9})
    waited=[];monkeypatch.setattr(c,'stage_directory',lambda _:directory);monkeypatch.setattr(c,'wait_pid',waited.append)
    with pytest.raises(RuntimeError,match='without durable'):c.wait_checkpoint({'status':'still_running','pid':7,'stage':{}},'sha',{})
    assert waited==[7,9]


def test_prepare_mode_never_captures_or_locks(monkeypatch):
    monkeypatch.setattr(c,'capture',lambda *a:pytest.fail('no subprocess allowed'))
    assert c.main(['--expected-freeze-sha256','not-yet-approved'])==0


def test_exclusive_output_keeps_prior_evidence(tmp_path):
    path=tmp_path/'x.json';c.write(path,{'first':True})
    with pytest.raises(FileExistsError):c.write(path,{'second':True})
    assert c.load(path)=={'first':True}
