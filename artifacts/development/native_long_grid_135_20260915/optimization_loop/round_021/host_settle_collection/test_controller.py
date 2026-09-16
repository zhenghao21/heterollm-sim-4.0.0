"""Python-only controller mocks. Never executes a probe, profiler, or clock command."""
import copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
P=Path(__file__).parent;sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('collect_under_test',P/'collect.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def protocol():return c.load(P.parent/'host_settle_probe_r2/protocol.json')
def manifest():return c.load(P.parent/'host_settle_probe_r2/build_manifest.json')
def test_plan_has_exact_domain_and_sequence():
    plan=c.plan(protocol(),manifest());assert len(plan)==18
    assert sum(x['mode']!='export' for x in plan)==12 and sum(x['mode']=='export' for x in plan)==6
    assert {x['arm'] for x in plan}=={'buffered'}
    expected=[('short','direct'),('short','profile'),('short','export'),('settled','profile'),('settled','direct'),('settled','export')]
    assert [(x['condition'],x['mode']) for x in plan[:6]]==expected
    assert [(x['condition'],x['mode']) for x in plan[6:12]]==[('settled','direct'),('settled','profile'),('settled','export'),('short','profile'),('short','direct'),('short','export')]
    for x in plan:
        ms=0 if x['condition']=='short' else 1000
        assert x['settle_ms']==ms and x['app_argv'][x['app_argv'].index('--settle-ms')+1]==str(ms)
        assert x['condition'] in Path(x['directory']).parts and x['pair_id'] in x['app_argv']
        assert 'host_settle_probe_r2' in x['app_argv'][0]
        if x['mode']=='profile':assert '--kill=false' in x['argv']

@pytest.mark.parametrize('field',['manifest_conditions','pilot_conditions','summary_conditions','executable','pilot_config','pairs','modes','formal'])
def test_domain_mutations_rejected(field):
    p=protocol();m=manifest()
    if field=='manifest_conditions':m['conditions']=[]
    elif field=='pilot_conditions':p['execution']['pilot']['conditions'][0]['settle_ms']=1
    elif field=='summary_conditions':p['execution']['conditions']=[]
    elif field=='executable':m['executable']['sha256']='0'*64
    elif field=='pilot_config':p['execution']['pilot']['config']='another'
    elif field=='pairs':p['execution']['pilot']['pairs_per_arm']=4
    elif field=='modes':p['execution']['pilot']['modes']=['direct']
    elif field=='formal':p['execution']['formal']=29
    with pytest.raises(ValueError):c.plan(p,m)

def test_wait_same_preserves_natural_exit_after_interrupt(monkeypatch):
    monkeypatch.setattr(c.time,'sleep',lambda _:None)
    class Proc:
        n=0
        def wait(self):
            self.n+=1
            if self.n==1:raise KeyboardInterrupt()
            return 7
    assert c.wait_same(Proc())==7

def frozen_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'HERE',tmp_path);m=manifest();p=protocol()
    mp=tmp_path/'manifest.json';pp=tmp_path/'protocol.json';proof=tmp_path/'runtime.json'
    c.write_new(pp,p);m['protocol']=c.ref(pp);c.write_new(mp,m)
    extras=[{'path':str(tmp_path/f'extra{i}'),'bytes':1,'sha256':'0'*64} for i in range(5)]
    c.write_new(proof,{'extra_runtime_refs':extras})
    f={'manifest':c.ref(mp),'protocol':c.ref(pp),'runtime_proof':c.ref(proof),'collector_inputs':[c.ref(mp)],'extra_runtime_refs':extras,'native_and_tool_refs':m['files']+extras,'smi':{'path':'mock-smi'},'plan':c.plan(p,m),'expected_native_processes':12,'expected_export_processes':6}
    monkeypatch.setattr(c,'verify',lambda r:r)
    return f

@pytest.mark.parametrize('mutation',['argv','condition','missing','runtime','manifest_protocol'])
def test_freeze_rejects_semantic_forgery(tmp_path,monkeypatch,mutation):
    f=frozen_fixture(tmp_path,monkeypatch)
    if mutation=='argv':f['plan'][0]['argv'][-1]='other'
    elif mutation=='condition':f['plan'][0]['condition']='settled'
    elif mutation=='missing':f['plan'].pop()
    elif mutation=='runtime':f['extra_runtime_refs']=f['extra_runtime_refs'][:4]
    else:f['protocol']={**f['protocol'],'sha256':'f'*64}
    c.write_new(tmp_path/'execution_freeze.json',f)
    with pytest.raises(ValueError):c.verified_freeze(c.ref(tmp_path/'execution_freeze.json')['sha256'])

def test_freeze_accepts_exact_canonical_plan(tmp_path,monkeypatch):
    f=frozen_fixture(tmp_path,monkeypatch);c.write_new(tmp_path/'execution_freeze.json',f)
    assert c.verified_freeze(c.ref(tmp_path/'execution_freeze.json')['sha256'])['plan']==f['plan']

def mock_controller(tmp_path,monkeypatch,reset_fail=False,stage_fail=False):
    f=frozen_fixture(tmp_path,monkeypatch);monkeypatch.setattr(c,'verified_freeze',lambda _:f);monkeypatch.setattr(c.time,'sleep',lambda _:None)
    commands=[];stages=[];reviews=[]
    class Proc:
        pid=123
        def __init__(self,args,stdout,stderr,**kwargs):
            commands.append(args);self.args=args
            if args[1].startswith('--query-gpu='):
                gpu=protocol()['runtime']['gpu_expected'];stdout.write(f"{gpu['uuid']},{gpu['driver']},2400".encode())
        def wait(self):return 1 if reset_fail and '-rgc' in self.args else 0
    monkeypatch.setattr(c.subprocess,'Popen',Proc)
    def stage(*args):
        stages.append(args[0])
        if stage_fail:raise RuntimeError('mock stage failure')
    def summary(sha,partial=False):
        if partial:
            reviews.append(len(stages));c.write_new(tmp_path/'continue.json',{'freeze_sha256':sha,'first_pair_reviewed':True})
        return {'mock_summary':True}
    monkeypatch.setattr(c,'stage_run',stage);monkeypatch.setattr(c,'summarize_pairs',summary)
    return commands,stages,reviews

def test_review_after_six_stages_both_conditions(tmp_path,monkeypatch):
    commands,stages,reviews=mock_controller(tmp_path,monkeypatch)
    assert c.run('a'*64)==0 and reviews==[6] and len(stages)==18
    assert {x['condition'] for x in stages[:6]}=={'short','settled'}
    assert commands[-1]==['mock-smi','-rgc']

@pytest.mark.parametrize('reset_fail,stage_fail',[(True,False),(False,True),(True,True)])
def test_failures_keep_full_denominator_and_reset(tmp_path,monkeypatch,reset_fail,stage_fail):
    commands,stages,reviews=mock_controller(tmp_path,monkeypatch,reset_fail,stage_fail)
    assert c.run('a'*64)==1
    result=c.load(tmp_path/'controller_result.json')
    assert len(result['stage_status'])==18 and result['expected_stages']==18 and commands[-1]==['mock-smi','-rgc']
    assert result['status']!='completed'

def test_launch_receipt_failure_still_waits_owned_process(tmp_path,monkeypatch):
    f=frozen_fixture(tmp_path,monkeypatch);monkeypatch.setattr(c,'verified_freeze',lambda _:f)
    stage=f['plan'][2];waited=[]
    class Proc:
        pid=12
        def __init__(self,*a,**kw):pass
        def wait(self):waited.append(True);return 0
    monkeypatch.setattr(c.subprocess,'Popen',Proc);real=c.write_new
    def write(path,doc):
        if Path(path).name=='launched.json':raise OSError('mock bookkeeping failure')
        return real(path,doc)
    monkeypatch.setattr(c,'write_new',write)
    with pytest.raises(RuntimeError,match='stage failed'):c.stage_run(stage,'a'*64,manifest(),protocol(),{})
    receipt=c.load(Path(stage['directory'])/'complete.json')
    assert waited==[True] and receipt['child_exited'] is True and receipt['status']=='failed'

def test_telemetry_cleanup_failure_is_durable(tmp_path,monkeypatch):
    f=frozen_fixture(tmp_path,monkeypatch);monkeypatch.setattr(c,'verified_freeze',lambda _:f);monkeypatch.setattr(c.time,'sleep',lambda _:None)
    stage=f['plan'][0]
    class Timer:
        def close(self):pass
    class Session:
        def __init__(self,**kw):
            self.ready=SimpleNamespace(wait=lambda _:True);self.done=SimpleNamespace(is_set=lambda:False);self.thread=SimpleNamespace(is_alive=lambda:False);self.errors=[];self.result=None
        def start(self):return self
        def stop(self):pass
        def close(self):raise OSError('mock close failure')
    class Proc:
        pid=12
        def __init__(self,*a,**kw):pass
        def wait(self):return 0
    monkeypatch.setattr(c,'Win32DeadlineTimer',Timer);monkeypatch.setattr(c,'SamplerSession',Session);monkeypatch.setattr(c.subprocess,'Popen',Proc)
    with pytest.raises(RuntimeError,match='stage failed'):c.stage_run(stage,'a'*64,manifest(),protocol(),{})
    result=c.load(Path(stage['directory'])/'complete.json')
    assert result['status']=='failed_cleanup' and result['child_exited'] and 'mock close failure' in result['cleanup_errors'][0]

def test_no_forced_child_termination():
    text=(P/'collect.py').read_text();assert '.kill(' not in text and '.terminate(' not in text


def test_postexit_freeze_failure_is_preserved(tmp_path,monkeypatch):
    f=frozen_fixture(tmp_path,monkeypatch);stage=f['plan'][2];checks=[]
    def verify_freeze(_):
        checks.append(True)
        if len(checks)>1:raise ValueError('mock frozen input changed')
        return f
    class Proc:
        pid=12
        def __init__(self,*a,**kw):pass
        def wait(self):return 0
    monkeypatch.setattr(c,'verified_freeze',verify_freeze);monkeypatch.setattr(c.subprocess,'Popen',Proc)
    with pytest.raises(RuntimeError,match='stage failed'):c.stage_run(stage,'a'*64,manifest(),protocol(),{})
    result=c.load(Path(stage['directory'])/'complete.json')
    assert result['status']=='failed_identity' and result['child_exited'] and 'frozen input changed' in result['error']

def test_reset_exception_keeps_controller_result(tmp_path,monkeypatch):
    commands,stages,reviews=mock_controller(tmp_path,monkeypatch)
    proc=c.subprocess.Popen
    def launch(args,**kw):
        if '-rgc' in args:raise OSError('mock reset launch failure')
        return proc(args,**kw)
    monkeypatch.setattr(c.subprocess,'Popen',launch)
    assert c.run('a'*64)==1
    result=c.load(tmp_path/'controller_result.json')
    assert result['status']=='failed_cleanup' and 'mock reset launch failure' in result['reset_error'] and len(result['stage_status'])==18

def test_unreadable_receipt_keeps_failed_denominator(tmp_path,monkeypatch):
    f=frozen_fixture(tmp_path,monkeypatch);stage=f['plan'][0];d=Path(stage['directory']);d.mkdir(parents=True);(d/'complete.json').write_text('{')
    result=c.stage_status(stage)
    assert result['process_status']=='unreadable_receipt' and result['condition']=='short'
