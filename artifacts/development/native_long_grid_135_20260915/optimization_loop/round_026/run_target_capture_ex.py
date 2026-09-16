"""Bounded synthetic path qualification; no LLM, no timing calibration.
--check is read-only. --execute refuses an unfinished/live simulation campaign.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse, hashlib, importlib.util, json, os, subprocess, sys
import psutil
P=Path(__file__).resolve().parent
LOOP=P.parent
CAP=P/'mmvq_target_capture_ex'
MANIFEST=CAP/'build_manifest.json'
MANIFEST_SHA='ce9e6047bb248a62cad8ec4348ceeaeb3e6721b6328752cbd2a9032db0615215'
HOST_TEST=CAP/'host_test.0001.json'
HOST_TEST_SHA='efe82c8ed3df0c56ea39178b45aa0db091b8a5ca0e60b4695c9b4607b78ae649'
CLOSED_CAMPAIGN=LOOP/'round_025/execution_closed.json'
CLOSED_CAMPAIGN_SHA='c42fd307580b4473c111010d7282e31a3c6130018bac3c43d187402730f3fa20'

def now():return datetime.now(timezone.utc).isoformat()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
build=load('r26_target_builder',CAP/'build.py')
def require(value,message):
    if not value:raise ValueError(message)
def check_ref(reference):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256','bytes'},'exact reference required')
    require(build.ref(reference['path'])==reference,'referenced bytes changed: '+reference['path'])
    return reference

def process_conflicts(records):
    found=[]
    for item in records:
        name=(item.get('name') or '').lower()
        args=item.get('cmdline') or []
        cmd=' '.join(args).replace('\\','/').lower()
        native=(name.startswith(('llama-','mmvq-','mmvq_','launch-recorder')) or name in {'test-backend-ops.exe','nsys.exe','ncu.exe'} or 'microbench' in name or ('probe' in name and name.endswith('.exe')))
        simulation=any(token in cmd for token in ('predict_stable_native_dataset.py','run_candidate.py','native_grid_predict.py','evaluate_identity_repair.py','evaluate_candidate.py','native_llama_compare.py',' -m heterollm_sim','--run-recorder-only','--run-correctness'))
        unknown_python=name.startswith('python') and item.get('cmdline') is None
        if native or simulation or unknown_python:
            found.append({'pid':item['pid'],'name':item.get('name'),'cmdline':args,'unknown_commandline':unknown_python})
    return found

def assert_idle():
    conflicts=process_conflicts([p.info for p in psutil.process_iter(['pid','name','cmdline'])])
    require(not conflicts,'simulation/native process still live or not inspectable: '+json.dumps(conflicts))

def campaign_finished():
    # R25 is closed by a reviewed, immutable startup-rejection receipt. Missing
    # status/lock files never count as evidence that a process stopped.
    require(CLOSED_CAMPAIGN.is_file(),'reviewed campaign closure not yet present')
    require(build.ref(CLOSED_CAMPAIGN)['sha256']==CLOSED_CAMPAIGN_SHA,'unreviewed campaign closure')
    assert_idle()
    closure=json.loads(CLOSED_CAMPAIGN.read_text(encoding='utf-8'))
    require(closure.get('status')=='candidate_startup_rejected' and closure.get('scored') is False and closure.get('on_terminal_predictions')==0,'unexpected campaign terminal scope')
    runner=load('r26_prior_campaign_check',LOOP/'round_025/run_candidate.py')
    require(runner.guard()==closure['controls_ref'],'prior campaign controls differ')
    runner.no_scores()
    runner.s.load_bundle(LOOP/'round_025/off','retained',closure['off_terminal_inputs'],full=True)
    check_ref(closure['on_freeze_ref']);check_ref(closure['driver_ref'])
    require(not list((LOOP/'round_025/on/predictions').glob('*.prediction.json')),'closed candidate acquired predictions')
    return build.ref(CLOSED_CAMPAIGN)

def verify_build():
    require(build.ref(MANIFEST)['sha256']==MANIFEST_SHA,'unreviewed recorder build')
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    require(build.verify_inputs()==manifest['inputs'],'recorder source/toolchain closure changed')
    for ref in manifest['compile_headers']:check_ref(ref)
    for key in ('target_executable','host_test_executable'):check_ref(manifest[key])
    require(build.ref(HOST_TEST)['sha256']==HOST_TEST_SHA,'unreviewed host-test receipt')
    test=json.loads(HOST_TEST.read_text(encoding='utf-8'))
    check_ref(test['build_manifest_ref']);check_ref(test['stdout_ref']);check_ref(test['stderr_ref'])
    require(test['build_manifest_ref']==build.ref(MANIFEST),'host tests cover another build')
    require(test.get('argv')==[manifest['host_test_executable']['path']],'host test executed another program')
    require(test.get('target_executable_executed') is False,'host test scope executed target')
    raw_test=json.loads(Path(test['stdout_ref']['path']).read_text(encoding='utf-8'))
    require(test.get('result')==raw_test,'wrapped host result differs from raw output')
    require(raw_test.get('retained_legacy_test_count')==38,'legacy tests not retained')
    require(len(raw_test.get('tests',[]))==len(set(raw_test.get('tests',[])))==63,'host test list mismatch')
    for key in ('GPU_context_created','CUPTI_subscription_created','CUDA_API_called','actual_GPU_callback_compatibility_validated'):
        require(raw_test.get(key) is False,'host-only flag differs: '+key)
    for key in ('GPU_CUPTI_GGML_DLL_imports','explicit_DLL_loads'):
        require(type(raw_test.get(key)) is int and raw_test[key]==0,'host test nonzero or invalid load count: '+key)
    require(test.get('returncode')==0 and test.get('inputs_unchanged') is True and test['result'].get('test_count')==63 and test['result'].get('status')=='passed','host qualification missing')
    require(manifest['host_GPU_import_count']==0 and test.get('gpu_execution_performed') is False,'host-only qualification scope differs')
    contract=json.loads((CAP/'source_contract.json').read_text(encoding='utf-8'))
    return manifest,contract

def write_new(path,payload):
    with Path(path).open('x',encoding='utf-8') as out:json.dump(payload,out,indent=2,ensure_ascii=False,allow_nan=False);out.write('\n')

def environment_for(contract):
    # Explicit process environment: never inherit injection DLLs or FORCE flags.
    keep={'SystemRoot','WINDIR','COMSPEC','TEMP','TMP','LOCALAPPDATA','APPDATA','USERPROFILE',
          'HOMEDRIVE','HOMEPATH','ProgramData','ProgramFiles','ProgramFiles(x86)'}
    env={k:v for k,v in os.environ.items() if k.lower() in {x.lower() for x in keep}}
    system_root=os.environ.get('SystemRoot','C:/Windows')
    env['SystemRoot']=system_root
    runtime_dirs=sorted({str(Path(x['path']).parent) for x in contract['target_modules']})
    env['PATH']=os.pathsep.join(runtime_dirs+['E:/cuda/bin',str(Path(system_root)/'System32'),system_root])
    env['GGML_CUDA_DISABLE_GRAPHS']='1'
    env['CAPTURE_CUPTI_DLL']=contract['CUPTI_runtime']['path']
    env['CAPTURE_RUNTIME_AUTHORIZED']='1'  # Existing research authorization; only after all gates pass.
    return env,runtime_dirs

def execute():
    previous=campaign_finished()
    manifest,contract=verify_build()
    assert_idle()  # Recheck after the longer identity checks, just before launch preparation.
    output=P/'target_capture_ex_run.0001'
    output.mkdir(exist_ok=False)
    evidence=output/'launches.json'
    env,runtime_dirs=environment_for(contract)
    argv=[manifest['target_executable']['path'],'--run-recorder-only',str(evidence.resolve())]
    start={'schema':'r26-synthetic-target-capture-start/v1','created_utc':now(),'driver_ref':build.ref(__file__),
           'build_manifest_ref':build.ref(MANIFEST),'host_test_ref':build.ref(HOST_TEST),'prior_campaign_closure_ref':previous,
           'argv':argv,'cuda_environment':{k:env[k] for k in ('GGML_CUDA_DISABLE_GRAPHS','CAPTURE_CUPTI_DLL','CAPTURE_RUNTIME_AUTHORIZED')},
           'environment_policy':'explicit OS minimum; no inherited injection or forced kernel flags',
           'native_LLM_executed':False,'timing_calibration_allowed':False}
    write_new(output/'start.json',start)
    finish={'schema':'r26-synthetic-target-capture-finish/v1','created_utc':None,'returncode':None,
            'status':'rejected','start_ref':build.ref(output/'start.json'),'performance_parameter_admitted':False}
    try:
        assert_idle()
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            result=subprocess.run(argv,env=env,cwd=runtime_dirs[0],stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        finish['returncode']=result.returncode
        require(result.returncode==0,'recorder exit failure: '+str(result.returncode))
        require(evidence.is_file(),'raw launch record missing')
        raw=json.loads(evidence.read_text(encoding='utf-8'))
        require(raw.get('status')=='qualified_runtime_pair' and raw.get('timed') is False and raw.get('performance_parameter_admitted') is False,'runtime pair not qualified')
        finish['runtime_pair_qualified']=True
    except Exception as error:
        finish['execution_error']=type(error).__name__+': '+str(error)
    try:
        verify_build();require(campaign_finished()==previous,'prior campaign changed')
        finish['identity_unchanged']=True
    except Exception as error:
        finish['identity_unchanged']=False;finish['identity_error']=type(error).__name__+': '+str(error)
    if finish.get('runtime_pair_qualified') is True and finish['identity_unchanged']:
        finish['status']='synthetic_runtime_pair_qualified'
    finish['created_utc']=now()
    for key,file in [('stdout_ref','stdout.log'),('stderr_ref','stderr.log'),('launches_ref','launches.json')]:
        finish[key]=build.ref(output/file) if (output/file).is_file() else None
    write_new(output/'finish.json',finish)
    require(finish['status']=='synthetic_runtime_pair_qualified','runtime qualification failed; saved failure terminal retained')
    print(json.dumps({'status':finish['status'],'performance_parameter_admitted':False,'finish_ref':build.ref(output/'finish.json')}))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--check',action='store_true');group.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.check:
        verify_build();print(json.dumps({'status':'build_closure_verified','GPU_executed':False,'execution_requires_closed_campaign_and_idle_processes':True}))
    else:execute()
