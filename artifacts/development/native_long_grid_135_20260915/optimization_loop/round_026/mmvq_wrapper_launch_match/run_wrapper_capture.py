"""Prepare/check without GPU; root may explicitly execute after the R27 terminal barrier.
This is an untimed synthetic qualification, never a latency calibration.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, os, subprocess
import psutil

P=Path(__file__).resolve().parent
LOOP=P.parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
build=load('wrapper_launch_builder',P/'build.py')
compare=load('wrapper_launch_comparator',P/'compare_records.py')
old=load('frozen_target_capture_driver',P.parent/'run_target_capture_ex.py')
need=build.need

def now():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):build.write_new(path,value)
def check_ref(expected):
    need(isinstance(expected,dict) and set(expected)=={'path','sha256','bytes'},'exact full reference required')
    build.check_ref(expected)

def process_conflicts(records,own_pid=None):
    records=[r for r in records if r.get('pid')!=own_pid]
    found=old.process_conflicts(records)
    found_pids={r['pid'] for r in found}
    additional=('freeze_candidate.py','verify_frozen_preflight.py','run_wrapper_capture.py',
        'run_target_capture_ex.py','run_wrapper_correctness.py','run_wrapper_correctness_v2.py')
    for row in records:
        cmd=' '.join(row.get('cmdline') or []).replace('\\','/').lower()
        if (row.get('name') or '').lower().startswith('python') and any(token in cmd for token in additional) and row['pid'] not in found_pids:
            found.append({'pid':row['pid'],'name':row.get('name'),'cmdline':row.get('cmdline'),'freeze_or_other_capture':True})
    return found

def assert_idle():
    found=process_conflicts([p.info for p in psutil.process_iter(['pid','name','cmdline'])],os.getpid())
    need(not found,'simulation/native/freeze process still active or uninspectable: '+json.dumps(found))

def campaign_closed():
    # A missing terminal file is not an idle/finished campaign. Validate the full
    # R27 terminal graph using its own frozen verifier, without running/scoring it.
    path=LOOP/'round_027/run_candidate.py'
    need((path.parent/'predictions_complete.json').is_file(),'R27 full262 terminal barrier not present')
    module=load('r27_terminal_read_only_guard',path)
    record,ref=module.barrier()
    need(record.get('terminal_count')==262 and record.get('failures_preserved') is True,'R27 terminal boundary differs')
    assert_idle()
    return {'driver_ref':build.ref(path),'barrier_ref':ref,'controls_ref':record['controls_ref']}

def verify_prepared():
    manifest=build.verify_build()
    test_path=P/'host_test_receipt.json'
    receipt=json.loads(test_path.read_text(encoding='utf8'))
    need(receipt.get('build_manifest_ref')==build.ref(P/'build_manifest.json'),'host receipt covers another build')
    need(receipt.get('GPU_executed') is False and receipt.get('inputs_unchanged') is True,'host receipt scope differs')
    need(type(receipt.get('layout_returncode')) is int and receipt['layout_returncode']==0,'layout tests not passed')
    need(type(receipt.get('comparison_returncode')) is int and receipt['comparison_returncode']==0,'comparison tests not passed')
    for key in ('build_manifest_ref','stdout_ref','stderr_ref','comparison_log_ref','shared_recorder_63_test_ref'):check_ref(receipt[key])
    layout=json.loads(Path(receipt['stdout_ref']['path']).read_text(encoding='utf8'))
    need(layout==receipt['layout_result'] and layout.get('status')=='passed' and layout.get('GPU_executed') is False,'layout raw receipt differs')
    source=json.loads((P/'source_manifest.json').read_text(encoding='utf8'))
    check_ref(source['target_run'])
    target=json.loads(Path(source['target_run']['path']).read_text(encoding='utf8'))
    compare.validate(target,'target')
    return manifest,source,{'build_ref':build.ref(P/'build_manifest.json'),'host_test_ref':build.ref(test_path),'target_ref':source['target_run']}

def environment_for(source):
    keep={'systemroot','windir','comspec','temp','tmp','localappdata','appdata','userprofile','homedrive','homepath','programdata','programfiles','programfiles(x86)'}
    env={key:value for key,value in os.environ.items() if key.lower() in keep}
    system_root=os.environ.get('SystemRoot','C:/Windows');env['SystemRoot']=system_root
    contract=json.loads((P.parent/'mmvq_target_capture_ex/source_contract.json').read_text(encoding='utf8'))
    cupti=contract['CUPTI_runtime'];check_ref(cupti)
    dirs=sorted({str(Path(ref['path']).parent) for ref in source['expected_runtime_dlls'].values()})
    env['PATH']=os.pathsep.join(dirs+[str(Path(system_root)/'System32'),system_root])
    env['GGML_CUDA_DISABLE_GRAPHS']='1'
    env['CAPTURE_CUPTI_DLL']=cupti['path'];env['CAPTURE_RUNTIME_AUTHORIZED']='1'
    return env

def collect_evidence(output,finish):
    for key,name in (('stdout_ref','stdout.log'),('stderr_ref','stderr.log'),('launches_ref','launches.json'),('comparison_ref','comparison.json')):
        path=output/name
        finish[key]=build.ref(path) if path.is_file() else None
    finish['retained_numeric_files']=[build.ref(path) for path in sorted(output.glob('launches.json.*.bin'))]

def execute_attempt(output,prepared,source,manifest,closure):
    # A rejected launch, invalid raw record, changed identity or numerical failure
    # always leaves a terminal receipt. CREATE_NEW prevents retry-overwrite.
    output=Path(output).resolve()
    need(output.parent==P and output.name.startswith('wrapper_capture_run.'),'run directory must be a new child of the prepared wrapper directory')
    env=environment_for(source)
    output.mkdir(exist_ok=False)
    evidence=output/'launches.json'
    argv=[manifest['executable']['path'],'--run-recorder-only',str(evidence)]
    start={'schema':'heterollm.mmvq-wrapper-launch-start/v1','created_utc':now(),'driver_ref':build.ref(__file__),
        'prepared_refs':prepared,'prior_campaign':closure,'argv':argv,
        'environment_policy':'OS allowlist; locked DLL directories; no inherited injection/FORCE flags',
        'cuda_environment':{key:env[key] for key in ('GGML_CUDA_DISABLE_GRAPHS','CAPTURE_CUPTI_DLL','CAPTURE_RUNTIME_AUTHORIZED')},
        'new_native_LLM':False,'timing_calibration_allowed':False,'performance_parameter_admitted':False}
    write_new(output/'start.json',start)
    finish={'schema':'heterollm.mmvq-wrapper-launch-finish/v1','status':'rejected','created_utc':None,
        'returncode':None,'start_ref':build.ref(output/'start.json'),'performance_parameter_admitted':False,
        'timing_equivalence_verified':False,'new_native_LLM':False}
    try:
        assert_idle()
        need(campaign_closed()==closure,'campaign state changed before wrapper launch')
        need(verify_prepared()[2]==prepared,'prepared identity changed before wrapper launch')
        assert_idle()
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            process=subprocess.run(argv,env=env,cwd=P,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        finish['returncode']=process.returncode
        need(process.returncode==0,'wrapper exited with failure '+str(process.returncode))
        raw=json.loads(evidence.read_text(encoding='utf8'))
        target=json.loads(Path(source['target_run']['path']).read_text(encoding='utf8'))
        matched=compare.compare(target,raw)
        matched['independent_raw_numerical_recheck']=compare.verify_numerical_files(raw,check_ref)
        matched['numerical_correctness_verified_by_this_comparison']=True
        matched['target_ref']=source['target_run'];matched['wrapper_ref']=build.ref(evidence)
        write_new(output/'comparison.json',matched)
        finish['launch_pair_matches']=True;finish['new_shim_numerically_qualified']=True
    except Exception as error:
        finish['execution_error']=type(error).__name__+': '+str(error)
    try:
        need(verify_prepared()[2]==prepared,'prepared inputs changed after execution')
        need(campaign_closed()==closure,'campaign inputs/terminal record changed after execution')
        assert_idle();finish['identity_unchanged']=True
    except Exception as error:
        finish['identity_unchanged']=False;finish['identity_error']=type(error).__name__+': '+str(error)
    if finish.get('launch_pair_matches') is True and finish.get('new_shim_numerically_qualified') is True and finish['identity_unchanged']:
        finish['status']='fixed_shape_effective_launch_and_numerical_match'
    collect_evidence(output,finish);finish['created_utc']=now();write_new(output/'finish.json',finish)
    need(finish['status']=='fixed_shape_effective_launch_and_numerical_match','qualification rejected; terminal evidence retained at '+str(output/'finish.json'))
    return build.ref(output/'finish.json')

def execute(output):
    manifest,source,prepared=verify_prepared();assert_idle();closure=campaign_closed()
    return execute_attempt(output,prepared,source,manifest,closure)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=P/'wrapper_capture_run.0001')
    args=parser.parse_args()
    if args.check:
        verify_prepared();print(json.dumps({'status':'prepared_identity_verified','GPU_executed':False,'runtime_match_verified':False,'requires_R27_full262_terminal_and_idle_processes':True}))
    else:print(json.dumps({'finish_ref':execute(args.output),'performance_parameter_admitted':False}))
