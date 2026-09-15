"""Run only in the idle window explicitly selected by root; never invoked by build."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def ref(path):
    path=Path(path).resolve();b=path.read_bytes();return {'path':str(path),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idle-window-confirmed',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if not args.idle_window_confirmed:parser.error('root must explicitly choose an idle measurement window; nothing was executed')
    receipt_path=ROOT/'build/build_receipt.json'
    build=json.loads(receipt_path.read_text());assert build['status']=='complete'
    for name in ('source','protocol','build_script'):
        expected=build['inputs'][name];assert ref(expected['path'])['sha256']==expected['sha256'],name+' identity changed'
    binary=build['binary_ref'];assert ref(binary['path'])['sha256']==binary['sha256']
    for expected in build['supporting_file_refs']:
        assert ref(expected['path'])['sha256']==expected['sha256'],'supporting protocol/tool changed'
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=(args.output_dir or ROOT/'runs'/stamp).resolve()
    if not output.is_relative_to(ROOT/'runs'):raise RuntimeError('output directory must be a new directory under launch_probe/runs')
    output.mkdir(parents=True,exist_ok=False)
    tool=Path(os.environ.get('WINDIR','C:/Windows'))/'System32/nvidia-smi.exe'
    query=['--query-gpu=uuid,name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']
    def snapshot(name):
        result=subprocess.run([str(tool),*query],text=True,capture_output=True,encoding='utf-8',errors='replace')
        value={'command':[str(tool),*query],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
               'created_utc':datetime.now(timezone.utc).isoformat(),'changes_device_configuration':False}
        path=output/(name+'.json');write(path,value)
        return value,ref(path)
    before,before_ref=snapshot('device_before')
    if before['returncode']!=0:raise RuntimeError('fresh driver/GPU identity snapshot failed; benchmark was not started')
    device=protocol['device'];matches=[line.split(',') for line in before['stdout'].splitlines() if line.split(',')[0].strip()==device['required_gpu_uuid']]
    if len(matches)!=1 or matches[0][2].strip()!=device['recorded_driver_version']:
        raise RuntimeError('fresh UUID/driver differs from predeclared protocol; benchmark was not started')
    raw=output/'raw.jsonl';command=[binary['path'],'--idle-window-confirmed','--protocol',str(ROOT/'protocol.json'),'--source',str(ROOT/'launch_probe.cu'),'--output',str(raw)]
    with (output/'process.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    after,after_ref=snapshot('device_after')
    runtime={'schema':'cuda-launch-sync-run-receipt/v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'command':command,'returncode':result.returncode,'status':'complete' if result.returncode==0 else 'failed',
        'build_receipt_ref':ref(receipt_path),'source_ref':build['inputs']['source'],'protocol_ref':build['inputs']['protocol'],
        'binary_ref':binary,'device_before_ref':before_ref,'device_after_ref':after_ref,'process_log_ref':ref(output/'process.log'),
        'raw_ref':ref(raw) if raw.exists() else None,'explicit_idle_window_confirmation':True,
        'native_DLL_modified':False,'coefficient_or_profile_written':False}
    write(output/'run_receipt.json',runtime)
    if result.returncode==0:
        analysis=subprocess.run([sys.executable,str(ROOT/'analyze_results.py'),'--run-directory',str(output)])
        return analysis.returncode
    return result.returncode
if __name__=='__main__':sys.exit(main())
