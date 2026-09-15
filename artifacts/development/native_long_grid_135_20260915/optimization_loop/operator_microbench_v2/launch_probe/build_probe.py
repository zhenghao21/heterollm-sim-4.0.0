"""Build only. This script never executes launch_probe.exe or device probes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build'

def ref(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    BUILD.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    protocol_ref=ref(ROOT/'protocol.json');source_ref=ref(ROOT/'launch_probe.cu')
    declared=(ROOT/'protocol.sha256').read_text().split()[0]
    assert protocol_ref['sha256']==declared,'predeclared protocol changed'
    assert protocol['burst_lengths']==[1,4,16,64]
    assert protocol['kernels']==['empty','low_load_integer']
    assert protocol['supply_modes']==['continuous_enqueue','stream_sync_each']
    assert protocol['warmup_repeats_per_configuration']==10 and protocol['formal_repeats_per_configuration']==30
    assert protocol['kernel_geometry']=={'blocks':1,'threads':32,'low_load_integer_iterations':16,'low_load_seed':305419896,'output_words':2048,'untouched_word':2779096485}
    receipt_path=BUILD/'build_receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text())['status']=='complete':
        raise RuntimeError('completed build already exists; this immutable probe must not be rebuilt in place')
    attempts_path=BUILD/'build_attempts.jsonl'
    attempts=[json.loads(line) for line in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
    if len(attempts)>=2:raise RuntimeError('two-build budget exhausted; no further compilation attempted')
    cuda=Path(os.environ.get('CUDA_PATH','E:/cuda')).resolve()
    nvcc=cuda/'bin/nvcc.exe'
    vcvars=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
    assert nvcc.is_file() and vcvars.is_file(),'required existing CUDA/MSVC toolchain absent'
    env_script=BUILD/'collect_msvc_env.cmd'
    env_script.write_text('@echo off\r\ncall "'+str(vcvars)+'" >nul\r\nif errorlevel 1 exit /b 1\r\nset\r\n',encoding='utf-8')
    raw_env=subprocess.check_output([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',str(env_script)],text=True,encoding='utf-8',errors='replace')
    environment=dict(os.environ)
    for line in raw_env.splitlines():
        if '=' in line and not line.startswith('='):
            key,value=line.split('=',1);environment[key]=value
    cl=Path(environment['VCToolsInstallDir'])/'bin/Hostx64/x64/cl.exe'
    version=subprocess.check_output([str(nvcc),'--version'],env=environment,text=True,encoding='utf-8',errors='replace')
    (BUILD/'nvcc_version.txt').write_text(version,encoding='utf-8')
    refs={'source':source_ref,'protocol':protocol_ref,'build_script':ref(__file__),'nvcc':ref(nvcc),'host_compiler':ref(cl),
          'vcvars64':ref(vcvars),'cuda_header':ref(cuda/'include/cuda.h'),'cuda_runtime_header':ref(cuda/'include/cuda_runtime_api.h'),
          'cuda_static_runtime':ref(cuda/'lib/x64/cudart_static.lib'),'nvcc_version':ref(BUILD/'nvcc_version.txt')}
    header=['#pragma once']
    strings={'PROBE_SOURCE_SHA256':source_ref['sha256'],'PROBE_PROTOCOL_SHA256':protocol_ref['sha256'],
        'PROBE_NVCC_SHA256':refs['nvcc']['sha256'],'PROBE_CL_SHA256':refs['host_compiler']['sha256'],
        'PROBE_BUILD_SCRIPT_SHA256':refs['build_script']['sha256'],'PROBE_NVCC_VERSION':version.strip(),
        'PROBE_EXPECTED_GPU_UUID':protocol['device']['required_gpu_uuid']}
    for key,value in strings.items():header.append('#define '+key+' '+json.dumps(value,ensure_ascii=True))
    numbers={'PROBE_WARMUP_REPEATS':10,'PROBE_FORMAL_REPEATS':30,'PROBE_THREADS':32,'PROBE_INTEGER_ITERATIONS':16,
             'PROBE_OUTPUT_WORDS':2048,'PROBE_LOW_LOAD_SEED':305419896,'PROBE_QPC_CONTROL_PAIRS':256}
    for key,value in numbers.items():header.append('#define '+key+' '+str(value)+'u')
    (BUILD/'build_identity.h').write_text('\n'.join(header)+'\n',encoding='utf-8')
    refs['generated_identity_header']=ref(BUILD/'build_identity.h')
    exe=BUILD/'launch_probe.exe'
    command=[str(nvcc),'-std=c++17','-O3','-arch=sm_120','--cudart','static','-Xcompiler','/EHsc','-Xcompiler','/W3',
             '-I',str(BUILD),str(ROOT/'launch_probe.cu'),'-o',str(exe),'-Xlinker','bcrypt.lib']
    attempt={'attempt':len(attempts)+1,'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,
        'source_ref':source_ref,'protocol_ref':protocol_ref,'timed_execution':False,'budget':2}
    with attempts_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(attempt)+'\n')
    log_path=BUILD/'compile.log'
    with log_path.open('a',encoding='utf-8') as log:
        log.write('\nBUILD ATTEMPT '+str(attempt['attempt'])+'\n'+json.dumps(command)+'\n');log.flush()
        completed=subprocess.run(command,cwd=BUILD,env=environment,stdout=log,stderr=subprocess.STDOUT,text=True)
    receipt={**attempt,'status':'complete' if completed.returncode==0 else 'failed','returncode':completed.returncode,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'inputs':refs,'compile_log_ref':ref(log_path),
        'no_probe_executed':True,'no_native_dll_modified':True,'no_Nsight_or_CUPTI_dependency':True,
        'device_identity_binding':protocol['device'],'model_profile_update':False}
    if completed.returncode==0:
        receipt['binary_ref']=ref(exe)
        receipt['supporting_file_refs']=[ref(ROOT/name) for name in ('run_probe.py','analyze_results.py','README.md')]
    write(receipt_path,receipt)
    print(json.dumps({'status':receipt['status'],'attempt':attempt['attempt'],'returncode':completed.returncode,
        'binary':str(exe),'build_receipt':str(receipt_path),'compile_log':str(log_path),'probe_executed':False},ensure_ascii=False))
    return completed.returncode
if __name__=='__main__':sys.exit(main())
