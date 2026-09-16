"""Prepare a host shim/recorder. Build/static ELF inspection only; no GPU run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys
P=Path(__file__).resolve().parent
OUT=P
ROOT=P.parents[5]
LOOP=P.parents[1]
EX=P.parent/'mmvq_target_capture_ex'
R6=LOOP/'round_024/mmvq_device_probe/r6_shared_abi'
CUDA=Path('E:/cuda')
VC=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64')
VS=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
SEM=ROOT/'source/llama.cpp-semantic'
LIB=SEM/'build-semantic-direct/ggml/src'
SOURCES=('build.py','wrapper_recorder.cpp','wrapper_identity.h','contiguous_layout.h','layout_host_test.cpp',
         'mmvq_contiguous_source.cu','source_manifest.json','QI5_ERRATUM.json','compare_records.py','test_compare_records.py','run_wrapper_capture.py','test_run_wrapper_capture.py','README.md')

def ref(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest,'bytes':path.stat().st_size}
def need(value,message):
    if not value:raise ValueError(message)
def check_ref(expected):
    need(ref(expected['path'])==expected,'identity changed: '+expected['path'])
def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def inputs():
    manifest=json.loads((P/'source_manifest.json').read_text(encoding='utf8'))
    for row in manifest['external_refs']:check_ref(row)
    for row in manifest['expected_runtime_dlls'].values():check_ref(row)
    proof=manifest['source_prefix'];check_ref(proof['original'])
    prefix=b''.join(Path(proof['original']['path']).read_bytes().splitlines(keepends=True)[:proof['through_line']])
    need(hashlib.sha256(prefix).hexdigest()==proof['sha256'],'prefix SHA differs')
    need((P/'mmvq_contiguous_source.cu').read_bytes().startswith(prefix),'original device/helper prefix changed')
    ex_build=load(EX/'build.py','shared_ExC_frozen_builder')
    ex_manifest=json.loads((EX/'build_manifest.json').read_text(encoding='utf8'))
    need(ex_build.verify_inputs()==ex_manifest['inputs'],'shared recorder source closure changed')
    for item in ex_manifest['compile_headers']:check_ref(item)
    return {'local_sources':[ref(P/name) for name in SOURCES],
        'external_refs':manifest['external_refs'],'runtime_refs':manifest['expected_runtime_dlls'],
        'shared_recorder_manifest':ref(EX/'build_manifest.json'),
        'compiler_refs':[ref(x) for x in (VC/'cl.exe',VC/'link.exe',VC/'dumpbin.exe',VS,CUDA/'bin/nvcc.exe',CUDA/'bin/cuobjdump.exe')],
        'libraries':[ref(LIB/name) for name in ('ggml-base.lib','ggml-cpu.lib','ggml.lib')]+[ref(CUDA/'lib/x64'/name) for name in ('cudart.lib','cuda.lib')]}

def command(text,stem,cwd=P):
    cmd=OUT/(stem+'.cmd');log=OUT/(stem+'.log')
    with cmd.open('x',encoding='utf8') as out:
        out.write('@echo off\ncall '+subprocess.list2cmdline([str(VS)])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+text+'\n')
    with log.open('xb') as out:
        process=subprocess.run(['cmd.exe','/c',str(cmd)],cwd=cwd,stdout=out,stderr=subprocess.STDOUT)
    need(process.returncode==0,'compile/static command failed; retained '+str(log))
    return {'command':text,'returncode':process.returncode,'command_ref':ref(cmd),'log_ref':ref(log)}
def arg_command(args,stem,cwd=P):return command(subprocess.list2cmdline([str(a) for a in args]),stem,cwd)

def verify_build():
    m=json.loads((P/'build_manifest.json').read_text(encoding='utf8'))
    need(inputs()==m['inputs'],'prepared build input closure differs')
    for r in [m['executable'],m['host_test_executable'],m['new_cuda_object'],*m['compiler_discovered_includes']]:check_ref(r)
    for pair in m['device_identity']['pairs'].values():
        check_ref(pair['target_cubin']);check_ref(pair['wrapper_cubin'])
        need(pair['bytes_equal'] is True and Path(pair['target_cubin']['path']).read_bytes()==Path(pair['wrapper_cubin']['path']).read_bytes(),'device identity changed')
    return m

def build_inner():
    before=inputs();source=json.loads((P/'source_manifest.json').read_text(encoding='utf8'))
    r6=json.loads((R6/'source_slice_receipt.json').read_text(encoding='utf8'))
    old=r6['compile_records'][0];obj=OUT/'mmvq_contiguous_source.obj'
    cuda_command=old['command'].replace(old['source'],str(P/'mmvq_contiguous_source.cu')).replace(old['object'],str(obj))
    need(str(P/'mmvq_contiguous_source.cu') in cuda_command and str(obj) in cuda_command,'narrow compile substitution failed')
    cuda_record=command(cuda_command,'cuda.compile')
    deps=OUT/'driver.dependencies.json';driver_obj=OUT/'wrapper_recorder.obj';exe=OUT/'mmvq-wrapper-launch-match.exe'
    include=[SEM/'ggml/include',SEM/'ggml/src',CUDA/'include',CUDA/'extras/CUPTI/include']
    cc=[VC/'cl.exe','/nologo','/std:c++17','/EHsc','/MD','/O2','/fp:strict','/DGGML_SHARED','/utf-8']+['/I'+str(x) for x in include]
    cpp_record=arg_command(cc+['/sourceDependencies',deps,'/c',P/'wrapper_recorder.cpp','/Fo'+str(driver_obj)],'driver.compile')
    link_record=arg_command([VC/'link.exe','/nologo','/OPT:REF','/OUT:'+str(exe),driver_obj,obj,
        R6/'quantize_same_source.obj',R6/'mmvq_runtime_support.obj','/LIBPATH:'+str(LIB),
        '/LIBPATH:'+str(CUDA/'lib/x64'),'ggml-base.lib','ggml-cpu.lib','ggml.lib','cudart.lib','cuda.lib','bcrypt.lib'],'driver.link')
    arg_command([VC/'dumpbin.exe','/DEPENDENTS',exe],'driver.imports')
    need('ggml-cuda.dll' not in (OUT/'driver.imports.log').read_text(encoding='utf8',errors='replace').lower(),'wrapper imported original backend')
    host=OUT/'mmvq-layout-host-test.exe'
    host_record=arg_command([VC/'cl.exe','/nologo','/std:c++17','/EHsc','/MD','/O2','/utf-8',P/'layout_host_test.cpp','/Fe'+str(host),'/Fo'+str(OUT/'layout_host_test.obj')],'host.compile')
    arg_command([VC/'dumpbin.exe','/DEPENDENTS',host],'host.imports')
    deptext=(OUT/'host.imports.log').read_text(encoding='utf8',errors='replace').lower()
    need(not any(x in deptext for x in ('cudart','cupti','nvcuda','ggml-')),'host test has GPU imports')
    directory=OUT/'static_device_code';directory.mkdir(exist_ok=False)
    dump_record=arg_command([CUDA/'bin/cuobjdump.exe','--extract-elf','all',exe],'static_device_code',directory)
    hashes={p:ref(p) for p in directory.glob('*.cubin')}
    pairs={}
    for name,expected in source['golden_cubins'].items():
        target=Path(expected['target_path']);need(ref(target)['sha256']==expected['target_sha256'],'golden cubin changed')
        matches=[p for p,r in hashes.items() if r['sha256']==expected['target_sha256']]
        need(len(matches)==1,'new '+name+' cubin not uniquely byte-identical to original')
        pairs[name]={'target_cubin':ref(target),'wrapper_cubin':ref(matches[0]),'bytes_equal':target.read_bytes()==matches[0].read_bytes()}
    need(inputs()==before,'inputs changed during build')
    headers=json.loads(deps.read_text(encoding='utf-8-sig'))['Data']['Includes']
    manifest={'schema':'heterollm.mmvq-wrapper-launch-match-build/v1','inputs':before,'executable':ref(exe),
        'host_test_executable':ref(host),'new_cuda_object':ref(obj),'compiler_discovered_includes':[ref(p) for p in headers],
        'build_records':[cuda_record,cpp_record,link_record,host_record],'device_identity':{'inspection':dump_record,'pairs':pairs},
        'GPU_executed':False,'target_native_DLL_rebuilt':False,'reused_unchanged_objects':[ref(R6/'quantize_same_source.obj'),ref(R6/'mmvq_runtime_support.obj')],
        'performance_parameter_admitted':False,'runtime_match_verified':False}
    write_new(P/'build_manifest.json',manifest)
    print(json.dumps({'status':'compiled_static_identity_passed','manifest':ref(P/'build_manifest.json'),'GPU_executed':False}))

def build():
    global OUT
    need(not (P/'build_manifest.json').exists(),'build already sealed; use a new revision')
    attempt=1
    while (P/('build_attempt.%04d'%attempt)).exists():attempt+=1
    OUT=P/('build_attempt.%04d'%attempt);OUT.mkdir(exist_ok=False)
    write_new(OUT/'start.json',{'schema':'wrapper-build-start/v1','inputs':inputs(),'GPU_executed':False})
    try:
        build_inner()
        write_new(OUT/'finish.json',{'schema':'wrapper-build-finish/v1','status':'compiled_static_identity_passed','manifest_ref':ref(P/'build_manifest.json'),'GPU_executed':False})
    except Exception as error:
        write_new(OUT/'finish.json',{'schema':'wrapper-build-finish/v1','status':'rejected','error':type(error).__name__+': '+str(error),'GPU_executed':False})
        raise

def host_test():
    manifest=verify_build();out=P/'host_layout.stdout.json';err=P/'host_layout.stderr.txt'
    with out.open('xb') as stdout,err.open('xb') as stderr:
        proc=subprocess.run([manifest['host_test_executable']['path']],cwd=P,stdout=stdout,stderr=stderr)
    testlog=P/'comparison_tests.log'
    with testlog.open('xb') as stdout:
        tests=subprocess.run([sys.executable,'-m','pytest','-q',str(P/'test_compare_records.py'),str(P/'test_run_wrapper_capture.py')],cwd=ROOT,stdout=stdout,stderr=subprocess.STDOUT)
    receipt={'schema':'heterollm.mmvq-wrapper-launch-host-tests/v1','build_manifest_ref':ref(P/'build_manifest.json'),
        'layout_returncode':proc.returncode,'comparison_returncode':tests.returncode,'stdout_ref':ref(out),'stderr_ref':ref(err),'comparison_log_ref':ref(testlog),
        'layout_result':json.loads(out.read_text(encoding='utf8')) if proc.returncode==0 else None,
        'shared_recorder_63_test_ref':ref(EX/'host_test.0001.json'),'GPU_executed':False,'inputs_unchanged':inputs()==manifest['inputs']}
    write_new(P/'host_test_receipt.json',receipt)
    need(proc.returncode==tests.returncode==0 and receipt['inputs_unchanged'],'host qualification failed; receipts retained')
    print(json.dumps({'status':'host_checks_passed','receipt':ref(P/'host_test_receipt.json'),'GPU_executed':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('check-inputs','build','host-test'));mode=parser.parse_args().mode
    if mode=='build':build()
    elif mode=='host-test':host_test()
    else:inputs();print('verified input closure; no GPU execution')
