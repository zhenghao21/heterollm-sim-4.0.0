"""Compile/static inspect only. No collector invocation occurs in this builder."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
P=Path(__file__).resolve().parent
ROOT=P.parents[5]
LOOP=P.parents[1]
CUDA=Path('E:/cuda')
SEM=ROOT/'source/llama.cpp-semantic'
LIB=SEM/'build-semantic-direct/ggml/src'
VC=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64')
VS=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
SOURCES=('protocol.json','collector.cpp','build.py','runner.py','test_runner.py')

def need(value,message):
    if not value:raise ValueError(message)
def now():return datetime.now(timezone.utc).isoformat()
def ref(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest,'bytes':path.stat().st_size}
def verify_ref(r):
    need(isinstance(r,dict) and set(r)=={'path','sha256','bytes'} and ref(r['path'])==r,'exact identity reference differs')
def references(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256','bytes'}:yield value
        else:
            for v in value.values():yield from references(v)
    elif isinstance(value,list):
        for v in value:yield from references(v)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def inputs():
    protocol=json.loads((P/'protocol.json').read_text(encoding='utf8'))
    for r in references(protocol):verify_ref(r)
    wrapper=load(LOOP/'round_026/mmvq_wrapper_launch_match/build.py','r26_unchanged_wrapper_builder')
    manifest=wrapper.verify_build()
    need(ref(LOOP/'round_026/mmvq_wrapper_launch_match/build_manifest.json')==protocol['identities']['wrapper_build'],'wrapper build changed')
    need(protocol['development']['M']==1 and protocol['held_out']['M']==4,'holdout boundary differs')
    need(protocol['sampling']['formal_calls_per_process']==64 and protocol['sampling']['initial_blocks']==5 and protocol['sampling']['extended_blocks']==10 and protocol['sampling']['wall_budget_seconds']==1200,'sampling budget changed')
    need(ref(P/'protocol.json')['sha256'] in (P/'collector.cpp').read_text(encoding='utf8'),'collector protocol digest stale')
    libs=[LIB/n for n in ('ggml-base.lib','ggml-cpu.lib','ggml.lib')]+[LIB/'ggml-cuda/ggml-cuda.lib']+[CUDA/'lib/x64'/n for n in ('cuda.lib','cudart.lib')]
    return {'sources':[ref(P/n) for n in SOURCES],'protocol_dependencies':list(references(protocol)),
        'compiler':[ref(x) for x in (VS,VC/'cl.exe',VC/'link.exe',VC/'dumpbin.exe',CUDA/'bin/cuobjdump.exe')],
        'libraries':[ref(x) for x in libs]}
def command(args,stem,out,cwd=None):
    line=subprocess.list2cmdline([str(x) for x in args]);cmd=out/(stem+'.cmd');log=out/(stem+'.log')
    with cmd.open('x',encoding='utf8') as f:f.write('@echo off\ncall '+subprocess.list2cmdline([str(VS)])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+line+'\n')
    with log.open('xb') as stream:result=subprocess.run(['cmd.exe','/c',str(cmd)],cwd=cwd or out,stdout=stream,stderr=subprocess.STDOUT)
    need(result.returncode==0,'static build failed; retained '+str(log))
    return {'command_ref':ref(cmd),'log_ref':ref(log),'returncode':result.returncode}
def build():
    need(not (P/'build_manifest.json').exists(),'sealed build exists; do not mutate it')
    before=inputs();n=1
    while (P/f'build_attempt.{n:04}').exists():n+=1
    out=P/f'build_attempt.{n:04}';out.mkdir();snapshots=out/'sources';snapshots.mkdir()
    for name in SOURCES:shutil.copyfile(P/name,snapshots/name)
    write_new(out/'start.json',{'created_utc':now(),'inputs':before,'source_snapshots':[ref(snapshots/name) for name in SOURCES],'GPU_executed':False})
    try:
        protocol=json.loads((P/'protocol.json').read_text(encoding='utf8'));ids=protocol['identities']
        obj=out/'collector.obj';exe=out/'gpu-operator-timing.exe';deps=out/'dependencies.json'
        args=[VC/'cl.exe','/nologo','/std:c++17','/EHsc','/MD','/O2','/fp:strict','/utf-8','/DGGML_SHARED']
        args+=['/I'+str(x) for x in (SEM/'ggml/include',SEM/'ggml/src',SEM/'vendor/nlohmann',CUDA/'include',CUDA/'extras/CUPTI/include')]
        records=[command(args+['/sourceDependencies',deps,'/c',P/'collector.cpp','/Fo'+str(obj)],'compile',out)]
        records.append(command([VC/'link.exe','/nologo','/OPT:REF','/OUT:'+str(exe),obj,ids['wrapper_object']['path'],ids['conversion_object']['path'],ids['support_object']['path'],
            '/LIBPATH:'+str(LIB),'/LIBPATH:'+str(LIB/'ggml-cuda'),'/LIBPATH:'+str(CUDA/'lib/x64'),'ggml-base.lib','ggml-cpu.lib','ggml.lib','ggml-cuda.lib','cudart.lib','cuda.lib','bcrypt.lib'],'link',out))
        records.append(command([VC/'dumpbin.exe','/DEPENDENTS',exe],'imports',out))
        imports=(out/'imports.log').read_text(encoding='utf8',errors='replace').lower()
        need('cupti' not in imports and 'nvperf' not in imports,'unobserved U/A must not import profiler DLLs')
        directory=out/'device_code';directory.mkdir()
        records.append(command([CUDA/'bin/cuobjdump.exe','--extract-elf','all',exe],'cubin',out,directory))
        device={}
        for name,pair in ids['golden_cubins'].items():
            matches=[f for f in directory.glob('*.cubin') if ref(f)['sha256']==pair['target_cubin']['sha256']]
            need(len(matches)==1,'wrapper device code not exact/unique '+name)
            device[name]={'target_ref':pair['target_cubin'],'new_ref':ref(matches[0]),'bytes_equal':matches[0].read_bytes()==Path(pair['target_cubin']['path']).read_bytes()}
        need(inputs()==before,'build inputs changed')
        includes=json.loads(deps.read_text(encoding='utf8-sig'))['Data']['Includes']
        manifest={'schema':'r28-gpu-timing-build/v1','created_utc':now(),'inputs':before,'headers':[ref(x) for x in includes],'executable':ref(exe),'object':ref(obj),'device_identity':device,'records':records,'GPU_executed':False,'cost_parameters_emitted':False}
        write_new(P/'build_manifest.json',manifest);write_new(out/'finish.json',{'status':'compiled_static_identity_passed','manifest_ref':ref(P/'build_manifest.json'),'GPU_executed':False})
        print(json.dumps({'status':'compiled_static_identity_passed','manifest_ref':ref(P/'build_manifest.json'),'GPU_executed':False}))
    except Exception as e:
        write_new(out/'finish.json',{'status':'rejected','error':type(e).__name__+': '+str(e),'GPU_executed':False});raise

def verify_build():
    m=json.loads((P/'build_manifest.json').read_text(encoding='utf8'));need(inputs()==m['inputs'],'sealed input closure changed')
    for r in references(m):verify_ref(r)
    return m

def host_test():
    m=verify_build();log=P/'host_tests.log'
    with log.open('xb') as stream:r=subprocess.run([sys.executable,'-m','pytest','-q',str(P/'test_runner.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    receipt={'schema':'r28-gpu-timing-host-tests/v1','created_utc':now(),'returncode':r.returncode,'manifest_ref':ref(P/'build_manifest.json'),'log_ref':ref(log),'inputs_unchanged':inputs()==m['inputs'],'GPU_executed':False}
    write_new(P/'host_test_receipt.json',receipt);need(r.returncode==0 and receipt['inputs_unchanged'],'host tests failed; receipt retained');print(json.dumps(receipt))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('check','build','host-test'));phase=parser.parse_args().phase
    if phase=='check':inputs();print('source closure checked; no GPU execution')
    elif phase=='build':build()
    else:host_test()
