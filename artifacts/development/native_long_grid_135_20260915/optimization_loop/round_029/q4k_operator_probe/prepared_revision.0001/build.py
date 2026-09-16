"""Host build and fixtures only. No GPU invocation in this entry."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys
import reference
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
LOOP=HERE.parent.parent
CUDA=Path('E:/cuda')
VS=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
VC=VS.parents[2]/'Tools/MSVC/14.44.35207/bin/Hostx64/x64'
SOURCES=('launch_decode.h','launch_metadata_json.h','locked_identity.h','fixture_lock.h','launch_recorder.cpp',
         'host_decode_test.cpp','reference.py','build.py','run_probe.py','test_probe.py','README.md','protocol.json')
LIB_DIRS=(ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/ggml/src',
          ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src',
          ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda',CUDA/'lib/x64')


def ref(path):
    p=Path(path).resolve(strict=True)
    with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':d}


def write_new(path,data):
    with Path(path).open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False)


def check_ref(r):
    if ref(r['path'])!=r:raise ValueError('input changed: '+r['path'])


def libraries():
    return [ref(next(d/n for d in LIB_DIRS if (d/n).is_file())) for n in ('ggml-base.lib','ggml-cuda.lib','cudart.lib')]


def prepare():
    fixture=reference.create_fixture()
    old=json.loads((LOOP/'round_026/mmvq_target_capture_ex/source_contract.json').read_text(encoding='utf8'))
    target=old['target_modules']+[ref(CUDA/'bin/cudart64_12.dll')]
    previous=LOOP/'round_026/mmvq_target_capture_ex'
    source=ROOT/'source/llama.cpp-semantic/ggml/src'
    evidence=[ref(previous/n) for n in ('launch_decode.h','launch_recorder.cpp','launch_metadata_json.h','host_decode_test.cpp','source_contract.json')]
    evidence += [ref(source/n) for n in ('ggml-cuda/mmvq.cu','ggml-cuda/vecdotq.cuh','ggml-cuda/quantize.cu','ggml-cuda/common.cuh','ggml-quants.c','ggml-common.h')]
    resources=ref(LOOP/'round_028/host_submission_probe/target_resource_usage.txt')
    main='_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'
    conversion='_Z13quantize_q8_1PKfPvxxxxxj5uint3'
    text=Path(resources['path']).read_text()
    if any(' Function '+symbol+':' not in text for symbol in (main,conversion)):raise ValueError('target kernel symbol not in pinned static dump')
    protocol={'schema':'r29-q4k-operator-protocol/v1','status':'prepared_runtime_unqualified',
      'shape':{'format':'Q4_K','type_id':12,'M':1,'K':2048,'N':2048,'ids':False,'fusion':False},
      'expected_main':{'symbol':main,'grid':[2048,1,1],'block':[32,4,1],'dynamic_shared_bytes':0,
        'reduction_K':2048,'small_K':False,'derivation':'generic warps4; Q4_K blocks=K/256=8; smallK requires8<8 (false)'},
      'expected_conversion':{'symbol':conversion,'grid':[8,1,1],'block':[256,1,1]},
      'expected_API':430,'PDL_attribute':{'id':6,'value':1},'expected_launches':2,
      'target_modules':target,'CUPTI':old['CUPTI_runtime'],'CUPTI_callback_runtime_API_version':130401,
      'compile_callback_API_version':26,'CUPTI_activity_timing_used':False,'HES_used':False,
      'callback_ABI_basis':'R26 pinned ExC parameter decoder; new shape still requires actual runtime validation',
      'source_refs':evidence,'resource_dump':resources,'fixture_manifest':ref(HERE/'fixture/manifest.json'),
      'fixture_files':fixture['files'],'GPU_uuid':'GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b',
      'one_untimed_warmup':True,'one_captured_replay':True,'no_auto_retry':True,
      'device_reads':'readback after capture disable and backend sync, never in callback',
      'numerical_gate':'full Q8_1 exact bytes and all2048 outputs within independent packed-Q4_K unsigned-amplitude gamma bound',
      'no_LLM_measurement':True,'no_timing':True,'cost_parameters':0,'cache_layout_for_static45_cells':'unproven',
      'held_out':[{'M':4,'K':2048,'N':2048},{'M':1,'K':1024,'N':2048},{'M':1,'K':4096,'N':2048}],
      'held_out_policy':'not executed; requires new fixture, decoder, source dispatch and runtime/numerical evidence',
      'lifecycle':'natural exit only; no timeout kill; any failure retained and blocks automatic repeat',
      'execution_guard_refs':[ref(LOOP/'round_028/host_service_controls/strict_gate.py')],
      'execution_prerequisite':'root serial scheduling; no live project native/simulator/GPU probe or identity diagnostic'}
    write_new(HERE/'protocol.json',protocol)
    fixture_dir=str((HERE/'fixture').resolve());sha=ref(HERE/'protocol.json')['sha256']
    lines=['#pragma once','namespace fixture_lock {','struct File {const char*path;const char*sha;};',
           'inline constexpr const char*directory='+json.dumps(fixture_dir)+';',
           'inline constexpr const char*protocol_sha='+json.dumps(sha)+';',
           'inline constexpr File files[]={']
    for r in fixture['files'].values():lines.append('{'+json.dumps(r['path'])+','+json.dumps(r['sha256'])+'},')
    lines+=['};','}']
    with (HERE/'fixture_lock.h').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'prepared':True,'GPU_executed':False,'fixture_manifest':ref(HERE/'fixture/manifest.json')}))


def inputs():
    p=json.loads((HERE/'protocol.json').read_text())
    externals=p['target_modules']+p['source_refs']+p['execution_guard_refs']+[p['CUPTI'],p['resource_dump'],p['fixture_manifest']]+list(p['fixture_files'].values())
    for r in externals:check_ref(r)
    return {'sources':[ref(HERE/n) for n in SOURCES],'external':externals,'libs':libraries(),
            'toolchain':[ref(VS),ref(VC/'cl.exe'),ref(VC/'dumpbin.exe')]}


def checked(args,label):
    cmd=HERE/(label+'.cmd');log=HERE/(label+'.log')
    with cmd.open('x') as f:f.write('@echo off\ncall '+subprocess.list2cmdline([str(VS)])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+subprocess.list2cmdline([str(x) for x in args])+'\n')
    with log.open('xb') as stream:
        child=subprocess.Popen(['cmd.exe','/c',str(cmd)],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        write_new(HERE/(label+'.child.json'),{'pid':child.pid})
        try:code=child.wait()
        except BaseException:
            write_new(HERE/(label+'.interrupted.json'),{'pid':child.pid,'child_may_be_live':child.poll() is None,'termination_requested':False});raise
    if code:raise RuntimeError('compile failed; preserved '+str(log))
    return {'command_ref':ref(cmd),'log_ref':ref(log),'returncode':code}


def build():
    before=inputs();attempt=1+len(list(HERE.glob('compile.target.*.cmd')))
    common=[VC/'cl.exe','/nologo','/showIncludes','/std:c++17','/EHsc','/O2','/fp:strict','/MD','/utf-8']
    common+=['/I'+str(x) for x in (ROOT/'source/llama.cpp-semantic/ggml/include',CUDA/'include',CUDA/'extras/CUPTI/include')]
    target=HERE/('q4k-target.%04d.exe'%attempt);host=HERE/('q4k-host-tests.%04d.exe'%attempt)
    records=[checked(common+['/DGGML_SHARED','/DGGML_BACKEND_SHARED',HERE/'launch_recorder.cpp','/Fe:'+str(target),'/Fo:'+str(target.with_suffix('.obj')),
               '/link',*[r['path'] for r in before['libs']],'bcrypt.lib'],'compile.target.%04d'%attempt),
             checked(common+[HERE/'host_decode_test.cpp','/Fe:'+str(host),'/Fo:'+str(host.with_suffix('.obj'))],'compile.host.%04d'%attempt)]
    deps=checked([VC/'dumpbin.exe','/nologo','/dependents',host],'imports.host.%04d'%attempt);records.append(deps)
    dlls=re.findall(r'^\s*([A-Za-z0-9_.-]+\.dll)\s*$',Path(deps['log_ref']['path']).read_text(errors='replace'),re.M|re.I)
    if not dlls or any(re.search('cuda|cupti|ggml|nvidia',v,re.I) for v in dlls):raise ValueError('host selftest must not import GPU')
    headers=set()
    for r in records[:2]:
        for line in Path(r['log_ref']['path']).read_text(errors='replace').splitlines():
            m=re.search(r'([A-Za-z]:[\\/].+)$',line)
            if m and Path(m.group(1).strip()).is_file():headers.add(str(Path(m.group(1).strip()).resolve()))
    if len(headers)<30 or inputs()!=before:raise ValueError('incomplete or changed build evidence')
    path=HERE/('build_manifest.%04d.json'%attempt)
    write_new(path,{'schema':'r29-q4k-build/v1','inputs':before,'headers':[ref(p) for p in sorted(headers)],
       'target_executable':ref(target),'host_executable':ref(host),'records':records,'host_imports':dlls,
       'target_executed':False,'GPU_executed':False,'target_DLL_rebuilt':False})
    print(json.dumps(ref(path)))


def verify(path):
    m=json.loads(Path(path).read_text())
    if m['inputs']!=inputs():raise ValueError('frozen inputs drift')
    for r in m['headers']+[m['target_executable'],m['host_executable']]:check_ref(r)
    return m


def host_test(path):
    m=verify(path)
    child=subprocess.Popen([m['host_executable']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    out,err=child.communicate()
    receipt={'manifest_ref':ref(path),'returncode':child.returncode,'stdout':out,'stderr':err,'GPU_executed':False,'target_executed':False}
    write_new(HERE/('host_test.%04d.json'%(len(list(HERE.glob('host_test.*.json')))+1)),receipt)
    if child.returncode:raise RuntimeError('host decoder test failed')
    print(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['prepare','build','host-test','verify']);ap.add_argument('--manifest',type=Path);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='build':build()
    elif a.mode=='host-test':host_test(a.manifest)
    else:verify(a.manifest);print('verified without target execution')
