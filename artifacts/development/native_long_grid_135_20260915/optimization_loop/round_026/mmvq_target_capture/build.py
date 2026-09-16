"""Build and host-only checks for the R26 recorder. No GPU execution action."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
CUDA = Path('E:/cuda')
VS = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
VC = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64')
SOURCES = ('launch_decode.h', 'launch_recorder.cpp', 'host_decode_test.cpp', 'locked_identity.h', 'source_contract.json', 'build.py')
INCLUDES = (ROOT/'source/llama.cpp-semantic/ggml/include', CUDA/'include', CUDA/'extras/CUPTI/include')
LIB_DIRS = (ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/ggml/src',
            ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src',
            ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda', CUDA/'lib/x64')

def ref(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as src: digest = hashlib.file_digest(src, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)

def write_new(path, data):
    with Path(path).open('x', encoding='utf8', newline='\n') as out:
        json.dump(data, out, indent=2); out.write('\n')

def imported_libraries():
    result = []
    for name in ('ggml-base.lib', 'ggml-cuda.lib', 'cudart.lib'):
        selected = next((d/name for d in LIB_DIRS if (d/name).is_file()), None)
        if selected is None: raise ValueError('Import library missing: '+name)
        result.append(selected)
    return result

def verify_inputs():
    contract = json.loads((HERE/'source_contract.json').read_text(encoding='utf8'))
    pinned = contract['target_modules'] + contract['source_files'] + contract['device_code_refs'] + contract['symbol_refs']
    pinned += [contract[k] for k in ('original_identity_lock', 'reference_recorder', 'CUPTI_runtime', 'device_code_identity_ref', 'hardware_ref')]
    for expected in pinned:
        if ref(expected['path']) != expected: raise ValueError('Pinned input changed: '+expected['path'])
    text = (HERE/'launch_decode.h').read_text(encoding='utf8')
    symbols = re.findall(r'inline constexpr const char \* k(?:Conversion|Main)Symbol = "([^"]+)";', text)
    if len(symbols) != 2: raise ValueError('Two exact accepted symbols required')
    symbol_text = '\n'.join(Path(item['path']).read_text(encoding='utf8') for item in contract['symbol_refs'])
    if not all(' Function '+symbol+':' in symbol_text for symbol in symbols):
        raise ValueError('Accepted symbol missing from locked target device-code resources')
    cupti = CUDA/'extras/CUPTI/include'
    for filename, required in {
        'generated_cuda_runtime_api_meta.h': ('cudaLaunchKernel_v7000_params', 'void **args', 'cudaStream_t stream'),
        'cupti_callbacks.h': ('functionReturnValue', 'symbolName', 'functionParams', 'cuptiEnableDomain'),
        'cupti_runtime_cbid.h': ('CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000',),
        'cupti_version.h': ('#define CUPTI_API_VERSION 26',),
    }.items():
        content = (cupti/filename).read_text(encoding='utf8')
        if not all(s in content for s in required): raise ValueError('Installed CUPTI header contract mismatch: '+filename)
        pinned.append(ref(cupti/filename))
    return { 'local_sources': [ref(HERE/name) for name in SOURCES], 'pinned_external_inputs': pinned,
             'toolchain': [ref(VS), ref(VC/'cl.exe'), ref(VC/'link.exe'), ref(VC/'dumpbin.exe')],
             'import_libraries': [ref(p) for p in imported_libraries()] }

def checked_command(args, stem, env):
    command = subprocess.list2cmdline([str(a) for a in args])
    cmd = HERE/(stem+'.cmd'); log = HERE/(stem+'.log')
    with cmd.open('x', encoding='utf8', newline='\n') as out:
        out.write('@echo off\ncall '+subprocess.list2cmdline([str(VS)])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+command+'\n')
    with log.open('xb') as out:
        p = subprocess.run(['cmd.exe', '/c', str(cmd)], cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT, check=False)
    if p.returncode: raise RuntimeError(f'Command failed with {p.returncode}; evidence preserved: {log}')
    return dict(command=command, returncode=p.returncode, command_ref=ref(cmd), log_ref=ref(log))

def header_refs(records):
    headers = set()
    for record in records:
        raw = Path(record['log_ref']['path']).read_bytes()
        try: text = raw.decode('utf-8-sig')
        except UnicodeDecodeError: text = raw.decode('gb18030')
        for line in text.splitlines():
            match = re.search(r'including file:\s*(.+)$', line, re.I) or re.search(r'^[^:]+:[^:]+:[ \t]+([A-Za-z]:[\\/].*)$', line)
            if match: headers.add(Path(match.group(1).strip()).resolve(strict=True))
    if len(headers)<30: raise ValueError('Incomplete compiler include evidence')
    return [ref(p) for p in sorted(headers)]

def build():
    before = verify_inputs()
    attempt = 1+len(list(HERE.glob('compile.target.*.cmd')))
    env=os.environ.copy();env['VSLANG']='1033'
    common=[VC/'cl.exe','/nologo','/showIncludes','/std:c++17','/EHsc','/O2','/fp:strict','/MD','/utf-8']
    common += ['/I'+str(p) for p in INCLUDES]
    target=HERE/f'mmvq-target-recorder.{attempt:04d}.exe'
    host=HERE/f'mmvq-host-decode-tests.{attempt:04d}.exe'
    target_args=common+['/DGGML_SHARED','/DGGML_BACKEND_SHARED',HERE/'launch_recorder.cpp','/Fe:'+str(target),
                        '/Fo:'+str(target.with_suffix('.obj')),'/link',*imported_libraries(),'bcrypt.lib']
    target_record=checked_command(target_args,f'compile.target.{attempt:04d}',env)
    host_args=common+[HERE/'host_decode_test.cpp','/Fe:'+str(host),'/Fo:'+str(host.with_suffix('.obj'))]
    host_record=checked_command(host_args,f'compile.host.{attempt:04d}',env)
    dependencies=checked_command([VC/'dumpbin.exe','/nologo','/dependents',host],f'host.dependencies.{attempt:04d}',env)
    dependency_text=Path(dependencies['log_ref']['path']).read_bytes().decode('utf8',errors='replace')
    dlls=re.findall(r'^\s*([A-Za-z0-9_.-]+\.dll)\s*$',dependency_text,re.M|re.I)
    if not dlls or any(re.search(r'cuda|cupti|ggml|nvidia',name,re.I) for name in dlls):
        raise ValueError('Host test must have zero GPU/CUPTI/GGML imports')
    if verify_inputs()!=before: raise ValueError('Inputs changed during build')
    manifest=dict(schema='heterollm.mmvq-target-capture-build/v1',status='compiled_not_GPU_executed',
                  inputs=before,target_executable=ref(target),host_test_executable=ref(host),
                  compile_records=[target_record,host_record],compile_headers=header_refs([target_record,host_record]),
                  host_dependencies=dependencies,host_imported_dlls=dlls,host_GPU_import_count=0,
                  target_executable_executed=False,gpu_execution_performed=False,
                  target_LLM_latency_used=False,source_built_kernel=False,original_DLL_rebuilt=False)
    manifest_path = HERE/'build_manifest.json' if not (HERE/'build_manifest.json').exists() else HERE/f'build_manifest.{attempt:04d}.json'
    write_new(manifest_path,manifest)
    print(json.dumps({'status':'compiled','target_executable':ref(target),'host_GPU_import_count':0,'manifest':ref(manifest_path)}))

def host_test(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['inputs']!=verify_inputs(): raise ValueError('Build/input identity differs')
    for expected in manifest['compile_headers']+[manifest['target_executable'],manifest['host_test_executable']]:
        if ref(expected['path'])!=expected: raise ValueError('Built dependency changed: '+expected['path'])
    attempt=1+len(list(HERE.glob('host_test.*.json')))
    stdout=HERE/f'host_test.{attempt:04d}.stdout.json';stderr=HERE/f'host_test.{attempt:04d}.stderr.txt'
    argv=[manifest['host_test_executable']['path']]
    with stdout.open('xb') as out,stderr.open('xb') as err:
        process=subprocess.run(argv,cwd=HERE,stdout=out,stderr=err,check=False)
    receipt=dict(schema='heterollm.mmvq-target-host-test-receipt/v1',argv=argv,returncode=process.returncode,
                 stdout_ref=ref(stdout),stderr_ref=ref(stderr),build_manifest_ref=ref(manifest_path),
                 target_executable_executed=False,gpu_execution_performed=False)
    if process.returncode==0: receipt['result']=json.loads(stdout.read_text(encoding='utf8'))
    receipt['inputs_unchanged']=manifest['inputs']==verify_inputs()
    write_new(HERE/f'host_test.{attempt:04d}.json',receipt)
    print(json.dumps(receipt['result'] if process.returncode==0 else receipt))
    if process.returncode or not receipt['inputs_unchanged']: raise RuntimeError('Host test failed; receipt retained')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['verify','build','host-test'])
    parser.add_argument('--manifest',type=Path,default=HERE/'build_manifest.json')
    args=parser.parse_args();mode=args.mode
    if mode=='build':build()
    elif mode=='host-test':host_test(args.manifest)
    else:print(json.dumps({'status':'inputs_verified','input_groups':list(verify_inputs()),'GPU_executed':False}))
