"""Prepare/compile/freeze only this standalone probe after parent review. Never runs GPU."""
from __future__ import annotations
import argparse, datetime, hashlib, json, re, subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE_FILES=['graph_submit_probe.cpp','math_reference.h','reference_host.cpp','frozen_module_guard.h','protocol.json','source_provenance.json','build_probe.py','invoke.ps1','reference_check.py','README.md']
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def ref(p):
    p=Path(p).resolve()
    if not p.is_file():raise RuntimeError('Missing file: '+str(p))
    return {'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}
def verify(items):
    if not items:raise RuntimeError('Empty identity set')
    names=set()
    for item in items:
        if not isinstance(item,dict) or not item.get('path') or not re.fullmatch('[a-f0-9]{64}',item.get('sha256') or '') or not isinstance(item.get('bytes'),int) or item['bytes']<=0:raise RuntimeError('Incomplete identity')
        if item['path'] in names:raise RuntimeError('Duplicate identity: '+item['path'])
        names.add(item['path'])
        if ref(item['path'])!=item:raise RuntimeError('Changed frozen input: '+item['path'])
def write_new(p,text):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def cstr(s):return json.dumps(s,ensure_ascii=True)
def main():
    a=argparse.ArgumentParser();a.add_argument('--compile',action='store_true');a.add_argument('--reviewed',action='store_true');opt=a.parse_args()
    protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'))
    provenance=json.loads((P/'source_provenance.json').read_text(encoding='utf-8'))
    verify(provenance['files']);verify(provenance['native_modules'])
    for item in provenance['copied_sources']:
        if ref(item['from']['path'])!=item['from'] or ref(item['to']['path'])!=item['to'] or item['from']['sha256']!=item['to']['sha256']:raise RuntimeError('Copied-source identity mismatch')
    if not opt.compile:
        print(json.dumps({'status':'source_dependencies_verified_not_compiled','configs':len(protocol['configs']),'gpu_access':False}));return
    if not opt.reviewed:raise RuntimeError('Parent source review required before compiling this new probe')
    forbidden=['build_manifest.json','prepared_identity.h','compile.cmd','compile.log','graph-submit-probe.exe','graph-submit-probe.obj','reference-host.exe','reference-host.obj','host_reference_validation.json']
    if any((P/x).exists() for x in forbidden):raise RuntimeError('Refusing to overwrite an existing build attempt; preserve it and prepare a new version')
    toolchain=provenance['toolchain'];compiler=Path(toolchain['compiler'])
    root=next(a for a in P.parents if (a/'pyproject.toml').exists() and (a/'source').is_dir())
    base_lib=root/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-base.lib'
    cuda_lib=root/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda/ggml-cuda.lib'
    common=[str(compiler),'/nologo','/showIncludes','/std:c++17','/EHsc','/O2','/fp:strict','/MD','/utf-8','/DGGML_SHARED','/DGGML_BACKEND_SHARED','/I'+str(root/'source/llama.cpp-semantic/ggml/include'),'/I'+toolchain['cuda_include']]
    probe=common+[str(P/'graph_submit_probe.cpp'),'/Fe:'+str(P/'graph-submit-probe.exe'),'/Fo:'+str(P/'graph-submit-probe.obj'),'/link',str(base_lib),str(cuda_lib),toolchain['cuda_library'],'bcrypt.lib']
    host=[str(compiler),'/nologo','/showIncludes','/std:c++17','/EHsc','/O2','/fp:strict','/MD','/utf-8',str(P/'reference_host.cpp'),'/Fe:'+str(P/'reference-host.exe'),'/Fo:'+str(P/'reference-host.obj')]
    cmd='@echo off\ncall '+subprocess.list2cmdline([toolchain['vcvars']])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+subprocess.list2cmdline(probe)+'\nif errorlevel 1 exit /b %errorlevel%\n'+subprocess.list2cmdline(host)+'\nexit /b %errorlevel%\n'
    write_new(P/'compile.cmd',cmd)
    tools=[toolchain['vcvars'],compiler,compiler.parent/'link.exe',toolchain['cuda_library'],protocol['profiler']['path']]
    before={x['path']:x for x in provenance['files']+provenance['native_modules']}
    for f in [*(P/x for x in SOURCE_FILES),P/'compile.cmd',*tools]:
        x=ref(f);before[x['path']]=x
    configs=protocol['configs'];environment=dict(protocol['runtime']['environment'])
    for name in protocol['runtime']['extra_clear_environment']:environment[name]=None
    expected=protocol['runtime']['gpu_expected'];major,minor=map(int,expected['compute_capability'].split('.'))
    header='#pragma once\nstruct FileIdentity {const char *path; const char *sha256;};\nstatic const FileIdentity frozen_files[]={\n'+''.join(' {'+cstr(x['path'])+','+cstr(x['sha256'])+'},\n' for x in sorted(before.values(),key=lambda x:x['path']))+'};\nstruct ModuleIdentity {const char *path; const char *hash;};\nstatic const ModuleIdentity frozen_modules[]={\n'+''.join(' {'+cstr(x['path'])+','+cstr(x['sha256'])+'},\n' for x in provenance['native_modules'])+'};\nstruct EnvIdentity {const char *name; const char *value;};\nstatic const EnvIdentity frozen_environment[]={\n'+''.join(' {'+cstr(k)+','+('nullptr' if v is None else cstr(v))+'},\n' for k,v in environment.items())+'};\nstruct ProbeConfig {const char *id; long long elements; int nodes;};\nstatic const ProbeConfig frozen_configs[]={\n'+''.join(' {'+cstr(x['id'])+','+str(x['elements'])+','+str(x['nodes'])+'},\n' for x in configs)+'};\nstatic const char *protocol_sha256='+cstr(digest(P/'protocol.json'))+';\nstatic const char *expected_gpu_uuid='+cstr(expected['uuid'])+';\nstatic const char *expected_gpu_name='+cstr(expected['name'])+';\nstatic const int expected_cc_major='+str(major)+';\nstatic const int expected_cc_minor='+str(minor)+';\n'
    write_new(P/'prepared_identity.h',header)
    with (P/'compile.log').open('xb') as log:
        result=subprocess.run(['cmd.exe','/d','/c',str(P/'compile.cmd')],cwd=P,stdout=log,stderr=subprocess.STDOUT,check=False)
    if result.returncode:raise RuntimeError('Compile failed; preserve compile.log and all outputs; no freeze published')
    verify(list(before.values()))
    rawlog=(P/'compile.log').read_bytes()
    try:text=rawlog.decode('utf-8-sig')
    except UnicodeDecodeError:text=rawlog.decode('cp936')
    includes=set()
    for line in text.splitlines():
        m=re.search(r'(?:包含文件:|including file:)\s*(.*)$',line)
        if m:includes.add(str(Path(m.group(1).strip()).resolve()))
    if len(includes)<50:raise RuntimeError('Incomplete compiler include closure; no freeze published')
    host_result=subprocess.run([str(P/'reference-host.exe')],cwd=P,capture_output=True,text=True,check=False)
    try:host_validation=json.loads(host_result.stdout)
    except json.JSONDecodeError:raise RuntimeError('Host reference did not return structured evidence')
    if host_result.returncode or host_validation.get('pass') is not True or host_validation.get('gpu_access') is not False:raise RuntimeError('Host reference validation failed')
    write_new(P/'host_reference_validation.json',json.dumps(host_validation,indent=2)+'\n')
    files=dict(before)
    for f in [*includes,P/'prepared_identity.h',P/'compile.log',P/'graph-submit-probe.exe',P/'reference-host.exe',P/'host_reference_validation.json']:
        x=ref(f);files[x['path']]=x
    verify(list(files.values()))
    manifest={'schema':'graph-submit-probe-build/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'compiled_host_tested_not_gpu_executed','gpu_access':False,'config_count':6,'timed_runs':0,'native_binary_modified':False,'native_bin':str(Path(provenance['native_modules'][0]['path']).parent),'executable':ref(P/'graph-submit-probe.exe'),'profiler':ref(protocol['profiler']['path']),'protocol':ref(P/'protocol.json'),'compile_header_count':len(includes),'host_reference':host_validation,'files':sorted(files.values(),key=lambda x:x['path']),'source_runtime_equivalence_proven':False,'calibration_ready':False,'limitations':['Actual device kernel counts and graph-replay mode await CUPTI validation','Graph trace observer effect must pass paired protocol gates','CPU pool/native server scheduling is outside this synthetic chain']}
    write_new(P/'build_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'frozen_files':len(files),'header_count':len(includes),'exe_sha256':manifest['executable']['sha256'],'gpu_access':False}))
if __name__=='__main__':main()
