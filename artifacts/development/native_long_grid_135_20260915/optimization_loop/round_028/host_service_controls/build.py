"""Compile/verify and pure-logic tests only. Counter/GPU probes are never run here."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
BASE=HERE.parent/'host_submission_probe'
VC=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64')
VARS=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat')
CRT_COOKIE=VC.parents[2]/'crt/src/vcruntime/gs_support.c'
SOURCES=('logic.h','counters.h','counter_pilot.cpp','service_probe.cpp','host_tests.cpp','build.py','analyze.py','run_controls.py','strict_gate.py','test_controls.py','protocol.json','README.md')
def ref(p):
 p=Path(p).resolve(strict=True);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write_new(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def verify_inputs():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
 external=p['campaign_barrier']['verifier_refs']+p['base_source_refs']+[p['predecessor_protocol_ref']]+p['target_modules']+p['source_refs']+[p['hardware_ref'],p['cupti_ref']]
 for r in external:
  if ref(r['path'])!=r:raise ValueError('external input drift: '+r['path'])
 dirs=[ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src',ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda',Path('E:/cuda/lib/x64')]
 libs=[next(d/n for d in dirs if(d/n).is_file()) for n in ('ggml-base.lib','ggml-cuda.lib','cudart.lib')]
 return dict(sources=[ref(HERE/n) for n in SOURCES],external=external,libraries=[ref(x) for x in libs],toolchain=[ref(VC/'cl.exe'),ref(VC/'dumpbin.exe'),ref(VARS),ref(CRT_COOKIE)])
def checked(args,label):
 cmd=HERE/(label+'.cmd');log=HERE/(label+'.log')
 with cmd.open('x',encoding='utf8') as f:f.write('@echo off\ncall '+subprocess.list2cmdline([str(VARS)])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+subprocess.list2cmdline([str(v) for v in args])+'\n')
 env=os.environ.copy();env['VSLANG']='1033'
 with log.open('xb') as f:
  child=subprocess.Popen(['cmd.exe','/c',str(cmd)],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
  write_new(HERE/(label+'.child.json'),dict(pid=child.pid))
  try:code=child.wait()
  except BaseException:
   write_new(HERE/(label+'.observation_interrupted.json'),dict(pid=child.pid,child_may_be_live=child.poll() is None,child_left_running=True));raise
 if code:raise ValueError('compile inspection failed; preserve '+str(log))
 return dict(command_ref=ref(cmd),log_ref=ref(log),returncode=code)
def validate_host_imports(imports,symbols,linkmap):
 # The linked MSVC startup cookie gathers entropy using QPC; it is not a probe.
 forbidden=('GetThreadTimes','QueryThreadCycleTime','QueryPerformanceFrequency','Sleep')
 if any(re.search(r'\b'+name+r'\b',imports) for name in forbidden):raise ValueError('pure logic test imports experiment timing API')
 if any(name in symbols for name in forbidden+('QueryPerformanceCounter',)):raise ValueError('pure logic object references experiment timing API')
 qpc='QueryPerformanceCounter' in imports
 if qpc and ('__security_init_cookie' not in linkmap or 'QueryPerformanceCounter' not in CRT_COOKIE.read_text(errors='replace')):raise ValueError('unexplained startup QPC dependency')
 return dict(experiment_timing_API_references_in_test_object=False,CRT_startup_QPC_import=qpc,CRT_source_ref=ref(CRT_COOKIE),runtime_clock_calls_instrumented=False)
def build():
 before=verify_inputs();attempt=len(list(HERE.glob('compile.service.*.cmd')))+1
 (HERE/'binding.h').write_text('#pragma once\n#define CONTROL_PROTOCOL_SHA "'+ref(HERE/'protocol.json')['sha256']+'"\n',encoding='utf8')
 include=[HERE,ROOT/'source/llama.cpp-semantic/ggml/include',Path('E:/cuda/include'),Path('E:/cuda/extras/CUPTI/include')]
 common=[VC/'cl.exe','/nologo','/showIncludes','/std:c++17','/EHsc','/O2','/fp:strict','/MD','/utf-8',*('/I'+str(p) for p in include)]
 variants={};records=[];headers=set()
 for label,source,prefix in [('service','service_probe.cpp','host-service-probe'),('pilot','counter_pilot.cpp','host-counter-probe'),('host','host_tests.cpp','host-service-logic-tests')]:
  exe=HERE/(prefix+'.%04d.exe'%attempt)
  args=common+(['/DGGML_SHARED','/DGGML_BACKEND_SHARED'] if label=='service' else [])+[HERE/source,'/Fe:'+str(exe),'/Fo:'+str(exe.with_suffix('.obj'))]
  if label=='service':args+=['/link',*[r['path'] for r in before['libraries']],'bcrypt.lib','psapi.lib']
  if label=='host':args+=['/link','/MAP:'+str(exe.with_suffix('.map'))]
  record=checked(args,'compile.'+label+'.%04d'%attempt);records.append(record)
  inspection=checked([VC/'dumpbin.exe','/nologo','/imports',exe],'imports.'+label+'.%04d'%attempt);records.append(inspection)
  imports=Path(inspection['log_ref']['path']).read_text(errors='replace')
  if label!='service' and re.search(r'cudart|nvcuda|cupti|ggml-',imports,re.I):raise ValueError('host/pilot CUDA dependency')
  host_evidence=None
  if label=='host':
   obj=checked([VC/'dumpbin.exe','/nologo','/symbols',exe.with_suffix('.obj')],'symbols.host.%04d'%attempt);records.append(obj)
   host_evidence=validate_host_imports(imports,Path(obj['log_ref']['path']).read_text(errors='replace'),exe.with_suffix('.map').read_text(errors='replace'))
   host_evidence.update(symbols_ref=obj['log_ref'],link_map_ref=ref(exe.with_suffix('.map')))
  for line in Path(record['log_ref']['path']).read_text(errors='replace').splitlines():
   m=re.search(r'([A-Za-z]:[\\/].+)$',line)
   if m and Path(m.group(1).strip()).is_file():headers.add(str(Path(m.group(1).strip()).resolve()))
  variants[label]=dict(executable=ref(exe),object=ref(exe.with_suffix('.obj')),imports_ref=inspection['log_ref'],executed=False,host_logic_evidence=host_evidence)
 if verify_inputs()!=before:raise ValueError('inputs changed during compile')
 out=dict(schema='host-service-controls-build/v1',inputs=before,headers=[ref(p) for p in sorted(headers)],binding_ref=ref(HERE/'binding.h'),variants=variants,records=records,any_experiment_timing_executed=False,any_GPU_executed=False,target_DLL_rebuilt=False)
 name=HERE/('build_manifest.%04d.json'%attempt);write_new(name,out);print(json.dumps({'manifest':ref(name),'status':'compiled_only'}))
def verify_manifest(path):
 m=json.loads(Path(path).read_text(encoding='utf8'))
 if m.get('schema')!='host-service-controls-build/v1' or not m.get('headers') or set(m.get('variants',{}))!={'host','pilot','service'} or not m.get('records'):raise ValueError('incomplete build manifest')
 if m['inputs']!=verify_inputs():raise ValueError('build input drift')
 for r in m['headers']+[m['binding_ref']]+[v['executable'] for v in m['variants'].values()]+[v['object'] for v in m['variants'].values()]:
  if ref(r['path'])!=r:raise ValueError('compiled identity drift')
 for row in m['records']:
  for key in ('command_ref','log_ref'):
   if ref(row[key]['path'])!=row[key]:raise ValueError('build evidence drift')
 host=m['variants']['host']['host_logic_evidence']
 for key in ('symbols_ref','link_map_ref','CRT_source_ref'):
  if ref(host[key]['path'])!=host[key]:raise ValueError('host source dependency evidence drift')
 validate_host_imports(Path(m['variants']['host']['imports_ref']['path']).read_text(errors='replace'),Path(host['symbols_ref']['path']).read_text(errors='replace'),Path(host['link_map_ref']['path']).read_text(errors='replace'))
 return m
def host_test(path):
 m=verify_manifest(path);exe=m['variants']['host']['executable']['path'];child=subprocess.Popen([exe],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
 try:stdout,stderr=child.communicate()
 except BaseException:
  write_new(HERE/('host_test_interrupted.%04d.json'%(len(list(HERE.glob('host_test_interrupted.*.json')))+1)),dict(pid=child.pid,child_may_be_live=child.poll() is None,child_left_running=True));raise
 receipt=dict(schema='host-service-pure-logic-test/v1',build_ref=ref(path),returncode=child.returncode,stdout=stdout,stderr=stderr,CRT_runtime_clock_calls_instrumented=False,any_experiment_timing_executed=False,any_GPU_executed=False)
 write_new(HERE/('host_test.%04d.json'%(len(list(HERE.glob('host_test.*.json')))+1)),receipt)
 if child.returncode:raise ValueError('host logic tests failed')
 print(stdout)
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['verify','build','host-test']);ap.add_argument('--manifest',type=Path);a=ap.parse_args()
 if a.action=='build':build()
 elif a.action=='verify':print(json.dumps(verify_inputs()))
 else:
  if a.manifest is None:raise ValueError('explicit manifest required')
  host_test(a.manifest)
