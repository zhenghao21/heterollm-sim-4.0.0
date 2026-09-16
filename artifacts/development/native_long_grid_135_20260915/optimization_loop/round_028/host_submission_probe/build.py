"""Host-only build/verify/self-test. No target probe or GPU-run action."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
CUDA=Path('E:/cuda')
VS=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
VC=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64')
INCLUDES=(ROOT/'source/llama.cpp-semantic/ggml/include',CUDA/'include',CUDA/'extras/CUPTI/include')
LIB_DIRS=(ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/ggml/src',ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src',ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda',CUDA/'lib/x64')
SOURCES=('probe.cpp','launch_decode.h','identity_support.h','locked_identity.h','host_tests.cpp','build.py','protocol.json','analyze.py','run_probe.py','test_host_probe.py','README.md')
def ref(path):
 path=Path(path).resolve(strict=True)
 with path.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h}
def write_new(path,obj):
 with Path(path).open('x',encoding='utf-8') as stream:json.dump(obj,stream,indent=2,ensure_ascii=False)
def verify_inputs():
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 external=protocol['target_modules']+protocol['source_refs']+[protocol[k] for k in ('resource_usage_ref','resource_dump_tool','parent_identity_ref','hardware_ref','cupti_ref')]
 for expected in external:
  if ref(expected['path'])!=expected:raise ValueError('Pinned evidence changed: '+expected['path'])
 symbol=protocol['kernel_symbol']
 if ' Function '+symbol+':' not in (HERE/'target_resource_usage.txt').read_text():raise ValueError('Target symbol unavailable')
 libs=[]
 for name in ('ggml-base.lib','ggml-cuda.lib','cudart.lib'):
  path=next((d/name for d in LIB_DIRS if(d/name).is_file()),None)
  if path is None:raise ValueError('Missing library '+name)
  libs.append(ref(path))
 return {'sources':[ref(HERE/x) for x in SOURCES],'external':external,'libraries':libs,'toolchain':[ref(VS),ref(VC/'cl.exe'),ref(VC/'dumpbin.exe')]}
def checked(args,name):
 cmd=HERE/(name+'.cmd');log=HERE/(name+'.log')
 with cmd.open('x',encoding='utf-8') as stream:stream.write('@echo off\ncall '+subprocess.list2cmdline([str(VS)])+' >nul\nif errorlevel 1 exit /b %errorlevel%\n'+subprocess.list2cmdline([str(x) for x in args])+'\n')
 env=os.environ.copy();env['VSLANG']='1033'
 with log.open('xb') as out:r=subprocess.run(['cmd.exe','/c',str(cmd)],cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Build failed; preserve '+str(log))
 return {'command_ref':ref(cmd),'log_ref':ref(log),'returncode':r.returncode}
def build():
 before=verify_inputs();attempt=len(list(HERE.glob('compile.target.*.cmd')))+1
 common=[VC/'cl.exe','/nologo','/showIncludes','/std:c++17','/EHsc','/O2','/fp:strict','/MD','/utf-8',*('/I'+str(x) for x in INCLUDES)]
 target=HERE/('host-submission-target.%04d.exe'%attempt);host=HERE/('host-decoder-tests.%04d.exe'%attempt)
 a=checked(common+['/DGGML_SHARED','/DGGML_BACKEND_SHARED',HERE/'probe.cpp','/Fe:'+str(target),'/Fo:'+str(target.with_suffix('.obj')),'/link',*[x['path'] for x in before['libraries']],'bcrypt.lib','psapi.lib'],'compile.target.%04d'%attempt)
 b=checked(common+[HERE/'host_tests.cpp','/Fe:'+str(host),'/Fo:'+str(host.with_suffix('.obj'))],'compile.host.%04d'%attempt)
 imports=checked([VC/'dumpbin.exe','/nologo','/dependents',host],'imports.host.%04d'%attempt)
 dlls=re.findall(r'^\s*([A-Za-z0-9_.-]+\.dll)\s*$',Path(imports['log_ref']['path']).read_text(errors='replace'),re.M|re.I)
 if not dlls or any(re.search('cuda|cupti|ggml|nvidia',x,re.I) for x in dlls):raise ValueError('Host test GPU imports forbidden')
 targetimports=checked([VC/'dumpbin.exe','/nologo','/imports',target],'imports.target.%04d'%attempt)
 headers=set()
 for record in (a,b):
  for line in Path(record['log_ref']['path']).read_text(errors='replace').splitlines():
   match=re.search(r'([A-Za-z]:[\\/].+)$',line)
   if match and Path(match.group(1).strip()).is_file():headers.add(str(Path(match.group(1).strip()).resolve()))
 if len(headers)<30:raise ValueError('Incomplete compiler include evidence')
 if verify_inputs()!=before:raise ValueError('Build inputs changed')
 manifest={'schema':'host-submission-build/v1','inputs':before,'headers':[ref(x) for x in sorted(headers)],'target_executable':ref(target),'host_executable':ref(host),'host_imports':dlls,'records':[a,b,imports,targetimports],'target_executed':False,'GPU_executed':False,'target_DLL_rebuilt':False}
 path=HERE/('build_manifest.%04d.json'%attempt);write_new(path,manifest);print(json.dumps({'manifest':ref(path),'status':'compiled_only'}))
def host_test(path):
 manifest=json.loads(Path(path).read_text())
 if manifest['inputs']!=verify_inputs():raise ValueError('Build inputs differ')
 for expected in manifest['headers']+[manifest['host_executable'],manifest['target_executable']]:
  if ref(expected['path'])!=expected:raise ValueError('Built file changed')
 result=subprocess.run([manifest['host_executable']['path']],capture_output=True,text=True,cwd=HERE)
 receipt={'schema':'host-submission-host-test/v1','build_ref':ref(path),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'target_executed':False,'GPU_executed':False}
 write_new(HERE/('host_test.%04d.json'%(len(list(HERE.glob('host_test.*.json')))+1)),receipt)
 if result.returncode:raise RuntimeError('Host decoder tests failed')
 print(result.stdout)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('verify','build','host-test'));parser.add_argument('--manifest',type=Path);a=parser.parse_args()
 if a.mode=='build':build()
 elif a.mode=='verify':print(json.dumps({'status':'verified','inputs':verify_inputs(),'GPU_executed':False}))
 else:
  if a.manifest is None:raise ValueError('Explicit manifest required')
  host_test(a.manifest)
