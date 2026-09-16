"""Explicit root compile/host checks; GPU capture still requires separate authorization."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys
P=Path(__file__).resolve().parent
R=P.parent/'operator_probe/r3'
CUDA=Path('E:/cuda');CUPTI=CUDA/'extras/CUPTI'
ROOT=P.parents[5]
NATIVE=ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/bin'
MODERN_CUPTI=Path('F:/codex_project/37_LLMsim/tools/nsys_cli/target-windows-x64/cupti64_134.dll')
MODERN_API_VERSION=130401
VS=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
CL=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe')
INCLUDE_DIRS=[ROOT/'source/llama.cpp-semantic/ggml/include',CUDA/'include',CUPTI/'include']
LIB_DIRS=[ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/ggml/src',ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src',ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda',CUDA/'lib/x64']


def ref(path):
 path=Path(path).resolve(strict=True)
 with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}


def imported_libs():
 result=[]
 for name in ('ggml-base.lib','ggml-cuda.lib','cudart.lib'):
  found=next((p/name for p in LIB_DIRS if (p/name).is_file()),None)
  if found is None:raise ValueError('import library unavailable: '+name)
  result.append(found)
 return result


def checks():
 lock=json.loads((R/'identity_lock.json').read_text(encoding='utf-8'));refs=[]
 for old in lock['files']:
  if Path(old['path']).suffix.lower()=='.dll' or Path(old['path']).name in ('quantize.cu','mmq.cuh','mmq.cu','common.cuh','ggml-cuda.cu'):
   now=ref(old['path'])
   if now!=old:raise ValueError('locked input changed: '+old['path'])
   refs.append(now)
 for name,required in [('generated_cuda_runtime_api_meta.h',['cudaLaunchKernel_v7000_params','void **args','cudaStream_t stream']),('cupti_callbacks.h',['symbolName','functionParams','cuptiSubscribe']),('cupti_runtime_cbid.h',['CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000']),('cupti_version.h',['#define CUPTI_API_VERSION 26'])]:
  path=CUPTI/'include'/name;text=path.read_text(encoding='utf-8')
  if not all(word in text for word in required):raise ValueError('installed CUPTI header mismatch')
  refs.append(ref(path))
 refs.extend(ref(x) for x in (P/'launch_recorder.cpp',Path(__file__),MODERN_CUPTI,CL,VS,*imported_libs()))
 return refs


def write_new(path,value):
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inspect','compile','host-api-test','run']);ap.add_argument('--root-idle-confirmed',action='store_true');ap.add_argument('--root-gpu-authorized',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
 before=checks()
 if a.mode=='inspect':print(json.dumps({'status':'headers_identity_checked','files':len(before),'selected_CUPTI':ref(MODERN_CUPTI),'compile_header_api_version':26,'selected_runtime_api_version':MODERN_API_VERSION,'GPU_executed':False}));return
 if not a.root_idle_confirmed:raise ValueError('root idle confirmation required; no compile/run during matrix')
 env=os.environ.copy();env['PATH']=str(NATIVE)+os.pathsep+str(CUDA/'bin')+os.pathsep+env.get('PATH','')
 env.update(CAPTURE_LOCKED_NATIVE_BIN=str(NATIVE),CAPTURE_CUPTI_DLL=str(MODERN_CUPTI),CAPTURE_CUPTI_API_VERSION=str(MODERN_API_VERSION),GGML_CUDA_DISABLE_GRAPHS='1')
 for key in ('CUDA_INJECTION64_PATH','NVTX_INJECTION64_PATH','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS'):env.pop(key,None)
 exe=P/'launch-recorder.exe'
 if a.mode=='compile':
  if (P/'build_manifest.json').exists():raise ValueError('refuse existing manifested build; choose a new revision')
  attempt=1+len(list(P.glob('compile*.log')));exe=P/f'launch-recorder.attempt{attempt:04d}.exe';obj=P/f'launch-recorder.attempt{attempt:04d}.obj'
  if exe.exists() or obj.exists():raise ValueError('refuse existing attempt artifacts')
  command='"'+str(CL)+'" /nologo /showIncludes /std:c++17 /EHsc /O2 /fp:strict /MD /utf-8 /DGGML_SHARED /DGGML_BACKEND_SHARED '+ ' '.join('/I"'+str(x)+'"' for x in INCLUDE_DIRS)+' "'+str(P/'launch_recorder.cpp')+'" /Fe:"'+str(exe)+'" /Fo:"'+str(obj)+'" /link '+' '.join('"'+str(x)+'"' for x in imported_libs())
  text='@echo off\ncall "'+str(VS)+'" >nul\nif errorlevel 1 exit /b %errorlevel%\n'+command+'\n'
  cmd=P/f'compile.{attempt:04d}.cmd';compile_log=P/f'compile.{attempt:04d}.log'
  with cmd.open('x',encoding='utf-8') as f:f.write(text)
  with compile_log.open('xb') as out:r=subprocess.run(['cmd.exe','/c',str(cmd)],cwd=P,env=env,stdout=out,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('compile failed; preserve compile.log')
  if checks()!=before:raise ValueError('input changed during compile')
  headers=set()
  raw_log=compile_log.read_bytes()
  try:log_text=raw_log.decode('utf-8-sig')
  except UnicodeDecodeError:log_text=raw_log.decode('gb18030')
  for line in log_text.splitlines():
   match=re.search(r'^.*?:.*?:[ \t]+([A-Za-z]:[\\/].*)$',line) or re.search(r'including file:[ \t]+(.*)$',line)
   if match:headers.add(Path(match.group(1).strip()))
  if len(headers)<30:raise ValueError('compile header evidence incomplete')
  manifest={'schema':'conversion-launch-recorder-build/v2','inputs':before,'compile_headers':[ref(p) for p in sorted(headers)],'compile_header_count':len(headers),
   'compiler_command':command,'compile_log':ref(compile_log),'exe':ref(exe),'selected_CUPTI':ref(MODERN_CUPTI),
   'compile_CUPTI_header_api_version':26,'selected_runtime_api_version':MODERN_API_VERSION,'legacy_CUPTI_import_library_linked':False,
   'runtime_loading':'absolute LoadLibraryEx; exact selected path/version; required legacy callback exports resolved by name',
   'actual_GPU_callback_compatibility_validated':False,'gpu_executed':False,'source_built_kernel':False,'original_DLL_rebuilt':False}
  write_new(P/'build_manifest.json',manifest);print(json.dumps({'compiled':True,'headers':len(headers),'exe':manifest['exe'],'GPU_executed':False}));return
 manifest=json.loads((P/'build_manifest.json').read_text(encoding='utf-8'));exe=Path(manifest['exe']['path'])
 if manifest['inputs']!=before or manifest['exe']!=ref(exe):raise ValueError('build identity mismatch')
 for item in manifest['compile_headers']:
  if ref(item['path'])!=item:raise ValueError('compile header changed: '+item['path'])
 if a.mode=='host-api-test':
  argv=[str(exe),'--host-api-test'];out=P/'host_api_test.stdout.txt';err=P/'host_api_test.stderr.txt'
 else:
  if not a.root_gpu_authorized:raise ValueError('explicit root GPU authorization required')
  if a.output is None or a.output.exists():raise ValueError('explicit new output required')
  argv=[str(exe),'--run-recorder-only',str(a.output.resolve())];out=Path(str(a.output)+'.stdout.txt');err=Path(str(a.output)+'.stderr.txt')
 with out.open('xb') as stdout,err.open('xb') as stderr:r=subprocess.run(argv,cwd=P,env=env,stdout=stdout,stderr=stderr,check=False)
 if checks()!=before:raise ValueError('inputs changed during invocation')
 receipt={'argv':argv,'returncode':r.returncode,'stdout_ref':ref(out),'stderr_ref':ref(err),'build_manifest_ref':ref(P/'build_manifest.json'),'GPU_executed':a.mode=='run'}
 if a.mode=='host-api-test':
  if r.returncode==0:receipt['result']=json.loads(out.read_text(encoding='utf-8'))
  write_new(P/'host_api_test.json',receipt)
 else:write_new(Path(str(a.output)+'.receipt.json'),receipt)
 print(json.dumps(receipt,ensure_ascii=False))
 if r.returncode:raise RuntimeError('invocation failed status '+str(r.returncode))

if __name__=='__main__':main()
