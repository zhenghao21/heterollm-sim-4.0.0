"""Root-authorized compile and host checks; only root may invoke the GPU run mode."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys
import analyze_replay
P=Path(__file__).resolve().parent
CAPTURE=P.parent
R=CAPTURE.parent/'operator_probe/r3'
ROOT=CAPTURE.parents[5]
CUDA=Path('E:/cuda');CUPTI=CUDA/'extras/CUPTI'
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
   if now!=old:raise ValueError('original native/source lock changed: '+old['path'])
   refs.append(now)
 origin=json.loads((P/'origin_lock.json').read_text(encoding='utf-8'))
 for old in origin['files']:
  if ref(old['path'])!=old:raise ValueError('frozen recorder/baseline source changed: '+old['path'])
  refs.append(old)
 for name,required in [('generated_cuda_runtime_api_meta.h',['cudaLaunchKernel_v7000_params','const void *func','void **args','cudaStream_t stream']),('cupti_callbacks.h',['symbolName','functionParams','cuptiSubscribe']),('cupti_runtime_cbid.h',['CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000']),('cupti_version.h',['#define CUPTI_API_VERSION 26'])]:
  path=CUPTI/'include'/name;text=path.read_text(encoding='utf-8')
  if not all(word in text for word in required):raise ValueError('installed CUPTI header mismatch: '+name)
  refs.append(ref(path))
 runtime_header=CUDA/'include/cuda_runtime_api.h';text=runtime_header.read_text(encoding='utf-8')
 for exact in ('cudaLaunchKernel(const void *func','cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)'):
  if exact not in text:raise ValueError('runtime func API signature changed: '+exact)
 refs.extend(ref(x) for x in (runtime_header,P/'launch_replay.cpp',P/'analyze_replay.py',P/'sample_baseline.json',P/'origin_lock.json',Path(__file__),MODERN_CUPTI,CL,VS,*imported_libs()))
 return refs

def write_new(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def run_environment():
 env=os.environ.copy();env['PATH']=str(NATIVE)+os.pathsep+str(CUDA/'bin')+os.pathsep+env.get('PATH','')
 env.update(CAPTURE_LOCKED_NATIVE_BIN=str(NATIVE),CAPTURE_CUPTI_DLL=str(MODERN_CUPTI),CAPTURE_CUPTI_API_VERSION=str(MODERN_API_VERSION),GGML_CUDA_DISABLE_GRAPHS='1')
 for key in ('CUDA_INJECTION64_PATH','NVTX_INJECTION64_PATH','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS'):env.pop(key,None)
 return env

def compile_build(before,env):
 if (P/'build_manifest.json').exists():raise ValueError('refuse existing manifested build; use a new revision')
 attempt=1+len(list(P.glob('compile.*.log')))
 exe=P/f'launch-replay.attempt{attempt:04d}.exe';obj=P/f'launch-replay.attempt{attempt:04d}.obj'
 if exe.exists() or obj.exists():raise ValueError('refuse existing attempt artifacts')
 command='"'+str(CL)+'" /nologo /showIncludes /std:c++17 /EHsc /O2 /fp:strict /MD /utf-8 /DGGML_SHARED /DGGML_BACKEND_SHARED '+ ' '.join('/I"'+str(x)+'"' for x in INCLUDE_DIRS)+' "'+str(P/'launch_replay.cpp')+'" /Fe:"'+str(exe)+'" /Fo:"'+str(obj)+'" /link '+' '.join('"'+str(x)+'"' for x in imported_libs())
 text='@echo off\ncall "'+str(VS)+'" >nul\nif errorlevel 1 exit /b %errorlevel%\n'+command+'\n'
 cmd=P/f'compile.{attempt:04d}.cmd';compile_log=P/f'compile.{attempt:04d}.log'
 with cmd.open('x',encoding='utf-8') as f:f.write(text)
 # Freeze each attempted source and command, including a failed attempt.
 with (P/f'compile.{attempt:04d}.source.cpp').open('xb') as f:f.write((P/'launch_replay.cpp').read_bytes())
 with compile_log.open('xb') as out:r=subprocess.run(['cmd.exe','/c',str(cmd)],cwd=P,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=180)
 if r.returncode:raise RuntimeError('compile failed; preserved '+str(compile_log))
 if checks()!=before:raise ValueError('input changed during compile')
 raw=compile_log.read_bytes()
 try:text=raw.decode('utf-8-sig')
 except UnicodeDecodeError:text=raw.decode('gb18030')
 headers=set()
 for line in text.splitlines():
  match=re.search(r'^.*?:.*?:[ \t]+([A-Za-z]:[\\/].*)$',line) or re.search(r'including file:[ \t]+(.*)$',line)
  if match:headers.add(Path(match.group(1).strip()))
 if len(headers)<30:raise ValueError('compile header evidence incomplete')
 manifest={'schema':'original-runtime-func-owned-replay-build/v1','inputs':before,'compile_headers':[ref(h) for h in sorted(headers)],'compile_header_count':len(headers),'compiler_command':command,'compile_log':ref(compile_log),'attempt_source':ref(P/f'compile.{attempt:04d}.source.cpp'),'exe':ref(exe),'selected_CUPTI':ref(MODERN_CUPTI),'compile_CUPTI_header_api_version':26,'selected_runtime_api_version':MODERN_API_VERSION,'legacy_CUPTI_import_library_linked':False,'runtime_loading':'absolute LoadLibraryEx; verified native DLL paths; runtime func from official CUDA runtime callback params','actual_GPU_replay_executed':False,'GPU_executed':False,'source_built_kernel':False,'original_DLL_rebuilt':False}
 write_new(P/'build_manifest.json',manifest)
 print(json.dumps({'compiled':True,'headers':len(headers),'exe':manifest['exe'],'GPU_executed':False}))

def verify_build(before):
 manifest=json.loads((P/'build_manifest.json').read_text(encoding='utf-8'));exe=Path(manifest['exe']['path'])
 if manifest['inputs']!=before or manifest['exe']!=ref(exe):raise ValueError('build identity mismatch')
 for item in manifest['compile_headers']:
  if ref(item['path'])!=item:raise ValueError('compile header changed: '+item['path'])
 return manifest,exe

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inspect','compile','host-api-test','run']);ap.add_argument('--root-idle-confirmed',action='store_true');ap.add_argument('--root-gpu-authorized',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
 before=checks()
 if a.mode=='inspect':print(json.dumps({'status':'frozen_original_and_API_signatures_checked','files':len(before),'selected_CUPTI':ref(MODERN_CUPTI),'GPU_executed':False}));return
 if not a.root_idle_confirmed:raise ValueError('explicit root idle confirmation required')
 env=run_environment()
 if a.mode=='compile':compile_build(before,env);return
 manifest,exe=verify_build(before)
 if a.mode=='host-api-test':
  argv=[str(exe),'--host-api-test'];out=P/'host_api_test.stdout.txt';err=P/'host_api_test.stderr.txt';receipt_path=P/'host_api_test.json'
  if any(x.exists() for x in (out,err,receipt_path)):raise ValueError('refuse existing host test artifacts')
 else:
  if not a.root_gpu_authorized:raise ValueError('GPU may be run only by root with explicit root authorization')
  if a.output is None:raise ValueError('explicit new result output required')
  result_path=a.output.resolve()
  if result_path.parent!=P:raise ValueError('replay_r1 owns output; choose a direct child here')
  out=Path(str(result_path)+'.stdout.txt');err=Path(str(result_path)+'.stderr.txt');receipt_path=Path(str(result_path)+'.receipt.json')
  all_outputs=[result_path,out,err,receipt_path,Path(str(result_path)+'.analysis.json'),*(Path(str(result_path)+s) for s in analyze_replay.SUFFIXES.values())]
  if any(x.exists() for x in all_outputs):raise ValueError('refuse existing result or sidecar file; select new output')
  argv=[str(exe),'--run-replay-only',str(result_path)]
 with out.open('xb') as stdout,err.open('xb') as stderr:r=subprocess.run(argv,cwd=P,env=env,stdout=stdout,stderr=stderr,check=False,timeout=180)
 receipt={'schema':'original-runtime-func-replay-invocation/v1','argv':argv,'returncode':r.returncode,'stdout_ref':ref(out),'stderr_ref':ref(err),'build_manifest_ref':ref(P/'build_manifest.json'),'GPU_executed':a.mode=='run','no_original_recorder_files_modified':True}
 error=None
 try:
  if checks()!=before:raise ValueError('inputs changed during invocation')
  verify_build(before)
  if a.mode=='host-api-test':
   if r.returncode:raise ValueError('C++ host test failed')
   receipt['result']=json.loads(out.read_text(encoding='utf-8'));receipt['python_decoder_host_test']=analyze_replay.host_test()
  else:
   if result_path.exists():receipt['result_ref']=ref(result_path)
   receipt['raw_artifacts']={key:ref(Path(str(result_path)+suffix)) for key,suffix in analyze_replay.SUFFIXES.items() if Path(str(result_path)+suffix).exists()}
   if r.returncode:raise ValueError('GPU diagnostic executable failed; preserve result and raw files')
   report=analyze_replay.analyze(result_path);analysis_path=Path(str(result_path)+'.analysis.json');write_new(analysis_path,report)
   receipt['analysis_ref']=ref(analysis_path)
   receipt['analysis_summary']={key:report[key] for key in ('target_row11_k33','sample_count','max_abs_prediction_minus_same_process_graph','max_abs_prediction_minus_archived_actual','all_replay_predictions_match_same_process_graph_under_frozen_tolerance','all_same_process_graph_samples_equal_archived_actual','same_process_original_path_failed_samples')}
 except Exception as e:error=str(e);receipt['verification_error']=error
 write_new(receipt_path,receipt);print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
 if error or r.returncode:raise RuntimeError(error or ('invocation failed '+str(r.returncode)))
if __name__=='__main__':main()
