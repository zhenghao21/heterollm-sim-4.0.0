"""Build only the CPU backend without OpenMP; retain all other annotation runtime files."""
from __future__ import annotations
import concurrent.futures,ctypes,datetime,hashlib,json,os,pathlib,shutil,subprocess,threading,time
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent/'llama.cpp-semantic';OLD=BASE/'build-semantic-direct';BUILD=ROOT/'build-native-thread-control';E=ROOT/'evidence'
M=json.loads((E/'source_manifest.json').read_text(encoding='utf-8'))
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def split(c):
 shell=ctypes.windll.shell32;shell.CommandLineToArgvW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)];shell.CommandLineToArgvW.restype=ctypes.POINTER(ctypes.c_wchar_p)
 n=ctypes.c_int();a=shell.CommandLineToArgvW(c,ctypes.byref(n))
 try:return [a[i] for i in range(n.value)]
 finally:ctypes.windll.kernel32.LocalFree(a)
def verify_protected():
 bad=[p for p,s in M['protected_sha256'].items() if digest(p)!=s]
 if bad:raise RuntimeError('Protected input changed: '+repr(bad))
 return True
verify_protected();BUILD.mkdir(exist_ok=True);(BUILD/'bin').mkdir(exist_ok=True)
for p in pathlib.Path(M['runtime_base']).iterdir():
 if p.is_file():shutil.copy2(p,BUILD/'bin'/p.name)
vs=pathlib.Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools');compiler=vs/'VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64'
vcvars=vs/'VC/Auxiliary/Build/vcvars64.bat';e=E/'capture_build_environment.cmd';e.write_text('@echo off\ncall "'+str(vcvars)+'" >nul\nset\n',encoding='utf-8')
raw=subprocess.check_output(['cmd.exe','/d','/c',str(e)],text=True,encoding='utf-8',errors='replace');env=dict(os.environ)
for line in raw.splitlines():
 k,sep,v=line.partition('=')
 if sep and k:env[k]=v
receipt={'schema':'native_thread_control_build_v1','status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime_base':M['runtime_base'],'runtime_output':str(BUILD/'bin'),'source_manifest_sha256':digest(E/'source_manifest.json'),'baseline_preverified':True,'steps':[],'readonly_input_sha256':{},'output_sha256':{},'compile_flags_removed':M['removed_flags'],'cuda_recompiled':False}
lock=threading.Lock()
def save():
 (E/'build_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
def run(label,args,logname):
 print(label,flush=True);t=time.monotonic();logpath=E/logname
 with logpath.open('w',encoding='utf-8') as f:
  f.write(subprocess.list2cmdline(args)+'\n');f.flush();p=subprocess.run(args,cwd=BUILD,env=env,stdout=f,stderr=subprocess.STDOUT)
 step={'label':label,'argv':args,'cwd':str(BUILD),'seconds':time.monotonic()-t,'returncode':p.returncode,'log':str(logpath),'log_sha256':digest(logpath)}
 with lock:receipt['steps'].append(step);save()
 if p.returncode:raise RuntimeError(label+' failed; see '+str(logpath))
 return step
entries=[]
for idx,x in enumerate(M['compile_units']):
 args=split(x['original_command']);removed=[];out=[]
 for a in args:
  if a=='-DGGML_USE_OPENMP' or a=='/DGGML_USE_OPENMP' or a.lower().startswith(('-openmp','/openmp')):
   removed.append(a);continue
  # PCH creates and consumers must use the new cache with the new macro set.
  a=a.replace(str(OLD),str(BUILD)).replace(OLD.as_posix(),BUILD.as_posix())
  if a.replace('\\','/').lower()==x['base_file'].replace('\\','/').lower():a=x['file']
  out.append(a)
 if not pathlib.Path(x['base_file']).is_relative_to(OLD):out.insert(1,'-I'+str(pathlib.Path(x['base_file']).parent))
 assert all('GGML_USE_OPENMP' not in a and not a.lower().startswith(('-openmp','/openmp')) for a in out)
 obj=BUILD/pathlib.Path(x['original_output']).relative_to(OLD);obj.parent.mkdir(parents=True,exist_ok=True)
 entries.append({'idx':idx,'x':x,'args':out,'obj':obj,'removed':removed,'pch':pathlib.Path(x['base_file']).name in ['cmake_pch.c','cmake_pch.cxx']})
 receipt['readonly_input_sha256'][x['base_file']]=digest(x['base_file'])
receipt['commands']=[{'file':e['x']['file'],'argv':e['args'],'removed':e['removed'],'output':str(e['obj'])} for e in entries];save()
def compile_one(e):
 run('compile '+pathlib.Path(e['x']['base_file']).name,e['args'],f"compile_{e['idx']:02d}.log")
 with lock:receipt['output_sha256'][str(e['obj'])]=digest(e['obj']);save()
for e in entries:
 if e['pch']:compile_one(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(compile_one,e) for e in entries if not e['pch']]
 for f in futures:f.result()
# All non-CPU libraries remain readonly; no CUDA compile or relink occurs.
base_lib=OLD/'ggml/src/ggml-base.lib';receipt['readonly_input_sha256'][str(base_lib)]=digest(base_lib)
intdir=BUILD/'ggml/src/CMakeFiles/ggml-cpu.dir';out=BUILD/'bin/ggml-cpu.dll';implib=BUILD/'ggml/src/ggml-cpu.lib'
linkargs=[*(str(e['obj']) for e in entries),str(base_lib),'kernel32.lib','user32.lib','gdi32.lib','winspool.lib','shell32.lib','ole32.lib','oleaut32.lib','uuid.lib','comdlg32.lib','advapi32.lib','/machine:x64','/INCREMENTAL:NO','/dll','/version:0.23','/out:'+str(out),'/implib:'+str(implib),'/pdb:'+str(out.with_suffix('.pdb'))]
rsp=E/'ggml-cpu.link.rsp';rsp.write_text('\n'.join(subprocess.list2cmdline([a]) for a in linkargs),encoding='utf-8')
cmake='C:/Users/A/AppData/Roaming/Python/Python312/site-packages/cmake/data/bin/cmake.exe';sdk=pathlib.Path('C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64')
run('link ggml-cpu.dll',[cmake,'-E','vs_link_dll','--intdir='+str(intdir),'--rc='+str(sdk/'rc.exe'),'--mt='+str(sdk/'mt.exe'),'--manifests','--',str(compiler/'link.exe'),'/nologo','@'+str(rsp)],'link.log')
run('inspect CPU imports',[str(compiler/'dumpbin.exe'),'/imports',str(out)],'cpu_imports.log')
imports=(E/'cpu_imports.log').read_text(encoding='utf-8').lower();assert 'vcomp' not in imports and 'libomp' not in imports and '__kmpc' not in imports,'OpenMP import remains'
run('startup version',[str(BUILD/'bin/llama-server.exe'),'--version'],'version_smoke.log')
receipt['baseline_postverified']=verify_protected()
receipt['unchanged_runtime_sha256']={}
for p in pathlib.Path(M['runtime_base']).iterdir():
 if p.is_file() and p.name!='ggml-cpu.dll':
  copied=BUILD/'bin'/p.name;assert digest(p)==digest(copied),p
  receipt['unchanged_runtime_sha256'][str(copied)]=digest(copied)
receipt['no_openmp_compile_flags']=True;receipt['no_openmp_imports']=True
receipt['old_cpu_dll_sha256']=digest(pathlib.Path(M['runtime_base'])/'ggml-cpu.dll');receipt['new_cpu_dll_sha256']=digest(out)
receipt['output_sha256'].update({str(p):digest(p) for p in (BUILD/'bin').iterdir() if p.is_file()})
receipt['build_script_sha256']=digest(pathlib.Path(__file__));receipt['status']='complete';receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
(E/'progress.json').write_text(json.dumps({'status':'complete','binary':str(BUILD/'bin/llama-server.exe'),'cpu_dll_sha256':digest(out),'all_other_runtime_files_unchanged':True,'no_openmp_imports':True,'performance_measurement_run':False},indent=2),encoding='utf-8')
print('COMPLETE '+str(BUILD/'bin/llama-server.exe'),flush=True)
