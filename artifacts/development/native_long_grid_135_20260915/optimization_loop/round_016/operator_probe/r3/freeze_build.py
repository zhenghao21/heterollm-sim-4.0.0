"""One-shot freeze after compilation and host-only tests. Never runs CUDA."""
from pathlib import Path
import json,hashlib,datetime,re
P=Path(__file__).resolve().parent
out=P/'build_manifest.json'
if out.exists():raise SystemExit('Already frozen; create a new version')
def ref(p):
 p=Path(p).resolve()
 if not p.is_file():raise RuntimeError('Missing file '+str(p))
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
lock=json.loads((P/'identity_lock.json').read_text());files={}
for old in lock['files']:
 now=ref(old['path'])
 if now!=old:raise RuntimeError('Changed locked dependency '+old['path'])
 files[now['path']]=now
includes=set()
for line in (P/'compile.log').read_text(encoding='utf-8-sig').splitlines():
 m=re.search(r'(?:包含文件:|including file:)\s*(.*)$',line)
 if m:includes.add(str(Path(m.group(1).strip()).resolve()))
if len(includes)<50:raise RuntimeError('Incomplete /showIncludes evidence')
for p in includes:
 r=ref(p);files[r['path']]=r
for p in P.iterdir():
 if p.is_file() and p.suffix.lower() in ('.py','.ps1','.cpp','.h','.cmd','.json','.md','.log','.exe') and p.name not in ('build_manifest.json','READY.json','identity_verification.json'):
  r=ref(p);files[r['path']]=r
for p in [r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe',r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\link.exe',r'E:\cuda\lib\x64\cudart.lib']:
 r=ref(p);files[r['path']]=r
profiler=ref(r'F:\codex_project\37_LLMsim\tools\nsys_cli\target-windows-x64\nsys.exe');files[profiler['path']]=profiler
manifest={'schema':'operator-surface-probe-build/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'compiled_host_tested_not_gpu_executed','gpu_access':False,'timed_runs':0,'native_binary_modified':False,'native_bin':lock['native_bin'],'source_commit':lock['source_commit'],'source_runtime_equivalence_proven':False,'config_count':26,'compile_header_count':len(includes),'executable':ref(P/'operator-surface-probe.exe'),'profiler':profiler,'profiler_version':'2026.5.1.161-265138896106v0','files':sorted(files.values(),key=lambda r:r['path']),'limitations':['No actual GPU dispatch or numeric validation yet','No calibration emitted','Nsight kernel extraction remains a separately versioned root task']}
out.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'frozen_files':len(files),'compile_headers':len(includes),'sha256':manifest['executable']['sha256'],'gpu_access':False}))
