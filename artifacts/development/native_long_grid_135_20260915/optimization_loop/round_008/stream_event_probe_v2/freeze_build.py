"""Freeze a compiled v2 probe and all execution/extraction inputs; no GPU access."""
from pathlib import Path
import datetime,hashlib,json
p=Path(__file__).resolve().parent
out=p/'build_manifest.json'
if out.exists():raise SystemExit('Manifest already frozen; make a new version instead')
def ref(path):
    path=Path(path).resolve()
    if not path.is_file():raise RuntimeError('Missing identity input: '+str(path))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
lock=json.loads((p/'identity_lock.json').read_text())
files=[]
for old in lock['files']:
    current=ref(old['path'])
    if current['sha256']!=old['sha256'] or current['bytes']!=old['bytes']:raise RuntimeError('Dependency changed: '+old['path'])
    files.append(current)
for name in ['stream_event_probe.cpp','identity_lock.h','identity_lock.json','protocol.json','invoke.ps1','run_frozen_matrix.ps1','assess.py','full_raw_audit.py','summarize_matrix.py','test_quality_tools.py','freeze_build.py','compile.cmd','README.md','copied_source_provenance.json','static_validation.json','check_environment.ps1','environment_validation.json','environment_echo.cpp','environment-echo.exe','test_quality_tools.log','powershell_validation.json','compile.log','stream-event-probe.exe']:
    value=ref(p/name)
    if not any(x['path']==value['path'] for x in files):files.append(value)
manifest={'schema':'stream-event-probe-build/v2','built_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'compiled_not_executed','gpu_access':False,'timed_runs':0,'native_binary_modified':False,'native_bin':lock['native_bin'],'native_source_commit':lock['source_commit'],'executable':ref(p/'stream-event-probe.exe'),'compile_log':ref(p/'compile.log'),'internal_abi':{'pointer_bits':64,'sizeof':16,'alignof':8,'device_offset':0,'context_offset':8},'llm_actual_inputs':[],'files':files,'readiness':'identity/static validation complete; GPU runtime and batch semantics remain unmeasured'}
out.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps({'frozen_files':len(files),'executable_sha256':manifest['executable']['sha256'],'gpu_access':False}))
