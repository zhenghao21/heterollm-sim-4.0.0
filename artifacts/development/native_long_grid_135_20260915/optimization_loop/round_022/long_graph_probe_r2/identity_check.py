"""Identity-only wrapper; never runs --run, CUDA initialization or measurement."""
from pathlib import Path
import json,os,subprocess,hashlib,datetime
P=Path(__file__).resolve().parent

def main():
    m=json.loads((P/'build_manifest.json').read_text());p=json.loads((P/'protocol.json').read_text())
    for r in m['files']:
        q=Path(r['path'])
        with q.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if q.stat().st_size!=r['bytes'] or digest!=r['sha256']:raise ValueError('frozen identity changed: '+r['path'])
    env=os.environ.copy();env['PATH']=m['native_bin']+';E:\\cuda\\bin;'+env.get('PATH','')
    for key,value in {**p['runtime']['environment'],**{x:None for x in p['runtime']['extra_clear_environment']}}.items():
        if value is None:env.pop(key,None)
        else:env[key]=value
    result=subprocess.run([m['executable']['path'],'--identity-check'],cwd=P,env=env,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    doc={'schema':'r22-long-graph-identity-only/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':[m['executable']['path'],'--identity-check'],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'gpu_work_executed':False}
    with (P/'identity_only_result.json').open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2)
    print(json.dumps(doc));return result.returncode
if __name__=='__main__':raise SystemExit(main())
