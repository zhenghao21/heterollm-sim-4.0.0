"""Bounded, append-only identity read experiment; no model inference or repair."""
import argparse
import ctypes as C
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import dual_hash as dh

HERE = Path(__file__).resolve().parent
R27 = HERE.parent
MAX_METADATA_BYTES = 2 * 1024**3


class ReadBudget:
    def __init__(self, limit=MAX_METADATA_BYTES): self.limit, self.used = limit, 0
    def read(self, path):
        path = Path(path)
        size = path.stat().st_size
        if size > 128*1024**2 or self.used+size > self.limit:
            raise ValueError('Metadata byte budget exhausted; no unbounded read')
        self.used += size
        with path.open('rb', buffering=0) as f:
            data = f.read(size)
        if len(data) != size or path.stat().st_size != size:
            raise ValueError('Metadata changed or incomplete')
        return data


def ref(path, budget):
    path = Path(path).resolve()
    data = budget.read(path)
    value = dh.openssl_sha256(); value.update(data)
    return {'path':str(path), 'bytes':len(data), 'sha256':value.hexdigest()}


def norm(value):
    return {'path':str(Path(value['path']).resolve()), 'bytes':value.get('bytes',value.get('size_bytes')),
            'sha256':value['sha256']}


def check_ref(value, budget):
    if ref(value['path'], budget) != norm(value): raise ValueError('Evidence identity changed: '+value['path'])
    return norm(value)


def document(path, budget): return json.loads(budget.read(path).decode('utf-8'))


def runtime_files():
    # Obtain the actually loaded OpenSSL DLL, not an arbitrary same-named file on PATH.
    kernel = C.WinDLL(str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/kernel32.dll'))
    kernel.GetModuleHandleW.argtypes=[C.c_wchar_p]; kernel.GetModuleHandleW.restype=C.c_void_p
    kernel.GetModuleFileNameW.argtypes=[C.c_void_p,C.c_wchar_p,C.c_uint32]; kernel.GetModuleFileNameW.restype=C.c_uint32
    handle=kernel.GetModuleHandleW('libcrypto-3-x64.dll')
    if not handle: raise RuntimeError('Loaded OpenSSL libcrypto identity unavailable; no fallback')
    buffer=C.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(handle,buffer,len(buffer)): raise RuntimeError('OpenSSL module path unavailable')
    system=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'
    return [Path(sys.executable),Path(dh._hashlib.__file__),Path(buffer.value),system/'bcrypt.dll',system/'bcryptprimitives.dll']


def crypto_environment():
    return {key:os.environ.get(key) for key in ('OPENSSL_ia32cap','OPENSSL_CONF','OPENSSL_MODULES')}


def verify_prepared(manifest_path,budget):
    m=document(manifest_path,budget)
    if m.get('schema')!='dual-hash-preparation/v1':raise ValueError('Preparation manifest missing')
    for item in m['files']+m['runtime_files']:check_ref(item,budget)
    p=document(HERE/'protocol.json',budget)
    if p['passes']!=['serial_01','parallel_01','parallel_02','parallel_03','parallel_04'] or p['parallel_workers']!=4:
        raise ValueError('Fixed pass budget differs')
    if p['maximum_model_read_bytes']!=5*p['target']['bytes'] or p['chunk_bytes']!=dh.MAX_CHUNK:
        raise ValueError('Fixed byte/chunk budget differs')
    if [str(x.resolve()).casefold() for x in runtime_files()]!=[str(Path(r['path']).resolve()).casefold() for r in m['runtime_files']]:
        raise ValueError('Loaded crypto runtime path differs')
    if crypto_environment()!=m['crypto_environment']:raise ValueError('Crypto process environment changed')
    return p,ref(manifest_path,budget)


def terminal_gate(protocol,budget):
    # Does not import R27's scoring/verification code or hash its GGUF files.
    for item in protocol['campaign_refs']:check_ref(item,budget)
    barrier_path=R27/'predictions_complete.json'
    data=budget.read(barrier_path)
    barrier=json.loads(data.decode('utf-8'))
    if (barrier.get('schema')!='r27-full262-terminal-barrier/v1' or barrier.get('terminal_count')!=262
        or set(barrier.get('arms',{}))!={'off','on'} or barrier.get('failures_preserved') is not True):
        raise ValueError('R27 full262 terminal barrier required, including preserved failures')
    controls=next(r for r in protocol['campaign_refs'] if Path(r['path']).name=='controls.json')
    if norm(barrier['controls_ref'])!=controls:raise ValueError('Barrier controls mismatch')
    counts={};read_refs=[]
    for arm in ('off','on'):
        locked=protocol['arms'][arm];actual=barrier['arms'][arm]
        if norm(actual['freeze_ref'])!=locked['freeze_ref']:raise ValueError('Barrier freeze mismatch')
        rows=actual['prediction_refs']
        if set(rows)!=set(locked['cell_ids']) or len(rows)!=131:raise ValueError('Incomplete terminal cell set')
        statuses={}
        for cell_id,item in rows.items():
            expected=(R27/arm/'predictions'/(cell_id+'.prediction.json')).resolve()
            if Path(item['path']).resolve()!=expected:raise ValueError('Prediction reference escaped frozen directory')
            payload=budget.read(expected)
            digest=dh.openssl_sha256();digest.update(payload)
            actual_ref={'path':str(expected),'bytes':len(payload),'sha256':digest.hexdigest()}
            if actual_ref!=norm(item):raise ValueError('Terminal prediction reference changed')
            row=json.loads(payload.decode('utf-8'))
            if (row.get('schema')!='stable-native-cell-prediction/v1' or row.get('cell_id')!=cell_id
                or row.get('status') not in ('predicted','failed','incomplete')
                or norm(row['freeze_ref'])!=locked['freeze_ref']
                or row.get('source_sha256')!=locked['source_sha256']
                or row.get('selection_sha256')!=locked['selection_sha256']):
                raise ValueError('Prediction is not a frozen terminal')
            if not row.get('finished_utc') or dh.datetime.datetime.fromisoformat(row['finished_utc'])>dh.datetime.datetime.fromisoformat(barrier['created_utc']):
                raise ValueError('Barrier predates terminal')
            statuses[row['status']]=statuses.get(row['status'],0)+1
            read_refs.append(actual_ref)
        counts[arm]=statuses
    digest=dh.openssl_sha256();digest.update(data)
    return {'barrier_ref':{'path':str(barrier_path.resolve()),'bytes':len(data),'sha256':digest.hexdigest()},
            'terminal_count':262,'status_counts_only':counts,'prediction_refs':read_refs,
            'model_metrics_read_for_scoring':False,'failed_results_preserved':True}


def conflicts(rows,current_pid=None):
    current_pid=os.getpid() if current_pid is None else current_pid
    bad=[]
    for row in rows:
        if row['pid']==current_pid:continue
        name=(row.get('name') or '').lower()
        args=row.get('cmdline');text=(' '.join(args) if isinstance(args,list) else args or '').replace('\\','/').lower()
        python=name.startswith(('python','pypy'))
        native=(name.startswith(('llama-','mmvq-','mmvq_','launch-recorder','host-submission-target'))
                or 'microbench' in name or ('probe' in name and name.endswith('.exe')) or name in ('nsys.exe','ncu.exe'))
        work=any(s in text for s in ('run_candidate.py','predict_stable_native_dataset.py','heterollm_sim',
                     'run_probe.py','run_wrapper_correctness','diagnostic.py run','--authorize-large-file-read'))
        project=python and '37_llmsim' in text
        known_host=python and not work and (('build.py' in text and any(x in text for x in (' build',' host-test',' verify')))
                     or ('pytest' in text and 'test_identity_diagnostic.py' in text))
        if native or (python and work) or (python and not args) or (project and not known_host):
            bad.append({'pid':row['pid'],'name':row.get('name'),'native_or_probe':native,
                        'project_or_execution_python':python and (work or project),'unknown_python':python and not args})
    return bad


def idle_snapshot():
    shell=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    command="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
    r=subprocess.run([str(shell),'-NoProfile','-NonInteractive','-Command',command],capture_output=True,
                     encoding='utf-8-sig',check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=json.loads(r.stdout);rows=[rows] if isinstance(rows,dict) else rows
    if not isinstance(rows,list) or not rows:raise ValueError('Process inventory unavailable')
    bad=conflicts([{'pid':x['ProcessId'],'name':x['Name'],'cmdline':x['CommandLine']} for x in rows])
    if bad:raise ValueError('Project not idle: '+json.dumps(bad))
    return {'utc':dh.now(),'inventory_count':len(rows),'conflicts':[],
            'continuous_exclusion_verified':False,'desktop_GPU_absence_required':False}


def memory_control(path, data=None, expected=None):
    if data is None:data=b'a'*1000000
    if expected is None:expected='cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'
    r={'schema':'dual-sha256-memory-control/v1','bytes':len(data),'expected_sha256':expected,
       'openssl_sha256':None,'cng_sha256':None,'status':'rejected','errors':[],'disk_input_bytes':0,
       'update_calls_per_engine':1,'fallback_used':False}
    try:
        left=dh.openssl_sha256();left.update(data);r['openssl_sha256']=left.hexdigest()
        with dh.CNGSHA256() as right:
            right.update(data);r['cng_sha256']=right.hexdigest()
        if r['openssl_sha256']!=expected or r['cng_sha256']!=expected:
            raise ValueError('Known memory vector mismatch; neither route accepted')
        r['status']='both_match_known_vector'
    except BaseException as error:r['errors'].append(type(error).__name__+': '+str(error))
    dh.write_new(path,r)
    if r['status']=='rejected':raise RuntimeError('Memory control failed; retained without model read')
    return r


def worker(protocol,output,event=None):
    if event is not None:event.wait()
    output=Path(output)
    # Exactly one 1,000,000-byte update in each independent engine per scheduled pass.
    memory_control(output.parent/(output.name+'.memory_control.json'))
    target=protocol['target']
    dh.scan_file(target['path'],output,target['bytes'],target['sha256'],chunk_bytes=protocol['chunk_bytes'])


def compare_chunks(root,labels,budget):
    logs=[[json.loads(line) for line in budget.read(Path(root)/label/'chunks.jsonl').decode('utf-8').splitlines()]
          for label in labels]
    differences=[]
    count=max(map(len,logs),default=0)
    for index in range(count):
        rows=[rows[index] if index<len(rows) else None for rows in logs]
        signature=lambda r:None if r is None else (r['index'],r['offset'],r['bytes'],r['openssl_sha256'],r['cng_sha256'])
        if any(signature(r)!=signature(rows[0]) for r in rows[1:]):
            differences.append({'index':index,'per_pass':dict(zip(labels,rows))})
    return {'compared_chunks':count,'differences':differences,'full_pass_consistency':not differences}


def execute(manifest_path):
    budget=ReadBudget();protocol,mref=verify_prepared(manifest_path,budget)
    output=HERE/'runs'/'read_comparison_0001';output.mkdir(parents=True,exist_ok=False)
    result={'schema':'dual-hash-comparison/v1','status':'rejected','started_utc':dh.now(),'errors':[],
            'model_read_upper_bound_bytes':0,'model_latency_measured':False,'cost_parameters':0,
            'repairs_or_retries':0,'old_failures_modified':False,'manifest_ref':mref,
            'pass_receipts':{},'children':[],'termination_requested':False}
    try:
        result['self_test']=dh.self_test()
        result['terminal_before']=terminal_gate(protocol,budget)
        result['idle_before']=idle_snapshot()
        dh.write_new(output/'start.json',{'utc':dh.now(),'manifest_ref':mref,'protocol':protocol,
             'terminal_barrier':result['terminal_before']['barrier_ref'],'serial_then_four_workers':True})
        result['model_read_upper_bound_bytes']+=protocol['target']['bytes']
        worker(protocol,output/'serial_01')
        serial=document(output/'serial_01/finish.json',budget)
        result['pass_receipts']['serial_01']=serial
        # Digest failures are retained; the predeclared parallel comparison is diagnostic, never a retry.
        if serial['bytes_read']!=protocol['target']['bytes'] or not serial['stat_unchanged']:
            raise ValueError('Serial read/stat incomplete; stop remaining model reads')
        result['terminal_between']=terminal_gate(protocol,budget)
        if result['terminal_between']!=result['terminal_before']:raise ValueError('R27 terminal evidence changed')
        result['idle_between']=idle_snapshot()
        if verify_prepared(manifest_path,budget)[1]!=mref:raise ValueError('Prepared identity changed')
        ctx=mp.get_context('spawn');event=ctx.Event();children=[]
        try:
            for label in protocol['passes'][1:]:
                child=ctx.Process(target=worker,args=(protocol,str(output/label),event),name=label)
                child.start();children.append((label,child))
                result['children'].append({'label':label,'pid':child.pid})
                result['model_read_upper_bound_bytes']+=protocol['target']['bytes']
        finally:
            # Release already-created children even if a later spawn failed. Never replace a failed worker.
            event.set()
            for label,child in children:child.join()
        for label,child in children:
            receipt=output/label/'finish.json'
            if child.exitcode!=0 or not receipt.exists():raise RuntimeError('Worker failed without complete receipt: '+label)
            result['pass_receipts'][label]=document(receipt,budget)
        if len(result['pass_receipts'])!=5:raise ValueError('Exactly five scheduled passes required')
        result['memory_controls']={label:document(output/(label+'.memory_control.json'),budget) for label in protocol['passes']}
        if any(r['status']!='both_match_known_vector' for r in result['memory_controls'].values()):raise ValueError('Memory vector rejected')
        result['chunk_comparison']=compare_chunks(output,protocol['passes'],budget)
        if any(r['status']!='agreed_expected_identity' for r in result['pass_receipts'].values()):
            result['errors'].append('At least one pass rejected; later agreement cannot repair it')
        if not result['chunk_comparison']['full_pass_consistency']:
            result['errors'].append('Serial/parallel chunks differ; entire comparison rejected')
    except BaseException as error:
        result['errors'].append(type(error).__name__+': '+str(error))
    finally:
        try:
            result['idle_after']=idle_snapshot()
            result['terminal_after']=terminal_gate(protocol,budget)
            if result.get('terminal_before')!=result['terminal_after']:raise ValueError('Terminal gate changed')
            if verify_prepared(manifest_path,budget)[1]!=mref:raise ValueError('Prepared evidence changed')
        except BaseException as error:result['errors'].append('post_gate: '+type(error).__name__+': '+str(error))
        result['metadata_read_bytes']=budget.used
        result['finished_utc']=dh.now()
        result['actual_model_read_bytes_reported']=sum(r['bytes_read'] for r in result['pass_receipts'].values())
        if not result['errors'] and len(result['pass_receipts'])==5:result['status']='five_pass_agreement_diagnostic_only'
        dh.write_new(output/'finish.json',result)
    if result['errors']:raise RuntimeError('Rejected diagnostic retained at '+str(output))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('self-test','run'))
    parser.add_argument('--manifest',type=Path,default=HERE/'preparation_manifest.0002.json')
    parser.add_argument('--authorize-large-file-read',action='store_true')
    a=parser.parse_args()
    if a.mode=='self-test':print(json.dumps(dh.self_test()));return
    if not a.authorize_large_file_read:raise ValueError('Explicit large-file execution switch required; preparation never reads model data')
    print(json.dumps(execute(a.manifest)))


if __name__=='__main__':main()
