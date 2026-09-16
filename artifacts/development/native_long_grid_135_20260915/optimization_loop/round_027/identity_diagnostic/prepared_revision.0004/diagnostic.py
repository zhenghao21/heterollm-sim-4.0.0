"""Bounded, append-only identity read experiment; no model inference or repair."""
import argparse
import ctypes as C
import json
import multiprocessing as mp
import os
import re
import shlex
from pathlib import Path
import subprocess
import sys
import dual_hash as dh

HERE = Path(__file__).resolve().parent
R27 = HERE.parent
MAX_METADATA_BYTES = 2 * 1024**3  # Small preparation/bootstrap only; execution uses checked formula.
MANIFEST_MAX_BYTES = 64 * 1024
PROTOCOL_MAX_BYTES = 1024 * 1024
PASS_RECEIPT_MAX_BYTES = 1024 * 1024
MEMORY_RECEIPT_MAX_BYTES = 64 * 1024


class ReadBudget:
    def __init__(self, limit=MAX_METADATA_BYTES): self.limit, self.used = limit, 0
    def read(self, path, *, max_bytes=128*1024**2):
        path = Path(path)
        size = path.stat().st_size
        if size > max_bytes or self.used+size > self.limit:
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


def document(path, budget, *, max_bytes=128*1024**2): return json.loads(budget.read(path,max_bytes=max_bytes).decode('utf-8'))


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


def _length(value, name):
    if type(value) is not int or value < 0:raise ValueError('Invalid byte count: '+name)
    return value


def chunk_log_upper_bound(target_bytes,chunk_bytes):
    target_bytes=_length(target_bytes,'target')
    if type(chunk_bytes) is not int or not 1<=chunk_bytes<=dh.MAX_CHUNK:raise ValueError('Invalid chunk bound')
    count=(target_bytes+chunk_bytes-1)//chunk_bytes
    # False is one byte longer than true. Maximal decimal index/offset/length
    # widths cover every emitted successful or disagreement line, including LF.
    row={'index':max(0,count-1),'offset':target_bytes,'bytes':chunk_bytes,
         'openssl_sha256':'f'*64,'cng_sha256':'f'*64,'equal':False}
    line_bytes=len((json.dumps(row)+'\n').encode('utf8'))
    return {'chunks_per_pass':count,'maximum_chunk_line_bytes':line_bytes,
            'maximum_log_bytes_per_pass':count*line_bytes}


def metadata_budget_formula(*,campaign_lengths,barrier_bytes,prediction_lengths,
                            source_lengths,runtime_lengths,historical_lengths,
                            target_bytes,chunk_bytes):
    """Pure arithmetic. Lengths budget work; they never replace full byte hashes."""
    if len(prediction_lengths)!=262:raise ValueError('Full262 prediction lengths required')
    groups={'campaign':campaign_lengths,'predictions':prediction_lengths,'sources':source_lengths,
            'runtime':runtime_lengths,'history':historical_lengths}
    totals={name:sum(_length(v,name) for v in values) for name,values in groups.items()}
    per_terminal=totals['campaign']+_length(barrier_bytes,'barrier')+totals['predictions']
    # verify_prepared: source/runtime/history once, selected protocol twice
    # (in manifest file refs + document), selected manifest twice (load + ref).
    per_verify=totals['sources']+totals['runtime']+totals['history']+2*PROTOCOL_MAX_BYTES+2*MANIFEST_MAX_BYTES
    chunks=chunk_log_upper_bound(target_bytes,chunk_bytes)
    terms={
      'three_full262_terminal_checks':3*per_terminal,
      'three_program_runtime_protocol_manifest_checks':3*per_verify,
      'one_budget_precheck_manifest_read':MANIFEST_MAX_BYTES,
      'five_new_chunk_logs':5*chunks['maximum_log_bytes_per_pass'],
      'five_pass_receipts':5*PASS_RECEIPT_MAX_BYTES,
      'five_memory_controls':5*MEMORY_RECEIPT_MAX_BYTES,
    }
    upper=sum(terms.values());rounding=1024**2
    return {'schema':'dual-hash-metadata-read-budget/v1','terminal_check_count':3,'verify_prepared_count':3,
      'prediction_reference_count':262,'per_terminal_bytes':per_terminal,'per_verify_upper_bound_bytes':per_verify,
      'exact_frozen_length_totals':totals,'barrier_bytes':barrier_bytes,'chunk_log_bound':chunks,
      'self_reference_caps':{'manifest_bytes':MANIFEST_MAX_BYTES,'protocol_bytes':PROTOCOL_MAX_BYTES},
      'new_receipt_caps':{'pass_finish_bytes':PASS_RECEIPT_MAX_BYTES,'memory_control_bytes':MEMORY_RECEIPT_MAX_BYTES},
      'terms':terms,'required_metadata_read_upper_bound_bytes':upper,
      'declared_metadata_read_limit_bytes':((upper+rounding-1)//rounding)*rounding,
      'rounding_bytes':rounding,'all_required_full_content_checks_retained':True,
      'length_precheck_is_identity_evidence':False,'stat_substituted_for_full_hash':False}


def planned_metadata_budget(protocol,manifest):
    snapshot=protocol['frozen_terminal_snapshot']
    protocol_name=protocol['protocol_filename']
    protocol_refs=[r for r in manifest['files'] if Path(r['path']).name==protocol_name]
    if len(protocol_refs)!=1:raise ValueError('Selected protocol must occur exactly once in frozen files')
    rows=[r for arm in ('off','on') for r in snapshot['arms'][arm]['prediction_refs'].values()]
    return metadata_budget_formula(campaign_lengths=[r['bytes'] for r in protocol['campaign_refs']],
      barrier_bytes=snapshot['barrier_ref']['bytes'],prediction_lengths=[r['bytes'] for r in rows],
      source_lengths=[r['bytes'] for r in manifest['files'] if Path(r['path']).name!=protocol_name],
      runtime_lengths=[r['bytes'] for r in manifest['runtime_files']],
      historical_lengths=[r['bytes'] for r in protocol['prior_attempt']['evidence_refs']],
      target_bytes=protocol['target']['bytes'],chunk_bytes=protocol['chunk_bytes'])


def configure_execution_budget(protocol,manifest_path,budget):
    """Fail before model reads if declared three-phase metadata work cannot fit."""
    manifest=document(manifest_path,budget,max_bytes=MANIFEST_MAX_BYTES)
    if protocol.get('schema')!='dual-hash-diagnostic-protocol/v2':raise ValueError('New budgeted protocol required; legacy failed batches cannot be resumed')
    prior=protocol.get('prior_attempt',{})
    if (prior.get('status')!='rejected' or prior.get('reported_passes')!=1
        or prior.get('reported_model_read_bytes')!=protocol['target']['bytes']
        or prior.get('serial_reused_in_new_batch') is not False
        or protocol.get('maximum_model_read_bytes')!=5*protocol['target']['bytes']
        or protocol.get('cumulative_model_read_upper_bound_bytes')!=6*protocol['target']['bytes']):
        raise ValueError('New five-pass/prior one-pass cumulative budget or history changed')
    expected=planned_metadata_budget(protocol,manifest)
    if protocol.get('metadata_budget_plan')!=expected or protocol.get('maximum_metadata_read_bytes')!=expected['declared_metadata_read_limit_bytes']:
        raise ValueError('Declared auxiliary byte budget does not match frozen full-check formula')
    selected=HERE/protocol['protocol_filename']
    if selected.stat().st_size>PROTOCOL_MAX_BYTES or Path(manifest_path).stat().st_size>MANIFEST_MAX_BYTES:
        raise ValueError('Self-reference byte cap exceeded')
    snapshot=protocol['frozen_terminal_snapshot']
    refs=protocol['campaign_refs']+[snapshot['barrier_ref']]+[
        r for arm in ('off','on') for r in snapshot['arms'][arm]['prediction_refs'].values()]
    refs+=manifest['files']+manifest['runtime_files']+protocol['prior_attempt']['evidence_refs']
    target=Path(protocol['target']['path']).resolve()
    for r in refs:
        path=Path(r['path']).resolve()
        if path==target:raise ValueError('Model content must never enter auxiliary read budget')
        if path.stat().st_size!=r['bytes']:raise ValueError('Frozen reference size changed; no budget expansion: '+str(path))
    if budget.used>expected['declared_metadata_read_limit_bytes']:raise ValueError('Preparation already exceeded declared execution budget')
    budget.limit=expected['declared_metadata_read_limit_bytes']
    return expected


def verify_prepared(manifest_path,budget):
    m=document(manifest_path,budget)
    if m.get('schema')!='dual-hash-preparation/v1':raise ValueError('Preparation manifest missing')
    for item in m['files']+m['runtime_files']:check_ref(item,budget)
    protocol_path=Path(m['protocol_ref']['path']) if m.get('revision')=='0004' else HERE/'protocol.json'
    if m.get('revision')=='0004' and protocol_path.resolve()!= (HERE/'protocol.0004.json').resolve():raise ValueError('Selected protocol path differs')
    p=document(protocol_path,budget,max_bytes=PROTOCOL_MAX_BYTES)
    if m.get('revision')=='0004':
        if norm(m['protocol_ref']) not in [norm(r) for r in m['files']]:raise ValueError('Protocol ref not in frozen code closure')
        for item in p['prior_attempt']['evidence_refs']:check_ref(item,budget)
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
    if protocol.get('frozen_terminal_snapshot'):
        expected_snapshot=protocol['frozen_terminal_snapshot'];digest=dh.openssl_sha256();digest.update(data)
        actual_barrier={'path':str(barrier_path.resolve()),'bytes':len(data),'sha256':digest.hexdigest()}
        if actual_barrier!=expected_snapshot['barrier_ref'] or barrier.get('arms')!=expected_snapshot['arms']:
            raise ValueError('Frozen full262 snapshot changed; declared budget and identity invalid')
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


def process_classification(row):
    name=(row.get('name') or '').lower()
    args=row.get('cmdline')
    if isinstance(args,str):
        try: args=[x.strip('"') for x in shlex.split(args,posix=False)]
        except ValueError: args=[]
    args=args or []
    tokens=[str(x).replace('\\','/').lower() for x in args]
    text=' '.join(tokens)
    python=name.startswith(('python','pypy'))
    if (name=='gpu-operator-timing.exe' or name.startswith(('llama-','mmvq-','mmvq_','launch-recorder','host-submission-target'))
        or 'microbench' in name or ('probe' in name and name.endswith('.exe')) or name in ('nsys.exe','ncu.exe')):
        return 'blocked_native'
    if not python:return 'other'
    if '--multiprocessing-fork' in tokens or 'multiprocessing.spawn' in text:return 'spawn_requires_parent'
    if not tokens:return 'unknown_python'
    scripts=[(i,x.rsplit('/',1)[-1]) for i,x in enumerate(tokens) if x.endswith('.py')]
    for i,script in scripts:
        mode=tokens[i+1] if i+1<len(tokens) else ''
        if script in ('runner.py','diagnostic.py') and mode in ('run','resume','extend','evaluate'):
            return 'blocked_execution'
    if any(x in text for x in ('run_candidate.py','predict_stable_native_dataset.py','heterollm_sim',
                              'run_probe.py','run_wrapper_correctness','--authorize-large-file-read')):
        return 'blocked_project'
    # A complete explicit command is required for host-only exemptions; never trust '-c' snippets.
    if '-c' not in tokens:
        if any(script=='diagnostic.py' and i+1<len(tokens) and tokens[i+1]=='self-test' for i,script in scripts):
            return 'allowed_host'
        if any(script=='build.py' and i+1<len(tokens) and tokens[i+1] in ('verify','build','host-test','check') for i,script in scripts):
            return 'allowed_host'
        tests=[script for i,script in scripts if script.startswith('test_')]
        if ('pytest' in tokens or any(x.endswith('/pytest.exe') for x in tokens)) and tests and set(tests)<={'test_identity_diagnostic.py','test_host_probe.py'}:
            return 'allowed_host'
    return 'unknown_project_python' if '37_llmsim' in text or 'gpu_operator_timing' in text else 'other_python'


def conflicts(rows,current_pid=None):
    current_pid=os.getpid() if current_pid is None else current_pid
    by_pid={row['pid']:row for row in rows};bad=[]
    def classify(pid,seen):
        if pid in seen:return 'unknown_parent_cycle'
        row=by_pid.get(pid)
        if row is None:return 'unknown_parent_missing'
        kind=process_classification(row)
        if kind!='spawn_requires_parent':return kind
        parent=row.get('ppid')
        if not parent:return 'unknown_spawn_parent'
        owner=classify(parent,seen|{pid})
        if owner=='allowed_host':return 'allowed_host_spawn'
        if owner=='allowed_host_spawn':return owner
        return 'blocked_spawn_parent' if owner.startswith('blocked_') else 'unknown_spawn_owner'
    for row in rows:
        if row['pid']==current_pid:continue
        kind=classify(row['pid'],set())
        if kind.startswith(('blocked_','unknown_')):
            bad.append({'pid':row['pid'],'ppid':row.get('ppid'),'name':row.get('name'),'reason':kind})
    return bad


def idle_snapshot():
    shell=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    command="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
    r=subprocess.run([str(shell),'-NoProfile','-NonInteractive','-Command',command],capture_output=True,
                     encoding='utf-8-sig',check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=json.loads(r.stdout);rows=[rows] if isinstance(rows,dict) else rows
    if not isinstance(rows,list) or not rows:raise ValueError('Process inventory unavailable')
    bad=conflicts([{'pid':x['ProcessId'],'ppid':x['ParentProcessId'],'name':x['Name'],'cmdline':x['CommandLine']} for x in rows])
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
    except BaseException as error:
        r['errors'].append(type(error).__name__+': '+str(error))
        r['exception_evidence']=dh.error_record(error)
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


def compare_chunks(root,labels,budget,protocol_target_bytes=None):
    logs=[[json.loads(line) for line in budget.read(Path(root)/label/'chunks.jsonl',max_bytes=chunk_log_upper_bound(protocol_target_bytes,dh.MAX_CHUNK)['maximum_log_bytes_per_pass'] if protocol_target_bytes is not None else 128*1024**2).decode('utf-8').splitlines()]
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
    calculated_budget=configure_execution_budget(protocol,manifest_path,budget)
    if protocol.get('run_directory')!='read_comparison_0002':raise ValueError('Fresh exclusive0002 output required')
    output=HERE/'runs'/protocol['run_directory'];output.mkdir(parents=True,exist_ok=False)
    result={'schema':'dual-hash-comparison/v1','status':'rejected','started_utc':dh.now(),'errors':[],
            'model_read_upper_bound_bytes':0,'model_latency_measured':False,'cost_parameters':0,
            'repairs_or_retries':0,'old_failures_modified':False,'manifest_ref':mref,
            'pass_receipts':{},'children':[],'termination_requested':False,
            'metadata_budget_precheck':calculated_budget,'prior_attempt':protocol['prior_attempt'],
            'prior_attempt_remains_rejected':True,'prior_serial_reused':False,
            'cumulative_model_read_upper_bound_bytes':protocol['prior_attempt']['reported_model_read_bytes'],
            'new_batch_model_read_limit_bytes':protocol['maximum_model_read_bytes']}
    try:
        result['self_test']=dh.self_test()
        result['terminal_before']=terminal_gate(protocol,budget)
        result['idle_before']=idle_snapshot()
        dh.write_new(output/'start.json',{'utc':dh.now(),'manifest_ref':mref,'protocol':protocol,
             'terminal_barrier':result['terminal_before']['barrier_ref'],'serial_then_four_workers':True})
        result['model_read_upper_bound_bytes']+=protocol['target']['bytes']
        worker(protocol,output/'serial_01')
        serial=document(output/'serial_01/finish.json',budget,max_bytes=PASS_RECEIPT_MAX_BYTES)
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
            result['pass_receipts'][label]=document(receipt,budget,max_bytes=PASS_RECEIPT_MAX_BYTES)
        if len(result['pass_receipts'])!=5:raise ValueError('Exactly five scheduled passes required')
        result['memory_controls']={label:document(output/(label+'.memory_control.json'),budget,max_bytes=MEMORY_RECEIPT_MAX_BYTES) for label in protocol['passes']}
        if any(r['status']!='both_match_known_vector' for r in result['memory_controls'].values()):raise ValueError('Memory vector rejected')
        result['chunk_comparison']=compare_chunks(output,protocol['passes'],budget,protocol_target_bytes=protocol['target']['bytes'])
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
        result['metadata_read_limit_bytes']=budget.limit
        result['cumulative_model_read_upper_bound_bytes']=protocol['prior_attempt']['reported_model_read_bytes']+result['model_read_upper_bound_bytes']
        result['finished_utc']=dh.now()
        result['actual_model_read_bytes_reported']=sum(r['bytes_read'] for r in result['pass_receipts'].values())
        result['cumulative_actual_model_read_bytes_reported']=protocol['prior_attempt']['reported_model_read_bytes']+result['actual_model_read_bytes_reported']
        if not result['errors'] and len(result['pass_receipts'])==5:result['status']='five_pass_agreement_diagnostic_only'
        dh.write_new(output/'finish.json',result)
    if result['errors']:raise RuntimeError('Rejected diagnostic retained at '+str(output))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('self-test','run'))
    parser.add_argument('--manifest',type=Path,default=HERE/'preparation_manifest.0004.json')
    parser.add_argument('--authorize-large-file-read',action='store_true')
    a=parser.parse_args()
    if a.mode=='self-test':print(json.dumps(dh.self_test()));return
    if not a.authorize_large_file_read:raise ValueError('Explicit large-file execution switch required; preparation never reads model data')
    print(json.dumps(execute(a.manifest)))


if __name__=='__main__':main()
