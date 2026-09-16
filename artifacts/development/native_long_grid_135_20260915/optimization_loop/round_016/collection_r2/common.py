"""Read-only identity and fixed policy helpers. Importing never opens a GPU."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
PROBE=ROUND/'operator_probe/r3'
PILOT=ROUND.parent/'round_015/trace_2026_5'
PROCESS_MASK=0xFFFFFFFFFF000000
POLICY={
    'config_count':26,'process_pairs':3,'calls_per_process':36,'formal_calls_per_process':30,
    'pair_orders':[['profile','direct'],['direct','profile'],['profile','direct']],
    'checkpoint_seconds':180,'terminate_on_checkpoint':False,
    'kernel_formal_p90_div_p10_max':1.50,'profile_process_median_max_relative_deviation':0.05,
    'profile_direct_host_median_max_relative_difference':0.20,
    'direct_host_formal_p90_div_p10_max':1.50,
    'all_first_warmup_formal_numeric_rows_pass_required':True,
    'all_profile_process_kernel_chain_and_source_path_required':True,
    'quantile_method':'linear interpolation at (n-1)*q',
    'median_deviation_center':'median of three profiled process medians',
    'host_ratio_denominator':'paired direct process host median',
    'within_profile_dispersion':'each of three formal kernel-union distributions',
    'target_sm_clock_mhz':2400,'sm_clock_tolerance_mhz':30,'telemetry_period_seconds':0.005,
    'maximum_formal_clock_bracket_gap_ms':25,'every_formal_interval_bracketed_required':True,
    'diagnostic_labels_only':True,'coefficients_emitted':False,'validation_group_never_fitted':True,
}

def utc():return datetime.now(timezone.utc).isoformat()

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path,document):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(document,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def ref(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}

def verify_ref(r):
    actual=ref(r['path'])
    if actual!=r:raise ValueError('identity changed: '+r['path'])
    return actual

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);exec(compile(Path(path).read_bytes(),str(path),'exec'),result.__dict__);return result

def quantile(values,q):
    values=sorted(values)
    if not values or not 0<=q<=1:raise ValueError('invalid quantile')
    i=(len(values)-1)*q;a=int(i);return values[a]+(values[min(a+1,len(values)-1)]-values[a])*(i-a)

def distribution(values):
    values=list(values)
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid positive samples')
    lo,hi=quantile(values,.1),quantile(values,.9)
    return {'count':len(values),'minimum_ns':min(values),'maximum_ns':max(values),'median_ns':statistics.median(values),'p10_ns':lo,'p90_ns':hi,'p90_div_p10':hi/lo}

def union_ns(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if type(start) is not int or type(end) is not int or end<start:raise ValueError('invalid interval')
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return sum(b-a for a,b in merged)

def expected_calls():return [('first_call',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]

class IdentityError(ValueError):
    """An approved measurement identity is missing, changed or incomplete."""


def _reference_map(refs,label):
    if not isinstance(refs,list) or not refs:raise IdentityError('empty or missing '+label)
    result={}
    for item in refs:
        if not isinstance(item,dict) or set(item)!={'path','sha256','bytes'}:
            raise IdentityError('invalid reference fields: '+label)
        if not isinstance(item['path'],str) or not Path(item['path']).is_absolute():raise IdentityError('absolute reference path required: '+label)
        if not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256']):raise IdentityError('invalid reference digest: '+label)
        if type(item['bytes']) is not int or item['bytes']<0:raise IdentityError('invalid reference size: '+label)
        key=str(Path(item['path']).resolve()).casefold()
        if key in result:raise IdentityError('duplicate reference: '+label)
        result[key]=item
    return result


def verify_freeze(path,expected_sha256,full_tools=True):
    """The caller supplies the approved SHA; neither this manifest nor ready can approve itself."""
    path=Path(path).resolve()
    try:
        if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):
            raise IdentityError('external approved freeze SHA256 required')
        actual=ref(path)
        if actual['sha256']!=expected_sha256:raise IdentityError('freeze differs from externally approved SHA256')
        freeze=load(path)
        if freeze.get('schema')!='operator-matrix-collection-freeze/v2':raise IdentityError('invalid collection freeze schema')
        ready=load(path.with_name('ready.json'))
        if ready.get('freeze_ref')!=actual:raise IdentityError('ready-to-approved-freeze binding mismatch')
        maps={name:_reference_map(freeze.get(name),name) for name in ('files','probe_files','tool_files','critical_tool_files')}
        def bound_reference(key):
            item=freeze.get(key)
            _reference_map([item],key);verify_ref(item)
            file_map=maps['files']
            if file_map.get(str(Path(item['path']).resolve()).casefold())!=item:raise IdentityError(key+' missing from local identity map')
            return item
        for key in ('protocol_ref','python','probe_manifest_ref','tool_inventory_ref','tool_signatures_ref'):bound_reference(key)
        if Path(freeze['protocol_ref']['path']).resolve()!=path.with_name('protocol.json'):raise IdentityError('protocol must belong to this revision')
        import sys
        if Path(freeze['python']['path']).resolve()!=Path(sys.executable).resolve():raise IdentityError('Python execution identity mismatch')
        protocol=load(freeze['protocol_ref']['path'])
        if protocol.get('schema')!='operator-matrix-collection-protocol/v2' or protocol.get('quality_policy')!=POLICY:raise IdentityError('protocol/policy drift')
        if protocol.get('probe_manifest_ref')!=freeze['probe_manifest_ref']:raise IdentityError('protocol probe manifest binding mismatch')
        for name in ('common.py','runner.py','worker.py','extract.py','quality.py','test_collection.py','README.md','protocol.json'):
            key=str(path.parent/name).casefold()
            if key not in maps['files']:raise IdentityError('required collector file missing: '+name)
        if Path(protocol['probe_root']).resolve()!=Path(freeze['probe_manifest_ref']['path']).resolve().parent:raise IdentityError('probe root does not match bound manifest')
        build=load(freeze['probe_manifest_ref']['path'])
        if build.get('schema')!='operator-surface-probe-build/v1' or maps['probe_files']!=_reference_map(build.get('files'),'bound probe manifest files'):
            raise IdentityError('probe identity set differs from bound build manifest')
        for name in ('protocol.json','stream_event_probe.cpp','source_reference.h','identity_lock.h','identity_lock.json','invoke.ps1','full_raw_audit.py','operator-surface-probe.exe'):
            key=str(Path(protocol['probe_root'])/name).casefold()
            if key not in maps['probe_files']:raise IdentityError('required probe file missing: '+name)
        if protocol.get('executable')!=build.get('executable'):raise IdentityError('probe executable binding mismatch')
        inventory=load(freeze['tool_inventory_ref']['path']);signatures=load(freeze['tool_signatures_ref']['path'])
        if maps['tool_files']!=_reference_map(inventory.get('files'),'bound tool inventory files'):raise IdentityError('tool identity set differs from bound inventory')
        if signatures.get('critical_nvidia_binary_signatures_valid') is not True or not signatures.get('critical_names'):raise IdentityError('critical tool signature evidence missing')
        names=set(signatures['critical_names'])
        if not names.issubset({Path(r['path']).name for r in inventory['files']}):raise IdentityError('named critical tool absent from inventory')
        expected_critical=[r for r in inventory['files'] if Path(r['path']).name in names or 'cupti' in Path(r['path']).name.lower()]
        if maps['critical_tool_files']!=_reference_map(expected_critical,'derived critical tool files'):raise IdentityError('critical tool identity set mismatch')
        profiler=protocol['nsys']['executable']
        if maps['tool_files'].get(str(Path(profiler['path']).resolve()).casefold())!=profiler:raise IdentityError('profiler absent from bound tool identity')
        refs=freeze['files']+freeze['probe_files']+(freeze['tool_files'] if full_tools else freeze['critical_tool_files'])
        for item in refs:verify_ref(item)
        if full_tools:
            tool_root=Path(inventory['root'])
            current={str(p.resolve()).casefold() for p in tool_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower()!='.pyc'}
            if current!=set(maps['tool_files']):raise IdentityError('live tool inventory membership changed')
        if ref(path)!=actual:raise IdentityError('freeze changed during verification')
        return {'passed':True,'utc':utc(),'checked_files':len(refs),'full_tool_inventory':full_tools,
                'external_approved_sha256':expected_sha256,'freeze_ref':actual,'required_sets_complete':True}
    except IdentityError:raise
    except (ValueError,KeyError,TypeError,OSError) as exc:raise IdentityError(str(exc)) from exc


def verify_clock_receipt(path,expected_sha256,gpu_uuid):
    before=ref(path)
    if before['sha256']!=expected_sha256:raise IdentityError('clock-control receipt SHA mismatch')
    document=load(path)
    if document.get('schema')!='operator-clock-control-receipt/v1' or document.get('gpu_uuid')!=gpu_uuid:
        raise IdentityError('clock-control receipt schema/GPU mismatch')
    if document.get('target_sm_clock_mhz')!=POLICY['target_sm_clock_mhz'] or document.get('sm_clock_tolerance_mhz')!=POLICY['sm_clock_tolerance_mhz']:
        raise IdentityError('clock-control target mismatch')
    if document.get('requested_lock_min_mhz')!=2400 or document.get('requested_lock_max_mhz')!=2400 or document.get('lock_command_returncode')!=0:
        raise IdentityError('clock-control lock command not successful')
    if not isinstance(document.get('command'),list) or '-lgc' not in document['command'] or '2400,2400' not in document['command']:
        raise IdentityError('clock-control command evidence missing')
    if not isinstance(document.get('created_utc'),str) or document.get('restore_on_exit_planned') is not True:
        raise IdentityError('clock-control lifecycle evidence missing')
    for key in ('stdout_ref','stderr_ref'):verify_ref(document[key])
    if ref(path)!=before:raise IdentityError('clock-control receipt changed while reading')
    return {'receipt_ref':before,'document':document,'scope':'External command evidence; not proof of actual in-call clock. Separate measured readback gate required.'}


def clock_readback_gate(app,samples):
    formal=[r for r in app.get('runs',[]) if r.get('phase')=='formal'];frequency=app.get('qpc_frequency');issues=[]
    if len(formal)!=30 or type(frequency) is not int or frequency<=0:return {'passed':False,'issues':['missing_formal_QPC_scope']}
    ordered=sorted((s for s in samples if type(s.get('qpc_ticks')) is int),key=lambda s:s['qpc_ticks'])
    used=set();coverage=[];max_ticks=frequency*POLICY['maximum_formal_clock_bracket_gap_ms']/1000
    for row in formal:
        begin,end=row.get('qpc_start'),row.get('qpc_end')
        if type(begin) is not int or type(end) is not int or end<begin:issues.append('invalid_formal_QPC');continue
        before=[(i,s) for i,s in enumerate(ordered) if s['qpc_ticks']<=begin]
        after=[(i,s) for i,s in enumerate(ordered) if s['qpc_ticks']>=end]
        if not before or not after:issues.append('formal_interval_not_bracketed');continue
        lo,left=before[-1];hi,right=after[0]
        if begin-left['qpc_ticks']>max_ticks or right['qpc_ticks']-end>max_ticks:issues.append('formal_clock_bracket_too_wide')
        for i in range(lo,hi+1):used.add(i)
        coverage.append({'index':row.get('index'),'sample_before_qpc':left['qpc_ticks'],'sample_after_qpc':right['qpc_ticks'],'observed_samples_inside':sum(begin<=s['qpc_ticks']<=end for s in ordered[lo:hi+1])})
    values=[]
    for i in sorted(used):
        reading=ordered[i].get('sm_mhz',{})
        if reading.get('status')!=0 or type(reading.get('value')) is not int:issues.append('SM_clock_readback_missing');continue
        values.append(reading['value'])
        if abs(reading['value']-POLICY['target_sm_clock_mhz'])>POLICY['sm_clock_tolerance_mhz']:issues.append('SM_clock_outside_target_domain')
    if not values:issues.append('no_formal_clock_readbacks')
    return {'passed':not issues,'issues':sorted(set(issues)),'target_sm_clock_mhz':POLICY['target_sm_clock_mhz'],
        'tolerance_mhz':POLICY['sm_clock_tolerance_mhz'],'formal_intervals':len(formal),'bracketed_intervals':len(coverage),
        'used_readback_samples':len(used),'sm_clock_min_mhz':min(values) if values else None,'sm_clock_max_mhz':max(values) if values else None,
        'coverage':coverage,'clock_locked_inferred':False,
        'scope':'Sampled NVML readbacks bracket every formal host interval in absolute QPC. They do not prove a continuous per-kernel frequency waveform.'}
