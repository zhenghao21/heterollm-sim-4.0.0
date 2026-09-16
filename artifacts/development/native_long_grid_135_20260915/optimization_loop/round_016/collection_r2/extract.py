"""Read-only actual-schema extraction and frozen matrix quality diagnostics."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sqlite3
from common import HERE,PROBE,PROCESS_MASK,load,write_new,ref,module,verify_freeze,distribution,union_ns,expected_calls,utc,clock_readback_gate,verify_clock_receipt
from quality import quality

REQUIRED={
 'NVTX_EVENTS':{'start','end','globalTid','text'},
 'CUPTI_ACTIVITY_KIND_RUNTIME':{'start','end','globalTid','correlationId','nameId','returnValue'},
 'CUPTI_ACTIVITY_KIND_KERNEL':{'start','end','globalPid','correlationId','deviceId','streamId','demangledName','shortName','gridX','gridY','gridZ','blockX','blockY','blockZ','staticSharedMemory','dynamicSharedMemory'},
 'StringIds':{'id','value'},
}
ROLE_NAMES={'quantize_q8_1':'conversion_mmvq','quantize_mmq_q8_1':'conversion_mmq','mul_mat_vec_q':'main_mmvq','mul_mat_q':'main_mmq','mul_mat_q_stream_k_fixup':'fixup_mmq'}


def role(kernel):return ROLE_NAMES.get(kernel['short_name_text'],'unsupported')


def validate_chain(kernels,config):
    ordered=sorted(kernels,key=lambda k:(k['start'],k['evidence_rowid']));roles=[role(k) for k in ordered];issues=[]
    if roles==['conversion_mmvq','main_mmvq']:
        observed_family='MMVQ';observed_source_path='MMVQ_Q8_1_HALF'
    elif roles in (['conversion_mmq','main_mmq'],['conversion_mmq','main_mmq','fixup_mmq']):
        observed_family='MMQ';observed_source_path='MMQ_Q8_1_D4_F32'
        conversions=[k for k in ordered if role(k)=='conversion_mmq']
        if not re.search(r'\(mmq_q8_1_ds_layout\)0\s*,',conversions[0]['demangled_name_text']):
            issues.append('MMQ_D4_layout_not_confirmed');observed_source_path=None
    else:observed_family='unsupported';observed_source_path=None;issues.append('unknown_or_incomplete_observed_kernel_chain')
    if observed_source_path!=config['expected_source_path']:issues.append('expected_source_path_mismatch')
    expected_type={'Q5_0':6,'Q8_0':8}.get(config['quant'])
    for kernel in ordered:
        if role(kernel).startswith(('main_','fixup_')) and (expected_type is None or not re.search(r'\(ggml_type\)'+str(expected_type)+r'\s*,',kernel['demangled_name_text'])):issues.append('kernel_quant_type_not_confirmed')
        if any(type(kernel.get(k)) is not int or kernel[k]<=0 for k in ('gridX','gridY','gridZ','blockX','blockY','blockZ')):issues.append('invalid_grid_or_block')
        if any(type(kernel.get(k)) is not int or kernel[k]<0 for k in ('staticSharedMemory','dynamicSharedMemory')):issues.append('invalid_shared_memory')
    if len({(k['deviceId'],k['streamId'],k['globalPid']) for k in ordered})!=1:issues.append('multiple_or_missing_device_stream_process')
    return {'complete':not issues,'observed_family':observed_family,'observed_source_path':observed_source_path,'roles':roles,'issues':issues,
        'fixup_observed':any(r=='fixup_mmq' for r in roles),'source_path_matches_observed_family':not issues}


def read_trace(path):
    path=Path(path).resolve(strict=True);before=ref(path);tables={};schema={}
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        names={r['name'] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name,required in REQUIRED.items():
            actual={r['name'] for r in db.execute('PRAGMA table_info("'+name+'")')}
            if not required.issubset(actual):raise ValueError('unsupported actual SQLite schema: '+name)
        for name in sorted(names):
            if name.startswith(('CUPTI_ACTIVITY_KIND_','NVTX','DIAGNOSTIC','PROFILER_OVERHEAD','TARGET_INFO_','ENUM_DIAGNOSTIC')) or name in ('StringIds','ENUM_NSYS_EVENT_CLASS'):
                schema[name]=[r['name'] for r in db.execute('PRAGMA table_info("'+name+'")')]
                records=[dict(r) for r in db.execute('SELECT rowid AS evidence_rowid,* FROM "'+name+'"')]
                for r in records:
                    for key,value in list(r.items()):
                        if isinstance(value,bytes):r[key]={'binary_hex':value.hex()}
                tables[name]=records
    if ref(path)!=before:raise ValueError('SQLite changed during read')
    strings={r['id']:r['value'] for r in tables['StringIds']}
    for r in tables['NVTX_EVENTS']:r['resolved_text']=strings.get(r.get('textId'),r.get('text'))
    for r in tables['CUPTI_ACTIVITY_KIND_RUNTIME']:r['name_text']=strings.get(r['nameId'])
    for r in tables['CUPTI_ACTIVITY_KIND_KERNEL']:
        r['demangled_name_text']=strings.get(r['demangledName'],'');r['short_name_text']=strings.get(r['shortName'],'')
    return {'schema':'operator-matrix-raw-nsys/v1','source_sqlite':before,'actual_schema':schema,'tables':tables,'time_unit':'ns'}


def correlate(raw,app,config):
    tables=raw['tables'];markers=tables['NVTX_EVENTS'];apis=tables['CUPTI_ACTIVITY_KIND_RUNTIME'];kernels=tables['CUPTI_ACTIVITY_KIND_KERNEL']
    wanted={r['nvtx_label']:(r['phase'],r['index']) for r in app['runs']};selected=[m for m in markers if m.get('resolved_text') in wanted]
    if len(selected)!=36 or len(wanted)!=36 or Counter(m['resolved_text'] for m in selected)!=Counter(wanted.keys()):raise ValueError('36 semantic NVTX ranges not uniquely captured')
    correlation=defaultdict(list)
    for k in kernels:correlation[(k['globalPid'],k['correlationId'])].append(k)
    used=set();calls=[]
    for marker in sorted(selected,key=lambda m:m['start']):
        begin,end,tid=marker['start'],marker['end'],marker['globalTid']
        if type(begin) is not int or type(end) is not int or begin>=end:raise ValueError('invalid NVTX interval')
        owned=[a for a in apis if a['globalTid']==tid and begin<=a['start'] and a['end']<=end]
        launches=[a for a in owned if a.get('name_text') and 'LaunchKernel' in a['name_text']]
        mapped=[];pairs=[];issues=[]
        for api in launches:
            found=correlation[(tid&PROCESS_MASK,api['correlationId'])]
            if len(found)!=1:issues.append('launch_lacks_unique_kernel')
            for kernel in found:
                if kernel['evidence_rowid'] in used:raise ValueError('kernel mapped more than once')
                used.add(kernel['evidence_rowid']);mapped.append(kernel)
                if kernel['start']<begin or kernel['end']>end:issues.append('kernel_outside_full_call_nvtx')
                pairs.append({'api':api,'kernel':kernel,'role':role(kernel),'start_offset_from_nvtx_ns':kernel['start']-begin,
                    'end_offset_from_nvtx_ns':kernel['end']-begin,'kernel_within_nvtx':begin<=kernel['start'] and kernel['end']<=end,
                    'launch_return_to_kernel_start_signed_ns':kernel['start']-api['end']})
        extra_gpu=[]
        owned_keys={(tid&PROCESS_MASK,a['correlationId']) for a in owned}
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
            extra_gpu.extend({'table':table,'event':event} for event in tables.get(table,[]) if (event.get('globalPid'),event.get('correlationId')) in owned_keys)
        if extra_gpu:issues.append('additional_captured_gpu_memory_activity_not_in_expected_kernel_chain')
        same_device_overlap=[];devices={k['deviceId'] for k in mapped};mapped_ids={k['evidence_rowid'] for k in mapped}
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
            for event in tables.get(table,[]):
                if table=='CUPTI_ACTIVITY_KIND_KERNEL' and event['evidence_rowid'] in mapped_ids:continue
                if event.get('deviceId') in devices and event['start']<end and event['end']>begin:
                    same_device_overlap.append({'table':table,'event':event,'overlap_with_full_call_ns':min(event['end'],end)-max(event['start'],begin),
                        'different_process':event.get('globalPid')!=(tid&PROCESS_MASK)})
        if same_device_overlap:issues.append('captured_external_gpu_activity_overlaps_full_call')
        chain=validate_chain(mapped,config);issues.extend(chain['issues'])
        phase,index=wanted[marker['resolved_text']]
        intervals=[(k['start'],k['end']) for k in mapped];span=max((k['end'] for k in mapped),default=0)-min((k['start'] for k in mapped),default=0)
        calls.append({'phase':phase,'index':index,'nvtx_verbatim':marker,'api_events_verbatim':owned,'kernel_launch_pairs':pairs,
            'chain':chain,'issues':issues,'additional_gpu_activity_verbatim':extra_gpu,'captured_external_gpu_overlap_verbatim':same_device_overlap,'kernel_union_ns':union_ns(intervals),'kernel_span_ns':span,'inter_kernel_gap_ns':span-union_ns(intervals),
            'launch_api_union_ns':union_ns((a['start'],a['end']) for a in launches),
            'host_runtime_api_union_ns':union_ns((a['start'],a['end']) for a in owned),
            'host_sync_api_union_ns':union_ns((a['start'],a['end']) for a in owned if a.get('name_text') and 'Synchronize' in a['name_text']),
            'nvtx_full_call_duration_ns':end-begin,'clock_policy':'Only Nsight SQLite timestamps combined. Full-call v2 NVTX contains target kernels and synchronize. Application raw QPC is preserved independently; no cross-domain subtraction.'})
    if [(c['phase'],c['index']) for c in calls]!=expected_calls():raise ValueError('captured call order/count mismatch')
    severities={r['id']:str(r.get('label',r.get('name',''))) for r in tables.get('ENUM_DIAGNOSTIC_SEVERITY_LEVEL',[])}
    diagnostics=tables.get('DIAGNOSTIC_EVENT',[])
    warnings=[d for d in diagnostics if d.get('severity') not in severities or re.search('warn|error|fatal',severities[d['severity']],re.I)]
    formal=[c['kernel_union_ns'] for c in calls if c['phase']=='formal']
    complete=all(not c['issues'] for c in calls)
    return {'schema':'operator-matrix-correlated-profile/v1','config':config,'calls':calls,'all36_chain_complete':complete,
        'source_path_matches_observed_family':all(c['chain']['source_path_matches_observed_family'] for c in calls),
        'formal_kernel_union':distribution(formal) if all(v>0 for v in formal) else {},
        'captured_kernels':len(kernels),'mapped_kernels':len(used),'unmapped_kernel_rowids':[k['evidence_rowid'] for k in kernels if k['evidence_rowid'] not in used],
        'all_diagnostics_verbatim':diagnostics,'warning_diagnostics_verbatim':warnings,
        'native_source_equivalence_proven':False,'empirical_runtime_dispatch_observed':complete,'calibration_eligible':False}


def numeric_document(path,config,freeze):
    before=ref(path);doc=load(path)
    if doc.get('schema')!='single-operator-surface-probe/v2' or doc.get('timing_contract',{}).get('id')!='actual-backend-single-graph-envelope/v2':raise ValueError('v2 raw and full-call timing contract required')
    protocol=load(freeze['protocol_ref']['path']);probe_root=Path(protocol['probe_root'])
    audit=module(probe_root/'full_raw_audit.py','frozen_operator_raw_audit').audit(doc,config)
    if doc.get('control_mode') is not False:audit['issues'].append('event_mode_required_for_profile_direct');audit['valid_raw']=False
    identity={r['path'].casefold():r for r in freeze['probe_files']}
    unbound=[];critical={Path(r['path']).name.casefold() for r in freeze['probe_files'] if Path(r['path']).name.casefold() in {'ggml-base.dll','ggml-cpu.dll','ggml-cuda.dll','operator-surface-probe.exe'}};seen=set()
    for item in doc.get('loaded_modules_before',[]):
        expected=identity.get(item['path'].casefold());name=Path(item['path']).name.casefold();seen.add(name)
        if name in critical and expected is None:audit['issues'].append('critical_loaded_module_unbound:'+name);audit['valid_raw']=False
        if expected and (item['sha256']!=expected['sha256'] or item['bytes']!=expected['bytes']):audit['issues'].append('loaded_module_identity_mismatch');audit['valid_raw']=False
        if expected is None:unbound.append(item)
    if not critical.issubset(seen):audit['issues'].append('critical_loaded_module_missing');audit['valid_raw']=False
    if ref(path)!=before:raise ValueError('application raw changed during numeric audit')
    signature={'M':doc['M'],'N':doc['N'],'K':doc['K'],'quant':doc['weight_format'],'input_sha256':doc['quantization']['input_sha256'],
        'packed_weight_sha256':doc['quantization']['packed_weight_sha256'],'environment':doc['environment'],
        'modules':sorted(doc['loaded_modules_before'],key=lambda r:r['path'].casefold()),'expected_source_path':doc['expected_source_path']}
    return doc,{'source':before,'audit':audit,'formal_host':distribution(r['host_wall_ns'] for r in doc['runs'] if r['phase']=='formal'),
        'signature':signature,'unbound_observed_cuda_or_os_runtime_modules':unbound,'QPC_absolute_boundaries_preserved_in_source':True}


def stage_complete(path,expected_freeze_sha256):
    r=load(path/'complete.json')
    protocol=load(HERE/'protocol.json');clock=r.get('clock_control_binding',{}).get('receipt_ref',{})
    binding=verify_clock_receipt(clock.get('path'),clock.get('sha256'),protocol['gpu_identity']['uuid'])
    anchor=load(HERE/'clock-control-binding.json')
    if anchor.get('receipt_ref')!=binding['receipt_ref']:raise ValueError('stage clock session differs from approved session')
    if r.get('external_approved_sha256')!=expected_freeze_sha256 or r.get('child_process_exited') is not True:raise ValueError('stage is not bound to approved freeze or confirmed exit')
    if r.get('status')!='completed' or r.get('returncode')!=0 or r.get('freeze_after',{}).get('passed') is not True:raise ValueError('process not completed successfully: '+str(path))
    for artifact in r.get('artifacts',[]):
        # Validate permanent process evidence; supervisor stdio may close immediately after receipt publication.
        if Path(artifact['path']).name in ('microbench.json','trace.nsys-rep','trace.sqlite','stdout.txt','stderr.txt','telemetry.json'):
            if ref(artifact['path'])!=artifact:raise ValueError('process artifact identity changed')
    return r


def extract_pair(config,pair,output,freeze,expected_freeze_sha256):
    base=HERE/'runs'/config['id']/f'pair_{pair+1:02d}';result={'pair':pair,'numerics_all_rows':False,'trace_chain_complete':False,
        'source_path_matches_observed_family':False,'trace_warning_free':False,'issues':[]}
    try:
        stages={}
        for mode in ('profile','direct','export'):
            try:stage_complete(base/mode,expected_freeze_sha256);stages[mode]={'complete':True}
            except (ValueError,KeyError,TypeError,OSError) as exc:stages[mode]={'complete':False,'reason':str(exc)}
        result['stage_statuses']=stages
        if not all(v['complete'] for v in stages.values()):raise ValueError('one or more stages failed or incomplete; all statuses retained')
        app,profile=numeric_document(base/'profile/microbench.json',config,freeze)
        direct_app,direct=numeric_document(base/'direct/microbench.json',config,freeze)
        result.update(profile_raw=profile,direct_raw=direct,profile_host=profile['formal_host'],direct_host=direct['formal_host'])
        gates={mode:clock_readback_gate(document,load(base/mode/'telemetry.json')['samples']) for mode,document in (('profile',app),('direct',direct_app))}
        result['clock_domain_gates']=gates;result['clock_domain_validated']=all(g['passed'] for g in gates.values())
        result['numerics_all_rows']=profile['audit']['valid_raw'] and direct['audit']['valid_raw'] and profile['signature']==direct['signature']
        trace=read_trace(base/'export/trace.sqlite');write_new(output/'raw_events.json',trace)
        mapped=correlate(trace,app,config);write_new(output/'mapped_calls.json',mapped)
        result.update(trace_chain_complete=mapped['all36_chain_complete'],source_path_matches_observed_family=mapped['source_path_matches_observed_family'],profile_kernel=mapped['formal_kernel_union'])
        messages=[]
        for mode in ('profile','export'):
            for name in ('stdout.txt','stderr.txt'):
                for line in (base/mode/name).read_text(encoding='utf-8',errors='replace').splitlines():
                    if re.search(r'\b(warning|error|fatal|unsupported|incompatible)\b',line,re.I):messages.append({'mode':mode,'file':name,'line':line})
        result['tool_warning_lines']=messages;result['trace_warning_free']=not messages and not mapped['warning_diagnostics_verbatim']
        result['trace_sources']={'raw':ref(output/'raw_events.json'),'mapped':ref(output/'mapped_calls.json'),'sqlite':trace['source_sqlite']}
    except (ValueError,KeyError,TypeError,OSError,sqlite3.Error) as exc:result['issues'].append(type(exc).__name__+': '+str(exc))
    write_new(output/'pair_summary.json',result);return result


def extract(output,expected_freeze_sha256):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(HERE.resolve()):raise ValueError('extraction output must be new and inside collection')
    before=verify_freeze(HERE/'freeze.json',expected_freeze_sha256);freeze=load(HERE/'freeze.json');protocol=load(HERE/'protocol.json');output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for config in protocol['configs']:
        pairs=[extract_pair(config,p,output/config['id']/f'pair_{p+1:02d}',freeze,expected_freeze_sha256) for p in range(3)]
        row=quality(config,pairs);write_new(output/config['id']/'quality.json',row);rows.append(row)
    after=verify_freeze(HERE/'freeze.json',expected_freeze_sha256)
    result={'schema':'operator-matrix-collection-summary/v1','utc':utc(),'freeze_ref':ref(HERE/'freeze.json'),'verification_before':before,'verification_after':after,
        'configs_required':26,'configs_reported':len(rows),'pairs_required':78,'formal_profile_calls_required':2340,'formal_direct_calls_required':2340,
        'diagnostic_accepted_configs':sum(r['measurement_cost_eligible'] for r in rows),'diagnostic_rejected_or_missing_configs':sum(not r['measurement_cost_eligible'] for r in rows),
        'by_group':{group:{'required':sum(c['group']==group for c in protocol['configs']),'accepted':sum(r['config']['group']==group and r['measurement_cost_eligible'] for r in rows)} for group in ('training','validation','aligned_control')},
        'source_runtime_equivalence_proven':False,'no_coefficients_emitted':True,'no_fit':True,'no_llm_actuals_read':True,'configs':rows}
    write_new(output/'summary.json',result);return {k:v for k,v in result.items() if k not in ('configs',)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-freeze-sha256',required=True);a=p.parse_args();print(json.dumps(extract(a.output,a.expected_freeze_sha256),ensure_ascii=False))
