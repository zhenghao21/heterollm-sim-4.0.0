"""Read-only bridge preparation. No GPU, no in-progress measurement reads, no profile invention.

Only --collection-complete-confirmed permits opening extracted measurement files.
The adapter intentionally fails closed on schema/clock/K evidence unsupported by
current resolver; caller cannot use a flag to bypass those semantic blockers.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
ROUND=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from heterollm_sim.kernel_calibration import effective_hardware_sha256, canonical_kernel_key


def load(path):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON field '+k)
            d[k]=v
        return d
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON '+value)))


def ref(path):
    path=Path(path).resolve(strict=True);before=path.stat();h=hashlib.sha256();count=0
    with path.open('rb') as stream:
        while chunk:=stream.read(1<<20):h.update(chunk);count+=len(chunk)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or count!=before.st_size:
        raise ValueError('file changed during bridge read: '+str(path))
    return {'path':str(path),'sha256':h.hexdigest(),'bytes':count}


def verify(reference):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256','bytes'}:raise ValueError('incomplete immutable reference')
    if ref(reference['path'])!=reference:raise ValueError('changed immutable evidence: '+reference['path'])
    return reference


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def schema_readiness(collector,probe,resolver_source):
    """Inspects only frozen protocol/code files; never opens collector runs."""
    protocol=load(collector/'protocol.json');probe_protocol=load(probe/'protocol.json');source=resolver_source.read_text()
    blockers=[]
    schema=protocol.get('schema')
    if schema=='operator-matrix-collection-protocol/v2' and 'protocol.get("schema") != "operator-matrix-collection-protocol/v1"' in source:
        blockers.append({'code':'resolver_collection_protocol_v2_unsupported','detail':'Current resolver requires collection-protocol/v1. Keep frozen v2 untouched; update the independent resolver adapter before producing an accepted profile.'})
    policy=protocol.get('quality_policy',{})
    if policy.get('every_formal_interval_bracketed_required') is True and 'profile_telemetry_id' not in source:
        blockers.append({'code':'resolver_clock_domain_evidence_unbound','detail':'Collection v2 has 2400MHz clock receipt and per-formal sampled readback gates. Current resolver bundle has no telemetry/clock ids or independently rederived clock gate. Do not silently drop these evidence fields.'})
    if 'key["role"] != "main" or config["expected_source_path"] != "MMVQ_Q8_1_HALF"' in source:
        scope='MMVQ main only; independently rejected conversion/MMQ/fixup'
    else:scope='requires explicit review of new resolver work contract'
    return {'collector_protocol':ref(collector/'protocol.json'),'probe_protocol':ref(probe/'protocol.json'),
        'resolver_source':ref(resolver_source),'blockers':blockers,'can_emit_accepted_profile':not blockers,
        'supported_role_scope':scope,'gpu_runs':0,'measurements_read':False,'effective_hardware_required_from_root':True,
        'caller_execution_context_constructed':False,'note':'Root must supply complete actual effective hardware config; bridge never creates caller phase/resource context.'}


def inventory_config(row,extraction,collector):
    """Retain every config result, and attach immutable original artifacts without guessing missing evidence."""
    config=row['config'];cid=config['id'];issues=[];references=[]
    if config.get('group')!='training':issues.append('holdout_excluded_from_calibration')
    if config.get('expected_source_path')!='MMVQ_Q8_1_HALF':issues.append('executed_K_contract_not_implemented_for_path')
    if row.get('measurement_cost_eligible') is not True:issues.append('collector_quality_rejected')
    if row.get('issues'):issues.extend('collector:'+str(v) for v in row['issues'])
    for pair in range(3):
        root=collector/'runs'/cid/f'pair_{pair+1:02d}'
        derived=extraction/cid/f'pair_{pair+1:02d}'
        paths=[derived/'raw_events.json',derived/'mapped_calls.json',derived/'pair_summary.json',root/'export/trace.sqlite']
        for mode in ('profile','direct'):
            paths.extend(root/mode/name for name in ('microbench.json','complete.json','telemetry.json','spec.json'))
        for mode in ('profile','export'):
            paths.extend(root/mode/name for name in ('stdout.txt','stderr.txt'))
        for path in paths:
            if not path.is_file():issues.append('missing_evidence:'+str(path))
            else:references.append(ref(path))
        for mode in ('profile','direct'):
            receipt_path=root/mode/'complete.json'
            if receipt_path.is_file():
                receipt=load(receipt_path)
                if receipt.get('status')!='completed' or receipt.get('returncode')!=0:issues.append(f'process_failed:{pair}:{mode}')
                clock=receipt.get('clock_control_binding',{}).get('receipt_ref')
                if not clock:issues.append(f'clock_receipt_missing:{pair}:{mode}')
                else:
                    verify(clock);references.append(clock)
                    for name in ('stdout_ref','stderr_ref'):
                        nested=load(clock['path']).get(name)
                        if not nested:issues.append('clock_command_log_missing')
                        else:verify(nested);references.append(nested)
    quality_path=extraction/cid/'quality.json'
    if quality_path.is_file():
        if load(quality_path)!=row:issues.append('summary_config_differs_from_quality_file')
        references.append(ref(quality_path))
    else:issues.append('config_quality_file_missing')
    unique={r['path']:r for r in references}
    return {'config':config,'candidate_role':'main','status':'candidate_requires_resolver_semantic_validation' if not issues else 'rejected_or_excluded',
        'issues':sorted(set(issues)),'evidence_refs':list(unique.values()),'declared_calibration_eligible':False}


def prepare(collector,output):
    collector=Path(collector).resolve(strict=True);protocol=load(collector/'protocol.json')
    probe=Path(protocol['probe_root']).resolve(strict=True)
    readiness=schema_readiness(collector,probe,ROOT/'src/heterollm_sim/kernel_calibration.py')
    readiness.update(schema='kernel-profile-bridge-readiness/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        collector_root=str(collector),expected_config_count=len(protocol['configs']),mode='preparation_only')
    write_new(output,readiness);return readiness


def inventory(collector,extraction,effective_hardware,output,*,collection_complete_confirmed=False):
    if collection_complete_confirmed is not True:raise ValueError('root confirmation of completed extraction required; do not read in-progress measurements')
    collector=Path(collector).resolve(strict=True);extraction=Path(extraction).resolve(strict=True)
    if not extraction.is_relative_to(collector):raise ValueError('extraction must belong to selected collector')
    protocol=load(collector/'protocol.json');summary=load(extraction/'summary.json')
    if summary.get('schema')!='operator-matrix-collection-summary/v1':raise ValueError('complete collector summary required')
    if summary.get('configs_required')!=26 or summary.get('configs_reported')!=26 or len(summary.get('configs',[]))!=26:
        raise ValueError('all26 outcomes including failures must be present')
    for boundary in ('verification_before','verification_after'):
        if summary.get(boundary,{}).get('passed') is not True:raise ValueError('extraction freeze verification missing')
    verify(summary['freeze_ref'])
    expected={c['id']:c for c in protocol['configs']};actual=[r['config']['id'] for r in summary['configs']]
    if len(set(actual))!=26 or set(actual)!=set(expected):raise ValueError('config denominator differs')
    effective=load(effective_hardware) if effective_hardware else None;hardware_digest=effective_hardware_sha256(effective) if effective is not None else None
    readiness=schema_readiness(collector,Path(protocol['probe_root']),ROOT/'src/heterollm_sim/kernel_calibration.py')
    rows=[]
    for row in summary['configs']:
        if row['config']!=expected[row['config']['id']]:raise ValueError('config mutated after freeze')
        rows.append(inventory_config(row,extraction,collector))
    candidates=sum(r['status']=='candidate_requires_resolver_semantic_validation' for r in rows)
    if candidates and effective is None:raise ValueError('candidate profile requires actual complete hardware configuration')
    stages=sum(1 for p in collector.glob('runs/*/pair_*/*/complete.json') if load(p).get('status')=='completed')
    completed_pairs=sum(all(pair.get('stage_statuses',{}).get(mode,{}).get('complete') is True for mode in ('profile','direct','export')) for row in summary['configs'] for pair in row.get('pairs',[]))
    result={'schema':'kernel-profile-bridge-inventory/v1','matrix_complete':completed_pairs==78,'completed_process_pairs':completed_pairs,'required_process_pairs':78,'completed_stages':stages,'required_stages':234,'status':'no_usable_calibration' if not candidates else 'candidates_blocked_pending_semantic_adapter',
        'summary_ref':ref(extraction/'summary.json'),'effective_hardware_ref':ref(effective_hardware) if effective_hardware else None,'effective_hardware_canonical_sha256':hardware_digest,
        'readiness':readiness,'config_denominator':26,'candidate_count':candidates,'rejected_or_excluded_count':26-candidates,
        'rows':rows,'profile_emitted':False,'model_coefficients_emitted':False,'native_llm_data_used':False,
        'reason':'This bridge inventory never promotes data. Resolver v2 compatibility and independent clock/K validation remain required.'}
    write_new(output,result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collector',type=Path,default=ROUND/'collection_r2');ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--extraction',type=Path);ap.add_argument('--effective-hardware',type=Path);ap.add_argument('--collection-complete-confirmed',action='store_true')
    a=ap.parse_args()
    if a.extraction:
        result=inventory(a.collector,a.extraction,a.effective_hardware,a.output,collection_complete_confirmed=a.collection_complete_confirmed)
    else:result=prepare(a.collector,a.output)
    print(json.dumps({k:result[k] for k in ('status','mode','can_emit_accepted_profile','candidate_count') if k in result}))
if __name__=='__main__':main()
