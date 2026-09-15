"""Summarize complete v2 matrix evidence without deleting rejected data."""
from pathlib import Path
import argparse,hashlib,json
from full_raw_audit import HERE,load,audit
from assess import assess

def verify_manifest(manifest):
    issues=[];entries=manifest.get('files',[])
    if not isinstance(entries,list) or not entries:return ['empty manifest file mapping']
    for entry in entries:
        try:
            path=Path(entry['path']);data=path.read_bytes()
            if not entry.get('sha256') or hashlib.sha256(data).hexdigest()!=entry['sha256'] or len(data)!=entry.get('bytes'):issues.append('frozen file changed:'+str(path))
        except Exception as exc:issues.append('frozen file unavailable:'+str(exc))
    for required in ['protocol.json','full_raw_audit.py','assess.py','summarize_matrix.py','invoke.ps1','run_frozen_matrix.ps1']:
        if len([x for x in entries if x.get('path')==str(HERE/required)])!=1:issues.append('mandatory frozen file absent:'+required)
    return issues

def raw_digest_issues(directory,audit_report,protocol):
    issues=[];rows=audit_report.get('rows',[])
    expected={(c['id'],mode) for c in protocol['configs'] for mode in ('event','control')}
    if not isinstance(rows,list) or len(rows)!=len(expected) or {(x.get('config'),x.get('mode')) for x in rows}!=expected:return ['raw audit mapping incomplete']
    for row in rows:
        path=directory/f"{row['config']}.{row['mode']}.json"
        try:
            if not row.get('sha256') or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:issues.append('raw changed after audit:'+row['config']+':'+row['mode'])
        except Exception as exc:issues.append('raw unavailable after audit:'+str(exc))
    return issues

def summarize(directory):
    protocol=load(HERE/'protocol.json');manifest=load(HERE/'build_manifest.json');issues=verify_manifest(manifest);assessments=[]
    try:execution=load(directory/'execution.json')
    except Exception as exc:execution={};issues.append('execution unavailable:'+str(exc))
    expected=[]
    for i,cfg in enumerate(protocol['configs']):
        for mode in (('event','control') if i%2==0 else ('control','event')):expected.append((cfg['id'],mode))
    rows=execution.get('runs',[])
    if [(x.get('config'),x.get('mode')) for x in rows]!=expected:issues.append('execution order/count mismatch')
    for row in rows:
        if row.get('succeeded') is not True or row.get('raw_exists') is not True or row.get('error') is not None:issues.append('execution failure:'+str(row.get('config'))+':'+str(row.get('mode')))
    for key,want in [('device_after_exit',0),('post_identity_error',None),('identity_unchanged',True),('full_raw_audit_exit',0)]:
        if key not in execution or execution[key]!=want:issues.append('execution gate:'+key)
    for name in ['identity_before.json','identity_after.json','device_before.csv','device_after.csv','processes_before.json','compute_processes_before.csv','background_load.json']:
        path=directory/name
        if not path.is_file() or (name!='compute_processes_before.csv' and path.stat().st_size==0):issues.append('missing evidence:'+name)
    try:
        before,after=load(directory/'identity_before.json'),load(directory/'identity_after.json')
        if not before.get('manifest_sha256') or before.get('manifest_sha256')!=after.get('manifest_sha256'):issues.append('identity receipt missing/mismatch')
        current=hashlib.sha256((HERE/'build_manifest.json').read_bytes()).hexdigest()
        if before.get('manifest_sha256')!=current:issues.append('manifest differs from execution')
    except Exception as exc:issues.append('identity receipt unavailable:'+str(exc))
    try:
        raw=load(directory/'full_raw_audit.json')
        if raw.get('complete_raw_passed') is not True:issues.append('complete raw audit rejected')
        issues.extend(raw_digest_issues(directory,raw,protocol))
        fresh_audit=audit(directory,protocol,manifest)
        if fresh_audit!=raw:issues.append('stored raw audit differs from current re-audit')
        if fresh_audit.get('complete_raw_passed') is not True:issues.append('current raw audit rejected')
    except Exception as exc:issues.append('complete raw audit unavailable:'+str(exc))
    records=execution.get('assessments',[])
    if [x.get('config') for x in records]!=[c['id'] for c in protocol['configs']]:issues.append('assessment execution count/order')
    for cfg in protocol['configs']:
        path=directory/f"{cfg['id']}.assessment.json"
        try:
            doc=load(path);accepted=doc.get('accepted') is True
            fresh=assess(load(directory/f"{cfg['id']}.event.json"),load(directory/f"{cfg['id']}.control.json"),cfg,protocol,manifest)
            if doc!=fresh:issues.append('stored assessment differs from current re-extraction:'+cfg['id'])
            records_for_config=[x for x in records if x.get('config')==cfg['id']]
            if len(records_for_config)!=1 or records_for_config[0].get('exit_code')!=(0 if accepted else 4):issues.append('assessment exit/result mismatch:'+cfg['id'])
            assessments.append({'config':cfg['id'],'group':cfg['group'],'accepted':accepted,'problems':doc.get('problems',[]),'statistics':doc.get('statistics',{}),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as exc:issues.append('assessment missing/invalid:'+cfg['id']+':'+str(exc));assessments.append({'config':cfg['id'],'group':cfg['group'],'accepted':False,'problems':['unavailable']})
    count=sum(x['accepted'] for x in assessments)
    return {'schema':'stream-event-matrix-summary/v2','run_directory':str(directory),'expected_configs':12,'expected_mode_runs':24,'completed_mode_runs':len(rows),'accepted_configs':count,'rejected_configs':12-count,'matrix_evidence_complete':not issues,'issues':issues,'all_diagnostic_gates_passed':not issues and count==12,'assessments':assessments,'calibration_eligible':False,'llm_actual_fit':False,'interpretation':'Diagnostic hot-buffer 64-graph batches only. Every failed, missing or rejected configuration remains in the fixed denominator. No timing parameter is generated.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();out=a.directory/'matrix_summary.json'
    if out.exists():raise SystemExit('refusing overwrite')
    result=summarize(a.directory);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='assessments'}));raise SystemExit(0 if result['all_diagnostic_gates_passed'] else 4)
