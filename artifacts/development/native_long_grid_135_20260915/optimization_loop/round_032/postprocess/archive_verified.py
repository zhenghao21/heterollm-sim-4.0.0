"""Explicit-reference R32 packet. No recursive campaign discovery, score or prediction.
Real execution is root-serial only, after prior postprocess naturally exits.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, hashlib, json, os, re, sys, tarfile, types

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
LEGACY=ROUND.parent/'round_024'
PINS={'split_evidence_packet.py':'e0f694666869c1032dab8c59bc75e5717aedb2299f129fb386271b3226e554bb',
      'restore_evidence.py':'a94b895b7250bb821ffd6839c03493758406b092d2cc057d1ebee337f2f41c76'}
ALLOWED_EXT={'.json','.jsonl','.log','.png','.svg','.md'}


def need(condition,message):
    if not condition:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def ref(path):
    path=Path(path).resolve();h=hashlib.sha256();count=0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):count+=len(block);h.update(block)
    return {'path':str(path),'bytes':count,'sha256':h.hexdigest()}


def normalize(r):return {'path':str(Path(r['path']).resolve()),'bytes':r.get('bytes',r.get('size_bytes')),'sha256':r['sha256']}


def load_helper(name):
    path=LEGACY/name;data=path.read_bytes()
    need(hashlib.sha256(data).hexdigest()==PINS[name],'legacy helper differs: '+name)
    module=types.ModuleType('verified_'+name.replace('.','_'));module.__file__=str(path)
    exec(compile(data,str(path),'exec'),module.__dict__)
    return module


def safe_member(root,path):
    root=Path(root).resolve();path=Path(path)
    rel=path.absolute().relative_to(root).as_posix()
    parts=rel.casefold().split('/')
    denied={'identity_diagnostic','source','sources','execution_source','control_source','__pycache__','.git'}
    need(not any(x in denied or x.startswith(('test','.test','source_snapshot','prepared_revision','archive')) for x in parts),
         'excluded source/fixture/archive path: '+rel)
    need(path.suffix.casefold() in ALLOWED_EXT,'excluded binary/source extension: '+rel)
    # Reject path traversal, junctions, symlinks and hardlinks before resolving target.
    for node in (path.absolute(),*path.absolute().parents):
        if node==root.parent:break
        st=node.lstat()
        need(not node.is_symlink() and not (getattr(st,'st_file_attributes',0)&1024),'link/reparse member forbidden')
        if node.is_file():need(st.st_nlink==1,'hardlinked member forbidden')
    need(path.resolve().is_relative_to(root),'member escaped root')
    return rel


class Selection:
    def __init__(self,root):self.root=Path(root).resolve();self.members={}
    def add(self,path,reason,expected=None):
        path=Path(path);relative=safe_member(self.root,path)
        r=ref(path)
        if expected is not None:need(r==normalize(expected),'reference mismatch: '+relative)
        row={'relative_path':relative,'bytes':r['bytes'],'sha256':r['sha256'],'selection_reason':reason}
        previous=self.members.get(relative.casefold())
        if previous is not None:need(previous['sha256']==row['sha256'] and previous['relative_path']==relative,'case collision or source changed')
        else:self.members[relative.casefold()]=row
        return load(path) if path.suffix=='.json' else None
    def bound(self,r,expected_path,reason):
        need(Path(r['path']).resolve()==Path(expected_path).resolve(),'reference is not canonical: '+reason)
        return self.add(expected_path,reason,r)
    def rows(self):return sorted(self.members.values(),key=lambda x:x['relative_path'])


def add_lifecycle(s, directory, ident, prediction, prediction_ref, freeze_ref):
    """Follow one canonical, sealed worker attempt; never discover test trees."""
    digest=hashlib.sha256(json.dumps({'cell_id':ident},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    attempt=directory/'runs/attempts'/digest
    need(attempt.is_dir() and not (attempt/'unresolved.json').exists(),'worker attempt absent or unresolved')
    execution_ref=prediction.get('worker_execution_ref')
    need(isinstance(execution_ref,dict),'terminal lacks worker execution')
    execution=s.bound(execution_ref,attempt/'execution.json','worker natural-exit execution')
    need(execution.get('schema')=='stable-native-worker-execution/v1' and execution.get('cell_id')==ident
         and normalize(execution['freeze_ref'])==normalize(freeze_ref)
         and execution.get('published_status')==prediction['status'],'execution identity differs')
    target=directory/'predictions'/(ident+'.prediction.json')
    need(Path(execution['official_result_path']).resolve()==target.resolve(),'execution output path differs')
    need(execution.get('wait_policy')=='natural_exit_soft_observation' and execution.get('hard_time_limit_enforced') is False
         and type(execution.get('observation_seconds')) in (int,float) and execution['observation_seconds']==600,'worker wait policy differs')
    start=s.bound(execution['attempt_ref'],attempt/'start.json','worker attempt start')
    need(start.get('schema')=='stable-native-worker-attempt/v1' and start.get('cell_id')==ident
         and normalize(start['freeze_ref'])==normalize(freeze_ref),'attempt identity differs')
    need(re.fullmatch(r'run\.[0-9]{4}',start.get('run_id','')) is not None,'invalid attempt run ID')
    need(Path(start['raw_result_path']).resolve()==(attempt/'worker-result.json').resolve()
         and Path(start['official_result_path']).resolve()==target.resolve(),'attempt output paths differ')
    if execution.get('spawned') is True:
        need(execution.get('natural_exit_observed') is True and type(execution.get('returncode')) is int,'worker natural exit unresolved')
        child=s.bound(execution['child_ref'],attempt/'child.json','worker child identity')
        need(type(child.get('pid')) is int and child['pid']>0 and normalize(child['attempt_ref'])==normalize(execution['attempt_ref'])
             and child.get('command')==start.get('command'),'worker child/attempt binding differs')
    else:
        need(execution.get('spawned') is False and prediction['status']=='failed' and execution.get('child_ref') is None,'unspawned worker cannot qualify')
        child=None
    raw=None
    if execution.get('raw_result_ref') is not None:
        raw=s.bound(execution['raw_result_ref'],attempt/'worker-result.json','original worker raw result')
    if prediction['status']=='predicted':
        need(execution.get('returncode')==0 and execution.get('result_identity_valid') is True and raw is not None and child is not None,'nonzero/invalid worker cannot qualify')
        need({k:v for k,v in prediction.items() if k not in ('worker_execution_ref','content_sha256')}==
             {k:v for k,v in raw.items() if k!='content_sha256'},'published prediction differs from retained raw')
    if execution.get('soft_deadline_ref') is not None:
        deadline=s.bound(execution['soft_deadline_ref'],attempt/'soft-deadline.json','worker soft observation deadline')
        need(deadline.get('schema')=='stable-native-soft-deadline/v1' and normalize(deadline['attempt_ref'])==normalize(execution['attempt_ref'])
             and deadline.get('hard_time_limit_enforced') is False and deadline.get('action')=='continue_waiting_for_natural_exit'
             and child is not None and deadline.get('pid')==child['pid'],'soft observation binding differs')
    interruptions=execution.get('observation_interruptions',0)
    need(type(interruptions) is int and 0<=interruptions<=10000,'invalid worker interruption count')
    for index in range(1,interruptions+1):
        item=s.add(attempt/('observation-interrupted.%04d.json'%index),'worker observer interruption')
        need(child is not None and item.get('pid')==child['pid'] and item.get('action')=='stop_new_launches_wait_for_natural_exit','worker observer interruption differs')
    s.add(attempt/'worker.log','worker process log')
    seal=s.add(attempt/'sealed.json','worker terminal seal')
    need(normalize(seal['execution_ref'])==normalize(execution_ref) and normalize(seal['prediction_ref'])==normalize(prediction_ref),'worker seal differs')
    return start['run_id'],attempt.name


def check_treatment(freeze):
    need(freeze.get('final_output_selection') is True and
         freeze.get('mmvq_hbm_mode','legacy_mma_output_wave')=='legacy_mma_output_wave',
         'both frontend arms require final output selection and legacy HBM mode')


def select(root,heatmap_dir='heatmaps.0001'):
    root=Path(root).resolve()
    # Fail before any score/prediction read when completion evidence is absent.
    for name in ('predictions_complete.json','report.json','grouped_paired_report.json'):
        need((root/name).is_file(),'completed barrier, scores and grouped report required: '+name)
    for arm in ('off','on'):
        need(not(root/arm/'runs/coordinator.lock').exists(),'live coordinator blocks archival selection')
    s=Selection(root)
    barrier=s.add(root/'predictions_complete.json','full262 barrier')
    need(barrier['schema']=='r32-full262-terminal-barrier/v1' and barrier['terminal_count']==262
         and set(barrier['arms'])=={'off','on'} and barrier['failures_preserved'] is True,'full262 barrier required')
    controls=s.bound(barrier['controls_ref'],root/'controls.json','barrier controls')
    receipt=s.bound(controls['freeze_receipt_ref'],root/'freeze_receipt.json','paired freeze receipt')
    s.bound(controls['protocol_ref'],root/'protocol.json','locked protocol')
    report=s.add(root/'report.json','existing scoring receipt')
    need(report['schema']=='r32-two-arm-scoring-receipt/v1' and report['denominator_per_arm']==131
         and report.get('formal_success') is False and set(report['arms'])=={'off','on'},'completed score receipt required')
    s.bound(report['barrier_ref'],root/'predictions_complete.json','score barrier binding')
    grouped=s.add(root/'grouped_paired_report.json','existing grouped report')
    need(grouped['schema']=='r32-grouped-paired-errors/v1' and grouped['fixed_cells']==131
         and grouped.get('fixed_metrics_per_arm')==393 and len(grouped.get('groups',{}))==6,'grouped report incomplete')
    s.bound(grouped['barrier_ref'],root/'predictions_complete.json','grouped barrier')
    s.bound(grouped['scoring_report_ref'],root/'report.json','grouped scoring binding')
    s.add(root/'grouped_paired_report.md','existing grouped report text')
    masks={}
    for arm in ('off','on'):
        directory=root/arm;entry=barrier['arms'][arm]
        freeze=s.bound(entry['freeze_ref'],directory/'freeze.json',arm+' frozen inputs')
        s.bound(receipt[arm],directory/'freeze.json','freeze receipt binding')
        ids=[row['cell_id'] for row in freeze['cells']]
        need(len(ids)==131 and len(set(ids))==131 and set(entry['prediction_refs'])==set(ids),'frozen131/prediction mask differs')
        check_treatment(freeze)
        masks[arm]=set(ids);attempt_runs={};attempt_names=set()
        for ident in ids:
            need(re.fullmatch(r'[A-Za-z0-9_-]+',ident) is not None and not ident.lower().startswith('test'),'unsafe cell id')
            prediction_ref=entry['prediction_refs'][ident]
            prediction=s.bound(prediction_ref,directory/'predictions'/(ident+'.prediction.json'),'barrier terminal prediction')
            need(prediction.get('schema')=='stable-native-cell-prediction/v1' and prediction.get('cell_id')==ident
                 and prediction.get('status') in ('predicted','failed','incomplete'),'nonterminal prediction')
            need(normalize(prediction['freeze_ref'])==normalize(entry['freeze_ref']),'prediction freeze binding')
            need(prediction['source_sha256']==freeze['source']['sha256'] and prediction['selection_sha256']==freeze['selection_sha256'],'prediction source selection binding')
            runid,attempt_name=add_lifecycle(s,directory,ident,prediction,prediction_ref,entry['freeze_ref'])
            attempt_runs[ident]=runid;attempt_names.add(attempt_name)
        # This direct, bounded inventory is a completeness check, not recursive selection.
        need({p.name for p in (directory/'runs/attempts').iterdir()}==attempt_names,'extra/unresolved worker attempt outside barrier')
        score_ref=report['arms'][arm]['score_ref'];score_path=Path(score_ref['path'])
        need(score_path.parent.resolve()==directory and score_path.name=='errors.0001.json','noncanonical score path')
        score=s.add(score_path,'existing score',score_ref)
        need(normalize(score['freeze_ref'])==normalize(entry['freeze_ref']),'score freeze mismatch')
        for stage,mapping in [('freeze',receipt['frozen_source_preflights']),('lock',controls['lock_preflights'])]:
            prepath=root/'preflight'/(stage+'_'+arm+'.json')
            pre=s.bound(mapping[arm],prepath,'referenced real preflight')
            need(pre['schema']=='r32-frozen-source-preflight/v1' and pre['stage']==stage and pre['arm']==arm
                 and pre['status']=='passed' and pre['verified_cells']==131,'preflight identity/status mismatch')
            need(normalize(pre['freeze_ref'])==normalize(entry['freeze_ref']),'preflight freeze mismatch')
            s.add(prepath.with_suffix('.log'),'canonical preflight log')
        runfiles=sorted((directory/'runs').glob('run.[0-9][0-9][0-9][0-9].finish.json'))
        need(runfiles,'completed run receipt missing');covered=set()
        for finish_path in runfiles:
            run=s.add(finish_path,'actual arm run finish');runid=finish_path.name.removesuffix('.finish.json')
            need(run['schema']=='stable-native-prediction-run/v1' and run['phase']=='finish' and run['run_id']==runid
                 and run.get('all_started_workers_exited_and_sealed') is True,'run identity or natural-exit seal differs')
            need(normalize(run['freeze_ref'])==normalize(entry['freeze_ref']),'run freeze mismatch')
            start=s.bound(run['start_ref'],directory/'runs'/(runid+'.start.json'),'actual arm run start')
            need(start['schema']==run['schema'] and start['phase']=='start' and start['run_id']==runid,'run start mismatch')
            need(normalize(start['freeze_ref'])==normalize(entry['freeze_ref']),'run start freeze mismatch')
            scheduled=set(start['scheduled_cell_ids'])
            need(len(scheduled)==len(start['scheduled_cell_ids']) and scheduled<=set(ids),'run schedules duplicate/unfrozen cells')
            runids=[]
            for row in run['cells']:
                ident=row['cell_id'];need(ident in scheduled and ident not in covered and attempt_runs[ident]==runid,'run result not uniquely scheduled')
                need(normalize(row['prediction_ref'])==normalize(entry['prediction_refs'][ident]),'run prediction differs from barrier')
                runids.append(ident)
            need(len(set(runids))==len(runids),'duplicate run terminal')
            need(set(runids)==scheduled or run.get('launches_stopped_after_observation_interrupt') is True,'unexplained partial scheduled run')
            for path in sorted((directory/'runs').glob(runid+'.observation-interrupted.[0-9][0-9][0-9][0-9].json')):
                s.add(path,'coordinator observer interruption')
            covered.update(runids)
        need(covered==set(ids),'run receipts do not cover all frozen predictions')
    need(masks['off']==masks['on'],'two arms frozen mask differs')
    need(re.fullmatch(r'heatmaps\.[0-9]{4}',heatmap_dir),'noncanonical heatmap directory')
    hp=root/heatmap_dir/'heatmaps.provenance.json';heat=s.add(hp,'completed heatmap provenance')
    need(heat['schema']=='r32-presentation-only-heatmaps/v1' and heat['fixed_cell_denominator']==131
         and heat.get('fixed_metric_denominator_per_arm')==393 and heat.get('fixed_deployment_groups')==6
         and set(heat['fixed_mask'])==masks['off'] and heat['changes_scores_or_acceptance'] is False,'heatmap proof mismatch')
    for key,filename in [('barrier_ref','predictions_complete.json'),('scoring_report_ref','report.json'),('grouped_report_ref','grouped_paired_report.json')]:
        s.bound(heat['source_evidence'][key],root/filename,'heatmap source binding')
    outputs={view+'_engine_error_heatmap'+suffix for view in ('off','on','delta') for suffix in ('.png','.svg')}
    need(len(heat['outputs'])==6 and {Path(x['path']).name for x in heat['outputs']}==outputs,'heatmap output set differs')
    for r in heat['outputs']:s.bound(r,root/heatmap_dir/Path(r['path']).name,'completed heatmap output')
    return s.rows()


def assert_previous_idle():
    import psutil
    conflicts=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid():continue
        row=process.info;cmd=' '.join(row.get('cmdline') or []).replace('\\','/').lower()
        if ((row.get('name') or '').lower().startswith('python') and
            (row.get('cmdline') is None or 'postprocess.py' in cmd or 'archive_verified.py' in cmd
             or 'run_candidate.py' in cmd or 'predict_stable_native_dataset.py' in cmd)):
            conflicts.append({'pid':row['pid'],'name':row['name']})
    need(not conflicts,'prior postprocessor/campaign still live or uninspectable; wait naturally: '+str(conflicts))


def package_selected(root,output,members,part_size=40*1024*1024,final_verifier=None):
    restore=load_helper('restore_evidence.py');split=load_helper('split_evidence_packet.py')
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(exist_ok=False)
    finish={'schema':'r32-reference-selected-archive/v1','status':'rejected','created_utc':now(),
            'originals_modified':False,'prior_rounds_modified':False,
            'selection':'explicit full262 closure; no recursive reference or directory walk',
            'large_model_hashed':False,'scoring_executed':False}
    try:
        need(members,'empty evidence selection')
        names=[row['relative_path'] for row in members]
        need(len(names)==len(set(n.casefold() for n in names)),'duplicate/colliding packet members')
        for row in members:
            source=root/row['relative_path'];need(safe_member(root,source)==row['relative_path'],'unsafe source')
            need(restore.digest_file(source)==(row['bytes'],row['sha256']),'source changed before packet')
        packet=output/restore.PACKET_NAME;index=output/restore.INDEX_NAME
        with packet.open('xb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
                with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                    for row in members:
                        info=tarfile.TarInfo(row['relative_path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                        with (root/row['relative_path']).open('rb') as source:tar.addfile(info,source)
            raw.flush();os.fsync(raw.fileno())
        expected={row['relative_path']:row for row in members}
        with tarfile.open(packet,'r:gz') as tar:
            for member in restore.archive_members(tar,expected):
                restore.check_member(tar,member,expected[member.name],compare_path=root/member.name)
        size,sha=restore.digest_file(packet)
        document={'schema':restore.SCHEMA,'created_utc':now(),'packet_name':packet.name,'packet_bytes':size,
                  'packet_sha256':sha,'member_count':len(members),'members':members,
                  'originals_preserved_locally':True,'member_bytes_compared_to_sources':True,
                  'selection_policy':finish['selection'],'identity_diagnostic_included':False,'lifecycle_raw_deduplication':'none; original raw and published bytes both retained'}
        write_new(index,document)
        parts_index=output/restore.PARTS_INDEX_NAME
        parts=split.split_packet(packet,index,output,parts_index,part_size=part_size)
        # Restore from PARTS into new paths, then compare every restored byte to canonical originals.
        assembled=output/'reassembled';assembled.mkdir();restored=output/'restored';restored.mkdir()
        result=restore.restore(assembled/packet.name,index,restored,parts_index=parts_index)
        for row in members:
            first=root/row['relative_path'];second=restored/row['relative_path']
            with first.open('rb') as a,second.open('rb') as b:
                while True:
                    x,y=a.read(1024*1024),b.read(1024*1024)
                    need(x==y,'restored member byte difference')
                    if not x:break
            need(restore.digest_file(first)==(row['bytes'],row['sha256']),'original changed after restore')
        if final_verifier is not None:final_verifier()
        finish.update(status='verified',packet_ref=ref(packet),index_ref=ref(index),parts_index_ref=ref(parts_index),
                      part_refs=[ref(output/r['filename']) for r in parts['parts']],restore=result,
                      member_count=len(members),parts_reassembled_verified=True,restored_bytes_compared=True,
                      publication_files=[index.name,parts_index.name,'finish.json']+[r['filename'] for r in parts['parts']])
    except BaseException as error:finish['error']=type(error).__name__+': '+str(error)
    write_new(output/'finish.json',finish)
    if finish['status']!='verified':raise RuntimeError('archive failed; exclusive partial outputs retained: '+str(output))
    return finish


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--output',type=Path,default=HERE/'archive_verified.0001')
    parser.add_argument('--heatmaps',default='heatmaps.0001');args=parser.parse_args()
    output=args.output.resolve()
    need(output.parent==HERE and re.fullmatch(r'archive_verified\.[0-9]{4}',output.name),'output must be exclusive postprocess/archive_verified.NNNN')
    assert_previous_idle()
    import importlib.util
    spec=importlib.util.spec_from_file_location('r32_archive_postprocess',HERE/'postprocess.py')
    post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
    context=post.load_completed()
    members=select(ROUND,args.heatmaps)
    assert_previous_idle()
    result=package_selected(ROUND,output,members,final_verifier=lambda:post.check_unchanged(context['evidence']))
    print(json.dumps({'status':result['status'],'members':result['member_count'],'output':str(output),
                      'publication_files':result['publication_files'],'prior_rounds_modified':False}))


if __name__=='__main__':main()
