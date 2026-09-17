"""Synthetic-only tests: no real score, figure, inference, archive or model read."""
from pathlib import Path
import hashlib,importlib.util,json,tarfile
import pytest
spec=importlib.util.spec_from_file_location('r30_archive_tests',Path(__file__).with_name('archive_verified.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8');return a.ref(path)


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'round';root.mkdir()
    ids=['real_cell_%03d'%i for i in range(131)]
    protocol=write(root/'protocol.json',{'schema':'synthetic-only-protocol'})
    armdata={};scores={};freeze_refs={};freeze_pre={};lock_pre={}
    for arm in ('off','on'):
        directory=root/arm
        fr=write(directory/'freeze.json',{'cells':[{'cell_id':i} for i in ids],
            'source':{'sha256':'source'},'selection_sha256':'selection','final_output_selection':True,
            'mmvq_hbm_mode':{'off':'legacy_mma_output_wave','on':'nominal_bandwidth_analytical_fallback'}[arm]})
        freeze_refs[arm]=fr;predrefs={}
        for index,ident in enumerate(ids):
            digest=hashlib.sha256(json.dumps({'cell_id':ident},sort_keys=True,separators=(',',':')).encode()).hexdigest()
            attempt=directory/'runs/attempts'/digest;target=directory/'predictions'/(ident+'.prediction.json')
            status='failed' if index==0 else 'predicted'
            command=['synthetic-worker',ident]
            start=write(attempt/'start.json',{'schema':'stable-native-worker-attempt/v1','cell_id':ident,
                'run_id':'run.0001','freeze_ref':fr,'command':command,
                'raw_result_path':str(attempt/'worker-result.json'),'official_result_path':str(target)})
            child=write(attempt/'child.json',{'pid':1000+index,'attempt_ref':start,'command':command})
            raw={'schema':'stable-native-cell-prediction/v1','cell_id':ident,'status':'predicted',
                 'freeze_ref':fr,'source_sha256':'source','selection_sha256':'selection'}
            rawref=write(attempt/'worker-result.json',raw)
            deadline=None
            if index==1:
                deadline=write(attempt/'soft-deadline.json',{'schema':'stable-native-soft-deadline/v1','attempt_ref':start,
                     'pid':1000+index,'hard_time_limit_enforced':False,'action':'continue_waiting_for_natural_exit'})
            execution=write(attempt/'execution.json',{'schema':'stable-native-worker-execution/v1','cell_id':ident,
                'freeze_ref':fr,'official_result_path':str(target),'published_status':status,
                'wait_policy':'natural_exit_soft_observation','hard_time_limit_enforced':False,'observation_seconds':600,
                'spawned':True,'natural_exit_observed':True,'returncode':7 if status=='failed' else 0,
                'result_identity_valid':status=='predicted','attempt_ref':start,'child_ref':child,
                'raw_result_ref':rawref,'soft_deadline_ref':deadline,'observation_interruptions':0})
            predrefs[ident]=write(target,{**raw,'status':status,'worker_execution_ref':execution})
            write(attempt/'sealed.json',{'execution_ref':execution,'prediction_ref':predrefs[ident]})
            (attempt/'worker.log').write_bytes(b'synthetic worker completed\r\n')
        armdata[arm]={'freeze_ref':fr,'prediction_refs':predrefs}
        scores[arm]={'score_ref':write(directory/'errors.0001.json',{'freeze_ref':fr})}
        start=write(directory/'runs/run.0001.start.json',{'schema':'stable-native-prediction-run/v1',
            'phase':'start','run_id':'run.0001','freeze_ref':fr,'scheduled_cell_ids':ids})
        write(directory/'runs/run.0001.finish.json',{'schema':'stable-native-prediction-run/v1','phase':'finish',
            'run_id':'run.0001','freeze_ref':fr,'start_ref':start,'all_started_workers_exited_and_sealed':True,
            'launches_stopped_after_observation_interrupt':False,'cells':[{'cell_id':i,'prediction_ref':predrefs[i]} for i in ids]})
        for stage,refs in [('freeze',freeze_pre),('lock',lock_pre)]:
            path=root/'preflight'/(stage+'_'+arm+'.json')
            refs[arm]=write(path,{'schema':'r30-frozen-source-preflight/v1','stage':stage,'arm':arm,
                'status':'passed','verified_cells':131,'freeze_ref':fr})
            path.with_suffix('.log').write_bytes(b'preflight\n')
    receipt=write(root/'freeze_receipt.json',{**freeze_refs,'frozen_source_preflights':freeze_pre})
    controls=write(root/'controls.json',{'protocol_ref':protocol,'freeze_receipt_ref':receipt,'lock_preflights':lock_pre})
    barrier=write(root/'predictions_complete.json',{'schema':'r30-full262-terminal-barrier/v1','terminal_count':262,
         'arms':armdata,'controls_ref':controls,'failures_preserved':True})
    report=write(root/'report.json',{'schema':'r30-two-arm-scoring-receipt/v1','denominator_per_arm':131,
         'formal_success':False,'barrier_ref':barrier,'arms':scores})
    grouped=write(root/'grouped_paired_report.json',{'schema':'r30-grouped-paired-errors/v1','fixed_cells':131,
         'fixed_metrics_per_arm':393,'groups':{str(i):{} for i in range(6)},'barrier_ref':barrier,'scoring_report_ref':report})
    (root/'grouped_paired_report.md').write_text('synthetic grouped report\n')
    outputs=[]
    for view in ('off','on','delta'):
        for ext in ('.png','.svg'):
            path=root/'heatmaps.0001'/(view+'_engine_error_heatmap'+ext);path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'fixture image\x00'+view.encode()+ext.encode());outputs.append(a.ref(path))
    write(root/'heatmaps.0001/heatmaps.provenance.json',{'schema':'r30-presentation-only-heatmaps/v1',
        'fixed_cell_denominator':131,'fixed_metric_denominator_per_arm':393,'fixed_deployment_groups':6,
        'fixed_mask':ids,'changes_scores_or_acceptance':False,
        'source_evidence':{'barrier_ref':barrier,'scoring_report_ref':report,'grouped_report_ref':grouped},'outputs':outputs})
    return root


def test_missing_barrier_or_grouped_refused_before_any_payload_read(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'load',lambda *args:pytest.fail('must refuse before opening result JSON'))
    with pytest.raises(ValueError,match='completed barrier'):a.select(tmp_path)
    (tmp_path/'predictions_complete.json').write_text('not parsed')
    with pytest.raises(ValueError,match='report.json'):a.select(tmp_path)
    (tmp_path/'report.json').write_text('not parsed')
    with pytest.raises(ValueError,match='grouped_paired_report'):a.select(tmp_path)


def test_fixture_prediction_names_and_source_copies_never_selected(campaign):
    before=a.select(campaign)
    for relative in ('identity_diagnostic/.test_tmp/predictions/cell.prediction.json','identity_diagnostic/.test_tmp/freeze.json',
        'test_extra/freeze.json','off/predictions/test_fake.prediction.json','off/source/freeze.json',
        'execution_source/freeze.json','on/runs/test_nested/run.0001.finish.json',
        'postprocess/archive.0001/freeze.json','on/predictions/unreferenced.prediction.json','extra.freeze.json','off/predictions/evil.exe'):
        path=campaign/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'not campaign evidence')
    assert a.select(campaign)==before
    assert sum(row['selection_reason']=='barrier terminal prediction' for row in before)==262
    assert sum(row['selection_reason']=='original worker raw result' for row in before)==262
    assert sum(row['selection_reason']=='worker natural-exit execution' for row in before)==262
    assert sum(row['selection_reason']=='worker soft observation deadline' for row in before)==2


@pytest.mark.parametrize('target',['identity_diagnostic/.test_tmp/freeze.json','test_nested/prediction.json','off/source/freeze.json'])
def test_reference_injection_rejected(campaign,target):
    path=campaign/target;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    barrier=a.load(campaign/'predictions_complete.json');barrier['arms']['off']['prediction_refs']['real_cell_000']=a.ref(path)
    write(campaign/'predictions_complete.json',barrier)
    with pytest.raises(ValueError):a.select(campaign)


@pytest.mark.parametrize('relative',['identity_diagnostic/freeze.json','test_fixtures/prediction.json','off/source/freeze.json','off/predictions/bad.dll'])
def test_direct_safe_member_rejects_fixture_source_and_binary(campaign,relative):
    path=campaign/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='excluded'):a.safe_member(campaign,path)


@pytest.mark.parametrize('failure',['live_lock','unresolved','missing_seal','extra_attempt','changed_raw','changed_terminal'])
def test_lifecycle_incomplete_or_changed_refused(campaign,failure):
    attempt=next((campaign/'off/runs/attempts').iterdir())
    if failure=='live_lock':(campaign/'off/runs/coordinator.lock').write_text('{}')
    elif failure=='unresolved':(attempt/'unresolved.json').write_text('{}')
    elif failure=='missing_seal':(attempt/'sealed.json').unlink()
    elif failure=='extra_attempt':(campaign/'off/runs/attempts'/'untracked').mkdir()
    elif failure=='changed_raw':(attempt/'worker-result.json').write_text('{}')
    else:(campaign/'off/predictions/real_cell_000.prediction.json').write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)):a.select(campaign)


def test_packet_split_and_parts_only_restore_are_byte_exact(campaign,tmp_path):
    members=a.select(campaign);output=tmp_path/'packet'
    result=a.package_selected(campaign,output,members,part_size=1024)
    assert result['status']=='verified' and result['parts_reassembled_verified'] and result['restored_bytes_compared']
    assert len(result['part_refs'])>1 and all(row['bytes']<=1024 for row in result['part_refs'])
    assert result['restore']['packet_source']=='assembled_from_parts'
    with tarfile.open(output/'detailed_evidence.tar.gz','r:gz') as tar:
        assert sorted(tar.getnames())==sorted(row['relative_path'] for row in members)
    assert all((output/'restored'/row['relative_path']).read_bytes()==(campaign/row['relative_path']).read_bytes() for row in members)
    # Raw and official differ by publication fields: both must survive separately.
    for relative in ('off/predictions/real_cell_000.prediction.json','on/predictions/real_cell_001.prediction.json'):
        pred=a.load(output/'restored'/relative)
        raw=Path(a.load(pred['worker_execution_ref']['path'])['raw_result_ref']['path']).relative_to(campaign)
        assert (output/'restored'/raw).read_bytes()!=(output/'restored'/relative).read_bytes()
    with pytest.raises(FileExistsError):a.package_selected(campaign,output,members)


def test_source_changed_before_archive_leaves_failed_terminal(campaign,tmp_path):
    members=a.select(campaign);(campaign/'protocol.json').write_text('{}')
    with pytest.raises(RuntimeError,match='partial'):a.package_selected(campaign,tmp_path/'packet',members)
    assert a.load(tmp_path/'packet/finish.json')['status']=='rejected'
