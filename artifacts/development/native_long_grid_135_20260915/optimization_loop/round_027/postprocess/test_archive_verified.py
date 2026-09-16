"""Synthetic-only archival tests; no real campaign packaging or score execution."""
from pathlib import Path
import json,sys,tarfile
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_verified as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8');return a.ref(path)


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'round';root.mkdir()
    ids=['real_cell_%03d'%i for i in range(131)]
    protocol=write(root/'protocol.json',{'schema':'synthetic-test-protocol'})
    armdata={};scores={};freeze_refs={};freeze_pre={};lock_pre={}
    for arm in ('off','on'):
        directory=root/arm
        fr=write(directory/'freeze.json',{'cells':[{'cell_id':i} for i in ids],
                 'source':{'sha256':'source'},'selection_sha256':'selection'})
        freeze_refs[arm]=fr;predrefs={}
        for ident in ids:
            predrefs[ident]=write(directory/'predictions'/(ident+'.prediction.json'),
                {'schema':'stable-native-cell-prediction/v1','cell_id':ident,'status':'predicted',
                 'freeze_ref':fr,'source_sha256':'source','selection_sha256':'selection'})
            (directory/'predictions'/(ident+'.run.0001.worker.log')).write_bytes(b'worker completed\r\n')
        armdata[arm]={'freeze_ref':fr,'prediction_refs':predrefs}
        scores[arm]={'score_ref':write(directory/'errors.0001.json',{'freeze_ref':fr})}
        start=write(directory/'runs/run.0001.start.json',{'schema':'stable-native-prediction-run/v1',
                    'phase':'start','run_id':'run.0001','freeze_ref':fr,'scheduled_cell_ids':ids})
        write(directory/'runs/run.0001.finish.json',{'schema':'stable-native-prediction-run/v1',
            'phase':'finish','run_id':'run.0001','freeze_ref':fr,'start_ref':start,
            'cells':[{'cell_id':i,'prediction_ref':predrefs[i]} for i in ids]})
        for stage,refs in [('freeze',freeze_pre),('lock',lock_pre)]:
            path=root/'preflight'/(stage+'_'+arm+'.json')
            refs[arm]=write(path,{'schema':'r27-frozen-source-preflight/v1','stage':stage,'arm':arm,
                                 'status':'passed','verified_cells':131,'freeze_ref':fr})
            path.with_suffix('.log').write_bytes(b'preflight\n')
    receipt=write(root/'freeze_receipt.json',{**freeze_refs,'frozen_source_preflights':freeze_pre})
    controls=write(root/'controls.json',{'protocol_ref':protocol,'freeze_receipt_ref':receipt,'lock_preflights':lock_pre})
    barrier=write(root/'predictions_complete.json',{'schema':'r27-full262-terminal-barrier/v1',
       'terminal_count':262,'arms':armdata,'controls_ref':controls,'failures_preserved':True})
    report=write(root/'report.json',{'schema':'r27-two-arm-scoring-receipt/v1','denominator_per_arm':131,
                  'barrier_ref':barrier,'arms':scores})
    grouped=write(root/'grouped_paired_report.json',{'schema':'r27-grouped-paired-errors/v1',
                  'fixed_cells':131,'barrier_ref':barrier,'scoring_report_ref':report})
    outputs=[]
    for view in ('off','on','delta'):
        for ext in ('.png','.svg'):
            p=root/'heatmaps.0001'/(view+'_engine_error_heatmap'+ext);p.parent.mkdir(exist_ok=True)
            p.write_bytes(b'fixture image\x00'+view.encode()+ext.encode());outputs.append(a.ref(p))
    write(root/'heatmaps.0001/heatmaps.provenance.json',{'schema':'r27-presentation-only-heatmaps/v1',
         'fixed_cell_denominator':131,'fixed_mask':ids,'changes_scores_or_acceptance':False,
         'source_evidence':{'barrier_ref':barrier,'scoring_report_ref':report,'grouped_report_ref':grouped},'outputs':outputs})
    return root


def test_fixture_prediction_freeze_names_never_selected(campaign):
    before=a.select(campaign)
    for relative in ('identity_diagnostic/.test_tmp/predictions/cell.prediction.json',
       'identity_diagnostic/.test_tmp/freeze.json','test_extra/freeze.json','off/predictions/test_fake.prediction.json',
       'off/source/freeze.json','on/runs/test_nested/run.0001.finish.json','postprocess/archive.0001/freeze.json',
       'on/predictions/unreferenced.prediction.json','extra.freeze.json','off/predictions/evil.exe'):
        p=campaign/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture must never enter')
    assert a.select(campaign)==before
    assert sum(x['selection_reason']=='barrier terminal prediction' for x in before)==262
    assert all('identity_diagnostic' not in r['relative_path'] for r in before)


@pytest.mark.parametrize('target',['identity_diagnostic/.test_tmp/freeze.json','test_nested/prediction.json','off/source/freeze.json'])
def test_reference_injection_is_rejected(campaign,target):
    p=campaign/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    barrier=a.load(campaign/'predictions_complete.json')
    barrier['arms']['off']['prediction_refs']['real_cell_000']=a.ref(p)
    write(campaign/'predictions_complete.json',barrier)
    # Reference binding itself rejects before payload can enter archive.
    with pytest.raises(ValueError):a.select(campaign)


@pytest.mark.parametrize('relative',['identity_diagnostic/freeze.json','test_fixtures/prediction.json',
                                   'off/source/freeze.json','off/predictions/bad.dll'])
def test_direct_safe_member_rejects_forbidden_paths(campaign,relative):
    p=campaign/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='excluded'):a.safe_member(campaign,p)


def test_changed_terminal_reference_rejected(campaign):
    (campaign/'off/predictions/real_cell_000.prediction.json').write_text('{}')
    with pytest.raises(ValueError,match='reference mismatch'):a.select(campaign)


def test_packet_split_and_parts_only_restore_are_byte_exact(campaign,tmp_path):
    rows=a.select(campaign)
    out=tmp_path/'packet'
    result=a.package_selected(campaign,out,rows,part_size=1024)
    assert result['status']=='verified' and result['parts_reassembled_verified'] and result['restored_bytes_compared']
    assert len(result['part_refs'])>1
    assert result['restore']['packet_source']=='assembled_from_parts'
    with tarfile.open(out/'detailed_evidence.tar.gz','r:gz') as tar:
        assert sorted(tar.getnames())==sorted(x['relative_path'] for x in rows)
    assert all((out/'restored'/x['relative_path']).read_bytes()==(campaign/x['relative_path']).read_bytes() for x in rows)
    with pytest.raises(FileExistsError):a.package_selected(campaign,out,rows)


def test_source_change_before_archive_has_failed_terminal(campaign,tmp_path):
    rows=a.select(campaign);(campaign/'protocol.json').write_text('{}')
    with pytest.raises(RuntimeError,match='partial'):a.package_selected(campaign,tmp_path/'packet',rows)
    assert a.load(tmp_path/'packet/finish.json')['status']=='rejected'
