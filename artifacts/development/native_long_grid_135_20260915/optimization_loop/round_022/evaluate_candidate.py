"""R22 four-way experiment phases; never executes a native benchmark.
freeze/lock/anchors/score-anchors/full/summary are independently scheduled.
Existing static freezes may be adopted before any predictions or scores exist.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

P=Path(__file__).resolve().parent
ROOT=P.parents[4]
STATE=P.parent/'state.json'
BASELINE=P.parent/'round_021/physical/freeze.json'
_spec=importlib.util.spec_from_file_location('r22_saved_ablation_report',P/'summarize_ablation.py')
s=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(s)
VARIANTS=s.VARIANTS


def now():return datetime.now(timezone.utc).isoformat()


def write_new(path,document):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(document,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return s.reference(path)


def native_lock():
    sys.path[:0]=[str(ROOT),str(ROOT/'src')]
    from tools.verify_fixed_native import verify_lock
    result=verify_lock(STATE)
    s.require(result.get('selection_sha256')==s.SELECTION_SHA and result.get('selected_cells')==s.DENOMINATOR,'fixed native selection changed')
    return result


def protocol_only():
    protocol,ref=s.read_json(P/'evaluation_protocol.json');s.check_protocol(protocol)
    for key,path in [('driver_ref',Path(__file__)),('summarizer_ref',P/'summarize_ablation.py')]:
        s.require(s.verify_reference(protocol[key])==s.reference(path),'protocol script changed: '+key)
    s.verify_reference(protocol['baseline_freeze_ref'])
    return protocol,ref


def reject_any_scores():
    for variant in VARIANTS:
        s.require(not list((P/variant).glob('errors.*.json')),'scores already exist; prediction phase cannot resume: '+variant)


def freeze_arguments(baseline,variant,hardware_document=None):
    mode=s.TREATMENTS[variant];loop=P.parent
    result=dict(data_root=Path(baseline['data_root']),
        model_snapshot_map={key:ref['path'] for key,ref in baseline['model_snapshot_map'].items()},
        runtime_build_audit_path=Path(baseline['runtime_build_audit']['audit_ref']['path']),
        recurrent_batching_contract_path=Path(baseline['recurrent_batching']['contract_ref']['path']),
        slot_order_contract_path=Path(baseline['slot_order']['contract_ref']['path']),
        host_offload_source_contract_path=Path(baseline['host_offload_source']['contract_ref']['path']),
        tensor_storage_contract_path=Path(baseline['tensor_storage']['contract_ref']['path']),tensor_storage_f32_hidden=True,
        gpu_invocation_contract_path=loop/'round_018/conversion_source_contract.json',
        gpu_mmq_source_costs=mode['mmq'],gpu_conversion_cta_costs=mode['cta'],
        sampling_contract_path=loop/'round_018/sampling_parity/static_sampling_contract.json' if mode['sampling'] else None,
        nonflash_kv_view_source_contract_path=loop/'round_021/nonflash_kv_view_source_contract.json' if mode['nonflash'] else None,
        mmvq_vector_issue_bound=mode['issue'])
    if mode['issue'] and hardware_document is not None:result['mmvq_issue_hardware_document_path']=hardware_document
    return result


def freeze(hardware_document=None):
    start=native_lock();reject_any_scores()
    s.require(not (P/'evaluation_controls.json').exists(),'controls already locked; cannot freeze again')
    for variant in VARIANTS:
        s.require(not list((P/variant/'predictions').glob('*.prediction.json')),'protocol/freeze preparation must precede predictions')
    baseline,baseline_ref=s.read_json(BASELINE)
    s.require(baseline.get('selection_sha256')==s.SELECTION_SHA,'baseline fixed selection differs')
    cells=s.unique_cells(baseline);ids=s.anchors(cells)
    path=P/'evaluation_protocol.json'
    if path.exists():
        protocol,_=protocol_only();s.require(protocol['anchor_ids']==ids,'anchor set changed')
    else:
        protocol={'schema':'mmvq-four-way-ablation/v1','created_utc':now(),'native_lock':start,
            'variants':list(VARIANTS),'candidate':s.CANDIDATE,'treatments':s.TREATMENTS,'declared_variants':s.DECLARED_VARIANTS,
            'anchor_ids':ids,'anchor_denominator':s.ANCHORS,'full_denominator':s.DENOMINATOR,
            'threshold_pct_strict':s.THRESHOLD,'metrics':list(s.METRICS),'gate_B':'unvalidated',
            'is_blind':False,'formal_acceptance':False,'native_remeasurement':False,'calibration_added':False,'accuracy_selected_subset':False,
            'baseline_freeze_ref':baseline_ref,'driver_ref':s.reference(Path(__file__)),'summarizer_ref':s.reference(P/'summarize_ablation.py'),
            'existing_static_freezes_adopted':[v for v in VARIANTS if (P/v/'freeze.json').is_file()],
            'registered_before_predictions':True,
            'workflow':'Freeze all four full131 variants, lock controls, save all80 terminal anchor predictions, then score all four; continue cta_issue to fixed131 while retaining failures.',
            'continuation_criterion':'Mechanism/source identity checks; never select cells or censor failures based on errors.',
            'issue_scope':'Conditional source/PTX dot issue floor with complete source snapshot and header-compilation condition; no actual DP4A throughput calibration.',
            'comparisons':[{'before':'current','after':'cta_only'},{'before':'cta_only','after':'cta_issue'}]}
        write_new(path,protocol)
    for variant in VARIANTS:
        output=P/variant
        if (output/'freeze.json').is_file():
            print('Adopt existing static freeze '+variant,flush=True)
            continue
        sys.path[:0]=[str(ROOT),str(ROOT/'src')]
        from tools.predict_stable_native_dataset import freeze_selection
        freeze_selection(Path(baseline['selection_ref']['path']),output,**freeze_arguments(baseline,variant,hardware_document))
        print('Frozen '+variant,flush=True)
    protocol_only();s.frozen_bundles(P,protocol)
    s.require(native_lock()==start,'native lock changed during freeze')


def lock_controls():
    start=native_lock();reject_any_scores();protocol,_=protocol_only()
    for variant in VARIANTS:s.require(not list((P/variant/'predictions').glob('*.prediction.json')),'controls must lock before predictions')
    _,sources,_=s.frozen_bundles(P,protocol)
    files=[Path(__file__),P/'summarize_ablation.py',P/'evaluation_protocol.json',*(P/v/'freeze.json' for v in VARIANTS)]
    payload={'schema':'mmvq-four-way-controls/v1','created_utc':now(),'variants':list(VARIANTS),
        'files':[s.reference(path) for path in files],'same_source_closure':True,
        'source_file_count':len(sources['current']),'source_content_sha256':s.stable_hash(sources['current'])}
    path=P/'evaluation_controls.json'
    if path.exists():s.check_controls(P)
    else:write_new(path,payload)
    guard();s.require(native_lock()==start,'native lock changed while locking controls')


def guard():
    lock=native_lock();protocol,controls=s.check_controls(P);protocol_only()
    bundles,sources,closure=s.frozen_bundles(P,protocol)
    s.require(s.stable_hash(sources['current'])==controls['source_content_sha256'] and len(sources['current'])==controls['source_file_count'],'locked source closure differs')
    return protocol,controls,bundles,lock


def run(argv):
    result=subprocess.call([str(arg) for arg in argv])
    if result:raise RuntimeError('simulation phase command failed with exit '+str(result))


def check_end(before):
    after=guard()
    s.require(after[1]==before[1] and after[3]==before[3],'control or native lock changed during phase')


def predict_variant(variant,ids,workers,timeout):
    output=P/variant
    argv=[sys.executable,output/'source/tools/predict_stable_native_dataset.py','--output',output,'--resume','--workers',workers,'--timeout-seconds',timeout]
    if ids is not None:
        for ident in ids:argv.extend(['--cell-id',ident])
    run(argv)


def prediction_inputs(bundle,full=False):
    ids=sorted(bundle['cells']) if full else bundle['ids']
    refs={ident:s.reference(bundle['directory']/'predictions'/(ident+'.prediction.json')) for ident in ids}
    inputs={'freeze_ref':bundle['freeze_ref'],'prediction_refs':refs}
    checked=s.load_bundle(bundle['directory'],bundle['variant'],inputs,full=full)
    return inputs,checked


def anchors(workers=4,timeout=600):
    before=guard();protocol,controls,bundles,_=before;reject_any_scores()
    barrier_path=P/'anchors_predictions.json'
    if barrier_path.exists():s.load_prediction_barrier(P,'anchors',controls);check_end(before);return
    for variant in VARIANTS:
        guard();predict_variant(variant,protocol['anchor_ids'],workers,timeout);check_end(before)
    inputs={variant:prediction_inputs(bundles[variant])[0] for variant in VARIANTS}
    check_end(before)
    write_new(barrier_path,{'schema':'r22-prediction-barrier/v1','phase':'anchors','created_utc':now(),
        'controls_ref':controls['controls'],'variants':inputs,'native_answers_used':False,'terminal_failures_preserved':True})
    s.load_prediction_barrier(P,'anchors',controls);check_end(before)


def score_anchors():
    before=guard();protocol,controls,_,_=before
    barrier,barrier_ref,bundles=s.load_prediction_barrier(P,'anchors',controls)
    receipt_path=P/'anchors_scores.json'
    if receipt_path.exists():s.assess(P,'anchors');check_end(before);return
    for variant,bundle in bundles.items():
        actual={path.name for path in (bundle['directory']/'predictions').glob('*.prediction.json')}
        s.require(actual=={ident+'.prediction.json' for ident in protocol['anchor_ids']},'full prediction appeared before all four anchor scores')
        s.require(not any(path.name!='errors.0001.json' for path in bundle['directory'].glob('errors.*.json')),'unexpected score sequence')
    scores={}
    for variant in VARIANTS:
        guard();s.load_prediction_barrier(P,'anchors',controls)
        path=P/variant/'errors.0001.json'
        if not path.exists():run([sys.executable,P/variant/'source/tools/predict_stable_native_dataset.py','--output',P/variant,'--score'])
        score_ref=s.reference(path)
        score=s.load_score(bundles[variant],'errors.0001.json',protocol['anchor_ids'],score_ref)
        s.require(s.timestamp(score['created_utc'])>=s.timestamp(barrier['created_utc']),'score preceded all-four prediction barrier')
        scores[variant]=score_ref;s.load_prediction_barrier(P,'anchors',controls);check_end(before)
    write_new(receipt_path,{'schema':'r22-score-receipt/v1','phase':'anchors','created_utc':now(),
        'controls_ref':controls['controls'],'prediction_barrier_ref':barrier_ref,'scores':scores})
    s.assess(P,'anchors');check_end(before)


def full(workers=4,timeout=600):
    before=guard();_,controls,bundles,_=before
    s.assess(P,'anchors')
    anchor_receipt,anchor_ref=s.read_phase_receipt(P,'anchors','scores',controls)
    path=P/'full_predictions.json'
    if not path.exists():
        s.require(not (P/s.CANDIDATE/'errors.0002.json').exists(),'full score exists before terminal prediction barrier')
        predict_variant(s.CANDIDATE,None,workers,timeout);check_end(before)
        inputs,checked=prediction_inputs(bundles[s.CANDIDATE],full=True)
        s.require(all(s.timestamp(pred['created_utc'])>=s.timestamp(anchor_receipt['created_utc']) for ident,pred in checked['predictions'].items() if ident not in checked['ids']),'full continuation preceded completed anchor scores')
        write_new(path,{'schema':'r22-prediction-barrier/v1','phase':'full','created_utc':now(),
            'controls_ref':controls['controls'],'variants':{s.CANDIDATE:inputs},'native_answers_used':False,
            'terminal_failures_preserved':True,'anchor_score_receipt_ref':anchor_ref})
    barrier,barrier_ref,full_bundles=s.load_prediction_barrier(P,'full',controls)
    if not (P/'full_scores.json').exists():
        scores_path=P/s.CANDIDATE/'errors.0002.json'
        s.require(not any(p.name not in ('errors.0001.json','errors.0002.json') for p in scores_path.parent.glob('errors.*.json')),'unexpected full score sequence')
        if not scores_path.exists():run([sys.executable,P/s.CANDIDATE/'source/tools/predict_stable_native_dataset.py','--output',P/s.CANDIDATE,'--score'])
        score_ref=s.reference(scores_path)
        score=s.load_score(full_bundles[s.CANDIDATE],'errors.0002.json',full_bundles[s.CANDIDATE]['planned'],score_ref)
        s.require(s.timestamp(score['created_utc'])>=s.timestamp(barrier['created_utc']),'full score preceded prediction barrier')
        s.load_prediction_barrier(P,'full',controls);check_end(before)
        write_new(P/'full_scores.json',{'schema':'r22-score-receipt/v1','phase':'full','created_utc':now(),
            'controls_ref':controls['controls'],'prediction_barrier_ref':barrier_ref,'scores':{s.CANDIDATE:score_ref}})
    s.assess(P,'full');check_end(before)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','lock','anchors','score-anchors','full','summary'])
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--timeout-seconds',type=float,default=600)
    parser.add_argument('--hardware-document',type=Path);parser.add_argument('--scope',choices=['anchors','full'],default='anchors');parser.add_argument('--heatmap',action='store_true')
    args=parser.parse_args(argv)
    s.require(1<=args.workers<=8 and args.timeout_seconds>0,'invalid execution budget')
    if args.phase=='freeze':freeze(args.hardware_document)
    elif args.phase=='lock':lock_controls()
    elif args.phase=='anchors':anchors(args.workers,args.timeout_seconds)
    elif args.phase=='score-anchors':score_anchors()
    elif args.phase=='full':full(args.workers,args.timeout_seconds)
    else:
        before=guard();result=s.assess(P,args.scope);paths=s.write_report(P,result,args.heatmap);check_end(before)
        print(json.dumps({'outputs':paths,'gate_A':result['gate_A'],'gate_B':result['gate_B']},ensure_ascii=False))

if __name__=='__main__':main()
