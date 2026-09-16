"""R25 fixed full131 off/on prediction barrier and scoring; never run native."""
from pathlib import Path
import argparse, importlib.util, subprocess, sys
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r25_freezer',P/'freeze_candidate.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
s=f.s
ARMS=('off','on')
def now():return datetime.now(timezone.utc).isoformat()
def command(arm,*args):
    subprocess.run([sys.executable,str(P/arm/'source/tools/predict_stable_native_dataset.py'),'--output',str(P/arm),*args],check=True)
def no_scores():
    s.require(not any(list((P/a).glob('errors.*.json')) for a in ARMS),'scores exist; cannot predict again')
def guard():
    c,cr=s.read_json(P/'controls.json')
    for ref in c['refs']:s.verify_reference(ref)
    receipt,_=s.read_json(P/'freeze_receipt.json');protocol,_=s.read_json(P/'protocol.json')
    s.require(s.verify_reference(receipt['protocol_ref'])==s.reference(P/'protocol.json'),'freeze protocol mismatch')
    for row in protocol['source_refs']:s.verify_reference(row['ref'])
    freezes={}
    for arm in ARMS:
        freezes[arm],ref=s.read_json(P/arm/'freeze.json')
        s.require(s.verify_reference(receipt[arm])==ref,'freeze receipt mismatch')
        s.source_content(freezes[arm])
        for evidence in s.evidence_closure(freezes[arm]):s.verify_source_evidence_reference(evidence)
        for evidence in freezes[arm]['final_output_selection_binding']['evidence_refs'] if arm=='on' else []:s.verify_source_evidence_reference(evidence)
    f.compare_inputs(freezes['off'],freezes['on'])
    verifier=f.load('r25_native_verifier',f.ROOT/'tools/verify_fixed_native.py')
    s.require(verifier.verify_lock(f.LOOP/'state.json')==protocol['native_lock'],'native lock changed')
    return cr

def lock():
    no_scores()
    s.require(not any(list((P/a/'predictions').glob('*.prediction.json')) for a in ARMS),'lock before prediction')
    refs=[s.reference(p) for p in [Path(__file__),P/'freeze_candidate.py',P/'protocol.json',P/'freeze_receipt.json',
          f.LOOP/'round_023/summarize_ablation.py',f.LOOP/'round_023/evaluate_candidate.py',f.ROOT/'tools/verify_fixed_native.py',
          *(P/a/'freeze.json' for a in ARMS)]]
    f.write_new(P/'controls.json',{'created_utc':now(),'refs':refs,'denominator':131,'arms':list(ARMS),'strict_threshold_pct':10})
    guard()

def terminal(arm):
    freeze,fr=s.read_json(P/arm/'freeze.json');cells=s.unique_cells(freeze)
    refs={p.name.removesuffix('.prediction.json'):s.reference(p) for p in (P/arm/'predictions').glob('*.prediction.json')}
    s.require(set(refs)==set(cells),'all131 terminal predictions required')
    inputs={'freeze_ref':fr,'prediction_refs':refs};s.load_bundle(P/arm,'retained',inputs,full=True);return inputs

def barrier():
    b,ref=s.read_json(P/'predictions_complete.json');s.require(s.verify_reference(b['controls_ref'])==guard(),'barrier controls differ')
    for arm in ARMS:
        loaded=s.load_bundle(P/arm,'retained',b['arms'][arm],full=True)
        s.require(all(s.timestamp(p['finished_utc'])<=s.timestamp(b['created_utc']) for p in loaded['predictions'].values()),'barrier predates predictions')
    return b,ref

def full(workers,timeout):
    before=guard();no_scores()
    if (P/'predictions_complete.json').exists():barrier();return
    for arm in ARMS:
        old=[s.reference(p) for p in (P/arm/'predictions').glob('*.prediction.json')]
        command(arm,'--resume','--workers',str(workers),'--timeout-seconds',str(timeout))
        for ref in old:s.verify_reference(ref)
    inputs={a:terminal(a) for a in ARMS};s.require(guard()==before,'controls changed')
    f.write_new(P/'predictions_complete.json',{'created_utc':now(),'controls_ref':before,'arms':inputs,'terminal_count':262})
    barrier()

def score():
    b,bref=barrier();results={}
    for arm in ARMS:
        files=list((P/arm).glob('errors.*.json'));s.require(all(p.name=='errors.0001.json' for p in files),'unexpected score sequence')
        if not files:command(arm,'--score')
        loaded=s.load_bundle(P/arm,'retained',b['arms'][arm],full=True)
        checked=s.load_score(loaded,'errors.0001.json',loaded['cells'],s.reference(P/arm/'errors.0001.json'))
        s.require(s.timestamp(checked['created_utc'])>=s.timestamp(b['created_utc']),'score predates barrier')
        results[arm]={'score_ref':checked['ref'],'summary':s.summarize(checked['rows'])}
    barrier()
    f.write_new(P/'report.json',{'created_utc':now(),'barrier_ref':bref,'arms':results,'denominator_per_arm':131,'gate_B':'unvalidated','blind':False,'formal_success':False})
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['lock','full','score']);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--timeout-seconds',type=int,default=600);a=parser.parse_args()
    s.require(1<=a.workers<=8 and a.timeout_seconds>0,'invalid budget')
    if a.phase=='lock':lock()
    elif a.phase=='full':full(a.workers,a.timeout_seconds)
    else:score()
