"""Preserve R25 candidate startup rejection without scoring or changing freezes."""
from pathlib import Path
import importlib.util,json,sys
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'on/source'),str(P/'on/source/src')]
from tools import native_final_output_binding as binding
spec=importlib.util.spec_from_file_location('r25_failure_driver',P/'run_candidate.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
s=r.s

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [row for key in sorted(set(a)|set(b)) for row in (differences(a[key],b[key],path+'/'+key) if key in a and key in b else [{'path':path+'/'+key,'frozen':a.get(key),'rederived':b.get(key)}])]
    return [] if a==b else [{'path':path,'frozen':a,'rederived':b}]
def main():
    controls=r.guard();r.no_scores()
    s.require(not (P/'predictions_complete.json').exists(),'completed campaign cannot be declared rejected')
    s.require(not list((P/'on/predictions').glob('*.prediction.json')),'on has predictions; inspect different failure')
    before=r.terminal('off')
    freeze,fr=s.read_json(P/'on/freeze.json');bad=[]
    for cell in freeze['cells']:
        i=cell['static_inputs'];proof=i[binding.INPUT_KEY]
        expected=binding.derive_cell(proof['source_contract'],cell_id=i['cell_id'],model_ref=i['prediction_model_ref'],
            model_scope={'architecture':proof['gguf_architecture']},config=i['config'],native_refs=[i['runtime_ref'],*i['runtime_module_refs']],sampling=i.get('sampling_binding'))
        diff=differences(proof,expected)
        s.require({x['path'] for x in diff}=={'/config/flash_attn','/config/op_offload','/content_sha256'},'unexpected mismatch scope')
        bad.append({'cell_id':cell['cell_id'],'differences':diff})
    s.require(len(bad)==131,'expected full candidate static rejection scope')
    report={'schema':'r25-closed-startup-rejection/v1','created_utc':r.now(),'status':'candidate_startup_rejected',
        'controls_ref':controls,'driver_ref':s.reference(__file__),'off_terminal_inputs':before,'on_freeze_ref':fr,
        'on_terminal_predictions':0,'fixed_denominator_per_arm':131,'observed_runner_returncode':1,
        'observed_error':'ValueError: final output binding: cell proof differs from frozen static inputs',
        'independently_rederived_mismatches':bad,'source_adapters_unchanged':True,'scored':False,'gate_A':'not_evaluable','gate_B':'unvalidated',
        'reason':'Per-cell proof captured raw config before flash_attention alias and verified host-offload normalization; worker checked normalized config.',
        'repair_requirement':'derive proof from final static_inputs in a new frozen version; preserve this rejected campaign'}
    s.require(r.guard()==controls,'controls changed')
    r.f.write_new(P/'execution_closed.json',report)
    print(json.dumps({'status':report['status'],'off_terminals':131,'on_terminals':0,'mismatches':len(bad),'scored':False,'receipt':s.reference(P/'execution_closed.json')}))
if __name__=='__main__':main()
