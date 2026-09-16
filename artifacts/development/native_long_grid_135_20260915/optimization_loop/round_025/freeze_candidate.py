"""Create two new source-identical final-selection freezes; no native or scoring."""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys
P=Path(__file__).resolve().parent
ROOT=P.parents[4]
LOOP=P.parent
BASE=LOOP/'round_024/repaired/freeze.json'
REPLACEMENTS={'src/heterollm_sim/final_layer_output_selection.py','src/heterollm_sim/planner.py','tools/predict_stable_native_dataset.py'}
ADDITIONS={'tools/native_final_output_binding.py'}
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=load('r25_evidence',LOOP/'round_023/summarize_ablation.py')
r=load('r25_recipe',LOOP/'round_023/evaluate_candidate.py')
def write_new(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def blob(commit,path):return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])
def snapshot(commit,baseline,destination):
    s.require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'full reviewed commit required')
    before=s.source_content(baseline)
    s.require(REPLACEMENTS<=set(before) and not ADDITIONS & set(before),'unexpected baseline membership')
    destination.mkdir(parents=True,exist_ok=False)
    refs=[]
    for name in sorted(set(before)|ADDITIONS):
        data=blob(commit,name) if name in REPLACEMENTS|ADDITIONS else (Path(baseline['source']['root'])/name).read_bytes()
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        reference=s.reference(target)
        if name not in REPLACEMENTS|ADDITIONS:s.require({k:reference[k] for k in ('sha256','bytes')}==before[name],'inherited source differs')
        refs.append({'relative':name,'origin':'reviewed_commit' if name in REPLACEMENTS|ADDITIONS else 'R24_frozen','ref':reference})
    return refs

def compare_inputs(off,on):
    a=s.unique_cells(off);b=s.unique_cells(on);s.require(set(a)==set(b),'cell membership differs')
    s.require(s.source_content(off)==s.source_content(on),'two arms need identical code')
    for ident in a:
        x=dict(a[ident]['static_inputs']);y=dict(b[ident]['static_inputs'])
        s.require('final_output_selection' not in x and 'final_output_selection_binding' not in x,'control must be default off')
        s.require(y.pop('final_output_selection',None) is True,'candidate switch missing')
        proof=y.pop('final_output_selection_binding',None)
        s.require(isinstance(proof,dict) and proof['status'] in ('conditional','uncovered'),'candidate proof missing')
        # Extractor copies differ by arm; compare only this known location.
        from copy import deepcopy
        x,y=deepcopy(x),deepcopy(y)
        for inputs in (x,y):
            ref=inputs['retained_kv_warmup_evidence']['extractor_ref']
            ref['path']='arm_source_copy'
        s.require(x==y,'non-treatment static difference: '+ident)
    return len(a)

def freeze(commit):
    s.require(blob(commit,str(Path(__file__).relative_to(ROOT)).replace('\\','/'))==Path(__file__).read_bytes(),'driver differs from reviewed commit')
    s.require(not (P/'protocol.json').exists(),'new freeze only; preserve any prior attempt')
    baseline,base_ref=s.read_json(BASE)
    verifier=load('r25_native_lock',ROOT/'tools/verify_fixed_native.py')
    native=verifier.verify_lock(LOOP/'state.json')
    s.require(native['selected_cells']==131 and native['selection_sha256']==s.SELECTION_SHA,'native changed')
    source=P/'execution_source';refs=snapshot(commit,baseline,source)
    protocol={'schema':'r25-final-selection-two-arm/v1','reviewed_commit':commit,'baseline_ref':base_ref,'native_lock':native,
        'driver_ref':s.reference(__file__),'recipe_ref':s.reference(LOOP/'round_023/evaluate_candidate.py'),
        'helper_ref':s.reference(LOOP/'round_023/summarize_ablation.py'),'source_refs':refs,
        'arms':['off','on'],'denominator':131,'strict_threshold_pct':10,'blind':False,'gate_B':'unvalidated',
        'affected_scope':'source-qualified ordinary completion; ordinary final FFN rows and hybrid final norm rows plus index/gather work',
        'expected_direction':'ordinary FFN work decreases for R<B; hybrid final norm work increases for R<B; gather/index adds work; no guaranteed latency direction',
        'workflow':'same-source off/on full predictions before score; preserve failures; no native remeasurement'}
    write_new(P/'protocol.json',protocol)
    api=load('r25_snapshot_api',source/'tools/predict_stable_native_dataset.py')
    for name,module in list(sys.modules.items()):
        if name=='tools' or name.startswith('tools.') or name=='heterollm_sim' or name.startswith('heterollm_sim.'):
            path=getattr(module,'__file__',None)
            s.require(path is None or Path(path).resolve().is_relative_to(source.resolve()),'import escaped snapshot: '+name)
    for arm in ('off','on'):
        args=r.freeze_arguments(baseline,'retained');args['final_output_selection']=arm=='on'
        api.freeze_selection(Path(baseline['selection_ref']['path']),P/arm,**args)
    off,oref=s.read_json(P/'off/freeze.json');on,nref=s.read_json(P/'on/freeze.json')
    count=compare_inputs(off,on)
    s.require(verifier.verify_lock(LOOP/'state.json')==native,'native changed during freeze')
    write_new(P/'freeze_receipt.json',{'protocol_ref':s.reference(P/'protocol.json'),'off':oref,'on':nref,'compared_cells':count})
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reviewed-commit',required=True)
    freeze(parser.parse_args().reviewed_commit)
