"""Pure synthetic R20 A/B entry regressions; no compile, DLL load, model, GPU or timing run."""
import copy, importlib.util, json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reset_entry',P/'entry.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
ID={'cpu_brand':'test','cpuid_signature':1,'active_processor_group_count':1,'active_processor_count_group0':16,'group':0,'logical_cpu':0,'thread_affinity_mask':1}
F=['os_max_mhz','os_reported_current_mhz','os_limit_mhz']
def stage(ticks=None, current=4500):
 b={'group':0,'logical_processor':0,'thread_affinity_mask':1,'os_max_mhz':5000,'os_reported_current_mhz':4500,'os_limit_mhz':100}; a={**b,'os_reported_current_mhz':current}; stable=b==a
 return {'stage':'original_dll_topk_apply','numeric_quality':'exact_pass','warmup_calls':16,'steady_repeats':64,'first_use_ticks':1,'steady_raw_ticks':[100]*64 if ticks is None else ticks,'reported_frequency_fields_checked':F,'reported_frequency_stable':stable,'timing_usable':stable,'diagnostic_only':not stable,'frequency_changed_diagnostic_only':not stable,'cpu_before':b,'cpu_after':a}
def doc(index, ticks=None, current=4500):
 arm=next(x['reset_policy'] for x in e.process_plan() if x['process_index']==index); cases=[]
 for v in (32768,131072,262144):
  for pat in ('monotone_ascending','deterministic_random_permutation'):
   cases.append({'vocabulary_size':v,'pattern':pat,'split':'quality_diagnosis','reset_policy':arm,'reset_outside_clock_window':True,'candidate_record_bytes':12,'top_k':1,'first_use_not_pooled_with_steady':True,'stages':[stage(ticks,current)]})
 return {'schema':'cpu-reset-independent-probe/v1','status':'complete','process_index':index,'process_id':100+index,'logical_cpu':0,'GPU_context_created':False,'model_loaded':False,'full_sampler_chain_measured':False,'actual_cpu_identity':copy.deepcopy(ID),'qpc_frequency':10_000_000,'observer_empty_bracket_ticks':[1]*64,'cases':cases}
def results(docs): return [{'document':d,'frequency':e.validate_result(d,d['process_index'],0,ID)} for d in docs]
def valid(): return [doc(x['process_index']) for x in e.process_plan()]
def test_plan_and_valid_quality():
 assert e.process_plan()==[{'process_index':i,'reset_policy':p} for i,p in [(0,'memcpy_baseline'),(3,'source_candidate_loop'),(1,'memcpy_baseline'),(4,'source_candidate_loop'),(2,'memcpy_baseline'),(5,'source_candidate_loop')]]
 q=e.summarize_quality(results(valid()),{'path':'id','sha256':'0'*64,'bytes':1}); assert q['stage_records_total']==36 and q['case_process_records_total']==36 and q['cross_process_case_stage_groups_total']==12 and q['timing_usable']
@pytest.mark.parametrize('mut,match',[('wrong_arm','wrong arm'),('candidate','independent stage'),('schema','measurement schema')])
def test_structure_rejections(mut,match):
 d=doc(0)
 if mut=='wrong_arm': d['cases'][0]['reset_policy']='source_candidate_loop'
 if mut=='candidate': d['cases'][0]['stages'].append(copy.deepcopy(d['cases'][0]['stages'][0])); d['cases'][0]['stages'][1]['stage']='candidate_loop'
 if mut=='schema': d['schema']='wrong'
 with pytest.raises(ValueError,match=match): e.validate_result(d,0,0,ID)
def test_pid_missing_and_mixed_arm_grouping_rejected():
 ds=valid(); ds[1]['process_id']=ds[0]['process_id']
 with pytest.raises(ValueError,match='six independent'): e.summarize_quality(results(ds),{'path':'id','sha256':'0'*64,'bytes':1})
 ds=valid()[:-1]
 with pytest.raises(ValueError,match='six independent'): e.summarize_quality(results(ds),{'path':'id','sha256':'0'*64,'bytes':1})
 ds=valid(); mixed=next(x for x in ds if x['process_index']==3); mixed['cases'][0]['reset_policy']='memcpy_baseline'
 with pytest.raises(ValueError,match='wrong arm'): e.validate_result(mixed,3,0,ID)
def test_frequency_dispersion_cross_and_observer_diagnostic():
 ds=valid(); ds[0]['cases'][0]['stages'][0]['cpu_after']['os_reported_current_mhz']=4400; st=ds[0]['cases'][0]['stages'][0]; st.update(reported_frequency_stable=False,timing_usable=False,diagnostic_only=True,frequency_changed_diagnostic_only=True)
 q=e.summarize_quality(results(ds),{'path':'id','sha256':'0'*64,'bytes':1}); assert not q['timing_usable'] and 'frequency_drift' in q['timing_quality_failure_reasons']
 ds=valid(); ds[0]['cases'][0]['stages'][0]['steady_raw_ticks']=[100]*57+[200]*7
 q=e.summarize_quality(results(ds),{'path':'id','sha256':'0'*64,'bytes':1}); assert 'steady_dispersion' in q['timing_quality_failure_reasons']
 ds=valid(); ds[5]=doc(5,[106]*64)
 q=e.summarize_quality(results(ds),{'path':'id','sha256':'0'*64,'bytes':1}); assert 'cross_process_median_deviation' in q['timing_quality_failure_reasons']
 ds=valid(); ds[0]['observer_empty_bracket_ticks']=[2]*64
 q=e.summarize_quality(results(ds),{'path':'id','sha256':'0'*64,'bytes':1}); assert 'observer_overhead' in q['timing_quality_failure_reasons']
def test_protocol_one_series_and_disclosed_sizes():
 p=json.loads((P/'protocol.json').read_text()); source=(P/'entry.py').read_text(); assert p['run_budget']['max_series_this_revision']==1 and 'already-used quality diagnosis point' in p['reset_ab']['holdout'] and "'holdout_vocabulary_sizes': []" in source and "'all_sizes_quality_diagnosis_only': True" in source
