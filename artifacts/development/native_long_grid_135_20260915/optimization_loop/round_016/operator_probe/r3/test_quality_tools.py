import unittest,copy,json
from pathlib import Path
from full_raw_audit import audit
from assess import assess
P=Path(__file__).resolve().parent
C=json.loads((P/'protocol.json').read_text())['configs'][0]
def fixture(control=False,wall_scale=1):
 count=min(4096,C['M']*C['N']);rows=[{'n_index':j,'m_index':0,'reference':1.,'math_reference':1.,'actual':1.,'math_absolute_error':0.,'path_reference':1.,'path_absolute_error':0.,'math_pass':True,'path_pass':True,'pass':True} for j in range(count)]
 numeric={'finite_all_outputs':True,'passed':True,'sample_count':count,'samples':rows,'path_max_absolute_error':0.,'path_rmse':0.,'math_passed':True}
 doc={'schema':'single-operator-surface-probe/v2','status':'measured','M':C['M'],'N':C['N'],'K':C['K'],'weight_format':C['quant'],'input_dtype':'F32','output_dtype':'F32','layout':'ordinary_contiguous_2d','device':'cuda','cuda_index':0,'threads':1,'graph_computations_per_batch':1,'graph_compute_calls':36,'warmup_requested':5,'formal_repeats_requested':30,'nvtx_enabled':True,'nvtx_scope':'one full graph call including event records, submit, wait and event queries; excluding cache sweep and numeric validation','expected_source_path':C['expected_source_path'],'actual_dispatch_trace_confirmed':False,'cache_policy':'untimed_read_write_sweep_at_least_4x_device_L2','gpu_l2_bytes':1<<20,'cache_eviction_bytes':128<<20,'modules_stable':True,'loaded_modules_before':[{'name':'same'}],'loaded_modules_after':[{'name':'same'}],'environment':{'GGML_CUDA_DISABLE_GRAPHS':'1'},'qpc_frequency':1000000000,'control_mode':control,'correctness_contract':{'absolute_tolerance':.05,'relative_tolerance':.03,'path_absolute_tolerance':.0001,'path_relative_tolerance':.00001,'reference_mode':'dual_math_and_source_path','source_runtime_equivalence_proven':False},'first_call_correctness':numeric,'final_correctness':numeric,'quantization':{'packed_weight_sha256':'a'*64,'input_sha256':'b'*64},'runs':[]}
 for j,(phase,index) in enumerate([('first_call',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]):
  base=100000+j*10000;keys=['qpc_pre_sync_start','qpc_pre_sync_end','qpc_evict_start','qpc_evict_submit_end','qpc_evict_end','qpc_nvtx_push_start','qpc_nvtx_push_end','qpc_start','qpc_record_begin_start','qpc_record_begin_end','qpc_submit_start','qpc_submit_end','qpc_record_end_start','qpc_record_end_end','qpc_wait_start','qpc_wait_end','qpc_end','qpc_nvtx_pop_start','qpc_nvtx_pop_end','qpc_validation_start','qpc_validation_end'];r={k:base+i*100*wall_scale for i,k in enumerate(keys)};r['qpc_wait_end']=r['qpc_end'];r.update(phase=phase,index=index,graph_computations=1,host_wall_ns=r['qpc_end']-r['qpc_start'],host_per_graph_ns=r['qpc_end']-r['qpc_start'],ggml_status=0,cuda_submit_status=0,cuda_wait_status=0,eviction_status=0,cuda_begin_record_status=0,cuda_end_record_status=0,cuda_query_after_wait=0,cuda_elapsed_status=0,cuda_query_before_wait=600,event_envelope_ms=None if control else .001,correctness=copy.deepcopy(numeric),nvtx_label=f"operator_surface/v1|phase={phase}|index={index}|op=MUL_MAT|M={C['M']}|N={C['N']}|K={C['K']}|quant={C['quant']}|input=F32|output=F32|layout=contiguous2d|expected_path={C['expected_source_path']}")
  if control:
   for k in ['qpc_record_begin_start','qpc_record_begin_end','qpc_record_end_start','qpc_record_end_end']:r[k]=0
  doc['runs'].append(r)
 return doc
class Quality(unittest.TestCase):
 def test_complete_modes(self):
  for mode in (True,False):self.assertTrue(audit(fixture(mode),C)['valid_raw'])
 def test_every_phase_checked(self):
  for i in (0,1,6,35):
   d=fixture();d['runs'][i]['correctness']['samples'][0]['actual']=2.;self.assertFalse(audit(d,C)['valid_raw'])
 def test_missing_run_reject(self):
  d=fixture();d['runs'].pop();self.assertFalse(audit(d,C)['valid_raw'])
 def test_cache_inside_nvtx_reject(self):
  d=fixture();d['runs'][0]['qpc_evict_end']=d['runs'][0]['qpc_submit_end'];self.assertFalse(audit(d,C)['valid_raw'])
 def test_math_error_diagnostic_retained(self):
  d=fixture();d['runs'][0]['correctness']['samples'][0]['math_absolute_error']=9;self.assertFalse(audit(d,C)['valid_raw'])
 def test_no_profile_no_calibration(self):
  r=assess(fixture(),fixture(True),None,C);self.assertFalse(r['timing_quality_pass']);self.assertFalse(r['calibration_eligible'])
 def test_profile_perturbation_reject(self):
  r=assess(fixture(),fixture(True),fixture(False,2),C);self.assertFalse(r['timing_quality_pass'])
 def test_good_quality_needs_trace(self):
  r=assess(fixture(),fixture(True),fixture(),C);self.assertTrue(r['timing_quality_pass']);self.assertFalse(r['calibration_eligible'])
 def test_identity_pair_reject(self):
  d=fixture();d['quantization']['input_sha256']='c'*64;self.assertEqual(assess(fixture(),fixture(True),d,C)['status'],'rejected_identity_pair')
 def test_submit_only_marker_rejected(self):
  d=fixture();d['runs'][0]['qpc_nvtx_pop_start']=d['runs'][0]['qpc_submit_end'];self.assertFalse(audit(d,C)['valid_raw'])
 def test_profiler_does_not_kill(self):
  self.assertIn('--kill=false',(P/'invoke.ps1').read_text())
 def test_single_graph_regression(self):
  s=(P/'stream_event_probe.cpp').read_text();self.assertNotIn('call<o.batch',s);self.assertEqual(s.count('ggml_backend_graph_compute_async(r.backend,g)'),1);self.assertIn('evict_bytes<size_t(l2_bytes)*4',s)
if __name__=='__main__':unittest.main(verbosity=2)
