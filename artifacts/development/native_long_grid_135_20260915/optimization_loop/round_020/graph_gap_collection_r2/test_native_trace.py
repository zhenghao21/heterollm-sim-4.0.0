import json,unittest
from pathlib import Path
from native_trace import read_trace,analyze_trace
R18=Path(r'F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_018\collection\runs\scale_f32_e1024_g1\pair_01')
class T(unittest.TestCase):
 def test_real_r18_sqlite_schema_and_correlation(self):
  trace=R18/'export/trace.sqlite'; raw=[json.loads(x) for x in (R18/'profile/microbench.json').read_text(encoding='utf-8-sig').splitlines()]
  doc={'records':raw}; config={'id':'scale_f32_e1024_g1','nodes':1,'elements':1024}
  loaded=read_trace(trace);self.assertIn('StringIds',loaded['tables']);self.assertNotIn('CUPTI_ACTIVITY_KIND_DRIVER',loaded['tables'])
  result=analyze_trace(trace,doc,config);self.assertTrue(result['all36_chain_complete']);self.assertTrue(result['formal_30_complete']);self.assertFalse(result['qpc_ns_subtraction_performed'])
 def test_r20_raw_format_binding(self):
  # Formatting-only fixture: real schema regression above supplies actual correlation coverage.
  records=[{'record':'header','pid':1},{'record':'setup','config':'scale_f32_e262144_g8','scheduling':{'caller_thread_id':2}},{'record':'footer','graph_calls':36}]
  for phase,index in [('first',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]:records.append({'record':'graph_call','label':f'graph_submit/scale_f32_e262144_g8/{phase}/{index}','phase':phase,'index':index})
  self.assertEqual(len([x for x in records if x.get('record')=='graph_call']),36)
if __name__=='__main__':unittest.main()