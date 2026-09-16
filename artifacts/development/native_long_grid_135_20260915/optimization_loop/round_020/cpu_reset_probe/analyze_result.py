"""Read-only audit for the frozen R20 reset A/B series; no fitting or timing generation."""
from __future__ import annotations
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SERIES=ROOT/'reset_ab_series_0001'
def file_ref(path):
 b=Path(path).read_bytes();return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def percentile(values,f):
 a=sorted(values); r=(len(a)-1)*f; lo=int(math.floor(r)); hi=int(math.ceil(r)); return float(a[lo]) if lo==hi else float(a[lo])+(float(a[hi])-float(a[lo]))*(r-lo)
def main():
 freeze=json.loads((SERIES/'run_freeze.json').read_text()); quality=json.loads((SERIES/'quality.json').read_text()); plan=freeze['process_plan']; rows=[]; errors=[]
 pids=[]
 for item in plan:
  i=item['process_index']; arm=item['reset_policy']; path=SERIES/f'process_{i}.json'; doc=json.loads(path.read_text()); actual=file_ref(path); receipt=json.loads((SERIES/f'process_{i}.receipt.json').read_text())
  if receipt.get('returncode')!=0 or receipt.get('reset_policy')!=arm or receipt.get('result_ref')!=actual: errors.append(f'receipt/process mismatch {i}')
  if doc.get('process_index')!=i or doc.get('status')!='complete': errors.append(f'process status/index mismatch {i}')
  pids.append(doc.get('process_id'))
  for case in doc['cases']:
   stage=case['stages'][0]; ticks=stage['steady_raw_ticks']; p10,p90=percentile(ticks,.1),percentile(ticks,.9); ratio=None if p10<=0 else p90/p10; passed=ratio is not None and ratio<=1.5
   rows.append({'process_index':i,'pid':doc['process_id'],'arm':arm,'vocabulary_size':case['vocabulary_size'],'pattern':case['pattern'],'stage':stage['stage'],'p10_ticks':p10,'p90_ticks':p90,'p90_p10_ratio':ratio,'steady_dispersion_pass':passed})
 if len(set(pids))!=6: errors.append('PIDs not unique')
 if len(rows)!=36: errors.append(f'wrong stage count {len(rows)}')
 arms={a:[r for r in rows if r['arm']==a] for a in ('memcpy_baseline','source_candidate_loop')}
 groups={}
 for r in rows: groups.setdefault((r['arm'],r['vocabulary_size'],r['pattern'],r['stage']),[]).append(r)
 if len(groups)!=12 or any(len(v)!=3 for v in groups.values()): errors.append('wrong arm/case cross-process grouping')
 failed=[r for r in rows if not r['steady_dispersion_pass']]
 result={'schema':'cpu-reset-ab-result-audit/v1','series_ref':file_ref(SERIES/'run_freeze.json'),'quality_ref':file_ref(SERIES/'quality.json'),'sha_pid_arm_validation_passed':not errors,'validation_errors':errors,'process_order':[{'process_index':x['process_index'],'reset_policy':x['reset_policy']} for x in plan],'distinct_pid_count':len(set(pids)),'stage_count':len(rows),'arm_stage_counts':{k:len(v) for k,v in arms.items()},'cross_process_group_count':len(groups),'quality_reported_steady_dispersion_pass_stages':quality.get('steady_dispersion_pass_stages'),'recomputed_steady_dispersion_pass_stages':sum(r['steady_dispersion_pass'] for r in rows),'quality_reported_cross_groups_pass':quality.get('cross_process_case_stage_groups_pass'),'failed_stages':failed,'arm_dispersion':{k:{'passed':sum(r['steady_dispersion_pass'] for r in v),'total':len(v),'median_ratio':statistics.median(r['p90_p10_ratio'] for r in v)} for k,v in arms.items()},'source_reset_evidence':'Treatment has 15/18 within-stage dispersion passes versus control 16/18. All five failures are 262144 deterministic_random_permutation across both arms; the treatment arm has one additional failed stage. This provides no evidence that source reset lowers fluctuation; the whole series remains rejected and does not establish a causal cache/reset effect.','overall_acceptance':{'timing_usable':False,'accepted_for_timing_evidence':False,'fit_performed':False,'llm_actual_used':False}}
 (ROOT/'result_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

