"""Controller-only tests: no prepare/run/native/GPU invocation."""
import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('collect_under_test',P/'collect.py'); c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def protocol(): return {'execution':{'pilot':{'id':'pilot','config':'scale_f32_e262144_g8','pairs_per_arm':3,'native_processes':12}},'profiler':{'profile_options':['--x']}}
def manifest(): return {'executable':{'path':'probe.exe'},'profiler':{'path':'nsys'}}
def test_plan_has_exact_12_native_6_export_and_pair_bound_argv():
 plan=c.plan(protocol(),manifest());assert len(plan)==18 and sum(x['mode']!='export' for x in plan)==12 and sum(x['mode']=='export' for x in plan)==6
 for pair in range(1,4):
  rows=[x for x in plan if x['pair']==pair];assert len(rows)==6
  for arm in ('control','buffered'):
   subset=[x for x in rows if x['arm']==arm];assert {x['mode'] for x in subset}=={'direct','profile','export'}
   assert len({x['pair_id'] for x in subset})==1
   for x in subset:
    if x['mode']!='export':assert x['app_argv'][-1]==x['raw'] and x['pair_id'] in x['app_argv']
def test_wait_same_preserves_natural_exit_after_interrupt():
 class Proc:
  n=0
  def wait(self):
   self.n+=1
   if self.n==1:raise KeyboardInterrupt()
   return 7
 assert c.wait_same(Proc())==7
def test_environment_only_explicitly_sets_runtime_keys(monkeypatch):
 monkeypatch.setenv('PATH','base');monkeypatch.setenv('DROP','x')
 p={'runtime':{'environment':{'KEEP':'1'},'extra_clear_environment':['DROP']}}
 env=c.environment(p,{'native_bin':'native'})
 assert env['PATH'].startswith('native;E:\\cuda\\bin;base') and env['KEEP']=='1' and 'DROP' not in env
def test_first_pair_gate_is_currently_too_early_for_cross_arm_review():
 plan=c.plan(protocol(),manifest())
 first_three=plan[:3]
 assert {x['arm'] for x in first_three}=={'control'}
 # The controller pauses at completed_stages==3, but summarize_pairs needs
 # direct/profile data for both arms of pair 1. This is a real root fix item.
 with pytest.raises(AssertionError): assert {x['arm'] for x in first_three}=={'control','buffered'}
def test_run_source_has_finally_clock_reset_and_no_child_kill():
 text=(P/'collect.py').read_text()
 assert "if lock_attempted:result['reset']=smi(['-rgc'],'reset')" in text
 assert '.kill(' not in text and '.terminate(' not in text
 assert "require(receipt['child_exited'],'cannot release a live child')" in text
