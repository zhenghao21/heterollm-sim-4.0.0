import json,sys
from pathlib import Path
from unittest.mock import patch
from heterollm_sim import planner
from heterollm_sim.serde import to_primitive
from tests.test_mmvq_mechanism import _qualified_mmvq_case
from tests.test_mmq_planner import scenario
rows=[]
for fmt in ('Q5_0','Q8_0'):
 for n in (1,2,4,8,64):
  def factory(**kwargs):return scenario(tokens=n,weight_format=fmt)
  with patch('tests.test_mmvq_mechanism.base_scenario',factory):case=_qualified_mmvq_case()
  result=planner.compile_scenario(case)
  tasks=[]
  for t in result.tasks:
   d=to_primitive(t);d.pop('metadata',None);tasks.append(d)
  rows.append({'format':fmt,'tokens':n,'tasks':tasks})
Path(sys.argv[1]).write_text(json.dumps({'planner':planner.__file__,'rows':rows},sort_keys=True),encoding='utf-8')
