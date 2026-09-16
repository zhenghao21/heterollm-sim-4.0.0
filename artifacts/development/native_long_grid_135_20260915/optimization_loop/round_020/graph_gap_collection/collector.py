"""R20 collection preparation: emits commands/freeze only; root owns all execution."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common import HERE,PROBE,load,ref,plan,validate_probe

def commands(protocol):
 out=[]; profile_raw={}
 for ordinal,s in enumerate(plan(protocol)):
  base=HERE/'runs'/f"{ordinal:02d}_{s['arm']}_p{s['pair']}_{s['mode']}"
  key=(s['pair'],s['arm'])
  if s['kind']=='native':
   raw=str(base.with_suffix('.jsonl'))
   if s['mode']=='profile': profile_raw[key]=raw
   out.append({**s,'ordinal':ordinal,'cwd':str(PROBE),'command':['powershell','-File',str(PROBE/'invoke.ps1'),'-Run','-Arm',s['arm'],'-Mode',s['mode'],'-Pair',str(s['pair']),'-Output',raw],'raw':raw,'natural_completion_required':True,'timeout_action':'retain_and_wait_no_kill'})
  else:
   if key not in profile_raw: raise ValueError('export missing its bound profile process')
   profile_path=Path(profile_raw[key]); rep=str(profile_path.with_suffix(''))+'.profile.nsys-rep'
   out.append({**s,'ordinal':ordinal,'profile_raw':profile_raw[key],'command':['nsys','export','--type','sqlite','--output',str(base.with_suffix('.sqlite')),rep],'trace_export':str(base.with_suffix('.sqlite'))})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','prepare']);ap.add_argument('--root-reviewed',action='store_true');a=ap.parse_args(); protocol=load(PROBE/'protocol.json'); validate_probe(protocol); payload={'schema':'graph-gap-r20-commands/v1','probe_protocol':ref(PROBE/'protocol.json'),'stages':commands(protocol),'sequential_launch':True,'collector_sets_clocks':False,'clock_reset_finally':'root-owned reset action runs after natural completion of started processes; no timed kill','calibration_ready':False}
 if a.action=='prepare':
  if not a.root_reviewed: raise SystemExit('root review required')
  manifest=PROBE/'build_manifest.json'
  if not manifest.exists(): raise SystemExit('compiled R20 build_manifest.json required; no launch attempted')
  payload['build_manifest']=ref(manifest)
 print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()