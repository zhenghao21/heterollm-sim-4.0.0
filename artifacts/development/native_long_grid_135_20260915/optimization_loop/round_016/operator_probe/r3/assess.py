"""Compare paired probe modes without treating traced timings as native truth."""
from pathlib import Path
import json,math,statistics,argparse,hashlib
from full_raw_audit import audit
P=Path(__file__).resolve().parent
def percentile(v,p):
 v=sorted(v);i=(len(v)-1)*p;a=int(i);return v[a]+(v[min(a+1,len(v)-1)]-v[a])*(i-a)
def summarize(d):
 r=[x for x in d['runs'] if x['phase']=='formal'];w=[x['host_wall_ns'] for x in r]
 return {'formal':len(w),'wall_median_ns':statistics.median(w),'wall_p90_over_p10':percentile(w,.9)/percentile(w,.1),'event_median_ns':None if d['control_mode'] else statistics.median(x['event_envelope_ms']*1e6 for x in r)}
def assess(direct,control,profile,config):
 docs={'direct':direct,'control':control,'profile':profile};audits={k:audit(v,config) for k,v in docs.items() if v is not None};result={'raw_audits':audits,'calibration_eligible':False,'dispatch_verified':False}
 if any(not a['valid_raw'] for a in audits.values()):result.update(status='rejected_raw');return result
 if direct['control_mode'] or not control['control_mode'] or (profile is not None and profile['control_mode']):result.update(status='rejected_mode_pair');return result
 stats={k:summarize(v) for k,v in docs.items() if v is not None};result['summaries']=stats
 signatures=[(v['M'],v['N'],v['K'],v['weight_format'],v['quantization']['packed_weight_sha256'],v['quantization']['input_sha256'],v['environment'],v['loaded_modules_before']) for v in docs.values() if v is not None]
 if any(s!=signatures[0] for s in signatures[1:]):result.update(status='rejected_identity_pair');return result
 event_control=abs(stats['direct']['wall_median_ns']-stats['control']['wall_median_ns'])/stats['control']['wall_median_ns'];result['event_control_relative_wall_difference']=event_control
 profile_direct=None if profile is None else abs(stats['profile']['wall_median_ns']-stats['direct']['wall_median_ns'])/stats['direct']['wall_median_ns'];result['profile_direct_relative_wall_difference']=profile_direct
 gates=all(s['wall_p90_over_p10']<=1.5 for s in stats.values()) and event_control<=.2 and profile_direct is not None and profile_direct<=.2
 result.update(status='timing_quality_pass_trace_dispatch_pending' if gates else 'unresolved_measurement_quality',timing_quality_pass=gates)
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--direct',required=True);ap.add_argument('--control',required=True);ap.add_argument('--profile');ap.add_argument('--output',required=True);a=ap.parse_args()
 c=next(c for c in json.loads((P/'protocol.json').read_text())['configs'] if c['id']==a.config);load=lambda s:None if not s else json.loads(Path(s).read_text(encoding='utf-8-sig'))
 r=assess(load(a.direct),load(a.control),load(a.profile),c);r['inputs']={k:{'path':str(Path(v).resolve()),'sha256':hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k,v in [('direct',a.direct),('control',a.control),('profile',a.profile)] if v};out=Path(a.output)
 if out.exists():raise SystemExit('Refuse overwrite')
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'calibration_eligible':False}))
if __name__=='__main__':main()
