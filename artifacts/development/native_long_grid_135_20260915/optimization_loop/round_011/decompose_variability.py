from pathlib import Path
import json,statistics,math,hashlib
r=Path(__file__).resolve().parent
run=r.parent/'round_010/stream_event_probe_dual_r2/runs/20260915T231129794Z'
def corr(x,y):
 a=statistics.mean(x);b=statistics.mean(y);sx=sum((v-a)**2 for v in x);sy=sum((v-b)**2 for v in y)
 return sum((u-a)*(v-b) for u,v in zip(x,y))/math.sqrt(sx*sy) if sx*sy else None
def spread(x):
 s=sorted(x);return {'median_us':statistics.median(x)/1000,'min_us':min(x)/1000,'max_us':max(x)/1000,'max_deviation_pct':100*max(abs(v-statistics.median(x)) for v in x)/statistics.median(x)}
rows=[]
for p in sorted(run.glob('*.event.json')):
 d=json.loads(p.read_text());data=[v for v in d['runs'] if v['phase']=='formal'];f=d['qpc_frequency'];scale=1e9/f
 host=[x['host_wall_ns'] for x in data];submit=[(x['qpc_submit_end']-x['qpc_submit_start'])*scale for x in data];wait=[(x['qpc_wait_end']-x['qpc_wait_start'])*scale for x in data];event=[x['event_envelope_ms']*1e6 for x in data]
 rows.append({'config':p.stem,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'batches':len(data),'host':spread(host),'submit':spread(submit),'wait':spread(wait),'event':spread(event),'corr_host_submit':corr(host,submit),'corr_event_submit':corr(event,submit),'median_submit_to_host':statistics.median(submit)/statistics.median(host),'not_additive':'event overlaps submit/wait; do not sum these intervals'})
out=r/'variability_decomposition.json'
if out.exists():raise SystemExit('no overwrite')
out.write_text(json.dumps({'scope':'read-only all12 configurations all30 formal batches; no filtering','rows':rows,'causality':'Correlation only, cannot distinguish OS scheduling/driver/firmware from device stalls; no coefficients inferred.'},indent=2))
for x in rows:print(x['config'],round(x['median_submit_to_host'],3),round(x['corr_host_submit'],3),round(x['corr_event_submit'],3))
