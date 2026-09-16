"""R21 host-only math and contract checks; no compile or native invocation."""
from pathlib import Path
from fractions import Fraction
import argparse,ast,json,struct,hashlib,math
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args()
 protocol=json.loads((P/'protocol.json').read_text());source=(P/'host_settle_probe.cpp').read_text();entry=(P/'host_settle_main.cpp').read_text()
 assert protocol['execution']['pilot']['conditions']==[{'id':'short','settle_ms':0},{'id':'settled','settle_ms':1000}]
 assert protocol['execution']['pilot']['native_processes']==12 and protocol['execution']['pilot']['expected_exports']==6
 assert (protocol['execution']['first'],protocol['execution']['warmup'],protocol['execution']['formal'])==(1,5,30)
 assert protocol['quality']['formal_p90_p10_max']==1.5 and protocol['quality']['three_process_median_relative_spread_max']==.05 and protocol['quality']['direct_profile_relative_median_wall_limit']==.2
 assert protocol['runtime']['environment']['GGML_CUDA_DISABLE_GRAPHS'] is None
 checked=0
 f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
 for n in range(-127,128):
  v=f32(n/256)
  for stage in range(1,33):
   v=f32(v*.5);expected=f32(float(Fraction(n,1<<(8+stage))))
   assert math.isfinite(v) and struct.pack('<f',v)==struct.pack('<f',expected);checked+=1
 body=source.split('ArmResult run_settle_buffered_arm',1)[1]
 assert body.index('kFirstCalls,calls,completed')<body.index('Range settle')<body.index('kFirstCalls,kWarmupCalls')<body.index('kFirstCalls+kWarmupCalls,kFormalCalls')
 assert 'catch(...) {flush_records();throw;}' in body and 'buffered_started' in body and 'settle_metadata' in body
 assert 'estimator_time_used' in body and 'iterations' in body and 'deadline=begin+frequency*settle_ms/1000' in body
 assert source.count('ggml_backend_synchronize(harness.backend)')==2 # one recorded-call path + one work-settle path
 assert 'int settle_ms = -1;' in entry and 'if (options.run && (options.arm != "buffered"' in entry
 assert 'graph_gap_probe::run_settle_buffered_arm' in entry and '}); });' not in entry
 assert 'graph_gap_probe::run_control_arm' not in entry
 for f in P.glob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
 d={'schema':'host-settle-preflight/v1','pass':True,'numerator_stage_checks':checked,'GPU_access':False,'compiled':False,'protocol_sha256':hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),'checks':['retained complete runtime and quality contract','exact F32 reference8160','first before settle before warmup/formal','bounded work window and no estimated cost use','catch-safe complete recorded buffers','entry requires explicit supported settle condition']}
 if args.output:
  with Path(args.output).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(d))
if __name__=='__main__':main()
