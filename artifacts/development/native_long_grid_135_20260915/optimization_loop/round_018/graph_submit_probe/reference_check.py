"""Host-only mathematical/protocol preparation check. Does not load native libraries."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import argparse, ast, hashlib, json, math, struct
P=Path(__file__).resolve().parent
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.pack('<f',x)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');a=ap.parse_args()
    protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'));configs=protocol['configs']
    assert len(configs)==6 and len({x['id'] for x in configs})==6
    assert {(x['elements'],x['nodes']) for x in configs}=={(e,n) for e in [1024,262144] for n in [1,8,32]}
    checked=0
    # Every possible input numerator is checked independently using rational arithmetic.
    # The generator cycles through all 255 values because gcd(73,255)==1.
    assert math.gcd(73,255)==1
    assert {((i*73+19)%255)-127 for i in range(255)}==set(range(-127,128))
    for numerator in range(-127,128):
        iterative=f32(numerator/256)
        for stage in range(1,33):
            iterative=f32(iterative*f32(0.5))
            exact=Fraction(numerator,1<<(8+stage));closed=f32(float(exact))
            assert math.isfinite(iterative) and bits(iterative)==bits(closed)
            if numerator:assert abs(closed)>=2**-40 and abs(closed)>=2**-126
            checked+=1
    e=protocol['execution'];assert e['first']+e['warmup']+e['formal']==36 and e['pairs']==3
    assert e['final_backend_synchronize_calls_per_graph']==1 and e['probe_cuda_events'] is False
    assert protocol['runtime']['environment']['GGML_CUDA_DISABLE_GRAPHS'] is None
    assert '--cuda-graph-trace=node' in protocol['profiler']['options']
    sources={}
    for f in P.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'));sources[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    result={'schema':'graph-submit-preparation-check/v1','pass':True,'gpu_access':False,'native_libraries_loaded':False,'numerator_stage_checks':checked,'configs':6,'protocol_sha256':hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),'cpp_sha256':hashlib.sha256((P/'graph_submit_probe.cpp').read_bytes()).hexdigest(),'checks':['six Cartesian configurations','all possible exact F32 numerator/stage references','no overflow/subnormal at 32 layers','native graph mode retained','explicit node-granularity trace required','Python source AST'],'python_sources_sha256':sources,'not_checked':['C++ compilation','native dynamic linking','GPU numerical results','actual kernel counts','PowerShell parser (separate host check)','hardware state','timing accuracy']}
    if a.output:
        with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
