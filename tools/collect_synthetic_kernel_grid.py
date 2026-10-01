"""Independent synthetic GGML/CUPTI collection. Never reads LLM measurements."""
import argparse, hashlib, json, os, sqlite3, statistics, subprocess
from pathlib import Path


def extract_case(raw_path, sqlite_path):
    raw = json.loads(Path(raw_path).read_text(encoding='utf-8'))
    if not (raw['status'] == 'measured' and raw['modules_stable'] and raw['first_call_correctness']['passed'] and raw['final_correctness']['passed']):
        raise ValueError('numerical or module identity gate failed')
    with sqlite3.connect(str(sqlite_path)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT k.*,s.value AS symbol FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
    groups = {}
    for row in rows:
        groups.setdefault(row['symbol'], []).append(row)
    kernels = []
    for symbol, events in groups.items():
        if len(events) != raw['graph_compute_calls']:
            raise ValueError('one-kernel-per-call extraction not valid for ' + symbol)
        fields = ('deviceId','streamId','registersPerThread','gridX','gridY','gridZ','blockX','blockY','blockZ','staticSharedMemory','dynamicSharedMemory')
        signatures = {tuple(e[f] for f in fields) for e in events}
        if len(signatures) != 1:
            raise ValueError('kernel geometry changed across repetitions')
        formal = events[1 + raw['warmup_requested']:]
        if len(formal) != raw['formal_repeats_requested']:
            raise ValueError('formal sample count mismatch')
        ns = [e['end'] - e['start'] for e in formal]
        kernels.append({'symbol':symbol,'geometry':dict(zip(fields,next(iter(signatures)))),
                        'samples_ns':ns,'median_ns':statistics.median(ns),'stddev_ns':statistics.stdev(ns)})
    if len(kernels) != 2 or not any('mul_mat_vec_q' in k['symbol'] for k in kernels):
        raise ValueError('expected independently observed conversion + MMVQ kernels')
    return {'shape':[raw['M'],raw['N'],raw['K']], 'format':raw['weight_format'],
            'cache_protocol':raw['cache_policy'],'kernels':kernels,
            'raw_sha256':hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
            'sqlite_sha256':hashlib.sha256(Path(sqlite_path).read_bytes()).hexdigest(),
            'module_identity':raw['loaded_modules_after'],'numerics_passed':True,
            'measurement_boundary':'cuda_device_kernel_interval',
            'transfer_to_llm_validated':False,'bandwidth_measured':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    exe=root/'artifacts/development/generic_gemm_microbench_v1/generic-gemm-microbench.exe'
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=protocol['executable_sha256']:raise ValueError('binary identity changed')
    nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.6.2/target-windows-x64/nsys.exe')
    env=dict(os.environ);env['PATH']=str(root/'source/llama.cpp-semantic/build-semantic-direct/bin')+';C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;'+env['PATH']
    rows=[]
    for fmt in protocol['formats']:
        for shape in protocol['training_shapes']+protocol['holdout_shapes']:
            ident=fmt+'_'+'_'.join(map(str,shape));raw=out/(ident+'.profiled.json');trace=out/(ident+'.nsys-rep');db=out/(ident+'.sqlite')
            cmd=[str(exe),'--device','cuda','--quant',fmt,'--m',str(shape[0]),'--n',str(shape[1]),'--k',str(shape[2]),'--threads','16','--warmup','3','--repeats','20','--samples','32','--seed','20260929','--atol','.05','--rtol','.03','--run','--output',str(raw)]
            if not trace.exists():
                with (out/(ident+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run([str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=false','--output='+str(out/ident),*cmd],env=env,stdout=log,stderr=log,timeout=120,check=True)
            if not db.exists():subprocess.run([str(nsys),'export','--type','sqlite','--output',str(db),str(trace)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60,check=True)
            row=extract_case(raw,db);row['split']='holdout' if shape in protocol['holdout_shapes'] else 'train';rows.append(row)
            (out/'kernel_measurements.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'rows':rows},indent=2),encoding='utf-8')
            print(ident,[(k['median_ns'],k['geometry']['registersPerThread']) for k in row['kernels']],flush=True)
if __name__=='__main__':main()
