"""Build bounded hot-cache diagnostic surfaces from passing independent holdouts.

Not an LLM profile installer: conversion and cache protocol retain separate owners.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heterollm_sim.kernel_model import KernelCapability,KernelSample
from heterollm_sim.serde import to_primitive

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    data=json.loads((a.directory/'kernel_measurements.json').read_text());gate=json.loads((a.directory/'holdout_validation.json').read_text());result=[]
    for item in gate['rows']:
        if not item['eligible'] or item['role']!='mul_mat_vec_q':continue
        rows=[r for r in data['rows'] if r['format']==item['format'] and r['split']=='train'];samples=[]
        for row in rows:
            k=next(k for k in row['kernels'] if item['role'] in k['symbol']);g=k['geometry']
            signature=json.dumps([k['symbol'],*[g[x] for x in ('blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory')]])
            samples.append(KernelSample(*row['shape'],k['median_ns'],k['median_ns'],k['stddev_ns'],len(k['samples_ns']),'sqlite-sha256:'+row['sqlite_sha256'],dispatch_signature=signature))
        if len({s.dispatch_signature for s in samples})!=1:raise ValueError('specialization mismatch')
        desc=KernelCapability('observed_mmvq_'+item['format'].lower(),(item['format'].lower(),),'fp32','decode','dp4a','independent CUPTI trace, main only',output_bits=32,
            min_shape=(1,2048,2048),max_shape=(1,4096,4096),samples=tuple(samples),surface_model='measured_surrogate',interpolation_space='linear',cache_protocol='same_buffers_repeated_hot_cache_no_flush',
            registers_per_thread=g['registersPerThread'],shared_memory_per_cta=g['staticSharedMemory']+g['dynamicSharedMemory'],warps_per_cta=g['blockX']*g['blockY']*g['blockZ']//32)
        result.append({'descriptor':to_primitive(desc),'holdout':item,'runtime_modules':rows[0]['module_identity']})
    payload={'schema':'diagnostic-verified-local-surfaces/v1','surfaces':result,'llm_installed':False,'missing_transfer_proofs':['cache_protocol','clock_domain','runtime_binary_equivalence','separate_conversion_owner'],'measurement_sha256':hashlib.sha256((a.directory/'kernel_measurements.json').read_bytes()).hexdigest()}
    (a.directory/'local_surfaces.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print('bounded surfaces',len(result))
if __name__=='__main__':main()
