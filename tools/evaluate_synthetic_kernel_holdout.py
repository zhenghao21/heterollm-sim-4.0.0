"""Evaluate untouched fixed-M holdouts without fitting to their answers."""
import argparse,json,math,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    data=json.loads((a.directory/'kernel_measurements.json').read_text(encoding='utf-8'));results=[]
    protocol=json.loads((a.directory/'protocol.json').read_text(encoding='utf-8-sig'));linear=protocol.get('interpolation')=='linear_shape'
    for fmt in sorted({r['format'] for r in data['rows']}):
        train=[r for r in data['rows'] if r['format']==fmt and r['split']=='train']
        held=[r for r in data['rows'] if r['format']==fmt and r['split']=='holdout']
        for test in held:
            m,n,k=test['shape'];lookup={tuple(r['shape']):r for r in train}
            na=sorted({r['shape'][1] for r in train});ka=sorted({r['shape'][2] for r in train})
            corners=[(m,n0,k0) for n0 in (na[0],na[-1]) for k0 in (ka[0],ka[-1])]
            if any(c not in lookup for c in corners):raise ValueError('incomplete joint cell')
            fn=(n-na[0])/(na[-1]-na[0]) if linear else math.log(n/na[0])/math.log(na[-1]/na[0]);fk=(k-ka[0])/(ka[-1]-ka[0]) if linear else math.log(k/ka[0])/math.log(ka[-1]/ka[0])
            for role in ('quantize_q8_1','mul_mat_vec_q'):
                select=lambda r:next(x for x in r['kernels'] if role in x['symbol'])
                points=[select(lookup[c]) for c in corners];truth=select(test)
                pred=sum(w*x['median_ns'] for w,x in zip(((1-fn)*(1-fk),(1-fn)*fk,fn*(1-fk),fn*fk),points))
                signatures={(x['symbol'],x['geometry']['blockX'],x['geometry']['blockY'],x['geometry']['registersPerThread'],x['geometry']['staticSharedMemory'],x['geometry']['dynamicSharedMemory']) for x in points+[truth]}
                ape=abs(pred-truth['median_ns'])/truth['median_ns']*100
                results.append({'format':fmt,'role':role,'holdout_shape':test['shape'],'prediction_ns':pred,'observed_ns':truth['median_ns'],'ape_pct':ape,'same_specialization':len(signatures)==1,'eligible':len(signatures)==1 and ape<10,'prediction_kind':'diagnostic_'+('linear' if linear else 'log')+'_shape_interpolation_not_installed'})
    result={'rows':results,'target_ape_pct':10,'all_eligible':all(r['eligible'] for r in results),'llm_accuracy_validated':False,'cache':'hot_same_buffers','profile_installed':False}
    (a.directory/'holdout_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
