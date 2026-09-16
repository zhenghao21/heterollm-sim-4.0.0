"""Read static kernel keys from a completed R27 prediction set; never read errors/native times."""
import argparse, collections, hashlib, json
from pathlib import Path

def digest(path):
    payload=path.read_bytes()
    return payload,hashlib.sha256(payload).hexdigest()

def verify_report(report_path):
    report=json.loads(report_path.read_text(encoding='utf-8'))
    assert report['schema']=='r28-static-mmvq-shape-coverage/v1'
    expected={r['cell_id']:r for r in report['cells']}
    assert len(expected)==131 and len(report['prediction_refs'])==131
    found={};shapes={}
    for ref in report['prediction_refs']:
        payload,sha=digest(Path(ref['path']))
        assert len(payload)==ref['bytes'] and sha==ref['sha256'], ref['path']
        prediction=json.loads(payload)
        cell=prediction['cell_id']
        assert cell not in found
        ledger=prediction.get('dispatch_summary',{}).get('gpu_invocations',{}).get('kernel_query_ledger')
        fixed=eligible=0
        for signature in (ledger or {}).get('signatures',[]):
            key=signature['key']
            if key['predicted_family']!='MMVQ':continue
            desc={k:key.get(k) for k in ('m','n','k_logical','k_executed','weight_formats')}
            sid=json.dumps(desc,sort_keys=True)
            group=shapes.setdefault(sid,dict(**desc,task_count=0,cells=set(),models=set(),cache_states=set(),native_dispatch_proven_all=True,layout_proven_all=True))
            group['task_count']+=signature['task_count'];group['cells'].add(cell)
            group['models'].add(prediction['model_key']+' / '+prediction['deployment'])
            group['cache_states'].add(key['cache_state'])
            group['native_dispatch_proven_all'] &= key.get('native_dispatch_proven') is True
            group['layout_proven_all'] &= key.get('layout_proven') is True
            if (key['m'],key['n'],key['k_logical'],key['weight_formats'])==(1,3072,4096,['Q5_0']):
                fixed+=signature['task_count']
                if key.get('calibration_eligible'):eligible+=signature['task_count']
        found[cell]=dict(cell_id=cell,status=prediction['status'],ledger_present=ledger is not None,
            ledger_complete=(ledger or {}).get('complete',False),unrepresented_tasks=(ledger or {}).get('unrepresented_tasks'),
            fixed_shape_tasks=fixed,eligible_fixed_shape_tasks=eligible)
    assert found==expected,'cell projection differs'
    def normalize(rows):
        result=[]
        for row in rows:
            value=dict(row)
            for k in ('cells','models','cache_states'):value[k]=sorted(value[k])
            result.append(value)
        return sorted(result,key=lambda x:json.dumps({k:x[k] for k in ('m','n','k_logical','k_executed','weight_formats')},sort_keys=True))
    assert normalize(shapes.values())==normalize(report['shapes']),'static signature projection differs'
    assert sum(v['fixed_shape_tasks'] for v in found.values())==report['fixed_shape_task_count']
    assert sum(v['eligible_fixed_shape_tasks'] for v in found.values())==report['eligible_fixed_shape_task_count']
    print(json.dumps(dict(status='verified',cells=len(found),signatures=len(shapes),fixed_shape_tasks=report['fixed_shape_task_count'],
        target_native_latency_read=False,errors_read=False,absence_scope='represented static signatures only; incomplete ledger is unknown')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=Path(__file__).with_name('static_kernel_coverage.0001.json'))
    a=p.parse_args();verify_report(a.report)
