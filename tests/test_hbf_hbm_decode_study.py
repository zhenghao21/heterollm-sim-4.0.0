from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hbf_hbm_decode_study import decode_measurement


def serving(times, *, prefill_end=None):
    events=[];requests={}
    for request_id, timestamps in times.items():
        requests[request_id]=NS(visible_output_tokens=len(timestamps),tpot_ns=(timestamps[-1]-timestamps[0])/(len(timestamps)-1))
        for i,time in enumerate(timestamps):
            events.append(NS(event_type='tokens_committed',request_id=request_id,timestamp_ns=time,
                             details={'visible_tokens':1,**({'source':'prefill'} if i==0 else {})}))
    start=prefill_end if prefill_end is not None else max(values[0] for values in times.values())
    batches=[NS(start_ns=0.,end_ns=start,items=[NS(phase='prefill')]),
             NS(start_ns=start,end_ns=max(values[-1] for values in times.values()),items=[NS(phase='decode')])]
    return NS(events=events,request_metrics=requests,batches=batches)


def test_prefill_duration_does_not_dilute_decode_rate():
    result=decode_measurement(serving({'a':[1000000.,1000010.,1000030.,1000060.]}))
    assert all(result['checks'].values())
    assert result['decode_tokens']==3
    assert result['node_decode_tps']==pytest.approx(50e6)
    assert result['pure_decode_tbt_ns']['p50']==20.


def test_concurrent_requests_use_shared_window_not_sum_of_request_rates():
    result=decode_measurement(serving({'a':[100.,110.,120.],'b':[100.,120.,140.]}))
    assert result['decode_tokens']==4
    assert result['node_decode_tps']==pytest.approx(100e6)
    assert result['pure_decode_tbt_ns']['count']==4
    assert result['pure_decode_tbt_ns']['p50']==15.


def test_prefill_tail_excludes_cross_boundary_tbt():
    result=decode_measurement(serving({'a':[100.,120.,150.]},prefill_end=110.))
    assert result['node_decode_tps']==pytest.approx(50e6)
    assert result['pure_decode_tbt_ns']['count']==1
    assert result['excluded_boundary_tbt_count']==1
    assert result['per_request'][0]['tpot_ns']==25.


def test_decode_work_before_last_prefill_is_not_called_pure():
    result=decode_measurement(serving({'a':[100.,105.,150.]},prefill_end=110.))
    assert not result['checks']['all_post_first_tokens_counted']


def test_mixed_inventory_cannot_produce_a_decode_comparison():
    from hbf_hbm_decode_study import summarize_decode
    rows=[{'id':arch,'parameters':{'model_id':'fixture'},'prompt_tokens':2048,
           'placement':arch,'fixed_inventory':{'hbm':capacity},'visible_output_tokens':64,
           'completed_requests':1,'decode_measurement':{'checks':{'valid':True}},
           'decode_core_checks':{'valid':True},'placement_verification':{'valid':True}}
          for arch,capacity in (('A_hbm_only',1024),('B_all_kv_hbf',512))]
    result=summarize_decode(rows,models=('fixture',),contexts=(2048,))
    assert result['comparisons']==[]
    assert not result['checks']['paired_inventory_identical']
    assert result['status']=='failed'


def test_failed_point_does_not_discard_other_completed_points(monkeypatch,tmp_path):
    from concurrent.futures import Future
    import hbf_hbm_decode_study as study
    class ImmediatePool:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def submit(self,function,point):
            future=Future()
            if point[2]=='A_hbm_only':
                future.set_exception(ValueError('physical capacity exceeded'))
            else:
                future.set_result({'id':'completed','decode_measurement':{'node_decode_tps':1.},
                                   'tpot_ns':{'p50':1e9},'wall_seconds':1.})
            return future
    monkeypatch.setattr(study,'ProcessPoolExecutor',ImmediatePool)
    with pytest.raises(study.StudyCasesError) as failure:
        study.run_decode_study(models=('fixture',),contexts=(1,),output=2,point_dir=tmp_path)
    assert len(failure.value.completed_rows)==1
    assert (tmp_path/'fixture-1-B_all_kv_hbf-2.json').exists()


@pytest.mark.parametrize('stale_contract', [
    'physical_core_unified_hardware_catalog_v2',
    'physical_core_unified_hardware_catalog_v3/serving-points-v4',
    'physical_core_unified_hardware_catalog_v3/serving-points-v8',
])
def test_geometry_revision_invalidates_decode_resume(monkeypatch, tmp_path, stale_contract):
    import json
    from concurrent.futures import Future
    import hbf_hbm_decode_study as study
    submissions = []
    for architecture in study.ARCHITECTURES:
        point = ('fixture', 1, architecture, 2)
        saved = {'memory_execution_contract': stale_contract,
                 'core_execution_audit': {'executed_physical_configurations': {'hbm0': {}}},
                 'parameters': study.point_parameters(point)}
        (tmp_path / ('-'.join(map(str, point)) + '.json')).write_text(json.dumps(saved), encoding='utf-8')
    class ImmediatePool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, point):
            submissions.append(point)
            future = Future()
            future.set_result({'id': point[2], 'memory_execution_contract': study.EXECUTION_CONTRACT,
                               'decode_measurement': {'node_decode_tps': 1.},
                               'tpot_ns': {'p50': 1e9}, 'wall_seconds': 1.})
            return future
    monkeypatch.setattr(study, 'ProcessPoolExecutor', ImmediatePool)
    monkeypatch.setattr(study, 'core_checks', lambda row: {'valid': True})
    monkeypatch.setattr(study, 'summarize_decode', lambda rows, **kwargs: {'rows': rows})
    result = study.run_decode_study(models=('fixture',), contexts=(1,), output=2,
                                    point_dir=tmp_path, resume=True)
    assert len(submissions) == len(result['rows']) == 2
    assert all(row['memory_execution_contract'] == study.EXECUTION_CONTRACT for row in result['rows'])
