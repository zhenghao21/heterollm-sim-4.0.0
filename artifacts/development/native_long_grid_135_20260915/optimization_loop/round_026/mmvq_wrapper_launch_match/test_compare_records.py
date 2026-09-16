from pathlib import Path
from copy import deepcopy
import importlib.util,json
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('launch_match_compare',P/'compare_records.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
TARGET=json.loads((P.parent/'target_capture_ex_run.0001/launches.json').read_text(encoding='utf8'))
def pair():
    a=deepcopy(TARGET);b=deepcopy(a);b['origin']='standalone_source_wrapper';b['gpu_graph_executed']=False;b['gpu_launch_pair_executed']=True;b['new_shim_numerical_qualified']=True;b['launch_pair_qualified']=True
    b['numerical_check']={'same_new_shim_used':True,'separate_post_capture_pass':True,'reference_fixed_before_GPU':True,'capture_includes_numerical_pass':False,'q8_bytes_compared':4608,'q8_byte_errors':0,'outputs_compared':3072,'output_failures':0}
    for k in b['graph_tensors']:b['graph_tensors'][k]+=4096
    for i,e in enumerate(b['launches']):
        for k in d.POINTER_FIELDS[i]:e['arguments'][k]+=4096
        e['function']+=8192;e['stream']+=16384;e['context']+=7;e['correlation']+=13
        for attr in e['attributes']:attr['opaque_value_bytes_hex']='01000000'+'cc'*60
    return a,b

def test_cross_process_ids_and_inactive_union_bytes_only_are_normalized():
    a,b=pair();assert d.compare(a,b)['status']=='effective_launch_pair_matches'

@pytest.mark.parametrize('field',['stride_channel_x','stride_channel_y','stride_channel_dst','stride_sample_x','stride_sample_y','stride_sample_dst'])
def test_every_degenerate_dimension_stride_is_still_compared(field):
    a,b=pair();b['launches'][1]['arguments'][field]=1
    with pytest.raises(ValueError,match='complete kernel scalar/layout'):d.compare(a,b)

@pytest.mark.parametrize('change',[lambda b:b['launches'][0]['attributes'][0].update(programmaticStreamSerializationAllowed=0),
    lambda b:b['launches'][0].update(geometry_observed=False),lambda b:b['launches'][1]['arguments'].update(vy=88),
    lambda b:b['launches'][1].update(stream=99),lambda b:b.update(memory_api_count=1),
    lambda b:b['launches'][1].update(return_code=3),lambda b:b['launches'][0]['attributes'][0].update(id=7),
    lambda b:b['launches'][1].update(api_id=211,api_name='cudaLaunchKernel')])
def test_PDL_observation_relationship_and_error_changes_are_rejected(change):
    a,b=pair();change(b)
    with pytest.raises(ValueError):d.compare(a,b)

def test_unknown_scalar_is_not_dropped():
    a,b=pair();b['launches'][1]['arguments']['hidden_scalar']=1
    with pytest.raises(ValueError):d.compare(a,b)

def test_bool_cannot_replace_integer_stride():
    a,b=pair();b['launches'][0]['arguments']['ne1']=True
    with pytest.raises(ValueError):d.compare(a,b)


def test_new_strides_cannot_inherit_old_numerical_pass():
    a,b=pair();b['new_shim_numerical_qualified']=False
    with pytest.raises(ValueError,match='new shim numerical'):d.compare(a,b)
