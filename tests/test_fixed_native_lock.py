import hashlib
import json
from pathlib import Path
import pytest
from tools.verify_fixed_native import verify_lock


def fixture(tmp_path):
    raw = tmp_path / 'raw.json'
    raw.write_text('{"timing_ms":1.2}', encoding='utf-8')
    def ref(path):
        data=path.read_bytes()
        return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps({'selected_cells':[{'cell_id':'one','native_actuals':[{'raw_ref':ref(raw)}]}]}),encoding='utf-8')
    state={'schema':'fixed-native-optimization-loop/v1','native_selection_ref':ref(selection),'native_raw_refs':[ref(raw)],'native_selected_cells':1,'native_remeasurement_allowed_in_this_loop':False}
    path=tmp_path/'state.json';path.write_text(json.dumps(state),encoding='utf-8')
    return path,state,raw,selection


def test_fixed_truth_passes(tmp_path):
    state,_,_,_=fixture(tmp_path)
    assert verify_lock(state)['formal_requests']==1


@pytest.mark.parametrize('target',['raw','selection'])
def test_changed_truth_fails_closed(tmp_path,target):
    state,_,raw,selection=fixture(tmp_path)
    path=raw if target=='raw' else selection
    data=path.read_bytes();path.write_bytes(data[:-1]+b' ')
    with pytest.raises(ValueError,match='changed'):
        verify_lock(state)


@pytest.mark.parametrize('field,value',[('native_raw_refs',[]),('native_selection_ref',None),('native_selected_cells',None),('native_remeasurement_allowed_in_this_loop',None)])
def test_missing_identity_is_not_a_pass(tmp_path,field,value):
    path,state,_,_=fixture(tmp_path);state[field]=value;path.write_text(json.dumps(state),encoding='utf-8')
    with pytest.raises(ValueError):verify_lock(path)
