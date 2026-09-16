import copy
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('r25_freeze_test',Path(__file__).with_name('freeze_candidate.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def pair(monkeypatch):
    monkeypatch.setattr(d.s,'unique_cells',lambda x:x['cells'])
    monkeypatch.setattr(d.s,'source_content',lambda x:x['source'])
    inputs={'shape':64,'retained_kv_warmup_evidence':{'extractor_ref':{'path':'off','sha256':'a','bytes':2}}}
    off={'source':{'same':'sha'},'cells':{'cell':{'static_inputs':inputs}}}
    on=copy.deepcopy(off);b=on['cells']['cell']['static_inputs'];b.update(final_output_selection=True,final_output_selection_binding={'status':'conditional'})
    b['retained_kv_warmup_evidence']['extractor_ref']['path']='on'
    return off,on

def test_only_selection_and_extractor_copy_change(monkeypatch):
    a,b=pair(monkeypatch);assert d.compare_inputs(a,b)==1
@pytest.mark.parametrize('change',['shape','digest','source','unknown'])
def test_other_changes_rejected(monkeypatch,change):
    a,b=pair(monkeypatch);i=b['cells']['cell']['static_inputs']
    if change=='shape':i['shape']=1
    if change=='digest':i['retained_kv_warmup_evidence']['extractor_ref']['sha256']='b'
    if change=='source':b['source']['same']='changed'
    if change=='unknown':i['unknown']=True
    with pytest.raises(ValueError):d.compare_inputs(a,b)
