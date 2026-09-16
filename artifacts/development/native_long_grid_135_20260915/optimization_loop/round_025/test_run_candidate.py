import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('r25_runner_test',Path(__file__).with_name('run_candidate.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def test_score_cannot_invoke_scorer_before_barrier(monkeypatch):
    calls=[]
    def reject():raise ValueError('missing barrier')
    monkeypatch.setattr(d,'barrier',reject);monkeypatch.setattr(d,'command',lambda *a:calls.append(a))
    with pytest.raises(ValueError,match='missing barrier'):d.score()
    assert calls==[]
def test_scores_block_prediction_even_when_guard_passes(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'P',tmp_path);(tmp_path/'off').mkdir();(tmp_path/'off/errors.0001.json').write_text('{}')
    monkeypatch.setattr(d,'guard',lambda:'valid');calls=[];monkeypatch.setattr(d,'command',lambda *a:calls.append(a))
    with pytest.raises(ValueError,match='scores exist'):d.full(4,600)
    assert calls==[]
def test_partial_terminal_set_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'P',tmp_path);monkeypatch.setattr(d.s,'read_json',lambda p:({},{}));monkeypatch.setattr(d.s,'unique_cells',lambda f:{'a':{},'b':{}})
    with pytest.raises(ValueError,match='all131'):d.terminal('on')
