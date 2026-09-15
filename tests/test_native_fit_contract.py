from pathlib import Path
import pytest
from tools.native_repeatability_experiment import command_for, expand_protocol


def config(tmp_path, fit=None):
    model=tmp_path/'m.gguf';model.write_bytes(b'fixture')
    p={'schema':'native-repeatability-protocol/v1','jobs':[{'id':'cell','model':str(model),'prompt':'one','output':32,'parallel':4,'gpu_layers':66}]}
    if fit is not None:p['jobs'][0]['fit_params']=fit
    return p


def test_fixed_full_gpu_fit_disabled_is_explicit(tmp_path,monkeypatch):
    import tools.native_repeatability_experiment as m
    monkeypatch.setattr(m,'DATA_ROOT',tmp_path)
    c=expand_protocol(config(tmp_path,False))[0];argv=command_for(Path('llama-server.exe'),c,1234)
    assert argv[argv.index('-ngl')+1]=='66'
    assert argv[argv.index('--fit')+1]=='off'


def test_legacy_omission_remains_omitted(tmp_path,monkeypatch):
    import tools.native_repeatability_experiment as m
    monkeypatch.setattr(m,'DATA_ROOT',tmp_path)
    assert '--fit' not in command_for(Path('llama-server.exe'),expand_protocol(config(tmp_path))[0],1234)


def test_fit_string_never_silently_enabled(tmp_path,monkeypatch):
    import tools.native_repeatability_experiment as m
    monkeypatch.setattr(m,'DATA_ROOT',tmp_path)
    with pytest.raises(ValueError,match='fit_params'):expand_protocol(config(tmp_path,'off'))
