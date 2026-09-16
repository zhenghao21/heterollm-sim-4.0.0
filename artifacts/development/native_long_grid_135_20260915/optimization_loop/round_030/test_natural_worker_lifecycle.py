"""Synthetic lifecycle tests. No model loading, inference or GPU activity."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from tools import predict_stable_native_dataset as api
from tests.test_predict_stable_native_dataset import fixture, document, seal

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("r30_natural_freezer",HERE/"freeze_candidate.py")
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
spec=importlib.util.spec_from_file_location("r30_natural_runner",HERE/"run_candidate.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def prepared(tmp_path,monkeypatch,count=1):
    path,selection,row,calls=fixture(tmp_path,monkeypatch)
    if count>1:
        rows=[]
        for i in range(count):
            item=copy.deepcopy(row);item["cell_id"]+="_%d"%i;rows.append(item)
        selection.update(selected_cells=rows,selected_count=count,selected_cell_ids=[x["cell_id"] for x in rows])
        for group in selection["coverage"]:
            group["selected_cells"]=count if group["model_key"]=="qwen38_gpu" else 0
            group["excluded_cells"]=27-group["selected_cells"]
        document(path,seal(selection))
    out=tmp_path/"on";api.freeze_selection(path,out,data_root=tmp_path)
    return out,calls


def fake_process(monkeypatch,mode):
    launched=[]
    class Child:
        def __init__(self,command,**kwargs):
            self.command=command;self.returncode=None;self.waits=0;self.pid=90000+len(launched)
            launched.append(self)
        def wait(self,timeout=None):
            self.waits+=1
            if mode=="late" and self.waits==1:raise subprocess.TimeoutExpired(self.command,timeout)
            if mode=="interrupt" and self.waits==1:raise KeyboardInterrupt("synthetic observer interruption")
            if mode=="unresolved":raise OSError("synthetic lost observation handle")
            if mode!="missing":
                args=self.command;raw=Path(args[args.index("--worker-result")+1])
                api.worker_cell(Path(args[args.index("--worker-freeze")+1]),args[args.index("--worker-cell")+1],raw)
                if mode=="bad_identity":
                    value=json.loads(raw.read_text());value["source_sha256"]="changed";raw.write_text(json.dumps(value))
            self.returncode=7 if mode=="nonzero" else 0
            return self.returncode
        def poll(self):return self.returncode
        def kill(self):pytest.fail("kill is forbidden")
        def terminate(self):pytest.fail("terminate is forbidden")
    monkeypatch.setattr(subprocess,"Popen",Child)
    return launched


def result(out):
    path=next((out/"predictions").glob("*.prediction.json"));doc,_=api.grid.read_document(path)
    execution=api.verify_worker_execution(doc,path)
    return doc,execution,path


def test_late_exit0_is_scoreable_and_keeps_raw_and_deadline(tmp_path,monkeypatch):
    out,_=prepared(tmp_path,monkeypatch);children=fake_process(monkeypatch,"late")
    manifest=api.run_predictions(out,timeout_seconds=600)
    doc,e,path=result(out)
    assert manifest["successful_cells"]==1 and doc["status"]=="predicted"
    assert e["late"] and e["soft_deadline_ref"] and e["natural_exit_observed"] and e["returncode"]==0
    assert children[0].waits==2 and not e["hard_time_limit_enforced"]
    before=api.grid.file_ref(path);api.run_predictions(out,resume=True)
    assert len(children)==1 and api.grid.file_ref(path)==before
    report=api.score_predictions(out)
    assert report["cells"][0]["execution_observation"]["late"] and report["overall"]["engine_ttft_ms"]["missing_cells"]==0
    monkeypatch.setattr(r,"P",tmp_path)
    assert r.verify_terminal_execution("on",doc)["late"]
    r.check_attempts_closed("on")


@pytest.mark.parametrize("mode,reason",[("nonzero","nonzero"),("missing","without a result"),("bad_identity","identity/status")])
def test_failed_execution_never_promotes_raw_predicted_status(tmp_path,monkeypatch,mode,reason):
    out,_=prepared(tmp_path,monkeypatch);children=fake_process(monkeypatch,mode)
    manifest=api.run_predictions(out)
    doc,e,path=result(out)
    assert manifest["failed_or_incomplete_cells"]==1 and doc["status"]=="failed" and reason in doc["reason"]
    if mode=="nonzero":
        raw,_=api.grid.read_document(e["raw_result_ref"]["path"])
        assert raw["status"]=="predicted" and e["returncode"]==7 and not e["result_identity_valid"]
    before=api.grid.file_ref(path);api.run_predictions(out,resume=True)
    assert len(children)==1 and before==api.grid.file_ref(path)


def test_observer_interrupt_stops_new_launches_but_drains_started_worker(tmp_path,monkeypatch):
    out,_=prepared(tmp_path,monkeypatch,3);children=fake_process(monkeypatch,"interrupt")
    manifest=api.run_predictions(out,workers=1)
    doc,e,_=result(out)
    assert len(children)==1 and children[0].waits==2 and e["observation_interruptions"]==1
    assert manifest["successful_cells"]==1 and manifest["pending_cells"]==2
    assert not(out/"runs/coordinator.lock").exists()
    finish,_=api.grid.read_document(out/"runs/run.0001.finish.json")
    assert finish["launches_stopped_after_observation_interrupt"] and finish["all_started_workers_exited_and_sealed"]


def test_unresolved_attempt_keeps_lock_and_blocks_duplicate_even_without_lock(tmp_path,monkeypatch):
    out,_=prepared(tmp_path,monkeypatch);children=fake_process(monkeypatch,"unresolved")
    with pytest.raises(OSError,match="lost observation"):api.run_predictions(out)
    assert(out/"runs/coordinator.lock").exists() and not list((out/"predictions").glob("*.prediction.json"))
    with pytest.raises(ValueError,match="unresolved"):api.run_predictions(out,resume=True)
    (out/"runs/coordinator.lock").unlink() # Simulate an erroneous external deletion in this temp fixture only.
    with pytest.raises(ValueError,match="unresolved"):api.run_predictions(out,resume=True)
    assert len(children)==1 and children[0].poll() is None
    monkeypatch.setattr(r,"P",tmp_path)
    with pytest.raises(ValueError,match="unresolved"):r.check_attempts_closed("on")


def test_terminal_tampering_or_missing_seal_cannot_be_resumed_or_scored(tmp_path,monkeypatch):
    out,_=prepared(tmp_path,monkeypatch);children=fake_process(monkeypatch,"success")
    api.run_predictions(out);doc,e,path=result(out)
    seal_path=Path(doc["worker_execution_ref"]["path"]).with_name("sealed.json")
    seal_path.unlink()
    with pytest.raises(ValueError,match="unresolved"):api.run_predictions(out,resume=True)
    with pytest.raises(ValueError,match="unresolved"):api.score_predictions(out)
    assert len(children)==1


def test_helper_interrupt_waits_naturally_and_stops_parent_phase(tmp_path,monkeypatch):
    monkeypatch.setattr(f,"P",tmp_path)
    class Child:
        pid=321;returncode=None
        def __init__(self,*a,**k):self.calls=0
        def communicate(self):
            self.calls+=1
            if self.calls==1:raise KeyboardInterrupt()
            self.returncode=0;return b"complete",b""
        def poll(self):return self.returncode
        def kill(self):pytest.fail("kill forbidden")
        def terminate(self):pytest.fail("terminate forbidden")
    monkeypatch.setattr(f.subprocess,"Popen",Child)
    with pytest.raises(InterruptedError,match="exited naturally"):f.natural_process(["never-run"],capture_output=True,purpose="preflight")
    receipt=json.loads(next((tmp_path/"process_observations").glob("*/finish.json")).read_text())
    assert receipt["natural_exit_observed"] and receipt["returncode"]==0 and receipt["observation_interruptions"]==1


def test_production_lifecycle_no_implicit_killing_helpers():
    for path in (HERE/"freeze_candidate.py",HERE/"run_candidate.py",Path(api.__file__)):
        tree=ast.parse(path.read_text(encoding="utf8"))
        for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)):
            assert call.func.attr not in ("kill","terminate"),str(path)
            assert not(isinstance(call.func.value,ast.Name) and call.func.value.id=="subprocess" and call.func.attr in ("run","check_output","call")),str(path)


def test_real_python_synthetic_worker_exits_naturally_without_model_loading(tmp_path,monkeypatch):
    selection,_,_,_=fixture(tmp_path,monkeypatch)
    script = r"""
import argparse,datetime,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--worker-freeze');p.add_argument('--worker-cell');p.add_argument('--worker-result');a=p.parse_args()
path=Path(a.worker_freeze).resolve();raw=path.read_bytes();f=json.loads(raw);entry=next(e for e in f['cells'] if e['cell_id']==a.worker_cell)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',': '),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
d={'schema':'stable-native-cell-prediction/v1','cell_id':a.worker_cell,'model_key':entry['model_key'],'deployment':entry['deployment'],
'created_utc':now,'finished_utc':now,'freeze_ref':{'path':str(path),'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
'source_sha256':f['source']['sha256'],'selection_sha256':f['selection_sha256'],'status':'predicted',
'native_answers_used':False,'calibration_applied':False,'formal_prediction_eligible':False,
'input_identity':{'static_inputs_sha256':digest(entry['static_inputs'])},
'aggregate':{k:{'median_ms':1.0} for k in ('engine_ttft_ms','engine_tpot_ms','engine_e2e_ms')}}
with Path(a.worker_result).open('x',encoding='utf8') as o:json.dump(d,o)
"""
    def source(destination):
        path=destination/'tools/predict_stable_native_dataset.py';path.parent.mkdir(parents=True);path.write_text(script,encoding='utf8')
        refs=[api.grid.file_ref(path)]
        return {'root':str(destination),'sha256':api.grid.stable_hash(refs),'files':refs}
    monkeypatch.setattr(api,'source_freeze',source)
    out=tmp_path/'on';api.freeze_selection(selection,out,data_root=tmp_path)
    manifest=api.run_predictions(out,workers=1,timeout_seconds=600)
    doc,e,_=result(out)
    assert manifest['successful_cells']==1 and doc['status']=='predicted' and e['returncode']==0
    assert e['natural_exit_observed'] and not(out/'runs/coordinator.lock').exists()
