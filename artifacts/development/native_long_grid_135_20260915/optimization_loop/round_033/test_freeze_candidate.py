"""R33 synthetic closure checks. No actual freeze, model, native or prediction."""
from pathlib import Path
import copy, hashlib, importlib.util, json, subprocess, sys
from types import SimpleNamespace
import pytest
spec=importlib.util.spec_from_file_location("r33_freeze_tests",Path(__file__).with_name("freeze_candidate.py"))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return d.s.reference(path)


def pair(tmp_path, monkeypatch):
    monkeypatch.setattr(d.s, "unique_cells", lambda value: value["cells"])
    monkeypatch.setattr(d.s, "source_content", lambda value: value["source_bytes"])
    monkeypatch.setattr(d,"SOURCE_COUNT",2)
    docs = []
    config = {key: 1 for key in d.CONFIG_KEYS}
    config.update(flash_attn=False, op_offload=True)
    for arm in ("off", "on"):
        source = tmp_path / arm / "source"
        extractor = write(source / "tools/retained_warmup_extractor.py", b"reviewed exact helper")
        pdf = write(tmp_path / arm / "evidence/hardware.pdf", b"reviewed exact PDF")
        retained = {"extractor_ref": extractor, "scope": "fixed"}
        contract = {"rows": 543, "evidence_sha256": d.s.stable_hash(retained)}
        retained["contract"] = contract
        inputs = {"config": config, "retained_kv_warmup_evidence": retained, "retained_kv_warmup_contract": copy.deepcopy(contract),
                  "mmvq_issue_evidence": {"hardware_document": {"ref": pdf}, "evidence_refs": [copy.deepcopy(pdf)]}}
        inputs.update(final_output_selection=True, final_output_selection_binding={"status": "conditional", "config": config.copy()})
        if arm == "on":inputs["mmvq_hbm_mode"]=d.MODES[arm]
        docs.append({"source": {"root": str(source)}, "source_bytes": {"same": "sha", "src/heterollm_sim/planner.py": arm},
            "final_output_selection":True, **({"mmvq_hbm_mode":d.MODES[arm]} if arm=="on" else {}),
            "mmvq_issue_bound": {"hardware_document": {"ref": pdf}},
            "cells": {str(i): {"static_inputs": copy.deepcopy(inputs), "preparation_error": None} for i in range(131)}})
    return docs


def fake_arm(tmp_path, stale):
    directory = tmp_path / "on"
    source = directory / "source"
    api = '''
import json
from pathlib import Path
from tools.native_final_output_binding import CONFIG_KEYS
class Grid:
    pass
grid=Grid()
def configuration(inputs):return None
def gpu_clock(inputs):return None
def verify_freeze_references(frozen):
    Path(frozen['call_marker']).write_text('actual frozen verifier was called')
    for c in frozen['cells']:
        i=c['static_inputs']
        if i['final_output_selection_binding']['config']!={k:i['config'].get(k) for k in CONFIG_KEYS}:
            raise ValueError('cell proof differs from frozen static inputs')
'''
    helper = '''
CONFIG_KEYS=('flash_attn','op_offload')
def verify_cell(inputs,verify_files=True):return inputs['final_output_selection_binding']
'''
    write(source / "tools/__init__.py", b"")
    write(source / "tools/predict_stable_native_dataset.py", api.encode())
    write(source / "tools/native_final_output_binding.py", helper.encode())
    config = {"flash_attn": False, "op_offload": True}
    proof_config = {"flash_attn": None, "op_offload": None} if stale else config.copy()
    cells = [{"cell_id": str(i), "preparation_error": None, "static_inputs": {"config": config,
        "final_output_selection":True,"mmvq_hbm_mode":d.MODES["on"],
        "final_output_selection_binding": {"config": proof_config, "status": "conditional"}}} for i in range(131)]
    marker = tmp_path / "verifier_called.txt"
    frozen = {"source": {"root": str(source), "sha256": "synthetic"}, "selected_denominator": 131,
        "final_output_selection":True,"mmvq_hbm_mode":d.MODES["on"],
        "selection_sha256": "synthetic", "cells": cells, "call_marker": str(marker)}
    write(directory / "freeze.json", json.dumps(frozen).encode())
    write(tmp_path / "protocol.json", b"{}")
    return directory, marker


def test_same_inputs_allow_only_planner_source_diff(tmp_path,monkeypatch):
    off,on=pair(tmp_path,monkeypatch)
    assert d.compare_inputs(off,on)==131
    assert d.compare_baseline_off(off,copy.deepcopy(off))==131

@pytest.mark.parametrize("change",["cost_source","same_source","member","config","mode","proof","native_evidence","preparation"])
def test_other_changes_fail_closed(tmp_path,monkeypatch,change):
    off,on=pair(tmp_path,monkeypatch);cell=on["cells"]["0"];i=cell["static_inputs"]
    if change=="cost_source":on["source_bytes"]["same"]="modified cost"
    elif change=="same_source":on["source_bytes"]["src/heterollm_sim/planner.py"]="off"
    elif change=="member":on["source_bytes"]["extra"]="file"
    elif change=="config":i["config"]["parallel"]=4
    elif change=="mode":on["mmvq_hbm_mode"]="nominal_bandwidth_analytical_fallback"
    elif change=="proof":i["final_output_selection_binding"]["config"]["op_offload"]=None
    elif change=="native_evidence":i["mmvq_issue_evidence"]["evidence_refs"][0]["sha256"]="a"*64
    else:cell["preparation_error"]="known failure"
    with pytest.raises(ValueError):d.compare_inputs(off,on)


def test_snapshot_uses_all_git_blobs_never_worktree_and_refuses_overwrite(tmp_path,monkeypatch):
    names=["src/heterollm_sim/planner.py","src/heterollm_sim/llama_graph_runtime.py","tools/predict_stable_native_dataset.py"]
    common={"tools/predict_stable_native_dataset.py":"b"*40}
    monkeypatch.setattr(d,"git_members",lambda c:names)
    monkeypatch.setattr(d,"git_blobs",lambda c,n:{name:(c+" committed LF\n").encode() for name in n})
    monkeypatch.setattr(d,"ROOT",tmp_path/"dirty")
    for name in names:write(d.ROOT/name,b"unreviewed WIP\r\n")
    refs=d.snapshot("a"*40,tmp_path/"snapshot",common)
    assert len(refs)==3
    for row in refs:
        origin=common.get(row["relative"],"a"*40)
        assert row["origin"]=="reviewed_git_blob" and row["commit"]==origin
        assert (tmp_path/"snapshot"/row["relative"]).read_bytes()==(origin+" committed LF\n").encode()
    with pytest.raises(FileExistsError):d.snapshot("a"*40,tmp_path/"snapshot",common)


def test_git_tree_rejects_extra_missing_and_nonfile_members(monkeypatch):
    monkeypatch.setattr(d,"SOURCE_COUNT",1)
    for raw in [b"",b"120000 blob "+b"a"*40+b"\tsrc/symlink.py\0",
                b"100644 blob "+b"a"*40+b"\tsrc/x.py\0"+b"100644 blob "+b"b"*40+b"\tsrc/y.py\0"]:
        monkeypatch.setattr(d,"natural_process",lambda *a,**k:SimpleNamespace(stdout=raw))
        with pytest.raises(ValueError):d.git_members("a"*40)


def test_git_batch_preserves_raw_bytes_and_rejects_tamper(monkeypatch):
    content=b"exact LF\n";oid=hashlib.sha1(("blob %d\0"%len(content)).encode()+content).hexdigest()
    packet=(oid+" blob "+str(len(content))+"\n").encode()+content+b"\n"
    seen=[]
    def process(*a,**k):seen.append(k["input"]);return SimpleNamespace(stdout=packet)
    monkeypatch.setattr(d,"natural_process",process)
    assert d.git_blobs("a"*40,["src/x.py"])["src/x.py"]==content
    assert seen==[("a"*40+":src/x.py\n").encode()]
    packet=packet.replace(b"exact",b"wrong")
    with pytest.raises(ValueError):d.git_blobs("a"*40,["src/x.py"])


def test_helpers_include_postprocess_and_archive_and_no_plan():
    names={p.relative_to(d.P).as_posix() for p in d.helper_paths() if p.is_relative_to(d.P)}
    assert {"postprocess/postprocess.py","postprocess/archive_verified.py"}<=names
    assert not any("plan" in n for n in names)


def test_unreviewed_commit_fails_before_any_native_read(monkeypatch):
    monkeypatch.setattr(d,"native_lock",lambda:pytest.fail("must not read real native"))
    with pytest.raises(ValueError,match="full reviewed commit"):d.freeze("8b715dc")


def test_partial_preparation_is_not_adopted(tmp_path,monkeypatch):
    monkeypatch.setattr(d,"P",tmp_path);monkeypatch.setattr(d,"check_commit",lambda _:None)
    (tmp_path/"execution_source").mkdir()
    monkeypatch.setattr(d,"native_lock",lambda:pytest.fail("must not read real native"))
    with pytest.raises(ValueError,match="never adopt"):d.freeze("a"*40)


def test_environment_dependency_identity_is_order_independent_and_changes_on_version(monkeypatch):
    class Dist:
        def __init__(self,name,version):self.metadata={"Name":name};self.version=version
        def read_text(self,name):return self.metadata["Name"]+self.version+name
        def locate_file(self,name):return Path(__file__).parent
    rows=[Dist("numpy","2.0"),Dist("ortools","9.15")]
    monkeypatch.setattr(d.importlib.metadata,"distributions",lambda:iter(rows))
    monkeypatch.setattr(d.s,"reference",lambda _: {"path":"python","bytes":1,"sha256":"a"*64})
    first=d.environment_identity();rows.reverse();assert d.environment_identity()==first
    rows[0].version="changed";assert d.environment_identity()!=first
    rows.clear()
    with pytest.raises(ValueError,match="required simulator dependencies"):d.environment_identity()


@pytest.mark.parametrize("stale",[False,True])
def test_fresh_preflight_invokes_real_frozen_verifier(tmp_path,stale):
    directory,marker=fake_arm(tmp_path,stale)
    receipt=tmp_path/"receipt.json"
    result=subprocess.run([sys.executable,str(d.P/"verify_frozen_preflight.py"),"--freeze",str(directory/"freeze.json"),
        "--protocol",str(tmp_path/"protocol.json"),"--stage","freeze","--arm","on","--receipt",str(receipt)],capture_output=True,text=True)
    assert marker.exists()
    record=json.loads(receipt.read_text())
    assert (result.returncode==0)==(not stale)
    assert record["status"]==("failed" if stale else "passed")
    assert record["native_run"] is False and record["prediction_run"] is False


def test_archive_accepts_omitted_or_explicit_legacy_but_rejects_nominal():
    spec=importlib.util.spec_from_file_location("r33_archive_test",d.P/"postprocess/archive_verified.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for mode in [None,"legacy_mma_output_wave"]:
        value={"final_output_selection":True}
        if mode is not None:value["mmvq_hbm_mode"]=mode
        module.check_treatment(value)
    with pytest.raises(ValueError):module.check_treatment({"final_output_selection":True,"mmvq_hbm_mode":"nominal_bandwidth_analytical_fallback"})


def origin_checker():
    spec=importlib.util.spec_from_file_location("r33_origin_tests",d.P/"verify_frozen_preflight.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_real_namespace_requires_all_search_locations_inside_arm(tmp_path):
    import importlib.machinery
    from types import ModuleType
    checker=origin_checker();root=tmp_path/"source";directory=root/"tools";directory.mkdir(parents=True)
    loader=importlib.machinery.NamespaceLoader("tools",[str(directory)],importlib.machinery.PathFinder)
    module=ModuleType("tools");module.__spec__=importlib.machinery.ModuleSpec("tools",loader,is_package=True)
    module.__spec__.submodule_search_locations=[str(directory)];module.__path__=[str(directory)]
    checker.assert_imports_confined(root,{"tools":module})
    outside=tmp_path/"foreign";outside.mkdir()
    module.__path__.append(str(outside));module.__spec__.submodule_search_locations.append(str(outside))
    with pytest.raises(ValueError,match="namespace origin"):checker.assert_imports_confined(root,{"tools":module})


@pytest.mark.parametrize("kind",["missing_file","none_module","foreign_file","spec_mismatch","empty_namespace"])
def test_unknown_nonempty_module_origins_are_rejected(tmp_path,kind):
    from types import ModuleType
    checker=origin_checker();root=tmp_path/"source";root.mkdir()
    name="heterollm_sim.real_module";module=ModuleType(name)
    if kind=="none_module":module=None
    elif kind in ("foreign_file","spec_mismatch"):
        location=(tmp_path if kind=="foreign_file" else root)/"module.py";location.write_text("x=1")
        module.__file__=str(location);module.__spec__=importlib.util.spec_from_file_location(name,location)
        if kind=="spec_mismatch":module.__spec__.origin=str(tmp_path/"missing.py")
    elif kind=="empty_namespace":
        module.__spec__=importlib.machinery.ModuleSpec(name,None,is_package=True);module.__path__=[]
    with pytest.raises((ValueError,OSError)):checker.assert_imports_confined(root,{name:module})


def test_preflight_catches_module_loaded_after_initial_origin_check(tmp_path):
    directory,marker=fake_arm(tmp_path,False)
    path=directory/"source/tools/predict_stable_native_dataset.py"
    text=path.read_text();text=text.replace("def configuration(inputs):return None", "def configuration(inputs):\n    import sys,types\n    sys.modules['heterollm_sim.late_unknown']=types.ModuleType('heterollm_sim.late_unknown')")
    path.write_text(text)
    receipt=tmp_path/"receipt.json"
    result=subprocess.run([sys.executable,str(d.P/"verify_frozen_preflight.py"),"--freeze",str(directory/"freeze.json"),
        "--protocol",str(tmp_path/"protocol.json"),"--stage","freeze","--arm","on","--receipt",str(receipt)],capture_output=True,text=True)
    assert marker.exists() and result.returncode!=0
    assert "late_unknown" in json.loads(receipt.read_text())["reason"]


def test_new_norm_binding_does_not_change_static_freeze_inputs_or_header_scope(tmp_path):
    import ast,os,re,struct
    from collections.abc import Mapping
    root=d.ROOT;oldroot=d.LOOP/"round_030/off/source"
    def committed(name):
        return subprocess.run(["git","-C",str(root),"show",d.BASELINE_COMMIT+":"+name],capture_output=True,check=True).stdout.decode()
    oldapi=(oldroot/"tools/predict_stable_native_dataset.py").read_text(encoding="utf-8")
    newapi=committed("tools/predict_stable_native_dataset.py")
    assert oldapi==newapi  # The real static constructor/extractor and freeze call chain are byte-identical.
    def nodes(source,names):
        tree=ast.parse(source)
        return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    oldgguf=(oldroot/"src/heterollm_sim/gguf_parity.py").read_text(encoding="utf-8")
    newgguf=committed("src/heterollm_sim/gguf_parity.py")
    names={"_read_string","_read_value"}
    assert [ast.dump(n) for n in nodes(oldgguf,names)]==[ast.dump(n) for n in nodes(newgguf,names)]
    assert "gguf_output_norm_binding" not in oldgguf and "gguf_output_norm_binding" in newgguf
    # A tiny actual GGUF metadata header, including norm epsilon. No tensor data
    # is necessary or read by this freeze-stage projection.
    def string(value):
        data=value.encode();return struct.pack("<Q",len(data))+data
    pairs=[("general.architecture",8,"qwen2"),("qwen2.block_count",4,2),
           ("qwen2.attention.head_count",4,4),("qwen2.attention.head_count_kv",4,2),
           ("qwen2.attention.layer_norm_rms_epsilon",6,1e-5)]
    raw=b"GGUF"+struct.pack("<IQQ",3,1,len(pairs))
    for key,kind,value in pairs:raw+=string(key)+struct.pack("<I",kind)+(string(value) if kind==8 else struct.pack("<f" if kind==6 else "<I",value))
    path=tmp_path/"tiny.gguf";path.write_bytes(raw)
    ref={"path":str(path.resolve()),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    outputs=[]
    for source,ggufsource in [(oldapi,oldgguf),(newapi,newgguf)]:
        selected=nodes(source,{"canonical_retained_model_ref","read_retained_gguf_scope","static_inputs"})
        class ReaderInjection(ast.NodeTransformer):
            def visit_ImportFrom(self,node):return None if node.module=="heterollm_sim.gguf_parity" else node
        selected=[ReaderInjection().visit(n) for n in selected]
        constants=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="STATIC_KEYS" for t in n.targets))
        namespace={"Path":Path,"Mapping":Mapping,"re":re,"os":os,"struct":struct,"GGUFError":ValueError,
            "MMVQ_HBM_MODE_LEGACY":"legacy_mma_output_wave","validate_mmvq_hbm_mode":lambda _:None,
            "measurement_clock_snapshot":lambda *a:([2400],"synthetic static clock",[]),
            "compiled_graph_evidence":lambda *a,**k:{"compiled_cuda_graphs":False},
            "grid":SimpleNamespace(resolve_data=lambda path,root:Path(path).resolve())}
        module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),constants,
                               *nodes(ggufsource,names),*selected],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),"synthetic-frozen-functions","exec"),namespace)
        inputs={"cell_id":"synthetic","model_key":"tiny","model_ref":ref,
            "native_runtime_refs":[{"path":str(tmp_path/"llama-server.exe"),"sha256":"b"*64}],
            "config":{"model":str(path.resolve()),"prompt_token_ids":[1,2],"output":2,"parallel":1,"gpu_layers":0},
            "static_hardware":{"frozen_hardware":{"gpu":{"uuid":"synthetic","clocks":{"sm_mhz":2400}}}}}
        outputs.append((namespace["read_retained_gguf_scope"](ref),namespace["static_inputs"](inputs,{},tmp_path)))
    assert outputs[0]==outputs[1]
    assert "gguf_output_norm_binding" not in repr(outputs[1])


@pytest.mark.parametrize("error",[OSError("observe IO"),SystemExit("observe exit"),KeyboardInterrupt("observe keyboard")])
def test_natural_wait_retains_first_observer_error_until_same_child_exits(tmp_path,monkeypatch,error):
    monkeypatch.setattr(d,"P",tmp_path);calls=[];delays=[]
    class Child:
        pid=100;returncode=None
        def communicate(self,input=None):
            calls.append(input)
            if len(calls)==1:raise error
            self.returncode=0;return b"complete",b""
        def poll(self):return self.returncode
        def kill(self):pytest.fail("must never kill")
        def terminate(self):pytest.fail("must never terminate")
    child=Child();spawns=[]
    monkeypatch.setattr(d.subprocess,"Popen",lambda *a,**k:spawns.append(a) or child)
    monkeypatch.setattr(d.time,"sleep",lambda delay:delays.append(delay))
    with pytest.raises(type(error)) as caught:d.natural_process(["synthetic"],input=b"once",capture_output=True)
    assert caught.value is error and child.returncode==0 and len(spawns)==1
    assert calls==[b"once",None] and delays and delays[0]>0
    finish=json.loads(next((tmp_path/"process_observations").glob("*/finish.json")).read_text())
    assert finish["natural_exit_observed"] and finish["parent_phase_rejected"] and not finish["automatic_retry"]


@pytest.mark.parametrize("stage",["child.json","observation-interrupted.0001.json","finish.json"])
def test_journal_failure_after_spawn_never_strands_live_child(tmp_path,monkeypatch,stage):
    monkeypatch.setattr(d,"P",tmp_path);write_real=d.write_new;calls=[]
    original=OSError("first observation");journal_error=OSError("journal failure")
    class Child:
        pid=100;returncode=None
        def communicate(self,input=None):
            calls.append(input)
            if stage.startswith("observation") and len(calls)==1:raise original
            self.returncode=0;return None,None
        def poll(self):return self.returncode
        def kill(self):pytest.fail("kill forbidden")
        def terminate(self):pytest.fail("terminate forbidden")
    child=Child()
    def writing(path,value):
        if path.name==stage:
            if stage=="finish.json":assert child.returncode==0
            else:assert child.returncode is None
            raise journal_error
        return write_real(path,value)
    monkeypatch.setattr(d,"write_new",writing)
    monkeypatch.setattr(d.subprocess,"Popen",lambda *a,**k:child)
    monkeypatch.setattr(d.time,"sleep",lambda _:None)
    with pytest.raises(OSError) as caught:d.natural_process(["synthetic"])
    assert child.returncode==0 and calls
    assert caught.value is (original if stage.startswith("observation") else journal_error)


def test_start_receipt_failure_never_launches_a_child(tmp_path,monkeypatch):
    monkeypatch.setattr(d,"P",tmp_path)
    monkeypatch.setattr(d,"write_new",lambda *a:(_ for _ in ()).throw(OSError("start receipt failure")))
    monkeypatch.setattr(d.subprocess,"Popen",lambda *a,**k:pytest.fail("no child without start receipt"))
    with pytest.raises(OSError,match="start receipt"):d.natural_process(["synthetic"])


def test_persistent_observation_and_journal_errors_back_off_then_reject(tmp_path,monkeypatch):
    monkeypatch.setattr(d,"P",tmp_path);calls=[];delays=[];write_real=d.write_new
    original=OSError("persistent observation failure")
    class Child:
        pid=100;returncode=None
        def communicate(self,input=None):
            calls.append(input)
            if len(calls)>4:self.returncode=0
            raise original
        def poll(self):return self.returncode
    child=Child()
    def writing(path,value):
        if path.name not in ("start.json","finish.json"):raise OSError("journal unavailable")
        return write_real(path,value)
    monkeypatch.setattr(d,"write_new",writing)
    monkeypatch.setattr(d.subprocess,"Popen",lambda *a,**k:child)
    monkeypatch.setattr(d.time,"sleep",lambda delay:delays.append(delay))
    with pytest.raises(OSError,match="journal unavailable"):d.natural_process(["synthetic"])
    assert child.returncode==0 and len(calls)==5 and delays==[.05,.1,.2,.4]
    finish=json.loads(next((tmp_path/"process_observations").glob("*/finish.json")).read_text())
    assert finish["first_error_stage"]=="journal:child.json" and finish["parent_phase_rejected"]


COMMON_PATH = "tools/predict_stable_native_dataset.py"
REVIEWED = "b" * 40


def source_protocol_fixture(tmp_path, monkeypatch, *, extra_source_diff=False):
    """Exercise the real protocol/provenance checks with tiny committed-source fixtures."""
    names = sorted(["src/heterollm_sim/planner.py", "src/heterollm_sim/cost_models.py", COMMON_PATH])
    monkeypatch.setattr(d, "P", tmp_path)
    monkeypatch.setattr(d, "BASE", tmp_path / "r30off.json")
    monkeypatch.setattr(d, "SOURCE_COUNT", len(names))
    monkeypatch.setattr(d, "check_commit", lambda commit: None)
    monkeypatch.setattr(d, "environment_identity", lambda: {"synthetic_runtime": "unchanged"})
    monkeypatch.setattr(d, "helper_paths", lambda: [])
    monkeypatch.setattr(d, "git_members", lambda commit: names)
    def blobs(commit, requested):
        return {name: ((commit if name in {COMMON_PATH, "src/heterollm_sim/planner.py"}
                        or extra_source_diff else "same-cost") + ":" + name).encode()
                for name in requested}
    monkeypatch.setattr(d, "git_blobs", blobs)
    baseline_ref = write(d.BASE, b"synthetic static R30/off")
    commits = {"off": d.BASELINE_COMMIT, "on": REVIEWED}
    common = {COMMON_PATH: REVIEWED}
    refs = {arm: d.snapshot(commit, tmp_path / "execution_source" / arm, common)
            for arm, commit in commits.items()}
    protocol = {"schema": "r33-clean-git-frontend-inputs/v1", "arms": list(d.ARMS),
        "denominator": 131, "strict_threshold_pct": 10, "gate_B": "unvalidated", "new_cost_coefficients": 0,
        "modes": d.MODES, "final_output_selection_both_arms": True,
        "worker_wait_policy": "natural_exit_soft_observation", "soft_observation_seconds": 600,
        "hard_time_limit_enforced": False, "reviewed_commit": REVIEWED,
        "source_commits": commits, "allowed_source_diff": sorted(d.ALLOWED_SOURCE_DIFF),
        "common_source_overrides": common, "python_environment": d.environment_identity(),
        "baseline_ref": baseline_ref, "helpers": [], "helper_copies": [], "source_refs": refs}
    write(tmp_path / "protocol.json", json.dumps(protocol).encode())
    return protocol


def test_protocol_accepts_common_patch_with_exact_per_file_origins(tmp_path, monkeypatch):
    expected = source_protocol_fixture(tmp_path, monkeypatch)
    actual, _ = d.read_protocol()
    assert actual == expected
    for arm, rows in actual["source_refs"].items():
        for row in rows:
            origin = REVIEWED if row["relative"] == COMMON_PATH else actual["source_commits"][arm]
            assert row["commit"] == origin
    off = {row["relative"]: row["ref"]["sha256"] for row in actual["source_refs"]["off"]}
    on = {row["relative"]: row["ref"]["sha256"] for row in actual["source_refs"]["on"]}
    assert {name for name in off if off[name] != on[name]} == d.ALLOWED_SOURCE_DIFF


@pytest.mark.parametrize("mutation", ["missing", "empty", "extra", "wrong_commit", "short_commit"])
def test_protocol_rejects_common_override_whitelist_or_commit_changes(tmp_path, monkeypatch, mutation):
    protocol = source_protocol_fixture(tmp_path, monkeypatch)
    if mutation == "missing": protocol.pop("common_source_overrides")
    elif mutation == "empty": protocol["common_source_overrides"] = {}
    elif mutation == "extra": protocol["common_source_overrides"]["src/heterollm_sim/cost_models.py"] = REVIEWED
    elif mutation == "wrong_commit": protocol["common_source_overrides"][COMMON_PATH] = d.BASELINE_COMMIT
    else: protocol["common_source_overrides"][COMMON_PATH] = REVIEWED[:7]
    write(tmp_path / "protocol.json", json.dumps(protocol).encode())
    with pytest.raises(ValueError, match="common source override whitelist/commit"):
        d.read_protocol()


@pytest.mark.parametrize("arm,path", [
    ("off", COMMON_PATH), ("on", COMMON_PATH),
    ("off", "src/heterollm_sim/cost_models.py"), ("on", "src/heterollm_sim/cost_models.py")])
def test_protocol_rejects_wrong_origin_on_each_common_or_arm_row(tmp_path, monkeypatch, arm, path):
    protocol = source_protocol_fixture(tmp_path, monkeypatch)
    row = next(row for row in protocol["source_refs"][arm] if row["relative"] == path)
    row["commit"] = "c" * 40
    write(tmp_path / "protocol.json", json.dumps(protocol).encode())
    with pytest.raises(ValueError, match="source origin differs"):
        d.read_protocol()


def test_protocol_rejects_old_diagnostic_bytes_even_with_refreshed_file_reference(tmp_path, monkeypatch):
    protocol = source_protocol_fixture(tmp_path, monkeypatch)
    row = next(row for row in protocol["source_refs"]["off"] if row["relative"] == COMMON_PATH)
    row["ref"] = write(Path(row["ref"]["path"]), d.git_blobs(d.BASELINE_COMMIT, [COMMON_PATH])[COMMON_PATH])
    write(tmp_path / "protocol.json", json.dumps(protocol).encode())
    with pytest.raises(ValueError, match="snapshot differs from reviewed blob"):
        d.read_protocol()


def test_protocol_rejects_any_additional_paired_source_difference(tmp_path, monkeypatch):
    source_protocol_fixture(tmp_path, monkeypatch, extra_source_diff=True)
    with pytest.raises(ValueError, match="exactly reviewed frontend planner change"):
        d.read_protocol()


@pytest.mark.parametrize("overrides", [None, {}, {"src/heterollm_sim/planner.py": REVIEWED},
    {COMMON_PATH: REVIEWED, "src/heterollm_sim/cost_models.py": REVIEWED}])
def test_materialization_rejects_undeclared_overrides_before_reading_git(monkeypatch, overrides):
    monkeypatch.setattr(d, "git_blobs", lambda *args: pytest.fail("invalid override must fail before any Git read"))
    with pytest.raises(ValueError, match="common source overrides must be exactly"):
        d.source_blobs(d.BASELINE_COMMIT, [COMMON_PATH], overrides)
