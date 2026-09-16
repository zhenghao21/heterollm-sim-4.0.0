"""Synthetic lowering and temporary fixture freezes only; no native/model/GPU run."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from heterollm_sim import planner, serving
from heterollm_sim.cost_models import MMVQ_HBM_MODE_LEGACY as LEGACY, MMVQ_HBM_MODE_NOMINAL as NOMINAL
from heterollm_sim.kernel_query_ledger import summarize_kernel_queries
from tests.test_mmvq_issue_bound import scenario
from tests.test_predict_stable_native_dataset import fixture as native_fixture
from tools import predict_stable_native_dataset as adapter

KEY="llama_cpp_mmvq_hbm_mode"


def with_mode(case,mode):
    return replace(case,workload=replace(case.workload,metadata={**case.workload.metadata,KEY:mode}))


def physical_demands(schedule):
    return {t.task_id:t.demands for t in schedule.tasks}


def test_default_legacy_keeps_graph_and_all_resource_costs():
    case=scenario("Q4_K")
    baseline=planner.compile_scenario(case);explicit=planner.compile_scenario(with_mode(case,LEGACY))
    assert baseline.tasks==explicit.tasks
    assert "mmvq_hbm_mode" not in baseline.manifest.metadata
    summary=explicit.manifest.metadata["mmvq_hbm_mode"]
    assert summary["applied_tasks"]==0 and summary["requested_tasks"]==0


def test_actual_planner_mode_changes_only_hbm_and_is_in_memo_key(monkeypatch):
    case=scenario("Q4_K");captured=[];original=planner._memoized_cost_estimate
    def observe(scene,key,factory):
        if key[0]=="gpu_gemm":captured.append(key)
        return original(scene,key,factory)
    monkeypatch.setattr(planner,"_memoized_cost_estimate",observe)
    before=planner.compile_scenario(case);after=planner.compile_scenario(with_mode(case,NOMINAL))
    assert {key[-1] for key in captured}=={LEGACY,NOMINAL}
    assert [(t.task_id,t.name,t.dependencies) for t in before.tasks]==[(t.task_id,t.name,t.dependencies) for t in after.tasks]
    changed=0
    for a,b in zip(before.tasks,after.tasks):
        assert len(a.demands)==len(b.demands)
        for x,y in zip(a.demands,b.demands):
            assert replace(y,service_ns=x.service_ns)==x
            if y.service_ns!=x.service_ns:
                changed+=1
                assert y.resource_id=="hbm0.hbm_fabric" and b.metadata["phase"]=="gpu_gemm"
    assert changed>0
    summary=after.manifest.metadata["mmvq_hbm_mode"]
    assert summary["requested_tasks"]==summary["applied_tasks"]>0
    assert summary["fallback_tasks"]==summary["requested_unaccounted_tasks"]==0
    assert summary["accuracy_validated_tasks"]==0 and not summary["bandwidth_saturation_proven"]
    ledger=summarize_kernel_queries(after.tasks)
    assert any(x["key"]["k_execution_source"]=="source_derived_mmvq_logical_K" for x in ledger["signatures"])


def test_missing_source_keeps_real_lowered_tasks_and_reports_fallback():
    case=scenario("Q4_K")
    components=tuple(replace(c,metadata={k:v for k,v in c.metadata.items() if k!="llama_cpp_conversion_source_contract"}) for c in case.hardware.components)
    case=replace(case,hardware=replace(case.hardware,components=components))
    before=planner.compile_scenario(case);after=planner.compile_scenario(with_mode(case,NOMINAL))
    assert physical_demands(before)==physical_demands(after)
    summary=after.manifest.metadata["mmvq_hbm_mode"]
    assert summary["applied_tasks"]==0 and summary["fallback_tasks"]==summary["gpu_gemm_tasks"]>0
    assert summary["requested_unaccounted_tasks"]==0 and summary["fallback_reason_counts"]


def test_serving_cohort_keeps_mode_coverage_in_retained_report():
    case=with_mode(scenario("Q4_K"),NOMINAL)
    cohort=serving.BatchCohort("synthetic-mode", "prefill",0,(serving.BatchItem("sample","prefill",4,4,logit_tokens=0),))
    cost=planner.estimate_serving_cohort_cost(case,cohort)
    mode=cost["metadata"]["mmvq_hbm_mode"]
    assert mode["requested_unaccounted_tasks"]==0 and mode["applied_tasks"]>0
    batch=SimpleNamespace(cost=SimpleNamespace(metadata=cost["metadata"]))
    fake=SimpleNamespace(serving=SimpleNamespace(batches=(batch,batch),scheduler_metrics=SimpleNamespace(total_batches=2)))
    summary=adapter.retained_dispatch_summary(fake)["mmvq_hbm_mode"]
    assert summary["applied_tasks"]==2*mode["applied_tasks"] and summary["fallback_tasks"]==2*mode["fallback_tasks"]
    assert summary["requested_modes"]==[NOMINAL] and summary["all_batches_summarized"]
    assert summary["accuracy_validated"] is False
    assert adapter.compact_dispatch_evidence(cost["metadata"])["mmvq_hbm_mode"]


@pytest.mark.parametrize("bad",[None,True,1,"measured"])
def test_bad_mode_rejected_at_static_and_streaming_boundaries(bad):
    for compile in (planner.compile_scenario,planner.compile_streaming_scenario):
        with pytest.raises(ValueError,match="mmvq_hbm_mode"):compile(with_mode(scenario(),bad))


def test_adapter_default_unchanged_and_explicit_mode_reaches_stubbed_predictor(tmp_path,monkeypatch):
    _,selection,row,calls=native_fixture(tmp_path,monkeypatch)
    implicit=adapter.static_inputs(row,selection,tmp_path);explicit=adapter.static_inputs(row,selection,tmp_path,mmvq_hbm_mode=LEGACY)
    assert implicit==explicit and "mmvq_hbm_mode" not in implicit
    assert adapter.apply_mmvq_hbm_static_contract(object(),{}) is not None
    inputs=adapter.static_inputs(row,selection,tmp_path,mmvq_hbm_mode=NOMINAL)
    assert {k:v for k,v in inputs.items() if k!="mmvq_hbm_mode"}==implicit
    output=adapter.predict_cell(inputs)
    assert calls["run"][0].workload.metadata[KEY]==NOMINAL
    assert output["input_identity"]["mmvq_hbm_mode"]==NOMINAL
    assert output["dispatch_qualification"]["mmvq_hbm_mode"]["accuracy_validated"] is False
    assert "native_latency_ms" not in json.dumps(inputs)


def test_new_freeze_binds_mode_per_cell_and_resume_rejects_mutation(tmp_path,monkeypatch):
    selection,_,_,_=native_fixture(tmp_path,monkeypatch)
    freeze=adapter.freeze_selection(selection,tmp_path/"new-nominal",data_root=tmp_path,mmvq_hbm_mode=NOMINAL)
    assert freeze["mmvq_hbm_mode"]==NOMINAL and freeze["cells"][0]["static_inputs"]["mmvq_hbm_mode"]==NOMINAL
    adapter.verify_freeze_references(freeze)
    bad=copy.deepcopy(freeze);del bad["cells"][0]["static_inputs"]["mmvq_hbm_mode"]
    with pytest.raises(ValueError,match="cell mode"):adapter.verify_freeze_references(bad)
    bad=copy.deepcopy(freeze);bad["mmvq_hbm_mode"]=None
    with pytest.raises(ValueError,match="mmvq_hbm_mode"):adapter.verify_mmvq_hbm_freeze_binding(bad)
    old={"cells":[{"static_inputs":{}},{"static_inputs":None,"preparation_error":"kept failure"}]}
    adapter.verify_mmvq_hbm_freeze_binding(old)


def test_cli_mode_is_initial_freeze_only(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(adapter,"freeze_selection",lambda *a,**k:calls.append(k))
    adapter.main(["--selection","synthetic.json","--output",str(tmp_path),"--freeze-only","--mmvq-hbm-mode",NOMINAL])
    assert calls[0]["mmvq_hbm_mode"]==NOMINAL
    for args in (["--output",str(tmp_path),"--resume"],["--worker-freeze","synthetic.json"]):
        with pytest.raises(SystemExit) as error:adapter.main([*args,"--mmvq-hbm-mode",NOMINAL])
        assert error.value.code==2


def test_legacy_frozen_choice_does_not_accept_pre_enabled_scenario():
    with pytest.raises(ValueError,match="conflicts"):adapter.apply_mmvq_hbm_static_contract(with_mode(scenario(),NOMINAL),{})


def host_declared_case(offload):
    from heterollm_sim.cost_models import HostGemmOffloadCapability
    case=scenario("Q4_K")
    placement=replace(case.placement,op_to_component={**case.placement.op_to_component,"attention":"cpu0","full0.norm":"cpu0"},
        tensor_to_component={**case.placement.tensor_to_component,"full0.attention_weights":"hostmem0"})
    profiles={kind:dict(registry) for kind,registry in case.component_profiles.items()}
    for ident,profile in profiles["gpu"].items():
        profiles["gpu"][ident]=replace(profile,host_gemm_offload=HostGemmOffloadCapability(minimum_m=1,evidence="synthetic mode route test") if offload else None)
    return replace(case,placement=placement,component_profiles=profiles)


@pytest.mark.parametrize("offload",[False,True])
def test_cpu_and_host_offload_select_mode_at_actual_execution_device(offload):
    base=host_declared_case(offload)
    before=planner.compile_scenario(base);after=planner.compile_scenario(with_mode(base,NOMINAL))
    prior={t.task_id:t for t in before.tasks}
    cpu=[t for t in after.tasks if t.metadata.get("phase")=="cpu_gemm"]
    for t in cpu:assert t.demands==prior[t.task_id].demands
    offloaded=[t for t in after.tasks if t.metadata.get("phase")=="gpu_gemm" and t.metadata.get("host_gemm_offload_applied")]
    assert bool(offloaded)==offload
    if not offload:assert cpu
    for t in offloaded:
        assert t.metadata["execution_component"]=="gpu0"
        assert t.metadata["cost_model"]["hbm_bandwidth"]["requested_mmvq_hbm_mode"]==NOMINAL
    assert after.manifest.metadata["mmvq_hbm_mode"]["requested_unaccounted_tasks"]==0


def test_existing_workload_metadata_configuration_round_trips_mode():
    from heterollm_sim.config import scenario_from_dict
    from heterollm_sim.serde import to_primitive
    from tests.test_config_v05 import reference_payload
    payload=reference_payload();payload["workload"]["metadata"][KEY]=NOMINAL
    parsed=scenario_from_dict(payload)
    assert planner._mmvq_hbm_mode(parsed)==NOMINAL
    payload["workload"]=to_primitive(parsed.workload)
    assert planner._mmvq_hbm_mode(scenario_from_dict(payload))==NOMINAL
