"""Small synthetic IR replay only; no real model parse, DES run or GPU."""
from dataclasses import replace
from heterollm_sim import planner, serving
from heterollm_sim.cost_models import MMVQ_HBM_MODE_LEGACY, MMVQ_HBM_MODE_NOMINAL
from tests.test_mmvq_issue_bound import scenario


def case(mode):
    base=scenario("Q4_K")
    return replace(base,workload=replace(base.workload,metadata={**base.workload.metadata,
        "llama_cpp_mmvq_hbm_mode":mode}))


def cohort(context,name):
    return serving.BatchCohort(name,"decode",0,(
        serving.BatchItem("synthetic","decode",1,context,kv_append_tokens=1,
                          kv_materialized_tokens=context,logit_tokens=1),))


def test_context_growth_template_replay_matches_fresh_for_both_modes():
    previous,target=cohort(32,"prior-context"),cohort(128,"target-context")
    outputs={}
    for mode in (MMVQ_HBM_MODE_LEGACY,MMVQ_HBM_MODE_NOMINAL):
        scene=case(mode)
        expected=planner.compile_serving_cohort_schedule(scene,target)
        with planner._compilation_scope(scene):
            planner.compile_serving_cohort_schedule(scene,previous)
            actual=planner.compile_serving_cohort_schedule(scene,target)
        assert actual==expected
        coverage=actual.manifest.metadata["mmvq_hbm_mode"]
        assert coverage["requested_mode"]==mode and coverage["requested_unaccounted_tasks"]==0
        if mode==MMVQ_HBM_MODE_NOMINAL:
            assert coverage["applied_tasks"]>0
            assert coverage["applied_tasks"]+coverage["fallback_tasks"]==coverage["gpu_gemm_tasks"]
        else:assert coverage["applied_tasks"]==0
        outputs[mode]=actual
    # Reopening the legacy scope after the nominal run must not reuse its rates.
    legacy_again=planner.compile_serving_cohort_schedule(case(MMVQ_HBM_MODE_LEGACY),target)
    assert legacy_again==outputs[MMVQ_HBM_MODE_LEGACY]
    assert [(t.task_id,t.name,t.dependencies) for t in outputs[MMVQ_HBM_MODE_LEGACY].tasks]==[(t.task_id,t.name,t.dependencies) for t in outputs[MMVQ_HBM_MODE_NOMINAL].tasks]
