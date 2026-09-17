"""Pure structural checks; no native executable, clocks or GPU calls."""
from dataclasses import replace
import pytest
from heterollm_sim.llama_graph_lifecycle import (
    GraphParameters, GraphState, InputCheck, process_ubatch, reserve, execution_failed,
)


def params(**changes):
    base = GraphParameters(equal_seqs=True, n_tokens=1, n_seq_tokens=1, n_seqs=1,
        n_seqs_unq=1, has_tokens=True, has_embeddings=False, owns_ubatch_data=True,
        sequence_ids=(7,), n_outputs=1, samplers=(), sampler_outputs=(),
        nextn_layer_offset=0, embeddings=False, embeddings_nextn=False,
        embeddings_nextn_masked=False, causal_attn=True, architecture="qwen35",
        graph_type="decoder", cvec_identity="none", loras_identity="none", cross_identity="none")
    return replace(base, **changes)


ROSTER = (("tokens", "embedding"), ("position", "position"))
CHECKS = (InputCheck("tokens", "embedding", (("token_elements", 1, 1),)),
          InputCheck("position", "position", (("position_elements", 1, 1),)))


def step(state, value=None, **changes):
    options = dict(input_roster=ROSTER, input_checks=CHECKS, disable_reuse=False, pipeline_parallel=False)
    options.update(changes)
    return process_ubatch(state, value or params(), **options)


def built():
    return step(GraphState()).state


def kinds(result):
    return tuple(event.kind for event in result.events)


def test_first_build_then_reuse_preserves_generation_and_cpu_only_events():
    first=step(GraphState())
    assert first.decision=="miss" and kinds(first)==("check","reset","build","alloc_plan","set_inputs","compute")
    second=step(first.state)
    assert second.decision=="hit" and kinds(second)==("check","reuse","set_inputs","compute")
    assert second.state is first.state and first.state.generation==1
    assert not first.native_execution_verified and not second.native_execution_verified
    assert all("gpu" not in kind for kind in kinds(first)+kinds(second))


def test_token_values_are_not_topology_parameters():
    # Caller may upload arbitrary new token values without changing this key.
    assert "token_values" not in GraphParameters.__dataclass_fields__
    assert step(built(), params()).decision=="hit"


@pytest.mark.parametrize("change", [
    {"n_tokens":2},{"n_seq_tokens":2},{"n_seqs":2},{"n_seqs_unq":2},
    {"n_outputs":0},{"equal_seqs":False},{"sequence_ids":(8,)},
    {"has_tokens":False,"has_embeddings":True},{"embeddings":True},
    {"embeddings_nextn":True},{"embeddings_nextn_masked":True},{"causal_attn":False},
    {"graph_type":"encoder"},{"architecture":"other"},{"nextn_layer_offset":1},
    {"cvec_identity":"adapter"},{"loras_identity":"lora"},{"cross_identity":"cross"},
    {"samplers":((7,"sampler"),),"sampler_outputs":((True,7),)},
])
def test_source_parameter_invalidations(change):
    result=step(built(),params(**change))
    assert result.decision=="miss" and "build" in kinds(result)


def test_disable_every_time_and_reserve_invalidation():
    state=built()
    for _ in range(3):
        result=step(state,disable_reuse=True)
        assert result.decision=="miss" and "reuse_disabled" in result.reasons
        state=result.state
    invalid=reserve(state)
    assert invalid.state.previous is None and invalid.decision=="invalidated"
    assert "reserve_plan" in kinds(invalid) and "buffer_expand" not in kinds(invalid)
    assert step(invalid.state).decision=="miss"


@pytest.mark.parametrize("change", [{"n_tokens":None},{"sequence_ids":None},{"loras_identity":None},{"samplers":None}])
def test_missing_source_parameters_never_hit(change):
    result=step(built(),params(**change))
    assert result.decision=="unknown" and "reuse" not in kinds(result)
    assert next(e for e in result.events if e.kind=="build").evidence=="conservative_unknown"


def test_unknown_disable_and_pipeline_do_not_claim_source_observation():
    assert step(built(),disable_reuse=None).decision=="unknown"
    result=step(built(),pipeline_parallel=None)
    assert "synchronization_requirement_unknown" in kinds(result)
    assert not result.native_execution_verified


def test_input_roster_missing_incomplete_or_unsupported_is_unknown():
    assert step(GraphState(params(),None)).decision=="unknown"
    assert step(built(),input_checks=CHECKS[:1]).decision=="unknown"
    state=GraphState(params(),(("special","unimplemented_arch_input"),))
    result=step(state,input_checks=(InputCheck("special","unimplemented_arch_input"),))
    assert result.decision=="unknown" and any("unsupported_input" in reason for reason in result.reasons)


def test_input_shape_and_recurrent_head_invalidation():
    roster=(("memory","mem_hybrid"),)
    values=dict(key_index_elements=1,mask_ne0=256,mask_ne1=1,mask_ne2=1,mask_ne3=1,
                copy_elements=2,main_elements=1,extra_elements=1,head=0,rs_z=0)
    state=GraphState(params(),roster)
    checks=lambda change: (InputCheck("memory","mem_hybrid",tuple((k,v,change.get(k,v)) for k,v in values.items())),)
    assert step(state,input_checks=checks({})).decision=="hit"
    for change in ({"mask_ne0":512},{"head":1},{"rs_z":1},{"extra_elements":2}):
        result=step(state,input_checks=checks(change))
        assert result.decision=="miss" and any("input_changed" in reason for reason in result.reasons)
    assert step(state,input_checks=(InputCheck("memory","mem_hybrid",(("head",0,0),)),)).decision=="unknown"


def test_sampler_binding_output_map_and_owned_data():
    p=params(samplers=((7,"sampler"),),sampler_outputs=((True,7),))
    state=GraphState(p,ROSTER)
    assert step(state,p).decision=="hit"
    assert step(state,replace(p,sampler_outputs=((False,7),))).decision=="miss"
    assert step(state,replace(p,sampler_outputs=None)).decision=="unknown"
    assert step(state,replace(p,owns_ubatch_data=False)).decision=="miss"


def test_non_equal_sequences_do_not_invent_participant_comparison():
    state=GraphState(params(equal_seqs=False,sequence_ids=(7,)),ROSTER)
    assert step(state,params(equal_seqs=False,sequence_ids=(8,))).decision=="hit"


def test_allocation_plan_does_not_imply_actual_buffer_expansion():
    first=step(GraphState())
    assert first.buffer_expansion=="unknown" and "buffer_expand" not in kinds(first)
    yes=step(GraphState(),buffer_expansion="required")
    assert kinds(yes).index("alloc_plan")<kinds(yes).index("buffer_expand")<kinds(yes).index("set_inputs")
    no=step(GraphState(),buffer_expansion="not_required")
    assert "alloc_plan" in kinds(no) and "buffer_expand" not in kinds(no)
    with pytest.raises(ValueError,match="reused graph"):step(built(),buffer_expansion="required")


def test_pipeline_dependency_before_set_inputs_only_on_reuse():
    result=step(built(),pipeline_parallel=True)
    assert kinds(result)==("check","reuse","synchronize_before_set_inputs","set_inputs","compute")


@pytest.mark.parametrize("change",[{"n_tokens":True},{"n_outputs":-1},{"equal_seqs":"unknown"},{"graph_type":""},{"samplers":((7,"a"),(7,"b"))}])
def test_invalid_typed_inputs_rejected(change):
    with pytest.raises(ValueError):params(**change)


def test_equal_sequences_checks_old_data_ownership_not_unneeded_new_ownership():
    assert step(GraphState(params(owns_ubatch_data=None),ROSTER)).decision=="unknown"
    assert step(GraphState(params(owns_ubatch_data=False),ROSTER)).decision=="miss"
    # With no samplers, allow_reuse reads ownership only from the old ubatch.
    assert step(built(),params(owns_ubatch_data=None)).decision=="hit"



def test_known_absent_and_unobserved_warm_initial_state_are_distinct():
    absent=GraphState()
    warm=GraphState.unknown()
    assert absent.previous is None and warm.previous is None
    assert not absent.previous_unknown and warm.previous_unknown
    assert warm.input_roster is None
    assert step(absent).decision=="miss"
    result=step(warm)
    assert result.decision=="unknown" and "previous_graph_unknown" in result.reasons
    assert "no_previous_graph" not in result.reasons
    assert "reuse" not in kinds(result) and not result.native_execution_verified
    assert next(e for e in result.events if e.kind=="build").evidence=="conservative_unknown"


@pytest.mark.parametrize("state",[GraphState(),GraphState.unknown()])
def test_disabled_reuse_is_known_miss_even_when_previous_graph_unknown(state):
    result=step(state,disable_reuse=True)
    assert result.decision=="miss" and result.reasons==("reuse_disabled",)
    assert all(e.evidence=="structural" for e in result.events)
    assert not result.native_execution_verified


def test_reserve_turns_unknown_graph_into_known_absent():
    result=reserve(GraphState.unknown(generation=3))
    assert result.decision=="invalidated" and result.state.previous is None
    assert not result.state.previous_unknown and result.state.generation==3
    assert step(result.state).decision=="miss"
    assert "scheduler_reset" in kinds(result)


@pytest.mark.parametrize("state",[GraphState(),GraphState.unknown(),GraphState(params(),ROSTER,5)])
def test_execution_failure_loses_knowledge_without_claiming_reset(state):
    result=execution_failed(state,reason="compute_returned_failure")
    assert result.decision=="invalidated" and result.state.previous_unknown
    assert result.state.previous is None and result.state.input_roster is None
    assert result.state.generation==state.generation
    assert kinds(result)==("execution_failed_graph_unknown",)
    assert result.buffer_expansion=="unknown" and not result.native_execution_verified
    assert step(result.state).decision=="unknown"
    assert step(result.state,disable_reuse=True).decision=="miss"
    assert not reserve(result.state).state.previous_unknown


def test_successful_modeled_transition_from_unknown_does_not_become_native_evidence():
    first=step(GraphState.unknown())
    assert first.decision=="unknown" and not first.state.previous_unknown
    assert first.state.previous==params() and first.state.generation==1
    second=step(first.state)
    assert second.decision=="hit"
    assert not first.native_execution_verified and not second.native_execution_verified
    assert execution_failed(first.state).state.previous_unknown


@pytest.mark.parametrize("kwargs",[
    {"previous":params(),"previous_unknown":True},
    {"input_roster":ROSTER,"previous_unknown":True},
    {"previous_unknown":"unknown"}, {"generation":-1},
])
def test_unknown_state_cannot_smuggle_fabricated_graph_parameters(kwargs):
    with pytest.raises(ValueError):GraphState(**kwargs)


def test_execution_failure_requires_explanatory_reason():
    with pytest.raises(ValueError):execution_failed(built(),reason="")
