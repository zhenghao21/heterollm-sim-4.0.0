"""Source-bound *structural* llama graph reuse transitions, without prices or I/O.

This models one llama_context's previous graph, not a global shape cache. All
records are modeled transitions, never proof of native execution. Callers must
supply the actual ubatch parameters and every cached graph input predicate.
Unknown compatibility uses an explicitly conservative rebuild; it is not a
source-proven miss or permission to claim native reuse. GraphState() means
known absence of a previous graph. After unobserved native warmup, use
GraphState.unknown(); a KV warmup contract does not identify the last graph.
After execution failure, use execution_failed() instead of reserve(): failure
does not prove that the native scheduler/previous graph was reset.

The successful-path boundary starts AFTER memory-context apply succeeds
(llama-context.cpp:1347-1352); the upstream memory-context owner must represent
that operation and its failure path. This module neither skips that ownership
requirement nor emits a second memory-apply operation.

Source paths and line references below document provenance only. They do not
verify the source version or runtime binary used by an execution. Integrators
must bind the locked source/runtime identities before claiming applicability.
"""
from dataclasses import dataclass, fields
from typing import Literal, Optional, Tuple

CONTEXT_SOURCE = "source/llama.cpp-semantic/src/llama-context.cpp:1347-1415"
PARAM_SOURCE = "source/llama.cpp-semantic/src/llama-graph.h:815-883"
INPUT_SOURCE = "source/llama.cpp-semantic/src/llama-graph.cpp:1406-1435"
ALLOC_SOURCE = "source/llama.cpp-semantic/ggml/src/ggml-backend.cpp:1611-1636,1989-2026"
RESERVE_SOURCE = "source/llama.cpp-semantic/src/llama-context.cpp:2429-2475"

# These are exact equality predicates from each named can_reuse method. A
# caller supplies (cached value, required value), not an unexplained bool.
# In particular, KV value indices are NOT checked by the locked source.
INPUT_PREDICATES = {
    "position": (("position_elements",), "llama-graph.cpp:149-154"),
    "out_ids": (("n_outputs",), "llama-graph.cpp:226-231"),
    "kq_mask": (("mask_ne0", "mask_ne1", "mask_ne2", "mask_ne3"), "llama-graph.cpp:48-64"),
    "attn_kv": (("key_index_elements", "mask_ne0", "mask_ne1", "mask_ne2", "mask_ne3"), "llama-graph.cpp:489-501"),
    "recurrent": (("copy_elements", "main_elements", "extra_elements", "head", "rs_z"), "llama-graph.cpp:345-360"),
    "mem_hybrid": (("key_index_elements", "mask_ne0", "mask_ne1", "mask_ne2", "mask_ne3", "copy_elements", "main_elements", "extra_elements", "head", "rs_z"), "llama-graph.cpp:1116-1136"),
    "embedding": ((), "llama-graph.cpp:85-91"),
}


@dataclass(frozen=True)
class GraphParameters:
    """None is unknown; use a stable explicit 'none' identity for absent adapters.

    Identities are context-local handles/keys, not content hashes. token values
    and positions intentionally do not form part of the topology key.
    samplers is the sorted (sequence id, sampler identity) mapping. When nonempty,
    sampler_outputs holds (output flag, first sequence id) for *every* token.
    """
    equal_seqs: Optional[bool] = None
    n_tokens: Optional[int] = None
    n_seq_tokens: Optional[int] = None
    n_seqs: Optional[int] = None
    n_seqs_unq: Optional[int] = None
    has_tokens: Optional[bool] = None
    has_embeddings: Optional[bool] = None
    owns_ubatch_data: Optional[bool] = None
    sequence_ids: Optional[Tuple[int, ...]] = None
    n_outputs: Optional[int] = None
    samplers: Optional[Tuple[Tuple[int, str], ...]] = None
    sampler_outputs: Optional[Tuple[Tuple[bool, int], ...]] = None
    nextn_layer_offset: Optional[int] = None
    embeddings: Optional[bool] = None
    embeddings_nextn: Optional[bool] = None
    embeddings_nextn_masked: Optional[bool] = None
    causal_attn: Optional[bool] = None
    architecture: Optional[str] = None
    graph_type: Optional[str] = None
    cvec_identity: Optional[str] = None
    loras_identity: Optional[str] = None
    cross_identity: Optional[str] = None


    def __post_init__(self):
        boolean_fields = ("equal_seqs", "has_tokens", "has_embeddings", "owns_ubatch_data", "embeddings", "embeddings_nextn", "embeddings_nextn_masked", "causal_attn")
        for name in boolean_fields:
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError("expected bool or unknown: " + name)
        for name in ("n_tokens", "n_seq_tokens", "n_seqs", "n_seqs_unq", "n_outputs", "nextn_layer_offset"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("expected nonnegative integer or unknown: " + name)
        for name in ("architecture", "graph_type", "cvec_identity", "loras_identity", "cross_identity"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("expected explicit identity or unknown: " + name)
        if self.samplers is not None:
            if not isinstance(self.samplers, tuple) or any(type(seq) is not int or not isinstance(identity, str) or not identity for seq, identity in self.samplers):
                raise ValueError("samplers must contain immutable sequence/identity bindings")
            if tuple(sorted(self.samplers)) != self.samplers or len({seq for seq, _ in self.samplers}) != len(self.samplers):
                raise ValueError("sampler bindings must be sorted and unique")
        if self.sequence_ids is not None and (not isinstance(self.sequence_ids, tuple) or any(type(seq) is not int for seq in self.sequence_ids)):
            raise ValueError("sequence participants must be an immutable integer tuple")
        if self.sampler_outputs is not None and (not isinstance(self.sampler_outputs, tuple) or any(type(flag) is not bool or type(seq) is not int for flag, seq in self.sampler_outputs)):
            raise ValueError("sampler outputs must be immutable flag/sequence pairs")


@dataclass(frozen=True)
class InputCheck:
    """One cached graph input's source predicates against the incoming ubatch.

    embedding uses token_elements only if has_tokens, embedding_rows only if
    has_embeddings. The entire cached input roster must be supplied, including
    unsupported inputs: unsupported kinds yield unknown instead of being skipped.
    """
    name: str
    kind: str
    comparisons: Tuple[Tuple[str, Optional[int], Optional[int]], ...] = ()


@dataclass(frozen=True)
class LifecycleEvent:
    kind: str
    source: str
    evidence: Literal["structural", "conservative_unknown"] = "structural"


@dataclass(frozen=True)
class GraphState:
    """One context's modeled graph state; default is explicitly known absent.

    previous_unknown distinguishes an unobserved warmup/failure state from a
    known reset. It never fabricates GraphParameters for an unidentified graph.
    generation counts modeled builds, not observed native builds.
    """
    previous: Optional[GraphParameters] = None
    input_roster: Optional[Tuple[Tuple[str, str], ...]] = None
    generation: int = 0
    previous_unknown: bool = False

    def __post_init__(self):
        if type(self.previous_unknown) is not bool:
            raise ValueError("previous_unknown must be bool")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if self.previous_unknown and (self.previous is not None or self.input_roster is not None):
            raise ValueError("unknown previous graph cannot carry invented parameters or inputs")

    @classmethod
    def unknown(cls, *, generation: int = 0) -> "GraphState":
        """Unobserved initial/warm graph, without invented parameters."""
        return cls(generation=generation, previous_unknown=True)


@dataclass(frozen=True)
class Transition:
    state: GraphState
    decision: Literal["hit", "miss", "unknown", "invalidated"]
    reasons: Tuple[str, ...]
    events: Tuple[LifecycleEvent, ...]
    buffer_expansion: Literal["unknown", "not_required", "required"] = "unknown"
    native_execution_verified: bool = False


def _parameter_reasons(old: GraphParameters, new: GraphParameters):
    unknown, mismatch = [], []
    conditional = {"owns_ubatch_data", "sequence_ids", "sampler_outputs", "has_tokens", "has_embeddings"}
    for item in fields(GraphParameters):
        if item.name in conditional:
            continue
        a, b = getattr(old, item.name), getattr(new, item.name)
        if a is None or b is None:
            unknown.append("unknown_parameter:" + item.name)
        elif a != b:
            mismatch.append("parameter_changed:" + item.name)
    presence = (old.has_tokens, new.has_tokens, old.has_embeddings, new.has_embeddings)
    if any(v is None for v in presence):
        unknown.append("unknown_token_embedding_presence")
    elif not ((not old.has_tokens and not new.has_tokens) or
              (not old.has_embeddings and not new.has_embeddings) or all(presence)):
        mismatch.append("token_embedding_mode_changed")
    if old.equal_seqs is True and new.equal_seqs is True:
        if old.owns_ubatch_data is None:
            unknown.append("unknown_old_ubatch_data_ownership")
        elif not old.owns_ubatch_data:
            mismatch.append("old_equal_sequences_data_not_owned")
        for params in (old, new):
            if params.sequence_ids is None or params.n_seqs_unq is None or len(params.sequence_ids) != params.n_seqs_unq:
                unknown.append("unknown_or_incomplete_sequence_participants")
        if old.sequence_ids is not None and new.sequence_ids is not None and old.sequence_ids != new.sequence_ids:
            mismatch.append("sequence_participants_changed")
    if old.samplers and new.samplers:
        for params in (old, new):
            if params.owns_ubatch_data is None:
                unknown.append("unknown_sampler_ubatch_data")
            elif not params.owns_ubatch_data:
                mismatch.append("sampler_ubatch_data_not_owned")
            if params.sampler_outputs is None or params.n_tokens is None or len(params.sampler_outputs) != params.n_tokens:
                unknown.append("unknown_or_incomplete_sampler_output_mapping")
        if old.sampler_outputs is not None and new.sampler_outputs is not None and old.sampler_outputs != new.sampler_outputs:
            mismatch.append("sampler_output_mapping_changed")
    return unknown, mismatch


def _input_reasons(params, checks, roster):
    unknown, mismatch = [], []
    actual = tuple((c.name, c.kind) for c in checks)
    if roster is None:
        unknown.append("cached_input_roster_unknown")
    elif len({n for n, _ in actual}) != len(actual) or set(actual) != set(roster):
        unknown.append("cached_input_roster_not_completely_covered")
    for check in checks:
        spec = INPUT_PREDICATES.get(check.kind)
        if spec is None:
            unknown.append("unsupported_input:" + check.name + ":" + check.kind)
            continue
        required = spec[0]
        if check.kind == "embedding":
            if params.has_tokens is None or params.has_embeddings is None:
                unknown.append("embedding_input_presence_unknown")
                continue
            required = (("token_elements",) if params.has_tokens else ()) + (("embedding_rows",) if params.has_embeddings else ())
        rows = {name: (a, b) for name, a, b in check.comparisons}
        if len(rows) != len(check.comparisons) or set(rows) != set(required):
            unknown.append("incomplete_or_unknown_input_predicates:" + check.name)
            continue
        for name, (a, b) in rows.items():
            if type(a) is not int or type(b) is not int:
                unknown.append("unknown_input_value:" + check.name + ":" + name)
            elif a != b:
                mismatch.append("input_changed:" + check.name + ":" + name)
    return unknown, mismatch


def process_ubatch(
    state: GraphState, params: GraphParameters, *,
    input_checks: Tuple[InputCheck, ...] = (),
    input_roster: Optional[Tuple[Tuple[str, str], ...]] = None,
    disable_reuse: Optional[bool] = None,
    pipeline_parallel: Optional[bool] = None,
    buffer_expansion: Literal["unknown", "not_required", "required"] = "unknown",
) -> Transition:
    """Plan the successful process_ubatch path; do not claim execution success.

    input_roster describes all inputs of the graph built for params. On a reuse
    check, input_checks must cover the previous state's complete roster. Unknown
    decisions conservatively rebuild and remain labeled unknown. Expansion is
    independent caller evidence, never inferred from a topology miss.
    """
    if disable_reuse is not None and type(disable_reuse) is not bool:
        raise ValueError("disable_reuse must be bool or unknown")
    if pipeline_parallel is not None and type(pipeline_parallel) is not bool:
        raise ValueError("pipeline_parallel must be bool or unknown")
    if input_roster is not None and (not isinstance(input_roster, tuple) or len({name for name, _ in input_roster}) != len(input_roster)):
        raise ValueError("input roster must have unique immutable names")
    if buffer_expansion not in ("unknown", "not_required", "required"):
        raise ValueError("invalid buffer expansion evidence")
    unknown, mismatch = [], []
    if disable_reuse is True:
        mismatch.append("reuse_disabled")
    elif state.previous_unknown:
        unknown.append("previous_graph_unknown")
        if disable_reuse is None:
            unknown.append("reuse_disable_state_unknown")
    elif state.previous is None:
        mismatch.append("no_previous_graph")
    else:
        if disable_reuse is None:
            unknown.append("reuse_disable_state_unknown")
        unknown_params, mismatch_params = _parameter_reasons(state.previous, params)
        unknown_inputs, mismatch_inputs = _input_reasons(params, input_checks, state.input_roster)
        unknown += unknown_params + unknown_inputs
        mismatch += mismatch_params + mismatch_inputs
    decision = "miss" if mismatch else "unknown" if unknown else "hit"
    events = [LifecycleEvent("check", CONTEXT_SOURCE)]
    if decision == "hit":
        if buffer_expansion == "required":
            raise ValueError("reused graph does not enter allocation planning")
        events.append(LifecycleEvent("reuse", CONTEXT_SOURCE))
        if pipeline_parallel is True:
            events.append(LifecycleEvent("synchronize_before_set_inputs", CONTEXT_SOURCE))
        elif pipeline_parallel is None:
            events.append(LifecycleEvent("synchronization_requirement_unknown", CONTEXT_SOURCE, "conservative_unknown"))
        next_state = state
    else:
        evidence = "conservative_unknown" if decision == "unknown" else "structural"
        events.extend(LifecycleEvent(kind, CONTEXT_SOURCE, evidence) for kind in ("reset", "build", "alloc_plan"))
        if buffer_expansion == "required":
            events.append(LifecycleEvent("buffer_expand", ALLOC_SOURCE, evidence))
        next_state = GraphState(params, input_roster, state.generation + 1)
    events.extend(LifecycleEvent(kind, CONTEXT_SOURCE) for kind in ("set_inputs", "compute"))
    return Transition(next_state, decision, tuple(dict.fromkeys(mismatch + unknown)), tuple(events), buffer_expansion)


def reserve(state: GraphState) -> Transition:
    """Record graph_reserve invalidation; reserve allocation size is not inferred."""
    return Transition(GraphState(generation=state.generation), "invalidated", ("graph_reserve_invalidates_previous",),
        tuple(LifecycleEvent(kind, RESERVE_SOURCE) for kind in ("scheduler_reset", "invalidate_previous", "reserve_graph_build", "reserve_plan")))



def execution_failed(state: GraphState, *, reason: str = "execution_failed") -> Transition:
    """Forget previous-graph knowledge after a failed actual execution.

    This expresses loss of applicability evidence, not a native reset or a
    successful recovery. No alloc, compute, or buffer expansion is inferred.
    The caller can subsequently supply a proven reset via reserve(), or model
    the next successful ubatch conservatively from this unknown state.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("execution failure reason must be nonempty")
    return Transition(
        GraphState.unknown(generation=state.generation), "invalidated",
        ("execution_failure_previous_graph_unknown", reason),
        (LifecycleEvent("execution_failed_graph_unknown", CONTEXT_SOURCE, "conservative_unknown"),),
    )
