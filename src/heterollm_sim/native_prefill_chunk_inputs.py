"""Stream native chunk graphs with explicitly declared future boundary inputs.

Future logical cursors live only in independent description runtimes. Their
physical kernels remain copies of the first real boundary: no sampled end
state is resumed and no earlier chunk's physical events are invented.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from .execution_control import _NativeLayerSamplingCutoff
from .ir import model_graph_execution_layers
from .native_layer_sampling import NativeLayerSamplingGuardError, _independent_kernel
from .native_stationary import cohort_layers, prefill_stage_ready


DECLARED_CHUNK_STATE_ASSUMPTION = (
    "Future Prefill chunks are sampled on independent copies of the first real boundary, with "
    "their true logical contexts declared as inputs. Prior chunks' controller, cache and stage "
    "frontier states are not reconstructed; the first-access DRAM row bound does not bound "
    "those omitted state differences. This is a cross-chunk boundary-state approximation.")
DECLARED_CHUNK_ADDRESS_ASSUMPTION = (
    "Native buffer identities, capacities, fixed-chunk ring offsets and access lengths are kept. "
    "Missing KV inputs are allocated by the copied native allocator without prior write events. "
    "Their addresses are not proven equal to a full native continuation's allocations.")


def _description_runtime(runtime, control):
    """Copy mutable planning/admission state together with an independent kernel."""
    source = runtime._execution_resource_kernel
    kernel, _ = _independent_kernel(source)
    # These are read-only inputs (or the caller's cancellation/progress
    # channel), not mutable simulated state. The source's resource ownership
    # mapping also follows _independent_kernel's immutable-copy contract.
    memo = {id(source): kernel, id(runtime.plan): runtime.plan,
            id(runtime.plan.scenario): runtime.plan.scenario,
            id(runtime.lowerer): runtime.lowerer,
            id(runtime.execution_control): control}
    described = deepcopy(runtime, memo)
    described.execution_control = control
    described._native_prefill_cohort_completed = None
    return described


def _lower_description(runtime, cohort):
    """Stop the ordinary lowering path before it submits any physical task."""
    captured = {}

    def capture(active, actual_cohort, cost, stages, request_ready_ns):
        captured.update(cohort=actual_cohort, stages=stages, request_ready_ns=request_ready_ns)
        raise _NativeLayerSamplingCutoff()

    runtime._native_layer_sampler = capture
    try:
        runtime._execute_cohort(cohort)
    except _NativeLayerSamplingCutoff:
        return captured["cohort"], captured["stages"], captured["request_ready_ns"]
    raise RuntimeError("declared Prefill chunk unexpectedly executed a native cohort")


def iter_prefill_chunk_inputs(runtime, first_cohort, first_stages, request_ready_ns, control, options):
    """Yield one Prepared graph at a time, without advancing ``runtime``.

    The first supplied graph is real. Each later graph is lowered by the
    ordinary native serving path on a fresh description copy: only that
    copy's logical cursor and admission reservations are set for its chunk.
    ``options`` is the caller's sampling configuration; this producer does
    not execute samples or choose a default approximation policy.
    """
    control.raise_if_cancelled()
    if (first_cohort.kind != "prefill" or len(first_cohort.items) != 1
            or first_cohort.items[0].phase != "prefill"):
        raise NativeLayerSamplingGuardError("native_prefill_chunk_inputs_require_one_prefill_request")
    first_item = first_cohort.items[0]
    state = runtime.states[first_item.request_id]
    if first_item.context_tokens or state.prefill_cursor or runtime.batches:
        raise NativeLayerSamplingGuardError("native_prefill_chunk_inputs_require_first_real_boundary")
    if runtime._external_executor is not None or runtime.residency_manager is not None:
        raise NativeLayerSamplingGuardError("native_prefill_chunk_inputs_require_unshared_native_boundary")
    layers = model_graph_execution_layers(runtime.plan.scenario.model.graph)
    context, index, total = 0, 0, state.spec.prompt_tokens
    while context < total:
        control.raise_if_cancelled()
        described = _description_runtime(runtime, control)
        logical_state = described.states[first_item.request_id]
        logical_state.prefill_cursor = context
        if index == 0:
            cohort, stages, ready_ns = first_cohort, first_stages, request_ready_ns
        else:
            cohort = described._prefill_cohort([logical_state], recompute=False)
            if cohort is None:
                raise NativeLayerSamplingGuardError("native_prefill_declared_chunk_cannot_be_admitted")
            cohort = replace(cohort, cohort_id=f"{first_cohort.cohort_id}.declared-chunk-{index:04d}")
            cohort, stages, ready_ns = _lower_description(described, cohort)
        item = cohort.items[0]
        if item.context_tokens != context or item.token_count <= 0:
            raise NativeLayerSamplingGuardError("native_prefill_declared_chunk_not_contiguous")
        ready = prefill_stage_ready(described, cohort, stages, ready_ns)
        prepared = cohort_layers(described, cohort, stages, ready_ns,
            "native-prefill-declared-inputs.", ready_by_stage=ready, allow_pipeline=True)
        prepared.update(
            chunk_index=index, chunk_id=cohort.cohort_id, prompt_tokens=total,
            chunk_prompt_tokens=item.token_count, context_tokens=context,
            native_stages=stages,
            stage_key_by_id={stage.stage_id: (stage.component_id, stage.stage_index) for stage in stages},
            stage_ready_ns=ready,
            stage_descriptors=tuple({"stage_id": stage.stage_id, "component_id": stage.component_id,
                "stage_index": stage.stage_index, "requires_device_fence": stage.requires_device_fence,
                "participates_in_device_fence": described._component_participates_in_device_fence(
                    stage.component_id),
                "dependencies": stage.dependencies} for stage in stages),
            chunk_input_boundary={
                "basis": "independent_copy_of_first_real_boundary_with_declared_logical_inputs",
                "is_native_continuation": False, "predecessor_events_fabricated": False,
                "future_physical_state_advanced": False,
                "source_boundary_is_real": index == 0,
                "address_equivalence_proven": False,
                "stage_frontier_basis": "first_real_boundary; cross_chunk_edges_composed_separately",
                "assumptions": [] if index == 0 else [
                    DECLARED_CHUNK_STATE_ASSUMPTION, DECLARED_CHUNK_ADDRESS_ASSUMPTION]},
            kv_geometry=tuple({"layer_id": layer.layer_id,
                "attention_chunk_size": layer.attention_chunk_size,
                "logical_context_tokens": context, "append_tokens": item.token_count,
                "historical_context_within_attention_block": (
                    context % layer.attention_chunk_size if layer.attention_chunk_size else context),
                "physical_write_offset_tokens": (
                    context % layer.attention_chunk_size if layer.attention_chunk_size else context)}
                for layer in layers))
        control.report("native_prefill_chunk_inputs", index + 1, unit="declared_chunks",
            message="已准备真实分段图与声明边界输入，未执行前序分段",
            metadata={"context_tokens": context, "chunk_tokens": item.token_count,
                      "native_continuation": False, "full_cohort_committed": False})
        yield prepared
        context += item.token_count
        index += 1
        # Do not retain earlier lowered DAGs in the streaming producer.
        del prepared, described, stages, cohort


__all__ = ["iter_prefill_chunk_inputs", "DECLARED_CHUNK_STATE_ASSUMPTION",
           "DECLARED_CHUNK_ADDRESS_ASSUMPTION"]
