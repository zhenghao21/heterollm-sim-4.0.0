"""Lower later Prefill chunks with explicitly declared future boundary inputs.

Future logical cursors live only in independent description runtimes. Their
physical kernels remain copies of the first real boundary: no sampled end
state is resumed and no earlier chunk's physical events are invented.
"""
from __future__ import annotations

from copy import deepcopy

from .execution_control import _NativeLayerSamplingCutoff
from .native_layer_sampling import _independent_kernel


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
