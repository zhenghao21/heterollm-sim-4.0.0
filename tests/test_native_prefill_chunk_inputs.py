"""Declared chunk descriptions preserve native geometry without live commits."""
from dataclasses import replace

import pytest

from heterollm_sim.execution_control import _NativeLayerSamplingCutoff
from heterollm_sim.native_layer_sampling import _core_bits, _state_bits, _TP_TIMELINE_MAPS
from heterollm_sim.native_prefill_chunk_inputs import (
    DECLARED_CHUNK_ADDRESS_ASSUMPTION, DECLARED_CHUNK_STATE_ASSUMPTION, iter_prefill_chunk_inputs)
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import _OnlineRuntime
from heterollm_sim.simulation_options import NativePrefillPassConfig, SimulationConfig
from test_native_prefill import _chunked_prefill


class _Identity:
    @staticmethod
    def copy(value):
        return value


def _live_boundary(runtime):
    kernel = runtime._execution_resource_kernel
    physical = kernel.physical_runtime
    return {
        "cores": {owner: _core_bits(active, _Identity()) for owner, active in physical.runtimes.items()},
        "timeline": _state_bits({name: getattr(physical.timeline, name) for name in _TP_TIMELINE_MAPS}),
        "allocations": _state_bits({owner: allocator._allocations
                                    for owner, allocator in physical.allocators.items()}),
        "resources": _state_bits(kernel.resource_available),
        "times_and_commits": _state_bits((runtime.now, kernel.makespan_ns,
            runtime._completed_decode_steps, len(runtime.batches),
            [(state.prefill_cursor, state.committed, state.kv_pages) for state in runtime.states.values()]))}


def test_thirteen_native_chunk_inputs_keep_live_state_and_fixed_ring_geometry(monkeypatch):
    original = _OnlineRuntime.__init__
    observed = []

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)

        def capture(active, cohort, cost, stages, request_ready_ns):
            before = _live_boundary(active)
            descriptions = iter_prefill_chunk_inputs(active, cohort, stages, request_ready_ns,
                active.execution_control, NativePrefillPassConfig())
            for prepared in descriptions:
                assert prepared["source"] is not active._execution_resource_kernel
                assert _live_boundary(active) == before
                assert prepared["stage_key_by_id"] == {
                    stage.stage_id: (stage.component_id, stage.stage_index)
                    for stage in prepared["native_stages"]}
                assert all("requires_device_fence" in stage for stage in prepared["stage_descriptors"])
                accesses = [(task.metadata.get("layer_id"), access)
                    for task in prepared["specs"] for access in task.metadata.get("memory_accesses", ())
                    if str(access.get("buffer_id", "")).startswith("kv:")]
                count, context = prepared["chunk_prompt_tokens"], prepared["context_tokens"]
                for layer, access in accesses:
                    if access["operation"] != "write":
                        continue
                    # h32, two KV heads, TP2: one BF16 8-wide head per rank.
                    assert access["byte_count"] == count * 16
                    assert access["offset_bytes"] == (context * 16 if layer in {
                        "layer-003", "layer-007"} else 0)
                    if layer not in {"layer-003", "layer-007"}:
                        assert access["allocation_size_bytes"] == 4 * 16
                if prepared["chunk_index"]:
                    assert DECLARED_CHUNK_STATE_ASSUMPTION in prepared["chunk_input_boundary"]["assumptions"]
                    assert DECLARED_CHUNK_ADDRESS_ASSUMPTION in prepared["chunk_input_boundary"]["assumptions"]
                    assert not prepared["chunk_input_boundary"]["address_equivalence_proven"]
                    assert any(layer == "layer-007" and access["operation"] == "read"
                               and access["offset_bytes"] == 0 and access["byte_count"] >= context * 16
                               for layer, access in accesses)
                observed.append((prepared["chunk_index"], context, count))
            assert _live_boundary(active) == before
            raise _NativeLayerSamplingCutoff()

        self._native_layer_sampler = capture

    monkeypatch.setattr(_OnlineRuntime, "__init__", initialize)
    scenario = _chunked_prefill(prompt=50, model_chunks=True)
    with pytest.raises(_NativeLayerSamplingCutoff):
        run_scenario(replace(scenario, simulation=SimulationConfig(mode="detailed")),
                     retention_policy="aggregate")
    assert observed == [(index, index * 4, 4 if index < 12 else 1) for index in range(13)]
