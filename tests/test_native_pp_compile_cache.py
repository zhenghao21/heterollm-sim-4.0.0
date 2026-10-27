"""A cohort namespace and a PP activation owner bind independently."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import planner
from test_decode_execution_acceleration import exact_values


@pytest.mark.parametrize("medium", ["HBM", "HBF"])
def test_static_projection_cache_rebinds_pp_owner_and_cohort_phase_separately(medium, monkeypatch):
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", medium, 32, 3)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        plan, router = context.parallel_plan(), context.router()
        layer = context.execution_layers()[0]

        def emit(cohort_id, history, group=0):
            builder = planner._TaskBuilder(replace(scenario.workload.requests[0],
                request_id=cohort_id, prompt_tokens=1, output_tokens=0))
            builder.request = replace(builder.request,
                request_id=cohort_id + f".pipeline-group{group:04d}")
            builder._cpu_graph_rows = 1
            builder._attention_invocation_lanes = (SimpleNamespace(request_id="scan-request",
                context_tokens=history + 1, kv_read_tokens=history, kv_materialized_tokens=1),)
            first = builder.add("input", planner.TaskCategory.COMPUTE)
            builder._rank_value_components[first] = {0: "gpu0"}
            planner._compile_parallel_layer_body(builder, scenario, plan, router, layer,
                token_batch=1, context_tokens=history, kv_read_tokens=history,
                kv_append_tokens=1, kv_materialized_tokens=1, linear_state_runtime=None,
                phase=cohort_id + f".decode.group{group:04d}", dependencies=(first,))
            return tuple(builder.tasks)

        before = emit("cohort-first", 15)
        cache = context.invariant(("static_input_projection_segments",), dict)
        assert len(cache) == 1
        template = next(iter(cache.values()))[0]
        hits = []
        original = planner._TaskSegmentTemplate.replay

        def counted(self, *args, **kwargs):
            if self is template:
                hits.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(planner._TaskSegmentTemplate, "replay", counted)
        cached = emit("cohort-second", 16)
        assert hits == [1] and len(cache) == 1
        other_group = emit("cohort-third", 17, group=1)
        assert hits == [1] and len(cache) == 2
        monkeypatch.setattr(planner, "_compile_static_input_projection",
            lambda *args, **kwargs: args[-1]())
        exact_values(cached, emit("cohort-second", 16))
        exact_values(other_group, emit("cohort-third", 17, group=1))
        old_views = [task.metadata["generic_kv_storage"] for task in before
                     if "generic_kv_storage" in task.metadata]
        new_views = [task.metadata["generic_kv_storage"] for task in cached
                     if "generic_kv_storage" in task.metadata]
        assert old_views and new_views and old_views != new_views
