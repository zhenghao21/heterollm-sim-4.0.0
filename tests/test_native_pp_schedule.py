"""Native request microbatches retain the physical stage DAG and costs."""
from dataclasses import replace

import pytest

from heterollm_sim.config import parallel_from_dict
from heterollm_sim.ir import ParallelSpec
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import BatchCohort, BatchItem, _OnlineRuntime
from heterollm_sim.planner import (_pipeline_serving_invocation_groups,
                                    _serving_invocation_groups)
from native_toy import toy_scenario, _analytical_toy_hardware, _moe_toy_model


def pp_scenario(microbatches=2, *, pp=2, ep=1, batch=4, tp=1):
    model = toy_scenario(layer_count=max(2, pp)).model
    if ep > 1:
        model = _moe_toy_model(max(2, pp))
    from exploring_hbf_native_config import build_native_comparison_scenario
    scenario = build_native_comparison_scenario(
        "llama3_405b", "HBM4", "SHORT", tp=tp, pp=pp, ep=ep, batch_size=batch,
        model_override=model,
        prompt_override=4, output_override=3, warmup_override=2,
        measurement_override=4,
    )
    scenario = _analytical_toy_hardware(scenario, .00001)
    return replace(scenario, placement=replace(scenario.placement,
        parallel=replace(scenario.placement.parallel, pipeline_microbatches=microbatches)))


@pytest.fixture(scope="module")
def pipeline_runs():
    original = _OnlineRuntime._schedule_execution_stages
    collected, first = {}, {}

    def capture(self, cohort, cost, stages, *args, **kwargs):
        degree = self.plan.scenario.placement.parallel.pipeline_microbatches
        if degree not in first:
            qkv = [dict(task.metadata["cost_model"])
                   for stage in stages for task in stage.execution_tasks
                   if task.metadata.get("projection_id") == "attention.qkv"
                   and task.metadata.get("cost_model", {}).get("operator_class") == "gemm"]
            first[degree] = qkv
        return original(self, cohort, cost, stages, *args, **kwargs)

    _OnlineRuntime._schedule_execution_stages = capture
    try:
        for degree in (1, 2, 4):
            collected[degree] = run_scenario(pp_scenario(degree), retention_policy="aggregate")
    finally:
        _OnlineRuntime._schedule_execution_stages = original
    return collected, first


def _gpu_stages(batch):
    return {row["stage_id"]: row for row in batch.cost.metadata["execution_stage_schedule"]
            if row["invocation_group_id"] and row["component_id"].startswith("gpu")
            and not row["stage_id"].endswith(".activation_transfer")}


def test_native_microbatches_obey_stage_fifo_and_overlap_other_stages(pipeline_runs):
    runs, _ = pipeline_runs
    for degree in (2, 4):
        for batch in runs[degree].serving.batches:
            rows = _gpu_stages(batch)
            for stage in (0, 1):
                previous = None
                for group in range(degree):
                    row = rows[f"operator-invocation-group-{group:04d}.stage{stage:04d}"]
                    if previous is not None:
                        assert row["start_ns"] >= previous["end_ns"] - 1e-6
                        assert previous["stage_id"] in row["dependencies"]
                    if stage:
                        source = rows[f"operator-invocation-group-{group:04d}.stage0000"]
                        assert row["start_ns"] >= source["end_ns"] - 1e-6
                    previous = row
            first = rows["operator-invocation-group-0000.stage0001"]
            next_first_stage = rows["operator-invocation-group-0001.stage0000"]
            assert next_first_stage["start_ns"] < first["end_ns"]
            assert first["start_ns"] < next_first_stage["end_ns"]


def test_pipeline_fill_and_drain_bubbles_are_retained(pipeline_runs):
    runs, _ = pipeline_runs
    rows = _gpu_stages(runs[4].serving.batches[0])
    first = rows["operator-invocation-group-0000.stage0000"]
    downstream = rows["operator-invocation-group-0000.stage0001"]
    upstream_tail = rows["operator-invocation-group-0003.stage0000"]
    last = rows["operator-invocation-group-0003.stage0001"]
    # Stage 1 waits for initial stage-0 work, then drains after stage 0 stops.
    assert downstream["start_ns"] > first["start_ns"]
    assert last["end_ns"] > upstream_tail["end_ns"]
    span = last["end_ns"] - first["start_ns"]
    assert span > sum(row["end_ns"] - row["start_ns"]
                      for row in rows.values() if row["component_id"] == "gpu1")


def test_slow_activation_send_can_overlap_the_next_microbatch_compute():
    scenario = pp_scenario()
    # Force the deliberately slow peer route: alternate paths through host
    # PCIe or the PD source must not bypass the link under test. These are
    # synthetic topology inputs; no elapsed time is divided or extrapolated.
    links = tuple(
        replace(link, bandwidth_gbps=.0001) if link.link_id.startswith("nvlink-") else
        replace(link, latency_ns=1e12) if link.protocol in {"PCIe", "InfiniBand"} else link
        for link in scenario.hardware.links
    )
    scenario = replace(scenario, hardware=replace(scenario.hardware, links=links))
    result = run_scenario(scenario, retention_policy="aggregate")
    first = result.serving.batches[0]
    stages = {row["stage_id"]: row for row in first.cost.metadata["execution_stage_schedule"]}
    compute = stages["operator-invocation-group-0000.stage0000"]
    send = stages["operator-invocation-group-0000.pp0000.activation_transfer"]
    successor = stages["operator-invocation-group-0001.stage0000"]
    consumer = stages["operator-invocation-group-0000.stage0001"]
    assert send["end_ns"] - send["start_ns"] > 2 * (compute["end_ns"] - compute["start_ns"])
    assert successor["start_ns"] >= compute["end_ns"]
    assert successor["end_ns"] < send["end_ns"]
    assert send["stage_id"] not in successor["dependencies"]
    assert consumer["start_ns"] >= send["end_ns"]
    assert send["stage_id"] in consumer["dependencies"]


def test_each_microbatch_relowers_real_small_gemm_geometry(pipeline_runs):
    _, qkv = pipeline_runs
    assert [len(qkv[degree]) for degree in (1, 2, 4)] == [2, 4, 8]
    for degree in (2, 4):
        wide, narrow = qkv[1][0], qkv[degree][0]
        assert narrow["activation_bytes"] * degree == wide["activation_bytes"]
        assert narrow["output_bytes"] * degree == wide["output_bytes"]
        assert narrow["weight_bytes"] == wide["weight_bytes"]
        # All toy M dimensions fit one MMA row tile. Splitting requests does
        # not divide issued compute service by the microbatch count.
        assert narrow["tile_count"] == wide["tile_count"]
        assert narrow["compute_service_ns"] == wide["compute_service_ns"]
        assert narrow["tile_utilization"] * degree == wide["tile_utilization"]


def _cohort(requests=4, tokens=1, ep=1):
    return BatchCohort("cohort-pipeline-test", "decode" if tokens == 1 else "prefill", 0,
        tuple(BatchItem(f"request-{index}", "decode" if tokens == 1 else "prefill", tokens, 4,
            main_tokens=tokens, draft_tokens=0, verifier_tokens=tokens,
            committed_tokens=tokens, logit_tokens=1, ep_group=index % ep)
            for index in range(requests)))


def test_pipeline_request_partition_preserves_ep_balance():
    scenario, cohort = pp_scenario(ep=2, batch=8), _cohort(requests=8, ep=2)
    groups = _pipeline_serving_invocation_groups(scenario,
        _serving_invocation_groups(scenario, cohort))
    assert len(groups) == 2
    for group in groups:
        assert [len(part.lanes) for part in group.ep_partitions(2)] == [2, 2]
    assert set(groups[0].request_ids).isdisjoint(groups[1].request_ids)


@pytest.mark.parametrize("tp,ep,batch", [(2, 1, 4), (1, 2, 8)])
def test_native_tp_ep_pipeline_keeps_distinct_overlapping_pp_stages(tp, ep, batch):
    result = run_scenario(pp_scenario(tp=tp, ep=ep, batch=batch), retention_policy="aggregate")
    assert result.serving.decode_measurement["complete"]
    for execution in result.serving.batches:
        stages = _gpu_stages(execution)
        assert len(stages) == 4
        first_stage = stages["operator-invocation-group-0000.stage0000"]
        first_downstream = stages["operator-invocation-group-0000.stage0001"]
        next_stage = stages["operator-invocation-group-0001.stage0000"]
        next_downstream = stages["operator-invocation-group-0001.stage0001"]
        assert next_stage["start_ns"] >= first_stage["end_ns"] - 1e-6
        assert next_downstream["start_ns"] >= first_downstream["end_ns"] - 1e-6
        assert next_stage["start_ns"] < first_downstream["end_ns"]


def test_token_ubatches_do_not_turn_one_prefill_owner_into_independent_requests():
    scenario = pp_scenario()
    scenario = replace(scenario, workload=replace(scenario.workload, scheduler=replace(
        scenario.workload.scheduler, max_num_batched_tokens=8, max_num_ubatch_tokens=2)))
    cohort = _cohort(requests=2, tokens=4)
    groups = _pipeline_serving_invocation_groups(scenario,
        _serving_invocation_groups(scenario, cohort))
    assert len(groups) == 4
    for left, right in ((groups[0], groups[1]), (groups[2], groups[3])):
        assert left.request_ids == right.request_ids
        assert left.pipeline_microbatch_index == right.pipeline_microbatch_index
        assert right.pipeline_predecessor_group_ids == (left.group_id,)
    assert groups[2].pipeline_predecessor_group_ids == ()


def test_default_microbatch_degree_preserves_pp1_invocation_graph():
    scenario, cohort = pp_scenario(1, pp=1), _cohort()
    raw = _serving_invocation_groups(scenario, cohort)
    assert _pipeline_serving_invocation_groups(scenario, raw) is raw
    assert len(raw) == 1 and raw[0].pipeline_microbatch_index is None


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_pipeline_microbatch_degree_is_a_positive_integer(invalid):
    with pytest.raises(ValueError, match="pipeline_microbatches"):
        ParallelSpec(pipeline_microbatches=invalid)


def test_pipeline_microbatch_degree_roundtrips_configuration():
    assert parallel_from_dict({"pp_degree": 2, "pipeline_microbatches": 4}).pipeline_microbatches == 4
    assert parallel_from_dict({}).pipeline_microbatches == 1
